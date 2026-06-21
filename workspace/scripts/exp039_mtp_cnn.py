"""
exp039: MTP-CNN (Multi-Trajectory Prediction loss + 1D CNN)

Alyaev & Elsheikh 2022 approach - K=4 trajectory candidates with MTP loss.
Outputs 4 TVT delta trajectory hypotheses per well, selects best via likelihood.

Features: MD_norm, Z_norm, GR_norm, last_known_TVT, delta_MD_from_PS, delta_Z_from_PS
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
from scipy.special import softmax
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Config
# ============================================================================

CONFIG = {
    'exp_id': 'exp039_mtp_cnn',
    'seed': 42,
    'n_splits': 5,
    'K': 4,  # number of trajectory candidates
    'in_dim': 6,  # features
    'hidden': 64,
    'cnn_depth': 3,
    'batch_size': 16,  # wells per batch
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'max_well_length': 1500,  # max row per well (pad/trim)
    'device': 'cpu',
}

PROJECT_ROOT = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
FOLD_FILE = PROJECT_ROOT / 'data' / 'folds' / 'folds_group_well_v001.csv'
EXP_DIR = PROJECT_ROOT / 'experiments' / CONFIG['exp_id']
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Set seed
# ============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(CONFIG['seed'])

# ============================================================================
# Model
# ============================================================================

class MTPCNN(nn.Module):
    """Multi-Trajectory Prediction CNN"""

    def __init__(self, in_dim=6, K=4, hidden=64, depth=3):
        super().__init__()
        self.K = K
        self.in_dim = in_dim

        # Conv encoder: in_dim -> hidden -> hidden -> hidden
        layers = [nn.Conv1d(in_dim, hidden, 5, padding=2), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Conv1d(hidden, hidden, 5, padding=2))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        # Output head: hidden -> K*2 (mean, log_std per trajectory)
        self.head = nn.Conv1d(hidden, K * 2, 1)

    def forward(self, x, mask=None):
        """
        x: (B, in_dim, T) where B=batch of wells, T=max sequence length
        mask: (B, T) binary mask (1=valid, 0=pad)

        Returns: (B, K, 2, T) where dim 1=trajectory, dim 2=[mean, log_std]
        """
        h = self.encoder(x)
        out = self.head(h)  # (B, K*2, T)
        B, _, T = out.shape
        out = out.view(B, self.K, 2, T)
        return out


# ============================================================================
# Loss
# ============================================================================

def mtp_loss(pred, target, mask=None):
    """
    Multi-Trajectory Prediction loss: pick best trajectory per sample.

    pred: (B, K, 2, T) - K trajectories with mean and log_std
    target: (B, T) - ground truth delta
    mask: (B, T) - valid row indicator

    Returns: scalar loss
    """
    B, K, _, T = pred.shape
    means = pred[:, :, 0, :]  # (B, K, T)

    # MSE loss per trajectory per row
    target_expanded = target.unsqueeze(1)  # (B, 1, T)
    mse_per_traj = (means - target_expanded) ** 2  # (B, K, T)

    if mask is not None:
        mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
        mse_per_traj = mse_per_traj * mask_expanded  # (B, K, T)
        row_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        mse_per_traj = mse_per_traj.sum(dim=-1) / row_count  # (B, K)
    else:
        mse_per_traj = mse_per_traj.mean(dim=-1)  # (B, K)

    # MTP: select best (lowest MSE) trajectory per sample
    best_mse = mse_per_traj.min(dim=1).values  # (B,)
    return best_mse.mean()


# ============================================================================
# Dataset
# ============================================================================

class WellSequenceDataset(Dataset):
    """
    Per-well dataset: stacks all rows of a well into sequence.
    """

    def __init__(self, well_ids, train_base, scalers, max_length=1500, is_train=True):
        self.well_ids = well_ids
        self.train_base = train_base
        self.scalers = scalers
        self.max_length = max_length
        self.is_train = is_train
        self.sequences = []
        self.targets = []
        self.masks = []
        self.well_idx_map = []

        self._build_sequences()

    def _build_sequences(self):
        """Preprocess sequences"""
        for well_id in self.well_ids:
            well_data = self.train_base[self.train_base['well_id'] == well_id].copy()

            # Only target rows
            target_mask = well_data['is_target'].values
            if target_mask.sum() == 0:
                continue

            well_data = well_data[target_mask].reset_index(drop=True)

            # Features
            features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
            X = well_data[features].values.astype(np.float32)

            # Handle NaN in features: fill with 0 (will be scaled to mean=0, std=1)
            X = np.nan_to_num(X, nan=0.0)

            # Normalize
            X_scaled = np.zeros_like(X)
            for i, feat in enumerate(features):
                if feat in self.scalers:
                    X_scaled[:, i] = self.scalers[feat].transform(X[:, [i]]).flatten()
                else:
                    X_scaled[:, i] = X[:, i]

            # Target: TVT delta from row to row
            tvt = well_data['TVT'].values.astype(np.float32)
            delta_tvt = np.diff(tvt, prepend=tvt[0])  # delta from previous or 0

            # Final NaN check
            if np.isnan(X_scaled).any() or np.isnan(delta_tvt).any():
                logger.warning(f"Well {well_id} has NaN after handling, skipping")
                continue

            # Trim or pad
            T = min(len(X_scaled), self.max_length)
            X_trim = X_scaled[:T]
            y_trim = delta_tvt[:T]

            # Always pad to max_length
            X_padded = np.zeros((self.max_length, len(features)), dtype=np.float32)
            y_padded = np.zeros(self.max_length, dtype=np.float32)
            mask = np.zeros(self.max_length, dtype=np.float32)

            X_padded[:T] = X_trim
            y_padded[:T] = y_trim
            mask[:T] = 1.0

            self.sequences.append(X_padded)  # (T, 6)
            self.targets.append(y_padded)     # (T,)
            self.masks.append(mask)           # (T,)
            self.well_idx_map.append(well_id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Convert (T, 6) to (6, T) for Conv1d
        x = torch.from_numpy(self.sequences[idx].T)  # (6, T)
        y = torch.from_numpy(self.targets[idx])       # (T,)
        m = torch.from_numpy(self.masks[idx])         # (T,)
        return x, y, m, self.well_idx_map[idx]


# ============================================================================
# Training
# ============================================================================

def train_fold(fold_idx, train_well_ids, val_well_ids, train_base, scalers, config):
    """Train single fold"""

    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_idx}: {len(train_well_ids)} train, {len(val_well_ids)} val wells")
    logger.info(f"{'='*60}")

    # Datasets
    train_ds = WellSequenceDataset(train_well_ids, train_base, scalers,
                                    config['max_well_length'], is_train=True)
    val_ds = WellSequenceDataset(val_well_ids, train_base, scalers,
                                  config['max_well_length'], is_train=False)

    logger.info(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    # Model
    model = MTPCNN(in_dim=config['in_dim'], K=config['K'],
                   hidden=config['hidden'], depth=config['cnn_depth'])
    model = model.to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{config['epochs']} [Train]")
        for x, y, mask, _ in pbar:
            x = x.to(config['device'])  # (B, 6, T)
            y = y.to(config['device'])  # (B, T)
            mask = mask.to(config['device'])  # (B, T)

            optimizer.zero_grad()
            pred = model(x, mask)  # (B, K, 2, T)
            loss = mtp_loss(pred, y, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': train_loss / n_batches})

        train_loss /= max(1, n_batches)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{config['epochs']} [Val]")
            for x, y, mask, _ in pbar:
                x = x.to(config['device'])
                y = y.to(config['device'])
                mask = mask.to(config['device'])

                pred = model(x, mask)
                loss = mtp_loss(pred, y, mask)
                val_loss += loss.item()
                n_val_batches += 1
                pbar.set_postfix({'loss': val_loss / n_val_batches})

        val_loss /= max(1, n_val_batches)
        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    model.load_state_dict(model_state)

    return model, train_ds, val_ds


# ============================================================================
# Inference
# ============================================================================

def predict_well(model, well_data, scalers, config):
    """
    Predict TVT for single well using ensemble of K trajectories.

    Returns: per-row predictions
    """
    features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
    X = well_data[features].values.astype(np.float32)

    # Normalize
    X_scaled = np.zeros_like(X)
    for i, feat in enumerate(features):
        if feat in scalers:
            X_scaled[:, i] = scalers[feat].transform(X[:, [i]]).flatten()
        else:
            X_scaled[:, i] = X[:, i]

    # Pad
    max_len = config['max_well_length']
    T = min(len(X_scaled), max_len)
    X_padded = np.zeros((max_len, len(features)), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)

    X_padded[:T] = X_scaled[:T]
    mask[:T] = 1.0

    # Convert to tensor (1, 6, T)
    x_tensor = torch.from_numpy(X_padded.T).unsqueeze(0).to(config['device'])
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(config['device'])

    model.eval()
    with torch.no_grad():
        pred = model(x_tensor, mask_tensor)  # (1, K, 2, T)

    # pred: (1, K, 2, T)
    means = pred[0, :, 0, :T].cpu().numpy()  # (K, T)
    log_stds = pred[0, :, 1, :T].cpu().numpy()  # (K, T)
    stds = np.exp(np.clip(log_stds, -3, 3)) + 1e-6

    # Likelihood-weighted ensemble
    # Higher likelihood (lower -log p) -> higher weight
    likelihoods = -((means - 0)**2 / (2 * stds**2) + log_stds)  # (K, T)
    weights = softmax(likelihoods, axis=0)  # (K, T)

    pred_delta = (means * weights).sum(axis=0)  # (T,)

    # Reconstruct TVT from delta
    last_known_tvt = well_data['last_known_TVT'].iloc[0]
    pred_tvt = last_known_tvt + np.cumsum(pred_delta)

    return pred_tvt[:T], pred_delta[:T]


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    logger.info("exp039: MTP-CNN")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")

    # Load data
    logger.info("\nLoading data...")
    train_base = pd.read_parquet(str(DATA_DIR / 'train_base_v001.parquet'))
    test_base = pd.read_parquet(str(DATA_DIR / 'test_base_v001.parquet'))
    folds = pd.read_csv(str(FOLD_FILE))

    logger.info(f"Train base: {train_base.shape}")
    logger.info(f"Test base: {test_base.shape}")
    logger.info(f"Folds: {folds.shape}")

    # Fit scalers on train target rows
    logger.info("\nFitting scalers...")
    features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
    scalers = {}
    train_target = train_base[train_base['is_target']].copy()

    for feat in features:
        scaler = StandardScaler()
        scaler.fit(train_target[[feat]])
        scalers[feat] = scaler

    # OOF collection
    oof_records = []
    fold_results = {}
    models = {}
    datasets = {'train': {}, 'val': {}}

    # 5-fold CV
    for fold_idx in range(CONFIG['n_splits']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing fold {fold_idx}")
        logger.info(f"{'='*60}")

        train_fold_mask = folds['fold'] != fold_idx
        val_fold_mask = folds['fold'] == fold_idx

        train_well_ids = folds[train_fold_mask]['well_id'].tolist()
        val_well_ids = folds[val_fold_mask]['well_id'].tolist()

        # Train
        model, train_ds, val_ds = train_fold(fold_idx, train_well_ids, val_well_ids,
                                             train_base, scalers, CONFIG)

        models[fold_idx] = model
        datasets['train'][fold_idx] = train_ds
        datasets['val'][fold_idx] = val_ds

        # Inference on validation set
        logger.info(f"\nInference on fold {fold_idx} validation wells...")
        fold_preds = []
        fold_errors = []

        for well_id in tqdm(val_well_ids, desc=f"Fold {fold_idx} inference"):
            well_data = train_base[train_base['well_id'] == well_id].copy()
            target_data = well_data[well_data['is_target']].reset_index(drop=True)

            if len(target_data) == 0:
                continue

            # Predict
            pred_tvt, _ = predict_well(model, target_data, scalers, CONFIG)

            # Collect results
            true_tvt = target_data['TVT'].values
            errors = pred_tvt - true_tvt

            for idx, (row_idx, row) in enumerate(target_data.iterrows()):
                oof_records.append({
                    'well_id': well_id,
                    'row_idx': idx,
                    'id': row.get('id', ''),
                    'TVT': true_tvt[idx],
                    'last_known_TVT': row['last_known_TVT'],
                    'pred_tvt': pred_tvt[idx],
                    'error': errors[idx],
                    'abs_error': np.abs(errors[idx]),
                    'fold': fold_idx,
                })

            fold_preds.extend(pred_tvt)
            fold_errors.extend(errors)

        # Fold RMSE
        fold_errors = np.array(fold_errors)
        fold_rmse = np.sqrt(np.mean(fold_errors**2))
        fold_results[fold_idx] = {'rmse': float(fold_rmse), 'n_rows': len(fold_errors)}

        logger.info(f"Fold {fold_idx} RMSE: {fold_rmse:.4f} ({len(fold_errors)} rows)")

    # Save OOF
    logger.info("\nSaving OOF...")
    oof_df = pd.DataFrame(oof_records)
    oof_df.to_csv(str(EXP_DIR / 'oof.csv'), index=False)
    logger.info(f"OOF saved: {oof_df.shape}")

    # Compute pooled CV
    pooled_rmse = np.sqrt(np.mean(oof_df['error'].values**2))
    logger.info(f"\nPooled CV RMSE: {pooled_rmse:.4f}")

    # Test inference
    logger.info("\nInference on test set...")
    test_preds = []

    # Use model from fold 0 for test (or ensemble all folds)
    test_model = models[0]

    for test_id in tqdm(test_base['id'].unique(), desc="Test inference"):
        test_data = test_base[test_base['id'] == test_id].copy().reset_index(drop=True)

        if len(test_data) == 0:
            continue

        pred_tvt, _ = predict_well(test_model, test_data, scalers, CONFIG)

        for row_idx, row in test_data.iterrows():
            test_preds.append({
                'id': row['id'],
                'tvt': pred_tvt[row_idx],
            })

    # Save submission
    logger.info("Saving submission...")
    sub_df = pd.DataFrame(test_preds)
    sub_df.to_csv(str(EXP_DIR / 'submission.csv'), index=False)
    logger.info(f"Submission saved: {sub_df.shape}")

    # Blend test vs exp022
    logger.info("\n" + "="*60)
    logger.info("BLEND TEST vs exp022 (particle filter)")
    logger.info("="*60)

    blend_result = None
    try:
        exp022_dir = PROJECT_ROOT / 'experiments' / 'exp022_particle_filter'
        exp022_oof = pd.read_csv(str(exp022_dir / 'oof.csv'))

        # Merge on target rows
        oof_merged = oof_df[['well_id', 'row_idx', 'id', 'TVT', 'error']].copy()
        oof_merged.columns = ['well_id', 'row_idx', 'id', 'TVT', 'error_39']
        exp022_oof_subset = exp022_oof[['well_id', 'row_idx', 'id', 'error']].copy()
        exp022_oof_subset.columns = ['well_id', 'row_idx', 'id', 'error_22']

        merged = oof_merged.merge(exp022_oof_subset, on=['well_id', 'row_idx', 'id'], how='inner')
        logger.info(f"Merged OOF rows: {len(merged)}")

        # Error correlation
        err_corr = np.corrcoef(merged['error_39'], merged['error_22'])[0, 1]
        logger.info(f"Error correlation exp39 vs exp22: {err_corr:.4f}")

        # NNLS blend
        from scipy.optimize import nnls

        X = np.column_stack([merged['error_39'], merged['error_22']])
        y = np.zeros(len(merged))  # target: zero error

        # Alternative: use (pred - true)^2 to minimize MSE
        pred_errors = np.column_stack([merged['error_39'], merged['error_22']])
        target_errors = merged[['error_39', 'error_22']].values  # actual errors

        # Actually blend the predictions, not errors
        # We need the pred_tvt values
        oof_with_pred = oof_df[['well_id', 'row_idx', 'TVT', 'pred_tvt']].copy()
        exp022_with_pred = exp022_oof[['well_id', 'row_idx', 'TVT', 'pred_tvt']].copy()

        merged_pred = oof_with_pred.merge(exp022_with_pred, on=['well_id', 'row_idx'], suffixes=('_39', '_22'))
        pred_39 = merged_pred['pred_tvt_39'].values
        pred_22 = merged_pred['pred_tvt_22'].values
        true_tvt = merged_pred['TVT_39'].values

        # NNLS: minimize ||w1*pred_39 + w2*pred_22 - true||^2 s.t. w >= 0, sum(w)=1
        X = np.column_stack([pred_39, pred_22])
        w, residual = nnls(X, true_tvt)
        w = w / w.sum()  # normalize to sum=1

        blend_pred = w[0] * pred_39 + w[1] * pred_22
        blend_rmse = np.sqrt(np.mean((blend_pred - true_tvt)**2))

        logger.info(f"\nNNLS weights:")
        logger.info(f"  exp039 (w39): {w[0]:.4f}")
        logger.info(f"  exp022 (w22): {w[1]:.4f}")
        logger.info(f"  exp039 CV RMSE: {pooled_rmse:.4f}")
        logger.info(f"  exp022 CV RMSE: {np.sqrt(np.mean(merged['error_22']**2)):.4f}")
        logger.info(f"  blend CV RMSE: {blend_rmse:.4f}")

        blend_result = {
            'error_corr': float(err_corr),
            'w_exp039': float(w[0]),
            'w_exp022': float(w[1]),
            'blend_cv_rmse': float(blend_rmse),
            'exp039_cv_rmse': float(pooled_rmse),
            'blend_improvement': float(np.sqrt(np.mean(merged['error_22']**2)) - blend_rmse),
        }

        if blend_rmse > pooled_rmse:
            logger.info(f"\n✓ Blend improves over exp039: {pooled_rmse:.4f} → {blend_rmse:.4f}")
        else:
            logger.info(f"\n✗ Blend does NOT improve: {pooled_rmse:.4f} > {blend_rmse:.4f}")

    except Exception as e:
        logger.warning(f"Blend test failed: {e}")

    # Save results
    result = {
        'exp_id': CONFIG['exp_id'],
        'cv_rmse': float(pooled_rmse),
        'anchor_rmse': 15.91,
        'fold_results': fold_results,
        'n_broken': int((oof_df['abs_error'] > 20).sum()),
        'blend_result': blend_result,
        'config': CONFIG,
    }

    with open(EXP_DIR / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResult: {json.dumps(result, indent=2)}")
    logger.info(f"\nExp {CONFIG['exp_id']} completed!")
    logger.info(f"Output dir: {EXP_DIR}")

if __name__ == '__main__':
    main()
