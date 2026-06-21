"""
exp039: MTP-CNN (Multi-Trajectory Prediction loss + 1D CNN) - Fold 0 only

Quick validation that full pipeline works
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
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Config (lightweight for testing, fold 0 only)
# ============================================================================

CONFIG = {
    'exp_id': 'exp039_mtp_cnn_fold0',
    'seed': 42,
    'n_splits': 5,
    'test_fold': 0,  # Only test fold 0
    'K': 4,
    'in_dim': 6,
    'hidden': 64,
    'cnn_depth': 3,
    'batch_size': 16,
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'early_stopping_patience': 5,
    'max_well_length': 1500,
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
    def __init__(self, in_dim=6, K=4, hidden=64, depth=3):
        super().__init__()
        self.K = K
        layers = [nn.Conv1d(in_dim, hidden, 5, padding=2), nn.ReLU()]
        for _ in range(depth - 1):
            layers.append(nn.Conv1d(hidden, hidden, 5, padding=2))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, K * 2, 1)

    def forward(self, x, mask=None):
        h = self.encoder(x)
        out = self.head(h)
        B, _, T = out.shape
        out = out.view(B, self.K, 2, T)
        return out


# ============================================================================
# Loss
# ============================================================================

def mtp_loss(pred, target, mask=None):
    B, K, _, T = pred.shape
    means = pred[:, :, 0, :]
    target_expanded = target.unsqueeze(1)
    mse_per_traj = (means - target_expanded) ** 2

    if mask is not None:
        mask_expanded = mask.unsqueeze(1)
        mse_per_traj = mse_per_traj * mask_expanded
        row_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mse_per_traj = mse_per_traj.sum(dim=-1) / row_count
    else:
        mse_per_traj = mse_per_traj.mean(dim=-1)

    best_mse = mse_per_traj.min(dim=1).values
    return best_mse.mean()


# ============================================================================
# Dataset
# ============================================================================

class WellSequenceDataset(Dataset):
    def __init__(self, well_ids, train_base, scalers, max_length=1500):
        self.well_ids = well_ids
        self.train_base = train_base
        self.scalers = scalers
        self.max_length = max_length
        self.sequences = []
        self.targets = []
        self.masks = []
        self.well_idx_map = []
        self._build_sequences()

    def _build_sequences(self):
        for well_id in self.well_ids:
            well_data = self.train_base[self.train_base['well_id'] == well_id].copy()
            target_mask = well_data['is_target'].values
            if target_mask.sum() == 0:
                continue

            well_data = well_data[target_mask].reset_index(drop=True)
            features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
            X = well_data[features].values.astype(np.float32)

            # Handle NaN
            X = np.nan_to_num(X, nan=0.0)

            X_scaled = np.zeros_like(X)
            for i, feat in enumerate(features):
                if feat in self.scalers:
                    X_scaled[:, i] = self.scalers[feat].transform(X[:, [i]]).flatten()
                else:
                    X_scaled[:, i] = X[:, i]

            if np.isnan(X_scaled).any():
                continue

            tvt = well_data['TVT'].values.astype(np.float32)
            delta_tvt = np.diff(tvt, prepend=tvt[0])

            T = min(len(X_scaled), self.max_length)
            X_padded = np.zeros((self.max_length, len(features)), dtype=np.float32)
            y_padded = np.zeros(self.max_length, dtype=np.float32)
            mask = np.zeros(self.max_length, dtype=np.float32)

            X_padded[:T] = X_scaled[:T]
            y_padded[:T] = delta_tvt[:T]
            mask[:T] = 1.0

            self.sequences.append(X_padded)
            self.targets.append(y_padded)
            self.masks.append(mask)
            self.well_idx_map.append(well_id)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx].T)
        y = torch.from_numpy(self.targets[idx])
        m = torch.from_numpy(self.masks[idx])
        return x, y, m, self.well_idx_map[idx]


# ============================================================================
# Training
# ============================================================================

def train_fold(fold_idx, train_well_ids, val_well_ids, train_base, scalers, config):
    logger.info(f"\nFold {fold_idx}: {len(train_well_ids)} train, {len(val_well_ids)} val wells")

    train_ds = WellSequenceDataset(train_well_ids, train_base, scalers, config['max_well_length'])
    val_ds = WellSequenceDataset(val_well_ids, train_base, scalers, config['max_well_length'])

    logger.info(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    model = MTPCNN(in_dim=config['in_dim'], K=config['K'],
                   hidden=config['hidden'], depth=config['cnn_depth'])
    model = model.to(config['device'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                          weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=2)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold_idx} Epoch {epoch+1}/{config['epochs']} [Train]")
        for x, y, mask, _ in pbar:
            x = x.to(config['device'])
            y = y.to(config['device'])
            mask = mask.to(config['device'])

            optimizer.zero_grad()
            pred = model(x, mask)
            loss = mtp_loss(pred, y, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': train_loss / n_batches})

        train_loss /= max(1, n_batches)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(model_state)
    return model, train_ds, val_ds


# ============================================================================
# Inference
# ============================================================================

def predict_well(model, well_data, scalers, config):
    features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
    X = well_data[features].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0)

    X_scaled = np.zeros_like(X)
    for i, feat in enumerate(features):
        if feat in scalers:
            X_scaled[:, i] = scalers[feat].transform(X[:, [i]]).flatten()
        else:
            X_scaled[:, i] = X[:, i]

    max_len = config['max_well_length']
    T = min(len(X_scaled), max_len)
    X_padded = np.zeros((max_len, len(features)), dtype=np.float32)
    mask = np.zeros(max_len, dtype=np.float32)

    X_padded[:T] = X_scaled[:T]
    mask[:T] = 1.0

    x_tensor = torch.from_numpy(X_padded.T).unsqueeze(0).to(config['device'])
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(config['device'])

    model.eval()
    with torch.no_grad():
        pred = model(x_tensor, mask_tensor)

    means = pred[0, :, 0, :T].cpu().numpy()
    likelihoods = -(means**2)
    weights = np.softmax(likelihoods, axis=0)
    pred_delta = (means * weights).sum(axis=0)

    last_known_tvt = well_data['last_known_TVT'].iloc[0]
    pred_tvt = last_known_tvt + np.cumsum(pred_delta)

    return pred_tvt[:T], pred_delta[:T]


# ============================================================================
# Main
# ============================================================================

def main():
    logger.info("exp039 MTP-CNN - Fold 0 only")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")

    train_base = pd.read_parquet(str(DATA_DIR / 'train_base_v001.parquet'))
    folds = pd.read_csv(str(FOLD_FILE))

    logger.info(f"Train base: {train_base.shape}")

    features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
    scalers = {}
    train_target = train_base[train_base['is_target']].copy()

    for feat in features:
        scaler = StandardScaler()
        scaler.fit(train_target[[feat]])
        scalers[feat] = scaler

    # Fold 0 only
    fold_idx = CONFIG['test_fold']
    train_fold_mask = folds['fold'] != fold_idx
    val_fold_mask = folds['fold'] == fold_idx

    train_well_ids = folds[train_fold_mask]['well_id'].tolist()
    val_well_ids = folds[val_fold_mask]['well_id'].tolist()

    logger.info(f"\nTraining fold {fold_idx}...")
    model, train_ds, val_ds = train_fold(fold_idx, train_well_ids, val_well_ids,
                                         train_base, scalers, CONFIG)

    # Inference on val set
    logger.info(f"\nInference on fold {fold_idx} validation wells...")
    oof_records = []
    fold_errors = []

    for well_id in tqdm(val_well_ids[:10], desc=f"Val inference (sample)"):  # First 10 only
        well_data = train_base[train_base['well_id'] == well_id].copy()
        target_data = well_data[well_data['is_target']].reset_index(drop=True)

        if len(target_data) == 0:
            continue

        pred_tvt, _ = predict_well(model, target_data, scalers, CONFIG)
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
            })

        fold_errors.extend(errors)

    fold_errors = np.array(fold_errors)
    fold_rmse = np.sqrt(np.mean(fold_errors**2))
    logger.info(f"Sample RMSE (10 wells): {fold_rmse:.4f}")

    result = {
        'exp_id': CONFIG['exp_id'],
        'fold': fold_idx,
        'sample_cv_rmse': float(fold_rmse),
        'config': CONFIG,
    }

    with open(EXP_DIR / 'result_fold0.json', 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nFold 0 test completed!")


if __name__ == '__main__':
    main()
