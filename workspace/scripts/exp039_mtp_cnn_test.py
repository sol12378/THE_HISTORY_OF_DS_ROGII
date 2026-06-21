"""
Quick test of exp039 MTP-CNN implementation: 5 wells, 2 epochs
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
# Config (lightweight)
# ============================================================================

CONFIG = {
    'exp_id': 'exp039_mtp_cnn_test',
    'seed': 42,
    'K': 4,
    'in_dim': 6,
    'hidden': 32,
    'cnn_depth': 2,
    'batch_size': 2,
    'epochs': 2,
    'lr': 1e-3,
    'device': 'cpu',
}

PROJECT_ROOT = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
FOLD_FILE = PROJECT_ROOT / 'data' / 'folds' / 'folds_group_well_v001.csv'

# ============================================================================
# Model
# ============================================================================

class MTPCNN(nn.Module):
    def __init__(self, in_dim=6, K=4, hidden=32, depth=2):
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
    """
    Multi-trajectory loss: pick best trajectory per sample.

    pred: (B, K, 2, T) - K trajectories with mean and log_std
    target: (B, T) - ground truth delta
    mask: (B, T) - valid row indicator
    """
    B, K, _, T = pred.shape
    means = pred[:, :, 0, :]  # (B, K, T)
    log_stds = pred[:, :, 1, :].clamp(-3, 3)  # (B, K, T)
    stds = torch.exp(log_stds) + 1e-6  # (B, K, T)

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

            # Handle NaN in features: fill with 0 (will be scaled to mean=0, std=1)
            X = np.nan_to_num(X, nan=0.0)

            X_scaled = np.zeros_like(X)
            for i, feat in enumerate(features):
                if feat in self.scalers:
                    X_scaled[:, i] = self.scalers[feat].transform(X[:, [i]]).flatten()
                else:
                    X_scaled[:, i] = X[:, i]

            tvt = well_data['TVT'].values.astype(np.float32)
            delta_tvt = np.diff(tvt, prepend=tvt[0])

            # Final NaN check
            if np.isnan(X_scaled).any() or np.isnan(delta_tvt).any():
                logger.warning(f"Well {well_id} still has NaN after handling, skipping")
                continue

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
# Main
# ============================================================================

def main():
    logger.info("Quick test: 5 wells, 2 epochs")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")

    # Load data
    logger.info("\nLoading data...")
    train_base = pd.read_parquet(str(DATA_DIR / 'train_base_v001.parquet'))
    folds = pd.read_csv(str(FOLD_FILE))

    # Get 5 test wells
    test_wells = folds[folds['fold'] == 0]['well_id'].head(5).tolist()
    logger.info(f"Test wells: {test_wells}")

    # Fit scalers
    logger.info("Fitting scalers...")
    features = ['MD', 'Z', 'GR', 'last_known_TVT', 'delta_MD_from_PS', 'delta_Z_from_PS']
    scalers = {}
    train_target = train_base[train_base['is_target']].copy()

    for feat in features:
        scaler = StandardScaler()
        scaler.fit(train_target[[feat]])
        scalers[feat] = scaler

    # Dataset
    ds = WellSequenceDataset(test_wells, train_base, scalers)
    logger.info(f"Dataset size: {len(ds)}")

    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=True)

    # Model
    model = MTPCNN(in_dim=CONFIG['in_dim'], K=CONFIG['K'],
                   hidden=CONFIG['hidden'], depth=CONFIG['cnn_depth'])
    model = model.to(CONFIG['device'])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # Training
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for x, y, mask, well_ids in pbar:
            x = x.to(CONFIG['device'])
            y = y.to(CONFIG['device'])
            mask = mask.to(CONFIG['device'])

            optimizer.zero_grad()
            pred = model(x, mask)
            loss = mtp_loss(pred, y, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': epoch_loss / n_batches})

        logger.info(f"Epoch {epoch+1} done: loss={epoch_loss / n_batches:.6f}")

    logger.info("\nTest completed successfully!")


if __name__ == '__main__':
    main()
