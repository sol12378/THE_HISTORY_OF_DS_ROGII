#!/usr/bin/env python3
"""exp053 v2: 物理組込み系列NN - simplified version with better debugging.

簡素化: 単一mode (multi-modalは後段)、MSEロスのみ
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rogii.training.baselines import (
    PRED_COL, TARGET_COL, target_rows, tvt_rmse, write_json,
    now_jst, build_submission, ensure_exp_dir, attach_folds, load_base_inputs
)

EXP_ID = "exp053_neural_tracker"
EXP_DIR = Path("experiments") / EXP_ID

# Hyperparameters
L = 128  # sequence length per well (shorter for stability)
EPOCHS = 25
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class SimpleOffsetTracker(nn.Module):
    """Simple sequence model: per-well GR sequence → offset trajectory."""

    def __init__(self, in_channels, hidden=32):
        super().__init__()
        self.in_channels = in_channels

        # Simple TCN backbone (3 dilated layers)
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=8)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Skip connection
        self.skip = nn.Conv1d(in_channels, hidden, kernel_size=1)

        # Output head
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: (B, C, L) → output: (B, L) offset trajectory"""
        skip = self.skip(x)

        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h))) + skip

        out = self.head(h).squeeze(1)  # (B, L)
        return out


class WellSequenceDataset(Dataset):
    def __init__(self, well_arrays, targets, masks, augment=False):
        self.well_arrays = torch.tensor(well_arrays, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.well_arrays)

    def __getitem__(self, idx):
        x = self.well_arrays[idx].clone()
        if self.augment and np.random.random() < 0.3:
            x[0] += torch.randn_like(x[0]) * 0.05
        return x, self.targets[idx], self.masks[idx]


def build_well_sequences(df):
    """Resample well sequences to fixed grid L."""
    features_list = []
    targets_list = []
    masks_list = []
    well_ids = []

    for wid, sub in df.groupby("well_id", sort=False):
        sub = sub.sort_values("row_idx")
        n = len(sub)

        pos = np.arange(n) / max(n - 1, 1)
        grid = np.linspace(0, 1, L)

        def resample(arr):
            return np.interp(grid, pos, np.asarray(arr, float))

        # GR normalization (handle NaNs)
        is_known = sub["is_known_tvt"].astype(bool).to_numpy()
        gr_raw = sub["GR"].to_numpy(float)
        is_gr_missing = sub["is_gr_missing"].astype(bool).to_numpy()

        # Forward fill NaNs
        gr = pd.Series(gr_raw).fillna(method='ffill').fillna(method='bfill').to_numpy()

        valid_known = is_known & ~is_gr_missing
        if valid_known.sum() >= 5:
            gr_mean = np.mean(gr[valid_known])
            gr_std = np.std(gr[valid_known]) + 1e-6
        else:
            gr_mean = np.nanmean(gr)
            gr_std = 1.0

        gr_norm = np.clip((gr - gr_mean) / gr_std, -5, 5)

        # dGR/dMD
        md = sub["MD"].to_numpy(float)
        gr_smooth = np.convolve(gr_norm, np.ones(5) / 5, mode='same')
        dgr_dmd = np.gradient(gr_smooth, md)
        dgr_dmd = np.clip(dgr_dmd * 50.0, -5, 5)  # reduced scaling

        # Geometric features
        anchor = float(sub["last_known_TVT"].iloc[0])
        anchor_md = float(sub["last_known_MD"].iloc[0])

        md_delta = sub["delta_MD_from_PS"].to_numpy(float) / 5000.0
        z_delta = sub["delta_Z_from_PS"].to_numpy(float) / 50.0

        ch = np.stack([
            resample(gr_norm),
            resample(dgr_dmd),
            resample(md_delta),
            resample(z_delta),
        ], axis=0).astype(np.float32)

        ch = np.clip(ch, -10, 10)

        # Target: TVT offset
        tvt = sub["TVT"].to_numpy(float)
        tvt_offset = (tvt - anchor) / 50.0
        tgt = resample(tvt_offset).astype(np.float32)
        tgt = np.clip(tgt, -20, 20)

        # Mask
        is_target = sub["is_target"].astype(bool).to_numpy()
        msk = resample(is_target.astype(float))
        msk = (msk > 0.3).astype(np.float32)

        if msk.sum() > 0:  # Only add if has target region
            features_list.append(ch)
            targets_list.append(tgt)
            masks_list.append(msk)
            well_ids.append(wid)

    print(f"  Built {len(features_list)} wells with targets")

    features = np.stack(features_list, axis=0)
    targets = np.stack(targets_list, axis=0)
    masks = np.stack(masks_list, axis=0)

    return features, targets, masks, well_ids, grid


def masked_mse_loss(pred, target, mask):
    """MSE loss only on masked region."""
    diff = (pred - target) ** 2
    masked = diff * mask
    return masked.sum() / (mask.sum() + 1e-6)


def validate(model, val_loader, device):
    """Compute validation RMSE."""
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for x, tgt, msk in val_loader:
            x, tgt, msk = x.to(device), tgt.to(device), msk.to(device)
            pred = model(x)
            loss = masked_mse_loss(pred, tgt, msk)

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item() * msk.sum().item()
                total_count += msk.sum().item()

    if total_count > 0:
        return np.sqrt(total_loss / total_count) * 50.0
    else:
        return float('inf')


def main():
    ensure_exp_dir(EXP_DIR)
    print(f"\n{'='*60}")
    print(f"[{EXP_ID}] 物理組込み系列NN (simplified v2)")
    print(f"{'='*60}")

    # Load data
    train, test, folds, sample = load_base_inputs(
        ROOT / "data/processed/train_base_v001.parquet",
        ROOT / "data/processed/test_base_v001.parquet",
        ROOT / "data/folds/folds_group_well_v001.csv",
        ROOT / "data/raw/sample_submission.csv"
    )

    train = attach_folds(train, folds)

    print("Building sequences...")
    Xtr, Ytr, Mtr, wtr, grid = build_well_sequences(train)
    Xte, Yte, Mte, wte, _ = build_well_sequences(test)

    print(f"  Train: {Xtr.shape} features, {Ytr.shape} targets, {Mtr.shape} masks")
    print(f"  Well count: {len(wtr)} train, {len(wte)} test")

    # Fold assignment
    wfold = train.groupby("well_id", sort=False)["fold"].first()
    fold_arr = np.array([wfold[w] for w in wtr])
    n_folds = len(np.unique(fold_arr))

    oof_grid_dict = {}
    test_grid_acc = np.zeros((len(wte), L))
    fold_rmses = []

    # Target rows
    tr_t = target_rows(train).copy()
    te_t = target_rows(test).copy()

    def backinterp(wid_list, grid_pred_dict, frame):
        out = np.zeros(len(frame))
        frame = frame.copy()
        frame["_o"] = np.arange(len(frame))

        for wid, sub in frame.groupby("well_id", sort=False):
            sub = sub.sort_values("row_idx")
            ridx = sub["row_idx"].to_numpy(float)
            nrows = float(sub["n_rows_in_well"].iloc[0])
            pos = ridx / max(nrows - 1, 1)

            if wid in grid_pred_dict:
                gp = grid_pred_dict[wid]
                out[sub["_o"].to_numpy()] = np.interp(pos, grid, gp)

        return out

    # 5-fold training
    for fold_id in sorted(np.unique(fold_arr)):
        print(f"\n--- Fold {fold_id} ---")

        val_mask = fold_arr == fold_id
        train_mask = ~val_mask

        train_ds = WellSequenceDataset(
            Xtr[train_mask], Ytr[train_mask], Mtr[train_mask], augment=True
        )
        val_ds = WellSequenceDataset(
            Xtr[val_mask], Ytr[val_mask], Mtr[val_mask], augment=False
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

        # Model
        model = SimpleOffsetTracker(in_channels=Xtr.shape[1], hidden=32).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

        best_val_rmse = float('inf')
        best_state = None
        patience = 0

        # Training
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            train_count = 0
            batch_count = 0

            for x, tgt, msk in train_loader:
                x, tgt, msk = x.to(DEVICE), tgt.to(DEVICE), msk.to(DEVICE)

                opt.zero_grad()
                pred = model(x)
                loss = masked_mse_loss(pred, tgt, msk)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                    train_loss += loss.item() * msk.sum().item()
                    train_count += msk.sum().item()
                    batch_count += 1

            if batch_count > 0:
                sched.step()

            # Validation
            val_rmse = validate(model, val_loader, DEVICE)

            if val_rmse < best_val_rmse - 1e-4:
                best_val_rmse = val_rmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_train = np.sqrt(train_loss / (train_count + 1e-6)) * 50.0 if train_count > 0 else 0
                print(f"  epoch {epoch+1:2d}: train_rmse={avg_train:.3f}, val_rmse={val_rmse:.3f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            print(f"  WARNING: No valid state for fold {fold_id}")

        model.eval()

        # OOF predictions
        with torch.no_grad():
            val_preds = []
            for x, _, _ in val_loader:
                x = x.to(DEVICE)
                pred = model(x)
                val_preds.append((pred * 50.0).cpu().numpy())

            val_preds = np.concatenate(val_preds, axis=0) if val_preds else np.zeros((len(val_mask), L))

            # Test predictions
            test_loader = DataLoader(
                WellSequenceDataset(Xte, Yte, Mte, augment=False),
                batch_size=BATCH, shuffle=False
            )
            test_preds = []
            for x, _, _ in test_loader:
                x = x.to(DEVICE)
                pred = model(x)
                test_preds.append((pred * 50.0).cpu().numpy())

            test_preds = np.concatenate(test_preds, axis=0) if test_preds else np.zeros((len(Xte), L))

        # Store OOF
        val_well_indices = np.where(val_mask)[0]
        for i, wid_idx in enumerate(val_well_indices):
            if i < len(val_preds):
                oof_grid_dict[wtr[wid_idx]] = val_preds[i]

        test_grid_acc += test_preds / n_folds

        print(f"  Fold {fold_id} best val RMSE = {best_val_rmse:.4f}")
        fold_rmses.append(best_val_rmse)

    # Store test grid
    test_grid_dict = {wte[j]: test_grid_acc[j] for j in range(len(wte))}

    # Back-interpolate
    print("\nBack-interpolating to original rows...")
    anchor_tr = tr_t["last_known_TVT"].to_numpy(float)
    anchor_te = te_t["last_known_TVT"].to_numpy(float)

    nn_oof_delta = backinterp(wtr, oof_grid_dict, tr_t)
    nn_test_delta = backinterp(wte, test_grid_dict, te_t)

    # Compute CV
    nn_cv = tvt_rmse(tr_t["TVT"], anchor_tr + nn_oof_delta)
    print(f"\nNeural Tracker OOF CV = {nn_cv:.6f}")

    pf_cv = 11.024
    anchor_cv = 15.909
    print(f"  vs PF (exp022):    {pf_cv:.3f} → diff = {nn_cv - pf_cv:+.3f}")
    print(f"  vs anchor (exp001): {anchor_cv:.3f} → diff = {nn_cv - anchor_cv:+.3f}")

    # Save OOF
    oof_df = tr_t[["id", "well_id", "row_idx", "TVT", "last_known_TVT"]].copy()
    oof_df[PRED_COL] = anchor_tr + nn_oof_delta
    oof_df.to_csv(EXP_DIR / "oof.csv", index=False)

    # Save submission
    test_out = test.copy()
    test_out.loc[test_out["is_target"].astype(bool), PRED_COL] = anchor_te + nn_test_delta
    build_submission(test_out, sample, PRED_COL).to_csv(EXP_DIR / "submission.csv", index=False)

    # Save result
    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "model": "simple-offset-tracker (dilated TCN)",
        "device": str(DEVICE),
        "fold": "folds_group_well_v001",
        "config": {
            "sequence_length": L,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LR,
        },
        "fold_rmses": [float(r) for r in fold_rmses],
        "nn_cv": float(nn_cv),
        "vs_pf": float(nn_cv - pf_cv),
        "vs_anchor": float(nn_cv - anchor_cv),
        "best_prev_cv": 10.062,
        "target": "blend素材・誤差相関確認",
        "leak_risk": "low"
    }

    write_json(EXP_DIR / "result.json", result)
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
