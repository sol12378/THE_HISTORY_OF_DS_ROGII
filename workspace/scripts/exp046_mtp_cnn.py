#!/usr/bin/env python3
"""
exp046 - MTP-CNN: Multi-modal Trajectory Prediction Loss for GR→TVT learning
Alyaev & Elsheikh 2022 implementation

Purpose: Capture multi-modal GR→TVT(offset) mapping using winner-takes-all MTP loss.
         Target blend complementarity with exp022 PF (error correlation < 0.8).

Strategy:
- MTP modes=5, per-row prediction (pred_steps=1)
- Sequence CNN (1D) with dilated conv + dropout
- 5-fold GroupKFold by well_id
- Strong regularization (dropout 0.3, weight_decay 1e-4)
- GR features: drift-corrected robust norm + geometry
- Output: pred_tvt (best mode) + mode indices + all mode predictions
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings("ignore")

# ============================================================================
# Config
# ============================================================================

REPO_ROOT = Path("/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction")
DATA_DIR = REPO_ROOT / "data"
EXP_DIR = REPO_ROOT / "experiments" / "exp046_mtp_cnn"
SCRIPT_DIR = REPO_ROOT / "scripts"

EXP_DIR.mkdir(parents=True, exist_ok=True)

# Logging
log_file = EXP_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hyperparameters
CONFIG = {
    "seed": 42,
    "device": "cpu",  # CPU only
    "mtp_modes": 5,
    "pred_steps": 1,
    "alpha_class": 0.1,
    "max_seq_len": 1500,
    "hidden_dim": 64,
    "num_layers": 3,
    "dropout": 0.3,
    "batch_size": 16,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 5,
    "grad_clip": 1.0,
    "smoke_test": False,  # Set True for quick validation
    "n_smoke_wells": 2,
    "smoke_epochs": 2,
}

logger.info(f"Config: {CONFIG}")

# ============================================================================
# MTP Loss
# ============================================================================

class MTPLoss(nn.Module):
    """
    Multiple-Trajectory-Prediction Loss (Alyaev & Elsheikh 2022)
    Winner-takes-all mode selection + negative log-likelihood

    Output shape: (batch, modes * (pred_steps + 1))
      - First modes * pred_steps: trajectory predictions
      - Last modes: logit probabilities
    """

    def __init__(self, modes: int = 5, pred_steps: int = 1, alpha_class: float = 0.1):
        super().__init__()
        self.modes = modes
        self.pred_steps = pred_steps
        self.alpha_class = alpha_class

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output: (batch, modes * (pred_steps + 1))
            target: (batch, pred_steps)

        Returns:
            scalar loss
        """
        batch_size = output.shape[0]
        expanded = output.view(batch_size, self.modes, self.pred_steps + 1)

        # Split pred and prob
        pred = expanded[:, :, :self.pred_steps]      # (batch, modes, pred_steps)
        prob_raw = expanded[:, :, self.pred_steps]    # (batch, modes)

        # L1 distance per mode (average over pred_steps)
        dists = (pred - target.unsqueeze(1)).abs().mean(dim=2)  # (batch, modes)

        # Best mode per sample (detach to avoid mode-collapse gradient)
        best_mode = dists.argmin(1).detach()  # (batch,)

        # Log softmax (stable)
        log_prob = F.log_softmax(prob_raw, dim=1)  # (batch, modes)

        # Loss: trajectory error at best mode - alpha * log_prob at best mode
        batch_idx = torch.arange(batch_size, device=output.device)
        trajectory_loss = dists[batch_idx, best_mode]
        log_prob_loss = log_prob[batch_idx, best_mode]
        loss = (trajectory_loss - self.alpha_class * log_prob_loss).mean()

        return loss


# ============================================================================
# Dataset
# ============================================================================

class GRTVTDataset(Dataset):
    """Per-well GR→TVT sequence dataset"""

    def __init__(
        self,
        well_ids: List[str],
        gr_sequences: Dict[str, np.ndarray],
        geom_sequences: Dict[str, np.ndarray],
        tvt_targets: Dict[str, np.ndarray],
        max_seq_len: int = 1500,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
    ):
        self.well_ids = well_ids
        self.gr_sequences = gr_sequences
        self.geom_sequences = geom_sequences
        self.tvt_targets = tvt_targets
        self.max_seq_len = max_seq_len
        self.scaler = scaler

        if fit_scaler and scaler is not None:
            # Fit on all wells
            all_seqs = np.vstack([gr_sequences[wid] for wid in well_ids if wid in gr_sequences])
            scaler.fit(all_seqs)

    def __len__(self):
        return len(self.well_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        well_id = self.well_ids[idx]

        gr_seq = self.gr_sequences[well_id]  # (seq_len, gr_features)
        geom_seq = self.geom_sequences[well_id]  # (seq_len, geom_features)
        tvt_seq = self.tvt_targets[well_id]  # (seq_len,)

        # Standardize GR
        if self.scaler is not None:
            gr_seq = self.scaler.transform(gr_seq)

        # Concatenate features
        X = np.hstack([gr_seq, geom_seq])  # (seq_len, all_features)

        # Pad to max_seq_len
        seq_len = X.shape[0]
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        if seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            X = np.vstack([X, np.zeros((pad_len, X.shape[1]))])
            tvt_seq = np.hstack([tvt_seq, np.zeros(pad_len)])
            mask[:seq_len] = 1.0
        else:
            X = X[:self.max_seq_len]
            tvt_seq = tvt_seq[:self.max_seq_len]
            mask[:] = 1.0

        return {
            "well_id": well_id,
            "X": torch.from_numpy(X).float(),  # (max_seq_len, features)
            "y": torch.from_numpy(tvt_seq).float(),  # (max_seq_len,)
            "mask": torch.from_numpy(mask).float(),  # (max_seq_len,)
        }


# ============================================================================
# Model
# ============================================================================

class MTPCNN(nn.Module):
    """
    MTP CNN: 1D dilated CNN for sequence→multi-modal output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mtp_modes: int = 5,
        pred_steps: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mtp_modes = mtp_modes
        self.pred_steps = pred_steps
        self.output_dim = mtp_modes * (pred_steps + 1)

        # Dilated 1D CNN layers
        layers = []
        in_channels = input_dim
        dilation = 1

        for i in range(num_layers):
            out_channels = hidden_dim
            kernel_size = 3

            layers.append(nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
                bias=True
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            in_channels = out_channels
            dilation = 2 ** (i + 1)  # Exponential dilation

        self.cnn = nn.Sequential(*layers)

        # Per-step output head
        self.head = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, seq_len, features)
            mask: (batch, seq_len) binary mask

        Returns:
            output: (batch, seq_len, mtp_modes * (pred_steps + 1))
        """
        # Transpose for 1D conv: (batch, features, seq_len)
        X = X.transpose(1, 2)

        # CNN processing
        out = self.cnn(X)  # (batch, hidden_dim, seq_len)

        # Transpose back: (batch, seq_len, hidden_dim)
        out = out.transpose(1, 2)

        # Per-step output
        out = self.head(out)  # (batch, seq_len, output_dim)

        # Mask: set invalid positions to 0
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        out = out * mask

        return out


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: float = 1.0,
) -> float:
    """Train one epoch, return mean loss"""
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch in loader:
        X = batch["X"].to(device)  # (batch, seq_len, features)
        y = batch["y"].to(device)  # (batch, seq_len)
        mask = batch["mask"].to(device)  # (batch, seq_len)

        optimizer.zero_grad()

        # Forward
        output = model(X, mask)  # (batch, seq_len, output_dim)

        # Loss per valid position
        valid_mask = mask > 0.5  # (batch, seq_len)
        valid_idx = torch.nonzero(valid_mask, as_tuple=True)

        if len(valid_idx[0]) > 0:
            valid_output = output[valid_idx]  # (num_valid, output_dim)
            valid_y = y[valid_idx]  # (num_valid,)

            loss = loss_fn(valid_output, valid_y.unsqueeze(-1))

            # Backward
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * valid_idx[0].shape[0]
            num_samples += valid_idx[0].shape[0]

    mean_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return mean_loss


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray]:
    """Evaluate, return loss and predictions"""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    all_preds = []
    all_targets = []
    all_masks = []

    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        output = model(X, mask)  # (batch, seq_len, output_dim)

        valid_mask = mask > 0.5
        valid_idx = torch.nonzero(valid_mask, as_tuple=True)

        if len(valid_idx[0]) > 0:
            valid_output = output[valid_idx]
            valid_y = y[valid_idx]

            loss = loss_fn(valid_output, valid_y.unsqueeze(-1))
            total_loss += loss.item() * valid_idx[0].shape[0]
            num_samples += valid_idx[0].shape[0]

        all_preds.append(output.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    mean_loss = total_loss / num_samples if num_samples > 0 else float('inf')

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    masks = np.vstack(all_masks)

    return mean_loss, preds, targets, masks


def predict_best_mode(output_np: np.ndarray, pred_steps: int = 1, modes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract best mode predictions from MTP output

    Args:
        output_np: (seq_len, modes * (pred_steps + 1))

    Returns:
        best_preds: (seq_len,)
        best_modes: (seq_len,)
    """
    seq_len = output_np.shape[0]
    output_reshaped = output_np.reshape(seq_len, modes, pred_steps + 1)

    pred = output_reshaped[:, :, :pred_steps]  # (seq_len, modes, pred_steps)
    prob_raw = output_reshaped[:, :, pred_steps]  # (seq_len, modes)

    # Argmax probability (greedy)
    best_modes = prob_raw.argmax(axis=1)  # (seq_len,)
    best_preds = pred[np.arange(seq_len), best_modes, 0]  # (seq_len,)

    return best_preds, best_modes


# ============================================================================
# Data Loading
# ============================================================================

def load_data(smoke_test: bool = False, n_smoke_wells: int = 2):
    """Load ROGII data from processed parquet + P2 features"""
    logger.info("Loading data...")

    # Processed train/test
    train_df = pd.read_parquet(DATA_DIR / "processed" / "train_base_v001.parquet")
    test_df = pd.read_parquet(DATA_DIR / "processed" / "test_base_v001.parquet")

    logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # P2 features
    train_p2 = pd.read_parquet(REPO_ROOT / "experiments" / "features_p2" / "train_p2.parquet")
    train_df = train_df.merge(train_p2, on=["well_id", "row_idx"], how="left")

    # Folds
    folds_df = pd.read_csv(DATA_DIR / "folds" / "folds_group_well_v001.csv")
    train_folds = folds_df[folds_df["split"] == "train"].copy()

    # Get unique wells, filter for only hidden rows (is_target=True)
    train_target = train_df[train_df["is_target"] == True].copy()
    unique_wells = train_folds["well_id"].unique()

    if smoke_test:
        unique_wells = unique_wells[:n_smoke_wells]
        logger.info(f"SMOKE TEST: Using {len(unique_wells)} wells")

    logger.info(f"Total wells: {len(unique_wells)}")

    # Build sequences per well (hidden rows only)
    gr_sequences = {}
    geom_sequences = {}
    tvt_targets = {}

    for well_id in unique_wells:
        well_data = train_target[train_target["well_id"] == well_id].sort_values("row_idx")

        if well_data.shape[0] == 0:
            logger.warning(f"Well {well_id} has no hidden rows")
            continue

        # GR: robust norm (drift-corrected, handle NaN)
        gr = well_data["GR"].fillna(well_data["GR"].mean()).values.astype(np.float32)
        gr_rolling_median = pd.Series(gr).rolling(window=51, center=True, min_periods=1).median().values
        gr_mad = np.abs(gr - gr_rolling_median).mean()
        gr_norm = (gr - gr_rolling_median) / (gr_mad + 1e-6)

        # Geometry + P2: MD, Z, delta_MD, delta_Z, dls_mean, dls_last30, inclination, tort_3d, knn_surface_minus_Z
        geom_cols = ["delta_MD_from_PS", "delta_Z_from_PS", "dls_mean", "dls_last30",
                     "inclination_last", "tort_3d", "knn_surface_minus_Z"]

        geom_dict = {}
        for col in geom_cols:
            if col in well_data.columns:
                geom_dict[col] = well_data[col].fillna(well_data[col].mean()).values.astype(np.float32)
            else:
                logger.warning(f"Column {col} missing for well {well_id}, using zeros")
                geom_dict[col] = np.zeros(well_data.shape[0], dtype=np.float32)

        geom_features = np.column_stack([geom_dict[col] for col in geom_cols]).astype(np.float32)

        # Target: delta TVT
        tvt = well_data["TVT"].values.astype(np.float32)
        last_known = well_data["last_known_TVT"].values.astype(np.float32)
        delta_tvt = tvt - last_known

        gr_sequences[well_id] = gr_norm.reshape(-1, 1)
        geom_sequences[well_id] = geom_features
        tvt_targets[well_id] = delta_tvt

    logger.info(f"Loaded {len(gr_sequences)} wells with sequences")

    return (
        train_df,
        test_df,
        train_folds,
        unique_wells,
        gr_sequences,
        geom_sequences,
        tvt_targets,
    )


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("exp046 MTP-CNN Training Start")
    logger.info("="*80)

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Load data
    train_df, test_df, train_folds, unique_wells, gr_sequences, geom_sequences, tvt_targets = load_data(
        smoke_test=CONFIG["smoke_test"],
        n_smoke_wells=CONFIG["n_smoke_wells"],
    )

    # Separate hidden rows for OOF
    train_target = train_df[train_df["is_target"] == True].copy()

    # Setup fold splits
    gkf = GroupKFold(n_splits=5)
    fold_assignments = {}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(
        train_folds, groups=train_folds["well_id"]
    )):
        train_wells = train_folds.iloc[train_idx]["well_id"].values
        val_wells = train_folds.iloc[val_idx]["well_id"].values

        fold_assignments[fold_idx] = {
            "train_wells": train_wells,
            "val_wells": val_wells,
        }

    logger.info(f"Fold setup complete: {len(fold_assignments)} folds")

    # Initialize loss and scalers
    loss_fn = MTPLoss(
        modes=CONFIG["mtp_modes"],
        pred_steps=CONFIG["pred_steps"],
        alpha_class=CONFIG["alpha_class"],
    ).to(CONFIG["device"])

    # Training per fold
    oof_list = []
    fold_rmses = {}

    # input_dim: GR (1) + geometry features (7: delta_MD, delta_Z, dls_mean, dls_last30, inclination, tort_3d, knn_surface_minus_Z)
    input_dim = 1 + 7

    for fold_idx in range(5):
        logger.info(f"\n{'='*80}")
        logger.info(f"FOLD {fold_idx}")
        logger.info(f"{'='*80}")

        train_wells = [w for w in unique_wells if w in fold_assignments[fold_idx]["train_wells"]]
        val_wells = [w for w in unique_wells if w in fold_assignments[fold_idx]["val_wells"]]

        logger.info(f"Train wells: {len(train_wells)}, Val wells: {len(val_wells)}")

        if len(val_wells) == 0:
            logger.warning(f"No validation wells in fold {fold_idx}, skipping")
            continue

        # Dataset & loader
        scaler_train = StandardScaler()
        ds_train = GRTVTDataset(
            train_wells,
            gr_sequences,
            geom_sequences,
            tvt_targets,
            max_seq_len=CONFIG["max_seq_len"],
            scaler=scaler_train,
            fit_scaler=True,
        )
        ds_val = GRTVTDataset(
            val_wells,
            gr_sequences,
            geom_sequences,
            tvt_targets,
            max_seq_len=CONFIG["max_seq_len"],
            scaler=scaler_train,
            fit_scaler=False,
        )

        loader_train = DataLoader(ds_train, batch_size=CONFIG["batch_size"], shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=CONFIG["batch_size"], shuffle=False)

        # Model
        model = MTPCNN(
            input_dim=input_dim,
            hidden_dim=CONFIG["hidden_dim"],
            num_layers=CONFIG["num_layers"],
            mtp_modes=CONFIG["mtp_modes"],
            pred_steps=CONFIG["pred_steps"],
            dropout=CONFIG["dropout"],
        ).to(CONFIG["device"])

        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

        num_epochs = CONFIG["smoke_epochs"] if CONFIG["smoke_test"] else CONFIG["epochs"]
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(num_epochs):
            train_loss = train_epoch(
                model, loader_train, loss_fn, optimizer, CONFIG["device"],
                grad_clip=CONFIG["grad_clip"],
            )

            val_loss, _, _, _ = eval_epoch(model, loader_val, loss_fn, CONFIG["device"])

            logger.info(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Validation OOF
        _, val_preds, val_targets, val_masks = eval_epoch(
            model, loader_val, loss_fn, CONFIG["device"]
        )

        # Extract best mode predictions
        val_rmses = []
        for well_idx, well_id in enumerate(val_wells):
            well_pred = val_preds[well_idx]
            well_mask = val_masks[well_idx]

            best_pred, best_mode = predict_best_mode(
                well_pred,
                pred_steps=CONFIG["pred_steps"],
                modes=CONFIG["mtp_modes"],
            )

            valid_mask = well_mask > 0.5
            valid_pred = best_pred[valid_mask]

            # Reconstruct TVT from hidden rows (use train_target which has full data)
            well_data = train_target[train_target["well_id"] == well_id].sort_values("row_idx").reset_index(drop=True)

            if well_data.shape[0] == 0:
                continue

            tvt_true = well_data["TVT"].values
            last_known_tvt = well_data["last_known_TVT"].values
            ids = well_data["id"].values  # Get actual id column

            # Create valid mask for this well's sequences
            seq_len = well_data.shape[0]

            if len(valid_pred) > 0:
                # valid_pred and best_mode have the same length (number of valid positions)
                pred_tvt = valid_pred + last_known_tvt[:len(valid_pred)]
                true_tvt = tvt_true[:len(valid_pred)]
                rmse = np.sqrt(np.mean((pred_tvt - true_tvt) ** 2))
                val_rmses.append(rmse)

                # OOF entry per valid row (only include rows where mask=1)
                valid_seq_indices = np.where(valid_mask)[0]
                for out_idx, (seq_idx, row_pred, row_mode) in enumerate(
                    zip(valid_seq_indices, valid_pred, best_mode[valid_mask])
                ):
                    if seq_idx < len(ids):
                        oof_list.append({
                            "well_id": well_id,
                            "row_idx": seq_idx,
                            "id": ids[seq_idx],
                            "TVT": tvt_true[seq_idx],
                            "last_known_TVT": last_known_tvt[seq_idx],
                            "pred_tvt": row_pred + last_known_tvt[seq_idx],
                            "pred_tvt_mode_idx": int(row_mode),
                            "fold": fold_idx,
                        })

        fold_cv = np.mean(val_rmses) if val_rmses else float('inf')
        fold_rmses[fold_idx] = fold_cv
        logger.info(f"Fold {fold_idx} CV RMSE: {fold_cv:.6f}")

    # Combine OOF
    oof_df = pd.DataFrame(oof_list)

    # Ensure id is sequential per well (like exp041 format)
    if 'id' in oof_df.columns:
        def assign_sequential_ids(group):
            group['id'] = range(len(group))
            return group

        oof_df = oof_df.groupby('well_id', group_keys=False).apply(assign_sequential_ids)
        logger.info(f"OOF with sequential ids: {oof_df.shape}, id range: {oof_df['id'].min()}-{oof_df['id'].max()}")

    if oof_df.shape[0] > 0:
        oof_df = oof_df.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
        oof_df.to_csv(EXP_DIR / "oof.csv", index=False)
        logger.info(f"OOF saved: {oof_df.shape}")

    # Pooled CV
    if "TVT" in oof_df.columns and "pred_tvt" in oof_df.columns:
        pooled_rmse = np.sqrt(np.mean((oof_df["TVT"].values - oof_df["pred_tvt"].values) ** 2))
        logger.info(f"Pooled CV RMSE: {pooled_rmse:.6f}")
    else:
        pooled_rmse = float('inf')

    # Error correlation vs exp022
    try:
        exp022_oof = pd.read_csv(REPO_ROOT / "experiments" / "exp022_particle_filter" / "oof.csv")
        oof_merged = oof_df.merge(exp022_oof[["id", "pred_tvt"]], on="id", suffixes=("", "_exp022"))
        if oof_merged.shape[0] > 0:
            error_exp046 = oof_merged["TVT"] - oof_merged["pred_tvt"]
            error_exp022 = oof_merged["TVT"] - oof_merged["pred_tvt_exp022"]
            error_corr = np.corrcoef(error_exp046, error_exp022)[0, 1]
            logger.info(f"Error correlation vs exp022: {error_corr:.6f}")
        else:
            error_corr = np.nan
    except Exception as e:
        logger.warning(f"Could not compute error correlation: {e}")
        error_corr = np.nan

    # Result JSON
    result = {
        "cv_rmse": float(pooled_rmse),
        "fold_rmses": {int(k): float(v) for k, v in fold_rmses.items()},
        "mtp_modes": CONFIG["mtp_modes"],
        "pred_steps": CONFIG["pred_steps"],
        "error_corr_vs_exp022": float(error_corr) if not np.isnan(error_corr) else None,
        "timing": "smoke" if CONFIG["smoke_test"] else "full",
    }

    with open(EXP_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Result: {result}")
    logger.info("="*80)
    logger.info("exp046 Training Complete")
    logger.info("="*80)

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["cv_rmse"] < 15 else 1)
