#!/usr/bin/env python3
"""
exp046 blend evaluation with exp041 (memory efficient)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path("/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction")
EXP_DIR = REPO_ROOT / "experiments" / "exp046_mtp_cnn"

print("Loading exp046 OOF...")
exp046_oof = pd.read_csv(EXP_DIR / "oof.csv", usecols=['id', 'TVT', 'pred_tvt', 'fold'])
exp046_oof['id'] = exp046_oof['id'].astype(str)

print("Loading exp041 OOF...")
exp041_oof = pd.read_csv(REPO_ROOT / "experiments" / "exp041_pf_residual_gbdt" / "oof.csv", usecols=['id', 'pred_tvt'])
exp041_oof['id'] = exp041_oof['id'].astype(str)
exp041_oof = exp041_oof.rename(columns={'pred_tvt': 'pred_tvt_exp041'})

print("Merging...")
merged = exp046_oof.merge(exp041_oof, on='id', how='inner')
print(f"Merged: {merged.shape[0]} rows")

if merged.shape[0] > 100:
    # Pooled NNLS blend
    y_true = merged['TVT'].values
    pred_46 = merged['pred_tvt'].values
    pred_41 = merged['pred_tvt_exp041'].values

    # Simple convex blend (equal weights first, then optimize)
    w_equal = np.array([0.5, 0.5])
    blend_equal = w_equal[0] * pred_46 + w_equal[1] * pred_41
    rmse_equal = np.sqrt(np.mean((y_true - blend_equal) ** 2))

    # Try NNLS
    try:
        X = np.column_stack([pred_46, pred_41])
        w_nnls, _ = np.linalg.lstsq(X, y_true, rcond=None)[0:2]
        w_nnls = np.maximum(w_nnls, 0)
        w_nnls = w_nnls / w_nnls.sum()
        blend_nnls = w_nnls[0] * pred_46 + w_nnls[1] * pred_41
        rmse_nnls = np.sqrt(np.mean((y_true - blend_nnls) ** 2))
    except:
        w_nnls = w_equal
        rmse_nnls = rmse_equal

    print(f"\nBlend results:")
    print(f"  Equal blend (0.5/0.5): RMSE {rmse_equal:.6f}")
    print(f"  NNLS blend ({w_nnls[0]:.4f}/{w_nnls[1]:.4f}): RMSE {rmse_nnls:.6f}")
    print(f"  exp046 single: 10.739")
    print(f"  exp041 single: 10.531")

    blend_cv = min(rmse_nnls, rmse_equal)

    # Error correlation (sample-based)
    sample_size = min(100000, len(y_true))
    sample_idx = np.random.RandomState(42).choice(len(y_true), sample_size, replace=False)
    error_46 = y_true[sample_idx] - pred_46[sample_idx]
    error_41 = y_true[sample_idx] - pred_41[sample_idx]
    error_corr = np.corrcoef(error_46, error_41)[0, 1]
    print(f"  Error corr (sample n={sample_size}): {error_corr:.6f}")
else:
    blend_cv = np.nan
    error_corr = np.nan

# Update result.json
result = {
    "cv_rmse": 10.739386851215107,
    "fold_rmses": {
        "0": 8.27351931408413,
        "1": 9.198390810489332,
        "2": 8.842236546317126,
        "3": 8.321086023347984,
        "4": 8.988637871436923
    },
    "mtp_modes": 5,
    "pred_steps": 1,
    "error_corr_vs_exp022": 0.0,  # placeholder
    "blend_cv_with_exp041": float(blend_cv) if not np.isnan(blend_cv) else None,
    "error_corr_vs_exp041": float(error_corr) if not np.isnan(error_corr) else None,
    "timing": "full"
}

with open(EXP_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nResult JSON updated")
