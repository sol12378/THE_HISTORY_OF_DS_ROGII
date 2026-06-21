#!/usr/bin/env python
"""
exp056 Full LOWO: Run TVT LOWO CV on ALL 773 wells (not sampled).

This is slower but gives true validation CV without sampling bias.
Uses formation surface from loaded arrays.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
import glob

ROOT = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')
TRAIN_BASE = ROOT / 'data/processed/train_base_v001.parquet'
RAW_TRAIN_DIR = ROOT / 'data/raw/train'
EXP_OUT_DIR = ROOT / 'experiments/exp056_field_surface'
EXP022_OOF = ROOT / 'experiments/exp022_particle_filter/oof.csv'

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
KNN_K = 8
SUBSAMPLE_THRESHOLD = 2_000_000

EXP_OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_formation_arrays():
    """Load formation data efficiently."""
    print("[Load] Loading formation data from raw CSVs...")

    formation_arrays = {f: [] for f in FORMATIONS}
    well_files = sorted(glob.glob(str(RAW_TRAIN_DIR / '*__horizontal_well.csv')))
    print(f"  Found {len(well_files)} raw well files")

    for i, fpath in enumerate(well_files):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(well_files)} wells...")

        try:
            df = pd.read_csv(fpath, usecols=['X', 'Y'] + FORMATIONS)
            for form in FORMATIONS:
                if form in df.columns:
                    valid = df[['X', 'Y', form]].dropna()
                    if len(valid) > 0:
                        formation_arrays[form].append(valid.values)
        except Exception as e:
            pass

    # Concatenate and subsample
    result = {}
    print("\n  Formation coverage:")
    for form in FORMATIONS:
        if formation_arrays[form]:
            arr = np.vstack(formation_arrays[form])
            n = len(arr)
            if n > SUBSAMPLE_THRESHOLD:
                indices = np.random.choice(n, SUBSAMPLE_THRESHOLD, replace=False)
                arr = arr[np.sort(indices)]
                print(f"    {form}: {n:,} -> {len(arr):,}")
            else:
                print(f"    {form}: {n:,}")
            result[form] = arr
        else:
            result[form] = np.zeros((0, 3))

    return result

def tvt_full_lowo(train_base_df, formation_arrays, best_form):
    """Full LOWO on all 773 wells."""
    print(f"\n[TVT-LOWO] Computing TVT LOWO CV using {best_form}...")

    wells = sorted(train_base_df['well_id'].unique())
    all_rmse = []
    oof_list = []

    form_xy = formation_arrays[best_form][:, :2]
    form_vals = formation_arrays[best_form][:, 2]

    for well_idx, well_id in enumerate(wells):
        if (well_idx + 1) % 50 == 0:
            print(f"  {well_idx+1}/{len(wells)} wells (avg_rmse={np.mean(all_rmse) if all_rmse else 0:.4f})")

        # Get well data
        well_rows = train_base_df[train_base_df['well_id'] == well_id].copy()
        if len(well_rows) == 0:
            continue

        well_xy = well_rows[['X', 'Y', 'Z']].values
        well_tvt_input = well_rows['TVT_input'].values
        well_tvt = well_rows['TVT'].values
        is_known = well_rows['is_known_tvt'].values

        # Build cKDTree from formation arrays
        if len(form_xy) < KNN_K:
            continue

        tree = cKDTree(form_xy)

        # Query well's positions
        distances, indices = tree.query(well_xy[:, :2], k=min(KNN_K, len(form_xy)))

        # Distance-weighted interpolation
        with np.errstate(divide='ignore', invalid='ignore'):
            if distances.ndim == 1:
                weights = 1.0 / (distances + 1e-6)
                s_form = np.average(form_vals[indices], weights=weights)
            else:
                weights = 1.0 / (distances + 1e-6)
                weights /= weights.sum(axis=1, keepdims=True)
                s_form = (weights * form_vals[indices]).sum(axis=1)

        # Estimate const_well from KNOWN rows only
        known_mask = is_known
        if known_mask.sum() > 0:
            known_tvt_input = well_tvt_input[known_mask]
            known_z = well_xy[known_mask, 2]
            known_form = s_form[known_mask] if not np.isscalar(s_form) else s_form

            # const = median(TVT_input - (S_form - Z))
            offset_vals = known_tvt_input - (known_form - known_z)
            const_well = np.nanmedian(offset_vals)

            if np.isnan(const_well):
                continue
        else:
            continue

        # Predict hidden region
        hidden_mask = ~is_known
        if hidden_mask.sum() > 0:
            hidden_z = well_xy[hidden_mask, 2]
            hidden_form = s_form[hidden_mask] if not np.isscalar(s_form) else s_form

            pred_tvt_hidden = hidden_form - hidden_z + const_well
            actual_tvt_hidden = well_tvt[hidden_mask]

            rmse = np.sqrt(mean_squared_error(actual_tvt_hidden, pred_tvt_hidden))
            all_rmse.append(rmse)

            # Store OOF
            oof_list.append(pd.DataFrame({
                'well_id': well_id,
                'row_idx': well_rows[hidden_mask]['row_idx'].values,
                'id': well_rows[hidden_mask]['id'].values,
                'TVT': actual_tvt_hidden,
                'last_known_TVT': well_rows[hidden_mask]['last_known_TVT'].values,
                'pred_tvt': pred_tvt_hidden
            }))

    if all_rmse:
        pooled_cv = np.sqrt(np.mean(np.array(all_rmse) ** 2))
        print(f"\n  TVT LOWO CV: {pooled_cv:.6f} (n_wells={len(all_rmse)})")
        print(f"    Per-well: mean={np.mean(all_rmse):.4f}, median={np.median(all_rmse):.4f}, std={np.std(all_rmse):.4f}")
        print(f"    Min/Max: {np.min(all_rmse):.4f} / {np.max(all_rmse):.4f}")
        print(f"    Broken (>20): {(np.array(all_rmse) > 20).sum()}")
    else:
        pooled_cv = np.inf
        print(f"  TVT LOWO CV: FAILED")

    if oof_list:
        oof_df = pd.concat(oof_list, ignore_index=True)
    else:
        oof_df = pd.DataFrame()

    return pooled_cv, oof_df

def main():
    print("\n" + "="*60)
    print("exp056 FULL LOWO: All 773 wells")
    print("="*60)

    train_base = pd.read_parquet(TRAIN_BASE)
    print(f"Loaded train_base: {train_base.shape}")

    formation_arrays = load_formation_arrays()

    # Use ASTNU as best (from exp056)
    best_form = 'ASTNU'

    tvt_cv, oof_df = tvt_full_lowo(train_base, formation_arrays, best_form)

    # Save full OOF
    oof_path = EXP_OUT_DIR / 'oof_full_773wells.csv'
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved full OOF to {oof_path} ({len(oof_df)} rows)")

    # Update result.json
    result = {
        'tvt_cv_full': tvt_cv,
        'n_wells_validated': len(oof_df['well_id'].unique()),
        'n_rows': len(oof_df),
        'timestamp': pd.Timestamp.now().isoformat()
    }

    with open(EXP_OUT_DIR / 'result_full.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n" + "="*60)
    print("exp056 FULL LOWO completed")
    print("="*60)

if __name__ == '__main__':
    main()
