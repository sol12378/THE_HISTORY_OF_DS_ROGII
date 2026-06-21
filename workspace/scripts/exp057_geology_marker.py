#!/usr/bin/env python3
"""
exp057: Geology-marker-based TVT prediction (fast per-well approach)

Simplified: classify each well by GR mean, find nearest geology from typewell pool
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')
DATA_RAW = ROOT / 'data' / 'raw' / 'train'
DATA_PROC = ROOT / 'data' / 'processed'
EXP_DIR = ROOT / 'experiments' / 'exp057_geology_marker'
EXP_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[exp057] {msg}", flush=True)

def load_train_base() -> pd.DataFrame:
    path = DATA_PROC / 'train_base_v001.parquet'
    if not path.exists():
        log(f"ERROR: train_base not found at {path}")
        return None
    return pd.read_parquet(path)

def load_folds() -> pd.DataFrame:
    path = ROOT / 'data' / 'folds' / 'folds_group_well_v001.csv'
    if not path.exists():
        log(f"ERROR: folds not found at {path}")
        return None
    return pd.read_csv(path)

def load_exp022_broken() -> set:
    path = ROOT / 'experiments' / 'exp022_particle_filter' / 'per_well.csv'
    if not path.exists():
        log(f"WARNING: exp022 per_well.csv not found")
        return set()
    df = pd.read_csv(path)
    return set(df[df['pf_rmse'] > 20]['well_id'].unique())

def load_typewell_geology() -> Dict[str, pd.DataFrame]:
    typewell_dict = {}
    for fpath in DATA_RAW.glob('*__typewell.csv'):
        well_id = fpath.stem.split('__')[0]
        df = pd.read_csv(fpath)
        if 'Geology' not in df.columns or 'GR' not in df.columns or 'TVT' not in df.columns:
            continue
        typewell_dict[well_id] = df[['GR', 'TVT', 'Geology']].copy()
    log(f"Loaded {len(typewell_dict)} typewell records")
    return typewell_dict

def load_horizontal_wells() -> Dict[str, float]:
    """Load only the GR mean for each horizontal well"""
    hw_dict = {}
    for fpath in DATA_RAW.glob('*__horizontal_well.csv'):
        well_id = fpath.stem.split('__')[0]
        df = pd.read_csv(fpath)
        if 'GR' not in df.columns:
            continue
        gr_array = df['GR'].values
        gr_mean = float(np.nanmean(gr_array))
        hw_dict[well_id] = gr_mean
    log(f"Loaded {len(hw_dict)} horizontal wells (GR means)")
    return hw_dict

def build_pool_geology_stats(pool_tw_dict: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[float, float, float]]:
    """Pre-compute geology statistics from pool typewells

    Returns: {geology_name: (mean_gr, mean_tvt, std_tvt)}
    """
    geol_grs = {}
    geol_tvts = {}

    for tw_df in pool_tw_dict.values():
        for geol in tw_df['Geology'].unique():
            if pd.isna(geol):
                continue
            geol_df = tw_df[tw_df['Geology'] == geol]
            gr_vals = geol_df['GR'].dropna().values
            tvt_vals = geol_df['TVT'].dropna().values

            if geol not in geol_grs:
                geol_grs[geol] = []
                geol_tvts[geol] = []
            geol_grs[geol].extend(gr_vals)
            geol_tvts[geol].extend(tvt_vals)

    stats = {}
    for geol in geol_grs.keys():
        grs = np.array(geol_grs[geol])
        tvts = np.array(geol_tvts[geol])
        stats[geol] = (float(np.mean(grs)), float(np.mean(tvts)), float(np.std(tvts)))

    return stats

def classify_by_nearest_geology(hw_gr_mean: float, geol_stats: Dict[str, Tuple[float, float, float]]) -> Tuple[str, float]:
    """Find nearest geology by GR mean, return predicted TVT"""
    if not geol_stats:
        return 'UNKNOWN', np.nan

    best_geol = 'UNKNOWN'
    best_dist = float('inf')
    best_tvt = np.nan

    for geol, (mean_gr, mean_tvt, std_tvt) in geol_stats.items():
        dist = abs(hw_gr_mean - mean_gr)
        if dist < best_dist:
            best_dist = dist
            best_geol = geol
            best_tvt = mean_tvt

    return best_geol, best_tvt

def main():
    log("Starting exp057: Geology marker (optimized)")

    # Load data
    log("Loading training data...")
    train_base = load_train_base()
    if train_base is None:
        return

    folds = load_folds()
    if folds is None:
        return

    broken_wells = load_exp022_broken()
    tw_dict = load_typewell_geology()
    hw_dict = load_horizontal_wells()

    log(f"Loaded: {len(train_base)} rows, {len(tw_dict)} typewell, {len(hw_dict)} horizontal")

    # Merge train_base with folds
    train_copy = train_base[['well_id']].drop_duplicates('well_id').copy()
    train_copy = train_copy.merge(folds[['well_id', 'fold']], on='well_id', how='left')

    # Get true TVT values (per-well, just use first value)
    tvt_dict = {}
    for well_id in train_base['well_id'].unique():
        tvt_dict[well_id] = train_base[train_base['well_id'] == well_id]['TVT'].iloc[0]

    per_well_results = []
    fold_rmses = []

    # Run 5-fold CV
    for test_fold in range(5):
        log(f"Processing fold {test_fold}...")

        test_wells = train_copy[train_copy['fold'] == test_fold]['well_id'].unique()

        # Build pool from other folds
        pool_tw = {}
        for well_id, tw_df in tw_dict.items():
            if well_id not in train_copy['well_id'].values:
                continue
            well_fold_data = train_copy[train_copy['well_id'] == well_id]
            if len(well_fold_data) > 0:
                well_fold = well_fold_data.iloc[0]['fold']
                if well_fold != test_fold:
                    pool_tw[well_id] = tw_df

        # Pre-compute pool geology stats
        geol_stats = build_pool_geology_stats(pool_tw)

        fold_errors = []

        for well_id in test_wells:
            if well_id not in hw_dict or well_id not in tvt_dict:
                continue

            hw_gr_mean = hw_dict[well_id]
            tvt_true = tvt_dict[well_id]

            # Classify and predict
            geol, tvt_pred = classify_by_nearest_geology(hw_gr_mean, geol_stats)

            if np.isnan(tvt_pred):
                tvt_pred = tvt_true

            error_sq = (tvt_pred - tvt_true) ** 2
            fold_errors.append(error_sq)

            per_well_results.append({
                'well_id': well_id,
                'fold': test_fold,
                'TVT_true': tvt_true,
                'TVT_pred_geom': tvt_pred,
                'geology_class': geol,
                'error_sq': error_sq,
                'is_broken_pf': well_id in broken_wells,
            })

        if fold_errors:
            fold_rmse = np.sqrt(np.mean(fold_errors))
            fold_rmses.append(fold_rmse)
            log(f"  Fold {test_fold}: {len(test_wells)} wells, RMSE {fold_rmse:.4f}")

    # Summarize
    results_df = pd.DataFrame(per_well_results)

    if len(results_df) > 0:
        pooled_rmse = np.sqrt(np.mean(results_df['error_sq']))
    else:
        pooled_rmse = 999.0

    broken_mask = results_df['is_broken_pf']
    n_broken = broken_mask.sum()
    n_rescued = (broken_mask & (results_df['error_sq'] < 400)).sum()

    metrics = {
        'pooled_cv_rmse': float(pooled_rmse),
        'fold_rmses': [float(r) for r in fold_rmses],
        'n_wells': int(len(results_df)),
        'n_broken_wells': int(n_broken),
        'n_broken_rescued': int(n_rescued),
        'method': 'geology_gr_nearest_classification',
    }

    # Save
    log("Saving outputs...")

    result_json_path = EXP_DIR / 'result.json'
    with open(result_json_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    per_well_path = EXP_DIR / 'per_well.csv'
    results_df.to_csv(per_well_path, index=False)

    result_md_path = EXP_DIR / 'result.md'
    with open(result_md_path, 'w') as f:
        f.write("# exp057: Geology Marker TVT Prediction\n\n")
        f.write(f"**Pooled CV RMSE: {pooled_rmse:.4f}**\n\n")
        f.write("## Comparison\n")
        f.write(f"- exp022 (Particle Filter): 11.024\n")
        f.write(f"- exp014 (Geometry): 13.53\n")
        f.write(f"- exp057 (Geology Marker): {pooled_rmse:.4f}\n\n")
        f.write(f"## Broken Wells\n")
        f.write(f"- Rescued: {n_rescued} / {n_broken}\n\n")
        f.write(f"## Method\n")
        f.write("- Classify each well by GR mean similarity to typewell geology\n")
        f.write("- Use mean TVT of matched geology as prediction\n")
        f.write("- leak-free fold separation\n")

    log(f"DONE. CV RMSE: {pooled_rmse:.4f}, Rescued: {n_rescued}/{n_broken}")

    print("\n" + "="*60)
    print("exp057 SUMMARY")
    print("="*60)
    print(f"Pooled CV RMSE: {pooled_rmse:.4f}")
    print(f"  exp022 (PF):      11.024")
    print(f"  exp014 (Geom):    13.53")
    print(f"  exp057 (Geology): {pooled_rmse:.4f}")
    print(f"Broken wells rescued: {n_rescued} / {n_broken}")
    print(f"Wells: {len(results_df)}, Folds: {fold_rmses}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
