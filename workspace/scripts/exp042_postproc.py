"""
exp042: Post-processing validation for exp041 OOF
- P3: Savitzky-Golay smoothing vs mean smoothing
- P4: Regime binning (well characteristics) with adaptive smoothing

Baseline (exp041): CV RMSE = 10.531
Goal: Improve via fold-consistent post-processing rules
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# Setup
# ============================================================================

ROOT = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')
EXP_ID = 'exp042'
EXP_DIR = ROOT / 'experiments' / EXP_ID
EXP_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = EXP_DIR / 'run.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Input paths
OOF_PATH = ROOT / 'experiments' / 'exp041_pf_residual_gbdt' / 'oof.csv'
TRAIN_BASE_PATH = ROOT / 'data' / 'processed' / 'train_base_v001.parquet'

def load_data():
    """Load OOF and well characteristics."""
    logger.info("Loading exp041 OOF...")
    oof = pd.read_csv(OOF_PATH)
    logger.info(f"OOF shape: {oof.shape}, columns: {oof.columns.tolist()}")

    logger.info("Loading train_base for well characteristics...")
    train_base = pd.read_parquet(TRAIN_BASE_PATH)
    logger.info(f"train_base shape: {train_base.shape}")

    # Compute well-level characteristics
    # hidden_length: per row (constant within well)
    # delta_Z_total: sum of absolute deltas per well
    well_chars = train_base.groupby('well_id').agg({
        'hidden_length': 'first',
        'delta_Z_from_PS': lambda x: np.abs(x).sum()
    }).rename(columns={'delta_Z_from_PS': 'delta_Z_total'}).reset_index()

    # Merge well characteristics back to train_base
    train_base = train_base.merge(well_chars[['well_id', 'delta_Z_total']], on='well_id', how='left')

    # Merge to OOF to get well characteristics
    oof = oof.merge(
        train_base[['row_idx', 'hidden_length', 'delta_Z_total']].drop_duplicates('row_idx'),
        on='row_idx',
        how='left'
    )

    logger.info(f"OOF after merge shape: {oof.shape}")
    logger.info(f"NaN counts: {oof[['hidden_length', 'delta_Z_total']].isna().sum()}")

    return oof, train_base

def compute_cv_rmse(y_true, y_pred):
    """Compute RMSE for CV evaluation."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def baseline_cv(oof):
    """Compute baseline CV (current mean-smoothed OOF)."""
    cv = compute_cv_rmse(oof['TVT'].values, oof['pred_tvt'].values)
    logger.info(f"Baseline mean-smooth CV RMSE: {cv:.6f}")
    return cv

# ============================================================================
# P3: Savitzky-Golay Smoothing
# ============================================================================

def apply_savgol_per_well(oof, window, polyorder):
    """Apply Savitzky-Golay filter per well."""
    oof = oof.copy()

    for well_id in oof['well_id'].unique():
        mask = oof['well_id'] == well_id
        well_data = oof.loc[mask].sort_values('row_idx')

        # Ensure window is valid
        n_rows = len(well_data)
        window_adj = min(window, n_rows)
        if window_adj % 2 == 0:
            window_adj -= 1
        window_adj = max(window_adj, polyorder + 1)

        if n_rows >= polyorder + 1:
            try:
                smoothed = savgol_filter(
                    well_data['pred_tvt'].values,
                    window_length=window_adj,
                    polyorder=polyorder
                )
                oof.loc[mask, 'pred_tvt'] = smoothed
            except Exception as e:
                logger.warning(f"Savgol failed for well {well_id}: {e}")

    return oof

def optimize_savgol(oof, gkf_split):
    """Grid search for best Savitzky-Golay parameters."""
    logger.info("\n=== P3: Savitzky-Goyal Optimization ===")

    windows = [51, 75, 101, 151]
    polyorders = [2, 3]

    results = []

    for window in windows:
        for polyorder in polyorders:
            logger.info(f"Testing savgol window={window} polyorder={polyorder}")

            fold_cvs = []
            for fold_idx, (train_idx, val_idx) in enumerate(gkf_split):
                oof_val = oof.iloc[val_idx].copy()
                oof_val = apply_savgol_per_well(oof_val, window, polyorder)

                fold_cv = compute_cv_rmse(oof_val['TVT'].values, oof_val['pred_tvt'].values)
                fold_cvs.append(fold_cv)
                logger.info(f"  Fold {fold_idx}: CV RMSE = {fold_cv:.6f}")

            pooled_cv = np.mean(fold_cvs)
            fold_consistency = np.std(fold_cvs)

            results.append({
                'method': 'savgol',
                'window': window,
                'polyorder': polyorder,
                'pooled_cv': pooled_cv,
                'fold_cvs': fold_cvs,
                'fold_mean': np.mean(fold_cvs),
                'fold_std': np.std(fold_cvs),
            })

            logger.info(f"  Pooled CV RMSE = {pooled_cv:.6f} (std={fold_consistency:.6f})")

    best_savgol = min(results, key=lambda x: x['pooled_cv'])
    logger.info(f"\nBest savgol: window={best_savgol['window']} polyorder={best_savgol['polyorder']} CV={best_savgol['pooled_cv']:.6f}")

    return best_savgol, results

# ============================================================================
# P4: Regime Binning with Adaptive Smoothing
# ============================================================================

def classify_wells(oof):
    """Classify wells by (hidden_length, delta_Z_span) into regimes."""
    oof = oof.copy()

    # Compute absolute delta_Z
    oof['z_span'] = oof['delta_Z_total'].abs()

    # Hidden length bins
    oof['hidden_bin'] = pd.cut(
        oof['hidden_length'],
        bins=[0, 2000, 5000, np.inf],
        labels=['short', 'mid', 'long'],
        include_lowest=True
    )

    # Z span bins
    oof['z_bin'] = pd.cut(
        oof['z_span'],
        bins=[0, 100, np.inf],
        labels=['flat', 'steep'],
        include_lowest=True
    )

    oof['regime'] = oof['hidden_bin'].astype(str) + '_' + oof['z_bin'].astype(str)

    logger.info("\nWell regime distribution:")
    regime_counts = oof.groupby('regime')['well_id'].nunique()
    logger.info(regime_counts)

    return oof

def optimize_smoothing_per_regime(oof, gkf_split):
    """Optimize smoothing window per regime."""
    logger.info("\n=== P4: Regime-Based Adaptive Smoothing ===")

    # Classify wells
    oof = classify_wells(oof)

    # Define regime-specific window ranges
    # Intuition: short+flat → small window (less noise), long → large window
    regime_windows = {
        'short_flat': [25, 51],
        'short_steep': [35, 51],
        'mid_flat': [51, 75],
        'mid_steep': [75, 101],
        'long_flat': [75, 151],
        'long_steep': [101, 151],
    }

    results = []
    best_per_regime = {}

    # For each regime, find best window
    for regime, windows in regime_windows.items():
        if regime not in oof['regime'].unique():
            logger.info(f"Regime {regime} not found, skipping")
            continue

        logger.info(f"\nOptimizing regime: {regime}")
        regime_results = []

        for window in windows:
            fold_cvs = []

            for fold_idx, (train_idx, val_idx) in enumerate(gkf_split):
                oof_val = oof.iloc[val_idx].copy()

                # Apply window only to this regime
                mask = oof_val['regime'] == regime
                if not mask.any():
                    continue

                oof_val_regime = oof_val[mask].copy()
                oof_val_regime = apply_savgol_per_well(oof_val_regime, window, polyorder=2)
                oof_val.loc[mask, 'pred_tvt'] = oof_val_regime['pred_tvt'].values

                fold_cv = compute_cv_rmse(oof_val['TVT'].values, oof_val['pred_tvt'].values)
                fold_cvs.append(fold_cv)

            if fold_cvs:
                regime_results.append({
                    'regime': regime,
                    'window': window,
                    'fold_cvs': fold_cvs,
                    'fold_mean': np.mean(fold_cvs),
                    'fold_std': np.std(fold_cvs),
                })
                logger.info(f"  window={window}: CV={np.mean(fold_cvs):.6f} (std={np.std(fold_cvs):.6f})")

        if regime_results:
            best = min(regime_results, key=lambda x: x['fold_mean'])
            best_per_regime[regime] = best
            logger.info(f"Best window for {regime}: {best['window']} (CV={best['fold_mean']:.6f})")
            results.extend(regime_results)

    # Full pooled CV with all regime windows
    logger.info("\nEvaluating full pooled CV with regime-specific windows...")
    fold_cvs_full = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf_split):
        oof_val = oof.iloc[val_idx].copy()

        # Apply best window per regime
        for regime, best_cfg in best_per_regime.items():
            mask = oof_val['regime'] == regime
            if not mask.any():
                continue

            oof_val_regime = oof_val[mask].copy()
            oof_val_regime = apply_savgol_per_well(oof_val_regime, best_cfg['window'], polyorder=2)
            oof_val.loc[mask, 'pred_tvt'] = oof_val_regime['pred_tvt'].values

        fold_cv = compute_cv_rmse(oof_val['TVT'].values, oof_val['pred_tvt'].values)
        fold_cvs_full.append(fold_cv)
        logger.info(f"Fold {fold_idx} pooled CV: {fold_cv:.6f}")

    pooled_cv_selector = np.mean(fold_cvs_full)
    logger.info(f"\nSelector pooled CV RMSE: {pooled_cv_selector:.6f} (std={np.std(fold_cvs_full):.6f})")

    return {
        'method': 'selector',
        'pooled_cv': pooled_cv_selector,
        'fold_cvs': fold_cvs_full,
        'fold_mean': np.mean(fold_cvs_full),
        'fold_std': np.std(fold_cvs_full),
        'best_per_regime': best_per_regime,
        'regime_results': results,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    logger.info(f"Starting {EXP_ID}: Post-processing Optimization")
    logger.info(f"Root: {ROOT}")

    # Load data
    oof, train_base = load_data()

    # Compute baseline CV
    baseline = baseline_cv(oof)

    # Setup GroupKFold by well_id
    logger.info("\nSetting up GroupKFold by well_id...")
    gkf = GroupKFold(n_splits=5)
    gkf_split = list(gkf.split(oof, groups=oof['well_id']))
    logger.info(f"GroupKFold splits: {len(gkf_split)}")

    # P3: Savitzky-Golay
    best_savgol, savgol_results = optimize_savgol(oof, gkf_split)

    # P4: Regime binning
    best_selector = optimize_smoothing_per_regime(oof, gkf_split)

    # ========================================================================
    # Decision Logic
    # ========================================================================

    logger.info("\n" + "="*70)
    logger.info("SUMMARY & DECISION")
    logger.info("="*70)

    logger.info(f"Baseline (mean-smooth):    CV RMSE = {baseline:.6f}")
    logger.info(f"Best Savgol (w={best_savgol['window']} p={best_savgol['polyorder']}): CV RMSE = {best_savgol['pooled_cv']:.6f}")
    logger.info(f"Best Selector:             CV RMSE = {best_selector['pooled_cv']:.6f}")

    # Fold consistency check
    savgol_consistent = all(cv < baseline + 0.01 for cv in best_savgol['fold_cvs'])  # loose threshold
    selector_consistent = all(cv < baseline + 0.01 for cv in best_selector['fold_cvs'])

    logger.info(f"\nFold consistency:")
    logger.info(f"  Savgol fold CVs: {[f'{cv:.6f}' for cv in best_savgol['fold_cvs']]}")
    logger.info(f"  Consistent (all < baseline+0.01)? {savgol_consistent}")
    logger.info(f"  Selector fold CVs: {[f'{cv:.6f}' for cv in best_selector['fold_cvs']]}")
    logger.info(f"  Consistent (all < baseline+0.01)? {selector_consistent}")

    # Select best method
    if best_savgol['pooled_cv'] < baseline and savgol_consistent:
        recommended = 'savgol'
        improvement = baseline - best_savgol['pooled_cv']
        best_cfg = best_savgol
        logger.info(f"\n✓ Recommended: Savgol (improvement: +{improvement:.6f})")
    elif best_selector['pooled_cv'] < baseline and selector_consistent:
        recommended = 'selector'
        improvement = baseline - best_selector['pooled_cv']
        best_cfg = best_selector
        logger.info(f"\n✓ Recommended: Selector (improvement: +{improvement:.6f})")
    else:
        recommended = 'mean_smooth'
        improvement = 0
        logger.info(f"\n✓ Recommended: Mean-smooth (no fold-consistent improvement found)")

    # ========================================================================
    # Generate best OOF
    # ========================================================================

    logger.info(f"\nGenerating best OOF with {recommended}...")
    oof_best = oof.copy()

    if recommended == 'savgol':
        window = best_savgol['window']
        polyorder = best_savgol['polyorder']
        logger.info(f"Applying savgol: window={window}, polyorder={polyorder}")
        oof_best = apply_savgol_per_well(oof_best, window, polyorder)

    elif recommended == 'selector':
        oof_best = classify_wells(oof_best)
        best_per_regime = best_selector['best_per_regime']
        logger.info(f"Applying regime-specific smoothing:")
        for regime, cfg in best_per_regime.items():
            logger.info(f"  {regime}: window={cfg['window']}")
            mask = oof_best['regime'] == regime
            if mask.any():
                oof_regime = oof_best[mask].copy()
                oof_regime = apply_savgol_per_well(oof_regime, cfg['window'], polyorder=2)
                oof_best.loc[mask, 'pred_tvt'] = oof_regime['pred_tvt'].values

    # Save best OOF (keep only relevant columns)
    oof_best_save = oof_best[['well_id', 'row_idx', 'id', 'TVT', 'last_known_TVT', 'pred_tvt']].copy()
    oof_best_path = EXP_DIR / 'oof_best.csv'
    oof_best_save.to_csv(oof_best_path, index=False)
    logger.info(f"Saved best OOF to {oof_best_path}")

    # Final CV check
    final_cv = compute_cv_rmse(oof_best['TVT'].values, oof_best['pred_tvt'].values)
    logger.info(f"Final OOF CV RMSE: {final_cv:.6f}")

    # ========================================================================
    # Result JSON
    # ========================================================================

    result = {
        'exp_id': EXP_ID,
        'method': 'postproc_validation',
        'baseline_cv_rmse': float(baseline),
        'baseline_method': 'mean_smoothing',
        'savgol_best': {
            'window': int(best_savgol['window']),
            'polyorder': int(best_savgol['polyorder']),
            'cv_rmse': float(best_savgol['pooled_cv']),
            'fold_cvs': [float(x) for x in best_savgol['fold_cvs']],
            'fold_std': float(best_savgol['fold_std']),
        },
        'selector_best': {
            'cv_rmse': float(best_selector['pooled_cv']),
            'fold_cvs': [float(x) for x in best_selector['fold_cvs']],
            'fold_std': float(best_selector['fold_std']),
            'best_per_regime': {
                regime: {
                    'window': int(cfg['window']),
                    'fold_mean': float(cfg['fold_mean']),
                }
                for regime, cfg in best_selector['best_per_regime'].items()
            }
        },
        'recommended_method': recommended,
        'improvement_vs_baseline': float(improvement),
        'final_cv_rmse': float(final_cv),
        'fold_consistency_savgol': savgol_consistent,
        'fold_consistency_selector': selector_consistent,
    }

    result_path = EXP_DIR / 'result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved result.json to {result_path}")

    logger.info("\n" + "="*70)
    logger.info("exp042 COMPLETE")
    logger.info("="*70)

    return result

if __name__ == '__main__':
    try:
        result = main()
        print("\n[SUCCESS]")
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        sys.exit(1)
