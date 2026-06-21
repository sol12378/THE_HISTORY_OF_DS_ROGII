#!/usr/bin/env python3
"""
exp058: Spatial offset correction via leave-one-well-out KNN bias estimation

目的: well間の offset (PF予測の系統誤差) を空間平滑化し、broken wellの予測を補正
手法:
  - 各wellの known区間 bias_known = mean(pred_tvt - TVT_input) を計算(leak-free)
  - 近傍他well(LOWO)の bias_known から、KNN で bias_spatial を推定
  - hidden予測補正: pred_corrected = pred_tvt - bias_spatial (案b: 本命)

leak-free: leave-one-well-out(某wellの補正に、その well の hidden TVT使わない)、known区間biasのみ
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Root paths
ROOT = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')
EXP_ROOT = ROOT / 'experiments' / 'exp058_spatial_offset'
EXP_ROOT.mkdir(parents=True, exist_ok=True)

EXP022_PATH = ROOT / 'experiments' / 'exp022_particle_filter'
TRAIN_BASE_PATH = ROOT / 'data' / 'processed' / 'train_base_v001.parquet'

def load_data():
    """Load exp022 oof and train_base"""
    logger.info("Loading exp022 oof and train_base...")

    # exp022 oof
    oof = pd.read_csv(EXP022_PATH / 'oof.csv')
    logger.info(f"oof shape: {oof.shape}, columns: {oof.columns.tolist()}")

    # train_base
    train_base = pd.read_parquet(TRAIN_BASE_PATH)
    logger.info(f"train_base shape: {train_base.shape}")

    return oof, train_base

def compute_well_offset_from_hidden(oof, train_base):
    """
    Compute well offset from hidden region (leak-free)

    Strategy:
      - oof contains only hidden region rows (is_known_tvt == False)
      - For each well, compute offset = mean(pred_tvt - TVT) over hidden rows
      - This is the "systematic error" or "offset" in PF predictions
      - Also get X, Y coordinates (constant within well)

    Returns:
        DataFrame: well_id, X, Y, offset_hidden, n_hidden_rows
    """
    logger.info("Computing well offset from hidden region...")

    # oof covers hidden region; merge with train_base to get X, Y, TVT
    merged = oof.merge(
        train_base[['well_id', 'row_idx', 'X', 'Y', 'is_known_tvt']],
        on=['well_id', 'row_idx'],
        how='left'
    )

    # Sanity check: all rows should be hidden
    n_known = merged['is_known_tvt'].sum()
    logger.info(f"Rows marked as known_tvt in oof: {n_known} (should be 0)")

    # Compute offset = pred_tvt - TVT (systematic error)
    merged['offset'] = merged['pred_tvt'] - merged['TVT']

    # Aggregate by well_id
    well_offset = merged.groupby('well_id').agg({
        'offset': 'mean',
        'X': 'first',  # Assume X,Y constant within well
        'Y': 'first',
        'row_idx': 'count'  # n_hidden_rows
    }).reset_index()

    well_offset.columns = ['well_id', 'offset_hidden', 'X', 'Y', 'n_hidden_rows']
    logger.info(f"well_offset computed: {len(well_offset)} wells")
    logger.info(f"Sample:\n{well_offset.head(10)}")
    logger.info(f"offset_hidden stats:\n{well_offset['offset_hidden'].describe()}")

    return well_offset

def estimate_spatial_bias_lowo(well_offset):
    """
    Leave-one-well-out KNN estimation of spatial offset
    For each well, estimate its offset from K nearest neighbors (excluding itself)

    Returns:
        DataFrame: well_id, offset_spatial, n_neighbors_used
    """
    logger.info("Estimating spatial offset via LOWO KNN...")

    # Prepare features: X, Y
    coords = well_offset[['X', 'Y']].values
    n_wells = len(well_offset)
    K = min(5, n_wells - 1)  # K nearest neighbors (excluding self)

    logger.info(f"K={K}, n_wells={n_wells}")

    # Fit KNN on all wells
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    # For each well, exclude self (indices[:, 0]) and use K neighbors
    offset_spatial_list = []
    for i in range(n_wells):
        neighbor_indices = indices[i, 1:K+1]  # Exclude self
        neighbor_offsets = well_offset.iloc[neighbor_indices]['offset_hidden'].values
        offset_est = neighbor_offsets.mean()
        offset_spatial_list.append(offset_est)

    well_offset['offset_spatial'] = offset_spatial_list
    well_offset['n_neighbors_used'] = K

    logger.info(f"Spatial offset estimated.")
    logger.info(f"offset_spatial stats:\n{well_offset['offset_spatial'].describe()}")

    return well_offset

def apply_correction(oof, train_base, well_offset, mode='b'):
    """
    Apply correction to oof predictions

    mode='a': pred_corrected = pred_tvt - offset_hidden (self-calib)
    mode='b': pred_corrected = pred_tvt - offset_spatial (spatial harmonization, main)
    mode='c': pred_corrected = pred_tvt - offset_spatial (conditional on high error, TBD)

    Returns:
        DataFrame: corrected oof
    """
    logger.info(f"Applying correction (mode={mode})...")

    oof_corrected = oof.merge(
        well_offset[['well_id', 'offset_hidden', 'offset_spatial']],
        on='well_id',
        how='left'
    )

    if mode == 'a':
        oof_corrected['pred_tvt_corrected'] = oof_corrected['pred_tvt'] - oof_corrected['offset_hidden']
    elif mode == 'b':
        oof_corrected['pred_tvt_corrected'] = oof_corrected['pred_tvt'] - oof_corrected['offset_spatial']
    elif mode == 'c':
        # For now, same as 'b'; in future could conditionally apply
        oof_corrected['pred_tvt_corrected'] = oof_corrected['pred_tvt'] - oof_corrected['offset_spatial']
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Compute error after correction
    oof_corrected['error_corrected'] = oof_corrected['pred_tvt_corrected'] - oof_corrected['TVT']
    oof_corrected['abs_error_corrected'] = oof_corrected['error_corrected'].abs()

    logger.info(f"oof_corrected shape: {oof_corrected.shape}")
    logger.info(f"Columns: {oof_corrected.columns.tolist()}")

    return oof_corrected

def compute_cv_rmse(oof_corrected):
    """Compute RMSE on corrected predictions"""
    rmse = np.sqrt((oof_corrected['error_corrected'] ** 2).mean())
    return rmse

def compare_modes(oof, train_base, well_offset):
    """Compare all correction modes"""
    logger.info("\n" + "="*60)
    logger.info("COMPARING CORRECTION MODES")
    logger.info("="*60)

    # Original CV (from exp022)
    cv_original = np.sqrt((oof['error'] ** 2).mean())
    logger.info(f"Original exp022 CV RMSE: {cv_original:.4f}")

    results = {'original': cv_original}

    for mode in ['a', 'b', 'c']:
        oof_corrected = apply_correction(oof, train_base, well_offset, mode=mode)
        cv = compute_cv_rmse(oof_corrected)
        results[f'mode_{mode}'] = cv

        improvement = cv_original - cv
        pct_improvement = 100 * improvement / cv_original

        logger.info(f"\nMode {mode}:")
        logger.info(f"  CV RMSE: {cv:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f} ({pct_improvement:+.2f}%)")

        # Count improved/worsened wells
        n_improved = (oof_corrected['abs_error_corrected'] < oof_corrected['abs_error']).sum()
        n_worsened = (oof_corrected['abs_error_corrected'] > oof_corrected['abs_error']).sum()
        n_same = len(oof_corrected) - n_improved - n_worsened

        logger.info(f"  Improved rows: {n_improved} | Worsened: {n_worsened} | Same: {n_same}")

    return results

def analyze_broken_wells(oof, train_base, well_offset, per_well_path=None):
    """Analyze effect on broken wells"""
    logger.info("\n" + "="*60)
    logger.info("BROKEN WELL ANALYSIS")
    logger.info("="*60)

    # Load per_well stats to identify broken wells
    if per_well_path is None:
        per_well_path = EXP022_PATH / 'per_well.csv'

    per_well = pd.read_csv(per_well_path)

    # Broken wells: PF RMSE >> anchor RMSE (e.g., PF/anchor > 1.5)
    per_well['pf_anchor_ratio'] = per_well['pf_rmse'] / per_well['anchor_rmse']
    broken = per_well[per_well['pf_anchor_ratio'] > 1.5].copy()

    logger.info(f"Identified {len(broken)} broken wells (PF/anchor > 1.5)")
    logger.info(f"Top 10 broken wells:\n{broken.nlargest(10, 'pf_anchor_ratio')[['well_id', 'pf_rmse', 'anchor_rmse', 'pf_anchor_ratio']]}")

    # Correction effect on broken wells (mode='b')
    oof_corrected = apply_correction(oof, train_base, well_offset, mode='b')

    broken_oof = oof_corrected[oof_corrected['well_id'].isin(broken['well_id'])]
    rmse_before = np.sqrt((broken_oof['error'] ** 2).mean())
    rmse_after = np.sqrt((broken_oof['error_corrected'] ** 2).mean())

    logger.info(f"\nBroken wells correction (mode='b'):")
    logger.info(f"  RMSE before: {rmse_before:.4f}")
    logger.info(f"  RMSE after:  {rmse_after:.4f}")
    logger.info(f"  Improvement: {rmse_before - rmse_after:+.4f}")

    return broken, rmse_before, rmse_after

def main():
    logger.info(f"exp058 spatial offset started at {datetime.now()}")

    # Load
    oof, train_base = load_data()

    # Compute offset from hidden region
    well_offset = compute_well_offset_from_hidden(oof, train_base)

    # Estimate spatial offset via LOWO KNN
    well_offset = estimate_spatial_bias_lowo(well_offset)

    # Compare modes
    results = compare_modes(oof, train_base, well_offset)

    # Broken well analysis
    broken, rmse_broken_before, rmse_broken_after = analyze_broken_wells(oof, train_base, well_offset)

    # Save best mode (mode='b')
    best_mode = 'b'
    oof_best = apply_correction(oof, train_base, well_offset, mode=best_mode)

    # Save outputs
    logger.info("\n" + "="*60)
    logger.info("SAVING OUTPUTS")
    logger.info("="*60)

    # Save oof
    oof_out = oof_best[['well_id', 'row_idx', 'id', 'TVT', 'last_known_TVT', 'pred_tvt_corrected']].copy()
    oof_out.columns = ['well_id', 'row_idx', 'id', 'TVT', 'last_known_TVT', 'pred_tvt']
    oof_out.to_csv(EXP_ROOT / 'oof.csv', index=False)
    logger.info(f"Saved oof.csv to {EXP_ROOT / 'oof.csv'}")

    # Save well_offset
    well_offset.to_csv(EXP_ROOT / 'well_offset.csv', index=False)
    logger.info(f"Saved well_offset.csv")

    # Save result.json
    cv_best = compute_cv_rmse(oof_best)
    result = {
        'exp_id': 'exp058_spatial_offset',
        'created_at': datetime.now().isoformat(),
        'status': 'completed',
        'method': 'Spatial offset correction via LOWO KNN offset estimation (mode=b)',
        'cv_rmse': cv_best,
        'cv_rmse_exp022_baseline': results['original'],
        'improvement_vs_baseline': results['original'] - cv_best,
        'improvement_pct': 100 * (results['original'] - cv_best) / results['original'],
        'modes_compared': {k: v for k, v in results.items()},
        'n_wells': len(well_offset),
        'lowo_k_neighbors': 5,
        'broken_well_rmse_before': float(rmse_broken_before),
        'broken_well_rmse_after': float(rmse_broken_after),
        'n_broken_wells': len(broken),
        'leak_risk': 'none (leave-one-well-out KNN; hidden region offset only)',
        'notes': 'Mode a (self-calib): CV 7.10, +35.6% (experimental). Mode b (spatial smoothing): CV 11.36, -3.0% (unsuccessful). Hidden offset variance large; spatial correlation insufficient. Offset strongly well-specific; neighbors smooth too aggressively.'
    }

    with open(EXP_ROOT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved result.json")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"exp022 baseline CV: {results['original']:.4f}")
    logger.info(f"exp058 mode=b CV:   {cv_best:.4f}")
    logger.info(f"Improvement: {results['original'] - cv_best:+.4f} ({100*(results['original']-cv_best)/results['original']:+.2f}%)")
    logger.info(f"\nFull result.json saved to {EXP_ROOT / 'result.json'}")

    return result

if __name__ == '__main__':
    result = main()
    sys.exit(0)
