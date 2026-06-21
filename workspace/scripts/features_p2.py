"""
P2 Features: 3D Tortuosity + Spatial KNN Formation Surface
============================================================

Leak-free features based on known TVT intervals:
1. 3D tortuosity metrics (per-well from known section)
2. Spatial KNN formation surface predictor (S(X,Y) ≈ TVT+Z)

Usage:
  python features_p2.py \
    --train_path data/processed/train_base_v001.parquet \
    --test_path data/processed/test_base_v001.parquet \
    --output_dir experiments/features_p2 \
    --n_neighbors 8 \
    --seed 42
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_3d_tortuosity(group_df: pd.DataFrame) -> dict:
    """
    Compute tortuosity metrics for a well's known TVT section.

    Params:
        group_df: rows for one well, sorted by row_idx

    Returns:
        dict with keys: tort_3d, tort_xy, tort_vert, dls_mean, dls_last30,
                        inclination_last, azimuth_change
        All values are scalars (broadcast to all rows of the well).
    """
    known_mask = group_df['is_known_tvt'].values

    if known_mask.sum() < 3:
        # Insufficient data: fallback to 1.0 for tortuosity, 0 for dls
        return {
            'tort_3d': 1.0,
            'tort_xy': 1.0,
            'tort_vert': 1.0,
            'dls_mean': 0.0,
            'dls_last30': 0.0,
            'inclination_last': 0.0,
            'azimuth_change': 0.0,
        }

    known_df = group_df[known_mask].copy()
    X = known_df['X'].values
    Y = known_df['Y'].values
    Z = known_df['Z'].values
    MD = known_df['MD'].values

    # === 3D Tortuosity ===
    # Path length: sum of consecutive segment lengths
    dX = np.diff(X)
    dY = np.diff(Y)
    dZ = np.diff(Z)
    segment_lengths = np.sqrt(dX**2 + dY**2 + dZ**2)
    total_path_length = segment_lengths.sum()

    # Straight-line distance from start to end
    straight_dist = np.sqrt((X[-1] - X[0])**2 + (Y[-1] - Y[0])**2 + (Z[-1] - Z[0])**2)

    if straight_dist > 1e-6:
        tort_3d = total_path_length / straight_dist
    else:
        tort_3d = 1.0

    # === XY (Horizontal) Tortuosity ===
    dX_h = np.diff(X)
    dY_h = np.diff(Y)
    segment_lengths_h = np.sqrt(dX_h**2 + dY_h**2)
    total_path_length_h = segment_lengths_h.sum()
    straight_dist_h = np.sqrt((X[-1] - X[0])**2 + (Y[-1] - Y[0])**2)

    if straight_dist_h > 1e-6:
        tort_xy = total_path_length_h / straight_dist_h
    else:
        tort_xy = 1.0

    # === Vertical Tortuosity (MD vs TVD) ===
    TVD_range = Z[-1] - Z[0]
    MD_range = MD[-1] - MD[0]

    if MD_range > 1e-6:
        tort_vert = MD_range / abs(TVD_range) if TVD_range != 0 else 1.0
    else:
        tort_vert = 1.0

    # === Dogleg Severity (DLS) ===
    # Direction vectors for each segment
    if len(segment_lengths) > 1:
        # Normalize direction vectors
        dirs = np.column_stack([dX, dY, dZ]) / segment_lengths[:, None]

        # Angle between consecutive direction vectors
        dot_products = np.sum(dirs[:-1] * dirs[1:], axis=1)
        # Clamp to [-1, 1] to avoid numerical issues with arccos
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(dot_products)

        # DLS: angle change per unit measured depth
        # For each angle measurement i, use MD span from i to i+2 (the interval between dir[i] and dir[i+1])
        dMD = np.diff(MD)  # MD differences between consecutive points
        # Each angle is between segment i and segment i+1
        # So use the MD span covering both segments: MD[i+2] - MD[i]
        dMD_span = dMD[1:] + dMD[:-1]  # MD span for each angle
        dLS_raw = angles / (dMD_span + 1e-8)  # rad/ft
        dls_mean = np.nanmean(dLS_raw)
        dls_mean = max(0.0, dls_mean)
    else:
        dls_mean = 0.0

    # === DLS over last 30 segments ===
    if len(segment_lengths) >= 30:
        dX_last = dX[-30:]
        dY_last = dY[-30:]
        dZ_last = dZ[-30:]
        seg_last = segment_lengths[-30:]
        MD_last = MD[-30:]

        if len(seg_last) > 1:
            dirs_last = np.column_stack([dX_last, dY_last, dZ_last]) / seg_last[:, None]
            dot_products_last = np.sum(dirs_last[:-1] * dirs_last[1:], axis=1)
            dot_products_last = np.clip(dot_products_last, -1, 1)
            angles_last = np.arccos(dot_products_last)

            # MD span for each angle
            dMD_last_array = np.diff(MD_last)
            dMD_span_last = dMD_last_array[1:] + dMD_last_array[:-1]
            if len(dMD_span_last) > 0:
                dls_raw_last = angles_last[:len(dMD_span_last)] / (dMD_span_last + 1e-8)
                dls_last30 = np.nanmean(dls_raw_last)
                dls_last30 = max(0.0, dls_last30)
            else:
                dls_last30 = dls_mean
        else:
            dls_last30 = dls_mean
    else:
        dls_last30 = dls_mean

    # === Inclination of Last Known Row ===
    # Inclination = atan2(horizontal displacement, vertical displacement)
    if len(X) > 1:
        horiz_disp = np.sqrt((X[-1] - X[-2])**2 + (Y[-1] - Y[-2])**2)
        vert_disp = abs(Z[-1] - Z[-2])
        inclination_last = np.arctan2(horiz_disp, vert_disp)
    else:
        inclination_last = 0.0

    # === Azimuth Change ===
    # Change in azimuth (direction in XY plane) from start to end
    if len(X) > 1:
        az_start = np.arctan2(Y[1] - Y[0], X[1] - X[0])
        az_end = np.arctan2(Y[-1] - Y[-2], X[-1] - X[-2])
        azimuth_change = abs(az_end - az_start)
        # Normalize to [0, pi]
        azimuth_change = min(azimuth_change, 2 * np.pi - azimuth_change)
    else:
        azimuth_change = 0.0

    return {
        'tort_3d': float(tort_3d),
        'tort_xy': float(tort_xy),
        'tort_vert': float(tort_vert),
        'dls_mean': float(dls_mean),
        'dls_last30': float(dls_last30),
        'inclination_last': float(inclination_last),
        'azimuth_change': float(azimuth_change),
    }


def build_knn_surface(
    train_df: pd.DataFrame,
    n_neighbors: int = 8,
    subsample_limit: Optional[int] = None,
) -> Tuple[cKDTree, np.ndarray]:
    """
    Build spatial KNN tree from train known TVT points.

    Returns:
        (tree, surface_values) where:
          - tree: cKDTree for (X, Y) coordinates
          - surface_values: array of (TVT_input + Z) for each point
    """
    known_mask = train_df['is_known_tvt'].values
    known_df = train_df[known_mask].copy()

    logger.info(f"Building KNN surface from {len(known_df)} known TVT points")

    # Surface value = TVT_input + Z (predicted surface at (X,Y))
    surface_values = (known_df['TVT_input'].values + known_df['Z'].values).astype(np.float32)

    # Coordinates for tree
    coords = known_df[['X', 'Y']].values.astype(np.float32)

    # Subsample if too large (memory constraint)
    if subsample_limit and len(coords) > subsample_limit:
        idx = np.random.choice(len(coords), subsample_limit, replace=False)
        coords = coords[idx]
        surface_values = surface_values[idx]
        logger.warning(f"Subsampled to {subsample_limit} points")

    tree = cKDTree(coords)
    logger.info(f"KNN tree built: {len(surface_values)} points")

    return tree, surface_values


def query_knn_surface(
    query_coords: np.ndarray,
    tree: cKDTree,
    surface_values: np.ndarray,
    n_neighbors: int = 8,
) -> np.ndarray:
    """
    Query KNN surface for each point in query_coords.

    Params:
        query_coords: (N, 2) array of (X, Y)
        tree: cKDTree from build_knn_surface
        surface_values: array of surface values
        n_neighbors: k for KNN

    Returns:
        (N,) array of distance-weighted average surface values
    """
    # Query with distance-weighted average
    distances, indices = tree.query(query_coords, k=n_neighbors)

    # Handle edge case of single point query
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Distance-weighted average (inverse distance weighting)
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    weights = 1.0 / (distances + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predictions = np.sum(surface_values[indices] * weights, axis=1)

    return predictions


def build_p2_features(
    df: pd.DataFrame,
    train_tree: Optional[cKDTree] = None,
    train_surface_values: Optional[np.ndarray] = None,
    n_neighbors: int = 8,
    split_name: str = 'train',
) -> pd.DataFrame:
    """
    Compute P2 features for all rows.

    If train_tree and train_surface_values are provided, uses them for KNN.
    Otherwise, builds from known points in df.

    Params:
        df: full dataframe (train or test)
        train_tree: pre-built KDTree (None → build from df's known points)
        train_surface_values: pre-computed surface values
        n_neighbors: k for KNN
        split_name: 'train' or 'test'

    Returns:
        DataFrame with P2 features added
    """
    logger.info(f"Building P2 features for {split_name}: {len(df)} rows")

    # === Tortuosity: per-well, broadcast to all rows ===
    logger.info("Computing tortuosity features...")
    tort_features = {}
    for well_id, group in df.groupby('well_id', sort=False):
        tort_dict = compute_3d_tortuosity(group)
        tort_features[well_id] = tort_dict

    # Broadcast to all rows
    p2_df = df.copy()
    for key in ['tort_3d', 'tort_xy', 'tort_vert', 'dls_mean', 'dls_last30', 'inclination_last', 'azimuth_change']:
        p2_df[key] = p2_df['well_id'].map(lambda w: tort_features[w][key])

    # === KNN Surface ===
    logger.info("Computing KNN formation surface features...")

    # For test split: always use train_tree
    # For train split: if tree provided, use it; else build from own known points
    if train_tree is not None:
        tree = train_tree
        surface_values = train_surface_values
    else:
        tree, surface_values = build_knn_surface(p2_df, n_neighbors=n_neighbors)

    # Query all rows
    coords = p2_df[['X', 'Y']].values.astype(np.float32)
    knn_pred = query_knn_surface(coords, tree, surface_values, n_neighbors=n_neighbors)

    p2_df['knn_surface'] = knn_pred
    p2_df['knn_surface_minus_Z'] = knn_pred - p2_df['Z'].values

    logger.info(f"P2 features complete: {len(p2_df)} rows, {len(p2_df.columns)} columns")

    return p2_df


def main():
    parser = argparse.ArgumentParser(description='Compute P2 leak-free features')
    parser.add_argument('--train_path', type=str, required=True, help='Train parquet path')
    parser.add_argument('--test_path', type=str, required=True, help='Test parquet path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--n_neighbors', type=int, default=8, help='K for KNN')
    parser.add_argument('--subsample_limit', type=int, default=100000, help='Max KNN tree points')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading train from {args.train_path}")
    train_df = pd.read_parquet(args.train_path)

    logger.info(f"Loading test from {args.test_path}")
    test_df = pd.read_parquet(args.test_path)

    # Build KNN tree from train known points
    logger.info("Building KNN surface from train known TVT points...")
    train_tree, train_surface_values = build_knn_surface(
        train_df,
        n_neighbors=args.n_neighbors,
        subsample_limit=args.subsample_limit
    )

    # Compute features
    logger.info("Computing P2 features for train...")
    train_p2 = build_p2_features(
        train_df,
        train_tree=train_tree,
        train_surface_values=train_surface_values,
        n_neighbors=args.n_neighbors,
        split_name='train'
    )

    logger.info("Computing P2 features for test...")
    test_p2 = build_p2_features(
        test_df,
        train_tree=train_tree,
        train_surface_values=train_surface_values,
        n_neighbors=args.n_neighbors,
        split_name='test'
    )

    # Select P2 feature columns only
    p2_cols = ['well_id', 'row_idx', 'tort_3d', 'tort_xy', 'tort_vert',
               'dls_mean', 'dls_last30', 'inclination_last', 'azimuth_change',
               'knn_surface', 'knn_surface_minus_Z']

    train_out = train_p2[p2_cols]
    test_out = test_p2[p2_cols]

    # Save
    train_out_path = output_dir / 'train_p2.parquet'
    test_out_path = output_dir / 'test_p2.parquet'

    logger.info(f"Saving train features to {train_out_path}")
    train_out.to_parquet(train_out_path, index=False)

    logger.info(f"Saving test features to {test_out_path}")
    test_out.to_parquet(test_out_path, index=False)

    # Validation & reporting
    logger.info("Validation checks...")

    # Check for nulls in target rows
    train_target_rows = train_df[train_df['is_target']].index
    null_count = train_p2.loc[train_target_rows, p2_cols[2:]].isnull().sum().sum()
    logger.info(f"Null count in target rows (P2 cols): {null_count}")

    # Check tort_3d range
    tort_3d_min = train_p2['tort_3d'].min()
    tort_3d_max = train_p2['tort_3d'].max()
    logger.info(f"tort_3d range: [{tort_3d_min:.4f}, {tort_3d_max:.4f}]")

    # Check dls_mean range
    dls_mean_min = train_p2['dls_mean'].min()
    dls_mean_max = train_p2['dls_mean'].max()
    logger.info(f"dls_mean range: [{dls_mean_min:.6f}, {dls_mean_max:.6f}]")

    # Correlation: knn_surface_minus_Z vs TVT (target rows only)
    target_rows = train_df['is_target'].values
    corr = np.corrcoef(
        train_p2.loc[target_rows, 'knn_surface_minus_Z'].values,
        train_df.loc[target_rows, 'TVT'].values
    )[0, 1]
    logger.info(f"knn_surface_minus_Z vs TVT correlation (target rows): {corr:.4f}")

    # Result JSON
    result = {
        'p2_features': p2_cols[2:],  # Exclude well_id, row_idx
        'tort_3d_range': [float(tort_3d_min), float(tort_3d_max)],
        'dls_mean_range': [float(dls_mean_min), float(dls_mean_max)],
        'knn_surface_minus_Z_vs_TVT_corr': float(corr),
        'train_shape': train_out.shape,
        'test_shape': test_out.shape,
        'null_count_in_target_rows': int(null_count),
        'knn_surface_points': len(train_surface_values),
    }

    result_path = output_dir / 'result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Result saved to {result_path}")

    # Summary
    logger.info("="*60)
    logger.info("P2 FEATURES SUMMARY")
    logger.info("="*60)
    logger.info(f"Features: {result['p2_features']}")
    logger.info(f"Train output: {train_out_path}")
    logger.info(f"Test output: {test_out_path}")
    logger.info(f"tort_3d range: {result['tort_3d_range']}")
    logger.info(f"dls_mean range: {result['dls_mean_range']}")
    logger.info(f"knn_surface_minus_Z vs TVT corr: {result['knn_surface_minus_Z_vs_TVT_corr']:.4f}")
    logger.info(f"KNN tree points: {result['knn_surface_points']}")
    logger.info(f"Null count (target rows): {result['null_count_in_target_rows']}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
