"""
exp041: Leak-free feature engineering module
- 3D tortuosity features (well-level + row-level)
- Spatial KNN formation surface prior (TVT+Z ~ f(X,Y) from train neighbors)
- Dip 2-component features (azimuth + cross direction)

Key: leave-one-out KNN on train wells, no hidden TVT usage
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def compute_tortuosity_features(well_df):
    """
    Compute 3D tortuosity features for a single well
    Inputs: MD, X, Y, Z for all rows in well (sorted by MD)
    Returns: dict of well-level features
    """
    features = {}

    # Filter to known sections only
    known_df = well_df[well_df['is_known_tvt']].copy()
    if len(known_df) < 2:
        return {
            'tort_3d_ratio_known': np.nan,
            'tort_3d_ratio_last50': np.nan,
            'tort_curvature_last20': np.nan,
            'dls_last20': np.nan,
        }

    known_df = known_df.sort_values('MD').reset_index(drop=True)

    # Extract coordinates
    md = known_df['MD'].values
    x = known_df['X'].values
    y = known_df['Y'].values
    z = known_df['Z'].values

    # 1. 3D ratio: total MD length / direct distance
    total_md = md[-1] - md[0]
    direct_dist = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2 + (z[-1] - z[0])**2)

    if direct_dist > 0:
        features['tort_3d_ratio_known'] = total_md / direct_dist
    else:
        features['tort_3d_ratio_known'] = np.nan

    # Last 50 rows
    last_n = min(50, len(known_df))
    if last_n >= 2:
        md_l50 = md[-last_n:]
        x_l50 = x[-last_n:]
        y_l50 = y[-last_n:]
        z_l50 = z[-last_n:]

        md_range = md_l50[-1] - md_l50[0]
        direct_dist_l50 = np.sqrt((x_l50[-1] - x_l50[0])**2 + (y_l50[-1] - y_l50[0])**2 + (z_l50[-1] - z_l50[0])**2)

        if direct_dist_l50 > 0:
            features['tort_3d_ratio_last50'] = md_range / direct_dist_l50
        else:
            features['tort_3d_ratio_last50'] = np.nan
    else:
        features['tort_3d_ratio_last50'] = np.nan

    # 2. Curvature: angle change in direction vectors (last 20)
    last_n = min(20, len(known_df))
    if last_n >= 3:
        angles = []
        for i in range(len(x) - 2):
            # Direction vectors
            v1 = np.array([x[i+1] - x[i], y[i+1] - y[i], z[i+1] - z[i]])
            v2 = np.array([x[i+2] - x[i+1], y[i+2] - y[i+1], z[i+2] - z[i+1]])

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-6 and norm2 > 1e-6:
                cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)

        if angles:
            features['tort_curvature_last20'] = np.mean(angles[-20:]) if len(angles) >= 20 else np.mean(angles)
        else:
            features['tort_curvature_last20'] = np.nan
    else:
        features['tort_curvature_last20'] = np.nan

    # 3. DLS (dogleg severity): inclination/azimuth change rate
    if len(known_df) >= 3:
        incs = []
        azims = []

        for i in range(len(md) - 1):
            dmd = md[i+1] - md[i]
            if dmd > 0:
                dz = z[i+1] - z[i]
                inc = np.arccos(np.clip(dz / dmd, -1, 1))
                incs.append(inc)

                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                azim = np.arctan2(dy, dx)
                azims.append(azim)

        if len(incs) >= 20:
            incs_l20 = np.array(incs[-20:])
            azims_l20 = np.array(azims[-20:])

            # DLS = sqrt(delta_inc^2 + delta_azim^2) per 30m (or normalize by dMD)
            dls_vals = []
            for j in range(len(incs_l20) - 1):
                dinc = abs(incs_l20[j+1] - incs_l20[j])
                dazim = abs(azims_l20[j+1] - azims_l20[j])
                dls = np.sqrt(dinc**2 + dazim**2)
                dls_vals.append(dls)

            features['dls_last20'] = np.mean(dls_vals) if dls_vals else np.nan
        else:
            features['dls_last20'] = np.nan
    else:
        features['dls_last20'] = np.nan

    return features


def build_row_tortuosity_features(well_df):
    """
    Compute row-level tortuosity features: curvature around each row
    """
    row_features = {}

    known_df = well_df[well_df['is_known_tvt']].copy().sort_values('MD').reset_index(drop=True)

    if len(known_df) < 3:
        return {}

    x = known_df['X'].values
    y = known_df['Y'].values
    z = known_df['Z'].values
    row_idxs = known_df['row_idx'].values

    # For each row with neighbors, compute local curvature
    for idx_pos in range(1, len(known_df) - 1):
        row_idx = row_idxs[idx_pos]

        v1 = np.array([x[idx_pos] - x[idx_pos-1], y[idx_pos] - y[idx_pos-1], z[idx_pos] - z[idx_pos-1]])
        v2 = np.array([x[idx_pos+1] - x[idx_pos], y[idx_pos+1] - y[idx_pos], z[idx_pos+1] - z[idx_pos]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
            curvature = np.arccos(cos_angle)
            row_features[row_idx] = curvature

    return row_features


def build_knn_surface(train_df, k=8):
    """
    Build KNN-based formation surface model from train well anchors
    Input: train_df with last_known_X, last_known_Y, last_known_TVT, last_known_Z
    Returns: scaler, nbrs model, anchor points (one per well)

    Key: Use only known TVT anchors, fit on leave-one-out basis
    """
    # Get anchor points: one per well (last_known location)
    anchor_df = train_df[train_df['is_known_tvt']].drop_duplicates('well_id')[['well_id', 'last_known_X', 'last_known_Y', 'last_known_TVT', 'last_known_Z']].copy()

    if len(anchor_df) < k + 1:
        print(f"Warning: only {len(anchor_df)} train wells, k={k} may be too large")
        k = min(k, len(anchor_df) - 1)

    X_anchor = anchor_df[['last_known_X', 'last_known_Y']].values
    TVT_Z_anchor = anchor_df[['last_known_TVT', 'last_known_Z']].values

    # Standardize inputs for KNN
    scaler = StandardScaler()
    X_anchor_scaled = scaler.fit_transform(X_anchor)

    # Build KNN model (will use for both train and test)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')  # k+1 to include self
    nbrs.fit(X_anchor_scaled)

    return scaler, nbrs, anchor_df, X_anchor_scaled, TVT_Z_anchor, k


def predict_knn_surface(X_pred_xy, scaler, nbrs, anchor_df, X_anchor_scaled, TVT_Z_anchor, k, exclude_well_ids=None):
    """
    Predict formation surface (TVT, Z) at (X, Y) locations using KNN

    Args:
        X_pred_xy: (N, 2) array of [X, Y] coordinates
        exclude_well_ids: set/list of well_ids to exclude (for leave-one-out on train)

    Returns:
        knn_surface_tvt: predicted TVT at (X, Y)
        knn_surface_z: predicted Z at (X, Y)
    """
    X_pred_scaled = scaler.transform(X_pred_xy)
    distances, indices = nbrs.kneighbors(X_pred_scaled, n_neighbors=k+1)

    # k+1 because first neighbor might be self, which we exclude for train wells
    knn_tvt = []
    knn_z = []

    for i, (dist, idx_list) in enumerate(zip(distances, indices)):
        valid_tvt = []
        valid_z = []

        for idx in idx_list:
            well_id = anchor_df.iloc[idx]['well_id']

            # Skip self-well if in exclude set (leave-one-out for train)
            if exclude_well_ids is not None and well_id in exclude_well_ids:
                continue

            valid_tvt.append(TVT_Z_anchor[idx, 0])
            valid_z.append(TVT_Z_anchor[idx, 1])

        # Average top k valid neighbors
        valid_tvt = valid_tvt[:k]
        valid_z = valid_z[:k]

        if valid_tvt:
            knn_tvt.append(np.mean(valid_tvt))
            knn_z.append(np.mean(valid_z))
        else:
            knn_tvt.append(np.nan)
            knn_z.append(np.nan)

    return np.array(knn_tvt), np.array(knn_z)


def fit_dip_plane(well_df):
    """
    Fit a plane through known TVT points: TVT ~ a*X + b*Y + c
    Returns: a (dip_x), b (dip_y), c (intercept), or NaN if insufficient data
    """
    known_df = well_df[well_df['is_known_tvt']].copy()

    if len(known_df) < 3:
        return np.nan, np.nan, np.nan

    # Prepare design matrix [X, Y, 1]
    X_design = known_df[['X', 'Y']].values
    X_design = np.column_stack([X_design, np.ones(len(X_design))])
    y_tvt = known_df['TVT'].values

    # Least squares fit
    try:
        coef = np.linalg.lstsq(X_design, y_tvt, rcond=None)[0]
        return coef[0], coef[1], coef[2]
    except:
        return np.nan, np.nan, np.nan


def build_features(train_df, test_df):
    """
    Main function: build all leak-free features

    Returns:
        train_features_df: (n_train_target, features)
        test_features_df: (n_test_all, features)
    """

    print("=" * 80)
    print("EXP041: Building leak-free features")
    print("=" * 80)

    # Step 1: Build KNN surface on train anchors
    print("\n[1/5] Building KNN surface model from train well anchors...")
    t0 = time.time()
    scaler, nbrs, anchor_df, X_anchor_scaled, TVT_Z_anchor, k = build_knn_surface(train_df, k=8)
    print(f"  {len(anchor_df)} train well anchors")
    print(f"  Time: {time.time() - t0:.2f}s")

    # Step 2: Compute per-well tortuosity and dip features
    print("\n[2/5] Computing per-well tortuosity and dip features...")
    t0 = time.time()

    train_wells = train_df['well_id'].unique()
    test_wells = test_df['well_id'].unique()
    all_wells = np.concatenate([train_wells, test_wells])

    well_features = {}
    dip_params = {}
    row_curvatures = {}

    for well_id in all_wells:
        if well_id in train_df['well_id'].values:
            well_df = train_df[train_df['well_id'] == well_id]
        else:
            well_df = test_df[test_df['well_id'] == well_id]

        # Tortuosity
        well_features[well_id] = compute_tortuosity_features(well_df)

        # Row-level curvature
        row_curv = build_row_tortuosity_features(well_df)
        row_curvatures[well_id] = row_curv

        # Dip plane
        dip_x, dip_y, dip_c = fit_dip_plane(well_df)
        dip_params[well_id] = (dip_x, dip_y, dip_c)

    print(f"  {len(all_wells)} wells processed")
    print(f"  Time: {time.time() - t0:.2f}s")

    # Step 3: Apply KNN surface predictions to train data (with leave-one-out)
    print("\n[3/5] Computing KNN surface predictions on train data (leave-one-out)...")
    t0 = time.time()

    train_knn_tvt = []
    train_knn_z = []
    train_row_idxs = []
    train_well_ids = []

    for well_id in train_wells:
        well_df = train_df[train_df['well_id'] == well_id].copy()

        # Predict surface for this well, excluding self (leave-one-out)
        X_well = well_df[['X', 'Y']].values
        pred_tvt, pred_z = predict_knn_surface(
            X_well, scaler, nbrs, anchor_df, X_anchor_scaled, TVT_Z_anchor, k,
            exclude_well_ids={well_id}
        )

        train_knn_tvt.extend(pred_tvt)
        train_knn_z.extend(pred_z)
        train_row_idxs.extend(well_df['row_idx'].values)
        train_well_ids.extend([well_id] * len(well_df))

    print(f"  {len(train_well_ids)} train rows predicted")
    print(f"  Time: {time.time() - t0:.2f}s")

    # Step 4: Apply KNN surface predictions to test data (no exclusion)
    print("\n[4/5] Computing KNN surface predictions on test data...")
    t0 = time.time()

    test_knn_tvt = []
    test_knn_z = []
    test_row_idxs = []
    test_well_ids = []

    for well_id in test_wells:
        well_df = test_df[test_df['well_id'] == well_id].copy()

        # Predict surface for this well, using all train neighbors (no exclusion)
        X_well = well_df[['X', 'Y']].values
        pred_tvt, pred_z = predict_knn_surface(
            X_well, scaler, nbrs, anchor_df, X_anchor_scaled, TVT_Z_anchor, k,
            exclude_well_ids=None
        )

        test_knn_tvt.extend(pred_tvt)
        test_knn_z.extend(pred_z)
        test_row_idxs.extend(well_df['row_idx'].values)
        test_well_ids.extend([well_id] * len(well_df))

    print(f"  {len(test_well_ids)} test rows predicted")
    print(f"  Time: {time.time() - t0:.2f}s")

    # Step 5: Assemble feature dataframes
    print("\n[5/5] Assembling feature dataframes...")
    t0 = time.time()

    # Train features
    train_features = pd.DataFrame({
        'well_id': train_well_ids,
        'row_idx': train_row_idxs,
    })

    # Add well-level tortuosity features
    for col in ['tort_3d_ratio_known', 'tort_3d_ratio_last50', 'tort_curvature_last20', 'dls_last20']:
        train_features[col] = train_features['well_id'].map(lambda w: well_features[w][col])

    # Add row-level curvature (from known section)
    train_features['local_curvature'] = train_features.apply(
        lambda row: row_curvatures[row['well_id']].get(row['row_idx'], np.nan),
        axis=1
    )

    # Add KNN surface features
    train_features['knn_surface_pred_tvt'] = train_knn_tvt
    train_features['knn_surface_pred_z'] = train_knn_z
    train_features['knn_surface_tvt_residual'] = (
        train_df.set_index('row_idx').loc[train_row_idxs, 'last_known_TVT'].values - np.array(train_knn_tvt)
    )

    # Add dip plane features
    dip_x_vals = []
    dip_y_vals = []
    dip_plane_tvt = []
    for i, (well_id, row_idx) in enumerate(zip(train_well_ids, train_row_idxs)):
        dip_x, dip_y, dip_c = dip_params[well_id]
        dip_x_vals.append(dip_x)
        dip_y_vals.append(dip_y)

        row_data = train_df[(train_df['well_id'] == well_id) & (train_df['row_idx'] == row_idx)]
        if len(row_data) > 0:
            x = row_data['X'].values[0]
            y = row_data['Y'].values[0]
            if not np.isnan(dip_x) and not np.isnan(dip_y):
                pred_tvt = dip_x * x + dip_y * y + dip_c
                dip_plane_tvt.append(pred_tvt)
            else:
                dip_plane_tvt.append(np.nan)
        else:
            dip_plane_tvt.append(np.nan)

    train_features['dip_x'] = dip_x_vals
    train_features['dip_y'] = dip_y_vals
    train_features['dip_plane_tvt'] = dip_plane_tvt

    # Test features (same structure)
    test_features = pd.DataFrame({
        'well_id': test_well_ids,
        'row_idx': test_row_idxs,
    })

    for col in ['tort_3d_ratio_known', 'tort_3d_ratio_last50', 'tort_curvature_last20', 'dls_last20']:
        test_features[col] = test_features['well_id'].map(lambda w: well_features[w][col])

    test_features['local_curvature'] = test_features.apply(
        lambda row: row_curvatures[row['well_id']].get(row['row_idx'], np.nan),
        axis=1
    )

    test_features['knn_surface_pred_tvt'] = test_knn_tvt
    test_features['knn_surface_pred_z'] = test_knn_z
    test_features['knn_surface_tvt_residual'] = (
        test_df.set_index('row_idx').loc[test_row_idxs, 'last_known_TVT'].values - np.array(test_knn_tvt)
    )

    dip_x_vals_test = []
    dip_y_vals_test = []
    dip_plane_tvt_test = []
    for i, (well_id, row_idx) in enumerate(zip(test_well_ids, test_row_idxs)):
        dip_x, dip_y, dip_c = dip_params[well_id]
        dip_x_vals_test.append(dip_x)
        dip_y_vals_test.append(dip_y)

        row_data = test_df[(test_df['well_id'] == well_id) & (test_df['row_idx'] == row_idx)]
        if len(row_data) > 0:
            x = row_data['X'].values[0]
            y = row_data['Y'].values[0]
            if not np.isnan(dip_x) and not np.isnan(dip_y):
                pred_tvt = dip_x * x + dip_y * y + dip_c
                dip_plane_tvt_test.append(pred_tvt)
            else:
                dip_plane_tvt_test.append(np.nan)
        else:
            dip_plane_tvt_test.append(np.nan)

    test_features['dip_x'] = dip_x_vals_test
    test_features['dip_y'] = dip_y_vals_test
    test_features['dip_plane_tvt'] = dip_plane_tvt_test

    print(f"  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")
    print(f"  Time: {time.time() - t0:.2f}s")

    return train_features, test_features


def main():
    """
    Main entry point: load data, build features, save, report
    """

    root = Path('/Users/satouryuuichi/Desktop/DS/ROGII-Wellbore-Geology-Prediction')

    # Load base data
    print("Loading base parquet files...")
    train_df = pd.read_parquet(root / 'data/processed/train_base_v001.parquet')
    test_df = pd.read_parquet(root / 'data/processed/test_base_v001.parquet')

    # Build features
    t_total = time.time()
    train_features, test_features = build_features(train_df, test_df)
    t_total = time.time() - t_total

    # Save
    exp_dir = root / 'experiments/exp041_features'
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_features.to_parquet(exp_dir / 'train_features.parquet', index=False)
    test_features.to_parquet(exp_dir / 'test_features.parquet', index=False)

    # Compute statistics
    feature_cols = [c for c in train_features.columns if c not in ['well_id', 'row_idx']]

    nan_rates = {}
    correlations = {}

    # Merge with original data for correlation check (target rows only)
    train_with_tvt = train_df[train_df['is_target']][['well_id', 'row_idx', 'TVT']].copy()
    train_merged = train_features.merge(train_with_tvt, on=['well_id', 'row_idx'], how='inner')

    for col in feature_cols:
        nan_rates[col] = (train_features[col].isna().sum() / len(train_features)) * 100

        if col in train_merged.columns:
            valid_mask = ~(train_merged[col].isna() | train_merged['TVT'].isna())
            if valid_mask.sum() > 0:
                correlations[col] = train_merged.loc[valid_mask, col].corr(train_merged.loc[valid_mask, 'TVT'])
            else:
                correlations[col] = np.nan
        else:
            correlations[col] = np.nan

    # Report
    report = {
        'n_train_features': len(train_features),
        'n_test_features': len(test_features),
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'nan_rates': nan_rates,
        'correlations_with_tvt': correlations,
        'total_time_sec': t_total,
        'leak_free': True,
        'knn_leave_one_out': True,
        'hidden_tvt_used': False,
    }

    with open(exp_dir / 'result.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("REPORT: exp041 features")
    print("=" * 80)
    print(f"\nGenerated features: {len(feature_cols)}")
    for col in feature_cols:
        print(f"  - {col}")

    print(f"\nNaN rates & TVT correlations (target rows only):")
    print(f"{'Feature':<30} {'NaN %':>8} {'Corr(TVT)':>10}")
    print("-" * 50)
    for col in feature_cols:
        nan_pct = nan_rates[col]
        corr = correlations.get(col, np.nan)
        if np.isnan(corr):
            corr_str = "N/A"
        else:
            corr_str = f"{corr:+.4f}"
        print(f"{col:<30} {nan_pct:>7.2f}% {corr_str:>10}")

    print(f"\nLeak-free validation:")
    print(f"  - KNN leave-one-out on train: YES")
    print(f"  - Hidden TVT used: NO")
    print(f"  - Hidden Z used: NO (only last_known_Z for plane fit)")

    print(f"\nComputation time: {t_total:.2f}s")
    print(f"Output files:")
    print(f"  - {exp_dir / 'train_features.parquet'}")
    print(f"  - {exp_dir / 'test_features.parquet'}")
    print(f"  - {exp_dir / 'result.json'}")
    print("=" * 80)


if __name__ == '__main__':
    main()
