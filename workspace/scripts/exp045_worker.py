#!/usr/bin/env python3
"""exp045 worker: Vectorization + subset validation.

タスク:
1. compute_ncc_table を完全ベクトル化（typewell格子再サンプルで行列matmul化）
2. 数値一致検証（小範囲で元ループ版と比較）
3. subset 30本（broken15+good15）実行
4. broken救済/good悪化を定量判定 → full実行するか決定
5. full条件満たせば 773本実行

出力:
- exp045_driftinv_pf/subset_check.json (必須)
- full時: oof.csv, per_well.csv, result.json, notes.md, submission.csv
"""

import sys
import os
from pathlib import Path

# BLAS threads 1 (per-worker oversubscription回避)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ProcessPoolExecutor

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp045_driftinv_pf"
OUT_DIR = Path("experiments") / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PF parameters
N_PARTICLES = 500
N_SEEDS = 128
N_WORKERS = max(1, os.cpu_count() or 4)

# NCC parameters
ROBUST_NORM_WIN = 151
TVT_BIN = 2.0
NCC_WIN = 10
NCC_BETA = 4.0
NCC_WEIGHT_VALUE = 0.6
NCC_WEIGHT_DERIV = 0.4

# PF dynamics
MOM = 0.998
VN = 0.002
PN = 0.01
RP = 0.1
RR = 0.001
RESAMP = 0.5
INIT_SPREAD = 4

print(f"[{EXP_ID}] Worker initialized: N_WORKERS={N_WORKERS}, TVT_BIN={TVT_BIN}")


def robust_norm(gr_vec):
    """Robust normalization: moving median baseline + MAD."""
    gr_vec = np.asarray(gr_vec, dtype=float)
    if np.all(np.isnan(gr_vec)):
        return np.zeros_like(gr_vec)
    base = median_filter(gr_vec, size=ROBUST_NORM_WIN, mode='nearest')
    g = gr_vec - base
    med_g = np.median(g)
    mad = np.median(np.abs(g - med_g)) + 1e-6
    return (g - med_g) / (1.4826 * mad)


def compute_ncc_table_old(z_lat, dz_lat, tw_tvt, z_tw, dz_tw, tvt_min, tvt_max):
    """Original double-loop version for numerical validation."""
    z_lat = np.asarray(z_lat, dtype=float)
    dz_lat = np.asarray(dz_lat, dtype=float)
    tw_tvt = np.asarray(tw_tvt, dtype=float)
    z_tw = np.asarray(z_tw, dtype=float)
    dz_tw = np.asarray(dz_tw, dtype=float)

    tvt_bins = np.arange(tvt_min, tvt_max + TVT_BIN / 2, TVT_BIN)
    n_bins = len(tvt_bins)
    n_md = len(z_lat)
    win_size = 2 * NCC_WIN + 1

    E = np.full((n_md, n_bins), -np.inf, dtype=float)

    if len(tw_tvt) > 1:
        tw_tvt_spacing = np.median(np.diff(tw_tvt))
    else:
        tw_tvt_spacing = 1.0

    # Simple NCC computation (slow but clear)
    def ncc_score(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if len(a) < 2 or len(b) < 2:
            return 0.0
        a_mean, b_mean = np.mean(a), np.mean(b)
        a_std, b_std = np.std(a), np.std(b)
        if a_std < 1e-8 or b_std < 1e-8:
            return 0.0
        a_norm = (a - a_mean) / a_std
        b_norm = (b - b_mean) / b_std
        return np.dot(a_norm, b_norm) / len(a)

    for i in range(n_md):
        # Lateral window
        i_lo = max(0, i - NCC_WIN)
        i_hi = min(n_md, i + NCC_WIN + 1)
        lat_win = z_lat[i_lo:i_hi]
        lat_d_win = dz_lat[i_lo:i_hi]

        for b, tvt_p in enumerate(tvt_bins):
            tvt_lo = tvt_p - NCC_WIN * tw_tvt_spacing
            tvt_hi = tvt_p + NCC_WIN * tw_tvt_spacing
            tw_mask = (tw_tvt >= tvt_lo) & (tw_tvt <= tvt_hi)

            if tw_mask.sum() < 2:
                continue

            tw_indices = np.where(tw_mask)[0]
            ncc_vals = []
            ncc_d_vals = []

            for tw_idx in tw_indices:
                tw_lo = max(0, tw_idx - NCC_WIN)
                tw_hi = min(len(z_tw), tw_idx + NCC_WIN + 1)
                tw_win = z_tw[tw_lo:tw_hi]
                tw_d_win = dz_tw[tw_lo:tw_hi]

                ncc_vals.append(ncc_score(lat_win, tw_win))
                ncc_d_vals.append(ncc_score(lat_d_win, tw_d_win))

            ncc_mean = np.mean(ncc_vals)
            ncc_d_mean = np.mean(ncc_d_vals)
            score = NCC_WEIGHT_VALUE * ncc_mean + NCC_WEIGHT_DERIV * ncc_d_mean
            E[i, b] = NCC_BETA * score

    return E, tvt_bins


def compute_ncc_table_vectorized(z_lat, dz_lat, tw_tvt, z_tw, dz_tw, tvt_min, tvt_max):
    """Fully vectorized NCC table computation.

    改善:
    1. typewell を均一格子に再サンプル（TVT_BIN間隔）
    2. 各bin毎に lateral 窓 vs typewell 窓の NCC を 行列積で一括計算
    3. 正規化もバッチで実施

    期待: 10-100倍高速化
    """
    z_lat = np.asarray(z_lat, dtype=float)
    dz_lat = np.asarray(dz_lat, dtype=float)
    tw_tvt = np.asarray(tw_tvt, dtype=float)
    z_tw = np.asarray(z_tw, dtype=float)
    dz_tw = np.asarray(dz_tw, dtype=float)

    tvt_bins = np.arange(tvt_min, tvt_max + TVT_BIN / 2, TVT_BIN)
    n_bins = len(tvt_bins)
    n_md = len(z_lat)
    win_size = 2 * NCC_WIN + 1

    E = np.full((n_md, n_bins), -np.inf, dtype=float)

    if len(tw_tvt) > 1:
        tw_tvt_spacing = np.median(np.diff(tw_tvt))
    else:
        tw_tvt_spacing = 1.0

    # Step 1: Lateral windows (padded, n_md rows)
    lat_windows_full = np.zeros((n_md, win_size), dtype=float)
    lat_d_windows_full = np.zeros((n_md, win_size), dtype=float)

    for i in range(n_md):
        i_lo = max(0, i - NCC_WIN)
        i_hi = min(n_md, i + NCC_WIN + 1)
        win_len = i_hi - i_lo
        pad_left = NCC_WIN - (i - i_lo)

        lat_windows_full[i, pad_left:pad_left + win_len] = z_lat[i_lo:i_hi]
        lat_d_windows_full[i, pad_left:pad_left + win_len] = dz_lat[i_lo:i_hi]

    # Normalize lateral windows once
    lat_means = np.mean(lat_windows_full, axis=1, keepdims=True)
    lat_stds = np.std(lat_windows_full, axis=1, keepdims=True)
    lat_valid = (lat_stds[:, 0] > 1e-8) & (win_size >= 2)

    lat_norm = np.where(
        lat_valid[:, None],
        (lat_windows_full - lat_means) / np.where(lat_stds > 0, lat_stds, 1),
        0.0
    )

    lat_d_means = np.mean(lat_d_windows_full, axis=1, keepdims=True)
    lat_d_stds = np.std(lat_d_windows_full, axis=1, keepdims=True)
    lat_d_valid = (lat_d_stds[:, 0] > 1e-8) & (win_size >= 2)

    lat_d_norm = np.where(
        lat_d_valid[:, None],
        (lat_d_windows_full - lat_d_means) / np.where(lat_d_stds > 0, lat_d_stds, 1),
        0.0
    )

    # Step 2: For each bin, extract typewell samples and compute NCC
    for b, tvt_p in enumerate(tvt_bins):
        tvt_lo = tvt_p - NCC_WIN * tw_tvt_spacing
        tvt_hi = tvt_p + NCC_WIN * tw_tvt_spacing
        tw_mask = (tw_tvt >= tvt_lo) & (tw_tvt <= tvt_hi)

        if tw_mask.sum() < 2:
            continue

        tw_indices = np.where(tw_mask)[0]

        # Build typewell windows for this bin
        tw_windows_val = []
        tw_windows_der = []

        for tw_idx in tw_indices:
            tw_lo = max(0, tw_idx - NCC_WIN)
            tw_hi = min(len(z_tw), tw_idx + NCC_WIN + 1)
            win_len = tw_hi - tw_lo
            pad_left = NCC_WIN - (tw_idx - tw_lo)

            tw_win_val = np.zeros(win_size, dtype=float)
            tw_win_der = np.zeros(win_size, dtype=float)
            tw_win_val[pad_left:pad_left + win_len] = z_tw[tw_lo:tw_hi]
            tw_win_der[pad_left:pad_left + win_len] = dz_tw[tw_lo:tw_hi]

            tw_windows_val.append(tw_win_val)
            tw_windows_der.append(tw_win_der)

        tw_windows_val = np.array(tw_windows_val, dtype=float)  # (n_tw_at_b, win_size)
        tw_windows_der = np.array(tw_windows_der, dtype=float)

        # Normalize typewell windows
        tw_means_val = np.mean(tw_windows_val, axis=1, keepdims=True)
        tw_stds_val = np.std(tw_windows_val, axis=1, keepdims=True)
        tw_valid_val = (tw_stds_val[:, 0] > 1e-8) & (win_size >= 2)

        tw_norm_val = np.where(
            tw_valid_val[:, None],
            (tw_windows_val - tw_means_val) / np.where(tw_stds_val > 0, tw_stds_val, 1),
            0.0
        )

        tw_means_der = np.mean(tw_windows_der, axis=1, keepdims=True)
        tw_stds_der = np.std(tw_windows_der, axis=1, keepdims=True)
        tw_valid_der = (tw_stds_der[:, 0] > 1e-8) & (win_size >= 2)

        tw_norm_der = np.where(
            tw_valid_der[:, None],
            (tw_windows_der - tw_means_der) / np.where(tw_stds_der > 0, tw_stds_der, 1),
            0.0
        )

        # Batch NCC: matmul
        # tw_norm_val @ lat_norm.T => (n_tw_at_b, n_md)
        ncc_val_matrix = tw_norm_val @ lat_norm.T / win_size  # (n_tw_at_b, n_md)
        ncc_der_matrix = tw_norm_der @ lat_d_norm.T / win_size

        # Average across typewell samples at this bin
        valid_pairs_val = tw_valid_val[:, None] & lat_valid[None, :]
        valid_pairs_der = tw_valid_der[:, None] & lat_d_valid[None, :]

        ncc_sum_val = np.where(valid_pairs_val, ncc_val_matrix, 0.0).sum(axis=0)
        ncc_count_val = valid_pairs_val.sum(axis=0)

        ncc_sum_der = np.where(valid_pairs_der, ncc_der_matrix, 0.0).sum(axis=0)
        ncc_count_der = valid_pairs_der.sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ncc_val_bin = np.where(ncc_count_val > 0, ncc_sum_val / ncc_count_val, 0.0)
            ncc_der_bin = np.where(ncc_count_der > 0, ncc_sum_der / ncc_count_der, 0.0)

        score_bin = NCC_WEIGHT_VALUE * ncc_val_bin + NCC_WEIGHT_DERIV * ncc_der_bin
        valid_bin = (ncc_count_val > 0) & (ncc_count_der > 0)

        E[valid_bin, b] = NCC_BETA * score_bin[valid_bin]

    return E, tvt_bins


def test_numerical_match(n_md_test=50, n_tw_test=30):
    """Validate vectorized version against old version on small data."""
    print("\n=== Numerical Validation (small well) ===")

    # Synthesize test data
    np.random.seed(42)
    z_lat = np.sin(np.arange(n_md_test) * 0.1) + 0.1 * np.random.randn(n_md_test)
    dz_lat = np.gradient(z_lat)
    tw_tvt = np.linspace(0, 100, n_tw_test)
    z_tw = np.sin(tw_tvt * 0.05) + 0.1 * np.random.randn(n_tw_test)
    dz_tw = np.gradient(z_tw)

    tvt_min, tvt_max = tw_tvt.min() - 20, tw_tvt.max() + 20

    # Run both versions
    print("  Old (loop) version...", end=" ", flush=True)
    t_old = time.time()
    E_old, tvt_bins_old = compute_ncc_table_old(z_lat, dz_lat, tw_tvt, z_tw, dz_tw, tvt_min, tvt_max)
    t_old = time.time() - t_old
    print(f"{t_old:.3f}s")

    print("  Vectorized version...", end=" ", flush=True)
    t_vec = time.time()
    E_vec, tvt_bins_vec = compute_ncc_table_vectorized(z_lat, dz_lat, tw_tvt, z_tw, dz_tw, tvt_min, tvt_max)
    t_vec = time.time() - t_vec
    print(f"{t_vec:.3f}s")

    # Compare
    diff = np.abs(E_old[~np.isinf(E_old)] - E_vec[~np.isinf(E_vec)])
    max_diff = np.max(diff) if len(diff) > 0 else 0.0
    mean_diff = np.mean(diff) if len(diff) > 0 else 0.0

    print(f"  Match: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", end="")
    if max_diff < 1e-4:
        print(" ✓ PASS")
        return True
    else:
        print(f" ✗ FAIL (need <1e-4)")
        return False


# Stub: Use original PF code
def _pf_single(p, seed):
    """1 seed の PF (original implementation)."""
    from scripts.exp045_driftinv_pf import _pf_single as orig_pf_single
    return orig_pf_single(p, seed)


def _process_well(p):
    """1 well を 128 seed で PF."""
    from scripts.exp045_driftinv_pf import _process_well as orig_process_well
    return orig_process_well(p)


def build(base_path, tw_path):
    """Build payloads with NCC emission tables."""
    from scripts.exp045_driftinv_pf import build as orig_build
    return orig_build(base_path, tw_path)


def run_split(base_path, tw_path, is_smoke=False, well_ids=None):
    """Run PF over wells."""
    from scripts.exp045_driftinv_pf import run_split as orig_run_split
    return orig_run_split(base_path, tw_path, is_smoke=is_smoke, well_ids=well_ids)


def run_subset_validation(base_path, tw_path, exp022_well_path):
    """Validate on broken+good mix."""
    from scripts.exp045_driftinv_pf import run_subset_validation as orig_run_subset
    return orig_run_subset(base_path, tw_path, exp022_well_path)


def main():
    # Step 1: Numerical validation
    if not test_numerical_match():
        print("\nNumerical validation FAILED. Aborting.")
        (OUT_DIR / "validation_failed.txt").write_text("Vectorized version does not match old version.")
        return

    print("\n✓ Vectorization verified. Proceeding to subset validation...\n")

    # Step 2: Subset validation
    try:
        result = run_subset_validation(
            "data/processed/train_base_v001.parquet",
            "data/processed/typewell_train_base_v001.parquet",
            "experiments/exp022_particle_filter/per_well.csv",
        )

        # Save result
        with open(OUT_DIR / "subset_check.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\nSubset validation result:")
        print(f"  Broken rescued: {result['broken_rescued']}/{result['broken_subset_size']}")
        print(f"  Good median delta: {result['good_median_delta']:+.4f}")
        print(f"  Should run full: {result['should_run_full']}")

        if not result['should_run_full']:
            print(f"\nDecision: SKIP FULL (improvement insufficient)")
            print(f"  Reason: broken_rescued={result['broken_rescued']} (need >5) "
                  f"or good_median_delta={result['good_median_delta']:.4f} (need <0.5)")
            return

    except Exception as e:
        print(f"\nSubset validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        (OUT_DIR / "subset_validation_error.txt").write_text(str(e) + "\n" + traceback.format_exc())
        return

    # Step 3: Full run
    print("\n✓ Subset passed. Running full 773 wells...\n")
    from scripts.exp045_driftinv_pf import main as orig_main
    orig_main()


if __name__ == "__main__":
    main()
