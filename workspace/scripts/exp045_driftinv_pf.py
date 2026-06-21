#!/usr/bin/env python3
"""exp045: Drift-Invariant NCC-based Emission Particle Filter.

Improves exp040 (CV 10.979) by replacing point-wise GR emission with window-based
normalized cross-correlation (NCC) + robust baseline removal (moving median + MAD).
Addresses exp031's failure (point-value normalization → CV 20.379) via window correlation.

完全leak-free: GR + typewell + anchor + Z + MD のみ使用。
計算: NCC テーブル事前計算 (TVT-bin × MD-row) + PF内で lookup。
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import json

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from numpy.lib.stride_tricks import sliding_window_view

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

# Set BLAS threads to 1 to avoid oversubscription with ProcessPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

EXP_ID = "exp045_driftinv_pf"
OUT_DIR = Path("experiments") / EXP_ID

N_PARTICLES = 500
N_SEEDS = 128
N_WORKERS = max(1, os.cpu_count() or 4)  # Use all cores

MOM = 0.998
VN = 0.002
PN = 0.01
RP = 0.1
RR = 0.001
RESAMP = 0.5
INIT_SPREAD = 4

# NCC tuning parameters
ROBUST_NORM_WIN = 151
TVT_BIN = 2.0  # ft
NCC_WIN = 10  # half-width in samples
NCC_BETA = 4.0  # higher score -> higher likelihood
NCC_WEIGHT_VALUE = 0.6
NCC_WEIGHT_DERIV = 0.4


def robust_norm(gr_vec):
    """Robust normalization: moving median baseline + MAD.

    Args:
        gr_vec: 1D array, GR values (possibly with NaN, should be pre-interpolated)

    Returns:
        Normalized GR (zero-ish mean, robust scale)
    """
    gr_vec = np.asarray(gr_vec, dtype=float)
    if np.all(np.isnan(gr_vec)):
        return np.zeros_like(gr_vec)

    # Moving median baseline removal
    base = median_filter(gr_vec, size=ROBUST_NORM_WIN, mode='nearest')
    g = gr_vec - base

    # MAD normalization
    med_g = np.median(g)
    mad = np.median(np.abs(g - med_g)) + 1e-6
    return (g - med_g) / (1.4826 * mad)


def ncc_compute(a, b):
    """Normalized cross-correlation in [-1, 1].

    Assumes a, b are already aligned (same length or interpolated).
    Returns scalar in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return 0.0

    a_mean = np.mean(a)
    b_mean = np.mean(b)
    a_std = np.std(a)
    b_std = np.std(b)

    if a_std < 1e-8 or b_std < 1e-8:
        return 0.0

    a_norm = (a - a_mean) / a_std
    b_norm = (b - b_mean) / b_std

    return np.dot(a_norm, b_norm) / len(a)


def _normalize_window_batch(windows):
    """Normalize a batch of windows (shape: n_windows, window_size) to zero-mean, unit-std.

    Returns normalized windows (same shape) and a mask of valid windows (len>=2 with non-zero std).
    Valid windows are normalized; invalid ones are zeroed.
    """
    n_windows, win_size = windows.shape

    # Compute mean and std for each window
    means = np.mean(windows, axis=1, keepdims=True)  # (n_windows, 1)
    stds = np.std(windows, axis=1, keepdims=True)    # (n_windows, 1)

    # Mark valid windows (non-zero std)
    valid = (stds[:, 0] > 1e-8) & (win_size >= 2)

    # Normalize (set invalid to 0)
    normalized = np.where(
        valid[:, None],
        (windows - means) / np.where(stds > 0, stds, 1),
        0.0
    )

    return normalized, valid


def _ncc_batch(lat_windows, tw_windows_set):
    """Compute NCC between lateral windows and typewell windows (vectorized).

    Args:
        lat_windows: (n_md, win_size) lateral windows
        tw_windows_set: list of (n_bins_b, win_size) typewell window batches, one per bin

    Returns:
        ncc_matrix: (n_md, n_bins) NCC scores

    Optimization: Pre-filter empty bins, use vectorized comparisons.
    """
    n_md, win_size = lat_windows.shape
    n_bins = len(tw_windows_set)

    # Normalize lateral windows once
    lat_norm, lat_valid = _normalize_window_batch(lat_windows)

    ncc_matrix = np.zeros((n_md, n_bins), dtype=float)

    # Pre-filter non-empty bins
    non_empty_bins = [b for b in range(n_bins) if len(tw_windows_set[b]) > 0]

    for b in non_empty_bins:
        tw_windows = tw_windows_set[b]  # (n_samples_at_b, win_size)

        # Normalize typewell windows
        tw_norm, tw_valid = _normalize_window_batch(tw_windows)

        # NCC: dot product of normalized windows, divided by window size (matmul, highly optimized)
        ncc_vals = np.dot(tw_norm, lat_norm.T) / win_size  # (n_samples_at_b, n_md)

        # Average across typewell samples at this bin
        valid_pairs = tw_valid[:, None] & lat_valid[None, :]
        ncc_sum = np.where(valid_pairs, ncc_vals, 0.0).sum(axis=0)
        ncc_count = valid_pairs.sum(axis=0)
        # Avoid divide-by-zero with explicit check
        with np.errstate(divide='ignore', invalid='ignore'):
            ncc_matrix[:, b] = np.where(ncc_count > 0, ncc_sum / ncc_count, 0.0)

    return ncc_matrix


def compute_ncc_table(z_lat, dz_lat, tw_tvt, z_tw, dz_tw, tvt_min, tvt_max):
    """Pre-compute NCC likelihood table: emission[i, b] for MD row i at TVT-bin b.

    Fully vectorized version using sliding_window_view and batch normalization.

    Args:
        z_lat, dz_lat: lateral normalized GR and its gradient (shape: n_md)
        tw_tvt: typewell TVT grid (shape: n_tw)
        z_tw, dz_tw: typewell normalized GR and its gradient (shape: n_tw)
        tvt_min, tvt_max: TVT range

    Returns:
        E: emission table (n_md, n_tvt_bins) - log-likelihood scores
        tvt_bins: TVT bin centers (shape: n_tvt_bins)
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

    # Initialize emission: -inf (no observation)
    E = np.full((n_md, n_bins), -np.inf, dtype=float)

    # Typewell TVT spacing
    if len(tw_tvt) > 1:
        tw_tvt_spacing = np.median(np.diff(tw_tvt))
    else:
        tw_tvt_spacing = 1.0

    # === LATERAL WINDOWS ===
    if n_md >= win_size:
        lat_windows = sliding_window_view(z_lat, win_size)  # (n_md - win_size + 1, win_size)
        lat_d_windows = sliding_window_view(dz_lat, win_size)
        # Pad to full n_md rows: for rows before/after, use direct window extraction
        lat_windows_full = np.zeros((n_md, win_size), dtype=float)
        lat_d_windows_full = np.zeros((n_md, win_size), dtype=float)

        for i in range(n_md):
            i_lo = max(0, i - NCC_WIN)
            i_hi = min(n_md, i + NCC_WIN + 1)
            win_len = i_hi - i_lo
            pad_left = NCC_WIN - (i - i_lo)

            lat_windows_full[i, pad_left:pad_left + win_len] = z_lat[i_lo:i_hi]
            lat_d_windows_full[i, pad_left:pad_left + win_len] = dz_lat[i_lo:i_hi]
    else:
        # Small well: use direct extraction
        lat_windows_full = np.zeros((n_md, win_size), dtype=float)
        lat_d_windows_full = np.zeros((n_md, win_size), dtype=float)
        for i in range(n_md):
            i_lo = max(0, i - NCC_WIN)
            i_hi = min(n_md, i + NCC_WIN + 1)
            win_len = i_hi - i_lo
            pad_left = NCC_WIN - (i - i_lo)
            lat_windows_full[i, pad_left:pad_left + win_len] = z_lat[i_lo:i_hi]
            lat_d_windows_full[i, pad_left:pad_left + win_len] = dz_lat[i_lo:i_hi]

    # === TYPEWELL WINDOWS per BIN ===
    tw_windows_val_set = []  # list of (n_samples_at_b, win_size) arrays
    tw_windows_der_set = []

    for b, tvt_p in enumerate(tvt_bins):
        tvt_lo = tvt_p - NCC_WIN * tw_tvt_spacing
        tvt_hi = tvt_p + NCC_WIN * tw_tvt_spacing
        tw_mask = (tw_tvt >= tvt_lo) & (tw_tvt <= tvt_hi)

        if tw_mask.sum() < 2:
            tw_windows_val_set.append(np.empty((0, win_size)))
            tw_windows_der_set.append(np.empty((0, win_size)))
            continue

        # Get typewell indices in the window
        tw_indices = np.where(tw_mask)[0]

        # For each typewell index, create a window around it
        tw_vals_at_b = []
        tw_ders_at_b = []

        for tw_idx in tw_indices:
            tw_win_lo = max(0, tw_idx - NCC_WIN)
            tw_win_hi = min(len(z_tw), tw_idx + NCC_WIN + 1)
            win_len = tw_win_hi - tw_win_lo
            pad_left = NCC_WIN - (tw_idx - tw_win_lo)

            # Create window of size win_size
            tw_window_val = np.zeros(win_size, dtype=float)
            tw_window_der = np.zeros(win_size, dtype=float)
            tw_window_val[pad_left:pad_left + win_len] = z_tw[tw_win_lo:tw_win_hi]
            tw_window_der[pad_left:pad_left + win_len] = dz_tw[tw_win_lo:tw_win_hi]

            tw_vals_at_b.append(tw_window_val)
            tw_ders_at_b.append(tw_window_der)

        tw_windows_val_set.append(np.array(tw_vals_at_b, dtype=float))
        tw_windows_der_set.append(np.array(tw_ders_at_b, dtype=float))

    # === BATCH NCC COMPUTATION ===
    ncc_val_matrix = _ncc_batch(lat_windows_full, tw_windows_val_set)  # (n_md, n_bins)
    ncc_der_matrix = _ncc_batch(lat_d_windows_full, tw_windows_der_set)

    # === BLEND & EMIT ===
    score_matrix = NCC_WEIGHT_VALUE * ncc_val_matrix + NCC_WEIGHT_DERIV * ncc_der_matrix

    # Mask invalid entries and apply beta
    valid_entries = ncc_val_matrix != 0  # If NCC was computed
    E[valid_entries] = NCC_BETA * score_matrix[valid_entries]

    return E, tvt_bins


def _pf_single(p, seed):
    """1 seed の PF。(pred_eval[n], log_lik)。

    Uses pre-computed NCC emission table instead of point-wise GR emission.
    """
    md_v = p["md_v"]
    z_v = p["z_v"]
    ir = p["ir"]
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0

    N = N_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (p["last_tvt"] + p["last_Z"]) + INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    prev_MD = p["last_MD"]
    log_lik = 0.0
    lo = p["tvt_min"] - 100
    hi = p["tvt_max"] + 100

    E = p["emission_table"]  # (n_md, n_tvt_bins)
    tvt_bins = p["tvt_bins"]

    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]

        # Lookup NCC emission from pre-computed table
        bin_idx = np.searchsorted(tvt_bins, tvt_p, side='left')
        bin_idx = np.clip(bin_idx - 1, 0, len(tvt_bins) - 1)

        # E[i, bin_idx] is log-likelihood; convert to likelihood
        log_lk_vals = E[i, bin_idx]
        lk = np.exp(np.minimum(log_lk_vals, 600.0))  # Clip to avoid overflow

        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    return res, log_lik


def _process_well(p):
    """1 well を 128 seed で PF シミュレーション (NCC emission)。

    戻り値: (wid, pred[n])。
    """
    wid = p["wid"]
    n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])

    # 128 seed で PF を走らせ、per-seed (preds, liks) を保存
    preds = np.empty((N_SEEDS, n))
    liks = np.empty(N_SEEDS)
    for s in range(N_SEEDS):
        preds[s], liks[s] = _pf_single(p, s)

    # Multi-scale temperature weighting (reuse exp040 strategy)
    SCALES = [3.0, 5.0, 8.0, 12.0]
    multi_pred = np.zeros(n)
    for scale in SCALES:
        wts = np.exp((liks - liks.max()) / scale)
        wts /= wts.sum()
        multi_pred += (wts[:, None] * preds).sum(0) / len(SCALES)

    return wid, multi_pred


def build(base_path, tw_path):
    """Build payloads with NCC emission tables pre-computed."""
    tr = pd.read_parquet(
        base_path,
        columns=[
            "well_id",
            "row_idx",
            "MD",
            "Z",
            "GR",
            "TVT",
            "TVT_input",
            "id",
            "is_target",
            "is_known_tvt",
            "is_gr_missing",
            "last_known_TVT",
        ],
    )
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    payloads = []
    out_frames = []

    for wid, g in sel.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue

        anchor = float(tgt["last_known_TVT"].iloc[0])
        out_frames.append(
            tgt[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy()
        )

        tw_g = tw_by_well.get(wid)
        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both")

        if tw_g is None or len(tw_g) < 2 or len(known) < 2:
            payloads.append(
                {"wid": wid, "no_tw": True, "anchor": anchor, "n_eval": int(len(tgt))}
            )
            continue

        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tw_tvt = tw_s["TVT"].to_numpy(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
        gr_v = gr_full[tgt_mask]

        # Robust normalization
        z_lat = robust_norm(gr_full)
        z_tw = robust_norm(tw_gr)
        dz_lat = np.gradient(z_lat)
        dz_tw = np.gradient(z_tw)

        # Compute NCC emission table
        tvt_min = tw_tvt.min() - 50
        tvt_max = tw_tvt.max() + 50
        E, tvt_bins = compute_ncc_table(z_lat, dz_lat, tw_tvt, z_tw, dz_tw, tvt_min, tvt_max)

        # Compute GR scale (for reference, though not used in NCC emission)
        k_tvt = known["TVT_input"].to_numpy(float)
        k_gr = known["GR"].fillna(0).to_numpy(float)
        tw_at_k = np.interp(k_tvt, tw_tvt, tw_gr)
        gs = float(np.clip(np.nanstd(k_gr - tw_at_k), 10.0, 60.0))

        # Initial rate from tail-30 known
        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float))
        dz = np.diff(tail["Z"].to_numpy(float))
        dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        ir = (
            float(np.median((dt + dz)[mm] / dm[mm]))
            if mm.sum() >= 3
            else 0.0
        )

        last = known.iloc[-1]
        payloads.append(
            {
                "wid": wid,
                "no_tw": False,
                "n_eval": int(len(tgt)),
                "md_v": tgt["MD"].to_numpy(float),
                "z_v": tgt["Z"].to_numpy(float),
                "gr_v": gr_v,
                "gs": gs,
                "ir": ir,
                "last_tvt": float(last["TVT_input"]),
                "last_Z": float(last["Z"]),
                "last_MD": float(last["MD"]),
                "anchor": anchor,
                "emission_table": E,
                "tvt_bins": tvt_bins,
                "tvt_min": tvt_min,
                "tvt_max": tvt_max,
            }
        )

    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


def run_split(base_path, tw_path, is_smoke=False, well_ids=None):
    """Run PF over all wells (or subset if smoke test).

    Args:
        base_path: path to train/test base parquet
        tw_path: path to typewell parquet
        is_smoke: if True, run on first 5 wells
        well_ids: if provided, only run on these well IDs (subset validation)

    Returns:
        out: DataFrame with well_id, row_idx, id, TVT, last_known_TVT, pred_tvt
    """
    payloads, out = build(base_path, tw_path)

    if is_smoke:
        payloads = payloads[:5]
        out = out[out["well_id"].isin([p["wid"] for p in payloads])]
        print(f"  Smoke test: {len(payloads)} wells, {len(out)} rows")
    elif well_ids is not None:
        well_ids_set = set(well_ids)
        payloads = [p for p in payloads if p["wid"] in well_ids_set]
        out = out[out["well_id"].isin(well_ids_set)]
        print(f"  Subset (well_ids): {len(payloads)} wells, {len(out)} rows")

    pred_by_wid = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred) in enumerate(ex.map(_process_well, payloads, chunksize=4)):
            pred_by_wid[wid] = pred
            if (k + 1) % max(1, len(payloads) // 10) == 0 or (k + 1) == len(payloads):
                print(f"    {k+1}/{len(payloads)} wells done", flush=True)

    # Assemble predictions aligned to out
    out = out.reset_index(drop=True)  # Reset to 0-based indices
    out["pred_tvt"] = np.nan

    for wid, pred_array in pred_by_wid.items():
        well_mask = out["well_id"] == wid
        well_indices = np.where(well_mask.values)[0]
        out.loc[well_indices, "pred_tvt"] = pred_array

    return out


def run_subset_validation(base_path, tw_path, exp022_well_path):
    """Validate NCC table on broken+good well mix before full run.

    Args:
        base_path: train base parquet
        tw_path: typewell train parquet
        exp022_well_path: path to exp022 per_well.csv

    Returns:
        dict with "broken_rescued", "good_degraded_count", "good_median_delta", "should_run_full"
    """
    # Load exp022 baseline
    exp022_well = pd.read_csv(exp022_well_path)

    # Select subset: broken (pf_rmse > 20) and good (pf_rmse < 8)
    broken_wells = exp022_well[exp022_well["pf_rmse"] > 20]["well_id"].tolist()
    good_wells = exp022_well[exp022_well["pf_rmse"] < 8]["well_id"].tolist()

    n_broken_target = min(15, len(broken_wells))
    n_good_target = min(15, len(good_wells))

    broken_subset = broken_wells[:n_broken_target]
    good_subset = good_wells[:n_good_target]
    subset_well_ids = broken_subset + good_subset

    print(f"\n  Subset validation: {len(broken_subset)} broken + {len(good_subset)} good = {len(subset_well_ids)} wells")

    # Run PF on subset
    subset_preds = run_split(base_path, tw_path, is_smoke=False, well_ids=subset_well_ids)

    # Compute per-well RMSE
    subset_well_rmse = []
    for wid, g in subset_preds.groupby("well_id"):
        rmse = tvt_rmse(g["TVT"], g["pred_tvt"])
        subset_well_rmse.append({"well_id": wid, "exp045_rmse": rmse})

    subset_df = pd.DataFrame(subset_well_rmse)

    # Compare with exp022
    subset_df = subset_df.merge(exp022_well[["well_id", "pf_rmse"]], on="well_id", how="left")
    subset_df.columns = ["well_id", "exp045_rmse", "exp022_rmse"]

    # Count broken rescued
    broken_rescued_mask = (
        (subset_df["exp022_rmse"] > 20) & (subset_df["exp045_rmse"] <= 20)
    )
    n_broken_rescued = int(broken_rescued_mask.sum())

    # Count good degraded (RMSE increase > 0.5)
    good_mask = subset_df["exp022_rmse"] < 8
    good_subset_df = subset_df[good_mask]
    if len(good_subset_df) > 0:
        delta_rmse = good_subset_df["exp045_rmse"] - good_subset_df["exp022_rmse"]
        good_median_delta = float(np.median(delta_rmse))
        n_good_degraded = int((delta_rmse > 0.5).sum())
    else:
        good_median_delta = 0.0
        n_good_degraded = 0

    # Decision logic: broken_rescued > 5 AND good_median_delta < 0.5
    should_run_full = (n_broken_rescued > 5) and (good_median_delta < 0.5)

    result = {
        "subset_wells": len(subset_well_ids),
        "broken_subset_size": len(broken_subset),
        "good_subset_size": len(good_subset),
        "broken_rescued": n_broken_rescued,
        "good_degraded_count": n_good_degraded,
        "good_median_delta": good_median_delta,
        "should_run_full": should_run_full,
    }

    # Save results
    subset_df.to_csv(OUT_DIR / "subset_check_details.csv", index=False)

    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[{EXP_ID}] Drift-Invariant NCC Particle Filter "
        f"(workers={N_WORKERS}, seeds={N_SEEDS}, particles={N_PARTICLES})"
    )
    print(f"  Emission: NCC window (robust norm + moving median, win={ROBUST_NORM_WIN}, W={NCC_WIN})")
    print(f"  NCC blend: {NCC_WEIGHT_VALUE:.1f}*value + {NCC_WEIGHT_DERIV:.1f}*deriv, beta={NCC_BETA}")
    print(f"  BLAS threads=1 (per-worker), ProcessPoolExecutor max_workers={N_WORKERS}")

    wall_start = time.time()

    # Smoke test: 5 wells (vectorization sanity check)
    print("\n=== SMOKE TEST (5 wells, vectorization check) ===", flush=True)
    smoke_start = time.time()
    try:
        smoke_preds = run_split(
            "data/processed/train_base_v001.parquet",
            "data/processed/typewell_train_base_v001.parquet",
            is_smoke=True,
        )
        smoke_cv = tvt_rmse(smoke_preds["TVT"], smoke_preds["pred_tvt"])
        smoke_time = time.time() - smoke_start
        print(f"✓ Smoke completed in {smoke_time:.1f}s, temp CV={smoke_cv:.6f}")
        assert len(smoke_preds) > 0, "Smoke test produced no predictions"
        assert not smoke_preds["pred_tvt"].isna().any(), "NaN predictions in smoke test"
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Subset validation: broken+good mix (30 wells)
    print("\n=== SUBSET VALIDATION (broken+good mix) ===", flush=True)
    subset_start = time.time()
    try:
        subset_result = run_subset_validation(
            "data/processed/train_base_v001.parquet",
            "data/processed/typewell_train_base_v001.parquet",
            "experiments/exp022_particle_filter/per_well.csv",
        )
        subset_time = time.time() - subset_start
        print(f"  Subset validation completed in {subset_time:.1f}s")
        print(f"    Broken rescued: {subset_result['broken_rescued']}/{subset_result['broken_subset_size']}")
        print(f"    Good median delta RMSE: {subset_result['good_median_delta']:+.4f}")
        print(f"    Good degraded >0.5: {subset_result['good_degraded_count']}/{subset_result['good_subset_size']}")

        # Save subset check result
        with open(OUT_DIR / "subset_check.json", "w") as f:
            json.dump(subset_result, f, indent=2)

        should_run_full = subset_result["should_run_full"]
        print(f"\n  Decision: {'RUN FULL' if should_run_full else 'SKIP FULL (emission improvement insufficient)'}")

        if not should_run_full:
            print(f"\n  Reason: broken_rescued={subset_result['broken_rescued']} (need >5) "
                  f"or good_median_delta={subset_result['good_median_delta']:.4f} (need <0.5)")
            print(f"  Conclusion: Emission improvement is marginal; full run not warranted.")
            return

    except Exception as e:
        print(f"✗ Subset validation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Full 773 wells (only if subset passes)
    print("\n=== FULL RUN (773 wells) ===", flush=True)
    full_start = time.time()
    preds = run_split(
        "data/processed/train_base_v001.parquet",
        "data/processed/typewell_train_base_v001.parquet",
        is_smoke=False,
    )
    cv = tvt_rmse(preds["TVT"], preds["pred_tvt"])
    anc = tvt_rmse(preds["TVT"], preds["last_known_TVT"])
    full_time = time.time() - full_start
    print(f"✓ Full 773 wells completed in {full_time:.1f}s")
    print(f"  Drift-invariant NCC PF CV = {cv:.6f}   anchor = {anc:.6f}")

    # Verify baseline comparison
    print(f"\n  === COMPARISON WITH BASELINES ===")
    print(f"    exp040 (multi-scale PF) = 10.979086")
    print(f"    exp022 (single-scale PF) = 11.024014")
    if cv < 10.979086:
        improvement = 10.979086 - cv
        print(f"    ✓ exp045 IMPROVEMENT: +{improvement:.6f}")
    else:
        degradation = cv - 10.979086
        print(f"    ✗ exp045 DEGRADATION: -{degradation:.6f}")

    # Well-level analysis
    preds["error"] = preds["pred_tvt"] - preds["TVT"]
    preds["abs_error"] = preds["error"].abs()
    preds.to_csv(OUT_DIR / "oof.csv", index=False)

    well_rows = []
    for wid, g in preds.groupby("well_id"):
        well_rows.append(
            {
                "well_id": wid,
                "n": len(g),
                "anchor_rmse": tvt_rmse(g["TVT"], g["last_known_TVT"]),
                "pf_rmse": tvt_rmse(g["TVT"], g["pred_tvt"]),
            }
        )
    well = pd.DataFrame(well_rows)
    well.to_csv(OUT_DIR / "per_well.csv", index=False)

    n_beat = int((well["pf_rmse"] < well["anchor_rmse"]).sum())
    n_broken = int((well["pf_rmse"] > 20).sum())
    print(f"  Drift-invariant NCC PF が anchor に勝つ well: {n_beat}/{len(well)}")
    print(f"  RMSE > 20 (broken): {n_broken}")

    # Broken well rescue analysis (vs exp022 baseline)
    exp022_well = pd.read_csv("experiments/exp022_particle_filter/per_well.csv")
    baseline_broken = set(exp022_well[exp022_well["pf_rmse"] > 20]["well_id"])
    rescued = well[(well["well_id"].isin(baseline_broken)) & (well["pf_rmse"] <= 20)]
    print(f"  Baseline (exp022) broken wells rescued to RMSE<=20: {len(rescued)}/{len(baseline_broken)}")

    # Good well stability (vs exp040)
    exp040_well = pd.read_csv("experiments/exp040_multiscale_pf/per_well.csv")
    good_wells = set(exp040_well[exp040_well["pf_rmse"] < 10]["well_id"])
    good_stability = well[well["well_id"].isin(good_wells)]
    exp040_good = exp040_well[exp040_well["pf_rmse"] < 10]
    merged_good = good_stability.merge(exp040_good[["well_id", "pf_rmse"]], on="well_id", suffixes=("", "_exp040"))
    if len(merged_good) > 0:
        delta_rmse = (merged_good["pf_rmse"] - merged_good["pf_rmse_exp040"]).values
        median_delta = float(np.median(delta_rmse))
        n_degraded = (delta_rmse > 0.1).sum()
        print(f"  Good well (exp040 pf<10) stability:")
        print(f"    Median delta RMSE: {median_delta:+.4f}")
        print(f"    Wells degraded >0.1: {n_degraded}/{len(merged_good)}")

    # Test submission
    print("\nTEST submission (drift-invariant NCC PF) ...", flush=True)
    test_preds = run_split(
        "data/processed/test_base_v001.parquet",
        "data/processed/typewell_test_base_v001.parquet",
        is_smoke=False,
    )
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(
        test_preds[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
        on="id",
        how="left",
        validate="one_to_one",
    )
    assert not sub["tvt"].isna().any()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"  submission rows: {len(sub)}")
    test_preds.to_csv(OUT_DIR / "test_pred.csv", index=False)

    truth = pd.read_parquet(
        "data/processed/train_base_v001.parquet", columns=["well_id", "row_idx", "TVT"]
    )
    tp = test_preds.merge(truth, on=["well_id", "row_idx"], how="left", suffixes=("", "_truth"))
    test_well_rmse = {}
    for wid, g in tp.groupby("well_id"):
        test_well_rmse[wid] = tvt_rmse(g["TVT_truth"], g["pred_tvt"])
        print(f"    test {wid}: drift-inv NCC PF vs train-truth RMSE = {test_well_rmse[wid]:.4f}")

    wall_time_sec = time.time() - wall_start

    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "method": f"Drift-Invariant NCC Particle Filter (vectorized, {N_WORKERS} workers, {N_SEEDS} seeds × {N_PARTICLES} particles)",
        "cv_rmse": float(cv),
        "anchor_rmse": float(anc),
        "n_wells": int(len(well)),
        "n_pf_beats_anchor": int(n_beat),
        "n_broken": int(n_broken),
        "n_broken_rescued_vs_exp022": int(len(rescued)),
        "test_well_rmse_vs_train_truth": test_well_rmse,
        "leak_risk": "none (no hidden TVT used; GR+typewell+anchor+Z+MD only)",
        "emission_params": {
            "robust_norm_window": ROBUST_NORM_WIN,
            "ncc_window_halfwidth": NCC_WIN,
            "tvt_bin_ft": TVT_BIN,
            "ncc_beta": NCC_BETA,
            "ncc_weight_value": NCC_WEIGHT_VALUE,
            "ncc_weight_deriv": NCC_WEIGHT_DERIV,
        },
        "compare": {
            "exp040_multiscale_pf": 10.979086,
            "exp022_single_scale": 11.024014,
            "anchor": float(anc),
        },
        "timing": {
            "smoke_test_sec": float(smoke_time),
            "subset_validation_sec": float(subset_time),
            "full_run_sec": float(full_time),
            "total_sec": float(wall_time_sec),
        },
        "notes": "Drift-invariant NCC-based emission replaces point-wise GR. "
        "Fully vectorized compute_ncc_table using sliding_window_view + batch normalization. "
        "Robust normalization (moving median + MAD) on both lateral and typewell GR. "
        "Window correlation (value + derivative, 0.6:0.4 blend) pre-computed as TVT-bin × MD lookup table. "
        "Multi-scale temperature re-weighting (scales 3,5,8,12) from 128 seeds. "
        "ProcessPoolExecutor with BLAS thread pinning. Smoke → subset validation → full pipeline.",
    }
    write_json(OUT_DIR / "result.json", result)

    twr = "\n".join(f"| {k} | {v:.4f} |" for k, v in test_well_rmse.items())
    (OUT_DIR / "notes.md").write_text(
        f"""# {EXP_ID} — Drift-Invariant NCC Particle Filter

## 手法
exp040 の点値GR尤度を **窓相互相関(NCC)** ベース + **robust baseline除去** に改良。
- **前処理**: 移動中央値 baseline → MAD 正規化（GR/typewell両側）
- **Emission**: lateral GR 窓 vs typewell GR 窓の NCC (value 60% + derivative 40%)
- **テーブル化**: TVT-bin × MD-row で事前計算、PF内で lookup
- **温度**: exp040と同じ multi-scale (3.0, 5.0, 8.0, 12.0)
- **完全leak-free**: GR + typewell + anchor + Z + MD のみ

## 結果 (773-well pooled CV)
| 手法 | CV RMSE |
|---|---|
| anchor | {anc:.6f} |
| **Drift-inv NCC PF (exp045)** | **{cv:.6f}** |
| exp040 multi-scale PF | 10.979086 |
| exp022 single-scale PF | 11.024014 |

ベースライン (exp040 10.979) との比較:
{f'✓ 改善: +{10.979086 - cv:.6f}' if cv < 10.979086 else f'✗ 悪化: -{cv - 10.979086:.6f}'}

Drift-inv NCC PF が anchor に勝つ well: {n_beat}/{len(well)}
RMSE > 20 (壊れたwell): {n_broken}

### broken well 救済 (exp022 ベース)
exp022 で RMSE>20 だった {len(baseline_broken)} 本中、exp045 で RMSE<=20 に改善: **{len(rescued)}本**

### good well 安定性 (exp040 ベース)
exp040 で pf_rmse<10 だった {len(good_wells)} 本の中で:
{f'Median delta RMSE: {median_delta:+.4f} (n_degraded={n_degraded})' if len(merged_good) > 0 else 'N/A'}

## 3 test well (Drift-inv NCC PF vs train真値, 参照"4.71 ft"と照合)
| well | RMSE |
|---|---|
{twr}

## Emission パラメータ
- robust_norm window: {ROBUST_NORM_WIN}
- NCC window半幅: {NCC_WIN} samples
- TVT bin幅: {TVT_BIN} ft
- NCC beta (尤度): {NCC_BETA}
- NCC weight (value:deriv): {NCC_WEIGHT_VALUE}:{NCC_WEIGHT_DERIV}

## 計算時間
- Smoke test (5 wells): {smoke_time:.1f} 秒
- Full (773 wells): {wall_time_sec - smoke_time:.1f} 秒
- 総計: {wall_time_sec:.1f} 秒

## 失敗分析 (該当時)
exp031 が点値正規化で失敗 (CV 20.379) した原因:
- 点値比較は GR baseline drift に脆弱
- typewell との offset が未知で点では照合不可

→ 窓相関 (NCC) で形状パターン照合に変更:
- baseline drift 不変 (window内の相対形状のみ比較)
- 微分項追加で peak timing 整合
- 移動中央値で local trend 除去

## リンク
[[exp040_multiscale_pf]] [[exp022_particle_filter]] [[exp031_pf_physical_lik]]
"""
    )

    print(f"\nCompleted in {wall_time_sec:.1f} sec")
    print(f"CV: {cv:.6f} (exp040 baseline: 10.979086)")
    if cv < 10.979086:
        improvement = 10.979086 - cv
        print(f"✓ Improvement: +{improvement:.6f}")
    else:
        degradation = cv - 10.979086
        print(f"✗ Degradation: {degradation:.6f}")


if __name__ == "__main__":
    main()
