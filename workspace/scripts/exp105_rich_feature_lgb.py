#!/usr/bin/env python3
"""exp105: Rich leak-free domain features + LightGBM (TVT delta from anchor).

戦略: 公開clean(LB7.297)/上位は「リッチなleak-freeドメイン特徴をGBMに融合」して強い。
我々の honest stack は CV~9.89 止まり=特徴が薄い。本実験は mycarta/rogii-geosteering-toolkit
(MIT) の手法を自前実装で移植し、特徴量を厚くして 9.89 を破ることを狙う。

実装する特徴群 (全て leak-free; 真TVTは評価のみ・学習特徴には typewell由来 + 既知幾何のみ):
  G1  multi-scale NCC implied-TVT vs typewell GR-TVTプロファイル (peak/width/secdist/pred_delta)
  G2  multi-scale NCC self-correlation (pre-PS lateral GR参照, peak idx -> TVT直接)
  G3  anchor 特徴 (last_known_tvt, md距離, eval-zone相対位置)
  G4  trajectory 特徴 (Z, dZ, dXY, inclination, signed azimuth sin/cos, dz/md, tvt-z lateral slope, dz_pred_delta)
  G5  local GR rolling (mean/std/min/max/range/slope, 複数window grm/grs)
  G6  Q-3D tortuosity (Jing2022 essence: 最小曲率inc/azi -> 区間 arc/chord ratio, DLS統計)
  G7  GR微分 (gr_d1, gr_d2) + affine GR較正 (cal_a, cal_b; TVT_inputのみ使用)
  G11 segment-well decomposition (early/mid/late TVT-vs-MD fit)
  G13 landing-zone state (PS時点の inc/state/dz_dmd/tvt slope/match quality)
  G14 well-length 幾何
  G_PF 我々のPF予測 (exp022 / exp100 の pred_tvt と anchorからのdelta)

CV: StratifiedGroupKFold by well (signed azimuth quadrant × median TVT bin × XY grid bin)
    + 既存 typewell-grouped fold でも評価 (leak膨張検査 / honest)。
評価は hidden tail (is_target) 行のみ。

Leak規約:
  - train-only地層列 (horizontal well Geology) は不使用。typewell GR-TVTプロファイルのみ参照。
  - 較正(cal_a,cal_b)は TVT_input (既知区間) のみ。
  - offset-well prior 系は fold外pool必須のため本実験では簡略 (per-well特徴に限定し fold leakを排除)。
  - 全特徴は (予測対象の真TVTを使わず) 既知GR/typewell/幾何/anchorのみから計算。

Usage:
  python scripts/exp105_rich_feature_lgb.py --smoke        # >=40 well subset
  python scripts/exp105_rich_feature_lgb.py                # full 773 well
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FOLDS = ROOT / "data" / "folds"
EXP_DIR = ROOT / "experiments" / "exp105_rich_feature_lgb"
CACHE = EXP_DIR / "features_cache.parquet"
EXP_DIR.mkdir(parents=True, exist_ok=True)

EVAL_STRIDE = 10          # 粗いstrideで特徴計算 -> 線形補間で全eval行へ (mycarta流, 高速化の要)
NCC_SCALES = (8, 15, 25)
SEARCH_RADIUS_FT = 80.0   # NCC peak探索をanchor TVT近傍に制限
RANDOM_STATE = 42
N_SPLITS = 5

# =====================================================================
# GR cleaning helpers (mycarta clean_gr port)
# =====================================================================

def _fill_short_gaps(x, max_gap=5):
    return pd.Series(x).interpolate(limit=max_gap, limit_direction="both").values


def _despike_hampel(x, window=11, n_mad=5.0):
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    diff = (s - med).abs()
    mad = diff.rolling(window, center=True, min_periods=1).median()
    thr = n_mad * 1.4826 * mad.replace(0, np.nan)
    return s.where(~(diff > thr), med).values


def _zscore(x):
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    if fin.sum() < 2:
        return np.zeros_like(x)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if sd < 1e-9:
        return np.zeros_like(x)
    out = (x - mu) / sd
    out[~fin] = 0.0
    return out


def clean_gr(gr):
    x = _fill_short_gaps(np.asarray(gr, dtype=float), max_gap=5)
    x = _despike_hampel(x, 11, 5.0)
    return _zscore(x)


# =====================================================================
# G1+G2: Multi-scale NCC (vectorised FFT) — mycarta port
# =====================================================================

def _next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def _batch_ncc(windows, reference):
    n_win, L = windows.shape
    M = reference.shape[0]
    nfft = _next_pow2(M + L - 1)
    W = np.fft.rfft(windows[:, ::-1], n=nfft, axis=1)
    R = np.fft.rfft(reference, n=nfft)
    C = np.fft.irfft(W * R[None, :], n=nfft, axis=1)
    valid = C[:, L - 1: L - 1 + (M - L + 1)] / L
    return valid


def _peak_features(profile):
    if profile.size == 0 or not np.any(np.isfinite(profile)):
        return 0.0, 0, 0.0, 0.0
    pi = int(np.nanargmax(profile))
    pv = float(profile[pi])
    half = pv / 2.0
    left = pi
    while left > 0 and profile[left - 1] >= half:
        left -= 1
    right = pi
    n = profile.size
    while right < n - 1 and profile[right + 1] >= half:
        right += 1
    pw = float(right - left + 1)
    masked = profile.copy()
    lo, hi = max(0, pi - 5), min(n, pi + 6)
    masked[lo:hi] = -np.inf
    if np.any(np.isfinite(masked)) and np.max(masked) > -np.inf:
        sd = float(abs(int(np.nanargmax(masked)) - pi))
    else:
        sd = 0.0
    return pv, pi, pw, sd


def ncc_features(eval_pos, lateral_gr_clean, reference_gr_clean,
                 ref_unit_ft, prefix, ref_origin_ft, last_known_tvt,
                 ref_tvt_lookup=None, scales=NCC_SCALES,
                 search_radius_ft=SEARCH_RADIUS_FT):
    n_eval = len(eval_pos)
    M = len(reference_gr_clean)
    stat_names = ("peak", "peakidx_ft", "width", "secdist", "pred_tvt", "pred_delta")
    if M < max(scales) + 2:
        cols = [f"{prefix}_{L}_{s}" for L in scales for s in stat_names]
        return pd.DataFrame(0.0, index=pd.Index(eval_pos, name="row_idx"), columns=cols)

    out = {}
    for L in scales:
        windows = np.empty((n_eval, L), dtype=float)
        for i, p in enumerate(eval_pos):
            start = max(0, p - L + 1)
            seg = lateral_gr_clean[start: p + 1]
            if len(seg) < L:
                seg = np.concatenate([np.zeros(L - len(seg)), seg])
            windows[i] = seg
        wm = windows.mean(axis=1, keepdims=True)
        ws = windows.std(axis=1, keepdims=True)
        ws = np.where(ws < 1e-9, 1.0, ws)
        windows = (windows - wm) / ws

        if M < L:
            for s in stat_names:
                out[f"{prefix}_{L}_{s}"] = np.zeros(n_eval)
            continue

        profiles = _batch_ncc(windows, reference_gr_clean)
        n_ref = profiles.shape[1]
        if search_radius_ft > 0 and n_ref > 0:
            if ref_tvt_lookup is None:
                anchor_idx = int(round((last_known_tvt - ref_origin_ft) / max(ref_unit_ft, 1e-9)))
            else:
                anchor_idx = int(np.nanargmin(np.abs(ref_tvt_lookup - last_known_tvt)))
            rad = int(round(search_radius_ft / max(ref_unit_ft, 1e-9)))
            lo, hi = max(0, anchor_idx - rad), min(n_ref, anchor_idx + rad + 1)
            if hi > lo:
                mask = np.full(n_ref, -np.inf)
                mask[lo:hi] = 0.0
                profiles = profiles + mask[None, :]

        peaks = np.zeros(n_eval); idxs = np.zeros(n_eval)
        widths = np.zeros(n_eval); secd = np.zeros(n_eval)
        for i in range(n_eval):
            pv, pi, pw, sd = _peak_features(profiles[i])
            peaks[i] = pv; idxs[i] = pi; widths[i] = pw; secd[i] = sd

        out[f"{prefix}_{L}_peak"] = peaks
        out[f"{prefix}_{L}_peakidx_ft"] = idxs * ref_unit_ft
        out[f"{prefix}_{L}_width"] = widths
        out[f"{prefix}_{L}_secdist"] = secd
        if ref_tvt_lookup is not None:
            safe = np.clip(idxs.astype(int), 0, len(ref_tvt_lookup) - 1)
            pred = ref_tvt_lookup[safe]
        else:
            pred = ref_origin_ft + idxs * ref_unit_ft
        out[f"{prefix}_{L}_pred_tvt"] = pred
        out[f"{prefix}_{L}_pred_delta"] = pred - last_known_tvt

    return pd.DataFrame(out, index=pd.Index(eval_pos, name="row_idx"))


# =====================================================================
# Per-well aggregates: azimuth, tvt-z fits, lateral start — mycarta port
# =====================================================================

def well_aggregates(md, x, y, z, tvti, last_known_idx):
    win, thr = 50, 0.05
    lat_start = None
    for i in range(last_known_idx, win, -1):
        seg, seg_md = z[i - win:i], md[i - win:i]
        if seg_md[-1] - seg_md[0] < 1e-6:
            continue
        if abs((seg[-1] - seg[0]) / (seg_md[-1] - seg_md[0])) > thr:
            lat_start = i
            break
    if lat_start is None:
        lat_start = max(0, last_known_idx - 1000)
    lat_start = max(0, min(lat_start + 50, last_known_idx - 100))

    lm, lx, ly, lz = md[lat_start:last_known_idx + 1], x[lat_start:last_known_idx + 1], \
        y[lat_start:last_known_idx + 1], z[lat_start:last_known_idx + 1]
    if len(lm) >= 5:
        dx, dy = np.diff(lx), np.diff(ly)
        azi_deg = np.degrees(np.arctan2(dx.sum(), dy.sum())) % 360
        dmd_t = lm[-1] - lm[0]
        dz_md = (lz[-1] - lz[0]) / dmd_t if abs(dmd_t) > 1e-6 else 0.0
    else:
        azi_deg, dz_md = 0.0, 0.0

    tvt_k, z_k = tvti[:last_known_idx + 1], z[:last_known_idx + 1]
    fin = np.isfinite(tvt_k) & np.isfinite(z_k)
    if fin.sum() >= 10:
        sl_f, ic_f, _, _, _ = linregress(z_k[fin], tvt_k[fin])
    else:
        sl_f, ic_f = -1.0, 0.0
    lat_tvt, lat_z = tvt_k[lat_start:], z_k[lat_start:]
    lf = np.isfinite(lat_tvt) & np.isfinite(lat_z)
    if lf.sum() >= 200 and lat_z[lf].std() > 1.0:
        sl_l, ic_l, _, _, _ = linregress(lat_z[lf], lat_tvt[lf])
        sl_l = float(np.clip(sl_l, -2.0, 2.0))
    else:
        sl_l, ic_l = 0.0, 0.0

    return {
        "well_azimuth_deg": azi_deg,
        "well_azimuth_sin": float(np.sin(np.radians(azi_deg))),
        "well_azimuth_cos": float(np.cos(np.radians(azi_deg))),
        "dz_per_md_lateral": float(dz_md),
        "tvt_z_a": float(sl_f), "tvt_z_b": float(ic_f),
        "tvt_z_a_lat": float(sl_l), "tvt_z_b_lat": float(ic_l),
        "lat_start": int(lat_start),
        "x_center": float(np.median(x)), "y_center": float(np.median(y)),
    }


# =====================================================================
# G3+G4: anchor / trajectory — mycarta port
# =====================================================================

def anchor_traj_features(eval_pos, md, x, y, z, last_known_idx, last_known_tvt, aggs):
    md_a, z_a, x_a, y_a = md[last_known_idx], z[last_known_idx], x[last_known_idx], y[last_known_idx]
    md_td = md[-1]
    eval_len = max(md_td - md_a, 1.0)
    em, ez, ex, ey = md[eval_pos], z[eval_pos], x[eval_pos], y[eval_pos]

    win = 30
    inc = np.zeros(len(eval_pos)); inc_rate = np.zeros(len(eval_pos))
    n = len(md)
    for i, p in enumerate(eval_pos):
        s, e = max(0, p - win), min(n, p + win + 1)
        if e - s < 3:
            continue
        dz, dmd = z[e - 1] - z[s], md[e - 1] - md[s]
        if dmd <= 1e-6:
            continue
        dh = np.hypot(x[e - 1] - x[s], y[e - 1] - y[s])
        inc[i] = np.degrees(np.arctan2(dh, abs(dz) + 1e-9))
        if e - s >= 5:
            zz, mm = z[s:e], md[s:e]
            try:
                inc_rate[i] = float(np.polyfit(mm - mm.mean(), zz - zz.mean(), 2)[0])
            except np.linalg.LinAlgError:
                inc_rate[i] = 0.0

    a, b = aggs["tvt_z_a"], aggs["tvt_z_b"]
    a_lat = aggs["tvt_z_a_lat"]
    resid_anchor = last_known_tvt - (a * z_a + b)
    dz_from_a = ez - z_a
    dz_pred = a_lat * dz_from_a

    return pd.DataFrame({
        "last_known_tvt": last_known_tvt,
        "md_dist_from_anchor": em - md_a,
        "md_dist_to_td": md_td - em,
        "eval_zone_relative_pos": (em - md_a) / eval_len,
        "z": ez,
        "delta_z_from_anchor": dz_from_a,
        "delta_xy_from_anchor": np.hypot(ex - x_a, ey - y_a),
        "inclination_deg": inc,
        "inclination_rate": inc_rate,
        "tvt_minus_az_b_at_anchor": resid_anchor,
        "well_azimuth_sin": aggs["well_azimuth_sin"],
        "well_azimuth_cos": aggs["well_azimuth_cos"],
        "dz_per_md_lateral": aggs["dz_per_md_lateral"],
        "tvt_z_slope_lat": a_lat,
        "dz_pred_delta": dz_pred,
    }, index=pd.Index(eval_pos, name="row_idx"))


# =====================================================================
# G5+G7: local GR rolling stats + derivatives + affine calibration
# =====================================================================

def gr_stats_features(eval_pos, lateral_gr_raw, tvti, last_known_idx, aggs):
    s30 = pd.Series(lateral_gr_raw).rolling(30, min_periods=1)
    base = pd.DataFrame({
        "gr_local_mean": s30.mean().values,
        "gr_local_std": s30.std().values,
        "gr_local_min": s30.min().values,
        "gr_local_max": s30.max().values,
    })
    base["gr_local_range"] = base["gr_local_max"] - base["gr_local_min"]
    n = len(lateral_gr_raw)
    idx = np.arange(n, dtype=float)
    sx = pd.Series(idx).rolling(30, min_periods=2).mean()
    sxx = pd.Series(idx ** 2).rolling(30, min_periods=2).mean()
    sy = pd.Series(lateral_gr_raw).rolling(30, min_periods=2).mean()
    sxy = pd.Series(idx * lateral_gr_raw).rolling(30, min_periods=2).mean()
    slope = (sxy - sx * sy) / (sxx - sx ** 2).replace(0, np.nan)
    base["gr_local_slope"] = slope.fillna(0.0).values

    # GR derivatives (gr_d1, gr_d2)
    gr_s = pd.Series(lateral_gr_raw)
    base["gr_d1"] = gr_s.diff().fillna(0.0).values
    base["gr_d2"] = gr_s.diff().diff().fillna(0.0).values

    # multi-window rolling mean/std (grm/grs)
    for w in (5, 21, 51, 101):
        sw = gr_s.rolling(w, min_periods=1)
        base[f"grm{w}"] = sw.mean().values
        base[f"grs{w}"] = sw.std().fillna(0.0).values

    out = base.iloc[eval_pos].copy()
    out.index = pd.Index(eval_pos, name="row_idx")

    # affine GR calibration (cal_a, cal_b): fit GR_lateral_known -> TVT_input
    # 既知区間のみ使用 (leak-free). 較正係数は per-well scalar.
    gk = lateral_gr_raw[:last_known_idx + 1]
    tk = tvti[:last_known_idx + 1]
    fin = np.isfinite(gk) & np.isfinite(tk)
    if fin.sum() >= 30 and np.nanstd(gk[fin]) > 1e-6:
        ca, cb, _, _, _ = linregress(gk[fin], tk[fin])
    else:
        ca, cb = 0.0, float(np.nanmean(tk[fin])) if fin.sum() > 0 else 0.0
    out["cal_a"] = float(ca)
    out["cal_b"] = float(cb)
    return out


# =====================================================================
# G6: Q-3D tortuosity essence (Jing2022) — lean, vectorised port
#   min-curvature-derived inc/azi from XYZ -> per-portion arc/chord ratio
#   + DLS statistics. Avoids full peak-valley pipeline for speed.
# =====================================================================

def tortuosity_features(eval_pos, md, x, y, z, lat_start, portion_ft=328.0, sub_ft=25):
    n = len(md)
    feat_cols = ["tort_incline", "tort_azimuth", "tort_q3d",
                 "dls_mean", "dls_max", "dls_std", "inc_var", "azi_var"]
    if n < 10:
        return pd.DataFrame(0.0, index=pd.Index(eval_pos, name="row_idx"), columns=feat_cols)

    stride = max(1, int(sub_ft))
    sidx = np.arange(0, n, stride)
    if sidx[-1] != n - 1:
        sidx = np.append(sidx, n - 1)
    ms, xs, ys, zs = md[sidx], x[sidx], y[sidx], z[sidx]

    # Inclination & azimuth from XYZ deltas (Z negative-down).
    dx, dy, dz = np.diff(xs), np.diff(ys), np.diff(zs)
    dmd = np.diff(ms)
    dmd = np.where(dmd <= 1e-6, 1e-6, dmd)
    dh = np.hypot(dx, dy)
    inc = np.degrees(np.arctan2(dh, np.abs(dz) + 1e-9))           # deg from vertical
    azi = np.degrees(np.arctan2(dx, dy)) % 360.0
    # dogleg severity (deg per 30ft) via change in unit tangent
    incr = np.radians(inc); azir = np.radians(azi)
    tx = np.sin(incr) * np.sin(azir); ty = np.sin(incr) * np.cos(azir); tz = np.cos(incr)
    cb = np.clip(tx[1:] * tx[:-1] + ty[1:] * ty[:-1] + tz[1:] * tz[:-1], -1, 1)
    beta = np.degrees(np.arccos(cb))
    dls = beta / dmd[1:] * 30.0 if len(dmd) > 1 else np.array([0.0])

    # Horizontal cumulative distance (chord coordinate)
    hd = np.concatenate([[0.0], np.cumsum(dh)])
    tvd = zs  # vertical projection
    # lateral offset perpendicular to mean azimuth (horizontal-section)
    hz_mask = ms >= md[lat_start] if lat_start < n else np.ones(len(ms), bool)
    if hz_mask.sum() >= 2:
        ma = np.arctan2(np.mean(np.sin(np.radians(azi[hz_mask[:-1]] if len(azi) else [0]))),
                        np.mean(np.cos(np.radians(azi[hz_mask[:-1]] if len(azi) else [0]))))
    else:
        ma = 0.0
    perp_x, perp_y = -np.cos(ma), np.sin(ma)  # perp to mean azimuth in (E=x,N=y)
    lat_off = xs * perp_x + ys * perp_y

    # Per-portion arc/chord tortuosity on inclined (hd vs tvd) & azimuth (hd vs lat_off) planes.
    def _portion_tort(px, py):
        if len(px) < 3:
            return 0.0
        ratios = []
        start = 0
        for k in range(1, len(px)):
            if px[k] - px[start] >= portion_ft or k == len(px) - 1:
                seg_x, seg_y = px[start:k + 1], py[start:k + 1]
                if len(seg_x) >= 3:
                    arc = np.sum(np.hypot(np.diff(seg_x), np.diff(seg_y)))
                    chord = np.hypot(seg_x[-1] - seg_x[0], seg_y[-1] - seg_y[0])
                    if chord > 1e-6:
                        ratios.append(arc / chord - 1.0)
                start = k
        return float(np.mean(ratios)) if ratios else 0.0

    t_inc = _portion_tort(hd, tvd)
    t_azi = _portion_tort(hd, lat_off)
    t_q3d = np.hypot(t_inc, t_azi)

    scalars = {
        "tort_incline": t_inc, "tort_azimuth": t_azi, "tort_q3d": float(t_q3d),
        "dls_mean": float(np.nanmean(dls)) if dls.size else 0.0,
        "dls_max": float(np.nanmax(dls)) if dls.size else 0.0,
        "dls_std": float(np.nanstd(dls)) if dls.size else 0.0,
        "inc_var": float(np.nanvar(inc)) if inc.size else 0.0,
        "azi_var": float(np.nanvar(np.sin(np.radians(azi)))) if azi.size else 0.0,
    }
    return pd.DataFrame([scalars] * len(eval_pos),
                        index=pd.Index(eval_pos, name="row_idx"), columns=feat_cols)


# =====================================================================
# G11: segment-well decomposition — mycarta port
# =====================================================================

def segment_decomp(md, tvti, last_known_idx, lat_start):
    lm, lt = md[lat_start:last_known_idx + 1], tvti[lat_start:last_known_idx + 1]
    n = len(lm)
    res = {}
    t1, t2 = n // 3, 2 * (n // 3)
    for name, s, e in [("early", 0, t1), ("mid", t1, t2), ("late", t2, n)]:
        sm, st = lm[s:e], lt[s:e]
        fin = np.isfinite(st) & np.isfinite(sm)
        if fin.sum() >= 10 and (sm[fin].max() - sm[fin].min()) > 1.0:
            sl, ic, r, _, _ = linregress(sm[fin], st[fin])
            r2 = float(r ** 2)
            if r2 < 0.2:
                sl, ic = 0.0, float(np.nanmean(st[fin]))
        else:
            sl, ic, r2 = 0.0, (float(np.nanmean(st[fin])) if fin.sum() else 0.0), 0.0
        res[f"seg_{name}_a"] = float(sl)
        res[f"seg_{name}_b"] = float(ic)
        res[f"seg_{name}_R2"] = float(r2)
    return res


# =====================================================================
# G13: landing-zone state — mycarta port (NCC mini-match on last 50ft)
# =====================================================================

def landing_features(md, x, y, z, tvti, last_known_idx,
                     lateral_gr_clean, typewell_gr_clean, tw_tvt_min, last_known_tvt):
    lo50 = max(0, last_known_idx - 50)
    lo100 = max(0, last_known_idx - 100)
    z50, md50 = z[lo50:last_known_idx + 1], md[lo50:last_known_idx + 1]
    dmd50 = float(md50[-1] - md50[0]) if len(md50) > 1 else 1.0
    dz_dmd = float((z50[-1] - z50[0]) / max(dmd50, 1e-6)) if len(z50) > 1 else 0.0
    state = 1.0 if dz_dmd > 0.05 else (-1.0 if dz_dmd < -0.05 else 0.0)

    tvt100, md100 = tvti[lo100:last_known_idx + 1], md[lo100:last_known_idx + 1]
    f100 = np.isfinite(tvt100) & np.isfinite(md100)
    if f100.sum() >= 10 and (md100[-1] - md100[0]) > 1.0:
        sl_tvt = linregress(md100[f100], tvt100[f100])[0]
    else:
        sl_tvt = 0.0

    if len(z50) >= 2:
        dx5, dy5, dz5 = np.diff(x[lo50:last_known_idx + 1]), np.diff(y[lo50:last_known_idx + 1]), np.diff(z50)
        inc_arr = np.degrees(np.arctan2(np.hypot(dx5, dy5), np.abs(dz5) + 1e-9))
        l_inc = float(np.nanmean(inc_arr))
    else:
        l_inc = 90.0

    tvt_pre = tvti[:last_known_idx + 1]
    fpre = np.isfinite(tvt_pre)
    mean_pre = float(np.nanmean(tvt_pre[fpre])) if fpre.sum() else last_known_tvt
    l_resid = float(last_known_tvt - mean_pre)
    tvt50 = tvti[lo50:last_known_idx + 1]
    f50 = np.isfinite(tvt50)
    l_std50 = float(np.nanstd(tvt50[f50])) if f50.sum() > 1 else 0.0

    M, L = len(typewell_gr_clean), 25
    pk_tvts, pk_vals = [], []
    if M >= L and last_known_idx > lo50:
        npos = min(5, last_known_idx - lo50 + 1)
        for p in np.linspace(lo50, last_known_idx, npos, dtype=int):
            start = max(0, p - L + 1)
            seg = lateral_gr_clean[start:p + 1]
            if len(seg) < L:
                seg = np.concatenate([np.zeros(L - len(seg)), seg])
            ws = float(seg.std())
            if ws > 1e-9:
                seg = (seg - seg.mean()) / ws
            prof = _batch_ncc(seg.reshape(1, -1), typewell_gr_clean).ravel()
            if prof.size and np.any(np.isfinite(prof)):
                pk_vals.append(float(np.nanmax(prof)))
                pk_tvts.append(tw_tvt_min + int(np.nanargmax(prof)) * 0.5)
    l_cons = float(np.std(pk_tvts)) if len(pk_tvts) > 1 else 0.0
    l_mq = float(np.mean(pk_vals)) if pk_vals else 0.0

    return {
        "landing_state": state, "landing_dz_dmd": dz_dmd,
        "landing_tvt_slope_pre_ps": float(sl_tvt), "landing_inclination": l_inc,
        "landing_target_residual": l_resid, "landing_tvt_std_50": l_std50,
        "landing_ncc_peak_consistency": l_cons, "landing_recent_match_quality": l_mq,
    }


# =====================================================================
# G14: well-length geometry — mycarta port
# =====================================================================

def well_length_features(md, tvti, last_known_idx, lat_start):
    heel, ps, td = float(md[lat_start]), float(md[last_known_idx]), float(md[-1])
    tot, pre, ev = td - heel, ps - heel, td - ps
    lat_tvt = tvti[lat_start:last_known_idx + 1]
    fin = np.isfinite(lat_tvt)
    rng = float(np.nanmax(lat_tvt[fin]) - np.nanmin(lat_tvt[fin])) if fin.sum() >= 2 else 0.0
    return {
        "total_lateral_md_length": tot,
        "pre_ps_lateral_md_length": pre,
        "eval_zone_md_length": ev,
        "eval_to_known_ratio": ev / max(pre, 1.0),
        "tvt_range_pre_ps": rng,
    }


# =====================================================================
# Per-well master feature builder
# =====================================================================

def build_well_features(args):
    well_id, horiz, tw = args
    horiz = horiz.sort_values("row_idx")
    md = horiz["MD"].values.astype(float)
    x = horiz["X"].values.astype(float)
    y = horiz["Y"].values.astype(float)
    z = horiz["Z"].values.astype(float)
    gr_raw = horiz["GR"].values.astype(float)
    tvti = horiz["TVT_input"].values.astype(float)
    tvt_true = horiz["TVT"].values.astype(float)
    row_idx_arr = horiz["row_idx"].values.astype(int)

    nan_mask = np.isnan(tvti)
    if not nan_mask.any():
        return None
    eval_local = np.where(nan_mask)[0]
    last_known_idx = int(eval_local[0] - 1)
    if last_known_idx < 0:
        return None
    last_known_tvt = float(tvti[last_known_idx])

    if len(eval_local) <= 2:
        coarse = eval_local
    else:
        coarse = eval_local[::EVAL_STRIDE]
        if coarse[-1] != eval_local[-1]:
            coarse = np.concatenate([coarse, eval_local[-1:]])

    lat_clean = clean_gr(gr_raw)
    tw = tw.sort_values("row_idx")
    tw_gr_clean = clean_gr(tw["GR"].values.astype(float))
    tw_tvt = tw["TVT"].values.astype(float)
    tw_tvt_min = float(tw_tvt[0]) if len(tw_tvt) else last_known_tvt

    aggs = well_aggregates(md, x, y, z, tvti, last_known_idx)
    lat_start = aggs["lat_start"]

    dfs = []
    # G1 typewell NCC
    dfs.append(ncc_features(coarse, lat_clean, tw_gr_clean, 0.5, "ncc_tw",
                            tw_tvt_min, last_known_tvt, ref_tvt_lookup=None))
    # G2 self NCC
    pre_gr = lat_clean[:last_known_idx + 1]
    pre_tvt = tvti[:last_known_idx + 1]
    if len(pre_gr) >= 50:
        dfs.append(ncc_features(coarse, lat_clean, pre_gr, 1.0, "ncc_self",
                                0.0, last_known_tvt, ref_tvt_lookup=pre_tvt))
    else:
        sn = ("peak", "peakidx_ft", "width", "secdist", "pred_tvt", "pred_delta")
        cols = [f"ncc_self_{L}_{s}" for L in NCC_SCALES for s in sn]
        dfs.append(pd.DataFrame(0.0, index=pd.Index(coarse, name="row_idx"), columns=cols))
    # G3+G4
    dfs.append(anchor_traj_features(coarse, md, x, y, z, last_known_idx, last_known_tvt, aggs))
    # G5+G7
    dfs.append(gr_stats_features(coarse, gr_raw, tvti, last_known_idx, aggs))
    # G6 tortuosity
    dfs.append(tortuosity_features(coarse, md, x, y, z, lat_start))
    # G11 segment (per-well scalar -> broadcast)
    g11 = segment_decomp(md, tvti, last_known_idx, lat_start)
    dfs.append(pd.DataFrame([g11] * len(coarse), index=pd.Index(coarse, name="row_idx")))
    # G13 landing
    g13 = landing_features(md, x, y, z, tvti, last_known_idx, lat_clean, tw_gr_clean, tw_tvt_min, last_known_tvt)
    dfs.append(pd.DataFrame([g13] * len(coarse), index=pd.Index(coarse, name="row_idx")))
    # G14 length
    g14 = well_length_features(md, tvti, last_known_idx, lat_start)
    dfs.append(pd.DataFrame([g14] * len(coarse), index=pd.Index(coarse, name="row_idx")))

    coarse_df = pd.concat(dfs, axis=1)
    coarse_df = coarse_df.loc[:, ~coarse_df.columns.duplicated()]
    full = coarse_df.reindex(eval_local).interpolate(method="linear", limit_direction="both").ffill().bfill()
    full["well_id"] = well_id
    full["row_idx"] = row_idx_arr[eval_local]
    full["x_center"] = aggs["x_center"]
    full["y_center"] = aggs["y_center"]
    full["well_azimuth_deg"] = aggs["well_azimuth_deg"]
    full["median_tvt"] = float(np.nanmedian(tvti[:last_known_idx + 1])) \
        if np.isfinite(tvti[:last_known_idx + 1]).any() else last_known_tvt
    full["anchor"] = last_known_tvt
    full["target"] = tvt_true[eval_local] - last_known_tvt  # delta target (eval only)
    full["tvt_true"] = tvt_true[eval_local]
    return full


# =====================================================================
# CV: StratifiedGroupKFold by well (azimuth quadrant × tvt bin × xy bin)
# =====================================================================

def make_well_strat_folds(well_meta):
    from sklearn.model_selection import StratifiedGroupKFold

    def az_quad(d):
        d = d % 360.0
        return 0 if d < 90 else (1 if d < 180 else (2 if d < 270 else 3))

    wm = well_meta.copy()
    wm["az"] = wm["well_azimuth_deg"].apply(az_quad).astype(int)
    wm["tvt_bin"] = pd.qcut(wm["median_tvt"].fillna(wm["median_tvt"].median()), 3,
                            labels=False, duplicates="drop").astype(int)
    wm["x_bin"] = pd.qcut(wm["x_center"], 2, labels=False, duplicates="drop").astype(int)
    wm["y_bin"] = pd.qcut(wm["y_center"], 2, labels=False, duplicates="drop").astype(int)
    wm["xy_bin"] = wm["x_bin"] * 2 + wm["y_bin"]
    strata = wm["az"].astype(str) + "_" + wm["tvt_bin"].astype(str) + "_" + wm["xy_bin"].astype(str)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold = pd.Series(-1, index=wm.index, dtype=int)
    for k, (_, vi) in enumerate(cv.split(np.zeros(len(wm)), strata, groups=wm["well_id"])):
        fold.iloc[vi] = k
    return dict(zip(wm["well_id"], fold.values))


# =====================================================================
# LGB training over folds
# =====================================================================

LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=5.0, reg_alpha=1.0,
    n_estimators=1500, verbose=-1, n_jobs=-1, seed=RANDOM_STATE,
)


def run_cv(feat, feature_cols, fold_map, fold_name):
    import lightgbm as lgb
    feat = feat.copy()
    feat["fold"] = feat["well_id"].map(fold_map)
    feat = feat[feat["fold"] >= 0]
    oof = np.zeros(len(feat))
    fold_rmse = []
    importances = np.zeros(len(feature_cols))
    X = feat[feature_cols].astype(np.float32).values
    yt = feat["target"].values
    anchor = feat["anchor"].values
    tvt_true = feat["tvt_true"].values
    folds = feat["fold"].values
    for k in range(N_SPLITS):
        tr = folds != k
        va = folds == k
        if va.sum() == 0:
            continue
        m = lgb.LGBMRegressor(**LGB_PARAMS)
        m.fit(X[tr], yt[tr],
              eval_set=[(X[va], yt[va])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        pred_delta = m.predict(X[va])
        oof[va] = pred_delta
        pred_tvt = anchor[va] + pred_delta
        rmse = float(np.sqrt(np.mean((pred_tvt - tvt_true[va]) ** 2)))
        fold_rmse.append(rmse)
        importances += m.feature_importances_
    pred_tvt_all = anchor + oof
    cv_rmse = float(np.sqrt(np.mean((pred_tvt_all - tvt_true) ** 2)))
    feat[f"pred_delta_{fold_name}"] = oof
    feat[f"pred_tvt_{fold_name}"] = pred_tvt_all
    imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    return cv_rmse, fold_rmse, imp, feat[["well_id", "row_idx", f"pred_tvt_{fold_name}", "tvt_true", "anchor"]]


# =====================================================================
# Main
# =====================================================================

def attach_pf_features(feat):
    """exp022 / exp100 PF予測を特徴として join. test側は exp022 のみ存在."""
    out = feat.copy()
    try:
        pf22 = pd.read_csv(EXP_DIR.parent / "exp022_particle_filter" / "oof.csv",
                           usecols=["well_id", "row_idx", "pred_tvt"])
        pf22 = pf22.rename(columns={"pred_tvt": "pf22_pred_tvt"})
        out = out.merge(pf22, on=["well_id", "row_idx"], how="left")
    except Exception as e:
        print(f"[warn] exp022 PF join failed: {e}")
        out["pf22_pred_tvt"] = np.nan
    try:
        pf100 = pd.read_csv(EXP_DIR.parent / "exp100_soft_likpf" / "oof.csv",
                            usecols=["well_id", "row_idx", "pred_tvt"])
        pf100 = pf100.rename(columns={"pred_tvt": "pf100_pred_tvt"})
        out = out.merge(pf100, on=["well_id", "row_idx"], how="left")
    except Exception as e:
        print(f"[warn] exp100 PF join failed: {e}")
        out["pf100_pred_tvt"] = np.nan
    # fallbacks
    out["pf22_pred_tvt"] = out["pf22_pred_tvt"].fillna(out["anchor"])
    out["pf100_pred_tvt"] = out["pf100_pred_tvt"].fillna(out["pf22_pred_tvt"])
    out["pf22_delta"] = out["pf22_pred_tvt"] - out["anchor"]
    out["pf100_delta"] = out["pf100_pred_tvt"] - out["anchor"]
    out["pf_mean_delta"] = 0.5 * (out["pf22_delta"] + out["pf100_delta"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help=">=40 well subset")
    ap.add_argument("--n-smoke", type=int, default=48)
    ap.add_argument("--rebuild", action="store_true", help="ignore feature cache")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[exp105] start smoke={args.smoke}")

    train = pd.read_parquet(PROC / "train_base_v001.parquet",
                            columns=["well_id", "row_idx", "MD", "X", "Y", "Z", "GR",
                                     "TVT_input", "TVT", "is_target"])
    tw_all = pd.read_parquet(PROC / "typewell_train_base_v001.parquet",
                             columns=["well_id", "row_idx", "TVT", "GR"])

    wells = sorted(train["well_id"].unique())
    if args.smoke:
        wells = wells[:max(40, args.n_smoke)]
        train = train[train["well_id"].isin(wells)]
        tw_all = tw_all[tw_all["well_id"].isin(wells)]
        print(f"[smoke] {len(wells)} wells")

    cache_ok = CACHE.exists() and not args.rebuild and not args.smoke
    if cache_ok:
        print(f"[cache] loading {CACHE}")
        feat = pd.read_parquet(CACHE)
    else:
        tw_grp = {w: g for w, g in tw_all.groupby("well_id")}
        tasks = []
        for w, g in train.groupby("well_id"):
            if w in tw_grp:
                tasks.append((w, g, tw_grp[w]))
        print(f"[build] {len(tasks)} wells, workers={args.workers}")
        results = []
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(build_well_features, t) for t in tasks]
            for fu in as_completed(futs):
                r = fu.result()
                if r is not None:
                    results.append(r)
                done += 1
                if done % 100 == 0:
                    print(f"  [build] {done}/{len(tasks)} wells  elapsed={time.time()-t0:.0f}s")
        feat = pd.concat(results, ignore_index=True)
        print(f"[build] done {feat.shape} in {time.time()-t0:.0f}s")
        if not args.smoke:
            feat.to_parquet(CACHE)
            print(f"[cache] saved {CACHE}")

    # attach PF predictions
    feat = attach_pf_features(feat)

    # feature columns = everything except identity / target / leak columns
    drop = {"well_id", "row_idx", "target", "tvt_true", "anchor",
            "x_center", "y_center", "well_azimuth_deg", "median_tvt",
            "pf22_pred_tvt", "pf100_pred_tvt"}
    feature_cols = [c for c in feat.columns if c not in drop]
    feature_cols = [c for c in feature_cols if feat[c].dtype != object]
    # drop all-NaN / constant
    keep = []
    for c in feature_cols:
        col = feat[c]
        if col.notna().sum() == 0:
            continue
        keep.append(c)
    feature_cols = keep
    feat[feature_cols] = feat[feature_cols].replace([np.inf, -np.inf], np.nan)
    print(f"[features] n_features={len(feature_cols)}  rows={len(feat)}")

    # build folds
    well_meta = feat.groupby("well_id").agg(
        well_azimuth_deg=("well_azimuth_deg", "first"),
        median_tvt=("median_tvt", "first"),
        x_center=("x_center", "first"),
        y_center=("y_center", "first"),
    ).reset_index()
    well_fold_map = make_well_strat_folds(well_meta)

    # typewell fold (honest)
    tw_fold = pd.read_csv(FOLDS / "folds_group_typewell_v001.csv")
    tw_fold_map = dict(zip(tw_fold["well_id"], tw_fold["fold"]))

    # CV runs
    print("[cv] well-fold (StratifiedGroupKFold)")
    cv_well, fr_well, imp, oof_well = run_cv(feat, feature_cols, well_fold_map, "wellfold")
    print(f"  well-fold CV RMSE = {cv_well:.4f}  folds={[round(r,3) for r in fr_well]}")

    print("[cv] typewell-fold (honest)")
    cv_tw, fr_tw, imp_tw, oof_tw = run_cv(feat, feature_cols, tw_fold_map, "twfold")
    print(f"  typewell-fold CV RMSE = {cv_tw:.4f}  folds={[round(r,3) for r in fr_tw]}")

    # ---- Ablation: drop a feature group, measure typewell-fold CV ----
    groups = {
        "ncc_tw": [c for c in feature_cols if c.startswith("ncc_tw")],
        "ncc_self": [c for c in feature_cols if c.startswith("ncc_self")],
        "tortuosity": [c for c in feature_cols if c.startswith(("tort_", "dls_", "inc_var", "azi_var"))],
        "landing": [c for c in feature_cols if c.startswith("landing_")],
        "calib": [c for c in feature_cols if c.startswith("cal_")],
        "gr_roll": [c for c in feature_cols if c.startswith(("grm", "grs", "gr_local", "gr_d"))],
        "pf": [c for c in feature_cols if c.startswith("pf")],
        "trajectory": [c for c in feature_cols if c in (
            "z", "delta_z_from_anchor", "delta_xy_from_anchor", "inclination_deg",
            "inclination_rate", "well_azimuth_sin", "well_azimuth_cos",
            "dz_per_md_lateral", "tvt_z_slope_lat", "dz_pred_delta")],
        "segment": [c for c in feature_cols if c.startswith("seg_")],
    }
    ablation = {}
    if not args.smoke:
        for gname, gcols in groups.items():
            sub = [c for c in feature_cols if c not in set(gcols)]
            if not gcols:
                continue
            cv_a, _, _, _ = run_cv(feat, sub, tw_fold_map, f"abl_{gname}")
            ablation[gname] = {"cv_typewell_without": round(cv_a, 4),
                               "delta_vs_full": round(cv_a - cv_tw, 4)}
            print(f"  [ablation] drop {gname:12s} -> typewell CV {cv_a:.4f}  (Δ {cv_a-cv_tw:+.4f})")
    else:
        # smoke: only run the 3 highest-value ablations to keep fast
        for gname in ("ncc_tw", "pf", "trajectory"):
            gcols = groups[gname]
            sub = [c for c in feature_cols if c not in set(gcols)]
            cv_a, _, _, _ = run_cv(feat, sub, tw_fold_map, f"abl_{gname}")
            ablation[gname] = {"cv_typewell_without": round(cv_a, 4),
                               "delta_vs_full": round(cv_a - cv_tw, 4)}

    # ---- OOF output (use well-fold OOF as primary) ----
    oof_out = oof_well.rename(columns={"pred_tvt_wellfold": "pred_tvt"})
    oof_out = oof_out.merge(
        oof_tw.rename(columns={"pred_tvt_twfold": "pred_tvt_twfold"})[["well_id", "row_idx", "pred_tvt_twfold"]],
        on=["well_id", "row_idx"], how="left")
    oof_out["error"] = oof_out["pred_tvt"] - oof_out["tvt_true"]
    oof_out["abs_error"] = oof_out["error"].abs()
    oof_out.to_csv(EXP_DIR / "oof.csv", index=False)

    # ---- importance ----
    imp_top = {k: round(float(v), 1) for k, v in imp.head(20).items()}

    runtime = time.time() - t0
    result = {
        "exp_id": "exp105_rich_feature_lgb",
        "smoke": bool(args.smoke),
        "n_wells": int(feat["well_id"].nunique()),
        "n_rows": int(len(feat)),
        "n_features": int(len(feature_cols)),
        "cv_rmse_wellfold": round(cv_well, 4),
        "cv_rmse_typewellfold": round(cv_tw, 4),
        "fold_rmse_wellfold": [round(r, 4) for r in fr_well],
        "fold_rmse_typewellfold": [round(r, 4) for r in fr_tw],
        "feature_importance_top20": imp_top,
        "ablation": ablation,
        "compare": {"exp104": 9.887, "exp022": 11.024, "public_clean_lb": 7.297},
        "beat_exp104_typewell": bool(cv_tw < 9.887),
        "beat_exp104_well": bool(cv_well < 9.887),
        "leak_risk": "low",
        "leak_notes": ("特徴=typewell GR-TVTプロファイル + 既知GR/幾何/anchor + 我々のPF予測のみ。"
                       "horizontal well Geology列不使用。較正(cal_a,cal_b)はTVT_input(既知)のみ。"
                       "真TVTは評価のみ。offset prior系はfold leak回避のため per-well特徴に限定。"
                       "PF OOFはleak-free(exp022/100)。fold割当はwell単位のみ。"),
        "runtime_sec": round(runtime, 1),
        "notes_short": (f"rich domain features (NCC/tortuosity/landing/calib/PF) LGB delta. "
                        f"well-fold={cv_well:.3f} typewell-fold={cv_tw:.3f} "
                        f"vs exp104=9.887 ({'BEAT' if cv_tw < 9.887 else 'miss'})"),
    }
    with open(EXP_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[done] result.json written. runtime={runtime:.0f}s")
    print(json.dumps({k: result[k] for k in
                      ("cv_rmse_wellfold", "cv_rmse_typewellfold", "n_features",
                       "beat_exp104_typewell", "runtime_sec")}, indent=1))
    return result


if __name__ == "__main__":
    main()
