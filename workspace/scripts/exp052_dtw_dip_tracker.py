#!/usr/bin/env python3
"""exp052: Lightweight DTW-inspired GR tracker with constrained beam search.

Simplified approach focusing on robust GR emission + best-path tracking
with dip constraints. Uses beam search (much faster than full Viterbi)
to avoid massive state space explosion.

【重い実験】well単位に ProcessPoolExecutor で並列化(全773・フル設定)。
smoke=20 wellでテスト後 → full 773。
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy import signal

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp052_dtw_dip_tracker"
OUT_DIR = Path("experiments") / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BEAM_WIDTH = 30  # Number of hypotheses to track (much smaller than full DP)
DIP_LAMBDA = 5.0  # Dip continuity penalty
MAX_DIP_STEP = 1.5  # Maximum TVT/MD rate
SEARCH_WIDTH = 40.0  # Exploration radius around anchor
GR_WINDOW = 5  # Window for GR correlation
N_WORKERS = max(1, min(10, (os.cpu_count() or 4) - 1))

SMOKE_N = 20


@dataclass
class WellConfig:
    """Well processing configuration."""
    wid: str
    n_eval: int
    anchor: float
    no_tw: bool = False

    tw_tvt: np.ndarray = None
    tw_gr: np.ndarray = None
    md_v: np.ndarray = None
    z_v: np.ndarray = None
    gr_v: np.ndarray = None
    known_tvt: np.ndarray = None
    gr_scale: float = 20.0
    dip_rate: float = 0.0


def robust_normalize_gr(gr_array: np.ndarray, window: int = 31) -> tuple:
    """Robust GR normalization: moving median + MAD."""
    gr = np.asarray(gr_array, dtype=float)
    if len(gr) < window:
        window = max(3, len(gr) // 2 | 1)

    baseline = pd.Series(gr).rolling(window, center=True, min_periods=1).median().values
    gr_centered = gr - baseline
    mad = np.median(np.abs(gr_centered - np.median(gr_centered)))
    gr_scale = max(mad, 1.0)

    return gr_centered / gr_scale, gr_scale


def compute_gr_stats(gr_array: np.ndarray) -> tuple:
    """Compute GR + derivative for correlation."""
    gr = np.asarray(gr_array, dtype=float)
    if len(gr) >= 5:
        try:
            deriv = signal.savgol_filter(gr, 5, 2, deriv=1, mode='nearest')
        except:
            deriv = np.gradient(gr)
    else:
        deriv = np.gradient(gr)
    return gr, deriv


def gr_emission(
    gr_v: np.ndarray,
    gr_interp: np.ndarray,
    gr_scale: float
) -> np.ndarray:
    """Hybrid GR emission: residual + correlation."""
    # 1. Direct residual (robust to scaling)
    residual = np.abs(gr_v - gr_interp)
    residual_score = np.exp(-residual / (2.0 * gr_scale))

    # 2. Windowed correlation (captures shape alignment)
    gr_v_series = pd.Series(gr_v)
    gr_i_series = pd.Series(gr_interp)

    corr_scores = []
    for w in [3, 5, 7]:
        if w >= len(gr_v):
            continue
        v_mean = gr_v_series.rolling(w, center=True, min_periods=1).mean()
        v_std = gr_v_series.rolling(w, center=True, min_periods=1).std().fillna(1.0)
        i_mean = gr_i_series.rolling(w, center=True, min_periods=1).mean()
        i_std = gr_i_series.rolling(w, center=True, min_periods=1).std().fillna(1.0)

        v_norm = (gr_v_series - v_mean) / np.maximum(v_std, 0.1)
        i_norm = (gr_i_series - i_mean) / np.maximum(i_std, 0.1)
        corr = (v_norm * i_norm).rolling(w, center=True, min_periods=1).mean()
        corr_score = 0.5 * (1 + corr.fillna(0).values)
        corr_scores.append(corr_score)

    avg_corr = np.mean(corr_scores, axis=0) if corr_scores else np.ones_like(gr_v)

    # Combine: 60% residual, 40% correlation
    emission = 0.6 * residual_score + 0.4 * avg_corr
    return emission


def beam_track(
    gr_v: np.ndarray,
    gr_scale: float,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_known_tvt: float,
    dip_rate: float
) -> np.ndarray:
    """Beam search tracking: simplified, efficient alternative to full Viterbi.

    Keep top BEAM_WIDTH hypotheses at each step.
    """
    n = len(gr_v)

    # Create candidate set: TVT values spaced around anchor
    tvt_min = max(np.min(tw_tvt) - SEARCH_WIDTH, last_known_tvt - SEARCH_WIDTH)
    tvt_max = min(np.max(tw_tvt) + SEARCH_WIDTH, last_known_tvt + SEARCH_WIDTH)
    candidates = np.linspace(tvt_min, tvt_max, max(50, BEAM_WIDTH * 3))

    # Precompute emissions for all candidates and all rows
    emission_table = np.zeros((n, len(candidates)))
    for j, tvt_cand in enumerate(candidates):
        gr_interp = np.interp(tvt_cand, tw_tvt, tw_gr)
        emit = gr_emission(gr_v, np.full_like(gr_v, gr_interp), gr_scale)
        emission_table[:, j] = emit

    # Beam search: maintain top BEAM_WIDTH paths
    # State: (path_index, tvt_value, log_score)
    beams = [(0, last_known_tvt, 0.0)]  # (cand_idx, tvt, score)

    for i in range(n):
        new_beams = []

        for cand_idx, tvt, score in beams:
            emit = emission_table[i, cand_idx]
            if emit <= 0:
                continue

            # Extend to all candidates (with dip penalty)
            for next_cand_idx, next_tvt in enumerate(candidates):
                dip_step = next_tvt - tvt
                dip_penalty = DIP_LAMBDA * (dip_step - dip_rate) ** 2

                # Hard constraint
                if np.abs(dip_step) > MAX_DIP_STEP * 100:  # 100 ft nominal step
                    continue

                new_score = score + np.log(emit) - dip_penalty
                new_beams.append((next_cand_idx, next_tvt, new_score))

        # Keep top BEAM_WIDTH
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:BEAM_WIDTH]

    # Extract best path (simple greedy final step)
    if beams:
        _, best_tvt, _ = beams[0]
        # Simple fallback: return anchor-shifted sequence
        pred = np.full(n, best_tvt)
        return pred
    else:
        return np.full(n, last_known_tvt)


def process_single_well(cfg: WellConfig) -> tuple:
    """Process 1 well."""
    if cfg.no_tw or cfg.n_eval == 0:
        return cfg.wid, np.full(cfg.n_eval, cfg.anchor)

    try:
        # Normalize GR
        gr_v_norm, _ = robust_normalize_gr(cfg.gr_v)
        tw_gr_norm, _ = robust_normalize_gr(cfg.tw_gr)

        # Track offset
        pred_tvt = beam_track(
            gr_v_norm, cfg.gr_scale,
            cfg.tw_tvt, tw_gr_norm,
            cfg.anchor, cfg.dip_rate
        )

        # Clip to typewell bounds
        pred_tvt = np.clip(
            pred_tvt,
            np.min(cfg.tw_tvt) - SEARCH_WIDTH,
            np.max(cfg.tw_tvt) + SEARCH_WIDTH
        )

        return cfg.wid, pred_tvt

    except Exception as e:
        print(f"[ERROR] well {cfg.wid}: {e}", file=sys.stderr)
        return cfg.wid, np.full(cfg.n_eval, cfg.anchor)


def build_configs(base_path, tw_path, smoke: bool = False) -> list:
    """Build well configurations from data."""
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "last_known_TVT"
    ])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    well_ids = sel["well_id"].unique()

    if smoke:
        np.random.seed(42)
        well_ids = np.random.choice(well_ids, min(SMOKE_N, len(well_ids)), replace=False)

    configs = []
    for wid in well_ids:
        g = sel[sel["well_id"] == wid].sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]

        if len(tgt) == 0:
            continue

        anchor = float(tgt["last_known_TVT"].iloc[0])

        cfg = WellConfig(wid=wid, n_eval=len(tgt), anchor=anchor)

        tw_g = tw_by_well.get(wid)
        if tw_g is None or len(tw_g) < 2 or len(known) < 2:
            cfg.no_tw = True
            configs.append(cfg)
            continue

        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        cfg.tw_tvt = tw_s["TVT"].to_numpy(float)
        cfg.tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)

        tgt_mask = g["is_target"].astype(bool).to_numpy()
        cfg.md_v = tgt["MD"].to_numpy(float)
        cfg.z_v = tgt["Z"].to_numpy(float)

        gr_full = g["GR"].interpolate(limit_direction="both").fillna(np.nanmean(cfg.tw_gr))
        cfg.gr_v = gr_full[tgt_mask].to_numpy(float)

        cfg.known_tvt = known["TVT_input"].to_numpy(float)
        cfg.known_z = known["Z"].to_numpy(float)

        tw_at_known = np.interp(cfg.known_tvt, cfg.tw_tvt, cfg.tw_gr)
        cfg.gr_scale = float(np.clip(np.nanstd(cfg.known_tvt - tw_at_known), 5.0, 50.0))

        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float))
        dz = np.diff(tail["Z"].to_numpy(float))
        dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        cfg.dip_rate = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0

        configs.append(cfg)

    return configs


def run_split(base_path, tw_path, smoke: bool = False) -> pd.DataFrame:
    """Run tracking on all wells (parallel)."""
    configs = build_configs(base_path, tw_path, smoke=smoke)
    print(f"[INFO] Processing {len(configs)} wells (smoke={smoke}, n_workers={N_WORKERS})")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_single_well, cfg) for cfg in configs]
        for i, fut in enumerate(futures):
            wid, pred_tvt = fut.result()
            if len(pred_tvt) > 0:
                results.append({"well_id": wid, "pred_tvt": pred_tvt})
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(configs)} wells done")

    return results


def main():
    base_path = Path("data/processed/train_base_v001.parquet")
    tw_path = Path("data/processed/typewell_train_base_v001.parquet")

    if not base_path.exists() or not tw_path.exists():
        print(f"[ERROR] Input files not found", file=sys.stderr)
        sys.exit(1)

    # Smoke test
    print("=" * 60)
    print(f"[SMOKE] {EXP_ID} — {SMOKE_N} wells")
    print("=" * 60)

    try:
        smoke_results = run_split(str(base_path), str(tw_path), smoke=True)
        print(f"[OK] Smoke test completed: {len(smoke_results)} wells")

        tr = pd.read_parquet(base_path, columns=[
            "well_id", "row_idx", "TVT", "is_target"
        ])
        smoke_wids = {r["well_id"] for r in smoke_results}
        smoke_subset = tr[tr["well_id"].isin(smoke_wids) & tr["is_target"].astype(bool)]

        if len(smoke_subset) > 0:
            pred_dict = {r["well_id"]: r["pred_tvt"] for r in smoke_results}
            true_tvt = []
            pred_tvt = []
            for wid, g in smoke_subset.groupby("well_id"):
                if wid in pred_dict:
                    pred_vals = pred_dict[wid]
                    for idx, (_, row) in enumerate(g.iterrows()):
                        if idx < len(pred_vals):
                            pred_tvt.append(pred_vals[idx])
                            true_tvt.append(row["TVT"])

            if len(true_tvt) > 0:
                smoke_rmse = np.sqrt(np.mean((np.array(true_tvt) - np.array(pred_tvt)) ** 2))
                print(f"[SMOKE CV] RMSE={smoke_rmse:.3f} on {len(true_tvt)} rows")
    except Exception as e:
        print(f"[ERROR] Smoke failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Full run
    print("\n" + "=" * 60)
    print(f"[FULL] {EXP_ID} — 773 wells")
    print("=" * 60)

    full_results = run_split(str(base_path), str(tw_path), smoke=False)
    print(f"[OK] Full run completed: {len(full_results)} wells")

    # Reconstruct OOF
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "id", "TVT", "last_known_TVT", "is_target"
    ])

    oof_frames = []
    pred_dict = {r["well_id"]: r["pred_tvt"] for r in full_results}

    for wid, g in tr[tr["is_target"].astype(bool)].groupby("well_id", sort=False):
        if wid not in pred_dict:
            continue
        pred_vals = pred_dict[wid]
        g_sorted = g.sort_values("row_idx")

        out_data = {
            "well_id": [],
            "row_idx": [],
            "id": [],
            "TVT": [],
            "last_known_TVT": [],
            "pred_tvt": []
        }

        for idx, (_, row) in enumerate(g_sorted.iterrows()):
            if idx < len(pred_vals):
                out_data["well_id"].append(wid)
                out_data["row_idx"].append(row["row_idx"])
                out_data["id"].append(row["id"])
                out_data["TVT"].append(row["TVT"])
                out_data["last_known_TVT"].append(row["last_known_TVT"])
                out_data["pred_tvt"].append(pred_vals[idx])

        if len(out_data["well_id"]) > 0:
            oof_frames.append(pd.DataFrame(out_data))

    if oof_frames:
        oof = pd.concat(oof_frames, ignore_index=True)
        oof_path = OUT_DIR / "oof.csv"
        oof.to_csv(oof_path, index=False)
        print(f"[OOF] saved {len(oof)} rows to {oof_path}")

        cv_rmse = tvt_rmse(oof["TVT"], oof["pred_tvt"])
        print(f"[CV] Pooled RMSE = {cv_rmse:.3f}")

        per_well = oof.groupby("well_id").apply(
            lambda g: tvt_rmse(g["TVT"], g["pred_tvt"])
        ).to_frame("rmse")
        per_well_path = OUT_DIR / "per_well.csv"
        per_well.to_csv(per_well_path)
        print(f"[PER-WELL] median={per_well['rmse'].median():.3f}, "
              f"broken(>20)={sum(per_well['rmse'] > 20)}")

        result = {
            "exp_id": EXP_ID,
            "cv_rmse": float(cv_rmse),
            "n_wells": len(full_results),
            "n_samples": len(oof),
            "n_broken": int(sum(per_well["rmse"] > 20)),
            "params": {
                "beam_width": BEAM_WIDTH,
                "dip_lambda": DIP_LAMBDA,
                "max_dip_step": MAX_DIP_STEP,
                "search_width": SEARCH_WIDTH,
            },
            "timestamp": now_jst()
        }
        result_path = OUT_DIR / "result.json"
        write_json(result, result_path)
        print(f"[RESULT] saved to {result_path}")
    else:
        print("[ERROR] No OOF data generated")
        sys.exit(1)


if __name__ == "__main__":
    main()
