"""exp073 Task B: Particle Filter OOF for all 773 train wells.

Applies the PF implementation from _decoded_exp026_b64.py to all train wells,
producing hidden-row TVT predictions without using hidden TVT (leak-free).

PF is well-independent - no folds needed.
Uses: known TVT_input + GR + typewell GR-TVT profile only.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII")
TRAIN_RAW_DIR = REPO / "data/raw/train"
TRAIN_BASE_PATH = REPO / "data/processed/train_base_v001.parquet"
OUT_DIR = REPO / "experiments/exp073_public_assets_integration"
OOF_PARTIAL_PATH = OUT_DIR / "oof_pf_partial.csv"
OOF_PATH = OUT_DIR / "oof_pf.csv"
RESULT_PATH = OUT_DIR / "result_pf.json"
PROGRESS_LOG = OUT_DIR / "pf_progress.log"
ERROR_LOG = OUT_DIR / "error_pf.log"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PF tuned config (identical to exp026) ─────────────────────────────────────
PF_SEEDS = 128
PF_PARTICLES = 500
PF_SCALE = 8.0
PF_INIT_SPREAD = 4.0
PF_PN = 0.01
PF_VN = 0.002
PF_MOM = 0.998
PF_RP = 0.1
PF_RR = 0.001
PF_RESAMP = 0.5

MAX_WORKERS = 12

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler(str(PROGRESS_LOG), mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── PF core (identical to exp026 source) ──────────────────────────────────────
def pf_single(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed):
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = PF_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    prev_MD = last_MD
    log_lik = 0.0
    lo = tw_tvt[0] - 100
    hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PF_PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.0)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < PF_RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(N)
            rate = rate[idx] + PF_RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    return res, log_lik


def pf_worker(p: dict):
    """Run PF for one well: 128 seed likelihood-weighted ensemble."""
    wid = p["wid"]
    n = p["n"]
    row_idxs = p["row_idxs"]
    if n == 0:
        return wid, np.zeros(0), row_idxs
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"]), row_idxs
    preds = np.empty((PF_SEEDS, n))
    liks = np.empty(PF_SEEDS)
    for s in range(PF_SEEDS):
        preds[s], liks[s] = pf_single(
            p["tw_tvt"], p["tw_gr"],
            p["md_v"], p["z_v"], p["gr_v"],
            p["gs"], p["ir"],
            p["last_tvt"], p["last_Z"], p["last_MD"],
            seed=s,
        )
    wts = np.exp((liks - liks.max()) / PF_SCALE)
    wts /= wts.sum()
    pred = (wts[:, None] * preds).sum(0)
    return wid, pred, row_idxs


def build_payload(well_id: str, well_df: pd.DataFrame) -> dict:
    """Build PF input payload for one train well (hidden rows only).

    LEAK-FREE: Only uses TVT_input (known rows), GR (all rows), and typewell.
    Hidden TVT is NOT used anywhere in payload construction.
    """
    g = well_df.sort_values("row_idx").reset_index(drop=True)
    known = g[g["is_known_tvt"].astype(bool)]
    tgt = g[g["is_target"].astype(bool)]  # hidden rows
    n = len(tgt)
    row_idxs = tgt["row_idx"].to_numpy(dtype=np.int64)

    anchor = float(tgt["last_known_TVT"].iloc[0]) if n > 0 else 0.0

    # Load typewell
    tw_path = TRAIN_RAW_DIR / f"{well_id}__typewell.csv"
    if not tw_path.exists():
        return {"wid": well_id, "n": n, "no_tw": True, "anchor": anchor, "row_idxs": row_idxs}

    tw_raw = pd.read_csv(tw_path)
    tw = tw_raw[["TVT", "GR"]].copy()

    if len(tw) < 2 or len(known) < 2 or n == 0:
        return {"wid": well_id, "n": n, "no_tw": True, "anchor": anchor, "row_idxs": row_idxs}

    tw_s = tw.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)

    # GR interpolation over all rows (no hidden TVT used here)
    gr_full = g["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    tgt_mask = g["is_target"].astype(bool).to_numpy()
    gr_v = gr_full[tgt_mask]

    # GR sigma from known rows
    k_tvt = known["TVT_input"].to_numpy(float)
    k_gr = known["GR"].fillna(0).to_numpy(float)
    gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10.0, 60.0))

    # Initial rate from known tail
    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0
    ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0

    last = known.iloc[-1]

    return {
        "wid": well_id,
        "n": n,
        "no_tw": False,
        "anchor": anchor,
        "row_idxs": row_idxs,
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
        "md_v": tgt["MD"].to_numpy(float),
        "z_v": tgt["Z"].to_numpy(float),
        "gr_v": gr_v,
        "gs": gs,
        "ir": ir,
        "last_tvt": float(last["TVT_input"]),
        "last_Z": float(last["Z"]),
        "last_MD": float(last["MD"]),
    }


def main():
    t0 = time.time()
    logger.info("Loading train_base...")
    train = pd.read_parquet(TRAIN_BASE_PATH)

    # hidden rows reference for validation
    hidden_ref = train[train["is_target"].astype(bool)][["well_id", "row_idx", "TVT"]].copy()
    hidden_ref["id"] = hidden_ref["well_id"] + "_" + hidden_ref["row_idx"].astype(str)
    logger.info(f"Hidden rows expected: {len(hidden_ref)}")

    # Build payloads per well (groupby avoids O(n^2) repeated filtering)
    logger.info("Building PF payloads...")
    payloads = []
    for wid, wdf in train.groupby("well_id", sort=False):
        payload = build_payload(wid, wdf)
        payloads.append(payload)
    logger.info(f"Payloads ready: {len(payloads)} wells")

    # Run PF in parallel
    results = {}  # wid -> (pred, row_idxs)
    done_count = 0
    n_total = len(payloads)

    logger.info(f"Starting PF with {MAX_WORKERS} workers, {PF_SEEDS} seeds x {PF_PARTICLES} particles")

    checkpoint_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_wid = {ex.submit(pf_worker, p): p["wid"] for p in payloads}
        for future in as_completed(future_to_wid):
            wid = future_to_wid[future]
            try:
                wid_ret, pred, row_idxs = future.result()
                results[wid_ret] = (pred, row_idxs)
                done_count += 1

                # Accumulate checkpoint rows
                for ri, pv in zip(row_idxs, pred):
                    checkpoint_rows.append((wid_ret, int(ri), float(pv)))

                if done_count % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done_count / elapsed
                    eta = (n_total - done_count) / rate if rate > 0 else 0
                    logger.info(
                        f"Progress: {done_count}/{n_total} wells | "
                        f"elapsed={elapsed/60:.1f}min | ETA={eta/60:.1f}min"
                    )

                if done_count % 100 == 0:
                    # Checkpoint save
                    chk = pd.DataFrame(checkpoint_rows, columns=["well_id", "row_idx", "pred_tvt"])
                    chk.to_csv(OOF_PARTIAL_PATH, index=False)
                    logger.info(f"Checkpoint saved: {len(chk)} rows")

            except Exception as e:
                logger.error(f"ERROR well {wid}: {e}")
                # Use anchor fallback
                p_match = next((p for p in payloads if p["wid"] == wid), None)
                if p_match:
                    results[wid] = (np.full(p_match["n"], p_match["anchor"]), p_match["row_idxs"])

    logger.info(f"All wells done. Building OOF dataframe...")

    # Assemble OOF
    oof_rows = []
    for wid, (pred, row_idxs) in results.items():
        for ri, pv in zip(row_idxs, pred):
            oof_rows.append({
                "well_id": wid,
                "row_idx": int(ri),
                "pred_tvt": float(pv),
            })

    oof = pd.DataFrame(oof_rows)
    oof["id"] = oof["well_id"] + "_" + oof["row_idx"].astype(str)

    # Merge with true TVT
    oof = oof.merge(
        hidden_ref[["id", "TVT"]].rename(columns={"TVT": "tvt_true"}),
        on="id", how="left"
    )
    oof = oof[["id", "well_id", "row_idx", "tvt_true", "pred_tvt"]]
    oof = oof.sort_values(["well_id", "row_idx"]).reset_index(drop=True)

    n_rows = len(oof)
    logger.info(f"OOF rows: {n_rows} (expected 3,783,989)")

    # Compute pooled RMSE
    valid = oof["tvt_true"].notna() & oof["pred_tvt"].notna()
    se = (oof.loc[valid, "tvt_true"] - oof.loc[valid, "pred_tvt"]) ** 2
    pooled_rmse = float(np.sqrt(se.mean()))
    logger.info(f"Pooled RMSE: {pooled_rmse:.4f}")

    # Per-well RMSE
    def well_rmse(g):
        v = g["tvt_true"].notna() & g["pred_tvt"].notna()
        if v.sum() == 0:
            return np.nan
        return float(np.sqrt(((g.loc[v, "tvt_true"] - g.loc[v, "pred_tvt"]) ** 2).mean()))

    per_well = oof.groupby("well_id").apply(well_rmse)
    per_well_stats = {
        "mean": float(per_well.mean()),
        "median": float(per_well.median()),
        "std": float(per_well.std()),
        "min": float(per_well.min()),
        "max": float(per_well.max()),
        "n_wells": int(per_well.notna().sum()),
    }
    logger.info(f"Per-well RMSE: mean={per_well_stats['mean']:.4f} median={per_well_stats['median']:.4f}")

    runtime_min = (time.time() - t0) / 60.0

    # Save OOF
    oof.to_csv(OOF_PATH, index=False)
    logger.info(f"OOF saved: {OOF_PATH}")

    # Save result json
    result = {
        "exp": "exp073_task_b_pf_oof",
        "pooled_rmse": pooled_rmse,
        "per_well_stats": per_well_stats,
        "n_rows": n_rows,
        "n_wells": len(results),
        "runtime_min": runtime_min,
        "pf_config": {
            "PF_SEEDS": PF_SEEDS,
            "PF_PARTICLES": PF_PARTICLES,
            "PF_SCALE": PF_SCALE,
            "PF_INIT_SPREAD": PF_INIT_SPREAD,
            "PF_PN": PF_PN,
            "PF_VN": PF_VN,
            "PF_MOM": PF_MOM,
            "PF_RP": PF_RP,
            "PF_RR": PF_RR,
            "PF_RESAMP": PF_RESAMP,
            "MAX_WORKERS": MAX_WORKERS,
        },
        "leak_notes": (
            "LEAK-FREE: hidden rows TVT (TVT column in train_base) is NOT used in payload construction. "
            "pf_build_payload only reads: TVT_input (known rows only), GR (all rows via interpolation), "
            "typewell GR-TVT profile. The TVT column is only merged AFTER prediction for RMSE evaluation. "
            "is_target rows are selected by is_known_tvt==False, which is derived from TVT_input being NaN, "
            "not from the TVT column. GR interpolation across all rows uses positional GR values (no TVT). "
            "pos = last_tvt + last_Z uses last KNOWN TVT_input (anchor), not hidden TVT."
        ),
        "gate_check": {
            "expected_rmse_range": [10.4, 11.6],
            "passed": 10.4 <= pooled_rmse <= 11.6,
            "expected_rows": 3783989,
            "rows_match": n_rows == 3783989,
        },
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result saved: {RESULT_PATH}")

    # Gate check
    if not result["gate_check"]["passed"]:
        logger.warning(
            f"GATE FAILED: pooled RMSE={pooled_rmse:.4f} outside [10.4, 11.6]"
        )
    if not result["gate_check"]["rows_match"]:
        logger.warning(
            f"ROW COUNT MISMATCH: got {n_rows}, expected 3783989"
        )

    logger.info(
        f"Done. pooled_rmse={pooled_rmse:.4f} n_rows={n_rows} runtime={runtime_min:.1f}min"
    )
    return result


if __name__ == "__main__":
    main()
