#!/usr/bin/env python3
"""exp040 smoke test — 5 wells only to verify multi-scale logic."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp040_multiscale_pf_smoke"
OUT_DIR = Path("experiments") / EXP_ID

N_PARTICLES = 500
N_SEEDS = 128
SCALES = [3.0, 5.0, 8.0, 12.0]
N_WORKERS = max(1, min(4, (os.cpu_count() or 4) - 1))

MOM = 0.998
VN = 0.002
PN = 0.01
RP = 0.1
RR = 0.001
RESAMP = 0.5
INIT_SPREAD = 4


def _pf_single(p, seed):
    """1 seed の PF。(pred_eval[n], log_lik)。"""
    tw_tvt = p["tw_tvt"]
    tw_gr = p["tw_gr"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
    gs = p["gs"]
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
    lo = tw_tvt[0] - 100
    hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.0)), 1e-300)
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
    """1 well を 128 seed で PF シミュレーション。

    per-seed (preds, log_liks) を集め、複数 SCALE で再加重・平均。
    戻り値: (wid, pred[n])。
    """
    wid = p["wid"]
    n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])

    # Step 1: 128 seed で PF を走らせ、per-seed (preds, log_liks) を保存
    preds = np.empty((N_SEEDS, n))
    liks = np.empty(N_SEEDS)
    for s in range(N_SEEDS):
        preds[s], liks[s] = _pf_single(p, s)

    # Step 2: 複数 SCALE で再加重・平均
    multi_pred = np.zeros(n)
    for scale in SCALES:
        wts = np.exp((liks - liks.max()) / scale)
        wts /= wts.sum()
        multi_pred += (wts[:, None] * preds).sum(0) / len(SCALES)

    return wid, multi_pred


def build(base_path, tw_path, max_wells=5):
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
    well_count = 0
    for wid, g in sel.groupby("well_id", sort=False):
        if well_count >= max_wells:
            break
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue
        well_count += 1
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

        k_tvt = known["TVT_input"].to_numpy(float)
        k_gr = known["GR"].fillna(0).to_numpy(float)
        tw_at_k = np.interp(k_tvt, tw_tvt, tw_gr)
        gs = float(np.clip(np.nanstd(k_gr - tw_at_k), 10.0, 60.0))
        # initial rate ir from tail-30 known
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
                "anchor": anchor,
            }
        )
    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


def run_split(base_path, tw_path, max_wells=5):
    payloads, out = build(base_path, tw_path, max_wells=max_wells)
    print(f"  Built {len(payloads)} payloads for {len(out)} rows", flush=True)
    pred_by_wid = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred) in enumerate(ex.map(_process_well, payloads, chunksize=2)):
            pred_by_wid[wid] = pred
            print(f"    well {k+1}/{len(payloads)}: {wid} done", flush=True)
    # assemble predictions aligned to out (row_idx order per well preserved in build)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        n = len(g)
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
    out = out.copy()
    out["pred_tvt"] = pred_col
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[{EXP_ID}] smoke test (5 wells) — Multi-scale Temperature PF "
        f"(seeds={N_SEEDS}, particles={N_PARTICLES}, scales={SCALES})"
    )

    wall_start = time.time()

    print("TRAIN 5 wells smoke test ...", flush=True)
    preds = run_split(
        "data/processed/train_base_v001.parquet",
        "data/processed/typewell_train_base_v001.parquet",
        max_wells=5,
    )
    if len(preds) > 0:
        cv = tvt_rmse(preds["TVT"], preds["pred_tvt"])
        anc = tvt_rmse(preds["TVT"], preds["last_known_TVT"])
        print(f"  Multi-scale PF CV = {cv:.6f}   anchor = {anc:.6f}")
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
        print(f"  Multi-scale PF が anchor に勝つ well: {n_beat}/{len(well)}")
        print(well)
    else:
        print("  ERROR: No predictions generated!")

    wall_time_sec = time.time() - wall_start
    print(f"\nSmoke test completed in {wall_time_sec:.1f} sec")


if __name__ == "__main__":
    main()
