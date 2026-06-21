#!/usr/bin/env python3
"""exp040: Multi-scale Temperature Particle Filter ensemble.

exp022 の per-seed (preds, log_liks) を保存し、複数の SCALE 温度(3.0, 5.0, 8.0, 12.0)
で再加重して平均。単一 SCALE=8.0 の exp022 (CV=11.024) から改善を狙う。

実装: PFシミュレーションは1回、得られた per-seed 結果を複数温度で再利用。
計算コストは exp022 とほぼ同じ。完全leak-free (hidden TVT不使用、GR+typewell+anchor+Z+MDのみ)。

**重い実験(>30分)**: well単位に ProcessPoolExecutor で並列化(全773・フル設定)。
"""

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

EXP_ID = "exp040_multiscale_pf"
OUT_DIR = Path("experiments") / EXP_ID

N_PARTICLES = 500
N_SEEDS = 128
SCALES = [3.0, 5.0, 8.0, 12.0]  # multi-scale temperatures
N_WORKERS = max(1, min(9, (os.cpu_count() or 4) - 1))

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


def build(base_path, tw_path):
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


def run_split(base_path, tw_path):
    payloads, out = build(base_path, tw_path)
    pred_by_wid = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred) in enumerate(ex.map(_process_well, payloads, chunksize=4)):
            pred_by_wid[wid] = pred
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(payloads)} wells done", flush=True)
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
        f"[{EXP_ID}] Multi-scale Temperature PF "
        f"(workers={N_WORKERS}, seeds={N_SEEDS}, particles={N_PARTICLES}, scales={SCALES})"
    )

    wall_start = time.time()

    print("TRAIN 773 wells (multi-scale PF) ...", flush=True)
    preds = run_split(
        "data/processed/train_base_v001.parquet",
        "data/processed/typewell_train_base_v001.parquet",
    )
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
    n_broken = int((well["pf_rmse"] > 20).sum())
    print(f"  Multi-scale PF が anchor に勝つ well: {n_beat}/{len(well)}", flush=True)
    print(f"  RMSE > 20 (broken): {n_broken}", flush=True)

    print("TEST submission (multi-scale PF) ...", flush=True)
    test_preds = run_split(
        "data/processed/test_base_v001.parquet",
        "data/processed/typewell_test_base_v001.parquet",
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
    print(f"  submission rows: {len(sub)}", flush=True)
    test_preds.to_csv(OUT_DIR / "test_pred.csv", index=False)
    print("  test_pred.csv saved", flush=True)

    truth = pd.read_parquet(
        "data/processed/train_base_v001.parquet", columns=["well_id", "row_idx", "TVT"]
    )
    tp = test_preds.merge(truth, on=["well_id", "row_idx"], how="left", suffixes=("", "_truth"))
    test_well_rmse = {}
    for wid, g in tp.groupby("well_id"):
        test_well_rmse[wid] = tvt_rmse(g["TVT_truth"], g["pred_tvt"])
        print(f"    test {wid}: multi-scale PF vs train-truth RMSE = {test_well_rmse[wid]:.4f}")

    wall_time_sec = time.time() - wall_start

    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "method": f"Multi-scale Temperature PF ({N_SEEDS}seed x {N_PARTICLES}part, scales={SCALES}), leak-free",
        "cv_rmse": cv,
        "anchor_rmse": anc,
        "n_wells": int(len(well)),
        "n_pf_beats_anchor": n_beat,
        "n_broken": n_broken,
        "scales": SCALES,
        "test_well_rmse_vs_train_truth": test_well_rmse,
        "leak_risk": "none (no hidden TVT used; GR+typewell+anchor+Z+MD only)",
        "compare": {
            "exp022_single_scale": 11.024,
            "anchor": anc,
        },
        "wall_time_sec": wall_time_sec,
        "notes": "Multi-scale temperature ensemble. Per-seed (preds, log_liks) from 128 seeds, "
        "re-weighted at 4 scales (3.0, 5.0, 8.0, 12.0) and averaged. "
        "Full-well GR interpolation. ProcessPoolExecutor over wells.",
    }
    write_json(OUT_DIR / "result.json", result)

    twr = "\n".join(f"| {k} | {v:.4f} |" for k, v in test_well_rmse.items())
    (OUT_DIR / "notes.md").write_text(
        f"""# {EXP_ID} — Multi-scale Temperature Particle Filter

## 手法
exp022 の PF フレームワークで、1回のシミュレーションから得た per-seed (preds, log_liks) を
複数の温度スケール (3.0, 5.0, 8.0, 12.0) で再加重。
各スケールでの予測を平均化。{N_SEEDS} seed × {N_PARTICLES} 粒子。
GRはwell全体を補間。**完全leak-free**。well単位ProcessPoolExecutor並列。

## 結果(773-well pooled CV)
| 手法 | CV RMSE |
|---|---|
| anchor | {anc:.6f} |
| **Multi-scale PF (exp040)** | **{cv:.6f}** |
| 参考: exp022 single-scale (SCALE=8.0) | 11.024000 |

Multi-scale PF が anchor に勝つ well: {n_beat}/{len(well)}
RMSE > 20 (broken): {n_broken}

## 3 test well (Multi-scale PF vs train真値, 参照"4.71 ft"と照合)
| well | RMSE |
|---|---|
{twr}

## 計算時間
{wall_time_sec:.1f} 秒

## リンク
[[exp022_particle_filter]] [[exp021_beam_track]] [[exp023_leak_lookup]]
"""
    )

    print(f"\nCompleted in {wall_time_sec:.1f} sec")
    print(f"CV: {cv:.6f} (exp022 baseline: 11.024)")
    improvement = 11.024 - cv
    if improvement > 0:
        print(f"✓ Improvement: +{improvement:.6f}")
    else:
        print(f"✗ Degradation: {improvement:.6f}")


if __name__ == "__main__":
    main()
