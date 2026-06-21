#!/usr/bin/env python3
"""exp030: Multi-typewell Particle Filter — 各wellに対しK=3のtypewell候補で並走

地質資料§3: ROGII公式 StarSteer/GeoAssist の設計思想 = 「自動で最も近いtypewellを選び、
複数offset wellから複数解釈を作り最良相関を選ぶ」。我々のexp022 PF は 1well=1typewell固定で
**過小利用**。47壊れwell の多くは「割当typewellが悪い」可能性が高い。

実装: 各wellに対し
  candidate 1: own typewell (= exp022 と同じ)
  candidate 2: 空間最近傍wellのtypewell (own と異なる場合)
  candidate 3: 2nd nearest
  → 各候補で独立にPFを32 seeds × 500粒子で実行
  → 各候補の log-likelihood を softmax(/SCALE) 重みで予測を加重平均

scale=64 seed相当の計算量(3x32=96)。完全leak-free。
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "exp030_multi_typewell_pf"
OUT_DIR = ROOT / "experiments" / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reduced seeds per candidate (original 128 × 1, here 32 × 3 = 96 equivalent)
N_PARTICLES = 500
N_SEEDS_PER_CAND = 32
N_CANDIDATES = 3
SCALE = 8.0
N_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))

# PF hyperparameters (same as exp022 tuned config)
MOM = 0.998
VN = 0.002
PN = 0.005
RP = 0.1
RR = 0.001
RESAMP = 0.5
INIT_SPREAD = 2.0


def _pf_single(cand, p, seed):
    """1 seed × 1 typewell candidate の PF実行。(pred_eval[n], log_lik)。"""
    tw_tvt = cand["tw_tvt"]
    tw_gr = cand["tw_gr"]
    gs = cand["gs"]
    ir = cand["ir"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
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
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
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
    """1 well を K candidates × N_SEEDS_PER_CAND で実行、最終 likelihood-weighted 予測。"""
    wid = p["wid"]
    n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0), {}, 0
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"]), {"used_candidates": 0}, 0
    cands = p["cands"]  # list of {tw_tvt, tw_gr, gs, ir, tw_id}
    K = len(cands)

    # K candidates × N_SEEDS_PER_CAND seeds
    cand_preds = np.empty((K, n))
    cand_liks = np.empty(K)
    cand_info = []
    for ci, cand in enumerate(cands):
        preds = np.empty((N_SEEDS_PER_CAND, n))
        liks = np.empty(N_SEEDS_PER_CAND)
        # use different seed range per candidate to avoid overlap
        for s in range(N_SEEDS_PER_CAND):
            preds[s], liks[s] = _pf_single(cand, p, ci * 1000 + s)
        # seed-level likelihood-weighted avg (intra-candidate)
        wts = np.exp((liks - liks.max()) / SCALE)
        wts /= wts.sum()
        cand_preds[ci] = (wts[:, None] * preds).sum(0)
        cand_liks[ci] = float(liks.max())  # best seed's likelihood as candidate score
        cand_info.append({"tw_id": cand.get("tw_id", f"cand{ci}"), "log_lik": cand_liks[ci]})

    # candidate-level likelihood-weighted avg
    cand_w = np.exp((cand_liks - cand_liks.max()) / SCALE)
    cand_w /= cand_w.sum()
    final = (cand_w[:, None] * cand_preds).sum(0)
    info = {
        "used_candidates": K,
        "cand_liks": cand_liks.tolist(),
        "cand_weights": cand_w.tolist(),
        "cand_info": cand_info,
        "best_cand": int(cand_liks.argmax()),
    }
    return wid, final, info, K


def _typewell_signature(tw_df):
    """typewellの内容ハッシュ風シグネチャ (重複検出用)。"""
    if len(tw_df) == 0:
        return ("empty",)
    return (
        round(float(tw_df["TVT"].min()), 1),
        round(float(tw_df["TVT"].max()), 1),
        round(float(tw_df["GR"].mean()), 1),
        round(float(tw_df["GR"].std()), 1),
        len(tw_df),
    )


def build(base_path, tw_path, processed_train_path=None):
    """Build payload list with K typewell candidates per well."""
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "X", "Y", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    # ---- well centroids (空間最近傍探索用) ----
    well_centroid = (
        tr.groupby("well_id", sort=False)
        .agg(X_mean=("X", "mean"), Y_mean=("Y", "mean"))
    )
    wid_list = well_centroid.index.tolist()
    centroids = well_centroid[["X_mean", "Y_mean"]].to_numpy(float)

    # typewell unique signatures map (well_id → tw_sig)
    well_to_sig = {}
    sig_to_wells = {}
    for wid in wid_list:
        tw_g = tw_by_well.get(wid)
        if tw_g is None or len(tw_g) < 2:
            well_to_sig[wid] = None
            continue
        sig = _typewell_signature(tw_g)
        well_to_sig[wid] = sig
        sig_to_wells.setdefault(sig, []).append(wid)

    # For target_set (typically test_wells or all train wells), pick K candidates
    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    payloads = []
    out_frames = []

    # Build a quick spatial index (KDTree-like)
    coord_arr = centroids
    wid_to_idx = {w: i for i, w in enumerate(wid_list)}

    for wid, g in sel.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])
        out_frames.append(tgt[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy())
        own_sig = well_to_sig.get(wid)
        if own_sig is None or len(known) < 2:
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor,
                             "n_eval": int(len(tgt))})
            continue

        # === 候補typewell選定 ===
        # candidate 1: own
        # candidate 2,3: 空間最近傍 distinct typewell
        my_idx = wid_to_idx.get(wid)
        if my_idx is None:
            order = []
        else:
            dist = np.linalg.norm(coord_arr - coord_arr[my_idx], axis=1)
            order = np.argsort(dist)  # increasing
        seen_sigs = {own_sig}
        cand_wids = [wid]
        for j in order:
            other_wid = wid_list[j]
            if other_wid == wid:
                continue
            other_sig = well_to_sig.get(other_wid)
            if other_sig is None or other_sig in seen_sigs:
                continue
            seen_sigs.add(other_sig)
            cand_wids.append(other_wid)
            if len(cand_wids) >= N_CANDIDATES:
                break

        # Build candidate payloads (precomputed typewell + gs + ir using own known)
        md_known = known["MD"].to_numpy(float)
        z_known = known["Z"].to_numpy(float)
        t_known = known["TVT_input"].to_numpy(float)
        gr_known = known["GR"].fillna(0).to_numpy(float)

        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float))
        dz = np.diff(tail["Z"].to_numpy(float))
        dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0

        cands = []
        for cwid in cand_wids:
            tw_g = tw_by_well.get(cwid)
            if tw_g is None or len(tw_g) < 2:
                continue
            tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
            tw_tvt = tw_s["TVT"].to_numpy(float)
            tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
            tw_at_k = np.interp(t_known, tw_tvt, tw_gr)
            gs = float(np.clip(np.nanstd(gr_known - tw_at_k), 10., 60.))
            cands.append({
                "tw_tvt": tw_tvt, "tw_gr": tw_gr, "gs": gs, "ir": ir, "tw_id": cwid,
            })

        if not cands:
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor,
                             "n_eval": int(len(tgt))})
            continue

        # Pre-compute hidden trajectory
        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both")
        # fill GR with own typewell GR mean as fallback (use first cand's tw mean)
        fallback_gr = float(np.nanmean(cands[0]["tw_gr"]))
        gr_full = gr_full.fillna(fallback_gr).to_numpy(float)
        gr_v = gr_full[tgt_mask]

        last = known.iloc[-1]
        payloads.append({
            "wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
            "cands": cands,
            "md_v": tgt["MD"].to_numpy(float),
            "z_v": tgt["Z"].to_numpy(float),
            "gr_v": gr_v,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]),
            "last_MD": float(last["MD"]), "anchor": anchor,
        })
    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


def run_split(base_path, tw_path):
    payloads, out = build(base_path, tw_path)
    pred_by_wid = {}
    info_by_wid = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred, info, kused) in enumerate(ex.map(_process_well, payloads, chunksize=2)):
            pred_by_wid[wid] = pred
            info_by_wid[wid] = info
            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed * len(payloads) / (k + 1) - elapsed
                print(f"  {k+1}/{len(payloads)} wells done, elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
    out = out.copy()
    out["pred_tvt"] = pred_col
    return out, info_by_wid


def main() -> None:
    t0 = time.time()
    print(f"[{EXP_ID}] multi-typewell PF — 773-well CV  (workers={N_WORKERS}, "
          f"candidates={N_CANDIDATES}, seeds/cand={N_SEEDS_PER_CAND}, particles={N_PARTICLES})")

    print("\n--- TRAIN 773 wells (multi-typewell PF) ---", flush=True)
    preds, info = run_split("data/processed/train_base_v001.parquet",
                            "data/processed/typewell_train_base_v001.parquet")
    def tvt_rmse(y, p): return float(np.sqrt(np.mean((np.array(y) - np.array(p)) ** 2)))
    cv = tvt_rmse(preds["TVT"], preds["pred_tvt"])
    anc = tvt_rmse(preds["TVT"], preds["last_known_TVT"])
    print(f"  multi-typewell PF CV = {cv:.6f}   anchor = {anc:.6f}")
    preds.to_csv(OUT_DIR / "oof.csv", index=False)

    well_rows = []
    for wid, g in preds.groupby("well_id"):
        well_rows.append({
            "well_id": wid, "n": len(g),
            "anchor_rmse": tvt_rmse(g["TVT"], g["last_known_TVT"]),
            "pf_rmse": tvt_rmse(g["TVT"], g["pred_tvt"]),
            "n_cand": info.get(wid, {}).get("used_candidates", 0),
            "best_cand": info.get(wid, {}).get("best_cand", -1),
        })
    well = pd.DataFrame(well_rows)
    well.to_csv(OUT_DIR / "per_well.csv", index=False)
    n_beat = int((well["pf_rmse"] < well["anchor_rmse"]).sum())
    n_broken = int((well["pf_rmse"] > 20).sum())
    print(f"  PF が anchor に勝つ well: {n_beat}/{len(well)}, broken(>20)={n_broken}")
    print(f"  best_cand distribution:")
    print(well["best_cand"].value_counts().sort_index())

    # === Compare with exp022 (own typewell only) ===
    try:
        old = pd.read_csv("experiments/exp022_particle_filter/per_well.csv")
        merged = well.merge(old[["well_id", "pf_rmse"]].rename(columns={"pf_rmse": "exp022_rmse"}),
                            on="well_id", how="left")
        improvement = merged["exp022_rmse"] - merged["pf_rmse"]
        n_better = int((improvement > 0).sum())
        n_worse = int((improvement < 0).sum())
        print(f"\n  vs exp022 (own typewell): better={n_better}, worse={n_worse}, "
              f"median delta={improvement.median():+.3f}")
        # 47壊れwellの内訳
        broken_old = old[old["pf_rmse"] > 20]
        broken_after = merged.set_index("well_id").loc[broken_old["well_id"].values]
        rescue = int((broken_after["pf_rmse"] <= 20).sum())
        print(f"  exp022の47壊れwell中、新PFで復活(<=20): {rescue}/47")
    except Exception as e:
        print(f"  exp022 比較失敗: {e}")

    print("\n--- TEST 3 wells (multi-typewell PF) ---", flush=True)
    test_preds, test_info = run_split("data/processed/test_base_v001.parquet",
                                       "data/processed/typewell_test_base_v001.parquet")
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(test_preds[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
                                on="id", how="left", validate="one_to_one")
    assert not sub["tvt"].isna().any()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    # Compare test wells with train truth
    truth = pd.read_parquet("data/processed/train_base_v001.parquet",
                            columns=["well_id", "row_idx", "TVT"])
    tp = test_preds.merge(truth, on=["well_id", "row_idx"], how="left", suffixes=("", "_truth"))
    print(f"  test wells (vs train truth):")
    for wid, g in tp.groupby("well_id"):
        rmse = tvt_rmse(g["TVT_truth"], g["pred_tvt"])
        print(f"    {wid}: {rmse:.4f}")

    summary = {
        "exp_id": EXP_ID,
        "method": f"Multi-typewell PF: K={N_CANDIDATES} candidates × {N_SEEDS_PER_CAND} seeds × {N_PARTICLES} particles",
        "cv_rmse": cv,
        "anchor_rmse": anc,
        "n_wells": int(len(well)),
        "n_pf_beats_anchor": n_beat,
        "n_broken": n_broken,
        "best_cand_distribution": well["best_cand"].value_counts().sort_index().to_dict(),
        "wall_time_sec": float(time.time() - t0),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
