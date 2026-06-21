#!/usr/bin/env python3
"""exp030c: Multi-typewell PF v2 — GR-similarity候補選定 + own-prior bias.

exp030b の弱点修正:
- spatial-only は地質ブロック異なる候補も拾う(失敗)
- 32 seeds は尤度推定ノイジー → SCALE上げてown優遇

地質資料§3.3:
- 候補選定 = 空間距離 + build区間GR波形類似度 + TVT range overlap
- own typewell に強prior bias、only switch if alternative >> own

scoring:
  score_cand = α·spatial_dist + β·GR_KL + γ·TVT_range_mismatch
              (低いほど良い)
  own_bonus: own typewell の log_lik に + bonus を加算
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "exp030c_multi_tw_gr_sim"
OUT_DIR = ROOT / "experiments" / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

S = 32           # seeds per candidate
N = 500          # particles
K_CAND = 3       # candidates
SCALE = 16.0     # higher → own dominance increases
OWN_BIAS = 2.0   # log_lik bonus for own typewell

# selection score weights
SEL_W_SPATIAL = 0.4
SEL_W_GR = 0.4
SEL_W_TVT = 0.2

# PF tuned config
INIT_SPREAD = 4.0
PN = 0.01
VN = 0.002
MOM = 0.998
RP = 0.1
RR = 0.001
RESAMP = 0.5
N_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))


def pf_well_vec(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed=0):
    """Vectorized PF (S,N) — same as exp030b."""
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + INIT_SPREAD * rng.standard_normal((S, N))
    rate = ir + 0.01 * rng.standard_normal((S, N))
    w = np.full((S, N), 1.0 / N)
    res = np.empty((S, n))
    log_lik = np.zeros(S)
    prev_MD = last_MD
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    arangeN = np.arange(N)
    base = (np.arange(S)[:, None] * 2.0)
    col_off = (np.arange(S)[:, None] * N)
    for i in range(n):
        dm = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal((S, N))
        pos = pos + rate * dm + PN * rng.standard_normal((S, N))
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p.ravel(), tw_tvt, tw_gr).reshape(S, N)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        avg = (w * lk).sum(axis=1)
        log_lik += np.log(np.maximum(avg, 1e-300))
        w = w * lk
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        neff = 1.0 / (w * w).sum(axis=1)
        need = neff < RESAMP * N
        if need.any():
            cum = np.cumsum(w, axis=1)
            u0 = rng.uniform(0, 1.0 / N, size=(S, 1))
            posn = u0 + arangeN[None, :] / N
            cflat = (cum + base).ravel()
            pflat = (posn + base).ravel()
            gidx = np.clip(np.searchsorted(cflat, pflat), 0, S * N - 1)
            lidx = np.clip(gidx.reshape(S, N) - col_off, 0, N - 1)
            pos_rs = np.take_along_axis(pos, lidx, axis=1) + RP * rng.standard_normal((S, N))
            rate_rs = np.take_along_axis(rate, lidx, axis=1) + RR * rng.standard_normal((S, N))
            m = need[:, None]
            pos = np.where(m, pos_rs, pos)
            rate = np.where(m, rate_rs, rate)
            w = np.where(m, 1.0 / N, w)
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        res[:, i] = (w * (pos - z_v[i])).sum(axis=1)
        prev_MD = md_v[i]
    wts = np.exp((log_lik - log_lik.max()) / SCALE); wts /= wts.sum()
    pred = (wts[:, None] * res).sum(axis=0)
    return pred, float(log_lik.max())


def gr_kl_divergence(gr_a, gr_b, n_bins=20):
    """軽量GR similarity: ヒストグラム KL風 ratio."""
    gr_a = np.asarray(gr_a)
    gr_b = np.asarray(gr_b)
    lo = float(min(np.nanmin(gr_a), np.nanmin(gr_b)))
    hi = float(max(np.nanmax(gr_a), np.nanmax(gr_b)))
    if hi - lo < 1e-6:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(gr_a, bins=bins); ha = ha / (ha.sum() + 1e-9) + 1e-9
    hb, _ = np.histogram(gr_b, bins=bins); hb = hb / (hb.sum() + 1e-9) + 1e-9
    return float(np.sum(ha * np.log(ha / hb)))


def _process_well(p):
    wid = p["wid"]
    n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0), {"used_candidates": 0}
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"]), {"used_candidates": 0}
    cands = p["cands"]
    K = len(cands)
    cand_preds = np.empty((K, n))
    cand_liks = np.empty(K)
    for ci, cand in enumerate(cands):
        pred, lik = pf_well_vec(
            cand["tw_tvt"], cand["tw_gr"],
            p["md_v"], p["z_v"], p["gr_v"],
            cand["gs"], cand["ir"],
            p["last_tvt"], p["last_Z"], p["last_MD"],
            seed=ci * 1000 + 7,
        )
        cand_preds[ci] = pred
        # own typewell に bonus
        cand_liks[ci] = lik + (OWN_BIAS if ci == 0 else 0.0)
    cand_w = np.exp((cand_liks - cand_liks.max()) / SCALE); cand_w /= cand_w.sum()
    final = (cand_w[:, None] * cand_preds).sum(0)
    info = {"used_candidates": K, "cand_liks": cand_liks.tolist(),
            "cand_weights": cand_w.tolist(),
            "best_cand": int(cand_liks.argmax()),
            "cand_wids": [c.get("tw_id", "") for c in cands]}
    return wid, final, info


def _typewell_signature(tw_df):
    if len(tw_df) == 0:
        return None
    return (round(float(tw_df["TVT"].min()), 1), round(float(tw_df["TVT"].max()), 1),
            round(float(tw_df["GR"].mean()), 1), round(float(tw_df["GR"].std()), 1), len(tw_df))


def build(base_path, tw_path):
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "X", "Y", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    well_summary = (
        tr.groupby("well_id", sort=False)
        .agg(X_mean=("X", "mean"), Y_mean=("Y", "mean"),
             TVT_med=("TVT_input", lambda x: float(x.dropna().median()) if x.dropna().size else 0.0))
    )
    wid_list = well_summary.index.tolist()
    coord = well_summary[["X_mean", "Y_mean"]].to_numpy(float)
    tvt_med_arr = well_summary["TVT_med"].to_numpy(float)
    wid_to_idx = {w: i for i, w in enumerate(wid_list)}

    # typewell signatures + GR histograms for similarity
    well_to_sig = {}
    well_to_gr_hist = {}
    for w in wid_list:
        tw_g = tw_by_well.get(w)
        well_to_sig[w] = _typewell_signature(tw_g) if (tw_g is not None and len(tw_g) >= 2) else None
        if tw_g is not None and len(tw_g) >= 2:
            well_to_gr_hist[w] = tw_g["GR"].dropna().to_numpy(float)
        else:
            well_to_gr_hist[w] = None

    # spatial distance normalization
    coord_n = (coord - coord.mean(0)) / (coord.std(0) + 1e-9)

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    payloads = []; out_frames = []
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
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor, "n_eval": int(len(tgt))})
            continue

        # ── multi-criteria candidate selection ──
        my_idx = wid_to_idx.get(wid)
        my_gr_hist = known["GR"].dropna().to_numpy(float)  # use lateral build-section GR as fingerprint
        my_tvt_med = float(known["TVT_input"].median())

        candidates_score = []
        for ow in wid_list:
            if ow == wid:
                continue
            o_sig = well_to_sig.get(ow)
            if o_sig is None or o_sig == own_sig:
                continue
            o_gr = well_to_gr_hist.get(ow)
            if o_gr is None or len(o_gr) < 10:
                continue
            j = wid_to_idx[ow]
            # spatial in normalized coords
            spatial_d = float(np.linalg.norm(coord_n[my_idx] - coord_n[j]))
            # GR similarity (KL)
            gr_kl = gr_kl_divergence(my_gr_hist, o_gr)
            # TVT range mismatch
            tvt_mismatch = abs(my_tvt_med - tvt_med_arr[j]) / 1000.0  # normalize
            score = (SEL_W_SPATIAL * spatial_d
                     + SEL_W_GR * min(gr_kl, 5.0)
                     + SEL_W_TVT * tvt_mismatch)
            candidates_score.append((score, ow))
        candidates_score.sort(key=lambda x: x[0])

        # always include own first, then top K-1 by score
        cand_wids = [wid] + [w for _, w in candidates_score[:K_CAND - 1]]

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
            cands.append({"tw_tvt": tw_tvt, "tw_gr": tw_gr, "gs": gs, "ir": ir, "tw_id": cwid})
        if not cands:
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor, "n_eval": int(len(tgt))})
            continue

        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both")
        fallback_gr = float(np.nanmean(cands[0]["tw_gr"]))
        gr_full = gr_full.fillna(fallback_gr).to_numpy(float)
        gr_v = gr_full[tgt_mask]
        last = known.iloc[-1]
        payloads.append({
            "wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
            "cands": cands,
            "md_v": tgt["MD"].to_numpy(float), "z_v": tgt["Z"].to_numpy(float), "gr_v": gr_v,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]),
            "last_MD": float(last["MD"]), "anchor": anchor,
        })
    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


def run_split(base_path, tw_path):
    payloads, out = build(base_path, tw_path)
    print(f"  payloads={len(payloads)}")
    pred_by_wid = {}; info_by_wid = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred, info) in enumerate(ex.map(_process_well, payloads, chunksize=2)):
            pred_by_wid[wid] = pred; info_by_wid[wid] = info
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                eta = el * len(payloads) / (k + 1) - el
                print(f"  {k+1}/{len(payloads)} el={el:.0f}s eta={eta:.0f}s", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
    out = out.copy(); out["pred_tvt"] = pred_col
    return out, info_by_wid


def main():
    t0 = time.time()
    print(f"[{EXP_ID}] multi-tw v2 (GR-sim + own bias)  S={S} N={N} K={K_CAND} SCALE={SCALE} OWN_BIAS={OWN_BIAS}")
    preds, info = run_split("data/processed/train_base_v001.parquet",
                            "data/processed/typewell_train_base_v001.parquet")
    def tvt_rmse(y, p): return float(np.sqrt(np.mean((np.array(y) - np.array(p)) ** 2)))
    cv = tvt_rmse(preds["TVT"], preds["pred_tvt"])
    anc = tvt_rmse(preds["TVT"], preds["last_known_TVT"])
    print(f"  CV = {cv:.6f}, anchor = {anc:.6f}")
    preds.to_csv(OUT_DIR / "oof.csv", index=False)
    well_rows = []
    for wid, g in preds.groupby("well_id"):
        well_rows.append({"well_id": wid, "n": len(g),
                          "anchor_rmse": tvt_rmse(g["TVT"], g["last_known_TVT"]),
                          "pf_rmse": tvt_rmse(g["TVT"], g["pred_tvt"]),
                          "best_cand": info.get(wid, {}).get("best_cand", -1)})
    well = pd.DataFrame(well_rows); well.to_csv(OUT_DIR / "per_well.csv", index=False)
    n_beat = int((well["pf_rmse"] < well["anchor_rmse"]).sum())
    n_broken = int((well["pf_rmse"] > 20).sum())
    print(f"  beat anchor: {n_beat}/{len(well)}, broken(>20): {n_broken}")
    print(f"  best_cand dist:\n{well['best_cand'].value_counts().sort_index()}")

    # vs exp022/exp030b
    for ref_name, ref_path in [("exp022", "experiments/exp022_particle_filter/per_well.csv"),
                                ("exp030b", "experiments/exp030b_multi_tw_vec/per_well.csv")]:
        try:
            old = pd.read_csv(ref_path)
            m = well.merge(old[["well_id", "pf_rmse"]].rename(columns={"pf_rmse": "ref"}),
                            on="well_id", how="left")
            d = m["ref"] - m["pf_rmse"]
            print(f"  vs {ref_name}: better={int((d>0).sum())} worse={int((d<0).sum())} mean delta={d.mean():+.3f}")
        except Exception as e:
            print(f"  vs {ref_name}: {e}")

    summary = {
        "exp_id": EXP_ID, "cv_rmse": cv, "anchor_rmse": anc,
        "n_wells": int(len(well)), "n_pf_beats_anchor": n_beat, "n_broken": n_broken,
        "best_cand_dist": well["best_cand"].value_counts().sort_index().to_dict(),
        "wall_time_sec": float(time.time() - t0),
        "params": {"S": S, "N": N, "K": K_CAND, "SCALE": SCALE, "OWN_BIAS": OWN_BIAS,
                   "SEL_W_SPATIAL": SEL_W_SPATIAL, "SEL_W_GR": SEL_W_GR, "SEL_W_TVT": SEL_W_TVT},
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
