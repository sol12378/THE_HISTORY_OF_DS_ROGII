#!/usr/bin/env python3
"""exp031: Multi-typewell PF + Physical Likelihood (§C-1/C-2 改造).

地質資料指針:
- §C-1: 生GR差分 → well正規化GR (baseline drift対策)。KCl mud+20APIや
  ツール較正差は加法的なので、z-score正規化や微分で消える。
- §C-2: 点単位GR一致でなく、極値(marker)tyingで局所的に強い拘束。

実装:
- Lateralの観測GR と Typewellの参照GR を、それぞれwell単位で z-score 正規化
- 加えて GR の 1次微分(window=5)を尤度に組み込む
- 尤度 = exp(-d_norm² / 2) × exp(-d_deriv² / 2)
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "exp031_pf_physical_lik"
OUT_DIR = ROOT / "experiments" / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

S = 32
N = 500
K_CAND = 3
SCALE = 8.0
INIT_SPREAD = 4.0
PN = 0.01
VN = 0.002
MOM = 0.998
RP = 0.1
RR = 0.001
RESAMP = 0.5
N_WORKERS = max(1, min(8, (os.cpu_count() or 4) - 1))

# Physical likelihood mixing weights
W_NORM_GR = 1.0
W_DERIV = 0.5    # GR微分項
DERIV_WINDOW = 5


def _normalize_gr(gr, gr_ref=None):
    """well単位 z-score 正規化. gr_ref = (mean, std)指定可."""
    gr = np.asarray(gr, dtype=float)
    if gr_ref is not None:
        mu, sd = gr_ref
    else:
        mu = float(np.nanmean(gr))
        sd = float(np.nanstd(gr) + 1e-6)
    return (gr - mu) / sd, (mu, sd)


def _gr_derivative(gr, window=DERIV_WINDOW):
    """中央差分風: gr[t+w]-gr[t-w]を線形補間して滑らかにdGR/dx."""
    gr = np.asarray(gr, dtype=float)
    n = len(gr)
    if n < 2 * window + 1:
        return np.zeros_like(gr)
    deriv = np.zeros_like(gr)
    deriv[window:n-window] = (gr[2*window:] - gr[:n-2*window]) / (2 * window)
    deriv[:window] = deriv[window]
    deriv[n-window:] = deriv[n-window-1]
    return deriv


def pf_well_physical(tw_tvt, tw_gr_norm, tw_gr_deriv, md_v, z_v,
                     gr_v_norm, gr_v_deriv, gs, ir, last_tvt, last_Z, last_MD, seed=0):
    """Physical likelihood PF (vectorized).
    入力 GR は既に well-normalized。gs は normalized scale (~1.0)。"""
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
        eg = np.interp(tvt_p.ravel(), tw_tvt, tw_gr_norm).reshape(S, N)
        eg_d = np.interp(tvt_p.ravel(), tw_tvt, tw_gr_deriv).reshape(S, N)
        # 正規化GR項
        d_norm = (gr_v_norm[i] - eg) / gs
        # 微分項
        d_deriv = (gr_v_deriv[i] - eg_d) / gs
        lk = np.maximum(
            np.exp(-0.5 * W_NORM_GR * np.minimum(d_norm * d_norm, 600.)) *
            np.exp(-0.5 * W_DERIV * np.minimum(d_deriv * d_deriv, 600.)),
            1e-300
        )
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
        pred, lik = pf_well_physical(
            cand["tw_tvt"], cand["tw_gr_norm"], cand["tw_gr_deriv"],
            p["md_v"], p["z_v"], p["gr_v_norm"], p["gr_v_deriv"],
            cand["gs"], cand["ir"],
            p["last_tvt"], p["last_Z"], p["last_MD"], seed=ci * 1000 + 7,
        )
        cand_preds[ci] = pred
        cand_liks[ci] = lik
    cand_w = np.exp((cand_liks - cand_liks.max()) / SCALE); cand_w /= cand_w.sum()
    final = (cand_w[:, None] * cand_preds).sum(0)
    info = {"used_candidates": K, "cand_liks": cand_liks.tolist(),
            "best_cand": int(cand_liks.argmax())}
    return wid, final, info


def _typewell_signature(tw_df):
    if len(tw_df) == 0:
        return ("empty",)
    return (round(float(tw_df["TVT"].min()), 1), round(float(tw_df["TVT"].max()), 1),
            round(float(tw_df["GR"].mean()), 1), round(float(tw_df["GR"].std()), 1), len(tw_df))


def build(base_path, tw_path):
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "X", "Y", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}
    well_centroid = tr.groupby("well_id", sort=False).agg(
        X_mean=("X", "mean"), Y_mean=("Y", "mean"))
    wid_list = well_centroid.index.tolist()
    coord = well_centroid[["X_mean", "Y_mean"]].to_numpy(float)
    wid_to_idx = {w: i for i, w in enumerate(wid_list)}
    well_to_sig = {}
    for w in wid_list:
        tw_g = tw_by_well.get(w)
        well_to_sig[w] = _typewell_signature(tw_g) if (tw_g is not None and len(tw_g) >= 2) else None

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

        # =========== lateral GR z-score 正規化 (全well GRから) ===========
        gr_full = g["GR"].interpolate(limit_direction="both")
        gr_full = gr_full.fillna(gr_full.mean()).to_numpy(float)
        lat_gr_norm, lat_ref = _normalize_gr(gr_full)
        lat_gr_deriv = _gr_derivative(lat_gr_norm)

        # known GR (anchor for gs estimation)
        gr_known = known["GR"].fillna(0).to_numpy(float)
        gr_known_norm, _ = _normalize_gr(gr_known, lat_ref)

        # candidate selection (spatial nearest)
        my_idx = wid_to_idx.get(wid)
        if my_idx is None:
            order = []
        else:
            dist = np.linalg.norm(coord - coord[my_idx], axis=1)
            order = np.argsort(dist)
        seen_sigs = {own_sig}
        cand_wids = [wid]
        for j in order:
            ow = wid_list[j]
            if ow == wid:
                continue
            osig = well_to_sig.get(ow)
            if osig is None or osig in seen_sigs:
                continue
            seen_sigs.add(osig)
            cand_wids.append(ow)
            if len(cand_wids) >= K_CAND:
                break

        t_known = known["TVT_input"].to_numpy(float)
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
            tw_gr_norm, tw_ref = _normalize_gr(tw_gr)
            tw_gr_deriv = _gr_derivative(tw_gr_norm)
            # gs (normalized space residual)
            tw_at_k = np.interp(t_known, tw_tvt, tw_gr_norm)
            gs = float(np.clip(np.nanstd(gr_known_norm - tw_at_k), 0.3, 3.0))
            cands.append({
                "tw_tvt": tw_tvt, "tw_gr_norm": tw_gr_norm, "tw_gr_deriv": tw_gr_deriv,
                "gs": gs, "ir": ir, "tw_id": cwid,
            })
        if not cands:
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor, "n_eval": int(len(tgt))})
            continue

        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_v_norm = lat_gr_norm[tgt_mask]
        gr_v_deriv = lat_gr_deriv[tgt_mask]
        last = known.iloc[-1]
        payloads.append({
            "wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
            "cands": cands,
            "md_v": tgt["MD"].to_numpy(float),
            "z_v": tgt["Z"].to_numpy(float),
            "gr_v_norm": gr_v_norm, "gr_v_deriv": gr_v_deriv,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]),
            "last_MD": float(last["MD"]), "anchor": anchor,
        })
    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


def run_split(base_path, tw_path):
    payloads, out = build(base_path, tw_path)
    print(f"  payloads={len(payloads)}")
    pred_by_wid = {}
    info_by_wid = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred, info) in enumerate(ex.map(_process_well, payloads, chunksize=2)):
            pred_by_wid[wid] = pred
            info_by_wid[wid] = info
            if (k + 1) % 50 == 0:
                el = time.time() - t0
                eta = el * len(payloads) / (k + 1) - el
                print(f"  {k+1}/{len(payloads)} wells el={el:.0f}s eta={eta:.0f}s ({el/(k+1):.2f}s/well)", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
    out = out.copy(); out["pred_tvt"] = pred_col
    return out, info_by_wid


def main():
    t0 = time.time()
    print(f"[{EXP_ID}] multi-tw + physical-likelihood PF  S={S} N={N} K={K_CAND} W_DERIV={W_DERIV}")
    print("\n--- TRAIN 773 wells ---", flush=True)
    preds, info = run_split("data/processed/train_base_v001.parquet",
                            "data/processed/typewell_train_base_v001.parquet")
    def tvt_rmse(y, p): return float(np.sqrt(np.mean((np.array(y) - np.array(p)) ** 2)))
    cv = tvt_rmse(preds["TVT"], preds["pred_tvt"])
    anc = tvt_rmse(preds["TVT"], preds["last_known_TVT"])
    print(f"\n  Physical-lik PF CV = {cv:.6f}   anchor = {anc:.6f}")
    preds.to_csv(OUT_DIR / "oof.csv", index=False)
    well_rows = []
    for wid, g in preds.groupby("well_id"):
        well_rows.append({
            "well_id": wid, "n": len(g),
            "anchor_rmse": tvt_rmse(g["TVT"], g["last_known_TVT"]),
            "pf_rmse": tvt_rmse(g["TVT"], g["pred_tvt"]),
        })
    well = pd.DataFrame(well_rows)
    well.to_csv(OUT_DIR / "per_well.csv", index=False)
    n_beat = int((well["pf_rmse"] < well["anchor_rmse"]).sum())
    n_broken = int((well["pf_rmse"] > 20).sum())
    print(f"  beat anchor: {n_beat}/{len(well)}, broken(>20): {n_broken}")

    try:
        old = pd.read_csv("experiments/exp022_particle_filter/per_well.csv")
        merged = well.merge(old[["well_id", "pf_rmse"]].rename(columns={"pf_rmse": "exp022_rmse"}),
                            on="well_id", how="left")
        improvement = merged["exp022_rmse"] - merged["pf_rmse"]
        print(f"  vs exp022 (own typewell, raw GR): median delta={improvement.median():+.3f}, "
              f"mean={improvement.mean():+.3f}")
        broken_old = old[old["pf_rmse"] > 20]
        broken_after = merged.set_index("well_id").loc[broken_old["well_id"].values]
        rescue = int((broken_after["pf_rmse"] <= 20).sum())
        print(f"  47壊れwell中、新PFで復活(<=20): {rescue}/47")
    except Exception as e:
        print(f"  比較失敗: {e}")

    print("\n--- TEST 3 wells ---", flush=True)
    test_preds, test_info = run_split("data/processed/test_base_v001.parquet",
                                       "data/processed/typewell_test_base_v001.parquet")
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(test_preds[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
                                on="id", how="left", validate="one_to_one")
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    truth = pd.read_parquet("data/processed/train_base_v001.parquet",
                            columns=["well_id", "row_idx", "TVT"])
    tp = test_preds.merge(truth, on=["well_id", "row_idx"], how="left", suffixes=("", "_truth"))
    print("  test wells RMSE:")
    for wid, g in tp.groupby("well_id"):
        rmse = tvt_rmse(g["TVT_truth"], g["pred_tvt"])
        print(f"    {wid}: {rmse:.4f}")

    summary = {"exp_id": EXP_ID, "cv_rmse": cv, "anchor_rmse": anc,
               "n_wells": int(len(well)), "n_pf_beats_anchor": n_beat, "n_broken": n_broken,
               "wall_time_sec": float(time.time() - t0),
               "params": {"S": S, "N": N, "K_CAND": K_CAND, "W_DERIV": W_DERIV, "DERIV_WINDOW": DERIV_WINDOW}}
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
