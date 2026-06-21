#!/usr/bin/env python3
"""exp071: Selector(regime適応トラッカー選択)を厳密leak-freeで測定(main直轄).

worker(exp069)は未完+fold-train RMSE 7.1という異常値(leak/bug疑い)のため再実験。
per-scale PF OOF を生成し、(n_eval, z_span) regime bin毎に最良variant(scale/hold重み)を
**fold内train wellのhidden RMSEのみ**で選択(GroupKFold nested)。fold-eval wellは
configを受け取るだけ=leak-free。基準 PF(scale8)=11.02。

variant: pred = (1-hold)*pf_scale + hold*last_known_tvt。scale∈{3,5,8,12}, hold∈{0,0.1,0.2}.
"""
from __future__ import annotations
import numpy as np, pandas as pd, json, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/exp071_selector"; OUT.mkdir(parents=True, exist_ok=True)
SCALES = [3.0, 5.0, 8.0, 12.0]
N_PART, N_SEED = 500, 128
INIT_SPREAD, PN, VN, MOM, RP, RR, RESAMP = 4.0, 0.01, 0.002, 0.998, 0.1, 0.001, 0.5
NW = max(1, min(10, (os.cpu_count() or 4)))


def pf_well_perscale(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed0=0):
    """vectorized (S,N) PF; return per-scale weighted preds dict + last_known."""
    n = len(md_v)
    if n == 0:
        return {f"s{sc:g}": np.zeros(0) for sc in SCALES}
    S, N = N_SEED, N_PART
    rng = np.random.default_rng(seed0)
    pos = (last_tvt + last_Z) + INIT_SPREAD * rng.standard_normal((S, N))
    rate = ir + 0.01 * rng.standard_normal((S, N))
    w = np.full((S, N), 1.0 / N); res = np.empty((S, n)); ll = np.zeros(S)
    prev = last_MD; lo, hi = tw_tvt[0] - 100, tw_tvt[-1] + 100
    arN = np.arange(N); base = (np.arange(S)[:, None] * 2.0); col = (np.arange(S)[:, None] * N)
    for i in range(n):
        dm = max(md_v[i] - prev, 1.0)
        rate = MOM * rate + VN * rng.standard_normal((S, N))
        pos = pos + rate * dm + PN * rng.standard_normal((S, N))
        tp = np.clip(pos - z_v[i], lo, hi); pos = tp + z_v[i]
        eg = np.interp(tp.ravel(), tw_tvt, tw_gr).reshape(S, N)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        ll += np.log(np.maximum((w * lk).sum(1), 1e-300))
        w = w * lk; w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        neff = 1.0 / (w * w).sum(1); need = neff < RESAMP * N
        if need.any():
            cum = np.cumsum(w, 1); u0 = rng.uniform(0, 1.0 / N, size=(S, 1))
            pf = (u0 + arN[None, :] / N + base).ravel(); cf = (cum + base).ravel()
            gi = np.clip(np.searchsorted(cf, pf), 0, S * N - 1)
            li = np.clip(gi.reshape(S, N) - col, 0, N - 1)
            m2 = need[:, None]
            pos = np.where(m2, np.take_along_axis(pos, li, 1) + RP * rng.standard_normal((S, N)), pos)
            rate = np.where(m2, np.take_along_axis(rate, li, 1) + RR * rng.standard_normal((S, N)), rate)
            w = np.where(m2, 1.0 / N, w); w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        res[:, i] = (w * (pos - z_v[i])).sum(1); prev = md_v[i]
    out = {}
    lln = ll - ll.max()
    for sc in SCALES:
        wt = np.exp(lln / sc); wt /= wt.sum()
        out[f"s{sc:g}"] = (wt[:, None] * res).sum(0)
    return out


def _proc(p):
    wid = p["wid"]; n = p["n"]
    if n == 0 or p.get("no_tw"):
        z = np.full(n, p["anchor"]); return wid, {f"s{sc:g}": z for sc in SCALES}
    return wid, pf_well_perscale(p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
                                 p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"])


def build():
    tr = pd.read_parquet("data/processed/train_base_v001.parquet", columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "last_known_TVT"])
    tw = pd.read_parquet("data/processed/typewell_train_base_v001.parquet", columns=["well_id", "TVT", "GR"])
    twb = {w: g for w, g in tw.groupby("well_id", sort=False)}
    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    pays = []; outs = []
    for wid, g in sel.groupby("well_id", sort=False):
        g = g.sort_values("row_idx"); kn = g[g["is_known_tvt"].astype(bool)]; tg = g[g["is_target"].astype(bool)]
        if len(tg) == 0:
            continue
        anc = float(tg["last_known_TVT"].iloc[0])
        outs.append(tg[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy())
        twg = twb.get(wid)
        if twg is None or len(twg) < 2 or len(kn) < 2:
            pays.append({"wid": wid, "n": len(tg), "no_tw": True, "anchor": anc}); continue
        tws = twg.sort_values("TVT").drop_duplicates("TVT")
        tt = tws["TVT"].to_numpy(float); tgr = tws["GR"].fillna(tws["GR"].mean()).to_numpy(float)
        grf = g["GR"].interpolate(limit_direction="both").fillna(np.nanmean(tgr)).to_numpy(float)
        tm = g["is_target"].astype(bool).to_numpy()
        kt = kn["TVT_input"].to_numpy(float); kg = kn["GR"].fillna(0).to_numpy(float)
        gs = float(np.clip(np.nanstd(kg - np.interp(kt, tt, tgr)), 10., 60.))
        tail = kn.tail(30); dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dmd = np.diff(tail["MD"].to_numpy(float)); mm = dmd > 0
        ir = float(np.median((dt + dz)[mm] / dmd[mm])) if mm.sum() >= 3 else 0.0
        last = kn.iloc[-1]
        pays.append({"wid": wid, "n": len(tg), "no_tw": False, "anchor": anc,
                     "tw_tvt": tt, "tw_gr": tgr, "md_v": tg["MD"].to_numpy(float),
                     "z_v": tg["Z"].to_numpy(float), "gr_v": grf[tm], "gs": gs, "ir": ir,
                     "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]), "last_MD": float(last["MD"])})
    return pays, pd.concat(outs, ignore_index=True)


def main():
    import time; t0 = time.time()
    print(f"[exp071] per-scale PF, workers={NW}", flush=True)
    pays, out = build()
    pred = {}
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for k, (wid, d) in enumerate(ex.map(_proc, pays, chunksize=2)):
            pred[wid] = d
            if (k + 1) % 150 == 0:
                print(f"  {k+1}/{len(pays)} ({time.time()-t0:.0f}s)", flush=True)
    # assemble per-scale columns
    for sc in SCALES:
        col = np.empty(len(out))
        for wid, g in out.groupby("well_id", sort=False):
            col[g.index.to_numpy()] = pred[wid][f"s{sc:g}"]
        out[f"pf_s{sc:g}"] = col
    out.to_csv(OUT / "perscale_oof.csv", index=False)

    # regime bins + folds
    base = pd.read_parquet("data/processed/train_base_v001.parquet", columns=["well_id", "row_idx", "Z", "is_target"])
    bt = base[base["is_target"]].groupby("well_id").agg(n_eval=("row_idx", "size"),
        z_span=("Z", lambda z: float(z.max() - z.min()))).reset_index()
    out = out.merge(bt, on="well_id", how="left")
    out = out.merge(pd.read_csv("data/folds/folds_group_well_v001.csv")[["well_id", "fold"]], on="well_id", how="left")
    n_thr = bt["n_eval"].median(); z_thr = bt["z_span"].quantile([0.33, 0.66]).values
    out["regime"] = (out["n_eval"] > n_thr).astype(int) + 2 * np.searchsorted(z_thr, out["z_span"].values)

    y = out["TVT"].to_numpy(float); a = out["last_known_TVT"].to_numpy(float); fold = out["fold"].to_numpy()
    def rmse(p, m=None):
        m = np.ones(len(y), bool) if m is None else m
        return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
    # variant grid
    variants = [(sc, h) for sc in SCALES for h in (0.0, 0.1, 0.2)]
    def vpred(sc, h):
        return (1 - h) * out[f"pf_s{sc:g}"].to_numpy(float) + h * a
    # nested: for each fold, pick best variant per regime on fold-TRAIN hidden RMSE
    sel_pred = np.empty(len(y))
    for f in sorted(set(fold)):
        tr = fold != f; va = fold == f
        for rg in sorted(out["regime"].unique()):
            rmask_tr = tr & (out["regime"].to_numpy() == rg)
            rmask_va = va & (out["regime"].to_numpy() == rg)
            if rmask_va.sum() == 0:
                continue
            best = (1e9, variants[0])
            for (sc, h) in variants:
                pr = vpred(sc, h)
                if rmask_tr.sum() > 0:
                    r = rmse(pr, rmask_tr)
                    if r < best[0]:
                        best = (r, (sc, h))
            sc, h = best[1]; sel_pred[rmask_va] = vpred(sc, h)[rmask_va]
    cv_sel = rmse(sel_pred)
    cv_s8 = rmse(out["pf_s8"].to_numpy(float))
    res = {"cv_selector_nested": round(cv_sel, 4), "cv_pf_scale8": round(cv_s8, 4),
           "cv_pf_scales_avg": round(rmse(np.mean([out[f"pf_s{sc:g}"] for sc in SCALES], 0)), 4),
           "n_regimes": int(out["regime"].nunique()),
           "leak_check": "variant chosen by fold-TRAIN hidden RMSE only; fold-eval gets config only",
           "wall_sec": round(time.time() - t0, 1)}
    # save selector oof for blend test
    out["pred_tvt"] = sel_pred
    out[["well_id", "row_idx", "id", "TVT", "last_known_TVT", "pred_tvt"]].to_csv(OUT / "oof.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
