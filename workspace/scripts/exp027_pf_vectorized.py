#!/usr/bin/env python3
"""exp027: Vectorized-across-seeds Particle Filter — CV equivalence check.

exp022/exp025bのPF(128 seed pythonループ)を (S,N)配列で一括演算にベクトル化。
リサンプリングはオフセット付き batched searchsorted で完全ベクトル化(pythonループ除去)。
RNGは単一(seed毎独立ループを廃止)なので乱数列は非ベクトル版と異なるが、同一アルゴリズム。
→ full773 CV が tuned PF(10.984) と seedノイズ内で一致するかを実測し、品質維持を確認する。
config = tuned (init_spread=4, pn=0.01)。完全leak-free。単一スレッドでも高速。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

OUT_DIR = Path("experiments") / "exp027_pf_vectorized"
# tuned config
S = 128            # seeds
N = 500            # particles
SCALE = 8.0
INIT_SPREAD = 4.0
PN = 0.01
VN = 0.002
MOM = 0.998
RP = 0.1
RR = 0.001
RESAMP = 0.5


def pf_well_vec(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed=0):
    """全S seedを (S,N) 配列で一括にPF。likelihood加重平均した (n,) 予測を返す。"""
    n = len(md_v)
    if n == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + INIT_SPREAD * rng.standard_normal((S, N))
    rate = ir + 0.01 * rng.standard_normal((S, N))
    w = np.full((S, N), 1.0 / N)
    res = np.empty((S, n))
    log_lik = np.zeros(S)
    prev_MD = last_MD
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    arangeN = np.arange(N)
    base = (np.arange(S)[:, None] * 2.0)        # disjoint per-seed offset for batched searchsorted
    col_off = (np.arange(S)[:, None] * N)        # local-index offset
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
            posn = u0 + arangeN[None, :] / N            # (S,N) sorted per row
            cflat = (cum + base).ravel()                # globally sorted
            pflat = (posn + base).ravel()               # globally sorted
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
    return (wts[:, None] * res).sum(axis=0)


def build(base_path, tw_path):
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}
    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    items = []; out_frames = []
    for wid, g in sel.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]; tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])
        out_frames.append(tgt[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy())
        tw_g = tw_by_well.get(wid)
        gr_full = g["GR"].interpolate(limit_direction="both")
        tgt_mask = g["is_target"].astype(bool).to_numpy()
        if tw_g is None or len(tw_g) < 2 or len(known) < 2:
            items.append((wid, None)); continue
        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tw_tvt = tw_s["TVT"].to_numpy(float); tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
        k_tvt = known["TVT_input"].to_numpy(float); k_gr = known["GR"].fillna(0).to_numpy(float)
        gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10., 60.))
        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
        last = known.iloc[-1]
        items.append((wid, dict(tw_tvt=tw_tvt, tw_gr=tw_gr,
                                md_v=tgt["MD"].to_numpy(float), z_v=tgt["Z"].to_numpy(float),
                                gr_v=gr_full[tgt_mask], gs=gs, ir=ir,
                                last_tvt=float(last["TVT_input"]), last_Z=float(last["Z"]),
                                last_MD=float(last["MD"]), anchor=anchor)))
    return items, pd.concat(out_frames, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[exp027] vectorized PF full-773  S={S} N={N} cfg(ispr={INIT_SPREAD},pn={PN})")
    items, out = build("data/processed/train_base_v001.parquet",
                       "data/processed/typewell_train_base_v001.parquet")
    pred_by_wid = {}
    t0 = time.time()
    for k, (wid, p) in enumerate(items):
        if p is None:
            pred_by_wid[wid] = None
        else:
            pred_by_wid[wid] = pf_well_vec(p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
                                           p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"])
        if (k + 1) % 150 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(items)} wells  ({el:.0f}s, {el/(k+1):.2f}s/well)", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        anc = float(g["last_known_TVT"].iloc[0])
        pr = pred_by_wid[wid]
        pred_col[g.index.to_numpy()] = anc if pr is None else pr
    out["pred_tvt"] = pred_col
    cv = tvt_rmse(out["TVT"], out["pred_tvt"])
    el = time.time() - t0
    print(f"\n[exp027] vectorized PF CV = {cv:.6f}  (tuned non-vec exp025b=10.983861)  total {el:.0f}s ({el/len(items):.2f}s/well)")
    out.to_csv(OUT_DIR / "oof.csv", index=False)
    write_json(OUT_DIR / "result.json", {
        "exp_id": "exp027_pf_vectorized", "created_at": now_jst(), "status": "completed",
        "cv_rmse": cv, "baseline_nonvec_exp025b": 10.983861, "S": S, "N": N,
        "seconds_total": el, "seconds_per_well": el / len(items),
        "leak_risk": "none", "notes": "Vectorized-across-seeds PF; single-thread; CV equivalence check vs non-vec."})


if __name__ == "__main__":
    main()
