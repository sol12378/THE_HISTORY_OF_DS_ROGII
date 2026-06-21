#!/usr/bin/env python3
"""exp027b: numba-JIT Particle Filter — full-773 CV (高速版, 並列可)。

PFの (seed × 行 × 粒子) ループを numba @njit で native compile。
np.random(legacy)/np.interp/np.searchsorted/np.cumsum を njit内で使用。
RNGは default_rng と異なるが同一アルゴリズム → CVが tuned PF(10.984) と
seedノイズ内で一致するかを実測。config = tuned(init_spread=4, pn=0.01)。完全leak-free。
Kaggle にも numba はプリインストールなので kernel でも同じJITが使える。
"""
from __future__ import annotations
import sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from numba import njit

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

OUT_DIR = Path("experiments") / "exp027b_pf_numba"
S = 128; N = 500; SCALE = 8.0
INIT_SPREAD = 4.0; PN = 0.01; VN = 0.002; MOM = 0.998; RP = 0.1; RR = 0.001; RESAMP = 0.5
N_WORKERS = max(1, min(9, (os.cpu_count() or 4) - 1))


@njit(cache=True, fastmath=True)
def _pf_well_njit(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD,
                  S, N, INIT_SPREAD, PN, VN, MOM, RP, RR, RESAMP, SCALE):
    n = md_v.shape[0]
    res = np.zeros((S, n))
    log_lik = np.zeros(S)
    lo = tw_tvt[0] - 100.0; hi = tw_tvt[-1] + 100.0
    thr = RESAMP * N
    for s in range(S):
        np.random.seed(s)
        pos = (last_tvt + last_Z) + INIT_SPREAD * np.random.standard_normal(N)
        rate = ir + 0.01 * np.random.standard_normal(N)
        w = np.full(N, 1.0 / N)
        prev_MD = last_MD
        ll = 0.0
        for i in range(n):
            dm = md_v[i] - prev_MD
            if dm < 1.0:
                dm = 1.0
            rn1 = np.random.standard_normal(N)
            rn2 = np.random.standard_normal(N)
            for j in range(N):
                rate[j] = MOM * rate[j] + VN * rn1[j]
                p = pos[j] + rate[j] * dm + PN * rn2[j]
                tp = p - z_v[i]
                if tp < lo:
                    tp = lo
                elif tp > hi:
                    tp = hi
                pos[j] = tp + z_v[i]
            tvt_p = pos - z_v[i]
            eg = np.interp(tvt_p, tw_tvt, tw_gr)
            avg = 0.0
            for j in range(N):
                dd = (gr_v[i] - eg[j]) / gs
                e = dd * dd
                if e > 600.0:
                    e = 600.0
                lk = np.exp(-0.5 * e)
                if lk < 1e-300:
                    lk = 1e-300
                avg += w[j] * lk
                w[j] = w[j] * lk
            if avg < 1e-300:
                avg = 1e-300
            ll += np.log(avg)
            sw = 0.0
            for j in range(N):
                sw += w[j]
            if sw <= 0.0:
                for j in range(N):
                    w[j] = 1.0 / N
            else:
                for j in range(N):
                    w[j] = w[j] / sw
            s2 = 0.0
            for j in range(N):
                s2 += w[j] * w[j]
            neff = 1.0 / s2
            if neff < thr:
                cum = np.cumsum(w)
                u0 = np.random.uniform(0.0, 1.0 / N)
                jr1 = np.random.standard_normal(N)
                jr2 = np.random.standard_normal(N)
                new_pos = np.empty(N)
                new_rate = np.empty(N)
                for j in range(N):
                    u = u0 + j / N
                    idx = np.searchsorted(cum, u)
                    if idx >= N:
                        idx = N - 1
                    new_pos[j] = pos[idx] + RP * jr1[j]
                    new_rate[j] = rate[idx] + RR * jr2[j]
                for j in range(N):
                    pos[j] = new_pos[j]
                    rate[j] = new_rate[j]
                    w[j] = 1.0 / N
            acc = 0.0
            for j in range(N):
                acc += w[j] * (pos[j] - z_v[i])
            res[s, i] = acc
            prev_MD = md_v[i]
        log_lik[s] = ll
    # likelihood-weighted seed ensemble
    mx = log_lik[0]
    for s in range(1, S):
        if log_lik[s] > mx:
            mx = log_lik[s]
    wsum = 0.0
    wts = np.empty(S)
    for s in range(S):
        wts[s] = np.exp((log_lik[s] - mx) / SCALE)
        wsum += wts[s]
    out = np.zeros(n)
    for s in range(S):
        ws = wts[s] / wsum
        for i in range(n):
            out[i] += ws * res[s, i]
    return out


def pf_well(p):
    return _pf_well_njit(p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
                         p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"],
                         S, N, INIT_SPREAD, PN, VN, MOM, RP, RR, RESAMP, SCALE)


def _worker(item):
    wid, p = item
    if p is None:
        return wid, None
    return wid, pf_well(p)


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
        tw_tvt = np.ascontiguousarray(tw_s["TVT"].to_numpy(float))
        tw_gr = np.ascontiguousarray(tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float))
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
        k_tvt = known["TVT_input"].to_numpy(float); k_gr = known["GR"].fillna(0).to_numpy(float)
        gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10., 60.))
        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
        last = known.iloc[-1]
        items.append((wid, dict(tw_tvt=tw_tvt, tw_gr=tw_gr,
                                md_v=np.ascontiguousarray(tgt["MD"].to_numpy(float)),
                                z_v=np.ascontiguousarray(tgt["Z"].to_numpy(float)),
                                gr_v=np.ascontiguousarray(gr_full[tgt_mask]),
                                gs=gs, ir=ir, last_tvt=float(last["TVT_input"]),
                                last_Z=float(last["Z"]), last_MD=float(last["MD"]), anchor=anchor)))
    return items, pd.concat(out_frames, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[exp027b] numba-JIT PF full-773  S={S} N={N}  workers={N_WORKERS}", flush=True)
    items, out = build("data/processed/train_base_v001.parquet",
                       "data/processed/typewell_train_base_v001.parquet")
    # warm-up JIT compile on first non-None well
    t0 = time.time()
    for wid, p in items:
        if p is not None:
            pf_well(p); break
    print(f"  JIT compiled in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    pred_by_wid = {}
    # 単一プロセス: numba njitは単独で高速。ProcessPoolExecutorはnumbaコンパイル/cacheロック競合で
    # macOS spawn時にデッドロックするため使わない。
    for k, (wid, p) in enumerate(items):
        pred_by_wid[wid] = None if p is None else pf_well(p)
        if (k + 1) % 150 == 0:
            el = time.time() - t1
            print(f"  {k+1}/{len(items)} wells ({el:.0f}s, {el/(k+1):.3f}s/well)", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        anc = float(g["last_known_TVT"].iloc[0]); pr = pred_by_wid[wid]
        pred_col[g.index.to_numpy()] = anc if pr is None else pr
    out["pred_tvt"] = pred_col
    cv = tvt_rmse(out["TVT"], out["pred_tvt"])
    el = time.time() - t1
    print(f"\n[exp027b] numba PF CV = {cv:.6f}  (tuned non-vec=10.983861)  run {el:.0f}s ({el/len(items):.3f}s/well, single-proc)", flush=True)
    out.to_csv(OUT_DIR / "oof.csv", index=False)
    write_json(OUT_DIR / "result.json", {
        "exp_id": "exp027b_pf_numba", "created_at": now_jst(), "status": "completed",
        "cv_rmse": cv, "baseline_nonvec": 10.983861, "S": S, "N": N,
        "run_seconds": el, "seconds_per_well": el / len(items), "workers": 1,
        "leak_risk": "none", "notes": "numba @njit PF (single-proc); CV equivalence check vs non-JIT."})


if __name__ == "__main__":
    main()
