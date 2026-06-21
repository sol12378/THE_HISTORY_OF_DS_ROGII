#!/usr/bin/env python3
"""exp025b: full-773 PF with a tuned config (self-contained, spawn-safe).

importlib経由で別モジュールのworker関数をProcessPoolExecutorに渡すと、macOS spawnで
子プロセスがそのモジュールをimportできずデッドロックする。→ PF関数を本ファイル内に直接定義。

usage: python exp025b_full.py --init_spread 4.0 --pn 0.01 --scale 8 --n_particles 500 --n_seeds 128
出力: experiments/exp025_pf_tuned/{oof.csv,submission.csv,result.json}
"""
from __future__ import annotations
import sys, os, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

OUT_DIR = Path("experiments") / "exp025_pf_tuned"
N_WORKERS = max(1, min(9, (os.cpu_count() or 4) - 1))
MOM = 0.998; RP = 0.1; RR = 0.001; RESAMP = 0.5


def _pf_single(p, seed, cfg):
    tw_tvt = p["tw_tvt"]; tw_gr = p["tw_gr"]; md_v = p["md_v"]; z_v = p["z_v"]; gr_v = p["gr_v"]
    gs = p["gs"]; ir = p["ir"]; n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = cfg["n_particles"]; PN = cfg["pn"]; VN = cfg["vn"]; ISPR = cfg["init_spread"]
    rng = np.random.default_rng(seed)
    pos = (p["last_tvt"] + p["last_Z"]) + ISPR * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N; res = np.empty(n); prev_MD = p["last_MD"]; log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr); d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N); rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    return res, log_lik


def _process_well(args):
    p, cfg = args
    wid = p["wid"]; n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])
    ns = cfg["n_seeds"]; preds = np.empty((ns, n)); liks = np.empty(ns)
    for s in range(ns):
        preds[s], liks[s] = _pf_single(p, s, cfg)
    wts = np.exp((liks - liks.max()) / cfg["scale"]); wts /= wts.sum()
    return wid, (wts[:, None] * preds).sum(0)


def build(base_path, tw_path):
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}
    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    payloads = []; out_frames = []
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
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor, "n_eval": int(len(tgt))})
            continue
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
        payloads.append({"wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
                         "tw_tvt": tw_tvt, "tw_gr": tw_gr,
                         "md_v": tgt["MD"].to_numpy(float), "z_v": tgt["Z"].to_numpy(float), "gr_v": gr_full[tgt_mask],
                         "gs": gs, "ir": ir, "last_tvt": float(last["TVT_input"]),
                         "last_Z": float(last["Z"]), "last_MD": float(last["MD"]), "anchor": anchor})
    return payloads, pd.concat(out_frames, ignore_index=True)


def run_cfg(payloads, out, cfg):
    pred_by_wid = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred) in enumerate(ex.map(_process_well, [(p, cfg) for p in payloads], chunksize=4)):
            pred_by_wid[wid] = pred
            if (k + 1) % 150 == 0:
                print(f"    {k+1}/{len(payloads)} wells", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
    o = out.copy(); o["pred_tvt"] = pred_col
    return o, tvt_rmse(o["TVT"], o["pred_tvt"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init_spread", type=float, default=2.0)
    ap.add_argument("--pn", type=float, default=0.005)
    ap.add_argument("--vn", type=float, default=0.002)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--n_particles", type=int, default=500)
    ap.add_argument("--n_seeds", type=int, default=128)
    a = ap.parse_args()
    cfg = {"n_seeds": a.n_seeds, "n_particles": a.n_particles, "scale": a.scale,
           "init_spread": a.init_spread, "pn": a.pn, "vn": a.vn}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[exp025b] full PF cfg={cfg} workers={N_WORKERS}", flush=True)

    print("TRAIN build ...", flush=True)
    payloads, out = build("data/processed/train_base_v001.parquet",
                          "data/processed/typewell_train_base_v001.parquet")
    print(f"TRAIN run ({len(payloads)} wells) ...", flush=True)
    oof, cv = run_cfg(payloads, out, cfg)
    anc = tvt_rmse(oof["TVT"], oof["last_known_TVT"])
    print(f"  tuned PF CV = {cv:.6f}  (exp022 baseline 11.024014, anchor {anc:.4f})", flush=True)
    oof["error"] = oof["pred_tvt"] - oof["TVT"]; oof["abs_error"] = oof["error"].abs()
    oof.to_csv(OUT_DIR / "oof.csv", index=False)

    print("TEST run ...", flush=True)
    pl_t, out_t = build("data/processed/test_base_v001.parquet",
                        "data/processed/typewell_test_base_v001.parquet")
    test_oof, _ = run_cfg(pl_t, out_t, cfg)
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(test_oof[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
                               on="id", how="left", validate="one_to_one")
    assert not sub["tvt"].isna().any()
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    write_json(OUT_DIR / "result.json", {
        "exp_id": "exp025_pf_tuned", "created_at": now_jst(), "status": "completed",
        "config": cfg, "cv_rmse": cv, "baseline_exp022": 11.024014, "anchor": anc,
        "leak_risk": "none", "notes": "Full-773 PF with tuned config (init_spread=4,pn=0.01); feed into exp026 re-blend."})
    print(f"[exp025b] 完了 CV={cv:.6f} -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
