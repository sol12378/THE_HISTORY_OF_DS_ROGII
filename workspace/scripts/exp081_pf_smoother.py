#!/usr/bin/env python
"""exp081 (Phase B): two-filter PF smoother — subset prototype.

Hypothesis: bad-band wells (8-20ft, 45% of error) drift in the late hidden section
because forward filtering can't use future GR. Run a forward PF (anchored at PS) and
an independent backward PF (reverse MD, diffuse end-prior), combine per-row by
inverse-variance (two-filter smoother). Future GR should correct mid-well drift.
Leak-free: test区間も末端までGR観測あり; hidden TVT never used.

SUBSET: bad+broken (rmse_blend>=8, ~207) + 50 good sample. 48 seeds. Compare
forward-only vs smoothed pooled/per-well RMSE, esp. bad-band median.
"""
import json, time, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
RAW = ROOT / "data/raw/train"
TB = ROOT / "data/processed/train_base_v001.parquet"
OUT = ROOT / "experiments/exp081_pf_smoother"; OUT.mkdir(parents=True, exist_ok=True)
FOREN = ROOT / "experiments/exp073_public_assets_integration/forensics_per_well.csv"
OOF_PF = ROOT / "experiments/exp073_public_assets_integration/oof_pf.csv"

import os
PF_SEEDS = int(os.environ.get("ROGII_SEEDS", "48")); PF_PARTICLES = 500; PF_SCALE = 8.0; PF_INIT_SPREAD = 4.0
FULL_RUN = os.environ.get("ROGII_FULL", "0") == "1"
PF_PN = 0.01; PF_VN = 0.002; PF_MOM = 0.998; PF_RP = 0.1; PF_RR = 0.001; PF_RESAMP = 0.5
BACK_INIT_SPREAD = 60.0   # diffuse end-prior (we don't know end TVT)
MAX_WORKERS = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(str(OUT/"log.txt"), mode="w", encoding="utf-8"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def pf_run(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, init_pos_mean, init_spread, ir, seed):
    """Generic PF over the given (already-ordered) sequence. Returns per-row (mean, var) of TVT."""
    n = len(md_v)
    if n == 0:
        return np.zeros(0), np.ones(0), 0.0
    N = PF_PARTICLES
    rng = np.random.default_rng(seed)
    pos = init_pos_mean + init_spread * rng.standard_normal(N)   # pos = TVT+Z space
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    mean = np.empty(n); var = np.empty(n)
    log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    prev_MD = md_v[0]
    for i in range(n):
        dm = md_v[i] - prev_MD  # signed; for backward pass this is negative
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm + PF_PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.0)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < PF_RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(N)
            rate = rate[idx] + PF_RR * rng.standard_normal(N); w = np.ones(N) / N
        tv = pos - z_v[i]
        m = float(np.dot(w, tv)); mean[i] = m
        var[i] = max(float(np.dot(w, (tv - m) ** 2)), 1e-3)
        prev_MD = md_v[i]
    return mean, var, log_lik


def ensemble(payload, direction):
    """Run PF_SEEDS ensemble in given direction; return per-row (mean,var) likelihood-weighted over seeds."""
    n = payload["n"]
    md, z, gr = payload["md_v"], payload["z_v"], payload["gr_v"]
    if direction == "back":
        md = md[::-1].copy(); z = z[::-1].copy(); gr = gr[::-1].copy()
        init_mean = payload["last_tvt_end"] + z[0]  # diffuse anyway
        init_spread = BACK_INIT_SPREAD; ir = -payload["ir"]
    else:
        init_mean = payload["last_tvt"] + payload["last_Z"]
        init_spread = PF_INIT_SPREAD; ir = payload["ir"]
    means = np.empty((PF_SEEDS, n)); vars = np.empty((PF_SEEDS, n)); liks = np.empty(PF_SEEDS)
    for s in range(PF_SEEDS):
        means[s], vars[s], liks[s] = pf_run(payload["tw_tvt"], payload["tw_gr"], md, z, gr,
                                            payload["gs"], init_mean, init_spread, ir, seed=s + (1000 if direction == "back" else 0))
    wts = np.exp((liks - liks.max()) / PF_SCALE); wts /= wts.sum()
    mean = (wts[:, None] * means).sum(0)
    var = (wts[:, None] * vars).sum(0)
    if direction == "back":
        mean = mean[::-1].copy(); var = var[::-1].copy()
    return mean, var


def worker(p):
    if p.get("no_tw", False):
        return p["wid"], np.full(p["n"], p["anchor"]), np.full(p["n"], p["anchor"]), p["row_idxs"]
    fmean, fvar = ensemble(p, "fwd")
    bmean, bvar = ensemble(p, "back")
    # inverse-variance two-filter combination
    sm = (fmean / fvar + bmean / bvar) / (1.0 / fvar + 1.0 / bvar)
    return p["wid"], fmean, sm, p["row_idxs"]


def build_payload(wid, g):
    g = g.sort_values("row_idx").reset_index(drop=True)
    known = g[g["is_known_tvt"].astype(bool)]; tgt = g[g["is_target"].astype(bool)]
    n = len(tgt); row_idxs = tgt["row_idx"].to_numpy(np.int64)
    anchor = float(tgt["last_known_TVT"].iloc[0]) if n > 0 else 0.0
    tw_path = RAW / f"{wid}__typewell.csv"
    if not tw_path.exists() or len(known) < 2 or n == 0:
        return {"wid": wid, "n": n, "no_tw": True, "anchor": anchor, "row_idxs": row_idxs}
    tw = pd.read_csv(tw_path)[["TVT", "GR"]]
    if len(tw) < 2:
        return {"wid": wid, "n": n, "no_tw": True, "anchor": anchor, "row_idxs": row_idxs}
    tw_s = tw.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float); tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    gr_full = g["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    tm = g["is_target"].astype(bool).to_numpy(); gr_v = gr_full[tm]
    k_tvt = known["TVT_input"].to_numpy(float); k_gr = known["GR"].fillna(0).to_numpy(float)
    gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10.0, 60.0))
    tail = known.tail(30); dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0; ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    last = known.iloc[-1]
    return {"wid": wid, "n": n, "no_tw": False, "anchor": anchor, "row_idxs": row_idxs,
            "tw_tvt": tw_tvt, "tw_gr": tw_gr, "md_v": tgt["MD"].to_numpy(float),
            "z_v": tgt["Z"].to_numpy(float), "gr_v": gr_v, "gs": gs, "ir": ir,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]),
            "last_tvt_end": anchor}  # diffuse, value irrelevant given BACK_INIT_SPREAD


def main():
    t0 = time.time()
    foren = pd.read_csv(FOREN)
    train = pd.read_parquet(TB)
    if FULL_RUN:
        log.info(f"FULL RUN: all {train['well_id'].nunique()} wells, {PF_SEEDS} seeds")
    else:
        bad = foren[foren["rmse_blend"] >= 8.0]["well_id"].tolist()
        good = foren[foren["rmse_blend"] < 8.0]["well_id"].sample(50, random_state=0).tolist()
        subset = set(bad + good)
        log.info(f"subset: {len(bad)} bad+broken + 50 good = {len(subset)}")
        train = train[train["well_id"].isin(subset)]
    truth = train[train["is_target"].astype(bool)][["well_id", "row_idx", "TVT"]].copy()
    truth["id"] = truth["well_id"] + "_" + truth["row_idx"].astype(str)
    payloads = [build_payload(w, g) for w, g in train.groupby("well_id", sort=False)]
    log.info(f"payloads {len(payloads)}; running fwd+back PF {PF_SEEDS} seeds")

    rows = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(worker, p): p["wid"] for p in payloads}
        done = 0
        for fut in as_completed(futs):
            wid, fwd, sm, ri = fut.result()
            for j, rj in enumerate(ri):
                rows.append((f"{wid}_{int(rj)}", float(fwd[j]), float(sm[j])))
            done += 1
            if done % 30 == 0:
                log.info(f"{done}/{len(payloads)} ({(time.time()-t0)/60:.1f}min)")

    df = pd.DataFrame(rows, columns=["id", "fwd", "sm"]).merge(truth[["id", "TVT", "well_id"]], on="id")
    pf128 = pd.read_csv(OOF_PF, usecols=["id", "pred_tvt"]).rename(columns={"pred_tvt": "pf128"})
    df = df.merge(pf128, on="id", how="left")

    def pooled(a): return float(np.sqrt(np.mean((df[a] - df["TVT"]) ** 2)))
    log.info(f"\n=== SUBSET {df['well_id'].nunique()} wells, {len(df)} rows ===")
    log.info(f"  pf128 (full 128seed fwd): {pooled('pf128'):.3f}")
    log.info(f"  fwd  (48seed fwd):        {pooled('fwd'):.3f}")
    log.info(f"  smoother (fwd+back):      {pooled('sm'):.3f}")
    # per-well, bad-band focus
    pw = df.groupby("well_id").apply(lambda g: pd.Series({
        "rmse_fwd": np.sqrt(np.mean((g["fwd"]-g["TVT"])**2)),
        "rmse_sm": np.sqrt(np.mean((g["sm"]-g["TVT"])**2))}), include_groups=False).reset_index()
    pw = pw.merge(foren[["well_id", "rmse_blend"]], on="well_id")
    badm = pw[pw["rmse_blend"] >= 8]
    goodm = pw[pw["rmse_blend"] < 8]
    log.info(f"  bad+broken({len(badm)}): fwd median {badm['rmse_fwd'].median():.2f} -> sm {badm['rmse_sm'].median():.2f} (improved {int((badm['rmse_sm']<badm['rmse_fwd']-0.5).sum())} wells)")
    log.info(f"  good({len(goodm)}): fwd median {goodm['rmse_fwd'].median():.2f} -> sm {goodm['rmse_sm'].median():.2f} (worsened>1ft: {int((goodm['rmse_sm']>goodm['rmse_fwd']+1).sum())})")
    sfx = "_full" if FULL_RUN else ""
    df.to_csv(OUT/f"preds{sfx}.csv", index=False)
    json.dump({"n_wells": int(df['well_id'].nunique()),
               "pooled_pf128": pooled('pf128'), "pooled_fwd": pooled('fwd'), "pooled_smoother": pooled('sm'),
               "bad_median_fwd": float(badm['rmse_fwd'].median()), "bad_median_sm": float(badm['rmse_sm'].median()),
               "bad_improved": int((badm['rmse_sm']<badm['rmse_fwd']-0.5).sum()),
               "good_worsened": int((goodm['rmse_sm']>goodm['rmse_fwd']+1).sum()),
               "runtime_min": (time.time()-t0)/60}, open(OUT/f"result{sfx}.json", "w"), indent=2)
    log.info(f"done {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
