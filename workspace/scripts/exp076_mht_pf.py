#!/usr/bin/env python
"""exp076 Phase2 loop#2: multi-restart (MHT-lite) PF on bad+broken wells.

Hypothesis: broken wells are wrong-branch lock (large |bias| 18-43ft). Current PF
init_spread=4ft is too narrow to reach a branch 20-40ft away. Run PF from several
discrete offset restarts {-40,-20,0,+20,+40}ft and pick, per well, the restart with
max total log-likelihood. = implementation of the 4-offset oracle (4.39).

SUBSET TEST: only wells with blend rmse >= 8 (bad+broken ~207). 48 seeds/restart to
keep it fast. Compare multi-restart vs single (offset=0) vs existing oof_pf.
Leak-free: same as exp073 PF (TVT_input/GR/typewell only; hidden TVT only for eval).
"""
import json, time, logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
RAW = ROOT / "data/raw/train"
TB = ROOT / "data/processed/train_base_v001.parquet"
OUT = ROOT / "experiments/exp076_mht_pf"
OUT.mkdir(parents=True, exist_ok=True)
FOREN = ROOT / "experiments/exp073_public_assets_integration/forensics_per_well.csv"
OOF_PF = ROOT / "experiments/exp073_public_assets_integration/oof_pf.csv"

# PF config (identical to exp073) except seeds reduced for subset test
PF_SEEDS = 48
PF_PARTICLES = 500
PF_SCALE = 8.0
PF_INIT_SPREAD = 4.0
PF_PN = 0.01; PF_VN = 0.002; PF_MOM = 0.998; PF_RP = 0.1; PF_RR = 0.001; PF_RESAMP = 0.5
RESTARTS = [-40.0, -20.0, 0.0, 20.0, 40.0]
MAX_WORKERS = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(str(OUT/"log.txt"), mode="w", encoding="utf-8"),
                              logging.StreamHandler()])
log = logging.getLogger(__name__)


def pf_single(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed, init_off):
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = PF_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (last_tvt + init_off + last_Z) + PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n); prev_MD = last_MD; log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PF_PN * rng.standard_normal(N)
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
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    return res, log_lik


def pf_worker(p):
    wid = p["wid"]; n = p["n"]; row_idxs = p["row_idxs"]
    if n == 0 or p.get("no_tw", False):
        return wid, np.full(n, p["anchor"]), row_idxs, 0.0
    best_pred = None; best_ll = -np.inf; best_off = 0.0
    for off in RESTARTS:
        preds = np.empty((PF_SEEDS, n)); liks = np.empty(PF_SEEDS)
        for s in range(PF_SEEDS):
            preds[s], liks[s] = pf_single(p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
                                          p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"], s, off)
        wts = np.exp((liks - liks.max()) / PF_SCALE); wts /= wts.sum()
        pred = (wts[:, None] * preds).sum(0)
        # restart score = max single-seed total loglik (branch evidence)
        score = float(liks.max())
        if off == 0.0:
            pred0 = pred
        if score > best_ll:
            best_ll = score; best_pred = pred; best_off = off
    return wid, best_pred, row_idxs, best_off, pred0


def build_payload(well_id, well_df):
    g = well_df.sort_values("row_idx").reset_index(drop=True)
    known = g[g["is_known_tvt"].astype(bool)]
    tgt = g[g["is_target"].astype(bool)]
    n = len(tgt); row_idxs = tgt["row_idx"].to_numpy(np.int64)
    anchor = float(tgt["last_known_TVT"].iloc[0]) if n > 0 else 0.0
    tw_path = RAW / f"{well_id}__typewell.csv"
    if not tw_path.exists():
        return {"wid": well_id, "n": n, "no_tw": True, "anchor": anchor, "row_idxs": row_idxs}
    tw = pd.read_csv(tw_path)[["TVT", "GR"]].copy()
    if len(tw) < 2 or len(known) < 2 or n == 0:
        return {"wid": well_id, "n": n, "no_tw": True, "anchor": anchor, "row_idxs": row_idxs}
    tw_s = tw.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float); tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    gr_full = g["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    tm = g["is_target"].astype(bool).to_numpy(); gr_v = gr_full[tm]
    k_tvt = known["TVT_input"].to_numpy(float); k_gr = known["GR"].fillna(0).to_numpy(float)
    gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10.0, 60.0))
    tail = known.tail(30); dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0; ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    last = known.iloc[-1]
    return {"wid": well_id, "n": n, "no_tw": False, "anchor": anchor, "row_idxs": row_idxs,
            "tw_tvt": tw_tvt, "tw_gr": tw_gr, "md_v": tgt["MD"].to_numpy(float),
            "z_v": tgt["Z"].to_numpy(float), "gr_v": gr_v, "gs": gs, "ir": ir,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]), "last_MD": float(last["MD"])}


def main():
    t0 = time.time()
    foren = pd.read_csv(FOREN)
    subset = foren[foren["rmse_blend"] >= 8.0]["well_id"].tolist()
    log.info(f"subset (rmse_blend>=8): {len(subset)} wells")
    train = pd.read_parquet(TB)
    train = train[train["well_id"].isin(subset)]
    truth = train[train["is_target"].astype(bool)][["well_id", "row_idx", "TVT"]].copy()
    truth["id"] = truth["well_id"] + "_" + truth["row_idx"].astype(str)

    payloads = [build_payload(wid, g) for wid, g in train.groupby("well_id", sort=False)]
    log.info(f"payloads {len(payloads)}; running MHT PF {len(RESTARTS)} restarts x {PF_SEEDS} seeds")
    rows_mht = []; rows_single = []; chosen = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(pf_worker, p): p["wid"] for p in payloads}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            wid, pred, ri, off = r[0], r[1], r[2], r[3]
            pred0 = r[4] if len(r) > 4 else pred
            chosen[wid] = off
            for j, rj in enumerate(ri):
                rows_mht.append((f"{wid}_{int(rj)}", float(pred[j])))
                rows_single.append((f"{wid}_{int(rj)}", float(pred0[j])))
            done += 1
            if done % 25 == 0:
                log.info(f"{done}/{len(payloads)} ({(time.time()-t0)/60:.1f}min)")

    mht = pd.DataFrame(rows_mht, columns=["id", "mht"]).merge(truth[["id", "TVT", "well_id"]], on="id")
    sg = pd.DataFrame(rows_single, columns=["id", "single"])
    mht = mht.merge(sg, on="id")
    oof_pf = pd.read_csv(OOF_PF, usecols=["id", "pred_tvt"]).rename(columns={"pred_tvt": "pf128"})
    mht = mht.merge(oof_pf, on="id", how="left")

    def pooled(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    log.info(f"\n=== SUBSET ({mht['well_id'].nunique()} wells, {len(mht)} rows) pooled RMSE ===")
    log.info(f"  existing oof_pf (128seed,1restart): {pooled(mht['pf128'], mht['TVT']):.3f}")
    log.info(f"  this single(48seed,off0):          {pooled(mht['single'], mht['TVT']):.3f}")
    log.info(f"  MHT multi-restart(48seed,5off):    {pooled(mht['mht'], mht['TVT']):.3f}")
    # per-well rescue
    pw = mht.groupby("well_id").apply(lambda g: pd.Series({
        "rmse_single": np.sqrt(np.mean((g["single"]-g["TVT"])**2)),
        "rmse_mht": np.sqrt(np.mean((g["mht"]-g["TVT"])**2))}), include_groups=False)
    rescued = int(((pw["rmse_single"] >= 20) & (pw["rmse_mht"] < 12)).sum())
    worse = int((pw["rmse_mht"] > pw["rmse_single"] + 3).sum())
    log.info(f"  wells rescued (single>=20 -> mht<12): {rescued}")
    log.info(f"  wells worsened (>+3ft): {worse}")
    log.info(f"  chosen offset dist: {pd.Series(chosen).value_counts().to_dict()}")
    mht.to_csv(OUT/"mht_subset_preds.csv", index=False)
    json.dump({"n_wells": int(mht['well_id'].nunique()),
               "pooled_pf128": pooled(mht['pf128'], mht['TVT']),
               "pooled_single48": pooled(mht['single'], mht['TVT']),
               "pooled_mht": pooled(mht['mht'], mht['TVT']),
               "rescued": rescued, "worsened": worse,
               "chosen_offsets": {str(k): int(v) for k, v in pd.Series(chosen).value_counts().items()},
               "runtime_min": (time.time()-t0)/60}, open(OUT/"result.json", "w"), indent=2)
    log.info(f"done {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
