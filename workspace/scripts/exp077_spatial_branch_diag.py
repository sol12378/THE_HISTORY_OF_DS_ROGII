#!/usr/bin/env python
"""exp077 loop#3 (direction change=geometry): can self-excluded spatial formation
surface RANK the per-well offset branch better than GR likelihood (ceiling 0.155)?

Leak-free LOWO: for each query well, build ASTNU surface from OTHER wells only,
offset = median over KNOWN rows of (TVT_input - (S_ASTNU - Z)); predict hidden
TVT = S - Z + offset. Compare to PF and truth on bad+broken wells.

Decisive metrics:
  - surface per-well RMSE on subset
  - on wrong-branch wells (|pf per-well bias|>15): does surface beat pf?
  - corr(surface-implied offset correction, true bias) vs likelihood ceiling 0.155
"""
import json, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
RAW = ROOT / "data/raw/train"
TB = ROOT / "data/processed/train_base_v001.parquet"
OUT = ROOT / "experiments/exp077_spatial_branch"; OUT.mkdir(parents=True, exist_ok=True)
FOREN = ROOT / "experiments/exp073_public_assets_integration/forensics_per_well.csv"
OOF_PF = ROOT / "experiments/exp073_public_assets_integration/oof_pf.csv"
FORM = "ASTNU"; KNN = 8; SUBSAMPLE = 2_000_000


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    foren = pd.read_csv(FOREN)
    subset = set(foren[foren["rmse_blend"] >= 8.0]["well_id"])
    log(f"subset {len(subset)} wells")

    # load ASTNU points from all wells (tree source), tag well_id index
    files = sorted(glob.glob(str(RAW / "*__horizontal_well.csv")))
    XY = []; VAL = []; WID = []
    wid_list = []
    for i, fp in enumerate(files):
        wid = Path(fp).name.split("__")[0]
        try:
            d = pd.read_csv(fp, usecols=["X", "Y", FORM])
        except Exception:
            continue
        d = d.dropna()
        if len(d) == 0:
            continue
        k = len(wid_list); wid_list.append(wid)
        XY.append(d[["X", "Y"]].values); VAL.append(d[FORM].values)
        WID.append(np.full(len(d), k, dtype=np.int32))
    XY = np.vstack(XY); VAL = np.concatenate(VAL); WID = np.concatenate(WID)
    if len(XY) > SUBSAMPLE:
        idx = np.random.RandomState(0).choice(len(XY), SUBSAMPLE, replace=False)
        XY, VAL, WID = XY[idx], VAL[idx], WID[idx]
    wid_to_k = {w: i for i, w in enumerate(wid_list)}
    log(f"formation pts {len(XY):,}, building tree")
    tree = cKDTree(XY)

    train = pd.read_parquet(TB)
    train = train[train["well_id"].isin(subset)]
    pf = pd.read_csv(OOF_PF, usecols=["id", "pred_tvt"]).rename(columns={"pred_tvt": "pf"})

    rows = []
    perwell = []
    for wid, g in train.groupby("well_id", sort=False):
        gk = g.sort_values("row_idx")
        kmask = gk["is_known_tvt"].astype(bool).values
        tmask = gk["is_target"].astype(bool).values
        if kmask.sum() < 3 or tmask.sum() == 0:
            continue
        xy = gk[["X", "Y"]].values; z = gk["Z"].values
        tvt_in = gk["TVT_input"].values; tvt = gk["TVT"].values
        myk = wid_to_k.get(wid, -1)
        # query k+buffer, exclude self-well points
        dist, ind = tree.query(xy, k=KNN + 25)
        s = np.empty(len(xy))
        for r in range(len(xy)):
            jj = ind[r]; keep = WID[jj] != myk
            jj = jj[keep][:KNN]; dd = dist[r][keep][:KNN]
            w = 1.0 / (dd + 1e-6); w /= w.sum()
            s[r] = (w * VAL[jj]).sum()
        offset = np.nanmedian(tvt_in[kmask] - (s[kmask] - z[kmask]))
        pred = s - z + offset
        # eval on hidden
        th = tvt[tmask]; ph = pred[tmask]
        rmse_surf = float(np.sqrt(np.mean((ph - th) ** 2)))
        rid = gk["row_idx"].values[tmask]
        for rj, pv, tv in zip(rid, ph, th):
            rows.append((f"{wid}_{int(rj)}", float(pv), float(tv)))
        perwell.append({"well_id": wid, "rmse_surf": rmse_surf})

    pw = pd.DataFrame(perwell).merge(foren[["well_id", "rmse_blend", "rmse_pf", "rmse_geom", "bias_blend"]], on="well_id")
    df = pd.DataFrame(rows, columns=["id", "surf", "TVT"]).merge(pf, on="id")
    df = df.merge(train.assign(id=train["well_id"]+"_"+train["row_idx"].astype(str))[["id","last_known_TVT"]], on="id", how="left")

    def pooled(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    log(f"\n=== SUBSET {len(pw)} wells ===")
    log(f"  surface pooled RMSE: {pooled(df['surf'], df['TVT']):.3f}")
    log(f"  pf      pooled RMSE: {pooled(df['pf'], df['TVT']):.3f}")
    # per-well: surface vs pf
    surf_better = int((pw["rmse_surf"] < pw["rmse_pf"]).sum())
    log(f"  wells surface<pf: {surf_better}/{len(pw)}")
    # wrong-branch wells (|bias_blend|>15): does surface beat pf there?
    wb = pw[pw["bias_blend"].abs() > 15]
    if len(wb):
        log(f"  wrong-branch wells (|bias|>15): {len(wb)}; surface<pf on {int((wb['rmse_surf']<wb['rmse_pf']).sum())}")
        log(f"    median rmse: surf={wb['rmse_surf'].median():.2f} pf={wb['rmse_pf'].median():.2f} geom={wb['rmse_geom'].median():.2f}")
    # corr: surface-implied per-well mean pred vs pf per-well mean pred vs truth (offset ranking)
    pm = df.groupby(df["id"].str.rsplit("_", n=1).str[0]).agg(
        surf=("surf", "mean"), pf=("pf", "mean"), tvt=("TVT", "mean")).reset_index()
    # true per-well offset vs anchor; correlation of (surf - pf) with (tvt - pf) = does surf point toward truth when pf is off?
    c1 = np.corrcoef(pm["surf"] - pm["pf"], pm["tvt"] - pm["pf"])[0, 1]
    log(f"  corr( surf-pf , true-pf ) = {c1:+.3f}  (>0 means surface points toward truth where pf errs; cf likelihood ceiling 0.155)")
    # blend test: simple 50/50 surf+pf on subset
    log(f"  0.5surf+0.5pf pooled: {pooled(0.5*df['surf']+0.5*df['pf'], df['TVT']):.3f}")

    pw.sort_values("rmse_surf").to_csv(OUT / "spatial_per_well.csv", index=False)
    json.dump({"n_wells": len(pw), "surf_pooled": pooled(df['surf'], df['TVT']),
               "pf_pooled": pooled(df['pf'], df['TVT']),
               "surf_better_than_pf": surf_better,
               "corr_surf_minus_pf_vs_true_minus_pf": float(c1),
               "blend_5050_pooled": pooled(0.5*df['surf']+0.5*df['pf'], df['TVT']),
               "runtime_min": (time.time()-t0)/60}, open(OUT/"result.json", "w"), indent=2)
    log(f"done {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
