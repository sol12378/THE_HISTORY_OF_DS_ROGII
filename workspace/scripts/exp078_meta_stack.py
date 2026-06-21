#!/usr/bin/env python
"""exp078 loop#5: learned per-well reweighting of OUR transferable components (pf, geom).

Tests whether a GBM meta-stacker conditioned on leak-free per-well/per-row features
(disagreement |pf-geom|, z_span, gr_std, md_since, row_frac) beats the fixed-weight
blend (exp026 = 0.683*pf + 0.392*geom). Nested GroupKFold by well = honest OOF.
NO borrowed external OOFs -> transfer should be honest (unlike exp073).

Baselines:
  - pf alone, geom alone
  - global NNLS(pf, geom)
  - exp026 recipe anchor + 0.683*(pf-a) + 0.392*(geom-a)
vs meta-stack (LGB on [pf_d, geom_d, features]).
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
EXP = ROOT / "experiments/exp073_public_assets_integration"
OUT = ROOT / "experiments/exp078_meta_stack"; OUT.mkdir(parents=True, exist_ok=True)
TB = ROOT / "data/processed/train_base_v001.parquet"
N_FOLDS = 5


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    t0 = time.time()
    pf = pd.read_csv(EXP/"oof_pf.csv", usecols=["id", "well_id", "pred_tvt"]).rename(columns={"pred_tvt": "pf"})
    geom = pd.read_csv(EXP/"oof_geom.csv", usecols=["id", "pred_tvt"]).rename(columns={"pred_tvt": "geom"})
    df = pf.merge(geom, on="id")
    tb = pd.read_parquet(TB, columns=["well_id", "row_idx", "MD", "Z", "GR", "is_known_tvt",
                                       "TVT", "last_known_TVT", "last_known_MD"])
    tb["id"] = tb["well_id"] + "_" + tb["row_idx"].astype(str)
    hid = tb[~tb["is_known_tvt"]].copy()
    df = df.merge(hid[["id", "MD", "Z", "GR", "TVT", "last_known_TVT", "last_known_MD"]], on="id")
    log(f"rows {len(df):,}")

    a = df["last_known_TVT"].to_numpy()
    df["pf_d"] = df["pf"] - a
    df["geom_d"] = df["geom"] - a
    df["y"] = df["TVT"].to_numpy() - a
    y = df["y"].to_numpy(); groups = df["well_id"].to_numpy()

    # leak-free features (no truth)
    df["disagree"] = np.abs(df["pf_d"] - df["geom_d"])
    df["md_since"] = np.log1p(np.clip(df["MD"] - df["last_known_MD"], 0, None))
    g = df.groupby("well_id", sort=False)
    df["dis_mean"] = g["disagree"].transform("mean")
    df["dis_std"] = g["disagree"].transform("std").fillna(0.0)
    df["z_span"] = g["Z"].transform(lambda s: s.max() - s.min())
    df["gr_std"] = g["GR"].transform("std").fillna(0.0)
    df["n_hidden"] = g["pf"].transform("size").astype(float)
    df["row_frac"] = g.cumcount() / df["n_hidden"].clip(lower=1)

    feats = ["pf_d", "geom_d", "disagree", "md_since", "Z", "GR",
             "dis_mean", "dis_std", "z_span", "gr_std", "n_hidden", "row_frac"]
    X = df[feats].to_numpy(np.float32)

    # baselines (pooled, in TVT space == delta space)
    log("=== baselines (pooled RMSE) ===")
    log(f"  pf alone:    {rmse(df['pf_d'], y):.4f}")
    log(f"  geom alone:  {rmse(df['geom_d'], y):.4f}")
    log(f"  exp026 0.683/0.392: {rmse(0.683*df['pf_d']+0.392*df['geom_d'], y):.4f}")

    # global NNLS(pf,geom) nested
    gkf = GroupKFold(N_FOLDS)
    nn_pred = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        Xtr = df[["pf_d", "geom_d"]].to_numpy()[tr]
        w, _ = nnls(Xtr, y[tr]); nn_pred[te] = df[["pf_d", "geom_d"]].to_numpy()[te] @ w
    log(f"  global NNLS(pf,geom) nested: {rmse(nn_pred, y):.4f}")

    # meta-stack: LGB nested
    log("=== meta-stack LGB nested ===")
    params = dict(objective="regression", num_leaves=63, learning_rate=0.05,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  min_child_samples=200, num_threads=14, verbose=-1)
    ms_pred = np.zeros(len(y)); fold_rmse = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMRegressor(**params, n_estimators=600)
        m.fit(X[tr], y[tr], eval_set=[(X[te], y[te])], eval_metric="rmse",
              callbacks=[lgb.early_stopping(40, verbose=False)])
        bi = int(m.best_iteration_ or 600)
        ms_pred[te] = m.predict(X[te], num_iteration=bi)
        fr = rmse(ms_pred[te], y[te]); fold_rmse.append(fr)
        log(f"  fold{k}: rmse={fr:.4f} (best_iter {bi})")
        if k == 0:
            imp = dict(zip(feats, m.feature_importances_.tolist()))
    ms = rmse(ms_pred, y)
    log(f"  meta-stack pooled: {ms:.4f}  folds={[round(f,3) for f in fold_rmse]}")
    log(f"  feature_importance(fold0): {sorted(imp.items(), key=lambda kv:-kv[1])}")

    exp026 = rmse(0.683*df['pf_d']+0.392*df['geom_d'], y)
    res = {"pf": rmse(df['pf_d'], y), "geom": rmse(df['geom_d'], y),
           "exp026_fixed": exp026, "global_nnls": rmse(nn_pred, y),
           "meta_stack": ms, "meta_folds": fold_rmse,
           "gain_vs_exp026": exp026 - ms, "importance": imp,
           "runtime_min": (time.time()-t0)/60}
    json.dump(res, open(OUT/"result.json", "w"), indent=2)
    log(f"\nGAIN vs exp026 fixed: {exp026-ms:+.4f} (meta {ms:.4f} vs exp026 {exp026:.4f})")
    log(f"done {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
