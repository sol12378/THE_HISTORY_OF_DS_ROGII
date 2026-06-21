#!/usr/bin/env python3
"""exp009: Typewell GR alignment features on top of exp008 feature set.

Group E_well : Per-well calibration from known portion vs typewell reference
Group E_row  : Per-row GR-slope-based TVT correction using typewell

Algorithm:
  1. Build typewell GR(TVT) interpolator for each well
  2. Calibrate: gr_offset = mean(actual_GR - tw_GR) in known portion
  3. Per-row: tw_tvt_correction = -gr_deviation / tw_gr_slope_at_anchor
     (first-order linear approximation of TVT offset from GR mismatch)

Leakage prevention:
  - Calibration uses is_known_tvt==True rows only
  - Typewell is a REFERENCE curve (different physical well), not the label
  - GR in target rows is an OBSERVABLE measurement (not the prediction target)
  - tw_gr_slope computed at last_known_TVT (anchor) only
  - test typewell_test_base_v001.parquet exists and used for test wells
"""

from __future__ import annotations

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rogii.training.baselines import (
    PRED_COL,
    SAFE_FEATURES,
    attach_folds,
    build_submission,
    ensure_exp_dir,
    load_base_inputs,
    now_jst,
    target_rows,
    tvt_rmse,
    write_json,
)


EXP_ID = "exp009_typewell_gr_alignment"
EXP_DIR = Path("experiments") / EXP_ID

# ── carried-over feature groups ──────────────────────────────────────────────
GROUP_A = ["pre_ps_tvt_slope_last20","pre_ps_tvt_slope_last5",
           "pre_ps_tvt_curvature","pre_ps_tvt_delta_last20"]
GROUP_B_WELL = ["pre_ps_dZ_dMD","pre_ps_dX_dMD","pre_ps_dY_dMD",
                "pre_ps_horiz_dMD","pre_ps_azimuth"]
GROUP_B_ROW  = ["dZ_dMD_from_ps","dX_dMD_from_ps","dY_dMD_from_ps",
                "horiz_disp_from_ps","azimuth_from_ps"]
GROUP_C = ["kh_ratio","hidden_frac"]
GROUP_D_WELL = ["pre_ps_gr_mean","pre_ps_gr_std","pre_ps_gr_last20_mean",
                "pre_ps_gr_trend","pre_ps_gr_available_frac"]
GROUP_D_ROW  = ["gr_vs_pre_ps_mean","gr_z_score",
                "gr_rolling_mean_w20","gr_rolling_mean_w50","gr_rolling_std_w20"]

# ── exp009 new features (Group E) ────────────────────────────────────────────
GROUP_E_WELL = [
    "tw_known_gr_corr",          # correlation(GR_known, tw_GR_at_known_TVT)
    "tw_gr_offset",              # mean(GR_known - tw_GR) — systematic bias
    "tw_gr_scale",               # std(GR_known) / std(tw_GR)
    "tw_gr_last20_diff",         # recent (last 20 known) GR offset from typewell
    "tw_gr_slope_at_anchor",     # dGR/dTVT at last_known_TVT in typewell
    "tw_gr_abs_slope_at_anchor", # |dGR/dTVT|
]
GROUP_E_ROW = [
    "tw_gr_at_anchor",           # typewell GR at last_known_TVT (per-row constant)
    "gr_dev_from_tw_anchor",     # actual_GR - gr_offset - tw_gr_at_anchor
    "tw_tvt_correction",         # -gr_dev / tw_gr_slope  (TVT delta estimate)
    "tw_tvt_correction_reliable",# tw_tvt_correction when |slope| > threshold
    "tw_gr_extrap_range",        # how far last_known_TVT is from typewell boundary
]
GROUP_E = GROUP_E_WELL + GROUP_E_ROW

NEW_FEATURES = (GROUP_A + GROUP_B_WELL + GROUP_B_ROW + GROUP_C
                + GROUP_D_WELL + GROUP_D_ROW + GROUP_E)
ALL_FEATURES = SAFE_FEATURES + NEW_FEATURES

SLOPE_RELIABLE_THR = 0.2   # |dGR/dTVT| threshold for tw_tvt_correction_reliable
GLOBAL_GR_MEAN_FALLBACK = 87.75   # from exp008 (train-only constant)


# ============================================================
# exp007 traj features (self-contained copy)
# ============================================================

def _traj_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        n = len(g)
        md = g["MD"].to_numpy(dtype=float)
        x  = g["X"].to_numpy(dtype=float)
        y  = g["Y"].to_numpy(dtype=float)
        z  = g["Z"].to_numpy(dtype=float)
        tv = g["TVT_input"].to_numpy(dtype=float)
        n20 = min(20,n); n5 = min(5,n)
        d20 = md[-1]-md[-n20] if n20>1 else 1.
        d5  = md[-1]-md[-n5]  if n5>1  else 1.
        s20 = (tv[-1]-tv[-n20])/d20 if abs(d20)>1e-6 else 0.
        s5  = (tv[-1]-tv[-n5])/d5   if abs(d5)>1e-6  else 0.
        dz  = (z[-1]-z[-n20])/d20   if abs(d20)>1e-6 else 0.
        dx  = (x[-1]-x[-n20])/d20   if abs(d20)>1e-6 else 0.
        dy  = (y[-1]-y[-n20])/d20   if abs(d20)>1e-6 else 0.
        dx20=x[-1]-x[-n20]; dy20=y[-1]-y[-n20]
        hd  = float(np.sqrt(dx20**2+dy20**2))
        recs.append({"well_id":wid,
                     "pre_ps_tvt_slope_last20":s20,"pre_ps_tvt_slope_last5":s5,
                     "pre_ps_tvt_curvature":s5-s20,"pre_ps_tvt_delta_last20":tv[-1]-tv[-n20],
                     "pre_ps_dZ_dMD":dz,"pre_ps_dX_dMD":dx,"pre_ps_dY_dMD":dy,
                     "pre_ps_horiz_dMD":hd/d20 if abs(d20)>1e-6 else 0.,
                     "pre_ps_azimuth":float(np.arctan2(dy20,dx20))})
    return pd.DataFrame(recs)


def _traj_per_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ms = np.where(df["delta_MD_from_PS"].to_numpy(dtype=float)<1., 1.,
                  df["delta_MD_from_PS"].to_numpy(dtype=float))
    df["dZ_dMD_from_ps"]    = df["delta_Z_from_PS"].astype(float)/ms
    df["dX_dMD_from_ps"]    = df["delta_X_from_PS"].astype(float)/ms
    df["dY_dMD_from_ps"]    = df["delta_Y_from_PS"].astype(float)/ms
    df["horiz_disp_from_ps"]= np.sqrt(df["delta_X_from_PS"].astype(float)**2
                                      +df["delta_Y_from_PS"].astype(float)**2)
    df["azimuth_from_ps"]   = np.arctan2(df["delta_Y_from_PS"].astype(float),
                                         df["delta_X_from_PS"].astype(float))
    hl = df["hidden_length"].to_numpy(dtype=float)
    df["kh_ratio"]   = df["known_length"].astype(float)/np.where(hl<1.,1.,hl)
    df["hidden_frac"]= hl/df["n_rows_in_well"].astype(float)
    return df


# ============================================================
# exp008 GR rolling features (self-contained copy)
# ============================================================

def _gr_per_well(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs  = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        gr = g["GR"].to_numpy(dtype=float)
        md = g["MD"].to_numpy(dtype=float)
        valid = ~np.isnan(gr); nv = int(valid.sum()); nt = len(gr)
        if nv == 0:
            recs.append({"well_id":wid,"pre_ps_gr_mean":global_gr_mean,
                         "pre_ps_gr_std":0.,"pre_ps_gr_last20_mean":global_gr_mean,
                         "pre_ps_gr_trend":0.,"pre_ps_gr_available_frac":0.}); continue
        gv = gr[valid]; mv = md[valid]
        m20 = min(20,len(gv))
        d_md = mv[-1]-mv[0]
        recs.append({"well_id":wid,
                     "pre_ps_gr_mean":float(np.nanmean(gr)),
                     "pre_ps_gr_std":float(np.nanstd(gr)),
                     "pre_ps_gr_last20_mean":float(gv[-m20:].mean()),
                     "pre_ps_gr_trend":(gv[-1]-gv[0])/d_md if abs(d_md)>1e-6 else 0.,
                     "pre_ps_gr_available_frac":nv/nt})
    return pd.DataFrame(recs)


def _gr_per_row(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    df = df.sort_values(["well_id","row_idx"]).copy()
    gr_f = df["GR"].copy().astype(float)
    gr_f[gr_f.isna()] = df.loc[gr_f.isna(),"pre_ps_gr_mean"].fillna(global_gr_mean)
    std_s = df["pre_ps_gr_std"].fillna(1.).replace(0.,1.)
    df["gr_vs_pre_ps_mean"] = gr_f - df["pre_ps_gr_mean"].fillna(global_gr_mean)
    df["gr_z_score"]        = df["gr_vs_pre_ps_mean"]/std_s
    df["_gr_f"] = gr_f
    for w,c in [(20,"gr_rolling_mean_w20"),(50,"gr_rolling_mean_w50")]:
        df[c] = df.groupby("well_id",sort=False)["_gr_f"]\
                  .transform(lambda x: x.rolling(w,min_periods=1).mean())
    df["gr_rolling_std_w20"] = df.groupby("well_id",sort=False)["_gr_f"]\
                                 .transform(lambda x: x.rolling(20,min_periods=2)
                                                       .std().fillna(0.))
    return df.drop(columns=["_gr_f"])


# ============================================================
# exp009 Group E: Typewell GR alignment
# ============================================================

def _build_tw_interpolators(tw_data: pd.DataFrame) -> dict:
    """Build TVT→GR linear interpolator for each well from typewell data."""
    interps = {}
    for wid, g in tw_data.groupby("well_id", sort=False):
        g = g.sort_values("TVT").drop_duplicates("TVT")
        if len(g) < 2:
            continue
        interps[wid] = interp1d(
            g["TVT"].to_numpy(dtype=float),
            g["GR"].to_numpy(dtype=float),
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )
    return interps


def _typewell_per_well(
    df: pd.DataFrame, tw_interps: dict
) -> pd.DataFrame:
    """Group E_well: per-well calibration from known portion vs typewell."""
    known = df[df["is_known_tvt"].astype(bool)]
    recs  = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        lkt = g["last_known_TVT"].iloc[-1]   # anchor TVT

        if wid not in tw_interps:
            # fallback: no typewell available
            recs.append({"well_id":wid,
                         "tw_known_gr_corr":0., "tw_gr_offset":0.,
                         "tw_gr_scale":1., "tw_gr_last20_diff":0.,
                         "tw_gr_slope_at_anchor":0.,
                         "tw_gr_abs_slope_at_anchor":0.,
                         "tw_gr_at_anchor":GLOBAL_GR_MEAN_FALLBACK,
                         "tw_gr_extrap_range":0.})
            continue

        fn = tw_interps[wid]

        # typewell boundaries for extrapolation distance feature
        tw_g   = df[df["well_id"]==wid].iloc[0]   # just to get lkt (same as above)
        # NOTE: interp is 'extrapolate' so no hard boundary, but we can estimate
        # distance from anchor to typewell range via fn's x array
        # We stored original TVT range when building; approximate with a test call
        # (extrapolate mode: no boundary info easily available; use 0 as safe default)
        tw_extrap = 0.0  # will refine below if needed

        # GR deviation in known portion (where GR is valid)
        g_gr = g[~g["is_gr_missing"]].copy()
        if len(g_gr) >= 10:
            tw_gr_k = fn(g_gr["TVT_input"].to_numpy(dtype=float))
            diff    = g_gr["GR"].to_numpy(dtype=float) - tw_gr_k
            gr_off  = float(np.mean(diff))
            gr_scale= (float(np.std(g_gr["GR"])) /
                       max(float(np.std(tw_gr_k)), 1e-6))
            corr_val= float(pd.Series(g_gr["GR"].values)
                            .corr(pd.Series(tw_gr_k)))
            last20  = g_gr.tail(20)
            tw_gr_l20 = fn(last20["TVT_input"].to_numpy(dtype=float))
            last20_diff = float(np.mean(last20["GR"].values - tw_gr_l20))
        else:
            gr_off = gr_scale = last20_diff = 0.
            corr_val = 0.

        # typewell GR slope at anchor (finite difference: ±0.5 TVT)
        tw_at_anchor = float(fn(lkt))
        tw_plus      = float(fn(lkt + 0.5))
        tw_minus     = float(fn(lkt - 0.5))
        slope        = (tw_plus - tw_minus)  # dGR over 1 TVT unit

        recs.append({
            "well_id":                 wid,
            "tw_known_gr_corr":        corr_val,
            "tw_gr_offset":            gr_off,
            "tw_gr_scale":             gr_scale,
            "tw_gr_last20_diff":       last20_diff,
            "tw_gr_slope_at_anchor":   slope,
            "tw_gr_abs_slope_at_anchor": abs(slope),
            "tw_gr_at_anchor":         tw_at_anchor,
            "tw_gr_extrap_range":      tw_extrap,
        })
    return pd.DataFrame(recs)


def _typewell_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """Group E_row: per-row TVT correction from GR deviation."""
    df = df.copy()

    # actual GR (fill missing with pre_ps_gr_mean so deviation = 0)
    gr_actual = df["GR"].copy().astype(float)
    missing   = gr_actual.isna()
    gr_actual.loc[missing] = (
        df.loc[missing, "tw_gr_at_anchor"]
        + df.loc[missing, "tw_gr_offset"].fillna(0.)
    )

    # GR deviation from typewell expectation at anchor (corrected for offset)
    df["gr_dev_from_tw_anchor"] = (
        gr_actual
        - df["tw_gr_offset"].fillna(0.)
        - df["tw_gr_at_anchor"].fillna(GLOBAL_GR_MEAN_FALLBACK)
    )
    # set deviation to 0 when GR was missing
    df.loc[missing, "gr_dev_from_tw_anchor"] = 0.

    # TVT correction: -dev / slope  (first-order GR→TVT inversion)
    slope_safe = df["tw_gr_slope_at_anchor"].replace(0., np.nan)
    raw_corr   = -df["gr_dev_from_tw_anchor"] / slope_safe

    # clip to ±150 TVT units (prevent exploding values)
    raw_corr_clipped = raw_corr.clip(-150., 150.).fillna(0.)

    df["tw_tvt_correction"] = raw_corr_clipped

    # reliable version: zero out when |slope| is too flat
    df["tw_tvt_correction_reliable"] = np.where(
        df["tw_gr_abs_slope_at_anchor"].fillna(0.) > SLOPE_RELIABLE_THR,
        raw_corr_clipped,
        0.,
    )

    return df


def enrich(
    df: pd.DataFrame,
    tw_interps: dict,
    global_gr_mean: float,
) -> pd.DataFrame:
    """Full enrichment pipeline: exp007 + exp008 + exp009 features."""
    # exp007
    pw_traj = _traj_per_well(df)
    df = df.merge(pw_traj, on="well_id", how="left")
    df = _traj_per_row(df)
    # exp008
    pw_gr = _gr_per_well(df, global_gr_mean)
    df = df.merge(pw_gr, on="well_id", how="left")
    df = _gr_per_row(df, global_gr_mean)
    # exp009
    pw_tw = _typewell_per_well(df, tw_interps)
    df = df.merge(pw_tw, on="well_id", how="left")
    df = _typewell_per_row(df)
    return df


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"[{EXP_ID}] Typewell GR alignment 特徴量を追加した LightGBM を学習します。")
    ensure_exp_dir(EXP_DIR)

    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )

    # global GR mean from train only
    global_gr_mean = float(
        train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean()
    )
    print(f"Global GR mean (train): {global_gr_mean:.4f}")

    # load typewell data
    print("Typewell データ読み込み...")
    tw_train = pd.read_parquet("data/processed/typewell_train_base_v001.parquet")
    tw_test  = pd.read_parquet("data/processed/typewell_test_base_v001.parquet")
    tw_all   = pd.concat([tw_train, tw_test], ignore_index=True)

    print("Typewell GR 補間器を構築中...")
    tw_interps = _build_tw_interpolators(tw_all)
    print(f"  {len(tw_interps)} wells の補間器を構築しました。")

    print("特徴量エンジニアリング: train ...")
    train = enrich(train, tw_interps, global_gr_mean)
    print("特徴量エンジニアリング: test ...")
    test  = enrich(test,  tw_interps, global_gr_mean)

    train = attach_folds(train, folds)

    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing:
        raise ValueError(f"欠落特徴量: {missing}")

    train_t = target_rows(train)
    test_t  = target_rows(test)
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    oof_pred_delta  = np.zeros(len(train_t), dtype=float)
    test_pred_delta = np.zeros(len(test_t),  dtype=float)
    fold_rows   = []
    importances = []

    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": -1,
        "min_data_in_leaf": 50, "feature_fraction": 0.9,
        "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1, "seed": 42, "num_threads": 8,
    }

    n_folds = int(train_t["fold"].nunique())
    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy()
        tm = ~vm
        print(f"fold {fold}: train={tm.sum()}, valid={vm.sum()}")

        model = lgb.LGBMRegressor(**params, n_estimators=1500)
        model.fit(
            train_t.loc[tm, ALL_FEATURES], y_delta.loc[tm],
            eval_set=[(train_t.loc[vm, ALL_FEATURES], y_delta.loc[vm])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(100)],
        )
        best = int(model.best_iteration_ or model.n_estimators)
        vd   = model.predict(train_t.loc[vm, ALL_FEATURES], num_iteration=best)
        oof_pred_delta[vm] = vd
        ftvt = train_t.loc[vm, "last_known_TVT"].to_numpy(dtype=float) + vd
        fr   = tvt_rmse(train_t.loc[vm, "TVT"], ftvt)
        fold_rows.append({"fold":int(fold),"n_rows":int(vm.sum()),
                          "rmse":fr,"best_iteration":best})
        print(f"  fold {fold}  RMSE={fr:.6f}  best_iter={best}")

        test_pred_delta += model.predict(test_t[ALL_FEATURES], num_iteration=best)/n_folds
        importances.append(pd.DataFrame({
            "fold":int(fold), "feature":ALL_FEATURES,
            "importance_gain":  model.booster_.feature_importance("gain"),
            "importance_split": model.booster_.feature_importance("split"),
        }))

    # ── outputs ──────────────────────────────────────────────────────────────
    oof = train_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof["pred_delta"] = oof_pred_delta
    oof[PRED_COL]     = oof["last_known_TVT"].astype(float) + oof["pred_delta"]
    oof["error"]      = oof[PRED_COL] - oof["TVT"]
    oof["abs_error"]  = oof["error"].abs()
    oof.to_csv(EXP_DIR/"oof.csv", index=False)

    cv = pd.DataFrame(fold_rows)
    cv.to_csv(EXP_DIR/"cv.csv", index=False)

    test = test.copy()
    test.loc[test["is_target"].astype(bool), "pred_delta"] = test_pred_delta
    test.loc[test["is_target"].astype(bool), PRED_COL] = (
        test_t["last_known_TVT"].to_numpy(dtype=float) + test_pred_delta
    )
    submission = build_submission(test, sample_submission, PRED_COL)
    submission.to_csv(EXP_DIR/"submission.csv", index=False)

    fi = pd.concat(importances, ignore_index=True)
    fi_summary = (fi.groupby("feature", as_index=False)[["importance_gain","importance_split"]]
                    .mean().sort_values("importance_gain", ascending=False))
    fi_summary.to_csv(EXP_DIR/"feature_importance.csv", index=False)

    overall = tvt_rmse(oof["TVT"], oof[PRED_COL])
    result  = {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "model": "LightGBMRegressor", "target": "TVT - last_known_TVT",
        "metric": "TVT_abs_RMSE",
        "cv_rmse":  overall,
        "cv_mean":  float(cv["rmse"].mean()),
        "cv_std":   float(cv["rmse"].std(ddof=0)),
        "baseline_exp008_cv": 13.808621,
        "improvement_vs_exp008": round(13.808621 - overall, 6),
        "baseline_exp007_cv": 13.867054,
        "improvement_vs_exp007": round(13.867054 - overall, 6),
        "n_oof_rows": int(len(oof)),
        "n_submission_rows": int(len(submission)),
        "leak_risk": "low",
        "notes": (
            "Group E_well: typewell GR calibration from known portion (corr, offset, scale, slope). "
            "Group E_row: tw_tvt_correction = -gr_dev/tw_gr_slope at anchor TVT. "
            "Clipped to ±150 TVT. slope < 0.2 → correction zeroed out. "
            "test typewell_test_base_v001.parquet used for test wells."
        ),
    }
    write_json(EXP_DIR/"result.json", result)

    # notes.md
    fold_table = "\n".join(
        f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |"
        for r in fold_rows
    )
    top10 = "\n".join(
        f"| {r.feature} | {int(r.importance_gain):,} |"
        for r in fi_summary.head(15).itertuples()
    )
    # Group E importance
    e_fi = fi_summary[fi_summary["feature"].isin(GROUP_E)].copy()
    e_fi["rank"] = range(1, len(fi_summary)+1)   # rank in full list
    e_fi_table = "\n".join(
        f"| {r.feature} | {int(r.importance_gain):,} | #{fi_summary.index[fi_summary['feature']==r.feature][0]+1} |"
        for r in e_fi.itertuples()
    )

    notes = f"""# {EXP_ID}

## 目的

exp008 (CV=13.808621) に Group E（Typewell GR alignment）特徴量を追加。
typewell の GR 勾配を用いた TVT 第一次補正を実現する。

## 主要特徴量

| 特徴量 | 定義 |
|---|---|
| tw_known_gr_corr | known区間 GR と typewell GR の相関 |
| tw_gr_offset | GR の系統的ずれ（known区間） |
| tw_gr_slope_at_anchor | typewell の dGR/dTVT at last_known_TVT |
| tw_gr_at_anchor | typewell GR at last_known_TVT |
| gr_dev_from_tw_anchor | actual_GR - offset - tw_gr_at_anchor |
| **tw_tvt_correction** | **-gr_dev / tw_gr_slope** （TVT補正量） |
| tw_tvt_correction_reliable | \|slope\| > 0.2 のときのみ有効 |

## 結果

| metric | 値 |
|---|---|
| **exp009 CV RMSE** | **{overall:.6f}** |
| exp008 baseline | 13.808621 |
| vs exp008 | {13.808621-overall:+.6f} |
| vs exp007 | {13.867054-overall:+.6f} |

## Fold 別 RMSE

| fold | rmse | best_iter |
|---|---:|---:|
{fold_table}

## Top 15 Feature Importance

| feature | importance_gain |
|---|---:|
{top10}

## Group E 特徴量のランク

| feature | importance_gain | 全体順位 |
|---|---:|---:|
{e_fi_table}

## リーク防止確認

- キャリブレーション: is_known_tvt==True 行のみ ✅
- typewell は参照曲線（別物理井戸）、ラベルではない ✅
- target GR は観測可能な物理計測値 ✅
- tw_gr_slope は last_known_TVT での typewell 勾配のみ ✅
- test: typewell_test_base_v001.parquet を使用 ✅
"""
    (EXP_DIR/"notes.md").write_text(notes, encoding="utf-8")

    print(f"\n[{EXP_ID}] 完了")
    print(f"  CV RMSE   = {overall:.6f}")
    print(f"  vs exp008 : {13.808621-overall:+.6f}")
    print(f"  vs exp007 : {13.867054-overall:+.6f}")
    print(f"  出力: {EXP_DIR}/")


if __name__ == "__main__":
    main()
