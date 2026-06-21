#!/usr/bin/env python3
"""exp014: Group F geometric trajectory extrapolation features (no typewell).

重要な前提（データ確認済み）:
  hidden(target)区間でも X/Y/Z/MD は完全既知（0% NaN）。隠れているのはTVTのみ。
  → 既知の3D軌跡と、known区間で観測した「TVTと幾何の関係（構造傾斜）」を使えば、
    hidden各行のTVT変化量(delta)を幾何的に外挿できる。

Group F（全て幾何ベース・typewell不使用・hiddenのTVTは一切使わない → leak-free）:
  per-well（known区間から構造傾斜を較正）:
    f_dtvt_dmd_l50 : 直近50 known行の TVT~MD 回帰傾き
    f_dtvt_dz_pre  : known区間の TVT~Z 回帰傾き（構造傾斜の鉛直成分）
    f_dtvt_dz_r2   : 上記回帰の R²（外挿信頼度）
  per-row（既知のhidden幾何へ傾斜を投影 → delta推定）:
    f_extrap_slope20_dMD : pre_ps_tvt_slope_last20 × delta_MD_from_PS  （MD線形外挿delta）
    f_extrap_slope5_dMD  : pre_ps_tvt_slope_last5  × delta_MD_from_PS
    f_extrap_quad_dMD    : slope20·dMD + 0.5·curvature·dMD²           （MD二次外挿delta）
    f_extrap_z           : f_dtvt_dz_pre × delta_Z_from_PS            （Z投影delta）
    f_extrap_disagree    : |f_extrap_slope20_dMD − f_extrap_z|        （外挿の不確実性）

baseline = exp008 (SAFE + A + B + C + D)。Group F を上乗せして CV を測る。
fold は exp008 と同一（folds_group_well_v001.csv）で公平比較。
"""

from __future__ import annotations

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rogii.training.baselines import (
    PRED_COL, SAFE_FEATURES, attach_folds, build_submission,
    ensure_exp_dir, load_base_inputs, now_jst, target_rows, tvt_rmse, write_json,
)

EXP_ID = "exp014_geom_extrap"
EXP_DIR = Path("experiments") / EXP_ID

# ── carried-over feature groups (exp008) ──
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

# ── exp014 new (Group F) ──
GROUP_F_WELL = ["f_dtvt_dmd_l50","f_dtvt_dz_pre","f_dtvt_dz_r2"]
GROUP_F_ROW  = ["f_extrap_slope20_dMD","f_extrap_slope5_dMD","f_extrap_quad_dMD",
                "f_extrap_z","f_extrap_disagree"]
GROUP_F = GROUP_F_WELL + GROUP_F_ROW

NEW_FEATURES = (GROUP_A + GROUP_B_WELL + GROUP_B_ROW + GROUP_C
                + GROUP_D_WELL + GROUP_D_ROW + GROUP_F)
ALL_FEATURES = SAFE_FEATURES + NEW_FEATURES


# ============================================================
# exp007 traj + exp008 GR builders (self-contained copies)
# ============================================================

def _traj_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        n = len(g)
        md = g["MD"].to_numpy(float); x = g["X"].to_numpy(float)
        y = g["Y"].to_numpy(float); z = g["Z"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float)
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
    ms = np.where(df["delta_MD_from_PS"].to_numpy(float)<1.,1.,
                  df["delta_MD_from_PS"].to_numpy(float))
    df["dZ_dMD_from_ps"]    = df["delta_Z_from_PS"].astype(float)/ms
    df["dX_dMD_from_ps"]    = df["delta_X_from_PS"].astype(float)/ms
    df["dY_dMD_from_ps"]    = df["delta_Y_from_PS"].astype(float)/ms
    df["horiz_disp_from_ps"]= np.sqrt(df["delta_X_from_PS"].astype(float)**2
                                      +df["delta_Y_from_PS"].astype(float)**2)
    df["azimuth_from_ps"]   = np.arctan2(df["delta_Y_from_PS"].astype(float),
                                         df["delta_X_from_PS"].astype(float))
    hl = df["hidden_length"].to_numpy(float)
    df["kh_ratio"]   = df["known_length"].astype(float)/np.where(hl<1.,1.,hl)
    df["hidden_frac"]= hl/df["n_rows_in_well"].astype(float)
    return df


def _gr_per_well(df: pd.DataFrame, gmean: float) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        gr = g["GR"].to_numpy(float); md = g["MD"].to_numpy(float)
        valid = ~np.isnan(gr); nv = int(valid.sum()); nt = len(gr)
        if nv == 0:
            recs.append({"well_id":wid,"pre_ps_gr_mean":gmean,"pre_ps_gr_std":0.,
                         "pre_ps_gr_last20_mean":gmean,"pre_ps_gr_trend":0.,
                         "pre_ps_gr_available_frac":0.}); continue
        gv = gr[valid]; mv = md[valid]; m20 = min(20,len(gv)); dmd = mv[-1]-mv[0]
        recs.append({"well_id":wid,"pre_ps_gr_mean":float(np.nanmean(gr)),
                     "pre_ps_gr_std":float(np.nanstd(gr)),
                     "pre_ps_gr_last20_mean":float(gv[-m20:].mean()),
                     "pre_ps_gr_trend":(gv[-1]-gv[0])/dmd if abs(dmd)>1e-6 else 0.,
                     "pre_ps_gr_available_frac":nv/nt})
    return pd.DataFrame(recs)


def _gr_per_row(df: pd.DataFrame, gmean: float) -> pd.DataFrame:
    df = df.sort_values(["well_id","row_idx"]).copy()
    grf = df["GR"].copy().astype(float)
    grf[grf.isna()] = df.loc[grf.isna(),"pre_ps_gr_mean"].fillna(gmean)
    sstd = df["pre_ps_gr_std"].fillna(1.).replace(0.,1.)
    df["gr_vs_pre_ps_mean"] = grf - df["pre_ps_gr_mean"].fillna(gmean)
    df["gr_z_score"]        = df["gr_vs_pre_ps_mean"]/sstd
    df["_grf"] = grf
    for w,c in [(20,"gr_rolling_mean_w20"),(50,"gr_rolling_mean_w50")]:
        df[c]=df.groupby("well_id",sort=False)["_grf"].transform(
            lambda x: x.rolling(w,min_periods=1).mean())
    df["gr_rolling_std_w20"]=df.groupby("well_id",sort=False)["_grf"].transform(
        lambda x: x.rolling(20,min_periods=2).std().fillna(0.))
    return df.drop(columns=["_grf"])


# ============================================================
# exp014 Group F: geometric extrapolation
# ============================================================

def _geom_per_well(df: pd.DataFrame) -> pd.DataFrame:
    """known区間から構造傾斜を較正（TVT~MD, TVT~Z 回帰）。"""
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        md = g["MD"].to_numpy(float); z = g["Z"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float)
        n = len(g)
        # dTVT/dMD over last 50 known points (local dip near PS)
        n50 = min(50, n)
        if n50 >= 2 and abs(md[-1]-md[-n50]) > 1e-6:
            dtvt_dmd_l50 = float(np.polyfit(md[-n50:], tv[-n50:], 1)[0])
        else:
            dtvt_dmd_l50 = 0.0
        # dTVT/dZ regression over all known (Z varies most in build section)
        if n >= 3 and np.ptp(z) > 1e-3:
            A = np.vstack([z, np.ones_like(z)]).T
            coef, res, *_ = np.linalg.lstsq(A, tv, rcond=None)
            slope_z = float(coef[0])
            pred = A @ coef
            ss_res = float(np.sum((tv - pred)**2))
            ss_tot = float(np.sum((tv - tv.mean())**2))
            r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-9 else 0.0
        else:
            slope_z = 0.0; r2 = 0.0
        recs.append({"well_id":wid,"f_dtvt_dmd_l50":dtvt_dmd_l50,
                     "f_dtvt_dz_pre":slope_z,"f_dtvt_dz_r2":r2})
    return pd.DataFrame(recs)


def _geom_per_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dmd = df["delta_MD_from_PS"].astype(float)
    dz  = df["delta_Z_from_PS"].astype(float)
    s20 = df["pre_ps_tvt_slope_last20"].astype(float)
    s5  = df["pre_ps_tvt_slope_last5"].astype(float)
    curv= df["pre_ps_tvt_curvature"].astype(float)
    df["f_extrap_slope20_dMD"] = s20 * dmd
    df["f_extrap_slope5_dMD"]  = s5  * dmd
    df["f_extrap_quad_dMD"]    = s20 * dmd + 0.5 * curv * dmd * dmd
    df["f_extrap_z"]           = df["f_dtvt_dz_pre"].astype(float) * dz
    df["f_extrap_disagree"]    = (df["f_extrap_slope20_dMD"] - df["f_extrap_z"]).abs()
    return df


def enrich(df: pd.DataFrame, gmean: float) -> pd.DataFrame:
    pw = _traj_per_well(df);  df = df.merge(pw, on="well_id", how="left")
    df = _traj_per_row(df)
    gw = _gr_per_well(df, gmean); df = df.merge(gw, on="well_id", how="left")
    df = _gr_per_row(df, gmean)
    fw = _geom_per_well(df);  df = df.merge(fw, on="well_id", how="left")
    df = _geom_per_row(df)
    return df


def main() -> None:
    print(f"[{EXP_ID}] Group F 幾何外挿特徴量を追加して LightGBM を学習します。")
    ensure_exp_dir(EXP_DIR)

    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )
    gmean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    print(f"Global GR mean (train): {gmean:.4f}")

    print("特徴量エンジニアリング: train ..."); train = enrich(train, gmean)
    print("特徴量エンジニアリング: test  ..."); test  = enrich(test,  gmean)
    train = attach_folds(train, folds)

    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing: raise ValueError(f"欠落特徴量: {missing}")

    train_t = target_rows(train); test_t = target_rows(test)
    y = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    oof_p = np.zeros(len(train_t)); test_p = np.zeros(len(test_t))
    fold_rows = []; importances = []
    params = {"objective":"regression","metric":"rmse","learning_rate":0.05,
              "num_leaves":63,"max_depth":-1,"min_data_in_leaf":50,
              "feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":1,
              "lambda_l2":1.0,"verbosity":-1,"seed":42,"num_threads":8}
    n_folds = int(train_t["fold"].nunique())

    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy(); tm = ~vm
        print(f"fold {fold}: train={tm.sum()}, valid={vm.sum()}")
        model = lgb.LGBMRegressor(**params, n_estimators=1500)
        model.fit(train_t.loc[tm, ALL_FEATURES], y.loc[tm],
                  eval_set=[(train_t.loc[vm, ALL_FEATURES], y.loc[vm])],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
        bi = int(model.best_iteration_ or model.n_estimators)
        vd = model.predict(train_t.loc[vm, ALL_FEATURES], num_iteration=bi)
        oof_p[vm] = vd
        ftvt = train_t.loc[vm,"last_known_TVT"].to_numpy(float) + vd
        fr = tvt_rmse(train_t.loc[vm,"TVT"], ftvt)
        fold_rows.append({"fold":int(fold),"n_rows":int(vm.sum()),"rmse":fr,"best_iteration":bi})
        print(f"  fold {fold}  RMSE={fr:.6f}  best_iter={bi}")
        test_p += model.predict(test_t[ALL_FEATURES], num_iteration=bi)/n_folds
        importances.append(pd.DataFrame({"fold":int(fold),"feature":ALL_FEATURES,
            "importance_gain":model.booster_.feature_importance("gain"),
            "importance_split":model.booster_.feature_importance("split")}))

    oof = train_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof["pred_delta"]=oof_p; oof[PRED_COL]=oof["last_known_TVT"].astype(float)+oof["pred_delta"]
    oof["error"]=oof[PRED_COL]-oof["TVT"]; oof["abs_error"]=oof["error"].abs()
    oof.to_csv(EXP_DIR/"oof.csv", index=False)
    cv = pd.DataFrame(fold_rows); cv.to_csv(EXP_DIR/"cv.csv", index=False)

    test = test.copy()
    test.loc[test["is_target"].astype(bool),"pred_delta"]=test_p
    test.loc[test["is_target"].astype(bool),PRED_COL]=(
        test_t["last_known_TVT"].to_numpy(float)+test_p)
    submission = build_submission(test, sample_submission, PRED_COL)
    submission.to_csv(EXP_DIR/"submission.csv", index=False)

    fi = pd.concat(importances, ignore_index=True)
    fi_s = (fi.groupby("feature",as_index=False)[["importance_gain","importance_split"]]
              .mean().sort_values("importance_gain",ascending=False))
    fi_s.to_csv(EXP_DIR/"feature_importance.csv", index=False)

    overall = tvt_rmse(oof["TVT"], oof[PRED_COL])
    result = {"exp_id":EXP_ID,"created_at":now_jst(),"status":"completed",
        "model":"LightGBMRegressor","target":"TVT - last_known_TVT","metric":"TVT_abs_RMSE",
        "cv_rmse":overall,"cv_mean":float(cv["rmse"].mean()),"cv_std":float(cv["rmse"].std(ddof=0)),
        "baseline_exp008_cv":13.808621,"improvement_vs_exp008":round(13.808621-overall,6),
        "feature_groups":{"safe":SAFE_FEATURES,"group_a":GROUP_A,"group_b_well":GROUP_B_WELL,
            "group_b_row":GROUP_B_ROW,"group_c":GROUP_C,"group_d_well":GROUP_D_WELL,
            "group_d_row":GROUP_D_ROW,"group_f_well":GROUP_F_WELL,"group_f_row":GROUP_F_ROW},
        "n_oof_rows":int(len(oof)),"n_submission_rows":int(len(submission)),"leak_risk":"low",
        "notes":("Group F: geometric extrapolation from known structural dip. "
                 "X/Y/Z/MD fully known in hidden region; only TVT hidden. "
                 "No typewell, no hidden-TVT used. Same folds as exp008.")}
    write_json(EXP_DIR/"result.json", result)

    fold_tbl = "\n".join(f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |" for r in fold_rows)
    top = "\n".join(f"| {r.feature} | {int(r.importance_gain):,} |" for r in fi_s.head(15).itertuples())
    f_rank = "\n".join(
        f"| {r.feature} | {int(r.importance_gain):,} | #{list(fi_s['feature']).index(r.feature)+1} |"
        for r in fi_s[fi_s['feature'].isin(GROUP_F)].itertuples())
    notes = f"""# {EXP_ID}

## 目的
exp008 (CV=13.808621) に Group F（幾何外挿）を追加。hidden区間でX/Y/Z/MDは既知なので、
known区間で較正した構造傾斜(dTVT/dMD, dTVT/dZ)を既知hidden幾何へ投影しTVT deltaを外挿する。

## 結果
| metric | 値 |
|---|---|
| **exp014 CV RMSE** | **{overall:.6f}** |
| exp008 baseline | 13.808621 |
| vs exp008 | {13.808621-overall:+.6f} |

## Fold 別
| fold | rmse | best_iter |
|---|---:|---:|
{fold_tbl}

## Top 15 importance
| feature | gain |
|---|---:|
{top}

## Group F のランク
| feature | gain | 全体順位 |
|---|---:|---:|
{f_rank}

## リーク防止
- 構造傾斜の較正は is_known_tvt==True 行のみ ✅
- hidden の X/Y/Z/MD は観測量（予測対象はTVTのみ） ✅
- hidden の TVT は一切使わない ✅
- fold は exp008 と同一（folds_group_well_v001.csv） ✅
"""
    (EXP_DIR/"notes.md").write_text(notes, encoding="utf-8")
    print(f"\n[{EXP_ID}] 完了  CV={overall:.6f}  vs exp008 {13.808621-overall:+.6f}  -> {EXP_DIR}/")


if __name__ == "__main__":
    main()
