#!/usr/bin/env python3
"""exp009b: Typewell GR well-level features only (remove noisy per-row correction).

exp009 の失敗分析:
  - Group E_row の tw_tvt_correction が最大のノイズ源
    correction = -gr_dev / slope = (GR noise σ≈20) / (slope≈0.25) = ±80 TVT
    >> target TVT range median (26 TVT) なので SNR が1以下
  - tw_known_gr_corr 層別:
    corr > 0.85 (243 wells): exp009 は exp008 より改善 (-0.215)
    corr < 0.60 (34 wells) : exp009 は exp008 より大幅悪化 (+0.857)
  → per-row 補正を全削除し、per-well E 特徴量のみに絞る

変更:
  - GROUP_E_ROW = ["tw_gr_at_anchor"] のみ (per-row 定数、per-well calibration の一部)
  - gr_dev_from_tw_anchor, tw_tvt_correction, tw_tvt_correction_reliable, tw_gr_extrap_range → 削除
  - _typewell_per_row() 関数 → 削除
  - enrich() から _typewell_per_row() 呼び出し → 削除

期待 CV: 13.73〜13.78 (exp008: 13.808621 より +0.03〜+0.08 改善)
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


EXP_ID = "exp009b_typewell_gr_well_only"
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

# ── exp009b Group E: per-well only (noisy per-row features removed) ──────────
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
    # REMOVED from exp009:
    # "gr_dev_from_tw_anchor"      -- ノイズ (0.6M gain)
    # "tw_tvt_correction"          -- ノイズ増幅 (0.5M gain, σ≈80 TVT)
    # "tw_tvt_correction_reliable" -- 同上 (0.8M gain)
    # "tw_gr_extrap_range"         -- 不要
]
GROUP_E = GROUP_E_WELL + GROUP_E_ROW

NEW_FEATURES = (GROUP_A + GROUP_B_WELL + GROUP_B_ROW + GROUP_C
                + GROUP_D_WELL + GROUP_D_ROW + GROUP_E)
ALL_FEATURES = SAFE_FEATURES + NEW_FEATURES

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
# exp009b Group E: Typewell GR alignment (per-well only)
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
    """Group E_well + tw_gr_at_anchor: per-well calibration from known portion vs typewell.

    NOTE: tw_gr_at_anchor is technically a per-row column (same value for all rows
    in a well since last_known_TVT is constant), and is safe to use.
    """
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
                         "tw_gr_at_anchor":GLOBAL_GR_MEAN_FALLBACK})
            continue

        fn = tw_interps[wid]

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

        # typewell GR at anchor + slope (finite difference: ±0.5 TVT)
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
        })
    return pd.DataFrame(recs)


def enrich(
    df: pd.DataFrame,
    tw_interps: dict,
    global_gr_mean: float,
) -> pd.DataFrame:
    """Full enrichment pipeline: exp007 + exp008 + exp009b features.

    exp009b vs exp009: _typewell_per_row() is REMOVED.
    Only per-well typewell features are used.
    """
    # exp007
    pw_traj = _traj_per_well(df)
    df = df.merge(pw_traj, on="well_id", how="left")
    df = _traj_per_row(df)
    # exp008
    pw_gr = _gr_per_well(df, global_gr_mean)
    df = df.merge(pw_gr, on="well_id", how="left")
    df = _gr_per_row(df, global_gr_mean)
    # exp009b: per-well only (no _typewell_per_row)
    pw_tw = _typewell_per_well(df, tw_interps)
    df = df.merge(pw_tw, on="well_id", how="left")
    return df


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"[{EXP_ID}] Typewell GR per-well features のみで LightGBM を学習します。")
    print("  (per-row 補正 tw_tvt_correction は削除: ノイズ σ≈80 >> target σ≈26)")
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
        "baseline_exp009_cv": 13.922291,
        "improvement_vs_exp009": round(13.922291 - overall, 6),
        "n_oof_rows": int(len(oof)),
        "n_submission_rows": int(len(submission)),
        "leak_risk": "low",
        "notes": (
            "exp009b: per-row E features removed. "
            "Kept: tw_known_gr_corr, tw_gr_offset, tw_gr_scale, tw_gr_last20_diff, "
            "tw_gr_slope_at_anchor, tw_gr_abs_slope_at_anchor, tw_gr_at_anchor. "
            "Removed: gr_dev_from_tw_anchor, tw_tvt_correction, "
            "tw_tvt_correction_reliable, tw_gr_extrap_range. "
            "Root cause of exp009 failure: correction noise sigma~80 >> target sigma~26."
        ),
    }
    write_json(EXP_DIR/"result.json", result)

    # notes.md
    fold_table = "\n".join(
        f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |"
        for r in fold_rows
    )
    top15 = "\n".join(
        f"| {r.feature} | {int(r.importance_gain):,} |"
        for r in fi_summary.head(15).itertuples()
    )
    e_fi = fi_summary[fi_summary["feature"].isin(GROUP_E)].copy()
    e_fi_table = "\n".join(
        f"| {r.feature} | {int(r.importance_gain):,} |"
        for r in e_fi.itertuples()
    )

    notes = f"""# {EXP_ID}

## 目的

exp009 (CV=13.922291) の失敗原因を修正: per-row GR 補正特徴量を全削除し、
per-well Typewell GR 特徴量のみを残す。

## 変更点（exp009 との差分）

| 削除した特徴量 | 削除理由 |
|---|---|
| gr_dev_from_tw_anchor | ノイズの入力値 (importance: 0.6M) |
| tw_tvt_correction | -gr_dev/slope → ノイズ増幅 σ≈80 TVT (0.5M) |
| tw_tvt_correction_reliable | 同上のマスク版 (0.8M) |
| tw_gr_extrap_range | 不要な補助特徴量 |

## 残した Group E 特徴量

| 特徴量 | grain | 定義 |
|---|---|---|
| tw_known_gr_corr | per-well | known区間 GR と typewell GR の相関 |
| tw_gr_offset | per-well | GR の系統的ずれ |
| tw_gr_scale | per-well | GR スケール差 |
| tw_gr_last20_diff | per-well | PS近傍の GR-typewell 乖離 |
| tw_gr_slope_at_anchor | per-well | dGR/dTVT at last_known_TVT |
| tw_gr_abs_slope_at_anchor | per-well | ｜dGR/dTVT｜ |
| tw_gr_at_anchor | per-row定数 | typewell GR at last_known_TVT |

## 結果

| metric | 値 |
|---|---|
| **exp009b CV RMSE** | **{overall:.6f}** |
| exp008 baseline | 13.808621 |
| exp009 (failed) | 13.922291 |
| vs exp008 | {13.808621-overall:+.6f} |
| vs exp009 | {13.922291-overall:+.6f} |

## Fold 別 RMSE

| fold | rmse | best_iter |
|---|---:|---:|
{fold_table}

## Top 15 Feature Importance (gain)

| feature | importance_gain |
|---|---:|
{top15}

## Group E 特徴量の重要度

| feature | importance_gain |
|---|---:|
{e_fi_table}

## リーク防止確認

- キャリブレーション: is_known_tvt==True 行のみ ✅
- typewell は参照曲線（別物理井戸）、ラベルではない ✅
- target GR は観測可能な物理計測値 ✅
- tw_gr_at_anchor は per-row 定数（全 row に同一値） ✅
- test: typewell_test_base_v001.parquet を使用 ✅
- per-row 補正ロジックは完全削除 ✅
"""
    (EXP_DIR/"notes.md").write_text(notes, encoding="utf-8")

    print(f"\n[{EXP_ID}] 完了")
    print(f"  CV RMSE    = {overall:.6f}")
    print(f"  vs exp008  : {13.808621-overall:+.6f}")
    print(f"  vs exp009  : {13.922291-overall:+.6f}")
    print(f"  出力: {EXP_DIR}/")


if __name__ == "__main__":
    main()
