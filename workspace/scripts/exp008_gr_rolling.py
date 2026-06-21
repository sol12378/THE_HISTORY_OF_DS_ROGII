#!/usr/bin/env python3
"""exp008: Add Group D (GR rolling) features on top of exp007b feature set.

Group D_well : Pre-PS GR statistics from known portion only
Group D_row  : Per-row causal (backward-only) GR rolling features

Leakage prevention rules:
  - pre-PS stats computed from is_known_tvt==True rows only
  - rolling windows are strictly backward-looking (Pandas default, center=False)
  - Missing GR filled: well-level pre_ps_gr_mean → GLOBAL_GR_MEAN (train-only constant)
  - No TVT used in any GR calculation
  - No cross-well GR averaging with label info
"""

from __future__ import annotations

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

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


EXP_ID = "exp008_gr_rolling"
EXP_DIR = Path("experiments") / EXP_ID

# ---- exp007 feature groups (carried over) ----
GROUP_A_FEATURES = [
    "pre_ps_tvt_slope_last20",
    "pre_ps_tvt_slope_last5",
    "pre_ps_tvt_curvature",
    "pre_ps_tvt_delta_last20",
]
GROUP_B_PREWELL_FEATURES = [
    "pre_ps_dZ_dMD",
    "pre_ps_dX_dMD",
    "pre_ps_dY_dMD",
    "pre_ps_horiz_dMD",
    "pre_ps_azimuth",
]
GROUP_B_PERROW_FEATURES = [
    "dZ_dMD_from_ps",
    "dX_dMD_from_ps",
    "dY_dMD_from_ps",
    "horiz_disp_from_ps",
    "azimuth_from_ps",
]
GROUP_C_FEATURES = ["kh_ratio", "hidden_frac"]

# ---- exp008 new features (Group D) ----
GROUP_D_WELL_FEATURES = [
    "pre_ps_gr_mean",
    "pre_ps_gr_std",
    "pre_ps_gr_last20_mean",
    "pre_ps_gr_trend",
    "pre_ps_gr_available_frac",
]
GROUP_D_ROW_FEATURES = [
    "gr_vs_pre_ps_mean",
    "gr_z_score",
    "gr_rolling_mean_w20",
    "gr_rolling_mean_w50",
    "gr_rolling_std_w20",
]
GROUP_D_FEATURES = GROUP_D_WELL_FEATURES + GROUP_D_ROW_FEATURES

NEW_FEATURES = (
    GROUP_A_FEATURES
    + GROUP_B_PREWELL_FEATURES
    + GROUP_B_PERROW_FEATURES
    + GROUP_C_FEATURES
    + GROUP_D_FEATURES
)

ALL_FEATURES = SAFE_FEATURES + NEW_FEATURES


# ============================================================
# exp007 feature builders (copied to keep this script self-contained)
# ============================================================

def _compute_per_well_traj_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """Group A + Group B pre-PS per-well features (from known portion)."""
    known = df_full.loc[df_full["is_known_tvt"].astype(bool)].copy()
    records = []
    for well_id, grp in known.groupby("well_id", sort=False):
        grp = grp.sort_values("row_idx")
        n = len(grp)
        md  = grp["MD"].to_numpy(dtype=float)
        x   = grp["X"].to_numpy(dtype=float)
        y   = grp["Y"].to_numpy(dtype=float)
        z   = grp["Z"].to_numpy(dtype=float)
        tvt = grp["TVT_input"].to_numpy(dtype=float)

        n20 = min(20, n); n5 = min(5, n)
        md_diff_20 = md[-1] - md[-n20] if n20 > 1 else 1.0
        md_diff_5  = md[-1] - md[-n5]  if n5  > 1 else 1.0

        tvt_slope_20 = (tvt[-1] - tvt[-n20]) / md_diff_20 if abs(md_diff_20) > 1e-6 else 0.0
        tvt_slope_5  = (tvt[-1] - tvt[-n5])  / md_diff_5  if abs(md_diff_5)  > 1e-6 else 0.0
        tvt_delta_20 = tvt[-1] - tvt[-n20]
        tvt_curv     = tvt_slope_5 - tvt_slope_20

        if abs(md_diff_20) > 1e-6:
            dz_dmd = (z[-1] - z[-n20]) / md_diff_20
            dx_dmd = (x[-1] - x[-n20]) / md_diff_20
            dy_dmd = (y[-1] - y[-n20]) / md_diff_20
        else:
            dz_dmd = dx_dmd = dy_dmd = 0.0

        dx_20 = x[-1] - x[-n20]; dy_20 = y[-1] - y[-n20]
        horiz_disp = float(np.sqrt(dx_20 ** 2 + dy_20 ** 2))
        horiz_dmd  = horiz_disp / md_diff_20 if abs(md_diff_20) > 1e-6 else 0.0
        azimuth    = float(np.arctan2(dy_20, dx_20))

        records.append({
            "well_id": well_id,
            "pre_ps_tvt_slope_last20": tvt_slope_20,
            "pre_ps_tvt_slope_last5":  tvt_slope_5,
            "pre_ps_tvt_curvature":    tvt_curv,
            "pre_ps_tvt_delta_last20": tvt_delta_20,
            "pre_ps_dZ_dMD":    dz_dmd,
            "pre_ps_dX_dMD":    dx_dmd,
            "pre_ps_dY_dMD":    dy_dmd,
            "pre_ps_horiz_dMD": horiz_dmd,
            "pre_ps_azimuth":   azimuth,
        })
    return pd.DataFrame(records)


def _add_perrow_traj_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group B per-row + Group C features."""
    df = df.copy()
    md_safe = np.where(df["delta_MD_from_PS"].to_numpy(dtype=float) < 1.0,
                       1.0, df["delta_MD_from_PS"].to_numpy(dtype=float))
    df["dZ_dMD_from_ps"]   = df["delta_Z_from_PS"].astype(float) / md_safe
    df["dX_dMD_from_ps"]   = df["delta_X_from_PS"].astype(float) / md_safe
    df["dY_dMD_from_ps"]   = df["delta_Y_from_PS"].astype(float) / md_safe
    df["horiz_disp_from_ps"] = np.sqrt(
        df["delta_X_from_PS"].astype(float) ** 2 + df["delta_Y_from_PS"].astype(float) ** 2
    )
    df["azimuth_from_ps"] = np.arctan2(
        df["delta_Y_from_PS"].astype(float), df["delta_X_from_PS"].astype(float)
    )
    hl = df["hidden_length"].to_numpy(dtype=float)
    df["kh_ratio"]    = df["known_length"].astype(float) / np.where(hl < 1.0, 1.0, hl)
    df["hidden_frac"] = hl / df["n_rows_in_well"].astype(float)
    return df


# ============================================================
# exp008 Group D feature builders
# ============================================================

def _compute_per_well_gr_features(
    df_full: pd.DataFrame,
    global_gr_mean: float,
) -> pd.DataFrame:
    """Group D_well: pre-PS GR statistics (from is_known_tvt rows only).

    Leakage-safe: uses only known-portion GR, never TVT.
    """
    known = df_full.loc[df_full["is_known_tvt"].astype(bool)].copy()
    records = []
    for well_id, grp in known.groupby("well_id", sort=False):
        grp   = grp.sort_values("row_idx")
        gr    = grp["GR"].to_numpy(dtype=float)
        md    = grp["MD"].to_numpy(dtype=float)
        valid = ~np.isnan(gr)
        n_valid = int(valid.sum())
        n_total = len(gr)

        if n_valid == 0:
            # 全欠損 well: グローバル平均で補完
            pre_ps_gr_mean      = global_gr_mean
            pre_ps_gr_std       = 0.0
            pre_ps_gr_last20    = global_gr_mean
            pre_ps_gr_trend     = 0.0
            pre_ps_available    = 0.0
        else:
            gr_valid = gr[valid]
            md_valid = md[valid]

            pre_ps_gr_mean   = float(np.nanmean(gr))
            pre_ps_gr_std    = float(np.nanstd(gr))
            pre_ps_available = n_valid / n_total

            # last 20 valid points の平均
            last20_gr = gr_valid[-min(20, len(gr_valid)):]
            pre_ps_gr_last20 = float(last20_gr.mean())

            # GR trend: simple slope over all valid known rows
            if len(gr_valid) >= 2:
                md_diff = md_valid[-1] - md_valid[0]
                pre_ps_gr_trend = (gr_valid[-1] - gr_valid[0]) / md_diff if abs(md_diff) > 1e-6 else 0.0
            else:
                pre_ps_gr_trend = 0.0

        records.append({
            "well_id":               well_id,
            "pre_ps_gr_mean":        pre_ps_gr_mean,
            "pre_ps_gr_std":         pre_ps_gr_std,
            "pre_ps_gr_last20_mean": pre_ps_gr_last20,
            "pre_ps_gr_trend":       pre_ps_gr_trend,
            "pre_ps_gr_available_frac": pre_ps_available,
        })
    return pd.DataFrame(records)


def _add_perrow_gr_features(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    """Group D_row: causal (backward-only) GR rolling features.

    Leakage-safe:
      - Rolling uses Pandas default (center=False) → backward-looking only
      - Missing GR filled with pre_ps_gr_mean → global_gr_mean (no TVT used)
      - Computation is per-well, no cross-well averaging
    """
    df = df.copy()
    # 1. fill missing GR: well-level pre_ps_gr_mean, fallback global
    gr_filled = df["GR"].copy().astype(float)
    # per-well fill using pre_ps_gr_mean (already merged)
    mask_nan = gr_filled.isna()
    gr_filled.loc[mask_nan] = df.loc[mask_nan, "pre_ps_gr_mean"].fillna(global_gr_mean)

    # 2. gr deviation features (per-row, uses only pre_ps stats)
    std_safe = df["pre_ps_gr_std"].fillna(1.0).replace(0.0, 1.0)
    df["gr_vs_pre_ps_mean"] = gr_filled - df["pre_ps_gr_mean"].fillna(global_gr_mean)
    df["gr_z_score"]        = df["gr_vs_pre_ps_mean"] / std_safe

    # 3. backward-only rolling windows
    # Sort is critical: must be (well_id, row_idx) order before rolling
    df = df.sort_values(["well_id", "row_idx"]).copy()
    df["_gr_filled"] = gr_filled  # temp column for rolling

    for window, col in [(20, "gr_rolling_mean_w20"),
                        (50, "gr_rolling_mean_w50")]:
        df[col] = (
            df.groupby("well_id", sort=False)["_gr_filled"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    df["gr_rolling_std_w20"] = (
        df.groupby("well_id", sort=False)["_gr_filled"]
        .transform(lambda x: x.rolling(20, min_periods=2).std().fillna(0.0))
    )

    df = df.drop(columns=["_gr_filled"])
    return df


def enrich(
    df: pd.DataFrame,
    global_gr_mean: float,
) -> pd.DataFrame:
    """Full feature enrichment pipeline for one split (train or test)."""
    # exp007 features
    per_well_traj = _compute_per_well_traj_features(df)
    df = df.merge(per_well_traj, on="well_id", how="left")
    df = _add_perrow_traj_features(df)

    # exp008 Group D
    per_well_gr = _compute_per_well_gr_features(df, global_gr_mean)
    df = df.merge(per_well_gr, on="well_id", how="left")
    df = _add_perrow_gr_features(df, global_gr_mean)

    return df


# ============================================================
# Main training loop
# ============================================================

def main() -> None:
    print(f"[{EXP_ID}] Group D (GR rolling) 特徴量を追加して LightGBM を学習します。")
    ensure_exp_dir(EXP_DIR)

    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )

    # global GR mean from TRAIN only (no test contamination)
    global_gr_mean = float(
        train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean()
    )
    print(f"Global GR mean (train): {global_gr_mean:.4f}")

    print("特徴量エンジニアリング: train ...")
    train = enrich(train, global_gr_mean)
    print("特徴量エンジニアリング: test ...")
    test  = enrich(test,  global_gr_mean)

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
        "objective":       "regression",
        "metric":          "rmse",
        "learning_rate":   0.05,
        "num_leaves":      63,
        "max_depth":       -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq":    1,
        "lambda_l2":       1.0,
        "verbosity":       -1,
        "seed":            42,
        "num_threads":     8,
    }

    n_folds = None
    for fold in sorted(train_t["fold"].unique()):
        valid_mask  = train_t["fold"].eq(fold).to_numpy()
        train_mask  = ~valid_mask
        if n_folds is None:
            n_folds = int(train_t["fold"].nunique())
        print(f"fold {fold}: train={train_mask.sum()}, valid={valid_mask.sum()}")

        model = lgb.LGBMRegressor(**params, n_estimators=1500)
        model.fit(
            train_t.loc[train_mask, ALL_FEATURES],
            y_delta.loc[train_mask],
            eval_set=[(train_t.loc[valid_mask, ALL_FEATURES], y_delta.loc[valid_mask])],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        best_iter   = int(model.best_iteration_ or model.n_estimators)
        valid_delta = model.predict(
            train_t.loc[valid_mask, ALL_FEATURES], num_iteration=best_iter
        )
        oof_pred_delta[valid_mask] = valid_delta
        fold_pred_tvt = train_t.loc[valid_mask, "last_known_TVT"].to_numpy(dtype=float) + valid_delta
        fold_rmse = tvt_rmse(train_t.loc[valid_mask, "TVT"], fold_pred_tvt)
        fold_rows.append({"fold": int(fold), "n_rows": int(valid_mask.sum()),
                          "rmse": fold_rmse, "best_iteration": best_iter})
        print(f"  fold {fold}  RMSE={fold_rmse:.6f}  best_iter={best_iter}")

        test_pred_delta += model.predict(
            test_t[ALL_FEATURES], num_iteration=best_iter
        ) / n_folds
        importances.append(pd.DataFrame({
            "fold": int(fold), "feature": ALL_FEATURES,
            "importance_gain":  model.booster_.feature_importance("gain"),
            "importance_split": model.booster_.feature_importance("split"),
        }))

    # ---- outputs ----
    oof = train_t[["id", "well_id", "row_idx", "fold", "TVT", "last_known_TVT"]].copy()
    oof["pred_delta"] = oof_pred_delta
    oof[PRED_COL]     = oof["last_known_TVT"].astype(float) + oof["pred_delta"]
    oof["error"]      = oof[PRED_COL] - oof["TVT"]
    oof["abs_error"]  = oof["error"].abs()
    oof.to_csv(EXP_DIR / "oof.csv", index=False)

    cv = pd.DataFrame(fold_rows)
    cv.to_csv(EXP_DIR / "cv.csv", index=False)

    test = test.copy()
    test.loc[test["is_target"].astype(bool), "pred_delta"] = test_pred_delta
    test.loc[test["is_target"].astype(bool), PRED_COL] = (
        test_t["last_known_TVT"].to_numpy(dtype=float) + test_pred_delta
    )
    submission = build_submission(test, sample_submission, PRED_COL)
    submission.to_csv(EXP_DIR / "submission.csv", index=False)

    fi = pd.concat(importances, ignore_index=True)
    fi_summary = (
        fi.groupby("feature", as_index=False)[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
    )
    fi_summary.to_csv(EXP_DIR / "feature_importance.csv", index=False)

    overall_rmse = tvt_rmse(oof["TVT"], oof[PRED_COL])
    result = {
        "exp_id":       EXP_ID,
        "created_at":   now_jst(),
        "status":       "completed",
        "model":        "LightGBMRegressor",
        "target":       "TVT - last_known_TVT",
        "prediction":   "last_known_TVT + pred_delta",
        "metric":       "TVT_abs_RMSE",
        "cv_rmse":      overall_rmse,
        "cv_mean":      float(cv["rmse"].mean()),
        "cv_std":       float(cv["rmse"].std(ddof=0)),
        "baseline_exp007b_cv": 13.853360,
        "improvement_vs_exp007b": round(13.853360 - overall_rmse, 6),
        "baseline_exp007_cv": 13.867054,
        "improvement_vs_exp007": round(13.867054 - overall_rmse, 6),
        "feature_groups": {
            "safe":           SAFE_FEATURES,
            "group_a":        GROUP_A_FEATURES,
            "group_b_prewell": GROUP_B_PREWELL_FEATURES,
            "group_b_perrow": GROUP_B_PERROW_FEATURES,
            "group_c":        GROUP_C_FEATURES,
            "group_d_well":   GROUP_D_WELL_FEATURES,
            "group_d_row":    GROUP_D_ROW_FEATURES,
        },
        "global_gr_mean_train": global_gr_mean,
        "n_oof_rows":       int(len(oof)),
        "n_submission_rows": int(len(submission)),
        "leak_risk":        "low",
        "notes": (
            "Group D_well: pre-PS GR mean/std/trend from known rows only. "
            "Group D_row: backward-only rolling (center=False), "
            "missing GR filled with pre_ps_gr_mean → global train mean. "
            "No TVT used in GR calculations."
        ),
    }
    write_json(EXP_DIR / "result.json", result)

    notes = f"""# {EXP_ID}

## 目的

exp007b (CV=13.853360) に Group D (GR rolling 特徴量) を追加し、
GRの文脈情報が TVT 予測に貢献するかを検証する。

## 追加特徴量

### Group D_well（per-well、既知区間から算出）

| 特徴量 | 定義 |
|---|---|
| pre_ps_gr_mean | known行 GR 平均 |
| pre_ps_gr_std | known行 GR 標準偏差 |
| pre_ps_gr_last20_mean | known末尾20行 GR 平均 |
| pre_ps_gr_trend | known行全体の GR 変化率 |
| pre_ps_gr_available_frac | known行の GR 有効割合 |

### Group D_row（per-row、因果ローリング）

| 特徴量 | 定義 |
|---|---|
| gr_vs_pre_ps_mean | GR - pre_ps_gr_mean |
| gr_z_score | (GR - mean) / std |
| gr_rolling_mean_w20 | 後ろ向き rolling mean (w=20) |
| gr_rolling_mean_w50 | 後ろ向き rolling mean (w=50) |
| gr_rolling_std_w20 | 後ろ向き rolling std (w=20) |

## リーク防止

- pre-PS 統計は is_known_tvt==True 行のみから計算
- rolling は center=False（後ろ向きのみ）
- 欠損補完: pre_ps_gr_mean → GLOBAL_GR_MEAN={global_gr_mean:.4f}（train のみから計算した定数）
- TVT・TVT_input は一切使用しない

## 結果

| metric | 値 |
|---|---|
| exp008 CV RMSE | **{overall_rmse:.6f}** |
| exp007b baseline | 13.853360 |
| vs exp007b | {13.853360 - overall_rmse:+.6f} |
| vs exp007 | {13.867054 - overall_rmse:+.6f} |

## Fold 別 RMSE

| fold | rmse | best_iter |
|---|---:|---:|
""" + "\n".join(
        f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |"
        for r in fold_rows
    ) + f"""

## Top 10 Feature Importance (gain)

| feature | importance_gain |
|---|---:|
""" + "\n".join(
        f"| {row.feature} | {int(row.importance_gain):,} |"
        for row in fi_summary.head(10).itertuples()
    ) + """
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(f"\n[{EXP_ID}] 完了")
    print(f"  CV RMSE = {overall_rmse:.6f}")
    print(f"  vs exp007b (+1.188): {13.853360 - overall_rmse:+.6f}")
    print(f"  vs exp007  (+1.188): {13.867054 - overall_rmse:+.6f}")
    print(f"  出力: {EXP_DIR}/")


if __name__ == "__main__":
    main()
