#!/usr/bin/env python3
"""exp007: Add Groups A+B+C features to LightGBM delta model.

Group A: Pre-PS TVT momentum (slope, curvature, delta from known portion)
Group B: Trajectory direction features (pre-PS direction + per-row from PS)
Group C: Well shape ratios (kh_ratio, hidden_frac)
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


EXP_ID = "exp007_traj_features"
EXP_DIR = Path("experiments") / EXP_ID

# New features grouped for ablation tracking
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

GROUP_C_FEATURES = [
    "kh_ratio",
    "hidden_frac",
]

NEW_FEATURES = (
    GROUP_A_FEATURES
    + GROUP_B_PREWELL_FEATURES
    + GROUP_B_PERROW_FEATURES
    + GROUP_C_FEATURES
)

ALL_FEATURES = SAFE_FEATURES + NEW_FEATURES


def compute_per_well_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """Compute per-well features from the known portion (is_known_tvt==True)."""
    known = df_full.loc[df_full["is_known_tvt"].astype(bool)].copy()

    records = []
    for well_id, grp in known.groupby("well_id", sort=False):
        grp = grp.sort_values("row_idx")
        n = len(grp)

        md = grp["MD"].to_numpy(dtype=float)
        x = grp["X"].to_numpy(dtype=float)
        y = grp["Y"].to_numpy(dtype=float)
        z = grp["Z"].to_numpy(dtype=float)
        tvt = grp["TVT_input"].to_numpy(dtype=float)

        n20 = min(20, n)
        n5 = min(5, n)

        # --- Group A: pre-PS TVT momentum ---
        md_diff_20 = md[-1] - md[-n20] if n20 > 1 else 1.0
        md_diff_5 = md[-1] - md[-n5] if n5 > 1 else 1.0

        if abs(md_diff_20) > 1e-6:
            tvt_slope_20 = (tvt[-1] - tvt[-n20]) / md_diff_20
        else:
            tvt_slope_20 = 0.0

        if abs(md_diff_5) > 1e-6:
            tvt_slope_5 = (tvt[-1] - tvt[-n5]) / md_diff_5
        else:
            tvt_slope_5 = 0.0

        tvt_delta_20 = tvt[-1] - tvt[-n20]
        tvt_curvature = tvt_slope_5 - tvt_slope_20

        # --- Group B pre-PS: trajectory direction in known portion ---
        if abs(md_diff_20) > 1e-6:
            dz_dmd = (z[-1] - z[-n20]) / md_diff_20
            dx_dmd = (x[-1] - x[-n20]) / md_diff_20
            dy_dmd = (y[-1] - y[-n20]) / md_diff_20
        else:
            dz_dmd = dx_dmd = dy_dmd = 0.0

        dx_20 = x[-1] - x[-n20]
        dy_20 = y[-1] - y[-n20]
        horiz_disp = float(np.sqrt(dx_20 ** 2 + dy_20 ** 2))
        horiz_dmd = horiz_disp / md_diff_20 if abs(md_diff_20) > 1e-6 else 0.0
        azimuth = float(np.arctan2(dy_20, dx_20))

        records.append(
            {
                "well_id": well_id,
                "pre_ps_tvt_slope_last20": tvt_slope_20,
                "pre_ps_tvt_slope_last5": tvt_slope_5,
                "pre_ps_tvt_curvature": tvt_curvature,
                "pre_ps_tvt_delta_last20": tvt_delta_20,
                "pre_ps_dZ_dMD": dz_dmd,
                "pre_ps_dX_dMD": dx_dmd,
                "pre_ps_dY_dMD": dy_dmd,
                "pre_ps_horiz_dMD": horiz_dmd,
                "pre_ps_azimuth": azimuth,
            }
        )

    return pd.DataFrame(records)


def add_per_row_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row Group B (from-PS direction) and Group C (shape ratio) features."""
    md_safe = np.where(df["delta_MD_from_PS"].to_numpy(dtype=float) < 1.0, 1.0, df["delta_MD_from_PS"].to_numpy(dtype=float))

    df = df.copy()
    df["dZ_dMD_from_ps"] = df["delta_Z_from_PS"].astype(float) / md_safe
    df["dX_dMD_from_ps"] = df["delta_X_from_PS"].astype(float) / md_safe
    df["dY_dMD_from_ps"] = df["delta_Y_from_PS"].astype(float) / md_safe
    df["horiz_disp_from_ps"] = np.sqrt(
        df["delta_X_from_PS"].astype(float) ** 2
        + df["delta_Y_from_PS"].astype(float) ** 2
    )
    df["azimuth_from_ps"] = np.arctan2(
        df["delta_Y_from_PS"].astype(float),
        df["delta_X_from_PS"].astype(float),
    )

    hl = df["hidden_length"].to_numpy(dtype=float)
    hl_safe = np.where(hl < 1.0, 1.0, hl)
    df["kh_ratio"] = df["known_length"].astype(float) / hl_safe
    df["hidden_frac"] = hl / df["n_rows_in_well"].astype(float)

    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-well and per-row features, merge into df."""
    per_well = compute_per_well_features(df)
    df = df.merge(per_well, on="well_id", how="left")
    df = add_per_row_features(df)
    return df


def main() -> None:
    print(
        f"[{EXP_ID}] Groups A+B+C 特徴量を追加した LightGBM delta モデルを実行します。"
    )
    ensure_exp_dir(EXP_DIR)

    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )

    print("特徴量エンジニアリング: train ...")
    train = enrich(train)
    print("特徴量エンジニアリング: test ...")
    test = enrich(test)

    train = attach_folds(train, folds)

    missing_train = [c for c in ALL_FEATURES if c not in train.columns]
    missing_test = [c for c in ALL_FEATURES if c not in test.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"必要な特徴量が不足しています。train: {missing_train}, test: {missing_test}"
        )

    train_t = target_rows(train)
    test_t = target_rows(test)
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    oof_pred_delta = np.zeros(len(train_t), dtype=float)
    test_pred_delta = np.zeros(len(test_t), dtype=float)
    fold_rows = []
    importances = []

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 42,
        "num_threads": 8,
    }

    for fold in sorted(train_t["fold"].unique()):
        valid_mask = train_t["fold"].eq(fold).to_numpy()
        train_mask = ~valid_mask
        print(
            f"fold {fold}: train={int(train_mask.sum())}, valid={int(valid_mask.sum())}"
        )

        model = lgb.LGBMRegressor(**params, n_estimators=600)
        model.fit(
            train_t.loc[train_mask, ALL_FEATURES],
            y_delta.loc[train_mask],
            eval_set=[
                (train_t.loc[valid_mask, ALL_FEATURES], y_delta.loc[valid_mask])
            ],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )

        valid_delta = model.predict(
            train_t.loc[valid_mask, ALL_FEATURES],
            num_iteration=model.best_iteration_,
        )
        oof_pred_delta[valid_mask] = valid_delta
        fold_pred_tvt = (
            train_t.loc[valid_mask, "last_known_TVT"].to_numpy(dtype=float)
            + valid_delta
        )
        fold_rmse = tvt_rmse(train_t.loc[valid_mask, "TVT"], fold_pred_tvt)
        fold_rows.append(
            {
                "fold": int(fold),
                "n_rows": int(valid_mask.sum()),
                "rmse": fold_rmse,
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )
        print(f"  fold {fold} RMSE={fold_rmse:.6f}, best_iter={model.best_iteration_}")

        n_folds = int(train_t["fold"].nunique())
        test_pred_delta += (
            model.predict(test_t[ALL_FEATURES], num_iteration=model.best_iteration_)
            / n_folds
        )
        importances.append(
            pd.DataFrame(
                {
                    "fold": int(fold),
                    "feature": ALL_FEATURES,
                    "importance_gain": model.booster_.feature_importance(
                        importance_type="gain"
                    ),
                    "importance_split": model.booster_.feature_importance(
                        importance_type="split"
                    ),
                }
            )
        )

    # --- Save OOF ---
    oof = train_t[["id", "well_id", "row_idx", "fold", "TVT", "last_known_TVT"]].copy()
    oof["pred_delta"] = oof_pred_delta
    oof[PRED_COL] = oof["last_known_TVT"].astype(float) + oof["pred_delta"]
    oof["error"] = oof[PRED_COL] - oof["TVT"]
    oof["abs_error"] = oof["error"].abs()
    oof.to_csv(EXP_DIR / "oof.csv", index=False)

    # --- Save CV ---
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(EXP_DIR / "cv.csv", index=False)

    # --- Save submission ---
    test = test.copy()
    test.loc[test["is_target"].astype(bool), "pred_delta"] = test_pred_delta
    test.loc[test["is_target"].astype(bool), PRED_COL] = (
        test_t["last_known_TVT"].to_numpy(dtype=float) + test_pred_delta
    )
    submission = build_submission(test, sample_submission, PRED_COL)
    submission.to_csv(EXP_DIR / "submission.csv", index=False)

    # --- Save feature importance ---
    fi = pd.concat(importances, ignore_index=True)
    fi_summary = (
        fi.groupby("feature", as_index=False)[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
    )
    fi_summary.to_csv(EXP_DIR / "feature_importance.csv", index=False)

    # --- Result JSON ---
    overall_rmse = tvt_rmse(oof["TVT"], oof[PRED_COL])
    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "model": "LightGBMRegressor",
        "target": "TVT - last_known_TVT",
        "prediction": "last_known_TVT + pred_delta",
        "metric": "TVT_abs_RMSE",
        "cv_rmse": overall_rmse,
        "cv_mean": float(cv["rmse"].mean()),
        "cv_std": float(cv["rmse"].std(ddof=0)),
        "baseline_exp003_cv": 15.054865,
        "improvement_vs_exp003": round(15.054865 - overall_rmse, 6),
        "feature_groups": {
            "safe_features": SAFE_FEATURES,
            "group_a": GROUP_A_FEATURES,
            "group_b_prewell": GROUP_B_PREWELL_FEATURES,
            "group_b_perrow": GROUP_B_PERROW_FEATURES,
            "group_c": GROUP_C_FEATURES,
        },
        "n_oof_rows": int(len(oof)),
        "n_submission_rows": int(len(submission)),
        "leak_risk": "low",
        "notes": (
            "Group A: pre-PS TVT slope/curvature (from TVT_input in known rows). "
            "Group B: trajectory direction (pre-PS per-well + per-row from PS). "
            "Group C: kh_ratio, hidden_frac. "
            "All features computed from observable data; no target leakage."
        ),
    }
    write_json(EXP_DIR / "result.json", result)

    # --- Notes markdown ---
    def rows_to_md(df: pd.DataFrame) -> str:
        header = " | ".join(str(c) for c in df.columns)
        sep = " | ".join(["---"] * len(df.columns))
        rows = "\n".join(
            " | ".join(str(v) for v in r) for r in df.itertuples(index=False)
        )
        return f"| {header} |\n| {sep} |\n" + "\n".join(
            f"| {' | '.join(str(v) for v in r)} |"
            for r in df.itertuples(index=False)
        )

    cv_disp = cv.copy()
    cv_disp["rmse"] = cv_disp["rmse"].round(6)
    top_fi = fi_summary.head(20).copy()
    top_fi["importance_gain"] = top_fi["importance_gain"].round(0).astype(int)

    notes = f"""# {EXP_ID}

## 目的

exp003 (LightGBM baseline, CV=15.054865) に Groups A+B+C 特徴量を追加し、TVT予測精度を改善する。

## 変更内容

- **Group A (Pre-PS TVT momentum)**: pre_ps_tvt_slope_last20, pre_ps_tvt_slope_last5, pre_ps_tvt_curvature, pre_ps_tvt_delta_last20
- **Group B pre-PS**: pre_ps_dZ_dMD, pre_ps_dX_dMD, pre_ps_dY_dMD, pre_ps_horiz_dMD, pre_ps_azimuth
- **Group B per-row**: dZ_dMD_from_ps, dX_dMD_from_ps, dY_dMD_from_ps, horiz_disp_from_ps, azimuth_from_ps
- **Group C**: kh_ratio (known/hidden), hidden_frac (hidden/total rows)
- LightGBMハイパーパラメータは exp003 と同一

## 結果

| metric | value |
|---|---|
| overall CV RMSE | {overall_rmse:.6f} |
| exp003 baseline | 15.054865 |
| 改善幅 | {15.054865 - overall_rmse:+.6f} |
| fold mean | {result['cv_mean']:.6f} |
| fold std | {result['cv_std']:.6f} |

## Fold別RMSE

{rows_to_md(cv_disp[['fold','rmse','best_iteration']])}

## Top 20 Feature Importance (gain)

{rows_to_md(top_fi[['feature','importance_gain']])}

## リーク懸念

- Group A は is_known_tvt 行の TVT_input を使う。train/testで観測済みデータ（PSより前）のみ使用。リーク低。
- Group B pre-PS は is_known_tvt 行のX/Y/Z座標から方向を計算。PSより前のデータのみ。リーク低。
- Group B per-row は delta_*_from_PS 列から計算。既存列と同等リスク（低〜中）。
- Group C は既存の known_length/hidden_length から計算。リーク低。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(f"\n[{EXP_ID}] 完了: CV RMSE={overall_rmse:.6f}  (exp003比 {15.054865 - overall_rmse:+.6f})")
    print(f"  出力: {EXP_DIR}/")


if __name__ == "__main__":
    main()
