#!/usr/bin/env python3
"""Run LightGBM delta baseline with group well folds."""

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


EXP_ID = "exp003_lgb_anchor_trajectory"
EXP_DIR = Path("experiments") / EXP_ID


def _available_features(df: pd.DataFrame) -> list[str]:
    missing = [col for col in SAFE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"必要な安全特徴量がありません: {missing}")
    return SAFE_FEATURES.copy()


def main() -> None:
    print("LightGBM baseline を開始します。仮説: anchor からの delta を安全特徴量で学習すると anchor baseline を改善できる。")
    ensure_exp_dir(EXP_DIR)

    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )
    train = attach_folds(train, folds)
    features = _available_features(train)

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
        print(f"fold {fold}: train={int(train_mask.sum())}, valid={int(valid_mask.sum())}")

        model = lgb.LGBMRegressor(
            **params,
            n_estimators=600,
        )
        model.fit(
            train_t.loc[train_mask, features],
            y_delta.loc[train_mask],
            eval_set=[(train_t.loc[valid_mask, features], y_delta.loc[valid_mask])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        valid_delta = model.predict(train_t.loc[valid_mask, features], num_iteration=model.best_iteration_)
        oof_pred_delta[valid_mask] = valid_delta
        fold_pred_tvt = train_t.loc[valid_mask, "last_known_TVT"].to_numpy(dtype=float) + valid_delta
        fold_rmse = tvt_rmse(train_t.loc[valid_mask, "TVT"], fold_pred_tvt)
        fold_rows.append(
            {
                "fold": int(fold),
                "n_rows": int(valid_mask.sum()),
                "rmse": fold_rmse,
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
            }
        )

        test_pred_delta += model.predict(test_t[features], num_iteration=model.best_iteration_) / train_t["fold"].nunique()
        importances.append(
            pd.DataFrame(
                {
                    "fold": int(fold),
                    "feature": features,
                    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                    "importance_split": model.booster_.feature_importance(importance_type="split"),
                }
            )
        )

    oof = train_t[["id", "well_id", "row_idx", "fold", "TVT", "last_known_TVT"]].copy()
    oof["pred_delta"] = oof_pred_delta
    oof[PRED_COL] = oof["last_known_TVT"].astype(float) + oof["pred_delta"]
    oof["error"] = oof[PRED_COL] - oof["TVT"]
    oof["abs_error"] = oof["error"].abs()
    oof.to_csv(EXP_DIR / "oof.csv", index=False)

    cv = pd.DataFrame(fold_rows)
    cv.to_csv(EXP_DIR / "cv.csv", index=False)

    test = test.copy()
    test.loc[test["is_target"].astype(bool), "pred_delta"] = test_pred_delta
    test.loc[test["is_target"].astype(bool), PRED_COL] = test_t["last_known_TVT"].to_numpy(dtype=float) + test_pred_delta
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
        "features": features,
        "n_oof_rows": int(len(oof)),
        "n_submission_rows": int(len(submission)),
        "leak_risk": "low_to_medium",
        "notes": "train-only marker columns と target-derived columns を除外し、anchor からの delta を学習。",
    }
    write_json(EXP_DIR / "result.json", result)

    notes = f"""# {EXP_ID}

## 目的

[[Anchor_Features]] と trajectory 系の安全特徴量で `TVT - last_known_TVT` を学習し、anchor baseline からの改善を確認する。

## 仮説

target 区間内の `MD/X/Y/Z/GR` と PS からの相対位置を使えば、単純な `last_known_TVT` 固定予測より TVT絶対値RMSEを下げられる。

## 変更内容

- 入力: `data/processed/train_base_v001.parquet`, `data/processed/test_base_v001.parquet`, `data/folds/folds_group_well_v001.csv`
- fold: `folds_group_well_v001.csv` の well 単位 Group fold
- target: `TVT - last_known_TVT`
- prediction: `last_known_TVT + pred_delta`
- 特徴量: `{", ".join(features)}`
- 除外: `TVT`, `TVT_input`, `id`, `is_target`, `is_known_tvt`, `split`, `well_id`, `source_path`, `fold` など train-only marker / target-derived / ID 系列

## 結果

- overall CV RMSE: {overall_rmse:.6f}
- fold mean RMSE: {result["cv_mean"]:.6f}
- fold std RMSE: {result["cv_std"]:.6f}
- OOF rows: {len(oof)}
- submission rows: {len(submission)}

## リーク懸念

target の真値 `TVT` は目的変数の delta 作成と valid 評価にのみ使用し、特徴量には入れていない。`is_target` や `is_known_tvt` など train/test の役割を直接示す marker columns も特徴量から除外した。`last_known_*` と `delta_*_from_PS` は PS 以前の anchor と各行の測定座標に基づく想定で、リーク懸念は低〜中。ただし base table 生成時に test 未知TVTや target 真値由来の情報が混入していないことを継続確認する。

## 次アクション

`feature_importance.csv` と OOF error slicing で、well長・hidden_length・PSからの距離別に誤差を見る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(f"完了: {EXP_DIR} / CV RMSE={overall_rmse:.6f}")


if __name__ == "__main__":
    main()
