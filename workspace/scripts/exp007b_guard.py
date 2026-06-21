#!/usr/bin/env python3
"""exp007b: n_estimators=1500 + 双方向 anchor guard。

exp007 の発見:
  - fold4 が n_estimators=600 を使い切り → underfitting
  - |mean_pred_delta| > 30 の wells で RMSE 38-48 の catastrophic 予測
  - exp006 は正方向 (mean_pred_delta > 30) のみガード、負方向が未対応

変更点:
  - n_estimators=1500 (early_stopping=50 は同じ)
  - ガードルール:
      |mean_pred_delta| > 30  OR  hidden_length > 8000  → alpha=0 (full anchor)
      それ以外                                          → alpha=1.0 (LGB そのまま)
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


EXP_ID = "exp007b_guard"
EXP_DIR = Path("experiments") / EXP_ID

ANCHOR_EXP = "exp001_anchor_baseline"

# Guard thresholds
ABS_MEAN_DELTA_THR = 30.0   # |mean_pred_delta| > this → full anchor
HIDDEN_LENGTH_THR = 8000.0  # hidden_length > this → full anchor

# Feature groups (exp007 と同一)
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


# ─── 特徴量エンジニアリング (exp007 と同一) ──────────────────────────────────

def compute_per_well_features(df_full: pd.DataFrame) -> pd.DataFrame:
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

        md_diff_20 = md[-1] - md[-n20] if n20 > 1 else 1.0
        md_diff_5 = md[-1] - md[-n5] if n5 > 1 else 1.0

        tvt_slope_20 = (tvt[-1] - tvt[-n20]) / md_diff_20 if abs(md_diff_20) > 1e-6 else 0.0
        tvt_slope_5 = (tvt[-1] - tvt[-n5]) / md_diff_5 if abs(md_diff_5) > 1e-6 else 0.0
        tvt_delta_20 = tvt[-1] - tvt[-n20]
        tvt_curvature = tvt_slope_5 - tvt_slope_20

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

        records.append({
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
        })
    return pd.DataFrame(records)


def add_per_row_features(df: pd.DataFrame) -> pd.DataFrame:
    md_safe = np.where(
        df["delta_MD_from_PS"].to_numpy(dtype=float) < 1.0,
        1.0,
        df["delta_MD_from_PS"].to_numpy(dtype=float),
    )
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
    per_well = compute_per_well_features(df)
    df = df.merge(per_well, on="well_id", how="left")
    df = add_per_row_features(df)
    return df


# ─── LGB 学習 ────────────────────────────────────────────────────────────────

def train_lgb(train_t: pd.DataFrame, test_t: pd.DataFrame, y_delta: pd.Series):
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

    oof_pred_delta = np.zeros(len(train_t), dtype=float)
    test_pred_delta = np.zeros(len(test_t), dtype=float)
    fold_rows = []
    importances = []
    n_folds = int(train_t["fold"].nunique())

    for fold in sorted(train_t["fold"].unique()):
        valid_mask = train_t["fold"].eq(fold).to_numpy()
        train_mask = ~valid_mask
        print(f"fold {fold}: train={int(train_mask.sum())}, valid={int(valid_mask.sum())}")

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

        valid_delta = model.predict(
            train_t.loc[valid_mask, ALL_FEATURES],
            num_iteration=model.best_iteration_,
        )
        oof_pred_delta[valid_mask] = valid_delta
        fold_pred_tvt = train_t.loc[valid_mask, "last_known_TVT"].to_numpy(dtype=float) + valid_delta
        fold_rmse = tvt_rmse(train_t.loc[valid_mask, "TVT"], fold_pred_tvt)
        best_iter = int(model.best_iteration_ or model.n_estimators)
        fold_rows.append({
            "fold": int(fold),
            "n_rows": int(valid_mask.sum()),
            "rmse": fold_rmse,
            "best_iteration": best_iter,
        })
        print(f"  fold {fold} RMSE={fold_rmse:.6f}, best_iter={best_iter}")

        test_pred_delta += (
            model.predict(test_t[ALL_FEATURES], num_iteration=model.best_iteration_)
            / n_folds
        )
        importances.append(pd.DataFrame({
            "fold": int(fold),
            "feature": ALL_FEATURES,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }))

    return oof_pred_delta, test_pred_delta, pd.DataFrame(fold_rows), pd.concat(importances, ignore_index=True)


# ─── anchor guard ────────────────────────────────────────────────────────────

def apply_anchor_guard(
    train_t: pd.DataFrame,
    oof_pred_delta: np.ndarray,
    anchor_oof_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LGB OOF に anchor guard を適用して blended OOF を返す。"""
    # anchor OOF をロード
    anchor_oof = pd.read_csv(
        anchor_oof_path, usecols=["well_id", "row_idx", "pred_tvt"]
    ).rename(columns={"pred_tvt": "anchor_pred_tvt"})

    oof = train_t[["id", "well_id", "row_idx", "fold", "TVT", "last_known_TVT"]].copy()
    oof["pred_delta"] = oof_pred_delta
    oof["lgbm_pred_tvt"] = oof["last_known_TVT"].astype(float) + oof["pred_delta"]

    oof = oof.merge(anchor_oof, on=["well_id", "row_idx"], how="left")
    assert not oof["anchor_pred_tvt"].isna().any(), "anchor OOF に欠損があります"

    # per-well guard features
    well_stats = oof.groupby("well_id", as_index=False).agg(
        mean_pred_delta=("pred_delta", "mean"),
        hidden_length=("lgbm_pred_tvt", "count"),  # placeholder, use base parquet value
    )
    # hidden_length は train_t から取る（row数ではなく実測値）
    hl = (
        train_t[["well_id", "hidden_length"]]
        .drop_duplicates("well_id")
    )
    well_stats = well_stats.drop(columns=["hidden_length"]).merge(hl, on="well_id", how="left")

    # ガードルール: |mean_pred_delta| > 30 OR hidden_length > 8000 → full anchor
    full_anchor_mask = (
        well_stats["mean_pred_delta"].abs().gt(ABS_MEAN_DELTA_THR)
        | well_stats["hidden_length"].astype(float).gt(HIDDEN_LENGTH_THR)
    )
    well_stats["guard_rule"] = "no_guard"
    well_stats.loc[full_anchor_mask, "guard_rule"] = "full_anchor"

    rule_map = well_stats.set_index("well_id")["guard_rule"].to_dict()
    oof["guard_rule"] = oof["well_id"].map(rule_map)
    oof["alpha"] = oof["guard_rule"].map({"full_anchor": 0.0, "no_guard": 1.0})

    oof[PRED_COL] = (
        oof["alpha"] * oof["lgbm_pred_tvt"]
        + (1.0 - oof["alpha"]) * oof["anchor_pred_tvt"]
    )
    oof["error"] = oof[PRED_COL] - oof["TVT"].astype(float)
    oof["abs_error"] = oof["error"].abs()

    return oof, well_stats


def apply_guard_to_submission(
    test_t: pd.DataFrame,
    test_pred_delta: np.ndarray,
    test_base: pd.DataFrame,
    anchor_sub_path: Path,
    well_stats_train: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> pd.DataFrame:
    """test 提出にガードを適用。test well の guard_rule を train rule から推定。"""
    anchor_sub = pd.read_csv(anchor_sub_path).rename(columns={"tvt": "anchor_tvt"})

    lgbm_pred_tvt = test_t["last_known_TVT"].to_numpy(dtype=float) + test_pred_delta

    # per-test-well mean pred_delta と hidden_length
    test_temp = test_t[["id", "well_id", "last_known_TVT", "hidden_length"]].copy()
    test_temp["lgbm_pred_tvt"] = lgbm_pred_tvt
    test_temp["pred_delta"] = test_pred_delta

    test_well_stats = test_temp.groupby("well_id", as_index=False).agg(
        mean_pred_delta=("pred_delta", "mean"),
    )
    hl_test = (
        test_temp[["well_id", "hidden_length"]]
        .drop_duplicates("well_id")
    )
    test_well_stats = test_well_stats.merge(hl_test, on="well_id", how="left")

    full_anchor_mask = (
        test_well_stats["mean_pred_delta"].abs().gt(ABS_MEAN_DELTA_THR)
        | test_well_stats["hidden_length"].astype(float).gt(HIDDEN_LENGTH_THR)
    )
    test_well_stats["guard_rule"] = "no_guard"
    test_well_stats.loc[full_anchor_mask, "guard_rule"] = "full_anchor"

    print("Guard rule 分布 (test):")
    print(test_well_stats["guard_rule"].value_counts().to_string())

    # test LGBM submission を作成（一時的）
    test_df_copy = test_base.copy()
    test_df_copy.loc[test_df_copy["is_target"].astype(bool), PRED_COL] = lgbm_pred_tvt
    lgbm_sub = build_submission(test_df_copy, sample_submission, PRED_COL)
    lgbm_sub = lgbm_sub.rename(columns={"tvt": "lgbm_tvt"})

    merged = lgbm_sub.merge(anchor_sub, on="id", how="left", validate="one_to_one")
    merged["well_id"] = merged["id"].str.rsplit("_", n=1).str[0]

    rule_map = test_well_stats.set_index("well_id")["guard_rule"].to_dict()
    merged["guard_rule"] = merged["well_id"].map(rule_map).fillna("no_guard")
    merged["alpha"] = merged["guard_rule"].map({"full_anchor": 0.0, "no_guard": 1.0})
    merged["tvt"] = merged["alpha"] * merged["lgbm_tvt"] + (1.0 - merged["alpha"]) * merged["anchor_tvt"]

    result = sample_submission[["id"]].merge(merged[["id", "tvt"]], on="id", how="left")
    assert not result["tvt"].isna().any(), "submission に欠損 tvt があります"
    return result, test_well_stats


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"[{EXP_ID}] n_estimators=1500 + 双方向 anchor guard を実行します。")
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
        raise ValueError(f"特徴量不足 train={missing_train} test={missing_test}")

    train_t = target_rows(train)
    test_t = target_rows(test)
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    # ── LGB 学習 ──────────────────────────────────────────────────────────────
    print("\n--- LGB 学習 (n_estimators=1500) ---")
    oof_pred_delta, test_pred_delta, cv, importances = train_lgb(train_t, test_t, y_delta)

    lgbm_rmse_raw = tvt_rmse(
        train_t["TVT"],
        train_t["last_known_TVT"].astype(float) + oof_pred_delta,
    )
    print(f"\nLGB raw CV RMSE: {lgbm_rmse_raw:.6f}")
    print("Fold別:")
    print(cv[["fold", "rmse", "best_iteration"]].to_string(index=False))

    cv.to_csv(EXP_DIR / "cv.csv", index=False)

    # ── Anchor guard ──────────────────────────────────────────────────────────
    print("\n--- Anchor guard 適用 ---")
    anchor_oof_path = Path("experiments") / ANCHOR_EXP / "oof.csv"
    oof_blended, well_stats_train = apply_anchor_guard(train_t, oof_pred_delta, anchor_oof_path)

    print("Guard rule 分布 (train):")
    print(well_stats_train["guard_rule"].value_counts().to_string())

    # per-rule RMSE
    for rule in ["full_anchor", "no_guard"]:
        subset = oof_blended[oof_blended["guard_rule"].eq(rule)]
        if len(subset) == 0:
            continue
        n_w = subset["well_id"].nunique()
        r_blend = tvt_rmse(subset["TVT"], subset[PRED_COL])
        r_lgb = tvt_rmse(subset["TVT"], subset["lgbm_pred_tvt"])
        r_anc = tvt_rmse(subset["TVT"], subset["anchor_pred_tvt"])
        print(f"  {rule:12s} ({n_w:3d} wells)  blended={r_blend:.4f}  lgbm={r_lgb:.4f}  anchor={r_anc:.4f}")

    blended_rmse = tvt_rmse(oof_blended["TVT"], oof_blended[PRED_COL])
    improvement_vs_exp007 = 13.867054 - blended_rmse
    print(f"\nBlended CV RMSE: {blended_rmse:.6f}  (exp007比 {improvement_vs_exp007:+.6f})")

    oof_blended.to_csv(EXP_DIR / "oof.csv", index=False)
    well_stats_train.to_csv(EXP_DIR / "well_guard_flags.csv", index=False)

    # ── Feature importance ────────────────────────────────────────────────────
    fi_summary = (
        importances.groupby("feature", as_index=False)[["importance_gain", "importance_split"]]
        .mean()
        .sort_values("importance_gain", ascending=False)
    )
    fi_summary.to_csv(EXP_DIR / "feature_importance.csv", index=False)

    # ── Submission ────────────────────────────────────────────────────────────
    print("\n--- Submission 生成 ---")
    anchor_sub_path = Path("experiments") / ANCHOR_EXP / "submission.csv"
    submission, test_well_stats = apply_guard_to_submission(
        test_t, test_pred_delta, test, anchor_sub_path, well_stats_train, sample_submission
    )
    submission.to_csv(EXP_DIR / "submission.csv", index=False)
    print(f"submission rows: {len(submission)}")

    # ── result.json ───────────────────────────────────────────────────────────
    anchor_oof = pd.read_csv(anchor_oof_path, usecols=["well_id", "row_idx", "TVT", "pred_tvt"])
    anchor_oof = anchor_oof.rename(columns={"pred_tvt": "anchor_pred_tvt"})
    anchor_rmse = tvt_rmse(anchor_oof["TVT"], anchor_oof["anchor_pred_tvt"])

    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "model": "LightGBMRegressor + anchor_guard",
        "target": "TVT - last_known_TVT",
        "prediction": "alpha * lgbm_pred_tvt + (1-alpha) * anchor_pred_tvt",
        "metric": "TVT_abs_RMSE",
        "lgbm_raw_cv_rmse": lgbm_rmse_raw,
        "blended_cv_rmse": blended_rmse,
        "anchor_cv_rmse": anchor_rmse,
        "exp007_cv_rmse": 13.867054,
        "improvement_vs_exp007": round(improvement_vs_exp007, 6),
        "improvement_vs_lgbm_raw": round(lgbm_rmse_raw - blended_rmse, 6),
        "guard_rules": {
            "full_anchor": f"|mean_pred_delta| > {ABS_MEAN_DELTA_THR} OR hidden_length > {HIDDEN_LENGTH_THR}",
            "no_guard": "otherwise",
        },
        "alphas": {"full_anchor": 0.0, "no_guard": 1.0},
        "train_rule_counts": well_stats_train["guard_rule"].value_counts().to_dict(),
        "test_rule_counts": test_well_stats["guard_rule"].value_counts().to_dict(),
        "n_estimators_max": 1500,
        "early_stopping": 50,
        "fold_results": cv.to_dict(orient="records"),
        "n_oof_rows": int(len(oof_blended)),
        "n_submission_rows": int(len(submission)),
        "leak_risk": "low",
        "notes": (
            "exp007 の fold4 が n_estimators=600 を使い切ったため 1500 に増加。"
            "guard 条件を正負対称 (|mean_pred_delta|>30) に拡張し、"
            "hidden_length>8000 との OR 条件で full anchor に切り替え。"
        ),
    }
    write_json(EXP_DIR / "result.json", result)

    # ── notes.md ─────────────────────────────────────────────────────────────
    def rows_to_md(df: pd.DataFrame) -> str:
        header = " | ".join(str(c) for c in df.columns)
        sep = " | ".join(["---"] * len(df.columns))
        return f"| {header} |\n| {sep} |\n" + "\n".join(
            f"| {' | '.join(str(v) for v in r)} |"
            for r in df.itertuples(index=False)
        )

    cv_disp = cv.copy()
    cv_disp["rmse"] = cv_disp["rmse"].round(6)
    top_fi = fi_summary.head(20).copy()
    top_fi["importance_gain"] = top_fi["importance_gain"].round(0).astype(int)

    train_rule_disp = well_stats_train["guard_rule"].value_counts().reset_index()
    train_rule_disp.columns = ["guard_rule", "n_wells"]

    notes = f"""# {EXP_ID}

## 目的

exp007 の後処理として、双方向 anchor guard を適用して catastrophic 予測を抑制する。

## 発見と仮説

- exp007 fold4: best_iteration=600 (n_estimators 上限) → underfitting
- `|mean_pred_delta7| > 30` の wells で RMSE 38-48 の catastrophic 予測が発生
- exp006 は `mean_pred_delta > 30` の正方向のみガード → 負方向ガード漏れ

## 変更内容

- **n_estimators**: 600 → 1500 (early_stopping=50 は同じ)
- **ガードルール** (exp007b 独自):
  - `|mean_pred_delta| > {ABS_MEAN_DELTA_THR}` または `hidden_length > {HIDDEN_LENGTH_THR}` → alpha=0 (full anchor)
  - それ以外 → alpha=1.0 (LGB そのまま)

## 結果

| metric | value |
|---|---|
| LGB raw CV RMSE | {lgbm_rmse_raw:.6f} |
| Blended CV RMSE | {blended_rmse:.6f} |
| exp007 CV RMSE | 13.867054 |
| 改善幅 vs exp007 | {improvement_vs_exp007:+.6f} |
| LGB raw 改善幅 | {lgbm_rmse_raw - blended_rmse:+.6f} |

## Fold別RMSE

{rows_to_md(cv_disp[['fold', 'rmse', 'best_iteration']])}

## Guard Rule 分布 (train)

{rows_to_md(train_rule_disp)}

## Top 20 Feature Importance (gain)

{rows_to_md(top_fi[['feature', 'importance_gain']])}

## リーク懸念

- `mean_pred_delta` は LGB 予測からのみ計算 (OOF/test prediction)。train target には触れない。リーク低。
- `hidden_length` は base table に存在し test でも同じ定義。リーク低。
- guard の alpha は 0 or 1 の固定値 (grid search なし) → OOF 過学習なし。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")

    print(f"\n[{EXP_ID}] 完了")
    print(f"  LGB raw CV:  {lgbm_rmse_raw:.6f}")
    print(f"  Blended CV:  {blended_rmse:.6f}  (exp007比 {improvement_vs_exp007:+.6f})")
    print(f"  出力: {EXP_DIR}/")


if __name__ == "__main__":
    main()
