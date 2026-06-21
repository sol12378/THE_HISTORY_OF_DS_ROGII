#!/usr/bin/env python3
"""Run anchor baseline: predict last_known_TVT for target rows."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rogii.training.baselines import (
    PRED_COL,
    attach_folds,
    build_submission,
    ensure_exp_dir,
    load_base_inputs,
    now_jst,
    target_rows,
    tvt_rmse,
    write_json,
)


EXP_ID = "exp001_anchor_baseline"
EXP_DIR = Path("experiments") / EXP_ID


def main() -> None:
    print("anchor baseline を開始します。仮説: PS直前の last_known_TVT をそのまま延長した場合の下限性能を確認します。")
    ensure_exp_dir(EXP_DIR)

    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )
    train = attach_folds(train, folds)

    train[PRED_COL] = train["last_known_TVT"].astype(float)
    test[PRED_COL] = test["last_known_TVT"].astype(float)

    oof = target_rows(train)[["id", "well_id", "row_idx", "fold", "TVT", "last_known_TVT", PRED_COL]].copy()
    oof["error"] = oof[PRED_COL] - oof["TVT"]
    oof["abs_error"] = oof["error"].abs()
    oof.to_csv(EXP_DIR / "oof.csv", index=False)

    rows = []
    for fold, fold_df in oof.groupby("fold", sort=True):
        rows.append({"fold": int(fold), "n_rows": int(len(fold_df)), "rmse": tvt_rmse(fold_df["TVT"], fold_df[PRED_COL])})
    cv = pd.DataFrame(rows)
    cv.to_csv(EXP_DIR / "cv.csv", index=False)

    submission = build_submission(test, sample_submission, PRED_COL)
    submission.to_csv(EXP_DIR / "submission.csv", index=False)

    overall_rmse = tvt_rmse(oof["TVT"], oof[PRED_COL])
    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "model": "anchor_last_known_TVT",
        "metric": "TVT_abs_RMSE",
        "cv_rmse": overall_rmse,
        "cv_mean": float(cv["rmse"].mean()),
        "cv_std": float(cv["rmse"].std(ddof=0)),
        "n_oof_rows": int(len(oof)),
        "n_submission_rows": int(len(submission)),
        "leak_risk": "low",
        "notes": "target rows の TVT を last_known_TVT で予測する単純基準線。",
    }
    write_json(EXP_DIR / "result.json", result)

    notes = f"""# {EXP_ID}

## 目的

[[Anchor_Features]] の最小構成として、target rows の `TVT` を `last_known_TVT` だけで予測する基準線を作る。

## 仮説

PS直前の既知TVTを水平に延長するだけでも、後続の delta model が超えるべき下限性能として有用。

## 変更内容

- 入力: `data/processed/train_base_v001.parquet`, `data/processed/test_base_v001.parquet`, `data/folds/folds_group_well_v001.csv`
- 予測: `pred_tvt = last_known_TVT`
- 評価: `is_target=True` の行だけで TVT絶対値RMSE

## 結果

- overall CV RMSE: {overall_rmse:.6f}
- fold mean RMSE: {result["cv_mean"]:.6f}
- fold std RMSE: {result["cv_std"]:.6f}
- OOF rows: {len(oof)}
- submission rows: {len(submission)}

## リーク懸念

`last_known_TVT` は target 区間より前の既知TVTから作られた anchor であり、target row の真値TVTや test の未知TVTは使っていない。リーク懸念は低い。ただし base table 生成ロジックが PS 以後の真値を参照していない前提に依存する。

## 次アクション

`TVT - last_known_TVT` を目的変数にした LightGBM baseline と比較し、anchor からの改善量を見る。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(f"完了: {EXP_DIR} / CV RMSE={overall_rmse:.6f}")


if __name__ == "__main__":
    main()
