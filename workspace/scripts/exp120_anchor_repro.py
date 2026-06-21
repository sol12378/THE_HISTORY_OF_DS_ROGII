#!/usr/bin/env python3
"""Leak-safe anchor baseline reproduction for ROGII."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rogii_paths import ensure_workspace_dirs, raw_dir, workspace_path


PRED_COL = "pred_tvt"


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def _rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(workspace_path("data", "processed", "train_base_v001.parquet")))
    parser.add_argument("--test", default=str(workspace_path("data", "processed", "test_base_v001.parquet")))
    parser.add_argument("--folds", default=str(workspace_path("data", "folds", "folds_group_well_v002.csv")))
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--exp-id", default="exp120_anchor_repro")
    args = parser.parse_args()

    ensure_workspace_dirs()
    raw = raw_dir(None)
    sample_path = args.sample_submission or str(raw / "sample_submission.csv")
    train = pd.read_parquet(args.train)
    test = pd.read_parquet(args.test)
    folds = pd.read_csv(args.folds)
    sample = pd.read_csv(sample_path)

    train_t = train.loc[train["is_target"].astype(bool)].copy()
    train_t = train_t.merge(
        folds.loc[folds["split"].eq("train"), ["well_id", "fold"]],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if train_t["fold"].isna().any():
        missing = train_t.loc[train_t["fold"].isna(), "well_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Missing folds for wells: {missing}")
    train_t["fold"] = train_t["fold"].astype(int)
    train_t[PRED_COL] = pd.to_numeric(train_t["last_known_TVT"], errors="coerce")
    train_t["error"] = train_t[PRED_COL] - pd.to_numeric(train_t["TVT"], errors="coerce")
    train_t["abs_error"] = train_t["error"].abs()

    fold_rows = []
    for fold, grp in train_t.groupby("fold", sort=True):
        fold_rows.append(
            {
                "fold": int(fold),
                "n_rows": int(len(grp)),
                "rmse": _rmse(grp["TVT"], grp[PRED_COL]),
            }
        )
    cv = pd.DataFrame(fold_rows)
    per_well_rows = []
    for (well_id, fold), grp in train_t.groupby(["well_id", "fold"], sort=False):
        per_well_rows.append(
            {
                "well_id": well_id,
                "fold": int(fold),
                "n_rows": int(len(grp)),
                "rmse": _rmse(grp["TVT"], grp[PRED_COL]),
            }
        )
    per_well = pd.DataFrame(per_well_rows)

    test_t = test.loc[test["is_target"].astype(bool), ["id", "last_known_TVT"]].copy()
    pred = test_t.rename(columns={"last_known_TVT": "tvt"})[["id", "tvt"]]
    submission = sample[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if submission["tvt"].isna().any():
        raise ValueError("Anchor submission contains missing predictions.")

    exp_dir = workspace_path("experiments", args.exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    train_t[["id", "well_id", "row_idx", "fold", "TVT", "last_known_TVT", PRED_COL, "error", "abs_error"]].to_csv(
        exp_dir / "oof.csv", index=False
    )
    cv.to_csv(exp_dir / "cv.csv", index=False)
    per_well.to_csv(exp_dir / "per_well.csv", index=False)
    submission.to_csv(exp_dir / "submission.csv", index=False)

    overall = _rmse(train_t["TVT"], train_t[PRED_COL])
    result = {
        "exp_id": args.exp_id,
        "created_at": _now(),
        "status": "completed",
        "model": "anchor_last_known_TVT",
        "metric": "rmse",
        "cv_rmse": overall,
        "cv_mean": float(cv["rmse"].mean()),
        "cv_std": float(cv["rmse"].std(ddof=0)),
        "n_oof_rows": int(len(train_t)),
        "n_submission_rows": int(len(submission)),
        "leakage_risk": "low",
        "notes": "Predict every target row with the last known TVT_input for that well.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (exp_dir / "notes.md").write_text(
        f"# {args.exp_id}\n\nAnchor baseline: predict target rows with `last_known_TVT`.\n\n"
        f"- CV RMSE: {overall:.6f}\n"
        f"- OOF rows: {len(train_t)}\n"
        f"- Submission rows: {len(submission)}\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
