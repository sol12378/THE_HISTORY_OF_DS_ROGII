#!/usr/bin/env python3
"""Create balanced well-level folds for ROGII target rows."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from rogii_paths import ensure_workspace_dirs, workspace_path


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(workspace_path("data", "processed", "train_base_v001.parquet")))
    parser.add_argument("--output", default=str(workspace_path("data", "folds", "folds_group_well_v002.csv")))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--exp-id", default="exp110_group_well_cv_v2")
    args = parser.parse_args()

    ensure_workspace_dirs()
    df = pd.read_parquet(args.input) if str(args.input).endswith(".parquet") else pd.read_csv(args.input)
    required = {"split", "well_id", "is_target", "row_idx", "hidden_length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for fold creation: {sorted(missing)}")

    well_stats = (
        df.loc[df["is_target"].astype(bool)]
        .groupby(["split", "well_id"], as_index=False)
        .agg(target_rows=("row_idx", "size"), hidden_length=("hidden_length", "max"))
    )
    if well_stats.empty:
        raise ValueError("No target rows found for fold creation.")

    fold_loads = [0 for _ in range(args.n_splits)]
    rows: list[dict] = []
    for row in well_stats.sort_values(["target_rows", "well_id"], ascending=[False, True]).itertuples(index=False):
        fold = min(range(args.n_splits), key=lambda idx: fold_loads[idx])
        fold_loads[fold] += int(row.target_rows)
        rows.append(
            {
                "split": row.split,
                "well_id": row.well_id,
                "fold": fold,
                "target_rows": int(row.target_rows),
                "hidden_length": int(row.hidden_length),
            }
        )

    folds = pd.DataFrame(rows).sort_values(["fold", "well_id"]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False)

    fold_summary = (
        folds.groupby("fold")
        .agg(n_wells=("well_id", "nunique"), target_rows=("target_rows", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    result = {
        "exp_id": args.exp_id,
        "created_at": _now(),
        "status": "completed",
        "strategy": "balanced_group_well",
        "input": str(args.input),
        "output": str(out),
        "n_splits": args.n_splits,
        "n_wells": int(folds["well_id"].nunique()),
        "fold_summary": fold_summary,
        "leakage_risk": "low",
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    exp_dir = workspace_path("experiments", args.exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (exp_dir / "notes.md").write_text(
        "# " + args.exp_id + "\n\nBalanced GroupKFold by `well_id`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
