#!/usr/bin/env python3
"""Build ROGII processed base tables and write a data-contract experiment."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from rogii_paths import ensure_workspace_dirs, raw_dir, workspace_path

SRC = workspace_path("src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rogii.data.build_base import build_all_base_tables


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--exp-id", default="exp100_data_contract")
    args = parser.parse_args()

    ensure_workspace_dirs()
    raw = raw_dir(args.raw_dir)
    if not raw.exists():
        raise FileNotFoundError(f"ROGII raw dir not found: {raw}")

    outputs = build_all_base_tables(
        raw_dir=raw,
        interim_dir=workspace_path("data", "interim"),
        processed_dir=workspace_path("data", "processed"),
    )

    train_h = len(list((raw / "train").glob("*__horizontal_well.csv")))
    test_h = len(list((raw / "test").glob("*__horizontal_well.csv")))
    train_png = len(list((raw / "train").glob("*.png")))
    sample = pd.read_csv(raw / "sample_submission.csv")
    train_base = pd.read_parquet(workspace_path("data", "processed", "train_base_v001.parquet"))
    test_base = pd.read_parquet(workspace_path("data", "processed", "test_base_v001.parquet"))

    test_target = test_base.loc[test_base["is_target"].astype(bool), "id"].reset_index(drop=True)
    sample_ids = sample["id"].reset_index(drop=True)
    sample_matches_test_tail = bool(sample_ids.equals(test_target))

    overlap_wells = sorted(set(test_base["well_id"]).intersection(set(train_base["well_id"])))
    danger_columns = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "Geology"]

    result = {
        "exp_id": args.exp_id,
        "created_at": _now(),
        "status": "completed",
        "raw_dir": str(raw),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "train_horizontal_wells": train_h,
        "test_horizontal_wells": test_h,
        "train_png": train_png,
        "sample_submission_shape": list(sample.shape),
        "train_base_shape": list(train_base.shape),
        "test_base_shape": list(test_base.shape),
        "test_target_rows": int(test_base["is_target"].sum()),
        "sample_matches_test_tail": sample_matches_test_tail,
        "public_test_train_overlap_wells": overlap_wells,
        "forbidden_train_only_columns": danger_columns,
        "leakage_risk": "documented_public_overlap",
    }

    exp_dir = workspace_path("experiments", args.exp_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    notes = f"""# {args.exp_id}

ROGII raw data contract check.

- raw dir: `{raw}`
- train horizontal wells: {train_h}
- test horizontal wells: {test_h}
- sample submission rows: {len(sample)}
- test target rows: {int(test_base["is_target"].sum())}
- sample ids match test missing tail: {sample_matches_test_tail}
- public test/train overlap wells: {", ".join(overlap_wells)}

Train-only danger columns remain forbidden in leak-safe experiments:
`{", ".join(danger_columns)}`.
"""
    (exp_dir / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
