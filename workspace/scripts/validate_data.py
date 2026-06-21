from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_table(path: Path) -> pd.DataFrame:
    _assert(path.exists(), f"必要なファイルがありません: {path}")
    return pd.read_parquet(path)


def validate_sample_submission_ids(test_base: pd.DataFrame, sample_submission_path: Path) -> dict[str, object]:
    sample = pd.read_csv(sample_submission_path)
    target_ids = test_base.loc[test_base["is_target"], "id"].astype(str).tolist()
    sample_ids = sample["id"].astype(str).tolist()
    _assert(target_ids == sample_ids, "sample_submission.id と test target id が完全一致しません。")
    return {"n_submission_ids": len(sample_ids)}


def validate_tvt_input_tail(df: pd.DataFrame, table_name: str) -> dict[str, object]:
    violations: list[str] = []
    for well_id, group in df.groupby("well_id", sort=False):
        missing = group["TVT_input"].isna().to_numpy()
        if not missing.any():
            continue
        first_missing = int(np.argmax(missing))
        if not bool(missing[first_missing:].all()):
            violations.append(str(well_id))
    _assert(
        not violations,
        f"{table_name}: TVT_input 欠損tailが連続していないwellがあります: {violations[:10]}",
    )
    return {"checked_wells": int(df["well_id"].nunique()), "violations": 0}


def validate_train_known_tvt(train_base: pd.DataFrame, tolerance: float = 1e-4) -> dict[str, object]:
    known = train_base[train_base["row_idx"] < train_base["ps_idx"]].copy()
    diff = (known["TVT_input"] - known["TVT"]).abs()
    bad = known[diff > tolerance]
    _assert(
        bad.empty,
        f"train の Prediction Start 前で TVT_input と TVT が一致しない行があります: {len(bad)} rows",
    )
    return {
        "checked_rows": int(len(known)),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "tolerance": tolerance,
    }


def validate_required_columns(df: pd.DataFrame, table_name: str) -> dict[str, object]:
    required = [
        "split",
        "well_id",
        "row_idx",
        "source_path",
        "MD",
        "X",
        "Y",
        "Z",
        "GR",
        "TVT_input",
        "TVT",
        "id",
        "is_target",
        "is_known_tvt",
        "is_gr_missing",
        "ps_idx",
        "n_rows_in_well",
        "known_length",
        "hidden_length",
        "last_known_TVT",
        "last_known_MD",
        "last_known_X",
        "last_known_Y",
        "last_known_Z",
        "delta_MD_from_PS",
        "delta_X_from_PS",
        "delta_Y_from_PS",
        "delta_Z_from_PS",
        "post_ps_step",
        "row_frac",
    ]
    missing = [column for column in required if column not in df.columns]
    _assert(not missing, f"{table_name}: 必須カラムがありません: {missing}")
    return {"n_columns": int(len(df.columns))}


def validate_meta_files(paths: list[Path]) -> dict[str, object]:
    missing = [path.with_suffix(path.suffix + ".meta.json") for path in paths if not path.with_suffix(path.suffix + ".meta.json").exists()]
    _assert(not missing, f"meta json がありません: {missing}")
    return {"checked_meta_files": len(paths)}


def validate_fold_file(train_base: pd.DataFrame, fold_path: Path, n_splits: int = 5) -> dict[str, object]:
    _assert(fold_path.exists(), f"fold file がありません: {fold_path}")
    folds = pd.read_csv(fold_path)
    required = {"split", "well_id", "fold", "target_rows"}
    missing = required - set(folds.columns)
    _assert(not missing, f"fold file に必要なカラムがありません: {sorted(missing)}")
    _assert(folds["well_id"].is_unique, "fold file の well_id が重複しています。")

    train_wells = set(train_base["well_id"].unique())
    fold_wells = set(folds["well_id"].unique())
    _assert(train_wells == fold_wells, "train wells と fold wells が一致しません。")

    fold_values = sorted(folds["fold"].unique().tolist())
    _assert(fold_values == list(range(n_splits)), f"fold値が期待と違います: {fold_values}")

    actual_target_rows = (
        train_base[train_base["is_target"]]
        .groupby("well_id")
        .size()
        .rename("actual_target_rows")
    )
    merged = folds.merge(actual_target_rows, on="well_id", how="left")
    bad = merged[merged["target_rows"] != merged["actual_target_rows"]]
    _assert(bad.empty, f"fold file の target_rows がbase tableと一致しません: {len(bad)} wells")

    return {
        "n_wells": int(len(folds)),
        "n_splits": int(n_splits),
        "fold_target_rows": folds.groupby("fold")["target_rows"].sum().astype(int).to_dict(),
        "fold_wells": folds.groupby("fold")["well_id"].nunique().astype(int).to_dict(),
    }


def main() -> None:
    processed_dir = PROJECT_ROOT / "data/processed"
    train_path = processed_dir / "train_base_v001.parquet"
    test_path = processed_dir / "test_base_v001.parquet"
    typewell_train_path = processed_dir / "typewell_train_base_v001.parquet"
    typewell_test_path = processed_dir / "typewell_test_base_v001.parquet"
    inventory_path = PROJECT_ROOT / "data/interim/raw_inventory_v001.json"
    fold_path = PROJECT_ROOT / "data/folds/folds_group_well_v001.csv"

    print("データ検証を開始します。")
    train_base = _load_table(train_path)
    test_base = _load_table(test_path)
    typewell_train = _load_table(typewell_train_path)
    typewell_test = _load_table(typewell_test_path)
    _assert(inventory_path.exists(), f"inventory がありません: {inventory_path}")

    results = {
        "inventory": json.loads(inventory_path.read_text(encoding="utf-8"))["by_split_table"],
        "train_required_columns": validate_required_columns(train_base, "train_base"),
        "test_required_columns": validate_required_columns(test_base, "test_base"),
        "meta_files": validate_meta_files([train_path, test_path, typewell_train_path, typewell_test_path]),
        "sample_submission_ids": validate_sample_submission_ids(
            test_base,
            PROJECT_ROOT / "data/raw/sample_submission.csv",
        ),
        "train_tvt_input_tail": validate_tvt_input_tail(train_base, "train_base"),
        "test_tvt_input_tail": validate_tvt_input_tail(test_base, "test_base"),
        "train_known_tvt": validate_train_known_tvt(train_base),
        "fold_file": validate_fold_file(train_base, fold_path),
        "typewell_rows": {
            "train": int(len(typewell_train)),
            "test": int(len(typewell_test)),
        },
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("データ検証は成功しました。")


if __name__ == "__main__":
    main()
