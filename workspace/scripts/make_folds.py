#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

N_SPLITS_DEFAULT = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/train_base_v001.parquet")
    parser.add_argument("--strategy", choices=["group_well"], default="group_well")
    parser.add_argument("--output", default="data/folds/folds_group_well_v001.csv")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    required = {"split", "well_id", "is_target", "hidden_length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fold作成に必要なカラムがありません: {sorted(missing)}")

    well_stats = (
        df[df["is_target"]]
        .groupby(["split", "well_id"], as_index=False)
        .agg(target_rows=("row_idx", "size"))
    )
    if well_stats.empty:
        raise ValueError("target rows がないためfoldを作成できません。")

    # target row数が大きいwellから、現在target row数が最も少ないfoldへ割り当てる。
    # これによりwell単位を守りつつ、評価行数の偏りを小さくする。
    fold_loads = [0 for _ in range(args.n_splits)]
    assignments: list[dict[str, object]] = []
    for row in well_stats.sort_values("target_rows", ascending=False).itertuples(index=False):
        fold = min(range(args.n_splits), key=lambda idx: fold_loads[idx])
        fold_loads[fold] += int(row.target_rows)
        assignments.append(
            {
                "split": row.split,
                "well_id": row.well_id,
                "fold": fold,
                "target_rows": int(row.target_rows),
            }
        )

    folds = pd.DataFrame(assignments).sort_values(["fold", "well_id"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_path, index=False)

    summary = {
        "strategy": args.strategy,
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "n_splits": args.n_splits,
        "fold_target_rows": folds.groupby("fold")["target_rows"].sum().astype(int).to_dict(),
        "fold_wells": folds.groupby("fold")["well_id"].nunique().astype(int).to_dict(),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
