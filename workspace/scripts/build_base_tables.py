from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rogii.data.build_base import build_all_base_tables


def main() -> None:
    print("base table生成を開始します。data/raw は読み取り専用で扱います。")
    outputs = build_all_base_tables(
        raw_dir=PROJECT_ROOT / "data/raw",
        interim_dir=PROJECT_ROOT / "data/interim",
        processed_dir=PROJECT_ROOT / "data/processed",
    )
    for name, path in outputs.items():
        print(f"生成完了: {name} -> {path}")


if __name__ == "__main__":
    main()
