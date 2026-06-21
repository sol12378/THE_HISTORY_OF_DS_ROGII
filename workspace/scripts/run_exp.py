#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_id = config["exp_id"]
    exp_dir = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    result = {
        "exp_id": exp_id,
        "created_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "status": "initialized",
        "cv_mean": None,
        "cv_std": None,
        "lb_score": None,
        "notes": "Training implementation pending.",
    }
    (exp_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (exp_dir / "notes.md").write_text(f"# {exp_id}\n\n## 目的\n\n## 結果\n\n## 解釈\n", encoding="utf-8")
    print(f"Initialized {exp_dir}")


if __name__ == "__main__":
    main()
