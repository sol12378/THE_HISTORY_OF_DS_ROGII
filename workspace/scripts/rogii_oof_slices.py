#!/usr/bin/env python3
"""Build per-well and slice diagnostics from a ROGII OOF file."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rogii_paths import workspace_path


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def _rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--oof", default=None)
    args = parser.parse_args()

    exp_dir = workspace_path("experiments", args.exp_id)
    oof_path = __import__("pathlib").Path(args.oof) if args.oof else exp_dir / "oof.csv"
    oof = pd.read_csv(oof_path)
    pred_col = "pred_tvt" if "pred_tvt" in oof.columns else "prediction"
    if pred_col not in oof.columns:
        raise ValueError("OOF must contain pred_tvt or prediction column.")

    per_well_rows = []
    for (well_id, fold), grp in oof.groupby(["well_id", "fold"], sort=False):
        per_well_rows.append(
            {
                "well_id": well_id,
                "fold": int(fold),
                "n_rows": int(len(grp)),
                "rmse": _rmse(grp["TVT"], grp[pred_col]),
            }
        )
    per_well = pd.DataFrame(per_well_rows).sort_values("rmse", ascending=False)
    per_well.to_csv(exp_dir / "per_well.csv", index=False)

    fold_rows = []
    for fold_id, grp in oof.groupby("fold", sort=True):
        fold_rows.append(
            {"fold": int(fold_id), "n_rows": int(len(grp)), "rmse": _rmse(grp["TVT"], grp[pred_col])}
        )
    fold = pd.DataFrame(fold_rows)
    fold.to_csv(exp_dir / "fold_slices.csv", index=False)

    result = {
        "exp_id": f"{args.exp_id}_oof_slices",
        "source_exp_id": args.exp_id,
        "created_at": _now(),
        "status": "completed",
        "oof": str(oof_path),
        "overall_rmse": _rmse(oof["TVT"], oof[pred_col]),
        "worst_wells": per_well.head(20).to_dict(orient="records"),
        "fold_slices": fold.to_dict(orient="records"),
    }
    (exp_dir / "slice_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
