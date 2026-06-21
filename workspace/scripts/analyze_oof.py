#!/usr/bin/env python3
"""Create OOF error slices for ROGII experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rogii.training.baselines import tvt_rmse


DEFAULT_MODEL_EXP = "exp003_lgb_anchor_trajectory"
DEFAULT_ANCHOR_EXP = "exp001_anchor_baseline"
DEFAULT_OUT_EXP = "exp004_oof_error_slicing"
TARGET_KEY = ["well_id", "row_idx"]
HIDDEN_BINS = [0, 199, 499, 999, 1999, np.inf]
HIDDEN_LABELS = ["1-199", "200-499", "500-999", "1000-1999", "2000+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-exp", default=DEFAULT_MODEL_EXP)
    parser.add_argument("--anchor-exp", default=DEFAULT_ANCHOR_EXP)
    parser.add_argument("--out-exp", default=DEFAULT_OUT_EXP)
    parser.add_argument("--train-base", default="data/processed/train_base_v001.parquet")
    return parser.parse_args()


def now_jst() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def metric_row(df: pd.DataFrame, pred_col: str, prefix: str = "") -> dict:
    err = df[pred_col].to_numpy(dtype=float) - df["TVT"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}mae": float(np.mean(abs_err)),
        f"{prefix}bias": float(np.mean(err)),
        f"{prefix}p95_abs_error": float(np.quantile(abs_err, 0.95)),
    }


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_rows"] = int(len(g))
        row["n_wells"] = int(g["well_id"].nunique())
        row.update(metric_row(g, "pred_tvt", "model_"))
        row.update(metric_row(g, "anchor_pred_tvt", "anchor_"))
        row["rmse_improvement_vs_anchor"] = row["anchor_rmse"] - row["model_rmse"]
        row["mae_improvement_vs_anchor"] = row["anchor_mae"] - row["model_mae"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_oof(model_exp: str, anchor_exp: str, train_base_path: Path) -> pd.DataFrame:
    model = pd.read_csv(Path("experiments") / model_exp / "oof.csv")
    anchor = pd.read_csv(Path("experiments") / anchor_exp / "oof.csv", usecols=TARGET_KEY + ["pred_tvt"])
    anchor = anchor.rename(columns={"pred_tvt": "anchor_pred_tvt"})
    base_cols = [
        "well_id",
        "row_idx",
        "is_target",
        "n_rows_in_well",
        "known_length",
        "hidden_length",
        "last_known_TVT",
        "delta_MD_from_PS",
        "delta_X_from_PS",
        "delta_Y_from_PS",
        "delta_Z_from_PS",
        "post_ps_step",
        "row_frac",
        "X",
        "Y",
        "Z",
    ]
    base = pd.read_parquet(train_base_path, columns=base_cols)
    base = base.loc[base["is_target"].astype(bool)].drop(columns=["is_target"]).copy()

    merged = model.merge(anchor, on=TARGET_KEY, how="left", validate="one_to_one")
    merged = merged.merge(base, on=TARGET_KEY, how="left", suffixes=("", "_base"), validate="one_to_one")
    if merged["anchor_pred_tvt"].isna().any():
        raise ValueError("anchor prediction missing after merge")
    if merged["hidden_length"].isna().any():
        raise ValueError("base metadata missing after merge")
    return merged


def add_hidden_length_bin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hidden_length_bin"] = pd.cut(
        out["hidden_length"],
        bins=HIDDEN_BINS,
        labels=HIDDEN_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def trajectory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well_id, g in df.sort_values(TARGET_KEY).groupby("well_id", sort=True):
        x = g["X"].to_numpy(dtype=float)
        y = g["Y"].to_numpy(dtype=float)
        z = g["Z"].to_numpy(dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        dxdy_step = np.sqrt(dx**2 + dy**2)
        heading = np.unwrap(np.arctan2(dy, dx)) if len(dx) else np.array([0.0])
        heading_change = np.abs(np.diff(heading)) if len(heading) > 1 else np.array([0.0])
        curvature_proxy = float(np.nanmean(heading_change)) if len(heading_change) else 0.0
        final_delta_z = float(g["delta_Z_from_PS"].iloc[-1])
        rows.append(
            {
                "well_id": well_id,
                "final_delta_z_from_ps": final_delta_z,
                "mean_abs_dz_step": float(np.nanmean(np.abs(dz))) if len(dz) else 0.0,
                "std_dz_step": float(np.nanstd(dz)) if len(dz) else 0.0,
                "mean_abs_dxdy_step": float(np.nanmean(dxdy_step)) if len(dxdy_step) else 0.0,
                "curvature_proxy": curvature_proxy,
                "azimuth_change_proxy": float(np.nanmax(heading) - np.nanmin(heading)) if len(heading) else 0.0,
            }
        )
    metrics = pd.DataFrame(rows)
    z_abs = metrics["final_delta_z_from_ps"].abs()
    flat_thr = float(z_abs.quantile(0.33))
    curved_thr = float(metrics["curvature_proxy"].quantile(0.67))
    metrics["vertical_shape"] = np.select(
        [
            z_abs.le(flat_thr),
            metrics["final_delta_z_from_ps"].gt(flat_thr),
            metrics["final_delta_z_from_ps"].lt(-flat_thr),
        ],
        ["flat", "upward", "downward"],
        default="flat",
    )
    metrics["curvature_shape"] = np.where(metrics["curvature_proxy"].ge(curved_thr), "curved", "smooth")
    metrics["trajectory_shape"] = metrics["vertical_shape"] + "_" + metrics["curvature_shape"]
    return metrics


def top_worst_wells(well_report: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    cols = [
        "well_id",
        "fold",
        "n_rows",
        "hidden_length",
        "model_rmse",
        "anchor_rmse",
        "rmse_improvement_vs_anchor",
        "model_mae",
        "model_bias",
        "model_p95_abs_error",
    ]
    return well_report.sort_values("model_rmse", ascending=False)[cols].head(n)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path("experiments") / args.out_exp
    ensure_dir(out_dir)

    df = load_oof(args.model_exp, args.anchor_exp, Path(args.train_base))
    df = add_hidden_length_bin(df)
    traj = trajectory_metrics(df)
    df = df.merge(traj, on="well_id", how="left", validate="many_to_one")

    overall = {
        "model_exp": args.model_exp,
        "anchor_exp": args.anchor_exp,
        "created_at": now_jst(),
        "n_rows": int(len(df)),
        "n_wells": int(df["well_id"].nunique()),
    }
    overall.update(metric_row(df, "pred_tvt", "model_"))
    overall.update(metric_row(df, "anchor_pred_tvt", "anchor_"))
    overall["rmse_improvement_vs_anchor"] = overall["anchor_rmse"] - overall["model_rmse"]
    overall["mae_improvement_vs_anchor"] = overall["anchor_mae"] - overall["model_mae"]

    well_report = summarize_group(df, ["well_id"])
    well_meta = (
        df.groupby("well_id", as_index=False)
        .agg(
            fold=("fold", "first"),
            hidden_length=("hidden_length", "first"),
            known_length=("known_length", "first"),
            final_delta_z_from_ps=("final_delta_z_from_ps", "first"),
            trajectory_shape=("trajectory_shape", "first"),
            vertical_shape=("vertical_shape", "first"),
            curvature_shape=("curvature_shape", "first"),
        )
    )
    well_report = well_report.merge(well_meta, on="well_id", how="left", validate="one_to_one")
    hidden_report = summarize_group(df, ["hidden_length_bin"])
    shape_report = summarize_group(df, ["trajectory_shape"])
    vertical_report = summarize_group(df, ["vertical_shape"])
    curvature_report = summarize_group(df, ["curvature_shape"])
    fold_report = summarize_group(df, ["fold"])
    worst_report = top_worst_wells(well_report)

    well_report.to_csv(out_dir / "well_error_report.csv", index=False)
    hidden_report.to_csv(out_dir / "hidden_length_error_report.csv", index=False)
    shape_report.to_csv(out_dir / "trajectory_shape_error_report.csv", index=False)
    vertical_report.to_csv(out_dir / "vertical_shape_error_report.csv", index=False)
    curvature_report.to_csv(out_dir / "curvature_shape_error_report.csv", index=False)
    fold_report.to_csv(out_dir / "fold_error_report.csv", index=False)
    worst_report.to_csv(out_dir / "worst_wells_top30.csv", index=False)
    traj.to_csv(out_dir / "trajectory_metrics_by_well.csv", index=False)

    result = {
        "exp_id": args.out_exp,
        "status": "completed",
        "analysis_target": args.model_exp,
        "baseline": args.anchor_exp,
        "metric": "TVT_abs_RMSE",
        "overall": overall,
        "worst_model_rmse_well": worst_report.iloc[0].to_dict(),
        "reports": [
            "well_error_report.csv",
            "hidden_length_error_report.csv",
            "trajectory_shape_error_report.csv",
            "vertical_shape_error_report.csv",
            "curvature_shape_error_report.csv",
            "fold_error_report.csv",
            "worst_wells_top30.csv",
            "trajectory_metrics_by_well.csv",
        ],
        "leak_risk": "low",
        "notes": "OOF predictions and train-base target-row metadata only. No raw data modification.",
    }
    write_json(out_dir / "result.json", result)

    notes = f"""# {args.out_exp}

## 目的

`{args.model_exp}` のOOFを、well別・hidden_length別・trajectory形状別に分解し、CVを落としている原因を特定する。

## 仮説

overall CVの悪化は全wellに均等に出るのではなく、long tailや曲がりの強いtrajectoryを持つ一部wellに集中している。

## 結果サマリ

- model RMSE: {overall["model_rmse"]:.6f}
- anchor RMSE: {overall["anchor_rmse"]:.6f}
- RMSE improvement vs anchor: {overall["rmse_improvement_vs_anchor"]:.6f}
- model MAE: {overall["model_mae"]:.6f}
- model bias: {overall["model_bias"]:.6f}
- rows: {overall["n_rows"]:,}
- wells: {overall["n_wells"]:,}

## 主要成果物

- `well_error_report.csv`
- `hidden_length_error_report.csv`
- `trajectory_shape_error_report.csv`
- `worst_wells_top30.csv`

## リーク懸念

OOF予測、anchor予測、train base tableのtarget-row metadataのみを使った後処理分析。モデル学習やsubmission生成には使っていないためリーク懸念は低い。ただし、この分析から作る特徴量は必ずtrain/test両方で同じ定義にする。
"""
    (out_dir / "notes.md").write_text(notes, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
