#!/usr/bin/env python3
"""exp005: Worst well pattern analysis and long-tail decomposition.

Goals:
  1. Fine-grained long-tail bins (2000+ → 6 sub-bins)
  2. Characterise wells where anchor is strong but LightGBM destroys it
  3. Find k-NN similar wells to the 3 key anchor-blown wells
"""

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

# ─── constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_EXP = "exp003_lgb_anchor_trajectory"
DEFAULT_ANCHOR_EXP = "exp001_anchor_baseline"
DEFAULT_OUT_EXP = "exp005_worst_well_analysis"
DEFAULT_PREV_ANALYSIS = "exp004_oof_error_slicing"
TARGET_KEY = ["well_id", "row_idx"]

# Finer long-tail bins (2000+ split into 6)
LONGTAIL_BINS = [0, 199, 499, 999, 1999, 2999, 3999, 4999, 5999, 7999, np.inf]
LONGTAIL_LABELS = [
    "1-199", "200-499", "500-999", "1000-1999",
    "2000-2999", "3000-3999", "4000-4999", "5000-5999", "6000-7999", "8000+",
]

# Wells whose pattern we want to understand
FOCUS_WELLS = ["727a3a10", "0390d174", "b95e7121"]

# Anchor-dominant threshold: anchor RMSE low → anchor is reliable
ANCHOR_DOMINANT_RMSE_THR = 15.0
# LightGBM-kills threshold: LightGBM worsens RMSE by this much
LGBM_KILLS_DIFF_THR = -10.0

BASE_COLS = [
    "well_id", "row_idx", "is_target",
    "hidden_length", "known_length",
    "last_known_TVT", "delta_MD_from_PS",
    "delta_X_from_PS", "delta_Y_from_PS", "delta_Z_from_PS",
    "post_ps_step", "row_frac", "X", "Y", "Z",
]

# ─── helpers ──────────────────────────────────────────────────────────────────

def now_jst() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def metric_row(df: pd.DataFrame, pred_col: str, prefix: str = "") -> dict:
    err = df[pred_col].to_numpy(dtype=float) - df["TVT"].to_numpy(dtype=float)
    abs_err = np.abs(err)
    return {
        f"{prefix}rmse": float(np.sqrt(np.mean(err ** 2))),
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
        rows.append(row)
    return pd.DataFrame(rows)


def load_oof(model_exp: str, anchor_exp: str, train_base_path: Path) -> pd.DataFrame:
    model = pd.read_csv(Path("experiments") / model_exp / "oof.csv")
    anchor = pd.read_csv(
        Path("experiments") / anchor_exp / "oof.csv",
        usecols=TARGET_KEY + ["pred_tvt"],
    ).rename(columns={"pred_tvt": "anchor_pred_tvt"})

    base = pd.read_parquet(train_base_path, columns=BASE_COLS)
    base = base.loc[base["is_target"].astype(bool)].drop(columns=["is_target"]).copy()

    merged = (
        model
        .merge(anchor, on=TARGET_KEY, how="left", validate="one_to_one")
        .merge(base, on=TARGET_KEY, how="left", validate="one_to_one")
    )
    assert not merged["anchor_pred_tvt"].isna().any(), "anchor predictions missing"
    assert not merged["hidden_length"].isna().any(), "base metadata missing"
    return merged


def trajectory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well_id, g in df.sort_values(TARGET_KEY).groupby("well_id", sort=True):
        x, y, z = (g[c].to_numpy(dtype=float) for c in ("X", "Y", "Z"))
        dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
        dxdy = np.sqrt(dx ** 2 + dy ** 2)
        heading = np.unwrap(np.arctan2(dy, dx)) if len(dx) else np.array([0.0])
        hchg = np.abs(np.diff(heading)) if len(heading) > 1 else np.array([0.0])
        rows.append({
            "well_id": well_id,
            "final_delta_z_from_ps": float(g["delta_Z_from_PS"].iloc[-1]),
            "mean_abs_dz_step": float(np.nanmean(np.abs(dz))) if len(dz) else 0.0,
            "std_dz_step": float(np.nanstd(dz)) if len(dz) else 0.0,
            "mean_abs_dxdy_step": float(np.nanmean(dxdy)) if len(dxdy) else 0.0,
            "curvature_proxy": float(np.nanmean(hchg)) if len(hchg) else 0.0,
            "azimuth_range": float(np.nanmax(heading) - np.nanmin(heading)) if len(heading) else 0.0,
        })
    traj = pd.DataFrame(rows)
    z_abs = traj["final_delta_z_from_ps"].abs()
    flat_thr = float(z_abs.quantile(0.33))
    curved_thr = float(traj["curvature_proxy"].quantile(0.67))
    traj["vertical_shape"] = np.select(
        [z_abs.le(flat_thr),
         traj["final_delta_z_from_ps"].gt(flat_thr),
         traj["final_delta_z_from_ps"].lt(-flat_thr)],
        ["flat", "upward", "downward"],
        default="flat",
    )
    traj["curvature_shape"] = np.where(traj["curvature_proxy"].ge(curved_thr), "curved", "smooth")
    traj["trajectory_shape"] = traj["vertical_shape"] + "_" + traj["curvature_shape"]
    return traj


# ─── analysis functions ───────────────────────────────────────────────────────

def build_well_table(df: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    """One row per well with error metrics + trajectory features."""
    well_err = []
    for well_id, g in df.groupby("well_id", sort=True):
        row = {"well_id": well_id}
        row["fold"] = int(g["fold"].iloc[0])
        row["n_rows"] = int(len(g))
        row["hidden_length"] = int(g["hidden_length"].iloc[0])
        row["known_length"] = int(g["known_length"].iloc[0])
        row.update(metric_row(g, "pred_tvt", "model_"))
        row.update(metric_row(g, "anchor_pred_tvt", "anchor_"))
        row["rmse_diff"] = row["anchor_rmse"] - row["model_rmse"]  # positive = LGB better
        well_err.append(row)
    well_df = pd.DataFrame(well_err)
    return well_df.merge(traj, on="well_id", how="left", validate="one_to_one")


def anchor_dominant_analysis(well_df: pd.DataFrame) -> pd.DataFrame:
    """Flag wells where anchor is strong and LightGBM makes things worse."""
    wdf = well_df.copy()
    wdf["anchor_dominant"] = wdf["anchor_rmse"] < ANCHOR_DOMINANT_RMSE_THR
    wdf["lgbm_kills"] = wdf["rmse_diff"] < LGBM_KILLS_DIFF_THR
    wdf["anchor_blown"] = wdf["anchor_dominant"] & wdf["lgbm_kills"]
    return wdf


def focus_well_profile(well_df: pd.DataFrame) -> dict:
    """Numeric profile of the 3 focus wells vs the rest."""
    feat_cols = [
        "hidden_length", "known_length",
        "anchor_rmse", "model_rmse", "rmse_diff",
        "final_delta_z_from_ps", "curvature_proxy",
        "mean_abs_dz_step", "std_dz_step", "mean_abs_dxdy_step",
    ]
    focus = well_df[well_df["well_id"].isin(FOCUS_WELLS)]
    rest = well_df[~well_df["well_id"].isin(FOCUS_WELLS)]
    blown = well_df[well_df.get("anchor_blown", pd.Series(False, index=well_df.index))]

    def stats(subset: pd.DataFrame, label: str) -> dict:
        out: dict = {"group": label, "n": len(subset)}
        for c in feat_cols:
            if c in subset.columns:
                out[f"{c}_mean"] = float(subset[c].mean())
                out[f"{c}_median"] = float(subset[c].median())
        return out

    profile = [stats(focus, "focus_wells"), stats(rest, "rest"), stats(blown, "anchor_blown")]
    if "trajectory_shape" in focus.columns:
        for label, subset in [("focus_wells", focus), ("anchor_blown", blown), ("rest", rest)]:
            shape_dist = subset["trajectory_shape"].value_counts(normalize=True).round(3).to_dict()
            for row in profile:
                if row["group"] == label:
                    row["shape_dist"] = shape_dist
    return {"profiles": profile}


def knn_similar_wells(
    well_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Find k nearest wells in feature space for each focus well."""
    feat_cols = [
        "hidden_length", "known_length",
        "anchor_rmse",
        "final_delta_z_from_ps", "curvature_proxy",
        "mean_abs_dz_step", "std_dz_step",
    ]
    available = [c for c in feat_cols if c in well_df.columns]
    mat = well_df[available].fillna(0).to_numpy(dtype=float)

    # Standardize
    std = mat.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    mat_norm = (mat - mat.mean(axis=0, keepdims=True)) / std

    well_ids = well_df["well_id"].to_numpy()
    rows = []
    for fw in FOCUS_WELLS:
        if fw not in well_ids:
            continue
        idx = int(np.where(well_ids == fw)[0][0])
        dists = np.sqrt(((mat_norm - mat_norm[idx]) ** 2).sum(axis=1))
        order = np.argsort(dists)
        for rank, j in enumerate(order[:k + 1]):
            if well_ids[j] == fw:
                continue
            rows.append({
                "query_well": fw,
                "neighbor_well": well_ids[j],
                "rank": rank,
                "l2_dist": float(dists[j]),
                "anchor_rmse": float(well_df.iloc[j]["anchor_rmse"]),
                "model_rmse": float(well_df.iloc[j]["model_rmse"]),
                "rmse_diff": float(well_df.iloc[j]["rmse_diff"]),
                "hidden_length": int(well_df.iloc[j]["hidden_length"]),
                "trajectory_shape": str(well_df.iloc[j].get("trajectory_shape", "")),
                "anchor_blown": bool(well_df.iloc[j].get("anchor_blown", False)),
            })
    return pd.DataFrame(rows)


def longtail_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Fine-grained hidden_length bins for 2000+ rows."""
    out = df.copy()
    out["longtail_bin"] = pd.cut(
        out["hidden_length"],
        bins=LONGTAIL_BINS,
        labels=LONGTAIL_LABELS,
        include_lowest=True,
        right=True,
    )
    return summarize_group(out, ["longtail_bin"])


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-exp", default=DEFAULT_MODEL_EXP)
    p.add_argument("--anchor-exp", default=DEFAULT_ANCHOR_EXP)
    p.add_argument("--out-exp", default=DEFAULT_OUT_EXP)
    p.add_argument("--train-base", default="data/processed/train_base_v001.parquet")
    p.add_argument("--knn-k", type=int, default=10)
    return p.parse_args()


def df_to_md(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "（データなし）"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows_str = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([header, sep] + rows_str)


def main() -> None:
    args = parse_args()
    out_dir = Path("experiments") / args.out_exp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading OOF data …")
    df = load_oof(args.model_exp, args.anchor_exp, Path(args.train_base))
    print(f"  rows: {len(df):,}  wells: {df['well_id'].nunique()}")

    print("Computing trajectory metrics …")
    traj = trajectory_metrics(df)

    print("Building well-level table …")
    well_df = build_well_table(df, traj)
    well_df = anchor_dominant_analysis(well_df)

    # ── 1. Long-tail finer bins ────────────────────────────────────────────
    print("Long-tail finer bins …")
    df_with_traj = df.merge(traj[["well_id", "trajectory_shape"]], on="well_id", how="left")
    longtail_report = longtail_analysis(df_with_traj)
    longtail_report.to_csv(out_dir / "longtail_finer_report.csv", index=False)

    # ── 2. Anchor-dominant wells ───────────────────────────────────────────
    blown_wells = well_df[well_df["anchor_blown"]].sort_values("rmse_diff")
    blown_wells.to_csv(out_dir / "anchor_blown_wells.csv", index=False)

    anchor_dom_wells = well_df[well_df["anchor_dominant"]].sort_values("rmse_diff")
    anchor_dom_wells.to_csv(out_dir / "anchor_dominant_wells.csv", index=False)

    # ── 3. Focus well profile ─────────────────────────────────────────────
    profile = focus_well_profile(well_df)

    # ── 4. k-NN similar wells ─────────────────────────────────────────────
    print("k-NN similarity …")
    neighbors = knn_similar_wells(well_df, k=args.knn_k)
    neighbors.to_csv(out_dir / "focus_well_neighbors.csv", index=False)

    # ── 5. Summary stats ──────────────────────────────────────────────────
    n_anchor_dominant = int(well_df["anchor_dominant"].sum())
    n_lgbm_kills = int(well_df["lgbm_kills"].sum())
    n_blown = int(well_df["anchor_blown"].sum())
    n_wells_total = int(len(well_df))

    result = {
        "exp_id": args.out_exp,
        "status": "completed",
        "created_at": now_jst(),
        "analysis_target": args.model_exp,
        "baseline": args.anchor_exp,
        "focus_wells": FOCUS_WELLS,
        "thresholds": {
            "anchor_dominant_rmse": ANCHOR_DOMINANT_RMSE_THR,
            "lgbm_kills_diff": LGBM_KILLS_DIFF_THR,
        },
        "summary": {
            "n_wells_total": n_wells_total,
            "n_anchor_dominant": n_anchor_dominant,
            "n_lgbm_kills": n_lgbm_kills,
            "n_anchor_blown": n_blown,
            "pct_anchor_dominant": round(n_anchor_dominant / n_wells_total * 100, 1),
            "pct_lgbm_kills": round(n_lgbm_kills / n_wells_total * 100, 1),
            "pct_anchor_blown": round(n_blown / n_wells_total * 100, 1),
        },
        "focus_well_profiles": profile,
        "longtail_bins": LONGTAIL_LABELS,
        "reports": [
            "longtail_finer_report.csv",
            "anchor_blown_wells.csv",
            "anchor_dominant_wells.csv",
            "focus_well_neighbors.csv",
        ],
        "leak_risk": "low",
    }

    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ── notes ─────────────────────────────────────────────────────────────
    blown_shapes = blown_wells["trajectory_shape"].value_counts().to_dict() if len(blown_wells) else {}
    focus_in_df = well_df[well_df["well_id"].isin(FOCUS_WELLS)]

    notes_lines = [
        f"# {args.out_exp}",
        "",
        "## 目的",
        "",
        "worst wellの共通点を特定し、「anchorが強いのにLightGBMが崩すwellに似ているwellを検出する。",
        "long tail（2000+）をさらに細かく切り、どの長さ帯で崩れるかを確認する。",
        "",
        "## long tail 細分化 (2000+)",
        "",
        df_to_md(longtail_report[longtail_report["longtail_bin"].astype(str).str.contains(r"\d000|\d+000|8000", regex=True)]),
        "",
        "## anchor_blown wells",
        f"- anchor RMSE < {ANCHOR_DOMINANT_RMSE_THR} かつ LightGBM改善 < {LGBM_KILLS_DIFF_THR} : {n_blown} wells",
        f"- trajectory_shape分布: {blown_shapes}",
        "",
        "## focus well profile",
    ]

    if len(focus_in_df):
        for _, row in focus_in_df.iterrows():
            notes_lines.append(
                f"- {row['well_id']}: hidden={row['hidden_length']}, anchor_rmse={row['anchor_rmse']:.1f}, "
                f"model_rmse={row['model_rmse']:.1f}, diff={row['rmse_diff']:.1f}, "
                f"shape={row.get('trajectory_shape','')}, curvature={row.get('curvature_proxy',0):.4f}"
            )

    notes_lines += [
        "",
        "## k-NN 類似well (上位5件 / focus well)",
        "",
        df_to_md(neighbors[neighbors["rank"] <= 5]) if len(neighbors) else "（データなし）",
        "",
        "## 次アクション",
        "",
        "1. anchor_blown wellsの共通特徴（hidden_length 帯、trajectory_shape）からルールベース guard を設計する。",
        "2. LightGBM delta を anchor_rmse_estimate で shrinkage する exp006 を実装する。",
        "3. k-NN 類似wellをtest setに展開し、同じ guard が必要なwellを事前フラグする。",
        "",
        "## リーク懸念",
        "",
        "exp003 OOF・exp001 anchor OOF・train base metadataのみ使用。モデル学習に未使用のためリーク懸念低。",
        "ただしanchor_rmse を特徴量に使う場合はtestでは推定値（e.g., anchor on holdout近似）が必要。",
    ]

    (out_dir / "notes.md").write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
