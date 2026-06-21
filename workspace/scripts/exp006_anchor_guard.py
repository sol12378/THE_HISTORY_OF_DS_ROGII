#!/usr/bin/env python3
"""exp006: Rule-based anchor guard.

After analysis in exp005, the following patterns predict LightGBM will destroy
an already-strong anchor:
  - hidden_length > 8000              → anchor always wins
  - final_delta_z_from_ps < -100 AND hidden_length > 4000  → strong guard
  - final_delta_z_from_ps < -50  AND hidden_length > 4000  → mild guard

This script blends exp003 (LightGBM) and exp001 (anchor) predictions using
well-level guard rules, all computed from features available in the test set.
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

from rogii.training.baselines import tvt_rmse, write_json, now_jst

# ─── constants ────────────────────────────────────────────────────────────────

EXP_ID = "exp006_anchor_guard"
DEFAULT_MODEL_EXP = "exp003_lgb_anchor_trajectory"
DEFAULT_ANCHOR_EXP = "exp001_anchor_baseline"

# Guard rule thresholds
FULL_ANCHOR_HIDDEN_THR = 8000       # wells longer than this → anchor only
DELTA_EXTREME_THR = 30.0            # per-well mean pred_delta > this → anchor only
STRONG_GUARD_DELTA_Z_THR = -100.0   # final_delta_z below this → strong guard
STRONG_GUARD_HIDDEN_THR = 4000
MILD_GUARD_DELTA_Z_THR = -50.0      # between -100 and -50 → mild guard
MILD_GUARD_HIDDEN_THR = 4000

# Alpha search grids (LightGBM weight; 1-alpha is anchor weight)
ALPHA_FULL_ANCHOR = 0.0
ALPHA_DELTA_EXTREME = 0.0
ALPHA_GRID_STRONG = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ALPHA_GRID_MILD   = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
ALPHA_NO_GUARD = 1.0


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_oof(model_exp: str, anchor_exp: str) -> pd.DataFrame:
    model = pd.read_csv(Path("experiments") / model_exp / "oof.csv",
                        usecols=["well_id", "row_idx", "fold", "TVT",
                                 "last_known_TVT", "pred_delta", "pred_tvt"])
    model = model.rename(columns={"pred_tvt": "lgbm_pred_tvt"})
    anchor = pd.read_csv(Path("experiments") / anchor_exp / "oof.csv",
                         usecols=["well_id", "row_idx", "pred_tvt"])
    anchor = anchor.rename(columns={"pred_tvt": "anchor_pred_tvt"})
    df = model.merge(anchor, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    assert not df["anchor_pred_tvt"].isna().any(), "anchor OOF missing"
    return df


def compute_well_mean_pred_delta_train(oof: pd.DataFrame) -> pd.DataFrame:
    """Per-well mean predicted delta from train OOF (pred_delta = lgbm - anchor)."""
    return oof.groupby("well_id", as_index=False).agg(mean_pred_delta=("pred_delta", "mean"))


def compute_well_mean_pred_delta_test(
    model_sub_path: Path,
    test_base_path: Path,
) -> pd.DataFrame:
    """Per-well mean predicted delta for test set.

    pred_delta = submission_tvt - last_known_TVT (constant per well).
    The 'id' column in test_base is used to join with submission.
    """
    sub = pd.read_csv(model_sub_path)
    base = pd.read_parquet(test_base_path,
                           columns=["well_id", "id", "is_target", "last_known_TVT"])
    base = base[base["is_target"].astype(bool)].copy()
    merged = sub.merge(base[["id", "well_id", "last_known_TVT"]], on="id", how="left")
    if merged["well_id"].isna().any():
        # Fallback: parse well_id from id string if join failed
        merged["well_id"] = merged["id"].str.rsplit("_", n=1).str[0]
        merged = merged.merge(base[["well_id", "last_known_TVT"]].drop_duplicates("well_id"),
                              on="well_id", how="left", suffixes=("", "_base"))
        last_known = "last_known_TVT_base" if "last_known_TVT_base" in merged.columns else "last_known_TVT"
        merged["pred_delta"] = merged["tvt"] - merged[last_known]
    else:
        merged["pred_delta"] = merged["tvt"] - merged["last_known_TVT"]
    return merged.groupby("well_id", as_index=False).agg(mean_pred_delta=("pred_delta", "mean"))


def compute_well_guard_features(parquet_path: Path) -> pd.DataFrame:
    """Compute per-well guard features from base table (train or test)."""
    cols = ["well_id", "row_idx", "is_target", "hidden_length", "delta_Z_from_PS"]
    df = pd.read_parquet(parquet_path, columns=cols)
    target = df.loc[df["is_target"].astype(bool)].copy()
    # Sort by row_idx to get the true last row per well
    well_feats = (
        target.sort_values(["well_id", "row_idx"])
        .groupby("well_id", as_index=False)
        .agg(
            hidden_length=("hidden_length", "first"),
            final_delta_z_from_ps=("delta_Z_from_PS", "last"),
        )
    )
    return well_feats


def assign_guard_rule(well_feats: pd.DataFrame) -> pd.DataFrame:
    """Assign guard rule per well. Priority: full_anchor > delta_extreme > strong_guard > mild_guard > no_guard."""
    wf = well_feats.copy()
    wf["guard_rule"] = "no_guard"

    mild_mask = (
        wf["final_delta_z_from_ps"].lt(MILD_GUARD_DELTA_Z_THR) &
        wf["hidden_length"].gt(MILD_GUARD_HIDDEN_THR)
    )
    strong_mask = (
        wf["final_delta_z_from_ps"].lt(STRONG_GUARD_DELTA_Z_THR) &
        wf["hidden_length"].gt(STRONG_GUARD_HIDDEN_THR)
    )
    delta_extreme_mask = (
        wf["mean_pred_delta"].gt(DELTA_EXTREME_THR)
        if "mean_pred_delta" in wf.columns
        else pd.Series(False, index=wf.index)
    )
    full_mask = wf["hidden_length"].gt(FULL_ANCHOR_HIDDEN_THR)

    wf.loc[mild_mask, "guard_rule"] = "mild_guard"
    wf.loc[strong_mask, "guard_rule"] = "strong_guard"
    wf.loc[delta_extreme_mask, "guard_rule"] = "delta_extreme"
    wf.loc[full_mask, "guard_rule"] = "full_anchor"
    return wf


def grid_search_alpha(oof: pd.DataFrame, well_rules: pd.DataFrame,
                      rule: str, alpha_grid: list[float]) -> float:
    """Find alpha that minimises RMSE for wells in a given rule group."""
    wells_in_rule = well_rules.loc[well_rules["guard_rule"].eq(rule), "well_id"]
    subset = oof[oof["well_id"].isin(wells_in_rule)]
    if len(subset) == 0:
        return alpha_grid[0]
    best_alpha, best_rmse = alpha_grid[0], float("inf")
    for alpha in alpha_grid:
        blended = alpha * subset["lgbm_pred_tvt"] + (1 - alpha) * subset["anchor_pred_tvt"]
        r = tvt_rmse(subset["TVT"], blended)
        if r < best_rmse:
            best_rmse, best_alpha = r, alpha
    return best_alpha


def blend_predictions(oof: pd.DataFrame, well_rules: pd.DataFrame,
                      alphas: dict[str, float]) -> pd.DataFrame:
    """Apply per-well alpha blending to the prediction column."""
    rule_map = well_rules.set_index("well_id")["guard_rule"].to_dict()
    alpha_map = {
        w: alphas[rule_map[w]] for w in oof["well_id"]
    }
    alpha_series = oof["well_id"].map(alpha_map).to_numpy(dtype=float)
    blended = alpha_series * oof["lgbm_pred_tvt"].to_numpy(dtype=float) + \
              (1 - alpha_series) * oof["anchor_pred_tvt"].to_numpy(dtype=float)
    out = oof.copy()
    out["guard_rule"] = oof["well_id"].map(rule_map)
    out["alpha"] = alpha_series
    out["pred_tvt"] = blended
    out["error"] = blended - oof["TVT"].to_numpy(dtype=float)
    out["abs_error"] = out["error"].abs()
    return out


def build_blended_submission(
    model_sub_path: Path,
    anchor_sub_path: Path,
    test_well_rules: pd.DataFrame,
    alphas: dict[str, float],
    test_base_path: Path,
    sample_sub_path: Path,
) -> pd.DataFrame:
    """Blend test submissions using the same guard rules."""
    model_sub = pd.read_csv(model_sub_path).rename(columns={"tvt": "lgbm_tvt"})
    anchor_sub = pd.read_csv(anchor_sub_path).rename(columns={"tvt": "anchor_tvt"})
    sub = model_sub.merge(anchor_sub, on="id", how="left", validate="one_to_one")

    # Parse well_id from id (format: {well_id}_{row_idx})
    sub["well_id"] = sub["id"].str.rsplit("_", n=1).str[0]

    rule_map = test_well_rules.set_index("well_id")["guard_rule"].to_dict()
    alpha_map = {w: alphas.get(rule_map.get(w, "no_guard"), ALPHA_NO_GUARD)
                 for w in sub["well_id"].unique()}
    sub["alpha"] = sub["well_id"].map(alpha_map)
    sub["tvt"] = sub["alpha"] * sub["lgbm_tvt"] + (1 - sub["alpha"]) * sub["anchor_tvt"]

    sample_sub = pd.read_csv(sample_sub_path)
    result = sample_sub[["id"]].merge(sub[["id", "tvt"]], on="id",
                                      how="left", validate="one_to_one")
    assert not result["tvt"].isna().any(), "missing tvt in blended submission"
    return result


def print_per_rule_rmse(oof: pd.DataFrame, well_rules: pd.DataFrame,
                        alphas: dict[str, float]) -> None:
    for rule in ["full_anchor", "delta_extreme", "strong_guard", "mild_guard", "no_guard"]:
        wells = well_rules.loc[well_rules["guard_rule"].eq(rule), "well_id"]
        subset = oof[oof["well_id"].isin(wells)]
        if len(subset) == 0:
            continue
        n_w = wells.shape[0]
        alpha = alphas[rule]
        blended = alpha * subset["lgbm_pred_tvt"] + (1 - alpha) * subset["anchor_pred_tvt"]
        r_blend = tvt_rmse(subset["TVT"], blended)
        r_lgb = tvt_rmse(subset["TVT"], subset["lgbm_pred_tvt"])
        r_anc = tvt_rmse(subset["TVT"], subset["anchor_pred_tvt"])
        print(f"  {rule:15s} ({n_w:3d} wells) alpha={alpha:.2f}  "
              f"blended={r_blend:.4f}  lgbm={r_lgb:.4f}  anchor={r_anc:.4f}")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-exp", default=DEFAULT_MODEL_EXP)
    p.add_argument("--anchor-exp", default=DEFAULT_ANCHOR_EXP)
    p.add_argument("--out-exp", default=EXP_ID)
    p.add_argument("--train-base", default="data/processed/train_base_v001.parquet")
    p.add_argument("--test-base", default="data/processed/test_base_v001.parquet")
    p.add_argument("--sample-sub", default="data/raw/sample_submission.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path("experiments") / args.out_exp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load OOF ───────────────────────────────────────────────────────────
    print("Loading OOF …")
    oof = load_oof(args.model_exp, args.anchor_exp)
    baseline_rmse = tvt_rmse(oof["TVT"], oof["lgbm_pred_tvt"])
    anchor_overall_rmse = tvt_rmse(oof["TVT"], oof["anchor_pred_tvt"])
    print(f"  LightGBM overall RMSE:  {baseline_rmse:.6f}")
    print(f"  Anchor overall RMSE:    {anchor_overall_rmse:.6f}")

    # ── 2. Compute guard features ─────────────────────────────────────────────
    print("Computing guard features (train) …")
    train_well_feats = compute_well_guard_features(Path(args.train_base))
    train_pred_delta = compute_well_mean_pred_delta_train(oof)
    train_well_feats = train_well_feats.merge(train_pred_delta, on="well_id", how="left")
    train_well_rules = assign_guard_rule(train_well_feats)

    print("Guard rule distribution (train):")
    print(train_well_rules["guard_rule"].value_counts().to_string())

    # ── 3. Grid search alphas ─────────────────────────────────────────────────
    print("\nGrid searching alpha values …")
    alpha_strong = grid_search_alpha(oof, train_well_rules, "strong_guard", ALPHA_GRID_STRONG)
    alpha_mild = grid_search_alpha(oof, train_well_rules, "mild_guard", ALPHA_GRID_MILD)

    alphas = {
        "full_anchor": ALPHA_FULL_ANCHOR,
        "delta_extreme": ALPHA_DELTA_EXTREME,
        "strong_guard": alpha_strong,
        "mild_guard": alpha_mild,
        "no_guard": ALPHA_NO_GUARD,
    }
    print(f"  full_anchor   alpha = {ALPHA_FULL_ANCHOR}")
    print(f"  delta_extreme alpha = {ALPHA_DELTA_EXTREME}")
    print(f"  strong_guard  alpha = {alpha_strong}")
    print(f"  mild_guard    alpha = {alpha_mild}")
    print(f"  no_guard      alpha = {ALPHA_NO_GUARD}")

    # ── 4. Apply blending to OOF ──────────────────────────────────────────────
    print("\nApplying guard blending …")
    oof_blended = blend_predictions(oof, train_well_rules, alphas)
    blended_rmse = tvt_rmse(oof_blended["TVT"], oof_blended["pred_tvt"])
    improvement = baseline_rmse - blended_rmse
    print(f"  Blended CV RMSE: {blended_rmse:.6f}  (improvement: {improvement:+.6f})")

    print("\nPer-rule breakdown:")
    print_per_rule_rmse(oof, train_well_rules, alphas)

    oof_blended.to_csv(out_dir / "oof.csv", index=False)
    train_well_rules.to_csv(out_dir / "well_guard_flags.csv", index=False)

    # ── 5. Apply to test ──────────────────────────────────────────────────────
    print("\nComputing guard features (test) …")
    model_sub_path = Path("experiments") / args.model_exp / "submission.csv"
    test_well_feats = compute_well_guard_features(Path(args.test_base))
    test_pred_delta = compute_well_mean_pred_delta_test(model_sub_path, Path(args.test_base))
    test_well_feats = test_well_feats.merge(test_pred_delta, on="well_id", how="left")
    test_well_rules = assign_guard_rule(test_well_feats)
    print("Guard rule distribution (test):")
    print(test_well_rules["guard_rule"].value_counts().to_string())

    anchor_sub_path = Path("experiments") / args.anchor_exp / "submission.csv"
    blended_sub = build_blended_submission(
        model_sub_path, anchor_sub_path,
        test_well_rules, alphas,
        Path(args.test_base), Path(args.sample_sub),
    )
    blended_sub.to_csv(out_dir / "submission.csv", index=False)
    print(f"  submission rows: {len(blended_sub)}")

    # ── 6. Guard params ───────────────────────────────────────────────────────
    guard_params = {
        "thresholds": {
            "full_anchor_hidden_thr": FULL_ANCHOR_HIDDEN_THR,
            "delta_extreme_thr": DELTA_EXTREME_THR,
            "strong_guard_delta_z_thr": STRONG_GUARD_DELTA_Z_THR,
            "strong_guard_hidden_thr": STRONG_GUARD_HIDDEN_THR,
            "mild_guard_delta_z_thr": MILD_GUARD_DELTA_Z_THR,
            "mild_guard_hidden_thr": MILD_GUARD_HIDDEN_THR,
        },
        "alphas": alphas,
        "alpha_grids": {
            "strong_guard": ALPHA_GRID_STRONG,
            "mild_guard": ALPHA_GRID_MILD,
        },
        "train_rule_counts": train_well_rules["guard_rule"].value_counts().to_dict(),
        "test_rule_counts": test_well_rules["guard_rule"].value_counts().to_dict(),
    }
    write_json(out_dir / "guard_params.json", guard_params)

    # ── 7. result.json ────────────────────────────────────────────────────────
    result = {
        "exp_id": args.out_exp,
        "status": "completed",
        "created_at": now_jst(),
        "model_exp": args.model_exp,
        "anchor_exp": args.anchor_exp,
        "metric": "TVT_abs_RMSE",
        "lgbm_cv_rmse": baseline_rmse,
        "anchor_cv_rmse": anchor_overall_rmse,
        "blended_cv_rmse": blended_rmse,
        "rmse_improvement_vs_lgbm": improvement,
        "rmse_improvement_vs_anchor": anchor_overall_rmse - blended_rmse,
        "alphas": alphas,
        "n_oof_rows": int(len(oof_blended)),
        "n_submission_rows": int(len(blended_sub)),
        "leak_risk": "low",
        "notes": (
            "Rule-based guard using hidden_length and final_delta_z_from_ps. "
            "Both features are available in test set without leakage. "
            "Alpha values tuned on OOF."
        ),
    }
    write_json(out_dir / "result.json", result)

    # ── 8. notes.md ──────────────────────────────────────────────────────────
    notes = f"""# {args.out_exp}

## 目的

exp003 LightGBM の後処理として、ルールベースの anchor guard を適用し、
「anchor が強いのに LightGBM が崩すwells」での大崩れを防ぐ。

## ガードルール

| rule | 条件 | alpha (LGB weight) |
|---|---|---|
| full_anchor | hidden_length > {FULL_ANCHOR_HIDDEN_THR} | {ALPHA_FULL_ANCHOR} |
| delta_extreme | mean_pred_delta > {DELTA_EXTREME_THR} | {ALPHA_DELTA_EXTREME} |
| strong_guard | final_delta_z < {STRONG_GUARD_DELTA_Z_THR} AND hidden_length > {STRONG_GUARD_HIDDEN_THR} | {alpha_strong} |
| mild_guard | final_delta_z < {MILD_GUARD_DELTA_Z_THR} AND hidden_length > {MILD_GUARD_HIDDEN_THR} (strong_guard非対象) | {alpha_mild} |
| no_guard | otherwise | {ALPHA_NO_GUARD} |

`pred = alpha * lgbm + (1 - alpha) * anchor`

## 結果

| metric | 値 |
|---|---|
| LightGBM CV RMSE | {baseline_rmse:.6f} |
| Anchor CV RMSE | {anchor_overall_rmse:.6f} |
| Blended CV RMSE | {blended_rmse:.6f} |
| improvement vs LightGBM | {improvement:+.6f} |
| improvement vs Anchor | {anchor_overall_rmse - blended_rmse:+.6f} |

## train guard rule 分布

{train_well_rules['guard_rule'].value_counts().to_string()}

## test guard rule 分布

{test_well_rules['guard_rule'].value_counts().to_string()}

## リーク懸念

guard 条件に使う `hidden_length` と `delta_Z_from_PS` は base table に存在し、
train/test 両方で同じ定義で計算できる。
alpha は train OOF 上でチューニングしているが、候補値が粗いため過学習懸念は低い。
"""
    (out_dir / "notes.md").write_text(notes, encoding="utf-8")

    print(f"\n完了: {out_dir}")
    print(f"LightGBM CV:  {baseline_rmse:.6f}")
    print(f"Blended CV:   {blended_rmse:.6f}  ({improvement:+.6f})")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
