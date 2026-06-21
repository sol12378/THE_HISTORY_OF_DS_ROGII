#!/usr/bin/env python3
"""exp033: 最終ブレンド — 利用可能な全componentでNNLS最適化.

components (利用可能なら自動取り込み):
- exp022 PF (own tw, raw GR) → "pf_orig"
- exp025_pf_tuned (tuned PF) → "pf_tuned"
- exp030b multi-tw PF → "pf_multi"
- exp031 multi-tw + physical-lik → "pf_phys"
- exp014 geom (LightGBM Group F) → "geom"
- exp018 tree blend (LGB+XGB+CB) → "trees"
- exp019 NN → "nn"
- exp020 attention → "attn"
- exp032 TabICL → "tabicl"

NNLS over delta-space (each component → delta from anchor), nested fold CV.
最終出力 = NNLS重みでblend + 平滑(w=101)。leak-free維持。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_TRAIN = ROOT / "data" / "processed" / "train_base_v001.parquet"
FOLDS_CSV = ROOT / "data" / "folds" / "folds_group_well_v001.csv"
OUT_DIR = ROOT / "experiments" / "exp033_final_blend"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENTS = {
    "pf_orig":   ("experiments/exp022_particle_filter/oof.csv", "pred_tvt"),
    "pf_tuned":  ("experiments/exp025_pf_tuned/oof.csv", "pred_tvt"),
    "pf_multi":  ("experiments/exp030b_multi_tw_vec/oof.csv", "pred_tvt"),
    "pf_phys":   ("experiments/exp031_pf_physical_lik/oof.csv", "pred_tvt"),
    "geom":      ("experiments/exp014_geom_extrap/oof.csv", "pred_tvt"),
    "trees":     ("experiments/exp018_model_blend/oof.csv", "pred_tvt"),
    "nn":        ("experiments/exp019_seq_nn/oof.csv", "pred_tvt"),
    "attn":      ("experiments/exp020_typewell_attn/oof.csv", "pred_tvt"),
    "tabicl":    ("experiments/exp032b_tabicl_fast/oof.csv", "pred_tvt"),
}

TEST_COMPONENTS = {
    "pf_orig":   ("experiments/exp022_particle_filter/submission.csv", "tvt"),
    "pf_tuned":  ("experiments/exp025_pf_tuned/submission.csv", "tvt"),
    "pf_multi":  ("experiments/exp030b_multi_tw_vec/submission.csv", "tvt"),
    "pf_phys":   ("experiments/exp031_pf_physical_lik/submission.csv", "tvt"),
    "geom":      ("experiments/exp014_geom_extrap/submission.csv", "tvt"),
    "trees":     ("experiments/exp018_model_blend/submission.csv", "tvt"),
    "nn":        ("experiments/exp019_seq_nn/submission.csv", "tvt"),
    "attn":      ("experiments/exp020_typewell_attn/submission.csv", "tvt"),
    "tabicl":    ("experiments/exp032b_tabicl_fast/submission.csv", "tvt"),
}


def load_oof_components():
    """Load all available components on training-side OOF."""
    loaded = {}
    for name, (path, col) in COMPONENTS.items():
        p = ROOT / path
        if p.exists():
            df = pd.read_csv(p)
            # tabicl uses different column name potentially
            if col not in df.columns:
                # try common alternatives
                for alt in ["tabicl_pred_tvt", "pred_tvt", "tvt"]:
                    if alt in df.columns:
                        col = alt
                        break
                else:
                    print(f"  WARN: {name} no prediction column in {path}")
                    continue
            # canonical key: well_id, row_idx
            df = df[["well_id", "row_idx", col]].rename(columns={col: name})
            loaded[name] = df
            print(f"  loaded {name} ({len(df)} rows)")
        else:
            print(f"  skip {name} (not found at {path})")
    return loaded


def main():
    print("== exp033_final_blend ==")
    print("\n[1] Loading components OOF...")
    oofs = load_oof_components()
    if not oofs:
        print("ERROR: no OOFs found.")
        return

    # Base data (anchor + target)
    base = pd.read_parquet(PROCESSED_TRAIN, columns=[
        "well_id", "row_idx", "TVT", "TVT_input", "is_target", "last_known_TVT"])
    base = base[base["is_target"].astype(bool)].copy()
    base = base[["well_id", "row_idx", "TVT", "last_known_TVT"]]
    folds = pd.read_csv(FOLDS_CSV).drop_duplicates(subset=["well_id"])[["well_id", "fold"]]
    base = base.merge(folds, on="well_id", how="left").dropna(subset=["fold"])
    base["fold"] = base["fold"].astype(int)

    # Merge all OOFs
    df = base.copy()
    for name, df_o in oofs.items():
        df = df.merge(df_o, on=["well_id", "row_idx"], how="left")

    # Drop rows where any component is NaN
    drop_cols = list(oofs.keys())
    before = len(df)
    df = df.dropna(subset=drop_cols).copy()
    after = len(df)
    print(f"  Dropped {before-after}/{before} rows due to missing components")
    print(f"  Components in blend: {drop_cols}")

    # === Convert to delta-space (each component - last_known_TVT) ===
    for c in drop_cols:
        df[f"{c}_d"] = df[c] - df["last_known_TVT"]
    df["y_d"] = df["TVT"] - df["last_known_TVT"]

    # === Component individual CV (sanity) ===
    print("\n[2] Per-component CV:")
    for c in drop_cols:
        rmse = float(np.sqrt(np.mean((df[c] - df["TVT"]) ** 2)))
        print(f"  {c:10s}: {rmse:.4f}")

    # === Correlation matrix in delta-space ===
    delta_cols = [f"{c}_d" for c in drop_cols]
    corr_delta_err = pd.DataFrame(
        {c: df[f"{c}_d"] - df["y_d"] for c in drop_cols}
    ).corr()
    print("\n[3] Component error correlation matrix:")
    print(corr_delta_err.round(3))

    # === Nested 5-fold NNLS blend ===
    print("\n[4] Nested 5-fold NNLS blend (leak-free)")
    y = df["y_d"].to_numpy(float)
    X = df[delta_cols].to_numpy(float)
    fold_arr = df["fold"].to_numpy(int)

    nested_pred_d = np.zeros(len(df), dtype=float)
    fold_weights = []
    for fold in sorted(np.unique(fold_arr)):
        va = fold_arr == fold
        tr = ~va
        coef, _ = nnls(X[tr], y[tr])
        nested_pred_d[va] = X[va] @ coef
        fold_weights.append(coef)
        # diagnostic
        wt_str = ", ".join(f"{c}={w:.3f}" for c, w in zip(drop_cols, coef))
        print(f"  fold{fold}: weights = {{{wt_str}}}")
    # average weights for final reporting
    avg_w = np.mean(fold_weights, axis=0)
    print(f"\n  Average weights:")
    for c, w in zip(drop_cols, avg_w):
        print(f"    {c}: {w:.4f}")

    # Pooled nested NNLS CV
    pred_tvt_blend = nested_pred_d + df["last_known_TVT"].to_numpy(float)
    true_tvt = df["TVT"].to_numpy(float)
    cv_blend = float(np.sqrt(np.mean((pred_tvt_blend - true_tvt) ** 2)))
    print(f"\n  NNLS nested CV (TVT) = {cv_blend:.6f}")

    # === Smoothing per-well (well内 row順 mean w=101) ===
    print("\n[5] Smoothing within well (w=101)...")
    df["pred_blend"] = pred_tvt_blend
    smoothed = []
    for wid, g in df.sort_values(["well_id", "row_idx"]).groupby("well_id", sort=False):
        s = g["pred_blend"].rolling(101, center=True, min_periods=1).mean()
        smoothed.append(pd.DataFrame({"well_id": g["well_id"], "row_idx": g["row_idx"], "pred_smooth": s.values}))
    sm = pd.concat(smoothed, ignore_index=True)
    df = df.merge(sm, on=["well_id", "row_idx"], how="left")
    cv_smooth = float(np.sqrt(np.mean((df["pred_smooth"] - df["TVT"]) ** 2)))
    print(f"  Smoothed CV = {cv_smooth:.6f}")

    df.to_csv(OUT_DIR / "oof.csv", index=False)

    summary = {
        "exp_id": "exp033_final_blend",
        "components": drop_cols,
        "individual_cv": {
            c: float(np.sqrt(np.mean((df[c] - df["TVT"]) ** 2))) for c in drop_cols
        },
        "blend_nested_cv": cv_blend,
        "blend_smoothed_cv": cv_smooth,
        "avg_weights": dict(zip(drop_cols, avg_w.tolist())),
        "fold_weights": [
            dict(zip(drop_cols, w.tolist())) for w in fold_weights
        ],
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    # === Generate test submission using same weights ===
    print("\n[6] Generating test submission...")
    # Load test components
    test_dfs = {}
    for name, (path, col) in TEST_COMPONENTS.items():
        p = ROOT / path
        if not p.exists() or name not in drop_cols:
            continue
        d = pd.read_csv(p)
        if col not in d.columns:
            for alt in ["pred_tvt", "tvt", "tabicl_pred_tvt"]:
                if alt in d.columns:
                    col = alt; break
        d = d[["id", col]].rename(columns={col: name})
        test_dfs[name] = d
        print(f"  loaded test {name}")

    if test_dfs:
        sample = pd.read_csv("data/raw/sample_submission.csv")[["id"]]
        # need anchor (last_known_TVT) for delta blend → load test_base
        te = pd.read_parquet("data/processed/test_base_v001.parquet",
                              columns=["id", "is_target", "last_known_TVT"])
        te = te[te["is_target"].astype(bool)][["id", "last_known_TVT"]]
        sub = sample.merge(te, on="id", how="left")
        for name, dd in test_dfs.items():
            sub = sub.merge(dd, on="id", how="left")
        # Components that have test predictions
        test_cols = [c for c in drop_cols if c in test_dfs]
        print(f"  test components: {test_cols}")
        # Renormalize weights over available test components
        w_test = np.array([avg_w[drop_cols.index(c)] for c in test_cols])
        w_test = w_test / w_test.sum() if w_test.sum() > 0 else w_test
        # Convert each to delta, blend, add back anchor
        for c in test_cols:
            sub[f"{c}_d"] = sub[c] - sub["last_known_TVT"]
        delta_sum = np.zeros(len(sub))
        for c, w in zip(test_cols, w_test):
            delta_sum += w * sub[f"{c}_d"].to_numpy(float)
        sub["tvt_blend"] = delta_sum + sub["last_known_TVT"]
        # Smooth per well (using id parsing: id = "wellid_rowidx")
        sub["well_id"] = sub["id"].str.rsplit("_", n=1).str[0]
        sub["row_idx_str"] = sub["id"].str.rsplit("_", n=1).str[-1].astype(int)
        sub = sub.sort_values(["well_id", "row_idx_str"]).reset_index(drop=True)
        smoothed_t = []
        for wid, g in sub.groupby("well_id", sort=False):
            s = g["tvt_blend"].rolling(101, center=True, min_periods=1).mean()
            smoothed_t.append(pd.DataFrame({"id": g["id"], "tvt": s.values}))
        out_sub = pd.concat(smoothed_t, ignore_index=True)
        # Order according to sample
        out_sub = sample.merge(out_sub, on="id", how="left")
        out_sub.to_csv(OUT_DIR / "submission.csv", index=False)
        print(f"  Saved test submission ({len(out_sub)} rows, NaN={out_sub['tvt'].isna().sum()})")

    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
