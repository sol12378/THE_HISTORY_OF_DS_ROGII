#!/usr/bin/env python3
"""exp024: Multi-stage blend — PF × geom × trees/NN (NNLS, nested-fold honest CV).

PF(exp022, CV11.02)は幾何系(geom/trees/NN, CV13.3-13.5)と誤差相関0.43と低く、
ブレンドで大きく伸びる(暫定 PF×geom=10.16)。全 leak-free base learner を
delta スケールで NNLS ブレンドし、nested 5-fold で正直な CV を出す。

base learners (全て leak-free OOF):
  pf    = exp022 Particle Filter
  geom  = exp014 Group F 幾何外挿
  trees = exp018 LGBM+XGB+Cat blend
  nn    = exp019 系列TCN blend
  attn  = exp020 typewell cross-attention blend
weights: NNLS on (pred_i - anchor) vs (TVT - anchor), 非負。
honest CV: 各foldでweightを他4foldで学習しhold-out予測(nested)。
+ well内 mean平滑(w=101)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp024_multistage_blend"
OUT_DIR = Path("experiments") / EXP_ID
KEY = ["well_id", "row_idx"]

OOF = {
    "pf":    ("experiments/exp022_particle_filter/oof.csv", "pred_tvt"),
    "geom":  ("experiments/exp014_geom_extrap/oof.csv", "pred_tvt"),
    "trees": ("experiments/exp018_model_blend/oof.csv", "pred_tvt"),
    "nn":    ("experiments/exp019_seq_nn/oof.csv", "pred_tvt"),
    "attn":  ("experiments/exp020_typewell_attn/oof.csv", "pred_tvt"),
}
SUB = {
    "pf":    "experiments/exp022_particle_filter/submission.csv",
    "geom":  "experiments/exp014_geom_extrap/submission.csv",
    "trees": "experiments/exp018_model_blend/submission.csv",
    "nn":    "experiments/exp019_seq_nn/submission.csv",
    "attn":  "experiments/exp020_typewell_attn/submission.csv",
}
SMOOTH_W = 101


def load_merged():
    base = pd.read_csv(OOF["pf"][0])[KEY + ["TVT", "last_known_TVT", "pred_tvt"]].rename(columns={"pred_tvt": "pf"})
    # fold from exp014
    f14 = pd.read_csv(OOF["geom"][0])[KEY + ["fold", "pred_tvt"]].rename(columns={"pred_tvt": "geom"})
    m = base.merge(f14, on=KEY, how="inner")
    for name in ["trees", "nn", "attn"]:
        d = pd.read_csv(OOF[name][0])[KEY + ["pred_tvt"]].rename(columns={"pred_tvt": name})
        m = m.merge(d, on=KEY, how="inner")
    return m


def fit_nnls(X_delta, y_delta):
    w, _ = nnls(X_delta, y_delta)
    return w


def blend_pred(df, cols, w, anchor_col="last_known_TVT"):
    a = df[anchor_col].to_numpy(float)
    delta = np.column_stack([df[c].to_numpy(float) - a for c in cols])
    return a + delta @ w


def smooth(df, col):
    out = df[[*KEY, col]].copy().sort_values(KEY)
    out["_o"] = np.arange(len(out))
    res = np.empty(len(out))
    for _, idx in out.groupby("well_id", sort=False).groups.items():
        s = out.loc[idx].sort_values("row_idx")
        res[s["_o"].to_numpy()] = s[col].rolling(SMOOTH_W, min_periods=1, center=True).mean().to_numpy()
    out[col + "_sm"] = res
    return out.sort_values("_o")[col + "_sm"].to_numpy()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] multi-stage NNLS blend (PF × geom × trees/NN)")
    m = load_merged()
    cols = ["pf", "geom", "trees", "nn", "attn"]
    print(f"  merged rows: {len(m)}  base learners: {cols}")

    a = m["last_known_TVT"].to_numpy(float)
    y = m["TVT"].to_numpy(float) - a

    # individual CVs
    print("  individual CV:")
    for c in cols + ["last_known_TVT"]:
        print(f"    {c:6s} = {tvt_rmse(m['TVT'], m[c] if c!='last_known_TVT' else a):.4f}")

    # error correlation matrix
    errs = {c: (m[c].to_numpy(float) - m["TVT"].to_numpy(float)) for c in cols}
    corr = pd.DataFrame(errs).corr()
    print("  error correlation:\n" + corr.round(2).to_string())

    # ---- nested 5-fold NNLS (honest) ----
    folds = sorted(m["fold"].dropna().unique())
    oof_blend = np.zeros(len(m))
    fold_w = {}
    for f in folds:
        tr = m["fold"] != f
        te = m["fold"] == f
        Xtr = np.column_stack([m.loc[tr, c].to_numpy(float) - a[tr.to_numpy()] for c in cols])
        w = fit_nnls(Xtr, y[tr.to_numpy()])
        fold_w[int(f)] = w
        Xte = np.column_stack([m.loc[te, c].to_numpy(float) - a[te.to_numpy()] for c in cols])
        oof_blend[te.to_numpy()] = a[te.to_numpy()] + Xte @ w
    cv_nested = tvt_rmse(m["TVT"], oof_blend)
    print(f"\n  nested-fold NNLS blend CV = {cv_nested:.6f}")
    for f, w in fold_w.items():
        print(f"    fold{f} w=" + ", ".join(f"{c}:{wi:.3f}" for c, wi in zip(cols, w)))

    # full-data weights (for test + reporting)
    Xall = np.column_stack([m[c].to_numpy(float) - a for c in cols])
    w_full = fit_nnls(Xall, y)
    print("  full-data weights: " + ", ".join(f"{c}:{wi:.3f}" for c, wi in zip(cols, w_full)))
    m["blend"] = a + Xall @ w_full
    cv_full = tvt_rmse(m["TVT"], m["blend"])
    print(f"  full-data blend CV (optimistic) = {cv_full:.6f}")

    # smoothing on nested blend
    m["oof_blend"] = oof_blend
    m["blend_sm"] = smooth(m.assign(_b=oof_blend).rename(columns={"_b": "oof_blend2"}), "oof_blend")
    cv_sm = tvt_rmse(m["TVT"], m["blend_sm"])
    print(f"  nested blend + smooth(w={SMOOTH_W}) CV = {cv_sm:.6f}")

    # fold-wise of nested+smooth
    print("  fold-wise (nested+smooth):")
    for f in folds:
        s = m[m["fold"] == f]
        print(f"    fold{int(f)}: {tvt_rmse(s['TVT'], s['blend_sm']):.4f}")

    # save oof
    out = m[KEY + ["fold", "TVT", "last_known_TVT", "oof_blend", "blend_sm"] + cols].copy()
    out.to_csv(OUT_DIR / "oof.csv", index=False)

    # ---- test submission with full-data weights ----
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].copy()
    sub["well_id"] = sub["id"].str.rsplit("_", n=1).str[0]
    sub["row_idx"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
    # anchor for test from any model submission via test_base
    tb = pd.read_parquet("data/processed/test_base_v001.parquet",
                         columns=["well_id", "row_idx", "is_target", "last_known_TVT"])
    tb = tb[tb["is_target"].astype(bool)]
    sub = sub.merge(tb[["well_id", "row_idx", "last_known_TVT"]], on=["well_id", "row_idx"], how="left")
    for name in cols:
        s = pd.read_csv(SUB[name]).rename(columns={"tvt": name})
        sub = sub.merge(s[["id", name]], on="id", how="left")
    at = sub["last_known_TVT"].to_numpy(float)
    Xt = np.column_stack([sub[c].to_numpy(float) - at for c in cols])
    sub["tvt"] = at + Xt @ w_full
    # smooth test
    sub = sub.sort_values(["well_id", "row_idx"])
    sub["tvt"] = sub.groupby("well_id", sort=False)["tvt"].transform(
        lambda x: x.rolling(SMOOTH_W, min_periods=1, center=True).mean())
    out_sub = sample[["id"]].merge(sub[["id", "tvt"]], on="id", how="left", validate="one_to_one")
    assert not out_sub["tvt"].isna().any()
    out_sub.to_csv(OUT_DIR / "submission.csv", index=False)

    result = {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "method": "NNLS multi-stage blend (delta scale) of pf/geom/trees/nn/attn + smoothing",
        "base_learners": cols,
        "individual_cv": {c: float(tvt_rmse(m["TVT"], m[c])) for c in cols},
        "anchor_cv": float(tvt_rmse(m["TVT"], a)),
        "error_corr": corr.round(3).to_dict(),
        "nested_fold_cv": cv_nested,
        "full_data_cv_optimistic": cv_full,
        "nested_plus_smooth_cv": cv_sm,
        "full_weights": {c: float(wi) for c, wi in zip(cols, w_full)},
        "fold_weights": {str(f): {c: float(wi) for c, wi in zip(cols, w)} for f, w in fold_w.items()},
        "prev_best_blend": 13.320964, "pf_alone": 11.024014,
        "leak_risk": "none (all leak-free OOF; weights nested-fold validated)",
    }
    write_json(OUT_DIR / "result.json", result)
    print(f"\n[{EXP_ID}] 完了 -> {OUT_DIR}  nested+smooth CV={cv_sm:.6f}")


if __name__ == "__main__":
    main()
