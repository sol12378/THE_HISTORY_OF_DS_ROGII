"""exp103: honest-CV NNLS blend (103a) + seed_std reliability gating (103b).

Lightweight: pandas/numpy/sklearn NNLS only. Uses existing OOF only (no PF re-run).
Leak-free: nested-fold NNLS weights decided out-of-fold; evaluated on both
well-fold and typewell-fold groupings. seed_std (from exp100, prediction-derived) used for gating.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(r"D:\Data_Science\autoresearch\competitions\rogii-wellbore-geology-prediction\workspace")
EXPDIR = ROOT / "experiments"
OUTDIR = EXPDIR / "exp103_oof_blend_gating"
OUTDIR.mkdir(parents=True, exist_ok=True)

# model -> (path, pred_col)
MODELS = {
    "exp022": ("exp022_particle_filter/oof.csv", "pred_tvt"),
    "exp008": ("exp008_gr_rolling/oof.csv", "pred_tvt"),
    "exp100": ("exp100_soft_likpf/oof.csv", "pred_tvt"),
    "exp099": ("exp099_marker_tiepoint/oof.csv", "pred"),
    "anchor": ("exp120_anchor_repro/oof.csv", "pred_tvt"),
}
MODEL_ORDER = ["exp022", "exp100", "exp008", "exp099", "anchor"]


def load():
    base = None
    for name in MODEL_ORDER:
        rel, col = MODELS[name]
        usecols = ["well_id", "row_idx", "TVT", col]
        if name == "exp100":
            usecols = usecols + ["seed_std"]
        df = pd.read_csv(EXPDIR / rel, usecols=usecols)
        df = df.rename(columns={col: f"pred_{name}"})
        if name == "exp100":
            df = df.rename(columns={"seed_std": "seed_std"})
        # keep TVT only from first
        if base is None:
            base = df
        else:
            keep = ["well_id", "row_idx", f"pred_{name}"]
            if name == "exp100":
                keep = keep + ["seed_std"]
            base = base.merge(df[keep], on=["well_id", "row_idx"], how="inner")
    return base


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def pooled_rmse_singles(df):
    out = {}
    y = df["TVT"].values
    for name in MODEL_ORDER:
        out[name] = rmse(df[f"pred_{name}"].values, y)
    return out


def corr_matrix(df):
    y = df["TVT"].values
    errs = {n: df[f"pred_{n}"].values - y for n in MODEL_ORDER}
    M = np.corrcoef(np.vstack([errs[n] for n in MODEL_ORDER]))
    return {MODEL_ORDER[i]: {MODEL_ORDER[j]: float(M[i, j]) for j in range(len(MODEL_ORDER))}
            for i in range(len(MODEL_ORDER))}


def nested_nnls_blend(df, fold_col):
    """Out-of-fold NNLS: for each fold, fit NNLS on other folds, predict held fold."""
    X = np.vstack([df[f"pred_{n}"].values for n in MODEL_ORDER]).T  # (N, M)
    y = df["TVT"].values
    folds = df[fold_col].values
    uf = sorted(np.unique(folds))
    pred = np.full(len(df), np.nan)
    weight_acc = np.zeros(len(MODEL_ORDER))
    for f in uf:
        tr = folds != f
        te = folds == f
        w, _ = nnls(X[tr], y[tr])
        pred[te] = X[te] @ w
        weight_acc += w
    avg_w = weight_acc / len(uf)
    cv = rmse(pred, y)
    return cv, {MODEL_ORDER[i]: float(avg_w[i]) for i in range(len(MODEL_ORDER))}, pred


def gating_sweep(df, fold_col, base_blend_pred):
    """seed_std reliability gating: where seed_std high (PF uncertain), shift toward anchor.
    Leak-free: seed_std is prediction-derived. Threshold decided per-fold out-of-fold
    by minimizing OOF RMSE on training folds.
    Gate: blend = (1-g)*base_blend + g*anchor, where g=1 if seed_std>thr else 0.
    """
    y = df["TVT"].values
    ss = df["seed_std"].values
    anchor = df["pred_anchor"].values
    folds = df[fold_col].values
    uf = sorted(np.unique(folds))
    # candidate thresholds from quantiles
    qs = np.quantile(ss[~np.isnan(ss)], np.linspace(0.5, 0.99, 25))
    cand = np.unique(np.round(qs, 6))

    def apply_gate(thr):
        g = (ss > thr).astype(float)
        return (1 - g) * base_blend_pred + g * anchor

    # fixed-threshold best (oracle-on-pooled, reported separately) vs nested
    # Nested: choose thr on train folds, apply to test fold (leak-free)
    pred_nested = np.full(len(df), np.nan)
    chosen_thrs = []
    for f in uf:
        tr = folds != f
        te = folds == f
        best_thr, best_c = None, np.inf
        for thr in cand:
            g = (ss[tr] > thr).astype(float)
            blended_tr = (1 - g) * base_blend_pred[tr] + g * anchor[tr]
            c = rmse(blended_tr, y[tr])
            if c < best_c:
                best_c, best_thr = c, thr
        chosen_thrs.append(float(best_thr))
        g_te = (ss[te] > best_thr).astype(float)
        pred_nested[te] = (1 - g_te) * base_blend_pred[te] + g_te * anchor[te]
    cv_nested = rmse(pred_nested, y)
    return cv_nested, chosen_thrs, cand.tolist()


def main():
    df = load()
    # valid rows: TVT not NaN and all preds finite
    pred_cols = [f"pred_{n}" for n in MODEL_ORDER]
    mask = df["TVT"].notna()
    for c in pred_cols:
        mask &= df[c].notna() & np.isfinite(df[c])
    df = df[mask].reset_index(drop=True)

    # join folds
    wf = pd.read_csv(ROOT / "data/folds/folds_group_well_v001.csv")[["well_id", "fold"]].drop_duplicates("well_id")
    wf = wf.rename(columns={"fold": "fold_well"})
    tf = pd.read_csv(ROOT / "data/folds/folds_group_typewell_v001.csv")[["well_id", "fold"]].drop_duplicates("well_id")
    tf = tf.rename(columns={"fold": "fold_tw"})
    df = df.merge(wf, on="well_id", how="inner").merge(tf, on="well_id", how="inner")

    n_rows = len(df)

    singles = pooled_rmse_singles(df)
    # represent singles per both fold groupings (single RMSE is pooled, same value, but report both keys)
    singles_out = {n: {"cv_wellfold": singles[n], "cv_typewellfold": singles[n]} for n in MODEL_ORDER}

    corr = corr_matrix(df)

    cv_w, w_w, pred_w = nested_nnls_blend(df, "fold_well")
    cv_tw, w_tw, pred_tw = nested_nnls_blend(df, "fold_tw")

    # gating uses typewell-fold (honest) base blend
    gate_cv, gate_thrs, gate_cand = gating_sweep(df, "fold_tw", pred_tw)
    improvement = cv_tw - gate_cv  # positive = gating helps

    # also well-fold gating for completeness
    gate_cv_w, gate_thrs_w, _ = gating_sweep(df, "fold_well", pred_w)

    result = {
        "n_rows_common": int(n_rows),
        "singles": singles_out,
        "corr_matrix": corr,
        "blend": {
            "model_order": MODEL_ORDER,
            "weights_wellfold": w_w,
            "weights_typewellfold": w_tw,
            "cv_wellfold": cv_w,
            "cv_typewellfold": cv_tw,
        },
        "gating": {
            "honest_typewellfold": {
                "best_cv": gate_cv,
                "improvement_vs_nogate": improvement,
                "chosen_thresholds_per_fold": gate_thrs,
            },
            "wellfold": {
                "best_cv": gate_cv_w,
                "improvement_vs_nogate": cv_w - gate_cv_w,
                "chosen_thresholds_per_fold": gate_thrs_w,
            },
            "threshold_candidates": gate_cand,
        },
        "notes_short": (
            "Honest=typewell-fold. NNLS weights & gating thresholds decided out-of-fold (leak-free). "
            "seed_std from exp100 prediction-derived."
        ),
    }
    with open(OUTDIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # notes.md
    best_single = min(singles, key=singles.get)
    lines = []
    lines.append("# exp103 OOF blend + seed_std gating\n")
    lines.append(f"共通行数 (n_rows_common): {n_rows:,}\n")
    lines.append("## 最良 honest CV (typewell-fold) / well-fold")
    lines.append(f"- blend cv_typewellfold (honest): **{cv_tw:.4f}**")
    lines.append(f"- blend cv_wellfold: {cv_w:.4f}")
    lines.append(f"- 最良単体: {best_single} = {singles[best_single]:.4f}")
    lines.append("")
    lines.append("## 各単体 pooled CV")
    for n in MODEL_ORDER:
        lines.append(f"- {n}: {singles[n]:.4f}")
    lines.append("")
    lines.append("## NNLS 重み (out-of-fold 平均)")
    lines.append("typewell-fold (honest):")
    for n in MODEL_ORDER:
        lines.append(f"- {n}: {w_tw[n]:.4f}")
    lines.append("well-fold:")
    for n in MODEL_ORDER:
        lines.append(f"- {n}: {w_w[n]:.4f}")
    lines.append("")
    lines.append("## 誤差相関 (corr matrix)")
    hdr = "       " + " ".join(f"{n:>8}" for n in MODEL_ORDER)
    lines.append("```")
    lines.append(hdr)
    for ni in MODEL_ORDER:
        row = f"{ni:>7}" + " ".join(f"{corr[ni][nj]:8.3f}" for nj in MODEL_ORDER)
        lines.append(row)
    lines.append("```")
    lines.append("")
    lines.append("## exp099/exp100 寄与")
    lines.append(f"- exp099 weight (typewell): {w_tw['exp099']:.4f}")
    lines.append(f"- exp100 weight (typewell): {w_tw['exp100']:.4f}")
    lines.append("")
    lines.append("## seed_std gating 効果 (leak-free, nested threshold)")
    lines.append(f"- honest(typewell): gate CV {gate_cv:.4f} vs no-gate {cv_tw:.4f} -> improvement {improvement:+.4f}")
    lines.append(f"- well-fold: gate CV {gate_cv_w:.4f} vs no-gate {cv_w:.4f} -> improvement {cv_w - gate_cv_w:+.4f}")
    lines.append("")
    lines.append("## 結論")
    lines.append(f"- 7.297 base 比: honest stack CV={cv_tw:.4f}")
    with open(OUTDIR / "notes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
