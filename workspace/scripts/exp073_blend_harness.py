#!/usr/bin/env python
"""exp073e: nested NNLS blend + postprocess ablation harness.

Inputs (produced by exp073 workers):
  - experiments/exp073_public_assets_integration/oof_geom.csv        (id, well_id, row_idx, tvt_true, pred_tvt)
  - experiments/exp073_public_assets_integration/oof_pf.csv          (same)
  - experiments/exp073_public_assets_integration/external_oof.parquet (id, well_id, target_delta, last_known_tvt, pred_delta columns)
  - experiments/exp073_public_assets_integration/oof_engineB.parquet  (id, well, pred_delta columns, target_delta)
  - data/processed/train_base_v001.parquet (geometry for postprocess)

Method:
  1. Align all components on id (hidden rows of 773 train wells).
  2. Delta space: pred_delta = pred_tvt - last_known_TVT for our components.
  3. Nested NNLS: GroupKFold(5) by well; fit NNLS weights on outer-train, predict outer-test.
  4. Postprocess ablation (one change at a time, fold-consistency gate >=4/5 folds):
     warmup damping (tau) -> smoothing (savgol/mean) -> robust U-projection (deg, beta).
  5. Report nested CV, per-fold RMSE, weight stability. Compare vs current best 9.086.

Leak notes:
  - All component OOFs are out-of-fold (GroupKFold by well) or per-well independent (PF).
  - Blend weights are fit nested (outer-train only) -> honest CV.
  - Postprocess params chosen on pooled OOF with fold-consistency gate; mild selection
    optimism acknowledged (same protocol as exp068/exp042).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
OUT = ROOT / "experiments" / "exp073_public_assets_integration"
TRAIN_BASE = ROOT / "data" / "processed" / "train_base_v001.parquet"
CURRENT_BEST = 9.086

N_FOLDS = 5


def pooled_rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def load_components():
    """Return df keyed by id with target_delta, per-component pred_delta, geometry."""
    base = pd.read_parquet(
        TRAIN_BASE,
        columns=[
            "well_id", "row_idx", "MD", "Z", "TVT", "TVT_input", "is_known_tvt",
            "last_known_TVT", "last_known_MD", "last_known_Z",
        ],
    )
    hidden = base[~base["is_known_tvt"]].copy()
    hidden["id"] = hidden["well_id"] + "_" + hidden["row_idx"].astype(str)
    hidden["target_delta"] = hidden["TVT"] - hidden["last_known_TVT"]
    df = hidden[[
        "id", "well_id", "row_idx", "MD", "Z", "target_delta",
        "last_known_TVT", "last_known_MD", "last_known_Z",
    ]].set_index("id")
    # well end MD for projection s-coordinate
    end_md = hidden.groupby("well_id")["MD"].max().rename("end_MD")
    df = df.join(end_md, on="well_id")

    comps = {}

    def add_ours(path, name):
        if not Path(path).exists():
            print(f"[skip] {name}: {path} not found")
            return
        o = pd.read_csv(path, usecols=["id", "pred_tvt"]).set_index("id")
        comps[name] = o["pred_tvt"]

    add_ours(OUT / "oof_geom.csv", "geom")
    add_ours(OUT / "oof_pf.csv", "pf")
    add_ours(OUT / "oof_tcn.csv", "tcn")
    add_ours(OUT / "oof_tcn_resid.csv", "tcn_resid")

    ext_path = OUT / "external_oof.parquet"
    if ext_path.exists():
        ext = pd.read_parquet(ext_path)
        ext = ext.set_index("id")
        pred_cols = [c for c in ext.columns if c.startswith(("pilk_", "rav_", "v11_"))]
        for c in pred_cols:
            comps[c] = ext[c]  # already delta
    else:
        print(f"[skip] external_oof.parquet not found")

    engb_path = OUT / "oof_engineB.parquet"
    if engb_path.exists():
        engb = pd.read_parquet(engb_path)
        idcol = "id" if "id" in engb.columns else engb.columns[0]
        engb = engb.set_index(idcol)
        for c in [c for c in engb.columns if c.startswith("engB_")]:
            comps[c] = engb[c]  # delta
    else:
        print(f"[skip] oof_engineB.parquet not found")

    # attach, converting our TVT-space preds to delta
    for name, s in comps.items():
        df[name] = s
        if name in ("geom", "pf", "tcn", "tcn_resid"):
            df[name] = df[name] - df["last_known_TVT"]

    comp_names = list(comps.keys())
    n0 = len(df)
    cov = {c: float(df[c].notna().mean()) for c in comp_names}
    df = df.dropna(subset=comp_names + ["target_delta"])
    print(f"rows: {n0} -> {len(df)} after align; coverage={cov}")
    return df, comp_names


def nested_nnls(df, comp_names):
    X = df[comp_names].to_numpy(dtype=np.float64)
    y = df["target_delta"].to_numpy(dtype=np.float64)
    groups = df["well_id"].to_numpy()
    pred = np.zeros_like(y)
    fold_id = np.zeros(len(y), dtype=int)
    weights = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        w, _ = nnls(X[tr], y[tr])
        pred[te] = X[te] @ w
        fold_id[te] = k
        weights.append(w)
    weights = np.array(weights)
    return pred, fold_id, weights


def per_fold_rmse(y, p, fold_id):
    return [pooled_rmse(y[fold_id == k], p[fold_id == k]) for k in range(N_FOLDS)]


# ---------------- postprocess ----------------

def apply_warmup(df, pred_delta, tau):
    md_since = (df["MD"] - df["last_known_MD"]).to_numpy()
    ramp = 1.0 - np.exp(-np.clip(md_since, 0, None) / tau)
    return pred_delta * ramp


def apply_smooth(df, pred_tvt, kind, window):
    out = pred_tvt.copy()
    order = np.argsort(df["_row_order"].to_numpy(), kind="stable")
    # group by well preserving row order
    tmp = pd.DataFrame({
        "well": df["well_id"].to_numpy(),
        "row": df["row_idx"].to_numpy(),
        "p": pred_tvt,
    })
    tmp["_pos"] = np.arange(len(tmp))
    tmp = tmp.sort_values(["well", "row"], kind="stable")
    res = np.empty(len(tmp))
    i = 0
    for well, g in tmp.groupby("well", sort=False):
        v = g["p"].to_numpy()
        n = len(v)
        if kind == "savgol":
            w = min(window, n if n % 2 == 1 else n - 1)
            if w >= 5:
                sm = savgol_filter(v, w, 3)
            else:
                sm = v
        else:  # mean
            w = min(window, n)
            sm = pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()
        res[i:i + n] = sm
        i += n
    tmp["sm"] = res
    tmp = tmp.sort_values("_pos", kind="stable")
    return tmp["sm"].to_numpy()


def robust_projection(df, pred_tvt, deg, beta):
    """U = pred + Z - anchor, robust polyfit vs normalized MD, replace beta*fit."""
    anchor = (df["last_known_TVT"] + df["last_known_Z"]).to_numpy()
    Z = df["Z"].to_numpy()
    md = df["MD"].to_numpy()
    lkmd = df["last_known_MD"].to_numpy()
    emd = df["end_MD"].to_numpy()
    out = pred_tvt.copy()
    tmp = pd.DataFrame({
        "well": df["well_id"].to_numpy(),
        "row": df["row_idx"].to_numpy(),
        "U": pred_tvt + Z - anchor,
        "s": np.where(emd > lkmd, (md - lkmd) / np.maximum(emd - lkmd, 1e-9), 0.0),
        "Z": Z, "anchor": anchor,
    })
    tmp["_pos"] = np.arange(len(tmp))
    res = np.empty(len(tmp))
    tmp = tmp.sort_values(["well", "row"], kind="stable")
    i = 0
    for well, g in tmp.groupby("well", sort=False):
        U = g["U"].to_numpy()
        s = g["s"].to_numpy()
        n = len(U)
        if n < max(deg + 2, 8):
            res[i:i + n] = U
            i += n
            continue
        w = np.ones(n)
        fit = U
        try:
            for _ in range(4):
                cs = np.polyfit(s, U, deg, w=w)
                fit = np.polyval(cs, s)
                r = U - fit
                sc = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
                w = 1.0 / (1.0 + (r / (2 * sc)) ** 2)
            res[i:i + n] = beta * fit + (1 - beta) * U
        except Exception:
            res[i:i + n] = U
        i += n
    tmp["proj"] = res
    tmp = tmp.sort_values("_pos", kind="stable")
    return tmp["proj"].to_numpy() + (-Z + anchor)  # back to TVT: fit + anchor - Z


def evaluate(df, y, fold_id, pred_tvt, label, report):
    truth_tvt = y + df["last_known_TVT"].to_numpy()
    rmse = pooled_rmse(truth_tvt, pred_tvt)
    folds = per_fold_rmse(truth_tvt, pred_tvt, fold_id)
    report.append({"step": label, "pooled": rmse, "folds": folds})
    print(f"  {label}: pooled={rmse:.4f} folds={[round(f,3) for f in folds]}")
    return rmse, folds


def consistent_gain(base_folds, new_folds, min_better=4):
    return sum(n < b for n, b in zip(new_folds, base_folds)) >= min_better


def main():
    t0 = time.time()
    df, comp_names = load_components()
    df["_row_order"] = np.arange(len(df))
    y = df["target_delta"].to_numpy(dtype=np.float64)

    print(f"\ncomponents ({len(comp_names)}): {comp_names}")
    report = []
    # individual component RMSEs
    indiv = {}
    for c in comp_names:
        indiv[c] = pooled_rmse(y, df[c].to_numpy())
    print("individual delta RMSE:", {k: round(v, 3) for k, v in sorted(indiv.items(), key=lambda kv: kv[1])})

    # ---- nested NNLS ----
    pred_delta, fold_id, weights = nested_nnls(df, comp_names)
    wmean = weights.mean(axis=0)
    wstd = weights.std(axis=0)
    print("\nNNLS weights (mean +/- std across folds):")
    for c, m, s in zip(comp_names, wmean, wstd):
        if m > 1e-4 or s > 1e-4:
            print(f"  {c}: {m:.3f} +/- {s:.3f}")

    pred_tvt = pred_delta + df["last_known_TVT"].to_numpy()
    base_rmse, base_folds = evaluate(df, y, fold_id, pred_tvt, "nnls_blend_raw", report)

    # ---- pruned NNLS (drop ~zero-weight comps) for parsimony check ----
    keep = [c for c, m in zip(comp_names, wmean) if m > 0.01]
    if len(keep) < len(comp_names):
        pred_delta_p, fold_id_p, weights_p = nested_nnls(df, keep)
        pred_tvt_p = pred_delta_p + df["last_known_TVT"].to_numpy()
        p_rmse, p_folds = evaluate(df, y, fold_id_p, pred_tvt_p, f"nnls_pruned({len(keep)})", report)
        if p_rmse <= base_rmse + 0.005:
            comp_names_used, pred_delta, fold_id, weights = keep, pred_delta_p, fold_id_p, weights_p
            pred_tvt, base_rmse, base_folds = pred_tvt_p, p_rmse, p_folds
        else:
            comp_names_used = comp_names
    else:
        comp_names_used = comp_names

    best_tvt = pred_tvt
    best_rmse, best_folds = base_rmse, base_folds
    recipe = {"components": comp_names_used,
              "weights_mean": {c: float(m) for c, m in zip(comp_names_used, weights.mean(axis=0))}}

    # ---- warmup ----
    print("\n[warmup tau ablation]")
    chosen_tau = None
    for tau in [50, 85, 120]:
        d2 = apply_warmup(df, best_tvt - df["last_known_TVT"].to_numpy(), tau)
        t2 = d2 + df["last_known_TVT"].to_numpy()
        r, f = evaluate(df, y, fold_id, t2, f"warmup_tau{tau}", report)
        if r < best_rmse and consistent_gain(best_folds, f):
            best_rmse, best_folds, best_tvt, chosen_tau = r, f, t2, tau
    recipe["warmup_tau"] = chosen_tau

    # ---- smoothing ----
    print("\n[smoothing ablation]")
    chosen_smooth = None
    for kind, win in [("savgol", 17), ("savgol", 61), ("mean", 101)]:
        t2 = apply_smooth(df, best_tvt, kind, win)
        r, f = evaluate(df, y, fold_id, t2, f"smooth_{kind}{win}", report)
        if r < best_rmse and consistent_gain(best_folds, f):
            best_rmse, best_folds, best_tvt, chosen_smooth = r, f, t2, (kind, win)
    recipe["smooth"] = chosen_smooth

    # ---- projection ----
    print("\n[projection ablation]")
    chosen_proj = None
    for deg in [4, 5]:
        for beta in [0.75, 1.0]:
            t2 = robust_projection(df, best_tvt, deg, beta)
            r, f = evaluate(df, y, fold_id, t2, f"proj_d{deg}_b{beta}", report)
            if r < best_rmse and consistent_gain(best_folds, f):
                best_rmse, best_folds, best_tvt, chosen_proj = r, f, t2, (deg, beta)
    recipe["projection"] = chosen_proj

    elapsed = (time.time() - t0) / 60
    result = {
        "nested_cv_raw_blend": base_rmse,
        "nested_cv_final": best_rmse,
        "final_folds": best_folds,
        "recipe": recipe,
        "individual_rmse": indiv,
        "weights_per_fold": {c: weights[:, i].tolist() for i, c in enumerate(comp_names_used)},
        "current_best_reference": CURRENT_BEST,
        "beats_current_best": best_rmse < CURRENT_BEST,
        "report": report,
        "n_rows": int(len(df)),
        "runtime_min": elapsed,
    }
    with open(OUT / "result_blend.json", "w") as fh:
        json.dump(result, fh, indent=2)
    # save final OOF
    out_oof = pd.DataFrame({
        "id": df.index, "well_id": df["well_id"].to_numpy(),
        "tvt_true": y + df["last_known_TVT"].to_numpy(), "pred_tvt": best_tvt,
    })
    out_oof.to_csv(OUT / "oof_blend_final.csv", index=False)
    print(f"\nFINAL nested CV: {best_rmse:.4f} (raw blend {base_rmse:.4f}, current best {CURRENT_BEST})")
    print(f"recipe: {recipe}")
    print(f"runtime: {elapsed:.1f} min")


if __name__ == "__main__":
    main()
