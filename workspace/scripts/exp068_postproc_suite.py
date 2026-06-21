#!/usr/bin/env python3
"""exp068: 後処理/blend系 20案のうち main直轄分を leak-free nested-fold で一括測定.

カバー案: 2 projection, 3 ramp, 5 blend重み, 6 proj degree, 10 ridge/nnls,
11 spline近似, 12 dip除去座標, 13 robust scale適応, 14 robust variants,
16 surface融合, 17 proj順序, 18 residual融合, 20 ablation.

全て leak-free: projection等は予測値・Z・anchor(known)のみ使用、真TVTは評価のみ。
blend重みは nested GroupKFold。基準: artifact+exp026 nested 9.27, PF 11.02。
"""
from __future__ import annotations
import numpy as np, pandas as pd, joblib, json
from pathlib import Path
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/exp068_postproc_suite"; OUT.mkdir(parents=True, exist_ok=True)
ART = "/tmp/artifacts/thbdh5765_rogii-v11-fresh-artifacts/models"


def robfit(s, y, deg=5, iters=4):
    s = np.asarray(s, float); y = np.asarray(y, float)
    if len(s) < deg + 2 or np.std(s) < 1e-9:
        return y.copy()
    try:
        c = np.polyfit(s, y, deg)
        for _ in range(iters):
            r = y - np.polyval(c, s)
            sc = np.median(np.abs(r)) * 1.4826 + 1e-6
            w = 1.0 / (1.0 + (r / (2.0 * sc)) ** 2)
            c = np.polyfit(s, y, deg, w=w)
        return np.polyval(c, s)
    except Exception:
        return y.copy()


def main():
    base = pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id", "row_idx", "id", "MD", "Z", "TVT", "TVT_input", "is_target",
                 "is_known_tvt", "last_known_TVT", "last_known_MD", "last_known_Z"])
    tgt = base[base["is_target"]].sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    fold = tgt.merge(pd.read_csv("data/folds/folds_group_well_v001.csv")[["well_id", "fold"]],
                     on="well_id", how="left")["fold"].to_numpy()
    y = tgt["TVT"].to_numpy(float); a = tgt["last_known_TVT"].to_numpy(float)
    Z = tgt["Z"].to_numpy(float); MD = tgt["MD"].to_numpy(float)
    lkMD = tgt["last_known_MD"].to_numpy(float); lkZ = tgt["last_known_Z"].to_numpy(float)
    wid = tgt["well_id"].to_numpy()

    def load(p):
        d = pd.read_csv(p)[["well_id", "row_idx", "pred_tvt"]]
        return tgt[["well_id", "row_idx"]].merge(d, on=["well_id", "row_idx"], how="left")["pred_tvt"].to_numpy(float)

    art = np.mean([joblib.load(f"{ART}/{m}/oof_preds.pkl") for m in
                   ["lightgbm-1", "lightgbm-2", "lightgbm-3", "catboost-1", "catboost-2", "catboost-3"]], 0)
    art_tvt = art + a
    p022 = load("experiments/exp022_particle_filter/oof.csv")
    p014 = load("experiments/exp014_geom_extrap/oof.csv")
    exp026 = a + 0.683 * (p022 - a) + 0.392 * (p014 - a)
    surf = load("experiments/exp067_field_lowo_blend/oof.csv")
    resid = load("experiments/exp041_pf_residual_gbdt/oof.csv")

    def rmse(p): return float(np.sqrt(np.nanmean((p - y) ** 2)))

    def nested(cols):
        X = np.nan_to_num(np.column_stack([c - a for c in cols])); oof = np.zeros(len(y))
        for f in sorted(set(fold)):
            tr = fold != f; va = fold == f
            w, _ = nnls(X[tr], (y - a)[tr]); oof[va] = X[va] @ w + a[va]
        return rmse(oof), oof

    # --- per-well projection on a prediction (leak-free) ---
    order = np.lexsort((tgt["row_idx"].to_numpy(), wid))
    # group row indices per well in sorted order
    well_groups = {}
    for i in order:
        well_groups.setdefault(wid[i], []).append(i)
    well_groups = {w: np.array(ix) for w, ix in well_groups.items()}

    def project(pred, deg=5, coord="U"):
        out = pred.copy()
        for w, ix in well_groups.items():
            mdv = MD[ix]; denom = max(mdv.max() - lkMD[ix][0], 1e-6)
            s = (mdv - lkMD[ix][0]) / denom
            anchor = a[ix][0] + lkZ[ix][0]
            if coord == "U":      # U = pred + Z - anchor (1次dip除去)
                u = pred[ix] + Z[ix] - anchor
                fit = robfit(s, u, deg)
                out[ix] = fit + anchor - Z[ix]
            else:                  # raw TVT projection
                out[ix] = robfit(s, pred[ix], deg)
        return out

    def ramp(pred, tau=85):
        md_since = np.maximum(MD - lkMD, 0.0)
        f = 1.0 - np.exp(-md_since / tau)
        return a + (pred - a) * f

    results = {}
    # baseline
    cv_base, oof_base = nested([art_tvt, exp026])
    results["baseline_artifact+exp026 (nested)"] = round(cv_base, 4)

    # 案2/6/11/12 projection (degree sweep, U vs raw)
    for deg in [3, 5, 7]:
        pj = project(oof_base, deg=deg, coord="U")
        results[f"proj_U_deg{deg}"] = round(rmse(pj), 4)
    for deg in [3, 5]:
        pj = project(oof_base, deg=deg, coord="raw")
        results[f"proj_raw_deg{deg}"] = round(rmse(pj), 4)

    # 案3 ramp
    results["ramp_tau85"] = round(rmse(ramp(oof_base, 85)), 4)
    results["ramp_tau300"] = round(rmse(ramp(oof_base, 300)), 4)

    # 案17 projection順序: 各component個別projection後にblend
    art_pj = project(art_tvt, 5, "U"); e26_pj = project(exp026, 5, "U")
    cv_pjfirst, _ = nested([art_pj, e26_pj])
    results["proj_each_then_blend"] = round(cv_pjfirst, 4)

    # 案16 surface融合, 案18 residual融合
    cv_surf, _ = nested([art_tvt, exp026, surf]); results["+surface"] = round(cv_surf, 4)
    cv_res, _ = nested([art_tvt, exp026, resid]); results["+residual_gbdt"] = round(cv_res, 4)
    cv_all, oof_all = nested([art_tvt, exp026, surf, resid, p022])
    results["+surf+resid+pf (all)"] = round(cv_all, 4)
    results["all + proj_U_deg5"] = round(rmse(project(oof_all, 5, "U")), 4)

    # 案5 blend重み: artifact vs exp026 のweight sweep
    bw = []
    for w_art in np.linspace(0, 1, 11):
        pr = w_art * art_tvt + (1 - w_art) * exp026
        bw.append((round(w_art, 2), round(rmse(pr), 4)))
    results["blend_weight_sweep_artifact"] = bw

    # 案20 ablation: 各成分単体 nested
    for nm, c in [("artifact", art_tvt), ("exp026", exp026), ("pf022", p022),
                  ("geom014", p014), ("surface", surf), ("residual", resid)]:
        results[f"solo_{nm}"] = round(rmse(c), 4)

    (OUT / "result.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
