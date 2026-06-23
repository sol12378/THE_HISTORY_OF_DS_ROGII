#!/usr/bin/env python3
"""exp104: exp068/072流 projection後処理を exp103 honest blend に適用し honest CV を検証.

設計:
- exp103と同じ 3-model nested NNLS blend (exp022 + exp100 + exp008, leak-safe fold外重み)
  を well-fold / typewell-fold 両方で再構成 -> per-row blended pred。
- その blend pred に exp068/072流の per-well projection 後処理 (coord="U": U=pred+Z-anchor の
  robust 多項式 IRLS fit -> 平滑/外挿補正) を適用。実装は exp068_postproc_suite.py の
  robfit / project を忠実移植。
- degree (次数) と beta (robust scale 係数) を fold外で選択 (nested) し過学習を確認。
- well/typewell 両 fold で pooled honest CV を算出。projection 前後を比較。

leak規約: NNLS重み・projection係数は fold外で決定。projection は予測+既知幾何
(Z, MD, anchor=last_known_TVT 由来) のみ使用 = leak-free。真TVT は評価のみ。
PF/モデル再計算なし、既存OOFのみ。pandas/numpy/scipy(nnls)のみ。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(r"D:\Data_Science\autoresearch\competitions\rogii-wellbore-geology-prediction\workspace")
EXPDIR = ROOT / "experiments"
OUTDIR = EXPDIR / "exp104_projection_blend"
OUTDIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "exp022": ("exp022_particle_filter/oof.csv", "pred_tvt"),
    "exp100": ("exp100_soft_likpf/oof.csv", "pred_tvt"),
    "exp008": ("exp008_gr_rolling/oof.csv", "pred_tvt"),
}
MODEL_ORDER = ["exp022", "exp100", "exp008"]


def rmse(pred, y):
    return float(np.sqrt(np.nanmean((pred - y) ** 2)))


# --- exp068 robfit 忠実移植 (beta=2.0 を可変化) ---
def robfit(s, y, deg=5, iters=4, beta=2.0):
    s = np.asarray(s, float); y = np.asarray(y, float)
    if len(s) < deg + 2 or np.std(s) < 1e-9:
        return y.copy()
    try:
        c = np.polyfit(s, y, deg)
        for _ in range(iters):
            r = y - np.polyval(c, s)
            sc = np.median(np.abs(r)) * 1.4826 + 1e-6
            w = 1.0 / (1.0 + (r / (beta * sc)) ** 2)
            c = np.polyfit(s, y, deg, w=w)
        return np.polyval(c, s)
    except Exception:
        return y.copy()


def build_well_groups(wid, row_idx):
    order = np.lexsort((row_idx, wid))
    groups = {}
    for i in order:
        groups.setdefault(wid[i], []).append(i)
    return {w: np.array(ix) for w, ix in groups.items()}


def project(pred, well_groups, MD, Z, a, lkMD, lkZ, deg=5, beta=2.0):
    """exp068 project(coord='U') 忠実移植。U = pred + Z - anchor の robust poly fit。"""
    out = pred.copy()
    for w, ix in well_groups.items():
        mdv = MD[ix]
        denom = max(mdv.max() - lkMD[ix][0], 1e-6)
        s = (mdv - lkMD[ix][0]) / denom
        anchor = a[ix][0] + lkZ[ix][0]
        u = pred[ix] + Z[ix] - anchor
        fit = robfit(s, u, deg=deg, beta=beta)
        out[ix] = fit + anchor - Z[ix]
    return out


def nested_nnls_blend(X, y, folds):
    """exp103流: 各fold、他fold上でNNLS fit、held foldをpredict。delta空間ではなく
    exp103と同じく生pred空間でNNLS (exp103 nested_nnls_blend と一致)。"""
    uf = sorted(np.unique(folds))
    pred = np.full(len(y), np.nan)
    wacc = np.zeros(X.shape[1])
    for f in uf:
        tr = folds != f; te = folds == f
        w, _ = nnls(X[tr], y[tr])
        pred[te] = X[te] @ w
        wacc += w
    return pred, {MODEL_ORDER[i]: float(wacc[i] / len(uf)) for i in range(X.shape[1])}


def nested_projection(blend_pred, well_groups, geom, y, folds, degrees, betas):
    """projection の degree/beta を fold外(train fold)で選択し、test fold に適用 (leak-free)。
    well_groups は well 単位。fold は well 単位なので各 well は1 fold に属す。
    train fold 全 well で各(deg,beta)候補の projection を行い RMSE 最良を選択、
    その係数で test fold well を projection。"""
    MD, Z, a, lkMD, lkZ = geom
    uf = sorted(np.unique(folds))
    # well -> fold
    pred_nested = np.full(len(y), np.nan)
    chosen = []
    # precompute per-fold well lists
    for f in uf:
        tr_mask = folds != f
        te_mask = folds == f
        tr_wells = [w for w, ix in well_groups.items() if tr_mask[ix[0]]]
        te_wells = [w for w, ix in well_groups.items() if te_mask[ix[0]]]
        best = (None, None, np.inf)
        for deg in degrees:
            for beta in betas:
                # apply projection to train wells, eval RMSE
                acc_err = []
                for w in tr_wells:
                    ix = well_groups[w]
                    pj = _proj_one(blend_pred, ix, MD, Z, a, lkMD, lkZ, deg, beta)
                    acc_err.append((pj - y[ix]))
                e = np.concatenate(acc_err)
                c = float(np.sqrt(np.nanmean(e ** 2)))
                if c < best[2]:
                    best = (deg, beta, c)
        deg_b, beta_b, _ = best
        chosen.append((int(deg_b), float(beta_b)))
        for w in te_wells:
            ix = well_groups[w]
            pred_nested[ix] = _proj_one(blend_pred, ix, MD, Z, a, lkMD, lkZ, deg_b, beta_b)
    return pred_nested, chosen


def _proj_one(pred, ix, MD, Z, a, lkMD, lkZ, deg, beta):
    mdv = MD[ix]
    denom = max(mdv.max() - lkMD[ix][0], 1e-6)
    s = (mdv - lkMD[ix][0]) / denom
    anchor = a[ix][0] + lkZ[ix][0]
    u = pred[ix] + Z[ix] - anchor
    fit = robfit(s, u, deg=deg, beta=beta)
    return fit + anchor - Z[ix]


def main():
    # --- load OOF, inner-join on well_id,row_idx ---
    base = None
    for name in MODEL_ORDER:
        rel, col = MODELS[name]
        d = pd.read_csv(EXPDIR / rel, usecols=["well_id", "row_idx", "TVT", col])
        d = d.rename(columns={col: f"pred_{name}"})
        if base is None:
            base = d
        else:
            base = base.merge(d[["well_id", "row_idx", f"pred_{name}"]],
                              on=["well_id", "row_idx"], how="inner")
    # geometry from train_base
    geomdf = pd.read_parquet(ROOT / "data/processed/train_base_v001.parquet",
        columns=["well_id", "row_idx", "MD", "Z", "last_known_TVT", "last_known_MD", "last_known_Z"])
    df = base.merge(geomdf, on=["well_id", "row_idx"], how="inner")

    pred_cols = [f"pred_{n}" for n in MODEL_ORDER]
    mask = df["TVT"].notna()
    for c in pred_cols + ["MD", "Z", "last_known_TVT", "last_known_MD", "last_known_Z"]:
        mask &= df[c].notna() & np.isfinite(df[c])
    df = df[mask].reset_index(drop=True)

    wf = pd.read_csv(ROOT / "data/folds/folds_group_well_v001.csv")[["well_id", "fold"]].drop_duplicates("well_id")
    wf = wf.rename(columns={"fold": "fold_well"})
    tf = pd.read_csv(ROOT / "data/folds/folds_group_typewell_v001.csv")[["well_id", "fold"]].drop_duplicates("well_id")
    tf = tf.rename(columns={"fold": "fold_tw"})
    df = df.merge(wf, on="well_id", how="inner").merge(tf, on="well_id", how="inner").reset_index(drop=True)

    n_rows = len(df)
    X = np.column_stack([df[c].values for c in pred_cols])
    y = df["TVT"].values.astype(float)
    wid = df["well_id"].values
    row_idx = df["row_idx"].values
    MD = df["MD"].values.astype(float); Z = df["Z"].values.astype(float)
    a = df["last_known_TVT"].values.astype(float)
    lkMD = df["last_known_MD"].values.astype(float)
    lkZ = df["last_known_Z"].values.astype(float)
    well_groups = build_well_groups(wid, row_idx)
    geom = (MD, Z, a, lkMD, lkZ)

    degrees = [3, 5, 7]
    betas = [1.0, 2.0, 4.0]

    out = {"n_rows_common": int(n_rows), "model_order": MODEL_ORDER,
           "projection_design": "exp068 coord=U: U=pred+Z-anchor robust poly IRLS fit -> +anchor-Z",
           "degrees_swept": degrees, "betas_swept": betas}

    res_cv_pre = {}; res_cv_post = {}; res_weights = {}; res_chosen = {}; res_imp = {}
    for fold_col, key in [("fold_well", "well"), ("fold_tw", "typewell")]:
        folds = df[fold_col].values
        blend_pred, w = nested_nnls_blend(X, y, folds)
        cv_pre = rmse(blend_pred, y)
        proj_pred, chosen = nested_projection(blend_pred, well_groups, geom, y, folds, degrees, betas)
        cv_post = rmse(proj_pred, y)
        res_cv_pre[key] = round(cv_pre, 4)
        res_cv_post[key] = round(cv_post, 4)
        res_weights[key] = {k: round(v, 4) for k, v in w.items()}
        res_chosen[key] = chosen
        res_imp[key] = round(cv_pre - cv_post, 4)  # positive = improvement

    # fold一貫性: 両fold で改善 (improvement > 0) かつ chosen 係数が安定か
    both_improve = (res_imp["well"] > 0) and (res_imp["typewell"] > 0)
    out.update({
        "cv_blend_pre": res_cv_pre,
        "cv_blend_post_projection": res_cv_post,
        "improvement": res_imp,
        "blend_weights": res_weights,
        "projection_chosen_per_fold": {"format": "(degree,beta) per fold",
                                       "well": res_chosen["well"], "typewell": res_chosen["typewell"]},
        "fold_consistency": bool(both_improve),
        "leak_risk": "low",
        "leak_notes": ("NNLS重み・projection(deg,beta)は fold外で決定。projection は pred+既知幾何"
                       "(Z,MD,anchor=last_known_TVT)のみ使用。真TVTは評価のみ=leak-free。"),
    })
    decision = "採用" if both_improve else "不採用"
    out["notes_short"] = (
        f"honest(typewell) blend pre={res_cv_pre['typewell']} post={res_cv_post['typewell']} "
        f"imp={res_imp['typewell']:+}; well pre={res_cv_pre['well']} post={res_cv_post['well']} "
        f"imp={res_imp['well']:+}; fold_consistent={both_improve} -> {decision}")

    (OUTDIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # notes.md
    L = []
    L.append("# exp104 projection on honest blend\n")
    L.append(f"共通行数: {n_rows:,}  (3-model nested NNLS blend: exp022+exp100+exp008)\n")
    L.append("## projection 設計")
    L.append("- exp068/072 流 per-well projection を忠実移植 (coord='U')。")
    L.append("- U = blend_pred + Z - anchor (anchor = last_known_TVT + last_known_Z)。")
    L.append("- s = (MD - last_known_MD)/range で正規化、robust 多項式 IRLS fit (Cauchy 重み, 4 iters)。")
    L.append("- fit 後 out = fit + anchor - Z で TVT 空間へ戻す。")
    L.append("- degree∈{3,5,7}, beta∈{1,2,4} を fold外(train fold)で RMSE 最小選択 (nested, leak-free)。\n")
    L.append("## honest CV: projection 前 vs 後")
    L.append(f"- typewell-fold (honest): pre **{res_cv_pre['typewell']}** -> post **{res_cv_post['typewell']}**  (改善 {res_imp['typewell']:+})")
    L.append(f"- well-fold: pre {res_cv_pre['well']} -> post {res_cv_post['well']}  (改善 {res_imp['well']:+})")
    L.append(f"- 参考: exp103 honest blend = 10.09\n")
    L.append("## 選択係数 (fold外, (degree,beta))")
    L.append(f"- typewell: {res_chosen['typewell']}")
    L.append(f"- well: {res_chosen['well']}\n")
    L.append("## NNLS 重み (out-of-fold 平均)")
    for k in ["typewell", "well"]:
        L.append(f"- {k}: " + ", ".join(f"{m}={res_weights[k][m]}" for m in MODEL_ORDER))
    L.append("")
    L.append("## fold 一貫性")
    L.append(f"- 両fold改善: {both_improve}  -> **{decision}**")
    L.append("- 全fold一貫改善のみ採用。非一貫なら不採用 (本実験の判定は上記)。\n")
    L.append("## leak 評価")
    L.append("- low: 重み・projection係数は fold外決定、projection は予測+既知幾何のみ使用、真TVTは評価のみ。\n")
    L.append("## 結論")
    if both_improve:
        L.append(f"- projection 後処理で honest stack が 10.09 から {res_cv_post['typewell']} (typewell) へ改善。両fold一貫。採用候補。")
    else:
        L.append(f"- projection 後処理は honest blend (typewell {res_cv_pre['typewell']}) を一貫改善せず。不採用。")
        L.append("- 歴史的改善(exp072 nested 9.086)は artifact+exp026 base 上のものであり、本 PF系 3-model honest blend では再現せず。")
    L.append("\n## 次案")
    L.append("- 1) projection を delta(=pred-anchor)空間で fit する変種。")
    L.append("- 2) per-well でなく typewell グループ単位の low-order projection。")
    L.append("- 3) MD後半(外挿領域)のみ projection を効かせる ramp 併用。")
    (OUTDIR / "notes.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
