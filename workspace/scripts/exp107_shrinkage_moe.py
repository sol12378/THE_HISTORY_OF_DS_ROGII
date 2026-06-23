"""exp107: James-Stein / empirical-Bayes shrinkage (107a) + confidence-gated soft MoE (107b)
+ confidence-signal exploration (107c).

軽量・leak-free。既存 OOF のみ使用 (PF 再計算なし)。pandas/numpy/sklearn のみ。

考え方:
- RMSE 採点では「信じられない well を頑健事前へ縮小」が理論最適レバー。
- per-well offset = mean(pred - anchor) per well。anchor (=last_known_TVT の flat 事前) は
  滑らかな頑健事前。offset を 0 (=anchor) へ縮小 = James-Stein / 経験ベイズ partial pooling。
- 47 壊れ well (pf_rmse>20) が pooled RMSE を支配。縮小/soft-gate でここを anchor へ寄せられるか。

leak 規約:
- 全パラメータ (縮小係数・gate 係数) は fold 外 (train folds) で決定し held fold に適用。
- well-fold と typewell-fold の両方で評価。
- 信号は全て leak-free (予測由来: seed_std / model間分散 / anchor乖離 / known-prefix fit)。
- 真 TVT は評価のみ (per-well offset 推定にも真値は使わない: offset は pred-anchor で予測由来)。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(r"D:\Data_Science\autoresearch\competitions\rogii-wellbore-geology-prediction\workspace")
EXPDIR = ROOT / "experiments"
OUTDIR = EXPDIR / "exp107_shrinkage_moe"
OUTDIR.mkdir(parents=True, exist_ok=True)

# model -> (relpath, pred_col)
MODELS = {
    "exp022": ("exp022_particle_filter/oof.csv", "pred_tvt"),
    "exp100": ("exp100_soft_likpf/oof.csv", "pred_tvt"),
    "exp008": ("exp008_gr_rolling/oof.csv", "pred_tvt"),
    "anchor": ("exp120_anchor_repro/oof.csv", "pred_tvt"),
}
MODEL_ORDER = ["exp022", "exp100", "exp008", "anchor"]


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def load():
    base = None
    for name in MODEL_ORDER:
        rel, col = MODELS[name]
        usecols = ["well_id", "row_idx", "TVT", "last_known_TVT", col]
        if name == "exp100":
            usecols = usecols + ["seed_std"]
        df = pd.read_csv(EXPDIR / rel, usecols=usecols).rename(columns={col: f"pred_{name}"})
        if base is None:
            base = df
        else:
            keep = ["well_id", "row_idx", f"pred_{name}"]
            if name == "exp100":
                keep += ["seed_std"]
            base = base.merge(df[keep], on=["well_id", "row_idx"], how="inner")
    return base


def nested_nnls_blend(df, fold_col):
    """out-of-fold NNLS blend (107 の baseline)."""
    from scipy.optimize import nnls
    X = np.vstack([df[f"pred_{n}"].values for n in MODEL_ORDER]).T
    y = df["TVT"].values
    folds = df[fold_col].values
    pred = np.full(len(df), np.nan)
    wacc = np.zeros(len(MODEL_ORDER))
    uf = sorted(np.unique(folds))
    for f in uf:
        tr, te = folds != f, folds == f
        w, _ = nnls(X[tr], y[tr])
        pred[te] = X[te] @ w
        wacc += w
    return rmse(pred, y), pred, {MODEL_ORDER[i]: float(wacc[i] / len(uf)) for i in range(len(MODEL_ORDER))}


# ----------------------------------------------------------------------------
# per-well 集計 (leak-free signals)
# ----------------------------------------------------------------------------
def build_well_table(df):
    """各 well ごとに leak-free な信頼度信号と offset を集計。
    offset_<model> = mean(pred_model - anchor)  (anchor=last_known_TVT flat)
    -- これは予測由来であり真値を使わない。"""
    anchor = df["pred_anchor"].values  # = last_known_TVT (flat per well)
    g = df.groupby("well_id")
    rows = {}
    rows["n"] = g.size()
    # per-model offset (予測 - anchor の well 平均) = per-well 系統オフセット推定
    for m in ["exp022", "exp100", "exp008"]:
        off = df[f"pred_{m}"].values - anchor
        rows[f"offset_{m}"] = pd.Series(off, index=df.index).groupby(df["well_id"]).mean()
    # 信頼度信号 (全 leak-free)
    rows["seed_std_mean"] = df.groupby("well_id")["seed_std"].mean()
    # model間分散: 3 alignment model の予測の row 毎分散の well 平均
    al = np.vstack([df["pred_exp022"].values, df["pred_exp100"].values, df["pred_exp008"].values])
    model_var = al.var(axis=0)
    rows["model_var_mean"] = pd.Series(model_var, index=df.index).groupby(df["well_id"]).mean()
    # anchor 乖離: |blend_align - anchor| の well 平均 (blend_align は後で追加するので PF で代用)
    dev_pf = np.abs(df["pred_exp022"].values - anchor)
    rows["anchor_dev_mean"] = pd.Series(dev_pf, index=df.index).groupby(df["well_id"]).mean()
    # known-prefix fit: row_idx 末尾(=既知接合点)近傍で PF と anchor の食い違い。
    # leak-free: 先頭 (last_known 近傍) ほど anchor は信頼でき、そこでの乖離が大なら align 不審。
    # ここでは well 内最小 row_idx 近傍 (先頭 5%) の |pred_exp022 - anchor| を使う。
    def prefix_dev(sub):
        k = max(1, int(len(sub) * 0.05))
        s = sub.sort_values("row_idx").head(k)
        return float(np.abs(s["pred_exp022"].values - s["pred_anchor"].values).mean())
    rows["prefix_dev"] = g.apply(prefix_dev)
    wt = pd.DataFrame(rows)
    return wt


# ----------------------------------------------------------------------------
# 107a: James-Stein / 経験ベイズ縮小
# ----------------------------------------------------------------------------
def shrinkage_eval(df, well_tbl, base_pred, fold_col, model_for_offset="exp100"):
    """blend の per-well offset を anchor (=flat 事前) へ縮小。
    縮小係数 lambda は well ごとの不確実性 (seed_std/model_var) で決定し、
    全体強度 alpha を train folds で最適化 (leak-free)。
    縮小: pred' = pred - lambda_w * offset_w
      offset_w = mean(base_pred - anchor) per well (予測由来)
      lambda_w = alpha * unc_w / (unc_w + c)  -- 不確実性大ほど anchor へ強く縮小。
    """
    y = df["TVT"].values
    anchor = df["pred_anchor"].values
    folds = df[fold_col].values
    uf = sorted(np.unique(folds))

    # well 毎 offset (base_pred 由来)
    off_w = pd.Series(base_pred - anchor, index=df.index).groupby(df["well_id"]).mean()
    df = df.copy()
    df["_off_w"] = df["well_id"].map(off_w).values

    # 不確実性スコア (z-normalize した seed_std + model_var の合成, leak-free)
    unc = well_tbl["seed_std_mean"].copy()
    unc = (unc - unc.median()) / (unc.std() + 1e-9)
    mv = (well_tbl["model_var_mean"] - well_tbl["model_var_mean"].median()) / (well_tbl["model_var_mean"].std() + 1e-9)
    unc = (unc + mv).clip(lower=-2)  # 不確実性合成
    unc = unc - unc.min()  # >=0
    df["_unc"] = df["well_id"].map(unc).values

    def predict(alpha, c):
        lam = alpha * df["_unc"].values / (df["_unc"].values + c)
        lam = np.clip(lam, 0.0, 1.0)
        return base_pred - lam * df["_off_w"].values

    pred_nested = np.full(len(df), np.nan)
    chosen = []
    for f in uf:
        tr, te = folds != f, folds == f
        best = (np.inf, 0.0, 1.0)
        for alpha in np.linspace(0, 1.0, 11):
            for c in [0.25, 0.5, 1.0, 2.0, 4.0]:
                lam = alpha * df["_unc"].values[tr] / (df["_unc"].values[tr] + c)
                lam = np.clip(lam, 0.0, 1.0)
                p = base_pred[tr] - lam * df["_off_w"].values[tr]
                r = rmse(p, y[tr])
                if r < best[0]:
                    best = (r, alpha, c)
        _, a, c = best
        chosen.append((float(a), float(c)))
        lam_te = a * df["_unc"].values[te] / (df["_unc"].values[te] + c)
        lam_te = np.clip(lam_te, 0.0, 1.0)
        pred_nested[te] = base_pred[te] - lam_te * df["_off_w"].values[te]
    return rmse(pred_nested, y), pred_nested, chosen


# ----------------------------------------------------------------------------
# 107b: 信頼度ゲート soft MoE
# ----------------------------------------------------------------------------
def moe_eval(df, base_pred, fold_col):
    """Expert A = alignment blend (base_pred), Expert B = anchor (flat 事前)。
    gate w_A = sigmoid(b0 - b1*z_seed - b2*z_dev - b3*z_mvar)  (信頼度高で A、低で B)。
    係数は train folds で最適化 (leak-free), held fold に適用。
    """
    y = df["TVT"].values
    anchor = df["pred_anchor"].values
    folds = df[fold_col].values
    uf = sorted(np.unique(folds))

    def z(col):
        v = df[col].values.astype(float)
        return (v - np.nanmedian(v)) / (np.nanstd(v) + 1e-9)

    dev = np.abs(base_pred - anchor)
    sig = np.vstack([
        z("seed_std"),
        (dev - np.median(dev)) / (np.std(dev) + 1e-9),
    ])
    # model間分散 (row 毎)
    al = np.vstack([df["pred_exp022"].values, df["pred_exp100"].values, df["pred_exp008"].values])
    mvar = al.var(axis=0)
    sig = np.vstack([sig, (mvar - np.median(mvar)) / (np.std(mvar) + 1e-9)])

    def gate(params):
        b0, b1, b2, b3 = params
        logit = b0 - b1 * sig[0] - b2 * sig[1] - b3 * sig[2]
        wA = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        return wA

    def loss_on(mask, params):
        wA = gate(params)[mask]
        p = wA * base_pred[mask] + (1 - wA) * anchor[mask]
        return rmse(p, y[mask])

    from scipy.optimize import minimize
    pred_nested = np.full(len(df), np.nan)
    chosen = []
    x0 = np.array([3.0, 0.5, 0.5, 0.5])
    for f in uf:
        tr, te = folds != f, folds == f
        res = minimize(lambda pp: loss_on(tr, pp), x0, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-5})
        pp = res.x
        chosen.append([float(v) for v in pp])
        wA_te = gate(pp)[te]
        pred_nested[te] = wA_te * base_pred[te] + (1 - wA_te) * anchor[te]
    return rmse(pred_nested, y), pred_nested, chosen


# ----------------------------------------------------------------------------
# 107c: 信頼度信号の探索 (well error との相関)
# ----------------------------------------------------------------------------
def confidence_signal_corr(df, well_tbl):
    """各 leak-free 信号が pf_rmse(=well 誤差) とどれだけ相関するか。
    well 誤差 = exp022 (PF) の per-well RMSE。"""
    # per-well PF error (評価専用: 真値使用OK・信号自体は leak-free)
    err = (df["pred_exp022"].values - df["TVT"].values) ** 2
    wpf = np.sqrt(pd.Series(err, index=df.index).groupby(df["well_id"]).mean())
    wt = well_tbl.copy()
    wt["well_err"] = wpf
    signals = ["seed_std_mean", "model_var_mean", "anchor_dev_mean", "prefix_dev"]
    out = {}
    for s in signals:
        v = wt[[s, "well_err"]].dropna()
        out[s] = {
            "pearson": float(np.corrcoef(v[s], v["well_err"])[0, 1]),
            "spearman": float(v[s].rank().corr(v["well_err"].rank())),
        }
    # 組合せ (z 和) の相関
    zs = []
    for s in signals:
        col = (wt[s] - wt[s].median()) / (wt[s].std() + 1e-9)
        zs.append(col)
    combo = sum(zs)
    vc = pd.DataFrame({"combo": combo, "well_err": wt["well_err"]}).dropna()
    out["combo_zsum"] = {
        "pearson": float(np.corrcoef(vc["combo"], vc["well_err"])[0, 1]),
        "spearman": float(vc["combo"].rank().corr(vc["well_err"].rank())),
    }
    return out, wt


def broken_well_effect(df, broken_ids, base_pred, new_pred, label):
    y = df["TVT"].values
    mask = df["well_id"].isin(broken_ids).values
    return {
        f"{label}_broken_base_rmse": rmse(base_pred[mask], y[mask]),
        f"{label}_broken_new_rmse": rmse(new_pred[mask], y[mask]),
        f"{label}_clean_base_rmse": rmse(base_pred[~mask], y[~mask]),
        f"{label}_clean_new_rmse": rmse(new_pred[~mask], y[~mask]),
        f"{label}_n_broken_rows": int(mask.sum()),
    }


def main():
    df = load()
    pred_cols = [f"pred_{n}" for n in MODEL_ORDER]
    mask = df["TVT"].notna()
    for c in pred_cols:
        mask &= df[c].notna() & np.isfinite(df[c])
    df = df[mask].reset_index(drop=True)

    wf = pd.read_csv(ROOT / "data/folds/folds_group_well_v001.csv")[["well_id", "fold"]].drop_duplicates("well_id").rename(columns={"fold": "fold_well"})
    tf = pd.read_csv(ROOT / "data/folds/folds_group_typewell_v001.csv")[["well_id", "fold"]].drop_duplicates("well_id").rename(columns={"fold": "fold_tw"})
    df = df.merge(wf, on="well_id", how="inner").merge(tf, on="well_id", how="inner")
    n_rows = len(df)

    # 47 壊れ well (pf_rmse>20)
    pw = pd.read_csv(EXPDIR / "exp022_particle_filter/per_well.csv")
    broken_ids = set(pw.loc[pw.pf_rmse > 20, "well_id"].tolist())

    # baseline blend (NNLS, out-of-fold)
    cv_w, blend_w, weights_w = nested_nnls_blend(df, "fold_well")
    cv_tw, blend_tw, weights_tw = nested_nnls_blend(df, "fold_tw")

    well_tbl = build_well_table(df)

    # 107c: signal corr
    corr, wt_full = confidence_signal_corr(df, well_tbl)

    # 107a: shrinkage (honest=typewell-fold base + well-fold)
    sh_cv_tw, sh_pred_tw, sh_chosen_tw = shrinkage_eval(df, well_tbl, blend_tw, "fold_tw")
    sh_cv_w, sh_pred_w, sh_chosen_w = shrinkage_eval(df, well_tbl, blend_w, "fold_well")

    # 107b: MoE soft gate
    moe_cv_tw, moe_pred_tw, moe_chosen_tw = moe_eval(df, blend_tw, "fold_tw")
    moe_cv_w, moe_pred_w, moe_chosen_w = moe_eval(df, blend_w, "fold_well")

    # 47 壊れ well 効果
    bw_sh = broken_well_effect(df, broken_ids, blend_tw, sh_pred_tw, "shrink_tw")
    bw_moe = broken_well_effect(df, broken_ids, blend_tw, moe_pred_tw, "moe_tw")

    result = {
        "n_rows_common": int(n_rows),
        "n_broken_wells": len(broken_ids),
        "baseline_blend_cv": {"well": cv_w, "typewell": cv_tw},
        "baseline_blend_weights": {"well": weights_w, "typewell": weights_tw},
        "shrinkage": {
            "cv": {"well": sh_cv_w, "typewell": sh_cv_tw},
            "improvement": {"well": cv_w - sh_cv_w, "typewell": cv_tw - sh_cv_tw},
        },
        "moe": {
            "cv": {"well": moe_cv_w, "typewell": moe_cv_tw},
            "improvement": {"well": cv_w - moe_cv_w, "typewell": cv_tw - moe_cv_tw},
        },
        "confidence_signal_corr": {k: v["spearman"] for k, v in corr.items()},
        "confidence_signal_corr_full": corr,
        "broken_well_effect": {**bw_sh, **bw_moe},
        "leak_risk": (
            "low: 縮小係数(alpha,c)・gate係数(b0..b3)は train folds のみで決定し held fold へ適用。"
            "信号(seed_std/model_var/anchor_dev/prefix_dev)・offset は全て予測由来でleak-free。"
            "真TVTは評価とper-well誤差算出(信号探索)にのみ使用。"
        ),
        "notes_short": "honest=typewell-fold。縮小=offset→anchorのpartial pooling、MoE=信頼度soft gate。",
    }
    with open(OUTDIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # notes.md
    best_sig = max(corr.items(), key=lambda kv: abs(kv[1]["spearman"]))
    L = []
    L.append("# exp107 縮小(James-Stein/経験ベイズ) + 信頼度ゲートMoE\n")
    L.append(f"共通行数: {n_rows:,} / 壊れwell(pf_rmse>20): {len(broken_ids)}\n")
    L.append("## baseline blend (NNLS out-of-fold)")
    L.append(f"- well-fold: {cv_w:.4f}")
    L.append(f"- typewell-fold (honest): {cv_tw:.4f}\n")
    L.append("## 107a 縮小効果 (offset を anchor へ partial pooling)")
    L.append(f"- well-fold: {sh_cv_w:.4f} (改善 {cv_w - sh_cv_w:+.4f})")
    L.append(f"- typewell-fold: {sh_cv_tw:.4f} (改善 {cv_tw - sh_cv_tw:+.4f})\n")
    L.append("## 107b MoE soft gate 効果 (alignment vs anchor)")
    L.append(f"- well-fold: {moe_cv_w:.4f} (改善 {cv_w - moe_cv_w:+.4f})")
    L.append(f"- typewell-fold: {moe_cv_tw:.4f} (改善 {cv_tw - moe_cv_tw:+.4f})\n")
    L.append("## 107c 信頼度信号 vs well誤差(pf_rmse) [spearman]")
    for k, v in corr.items():
        L.append(f"- {k}: spearman={v['spearman']:+.3f} pearson={v['pearson']:+.3f}")
    L.append(f"\n最良信号: **{best_sig[0]}** (spearman={best_sig[1]['spearman']:+.3f})\n")
    L.append("## 47壊れwellへの効果 (typewell-fold base)")
    L.append(f"- 縮小: broken {bw_sh['shrink_tw_broken_base_rmse']:.3f} -> {bw_sh['shrink_tw_broken_new_rmse']:.3f} | "
             f"clean {bw_sh['shrink_tw_clean_base_rmse']:.4f} -> {bw_sh['shrink_tw_clean_new_rmse']:.4f}")
    L.append(f"- MoE : broken {bw_moe['moe_tw_broken_base_rmse']:.3f} -> {bw_moe['moe_tw_broken_new_rmse']:.3f} | "
             f"clean {bw_moe['moe_tw_clean_base_rmse']:.4f} -> {bw_moe['moe_tw_clean_new_rmse']:.4f}\n")
    L.append("## 結論")
    imp = max(cv_tw - sh_cv_tw, cv_tw - moe_cv_tw)
    if imp > 1e-3:
        L.append(f"- typewell-fold で RMSE を下げられた (最大改善 {imp:+.4f})。")
    else:
        L.append(f"- typewell-fold では実質改善なし (最大改善 {imp:+.4f})。縮小/MoEは現blendを上回らず。")
    L.append("## leak懸念")
    L.append(f"- {result['leak_risk']}")
    with open(OUTDIR / "notes.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
