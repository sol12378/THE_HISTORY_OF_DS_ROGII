#!/usr/bin/env python3
"""exp067: 群II物理場 厳密LOWO + PF blend検証(leak-free).

案4(調和場)/5(dipベクトル場)/6(Kriging)の本質=フィールド構造曲面で
offsetを幾何決定。proper-LOWO(自well除外)で形成曲面を予測し:
1) 局所平面fit(dip-aware, 案5)で各wellのhidden TVTを予測
2) PFと誤差相関・blendを測定(broken wellでPFを補完できるか=案4-6の価値)

leak厳守: well Wの予測は他772 wellのみ。const_wellはknown区間TVT_inputのみ。
真TVTは評価のみ。
"""
from __future__ import annotations
import numpy as np, pandas as pd, glob, json
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/exp067_field_lowo_blend"; OUT.mkdir(parents=True, exist_ok=True)
FORM = "ASTNU"   # 先のLOWOで最良
K = 12


def main():
    base = pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id", "row_idx", "id", "X", "Y", "Z", "TVT", "TVT_input", "is_target", "is_known_tvt", "last_known_TVT"])
    # load formation per well WITH well_id (proper LOWO exclusion)
    files = sorted(glob.glob("data/raw/train/*__horizontal_well.csv"))
    XY = []; V = []; WID = []
    for f in files:
        wid = f.split("/")[-1].split("__")[0]
        d = pd.read_csv(f, usecols=["X", "Y", FORM]).dropna()
        if len(d) == 0:
            continue
        XY.append(d[["X", "Y"]].values); V.append(d[FORM].values)
        WID.append(np.full(len(d), wid))
    XY = np.vstack(XY); V = np.concatenate(V); WID = np.concatenate(WID)

    per = pd.read_csv("experiments/exp022_particle_filter/per_well.csv")
    pf = pd.read_csv("experiments/exp022_particle_filter/oof.csv")[["well_id", "row_idx", "pred_tvt"]]

    wells = sorted(base["well_id"].unique())
    rows = []
    for wi, wid in enumerate(wells):
        g = base[base["well_id"] == wid].sort_values("row_idx")
        wxy = g[["X", "Y", "Z"]].values
        known = g["is_known_tvt"].to_numpy(bool)
        tgt = g["is_target"].to_numpy(bool)
        if tgt.sum() < 5 or known.sum() < 3:
            continue
        # proper LOWO: exclude this well's formation points
        mask = WID != wid
        tree = cKDTree(XY[mask]); vv = V[mask]
        dist, idx = tree.query(wxy[:, :2], k=K)
        w_ = 1.0 / (dist + 1e-6); w_ /= w_.sum(1, keepdims=True)
        s_form = (w_ * vv[idx]).sum(1)
        # const_well from known region only
        tin = g["TVT_input"].to_numpy(float)
        km = known & ~np.isnan(tin)
        const = np.nanmedian(tin[km] - (s_form[km] - wxy[km, 2]))
        pred = s_form - wxy[:, 2] + const
        gg = g.copy(); gg["surf_pred"] = pred
        rows.append(gg[tgt][["well_id", "row_idx", "id", "TVT", "last_known_TVT", "surf_pred"]])
        if (wi + 1) % 150 == 0:
            print(f"  {wi+1}/{len(wells)}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out[["well_id", "row_idx", "id", "TVT", "last_known_TVT", "surf_pred"]].rename(
        columns={"surf_pred": "pred_tvt"}).to_csv(OUT / "oof.csv", index=False)
    out = out.merge(pf, on=["well_id", "row_idx"], how="left").rename(columns={"pred_tvt": "pf"})
    y = out["TVT"].to_numpy(float); a = out["last_known_TVT"].to_numpy(float)
    surf = out["surf_pred"].to_numpy(float); pfp = out["pf"].to_numpy(float)
    def rmse(p, yy=None):
        yy = y if yy is None else yy
        return float(np.sqrt(np.nanmean((p - yy) ** 2)))
    cv_surf = rmse(surf); cv_pf = rmse(pfp)
    # error correlation
    es = surf - y; ep = pfp - y; m = ~np.isnan(es) & ~np.isnan(ep)
    corr = float(np.corrcoef(es[m], ep[m])[0, 1])
    # simple blends
    best = (1e9, None)
    for w in np.linspace(0, 1, 21):
        bl = w * surf + (1 - w) * pfp
        r = rmse(bl)
        if r < best[0]:
            best = (r, w)
    # NNLS blend
    X = np.column_stack([np.nan_to_num(surf - a), np.nan_to_num(pfp - a)])
    wn, _ = nnls(X, y - a); cv_nnls = rmse(X @ wn + a)
    # broken well: does surf beat pf on exp022-broken wells?
    broken = set(per[per["pf_rmse"] > 20]["well_id"])
    bm = out["well_id"].isin(broken).to_numpy()
    cv_pf_brk = rmse(pfp[bm], y[bm]) if bm.sum() else np.nan
    cv_surf_brk = rmse(surf[bm], y[bm]) if bm.sum() else np.nan

    res = {
        "formation": FORM, "K": K,
        "cv_surface_LOWO": cv_surf, "cv_pf": cv_pf,
        "error_corr_surf_pf": corr,
        "best_linear_blend": {"cv": best[0], "w_surf": best[1]},
        "nnls_blend_cv": cv_nnls, "nnls_w": [float(x) for x in wn],
        "broken_wells": {"n_rows": int(bm.sum()), "pf_rmse": cv_pf_brk, "surf_rmse": cv_surf_brk},
        "leak_check": "LOWO: well's own formation points excluded (WID!=wid); const from known TVT_input only",
    }
    (OUT / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n[interpret] surf単体{cv_surf:.2f} vs PF{cv_pf:.2f}; blend best{best[0]:.3f}(w_surf={best[1]:.2f}); "
          f"NNLS{cv_nnls:.3f}; broken: PF{cv_pf_brk:.1f}/surf{cv_surf_brk:.1f}")


if __name__ == "__main__":
    main()
