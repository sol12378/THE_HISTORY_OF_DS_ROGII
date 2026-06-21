#!/usr/bin/env python3
"""exp026: final blend with tuned PF (exp025_pf_tuned) × geom × trees/nn/attn.

exp024 と同じ NNLS nested-fold ブレンド+平滑だが、PFソースを tuned PF(exp025_pf_tuned)に差替え。
--pf-exp で PF実験を指定(default exp025_pf_tuned)。最終 submission を出力。
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp026_final_blend"
OUT_DIR = Path("experiments") / EXP_ID
KEY = ["well_id", "row_idx"]
SMOOTH_W = 101
GEOM = {"geom": "exp014_geom_extrap", "trees": "exp018_model_blend",
        "nn": "exp019_seq_nn", "attn": "exp020_typewell_attn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pf-exp", default="exp025_pf_tuned")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] final blend, PF source = {a.pf_exp}")

    pf = pd.read_csv(f"experiments/{a.pf_exp}/oof.csv")[KEY + ["TVT", "last_known_TVT", "pred_tvt"]].rename(columns={"pred_tvt": "pf"})
    m = pf.merge(pd.read_csv(f"experiments/{GEOM['geom']}/oof.csv")[KEY + ["fold", "pred_tvt"]].rename(columns={"pred_tvt": "geom"}), on=KEY)
    for name in ["trees", "nn", "attn"]:
        m = m.merge(pd.read_csv(f"experiments/{GEOM[name]}/oof.csv")[KEY + ["pred_tvt"]].rename(columns={"pred_tvt": name}), on=KEY)
    cols = ["pf", "geom", "trees", "nn", "attn"]
    a_arr = m["last_known_TVT"].to_numpy(float)
    y = m["TVT"].to_numpy(float) - a_arr
    print(f"  rows={len(m)}  pf CV={tvt_rmse(m['TVT'],m['pf']):.4f}")

    folds = sorted(m["fold"].dropna().unique())
    oof_blend = np.zeros(len(m)); fold_w = {}
    for f in folds:
        tr = (m["fold"] != f).to_numpy(); te = (m["fold"] == f).to_numpy()
        Xtr = np.column_stack([m.loc[tr, c].to_numpy(float) - a_arr[tr] for c in cols])
        w, _ = nnls(Xtr, y[tr]); fold_w[int(f)] = w
        Xte = np.column_stack([m.loc[te, c].to_numpy(float) - a_arr[te] for c in cols])
        oof_blend[te] = a_arr[te] + Xte @ w
    cv_nested = tvt_rmse(m["TVT"], oof_blend)
    Xall = np.column_stack([m[c].to_numpy(float) - a_arr for c in cols])
    w_full, _ = nnls(Xall, y)
    print(f"  nested CV={cv_nested:.6f}  weights=" + ", ".join(f"{c}:{wi:.3f}" for c, wi in zip(cols, w_full)))

    m["oof_blend"] = oof_blend
    m = m.sort_values(KEY)
    m["blend_sm"] = m.groupby("well_id", sort=False)["oof_blend"].transform(
        lambda x: x.rolling(SMOOTH_W, min_periods=1, center=True).mean())
    cv_sm = tvt_rmse(m["TVT"], m["blend_sm"])
    print(f"  nested+smooth CV={cv_sm:.6f}")
    for f in folds:
        s = m[m["fold"] == f]; print(f"    fold{int(f)}: {tvt_rmse(s['TVT'],s['blend_sm']):.4f}")
    m[KEY + ["fold", "TVT", "last_known_TVT", "oof_blend", "blend_sm"] + cols].to_csv(OUT_DIR / "oof.csv", index=False)

    # test submission
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].copy()
    sub["well_id"] = sub["id"].str.rsplit("_", n=1).str[0]
    sub["row_idx"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
    tb = pd.read_parquet("data/processed/test_base_v001.parquet", columns=["well_id", "row_idx", "is_target", "last_known_TVT"])
    tb = tb[tb["is_target"].astype(bool)]
    sub = sub.merge(tb[["well_id", "row_idx", "last_known_TVT"]], on=["well_id", "row_idx"], how="left")
    SUB = {"pf": f"experiments/{a.pf_exp}/submission.csv", "geom": f"experiments/{GEOM['geom']}/submission.csv",
           "trees": f"experiments/{GEOM['trees']}/submission.csv", "nn": f"experiments/{GEOM['nn']}/submission.csv",
           "attn": f"experiments/{GEOM['attn']}/submission.csv"}
    for name in cols:
        sub = sub.merge(pd.read_csv(SUB[name]).rename(columns={"tvt": name})[["id", name]], on="id", how="left")
    at = sub["last_known_TVT"].to_numpy(float)
    sub["tvt"] = at + np.column_stack([sub[c].to_numpy(float) - at for c in cols]) @ w_full
    sub = sub.sort_values(["well_id", "row_idx"])
    sub["tvt"] = sub.groupby("well_id", sort=False)["tvt"].transform(lambda x: x.rolling(SMOOTH_W, min_periods=1, center=True).mean())
    out_sub = sample[["id"]].merge(sub[["id", "tvt"]], on="id", how="left", validate="one_to_one")
    assert not out_sub["tvt"].isna().any()
    out_sub.to_csv(OUT_DIR / "submission.csv", index=False)

    write_json(OUT_DIR / "result.json", {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "pf_source": a.pf_exp, "base_learners": cols,
        "pf_alone_cv": float(tvt_rmse(m["TVT"], m["pf"])),
        "nested_fold_cv": cv_nested, "nested_plus_smooth_cv": cv_sm,
        "full_weights": {c: float(wi) for c, wi in zip(cols, w_full)},
        "fold_weights": {str(f): {c: float(wi) for c, wi in zip(cols, w)} for f, w in fold_w.items()},
        "prev_best_exp024": 10.076887, "leak_risk": "none (leak-free, nested-fold weights)"})
    print(f"[{EXP_ID}] 完了 nested+smooth CV={cv_sm:.6f} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
