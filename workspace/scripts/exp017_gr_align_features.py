#!/usr/bin/env python3
"""exp017: GR-typewell系列照合特徴量(Group H)を学習モデルに投入する最終GR実験。

背景(2026-06-02 oracle診断, [[gr-offset-ceiling]]):
  geom path(exp014)は TVT(MD)形状をほぼ正しく与える。残差は低周波offset。
  glob_oracle(1param/well)=8.2 / seg_oracle(4param/well)=4.4 → 構造上はCV<=5可能。
  しかしGRが選ぶoffsetは真offsetと corr=+0.155 と弱い。直接適用(diag_seq_align)は
  geomを超えない(13.67〜)。→ 直接照合ではなく「弱offset信号をtree modelにfeature化して
  shrink利用」が唯一の現実的GR活用。本実験でそれを定量化する。

設計:
  base = exp014 の全特徴量(SAFE+A+B+C+D+F)。
  +Group H (GR alignment, 完全leak-free。観測GR+typewell+幾何centerのみ。hidden TVT不使用):
    per-well: h_align_offset(band±30でGRコスト最小のoffset), h_align_gain(cost0-best),
              h_tw_corr(known較正相関), h_cal_a/h_cal_b(GR較正係数)
    per-row : h_gr_resid_at_geom(geom centerでのGR残差), h_local_offset(局所窓offset)
  center = last_known_TVT + f_extrap_quad_dMD (exp014幾何の主項, leak-free, self-contained)

評価: **typewell-grouped fold** (folds_group_typewell_v001.csv) — exp011 leak回避の必須条件。
  同fold上で base-only と base+H を学習し、Hの寄与を公平に切り出す。
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from rogii.training.baselines import (PRED_COL, SAFE_FEATURES, target_rows,
    tvt_rmse, write_json, now_jst, build_submission, ensure_exp_dir)

# import exp014 builders
spec = importlib.util.spec_from_file_location("exp014", ROOT/"scripts"/"exp014_geom_extrap.py")
exp014 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp014)

EXP_ID = "exp017_gr_align_features"
EXP_DIR = Path("experiments") / EXP_ID
GROUP_H_WELL = ["h_align_offset","h_align_gain","h_tw_corr","h_cal_a","h_cal_b"]
GROUP_H_ROW  = ["h_gr_resid_at_geom","h_local_offset"]
GROUP_H = GROUP_H_WELL + GROUP_H_ROW
BASE_FEATURES = exp014.ALL_FEATURES
ALL_FEATURES = BASE_FEATURES + GROUP_H
BAND, STEP, LOCAL_W = 30.0, 0.5, 60


def add_group_h(df: pd.DataFrame, tw_by: dict) -> pd.DataFrame:
    """GR alignment 特徴量。center = last_known_TVT + f_extrap_quad_dMD。"""
    df = df.sort_values(["well_id","row_idx"]).copy()
    center_all = df["last_known_TVT"].astype(float) + df["f_extrap_quad_dMD"].astype(float)
    df["_center"] = center_all
    cs = np.arange(-BAND, BAND+STEP, STEP)
    well_recs = []
    resid = np.full(len(df), 0.0); local_off = np.full(len(df), 0.0)
    pos = {wid: idx.to_numpy() for wid, idx in df.groupby("well_id", sort=False).groups.items()}
    is_tgt = df["is_target"].astype(bool).to_numpy()
    is_known = df["is_known_tvt"].astype(bool).to_numpy()
    is_grmiss = df["is_gr_missing"].astype(bool).to_numpy()
    GR = df["GR"].to_numpy(float); TVTin = df["TVT_input"].to_numpy(float)
    center = df["_center"].to_numpy(float)
    df_index_arr = np.arange(len(df))
    for wid, rows in pos.items():
        tw_g = tw_by.get(wid)
        sub = rows
        ktgt = sub[is_tgt[sub]]
        rec = {"well_id":wid,"h_align_offset":0.0,"h_align_gain":0.0,
               "h_tw_corr":np.nan,"h_cal_a":1.0,"h_cal_b":0.0}
        if tw_g is None or len(tw_g) < 2 or len(ktgt) == 0:
            well_recs.append(rec); continue
        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tvt_f = tw_s["TVT"].to_numpy(float); gr_f = tw_s["GR"].to_numpy(float)
        kn = sub[is_known[sub] & ~is_grmiss[sub]]
        if len(kn) >= 10:
            twk = np.interp(TVTin[kn], tvt_f, gr_f); obs_k = GR[kn]
            A = np.vstack([twk, np.ones_like(twk)]).T
            coef,*_ = np.linalg.lstsq(A, obs_k, rcond=None)
            a,b = float(coef[0]), float(coef[1])
            if abs(a) < 1e-3: a,b = 1.0,0.0
            rec["h_cal_a"], rec["h_cal_b"] = a,b
            if np.std(twk) > 1e-6:
                rec["h_tw_corr"] = float(np.corrcoef(obs_k, twk)[0,1])
        else:
            a,b = 1.0,0.0
        # target rows with GR
        tg = ktgt[~is_grmiss[ktgt]]
        if len(tg) >= 8:
            tw_eq = (GR[tg]-b)/a; cen = center[tg]
            # vectorized global-offset cost over band
            pred = np.interp((cen[:,None]+cs[None,:]).ravel(), tvt_f, gr_f).reshape(len(tg),-1)
            costs = np.mean(np.abs(tw_eq[:,None]-pred), axis=0)
            j = int(np.argmin(costs))
            rec["h_align_offset"] = float(cs[j])
            rec["h_align_gain"] = float(costs[len(cs)//2] - costs[j])  # cost(c~0) - best
            resid[tg] = tw_eq - np.interp(cen, tvt_f, gr_f)
            # per-row local offset: vectorized rolling-window argmin
            if len(tg) >= LOCAL_W:
                csl = np.arange(-15, 15+STEP, STEP)
                M = np.abs(tw_eq[:,None] -
                    np.interp((cen[:,None]+csl[None,:]).ravel(), tvt_f, gr_f).reshape(len(tg),-1))
                cs2 = np.vstack([np.zeros((1,M.shape[1])), np.cumsum(M,axis=0)])
                h = LOCAL_W//2; ar = np.arange(len(tg))
                lo = np.clip(ar-h,0,len(tg)); hi = np.clip(ar+h,0,len(tg))
                win = (cs2[hi]-cs2[lo])/np.maximum(hi-lo,1)[:,None]
                local_off[tg] = csl[np.argmin(win,axis=1)]
        well_recs.append(rec)
    wr = pd.DataFrame(well_recs)
    df = df.merge(wr, on="well_id", how="left")
    df["h_gr_resid_at_geom"] = resid
    df["h_local_offset"] = local_off
    return df.drop(columns=["_center"])


def run_model(train_t, test_t, y, features, folds_map, tag):
    params = {"objective":"regression","metric":"rmse","learning_rate":0.05,
              "num_leaves":63,"max_depth":-1,"min_data_in_leaf":50,
              "feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":1,
              "lambda_l2":1.0,"verbosity":-1,"seed":42,"num_threads":8}
    fold = train_t["fold"].to_numpy()
    oof = np.zeros(len(train_t)); test_p = np.zeros(len(test_t))
    n_folds = len(np.unique(fold)); fold_rows = []; imps = []
    for f in sorted(np.unique(fold)):
        vm = fold == f; tm = ~vm
        m = lgb.LGBMRegressor(**params, n_estimators=1500)
        m.fit(train_t.loc[tm, features], y[tm],
              eval_set=[(train_t.loc[vm, features], y[vm])], eval_metric="rmse",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        bi = int(m.best_iteration_ or 1500)
        vd = m.predict(train_t.loc[vm, features], num_iteration=bi); oof[vm] = vd
        ftvt = train_t.loc[vm,"last_known_TVT"].to_numpy(float)+vd
        fr = tvt_rmse(train_t.loc[vm,"TVT"], ftvt)
        fold_rows.append({"fold":int(f),"rmse":fr,"best_iter":bi})
        print(f"  [{tag}] fold {f} RMSE={fr:.6f} best_iter={bi}")
        test_p += m.predict(test_t[features], num_iteration=bi)/n_folds
        imps.append(pd.DataFrame({"feature":features,
            "gain":m.booster_.feature_importance("gain")}))
    ftvt_all = train_t["last_known_TVT"].to_numpy(float)+oof
    overall = tvt_rmse(train_t["TVT"], ftvt_all)
    fi = pd.concat(imps).groupby("feature",as_index=False)["gain"].mean().sort_values("gain",ascending=False)
    return overall, oof, test_p, pd.DataFrame(fold_rows), fi


def main():
    ensure_exp_dir(EXP_DIR)
    print(f"[{EXP_ID}] GR alignment features (Group H) on typewell-grouped fold")
    train = pd.read_parquet("data/processed/train_base_v001.parquet")
    test = pd.read_parquet("data/processed/test_base_v001.parquet")
    tw_tr = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
                            columns=["well_id","TVT","GR"])
    tw_te = pd.read_parquet("data/processed/typewell_test_base_v001.parquet",
                            columns=["well_id","TVT","GR"])
    tw_by_tr = {w:g for w,g in tw_tr.groupby("well_id",sort=False)}
    tw_by_te = {w:g for w,g in tw_te.groupby("well_id",sort=False)}
    folds = pd.read_csv("data/folds/folds_group_typewell_v001.csv")
    sample = pd.read_csv("data/raw/sample_submission.csv")

    gmean = float(train.loc[~train["is_gr_missing"].astype(bool),"GR"].mean())
    print("enrich train (exp014 base) ..."); train = exp014.enrich(train, gmean)
    print("enrich test  (exp014 base) ..."); test  = exp014.enrich(test, gmean)
    print("Group H train ..."); train = add_group_h(train, tw_by_tr)
    print("Group H test  ..."); test  = add_group_h(test, tw_by_te)

    fmap = folds.set_index("well_id")["fold"]
    train["fold"] = train["well_id"].map(fmap)
    if train["fold"].isna().any(): raise ValueError("fold欠損")
    train["fold"] = train["fold"].astype(int)

    train_t = target_rows(train).reset_index(drop=True)
    test_t = target_rows(test).reset_index(drop=True)
    y = (train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)).to_numpy()

    print("\n--- baseline: exp014 features ONLY (typewell fold) ---")
    base_cv, base_oof, _, base_folds, _ = run_model(train_t, test_t, y, BASE_FEATURES, fmap, "base")
    print("\n--- exp017: base + Group H (typewell fold) ---")
    h_cv, h_oof, h_test, h_folds, fi = run_model(train_t, test_t, y, ALL_FEATURES, fmap, "base+H")

    # save oof/submission for base+H
    oof = train_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof["pred_delta"]=h_oof; oof[PRED_COL]=oof["last_known_TVT"].astype(float)+h_oof
    oof.to_csv(EXP_DIR/"oof.csv", index=False)
    test_out = test.copy()
    test_out.loc[test_out["is_target"].astype(bool), PRED_COL] = (
        test_t["last_known_TVT"].to_numpy(float)+h_test)
    build_submission(test_out, sample, PRED_COL).to_csv(EXP_DIR/"submission.csv", index=False)
    fi.to_csv(EXP_DIR/"feature_importance.csv", index=False)
    h_folds.to_csv(EXP_DIR/"cv.csv", index=False)

    h_rank = {r.feature:i+1 for i,r in enumerate(fi.itertuples())}
    result = {"exp_id":EXP_ID,"created_at":now_jst(),"status":"completed",
        "fold":"folds_group_typewell_v001 (leak-safe)",
        "base_cv_typewell_fold":base_cv,"exp017_cv_typewell_fold":h_cv,
        "group_h_gain":round(base_cv-h_cv,6),
        "note_exp008_wellfold_ref":13.808621,"exp015_best_wellfold":13.520383,
        "group_h_features":GROUP_H,
        "group_h_importance_rank":{f:h_rank.get(f) for f in GROUP_H},
        "leak_risk":"low (typewell-grouped fold; H uses only obs GR+typewell+geom, no hidden TVT)",
        "conclusion_ref":"[[gr-offset-ceiling]] oracle ceiling ~8.2; GR offset corr 0.155"}
    write_json(EXP_DIR/"result.json", result)
    print(f"\n=== RESULT (typewell-grouped fold) ===")
    print(f"  base (exp014 feats) CV = {base_cv:.6f}")
    print(f"  base + Group H  CV     = {h_cv:.6f}")
    print(f"  Group H gain           = {base_cv-h_cv:+.6f}")
    print(f"  Group H importance ranks: "+", ".join(f"{f}=#{h_rank.get(f)}" for f in GROUP_H))

if __name__=="__main__":
    main()
