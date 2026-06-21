#!/usr/bin/env python3
"""exp018: モデル多様化 blend (LGBM + XGBoost + CatBoost) on exp014幾何特徴量。

戦略 Tier2 優先A。木モデル間の誤差非相関でCVを底上げ。
- 特徴量: exp014 (SAFE+A+B+C+D+F)。typewell不使用 → well-grouped fold は leak-free。
- fold: folds_group_well_v001 (exp008/014/015と同一、exp015 13.520と直接比較可)。
- target: TVT - last_known_TVT。
- blend: OOFで等加重 と 凸最適化加重 を両方評価。headlineは頑健な等加重を基本に、
  凸加重が全fold一貫改善のときのみ採用。
- 出力 oof.csv/submission.csv は後段で exp015 平滑化を適用可能。
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from rogii.training.baselines import (PRED_COL, target_rows, tvt_rmse, write_json,
    now_jst, build_submission, ensure_exp_dir, attach_folds, load_base_inputs)

spec = importlib.util.spec_from_file_location("exp014", ROOT/"scripts"/"exp014_geom_extrap.py")
exp014 = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp014)

EXP_ID = "exp018_model_blend"
EXP_DIR = Path("experiments") / EXP_ID
FEATURES = exp014.ALL_FEATURES


def fit_lgbm(Xtr, ytr, Xva, yva, Xte):
    p = {"objective":"regression","metric":"rmse","learning_rate":0.05,"num_leaves":63,
         "max_depth":-1,"min_data_in_leaf":50,"feature_fraction":0.9,"bagging_fraction":0.9,
         "bagging_freq":1,"lambda_l2":1.0,"verbosity":-1,"seed":42,"num_threads":8}
    m = lgb.LGBMRegressor(**p, n_estimators=3000)
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],eval_metric="rmse",
          callbacks=[lgb.early_stopping(60,verbose=False)])
    bi = int(m.best_iteration_ or 3000)
    return m.predict(Xva,num_iteration=bi), m.predict(Xte,num_iteration=bi)

def fit_xgb(Xtr, ytr, Xva, yva, Xte):
    m = XGBRegressor(n_estimators=3000, learning_rate=0.05, max_depth=7,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, min_child_weight=20,
        early_stopping_rounds=60, eval_metric="rmse", n_jobs=8, random_state=42, tree_method="hist")
    m.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    return m.predict(Xva), m.predict(Xte)

def fit_cat(Xtr, ytr, Xva, yva, Xte):
    m = CatBoostRegressor(iterations=3000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
        loss_function="RMSE", random_seed=42, thread_count=8, early_stopping_rounds=60, verbose=False)
    m.fit(Xtr,ytr,eval_set=(Xva,yva),use_best_model=True)
    return m.predict(Xva), m.predict(Xte)

MODELS = {"lgbm":fit_lgbm, "xgb":fit_xgb, "cat":fit_cat}


def main():
    ensure_exp_dir(EXP_DIR)
    print(f"[{EXP_ID}] model diversification blend on exp014 features")
    train, test, folds, sample = load_base_inputs(
        ROOT/"data/processed/train_base_v001.parquet",
        ROOT/"data/processed/test_base_v001.parquet",
        ROOT/"data/folds/folds_group_well_v001.csv",
        ROOT/"data/raw/sample_submission.csv")
    gmean = float(train.loc[~train["is_gr_missing"].astype(bool),"GR"].mean())
    print("enrich train ..."); train = exp014.enrich(train, gmean)
    print("enrich test  ..."); test  = exp014.enrich(test, gmean)
    train = attach_folds(train, folds)

    tr_t = target_rows(train).reset_index(drop=True)
    te_t = target_rows(test).reset_index(drop=True)
    y = (tr_t["TVT"].astype(float) - tr_t["last_known_TVT"].astype(float)).to_numpy()
    anchor_tr = tr_t["last_known_TVT"].to_numpy(float)
    anchor_te = te_t["last_known_TVT"].to_numpy(float)
    fold = tr_t["fold"].to_numpy()
    folds_u = sorted(np.unique(fold)); nf = len(folds_u)

    oof = {k: np.zeros(len(tr_t)) for k in MODELS}
    tep = {k: np.zeros(len(te_t)) for k in MODELS}
    for f in folds_u:
        vm = fold==f; tm=~vm
        Xtr,Xva = tr_t.loc[tm,FEATURES], tr_t.loc[vm,FEATURES]
        ytr,yva = y[tm], y[vm]
        for name,fn in MODELS.items():
            vd, td = fn(Xtr,ytr,Xva,yva,te_t[FEATURES])
            oof[name][vm]=vd; tep[name]+=td/nf
            fr = tvt_rmse(tr_t.loc[vm,"TVT"], anchor_tr[vm]+vd)
            print(f"  fold{f} {name:5s} RMSE={fr:.6f}")

    # per-model CV
    cv = {k: tvt_rmse(tr_t["TVT"], anchor_tr+oof[k]) for k in MODELS}
    # equal-weight blend
    eq = np.mean([oof[k] for k in MODELS], axis=0)
    cv_eq = tvt_rmse(tr_t["TVT"], anchor_tr+eq)
    # convex-optimized weights on OOF
    names = list(MODELS); O = np.vstack([oof[k] for k in names]).T  # (n,3)
    true_delta = y
    def loss(w):
        pred = O@w
        return float(np.sqrt(np.mean((true_delta-pred)**2)))
    cons = ({"type":"eq","fun":lambda w: w.sum()-1},)
    res = minimize(loss, np.full(len(names),1/len(names)), method="SLSQP",
                   bounds=[(0,1)]*len(names), constraints=cons)
    w_opt = res.x
    blend_opt = O@w_opt
    cv_opt = tvt_rmse(tr_t["TVT"], anchor_tr+blend_opt)
    # fold consistency of opt vs best single
    best_single = min(cv, key=cv.get)
    fold_opt_consistent = True
    for f in folds_u:
        vm=fold==f
        r_opt = tvt_rmse(tr_t.loc[vm,"TVT"], anchor_tr[vm]+(O[vm]@w_opt))
        r_bs  = tvt_rmse(tr_t.loc[vm,"TVT"], anchor_tr[vm]+oof[best_single][vm])
        if r_opt > r_bs + 1e-6: fold_opt_consistent=False

    # choose blend: prefer opt if it beats eq AND is fold-consistent vs best single
    use_opt = (cv_opt <= cv_eq) and fold_opt_consistent
    chosen = "opt" if use_opt else "eq"
    blend_oof = blend_opt if use_opt else eq
    blend_te  = (np.vstack([tep[k] for k in names]).T @ w_opt) if use_opt else np.mean([tep[k] for k in MODELS],axis=0)
    cv_final = cv_opt if use_opt else cv_eq

    # save oof/submission (for exp015 smoothing)
    oof_df = tr_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof_df["pred_delta"]=blend_oof; oof_df[PRED_COL]=anchor_tr+blend_oof
    for k in MODELS: oof_df[f"pred_{k}"]=anchor_tr+oof[k]
    oof_df.to_csv(EXP_DIR/"oof.csv", index=False)
    test_out = test.copy()
    test_out.loc[test_out["is_target"].astype(bool), PRED_COL]=anchor_te+blend_te
    build_submission(test_out, sample, PRED_COL).to_csv(EXP_DIR/"submission.csv", index=False)

    # error correlation matrix (diversity check)
    err = {k: (anchor_tr+oof[k]) - tr_t["TVT"].to_numpy(float) for k in MODELS}
    corr = pd.DataFrame(err).corr()

    result = {"exp_id":EXP_ID,"created_at":now_jst(),"status":"completed",
        "fold":"folds_group_well_v001","model":"LGBM+XGB+CatBoost blend",
        "per_model_cv":cv, "cv_equal_weight":cv_eq, "cv_opt_weight":cv_opt,
        "opt_weights":dict(zip(names, [round(float(x),4) for x in w_opt])),
        "chosen_blend":chosen, "cv_final_preSmooth":cv_final,
        "exp015_best_ref":13.520383, "exp014_ref":13.525189,
        "improvement_vs_exp014":round(13.525189-cv_final,6),
        "fold_opt_consistent":fold_opt_consistent,
        "error_corr":corr.round(4).to_dict(),
        "leak_risk":"low (geom features only, well-grouped fold leak-free; blend weights from OOF)",
        "note":"run exp015_seq_smooth --model-exp exp018_model_blend for final smoothed CV"}
    write_json(EXP_DIR/"result.json", result)
    print("\n=== RESULT (well-grouped fold, pre-smoothing) ===")
    for k in MODELS: print(f"  {k:5s} CV = {cv[k]:.6f}")
    print(f"  equal-weight blend = {cv_eq:.6f}")
    print(f"  opt-weight blend   = {cv_opt:.6f}  weights={dict(zip(names,[round(float(x),3) for x in w_opt]))}")
    print(f"  CHOSEN ({chosen}) = {cv_final:.6f}   vs exp014 {13.525189-cv_final:+.6f} / exp015 {13.520383-cv_final:+.6f}")
    print(f"\n  error correlation:\n{corr.round(3).to_string()}")

if __name__=="__main__":
    main()
