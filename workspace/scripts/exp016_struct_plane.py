#!/usr/bin/env python3
"""exp016: Group G = 3D structural plane extrapolation, on top of exp014 (Group F).

地質的動機: TVT(層序的鉛直位置)は、平面的に傾斜する地層では位置(X,Y,Z)の線形関数。
  TVT ≈ cx·X + cy·Y + cz·Z + c0  （bedding平面への射影）
hidden区間でも X/Y/Z は完全既知 → known区間で平面を較正し、既知hidden座標へ適用すれば
TVT delta を構造傾斜の3成分(縦+横)込みで外挿できる。
Group F の f_extrap_z(縦成分のみ) を、横方向dipまで含めて一般化したもの。

Group G（leak-free: known区間のみで較正、hiddenのTVT不使用、X/Y/Z/MDは観測量）:
  per-well（座標は well 内で中心化・スケーリングして数値安定化）:
    g_plane_full_r2   : known全体の TVT~X,Y,Z 平面フィット R²
    g_plane_cz        : Z係数（縦成分dip）
    g_plane_choriz    : 水平係数の大きさ sqrt(cx²+cy²)（横成分dip強度）
    g_plane_local_r2  : 直近300 known行での平面フィット R²
  per-row（既知hidden座標へ平面適用 → delta推定、±200にclip）:
    g_plane_full_delta  : 全体平面 pred(X,Y,Z) − last_known_TVT
    g_plane_local_delta : 局所平面 pred − last_known_TVT
    g_plane_disagree    : |g_plane_full_delta − f_extrap_quad_dMD|（外挿クロスチェック）

baseline = exp014 (SAFE + A+B+C+D+F)。Group G を上乗せ。fold は exp008/014 と同一。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rogii.training.baselines import (
    PRED_COL, SAFE_FEATURES, attach_folds, build_submission,
    ensure_exp_dir, load_base_inputs, now_jst, target_rows, tvt_rmse, write_json,
)

# import exp014 builders (module main is __main__-guarded, safe to import)
_spec = importlib.util.spec_from_file_location("exp014", ROOT / "scripts" / "exp014_geom_extrap.py")
exp014 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp014)

EXP_ID = "exp016_struct_plane"
EXP_DIR = Path("experiments") / EXP_ID

GROUP_G_WELL = ["g_plane_full_r2", "g_plane_cz", "g_plane_choriz", "g_plane_local_r2"]
GROUP_G_ROW  = ["g_plane_full_delta", "g_plane_local_delta", "g_plane_disagree"]
GROUP_G = GROUP_G_WELL + GROUP_G_ROW

ALL_FEATURES = exp014.ALL_FEATURES + GROUP_G
PLANE_CLIP = 200.0
LOCAL_K = 300


def _fit_plane(x, y, z, tv):
    """中心化・スケーリングして TVT~X,Y,Z をOLS。(coef_centered, mean, scale, r2) を返す。
    coef は元スケールの (cx,cy,cz,c0) に戻す。"""
    n = len(tv)
    if n < 6:
        return None
    XYZ = np.column_stack([x, y, z]).astype(float)
    mu = XYZ.mean(axis=0)
    sd = XYZ.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Xs = (XYZ - mu) / sd
    if np.ptp(Xs, axis=0).max() < 1e-6:
        return None
    A = np.column_stack([Xs, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, tv, rcond=None)
    pred = A @ coef
    if not np.all(np.isfinite(pred)):
        return None
    ss_res = float(np.sum((tv - pred) ** 2))
    ss_tot = float(np.sum((tv - tv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
    # back to original scale: tv = sum(coef_i*(c_i-mu_i)/sd_i) + coef_3
    cx, cy, cz = coef[0] / sd[0], coef[1] / sd[1], coef[2] / sd[2]
    c0 = coef[3] - (coef[0] * mu[0] / sd[0] + coef[1] * mu[1] / sd[1] + coef[2] * mu[2] / sd[2])
    return np.array([cx, cy, cz, c0]), float(r2)


def _plane_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    coefs = {}
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        x = g["X"].to_numpy(float); y = g["Y"].to_numpy(float)
        z = g["Z"].to_numpy(float); tv = g["TVT_input"].to_numpy(float)
        full = _fit_plane(x, y, z, tv)
        k = min(LOCAL_K, len(g))
        local = _fit_plane(x[-k:], y[-k:], z[-k:], tv[-k:]) if k >= 6 else None
        if full is None:
            cf, r2 = np.array([0., 0., -1., 0.]), 0.0
        else:
            cf, r2 = full
        if local is None:
            lcf, lr2 = cf, 0.0
        else:
            lcf, lr2 = local
        coefs[wid] = (cf, lcf)
        recs.append({"well_id": wid, "g_plane_full_r2": r2,
                     "g_plane_cz": float(cf[2]),
                     "g_plane_choriz": float(np.hypot(cf[0], cf[1])),
                     "g_plane_local_r2": lr2})
    return pd.DataFrame(recs), coefs


def _plane_per_row(df: pd.DataFrame, coefs: dict) -> pd.DataFrame:
    df = df.copy()
    x = df["X"].to_numpy(float); y = df["Y"].to_numpy(float); z = df["Z"].to_numpy(float)
    lkt = df["last_known_TVT"].to_numpy(float)
    full_pred = np.full(len(df), np.nan)
    local_pred = np.full(len(df), np.nan)
    wid_arr = df["well_id"].to_numpy()
    # group indices
    for wid, idx in df.groupby("well_id", sort=False).groups.items():
        if wid not in coefs:
            continue
        cf, lcf = coefs[wid]
        ii = df.index.get_indexer(idx)
        full_pred[ii] = cf[0]*x[ii] + cf[1]*y[ii] + cf[2]*z[ii] + cf[3]
        local_pred[ii] = lcf[0]*x[ii] + lcf[1]*y[ii] + lcf[2]*z[ii] + lcf[3]
    df["g_plane_full_delta"] = np.clip(full_pred - lkt, -PLANE_CLIP, PLANE_CLIP)
    df["g_plane_local_delta"] = np.clip(local_pred - lkt, -PLANE_CLIP, PLANE_CLIP)
    df["g_plane_disagree"] = (df["g_plane_full_delta"] - df["f_extrap_quad_dMD"].astype(float)).abs()
    return df


def enrich(df: pd.DataFrame, gmean: float) -> pd.DataFrame:
    df = exp014.enrich(df, gmean)            # Group A/B/C/D/F
    pw, coefs = _plane_per_well(df)
    df = df.merge(pw, on="well_id", how="left")
    df = _plane_per_row(df, coefs)
    return df


def main() -> None:
    print(f"[{EXP_ID}] Group G 3D構造平面外挿を追加して LightGBM を学習します。")
    ensure_exp_dir(EXP_DIR)
    train, test, folds, sample_submission = load_base_inputs(
        Path("data/processed/train_base_v001.parquet"),
        Path("data/processed/test_base_v001.parquet"),
        Path("data/folds/folds_group_well_v001.csv"),
        Path("data/raw/sample_submission.csv"),
    )
    gmean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    print(f"Global GR mean (train): {gmean:.4f}")
    print("特徴量エンジニアリング: train ..."); train = enrich(train, gmean)
    print("特徴量エンジニアリング: test  ..."); test  = enrich(test,  gmean)
    train = attach_folds(train, folds)

    missing = [c for c in ALL_FEATURES if c not in train.columns]
    if missing: raise ValueError(f"欠落特徴量: {missing}")

    train_t = target_rows(train); test_t = target_rows(test)
    y = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)
    oof_p = np.zeros(len(train_t)); test_p = np.zeros(len(test_t))
    fold_rows = []; importances = []
    params = {"objective":"regression","metric":"rmse","learning_rate":0.05,
              "num_leaves":63,"max_depth":-1,"min_data_in_leaf":50,
              "feature_fraction":0.9,"bagging_fraction":0.9,"bagging_freq":1,
              "lambda_l2":1.0,"verbosity":-1,"seed":42,"num_threads":8}
    n_folds = int(train_t["fold"].nunique())
    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy(); tm = ~vm
        print(f"fold {fold}: train={tm.sum()}, valid={vm.sum()}")
        model = lgb.LGBMRegressor(**params, n_estimators=1500)
        model.fit(train_t.loc[tm, ALL_FEATURES], y.loc[tm],
                  eval_set=[(train_t.loc[vm, ALL_FEATURES], y.loc[vm])],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
        bi = int(model.best_iteration_ or model.n_estimators)
        vd = model.predict(train_t.loc[vm, ALL_FEATURES], num_iteration=bi)
        oof_p[vm] = vd
        ftvt = train_t.loc[vm,"last_known_TVT"].to_numpy(float) + vd
        fr = tvt_rmse(train_t.loc[vm,"TVT"], ftvt)
        fold_rows.append({"fold":int(fold),"n_rows":int(vm.sum()),"rmse":fr,"best_iteration":bi})
        print(f"  fold {fold}  RMSE={fr:.6f}  best_iter={bi}")
        test_p += model.predict(test_t[ALL_FEATURES], num_iteration=bi)/n_folds
        importances.append(pd.DataFrame({"fold":int(fold),"feature":ALL_FEATURES,
            "importance_gain":model.booster_.feature_importance("gain"),
            "importance_split":model.booster_.feature_importance("split")}))

    oof = train_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof["pred_delta"]=oof_p; oof[PRED_COL]=oof["last_known_TVT"].astype(float)+oof["pred_delta"]
    oof["error"]=oof[PRED_COL]-oof["TVT"]; oof["abs_error"]=oof["error"].abs()
    oof.to_csv(EXP_DIR/"oof.csv", index=False)
    cv = pd.DataFrame(fold_rows); cv.to_csv(EXP_DIR/"cv.csv", index=False)

    test = test.copy()
    test.loc[test["is_target"].astype(bool),"pred_delta"]=test_p
    test.loc[test["is_target"].astype(bool),PRED_COL]=(test_t["last_known_TVT"].to_numpy(float)+test_p)
    submission = build_submission(test, sample_submission, PRED_COL)
    submission.to_csv(EXP_DIR/"submission.csv", index=False)

    fi = pd.concat(importances, ignore_index=True)
    fi_s = (fi.groupby("feature",as_index=False)[["importance_gain","importance_split"]]
              .mean().sort_values("importance_gain",ascending=False))
    fi_s.to_csv(EXP_DIR/"feature_importance.csv", index=False)

    overall = tvt_rmse(oof["TVT"], oof[PRED_COL])
    result = {"exp_id":EXP_ID,"created_at":now_jst(),"status":"completed",
        "model":"LightGBMRegressor","target":"TVT - last_known_TVT","metric":"TVT_abs_RMSE",
        "cv_rmse":overall,"cv_mean":float(cv["rmse"].mean()),"cv_std":float(cv["rmse"].std(ddof=0)),
        "baseline_exp014_cv":13.525189,"improvement_vs_exp014":round(13.525189-overall,6),
        "baseline_exp008_cv":13.808621,"improvement_vs_exp008":round(13.808621-overall,6),
        "group_g_well":GROUP_G_WELL,"group_g_row":GROUP_G_ROW,
        "n_oof_rows":int(len(oof)),"n_submission_rows":int(len(submission)),"leak_risk":"low",
        "notes":("Group G: 3D structural plane TVT~X,Y,Z fit on known region, applied to known "
                 "hidden coords. Centered/scaled for stability, delta clipped ±200. No hidden TVT.")}
    write_json(EXP_DIR/"result.json", result)

    fold_tbl = "\n".join(f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |" for r in fold_rows)
    g_rank = "\n".join(
        f"| {r.feature} | {int(r.importance_gain):,} | #{list(fi_s['feature']).index(r.feature)+1} |"
        for r in fi_s[fi_s['feature'].isin(GROUP_G)].itertuples())
    notes = f"""# {EXP_ID}

## 目的
exp014 (CV=13.525189) に Group G（3D構造平面外挿）を追加。
TVT≈傾斜地層の平面 → known区間で TVT~X,Y,Z を較正し、既知hidden座標へ適用。
Group F の縦成分(dTVT/dZ)を横方向dip込みに一般化。

## 結果
| metric | 値 |
|---|---|
| **exp016 CV RMSE** | **{overall:.6f}** |
| exp014 baseline | 13.525189 |
| vs exp014 | {13.525189-overall:+.6f} |
| vs exp008 | {13.808621-overall:+.6f} |

## Fold 別
| fold | rmse | best_iter |
|---|---:|---:|
{fold_tbl}

## Group G のランク
| feature | gain | 全体順位 |
|---|---:|---:|
{g_rank}

## リーク防止
- 平面較正は is_known_tvt==True 行のみ ✅ / hidden X/Y/Z は観測量 ✅ / hidden TVT不使用 ✅
- fold は exp008/014 と同一 ✅ / delta は ±200 clip ✅
"""
    (EXP_DIR/"notes.md").write_text(notes, encoding="utf-8")
    print(f"\n[{EXP_ID}] 完了  CV={overall:.6f}  vs exp014 {13.525189-overall:+.6f}  vs exp008 {13.808621-overall:+.6f}")


if __name__ == "__main__":
    main()
