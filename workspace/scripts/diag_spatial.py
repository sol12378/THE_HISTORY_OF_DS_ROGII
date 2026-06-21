#!/usr/bin/env python3
"""diag: 多井戸 空間構造モデル TVT~f(X,Y,Z) は hidden TVT を当てられるか?

仮説: per-well offset(誤差の正体)は空間的・構造的量。全wellのknown区間を結合した
空間モデルなら、隣接井戸の構造が hidden区間のoffsetを拘束できる(これまで未着手)。
corr(TVT,Z)=-0.96 と強い空間構造あり。

検証(全て leak-free: hidden TVT不使用):
  A_strict : val well 以外のknownで学習→val hidden予測 (隣接井戸だけで当たるか)
  B_realistic: 全well known(val自身のknown含む)で学習→val hidden予測 (test条件)
  C_z_only : TVT~Z 単回帰(global)
比較基準: anchor=15.91, exp014=13.53, 現best=13.33。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse

def main():
    d=pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id","row_idx","X","Y","Z","MD","TVT","is_known_tvt","is_target","last_known_TVT"])
    folds=pd.read_csv("data/folds/folds_group_well_v001.csv")
    fmap=folds[folds.split=="train"].set_index("well_id")["fold"]
    d["fold"]=d["well_id"].map(fmap)
    known=d[d.is_known_tvt.astype(bool)].copy()
    tgt=d[d.is_target.astype(bool)].copy()
    feats=["X","Y","Z","MD"]
    p={"objective":"regression","metric":"rmse","learning_rate":0.05,"num_leaves":255,
       "min_data_in_leaf":100,"feature_fraction":0.9,"bagging_fraction":0.8,"bagging_freq":1,
       "verbosity":-1,"seed":42,"num_threads":8}

    # subsample known for speed
    known_s=known.sample(frac=0.25,random_state=42)

    # C: TVT~Z global linear
    a,b=np.polyfit(known.Z, known.TVT,1)
    predC=a*tgt.Z+b
    rmseC=tvt_rmse(tgt.TVT,predC)

    # A_strict: well-grouped, train on other wells' known
    oofA=np.zeros(len(tgt)); tgt_fold=tgt.fold.to_numpy()
    for f in sorted(tgt.fold.dropna().unique()):
        tr=known_s[known_s.fold!=f]
        m=lgb.LGBMRegressor(**p,n_estimators=600)
        m.fit(tr[feats],tr.TVT)
        vm=tgt_fold==f
        oofA[vm]=m.predict(tgt.loc[vm,feats])
    rmseA=tvt_rmse(tgt.TVT,oofA)

    # B_realistic: train on ALL wells' known (incl val's known), predict hidden
    mB=lgb.LGBMRegressor(**p,n_estimators=800)
    mB.fit(known_s[feats],known_s.TVT)
    predB=mB.predict(tgt[feats])
    rmseB=tvt_rmse(tgt.TVT,predB)

    # B + per-well anchor recentering: shift each well's prediction so known-section bias=0
    # (use known rows to estimate model bias per well, leak-free)
    kpredB=mB.predict(known[feats])
    known["_biasB"]=kpredB-known.TVT
    wbias=known.groupby("well_id")["_biasB"].mean()
    tgt["_biasB"]=tgt.well_id.map(wbias).fillna(0.0)
    predB_rc=predB-tgt["_biasB"].to_numpy()
    rmseB_rc=tvt_rmse(tgt.TVT,predB_rc)

    print("=== 多井戸空間モデル TVT~f(X,Y,Z,MD) (hidden行RMSE) ===")
    print(f"  anchor基準           = 15.91")
    print(f"  exp014 (per-well幾何) = 13.53")
    print(f"  現best (blend)        = 13.33")
    print(f"  --- 空間モデル ---")
    print(f"  C: TVT~Z global linear        = {rmseC:.4f}")
    print(f"  A_strict (隣接井戸のみ)        = {rmseA:.4f}")
    print(f"  B_realistic (全known)         = {rmseB:.4f}")
    print(f"  B + per-well known bias補正    = {rmseB_rc:.4f}")
    print("\n  解釈: B系がexp014(13.53)を下回れば空間モデルが本命。")
    print("        per-well bias補正は『known区間で空間モデルのズレを測りhiddenを補正』=offset回収の試み。")

if __name__=="__main__":
    main()
