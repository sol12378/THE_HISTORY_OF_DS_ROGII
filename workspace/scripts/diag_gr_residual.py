#!/usr/bin/env python3
"""diag: 真のTVTにおけるGR残差を測定し、GR_TOL依存でない上限を出す。

問い: typewell GR profile は、真のTVTにおいて lateral の観測GR(較正済)を
      どれだけ正確に再現するか? = GR照合の原理的ノイズ床。
      残差が小さければGR照合は原理上有効、大きければ不可能。

さらに oracle を「適応tol(known残差stdのk倍)」で再測定し、tol=6 が
上限を不当に低く見せていないか確認する。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse

def main():
    tr = pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id","row_idx","GR","TVT","TVT_input","is_target",
                 "is_known_tvt","is_gr_missing","last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
        columns=["well_id","TVT","GR"])
    tw_by = {w:g for w,g in tw_all.groupby("well_id", sort=False)}

    known_res_std=[]; truth_res=[]; well_truthres=[]
    for wid, g in tr[tr.is_target.astype(bool)|tr.is_known_tvt.astype(bool)].groupby("well_id", sort=False):
        known = g[g.is_known_tvt.astype(bool)]
        tgt = g[g.is_target.astype(bool)]
        tw_g = tw_by.get(wid)
        if tw_g is None or len(tw_g)<2 or len(tgt)==0: continue
        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tvt_f=tw_s.TVT.to_numpy(float); gr_f=tw_s.GR.to_numpy(float)
        kv = known[~known.is_gr_missing.astype(bool)]
        if len(kv)<10: continue
        tw_at_known=np.interp(kv.TVT_input.to_numpy(float),tvt_f,gr_f)
        obs_k=kv.GR.to_numpy(float)
        A=np.vstack([tw_at_known,np.ones_like(tw_at_known)]).T
        coef,*_=np.linalg.lstsq(A,obs_k,rcond=None)
        a,b=float(coef[0]),float(coef[1])
        if abs(a)<1e-3: a,b=1.0,0.0
        # known残差 (較正後 obs vs tw)
        kres = obs_k - (a*tw_at_known+b)
        known_res_std.append(np.std(kres))
        # target: 真TVTでのtw_GR vs 観測GR(較正後 tw scaleへ)
        tv=tgt[~tgt.is_gr_missing.astype(bool)]
        if len(tv)==0: continue
        obs=tv.GR.to_numpy(float)
        tw_at_true=np.interp(tv.TVT.to_numpy(float),tvt_f,gr_f)
        tw_equiv=(obs-b)/a
        r=tw_equiv-tw_at_true
        truth_res.append(np.abs(r))
        well_truthres.append({"well_id":wid,"med_abs_truth_res":float(np.median(np.abs(r))),
                              "known_res_std":float(np.std(kres))})

    tr_all=np.concatenate(truth_res)
    ks=np.array(known_res_std)
    print("=== 真TVTでのGR残差 |tw_equiv_obs - tw_GR(true)| (typewellスケール) ===")
    for q in [10,25,50,75,90]:
        print(f"  P{q:2d} = {np.percentile(tr_all,q):7.3f}")
    print(f"  mean= {tr_all.mean():7.3f}")
    print("\n=== known区間の較正後残差std (well毎) ===")
    print(f"  median well known_res_std = {np.median(ks):.3f}")
    print(f"  P75 = {np.percentile(ks,75):.3f}  P90 = {np.percentile(ks,90):.3f}")
    print("\n解釈: 真TVTでのGR残差中央値が large(>known残差std) なら、typewell GRは")
    print("真TVTにおいてlateral GRを再現できず、GR照合は原理的に不可能。")
    wt=pd.DataFrame(well_truthres)
    wt.to_csv("experiments/diag_gr_ceiling/truth_residual_per_well.csv",index=False)
    # 真残差 vs known残差: GRが効くなら同等のはず
    ratio = wt.med_abs_truth_res/(wt.known_res_std+1e-9)
    print(f"\n  well毎 median(真残差/known残差std) = {np.median(ratio):.3f}")
    print("  (1付近ならGRは真TVTで整合=有効。>>1なら多価/不整合で照合不能)")

if __name__=="__main__":
    main()
