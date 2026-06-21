#!/usr/bin/env python3
"""diag: 系列GR照合(sequence alignment)が geom prior を改善するか。

着想: GR点単位ノイズは大(std~12.8)だが、hidden区間のGR(MD)系列全体を
typewell GR(TVT)プロファイルに整合させればノイズが平均化され、TVT offset を
高精度に決められる(=井戸間対比/log correlation の標準手法)。

geom(exp014)が TVT(MD) の形状を与える。GR整合で滑らかな offset 補正を加える。

テスト段階:
 (1) global offset: well毎に単一定数c。realistic(GRで決める) vs oracle(真値で決める)。
 (2) segment offset: hiddenをKセグメントに分け各々でc。realistic vs oracle。
全て leak-free(realisticは真値不使用)。oracleは上限測定用。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse

BANDS=[8.0,15.0,25.0,40.0]; STEP=0.5; N_SEG=4; LAM=0.02

def main():
    tr = pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id","row_idx","GR","TVT","TVT_input","is_target",
                 "is_known_tvt","is_gr_missing","last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
        columns=["well_id","TVT","GR"])
    tw_by={w:g for w,g in tw_all.groupby("well_id",sort=False)}
    geom=pd.read_csv("experiments/exp014_geom_extrap/oof.csv",
        usecols=["well_id","row_idx","pred_tvt"]).rename(columns={"pred_tvt":"geom"})
    gm={(r.well_id,r.row_idx):r.geom for r in geom.itertuples()}

    parts=[]; orac_glob_off=[]
    for wid,g in tr[tr.is_target.astype(bool)|tr.is_known_tvt.astype(bool)].groupby("well_id",sort=False):
        g=g.sort_values("row_idx")
        known=g[g.is_known_tvt.astype(bool)]; tgt=g[g.is_target.astype(bool)].copy()
        if len(tgt)==0: continue
        anchor=float(tgt.last_known_TVT.iloc[0]); n=len(tgt)
        ri=tgt.row_idx.to_numpy()
        geomp=np.array([gm.get((wid,r),anchor) for r in ri],float)
        true=tgt.TVT.to_numpy(float)
        out={"well_id":np.full(n,wid),"row_idx":ri,"TVT":true,"geom":geomp,"anchor":np.full(n,anchor)}
        tw_g=tw_by.get(wid)
        if tw_g is None or len(tw_g)<2:
            out["glob_real"]=geomp;out["glob_oracle"]=geomp
            out["seg_real"]=geomp;out["seg_oracle"]=geomp
            parts.append(pd.DataFrame(out));continue
        tw_s=tw_g.sort_values("TVT").drop_duplicates("TVT")
        tvt_f=tw_s.TVT.to_numpy(float);gr_f=tw_s.GR.to_numpy(float)
        kv=known[~known.is_gr_missing.astype(bool)]
        if len(kv)>=10:
            twk=np.interp(kv.TVT_input.to_numpy(float),tvt_f,gr_f)
            A=np.vstack([twk,np.ones_like(twk)]).T
            coef,*_=np.linalg.lstsq(A,kv.GR.to_numpy(float),rcond=None)
            a,b=float(coef[0]),float(coef[1])
            if abs(a)<1e-3:a,b=1.0,0.0
        else: a,b=1.0,0.0
        obs=tgt.GR.to_numpy(float); has=~tgt.is_gr_missing.astype(bool).to_numpy()
        tw_equiv=(obs-b)/a

        def abscost(idx,c):
            ii=idx[has[idx]]
            if len(ii)<8: return np.inf
            pred_gr=np.interp(geomp[ii]+c,tvt_f,gr_f)
            return float(np.mean(np.abs(tw_equiv[ii]-pred_gr)))+LAM*c*c
        # oracle global offset (1 param)
        cs_wide=np.arange(-60,60+STEP,STEP)
        terr=np.array([np.sqrt(np.mean((geomp+c-true)**2)) for c in cs_wide])
        c_or=cs_wide[np.argmin(terr)]; orac_glob_off.append(abs(c_or))
        out["glob_oracle"]=geomp+c_or
        # realistic per band
        allidx=np.arange(n)
        for B in BANDS:
            cs=np.arange(-B,B+STEP,STEP)
            cc=np.array([abscost(allidx,c) for c in cs])
            cr=cs[np.argmin(cc)] if np.isfinite(cc).any() else 0.0
            out[f"glob_real_b{int(B)}"]=geomp+cr
        # segment oracle (4 param) + realistic segment (band=15)
        seg_o=geomp.copy(); seg_r=geomp.copy()
        bounds=np.linspace(0,n,N_SEG+1).astype(int)
        csB=np.arange(-15,15+STEP,STEP)
        for s in range(N_SEG):
            idx=np.arange(bounds[s],bounds[s+1])
            if len(idx)==0: continue
            te=np.array([np.sqrt(np.mean((geomp[idx]+c-true[idx])**2)) for c in cs_wide])
            seg_o[idx]=geomp[idx]+cs_wide[np.argmin(te)]
            cc=np.array([abscost(idx,c) for c in csB])
            cr=csB[np.argmin(cc)] if np.isfinite(cc).any() else 0.0
            seg_r[idx]=geomp[idx]+cr
        out["seg_oracle"]=seg_o; out["seg_real_b15"]=seg_r
        parts.append(pd.DataFrame(out))
    P=pd.concat(parts,ignore_index=True)
    print("=== sequence alignment (全target行 RMSE) ===")
    cols=["anchor","geom","glob_oracle","seg_oracle","seg_real_b15"]+\
         [f"glob_real_b{int(B)}" for B in BANDS]
    for k in cols:
        print(f"  {k:18s} = {tvt_rmse(P.TVT,P[k]):.4f}")
    print(f"\n  oracle global offset |c|: median={np.median(orac_glob_off):.2f} "
          f"P75={np.percentile(orac_glob_off,75):.2f} P90={np.percentile(orac_glob_off,90):.2f}")
    P.to_csv("experiments/diag_gr_ceiling/seq_align_rows.csv",index=False)

if __name__=="__main__":
    main()
