#!/usr/bin/env python3
"""diag: GRが選ぶoffsetは真のoffsetと相関するか? = GR offset信号の存否(決定的判定)。

per-well: c_real(GR絶対コスト最小,band8) / c_real_corr(GR相関最大,band15) /
          c_oracle(真値RMSE最小) / tw_corr(known区間GR一致度)。
報告: corr(c_real, c_oracle) 全体 & tw_corrビン別。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

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
    rows=[]
    for wid,g in tr[tr.is_target.astype(bool)|tr.is_known_tvt.astype(bool)].groupby("well_id",sort=False):
        g=g.sort_values("row_idx")
        known=g[g.is_known_tvt.astype(bool)]; tgt=g[g.is_target.astype(bool)]
        if len(tgt)==0: continue
        anchor=float(tgt.last_known_TVT.iloc[0])
        ri=tgt.row_idx.to_numpy()
        geomp=np.array([gm.get((wid,r),anchor) for r in ri],float)
        true=tgt.TVT.to_numpy(float)
        tw_g=tw_by.get(wid)
        if tw_g is None or len(tw_g)<2: continue
        tw_s=tw_g.sort_values("TVT").drop_duplicates("TVT")
        tvt_f=tw_s.TVT.to_numpy(float);gr_f=tw_s.GR.to_numpy(float)
        kv=known[~known.is_gr_missing.astype(bool)]
        if len(kv)>=10:
            twk=np.interp(kv.TVT_input.to_numpy(float),tvt_f,gr_f)
            A=np.vstack([twk,np.ones_like(twk)]).T
            coef,*_=np.linalg.lstsq(A,kv.GR.to_numpy(float),rcond=None)
            a,b=float(coef[0]),float(coef[1])
            if abs(a)<1e-3:a,b=1.0,0.0
            tw_corr=float(np.corrcoef(kv.GR.to_numpy(float),twk)[0,1]) if np.std(twk)>1e-6 else np.nan
        else: a,b,tw_corr=1.0,0.0,np.nan
        obs=tgt.GR.to_numpy(float); has=~tgt.is_gr_missing.astype(bool).to_numpy()
        if has.sum()<8: continue
        tw_equiv=(obs-b)/a
        oi=np.where(has)[0]; o=tw_equiv[oi]; gp=geomp[oi]
        cs=np.arange(-30,30+0.5,0.5)
        # abs cost
        abc=[np.mean(np.abs(o-np.interp(gp+c,tvt_f,gr_f))) for c in cs]
        c_abs=cs[int(np.argmin(abc))]
        # corr cost
        cor=[]
        for c in cs:
            pg=np.interp(gp+c,tvt_f,gr_f)
            cor.append(-(np.corrcoef(o,pg)[0,1] if np.std(pg)>1e-6 else 0))
        c_cor=cs[int(np.argmin(cor))]
        # oracle
        te=[np.sqrt(np.mean((geomp+c-true)**2)) for c in cs]
        c_or=cs[int(np.argmin(te))]
        rows.append({"well_id":wid,"tw_corr":tw_corr,"c_abs":c_abs,"c_cor":c_cor,
                     "c_oracle":c_or,"hidden_n":len(tgt)})
    D=pd.DataFrame(rows)
    print(f"wells={len(D)}")
    def rep(name,col):
        d=D.dropna(subset=[col,"c_oracle"])
        print(f"  corr({name:6s}, oracle) all = {d[col].corr(d.c_oracle):+.3f}")
    print("=== GR選択offset vs 真offset 相関 (符号付き, 高いほどGR有効) ===")
    rep("c_abs","c_abs"); rep("c_cor","c_cor")
    D["tw_bin"]=pd.cut(D.tw_corr,[-2,0.7,0.85,1.01],labels=["<0.7","0.7-0.85",">0.85"])
    print("\n=== tw_corrビン別 corr(c_abs, oracle) / well数 ===")
    for bn,gg in D.groupby("tw_bin",observed=True):
        gg=gg.dropna(subset=["c_abs","c_oracle"])
        print(f"  {str(bn):10s} n={len(gg):4d}  corr={gg.c_abs.corr(gg.c_oracle):+.3f}")
    print(f"\n  真offset std={D.c_oracle.std():.2f}  GR(abs)offset std={D.c_abs.std():.2f}")
    D.to_csv("experiments/diag_gr_ceiling/offset_signal_per_well.csv",index=False)

if __name__=="__main__":
    main()
