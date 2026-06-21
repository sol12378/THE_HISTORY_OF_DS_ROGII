#!/usr/bin/env python3
"""diag: known区間の自己ホールドアウトで「geom外挿のbias」を推定し、
真の hidden offset を予測できるか? (leak-free offset信号の探索)。

着想([[gr-offset-ceiling]]): geom形状は正しく残差は低周波offset。
GRはoffsetを当てられない(corr0.155)。代わりに known区間内で同じ外挿を
擬似実行しbiasを測れば、hidden offsetの予測子になるかもしれない(完全leak-free)。

各well: known を split(既定70%)で分け、前半で last20/last50 slope+curv を較正、
後半known(擬似hidden)へ二次外挿。pseudo_offset = mean(true - pred)。
これを真 hidden offset(c_oracle, diag_offset_signalより)と相関。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

def extrap_bias(md, tv, split_frac):
    n=len(md)
    s=int(n*split_frac)
    if s<25 or n-s<10: return None
    md0,tv0=md[:s],tv[:s]; md1,tv1=md[s:],tv[s:]
    n20=min(20,s); n50=min(50,s)
    d20=md0[-1]-md0[-n20]; d50=md0[-1]-md0[-n50]
    if abs(d20)<1e-6: return None
    s20=(tv0[-1]-tv0[-n20])/d20
    s5n=min(5,s); d5=md0[-1]-md0[-s5n]
    s5=(tv0[-1]-tv0[-s5n])/d5 if abs(d5)>1e-6 else s20
    curv=s5-s20
    base_tv=tv0[-1]; base_md=md0[-1]
    dmd=md1-base_md
    pred=base_tv + s20*dmd + 0.5*curv*dmd*dmd
    return float(np.mean(tv1-pred)), float(np.mean(np.abs(tv1-pred)))

def main():
    tr=pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id","row_idx","MD","TVT","TVT_input","is_target","is_known_tvt","last_known_TVT"])
    orc=pd.read_csv("experiments/diag_gr_ceiling/offset_signal_per_well.csv",
        usecols=["well_id","c_oracle"])
    oc={r.well_id:r.c_oracle for r in orc.itertuples()}
    rows=[]
    for wid,g in tr[tr.is_known_tvt.astype(bool)|tr.is_target.astype(bool)].groupby("well_id",sort=False):
        g=g.sort_values("row_idx")
        kn=g[g.is_known_tvt.astype(bool)]
        if len(kn)<40 or wid not in oc: continue
        md=kn.MD.to_numpy(float); tv=kn.TVT_input.to_numpy(float)
        recs={}
        for sf in [0.6,0.7,0.8]:
            r=extrap_bias(md,tv,sf)
            if r: recs[f"bias{int(sf*100)}"]=r[0]
        if not recs: continue
        recs["well_id"]=wid; recs["c_oracle"]=oc[wid]; recs["known_n"]=len(kn)
        rows.append(recs)
    D=pd.DataFrame(rows)
    print(f"wells with self-calib = {len(D)}")
    for c in ["bias60","bias70","bias80"]:
        if c in D:
            d=D.dropna(subset=[c,"c_oracle"])
            print(f"  corr({c}, true hidden offset) = {d[c].corr(d.c_oracle):+.3f}  (n={len(d)})")
    # combined (mean of available biases)
    bc=[c for c in ["bias60","bias70","bias80"] if c in D]
    D["bias_mean"]=D[bc].mean(axis=1)
    d=D.dropna(subset=["bias_mean","c_oracle"])
    print(f"  corr(bias_mean, true offset) = {d.bias_mean.corr(d.c_oracle):+.3f}")
    print(f"\n  比較: GR offset信号 corr=0.155。self-calibが上回れば leak-free offset補正の芽。")
    D.to_csv("experiments/diag_gr_ceiling/self_calib_per_well.csv",index=False)

if __name__=="__main__":
    main()
