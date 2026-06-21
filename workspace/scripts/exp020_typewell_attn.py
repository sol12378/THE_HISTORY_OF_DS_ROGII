#!/usr/bin/env python3
"""exp020: typewell-aware cross-attention 系列モデル (offset回収の本命)。

これまでの全手法が捉えられなかった per-well offset を、唯一未着手の paradigm =
「lateral GR系列 ↔ typewell GR-TVTプロファイルの微分可能な対応付け」で取りにいく。
exp019のNNはtypewellを入力していなかった(盲点)。本実験はtypewellを cross-attention で投入。

機構:
- TCN が lateral(GR+幾何)系列を encode → 各行 context。
- typewell GR(TVT) を anchor±band の TVTグリッド(M点)へ resample、small convで encode。
- cross-attention: query=lateral行context, key/value=typewell位置。
  prior bias = -alpha*(tw_TVT_rel - geom_prior_delta)^2 で幾何prior近傍へ誘導(cycle-skip防止)。
  → GR形状が prior窓内で offset を微修正できるか学習。
- 出力 = geom_prior_delta + residual (GR無効ならresid->0でexp019に縮退)。

leak-free: hidden TVT不使用、calibrationはknownのみ、typewellは参照曲線、well-grouped fold。
最後に exp018/exp019 と blend。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from rogii.training.baselines import (PRED_COL, target_rows, tvt_rmse, write_json,
    now_jst, build_submission, ensure_exp_dir, attach_folds, load_base_inputs)
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(42); np.random.seed(42)

EXP_ID="exp020_typewell_attn"; EXP_DIR=Path("experiments")/EXP_ID
L=1024; M=192; BAND=80.0; SCALE=50.0; EPOCHS=45; BATCH=8; LR=2e-3; DEV=torch.device("cpu")


def add_slope20(df):
    recs={}; kn=df[df["is_known_tvt"].astype(bool)]
    for wid,g in kn.groupby("well_id",sort=False):
        g=g.sort_values("row_idx"); md=g["MD"].to_numpy(float); tv=g["TVT_input"].to_numpy(float)
        n=len(g); n20=min(20,n); d=md[-1]-md[-n20] if n20>1 else 1.
        recs[wid]=(tv[-1]-tv[-n20])/d if abs(d)>1e-6 else 0.0
    df["_slope20"]=df["well_id"].map(recs).fillna(0.0); return df


def build_arrays2(df, tw_by):
    g=df.sort_values(["well_id","row_idx"])
    Xl=[]; Y=[]; Msk=[]; Xt=[]; wid_order=[]
    grid=np.linspace(0,1,L); tw_rel_grid=np.linspace(-BAND,BAND,M)
    for wid,sub in g.groupby("well_id",sort=False):
        n=len(sub); pos=np.arange(n)/max(n-1,1)
        def rs(a): return np.interp(grid,pos,np.asarray(a,float))
        is_known=sub["is_known_tvt"].astype(float).to_numpy()
        is_tgt=sub["is_target"].astype(float).to_numpy()
        gr=sub["GR"].to_numpy(float); grmiss=sub["is_gr_missing"].astype(float).to_numpy()
        kmask=is_known.astype(bool)&(grmiss<0.5)
        gm,gs=(np.mean(gr[kmask]),np.std(gr[kmask])+1e-6) if kmask.sum()>=5 else (0.0,1.0)
        gr_norm=np.where(grmiss<0.5,(gr-gm)/gs,0.0)
        anchor=float(sub["last_known_TVT"].iloc[0])
        known_delta=np.where(is_known>0.5,(sub["TVT_input"].to_numpy(float)-anchor)/SCALE,0.0)
        dMD=sub["delta_MD_from_PS"].to_numpy(float)/5000.; dZ=sub["delta_Z_from_PS"].to_numpy(float)/50.
        dX=sub["delta_X_from_PS"].to_numpy(float)/5000.; dY=sub["delta_Y_from_PS"].to_numpy(float)/5000.
        z_rel=(sub["Z"].to_numpy(float)-float(sub["last_known_Z"].iloc[0]))/50.
        slope20=float(sub["_slope20"].iloc[0]); geom_delta=slope20*sub["delta_MD_from_PS"].to_numpy(float)
        lat=np.stack([rs(gr_norm),rs(grmiss),rs(is_known),rs(known_delta),
                      rs(dMD),rs(dZ),rs(dX),rs(dY),rs(z_rel),grid,rs(geom_delta)/SCALE],0).astype(np.float32)
        Xl.append(lat)
        tg=sub["TVT"].to_numpy(float); delta_true=(tg-anchor)/SCALE
        Y.append(rs(np.where(is_tgt>0.5,delta_true,0.0)).astype(np.float32))
        Msk.append((rs(is_tgt)>0.5).astype(np.float32))
        tw=tw_by.get(wid)
        if tw is not None and len(tw)>=2:
            tws=tw.sort_values("TVT").drop_duplicates("TVT")
            tvt_f=tws["TVT"].to_numpy(float); grf=tws["GR"].to_numpy(float)
            tw_tvt=anchor+tw_rel_grid; tw_gr=np.interp(tw_tvt,tvt_f,grf)
            valid=((tw_tvt>=tvt_f.min())&(tw_tvt<=tvt_f.max())).astype(np.float32)
            tw_gr_norm=np.where(valid>0,(tw_gr-gm)/gs,0.0)
        else:
            tw_gr_norm=np.zeros(M,np.float32); valid=np.zeros(M,np.float32)
        Xt.append(np.stack([tw_gr_norm,tw_rel_grid/SCALE,valid],0).astype(np.float32))
        wid_order.append(wid)
    return np.stack(Xl),np.stack(Y),np.stack(Msk),np.stack(Xt),wid_order,grid


class Model(nn.Module):
    def __init__(self,cin_l=11,cin_t=3,d=48):
        super().__init__()
        layers=[]; prev=cin_l
        for dil in (1,2,4,8,16,32):
            layers+=[nn.Conv1d(prev,d,3,padding=dil,dilation=dil),nn.BatchNorm1d(d),nn.GELU(),nn.Dropout(0.1)]
            prev=d
        self.tcn=nn.Sequential(*layers)
        self.tw_enc=nn.Sequential(nn.Conv1d(cin_t,d,3,padding=1),nn.GELU(),nn.Conv1d(d,d,3,padding=1),nn.GELU())
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d)
        self.alpha=nn.Parameter(torch.tensor(2.0))   # prior bias強度(learnable)
        self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Dropout(0.1),nn.Linear(d,1))
        self.d=d
    def forward(self,xl,xt):
        # xl:(B,Cl,L) xt:(B,Ct,M)
        B=xl.shape[0]
        H=self.tcn(xl).transpose(1,2)                 # (B,L,d)
        T=self.tw_enc(xt).transpose(1,2)              # (B,M,d)
        Q=self.q(H); K=self.k(T); V=self.v(T)         # (B,L,d),(B,M,d)
        scores=torch.bmm(Q,K.transpose(1,2))/ (self.d**0.5)   # (B,L,M)
        prior=xl[:,10,:]                              # geom_delta_norm (B,L)  (TVT_rel/SCALE prior)
        tw_rel=xt[:,1,:]                              # (B,M) tw_TVT_rel/SCALE
        valid=xt[:,2,:]                               # (B,M)
        bias=-F.softplus(self.alpha)*(tw_rel.unsqueeze(1)-prior.unsqueeze(2))**2   # (B,L,M)
        scores=scores+bias
        scores=scores.masked_fill(valid.unsqueeze(1)<0.5,-1e9)
        # wells with no valid tw: all -1e9 -> softmax uniform; guard
        no_tw=(valid.sum(1)<1).view(B,1,1)
        attn=torch.softmax(scores,dim=2)
        ctx=torch.bmm(attn,V)                         # (B,L,d)
        ctx=torch.where(no_tw, torch.zeros_like(ctx), ctx)
        resid=self.head(torch.cat([H,ctx],dim=2)).squeeze(-1)  # (B,L)
        return prior+resid


def main():
    ensure_exp_dir(EXP_DIR)
    print(f"[{EXP_ID}] typewell-aware cross-attention model")
    train,test,folds,sample=load_base_inputs(
        ROOT/"data/processed/train_base_v001.parquet",ROOT/"data/processed/test_base_v001.parquet",
        ROOT/"data/folds/folds_group_well_v001.csv",ROOT/"data/raw/sample_submission.csv")
    tw_tr=pd.read_parquet("data/processed/typewell_train_base_v001.parquet",columns=["well_id","TVT","GR"])
    tw_te=pd.read_parquet("data/processed/typewell_test_base_v001.parquet",columns=["well_id","TVT","GR"])
    tw_by_tr={w:g for w,g in tw_tr.groupby("well_id",sort=False)}
    tw_by_te={w:g for w,g in tw_te.groupby("well_id",sort=False)}
    train=add_slope20(train); test=add_slope20(test); train=attach_folds(train,folds)
    print("build train ..."); Xl,Y,Mk,Xt,wtr,grid=build_arrays2(train,tw_by_tr)
    print("build test  ..."); Xl2,_,_,Xt2,wte,_=build_arrays2(test,tw_by_te)
    wfold=train.groupby("well_id",sort=False)["fold"].first(); fold_arr=np.array([wfold[w] for w in wtr])
    Xl=torch.tensor(Xl);Y=torch.tensor(Y);Mk=torch.tensor(Mk);Xt=torch.tensor(Xt)
    Xl2=torch.tensor(Xl2);Xt2=torch.tensor(Xt2)

    tr_t=target_rows(train).copy(); te_t=target_rows(test).copy()
    def backinterp(frame,grid_by_wid):
        out=np.zeros(len(frame)); frame=frame.copy(); frame["_o"]=np.arange(len(frame))
        for wid,sub in frame.groupby("well_id",sort=False):
            sub=sub.sort_values("row_idx"); nrw=float(sub["n_rows_in_well"].iloc[0])
            pos=sub["row_idx"].to_numpy(float)/max(nrw-1,1)
            out[sub["_o"].to_numpy()]=np.interp(pos,grid,grid_by_wid[wid])
        return out

    oof_grid={}; test_acc=np.zeros((len(wte),L)); nf=len(np.unique(fold_arr))
    for f in sorted(np.unique(fold_arr)):
        vm=fold_arr==f; tm=~vm; idx_tr=np.where(tm)[0]
        model=Model().to(DEV); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
        best=1e9; best_state=None; pat=0
        for ep in range(EPOCHS):
            model.train(); np.random.shuffle(idx_tr)
            for i in range(0,len(idx_tr),BATCH):
                bi=idx_tr[i:i+BATCH]
                pred=model(Xl[bi].to(DEV),Xt[bi].to(DEV)); yb=Y[bi].to(DEV); mb=Mk[bi].to(DEV)
                loss=((pred-yb)**2*mb).sum()/(mb.sum()+1e-6)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            sched.step(); model.eval()
            with torch.no_grad():
                pv=model(Xl[vm].to(DEV),Xt[vm].to(DEV)); yb=Y[vm].to(DEV); mb=Mk[vm].to(DEV)
                vr=float(torch.sqrt(((pv-yb)**2*mb).sum()/(mb.sum()+1e-6)))*SCALE
            if vr<best-1e-4: best=vr; best_state={k:v.clone() for k,v in model.state_dict().items()}; pat=0
            else:
                pat+=1
                if pat>=8: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            vp=model(Xl[vm].to(DEV),Xt[vm].to(DEV)).cpu().numpy()*SCALE
            tp=model(Xl2.to(DEV),Xt2.to(DEV)).cpu().numpy()*SCALE
        for j,wi in enumerate(np.where(vm)[0]): oof_grid[wtr[wi]]=vp[j]
        test_acc+=tp/nf
        print(f"  fold{f} val RMSE(grid)={best:.4f}  alpha={float(F.softplus(model.alpha)):.3f}")
    test_grid={wte[j]:test_acc[j] for j in range(len(wte))}
    anchor_tr=tr_t["last_known_TVT"].to_numpy(float); anchor_te=te_t["last_known_TVT"].to_numpy(float)
    nn_oof=backinterp(tr_t,oof_grid); nn_test=backinterp(te_t,test_grid)
    nn_cv=tvt_rmse(tr_t["TVT"],anchor_tr+nn_oof)
    print(f"\n  exp020 NN OOF CV = {nn_cv:.6f}")

    # blend with exp018 + exp019
    e18=pd.read_csv("experiments/exp018_model_blend/oof.csv",usecols=["well_id","row_idx","pred_delta"]).rename(columns={"pred_delta":"e18"})
    e19=pd.read_csv("experiments/exp019_seq_nn/oof.csv",usecols=["well_id","row_idx","pred_delta"]).rename(columns={"pred_delta":"e19"})
    tr_t=tr_t.merge(e18,on=["well_id","row_idx"],how="left").merge(e19,on=["well_id","row_idx"],how="left")
    true_d=(tr_t["TVT"].astype(float)-anchor_tr).to_numpy()
    from scipy.optimize import minimize
    O=np.vstack([tr_t["e18"].to_numpy(float),tr_t["e19"].to_numpy(float),nn_oof]).T
    loss=lambda w: float(np.sqrt(np.mean((true_d-O@w)**2)))
    res=minimize(loss,[0.6,0.2,0.2],method="SLSQP",bounds=[(0,1)]*3,constraints=({"type":"eq","fun":lambda w:w.sum()-1},))
    w=res.x; blend=O@w; cv_blend=tvt_rmse(tr_t["TVT"],anchor_tr+blend)
    cv_e18=tvt_rmse(tr_t["TVT"],anchor_tr+tr_t["e18"].to_numpy(float))
    err020=anchor_tr+nn_oof-tr_t["TVT"].to_numpy(float); err18=anchor_tr+tr_t["e18"].to_numpy(float)-tr_t["TVT"].to_numpy(float)
    ecorr=float(np.corrcoef(err020,err18)[0,1])

    oof_df=tr_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof_df["pred_delta"]=blend; oof_df[PRED_COL]=anchor_tr+blend; oof_df["pred_nn020"]=anchor_tr+nn_oof
    oof_df.to_csv(EXP_DIR/"oof.csv",index=False)
    # test blend (reconstruct e18/e19 test deltas)
    def load_test_delta(path):
        s=pd.read_csv(path); mp=dict(zip(s["id"],s["tvt"]));
        return te_t["id"].map(mp).to_numpy(float)-anchor_te
    e18t=load_test_delta("experiments/exp018_model_blend/submission.csv")
    e19t=load_test_delta("experiments/exp019_seq_nn/submission.csv")
    blend_test=w[0]*e18t+w[1]*e19t+w[2]*nn_test
    test_out=test.copy(); test_out.loc[test_out["is_target"].astype(bool),PRED_COL]=anchor_te+blend_test
    build_submission(test_out,sample,PRED_COL).to_csv(EXP_DIR/"submission.csv",index=False)

    result={"exp_id":EXP_ID,"created_at":now_jst(),"status":"completed",
        "model":"typewell cross-attention TCN","grid_L":L,"tw_M":M,"band":BAND,
        "nn_cv":nn_cv,"exp018_cv":cv_e18,"blend3_cv":cv_blend,
        "blend_weights":{"exp018":round(float(w[0]),4),"exp019":round(float(w[1]),4),"exp020":round(float(w[2]),4)},
        "improvement_vs_exp018":round(cv_e18-cv_blend,6),"err_corr_020_vs_018":round(ecorr,4),
        "prev_best":13.331211,"leak_risk":"low (no hidden TVT, typewell ref only, well-grouped fold)"}
    write_json(EXP_DIR/"result.json",result)
    print("\n=== RESULT ===")
    print(f"  exp020 NN CV      = {nn_cv:.6f}")
    print(f"  exp018 CV         = {cv_e18:.6f}")
    print(f"  3-model blend CV  = {cv_blend:.6f}  (w={np.round(w,3)})  vs exp018 {cv_e18-cv_blend:+.6f}")
    print(f"  err corr(020,018) = {ecorr:.4f}")

if __name__=="__main__":
    main()
