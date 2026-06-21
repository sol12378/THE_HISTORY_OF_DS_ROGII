#!/usr/bin/env python3
"""exp019: per-well 系列NN (dilated 1D CNN / TCN) + exp018 blend。

ユーザー要望: NN modelを加えたensembleでCVを引き下げる。
これまでの全手法(tree+pointwise特徴)が捉えられない唯一の paradigm =
**well全体のGR+幾何 系列を端から端まで見る系列モデル**。oracleは「geom形状は正しく
残差は低周波offset(seg_oracle=4.4)」を示す。系列NNはgeom prior上の残差を学習し、
GR系列形状から offset を補正できるか試す(現実的最大の試み)。

設計:
- 各wellを row_idx順に並べ L=1024 の固定グリッドへ線形リサンプル(低周波offsetには十分)。
- 入力ch: GR(well内known正規化)+missing flag, is_known, known_delta(known区間のTVT-anchor),
          dMD/dZ/dX/dY/z_rel(幾何), row_pos, geom_delta_prior(slope20*dMD)。
- 出力: NNは **residual** を予測し pred_delta = geom_delta_prior + residual。
- loss: hidden(target)グリッド点のみ masked MSE(deltaを/50正規化)。
- fold: folds_group_well_v001(exp018と同一)。OOFを元の行へ補間し本来のTVT-RMSEで評価。
- 最後に exp018(LGBM+XGB+CatBoost) blendと凸加重blend → 必要なら exp015平滑化。
leak-free: typewell不使用、hidden TVT不使用(known_delta/anchorはknownのみ)。well-grouped fold。
"""
from __future__ import annotations
import sys, importlib.util, math
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from rogii.training.baselines import (PRED_COL, target_rows, tvt_rmse, write_json,
    now_jst, build_submission, ensure_exp_dir, attach_folds, load_base_inputs)

import torch, torch.nn as nn
torch.manual_seed(42); np.random.seed(42)

EXP_ID = "exp019_seq_nn"
EXP_DIR = Path("experiments") / EXP_ID
L = 1024
SCALE = 50.0
EPOCHS = 60
BATCH = 16
LR = 2e-3
DEV = torch.device("cpu")


def build_well_arrays(train):
    """各wellを固定グリッドにリサンプル。戻り: arrays dict。"""
    feats = []
    wid_order = []
    targets = []; masks = []; row_index = []  # for mapping back
    # per-well known GR stats for normalization
    g = train.sort_values(["well_id","row_idx"])
    for wid, sub in g.groupby("well_id", sort=False):
        n = len(sub)
        pos = np.arange(n)/max(n-1,1)
        grid = np.linspace(0,1,L)
        def rs(a):  # resample to grid
            return np.interp(grid, pos, np.asarray(a,float))
        is_known = sub["is_known_tvt"].astype(float).to_numpy()
        is_tgt = sub["is_target"].astype(float).to_numpy()
        gr = sub["GR"].to_numpy(float); grmiss = sub["is_gr_missing"].astype(float).to_numpy()
        kmask = is_known.astype(bool) & (grmiss<0.5)
        if kmask.sum()>=5:
            gm, gs = np.mean(gr[kmask]), np.std(gr[kmask])+1e-6
        else:
            gm, gs = np.nanmean(gr) if np.isfinite(np.nanmean(gr)) else 0.0, 1.0
        gr_norm = np.where(grmiss<0.5,(gr-gm)/gs,0.0)
        anchor = float(sub["last_known_TVT"].iloc[0])
        tvt_in = sub["TVT_input"].to_numpy(float)
        known_delta = np.where(is_known>0.5,(tvt_in-anchor)/SCALE,0.0)
        dMD = sub["delta_MD_from_PS"].to_numpy(float)/5000.0
        dZ  = sub["delta_Z_from_PS"].to_numpy(float)/50.0
        dX  = sub["delta_X_from_PS"].to_numpy(float)/5000.0
        dY  = sub["delta_Y_from_PS"].to_numpy(float)/5000.0
        z_rel = (sub["Z"].to_numpy(float)-float(sub["last_known_Z"].iloc[0]))/50.0
        # geom prior delta = slope20 * delta_MD_from_PS
        slope20 = float(sub["_slope20"].iloc[0])
        geom_delta = slope20 * sub["delta_MD_from_PS"].to_numpy(float)
        ch = np.stack([rs(gr_norm), rs(grmiss), rs(is_known), rs(known_delta),
                       rs(dMD), rs(dZ), rs(dX), rs(dY), rs(z_rel), grid,
                       rs(geom_delta)/SCALE], 0).astype(np.float32)  # (C,L)
        tg = sub["TVT"].to_numpy(float)
        delta_true = (tg - anchor)/SCALE
        tgt_grid = rs(np.where(is_tgt>0.5, delta_true, 0.0))
        msk_grid = (rs(is_tgt) > 0.5).astype(np.float32)
        feats.append(ch); targets.append(tgt_grid.astype(np.float32))
        masks.append(msk_grid); wid_order.append(wid)
    return (np.stack(feats), np.stack(targets), np.stack(masks),
            wid_order, grid)


class TCN(nn.Module):
    def __init__(self, cin, ch=64, dil=(1,2,4,8,16,32)):
        super().__init__()
        layers=[]; prev=cin
        for d in dil:
            layers += [nn.Conv1d(prev,ch,3,padding=d,dilation=d), nn.BatchNorm1d(ch),
                       nn.GELU(), nn.Dropout(0.1)]
            prev=ch
        self.body=nn.Sequential(*layers)
        self.head=nn.Conv1d(ch,1,1)
        self.skip=nn.Conv1d(cin,ch,1)
    def forward(self,x):
        h=self.body(x)+self.skip(x)
        return self.head(h).squeeze(1)  # (B,L) residual (normalized)


def main():
    ensure_exp_dir(EXP_DIR)
    print(f"[{EXP_ID}] per-well sequence TCN + exp018 blend")
    train, test, folds, sample = load_base_inputs(
        ROOT/"data/processed/train_base_v001.parquet",
        ROOT/"data/processed/test_base_v001.parquet",
        ROOT/"data/folds/folds_group_well_v001.csv",
        ROOT/"data/raw/sample_submission.csv")

    # precompute slope20 per well (geom prior), known section
    def add_slope20(df):
        recs={}
        kn=df[df["is_known_tvt"].astype(bool)]
        for wid,gsub in kn.groupby("well_id",sort=False):
            gsub=gsub.sort_values("row_idx")
            md=gsub["MD"].to_numpy(float); tv=gsub["TVT_input"].to_numpy(float)
            n=len(gsub); n20=min(20,n); d=md[-1]-md[-n20] if n20>1 else 1.
            recs[wid]=(tv[-1]-tv[-n20])/d if abs(d)>1e-6 else 0.0
        df["_slope20"]=df["well_id"].map(recs).fillna(0.0); return df
    train=add_slope20(train); test=add_slope20(test)
    train=attach_folds(train,folds)

    print("build train grids ..."); Xtr,Ytr,Mtr,wtr,grid = build_well_arrays(train)
    print("build test grids  ..."); Xte,_,_,wte,_ = build_well_arrays(test)
    cin=Xtr.shape[1]
    wfold = train.groupby("well_id",sort=False)["fold"].first()
    fold_arr = np.array([wfold[w] for w in wtr])

    # original-row frames for back-interp (target rows)
    tr_t = target_rows(train).copy()
    te_t = target_rows(test).copy()
    def backinterp(wid_list, grid_pred_by_wid, frame):
        out=np.zeros(len(frame))
        frame=frame.copy(); frame["_o"]=np.arange(len(frame))
        for wid, sub in frame.groupby("well_id",sort=False):
            sub=sub.sort_values("row_idx")
            ridx=sub["row_idx"].to_numpy(float)
            # position of target rows within full well = row_idx/(n_rows-1)
            nrw=float(sub["n_rows_in_well"].iloc[0])
            pos=ridx/max(nrw-1,1)
            gp=grid_pred_by_wid[wid]
            out[sub["_o"].to_numpy()]=np.interp(pos, grid, gp)
        return out

    Xtr_t=torch.tensor(Xtr); Ytr_t=torch.tensor(Ytr); Mtr_t=torch.tensor(Mtr)
    Xte_t=torch.tensor(Xte)
    # geom prior delta on grid (normalized) is channel index 10
    GEOM_CH=10
    oof_grid={}; test_grid_acc=np.zeros((len(wte),L)); nf=len(np.unique(fold_arr))
    fold_rmse=[]
    for f in sorted(np.unique(fold_arr)):
        vm=fold_arr==f; tm=~vm
        model=TCN(cin).to(DEV)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
        idx_tr=np.where(tm)[0]
        best_val=1e9; best_state=None; patience=0
        for ep in range(EPOCHS):
            model.train(); np.random.shuffle(idx_tr)
            for i in range(0,len(idx_tr),BATCH):
                bi=idx_tr[i:i+BATCH]
                xb=Xtr_t[bi].to(DEV); yb=Ytr_t[bi].to(DEV); mb=Mtr_t[bi].to(DEV)
                resid=model(xb)
                pred=xb[:,GEOM_CH,:]+resid   # residual on geom prior (normalized)
                loss=((pred-yb)**2*mb).sum()/(mb.sum()+1e-6)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            sched.step()
            # val RMSE on grid (normalized->TVT scale approx; use grid masked)
            model.eval()
            with torch.no_grad():
                xb=Xtr_t[vm].to(DEV); yb=Ytr_t[vm].to(DEV); mb=Mtr_t[vm].to(DEV)
                pred=xb[:,GEOM_CH,:]+model(xb)
                vrmse=float(torch.sqrt(((pred-yb)**2*mb).sum()/(mb.sum()+1e-6)))*SCALE
            if vrmse<best_val-1e-4:
                best_val=vrmse; best_state={k:v.clone() for k,v in model.state_dict().items()}; patience=0
            else:
                patience+=1
                if patience>=10: break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp=(Xtr_t[vm][:,GEOM_CH,:]+model(Xtr_t[vm].to(DEV))).cpu().numpy()*SCALE
            tp=(Xte_t[:,GEOM_CH,:]+model(Xte_t.to(DEV))).cpu().numpy()*SCALE
        for j,wi in enumerate(np.where(vm)[0]):
            oof_grid[wtr[wi]]=vp[j]
        test_grid_acc += tp/nf
        print(f"  fold{f} grid val RMSE(norm-grid)={best_val:.4f}")
        fold_rmse.append(best_val)

    test_grid={wte[j]:test_grid_acc[j] for j in range(len(wte))}
    # back-interp to original target rows
    print("back-interp to rows ...")
    anchor_tr=tr_t["last_known_TVT"].to_numpy(float)
    anchor_te=te_t["last_known_TVT"].to_numpy(float)
    nn_oof_delta=backinterp(wtr, oof_grid, tr_t)
    nn_test_delta=backinterp(wte, test_grid, te_t)
    nn_cv=tvt_rmse(tr_t["TVT"], anchor_tr+nn_oof_delta)
    print(f"\n  NN OOF CV = {nn_cv:.6f}")

    # ---- blend with exp018 ----
    e18=pd.read_csv(EXP_DIR.parent/"exp018_model_blend"/"oof.csv",
                    usecols=["well_id","row_idx","pred_delta"]).rename(columns={"pred_delta":"e18"})
    tr_t=tr_t.merge(e18,on=["well_id","row_idx"],how="left")
    nn_s=pd.Series(nn_oof_delta,index=tr_t.index)
    e18d=tr_t["e18"].to_numpy(float)
    true_delta=(tr_t["TVT"].astype(float)-anchor_tr).to_numpy()
    from scipy.optimize import minimize
    O=np.vstack([e18d,nn_oof_delta]).T
    def loss(w): return float(np.sqrt(np.mean((true_delta-O@w)**2)))
    res=minimize(loss,[0.7,0.3],method="SLSQP",bounds=[(0,1)]*2,
                 constraints=({"type":"eq","fun":lambda w:w.sum()-1},))
    w=res.x; blend_oof=O@w
    cv_blend=tvt_rmse(tr_t["TVT"],anchor_tr+blend_oof)
    cv_e18=tvt_rmse(tr_t["TVT"],anchor_tr+e18d)
    # corr of errors
    err_nn=anchor_tr+nn_oof_delta-tr_t["TVT"].to_numpy(float)
    err_e18=anchor_tr+e18d-tr_t["TVT"].to_numpy(float)
    ecorr=float(np.corrcoef(err_nn,err_e18)[0,1])

    # test blend
    e18t=pd.read_csv(EXP_DIR.parent/"exp018_model_blend"/"submission.csv")
    # exp018 submission is final TVT; reconstruct delta via anchor from te_t order
    e18t_map=dict(zip(e18t["id"],e18t["tvt"]))
    e18_test_tvt=te_t["id"].map(e18t_map).to_numpy(float)
    e18_test_delta=e18_test_tvt-anchor_te
    blend_test_delta=w[0]*e18_test_delta+w[1]*nn_test_delta

    # save
    oof_df=tr_t[["id","well_id","row_idx","fold","TVT","last_known_TVT"]].copy()
    oof_df["pred_delta"]=blend_oof; oof_df[PRED_COL]=anchor_tr+blend_oof
    oof_df["pred_nn"]=anchor_tr+nn_oof_delta; oof_df["pred_e18"]=anchor_tr+e18d
    oof_df.to_csv(EXP_DIR/"oof.csv",index=False)
    test_out=test.copy()
    test_out.loc[test_out["is_target"].astype(bool),PRED_COL]=anchor_te+blend_test_delta
    build_submission(test_out,sample,PRED_COL).to_csv(EXP_DIR/"submission.csv",index=False)

    result={"exp_id":EXP_ID,"created_at":now_jst(),"status":"completed",
        "fold":"folds_group_well_v001","model":"per-well TCN residual on geom prior",
        "grid_L":L,"epochs":EPOCHS,"nn_cv":nn_cv,"exp018_cv":cv_e18,
        "blend_weights":{"exp018":round(float(w[0]),4),"nn":round(float(w[1]),4)},
        "blend_cv_preSmooth":cv_blend,"improvement_vs_exp018":round(cv_e18-cv_blend,6),
        "err_corr_nn_vs_exp018":round(ecorr,4),
        "best_prev":13.340426,"target_user":"CV ~5",
        "leak_risk":"low (no typewell, no hidden TVT, well-grouped fold)",
        "note":"run exp015_seq_smooth --model-exp exp019_seq_nn for smoothed CV"}
    write_json(EXP_DIR/"result.json",result)
    print("\n=== RESULT ===")
    print(f"  NN CV               = {nn_cv:.6f}")
    print(f"  exp018 CV           = {cv_e18:.6f}")
    print(f"  blend CV (w={np.round(w,3)}) = {cv_blend:.6f}  vs exp018 {cv_e18-cv_blend:+.6f}")
    print(f"  err corr(NN, exp018)= {ecorr:.4f}")

if __name__=="__main__":
    main()
