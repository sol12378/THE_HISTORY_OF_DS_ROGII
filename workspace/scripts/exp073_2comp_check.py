#!/usr/bin/env python
"""Check: can a simple 2-level submission keep the gain?
  PUBLIC = nested NNLS of all rav_*/pilk_* components (their world)
  OURS   = nested NNLS of pf/geom/tcn_resid (our world)
  then nested NNLS{PUBLIC, OURS} + projection.
Compares to full joint blend (8.690) and current best (9.086).
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp073_blend_harness import load_components, robust_projection, N_FOLDS

def nested(df, cols, y, groups):
    X = df[cols].to_numpy(float); pred = np.zeros(len(y)); fid = np.zeros(len(y), int)
    for k,(tr,te) in enumerate(GroupKFold(N_FOLDS).split(X,y,groups)):
        w,_ = nnls(X[tr], y[tr]); pred[te] = X[te]@w; fid[te]=k
    return pred, fid

df, comp = load_components()
y = df["target_delta"].to_numpy(float); groups = df["well_id"].to_numpy()
lk = df["last_known_TVT"].to_numpy()

pub_cols = [c for c in comp if c.startswith(("pilk_","rav_"))]
our_cols = [c for c in comp if c in ("pf","geom","tcn_resid")]
pub,_ = nested(df, pub_cols, y, groups)
our,_ = nested(df, our_cols, y, groups)
def rmse(d): return float(np.sqrt(np.mean((y+lk-(d+lk))**2)))
print(f"PUBLIC stack nested delta-RMSE: {rmse(pub):.4f}")
print(f"OURS   stack nested delta-RMSE: {rmse(our):.4f}")
print(f"corr(PUBLIC,OURS) err: {np.corrcoef(pub-y, our-y)[0,1]:.3f}")

df2 = df.copy(); df2["_pub"]=pub; df2["_our"]=our
pred2, fid2 = nested(df2, ["_pub","_our"], y, groups)
# weights
X2=df2[["_pub","_our"]].to_numpy(float); ws=[]
for tr,te in GroupKFold(N_FOLDS).split(X2,y,groups):
    w,_=nnls(X2[tr],y[tr]); ws.append(w)
ws=np.array(ws).mean(0); ws=ws/ws.sum()
print(f"2-comp weights: PUBLIC={ws[0]:.3f} OURS={ws[1]:.3f}")
tvt2 = pred2 + lk
print(f"2-comp nested CV (pre-proj): {float(np.sqrt(np.mean((y+lk-tvt2)**2))):.4f}")
proj = robust_projection(df2, tvt2, 4, 0.75)
print(f"2-comp nested CV (proj d4 b0.75): {float(np.sqrt(np.mean((y+lk-proj)**2))):.4f}")
print("(full joint blend=8.690, current best=9.086)")
