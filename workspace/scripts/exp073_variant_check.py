#!/usr/bin/env python
"""Which external subset is needed? Compare blend CV (+proj d4 b0.75) for:
  ours-only, ours+rav, ours+pilk, ours+both(joint).
Goal: find the simplest reproducible submission that keeps the gain.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp073_blend_harness import load_components, robust_projection, N_FOLDS

df, comp = load_components()
y = df["target_delta"].to_numpy(float); g = df["well_id"].to_numpy(); lk = df["last_known_TVT"].to_numpy()
ours = [c for c in comp if c in ("pf", "geom", "tcn_resid")]
rav = [c for c in comp if c.startswith("rav_")]
pilk = [c for c in comp if c.startswith("pilk_")]

def blend_cv(cols, proj=True):
    X = df[cols].to_numpy(float); pred = np.zeros(len(y)); fid = np.zeros(len(y), int); ws=[]
    for k,(tr,te) in enumerate(GroupKFold(N_FOLDS).split(X,y,g)):
        w,_ = nnls(X[tr], y[tr]); pred[te] = X[te]@w; fid[te]=k; ws.append(w)
    tvt = pred + lk
    raw = float(np.sqrt(np.mean((y+lk-tvt)**2)))
    if proj:
        pj = robust_projection(df, tvt, 4, 0.75)
        pr = float(np.sqrt(np.mean((y+lk-pj)**2)))
    else:
        pr = raw
    w = np.array(ws).mean(0)
    nz = {c: round(float(wi),3) for c,wi in zip(cols,w) if wi>0.01}
    return raw, pr, nz

# DEPLOYABLE set: components we can faithfully reproduce at scoring time.
# EXCLUDE pilk_tcn (inference broken: hidden-only cold-start, OOF mean +1.6 vs deploy -35.6)
# and our tcn (weak). Keep pf, geom (ours, reliable) + ravaghi (koolbox Trainer) + pilk_cat (catboost).
deployable = ["pf", "geom"] + rav + ["pilk_cat"]
deployable_min = ["pf", "geom", "rav_lgb3", "rav_cb1", "rav_cb2", "pilk_cat"]

for name, cols in [("ours-only", ours), ("ours+rav", ours+rav),
                   ("ours+pilk", ours+pilk), ("ours+both(joint)", ours+rav+pilk),
                   ("DEPLOYABLE(no pilk_tcn)", deployable),
                   ("DEPLOYABLE_min", deployable_min)]:
    raw, pr, nz = blend_cv(cols)
    print(f"{name:20s} raw={raw:.4f}  proj={pr:.4f}  weights={nz}")
print("(current best=9.086)")
