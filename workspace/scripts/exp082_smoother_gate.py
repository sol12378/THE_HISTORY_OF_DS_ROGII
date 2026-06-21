#!/usr/bin/env python
"""exp082 (loop-exp#2): offline agreement-gating of exp081 smoother.
Keep smoother only where it agrees with forward (|sm-fwd|<=T); else forward.
Tests if the SAFE subset of backward corrections helps (avoids catastrophic wrong-branch)."""
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII\experiments\exp081_pf_smoother")
FOREN = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII\experiments\exp073_public_assets_integration\forensics_per_well.csv")
d = pd.read_csv(OUT / "preds.csv")
foren = pd.read_csv(FOREN)[["well_id", "rmse_blend"]]

def pooled(col):
    return float(np.sqrt(np.mean((d[col] - d["TVT"]) ** 2)))

print("forward pooled :", round(pooled("fwd"), 3))
print("smoother pooled:", round(pooled("sm"), 3))
dis = (d["sm"] - d["fwd"]).abs()
print("|sm-fwd| q[.5,.75,.9,.95,.99]:", [round(q, 2) for q in dis.quantile([.5, .75, .9, .95, .99]).tolist()])
best = None
for T in [1, 2, 4, 8, 15, 30]:
    g = np.where(dis <= T, d["sm"], d["fwd"])
    r = float(np.sqrt(np.mean((g - d["TVT"]) ** 2)))
    frac = float((dis <= T).mean())
    print(f"  gate T={T:>2}: pooled={r:.3f}  (sm used on {frac*100:.0f}% rows)")
    if best is None or r < best[1]:
        best = (T, r)
T = best[0]
d["gated"] = np.where(dis <= T, d["sm"], d["fwd"])
pw = d.groupby("well_id").apply(
    lambda x: pd.Series({"fwd": np.sqrt(np.mean((x["fwd"] - x["TVT"]) ** 2)),
                         "gated": np.sqrt(np.mean((x["gated"] - x["TVT"]) ** 2))}),
    include_groups=False).reset_index().merge(foren, on="well_id")
bad = pw[pw["rmse_blend"] >= 8]; good = pw[pw["rmse_blend"] < 8]
print(f"\nbest gate T={T} pooled={best[1]:.3f} (forward {pooled('fwd'):.3f})")
print(f"  bad+broken({len(bad)}): fwd med {bad['fwd'].median():.2f} -> gated {bad['gated'].median():.2f} "
      f"(improved>0.5: {int((bad['gated']<bad['fwd']-0.5).sum())}, worsened>0.5: {int((bad['gated']>bad['fwd']+0.5).sum())})")
print(f"  good({len(good)}): fwd med {good['fwd'].median():.2f} -> gated {good['gated'].median():.2f} "
      f"(worsened>1: {int((good['gated']>good['fwd']+1).sum())})")
print(f"VERDICT: gated smoother {'BEATS' if best[1] < pooled('fwd') - 0.05 else 'does NOT beat'} forward on this subset")
