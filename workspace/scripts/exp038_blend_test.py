#!/usr/bin/env python3
"""Blend test: exp038 (Viterbi) + exp022 (PF).

Compare error correlation and evaluate NNLS blend (Differential features: pred - last_known_TVT).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse

def main():
    print("[exp038 BLEND TEST] exp038 (Viterbi) + exp022 (PF)")

    # Load both OOF files
    print("Loading OOF files...", flush=True)
    p22 = pd.read_csv("experiments/exp022_particle_filter/oof.csv")[
        ["well_id", "row_idx", "TVT", "pred_tvt", "last_known_TVT"]
    ].rename(columns={"pred_tvt": "p22"})

    p38 = pd.read_csv("experiments/exp038_dp_viterbi/oof.csv")[
        ["well_id", "row_idx", "pred_tvt"]
    ].rename(columns={"pred_tvt": "p38"})

    # Merge on well_id, row_idx
    print("Merging OOF files...", flush=True)
    m = p22.merge(p38, on=["well_id", "row_idx"], how="inner")
    print(f"  Merged rows: {len(m)} / {len(p22)} (p22), {len(p38)} (p38)")

    # Compute errors
    m["e22"] = m["p22"] - m["TVT"]
    m["e38"] = m["p38"] - m["TVT"]

    # Compute error correlation
    corr = m[["e22", "e38"]].corr().iloc[0, 1]
    print(f"\nError correlation (p22 vs p38): {corr:.4f}")

    # Compute individual RMSEs
    rmse22 = tvt_rmse(m["TVT"], m["p22"])
    rmse38 = tvt_rmse(m["TVT"], m["p38"])
    rmse_anchor = tvt_rmse(m["TVT"], m["last_known_TVT"])

    print(f"Individual RMSEs:")
    print(f"  anchor:  {rmse_anchor:.6f}")
    print(f"  exp022:  {rmse22:.6f}")
    print(f"  exp038:  {rmse38:.6f}")

    # Blend via differential features
    print(f"\n[NNLS Blend] Using differential features (pred - last_known_TVT)")
    m["p22d"] = m["p22"] - m["last_known_TVT"]
    m["p38d"] = m["p38"] - m["last_known_TVT"]
    m["yd"] = m["TVT"] - m["last_known_TVT"]

    # NNLS on differential
    print(f"  Matrix shape: {m[['p22d','p38d']].shape}")
    print(f"  Target shape: {m['yd'].shape}")

    w, residual = nnls(m[["p22d", "p38d"]].values, m["yd"].values)
    print(f"  NNLS weights: w22={w[0]:.4f}, w38={w[1]:.4f}")
    print(f"  Sum: {sum(w):.4f}")

    # Blend prediction
    pred_blend = m[["p22d", "p38d"]].values @ w + m["last_known_TVT"].values
    rmse_blend = tvt_rmse(m["TVT"], pred_blend)

    print(f"\nBlend results:")
    print(f"  NNLS blend CV: {rmse_blend:.6f}")
    print(f"  Improvement vs anchor: {rmse_anchor - rmse_blend:+.6f}")
    print(f"  Improvement vs exp022: {rmse22 - rmse_blend:+.6f}")
    print(f"  Improvement vs exp038: {rmse38 - rmse_blend:+.6f}")

    # Detailed analysis
    print(f"\n[Detail] Error analysis:")
    print(f"  exp022 (p22):")
    print(f"    mean error: {m['e22'].mean():+.4f}")
    print(f"    median error: {m['e22'].median():+.4f}")
    print(f"    std error: {m['e22'].std():.4f}")
    print(f"  exp038 (p38):")
    print(f"    mean error: {m['e38'].mean():+.4f}")
    print(f"    median error: {m['e38'].median():+.4f}")
    print(f"    std error: {m['e38'].std():.4f}")

    # Count wells where each method wins
    p22_wins = (m.groupby("well_id").apply(lambda g: tvt_rmse(g["TVT"], g["p22"]) < tvt_rmse(g["TVT"], g["p38"])).sum())
    p38_wins = (m.groupby("well_id").apply(lambda g: tvt_rmse(g["TVT"], g["p38"]) < tvt_rmse(g["TVT"], g["p22"])).sum())
    print(f"\nWell-level wins:")
    print(f"  exp022 wins: {p22_wins} wells")
    print(f"  exp038 wins: {p38_wins} wells")
    print(f"  Tied: {773 - p22_wins - p38_wins} wells")

    # Summary
    print(f"\n{'='*60}")
    if corr < 0.7:
        print(f"✓ Low error correlation ({corr:.4f}) → blend has potential")
    else:
        print(f"✗ High error correlation ({corr:.4f}) → blend may not help much")

    if rmse_blend < rmse22 and rmse_blend < rmse38:
        print(f"✓ Blend improves both methods")
    elif rmse_blend < rmse22 or rmse_blend < rmse38:
        print(f"✓ Blend improves at least one method")
    else:
        print(f"✗ Blend does not improve individual methods")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
