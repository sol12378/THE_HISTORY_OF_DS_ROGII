#!/usr/bin/env python3
"""exp038 + exp022 blend test (仕様通り)."""

import numpy as np
import pandas as pd
from scipy.optimize import nnls

def blend_exp038_exp022():
    """Blend exp038 Viterbi と exp022 PF。"""
    print("\n=== Blend Test: exp038 (Viterbi) + exp022 (PF) ===\n")

    # Load OOF
    p22 = pd.read_csv('experiments/exp022_particle_filter/oof.csv')[
        ['well_id', 'row_idx', 'TVT', 'pred_tvt', 'last_known_TVT']
    ].rename(columns={'pred_tvt': 'p22'})

    p38 = pd.read_csv('experiments/exp038_dp_viterbi/oof.csv')[
        ['well_id', 'row_idx', 'pred_tvt']
    ].rename(columns={'pred_tvt': 'p38'})

    # Merge
    m = p22.merge(p38, on=['well_id', 'row_idx'])

    # Errors
    m['e22'] = m['p22'] - m['TVT']
    m['e38'] = m['p38'] - m['TVT']

    # Error correlation
    corr = m[['e22', 'e38']].corr().iloc[0, 1]
    print(f"Error correlation: {corr:.3f}")

    # Delta from anchor
    m['p22d'] = m['p22'] - m['last_known_TVT']
    m['p38d'] = m['p38'] - m['last_known_TVT']
    m['yd'] = m['TVT'] - m['last_known_TVT']

    # NNLS
    w, _ = nnls(m[['p22d', 'p38d']].values, m['yd'].values)
    pred = m[['p22d', 'p38d']].values @ w + m['last_known_TVT'].values
    blend_rmse = np.sqrt(((pred - m.TVT) ** 2).mean())

    print(f"NNLS w22={w[0]:.3f}, w38={w[1]:.3f}")
    print(f"Blend CV RMSE: {blend_rmse:.4f}")
    print(f"\nRefs:")
    print(f"  exp022 (PF):    {np.sqrt(((m.p22 - m.TVT)**2).mean()):.4f}")
    print(f"  exp038 (Viterbi): {np.sqrt(((m.p38 - m.TVT)**2).mean()):.4f}")
    print(f"  anchor:         {np.sqrt(((m.last_known_TVT - m.TVT)**2).mean()):.4f}")
    print(f"\nBlend value: {'YES' if corr < 0.7 else 'Weak'} (corr={corr:.3f})")

if __name__ == "__main__":
    blend_exp038_exp022()
