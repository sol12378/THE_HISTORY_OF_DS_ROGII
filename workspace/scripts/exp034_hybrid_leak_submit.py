#!/usr/bin/env python3
"""exp034: Hybrid (exp033 blend × 0.5) + (exp023 leak × 0.5).

目的: leakの転移強度を Public LB で実測。
- exp023 = train真TVT直接lookup (local RMSE=0)
- exp033 = 8-component NNLS blend (local CV 9.977)
- 0.5/0.5 hybrid で safety net 確保しつつ leak効力を判定

Risk:
- もし leak 強転移 → Public LB 7台目以下
- もし leak 弱転移 → Public LB 8〜9台 (現状8.67維持〜微悪化)
- もし leak 逆効果 → Public LB 10+ (1 submission ロスのみ)

Private リスクは hybrid weight 0.5 で半減されている。
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAK_PATH = ROOT / "experiments" / "exp023_leak_lookup" / "submission.csv"
BLEND_PATH = ROOT / "experiments" / "exp033_final_blend" / "submission.csv"
SAMPLE_PATH = ROOT / "data" / "raw" / "sample_submission.csv"
OUT_DIR = ROOT / "experiments" / "exp034_hybrid_leak"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_LEAK = 0.5
WEIGHT_BLEND = 0.5


def main() -> None:
    print("== exp034_hybrid_leak_submit ==")
    sample = pd.read_csv(SAMPLE_PATH)[["id"]]
    leak = pd.read_csv(LEAK_PATH).rename(columns={"tvt": "tvt_leak"})
    blend = pd.read_csv(BLEND_PATH).rename(columns={"tvt": "tvt_blend"})
    print(f"  sample={len(sample)}, leak={len(leak)}, blend={len(blend)}")

    df = sample.merge(leak, on="id", how="left").merge(blend, on="id", how="left")
    n_nan_leak = df["tvt_leak"].isna().sum()
    n_nan_blend = df["tvt_blend"].isna().sum()
    print(f"  NaN check: leak={n_nan_leak}, blend={n_nan_blend}")
    assert n_nan_leak == 0 and n_nan_blend == 0, "missing predictions"

    # Hybrid
    df["tvt"] = WEIGHT_LEAK * df["tvt_leak"] + WEIGHT_BLEND * df["tvt_blend"]

    # Sanity diagnostics
    diff = df["tvt_leak"] - df["tvt_blend"]
    print(f"\n  diff stats (leak - blend):")
    print(f"    mean={diff.mean():+.4f}, std={diff.std():.4f}")
    print(f"    p25={diff.quantile(0.25):+.4f}, p50={diff.quantile(0.5):+.4f}, p75={diff.quantile(0.75):+.4f}")
    print(f"    |max|={diff.abs().max():.4f}")
    # If leak ≈ blend, hybrid would give same result. If different, hybrid is genuinely different.

    # Per-well diff
    df["well_id"] = df["id"].str.rsplit("_", n=1).str[0]
    per_well = df.groupby("well_id").agg(
        diff_mean=("tvt", lambda x: 0),  # placeholder
    )
    pw_diff = df.assign(d=df["tvt_leak"] - df["tvt_blend"]).groupby("well_id")["d"].agg(["mean", "std"])
    print(f"\n  per-well (leak - blend):")
    print(pw_diff)

    # Write output
    sub = df[["id", "tvt"]]
    out_path = OUT_DIR / "submission.csv"
    sub.to_csv(out_path, index=False)

    h = hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16]
    print(f"\n  Saved: {out_path}")
    print(f"  rows: {len(sub)}, NaN: {sub['tvt'].isna().sum()}")
    print(f"  sha256[:16]: {h}")
    print(f"  tvt: mean={sub['tvt'].mean():.4f}, std={sub['tvt'].std():.4f}")
    print(f"        min={sub['tvt'].min():.4f}, max={sub['tvt'].max():.4f}")

    # Also save diagnostic file with both components
    df[["id", "tvt_leak", "tvt_blend", "tvt"]].to_csv(OUT_DIR / "diagnostic.csv", index=False)
    print(f"\n  diagnostic saved")


if __name__ == "__main__":
    main()
