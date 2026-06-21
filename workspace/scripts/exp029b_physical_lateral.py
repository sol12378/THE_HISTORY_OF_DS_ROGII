#!/usr/bin/env python3
"""exp029b — lateral-only物理モデル.

exp029の失敗原因: last K=200 known rowsにbuildセクションが混入し、急dipが反映された。
lateral区間ではTVT変化は微小(~0.01ft/row)。**lateral区間のdipのみ**を使い線形外挿。

Sunny出力の0.01ft/row刻みパターンと一致する設計。

variants:
- physE: last 30 known rows で dTVT/dMD → MD外挿
- physF: last 30 known rows で dTVT/dZ → Z軌道に投影
- physG: ロバスト中央値 dTVT/dMD (last 50)
- physH: TVT_input 単純延長 (= anchor baseline)
- phys_blend: mean of physE/F/G + anchor補正
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "train_base_v001.parquet"
OUT_DIR = ROOT / "experiments" / "exp029b_physical_lateral"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_phys_well(df_well: pd.DataFrame) -> pd.DataFrame:
    df_well = df_well.sort_values("MD").reset_index(drop=True)
    known = df_well["TVT_input"].notna() & df_well["TVT"].notna()
    n = len(df_well)
    md = df_well["MD"].to_numpy(dtype=float)
    z = df_well["Z"].to_numpy(dtype=float)
    tvt = df_well["TVT"].to_numpy(dtype=float)

    last_idx = np.where(known.values)[0]
    if len(last_idx) < 30:
        last_tvt = tvt[last_idx[-1]] if len(last_idx) else 0.0
        return pd.DataFrame({
            "MD": md, "is_hidden": ~known.values,
            "physE_MD": [last_tvt] * n,
            "physF_Z": [last_tvt] * n,
            "physG_robust": [last_tvt] * n,
            "physH_anchor": [last_tvt] * n,
        })

    last_md = md[last_idx[-1]]
    last_tvt = tvt[last_idx[-1]]
    last_z = z[last_idx[-1]]

    # === physE: 直近30行の MD-TVT 線形 ===
    K = min(30, len(last_idx))
    use = last_idx[-K:]
    if md[use].std() > 1e-6:
        slope_md, _ = np.polyfit(md[use], tvt[use], 1)
    else:
        slope_md = 0.0
    physE = last_tvt + slope_md * (md - last_md)

    # === physF: 直近30行の Z-TVT 線形 ===
    if z[use].std() > 1e-6:
        slope_z, _ = np.polyfit(z[use], tvt[use], 1)
    else:
        slope_z = 0.0
    physF = last_tvt + slope_z * (z - last_z)

    # === physG: dTVT/dMD ロバスト中央値 (last 50) ===
    K2 = min(50, len(last_idx) - 1)
    use2 = last_idx[-(K2 + 1):]
    dtvt = np.diff(tvt[use2])
    dmd = np.diff(md[use2])
    rate = np.median(dtvt / np.where(dmd != 0, dmd, 1e-6))
    physG = last_tvt + rate * (md - last_md)

    # === physH: pure anchor 延長 ===
    physH = np.full(n, last_tvt)
    # known区間は真値で埋める (RMSE安定のため)
    for variant in [physE, physF, physG, physH]:
        variant[last_idx] = tvt[last_idx]

    return pd.DataFrame({
        "MD": md, "is_hidden": ~known.values,
        "physE_MD": physE,
        "physF_Z": physF,
        "physG_robust": physG,
        "physH_anchor": physH,
    })


def main() -> None:
    t0 = time.time()
    print("== exp029b_physical_lateral ==")
    df = pd.read_parquet(PROCESSED)
    df = df[df["split"] == "train"].copy()
    print(f"rows={len(df)}, wells={df['well_id'].nunique()}")

    preds = []
    for wid, g in df.groupby("well_id", sort=False):
        out = fit_phys_well(g)
        out["well_id"] = wid
        out["row_idx"] = g["row_idx"].values
        preds.append(out)
    pred_df = pd.concat(preds, ignore_index=True)

    tgt_mask = df["TVT_input"].isna() & df["TVT"].notna()
    df_tgt = df.loc[tgt_mask, ["well_id", "row_idx", "TVT"]].copy()
    pred_tgt = pred_df.merge(df_tgt, on=["well_id", "row_idx"], how="inner")
    print(f"target rows: {len(pred_tgt)}")

    method_cols = ["physE_MD", "physF_Z", "physG_robust", "physH_anchor"]
    # blend: mean over methods (excluding anchor for stability test)
    pred_tgt["phys_blend"] = pred_tgt[["physE_MD", "physF_Z", "physG_robust"]].mean(axis=1)
    method_cols.append("phys_blend")

    results = {}
    for c in method_cols:
        rmse = float(np.sqrt(np.mean((pred_tgt[c] - pred_tgt["TVT"]) ** 2)))
        results[c] = rmse
        print(f"  {c}: pooled RMSE = {rmse:.4f}")

    # === Save full predictions for blend later ===
    pred_df[["well_id", "row_idx", "MD", "physE_MD", "physF_Z", "physG_robust", "physH_anchor"]].to_parquet(
        OUT_DIR / "phys_full.parquet", index=False
    )
    pred_tgt[["well_id", "row_idx", "TVT"] + method_cols].to_parquet(
        OUT_DIR / "oof_phys.parquet", index=False
    )

    # per-well RMSE summary (phys_blend)
    pwr = pred_tgt.groupby("well_id", group_keys=False).apply(
        lambda d: np.sqrt(np.mean((d["phys_blend"] - d["TVT"]) ** 2))
    )
    print(f"\nphys_blend per-well RMSE: med={pwr.median():.3f}, p75={pwr.quantile(0.75):.3f}, max={pwr.max():.3f}")

    summary = {
        "exp": "exp029b_physical_lateral",
        "results": results,
        "anchor_baseline_rmse": 15.9099,
        "phys_blend_per_well": {
            "median": float(pwr.median()),
            "p75": float(pwr.quantile(0.75)),
            "p90": float(pwr.quantile(0.9)),
            "max": float(pwr.max()),
            "n_gt20": int((pwr > 20).sum()),
        },
        "wall_time_sec": float(time.time() - t0),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nTime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
