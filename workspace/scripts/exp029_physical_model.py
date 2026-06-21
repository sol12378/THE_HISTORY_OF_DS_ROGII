#!/usr/bin/env python3
"""exp029_physical_model — Sunny相当の物理モデル(dip+fault offsetフィット型).

Sunny出力の特徴: TVTが超滑らか (0.01ft/row刻みの線形)。これは
US特許10318662型の dip+fault offset フィッティング = 構造の物理パラメトリックモデル。

我々のexp014(Group F)はLightGBMでdelta学習しており非線形。Sunnyは直接パラメトリック
線形外挿で「形状アンカー」になる。誤差相関が低く blendに効く可能性。

実装する物理モデル variants:
- physA: TVT-MD 線形外挿 (last K rows)
- physB: TVT-Z 線形外挿 (build section)
- physC: TVT-Z 局所線形 (last K rows)
- physD: TVT-(Z, MD, X, Y) 平面外挿 (build section)
- blend: equal mean

全773 wellで well-disjoint OOF (foldの概念なし、各wellは自分のbuild→hidden外挿のみ)
→ 完全 leak-free。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "train_base_v001.parquet"
OUT_DIR = ROOT / "experiments" / "exp029_physical_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# パラメータ
K_LOCAL = 200  # 局所dipに使う直近行数


def fit_physical_well(df_well: pd.DataFrame) -> pd.DataFrame:
    """1 wellに対し物理モデル予測を計算。

    df_well: well行を MD 昇順で持つDataFrame。columns:
        MD, X, Y, Z, GR, TVT (known区間のみ非NaN), TVT_input (PS前のみ非NaN)
    return: 各row予測値 + meta
    """
    df_well = df_well.sort_values("MD").reset_index(drop=True)
    known = df_well["TVT_input"].notna() & df_well["TVT"].notna()
    n = len(df_well)
    if known.sum() < 30:
        # known が極端に少ない wellはanchor extension
        last_tvt = df_well.loc[known, "TVT"].iloc[-1] if known.any() else 0
        out = pd.DataFrame({
            "physA": [last_tvt] * n,
            "physB": [last_tvt] * n,
            "physC": [last_tvt] * n,
            "physD": [last_tvt] * n,
        })
        out["MD"] = df_well["MD"].values
        out["is_hidden"] = ~known.values
        return out

    known_idx = np.where(known.values)[0]
    last_known_idx = known_idx[-1]
    md_arr = df_well["MD"].to_numpy(dtype=float)
    z_arr = df_well["Z"].to_numpy(dtype=float)
    x_arr = df_well["X"].to_numpy(dtype=float)
    y_arr = df_well["Y"].to_numpy(dtype=float)
    tvt_arr = df_well["TVT"].to_numpy(dtype=float)

    # === Method A: TVT-MD 局所線形 (last K) ===
    use = known_idx[-min(K_LOCAL, len(known_idx)):]
    a_md, a_int = np.polyfit(md_arr[use], tvt_arr[use], 1)
    physA = a_int + a_md * md_arr

    # === Method B: TVT-Z 全体線形 (build) ===
    if z_arr[known].std() > 1e-6:
        b_md, b_int = np.polyfit(z_arr[known], tvt_arr[known], 1)
        physB = b_int + b_md * z_arr
    else:
        physB = np.full(n, tvt_arr[last_known_idx])

    # === Method C: TVT-Z 局所線形 (last K) ===
    use_z = z_arr[use]
    if use_z.std() > 1e-6:
        c_md, c_int = np.polyfit(use_z, tvt_arr[use], 1)
        physC = c_int + c_md * z_arr
    else:
        physC = np.full(n, tvt_arr[last_known_idx])

    # === Method D: TVT ~ (Z, X, Y) 平面 (build all, ridge)===
    Xd = np.stack([z_arr[known], x_arr[known], y_arr[known], np.ones(known.sum())], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(Xd, tvt_arr[known], rcond=None)
        Xd_pred = np.stack([z_arr, x_arr, y_arr, np.ones(n)], axis=1)
        physD = Xd_pred @ coef
    except Exception:
        physD = physB.copy()

    out = pd.DataFrame({
        "MD": md_arr,
        "physA_tvtMD": physA,
        "physB_tvtZ_all": physB,
        "physC_tvtZ_local": physC,
        "physD_plane": physD,
        "is_hidden": ~known.values,
    })
    return out


def main() -> None:
    t0 = time.time()
    print("== exp029_physical_model ==")
    df = pd.read_parquet(PROCESSED)
    df = df[df["split"] == "train"].copy()
    print(f"rows={len(df)}, wells={df['well_id'].nunique()}")

    # well毎に物理モデル予測
    preds = []
    for wid, g in df.groupby("well_id", sort=False):
        out = fit_physical_well(g)
        out["well_id"] = wid
        out["row_idx"] = g["row_idx"].values
        out["TVT_true"] = g["TVT"].values
        preds.append(out)
    pred_df = pd.concat(preds, ignore_index=True)
    print(f"prediction rows: {len(pred_df)}")

    # === Evaluate on target rows (hidden) ===
    # target rows = TVT_input NaN & TVT not NaN
    tgt_mask = df["TVT_input"].isna() & df["TVT"].notna()
    df_tgt = df.loc[tgt_mask, ["well_id", "row_idx", "TVT"]].copy()
    pred_tgt = pred_df.merge(df_tgt, on=["well_id", "row_idx"], how="inner")
    print(f"target rows: {len(pred_tgt)}")

    # blend equal mean
    method_cols = ["physA_tvtMD", "physB_tvtZ_all", "physC_tvtZ_local", "physD_plane"]
    pred_tgt["phys_mean"] = pred_tgt[method_cols].mean(axis=1)
    method_cols.append("phys_mean")

    # pooled RMSE per method
    results = {}
    for c in method_cols:
        rmse = float(np.sqrt(np.mean((pred_tgt[c] - pred_tgt["TVT"]) ** 2)))
        results[c] = {"pooled_rmse": rmse}
        print(f"  {c}: pooled RMSE = {rmse:.4f}")

    # per-well RMSE distribution for blend
    pwr = pred_tgt.groupby("well_id").apply(
        lambda d: np.sqrt(np.mean((d["phys_mean"] - d["TVT"]) ** 2))
    )
    print(f"\nphys_mean per-well RMSE: median={pwr.median():.3f}, p25={pwr.quantile(0.25):.3f}, p75={pwr.quantile(0.75):.3f}, max={pwr.max():.3f}")

    # 比較: 既存 anchor baseline (last_known_TVT extension)
    df_a = df[tgt_mask][["well_id", "row_idx", "TVT", "last_known_TVT"]].copy()
    anchor_rmse = float(np.sqrt(np.mean((df_a["last_known_TVT"] - df_a["TVT"]) ** 2)))
    print(f"\nanchor baseline pooled RMSE: {anchor_rmse:.4f}")

    # save predictions
    pred_tgt[["well_id", "row_idx", "TVT"] + method_cols].to_parquet(
        OUT_DIR / "oof_phys.parquet", index=False
    )

    # Also export full prediction (incl. hidden) for blend use later
    pred_df[["well_id", "row_idx", "MD"] + method_cols[:-1]].to_parquet(
        OUT_DIR / "phys_full.parquet", index=False
    )

    summary = {
        "exp": "exp029_physical_model",
        "n_wells": int(df["well_id"].nunique()),
        "n_target_rows": int(len(pred_tgt)),
        "anchor_baseline_rmse": anchor_rmse,
        "methods": results,
        "phys_mean_per_well_rmse": {
            "median": float(pwr.median()),
            "p25": float(pwr.quantile(0.25)),
            "p75": float(pwr.quantile(0.75)),
            "p90": float(pwr.quantile(0.9)),
            "max": float(pwr.max()),
            "n_broken_gt20": int((pwr > 20).sum()),
        },
        "wall_time_sec": float(time.time() - t0),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUT_DIR}/")
    print(f"Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
