#!/usr/bin/env python3
"""make_folds_strat — StratifiedGroupKFold by (azimuth, median TVT, spatial cluster).

Niccoli助言: train/test wellは空間的に交互配置(補間問題)。GroupKFold(well_id)は
過度に悲観。azimuth signed(updip/downdip)、median TVT、空間位置で層化したGroupKFoldを
作る。各fold内のwellがtest分布に近くなる。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "train_base_v001.parquet"
OUT_PATH = ROOT / "data" / "folds" / "folds_strat_v001.csv"
N_SPLITS = 5
SEED = 42


def build_well_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df["split"] == "train"].groupby("well_id", sort=False).agg(
        n_rows=("row_idx", "size"),
        target_rows=("is_target", "sum"),
        X_mean=("X", "mean"),
        Y_mean=("Y", "mean"),
        Z_mean=("Z", "mean"),
        TVT_median=("TVT", "median"),
        delta_X=("delta_X_from_PS", "last"),
        delta_Y=("delta_Y_from_PS", "last"),
        delta_Z=("delta_Z_from_PS", "last"),
    ).reset_index()
    g["azim_rad"] = np.arctan2(g["delta_Y"], g["delta_X"])
    g["azim_deg"] = np.degrees(g["azim_rad"])
    g["updip_sign"] = np.sign(g["delta_Z"]).astype(int)  # +1 updip, -1 downdip, 0 flat
    return g


def stratify_label(g: pd.DataFrame, n_clusters: int = 6) -> np.ndarray:
    # 1) 空間クラスタ (X,Y kmeans)
    coords = g[["X_mean", "Y_mean"]].to_numpy()
    coords_n = (coords - coords.mean(0)) / (coords.std(0) + 1e-9)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=SEED).fit(coords_n)
    spatial = km.labels_

    # 2) TVT 4-quartile
    tvt_q = pd.qcut(g["TVT_median"], q=4, labels=False, duplicates="drop")
    # 3) azimuth 4-sector
    azim_q = pd.cut(
        g["azim_deg"],
        bins=[-180, -90, 0, 90, 180],
        labels=False, include_lowest=True,
    )
    # 4) updip vs downdip (3-way)
    udip = (g["updip_sign"] + 1).astype(int)  # 0,1,2

    # combine into composite stratification label, but keep number of strata small
    # so that StratifiedKFold can split. Use spatial*tvt_q*udip (skip azim to avoid sparse)
    label = (spatial.astype(int) * 12) + (tvt_q.astype(int) * 3) + udip
    return label, dict(spatial=spatial, tvt_q=tvt_q, azim_q=azim_q, udip=udip)


def assign_folds(g: pd.DataFrame, label: np.ndarray) -> np.ndarray:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = np.full(len(g), -1, dtype=int)
    # well単位の入力なのでwellごとに直接foldを付ける
    # ダミーx (kfoldが必要)
    x_dummy = np.arange(len(g)).reshape(-1, 1)
    # 稀少クラスは fold をmergeする必要があるが StratifiedKFold が自動でやってくれる
    for f, (_, va_idx) in enumerate(skf.split(x_dummy, label)):
        folds[va_idx] = f
    return folds


def main() -> None:
    print("Loading processed train parquet...")
    df = pd.read_parquet(PROCESSED)
    g = build_well_summary(df)
    print(f"wells: {len(g)}")

    label, parts = stratify_label(g)
    folds = assign_folds(g, label)
    g["fold"] = folds
    g["split"] = "train"
    g["spatial_cluster"] = parts["spatial"]
    g["tvt_q"] = parts["tvt_q"].astype(int)
    g["azim_q"] = parts["azim_q"].astype(int)
    g["updip"] = parts["udip"]

    # 確認: fold内のtarget rows・azim・TVT分布
    print("\nFold balance:")
    print(g.groupby("fold").agg(
        n_wells=("well_id", "count"),
        target_rows=("target_rows", "sum"),
        TVT_mean=("TVT_median", "mean"),
        delta_Z_mean=("delta_Z", "mean"),
    ))

    out_cols = ["split", "well_id", "fold", "target_rows",
                "spatial_cluster", "tvt_q", "azim_q", "updip",
                "X_mean", "Y_mean", "TVT_median", "delta_Z", "azim_deg"]
    g[out_cols].to_csv(OUT_PATH, index=False)
    meta = {
        "strategy": "StratifiedGroupKFold(spatial_kmeans6 x tvt_q4 x updip3)",
        "n_splits": N_SPLITS,
        "n_wells": int(len(g)),
        "seed": SEED,
        "fold_distribution": g.groupby("fold").agg(
            n_wells=("well_id", "count"),
            target_rows=("target_rows", "sum"),
        ).reset_index().to_dict(orient="records"),
    }
    OUT_PATH.with_suffix(".csv.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
