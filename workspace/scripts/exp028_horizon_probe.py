#!/usr/bin/env python3
"""exp028_horizon_probe — horizon特徴量の価値を素早く測定する。

A案検証: (MD, X, Y, Z, GR + 派生) から 6本のhorizon (ANCC..BUDA) を
LightGBMで予測可能か。GroupKFold(by well_id) OOFで R²/RMSE を出す。

その後、予測horizonをTVT予測モデルに加えた効果を簡易計測。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN = ROOT / "data" / "raw" / "train"
PROCESSED = ROOT / "data" / "processed" / "train_base_v001.parquet"
FOLDS_CSV = ROOT / "data" / "folds" / "folds_group_well_v001.csv"
OUT_DIR = ROOT / "experiments" / "exp028_horizon_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def load_horizons() -> pd.DataFrame:
    """train全wellからhorizonをロードしてrow_idxとwell_idでひもづける。"""
    rows = []
    for csv_path in sorted(RAW_TRAIN.glob("*__horizontal_well.csv")):
        well_id = csv_path.name.split("__")[0]
        df = pd.read_csv(csv_path)
        df["well_id"] = well_id
        df["row_idx"] = np.arange(len(df))
        rows.append(df[["well_id", "row_idx", *HORIZONS]])
    return pd.concat(rows, ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """軽量な特徴量セット。well内ロール統計を追加。"""
    df = df.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    g = df.groupby("well_id", sort=False)

    df["GR_roll30_mean"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["GR_roll30_std"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=5).std()).fillna(0)
    df["GR_roll100_mean"] = g["GR"].transform(lambda x: x.rolling(100, min_periods=1).mean())
    df["dZ_dMD"] = g["Z"].transform(lambda x: x.diff()).fillna(0) / g["MD"].transform(lambda x: x.diff()).replace(0, np.nan).fillna(1)
    df["dX_dMD"] = g["X"].transform(lambda x: x.diff()).fillna(0) / g["MD"].transform(lambda x: x.diff()).replace(0, np.nan).fillna(1)
    df["dY_dMD"] = g["Y"].transform(lambda x: x.diff()).fillna(0) / g["MD"].transform(lambda x: x.diff()).replace(0, np.nan).fillna(1)
    df["MD_norm"] = df["MD"] - g["MD"].transform("min")
    return df


def main() -> None:
    t0 = time.time()
    print("== exp028_horizon_probe ==")
    print("Step1: load processed base + horizons")
    base = pd.read_parquet(PROCESSED)
    base = base[base["split"] == "train"].copy()  # safety
    hz = load_horizons()
    df = base.merge(hz, on=["well_id", "row_idx"], how="left")
    miss = df[HORIZONS].isna().any(axis=1).sum()
    print(f"  rows={len(df)}, horizon missing rows={miss}")

    print("Step2: build features")
    df = build_features(df)

    folds = pd.read_csv(FOLDS_CSV)
    folds = folds.drop_duplicates(subset=["well_id"])[["well_id", "fold"]]
    df = df.merge(folds, on="well_id", how="left")
    if df["fold"].isna().any():
        n = df["fold"].isna().sum()
        print(f"  WARN: {n} rows missing fold, dropping")
        df = df.dropna(subset=["fold"]).copy()
    df["fold"] = df["fold"].astype(int)

    FEATS = [
        "MD", "X", "Y", "Z", "GR",
        "GR_roll30_mean", "GR_roll30_std", "GR_roll100_mean",
        "dZ_dMD", "dX_dMD", "dY_dMD", "MD_norm",
        "delta_MD_from_PS", "delta_Z_from_PS",
        "known_length", "hidden_length",
    ]
    FEATS = [f for f in FEATS if f in df.columns]
    print(f"  features used: {FEATS}")

    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.07,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        verbosity=-1,
        seed=42,
        num_threads=8,
    )

    print("Step3: per-horizon GroupKFold OOF prediction")
    results = {}
    oof_all = {h: np.full(len(df), np.nan) for h in HORIZONS}

    for h in HORIZONS:
        print(f"\n  --- target = {h} ---")
        y = df[h].astype(float).to_numpy()
        oof = np.full(len(df), np.nan)
        fold_rmses = []
        for fold in sorted(df["fold"].unique()):
            va = df["fold"].eq(fold).to_numpy()
            tr = ~va
            dtrain = lgb.Dataset(df.loc[tr, FEATS], label=y[tr])
            dvalid = lgb.Dataset(df.loc[va, FEATS], label=y[va])
            model = lgb.train(
                params, dtrain, num_boost_round=400,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            pred = model.predict(df.loc[va, FEATS], num_iteration=model.best_iteration)
            oof[va] = pred
            rmse = float(np.sqrt(np.mean((pred - y[va]) ** 2)))
            fold_rmses.append(rmse)
            print(f"    fold{fold}: rmse={rmse:.3f}, best_iter={model.best_iteration}")
        rmse_pooled = float(np.sqrt(np.mean((oof - y) ** 2)))
        ss_res = float(np.sum((oof - y) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        results[h] = {
            "pooled_rmse": rmse_pooled,
            "r2": r2,
            "fold_rmses": fold_rmses,
            "y_std": float(y.std()),
            "y_range": [float(y.min()), float(y.max())],
        }
        oof_all[h] = oof
        print(f"  {h}: pooled RMSE={rmse_pooled:.3f}, R²={r2:.4f}, y_std={y.std():.2f}")

    print("\nStep4: save OOF + summary")
    oof_df = pd.DataFrame({h + "_oof_pred": oof_all[h] for h in HORIZONS})
    oof_df["well_id"] = df["well_id"].values
    oof_df["row_idx"] = df["row_idx"].values
    oof_df["fold"] = df["fold"].values
    oof_df.to_parquet(OUT_DIR / "horizon_oof.parquet", index=False)

    summary = {
        "exp": "exp028_horizon_probe",
        "n_rows": int(len(df)),
        "features": FEATS,
        "results_per_horizon": results,
        "wall_time_sec": float(time.time() - t0),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUT_DIR}/")
    print(f"Total time: {time.time()-t0:.1f}s")

    print("\n== Summary ==")
    print(f"{'horizon':<8} {'RMSE':>8} {'R²':>8} {'y_std':>8}")
    for h in HORIZONS:
        r = results[h]
        print(f"{h:<8} {r['pooled_rmse']:>8.3f} {r['r2']:>8.4f} {r['y_std']:>8.2f}")


if __name__ == "__main__":
    main()
