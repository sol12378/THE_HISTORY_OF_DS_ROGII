#!/usr/bin/env python3
"""exp028b — horizonの「Z相対オフセット」を真のシグナルとして測定。

R²=0.997の正体は「horizon≈Z+定数」の自明な部分が支配的のはず。
本当の有用情報は (horizon - Z) の予測精度。これを測る。

加えて、最も予測しやすい horizon を1つ使い TVT 予測へ投入したときの
CV変化を簡易計測する（OOFリーク無し）。
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
PROCESSED_TRAIN = ROOT / "data" / "processed" / "train_base_v001.parquet"
FOLDS_CSV = ROOT / "data" / "folds" / "folds_group_well_v001.csv"
OUT_DIR = ROOT / "experiments" / "exp028b_horizon_resid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def load_horizons() -> pd.DataFrame:
    rows = []
    for csv_path in sorted(RAW_TRAIN.glob("*__horizontal_well.csv")):
        well_id = csv_path.name.split("__")[0]
        df = pd.read_csv(csv_path)
        df["well_id"] = well_id
        df["row_idx"] = np.arange(len(df))
        rows.append(df[["well_id", "row_idx", *HORIZONS]])
    return pd.concat(rows, ignore_index=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    g = df.groupby("well_id", sort=False)
    df["GR_roll30_mean"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["GR_roll30_std"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=5).std()).fillna(0)
    df["GR_roll100_mean"] = g["GR"].transform(lambda x: x.rolling(100, min_periods=1).mean())
    df["dZ_dMD"] = (g["Z"].diff().fillna(0) / g["MD"].diff().replace(0, np.nan)).fillna(0)
    df["MD_norm"] = df["MD"] - g["MD"].transform("min")
    df["Z_norm"] = df["Z"] - g["Z"].transform("first")
    return df


def fit_oof(df: pd.DataFrame, feats: list[str], target: np.ndarray, mask_valid: np.ndarray) -> tuple[np.ndarray, dict]:
    """指定特徴量・指定マスク内で5-fold OOF。target が NaN の行は除外。"""
    oof = np.full(len(df), np.nan)
    fold_info = {}
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
    for fold in sorted(df["fold"].unique()):
        va = df["fold"].eq(fold).to_numpy() & mask_valid
        tr = (~df["fold"].eq(fold).to_numpy()) & mask_valid
        if tr.sum() == 0 or va.sum() == 0:
            continue
        dtrain = lgb.Dataset(df.loc[tr, feats], label=target[tr])
        dvalid = lgb.Dataset(df.loc[va, feats], label=target[va])
        model = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(25), lgb.log_evaluation(0)],
        )
        pred = model.predict(df.loc[va, feats], num_iteration=model.best_iteration)
        oof[va] = pred
        rmse = float(np.sqrt(np.mean((pred - target[va]) ** 2)))
        fold_info[int(fold)] = {"rmse": rmse, "best_iter": int(model.best_iteration), "n_val": int(va.sum())}
    return oof, fold_info


def main() -> None:
    t0 = time.time()
    print("== exp028b_horizon_resid ==")

    base = pd.read_parquet(PROCESSED_TRAIN)
    base = base[base["split"] == "train"].copy()
    hz = load_horizons()
    df = base.merge(hz, on=["well_id", "row_idx"], how="left")
    df = build_features(df)
    folds = pd.read_csv(FOLDS_CSV).drop_duplicates(subset=["well_id"])[["well_id", "fold"]]
    df = df.merge(folds, on="well_id", how="left").dropna(subset=["fold"]).copy()
    df["fold"] = df["fold"].astype(int)

    FEATS = [
        "MD", "X", "Y", "Z", "GR",
        "GR_roll30_mean", "GR_roll30_std", "GR_roll100_mean",
        "dZ_dMD", "MD_norm", "Z_norm",
        "delta_MD_from_PS", "delta_Z_from_PS",
        "known_length", "hidden_length",
    ]
    FEATS = [f for f in FEATS if f in df.columns]

    # ---- horizon absolute Z (任務1) ----
    # ---- horizon - Z 残差 (任務2: 真のシグナル) ----
    print("\n[1] Horizon absolute Z vs (horizon - Z) residual prediction")
    summary = {}
    horizon_oof = {}
    for h in HORIZONS:
        y_abs = df[h].astype(float).to_numpy()
        mask_v = ~np.isnan(y_abs)
        if mask_v.sum() == 0:
            print(f"  {h}: completely missing, skip")
            continue
        z = df["Z"].astype(float).to_numpy()
        y_resid = y_abs - z

        oof_abs, _ = fit_oof(df, FEATS, y_abs, mask_v)
        oof_res, _ = fit_oof(df, FEATS, y_resid, mask_v)

        # baseline: y_abs ~ Z + per-well-fixed offset (no model)
        df_h = pd.DataFrame({"well_id": df["well_id"], "y": y_resid, "fold": df["fold"]})
        df_h = df_h[mask_v].copy()
        # leave-one-well-out is heavy. use fold-level mean instead:
        # baseline_resid_pred = train-fold-mean of y_resid per well — but well is fold-disjoint,
        # so just use global train mean per fold (very crude).
        baseline_pred = np.full(mask_v.sum(), np.nan)
        for f in sorted(df_h["fold"].unique()):
            tr = df_h["fold"] != f
            va = df_h["fold"] == f
            baseline_pred[va.to_numpy()] = df_h.loc[tr, "y"].mean()

        def rmse(p, y): return float(np.sqrt(np.mean((p - y) ** 2)))
        def r2(p, y):
            ss_r = np.sum((p - y) ** 2); ss_t = np.sum((y - y.mean()) ** 2)
            return float(1 - ss_r / ss_t) if ss_t > 0 else float("nan")

        rmse_abs = rmse(oof_abs[mask_v], y_abs[mask_v])
        rmse_res = rmse(oof_res[mask_v], y_resid[mask_v])
        rmse_base = rmse(baseline_pred, y_resid[mask_v])
        r2_abs = r2(oof_abs[mask_v], y_abs[mask_v])
        r2_res = r2(oof_res[mask_v], y_resid[mask_v])

        summary[h] = {
            "n_valid": int(mask_v.sum()),
            "abs_rmse": rmse_abs,
            "abs_r2": r2_abs,
            "abs_y_std": float(y_abs[mask_v].std()),
            "resid_rmse": rmse_res,
            "resid_r2": r2_res,
            "resid_y_std": float(y_resid[mask_v].std()),
            "resid_baseline_rmse": rmse_base,
            "improvement_over_baseline": rmse_base - rmse_res,
        }
        horizon_oof[h] = oof_res

        print(f"  {h}: |Z|={rmse_abs:6.2f} (R²={r2_abs:.4f}, std={y_abs[mask_v].std():6.1f}) | "
              f"resid={rmse_res:6.2f} (R²={r2_res:.4f}, std={y_resid[mask_v].std():6.2f}, baseline={rmse_base:.2f})")

    # ---- 任務3: TVT予測に予測horizonを足したときのCV変化 ----
    print("\n[2] TVT prediction: baseline vs +predicted_horizon_residuals")

    # target rows のみ
    tgt = df[df["is_target"] == True].copy() if "is_target" in df.columns else df[df["TVT"].notna()].copy()
    # TVT - last_known_TVT を学習
    if "last_known_TVT" not in tgt.columns:
        print("  WARN: last_known_TVT missing, skip TVT eval")
    else:
        tgt["y_delta"] = tgt["TVT"].astype(float) - tgt["last_known_TVT"].astype(float)
        BASE_FEATS = [
            "MD", "X", "Y", "Z", "GR",
            "GR_roll30_mean", "GR_roll30_std",
            "dZ_dMD", "delta_MD_from_PS", "delta_Z_from_PS", "delta_X_from_PS", "delta_Y_from_PS",
            "post_ps_step", "row_frac", "known_length", "hidden_length", "MD_norm",
        ]
        BASE_FEATS = [f for f in BASE_FEATS if f in tgt.columns]
        # +horizon版（予測値、OOFリーク無し）
        df_idx = df.set_index(["well_id", "row_idx"]) if not df.index.equals(pd.RangeIndex(len(df))) else df
        for h, oof in horizon_oof.items():
            df_h = pd.DataFrame({"well_id": df["well_id"], "row_idx": df["row_idx"], f"{h}_resid_oof": oof})
            tgt = tgt.merge(df_h, on=["well_id", "row_idx"], how="left")
        HZ_FEATS = [f"{h}_resid_oof" for h in horizon_oof]
        FULL_FEATS = BASE_FEATS + HZ_FEATS

        def cv_score(feats):
            oof = np.zeros(len(tgt))
            y = tgt["y_delta"].astype(float).to_numpy()
            for fold in sorted(tgt["fold"].unique()):
                va = tgt["fold"].eq(fold).to_numpy()
                tr = ~va
                dtrain = lgb.Dataset(tgt.loc[tr, feats], label=y[tr])
                dvalid = lgb.Dataset(tgt.loc[va, feats], label=y[va])
                model = lgb.train(
                    dict(objective="regression", metric="rmse", learning_rate=0.05,
                         num_leaves=63, min_data_in_leaf=50, feature_fraction=0.9,
                         bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
                         verbosity=-1, seed=42, num_threads=8),
                    dtrain, num_boost_round=800, valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
                )
                oof[va] = model.predict(tgt.loc[va, feats], num_iteration=model.best_iteration)
            # convert delta -> TVT
            pred_tvt = oof + tgt["last_known_TVT"].astype(float).to_numpy()
            true_tvt = tgt["TVT"].astype(float).to_numpy()
            rmse = float(np.sqrt(np.mean((pred_tvt - true_tvt) ** 2)))
            return rmse, oof

        rmse_base, _ = cv_score(BASE_FEATS)
        rmse_full, _ = cv_score(FULL_FEATS)
        print(f"  TVT-CV baseline (n_feat={len(BASE_FEATS)}): {rmse_base:.4f}")
        print(f"  TVT-CV +horizons (n_feat={len(FULL_FEATS)}): {rmse_full:.4f}")
        print(f"  delta: {rmse_full - rmse_base:+.4f}")
        summary["TVT_cv"] = {
            "baseline_rmse": rmse_base,
            "with_horizon_rmse": rmse_full,
            "delta": rmse_full - rmse_base,
            "n_base_feats": len(BASE_FEATS),
            "n_hz_feats": len(HZ_FEATS),
        }

    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {OUT_DIR}/")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
