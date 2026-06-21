#!/usr/bin/env python3
"""exp032: TabICL (foundation tabular ICL) for TVT-delta regression.

kojimar v10 stackと同様のTabICL利用。我々のgeom特徴(exp014相当)とdelta_TVTを使い、
GroupKFold by well_id で5-fold OOF。Blendに加える diversification 要素。

TabICL は in-context learning なので大規模contextが必要だが、メモリ制約から
fold内のtrain rowsを subsample(20k) して context にする。

ライト目の試運転: まず subset (10wells) で動作確認、その後full 773wells。
"""
from __future__ import annotations
import json, time, gc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "exp032_tabicl"
OUT_DIR = ROOT / "experiments" / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_TRAIN = ROOT / "data" / "processed" / "train_base_v001.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "test_base_v001.parquet"
FOLDS_CSV = ROOT / "data" / "folds" / "folds_group_well_v001.csv"

CONTEXT_SUBSAMPLE = 20000   # rows for ICL context per fold
SEED = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    g = df.groupby("well_id", sort=False)
    df["GR_roll30"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["GR_roll30s"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=5).std()).fillna(0)
    df["GR_roll100"] = g["GR"].transform(lambda x: x.rolling(100, min_periods=1).mean())
    df["dZ_dMD"] = (g["Z"].diff() / g["MD"].diff().replace(0, np.nan)).fillna(0)
    return df


def main():
    import torch
    t0 = time.time()
    print(f"== {EXP_ID} ==")
    print(f"torch device available: cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}")

    print("Loading data...")
    tr = pd.read_parquet(PROCESSED_TRAIN)
    te = pd.read_parquet(PROCESSED_TEST)
    folds = pd.read_csv(FOLDS_CSV)
    folds = folds.drop_duplicates(subset=["well_id"])[["well_id", "fold"]]

    tr = tr[tr["split"] == "train"].copy()
    tr = build_features(tr)
    te = build_features(te)
    tr = tr.merge(folds, on="well_id", how="left").dropna(subset=["fold"]).copy()
    tr["fold"] = tr["fold"].astype(int)

    FEATS = ["MD", "X", "Y", "Z", "GR",
             "GR_roll30", "GR_roll30s", "GR_roll100", "dZ_dMD",
             "delta_MD_from_PS", "delta_X_from_PS", "delta_Y_from_PS", "delta_Z_from_PS",
             "last_known_TVT", "known_length", "hidden_length", "row_frac", "post_ps_step"]
    FEATS = [f for f in FEATS if f in tr.columns]
    print(f"features: {FEATS}")

    # target = TVT - last_known_TVT (delta)
    tr_t = tr[tr["is_target"].astype(bool)].copy()
    tr_t["y_delta"] = tr_t["TVT"].astype(float) - tr_t["last_known_TVT"].astype(float)
    te_t = te[te["is_target"].astype(bool)].copy()
    print(f"train target rows={len(tr_t)}, test target rows={len(te_t)}")

    # Import TabICL after env check
    from tabicl import TabICLRegressor

    oof = np.zeros(len(tr_t), dtype=float)
    test_pred = np.zeros(len(te_t), dtype=float)
    fold_results = []

    rng = np.random.default_rng(SEED)
    for fold in sorted(tr_t["fold"].unique()):
        print(f"\n=== fold {fold} ===")
        va_mask = tr_t["fold"].eq(fold).to_numpy()
        tr_mask = ~va_mask
        n_tr = int(tr_mask.sum())
        n_va = int(va_mask.sum())
        print(f"  train rows: {n_tr}, valid rows: {n_va}")

        # Subsample train rows for context (memory)
        if n_tr > CONTEXT_SUBSAMPLE:
            idx_tr = np.where(tr_mask)[0]
            sel = rng.choice(idx_tr, size=CONTEXT_SUBSAMPLE, replace=False)
            X_tr = tr_t.iloc[sel][FEATS].astype(float).values
            y_tr = tr_t.iloc[sel]["y_delta"].astype(float).values
        else:
            X_tr = tr_t.loc[tr_mask, FEATS].astype(float).values
            y_tr = tr_t.loc[tr_mask, "y_delta"].astype(float).values
        X_va = tr_t.loc[va_mask, FEATS].astype(float).values
        y_va = tr_t.loc[va_mask, "y_delta"].astype(float).values

        t1 = time.time()
        model = TabICLRegressor(
            n_estimators=4, batch_size=4, verbose=True,
            random_state=SEED + fold, n_jobs=1,
        )
        print(f"  fitting TabICL ({len(X_tr)} context rows)...")
        model.fit(X_tr, y_tr)
        print(f"  predicting valid ({len(X_va)} rows)...")
        pred_va = model.predict(X_va, output_type="mean")
        rmse_delta = float(np.sqrt(np.mean((pred_va - y_va) ** 2)))
        # convert to TVT
        pred_va_tvt = pred_va + tr_t.loc[va_mask, "last_known_TVT"].astype(float).values
        true_va_tvt = tr_t.loc[va_mask, "TVT"].astype(float).values
        rmse_tvt = float(np.sqrt(np.mean((pred_va_tvt - true_va_tvt) ** 2)))
        print(f"  fold{fold} delta-RMSE={rmse_delta:.4f}, TVT-RMSE={rmse_tvt:.4f}, time={time.time()-t1:.1f}s")

        oof[va_mask] = pred_va

        # test prediction (accumulate, will average across folds)
        print(f"  predicting test ({len(te_t)} rows)...")
        X_te = te_t[FEATS].astype(float).values
        pred_te = model.predict(X_te, output_type="mean")
        test_pred += pred_te / 5.0  # 5-fold avg

        fold_results.append({"fold": fold, "delta_rmse": rmse_delta, "tvt_rmse": rmse_tvt})
        del model; gc.collect()

    # Pooled CV
    oof_tvt = oof + tr_t["last_known_TVT"].astype(float).to_numpy()
    true_tvt = tr_t["TVT"].astype(float).to_numpy()
    cv_tvt = float(np.sqrt(np.mean((oof_tvt - true_tvt) ** 2)))
    print(f"\n=== TabICL 5-fold CV: TVT-RMSE = {cv_tvt:.4f} ===")

    # Save OOF + test
    tr_t["tabicl_oof_delta"] = oof
    tr_t[["well_id", "row_idx", "id", "TVT", "last_known_TVT", "tabicl_oof_delta"]].to_csv(
        OUT_DIR / "oof.csv", index=False
    )
    te_t["tabicl_pred_delta"] = test_pred
    te_t["tabicl_pred_tvt"] = test_pred + te_t["last_known_TVT"].astype(float)
    te_t[["well_id", "row_idx", "id", "last_known_TVT", "tabicl_pred_delta", "tabicl_pred_tvt"]].to_csv(
        OUT_DIR / "test_pred.csv", index=False
    )

    # Submission (TabICL alone)
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(te_t[["id", "tabicl_pred_tvt"]].rename(columns={"tabicl_pred_tvt": "tvt"}),
                                on="id", how="left", validate="one_to_one")
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    summary = {
        "exp_id": EXP_ID,
        "cv_tvt_rmse": cv_tvt,
        "fold_results": fold_results,
        "context_subsample": CONTEXT_SUBSAMPLE,
        "features": FEATS,
        "wall_time_sec": float(time.time() - t0),
    }
    (OUT_DIR / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
