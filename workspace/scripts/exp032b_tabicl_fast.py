#!/usr/bin/env python3
"""exp032b: TabICL — fast version (small context, predict only target rows).

context=5000 per fold subsample, predict ALL validation target rows (full OOF).
TabICLは regressor として ICL ベース。
推論時間: ~5-20分予測か (要計測)。
"""
from __future__ import annotations
import json, time, gc
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ID = "exp032b_tabicl_fast"
OUT_DIR = ROOT / "experiments" / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_TRAIN = ROOT / "data" / "processed" / "train_base_v001.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "test_base_v001.parquet"
FOLDS_CSV = ROOT / "data" / "folds" / "folds_group_well_v001.csv"

CONTEXT_SUBSAMPLE = 5000
SEED = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    g = df.groupby("well_id", sort=False)
    df["GR_roll30"] = g["GR"].transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["GR_roll100"] = g["GR"].transform(lambda x: x.rolling(100, min_periods=1).mean())
    df["dZ_dMD"] = (g["Z"].diff() / g["MD"].diff().replace(0, np.nan)).fillna(0)
    return df


def main():
    t0 = time.time()
    print(f"== {EXP_ID} ==")

    tr = pd.read_parquet(PROCESSED_TRAIN)
    te = pd.read_parquet(PROCESSED_TEST)
    folds = pd.read_csv(FOLDS_CSV).drop_duplicates(subset=["well_id"])[["well_id", "fold"]]

    tr = tr[tr["split"] == "train"].copy()
    tr = build_features(tr)
    te = build_features(te)
    tr = tr.merge(folds, on="well_id", how="left").dropna(subset=["fold"]).copy()
    tr["fold"] = tr["fold"].astype(int)

    FEATS = ["MD", "X", "Y", "Z", "GR", "GR_roll30", "GR_roll100", "dZ_dMD",
             "delta_MD_from_PS", "delta_Z_from_PS",
             "known_length", "hidden_length", "row_frac", "post_ps_step"]
    FEATS = [f for f in FEATS if f in tr.columns]

    # NaN除去 (TabICLは NaN を受け付けない)
    for col in FEATS:
        if tr[col].isna().any():
            med = tr[col].median()
            tr[col] = tr[col].fillna(med)
            te[col] = te[col].fillna(med)

    tr_t = tr[tr["is_target"].astype(bool)].copy()
    tr_t["y_delta"] = tr_t["TVT"].astype(float) - tr_t["last_known_TVT"].astype(float)
    te_t = te[te["is_target"].astype(bool)].copy()
    print(f"target rows train={len(tr_t)} test={len(te_t)}")

    from tabicl import TabICLRegressor
    oof = np.full(len(tr_t), np.nan, dtype=float)
    test_pred = np.zeros(len(te_t), dtype=float)
    fold_results = []
    rng = np.random.default_rng(SEED)

    for fold in sorted(tr_t["fold"].unique()):
        print(f"\n=== fold {fold} ===")
        va_mask = tr_t["fold"].eq(fold).to_numpy()
        tr_mask = ~va_mask
        idx_tr = np.where(tr_mask)[0]
        sel = rng.choice(idx_tr, size=min(CONTEXT_SUBSAMPLE, len(idx_tr)), replace=False)
        X_ctx = tr_t.iloc[sel][FEATS].astype(float).values
        y_ctx = tr_t.iloc[sel]["y_delta"].astype(float).values
        X_va = tr_t.loc[va_mask, FEATS].astype(float).values
        # subsample validation for speed (500 random + full per-well 1 row)
        n_va_full = len(X_va)
        print(f"  context={len(X_ctx)}, valid_full={n_va_full}")

        t1 = time.time()
        model = TabICLRegressor(n_estimators=2, batch_size=2, verbose=False,
                                 random_state=SEED + fold, n_jobs=1)
        model.fit(X_ctx, y_ctx)
        print(f"  fit: {time.time()-t1:.1f}s")

        t2 = time.time()
        # validation predict in chunks
        chunk = 5000
        preds = np.empty(n_va_full)
        for s in range(0, n_va_full, chunk):
            e = min(s + chunk, n_va_full)
            preds[s:e] = model.predict(X_va[s:e], output_type="mean")
        print(f"  predict valid: {time.time()-t2:.1f}s ({n_va_full/((time.time()-t2)+1e-9):.0f} rows/s)")

        oof[va_mask] = preds
        y_va = tr_t.loc[va_mask, "y_delta"].astype(float).values
        rmse_delta = float(np.sqrt(np.mean((preds - y_va) ** 2)))
        true_va_tvt = tr_t.loc[va_mask, "TVT"].astype(float).values
        last_va = tr_t.loc[va_mask, "last_known_TVT"].astype(float).values
        rmse_tvt = float(np.sqrt(np.mean((preds + last_va - true_va_tvt) ** 2)))
        print(f"  fold{fold} delta-RMSE={rmse_delta:.4f}, TVT-RMSE={rmse_tvt:.4f}")

        # test predict
        t3 = time.time()
        X_te = te_t[FEATS].astype(float).values
        ntest = len(X_te)
        preds_te = np.empty(ntest)
        for s in range(0, ntest, chunk):
            e = min(s + chunk, ntest)
            preds_te[s:e] = model.predict(X_te[s:e], output_type="mean")
        test_pred += preds_te / 5.0
        print(f"  predict test: {time.time()-t3:.1f}s")
        fold_results.append({"fold": int(fold), "delta_rmse": rmse_delta, "tvt_rmse": rmse_tvt})
        del model; gc.collect()

    oof_tvt = oof + tr_t["last_known_TVT"].astype(float).to_numpy()
    true_tvt = tr_t["TVT"].astype(float).to_numpy()
    cv_tvt = float(np.sqrt(np.mean((oof_tvt - true_tvt) ** 2)))
    print(f"\nTabICL 5-fold TVT CV = {cv_tvt:.4f}")

    tr_t["pred_tvt"] = oof_tvt
    tr_t[["well_id", "row_idx", "id", "TVT", "last_known_TVT", "pred_tvt"]].to_csv(
        OUT_DIR / "oof.csv", index=False)

    te_t["pred_tvt"] = test_pred + te_t["last_known_TVT"].astype(float)
    te_t[["well_id", "row_idx", "id", "last_known_TVT", "pred_tvt"]].to_csv(
        OUT_DIR / "test_pred.csv", index=False)

    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(te_t[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
                                on="id", how="left", validate="one_to_one")
    sub.to_csv(OUT_DIR / "submission.csv", index=False)

    (OUT_DIR / "result.json").write_text(json.dumps({
        "exp_id": EXP_ID, "cv_tvt_rmse": cv_tvt,
        "fold_results": fold_results, "context_subsample": CONTEXT_SUBSAMPLE,
        "features": FEATS, "wall_time_sec": float(time.time() - t0),
    }, indent=2, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
