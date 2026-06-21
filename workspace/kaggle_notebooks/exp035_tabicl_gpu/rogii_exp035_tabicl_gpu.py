"""ROGII exp035 — TabICL (Tabular In-Context Learning) on GPU.

TabICL は事前学習済みタブラーTransformer。fit(X_ctx, y_ctx) は context保存のみで
学習なし、predict(X_query) で in-context inference。
ローカルCPU実行は時間/メモリ制約で頓挫したので、Kaggle T4 GPU で完走させる。

構成:
- features: 我々のgeom特徴量set (SAFE + Group A/B/C/D/F の row-level subset)
- target: delta = TVT - last_known_TVT
- 5-fold GroupKFold by well_id
- context per fold: 15000 rows random subsample
- OOF: 全validation target rows
- Test predict: 14151 rows × 5fold 平均

requirements:
- enable_gpu: true (T4)
- enable_internet: true (tabicl 初回downloadのためHugging Face接続)
"""
from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ── locations ──────────────────────────────────────────────────────────────────
def find_input_dir() -> Path:
    for root in (Path("/kaggle/input"), Path("data/raw"), Path("data")):
        if root.exists():
            hits = list(root.rglob("sample_submission.csv"))
            if hits:
                return hits[0].parent
    raise FileNotFoundError("sample_submission.csv not found")


INPUT_DIR = find_input_dir()
TRAIN_DIR = INPUT_DIR / "train"
TEST_DIR = INPUT_DIR / "test"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

# ── tabicl install ─────────────────────────────────────────────────────────────
def ensure_tabicl():
    try:
        import tabicl  # noqa: F401
        print("tabicl already installed", flush=True)
        return
    except ImportError:
        pass
    print("installing tabicl...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabicl"])
    print("tabicl installed", flush=True)


# ── config ─────────────────────────────────────────────────────────────────────
N_SPLITS = 5
CONTEXT_SUBSAMPLE = 5000    # rows in TabICL context per fold (CPU-friendly)
N_ESTIMATORS = 2            # TabICL ensemble depth (CPU-friendly)
BATCH_SIZE = 2
PREDICT_CHUNK = 4000
SEED = 42
DEVICE = "cpu"              # Kaggle P100 (sm_60) is PyTorch-incompatible

# ── feature reconstruction (replicates exp026 kernel logic for geom features) ──
def well_id_from_path(p: Path) -> str:
    return p.name.split("__", 1)[0]


def natural_key(p: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)]


def build_base_for_file(path: Path, split: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    well_id = well_id_from_path(path)
    df = pd.DataFrame(index=raw.index)
    df["split"] = split
    df["well_id"] = well_id
    df["row_idx"] = raw.index.astype("int64")
    for col in ["MD", "X", "Y", "Z", "GR", "TVT_input", "TVT"]:
        df[col] = raw[col] if col in raw.columns else pd.NA
    df["TVT"] = pd.to_numeric(df["TVT"], errors="coerce")
    df["TVT_input"] = pd.to_numeric(df["TVT_input"], errors="coerce")

    missing = df["TVT_input"].isna()
    ps_idx = int(missing.idxmax()) if missing.any() else len(df)
    n_rows = len(df)
    known_length = ps_idx if missing.any() else n_rows
    hidden_length = n_rows - known_length
    is_target = (df["row_idx"] >= ps_idx) if missing.any() else pd.Series(False, index=df.index)
    anchor_idx = max(known_length - 1, 0)
    anchor = df.loc[anchor_idx, ["MD", "X", "Y", "Z", "TVT_input"]]
    df["id"] = pd.NA
    if split == "test":
        df.loc[is_target, "id"] = df.loc[is_target, "row_idx"].map(lambda r: f"{well_id}_{r}")
    df["is_target"] = is_target.astype(bool)
    df["is_known_tvt"] = df["TVT_input"].notna()
    df["is_gr_missing"] = df["GR"].isna()
    df["n_rows_in_well"] = int(n_rows)
    df["known_length"] = int(known_length)
    df["hidden_length"] = int(hidden_length)
    df["last_known_TVT"] = anchor["TVT_input"]
    df["last_known_MD"] = anchor["MD"]
    df["last_known_Z"] = anchor["Z"]
    df["delta_MD_from_PS"] = df["MD"] - anchor["MD"]
    df["delta_X_from_PS"] = df["X"] - anchor["X"]
    df["delta_Y_from_PS"] = df["Y"] - anchor["Y"]
    df["delta_Z_from_PS"] = df["Z"] - anchor["Z"]
    df["post_ps_step"] = (df["row_idx"] - ps_idx).clip(lower=0)
    df["row_frac"] = df["row_idx"] / max(n_rows - 1, 1)
    return df


def load_base(split_dir: Path, split: str) -> pd.DataFrame:
    paths = sorted(split_dir.glob("*__horizontal_well.csv"), key=natural_key)
    return pd.concat([build_base_for_file(p, split) for p in paths], ignore_index=True)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    g = df.groupby("well_id", sort=False)
    df["GR_roll20"] = g["GR"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["GR_roll50"] = g["GR"].transform(lambda x: x.rolling(50, min_periods=1).mean())
    df["GR_roll100"] = g["GR"].transform(lambda x: x.rolling(100, min_periods=1).mean())
    df["GR_roll20s"] = g["GR"].transform(lambda x: x.rolling(20, min_periods=5).std()).fillna(0)
    df["dZ_dMD"] = (g["Z"].diff() / g["MD"].diff().replace(0, np.nan)).fillna(0)
    df["dX_dMD"] = (g["X"].diff() / g["MD"].diff().replace(0, np.nan)).fillna(0)
    df["dY_dMD"] = (g["Y"].diff() / g["MD"].diff().replace(0, np.nan)).fillna(0)
    return df


def make_folds(train: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
    stats = (train[train["is_target"]].groupby("well_id", as_index=False)
             .agg(target_rows=("row_idx", "size")))
    loads = [0] * n_splits
    fold_of = {}
    for row in stats.sort_values("target_rows", ascending=False).itertuples(index=False):
        f = min(range(n_splits), key=lambda i: loads[i])
        loads[f] += int(row.target_rows)
        fold_of[row.well_id] = f
    return fold_of


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"INPUT_DIR={INPUT_DIR}", flush=True)

    ensure_tabicl()
    from tabicl import TabICLRegressor
    import torch
    print(f"torch.cuda.is_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # ── load + features ──
    print("\n[1/3] loading data + features...", flush=True)
    train = load_base(TRAIN_DIR, "train")
    test = load_base(TEST_DIR, "test")
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    train = add_rolling_features(train)
    test = add_rolling_features(test)
    print(f"  train rows={len(train)}, test rows={len(test)}", flush=True)

    fold_of = make_folds(train)
    train["fold"] = train["well_id"].map(fold_of)
    train_t = train[train["is_target"].astype(bool)].copy()
    test_t = test[test["is_target"].astype(bool)].copy()
    train_t["y_delta"] = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)
    print(f"  train target rows={len(train_t)}, test target rows={len(test_t)}", flush=True)

    FEATS = ["MD", "X", "Y", "Z", "GR",
             "GR_roll20", "GR_roll50", "GR_roll100", "GR_roll20s",
             "dZ_dMD", "dX_dMD", "dY_dMD",
             "delta_MD_from_PS", "delta_X_from_PS", "delta_Y_from_PS", "delta_Z_from_PS",
             "last_known_TVT", "last_known_MD", "last_known_Z",
             "known_length", "hidden_length", "row_frac", "post_ps_step"]
    FEATS = [f for f in FEATS if f in train_t.columns]

    # NaN impute
    for c in FEATS:
        if train_t[c].isna().any() or test_t[c].isna().any():
            med = pd.concat([train_t[c], test_t[c]]).median()
            train_t[c] = train_t[c].fillna(med)
            test_t[c] = test_t[c].fillna(med)
    print(f"  features: {FEATS}", flush=True)

    # ── 5-fold TabICL OOF + test predict ──
    print("\n[2/3] TabICL 5-fold OOF + test predict on GPU...", flush=True)
    oof = np.full(len(train_t), np.nan, dtype=float)
    test_pred_delta = np.zeros(len(test_t), dtype=float)
    fold_results = []
    rng = np.random.default_rng(SEED)

    for fold in sorted(train_t["fold"].unique()):
        fold_t0 = time.time()
        print(f"\n=== fold {fold} ===", flush=True)
        va_mask = train_t["fold"].eq(fold).to_numpy()
        tr_mask = ~va_mask
        idx_tr = np.where(tr_mask)[0]
        sel = rng.choice(idx_tr, size=min(CONTEXT_SUBSAMPLE, len(idx_tr)), replace=False)
        X_ctx = train_t.iloc[sel][FEATS].astype(float).values
        y_ctx = train_t.iloc[sel]["y_delta"].astype(float).values
        X_va = train_t.loc[va_mask, FEATS].astype(float).values
        n_va = len(X_va)
        print(f"  context={len(X_ctx)} rows, validation={n_va} rows", flush=True)

        model = TabICLRegressor(
            n_estimators=N_ESTIMATORS, batch_size=BATCH_SIZE,
            verbose=False, random_state=int(SEED + fold), n_jobs=1,
            device=DEVICE,
        )
        t1 = time.time()
        model.fit(X_ctx, y_ctx)
        print(f"  fit: {time.time()-t1:.1f}s", flush=True)

        # validation predict (chunked)
        t2 = time.time()
        preds_va = np.empty(n_va, dtype=float)
        for s in range(0, n_va, PREDICT_CHUNK):
            e = min(s + PREDICT_CHUNK, n_va)
            preds_va[s:e] = model.predict(X_va[s:e], output_type="mean")
        print(f"  predict val: {time.time()-t2:.1f}s ({n_va/(time.time()-t2+1e-9):.0f} rows/s)", flush=True)
        oof[va_mask] = preds_va

        y_va = train_t.loc[va_mask, "y_delta"].astype(float).values
        last_va = train_t.loc[va_mask, "last_known_TVT"].astype(float).values
        true_va = train_t.loc[va_mask, "TVT"].astype(float).values
        rmse_delta = float(np.sqrt(np.mean((preds_va - y_va) ** 2)))
        rmse_tvt = float(np.sqrt(np.mean((preds_va + last_va - true_va) ** 2)))
        print(f"  fold{fold} delta-RMSE={rmse_delta:.4f}, TVT-RMSE={rmse_tvt:.4f}", flush=True)

        # test predict
        t3 = time.time()
        X_te = test_t[FEATS].astype(float).values
        nt = len(X_te)
        preds_te = np.empty(nt, dtype=float)
        for s in range(0, nt, PREDICT_CHUNK):
            e = min(s + PREDICT_CHUNK, nt)
            preds_te[s:e] = model.predict(X_te[s:e], output_type="mean")
        test_pred_delta += preds_te / N_SPLITS
        print(f"  predict test: {time.time()-t3:.1f}s", flush=True)
        fold_results.append({
            "fold": int(fold), "delta_rmse": rmse_delta, "tvt_rmse": rmse_tvt,
            "fold_time_sec": float(time.time() - fold_t0),
        })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── pooled metrics + outputs ──
    print("\n[3/3] aggregate + save...", flush=True)
    oof_tvt = oof + train_t["last_known_TVT"].astype(float).to_numpy()
    true_tvt = train_t["TVT"].astype(float).to_numpy()
    cv_tvt = float(np.sqrt(np.mean((oof_tvt - true_tvt) ** 2)))
    print(f"  TabICL 5-fold TVT-RMSE = {cv_tvt:.4f}", flush=True)

    # save OOF
    oof_df = train_t[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy()
    oof_df["pred_tvt"] = oof_tvt
    oof_df.to_csv(OUT_DIR / "oof.csv", index=False)

    # save test predictions
    test_t["pred_tvt"] = test_pred_delta + test_t["last_known_TVT"].astype(float).to_numpy()
    test_t[["well_id", "row_idx", "id", "last_known_TVT", "pred_tvt"]].to_csv(
        OUT_DIR / "test_pred.csv", index=False)

    # submission (TabICL alone)
    submission = sample[["id"]].merge(
        test_t[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
        on="id", how="left")
    if submission["tvt"].isna().any():
        raise ValueError(f"submission has {submission['tvt'].isna().sum()} NaN")
    submission.to_csv(OUT_DIR / "submission.csv", index=False)

    import json
    (OUT_DIR / "result.json").write_text(json.dumps({
        "exp_id": "exp035_tabicl_gpu",
        "cv_tvt_rmse": cv_tvt,
        "fold_results": fold_results,
        "context_subsample": CONTEXT_SUBSAMPLE,
        "features": FEATS,
        "wall_time_sec": float(time.time() - t0),
    }, indent=2, ensure_ascii=False, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s, CV TVT-RMSE = {cv_tvt:.4f}")
    print(f"submission.csv rows={len(submission)}, tvt[{submission.tvt.min():.1f}, {submission.tvt.max():.1f}]")


if __name__ == "__main__":
    main()
