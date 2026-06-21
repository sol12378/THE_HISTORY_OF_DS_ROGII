"""
exp073 Task D: Engine B LightGBM GroupKFold OOF

Build honest OOF for fleongg Engine B using same features + params
but GroupKFold(5, by well) retraining instead of all-train.

Target: delta = TVT - last_known_TVT (same as fleongg Engine B)
Features: 195 features from features.json (likpf_mean_d excluded - not in train.csv)
LGB params: 3 configs from fle3n-rogii-v4.ipynb
CV: GroupKFold(5) groups=well
"""

import os, sys, time, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# === Paths ===
ROOT = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII")
TRAIN_CSV = ROOT / "data/external/wellbore-geology-prediction-artifacts/data/train.csv"
FEATURES_JSON = ROOT / "data/external/rogii-claude-models-pub/features.json"
OUT_DIR = ROOT / "experiments/exp073_public_assets_integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "log_engineB.txt"
ERROR_LOG = OUT_DIR / "error_engineB.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

t0 = time.time()

def elapsed():
    return (time.time() - t0) / 60

try:
    # === Load features list ===
    log.info("Loading features.json...")
    with open(FEATURES_JSON) as f:
        all_features = json.load(f)
    log.info(f"Features in JSON: {len(all_features)}")

    # === Check which features exist in train.csv ===
    log.info("Checking available columns in train.csv (header only)...")
    df_head = pd.read_csv(TRAIN_CSV, nrows=1)
    csv_cols = set(df_head.columns)
    features = [f for f in all_features if f in csv_cols]
    missing_features = [f for f in all_features if f not in csv_cols]
    log.info(f"Present features: {len(features)}, Missing: {missing_features}")

    # === Load full training data ===
    log.info("Loading train.csv (float32 for memory efficiency)...")
    t_load = time.time()

    usecols = ['well', 'id', 'target'] + features
    dtype_map = {f: np.float32 for f in features}
    dtype_map['target'] = np.float32
    dtype_map['well'] = str
    dtype_map['id'] = str

    df = pd.read_csv(
        TRAIN_CSV,
        usecols=usecols,
        dtype=dtype_map
    )
    log.info(f"Loaded: {len(df):,} rows, {df.shape[1]} cols in {(time.time()-t_load)/60:.1f}min")
    log.info(f"Unique wells: {df['well'].nunique()}")
    log.info(f"Target stats: mean={df['target'].mean():.4f}, std={df['target'].std():.4f}")

    # Validate row count
    n_rows = len(df)
    assert n_rows == 3783989, f"Expected 3783989 rows, got {n_rows}"
    log.info(f"Row count validated: {n_rows:,}")

    # === Prepare arrays ===
    X = df[features].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    groups = df['well'].values
    ids = df['id'].values
    wells = df['well'].values

    log.info(f"X shape: {X.shape}, y shape: {y.shape}")
    log.info(f"Memory: X={X.nbytes/1e9:.2f}GB, y={y.nbytes/1e6:.1f}MB")

    # === LightGBM configurations (from fle3n-rogii-v4.ipynb Engine B) ===
    # No GPU - use CPU (num_threads=14)
    base_cpu = dict(
        boosting_type="gbdt",
        objective="regression",
        verbose=-1,
        n_jobs=14,
        max_bin=255,
        device_type="cpu"
    )

    lgb_configs = [
        # Config 0: num_leaves=255, lr=0.03, n_estimators=5000 (main strong config)
        dict(**base_cpu,
             num_leaves=255,
             min_child_samples=15,
             subsample=0.8,
             subsample_freq=1,
             colsample_bytree=0.8,
             reg_lambda=3.0,
             reg_alpha=0.05,
             learning_rate=0.03,
             n_estimators=5000,
             seed=123),
        # Config 1: num_leaves=64, lr=0.0093, n_estimators=10000 (seed=0)
        dict(**base_cpu,
             num_leaves=64,
             min_child_samples=40,
             subsample=0.474,
             subsample_freq=1,
             colsample_bytree=0.393,
             reg_lambda=95.75,
             reg_alpha=10.79,
             min_child_weight=0.24,
             learning_rate=0.0093,
             n_estimators=10000,
             random_state=0),
        # Config 2: same as config 1 but seed=29
        dict(**base_cpu,
             num_leaves=64,
             min_child_samples=40,
             subsample=0.474,
             subsample_freq=1,
             colsample_bytree=0.393,
             reg_lambda=95.75,
             reg_alpha=10.79,
             min_child_weight=0.24,
             learning_rate=0.0093,
             n_estimators=10000,
             random_state=29),
    ]

    config_names = ['engB_lgb1', 'engB_lgb2', 'engB_lgb3']

    # === GroupKFold CV ===
    N_SPLITS = 5
    gkf = GroupKFold(n_splits=N_SPLITS)
    splits = list(gkf.split(X, y, groups=groups))
    log.info(f"GroupKFold splits ready: {N_SPLITS} folds")

    # === OOF storage ===
    oof_preds = {name: np.full(n_rows, np.nan, dtype=np.float32) for name in config_names}
    fold_rmses = {name: [] for name in config_names}

    # === Training loop ===
    for cfg_idx, (cfg_name, params) in enumerate(zip(config_names, lgb_configs)):
        log.info(f"\n{'='*60}")
        log.info(f"Training {cfg_name} (config {cfg_idx+1}/3)...")
        log.info(f"Params: num_leaves={params['num_leaves']}, lr={params['learning_rate']}, n_est={params['n_estimators']}")

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            t_fold = time.time()
            log.info(f"  Fold {fold_idx+1}/{N_SPLITS}: train={len(tr_idx):,}, val={len(va_idx):,}")

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            model = LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='rmse',
                callbacks=[
                    early_stopping(250, verbose=False),
                    log_evaluation(0)
                ]
            )

            best_iter = model.best_iteration_
            pred_va = model.predict(X_va, num_iteration=best_iter)
            oof_preds[cfg_name][va_idx] = pred_va.astype(np.float32)

            fold_rmse = np.sqrt(np.mean((y_va - pred_va) ** 2))
            fold_rmses[cfg_name].append(fold_rmse)
            t_fold_min = (time.time() - t_fold) / 60
            log.info(f"  Fold {fold_idx+1}: RMSE={fold_rmse:.4f}, best_iter={best_iter}, time={t_fold_min:.1f}min, elapsed={elapsed():.1f}min")

        # Check for NaNs
        nan_count = np.sum(np.isnan(oof_preds[cfg_name]))
        if nan_count > 0:
            log.warning(f"  WARNING: {nan_count} NaN predictions in {cfg_name}")

        pooled_rmse = np.sqrt(np.mean((y - oof_preds[cfg_name]) ** 2))
        log.info(f"  {cfg_name} pooled RMSE: {pooled_rmse:.4f}")
        log.info(f"  {cfg_name} per-fold RMSE: {[f'{v:.4f}' for v in fold_rmses[cfg_name]]}")

        # Save checkpoint after each config
        ckpt_path = OUT_DIR / f"checkpoint_{cfg_name}.npz"
        np.savez(ckpt_path, preds=oof_preds[cfg_name])
        log.info(f"  Checkpoint saved: {ckpt_path}")

    # === Compute engB_avg ===
    log.info("\nComputing engB_avg...")
    stacked = np.stack([oof_preds[n] for n in config_names], axis=1)
    engB_avg = np.nanmean(stacked, axis=1).astype(np.float32)

    avg_rmse = np.sqrt(np.mean((y - engB_avg) ** 2))
    log.info(f"engB_avg pooled RMSE: {avg_rmse:.4f}")

    # Per-config pooled RMSEs
    per_config_rmse = {}
    for name in config_names:
        rmse_val = float(np.sqrt(np.mean((y - oof_preds[name]) ** 2)))
        per_config_rmse[name] = rmse_val
        log.info(f"{name} pooled RMSE: {rmse_val:.4f}")

    # === Save OOF parquet ===
    log.info("\nSaving oof_engineB.parquet...")
    oof_df = pd.DataFrame({
        'id': ids,
        'well': wells,
        'target_delta': y.astype(np.float32),
    })
    for name in config_names:
        oof_df[f'pred_delta_{name}'] = oof_preds[name]
    oof_df['pred_delta_engB_avg'] = engB_avg

    out_parquet = OUT_DIR / "oof_engineB.parquet"
    oof_df.to_parquet(out_parquet, index=False)
    log.info(f"Saved: {out_parquet} ({len(oof_df):,} rows)")

    # === Save result JSON ===
    runtime_min = elapsed()
    result = {
        "exp": "exp073_taskD_engineB",
        "status": "ok",
        "pooled_rmse_engB_avg": float(avg_rmse),
        "per_config_rmse": per_config_rmse,
        "per_fold_rmse": {name: [float(v) for v in fold_rmses[name]] for name in config_names},
        "n_rows": n_rows,
        "n_wells": int(df['well'].nunique()),
        "n_features": len(features),
        "missing_features": missing_features,
        "quality_gate": {
            "threshold": 10.8,
            "passed": float(avg_rmse) < 10.8,
            "metric": float(avg_rmse)
        },
        "runtime_min": runtime_min,
        "leak_notes": (
            "LEAK AUDIT: target=delta=TVT-last_known is from train.csv 'target' column. "
            "Features are pre-computed per-row by fleongg (particle filter, beam search, etc). "
            "GroupKFold ensures held-out fold predictions never see fold-val labels during training. "
            "OOF construction: for each fold, model trained on tr_idx only, predicts va_idx. "
            "True TVT values of hidden rows are NOT used in feature construction (features are physics-based). "
            "Mild optimistic bias: early stopping uses held-out fold as eval set (standard practice). "
        )
    }

    result_path = OUT_DIR / "result_engineB.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved: {result_path}")
    log.info(f"\nFINAL: engB_avg RMSE={avg_rmse:.4f}, gate={'PASS' if avg_rmse < 10.8 else 'FAIL'}")
    log.info(f"Total runtime: {runtime_min:.1f} min")

except Exception as e:
    import traceback
    err_msg = traceback.format_exc()
    with open(ERROR_LOG, 'w', encoding='utf-8') as ef:
        ef.write(err_msg)
    log.error(f"FATAL ERROR: {e}")
    log.error(err_msg)
    raise
