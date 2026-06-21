#!/usr/bin/env python3
"""exp041: PF-residual GBDT — well-aware residual correction.

Architecture:
  final_TVT = PF_pred + ML_residual_correction

where ML learns (TVT - PF_pred) from features, trained only on target rows (fold-aware CV).
Models: LightGBM, XGBoost, CatBoost (3-model ensemble, 5-fold GroupKFold by well_id).

OOF & test: residual_pred, then TVT_final = PF_pred + residual_pred.
Post-process: well-internal mean smoothing (w=101).

Inputs:
- PF base oof: exp040 (or exp022 if exp040 unavailable)
- Base features v001
- P2 engineered features
- Folds (group well)

Outputs:
- oof.csv, test_pred.csv, result.json
"""
from __future__ import annotations
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter

# ML libraries
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("WARNING: lightgbm not installed")

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("WARNING: xgboost not installed")

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
    print("WARNING: catboost not installed")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_TRAIN = ROOT / "data" / "processed" / "train_base_v001.parquet"
PROCESSED_TEST = ROOT / "data" / "processed" / "test_base_v001.parquet"
P2_TRAIN = ROOT / "experiments" / "features_p2" / "train_p2.parquet"
P2_TEST = ROOT / "experiments" / "features_p2" / "test_p2.parquet"
FOLDS_CSV = ROOT / "data" / "folds" / "folds_group_well_v001.csv"

# PF base: prefer exp040, fallback exp022
PF_BASE_OOF_040 = ROOT / "experiments" / "exp040_multiscale_pf" / "oof.csv"
PF_BASE_OOF_022 = ROOT / "experiments" / "exp022_particle_filter" / "oof.csv"
PF_BASE_TEST_040 = ROOT / "experiments" / "exp040_multiscale_pf" / "submission.csv"
PF_BASE_TEST_022 = ROOT / "experiments" / "exp022_particle_filter" / "submission.csv"

OUT_DIR = ROOT / "experiments" / "exp041_pf_residual_gbdt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Configuration
CFG = {
    "lgb_params": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "verbose": -1,
    },
    "lgb_n_estimators": 2000,
    "lgb_early_stop": 80,
    "xgb_params": {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
    },
    "xgb_n_estimators": 2000,
    "xgb_early_stop": 80,
    "cb_params": {
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "early_stopping_rounds": 80,
        "verbose": False,
        "loss_function": "RMSE",
    },
    "n_folds": 5,
    "smooth_window": 101,
    "smoke_test_n_wells": 5,
}

# Feature engineering
BASE_FEATURES = [
    "MD", "X", "Y", "Z", "GR",
    "delta_MD_from_PS",
    "delta_X_from_PS", "delta_Y_from_PS", "delta_Z_from_PS",
    "post_ps_step", "row_frac", "known_length", "hidden_length",
]

P2_FEATURES = [
    "tort_3d", "tort_xy", "tort_vert",
    "dls_mean", "dls_last30",
    "inclination_last", "azimuth_change",
    "knn_surface", "knn_surface_minus_Z",
]


def load_pf_base() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load PF base OOF & test, prefer exp040, fallback exp022."""
    if PF_BASE_OOF_040.exists():
        logger.info(f"Loading PF base from exp040: {PF_BASE_OOF_040}")
        oof = pd.read_csv(PF_BASE_OOF_040)
        test_sub = pd.read_csv(PF_BASE_TEST_040)
        source = "exp040"
    elif PF_BASE_OOF_022.exists():
        logger.info(f"Loading PF base from exp022 (exp040 unavailable): {PF_BASE_OOF_022}")
        oof = pd.read_csv(PF_BASE_OOF_022)
        test_sub = pd.read_csv(PF_BASE_TEST_022)
        source = "exp022"
    else:
        raise FileNotFoundError(f"No PF base found at {PF_BASE_OOF_040} or {PF_BASE_OOF_022}")

    logger.info(f"PF base OOF shape: {oof.shape}, source: {source}")
    logger.info(f"PF base test shape: {test_sub.shape}")
    return oof, test_sub, source


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    """Engineer features: base + P2 + PF deltas + GR rolling."""
    logger.info("Engineering features...")

    # Load P2 features
    p2_train = pd.read_parquet(P2_TRAIN)
    p2_test = pd.read_parquet(P2_TEST)

    # Merge P2 into train and test
    train_df = train_df.merge(p2_train, on=["well_id", "row_idx"], how="left")
    test_df = test_df.merge(p2_test, on=["well_id", "row_idx"], how="left")

    # Add PF delta and other derived features
    train_df["pf_delta"] = train_df["pred_tvt"] - train_df["last_known_TVT"]
    # Test doesn't have last_known_TVT, so use pred_tvt as anchor
    if "last_known_TVT" in test_df.columns:
        test_df["pf_delta"] = test_df["pred_tvt"] - test_df["last_known_TVT"]
    else:
        test_df["pf_delta"] = 0

    # GR rolling stats (within well)
    for df in [train_df, test_df]:
        if "GR" in df.columns:
            df["gr_roll20_mean"] = df.groupby("well_id")["GR"].transform(
                lambda x: x.rolling(window=20, center=True, min_periods=1).mean()
            )
            df["gr_roll50_mean"] = df.groupby("well_id")["GR"].transform(
                lambda x: x.rolling(window=50, center=True, min_periods=1).mean()
            )
            df["gr_roll20_std"] = df.groupby("well_id")["GR"].transform(
                lambda x: x.rolling(window=20, center=True, min_periods=1).std()
            )
        else:
            logger.warning("GR not in dataframe, skipping GR rolling features")
            df["gr_roll20_mean"] = 0
            df["gr_roll50_mean"] = 0
            df["gr_roll20_std"] = 0

    # Feature list (only use those that exist)
    potential_features = (
        BASE_FEATURES
        + ["last_known_TVT", "pred_tvt", "pf_delta"]
        + P2_FEATURES
        + ["gr_roll20_mean", "gr_roll50_mean", "gr_roll20_std"]
    )

    feature_list = [f for f in potential_features if f in train_df.columns]
    logger.info(f"Available features: {len(feature_list)} / {len(potential_features)}")
    missing = [f for f in potential_features if f not in train_df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    # Fill NaN values with median
    for feat in feature_list:
        if train_df[feat].dtype in ['float64', 'float32', 'int64', 'int32']:
            median_val = train_df[feat].median()
            train_df[feat] = train_df[feat].fillna(median_val)
            test_df[feat] = test_df[feat].fillna(median_val)
        else:
            train_df[feat] = train_df[feat].fillna(0)
            test_df[feat] = test_df[feat].fillna(0)

    logger.info(f"Features engineered: {len(feature_list)} features")
    return feature_list, train_df, test_df


def train_lgb_fold(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    fold_id: int
) -> Tuple[object, np.ndarray, float]:
    """Train LightGBM for one fold."""
    if lgb is None:
        raise ImportError("lightgbm required but not installed")

    params = CFG["lgb_params"].copy()
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=CFG["lgb_n_estimators"],
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(CFG["lgb_early_stop"], verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    y_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    logger.info(f"  LGB fold {fold_id}: RMSE={rmse:.6f}")
    return model, y_pred, rmse


def train_xgb_fold(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    fold_id: int
) -> Tuple[object, np.ndarray, float]:
    """Train XGBoost for one fold."""
    if xgb is None:
        raise ImportError("xgboost required but not installed")

    params = CFG["xgb_params"].copy()
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)

    evals = [(train_data, "train"), (val_data, "eval")]
    evals_result = {}
    model = xgb.train(
        params,
        train_data,
        num_boost_round=CFG["xgb_n_estimators"],
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=CFG["xgb_early_stop"],
        verbose_eval=False,
    )

    y_pred = model.predict(val_data)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    logger.info(f"  XGB fold {fold_id}: RMSE={rmse:.6f}")
    return model, y_pred, rmse


def train_cb_fold(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    fold_id: int
) -> Tuple[object, np.ndarray, float]:
    """Train CatBoost for one fold."""
    if CatBoostRegressor is None:
        raise ImportError("catboost required but not installed")

    params = CFG["cb_params"].copy()
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    y_pred = model.predict(X_val)
    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
    logger.info(f"  CB fold {fold_id}: RMSE={rmse:.6f}")
    return model, y_pred, rmse


def cv_residual_gbdt(
    train_df: pd.DataFrame,
    feature_list: List[str],
    smoke_test: bool = False
) -> Tuple[np.ndarray, List[Dict], float, int]:
    """Cross-validation: train 3-model ensemble on residuals (fold-aware)."""
    logger.info("\n[CV] Starting residual GBDT cross-validation...")

    # Prepare target: residual = TVT - PF_pred
    train_df["residual"] = train_df["TVT"] - train_df["pred_tvt"]
    y = train_df["residual"].values

    # Folds
    folds = pd.read_csv(FOLDS_CSV).drop_duplicates(subset=["well_id"])
    train_df = train_df.merge(folds[["well_id", "fold"]], on="well_id", how="left")
    train_df["fold"] = train_df["fold"].fillna(0).astype(int)

    # GroupKFold on well_id
    gkf = GroupKFold(n_splits=CFG["n_folds"])
    groups = train_df["well_id"].values
    well_ids_unique = train_df["well_id"].unique()
    if smoke_test:
        well_ids_unique = well_ids_unique[:CFG["smoke_test_n_wells"]]
        logger.info(f"SMOKE TEST: using {len(well_ids_unique)} wells")

    # Initialize OOF arrays
    oof_residual = np.zeros(len(train_df))
    oof_residual_lgb = np.zeros(len(train_df))
    oof_residual_xgb = np.zeros(len(train_df))
    oof_residual_cb = np.zeros(len(train_df))

    models_all = {"lgb": [], "xgb": [], "cb": []}
    fold_rmses = {"lgb": [], "xgb": [], "cb": []}
    fold_count = 0

    # Manual split on well_id
    for fold_id in range(CFG["n_folds"]):
        # Deterministic train/val split by well
        val_well_mask = (train_df["fold"] == fold_id)
        if smoke_test:
            val_well_mask = val_well_mask & (train_df["well_id"].isin(well_ids_unique))

        train_mask = ~val_well_mask & (train_df["well_id"].isin(well_ids_unique))

        if train_mask.sum() == 0 or val_well_mask.sum() == 0:
            logger.warning(f"Fold {fold_id}: skipping (empty split)")
            continue

        X_train = train_df.loc[train_mask, feature_list].copy()
        y_train = y[train_mask]
        X_val = train_df.loc[val_well_mask, feature_list].copy()
        y_val = y[val_well_mask]

        logger.info(f"\nFold {fold_id}: train={len(X_train)}, val={len(X_val)}")

        # LightGBM
        if lgb is not None:
            try:
                lgb_model, lgb_pred, lgb_rmse = train_lgb_fold(X_train, y_train, X_val, y_val, fold_id)
                models_all["lgb"].append(lgb_model)
                oof_residual_lgb[val_well_mask] = lgb_pred
                fold_rmses["lgb"].append(lgb_rmse)
            except Exception as e:
                logger.error(f"LGB fold {fold_id} failed: {e}")

        # XGBoost
        if xgb is not None:
            try:
                xgb_model, xgb_pred, xgb_rmse = train_xgb_fold(X_train, y_train, X_val, y_val, fold_id)
                models_all["xgb"].append(xgb_model)
                oof_residual_xgb[val_well_mask] = xgb_pred
                fold_rmses["xgb"].append(xgb_rmse)
            except Exception as e:
                logger.error(f"XGB fold {fold_id} failed: {e}")

        # CatBoost
        if CatBoostRegressor is not None:
            try:
                cb_model, cb_pred, cb_rmse = train_cb_fold(X_train, y_train, X_val, y_val, fold_id)
                models_all["cb"].append(cb_model)
                oof_residual_cb[val_well_mask] = cb_pred
                fold_rmses["cb"].append(cb_rmse)
            except Exception as e:
                logger.error(f"CB fold {fold_id} failed: {e}")

        fold_count += 1

    # Average ensemble
    n_models = sum([len(m) for m in models_all.values()])
    logger.info(f"\nTotal models trained: {n_models} ({fold_count} folds × {n_models // max(fold_count, 1)})")

    if fold_count == 0:
        raise RuntimeError("No folds trained successfully")

    # Weighted average (equal weight for now)
    counts = [len(m) > 0 for m in models_all.values()]
    if counts[0] > 0:
        oof_residual += oof_residual_lgb
    if counts[1] > 0:
        oof_residual += oof_residual_xgb
    if counts[2] > 0:
        oof_residual += oof_residual_cb
    oof_residual /= sum(counts)

    # CV RMSE
    cv_rmse = np.sqrt(np.mean((y - oof_residual) ** 2))
    logger.info(f"\nOOF Ensemble RMSE (3-model avg): {cv_rmse:.6f}")

    return oof_residual, models_all, cv_rmse, fold_count


def smooth_predictions(df: pd.DataFrame, pred_col: str, window: int = 101) -> np.ndarray:
    """Smooth predictions within each well using savgol filter."""
    smoothed = df[pred_col].copy()
    for well_id in df["well_id"].unique():
        mask = df["well_id"] == well_id
        indices = np.where(mask)[0]
        if len(indices) < window:
            # Simple mean smoothing for small wells
            smoothed.iloc[mask] = df.loc[mask, pred_col].mean()
        else:
            try:
                smoothed.iloc[indices] = savgol_filter(
                    df.loc[mask, pred_col].values,
                    window_length=window,
                    polyorder=3
                )
            except Exception:
                # Fallback to rolling mean
                smoothed.iloc[indices] = df.loc[mask, pred_col].rolling(
                    window=window, center=True, min_periods=1
                ).mean().values

    return smoothed.values


def predict_test(
    test_df: pd.DataFrame,
    feature_list: List[str],
    models_all: Dict[str, List]
) -> np.ndarray:
    """Predict test residuals using ensemble of trained models."""
    logger.info("\n[Test] Predicting test residuals...")

    X_test = test_df[feature_list].copy()
    test_residual = np.zeros(len(test_df))
    n_models = 0

    # LightGBM predictions
    if lgb is not None and len(models_all["lgb"]) > 0:
        for model in models_all["lgb"]:
            test_residual += model.predict(X_test)
        n_models += len(models_all["lgb"])
        logger.info(f"  LGB: {len(models_all['lgb'])} models")

    # XGBoost predictions
    if xgb is not None and len(models_all["xgb"]) > 0:
        for model in models_all["xgb"]:
            X_test_dmatrix = xgb.DMatrix(X_test)
            test_residual += model.predict(X_test_dmatrix)
        n_models += len(models_all["xgb"])
        logger.info(f"  XGB: {len(models_all['xgb'])} models")

    # CatBoost predictions
    if CatBoostRegressor is not None and len(models_all["cb"]) > 0:
        for model in models_all["cb"]:
            test_residual += model.predict(X_test)
        n_models += len(models_all["cb"])
        logger.info(f"  CB: {len(models_all['cb'])} models")

    if n_models > 0:
        test_residual /= n_models
        logger.info(f"  Ensemble avg ({n_models} models)")
    else:
        raise RuntimeError("No test predictions generated")

    return test_residual


def main(smoke_test: bool = False):
    logger.info("=" * 70)
    logger.info("exp041: PF-residual GBDT")
    logger.info(f"smoke_test={smoke_test}")
    logger.info("=" * 70)

    try:
        # [1] Load PF base
        logger.info("\n[1] Loading PF base...")
        pf_oof, pf_test, pf_source = load_pf_base()

        # [2] Load base features
        logger.info("\n[2] Loading base features...")
        base_train = pd.read_parquet(PROCESSED_TRAIN, columns=[
            "well_id", "row_idx", "TVT", "last_known_TVT", "last_known_Z"
        ] + BASE_FEATURES)
        base_test = pd.read_parquet(PROCESSED_TEST, columns=[
            "well_id", "row_idx", "id", "last_known_TVT", "last_known_Z"
        ] + [f for f in BASE_FEATURES if f != "TVT"])

        # Filter to target rows (is_target=True)
        base_all = pd.read_parquet(PROCESSED_TRAIN, columns=["well_id", "row_idx", "is_target"])
        target_rows = base_all[base_all["is_target"].astype(bool)][["well_id", "row_idx"]].copy()
        base_train = base_train.merge(target_rows, on=["well_id", "row_idx"], how="inner")

        logger.info(f"  train shape: {base_train.shape}, test shape: {base_test.shape}")

        # [3] Merge PF predictions
        logger.info("\n[3] Merging PF predictions...")
        pf_oof = pf_oof[["well_id", "row_idx", "pred_tvt"]].rename(columns={"pred_tvt": "pred_tvt"})
        train_df = base_train.merge(pf_oof, on=["well_id", "row_idx"], how="inner")
        logger.info(f"  train merged shape: {train_df.shape}")

        pf_test = pf_test[["id", "tvt"]].rename(columns={"tvt": "pred_tvt"})
        pf_test["row_idx"] = pf_test.index
        test_df = base_test.merge(pf_test[["row_idx", "pred_tvt"]], on="row_idx", how="inner")
        logger.info(f"  test merged shape: {test_df.shape}")

        # [4] Engineer features
        feature_list, train_df, test_df = engineer_features(train_df, test_df)

        # [5] CV residual training
        oof_residual, models_all, cv_rmse, fold_count = cv_residual_gbdt(
            train_df, feature_list, smoke_test=smoke_test
        )

        # [6] Add residual to train, create final OOF
        train_df["oof_residual"] = oof_residual
        train_df["pred_tvt_final"] = train_df["pred_tvt"] + train_df["oof_residual"]

        # Smooth final predictions
        train_df["pred_tvt_final_smoothed"] = smooth_predictions(
            train_df, "pred_tvt_final", window=CFG["smooth_window"]
        )

        # [7] Test predictions
        test_residual = predict_test(test_df, feature_list, models_all)
        test_df["pred_residual"] = test_residual
        test_df["pred_tvt_final"] = test_df["pred_tvt"] + test_df["pred_residual"]

        # Smooth test predictions
        test_df["pred_tvt_final_smoothed"] = smooth_predictions(
            test_df, "pred_tvt_final", window=CFG["smooth_window"]
        )

        # [8] Calculate CV metrics
        pf_base_cv = np.sqrt(np.mean((train_df["TVT"] - train_df["pred_tvt"]) ** 2))
        residual_cv = np.sqrt(np.mean((train_df["TVT"] - train_df["pred_tvt"] - train_df["oof_residual"]) ** 2))
        final_cv = np.sqrt(np.mean((train_df["TVT"] - train_df["pred_tvt_final"]) ** 2))
        final_cv_smoothed = np.sqrt(np.mean((train_df["TVT"] - train_df["pred_tvt_final_smoothed"]) ** 2))

        logger.info("\n[Results]")
        logger.info(f"  PF base CV RMSE: {pf_base_cv:.6f}")
        logger.info(f"  Residual model CV RMSE: {residual_cv:.6f}")
        logger.info(f"  Final TVT CV RMSE: {final_cv:.6f}")
        logger.info(f"  Final TVT smoothed CV RMSE: {final_cv_smoothed:.6f}")
        logger.info(f"  Improvement: {pf_base_cv - final_cv_smoothed:.6f}")

        # [9] Save OOF
        oof_output = train_df[["well_id", "row_idx", "TVT", "last_known_TVT", "pred_tvt", "pred_tvt_final", "pred_tvt_final_smoothed"]].copy()
        oof_output.columns = ["well_id", "row_idx", "TVT", "last_known_TVT", "pred_tvt", "pred_tvt_final", "pred_tvt_final_smoothed"]
        # Add id column (fake for OOF, used for submission alignment)
        oof_output["id"] = range(len(oof_output))
        # Use smoothed as final prediction
        oof_output["pred_tvt"] = oof_output["pred_tvt_final_smoothed"]
        oof_output = oof_output[["well_id", "row_idx", "id", "TVT", "last_known_TVT", "pred_tvt"]]

        oof_output.to_csv(OUT_DIR / "oof.csv", index=False)
        logger.info(f"  OOF saved: {OUT_DIR / 'oof.csv'}")

        # [10] Save test submission
        test_output = test_df[["id", "pred_tvt_final_smoothed"]].copy()
        test_output.columns = ["id", "tvt"]
        test_output.to_csv(OUT_DIR / "test_pred.csv", index=False)
        logger.info(f"  Test pred saved: {OUT_DIR / 'test_pred.csv'}")

        # [11] Save result.json
        result = {
            "exp_id": "exp041",
            "method": "pf_residual_gbdt",
            "pf_base_source": pf_source,
            "pf_base_cv_rmse": float(pf_base_cv),
            "residual_model_cv_rmse": float(residual_cv),
            "cv_rmse": float(final_cv_smoothed),
            "n_folds": fold_count,
            "n_features": len(feature_list),
            "n_samples_train": len(train_df),
            "n_samples_test": len(test_df),
            "improvement_vs_pf": float(pf_base_cv - final_cv_smoothed),
            "smoke_test": smoke_test,
            "models_trained": {
                "lgb": len(models_all["lgb"]),
                "xgb": len(models_all["xgb"]),
                "cb": len(models_all["cb"]),
            },
        }
        with open(OUT_DIR / "result.json", "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Result saved: {OUT_DIR / 'result.json'}")

        logger.info("\n" + "=" * 70)
        logger.info("exp041 COMPLETED")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"\nFATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    smoke_test = "--smoke" in sys.argv or "smoke" in os.environ.get("EXP_MODE", "")
    main(smoke_test=smoke_test)
