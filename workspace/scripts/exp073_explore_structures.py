"""
exp073_explore_structures.py
Explore the structure of all three external data sources before building unified OOF table.
"""
import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

OUT_DIR = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII/experiments/exp073_public_assets_integration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG = OUT_DIR / "explore_log.txt"

def log(msg):
    print(msg)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log("=" * 70)
log("EXPLORING EXTERNAL ASSET STRUCTURES")
log("=" * 70)

# ============================
# 1. pilkwang train_gt.parquet
# ============================
log("\n--- 1. PILKWANG train_gt.parquet ---")
pilk_dir = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII/data/external/rogii-model-package")
train_gt = pd.read_parquet(pilk_dir / "oof/train_gt.parquet")
log(f"Shape: {train_gt.shape}")
log(f"Columns: {list(train_gt.columns)}")
log(f"dtypes:\n{train_gt.dtypes.to_string()}")
log(f"Head:\n{train_gt.head(3).to_string()}")
log(f"Null counts:\n{train_gt.isnull().sum().to_string()}")

# Check key columns
for col in ['well_id', 'row_idx', 'row_index', 'id', 'target', 'delta', 'TVT', 'last_known_tvt', 'last_known_TVT']:
    if col in train_gt.columns:
        log(f"  col '{col}' present, sample: {train_gt[col].iloc[:5].tolist()}")

# ============================
# 2. pilkwang OOF arrays
# ============================
log("\n--- 2. PILKWANG OOF .npy arrays ---")
for name in ['xgb_oof', 'catboost_oof', 'hgb_oof', 'lgb_oof', 'sequence_tcn_oof', 'blend_oof', 'blend_oof_postprocessed']:
    arr = np.load(pilk_dir / f"oof/{name}.npy")
    log(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")

# ============================
# 3. v11 OOF preds
# ============================
log("\n--- 3. V11 oof_preds.pkl structures ---")
v11_dir = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII/data/external/rogii-v11-fresh-artifacts/models")
for model_key in ['lightgbm-1', 'lightgbm-2', 'lightgbm-3', 'catboost-1', 'catboost-2', 'catboost-3']:
    oof_path = v11_dir / model_key / "oof_preds.pkl"
    if oof_path.exists():
        with open(oof_path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            log(f"  {model_key}/oof_preds: ndarray shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, pd.DataFrame):
            log(f"  {model_key}/oof_preds: DataFrame shape={obj.shape}, columns={list(obj.columns)}")
            log(f"    head: {obj.head(2).to_string()}")
        elif isinstance(obj, dict):
            log(f"  {model_key}/oof_preds: dict keys={list(obj.keys())}")
            for k, v in obj.items():
                if hasattr(v, 'shape'):
                    log(f"    {k}: shape={v.shape}")
                else:
                    log(f"    {k}: type={type(v)}, value={str(v)[:100]}")
        else:
            log(f"  {model_key}/oof_preds: type={type(obj)}")
    else:
        log(f"  {model_key}/oof_preds.pkl NOT FOUND")

# ============================
# 4. ravaghi trainer pickles
# ============================
log("\n--- 4. RAVAGHI trainer pickles ---")
rav_dir = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII/data/external/wellbore-geology-prediction-artifacts/models")
rav_model_dirs = ['lightgbm-1', 'lightgbm-2', 'lightgbm-3', 'catboost-1', 'catboost-2']

for model_key in rav_model_dirs:
    model_path = rav_dir / model_key
    pkl_files = list(model_path.glob("*.pkl"))
    log(f"  {model_key}: pkl files = {[f.name for f in pkl_files]}")
    if pkl_files:
        try:
            import joblib
            trainer = joblib.load(pkl_files[0])
            log(f"    type={type(trainer)}")
            log(f"    attrs={[a for a in dir(trainer) if not a.startswith('__')][:30]}")
            # check oof_preds
            for attr in ['oof_preds', 'oof_predictions', 'val_preds', 'overall_score', 'cv_scores']:
                if hasattr(trainer, attr):
                    val = getattr(trainer, attr)
                    if hasattr(val, 'shape'):
                        log(f"    {attr}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        log(f"    {attr}: {str(val)[:200]}")
        except Exception as e:
            log(f"    ERROR loading {pkl_files[0].name}: {e}")

# ============================
# 5. ravaghi train.csv header (usecols only)
# ============================
log("\n--- 5. RAVAGHI train.csv ---")
rav_csv = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII/data/external/wellbore-geology-prediction-artifacts/data/train.csv")
if rav_csv.exists():
    # Read just header first
    header_df = pd.read_csv(rav_csv, nrows=3)
    log(f"  All columns: {list(header_df.columns)}")
    # Try loading usecols
    target_cols = ['well', 'id', 'last_known_tvt', 'target']
    avail_cols = [c for c in target_cols if c in header_df.columns]
    log(f"  Requested cols available: {avail_cols}")
    log(f"  Shape of header_df (3 rows): {header_df.shape}")
    log(f"  head:\n{header_df[avail_cols[:4]].head(3).to_string() if avail_cols else 'none'}")
else:
    log("  train.csv NOT FOUND")

# ============================
# 6. train_base check
# ============================
log("\n--- 6. TRAIN_BASE hidden rows ---")
tb = pd.read_parquet("E:/kaggle/THE_HISTORY_OF_DS_ROGII/data/processed/train_base_v001.parquet")
hidden = tb[~tb['is_known_tvt']].copy()
log(f"  train_base shape: {tb.shape}")
log(f"  hidden rows: {len(hidden)}")
log(f"  hidden columns: {list(hidden.columns)}")
log(f"  unique well_ids: {hidden['well_id'].nunique()}")
log(f"  id sample: {hidden['id'].head(5).tolist()}")

log("\nDONE exploration")
