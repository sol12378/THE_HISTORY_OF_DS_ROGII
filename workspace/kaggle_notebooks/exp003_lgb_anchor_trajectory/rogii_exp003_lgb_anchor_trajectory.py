from pathlib import Path
import re

import lightgbm as lgb
import numpy as np
import pandas as pd


def find_input_dir() -> Path:
    candidates = list(Path("/kaggle/input").rglob("sample_submission.csv"))
    if not candidates:
        raise FileNotFoundError(f"sample_submission.csv not found under /kaggle/input. Found: {list(Path('/kaggle/input').glob('*'))}")
    return candidates[0].parent


INPUT_DIR = find_input_dir()
TRAIN_DIR = INPUT_DIR / "train"
TEST_DIR = INPUT_DIR / "test"
SAMPLE_SUB_PATH = INPUT_DIR / "sample_submission.csv"
OUT_PATH = Path("/kaggle/working/submission.csv")

PRED_COL = "pred_tvt"
SAFE_FEATURES = [
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "is_gr_missing",
    "n_rows_in_well",
    "known_length",
    "hidden_length",
    "last_known_TVT",
    "last_known_MD",
    "last_known_X",
    "last_known_Y",
    "last_known_Z",
    "delta_MD_from_PS",
    "delta_X_from_PS",
    "delta_Y_from_PS",
    "delta_Z_from_PS",
    "post_ps_step",
    "row_frac",
]


def well_id_from_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def load_horizontal(split_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(split_dir.glob("*__horizontal_well.csv"), key=natural_key):
        df = pd.read_csv(path)
        df["well_id"] = well_id_from_path(path)
        df["row_idx"] = np.arange(len(df), dtype=np.int32)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_base_features(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split
    out["GR"] = out["GR"].astype(float)
    out["is_gr_missing"] = out["GR"].isna().astype(np.int8)
    out["GR"] = out["GR"].fillna(out.groupby("well_id")["GR"].transform("median")).fillna(0.0)

    has_tvt_input = out["TVT_input"].notna()
    if split == "train":
        out["is_target"] = out["TVT_input"].isna()
    else:
        sample_ids = set(pd.read_csv(SAMPLE_SUB_PATH)["id"].astype(str))
        out["id"] = out["well_id"].astype(str) + "_" + out["row_idx"].astype(str)
        out["is_target"] = out["id"].isin(sample_ids)
    out["is_known_tvt"] = has_tvt_input.astype(np.int8)

    pieces = []
    for _, g in out.groupby("well_id", sort=False):
        g = g.copy()
        known = g.loc[g["TVT_input"].notna()]
        if len(known) == 0:
            raise ValueError(f"No known TVT_input rows for well {g['well_id'].iloc[0]}")
        anchor = known.iloc[-1]
        n = len(g)
        g["n_rows_in_well"] = n
        g["known_length"] = len(known)
        g["hidden_length"] = int(g["is_target"].sum())
        g["last_known_TVT"] = float(anchor["TVT_input"])
        g["last_known_MD"] = float(anchor["MD"])
        g["last_known_X"] = float(anchor["X"])
        g["last_known_Y"] = float(anchor["Y"])
        g["last_known_Z"] = float(anchor["Z"])
        g["delta_MD_from_PS"] = g["MD"].astype(float) - float(anchor["MD"])
        g["delta_X_from_PS"] = g["X"].astype(float) - float(anchor["X"])
        g["delta_Y_from_PS"] = g["Y"].astype(float) - float(anchor["Y"])
        g["delta_Z_from_PS"] = g["Z"].astype(float) - float(anchor["Z"])
        g["post_ps_step"] = np.maximum(g["row_idx"].to_numpy() - int(anchor["row_idx"]), 0)
        g["row_frac"] = g["row_idx"].to_numpy() / max(n - 1, 1)
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


train = add_base_features(load_horizontal(TRAIN_DIR), "train")
test = add_base_features(load_horizontal(TEST_DIR), "test")

train_t = train.loc[train["is_target"].astype(bool)].copy()
test_t = test.loc[test["is_target"].astype(bool)].copy()
sample = pd.read_csv(SAMPLE_SUB_PATH)

y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 42,
    "num_threads": 4,
    "n_estimators": 600,
}

model = lgb.LGBMRegressor(**params)
model.fit(train_t[SAFE_FEATURES], y_delta)
pred_delta = model.predict(test_t[SAFE_FEATURES])
test_t[PRED_COL] = test_t["last_known_TVT"].astype(float).to_numpy() + pred_delta
submission = sample[["id"]].merge(test_t[["id", PRED_COL]].rename(columns={PRED_COL: "tvt"}), on="id", how="left")
if submission["tvt"].isna().any():
    raise ValueError("submission contains missing predictions")
submission.to_csv(OUT_PATH, index=False)
print(f"wrote {OUT_PATH} rows={len(submission)}")
