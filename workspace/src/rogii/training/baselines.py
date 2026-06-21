"""Baseline helpers for fast OOF experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TARGET_COL = "TVT"
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


def tvt_rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute RMSE on absolute TVT values."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def load_base_inputs(
    train_path: Path,
    test_path: Path,
    folds_path: Path,
    sample_submission_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    folds = pd.read_csv(folds_path)
    sample_submission = pd.read_csv(sample_submission_path)
    return train, test, folds, sample_submission


def attach_folds(train: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    fold_map = folds.loc[folds["split"].eq("train"), ["well_id", "fold"]]
    merged = train.merge(fold_map, on="well_id", how="left", validate="many_to_one")
    if merged["fold"].isna().any():
        missing = merged.loc[merged["fold"].isna(), "well_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"fold が見つからない well_id があります: {missing}")
    merged["fold"] = merged["fold"].astype(int)
    return merged


def target_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["is_target"].astype(bool)].copy()


def build_submission(
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    pred_col: str = PRED_COL,
) -> pd.DataFrame:
    pred = target_rows(test)[["id", pred_col]].rename(columns={pred_col: "tvt"})
    submission = sample_submission[["id"]].merge(pred, on="id", how="left", validate="one_to_one")
    if submission["tvt"].isna().any():
        missing = submission.loc[submission["tvt"].isna(), "id"].head(10).tolist()
        raise ValueError(f"submission に予測が無い id があります: {missing}")
    return submission


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_jst() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def ensure_exp_dir(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
