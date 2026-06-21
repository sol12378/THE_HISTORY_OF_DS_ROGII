"""ROGII exp008 GR rolling — Kaggle Notebook submission script.

self-contained reproduction of experiments/exp008_gr_rolling (CV=13.808621).
features: SAFE + Group A (pre-PS TVT slope/curv) + Group B (per-row + per-well dir)
         + Group C (kh_ratio, hidden_frac) + Group D (pre-PS GR stats + rolling).
Group E (typewell) is intentionally EXCLUDED (sealed: see exp011 leak test).

training: 5-fold GroupKFold-by-well (greedy load-balanced, same as make_folds.py),
          LightGBM delta regression (n_estimators=1500, early_stopping=50),
          test predictions averaged over the 5 fold models — identical to exp008 main loop.

This script reconstructs base features directly from raw CSVs, replicating
src/rogii/data/build_base.py (GR kept NaN; is_gr_missing flag) so the model input
matches train_base_v001.parquet exactly.
"""
from pathlib import Path
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

# ── input/output locations ───────────────────────────────────────────────────
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
OUT_PATH = OUT_DIR / "submission.csv"

PRED_COL = "pred_tvt"
N_SPLITS = 5

SAFE_FEATURES = [
    "MD", "X", "Y", "Z", "GR", "is_gr_missing", "n_rows_in_well",
    "known_length", "hidden_length", "last_known_TVT", "last_known_MD",
    "last_known_X", "last_known_Y", "last_known_Z", "delta_MD_from_PS",
    "delta_X_from_PS", "delta_Y_from_PS", "delta_Z_from_PS",
    "post_ps_step", "row_frac",
]
GROUP_A = ["pre_ps_tvt_slope_last20", "pre_ps_tvt_slope_last5",
           "pre_ps_tvt_curvature", "pre_ps_tvt_delta_last20"]
GROUP_B_WELL = ["pre_ps_dZ_dMD", "pre_ps_dX_dMD", "pre_ps_dY_dMD",
                "pre_ps_horiz_dMD", "pre_ps_azimuth"]
GROUP_B_ROW = ["dZ_dMD_from_ps", "dX_dMD_from_ps", "dY_dMD_from_ps",
               "horiz_disp_from_ps", "azimuth_from_ps"]
GROUP_C = ["kh_ratio", "hidden_frac"]
GROUP_D_WELL = ["pre_ps_gr_mean", "pre_ps_gr_std", "pre_ps_gr_last20_mean",
                "pre_ps_gr_trend", "pre_ps_gr_available_frac"]
GROUP_D_ROW = ["gr_vs_pre_ps_mean", "gr_z_score", "gr_rolling_mean_w20",
               "gr_rolling_mean_w50", "gr_rolling_std_w20"]
ALL_FEATURES = (SAFE_FEATURES + GROUP_A + GROUP_B_WELL + GROUP_B_ROW
                + GROUP_C + GROUP_D_WELL + GROUP_D_ROW)


# ── base feature reconstruction (replicates src/rogii/data/build_base.py) ─────
def well_id_from_path(path: Path) -> str:
    return path.name.split("__", 1)[0]


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


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
    df["last_known_X"] = anchor["X"]
    df["last_known_Y"] = anchor["Y"]
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


# ── Group A/B/C trajectory features (from exp007/exp008) ──────────────────────
def traj_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        md = g["MD"].to_numpy(float); x = g["X"].to_numpy(float)
        y = g["Y"].to_numpy(float); z = g["Z"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float); n = len(g)
        n20 = min(20, n); n5 = min(5, n)
        d20 = md[-1] - md[-n20] if n20 > 1 else 1.
        d5 = md[-1] - md[-n5] if n5 > 1 else 1.
        s20 = (tv[-1] - tv[-n20]) / d20 if abs(d20) > 1e-6 else 0.
        s5 = (tv[-1] - tv[-n5]) / d5 if abs(d5) > 1e-6 else 0.
        dz = (z[-1] - z[-n20]) / d20 if abs(d20) > 1e-6 else 0.
        dx = (x[-1] - x[-n20]) / d20 if abs(d20) > 1e-6 else 0.
        dy = (y[-1] - y[-n20]) / d20 if abs(d20) > 1e-6 else 0.
        dx20 = x[-1] - x[-n20]; dy20 = y[-1] - y[-n20]
        hd = float(np.sqrt(dx20 ** 2 + dy20 ** 2))
        recs.append({"well_id": wid,
                     "pre_ps_tvt_slope_last20": s20, "pre_ps_tvt_slope_last5": s5,
                     "pre_ps_tvt_curvature": s5 - s20, "pre_ps_tvt_delta_last20": tv[-1] - tv[-n20],
                     "pre_ps_dZ_dMD": dz, "pre_ps_dX_dMD": dx, "pre_ps_dY_dMD": dy,
                     "pre_ps_horiz_dMD": hd / d20 if abs(d20) > 1e-6 else 0.,
                     "pre_ps_azimuth": float(np.arctan2(dy20, dx20))})
    return pd.DataFrame(recs)


def traj_per_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ms = np.where(df["delta_MD_from_PS"].to_numpy(float) < 1., 1., df["delta_MD_from_PS"].to_numpy(float))
    df["dZ_dMD_from_ps"] = df["delta_Z_from_PS"].astype(float) / ms
    df["dX_dMD_from_ps"] = df["delta_X_from_PS"].astype(float) / ms
    df["dY_dMD_from_ps"] = df["delta_Y_from_PS"].astype(float) / ms
    df["horiz_disp_from_ps"] = np.sqrt(df["delta_X_from_PS"].astype(float) ** 2
                                       + df["delta_Y_from_PS"].astype(float) ** 2)
    df["azimuth_from_ps"] = np.arctan2(df["delta_Y_from_PS"].astype(float),
                                       df["delta_X_from_PS"].astype(float))
    hl = df["hidden_length"].to_numpy(float)
    df["kh_ratio"] = df["known_length"].astype(float) / np.where(hl < 1., 1., hl)
    df["hidden_frac"] = hl / df["n_rows_in_well"].astype(float)
    return df


# ── Group D GR features (from exp008) ─────────────────────────────────────────
def gr_per_well(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        gr = g["GR"].to_numpy(float); md = g["MD"].to_numpy(float)
        valid = ~np.isnan(gr); nv = int(valid.sum()); nt = len(gr)
        if nv == 0:
            recs.append({"well_id": wid, "pre_ps_gr_mean": global_gr_mean,
                         "pre_ps_gr_std": 0., "pre_ps_gr_last20_mean": global_gr_mean,
                         "pre_ps_gr_trend": 0., "pre_ps_gr_available_frac": 0.}); continue
        gv = gr[valid]; mv = md[valid]; m20 = min(20, len(gv)); d_md = mv[-1] - mv[0]
        recs.append({"well_id": wid,
                     "pre_ps_gr_mean": float(np.nanmean(gr)),
                     "pre_ps_gr_std": float(np.nanstd(gr)),
                     "pre_ps_gr_last20_mean": float(gv[-m20:].mean()),
                     "pre_ps_gr_trend": (gv[-1] - gv[0]) / d_md if abs(d_md) > 1e-6 else 0.,
                     "pre_ps_gr_available_frac": nv / nt})
    return pd.DataFrame(recs)


def gr_per_row(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    df = df.sort_values(["well_id", "row_idx"]).copy()
    gr_f = df["GR"].copy().astype(float)
    gr_f[gr_f.isna()] = df.loc[gr_f.isna(), "pre_ps_gr_mean"].fillna(global_gr_mean)
    std_s = df["pre_ps_gr_std"].fillna(1.).replace(0., 1.)
    df["gr_vs_pre_ps_mean"] = gr_f - df["pre_ps_gr_mean"].fillna(global_gr_mean)
    df["gr_z_score"] = df["gr_vs_pre_ps_mean"] / std_s
    df["_gr_f"] = gr_f
    for w, c in [(20, "gr_rolling_mean_w20"), (50, "gr_rolling_mean_w50")]:
        df[c] = df.groupby("well_id", sort=False)["_gr_f"].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
    df["gr_rolling_std_w20"] = df.groupby("well_id", sort=False)["_gr_f"].transform(
        lambda x: x.rolling(20, min_periods=2).std().fillna(0.))
    return df.drop(columns=["_gr_f"])


def enrich(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    df = df.merge(traj_per_well(df), on="well_id", how="left")
    df = traj_per_row(df)
    df = df.merge(gr_per_well(df, global_gr_mean), on="well_id", how="left")
    df = gr_per_row(df, global_gr_mean)
    return df


# ── folds (greedy load-balanced GroupKFold by well, replicates make_folds.py) ─
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


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"INPUT_DIR={INPUT_DIR}")
    train = load_base(TRAIN_DIR, "train")
    test = load_base(TEST_DIR, "test")
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    global_gr_mean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    print(f"global_gr_mean={global_gr_mean:.4f}")

    train = enrich(train, global_gr_mean)
    test = enrich(test, global_gr_mean)

    fold_of = make_folds(train)
    train["fold"] = train["well_id"].map(fold_of)

    train_t = train[train["is_target"].astype(bool)].copy()
    test_t = test[test["is_target"].astype(bool)].copy()
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    params = {
        "objective": "regression", "metric": "rmse", "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 50,
        "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
        "lambda_l2": 1.0, "verbosity": -1, "seed": 42, "num_threads": 4,
    }

    test_pred_delta = np.zeros(len(test_t), dtype=float)
    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy(); tm = ~vm
        model = lgb.LGBMRegressor(**params, n_estimators=1500)
        model.fit(train_t.loc[tm, ALL_FEATURES], y_delta.loc[tm],
                  eval_set=[(train_t.loc[vm, ALL_FEATURES], y_delta.loc[vm])],
                  eval_metric="rmse",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(model.best_iteration_ or model.n_estimators)
        test_pred_delta += model.predict(test_t[ALL_FEATURES], num_iteration=best) / N_SPLITS
        print(f"fold {fold}: best_iter={best}")

    test_t[PRED_COL] = test_t["last_known_TVT"].astype(float).to_numpy() + test_pred_delta
    submission = sample[["id"]].merge(
        test_t[["id", PRED_COL]].rename(columns={PRED_COL: "tvt"}), on="id", how="left")
    if submission["tvt"].isna().any():
        raise ValueError("submission contains missing predictions")
    submission.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} rows={len(submission)}")


if __name__ == "__main__":
    main()
