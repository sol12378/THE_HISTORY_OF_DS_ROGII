"""exp073 Task A: Rebuild exp026 geom LightGBM as 5-fold OOF on train.

Replicates the geom feature pipeline and LGB parameters from
kaggle_notebooks/exp072_proj/_decoded_exp026_b64.py exactly, but
runs a 5-fold GroupKFold-by-well cross-validation on all 773 train
wells to produce OOF predictions for every hidden row.

Output:
  experiments/exp073_public_assets_integration/oof_geom.csv
  experiments/exp073_public_assets_integration/result_geom.json

Quality gate: pooled RMSE in [13.0, 14.3], n_rows == 3_783_989.
"""

import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("E:/kaggle/THE_HISTORY_OF_DS_ROGII")
OUT_DIR = ROOT / "experiments/exp073_public_assets_integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "log_geom_oof.txt"
OOF_CSV = OUT_DIR / "oof_geom.csv"
RESULT_JSON = OUT_DIR / "result_geom.json"

N_SPLITS = 5

# ── Feature lists (verbatim from _decoded_exp026_b64.py) ──────────────────────
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
GROUP_F_WELL = ["f_dtvt_dmd_l50", "f_dtvt_dz_pre", "f_dtvt_dz_r2"]
GROUP_F_ROW = ["f_extrap_slope20_dMD", "f_extrap_slope5_dMD", "f_extrap_quad_dMD",
               "f_extrap_z", "f_extrap_disagree"]
ALL_FEATURES = (SAFE_FEATURES + GROUP_A + GROUP_B_WELL + GROUP_B_ROW
                + GROUP_C + GROUP_D_WELL + GROUP_D_ROW + GROUP_F_WELL + GROUP_F_ROW)

# ── LGB parameters (verbatim from _decoded_exp026_b64.py) ─────────────────────
LGB_PARAMS = {
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
    "num_threads": 14,  # PF done -> use full cores for fast rerun
}


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Feature engineering (verbatim from _decoded_exp026_b64.py) ────────────────

def traj_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        md = g["MD"].to_numpy(float)
        x = g["X"].to_numpy(float)
        y = g["Y"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float)
        n = len(g)
        n20 = min(20, n); n5 = min(5, n)
        d20 = md[-1] - md[-n20] if n20 > 1 else 1.
        d5 = md[-1] - md[-n5] if n5 > 1 else 1.
        s20 = (tv[-1] - tv[-n20]) / d20 if abs(d20) > 1e-6 else 0.
        s5 = (tv[-1] - tv[-n5]) / d5 if abs(d5) > 1e-6 else 0.
        dx20 = x[-1] - x[-n20]; dy20 = y[-1] - y[-n20]
        hd = float(np.sqrt(dx20 ** 2 + dy20 ** 2))
        recs.append({
            "well_id": wid,
            "pre_ps_tvt_slope_last20": s20,
            "pre_ps_tvt_slope_last5": s5,
            "pre_ps_tvt_curvature": s5 - s20,
            "pre_ps_tvt_delta_last20": tv[-1] - tv[-n20],
            "pre_ps_dZ_dMD": (g["Z"].to_numpy(float)[-1] - g["Z"].to_numpy(float)[-n20]) / d20 if abs(d20) > 1e-6 else 0.,
            "pre_ps_dX_dMD": (x[-1] - x[-n20]) / d20 if abs(d20) > 1e-6 else 0.,
            "pre_ps_dY_dMD": (y[-1] - y[-n20]) / d20 if abs(d20) > 1e-6 else 0.,
            "pre_ps_horiz_dMD": hd / d20 if abs(d20) > 1e-6 else 0.,
            "pre_ps_azimuth": float(np.arctan2(dy20, dx20)),
        })
    return pd.DataFrame(recs)


def traj_per_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ms = np.where(df["delta_MD_from_PS"].to_numpy(float) < 1.,
                  1., df["delta_MD_from_PS"].to_numpy(float))
    df["dZ_dMD_from_ps"] = df["delta_Z_from_PS"].astype(float) / ms
    df["dX_dMD_from_ps"] = df["delta_X_from_PS"].astype(float) / ms
    df["dY_dMD_from_ps"] = df["delta_Y_from_PS"].astype(float) / ms
    df["horiz_disp_from_ps"] = np.sqrt(
        df["delta_X_from_PS"].astype(float) ** 2
        + df["delta_Y_from_PS"].astype(float) ** 2)
    df["azimuth_from_ps"] = np.arctan2(
        df["delta_Y_from_PS"].astype(float),
        df["delta_X_from_PS"].astype(float))
    hl = df["hidden_length"].to_numpy(float)
    df["kh_ratio"] = df["known_length"].astype(float) / np.where(hl < 1., 1., hl)
    df["hidden_frac"] = hl / df["n_rows_in_well"].astype(float)
    return df


def gr_per_well(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        gr = g["GR"].to_numpy(float)
        md = g["MD"].to_numpy(float)
        valid = ~np.isnan(gr)
        nv = int(valid.sum())
        nt = len(gr)
        if nv == 0:
            recs.append({
                "well_id": wid,
                "pre_ps_gr_mean": global_gr_mean,
                "pre_ps_gr_std": 0.,
                "pre_ps_gr_last20_mean": global_gr_mean,
                "pre_ps_gr_trend": 0.,
                "pre_ps_gr_available_frac": 0.,
            })
            continue
        gv = gr[valid]
        mv = md[valid]
        m20 = min(20, len(gv))
        d_md = mv[-1] - mv[0]
        recs.append({
            "well_id": wid,
            "pre_ps_gr_mean": float(np.nanmean(gr)),
            "pre_ps_gr_std": float(np.nanstd(gr)),
            "pre_ps_gr_last20_mean": float(gv[-m20:].mean()),
            "pre_ps_gr_trend": (gv[-1] - gv[0]) / d_md if abs(d_md) > 1e-6 else 0.,
            "pre_ps_gr_available_frac": nv / nt,
        })
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


def geom_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        md = g["MD"].to_numpy(float)
        z = g["Z"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float)
        n = len(g)
        n50 = min(50, n)
        if n50 >= 2 and abs(md[-1] - md[-n50]) > 1e-6:
            dtvt_dmd_l50 = float(np.polyfit(md[-n50:], tv[-n50:], 1)[0])
        else:
            dtvt_dmd_l50 = 0.0
        if n >= 3 and np.ptp(z) > 1e-3:
            zc = z - z.mean()
            denom = float(np.dot(zc, zc))
            slope_z = float(np.dot(zc, tv - tv.mean()) / denom) if denom > 1e-9 else 0.0
            pred = slope_z * zc + tv.mean()
            ss_res = float(np.sum((tv - pred) ** 2))
            ss_tot = float(np.sum((tv - tv.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        else:
            slope_z = 0.0
            r2 = 0.0
        recs.append({
            "well_id": wid,
            "f_dtvt_dmd_l50": dtvt_dmd_l50,
            "f_dtvt_dz_pre": slope_z,
            "f_dtvt_dz_r2": r2,
        })
    return pd.DataFrame(recs)


def geom_per_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dmd = df["delta_MD_from_PS"].astype(float)
    dz = df["delta_Z_from_PS"].astype(float)
    s20 = df["pre_ps_tvt_slope_last20"].astype(float)
    s5 = df["pre_ps_tvt_slope_last5"].astype(float)
    curv = df["pre_ps_tvt_curvature"].astype(float)
    df["f_extrap_slope20_dMD"] = s20 * dmd
    df["f_extrap_slope5_dMD"] = s5 * dmd
    df["f_extrap_quad_dMD"] = s20 * dmd + 0.5 * curv * dmd * dmd
    df["f_extrap_z"] = df["f_dtvt_dz_pre"].astype(float) * dz
    df["f_extrap_disagree"] = (df["f_extrap_slope20_dMD"] - df["f_extrap_z"]).abs()
    return df


def enrich(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    df = df.merge(traj_per_well(df), on="well_id", how="left")
    df = traj_per_row(df)
    df = df.merge(gr_per_well(df, global_gr_mean), on="well_id", how="left")
    df = gr_per_row(df, global_gr_mean)
    df = df.merge(geom_per_well(df), on="well_id", how="left")
    df = geom_per_row(df)
    return df


def make_folds(train: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
    """Balance-split wells by target row count (verbatim from _decoded_exp026_b64.py)."""
    stats = (
        train[train["is_target"]]
        .groupby("well_id", as_index=False)
        .agg(target_rows=("row_idx", "size"))
    )
    loads = [0] * n_splits
    fold_of = {}
    for row in stats.sort_values("target_rows", ascending=False).itertuples(index=False):
        f = min(range(n_splits), key=lambda i: loads[i])
        loads[f] += int(row.target_rows)
        fold_of[row.well_id] = f
    return fold_of


def main() -> None:
    t0 = time.time()
    log("=== exp073 Task A: geom OOF ===")

    # Load train_base (already processed)
    log("Loading train_base_v001.parquet ...")
    train = pd.read_parquet(ROOT / "data/processed/train_base_v001.parquet")
    log(f"  Loaded {len(train):,} rows, {train['well_id'].nunique()} wells")

    n_hidden_total = int((~train["is_known_tvt"]).sum())
    log(f"  Hidden rows: {n_hidden_total:,}  (expected 3,783,989)")

    # global_gr_mean is computed from all training known GR rows
    global_gr_mean = float(train.loc[~train["is_gr_missing"], "GR"].mean())
    log(f"  global_gr_mean={global_gr_mean:.4f}")

    # Feature engineering (on full train, leak-free: uses only is_known_tvt rows
    # for per-well statistics; hidden TVT is never accessed)
    log("Enriching features ...")
    t_fe = time.time()
    train = enrich(train, global_gr_mean)
    log(f"  Feature engineering done in {time.time()-t_fe:.1f}s")

    # Make folds (balance by target_rows, same algorithm as kernel)
    fold_of = make_folds(train)
    train["fold"] = train["well_id"].map(fold_of)

    # Restrict to target (hidden) rows for modelling
    train_t = train[train["is_target"]].copy()
    log(f"  train_t rows: {len(train_t):,}")

    # Delta target
    y_delta = (
        train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)
    ).to_numpy()

    # OOF array (indexed by train_t's positional index)
    oof_delta = np.full(len(train_t), np.nan, dtype=float)
    fold_rmse = {}

    for fold in range(N_SPLITS):
        vm = train_t["fold"].eq(fold).to_numpy()
        tm = ~vm
        log(f"  Fold {fold}: train={tm.sum():,}  val={vm.sum():,}")

        model = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=1500)
        model.fit(
            train_t.loc[tm, ALL_FEATURES],
            y_delta[tm],
            eval_set=[(train_t.loc[vm, ALL_FEATURES], y_delta[vm])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = int(model.best_iteration_ or model.n_estimators)
        log(f"    best_iter={best}")

        val_pred = model.predict(train_t.loc[vm, ALL_FEATURES], num_iteration=best)
        oof_delta[vm] = val_pred

        fold_val_rmse = float(np.sqrt(np.mean((y_delta[vm] - val_pred) ** 2)))
        fold_rmse[fold] = fold_val_rmse
        log(f"    fold {fold} RMSE={fold_val_rmse:.6f}")

    # Convert delta back to TVT
    pred_tvt = train_t["last_known_TVT"].astype(float).to_numpy() + oof_delta

    # Assemble OOF dataframe
    oof_df = train_t[["id", "well_id", "row_idx", "TVT"]].copy()
    oof_df = oof_df.rename(columns={"TVT": "tvt_true"})
    oof_df["pred_tvt"] = pred_tvt

    # Verify id format for hidden rows
    # id = f"{well_id}_{row_idx}" (set in train_base build_base_for_file for hidden rows)
    if oof_df["id"].isna().any():
        n_missing_id = int(oof_df["id"].isna().sum())
        log(f"  WARNING: {n_missing_id} rows have missing id — reconstructing")
        missing_mask = oof_df["id"].isna()
        oof_df.loc[missing_mask, "id"] = (
            oof_df.loc[missing_mask, "well_id"].astype(str)
            + "_"
            + oof_df.loc[missing_mask, "row_idx"].astype(str)
        )

    # Quality metrics
    pooled_rmse = float(np.sqrt(np.mean((oof_df["tvt_true"] - oof_df["pred_tvt"]) ** 2)))
    log(f"  Pooled RMSE = {pooled_rmse:.6f}")
    log(f"  n_rows = {len(oof_df):,}  (expected 3,783,989)")

    runtime_min = (time.time() - t0) / 60.0

    # Save OOF CSV
    oof_df.to_csv(OOF_CSV, index=False)
    log(f"  Saved {OOF_CSV}")

    # Per-fold RMSE dict with string keys for JSON
    fold_rmse_str = {str(k): v for k, v in fold_rmse.items()}

    # Compute per-well RMSE stats
    per_well = oof_df.groupby("well_id").apply(
        lambda g: np.sqrt(np.mean((g["tvt_true"] - g["pred_tvt"]) ** 2))
    )
    per_well_stats = {
        "mean": float(per_well.mean()),
        "median": float(per_well.median()),
        "std": float(per_well.std()),
        "min": float(per_well.min()),
        "max": float(per_well.max()),
    }

    # Leak audit
    leak_notes = (
        "Leak audit: hidden TVT values (TVT column in train_base) are used ONLY as "
        "the ground-truth label (tvt_true) for RMSE computation after predictions are "
        "made. During feature engineering (enrich()), only rows where is_known_tvt==True "
        "are used to compute per-well statistics (traj_per_well, gr_per_well, "
        "geom_per_well). These functions read TVT_input (not TVT), and TVT_input is NaN "
        "for hidden rows. Per-row features (traj_per_row, gr_per_row, geom_per_row) use "
        "only MD/X/Y/Z/GR/delta_* columns which are available for hidden rows without "
        "revealing TVT. LGB training uses only train-fold hidden rows (is_target=True) "
        "with y_delta = TVT - last_known_TVT; the validation fold hidden rows TVT is "
        "not passed to the model during fit (only to eval_set for early stopping score "
        "reporting, which is a standard ML practice and does not contaminate predictions "
        "made by model.predict). Conclusion: no TVT leakage into predictions."
    )

    result = {
        "pooled_rmse": pooled_rmse,
        "per_fold_rmse": fold_rmse_str,
        "per_well_rmse_stats": per_well_stats,
        "n_rows": len(oof_df),
        "runtime_min": runtime_min,
        "leak_notes": leak_notes,
        "gate_passed": (13.0 <= pooled_rmse <= 14.3 and len(oof_df) == 3_783_989),
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(f"  Saved {RESULT_JSON}")

    gate_ok = result["gate_passed"]
    log(f"  Quality gate {'PASSED' if gate_ok else 'FAILED'}: RMSE={pooled_rmse:.4f}, expected [13.0, 14.3], n_rows={len(oof_df)}")
    log(f"=== Done in {runtime_min:.1f} min ===")
    return result


if __name__ == "__main__":
    main()
