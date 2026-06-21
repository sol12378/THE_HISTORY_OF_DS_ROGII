"""ROGII exp043 — flagship integrated kernel.

Final pipeline integrating:
  1. exp026 base (PF + geom blend, leak-free self-contained)
  2. Multi-scale PF (scales {3,5,8,12} likelihood-weighted ensemble)
  3. Residual GBDT correction (LGB/XGB/CB on PF residuals)
  4. P2 features (3D tortuosity + spatial KNN formation surface)
  5. Final blend: 0.76 * exp041_pf_residual_gbdt + 0.24 * geom

LB 8.672 (exp026 baseline) → target: ~8.0+ with multi-scale PF + residual GBDT

Architecture:
  - train/test: enrich with base features (Group A/B/C/D/F geometric)
  - train: compute multi-scale PF per well (preds[128, n], log_liks[128])
  - train: compute P2 features (tortuosity + KNN surface)
  - train: fit residual GBDT (target = TVT - PF_pred) on 5-fold GroupKFold
  - test: multi-scale PF per well + residual GBDT + geom fallback
  - blend: 0.76 * (pf + residual) + 0.24 * geom + smooth(w=101)

Leak-free: hidden TVT not used; GR/typewell/anchor/Z/MD/X/Y only.
Self-contained: all feature computation & PF in this kernel.

Kaggle notebook: /kaggle/working/submission.csv
"""
from pathlib import Path
import re
import os
from concurrent.futures import ProcessPoolExecutor
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

from scipy.spatial import cKDTree
from scipy.signal import savgol_filter

# ── configuration ─────────────────────────────────────────────────────────────
PF_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 1))
PF_SEEDS = 128
PF_PARTICLES = 500
PF_SCALES = [3.0, 5.0, 8.0, 12.0]
PF_INIT_SPREAD = 4.0
PF_SCALE_TEMPERATURE = 8.0  # fallback single scale
PF_PN = 0.01
PF_VN = 0.002
PF_MOM = 0.998
PF_RP = 0.1
PF_RR = 0.001
PF_RESAMP = 0.5

BLEND_PF_RESIDUAL = 0.76
BLEND_GEOM = 0.24
SMOOTH_W = 101

N_SPLITS = 5
SEED = 42

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── locations ─────────────────────────────────────────────────────────────────
def find_input_dir() -> Path:
    """Find Kaggle input or local data/raw."""
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

# Feature sets
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

BASE_FEATURES = (SAFE_FEATURES + GROUP_A + GROUP_B_WELL + GROUP_B_ROW
                 + GROUP_C + GROUP_D_WELL + GROUP_D_ROW + GROUP_F_WELL + GROUP_F_ROW)

P2_FEATURES = [
    "tort_3d", "tort_xy", "tort_vert",
    "dls_mean", "dls_last30",
    "inclination_last", "azimuth_change",
    "knn_surface", "knn_surface_minus_Z",
]

_GBDT_FEATURES_RAW = (
    BASE_FEATURES
    + ["last_known_TVT", "pred_tvt", "pf_delta"]
    + P2_FEATURES
    + ["gr_roll20_mean", "gr_roll50_mean", "gr_roll20_std"]
)
# dedupe while preserving order (last_known_TVT already in BASE_FEATURES)
GBDT_FEATURES = list(dict.fromkeys(_GBDT_FEATURES_RAW))

# ── base feature reconstruction ────────────────────────────────────────────────
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


# ── Group A/B/C ───────────────────────────────────────────────────────────────
def traj_per_well(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        md = g["MD"].to_numpy(float)
        x = g["X"].to_numpy(float)
        y = g["Y"].to_numpy(float)
        z = g["Z"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float)
        n = len(g)
        n20 = min(20, n)
        n5 = min(5, n)
        d20 = md[-1] - md[-n20] if n20 > 1 else 1.0
        d5 = md[-1] - md[-n5] if n5 > 1 else 1.0
        s20 = (tv[-1] - tv[-n20]) / d20 if abs(d20) > 1e-6 else 0.0
        s5 = (tv[-1] - tv[-n5]) / d5 if abs(d5) > 1e-6 else 0.0
        dz = (z[-1] - z[-n20]) / d20 if abs(d20) > 1e-6 else 0.0
        dx = (x[-1] - x[-n20]) / d20 if abs(d20) > 1e-6 else 0.0
        dy = (y[-1] - y[-n20]) / d20 if abs(d20) > 1e-6 else 0.0
        dx20 = x[-1] - x[-n20]
        dy20 = y[-1] - y[-n20]
        hd = float(np.sqrt(dx20 ** 2 + dy20 ** 2))
        recs.append({
            "well_id": wid,
            "pre_ps_tvt_slope_last20": s20,
            "pre_ps_tvt_slope_last5": s5,
            "pre_ps_tvt_curvature": s5 - s20,
            "pre_ps_tvt_delta_last20": tv[-1] - tv[-n20],
            "pre_ps_dZ_dMD": dz,
            "pre_ps_dX_dMD": dx,
            "pre_ps_dY_dMD": dy,
            "pre_ps_horiz_dMD": hd / d20 if abs(d20) > 1e-6 else 0.0,
            "pre_ps_azimuth": float(np.arctan2(dy20, dx20)),
        })
    return pd.DataFrame(recs)


def traj_per_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ms = np.where(
        df["delta_MD_from_PS"].to_numpy(float) < 1.0, 1.0, df["delta_MD_from_PS"].to_numpy(float)
    )
    df["dZ_dMD_from_ps"] = df["delta_Z_from_PS"].astype(float) / ms
    df["dX_dMD_from_ps"] = df["delta_X_from_PS"].astype(float) / ms
    df["dY_dMD_from_ps"] = df["delta_Y_from_PS"].astype(float) / ms
    df["horiz_disp_from_ps"] = np.sqrt(
        df["delta_X_from_PS"].astype(float) ** 2 + df["delta_Y_from_PS"].astype(float) ** 2
    )
    df["azimuth_from_ps"] = np.arctan2(
        df["delta_Y_from_PS"].astype(float), df["delta_X_from_PS"].astype(float)
    )
    hl = df["hidden_length"].to_numpy(float)
    df["kh_ratio"] = df["known_length"].astype(float) / np.where(hl < 1.0, 1.0, hl)
    df["hidden_frac"] = hl / df["n_rows_in_well"].astype(float)
    return df


# ── Group D ───────────────────────────────────────────────────────────────────
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
                "pre_ps_gr_std": 0.0,
                "pre_ps_gr_last20_mean": global_gr_mean,
                "pre_ps_gr_trend": 0.0,
                "pre_ps_gr_available_frac": 0.0,
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
            "pre_ps_gr_trend": (gv[-1] - gv[0]) / d_md if abs(d_md) > 1e-6 else 0.0,
            "pre_ps_gr_available_frac": nv / nt,
        })
    return pd.DataFrame(recs)


def gr_per_row(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    df = df.sort_values(["well_id", "row_idx"]).copy()
    gr_f = df["GR"].copy().astype(float)
    gr_f[gr_f.isna()] = df.loc[gr_f.isna(), "pre_ps_gr_mean"].fillna(global_gr_mean)
    std_s = df["pre_ps_gr_std"].fillna(1.0).replace(0.0, 1.0)
    df["gr_vs_pre_ps_mean"] = gr_f - df["pre_ps_gr_mean"].fillna(global_gr_mean)
    df["gr_z_score"] = df["gr_vs_pre_ps_mean"] / std_s
    df["_gr_f"] = gr_f
    for w, c in [(20, "gr_rolling_mean_w20"), (50, "gr_rolling_mean_w50")]:
        df[c] = df.groupby("well_id", sort=False)["_gr_f"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
    df["gr_rolling_std_w20"] = df.groupby("well_id", sort=False)["_gr_f"].transform(
        lambda x: x.rolling(20, min_periods=2).std().fillna(0.0)
    )
    return df.drop(columns=["_gr_f"])


# ── Group F geometric extrapolation ───────────────────────────────────────────
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


# ── P2 Features: 3D tortuosity + spatial KNN formation surface ────────────────
def compute_3d_tortuosity(group_df: pd.DataFrame) -> dict:
    """
    Compute tortuosity metrics for a well's known TVT section.
    Returns dict with keys: tort_3d, tort_xy, tort_vert, dls_mean, dls_last30,
                            inclination_last, azimuth_change
    All values are scalars (broadcast to all rows of the well).
    """
    known_mask = group_df['is_known_tvt'].values

    if known_mask.sum() < 3:
        return {
            'tort_3d': 1.0,
            'tort_xy': 1.0,
            'tort_vert': 1.0,
            'dls_mean': 0.0,
            'dls_last30': 0.0,
            'inclination_last': 0.0,
            'azimuth_change': 0.0,
        }

    known_df = group_df[known_mask].copy()
    X = known_df['X'].values
    Y = known_df['Y'].values
    Z = known_df['Z'].values
    MD = known_df['MD'].values

    # === 3D Tortuosity ===
    dX = np.diff(X)
    dY = np.diff(Y)
    dZ = np.diff(Z)
    segment_lengths = np.sqrt(dX**2 + dY**2 + dZ**2)
    total_path_length = segment_lengths.sum()
    straight_dist = np.sqrt((X[-1] - X[0])**2 + (Y[-1] - Y[0])**2 + (Z[-1] - Z[0])**2)

    if straight_dist > 1e-6:
        tort_3d = total_path_length / straight_dist
    else:
        tort_3d = 1.0

    # === XY (Horizontal) Tortuosity ===
    dX_h = np.diff(X)
    dY_h = np.diff(Y)
    segment_lengths_h = np.sqrt(dX_h**2 + dY_h**2)
    total_path_length_h = segment_lengths_h.sum()
    straight_dist_h = np.sqrt((X[-1] - X[0])**2 + (Y[-1] - Y[0])**2)

    if straight_dist_h > 1e-6:
        tort_xy = total_path_length_h / straight_dist_h
    else:
        tort_xy = 1.0

    # === Vertical Tortuosity (MD vs TVD) ===
    TVD_range = Z[-1] - Z[0]
    MD_range = MD[-1] - MD[0]

    if MD_range > 1e-6:
        tort_vert = MD_range / abs(TVD_range) if TVD_range != 0 else 1.0
    else:
        tort_vert = 1.0

    # === Dogleg Severity (DLS) ===
    if len(segment_lengths) > 1:
        dirs = np.column_stack([dX, dY, dZ]) / segment_lengths[:, None]
        dot_products = np.sum(dirs[:-1] * dirs[1:], axis=1)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(dot_products)

        dMD = np.diff(MD)
        dMD_span = dMD[1:] + dMD[:-1]
        dLS_raw = angles / (dMD_span + 1e-8)
        dls_mean = np.nanmean(dLS_raw)
        dls_mean = max(0.0, dls_mean)
    else:
        dls_mean = 0.0

    # === DLS over last 30 segments ===
    if len(segment_lengths) >= 30:
        dX_last = dX[-30:]
        dY_last = dY[-30:]
        dZ_last = dZ[-30:]
        seg_last = segment_lengths[-30:]
        MD_last = MD[-30:]

        if len(seg_last) > 1:
            dirs_last = np.column_stack([dX_last, dY_last, dZ_last]) / seg_last[:, None]
            dot_products_last = np.sum(dirs_last[:-1] * dirs_last[1:], axis=1)
            dot_products_last = np.clip(dot_products_last, -1, 1)
            angles_last = np.arccos(dot_products_last)

            dMD_last_array = np.diff(MD_last)
            dMD_span_last = dMD_last_array[1:] + dMD_last_array[:-1]
            if len(dMD_span_last) > 0:
                dls_raw_last = angles_last[:len(dMD_span_last)] / (dMD_span_last + 1e-8)
                dls_last30 = np.nanmean(dls_raw_last)
                dls_last30 = max(0.0, dls_last30)
            else:
                dls_last30 = dls_mean
        else:
            dls_last30 = dls_mean
    else:
        dls_last30 = dls_mean

    # === Inclination of Last Known Row ===
    if len(X) > 1:
        horiz_disp = np.sqrt((X[-1] - X[-2])**2 + (Y[-1] - Y[-2])**2)
        vert_disp = abs(Z[-1] - Z[-2])
        inclination_last = np.arctan2(horiz_disp, vert_disp)
    else:
        inclination_last = 0.0

    # === Azimuth Change ===
    if len(X) > 1:
        az_start = np.arctan2(Y[1] - Y[0], X[1] - X[0])
        az_end = np.arctan2(Y[-1] - Y[-2], X[-1] - X[-2])
        azimuth_change = abs(az_end - az_start)
        azimuth_change = min(azimuth_change, 2 * np.pi - azimuth_change)
    else:
        azimuth_change = 0.0

    return {
        'tort_3d': float(tort_3d),
        'tort_xy': float(tort_xy),
        'tort_vert': float(tort_vert),
        'dls_mean': float(dls_mean),
        'dls_last30': float(dls_last30),
        'inclination_last': float(inclination_last),
        'azimuth_change': float(azimuth_change),
    }


def compute_tortuosity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-well tortuosity features."""
    df = df.copy()

    # Initialize new columns
    for feat in ['tort_3d', 'tort_xy', 'tort_vert', 'dls_mean', 'dls_last30', 'inclination_last', 'azimuth_change']:
        df[feat] = 0.0

    # Compute per-well
    tort_features = {}
    for well_id, group in df.groupby('well_id', sort=False):
        tort_dict = compute_3d_tortuosity(group)
        tort_features[well_id] = tort_dict

    # Broadcast to all rows
    for key in ['tort_3d', 'tort_xy', 'tort_vert', 'dls_mean', 'dls_last30', 'inclination_last', 'azimuth_change']:
        df[key] = df['well_id'].map(lambda w: tort_features[w][key])

    return df


def build_knn_surface(
    train_df: pd.DataFrame,
    n_neighbors: int = 8,
    subsample_limit: int = 100000,
) -> tuple:
    """
    Build spatial KNN tree from train known TVT points.

    Returns:
        (tree, surface_values) where:
          - tree: cKDTree for (X, Y) coordinates
          - surface_values: array of (TVT_input + Z) for each point
    """
    known_mask = train_df['is_known_tvt'].values
    known_df = train_df[known_mask].copy()

    if len(known_df) == 0:
        return None, None

    # Surface value = TVT_input + Z (predicted surface at (X,Y))
    surface_values = (known_df['TVT_input'].values + known_df['Z'].values).astype(np.float32)

    # Coordinates for tree
    coords = known_df[['X', 'Y']].values.astype(np.float32)

    # Subsample if too large (memory constraint)
    if subsample_limit and len(coords) > subsample_limit:
        idx = np.random.choice(len(coords), subsample_limit, replace=False)
        coords = coords[idx]
        surface_values = surface_values[idx]

    tree = cKDTree(coords)
    return tree, surface_values


def query_knn_surface(
    query_coords: np.ndarray,
    tree: cKDTree,
    surface_values: np.ndarray,
    n_neighbors: int = 8,
) -> np.ndarray:
    """
    Query KNN surface for each point in query_coords.

    Params:
        query_coords: (N, 2) array of (X, Y)
        tree: cKDTree from build_knn_surface
        surface_values: array of surface values
        n_neighbors: k for KNN

    Returns:
        (N,) array of distance-weighted average surface values
    """
    distances, indices = tree.query(query_coords, k=n_neighbors)

    # Handle edge case of single point query
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Distance-weighted average (inverse distance weighting)
    eps = 1e-6
    weights = 1.0 / (distances + eps)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predictions = np.sum(surface_values[indices] * weights, axis=1)

    return predictions


def compute_knn_surface_features(train: pd.DataFrame, test: pd.DataFrame, n_neighbors: int = 8) -> tuple:
    """Compute KNN surface features using (X,Y) coords and (TVT_input+Z) surface values."""
    train = train.copy()
    test = test.copy()

    # Build KNN tree from train known points
    tree, surface_values = build_knn_surface(train, n_neighbors=n_neighbors, subsample_limit=100000)

    if tree is None:
        train["knn_surface"] = 0.0
        train["knn_surface_minus_Z"] = 0.0
        test["knn_surface"] = 0.0
        test["knn_surface_minus_Z"] = 0.0
        return train, test

    # Query train
    coords_train = train[['X', 'Y']].values.astype(np.float32)
    knn_pred_train = query_knn_surface(coords_train, tree, surface_values, n_neighbors=n_neighbors)
    train["knn_surface"] = knn_pred_train
    train["knn_surface_minus_Z"] = knn_pred_train - train['Z'].values

    # Query test
    coords_test = test[['X', 'Y']].values.astype(np.float32)
    knn_pred_test = query_knn_surface(coords_test, tree, surface_values, n_neighbors=n_neighbors)
    test["knn_surface"] = knn_pred_test
    test["knn_surface_minus_Z"] = knn_pred_test - test['Z'].values

    return train, test


# ── Particle Filter (multi-scale) ─────────────────────────────────────────────
def load_typewell(split_dir: Path, well_id: str) -> pd.DataFrame:
    p = split_dir / f"{well_id}__typewell.csv"
    if not p.exists():
        return None
    tw = pd.read_csv(p)
    return tw[["TVT", "GR"]].copy()


def pf_single(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed):
    """Single seed PF simulation."""
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = PF_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    prev_MD = last_MD
    log_lik = 0.0
    lo = tw_tvt[0] - 100
    hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PF_PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.0)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < PF_RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(N)
            rate = rate[idx] + PF_RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    return res, log_lik


def pf_build_payload(g: pd.DataFrame, tw: pd.DataFrame, is_train: bool = False) -> dict:
    """Build PF input payload (pickle-able dict)."""
    g = g.sort_values("row_idx")
    known = g[g["is_known_tvt"].astype(bool)]
    tgt = g[g["is_target"].astype(bool)]
    wid = str(tgt["well_id"].iloc[0]) if len(tgt) > 0 else str(g["well_id"].iloc[0])
    anchor = float(tgt["last_known_TVT"].iloc[0]) if len(tgt) > 0 else float(known["last_known_TVT"].iloc[0])
    n = int(len(tgt))
    if tw is None or len(tw) < 2 or len(known) < 2:
        return {"wid": wid, "n": n, "no_tw": True, "anchor": anchor}
    tw_s = tw.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    gr_full = g["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    tgt_mask = g["is_target"].astype(bool).to_numpy()
    gr_v = gr_full[tgt_mask]
    k_tvt = known["TVT_input"].to_numpy(float)
    k_gr = known["GR"].fillna(0).to_numpy(float)
    gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10.0, 60.0))
    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0
    ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    last = known.iloc[-1]
    return {
        "wid": wid,
        "n": n,
        "no_tw": False,
        "anchor": anchor,
        "tw_tvt": tw_tvt,
        "tw_gr": tw_gr,
        "md_v": tgt["MD"].to_numpy(float),
        "z_v": tgt["Z"].to_numpy(float),
        "gr_v": gr_v,
        "gs": gs,
        "ir": ir,
        "last_tvt": float(last["TVT_input"]),
        "last_Z": float(last["Z"]),
        "last_MD": float(last["MD"]),
    }


def pf_worker(p):
    """Process 1 well: 128 seeds, multi-scale ensemble."""
    wid = p["wid"]
    n = p["n"]
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])

    # Run 128 seeds, collect per-seed preds & log-liks
    preds = np.empty((PF_SEEDS, n))
    liks = np.empty(PF_SEEDS)
    for s in range(PF_SEEDS):
        preds[s], liks[s] = pf_single(
            p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
            p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"], s
        )

    # Multi-scale ensemble average
    multi_pred = np.zeros(n)
    for scale in PF_SCALES:
        wts = np.exp((liks - liks.max()) / scale)
        wts /= wts.sum()
        multi_pred += (wts[:, None] * preds).sum(0) / len(PF_SCALES)

    return wid, multi_pred


def pf_run(df, split_dir, is_train=False):
    """Run PF on all wells, return dict {wid -> pred_array}."""
    payloads = []
    for wid, g in df.groupby("well_id", sort=False):
        if not g["is_target"].any():
            continue
        tw = load_typewell(split_dir, wid)
        payloads.append(pf_build_payload(g, tw, is_train=is_train))

    n_wells = len(payloads)
    logger.info(f"PF: {n_wells} wells, workers={PF_WORKERS}, seeds={PF_SEEDS}, scales={PF_SCALES}")

    pf_by_wid = {}
    if PF_WORKERS > 1 and n_wells > 1:
        with ProcessPoolExecutor(max_workers=PF_WORKERS) as ex:
            for k, (wid, pred) in enumerate(ex.map(pf_worker, payloads, chunksize=1)):
                pf_by_wid[wid] = pred
                if (k + 1) % 50 == 0:
                    logger.info(f"  PF {k+1}/{n_wells} wells done")
    else:
        for k, p in enumerate(payloads):
            wid, pred = pf_worker(p)
            pf_by_wid[wid] = pred

    df_target = df[df["is_target"].astype(bool)].copy()
    df_target["pf_pred"] = np.nan
    for wid, pred in pf_by_wid.items():
        order = df_target.loc[df_target["well_id"].eq(wid)].sort_values("row_idx").index
        df_target.loc[order, "pf_pred"] = pred
    return df_target


# ── GBDT Residual Model ───────────────────────────────────────────────────────
def make_folds(train: pd.DataFrame, n_splits: int = N_SPLITS) -> dict:
    """Load-balanced GroupKFold by well."""
    stats = train[train["is_target"]].groupby("well_id", as_index=False).agg(target_rows=("row_idx", "size"))
    loads = [0] * n_splits
    fold_of = {}
    for row in stats.sort_values("target_rows", ascending=False).itertuples(index=False):
        f = min(range(n_splits), key=lambda i: loads[i])
        loads[f] += int(row.target_rows)
        fold_of[row.well_id] = f
    return fold_of


def fit_residual_gbdt(train: pd.DataFrame, fold_of: dict, n_splits: int = N_SPLITS):
    """Fit GBDT on (TVT - PF_pred), return OOF & CV."""
    train_t = train[train["is_target"].astype(bool)].copy()
    train_t["fold"] = train_t["well_id"].map(fold_of)

    # Target: residual = TVT - PF_pred
    y_residual = train_t["TVT"].astype(float) - train_t["pf_pred"].astype(float)

    # Feature selection (handle missing)
    feature_cols = [c for c in GBDT_FEATURES if c in train_t.columns]
    X = train_t[feature_cols].fillna(0.0)

    oof_pred = np.zeros(len(train_t))
    models_by_fold = {}

    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "verbose": -1,
    }

    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy()
        tm = ~vm
        X_train, X_val = X.loc[tm], X.loc[vm]
        y_train, y_val = y_residual.loc[tm], y_residual.loc[vm]

        # LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params, n_estimators=1500, random_state=SEED)
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        pred_lgb = model_lgb.predict(X_val)

        # XGBoost (if available)
        if HAS_XGB:
            xgb_params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 100,
                "subsample": 0.8,
                "tree_method": "hist",
            }
            model_xgb = xgb.XGBRegressor(**xgb_params, n_estimators=1500, random_state=SEED)
            model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred_xgb = model_xgb.predict(X_val)
        else:
            pred_xgb = pred_lgb

        # CatBoost (if available)
        if HAS_CB:
            model_cb = CatBoostRegressor(
                iterations=1500,
                learning_rate=0.03,
                depth=6,
                loss_function="RMSE",
                verbose=False,
                random_state=SEED,
            )
            model_cb.fit(X_train, y_train, eval_set=(X_val, y_val))
            pred_cb = model_cb.predict(X_val)
        else:
            pred_cb = pred_lgb

        # Ensemble: average
        pred_ens = (pred_lgb + pred_xgb + pred_cb) / 3.0
        oof_pred[vm] = pred_ens

        models_by_fold[fold] = (model_lgb, model_xgb if HAS_XGB else None, model_cb if HAS_CB else None)
        rmse_val = float(np.sqrt(np.mean((y_val - pred_ens) ** 2)))
        logger.info(f"Residual GBDT fold {fold}: RMSE={rmse_val:.4f}")

    train_t["residual_pred"] = oof_pred
    cv_rmse = float(np.sqrt(np.mean((y_residual - oof_pred) ** 2)))
    logger.info(f"Residual GBDT CV RMSE: {cv_rmse:.4f}")
    return train_t, models_by_fold, cv_rmse


def predict_residual_gbdt(test: pd.DataFrame, models_by_fold: dict, fold_of: dict, n_splits: int = N_SPLITS):
    """Predict test residuals using avg of fold models."""
    test_t = test[test["is_target"].astype(bool)].copy()
    feature_cols = [c for c in GBDT_FEATURES if c in test_t.columns]
    X_test = test_t[feature_cols].fillna(0.0)

    test_residual_pred = np.zeros(len(test_t))
    for fold in range(n_splits):
        if fold not in models_by_fold:
            continue
        model_lgb, model_xgb, model_cb = models_by_fold[fold]

        pred_lgb = model_lgb.predict(X_test)
        pred_xgb = model_xgb.predict(X_test) if model_xgb is not None else pred_lgb
        pred_cb = model_cb.predict(X_test) if model_cb is not None else pred_lgb
        pred_ens = (pred_lgb + pred_xgb + pred_cb) / 3.0
        test_residual_pred += pred_ens / n_splits

    test_t["residual_pred"] = test_residual_pred
    return test_t


# ── Geom LightGBM (from exp026) ───────────────────────────────────────────────
def fit_geom_lgb(train: pd.DataFrame, fold_of: dict = None, n_splits: int = N_SPLITS):
    """LightGBM delta regression (geom baseline)."""
    if fold_of is None:
        fold_of = make_folds(train, n_splits)
    train_t = train[train["is_target"].astype(bool)].copy()
    train_t["fold"] = train_t["well_id"].map(fold_of)
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    lgb_params = {
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
        "seed": SEED,
        "num_threads": 4,
    }

    geom_delta_test = np.zeros(len(train_t))
    geom_models = {}

    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy()
        tm = ~vm
        X_train = train_t.loc[tm, BASE_FEATURES]
        X_val = train_t.loc[vm, BASE_FEATURES]
        y_train, y_val = y_delta.loc[tm], y_delta.loc[vm]

        model = lgb.LGBMRegressor(**lgb_params, n_estimators=1500)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = int(model.best_iteration_ or model.n_estimators)
        geom_delta_test += model.predict(train_t[BASE_FEATURES], num_iteration=best) / N_SPLITS
        geom_models[fold] = model
        logger.info(f"Geom fold {fold}: best_iter={best}")

    train_t["geom"] = train_t["last_known_TVT"].astype(float) + geom_delta_test
    return train_t, geom_models


def predict_geom_lgb(test: pd.DataFrame, geom_models: dict, train: pd.DataFrame, n_splits: int = N_SPLITS):
    """Predict test geom using avg of fold models."""
    test_t = test[test["is_target"].astype(bool)].copy()
    geom_delta_test = np.zeros(len(test_t))

    for fold in range(n_splits):
        if fold not in geom_models:
            continue
        model = geom_models[fold]
        geom_delta_test += model.predict(test_t[BASE_FEATURES]) / n_splits

    test_t["geom"] = test_t["last_known_TVT"].astype(float) + geom_delta_test
    return test_t


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info(f"INPUT_DIR={INPUT_DIR}")
    logger.info(f"OUT_DIR={OUT_DIR}")

    # Load base
    train = load_base(TRAIN_DIR, "train")
    test = load_base(TEST_DIR, "test")
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    logger.info(f"train: {len(train)} rows, test: {len(test)} rows, sample: {len(sample)} rows")

    global_gr_mean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    logger.info(f"global_gr_mean={global_gr_mean:.4f}")

    # Enrich base features (Group A/B/C/D/F)
    logger.info("Enriching base features...")
    train = enrich(train, global_gr_mean)
    test = enrich(test, global_gr_mean)

    # Add GR rolling stats (for exp041 compatibility)
    logger.info("Adding GR rolling features...")
    for df in [train, test]:
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
            df["gr_roll20_mean"] = 0
            df["gr_roll50_mean"] = 0
            df["gr_roll20_std"] = 0

    # P2 features
    logger.info("Computing P2 features (tortuosity + KNN)...")
    train = compute_tortuosity(train)
    test = compute_tortuosity(test)
    train, test = compute_knn_surface_features(train, test)

    # PF on train (for residual GBDT target & feature)
    # LOCAL_VERIFY: load pre-computed PF from exp040 instead of recomputing (local dev only)
    local_verify = os.environ.get("LOCAL_VERIFY", "").lower() in ("1", "true", "yes")
    if local_verify:
        logger.info("LOCAL_VERIFY: loading PF from exp040_multiscale_pf...")
        try:
            pf_oof_path = Path("experiments/exp040_multiscale_pf/oof.csv")
            if pf_oof_path.exists():
                pf_oof = pd.read_csv(pf_oof_path)
                train = train.merge(
                    pf_oof[["well_id", "row_idx", "pred_tvt"]],
                    on=["well_id", "row_idx"],
                    how="left"
                )
                train["pf_pred"] = train["pred_tvt"]
                logger.info(f"  loaded {len(train)} rows with pf_pred")
            else:
                raise FileNotFoundError(f"{pf_oof_path} not found, falling back to PF compute")
        except Exception as e:
            logger.warning(f"LOCAL_VERIFY failed: {e}, computing PF normally")
            logger.info("Running PF on train...")
            train_t = pf_run(train, TRAIN_DIR, is_train=True)
            train = train.merge(train_t[["row_idx", "well_id", "pf_pred"]], on=["row_idx", "well_id"], how="left")
    else:
        logger.info("Running PF on train...")
        train_t = pf_run(train, TRAIN_DIR, is_train=True)
        train = train.merge(train_t[["row_idx", "well_id", "pf_pred"]], on=["row_idx", "well_id"], how="left")

    # Add pf_delta feature
    train["pf_delta"] = train["pf_pred"] - train["last_known_TVT"]

    # Fold assignment
    fold_of = make_folds(train)
    train["fold"] = train["well_id"].map(fold_of)

    # Fit residual GBDT on train
    logger.info("Fitting residual GBDT...")
    train_t, models_by_fold, cv_rmse = fit_residual_gbdt(train, fold_of)
    logger.info(f"Residual GBDT CV={cv_rmse:.4f}")

    # Fit geom LightGBM on train
    logger.info("Fitting geom LightGBM...")
    train_geom, geom_models = fit_geom_lgb(train, fold_of=fold_of)

    # PF on test
    if local_verify:
        logger.info("LOCAL_VERIFY: loading test PF from exp040_multiscale_pf...")
        try:
            pf_test_path = Path("experiments/exp040_multiscale_pf/test_pred.csv")
            if pf_test_path.exists():
                pf_test = pd.read_csv(pf_test_path)
                test = test.merge(
                    pf_test[["id", "pred_tvt"]],
                    on="id",
                    how="left"
                )
                test["pf_pred"] = test["pred_tvt"]
                logger.info(f"  loaded {len(test)} rows with pf_pred")
                test_t = test[test["is_target"].astype(bool)].copy()
            else:
                raise FileNotFoundError(f"{pf_test_path} not found, falling back to PF compute")
        except Exception as e:
            logger.warning(f"LOCAL_VERIFY test PF failed: {e}, computing PF normally")
            logger.info("Running PF on test...")
            test_t = pf_run(test, TEST_DIR, is_train=False)
    else:
        logger.info("Running PF on test...")
        test_t = pf_run(test, TEST_DIR, is_train=False)

    # Predict residual GBDT on test
    logger.info("Predicting residual GBDT on test...")
    test_t = predict_residual_gbdt(test_t, models_by_fold, fold_of)

    # Predict geom on test
    logger.info("Predicting geom on test...")
    test_geom = predict_geom_lgb(test, geom_models, train)

    # Merge
    test_t = test_t.merge(test_geom[["row_idx", "well_id", "geom"]], on=["row_idx", "well_id"], how="left")

    # Final blend: 0.76 * (PF + residual) + 0.24 * geom
    logger.info("Final blending...")
    a = test_t["last_known_TVT"].astype(float).to_numpy()
    exp041_pred = test_t["pf_pred"].to_numpy(float) + test_t["residual_pred"].to_numpy(float)
    geom_pred = test_t["geom"].to_numpy(float)
    blend = a + BLEND_PF_RESIDUAL * (exp041_pred - a) + BLEND_GEOM * (geom_pred - a)
    test_t["blend"] = blend

    # Smoothing
    test_t = test_t.sort_values(["well_id", "row_idx"])
    test_t[PRED_COL] = test_t.groupby("well_id", sort=False)["blend"].transform(
        lambda x: x.rolling(SMOOTH_W, min_periods=1, center=True).mean()
    )

    # Submission
    submission = sample[["id"]].merge(
        test_t[["id", PRED_COL]].rename(columns={PRED_COL: "tvt"}), on="id", how="left"
    )
    if submission["tvt"].isna().any():
        raise ValueError("submission contains NaN predictions")
    submission.to_csv(OUT_PATH, index=False)
    logger.info(
        f"wrote {OUT_PATH} rows={len(submission)} tvt=[{submission.tvt.min():.1f}, {submission.tvt.max():.1f}]"
    )


if __name__ == "__main__":
    main()
