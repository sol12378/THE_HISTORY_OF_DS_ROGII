"""ROGII exp073 SUBMIT   public-assets integration blend, deployable-only.
SINGLE SELF-CONTAINED FILE (no utility-dataset imports). Attach only:
  - competition data (rogii-wellbore-geology-prediction)
  - ravaghi/wellbore-geology-prediction-artifacts
  - pilkwang/rogii-model-package
  - phongnguyn23021656/koolbox-offline  (pip-installable koolbox wheel for ravaghi Trainer pickles)

================================ RECIPE ================================
pilk_tcn               (hidden-only     delta mean -35.6 vs OOF +1.6)
                      OOF NNLS  fit   nested CV 8.79 (leak-free) 

final_delta (DELTA space = TVT - last_known_TVT, fixed weights, NOT normalized):
    0.440*pf + 0.096*geom + 0.284*rav_lgb3 + 0.031*rav_cb1 + 0.030*rav_cb2 + 0.348*pilk_cat
  - pf, geom         :    exp026 (PF GR-tracker + geom LightGBM)
  - rav_lgb3/cb1/cb2 : ravaghi koolbox Trainer pickles (lightgbm-3 / catboost-1 / catboost-2)
  - pilk_cat         : pilkwang rogii-model-package drift_ncc CatBoost (load_model)

Post-processing (per well, in order):
  1. warmup decay: final_delta *= (1 - exp(-max(MD - last_known_MD, 0)/85))
  2. tvt = last_known_TVT + final_delta
  3. per-well rolling mean window=101 (center, min_periods=1)
  4. robust projection deg=4, beta=0.75 (U = tvt+Z - anchor vs normalized MD s)

LEAK-FREE: model inference + known geometry only. No train-TVT lookup, no overlap
override, no tvt_from_contacts. Hidden-row TVT is never read for prediction.
"""
import os, sys, json, time, warnings, traceback, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    def _find_comp():
        for p in [Path("/kaggle/input/rogii-wellbore-geology-prediction"),
                  Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")]:
            if (p / "test").exists():
                return p
        for p in Path("/kaggle/input").glob("*/sample_submission.csv"):
            return p.parent
        raise FileNotFoundError("competition data not found")
    DATA = _find_comp()
    EXT = Path("/kaggle/input")
    RAV_ROOTS = [EXT / "wellbore-geology-prediction-artifacts",
                 EXT / "datasets/ravaghi/wellbore-geology-prediction-artifacts"]
    PKG_ROOTS = [EXT / "rogii-model-package",
                 EXT / "rogii-model-package/rogii_model_package",
                 EXT / "datasets/pilkwang/rogii-model-package"]
    WORK = Path("/kaggle/working")
else:
    REPO = Path(__file__).resolve().parent.parent.parent
    DATA = REPO / "data" / "raw"
    EXT = REPO / "data" / "external"
    RAV_ROOTS = [EXT / "wellbore-geology-prediction-artifacts"]
    PKG_ROOTS = [EXT / "rogii-model-package"]
    WORK = REPO / "experiments" / "exp073_public_assets_integration"

WORK.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = DATA / "train"
TEST_DIR = DATA / "test"
SAMPLE_PATH = DATA / "sample_submission.csv"
OUT_PATH = WORK / "submission.csv"

# DELTA-space fixed blend weights (deployable-only, NOT normalized)
W = dict(pf=0.440, geom=0.096, rav_lgb3=0.284, rav_cb1=0.031, rav_cb2=0.030, pilk_cat=0.348)

TAU = 85.0
SMOOTH_W = 101
PROJ_DEG = 4
PROJ_BETA = 0.75
PROJ_ITERS = 4
PROJ_MIN_KNOWN = 8

_log_lines = []
def log(msg):
    line = f"[exp073] {msg}"
    print(line, flush=True)
    _log_lines.append(line)


# ====================================================================
# ================ BLOCK A: exp026 (pf + geom) inlined ===============
# ====================================================================
import re
import lightgbm as lgb

E026_N_SPLITS = 5
E026_PF_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))
E026_PF_SEEDS = 128
E026_PF_PARTICLES = 500
E026_PF_SCALE = 8.0
E026_PF_INIT_SPREAD = 4.0
E026_PF_PN = 0.01
E026_PF_VN = 0.002
E026_PF_MOM = 0.998
E026_PF_RP = 0.1
E026_PF_RR = 0.001
E026_PF_RESAMP = 0.5

_E_SAFE = [
    "MD", "X", "Y", "Z", "GR", "is_gr_missing", "n_rows_in_well",
    "known_length", "hidden_length", "last_known_TVT", "last_known_MD",
    "last_known_X", "last_known_Y", "last_known_Z", "delta_MD_from_PS",
    "delta_X_from_PS", "delta_Y_from_PS", "delta_Z_from_PS",
    "post_ps_step", "row_frac"]
_E_GA = ["pre_ps_tvt_slope_last20", "pre_ps_tvt_slope_last5",
         "pre_ps_tvt_curvature", "pre_ps_tvt_delta_last20"]
_E_GBW = ["pre_ps_dZ_dMD", "pre_ps_dX_dMD", "pre_ps_dY_dMD",
          "pre_ps_horiz_dMD", "pre_ps_azimuth"]
_E_GBR = ["dZ_dMD_from_ps", "dX_dMD_from_ps", "dY_dMD_from_ps",
          "horiz_disp_from_ps", "azimuth_from_ps"]
_E_GC = ["kh_ratio", "hidden_frac"]
_E_GDW = ["pre_ps_gr_mean", "pre_ps_gr_std", "pre_ps_gr_last20_mean",
          "pre_ps_gr_trend", "pre_ps_gr_available_frac"]
_E_GDR = ["gr_vs_pre_ps_mean", "gr_z_score", "gr_rolling_mean_w20",
          "gr_rolling_mean_w50", "gr_rolling_std_w20"]
_E_GFW = ["f_dtvt_dmd_l50", "f_dtvt_dz_pre", "f_dtvt_dz_r2"]
_E_GFR = ["f_extrap_slope20_dMD", "f_extrap_slope5_dMD", "f_extrap_quad_dMD",
          "f_extrap_z", "f_extrap_disagree"]
E026_ALL_FEATURES = (_E_SAFE + _E_GA + _E_GBW + _E_GBR + _E_GC + _E_GDW + _E_GDR + _E_GFW + _E_GFR)


def _e_well_id_from_path(path):
    return path.name.split("__", 1)[0]
def _e_natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path.name)]


def _e_build_base_for_file(path, split):
    raw = pd.read_csv(path)
    well_id = _e_well_id_from_path(path)
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


def _e_load_base(split_dir, split):
    paths = sorted(split_dir.glob("*__horizontal_well.csv"), key=_e_natural_key)
    return pd.concat([_e_build_base_for_file(p, split) for p in paths], ignore_index=True)


def _e_traj_per_well(df):
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


def _e_traj_per_row(df):
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


def _e_gr_per_well(df, global_gr_mean):
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


def _e_gr_per_row(df, global_gr_mean):
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


def _e_geom_per_well(df):
    known = df[df["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        md = g["MD"].to_numpy(float); z = g["Z"].to_numpy(float)
        tv = g["TVT_input"].to_numpy(float); n = len(g)
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
            ss_res = float(np.sum((tv - pred) ** 2)); ss_tot = float(np.sum((tv - tv.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0
        else:
            slope_z = 0.0; r2 = 0.0
        recs.append({"well_id": wid, "f_dtvt_dmd_l50": dtvt_dmd_l50,
                     "f_dtvt_dz_pre": slope_z, "f_dtvt_dz_r2": r2})
    return pd.DataFrame(recs)


def _e_geom_per_row(df):
    df = df.copy()
    dmd = df["delta_MD_from_PS"].astype(float); dz = df["delta_Z_from_PS"].astype(float)
    s20 = df["pre_ps_tvt_slope_last20"].astype(float); s5 = df["pre_ps_tvt_slope_last5"].astype(float)
    curv = df["pre_ps_tvt_curvature"].astype(float)
    df["f_extrap_slope20_dMD"] = s20 * dmd
    df["f_extrap_slope5_dMD"] = s5 * dmd
    df["f_extrap_quad_dMD"] = s20 * dmd + 0.5 * curv * dmd * dmd
    df["f_extrap_z"] = df["f_dtvt_dz_pre"].astype(float) * dz
    df["f_extrap_disagree"] = (df["f_extrap_slope20_dMD"] - df["f_extrap_z"]).abs()
    return df


def _e_enrich(df, global_gr_mean):
    df = df.merge(_e_traj_per_well(df), on="well_id", how="left")
    df = _e_traj_per_row(df)
    df = df.merge(_e_gr_per_well(df, global_gr_mean), on="well_id", how="left")
    df = _e_gr_per_row(df, global_gr_mean)
    df = df.merge(_e_geom_per_well(df), on="well_id", how="left")
    df = _e_geom_per_row(df)
    return df


def _e_make_folds(train, n_splits=E026_N_SPLITS):
    stats = (train[train["is_target"]].groupby("well_id", as_index=False)
             .agg(target_rows=("row_idx", "size")))
    loads = [0] * n_splits
    fold_of = {}
    for row in stats.sort_values("target_rows", ascending=False).itertuples(index=False):
        f = min(range(n_splits), key=lambda i: loads[i])
        loads[f] += int(row.target_rows)
        fold_of[row.well_id] = f
    return fold_of


def _e_load_typewell(split_dir, well_id):
    tw = pd.read_csv(split_dir / f"{well_id}__typewell.csv")
    return tw[["TVT", "GR"]].copy()


def _e_pf_single(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed):
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = E026_PF_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + E026_PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n); prev_MD = last_MD; log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = E026_PF_MOM * rate + E026_PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + E026_PF_PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr); d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < E026_PF_RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + E026_PF_RP * rng.standard_normal(N); rate = rate[idx] + E026_PF_RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    return res, log_lik


def _e_pf_build_payload(g, tw):
    g = g.sort_values("row_idx")
    known = g[g["is_known_tvt"].astype(bool)]
    tgt = g[g["is_target"].astype(bool)]
    wid = str(tgt["well_id"].iloc[0])
    anchor = float(tgt["last_known_TVT"].iloc[0])
    n = int(len(tgt))
    if tw is None or len(tw) < 2 or len(known) < 2:
        return {"wid": wid, "n": n, "no_tw": True, "anchor": anchor}
    tw_s = tw.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float); tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    gr_full = g["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    tgt_mask = g["is_target"].astype(bool).to_numpy()
    gr_v = gr_full[tgt_mask]
    k_tvt = known["TVT_input"].to_numpy(float); k_gr = known["GR"].fillna(0).to_numpy(float)
    gs = float(np.clip(np.nanstd(k_gr - np.interp(k_tvt, tw_tvt, tw_gr)), 10., 60.))
    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float)); dz = np.diff(tail["Z"].to_numpy(float)); dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0
    ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    last = known.iloc[-1]
    return {"wid": wid, "n": n, "no_tw": False, "anchor": anchor,
            "tw_tvt": tw_tvt, "tw_gr": tw_gr,
            "md_v": tgt["MD"].to_numpy(float), "z_v": tgt["Z"].to_numpy(float), "gr_v": gr_v,
            "gs": gs, "ir": ir, "last_tvt": float(last["TVT_input"]),
            "last_Z": float(last["Z"]), "last_MD": float(last["MD"])}


def _e_pf_worker(p):
    wid = p["wid"]; n = p["n"]
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])
    preds = np.empty((E026_PF_SEEDS, n)); liks = np.empty(E026_PF_SEEDS)
    for s in range(E026_PF_SEEDS):
        preds[s], liks[s] = _e_pf_single(p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
                                         p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"], s)
    wts = np.exp((liks - liks.max()) / E026_PF_SCALE); wts /= wts.sum()
    return wid, (wts[:, None] * preds).sum(0)


def compute_exp026():
    """Returns DataFrame[id, pf_delta, geom_delta] from raw test (self-contained)."""
    train = _e_load_base(TRAIN_DIR, "train")
    test = _e_load_base(TEST_DIR, "test")
    gmean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    train = _e_enrich(train, gmean)
    test = _e_enrich(test, gmean)
    fold_of = _e_make_folds(train)
    train["fold"] = train["well_id"].map(fold_of)
    train_t = train[train["is_target"].astype(bool)].copy()
    test_t = test[test["is_target"].astype(bool)].copy()
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
              "num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 50,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
              "lambda_l2": 1.0, "verbosity": -1, "seed": 42, "num_threads": 4}
    test_geom_delta = np.zeros(len(test_t), dtype=float)
    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy(); tm = ~vm
        model = lgb.LGBMRegressor(**params, n_estimators=1500)
        model.fit(train_t.loc[tm, E026_ALL_FEATURES], y_delta.loc[tm],
                  eval_set=[(train_t.loc[vm, E026_ALL_FEATURES], y_delta.loc[vm])],
                  eval_metric="rmse", callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(model.best_iteration_ or model.n_estimators)
        test_geom_delta += model.predict(test_t[E026_ALL_FEATURES], num_iteration=best) / E026_N_SPLITS
    test_t = test_t.copy()
    test_t["geom_delta"] = test_geom_delta

    payloads = []
    for wid, g in test.groupby("well_id", sort=False):
        if not g["is_target"].any():
            continue
        payloads.append(_e_pf_build_payload(g, _e_load_typewell(TEST_DIR, wid)))
    pf_by_wid = {}
    n_wells = len(payloads)
    if E026_PF_WORKERS > 1 and n_wells > 1:
        try:
            with ProcessPoolExecutor(max_workers=E026_PF_WORKERS) as ex:
                for wid, pred in ex.map(_e_pf_worker, payloads, chunksize=1):
                    pf_by_wid[wid] = pred
        except Exception as ex:
            log(f"PF multiprocess failed ({ex}); falling back to single-thread")
            pf_by_wid = {}
            for p in payloads:
                wid, pred = _e_pf_worker(p)
                pf_by_wid[wid] = pred
    else:
        for p in payloads:
            wid, pred = _e_pf_worker(p)
            pf_by_wid[wid] = pred
    test_t["pf"] = np.nan
    for wid, pred in pf_by_wid.items():
        order = test_t.loc[test_t["well_id"].eq(wid)].sort_values("row_idx").index
        test_t.loc[order, "pf"] = pred
    a = test_t["last_known_TVT"].astype(float).to_numpy()
    test_t["pf_delta"] = test_t["pf"].to_numpy(float) - a

    out = test_t[["id", "pf_delta", "geom_delta"]].copy()
    out["id"] = out["id"].astype(str)
    log(f"exp026 OK: {len(out)} rows  pf_delta[{out.pf_delta.min():.1f},{out.pf_delta.max():.1f}]  "
        f"geom_delta[{out.geom_delta.min():.1f},{out.geom_delta.max():.1f}]")
    return out


# ====================================================================
# =========== BLOCK B: ravaghi feature builder inlined ===============
# (extracted verbatim from pilkwang cell9 build_well/build_dataset;
#  LEAK-FREE: tvt_from_contacts / train-TVT lookup / selector NOT copied)
# ====================================================================
from scipy.interpolate import interp1d  # noqa: F401 (kept for parity)
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
try:
    from numba import njit
except Exception:
    def njit(*a, **k):
        def _d(f): return f
        return _d

R_SEED = 42
R_NCPU = min(4, multiprocessing.cpu_count())
R_FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
R_PLANE_K = 10; R_DENSE_SPW = 60; R_DENSE_K = 20
R_BEAMS = [
    (10, 20.0, 144.0, 2, "cons"), (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"), (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"), (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff")]
R_PF_N = 600; R_ANCC_N = 600
R_PF_MOM = 0.993; R_PF_VN = 0.005; R_PF_PN = 0.01
R_PF_GR_SIG_MIN = 10.; R_PF_GR_SIG_MAX = 60.; R_PF_GR_SIG_DEF = 30.
R_PF_RESAMP = 0.5
R_PF_ROUGH_P = 0.2; R_PF_ROUGH_V = 0.003; R_PF_GR_WIN = 5; R_PF_GR_WT = 0.3
R_ANCC_ALPHA = 0.998; R_ANCC_RN = 0.002; R_ANCC_PN = 0.005
R_ANCC_IS = 0.3; R_ANCC_RP = 0.1; R_ANCC_RR = 0.001


@njit(cache=True)
def _r_interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i] * (1. - t) + grid[i + 1] * t


@njit(cache=True)
def _r_resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N + 1)
    for j in range(N): cum[j + 1] = cum[j] + w[j]
    u0 = np.random.uniform(0., 1. / N)
    np2 = np.empty(N); na = np.empty(N); ci = 0
    for j in range(N):
        u = u0 + j / N
        while ci < N - 1 and cum[ci + 1] < u: ci += 1
        np2[j] = pos[ci] + rp * np.random.randn()
        na[j] = aux[ci] + rv * np.random.randn()
    return np2, na


@njit(cache=True)
def _r_beam_jit(sgr, tw_gr, si, BS, mc, es):
    n = len(sgr); nt = len(tw_gr); MAX = BS * 6
    bidx = np.zeros(BS, np.int64); bidx[0] = si
    bcost = np.full(BS, 1e30); bcost[0] = 0.; bn = np.int64(1)
    hI = np.zeros((n, BS), np.int64); hP = np.zeros((n, BS), np.int64)
    cI = np.zeros(MAX, np.int64); cC = np.full(MAX, 1e30); cP = np.zeros(MAX, np.int64)
    for step in range(n):
        gv = sgr[step]; nc = np.int64(0)
        for bi in range(bn):
            idx = bidx[bi]; cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt: continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es + mc * (d if d >= 0 else -d)
                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni: fnd = ci; break
                if fnd >= 0:
                    if tot < cC[fnd]: cC[fnd] = tot; cP[fnd] = bi
                else:
                    if nc < MAX: cI[nc] = ni; cC[nc] = tot; cP[nc] = bi; nc += 1
        kept = min(BS, nc)
        for i in range(kept):
            mi = i
            for j in range(i + 1, nc):
                if cC[j] < cC[mi]: mi = j
            if mi != i:
                cI[i], cI[mi] = cI[mi], cI[i]
                cC[i], cC[mi] = cC[mi], cC[i]
                cP[i], cP[mi] = cP[mi], cP[i]
        hI[step, :kept] = cI[:kept]; hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]; bcost[:kept] = cC[:kept]; bn = kept
    best = np.int64(0)
    for b in range(1, bn):
        if bcost[b] < bcost[best]: best = b
    path = np.zeros(n, np.int64); b = best
    for s in range(n - 1, -1, -1): path[s] = hI[s, b]; b = hP[s, b]
    return path


@njit(cache=True)
def _r_pf_ancc(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N,
               ALPHA, RN, PN, IS, RP, RR, RESAMP):
    pos = np.empty(N); rate = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ls + IS * np.random.randn()
        rate[j] = ir + 0.01 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.
    for i in range(len(md_v)):
        dm = md_v[i] - pm; dm = max(dm, 1.)
        for j in range(N):
            rate[j] = ALPHA * rate[j] + RN * np.random.randn()
            pos[j] += rate[j] * dm + PN * np.random.randn()
            tvt_j = pos[j] - z_v[i]
            tvt_j = max(tvt_j, vmin - 50.); tvt_j = min(tvt_j, vmin + len(gg) * step + 50.)
            pos[j] = tvt_j + z_v[i]
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                eg = _r_interp1(gg, pos[j] - z_v[i], vmin, step)
                d = (gr_v[i] - eg) / gs
                lk = max(np.exp(-0.5 * d * d) if d * d < 600. else 0., 1e-300)
                w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        ne = 0.
        for j in range(N): ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, rate = _r_resamp(pos, rate, w, N, RP, RR)
            for j in range(N): w[j] = 1. / N
        tv = 0.
        for j in range(N): tv += w[j] * (pos[j] - z_v[i])
        pts[i] = tv; va = 0.
        for j in range(N): va += w[j] * (pos[j] - z_v[i] - tv) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]
    return pts, std_


@njit(cache=True)
def _r_pf_z(md_v, z_v, gr_v, gr_sm_v, gg_p, gg_s, vmin, step,
           gs, ip, iv, beta, icpt, zsig, N,
           MOM, VN, PN, GR_WT, RP, RV, RESAMP):
    pos = np.empty(N); vel = np.empty(N); w = np.ones(N) / N
    for j in range(N):
        pos[j] = ip + 0.5 * np.random.randn()
        vel[j] = iv + 0.02 * np.random.randn()
    pts = np.empty(len(md_v)); std_ = np.empty(len(md_v)); pm = md_v[0] - 1.; pz = z_v[0] - 1.
    for i in range(len(md_v)):
        dm = md_v[i] - pm; dm = max(dm, 1.)
        dzd = (z_v[i] - pz) / dm; ve = beta * dzd + icpt
        for j in range(N):
            vel[j] = MOM * vel[j] + VN * np.random.randn()
            pos[j] += vel[j] * dm + PN * np.random.randn()
            pos[j] = max(pos[j], vmin - 50.); pos[j] = min(pos[j], vmin + len(gg_p) * step + 50.)
        if not np.isnan(gr_v[i]):
            ws = 0.
            for j in range(N):
                ep = _r_interp1(gg_p, pos[j], vmin, step)
                dp = (gr_v[i] - ep) / gs
                lp = max(np.exp(-0.5 * dp * dp) if dp * dp < 600. else 0., 1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es = _r_interp1(gg_s, pos[j], vmin, step)
                    ds = (gr_sm_v[i] - es) / (gs * 1.5)
                    ls = max(np.exp(-0.5 * ds * ds) if ds * ds < 600. else 0., 1e-300)
                    lk = (1. - GR_WT) * lp + GR_WT * ls
                else: lk = lp
                lk = max(lk, 1e-300); w[j] *= lk; ws += w[j]
            if ws > 0.:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1. / N
        ws2 = 0.
        for j in range(N):
            dv = (vel[j] - ve) / max(zsig * 2., 0.005)
            lz = max(np.exp(-0.5 * dv * dv) if dv * dv < 600. else 0., 1e-300)
            w[j] *= lz; ws2 += w[j]
        if ws2 > 0.:
            for j in range(N): w[j] /= ws2
        else:
            for j in range(N): w[j] = 1. / N
        ne = 0.
        for j in range(N): ne += w[j] * w[j]
        if 1. / ne < RESAMP * N:
            pos, vel = _r_resamp(pos, vel, w, N, RP, RV)
            for j in range(N): w[j] = 1. / N
        wm = 0.
        for j in range(N): wm += w[j] * pos[j]
        pts[i] = wm; va = 0.
        for j in range(N): va += w[j] * (pos[j] - wm) ** 2
        std_[i] = va ** 0.5; pm = md_v[i]; pz = z_v[i]
    return pts, std_


def _r_grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def _r_gr_sig(hw, tw_tvt, tw_gr):
    kn = hw[hw['TVT_input'].notna() & hw['GR'].notna()]
    if len(kn) < 20: return float(R_PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values - np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)),
                         R_PF_GR_SIG_MIN, R_PF_GR_SIG_MAX))


def _r_nn(arr, v):
    i = int(np.searchsorted(arr, v, 'left'))
    if i >= len(arr): return len(arr) - 1
    if i > 0 and abs(arr[i - 1] - v) <= abs(arr[i] - v): return i - 1
    return i


def _r_smooth(vals, fb, r):
    s = pd.Series(vals, dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r * 2 + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy(np.float32)


def _r_beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r):
    si = _r_nn(tw_tvt, start_tvt)
    sgr = _r_smooth(gr_h, float(np.nanmean(tw_gr)), r).astype(np.float64)
    path = _r_beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def _r_run_pf_ancc(hw, tw_tvt, tw_gr, N=R_ANCC_N):
    gs = _r_gr_sig(hw, tw_tvt, tw_gr)
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    ls = float(kn['TVT_input'].iloc[-1] + kn['Z'].iloc[-1])
    tail = kn.tail(30); dt = np.diff(tail['TVT_input'].values)
    dz = np.diff(tail['Z'].values); dm = np.diff(tail['MD'].values); m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.
    gg, gmin, gst = _r_grid(tw_tvt, tw_gr)
    pts, std = _r_pf_ancc(ev['MD'].values.astype(np.float64), ev['Z'].values.astype(np.float64),
                          ev['GR'].values.astype(np.float64), gg, gmin, gst,
                          gs, ls, ir, N, R_ANCC_ALPHA, R_ANCC_RN, R_ANCC_PN, R_ANCC_IS, R_ANCC_RP, R_ANCC_RR, R_PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


def _r_run_pf_z(hw, tw_tvt, tw_gr, N=R_PF_N):
    gs = _r_gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(R_PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0: return np.array([]), np.array([])
    dz_k = np.diff(kna['Z'].values); dvt = np.diff(kna['TVT_input'].values)
    dmd_k = np.diff(kna['MD'].values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2] / dmd_k[m2]; vt = dvt[m2] / dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)]); c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt - (c[0] * vz + c[1]))), 0.001)
    else: beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20); dvt2 = np.diff(t2['TVT_input'].values); dmd2 = np.diff(t2['MD'].values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3] / dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _r_grid(tw_tvt, tw_gr)
    gs2, _, _ = _r_grid(tw_tvt, tw_s)
    gr_sm = hw['GR'].rolling(R_PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _r_pf_z(ev['MD'].values.astype(np.float64), ev['Z'].values.astype(np.float64),
                       ev['GR'].values.astype(np.float64),
                       gr_sm.loc[ev.index].values.astype(np.float64),
                       gg, gs2, gmin, gst, gs, float(kna['TVT_input'].iloc[-1]), iv,
                       beta, icpt, zsig, N,
                       R_PF_MOM, R_PF_VN, R_PF_PN, R_PF_GR_WT, R_PF_ROUGH_P, R_PF_ROUGH_V, R_PF_RESAMP)
    return pts.astype(np.float32), std.astype(np.float32)


# numba warmup
_r_md = np.linspace(1, 50, 20, np.float64); _r_z = np.zeros(20, np.float64); _r_gr = np.full(20, 50., np.float64)
_r_gg = np.linspace(45, 55, 100, np.float64)
_r_pf_ancc(_r_md, _r_z, _r_gr, _r_gg, 45., 0.1, 20., 50., 0., 8, 0.998, 0.002, 0.005, 0.3, 0.1, 0.001, 0.5)
_r_pf_z(_r_md, _r_z, _r_gr, _r_gr, _r_gg, _r_gg, 45., 0.1, 20., 50., 0., -1., 0., 0.1, 8, 0.993, 0.005, 0.01, 0.3, 0.2, 0.003, 0.5)
_r_beam_jit(np.random.randn(30), np.random.randn(50), 25, 8, 15., 100.)


def _r_robust_slope(x, y, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-6: return 0.
    return float(np.polyfit(x[m], y[m], 1)[0])


def _r_affine_cal(kgr, tw_at_k, min_pts=20):
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6:
        return 1., float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1); return float(a), float(b)


def _r_seg_b_well(ktvt, kz, form_col):
    bv = ktvt + kz - form_col; n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n)); w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


def _r_multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    out = []
    for hw in hws:
        win = 2 * hw + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().values.astype(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32); M = len(sts)
        if M == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]].astype(np.float32)
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hw, mode='edge')
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]].astype(np.float32)
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win; best = ncc.argmax(1); score = ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best] + hw, 0, nk - 1)].astype(np.float32), score))
    tvts = np.stack([o[0] for o in out], 1); scores = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * scores); sw /= sw.sum(1, keepdims=True) + 1e-9
    sc_ens = (tvts * sw).sum(1).astype(np.float32)
    return out, sc_ens


class _RFormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            p = data_dir / f'{wid}__horizontal_well.csv'
            try: df = pd.read_csv(p, usecols=['X', 'Y'] + R_FORMATIONS).dropna()
            except: continue
            if len(df) == 0: continue
            row = {'wid': wid, 'x': float(df['X'].median()), 'y': float(df['Y'].median())}
            for c in R_FORMATIONS: row[f'{c}_m'] = float(df[c].median())
            rows.append(row)
        self.df = pd.DataFrame(rows); self.wmap = {w: i for i, w in enumerate(self.df['wid'])}
        xy = self.df[['x', 'y']].to_numpy(); self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale)
        self.xa = self.df['x'].to_numpy(); self.ya = self.df['y'].to_numpy()
        self.fa = self.df[[f'{c}_m' for c in R_FORMATIONS]].to_numpy(np.float64)

    def impute(self, xy_q, self_wid=None, k=R_PLANE_K):
        q = xy_q / self.scale; nf = min(k + 5, len(self.df))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid in self.wmap: dist = np.where(idx == self.wmap[self_wid], np.inf, dist)
        ordd = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordd, 1); ik = np.take_along_axis(idx, ordd, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.).astype(np.float64)
        xn = self.xa[ik]; yn = self.ya[ik]; fn = self.fa[ik]; wx = w * xn; wy = w * yn
        A = np.zeros((len(q), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(1); A[:, 0, 1] = (wx * yn).sum(1); A[:, 0, 2] = wx.sum(1)
        A[:, 1, 0] = A[:, 0, 1]; A[:, 1, 1] = (wy * yn).sum(1); A[:, 1, 2] = wy.sum(1)
        A[:, 2, 0] = A[:, 0, 2]; A[:, 2, 1] = A[:, 1, 2]; A[:, 2, 2] = w.sum(1)
        A[:, 0, 0] += 1e-9; A[:, 1, 1] += 1e-9; A[:, 2, 2] += 1e-9
        rhs = np.stack([(wx[:, :, None] * fn).sum(1), (wy[:, :, None] * fn).sum(1), (w[:, :, None] * fn).sum(1)], 1)
        try: coef = np.linalg.solve(A, rhs)
        except:
            coef = np.zeros((len(q), 3, 6))
            for r in range(len(q)):
                try: coef[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except: pass
        Xq = xy_q[:, 0]; Yq = xy_q[:, 1]
        pred = (Xq[:, None] * coef[:, 0, :] + Yq[:, None] * coef[:, 1, :] + coef[:, 2, :]).astype(np.float32)
        pred[~vk.any(1)] = self.fa.mean(0)
        return pred, np.where(vk, dk, np.inf).min(1).astype(np.float32)


class _RDenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=R_DENSE_SPW):
        xs, ys, anccs, wids = [], [], [], []
        for wid in well_ids:
            p = data_dir / f'{wid}__horizontal_well.csv'
            try: df = pd.read_csv(p, usecols=['X', 'Y', 'ANCC']).dropna()
            except: continue
            if len(df) == 0: continue
            ix = np.linspace(0, len(df) - 1, min(spw, len(df)), dtype=int); s = df.iloc[ix]
            xs.append(s['X'].values); ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values); wids.extend([wid] * len(s))
        self.xy = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
        self.ancc = np.concatenate(anccs).astype(np.float32); self.wids = np.array(wids)
        self.scale = np.where(self.xy.std(0) < 1e-3, 1., self.xy.std(0))
        self.tree = cKDTree(self.xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=R_DENSE_K, nfetch=5000):
        xy_q = np.atleast_2d(xy_q); q = xy_q / self.scale; nf = min(nfetch, len(self.ancc))
        dist, idx = self.tree.query(q, k=nf, workers=-1)
        if self_wid: dist = np.where(self.wids[idx] == self_wid, np.inf, dist)
        ordd = np.argpartition(dist, min(k - 1, nf - 1), 1)[:, :k]
        dk = np.take_along_axis(dist, ordd, 1); ik = np.take_along_axis(idx, ordd, 1)
        vk = np.isfinite(dk); w = np.where(vk, 1. / (dk + 1e-3), 0.)
        sw = w.sum(1); safe = np.where(sw < 1e-9, 1., sw); an = self.ancc[ik]
        ap = (an * w).sum(1) / safe; ap = np.where(sw < 1e-9, float(self.ancc.mean()), ap)
        var = ((an - ap[:, None]) ** 2 * w).sum(1) / safe
        return ap.astype(np.float32), np.sqrt(np.maximum(var, 0.)).astype(np.float32), np.where(vk, dk, np.inf).min(1).astype(np.float32)


_R_FI = None; _R_DI = None


def _r_build_imputers(train_dir):
    global _R_FI, _R_DI
    train_dir = Path(train_dir)
    hw_paths = sorted(train_dir.glob('*__horizontal_well.csv'))
    train_wids = [p.stem.replace('__horizontal_well', '') for p in hw_paths]
    _R_FI = _RFormationPlaneKNN(train_wids, train_dir)
    _R_DI = _RDenseANCCImputer(train_wids, train_dir)
    return _R_FI, _R_DI


R_ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
R_BEAM_OFFS = np.array([-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], np.float32)
R_SC_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)
R_PF_OFFS = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)


def _r_build_well(hw_path, tw_path, is_train):
    global _R_FI, _R_DI
    wid = Path(hw_path).stem.replace('__horizontal_well', '')
    try:
        hw = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values('TVT')
    except: return None
    if is_train and 'TVT' not in hw.columns: return None
    kn = hw[hw['TVT_input'].notna()]; ev = hw[hw['TVT_input'].isna()]
    if len(ev) == 0 or len(kn) < 10: return None
    if is_train and hw['TVT'].isna().all(): return None
    tw_tvt = tw['TVT'].to_numpy(np.float32); tw_gr = tw['GR'].to_numpy(np.float32)
    if len(tw_tvt) < 3: return None

    pf_a, std_a = _r_run_pf_ancc(hw, tw_tvt, tw_gr)
    if len(pf_a) == 0: return None
    pf_z, std_z = _r_run_pf_z(hw, tw_tvt, tw_gr)
    pf_use = pf_a.astype(np.float32); std_use = std_a.astype(np.float32)
    has_z = len(pf_z) == len(pf_a) and not np.any(np.isnan(pf_z))

    lk = kn.iloc[-1]; last_tvt = float(lk['TVT_input'])
    gr_full = hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr = gr_full.iloc[:len(kn)].to_numpy(np.float32)

    bpaths = {}
    for (bs, mc, es, r, tag) in R_BEAMS:
        bpaths[tag] = _r_beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
    beam_ref = (bpaths['cons'] + bpaths['sm5']) / 2.

    ktvt = kn['TVT_input'].to_numpy(np.float32)
    sc_res, sc_ens = _r_multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3)
    sc8, sc8s = sc_res[0]; sc15, sc15s = sc_res[1]; sc25, sc25s = sc_res[2]
    sc_cons = (sc8 + sc15 + sc25) / 3.
    sc_trust = float(np.clip(len(kn) / 200., 0., 0.6))
    hyb_ref = (1 - sc_trust) * beam_ref + sc_trust * sc_ens

    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr).astype(np.float32)
    a_cal, b_cal = _r_affine_cal(kgr, tw_at_k)
    kmd = kn['MD'].to_numpy(np.float32); kz = kn['Z'].to_numpy(np.float32)
    pfx_rmse = float(np.sqrt(np.mean((kgr - tw_at_k) ** 2)))
    slp_all = _r_robust_slope(kmd, ktvt); slp_50 = _r_robust_slope(kmd[-50:], ktvt[-50:])
    slp_z = _r_robust_slope(kz, ktvt)

    swid = wid if is_train else None
    xy_ev = ev[['X', 'Y']].to_numpy(np.float64); xy_kn = kn[['X', 'Y']].to_numpy(np.float64)
    form_ev, knn_d = _R_FI.impute(xy_ev, self_wid=swid)
    form_kn, _ = _R_FI.impute(xy_kn, self_wid=swid)
    z_kn = kn['Z'].to_numpy(np.float32); z_ev = ev['Z'].to_numpy(np.float32)

    tvt_fs = {}; form_rmse = {}; form_list = []
    for fi2, fn in enumerate(R_FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = _r_seg_b_well(ktvt, z_kn, form_kn[:, fi2])
        tvt_f = (-z_ev + form_ev[:, fi2] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi2] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi2] + b_late).astype(np.float32)
        tvt_fs[f'tvtF_{fn}'] = tvt_f; tvt_fs[f'tvtFw_{fn}'] = tvt_fw
        tvt_fs[f'tvtF50_{fn}'] = tvt_f50
        tvt_fs[f'bw_{fn}'] = np.float32(b_full); tvt_fs[f'bww_{fn}'] = np.float32(b_wls)
        tvt_fs[f'bw50_{fn}'] = np.float32(b_late)
        tvt_fs[f'bw_early_{fn}'] = np.float32(b_early)
        tvt_fs[f'bw_mid_{fn}'] = np.float32(b_mid)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi2] + b_full)) ** 2)))
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    form_mean_d = (fs.mean(1) - last_tvt).astype(np.float32)
    form_std_d = fs.std(1).astype(np.float32)
    form_rng_d = (fs.max(1) - fs.min(1)).astype(np.float32)

    d_ancc, d_std, d_dist = _R_DI.impute(xy_ev, self_wid=swid)
    d_kn, d_std_kn, _ = _R_DI.impute(xy_kn, self_wid=swid)
    b_vd = ktvt + z_kn - d_kn
    _, b_de, b_dm, b_dl, b_dw = _r_seg_b_well(ktvt, z_kn, d_kn)
    b_d = float(np.median(b_vd))
    tvt_dense = (-z_ev + d_ancc + b_d).astype(np.float32)
    tvt_densew = (-z_ev + d_ancc + b_dw).astype(np.float32)
    tvt_dense50 = (-z_ev + d_ancc + b_dl).astype(np.float32)
    res_kn = ktvt + z_kn - d_kn
    d_rmse = float(np.sqrt(np.mean(res_kn ** 2))); d_bias = float(np.mean(res_kn)); d_nb_std = float(np.mean(d_std_kn))

    all_sigs = [pf_use] + [p for p in bpaths.values()] + [sc8, sc15, sc25, sc_ens, tvt_fs['tvtF_ANCC'], tvt_dense]
    sig_mat = np.stack(all_sigs, 1)
    sig_std = sig_mat.std(1).astype(np.float32)
    sig_mean = (sig_mat.mean(1) - last_tvt).astype(np.float32)

    gr_s = pd.Series(gr_full.values); rolls = {}
    for w in [5, 21, 51, 101]:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f'grm{w}'] = r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}'] = r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1, 5, 15, 30]:
        rolls[f'glag{lag}'] = gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}'] = gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1 = gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2 = gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env = gr_s.rolling(21, center=True, min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg = np.sqrt(np.maximum((gr_s ** 2).rolling(21, center=True, min_periods=1).mean(), 0.)
                     ).iloc[ev.index].values.astype(np.float32)

    hmd = ev['MD'].to_numpy(np.float32); md_since = hmd - float(lk['MD'])
    slp_b_all = (last_tvt + slp_all * md_since).astype(np.float32)
    slp_b_50 = (last_tvt + slp_50 * md_since).astype(np.float32)

    mdd = hw['MD'].diff().replace(0, np.nan)
    dzdmd = (hw['Z'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd = (hw['X'].diff() / mdd).iloc[ev.index].values.astype(np.float32)
    dydmd = (hw['Y'].diff() / mdd).iloc[ev.index].values.astype(np.float32)

    nh = len(ev); frac = (np.arange(nh) / max(nh - 1, 1)).astype(np.float32)
    def sc(v): return np.full(nh, np.float32(v), np.float32)

    feats = {
        'well': wid, 'id': [f'{wid}_{i}' for i in ev.index],
        'last_known_tvt': sc(last_tvt),
        'pf_ancc': pf_use, 'pf_ancc_std': std_use,
        'pf_ancc_delta': (pf_use - last_tvt).astype(np.float32),
        'pf_z': (pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta': ((pf_z - last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z': ((pf_use - pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f'beam_{t}_d': (p - np.float32(last_tvt)).astype(np.float32) for t, p in bpaths.items()},
        'beam_mean_d': np.stack([(p - last_tvt) for p in bpaths.values()], 1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p - last_tvt) for p in bpaths.values()], 1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p - last_tvt) for p in bpaths.values()], 1), 1).astype(np.float32),
        'sc8_d': (sc8 - np.float32(last_tvt)).astype(np.float32), 'sc8_sc': sc8s,
        'sc15_d': (sc15 - np.float32(last_tvt)).astype(np.float32), 'sc15_sc': sc15s,
        'sc25_d': (sc25 - np.float32(last_tvt)).astype(np.float32), 'sc25_sc': sc25s,
        'sc_cons_d': (sc_cons - np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d': (sc_ens - np.float32(last_tvt)).astype(np.float32),
        'sc_trust': sc(sc_trust), 'hyb_d': (hyb_ref - np.float32(last_tvt)).astype(np.float32),
        'sig_std': sig_std, 'sig_mean_d': sig_mean,
        **tvt_fs,
        **{f'frm_rmse_{fn}': sc(form_rmse[fn]) for fn in R_FORMATIONS},
        'form_mean_d': form_mean_d, 'form_std_d': form_std_d, 'form_rng_d': form_rng_d,
        'spatial_ancc_d': (form_ev[:, 0] - np.float32(np.interp(last_tvt, tw_tvt, tw_gr))),
        'spatial_knn_dist': knn_d,
        'dense_ancc': d_ancc, 'dense_std': d_std, 'dense_dist': d_dist,
        'tvt_dense_d': (tvt_dense - last_tvt).astype(np.float32),
        'tvt_densew_d': (tvt_densew - last_tvt).astype(np.float32),
        'tvt_dense50_d': (tvt_dense50 - last_tvt).astype(np.float32),
        'dense_rmse': sc(d_rmse), 'dense_bias': sc(d_bias), 'dense_nb_std': sc(d_nb_std),
        'pf_vs_spatial': (pf_use - tvt_fs['tvtF_ANCC']).astype(np.float32),
        'pf_vs_dense': (pf_use - tvt_dense).astype(np.float32),
        'spatial_vs_dense': (tvt_fs['tvtF_ANCC'] - tvt_dense).astype(np.float32),
        'beam_vs_spatial': (bpaths['cons'] - tvt_fs['tvtF_ANCC']).astype(np.float32),
        'sc_vs_beam': (sc_ens - bpaths['cons']).astype(np.float32),
        'cal_a': sc(a_cal), 'cal_b': sc(b_cal),
        'pfx_rmse': sc(pfx_rmse), 'known_len': sc(len(kn)), 'eval_len': sc(nh),
        'slp_all': sc(slp_all), 'slp_50': sc(slp_50), 'slp_z': sc(slp_z),
        'slp_b_d_all': (slp_b_all - last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50 - last_tvt).astype(np.float32),
        'ktvt_range': sc(float(np.ptp(ktvt))), 'ktvt_std': sc(float(ktvt.std())),
        'md_since': md_since, 'frac': frac, 'frac2': frac ** 2, 'sqrt_frac': np.sqrt(frac),
        'z': z_ev,
        'dx': (ev['X'] - float(lk['X'])).to_numpy(np.float32),
        'dy': (ev['Y'] - float(lk['Y'])).to_numpy(np.float32),
        'dz': (z_ev - float(lk['Z'])).astype(np.float32),
        'dxy': np.sqrt((ev['X'] - float(lk['X'])) ** 2 + (ev['Y'] - float(lk['Y'])) ** 2).to_numpy(np.float32),
        'dzdmd': dzdmd, 'dxdmd': dxdmd, 'dydmd': dydmd,
        'gr': hgr, 'gr_d1': gr_d1, 'gr_d2': gr_d2, 'gr_env': gr_env, 'gr_nrg': gr_nrg,
        'gr_vs_tw_anc': hgr - np.float32(np.interp(last_tvt, tw_tvt, tw_gr)),
        'gr_vs_slp_all': hgr - np.interp(slp_b_all, tw_tvt, tw_gr).astype(np.float32),
        **{f'tda{int(o)}': hgr - np.float32(np.interp(last_tvt + o, tw_tvt, tw_gr)) for o in R_ANCH_OFFS},
        **{f'tdbc{int(o)}': hgr - np.interp(beam_ref + o, tw_tvt, tw_gr).astype(np.float32) for o in R_BEAM_OFFS},
        **{f'tdsc{int(o)}': hgr - np.interp(sc_ens + o, tw_tvt, tw_gr).astype(np.float32) for o in R_SC_OFFS},
        **{f'tdpf{int(o)}': hgr - np.interp(pf_use + o, tw_tvt, tw_gr).astype(np.float32) for o in R_PF_OFFS},
        'tw_range': sc(float(np.ptp(tw_tvt))), 'tw_gr_mean': sc(float(tw_gr.mean())),
    }
    for k, v in rolls.items(): feats[k] = v
    result = pd.DataFrame(feats)
    if is_train:
        if 'TVT' not in ev.columns or ev['TVT'].isna().all(): return None
        result['target'] = (ev['TVT'].to_numpy(np.float32) - np.float32(last_tvt))
    return result


def _r_build_dataset(paths, is_train, label):
    args = [(str(p), str(p.parent / f'{p.stem.replace("__horizontal_well", "")}__typewell.csv'), is_train)
            for p in paths
            if (p.parent / f'{p.stem.replace("__horizontal_well", "")}__typewell.csv').exists()]
    res = Parallel(n_jobs=R_NCPU, prefer='threads', verbose=0)(
        delayed(_r_build_well)(hp, tp, it) for hp, tp, it in args)
    parts = [r for r in res if r is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _ensure_koolbox():
    """ravaghi Trainer pickles   koolbox   import          
          pip      Kaggle(internet   )   offline wheel dataset
    (phongnguyn23021656/koolbox-offline)    pip install / sys.path      """
    try:
        import koolbox  # noqa: F401
        return
    except Exception:
        pass
    if not ON_KAGGLE:
        return  # local: let the real ImportError surface later
    import subprocess, glob as _glob
    roots = [EXT / "koolbox-offline", EXT / "datasets/phongnguyn23021656/koolbox-offline"]
    root = next((p for p in roots if p.exists()), None)
    if root is None:
        log("WARN koolbox-offline dataset not found; ravaghi unpickle may fail")
        return
    whls = sorted(_glob.glob(str(root / "**" / "koolbox*.whl"), recursive=True))
    if whls:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                            "--find-links", str(root), whls[0]], check=True)
            log(f"koolbox installed offline from {whls[0]}")
            return
        except Exception as ex:
            log(f"WARN koolbox pip install failed: {ex}; falling back to sys.path")
    # fallback: add a directory that contains a `koolbox/` package to sys.path
    for cand in [root, *[Path(p).parent for p in _glob.glob(str(root / "**" / "koolbox" / "__init__.py"), recursive=True)]]:
        sys.path.insert(0, str(Path(cand)))
    log(f"koolbox sys.path fallback from {root}")


def compute_ravaghi():
    """Returns DataFrame[id, rav_lgb3, rav_cb1, rav_cb2] (delta space)."""
    import joblib
    _ensure_koolbox()
    _r_build_imputers(TRAIN_DIR)
    test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
    df = _r_build_dataset(test_paths, is_train=False, label="test")
    df["id"] = df["id"].astype(str)

    rav_root = next((p for p in RAV_ROOTS if p.exists()), None)
    if rav_root is None:
        raise FileNotFoundError(f"ravaghi artifacts not found in {RAV_ROOTS}")
    rav_cols = [c for c in pd.read_csv(rav_root / "data" / "train.csv", nrows=1).columns
                if c not in {"well", "id", "target"}]
    X = df[rav_cols]

    def _pred(model_dir):
        pk = sorted((rav_root / "models" / model_dir).glob("*.pkl"))
        tr = joblib.load(pk[0])
        return np.asarray(tr.predict(X), dtype=float)

    out = pd.DataFrame({"id": df["id"].to_numpy()})
    out["rav_lgb3"] = _pred("lightgbm-3")
    out["rav_cb1"] = _pred("catboost-1")
    out["rav_cb2"] = _pred("catboost-2")
    for c in ["rav_lgb3", "rav_cb1", "rav_cb2"]:
        log(f"ravaghi {c} delta[{out[c].min():.1f},{out[c].max():.1f}] mean={out[c].mean():.2f}")
    return out


# ====================================================================
# ====== BLOCK C: pilkwang model-package CatBoost (pilk_cat) =========
# (pilk_tcn      : hidden-only frame         TCN load/       )
# ====================================================================
def compute_pilkwang():
    """Returns DataFrame[id, pilk_cat] (delta space). CatBoost only."""
    import importlib.util
    pkg = next((p for p in PKG_ROOTS if (p / "metadata" / "model_package_manifest.json").exists()), None)
    if pkg is None:
        raise FileNotFoundError(f"rogii-model-package not found in {PKG_ROOTS}")
    man = json.load(open(pkg / "metadata" / "model_package_manifest.json"))
    fc = json.load(open(pkg / "feature_builders" / "feature_columns.json"))

    os.environ["ROGII_DATA_DIR"] = str(DATA)
    sys.path.insert(0, str(pkg)); sys.path.insert(0, str(pkg / "feature_builders"))
    sys.modules.pop("rogii_sidecar_feature_builder", None)
    spec = importlib.util.spec_from_file_location(
        "rogii_sidecar_feature_builder", str(pkg / "feature_builders" / "build_features.py"))
    fb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fb
    spec.loader.exec_module(fb)
    sample = pd.read_csv(SAMPLE_PATH)[["id"]]
    ff = fb.build_features(data_dir=DATA, sample=sample)
    ff["id"] = ff["id"].astype(str)

    def cols_for(entry):
        if isinstance(entry.get("feature_columns"), list):
            return entry["feature_columns"]
        fs = entry.get("feature_set")
        if isinstance(fc, dict):
            if fs and isinstance(fc.get(fs), list):
                return fc[fs]
            if isinstance(fc.get("columns"), list):
                return fc["columns"]
        return fc
    ents = {e["model_name"]: e for e in man["models"]}

    out = pd.DataFrame({"id": ff["id"].to_numpy()})
    from catboost import CatBoostRegressor
    e = ents["catboost_alltrain"]; cols = cols_for(e)
    cm = CatBoostRegressor(); cm.load_model(str(pkg / e["path"]))
    pc = np.asarray(cm.predict(ff[cols].replace([np.inf, -np.inf], np.nan)), dtype=float)
    out["pilk_cat"] = pc
    log(f"pilk_cat delta[{pc.min():.1f},{pc.max():.1f}] mean={pc.mean():.2f} std={pc.std():.2f}")
    return out


# ====================================================================
# ==================== BLOCK D: blend + postprocess ==================
# ====================================================================
def _robfit(s, yv, deg, iters):
    s = np.asarray(s, float); yv = np.asarray(yv, float)
    if len(s) < deg + 2 or np.std(s) < 1e-9:
        return yv.copy()
    try:
        c = np.polyfit(s, yv, deg)
        for _ in range(iters):
            r = yv - np.polyval(c, s)
            scv = np.median(np.abs(r)) * 1.4826 + 1e-6
            w = 1.0 / (1.0 + (r / (2.0 * scv)) ** 2)
            c = np.polyfit(s, yv, deg, w=w)
        return np.polyval(c, s)
    except Exception:
        return yv.copy()


_ANCHOR = {}; _MD = {}; _LASTMD = {}
def _build_anchor_maps():
    if _ANCHOR:
        return
    for p in sorted(TEST_DIR.glob("*__horizontal_well.csv")):
        wid = p.stem.replace("__horizontal_well", "")
        hw = pd.read_csv(p)
        kn = hw[hw["TVT_input"].notna()]
        if len(kn) == 0:
            continue
        last = kn.iloc[-1]
        last_tvt = float(last["TVT_input"]); last_md = float(last["MD"])
        ev = hw[hw["TVT_input"].isna()]
        for ridx in ev.index:
            _id = f"{wid}_{ridx}"
            _ANCHOR[_id] = last_tvt
            _MD[_id] = float(hw["MD"].iloc[ridx])
            _LASTMD[_id] = last_md


def main():
    log(f"ON_KAGGLE={ON_KAGGLE}  DATA={DATA}")
    sample = pd.read_csv(SAMPLE_PATH)
    sample["id"] = sample["id"].astype(str)

    t0 = time.time(); exp026 = compute_exp026(); log(f"exp026 {time.time()-t0:.0f}s")
    t0 = time.time(); rav = compute_ravaghi(); log(f"ravaghi {time.time()-t0:.0f}s")
    t0 = time.time(); pilk = compute_pilkwang(); log(f"pilkwang {time.time()-t0:.0f}s")

    df = sample[["id"]].merge(exp026, on="id", how="left") \
        .merge(rav, on="id", how="left").merge(pilk, on="id", how="left")
    try:
        df.to_parquet(WORK / "_components_submit.parquet")
    except Exception:
        df.to_csv(WORK / "_components_submit.csv", index=False)

    comp_map = {"pf": "pf_delta", "geom": "geom_delta", "rav_lgb3": "rav_lgb3",
                "rav_cb1": "rav_cb1", "rav_cb2": "rav_cb2", "pilk_cat": "pilk_cat"}
    pf_fallback = df["pf_delta"].fillna(0.0).to_numpy(float)
    final_delta = np.zeros(len(df), dtype=float)
    for name, w in W.items():
        col = comp_map[name]
        v = df[col].to_numpy(float)
        nmiss = int(np.isnan(v).sum())
        if nmiss:
            log(f"WARN component {name} has {nmiss} NaN -> filled with pf_delta fallback")
            v = np.where(np.isnan(v), pf_fallback, v)
        final_delta += w * v

    _build_anchor_maps()
    anchor_tvt = df["id"].map(_ANCHOR).to_numpy(float)
    md = df["id"].map(_MD).to_numpy(float)
    last_md = df["id"].map(_LASTMD).to_numpy(float)

    # 1. warmup decay applied to delta
    decay = 1.0 - np.exp(-np.maximum(md - last_md, 0.0) / TAU)
    decay = np.where(np.isfinite(decay), decay, 1.0)
    final_delta = final_delta * decay

    # 2. tvt
    df["tvt"] = anchor_tvt + final_delta
    df["well"] = df["id"].str.rsplit("_", n=1).str[0]
    df["row_idx"] = df["id"].str.rsplit("_", n=1).str[1].astype(int)
    final = dict(zip(df["id"].values, df["tvt"].astype(float).values))

    hw_cache = {}
    def load_hw(wid):
        if wid not in hw_cache:
            hw_cache[wid] = pd.read_csv(TEST_DIR / f"{wid}__horizontal_well.csv")
        return hw_cache[wid]

    n_smooth = 0; n_proj = 0
    for wid, g in df.groupby("well", sort=False):
        gi = g.sort_values("row_idx")
        ids = gi["id"].values
        v = gi["tvt"].to_numpy(float)
        # 3. rolling mean window=101
        v = pd.Series(v).rolling(SMOOTH_W, min_periods=1, center=True).mean().to_numpy()
        n_smooth += 1
        # 4. robust projection (deg=4, beta=0.75)
        try:
            hw = load_hw(wid)
            kn = hw[hw["TVT_input"].notna()]
            if len(kn) >= PROJ_MIN_KNOWN:
                last = kn.iloc[-1]
                anchor = float(last["TVT_input"]) + float(last["Z"])
                ps = float(last["MD"]); end = float(hw["MD"].iloc[-1])
                ri = gi["row_idx"].values
                Zv = hw["Z"].values[ri].astype(float)
                mdv = hw["MD"].values[ri].astype(float)
                s = (mdv - ps) / max(end - ps, 1e-6)
                U = (v + Zv) - anchor
                fit = _robfit(s, U, PROJ_DEG, PROJ_ITERS)
                newU = PROJ_BETA * fit + (1.0 - PROJ_BETA) * U
                v = newU + anchor - Zv
                n_proj += 1
        except Exception as ex:
            log(f"proj skip {wid}: {ex}")
        for _id, _v in zip(ids, v):
            final[_id] = float(_v)
    log(f"smoothed {n_smooth} wells, projected {n_proj} wells")

    out = sample[["id"]].copy()
    out["tvt"] = out["id"].map(final)
    if out["tvt"].isna().any():
        gmean = float(np.nanmean(out["tvt"].to_numpy(float)))
        nbad = int(out["tvt"].isna().sum())
        out["tvt"] = out["tvt"].fillna(gmean)
        log(f"WARN filled {nbad} NaN ids with global mean {gmean:.1f}")

    assert list(out.columns) == ["id", "tvt"], out.columns.tolist()
    assert len(out) == len(sample), (len(out), len(sample))
    assert set(out["id"]) == set(sample["id"]), "id set mismatch"
    assert np.isfinite(out["tvt"].to_numpy(float)).all(), "non-finite tvt"
    out.to_csv(OUT_PATH, index=False)
    log(f"WROTE {OUT_PATH} rows={len(out)} tvt[{out.tvt.min():.1f},{out.tvt.max():.1f}] mean={out.tvt.mean():.1f}")
    (WORK / "submit_build_log2.txt").write_text("\n".join(_log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
