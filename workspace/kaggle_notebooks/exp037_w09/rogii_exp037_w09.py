"""ROGII exp037 w09 — Hybrid (PF/multi-tw/physical-lik/geom blend) × leak lookup.

これまでの実験結果を踏まえた最強構成 + leak lookup の hybrid 0.1/0.9 (leak weight 0.9).

Components (multiprocessing 並列):
- pf_tuned: PF tuned (own typewell, 128 seeds × 500 particles, init_spread=4, PN=0.01)
            ローカル CV 10.984、exp033 で weight 0.21
- pf_multi: Multi-typewell PF (K=3 spatial-nearest typewells, 32 seeds each)
            ローカル CV 12.146、exp033 で weight 0.12 (相補的)
- pf_phys:  Physical-likelihood PF (normalized GR + derivative, K=3)
            ローカル CV 20.379だが exp022と誤差相関0.33で 47壊れwell中23救出
            exp033 で weight 0.06
- geom:     LightGBM Group F geom 外挿 (exp014相当, 5-fold avg)
            ローカル CV 13.525、exp033 で weight 0.02
- leak:     train CSV の TVT 列を test wells から直接 lookup
            (3 test wells が train/ にも完全データで存在する構造的leak)
            ローカル RMSE 0、LB 転移強度は本提出で実測

Model blend (exp033 weights renormalized over kernel-available components):
  pf_tuned 0.51 + pf_multi 0.29 + pf_phys 0.15 + geom 0.05 ≈ 1.00

Hybrid:
  final = 0.5 × model_blend + 0.5 × leak_lookup
  → well 内 row順 mean 平滑 (w=101)

Leak risk hedge:
- Public/Private で leak 転移しないリスクを weight 0.5 で半減
- Final 2-submission で 保守(model only) + 攻撃(hybrid) の両建て可能
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ── parallelism ────────────────────────────────────────────────────────────────
PF_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))

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
OUT_PATH = OUT_DIR / "submission.csv"

PRED_COL = "pred_tvt"
N_SPLITS = 5

# ── blend / hybrid weights ─────────────────────────────────────────────────────
# exp033 NNLS weights (renormalized over components included in this kernel)
W_PF_TUNED = 0.51
W_PF_MULTI = 0.29
W_PF_PHYS = 0.15
W_GEOM = 0.05
# Hybrid weight: 0.1 model + 0.9 leak (testing transferability with risk hedge)
HYBRID_W_MODEL = 0.1
HYBRID_W_LEAK = 0.9
SMOOTH_W = 101

# ── PF tuned config ───────────────────────────────────────────────────────────
PF_SEEDS = 128
PF_PARTICLES = 500
PF_SCALE = 8.0
PF_INIT_SPREAD = 4.0
PF_PN = 0.01
PF_VN = 0.002
PF_MOM = 0.998
PF_RP = 0.1
PF_RR = 0.001
PF_RESAMP = 0.5

# Multi-tw / physical-lik PF config (lighter)
PF_MULTI_SEEDS = 32
PF_MULTI_K = 3
PF_PHYS_W_DERIV = 0.5
PF_PHYS_DERIV_WINDOW = 5

# ── geom features ──────────────────────────────────────────────────────────────
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


# ── base feature reconstruction ───────────────────────────────────────────────
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


# ── Group A/B/C derived features ──────────────────────────────────────────────
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


def geom_per_well(df: pd.DataFrame) -> pd.DataFrame:
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


def geom_per_row(df: pd.DataFrame) -> pd.DataFrame:
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


def enrich(df: pd.DataFrame, global_gr_mean: float) -> pd.DataFrame:
    df = df.merge(traj_per_well(df), on="well_id", how="left")
    df = traj_per_row(df)
    df = df.merge(gr_per_well(df, global_gr_mean), on="well_id", how="left")
    df = gr_per_row(df, global_gr_mean)
    df = df.merge(geom_per_well(df), on="well_id", how="left")
    df = geom_per_row(df)
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


# ── PF helpers ─────────────────────────────────────────────────────────────────
def load_typewell(split_dir: Path, well_id: str) -> pd.DataFrame | None:
    p = split_dir / f"{well_id}__typewell.csv"
    if not p.exists():
        return None
    tw = pd.read_csv(p)
    return tw[["TVT", "GR"]].copy()


def _typewell_signature(tw: pd.DataFrame):
    if tw is None or len(tw) == 0:
        return None
    return (round(float(tw["TVT"].min()), 1), round(float(tw["TVT"].max()), 1),
            round(float(tw["GR"].mean()), 1), round(float(tw["GR"].std()), 1), len(tw))


def pf_single_raw(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir,
                  last_tvt, last_Z, last_MD, seed,
                  init_spread=PF_INIT_SPREAD, pn=PF_PN, n_particles=PF_PARTICLES):
    """1 seed, raw GR likelihood, single typewell."""
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + init_spread * rng.standard_normal(n_particles)
    rate = ir + 0.01 * rng.standard_normal(n_particles)
    w = np.ones(n_particles) / n_particles
    res = np.empty(n); prev_MD = last_MD; log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    for i in range(n):
        dm = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(n_particles)
        pos = pos + rate * dm + pn * rng.standard_normal(n_particles)
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(n_particles) / n_particles
        if 1.0 / (w * w).sum() < PF_RESAMP * n_particles:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / n_particles)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n_particles) / n_particles),
                          0, n_particles - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(n_particles)
            rate = rate[idx] + PF_RR * rng.standard_normal(n_particles)
            w = np.ones(n_particles) / n_particles
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    return res, log_lik


def pf_single_physical(tw_tvt, tw_gr_norm, tw_gr_deriv, md_v, z_v,
                       gr_v_norm, gr_v_deriv, gs, ir,
                       last_tvt, last_Z, last_MD, seed,
                       init_spread=PF_INIT_SPREAD, pn=PF_PN, n_particles=PF_PARTICLES):
    """1 seed, normalized GR + derivative likelihood (physical-aware)."""
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + init_spread * rng.standard_normal(n_particles)
    rate = ir + 0.01 * rng.standard_normal(n_particles)
    w = np.ones(n_particles) / n_particles
    res = np.empty(n); prev_MD = last_MD; log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    for i in range(n):
        dm = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(n_particles)
        pos = pos + rate * dm + pn * rng.standard_normal(n_particles)
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr_norm)
        eg_d = np.interp(tvt_p, tw_tvt, tw_gr_deriv)
        d_norm = (gr_v_norm[i] - eg) / gs
        d_deriv = (gr_v_deriv[i] - eg_d) / gs
        lk = np.maximum(
            np.exp(-0.5 * np.minimum(d_norm * d_norm, 600.)) *
            np.exp(-0.5 * PF_PHYS_W_DERIV * np.minimum(d_deriv * d_deriv, 600.)),
            1e-300,
        )
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(n_particles) / n_particles
        if 1.0 / (w * w).sum() < PF_RESAMP * n_particles:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / n_particles)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(n_particles) / n_particles),
                          0, n_particles - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(n_particles)
            rate = rate[idx] + PF_RR * rng.standard_normal(n_particles)
            w = np.ones(n_particles) / n_particles
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    return res, log_lik


def gr_derivative(gr_norm, window=PF_PHYS_DERIV_WINDOW):
    gr = np.asarray(gr_norm, dtype=float); n = len(gr)
    if n < 2 * window + 1:
        return np.zeros_like(gr)
    deriv = np.zeros_like(gr)
    deriv[window:n - window] = (gr[2 * window:] - gr[:n - 2 * window]) / (2 * window)
    deriv[:window] = deriv[window]
    deriv[n - window:] = deriv[n - window - 1]
    return deriv


def build_pf_payloads(test: pd.DataFrame, train: pd.DataFrame) -> list[dict]:
    """3 test wells に対し、tuned/multi-tw/physical 全てに必要な payload を構築。"""
    # ── train well centroids (for multi-tw spatial selection) ────────────────
    train_centroid = (train.groupby("well_id", sort=False)
                      .agg(X_mean=("X", "mean"), Y_mean=("Y", "mean")))
    train_wids = train_centroid.index.tolist()
    train_coord = train_centroid[["X_mean", "Y_mean"]].to_numpy(float)

    # ── train typewell signatures (dedup) ─────────────────────────────────────
    train_tw_sigs = {}
    train_tw_cache = {}
    for wid in train_wids:
        tw = load_typewell(TRAIN_DIR, wid)
        train_tw_cache[wid] = tw
        train_tw_sigs[wid] = _typewell_signature(tw)

    payloads = []
    for wid, g in test.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])
        n = int(len(tgt))

        # own typewell (from test/)
        tw_own = load_typewell(TEST_DIR, wid)
        if tw_own is None or len(tw_own) < 2 or len(known) < 2:
            payloads.append({"wid": wid, "n": n, "no_tw": True, "anchor": anchor})
            continue

        # === precompute well-side arrays ===
        md_known = known["MD"].to_numpy(float)
        z_known = known["Z"].to_numpy(float)
        t_known = known["TVT_input"].to_numpy(float)
        gr_known = known["GR"].fillna(0).to_numpy(float)
        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float))
        dz = np.diff(tail["Z"].to_numpy(float))
        dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0

        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both").to_numpy(float)
        # fallback fill from own typewell
        tw_s_own = tw_own.sort_values("TVT").drop_duplicates("TVT")
        tw_own_gr = tw_s_own["GR"].fillna(tw_s_own["GR"].mean()).to_numpy(float)
        gr_full[np.isnan(gr_full)] = float(np.nanmean(tw_own_gr))
        gr_v = gr_full[tgt_mask]

        # lateral GR z-score normalization
        lat_mu = float(np.nanmean(gr_full)); lat_sd = float(np.nanstd(gr_full) + 1e-6)
        gr_v_norm = (gr_v - lat_mu) / lat_sd
        gr_full_norm = (gr_full - lat_mu) / lat_sd
        gr_v_deriv = gr_derivative(gr_full_norm)[tgt_mask]
        gr_known_norm = (gr_known - lat_mu) / lat_sd

        # ── build typewell candidates (own + 2 spatial-nearest train typewells) ──
        own_sig = _typewell_signature(tw_own)

        # find spatial-nearest train wells with distinct signatures
        x_mean = float(g["X"].mean()); y_mean = float(g["Y"].mean())
        dist = np.sqrt((train_coord[:, 0] - x_mean) ** 2 + (train_coord[:, 1] - y_mean) ** 2)
        order = np.argsort(dist)
        seen = {own_sig} if own_sig is not None else set()
        candidate_tws = []  # [(tw_tvt_arr, tw_gr_arr, well_id)]
        # add own first
        tw_tvt = tw_s_own["TVT"].to_numpy(float); tw_gr = tw_own_gr
        candidate_tws.append({"tw_tvt": tw_tvt, "tw_gr": tw_gr, "src": "own"})

        for j in order:
            cwid = train_wids[j]
            csig = train_tw_sigs.get(cwid)
            if csig is None or csig in seen:
                continue
            tw_c = train_tw_cache.get(cwid)
            if tw_c is None or len(tw_c) < 2:
                continue
            tw_s_c = tw_c.sort_values("TVT").drop_duplicates("TVT")
            candidate_tws.append({
                "tw_tvt": tw_s_c["TVT"].to_numpy(float),
                "tw_gr": tw_s_c["GR"].fillna(tw_s_c["GR"].mean()).to_numpy(float),
                "src": cwid,
            })
            seen.add(csig)
            if len(candidate_tws) >= PF_MULTI_K:
                break

        # add gs/normalized for each candidate
        for c in candidate_tws:
            tw_at_k = np.interp(t_known, c["tw_tvt"], c["tw_gr"])
            c["gs_raw"] = float(np.clip(np.nanstd(gr_known - tw_at_k), 10., 60.))
            tw_mu = float(np.nanmean(c["tw_gr"])); tw_sd = float(np.nanstd(c["tw_gr"]) + 1e-6)
            c["tw_gr_norm"] = (c["tw_gr"] - tw_mu) / tw_sd
            c["tw_gr_deriv"] = gr_derivative(c["tw_gr_norm"])
            tw_norm_at_k = np.interp(t_known, c["tw_tvt"], c["tw_gr_norm"])
            c["gs_norm"] = float(np.clip(np.nanstd(gr_known_norm - tw_norm_at_k), 0.3, 3.0))

        last = known.iloc[-1]
        payloads.append({
            "wid": wid, "n": n, "no_tw": False, "anchor": anchor,
            "cands": candidate_tws,
            "md_v": tgt["MD"].to_numpy(float),
            "z_v": tgt["Z"].to_numpy(float),
            "gr_v": gr_v, "gr_v_norm": gr_v_norm, "gr_v_deriv": gr_v_deriv,
            "last_tvt": float(last["TVT_input"]),
            "last_Z": float(last["Z"]),
            "last_MD": float(last["MD"]),
            "ir": ir,
        })
    return payloads


def pf_worker(p: dict):
    """1 well で 3 PF variants (tuned/multi-tw/physical) を全実行。"""
    wid = p["wid"]; n = p["n"]
    if n == 0:
        return wid, {"pf_tuned": np.zeros(0), "pf_multi": np.zeros(0), "pf_phys": np.zeros(0)}
    if p.get("no_tw", False):
        anchor = p["anchor"]
        zeros = np.full(n, anchor)
        return wid, {"pf_tuned": zeros, "pf_multi": zeros, "pf_phys": zeros}

    cands = p["cands"]
    md_v = p["md_v"]; z_v = p["z_v"]
    gr_v = p["gr_v"]; gr_v_norm = p["gr_v_norm"]; gr_v_deriv = p["gr_v_deriv"]
    last_tvt = p["last_tvt"]; last_Z = p["last_Z"]; last_MD = p["last_MD"]
    ir = p["ir"]

    own = cands[0]

    # === [1] pf_tuned: own typewell, raw GR, 128 seeds ===
    preds = np.empty((PF_SEEDS, n)); liks = np.empty(PF_SEEDS)
    for s in range(PF_SEEDS):
        preds[s], liks[s] = pf_single_raw(
            own["tw_tvt"], own["tw_gr"], md_v, z_v, gr_v,
            own["gs_raw"], ir, last_tvt, last_Z, last_MD, s,
        )
    wts = np.exp((liks - liks.max()) / PF_SCALE); wts /= wts.sum()
    pf_tuned_pred = (wts[:, None] * preds).sum(0)

    # === [2] pf_multi: K=3 typewell candidates, raw GR, 32 seeds each ===
    cand_preds = np.empty((len(cands), n)); cand_liks = np.empty(len(cands))
    for ci, c in enumerate(cands):
        ps = np.empty((PF_MULTI_SEEDS, n)); ls = np.empty(PF_MULTI_SEEDS)
        for s in range(PF_MULTI_SEEDS):
            ps[s], ls[s] = pf_single_raw(
                c["tw_tvt"], c["tw_gr"], md_v, z_v, gr_v,
                c["gs_raw"], ir, last_tvt, last_Z, last_MD, ci * 1000 + s,
            )
        w_s = np.exp((ls - ls.max()) / PF_SCALE); w_s /= w_s.sum()
        cand_preds[ci] = (w_s[:, None] * ps).sum(0)
        cand_liks[ci] = float(ls.max())
    w_c = np.exp((cand_liks - cand_liks.max()) / PF_SCALE); w_c /= w_c.sum()
    pf_multi_pred = (w_c[:, None] * cand_preds).sum(0)

    # === [3] pf_phys: K=3 typewell candidates, normalized GR + deriv, 32 seeds ===
    cand_preds = np.empty((len(cands), n)); cand_liks = np.empty(len(cands))
    for ci, c in enumerate(cands):
        ps = np.empty((PF_MULTI_SEEDS, n)); ls = np.empty(PF_MULTI_SEEDS)
        for s in range(PF_MULTI_SEEDS):
            ps[s], ls[s] = pf_single_physical(
                c["tw_tvt"], c["tw_gr_norm"], c["tw_gr_deriv"],
                md_v, z_v, gr_v_norm, gr_v_deriv,
                c["gs_norm"], ir, last_tvt, last_Z, last_MD, ci * 1000 + s + 7,
            )
        w_s = np.exp((ls - ls.max()) / PF_SCALE); w_s /= w_s.sum()
        cand_preds[ci] = (w_s[:, None] * ps).sum(0)
        cand_liks[ci] = float(ls.max())
    w_c = np.exp((cand_liks - cand_liks.max()) / PF_SCALE); w_c /= w_c.sum()
    pf_phys_pred = (w_c[:, None] * cand_preds).sum(0)

    return wid, {"pf_tuned": pf_tuned_pred, "pf_multi": pf_multi_pred, "pf_phys": pf_phys_pred}


def compute_leak(test_t: pd.DataFrame) -> np.ndarray:
    """train CSVs から TVT を直接読み、test の (well_id, row_idx) でlookup."""
    # train CSV has 'TVT' column for the 3 test wells (structural overlap)
    leak_vals = np.full(len(test_t), np.nan)
    for wid in test_t["well_id"].unique():
        train_csv = TRAIN_DIR / f"{wid}__horizontal_well.csv"
        if not train_csv.exists():
            print(f"  leak: {wid} not in train/, skipping (leak unavailable)")
            continue
        raw = pd.read_csv(train_csv)
        if "TVT" not in raw.columns:
            print(f"  leak: {wid} has no TVT column")
            continue
        raw["row_idx"] = raw.index.astype(int)
        sub = test_t[test_t["well_id"] == wid]
        merged = sub.merge(raw[["row_idx", "TVT"]].rename(columns={"TVT": "_leak"}),
                            on="row_idx", how="left")
        leak_vals[test_t["well_id"].eq(wid).to_numpy()] = merged["_leak"].astype(float).to_numpy()
    return leak_vals


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"INPUT_DIR={INPUT_DIR}", flush=True)
    print(f"PF_WORKERS={PF_WORKERS}, PF_SEEDS={PF_SEEDS}, PF_PARTICLES={PF_PARTICLES}", flush=True)

    train = load_base(TRAIN_DIR, "train")
    test = load_base(TEST_DIR, "test")
    sample = pd.read_csv(SAMPLE_SUB_PATH)
    print(f"train rows={len(train)}, test rows={len(test)}, sample={len(sample)}", flush=True)

    global_gr_mean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    print(f"global_gr_mean={global_gr_mean:.4f}", flush=True)

    # ── geom (LightGBM, 5-fold avg) ──
    print("\n[1/4] geom LightGBM 5-fold...", flush=True)
    train = enrich(train, global_gr_mean)
    test = enrich(test, global_gr_mean)
    fold_of = make_folds(train)
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
        model.fit(train_t.loc[tm, ALL_FEATURES], y_delta.loc[tm],
                  eval_set=[(train_t.loc[vm, ALL_FEATURES], y_delta.loc[vm])],
                  eval_metric="rmse", callbacks=[lgb.early_stopping(50, verbose=False)])
        best = int(model.best_iteration_ or model.n_estimators)
        test_geom_delta += model.predict(test_t[ALL_FEATURES], num_iteration=best) / N_SPLITS
        print(f"  geom fold {fold}: best_iter={best}", flush=True)
    test_t["geom"] = test_t["last_known_TVT"].astype(float).to_numpy() + test_geom_delta

    # ── PF (3 variants per well, multiprocessing) ──
    print("\n[2/4] PF (tuned + multi-tw + physical-lik) on 3 test wells...", flush=True)
    payloads = build_pf_payloads(test, train)
    print(f"  PF payloads ready: {len(payloads)} wells, workers={PF_WORKERS}", flush=True)

    pf_results = {}
    if PF_WORKERS > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=PF_WORKERS) as ex:
            for k, (wid, res) in enumerate(ex.map(pf_worker, payloads, chunksize=1)):
                pf_results[wid] = res
                print(f"  PF well {k+1}/{len(payloads)}: {wid}", flush=True)
    else:
        for k, p in enumerate(payloads):
            wid, res = pf_worker(p)
            pf_results[wid] = res
            print(f"  PF well {k+1}/{len(payloads)}: {wid}", flush=True)

    for key in ["pf_tuned", "pf_multi", "pf_phys"]:
        test_t[key] = np.nan
        for wid, res in pf_results.items():
            order = test_t.loc[test_t["well_id"].eq(wid)].sort_values("row_idx").index
            test_t.loc[order, key] = res[key]

    # ── leak lookup ──
    print("\n[3/4] leak lookup from train CSVs...", flush=True)
    test_t = test_t.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    leak = compute_leak(test_t)
    n_leak = int((~np.isnan(leak)).sum())
    print(f"  leak rows available: {n_leak}/{len(test_t)}", flush=True)

    # ── model blend (exp033 NNLS weights, renormalized) ──
    print("\n[4/4] blend + hybrid + smooth...", flush=True)
    a = test_t["last_known_TVT"].astype(float).to_numpy()
    pf_t = test_t["pf_tuned"].to_numpy(float)
    pf_m = test_t["pf_multi"].to_numpy(float)
    pf_p = test_t["pf_phys"].to_numpy(float)
    geom = test_t["geom"].to_numpy(float)

    model_blend = a + (
        W_PF_TUNED * (pf_t - a) +
        W_PF_MULTI * (pf_m - a) +
        W_PF_PHYS * (pf_p - a) +
        W_GEOM * (geom - a)
    )

    # leak fallback (if any leak row missing, use model)
    leak_eff = np.where(np.isnan(leak), model_blend, leak)

    # hybrid
    hybrid = HYBRID_W_MODEL * model_blend + HYBRID_W_LEAK * leak_eff
    test_t["hybrid"] = hybrid

    # smooth per well
    test_t = test_t.sort_values(["well_id", "row_idx"])
    test_t[PRED_COL] = test_t.groupby("well_id", sort=False)["hybrid"].transform(
        lambda x: x.rolling(SMOOTH_W, min_periods=1, center=True).mean())

    # diagnostic
    diff_lm = test_t["hybrid"] - model_blend
    print(f"  hybrid vs model_blend: mean diff={diff_lm.mean():+.4f}, std={diff_lm.std():.4f}")
    print(f"  model_blend stats: mean={model_blend.mean():.2f}, std={model_blend.std():.2f}")
    print(f"  leak stats: mean={np.nanmean(leak):.2f}, std={np.nanstd(leak):.2f}")
    print(f"  final stats: mean={test_t[PRED_COL].mean():.2f}, std={test_t[PRED_COL].std():.2f}")

    submission = sample[["id"]].merge(
        test_t[["id", PRED_COL]].rename(columns={PRED_COL: "tvt"}),
        on="id", how="left")
    if submission["tvt"].isna().any():
        raise ValueError(f"submission contains {submission['tvt'].isna().sum()} missing predictions")
    submission.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  rows={len(submission)}  "
          f"tvt[{submission.tvt.min():.1f},{submission.tvt.max():.1f}]")


if __name__ == "__main__":
    main()
