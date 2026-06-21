"""ROGII exp026 — PF × geom self-contained blend (Kaggle Notebook submission).

最終best(ローカル)= PF(GR-typewell系列トラッカー) × geom(幾何外挿LightGBM) の NNLSブレンド+平滑。
本kernelは PF と geom を **raw CSVから自己完結で再計算**して提出する。

CV (local, leak-free nested-fold):
  PF(tuned) 10.984 / geom(exp014) 13.525 → 0.683*PF + 0.392*geom + 平滑(w101) = 10.106

構成:
  geom : SAFE + Group A/B/C/D(GR rolling) + Group F(幾何外挿) を LightGBM delta回帰
         (5-fold GroupKFold-by-well, test=5fold平均)。exp014相当。
  PF   : 各test wellで typewell GR-TVTプロファイルに対し状態 pos=TVT+Z を粒子フィルタ追跡
         (init_spread=4, PN=0.01, 128seed×500粒子, 尤度加重)。tuned config。
  blend: anchor + 0.683*(PF-anchor) + 0.392*(geom-anchor) → well内 row順 mean平滑(w=101)。

leak-free: hidden TVT不使用。GR/typewell/anchor/既知幾何(X/Y/Z/MD)のみ。
"""
from pathlib import Path
import re
import os
from concurrent.futures import ProcessPoolExecutor

import lightgbm as lgb
import numpy as np
import pandas as pd

PF_WORKERS = max(1, min(4, (os.cpu_count() or 2) - 1))  # Kaggle CPU ~4 cores

# ── locations ─────────────────────────────────────────────────────────────────
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

# blend (from local 2-way NNLS on leak-free OOF) + smoothing
BLEND_PF = 0.683
BLEND_GEOM = 0.392
SMOOTH_W = 101

# PF tuned config
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


# ── Group A/B/C ───────────────────────────────────────────────────────────────
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


# ── Group D ───────────────────────────────────────────────────────────────────
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


# ── Group F geometric extrapolation (from exp014) ─────────────────────────────
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


# ── Particle Filter (tuned) on test wells ─────────────────────────────────────
def load_typewell(split_dir: Path, well_id: str) -> pd.DataFrame:
    p = split_dir / f"{well_id}__typewell.csv"
    tw = pd.read_csv(p)
    return tw[["TVT", "GR"]].copy()


def pf_single(tw_tvt, tw_gr, md_v, z_v, gr_v, gs, ir, last_tvt, last_Z, last_MD, seed):
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = PF_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (last_tvt + last_Z) + PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n); prev_MD = last_MD; log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PF_PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr); d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk; ws = w.sum(); w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < PF_RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(N); rate = rate[idx] + PF_RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i])); prev_MD = md_v[i]
    return res, log_lik


def pf_build_payload(g: pd.DataFrame, tw: pd.DataFrame) -> dict:
    """1 well のPF入力(numpy配列)を組み立て。pickle可能なdictで返す(マルチプロセス用)。"""
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


def pf_worker(p: dict):
    """1 well を 128 seed 尤度加重アンサンブル(マルチプロセスのunit)。(wid, pred) を返す。"""
    wid = p["wid"]; n = p["n"]
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])
    preds = np.empty((PF_SEEDS, n)); liks = np.empty(PF_SEEDS)
    for s in range(PF_SEEDS):
        preds[s], liks[s] = pf_single(p["tw_tvt"], p["tw_gr"], p["md_v"], p["z_v"], p["gr_v"],
                                      p["gs"], p["ir"], p["last_tvt"], p["last_Z"], p["last_MD"], s)
    wts = np.exp((liks - liks.max()) / PF_SCALE); wts /= wts.sum()
    return wid, (wts[:, None] * preds).sum(0)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"INPUT_DIR={INPUT_DIR}")
    train = load_base(TRAIN_DIR, "train")
    test = load_base(TEST_DIR, "test")
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    global_gr_mean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())
    print(f"global_gr_mean={global_gr_mean:.4f}")

    # ── geom (LightGBM, 5-fold avg test) ──
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
        print(f"geom fold {fold}: best_iter={best}")
    test_t = test_t.copy()
    test_t["geom"] = test_t["last_known_TVT"].astype(float).to_numpy() + test_geom_delta

    # ── PF on each test well (multiprocess over wells; CV-identical to single-thread) ──
    payloads = []
    for wid, g in test.groupby("well_id", sort=False):
        if not g["is_target"].any():
            continue
        payloads.append(pf_build_payload(g, load_typewell(TEST_DIR, wid)))
    n_wells = len(payloads)
    print(f"PF: {n_wells} test wells, workers={PF_WORKERS}, seeds={PF_SEEDS}")
    pf_by_wid = {}
    if PF_WORKERS > 1 and n_wells > 1:
        with ProcessPoolExecutor(max_workers=PF_WORKERS) as ex:
            for k, (wid, pred) in enumerate(ex.map(pf_worker, payloads, chunksize=1)):
                pf_by_wid[wid] = pred
                if (k + 1) % 50 == 0:
                    print(f"  PF {k+1}/{n_wells} wells", flush=True)
    else:
        for k, p in enumerate(payloads):
            wid, pred = pf_worker(p)
            pf_by_wid[wid] = pred
    test_t["pf"] = np.nan
    for wid, pred in pf_by_wid.items():
        order = test_t.loc[test_t["well_id"].eq(wid)].sort_values("row_idx").index
        test_t.loc[order, "pf"] = pred

    # ── blend + smooth ──
    a = test_t["last_known_TVT"].astype(float).to_numpy()
    blend = a + BLEND_PF * (test_t["pf"].to_numpy(float) - a) + BLEND_GEOM * (test_t["geom"].to_numpy(float) - a)
    test_t["blend"] = blend
    test_t = test_t.sort_values(["well_id", "row_idx"])
    test_t[PRED_COL] = test_t.groupby("well_id", sort=False)["blend"].transform(
        lambda x: x.rolling(SMOOTH_W, min_periods=1, center=True).mean())

    submission = sample[["id"]].merge(
        test_t[["id", PRED_COL]].rename(columns={PRED_COL: "tvt"}), on="id", how="left")
    if submission["tvt"].isna().any():
        raise ValueError("submission contains missing predictions")
    submission.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} rows={len(submission)}  tvt[{submission.tvt.min():.1f},{submission.tvt.max():.1f}]")


if __name__ == "__main__":
    main()
