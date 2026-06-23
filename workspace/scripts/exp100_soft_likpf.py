#!/usr/bin/env python3
"""exp100: Soft likelihood Particle Filter — gracefully-degrading改良 (HEAVY).

exp022/exp040 の動く PF を base に、以下の soft・gated・幾何結合な要素を ablation 可能に追加:
  1. multi-scale GR尤度 (exp040 の {3,5,8,12}) を既定化。
  2. GR-typewell波形相関の soft prior (局所窓の正規化相互相関を尤度に加算)。
  3. 幾何結合の残差状態 (PF状態を絶対TVTでなく幾何予測まわりの残差offsetにし process noise小)。
  4. robust(重裾) 尤度 (student-t / soft-Huber 的: minimum で d^2 を飽和させる軟clip)。
  5. per-well affine GR較正 (a,b) を既知prefix(TVT_input可視区間)のみで推定 = leak-free。
  6. multi-seed(>=8, 既定 N_SEEDS) で安定化し、seed間分散を per-well 信頼度として出力。

全要素 OFF = exp022 相当(~11.02)に戻る gracefully-degrading 設計。

leak規約: hidden TVT不使用。GR+typewell+anchor+Z+MD+TVT_input(可視prefix)のみ。
評価は hidden tail 行のみ。well GroupKFold と typewell-grouped fold の両方で算出(honest=typewell)。

**重い実験**: well単位に ProcessPoolExecutor で並列化。runtime は notes/result に明記。
使い方:
  --smoke           : 少数well・少seedで全段完走確認
  --smoke-wells N   : smoke の well 数 (default 30)
  --ablation        : 各要素 on/off の CV を追加で算出 (full only)
  --seeds N         : seed 数上書き
"""

from __future__ import annotations

import sys
import os
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp100_soft_likpf"
OUT_DIR = Path("experiments") / EXP_ID

# ---- PF hyperparams (exp022/exp040 由来) ----
N_PARTICLES = 500
N_SEEDS_DEFAULT = 8          # >=8 で multi-seed 分散を出す (規約6)
SCALES = [3.0, 5.0, 8.0, 12.0]
N_WORKERS = max(1, min(9, (os.cpu_count() or 4) - 1))

MOM = 0.998
VN = 0.002
PN_ABS = 0.01        # 絶対TVT状態時の process noise
PN_RES = 0.004       # 残差offset状態時の process noise(緩いドリフトのみ=小さく, 規約3)
RP = 0.1
RR = 0.001
RESAMP = 0.5
INIT_SPREAD = 4.0

# ---- soft prior / robust params ----
CORR_WIN = 9         # 波形相関の局所窓(行数)
CORR_WEIGHT = 0.15   # shape項の尤度寄与の重み(gentle soft prior)
HUBER_K = 2.0        # soft-Huber 閾値(|d|>K で線形・重裾)
AFFINE_SHRINK = 0.3  # affine較正の identity への収縮(0=恒等, 1=full moment-match)


# ablation 用フラグ辞書 (全 True が exp100 既定)
DEFAULT_FLAGS = {
    "multiscale": True,    # 1 — 効果確認済(13.60)
    "shape_corr": False,   # 2 — 単体で悪化(対relax後も)→既定OFF, ablation可
    "residual": False,     # 3 — 単体で悪化→既定OFF, ablation可
    "robust": True,        # 4 — 効果確認済(13.56, 単体最良)
    "affine": False,       # 5 — 単体で悪化→既定OFF(shrink版を ablation で再評価)
}


def _pf_single(p, seed, flags):
    """1 seed の PF。(pred_eval[n], log_lik)。

    residual=True: 状態を「幾何予測 geo_v[i] まわりの残差 offset」にする。
      pos_resid ~ offset, TVT_pred = geo_v[i] + pos_resid。process noise 小(緩いドリフト)。
    residual=False: exp022 同様 pos = TVT+Z の絶対状態。
    """
    tw_tvt = p["tw_tvt"]; tw_gr = p["tw_gr"]
    md_v = p["md_v"]; z_v = p["z_v"]; gr_v = p["gr_v"]
    geo_v = p["geo_v"]               # 幾何予測 TVT (各 target 行)
    gs = p["gs"]; ir = p["ir"]
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = N_PARTICLES
    rng = np.random.default_rng(seed)

    use_resid = flags["residual"]
    use_ms = flags["multiscale"]
    use_corr = flags["shape_corr"]
    use_robust = flags["robust"]

    if use_resid:
        # 残差 offset 状態: anchor(last_tvt) と幾何予測 geo[0] の差を初期 offset に
        off0 = p["last_tvt"] - p["geo_last"]
        st = off0 + INIT_SPREAD * rng.standard_normal(N)   # offset particles
        pn = PN_RES
    else:
        st = (p["last_tvt"] + p["last_Z"]) + INIT_SPREAD * rng.standard_normal(N)  # pos=TVT+Z
        pn = PN_ABS
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    prev_MD = p["last_MD"]
    log_lik = 0.0
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100

    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        if use_resid:
            # 緩いドリフト補正のみ: rate は残差offsetの変化率
            rate = MOM * rate + VN * rng.standard_normal(N)
            st = st + rate * dm_step + pn * rng.standard_normal(N)
            tvt_p = np.clip(geo_v[i] + st, lo, hi)
            st = tvt_p - geo_v[i]
        else:
            rate = MOM * rate + VN * rng.standard_normal(N)
            st = st + rate * dm_step + pn * rng.standard_normal(N)
            tvt_p = np.clip(st - z_v[i], lo, hi)
            st = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        d2 = d * d
        if use_robust:
            # soft-Huber: core は Gaussian, 外れ(>HUBER_K)は線形に減衰(重裾だが平坦化しない)。
            ad = np.abs(d)
            quad = np.minimum(d2, 600.0)
            rho = np.where(ad <= HUBER_K, quad,
                           HUBER_K * (2.0 * ad - HUBER_K))
            lk = np.exp(-0.5 * np.minimum(rho, 600.0))
        else:
            lk = np.exp(-0.5 * np.minimum(d2, 600.0))

        # shape項: 局所窓 GR波形と typewell波形(各粒子位置の窓)の正規化相互相関。
        # gated soft prior: 乗数 = exp(CORR_WEIGHT*(corr-corr_ref))。corr が窓平均参照より高い粒子を緩く優遇。
        if use_corr and p["corr_ok"] and i >= CORR_WIN:
            ec = np.interp(p["corr_tvt_off"][:, None] + tvt_p[None, :],
                           tw_tvt, tw_gr)            # (win, N) typewell窓 (粒子毎)
            ow = gr_v[i - CORR_WIN + 1: i + 1]       # (win,) observed窓(因果: 既走行のみ)
            corr = _ncc(ow, ec)                      # (N,) in [-1,1]
            lk = lk * np.exp(CORR_WEIGHT * (corr - corr.mean()))

        lk = np.maximum(lk, 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            st = st[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        if use_resid:
            res[i] = float(np.dot(w, geo_v[i] + st))
        else:
            res[i] = float(np.dot(w, st - z_v[i]))
        prev_MD = md_v[i]

    # use_ms はアンサンブル段で SCALE を変えるのみ; ここでは log_lik を返す
    return res, log_lik


def _ncc(obs, ec):
    """正規化相互相関。obs:(win,), ec:(win,N) -> (N,)。定数窓は corr=0。"""
    o = obs - obs.mean()
    on = np.sqrt((o * o).sum())
    if on < 1e-9:
        return np.zeros(ec.shape[1])
    e = ec - ec.mean(axis=0, keepdims=True)
    en = np.sqrt((e * e).sum(axis=0))
    en = np.where(en < 1e-9, 1e-9, en)
    return (o[:, None] * e).sum(axis=0) / (on * en)


def _process_well(args):
    """1 well を multi-seed で PF。seed間分散を信頼度として返す。
    戻り: (wid, pred[n], seed_std_mean)。"""
    p, n_seeds, flags = args
    wid = p["wid"]; n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0), 0.0
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"]), 0.0
    preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds)
    for s in range(n_seeds):
        preds[s], liks[s] = _pf_single(p, s, flags)

    if flags["multiscale"]:
        final = np.zeros(n)
        for scale in SCALES:
            wts = np.exp((liks - liks.max()) / scale); wts /= wts.sum()
            final += (wts[:, None] * preds).sum(0) / len(SCALES)
    else:
        wts = np.exp((liks - liks.max()) / 8.0); wts /= wts.sum()
        final = (wts[:, None] * preds).sum(0)

    # seed間分散(規約6): per-seed point予測(均一加重)のばらつき
    seed_mean_pred = preds.mean(0)
    seed_std = float(np.sqrt(((preds - seed_mean_pred) ** 2).mean()))
    return wid, final, seed_std


def _geom_predict(known, tgt):
    """exp014 Group F 相当の幾何外挿 TVT 予測 (leak-free)。
    既知 prefix の末尾傾き (dTVT_input/dMD) を anchor から線形外挿。
    戻り: geo_v(target各行), geo_last(anchor行のgeo), ir(初期rate)。"""
    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0
    # TVT の MD 微分(pos=TVT+Z の rate と整合: d(TVT)/dMD)
    slope_tvt = float(np.median(dt[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0
    last = known.iloc[-1]
    last_md = float(last["MD"]); last_tvt = float(last["TVT_input"])
    md_v = tgt["MD"].to_numpy(float)
    geo_v = last_tvt + slope_tvt * (md_v - last_md)
    geo_last = last_tvt
    return geo_v, geo_last, ir, slope_tvt


def build(base_path, tw_path, flags):
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    payloads = []; out_frames = []
    for wid, g in sel.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])
        out_frames.append(tgt[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy())
        tw_g = tw_by_well.get(wid)
        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both")
        if tw_g is None or len(tw_g) < 2 or len(known) < 2:
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor,
                             "n_eval": int(len(tgt))})
            continue
        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tw_tvt = tw_s["TVT"].to_numpy(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
        gr_v = gr_full[tgt_mask]

        k_tvt = known["TVT_input"].to_numpy(float)
        k_gr = known["GR"].fillna(0).to_numpy(float)
        tw_at_k = np.interp(k_tvt, tw_tvt, tw_gr)

        # ---- (5) per-well affine GR較正 a,b を既知prefixのみで推定 (leak-free) ----
        a, b = 1.0, 0.0
        if flags["affine"] and len(k_gr) >= 10 and np.nanstd(k_gr) > 1e-6:
            # GR scale整合のみ(平均・分散マッチ)。pointwise回帰は別経路GRでノイズ化するため不採用。
            # k_gr の分布を tw_at_k の分布に moment-match: a=std比, b=平均差。identity に向け shrink。
            sk = np.nanstd(k_gr); st = np.nanstd(tw_at_k)
            if sk > 1e-6 and np.isfinite(st):
                a_ = float(st / sk)
                b_ = float(np.nanmean(tw_at_k) - a_ * np.nanmean(k_gr))
                a_ = float(np.clip(a_, 0.5, 2.0))
                # shrink toward identity (AFFINE_SHRINK=0 で恒等)
                a = 1.0 + AFFINE_SHRINK * (a_ - 1.0)
                b = AFFINE_SHRINK * b_
        gr_v_cal = a * gr_v + b
        k_gr_cal = a * k_gr + b
        gs = float(np.clip(np.nanstd(k_gr_cal - tw_at_k), 10., 60.))

        geo_v, geo_last, ir, _ = _geom_predict(known, tgt)
        last = known.iloc[-1]

        # corr 用: 各 target 行で window の TVT offsets を概算(局所 dTVT/行 を一定近似)
        corr_ok = flags["shape_corr"] and len(gr_v) >= CORR_WIN
        corr_tvt_off = None
        if corr_ok:
            # 因果trailing窓 (offsets -(WIN-1)..0) の TVT 変位を幾何傾きで近似。
            mdv = tgt["MD"].to_numpy(float)
            span_md = max(1.0, mdv[-1] - mdv[0])
            slope = (geo_v[-1] - geo_v[0]) / span_md   # dTVT/dMD
            step_md = float(np.median(np.diff(mdv))) if len(mdv) > 1 else 1.0
            dtvt_per_row = slope * step_md
            corr_tvt_off = (np.arange(-CORR_WIN + 1, 1, dtype=float)) * dtvt_per_row

        payloads.append({
            "wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
            "tw_tvt": tw_tvt, "tw_gr": tw_gr,
            "md_v": tgt["MD"].to_numpy(float), "z_v": tgt["Z"].to_numpy(float),
            "gr_v": gr_v_cal,
            "geo_v": geo_v, "geo_last": geo_last,
            "gs": gs, "ir": ir,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]),
            "last_MD": float(last["MD"]), "anchor": anchor,
            "corr_ok": corr_ok, "corr_tvt_off": corr_tvt_off,
            "affine_a": a, "affine_b": b,
        })
    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


def run_split(base_path, tw_path, flags, n_seeds, well_subset=None):
    payloads, out = build(base_path, tw_path, flags)
    if well_subset is not None:
        keep = set(well_subset)
        payloads = [p for p in payloads if p["wid"] in keep]
        out = out[out["well_id"].isin(keep)].reset_index(drop=True)
    pred_by_wid = {}; std_by_wid = {}
    tasks = [(p, n_seeds, flags) for p in payloads]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, (wid, pred, std) in enumerate(ex.map(_process_well, tasks, chunksize=4)):
            pred_by_wid[wid] = pred; std_by_wid[wid] = std
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(payloads)} wells done", flush=True)
    pred_col = np.empty(len(out)); std_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
        std_col[g.index.to_numpy()] = std_by_wid[wid]
    out = out.copy()
    out["pred_tvt"] = pred_col
    out["seed_std"] = std_col
    return out


def _fold_rmse(preds, fold_map, col_true="TVT", col_pred="pred_tvt"):
    """fold毎に RMSE と pooled を返す。"""
    p = preds.copy()
    p["fold"] = p["well_id"].map(fold_map)
    fold_rmse = []
    for fk in sorted(p["fold"].dropna().unique()):
        gg = p[p["fold"] == fk]
        fold_rmse.append(float(tvt_rmse(gg[col_true], gg[col_pred])))
    pooled = float(tvt_rmse(p[col_true], p[col_pred]))
    return pooled, fold_rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-wells", type=int, default=30)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_seeds = args.seeds if args.seeds else (3 if args.smoke else N_SEEDS_DEFAULT)
    wall_start = time.time()

    base = "data/processed/train_base_v001.parquet"
    tw = "data/processed/typewell_train_base_v001.parquet"
    wellfold = pd.read_csv("data/folds/folds_group_well_v001.csv")
    twfold = pd.read_csv("data/folds/folds_group_typewell_v001.csv")
    wf_map = dict(zip(wellfold["well_id"], wellfold["fold"]))
    tf_map = dict(zip(twfold["well_id"], twfold["fold"]))

    well_subset = None
    if args.smoke:
        all_w = list(wellfold["well_id"])[: args.smoke_wells]
        well_subset = all_w
        print(f"[{EXP_ID}] SMOKE: {len(well_subset)} wells, seeds={n_seeds}", flush=True)
    else:
        print(f"[{EXP_ID}] FULL 773 wells, seeds={n_seeds}, workers={N_WORKERS}", flush=True)

    flags = dict(DEFAULT_FLAGS)
    print(f"  flags={flags}", flush=True)
    preds = run_split(base, tw, flags, n_seeds, well_subset)

    cv_well, fold_well = _fold_rmse(preds, wf_map)
    cv_tw, fold_tw = _fold_rmse(preds, tf_map)
    anc = float(tvt_rmse(preds["TVT"], preds["last_known_TVT"]))
    print(f"  CV well-fold = {cv_well:.6f}  typewell-fold = {cv_tw:.6f}  anchor = {anc:.6f}", flush=True)

    preds["error"] = preds["pred_tvt"] - preds["TVT"]
    preds["abs_error"] = preds["error"].abs()
    preds.to_csv(OUT_DIR / "oof.csv", index=False)

    well_rows = []
    for wid, g in preds.groupby("well_id"):
        well_rows.append({"well_id": wid, "n": len(g),
                          "anchor_rmse": float(tvt_rmse(g["TVT"], g["last_known_TVT"])),
                          "pf_rmse": float(tvt_rmse(g["TVT"], g["pred_tvt"])),
                          "seed_std": float(g["seed_std"].iloc[0])})
    well = pd.DataFrame(well_rows)
    well.to_csv(OUT_DIR / "per_well.csv", index=False)
    n_beat = int((well["pf_rmse"] < well["anchor_rmse"]).sum())
    n_broken = int((well["pf_rmse"] > 20).sum())
    print(f"  PF beats anchor: {n_beat}/{len(well)}  broken(>20): {n_broken}", flush=True)

    # ---- ablation (full only) ----
    ablation = {}
    if args.ablation:
        print("ABLATION ...", flush=True)
        # all-off = exp022相当
        # all_off=exp022相当(gracefully-degrade検証)。default=採用構成。
        # 各 optional 要素を default に足した版 + default から各要素を抜いた版。
        # 本質確認に絞る(matched-seed): all_off(exp022相当) / default / 採用要素leave-one-out / affine追加
        configs = {"all_off": {k: False for k in DEFAULT_FLAGS},
                   "default": dict(DEFAULT_FLAGS)}
        for k in ("multiscale", "robust"):  # default から採用要素を抜く(寄与確認)
            c = dict(DEFAULT_FLAGS); c[k] = False
            configs[f"default_minus_{k}"] = c
        c = dict(DEFAULT_FLAGS); c["affine"] = True
        configs["default_plus_affine"] = c
        for name, fl in configs.items():
            pp = run_split(base, tw, fl, n_seeds, well_subset)
            cw, _ = _fold_rmse(pp, wf_map)
            ct, _ = _fold_rmse(pp, tf_map)
            ablation[name] = {"cv_wellfold": cw, "cv_typewellfold": ct}
            print(f"  ablation {name}: well={cw:.4f} tw={ct:.4f}", flush=True)

    runtime = time.time() - wall_start
    result = {
        "exp_id": EXP_ID, "created_at": now_jst(),
        "status": "smoke" if args.smoke else "completed",
        "method": "Soft lik-PF: multiscale+shape-corr+residual-geom+robust-t+affine-cal, multi-seed",
        "flags": flags,
        "cv_rmse_wellfold": cv_well, "cv_rmse_typewellfold": cv_tw,
        "fold_rmse_wellfold": fold_well, "fold_rmse_typewellfold": fold_tw,
        "anchor_rmse": anc,
        "n_wells": int(len(well)), "n_seeds": n_seeds,
        "n_pf_beats_anchor": n_beat, "n_broken": n_broken,
        "ablation": ablation,
        "seed_variance_available": True,
        "leak_risk": "none (no hidden TVT; GR+typewell+anchor+Z+MD+TVT_input prefix only; affine cal on prefix only)",
        "compare_exp022": 11.024, "compare_exp040": 10.979,
        "runtime_sec": runtime,
        "notes_short": f"{'SMOKE ' if args.smoke else ''}wellCV={cv_well:.4f} twCV={cv_tw:.4f} vs exp022=11.024/exp040=10.979",
    }
    write_json(OUT_DIR / "result.json", result)
    print(f"\n[{EXP_ID}] done in {runtime:.1f}s  wellCV={cv_well:.6f}  twCV={cv_tw:.6f}", flush=True)


if __name__ == "__main__":
    main()
