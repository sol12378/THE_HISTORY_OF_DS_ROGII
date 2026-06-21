#!/usr/bin/env python3
"""exp066: 群IV信号変換 emission品質テスト(leak-free評価).

核心仮説: lateral GR↔typewell GR照合のoffset局在化を、drift不変な信号表現
(Hilbert位相/微分/散乱/最適輸送)が生GRより改善するか(特にbroken well)。

評価方法(leak-free): 各wellのhidden区間で、各MD点の真TVT近傍に候補TVT格子を張り、
各候補で「lateral GR窓 vs typewell GR窓(その候補TVT位置)」の照合スコアを計算。
スコアが最大の候補TVTと真TVTの差(局在化誤差)を測る。
真TVTは候補格子の中心決定と評価にのみ使用=モデル(照合)自体はGRのみ。
→ これは「emissionの識別力」を測る研究評価。照合がTVTを当てられるか。

representations:
- raw: 生GR差の二乗(現行PF emission)
- norm: robust正規化GR(移動中央値+MAD)
- deriv: 正規化GRの微分(drift不変)
- hilbert: 瞬時位相(振幅drift完全不変)
- wasserstein: 窓内GR分布のWasserstein距離
- wavelet: pywt近似係数の相関
"""
from __future__ import annotations
import numpy as np, pandas as pd, glob, json
from pathlib import Path
from scipy.signal import hilbert
from scipy.ndimage import median_filter
from scipy.stats import wasserstein_distance
import pywt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/exp066_emission_test"; OUT.mkdir(parents=True, exist_ok=True)
W = 15           # 照合窓 半幅(行)
GRID = 60        # 候補TVT格子半幅(ft)
GRID_STEP = 1.0


def robust_norm(g, win=151):
    g = np.asarray(g, float)
    base = median_filter(g, size=min(win, len(g) | 1), mode="nearest")
    r = g - base
    mad = np.median(np.abs(r - np.median(r))) + 1e-6
    return r / (1.4826 * mad)


def inst_phase(g):
    z = robust_norm(g)
    return np.angle(hilbert(z))


def wavelet_approx(g):
    z = robust_norm(g)
    try:
        c = pywt.wavedec(z, "db2", level=2)
        a = c[0]
        return np.interp(np.linspace(0, 1, len(z)), np.linspace(0, 1, len(a)), a)
    except Exception:
        return z


def score_window(lat_win, tw_win, kind):
    """高いほど良い一致。kind別の照合スコア."""
    if len(lat_win) < 3 or len(tw_win) < 3:
        return -1e9
    n = min(len(lat_win), len(tw_win))
    a = np.asarray(lat_win[:n], float); b = np.asarray(tw_win[:n], float)
    if kind == "wasserstein":
        return -wasserstein_distance(a, b)
    # NCC (zero-mean unit-var) for shape-based; negative SSE for raw
    if kind == "raw":
        return -np.mean((a - b) ** 2)
    am = a - a.mean(); bm = b - b.mean()
    da = np.sqrt((am ** 2).sum()); db = np.sqrt((bm ** 2).sum())
    if da < 1e-6 or db < 1e-6:
        return -1e9
    return float((am * bm).sum() / (da * db))


def main():
    base = pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "is_target", "is_known_tvt"])
    tw = pd.read_parquet("data/processed/typewell_train_base_v001.parquet", columns=["well_id", "TVT", "GR"])
    tw_by = {w: g.sort_values("TVT") for w, g in tw.groupby("well_id", sort=False)}
    per = pd.read_csv("experiments/exp022_particle_filter/per_well.csv")
    broken = per[per["pf_rmse"] > 20]["well_id"].tolist()[:15]
    good = per[per["pf_rmse"] < 5]["well_id"].tolist()[:15]
    wells = broken + good
    label = {**{w: "broken" for w in broken}, **{w: "good" for w in good}}

    KINDS = ["raw", "norm", "deriv", "hilbert", "wasserstein", "wavelet"]
    # localization error per kind per group
    errs = {k: {"broken": [], "good": []} for k in KINDS}

    rng = np.random.default_rng(0)
    for wid in wells:
        g = base[base["well_id"] == wid].sort_values("row_idx")
        twg = tw_by.get(wid)
        if twg is None or len(twg) < 10:
            continue
        tw_tvt = twg["TVT"].to_numpy(float)
        tw_gr_raw = twg["GR"].fillna(twg["GR"].mean()).to_numpy(float)
        # precompute typewell representations on its TVT grid
        tw_rep = {
            "raw": tw_gr_raw, "norm": robust_norm(tw_gr_raw),
            "deriv": np.gradient(robust_norm(tw_gr_raw)),
            "hilbert": inst_phase(tw_gr_raw), "wasserstein": robust_norm(tw_gr_raw),
            "wavelet": wavelet_approx(tw_gr_raw),
        }
        lat_gr = g["GR"].interpolate(limit_direction="both").fillna(tw_gr_raw.mean()).to_numpy(float)
        lat_rep = {
            "raw": lat_gr, "norm": robust_norm(lat_gr), "deriv": np.gradient(robust_norm(lat_gr)),
            "hilbert": inst_phase(lat_gr), "wasserstein": robust_norm(lat_gr), "wavelet": wavelet_approx(lat_gr),
        }
        tgt = g["is_target"].to_numpy(bool)
        true_tvt = g["TVT"].to_numpy(float)
        tw_spacing = np.median(np.diff(tw_tvt)) if len(tw_tvt) > 1 else 0.5
        idx_tgt = np.where(tgt)[0]
        if len(idx_tgt) < 30:
            continue
        sel = rng.choice(idx_tgt, size=min(40, len(idx_tgt)), replace=False)  # sample rows for speed
        for i in sel:
            if i - W < 0 or i + W >= len(lat_gr):
                continue
            t_true = true_tvt[i]
            cand = np.arange(t_true - GRID, t_true + GRID + 1e-9, GRID_STEP)  # 真TVT中心格子(評価用)
            for k in KINDS:
                latw = lat_rep[k][i - W:i + W + 1]
                best_s = -1e18; best_t = cand[0]
                for ct in cand:
                    # typewell窓: ct中心 ±W*tw_spacing をTVT軸で取り出し
                    lo = ct - W * tw_spacing; hi = ct + W * tw_spacing
                    m = (tw_tvt >= lo) & (tw_tvt <= hi)
                    if m.sum() < 3:
                        continue
                    tww = tw_rep[k][m]
                    # resample to window length
                    tww = np.interp(np.linspace(0, 1, len(latw)), np.linspace(0, 1, len(tww)), tww)
                    s = score_window(latw, tww, k)
                    if s > best_s:
                        best_s = s; best_t = ct
                errs[k][label[wid]].append(abs(best_t - t_true))

    res = {}
    print(f"{'kind':<12}{'broken_MAE':>12}{'good_MAE':>12}")
    for k in KINDS:
        b = float(np.mean(errs[k]["broken"])) if errs[k]["broken"] else np.nan
        gd = float(np.mean(errs[k]["good"])) if errs[k]["good"] else np.nan
        res[k] = {"broken_localization_MAE_ft": b, "good_localization_MAE_ft": gd,
                  "n_broken": len(errs[k]["broken"]), "n_good": len(errs[k]["good"])}
        print(f"{k:<12}{b:>12.3f}{gd:>12.3f}")
    (OUT / "result.json").write_text(json.dumps({
        "note": "Localization MAE: |argmax-score TVT - true TVT| over hidden rows. lower=better emission. truth used for grid center+eval only (leak-free model).",
        "window_halfwidth": W, "grid_halfwidth_ft": GRID, "results": res,
    }, indent=2))
    print("\n[interpret] raw が基準。これより broken_MAE が小さい表現=drift不変が効く=有望")
    print(f"saved {OUT}/result.json")


if __name__ == "__main__":
    main()
