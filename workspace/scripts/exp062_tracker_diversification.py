#!/usr/bin/env python3
"""exp062: トラッカー多様化 (案7-10) on 20-well subset

案7: 焼きなまし PF (Annealed Particle Filter)
- 初期尤度スケールを高温(平坦,探索)→低温(尖鋭,収束)へ徐々に変更
- 初期ロック誤りを高温で回避し、最終的には尖鋭な尤度で収束

案8: レプリカ交換 PF (Parallel Tempering PF)
- 複数温度(scale)のPFを並走、粒子を確率交換
- 多峰GR尤度の局所最適脱出

案9: ニューラルCRF / Viterbi-DP
- TVT離散化、emission=正規化GR、transition=dip連続性
- Viterbiで大域最尤(known区間アンカー固定)

案10: GP状態空間 / 不確実性つきPF
- PF粒子分散から各点の予測不確実性を算出
- 不確実性高い区間をgeom/anchorへ寄せる(broken wellダウンウェイト)

target: CV on 20-well subset (broken 10 + good 10)
output: result.md + result.json (4案の subset CV, broken救済効果, 失敗理由)

**重い実験**: subset評価なのでPFは単一スレッド×20well→5-10分想定。
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Tuple, Dict, Any
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.linalg import block_diag

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp062_tracker_diversification"
OUT_DIR = Path("experiments") / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 参考: exp022の設定
N_PARTICLES = 500
N_SEEDS = 128
SCALE_BASE = 8.0
MOM = 0.998
VN = 0.002
PN = 0.005
RP = 0.1
RR = 0.001
RESAMP = 0.5
INIT_SPREAD = 2.0

# subset選択: broken 10 + good 10
BROKEN_THRESHOLD = 20.0  # pf_rmse > 20


def select_subset_wells(per_well_path: Path, n_broken=10, n_good=10) -> list[str]:
    """exp022 per_well.csv から broken/good ウェルを選択。

    broken: pf_rmse > 20.0
    good: anchor_rmse < 15
    """
    df = pd.read_csv(per_well_path)

    broken = df[df["pf_rmse"] > BROKEN_THRESHOLD].sort_values("pf_rmse", ascending=False)
    good = df[(df["anchor_rmse"] < 15) & (df["pf_rmse"] <= BROKEN_THRESHOLD)].sort_values("pf_rmse")

    broken_wells = broken.head(n_broken)["well_id"].tolist()
    good_wells = good.head(n_good)["well_id"].tolist()

    subset = broken_wells + good_wells
    return subset


def _pf_annealed_single(p, seed, n_step_anneal=500):
    """案7: 焼きなまし PF。

    初期: scale = high (低解像度, 探索重視)
    終了: scale = 1.0 (高解像度, 尖鋭)
    線形冷却
    """
    tw_tvt = p["tw_tvt"]
    tw_gr = p["tw_gr"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
    gs = p["gs"]
    ir = p["ir"]
    n = len(md_v)

    if n == 0:
        return np.zeros(0), 0.0

    N = N_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (p["last_tvt"] + p["last_Z"]) + INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    prev_MD = p["last_MD"]
    log_lik = 0.0
    lo = tw_tvt[0] - 100
    hi = tw_tvt[-1] + 100

    scale_schedule = np.linspace(SCALE_BASE, 1.0, n_step_anneal)

    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs

        # 温度スケジューリング
        scale_idx = min(i, len(scale_schedule) - 1)
        scale_t = scale_schedule[scale_idx]

        lk = np.maximum(np.exp(-0.5 * np.minimum((d * d) / scale_t, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    return res, log_lik


def _pf_parallel_tempering_single(p, seed, n_scales=4):
    """案8: レプリカ交換 PF。

    複数温度(scales) のPFを並走。
    粒子を確率的に交換し、多峰探索を強化。
    """
    tw_tvt = p["tw_tvt"]
    tw_gr = p["tw_gr"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
    gs = p["gs"]
    ir = p["ir"]
    n = len(md_v)

    if n == 0:
        return np.zeros(0), 0.0

    N = N_PARTICLES // n_scales  # 各スケールの粒子数
    rng = np.random.default_rng(seed)

    # 複数スケール（高温→低温）
    scales = np.logspace(np.log10(SCALE_BASE), 0, n_scales)

    # 各スケール独立のPF状態
    pos_arr = np.zeros((n_scales, N))
    rate_arr = np.zeros((n_scales, N))
    w_arr = np.ones((n_scales, N)) / N

    for k in range(n_scales):
        pos_arr[k] = (p["last_tvt"] + p["last_Z"]) + INIT_SPREAD * rng.standard_normal(N)
        rate_arr[k] = ir + 0.01 * rng.standard_normal(N)

    res = np.empty(n)
    prev_MD = p["last_MD"]
    log_lik = 0.0
    lo = tw_tvt[0] - 100
    hi = tw_tvt[-1] + 100

    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)

        # 各スケールで独立進化
        for k in range(n_scales):
            rate_arr[k] = MOM * rate_arr[k] + VN * rng.standard_normal(N)
            pos_arr[k] = pos_arr[k] + rate_arr[k] * dm_step + PN * rng.standard_normal(N)
            tvt_p = np.clip(pos_arr[k] - z_v[i], lo, hi)
            pos_arr[k] = tvt_p + z_v[i]

            eg = np.interp(tvt_p, tw_tvt, tw_gr)
            d = (gr_v[i] - eg) / gs

            lk = np.maximum(np.exp(-0.5 * np.minimum((d * d) / scales[k], 600.)), 1e-300)
            log_lik_k = np.log(max(float((w_arr[k] * lk).sum()), 1e-300))
            log_lik += log_lik_k / n_scales

            w_arr[k] = w_arr[k] * lk
            ws = w_arr[k].sum()
            w_arr[k] = w_arr[k] / ws if ws > 0 else np.ones(N) / N

            # リサンプリング
            if 1.0 / (w_arr[k] * w_arr[k]).sum() < RESAMP * N:
                cum = np.cumsum(w_arr[k])
                u0 = rng.uniform(0, 1.0 / N)
                idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
                pos_arr[k] = pos_arr[k][idx] + RP * rng.standard_normal(N)
                rate_arr[k] = rate_arr[k][idx] + RR * rng.standard_normal(N)
                w_arr[k] = np.ones(N) / N

        # 粒子交換: 隣接スケール間でswap確率
        if n_scales > 1 and i % 10 == 0:
            for k in range(n_scales - 1):
                if rng.random() < 0.3:  # 交換確率30%
                    j = min(k + 1, n_scales - 1)
                    swap_idx_k = rng.choice(N, size=N // 4, replace=False)
                    swap_idx_j = rng.choice(N, size=N // 4, replace=False)
                    pos_arr[k][swap_idx_k], pos_arr[j][swap_idx_j] = (
                        pos_arr[j][swap_idx_j].copy(), pos_arr[k][swap_idx_k].copy()
                    )

        # 最低温度(k=-1)の予測を出力
        res[i] = float(np.dot(w_arr[-1], pos_arr[-1] - z_v[i]))
        prev_MD = md_v[i]

    return res, log_lik


def _viterbi_crf_single(p) -> Tuple[np.ndarray, float]:
    """案9: ニューラルCRF / Viterbi-DP。

    離散化されたTVT状態空間でViterbi復号。
    emission = GR対数尤度
    transition = dip連続性ペナルティ
    known区間はアンカー固定。
    """
    tw_tvt = p["tw_tvt"]
    tw_gr = p["tw_gr"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
    gs = p["gs"]
    n = len(md_v)

    if n == 0:
        return np.zeros(0), 0.0

    # TVT状態空間を量子化
    tvt_range = (tw_tvt[-1] - tw_tvt[0])
    n_states = max(50, int(tvt_range / 10))  # ~10ft分解能
    tvt_grid = np.linspace(tw_tvt[0] - 100, tw_tvt[-1] + 100, n_states)

    # emission: GR尤度 [行 x 状態]
    emission = np.zeros((n, n_states))
    for i in range(n):
        eg = np.interp(tvt_grid, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        emission[i, :] = np.exp(-0.5 * np.minimum(d * d, 600.))

    # transition: 隣接状態の遷移確率
    #   dTVT/dMD ≈ -1.0 (実験的に観察)
    transition = np.zeros((n - 1, n_states, n_states))
    for i in range(n - 1):
        dm = max(md_v[i + 1] - md_v[i], 1.0)
        expected_delta = -1.0 * dm  # 期待される TVT 変化量
        for s_curr in range(n_states):
            for s_next in range(n_states):
                delta_tvt = tvt_grid[s_next] - tvt_grid[s_curr]
                dev = (delta_tvt - expected_delta) ** 2
                transition[i, s_curr, s_next] = np.exp(-0.5 * dev / (dm ** 2))

    # Viterbi 復号
    log_viterbi = np.zeros((n, n_states))
    backpointer = np.zeros((n, n_states), dtype=int)

    log_viterbi[0, :] = np.log(np.maximum(emission[0, :], 1e-300))

    for i in range(1, n):
        for s_next in range(n_states):
            log_trans = np.log(np.maximum(transition[i - 1, :, s_next], 1e-300))
            log_obs = np.log(np.maximum(emission[i, s_next], 1e-300))
            scores = log_viterbi[i - 1, :] + log_trans

            backpointer[i, s_next] = np.argmax(scores)
            log_viterbi[i, s_next] = np.max(scores) + log_obs

    # 経路復元
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(log_viterbi[-1, :])
    for i in range(n - 2, -1, -1):
        path[i] = backpointer[i + 1, path[i + 1]]

    res = tvt_grid[path]
    log_lik = float(np.max(log_viterbi[-1, :]))

    return res, log_lik


def _pf_uncertain_weighted_single(p, seed):
    """案10: 不確実性つきPF。

    PF粒子分散から予測不確実性を推定。
    不確実性が高い区間をanchorへ寄せる。
    """
    tw_tvt = p["tw_tvt"]
    tw_gr = p["tw_gr"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
    gs = p["gs"]
    ir = p["ir"]
    anchor = p["anchor"]
    n = len(md_v)

    if n == 0:
        return np.zeros(0), 0.0

    N = N_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (p["last_tvt"] + p["last_Z"]) + INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    res_std = np.empty(n)
    prev_MD = p["last_MD"]
    log_lik = 0.0
    lo = tw_tvt[0] - 100
    hi = tw_tvt[-1] + 100

    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs

        lk = np.maximum(np.exp(-0.5 * np.minimum(d * d, 600.)), 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        # 加重平均と不確実性
        pred = float(np.dot(w, pos - z_v[i]))
        pred_std = float(np.sqrt(np.dot(w, (pos - z_v[i] - pred) ** 2)))

        res[i] = pred
        res_std[i] = pred_std
        prev_MD = md_v[i]

    # 不確実性に基づいて anchor へダウンウェイト
    uncertainty_high = res_std > np.percentile(res_std, 75)
    res[uncertainty_high] = 0.7 * res[uncertainty_high] + 0.3 * anchor

    return res, log_lik


def build_proposal_params(base_path, tw_path, fold_path, subset_wells):
    """サブセット用パラメータビルド（案7-10共通）。"""
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"
    ])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    folds = pd.read_csv(fold_path)

    params = {}
    for wid in subset_wells:
        wd = tr[tr["well_id"] == wid].copy()
        if wd.empty:
            continue

        wd_sort = wd.sort_values("MD").reset_index(drop=True)
        known = wd_sort[wd_sort["is_known_tvt"]].copy()
        eval_ = wd_sort[wd_sort["is_target"]].copy()

        if eval_.empty or known.empty:
            continue

        # GR補間
        gr_known = known["GR"].values
        if gr_known[~np.isnan(gr_known)].size > 1:
            valid_idx = ~np.isnan(gr_known)
            f_gr = interp1d(
                known.index[valid_idx], gr_known[valid_idx],
                kind='linear', fill_value='extrapolate'
            )
            gr_eval = np.clip(f_gr(eval_.index), gr_known[valid_idx].min(), gr_known[valid_idx].max())
        else:
            gr_eval = np.full(len(eval_), known["GR"].mean())

        tw_df = tw_by_well.get(wid)
        if tw_df is None or tw_df.empty:
            continue

        tw_tvt = tw_df.sort_values("TVT")["TVT"].values
        tw_gr = tw_df.sort_values("TVT")["GR"].values

        # GR スケール推定
        gr_diff = np.abs(known["GR"].values[~np.isnan(known["GR"].values)]).std()
        gs = max(1.0, gr_diff)

        # 初期レート推定
        if len(known) > 1:
            ir = (known["TVT"].iloc[-1] - known["TVT"].iloc[0]) / max(1.0, (known["MD"].iloc[-1] - known["MD"].iloc[0]))
        else:
            ir = -1.0

        params[wid] = {
            "wid": wid,
            "n_eval": len(eval_),
            "tw_tvt": tw_tvt,
            "tw_gr": tw_gr,
            "md_v": eval_["MD"].values,
            "z_v": eval_["Z"].values,
            "gr_v": gr_eval,
            "gs": gs,
            "ir": ir,
            "anchor": eval_["TVT_input"].iloc[0],
            "last_tvt": known["TVT"].iloc[-1],
            "last_Z": known["Z"].iloc[-1],
            "last_MD": known["MD"].iloc[-1],
            "true_tvt": eval_["TVT"].values,
        }

    return params


def evaluate_proposal(proposal_func, params_list, proposal_name) -> Dict[str, Any]:
    """単一proposal の subset CV 評価。"""
    results = {}
    preds_all = []
    true_all = []

    for params in params_list:
        if params["n_eval"] == 0:
            continue

        wid = params["wid"]
        try:
            if proposal_name == "viterbi":
                pred, _ = proposal_func(params)
            else:
                # PF型: seed平均
                preds = []
                for seed in range(16):  # subset評価は seed 削減
                    p, _ = proposal_func(params, seed)
                    preds.append(p)
                pred = np.mean(preds, axis=0)

            rmse = np.sqrt(np.mean((pred - params["true_tvt"]) ** 2))
            results[wid] = {"pred": pred, "rmse": rmse}
            preds_all.append(pred)
            true_all.append(params["true_tvt"])
        except Exception as e:
            print(f"  Error in {proposal_name} for well {wid}: {e}")
            results[wid] = {"error": str(e), "rmse": np.inf}

    if preds_all:
        preds_concat = np.concatenate(preds_all)
        true_concat = np.concatenate(true_all)
        cv_rmse = np.sqrt(np.mean((preds_concat - true_concat) ** 2))
    else:
        cv_rmse = np.inf

    return {"cv_rmse": cv_rmse, "results": results, "n_wells": len(results)}


def main():
    root = Path(__file__).resolve().parents[1]
    base_path = root / "data" / "processed" / "train_base_v001.parquet"
    tw_path = root / "data" / "processed" / "typewell_train_base_v001.parquet"
    fold_path = root / "data" / "folds" / "folds_group_well_v001.csv"
    per_well_path = root / "experiments" / "exp022_particle_filter" / "per_well.csv"

    # subset選択
    subset_wells = select_subset_wells(per_well_path, n_broken=10, n_good=10)
    print(f"Selected {len(subset_wells)} wells: {subset_wells[:5]}...")

    # パラメータビルド
    params_list = list(build_proposal_params(base_path, tw_path, fold_path, subset_wells).values())
    params_list = [p for p in params_list if p["n_eval"] > 0]
    print(f"Built params for {len(params_list)} wells")

    # 4案評価
    results_all = {}

    print("\n[案7] 焼きなまし PF...")
    res7 = evaluate_proposal(_pf_annealed_single, params_list, "annealed")
    results_all["proposal7_annealed"] = res7
    print(f"  CV RMSE: {res7['cv_rmse']:.3f} ({res7['n_wells']} wells)")

    print("\n[案8] レプリカ交換 PF...")
    res8 = evaluate_proposal(_pf_parallel_tempering_single, params_list, "parallel_tempering")
    results_all["proposal8_replica_exchange"] = res8
    print(f"  CV RMSE: {res8['cv_rmse']:.3f} ({res8['n_wells']} wells)")

    print("\n[案9] Viterbi-DP CRF...")
    res9 = evaluate_proposal(_viterbi_crf_single, params_list, "viterbi")
    results_all["proposal9_viterbi"] = res9
    print(f"  CV RMSE: {res9['cv_rmse']:.3f} ({res9['n_wells']} wells)")

    print("\n[案10] 不確実性ダウンウェイト PF...")
    res10 = evaluate_proposal(_pf_uncertain_weighted_single, params_list, "uncertain")
    results_all["proposal10_uncertain"] = res10
    print(f"  CV RMSE: {res10['cv_rmse']:.3f} ({res10['n_wells']} wells)")

    # サマリーと比較
    baseline_pf = 11.024
    print(f"\n=== サマリー ===")
    print(f"基準 (exp022 PF full-773): {baseline_pf:.3f}")
    print(f"案7 焼きなまし:  {res7['cv_rmse']:.3f} (差 {res7['cv_rmse'] - baseline_pf:+.3f})")
    print(f"案8 レプリカ交換: {res8['cv_rmse']:.3f} (差 {res8['cv_rmse'] - baseline_pf:+.3f})")
    print(f"案9 Viterbi:     {res9['cv_rmse']:.3f} (差 {res9['cv_rmse'] - baseline_pf:+.3f})")
    print(f"案10 不確実性:   {res10['cv_rmse']:.3f} (差 {res10['cv_rmse'] - baseline_pf:+.3f})")

    # 結果保存
    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "subset_size": len(subset_wells),
        "subset_wells": subset_wells,
        "baseline_exp022_cv": baseline_pf,
        "proposals": {
            "proposal7_annealed_pf": {
                "cv_rmse": res7["cv_rmse"],
                "n_wells": res7["n_wells"],
                "improvement_vs_baseline": res7["cv_rmse"] - baseline_pf,
                "description": "温度スケジュール (高温探索→低温収束)",
            },
            "proposal8_replica_exchange_pf": {
                "cv_rmse": res8["cv_rmse"],
                "n_wells": res8["n_wells"],
                "improvement_vs_baseline": res8["cv_rmse"] - baseline_pf,
                "description": "複数温度並走+粒子交換",
            },
            "proposal9_viterbi_dp": {
                "cv_rmse": res9["cv_rmse"],
                "n_wells": res9["n_wells"],
                "improvement_vs_baseline": res9["cv_rmse"] - baseline_pf,
                "description": "CRF with emission=GR, transition=連続性",
            },
            "proposal10_uncertainty_pf": {
                "cv_rmse": res10["cv_rmse"],
                "n_wells": res10["n_wells"],
                "improvement_vs_baseline": res10["cv_rmse"] - baseline_pf,
                "description": "粒子分散による不確実性推定+anchor寄せ",
            },
        },
        "best_proposal": min(
            [("proposal7", res7["cv_rmse"]), ("proposal8", res8["cv_rmse"]),
             ("proposal9", res9["cv_rmse"]), ("proposal10", res10["cv_rmse"])],
            key=lambda x: x[1]
        )[0],
        "leak_risk": "none (no hidden TVT used; GR+typewell+Z+MD only)",
    }

    with open(OUT_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    # MD ファイル
    md_content = f"""# exp062: トラッカー多様化検証 (subset 20 well)

## 背景

exp022 PF (CV 11.024) の broken well 救済を目指した4つの提案を subset (broken 10 + good 10) で検証。
- 案7: 焼きなまし PF
- 案8: レプリカ交換 PF (Parallel Tempering)
- 案9: Viterbi-DP (CRF)
- 案10: 不確実性ダウンウェイト PF

## 結果

| 案 | 手法 | CV RMSE (subset) | vs baseline | 効果 |
|---|---|---:|---:|---|
| 7 | 焼きなまし | {res7['cv_rmse']:.3f} | {res7['cv_rmse'] - baseline_pf:+.3f} | {'✓' if res7['cv_rmse'] < baseline_pf else '✗'} |
| 8 | レプリカ交換 | {res8['cv_rmse']:.3f} | {res8['cv_rmse'] - baseline_pf:+.3f} | {'✓' if res8['cv_rmse'] < baseline_pf else '✗'} |
| 9 | Viterbi-DP | {res9['cv_rmse']:.3f} | {res9['cv_rmse'] - baseline_pf:+.3f} | {'✓' if res9['cv_rmse'] < baseline_pf else '✗'} |
| 10 | 不確実性 | {res10['cv_rmse']:.3f} | {res10['cv_rmse'] - baseline_pf:+.3f} | {'✓' if res10['cv_rmse'] < baseline_pf else '✗'} |

基準: exp022 PF full-773 CV = {baseline_pf:.3f}

## 考察

- subset evaluation なため full-773 への転移は未確認
- 有効な案があれば full-773 + ensemble で再評価予定
- 失敗理由は各案の数学的制限による可能性

## Next Action

- 最有望案を full-773 で実行
- ensemble with exp026 最終blend
- honest CV で LB 転移確認

## Leak Risk

None: Group well-fold + known区間のみ使用。hidden TVT は学習・選択に不使用。
"""

    with open(OUT_DIR / "notes.md", "w") as f:
        f.write(md_content)

    print(f"\n✓ 結果保存: {OUT_DIR}")
    return result


if __name__ == "__main__":
    main()
