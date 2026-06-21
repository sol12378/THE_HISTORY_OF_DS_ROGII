#!/usr/bin/env python3
"""Debug version: run Viterbi sequentially (no multiprocessing) on first well."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse

# Constants
N_BINS = 50
BIN_RANGE = 50.0
TRANSITION_SIGMA = 1.0


def _viterbi_single_debug(p):
    """Viterbi for debugging."""
    wid = p["wid"]
    n = int(p["n_eval"])

    print(f"  [DEBUG] {wid}: n_eval={n}")

    if n == 0:
        return wid, np.zeros(0)

    if p.get("no_tw", False):
        print(f"    -> no_tw, returning anchor")
        return wid, np.full(n, p["anchor"])

    anchor = p["anchor"]
    tw_tvt = p["tw_tvt"]
    tw_gr = p["tw_gr"]
    md_v = p["md_v"]
    z_v = p["z_v"]
    gr_v = p["gr_v"]
    gs = p["gs"]
    prior_dtvt_per_md = p["prior_dtvt_per_md"]

    print(f"    tw_tvt range: [{tw_tvt.min():.2f}, {tw_tvt.max():.2f}]")
    print(f"    md_v range: [{md_v.min():.2f}, {md_v.max():.2f}]")
    print(f"    gs={gs:.4f}, prior_dtvt_per_md={prior_dtvt_per_md:.6f}")

    bin_center = np.arange(N_BINS) - N_BINS // 2

    INF = 1e20
    cost = np.full((n, N_BINS), INF, dtype=np.float64)
    parent = np.full((n, N_BINS), -1, dtype=np.int32)

    init_state = N_BINS // 2
    print(f"    init_state={init_state} (TVT={anchor})")

    # Initialize first step
    for state in range(N_BINS):
        tvt_at_state = anchor + bin_center[state]
        tw_gr_interp = np.interp(tvt_at_state, tw_tvt, tw_gr)
        obs_gr = gr_v[0]
        residual = obs_gr - tw_gr_interp
        emission_cost = 0.5 * (residual / gs) ** 2

        if state == init_state:
            cost[0, state] = emission_cost
        else:
            start_penalty = 0.5 * ((bin_center[state]) / TRANSITION_SIGMA) ** 2
            cost[0, state] = start_penalty + emission_cost

    print(f"    Step 0 cost range: [{cost[0].min():.4f}, {cost[0].max():.4f}]")

    # DP forward pass
    for md_idx in range(1, n):
        if md_idx % 500 == 0:
            print(f"      md_idx={md_idx}/{n}")

        dm = md_v[md_idx] - md_v[md_idx - 1]
        if dm <= 0:
            dm = 1.0

        expected_dtvt = prior_dtvt_per_md * dm

        for curr_state in range(N_BINS):
            tvt_at_curr = anchor + bin_center[curr_state]
            tw_gr_interp = np.interp(tvt_at_curr, tw_tvt, tw_gr)
            obs_gr = gr_v[md_idx]
            residual = obs_gr - tw_gr_interp
            emission_cost = 0.5 * (residual / gs) ** 2

            for prev_state in range(N_BINS):
                if cost[md_idx - 1, prev_state] >= INF:
                    continue

                tvt_at_prev = anchor + bin_center[prev_state]
                dtvt_actual = tvt_at_curr - tvt_at_prev

                transition_cost = 0.5 * ((dtvt_actual - expected_dtvt) / TRANSITION_SIGMA) ** 2
                total_cost = cost[md_idx - 1, prev_state] + transition_cost + emission_cost

                if total_cost < cost[md_idx, curr_state]:
                    cost[md_idx, curr_state] = total_cost
                    parent[md_idx, curr_state] = prev_state

        step_cost_min = cost[md_idx].min()
        if step_cost_min >= INF:
            print(f"      WARNING: step {md_idx} all costs = INF")

    print(f"    Final cost range: [{cost[-1].min():.4f}, {cost[-1].max():.4f}]")

    # Backtrack
    final_state = np.argmin(cost[-1, :])
    print(f"    final_state={final_state} (TVT={anchor + bin_center[final_state]:.2f})")

    path = np.zeros(n, dtype=np.int32)
    path[-1] = final_state
    for md_idx in range(n - 2, -1, -1):
        prev_state = parent[md_idx + 1, path[md_idx + 1]]
        if prev_state < 0:
            path[md_idx] = path[md_idx + 1]
        else:
            path[md_idx] = prev_state

    pred_tvt = anchor + bin_center[path].astype(np.float64)
    print(f"    pred_tvt range: [{pred_tvt.min():.2f}, {pred_tvt.max():.2f}]")

    return wid, pred_tvt


def main():
    print("[DEBUG] Loading data...")
    tr = pd.read_parquet("data/processed/train_base_v001.parquet", columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
                             columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]

    print("[DEBUG] Processing first well...")
    all_wells = list(sel.groupby("well_id", sort=False).groups.keys())
    wid = all_wells[0]

    g = tr[tr["well_id"] == wid].sort_values("row_idx")
    known = g[g["is_known_tvt"].astype(bool)]
    tgt = g[g["is_target"].astype(bool)]

    anchor = float(tgt["last_known_TVT"].iloc[0])
    tw_g = tw_by_well[wid]
    tgt_mask = g["is_target"].astype(bool).to_numpy()
    gr_full = g["GR"].interpolate(limit_direction="both")

    tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
    gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    gr_v = gr_full[tgt_mask]

    k_tvt = known["TVT_input"].to_numpy(float)
    k_gr = known["GR"].fillna(0).to_numpy(float)
    tw_at_k = np.interp(k_tvt, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(k_gr - tw_at_k), 10., 60.))

    tail = known.tail(30)
    dt = np.diff(tail["TVT_input"].to_numpy(float))
    dz = np.diff(tail["Z"].to_numpy(float))
    dm = np.diff(tail["MD"].to_numpy(float))
    mm = dm > 0
    if mm.sum() >= 3:
        prior_dtvt_per_md = float(np.median((dt + dz)[mm] / dm[mm]))
    else:
        prior_dtvt_per_md = 0.0

    payload = {
        "wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
        "tw_tvt": tw_tvt, "tw_gr": tw_gr,
        "md_v": tgt["MD"].to_numpy(float),
        "z_v": tgt["Z"].to_numpy(float),
        "gr_v": gr_v,
        "gs": gs,
        "prior_dtvt_per_md": prior_dtvt_per_md,
        "anchor": anchor,
    }

    print(f"[DEBUG] Running Viterbi...")
    wid_ret, pred = _viterbi_single_debug(payload)

    rmse = tvt_rmse(tgt["TVT"].values, pred)
    anchor_rmse = tvt_rmse(tgt["TVT"].values, np.full(len(pred), anchor))
    print(f"\n[DEBUG] Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Anchor RMSE: {anchor_rmse:.6f}")
    print(f"  Improvement: {anchor_rmse - rmse:+.6f}")


if __name__ == "__main__":
    main()
