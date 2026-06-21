#!/usr/bin/env python3
"""Smoke test for exp038: test on 5 wells to verify correctness before full run."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

# Import the Viterbi function
import importlib.util
spec = importlib.util.spec_from_file_location("exp038", "scripts/exp038_dp_viterbi.py")
exp038 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp038)

EXP_ID = "exp038_dp_viterbi_smoke"
OUT_DIR = Path("experiments") / EXP_ID

def smoke_test():
    """Run Viterbi on first 5 wells."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp038 SMOKE TEST] 5-well test run...")

    # Load data
    tr = pd.read_parquet("data/processed/train_base_v001.parquet", columns=[
        "well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
                             columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    all_wells = list(sel.groupby("well_id", sort=False).groups.keys())[:5]

    print(f"  Testing wells: {all_wells}")

    # Manual inline test of _viterbi_single on first well
    for wid in all_wells:
        g = tr[tr["well_id"] == wid].sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]

        if len(tgt) == 0:
            continue

        anchor = float(tgt["last_known_TVT"].iloc[0])
        tw_g = tw_by_well.get(wid)
        tgt_mask = g["is_target"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both")

        if tw_g is None or len(tw_g) < 2 or len(known) < 2:
            print(f"    {wid}: no_tw (anchor fallback)")
            continue

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

        wid_ret, pred = exp038._viterbi_single(payload)
        rmse = tvt_rmse(tgt["TVT"].values, pred)
        anchor_rmse = tvt_rmse(tgt["TVT"].values, np.full(len(pred), anchor))
        print(f"    {wid}: n={len(pred):3d}, Viterbi RMSE={rmse:8.4f}, anchor={anchor_rmse:8.4f}")

    print("[exp038 SMOKE TEST] OK ✓")

if __name__ == "__main__":
    smoke_test()
