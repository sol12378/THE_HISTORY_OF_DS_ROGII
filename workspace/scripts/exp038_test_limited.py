#!/usr/bin/env python3
"""Test exp038 on first 10 wells to debug hangs."""

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

# Import the functions from exp038
import importlib.util
spec = importlib.util.spec_from_file_location("exp038", "scripts/exp038_dp_viterbi.py")
exp038 = importlib.util.module_from_spec(spec)
sys.modules["exp038"] = exp038
spec.loader.exec_module(exp038)

EXP_ID = "exp038_test_10wells"
OUT_DIR = Path("experiments") / EXP_ID

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{EXP_ID}] Testing on first 10 wells...")
    print(f"  Current time: {time.strftime('%H:%M:%S')}")

    print("Building payloads (train)...", flush=True)
    payloads, out = exp038.build("data/processed/train_base_v001.parquet",
                                  "data/processed/typewell_train_base_v001.parquet")
    print(f"  Total payloads: {len(payloads)}")
    payloads = payloads[:10]
    out = out[out["well_id"].isin([p["wid"] for p in payloads])]
    print(f"  Limited to: {len(payloads)} payloads, {len(out)} rows")

    print("Running Viterbi (sequential)...", flush=True)
    t0 = time.time()
    pred_by_wid = {}

    for k, p in enumerate(payloads):
        print(f"  {k+1}/{len(payloads)} ...", flush=True)
        wid, pred = exp038._viterbi_single(p)
        pred_by_wid[wid] = pred
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s total", flush=True)

    print("Assembling predictions...", flush=True)
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        n = len(g)
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]

    out = out.copy()
    out["pred_tvt"] = pred_col

    cv = tvt_rmse(out["TVT"], out["pred_tvt"])
    anc = tvt_rmse(out["TVT"], out["last_known_TVT"])

    print(f"\n[{EXP_ID}] Results (10 wells):")
    print(f"  CV: {cv:.6f}")
    print(f"  Anchor: {anc:.6f}")
    print(f"  Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
