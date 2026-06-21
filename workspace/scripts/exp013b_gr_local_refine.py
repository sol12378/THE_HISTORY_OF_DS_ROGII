#!/usr/bin/env python3
"""exp013b: GR local-refinement test (Track 2 Phase 1, follow-up).

exp013 で素朴なGR直接照合(±110窓)は全滅(RMSE 24-46 >> anchor 15.9)を確認。
原因 = hidden TVTのdelta幅±100に対しGRが多価で遠方偽マッチを拾う。

本追試の問い: **狭窓 + 幾何prior中心** なら、GRは予測を局所微修正できるか?
  center ∈ {anchor, exp008予測} の ±w窓内で、GR照合TVT(centerに最も近い解)を採用。
  w を 10/20/30 と振る。blend(GR-match と center) も評価。

これが効けば「GRは幾何の上で微修正に使える」= Approach C に部分GO。
効かなければ「現データ・現手法ではGR-typewellは使えない」= NO-GO確定。

完全leak-free (推定にラベルTVT不使用、exp008はOOF予測)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp013b_gr_local_refine"
OUT_DIR = Path("experiments") / EXP_ID
GRID_STEP = 0.25
WINDOWS = [10.0, 20.0, 30.0]
BLEND_WEIGHTS = [0.3, 0.5, 1.0]  # GR-match weight; 1.0 = pure match


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] GR local-refinement test")

    tr = pd.read_parquet("data/processed/train_base_v001.parquet",
                         columns=["well_id", "row_idx", "GR", "TVT", "TVT_input",
                                  "is_target", "is_known_tvt", "is_gr_missing",
                                  "last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
                             columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: g.sort_values("TVT").drop_duplicates("TVT") for w, g in tw_all.groupby("well_id", sort=False)}

    oof8 = pd.read_csv("experiments/exp008_gr_rolling/oof.csv",
                       usecols=["well_id", "row_idx", "pred_tvt"]).rename(columns={"pred_tvt": "exp008_tvt"})
    oof8_map = oof8.set_index(["well_id", "row_idx"])["exp008_tvt"]

    parts = []
    for wid, g in tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)].groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)].copy()
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])
        tw_g = tw_by_well.get(wid)

        # exp008 center
        idx = list(zip(tgt["well_id"], tgt["row_idx"]))
        e8 = oof8_map.reindex(idx).to_numpy(float)
        e8 = np.where(np.isnan(e8), anchor, e8)

        has_gr = ~tgt["is_gr_missing"].astype(bool).to_numpy()
        gr_obs = tgt["GR"].to_numpy(float)

        out = {"well_id": tgt["well_id"].to_numpy(), "row_idx": tgt["row_idx"].to_numpy(),
               "TVT": tgt["TVT"].to_numpy(float), "anchor": np.full(len(tgt), anchor),
               "exp008": e8}

        if tw_g is None or len(tw_g) < 2:
            for c in ["anchor", "exp008"]:
                for w in WINDOWS:
                    out[f"matchA_{c}_w{int(w)}"] = out[c].copy()
                    out[f"matchE_{c}_w{int(w)}"] = out[c].copy()
            parts.append(pd.DataFrame(out))
            continue

        tw_tvt = tw_g["TVT"].to_numpy(float)
        tw_gr = tw_g["GR"].to_numpy(float)

        # calibration offset (known)
        kv = known[~known["is_gr_missing"].astype(bool)]
        if len(kv) >= 10:
            exp_gr = np.interp(kv["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
            offset = float(np.median(kv["GR"].to_numpy(float) - exp_gr))
        else:
            offset = 0.0
        target_gr = gr_obs - offset

        for c, center in [("anchor", out["anchor"]), ("exp008", out["exp008"])]:
            for w in WINDOWS:
                pred_match = center.copy()  # default = center (and for missing GR)
                for i in np.where(has_gr)[0]:
                    lo = max(center[i] - w, tw_tvt.min())
                    hi = min(center[i] + w, tw_tvt.max())
                    if hi - lo < GRID_STEP:
                        continue
                    grid = np.arange(lo, hi + GRID_STEP, GRID_STEP)
                    ggr = np.interp(grid, tw_tvt, tw_gr)
                    # nearest-to-center tiebreak: minimize |GR diff| + tiny*|tvt-center|
                    cost = np.abs(ggr - target_gr[i]) + 0.01 * np.abs(grid - center[i])
                    pred_match[i] = grid[int(np.argmin(cost))]
                out[f"matchA_{c}_w{int(w)}"] = pred_match
                # smoothed
                out[f"matchE_{c}_w{int(w)}"] = pd.Series(pred_match).rolling(
                    25, min_periods=1, center=True).median().to_numpy()
        parts.append(pd.DataFrame(out))

    preds = pd.concat(parts, ignore_index=True)
    preds.to_csv(OUT_DIR / "row_predictions.csv", index=False)

    res = {"anchor": tvt_rmse(preds["TVT"], preds["anchor"]),
           "exp008": tvt_rmse(preds["TVT"], preds["exp008"])}
    # pure match + blends with center
    for c in ["anchor", "exp008"]:
        for w in WINDOWS:
            mA = preds[f"matchA_{c}_w{int(w)}"].to_numpy(float)
            mE = preds[f"matchE_{c}_w{int(w)}"].to_numpy(float)
            ctr = preds[c].to_numpy(float)
            res[f"matchA_{c}_w{int(w)}"] = tvt_rmse(preds["TVT"], mA)
            res[f"matchE_{c}_w{int(w)}_smooth"] = tvt_rmse(preds["TVT"], mE)
            for bw in BLEND_WEIGHTS:
                blend = bw * mA + (1 - bw) * ctr
                res[f"blend_{c}_w{int(w)}_bw{bw}"] = tvt_rmse(preds["TVT"], blend)

    # best vs exp008
    best_key = min(res, key=res.get)
    print("\n=== RMSE (全target行) ===")
    for k in sorted(res, key=res.get):
        flag = "  <-- BEST" if k == best_key else ("  (exp008)" if k == "exp008" else "")
        print(f"  {k:34s} = {res[k]:.6f}{flag}")

    result = {"exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
              "method": "GR local refinement around geometric prior (leak-free)",
              "grid_step": GRID_STEP, "windows": WINDOWS,
              "rmse": res, "best": {"key": best_key, "rmse": res[best_key]},
              "exp008_ref": res["exp008"], "anchor_ref": res["anchor"],
              "verdict_hint": "GO if any variant < exp008; partial-GO if < anchor; NO-GO if all >= exp008"}
    write_json(OUT_DIR / "result.json", result)
    print(f"\n[{EXP_ID}] 完了 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
