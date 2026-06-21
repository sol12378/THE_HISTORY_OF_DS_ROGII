#!/usr/bin/env python3
"""exp013: GR-typewell alignment GO/NO-GO signal test (Track 2 Phase 1).

戦略の核心判断: hidden行の観測GRをtypewellのGR-TVTプロファイルに照合して
TVTを直接復元したとき、anchor(delta=0, RMSE=15.91)より良いか?
exp008(13.81)に匹敵/凌駕する信号があれば本丸投資(Approach C)へGO。

**完全leak-free**: 推定にラベルTVTを一切使わない。
  - 観測量: hidden行のGR (入力。予測対象ではない)
  - 参照: typewell GR-TVTプロファイル (別物理井戸の参照曲線)
  - anchor: last_known_TVT (既知)
  - キャリブレーション offset: known区間のみ
→ foldも学習も不要。全773 wellsで直接RMSEを測定できる。

手法:
  各wellで typewell GR(TVT) を構築。known区間で gr_offset を較正。
  各hidden行(GR有)について、anchor±W の窓内の typewell TVTグリッドから
  |tw_GR(tvt) - (obs_GR - offset)| を最小化する TVT を探索。
  曖昧性(GR-TVT多価)は (A)nearest-to-anchor と (B)系列連続性 で解決。
  GR欠損行は anchor にフォールバック。

出力: anchor / variantA / variantA_smooth / variantB(系列) の RMSE、
      tw相関ビン別内訳、anchorに勝つwell割合。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp013_gr_match_go_nogo"
OUT_DIR = Path("experiments") / EXP_ID

WINDOW = 110.0       # anchor ± WINDOW (TVT) を探索範囲に (delta実測 ±104)
GRID_STEP = 0.5      # typewell TVTグリッド刻み
CONT_LAMBDA = 0.15   # 系列連続性ペナルティ (GR単位/TVT単位の重み)
SMOOTH_W = 25        # variantA_smooth の rolling median 窓


def build_tw_grid(tw_g: pd.DataFrame, anchor: float):
    """typewellを anchor±WINDOW のTVTグリッドに補間。(grid_tvt, grid_gr) を返す。"""
    tw_g = tw_g.sort_values("TVT").drop_duplicates("TVT")
    tvt = tw_g["TVT"].to_numpy(float)
    gr = tw_g["GR"].to_numpy(float)
    if len(tvt) < 2:
        return None, None
    lo = max(anchor - WINDOW, tvt.min())
    hi = min(anchor + WINDOW, tvt.max())
    if hi - lo < GRID_STEP:
        return None, None
    grid_tvt = np.arange(lo, hi + GRID_STEP, GRID_STEP)
    grid_gr = np.interp(grid_tvt, tvt, gr)
    return grid_tvt, grid_gr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] GR-typewell GO/NO-GO signal test")

    tr = pd.read_parquet("data/processed/train_base_v001.parquet",
                         columns=["well_id", "row_idx", "GR", "TVT", "TVT_input",
                                  "is_target", "is_known_tvt", "is_gr_missing",
                                  "last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
                             columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: g for w, g in tw_all.groupby("well_id", sort=False)}

    # exp008 OOF (参照比較用)
    oof8 = pd.read_csv("experiments/exp008_gr_rolling/oof.csv",
                       usecols=["well_id", "row_idx", "pred_tvt"]).rename(
        columns={"pred_tvt": "exp008_tvt"})

    well_rows = []
    all_parts = []   # per-row predictions for global RMSE

    for wid, g in tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)].groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)].copy()
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])

        tw_g = tw_by_well.get(wid)
        if tw_g is None or len(tw_g) < 2:
            # no typewell → all anchor
            tgt["pred_A"] = anchor
            tgt["pred_As"] = anchor
            tgt["pred_B"] = anchor
            tgt["tw_corr"] = np.nan
            all_parts.append(tgt[["well_id", "row_idx", "TVT", "last_known_TVT",
                                  "pred_A", "pred_As", "pred_B", "tw_corr"]])
            continue

        grid_tvt, grid_gr = build_tw_grid(tw_g, anchor)
        if grid_tvt is None:
            tgt["pred_A"] = anchor; tgt["pred_As"] = anchor; tgt["pred_B"] = anchor
            tgt["tw_corr"] = np.nan
            all_parts.append(tgt[["well_id", "row_idx", "TVT", "last_known_TVT",
                                  "pred_A", "pred_As", "pred_B", "tw_corr"]])
            continue

        # ── calibration offset from known portion ──
        tw_sorted = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tw_tvt_full = tw_sorted["TVT"].to_numpy(float)
        tw_gr_full = tw_sorted["GR"].to_numpy(float)
        k_valid = known[~known["is_gr_missing"].astype(bool)]
        if len(k_valid) >= 10:
            exp_gr = np.interp(k_valid["TVT_input"].to_numpy(float), tw_tvt_full, tw_gr_full)
            offset = float(np.median(k_valid["GR"].to_numpy(float) - exp_gr))
            tw_corr = float(pd.Series(k_valid["GR"].to_numpy(float)).corr(pd.Series(exp_gr)))
        else:
            offset = 0.0
            tw_corr = np.nan

        # ── per-row GR match ──
        gr_obs = tgt["GR"].to_numpy(float)
        has_gr = ~tgt["is_gr_missing"].astype(bool).to_numpy()
        target_gr = gr_obs - offset  # calibrated to typewell scale

        # variant A: nearest GR match in window (vectorized)
        # diff matrix (n_rows x n_grid) only for rows with GR
        pred_A = np.full(len(tgt), anchor, dtype=float)
        anchor_idx = int(np.argmin(np.abs(grid_tvt - anchor)))
        idx_rows = np.where(has_gr)[0]
        if len(idx_rows) > 0:
            tg = target_gr[idx_rows]
            # |grid_gr - tg|: shape (len(idx_rows), n_grid)
            diff = np.abs(grid_gr[None, :] - tg[:, None])
            best = np.argmin(diff, axis=1)
            pred_A[idx_rows] = grid_tvt[best]

        # variant B: sequential continuity DP-ish (greedy)
        # pick tvt minimizing |gr - target| + lambda*|tvt - prev|
        pred_B = np.full(len(tgt), anchor, dtype=float)
        prev = anchor
        gtvt = grid_tvt
        for i in range(len(tgt)):
            if has_gr[i]:
                cost = np.abs(grid_gr - target_gr[i]) + CONT_LAMBDA * np.abs(gtvt - prev)
                j = int(np.argmin(cost))
                prev = gtvt[j]
            pred_B[i] = prev

        # variant A_smooth: rolling median of pred_A
        pred_As = pd.Series(pred_A).rolling(SMOOTH_W, min_periods=1, center=True).median().to_numpy()
        # missing-GR rows keep anchor in A; smoothing may bleed — restore anchor where no GR neighbor
        pred_As[~has_gr] = pred_As[~has_gr]  # keep smoothed (continuity OK)

        tgt["pred_A"] = pred_A
        tgt["pred_As"] = pred_As
        tgt["pred_B"] = pred_B
        tgt["tw_corr"] = tw_corr
        all_parts.append(tgt[["well_id", "row_idx", "TVT", "last_known_TVT",
                              "pred_A", "pred_As", "pred_B", "tw_corr"]])

        # per-well RMSE
        rA = tvt_rmse(tgt["TVT"], tgt["pred_A"])
        rB = tvt_rmse(tgt["TVT"], tgt["pred_B"])
        rAs = tvt_rmse(tgt["TVT"], tgt["pred_As"])
        ranc = tvt_rmse(tgt["TVT"], tgt["last_known_TVT"])
        well_rows.append({"well_id": wid, "n": len(tgt), "tw_corr": tw_corr,
                          "anchor_rmse": ranc, "A_rmse": rA, "As_rmse": rAs, "B_rmse": rB,
                          "frac_gr": float(has_gr.mean())})

    preds = pd.concat(all_parts, ignore_index=True)
    preds.to_csv(OUT_DIR / "row_predictions.csv", index=False)
    wells = pd.DataFrame(well_rows)
    wells.to_csv(OUT_DIR / "per_well.csv", index=False)

    # merge exp008
    preds = preds.merge(oof8, on=["well_id", "row_idx"], how="left")

    # ── global RMSE ──
    res = {
        "anchor": tvt_rmse(preds["TVT"], preds["last_known_TVT"]),
        "variantA_nearest": tvt_rmse(preds["TVT"], preds["pred_A"]),
        "variantA_smooth": tvt_rmse(preds["TVT"], preds["pred_As"]),
        "variantB_sequential": tvt_rmse(preds["TVT"], preds["pred_B"]),
        "exp008_ref": tvt_rmse(preds.dropna(subset=["exp008_tvt"])["TVT"],
                               preds.dropna(subset=["exp008_tvt"])["exp008_tvt"]),
    }
    # GR-rows-only ceiling (どこまでGRが効くか)
    gr_only = preds[preds["pred_A"] != preds["last_known_TVT"]]  # rough proxy; use has_gr better
    # better: recompute on rows where GR present via row_predictions has implicit; skip, use blend below

    # ── blend: GR-match where reliable, else anchor (corr threshold) ──
    # well-level: use variantB if tw_corr high, else anchor
    wcorr = wells.set_index("well_id")["tw_corr"]
    preds["tw_corr_w"] = preds["well_id"].map(wcorr)
    for thr in [0.0, 0.3, 0.5, 0.7]:
        use = preds["tw_corr_w"].fillna(-1) >= thr
        blendB = np.where(use, preds["pred_B"], preds["last_known_TVT"])
        res[f"blendB_corr>={thr}"] = tvt_rmse(preds["TVT"], blendB)

    # ── corr-bin breakdown (per-well mean RMSE) ──
    wells["corr_bin"] = pd.cut(wells["tw_corr"], [-2, 0.3, 0.5, 0.7, 0.85, 1.01],
                               labels=["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.85", ">0.85"])
    binstat = wells.groupby("corr_bin", observed=True).agg(
        n_wells=("well_id", "size"),
        anchor=("anchor_rmse", "mean"),
        A=("A_rmse", "mean"),
        As=("As_rmse", "mean"),
        B=("B_rmse", "mean"),
    ).reset_index()
    binstat.to_csv(OUT_DIR / "corr_bin_breakdown.csv", index=False)

    n_B_wins = int((wells["B_rmse"] < wells["anchor_rmse"]).sum())
    n_As_wins = int((wells["As_rmse"] < wells["anchor_rmse"]).sum())

    print("\n=== GLOBAL RMSE (全target行) ===")
    for k, v in res.items():
        print(f"  {k:24s} = {v:.6f}")
    print(f"\n  B が anchor に勝つ well: {n_B_wins}/{len(wells)}")
    print(f"  As が anchor に勝つ well: {n_As_wins}/{len(wells)}")
    print("\n=== corr-bin別 (per-well平均RMSE) ===")
    print(binstat.to_string(index=False))

    result = {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "method": "leak-free GR-typewell matching (no label, no fold)",
        "window": WINDOW, "grid_step": GRID_STEP, "cont_lambda": CONT_LAMBDA,
        "smooth_w": SMOOTH_W,
        "global_rmse": res,
        "n_wells": int(len(wells)),
        "n_B_beats_anchor": n_B_wins,
        "n_As_beats_anchor": n_As_wins,
        "corr_bin_breakdown": binstat.to_dict(orient="records"),
        "leak_risk": "none (no label used in estimation)",
        "go_nogo_note": (
            "GO if any GR-match variant or blend approaches exp008 (13.81) or "
            "clearly beats anchor (15.91) in high-corr wells. "
            "NO-GO/redesign if GR-match >= anchor everywhere."
        ),
    }
    write_json(OUT_DIR / "result.json", result)
    print(f"\n[{EXP_ID}] 完了 -> {OUT_DIR}")


if __name__ == "__main__":
    main()
