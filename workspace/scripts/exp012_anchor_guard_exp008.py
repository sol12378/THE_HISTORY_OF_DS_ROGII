#!/usr/bin/env python3
"""exp012: Anchor guard post-processing on top of exp008 (best model).

exp006 で exp003(CV15.05) に anchor guard を適用し +0.330 改善した。
しかし exp007b(CV13.85) では双方向guardが逆効果だった。
本実験は best model exp008(CV13.808) に、exp006 と同じ一方向guardを適用し、
さらに閾値を exp008 OOF 上で再評価して、強モデルでもまだ
「長尺well / delta暴走well」で anchor に救われる余地があるかを検証する。

設計方針:
  - exp006 と同じ feature-based guard (hidden_length, final_delta_z, mean_pred_delta)
  - alpha は exp008 OOF 上で grid search (過学習を避け粗いグリッド)
  - 閾値そのものも複数候補で sweep し、最良 OOF を選ぶ
  - guard 条件は test で計算可能な特徴量のみ (leak-free)

注意: test 3 wells は全て hidden < 8000 かつ delta穏当 → public LB は変わらない見込み。
これは CV(773 wells)/private 向けの底上げ。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp012_anchor_guard_exp008"
MODEL_EXP = "exp008_gr_rolling"
ANCHOR_EXP = "exp001_anchor_baseline"

OUT_DIR = Path("experiments") / EXP_ID
TRAIN_BASE = Path("data/processed/train_base_v001.parquet")
TEST_BASE = Path("data/processed/test_base_v001.parquet")
SAMPLE_SUB = Path("data/raw/sample_submission.csv")

# guard 閾値の sweep 候補
FULL_ANCHOR_HIDDEN_GRID = [6000, 7000, 8000, 10000, 1e9]   # 1e9 = 無効
DELTA_EXTREME_GRID = [20.0, 30.0, 50.0, 1e9]               # 1e9 = 無効
# strong/mild guard (alpha blend)
STRONG_DELTA_Z_THR = -100.0
STRONG_HIDDEN_THR = 4000
MILD_DELTA_Z_THR = -50.0
MILD_HIDDEN_THR = 4000
ALPHA_GRID = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]  # LGB weight (1.0 = no guard)


def load_oof() -> pd.DataFrame:
    model = pd.read_csv(
        OUT_DIR.parent / MODEL_EXP / "oof.csv",
        usecols=["well_id", "row_idx", "fold", "TVT",
                 "last_known_TVT", "pred_delta", "pred_tvt"],
    ).rename(columns={"pred_tvt": "lgbm_pred_tvt"})
    anchor = pd.read_csv(
        OUT_DIR.parent / ANCHOR_EXP / "oof.csv",
        usecols=["well_id", "row_idx", "pred_tvt"],
    ).rename(columns={"pred_tvt": "anchor_pred_tvt"})
    df = model.merge(anchor, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    assert not df["anchor_pred_tvt"].isna().any(), "anchor OOF missing"
    return df


def compute_well_features(parquet_path: pd.DataFrame) -> pd.DataFrame:
    cols = ["well_id", "row_idx", "is_target", "hidden_length", "delta_Z_from_PS"]
    df = pd.read_parquet(parquet_path, columns=cols)
    t = df.loc[df["is_target"].astype(bool)].copy()
    wf = (
        t.sort_values(["well_id", "row_idx"])
        .groupby("well_id", as_index=False)
        .agg(hidden_length=("hidden_length", "first"),
             final_delta_z=("delta_Z_from_PS", "last"))
    )
    return wf


def per_well_anchor_vs_lgbm(oof: pd.DataFrame) -> pd.DataFrame:
    """各 well で lgbm RMSE と anchor RMSE を比較。"""
    rows = []
    for wid, g in oof.groupby("well_id"):
        r_lgb = tvt_rmse(g["TVT"], g["lgbm_pred_tvt"])
        r_anc = tvt_rmse(g["TVT"], g["anchor_pred_tvt"])
        rows.append({
            "well_id": wid,
            "n_rows": len(g),
            "lgbm_rmse": r_lgb,
            "anchor_rmse": r_anc,
            "anchor_wins_by": r_anc - r_lgb,  # 負 = anchor が勝つ
            "mean_pred_delta": float(g["pred_delta"].mean()),
        })
    return pd.DataFrame(rows)


def assign_rules(wf: pd.DataFrame, full_hidden_thr: float, delta_extreme_thr: float) -> pd.Series:
    rule = pd.Series("no_guard", index=wf.index)
    mild = wf["final_delta_z"].lt(MILD_DELTA_Z_THR) & wf["hidden_length"].gt(MILD_HIDDEN_THR)
    strong = wf["final_delta_z"].lt(STRONG_DELTA_Z_THR) & wf["hidden_length"].gt(STRONG_HIDDEN_THR)
    dext = wf["mean_pred_delta"].abs().gt(delta_extreme_thr)
    full = wf["hidden_length"].gt(full_hidden_thr)
    rule[mild] = "mild_guard"
    rule[strong] = "strong_guard"
    rule[dext] = "delta_extreme"
    rule[full] = "full_anchor"
    return rule


def blend_rmse(oof: pd.DataFrame, rule_map: dict, alphas: dict) -> float:
    a = oof["well_id"].map(lambda w: alphas[rule_map[w]]).to_numpy(float)
    blended = a * oof["lgbm_pred_tvt"].to_numpy(float) + (1 - a) * oof["anchor_pred_tvt"].to_numpy(float)
    return tvt_rmse(oof["TVT"], blended)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] anchor guard on best model {MODEL_EXP}")

    oof = load_oof()
    base_rmse = tvt_rmse(oof["TVT"], oof["lgbm_pred_tvt"])
    anchor_rmse = tvt_rmse(oof["TVT"], oof["anchor_pred_tvt"])
    print(f"  exp008 OOF RMSE  = {base_rmse:.6f}")
    print(f"  anchor OOF RMSE  = {anchor_rmse:.6f}")

    # ── per-well anchor vs lgbm diagnostic ──
    pw = per_well_anchor_vs_lgbm(oof)
    pw.to_csv(OUT_DIR / "per_well_anchor_vs_lgbm.csv", index=False)
    n_anchor_wins = int((pw["anchor_wins_by"] > 0).sum())
    n_anchor_big = int((pw["anchor_wins_by"] > 5).sum())
    print(f"  anchor が勝つ well: {n_anchor_wins}/{len(pw)} "
          f"(うち +5以上: {n_anchor_big})")

    # ── train guard features ──
    wf = compute_well_features(TRAIN_BASE)
    wf = wf.merge(pw[["well_id", "mean_pred_delta"]], on="well_id", how="left")

    # ── sweep thresholds × alpha ──
    best = {"rmse": base_rmse, "full_hidden_thr": 1e9, "delta_extreme_thr": 1e9,
            "alpha_strong": 1.0, "alpha_mild": 1.0}
    for fh in FULL_ANCHOR_HIDDEN_GRID:
        for de in DELTA_EXTREME_GRID:
            rule = assign_rules(wf, fh, de)
            rule_map = dict(zip(wf["well_id"], rule))
            # alpha grid search per rule group (strong, mild). full/delta = 0.0 fixed.
            for a_strong in ALPHA_GRID:
                for a_mild in ALPHA_GRID:
                    alphas = {
                        "full_anchor": 0.0,
                        "delta_extreme": 0.0,
                        "strong_guard": a_strong,
                        "mild_guard": a_mild,
                        "no_guard": 1.0,
                    }
                    r = blend_rmse(oof, rule_map, alphas)
                    if r < best["rmse"] - 1e-9:
                        best = {"rmse": r, "full_hidden_thr": fh, "delta_extreme_thr": de,
                                "alpha_strong": a_strong, "alpha_mild": a_mild}

    improvement = base_rmse - best["rmse"]
    print(f"\n  BEST blended RMSE = {best['rmse']:.6f}  (improvement {improvement:+.6f})")
    print(f"  full_hidden_thr={best['full_hidden_thr']}, delta_extreme_thr={best['delta_extreme_thr']}, "
          f"alpha_strong={best['alpha_strong']}, alpha_mild={best['alpha_mild']}")

    # ── final OOF with best config ──
    best_alphas = {
        "full_anchor": 0.0, "delta_extreme": 0.0,
        "strong_guard": best["alpha_strong"], "mild_guard": best["alpha_mild"],
        "no_guard": 1.0,
    }
    final_rule = assign_rules(wf, best["full_hidden_thr"], best["delta_extreme_thr"])
    final_rule_map = dict(zip(wf["well_id"], final_rule))
    wf["guard_rule"] = final_rule
    print("\n  train guard rule 分布:")
    print(wf["guard_rule"].value_counts().to_string())

    # per-rule breakdown
    print("\n  per-rule RMSE:")
    for rl in ["full_anchor", "delta_extreme", "strong_guard", "mild_guard", "no_guard"]:
        wells = wf.loc[wf["guard_rule"].eq(rl), "well_id"]
        sub = oof[oof["well_id"].isin(wells)]
        if len(sub) == 0:
            continue
        a = best_alphas[rl]
        bl = a * sub["lgbm_pred_tvt"] + (1 - a) * sub["anchor_pred_tvt"]
        print(f"    {rl:14s}({wells.nunique():3d} wells) a={a:.2f} "
              f"blend={tvt_rmse(sub['TVT'], bl):.4f} "
              f"lgb={tvt_rmse(sub['TVT'], sub['lgbm_pred_tvt']):.4f} "
              f"anc={tvt_rmse(sub['TVT'], sub['anchor_pred_tvt']):.4f}")

    a_arr = oof["well_id"].map(lambda w: best_alphas[final_rule_map[w]]).to_numpy(float)
    oof_out = oof.copy()
    oof_out["guard_rule"] = oof["well_id"].map(final_rule_map)
    oof_out["alpha"] = a_arr
    oof_out["pred_tvt"] = a_arr * oof["lgbm_pred_tvt"] + (1 - a_arr) * oof["anchor_pred_tvt"]
    oof_out["error"] = oof_out["pred_tvt"] - oof["TVT"]
    oof_out["abs_error"] = oof_out["error"].abs()
    oof_out.to_csv(OUT_DIR / "oof.csv", index=False)
    wf.to_csv(OUT_DIR / "well_guard_flags.csv", index=False)

    # ── apply to test submission ──
    model_sub = pd.read_csv(OUT_DIR.parent / MODEL_EXP / "submission.csv").rename(columns={"tvt": "lgbm_tvt"})
    anchor_sub = pd.read_csv(OUT_DIR.parent / ANCHOR_EXP / "submission.csv").rename(columns={"tvt": "anchor_tvt"})
    sub = model_sub.merge(anchor_sub, on="id", how="left", validate="one_to_one")
    sub["well_id"] = sub["id"].str.rsplit("_", n=1).str[0]

    test_wf = compute_well_features(TEST_BASE)
    # test mean_pred_delta from submission
    base_t = pd.read_parquet(TEST_BASE, columns=["well_id", "id", "is_target", "last_known_TVT"])
    base_t = base_t[base_t["is_target"].astype(bool)]
    mp = model_sub.merge(base_t[["id", "well_id", "last_known_TVT"]], on="id", how="left")
    mp["pred_delta"] = mp["lgbm_tvt"] - mp["last_known_TVT"]
    test_mp = mp.groupby("well_id", as_index=False).agg(mean_pred_delta=("pred_delta", "mean"))
    test_wf = test_wf.merge(test_mp, on="well_id", how="left")
    test_rule = assign_rules(test_wf, best["full_hidden_thr"], best["delta_extreme_thr"])
    test_wf["guard_rule"] = test_rule
    test_rule_map = dict(zip(test_wf["well_id"], test_rule))
    print("\n  test guard rule 分布:")
    print(test_wf["guard_rule"].value_counts().to_string())

    sub["alpha"] = sub["well_id"].map(lambda w: best_alphas.get(test_rule_map.get(w, "no_guard"), 1.0))
    sub["tvt"] = sub["alpha"] * sub["lgbm_tvt"] + (1 - sub["alpha"]) * sub["anchor_tvt"]
    sample = pd.read_csv(SAMPLE_SUB)
    out_sub = sample[["id"]].merge(sub[["id", "tvt"]], on="id", how="left", validate="one_to_one")
    assert not out_sub["tvt"].isna().any()
    out_sub.to_csv(OUT_DIR / "submission.csv", index=False)
    test_changed = int((sub["alpha"] < 1.0).sum() > 0)

    # ── outputs ──
    result = {
        "exp_id": EXP_ID,
        "created_at": now_jst(),
        "status": "completed",
        "model_exp": MODEL_EXP,
        "anchor_exp": ANCHOR_EXP,
        "metric": "TVT_abs_RMSE",
        "lgbm_cv_rmse": base_rmse,
        "anchor_cv_rmse": anchor_rmse,
        "blended_cv_rmse": best["rmse"],
        "improvement_vs_lgbm": improvement,
        "best_config": {
            "full_anchor_hidden_thr": best["full_hidden_thr"],
            "delta_extreme_thr": best["delta_extreme_thr"],
            "alpha_strong": best["alpha_strong"],
            "alpha_mild": best["alpha_mild"],
        },
        "n_anchor_wins_wells": n_anchor_wins,
        "n_anchor_wins_big5": n_anchor_big,
        "train_rule_counts": wf["guard_rule"].value_counts().to_dict(),
        "test_rule_counts": test_wf["guard_rule"].value_counts().to_dict(),
        "test_submission_changed": bool(test_changed),
        "leak_risk": "low",
        "notes": "Feature-based anchor guard on exp008. Thresholds & alphas tuned on exp008 OOF (coarse grid).",
    }
    write_json(OUT_DIR / "result.json", result)
    print(f"\n[{EXP_ID}] 完了 -> {OUT_DIR}")
    print(json.dumps(result["best_config"], ensure_ascii=False))


if __name__ == "__main__":
    main()
