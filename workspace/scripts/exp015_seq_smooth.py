#!/usr/bin/env python3
"""exp015: Sequence smoothing post-process (Track 2 優先B).

データ確認済み: 真のTVTは MD/row_idx に沿って極めて滑らか
  （隣接行 |ΔTVT| 中央値0.01, 75%点0.02）。
一方 LightGBM(exp008) の予測は隣接差 平均0.029・max20 とジャンプ/ジッターを含む。
→ well内で予測TVTを系列平滑化すれば、真の滑らかさに近づき RMSE 改善が期待できる。

完全 leak-free（既存予測を well内 row_idx 順に平滑化するだけ。ラベル不使用）。
OOF上で手法・窓を sweep し、fold非一貫でないかも確認。同設定を test submission に適用。

usage: python exp015_seq_smooth.py --model-exp exp014_geom_extrap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp015_seq_smooth"


def smooth_series(v: np.ndarray, method: str, w: int, poly: int = 2) -> np.ndarray:
    n = len(v)
    if n < 3 or w < 3:
        return v
    ww = min(w, n if n % 2 == 1 else n - 1)
    if ww < 3:
        return v
    if method == "median":
        return pd.Series(v).rolling(ww, min_periods=1, center=True).median().to_numpy()
    if method == "mean":
        return pd.Series(v).rolling(ww, min_periods=1, center=True).mean().to_numpy()
    if method == "savgol":
        if ww % 2 == 0:
            ww -= 1
        if ww <= poly:
            return v
        return savgol_filter(v, ww, poly)
    raise ValueError(method)


def apply_smoothing(df: pd.DataFrame, pred_col: str, method: str, w: int) -> np.ndarray:
    """well内 row_idx 順に pred_col を平滑化して返す（df全体の順序を保持）。"""
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    out = np.empty(len(df))
    for _, idx in df.groupby("well_id", sort=False).groups.items():
        sub = df.loc[idx].sort_values("row_idx")
        sm = smooth_series(sub[pred_col].to_numpy(float), method, w)
        out[sub["_orig_order"].to_numpy()] = sm
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-exp", default="exp008_gr_rolling")
    args = ap.parse_args()
    model_exp = args.model_exp
    out_dir = Path("experiments") / f"{EXP_ID}_{model_exp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] sequence smoothing on {model_exp}")

    oof = pd.read_csv(Path("experiments") / model_exp / "oof.csv")
    base_rmse = tvt_rmse(oof["TVT"], oof["pred_tvt"])
    print(f"  baseline OOF RMSE = {base_rmse:.6f}")

    methods = ["median", "mean", "savgol"]
    windows = [5, 7, 11, 15, 21, 31, 51, 71, 101]

    rows = []
    best = {"rmse": base_rmse, "method": "none", "w": 0}
    for m in methods:
        for w in windows:
            sm = apply_smoothing(oof, "pred_tvt", m, w)
            r = tvt_rmse(oof["TVT"], sm)
            rows.append({"method": m, "w": w, "rmse": r, "delta": base_rmse - r})
            if r < best["rmse"] - 1e-9:
                best = {"rmse": r, "method": m, "w": w}
    sweep = pd.DataFrame(rows).sort_values("rmse")
    sweep.to_csv(out_dir / "sweep.csv", index=False)
    print("  top5 sweep:")
    print(sweep.head(5).to_string(index=False))
    print(f"\n  BEST: {best['method']} w={best['w']}  RMSE={best['rmse']:.6f}  "
          f"(improvement {base_rmse-best['rmse']:+.6f})")

    # fold-wise consistency check
    print("\n  fold別 improvement:")
    fold_deltas = {}
    if best["method"] != "none":
        for f in sorted(oof["fold"].unique()):
            s = oof[oof["fold"] == f]
            sm = apply_smoothing(s, "pred_tvt", best["method"], best["w"])
            r0 = tvt_rmse(s["TVT"], s["pred_tvt"]); r1 = tvt_rmse(s["TVT"], sm)
            fold_deltas[int(f)] = r0 - r1
            print(f"    fold{f}: {r0:.4f} -> {r1:.4f}  ({r0-r1:+.4f})")

    # apply to OOF + submission
    if best["method"] != "none":
        oof["pred_tvt_smooth"] = apply_smoothing(oof, "pred_tvt", best["method"], best["w"])
        oof["error"] = oof["pred_tvt_smooth"] - oof["TVT"]
        oof["abs_error"] = oof["error"].abs()
        oof.to_csv(out_dir / "oof.csv", index=False)

        sub = pd.read_csv(Path("experiments") / model_exp / "submission.csv")
        sub["well_id"] = sub["id"].str.rsplit("_", n=1).str[0]
        sub["row_idx"] = sub["id"].str.rsplit("_", n=1).str[1].astype(int)
        sub = sub.rename(columns={"tvt": "pred_tvt"})
        sub["tvt"] = apply_smoothing(sub, "pred_tvt", best["method"], best["w"])
        sub[["id", "tvt"]].to_csv(out_dir / "submission.csv", index=False)
        n_test_changed = int((sub["tvt"] != sub["pred_tvt"]).sum())
    else:
        n_test_changed = 0

    result = {"exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
              "model_exp": model_exp, "metric": "TVT_abs_RMSE",
              "baseline_cv_rmse": base_rmse, "best_cv_rmse": best["rmse"],
              "improvement": base_rmse - best["rmse"],
              "best_method": best["method"], "best_window": best["w"],
              "fold_deltas": fold_deltas,
              "fold_consistent": bool(all(v >= 0 for v in fold_deltas.values())) if fold_deltas else False,
              "n_test_rows_changed": n_test_changed,
              "leak_risk": "none (post-process on existing preds, no label)",
              "notes": "Per-well row_idx-ordered smoothing. Swept median/mean/savgol over windows."}
    write_json(out_dir / "result.json", result)
    print(f"\n[{EXP_ID}] 完了 -> {out_dir}")


if __name__ == "__main__":
    main()
