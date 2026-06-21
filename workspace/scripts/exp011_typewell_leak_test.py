#!/usr/bin/env python3
"""exp011: Typewell リーク検証 — GroupKFold by typewell (重複 typewell 同一 fold).

仮説:
  exp009b の Group E 特徴量が importance 上位なのに CV を悪化させた。
  well_id GroupKFold では、同一 typewell 曲線を共有する 34 wells (13 グループ,
  うち 10 グループが複数 fold にまたがる) が train/val 間でリーク露出している。
  Group E は typewell 由来なので、共有 typewell 経由で val 側が train 情報の恩恵を
  受け、CV が楽観的になっている可能性がある (exp010 診断: 共有 wells は -0.182 改善 vs
  非共有 +0.090 悪化)。

設計 (リーク分離のための最小移動 fold):
  - 既存 well-fold (folds_group_well_v001.csv) を基準にする。
  - 重複 typewell グループのメンバーを、グループ代表 (min well_id) の fold に強制集約。
    → 最大 34 wells だけが移動し、fold 構成ノイズを最小化してリークだけを除去。
  - この "deduped fold" 上で 2 構成を学習:
      config_baseline = exp008 特徴量 (Group E なし)
      config_E        = exp009b 特徴量 (Group E あり)
  - 比較:
      well-fold:   E - baseline = 13.932155 - 13.808621 = +0.123534 (E は悪化)
      deduped:     E - baseline = ?
    deduped で E-baseline gap が更に拡大 → リークが well-fold で E を過大評価していた。
    gap がほぼ不変 → リークは軽微で、悪化は純粋な overfit。

注意: 学習は LightGBM 1500 木 × 5 fold × 2 構成 = 重い (~30 分想定)。
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import hashlib
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rogii.training.baselines import (  # noqa: E402
    PRED_COL, SAFE_FEATURES, target_rows, tvt_rmse, write_json, now_jst,
)

# ── exp009b モジュールを動的 import (特徴量関数を再利用) ──────────────────────
_spec = importlib.util.spec_from_file_location(
    "exp009b_mod", ROOT / "scripts" / "exp009b_typewell_gr_well_only.py")
exp009b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp009b)

EXP_ID = "exp011_typewell_leak_test"
EXP_DIR = Path("experiments") / EXP_ID
EXP_DIR.mkdir(parents=True, exist_ok=True)

# baseline (exp008) features = exp009b ALL_FEATURES から Group E を除く
FEATURES_E = exp009b.ALL_FEATURES
FEATURES_BASELINE = [f for f in FEATURES_E if f not in exp009b.GROUP_E]

PARAMS = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 63, "max_depth": -1,
    "min_data_in_leaf": 50, "feature_fraction": 0.9,
    "bagging_fraction": 0.9, "bagging_freq": 1,
    "lambda_l2": 1.0, "verbosity": -1, "seed": 42, "num_threads": 8,
}


def typewell_signature(tw: pd.DataFrame) -> pd.DataFrame:
    def sig(g):
        g = g.sort_values("row_idx")
        arr = np.round(g[["TVT", "GR"]].to_numpy(dtype=float), 3)
        return hashlib.md5(arr.tobytes()).hexdigest()
    s = tw.groupby("well_id").apply(sig, include_groups=False).rename("tw_sig")
    return s.reset_index()


def build_deduped_folds(folds: pd.DataFrame, sig_df: pd.DataFrame) -> pd.DataFrame:
    """重複 typewell グループを単一 fold に集約した fold マップを返す (well_id, fold)。"""
    fmap = folds.loc[folds["split"].eq("train"), ["well_id", "fold"]].copy()
    fmap = fmap.merge(sig_df, on="well_id", how="left")
    # 各 typewell グループの代表 fold = min well_id の fold
    rep = (fmap.sort_values("well_id")
                .groupby("tw_sig")["fold"].first().rename("fold_dedup"))
    fmap = fmap.merge(rep, on="tw_sig", how="left")
    fmap["fold_new"] = fmap["fold_dedup"].fillna(fmap["fold"]).astype(int)
    n_moved = int((fmap["fold_new"] != fmap["fold"]).sum())
    print(f"  deduped fold: {n_moved} wells が移動 (重複 typewell 集約)")
    return fmap[["well_id", "fold_new"]].rename(columns={"fold_new": "fold"}), n_moved


def run_cv(train_t: pd.DataFrame, y_delta: pd.Series, features: list[str],
           tag: str) -> dict:
    oof = np.zeros(len(train_t), dtype=float)
    fold_rows = []
    n_folds = int(train_t["fold"].nunique())
    for fold in sorted(train_t["fold"].unique()):
        vm = train_t["fold"].eq(fold).to_numpy()
        tm = ~vm
        model = lgb.LGBMRegressor(**PARAMS, n_estimators=1500)
        model.fit(
            train_t.loc[tm, features], y_delta.loc[tm],
            eval_set=[(train_t.loc[vm, features], y_delta.loc[vm])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        best = int(model.best_iteration_ or model.n_estimators)
        vd = model.predict(train_t.loc[vm, features], num_iteration=best)
        oof[vm] = vd
        ftvt = train_t.loc[vm, "last_known_TVT"].to_numpy(dtype=float) + vd
        fr = tvt_rmse(train_t.loc[vm, "TVT"], ftvt)
        fold_rows.append({"fold": int(fold), "n_rows": int(vm.sum()),
                          "rmse": fr, "best_iteration": best})
        print(f"    [{tag}] fold {fold}  RMSE={fr:.6f}  best_iter={best}")
    ftvt_all = train_t["last_known_TVT"].to_numpy(dtype=float) + oof
    overall = tvt_rmse(train_t["TVT"], ftvt_all)
    return {"cv_rmse": overall,
            "cv_mean": float(np.mean([r["rmse"] for r in fold_rows])),
            "folds": fold_rows}


def main() -> None:
    print(f"[{EXP_ID}] typewell リーク検証 (重い: LGBM 1500 木 ×5fold ×2構成)")

    train = pd.read_parquet("data/processed/train_base_v001.parquet")
    test = pd.read_parquet("data/processed/test_base_v001.parquet")
    folds = pd.read_csv("data/folds/folds_group_well_v001.csv")

    global_gr_mean = float(train.loc[~train["is_gr_missing"].astype(bool), "GR"].mean())

    tw_train = pd.read_parquet("data/processed/typewell_train_base_v001.parquet")
    tw_test = pd.read_parquet("data/processed/typewell_test_base_v001.parquet")
    tw_all = pd.concat([tw_train, tw_test], ignore_index=True)
    tw_interps = exp009b._build_tw_interpolators(tw_all)

    # deduped fold 構築 (train typewell の重複で集約)
    sig_df = typewell_signature(tw_train)
    dedup_map, n_moved = build_deduped_folds(folds, sig_df)
    dedup_map.to_csv("data/folds/folds_group_typewell_v001.csv", index=False)

    print("特徴量エンジニアリング (exp009b enrich)...")
    train = exp009b.enrich(train, tw_interps, global_gr_mean)

    # deduped fold を付与
    train = train.merge(dedup_map, on="well_id", how="left", validate="many_to_one")
    assert not train["fold"].isna().any()
    train["fold"] = train["fold"].astype(int)

    train_t = target_rows(train)
    y_delta = train_t["TVT"].astype(float) - train_t["last_known_TVT"].astype(float)

    print(f"  baseline features (no E): {len(FEATURES_BASELINE)}")
    print(f"  E features:               {len(FEATURES_E)}")

    print("\n=== config_baseline (exp008 features, deduped fold) ===")
    res_base = run_cv(train_t, y_delta, FEATURES_BASELINE, "base")
    print("\n=== config_E (exp009b features, deduped fold) ===")
    res_e = run_cv(train_t, y_delta, FEATURES_E, "E")

    # ── 比較 ────────────────────────────────────────────────────────
    WELL_BASE = 13.808621   # exp008 well-fold CV
    WELL_E = 13.932155      # exp009b well-fold CV
    gap_well = WELL_E - WELL_BASE
    gap_dedup = res_e["cv_rmse"] - res_base["cv_rmse"]

    result = {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "purpose": "typewell leak verification via deduped GroupKFold",
        "n_wells_moved": n_moved,
        "fold_scheme": "deduped (shared typewell forced into same fold)",
        "well_fold": {"baseline_cv": WELL_BASE, "E_cv": WELL_E,
                      "E_minus_baseline": round(gap_well, 6)},
        "deduped_fold": {"baseline_cv": round(res_base["cv_rmse"], 6),
                         "E_cv": round(res_e["cv_rmse"], 6),
                         "E_minus_baseline": round(gap_dedup, 6)},
        "gap_change_dedup_minus_well": round(gap_dedup - gap_well, 6),
        "baseline_folds": res_base["folds"],
        "E_folds": res_e["folds"],
        "leak_verdict": (
            "LEAK_CONFIRMED: E penalty grows under deduped folds"
            if gap_dedup > gap_well + 0.02 else
            "LEAK_NEGLIGIBLE: E penalty stable → damage is genuine overfit"
            if abs(gap_dedup - gap_well) <= 0.02 else
            "INCONCLUSIVE: E penalty shrank under deduped folds"
        ),
    }
    write_json(EXP_DIR / "result.json", result)

    fb = "\n".join(f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |"
                   for r in res_base["folds"])
    fe = "\n".join(f"| {r['fold']} | {r['rmse']:.6f} | {r['best_iteration']} |"
                   for r in res_e["folds"])
    notes = f"""# {EXP_ID}: typewell リーク検証

## 設計
- 重複 typewell グループを単一 fold に集約した deduped fold を構築 ({n_moved} wells 移動)。
- 同じ deduped fold 上で exp008 特徴量(E無し)と exp009b 特徴量(E有り)を学習。
- E の悪化幅が fold 方式で変わるかを比較してリークを判定。

## 結果

| fold 方式 | baseline (E無し) | E有り | E − baseline |
|---|---:|---:|---:|
| well-fold (既存) | {WELL_BASE:.6f} | {WELL_E:.6f} | {gap_well:+.6f} |
| deduped (typewell集約) | {res_base['cv_rmse']:.6f} | {res_e['cv_rmse']:.6f} | {gap_dedup:+.6f} |

- gap 変化 (deduped − well): {gap_dedup - gap_well:+.6f}
- **判定: {result['leak_verdict']}**

## baseline fold 別 (deduped)
| fold | rmse | best_iter |
|---|---:|---:|
{fb}

## E fold 別 (deduped)
| fold | rmse | best_iter |
|---|---:|---:|
{fe}

## 解釈
- gap が +0.02 以上拡大 → well-fold が共有 typewell リークで E を過大評価していた。
- gap がほぼ不変 → リークは軽微、悪化は overfit。Group E 封印の妥当性を補強。
"""
    (EXP_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print("\n" + notes)
    print(f"[{EXP_ID}] 完了 → {EXP_DIR}/")


if __name__ == "__main__":
    main()
