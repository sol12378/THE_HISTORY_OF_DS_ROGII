#!/usr/bin/env python3
"""exp010: OOF error slicing — exp009b の fold2 (+0.438) / fold4 (+0.177) 悪化要因分析.

目的:
  - exp009b (Group E あり) が exp008 (Group E なし) に対し、どの well で悪化したか特定する。
  - fold2 が +0.438 と最も崩れた原因を tw_known_gr_corr 区間別に層別する。
  - exp009 の分析 (corr>0.85 で改善) と矛盾する原因が corr 以外にあるか検証する。
  - 併せて、重複 typewell を共有する 34 wells がリーク露出しているか確認する (Action 2 診断)。

入力:
  - experiments/exp008_gr_rolling/oof.csv         (Group E なし baseline)
  - experiments/exp009b_typewell_gr_well_only/oof.csv (Group E あり)
  - data/processed/train_base_v001.parquet        (known GR)
  - data/processed/typewell_train_base_v001.parquet (typewell 曲線)

出力: experiments/exp010_oof_slicing_fold24/
"""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

EXP_DIR = Path("experiments") / "exp010_oof_slicing_fold24"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def per_well_abs_error(oof: pd.DataFrame) -> pd.DataFrame:
    """well 単位 RMSE と平均 abs_error。"""
    g = oof.groupby("well_id")
    out = pd.DataFrame({
        "fold": g["fold"].first(),
        "n_rows": g.size(),
        "rmse": g["error"].apply(lambda e: float(np.sqrt(np.mean(e.to_numpy() ** 2)))),
        "mae": g["abs_error"].mean(),
    })
    return out.reset_index()


def build_tw_interps(tw: pd.DataFrame) -> dict:
    interps = {}
    for wid, g in tw.groupby("well_id", sort=False):
        g = g.sort_values("TVT").drop_duplicates("TVT")
        if len(g) < 2:
            continue
        interps[wid] = interp1d(
            g["TVT"].to_numpy(dtype=float), g["GR"].to_numpy(dtype=float),
            kind="linear", bounds_error=False, fill_value="extrapolate",
        )
    return interps


def tw_known_gr_corr_per_well(train: pd.DataFrame, interps: dict) -> pd.DataFrame:
    """exp009b と同じ定義で tw_known_gr_corr を再計算。"""
    known = train[train["is_known_tvt"].astype(bool)]
    recs = []
    for wid, g in known.groupby("well_id", sort=False):
        g_gr = g[~g["is_gr_missing"].astype(bool)]
        if wid not in interps or len(g_gr) < 10:
            recs.append({"well_id": wid, "tw_known_gr_corr": np.nan,
                         "n_known_gr": int(len(g_gr))})
            continue
        fn = interps[wid]
        tw_gr_k = fn(g_gr["TVT_input"].to_numpy(dtype=float))
        corr = float(pd.Series(g_gr["GR"].values).corr(pd.Series(tw_gr_k)))
        recs.append({"well_id": wid, "tw_known_gr_corr": corr,
                     "n_known_gr": int(len(g_gr))})
    return pd.DataFrame(recs)


def typewell_signature(tw: pd.DataFrame) -> pd.DataFrame:
    """各 well の typewell 曲線をハッシュ化し、重複グループを検出。"""
    def sig(g):
        g = g.sort_values("row_idx")
        arr = np.round(g[["TVT", "GR"]].to_numpy(dtype=float), 3)
        return hashlib.md5(arr.tobytes()).hexdigest()
    sigs = tw.groupby("well_id").apply(sig, include_groups=False)
    sigs = sigs.rename("tw_sig").reset_index()
    counts = sigs["tw_sig"].value_counts()
    sigs["tw_group_size"] = sigs["tw_sig"].map(counts)
    sigs["shares_typewell"] = sigs["tw_group_size"] > 1
    return sigs


def main() -> None:
    print("[exp010] OOF 読み込み...")
    oof8 = pd.read_csv("experiments/exp008_gr_rolling/oof.csv")
    oof9b = pd.read_csv("experiments/exp009b_typewell_gr_well_only/oof.csv")

    # fold 割り当てが一致することを確認
    f8 = oof8.groupby("well_id")["fold"].first()
    f9 = oof9b.groupby("well_id")["fold"].first()
    assert (f8 == f9.reindex(f8.index)).all(), "fold 割り当てが exp008/exp009b で不一致"

    pw8 = per_well_abs_error(oof8).rename(columns={"rmse": "rmse_e8", "mae": "mae_e8"})
    pw9 = per_well_abs_error(oof9b).rename(columns={"rmse": "rmse_e9b", "mae": "mae_e9b"})
    pw = pw8.merge(pw9[["well_id", "rmse_e9b", "mae_e9b"]], on="well_id")
    pw["d_rmse"] = pw["rmse_e9b"] - pw["rmse_e8"]   # +なら exp009b が悪化
    pw["d_mae"] = pw["mae_e9b"] - pw["mae_e8"]

    # tw_known_gr_corr 再計算
    print("[exp010] typewell 補間器構築 + tw_known_gr_corr 再計算...")
    train = pd.read_parquet("data/processed/train_base_v001.parquet")
    tw = pd.read_parquet("data/processed/typewell_train_base_v001.parquet")
    tw_test = pd.read_parquet("data/processed/typewell_test_base_v001.parquet")
    interps = build_tw_interps(pd.concat([tw, tw_test], ignore_index=True))
    corr_df = tw_known_gr_corr_per_well(train, interps)
    pw = pw.merge(corr_df, on="well_id", how="left")

    # typewell 重複シグネチャ (リーク露出診断)
    sig_df = typewell_signature(tw)
    pw = pw.merge(sig_df[["well_id", "tw_group_size", "shares_typewell"]],
                  on="well_id", how="left")

    pw.to_csv(EXP_DIR / "per_well_comparison.csv", index=False)

    # ── fold 別サマリ ──────────────────────────────────────────────
    fold_summary = pw.groupby("fold").agg(
        n_wells=("well_id", "size"),
        mean_d_rmse=("d_rmse", "mean"),
        median_d_rmse=("d_rmse", "median"),
        n_worse=("d_rmse", lambda s: int((s > 0).sum())),
        n_better=("d_rmse", lambda s: int((s < 0).sum())),
    ).reset_index()

    # ── corr 区間別 (全体 + fold2/fold4) ───────────────────────────
    bins = [-1.01, 0.0, 0.60, 0.85, 1.01]
    labels = ["corr<0", "0-0.60", "0.60-0.85", "0.85-1.0"]
    pw["corr_bin"] = pd.cut(pw["tw_known_gr_corr"], bins=bins, labels=labels)

    def corr_table(sub):
        t = sub.groupby("corr_bin", observed=False).agg(
            n_wells=("well_id", "size"),
            mean_d_rmse=("d_rmse", "mean"),
            mean_rmse_e8=("rmse_e8", "mean"),
            mean_rmse_e9b=("rmse_e9b", "mean"),
        ).reset_index()
        return t

    corr_all = corr_table(pw)
    corr_f2 = corr_table(pw[pw["fold"] == 2])
    corr_f4 = corr_table(pw[pw["fold"] == 4])

    # ── fold2 で最も悪化した wells ──────────────────────────────────
    f2 = pw[pw["fold"] == 2].sort_values("d_rmse", ascending=False)
    top_worse_f2 = f2.head(20)[["well_id", "n_rows", "rmse_e8", "rmse_e9b",
                                "d_rmse", "tw_known_gr_corr", "n_known_gr",
                                "tw_group_size"]]
    f4 = pw[pw["fold"] == 4].sort_values("d_rmse", ascending=False)
    top_worse_f4 = f4.head(20)[["well_id", "n_rows", "rmse_e8", "rmse_e9b",
                                "d_rmse", "tw_known_gr_corr", "n_known_gr",
                                "tw_group_size"]]

    # ── 重複 typewell well のリーク露出診断 ─────────────────────────
    shared = pw[pw["shares_typewell"]].copy()
    # 各重複グループが複数 fold にまたがっているか
    shared_sig = pw.merge(sig_df[["well_id", "tw_sig"]], on="well_id")
    grp = shared_sig[shared_sig["shares_typewell"]].groupby("tw_sig").agg(
        group_size=("well_id", "size"),
        n_folds_spanned=("fold", "nunique"),
        mean_d_rmse=("d_rmse", "mean"),
    ).reset_index()
    n_split_groups = int((grp["n_folds_spanned"] > 1).sum())

    shared_vs_not = pw.groupby("shares_typewell").agg(
        n_wells=("well_id", "size"),
        mean_d_rmse=("d_rmse", "mean"),
        median_d_rmse=("d_rmse", "median"),
    ).reset_index()

    # ── 出力 ───────────────────────────────────────────────────────
    overall_d = pw["d_rmse"].mean()

    def fmt(df):
        return df.to_string(index=False)

    report = f"""# exp010: OOF error slicing — exp009b fold2/fold4 悪化分析

## 全体
- 比較対象: exp008 (Group E なし, CV=13.808621) vs exp009b (Group E あり, CV=13.932155)
- well 単位 RMSE 差 (exp009b - exp008) の平均: {overall_d:+.4f}  (+ = exp009b 悪化)
- 悪化した well 数: {int((pw['d_rmse']>0).sum())} / {len(pw)}
- 改善した well 数: {int((pw['d_rmse']<0).sum())} / {len(pw)}

## fold 別サマリ
{fmt(fold_summary)}

## corr 区間別 — 全 wells
{fmt(corr_all)}

## corr 区間別 — fold2 (最大悪化 +0.438)
{fmt(corr_f2)}

## corr 区間別 — fold4 (+0.177)
{fmt(corr_f4)}

## fold2 で最も悪化した上位 20 wells
{fmt(top_worse_f2)}

## fold4 で最も悪化した上位 20 wells
{fmt(top_worse_f4)}

## 重複 typewell リーク露出診断 (Action 2 補助)
- 重複 typewell を共有する well 数: {int(pw['shares_typewell'].sum())} / {len(pw)}
- 重複グループ数: {len(grp)}
- うち複数 fold にまたがるグループ数 (リーク露出): {n_split_groups}

### 共有 vs 非共有 wells の悪化度
{fmt(shared_vs_not)}

### 重複グループ別詳細
{fmt(grp)}
"""
    (EXP_DIR / "report.md").write_text(report, encoding="utf-8")
    corr_all.to_csv(EXP_DIR / "corr_bin_all.csv", index=False)
    corr_f2.to_csv(EXP_DIR / "corr_bin_fold2.csv", index=False)
    corr_f4.to_csv(EXP_DIR / "corr_bin_fold4.csv", index=False)
    fold_summary.to_csv(EXP_DIR / "fold_summary.csv", index=False)
    grp.to_csv(EXP_DIR / "shared_typewell_groups.csv", index=False)

    print(report)
    print(f"[exp010] 出力: {EXP_DIR}/")


if __name__ == "__main__":
    main()
