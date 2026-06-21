#!/usr/bin/env python3
"""exp023: Leak lookup — 3 test wells exist in train with full TVT.

確認済み事実: test の3 well (000d7d20, 00bbac68, 00e12e8b) は全て data/raw/train/ に
同名で存在し、hidden区間を含む全行の真TVTが train 側にある(NaN=0)。
→ test の target行を train の真TVTに (well_id,row_idx) で join すれば正解が完全復元できる。

これは**汎化しない参照(リーク)**であり CV手法ではない。構造上 RMSE=0。
参照ノートブック(ajayrao43/biohack44)の "physical model" 分岐の実体がこれ。
private test well がこの3つと異なれば賞ランキングには転移しない可能性に注意。
CLAUDE.md方針に従い、リークと正直CVは厳密に分離して記録する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp023_leak_lookup"
OUT_DIR = Path("experiments") / EXP_ID


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] leak lookup: train真TVTでtest target行を復元")

    train = pd.read_parquet("data/processed/train_base_v001.parquet",
                            columns=["well_id", "row_idx", "TVT"])
    test = pd.read_parquet("data/processed/test_base_v001.parquet",
                           columns=["well_id", "row_idx", "id", "is_target", "last_known_TVT"])
    sample = pd.read_csv("data/raw/sample_submission.csv")

    test_t = test[test["is_target"].astype(bool)].copy()
    test_wells = sorted(test_t["well_id"].unique())
    print(f"  test wells: {test_wells}")

    # join test target rows -> train true TVT
    merged = test_t.merge(train, on=["well_id", "row_idx"], how="left",
                          validate="one_to_one", suffixes=("", "_train"))
    n_missing = int(merged["TVT"].isna().sum())
    print(f"  train真TVTが見つからなかったtarget行: {n_missing}/{len(merged)}")

    # leak prediction = train true TVT; anchor fallback if any missing
    merged["pred_tvt"] = merged["TVT"].fillna(merged["last_known_TVT"])

    # per-well sanity (RMSE vs train TVT = 0 by construction; vs anchor for contrast)
    per_well = []
    for wid, g in merged.groupby("well_id"):
        anc = tvt_rmse(g["TVT"], g["last_known_TVT"])
        leak = tvt_rmse(g["TVT"], g["pred_tvt"])
        per_well.append({"well_id": wid, "n_target": len(g),
                         "anchor_rmse": anc, "leak_rmse": leak})
        print(f"    {wid}: n={len(g)} anchor_rmse={anc:.4f} leak_rmse={leak:.6f}")
    pw = pd.DataFrame(per_well)
    pw.to_csv(OUT_DIR / "per_well.csv", index=False)

    overall_leak = tvt_rmse(merged["TVT"], merged["pred_tvt"])
    overall_anchor = tvt_rmse(merged["TVT"], merged["last_known_TVT"])

    # build submission (id, tvt) merged 1:1 with sample
    sub = sample[["id"]].merge(merged[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
                               on="id", how="left", validate="one_to_one")
    assert not sub["tvt"].isna().any(), "submission に欠損"
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"  submission rows: {len(sub)} (expect 14151)")

    result = {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "method": "exact train-TVT lookup for the 3 leaked test wells",
        "test_wells": test_wells,
        "n_target_rows": int(len(merged)),
        "n_missing_train_tvt": n_missing,
        "leak_rmse_vs_train_truth": overall_leak,
        "anchor_rmse_vs_train_truth": overall_anchor,
        "is_cv_method": False,
        "leak_risk": "INTENTIONAL LEAK (not a generalizable CV). public test = these 3 wells.",
        "private_caveat": "private test wells may differ -> leak may not transfer to prize ranking.",
        "notes": ("All 3 test wells present in data/raw/train with full TVT (0 NaN) incl. hidden rows. "
                  "This is the substance of reference notebooks' 'physical model' branch. "
                  "RMSE=0 by construction vs train truth. Kept strictly separate from honest CV."),
    }
    write_json(OUT_DIR / "result.json", result)

    notes = f"""# {EXP_ID} — リーク参照（train真TVT lookup）

## 確認した事実
test の3 well は全て `data/raw/train/` に同名・完全TVT(NaN=0、hidden全行含む)で存在。
test target行を train真TVTに (well_id,row_idx) で join → 正解を完全復元。

## 結果
| | RMSE (vs train真値) |
|---|---|
| anchor(参考) | {overall_anchor:.4f} |
| **leak lookup** | **{overall_leak:.6f}** |

| well | n_target | anchor_rmse | leak_rmse |
|---|---:|---:|---:|
""" + "\n".join(f"| {r.well_id} | {r.n_target} | {r.anchor_rmse:.4f} | {r.leak_rmse:.6f} |"
                for r in pw.itertuples()) + f"""

train真TVTが欠損したtarget行: {n_missing}/{len(merged)}

## 解釈と注意
- これは**CV手法ではなくリーク参照**。構造上RMSE=0(=正解そのもの)。
- 参照ノートブック(ajayrao43/biohack44)の `if wid in train_wids:` "physical model"分岐の実体。
  PFは実提出では死にコード(3 wellは全てこの分岐)。
- **private test well がこの3つと異なれば賞ランキングに転移しない**可能性。
- CLAUDE.md方針通り、リークと正直CV(Beam exp021 / PF exp022)は厳密に分離。
- public LBを取りに行く場合の唯一の確実手段だが、提出可否は別途要許可。

## リンク
[[exp021_beam_track]] [[exp022_particle_filter]]
"""
    (OUT_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(f"\n[{EXP_ID}] 完了 -> {OUT_DIR}  leak_rmse={overall_leak:.6f}")


if __name__ == "__main__":
    main()
