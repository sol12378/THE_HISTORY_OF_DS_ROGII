#!/usr/bin/env python3
"""exp021: Beam Search GR-typewell tracker — honest 773-well CV.

参照ノートブック(ajayrao43/biohack44)の beam_search を忠実移植し、
全773 train well で正直なリークなしCVを算出する。

手法(DTW型ハードトラッカー):
  typewell GR-TVT曲線上で現在index(bidx)を1行±2ステップだけ動かす。
  各ステップ cost = GR誤差²/es + 移動コスト(mc×|move|)。bs本の仮説を保持し累積コスト最小経路。
  14 config を平均アンサンブル。GR欠損は well全体を補間してから使用。
完全leak-free: 入力は GR + typewell + anchor のみ。hidden区間のTVTは一切使わない。

CV枠組み(exp013と同じ): 各wellで known=is_known_tvt, target=is_target。
last_tvt=anchor から target区間をトラッキング予測し、真TVTと pooled tvt_rmse で比較。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp021_beam_track"
OUT_DIR = Path("experiments") / EXP_ID

# 14 beam configs (bs=beam size, mc=motion cost, es=GR error scale, r=savgol radius)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5), (20, 4.0, 36.0, 3), (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2), (20, 30.0, 200.0, 2), (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3), (10, 40.0, 300.0, 1), (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2), (10, 50.0, 400.0, 0),
]


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    """Vectorized beam search for TVT tracking via GR matching (faithful port)."""
    n = len(hgr); nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])
    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2., 1., 0., 1., 2.])
    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf); bcost[0] = 0.
    bn = 1
    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)
        ni_f = ni.flatten(); tot_f = tot.flatten(); vf = valid.flatten()
        ni_f = ni_f[vf]; tot_f = tot_f[vf]
        order = np.argsort(tot_f); ni_s = ni_f[order]; tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]; tot_u = tot_s[first]
        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]
        bidx[:kept] = ni_u[top]; bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]; bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]
    return result


def track_well(g, tw_g, use_offset=False):
    """1 well を 14 config beam で予測。target区間の予測配列を返す(row_idx順)。"""
    g = g.sort_values("row_idx")
    known = g[g["is_known_tvt"].astype(bool)]
    tgt = g[g["is_target"].astype(bool)]
    if len(tgt) == 0:
        return None
    anchor = float(tgt["last_known_TVT"].iloc[0])
    if tw_g is None or len(tw_g) < 2:
        return np.full(len(tgt), anchor)

    tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
    tw_tvt = tw_s["TVT"].to_numpy(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)

    # GR: interpolate full-well series, then take target rows (faithful to reference)
    gr_full = g["GR"].interpolate(limit_direction="both")
    gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    tgt_pos = np.where(g["is_target"].astype(bool).to_numpy())[0]
    hgr = gr_full[tgt_pos]

    offset = 0.0
    if use_offset:
        kv = known[~known["is_gr_missing"].astype(bool)]
        if len(kv) >= 10:
            exp_gr = np.interp(kv["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
            offset = float(np.median(kv["GR"].to_numpy(float) - exp_gr))
    hgr_use = hgr - offset

    beam_results = [beam_search(hgr_use, tw_tvt, tw_gr, anchor, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]
    return np.stack(beam_results, 0).mean(0)


def run_split(base_path, tw_path, use_offset=False):
    tr = pd.read_parquet(base_path, columns=[
        "well_id", "row_idx", "GR", "TVT", "TVT_input", "id",
        "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"])
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    parts = []
    well_rows = []
    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    groups = list(sel.groupby("well_id", sort=False))
    for j, (wid, g) in enumerate(groups):
        tgt = g[g["is_target"].astype(bool)].sort_values("row_idx")
        if len(tgt) == 0:
            continue
        pred = track_well(g, tw_by_well.get(wid), use_offset=use_offset)
        anchor = float(tgt["last_known_TVT"].iloc[0])
        out = tgt[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy()
        out["pred_tvt"] = pred
        parts.append(out)
        if "TVT" in tgt and tgt["TVT"].notna().all():
            well_rows.append({"well_id": wid, "n": len(tgt),
                              "anchor_rmse": tvt_rmse(tgt["TVT"], tgt["last_known_TVT"]),
                              "beam_rmse": tvt_rmse(tgt["TVT"], pred)})
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{len(groups)} wells done")
    preds = pd.concat(parts, ignore_index=True)
    return preds, pd.DataFrame(well_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Beam Search GR-typewell tracker — 773-well CV")

    # ---- TRAIN CV (faithful, no offset) ----
    print("TRAIN 773 wells (faithful, no offset) ...")
    preds, well = run_split("data/processed/train_base_v001.parquet",
                            "data/processed/typewell_train_base_v001.parquet",
                            use_offset=False)
    cv = tvt_rmse(preds["TVT"], preds["pred_tvt"])
    anc = tvt_rmse(preds["TVT"], preds["last_known_TVT"])
    print(f"  beam CV (faithful) = {cv:.6f}   anchor = {anc:.6f}")

    preds["error"] = preds["pred_tvt"] - preds["TVT"]
    preds["abs_error"] = preds["error"].abs()
    preds.to_csv(OUT_DIR / "oof.csv", index=False)
    well.to_csv(OUT_DIR / "per_well.csv", index=False)
    n_beat = int((well["beam_rmse"] < well["anchor_rmse"]).sum())
    print(f"  beam が anchor に勝つ well: {n_beat}/{len(well)}")

    # ---- TRAIN CV with offset calibration variant ----
    print("TRAIN 773 wells (+offset calibration) ...")
    preds_o, well_o = run_split("data/processed/train_base_v001.parquet",
                                "data/processed/typewell_train_base_v001.parquet",
                                use_offset=True)
    cv_o = tvt_rmse(preds_o["TVT"], preds_o["pred_tvt"])
    print(f"  beam CV (+offset) = {cv_o:.6f}")

    # ---- TEST submission (use better variant) ----
    use_off = cv_o < cv
    print(f"TEST submission (use_offset={use_off}) ...")
    test_preds, _ = run_split("data/processed/test_base_v001.parquet",
                              "data/processed/typewell_test_base_v001.parquet",
                              use_offset=use_off)
    sample = pd.read_csv("data/raw/sample_submission.csv")
    sub = sample[["id"]].merge(test_preds[["id", "pred_tvt"]].rename(columns={"pred_tvt": "tvt"}),
                               on="id", how="left", validate="one_to_one")
    assert not sub["tvt"].isna().any(), "submission欠損"
    sub.to_csv(OUT_DIR / "submission.csv", index=False)
    print(f"  submission rows: {len(sub)}")

    # 3 test wells per-well (vs train truth, for sanity vs reference 4.71 claim)
    test_truth = pd.read_parquet("data/processed/train_base_v001.parquet",
                                 columns=["well_id", "row_idx", "TVT"])
    tp = test_preds.merge(test_truth, on=["well_id", "row_idx"], how="left", suffixes=("", "_truth"))
    test_well_rmse = {}
    for wid, gg in tp.groupby("well_id"):
        r = tvt_rmse(gg["TVT_truth"], gg["pred_tvt"])
        test_well_rmse[wid] = r
        print(f"    test {wid}: beam vs train-truth RMSE = {r:.4f}")

    result = {
        "exp_id": EXP_ID, "created_at": now_jst(), "status": "completed",
        "method": "Beam Search GR-typewell tracker (14-config ensemble), leak-free",
        "beam_configs": len(BEAM_CONFIGS),
        "cv_rmse_faithful": cv, "cv_rmse_offset": cv_o,
        "anchor_rmse": anc, "best_cv": min(cv, cv_o),
        "n_wells": int(len(well)), "n_beam_beats_anchor": n_beat,
        "test_well_rmse_vs_train_truth": test_well_rmse,
        "leak_risk": "none (no hidden TVT used; GR+typewell+anchor only)",
        "compare": {"exp014_geom": 13.525189, "best_blend_exp020": 13.320964, "anchor": anc},
        "notes": ("Faithful port of reference beam_search. Tracks typewell index ±2/row with "
                  "motion cost + GR error cost; 14-config mean. GR interpolated full-well first."),
    }
    write_json(OUT_DIR / "result.json", result)

    twr = "\n".join(f"| {k} | {v:.4f} |" for k, v in test_well_rmse.items())
    notes = f"""# {EXP_ID} — Beam Search GR-typewell トラッカー

## 手法
参照ノートの beam_search を忠実移植(DTW型)。typewell index を1行±2ステップ動かし
cost=GR誤差²/es + 移動コスト の累積最小経路を bs本ビームで探索。14 config平均。
GRは well全体を補間してから使用。**完全leak-free**(hidden TVT不使用)。

## 結果(773-well pooled CV)
| 手法 | CV RMSE |
|---|---|
| anchor | {anc:.6f} |
| **beam (faithful)** | **{cv:.6f}** |
| beam (+offset較正) | {cv_o:.6f} |
| 参考: exp014 geom | 13.525189 |
| 参考: best blend(exp020) | 13.320964 |

beam が anchor に勝つ well: {n_beat}/{len(well)}

## 3 test well (beam vs train真値, 参照の"4.71 ft"主張と照合)
| well | RMSE |
|---|---|
{twr}

## 解釈
(実行後に追記)

## リンク
[[exp022_particle_filter]] [[exp023_leak_lookup]] [[exp014_geom_extrap]]
"""
    (OUT_DIR / "notes.md").write_text(notes, encoding="utf-8")
    print(f"\n[{EXP_ID}] 完了 -> {OUT_DIR}  best_cv={min(cv,cv_o):.6f}")


if __name__ == "__main__":
    main()
