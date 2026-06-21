"""exp073 sanity check (leak-free 予測には一切影響しない、診断のみ)。

タスク2: pilk_cat 推論 delta 分布 vs package OOF (catboost_oof.npy) の突合。
  - 全train OOF mean (期待 ~+1.66) と、3 test well の hidden 行に限定した OOF mean を比較。
  - 推論側の pilk_cat mean とどちらに整合するかを判定。
タスク3: 3 test well は train にも存在し真 TVT を持つ。submission を真 TVT(hidden 行=row_idx 一致)
  と join し blend / 各成分単体の in-distribution RMSE を計算 (gross check, 楽観的)。

真 TVT は RMSE 計算にのみ使用。予測生成には一切使わない。
出力ログは submit_build_log2.txt に追記。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
EXT = REPO / "data" / "external"
RAW = REPO / "data" / "raw"
WORK = REPO / "experiments" / "exp073_public_assets_integration"
COMP = WORK / "_components_submit.parquet"
SUB = WORK / "submission.csv"
OOF = EXT / "rogii-model-package" / "oof" / "catboost_oof.npy"
GT = EXT / "rogii-model-package" / "oof" / "train_gt.parquet"
LOG = WORK / "submit_build_log2.txt"

# DELTA-space fixed blend weights (rogii_exp073_submit.py と一致)
W = dict(pf=0.440, geom=0.096, rav_lgb3=0.284, rav_cb1=0.031, rav_cb2=0.030, pilk_cat=0.348)
COMP_MAP = {"pf": "pf_delta", "geom": "geom_delta", "rav_lgb3": "rav_lgb3",
            "rav_cb1": "rav_cb1", "rav_cb2": "rav_cb2", "pilk_cat": "pilk_cat"}

lines = []
def log(m):
    s = f"[sanity] {m}"; print(s, flush=True); lines.append(s)


def main():
    df = pd.read_parquet(COMP); df["id"] = df["id"].astype(str)
    sub = pd.read_csv(SUB); sub["id"] = sub["id"].astype(str)
    oof = np.load(OOF)
    gt = pd.read_parquet(GT); gt["id"] = gt["id"].astype(str)
    gt = gt.reset_index(drop=True)
    gt["oof_cat"] = oof  # row-aligned per earlier verification

    # ---------- タスク2: pilk_cat 忠実性 ----------
    log("==== TASK2: pilk_cat fidelity vs package OOF ====")
    infer_mean = float(df["pilk_cat"].mean())
    log(f"pilk_cat INFER (test hidden rows, n={len(df)}): "
        f"mean={infer_mean:.3f} std={df['pilk_cat'].std():.3f} "
        f"min={df['pilk_cat'].min():.2f} max={df['pilk_cat'].max():.2f}")
    log(f"catboost OOF全train (n={len(oof)}): mean={float(np.nanmean(oof)):.3f} "
        f"std={float(np.nanstd(oof)):.3f}")
    # 3 test well は train にも存在 -> OOF を同well の hidden(=eval, sub id) 行に限定
    sub_ids = set(sub["id"])
    gt_sub = gt[gt["id"].isin(sub_ids)].copy()
    log(f"OOF restricted to the SAME 3 wells' eval rows (n={len(gt_sub)}): "
        f"oof_cat mean={float(gt_sub['oof_cat'].mean()):.3f} "
        f"std={float(gt_sub['oof_cat'].std()):.3f}")
    log(f"true delta (target_delta_from_last_known) on those eval rows: "
        f"mean={float(gt_sub['target_delta_from_last_known'].mean()):.3f}")
    # 同一well・同一行での直接突合 (OOF predicted vs INFER predicted)
    m = df[["id", "pilk_cat"]].merge(
        gt_sub[["id", "oof_cat", "target_delta_from_last_known"]], on="id", how="inner")
    if len(m):
        d = m["pilk_cat"] - m["oof_cat"]
        log(f"per-row INFER vs OOF (same wells, n={len(m)}): "
            f"mean(infer)={m['pilk_cat'].mean():.3f} mean(oof)={m['oof_cat'].mean():.3f} "
            f"mean(infer-oof)={float(d.mean()):.3f} medabs={float(d.abs().median()):.3f} "
            f"corr={float(np.corrcoef(m['pilk_cat'], m['oof_cat'])[0,1]):.3f}")
        log("判定: 全train OOF mean(+1.66)は train全wellのdelta分布。3 test wellは"
            "deltaがやや負側に寄る分布で、OOFをその同well eval行に限定すると "
            f"{float(gt_sub['oof_cat'].mean()):.2f}。推論mean {infer_mean:.2f} は "
            "この同well限定OOFと比較すべきで、per-row相関/medabsで忠実性を評価。")
    else:
        log("WARN: 3 well の id が gt と一致せず per-row 突合不可")

    # ---------- タスク3: 3-well in-distribution RMSE ----------
    log("==== TASK3: 3-well in-distribution RMSE (true TVT join) ====")
    # 真 TVT を train horizontal well から row_idx 一致で取得
    true_tvt = {}
    anchor = {}  # last_known_TVT (delta->tvt 復元用)
    for wid in sorted({i.rsplit("_", 1)[0] for i in sub["id"]}):
        hw = pd.read_csv(RAW / "train" / f"{wid}__horizontal_well.csv")
        tv = pd.to_numeric(hw["TVT"], errors="coerce").to_numpy(float)
        for rid in sub["id"]:
            if rid.rsplit("_", 1)[0] != wid:
                continue
            ridx = int(rid.rsplit("_", 1)[1])
            if 0 <= ridx < len(tv) and np.isfinite(tv[ridx]):
                true_tvt[rid] = float(tv[ridx])
    log(f"true TVT joined for {len(true_tvt)}/{len(sub)} rows")

    # blend submission RMSE
    s = sub[sub["id"].isin(true_tvt)].copy()
    yt = s["id"].map(true_tvt).to_numpy(float)
    yp = s["tvt"].to_numpy(float)
    blend_rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    log(f"BLEND submission RMSE = {blend_rmse:.3f} ft (n={len(s)})  "
        f"pred[{yp.min():.1f},{yp.max():.1f}] true[{yt.min():.1f},{yt.max():.1f}]")

    # 各成分単体: anchor + component_delta (後処理なしの素のdelta評価)。
    g = gt[gt["id"].isin(true_tvt)][["id", "last_known_TVT", "target_tvt"]].copy()
    base = df.merge(g, on="id", how="inner")
    yt2 = base["target_tvt"].to_numpy(float)
    a = base["last_known_TVT"].to_numpy(float)
    log(f"per-component raw-delta RMSE (anchor + w*delta, 後処理なし, n={len(base)}):")
    for name, col in COMP_MAP.items():
        pred = a + base[col].to_numpy(float)
        rmse = float(np.sqrt(np.mean((pred - yt2) ** 2)))
        log(f"  {name:9s} (raw delta) RMSE={rmse:.3f}  delta_mean={base[col].mean():.2f}")
    # raw blend (後処理なし) も
    fd = np.zeros(len(base))
    for name, w in W.items():
        fd += w * base[COMP_MAP[name]].to_numpy(float)
    raw_blend = float(np.sqrt(np.mean((a + fd - yt2) ** 2)))
    log(f"  RAW BLEND (no postproc) RMSE={raw_blend:.3f}  (vs post-proc submission {blend_rmse:.3f})")

    # gross check 判定
    verdict = "OK (数ft〜十数ft, 50ft級でない)" if blend_rmse < 25 else "WARN (>25ft, 要調査)"
    log(f"GROSS CHECK: blend RMSE {blend_rmse:.2f}ft -> {verdict}")

    # append to build log
    prev = LOG.read_text(encoding="utf-8") if LOG.exists() else ""
    LOG.write_text(prev + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    log(f"appended to {LOG}")


if __name__ == "__main__":
    main()
