#!/usr/bin/env python3
"""diag: GR-typewell matching の理論上限(oracle ceiling)を測定する。

目的: GRがTVTについて持つ情報量の上限を知る。CV<=5がGR照合で物理的に
到達可能かを判定するための最重要診断。

oracle: 各hidden行で、較正済み観測GRに「GR的に整合する」typewell TVT候補の中から
        TRUE TVT に最も近いものを選ぶ(=多価性を完璧に解消できた場合の上限)。
        これは提出には使えない(真値を使う)が、上限の測定には有効。

較正: known区間で obs_GR ↔ tw_GR(at known TVT_input) を線形回帰 (obs≈a*tw+b)。
      逆変換 tw_equiv = (obs-b)/a で観測GRをtypewellスケールへ。

variants:
  anchor                 : delta=0 基準
  nearest_gr_window      : 窓内でGR最近傍(非oracle, exp013相当・較正改善版)
  oracle_window_W        : 窓 anchor±W 内でGR整合候補のうち真値最近 (ceiling@window)
  oracle_full            : typewell全域でGR整合候補のうち真値最近 (絶対ceiling)
  oracle_geomprior       : exp014 geom予測±band 内でGR整合候補のうち真値最近
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, now_jst

OUT = Path("experiments") / "diag_gr_ceiling"
GRID_STEP = 0.5
GR_TOL = 6.0          # GR整合とみなす|Δtw_GR|許容(typewellスケール)
WINDOWS = [40.0, 60.0, 110.0]
GEOM_BAND = 25.0      # geom prior 周りの探索半径

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet("data/processed/train_base_v001.parquet",
        columns=["well_id","row_idx","GR","TVT","TVT_input","is_target",
                 "is_known_tvt","is_gr_missing","last_known_TVT"])
    tw_all = pd.read_parquet("data/processed/typewell_train_base_v001.parquet",
        columns=["well_id","TVT","GR"])
    tw_by = {w:g for w,g in tw_all.groupby("well_id", sort=False)}
    # exp014 geom prior (current strong leak-free prior)
    geom = pd.read_csv("experiments/exp014_geom_extrap/oof.csv",
        usecols=["well_id","row_idx","pred_tvt"]).rename(columns={"pred_tvt":"geom_tvt"})
    geom_map = {(r.well_id, r.row_idx): r.geom_tvt for r in geom.itertuples()}

    parts = []
    for wid, g in tr[tr.is_target.astype(bool) | tr.is_known_tvt.astype(bool)].groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g.is_known_tvt.astype(bool)]
        tgt = g[g.is_target.astype(bool)].copy()
        if len(tgt)==0: continue
        anchor = float(tgt.last_known_TVT.iloc[0])
        tw_g = tw_by.get(wid)
        n = len(tgt)
        out = {"well_id": np.full(n,wid), "row_idx": tgt.row_idx.to_numpy(),
               "TVT": tgt.TVT.to_numpy(float), "anchor": np.full(n,anchor)}
        geomp = np.array([geom_map.get((wid,ri), anchor) for ri in tgt.row_idx.to_numpy()], float)
        out["geom"] = geomp
        if tw_g is None or len(tw_g)<2:
            for w in WINDOWS: out[f"oracle_w{int(w)}"]=np.full(n,anchor)
            out["nearest_w60"]=np.full(n,anchor); out["oracle_full"]=np.full(n,anchor)
            out["oracle_geom"]=geomp
            parts.append(pd.DataFrame(out)); continue
        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tvt_f = tw_s.TVT.to_numpy(float); gr_f = tw_s.GR.to_numpy(float)
        # fine grid over full tw range
        grid = np.arange(tvt_f.min(), tvt_f.max()+GRID_STEP, GRID_STEP)
        grid_gr = np.interp(grid, tvt_f, gr_f)
        # calibration on known section
        kv = known[~known.is_gr_missing.astype(bool)]
        if len(kv)>=10:
            tw_at_known = np.interp(kv.TVT_input.to_numpy(float), tvt_f, gr_f)
            obs_k = kv.GR.to_numpy(float)
            A = np.vstack([tw_at_known, np.ones_like(tw_at_known)]).T
            coef,*_ = np.linalg.lstsq(A, obs_k, rcond=None)
            a,b = float(coef[0]), float(coef[1])
            if abs(a)<1e-3: a=1.0; b=0.0
        else:
            a,b = 1.0, 0.0
        gr_obs = tgt.GR.to_numpy(float)
        has = ~tgt.is_gr_missing.astype(bool).to_numpy()
        tw_equiv = (gr_obs - b)/a   # observed GR mapped to typewell scale

        def oracle(center_arr, halfwin):
            pred = center_arr.copy()
            for i in range(n):
                if not has[i]: continue
                lo,hi = center_arr[i]-halfwin, center_arr[i]+halfwin
                m = (grid>=lo)&(grid<=hi)
                if not m.any(): continue
                cg = grid_gr[m]; ct = grid[m]
                consistent = np.abs(cg - tw_equiv[i]) <= GR_TOL
                cand = ct[consistent] if consistent.any() else ct[[np.argmin(np.abs(cg-tw_equiv[i]))]]
                pred[i] = cand[np.argmin(np.abs(cand - out["TVT"][i]))]  # ORACLE pick
            return pred
        def nearest(center_arr, halfwin):
            pred = center_arr.copy()
            for i in range(n):
                if not has[i]: continue
                lo,hi = center_arr[i]-halfwin, center_arr[i]+halfwin
                m=(grid>=lo)&(grid<=hi)
                if not m.any(): continue
                cg=grid_gr[m]; ct=grid[m]
                pred[i]=ct[np.argmin(np.abs(cg-tw_equiv[i]))]
            return pred

        anc = np.full(n,anchor)
        for w in WINDOWS: out[f"oracle_w{int(w)}"]=oracle(anc, w)
        out["nearest_w60"]=nearest(anc, 60.0)
        out["oracle_full"]=oracle(anc, 1e9)
        out["oracle_geom"]=oracle(geomp, GEOM_BAND)
        parts.append(pd.DataFrame(out))

    P = pd.concat(parts, ignore_index=True)
    P.to_csv(OUT/"rows.csv", index=False)
    res = {"anchor": tvt_rmse(P.TVT,P.anchor),
           "geom_exp014": tvt_rmse(P.TVT,P.geom),
           "nearest_gr_w60": tvt_rmse(P.TVT,P.nearest_w60),
           "oracle_full_range": tvt_rmse(P.TVT,P.oracle_full),
           "oracle_geomprior_band25": tvt_rmse(P.TVT,P.oracle_geom)}
    for w in WINDOWS: res[f"oracle_window_{int(w)}"]=tvt_rmse(P.TVT,P[f"oracle_w{int(w)}"])
    print("\n=== GR matching ceiling (全target行 RMSE) ===")
    for k,v in res.items(): print(f"  {k:28s} = {v:.4f}")
    (OUT/"result.json").write_text(json.dumps({"created":now_jst(),"gr_tol":GR_TOL,
        "rmse":res},indent=2))
    print("\nsaved ->", OUT)

if __name__=="__main__":
    main()
