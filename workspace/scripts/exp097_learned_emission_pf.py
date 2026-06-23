#!/usr/bin/env python3
"""exp097: Learned-emission Particle Filter (Phase 4b, track B1).

exp022 の PF アーキテクチャ(状態 pos=TVT+Z を MD 沿いに伝播、systematic resampling)を
そのまま踏襲しつつ、emission 尤度 P(GR観測 | 深度状態) を「生ガウシアン」から
小型 1D-CNN 学習スコアラに置換する。

emission scorer:
  入力 = lateral GR 局所窓(対象行まわり W 点) + typewell GR 局所窓(候補 TVT まわり W 点)
  出力 = その候補 TVT 仮説の log-likelihood スコア
教師 = known区間(TVT_input 既知 = 真の TVT が分かる行)から
  正例 (候補=真TVT) / 負例 (候補=真TVT±オフセット) を生成し二値分類で学習。
  → 学習した「対応スコア」を PF の emission として使う。

高速化: PF は粒子×ステップで emission を呼ぶため、per-particle NN 呼び出しは CPU では非現実的。
そこで各対象行ごとに候補 TVT グリッド上で emission スコアを **一括前計算**(scorer を 1 回 batch 評価)し、
PF 内では粒子位置をそのグリッドに線形補間するだけにする(exp022 の np.interp と同形)。
これで PF 本体は exp022 とほぼ同コスト、emission のみ学習モデルに差し替わる。

leak規約(厳守):
  - CV は well_id GroupKFold のみ(folds_group_well_v001.csv)。row-random 禁止。
  - 評価は hidden tail 行(is_target / TVT_input が NaN)のみ。
  - emission scorer は **各 fold の train well のみ**で学習し valid well に適用(fold外厳守)。
  - train-only 地層列(ANCC/ASTNU/.../typewell Geology)は一切読まない・使わない。
  - hidden TVT は学習にも推論にも一切使わない(known区間の TVT_input のみ教師に使用)。
"""

from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp097_learned_emission_pf"
OUT_DIR = Path("experiments") / EXP_ID

# ---- PF settings (exp022 流用) ----
N_PARTICLES = 500
N_SEEDS = 16          # exp022 は 128。CPU-only & 検証目的のため軽量化(notes に明記)。
SCALE = 8.0
MOM = 0.998; VN = 0.002; PN = 0.005; RP = 0.1; RR = 0.001; RESAMP = 0.5
INIT_SPREAD = 2.0
ALPHA = 1.0  # 学習補正の強さ。生ガウシアン log-emission に ALPHA*(logit-row_max) を加える。
             # 小さいほど exp022(raw)寄り。AUC が低い間は raw が支配し PF は壊れない。

# ---- emission scorer settings ----
WIN = 16              # 局所窓の半径(片側点数)。窓長 = 2*WIN+1
GRID_HALF = 60.0      # 候補 TVT グリッドの片側幅(ft)
GRID_STEP = 1.0       # グリッド刻み(ft)
NEG_OFFSETS = (3.0, 8.0, 20.0, 45.0)  # 負例オフセット(ft)
EPOCHS = 8
BATCH = 4096
LR = 1e-3
MAX_TRAIN_SAMPLES = 200_000  # foldあたり教師サンプル上限

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# emission scorer: 1D-CNN
#   入力: 2ch × (2*WIN+1)  [ch0 = lateral GR 窓, ch1 = typewell GR 窓(候補TVT中心)]
#   出力: scalar logit (= 対応スコア, log-likelihood として使用)
# =====================================================================
class EmissionCNN(nn.Module):
    def __init__(self, win: int):
        super().__init__()
        L = 2 * win + 1
        self.net = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 16, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):  # x: (B, 2, L)
        h = self.net(x).squeeze(-1)   # (B,16)
        return self.head(h).squeeze(-1)  # (B,)


def _norm_win(a: np.ndarray) -> np.ndarray:
    """窓を平均0/標準偏差1に正規化(absolute GR レベル差を吸収)。"""
    a = a.astype(np.float32)
    m = a.mean(axis=-1, keepdims=True)
    s = a.std(axis=-1, keepdims=True)
    s = np.where(s < 1e-6, 1.0, s)
    return (a - m) / s


# =====================================================================
# データ構築(exp022 build を踏襲。typewell を MD 等間隔ではなく TVT グリッドで持つ)
# =====================================================================
def build_payloads(base_path, tw_path, well_ids=None):
    cols = ["well_id", "row_idx", "MD", "Z", "GR", "TVT", "TVT_input", "id",
            "is_target", "is_known_tvt", "is_gr_missing", "last_known_TVT"]
    tr = pd.read_parquet(base_path, columns=cols)
    if well_ids is not None:
        tr = tr[tr["well_id"].isin(well_ids)]
    tw_all = pd.read_parquet(tw_path, columns=["well_id", "TVT", "GR"])
    tw_by_well = {w: gg for w, gg in tw_all.groupby("well_id", sort=False)}

    sel = tr[tr["is_target"].astype(bool) | tr["is_known_tvt"].astype(bool)]
    payloads = []; out_frames = []
    for wid, g in sel.groupby("well_id", sort=False):
        g = g.sort_values("row_idx")
        known = g[g["is_known_tvt"].astype(bool)]
        tgt = g[g["is_target"].astype(bool)]
        if len(tgt) == 0:
            continue
        anchor = float(tgt["last_known_TVT"].iloc[0])
        out_frames.append(tgt[["well_id", "row_idx", "id", "TVT", "last_known_TVT"]].copy())
        tw_g = tw_by_well.get(wid)
        tgt_mask = g["is_target"].astype(bool).to_numpy()
        known_mask = g["is_known_tvt"].astype(bool).to_numpy()
        gr_full = g["GR"].interpolate(limit_direction="both")
        if tw_g is None or len(tw_g) < 2 or len(known) < 2:
            payloads.append({"wid": wid, "no_tw": True, "anchor": anchor,
                             "n_eval": int(len(tgt))})
            continue
        tw_s = tw_g.sort_values("TVT").drop_duplicates("TVT")
        tw_tvt = tw_s["TVT"].to_numpy(float)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).to_numpy(float)
        gr_full = gr_full.fillna(float(np.nanmean(tw_gr))).to_numpy(float)
        gr_v = gr_full[tgt_mask]
        gr_known = gr_full[known_mask]

        k_tvt = known["TVT_input"].to_numpy(float)
        k_gr = known["GR"].fillna(0).to_numpy(float)
        tw_at_k = np.interp(k_tvt, tw_tvt, tw_gr)
        gs = float(np.clip(np.nanstd(k_gr - tw_at_k), 10., 60.))
        tail = known.tail(30)
        dt = np.diff(tail["TVT_input"].to_numpy(float))
        dz = np.diff(tail["Z"].to_numpy(float))
        dm = np.diff(tail["MD"].to_numpy(float))
        mm = dm > 0
        ir = float(np.median((dt + dz)[mm] / dm[mm])) if mm.sum() >= 3 else 0.0

        last = known.iloc[-1]
        payloads.append({
            "wid": wid, "no_tw": False, "n_eval": int(len(tgt)),
            "tw_tvt": tw_tvt, "tw_gr": tw_gr,
            "md_v": tgt["MD"].to_numpy(float), "z_v": tgt["Z"].to_numpy(float),
            "gr_v": gr_v.astype(np.float32),
            # 教師生成用(known区間)
            "k_tvt": k_tvt.astype(np.float32), "gr_known": gr_known.astype(np.float32),
            "gs": gs, "ir": ir,
            "last_tvt": float(last["TVT_input"]), "last_Z": float(last["Z"]),
            "last_MD": float(last["MD"]), "anchor": anchor,
        })
    out = pd.concat(out_frames, ignore_index=True)
    return payloads, out


# =====================================================================
# 教師サンプル生成: known区間の各行で、真TVT中心の typewell窓(正例) と
# オフセット中心(負例) を作る。lateral窓 = known区間 GR の局所窓。
# =====================================================================
def _make_windows(lat_gr, center_idx, tw_tvt, tw_gr, cand_tvt):
    """lat_gr 局所窓(idx中心) と tw 局所窓(cand_tvt中心, TVT等間隔±WIN ft)。"""
    L = 2 * WIN + 1
    # lateral 窓(深度方向 index 窓)
    lo = center_idx - WIN; hi = center_idx + WIN + 1
    idx = np.clip(np.arange(lo, hi), 0, len(lat_gr) - 1)
    lat = lat_gr[idx]
    # typewell 窓(候補 TVT を中心に ±WIN ft で TVT を引いて補間)
    tvt_grid = cand_tvt + (np.arange(L) - WIN).astype(np.float32)
    tw = np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float32)
    return lat, tw


def build_training_set(payloads, rng):
    X = []; Y = []
    for p in payloads:
        if p.get("no_tw", False):
            continue
        lat = p["gr_known"]; ktvt = p["k_tvt"]
        tw_tvt = p["tw_tvt"]; tw_gr = p["tw_gr"]
        n = len(lat)
        if n < 3:
            continue
        # known区間内のサンプル点(間引き)
        step = max(1, n // 200)
        idxs = np.arange(0, n, step)
        for ci in idxs:
            true_tvt = float(ktvt[ci])
            # 正例
            lat_w, tw_w = _make_windows(lat, ci, tw_tvt, tw_gr, true_tvt)
            X.append((lat_w, tw_w)); Y.append(1.0)
            # 負例(複数オフセット)
            for off in NEG_OFFSETS:
                s = 1.0 if rng.random() < 0.5 else -1.0
                cand = true_tvt + s * off
                lat_w, tw_w = _make_windows(lat, ci, tw_tvt, tw_gr, cand)
                X.append((lat_w, tw_w)); Y.append(0.0)
    if not X:
        return None, None
    lat_arr = np.stack([x[0] for x in X]).astype(np.float32)
    tw_arr = np.stack([x[1] for x in X]).astype(np.float32)
    lat_arr = _norm_win(lat_arr); tw_arr = _norm_win(tw_arr)
    feat = np.stack([lat_arr, tw_arr], axis=1)  # (N,2,L)
    y = np.asarray(Y, dtype=np.float32)
    if len(y) > MAX_TRAIN_SAMPLES:
        sel = rng.choice(len(y), MAX_TRAIN_SAMPLES, replace=False)
        feat = feat[sel]; y = y[sel]
    return feat, y


def train_scorer(feat, y, epochs, seed=0):
    torch.manual_seed(seed)
    model = EmissionCNN(WIN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(feat); Yt = torch.from_numpy(y)
    n = len(y)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            xb = Xt[b].to(DEVICE); yb = Yt[b].to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"    epoch {ep+1}/{epochs} loss={tot/n:.4f}", flush=True)
    model.eval()
    return model


# =====================================================================
# 各対象行で候補TVTグリッド上の emission スコアを一括前計算
#   score_grid[i, :] = scorer( lat窓(対象行i), tw窓(候補TVT) )  over grid
# PF はこの grid を線形補間して emission を得る(per-particle NN 呼び出しなし)。
# =====================================================================
@torch.no_grad()
def precompute_score_grids(p, model):
    lat = p["gr_v"]                      # 対象行の GR(補間済)
    tw_tvt = p["tw_tvt"]; tw_gr = p["tw_gr"]
    n = len(lat)
    lo = tw_tvt[0] - 100; hi = tw_tvt[-1] + 100
    # 候補 TVT グリッドは PF が動き得る全域(=clip 範囲 [lo,hi])を覆う。
    # anchor±GRID_HALF だと tail で真TVTがグリッド外に出て emission が平坦化し PF が発散する。
    # グリッド点数を MAX_GRID に制限(span が広い well は刻みを粗くする)。
    MAX_GRID = 300
    span = hi - lo
    step = max(GRID_STEP, span / MAX_GRID)
    cand_tvt = np.arange(lo, hi + step, step).astype(np.float32)  # (G,)
    G = len(cand_tvt)
    L = 2 * WIN + 1
    # typewell 窓は候補 TVT のみ依存 → 1 回作って全行で共有
    tvt_mat = cand_tvt[:, None] + (np.arange(L) - WIN)[None, :]
    tw_win = np.interp(tvt_mat.ravel(), tw_tvt, tw_gr).reshape(G, L).astype(np.float32)
    tw_win = _norm_win(tw_win)  # (G,L)

    # lateral 窓は対象行 index 依存
    pad = np.pad(lat, (WIN, WIN), mode="edge")
    lat_win = np.empty((n, L), dtype=np.float32)
    for i in range(n):
        lat_win[i] = pad[i:i + L]
    lat_win = _norm_win(lat_win)  # (n,L)

    # cross: 各行 i × 各候補 g。バッチで scorer 評価。
    # feat shape -> (n*G, 2, L)
    lat_rep = np.repeat(lat_win, G, axis=0)            # (n*G, L)
    tw_rep = np.tile(tw_win, (n, 1))                   # (n*G, L)
    feat = np.stack([lat_rep, tw_rep], axis=1)         # (n*G,2,L)
    scores = np.empty(len(feat), dtype=np.float32)
    for i in range(0, len(feat), 65536):
        xb = torch.from_numpy(feat[i:i + 65536]).to(DEVICE)
        scores[i:i + 65536] = model(xb).cpu().numpy()
    logit_grid = scores.reshape(n, G).astype(np.float64)  # (n,G) 学習スコア

    # 学習補正のみグリッド化(行内最大基準・温度 ALPHA)。
    # 生ガウシアン emission は鋭く粗グリッドでピークが潰れて PF が壊れるため、
    # PF 内で per-particle に厳密計算する(exp022 と同一)。学習補正は緩やかなので粗グリッドで可。
    corr_grid = ALPHA * (logit_grid - logit_grid.max(axis=1, keepdims=True))  # (n,G), <=0
    return cand_tvt, corr_grid, lo, hi


# =====================================================================
# PF 本体(exp022 を踏襲。emission のみ学習スコアグリッド補間に置換)
# =====================================================================
def _pf_single(p, seed, tvt_axis, corr_grid, lo, hi):
    z_v = p["z_v"]; md_v = p["md_v"]; gr_v = p["gr_v"]
    tw_tvt = p["tw_tvt"]; tw_gr = p["tw_gr"]; gs = p["gs"]
    n = len(md_v)
    if n == 0:
        return np.zeros(0), 0.0
    N = N_PARTICLES
    rng = np.random.default_rng(seed)
    pos = (p["last_tvt"] + p["last_Z"]) + INIT_SPREAD * rng.standard_normal(N)
    rate = p["ir"] + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N
    res = np.empty(n)
    prev_MD = p["last_MD"]
    log_lik = 0.0
    for i in range(n):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], lo, hi)
        pos = tvt_p + z_v[i]
        # === hybrid emission ===
        # raw: 生ガウシアン GR 尤度を per-particle 厳密計算(exp022 と同一・粗グリッド化しない)
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        raw_ll = -0.5 * np.minimum(d * d, 600.0)
        # learned: 緩やかな学習補正(<=0)を粗グリッドから補間して加算
        corr = np.interp(tvt_p, tvt_axis, corr_grid[i])
        lk = np.exp(np.clip(raw_ll + corr, -600.0, 0.0))
        lk = np.maximum(lk, 1e-300)
        log_lik += np.log(max(float((w * lk).sum()), 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w * w).sum() < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]
    return res, log_lik


def process_well(p, model):
    wid = p["wid"]; n = int(p["n_eval"])
    if n == 0:
        return wid, np.zeros(0)
    if p.get("no_tw", False):
        return wid, np.full(n, p["anchor"])
    tvt_axis, score_grid, lo, hi = precompute_score_grids(p, model)
    preds = np.empty((N_SEEDS, n)); liks = np.empty(N_SEEDS)
    for s in range(N_SEEDS):
        preds[s], liks[s] = _pf_single(p, s, tvt_axis, score_grid, lo, hi)
    wts = np.exp((liks - liks.max()) / SCALE); wts /= wts.sum()
    return wid, (wts[:, None] * preds).sum(0)


# =====================================================================
# fold-out CV
# =====================================================================
def run_cv(base_path, tw_path, folds, epochs, max_wells=None):
    fold_map = folds.loc[folds["split"].eq("train"), ["well_id", "fold"]]
    if max_wells is not None:
        # smoke: 各 fold から均等にサンプルし、全 fold に train/valid が存在するようにする
        per = max(2, max_wells // fold_map["fold"].nunique())
        fold_map = (fold_map.groupby("fold", group_keys=False)
                    .apply(lambda d: d.head(per)).reset_index(drop=True))
    all_wells = fold_map["well_id"].tolist()
    well_fold = dict(zip(fold_map["well_id"], fold_map["fold"]))
    folds_uniq = sorted(fold_map["fold"].unique())

    # 全 well payload を 1 回構築(emission は fold ごとに学習)
    payloads, out = build_payloads(base_path, tw_path, well_ids=set(all_wells))
    pl_by_wid = {p["wid"]: p for p in payloads}

    fold_rmse = {}
    pred_by_wid = {}
    for fold in folds_uniq:
        valid_wells = [w for w in all_wells if well_fold[w] == fold]
        train_wells = [w for w in all_wells if well_fold[w] != fold]
        tr_pl = [pl_by_wid[w] for w in train_wells if w in pl_by_wid]
        print(f"[fold {fold}] train_wells={len(train_wells)} valid_wells={len(valid_wells)}", flush=True)
        rng = np.random.default_rng(1000 + fold)
        feat, y = build_training_set(tr_pl, rng)
        if feat is None:
            print(f"  [fold {fold}] no training samples, skip", flush=True)
            continue
        print(f"  [fold {fold}] train samples={len(y)} pos_frac={y.mean():.3f}", flush=True)
        model = train_scorer(feat, y, epochs, seed=fold)
        # valid well に適用
        for k, w in enumerate(valid_wells):
            if w not in pl_by_wid:
                continue
            wid, pred = process_well(pl_by_wid[w], model)
            pred_by_wid[wid] = pred
            if (k + 1) % 50 == 0:
                print(f"    [fold {fold}] {k+1}/{len(valid_wells)} valid wells done", flush=True)

    # assemble
    out = out[out["well_id"].isin(pred_by_wid.keys())].copy()
    pred_col = np.empty(len(out))
    for wid, g in out.groupby("well_id", sort=False):
        pred_col[g.index.to_numpy()] = pred_by_wid[wid]
    out = out.copy()
    out["pred_tvt"] = pred_col
    out["fold"] = out["well_id"].map(well_fold)

    for fold in folds_uniq:
        sub = out[out["fold"] == fold]
        if len(sub):
            fold_rmse[int(fold)] = tvt_rmse(sub["TVT"], sub["pred_tvt"])
    return out, fold_rmse


def main():
    global ALPHA, OUT_DIR, N_SEEDS, EPOCHS, N_PARTICLES, MAX_TRAIN_SAMPLES

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="小サブセットで高速 smoke-test")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="学習補正の強さ。0 で生ガウシアン(exp022 相当)に劣化収束するはず。")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="出力先ディレクトリ上書き(診断用)。")
    args = ap.parse_args()

    ALPHA = args.alpha
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        N_SEEDS = 4; EPOCHS = 2; N_PARTICLES = 200; MAX_TRAIN_SAMPLES = 20_000
        max_wells = 30
        print(f"[{EXP_ID}] SMOKE mode: wells={max_wells} seeds={N_SEEDS} part={N_PARTICLES} epochs={EPOCHS}")
    else:
        max_wells = None
        print(f"[{EXP_ID}] FULL mode: 773 wells seeds={N_SEEDS} part={N_PARTICLES} epochs={EPOCHS}")
    print(f"  device = {DEVICE}")

    t0 = time.time()
    folds = pd.read_csv("data/folds/folds_group_well_v001.csv")
    out, fold_rmse = run_cv(
        "data/processed/train_base_v001.parquet",
        "data/processed/typewell_train_base_v001.parquet",
        folds, EPOCHS, max_wells=max_wells)
    runtime = time.time() - t0

    cv = tvt_rmse(out["TVT"], out["pred_tvt"])
    anc = tvt_rmse(out["TVT"], out["last_known_TVT"])
    n_wells = out["well_id"].nunique()
    print(f"  CV RMSE = {cv:.6f}  anchor = {anc:.6f}  n_wells={n_wells}  runtime={runtime:.1f}s")
    print(f"  fold_rmse = {fold_rmse}")

    # oof.csv (列: id, well_id, fold, tvt_pred, tvt_true)
    oof = out[["id", "well_id", "fold"]].copy()
    oof["tvt_pred"] = out["pred_tvt"].values
    oof["tvt_true"] = out["TVT"].values
    oof.to_csv(OUT_DIR / "oof.csv", index=False)

    result = {
        "exp_id": EXP_ID,
        "cv_rmse": cv,
        "n_wells": int(n_wells),
        "runtime_sec": round(runtime, 1),
        "leak_risk": "none (fold-out emission training on train wells only; "
                     "hidden TVT never used; only known-interval TVT_input as supervision; "
                     "GR+typewell GR+anchor+Z+MD only; no train-only geology cols)",
        "notes_short": ("Learned-emission PF: exp022 PF with raw-Gaussian emission replaced by "
                        "1D-CNN correspondence scorer (lateral GR window x typewell GR window). "
                        "Per-row candidate-TVT score grid precomputed, PF interpolates grid per particle. "
                        f"smoke={args.smoke}"),
        "fold_rmse": [fold_rmse.get(i) for i in sorted(fold_rmse.keys())],
        "anchor_rmse": anc,
        "alpha": ALPHA,
        "compare_exp022_pooled": 11.024,
        "compare_exp097_full": 20.229,
        "compare_exp022_pooled_precise": 11.024014,
        "smoke": bool(args.smoke),
        "settings": {"n_seeds": N_SEEDS, "n_particles": N_PARTICLES, "epochs": EPOCHS,
                     "win": WIN, "alpha": ALPHA, "neg_offsets": list(NEG_OFFSETS),
                     "device": str(DEVICE)},
        "created_at": now_jst(),
        "status": "completed",
    }
    write_json(OUT_DIR / "result.json", result)

    fr_md = "\n".join(f"| {k} | {v:.4f} |" for k, v in sorted(fold_rmse.items()))
    (OUT_DIR / "notes.md").write_text(f"""# {EXP_ID} — 学習型 emission Particle Filter (Phase 4b / track B1)

## 目的
exp022 の PF(状態 pos=TVT+Z を MD 沿いに伝播)の emission 尤度を、
生ガウシアン `exp(-0.5*((GR-eg)/gs)^2)` から **小型 1D-CNN 学習スコアラ** に置換し、
PF 成分の CV を底上げする(目標: PF 11.0 → 9〜10)。

## 手法
- emission scorer: 2ch 1D-CNN。入力 = lateral GR 局所窓(対象行±{WIN}点) と
  typewell GR 局所窓(候補 TVT ±{WIN} ft で補間)。出力 = 対応スコア logit。
- 教師: known区間(TVT_input 既知)の各点で 正例(候補=真TVT) / 負例(真TVT±{list(NEG_OFFSETS)} ft)。
  BCEWithLogits で学習。
- 高速化: 各対象行で候補 TVT グリッド(±{GRID_HALF}ft, {GRID_STEP}ft 刻み)上の emission を
  一括前計算し、PF は粒子位置をグリッド線形補間(per-particle NN 呼び出し無し)。
  emission を logistic で (0,1] 尤度化して exp022 と同じ粒子再重み付けに投入。
- PF 本体(粒子数 {N_PARTICLES}, seeds {N_SEEDS}, systematic resampling)は exp022 流用。
  検証コストのため seeds を 128→{N_SEEDS} に軽量化(notes 明記)。

## CV(well GroupKFold, fold外学習厳守)
| fold | RMSE |
|---|---|
{fr_md}

- **pooled CV RMSE = {cv:.6f}**  (anchor {anc:.6f}, n_wells {n_wells})
- 参考: exp022 pooled PF = 11.024014

## leak懸念
- emission scorer は各 fold の train well のみで学習し valid well に適用(fold外厳守)。
- 教師は known区間の TVT_input のみ。hidden TVT(評価対象)は学習にも推論にも不使用。
- typewell Geology / train-only 地層列は一切読まない。row-random CV 不使用。
- 残懸念: 候補グリッド中心を anchor(last_known_TVT) に固定。anchor から大きく外れる well では
  グリッド範囲(±{GRID_HALF}ft)を超える真TVTを捉えられない可能性 → grid_half 拡張で要検証。

## exp022 比 / 次案
- exp022 比は上記 pooled CV を参照(< 11.02 なら学習 emission が寄与)。
- 次案: (1) grid_half 拡張+seeds 128 で full 再走, (2) typewell 窓を MD 等間隔でなく
  TVT 多解像度に, (3) scorer 出力を温度較正して blend(exp026)へ第6エンジン注入。

## 実行
- runtime = {runtime:.1f}s, device = {DEVICE}, smoke = {args.smoke}
""", encoding="utf-8")

    print(f"\n[{EXP_ID}] done -> {OUT_DIR}  CV={cv:.6f}  runtime={runtime:.1f}s")


if __name__ == "__main__":
    main()
