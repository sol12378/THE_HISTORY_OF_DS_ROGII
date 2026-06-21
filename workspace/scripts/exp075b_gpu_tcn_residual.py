#!/usr/bin/env python
"""exp075b: GPU TCN learning the RESIDUAL on the geom prior (decorrelated source).

Difference vs exp075:
  - target = TVT - geom_oof_pred   (learn what geom misses)
  - geom prior (relative to anchor) added as an input feature
  - final pred_tvt = geom_oof_pred + residual_pred
Leak-free: geom_oof is out-of-fold; TCN uses GroupKFold by well; loss on hidden rows only.

Output: experiments/exp073_public_assets_integration/oof_tcn_resid.csv
        columns [id, well_id, row_idx, tvt_true, pred_tvt]
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
OUT = ROOT / "experiments" / "exp073_public_assets_integration"
TRAIN_BASE = ROOT / "data" / "processed" / "train_base_v001.parquet"
GEOM_OOF = OUT / "oof_geom.csv"
LOG = OUT / "log_tcn_resid.txt"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
MAX_EPOCHS = 22
PATIENCE = 5
LR = 1.5e-3
BATCH_WELLS = 16
CH = 64
DILATIONS = [1, 2, 4, 8, 16, 32, 64]
SEED = 42


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_sequences(train, geom_pred_by_id):
    train = train.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    train["id"] = train["well_id"] + "_" + train["row_idx"].astype(str)
    g = train.groupby("well_id", sort=False)

    gr = train["GR"].astype(float).groupby(train["well_id"]).transform(
        lambda s: s.interpolate(limit_direction="both"))
    global_gr_mean = float(gr.mean())
    gr = gr.fillna(global_gr_mean)
    train["_gr"] = gr.to_numpy()
    train["_gr_d1"] = g["_gr"].transform(lambda s: s.diff().fillna(0.0))
    train["_gr_rm"] = g["_gr"].transform(lambda s: s.rolling(20, min_periods=1, center=True).mean())
    train["_gr_rs"] = g["_gr"].transform(lambda s: s.rolling(20, min_periods=1, center=True).std().fillna(0.0))

    lk = train["last_known_TVT"].astype(float).to_numpy()
    train["_tvt_in_rel"] = np.where(train["is_known_tvt"].to_numpy(),
                                    train["TVT_input"].astype(float).to_numpy() - lk, 0.0)
    train["_md_since"] = np.log1p(np.clip(train["MD"].astype(float) - train["last_known_MD"].astype(float), 0, None))
    train["_is_known"] = train["is_known_tvt"].astype(float)
    dz = train["delta_Z_from_PS"].astype(float).to_numpy()
    train["_dxy"] = np.sqrt(train["delta_X_from_PS"].astype(float) ** 2 + train["delta_Y_from_PS"].astype(float) ** 2)
    dmd = (train["MD"].astype(float) - train["last_known_MD"].astype(float)).to_numpy()
    train["_dz_dmd"] = np.where(np.abs(dmd) > 1e-6, dz / dmd, 0.0)

    # geom prior (TVT space) per row: hidden rows from OOF, known rows = anchor
    gp = train["id"].map(geom_pred_by_id).to_numpy()
    gp = np.where(np.isfinite(gp), gp, lk)  # known/missing -> anchor
    train["_geom_rel"] = gp - lk  # relative to anchor

    feat_cols = ["_gr", "_gr_d1", "_gr_rm", "_gr_rs", "Z", "delta_Z_from_PS",
                 "delta_X_from_PS", "delta_Y_from_PS", "_md_since", "row_frac",
                 "_tvt_in_rel", "_is_known", "_dxy", "_dz_dmd", "_geom_rel"]
    X = train[feat_cols].astype(np.float32).to_numpy()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd

    tvt = train["TVT"].astype(float).to_numpy()
    resid = tvt - gp                      # target = residual on geom prior
    is_hidden = (~train["is_known_tvt"]).to_numpy()
    well_ids = train["well_id"].to_numpy()
    row_idx = train["row_idx"].to_numpy()

    seqs, start = [], 0
    for wid, gdf in g:
        n = len(gdf); sl = slice(start, start + n)
        seqs.append({"wid": wid, "X": X[sl], "y": resid[sl].astype(np.float32),
                     "hidden": is_hidden[sl], "row_idx": row_idx[sl],
                     "geom": gp[sl], "tvt_true": tvt[sl]})
        start += n
    return seqs, feat_cols


class TCN(nn.Module):
    def __init__(self, n_feat, ch=CH, dils=DILATIONS, k=3, drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, ch, 1)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "c1": nn.Conv1d(ch, ch, k, padding=(k - 1) * d // 2, dilation=d),
            "c2": nn.Conv1d(ch, ch, k, padding=(k - 1) * d // 2, dilation=d),
            "bn1": nn.BatchNorm1d(ch), "bn2": nn.BatchNorm1d(ch)}) for d in dils])
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.GELU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            r = h
            h = torch.relu(b["bn1"](b["c1"](h)))
            h = self.drop(h)
            h = b["bn2"](b["c2"](h))
            h = torch.relu(h + r)
        return self.head(h).squeeze(1)


def make_batches(idxs, seqs, shuffle, rng):
    order = sorted(idxs, key=lambda i: len(seqs[i]["row_idx"]))
    batches = [order[i:i + BATCH_WELLS] for i in range(0, len(order), BATCH_WELLS)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(batch_idx, seqs):
    L = max(len(seqs[i]["row_idx"]) for i in batch_idx)
    B, nf = len(batch_idx), seqs[batch_idx[0]]["X"].shape[1]
    X = np.zeros((B, L, nf), np.float32); y = np.zeros((B, L), np.float32); hm = np.zeros((B, L), np.float32)
    for b, i in enumerate(batch_idx):
        s = seqs[i]; n = len(s["row_idx"])
        X[b, :n] = s["X"]; y[b, :n] = s["y"]; hm[b, :n] = s["hidden"].astype(np.float32)
    return torch.from_numpy(X).permute(0, 2, 1), torch.from_numpy(y), torch.from_numpy(hm)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    open(LOG, "w").close()
    t0 = time.time()
    log(f"device={DEV}")
    geom = pd.read_csv(GEOM_OOF, usecols=["id", "pred_tvt"])
    geom_pred_by_id = dict(zip(geom["id"].astype(str), geom["pred_tvt"].astype(float)))
    log(f"geom OOF loaded: {len(geom_pred_by_id):,} ids")
    train = pd.read_parquet(TRAIN_BASE)
    seqs, feat_cols = build_sequences(train, geom_pred_by_id)
    log(f"built {len(seqs)} seqs, {len(feat_cols)} features")
    n_feat = len(feat_cols)
    wells = np.array([s["wid"] for s in seqs])

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred, fold_rmses = {}, []
    rng = np.random.default_rng(SEED)
    for fold, (tr, va) in enumerate(gkf.split(wells, groups=wells)):
        model = TCN(n_feat).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=MAX_EPOCHS * ((len(tr) + BATCH_WELLS - 1) // BATCH_WELLS))
        scaler = torch.cuda.amp.GradScaler()
        huber = nn.HuberLoss(delta=8.0, reduction="none")
        best, best_state, bad = 1e9, None, 0
        for ep in range(MAX_EPOCHS):
            model.train()
            for bidx in make_batches(tr, seqs, True, rng):
                Xt, yt, hm = collate(bidx, seqs)
                Xt, yt, hm = Xt.to(DEV), yt.to(DEV), hm.to(DEV)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = model(Xt)
                    loss = (huber(pred, yt) * hm).sum() / hm.sum().clamp_min(1)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt); scaler.update(); sched.step()
            # validate on final TVT (geom + resid)
            model.eval(); se = cnt = 0
            with torch.no_grad(), torch.cuda.amp.autocast():
                for bidx in make_batches(va, seqs, False, rng):
                    Xt, _, _ = collate(bidx, seqs)
                    pred = model(Xt.to(DEV)).float().cpu().numpy()
                    for b, i in enumerate(bidx):
                        s = seqs[i]; n = len(s["row_idx"]); m = s["hidden"]
                        ptvt = pred[b, :n][m] + s["geom"][m]
                        se += float(np.sum((ptvt - s["tvt_true"][m]) ** 2)); cnt += int(m.sum())
            vr = (se / max(cnt, 1)) ** 0.5
            if vr < best - 1e-4:
                best, bad = vr, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            log(f"  fold{fold} ep{ep} val_tvt_rmse={vr:.4f} best={best:.4f}")
            if bad >= PATIENCE:
                break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for bidx in make_batches(va, seqs, False, rng):
                Xt, _, _ = collate(bidx, seqs)
                pred = model(Xt.to(DEV)).float().cpu().numpy()
                for b, i in enumerate(bidx):
                    s = seqs[i]; n = len(s["row_idx"]); m = s["hidden"]
                    ptvt = pred[b, :n][m] + s["geom"][m]
                    for rj, pv in zip(s["row_idx"][m], ptvt):
                        oof_pred[f"{s['wid']}_{int(rj)}"] = float(pv)
        fold_rmses.append(best)
        log(f"fold{fold} done best={best:.4f} elapsed={(time.time()-t0)/60:.1f}min")

    rows = []
    for s in seqs:
        m = s["hidden"]
        for rj, tv in zip(s["row_idx"][m], s["tvt_true"][m]):
            iid = f"{s['wid']}_{int(rj)}"
            rows.append((iid, s["wid"], int(rj), float(tv), oof_pred.get(iid, np.nan)))
    oof = pd.DataFrame(rows, columns=["id", "well_id", "row_idx", "tvt_true", "pred_tvt"])
    valid = oof["pred_tvt"].notna()
    pooled = float(np.sqrt(np.mean((oof.loc[valid, "tvt_true"] - oof.loc[valid, "pred_tvt"]) ** 2)))
    oof.to_csv(OUT / "oof_tcn_resid.csv", index=False)
    with open(OUT / "result_tcn_resid.json", "w", encoding="utf-8") as f:
        json.dump({"exp": "exp075b_gpu_tcn_residual", "pooled_rmse": pooled,
                   "fold_val_rmses": fold_rmses, "n_rows": int(len(oof)),
                   "n_valid": int(valid.sum()), "features": feat_cols,
                   "runtime_min": (time.time() - t0) / 60, "device": str(DEV),
                   "leak_notes": "residual on geom OOF (out-of-fold); GroupKFold by well; loss on hidden only."},
                  f, indent=2, ensure_ascii=False)
    log(f"DONE pooled_rmse={pooled:.4f} rows={len(oof)} runtime={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
