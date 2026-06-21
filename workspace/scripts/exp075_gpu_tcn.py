#!/usr/bin/env python
"""exp075: GPU sequence TCN -> TVT delta OOF (decorrelated blend source).

Leak-free:
  - GroupKFold(5) by well -> honest OOF.
  - Target = TVT - last_known_TVT, loss computed ONLY on hidden rows.
  - Inputs use geometry (X,Y,Z,MD all known for hidden rows), GR (interpolated),
    and TVT_input relative-to-anchor on KNOWN rows only (hidden TVT never fed).
  - Bidirectional dilated convs are valid: only TVT is hidden; the trajectory
    geometry is fully observed, so non-causal context introduces no target leak.

Output: experiments/exp073_public_assets_integration/oof_tcn.csv
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
OUT.mkdir(parents=True, exist_ok=True)
TRAIN_BASE = ROOT / "data" / "processed" / "train_base_v001.parquet"
LOG = OUT / "log_tcn.txt"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5
MAX_EPOCHS = 18
PATIENCE = 4
LR = 1.5e-3
BATCH_WELLS = 16
CH = 64
DILATIONS = [1, 2, 4, 8, 16, 32, 64]
SEED = 42


def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_sequences(train):
    """Return list of per-well dicts with feature matrix, target, mask, ids."""
    train = train.sort_values(["well_id", "row_idx"]).reset_index(drop=True)
    g = train.groupby("well_id", sort=False)

    # per-row leak-free features
    gr = train["GR"].astype(float)
    gr = gr.groupby(train["well_id"]).transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    global_gr_mean = float(gr.mean())
    gr = gr.fillna(global_gr_mean)
    train["_gr"] = gr.to_numpy()
    train["_gr_d1"] = g["_gr"].transform(lambda s: s.diff().fillna(0.0))
    train["_gr_rm"] = g["_gr"].transform(lambda s: s.rolling(20, min_periods=1, center=True).mean())
    train["_gr_rs"] = g["_gr"].transform(lambda s: s.rolling(20, min_periods=1, center=True).std().fillna(0.0))

    lk = train["last_known_TVT"].astype(float).to_numpy()
    train["_tvt_in_rel"] = np.where(
        train["is_known_tvt"].to_numpy(),
        train["TVT_input"].astype(float).to_numpy() - lk,
        0.0,
    )
    train["_md_since"] = np.log1p(np.clip(train["MD"].astype(float) - train["last_known_MD"].astype(float), 0, None))
    train["_is_known"] = train["is_known_tvt"].astype(float)
    dz = train["delta_Z_from_PS"].astype(float).to_numpy()
    train["_dxy"] = np.sqrt(train["delta_X_from_PS"].astype(float) ** 2 + train["delta_Y_from_PS"].astype(float) ** 2)
    dmd = (train["MD"].astype(float) - train["last_known_MD"].astype(float)).to_numpy()
    train["_dz_dmd"] = np.where(np.abs(dmd) > 1e-6, dz / dmd, 0.0)

    feat_cols = [
        "_gr", "_gr_d1", "_gr_rm", "_gr_rs", "Z", "delta_Z_from_PS",
        "delta_X_from_PS", "delta_Y_from_PS", "_md_since", "row_frac",
        "_tvt_in_rel", "_is_known", "_dxy", "_dz_dmd",
    ]
    X = train[feat_cols].astype(np.float32).to_numpy()
    # standardize globally (input normalization; not target leakage)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    X = (X - mu) / sd

    target = (train["TVT"].astype(float).to_numpy() - lk)
    is_hidden = (~train["is_known_tvt"]).to_numpy()
    well_ids = train["well_id"].to_numpy()
    row_idx = train["row_idx"].to_numpy()
    tvt_true = train["TVT"].astype(float).to_numpy()

    seqs = []
    start = 0
    for wid, gdf in g:
        n = len(gdf)
        sl = slice(start, start + n)
        seqs.append({
            "wid": wid,
            "X": X[sl],
            "y": target[sl].astype(np.float32),
            "hidden": is_hidden[sl],
            "row_idx": row_idx[sl],
            "lk": lk[sl],
            "tvt_true": tvt_true[sl],
        })
        start += n
    return seqs, feat_cols


class TCN(nn.Module):
    def __init__(self, n_feat, ch=CH, dils=DILATIONS, k=3, drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_feat, ch, 1)
        blocks = []
        for d in dils:
            pad = (k - 1) * d // 2
            blocks.append(nn.ModuleDict({
                "c1": nn.Conv1d(ch, ch, k, padding=pad, dilation=d),
                "c2": nn.Conv1d(ch, ch, k, padding=pad, dilation=d),
                "bn1": nn.BatchNorm1d(ch),
                "bn2": nn.BatchNorm1d(ch),
            }))
        self.blocks = nn.ModuleList(blocks)
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.GELU(), nn.Conv1d(ch, 1, 1))

    def forward(self, x):  # x: (B, n_feat, L)
        h = self.inp(x)
        for b in self.blocks:
            r = h
            h = torch.relu(b["bn1"](b["c1"](h)))
            h = self.drop(h)
            h = b["bn2"](b["c2"](h))
            h = torch.relu(h + r)
        return self.head(h).squeeze(1)  # (B, L)


def make_batches(idxs, seqs, shuffle, rng):
    order = list(idxs)
    # sort by length to minimize padding, then chunk
    order.sort(key=lambda i: len(seqs[i]["row_idx"]))
    batches = [order[i:i + BATCH_WELLS] for i in range(0, len(order), BATCH_WELLS)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(batch_idx, seqs):
    L = max(len(seqs[i]["row_idx"]) for i in batch_idx)
    B = len(batch_idx)
    nf = seqs[batch_idx[0]]["X"].shape[1]
    X = np.zeros((B, L, nf), np.float32)
    y = np.zeros((B, L), np.float32)
    hm = np.zeros((B, L), np.float32)
    for b, i in enumerate(batch_idx):
        s = seqs[i]
        n = len(s["row_idx"])
        X[b, :n] = s["X"]
        y[b, :n] = s["y"]
        hm[b, :n] = s["hidden"].astype(np.float32)
    Xt = torch.from_numpy(X).permute(0, 2, 1)  # (B, nf, L)
    return Xt, torch.from_numpy(y), torch.from_numpy(hm)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    open(LOG, "w").close()
    t0 = time.time()
    log(f"device={DEV}")
    train = pd.read_parquet(TRAIN_BASE)
    log(f"loaded {len(train):,} rows, {train['well_id'].nunique()} wells")

    seqs, feat_cols = build_sequences(train)
    log(f"built {len(seqs)} well sequences, {len(feat_cols)} features")
    n_feat = len(feat_cols)
    wells = np.array([s["wid"] for s in seqs])

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = {}  # id -> pred delta
    fold_rmses = []
    rng = np.random.default_rng(SEED)

    for fold, (tr, va) in enumerate(gkf.split(wells, groups=wells)):
        model = TCN(n_feat).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=MAX_EPOCHS * ((len(tr) + BATCH_WELLS - 1) // BATCH_WELLS)
        )
        scaler = torch.cuda.amp.GradScaler()
        huber = nn.HuberLoss(delta=10.0, reduction="none")

        best_rmse, best_state, bad = 1e9, None, 0
        for ep in range(MAX_EPOCHS):
            model.train()
            batches = make_batches(tr, seqs, True, rng)
            for bidx in batches:
                Xt, yt, hm = collate(bidx, seqs)
                Xt, yt, hm = Xt.to(DEV), yt.to(DEV), hm.to(DEV)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = model(Xt)
                    loss = (huber(pred, yt) * hm).sum() / hm.sum().clamp_min(1)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

            # validate
            model.eval()
            se, cnt = 0.0, 0
            vb = make_batches(va, seqs, False, rng)
            with torch.no_grad(), torch.cuda.amp.autocast():
                for bidx in vb:
                    Xt, yt, hm = collate(bidx, seqs)
                    Xt = Xt.to(DEV)
                    pred = model(Xt).float().cpu().numpy()
                    for b, i in enumerate(bidx):
                        s = seqs[i]
                        n = len(s["row_idx"])
                        mask = s["hidden"]
                        p = pred[b, :n][mask]
                        t = s["y"][mask]
                        se += float(np.sum((p - t) ** 2))
                        cnt += int(mask.sum())
            vr = (se / max(cnt, 1)) ** 0.5
            if vr < best_rmse - 1e-4:
                best_rmse, bad = vr, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            log(f"  fold{fold} ep{ep} val_rmse={vr:.4f} best={best_rmse:.4f}")
            if bad >= PATIENCE:
                break

        model.load_state_dict(best_state)
        model.eval()
        vb = make_batches(va, seqs, False, rng)
        with torch.no_grad(), torch.cuda.amp.autocast():
            for bidx in vb:
                Xt, yt, hm = collate(bidx, seqs)
                Xt = Xt.to(DEV)
                pred = model(Xt).float().cpu().numpy()
                for b, i in enumerate(bidx):
                    s = seqs[i]
                    n = len(s["row_idx"])
                    mask = s["hidden"]
                    ridx = s["row_idx"][mask]
                    pdelta = pred[b, :n][mask]
                    ptvt = pdelta + s["lk"][mask]
                    for rj, pv in zip(ridx, ptvt):
                        oof_pred[f"{s['wid']}_{int(rj)}"] = float(pv)
        fold_rmses.append(best_rmse)
        log(f"fold{fold} done best_val_rmse={best_rmse:.4f} elapsed={(time.time()-t0)/60:.1f}min")

    # assemble OOF
    rows = []
    for s in seqs:
        mask = s["hidden"]
        for rj, tv in zip(s["row_idx"][mask], s["tvt_true"][mask]):
            iid = f"{s['wid']}_{int(rj)}"
            rows.append((iid, s["wid"], int(rj), float(tv), oof_pred.get(iid, np.nan)))
    oof = pd.DataFrame(rows, columns=["id", "well_id", "row_idx", "tvt_true", "pred_tvt"])
    valid = oof["pred_tvt"].notna()
    pooled = float(np.sqrt(np.mean((oof.loc[valid, "tvt_true"] - oof.loc[valid, "pred_tvt"]) ** 2)))
    oof.to_csv(OUT / "oof_tcn.csv", index=False)
    result = {
        "exp": "exp075_gpu_tcn",
        "pooled_rmse": pooled,
        "fold_val_rmses": fold_rmses,
        "n_rows": int(len(oof)),
        "n_valid": int(valid.sum()),
        "features": feat_cols,
        "runtime_min": (time.time() - t0) / 60,
        "device": str(DEV),
        "leak_notes": "GroupKFold by well; loss on hidden rows only; hidden TVT never fed; bidirectional convs valid (geometry fully observed).",
    }
    with open(OUT / "result_tcn.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"DONE pooled_rmse={pooled:.4f} rows={len(oof)} runtime={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
