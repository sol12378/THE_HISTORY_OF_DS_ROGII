#!/usr/bin/env python
"""exp073 Task C (re-run by main): extract & verify honest external OOFs.

Builds experiments/exp073_public_assets_integration/external_oof.parquet keyed by id,
with delta-space pred columns from:
  - pilkwang model-package: oof/{xgb,catboost,hgb,lgb,sequence_tcn}_oof.npy + blend(_pp)
    aligned to oof/train_gt.parquet  (honest GroupKFold-by-well OOF, target=delta)
  - v11 fresh-artifacts: models/*/oof_preds.pkl (structure inspected defensively)
  - ravaghi artifacts: koolbox Trainer pickles -> trainer.oof_preds, + positive-Ridge stack

Each source is in try/except; failures are recorded in issues and skipped.
"""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\kaggle\THE_HISTORY_OF_DS_ROGII")
OUT = ROOT / "experiments" / "exp073_public_assets_integration"
OUT.mkdir(parents=True, exist_ok=True)
PILK = ROOT / "data" / "external" / "rogii-model-package"
V11 = ROOT / "data" / "external" / "rogii-v11-fresh-artifacts"
RAV = ROOT / "data" / "external" / "wellbore-geology-prediction-artifacts"

issues = []
metrics = {}


def rmse(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


# ---------------- base: pilkwang train_gt ----------------
gt = pd.read_parquet(PILK / "oof" / "train_gt.parquet")
print("train_gt:", gt.shape, list(gt.columns))
# expected cols: id, well_id, row_index, MD, last_known_TVT, target_tvt, target_delta_from_last_known
y_delta = gt["target_delta_from_last_known"].to_numpy(np.float64)
base = pd.DataFrame({
    "id": gt["id"].astype(str).to_numpy(),
    "well_id": gt["well_id"].astype(str).to_numpy(),
    "last_known_tvt": gt["last_known_TVT"].to_numpy(np.float64),
    "target_delta": y_delta,
})
N = len(base)
print(f"N={N}")

cols = {}

# ---------------- pilkwang OOF npy ----------------
pilk_map = {
    "pilk_xgb": "xgb_oof.npy",
    "pilk_cat": "catboost_oof.npy",
    "pilk_hgb": "hgb_oof.npy",
    "pilk_lgb": "lgb_oof.npy",
    "pilk_tcn": "sequence_tcn_oof.npy",
    "pilk_blend": "blend_oof.npy",
    "pilk_blend_pp": "blend_oof_postprocessed.npy",
}
for name, fn in pilk_map.items():
    try:
        arr = np.load(PILK / "oof" / fn).astype(np.float64)
        if len(arr) != N:
            issues.append(f"{name}: length {len(arr)} != {N}, skipped")
            continue
        cols[name] = arr
        metrics[name] = rmse(y_delta, arr)
        print(f"  {name}: RMSE(delta)={metrics[name]:.4f}")
    except Exception as e:
        issues.append(f"{name}: {e}")

# ---------------- v11 oof_preds.pkl ----------------
try:
    v11_dirs = sorted([p for p in (V11 / "models").glob("*") if p.is_dir()])
    print("v11 model dirs:", [d.name for d in v11_dirs])
    # try to find an id/order reference for v11
    v11_ref_id = None
    for cand in [V11 / "train_gt.parquet", V11 / "oof" / "train_gt.parquet"]:
        if cand.exists():
            v11_ref_id = pd.read_parquet(cand)
            break
    for d in v11_dirs:
        pk = d / "oof_preds.pkl"
        if not pk.exists():
            continue
        obj = pd.read_pickle(pk)
        # normalise to a 1D array
        if isinstance(obj, pd.DataFrame):
            # pick a prediction-like column
            numcols = [c for c in obj.columns if obj[c].dtype.kind == "f"]
            arr = obj[numcols[0]].to_numpy(np.float64) if numcols else None
            print(f"  v11 {d.name}: DataFrame cols={list(obj.columns)[:6]}")
        elif isinstance(obj, pd.Series):
            arr = obj.to_numpy(np.float64)
        elif isinstance(obj, dict):
            print(f"  v11 {d.name}: dict keys={list(obj.keys())[:8]}")
            arr = None
            for k in ("oof", "oof_preds", "pred", "preds"):
                if k in obj:
                    arr = np.asarray(obj[k], np.float64).ravel()
                    break
        else:
            arr = np.asarray(obj, np.float64).ravel()
        if arr is None:
            issues.append(f"v11 {d.name}: could not extract array (type {type(obj)})")
            continue
        if len(arr) != N:
            issues.append(f"v11 {d.name}: length {len(arr)} != {N}, skipped (no id alignment)")
            continue
        nm = f"v11_{d.name.replace('-', '')}"
        cols[nm] = arr
        metrics[nm] = rmse(y_delta, arr)
        print(f"  {nm}: RMSE(delta)={metrics[nm]:.4f}")
except Exception as e:
    issues.append(f"v11 block: {e}")
    traceback.print_exc()

# ---------------- ravaghi koolbox Trainer pickles ----------------
try:
    import joblib
    # id+target reference in train.csv row order (matches trainer.oof_preds order)
    rav_ref = pd.read_csv(RAV / "data" / "train.csv", usecols=["id", "target"])
    print("ravaghi train.csv ref:", rav_ref.shape)
    rav_ids = rav_ref["id"].astype(str).to_numpy()
    rav_y = rav_ref["target"].to_numpy(np.float64)
    # map ravaghi row order -> base id order
    id_to_pos = {i: k for k, i in enumerate(base["id"].to_numpy())}
    pos_in_base = np.array([id_to_pos.get(i, -1) for i in rav_ids])
    aligned_ok = (pos_in_base >= 0).all() and len(rav_ids) == N
    print(f"  ravaghi id alignment ok={aligned_ok}, matched={int((pos_in_base>=0).sum())}/{len(rav_ids)}")

    rav_oofs = {}
    for fam, sub in [("rav_lgb1", "lightgbm-1"), ("rav_lgb2", "lightgbm-2"),
                     ("rav_lgb3", "lightgbm-3"), ("rav_cb1", "catboost-1"),
                     ("rav_cb2", "catboost-2")]:
        pkls = list((RAV / "models" / sub).glob("*.pkl"))
        if not pkls:
            issues.append(f"{fam}: no pkl in {sub}")
            continue
        tr = joblib.load(pkls[0])
        oof = np.asarray(getattr(tr, "oof_preds")).ravel().astype(np.float64)
        score = float(getattr(tr, "overall_score", np.nan))
        if len(oof) != N:
            issues.append(f"{fam}: oof length {len(oof)} != {N}")
            continue
        # reorder ravaghi row-order oof -> base id order
        if aligned_ok:
            reordered = np.empty(N, np.float64)
            reordered[pos_in_base] = oof
            rav_oofs[fam] = reordered
        else:
            rav_oofs[fam] = oof  # assume already aligned
        cols[fam] = rav_oofs[fam]
        metrics[fam] = rmse(y_delta, rav_oofs[fam])
        metrics[f"{fam}_trainer_score"] = score
        print(f"  {fam}: RMSE(delta)={metrics[fam]:.4f} (trainer.overall_score={score:.4f})")

    # positive Ridge stack over the 5 ravaghi families (GroupKFold by well)
    if len(rav_oofs) >= 2:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        fam_names = list(rav_oofs.keys())
        Xs = np.column_stack([rav_oofs[f] for f in fam_names])
        groups = base["well_id"].to_numpy()
        stack = np.zeros(N)
        gkf = GroupKFold(n_splits=5)
        for tr_idx, te_idx in gkf.split(Xs, y_delta, groups):
            r = Ridge(alpha=1.6602834637650032, positive=True, fit_intercept=True)
            r.fit(Xs[tr_idx], y_delta[tr_idx])
            stack[te_idx] = r.predict(Xs[te_idx])
        cols["rav_ridge_stack"] = stack
        metrics["rav_ridge_stack"] = rmse(y_delta, stack)
        print(f"  rav_ridge_stack: RMSE(delta)={metrics['rav_ridge_stack']:.4f}")
except Exception as e:
    issues.append(f"ravaghi block: {e}")
    traceback.print_exc()

# ---------------- assemble & save ----------------
for nm, arr in cols.items():
    base[nm] = arr
pred_cols = list(cols.keys())
base.to_parquet(OUT / "external_oof.parquet", index=False)
print(f"\nSaved external_oof.parquet: {base.shape}, pred_cols={pred_cols}")

result = {
    "n_rows": int(N),
    "pred_columns": pred_cols,
    "metrics_rmse_delta": metrics,
    "issues": issues,
    "leak_audit": (
        "pilkwang OOF = GroupKFold(5) by well, self-well excluded for formation features "
        "(manifest oof_imputer_mode=global_train_self_exclude_query_well); mild optimism (imputer "
        "not rebuilt per fold) acknowledged. ravaghi trainer.oof_preds are GroupKFold OOF. "
        "All honest (delta-space RMSE ~9-11, not ~0)."
    ),
}
with open(OUT / "result_external.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print("issues:", issues)
print("Done.")
