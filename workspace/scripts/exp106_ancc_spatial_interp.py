#!/usr/bin/env python
"""
exp106: ANCC spatial interpolation (LOWO) -> closed-form TVT.

Background (decisive):
  Closed form TVT = ANCC - Z + b_well (Pearson -1.0000, residual ~0.007ft).
  The dominant lever is honest spatial interpolation of the ANCC formation-top
  surface at the evaluation well's (X,Y), with strict Leave-One-Well-Out (LOWO):
  the eval well's OWN formation points must NEVER enter the interpolation source.
  exp056 failed exactly here (self-leak: used all points incl. the eval well).

Data reality on this machine (verified):
  - Raw per-well formation CSVs (data/raw/train/*__horizontal_well.csv) are GONE.
  - The only ANCC surface signal available is `pf_ancc` from the external table
    data/external/wellbore-geology-prediction-artifacts/data/train.csv, which is
    ITSELF an existing TVT-space plane-fit interpolation (corr 0.9997 vs TVT,
    std ~14ft). It exists only for target (hidden tail) rows.
  - `bw_ANCC` is constant per well = the well intercept b_well (already fit).
  - train_base target id = well_id + "_" + row_idx (matches external `id`).

So this experiment does what is honestly possible: re-interpolate the available
ANCC surface (pf_ancc) with STRICT LOWO (eval well excluded from source), then
apply the closed form / per-well bias calibration from TVT_input only, and report
honest well-fold and typewell-fold CV. This both (a) fixes exp056's self-leak and
(b) bounds how much pure spatial re-interpolation can recover given the surface
signal we actually have.

Methods compared:
  knn   : plane-fit / IDW KNN interpolation of pf_ancc surface, LOWO. Predict TVT
          = interp_surface (pf_ancc is already TVT-space) + per-well bias from known.
  plane : local plane fit (X,Y -> surface) over K LOWO neighbors (konbu17 style).
  consensus: median over multiple K (robustness).

CV: GroupKFold by well (folds_group_well_v001) and by typewell
    (folds_group_typewell_v001). Evaluate on hidden tail (target) rows only.

LOWO guarantee: the KDTree source is built ONLY from rows whose well_id != eval
well. Verified per fold. Calibration b_well uses TVT_input known rows only.
"""
import argparse, json, time, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'experiments' / '_ancc_work.parquet'          # target rows + pf_ancc/bw_ANCC
TRAIN_BASE = ROOT / 'data/processed/train_base_v001.parquet'
EXT_SLIM = ROOT / 'experiments/_ancc_ext_slim.parquet'
FOLD_WELL = ROOT / 'data/folds/folds_group_well_v001.csv'
FOLD_TW = ROOT / 'data/folds/folds_group_typewell_v001.csv'
OUT = ROOT / 'experiments/exp106_ancc_spatial_interp'
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m):
    print(m, flush=True)
    LOG.append(str(m))

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def build_work():
    """Build/cache the working table: target (hidden) rows with real X,Y,Z,TVT and
    pf_ancc surface; plus known (TVT_input) rows per well for calibration."""
    if WORK.exists() and EXT_SLIM.exists():
        m = pd.read_parquet(WORK)
    else:
        tb = pd.read_parquet(TRAIN_BASE,
            columns=['well_id','row_idx','X','Y','Z','TVT','TVT_input','is_known_tvt','is_target'])
        tgt = tb[tb['is_target']].copy()
        tgt['id'] = tgt['well_id'].astype(str) + '_' + tgt['row_idx'].astype(str)
        if EXT_SLIM.exists():
            ext = pd.read_parquet(EXT_SLIM)[['id','pf_ancc','bw_ANCC']]
        else:
            cols = ['well','id','pf_ancc','bw_ANCC','tvtF_ANCC','last_known_tvt']
            chs = []
            for ch in pd.read_csv(ROOT/'data/external/wellbore-geology-prediction-artifacts/data/train.csv',
                    usecols=cols, dtype={'well':str,'id':str}, chunksize=500000, low_memory=False):
                chs.append(ch)
            ext = pd.concat(chs, ignore_index=True)
            ext.to_parquet(EXT_SLIM)
            ext = ext[['id','pf_ancc','bw_ANCC']]
        m = tgt.merge(ext, on='id', how='inner')
        m = m[['well_id','X','Y','Z','TVT','TVT_input','is_known_tvt','pf_ancc','bw_ANCC']]
        m.to_parquet(WORK)
    # known calibration rows
    tb = pd.read_parquet(TRAIN_BASE,
        columns=['well_id','X','Y','Z','TVT_input','is_known_tvt','is_target'])
    known = tb[tb['is_known_tvt']].copy()
    known = known[['well_id','X','Y','Z','TVT_input']]
    return m, known


def interp_surface_lowo(tree, src_xy, src_val, qxy, k, method):
    """Interpolate surface at qxy from prebuilt `tree` over (src_xy, src_val).
    src already excludes eval well. method: 'idw' KNN, or 'plane' local plane fit."""
    if tree is None or len(src_xy) < max(k, 4):
        return np.full(len(qxy), np.nan)
    kk = min(k, len(src_xy))
    dist, idx = tree.query(qxy, k=kk, workers=-1)
    if dist.ndim == 1:
        dist = dist[:, None]; idx = idx[:, None]
    if method == 'idw':
        w = 1.0 / (dist + 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        return (w * src_val[idx]).sum(axis=1)
    # local plane fit (konbu17 style): weighted LS val ~ a + b*(x-qx) + c*(y-qy) over
    # the k neighbors per query; value at query == intercept a. Vectorized via batched
    # 3x3 normal equations. n=#queries, k=neighbors.
    nx = src_xy[:, 0][idx]; ny = src_xy[:, 1][idx]; nv = src_val[idx]   # (n,k)
    dx = nx - qxy[:, 0:1]; dy = ny - qxy[:, 1:2]                         # centered
    w = 1.0 / (dist + 1e-6)                                             # (n,k) weights
    # design basis B = [1, dx, dy] -> (n,k,3)
    B = np.stack([np.ones_like(dx), dx, dy], axis=2)
    Wd = w[:, :, None]                                                  # (n,k,1)
    # normal eqs: (B^T W B) a = B^T W v
    ATA = np.einsum('nki,nkj->nij', B * Wd, B)                          # (n,3,3)
    ATb = np.einsum('nki,nk->ni', B * Wd, nv)                           # (n,3)
    ATA += np.eye(3)[None] * 1e-3                                       # ridge for stability
    try:
        coef = np.linalg.solve(ATA, ATb[:, :, None])[:, :, 0]          # (n,3)
        val = coef[:, 0]                                               # intercept == value at query
    except np.linalg.LinAlgError:
        val = np.full(len(qxy), np.nan)
    # IDW fallback / clamp against unstable extrapolation
    wn = w / w.sum(axis=1, keepdims=True)
    idw_val = (wn * nv).sum(axis=1)
    lo = nv.min(axis=1); hi = nv.max(axis=1); span = hi - lo + 1e-9
    bad = ~np.isfinite(val) | (val < lo - span) | (val > hi + span)
    val = np.where(bad, idw_val, val)
    return val


def run_method(work, known, fold_map, method, k, calibrate, eval_wells_only=None, src_pool=None):
    """One CV pass. Returns dict with overall + per-fold RMSE and oof frame.

    src_pool: full work table to interpolate FROM (defaults to `work`). In smoke we
    evaluate a few wells (`eval_wells_only`) but keep the FULL source pool so neighbor
    geometry is realistic. LOWO is still enforced (eval well removed from source)."""
    if src_pool is None:
        src_pool = work
    wells = work['well_id'].unique()
    if eval_wells_only is not None:
        wells = [w for w in wells if w in set(eval_wells_only)]
    # source pools keyed by well for LOWO removal: stack target-row (X,Y,pf_ancc)
    src_well = src_pool['well_id'].to_numpy()
    src_xy_all = src_pool[['X','Y']].to_numpy()
    src_val_all = src_pool['pf_ancc'].to_numpy()

    folds = sorted(set(fold_map.get(w) for w in wells))
    work_by_well = {w: g for w, g in work.groupby('well_id')}
    known_by_well = {w: g for w, g in known.groupby('well_id')}
    oof_parts = []
    fold_rmse = {}
    leak_checks = []
    for f in folds:
        eval_wells = [w for w in wells if fold_map.get(w) == f]
        if not eval_wells:
            continue
        eval_set = set(eval_wells)
        # LOWO source = source rows whose well is NOT an eval well in this fold.
        keep = ~np.isin(src_well, list(eval_set))
        s_xy = src_xy_all[keep]; s_val = src_val_all[keep]; s_w = src_well[keep]
        leak = len(set(s_w.tolist()) & eval_set)
        leak_checks.append(leak)
        tree = cKDTree(s_xy) if len(s_xy) >= max(k, 4) else None
        fparts = []
        for w in eval_wells:
            wm = work_by_well[w]
            qxy = wm[['X','Y']].to_numpy()
            interp = interp_surface_lowo(tree, s_xy, s_val, qxy, k, method)
            pred = interp.copy()
            bias = 0.0
            if calibrate:
                kk = known_by_well.get(w)
                if kk is not None and len(kk):
                    kqxy = kk[['X','Y']].to_numpy()
                    kinterp = interp_surface_lowo(tree, s_xy, s_val, kqxy, k, method)
                    d = kk['TVT_input'].to_numpy() - kinterp
                    d = d[~np.isnan(d)]
                    if len(d):
                        bias = float(np.median(d))
                pred = interp + bias
            fparts.append(pd.DataFrame({
                'well_id': w, 'fold': f, 'TVT': wm['TVT'].to_numpy(),
                'pred': pred, 'interp': interp, 'bias': bias}))
        oof_parts.extend(fparts)
        ow = pd.concat(fparts) if fparts else None
        fold_rmse[f] = rmse(ow['TVT'], ow['pred']) if ow is not None else float('nan')
        log(f"    fold {f}: {len(eval_wells)} wells, RMSE={fold_rmse[f]:.3f}, leak_wells_in_src={leak}")
    oof = pd.concat(oof_parts, ignore_index=True)
    overall = rmse(oof['TVT'], oof['pred'])
    return {'overall': overall, 'fold_rmse': fold_rmse,
            'leak_ok': all(x == 0 for x in leak_checks)}, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--k', type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    log(f"[1] Building work table (smoke={args.smoke})...")
    work, known = build_work()
    log(f"    work rows={len(work):,} wells={work['well_id'].nunique()} known rows={len(known):,}")

    fw = pd.read_csv(FOLD_WELL)
    ftw = pd.read_csv(FOLD_TW)
    wellfold = dict(zip(fw['well_id'].astype(str), fw['fold']))
    twfold = dict(zip(ftw['well_id'].astype(str), ftw['fold']))

    # Source pool: subsample pf_ancc surface to ~every 10th row per well to keep the
    # per-fold KDTree memory/time bounded (the surface is smooth along the trajectory,
    # so this preserves geometry). LOWO removal happens per fold.
    src_pool = work.groupby('well_id', group_keys=False).apply(
        lambda g: g.iloc[::10]).reset_index(drop=True)
    log(f"    source pool subsampled to {len(src_pool):,} points "
        f"({src_pool['well_id'].nunique()} wells)")
    eval_only = None
    if args.smoke:
        # Evaluate ~50 wells but keep the FULL source pool so neighbor geometry is realistic.
        eval_only = sorted(work['well_id'].unique())[:50]
        log(f"    SMOKE: evaluating {len(eval_only)} wells, full {src_pool['well_id'].nunique()}-well source")

    results = {}
    oofs = {}
    # Primary = idw_raw: LOWO-interpolated pf_ancc surface used DIRECTLY as TVT.
    # (pf_ancc is already TVT-space; the closed form ANCC-Z+b_well is baked in, so an
    #  extra per-well heel bias double-counts and transfers the heel offset to the toe
    #  of the lateral -> we verified that hurts. Surface-direct is the honest closed form.)
    # We also run a consensus (median over K in {6,8,12,20}) and a plane fit, plus the
    # heel-calibrated variant for the record (shown to be worse).
    method_specs = [('idw_raw','idw',False), ('plane_raw','plane',False),
                    ('idw_calib','idw',True)]
    for tag, method, calib in method_specs:
        log(f"[2] method={tag} (well-fold)...")
        r, oof = run_method(work, known, wellfold, method, args.k, calib,
                            eval_wells_only=eval_only, src_pool=src_pool)
        log(f"    -> well-fold RMSE={r['overall']:.3f} leak_ok={r['leak_ok']}")
        results[tag+'_wellfold'] = r
        oofs[tag] = oof

    # consensus: median over several K of the raw IDW surface (robustness)
    log("[2c] consensus (median over K in 6,8,12,20, well-fold)...")
    cons_parts = []
    for kk in [6, 8, 12, 20]:
        _, o = run_method(work, known, wellfold, 'idw', kk, False,
                          eval_wells_only=eval_only, src_pool=src_pool)
        cons_parts.append(o[['well_id','fold','TVT','pred']].rename(columns={'pred': f'p{kk}'}))
    cons = cons_parts[0][['well_id','fold','TVT']].copy()
    for c in cons_parts:
        col = [x for x in c.columns if x.startswith('p')][0]
        cons[col] = c[col].to_numpy()
    pcols = [c for c in cons.columns if c.startswith('p')]
    cons['pred'] = cons[pcols].median(axis=1)
    cons_rmse = rmse(cons['TVT'], cons['pred'])
    log(f"    -> consensus well-fold RMSE={cons_rmse:.3f}")
    results['consensus_wellfold'] = {'overall': cons_rmse}

    # typewell-fold for the primary (idw_raw)
    log("[3] idw_raw (typewell-fold)...")
    r_tw, oof_tw = run_method(work, known, twfold, 'idw', args.k, False,
                              eval_wells_only=eval_only, src_pool=src_pool)
    log(f"    -> typewell-fold RMSE={r_tw['overall']:.3f} leak_ok={r_tw['leak_ok']}")
    results['idw_raw_typewellfold'] = r_tw

    best = oofs['idw_raw']
    # ANCC interpolation error = how well the LOWO surface itself matches TVT (==primary here)
    ancc_interp_rmse = rmse(best['TVT'], best['interp'])

    # ---- residual GBM on primary OOF (well-fold), real geometry features ----
    gbm_cv = None
    try:
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold
        # join geometry (Z,X,Y) back by positional order within each well
        geo = work[['well_id','X','Y','Z']].copy()
        b = best.copy()
        b = b.dropna(subset=['pred']).reset_index(drop=True)
        # row-aligned merge: both `best` and `work` iterate wells in groupby order; rebuild
        # a stable key using cumulative count within well
        b['rk'] = b.groupby('well_id').cumcount()
        geo['rk'] = geo.groupby('well_id').cumcount()
        b = b.merge(geo, on=['well_id','rk'], how='left')
        b['resid'] = b['TVT'] - b['pred']
        feat = b[['interp','X','Y','Z']].copy()
        groups = b['well_id']
        gkf = GroupKFold(n_splits=5)
        pred_resid = np.zeros(len(b))
        for tr, va in gkf.split(feat, b['resid'], groups):
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=200, verbose=-1)
            m.fit(feat.iloc[tr], b['resid'].iloc[tr])
            pred_resid[va] = m.predict(feat.iloc[va])
        gbm_pred = b['pred'].to_numpy() + pred_resid
        gbm_cv = rmse(b['TVT'], gbm_pred)
        log(f"[4] residual GBM well-fold RMSE={gbm_cv:.3f} (base {results['idw_raw_wellfold']['overall']:.3f})")
    except Exception as e:
        log(f"[4] residual GBM skipped: {e}")

    runtime = time.time() - t0
    best.to_csv(OUT / 'oof.csv', index=False)

    result = {
        'experiment': 'exp106_ancc_spatial_interp',
        'smoke': args.smoke,
        'k': args.k,
        'primary_method': 'idw_raw (LOWO-interpolated pf_ancc surface used directly as TVT)',
        'method': {
            'idw_raw': {'cv_rmse_wellfold': results['idw_raw_wellfold']['overall'],
                        'fold_rmse': results['idw_raw_wellfold']['fold_rmse']},
            'plane_raw': {'cv_rmse_wellfold': results['plane_raw_wellfold']['overall'],
                          'fold_rmse': results['plane_raw_wellfold']['fold_rmse']},
            'idw_calib_heelbias': {'cv_rmse_wellfold': results['idw_calib_wellfold']['overall'],
                                   'note': 'heel-bias calibration HURTS (transfers heel offset to toe)'},
            'consensus_medianK': {'cv_rmse_wellfold': results['consensus_wellfold']['overall']},
        },
        'cv_rmse_wellfold': results['idw_raw_wellfold']['overall'],
        'cv_rmse_typewellfold': results['idw_raw_typewellfold']['overall'],
        'ancc_interp_rmse_ft': ancc_interp_rmse,
        'with_residual_gbm': {'cv': gbm_cv},
        'fold_rmse': results['idw_raw_wellfold']['fold_rmse'],
        'leak_risk': {
            'LOWO_enforced': True,
            'source_excludes_eval_well': results['idw_raw_wellfold']['leak_ok'] and
                                          results['idw_raw_typewellfold']['leak_ok'],
            'note': 'KDTree source built only from rows whose well_id is NOT an eval '
                    'well in the fold (verified leak_wells_in_src==0 every fold). '
                    'Fixes exp056 self-leak (which deliberately used all points incl. '
                    'the eval well for speed).'},
        'compare': {'exp104': 9.89, 'exp105_tw': 9.53, 'konbu17_lb': 11.9},
        'data_caveat': ('Raw per-well formation CSVs are absent on this machine; the only '
                        'ANCC surface available is pf_ancc from the external table, which is '
                        'itself an existing TVT-space plane-fit interpolation (corr 0.9997 vs '
                        'TVT, ~14ft). This experiment honestly RE-interpolates that surface '
                        'under strict LOWO; it cannot reconstruct raw geological ANCC depth, '
                        'so it bounds rather than improves the spatial lever.'),
        'runtime_sec': runtime,
        'notes_short': 'Strict-LOWO re-interpolation of ANCC(pf_ancc) surface used directly as TVT.',
    }
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    with open(OUT / 'run.log', 'w') as f:
        f.write('\n'.join(LOG))
    log(f"[done] runtime={runtime:.1f}s wellfold={result['cv_rmse_wellfold']:.3f} "
        f"twfold={result['cv_rmse_typewellfold']:.3f} interp_rmse={ancc_interp_rmse:.3f}")
    return result


if __name__ == '__main__':
    main()
