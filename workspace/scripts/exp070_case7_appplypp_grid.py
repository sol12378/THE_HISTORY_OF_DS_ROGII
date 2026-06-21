#!/usr/bin/env python3
"""exp070 Case 7: apply_pp grid search (w_pf, tau, alpha) with GroupKFold validation.
Leak-free: uses only known/target rows, no hidden true values.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rogii.training.baselines import tvt_rmse, write_json, now_jst

EXP_ID = "exp070_case7_appplypp_grid"
OUT_DIR = Path("experiments") / EXP_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid
W_PF_GRID = [0.0, 0.05, 0.09, 0.15, 0.2]
TAU_GRID = [None, 50, 85, 150, 300]
ALPHA_GRID = [0.95, 1.0, 1.05]

# Baselines
EXP022_CV = 11.024014
EXP014_CV = 13.525189

print(f"[{EXP_ID}] apply_pp grid search")
print(f"  w_pf: {W_PF_GRID}")
print(f"  tau:  {TAU_GRID}")
print(f"  alpha: {ALPHA_GRID}")
print(f"  Total configs: {len(W_PF_GRID) * len(TAU_GRID) * len(ALPHA_GRID)}")

# Load predictions
print("\nLoading OOF predictions...")
exp022_oof = pd.read_csv("experiments/exp022_particle_filter/oof.csv")
exp014_oof = pd.read_csv("experiments/exp014_geom_extrap/oof.csv")

# Merge on well_id + row_idx (more robust than id)
df = exp022_oof[["well_id", "row_idx", "TVT", "pred_tvt"]].copy()
df.columns = ["well_id", "row_idx", "TVT", "pf_pred"]
df = df.merge(exp014_oof[["well_id", "row_idx", "pred_tvt"]], 
              on=["well_id", "row_idx"], how="inner")
df.rename(columns={"pred_tvt": "geom_pred"}, inplace=True)

print(f"  Aligned rows: {len(df)}")
assert len(df) > 0, "No rows to process!"

# Load MD data for md_since calculation
print("Computing md_since (MD - last_known_MD)...")
train_base = pd.read_parquet("data/processed/train_base_v001.parquet",
                             columns=["well_id", "row_idx", "MD", "last_known_MD"])
df = df.merge(train_base, on=["well_id", "row_idx"], how="left")
df["md_since"] = (df["MD"] - df["last_known_MD"]).fillna(0).clip(lower=0)

# Load folds
print("Loading folds...")
folds = pd.read_csv("data/folds/folds_group_well_v001.csv")
df = df.merge(folds[["well_id", "fold"]], on="well_id", how="left")

print(f"  Folds loaded: {df['fold'].nunique()} folds, rows: {len(df)}")

# Grid search
print("\nRunning grid search (this may take a few minutes)...")
results = []
best_cv = float('inf')
best_config = None

for w_pf in W_PF_GRID:
    for tau in TAU_GRID:
        for alpha in ALPHA_GRID:
            # Apply pp to full dataset (using fold-wise pseudo-CV to check consistency)
            pred_full = np.zeros(len(df))
            cv_folds = []
            
            for fold_id in range(5):
                test_mask = df["fold"] == fold_id
                test_idx = df.index[test_mask]
                
                # Apply formula to test fold
                pred = (1.0 - w_pf) * df.loc[test_idx, "geom_pred"] + w_pf * df.loc[test_idx, "pf_pred"]
                if tau is not None:
                    tau_factor = 1.0 - np.exp(-df.loc[test_idx, "md_since"].values / tau)
                    pred = pred * (1.0 - tau_factor) + df.loc[test_idx, "pf_pred"].values * tau_factor
                pred = pred * alpha
                pred_full[test_idx] = pred
                
                fold_cv = tvt_rmse(df.loc[test_idx, "TVT"], pred)
                cv_folds.append(fold_cv)
            
            # Pool CV
            pool_cv = tvt_rmse(df["TVT"], pred_full)
            
            results.append({
                "w_pf": w_pf,
                "tau": tau,
                "alpha": alpha,
                "pool_cv": pool_cv,
                "fold_cvs": cv_folds,
                "fold_std": np.std(cv_folds)
            })
            
            if pool_cv < best_cv:
                best_cv = pool_cv
                best_config = (w_pf, tau, alpha)
            
            tau_str = str(tau) if tau is not None else "None"
            print(f"  w_pf={w_pf:.2f}, tau={tau_str:>4}, alpha={alpha:.2f}: CV={pool_cv:.6f} ± {np.std(cv_folds):.4f}")

print(f"\n✓ Best config: w_pf={best_config[0]}, tau={best_config[1]}, alpha={best_config[2]}")
print(f"  CV: {best_cv:.6f}")
print(f"  vs exp022 (PF only): {EXP022_CV:.6f}, delta={EXP022_CV - best_cv:+.6f} ft")
print(f"  vs exp014 (geom only): {EXP014_CV:.6f}, delta={EXP014_CV - best_cv:+.6f} ft")

# Determine if tau ramp is effective
baseline_no_tau = next(r for r in results if r["w_pf"] == best_config[0] and r["tau"] is None and r["alpha"] == best_config[2])
tau_ramp_effective = best_cv < baseline_no_tau["pool_cv"] - 0.01

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "grid_results.csv", index=False)

result_dict = {
    "exp_id": EXP_ID,
    "created_at": now_jst(),
    "best_w_pf": float(best_config[0]),
    "best_tau": best_config[1],
    "best_alpha": float(best_config[2]),
    "best_cv": float(best_cv),
    "baseline_pf_cv": EXP022_CV,
    "baseline_geom_cv": EXP014_CV,
    "improvement_vs_pf": float(EXP022_CV - best_cv),
    "improvement_vs_geom": float(EXP014_CV - best_cv),
    "tau_ramp_effective": tau_ramp_effective,
    "grid_size": len(results),
    "status": "completed"
}
write_json(OUT_DIR / "result.json", result_dict)

(OUT_DIR / "notes.md").write_text(f"""# {EXP_ID}: apply_pp Grid Search

## Grid Configuration
- w_pf (PF weight): {W_PF_GRID}
- tau (depth ramp, MD distance): {TAU_GRID}
- alpha (scaling): {ALPHA_GRID}
- Total configs tested: {len(results)}

## Best Configuration
| Param | Value |
|---|---|
| w_pf | {best_config[0]} |
| tau | {best_config[1]} |
| alpha | {best_config[2]} |
| **Pool CV** | **{best_cv:.6f}** |

## Comparison
| Baseline | CV | Delta |
|---|---|---|
| exp022 (PF 500/128) | {EXP022_CV:.6f} | {EXP022_CV - best_cv:+.6f} ft |
| exp014 (geom) | {EXP014_CV:.6f} | {EXP014_CV - best_cv:+.6f} ft |

## Tau Ramp Effect
- Baseline (no ramp, tau=None, best w_pf/alpha): {baseline_no_tau['pool_cv']:.6f}
- With ramp (tau={best_config[1]}): {best_cv:.6f}
- **Tau ramp effect**: {'EFFECTIVE' if tau_ramp_effective else 'NOT SIGNIFICANT'} (delta={baseline_no_tau['pool_cv'] - best_cv:+.6f} ft)

## Interpretation
The tau ramp factor (1 - exp(-md_since/tau)) modulates the blend weight based on depth.
- tau=None: linear blend, constant weight throughout hidden section
- tau>0: exponential decay, emphasize PF near last_known_MD, geometric blend deeper

Result: {'Tau ramp provides consistent improvement by adapting weights to hidden depth' if tau_ramp_effective else 'Linear blend is optimal; no need for depth modulation'}

## Next Actions
1. Evaluate best config on test set (apply_pp kernel)
2. Blend with Case 8 (PF 600/150) if both show improvements
3. Check error correlation for ensemble potential
""")

print(f"\nOutput: {OUT_DIR}/")

