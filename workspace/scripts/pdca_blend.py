"""
Anti-overfit blend automation foundation.

Implements nested-fold blend optimization with per-fold consistency gates,
parsimony penalties, weight stability checks, and CV-LB gap prediction.

Usage:
    python pdca_blend.py \
        --oofs experiments/exp022_particle_filter/oof.csv experiments/exp014_geom_extrap/oof.csv \
        --out experiments/pdca_runs/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import GroupKFold


def load_and_validate_oofs(oof_paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load multiple OOF files and perform inner join on well_id, row_idx.

    Args:
        oof_paths: List of paths to oof.csv files

    Returns:
        Merged DataFrame with all predictions and list of component names

    Raises:
        ValueError: If join results in no rows or mismatched data types
    """
    component_names = []
    dfs = []

    for i, path in enumerate(oof_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"OOF file not found: {path}")

        df = pd.read_csv(path)

        # Validate required columns
        required_cols = {"well_id", "row_idx", "pred_tvt"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"OOF file {path} missing required columns. "
                f"Has: {df.columns.tolist()}, needs: {required_cols}"
            )

        # Ensure consistent types
        df["well_id"] = df["well_id"].astype(str)
        df["row_idx"] = df["row_idx"].astype(int)
        df["pred_tvt"] = df["pred_tvt"].astype(float)

        # Extract component name from path
        comp_name = Path(path).parent.name
        component_names.append(comp_name)

        # Rename prediction column to avoid conflicts
        df = df[["well_id", "row_idx", "pred_tvt"]].copy()
        df.rename(columns={"pred_tvt": f"pred_{comp_name}"}, inplace=True)

        dfs.append(df)

    # Inner join on well_id, row_idx
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=["well_id", "row_idx"], how="inner")

    if len(merged) == 0:
        raise ValueError(
            "Inner join on well_id, row_idx produced 0 rows. "
            "Check data alignment and types."
        )

    return merged, component_names


def filter_target_rows(df: pd.DataFrame, original_oof_path: str) -> pd.DataFrame:
    """
    Filter to target rows (TVT not null) using the first OOF file's TVT column.

    Args:
        df: Merged DataFrame
        original_oof_path: Path to original OOF for TVT reference

    Returns:
        Filtered DataFrame with only target rows
    """
    ref_df = pd.read_csv(original_oof_path)[["well_id", "row_idx", "TVT"]]
    ref_df["well_id"] = ref_df["well_id"].astype(str)
    ref_df["row_idx"] = ref_df["row_idx"].astype(int)

    df = df.merge(ref_df, on=["well_id", "row_idx"], how="inner")
    df = df[df["TVT"].notna()].copy()

    if len(df) == 0:
        raise ValueError("No target rows (TVT not null) after filtering.")

    return df


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nested_fold_blend(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    n_splits: int = 5,
) -> Tuple[np.ndarray, List[float], Dict]:
    """
    Nested GroupKFold blend with per-fold consistency gate.

    Outer loop: for each fold, use other folds to learn weights, evaluate on this fold.
    Inner loop (implicit): weights are learned by NNLS minimization.

    Args:
        X: Component predictions (n_samples, n_components)
        y: Target (n_samples,)
        group_ids: Group IDs for GroupKFold (n_samples,)
        n_splits: Number of folds

    Returns:
        Tuple of:
        - blended predictions (n_samples,)
        - list of fold CVs
        - dict with per-fold details (weights, consistency info)
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_cvs = []
    fold_weights_list = []
    per_fold_detail = {}

    blended_pred = np.zeros_like(y)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Learn weights on train fold using NNLS
        # Solve: min ||y_train - X_train @ w||^2 with w >= 0
        weights, _ = nnls(X_train, y_train)

        # Normalize weights to sum to 1
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback to equal weights if all zeros
            weights = np.ones(X.shape[1]) / X.shape[1]

        # Evaluate on test fold
        pred_test = X_test @ weights
        rmse_test = compute_rmse(y_test, pred_test)
        fold_cvs.append(rmse_test)
        fold_weights_list.append(weights)
        blended_pred[test_idx] = pred_test

        per_fold_detail[f"fold_{fold_idx}"] = {
            "cv": rmse_test,
            "weights": weights.tolist(),
        }

    return blended_pred, fold_cvs, per_fold_detail


def check_per_fold_consistency(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    comp_names: List[str],
    n_splits: int = 5,
) -> Dict:
    """
    For each component, check if it improves CV in at least 4/5 folds.

    Args:
        X: Component predictions (n_samples, n_components)
        y: Target (n_samples,)
        group_ids: Group IDs
        comp_names: Component names
        n_splits: Number of folds

    Returns:
        Dict with consistency gate results
    """
    gkf = GroupKFold(n_splits=n_splits)

    # Baseline: use first component as reference
    baseline_folds = []
    component_improves = {name: 0 for name in comp_names}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Baseline CV (first component only)
        baseline_weights = np.array([1.0] + [0.0] * (X.shape[1] - 1))
        baseline_pred = X_test @ baseline_weights
        baseline_rmse = compute_rmse(y_test, baseline_pred)
        baseline_folds.append(baseline_rmse)

        # Check each component
        for comp_idx, comp_name in enumerate(comp_names):
            if comp_idx == 0:
                continue  # Skip baseline

            # Use all components up to this one
            X_train_subset = X_train[:, : comp_idx + 1]
            X_test_subset = X_test[:, : comp_idx + 1]

            weights, _ = nnls(X_train_subset, y_train)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(X_train_subset.shape[1]) / X_train_subset.shape[1]

            pred = X_test_subset @ weights
            rmse = compute_rmse(y_test, pred)

            if rmse < baseline_rmse:
                component_improves[comp_name] += 1

    # Gate: component passes if improves in >= 4/5 folds
    passed_components = {
        name: count >= 4 for name, count in component_improves.items()
    }

    return {
        "baseline_cvs": baseline_folds,
        "component_fold_improvements": component_improves,
        "passed_components": passed_components,
    }


def compute_weight_stability(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    comp_names: List[str],
    n_splits: int = 5,
) -> Dict:
    """
    Compute weight stability (CV of weights across folds).

    Args:
        X: Component predictions
        y: Target
        group_ids: Group IDs
        comp_names: Component names
        n_splits: Number of folds

    Returns:
        Dict with per-component weight stability metrics
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_weights_list = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, group_ids)):
        X_train, y_train = X[train_idx], y[train_idx]

        weights, _ = nnls(X_train, y_train)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(X.shape[1]) / X.shape[1]

        fold_weights_list.append(weights)

    fold_weights_array = np.array(fold_weights_list)  # (n_splits, n_components)

    stability_info = {}
    for comp_idx, comp_name in enumerate(comp_names):
        weights_across_folds = fold_weights_array[:, comp_idx]
        mean_weight = weights_across_folds.mean()
        std_weight = weights_across_folds.std()

        # Coefficient of variation
        if mean_weight > 1e-6:
            cv_weight = std_weight / mean_weight
        else:
            cv_weight = 0.0

        stability_info[comp_name] = {
            "mean_weight": float(mean_weight),
            "std_weight": float(std_weight),
            "cv_weight": float(cv_weight),
            "is_stable": cv_weight < 0.5,  # Gate: cv < 0.5
        }

    return stability_info


def apply_parsimony_penalty(
    weights: np.ndarray, comp_names: List[str], threshold: float = 0.05
) -> Tuple[np.ndarray, List[str]]:
    """
    Remove components with weight < threshold.

    Args:
        weights: Component weights
        comp_names: Component names
        threshold: Minimum weight to keep

    Returns:
        Tuple of filtered weights (renormalized) and filtered comp names
    """
    mask = weights >= threshold
    filtered_weights = weights[mask]
    filtered_names = [name for name, keep in zip(comp_names, mask) if keep]

    # Renormalize
    if filtered_weights.sum() > 0:
        filtered_weights = filtered_weights / filtered_weights.sum()

    return filtered_weights, filtered_names


def compute_confidence_level(
    fold_cvs: List[float],
    stability_info: Dict,
    comp_names: List[str],
    n_components: int,
) -> str:
    """
    Assign confidence level based on fold consistency and weight stability.

    Args:
        fold_cvs: CV scores for each fold
        stability_info: Per-component stability metrics
        comp_names: Component names
        n_components: Total number of components

    Returns:
        Confidence level: "high", "medium", or "low"
    """
    # Check fold consistency: all folds improve?
    fold_std = np.std(fold_cvs)
    fold_cv_coeff = fold_std / np.mean(fold_cvs) if np.mean(fold_cvs) > 0 else 0
    fold_consistent = fold_cv_coeff < 0.15  # Low CV coefficient means stable

    # Check weight stability: all non-zero weights stable?
    all_weights_stable = all(
        info["is_stable"]
        for name, info in stability_info.items()
        if info["mean_weight"] > 0.05
    )

    # Check component count: prefer <= 3
    component_count_ok = n_components <= 3

    if fold_consistent and all_weights_stable and component_count_ok:
        return "high"
    elif fold_consistent and all_weights_stable:
        return "medium"
    else:
        return "low"


def predict_lb(cv: float, confidence: str) -> Tuple[float, str]:
    """
    Predict Kaggle LB from local CV using historical CV-LB gap.

    Historical data from MEMORY:
    - exp026: CV 10.062, LB 8.672, gap +1.390
    - exp008: CV 13.808, LB 12.339, gap +1.469
    - exp033: CV 9.977, LB (unknown, ~8.5 estimated)

    Average gap: ~1.40 (consistent for clean blends)

    Args:
        cv: Local CV score
        confidence: Confidence level ("high", "medium", "low")

    Returns:
        Tuple of predicted LB and uncertainty range
    """
    gap = 1.40
    predicted_lb = cv - gap

    if confidence == "high":
        uncertainty = 0.2
        range_str = f"{predicted_lb - uncertainty:.3f} ± {uncertainty:.3f}"
    elif confidence == "medium":
        uncertainty = 0.35
        range_str = f"{predicted_lb - uncertainty:.3f} ± {uncertainty:.3f}"
    else:
        uncertainty = 0.5
        range_str = f"{predicted_lb - uncertainty:.3f} ± {uncertainty:.3f}"

    return predicted_lb, range_str


def main():
    parser = argparse.ArgumentParser(
        description="Anti-overfit blend automation with nested-fold CV"
    )
    parser.add_argument(
        "--oofs",
        nargs="+",
        required=True,
        help="Paths to OOF CSV files to blend",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/pdca_runs/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp for run (default: current time)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Load OOFs
    print(f"Loading {len(args.oofs)} OOF files...")
    try:
        merged_df, comp_names = load_and_validate_oofs(args.oofs)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Filter to target rows
    print(f"Filtering to target rows (TVT not null)...")
    try:
        merged_df = filter_target_rows(merged_df, args.oofs[0])
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Final dataset: {len(merged_df)} rows, {len(comp_names)} components")

    # Extract data for blending
    X = np.column_stack(
        [merged_df[f"pred_{name}"].values for name in comp_names]
    )
    y = merged_df["TVT"].values
    group_ids = merged_df["well_id"].values

    print(f"\nRunning nested-fold blend (n_splits={args.n_splits})...")
    blended_pred, fold_cvs, per_fold_detail = nested_fold_blend(
        X, y, group_ids, n_splits=args.n_splits
    )

    nested_cv = compute_rmse(y, blended_pred)
    print(f"  Nested CV: {nested_cv:.6f}")
    print(f"  Per-fold CVs: {[f'{cv:.6f}' for cv in fold_cvs]}")

    # Per-fold consistency gate
    print(f"\nChecking per-fold consistency gate...")
    consistency_result = check_per_fold_consistency(
        X, y, group_ids, comp_names, n_splits=args.n_splits
    )

    passed_comps = consistency_result["passed_components"]
    print(f"  Components passing consistency gate (>=4/5 folds):")
    for name, passed in passed_comps.items():
        count = consistency_result["component_fold_improvements"][name]
        status = "PASS" if passed else "FAIL"
        print(f"    {name}: {count}/5 folds ({status})")

    # Weight stability
    print(f"\nAnalyzing weight stability...")
    stability_info = compute_weight_stability(
        X, y, group_ids, comp_names, n_splits=args.n_splits
    )

    for name, info in stability_info.items():
        status = "STABLE" if info["is_stable"] else "UNSTABLE"
        print(
            f"  {name}: weight={info['mean_weight']:.4f}, "
            f"CV={info['cv_weight']:.3f} ({status})"
        )

    # Final blend weights (on full data)
    print(f"\nLearning final blend weights on full data...")
    final_weights, _ = nnls(X, y)
    if final_weights.sum() > 0:
        final_weights = final_weights / final_weights.sum()
    else:
        final_weights = np.ones(len(comp_names)) / len(comp_names)

    # Apply parsimony penalty
    filtered_weights, filtered_names = apply_parsimony_penalty(
        final_weights, comp_names, threshold=0.05
    )

    print(f"  Final weights (after parsimony threshold 0.05):")
    for name, weight in zip(filtered_names, filtered_weights):
        print(f"    {name}: {weight:.4f}")

    # Compute confidence level
    confidence = compute_confidence_level(
        fold_cvs, stability_info, comp_names, len(filtered_names)
    )
    print(f"\nConfidence level: {confidence.upper()}")

    # Predict LB
    predicted_lb, lb_range = predict_lb(nested_cv, confidence)
    print(f"Predicted Kaggle LB: {predicted_lb:.3f} (range: {lb_range})")

    # Compile results
    result = {
        "timestamp": args.timestamp or datetime.now().isoformat(),
        "oof_paths": args.oofs,
        "component_names": comp_names,
        "n_rows": len(merged_df),
        "n_components": len(comp_names),
        "nested_cv": float(nested_cv),
        "fold_cvs": [float(cv) for cv in fold_cvs],
        "final_weights": {name: float(w) for name, w in zip(comp_names, final_weights)},
        "filtered_weights": {name: float(w) for name, w in zip(filtered_names, filtered_weights)},
        "weight_stability": {
            name: {
                "mean_weight": float(info["mean_weight"]),
                "std_weight": float(info["std_weight"]),
                "cv_weight": float(info["cv_weight"]),
                "is_stable": bool(info["is_stable"]),
            }
            for name, info in stability_info.items()
        },
        "consistency_gate": {
            "component_fold_improvements": consistency_result["component_fold_improvements"],
            "passed_components": passed_comps,
        },
        "per_fold_detail": per_fold_detail,
        "confidence": confidence,
        "predicted_lb": float(predicted_lb),
        "predicted_lb_range": lb_range,
        "gap_assumption": 1.40,
    }

    # Save results
    if args.timestamp:
        run_name = f"run_{args.timestamp}"
    else:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    result_path = os.path.join(args.out, f"{run_name}.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    # Summary
    print("\n" + "=" * 60)
    print("BLEND SUMMARY")
    print("=" * 60)
    print(f"Components: {len(comp_names)} ({', '.join(comp_names)})")
    print(f"Target rows: {len(merged_df)}")
    print(f"Nested CV: {nested_cv:.6f}")
    print(f"Fold CVs: min={min(fold_cvs):.6f}, max={max(fold_cvs):.6f}")
    print(f"Fold stability (CV coeff): {np.std(fold_cvs)/np.mean(fold_cvs):.3f}")
    print(f"Final weights (top 3): {sorted(zip(comp_names, final_weights), key=lambda x: x[1], reverse=True)[:3]}")
    print(f"Confidence: {confidence.upper()}")
    print(f"Predicted LB: {predicted_lb:.3f} ({lb_range})")
    print("=" * 60)


if __name__ == "__main__":
    main()
