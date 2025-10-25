# core/breath_outliers.py
"""
Outlier detection and calculation failure identification for breath metrics.

This module identifies problematic breath cycles using two approaches:
1. Statistical outlier detection for key metrics (IF, Ti, Te, amplitudes, areas)
2. Calculation failure detection (NaN values in computed metrics)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional


def detect_metric_outliers(metric_array: np.ndarray,
                          onsets: np.ndarray,
                          n_std: float = 3.0,
                          global_mean: Optional[float] = None,
                          global_std: Optional[float] = None) -> np.ndarray:
    """
    Detect outliers in a metric array using statistical threshold (mean ± n_std).

    Args:
        metric_array: Stepwise metric values (same length as signal)
        onsets: Breath onset indices to define cycle boundaries
        n_std: Number of standard deviations for outlier threshold
        global_mean: Pre-computed mean across all sweeps (if None, compute from this sweep only)
        global_std: Pre-computed std across all sweeps (if None, compute from this sweep only)

    Returns:
        Boolean mask (same length as metric_array) marking outlier regions
    """
    N = len(metric_array)
    outlier_mask = np.zeros(N, dtype=bool)

    if onsets is None or len(onsets) < 3:
        return outlier_mask

    # Extract per-cycle values (one value per breath cycle)
    cycle_values = []
    cycle_bounds = []

    for i in range(len(onsets)):
        start = int(onsets[i])
        end = int(onsets[i + 1]) if i + 1 < len(onsets) else N

        # Get metric value for this cycle (should be constant across cycle)
        cycle_segment = metric_array[start:end]
        valid_vals = cycle_segment[~np.isnan(cycle_segment)]

        if len(valid_vals) > 0:
            cycle_value = valid_vals[0]  # Take first valid value (all should be same)
            cycle_values.append(cycle_value)
            cycle_bounds.append((start, end))

    if len(cycle_values) < 3:
        return outlier_mask

    # Use global statistics if provided, otherwise compute from current sweep
    if global_mean is not None and global_std is not None:
        mean_val = global_mean
        std_val = global_std
    else:
        # Compute statistics on cycle values (per-sweep mode)
        cycle_values_arr = np.array(cycle_values)
        mean_val = np.mean(cycle_values_arr)
        std_val = np.std(cycle_values_arr)

    if std_val == 0:
        return outlier_mask

    # Mark cycles that are outliers
    lower_bound = mean_val - n_std * std_val
    upper_bound = mean_val + n_std * std_val

    for i, (start, end) in enumerate(cycle_bounds):
        if i < len(cycle_values):
            val = cycle_values[i]
            if val < lower_bound or val > upper_bound:
                outlier_mask[start:end] = True

    return outlier_mask


def detect_calculation_failures(metric_array: np.ndarray,
                                onsets: np.ndarray) -> np.ndarray:
    """
    Detect breath cycles where metric calculation failed (NaN values).

    Args:
        metric_array: Stepwise metric values (same length as signal)
        onsets: Breath onset indices to define cycle boundaries

    Returns:
        Boolean mask marking cycles with calculation failures
    """
    N = len(metric_array)
    failure_mask = np.zeros(N, dtype=bool)

    if onsets is None or len(onsets) == 0:
        return failure_mask

    # Check each cycle for NaN values
    for i in range(len(onsets)):
        start = int(onsets[i])
        end = int(onsets[i + 1]) if i + 1 < len(onsets) else N

        cycle_segment = metric_array[start:end]

        # If entire cycle is NaN or has any NaN, mark as failure
        if len(cycle_segment) > 0 and np.all(np.isnan(cycle_segment)):
            failure_mask[start:end] = True

    return failure_mask


def identify_problematic_breaths(t: np.ndarray,
                                 y: np.ndarray,
                                 sr_hz: float,
                                 peaks: np.ndarray,
                                 onsets: np.ndarray,
                                 offsets: np.ndarray,
                                 expmins: np.ndarray,
                                 expoffs: np.ndarray,
                                 metrics_dict: Dict[str, np.ndarray],
                                 outlier_threshold: float = 3.0,
                                 outlier_metrics: list = None,
                                 global_stats: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify problematic breath cycles using outlier detection and failure detection.

    Args:
        t, y, sr_hz: Time, signal, sample rate
        peaks, onsets, offsets, expmins, expoffs: Breath event indices
        metrics_dict: Dictionary of computed metrics (from metrics.METRICS)
        outlier_threshold: Number of standard deviations for outlier detection
        outlier_metrics: List of metric keys to check (default: all numeric metrics)
        global_stats: Dict[metric_key, (mean, std)] - Pre-computed global statistics across all sweeps
                     If None, compute statistics from current sweep only (per-sweep mode)

    Returns:
        Tuple of (outlier_mask, failure_mask):
        - outlier_mask: Boolean mask marking outlier regions (orange highlighting)
        - failure_mask: Boolean mask marking calculation failure regions (red highlighting)
    """
    N = len(y)
    outlier_mask = np.zeros(N, dtype=bool)
    failure_mask = np.zeros(N, dtype=bool)

    if onsets is None or len(onsets) < 3:
        return outlier_mask, failure_mask

    # Metrics to check for outliers (use provided list or default)
    if outlier_metrics is None:
        outlier_metrics = ["if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"]

    mode = "cross-sweep" if global_stats is not None else "per-sweep"
    print(f"\n=== Breath Outlier Detection ({mode}) ===")
    print(f"Analyzing {len(onsets)} breath cycles")

    # Check each metric for outliers and failures
    for metric_key in outlier_metrics:
        if metric_key not in metrics_dict:
            continue

        metric_array = metrics_dict[metric_key]

        # Get global statistics for this metric (if available)
        global_mean = None
        global_std = None
        if global_stats and metric_key in global_stats:
            global_mean, global_std = global_stats[metric_key]

        # Detect outliers (orange)
        metric_outlier_mask = detect_metric_outliers(
            metric_array, onsets, n_std=outlier_threshold,
            global_mean=global_mean, global_std=global_std
        )
        outlier_count = np.sum(metric_outlier_mask)

        # Detect calculation failures (red)
        metric_failure_mask = detect_calculation_failures(metric_array, onsets)
        failure_count = np.sum(metric_failure_mask)

        if outlier_count > 0 or failure_count > 0:
            print(f"  {metric_key}: {outlier_count} outlier points, {failure_count} failure points")
            outlier_mask = outlier_mask | metric_outlier_mask
            failure_mask = failure_mask | metric_failure_mask

    total_outlier_points = np.sum(outlier_mask)
    total_failure_points = np.sum(failure_mask)
    print(f"Total outlier points: {total_outlier_points}/{N} ({100*total_outlier_points/N:.1f}%)")
    print(f"Total failure points: {total_failure_points}/{N} ({100*total_failure_points/N:.1f}%)")
    print(f"================================\n")

    return outlier_mask, failure_mask


def get_problematic_cycle_details(onsets: np.ndarray,
                                  problem_mask: np.ndarray) -> List[Dict]:
    """
    Get details about which specific cycles are problematic.

    Returns list of dicts with cycle_idx and whether it's problematic.
    """
    N = len(problem_mask)
    cycle_details = []

    if onsets is None or len(onsets) == 0:
        return cycle_details

    for i in range(len(onsets)):
        start = int(onsets[i])
        end = int(onsets[i + 1]) if i + 1 < len(onsets) else N

        # Check if any part of this cycle is problematic
        is_problematic = np.any(problem_mask[start:end])

        cycle_details.append({
            'cycle_idx': i,
            'start': start,
            'end': end,
            'is_problematic': is_problematic
        })

    return cycle_details


def compute_global_metric_statistics(all_metrics_by_sweep: Dict[int, Dict[str, np.ndarray]],
                                     all_onsets_by_sweep: Dict[int, np.ndarray],
                                     outlier_metrics: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute global statistics (mean, std) for each metric across ALL sweeps.

    Args:
        all_metrics_by_sweep: Dict[sweep_idx, Dict[metric_key, metric_array]]
        all_onsets_by_sweep: Dict[sweep_idx, onsets_array]
        outlier_metrics: List of metric keys to compute statistics for

    Returns:
        Dict[metric_key, (mean, std)] - Global statistics for each metric
    """
    global_stats = {}

    for metric_key in outlier_metrics:
        # Collect all cycle values for this metric across all sweeps
        all_cycle_values = []

        for sweep_idx in all_metrics_by_sweep:
            if sweep_idx not in all_onsets_by_sweep:
                continue

            metrics_dict = all_metrics_by_sweep[sweep_idx]
            onsets = all_onsets_by_sweep[sweep_idx]

            if metric_key not in metrics_dict:
                continue

            metric_array = metrics_dict[metric_key]
            N = len(metric_array)

            # Extract per-cycle values for this sweep
            for i in range(len(onsets)):
                start = int(onsets[i])
                end = int(onsets[i + 1]) if i + 1 < len(onsets) else N

                cycle_segment = metric_array[start:end]
                valid_vals = cycle_segment[~np.isnan(cycle_segment)]

                if len(valid_vals) > 0:
                    cycle_value = valid_vals[0]
                    all_cycle_values.append(cycle_value)

        # Compute global statistics from all collected cycle values
        if len(all_cycle_values) >= 3:
            all_cycle_values = np.array(all_cycle_values)
            mean_val = np.mean(all_cycle_values)
            std_val = np.std(all_cycle_values)
            global_stats[metric_key] = (mean_val, std_val)
            print(f"[global-stats] {metric_key}: mean={mean_val:.3f}, std={std_val:.3f} ({len(all_cycle_values)} cycles)")

    return global_stats