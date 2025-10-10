# core/robust_metrics.py
"""
ROBUST METRICS MODULE

This module provides enhanced versions of breath analysis metrics with comprehensive
error handling, graceful degradation, and multiple fallback strategies.

Key improvements over standard metrics:
1. Graceful handling of missing or misaligned breath events
2. Multiple fallback strategies when primary methods fail
3. Bounds checking to prevent index errors
4. NaN handling without cascading failures
5. Consistent output array lengths regardless of input quality
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import warnings


def _safe_array_get(arr: Optional[np.ndarray], idx: int, default=np.nan):
    """Safely get array element with bounds checking."""
    if arr is None or len(arr) == 0:
        return default
    if 0 <= idx < len(arr):
        return arr[idx]
    return default


def _robust_cycle_iterator(peaks, onsets, offsets, expmins, expoffs, N):
    """
    Iterator that yields robust cycle information, handling mismatched array lengths.

    Yields:
        (cycle_idx, peak_idx, onset_idx, offset_idx, expmin_idx, expoff_idx, cycle_start, cycle_end)

    All indices are validated and may be None if data is missing.
    cycle_start and cycle_end are always valid bounds.
    """
    if peaks is None or len(peaks) == 0:
        return

    n_peaks = len(peaks)

    for i in range(n_peaks):
        peak_idx = peaks[i]

        # Determine cycle boundaries
        if onsets is not None and len(onsets) > i:
            cycle_start = onsets[i]
            if i + 1 < len(onsets):
                cycle_end = onsets[i + 1]
            else:
                cycle_end = N
        else:
            # Fallback to peak-based boundaries
            if i > 0:
                cycle_start = max(0, int(0.5 * (peaks[i-1] + peak_idx)))
            else:
                cycle_start = max(0, peak_idx - int(0.5 * (peaks[1] - peaks[0])) if n_peaks > 1 else 0)

            if i < n_peaks - 1:
                cycle_end = min(N, int(0.5 * (peak_idx + peaks[i+1])))
            else:
                cycle_end = N

        # Validate and get other event indices
        onset_idx = _safe_array_get(onsets, i, None)
        offset_idx = _safe_array_get(offsets, i, None)
        expmin_idx = _safe_array_get(expmins, i, None)
        expoff_idx = _safe_array_get(expoffs, i, None)

        yield (i, peak_idx, onset_idx, offset_idx, expmin_idx, expoff_idx, cycle_start, cycle_end)


def _fill_stepwise(N: int, boundaries: List[int], values: List[float]) -> np.ndarray:
    """Fill array with stepwise constant values between boundaries."""
    result = np.full(N, np.nan, dtype=float)

    for i, (start, value) in enumerate(zip(boundaries, values)):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else N
        if not np.isnan(value):
            result[start:end] = value

    return result


def robust_compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    ROBUST instantaneous frequency calculation with multiple fallback strategies.

    Strategies:
    1. Onset-to-onset intervals (preferred)
    2. Peak-to-peak intervals (fallback)
    3. Fixed reasonable estimate for isolated peaks
    """
    N = len(y)
    result = np.full(N, np.nan, dtype=float)

    if peaks is None or len(peaks) == 0:
        return result

    boundaries = []
    values = []

    # Use onsets if available, otherwise peaks
    if onsets is not None and len(onsets) >= 2:
        time_points = [t[max(0, min(idx, N-1))] for idx in onsets]
        boundaries = [max(0, min(idx, N-1)) for idx in onsets]
        reference_points = onsets
    elif len(peaks) >= 2:
        time_points = [t[max(0, min(idx, N-1))] for idx in peaks]
        boundaries = [max(0, min(idx, N-1)) for idx in peaks]
        reference_points = peaks
    else:
        # Single peak - estimate reasonable frequency
        boundaries = [0]
        values = [1.0]  # 1 Hz default
        return _fill_stepwise(N, boundaries, values)

    # Calculate intervals
    for i in range(len(time_points) - 1):
        dt = time_points[i + 1] - time_points[i]
        if dt > 0:
            freq_hz = 1.0 / dt
            # Sanity check - breathing frequency should be reasonable
            if 0.1 <= freq_hz <= 10.0:  # 0.1-10 Hz is reasonable for breathing
                values.append(freq_hz)
            else:
                values.append(np.nan)
        else:
            values.append(np.nan)

    # Extend last value to end
    if values:
        values.append(values[-1])
    else:
        values = [np.nan]

    return _fill_stepwise(N, boundaries, values)


def robust_compute_amp_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    ROBUST inspiratory amplitude calculation with comprehensive fallback strategies.

    Strategies:
    1. Peak amplitude relative to onset
    2. Peak amplitude relative to cycle start
    3. Peak amplitude relative to previous expiratory minimum
    4. Absolute peak amplitude
    """
    N = len(y)

    if peaks is None or len(peaks) == 0:
        return np.full(N, np.nan, dtype=float)

    boundaries = []
    values = []

    for cycle_data in _robust_cycle_iterator(peaks, onsets, offsets, expmins, expoffs, N):
        (i, peak_idx, onset_idx, offset_idx, expmin_idx, expoff_idx, cycle_start, cycle_end) = cycle_data

        boundaries.append(cycle_start)

        if peak_idx is None or peak_idx >= N:
            values.append(np.nan)
            continue

        peak_value = y[peak_idx]
        baseline_value = np.nan

        # Strategy 1: Use onset as baseline
        if onset_idx is not None and 0 <= onset_idx < N:
            baseline_value = y[onset_idx]
        # Strategy 2: Use cycle start as baseline
        elif 0 <= cycle_start < N:
            baseline_value = y[cycle_start]
        # Strategy 3: Use previous expiratory minimum
        elif i > 0 and expmins is not None and i-1 < len(expmins):
            prev_expmin = expmins[i-1]
            if 0 <= prev_expmin < N:
                baseline_value = y[prev_expmin]

        # Calculate amplitude
        if not np.isnan(baseline_value):
            amplitude = peak_value - baseline_value
        else:
            # Strategy 4: Absolute amplitude (less ideal but better than NaN)
            amplitude = peak_value

        values.append(amplitude)

    # Ensure we have final boundary
    if boundaries:
        if boundaries[-1] != N:
            # Extend last value to end
            if values:
                values.append(values[-1])
            boundaries.append(N)

    return _fill_stepwise(N, boundaries, values)


def robust_compute_ti(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    ROBUST inspiratory time calculation with multiple fallback strategies.

    Strategies:
    1. Onset to offset time (preferred)
    2. Estimated fraction of cycle time based on typical patterns
    3. Fixed reasonable estimate based on frequency
    """
    N = len(y)

    if peaks is None or len(peaks) == 0:
        return np.full(N, np.nan, dtype=float)

    boundaries = []
    values = []

    for cycle_data in _robust_cycle_iterator(peaks, onsets, offsets, expmins, expoffs, N):
        (i, peak_idx, onset_idx, offset_idx, expmin_idx, expoff_idx, cycle_start, cycle_end) = cycle_data

        boundaries.append(cycle_start)

        # Strategy 1: Direct onset to offset measurement
        if (onset_idx is not None and offset_idx is not None and
            0 <= onset_idx < N and 0 <= offset_idx < N and offset_idx > onset_idx):
            ti = t[offset_idx] - t[onset_idx]
            values.append(ti)
        # Strategy 2: Estimate based on cycle duration (typical Ti/Ttot ~ 0.4-0.5)
        elif cycle_end > cycle_start and cycle_end <= N:
            cycle_duration = t[min(cycle_end-1, N-1)] - t[cycle_start]
            estimated_ti = cycle_duration * 0.45  # Typical inspiratory fraction
            values.append(estimated_ti)
        else:
            values.append(np.nan)

    if boundaries and boundaries[-1] != N:
        if values:
            values.append(values[-1])
        boundaries.append(N)

    return _fill_stepwise(N, boundaries, values)


def create_robust_metrics_registry() -> Dict[str, Callable]:
    """Create registry of robust metrics functions."""
    return {
        "if": robust_compute_if,
        "amp_insp": robust_compute_amp_insp,
        "ti": robust_compute_ti,
        # Add more robust metrics as needed
    }


# Integration function to enhance existing metrics
def enhance_metrics_with_robust_fallbacks(original_metrics: Dict[str, Callable]) -> Dict[str, Callable]:
    """
    Enhance existing metrics dictionary with robust fallbacks.

    This function wraps existing metric functions to add error handling and
    fallback strategies while preserving the original behavior when possible.
    """
    robust_registry = create_robust_metrics_registry()
    enhanced_metrics = {}

    for key, original_func in original_metrics.items():
        if key in robust_registry:
            # Use robust version
            enhanced_metrics[key] = robust_registry[key]
        else:
            # Wrap original function with basic error handling
            def wrapped_metric(original=original_func):
                def safe_metric(*args, **kwargs):
                    try:
                        return original(*args, **kwargs)
                    except Exception as e:
                        # Log warning but don't crash
                        warnings.warn(f"Metric '{key}' failed: {e}. Returning NaN array.")
                        N = len(args[1]) if len(args) > 1 else 1000  # args[1] should be y
                        return np.full(N, np.nan, dtype=float)
                return safe_metric
            enhanced_metrics[key] = wrapped_metric()

    return enhanced_metrics