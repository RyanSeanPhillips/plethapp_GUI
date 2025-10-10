# core/stim.py
from __future__ import annotations
import numpy as np
from typing import Tuple, List, Dict

def detect_threshold_crossings(
    y: np.ndarray,
    t: np.ndarray,
    thresh: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]], Dict[str, float]]:
    """
    Returns:
      on_idx: np.ndarray of sample indices where y crosses upward through +thresh
      off_idx: np.ndarray of sample indices where y crosses downward through +thresh
      spans_s: list of (t_on, t_off) in seconds (paired windows; unpaired tail is ignored)
      metrics: dict with 'pulse_width_s', 'duration_s', 'freq_hz' (if inferable)

    Notes:
      - Crossings use sign change on (y - thresh).
      - If the trace starts above threshold, we wait for the first downward crossing
        to close any leading span.
      - Frequency is estimated from consecutive onset-onset intervals (period).
        If only one pulse exists, 'freq_hz' is omitted.
    """
    y_shift = y - thresh
    above = y_shift > 0

    # Upward 0->1 transitions = onsets; downward 1->0 transitions = offsets
    on_idx = np.where((~above[:-1]) & (above[1:]))[0] + 1
    off_idx = np.where((above[:-1]) & (~above[1:]))[0] + 1

    # Pair them into (on, off) with order constraints
    spans: List[Tuple[int, int]] = []
    i = j = 0
    n_on, n_off = len(on_idx), len(off_idx)
    while i < n_on and j < n_off:
        # Ensure onset precedes offset
        if on_idx[i] < off_idx[j]:
            spans.append((on_idx[i], off_idx[j]))
            i += 1
            j += 1
        else:
            # Offset before any onset: skip offset
            j += 1

    # Convert spans to time (seconds)
    spans_s = [(float(t[i0]), float(t[i1])) for (i0, i1) in spans]

    # Metrics
    metrics: Dict[str, float] = {}
    if spans:
        widths = [t[i1] - t[i0] for (i0, i1) in spans]
        metrics["pulse_width_s"] = float(np.median(widths))  # robust typical width

        duration = spans_s[-1][1] - spans_s[0][0]
        metrics["duration_s"] = float(duration)

        # Frequency from onset->onset period (preferred)
        if len(on_idx) >= 2:
            periods = np.diff(t[on_idx])
            pos_periods = periods[periods > 0]
            if len(pos_periods):
                metrics["freq_hz"] = float(1.0 / np.median(pos_periods))
        # (If you *really* want 1/(offset(1)-onset(1)), that's the inverse pulse width;
        # we already expose pulse_width_s above.)
    return on_idx, off_idx, spans_s, metrics
