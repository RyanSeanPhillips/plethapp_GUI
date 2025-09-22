# core/metrics.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable

"""
Scaffold for y2 metrics derived from breath features.

Each compute_* function must return an array of shape (N,) where N == len(y),
ideally stepwise-constant over breath spans (NaN where undefined).
Inputs:
  - t:        time (s), shape (N,)
  - y:        processed pleth, shape (N,)
  - sr_hz:    sample rate (Hz)
  - peaks:    inspiratory peaks (np.int indices)
  - onsets:   breath onset indices (np.int)
  - offsets:  breath offset indices (np.int)
  - expmins:  expiratory minima indices (np.int)
"""

# (human label, key)
METRIC_SPECS: List[Tuple[str, str]] = [
    ("Instantaneous frequency (Hz)",          "if"),
    ("Inspiratory amplitude",                 "amp_insp"),
    ("Expiratory amplitude",                  "amp_exp"),
    ("Inspiratory area",                      "area_insp"),
    ("Expiratory area",                       "area_exp"),
    ("Ti (on→off)",                           "ti"),
    ("Te (off→next on)",                      "te"),
    ("Vent proxy (insp area / cycle dur)",    "vent_proxy"),
    ("d/dt (1st derivative)",                 "d1"),
    ("d²/dt² (2nd derivative)",               "d2"),
]


def _step_fill(n: int, spans: List[Tuple[int, int]], values: List[float]) -> np.ndarray:
    """Fill piecewise-constant values across [i0, i1) spans; NaN elsewhere."""
    out = np.full(n, np.nan, dtype=float)
    for (i0, i1), v in zip(spans, values):
        i0c = max(0, int(i0))
        i1c = min(n, int(i1))
        if i1c > i0c:
            out[i0c:i1c] = v
    return out


def _breath_spans_from_onsets(onsets: np.ndarray, N: int) -> List[Tuple[int, int]]:
    """Onset→next onset spans (last one runs to N)."""
    if onsets is None or len(onsets) == 0:
        return []
    spans: List[Tuple[int, int]] = []
    for i in range(len(onsets) - 1):
        spans.append((int(onsets[i]), int(onsets[i + 1])))
    spans.append((int(onsets[-1]), N))
    return spans


def _spans_from_bounds(bounds: np.ndarray, N: int) -> List[Tuple[int, int]]:
    """
    Convert a sorted array of boundary indices (e.g., onsets or peaks)
    into half-open spans [(b0,b1), (b1,b2), ..., (b_last, N)].
    """
    bounds = np.asarray(bounds, dtype=int)
    if bounds.size == 0:
        return []
    spans: List[Tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        spans.append((int(bounds[i]), int(bounds[i + 1])))
    spans.append((int(bounds[-1]), int(N)))
    return spans


# Add near the other helpers
def _trapz_seg(t: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
    """
    Trapezoid integral of y over inclusive indices [i0, i1].
    Returns NaN if the segment is empty or invalid.
    """
    N = len(y)
    i0c = max(0, min(int(i0), N - 1))
    i1c = max(0, min(int(i1), N - 1))
    if i1c <= i0c:
        return np.nan
    return float(np.trapz(y[i0c:i1c + 1], t[i0c:i1c + 1]))


# ---------- STUB COMPUTATIONS (return NaNs for now) ----------

################################################################################
#####IF Calculation
################################################################################
# def compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
#     """
#     Instantaneous frequency as a stepwise signal (breaths/min).
#     Prefer onset→onset cycle times; fall back to peak→peak if onsets unavailable.
#     """
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     # choose cycle boundaries
#     bounds = None
#     if onsets is not None and len(onsets) >= 2:
#         bounds = np.asarray(onsets, dtype=int)
#     elif peaks is not None and len(peaks) >= 2:
#         bounds = np.asarray(peaks, dtype=int)

#     if bounds is None or len(bounds) < 2:
#         return out  # not enough info

#     # compute BPM for each cycle
#     vals = []
#     spans = _spans_from_bounds(bounds, N)
#     for i in range(len(bounds) - 1):
#         i0, i1 = int(bounds[i]), int(bounds[i + 1])
#         dt = float(t[i1] - t[i0])
#         bpm = 60.0 / dt if dt > 0 else np.nan
#         vals.append(bpm)
#     # extend last span with last BPM value
#     vals.append(vals[-1] if len(vals) else np.nan)

#     return _step_fill(N, spans, vals)

def compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None):
    """
    Instantaneous frequency in Hz (not breaths/min).
    Stepwise constant over onset→next-onset spans.
    """
    import numpy as np

    N = len(y)
    out = np.full(N, np.nan, dtype=float)
    if onsets is None or len(onsets) < 2:
        return out

    on = np.asarray(onsets, dtype=int)

    # Reuse your helpers if you have them in metrics.py
    spans = _spans_from_bounds(on, N)

    vals = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])
        dt = float(t[i1] - t[i0])
        vals.append((1.0 / dt) if dt > 0 else np.nan)  # Hz

    # extend last span with last value (same behavior as your other metrics)
    vals.append(vals[-1] if vals else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####INSP AMP Calculation
################################################################################
def compute_amp_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Inspiratory amplitude per cycle: peak - onset.
    We take the FIRST peak inside each onset→next-onset span.
    If offsets are provided and that peak is beyond the offset, mark NaN.
    Returns stepwise-constant array over onset-bounded spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])
        # first peak in this cycle
        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        # if offsets exist, ensure peak lies before the offset for this cycle
        if offsets is not None and i < len(offsets):
            off_i = int(offsets[i])
            if p > off_i:
                vals.append(np.nan)
                continue

        vals.append(float(y[p] - y[i0]))

    # extend last span to N with the last valid value (consistent with IF)
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####ExpAmP Calculation
################################################################################
def compute_amp_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Expiratory amplitude per cycle: next_onset - expiratory_minimum_in_cycle.
    Prefer the precomputed expiratory minima; if none inside the cycle,
    fall back to argmin(y) over that cycle window.
    Returns stepwise-constant array over onset-bounded spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2:
        return out

    on = np.asarray(onsets, dtype=int)
    em = np.asarray(expmins, dtype=int) if (expmins is not None and len(expmins)) else np.array([], dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        exp_idx = None
        if em.size:
            mask = (em >= i0) & (em < i1)
            if np.any(mask):
                exp_idx = int(em[mask][0])

        # fallback: minimum of y within the cycle if no labeled expmin
        if exp_idx is None:
            seg = y[i0:i1]
            if seg.size == 0:
                vals.append(np.nan)
                continue
            exp_idx = int(i0 + int(np.argmin(seg)))

        # guard
        if exp_idx >= i1:
            vals.append(np.nan)
            continue

        # vals.append(float(y[i1] - y[exp_idx]))
        vals.append(float(y[exp_idx] - y[i1]))


    # extend last span with most recent value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)




################################################################################
#####Insp Area Calculation
################################################################################
def compute_area_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Inspiratory area per cycle: integral of y from onset -> offset.
    Returns a stepwise-constant array over onset→next-onset spans.
    If an offset is missing or out of range for a cycle, that cycle is NaN.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 1 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    # Step spans are always onset -> next onset (last extends to N)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # Need a valid offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])

        # Guard: offset must lie within (i0, i1]
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        # Trapezoid integral over [i0, oi] (inclusive of oi via +1 slice end)
        left = max(0, i0)
        right = min(N, oi + 1)
        if right - left < 2:
            vals.append(np.nan)
            continue
        area = float(np.trapz(y[left:right], t[left:right]))
        vals.append(area)

    # Extend last span to N with last valid value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####Exp Area Calculation
################################################################################

# def compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins) -> np.ndarray:
#     """
#     Expiratory area per cycle: integral of y from offset -> next onset.
#     Returns a stepwise-constant array over onset→next-onset spans.
#     If offset is missing for a cycle, that cycle is NaN.
#     """
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
#         return out

#     on = np.asarray(onsets, dtype=int)
#     off = np.asarray(offsets, dtype=int)

#     spans = _spans_from_bounds(on, N)

#     vals: list[float] = []
#     for i in range(len(on) - 1):
#         i0 = int(on[i])
#         i1 = int(on[i + 1])

#         if i >= len(off):
#             vals.append(np.nan)
#             continue
#         oi = int(off[i])

#         # Guard: offset must lie within [i0, i1)
#         if not (i0 <= oi < i1):
#             vals.append(np.nan)
#             continue

#         # Trapezoid integral over [oi, i1]
#         left = max(0, oi)
#         right = min(N, i1 + 1)
#         if right - left < 2:
#             vals.append(np.nan)
#             continue
#         area = float(np.trapz(y[left:right], t[left:right]))
#         vals.append(area)

#     # Extend last span with last value
#     vals.append(vals[-1] if len(vals) else np.nan)
#     return _step_fill(N, spans, vals)

def compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Expiratory area per cycle: integral of y from inspiratory offset -> expiratory offset.
    Returns a stepwise-constant array over onset→next-onset spans.

    If an expiratory offset for a cycle is missing or invalid, that cycle is NaN.
    (Optionally, you could fall back to next onset if you want the old behavior.)
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on  = np.asarray(onsets,  dtype=int)
    off = np.asarray(offsets, dtype=int)
    exo = np.asarray(expoffs, dtype=int) if (expoffs is not None and len(expoffs)) else np.array([], dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])         # onset (start of cycle)
        i1 = int(on[i + 1])     # next onset (end of cycle)

        # need a valid inspiratory offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])

        # need a valid expiratory offset for this cycle
        if i >= len(exo):
            vals.append(np.nan)
            continue
        ei = int(exo[i])

        # guards: oi must be inside cycle, and expoff must be after oi and before next onset
        if not (i0 <= oi < i1) or not (oi < ei <= i1):
            vals.append(np.nan)
            continue

        # trapezoid integral over [oi, ei] (inclusive of ei via +1 end)
        left  = max(0, oi)
        right = min(N, ei + 1)
        if right - left < 2:
            vals.append(np.nan)
            continue

        area = float(np.trapz(y[left:right], t[left:right]))
        vals.append(area)

    # extend last span with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)



################################################################################
#####Ti Calculation
################################################################################
def compute_ti(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Ti duration per cycle (seconds): time from onset -> offset.
    Returns a stepwise-constant array over onset→next-onset spans.
    If no valid offset for a cycle, that cycle is NaN.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 1 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])

        # offset must lie in (i0, i1]
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        dt = float(t[oi] - t[i0])
        vals.append(dt if dt > 0 else np.nan)

    # extend last span to N with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####Te Calculation
################################################################################
# def compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins) -> np.ndarray:
#     """
#     Te duration per cycle (seconds): time from offset -> next onset.
#     Returns a stepwise-constant array over onset→next-onset spans.
#     If no valid offset for a cycle, that cycle is NaN.
#     """
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
#         return out

#     on = np.asarray(onsets, dtype=int)
#     off = np.asarray(offsets, dtype=int)

#     spans = _spans_from_bounds(on, N)

#     vals: list[float] = []
#     for i in range(len(on) - 1):
#         i0 = int(on[i])
#         i1 = int(on[i + 1])

#         if i >= len(off):
#             vals.append(np.nan)
#             continue
#         oi = int(off[i])

#         # offset must lie in [i0, i1)
#         if not (i0 <= oi < i1):
#             vals.append(np.nan)
#             continue

#         dt = float(t[i1] - t[oi])
#         vals.append(dt if dt > 0 else np.nan)

#     # extend last span to N with last value
#     vals.append(vals[-1] if len(vals) else np.nan)
#     return _step_fill(N, spans, vals)

def compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Te per cycle (seconds): time from inspiratory offset -> expiratory offset.
    Returns a stepwise-constant array over onset→next-onset spans.
    If either offset or expiratory offset for a cycle is missing/invalid, that cycle is NaN.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if (onsets is None or len(onsets) < 2 or
        offsets is None or len(offsets) < 1 or
        expoffs is None or len(expoffs) < 1):
        return out

    on  = np.asarray(onsets,  dtype=int)
    off = np.asarray(offsets, dtype=int)
    exo = np.asarray(expoffs, dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])         # onset (start of cycle)
        i1 = int(on[i + 1])     # next onset (end of cycle)

        if i >= len(off) or i >= len(exo):
            vals.append(np.nan)
            continue

        oi = int(off[i])        # inspiratory offset
        ei = int(exo[i])        # expiratory offset

        # Guards: offset inside cycle; expoff strictly after offset and before/equal next onset
        if not (i0 <= oi < i1) or not (oi < ei <= i1):
            vals.append(np.nan)
            continue

        dt = float(t[ei] - t[oi])
        vals.append(dt if dt > 0 else np.nan)

    # Extend last span with last value
    vals.append(vals[-1] if vals else np.nan)
    return _step_fill(N, spans, vals)





################################################################################
#####Vent proxy Calculation
################################################################################
def compute_vent_proxy(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Vent proxy = (inspiratory area) / (cycle duration), in arbitrary units per second.
    Inspiratory area: integral from onset -> offset.
    Cycle duration: onset -> next onset.
    Returns a stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # need a valid offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])
        # offset must lie inside the cycle
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        area_insp = _trapz_seg(t, y, i0, oi)
        dur = float(t[i1] - t[i0])
        vals.append(area_insp / dur if (dur > 0 and np.isfinite(area_insp)) else np.nan)

    # extend final span with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)

################################################################################
#####d/dt(pleth signal) Calculation
################################################################################
def compute_d1(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    First derivative dy/dt using np.gradient with the time vector (same length as y).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y) or len(t) < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, t)

################################################################################
#####d^2/dt^2(pleth signal) Calculation
################################################################################
def compute_d2(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Second derivative d²y/dt² via two successive gradients with the time vector.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y) or len(t) < 3:
        return np.full_like(y, np.nan, dtype=float)
    d1 = np.gradient(y, t)
    return np.gradient(d1, t)



# Registry: key -> function
METRICS: Dict[str, Callable] = {
    "if":         compute_if,
    "amp_insp":   compute_amp_insp,
    "amp_exp":    compute_amp_exp,
    "area_insp":  compute_area_insp,
    "area_exp":   compute_area_exp,
    "ti":         compute_ti,
    "te":         compute_te,
    "vent_proxy": compute_vent_proxy,
    "d1":         compute_d1,
    "d2":         compute_d2,
}
