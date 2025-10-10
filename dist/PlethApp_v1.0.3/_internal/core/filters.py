# core/filters.py
from __future__ import annotations
import numpy as np
from functools import lru_cache

try:
    from scipy.signal import butter, sosfiltfilt
except Exception as e:
    raise RuntimeError(
        "SciPy is required for Butterworth filtering. Install with: pip install scipy"
    ) from e


# def mean_subtract(y: np.ndarray, mean_val: float | None) -> np.ndarray:
#     return y - (0.0 if mean_val is None else float(mean_val))


try:
    from scipy.ndimage import uniform_filter1d
    def _movmean(x: np.ndarray, n: int) -> np.ndarray:
        # Fast O(N) moving average with good edge handling
        return uniform_filter1d(x, size=n, mode='nearest')
except Exception:
    def _movmean(x: np.ndarray, n: int) -> np.ndarray:
        # Fallback: simple convolution
        if n <= 1:
            return x
        k = np.ones(int(n), dtype=float) / float(n)
        return np.convolve(x, k, mode="same")


def rolling_mean_subtract_1d(y: np.ndarray, sr_hz: float, window_seconds: float) -> np.ndarray:
    """
    Subtract a rolling average (centered approximate) from y.
    window_seconds: rolling window length in seconds (e.g., 5.0)
    """
    y = np.asarray(y, dtype=float)
    if window_seconds is None or window_seconds <= 0 or not np.isfinite(window_seconds):
        return y

    n = max(3, int(round(window_seconds * float(sr_hz))))
    # enforce odd size for nicer centering (optional)
    if n % 2 == 0:
        n += 1

    baseline = _movmean(y, n)
    return y - baseline




def invert(y: np.ndarray) -> np.ndarray:
    return -y


def _sanitize_cutoff(c: float, nyq: float) -> float:
    # keep within (0, nyq); also avoid exactly 0 or >= nyq which break butter()
    eps = 1e-9
    c = max(eps, min(float(c), nyq * (1 - 2e-3)))
    return c


@lru_cache(maxsize=128)
def _design_low(sr_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    nyq = 0.5 * sr_hz
    c = _sanitize_cutoff(cutoff_hz, nyq)
    wn = c / nyq
    return butter(order, wn, btype="lowpass", output="sos")


@lru_cache(maxsize=128)
def _design_high(sr_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    nyq = 0.5 * sr_hz
    c = _sanitize_cutoff(cutoff_hz, nyq)
    wn = c / nyq
    return butter(order, wn, btype="highpass", output="sos")


@lru_cache(maxsize=128)
def _design_band(sr_hz: float, low_cut_hz: float, high_cut_hz: float, order: int) -> np.ndarray:
    """
    Band-pass between low_cut_hz (HP edge) and high_cut_hz (LP edge).
    Requires: 0 < low_cut < high_cut < Nyquist
    """
    nyq = 0.5 * sr_hz
    lo = _sanitize_cutoff(low_cut_hz, nyq)
    hi = _sanitize_cutoff(high_cut_hz, nyq)
    if not (lo < hi):
        # Fallback: if invalid, return a low-pass at hi (safer default)
        return _design_low(sr_hz, hi, order)
    wn = (lo / nyq, hi / nyq)
    return butter(order, wn, btype="bandpass", output="sos")


def _sosfiltfilt_safe(sos: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    sosfiltfilt can raise if the signal is too short for padding.
    Fall back to sosfiltfilt with smaller padlen; if still bad, return original y.
    """
    try:
        return sosfiltfilt(sos, y, axis=0)
    except ValueError:
        # Try a minimal padding; if still too short, just return y
        try:
            # padlen must be less than data length; choose something conservative
            padlen = max(0, min(99, y.shape[0] // 3))
            return sosfiltfilt(sos, y, axis=0, padlen=padlen)
        except Exception:
            return y


def low_pass_1d(y: np.ndarray, sr_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = _design_low(sr_hz, cutoff_hz, order)
    return _sosfiltfilt_safe(sos, y)


def high_pass_1d(y: np.ndarray, sr_hz: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    sos = _design_high(sr_hz, cutoff_hz, order)
    return _sosfiltfilt_safe(sos, y)


def band_pass_1d(y: np.ndarray, sr_hz: float,
                 low_cut_hz: float, high_cut_hz: float, order: int = 4) -> np.ndarray:
    """
    Band-pass where low_cut_hz is the HIGH-PASS edge (lower bound),
    and high_cut_hz is the LOW-PASS edge (upper bound).
    """
    sos = _design_band(sr_hz, low_cut_hz, high_cut_hz, order)
    return _sosfiltfilt_safe(sos, y)


# def apply_all_1d(y: np.ndarray, sr_hz: float,
#                  use_low: bool,  low_hz: float | None,
#                  use_high: bool, high_hz: float | None,
#                  use_mean: bool, mean_val: float | None,
#                  use_inv: bool,
#                  order: int = 4) -> np.ndarray:
#     """
#     Applies mean subtraction, Butterworth HP/LP (or band-pass if both are enabled),
#     and optional inversion. Uses cached SOS for speed.
#     """
#     out = y

#     # Mean subtraction first (DC/baseline)
#     if use_mean and mean_val is not None:
#         out = mean_subtract(out, mean_val)

#     # Combine HP+LP into one band-pass when both are valid
#     has_lp = use_low and (low_hz is not None)
#     has_hp = use_high and (high_hz is not None)

#     if has_lp and has_hp:
#         # high_hz is lower bound (HP), low_hz is upper bound (LP)
#         out = band_pass_1d(out, sr_hz, low_cut_hz=high_hz, high_cut_hz=low_hz, order=order)
#     elif has_hp:
#         out = high_pass_1d(out, sr_hz, cutoff_hz=high_hz, order=order)
#     elif has_lp:
#         out = low_pass_1d(out, sr_hz, cutoff_hz=low_hz, order=order)

#     if use_inv:
#         out = invert(out)

#     return out

# def apply_all_1d(
#     y: np.ndarray, sr_hz: float,
#     use_low: bool,  low_hz: float | None,
#     use_high: bool, high_hz: float | None,
#     use_mean_sub: bool, mean_param: float | None,   # NOW interpreted as window_seconds
#     use_invert: bool
# ) -> np.ndarray:

#     x = np.asarray(y, dtype=float)

#     # Low-pass first (optional)
#     if use_low and low_hz is not None and low_hz > 0:
#         x = butter_lowpass_1d(x, sr_hz, low_hz)

#     # High-pass (optional)
#     if use_high and high_hz is not None and high_hz > 0:
#         x = butter_highpass_1d(x, sr_hz, high_hz)

#     # Rolling baseline subtraction (optional)
#     if use_mean_sub and mean_param is not None and mean_param > 0:
#         # mean_param is interpreted as window_seconds
#         x = rolling_mean_subtract_1d(x, sr_hz, mean_param)

#     if use_invert:
#         x = -x

#     return x

import numpy as np

def apply_all_1d(
    y: np.ndarray,
    sr_hz: float,
    use_low: bool,  low_hz: float | None,
    use_high: bool, high_hz: float | None,
    use_mean: bool, mean_param: float | None,
    use_inv: bool,
    order: int = 4,
    mean_mode: str = "window",   # "window" (rolling) or "const" (subtract a number)
) -> np.ndarray:
    """
    Pipeline:
    - (optional) constant mean subtraction  (if mean_mode == 'const')
    - Butterworth HP/LP or band-pass using existing helpers: high_pass_1d / low_pass_1d / band_pass_1d
    - (optional) rolling baseline subtraction (if mean_mode == 'window', mean_param = window_seconds)
    - (optional) invert

    NOTE: This intentionally uses your original filter functions so HP/LP keep working.
    """
    x = np.asarray(y, dtype=float)

    # 1) optional constant mean subtraction
    if use_mean and mean_param is not None and mean_mode == "const":
        # expects a numeric DC value to subtract
        x = mean_subtract(x, float(mean_param))

    # 2) HP/LP/BP (unchanged logic)
    has_lp = use_low  and (low_hz  is not None)
    has_hp = use_high and (high_hz is not None)

    if has_lp and has_hp:
        # Band-pass when HP cutoff (high_hz) < LP cutoff (low_hz)
        if float(high_hz) < float(low_hz):
            x = band_pass_1d(x, sr_hz, low_cut_hz=float(high_hz), high_cut_hz=float(low_hz), order=order)
        else:
            # Graceful fallback if user enters overlapping/invalid cutoffs:
            # apply sequential HP then LP (still yields a BP effect)
            x = high_pass_1d(x, sr_hz, cutoff_hz=float(high_hz), order=order)
            x = low_pass_1d(x,  sr_hz, cutoff_hz=float(low_hz),  order=order)
    elif has_hp:
        x = high_pass_1d(x, sr_hz, cutoff_hz=float(high_hz), order=order)
    elif has_lp:
        x = low_pass_1d(x,  sr_hz, cutoff_hz=float(low_hz),  order=order)

    # 3) optional rolling baseline subtraction (window in seconds)
    if use_mean and mean_param is not None and mean_mode == "window":
        x = rolling_mean_subtract_1d(x, sr_hz, window_seconds=float(mean_param))

    # 4) optional invert
    if use_inv:
        x = invert(x)

    return x
