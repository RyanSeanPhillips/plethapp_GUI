from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

def load_abf(path: Path) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)
    """
    try:
        import pyabf
    except ImportError:
        raise RuntimeError("pyabf not installed. pip install pyabf")

    abf = pyabf.ABF(str(path))
    sr_hz = float(abf.dataRate)
    chan_names = [abf.adcNames[c] for c in range(abf.channelCount)]

    # Collect sweeps into arrays per channel
    sweeps_by_channel: Dict[str, np.ndarray] = {}
    n_sweeps = abf.sweepCount
    n_samples = abf.sweepPointCount

    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        M = np.empty((n_samples, n_sweeps), dtype=float)
        for s in range(n_sweeps):
            abf.setSweep(sweepNumber=s, channel=ch)
            M[:, s] = abf.sweepY
        sweeps_by_channel[chan_names[ch]] = M

    # time vector for one sweep
    t = np.arange(n_samples, dtype=float) / sr_hz
    return sr_hz, sweeps_by_channel, chan_names, t
