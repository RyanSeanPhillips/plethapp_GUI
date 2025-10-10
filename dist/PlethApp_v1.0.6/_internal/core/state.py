from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Tuple


@dataclass
class AppState:
    # File & channels
    in_path: Optional[Path] = None
    file_info: List[Dict] = field(default_factory=list)  # For multi-file concatenation: [{'path', 'sweep_start', 'sweep_end'}]
    channel_names: List[str] = field(default_factory=list)
    analyze_chan: Optional[str] = None
    stim_chan: Optional[str] = None
    # core/state.py  (inside AppState dataclass)
    stim_onsets_by_sweep: dict[int, np.ndarray] = field(default_factory=dict)
    stim_offsets_by_sweep: dict[int, np.ndarray] = field(default_factory=dict)
    stim_spans_by_sweep: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    stim_metrics_by_sweep: dict[int, dict] = field(default_factory=dict)


    # Raw & processed
    sr_hz: Optional[float] = None
    sweeps: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n_samples, n_sweeps) or 2D
    t: Optional[np.ndarray] = None

    # Filters
    use_low: bool = False
    use_high: bool = False
    use_mean_sub: bool = False
    use_invert: bool = False
    low_hz: Optional[float] = None
    high_hz: Optional[float] = None
    mean_val: float = 0.0

    # Navigation
    sweep_idx: int = 0
    window_start_s: float = 0.0
    window_dur_s: float = 10.0

    # Peaks & edits
    peaks_by_sweep: Dict[int, np.ndarray] = field(default_factory=dict)
    omitted_points: Dict[int, List[int]] = field(default_factory=dict)       # sweep -> sample idxs
    omitted_ranges: Dict[int, List[Tuple[int,int]]] = field(default_factory=dict)  # sweep -> [(i0,i1), ...]


    # Cached processed data (per sweep)
    proc_cache: Dict[Tuple[str,int], np.ndarray] = field(default_factory=dict)  # (chan, sweep) -> y
