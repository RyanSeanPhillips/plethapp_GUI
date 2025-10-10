import numpy as np
from typing import Dict, List, Tuple

def add_peak(peaks_by_sweep: Dict[int, np.ndarray], sweep: int, idx: int):
    arr = peaks_by_sweep.get(sweep, np.empty(0, dtype=int))
    arr = np.unique(np.concatenate([arr, [idx]]))
    peaks_by_sweep[sweep] = arr

def delete_peak(peaks_by_sweep: Dict[int, np.ndarray], sweep: int, idx: int, tol: int = 3):
    arr = peaks_by_sweep.get(sweep, np.empty(0, dtype=int))
    if arr.size:
        arr = arr[np.abs(arr - idx) > tol]
    peaks_by_sweep[sweep] = arr

def omit_point(omitted_points: Dict[int, List[int]], sweep: int, idx: int):
    omitted_points.setdefault(sweep, []).append(idx)

def omit_range(omitted_ranges: Dict[int, List[Tuple[int,int]]], sweep: int, i0: int, i1: int):
    if i1 < i0:
        i0, i1 = i1, i0
    omitted_ranges.setdefault(sweep, []).append((i0, i1))
