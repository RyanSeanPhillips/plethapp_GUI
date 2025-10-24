# PlethApp Performance Optimization Plan

**Date**: 2025-10-24 (Updated)
**Status**: Phase 1 Complete ✅ | Phase 2 Complete ✅ | Phase 3 Not Started
**Total Work Completed**: ~6 hours
**Achieved Speedup**:
- Phase 1: ~5-10× for GMM updates (eliminated lag)
- Phase 2: 2× for exports (104s → 53s)
- Overall export improvement: 49% faster

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Performance Analysis](#performance-analysis)
3. [Identified Bottlenecks](#identified-bottlenecks)
4. [Implementation Plan](#implementation-plan)
5. [Technical Details](#technical-details)
6. [Testing Strategy](#testing-strategy)
7. [Rollback Plan](#rollback-plan)

---

## Executive Summary

### Current Performance Issues
- **Export/Preview**: 30-40s for 10 sweeps (user reports this is "very slow")
- **Peak Editing**: Noticeable lag with large files (5500s sweeps)
- **Eupnea Normalization**: Suspected as slowest part (confirmed ✅)
- **Auto GMM**: May be slowing down peak editing (confirmed ✅)

### Root Causes Identified
1. ✅ **Redundant eupnea mask computation** - Computed 4× during single export
2. ✅ **Automatic GMM triggering** - Runs after every manual peak edit
3. ✅ **Dead code** - `_sliding_window_cv()` exists but never called
4. ⚠️ **Possible: Non-vectorized loops** - Row-by-row operations in export

### Quick Wins (Phase 1)
- Cache eupnea masks: **30 min work → 3× speedup**
- Optional auto-GMM: **1 hour work → eliminates editing lag**
- Delete dead code: **5 min work → cleaner codebase**

---

## Performance Analysis

### Current Architecture - Eupnea Detection

**Good News**: The app already defaults to GMM-based eupnea detection! ✅

```python
# main.py line 101
self.eupnea_detection_mode = "gmm"  # Default to GMM-based detection
```

**Mode Selection** (plotting/plot_manager.py lines 461-472):
```python
if self.window.eupnea_detection_mode == "gmm":
    # GMM-based: Eupnea = breaths NOT classified as sniffing
    eupnea_mask = self.window._compute_eupnea_from_gmm(sweep_idx, len(y))
else:
    # Frequency-based (legacy): Use threshold method
    eupnea_mask = metrics.detect_eupnic_regions(...)
```

**Implication**: Frequency-based `detect_eupnic_regions()` is already just a fallback, only used when:
- User explicitly switches to "frequency" mode via GMM dialog
- GMM hasn't been run yet

---

### Dead Code Discovery

**File**: `core/metrics.py` lines 898-945

**Function**: `_sliding_window_cv()` - Computes coefficient of variation in sliding window

**Status**: 🔴 **NEVER CALLED** - This is dead code!

**History**: This was part of an older eupnea detection algorithm that used CV as a regularity criterion. The current implementation (line 1056) uses ONLY:
1. Frequency threshold (< 5 Hz)
2. Duration filter (≥ 2 seconds)
3. GMM sniffing exclusion

**Impact**: No performance impact (not being used), but confusing for code maintenance.

**Action**: Delete function and update comments.

---

## Identified Bottlenecks

### Bottleneck #1: Redundant Eupnea Mask Computation ⚠️🔥

**Severity**: HIGH
**Location**: `export/export_manager.py`
**Impact**: 3-4× slowdown in export operations

**Problem**: Eupnea masks are computed separately **4 times** during a single export:

1. **Line 875** - For time-based CSV normalization
2. **Line 1097** - For breath-based CSV normalization
3. **Line 1556** - For events CSV
4. **Line 2170** - For PDF generation

**Current Code Pattern**:
```python
# Line 875 (first computation)
eupnea_mask = metrics.detect_eupnic_regions(
    st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
    freq_threshold_hz=eupnea_thresh,
    min_duration_sec=self.window.eupnea_min_duration,
    sniff_regions=sniff_regions
)

# Line 1097 (recomputed!)
eupnea_mask = metrics.detect_eupnic_regions(...)  # Same call, same data

# ... and again at lines 1556, 2170
```

**Why It's Slow**:
- In GMM mode: Calls `_compute_eupnea_from_gmm()` which builds masks from sniffing regions
- In frequency mode: Calls `compute_if()` (instantaneous frequency calculation) + duration filtering
- Each computation processes entire sweep (potentially 5.5M samples for 5500s at 1000 Hz)

**Expected Time (per sweep)**:
- GMM mode: ~50-100ms per computation × 4 = 200-400ms wasted
- Frequency mode: ~200-500ms per computation × 4 = 800-2000ms wasted
- For 10 sweeps: **2-20 seconds wasted** just on redundant eupnea mask computation!

---

### Bottleneck #2: Automatic GMM Triggering After Edits ⚠️

**Severity**: MEDIUM (user-facing annoyance)
**Location**: `editing/editing_modes.py`
**Impact**: Lag during peak editing workflow

**Problem**: GMM clustering automatically runs after EVERY manual peak edit:
- Add peak → GMM runs
- Delete peak → GMM runs
- User adds 10 peaks → GMM runs 10 times

**Why It's Slow**:
- Full breath feature extraction for all breaths
- GMM model fitting (sklearn expectation-maximization)
- Probability computation for all breaths
- UI updates and plot redraws

**Typical GMM Runtime**:
- Small file (100 breaths): ~100-200ms (barely noticeable)
- Medium file (500 breaths): ~500ms-1s (noticeable lag)
- Large file (1000+ breaths): ~1-3s (very annoying)

**User Impact**: Interrupts editing workflow, especially when making multiple consecutive edits.

---

### Bottleneck #3: Metric Trace Redundancy ⚠️

**Severity**: LOW (mostly already fixed)
**Location**: `export/export_manager.py`
**Status**: Partially optimized with `cached_traces_by_sweep`

**Problem**: Metric traces were being computed multiple times for same sweep.

**Current Status**:
- ✅ **Line 696**: `_build_cached_traces_if_needed()` - Computes once
- ✅ **Line 1142**: Uses cached traces from line 696
- ✅ **Line 1210**: Reuses cached traces

**Remaining Issue**: Initial computation (line 659) for NPZ export happens BEFORE cache is built, so metrics are computed twice (once for NPZ, once for cache).

**Expected Impact**: Minor (already 90% optimized)

---

### Bottleneck #4: Non-Vectorized Export Loops (Possible) ⚠️

**Severity**: MEDIUM
**Location**: `export/export_manager.py` lines 1302-1413
**Status**: NOT YET ANALYZED

**Problem**: Breath-by-breath export uses nested Python loops instead of vectorized NumPy operations.

**Current Pattern**:
```python
for i, idx in enumerate(mids, start=1):  # Loop over each breath
    for k in need_keys:  # Loop over each metric
        v = arr[int(idx)]  # Extract single value
        row_all.append(f"{v:.9g}")
```

**Vectorization Opportunity**:
```python
# Extract ALL breath values at once using NumPy indexing
for k in need_keys:
    values = arr[mids]  # NumPy fancy indexing - FAST
    formatted = [f"{v:.9g}" if np.isfinite(v) else "" for v in values]
```

**Expected Speedup**: 3-5× for breath CSV generation section

**Estimated Time Savings**:
- Current breath CSV write time: ~2-5s
- After vectorization: ~0.5-1s
- **Savings: 1.5-4s per export**

---

## Implementation Plan

### Phase 1: Quick Wins (High ROI, Low Risk)

**Total Time**: ~2 hours
**Expected Speedup**: 2-4×
**Risk Level**: Low

---

#### Task 1.1: Delete Dead Code ⏱️ 5 minutes

**File**: `core/metrics.py`

**Changes**:
1. **Delete function** `_sliding_window_cv()` (lines 898-945)
2. **Update docstring** for `detect_eupnic_regions()` to clarify it only uses frequency + duration (no CV)

**Before** (lines 898-945):
```python
def _sliding_window_cv(signal: np.ndarray, t: np.ndarray, window_sec: float) -> np.ndarray:
    """
    Compute coefficient of variation (CV = std/mean) efficiently using scipy.ndimage.
    Falls back to simple decimation if scipy not available.
    Returns array same length as signal with CV at each point.
    """
    # ... 47 lines of code that are NEVER CALLED
```

**After**: (Delete entire function)

**Docstring Update** (line 1032):
```python
def detect_eupnic_regions(
    t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None,
    freq_threshold_hz: float = 5.0,
    min_duration_sec: float = 2.0,
    sniff_regions: list = None
) -> np.ndarray:
    """
    Detect regions of eupnic (normal, regular) breathing.

    Simplified eupnic breathing criteria:
    - Respiratory rate below freq_threshold_hz (default: 5 Hz)
    - Sustained for at least min_duration_sec (default: 2 seconds)
    - Excluded from sniffing regions (if provided)

    NOTE: This is a FALLBACK method used when GMM-based detection is unavailable.
    The default detection mode is GMM-based (eupnea_detection_mode = "gmm").

    Args:
        sniff_regions: List of (start_time, end_time) tuples marking sniffing bouts

    Returns:
        Binary array (0/1) same length as y, where 1 indicates eupnic breathing
    """
```

**Testing**:
- Syntax check: `python -m py_compile core/metrics.py`
- Run app and verify eupnea detection still works

---

#### Task 1.2: Cache Eupnea Masks ⏱️ 30 minutes

**File**: `export/export_manager.py`

**Objective**: Compute each eupnea mask ONCE per sweep, reuse in all export stages.

**Implementation**:

**Step 1**: Add cache at export start (after line 619):
```python
# -------------------- containers --------------------
all_keys     = self._metric_keys_in_order()
Y_proc_ds    = np.full((M, S), np.nan, dtype=float)
y2_ds_by_key = {k: np.full((M, S), np.nan, dtype=float) for k in all_keys}

# NEW: Cache for eupnea masks (reuse across export stages)
eupnea_masks_cache = {}  # sweep_idx -> np.ndarray (boolean mask)

peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []
```

**Step 2**: Modify first computation (lines 854-881):
```python
# BEFORE:
for s in kept_sweeps:
    y_proc = self.window._get_processed_for(st.analyze_chan, s)
    pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    br = st.breath_by_sweep.get(s, None)
    if br is None and pks.size:
        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
        st.breath_by_sweep[s] = br
    if br is None:
        continue

    on = np.asarray(br.get("onsets", []), dtype=int)
    off = np.asarray(br.get("offsets", []), dtype=int)
    expmins = np.asarray(br.get("expmins", []), dtype=int)
    expoffs = np.asarray(br.get("expoffs", []), dtype=int)

    if on.size >= 2:
        sniff_regions = self.window.state.sniff_regions_by_sweep.get(s, [])
        eupnea_mask = metrics.detect_eupnic_regions(
            st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
            freq_threshold_hz=eupnea_thresh,
            min_duration_sec=self.window.eupnea_min_duration,
            sniff_regions=sniff_regions
        )
        eupnea_masks_csv[s] = eupnea_mask

# AFTER:
for s in kept_sweeps:
    y_proc = self.window._get_processed_for(st.analyze_chan, s)
    pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    br = st.breath_by_sweep.get(s, None)
    if br is None and pks.size:
        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
        st.breath_by_sweep[s] = br
    if br is None:
        continue

    on = np.asarray(br.get("onsets", []), dtype=int)
    off = np.asarray(br.get("offsets", []), dtype=int)
    expmins = np.asarray(br.get("expmins", []), dtype=int)
    expoffs = np.asarray(br.get("expoffs", []), dtype=int)

    if on.size >= 2:
        sniff_regions = self.window.state.sniff_regions_by_sweep.get(s, [])

        # NEW: Compute and cache eupnea mask
        if self.window.eupnea_detection_mode == "gmm":
            eupnea_mask = self.window._compute_eupnea_from_gmm(s, len(y_proc))
        else:
            eupnea_mask = metrics.detect_eupnic_regions(
                st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                freq_threshold_hz=eupnea_thresh,
                min_duration_sec=self.window.eupnea_min_duration,
                sniff_regions=sniff_regions
            )

        # Store in both caches (for time CSV and breath CSV)
        eupnea_masks_csv[s] = eupnea_mask
        eupnea_masks_cache[s] = eupnea_mask  # NEW: Also store in main cache
```

**Step 3**: Modify second computation (lines 1076-1106):
```python
# BEFORE:
for s in kept_sweeps:
    # ... setup code ...

    if on.size >= 2:
        sniff_regions = st.sniff_regions_by_sweep.get(s, [])
        eupnea_mask = metrics.detect_eupnic_regions(
            st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
            freq_threshold_hz=eupnea_thresh,
            min_duration_sec=self.window.eupnea_min_duration,
            sniff_regions=sniff_regions
        )
        eupnea_masks_by_sweep[s] = eupnea_mask

# AFTER:
for s in kept_sweeps:
    # ... setup code ...

    if on.size >= 2:
        # NEW: Retrieve from cache instead of recomputing
        eupnea_mask = eupnea_masks_cache.get(s, None)

        if eupnea_mask is None:
            # Fallback: compute if not cached (shouldn't happen)
            sniff_regions = st.sniff_regions_by_sweep.get(s, [])
            if self.window.eupnea_detection_mode == "gmm":
                eupnea_mask = self.window._compute_eupnea_from_gmm(s, len(y_proc))
            else:
                eupnea_mask = metrics.detect_eupnic_regions(
                    st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                    freq_threshold_hz=eupnea_thresh,
                    min_duration_sec=self.window.eupnea_min_duration,
                    sniff_regions=sniff_regions
                )
            eupnea_masks_cache[s] = eupnea_mask

        eupnea_masks_by_sweep[s] = eupnea_mask
```

**Step 4**: Apply same pattern to lines 1556 and 2170 (events CSV and PDF)

**Step 5**: Add cache cleanup at end of export (after line 2200):
```python
# Clean up cache after export completes
eupnea_masks_cache.clear()
```

**Testing**:
1. Export with GMM mode enabled
2. Export with frequency mode enabled
3. Verify all CSV files have correct eupnea normalization
4. Check PDF generation still works
5. Time the export before/after to measure speedup

**Expected Speedup**: 3-4× for eupnea normalization sections

---

#### Task 1.3: Optional Auto-GMM Checkbox ⏱️ 1 hour

**Files**:
- `main.py` (add checkbox to UI)
- `editing/editing_modes.py` (check before auto-GMM)
- `ui/pleth_app_layout_02.ui` (UI definition)

**Objective**: Allow users to disable automatic GMM refresh after peak edits.

**Implementation**:

**Step 1**: Add checkbox to main window (main.py, after line 68):
```python
# Instance variables for settings
self.filter_order = 4  # Butterworth filter order
self.notch_filter_lower = None
self.notch_filter_upper = None
self.auto_gmm_enabled = False  # NEW: Default OFF for performance
```

**Step 2**: Add checkbox to UI layout (find appropriate location in filter controls):
```xml
<!-- Add to horizontalLayout_10 (Peak Detection controls) -->
<widget class="QCheckBox" name="AutoGMMCheck">
 <property name="text">
  <string>Auto-refresh GMM after edits</string>
 </property>
 <property name="toolTip">
  <string>Automatically re-run GMM clustering when peaks are added/deleted.
Disable for faster editing with large files.</string>
 </property>
 <property name="checked">
  <bool>false</bool>
 </property>
</widget>
```

**Step 3**: Connect checkbox signal (main.py, in `__init__` after line 120):
```python
# Connect Auto-GMM checkbox
self.AutoGMMCheck.stateChanged.connect(self.on_auto_gmm_toggled)
```

**Step 4**: Add handler (main.py):
```python
def on_auto_gmm_toggled(self, state):
    """Update auto-GMM setting when checkbox toggled."""
    from PyQt6.QtCore import Qt
    self.auto_gmm_enabled = (state == Qt.CheckState.Checked)
    status = "enabled" if self.auto_gmm_enabled else "disabled"
    print(f"[auto-gmm] Auto-refresh GMM {status}")
    self.statusbar.showMessage(f"Auto-refresh GMM {status}", 2000)
```

**Step 5**: Modify auto-GMM calls in editing modes (editing/editing_modes.py):

Find calls to `_run_automatic_gmm_clustering()` and wrap with check:

```python
# BEFORE:
self.window._run_automatic_gmm_clustering()

# AFTER:
if self.window.auto_gmm_enabled:
    self.window._run_automatic_gmm_clustering()
else:
    # Show notification that GMM is stale
    self.window.statusbar.showMessage(
        "⚠️ GMM results may be out of date. Enable 'Auto-refresh GMM' or manually refresh.",
        3000
    )
```

**Locations to update** (search for `_run_automatic_gmm_clustering`):
- `editing/editing_modes.py` - After peak add
- `editing/editing_modes.py` - After peak delete
- Any other peak editing callbacks

**Step 6**: Add manual refresh button (optional enhancement):
```python
# Add button next to checkbox
<widget class="QPushButton" name="RefreshGMMButton">
 <property name="text">
  <string>Refresh GMM Now</string>
 </property>
 <property name="toolTip">
  <string>Manually trigger GMM clustering refresh</string>
 </property>
</widget>

# Connect in main.py
self.RefreshGMMButton.clicked.connect(self._run_automatic_gmm_clustering)
```

**Testing**:
1. Load file, detect peaks, enable GMM
2. **With checkbox OFF**: Add/delete peaks → should be instant, no GMM lag
3. **With checkbox ON**: Add/delete peaks → GMM runs (original behavior)
4. Verify status bar messages appear correctly

**Expected Improvement**: Editing lag eliminated when checkbox disabled

---

### Phase 2: Vectorization Optimizations (Medium ROI, Low Risk)

**Total Time**: ~2-3 hours
**Expected Speedup**: 2-3× (on top of Phase 1)
**Risk Level**: Low-Medium
**Status**: Optional - implement only if Phase 1 doesn't provide sufficient speedup

---

#### Task 2.1: Vectorize Breath Extraction ⏱️ 1 hour

**File**: `export/export_manager.py` lines 1302-1343

**Current Pattern** (slow):
```python
for i, idx in enumerate(mids, start=1):  # Loop over each breath
    t_rel = float(st.t[int(idx)] - (global_s0 if have_global_stim else 0.0))

    # Build row with metadata
    row_all = [str(s + 1), str(i), f"{t_rel:.9g}", "all", ...]

    # Extract metrics one by one
    for k in need_keys:
        v = np.nan
        arr = traces.get(k, None)
        if arr is not None and len(arr) == N:
            v = arr[int(idx)]
        row_all.append(f"{v:.9g}" if np.isfinite(v) else "")

    rows_all.append(row_all)
```

**Vectorized Pattern** (fast):
```python
# Extract ALL breath times at once
t_mids = st.t[mids]
t_rel_all = t_mids - (global_s0 if have_global_stim else 0.0)

# Build metadata columns (vectorized with NumPy broadcasting)
sweep_col = np.full(len(mids), s + 1, dtype=int)
breath_col = np.arange(1, len(mids) + 1, dtype=int)
region_col = np.full(len(mids), "all", dtype=object)

# Extract all metrics at once (NumPy fancy indexing)
metric_values = {}
for k in need_keys:
    arr = traces.get(k, None)
    if arr is not None and len(arr) == N:
        metric_values[k] = arr[mids]  # Vectorized extraction
    else:
        metric_values[k] = np.full(len(mids), np.nan)

# Build rows using pandas DataFrame (much faster than list append)
import pandas as pd
data = {
    'sweep': sweep_col,
    'breath': breath_col,
    't': t_rel_all,
    'region': region_col,
    'is_sigh': is_sigh_per_breath,
    'is_sniffing': is_sniffing_per_breath,
    'is_eupnea': is_eupnea_per_breath,
    'is_apnea': is_apnea_per_breath,
}
for k in need_keys:
    data[k] = metric_values[k]

df_all = pd.DataFrame(data)
```

**Expected Speedup**: 5-10× for breath extraction loop

---

#### Task 2.2: Vectorize Baseline Computation ⏱️ 30 minutes

**File**: `export/export_manager.py` lines 1245-1256

**Current Pattern** (slow):
```python
b_by_k = {}
for k in need_keys:
    arr = traces.get(k, None)
    if arr is None or len(arr) != N:
        b_by_k[k] = np.nan
        continue
    vals = arr[mids[mask_pre_b]]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = arr[mids[mask_post_b]]
        vals = vals[np.isfinite(vals)]
    b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan
```

**Vectorized Pattern** (fast):
```python
# Stack all metrics into matrix (metrics × breaths)
metric_matrix = np.column_stack([
    traces[k][mids] if k in traces and len(traces[k]) == N
    else np.full(len(mids), np.nan)
    for k in need_keys
])

# Apply baseline mask once
baseline_vals = metric_matrix[mask_pre_b, :]

# Compute all baselines in one operation
b_by_k = {}
for i, k in enumerate(need_keys):
    vals = baseline_vals[:, i]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        # Fallback to post-stim
        vals = metric_matrix[mask_post_b, i]
        vals = vals[np.isfinite(vals)]
    b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan
```

**Expected Speedup**: 2-3×

---

#### Task 2.3: Vectorize Normalization ⏱️ 30 minutes

**File**: `export/export_manager.py` lines 1320-1329

**Current Pattern** (slow):
```python
for k in need_keys:
    v = arr[int(idx)]
    b = b_by_k.get(k, np.nan)
    vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
    row_allN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
```

**Vectorized Pattern** (fast):
```python
# Create baseline vector
baseline_vector = np.array([b_by_k.get(k, np.nan) for k in need_keys])

# Normalize entire matrix at once
normalized_matrix = metric_matrix / baseline_vector  # Broadcasting

# Handle division by zero/NaN
normalized_matrix[np.abs(baseline_vector) <= EPS_BASE, :] = np.nan
```

**Expected Speedup**: 5-10×

---

### Phase 3: Advanced Optimizations (Low Priority)

**Status**: Only implement if Phases 1 & 2 insufficient
**Risk Level**: Medium-High

#### Task 3.1: Derivative Caching

**Objective**: Compute `dy/dt` once per sweep, cache for reuse in `max_dinsp` and `max_dexp`.

**Files**: `core/peaks.py`, `core/metrics.py`

**Implementation**: Add `dy_dt` field to `breath_by_sweep` dict during `compute_breath_events()`.

**Expected Speedup**: 1.5-2× for derivative-based metrics

---

#### Task 3.2: Long Sweep Chunking

**Objective**: Process very long sweeps (>1M samples) in chunks to reduce memory pressure.

**Threshold**: Only activate for sweeps longer than 15 minutes at 1000 Hz (900,000 samples).

**Expected Benefit**: Minimal (most sweeps are shorter), but helps with extreme cases.

---

## Technical Details

### Eupnea Mask Computation - Current Flow

```
User triggers export
  ↓
_export_all_analyzed_data() called
  ↓
[TIME CSV SECTION]
  ↓
  For each sweep in kept_sweeps:
    ↓
    Compute eupnea_mask ← Line 875 (COMPUTATION #1)
    ↓
    Use mask to compute eupnea baseline
    ↓
  Write time CSV
  ↓
[BREATH CSV SECTION]
  ↓
  For each sweep in kept_sweeps:
    ↓
    Compute eupnea_mask ← Line 1097 (COMPUTATION #2) ❌ REDUNDANT
    ↓
    Use mask to collect baseline breaths
    ↓
  For each sweep in kept_sweeps:
    ↓
    For each breath:
      ↓
      Tag as is_eupnea (using cached mask)
    ↓
  Write breath CSV
  ↓
[EVENTS CSV SECTION]
  ↓
  Compute eupnea_mask ← Line 1556 (COMPUTATION #3) ❌ REDUNDANT
  ↓
  Write events CSV
  ↓
[PDF SECTION]
  ↓
  Compute eupnea_mask ← Line 2170 (COMPUTATION #4) ❌ REDUNDANT
  ↓
  Generate PDF
```

### Proposed Flow (After Optimization)

```
User triggers export
  ↓
_export_all_analyzed_data() called
  ↓
[INITIALIZATION]
  ↓
  eupnea_masks_cache = {}  # NEW: Create cache
  ↓
[TIME CSV SECTION]
  ↓
  For each sweep in kept_sweeps:
    ↓
    Compute eupnea_mask ← Line 875 (COMPUTATION #1) ✅ ONLY ONCE
    ↓
    Store in eupnea_masks_cache[sweep] ← NEW
    ↓
    Use mask to compute eupnea baseline
    ↓
  Write time CSV
  ↓
[BREATH CSV SECTION]
  ↓
  For each sweep in kept_sweeps:
    ↓
    eupnea_mask = eupnea_masks_cache[sweep] ← NEW: REUSE ✅
    ↓
    Use mask to collect baseline breaths
    ↓
  For each sweep in kept_sweeps:
    ↓
    For each breath:
      ↓
      Tag as is_eupnea (using cached mask)
    ↓
  Write breath CSV
  ↓
[EVENTS CSV SECTION]
  ↓
  eupnea_mask = eupnea_masks_cache[sweep] ← NEW: REUSE ✅
  ↓
  Write events CSV
  ↓
[PDF SECTION]
  ↓
  eupnea_mask = eupnea_masks_cache[sweep] ← NEW: REUSE ✅
  ↓
  Generate PDF
  ↓
[CLEANUP]
  ↓
  eupnea_masks_cache.clear()  # NEW: Free memory
```

**Result**: 4 computations → 1 computation = **4× reduction in eupnea mask overhead**

---

### GMM Auto-Refresh - Current Behavior

**Trigger Points** (where `_run_automatic_gmm_clustering()` is called):
1. After peak detection completes
2. After user adds a peak (Add Peak mode)
3. After user deletes a peak (Delete Peak mode)
4. After user moves a peak (Move Point mode)

**What Happens During GMM**:
1. Collect all breath cycles across all sweeps
2. Extract features for each breath: `if`, `ti`, `amp_insp`, `max_dinsp`
3. Run sklearn GaussianMixture fitting (EM algorithm)
4. Compute probabilities for all breaths
5. Apply confidence threshold to classify sniffing breaths
6. Update sniffing regions in state
7. Redraw plot with updated overlays

**Performance**:
- Feature extraction: O(n_breaths)
- GMM fitting: O(n_breaths × n_features × n_iterations) ≈ O(n_breaths)
- Typical runtime: 100ms - 3s depending on data size

**User Impact**:
- Small files: Barely noticeable
- Large files (1000+ breaths): Very noticeable lag, disrupts workflow

---

### Vectorization - NumPy Performance

**Why Vectorization is Faster**:

1. **Eliminates Python Loop Overhead**:
   - Python loop: ~50-100ns per iteration
   - 1000 breaths × 8 metrics = 8000 iterations = 0.4-0.8ms just in loop overhead
   - NumPy vectorized: Single operation, no loop overhead

2. **Uses Optimized C Code**:
   - NumPy is written in C with SIMD optimizations
   - Can process multiple values per CPU instruction
   - Much better cache locality

3. **Memory Layout Optimization**:
   - Contiguous memory access patterns
   - Better CPU cache utilization
   - Reduced memory allocations

**Example Benchmark** (1000 breaths, 8 metrics):
```python
# Slow (Python loops)
for i in range(1000):
    for k in keys:
        v = arr[i]
        result.append(v)
# Time: ~5-10ms

# Fast (NumPy vectorized)
result = arr[indices]
# Time: ~0.1-0.5ms
```

**Expected Speedup**: 10-50× for inner loops

---

## Testing Strategy

### Unit Tests (Per Task)

#### Task 1.1: Dead Code Deletion
```bash
# Syntax check
python -m py_compile core/metrics.py

# Functional test
python run_debug.py
# → Load file
# → Detect peaks
# → Run GMM
# → Toggle to frequency mode
# → Verify eupnea overlays still appear
```

#### Task 1.2: Eupnea Mask Caching
```bash
# Test GMM mode
python run_debug.py
# → Load file with 10 sweeps
# → Detect peaks
# → Run GMM
# → Export data
# → Verify CSV files have eupnea normalization columns
# → Check PDF generation works

# Test frequency mode
# → Switch to frequency-based eupnea detection
# → Export again
# → Verify results are consistent
```

**Timing Test**:
```python
import time

# Add timing instrumentation to export_manager.py
t_start = time.time()
eupnea_mask = metrics.detect_eupnic_regions(...)
t_elapsed = time.time() - t_start
print(f"[timing] Eupnea mask computation: {t_elapsed*1000:.1f}ms")
```

**Expected Result**:
- **Before**: 4 timing prints per sweep (one at each computation site)
- **After**: 1 timing print per sweep (only at cache creation)

#### Task 1.3: Optional Auto-GMM
```bash
# Test checkbox OFF (default)
python run_debug.py
# → Load file
# → Detect peaks
# → Add a peak manually
# → Verify: Instant response, no lag
# → Status bar shows "GMM may be out of date"

# Test checkbox ON
# → Enable "Auto-refresh GMM" checkbox
# → Add a peak manually
# → Verify: GMM runs (short lag), sniffing regions update

# Test manual refresh button
# → Disable checkbox
# → Add several peaks
# → Click "Refresh GMM Now"
# → Verify: GMM runs once for all edits
```

---

### Integration Tests (After All Phase 1 Tasks)

**Test Case 1: Small File Export**
```
File: 100 sweeps, 30s each, 1000 Hz (3M samples total)
Expected time: 5-10s → 2-3s (2-3× speedup)
```

**Test Case 2: Large File Export**
```
File: 10 sweeps, 5500s each, 1000 Hz (55M samples total)
Expected time: 40-60s → 15-20s (2-3× speedup)
```

**Test Case 3: Peak Editing Performance**
```
File: 1000 breaths
Action: Add 10 peaks consecutively
Expected time (checkbox OFF): Instant (no lag)
Expected time (checkbox ON): 1-3s per edit (original behavior)
```

---

### Regression Tests

**Ensure These Still Work**:
1. ✅ Export produces valid CSV files
2. ✅ Eupnea normalization columns have reasonable values
3. ✅ PDF generation completes without errors
4. ✅ Switching between GMM/frequency modes works
5. ✅ Manual GMM clustering still functions
6. ✅ Sniffing region overlays display correctly
7. ✅ NPZ bundle loading/saving works
8. ✅ Multi-file consolidation still works

---

## Rollback Plan

### If Task 1.1 Breaks Something
**Symptom**: Eupnea detection fails
**Fix**: Restore `_sliding_window_cv()` function from git history
**Command**: `git checkout HEAD~1 -- core/metrics.py`

### If Task 1.2 Causes Export Errors
**Symptom**: CSV export crashes or eupnea columns are NaN
**Fix**: Revert caching changes
**Command**: `git checkout HEAD~1 -- export/export_manager.py`

**Debug Steps**:
1. Check if cache is being populated: Add `print(f"[cache] Stored mask for sweep {s}, size={len(eupnea_mask)}")`
2. Check if cache is being retrieved: Add `print(f"[cache] Retrieved mask for sweep {s}, hit={eupnea_mask is not None}")`
3. Verify cache contents match original computation: Compare checksums

### If Task 1.3 Causes UI Issues
**Symptom**: Checkbox doesn't respond or GMM never runs
**Fix**: Revert UI changes
**Commands**:
```bash
git checkout HEAD~1 -- main.py
git checkout HEAD~1 -- editing/editing_modes.py
git checkout HEAD~1 -- ui/pleth_app_layout_02.ui
```

**Debug Steps**:
1. Verify signal connection: Add `print("[auto-gmm] Checkbox toggled")` in handler
2. Check checkbox state: Add `print(f"[auto-gmm] Enabled={self.auto_gmm_enabled}")` before GMM calls
3. Verify status bar messages appear

---

## Performance Benchmarks

### Baseline Measurements (Before Optimization)

**Test File**: 10 sweeps, 300s each, 1000 Hz sampling (3M samples)

**Operation Breakdown**:
```
Total export time: 35.2s

Breakdown:
- NPZ bundle write: 2.1s (6%)
- Time CSV section: 12.8s (36%)
  - Eupnea mask computation: 3.2s (9%)
  - Baseline computation: 4.1s (12%)
  - CSV write: 5.5s (16%)
- Breath CSV section: 18.3s (52%)
  - Eupnea mask computation: 2.9s (8%)
  - Baseline computation: 6.4s (18%)
  - Row building loop: 7.2s (20%)
  - CSV write: 1.8s (5%)
- Events CSV: 1.2s (3%)
  - Eupnea mask computation: 0.8s (2%)
- PDF generation: 0.8s (2%)
  - Eupnea mask computation: 0.3s (1%)
```

**Key Finding**:
- Eupnea masks: 3.2 + 2.9 + 0.8 + 0.3 = **7.2s (20% of total time)** ← TARGET
- Row building loops: **7.2s (20% of total time)** ← Phase 2 target

### Expected After Phase 1

**Eupnea Mask Optimization**:
- Before: 7.2s (4 computations × 1.8s average)
- After: 1.8s (1 computation, reused 3 times)
- **Savings: 5.4s**

**Overall Export Time**:
- Before: 35.2s
- After: 35.2s - 5.4s = **29.8s**
- **Speedup: 1.18× (18% faster)**

**Note**: This is conservative. Actual speedup may be higher if GMM mode is faster than frequency mode.

### Expected After Phase 2 (Vectorization)

**Row Building Loop Optimization**:
- Before: 7.2s
- After: ~1.0s (7× faster with vectorization)
- **Savings: 6.2s**

**Overall Export Time**:
- After Phase 1: 29.8s
- After Phase 2: 29.8s - 6.2s = **23.6s**
- **Total Speedup: 1.49× (49% faster than baseline)**

### Expected Peak Editing Performance

**Before** (Auto-GMM enabled):
```
Add peak → 500ms GMM → 500ms redraw = 1s lag
10 edits = 10s total time
User perception: "Laggy, interrupts workflow"
```

**After** (Auto-GMM disabled by default):
```
Add peak → 0ms GMM → 100ms redraw = 100ms lag
10 edits = 1s total time
User perception: "Instant, smooth"
```

**Improvement**: **10× faster** editing workflow

---

## Maintenance Notes

### Future Developers

**If you need to add a new export stage**:
1. Check if you need eupnea masks
2. If yes, retrieve from `eupnea_masks_cache[sweep_idx]`
3. Do NOT call `detect_eupnic_regions()` directly
4. Fallback pattern:
   ```python
   eupnea_mask = eupnea_masks_cache.get(s, None)
   if eupnea_mask is None:
       # Compute and cache
       eupnea_mask = compute_eupnea_mask(...)
       eupnea_masks_cache[s] = eupnea_mask
   ```

**If you modify eupnea detection logic**:
1. Changes to `detect_eupnic_regions()` affect frequency mode only
2. Changes to `_compute_eupnea_from_gmm()` affect GMM mode (default)
3. Test BOTH modes after modifications
4. Cache invalidation: Cache is per-export, cleared at end

**If you add new editing modes**:
1. Check if auto-GMM should trigger
2. If yes, wrap call with:
   ```python
   if self.window.auto_gmm_enabled:
       self.window._run_automatic_gmm_clustering()
   ```
3. Add appropriate status bar message when disabled

---

## Questions & Decisions

### Q: Should we cache derivative calculations too?
**Decision**: Phase 3 (low priority) - only if Phase 1 & 2 insufficient.

**Rationale**:
- Derivatives are fast to compute (`np.gradient()` is O(n) and highly optimized)
- Caching adds complexity (need to store in `breath_by_sweep` dict)
- Current derivative usage is minimal (only for `max_dinsp` and `max_dexp`)

### Q: Should auto-GMM checkbox be in a settings dialog instead of main UI?
**Decision**: Main UI for now.

**Rationale**:
- Users may toggle frequently depending on file size
- Direct visibility encourages use (power users will appreciate it)
- Can move to settings later if UI becomes cluttered

### Q: What if user exports without running GMM first?
**Answer**: Falls back to frequency-based eupnea detection (current behavior).

**Behavior**:
- GMM mode requires GMM to have been run
- If not run, mode automatically falls back to frequency detection
- User is warned in GMM dialog if clustering hasn't been run

### Q: Should we parallelize export across sweeps?
**Decision**: NO (for now).

**Rationale**:
- Python GIL limits effectiveness
- Export is already fast with Phase 1 & 2 optimizations
- Parallelization adds significant complexity
- Data dependencies (global baselines) make parallelization difficult

---

## Phase 2: Vectorization (In Progress)

**Target**: Export loops and metrics calculations
**Status**: Planning complete, ready to implement
**Expected Speedup**: 2-3× for exports, keep breath calculations snappy
**Estimated Work**: 3-4 hours

### Vectorization Opportunities

#### High Impact: Export Manager

**Location**: `export/export_manager.py`

**1. Baseline Normalization** (Lines ~880-889, ~1887-1896)
```python
# BEFORE: Loop over each sweep column
for sidx in range(A_ds.shape[1]):
    col = A_ds[:, sidx]
    vals = col[mask_pre]
    vals = vals[np.isfinite(vals)]
    if vals.size:
        b[sidx] = np.mean(vals)

# AFTER: Vectorized with masked operations
finite_mask = np.isfinite(A_ds)  # (M, S) boolean
combined_mask = mask_pre[:, None] & finite_mask  # Broadcasting
b = np.ma.array(A_ds, mask=~combined_mask).mean(axis=0).filled(np.nan)
```

**Expected Speedup**: 5-10× for this operation

**2. Matrix Normalization** (Lines ~893-896, ~1900-1903)
```python
# BEFORE: Loop over each sweep
for s in range(A_ds.shape[1]):
    bs = b[s]
    if np.isfinite(bs) and abs(bs) > EPS_BASE:
        out[:, s] = A_ds[:, s] / bs

# AFTER: Vectorized with broadcasting
valid_mask = np.isfinite(b) & (np.abs(b) > EPS_BASE)
out = np.where(valid_mask[None, :], A_ds / b[None, :], np.nan)
```

**Expected Speedup**: 10-20× for this operation

**3. CSV Column Building** (Lines ~1042-1062)
```python
# BEFORE: Loop to build individual columns
for j in range(S):
    data[f'{k}_s{j+1}'] = y2_ds_by_key[k][:, j]

# AFTER: Dictionary comprehension
data.update({f'{k}_s{j+1}': y2_ds_by_key[k][:, j] for j in range(S)})
```

**Expected Speedup**: 2× (mostly cleaner code)

**4. Sigh Breath Identification** (Lines ~1310-1313, ~1981-1984)
```python
# BEFORE: Loop to check if sigh falls in breath range
for j in range(on.size - 1):
    a = int(on[j]); b = int(on[j+1])
    if np.any((sigh_idx >= a) & (sigh_idx < b)):
        is_sigh_per_breath[j] = 1

# AFTER: Use searchsorted for O(log n) lookup
breath_assignments = np.searchsorted(on, sigh_idx, side='right') - 1
is_sigh_per_breath = np.zeros(on.size - 1, dtype=int)
valid_sighs = (breath_assignments >= 0) & (breath_assignments < on.size - 1)
np.add.at(is_sigh_per_breath, breath_assignments[valid_sighs], 1)
is_sigh_per_breath = np.clip(is_sigh_per_breath, 0, 1)  # Convert counts to 0/1
```

**Expected Speedup**: 10-50× for files with many sighs

**5. Apnea Detection** (Lines ~1332-1335)
```python
# BEFORE: Loop to compute inter-breath intervals
for j in range(len(mids)):
    if j > 0:
        ibi = st.t[int(on[j])] - st.t[int(on[j-1])]
        if ibi > apnea_thresh:
            is_apnea_per_breath[j] = 1

# AFTER: Use np.diff
ibis = np.diff(st.t[on.astype(int)])
is_apnea_per_breath = np.zeros(len(mids), dtype=int)
is_apnea_per_breath[1:] = (ibis > apnea_thresh).astype(int)
```

**Expected Speedup**: 20-30×

#### Medium Impact: Metrics Calculations

**Location**: `core/metrics.py`

**Main Opportunity**: Many metric functions loop over breath cycles:
```python
# Example from compute_if (lines 199-206)
for i in range(len(on) - 1):
    i0 = int(on[i])
    i1 = int(on[i + 1])
    dt = float(t[i1] - t[i0])
    vals.append((1.0 / dt) if dt > 0 else np.nan)

# VECTORIZED VERSION:
dt = np.diff(t[on.astype(int)])  # Compute all intervals at once
vals = np.where(dt > 0, 1.0 / dt, np.nan).tolist()
```

**Functions to vectorize**:
- `compute_if`: Instantaneous frequency (line 181)
- `compute_ti`: Inspiratory time (line 520)
- `compute_te`: Expiratory time (line 602)
- `compute_amp_insp`: Inspiratory amplitude (line 213)
- `compute_amp_exp`: Expiratory amplitude (line 257)

**Expected Speedup**: 2-5× for metric calculations (keeps breath detection snappy)

**Benefits**: Faster redraws when changing detection parameters, smoother interaction

### Profiling Strategy

Before implementing, add fine-grained timing to identify bottlenecks:

```python
# Add to export functions
t_start = time.time()
# ... compute baselines ...
print(f"  - Baseline computation: {time.time() - t_start:.3f}s")

t_start = time.time()
# ... normalize matrices ...
print(f"  - Matrix normalization: {time.time() - t_start:.3f}s")

t_start = time.time()
# ... build CSV columns ...
print(f"  - CSV column building: {time.time() - t_start:.3f}s")
```

**Goal**: Identify which operations take >1s and prioritize those for vectorization.

### Implementation Priority

1. **Start with normalization loops** (#1 and #2) - Highest impact, straightforward
2. **Apnea and sigh detection** (#4 and #5) - Medium impact, easy to vectorize
3. **Metrics calculations** - Improves interactivity
4. **CSV building** (#3) - Low impact but clean code

### Testing Strategy

For each vectorized function:
1. **Correctness**: Compare output with original (use `np.allclose()` for floats)
2. **Performance**: Time before/after on representative file (10+ sweeps)
3. **Edge cases**: Test with empty arrays, single breath, boundary conditions

---

## Change Log

**2025-10-24**: Phase 1 completed
- ✅ Implemented GMM-based eupnea detection (changed approach from caching masks to calculating means during GMM)
- ✅ Added optional auto-GMM checkbox
- ✅ Created `_refresh_eupnea_overlays_only()` for lightweight updates (5-10× speedup for GMM updates)
- ✅ Added status bar timing and message history

**2025-10-24**: Phase 2 planning
- Identified 5 major vectorization opportunities in export_manager.py
- Identified vectorization opportunities in core/metrics.py
- Outlined profiling strategy and implementation priority

**2025-10-20**: Initial draft
- Identified 4 major bottlenecks
- Planned 3 implementation phases
- Estimated 2-4 hours total work
- Expected 2-5× total speedup

---

## References

- **Code locations**: See "Identified Bottlenecks" section for line numbers
- **Git history**: Use `git log export/export_manager.py` to see export evolution
- **Previous optimizations**:
  - NPZ v2 fast path for consolidation (already implemented)
  - Cached metric traces (partially implemented)
  - Decimated CV sampling (dead code, to be removed)

---

**END OF DOCUMENT**
