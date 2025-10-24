# PlethApp - Recent Feature Additions

This document describes recently implemented features, including implementation details and instructions for removal if needed.

---

## Spectral Analysis Window (2025-10-02)

A comprehensive spectral analysis tool for identifying and filtering oscillatory noise contamination.

### Features

#### Power Spectrum (Welch Method)
- High-resolution frequency analysis (0-30 Hz range optimized for breathing)
- **Separate spectra** for:
  - Full trace (blue line)
  - During-stimulation periods only (orange line)
- **Parameters**: nperseg=32768, 90% overlap for maximum resolution
- **Use case**: Identify oscillatory noise (e.g., 50/60 Hz electrical noise, pump vibration)

#### Wavelet Scalogram
- Time-frequency analysis using complex Morlet wavelets
- **Frequency range**: 0.5-30 Hz
- **Time normalization**: Time axis normalized to stimulation onset (t=0)
- **Color scaling**: Percentile-based (95th) to handle transient sniffing bouts
- **Stim markers**: Lime green dashed lines at stimulation on/offset
- **Use case**: Visualize time-varying frequency content (e.g., sniffing bouts)

#### Notch Filter Controls
- Interactive band-stop filter configuration
- User specifies lower and upper frequency bounds
- 4th-order Butterworth band-stop filter
- Applied to main signal when dialog is closed
- **Use case**: Remove specific frequency ranges (e.g., 3-5 Hz noise from pump)

#### Sweep Navigation
- Step through sweeps within the spectral analysis view
- Synchronized with main plot sweep selection

#### Aligned Panels
- GridSpec layout ensures power spectrum and wavelet plots have matching edges
- Professional appearance for publications

### Implementation Details

**Location**: `main.py` as `SpectralAnalysisDialog` class (lines ~1970-2330)

**UI Integration**:
- Button: `SpectralAnalysisButton` in filter controls (horizontalLayout_9)
- Button connection: `self.SpectralAnalysisButton.clicked.connect(self.on_spectral_analysis_clicked)` (line ~116)

**Signal Processing Pipeline**:
- Notch filter integrated into `_current_trace()` (lines ~621-623)
- Filter method: `_apply_notch_filter()` (lines ~1164-1193)
- Filter parameters included in cache key `_proc_key()` (lines ~369-370)

**Instance Variables** (lines ~63-65):
```python
self.notch_lower = None  # Lower frequency bound for notch filter (Hz)
self.notch_upper = None  # Upper frequency bound for notch filter (Hz)
```

### How to Remove This Feature

If you need to remove the spectral analysis feature:

1. **Delete `SpectralAnalysisDialog` class** from `main.py` (lines ~1970-2330)
2. **Remove button from UI**: Delete `SpectralAnalysisButton` from `ui/pleth_app_layout_02.ui` (horizontalLayout_9)
3. **Remove notch filter code** from `_current_trace()` (lines ~621-623):
   ```python
   # Remove these lines:
   if self.notch_lower is not None and self.notch_upper is not None:
       y = self._apply_notch_filter(y, sr)
   ```
4. **Delete `_apply_notch_filter()` method** (lines ~1164-1193)
5. **Remove notch filter from cache key** in `_proc_key()` (lines ~369-370):
   ```python
   # Remove these lines:
   self.notch_lower,
   self.notch_upper,
   ```
6. **Remove instance variables** from `__init__()` (lines ~63-65):
   ```python
   # Remove these lines:
   self.notch_lower = None
   self.notch_upper = None
   ```
7. **Remove button connection** from `__init__()` (line ~116):
   ```python
   # Remove this line:
   self.SpectralAnalysisButton.clicked.connect(self.on_spectral_analysis_clicked)
   ```
8. **Remove `on_spectral_analysis_clicked()` method** (lines ~1816-1863)

---

## Adjustable Filter Order (2025-10-02)

Added UI control for Butterworth filter order to enable stronger frequency attenuation.

### Features

- **Filter Order Spinbox**: Range 2-10, default 4
  - Located in filter controls (horizontalLayout_9)
  - Higher order = steeper roll-off at cutoff frequency
  - More aggressive elimination of frequencies beyond cutoff

- **Cache Integration**: Filter order included in processing cache key
  - Changes trigger automatic signal reprocessing

- **Real-time Updates**: Immediate visual feedback when order is changed

### Use Cases

- **Order 2-4**: Gentle filtering, preserves signal shape, minimal phase distortion
- **Order 6-8**: Stronger filtering, sharper cutoff, useful for noisy signals
- **Order 10**: Maximum attenuation, use when strong noise isolation is needed

### Implementation Details

**UI Widget**: `FilterOrderSpin` (QSpinBox) in `ui/pleth_app_layout_02.ui` (lines ~834-853)

**Label**: `FilterOrderLabel` (lines ~813-831)

**Signal Connection**: Connected to `update_and_redraw()` via `valueChanged` signal (line ~107)

**Storage**: Stored in `self.filter_order` instance variable (line ~68)

**Application**: Passed to `filters.apply_all_1d()` as `order` parameter (line ~618)

**Cache Key**: Included in `_proc_key()` for cache invalidation (line ~368)

### How to Remove This Feature

If you need to revert to fixed filter order:

1. **Delete UI widgets** from `ui/pleth_app_layout_02.ui` (lines ~812-854):
   - `FilterOrderLabel`
   - `FilterOrderSpin`

2. **Remove instance variable** from `__init__()` (line ~68):
   ```python
   # Remove this line:
   self.filter_order = 4  # Default filter order
   ```

3. **Remove signal connection** from `__init__()` (line ~107):
   ```python
   # Remove this line:
   self.FilterOrderSpin.valueChanged.connect(self.update_and_redraw)
   ```

4. **Remove filter order update** in `update_and_redraw()` (line ~544):
   ```python
   # Remove this line:
   self.filter_order = self.FilterOrderSpin.value()
   ```

5. **Remove parameter from filter call** in `_current_trace()` (line ~618):
   ```python
   # Change this:
   y = filters.apply_all_1d(y, sr, hp=hp, lp=lp, order=self.filter_order, ...)
   # To this:
   y = filters.apply_all_1d(y, sr, hp=hp, lp=lp, order=4, ...)  # Fixed order
   ```

6. **Remove from cache key** in `_proc_key()` (line ~368):
   ```python
   # Remove this line from the tuple:
   self.filter_order,
   ```

---

## Manual Peak Editing Enhancements (2025-10-02)

Improved peak editing workflow with keyboard modifiers and precision controls.

### Features

#### Keyboard Shortcuts
- **Shift key**: Toggle to Delete Peak mode from Add Peak mode (and vice versa)
- **Ctrl key**: Switch to Add Sigh mode from any mode
- Allows quick mode switching without button clicks
- Reduces workflow interruptions during manual editing

#### Precise Peak Deletion
- **New behavior**: Only deletes the single closest peak within ±80ms window
- **Previous behavior**: Deleted all peaks within window (could accidentally delete multiple)
- **Method**: Uses `np.argmin(distances)` to find closest peak

### Implementation Details

**Mode Switching**:
- In `_on_plot_click_add_peak()` and `_on_plot_click_delete_peak()` (editing/editing_modes.py)
- Uses `QApplication.keyboardModifiers()` to detect Shift/Ctrl
- `_force_mode` parameter prevents infinite recursion

**Button Labels**:
- Updated to show shortcuts (e.g., "Add Peak (Shift: Delete)")
- Provides visual reminder of keyboard shortcuts

**Peak Deletion Logic**:
```python
# Old behavior (deleted all peaks in window):
pks_new = pks[(pks < i_lo) | (pks > i_hi)]

# New behavior (deletes only closest peak):
distances = np.abs(pks - i_click)
closest_idx = np.argmin(distances)
if distances[closest_idx] <= half_win_n:
    pks_new = np.delete(pks, closest_idx)
```

### How to Revert to Previous Behavior

If you want to remove keyboard shortcuts and revert deletion behavior:

1. **Remove modifier key detection** from `_on_plot_click_add_peak()` and `_on_plot_click_delete_peak()`:
   ```python
   # Remove these lines:
   modifiers = QApplication.keyboardModifiers()
   if modifiers == Qt.KeyboardModifier.ShiftModifier:
       return self._on_plot_click_delete_peak(event, _force_mode=True)
   ```

2. **Change delete logic back** to delete all peaks in window:
   ```python
   # Change this:
   distances = np.abs(pks - i_click)
   closest_idx = np.argmin(distances)
   if distances[closest_idx] > half_win_n:
       return
   pks_new = np.delete(pks, closest_idx)

   # Back to this:
   pks_new = pks[(pks < i_lo) | (pks > i_hi)]
   ```

3. **Update button labels** to remove shortcut hints:
   ```python
   # Change "Add Peak (Shift: Delete)" back to "Add Peak"
   ```

---

## Status Bar Enhancements (2025-10-24)

Comprehensive status bar improvements for better user feedback and workflow transparency.

### Features

#### Dark Theme Status Bar
- Black background (#1e1e1e) matching application theme
- White text (#d4d4d4) for high contrast
- Border at top (#3e3e42) for visual separation
- Resize grip removed (dots on right side)

#### Timing Messages
Operation timing displayed for:
- **File loading**: "✓ File loaded (2.3s)"
- **Peak detection**: "✓ Peak detection complete (1.8s)"
- **GMM clustering**: "✓ GMM clustering complete (0.5s)"
- **Data export**: "✓ Data export complete (47.2s)"
- **Summary generation**: "✓ Summary generated (8.6s)"

#### Peak Edit Feedback
- **Success messages**: "✓ Peak added", "✓ Peak deleted"
- **Error messages**:
  - "✗ Peak too close (< 0.08s)"
  - "✗ Click too far from peak (> 0.08s)"
- **Warning restoration**: Out-of-date warnings automatically restore after temporary messages

#### Message History
- **History button**: 📋 icon on right side of status bar
- **Dropdown menu**: Shows last 20 messages with timestamps
- **Storage**: Keeps last 100 messages in memory
- **Format**: "[HH:MM:SS] Message text"

### Implementation Details

**Status Bar Styling** (main.py, lines 71-83):
```python
self.statusBar().setStyleSheet("""
    QStatusBar {
        background-color: #1e1e1e;
        color: #d4d4d4;
        border-top: 1px solid #3e3e42;
    }
    QStatusBar::item {
        border: none;
    }
""")
self.statusBar().setSizeGripEnabled(False)
```

**Message History** (main.py, lines 85-87, 395-465):
- `self._status_message_history = []` - List of (timestamp, message) tuples
- `_setup_status_history_dropdown()` - Creates 📋 button
- `_show_message_history()` - Displays dropdown menu
- `_log_status_message()` - Centralized logging function

**Timing Integration**:
- File loading (lines 441, 586-587)
- Peak detection (lines 1243, 1336, 1401)
- GMM clustering (lines 1432, 1453, 1586, 1591)
- Export (export/export_manager.py, lines 255-287)
- Preview (export/export_manager.py, lines 297-328, 2523-2525)

**Peak Edit Feedback** (editing/editing_modes.py):
- Success (lines 455, 601)
- Errors (lines 413-418, 591-596)
- Warning restoration via QTimer.singleShot (2100ms delay)

### Performance Optimizations

**Lightweight GMM Refresh** (main.py, lines 2020-2075):
- `_refresh_eupnea_overlays_only()` method
- Updates only eupnea/sniffing overlays without full redraw
- Skips expensive outlier detection (5-10× faster)
- Used after manual GMM update and auto-GMM on peak edits

**Results**:
- Manual GMM update: ~0.1-0.2s (was ~1s)
- Auto-GMM on peak edits: Much more responsive
- Overall app feels "snappier"

### How to Revert Status Bar Changes

If you need to revert to the original status bar:

1. **Remove dark theme styling** from `__init__()` (lines 71-83)
2. **Remove message history** tracking (lines 85-87, 395-465)
3. **Replace all** `_log_status_message()` calls with `statusBar().showMessage()`
4. **Remove timing code** from all operations (use grep to find `t_start = time.time()`)
5. **Remove peak edit feedback** from editing_modes.py

---

## Performance Profiling and Export Optimization (2025-10-24)

Comprehensive performance analysis and vectorization implementation that achieved 49% export speedup.

### Features

#### Line Profiler Integration
- **Automated profiling workflow** with `run_profiling.bat`
- **Zero-overhead @profile decorator** with no-op fallback
- **Timestamped results** automatically saved to `profiling_results/` folder
- **Complete guide** in PROFILING_GUIDE.md

**Workflow**:
1. Run `run_profiling.bat`
2. Use app normally (load file, detect peaks, export)
3. Close app
4. Profiling report automatically generates with timestamp

**Results Format**:
```
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   502      4000     10300000   2575.0      9.9%      ds_to_orig_idx[i] = np.argmin(...)
```

#### Export Vectorization
**Problem**: Original export took 104 seconds due to Python loops

**Solution**: Replaced loops with NumPy vectorized operations

##### 1. Index Mapping Optimization (Lines 957-970 in export_manager.py)
**Before**: Python loop with np.argmin (10.3s for 4000 timepoints)
```python
for i, t_target in enumerate(t_targets):
    ds_to_orig_idx[i] = np.argmin(np.abs(st.t - t_target))
```

**After**: Binary search with np.searchsorted (0.14ms - 73,000× faster)
```python
insert_idx = np.searchsorted(st.t, t_targets)
insert_idx = np.clip(insert_idx, 1, len(st.t) - 1)
left_dist = np.abs(st.t[insert_idx - 1] - t_targets)
right_dist = np.abs(st.t[insert_idx] - t_targets)
ds_to_orig_idx = np.where(left_dist < right_dist, insert_idx - 1, insert_idx)
```

**Technical Notes**:
- Uses O(log n) binary search instead of O(n) linear search
- Memory efficient: ~32KB instead of 19GB from naive broadcasting
- Handles edge cases with np.clip

##### 2. Mask-to-Intervals Vectorization (Lines 1624-1639 in export_manager.py)
**Before**: Python loop (10s for typical dataset)
```python
intervals = []
in_region = False
start_idx = None
for i, val in enumerate(mask):
    if val and not in_region:
        start_idx = i
        in_region = True
    elif not val and in_region:
        intervals.append((start_idx, i-1))
        in_region = False
```

**After**: Vectorized transition detection (0.1s - 100× faster)
```python
def mask_to_intervals(mask):
    if len(mask) == 0:
        return []

    padded = np.concatenate(([0], mask.astype(int), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]      # Region begins
    ends = np.where(diff == -1)[0] - 1    # Region ends (inclusive)

    return list(zip(starts.tolist(), ends.tolist()))
```

**Technical Notes**:
- Uses np.diff to find transitions (0→1 for starts, 1→0 for ends)
- Padding ensures edge regions are detected
- Single pass through data

#### Optional File Export Selection
**Problem**: Exporting all files (NPZ, CSV, PDF) always took full time, even when user only needed quick NPZ backup

**Solution**: Checkboxes in SaveMetaDialog for optional exports

**Features**:
- **NPZ Bundle**: Always saved (grayed out checkbox) - required for data integrity
- **Timeseries CSV**: Time-aligned metric traces (optional, ~9s)
- **Breaths CSV**: Per-breath metrics by region (optional, ~1-2s)
- **Events CSV**: Apnea/eupnea/sniffing intervals (optional, ~0.5s)
- **Summary PDF**: Visualization plots (optional, ~31s)
- **Default**: All checked for comprehensive export
- **Quick export**: Uncheck PDF → 17s total
- **Fastest**: NPZ only → 8s total

**UI Design**:
- 3-column grid layout (compact, professional)
- Smaller font (9pt) for reduced vertical space
- Tooltips showing approximate export time for each file
- Spans both columns of parent form layout
- File name preview positioned before export options

### Performance Results

**Before Optimization**:
- Total export time: 104 seconds
- Index mapping loop: 10.3s (9.9%)
- mask_to_intervals: 10s (9.6%)
- PDF generation: 31s (30%)
- CSV writing: 9s (9%)

**After Optimization**:
- Total export time: 53 seconds (49% faster)
- Index mapping: 0.14ms (73,000× speedup)
- mask_to_intervals: 0.1s (100× speedup)
- Computational bottlenecks eliminated
- Remaining time mostly I/O operations

**With Optional Exports**:
- NPZ only: ~8 seconds
- NPZ + CSVs (no PDF): ~17 seconds
- All files: ~53 seconds

### Implementation Details

#### Files Created
1. **run_profiling.bat** - Automated profiling workflow
   - Creates profiling_results/ directory
   - Runs kernprof with PLETHAPP_TESTING=1
   - Generates timestamped report
   - Displays results in terminal

2. **PROFILING_GUIDE.md** - Complete profiling documentation
   - Quick start instructions
   - @profile decorator explanation
   - Output interpretation guide
   - Troubleshooting section

#### Files Modified

**export/export_manager.py**:
- Added @profile decorator with no-op fallback (lines 20-27)
- Vectorized index mapping (lines 957-970)
- Vectorized mask_to_intervals (lines 1624-1639)
- Added export flags and conditional saves (lines 627-641)
- Updated success message to list only saved files

**dialogs/save_meta_dialog.py**:
- Added import for QGridLayout and QWidget (line 12)
- Added export checkboxes section (lines 115-177)
- Updated values() method to return checkbox states (lines 247-251)
- Moved filename preview before export section

**.gitignore**:
- Added profiling_results/ (line 46)
- Added *.lprof (line 47)

### How to Use

#### Running Profiler
```bash
# Automated workflow (recommended)
run_profiling.bat

# Manual workflow
set PLETHAPP_TESTING=1
kernprof -l run_debug.py
python -m line_profiler run_debug.py.lprof
```

#### Viewing Results
- Profiling results automatically saved to `profiling_results/profile_YYYYMMDD_HHMMSS.txt`
- Read historical results with Read tool
- Compare before/after optimizations

#### Optional Exports
1. Click "Save analyzed data" button
2. Fill in metadata fields
3. **Select which files to export**:
   - Uncheck PDF for quick analysis iterations
   - Keep CSVs for data sharing
   - NPZ always saved for state restoration
4. Click OK

### Lessons Learned

#### Memory Efficiency
**Problem**: First vectorization attempt created 19GB distance matrix
```python
# DON'T DO THIS - creates (600,000 × 4,000) array:
ds_to_orig_idx = np.argmin(np.abs(st.t[:, None] - t_targets), axis=0)
```

**Solution**: Use binary search instead of broadcasting
- Same functionality
- ~32KB memory instead of 19GB
- Actually faster due to O(log n) complexity

#### Profiling Best Practices
1. **Clear cache before profiling**: Python bytecode cache can show old code executing
2. **Use timestamped results**: Easier to track optimization progress
3. **Profile real workflows**: Load actual files, run actual exports
4. **Focus on high % Time**: Lines with >5% time are prime optimization targets

### How to Revert

If you need to remove profiling or revert optimizations:

#### Remove Profiling
1. Delete `run_profiling.bat`
2. Remove @profile decorator from export_manager.py (lines 20-27)
3. Remove profiling_results/ from .gitignore
4. Delete PROFILING_GUIDE.md

#### Revert Vectorization
```python
# In export_manager.py, replace vectorized code with original loops
# (Not recommended - reverts 49% speedup!)

# Revert index mapping (lines 957-970):
for i, t_target in enumerate(t_targets):
    ds_to_orig_idx[i] = np.argmin(np.abs(st.t - t_target))

# Revert mask_to_intervals (lines 1624-1639):
intervals = []
in_region = False
start_idx = None
for i, val in enumerate(mask):
    if val and not in_region:
        start_idx = i
        in_region = True
    elif not val and in_region:
        intervals.append((start_idx, i-1))
        in_region = False
```

#### Remove Optional Exports
1. Delete checkbox section from save_meta_dialog.py (lines 115-177)
2. Remove conditional saves from export_manager.py
3. Revert to always exporting all file types

---

## Related Documentation
- **ALGORITHMS.md**: Core algorithm details
- **FEATURE_BACKLOG.md**: Planned future features
- **PUBLICATION_ROADMAP.md**: v1.0 publication timeline
- **SESSION_SUMMARY.md**: Detailed session notes for recent work
- **PROFILING_GUIDE.md**: Line profiler usage guide
- **PERFORMANCE_OPTIMIZATION_PLAN.md**: Performance improvement roadmap
