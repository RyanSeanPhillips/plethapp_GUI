# PlethApp Implementation Plan - October 7, 2025

## Executive Summary

This plan addresses 5 major tasks for PlethApp development:
1. **Sniffing Data Export & Consolidation** - Integrate sniffing bout annotations into save/consolidation workflows
2. **NPZ File Reopening** - Enable loading saved .npz files to resume analysis sessions
3. **ML Training Data Format** - Structure data exports for machine learning model training
4. **Code Cleanup** - Remove ~4,000+ lines of commented-out code
5. **Code Restructuring** - Improve modularity and organization of 13,000+ line main.py

**Total Estimated Time**: 25-35 hours across all tasks

---

## Current State Analysis

### Codebase Statistics
- **main.py**: 13,213 lines (primary application logic)
- **core/ modules**: ~5,904 lines total (20 Python files)
- **Commented code**: ~4,158 lines in main.py alone
- **Key data structures**:
  - `state.sniff_regions_by_sweep`: `dict[int, list[tuple[float, float]]]` - manual sniffing annotations
  - `state.peaks_by_sweep`: `dict[int, np.ndarray]` - detected inspiratory peaks
  - `state.breath_by_sweep`: `dict[int, dict]` - breath events (onsets, offsets, expmins, expoffs)
  - `state.sigh_by_sweep`: `dict[int, np.ndarray]` - sigh peak indices

### Current Export Format (.npz v2)
**Files saved per analysis**:
1. `<base>_bundle.npz` - Binary data bundle
2. `<base>_timeseries.csv` - Time-series metrics with normalization
3. `<base>_breaths.csv` - Breath-by-breath metrics
4. `<base>_events.csv` - Event intervals (stimulus, apnea, eupnea)
5. `<base>_summary.pdf` - Visual summary

**NPZ v2 Structure** (lines 5757-6006):
```python
{
    'npz_version': 2,
    't_ds': np.ndarray,                    # Downsampled time (raw)
    'timeseries_t': np.ndarray,            # Time for CSV (relative to stim)
    'Y_proc_ds': np.ndarray,               # Processed signal (M, S)
    'peaks_by_sweep': np.ndarray[object],  # Peak indices per sweep
    'onsets_by_sweep': np.ndarray[object], # Breath onsets
    'offsets_by_sweep': np.ndarray[object],
    'expmins_by_sweep': np.ndarray[object],
    'expoffs_by_sweep': np.ndarray[object],
    'sigh_idx_by_sweep': np.ndarray[object],
    'stim_spans_by_sweep': np.ndarray[object],
    'meta_json': str,                      # JSON-encoded metadata
    'timeseries_keys': list[str],          # Metric names
    'ts_raw_{metric}': np.ndarray,         # Raw metric timeseries
    'ts_norm_{metric}': np.ndarray,        # Normalized (per-sweep baseline)
    'ts_eupnea_{metric}': np.ndarray,      # Normalized (eupnea baseline)
}
```

**Missing from NPZ**: Sniffing regions, omitted_sweeps, filter parameters, manual edits

### Current Consolidation Workflow
- Users select multiple CSV files from Curation tab
- App searches for matching `_bundle.npz` files
- Fast path: Loads NPZ v2 data directly (no CSV parsing)
- Fallback: Parses CSV files if NPZ missing/old version
- Consolidates across files: mean, SEM, time-aligned averaging

---

## Task 1: Sniffing Data Export & Consolidation

### Priority: HIGH
**Estimated Time**: 5-6 hours

### 1.1 Add Sniffing to NPZ Bundle

**Location**: `main.py` lines 5721-6006 (`_export_all_analyzed_data`)

**Changes**:
```python
# Add after line 5765 (sigh_obj creation):
# Pack sniffing regions (aligned with kept_sweeps)
sniff_obj = np.empty(S, dtype=object)
for col, s in enumerate(kept_sweeps):
    regions = st.sniff_regions_by_sweep.get(s, [])
    # Convert to numpy array of shape (N_regions, 2)
    sniff_obj[col] = np.array(regions, dtype=float).reshape(-1, 2) if regions else np.empty((0, 2), dtype=float)

# Add to NPZ save dict (line 5769):
_npz_timeseries_data['sniff_regions_by_sweep'] = sniff_obj
```

**Backward Compatibility**: Safe - only adds new key, old code ignores it

### 1.2 Add Sniffing to Events CSV

**Location**: `main.py` lines 6398-6538 (events CSV writing)

**Changes**:
```python
# Add after eupnea intervals (line 6527), before writing CSV:

# Add sniffing bout intervals
sniff_regions = st.sniff_regions_by_sweep.get(s, [])
for start_time, end_time in sniff_regions:
    # start_time and end_time are already in seconds

    # Convert to relative time if global stim available
    if have_global_stim:
        start_time_rel = start_time - global_s0
        end_time_rel = end_time - global_s0
    else:
        start_time_rel = start_time
        end_time_rel = end_time

    duration = end_time - start_time

    events_rows.append([
        str(s + 1),
        "sniffing",  # New event type
        f"{start_time_rel:.9g}",
        f"{end_time_rel:.9g}",
        f"{duration:.9g}"
    ])
```

**Test Cases**:
- File with sniffing annotations → events CSV includes "sniffing" rows
- File without sniffing → no "sniffing" rows (graceful)
- Multiple sniffing bouts per sweep → all exported correctly
- Time alignment with stimulus → relative times match other events

### 1.3 Add Sniffing Consolidation

**Location**: `main.py` lines 8665-8836 (`on_consolidate_save_data_clicked`)

**Implementation Strategy**: Create separate sheet "Sniffing Bouts" in Excel output

**Changes**:
```python
# Add after 'events' consolidation (around line 8815):

# Consolidate sniffing bout data
sniffing_data = []
for (root_name, root_path), group in file_groups.items():
    events_path = group.get('events')
    if events_path:
        df = pd.read_csv(events_path)
        # Filter for sniffing events only
        sniff_df = df[df['event_type'] == 'sniffing'].copy()
        if not sniff_df.empty:
            sniff_df['file'] = root_name
            sniffing_data.append(sniff_df)

if sniffing_data:
    consolidated_data['sniffing'] = {
        'data': pd.concat(sniffing_data, ignore_index=True),
        'description': 'Sniffing bout events across all files'
    }
else:
    # No sniffing data found - add warning
    if '_warnings' not in consolidated_data:
        consolidated_data['_warnings'] = []
    consolidated_data['_warnings'].append("No sniffing bout data found in selected files.")
```

**Output Format**: Excel sheet with columns:
- `file` - Source filename
- `sweep` - Sweep number
- `event_type` - Always "sniffing"
- `start_time` - Relative to stimulus (seconds)
- `end_time` - Relative to stimulus
- `duration` - Bout duration (seconds)

**Alternative Enhancement** (optional, +2 hours):
Add time-binned sniffing metrics similar to breathing frequency:
- Sniffing bout frequency (bouts/second in time bins)
- Mean bout duration per time bin
- Fraction of time spent sniffing per bin

### 1.4 Testing Strategy

**Phase 1 - Export** (30 min):
1. Load ABF with manual sniffing annotations
2. Run "Save Analyzed Data"
3. Verify `_events.csv` contains "sniffing" rows
4. Verify `_bundle.npz` contains `sniff_regions_by_sweep` key
5. Check time alignment matches other events

**Phase 2 - Consolidation** (30 min):
1. Create 2-3 files with sniffing annotations
2. Run consolidation workflow
3. Verify Excel output has "Sniffing Bouts" sheet
4. Check data integrity (times, durations, sweep numbers)

**Phase 3 - Backward Compatibility** (30 min):
1. Load old CSV files (pre-sniffing feature)
2. Verify consolidation doesn't crash
3. Check graceful warnings for missing sniffing data
4. Test with mixed new/old files

**Files to Modify**:
- `main.py` (3 locations: NPZ save, events CSV, consolidation)

**Files to Create**:
- None (pure modification of existing code)

---

## Task 2: NPZ File Reopening

### Priority: HIGH
**Estimated Time**: 8-10 hours

### 2.1 Feature Requirements

**Use Case**: User saves analysis, closes app, later reopens .npz to review/modify/re-export

**Must Restore**:
- Raw data (original ABF traces) OR processed downsampled traces
- Peaks, breath events, sigh annotations
- Sniffing region markers
- Filter settings (low/high Hz, invert, mean_sub)
- Omitted sweeps
- Stim detection results
- All manual edits

**UI/UX**:
- Add "Open Analysis (.npz)" button next to "Browse" button
- OR: Detect .npz file in "Browse" dialog and auto-load
- Status bar: "Loaded from saved analysis: [filename]"
- Enable all editing modes after load (add/delete peaks, mark sniffing, etc.)

### 2.2 Data Structure Enhancement

**Problem**: Current NPZ v2 doesn't store enough for full restoration

**Solution**: Add to NPZ save (line ~5753):

```python
meta = {
    # ... existing keys ...

    # Filter settings
    "use_low": bool(st.use_low),
    "use_high": bool(st.use_high),
    "use_mean_sub": bool(st.use_mean_sub),
    "use_invert": bool(st.use_invert),
    "low_hz": float(st.low_hz) if st.low_hz else None,
    "high_hz": float(st.high_hz) if st.high_hz else None,
    "mean_val": float(st.mean_val),
    "filter_order": int(self.filter_order),

    # Notch filter (if active)
    "notch_filter_lower": float(self.notch_filter_lower) if self.notch_filter_lower else None,
    "notch_filter_upper": float(self.notch_filter_upper) if self.notch_filter_upper else None,

    # Navigation state
    "window_dur_s": float(st.window_dur_s),

    # Channel info (critical for restoration)
    "channel_names": list(st.channel_names),
    "stim_chan": str(st.stim_chan) if st.stim_chan else None,
}

# Add separate top-level keys (can't JSON-serialize numpy arrays):
_npz_timeseries_data['omitted_sweeps'] = np.array(sorted(st.omitted_sweeps), dtype=int)
_npz_timeseries_data['sniff_regions_by_sweep'] = sniff_obj  # From Task 1
_npz_timeseries_data['omitted_points_by_sweep'] = ... # If needed for point edits
_npz_timeseries_data['omitted_ranges_by_sweep'] = ... # If needed for range edits
```

### 2.3 Load Implementation

**New Method**: `load_npz_analysis(self, npz_path: Path)`

**Location**: Add to `MainWindow` class after `on_browse_clicked` (~line 550)

**Implementation**:
```python
def on_open_analysis_clicked(self):
    """Open a saved .npz analysis bundle to resume work."""
    start_dir = self.settings.value("last_browse_dir", str(Path.home()))

    path, _ = QFileDialog.getOpenFileName(
        self, "Open Saved Analysis",
        start_dir,
        "PlethApp Analysis (*.npz);;All Files (*)"
    )
    if not path:
        return

    npz_path = Path(path)
    self.settings.setValue("last_browse_dir", str(npz_path.parent))

    try:
        self._load_npz_analysis(npz_path)
    except Exception as e:
        QMessageBox.critical(self, "Load Failed", f"Could not load analysis:\n{e}")
        import traceback
        traceback.print_exc()

def _load_npz_analysis(self, npz_path: Path):
    """Load complete analysis state from NPZ bundle."""
    # Load NPZ
    data = np.load(npz_path, allow_pickle=True)

    # Check version
    version = data.get('npz_version', None)
    if version is None or int(version) < 2:
        QMessageBox.warning(self, "Old Format",
            "This is an older NPZ format without full analysis state.\n"
            "Only basic data will be restored.")

    # Parse metadata
    meta = json.loads(str(data['meta_json']))

    st = self.state

    # Restore file info
    st.in_path = Path(meta['abf_path']) if meta.get('abf_path') else None
    st.sr_hz = float(meta['sr_hz'])
    st.channel_names = list(meta['channel_names'])
    st.analyze_chan = str(meta['analyze_channel'])
    st.stim_chan = meta.get('stim_chan', None)

    # Restore time array
    st.t = data['t_ds']  # Downsampled time (raw)
    N = len(st.t)

    # Restore processed signal
    Y_proc_ds = data['Y_proc_ds']  # Shape (M, S)
    M, S = Y_proc_ds.shape

    # Reconstruct sweeps dictionary
    # NOTE: We only have downsampled processed data, not raw ABF data
    # This is a limitation - can't go back to raw signal
    st.sweeps = {st.analyze_chan: Y_proc_ds}

    # Restore kept/omitted sweeps
    kept_sweeps = [int(x) for x in meta['kept_sweeps']]
    omitted_sweeps = set(int(x) for x in meta.get('omitted_sweeps', []))
    st.omitted_sweeps = omitted_sweeps

    # Restore peaks and breath events
    st.peaks_by_sweep = {}
    st.breath_by_sweep = {}

    peaks_obj = data['peaks_by_sweep']
    onsets_obj = data['onsets_by_sweep']
    offsets_obj = data['offsets_by_sweep']
    expmins_obj = data['expmins_by_sweep']
    expoffs_obj = data['expoffs_by_sweep']

    for col, s in enumerate(kept_sweeps):
        st.peaks_by_sweep[s] = np.asarray(peaks_obj[col], dtype=int)
        st.breath_by_sweep[s] = {
            'onsets': np.asarray(onsets_obj[col], dtype=int),
            'offsets': np.asarray(offsets_obj[col], dtype=int),
            'expmins': np.asarray(expmins_obj[col], dtype=int),
            'expoffs': np.asarray(expoffs_obj[col], dtype=int),
        }

    # Restore sighs
    st.sigh_by_sweep = {}
    sigh_obj = data.get('sigh_idx_by_sweep', None)
    if sigh_obj is not None:
        for col, s in enumerate(kept_sweeps):
            st.sigh_by_sweep[s] = np.asarray(sigh_obj[col], dtype=int)

    # Restore sniffing regions
    st.sniff_regions_by_sweep = {}
    sniff_obj = data.get('sniff_regions_by_sweep', None)
    if sniff_obj is not None:
        for col, s in enumerate(kept_sweeps):
            regions_arr = sniff_obj[col]
            if regions_arr.size > 0:
                st.sniff_regions_by_sweep[s] = [(float(r[0]), float(r[1])) for r in regions_arr]

    # Restore stimulus spans
    st.stim_spans_by_sweep = {}
    stim_obj = data.get('stim_spans_by_sweep', None)
    if stim_obj is not None:
        for col, s in enumerate(kept_sweeps):
            spans_arr = stim_obj[col]
            if spans_arr.size > 0:
                st.stim_spans_by_sweep[s] = [(float(r[0]), float(r[1])) for r in spans_arr]

    # Restore filter settings
    st.use_low = bool(meta.get('use_low', False))
    st.use_high = bool(meta.get('use_high', False))
    st.use_mean_sub = bool(meta.get('use_mean_sub', False))
    st.use_invert = bool(meta.get('use_invert', False))
    st.low_hz = meta.get('low_hz', None)
    st.high_hz = meta.get('high_hz', None)
    st.mean_val = meta.get('mean_val', 0.0)

    self.filter_order = int(meta.get('filter_order', 4))
    self.notch_filter_lower = meta.get('notch_filter_lower', None)
    self.notch_filter_upper = meta.get('notch_filter_upper', None)

    # Restore navigation
    st.window_dur_s = float(meta.get('window_dur_s', 10.0))
    st.sweep_idx = 0
    st.window_start_s = 0.0

    # Update UI controls
    self._populate_channel_dropdowns()
    self.AnalyzeChanSelect.setCurrentText(st.analyze_chan)
    if st.stim_chan:
        self.StimChanSelect.setCurrentText(st.stim_chan)

    # Update filter controls
    self.LowPassCheck.setChecked(st.use_low)
    self.HighPassCheck.setChecked(st.use_high)
    self.MeanSubCheck.setChecked(st.use_mean_sub)
    self.InvertCheck.setChecked(st.use_invert)
    if st.low_hz:
        self.LowPassHz.setText(str(st.low_hz))
    if st.high_hz:
        self.HighPassHz.setText(str(st.high_hz))
    self.FilterOrderSpin.setValue(self.filter_order)

    # Update status
    self.statusbar.showMessage(f"Loaded analysis: {npz_path.name}", 5000)
    self.setWindowTitle(f"PlethAnalysis v{VERSION_STRING} - {npz_path.stem}")

    # Redraw
    self.redraw_main_plot()

    print(f"[NPZ] ✓ Analysis restored from {npz_path.name}")
    print(f"  - {S} sweeps ({len(omitted_sweeps)} omitted)")
    print(f"  - {sum(len(st.peaks_by_sweep.get(s, [])) for s in kept_sweeps)} total peaks")
    print(f"  - {sum(len(st.sniff_regions_by_sweep.get(s, [])) for s in kept_sweeps)} sniffing regions")
```

### 2.4 Critical Limitation: Raw Data Not Saved

**Problem**: Current NPZ saves downsampled *processed* data, not raw ABF traces

**Implications**:
- Can't change filter settings after reopening (data already filtered)
- Can't access original sample rate (downsampled to 50 Hz)
- Can't reload from original ABF if file moved

**Solutions** (choose one):

**Option A - Hybrid: Save ABF path, reload raw on open** (RECOMMENDED)
```python
# On load:
if st.in_path and st.in_path.exists():
    # Reload raw ABF data
    sr_hz, raw_sweeps, channels, t_raw = abf_io.load_abf(st.in_path)
    st.sweeps = raw_sweeps
    st.t = t_raw
    st.sr_hz = sr_hz
    # Now user can change filters and re-detect peaks
else:
    # Fallback to downsampled data
    st.sweeps = {st.analyze_chan: Y_proc_ds}
    st.t = data['t_ds']
    QMessageBox.warning(self, "Raw Data Unavailable",
        f"Could not find original file:\n{st.in_path}\n\n"
        "Loaded downsampled data only. Filter changes disabled.")
```
**Pros**: Small NPZ files, full flexibility if ABF available
**Cons**: Breaks if ABF moved/renamed

**Option B - Save raw data in NPZ** (+large file size)
```python
# On save: Add raw sweeps to NPZ
_npz_timeseries_data['raw_sweeps'] = {ch: st.sweeps[ch] for ch in st.channel_names}
_npz_timeseries_data['raw_t'] = st.t
```
**Pros**: Self-contained, works even if ABF deleted
**Cons**: NPZ files become 10-50× larger (hundreds of MB)

**Option C - Save compressed raw data** (best of both worlds, +3 hours implementation)
```python
# Save raw data with aggressive compression
np.savez_compressed(npz_path,
    raw_sweeps_compressed=compress_sweeps(st.sweeps),  # Custom compression
    **_npz_timeseries_data
)
```
**Pros**: Smaller than Option B, self-contained
**Cons**: Requires custom compression logic

**Recommendation**: Start with **Option A** (hybrid). Add UI indicator when raw data unavailable.

### 2.5 Re-save / Update Workflow

**Feature**: When user opens NPZ, edits data, and clicks "Save Analyzed Data" again

**Expected Behavior**:
1. Auto-populate filename with same base name
2. Detect existing CSV files: `_timeseries.csv`, `_breaths.csv`, `_events.csv`
3. Prompt: "Files exist. Overwrite?"
4. If yes: Update all CSVs + NPZ with new data

**Implementation**:
```python
# In SaveMetaDialog initialization:
if hasattr(parent, '_npz_source_path'):
    # We're re-saving an opened NPZ analysis
    npz_path = parent._npz_source_path
    suggested_base = npz_path.stem.replace('_bundle', '')
    # Pre-populate dialog with original metadata
    self.preview_line.setText(suggested_base)
    # ... populate other fields from parent._save_meta

# In _export_all_analyzed_data:
# Before save, check for existing files
if npz_path.exists():
    reply = QMessageBox.question(self, "Overwrite?",
        f"Analysis files already exist:\n{npz_path.parent}\n\n"
        "Overwrite all files?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    if reply == QMessageBox.StandardButton.No:
        return
```

### 2.6 UI Changes

**Add Button**: "Open Analysis" next to "Browse" button
**Button Text**: `OpenAnalysisButton`
**Wire in `__init__`**:
```python
self.OpenAnalysisButton.clicked.connect(self.on_open_analysis_clicked)
```

**Add to UI file**: `ui/pleth_app_layout_02_horizontal.ui`
- Insert button in `horizontalLayout_7` (Browse/File Selection row)
- Place after `BrowseButton`

### 2.7 Testing Strategy

**Test Case 1 - Basic Roundtrip**:
1. Load ABF, detect peaks, mark sniffing, save analysis
2. Close app
3. Reopen app, click "Open Analysis", select NPZ
4. Verify: peaks displayed, sniffing regions shown, sweep count correct
5. Edit: add more peaks, delete a sniffing region
6. Re-save → verify files overwritten

**Test Case 2 - Missing ABF**:
1. Save analysis, move/rename original ABF
2. Open NPZ
3. Verify: warning shown, downsampled data loaded
4. Try to change filters → verify disabled or warning shown

**Test Case 3 - Old Format**:
1. Open NPZ v1 file (if any exist)
2. Verify: warning shown about limited restoration

**Files to Modify**:
- `main.py` (new methods, NPZ save enhancement)
- `ui/pleth_app_layout_02_horizontal.ui` (new button)
- `core/state.py` (verify all state fields documented)

---

## Task 3: ML Training Data Format

### Priority: MEDIUM
**Estimated Time**: 6-8 hours

### 3.1 Requirements Analysis

**ML Model Goals**:
1. **Breath Event Detection**: Predict onset/offset/peak indices from raw signal
2. **Breath Classification**: Identify eupnea, sniffing, sighs, apneas from features

**Training Data Needs**:
- **Raw signal segments** (input features)
- **Ground truth labels** (breath events, classifications)
- **Temporal context** (time windows around each breath)
- **Metadata** (sampling rate, filtering applied, species/condition)

### 3.2 Current NPZ Format Suitability

**Pros**:
- ✅ Already has raw processed signal (`Y_proc_ds`)
- ✅ Has breath events (onsets, offsets, peaks)
- ✅ Has classifications (sighs, sniffing regions, eupnea masks)
- ✅ Structured format (numpy arrays)
- ✅ Metadata included

**Cons**:
- ❌ Downsampled to 50 Hz (may lose important features)
- ❌ Only processed signal, not raw (filter artifacts)
- ❌ No per-breath feature vectors (would need to recompute)
- ❌ Time-series focused, not breath-focused

**Recommendation**: Current NPZ is **NOT optimal** for ML training, but can work as intermediate format

### 3.3 Proposed ML Data Export Format

**Approach**: Add separate "Export for ML" button that creates ML-optimized format

**Output Structure**: HDF5 file (better than NPZ for large datasets + metadata)

```python
ml_export.h5:
  /metadata
    - sampling_rate: float
    - n_files: int
    - filter_settings: dict
    - species: str (from UI metadata)
    - experimental_condition: str

  /signals
    - raw_segments: (N_breaths, segment_length) float32
    - processed_segments: (N_breaths, segment_length) float32
    - time_vector: (segment_length,) float32  # Relative time centered on peak

  /labels
    - breath_type: (N_breaths,) str  # 'normal', 'sigh', 'sniff', 'apnea'
    - peak_idx: (N_breaths,) int32  # Peak index within segment
    - onset_idx: (N_breaths,) int32
    - offset_idx: (N_breaths,) int32
    - expmin_idx: (N_breaths,) int32
    - expoff_idx: (N_breaths,) int32

  /features
    - instantaneous_freq: (N_breaths,) float32
    - amplitude_insp: (N_breaths,) float32
    - amplitude_exp: (N_breaths,) float32
    - ti: (N_breaths,) float32
    - te: (N_breaths,) float32
    - area_insp: (N_breaths,) float32
    - area_exp: (N_breaths,) float32

  /context
    - file_id: (N_breaths,) int32  # Which source file
    - sweep_id: (N_breaths,) int32
    - breath_id: (N_breaths,) int32  # Breath number within sweep
    - is_manual_edit: (N_breaths,) bool  # User manually adjusted?
    - quality_score: (N_breaths,) float32  # Optional outlier score
```

### 3.4 Implementation Strategy

**Option 1 - Export from PlethApp** (+6 hours)

Add new export mode to PlethApp:

```python
def on_export_ml_data_clicked(self):
    """Export analysis data in ML-ready format (HDF5)."""
    # Prompt for output location
    out_path, _ = QFileDialog.getSaveFileName(
        self, "Export ML Training Data",
        str(Path.cwd() / "ml_training_data.h5"),
        "HDF5 Files (*.h5);;All Files (*)"
    )
    if not out_path:
        return

    self._export_ml_format(Path(out_path))

def _export_ml_format(self, out_path: Path):
    """Export current analysis to ML training format."""
    import h5py

    st = self.state
    kept_sweeps = [s for s in range(self._sweep_count()) if s not in st.omitted_sweeps]

    # Collect breath segments
    all_segments_raw = []
    all_segments_proc = []
    all_labels = []
    all_features = []
    all_context = []

    segment_length = int(3.0 * st.sr_hz)  # 3 second windows
    half_window = segment_length // 2

    file_id = 0  # Single file for now

    for s in kept_sweeps:
        y_raw = st.sweeps[st.analyze_chan][:, s]  # Raw signal
        y_proc = self._get_processed_for(st.analyze_chan, s)  # Processed

        pks = st.peaks_by_sweep.get(s, np.array([], dtype=int))
        br = st.breath_by_sweep.get(s, {})

        onsets = br.get('onsets', np.array([], dtype=int))
        offsets = br.get('offsets', np.array([], dtype=int))
        expmins = br.get('expmins', np.array([], dtype=int))
        expoffs = br.get('expoffs', np.array([], dtype=int))

        sighs = set(st.sigh_by_sweep.get(s, []))
        sniff_regions = st.sniff_regions_by_sweep.get(s, [])

        # For each peak, extract segment and labels
        for i, pk in enumerate(pks):
            # Extract segment centered on peak
            start_idx = max(0, pk - half_window)
            end_idx = min(len(y_proc), pk + half_window)

            # Pad if needed
            seg_raw = np.zeros(segment_length, dtype=np.float32)
            seg_proc = np.zeros(segment_length, dtype=np.float32)

            actual_len = end_idx - start_idx
            seg_raw[:actual_len] = y_raw[start_idx:end_idx]
            seg_proc[:actual_len] = y_proc[start_idx:end_idx]

            # Determine breath type
            breath_type = 'normal'
            if pk in sighs:
                breath_type = 'sigh'
            else:
                # Check if peak in sniffing region
                t_pk = st.t[pk]
                for sn_start, sn_end in sniff_regions:
                    if sn_start <= t_pk <= sn_end:
                        breath_type = 'sniff'
                        break

            # Find corresponding breath events (relative to segment start)
            onset_rel = onsets[i] - start_idx if i < len(onsets) else -1
            offset_rel = offsets[i] - start_idx if i < len(offsets) else -1
            expmin_rel = expmins[i] - start_idx if i < len(expmins) else -1
            expoff_rel = expoffs[i] - start_idx if i < len(expoffs) else -1
            peak_rel = pk - start_idx

            # Compute features for this breath
            features = self._compute_breath_features(y_proc, st.t, pk, br, i)

            all_segments_raw.append(seg_raw)
            all_segments_proc.append(seg_proc)
            all_labels.append({
                'breath_type': breath_type,
                'peak_idx': peak_rel,
                'onset_idx': onset_rel,
                'offset_idx': offset_rel,
                'expmin_idx': expmin_rel,
                'expoff_idx': expoff_rel,
            })
            all_features.append(features)
            all_context.append({
                'file_id': file_id,
                'sweep_id': s,
                'breath_id': i,
            })

    # Write to HDF5
    with h5py.File(out_path, 'w') as f:
        # Metadata
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['sampling_rate'] = st.sr_hz
        meta_grp.attrs['n_files'] = 1
        meta_grp.attrs['segment_length'] = segment_length

        # Signals
        sig_grp = f.create_group('signals')
        sig_grp.create_dataset('raw_segments', data=np.array(all_segments_raw))
        sig_grp.create_dataset('processed_segments', data=np.array(all_segments_proc))
        sig_grp.create_dataset('time_vector', data=np.linspace(-1.5, 1.5, segment_length))

        # Labels
        lbl_grp = f.create_group('labels')
        breath_types = [lb['breath_type'] for lb in all_labels]
        lbl_grp.create_dataset('breath_type', data=np.array(breath_types, dtype='S10'))
        lbl_grp.create_dataset('peak_idx', data=[lb['peak_idx'] for lb in all_labels])
        lbl_grp.create_dataset('onset_idx', data=[lb['onset_idx'] for lb in all_labels])
        lbl_grp.create_dataset('offset_idx', data=[lb['offset_idx'] for lb in all_labels])
        # ... (similar for other indices)

        # Features
        feat_grp = f.create_group('features')
        if all_features:
            for key in all_features[0].keys():
                feat_grp.create_dataset(key, data=[f[key] for f in all_features])

        # Context
        ctx_grp = f.create_group('context')
        ctx_grp.create_dataset('file_id', data=[c['file_id'] for c in all_context])
        ctx_grp.create_dataset('sweep_id', data=[c['sweep_id'] for c in all_context])
        ctx_grp.create_dataset('breath_id', data=[c['breath_id'] for c in all_context])

    print(f"[ML] ✓ Exported {len(all_segments_raw)} breath segments to {out_path.name}")
    QMessageBox.information(self, "ML Export Complete",
        f"Exported {len(all_segments_raw)} breath segments\n{out_path.name}")
```

**Option 2 - Separate Conversion Script** (+2 hours, RECOMMENDED)

Create standalone script that reads NPZ files and generates ML format:

```python
# scripts/npz_to_ml_format.py

import numpy as np
import h5py
import json
from pathlib import Path

def convert_npz_to_ml(npz_paths: list[Path], output_h5: Path):
    """
    Convert multiple PlethApp NPZ bundles into unified ML training dataset.

    Args:
        npz_paths: List of paths to _bundle.npz files
        output_h5: Output path for HDF5 ML dataset
    """
    # ... similar logic to Option 1, but batch processes multiple files
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert PlethApp NPZ to ML format')
    parser.add_argument('input_dir', type=Path, help='Directory containing _bundle.npz files')
    parser.add_argument('output', type=Path, help='Output .h5 file')
    args = parser.parse_args()

    npz_files = list(args.input_dir.rglob('*_bundle.npz'))
    print(f"Found {len(npz_files)} NPZ files")
    convert_npz_to_ml(npz_files, args.output)
```

**Advantages of Option 2**:
- Keeps PlethApp focused on analysis, not ML pipeline
- Easier to iterate on ML format without rebuilding app
- Can run as batch job on server
- Separates concerns (GUI vs ML preprocessing)

### 3.5 Alternative: Just Use Existing NPZ + CSV

**Simplest Approach** (0 hours):

Current NPZ format is actually quite good for ML:
- Load NPZ in training script
- Extract `Y_proc_ds` as features
- Extract `peaks_by_sweep` as labels
- Use `sniff_regions_by_sweep`, `sigh_idx_by_sweep` for classification

**Example training script**:
```python
# ml/train_breath_detector.py

import numpy as np
from pathlib import Path

# Load all NPZ files
data_dir = Path('Pleth_App_analysis')
npz_files = list(data_dir.rglob('*_bundle.npz'))

X_train = []  # Signal windows
y_train = []  # Peak locations

for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=True)

    Y_proc = data['Y_proc_ds']  # Shape (M, S)
    peaks_by_sweep = data['peaks_by_sweep']

    for sweep_idx, peaks in enumerate(peaks_by_sweep):
        signal = Y_proc[:, sweep_idx]
        # Create training examples
        # ... sliding window + peak labels
```

**Pros**: Zero implementation time, works with existing exports
**Cons**: Less structured, requires custom preprocessing in ML code

**Recommendation**: Start with this approach, add dedicated ML export later if needed

### 3.6 Documentation for ML Users

Create `docs/ML_TRAINING_GUIDE.md`:

```markdown
# Machine Learning Training Data Guide

## Using PlethApp NPZ Files for ML Training

PlethApp's `.npz` bundle files contain all data needed for training breath detection models.

### NPZ Structure
...

### Example: Loading NPZ in Python
...

### Example: Training a Simple Detector
...

### Suggested Model Architectures
- 1D CNN for peak detection
- LSTM for breath sequence modeling
- Transformer for long-range dependencies

### Augmentation Strategies
- Random time shifts
- Amplitude scaling
- Baseline drift simulation
```

---

## Task 4: Code Cleanup - Remove Commented Code

### Priority: MEDIUM
**Estimated Time**: 4-5 hours

### 4.1 Scope Assessment

**Current State**:
- ~4,158 commented lines in `main.py` (31% of file)
- Many are old function versions with new version pasted above/below
- Some are experimental features tried and abandoned
- Some are notes/TODOs

**Examples Found**:
```python
Line 4092-4560: Commented _export_all_analyzed_data (old version)
Line 4566-5000: Another commented _export_all_analyzed_data
Line 5007-5455: Yet another commented version
Line 8320-8663: Multiple commented on_consolidate_save_data_clicked versions
Line 11307-12432: Many commented _save_consolidated_to_excel versions
```

### 4.2 Cleanup Strategy

**Phase 1 - Automated Removal** (2 hours)

Create script to identify and optionally remove commented code:

```python
# scripts/clean_commented_code.py

import re
from pathlib import Path

def find_commented_functions(file_path: Path):
    """Find large blocks of commented-out functions."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    in_comment_block = False
    block_start = None
    block_lines = []

    for i, line in enumerate(lines, 1):
        # Match lines starting with "    # def" or "    #     " (indented comments)
        if re.match(r'^\s+# (def |class |@|    )', line):
            if not in_comment_block:
                block_start = i
                in_comment_block = True
            block_lines.append(i)
        elif in_comment_block:
            # Check if we left comment block
            if not line.strip().startswith('#') and line.strip():
                # End of block
                blocks.append((block_start, block_lines[-1], len(block_lines)))
                in_comment_block = False
                block_lines = []

    return blocks

def create_backup(file_path: Path):
    """Create timestamped backup."""
    import shutil
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = file_path.with_suffix(f'.backup_{timestamp}.py')
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def remove_comment_blocks(file_path: Path, blocks: list, dry_run=True):
    """Remove identified comment blocks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if dry_run:
        print(f"\n[DRY RUN] Would remove {len(blocks)} blocks ({sum(b[2] for b in blocks)} lines):")
        for start, end, count in blocks:
            print(f"  Lines {start}-{end} ({count} lines)")
            # Show first line as preview
            preview = lines[start-1].strip()
            if len(preview) > 80:
                preview = preview[:77] + '...'
            print(f"    {preview}")
        return None

    # Actually remove blocks (in reverse order to preserve line numbers)
    blocks_sorted = sorted(blocks, key=lambda b: b[0], reverse=True)
    lines_removed = 0

    for start, end, count in blocks_sorted:
        # Delete lines[start-1:end]  (inclusive)
        del lines[start-1:end]
        lines_removed += count

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print(f"✓ Removed {lines_removed} lines from {len(blocks)} comment blocks")
    return lines_removed

if __name__ == '__main__':
    main_py = Path(__file__).parent.parent / 'main.py'

    blocks = find_commented_functions(main_py)
    print(f"Found {len(blocks)} large commented code blocks in main.py")

    # Dry run first
    remove_comment_blocks(main_py, blocks, dry_run=True)

    # Prompt for actual removal
    response = input("\nProceed with removal? (yes/no): ")
    if response.lower() == 'yes':
        backup_path = create_backup(main_py)
        removed = remove_comment_blocks(main_py, blocks, dry_run=False)
        print(f"\nBackup saved to: {backup_path}")
        print(f"Removed {removed} lines. Please test the application!")
```

**Phase 2 - Manual Review** (2-3 hours)

Some commented code should be **kept** temporarily:
- Active TODO comments
- Explanatory comments for complex logic
- Disabled features planned for re-implementation

**Review Process**:
1. Run script in dry-run mode
2. Review list of blocks to be removed
3. Manually check any blocks with "TODO", "FIXME", "NOTE", etc.
4. Move important comments to separate notes file
5. Run script to remove approved blocks
6. Test application thoroughly
7. Commit with descriptive message: "Clean up 4000+ lines of commented code"

### 4.3 Preserve Important Information

Before deleting, extract any important notes:

```python
# scripts/extract_important_comments.py

def extract_todos_and_notes(file_path: Path, output_md: Path):
    """Extract TODO, FIXME, NOTE comments to markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    important = []
    for i, line in enumerate(lines, 1):
        if any(keyword in line.upper() for keyword in ['TODO:', 'FIXME:', 'XXX:', 'HACK:', 'NOTE:']):
            important.append(f"Line {i}: {line.strip()}")

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write(f"# Important Comments from {file_path.name}\n\n")
        f.write(f"Extracted: {datetime.now()}\n\n")
        f.write('\n'.join(important))

    print(f"Extracted {len(important)} important comments to {output_md}")
```

### 4.4 Testing After Cleanup

**Critical Test Cases**:
1. Load ABF file → verify loads correctly
2. Detect peaks → verify peak detection works
3. Save analyzed data → verify all exports generate
4. Open Curation tab → verify consolidation works
5. Spectral analysis dialog → verify opens and functions
6. Manual editing (add/delete peaks, mark sniffing) → verify all modes work

**If any test fails**: Restore from backup, identify which block was needed, restore it

---

## Task 5: Code Restructuring & Modularization

### Priority: LOW (but high impact)
**Estimated Time**: 8-12 hours

### 5.1 Current Organization Issues

**main.py Problems** (13,213 lines):
- All dialogs defined as nested classes (1000+ lines)
- Export logic mixed with UI code (1000+ lines)
- Consolidation logic in UI file (3000+ lines)
- Plotting code in UI file (scattered throughout)
- No clear separation of concerns

**core/ Organization**:
- ✅ Good: peaks.py, filters.py, metrics.py, stim.py (focused modules)
- ⚠️ Underutilized: export.py is minimal, could do more
- ❌ Missing: dialogs/, consolidation/, ml_export/

### 5.2 Proposed New Structure

```
plethapp/
├── main.py (shrink to ~3000 lines)
│   - Application bootstrap
│   - Main window class (UI wiring only)
│   - Event handlers (delegate to modules)
│
├── core/
│   ├── state.py             [existing]
│   ├── abf_io.py            [existing]
│   ├── filters.py           [existing]
│   ├── peaks.py             [existing]
│   ├── metrics.py           [existing]
│   ├── stim.py              [existing]
│   ├── navigation.py        [existing]
│   ├── editing.py           [existing]
│   ├── plotting.py          [existing]
│   │
│   ├── export.py            [EXPAND]
│   │   - export_npz_bundle()
│   │   - export_timeseries_csv()
│   │   - export_breaths_csv()
│   │   - export_events_csv()
│   │   - export_summary_pdf()
│   │   - load_npz_bundle()  [NEW]
│   │
│   └── io/                  [existing]
│       ├── son64_loader.py
│       └── ...
│
├── ui/
│   ├── dialogs/             [NEW - extract from main.py]
│   │   ├── __init__.py
│   │   ├── save_meta_dialog.py
│   │   ├── eupnea_params_dialog.py
│   │   ├── outlier_metrics_dialog.py
│   │   ├── spectral_analysis_dialog.py
│   │   └── summary_preview_dialog.py
│   │
│   └── pleth_app_layout_02_horizontal.ui [existing]
│
├── curation/                [NEW - extract from main.py]
│   ├── __init__.py
│   ├── file_scanner.py
│   │   - scan_csv_groups()
│   │   - populate_file_list()
│   │
│   ├── consolidation.py
│   │   - consolidate_means_files()
│   │   - consolidate_breaths_histograms()
│   │   - consolidate_events()
│   │   - consolidate_from_npz()
│   │
│   └── excel_writer.py
│       - save_consolidated_to_excel()
│
└── ml/                      [NEW - future ML tools]
    ├── __init__.py
    ├── export_ml_format.py
    └── training_utils.py
```

### 5.3 Refactoring Plan

**Step 1 - Extract Dialogs** (3 hours)

Move dialog classes from main.py to separate files:

```python
# ui/dialogs/save_meta_dialog.py

from PyQt6.QtWidgets import QDialog, QFormLayout, ...

class SaveMetaDialog(QDialog):
    """Dialog for setting metadata when saving analyzed data."""
    def __init__(self, abf_name, channel, parent=None, auto_stim="", history=None):
        # ... (move entire class from main.py ~line 3694)
        pass
```

Then in main.py:
```python
from ui.dialogs.save_meta_dialog import SaveMetaDialog

# Remove class definition, just use imported class
dlg = SaveMetaDialog(abf_name=..., channel=..., parent=self, ...)
```

**Repeat for**:
- EupneaParamsDialog → ui/dialogs/eupnea_params_dialog.py
- OutlierMetricsDialog → ui/dialogs/outlier_metrics_dialog.py
- SpectralAnalysisDialog → ui/dialogs/spectral_analysis_dialog.py

**Step 2 - Extract Export Logic** (3 hours)

Move export functions from main.py to core/export.py:

```python
# core/export.py

def export_analysis_bundle(
    state: AppState,
    save_dir: Path,
    save_base: str,
    meta_dict: dict,
    include_traces: bool = True,
    progress_callback=None
) -> dict:
    """
    Export complete analysis to NPZ + CSV + PDF.

    Returns:
        dict with paths: {'npz': Path, 'timeseries_csv': Path, ...}
    """
    # Move entire _export_all_analyzed_data logic here
    # Return dict of saved file paths
    pass

def load_analysis_bundle(npz_path: Path) -> AppState:
    """
    Load complete analysis from NPZ bundle.

    Returns:
        Restored AppState object
    """
    # Move NPZ loading logic here
    pass
```

Then in main.py:
```python
from core.export import export_analysis_bundle, load_analysis_bundle

def on_save_analyzed_clicked(self):
    # ... UI prompts ...
    result = export_analysis_bundle(
        self.state,
        save_dir=final_dir,
        save_base=suggested,
        meta_dict=vals,
        progress_callback=self._update_progress
    )
    # ... show success message ...

def on_open_analysis_clicked(self):
    # ... file dialog ...
    restored_state = load_analysis_bundle(npz_path)
    self.state = restored_state
    self._refresh_ui_from_state()
```

**Step 3 - Extract Consolidation Logic** (3 hours)

Move consolidation from main.py to curation/ module:

```python
# curation/consolidation.py

def consolidate_timeseries(
    files: list[tuple[str, Path]],
    metrics: list[str],
    npz_data_by_root: dict = None
) -> dict:
    """Consolidate time-series metrics across multiple files."""
    # Move _consolidate_means_files logic here
    pass

def consolidate_breath_histograms(
    files: list[tuple[str, Path]]
) -> dict:
    """Consolidate breath-by-breath histograms."""
    # Move histogram logic here
    pass

def consolidate_events(
    files: list[tuple[str, Path]]
) -> dict:
    """Consolidate event data (stimulus, apnea, eupnea, sniffing)."""
    # Move event consolidation logic here
    pass
```

```python
# curation/excel_writer.py

def save_consolidated_to_excel(
    consolidated_data: dict,
    save_path: Path,
    include_plots: bool = True
) -> None:
    """Save all consolidated data to multi-sheet Excel file."""
    # Move _save_consolidated_to_excel logic here
    pass
```

Then in main.py:
```python
from curation.consolidation import consolidate_timeseries, consolidate_events
from curation.excel_writer import save_consolidated_to_excel

def on_consolidate_save_data_clicked(self):
    # ... file selection ...
    consolidated = {}
    consolidated.update(consolidate_timeseries(means_files, selected_metrics))
    consolidated.update(consolidate_events(events_files))
    save_consolidated_to_excel(consolidated, save_path)
```

**Step 4 - Extract File Scanning** (1 hour)

```python
# curation/file_scanner.py

def scan_csv_groups(base_dir: Path) -> dict:
    """Scan directory for matching CSV groups."""
    # Move _scan_csv_groups logic here
    pass

def get_csv_metadata(csv_path: Path) -> dict:
    """Extract metadata from CSV file (subject, stim params, etc.)."""
    # Helper for displaying file info
    pass
```

**Step 5 - Update Imports & Test** (1 hour)

After each extraction:
1. Update imports in main.py
2. Run application
3. Test affected features
4. Fix any import errors
5. Commit changes

**Example commit sequence**:
```bash
git commit -m "Extract dialogs to ui/dialogs/ module"
git commit -m "Move export logic to core/export.py"
git commit -m "Move consolidation to curation/ module"
git commit -m "Create curation/file_scanner.py"
git commit -m "Final restructuring cleanup"
```

### 5.4 Benefits of Restructuring

**Immediate Benefits**:
- main.py shrinks from 13,000 → ~3,000 lines (77% reduction)
- Modules are testable in isolation
- Easier to find/modify specific functionality
- Faster for Claude to read and understand code
- Reduced token usage in future conversations

**Long-term Benefits**:
- Can add unit tests for each module
- Multiple developers can work on different modules
- Can package core/ as standalone library
- Easier to add new export formats
- Simpler to maintain and debug

### 5.5 Backward Compatibility

**No changes to**:
- File formats (.npz, .csv, .pdf)
- UI layout or behavior
- User workflows
- Saved data

All changes are **internal refactoring only**

### 5.6 Testing Strategy

After each module extraction:

**Unit Tests** (optional, +2 hours):
```python
# tests/test_export.py

from core.export import export_analysis_bundle, load_analysis_bundle

def test_export_roundtrip():
    """Test save and load produces identical state."""
    # Create dummy state
    # Export to tempdir
    # Load back
    # Assert equality
    pass
```

**Integration Tests** (required):
1. Run through full workflow: load → analyze → save → close → reopen
2. Test consolidation with 3 files
3. Test all dialogs still open/function
4. Test spectral analysis
5. Test manual editing modes

---

## Implementation Priority & Timeline

### Phase 1: Critical Features (Week 1)
**Days 1-2**: Task 1 - Sniffing Export & Consolidation (5-6 hours)
**Days 3-4**: Task 2 - NPZ Reopening (8-10 hours)

**Total**: 13-16 hours (2 days of focused work)

### Phase 2: Quality & Maintenance (Week 2)
**Days 1-2**: Task 4 - Code Cleanup (4-5 hours)
**Days 3-4**: Task 5 - Restructuring (8-12 hours)

**Total**: 12-17 hours (2 days of focused work)

### Phase 3: ML Infrastructure (Future)
**Week 3**: Task 3 - ML Export Format (6-8 hours)
- Can be done after Phase 1&2
- Separate developer can work on this in parallel

---

## Risk Assessment & Mitigation

### High Risk Areas

**Risk 1: NPZ reopening breaks with missing ABF**
- **Mitigation**: Implement hybrid approach (reload ABF if exists, fallback to downsampled)
- **Testing**: Create test with moved/deleted ABF files

**Risk 2: Code cleanup removes needed code**
- **Mitigation**: Create timestamped backups before deletion
- **Testing**: Full regression testing after cleanup

**Risk 3: Restructuring breaks imports**
- **Mitigation**: Incremental commits, test after each module
- **Testing**: Automated import checking script

### Medium Risk Areas

**Risk 4: Sniffing consolidation has edge cases**
- **Mitigation**: Handle empty sniffing data gracefully
- **Testing**: Test with files without sniffing annotations

**Risk 5: Large NPZ files if raw data included**
- **Mitigation**: Start with hybrid approach (ABF path only)
- **Testing**: Monitor NPZ file sizes

### Low Risk Areas

**Risk 6: ML export format not optimal**
- **Mitigation**: Start with separate converter script (not in main app)
- **Testing**: Can iterate without affecting main app

---

## Success Criteria

### Task 1 - Sniffing Export
- ✅ Sniffing regions saved to NPZ
- ✅ Sniffing events in events.csv
- ✅ Sniffing sheet in consolidated Excel
- ✅ Backward compatible (old files without sniffing work)

### Task 2 - NPZ Reopening
- ✅ Can open saved NPZ and continue work
- ✅ All peaks, breaths, sighs, sniffing regions restored
- ✅ Can re-save and overwrite existing files
- ✅ Warning shown if ABF unavailable

### Task 3 - ML Export
- ✅ Clear documentation for using NPZ in ML training
- ✅ (Optional) Dedicated ML export format created
- ✅ Example training script provided

### Task 4 - Code Cleanup
- ✅ 4000+ lines of commented code removed
- ✅ Important TODOs/notes preserved in separate file
- ✅ All features still work after cleanup
- ✅ Backup created before deletion

### Task 5 - Restructuring
- ✅ main.py under 4000 lines (70% reduction)
- ✅ Dialogs in ui/dialogs/
- ✅ Export in core/export.py
- ✅ Consolidation in curation/
- ✅ All features still work
- ✅ No change to user experience

---

## Questions for Review

Before starting implementation, confirm:

1. **NPZ Reopening - Raw Data**:
   - Should we save raw ABF data in NPZ (large files)?
   - Or use hybrid approach (reload ABF, fallback to downsampled)?
   - **Recommendation**: Hybrid approach

2. **ML Export**:
   - Should this be in main app or separate script?
   - What ML use cases should we prioritize?
   - **Recommendation**: Document NPZ usage first, add dedicated export later

3. **Code Cleanup**:
   - Should we review commented code manually before deletion?
   - Any specific blocks to preserve?
   - **Recommendation**: Yes, manual review of 30 minutes first

4. **Restructuring**:
   - Do we need unit tests after restructuring?
   - Should we add linting/type checking at this stage?
   - **Recommendation**: Integration tests only for now, defer formal testing

5. **Timeline**:
   - Does 4-week timeline work for your schedule?
   - Should we prioritize differently?
   - **Recommendation**: Phases 1+2 are high priority, Phase 3 can wait

---

## Next Steps

**Today (Planning)**:
- ✅ Review this plan
- ⏳ Answer questions above
- ⏳ Prioritize tasks if timeline too long

**Tomorrow (Start Implementation)**:
- Begin Task 1: Sniffing export (5-6 hours)
- Test thoroughly before moving to Task 2

**This Week**:
- Complete Tasks 1+2 (critical features)
- Create backups before any changes
- Document any issues/edge cases found

---

## Appendix: File Location Reference

**Key files for each task**:

**Task 1 (Sniffing)**:
- `main.py` lines 5721-6006 (NPZ save)
- `main.py` lines 6398-6538 (events CSV)
- `main.py` lines 8665-8836 (consolidation)
- `core/metrics.py` lines 889-894 (detect_eupnic_regions)

**Task 2 (NPZ Reopen)**:
- `main.py` ~line 550 (add new button handler)
- `main.py` lines 10642-10720 (existing NPZ load helpers)
- `ui/pleth_app_layout_02_horizontal.ui` (add button)

**Task 3 (ML Export)**:
- New file: `scripts/npz_to_ml_format.py`
- New file: `docs/ML_TRAINING_GUIDE.md`

**Task 4 (Cleanup)**:
- `main.py` (entire file - remove commented code)
- New file: `scripts/clean_commented_code.py`

**Task 5 (Restructure)**:
- Extract from `main.py` to:
  - `ui/dialogs/*.py`
  - `core/export.py`
  - `curation/*.py`
