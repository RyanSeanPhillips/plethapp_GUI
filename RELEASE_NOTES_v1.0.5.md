# PlethApp v1.0.5 Release Notes

**Release Date:** 2025-10-08

## Major New Features

### Multi-File ABF Concatenation
Load multiple ABF files simultaneously and view them as sequential sweeps. Perfect for analyzing long recording sessions split across multiple files.

**Key Features:**
- **Multi-Select File Dialog**: Select multiple files at once using Browse button
- **Automatic Validation**: Ensures files are compatible before loading
  - Same file type (.abf or .smrx)
  - Same number of channels
  - Same channel names
  - Same sample rate (±0.1 Hz tolerance)
- **Smart Warnings**: Optional warnings for:
  - Different sweep lengths (will be padded with NaN)
  - Different dates in filenames
- **Seamless Integration**: Multi-file data appears as one continuous dataset

**Example Usage:**
- File 1: 10 sweeps → Sweeps 0-9
- File 2: 5 sweeps → Sweeps 10-14
- File 3: 3 sweeps → Sweeps 15-17
- Total: 18 sweeps to navigate

### NaN Padding for Variable-Length Sweeps
Handles ABF files with different recording durations automatically.

**How It Works:**
- Identifies the longest sweep across all files
- Pads shorter sweeps with NaN values at the end
- Displays original vs. padded duration in success message
- Example: "File 2: sweeps 1-1 (padded: 24.27s → 28.06s)"

### Enhanced Filtering with NaN Support
Butterworth filters now handle NaN-padded data correctly.

**Technical Details:**
- Detects NaN values in input signal
- Extracts only the valid (non-NaN) region
- Applies all filters (low-pass, high-pass, mean subtraction, invert) to valid portion only
- Reconstructs full array with filtered data and NaN padding
- Prevents scipy filter propagation of NaN values throughout entire array

## UI Improvements

### Multi-Channel Grid View Sweep Navigation
Use Prev/Next Sweep buttons while viewing "All Channels" grid mode.

**Benefits:**
- Quickly scan through all sweeps in overview mode
- Compare all channels across different sweeps
- Spot recording issues or artifacts before detailed analysis
- Title updates automatically: "All channels | sweep 2 | file 2/3 (filename.abf)"

### Enhanced File Information Display
Plot titles now show which file each sweep came from when multiple files are loaded.

**Display Format:**
- Single channel: "Channel Name | sweep 15 | file 2/3 (25130086.abf)"
- Multi-channel: "All channels | sweep 15 | file 2/3 (25130086.abf)"

## Technical Improvements

### File Metadata Tracking
- Added `file_info` field to `AppState` dataclass
- Stores sweep ranges and file paths for each loaded file
- Format: `[{'path': Path, 'sweep_start': int, 'sweep_end': int, 'padded': bool, ...}]`

### Validation and Error Handling
- Comprehensive pre-load validation with detailed error messages
- Graceful handling of edge cases (empty files, corrupted data)
- Clear user feedback via dialog boxes

### Performance Optimizations
- Fast metadata-only validation (doesn't load full data until approved)
- Efficient NaN padding with numpy operations
- Minimal memory overhead for multi-file workflows

## Bug Fixes

- **Fixed:** Scipy Butterworth filters now handle NaN values correctly (previously turned entire array to NaN)
- **Fixed:** Multi-file concatenation properly calculates max sweep length from actual data, not time array
- **Fixed:** Array bounds checking prevents index errors during concatenation
- **Fixed:** Grid view now updates when navigating sweeps (was stuck on first sweep)

## Files Modified

- `main.py`: Multi-file loading, browse dialog, title display, grid view navigation
- `core/abf_io.py`: Validation, concatenation, NaN padding logic
- `core/filters.py`: NaN-aware filtering with `_apply_filters_to_valid()` helper
- `core/state.py`: Added `file_info` field for metadata tracking
- `version_info.py`: Version bump to 1.0.5

## Known Limitations

- Multi-file concatenation currently only supports ABF files (SMRX support coming in future release)
- NaN padding assumes contiguous valid data followed by NaN padding (does not handle gaps in middle of data)

## Upgrade Notes

- No breaking changes - all existing workflows remain compatible
- Single-file loading behavior unchanged
- New features are opt-in (multi-file selection is optional)

## Build Instructions

```bash
# Windows
build_executable.bat

# Cross-platform
python build_executable.py
```

Output: `dist/PlethApp_v1.0.5/PlethApp_v1.0.5.exe`

## Acknowledgments

Version 1.0.5 developed with assistance from Claude Code (Anthropic).

---

**Previous Version:** v1.0.4
**Next Planned Version:** v1.0.6 (NPZ file save/load with full state restoration)
