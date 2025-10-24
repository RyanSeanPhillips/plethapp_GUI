# PlethApp - Supported File Formats

PlethApp supports multiple neurophysiology file formats through a modular I/O architecture located in `core/io/`. All loaders return data in a standardized format for seamless integration with the application.

## Standardized Output Format

All file loaders return:
```python
sr_hz, sweeps_by_channel, channel_names, t = load_function('file.ext')

# Where:
# sr_hz: float - Sample rate in Hz
# sweeps_by_channel: dict[str, np.ndarray] - channel_name -> (n_samples, n_sweeps)
# channel_names: list[str] - Ordered list of channel names
# t: np.ndarray - Time vector (optional, can be None)
```

---

## Axon Binary Format (ABF) Files

### Overview
ABF is the native format for Axon Instruments/Molecular Devices acquisition systems (pCLAMP, Clampex).

### Implementation
- **Loader**: `core/abf_io.py` (dispatcher) uses pyabf library
- **Dependencies**: `pyabf` (installed via pip)

### Supported Features
- ABF1 and ABF2 formats
- Multi-sweep episodic recordings
- Multi-channel recordings
- Full metadata (protocol, sample rate, units, gains)
- Sweep timing and stimulus information

### Usage Example
```python
from core.abf_io import load_abf_or_similar

# Single file
sr_hz, sweeps_by_channel, channel_names, t = load_abf_or_similar('recording.abf')

# Multi-file concatenation (see FEATURE_BACKLOG.md #3)
# Coming soon: load multiple ABF files as sequential sweeps
```

### Common Issues
- **Channel naming**: ABF channel names may be generic (e.g., "IN 0", "IN 1"). Consider renaming in Clampex before acquisition.
- **Sweep timing**: Sweep start times available via pyabf metadata
- **Large files**: ABF files >2GB may have performance issues with pyabf

---

## Spike2 SMRX Files (SON64 Format)

### Overview
SMRX is the native format for CED (Cambridge Electronic Design) Spike2 acquisition software, widely used in neurophysiology.

### Implementation
PlethApp reads SMRX files using the official CED SON64 library via a custom ctypes wrapper.

**Files**:
- **core/io/son64_dll_loader.py**: Low-level ctypes wrapper for `ceds64int.dll`
- **core/io/son64_loader.py**: High-level loader that converts SMRX data to PlethApp format
- **core/io/s2rx_parser.py**: Parser for Spike2 `.s2rx` XML configuration files

**Dependencies**:
Requires CED MATLAB SON library (CEDS64ML) installed at `C:\CEDMATLAB\CEDS64ML\`

### Key Technical Details

#### Accurate Timing
Uses `S64ChanDivide` to get actual sample interval in ticks (critical for correct timing):
```python
time[i] = (first_tick + i * chan_divide) * time_base
```
This ensures sample-accurate timing even with unusual sample rates.

#### Multi-Segment Data
Handles files with gaps (multi-segment recordings) by:
1. Reading all segments for each channel
2. Concatenating segments into continuous arrays
3. Maintaining correct timing across gaps

#### Multi-Rate Channel Support
- Automatically detects different sample rates across channels
- Resamples all channels to lowest common rate (avoids upsampling artifacts)
- Uses `scipy.interpolate.interp1d` for high-quality resampling

#### Channel Visibility Filtering
**Automatic channel hiding**:
- Reads `.s2rx` configuration file (same name as `.smrx` file)
- Respects `Vis="0"` attribute in channel settings
- Hides channels marked as hidden in Spike2
- Defaults to showing all channels if `.s2rx` not found
- Channels not mentioned in `.s2rx` default to visible

### Supported Features
- Waveform channels (ADC and ADCMARK types)
- Multiple sample rates with automatic alignment
- Full time range with accurate tick-based timing
- Channel metadata (titles, units, scale, offset)
- Multi-segment recordings

### Usage Example
```python
from core.io.son64_loader import load_son64

# Load file (returns PlethApp format)
sr_hz, sweeps_by_channel, channel_names, t = load_son64('recording.smrx')
# sweeps_by_channel[channel_name] -> shape (n_samples, 1) - continuous recording
```

### Important Limitations

#### File Locking
**CRITICAL**: SMRX files must be closed in Spike2 before opening in PlethApp.
- CED DLL uses exclusive file locking
- Error code -1 from S64Open indicates file is locked or path issue
- **Solution**: Close file in Spike2, then open in PlethApp

#### Channel Type Support
Only waveform channels (kind 1 or 7) are loaded:
- ADC channels (continuous waveforms)
- ADCMARK channels (waveforms with markers)
- Event channels, text markers, and real markers are not supported

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| Error code -1 from S64Open | File locked in Spike2 | Close file in Spike2 |
| Error code -1 from S64Open | Invalid file path | Check path, try absolute path |
| Data duplication | Incorrect sample rate calculation | Fixed in current version (uses S64ChanDivide) |
| Missing channels | Non-waveform channels | Only ADC/ADCMARK channels loaded |
| Empty data | No waveform channels in file | Check file contents in Spike2 |

### Installation Requirements

1. **Install CED MATLAB SON Library**:
   - Download from CED website: http://ced.co.uk/downloads/latest/matlab
   - Install to `C:\CEDMATLAB\CEDS64ML\`
   - Verify `ceds64int.dll` exists in installation directory

2. **Verify Installation**:
   ```python
   from core.io.son64_dll_loader import SON64Library
   lib = SON64Library()  # Should not raise an error
   ```

---

## European Data Format (EDF) Files

### Overview
EDF and EDF+ are standard formats for physiological recordings, commonly used in sleep studies, EEG, and respiratory monitoring.

### Implementation
- **Loader**: `core/io/edf_loader.py` uses pyedflib library
- **Dependencies**: `pyedflib` (requires numpy<2.0 as of version 0.1.30)

### Key Technical Details

#### Annotation Channel Filtering
- Automatically filters out EDF Annotations channels (EDF+ specific)
- Only waveform channels with actual data are loaded
- Annotation data is ignored (not currently used by PlethApp)

#### Multi-Rate Channel Support
- Resamples all channels to lowest sample rate
- Avoids upsampling artifacts
- Uses `scipy.interpolate.interp1d` for resampling

#### Channel Naming
- Channel names include units when available (e.g., "Respiration (mmHg)")
- Handles duplicate channel names by appending channel index
- Physical units extracted from EDF header

### Supported Features
- EDF and EDF+ formats
- Multi-rate waveform channels (automatically aligned to lowest rate)
- Channel metadata (names, units, sample rates)
- Physical units display
- Continuous recordings (treated as single sweep in PlethApp format)

### Usage Example
```python
from core.io.edf_loader import load_edf

# Load file (returns PlethApp format)
sr_hz, sweeps_by_channel, channel_names, t = load_edf('recording.edf')
# sweeps_by_channel[channel_name] -> shape (n_samples, 1) - continuous recording
```

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| ImportError for pyedflib | Library not installed | `pip install pyedflib` |
| Numpy version conflict | pyedflib requires numpy<2.0 | `pip install "numpy<2.0"` |
| Missing channels | Only annotation channels in file | Check file contents with EDFbrowser |
| Empty file error | No valid waveform data | Verify EDF file is not corrupted |

### Installation Requirements

```bash
pip install pyedflib
# If using numpy 2.0+, downgrade:
pip install "numpy<2.0"
```

---

## Planned Format Support

### CSV/Text Time-Series (Planned)
See FEATURE_BACKLOG.md #4 for detailed implementation plan.

**Features**:
- Generic CSV/text file import
- Column selection UI
- Auto-detect headers, delimiters, sample rate
- File preview with first 20 rows

**Use cases**:
- Import data from custom acquisition systems
- Load exported data from other analysis software
- Test with synthetic data

### SpikeGLX (Planned)
Neuropixels data format support (post v1.0).

### TDT (Planned)
Tucker-Davis Technologies format support (post v1.0).

---

## File Format Dispatcher

The main file loading logic in `core/abf_io.py` acts as a dispatcher that routes to the appropriate loader based on file extension:

```python
def load_abf_or_similar(file_path: str):
    """
    Universal file loader - dispatches to appropriate loader based on extension.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext == '.abf':
        return load_abf(file_path)
    elif ext == '.smrx':
        return load_son64(file_path)
    elif ext == '.edf':
        return load_edf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
```

This architecture makes it easy to add new formats - just add a new loader and register it in the dispatcher.

---

## Multi-File Concatenation (Planned)

See FEATURE_BACKLOG.md #3 for detailed implementation plan.

**Planned features**:
- Multi-select ABF files in file dialog
- Treat files as sequential sweeps
- Validation checks:
  - Same number of channels
  - Same channel names
  - Same sample rate
  - Same date (from filename format: `YYYYMMDD####.abf`)
- Warning dialog if validation fails
- Display concatenated file info in status bar

---

## Future: Universal Data Loader Framework

See MULTI_APP_STRATEGY.md for long-term vision of extracting file loaders into a standalone `neurodata-io` package that can be used across multiple applications.

**Benefits**:
- Write loader once, use in all apps (PlethApp, photometry, ephys)
- Consistent API across all neurophysiology file formats
- Easy to add new formats
- Community contributions

---

## Related Documentation
- **MULTI_APP_STRATEGY.md**: Universal data loader framework (long-term vision)
- **FEATURE_BACKLOG.md**: Planned file format features
- **CLAUDE.md**: Quick reference for development commands
