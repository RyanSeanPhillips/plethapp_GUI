# Multi-App Neurophysiology Ecosystem - Strategic Roadmap

**Status**: Preliminary planning document (not for immediate implementation)
**Timeline**: Long-term goal after PlethApp v1.0 publication + ML features complete
**Last Updated**: 2025-10-20

---

## Vision

Build a **unified neurophysiology analysis ecosystem** with reusable infrastructure that can power multiple analysis applications:

- **PlethApp** (breath analysis) - Already built, 90% modular
- **PhotometryApp** (fiber photometry) - Future application
- **NeuropixelsViewer** (ephys data) - Future application
- **[Your custom app here]** - Easy to build with shared foundation

---

## Core Infrastructure Packages

### 1. `neurodata-io` - Universal File Loader

**Purpose**: Single API for loading all neurophysiology file formats

**Supported Formats**:
- Axon Binary Format (.abf) - ✅ Already implemented in PlethApp
- Spike2 SON64 (.smrx) - ✅ Already implemented in PlethApp
- European Data Format (.edf) - ✅ Already implemented in PlethApp
- Generic CSV/TSV time-series - Planned
- SpikeGLX binary (.bin) - For Neuropixels data
- Tucker-Davis Systems (.tdt) - For photometry/ephys
- Multi-file workflows - For photometry sessions

**Unified API**:
```python
from neurodata_io import load_data

# Auto-detect format from file extension
data = load_data("recording.abf")
data = load_data("recording.smrx")

# Multi-file concatenation
data = load_data(
    ["file1.abf", "file2.abf", "file3.abf"],
    strategy="concatenate"
)

# Standardized output across all formats
{
    'signals': dict[str, np.ndarray],  # channel_name -> (n_samples, n_trials)
    'sample_rate': float,
    'time': np.ndarray,
    'metadata': dict,
    'source_files': list[Path]
}
```

**Implementation Source**:
- Extract from PlethApp `core/abf_io.py` and `core/io/`
- Already 80% complete - just needs packaging!

---

### 2. `neuroviz` - Reusable PyQt6 UI Components

**Purpose**: Drop-in matplotlib integration and scientific UI widgets

**Components**:
- `PlotHost`: Matplotlib canvas with toolbar (from PlethApp `core/plotting.py`)
- `ChannelSelector`: Dropdown for selecting data channels
- `FilterControls`: Lo/Hi pass filter spinboxes with order selection
- `NavigationPanel`: Window/sweep navigation controls
- `ProgressDialog`: File loading progress with cancel button
- Dark theme CSS and matplotlib styles

**Usage Example**:
```python
from neuroviz import PlotHost, ChannelSelector, apply_dark_theme

# Create plot widget (handles matplotlib backend automatically)
plot_widget = PlotHost()
plot_widget.show_trace(y, t, title="Signal")

# Add reusable channel selector
channel_selector = ChannelSelector(channel_names=["Ch1", "Ch2"])
channel_selector.channel_changed.connect(on_channel_changed)

# Apply consistent dark theme
apply_dark_theme(app)
```

**Implementation Source**:
- Extract from PlethApp `core/plotting.py`, `core/navigation.py`
- UI components from PlethApp's common patterns

---

### 3. `neurocore` - Signal Processing & ML Framework

**Purpose**: Shared analysis utilities and machine learning tools

**Modules**:
- **Filtering**: Butterworth filters, notch filters, mean subtraction (from PlethApp `core/filters.py`)
- **Event Detection**: Generic peak/event detection algorithms
- **Resampling**: Common downsampling/upsampling with anti-aliasing
- **Alignment**: Multi-signal temporal alignment utilities
- **ML Framework**:
  - Breath classifier (PlethApp-specific)
  - Artifact detector (photometry-specific)
  - Base classifier class (shared foundation)
  - Active learning utilities

**Usage Example**:
```python
from neurocore import filters, event_detection
from neurocore.ml import BreathClassifier

# Apply butterworth filter (same as PlethApp)
y_filtered = filters.butterworth_lowpass(y, cutoff_hz=10, sr_hz=1000, order=4)

# Detect peaks/events
peaks = event_detection.find_peaks(y, threshold=0.5, prominence=0.1)

# Use pre-trained ML classifier
classifier = BreathClassifier()
breath_types = classifier.predict(breath_features)
```

**Implementation Source**:
- Extract from PlethApp `core/filters.py`, `core/metrics.py`, `core/peaks.py`
- Add ML components from PUBLICATION_ROADMAP.md

---

## Application Architecture

### Shared Foundation (60-70% of code)
All apps use the same base infrastructure:
```
your_app/
├── main.py                    # App-specific UI and logic
├── requirements.txt           # Includes neurodata-io, neuroviz, neurocore
├── domain/                    # Domain-specific algorithms
│   ├── analysis.py           # Your unique analysis methods
│   └── models.py             # Data models specific to your domain
└── ui/
    └── main_window.ui        # PyQt6 UI file
```

### Dependencies in pyproject.toml
```toml
dependencies = [
    "neurodata-io>=1.0",       # File loading
    "neuroviz>=1.0",           # UI components
    "neurocore>=1.0",          # Signal processing
    "pyqt6",
    "numpy",
    "scipy",
    "matplotlib"
]
```

---

## Example: PhotometryApp (Future Application)

**Unique Logic** (30-40% custom code):
- ΔF/F0 normalization
- Isosbestic correction (405nm vs 465nm signals)
- PETH (Peri-Event Time Histogram) around behavioral events
- Exponential photobleaching correction
- Multi-session alignment

**Reused Infrastructure** (60-70% from packages):
- File loading: `neurodata_io.load_data("photometry.csv")`
- Plotting: `neuroviz.PlotHost()` for traces
- Filtering: `neurocore.filters.butterworth_lowpass()`
- UI widgets: `neuroviz.ChannelSelector`, `neuroviz.FilterControls`

**Development Time**: 8-12 hours (vs 30+ hours from scratch!)

---

## Example: NeuropixelsViewer (Future Application)

**Unique Logic**:
- Raster plots (spikes across trials)
- LFP spectrograms
- Spike waveform visualization
- Integration with Kilosort/Phy output

**Reused Infrastructure**:
- File loading: `neurodata_io.load_data("recording.bin", format="spikeglx")`
- Plotting: `neuroviz.PlotHost()` with custom renderers
- Filtering: `neurocore.filters` for LFP preprocessing
- Navigation: `neuroviz.NavigationPanel` for time windows

**Development Time**: 12-16 hours (vs 40+ hours from scratch!)

---

## Migration Plan for PlethApp

### Goal: Extract packages without breaking PlethApp

**Strategy**: Gradual transition with backward compatibility

```python
# In PlethApp main.py (during transition period)
try:
    # Try to use new package
    from neurodata_io import load_data as load_data_file
    from neuroviz import PlotHost
    from neurocore import filters
    print("[PlethApp] Using shared packages (neurodata-io, neuroviz, neurocore)")
except ImportError:
    # Fallback to local copies
    from core.abf_io import load_data_file
    from core.plotting import PlotHost
    from core import filters
    print("[PlethApp] Using local modules (packages not installed)")
```

**Transition Steps**:
1. Create packages with extracted code
2. Test packages independently
3. Install packages in PlethApp environment
4. Run full PlethApp test suite (ensure no regressions)
5. Once stable, remove local copies and depend only on packages

---

## Benefits

### For Development
- **Faster new apps**: 60-70% of code already written
- **Consistent UX**: All apps have same look and feel
- **Single maintenance point**: Fix bug once, benefits all apps
- **Easier testing**: Test infrastructure packages independently

### For Career/Portfolio
- **Infrastructure design experience**: Highly valued in industry
- **Multiple citable packages**: 3+ Zenodo DOIs for publications
- **Adoption potential**: Other labs can use your packages
- **Professional credibility**: Shows systems thinking, not just scripting

### For Publications
- **PlethApp paper**: Cites neurodata-io, neuroviz, neurocore as dependencies
- **PhotometryApp paper**: Cites same packages (cumulative citations!)
- **Infrastructure paper**: Separate publication for the framework itself
- **Cross-domain impact**: Framework enables work beyond just breathing

---

## Implementation Timeline

### Phase 1: Package Extraction (4 hours)
- Create `neurodata-io` package (extract from `core/abf_io.py`, `core/io/`)
- Create `neuroviz` package (extract from `core/plotting.py`, `core/navigation.py`)
- Create `neurocore` package (extract from `core/filters.py`, signal processing utils)
- Test PlethApp with new packages (ensure no regressions)

### Phase 2: Validation via New App (8-16 hours)
**Option A**: Build PhotometryApp using packages (validates file I/O, filtering, UI)
**Option B**: Build NeuropixelsViewer using packages (validates plotting, navigation)

### Phase 3: Publication & Distribution (2 hours)
- Publish packages to PyPI: `pip install neurodata-io neuroviz neurocore`
- Create Zenodo DOIs for each package
- Write READMEs with examples and documentation
- Update PlethApp dependencies in `pyproject.toml`

### Total Time Investment
- **Initial setup**: 4 hours (extraction + testing)
- **Validation**: 8-16 hours (build second app)
- **Publication**: 2 hours (PyPI + documentation)
- **Total**: 14-22 hours

### Time Savings
- **PhotometryApp from scratch**: ~35 hours
- **PhotometryApp with packages**: ~10 hours
- **Savings**: 25 hours (and every future app saves 20-30 hours!)

---

## Prerequisites

**Before starting package extraction**:
1. ✅ PlethApp v1.0 publication submitted (JOSS)
2. ✅ ML breath classifier implemented and validated
3. ✅ Performance optimizations complete (Phase 1 from PERFORMANCE_OPTIMIZATION_PLAN.md)
4. ✅ PlethApp codebase stable and well-tested

**Timing**: After 12-week PlethApp v1.0 timeline in PUBLICATION_ROADMAP.md

---

## Risk Mitigation

### Risk 1: Breaking PlethApp during extraction
**Mitigation**: Use transition imports with fallbacks (see Migration Plan above)

### Risk 2: Package maintenance overhead
**Mitigation**: Start with simple semantic versioning, only break compatibility in major versions

### Risk 3: Over-engineering too early
**Mitigation**: Only extract after PlethApp v1.0 is stable and proven

### Risk 4: Packages too PlethApp-specific
**Mitigation**: Validate with second app (PhotometryApp) before publishing

---

## Success Metrics

### Technical Metrics
- **Code reuse**: ≥60% of new app code comes from packages
- **Development speed**: New app built in ≤2 weeks
- **Bug propagation**: Bugfix in package fixes all apps

### Career Metrics
- **Adoptions**: ≥3 apps using the packages (your own)
- **External users**: ≥1 other lab tries the packages
- **Citations**: ≥3 separate Zenodo DOIs
- **Industry interviews**: Can discuss infrastructure design and reusable architecture

---

## Decision Points

### When to Start?
**Not now** - wait until:
- PlethApp v1.0 published and stable
- ML classifier implemented and validated
- You have time to build a second app to validate the packages

### Should You Do This At All?
**Yes, if**:
- You plan to build 2+ more analysis apps
- You want infrastructure/architecture experience for industry jobs
- You enjoy systems design and creating reusable tools

**No, if**:
- PlethApp is your only planned application
- You prefer domain science over software infrastructure
- Timeline constraints require focus on immediate publications only

---

## Alternative: Lighter Approach

If full package extraction seems like too much, consider a **lighter shared library**:

```
shared_neuro_utils/  (single package, not three)
├── io.py              # File loaders
├── plotting.py        # PlotHost only
├── filters.py         # Signal processing
└── __init__.py
```

**Benefits**: Faster to set up (2 hours vs 4 hours)
**Tradeoffs**: Less modular, harder to version independently

---

## Next Steps (When Ready)

1. **Review this document** with fresh perspective after PlethApp v1.0 is published
2. **Decide on scope**: Full 3-package extraction vs lighter shared library
3. **Choose validation app**: PhotometryApp (easier) or NeuropixelsViewer (more complex)
4. **Block time**: 14-22 hours for full implementation
5. **Execute Phase 1**: Package extraction and testing

---

## Related Documents

- **PUBLICATION_ROADMAP.md** - PlethApp v1.0 publication timeline (prerequisite)
- **PERFORMANCE_OPTIMIZATION_PLAN.md** - Performance work before extraction
- **CLAUDE.md** - Current PlethApp architecture documentation

---

**Remember**: This is a **strategic vision**, not an immediate task. Focus on PlethApp v1.0 publication first, then revisit this plan when you're ready to build your second neurophysiology app.
