# PlethApp - Architecture & Design

This document describes the software architecture, design patterns, and technical implementation details of PlethApp.

## State Management

### AppState (core/state.py)

The application uses a centralized state system that manages all data and analysis results. This provides a single source of truth and simplifies data flow.

**Managed State**:
- **Current data and metadata**: Loaded file information, sample rates, channel data
- **Peak detection results**: Detected peaks, onsets, offsets, expiratory minima
- **Filter parameters**: Current filter settings (high-pass, low-pass, order, notch)
- **Navigation state**: Current sweep, channel, time window
- **User annotations**: Manual peak edits, sighs, custom markers
- **GMM clustering results**: Eupnea/sniffing classification probabilities
- **Outlier detection**: Flagged breath cycles based on cross-sweep statistics

**Key Advantages**:
- Single source of truth prevents state synchronization bugs
- Easy to serialize entire state for save/load functionality
- Simplifies undo/redo implementation (future feature)
- Clear data flow: UI → State → Processing → UI

**State Access Pattern**:
```python
st = self.app_state  # Main window has reference to AppState
st.analyze_chan = "Pleth"  # Update state
peaks = st.peaks_by_sweep.get(sweep_idx, np.array([]))  # Read state
```

---

## UI Architecture

### Main Window Design

**Layout System**: Grid-based layout (`gridLayout_3`) with three rows:
1. **Row 0**: Control panels (left-aligned, fixed size)
2. **Row 1**: Main plot (expands to fill space)
3. **Row 2**: Navigation controls (left-aligned, fixed size)

**UI Definition**: `ui/pleth_app_layout_02.ui` - PyQt6 Designer file
- Converted to Python code at runtime via `uic.loadUi()`
- Allows visual UI editing in Qt Designer
- Preserves layout properties in XML format

### PlotHost Integration

**Custom matplotlib integration** (`PlotHost` class in `core/plotting.py`):
- Embeds matplotlib figures within PyQt6 widgets
- Manages canvas refresh and event handling
- Provides methods for updating plot artists (lines, regions, markers)
- Optimized for scientific data visualization

**Key Methods**:
- `update_main_plot()`: Full plot redraw with all data and annotations
- `update_region_overlays()`: Lightweight overlay update (eupnea, apnea, outliers)
- `update_peak_artists()`: Update peak markers without full redraw
- `canvas.draw_idle()`: Deferred rendering for responsive UI

### Responsive Design

**Control Panel Layout**:
- Controls remain left-justified (don't stretch)
- Uses `Qt::AlignLeft|Qt::AlignTop` alignment properties
- Fixed width prevents UI element distortion

**Main Plot**:
- Expands horizontally and vertically with window resize
- No alignment property (allows stretching)
- Maintains aspect ratio of data, not plot frame

### Dark Theme

**Custom CSS styling** optimized for scientific data visualization:
- Dark background reduces eye strain during long analysis sessions
- High contrast for plot elements (peaks, regions, text)
- Consistent color scheme across all dialogs and windows

**Styled Elements**:
- Status bar: Black background (#1e1e1e), white text (#d4d4d4)
- Buttons: Rounded edges, hover effects
- Dialogs: Dark backgrounds with high-contrast text
- Plot: Dark gray background, white axes, colored data

---

## Qt Designer Left-Alignment Fix

**IMPORTANT**: Qt Designer tends to remove alignment attributes when saving `.ui` files. To maintain left-justified controls, manually re-apply these properties after editing in Qt Designer.

### Required Alignment Properties

#### 1. Main Grid Layout (gridLayout_3)

```xml
<item row="0" column="0" alignment="Qt::AlignLeft|Qt::AlignTop">
  <layout class="QVBoxLayout" name="verticalLayout_8">
    <!-- Control panels -->
  </layout>
</item>

<item row="1" column="0">
  <!-- NO alignment - plot must expand -->
  <widget class="QWidget" name="MainPlot">
  </widget>
</item>

<item row="2" column="0" alignment="Qt::AlignLeft|Qt::AlignTop">
  <layout class="QHBoxLayout" name="horizontalLayout_12">
    <!-- Navigation controls -->
  </layout>
</item>
```

#### 2. Top Controls Container (verticalLayout_8)

```xml
<layout class="QVBoxLayout" name="verticalLayout_8">
  <property name="alignment">
    <set>Qt::AlignLeft|Qt::AlignTop</set>
  </property>
  <!-- Control panels -->
</layout>
```

#### 3. Horizontal Layouts Inside verticalLayout_8

Each of these layouts needs left-center alignment:
- `horizontalLayout_7` (Browse/File Selection)
- `horizontalLayout_8` (Channel Selection)
- `horizontalLayout_9` (Filters)
- `horizontalLayout_10` (Peak Detection)

```xml
<layout class="QHBoxLayout" name="horizontalLayout_7">
  <property name="alignment">
    <set>Qt::AlignLeft|Qt::AlignVCenter</set>
  </property>
  <item>
    <!-- widgets here -->
  </item>
</layout>
```

### How to Apply After Qt Designer Editing

1. **Open `.ui` file in text editor** (VS Code, Notepad++)
2. **Find grid layout item** for row 0 (control panels)
3. **Add alignment attribute** to `<item>` tag:
   ```xml
   <item row="0" column="0" alignment="Qt::AlignLeft|Qt::AlignTop">
   ```
4. **Find verticalLayout_8** and add alignment property
5. **Find each horizontal layout** inside and add alignment property
6. **Save file** and test in application

**Note**: Always make a backup before manually editing `.ui` files.

---

## Performance Optimizations

### 1. Decimated Sampling

**Strategy**: Reduce sample rate for metrics calculations that don't require full resolution.

**Implementation**:
- Regularity score: 10 Hz effective sample rate (was 1000 Hz)
- Statistical calculations: Sample every 100th point
- Visual plotting: Downsample when >10,000 points in view

**Results**: 10-100× speedup with negligible accuracy loss

### 2. Lazy Loading

**Strategy**: Load and process data only when needed.

**Implementation**:
- Channel data cached only when selected
- Processed traces cached with parameter-based invalidation
- Peak detection results cached per sweep

**Results**: Reduced memory usage, faster file loading

### 3. Efficient Plotting

**Strategy**: Update only changed elements instead of full redraw.

**Implementation**:
- `_refresh_eupnea_overlays_only()`: Updates region overlays without recomputing metrics
- Artist reuse: Modify existing plot artists instead of creating new ones
- `canvas.draw_idle()`: Batches draw requests for better performance

**Specific Optimizations**:
- **GMM update**: 5-10× faster by skipping outlier detection recomputation
- **Peak editing**: Lightweight refresh instead of full redraw
- **Region overlays**: Separate artist updates for eupnea, apnea, sniffing, outliers

**Results**:
- GMM clustering update: ~0.1-0.2s (was ~1s)
- Peak add/delete: ~0.05s (was ~0.2s)
- Application feels "snappier" during interactive editing

### 4. Caching Strategy

**Processing Cache** (`_proc_key()` method):
```python
def _proc_key(self):
    """Generate cache key based on processing parameters."""
    return (
        self.app_state.analyze_chan,
        self.sweep_idx,
        self.filter_hp,
        self.filter_lp,
        self.filter_order,
        self.notch_lower,
        self.notch_upper,
        self.subtract_mean,
        self.invert_signal,
    )
```

**Cache Invalidation**:
- Automatic when any filter parameter changes
- Manual invalidation when raw data changes
- Separate caches for different processing stages

### 5. Background Processing (Future)

**Planned**: Move heavy computations to background threads
- Outlier detection across all sweeps
- GMM clustering on large datasets
- Export operations for multiple sweeps

**Benefits**: Non-blocking UI during long operations

---

## Build System

### PyInstaller Configuration

**Spec File**: `pleth_app.spec` - Comprehensive configuration for executable creation

**Key Settings**:
- **Directory distribution**: Fast startup (~6s) vs single-file (~15s)
- **Icon integration**: Custom application icons for Windows
- **Hidden imports**: Explicit imports for dynamically loaded modules
- **Excluded modules**: Remove conflicting Qt bindings (PySide2, PyQt5)
- **Data files**: Include UI files, images, assets

**Build Methods**:
1. **Batch script** (Windows): `build_executable.bat`
2. **Python script** (cross-platform): `build_executable.py`
3. **Manual**: `python version_info.py && pyinstaller --clean pleth_app.spec`

### Version Management

**Version Info**: `version_info.py` generates Windows version resource
- Embedded in executable metadata
- Visible in file properties dialog
- Includes company, copyright, version number

**Version Format**: `v0.9.x` (pre-release), moving to `v1.0.0` (stable)

### Dependency Management

**Careful module exclusion** to prevent conflicts:
```python
excludes=['PySide2', 'PySide6', 'PyQt5', 'tkinter']
```

**Hidden imports** for dynamically loaded modules:
```python
hiddenimports=['scipy.signal', 'sklearn.mixture', 'pyedflib']
```

### Distribution Structure

```
dist/PlethApp_v1.0.0/
├── PlethApp.exe          # Main executable
├── _internal/            # Dependencies
│   ├── numpy/
│   ├── scipy/
│   ├── matplotlib/
│   └── ...
├── ui/                   # UI definition files
├── images/               # Application assets
└── examples/             # Sample data (optional)
```

---

## Deployment Notes

### Target Platform
- **Primary**: Windows 10/11 (64-bit)
- **Future**: macOS, Linux (PyQt6 is cross-platform)

### Distribution Size
- **Typical**: ~200-400MB due to scientific libraries
- **Compressed**: ~100-150MB (7zip, self-extracting)

### Startup Time
- **Directory distribution**: ~6 seconds (cold start)
- **Single-file distribution**: ~15 seconds (extraction overhead)

### Runtime Dependencies
- **Self-contained**: No Python installation required
- **External**: CED MATLAB SON library (for SMRX file support only)
  - Not included in distribution
  - User must install separately if using SMRX files
  - Path: `C:\CEDMATLAB\CEDS64ML\`

### Installation
- **No installer required**: Unzip and run
- **User data**: Stored in user directory (future: settings, cache)
- **Updates**: Manual download and replace (future: auto-update)

---

## Architecture Philosophy

### Current State (v0.9.x)

PlethApp is designed with **modularity by default**:

**Core modules (`core/`)**: Domain-agnostic signal processing tools
- `filters.py`: Generic signal filtering (can be used for any time series)
- `peaks.py`: Peak detection (adaptable to any signal with peaks)
- `metrics.py`: Breath metrics (breath-specific, but extensible pattern)

**File loaders (`core/io/`)**: Standardized output format
- All loaders return same data structure
- Easy to add new formats
- Prepared for extraction into `neurodata-io` package

**UI components**: Don't depend on breath-specific logic
- `core/plotting.py`: Generic matplotlib/PyQt6 integration
- `core/navigation.py`: Time-based navigation (works for any time series)

**State management (`core/state.py`)**: Separates data from processing
- Pure data container
- No processing logic in state class
- Easy to serialize/deserialize

**This 90% modular architecture makes PlethApp ready for**:
1. Integration of ML classifiers (no major refactoring needed)
2. Extraction into standalone packages (already organized by function)
3. Reuse in photometry/ephys applications (generic interfaces)

### Future State (v2.0+)

After v1.0 publication, PlethApp will become the **reference implementation** of:

**neurodata-io**: Universal neurophysiology file loader
- ABF, SMRX, EDF, CSV, SpikeGLX, TDT formats
- Consistent API across all formats
- Used by PlethApp, photometry apps, ephys tools

**neuroviz**: Scientific plotting and UI components
- PyQt6 matplotlib integration
- Time-series navigation widgets
- Dark theme templates

**neurocore**: Signal processing and ML framework
- Filtering, peak detection, feature extraction
- ML breath classification
- Generic enough for multiple signal types

**Benefits**:
- Other researchers can use these packages in their tools
- Bug fixes propagate across all dependent applications
- PlethApp gets cited when these packages are used
- Demonstrates infrastructure design skills

---

## Design Patterns

### Model-View Separation
- **Model**: `AppState` holds all data
- **View**: PyQt6 UI displays data and receives user input
- **Controller**: Main window methods coordinate between model and view

### Observer Pattern (Implicit)
- UI elements trigger `update_and_redraw()` on parameter changes
- Processing functions observe state via `app_state` reference
- Plot updates observe processing results

### Cache-Aside Pattern
- Processing results cached with parameter-based keys
- Cache checked before expensive recomputation
- Cache invalidated when parameters change

### Strategy Pattern
- Multiple peak detection strategies (primary, fallback, emergency)
- Multiple file loaders (ABF, SMRX, EDF) with common interface
- Multiple metrics calculation methods (standard, robust)

---

## Future Architecture Improvements

### Planned Enhancements (Post v1.0)

1. **Plugin Architecture**: Allow custom algorithms and file formats
2. **Event Bus**: Replace implicit observer pattern with explicit event system
3. **Command Pattern**: Enable undo/redo for all operations
4. **Dependency Injection**: Improve testability and modularity
5. **Type Hints**: Full type coverage for better IDE support and static analysis

---

## Related Documentation
- **ALGORITHMS.md**: Algorithmic details and performance characteristics
- **PERFORMANCE_OPTIMIZATION_PLAN.md**: Detailed performance improvement roadmap
- **MULTI_APP_STRATEGY.md**: Long-term architecture vision (multi-app ecosystem)
- **BUILD_INSTRUCTIONS.md**: Detailed build process and troubleshooting
