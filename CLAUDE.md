# PlethApp - Breath Analysis Application

## Overview
PlethApp is a PyQt6-based desktop application for advanced respiratory signal analysis. It provides comprehensive tools for breath pattern detection, eupnea/apnea identification, and breathing regularity assessment.

## Project Structure
```
plethapp/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── pleth_app.spec            # PyInstaller configuration
├── build_executable.bat      # Windows build script
├── build_executable.py       # Cross-platform build script
├── version_info.py           # Version metadata generator
├── run_debug.py              # Debug launcher for testing
├── BUILD_INSTRUCTIONS.md     # Detailed build documentation
├── core/                     # Core application modules
│   ├── state.py             # Application state management
│   ├── abf_io.py            # ABF file I/O operations
│   ├── filters.py           # Signal filtering functions
│   ├── plotting.py          # Matplotlib integration and plot management
│   ├── stim.py              # Stimulus detection algorithms
│   ├── peaks.py             # Peak detection and breath onset/offset analysis
│   ├── metrics.py           # Breathing metrics and pattern analysis
│   ├── navigation.py        # Data navigation utilities
│   ├── editing.py           # Manual peak editing tools
│   └── export.py            # Data export functionality
├── ui/                      # PyQt6 UI definition files
│   └── pleth_app_layout_02.ui # Main application UI layout
├── images/                  # Application icons and assets
├── assets/                  # Additional application assets
└── examples/               # Sample data files
```

## Key Features
- **Advanced Peak Detection**: Sophisticated algorithms for detecting inspiratory peaks, expiratory minima, and breath onsets/offsets
- **Enhanced Expiratory Offset Detection**: Uses both signal zero crossings and derivative analysis with amplitude constraints
- **Eupnea Detection**: Identifies regions of normal, regular breathing patterns
- **Apnea Detection**: Detects breathing gaps longer than configurable thresholds (default: 0.5 seconds)
- **Breathing Regularity Score**: RMSSD-based assessment of breathing pattern variability
- **Real-time Visual Overlays**: Automatic green/red line overlays for eupnea and apnea regions
- **Interactive Data Navigation**: Window-based and sweep-based data exploration
- **Manual Peak Editing**: Add/delete peaks and annotate sighs
- **Data Export**: CSV export of analyzed breathing metrics
- **Multi-format Support**: ABF file format support with extensible I/O architecture

## Core Algorithms

### Peak Detection (core/peaks.py)
- **find_peaks()**: Main peak detection using scipy with configurable threshold, prominence, and distance parameters
- **compute_breath_events()**: **ENHANCED** robust breath event detection with multiple fallback strategies:
  - **Robust Onset Detection**: Zero crossing in y signal → dy/dt crossing → fixed fraction fallback → boundary-based fallback
  - **Robust Offset Detection**: Similar multi-level fallback approach ensuring valid breath boundaries
  - **Enhanced Expiratory Detection**: Derivative zero crossing → actual minimum fallback → midpoint estimation
  - **Expiratory Offset Detection**: **ENHANCED** dual-candidate method:
    - Finds both signal zero crossing AND derivative zero crossing with 50% amplitude threshold
    - Selects whichever occurs EARLIER (more physiologically accurate timing)
    - Robust fallbacks for edge cases
  - **Edge Effect Protection**: Special handling for peaks near trace boundaries
  - **Consistent Output**: Always returns arrays of appropriate lengths with bounds checking

### Breathing Pattern Analysis (core/metrics.py + core/robust_metrics.py)
- **detect_eupnic_regions()**: Identifies normal breathing regions using frequency (<5Hz) and duration (≥2s) criteria
- **detect_apneas()**: Detects breathing gaps based on inter-breath intervals
- **compute_regularity_score()**: RMSSD calculation with performance-optimized decimated sampling
- **compute_breath_metrics()**: Comprehensive breath-by-breath analysis including:
  - Breath duration and frequency
  - Peak amplitudes and timing
  - Inter-breath intervals
  - Respiratory variability measures

#### **NEW: Robust Metrics Framework (core/robust_metrics.py)**
Advanced error-resistant metrics calculation system:
- **Graceful Degradation**: Continues processing when individual breath cycles fail
- **Multiple Fallback Strategies**: Each metric has 3-4 fallback methods
- **Bounds Protection**: Comprehensive array bounds checking prevents crashes
- **Misaligned Data Handling**: Works with arrays of different lengths
- **NaN Isolation**: Failed calculations don't cascade to other cycles
- **Consistent Output**: Always produces valid arrays regardless of input quality

**Key Robust Metrics:**
- `robust_compute_if()`: Instantaneous frequency with onset→onset, peak→peak, and estimation fallbacks
- `robust_compute_amp_insp()`: Inspiratory amplitude with baseline detection strategies
- `robust_compute_ti()`: Inspiratory timing with direct, estimated, and default calculations

### Signal Processing (core/filters.py)
- Low-pass and high-pass Butterworth filtering
- Mean subtraction with configurable time windows
- Signal inversion capabilities
- Real-time filter parameter adjustment

## Development Commands

### Running in Development Mode
```bash
python run_debug.py
```
This launches the application with import checking and debug output.

### Building Executable
```bash
# Method 1: Using batch script (Windows)
build_executable.bat

# Method 2: Using Python script (cross-platform)
python build_executable.py

# Method 3: Manual PyInstaller
python version_info.py
pyinstaller --clean pleth_app.spec
```

### Testing
- Test the source application using `python run_debug.py`
- Test the built executable on a clean machine without Python installed
- Verify all features work: file loading, peak detection, filtering, export

## Lint & Typecheck Commands
Currently no formal linting setup. Consider adding:
```bash
# Future recommendations:
# pip install flake8 mypy
# flake8 core/ main.py
# mypy core/ main.py
```

## Architecture Notes

### State Management
The application uses a centralized state system (`core.state.AppState`) that manages:
- Current data and metadata
- Peak detection results
- Filter parameters
- Navigation state
- User annotations

### UI Architecture
- **Main Window**: `pleth_app_layout_02.ui` - Grid-based layout with left-aligned controls
- **Plotting Integration**: Custom `PlotHost` class managing matplotlib figures within PyQt6
- **Responsive Design**: MainPlot stretches with window, controls remain left-justified
- **Dark Theme**: Custom CSS styling optimized for scientific data visualization

#### **IMPORTANT: Qt Designer Left-Alignment Fix**
Qt Designer tends to remove alignment attributes when saving `.ui` files. To maintain left-justified controls:

1. **Main Grid Layout** (`gridLayout_3`):
   - Row 0: Add `alignment="Qt::AlignLeft|Qt::AlignTop"` to item containing `verticalLayout_8`
   - Row 1: **NO alignment** on MainPlot (must expand to fill space)
   - Row 2: Add `alignment="Qt::AlignLeft|Qt::AlignTop"` to item containing `horizontalLayout_12`

2. **Top Controls Container** (`verticalLayout_8`):
   - Add layout property: `<property name="alignment"><set>Qt::AlignLeft|Qt::AlignTop</set></property>`

3. **Horizontal Layouts Inside verticalLayout_8**:
   - `horizontalLayout_7` (Browse/File Selection)
   - `horizontalLayout_8` (Channel Selection)
   - `horizontalLayout_9` (Filters)
   - `horizontalLayout_10` (Peak Detection)
   - Each needs: `<property name="alignment"><set>Qt::AlignLeft|Qt::AlignVCenter</set></property>`

**Example XML snippet for horizontal layout:**
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

**Note**: After editing in Qt Designer, these alignment properties may be removed. Re-apply them using the Edit tool or a text editor.

### Performance Optimizations
- **Decimated Sampling**: Metrics calculations use reduced sample rates (~0.1s intervals) for performance
- **Lazy Loading**: Data loaded on-demand to minimize memory usage
- **Efficient Plotting**: Optimized matplotlib backend integration

### Build System
- **PyInstaller Configuration**: Comprehensive `.spec` file with proper dependency handling
- **Directory Distribution**: Fast startup times (~6 seconds) vs single-file distribution
- **Icon Integration**: Custom application icons for Windows executable
- **Dependency Management**: Careful exclusion of conflicting Qt bindings and unused modules

## Deployment Notes
- **Target Platform**: Windows 10/11 (primary), extensible to other platforms
- **Distribution Size**: ~200-400MB due to scientific libraries
- **Startup Time**: ~6 seconds (directory distribution)
- **Runtime Dependencies**: Self-contained executable with no Python installation required

## Robustness Features

### **Peak Detection Robustness**
- **Multi-level Fallbacks**: Each breath event type uses 3-4 detection strategies
- **Edge Effect Handling**: Special processing for peaks near trace boundaries
- **Noisy Signal Tolerance**: Derivative filtering and amplitude constraints reduce false detections
- **Emergency Fallbacks**: Creates reasonable estimates when all primary methods fail

### **Metrics Calculation Robustness**
- **Isolated Failures**: One bad breath cycle doesn't break entire sweep analysis
- **Array Length Mismatches**: Handles cases where onsets, offsets, and peaks arrays have different lengths
- **Bounds Checking**: All array accesses are validated to prevent index errors
- **Graceful Degradation**: Returns partial results when possible rather than complete failure

### **Usage: Enabling Robust Mode**
To enable the enhanced robust metrics (optional):

1. **Uncomment the integration code** in `core/metrics.py` (lines 966-971):
   ```python
   try:
       from core.robust_metrics import enhance_metrics_with_robust_fallbacks
       METRICS = enhance_metrics_with_robust_fallbacks(METRICS)
       print("Enhanced metrics with robust fallbacks enabled.")
   except ImportError:
       print("Robust metrics module not available. Using standard metrics.")
   ```

2. **Restart the application** - the robust metrics will be used automatically for all calculations

## Future Enhancements
- Code signing for professional distribution
- Automated testing framework with edge case coverage
- Additional file format support (EDF, WFDB)
- Real-time data acquisition capabilities
- Advanced statistical analysis modules
- Plugin architecture for custom algorithms
- Automated quality assessment and signal validation

## Contributing
When modifying the codebase:
1. Test changes using `run_debug.py` first
2. Follow existing code conventions (see main.py imports and core module structure)
3. Update this CLAUDE.md file for significant architectural changes
4. Rebuild executable and test on clean machine before distribution
5. Update version info in `version_info.py` for releases

## Contact & Support
For technical issues or enhancement requests, refer to the project documentation or create an issue in the repository.