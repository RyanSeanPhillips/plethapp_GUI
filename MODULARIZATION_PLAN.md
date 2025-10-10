# PlethApp Code Modularization Plan

## Current State Analysis

**Problem**: `main.py` has grown to ~10,000+ lines with multiple responsibilities mixed together:
- Main window UI management
- File I/O and data loading
- Signal processing and filtering
- Peak detection and breath analysis
- Multiple dialog classes (GMM, spectral analysis, outlier detection, etc.)
- Plotting and visualization
- Data export
- Manual editing modes
- Navigation logic
- Caching systems

**Goal**: Break down `main.py` into focused, maintainable modules while preserving all functionality.

---

## Phase 1: Extract Dialog Classes (Low Risk, High Impact)

### Priority: HIGH (Start here)
### Estimated Time: 4-6 hours
### Risk Level: LOW (dialogs are self-contained)

**What to do**:
1. Create new directory: `dialogs/`
2. Move each dialog class to its own file:
   - `dialogs/gmm_clustering_dialog.py` - GMM/Eupnea/Sniffing detection (~800 lines)
   - `dialogs/spectral_analysis_dialog.py` - Power spectrum, wavelet, notch filter (~400 lines)
   - `dialogs/outlier_threshold_dialog.py` - Outlier metric selection (~200 lines)
   - `dialogs/consolidated_export_dialog.py` - Multi-file export (~300 lines)
   - `dialogs/sweep_omit_dialog.py` - Sweep curation (~150 lines)

**Benefits**:
- Removes ~2000 lines from `main.py`
- Each dialog is independently testable
- Easier to add new dialogs in the future
- Clear separation of concerns

**Implementation Steps**:
```python
# Example for GMM dialog:
# File: dialogs/gmm_clustering_dialog.py

from PyQt6.QtWidgets import QDialog, QVBoxLayout, ...
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np
from sklearn.mixture import GaussianMixture

class GMMClusteringDialog(QDialog):
    """Dialog for GMM-based eupnea/sniffing detection."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Eupnea / Sniffing Detection")
        # ... rest of dialog code ...

# Then in main.py:
from dialogs.gmm_clustering_dialog import GMMClusteringDialog

class MainWindow(QMainWindow):
    def on_gmm_button_clicked(self):
        dialog = GMMClusteringDialog(self)
        dialog.exec()
```

**Testing Strategy**:
- Move one dialog at a time
- Test dialog functionality after each move
- Ensure main window signals/slots still work
- Check that dialog can access main window state properly

---

## Phase 2: Extract Editing Modes (Medium Risk, Medium Impact)

### Priority: MEDIUM
### Estimated Time: 3-4 hours
### Risk Level: MEDIUM (tightly coupled to canvas events)

**What to do**:
1. Create `editing/` directory
2. Move editing mode logic to separate modules:
   - `editing/peak_editor.py` - Add/delete peaks, manual editing
   - `editing/sigh_editor.py` - Sigh annotation mode
   - `editing/move_point_editor.py` - Drag-to-reposition peaks
   - `editing/sniff_region_editor.py` - Manual sniffing region drawing
   - `editing/base_editor.py` - Base class for all editing modes

**Design Pattern**: State pattern for editing modes
```python
# File: editing/base_editor.py
class BaseEditor:
    """Base class for all editing modes."""
    def __init__(self, main_window):
        self.main_window = main_window

    def on_click(self, event): pass
    def on_motion(self, event): pass
    def on_release(self, event): pass
    def on_key_press(self, event): pass
    def activate(self): pass
    def deactivate(self): pass

# File: editing/peak_editor.py
class PeakEditor(BaseEditor):
    """Handles add/delete peak operations."""
    # ... implementation ...

# In main.py:
from editing.peak_editor import PeakEditor
from editing.sigh_editor import SighEditor

class MainWindow(QMainWindow):
    def __init__(self):
        # ...
        self.peak_editor = PeakEditor(self)
        self.sigh_editor = SighEditor(self)
        self.active_editor = None

    def _on_canvas_click(self, event):
        if self.active_editor:
            self.active_editor.on_click(event)
```

**Benefits**:
- Clear separation of editing concerns
- Easy to add new editing modes
- Testable in isolation
- Reduces main window complexity

---

## Phase 3: Extract Navigation Logic (Low Risk, Medium Impact)

### Priority: MEDIUM
### Estimated Time: 2-3 hours
### Risk Level: LOW (well-encapsulated functionality)

**What to do**:
1. Create `navigation/` directory
2. Move navigation code:
   - `navigation/sweep_navigator.py` - Sweep-based navigation
   - `navigation/window_navigator.py` - Time-window navigation
   - `navigation/navigator_manager.py` - Toggle between modes

**Current navigation code** (~300 lines):
- `on_prev_sweep()`, `on_next_sweep()`
- `on_window_left()`, `on_window_right()`
- `_toggle_navigation_mode()`
- Window size calculations

**New structure**:
```python
# File: navigation/navigator_manager.py
class NavigatorManager:
    """Manages navigation mode switching."""
    def __init__(self, main_window):
        self.main_window = main_window
        self.sweep_nav = SweepNavigator(main_window)
        self.window_nav = WindowNavigator(main_window)
        self.current_nav = self.sweep_nav

    def toggle_mode(self):
        if self.current_nav == self.sweep_nav:
            self.current_nav = self.window_nav
        else:
            self.current_nav = self.sweep_nav
        self.main_window.redraw_main_plot()

    def next(self):
        self.current_nav.next()

    def prev(self):
        self.current_nav.prev()
```

---

## Phase 4: Extract Plotting Logic (Medium Risk, High Impact)

### Priority: HIGH
### Estimated Time: 5-7 hours
### Risk Level: MEDIUM (core visualization logic)

**What to do**:
1. Expand `core/plotting.py` with specialized plotters:
   - `plotting/breath_plotter.py` - Main trace + peak markers
   - `plotting/region_plotter.py` - Eupnea/apnea/outlier overlays
   - `plotting/y2_plotter.py` - Secondary axis metrics
   - `plotting/threshold_plotter.py` - Threshold lines
   - `plotting/stim_plotter.py` - Stimulus markers

**Current plotting code** (~800 lines in `main.py`):
- `redraw_main_plot()` - massive method with many responsibilities
- `_refresh_threshold_lines()`
- `_draw_region_overlays()`
- Y2 axis management
- Stim marker drawing

**New structure**:
```python
# File: plotting/plot_manager.py
class PlotManager:
    """Orchestrates all plotting operations."""
    def __init__(self, plot_host, state):
        self.host = plot_host
        self.state = state
        self.breath_plotter = BreathPlotter(plot_host)
        self.region_plotter = RegionPlotter(plot_host)
        self.y2_plotter = Y2Plotter(plot_host)
        self.stim_plotter = StimPlotter(plot_host)
        self.threshold_plotter = ThresholdPlotter(plot_host)

    def redraw(self, sweep_idx, trace_data, breath_data, region_data):
        """Main redraw orchestration."""
        self.host.clear_main()
        self.breath_plotter.plot_trace(trace_data, breath_data)
        self.region_plotter.plot_regions(region_data)
        self.stim_plotter.plot_stim_markers(self.state.stim_onsets_by_sweep)
        self.threshold_plotter.plot_threshold(self.state.threshold_value)
        if self.state.y2_metric_key:
            self.y2_plotter.plot_metric(sweep_idx)
        self.host.canvas.draw_idle()

# In main.py:
from plotting.plot_manager import PlotManager

class MainWindow(QMainWindow):
    def __init__(self):
        # ...
        self.plot_manager = PlotManager(self.plot_host, self.state)

    def redraw_main_plot(self):
        # Much simpler now!
        s = self.state.sweep_idx
        trace = self._current_trace()
        breath_data = self.state.breath_by_sweep.get(s)
        region_data = self._get_region_data(s)
        self.plot_manager.redraw(s, trace, breath_data, region_data)
```

**Benefits**:
- Reduces `redraw_main_plot()` from ~300 lines to ~20 lines
- Each plotter is independently testable
- Easy to add new plot overlays
- Clear responsibilities

---

## Phase 5: Extract Export Logic (Low Risk, Medium Impact)

### Priority: MEDIUM
### Estimated Time: 2-3 hours
### Risk Level: LOW (already partially in core/export.py)

**What to do**:
1. Move all export functions to `core/export.py`:
   - Move `on_export_breath_metrics_clicked()` logic
   - Move `on_export_events_clicked()` logic
   - Move `on_consolidate_save_data_clicked()` logic
   - Keep only dialog creation in `main.py`

**New structure**:
```python
# File: core/export.py
class BreathExporter:
    """Handles breath-by-breath CSV export."""
    @staticmethod
    def export_breath_metrics(state, output_path):
        # ... logic currently in main.py ...

class EventExporter:
    """Handles event-level CSV export."""
    @staticmethod
    def export_events(state, output_path):
        # ... logic currently in main.py ...

class ConsolidatedExporter:
    """Handles multi-file consolidated export."""
    @staticmethod
    def export_consolidated(file_list, params, output_path):
        # ... logic currently in main.py ...

# In main.py:
from core.export import BreathExporter, EventExporter, ConsolidatedExporter

class MainWindow(QMainWindow):
    def on_export_breath_metrics_clicked(self):
        path, _ = QFileDialog.getSaveFileName(...)
        if path:
            BreathExporter.export_breath_metrics(self.state, path)
            QMessageBox.information(self, "Success", "Exported!")
```

---

## Phase 6: Extract Cache Management (Low Risk, Low Impact)

### Priority: LOW
### Estimated Time: 1-2 hours
### Risk Level: LOW (internal optimization)

**What to do**:
1. Create `caching/trace_cache.py`
2. Move trace processing cache logic to dedicated class

**Current cache code**:
```python
# In main.py:
self._global_trace_cache = {}  # Dict[(chan, sweep, filter_params), processed_trace]

def _current_trace(self):
    key = self._proc_key(chan, sweep)
    if key in self._global_trace_cache:
        return self._global_trace_cache[key]
    # ... process trace ...
    self._global_trace_cache[key] = result
    return result
```

**New structure**:
```python
# File: caching/trace_cache.py
class TraceCache:
    """LRU cache for processed traces."""
    def __init__(self, max_size=100):
        self._cache = {}
        self._max_size = max_size

    def get(self, chan, sweep, filter_params):
        key = self._make_key(chan, sweep, filter_params)
        return self._cache.get(key)

    def set(self, chan, sweep, filter_params, trace):
        key = self._make_key(chan, sweep, filter_params)
        self._cache[key] = trace
        self._evict_if_needed()

    def clear(self):
        self._cache.clear()

# In main.py:
from caching.trace_cache import TraceCache

class MainWindow(QMainWindow):
    def __init__(self):
        self.trace_cache = TraceCache()
```

---

## Phase 7: Refactor MainWindow (High Risk, High Impact)

### Priority: LOW (do this LAST)
### Estimated Time: 8-10 hours
### Risk Level: HIGH (core refactor)

**What to do** (only after all above phases complete):
1. Break `MainWindow` into focused mixins or sub-controllers
2. Possible structure:
   - `ui/main_window.py` - Core window, UI initialization
   - `ui/file_controller.py` - File loading, channel selection
   - `ui/filter_controller.py` - Filter parameter management
   - `ui/peak_controller.py` - Peak detection control
   - `ui/curation_controller.py` - Multi-file curation tab

**This is the riskiest phase** - only do this if:
- All previous phases are complete and tested
- You have comprehensive tests
- You're willing to accept significant debugging time

---

## Implementation Order (Recommended)

### Week 1: Low-Hanging Fruit
1. **Day 1-2**: Phase 1 (Extract Dialogs) - Move GMM and Spectral dialogs
2. **Day 3**: Phase 1 continued - Move remaining dialogs
3. **Day 4**: Phase 3 (Extract Navigation) - Quick win
4. **Day 5**: Phase 6 (Extract Cache) - Another quick win

### Week 2: Core Improvements
5. **Day 1-2**: Phase 5 (Extract Export) - Consolidate export logic
6. **Day 3-5**: Phase 4 (Extract Plotting) - High impact refactor

### Week 3: Advanced (Optional)
7. **Day 1-3**: Phase 2 (Extract Editing) - More complex refactor
8. **Day 4-5**: Phase 7 (Refactor MainWindow) - Only if needed

---

## Testing Strategy for Each Phase

### Before Refactoring:
1. Test all features manually, document expected behavior
2. Create a "golden dataset" test file
3. Export metrics/events as reference outputs

### During Refactoring:
1. Move code incrementally (one dialog/module at a time)
2. Test after each move
3. Use `run_testing.bat` for rapid iteration

### After Each Phase:
1. Verify all features still work
2. Compare exported data to reference outputs
3. Check for performance regressions
4. Test edge cases (empty sweeps, missing channels, etc.)

---

## Git Strategy

### Branch Structure:
- `main` - Stable version (current code)
- `refactor/phase1-dialogs` - First refactor branch
- `refactor/phase2-editing` - Second refactor branch
- etc.

### Commit Strategy:
- Small, atomic commits: "Move GMMClusteringDialog to dialogs/"
- Test after each commit
- Can revert easily if something breaks

### Merge Strategy:
- Only merge phase to `main` when fully tested
- Keep old code in comments initially (delete after confirmed working)
- Tag releases: `v1.0.5-pre-refactor`, `v1.1.0-post-phase1`, etc.

---

## Risks and Mitigation

### Risk 1: Breaking existing functionality
**Mitigation**: Test extensively, use testing mode, keep reference outputs

### Risk 2: Import circular dependencies
**Mitigation**: Use dependency injection, pass main_window as parameter, avoid importing main.py in submodules

### Risk 3: Performance regression
**Mitigation**: Profile before/after, keep caching systems intact

### Risk 4: Loss of state between modules
**Mitigation**: Keep `AppState` in `core/state.py`, pass references, avoid duplicating state

---

## Benefits Summary

After completing Phases 1-6:
- **main.py**: ~10,000 lines → ~4,000 lines (60% reduction)
- **Maintainability**: Much easier to find and fix bugs
- **Testability**: Individual components can be unit tested
- **Collaboration**: Multiple developers can work on different modules
- **Onboarding**: New developers understand code faster
- **Future Features**: Easier to add new dialogs, editors, exporters

---

## Decision: Start or Wait?

### Start Now If:
- You have time for 2-3 weeks of refactoring
- You're planning to add many new features
- Multiple people will work on the code
- You want to publish as open-source package

### Wait Until Later If:
- You need to ship new features urgently
- The current code is "working well enough"
- You're the only developer
- You plan to do a complete rewrite later

---

## Recommendation

**Start with Phase 1 (Dialogs)** - It's:
- Low risk (dialogs are self-contained)
- High impact (removes ~2000 lines immediately)
- Quick (can finish in 1-2 days)
- Easy to test (dialogs are independent)
- Reversible (easy to move back if issues)

After Phase 1, reassess whether to continue based on:
- How much time it took
- How many issues you encountered
- Whether the benefits are worth the effort

Would you like me to start with Phase 1 now, or would you prefer to continue adding features with the current structure?
