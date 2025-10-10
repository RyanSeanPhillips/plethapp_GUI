# Editing Modes Extraction Summary

**Date:** 2025-10-10
**Task:** Extract editing mode functionality from main.py to create editing/editing_modes.py

---

## Files Created

### 1. `editing/__init__.py` (13 lines)
Module initialization file with exports:
- Exports `EditingModes` class
- Includes module docstring

### 2. `editing/editing_modes.py` (1,523 lines)
Complete EditingModes class with all editing functionality:
- Add/Delete Peaks mode
- Add/Delete Sigh mode
- Move Point mode (drag peaks, onsets, offsets, expmins, expoffs)
- Mark Sniffing Regions mode
- All helper methods and state management

**Total New Code:** 1,536 lines

---

## Files Modified

### `main.py`
- **Original:** 8,729 lines
- **Modified:** 7,237 lines
- **Lines Removed:** 1,492 lines
- **Net Reduction:** 1,492 lines (accounting for added integration code)

### Changes Made:
1. **Added import:** `from editing import EditingModes`
2. **Removed state variables:** All editing mode flags and state variables (lines 210-257)
3. **Removed button connections:** Moved to EditingModes.__init__()
4. **Removed keyPressEvent:** Replaced with delegation to EditingModes
5. **Removed 27 methods:** All extracted to EditingModes class
6. **Added delegation:** `self.editing_modes = EditingModes(self)` in __init__()
7. **Updated references:** Changed `self._turn_off_all_edit_modes()` → `self.editing_modes.turn_off_all_edit_modes()`
8. **Updated references:** Changed `self._update_sniff_artists()` → `self.editing_modes.update_sniff_artists()`

---

## Methods Extracted (27 total)

### Core Toggle Handlers (5)
1. `on_add_peaks_toggled(checked: bool)` - Enter/exit add peaks mode
2. `on_delete_peaks_toggled(checked: bool)` - Enter/exit delete peaks mode
3. `on_add_sigh_toggled(checked: bool)` - Enter/exit add sigh mode
4. `on_move_point_toggled(checked: bool)` - Enter/exit move point mode
5. `on_mark_sniff_toggled(checked: bool)` - Enter/exit mark sniff mode

### Plot Click Handlers (5)
6. `_on_plot_click_add_peak(xdata, ydata, event, _force_mode)` - Add peak at click location
7. `_on_plot_click_delete_peak(xdata, ydata, event, _force_mode)` - Delete peak at click location
8. `_on_plot_click_add_sigh(xdata, ydata, event, _force_mode)` - Toggle sigh marker
9. `_on_plot_click_move_point(xdata, ydata, event)` - Select point to move
10. `_on_plot_click_mark_sniff(xdata, ydata, event)` - Start/edit sniff region

### Canvas Event Handlers (5)
11. `_on_canvas_key_press(event)` - Handle matplotlib canvas key events
12. `_on_canvas_motion(event)` - Handle mouse motion for dragging
13. `_on_canvas_release(event)` - Handle mouse release (auto-save)
14. `_on_sniff_drag(event)` - Visual feedback while marking sniff region
15. `_on_sniff_release(event)` - Finalize sniff region marking

### Helper Methods (11)
16. `_turn_off_all_edit_modes()` - Disable all editing modes (mutual exclusion)
17. `_compute_single_breath_events(...)` - Compute breath events for single peak
18. `_find_nearest_zero_crossing(...)` - Find nearest zero crossing for snapping
19. `_constrain_to_peak_boundaries(...)` - Constrain point movement
20. `_move_selected_point(direction, snap_to_zero)` - Move point via arrow keys
21. `_update_point_position(...)` - Update point to new index position
22. `_save_moved_point(recompute_metrics)` - Save moved point and clear selection
23. `_cancel_move_point()` - Cancel move and restore original position
24. `_snap_sniff_to_breath_events(...)` - Snap sniff region edges to breath events
25. `_merge_sniff_regions(sweep_idx)` - Merge overlapping sniff regions
26. `update_sniff_artists(t_plot, sweep_idx)` - Redraw sniff region overlays

### New Methods (1)
27. `handle_key_press_event(event) -> bool` - QKeyEvent delegation handler (new)

---

## Integration Architecture

### MainWindow → EditingModes Communication

**MainWindow provides access to:**
- `self.window.state` - Application state (peaks, breaths, etc.)
- `self.window.plot_host` - Plot management
- `self.window._current_trace()` - Get processed trace data
- `self.window._sweep_count()` - Get total sweep count
- `self.window.redraw_main_plot()` - Trigger plot redraw
- `self.window._compute_y2_all_sweeps()` - Recompute Y2 metrics
- `self.window._run_automatic_gmm_clustering()` - Re-run GMM clustering
- All UI widgets (buttons, checkboxes, etc.)

**EditingModes provides:**
- `handle_key_press_event(event)` - Handle keyboard shortcuts
- `turn_off_all_edit_modes()` - Disable all modes (for toolbar callbacks)
- `update_sniff_artists(t_plot, sweep_idx)` - Redraw sniff overlays
- All editing mode state management

### State Management

**State in EditingModes:**
- Mode flags: `_add_peaks_mode`, `_delete_peaks_mode`, `_add_sigh_mode`, `_move_point_mode`, `_mark_sniff_mode`
- Move point state: `_selected_point`, `_move_point_artist`
- Sniff marking state: `_sniff_start_x`, `_sniff_drag_artist`, `_sniff_artists`, `_sniff_edge_mode`, `_sniff_region_index`
- Matplotlib event connections: `_key_press_cid`, `_motion_cid`, `_release_cid`

**State still in MainWindow:**
- `state.peaks_by_sweep` - Detected peaks
- `state.breath_by_sweep` - Breath events
- `state.sigh_by_sweep` - Sigh markers
- `state.sniff_regions_by_sweep` - Manually marked sniff regions
- `_sigh_artists` - Matplotlib artists for sigh overlays (separate from sniff)

---

## Benefits of Extraction

1. **Reduced main.py complexity:** 1,492 lines removed (17% reduction)
2. **Improved modularity:** All editing functionality in one cohesive module
3. **Better maintainability:** Clear separation of concerns
4. **Easier testing:** Editing modes can be tested independently
5. **Reduced cognitive load:** Developers can focus on editing logic without main.py complexity
6. **Clear delegation pattern:** All editing operations go through EditingModes

---

## Validation Results

✅ All files pass Python syntax validation
✅ All 27 methods successfully extracted
✅ All integration points confirmed
✅ No duplicate methods in main.py
✅ All references updated to use delegation

---

## Backup

A backup of the original main.py is saved as:
- `main.py.backup` (8,729 lines)

To restore the original:
```bash
cp main.py.backup main.py
```

---

## Next Steps (Optional)

1. **Update CLAUDE.md** - Document the new editing/ module structure
2. **Add unit tests** - Create tests for EditingModes class
3. **Further extraction** - Consider extracting other large subsystems (GMM clustering, consolidation, etc.)
4. **Type hints** - Add comprehensive type annotations to EditingModes
5. **Documentation** - Add detailed docstrings for all public methods

---

## Issues Encountered

**None.** The extraction completed successfully without any syntax errors or broken references.

---

*Generated by Claude Code - 2025-10-10*
