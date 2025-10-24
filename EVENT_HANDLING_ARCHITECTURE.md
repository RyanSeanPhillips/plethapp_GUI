# Event Handling Architecture - PlethApp

**Last Updated**: 2025-10-13
**Status**: Complete and tested

---

## Overview

PlethApp uses a sophisticated event handling system to manage user interactions with matplotlib plots embedded in PyQt6. This document explains how mouse clicks, editing modes, and plot redraws interact, with special attention to the challenges of Y2 axis (twinx) overlays.

---

## Architecture Summary

The event handling system operates across four layers:

1. **Qt Layer**: Raw OS events → `FigureCanvasQTAgg.mousePressEvent()`
2. **Matplotlib Canvas**: Event dispatch via `mpl_connect()` callbacks
3. **PlotHost**: Central routing and callback management
4. **Editing Modes**: Mode-specific handlers for user interactions

All layers work together to ensure clicks reach the appropriate handlers even after figure redraws and Y2 axis changes.

---

## Critical Design Decisions

### 1. Handler Registration Order

**Key Principle**: Register our handlers BEFORE creating the matplotlib toolbar.

**Why**: Matplotlib calls handlers in registration order. By registering first, we ensure our click handler sees every click before toolbar modes (zoom/pan) can intercept them.

```python
# core/plotting.py, lines 28-31
self._cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll)
self._cid_button = self.canvas.mpl_connect('button_press_event', self._on_button)

# Toolbar created AFTER our handlers
self.toolbar = NavigationToolbar(self.canvas, self)
```

### 2. Y2 Axis Mouse Event Blocking

**Key Principle**: Disable mouse events on Y2 axis immediately after creation.

**Why**: `ax.twinx()` creates an overlay axis that intercepts mouse clicks. Disabling its mouse events allows clicks to pass through to the main axis.

```python
# core/plotting.py, lines 474-477
ax_y2.set_navigate(False)  # Disable navigation (zoom/pan)
ax_y2.patch.set_visible(False)  # Make background transparent
```

### 3. Callback Restoration After Redraws

**Key Principle**: Restore editing mode callbacks after every `fig.clear()`.

**Why**: `fig.clear()` destroys and recreates axes. Canvas-level matplotlib connections survive, but our application-level callback registrations do not.

**Implementation**:
```python
# plotting/plot_manager.py, lines 37-40
if self.window.single_panel_mode:
    self._draw_single_panel_plot()
    # Restore editing mode connections after redraw
    self._restore_editing_mode_connections()
```

### 4. Qt-Level Failsafe

**Key Principle**: Override Qt's `mousePressEvent()` to provide backup click handling.

**Why**: Provides robustness against unforeseen edge cases where matplotlib's callback system fails.

```python
# core/plotting.py, lines 33-37
self._original_mouse_press = self.canvas.mousePressEvent
self.canvas.mousePressEvent = self._qt_mouse_press_override
```

**Note**: This failsafe is rarely triggered but ensures editing modes always work.

---

## Event Flow

```
User Click
    ↓
Qt mousePressEvent() [failsafe override intercepts]
    ↓
Matplotlib canvas dispatch (in registration order)
    ├─ PlotHost._on_button() [OUR handler - registered first]
    ├─ NavigationToolbar handlers
    └─ Other handlers
    ↓
PlotHost._on_button() checks for registered callback
    ↓
If callback exists: Forward to EditingModes handler
    ↓
EditingModes._on_plot_click_*() processes the click
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Clearing Callback at Wrong Time

**Problem**: Code calls `clear_click_callback()` while a mode is active.

**Example Bug** (fixed in this session):
```python
def on_y2_metric_changed(self):
    self.redraw_main_plot()  # Restores callback
    # ORPHANED CODE (wrong indentation)
    self.plot_host.clear_click_callback()  # BUG: Clears it immediately!
```

**Solution**: Only clear callbacks when deactivating modes, not after redraws.

### Pitfall 2: Not Restoring After Redraw

**Problem**: `fig.clear()` invalidates callback, but no restoration happens.

**Solution**: Call `_restore_editing_mode_connections()` after every `fig.clear()`.

### Pitfall 3: Y2 Axis Blocks Clicks

**Problem**: Y2 axis created with `twinx()` intercepts mouse events.

**Solution**: Disable Y2 mouse events with `set_navigate(False)` and `patch.set_visible(False)`.

---

## Testing Checklist

All scenarios have been tested and work correctly:

- [x] Mark Sniff works with Y2 axis displayed
- [x] Mark Sniff works when changing Y2 metrics (no toggle needed)
- [x] All 5 editing modes work with Y2 axis
- [x] Editing modes work after sweep navigation
- [x] Editing modes work after filter changes
- [x] Mode buttons correctly toggle editing modes on/off

---

## Code References

### Key Files
- `core/plotting.py` - PlotHost class, event handling core
- `plotting/plot_manager.py` - Plot orchestration, callback restoration
- `editing/editing_modes.py` - Editing mode implementations

### Key Line Numbers (as of 2025-10-13)

**core/plotting.py**:
- Lines 28-31: Event handler registration (before toolbar)
- Lines 33-37: Qt failsafe override
- Lines 159-174: `_on_button()` main click handler
- Lines 200-240: Qt failsafe implementation
- Lines 474-477: Y2 mouse event blocking
- Lines 701-706: Callback get/set/clear methods

**plotting/plot_manager.py**:
- Lines 37-40: Restoration call
- Lines 433-493: `_restore_editing_mode_connections()` method

**main.py**:
- Lines 1699-1714: `on_y2_metric_changed()` method (orphaned code removed)

---

## Conclusion

The event handling system is now robust and reliable. Editing modes work correctly in all scenarios, including:
- With and without Y2 axis
- After Y2 metric changes
- After sweep navigation and filter changes
- With toolbar modes active/inactive

The key to this robustness is the combination of:
1. Handler priority (register before toolbar)
2. Y2 mouse event blocking
3. Callback restoration after redraws
4. Qt-level failsafe for edge cases

---

**Document Version**: 1.0
**Author**: Claude (Anthropic)
**Last Tested**: PlethApp v1.0.5 (2025-10-13)
