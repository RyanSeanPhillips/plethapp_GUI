# Event Handling Architecture Analysis: PlethApp Click Event Flow

**Date:** 2025-10-13
**Issue:** Editing modes (Mark Sniff, Add Peaks, etc.) stop working after Y2 axis changes
**Analysis Target:** Complete matplotlib event connection lifecycle

---

## Executive Summary

This document provides a comprehensive analysis of the click event handling architecture in PlethApp, a PyQt6/Matplotlib application. The analysis reveals a **critical architectural issue**: matplotlib event connections survive canvas redraws but NOT figure clears (`fig.clear()`), leading to broken editing modes when Y2 axis changes trigger plot redraws.

**Root Cause:** The `show_trace_with_spans()` method calls `fig.clear()`, which destroys the axes but NOT the canvas. Canvas-level event connections (`_cid_button`, `_cid_scroll`) remain intact, but the callback registration in PlotHost (`_external_click_cb`) gets orphaned because the restoration logic in PlotManager only runs AFTER Y2 axis addition.

---

## 1. Architecture Overview

### Module Hierarchy

```
MainWindow (main.py)
    |
    +-- PlotHost (core/plotting.py)
    |     |
    |     +-- Figure (matplotlib)
    |     +-- Canvas (FigureCanvasQTAgg)
    |     +-- Toolbar (NavigationToolbar2QT)
    |
    +-- PlotManager (plotting/plot_manager.py)
    |     |
    |     +-- Orchestrates plotting operations
    |     +-- Manages editing mode restoration
    |
    +-- EditingModes (editing/editing_modes.py)
    |     |
    |     +-- Manages interactive editing states
    |     +-- Provides click/drag/motion callbacks
    |
    +-- NavigationManager (core/navigation_manager.py)
          |
          +-- Controls data navigation (sweeps, windows)
```

### Key Responsibilities

| Module | Responsibility | Event Connection Type |
|--------|---------------|----------------------|
| **PlotHost** | Owns Figure/Canvas, provides plotting API | **Persistent canvas connections** (`_cid_button`, `_cid_scroll`) |
| **PlotManager** | Orchestrates plot updates, manages restoration | **No direct connections** (calls PlotHost methods) |
| **EditingModes** | Manages mode states, provides callbacks | **Temporary mode-specific connections** (`_motion_cid`, `_release_cid`, `_key_press_cid`) |
| **MainWindow** | Coordinates all modules, handles UI events | **No matplotlib connections** (uses Qt signals) |

---

## 2. Complete Click Event Flow

### 2.1 Initial Setup (Application Launch)

```
PlotHost.__init__()
├─ Create figure: self.fig = plt.figure()
├─ Create canvas: self.canvas = FigureCanvas(self.fig)
├─ Connect persistent events (CANVAS level):
│   ├─ self._cid_scroll = canvas.mpl_connect('scroll_event', self._on_scroll)
│   └─ self._cid_button = canvas.mpl_connect('button_press_event', self._on_button)
└─ Initialize callback placeholder: self._external_click_cb = None
```

**Key Point:** These connections are made to the **canvas**, not the figure or axes. They survive `fig.clear()`.

### 2.2 User Activates "Mark Sniff" Mode

```
EditingModes.on_mark_sniff_toggled(checked=True)
├─ Set mode flag: self._mark_sniff_mode = True
├─ Turn off toolbar modes: plot_host.turn_off_toolbar_modes()
├─ Register click callback:
│   └─ plot_host.set_click_callback(self._on_plot_click_mark_sniff)
│       └─ Sets: plot_host._external_click_cb = callback
├─ Connect motion/release events (CANVAS level):
│   ├─ self._motion_cid = canvas.mpl_connect('motion_notify_event', self._on_sniff_drag)
│   └─ self._release_cid = canvas.mpl_connect('button_release_event', self._on_sniff_release)
└─ Set cursor: plot_host.setCursor(Qt.CursorShape.CrossCursor)
```

**Connection Summary (Mark Sniff Active):**
- **Persistent:** `_cid_button` (PlotHost) → `_on_button()` → `_external_click_cb()`
- **Persistent:** `_cid_scroll` (PlotHost) → `_on_scroll()`
- **Temporary:** `_motion_cid` (EditingModes) → `_on_sniff_drag()`
- **Temporary:** `_release_cid` (EditingModes) → `_on_sniff_release()`
- **Callback:** `_external_click_cb` → `EditingModes._on_plot_click_mark_sniff()`

### 2.3 Click Event Propagation (Normal Operation)

```
User clicks on plot
    ↓
Matplotlib canvas receives button_press_event
    ↓
Canvas dispatches to all connected callbacks
    ↓
PlotHost._on_button(event)  [via _cid_button]
    ↓
Checks: event.inaxes is not None? event.xdata is not None?
    ↓
Checks: self._external_click_cb is not None?
    ↓
YES → Forwards to callback:
    self._external_click_cb(event.xdata, event.ydata, event)
    ↓
EditingModes._on_plot_click_mark_sniff(xdata, ydata, event)
    ↓
Stores start position: self._sniff_start_x = xdata
    ↓
User drags mouse
    ↓
_on_sniff_drag() receives motion_notify_event [via _motion_cid]
    ↓
Draws preview rectangle: ax.axvspan(...)
    ↓
User releases mouse
    ↓
_on_sniff_release() receives button_release_event [via _release_cid]
    ↓
Creates sniff region in state
    ↓
Calls: self.window.redraw_main_plot()
```

**Critical Observation:** The click callback chain requires:
1. `_cid_button` connection (canvas → PlotHost)
2. `_external_click_cb` registration (PlotHost → EditingModes)
3. Mode-specific connections (EditingModes → canvas)

---

## 3. Event Connection Lifecycle

### 3.1 Connection Types and Storage

| Connection ID | Created In | Stored In | Scope | Lifecycle |
|--------------|-----------|-----------|-------|-----------|
| `_cid_button` | PlotHost.__init__() | PlotHost | **Canvas** | **Persistent** (entire app lifetime) |
| `_cid_scroll` | PlotHost.__init__() | PlotHost | **Canvas** | **Persistent** (entire app lifetime) |
| `_motion_cid` | EditingModes (mode activation) | EditingModes | **Canvas** | **Temporary** (mode duration) |
| `_release_cid` | EditingModes (mode activation) | EditingModes | **Canvas** | **Temporary** (mode duration) |
| `_key_press_cid` | EditingModes (Move Point mode) | EditingModes | **Canvas** | **Temporary** (mode duration) |

### 3.2 Connection Survival Rules (CRITICAL)

```python
# Matplotlib connection behavior:

# 1. Canvas-level connections (mpl_connect to canvas):
canvas.mpl_connect('button_press_event', handler)
# → Survives fig.clear() ✓
# → Survives ax.clear() ✓
# → Survives canvas.draw() ✓
# → Destroyed by canvas destruction only

# 2. Axes-level connections (callbacks.connect to axes):
ax.callbacks.connect('xlim_changed', handler)
# → Destroyed by fig.clear() ✗
# → Destroyed by ax.clear() ✗
# → Survives canvas.draw() ✓

# 3. Callback registration (application-level):
plot_host._external_click_cb = callback_function
# → No automatic survival mechanism
# → Must be manually restored after redraws
```

**Key Insight:** Canvas connections survive `fig.clear()`, but callback REGISTRATIONS (like `_external_click_cb`) do NOT have any automatic restoration mechanism.

### 3.3 Disconnection Points

```python
# Temporary connections are disconnected when:

# 1. Mode is turned off
EditingModes.on_mark_sniff_toggled(checked=False):
    if self._motion_cid is not None:
        canvas.mpl_disconnect(self._motion_cid)
        self._motion_cid = None
    if self._release_cid is not None:
        canvas.mpl_disconnect(self._release_cid)
        self._release_cid = None

# 2. Mode is switched (mutual exclusion)
EditingModes.on_add_peaks_toggled(checked=True):
    # Turns off Mark Sniff mode, which disconnects its events
    if self._mark_sniff_mode:
        self._mark_sniff_mode = False
        # ... disconnect motion/release events

# 3. Toolbar mode is activated
PlotHost.set_toolbar_callback():
    # When toolbar zoom/pan is clicked, callback triggers:
    # → EditingModes.turn_off_all_edit_modes()
    # → All mode connections are disconnected
```

---

## 4. Y2 Axis Addition Flow (THE PROBLEM)

### 4.1 User Selects Y2 Metric (e.g., "IF")

```
Y2Combo.currentTextChanged signal
    ↓
MainWindow.on_y2_combo_change()  [hypothetical - actual handler TBD]
    ↓
Sets: self.state.y2_metric_key = "if"
    ↓
Calls: self._compute_y2_all_sweeps()
    ↓
Computes metric for all sweeps, stores in: state.y2_values_by_sweep
    ↓
Calls: self.redraw_main_plot()
```

### 4.2 Redraw Sequence (WITH Y2 Metric)

```
MainWindow.redraw_main_plot()
    ↓
PlotManager.redraw_main_plot()
    ↓
PlotManager._draw_single_panel_plot()
    ↓
[Step 1] Plot base trace:
    PlotHost.show_trace_with_spans(t, y, spans, title, ylabel)
        ↓
        CRITICAL: fig.clear()  ← DESTROYS ALL AXES
        ↓
        Creates new ax_main: self.ax_main = fig.add_subplot(111)
        ↓
        Clears old references:
            self.ax_y2 = None
            self.line_y2 = None
            self.scatter_peaks = None
            ...
        ↓
        Plots trace: ax_main.plot(t, y, ...)
        ↓
        canvas.draw_idle()
    ↓
[Step 2] Draw peak markers:
    PlotManager._draw_peak_markers(s, t_plot, y)
        → Calls: PlotHost.update_peaks(t_peaks, y_peaks)
    ↓
[Step 3] Draw breath markers:
    PlotManager._draw_breath_markers(s, t_plot, y)
        → Calls: PlotHost.update_breath_markers(...)
    ↓
[Step 4] Update sniff overlays:
    EditingModes.update_sniff_artists(t_plot, s)
        → Draws purple rectangles for marked sniff regions
    ↓
[Step 5] Draw Y2 metric:
    PlotManager._draw_y2_metric(s, t, t_plot)
        ↓
        Calls: PlotHost.add_or_update_y2(t_plot, arr, label, color)
            ↓
            Creates twin axis: self.ax_y2 = ax.twinx()
            ↓
            IMPORTANT: Disables mouse events on Y2 axis:
                ax_y2.set_navigate(False)
                ax_y2.patch.set_visible(False)
            ↓
            Plots Y2 line: ax_y2.plot(t, y2, ...)
            ↓
            canvas.draw_idle()
    ↓
[Step 6] Draw region overlays:
    PlotManager._draw_region_overlays(s, t, y, t_plot)
        → Draws eupnea/apnea/outlier overlays
    ↓
[Step 7] Restore editing mode connections:  ← THIS IS THE FIX ATTEMPT
    PlotManager._restore_editing_mode_connections()
        ↓
        Reconnects _cid_button:
            canvas.mpl_disconnect(plot_host._cid_button)
            plot_host._cid_button = canvas.mpl_connect('button_press_event', plot_host._on_button)
        ↓
        IF Mark Sniff mode is active:
            Re-registers callback:
                plot_host.set_click_callback(editing._on_plot_click_mark_sniff)
            ↓
            Reconnects motion/release:
                editing._motion_cid = canvas.mpl_connect('motion_notify_event', ...)
                editing._release_cid = canvas.mpl_connect('button_release_event', ...)
```

### 4.3 The Failure Scenario (BEFORE FIX)

**Timeline:**

1. **T0:** User activates Mark Sniff mode
   - `_external_click_cb` = `_on_plot_click_mark_sniff` ✓
   - `_motion_cid` connected ✓
   - `_release_cid` connected ✓

2. **T1:** User selects Y2 metric (IF)
   - `redraw_main_plot()` called
   - `show_trace_with_spans()` calls `fig.clear()`
   - **All axes destroyed, but canvas connections survive**

3. **T2:** Y2 axis added
   - `add_or_update_y2()` creates twin axis
   - Y2 axis mouse events disabled (correct)

4. **T3:** User clicks on plot
   - Canvas dispatches to `_cid_button` ✓ (survived `fig.clear()`)
   - `_on_button()` called ✓
   - Checks `_external_click_cb is not None`
   - **PROBLEM:** `_external_click_cb` was never restored!
   - Click is NOT forwarded to `_on_plot_click_mark_sniff()`

5. **T4:** User drags mouse
   - `_motion_cid` receives event ✓ (survived `fig.clear()`)
   - `_on_sniff_drag()` called
   - Tries to draw on `ax_main`
   - **PROBLEM:** `_sniff_start_x` is None (never set because click wasn't handled)
   - Early return, no drag preview shown

**Result:** Mark Sniff mode appears active (button checked, cursor changed), but clicks do nothing.

---

## 5. Current Fix Analysis (PlotManager._restore_editing_mode_connections)

### 5.1 Fix Implementation (plotting/plot_manager.py, lines 433-515)

```python
def _restore_editing_mode_connections(self):
    """Restore matplotlib event connections for active editing modes after redraw."""
    editing = self.window.editing_modes

    # STEP 1: Reconnect main button press handler
    print(f"[editing-debug] Reconnecting main button press event handler")
    if hasattr(self.window.plot_host, '_cid_button') and self.window.plot_host._cid_button is not None:
        try:
            self.window.plot_host.canvas.mpl_disconnect(self.window.plot_host._cid_button)
        except:
            pass
    self.window.plot_host._cid_button = self.window.plot_host.canvas.mpl_connect(
        'button_press_event',
        self.window.plot_host._on_button
    )
    print(f"[editing-debug] Reconnected button press: cid={self.window.plot_host._cid_button}")

    # STEP 2: Restore Mark Sniff mode connections
    if getattr(editing, "_mark_sniff_mode", False):
        print(f"[editing-debug] Mark Sniff mode is active - reconnecting matplotlib events")

        # Re-register click callback (THE KEY RESTORATION)
        self.window.plot_host.set_click_callback(editing._on_plot_click_mark_sniff)
        print(f"[editing-debug] Re-registered click callback: {self.window.plot_host._external_click_cb}")

        # Disconnect old connections
        if editing._motion_cid is not None:
            try:
                self.window.plot_host.canvas.mpl_disconnect(editing._motion_cid)
            except:
                pass
        if editing._release_cid is not None:
            try:
                self.window.plot_host.canvas.mpl_disconnect(editing._release_cid)
            except:
                pass

        # Reconnect with fresh connections
        editing._motion_cid = self.window.plot_host.canvas.mpl_connect('motion_notify_event', editing._on_sniff_drag)
        editing._release_cid = self.window.plot_host.canvas.mpl_connect('button_release_event', editing._on_sniff_release)
        print(f"[editing-debug] Reconnected: motion_cid={editing._motion_cid}, release_cid={editing._release_cid}")

    # STEP 3-5: Similar restoration for Add Peaks, Delete Peaks, Add Sigh, Move Point modes
    # [Similar logic for other modes...]
```

### 5.2 Why the Fix SHOULD Work

**Theory:**

1. **Button connection restoration:** The `_cid_button` is explicitly reconnected, ensuring the canvas → PlotHost path works.

2. **Callback re-registration:** The critical line:
   ```python
   self.window.plot_host.set_click_callback(editing._on_plot_click_mark_sniff)
   ```
   Restores the `_external_click_cb` reference that was lost during redraw.

3. **Motion/release reconnection:** Even though these connections survive `fig.clear()`, they're defensively reconnected to ensure fresh connection IDs.

4. **Called at right time:** The restoration happens AFTER `show_trace_with_spans()` and AFTER `add_or_update_y2()`, so the new axes are already in place.

### 5.3 Why the Fix Might NOT Work (Debug Required)

**Potential Issues:**

1. **Restoration called too early:**
   - If `_restore_editing_mode_connections()` is called BEFORE `_draw_y2_metric()`, the Y2 axis isn't created yet.
   - **CHECK:** Verify call order in `_draw_single_panel_plot()`

2. **Y2 axis event blocking:**
   - The Y2 axis has `set_navigate(False)` and `patch.set_visible(False)`, but might still be intercepting clicks.
   - **CHECK:** Verify `event.inaxes` in `_on_button()` debug output. Is it `ax_main` or `ax_y2`?

3. **Callback overwritten:**
   - Some other code path might be clearing `_external_click_cb` after restoration.
   - **CHECK:** Add debug print in `PlotHost.set_click_callback()` and `clear_click_callback()`

4. **Connection ID stale:**
   - Old `_cid_button` might not be disconnected properly, leading to duplicate handlers.
   - **CHECK:** Verify disconnect succeeds (no exceptions)

5. **Toolbar interference:**
   - If toolbar mode is activated during/after restoration, it clears callbacks.
   - **CHECK:** Add debug to `turn_off_toolbar_modes()`

---

## 6. Other Redraw Scenarios (Comparison)

### 6.1 Sweep Navigation (Working)

```
User clicks Next Sweep button
    ↓
NavigationManager.next_sweep()
    ↓
Updates: self.main.state.sweep_idx += 1
    ↓
Calls: self.main.redraw_main_plot()
    ↓
PlotManager.redraw_main_plot()
    ↓
Calls: show_trace_with_spans()  [fig.clear() happens]
    ↓
Calls: _restore_editing_mode_connections()  ✓
    ↓
Result: Mark Sniff mode WORKS (callback restored)
```

**Why this works:** Same restoration logic runs after every `show_trace_with_spans()` call.

### 6.2 Filter Parameter Change (Working)

```
User adjusts Low-pass filter cutoff
    ↓
Signal: valueChanged
    ↓
MainWindow.update_and_redraw()
    ↓
Calls: self.redraw_main_plot()
    ↓
Same restoration as above ✓
```

**Why this works:** All parameter changes go through `redraw_main_plot()` → restoration happens.

### 6.3 Peak Detection (Working)

```
User clicks "Detect Peaks" button
    ↓
MainWindow.on_detect_peaks()
    ↓
Computes peaks for all sweeps
    ↓
Calls: self.redraw_main_plot()
    ↓
Same restoration as above ✓
```

### 6.4 Y2 Metric Selection (BROKEN - SHOULD BE FIXED)

```
User selects Y2 metric
    ↓
MainWindow [handler TBD]
    ↓
Calls: self._compute_y2_all_sweeps()
    ↓
Calls: self.redraw_main_plot()
    ↓
PlotManager.redraw_main_plot()
    ↓
Calls: show_trace_with_spans()  [fig.clear()]
    ↓
Calls: _draw_y2_metric()  [creates twin axis]
    ↓
Calls: _restore_editing_mode_connections()  ← FIX ADDED HERE
    ↓
Result: SHOULD work now (callback restored after Y2 creation)
```

**Expected outcome:** If the fix is implemented correctly, this should now work.

---

## 7. Root Cause Summary

### The Fundamental Problem

**Matplotlib's event system operates at THREE levels:**

1. **Canvas level:** Direct connections to canvas via `mpl_connect()` (survive `fig.clear()`)
2. **Axes level:** Callbacks registered to axes via `ax.callbacks.connect()` (destroyed by `fig.clear()`)
3. **Application level:** Custom callback registrations like `_external_click_cb` (NO automatic survival)

**PlethApp's architecture mixes these levels:**

- **Persistent plumbing:** `_cid_button` (canvas level) → survives ✓
- **Fragile routing:** `_external_click_cb` (application level) → broken by redraw ✗
- **Temporary connections:** `_motion_cid`, `_release_cid` (canvas level) → survive but useless without routing ✗

### Why Y2 Axis Addition Exposed the Bug

Before Y2 feature:
- Fewer redraws in typical workflow
- Users less likely to change plot structure after activating editing modes

After Y2 feature:
- Users toggle Y2 metrics frequently during analysis
- Each toggle triggers full redraw with `fig.clear()`
- Callback registration lost, connections orphaned

### The "Fix" vs. True Solution

**Current fix (_restore_editing_mode_connections):**
- **Pros:** Defensive, explicit, catches all mode states
- **Cons:** Must be called after EVERY redraw, easy to forget, maintenance burden

**Better architectural solution:**
- Store active mode state centrally
- Make `PlotHost.set_click_callback()` check mode state on EVERY `_on_button()` call
- Eliminate need for manual restoration

**Example (hypothetical):**
```python
# In PlotHost._on_button():
def _on_button(self, event):
    if event.inaxes is None:
        return

    if event.dblclick:
        # ... double-click handling
        return

    # ALWAYS query editing modes for active callback
    if hasattr(self, '_editing_modes'):
        active_callback = self._editing_modes.get_active_callback()
        if active_callback and event.xdata is not None:
            active_callback(event.xdata, event.ydata, event)
            return

    # Fallback to registered callback
    if self._external_click_cb is not None and event.xdata is not None:
        self._external_click_cb(event.xdata, event.ydata, event)
```

This eliminates the need for restoration entirely.

---

## 8. Diagnostic Checklist

### To Verify the Fix Works

Run the application with debug mode and follow these steps:

1. **Activate Mark Sniff Mode**
   - Click "Mark Sniff" button
   - **Expected console output:**
     ```
     [mark-sniff] Mode activated
     _external_click_cb = <function _on_plot_click_mark_sniff>
     _motion_cid = 5
     _release_cid = 6
     ```

2. **Test Click WITHOUT Y2**
   - Click on plot
   - **Expected console output:**
     ```
     [plot-click-debug] Click detected: inaxes=AxesSubplot(...), callback=True
     [plot-click-debug] Forwarding to callback: xdata=1.234
     [mark-sniff] Started new region at x=1.234
     ```

3. **Select Y2 Metric (IF)**
   - Choose "IF" from Y2 dropdown
   - **Expected console output:**
     ```
     [editing-debug] Reconnecting main button press event handler
     [editing-debug] Reconnected button press: cid=7
     [editing-debug] Mark Sniff mode is active - reconnecting matplotlib events
     [editing-debug] Re-registered click callback: <function _on_plot_click_mark_sniff>
     [editing-debug] Reconnected: motion_cid=8, release_cid=9
     ```

4. **Test Click WITH Y2**
   - Click on plot
   - **Expected console output (if working):**
     ```
     [plot-click-debug] Click detected: inaxes=AxesSubplot(...), callback=True
     [plot-click-debug] Forwarding to callback: xdata=2.345
     [mark-sniff] Started new region at x=2.345
     ```
   - **Expected console output (if broken):**
     ```
     [plot-click-debug] Click detected: inaxes=AxesSubplot(...), callback=False
     [plot-click-debug] Not forwarding - callback=False, xdata=2.345
     ```

5. **Test Drag and Release**
   - Click and drag on plot
   - **Expected:** Purple preview rectangle appears during drag
   - Release mouse
   - **Expected:** Sniff region saved, purple overlay rendered

### Key Debug Outputs to Check

| Output | What It Means | Fix Status |
|--------|--------------|-----------|
| `callback=False` after Y2 | `_external_click_cb` is None | **FIX FAILED** |
| `callback=True` but no forwarding | `event.xdata` is None or wrong axes | **CHECK Y2 EVENT BLOCKING** |
| `Reconnected button press: cid=X` | Button connection restored | **FIX RUNNING** |
| `Re-registered click callback` | Callback registration restored | **FIX RUNNING** |
| `[mark-sniff] Started new region` | Click reached Mark Sniff handler | **FIX WORKING** |

---

## 9. Recommended Next Steps

### Immediate Actions

1. **Verify fix is enabled:**
   - Check `PlotManager._draw_single_panel_plot()` calls `_restore_editing_mode_connections()`
   - Verify it's called AFTER `_draw_y2_metric()`

2. **Add comprehensive debug logging:**
   - Instrument `PlotHost._on_button()` to log `event.inaxes`, `_external_click_cb`
   - Log every call to `set_click_callback()` and `clear_click_callback()`
   - Log every fig.clear() occurrence

3. **Test all editing modes:**
   - Mark Sniff
   - Add Peaks
   - Delete Peaks
   - Add Sigh
   - Move Point

4. **Test all Y2 metrics:**
   - IF (Instantaneous Frequency)
   - Sniffing Confidence
   - Eupnea Confidence
   - Any other metrics

### Long-Term Improvements

1. **Eliminate manual restoration:**
   - Implement query-based callback resolution in `_on_button()`
   - Store mode state centrally, not just in button checkboxes

2. **Add event connection audit:**
   - Method to list all active matplotlib connections
   - Warning system when connections are orphaned

3. **Separate canvas from figure lifecycle:**
   - Use `ax.cla()` instead of `fig.clear()` where possible
   - Only clear figures when absolutely necessary

4. **Connection pooling:**
   - Reuse connection IDs instead of disconnect/reconnect
   - Track connection state in a registry

---

## 10. Conclusion

The event handling architecture in PlethApp is fundamentally sound but fragile due to the mixing of matplotlib's connection system with application-level callback registration. The Y2 axis addition exposed this fragility because it triggers full figure redraws (`fig.clear()`) which destroy axes but not canvas connections, leaving callback registrations orphaned.

The implemented fix (`_restore_editing_mode_connections()`) should resolve the immediate issue by explicitly restoring callback registrations after every redraw. However, the underlying architectural issue remains: **manual restoration is required after every redraw**, creating maintenance burden and potential for future bugs.

**Final Verdict:** The fix is theoretically correct and should work IF:
1. It's called at the right time (after Y2 axis creation)
2. No other code clears `_external_click_cb` after restoration
3. The Y2 axis doesn't intercept clicks before they reach `ax_main`

If the fix doesn't work, the most likely culprits are:
- **Call order:** Restoration happens before Y2 axis creation
- **Event interception:** Y2 axis blocks clicks despite `set_navigate(False)`
- **Callback clearing:** Some code path clears the callback after restoration

Follow the diagnostic checklist above to identify the specific failure point.

---

**Document Version:** 1.0
**Author:** Claude (Anthropic)
**Last Updated:** 2025-10-13
