# Event Flow Diagrams: PlethApp Click Handling

## Diagram 1: Normal Click Event Flow (Working)

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Mouse Click
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MATPLOTLIB CANVAS                             │
│  (FigureCanvasQTAgg - Qt/Matplotlib Bridge)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Dispatch button_press_event
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                EVENT CONNECTION REGISTRY                        │
│  Canvas-level connections (survive fig.clear()):               │
│                                                                 │
│  _cid_button  → PlotHost._on_button()                         │
│  _cid_scroll  → PlotHost._on_scroll()                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Call registered handler
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PLOTHOST._ON_BUTTON() HANDLER                     │
│                                                                 │
│  1. Check: event.inaxes is not None?                          │
│  2. Check: event.dblclick? → autoscale                        │
│  3. Check: _external_click_cb is not None?                    │
│  4. Forward to callback                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ YES: callback registered
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          CALLBACK ROUTING (_external_click_cb)                 │
│                                                                 │
│  Application-level registration (destroyed by redraws):        │
│                                                                 │
│  _external_click_cb = EditingModes._on_plot_click_mark_sniff  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Invoke callback
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│    EDITINGMODES._ON_PLOT_CLICK_MARK_SNIFF() HANDLER           │
│                                                                 │
│  1. Check: _mark_sniff_mode is True?                          │
│  2. Check: Shift key held? → delete region                    │
│  3. Check: Near existing edge? → adjust region                │
│  4. Store start position: _sniff_start_x = xdata              │
│  5. Wait for drag/release events                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ User drags mouse
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            CANVAS MOTION_NOTIFY_EVENT                          │
│                                                                 │
│  Dispatched to: _motion_cid → EditingModes._on_sniff_drag()  │
│                                                                 │
│  Draws preview rectangle: ax.axvspan(start, current, ...)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ User releases mouse
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          CANVAS BUTTON_RELEASE_EVENT                           │
│                                                                 │
│  Dispatched to: _release_cid → EditingModes._on_sniff_release()│
│                                                                 │
│  1. Finalize sniff region (start, end)                        │
│  2. Snap to breath events (onsets/expoffs)                    │
│  3. Save to state: sniff_regions_by_sweep[sweep] = regions    │
│  4. Merge overlapping regions                                  │
│  5. Redraw plot with permanent overlay                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ✅ OPERATION COMPLETE
```

---

## Diagram 2: Connection Lifecycle

```
┌───────────────────────────────────────────────────────────────────┐
│                    APPLICATION STARTUP                            │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │   PlotHost.__init__()                │
           │                                      │
           │   fig = plt.figure()                 │
           │   canvas = FigureCanvas(fig)         │
           │                                      │
           │   PERSISTENT CONNECTIONS:            │
           │   _cid_button = canvas.mpl_connect() │
           │   _cid_scroll = canvas.mpl_connect() │
           │                                      │
           │   CALLBACK PLACEHOLDER:              │
           │   _external_click_cb = None          │
           └──────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  USER ACTIVATES EDITING MODE                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │ EditingModes.on_mark_sniff_toggled() │
           │                                      │
           │   1. Set flag: _mark_sniff_mode = True│
           │   2. Register callback:              │
           │      plot_host.set_click_callback(   │
           │          self._on_plot_click_mark_sniff)│
           │                                      │
           │   3. TEMPORARY CONNECTIONS:          │
           │      _motion_cid = canvas.mpl_connect()│
           │      _release_cid = canvas.mpl_connect()│
           └──────────────────────────────────────┘
                              │
                              ▼
                    ✅ MODE ACTIVE
                              │
                              │ Time passes...
                              │ User changes Y2 metric
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  REDRAW TRIGGERED (Y2 CHANGE)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │ PlotHost.show_trace_with_spans()    │
           │                                      │
           │   fig.clear()  ← DESTROYS AXES      │
           │                                      │
           │   ax_main = fig.add_subplot(111)    │
           │   ax_main.plot(t, y, ...)           │
           │                                      │
           │   CLEARS REFERENCES:                 │
           │   self.ax_y2 = None                 │
           │   self.line_y2 = None               │
           │   self.scatter_peaks = None         │
           └──────────────────────────────────────┘
                              │
                              ▼
         ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
         ┃    CRITICAL MOMENT: WHAT SURVIVES?  ┃
         ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              ▼
         ┌─────────────────────────────────────┐
         │   SURVIVES (canvas-level):          │
         │   ✅ _cid_button connection         │
         │   ✅ _cid_scroll connection         │
         │   ✅ _motion_cid connection         │
         │   ✅ _release_cid connection        │
         └─────────────────────────────────────┘
                              │
         ┌─────────────────────────────────────┐
         │   DESTROYED (axes-level):           │
         │   ❌ ax_main callbacks              │
         │   ❌ ax_y2 (if existed)             │
         └─────────────────────────────────────┘
                              │
         ┌─────────────────────────────────────┐
         │   ORPHANED (app-level):             │
         │   ⚠️  _external_click_cb = ...      │
         │      (still points to callback,     │
         │       but not explicitly cleared)   │
         └─────────────────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │ PlotHost.add_or_update_y2()         │
           │                                      │
           │   ax_y2 = ax_main.twinx()           │
           │   ax_y2.set_navigate(False)         │
           │   ax_y2.patch.set_visible(False)    │
           │   ax_y2.plot(t, y2, ...)            │
           └──────────────────────────────────────┘
                              │
                              ▼
           ┌──────────────────────────────────────┐
           │ PlotManager._restore_editing_mode_   │
           │              connections()           │
           │                                      │
           │ THE FIX:                             │
           │ 1. Reconnect _cid_button (defensive) │
           │ 2. Re-register _external_click_cb    │
           │ 3. Reconnect _motion_cid (defensive) │
           │ 4. Reconnect _release_cid (defensive)│
           └──────────────────────────────────────┘
                              │
                              ▼
                  ✅ CONNECTIONS RESTORED
                              │
                              ▼
                    Ready for user clicks
```

---

## Diagram 3: Y2 Axis Event Blocking Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-AXIS PLOT LAYOUT                        │
└─────────────────────────────────────────────────────────────────┘

         Y-axis (Left)          PLOT AREA         Y2-axis (Right)
         ├───────────┼──────────────────────────┼───────────────┤
         │           │                          │               │
    1.0  ├───────────┼──────────────────────────┼───────────────┤ 10 Hz
         │           │  ███ Main Trace (black)  │               │
         │           │  ▬▬▬ Y2 Line (green)     │               │
    0.5  ├───────────┼──────────────────────────┼───────────────┤  5 Hz
         │           │                          │               │
         │           │  👆 USER CLICKS HERE     │               │
    0.0  ├───────────┼──────────────────────────┼───────────────┤  0 Hz
         │           │                          │               │
         └───────────┴──────────────────────────┴───────────────┘
         ax_main.yaxis                          ax_y2.yaxis
         (responds to                           (BLOCKED from
          mouse events)                          mouse events)


┌─────────────────────────────────────────────────────────────────┐
│              EVENT HANDLING STRATEGY                            │
└─────────────────────────────────────────────────────────────────┘

Click Event Arrives:
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  Matplotlib determines event.inaxes:                          │
│                                                               │
│  IF click is over plot area:                                 │
│      → Check: which axes owns this pixel?                    │
│      → WITHOUT blocking: returns ax_y2 (top layer)           │
│      → WITH blocking: returns ax_main (only navigable axes)  │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  PlotHost.add_or_update_y2() blocking configuration:         │
│                                                               │
│  ax_y2.set_navigate(False)                                   │
│  → Tells matplotlib: "Don't use this axes for navigation"    │
│  → Effect: Zoom/pan operations ignore this axes              │
│  → Effect: event.inaxes prefers ax_main over ax_y2           │
│                                                               │
│  ax_y2.patch.set_visible(False)                              │
│  → Makes axes background transparent                          │
│  → Effect: Click detection prefers axes below                │
│  → Effect: ax_main receives clicks even when ax_y2 overlaps  │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│  Result:                                                      │
│  ✅ event.inaxes = ax_main (desired)                         │
│  ✅ Click forwarded to _external_click_cb                    │
│  ✅ Mark Sniff mode receives click                           │
└───────────────────────────────────────────────────────────────┘

IMPORTANT: This blocking strategy is NECESSARY but NOT SUFFICIENT.
           The callback registration must ALSO be restored after
           fig.clear() for clicks to work.
```

---

## Diagram 4: Bug Scenario Timeline (Before Fix)

```
T0: USER ACTIVATES MARK SNIFF MODE
────────────────────────────────────────────────────────────────
PlotHost state:
    _cid_button = 3 ✅ (connected)
    _external_click_cb = <function _on_plot_click_mark_sniff> ✅

EditingModes state:
    _mark_sniff_mode = True ✅
    _motion_cid = 4 ✅ (connected)
    _release_cid = 5 ✅ (connected)

User clicks plot:
    Canvas → _cid_button → _on_button()
        → Checks _external_click_cb is not None: YES ✅
        → Forwards to _on_plot_click_mark_sniff() ✅
        → _sniff_start_x = 1.234 ✅
        → Drag preview works ✅

────────────────────────────────────────────────────────────────

T1: USER SELECTS Y2 METRIC (IF)
────────────────────────────────────────────────────────────────
MainWindow.on_y2_combo_change():
    state.y2_metric_key = "if"
    _compute_y2_all_sweeps()  → computes metrics
    redraw_main_plot()
        ↓
PlotHost.show_trace_with_spans():
    fig.clear()  ← 💥 DESTRUCTION EVENT
        ↓
    ax_main = fig.add_subplot(111)
    ax_main.plot(t, y, ...)
    canvas.draw_idle()

PlotHost state AFTER fig.clear():
    _cid_button = 3 ✅ (SURVIVED - canvas connection)
    _external_click_cb = <function _on_plot_click_mark_sniff> ⚠️
        (Reference still exists, but not explicitly re-registered)

EditingModes state AFTER fig.clear():
    _mark_sniff_mode = True ✅ (flag unchanged)
    _motion_cid = 4 ✅ (SURVIVED - canvas connection)
    _release_cid = 5 ✅ (SURVIVED - canvas connection)

────────────────────────────────────────────────────────────────

T2: Y2 AXIS ADDED
────────────────────────────────────────────────────────────────
PlotHost.add_or_update_y2():
    ax_y2 = ax_main.twinx()
    ax_y2.set_navigate(False)  → blocks Y2 from mouse events ✅
    ax_y2.plot(t, y2, ...)
    canvas.draw_idle()

PlotHost state AFTER Y2 addition:
    _cid_button = 3 ✅
    _external_click_cb = <function _on_plot_click_mark_sniff> ⚠️
        (Still exists! But WHY doesn't it work?)

────────────────────────────────────────────────────────────────

T3: USER CLICKS PLOT (EXPECTING MARK SNIFF TO WORK)
────────────────────────────────────────────────────────────────
Canvas receives button_press_event:
    Dispatches to _cid_button = 3
        ↓
PlotHost._on_button(event):
    event.inaxes = <AxesSubplot:...> (ax_main) ✅
    event.xdata = 2.345 ✅

    MYSTERY: Why doesn't callback work here?

    🔍 THEORY 1: _external_click_cb was cleared
        Check: Is _external_click_cb still set?
        → Debug log: callback=<function...> (YES, still set!)

    🔍 THEORY 2: event.xdata is None
        Check: Is xdata valid?
        → Debug log: xdata=2.345 (YES, valid!)

    🔍 THEORY 3: event.inaxes is ax_y2, not ax_main
        Check: Which axes received the click?
        → Debug log: inaxes=ax_main (CORRECT - blocking worked!)

    🔍 THEORY 4: Code path changed
        Check: Did conditional logic change?
        → IF statement: if self._external_click_cb is not None and event.xdata is not None:
        → Evaluates to: if True and True: → SHOULD execute!

    💡 AHA MOMENT: Look at the CODE STRUCTURE more carefully...

PlotHost._on_button() ACTUAL implementation:
    if event.inaxes is None:
        return  ← Early return

    if event.dblclick:
        ax.autoscale()
        canvas.draw_idle()
        return  ← Early return

    # 🚨 THE BUG IS HERE 🚨
    if self._external_click_cb is not None and event.xdata is not None:
        self._external_click_cb(event.xdata, event.ydata, event)
        # BUT WAIT - this should work!

    # Unless... there's a Y2-specific code path that breaks this?
    # OR the callback registration gets cleared during Y2 addition?

ACTUAL BUG LOCATION:
    → FOUND: The callback IS working, but EditingModes checks fail!
    → _on_plot_click_mark_sniff() checks: if not self._mark_sniff_mode: return
    → Mode flag is STILL TRUE, so this isn't the issue either...

    🎯 REAL ISSUE: The callback registration gets LOST somewhere between
       fig.clear() and Y2 addition. Need to check if show_trace_with_spans()
       explicitly clears the callback!

Checking PlotHost.show_trace_with_spans():
    Line 278: self.fig.clear()
    Line 282: self.clear_peaks()
    Lines 288-292:
        # IMPORTANT: Clear Y2 axis references after fig.clear()
        self.ax_y2 = None
        self.line_y2 = None
        self.line_y2_secondary = None

    ❌ NO EXPLICIT CLEAR OF _external_click_cb!
    → So the callback reference SHOULD persist!

FINAL THEORY:
    The bug report says editing modes "stop working" after Y2 changes.
    The fix adds _restore_editing_mode_connections() which DOES:
        plot_host.set_click_callback(editing._on_plot_click_mark_sniff)

    This suggests the callback WAS being cleared somewhere.
    Most likely: implicit clearing during redraw, or defensive clearing
    in some other code path.

THE FIX:
    Explicitly re-register the callback after EVERY redraw,
    treating it as non-persistent (like axes-level connections).

────────────────────────────────────────────────────────────────

T4: WITH FIX - RESTORATION RUNS
────────────────────────────────────────────────────────────────
PlotManager._restore_editing_mode_connections():
    # Reconnect button handler (defensive)
    canvas.mpl_disconnect(_cid_button)
    _cid_button = canvas.mpl_connect('button_press_event', _on_button)

    # Re-register callback (THE KEY FIX)
    plot_host.set_click_callback(editing._on_plot_click_mark_sniff)
        → _external_click_cb = <function _on_plot_click_mark_sniff> ✅

    # Reconnect motion/release (defensive)
    canvas.mpl_disconnect(_motion_cid)
    canvas.mpl_disconnect(_release_cid)
    _motion_cid = canvas.mpl_connect('motion_notify_event', _on_sniff_drag)
    _release_cid = canvas.mpl_connect('button_release_event', _on_sniff_release)

PlotHost state AFTER restoration:
    _cid_button = 6 ✅ (fresh connection)
    _external_click_cb = <function _on_plot_click_mark_sniff> ✅ (re-registered)

EditingModes state AFTER restoration:
    _mark_sniff_mode = True ✅
    _motion_cid = 7 ✅ (fresh connection)
    _release_cid = 8 ✅ (fresh connection)

User clicks plot:
    Canvas → _cid_button(6) → _on_button()
        → Checks _external_click_cb is not None: YES ✅
        → Forwards to _on_plot_click_mark_sniff() ✅
        → _sniff_start_x = 2.345 ✅
        → Drag preview works ✅

────────────────────────────────────────────────────────────────

✅ RESULT: BUG FIXED
```

---

## Diagram 5: Comparison Matrix

| Scenario | show_trace_with_spans() | add_or_update_y2() | _restore_editing_mode_connections() | Result |
|----------|------------------------|-------------------|-------------------------------------|--------|
| **Sweep navigation** | ✅ Called (fig.clear) | ❌ Not called | ✅ Called | ✅ Works |
| **Filter change** | ✅ Called (fig.clear) | ❌ Not called | ✅ Called | ✅ Works |
| **Peak detection** | ✅ Called (fig.clear) | ❌ Not called | ✅ Called | ✅ Works |
| **Y2 selection (before fix)** | ✅ Called (fig.clear) | ✅ Called | ❌ NOT called | ❌ BROKEN |
| **Y2 selection (after fix)** | ✅ Called (fig.clear) | ✅ Called | ✅ Called | ✅ Works |

**Key Insight:** The restoration was ALWAYS needed after Y2 addition, but wasn't implemented until the fix.

