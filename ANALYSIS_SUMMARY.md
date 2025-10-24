# PlethApp Event Handling Analysis - Executive Summary

**Analysis Date:** 2025-10-13
**Issue:** Editing modes stop working after Y2 axis changes
**Status:** Fix implemented, verification required

---

## Key Findings

### 1. Root Cause Identified

The bug stems from **matplotlib's multi-level event system** interacting with PlethApp's modular architecture:

```
Canvas Connections (survive fig.clear())
    ├─ _cid_button ✅ Persistent
    ├─ _cid_scroll ✅ Persistent
    └─ Mode-specific (_motion_cid, _release_cid) ✅ Persistent

Axes Connections (destroyed by fig.clear())
    └─ axes callbacks ❌ Destroyed

Application-Level Registration (no automatic persistence)
    └─ _external_click_cb ⚠️ Orphaned during redraw
```

**The Problem:** When `show_trace_with_spans()` calls `fig.clear()`, it destroys all axes but not canvas connections. The `_external_click_cb` reference that routes clicks to editing modes needs manual restoration.

---

## 2. Why Y2 Axis Addition Exposed the Bug

| Operation | Calls fig.clear()? | Restoration Ran? | Result |
|-----------|-------------------|------------------|--------|
| Sweep navigation | YES | YES | ✅ Works |
| Filter changes | YES | YES | ✅ Works |
| Peak detection | YES | YES | ✅ Works |
| **Y2 metric selection** | **YES** | **NO (before fix)** | **❌ Broken** |

The Y2 axis addition code path didn't include the restoration logic that other operations had.

---

## 3. The Implemented Fix

**Location:** `plotting/plot_manager.py`, method `_restore_editing_mode_connections()`

**What it does:**
1. Reconnects `_cid_button` (defensive, canvas connection already survives)
2. **Re-registers `_external_click_cb`** (THE CRITICAL FIX)
3. Reconnects motion/release events (defensive)
4. Repeats for all editing modes: Mark Sniff, Add Peaks, Delete Peaks, Add Sigh, Move Point

**When it runs:**
- Called at end of `PlotManager._draw_single_panel_plot()`
- Executes AFTER `show_trace_with_spans()` (fig.clear)
- Executes AFTER `_draw_y2_metric()` (Y2 axis creation)

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         MainWindow                          │
│  (Coordinates modules, handles UI signals)                  │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼──────────────────┐
            │                 │                  │
            ▼                 ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  PlotHost    │  │ PlotManager  │  │EditingModes  │
    │              │  │              │  │              │
    │ • Figure     │  │ • Orchestrate│  │ • Mode flags │
    │ • Canvas     │  │   plotting   │  │ • Callbacks  │
    │ • Persistent │  │ • Restore    │  │ • Temp       │
    │   events     │  │   connections│  │   connections│
    └──────────────┘  └──────────────┘  └──────────────┘
            │                 │                  │
            └─────────────────┼──────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
         ┌──────────────┐          ┌──────────────┐
         │ Matplotlib   │          │NavigationMgr │
         │ Figure/Axes  │          │ (Sweep Nav)  │
         └──────────────┘          └──────────────┘
```

---

## 5. Event Flow (Simplified)

### Normal Click (Working)
```
User Click
    → Canvas dispatches event
    → _cid_button calls PlotHost._on_button()
    → Checks _external_click_cb is not None
    → Forwards to EditingModes._on_plot_click_mark_sniff()
    → Handles click (store position, wait for drag)
```

### After Y2 Addition (Before Fix)
```
User Click
    → Canvas dispatches event
    → _cid_button calls PlotHost._on_button()
    → Checks _external_click_cb is not None
    → ❌ CALLBACK WAS CLEARED/ORPHANED
    → No forwarding, click ignored
```

### After Y2 Addition (With Fix)
```
User Click
    → Canvas dispatches event
    → _cid_button calls PlotHost._on_button()
    → Checks _external_click_cb is not None
    → ✅ CALLBACK RESTORED BY FIX
    → Forwards to EditingModes._on_plot_click_mark_sniff()
    → Handles click normally
```

---

## 6. Verification Checklist

To confirm the fix works, test this sequence:

### Test Procedure
1. **Launch application**
2. **Load data file**
3. **Detect peaks** (to have breath events)
4. **Activate Mark Sniff mode** (button should highlight)
5. **Click and drag on plot** → Verify purple preview appears ✅
6. **Release mouse** → Verify sniff region is saved (purple overlay) ✅
7. **Select Y2 metric "IF"** from dropdown
8. **Click and drag on plot again** → Verify still works ✅

### Expected Console Output (if fix is working)
```
[editing-debug] Reconnecting main button press event handler
[editing-debug] Reconnected button press: cid=7
[editing-debug] Mark Sniff mode is active - reconnecting matplotlib events
[editing-debug] Re-registered click callback: <function _on_plot_click_mark_sniff>
[editing-debug] Reconnected: motion_cid=8, release_cid=9
[plot-click-debug] Click detected: inaxes=AxesSubplot(...), callback=True
[plot-click-debug] Forwarding to callback: xdata=2.345
[mark-sniff] Started new region at x=2.345
```

### Expected Console Output (if fix is NOT working)
```
[plot-click-debug] Click detected: inaxes=AxesSubplot(...), callback=False
[plot-click-debug] Not forwarding - callback=False, xdata=2.345
```

---

## 7. Potential Issues to Investigate

If the fix doesn't work, check these:

| Issue | Diagnostic | Solution |
|-------|-----------|----------|
| **Restoration called too early** | Check if Y2 axis exists when restoration runs | Move restoration call after Y2 creation |
| **Callback cleared elsewhere** | Add logging to `set_click_callback()` and `clear_click_callback()` | Find rogue clearing code |
| **Y2 axis blocks clicks** | Check `event.inaxes` in debug output | Verify `set_navigate(False)` is applied |
| **Connection ID issues** | Verify disconnect succeeds before reconnect | Add try/except around disconnect |
| **Toolbar interference** | Check if toolbar mode activates during redraw | Ensure `turn_off_toolbar_modes()` called |

---

## 8. Comparison with Other Editing Modes

| Mode | Click Handler | Motion Handler | Release Handler | Restoration |
|------|--------------|----------------|----------------|-------------|
| **Mark Sniff** | `_on_plot_click_mark_sniff` | `_on_sniff_drag` | `_on_sniff_release` | ✅ Implemented |
| **Add Peaks** | `_on_plot_click_add_peak` | ❌ None | ❌ None | ✅ Implemented |
| **Delete Peaks** | `_on_plot_click_delete_peak` | ❌ None | ❌ None | ✅ Implemented |
| **Add Sigh** | `_on_plot_click_add_sigh` | ❌ None | ❌ None | ✅ Implemented |
| **Move Point** | `_on_plot_click_move_point` | `_on_canvas_motion` | `_on_canvas_release` | ✅ Implemented |

**All modes should work** after Y2 changes if the fix is implemented correctly.

---

## 9. Files Modified

### Primary Files
- `plotting/plot_manager.py` - Added `_restore_editing_mode_connections()` method
- `core/plotting.py` - Existing PlotHost class (no changes needed for fix)
- `editing/editing_modes.py` - Existing EditingModes class (no changes needed for fix)

### Documentation Created
- `EVENT_HANDLING_ARCHITECTURE.md` - Comprehensive 10-section analysis
- `EVENT_FLOW_DIAGRAM.md` - 5 ASCII diagrams showing event flows
- `ANALYSIS_SUMMARY.md` - This executive summary

---

## 10. Recommended Follow-up Actions

### Immediate (After Verification)
1. ✅ Test all editing modes after Y2 changes
2. ✅ Test all Y2 metrics (IF, Sniff Conf, Eupnea Conf)
3. ✅ Test mode switching after Y2 changes
4. ✅ Add regression test to test suite

### Short-term (Next Sprint)
1. Add event connection audit system
2. Implement query-based callback resolution (eliminate manual restoration)
3. Add warning system for orphaned connections
4. Document connection lifecycle for future developers

### Long-term (Architecture)
1. Separate canvas lifecycle from figure lifecycle
2. Use `ax.cla()` instead of `fig.clear()` where possible
3. Implement connection pooling/registry
4. Create centralized event routing system

---

## 11. Conclusion

The fix is **theoretically sound** and should resolve the issue. The root cause was clearly identified as callback registration loss during figure redraws. The restoration logic explicitly re-registers callbacks after every redraw, which is the correct approach given matplotlib's event system design.

**Confidence Level:** 90% - The fix should work if:
- It's called at the right time (after Y2 axis creation) ✅
- No other code clears callbacks after restoration (needs verification)
- Y2 axis doesn't intercept clicks (already has `set_navigate(False)`) ✅

**Next Step:** Run the verification checklist and examine console debug output to confirm the fix is working as expected.

---

**Document Version:** 1.0
**Analysis Completed By:** Claude (Anthropic)
**Files Analyzed:**
- `core/plotting.py` (940 lines)
- `plotting/plot_manager.py` (515 lines)
- `editing/editing_modes.py` (1532 lines)
- Plus searches across entire codebase for event connections

**Total Analysis Time:** Approximately 45 minutes
**Lines of Code Analyzed:** ~3000+ lines across multiple modules
