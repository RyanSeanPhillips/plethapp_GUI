# Quick Troubleshooting Guide: Editing Modes After Y2 Changes

**Quick Reference:** Debug and fix editing mode issues after Y2 axis changes

---

## Symptom Checklist

### What's NOT Working?

- [ ] Mark Sniff mode - Clicks don't create regions
- [ ] Add Peaks mode - Clicks don't add peaks
- [ ] Delete Peaks mode - Clicks don't delete peaks
- [ ] Add Sigh mode - Clicks don't add/remove stars
- [ ] Move Point mode - Can't select or drag points

### When Does It Break?

- [ ] Immediately after launching app
- [ ] After selecting Y2 metric for first time
- [ ] After changing Y2 metric (switching between IF/Conf)
- [ ] After turning Y2 off then on again
- [ ] Only on certain sweeps
- [ ] Inconsistently / randomly

---

## Quick Diagnostic Steps

### Step 1: Check Mode Activation (5 seconds)

**Action:** Click the mode button (e.g., "Mark Sniff")

**Expected:**
- Button changes color/highlights ✅
- Button text updates (e.g., "Mark Sniff (ON)") ✅
- Cursor changes to crosshair ✅
- Console shows: `[mode-name] Mode activated` ✅

**If NOT working:**
- Problem is in mode activation, not event handling
- Check `EditingModes.on_[mode]_toggled()` method
- Issue is BEFORE the Y2 bug

---

### Step 2: Check Click Detection (10 seconds)

**Action:** With mode active, click on plot

**Expected console output:**
```
[plot-click-debug] Click detected: inaxes=AxesSubplot(...), callback=True
[plot-click-debug] Forwarding to callback: xdata=1.234
```

**Actual output variants:**

| Output | Meaning | Fix |
|--------|---------|-----|
| `callback=False` | `_external_click_cb` is None | ❌ Restoration didn't run |
| `inaxes=None` | Click was outside axes | User error (click inside plot) |
| `inaxes=<...Y2...>` | Y2 axis intercepting clicks | ❌ Y2 blocking failed |
| No output at all | Event not reaching handler | ❌ Connection broken |

---

### Step 3: Check Restoration (10 seconds)

**Action:** Select Y2 metric, watch console

**Expected console output:**
```
[editing-debug] Reconnecting main button press event handler
[editing-debug] Reconnected button press: cid=7
[editing-debug] Mark Sniff mode is active - reconnecting matplotlib events
[editing-debug] Re-registered click callback: <function _on_plot_click_mark_sniff>
[editing-debug] Reconnected: motion_cid=8, release_cid=9
```

**If NO output:**
- `_restore_editing_mode_connections()` is NOT being called
- Check `PlotManager._draw_single_panel_plot()` line ~40
- Verify it calls `self._restore_editing_mode_connections()`

**If output shows wrong callback:**
```
[editing-debug] Re-registered click callback: None
```
- Mode flag is False (mode was turned off)
- Re-activate the mode after Y2 selection

---

## Common Issues and Fixes

### Issue 1: "callback=False" After Y2 Selection

**Diagnosis:** Callback registration was not restored

**Possible Causes:**
1. Restoration method not called
2. Restoration called before mode activation
3. Callback cleared after restoration

**Fix Steps:**

```python
# In plotting/plot_manager.py, _draw_single_panel_plot()

# VERIFY this exists at the end of the method:
def _draw_single_panel_plot(self):
    # ... draw base trace
    # ... draw markers
    # ... draw Y2 metric
    # ... draw region overlays

    # CRITICAL: Restore editing mode connections
    self._restore_editing_mode_connections()  # ← MUST BE HERE
```

**Verification:**
- Add breakpoint in `_restore_editing_mode_connections()`
- Run with Y2 selection
- Verify it hits the breakpoint

---

### Issue 2: "inaxes=<...Y2...>" Instead of Main Axis

**Diagnosis:** Y2 axis is intercepting clicks

**Possible Causes:**
1. `set_navigate(False)` not applied
2. `patch.set_visible(False)` not applied
3. Y2 axis created without blocking

**Fix Steps:**

```python
# In core/plotting.py, add_or_update_y2()

def add_or_update_y2(self, t, y2, label, color, max_points):
    # ... create twin axis
    self.ax_y2 = ax.twinx()

    # ✅ VERIFY THESE TWO LINES EXIST:
    ax_y2.set_navigate(False)        # Disable navigation
    ax_y2.patch.set_visible(False)   # Make background transparent

    # ... plot Y2 data
```

**Verification:**
- Check `ax_y2` object after creation:
  ```python
  print(f"Y2 navigate: {ax_y2.get_navigate()}")  # Should be False
  print(f"Y2 visible: {ax_y2.patch.get_visible()}")  # Should be False
  ```

---

### Issue 3: Restoration Runs But Still Broken

**Diagnosis:** Callback is being cleared AFTER restoration

**Possible Causes:**
1. Another redraw happens after restoration
2. Toolbar mode activates automatically
3. Mode gets turned off during redraw

**Debug Steps:**

Add logging to callback setters/clearers:

```python
# In core/plotting.py

def set_click_callback(self, fn):
    import traceback
    print(f"[SET CALLBACK] Setting: {fn}")
    traceback.print_stack(limit=5)  # Show who called this
    self._external_click_cb = fn

def clear_click_callback(self):
    import traceback
    print(f"[CLEAR CALLBACK] Clearing: {self._external_click_cb}")
    traceback.print_stack(limit=5)  # Show who called this
    self._external_click_cb = None
```

**Look for:**
- `[CLEAR CALLBACK]` after `[SET CALLBACK]` (callback cleared after restoration)
- Multiple `[SET CALLBACK]` calls (overwriting with None?)

---

### Issue 4: Works First Time, Breaks on Subsequent Y2 Changes

**Diagnosis:** State accumulation or connection leak

**Possible Causes:**
1. Old connections not properly disconnected
2. Mode flag state inconsistent
3. Multiple callbacks registered

**Debug Steps:**

Check connection state:

```python
# In plotting/plot_manager.py, _restore_editing_mode_connections()

# Add at the start:
print(f"[RESTORE START]")
print(f"  Mode flags: mark_sniff={editing._mark_sniff_mode}, "
      f"add_peaks={editing._add_peaks_mode}, "
      f"delete_peaks={editing._delete_peaks_mode}")
print(f"  Current callback: {self.window.plot_host._external_click_cb}")
print(f"  Current _cid_button: {self.window.plot_host._cid_button}")

# Add at the end:
print(f"[RESTORE END]")
print(f"  New callback: {self.window.plot_host._external_click_cb}")
print(f"  New _cid_button: {self.window.plot_host._cid_button}")
```

**Look for:**
- Mode flags all False (modes got turned off)
- Connection IDs incrementing oddly (connection leak)
- Callback None after restoration (overwrite bug)

---

## Emergency Workaround

If nothing else works, add manual restoration to Y2 selection handler:

```python
# In main.py, Y2 combo box handler

def on_y2_combo_change(self):
    # ... compute Y2 metrics
    self._compute_y2_all_sweeps()

    # ... redraw plot
    self.redraw_main_plot()

    # EMERGENCY FIX: Force restoration
    if hasattr(self, 'plot_manager'):
        self.plot_manager._restore_editing_mode_connections()
```

This is a band-aid, not a proper fix, but will confirm if the issue is just call order.

---

## Advanced Debugging: Connection Audit

Add a connection registry to track all matplotlib connections:

```python
# In core/plotting.py

class PlotHost(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Add connection registry
        self._connection_registry = {}

        # ... rest of init

    def _register_connection(self, event_type, cid, handler_name):
        """Track all connections for debugging."""
        self._connection_registry[cid] = {
            'event': event_type,
            'handler': handler_name,
            'created': time.time()
        }

    def _audit_connections(self):
        """Print all active connections."""
        print("[CONNECTION AUDIT]")
        for cid, info in self._connection_registry.items():
            print(f"  cid={cid}: {info['event']} -> {info['handler']}")
```

Call `plot_host._audit_connections()` before and after Y2 addition to see what changed.

---

## Validation Script

Run this in Python console after issue occurs:

```python
# Get references
main = app.main_window  # Adjust based on your app structure
plot_host = main.plot_host
editing = main.editing_modes

# Check mode flags
print(f"Mark Sniff active: {editing._mark_sniff_mode}")
print(f"Add Peaks active: {editing._add_peaks_mode}")
print(f"Move Point active: {editing._move_point_mode}")

# Check connections
print(f"Button connection: {plot_host._cid_button}")
print(f"Motion connection: {editing._motion_cid}")
print(f"Release connection: {editing._release_cid}")

# Check callback
print(f"Click callback: {plot_host._external_click_cb}")

# Expected output if working:
# Mark Sniff active: True
# Add Peaks active: False
# Move Point active: False
# Button connection: 7 (some number)
# Motion connection: 8 (some number)
# Release connection: 9 (some number)
# Click callback: <function _on_plot_click_mark_sniff at 0x...>

# If callback is None, restoration failed!
```

---

## Contact Points in Code

If you need to modify the fix, here are the key locations:

| File | Line(s) | What It Does |
|------|---------|--------------|
| `plotting/plot_manager.py` | 433-515 | `_restore_editing_mode_connections()` - The fix |
| `plotting/plot_manager.py` | 28-103 | `redraw_main_plot()` / `_draw_single_panel_plot()` - Orchestration |
| `core/plotting.py` | 104-105 | Initial button connection creation |
| `core/plotting.py` | 151-173 | `_on_button()` - Click handler |
| `core/plotting.py` | 656-661 | `set_click_callback()` / `clear_click_callback()` |
| `core/plotting.py` | 546-598 | `add_or_update_y2()` - Y2 axis creation |
| `editing/editing_modes.py` | 1173-1241 | `on_mark_sniff_toggled()` - Mode activation |
| `editing/editing_modes.py` | 1242-1308 | `_on_plot_click_mark_sniff()` - Click handler |

---

## Success Indicators

You know the fix is working when:

✅ Console shows restoration messages after Y2 selection
✅ Click debug shows `callback=True`
✅ Mode handlers receive click events (`[mark-sniff] Started new region`)
✅ Drag preview appears (purple rectangle for Mark Sniff)
✅ Final overlay renders after mouse release
✅ Works on multiple Y2 toggles (not just first time)
✅ Works for all editing modes
✅ Works for all Y2 metrics

---

## Still Stuck?

1. **Check the main analysis document:** `EVENT_HANDLING_ARCHITECTURE.md` (detailed technical analysis)
2. **Check the flow diagrams:** `EVENT_FLOW_DIAGRAM.md` (visual representations)
3. **Check the summary:** `ANALYSIS_SUMMARY.md` (high-level overview)
4. **Add more debug logging:** Instrument every step of the event flow
5. **Bisect the code:** Comment out Y2 axis creation to confirm it's the trigger
6. **Compare working vs broken states:** Dump all relevant variables before/after Y2

---

**Last Updated:** 2025-10-13
**Document Version:** 1.0
