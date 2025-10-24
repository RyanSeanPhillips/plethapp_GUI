# Event Channel & Bout Marking Implementation Guide

## Status: Foundation Complete, Integration Needed

### ✅ Completed (Phase 1)
1. **State Management** (`core/state.py`):
   - Added `event_channel: Optional[str]` field
   - Added `bout_annotations: Dict[int, List[Dict]]` field for storing bouts per sweep
   - Bout format: `{'start_time': float, 'end_time': float, 'id': int}`

### 🚧 Next Steps (Phase 2 - UI Integration)

#### Step 1: Add Event Channel Dropdown Widget

Add this code in `main.py` `__init__()` method, after the UI is loaded (around line 59):

```python
# Create Event Channel dropdown dynamically
from PyQt6.QtWidgets import QLabel, QComboBox

# Create label
self.EventChannelLabel = QLabel("Event Channel:")
self.EventChannelLabel.setObjectName("EventChannelLabel")
self.EventChannelLabel.setFont(QFont())
self.EventChannelLabel.font().setPointSize(11)
self.EventChannelLabel.setMaximumWidth(100)

# Create combobox
self.EventChannelCombo = QComboBox()
self.EventChannelCombo.setObjectName("EventChannelCombo")
self.EventChannelCombo.setMaximumWidth(225)
self.EventChannelCombo.addItem("None")  # Default option

# Insert into horizontal layout (after AnalyzeChanSelect, before StimChanSelect)
# Find the layout - it's called horizontalLayout_8
channel_layout = self.findChild(QHBoxLayout, "horizontalLayout_8")
if channel_layout:
    # Find position after AnalyzeChanSelect
    analyze_combo_index = None
    for i in range(channel_layout.count()):
        widget = channel_layout.itemAt(i).widget()
        if widget and widget.objectName() == "AnalyzeChanSelect":
            analyze_combo_index = i
            break

    if analyze_combo_index is not None:
        # Insert after AnalyzeChanSelect
        channel_layout.insertWidget(analyze_combo_index + 1, self.EventChannelLabel)
        channel_layout.insertWidget(analyze_combo_index + 2, self.EventChannelCombo)

# Connect to handler
self.EventChannelCombo.currentTextChanged.connect(self.on_event_channel_changed)
```

#### Step 2: Populate Event Channel Dropdown

In `update_and_redraw()` method, populate the dropdown with available channels:

```python
# Around line 540, after populating AnalyzeChanSelect
self.EventChannelCombo.blockSignals(True)
self.EventChannelCombo.clear()
self.EventChannelCombo.addItem("None")
for ch in st.channel_names:
    self.EventChannelCombo.addItem(ch)

# Restore previous selection
if st.event_channel and st.event_channel in st.channel_names:
    idx = self.EventChannelCombo.findText(st.event_channel)
    if idx >= 0:
        self.EventChannelCombo.setCurrentIndex(idx)
else:
    self.EventChannelCombo.setCurrentIndex(0)

self.EventChannelCombo.blockSignals(False)
```

#### Step 3: Handle Event Channel Selection

Add this method in `main.py`:

```python
def on_event_channel_changed(self, text):
    """Handle event channel selection change."""
    st = self.state
    if text == "None":
        st.event_channel = None
    else:
        st.event_channel = text

    # Redraw with new layout
    self.redraw_main_plot()
```

#### Step 4: Modify Plotting to Support Dual Subplot

Modify `_draw_main()` method to create dual subplot layout when event channel is selected:

```python
def _draw_main(self):
    """Draw main plot with optional event channel subplot."""
    st = self.state
    # ... existing setup code ...

    # Determine if we need dual subplot
    use_event_subplot = (st.event_channel is not None)

    if use_event_subplot:
        # Create dual subplot layout with shared x-axis
        from matplotlib.gridspec import GridSpec

        fig = self.plot_host.fig
        fig.clear()

        gs = GridSpec(2, 1, height_ratios=[0.7, 0.3], hspace=0.05, figure=fig)
        ax_pleth = fig.add_subplot(gs[0])
        ax_event = fig.add_subplot(gs[1], sharex=ax_pleth)

        # Hide x-axis labels on top plot
        ax_pleth.tick_params(labelbottom=False)

        # Plot pleth trace on top subplot
        self._plot_pleth_trace(ax_pleth, y, t_full)

        # Plot event channel on bottom subplot
        self._plot_event_trace(ax_event, t_full)

        # Plot bout annotations if any exist
        if swp in st.bout_annotations and st.bout_annotations[swp]:
            self._plot_bout_annotations(ax_pleth, ax_event, st.bout_annotations[swp])

    else:
        # Single plot mode (current behavior)
        fig = self.plot_host.fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._plot_pleth_trace(ax, y, t_full)

    fig.canvas.draw_idle()
```

#### Step 5: Add Event Trace Plotting Method

```python
def _plot_event_trace(self, ax, t_full):
    """Plot event channel trace on given axis."""
    st = self.state
    swp = st.sweep_idx

    # Get event channel data
    event_data_full = st.sweeps[st.event_channel][:, swp]

    # Apply window slicing if needed
    in_window = (t_full >= st.window_start_s) & (t_full < st.window_start_s + st.window_dur_s)
    t_win = t_full[in_window]
    event_data = event_data_full[in_window]

    # Plot continuous trace
    ax.plot(t_win, event_data, 'b-', linewidth=1, label=st.event_channel)
    ax.set_ylabel(f'{st.event_channel}', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
```

#### Step 6: Add Mark Bout Button

Add button creation in `__init__()`:

```python
# Create Mark Bout button near other editing buttons
self.MarkBoutButton = QPushButton("Mark Bout")
self.MarkBoutButton.setObjectName("MarkBoutButton")
self.MarkBoutButton.setCheckable(True)  # Toggle button
self.MarkBoutButton.setStyleSheet("""
    QPushButton {
        background-color: #3498db;
        color: white;
        border-radius: 6px;
        padding: 5px 10px;
    }
    QPushButton:hover {
        background-color: #2980b9;
    }
    QPushButton:checked {
        background-color: #1abc9c;
        border: 2px solid #16a085;
    }
""")
self.MarkBoutButton.setMinimumWidth(100)
self.MarkBoutButton.setMaximumWidth(100)
self.MarkBoutButton.setToolTip("Click twice to mark bout start and end times")

# Add to editing buttons grid (gridLayout_2)
editing_grid = self.findChild(QGridLayout, "gridLayout_2")
if editing_grid:
    # Add in row 3, column 0 (below other buttons)
    editing_grid.addWidget(self.MarkBoutButton, 3, 0)

# Connect handler
self.MarkBoutButton.clicked.connect(self.on_mark_bout_clicked)
```

#### Step 7: Implement Bout Marking Logic

Add these methods:

```python
def on_mark_bout_clicked(self, checked):
    """Toggle bout marking mode."""
    if checked:
        # Enable bout marking mode
        self.bout_start_time = None
        self.bout_marking_active = True
        self.MarkBoutButton.setText("Mark Bout (Active)")

        # Connect plot click handler
        self.plot_click_cid = self.plot_host.fig.canvas.mpl_connect(
            'button_press_event', self._on_plot_click_mark_bout
        )
    else:
        # Disable bout marking mode
        self.bout_marking_active = False
        self.bout_start_time = None
        self.MarkBoutButton.setText("Mark Bout")

        # Disconnect handler
        if hasattr(self, 'plot_click_cid'):
            self.plot_host.fig.canvas.mpl_disconnect(self.plot_click_cid)

def _on_plot_click_mark_bout(self, event):
    """Handle plot click for bout marking."""
    if event.inaxes is None:
        return

    st = self.state
    click_time = event.xdata

    if self.bout_start_time is None:
        # First click - set start time
        self.bout_start_time = click_time
        print(f"Bout start: {click_time:.3f}s")

        # Draw vertical line at start
        event.inaxes.axvline(click_time, color='green', linestyle='--', linewidth=2, alpha=0.7)
        self.plot_host.fig.canvas.draw_idle()
    else:
        # Second click - set end time and create bout
        bout_end = click_time
        bout_start = self.bout_start_time

        # Ensure start < end
        if bout_end < bout_start:
            bout_start, bout_end = bout_end, bout_start

        # Create bout annotation
        swp = st.sweep_idx
        if swp not in st.bout_annotations:
            st.bout_annotations[swp] = []

        # Generate unique ID
        bout_id = len(st.bout_annotations[swp]) + 1

        bout = {
            'start_time': bout_start,
            'end_time': bout_end,
            'id': bout_id
        }
        st.bout_annotations[swp].append(bout)

        print(f"Bout created: {bout_start:.3f}s - {bout_end:.3f}s")

        # Reset for next bout
        self.bout_start_time = None

        # Redraw to show bout
        self.redraw_main_plot()
```

#### Step 8: Visualize Bout Annotations

```python
def _plot_bout_annotations(self, ax_pleth, ax_event, bouts):
    """Plot bout annotations as shaded regions on both subplots."""
    for bout in bouts:
        start = bout['start_time']
        end = bout['end_time']

        # Shaded region on both subplots
        ax_pleth.axvspan(start, end, alpha=0.2, color='cyan', label=f"Bout {bout['id']}")
        ax_event.axvspan(start, end, alpha=0.2, color='cyan')

        # Vertical lines at boundaries
        ax_pleth.axvline(start, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_pleth.axvline(end, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_event.axvline(start, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
        ax_event.axvline(end, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
```

#### Step 9: Export Bout Data

Add bout export to `core/export.py`:

```python
def export_bout_annotations(state: AppState, base_path: Path):
    """Export bout annotations to CSV."""
    import pandas as pd

    bout_rows = []
    for sweep_idx in sorted(state.bout_annotations.keys()):
        for bout in state.bout_annotations[sweep_idx]:
            bout_rows.append({
                'sweep': sweep_idx,
                'bout_id': bout['id'],
                'start_time_s': bout['start_time'],
                'end_time_s': bout['end_time'],
                'duration_s': bout['end_time'] - bout['start_time']
            })

    if bout_rows:
        df = pd.DataFrame(bout_rows)
        out_path = base_path.parent / f"{base_path.stem}_bouts.csv"
        df.to_csv(out_path, index=False)
        print(f"Exported bout annotations to: {out_path}")
        return out_path
    return None
```

Call this in the main export function.

---

## Advanced Features (Phase 3 - Optional)

### Bout Deletion
1. Click on bout to select (highlight in different color)
2. Press Delete key or click Delete Bout button
3. Remove from `state.bout_annotations[sweep_idx]`

### Bout Boundary Editing
1. Click and hold near bout edge (within ±0.5s)
2. Drag to adjust start/end time
3. Update bout dict on mouse release

### Multiple Event Channels
- Change `event_channel` to `event_channels: List[str]`
- Create N subplots dynamically based on number of selected channels
- Use `GridSpec` with dynamic height ratios

---

## Testing Checklist

- [ ] Event channel dropdown appears in UI
- [ ] Dropdown populated with all channels
- [ ] Selecting "None" shows single plot
- [ ] Selecting channel shows dual subplot
- [ ] Event trace displays correctly
- [ ] Mark Bout button toggles mode
- [ ] First click marks start (green line)
- [ ] Second click marks end (red line)
- [ ] Bout region shaded in cyan
- [ ] Bout persists across navigation
- [ ] Bout exports to CSV

---

## Known Limitations
- Bouts are not yet editable after creation
- No bout deletion UI (can be cleared by deleting from state manually)
- Only supports one event channel at a time
- Bout annotations not included in NPZ save/load (add to state serialization)

---

## File Locations
- **State**: `core/state.py` (✅ Complete)
- **Main UI Logic**: `main.py` (🚧 Needs integration)
- **Export**: `core/export.py` (🚧 Needs bout export function)
- **Documentation**: This file

---

## Questions?
Refer to existing code patterns:
- Sniff marking: `markSniffButton` and `_on_mark_sniff_clicked()`
- Peak editing: `addPeaksButton` and `_on_plot_click_add_peak()`
- Dual axis plotting: `_draw_main()` with y2 metrics
