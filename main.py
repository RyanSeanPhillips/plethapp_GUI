from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtGui import QIcon

import re
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QLabel,
    QDialogButtonBox, QPushButton, QHBoxLayout, QCheckBox
)

import csv, json



from pathlib import Path
import sys
import numpy as np

from core.state import AppState
from core import abf_io, filters
from core.plotting import PlotHost
from core import stim as stimdet   # stim detection
from core import peaks as peakdet   # peak detection
from core import metrics  # calculation of breath metrics




ORG = "PlethApp"
APP = "PlethApp"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui_file = Path(__file__).parent / "ui" / "pleth_app_layout.ui"
        uic.loadUi(ui_file, self)
        icon_path = Path(__file__).parent / "assets" / "plethapp_thumbnail.ico"
        self.setWindowIcon(QIcon(str(icon_path)))
        # after uic.loadUi(ui_file, self)
        from PyQt6.QtWidgets import QWidget
        for w in self.findChildren(QWidget):
            if w.property("startHidden") is True:
                w.hide()

        self.setWindowTitle("PlethApp")

        self.settings = QSettings(ORG, APP)
        self.state = AppState()
        self.single_panel_mode = False  # flips True after stim channel selection
        # store peaks per sweep
        self.state.peaks_by_sweep = {}
        self.state.breath_by_sweep = {}

        # Y2 plotting
        self.state.y2_metric_key = None
        self.state.y2_values_by_sweep = {}

        # --- Embed Matplotlib into MainPlot (QFrame in Designer) ---
        self.plot_host = PlotHost(self.MainPlot)
        layout = self.MainPlot.layout()
        if layout is None:
            from PyQt6.QtWidgets import QVBoxLayout
            layout = QVBoxLayout(self.MainPlot)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_host)

        # --- Wire browse ---
        self.BrowseButton.clicked.connect(self.on_browse_clicked)

        # --- Wire channel selection ---
        self.AnalyzeChanSelect.currentIndexChanged.connect(self.on_analyze_channel_changed)
        self.StimChanSelect.currentIndexChanged.connect(self.on_stim_channel_changed)
        self.ApplyChanPushButton.clicked.connect(self.on_apply_channels_clicked)
        self.ApplyChanPushButton.setEnabled(False)  # disabled until something changes

        # Track pending (unapplied) selections
        self._pending_analyze_idx = None
        self._pending_stim_idx = None


        # --- Wire filter controls ---
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(150)       # ms
        self._redraw_timer.timeout.connect(self.redraw_main_plot)

        # filters: commit-on-finish, not per key
        self.LowPassVal.editingFinished.connect(self.update_and_redraw)
        self.HighPassVal.editingFinished.connect(self.update_and_redraw)
        self.MeanSubractVal.editingFinished.connect(self.update_and_redraw)

        # checkboxes toggled immediately, but we debounce the draw
        self.LowPass_checkBox.toggled.connect(self.update_and_redraw)
        self.HighPass_checkBox.toggled.connect(self.update_and_redraw)
        self.MeanSubtract_checkBox.toggled.connect(self.update_and_redraw)
        self.InvertSignal_checkBox.toggled.connect(self.update_and_redraw)


        # --- Wire sweep navigation ---
        self.PrevSweepButton.clicked.connect(self.on_prev_sweep)
        self.NextSweepButton.clicked.connect(self.on_next_sweep)
        self.SnaptoSweepButton.clicked.connect(self.on_snap_to_sweep)
        

        # --- Wire window navigation ---
        self.PrevWindowButton.clicked.connect(self.on_prev_window)
        self.NextWindowButton.clicked.connect(self.on_next_window)
        self.SnaptoWindowButton.clicked.connect(self.on_snap_to_window)
        self.WindowRangeValue.setText("20")# Default window length
        # overlap settings for window stepping
        self._win_overlap_frac = 0.10   # 10% of the window length
        self._win_min_overlap_s = 0.50  # but at least 0.5 s overlap

        self._win_left = None# Track current window left edge (in "display time" coordinates)


        # --- Peak-detect UI wiring ---
        self.ApplyPeakFindPushButton.setEnabled(False)  # stays disabled until threshold typed
        self.ApplyPeakFindPushButton.clicked.connect(self.on_apply_peak_find_clicked)
        # --- live threshold line on the main plot ---
        self._threshold_value = None
        self._thresh_line_artists = []
        self.ThreshVal.textChanged.connect(self._on_thresh_text_changed)


        self.ThreshVal.textChanged.connect(self._maybe_enable_peak_apply)
        self.PeakPromValue.textChanged.connect(self._maybe_enable_peak_apply)
        self.MinPeakDistValue.textChanged.connect(self._maybe_enable_peak_apply)
        self.PeakDetectionDirection.currentIndexChanged.connect(self._maybe_enable_peak_apply)

        # Default for refractory period / min peak distance (seconds)
        self.MinPeakDistValue.setText("0.05")

        # --- y2 metric dropdown (choices only; plotting later) ---
        self.y2plot_dropdown.clear()
        self.state.y2_values_by_sweep.clear()
        self.plot_host.clear_y2()

        self.y2plot_dropdown.addItem("None", userData=None)
        for label, key in metrics.METRIC_SPECS:
            self.y2plot_dropdown.addItem(label, userData=key)

        # ADD/DELETE Peak Mode: track selection in state
        self.state.y2_metric_key = None
        self.y2plot_dropdown.currentIndexChanged.connect(self.on_y2_metric_changed)

        self._add_peaks_mode = False
        self._delete_peaks_mode = False
        self._peak_edit_half_win_s = 0.08

        self.addPeaksButton.setCheckable(True)
        self.deletePeaksButton.setCheckable(True)
        self.addPeaksButton.toggled.connect(self.on_add_peaks_toggled)
        self.deletePeaksButton.toggled.connect(self.on_delete_peaks_toggled)

        #wire save analyzed data button
        self.SaveAnalyzedDataButton.clicked.connect(self.on_save_analyzed_clicked)



        # Defaults: 0.5–20 Hz band, all off initially
        self.HighPassVal.setText("0.5")
        self.LowPassVal.setText("20")
        self.HighPass_checkBox.setChecked(False)
        self.LowPass_checkBox.setChecked(True)
        self.MeanSubtract_checkBox.setChecked(False)
        self.InvertSignal_checkBox.setChecked(False)

        # Push defaults into state (no-op if no data yet)
        self.update_and_redraw()

    # ---------- File browse ----------
    def on_browse_clicked(self):
        last_dir = self.settings.value("last_dir", str(Path.home()))
        if not Path(str(last_dir)).exists():
            last_dir = str(Path.home())

        path, _ = QFileDialog.getOpenFileName(
            self, "Select ABF", last_dir, "ABF (*.abf);;All Files (*.*)"
        )
        if not path:
            return
        self.settings.setValue("last_dir", str(Path(path).parent))
        self.BrowseFilePath.setText(path)
        self.load_file(Path(path))

    def load_file(self, path: Path):
        try:
            sr, sweeps_by_ch, ch_names, t = abf_io.load_abf(path)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return


        st = self.state
        st.in_path = path
        st.sr_hz = sr
        st.sweeps = sweeps_by_ch
        st.channel_names = ch_names
        st.t = t
        st.sweep_idx = 0
        self._win_left = None

        # Reset peak results
        if not hasattr(st, "peaks_by_sweep"):
            st.peaks_by_sweep = {}
            st.peaks_by_sweep = {}
            st.breath_by_sweep = {}
        else:
            st.peaks_by_sweep.clear()


        # Reset Apply button and its enable logic
        self.ApplyPeakFindPushButton.setEnabled(False)
        self._maybe_enable_peak_apply()


        


        # Fill combos safely (no signal during population)
        self.AnalyzeChanSelect.blockSignals(True)
        self.AnalyzeChanSelect.clear()
        self.AnalyzeChanSelect.addItems(ch_names)
        self.AnalyzeChanSelect.setCurrentIndex(0)  # default analyze = first channel
        self.AnalyzeChanSelect.blockSignals(False)

        self.StimChanSelect.blockSignals(True)
        self.StimChanSelect.clear()
        self.StimChanSelect.addItem("None")        # default
        self.StimChanSelect.addItems(ch_names)
        self.StimChanSelect.setCurrentIndex(0)     # select "None"
        self.StimChanSelect.blockSignals(False)

        # Reset deferred state & button
        self._pending_analyze_idx = None
        self._pending_stim_idx = None
        # self.ApplyChanPushButton.setEnabled(False)
        self._update_apply_button_enabled()

        #Clear peaks
        self.state.peaks_by_sweep.clear()



        # Default analyze channel: first
        if ch_names:
            st.analyze_chan = ch_names[0]

        # No stim selected by default
        st.stim_chan = None
        st.stim_onsets_by_sweep.clear()
        st.stim_offsets_by_sweep.clear()
        st.stim_spans_by_sweep.clear()
        st.stim_metrics_by_sweep.clear()

        # Start in multi-panel (all channels) view
        self.single_panel_mode = False
        self.plot_host.clear_saved_view("grid")  # fresh autoscale for grid
        self.plot_all_channels()

        # start with Apply enabled so the user can jump to single panel immediately
        self.ApplyChanPushButton.setEnabled(True)

    def _proc_key(self, chan: str, sweep: int):
        st = self.state
        return (
            chan, sweep,
            st.use_low,  st.low_hz,
            st.use_high, st.high_hz,
            st.use_mean_sub, st.mean_val,
            st.use_invert
        )

    def plot_all_channels(self):
        """Plot the current sweep for every channel, one panel per channel."""
        st = self.state
        if not st.channel_names or st.t is None:
            return
        s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))

        traces = []
        for ch_name in st.channel_names:
            Y = st.sweeps[ch_name]           # (n_samples, n_sweeps)
            y = Y[:, s]                      # current sweep (1D)
            y_proc = filters.apply_all_1d(
                y, st.sr_hz,
                st.use_low,  st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert
            )
            traces.append((st.t, y_proc, ch_name))

        self.plot_host.show_multi_grid(
            traces,
            title=f"All channels | sweep {s+1}",
            max_points_per_trace=6000
        )

    def on_analyze_channel_changed(self, idx: int):
        # record pending selection; enable Apply
        self._pending_analyze_idx = idx
        self.ApplyChanPushButton.setEnabled(True)

    def on_stim_channel_changed(self, idx: int):
        # record pending selection; enable Apply
        self._pending_stim_idx = idx
        self.ApplyChanPushButton.setEnabled(True)

    def _update_apply_button_enabled(self):
        st = self.state
        has_data = bool(st.channel_names)
        has_analyze = has_data and (st.analyze_chan in st.channel_names)

        changed = False

        # pending analyze differs?
        if self._pending_analyze_idx is not None and 0 <= self._pending_analyze_idx < len(st.channel_names):
            current_an_idx = st.channel_names.index(st.analyze_chan) if st.analyze_chan in st.channel_names else -1
            if self._pending_analyze_idx != current_an_idx:
                changed = True

        # pending stim differs? (UI idx: 0 = None)
        if self._pending_stim_idx is not None:
            current_stim_ui_idx = 0 if (st.stim_chan is None) else (st.channel_names.index(st.stim_chan) + 1)
            if self._pending_stim_idx != current_stim_ui_idx:
                changed = True

        # Enable if: we have an analyze channel AND (we're still in grid mode OR there are pending changes)
        enable = has_analyze and ((not self.single_panel_mode) or changed)
        self.ApplyChanPushButton.setEnabled(enable)

    def on_apply_channels_clicked(self):
        st = self.state
        something_changed = False
        mode_changed = False  # grid <-> single

        # ---- Apply analyze channel (if pending) ----
        if self._pending_analyze_idx is not None and 0 <= self._pending_analyze_idx < len(st.channel_names):
            new_an = st.channel_names[self._pending_analyze_idx]
            if new_an != st.analyze_chan:
                st.analyze_chan = new_an
                st.proc_cache.clear()
                something_changed = True

        # ---- Apply stim channel (if pending) ----
        if self._pending_stim_idx is not None:
            idx = self._pending_stim_idx
            if idx <= 0:
                # "None" → clear stim, but go/stay in single-panel without spans
                if st.stim_chan is not None:
                    st.stim_chan = None
                    st.stim_onsets_by_sweep.clear()
                    st.stim_offsets_by_sweep.clear()
                    st.stim_spans_by_sweep.clear()
                    st.stim_metrics_by_sweep.clear()
                    something_changed = True
                if not self.single_panel_mode:
                    self.single_panel_mode = True
                    mode_changed = True
            else:
                if 1 <= idx <= len(st.channel_names):
                    new_stim = st.channel_names[idx - 1]
                    if (new_stim != st.stim_chan) or (not self.single_panel_mode):
                        st.stim_chan = new_stim
                        self._compute_stim_for_current_sweep()
                        if not self.single_panel_mode:
                            self.single_panel_mode = True
                            mode_changed = True
                        st.proc_cache.clear()
                        something_changed = True

        # If nothing changed but we’re still in grid mode, switch to single-panel
        if not something_changed and not self.single_panel_mode:
            self.single_panel_mode = True
            mode_changed = True
            something_changed = True

        # ALWAYS reset x-range on Apply (full autoscale next draw)
        # We’ll be in single-panel after Apply in all the flows above.
        self.plot_host.clear_saved_view("single")

        # Clear pending and disable Apply until there’s another change
        self._pending_analyze_idx = None
        self._pending_stim_idx = None
        self.ApplyChanPushButton.setEnabled(False)

        if something_changed or mode_changed:
            self.redraw_main_plot()

    def _compute_stim_for_current_sweep(self, thresh: float = 1.0):
        st = self.state
        if not st.stim_chan or st.stim_chan not in st.sweeps:
            return
        Y = st.sweeps[st.stim_chan]
        s = max(0, min(st.sweep_idx, Y.shape[1] - 1))
        y = Y[:, s]
        t = st.t

        on_idx, off_idx, spans_s, metrics = stimdet.detect_threshold_crossings(y, t, thresh=thresh)
        st.stim_onsets_by_sweep[s] = on_idx
        st.stim_offsets_by_sweep[s] = off_idx
        st.stim_spans_by_sweep[s] = spans_s
        st.stim_metrics_by_sweep[s] = metrics

        # Debug print
        if metrics:
            pw = metrics.get("pulse_width_s")
            dur = metrics.get("duration_s")
            hz = metrics.get("freq_hz")
            msg = f"[stim] sweep {s}: width={pw:.6f}s, duration={dur:.6f}s"
            if hz:
                msg += f", freq={hz:.3f}Hz"
            print(msg)

    # ---------- Filters & redraw ----------
    def update_and_redraw(self, *args):
        st = self.state

        # checkboxes
        st.use_low       = self.LowPass_checkBox.isChecked()
        st.use_high      = self.HighPass_checkBox.isChecked()
        st.use_mean_sub  = self.MeanSubtract_checkBox.isChecked()
        st.use_invert    = self.InvertSignal_checkBox.isChecked()


        # Peaks/breaths no longer valid if filters change
        if hasattr(self.state, "peaks_by_sweep"):
            self.state.peaks_by_sweep.clear()
        if hasattr(self.state, "breath_by_sweep"):
            self.state.breath_by_sweep.clear()
        self._maybe_enable_peak_apply()  # re-enable Apply Peak if threshold is valid

        # Peaks/breaths/y2 no longer valid if filters change
        if hasattr(self.state, "peaks_by_sweep"):
            self.state.peaks_by_sweep.clear()
        if hasattr(self.state, "breath_by_sweep"):
            self.state.breath_by_sweep.clear()
            self.state.y2_values_by_sweep.clear()
            self.plot_host.clear_y2()



        def _val_if_enabled(line, checked: bool, cast=float, default=None):
            """Return a numeric value only if the box is checked and a value exists."""
            if not checked:
                return None
            txt = line.text().strip()
            if not txt:
                return None
            try:
                return cast(txt)
            except ValueError:
                return None

        # only take values if box is checked AND entry is valid
        st.low_hz  = _val_if_enabled(self.LowPassVal, st.use_low, float, None)
        st.high_hz = _val_if_enabled(self.HighPassVal, st.use_high, float, None)
        st.mean_val= _val_if_enabled(self.MeanSubractVal, st.use_mean_sub, float, 0.0)

        # If the checkbox is checked but the box is empty/invalid, disable that filter automatically
        if st.use_low and st.low_hz is None:
            st.use_low = False
        if st.use_high and st.high_hz is None:
            st.use_high = False
        if st.use_mean_sub and st.mean_val is None:
            st.use_mean_sub = False

        # Invalidate processed cache
        st.proc_cache.clear()

        # Debounce redraw
        self._redraw_timer.start()

    def _current_trace(self):
        """Return (t, y_proc) for analyze channel & current sweep, using cached processing."""
        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return None, None

        Y = st.sweeps[st.analyze_chan]
        s = max(0, min(st.sweep_idx, Y.shape[1] - 1))
        key = self._proc_key(st.analyze_chan, s)

        # Fast path: reuse processed data if settings didn't change
        if key in st.proc_cache:
            return st.t, st.proc_cache[key]

        # Compute once, cache, and return
        y = Y[:, s]
        y2 = filters.apply_all_1d(
            y, st.sr_hz,
            st.use_low,  st.low_hz,
            st.use_high, st.high_hz,
            st.use_mean_sub, st.mean_val,
            st.use_invert
        )
        st.proc_cache[key] = y2
        return st.t, y2

    def redraw_main_plot(self):
        st = self.state
        if st.t is None:
            return

        if self.single_panel_mode:
            t, y = self._current_trace()
            if t is None:
                return
            s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))
            spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []

            # Normalize time to first stim onset if available
            if st.stim_chan and spans:
                t0 = spans[0][0]
                t_plot = t - t0
                spans_plot = [(a - t0, b - t0) for (a, b) in spans]
            else:
                t0 = 0.0
                t_plot = t
                spans_plot = spans

            # base trace
            self.plot_host.show_trace_with_spans(
                t_plot, y, spans_plot,
                title=f"{st.analyze_chan or ''} | sweep {s+1}",
                max_points=None
            )

            # Overlay peaks for the current sweep (if computed)
            pks = getattr(st, "peaks_by_sweep", {}).get(s, None)
            if pks is not None and len(pks):
                # Use the SAME time axis you plotted (t_plot) so indices align
                t_peaks = t_plot[pks]
                y_peaks = y[pks]
                self.plot_host.update_peaks(t_peaks, y_peaks, size=20)
            else:
                # ensure old scatter is gone if no peaks for this sweep
                self.plot_host.clear_peaks()

            # Overlay peaks for the current sweep (if computed)
            pks = getattr(st, "peaks_by_sweep", {}).get(s, None)
            if pks is not None and len(pks):
                t_peaks = t_plot[pks]
                y_peaks = y[pks]
                self.plot_host.update_peaks(t_peaks, y_peaks, size=24)
            else:
                self.plot_host.clear_peaks()

            # Overlay breath markers (if computed)
            br = getattr(st, "breath_by_sweep", {}).get(s, None)
            if br:
                on_idx  = br.get("onsets",  [])
                off_idx = br.get("offsets", [])
                ex_idx  = br.get("expmins", [])
                exoff_idx= br.get("expoffs", [])

                t_on  = t_plot[on_idx]  if len(on_idx)  else None
                y_on  = y[on_idx]       if len(on_idx)  else None
                t_off = t_plot[off_idx] if len(off_idx) else None
                y_off = y[off_idx]      if len(off_idx) else None
                t_exp = t_plot[ex_idx]  if len(ex_idx)  else None
                y_exp = y[ex_idx]       if len(ex_idx)  else None
                t_exof = t_plot[exoff_idx] if len(exoff_idx) else None
                y_exof = y[exoff_idx]      if len(exoff_idx) else None

                self.plot_host.update_breath_markers(
                    t_on=t_on, y_on=y_on,
                    t_off=t_off, y_off=y_off,
                    t_exp=t_exp, y_exp=y_exp,
                    t_exoff=t_exof, y_exoff=y_exof,
                    size=36
                )
            else:
                self.plot_host.clear_breath_markers()
            
            # ---- Y2 metric (if selected and available for this sweep) ----
            key = getattr(st, "y2_metric_key", None)
            if key:
                arr = st.y2_values_by_sweep.get(s, None)
                if arr is not None and len(arr) == len(t):
                    # Use the same time axis (t_plot) you've just plotted
                    # label = "IF (breaths/min)" if key == "if" else key
                    label = "IF (Hz)" if key == "if" else key
                    # self.plot_host.add_or_update_y2(t_plot, arr, label=label, max_points=None)
                    self.plot_host.add_or_update_y2(t_plot, arr, label=label, color="#39FF14", max_points=None)
                else:
                    self.plot_host.clear_y2()
            else:
                self.plot_host.clear_y2()
            self._refresh_threshold_lines()


    def _maybe_enable_peak_apply(self):
        """
        Enable ApplyPeakFindPushButton if there's a numeric threshold AND we have data.
        We keep it simple: whenever user edits any peak param, allow pressing Apply again.
        """
        st = self.state
        has_data = bool(st.channel_names)
        txt = self.ThreshVal.text().strip()
        try:
            float(txt)
            ok_thresh = True
        except Exception:
            ok_thresh = False

        self.ApplyPeakFindPushButton.setEnabled(has_data and ok_thresh)

    ##################################################
    ##threshold plotting                            ##
    ##################################################
    def _on_thresh_text_changed(self, *_):
        """
        Called whenever the ThreshVal text changes.
        Parses the float (or clears if invalid) and refreshes the dashed line(s).
        """
        self._threshold_value = self._parse_float(self.ThreshVal)  # None if invalid/empty
        self._refresh_threshold_lines()

    def _refresh_threshold_lines(self):
        """
        (Re)draw the dashed threshold line on the current visible axes.
        Called after typing in ThreshVal and after redraws that clear the axes.
        """
        fig = getattr(self.plot_host, "fig", None)
        canvas = getattr(self.plot_host, "canvas", None)
        if fig is None or canvas is None or not fig.axes:
            return

        # Remove previous lines if they exist
        for ln in getattr(self, "_thresh_line_artists", []):
            try:
                ln.remove()
            except Exception:
                pass
        self._thresh_line_artists = []

        y = getattr(self, "_threshold_value", None)
        if y is None:
            # Nothing to draw
            canvas.draw_idle()
            return

        # Decide which axes to draw on:
        # - In single-panel mode, draw only on the first axes (main plot).
        # - In grid mode, you can choose all axes or just the first; here we do first only.
        axes = fig.axes[:1] if getattr(self, "single_panel_mode", False) else fig.axes[:1]

        for ax in axes:
            # line = ax.axhline(
            #     y,
            #     linestyle=(0, (5, 5)),  # dashed
            #     linewidth=1.2,
            #     alpha=0.9,
            #     zorder=5,
            # )
            line = ax.axhline(
                y,
                color="red",              # make it red
                linewidth=1.0,            # a touch thinner (optional)
                linestyle=(0, (2, 2)),    # smaller dash pattern: 2 on, 2 off
                alpha=0.95,
                zorder=5,
            )

            self._thresh_line_artists.append(line)

        canvas.draw_idle()


    ##################################################
    ##Sweep navigation                              ##
    ##################################################
    def _num_sweeps(self) -> int:
        """Return total sweep count from the first channel (0 if no data)."""
        st = self.state
        if not st.sweeps:
            return 0
        first = next(iter(st.sweeps.values()))
        return int(first.shape[1]) if first is not None else 0

    def on_prev_sweep(self):
        st = self.state
        n = self._num_sweeps()
        if n == 0:
            return
        if st.sweep_idx > 0:
            st.sweep_idx -= 1
            # recompute stim spans for this sweep if a stim channel is selected
            if st.stim_chan:
                self._compute_stim_for_current_sweep()
            self.redraw_main_plot()

    def on_next_sweep(self):
        st = self.state
        n = self._num_sweeps()
        if n == 0:
            return
        if st.sweep_idx < n - 1:
            st.sweep_idx += 1
            if st.stim_chan:
                self._compute_stim_for_current_sweep()
            self.redraw_main_plot()

    def on_snap_to_sweep(self):
        # clear saved zoom so the next draw autoscales to full sweep range
        self.plot_host.clear_saved_view("single" if self.single_panel_mode else "grid")
        self.redraw_main_plot()

    ##################################################
    ##Window navigation (relative to current window)##
    ##################################################
    def _parse_window_seconds(self) -> float:
        """Read WindowRangeValue (seconds). Returns a positive float, default 20."""
        try:
            val = float(self.WindowRangeValue.text().strip())
            if val > 0:
                return val
        except Exception:
            pass
        return 20.0

    def _window_step(self, W: float) -> float:
        """
        Step size when paging windows: W - overlap,
        where overlap is max(min_overlap, frac * W).
        """
        overlap = max(self._win_min_overlap_s, self._win_overlap_frac * W)
        step = max(0.0, W - overlap)
        # avoid zero step for tiny windows
        if step <= 0:
            step = 0.9 * W
        return step

    def on_snap_to_window(self):
        """Jump to start of current sweep (normalized domain if applicable)."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return
        W = self._parse_window_seconds()
        left = float(t[0])
        self._set_window(left=left, width=W)

    # def on_next_window(self):
    #     """Step forward; if at end of sweep, go to first window of next sweep."""
    #     t = self._current_t_plot()
    #     if t is None or t.size == 0:
    #         return
    #     W = self._parse_window_seconds()
    #     step = self._window_step(W)

    #     # initialize from current axes if needed
    #     if self._win_left is None:
    #         ax = self.plot_host.fig.axes[0] if self.plot_host.fig.axes else None
    #         self._win_left = float(ax.get_xlim()[0]) if ax else float(t[0])

    #     max_left = float(t[-1]) - W
    #     # normal step within this sweep?
    #     if self._win_left + step <= max_left + 1e-12:
    #         self._set_window(left=self._win_left + step, width=W)
    #         return

    #     # otherwise hop to next sweep if available
    #     s_count = self._sweep_count()
    #     if self.state.sweep_idx < s_count - 1:
    #         self.state.sweep_idx += 1
    #         # compute stim spans for the new sweep if a stim channel is set
    #         if self.state.stim_chan:
    #             self._compute_stim_for_current_sweep()

    #         # redraw single/grid view (kept consistent with your logic)
    #         self.redraw_main_plot()

    #         # new sweep domain
    #         t2 = self._current_t_plot()
    #         if t2 is None or t2.size == 0:
    #             return
    #         self._set_window(left=float(t2[0]), width=W)
    #     else:
    #         # clamp to last window of this (final) sweep
    #         last_left = max(float(t[0]), max_left)
    #         self._set_window(left=last_left, width=W)

    # def on_prev_window(self):
        """Step backward; if at start of sweep, go to last window of previous sweep."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return
        W = self._parse_window_seconds()
        step = self._window_step(W)

        # initialize from current axes if needed
        if self._win_left is None:
            ax = self.plot_host.fig.axes[0] if self.plot_host.fig.axes else None
            self._win_left = float(ax.get_xlim()[0]) if ax else float(t[0])

        min_left = float(t[0])
        # normal step within this sweep?
        if self._win_left - step >= min_left - 1e-12:
            self._set_window(left=self._win_left - step, width=W)
            return

        # otherwise hop to previous sweep if available
        if self.state.sweep_idx > 0:
            self.state.sweep_idx -= 1
            # compute stim spans for the new sweep if a stim channel is set
            if self.state.stim_chan:
                self._compute_stim_for_current_sweep()

            self.redraw_main_plot()

            # new sweep domain
            t2 = self._current_t_plot()
            if t2 is None or t2.size == 0:
                return
            # last window of previous sweep
            last_left = max(float(t2[0]), float(t2[-1]) - W)
            self._set_window(left=last_left, width=W)
        else:
            # clamp to first window of this (first) sweep
            self._set_window(left=min_left, width=W)

    def on_next_window(self):
        """Step forward; if stepping past end, first show the last full window once,
        then on the next press hop to the first window of the next sweep."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return

        W = self._parse_window_seconds()
        step = self._window_step(W)

        # Initialize left edge if needed
        if self._win_left is None:
            ax = self.plot_host.fig.axes[0] if self.plot_host.fig.axes else None
            self._win_left = float(ax.get_xlim()[0]) if ax else float(t[0])

        # Use an effective width that never exceeds this sweep's duration
        dur = float(t[-1] - t[0])
        W_eff = min(W, max(1e-6, dur))
        max_left = float(t[-1]) - W_eff
        eps = 1e-9

        # Normal step within this sweep?
        if self._win_left + step <= max_left + eps:
            self._set_window(left=self._win_left + step, width=W_eff)
            return

        # Not enough room for a full step:
        # 1) If we're not yet at the last full window, show it once.
        if self._win_left < max_left - eps:
            self._set_window(left=max_left, width=W_eff)
            return

        # 2) Already at last full window -> hop to next sweep if possible
        s_count = self._sweep_count()
        if self.state.sweep_idx < s_count - 1:
            self.state.sweep_idx += 1
            if self.state.stim_chan:
                self._compute_stim_for_current_sweep()
            self.redraw_main_plot()

            t2 = self._current_t_plot()
            if t2 is None or t2.size == 0:
                return
            dur2 = float(t2[-1] - t2[0])
            W_eff2 = min(W, max(1e-6, dur2))
            self._set_window(left=float(t2[0]), width=W_eff2)
        else:
            # No next sweep: stay clamped at the last full window
            self._set_window(left=max_left, width=W_eff)

    def on_prev_window(self):
        """Step backward; if stepping before start, first show the first full window once,
        then on the next press hop to the last window of the previous sweep."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return

        W = self._parse_window_seconds()
        step = self._window_step(W)

        if self._win_left is None:
            ax = self.plot_host.fig.axes[0] if self.plot_host.fig.axes else None
            self._win_left = float(ax.get_xlim()[0]) if ax else float(t[0])

        dur = float(t[-1] - t[0])
        W_eff = min(W, max(1e-6, dur))
        min_left = float(t[0])
        eps = 1e-9

        # Normal step within this sweep?
        if self._win_left - step >= min_left - eps:
            self._set_window(left=self._win_left - step, width=W_eff)
            return

        # Not enough room for a full step:
        # 1) If we're not yet at the first full window, show it once.
        if self._win_left > min_left + eps:
            self._set_window(left=min_left, width=W_eff)
            return

        # 2) Already at first window -> hop to previous sweep if possible
        if self.state.sweep_idx > 0:
            self.state.sweep_idx -= 1
            if self.state.stim_chan:
                self._compute_stim_for_current_sweep()
            self.redraw_main_plot()

            t2 = self._current_t_plot()
            if t2 is None or t2.size == 0:
                return
            dur2 = float(t2[-1] - t2[0])
            W_eff2 = min(W, max(1e-6, dur2))
            last_left = max(float(t2[0]), float(t2[-1]) - W_eff2)
            self._set_window(left=last_left, width=W_eff2)
        else:
            # No previous sweep: stay clamped at the first window
            self._set_window(left=min_left, width=W_eff)

    def _sweep_count(self) -> int:
        st = self.state
        if not st.sweeps:
            return 0
        any_ch = next(iter(st.sweeps.values()))
        return any_ch.shape[1]

    def _current_t_plot(self):
        """Time axis exactly like the one used in redraw (normalized if stim spans exist)."""
        st = self.state
        if st.t is None:
            return None
        s = max(0, min(st.sweep_idx, self._sweep_count()-1))
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            return st.t - t0
        return st.t

    def _set_window(self, left: float, width: float):
        """Apply x-limits and remember left edge for subsequent steps."""
        ax = self.plot_host.fig.axes[0] if self.plot_host.fig.axes else None
        if ax is None:
            return
        right = left + max(0.01, float(width))
        ax.set_xlim(left, right)
        self._win_left = float(left)
        self.plot_host.canvas.draw_idle()

    ##################################################
    ##Window navigation (relative to current window)##
    ##################################################
    def _parse_float(self, line_edit):
        txt = line_edit.text().strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            return None

    def _on_peak_param_changed(self, *args):
        """Enable Apply if threshold is a valid number and we have data."""
        th = self._parse_float(self.ThreshVal)
        has_data = (self.state.t is not None) and (self.state.analyze_chan in (self.state.sweeps or {}))
        self.ApplyPeakFindPushButton.setEnabled(th is not None and has_data)

    # def on_apply_peak_find_clicked(self):
    #     st = self.state

    #     # must have data & an analyze channel
    #     if st.t is None or st.analyze_chan not in st.sweeps:
    #         self.ApplyPeakFindPushButton.setEnabled(False)
    #         return

    #     # get current processed trace (what user sees)
    #     t, y = self._current_trace()
    #     if t is None or y is None:
    #         self.ApplyPeakFindPushButton.setEnabled(False)
    #         return

    #     # read params
    #     thresh = self._parse_float(self.ThreshVal)  # REQUIRED (button only enables if valid)
    #     prom   = self._parse_float(self.PeakPromValue)           # optional
    #     mind_s = self._parse_float(self.MinPeakDistValue)        # optional (seconds)
    #     mind_n = int(round(mind_s * st.sr_hz)) if (mind_s is not None and st.sr_hz) else None
    #     direction = (self.PeakDetectionDirection.currentText() or "Up").strip().lower()
    #     if direction not in ("up", "down"):
    #         direction = "up"

    #     # compute peaks
    #     import numpy as np
    #     s = max(0, min(st.sweep_idx, st.sweeps[st.analyze_chan].shape[1] - 1))
    #     pks = peakdet.detect_peaks(
    #         y=np.asarray(y),
    #         sr_hz=st.sr_hz,
    #         thresh=thresh,
    #         prominence=prom,
    #         min_dist_samples=mind_n,
    #         direction=direction
    #     )

    #     # store per sweep
    #     st.peaks_by_sweep[s] = pks

    #     # draw (single panel only shows peaks)
    #     self.redraw_main_plot()

    #     # disable until params change again (grays out)
    #     self.ApplyPeakFindPushButton.setEnabled(False)

    def _get_processed_for(self, chan: str, sweep_idx: int):
        """Return processed y for (channel, sweep_idx) using the same cache key logic."""
        st = self.state
        Y = st.sweeps[chan]
        s = max(0, min(sweep_idx, Y.shape[1]-1))
        key = (chan, s, st.use_low, st.low_hz, st.use_high, st.high_hz, st.use_mean_sub, st.mean_val, st.use_invert)
        if key in st.proc_cache:
            return st.proc_cache[key]
        y = Y[:, s]
        y2 = filters.apply_all_1d(
            y, st.sr_hz,
            st.use_low,  st.low_hz,
            st.use_high, st.high_hz,
            st.use_mean_sub, st.mean_val,
            st.use_invert
        )
        st.proc_cache[key] = y2
        return y2

    
    def on_apply_peak_find_clicked(self):
        """
        Run peak detection on the ANALYZE channel for ALL sweeps,
        store indices per sweep, and redraw current sweep with peaks + breath markers.
        """
        st = self.state
        if not st.channel_names or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            self.ApplyPeakFindPushButton.setEnabled(False)
            return

        # Parse UI parameters
        def _num(line):
            txt = line.text().strip()
            return float(txt) if txt else None

        thresh = _num(self.ThreshVal)                 # required (button enabled only if valid)
        prom   = _num(self.PeakPromValue)             # optional
        min_d  = _num(self.MinPeakDistValue)          # seconds
        direction = (self.PeakDetectionDirection.currentText() or "up").strip().lower()
        if direction not in ("up", "down"):
            direction = "up"

        min_dist_samples = None
        if min_d is not None and min_d > 0:
            min_dist_samples = max(1, int(round(min_d * st.sr_hz)))

        # Detect on ALL sweeps for the analyze channel
        any_chan = next(iter(st.sweeps.values()))
        n_sweeps = any_chan.shape[1]
        st.peaks_by_sweep.clear()
        st.breath_by_sweep.clear()

        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            pks, breaths = peakdet.detect_peaks_and_breaths(
                y=y_proc, sr_hz=st.sr_hz,
                thresh=thresh,
                prominence=prom,
                min_dist_samples=min_dist_samples,
                direction=direction,
            )
            st.peaks_by_sweep[s] = pks
            st.breath_by_sweep[s] = breaths  # dict with 'onsets', 'offsets', 'expmins'

        # Disable until parameters change again and redraw current sweep
        self.ApplyPeakFindPushButton.setEnabled(False)
        # If a Y2 metric is selected, recompute it now that peaks/breaths changed
        if getattr(self.state, "y2_metric_key", None):
            self._compute_y2_all_sweeps()
            self.plot_host.clear_y2()

        self.redraw_main_plot()


    ##################################################
    ##Window navigation (relative to current window)##
    ##################################################
    def on_y2_metric_changed(self, idx: int):
        key = self.y2plot_dropdown.itemData(idx)
        self.state.y2_metric_key = key  # None or one of metrics.METRICS keys
        # Next step (later): compute metric for current sweep(s) and draw on y2 axis


    def _compute_y2_all_sweeps(self):
        """Compute active y2 metric for ALL sweeps on the analyze channel."""
        st = self.state
        key = getattr(st, "y2_metric_key", None)
        if not key:
            st.y2_values_by_sweep.clear()
            return
        if key not in metrics.METRICS:
            st.y2_values_by_sweep.clear()
            return
        if st.t is None or st.analyze_chan not in st.sweeps:
            st.y2_values_by_sweep.clear()
            return

        fn = metrics.METRICS[key]
        any_ch = next(iter(st.sweeps.values()))
        n_sweeps = any_ch.shape[1]
        st.y2_values_by_sweep = {}

        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            # pull peaks/breaths if available
            pks = getattr(st, "peaks_by_sweep", {}).get(s, None)
            # breaths = getattr(st, "breath_by_sweep", {}).get(s, {}) if hasattr(st, "breath_by_sweep") else {}
            breaths = getattr(st, "breath_by_sweep", {}).get(s, {})
            on = breaths.get("onsets", None)
            off = breaths.get("offsets", None)
            exm = breaths.get("expmins", None)
            exo = breaths.get("expoffs", None)

            y2 = fn(st.t, y_proc, st.sr_hz, pks, on, off, exm, exo)
            st.y2_values_by_sweep[s] = y2

    def on_y2_metric_changed(self, idx: int):
        key = self.y2plot_dropdown.itemData(idx)
        self.state.y2_metric_key = key  # None or e.g. "if"

        # Recompute Y2 (needs peaks/breaths for most metrics; IF falls back to peaks)
        self._compute_y2_all_sweeps()

        # Force a redraw of current sweep
        # Also reset Y2 axis so it rescales to new data
        self.plot_host.clear_y2()
        self.redraw_main_plot()

    ##################################################
    ##ADD Peaks Button##
    ##################################################
    # def on_add_peaks_toggled(self, checked: bool):
    #     """Enter/exit Add Peaks mode."""
    #     self._add_peaks_mode = checked

    #     if checked:
    #         # Visual hint
    #         self.addPeaksButton.setText("Add Peaks (ON)")
    #         # Send plot clicks to our handler
    #         self.plot_host.set_click_callback(self._on_plot_click_add_peak)
    #         # Nice crosshair cursor while adding peaks
    #         self.plot_host.setCursor(Qt.CursorShape.CrossCursor)
    #     else:
    #         self.addPeaksButton.setText("Add Peaks")
    #         self.plot_host.clear_click_callback()
    #         self.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    # def on_add_peaks_toggled(self, checked: bool):
    #     """Enter/exit Add Peaks mode (mutually exclusive with Delete mode)."""
    #     self._add_peaks_mode = checked

    #     if checked:
    #         # Turn off delete mode if active
    #         if getattr(self, "_delete_peaks_mode", False):
    #             self._delete_peaks_mode = False
    #             self.deletePeaksButton.blockSignals(True)
    #             self.deletePeaksButton.setChecked(False)
    #             self.deletePeaksButton.blockSignals(False)

    #         self.addPeaksButton.setText("Add Peaks (ON)")
    #         self.plot_host.set_click_callback(self._on_plot_click_add_peak)
    #         self.plot_host.setCursor(Qt.CursorShape.CrossCursor)
    #     else:
    #         self.addPeaksButton.setText("Add Peaks")
    #         # Only clear callback if no other edit mode is active
    #         if not getattr(self, "_delete_peaks_mode", False):
    #             self.plot_host.clear_click_callback()
    #             self.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    def on_add_peaks_toggled(self, checked: bool):
        """Enter/exit Add Peaks mode, mutually exclusive with Delete mode."""
        self._add_peaks_mode = checked

        if checked:
            # Turn OFF delete mode visually and internally, without triggering its slot.
            if getattr(self, "_delete_peaks_mode", False):
                self._delete_peaks_mode = False
                self.deletePeaksButton.blockSignals(True)
                self.deletePeaksButton.setChecked(False)
                self.deletePeaksButton.blockSignals(False)
                self.deletePeaksButton.setText("Delete Peaks")

            self.addPeaksButton.setText("Add Peaks (ON)")
            self.plot_host.set_click_callback(self._on_plot_click_add_peak)
            self.plot_host.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.addPeaksButton.setText("Add Peaks")
            # Only clear callbacks/cursor if no other edit mode is active
            if not getattr(self, "_delete_peaks_mode", False):
                self.plot_host.clear_click_callback()
                self.plot_host.setCursor(Qt.CursorShape.ArrowCursor)


    def _on_plot_click_add_peak(self, xdata, ydata, event):
        # Only when "Add Peaks (ON)"
        def _on_plot_click_add_peak(self, xdata, ydata, event):
            # Only when "Add Peaks (ON)" and only left clicks
            if not getattr(self, "_add_peaks_mode", False) or getattr(event, "button", 1) != 1:
                return
            if event.inaxes is None or xdata is None:
                return
            ...

        if not getattr(self, "_add_peaks_mode", False):
            return
        if event.inaxes is None or xdata is None:
            return

        import numpy as np
        st = self.state
        if st.t is None or st.analyze_chan not in st.sweeps:
            return

        # Current sweep + processed trace (what user sees)
        s = max(0, min(st.sweep_idx, self._sweep_count() - 1))
        t, y = self._current_trace()
        if t is None or y is None:
            return

        # Use the same time basis as the plot (normalized if stim)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # ±080 ms search window around click
        half_win_s = 0.08
        half_win_n = max(1, int(round(half_win_s * st.sr_hz)))
        i_center = int(np.clip(np.searchsorted(t_plot, float(xdata)), 0, len(t_plot) - 1))
        i0 = max(0, i_center - half_win_n)
        i1 = min(len(y) - 1, i_center + half_win_n)
        if i1 <= i0:
            return

        # Local extremum depends on the detection direction
        direction = (self.PeakDetectionDirection.currentText() or "up").strip().lower()
        seg = y[i0:i1 + 1]
        loc = int(np.argmin(seg)) if direction == "down" else int(np.argmax(seg))
        i_peak = i0 + loc

        # ---- NEW: enforce minimum separation from existing peaks ----
        # Use UI "MinPeakDistValue" if valid; fallback to 0.05 s
        try:
            sep_s = float(self.MinPeakDistValue.text().strip())
            if not (sep_s > 0):
                raise ValueError
        except Exception:
            sep_s = 0.05
        sep_n = max(1, int(round(sep_s * st.sr_hz)))

        pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
        if pks.size and np.any(np.abs(pks - i_peak) <= sep_n):
            print(f"[add-peak] Rejected: candidate within {sep_s:.3f}s of an existing peak.")
            return

        # Insert, sort, and store
        pks_new = np.sort(np.append(pks, i_peak))
        st.peaks_by_sweep[s] = pks_new

        # Recompute breath markers with the updated peaks
        breaths = peakdet.compute_breath_events(y, pks_new, st.sr_hz)  # uses default deadband
        st.breath_by_sweep[s] = breaths

        # Recompute Y2 metric if selected
        if getattr(st, "y2_metric_key", None):
            self._compute_y2_all_sweeps()

        # Refresh plot
        self.redraw_main_plot()

    # def on_delete_peaks_toggled(self, checked: bool):
    #     """Enter/exit Delete Peaks mode (mutually exclusive with Add mode)."""
    #     self._delete_peaks_mode = checked

    #     if checked:
    #         # Turn off add mode if active
    #         if getattr(self, "_add_peaks_mode", False):
    #             self._add_peaks_mode = False
    #             self.addPeaksButton.blockSignals(True)
    #             self.addPeaksButton.setChecked(False)
    #             self.addPeaksButton.blockSignals(False)

    #         self.deletePeaksButton.setText("Delete Peaks (ON)")
    #         self.plot_host.set_click_callback(self._on_plot_click_delete_peak)
    #         self.plot_host.setCursor(Qt.CursorShape.CrossCursor)
    #     else:
    #         self.deletePeaksButton.setText("Delete Peaks")
    #         # Only clear callback if no other edit mode is active
    #         if not getattr(self, "_add_peaks_mode", False):
    #             self.plot_host.clear_click_callback()
    #             self.plot_host.setCursor(Qt.CursorShape.ArrowCursor)
    
    def on_delete_peaks_toggled(self, checked: bool):
        """Enter/exit Delete Peaks mode, mutually exclusive with Add mode."""
        self._delete_peaks_mode = checked

        if checked:
            # Turn OFF add mode visually and internally, without triggering its slot.
            if getattr(self, "_add_peaks_mode", False):
                self._add_peaks_mode = False
                self.addPeaksButton.blockSignals(True)
                self.addPeaksButton.setChecked(False)
                self.addPeaksButton.blockSignals(False)
                self.addPeaksButton.setText("Add Peaks")

            self.deletePeaksButton.setText("Delete Peaks (ON)")
            self.plot_host.set_click_callback(self._on_plot_click_delete_peak)
            self.plot_host.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.deletePeaksButton.setText("Delete Peaks")
            # Only clear callbacks/cursor if no other edit mode is active
            if not getattr(self, "_add_peaks_mode", False):
                self.plot_host.clear_click_callback()
                self.plot_host.setCursor(Qt.CursorShape.ArrowCursor)


    def _on_plot_click_delete_peak(self, xdata, ydata, event):
        # Only when "Delete Peaks (ON)" and only left clicks
        if not getattr(self, "_delete_peaks_mode", False) or getattr(event, "button", 1) != 1:
            return
        if event.inaxes is None or xdata is None:
            return

        st = self.state
        if st.t is None or st.analyze_chan not in st.sweeps:
            return

        # Current sweep & processed trace
        s = max(0, min(st.sweep_idx, self._sweep_count() - 1))
        t, y = self._current_trace()
        if t is None or y is None:
            return

        # Match the plotted time basis (normalized if stim spans exist)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # Same ±80 ms window you use for adding peaks
        half_win_s = float(getattr(self, "_peak_edit_half_win_s", 0.08))
        half_win_n = max(1, int(round(half_win_s * st.sr_hz)))

        i_center = int(np.clip(np.searchsorted(t_plot, float(xdata)), 0, len(t_plot) - 1))
        i0 = max(0, i_center - half_win_n)
        i1 = min(len(y) - 1, i_center + half_win_n)

        # Existing peaks for this sweep
        pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
        if pks.size == 0:
            print("[delete-peak] No peaks to delete in this sweep.")
            return

        # Delete any peaks inside [i0, i1] (inclusive)
        mask_keep = (pks < i0) | (pks > i1)
        removed = pks[~mask_keep]
        if removed.size == 0:
            print("[delete-peak] No peak within the click window.")
            return

        pks_new = pks[mask_keep]
        st.peaks_by_sweep[s] = pks_new

        # Recompute breaths with updated peaks (fallback to your available signature)
        try:
            breaths = peakdet.compute_breath_events(y, pks_new, st.sr_hz)
        except TypeError:
            breaths = peakdet.compute_breath_events(y, pks_new)  # older signature

        st.breath_by_sweep[s] = breaths

        # If a Y2 metric is selected, recompute
        if getattr(st, "y2_metric_key", None):
            self._compute_y2_all_sweeps()

        # Redraw
        self.redraw_main_plot()


    ##################################################
    ##Save Data to File                             ##
    ##################################################
    def _sanitize_token(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = s.replace(" ", "_")
        # allow alnum, underscore, hyphen, dot
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        # squeeze repeats of underscores/hyphens
        s = re.sub(r"_+", "_", s)
        s = re.sub(r"-+", "-", s)
        return s

    # class SaveMetaDialog(QDialog):
    #     # def __init__(self, abf_name: str, channel: str, parent=None):
    #     #     super().__init__(parent)
    #     #     self.setWindowTitle("Save analyzed data — name builder")

    #     #     self._abf_name = abf_name
    #     #     self._channel = channel

    #     #     lay = QFormLayout(self)

    #     #     self.le_stim = QLineEdit(self)
    #     #     self.le_stim.setPlaceholderText("e.g., 30Hz15s10ms or 25msPulse")
    #     #     lay.addRow("Stimulation type:", self.le_stim)

    #     #     self.le_power = QLineEdit(self)
    #     #     self.le_power.setPlaceholderText("e.g., 8mW")
    #     #     lay.addRow("Laser power:", self.le_power)

    #     #     self.cb_sex = QComboBox(self)
    #     #     self.cb_sex.addItems(["", "M", "F", "Unknown"])
    #     #     lay.addRow("Sex:", self.cb_sex)

    #     #     self.le_animal = QLineEdit(self)
    #     #     self.le_animal.setPlaceholderText("e.g., MouseA42")
    #     #     lay.addRow("Animal ID:", self.le_animal)

    #     #     # Read-only info
    #     #     self.lbl_abf = QLabel(abf_name, self)
    #     #     self.lbl_chn = QLabel(channel or "", self)
    #     #     lay.addRow("ABF file:", self.lbl_abf)
    #     #     lay.addRow("Channel:", self.lbl_chn)

    #     #     # Live preview
    #     #     self.lbl_preview = QLabel("", self)
    #     #     self.lbl_preview.setStyleSheet("color:#b6bfda;")
    #     #     lay.addRow("Preview:", self.lbl_preview)

    #     #     # Buttons
    #     #     btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
    #     #     lay.addRow(btns)
    #     #     btns.accepted.connect(self.accept)
    #     #     btns.rejected.connect(self.reject)

    #     #     # Update preview when fields change
    #     #     self.le_stim.textChanged.connect(self._update_preview)
    #     #     self.le_power.textChanged.connect(self._update_preview)
    #     #     self.cb_sex.currentTextChanged.connect(self._update_preview)
    #     #     self.le_animal.textChanged.connect(self._update_preview)

    #     #     self._update_preview()

    #     def __init__(self, abf_name: str, channel: str, parent=None, auto_stim: str = ""):
    #         super().__init__(parent)
    #         self.setWindowTitle("Save analyzed data — name builder")

    #         self._abf_name = abf_name
    #         self._channel = channel

    #         lay = QFormLayout(self)

    #         self.le_stim = QLineEdit(self)
    #         self.le_stim.setPlaceholderText("e.g., 30Hz15s10ms or 25msPulse")
    #         if auto_stim:
    #             self.le_stim.setText(auto_stim)  # <- prefill from detected stim
    #         lay.addRow("Stimulation type:", self.le_stim)

    #         self.le_power = QLineEdit(self)
    #         self.le_power.setPlaceholderText("e.g., 8mW")
    #         lay.addRow("Laser power:", self.le_power)

    #         self.cb_sex = QComboBox(self)
    #         self.cb_sex.addItems(["", "M", "F", "Unknown"])
    #         lay.addRow("Sex:", self.cb_sex)

    #         self.le_animal = QLineEdit(self)
    #         self.le_animal.setPlaceholderText("e.g., MouseA42")
    #         lay.addRow("Animal ID:", self.le_animal)

    #         # Read-only info
    #         self.lbl_abf = QLabel(abf_name, self)
    #         self.lbl_chn = QLabel(channel or "", self)
    #         lay.addRow("ABF file:", self.lbl_abf)
    #         lay.addRow("Channel:", self.lbl_chn)

    #         # Live preview
    #         self.lbl_preview = QLabel("", self)
    #         self.lbl_preview.setStyleSheet("color:#b6bfda;")
    #         lay.addRow("Preview:", self.lbl_preview)

    #         # Buttons
    #         btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
    #         lay.addRow(btns)
    #         btns.accepted.connect(self.accept)
    #         btns.rejected.connect(self.reject)

    #         # Update preview when fields change
    #         self.le_stim.textChanged.connect(self._update_preview)
    #         self.le_power.textChanged.connect(self._update_preview)
    #         self.cb_sex.currentTextChanged.connect(self._update_preview)
    #         self.le_animal.textChanged.connect(self._update_preview)

    #         self._update_preview()

    #     def _san(self, s: str) -> str:
    #         s = (s or "").strip()
    #         s = s.replace(" ", "_")
    #         s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    #         s = re.sub(r"_+", "_", s)
    #         s = re.sub(r"-+", "-", s)
    #         return s

    #     def _update_preview(self):
    #         stim   = self._san(self.le_stim.text())
    #         power  = self._san(self.le_power.text())
    #         sex    = self._san(self.cb_sex.currentText())
    #         animal = self._san(self.le_animal.text())
    #         abf    = self._san(self._abf_name)
    #         ch     = self._san(self._channel)

    #         parts = [p for p in (stim, power, sex, animal, abf, ch) if p]
    #         preview = "_".join(parts) if parts else "analysis"
    #         self.lbl_preview.setText(preview)

    #     def values(self) -> dict:
    #         return {
    #             "stim":   self.le_stim.text().strip(),
    #             "power":  self.le_power.text().strip(),
    #             "sex":    self.cb_sex.currentText().strip(),
    #             "animal": self.le_animal.text().strip(),
    #             "abf":    self._abf_name,
    #             "chan":   self._channel,
    #             "preview": self.lbl_preview.text().strip(),
    #         }

    # class SaveMetaDialog(QDialog):
    #     def __init__(self, abf_name: str, channel: str, parent=None, auto_stim: str = ""):
    #         super().__init__(parent)
    #         self.setWindowTitle("Save analyzed data — name builder")

    #         self._abf_name = abf_name
    #         self._channel = channel

    #         lay = QFormLayout(self)

    #         # NEW: Mouse Strain
    #         self.le_strain = QLineEdit(self)
    #         self.le_strain.setPlaceholderText("e.g., VgatCre")
    #         lay.addRow("Mouse Strain:", self.le_strain)

    #         # NEW: Virus
    #         self.le_virus = QLineEdit(self)
    #         self.le_virus.setPlaceholderText("e.g., ConFoff-ChR2")
    #         lay.addRow("Virus:", self.le_virus)

    #         # Stimulation type (can be auto-populated)
    #         self.le_stim = QLineEdit(self)
    #         self.le_stim.setPlaceholderText("e.g., 20Hz10s15ms or 15msPulse")
    #         if auto_stim:
    #             self.le_stim.setText(auto_stim)
    #         lay.addRow("Stimulation type:", self.le_stim)

    #         self.le_power = QLineEdit(self)
    #         self.le_power.setPlaceholderText("e.g., 8mW")
    #         lay.addRow("Laser power:", self.le_power)

    #         self.cb_sex = QComboBox(self)
    #         self.cb_sex.addItems(["", "M", "F", "Unknown"])
    #         lay.addRow("Sex:", self.cb_sex)

    #         self.le_animal = QLineEdit(self)
    #         self.le_animal.setPlaceholderText("e.g., 25121004")
    #         lay.addRow("Animal ID:", self.le_animal)

    #         # Read-only info
    #         self.lbl_abf = QLabel(abf_name, self)
    #         self.lbl_chn = QLabel(channel or "", self)
    #         lay.addRow("ABF file:", self.lbl_abf)
    #         lay.addRow("Channel:", self.lbl_chn)

    #         # Live preview
    #         self.lbl_preview = QLabel("", self)
    #         self.lbl_preview.setStyleSheet("color:#b6bfda;")
    #         lay.addRow("Preview:", self.lbl_preview)

    #         # Buttons
    #         btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
    #         lay.addRow(btns)
    #         btns.accepted.connect(self.accept)
    #         btns.rejected.connect(self.reject)

    #         # Update preview on change
    #         self.le_strain.textChanged.connect(self._update_preview)
    #         self.le_virus.textChanged.connect(self._update_preview)
    #         self.le_stim.textChanged.connect(self._update_preview)
    #         self.le_power.textChanged.connect(self._update_preview)
    #         self.cb_sex.currentTextChanged.connect(self._update_preview)
    #         self.le_animal.textChanged.connect(self._update_preview)

    #         self._update_preview()

    #     # --- Helpers: light canonicalization + sanitization ---
    #     def _norm_token(self, s: str) -> str:
    #         """
    #         Light canonicalization for common tokens:
    #         - 'chr2' -> 'ChR2'
    #         - 'gcamp6f' -> 'GCaMP6f'
    #         - '...cre' suffix -> '...Cre'
    #         Does NOT force all-caps or remove hyphens; just tidies typical cases.
    #         """
    #         s0 = (s or "").strip()
    #         if not s0:
    #             return ""
    #         s1 = s0.replace(" ", "")
    #         # chr2 -> ChR2 (case-insensitive)
    #         s1 = re.sub(r"(?i)chr\s*2", "ChR2", s1)
    #         # gcamp6f -> GCaMP6f
    #         s1 = re.sub(r"(?i)gcamp\s*6f", "GCaMP6f", s1)
    #         # "...cre" (end) -> "...Cre"
    #         s1 = re.sub(r"(?i)([A-Za-z0-9_-]*?)cre$", lambda m: (m.group(1) or "") + "Cre", s1)
    #         return s1

    #     def _san(self, s: str) -> str:
    #         """Allow alnum, underscore, hyphen, dot; collapse repeated underscores/hyphens."""
    #         s = (s or "").strip()
    #         s = s.replace(" ", "_")
    #         s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    #         s = re.sub(r"_+", "_", s)
    #         s = re.sub(r"-+", "-", s)
    #         return s

    #     def _update_preview(self):
    #         # Read & normalize
    #         strain = self._norm_token(self.le_strain.text())
    #         virus  = self._norm_token(self.le_virus.text())

    #         stim   = self.le_stim.text().strip()
    #         power  = self.le_power.text().strip()
    #         sex    = self.cb_sex.currentText().strip()
    #         animal = self.le_animal.text().strip()
    #         abf    = self._abf_name
    #         ch     = self._channel

    #         # Sanitize for filename
    #         strain_s = self._san(strain)
    #         virus_s  = self._san(virus)
    #         stim_s   = self._san(stim)
    #         power_s  = self._san(power)
    #         sex_s    = self._san(sex)
    #         animal_s = self._san(animal)
    #         abf_s    = self._san(abf)
    #         ch_s     = self._san(ch)

    #         # STANDARD ORDER:
    #         # Strain_Virus_Sex_Animal_Stim_Power_ABF_Channel
    #         parts = [p for p in (strain_s, virus_s, sex_s, animal_s, stim_s, power_s, abf_s, ch_s) if p]
    #         preview = "_".join(parts) if parts else "analysis"
    #         self.lbl_preview.setText(preview)

    #     def values(self) -> dict:
    #         return {
    #             "strain": self.le_strain.text().strip(),
    #             "virus":  self.le_virus.text().strip(),
    #             "stim":   self.le_stim.text().strip(),
    #             "power":  self.le_power.text().strip(),
    #             "sex":    self.cb_sex.currentText().strip(),
    #             "animal": self.le_animal.text().strip(),
    #             "abf":    self._abf_name,
    #             "chan":   self._channel,
    #             "preview": self.lbl_preview.text().strip(),
    #         }

    class SaveMetaDialog(QDialog):
        def __init__(self, abf_name: str, channel: str, parent=None, auto_stim: str = ""):
            super().__init__(parent)
            self.setWindowTitle("Save analyzed data — name builder")

            self._abf_name = abf_name
            self._channel = channel

            lay = QFormLayout(self)

            # Mouse Strain
            self.le_strain = QLineEdit(self)
            self.le_strain.setPlaceholderText("e.g., VgatCre")
            lay.addRow("Mouse Strain:", self.le_strain)

            # Virus
            self.le_virus = QLineEdit(self)
            self.le_virus.setPlaceholderText("e.g., ConFoff-ChR2")
            lay.addRow("Virus:", self.le_virus)

            # Stimulation type (can be auto-populated)
            self.le_stim = QLineEdit(self)
            self.le_stim.setPlaceholderText("e.g., 20Hz10s15ms or 15msPulse")
            if auto_stim:
                self.le_stim.setText(auto_stim)
            lay.addRow("Stimulation type:", self.le_stim)

            self.le_power = QLineEdit(self)
            self.le_power.setPlaceholderText("e.g., 8mW")
            lay.addRow("Laser power:", self.le_power)

            self.cb_sex = QComboBox(self)
            self.cb_sex.addItems(["", "M", "F", "Unknown"])
            lay.addRow("Sex:", self.cb_sex)

            self.le_animal = QLineEdit(self)
            self.le_animal.setPlaceholderText("e.g., 25121004")
            lay.addRow("Animal ID:", self.le_animal)

            # Read-only info
            self.lbl_abf = QLabel(abf_name, self)
            self.lbl_chn = QLabel(channel or "", self)
            lay.addRow("ABF file:", self.lbl_abf)
            lay.addRow("Channel:", self.lbl_chn)

            # NEW: choose location checkbox
            self.cb_choose_dir = QCheckBox("Let me choose where to save", self)
            self.cb_choose_dir.setToolTip("If unchecked, files go to a 'Pleth_App_Analysis' folder automatically.")
            lay.addRow("", self.cb_choose_dir)

            # Live preview
            self.lbl_preview = QLabel("", self)
            self.lbl_preview.setStyleSheet("color:#b6bfda;")
            lay.addRow("Preview:", self.lbl_preview)

            # Buttons
            btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
            lay.addRow(btns)
            btns.accepted.connect(self.accept)
            btns.rejected.connect(self.reject)

            # Update preview on change
            self.le_strain.textChanged.connect(self._update_preview)
            self.le_virus.textChanged.connect(self._update_preview)
            self.le_stim.textChanged.connect(self._update_preview)
            self.le_power.textChanged.connect(self._update_preview)
            self.cb_sex.currentTextChanged.connect(self._update_preview)
            self.le_animal.textChanged.connect(self._update_preview)

            self._update_preview()

        # --- Helpers: light canonicalization + sanitization ---
        def _norm_token(self, s: str) -> str:
            s0 = (s or "").strip()
            if not s0:
                return ""
            s1 = s0.replace(" ", "")
            s1 = re.sub(r"(?i)chr\s*2", "ChR2", s1)            # chr2 -> ChR2
            s1 = re.sub(r"(?i)gcamp\s*6f", "GCaMP6f", s1)      # gcamp6f -> GCaMP6f
            s1 = re.sub(r"(?i)([A-Za-z0-9_-]*?)cre$", lambda m: (m.group(1) or "") + "Cre", s1)  # ...cre -> ...Cre
            return s1

        def _san(self, s: str) -> str:
            s = (s or "").strip()
            s = s.replace(" ", "_")
            s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
            s = re.sub(r"_+", "_", s)
            s = re.sub(r"-+", "-", s)
            return s

        def _update_preview(self):
            # Read & normalize
            strain = self._norm_token(self.le_strain.text())
            virus  = self._norm_token(self.le_virus.text())

            stim   = self.le_stim.text().strip()
            power  = self.le_power.text().strip()
            sex    = self.cb_sex.currentText().strip()
            animal = self.le_animal.text().strip()
            abf    = self._abf_name
            ch     = self._channel

            # Sanitize for filename
            strain_s = self._san(strain)
            virus_s  = self._san(virus)
            stim_s   = self._san(stim)
            power_s  = self._san(power)
            sex_s    = self._san(sex)
            animal_s = self._san(animal)
            abf_s    = self._san(abf)
            ch_s     = self._san(ch)

            # STANDARD ORDER:
            # Strain_Virus_Sex_Animal_Stim_Power_ABF_Channel
            parts = [p for p in (strain_s, virus_s, sex_s, animal_s, stim_s, power_s, abf_s, ch_s) if p]
            preview = "_".join(parts) if parts else "analysis"
            self.lbl_preview.setText(preview)

        def values(self) -> dict:
            return {
                "strain": self.le_strain.text().strip(),
                "virus":  self.le_virus.text().strip(),
                "stim":   self.le_stim.text().strip(),
                "power":  self.le_power.text().strip(),
                "sex":    self.cb_sex.currentText().strip(),
                "animal": self.le_animal.text().strip(),
                "abf":    self._abf_name,
                "chan":   self._channel,
                "preview": self.lbl_preview.text().strip(),
                "choose_dir": bool(self.cb_choose_dir.isChecked()),
            }


    def _suggest_stim_string(self) -> str:
        """
        Build a stim name like '20Hz10s15ms' from detected stim metrics
        or '15msPulse' / '5sPulse' for single pulses.
        Rounding:
        - freq_hz -> nearest Hz
        - duration_s -> nearest second
        - pulse_width_s -> nearest millisecond (or nearest second if >1s)
        """
        st = self.state
        if not getattr(st, "stim_chan", None):
            return ""

        # Make sure current sweep has metrics if possible
        try:
            self._compute_stim_for_current_sweep()
        except Exception:
            pass

        # Prefer current sweep, else first one that has metrics
        m = st.stim_metrics_by_sweep.get(st.sweep_idx) if hasattr(st, "stim_metrics_by_sweep") else None
        if not m:
            for _s, _m in getattr(st, "stim_metrics_by_sweep", {}).items():
                if _m:
                    m = _m
                    break
        if not m:
            return ""

        freq_hz   = m.get("freq_hz", None)
        dur_s     = m.get("duration_s", None)
        pw_s      = m.get("pulse_width_s", None)

        def _round_int(x):  # safe int rounding
            if x is None:
                return None
            try:
                return int(round(float(x)))
            except Exception:
                return None

        # TRAIN: if freq present and > 0, use Freq + Duration + PW
        if freq_hz is not None and np.isfinite(freq_hz) and freq_hz > 0:
            F = _round_int(freq_hz)              # Hz
            D = _round_int(dur_s)                # s
            MS = _round_int((pw_s or 0) * 1000)  # ms

            parts = []
            if F is not None and F > 0:  parts.append(f"{F}Hz")
            if D is not None and D > 0:  parts.append(f"{D}s")
            if MS is not None and MS > 0: parts.append(f"{MS}ms")
            return "".join(parts) if parts else ""

        # SINGLE PULSE (no usable freq)
        # Prefer pulse width; if missing, fall back to duration
        width_s = pw_s if (pw_s is not None and np.isfinite(pw_s)) else dur_s
        if width_s is None:
            return ""

        if width_s < 1.0:
            ms = _round_int(width_s * 1000)
            if ms is None or ms <= 0:
                ms = 1
            return f"{ms}msPulse"
        else:
            secs = _round_int(width_s)
            if secs is None or secs <= 0:
                secs = 1
            return f"{secs}sPulse"


    # def on_save_analyzed_clicked(self):
    #     st = self.state
    #     if not getattr(st, "in_path", None):
    #         QMessageBox.information(self, "Save analyzed data", "Open an ABF first.")
    #         return

    #     # Build metadata → previewed base name
    #     abf_stem = st.in_path.stem
    #     chan = st.analyze_chan or ""
    #     dlg = self.SaveMetaDialog(abf_name=abf_stem, channel=chan, parent=self)
    #     if dlg.exec() != QDialog.DialogCode.Accepted:
    #         return
    #     vals = dlg.values()
    #     suggested = self._sanitize_token(vals["preview"]) or "analysis"

    #     # Choose destination file (we'll treat the chosen *name* as the base; you can
    #     # append your own suffixes like _peaks.csv, _metrics.csv later).
    #     last_dir = self.settings.value("save_dir", str(st.in_path.parent))
    #     from PyQt6.QtWidgets import QFileDialog
    #     path, _ = QFileDialog.getSaveFileName(
    #         self,
    #         "Choose where to save (this will be your base name)",
    #         str(Path(last_dir) / f"{suggested}.csv"),
    #         "CSV (*.csv);;All Files (*.*)"
    #     )
    #     if not path:
    #         return

    #     p = Path(path)
    #     # Remember directory for next time
    #     self.settings.setValue("save_dir", str(p.parent))

    #     # Store base pieces for your later multi-file writes
    #     self._save_dir = p.parent
    #     # Use the *stem* as the base (allows user to edit the name in the Save dialog)
    #     self._save_base = p.stem

    #     # Also keep the metadata if you want to embed in files later
    #     self._save_meta = vals  # dict with stim, power, sex, animal, abf, chan, preview

    #     # If you want to immediately write one "manifest" CSV (optional), do it here:
    #     # self._write_manifest_csv(self._save_dir / f"{self._save_base}_manifest.csv")

    #     try:
    #         self.statusbar.showMessage(f"Save base set → {self._save_dir / self._save_base}", 4000)
    #     except Exception:
    #         pass
    #     print(f"[save] base path set: {self._save_dir / self._save_base}")

    # def on_save_analyzed_clicked(self):
    #     st = self.state
    #     if not getattr(st, "in_path", None):
    #         QMessageBox.information(self, "Save analyzed data", "Open an ABF first.")
    #         return

    #     # Build metadata → previewed base name
    #     abf_stem = st.in_path.stem
    #     chan = st.analyze_chan or ""
    #     dlg = self.SaveMetaDialog(abf_name=abf_stem, channel=chan, parent=self)
    #     if dlg.exec() != QDialog.DialogCode.Accepted:
    #         return
    #     vals = dlg.values()
    #     suggested = self._sanitize_token(vals["preview"]) or "analysis"

    #     # Pick a base filename (we'll append our own suffixes)
    #     last_dir = self.settings.value("save_dir", str(st.in_path.parent))
    #     path, _ = QFileDialog.getSaveFileName(
    #         self,
    #         "Choose where to save (this will be your base name)",
    #         str(Path(last_dir) / f"{suggested}.csv"),
    #         "CSV (*.csv);;All Files (*.*)"
    #     )
    #     if not path:
    #         return

    #     p = Path(path)
    #     self.settings.setValue("save_dir", str(p.parent))

    #     self._save_dir = p.parent
    #     self._save_base = p.stem
    #     self._save_meta = vals

    #     try:
    #         self.statusbar.showMessage(f"Save base set → {self._save_dir / self._save_base}", 4000)
    #     except Exception:
    #         pass
    #     print(f"[save] base path set: {self._save_dir / self._save_base}")

    #     # >>> ACTUALLY WRITE THE FILES NOW <<<
    #     try:
    #         self._export_all_analyzed_data()
    #     except Exception as e:
    #         QMessageBox.critical(self, "Save error", str(e))

    # def on_save_analyzed_clicked(self):
    #     st = self.state
    #     if not getattr(st, "in_path", None):
    #         QMessageBox.information(self, "Save analyzed data", "Open an ABF first.")
    #         return

    #     abf_stem = st.in_path.stem
    #     chan = st.analyze_chan or ""

    #     auto_stim = self._suggest_stim_string()  # <- NEW

    #     dlg = self.SaveMetaDialog(
    #         abf_name=abf_stem,
    #         channel=chan,
    #         auto_stim=auto_stim,   # <- NEW
    #         parent=self
    #     )
    #     if dlg.exec() != QDialog.DialogCode.Accepted:
    #         return
    #     vals = dlg.values()
    #     suggested = self._sanitize_token(vals["preview"]) or "analysis"

    #     # Pick a base filename (we'll append our own suffixes)
    #     last_dir = self.settings.value("save_dir", str(st.in_path.parent))
    #     path, _ = QFileDialog.getSaveFileName(
    #         self,
    #         "Choose where to save (this will be your base name)",
    #         str(Path(last_dir) / f"{suggested}.csv"),
    #         "CSV (*.csv);;All Files (*.*)"
    #     )
    #     if not path:
    #         return

    #     p = Path(path)
    #     self.settings.setValue("save_dir", str(p.parent))

    #     self._save_dir = p.parent
    #     self._save_base = p.stem
    #     self._save_meta = vals

    #     try:
    #         self.statusbar.showMessage(f"Save base set → {self._save_dir / self._save_base}", 4000)
    #     except Exception:
    #         pass
    #     print(f"[save] base path set: {self._save_dir / self._save_base}")

    #     # >>> ACTUALLY WRITE THE FILES NOW <<<
    #     try:
    #         self._export_all_analyzed_data()
    #     except Exception as e:
    #         QMessageBox.critical(self, "Save error", str(e))


    # def on_save_analyzed_clicked(self):
    #     st = self.state
    #     if not getattr(st, "in_path", None):
    #         QMessageBox.information(self, "Save analyzed data", "Open an ABF first.")
    #         return

    #     # --- Build an auto stim string from current sweep metrics, if available ---
    #     def _auto_stim_from_metrics() -> str:
    #         s = max(0, min(getattr(st, "sweep_idx", 0), self._sweep_count()-1))
    #         m = st.stim_metrics_by_sweep.get(s, {}) if getattr(st, "stim_metrics_by_sweep", None) else {}
    #         if not m:
    #             return ""
    #         def _ri(x):  # round to nearest int safely
    #             try: return int(round(float(x)))
    #             except Exception: return None

    #         f = _ri(m.get("freq_hz"))
    #         d = _ri(m.get("duration_s"))
    #         pw_s = m.get("pulse_width_s")
    #         n_pulses = m.get("n_pulses", None)

    #         # If we can form the standard Freq+Dur+PW form
    #         if f is not None and d is not None and pw_s is not None:
    #             if pw_s >= 1.0:
    #                 pw_str = f"{_ri(pw_s)}s"
    #             else:
    #                 pw_str = f"{_ri(pw_s * 1000)}ms"
    #             return f"{f}Hz{d}s{pw_str}"

    #         # Single-pulse fallback
    #         if pw_s is not None and (n_pulses == 1 or f is None):
    #             if pw_s >= 1.0:
    #                 return f"{_ri(pw_s)}sPulse"
    #             else:
    #                 return f"{_ri(pw_s * 1000)}msPulse"

    #         return ""

    #     abf_stem = st.in_path.stem
    #     chan = st.analyze_chan or ""
    #     auto_stim = _auto_stim_from_metrics()

    #     # --- Name builder dialog (with auto stim suggestion) ---
    #     dlg = self.SaveMetaDialog(abf_name=abf_stem, channel=chan, parent=self, auto_stim=auto_stim)
    #     if dlg.exec() != QDialog.DialogCode.Accepted:
    #         return

    #     vals = dlg.values()
    #     suggested = self._sanitize_token(vals["preview"]) or "analysis"
    #     want_picker = bool(vals.get("choose_dir", False))

    #     # --- Decide root folder ---
    #     # Default root = ABF folder (remember it); user can override when picker is checked.
    #     default_root = Path(self.settings.value("save_root", str(st.in_path.parent)))
    #     root: Path

    #     if want_picker:
    #         chosen = QFileDialog.getExistingDirectory(
    #             self,
    #             "Choose a folder (files will go into a 'Pleth_App_Analysis' subfolder here)",
    #             str(default_root)
    #         )
    #         if not chosen:
    #             return
    #         root = Path(chosen)
    #     else:
    #         root = default_root

    #     # Always save into a Pleth_App_Analysis subfolder of the chosen root
    #     final_dir = (root / "Pleth_App_Analysis")
    #     try:
    #         final_dir.mkdir(parents=True, exist_ok=True)
    #     except Exception as e:
    #         QMessageBox.critical(self, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
    #         return

    #     # Remember last chosen root for next time
    #     self.settings.setValue("save_root", str(root))

    #     # Base pieces for subsequent writes
    #     self._save_dir = final_dir
    #     self._save_base = suggested
    #     self._save_meta = vals  # includes your new strain/virus fields, etc.

    #     # Feedback + save now
    #     base_path = self._save_dir / self._save_base
    #     print(f"[save] base path set: {base_path}")
    #     try:
    #         self.statusbar.showMessage(f"Saving to: {base_path}", 4000)
    #     except Exception:
    #         pass

    #     # Kick off the actual export
    #     self._export_all_analyzed_data()

    # def on_save_analyzed_clicked(self):
    #     st = self.state
    #     if not getattr(st, "in_path", None):
    #         QMessageBox.information(self, "Save analyzed data", "Open an ABF first.")
    #         return

    #     # --- Build an auto stim string from current sweep metrics, if available ---
    #     def _auto_stim_from_metrics() -> str:
    #         s = max(0, min(getattr(st, "sweep_idx", 0), self._sweep_count()-1))
    #         m = st.stim_metrics_by_sweep.get(s, {}) if getattr(st, "stim_metrics_by_sweep", None) else {}
    #         if not m:
    #             return ""
    #         def _ri(x):
    #             try: return int(round(float(x)))
    #             except Exception: return None

    #         f = _ri(m.get("freq_hz"))
    #         d = _ri(m.get("duration_s"))
    #         pw_s = m.get("pulse_width_s")
    #         n_pulses = m.get("n_pulses", None)

    #         if f is not None and d is not None and pw_s is not None:
    #             if pw_s >= 1.0:
    #                 pw_str = f"{_ri(pw_s)}s"
    #             else:
    #                 pw_str = f"{_ri(pw_s * 1000)}ms"
    #             return f"{f}Hz{d}s{pw_str}"

    #         if pw_s is not None and (n_pulses == 1 or f is None):
    #             if pw_s >= 1.0:
    #                 return f"{_ri(pw_s)}sPulse"
    #             else:
    #                 return f"{_ri(pw_s * 1000)}msPulse"

    #         return ""

    #     abf_stem = st.in_path.stem
    #     chan = st.analyze_chan or ""
    #     auto_stim = _auto_stim_from_metrics()

    #     # --- Name builder dialog (with auto stim suggestion) ---
    #     dlg = self.SaveMetaDialog(abf_name=abf_stem, channel=chan, parent=self, auto_stim=auto_stim)
    #     if dlg.exec() != QDialog.DialogCode.Accepted:
    #         return

    #     vals = dlg.values()
    #     suggested = self._sanitize_token(vals["preview"]) or "analysis"
    #     want_picker = bool(vals.get("choose_dir", False))

    #     # --- Resolve final directory according to your new rules ---
    #     default_root = Path(self.settings.value("save_root", str(st.in_path.parent)))
    #     target_lower = "pleth_app_analysis"

    #     def _nearest_analysis_ancestor(p: Path) -> Path | None:
    #         # return the closest ancestor (including self) named Pleth_App_Analysis (case-insensitive)
    #         for cand in [p] + list(p.parents):
    #             if cand.name.lower() == target_lower:
    #                 return cand
    #         return None

    #     if want_picker:
    #         chosen = QFileDialog.getExistingDirectory(
    #             self,
    #             "Choose a folder (files may go into an existing 'Pleth_App_Analysis' here)",
    #             str(default_root)
    #         )
    #         if not chosen:
    #             return
    #         chosen_path = Path(chosen)

    #         # 1) If the chosen path is inside an ancestor named Pleth_App_Analysis → save THERE (the ancestor).
    #         anc = _nearest_analysis_ancestor(chosen_path)
    #         if anc is not None:
    #             final_dir = anc
    #         else:
    #             # 2) If the chosen path contains an existing Pleth_App_Analysis (or Pleth_App_analysis) subfolder → use it.
    #             sub_exact = chosen_path / "Pleth_App_Analysis"
    #             sub_variant = chosen_path / "Pleth_App_analysis"
    #             if sub_exact.is_dir():
    #                 final_dir = sub_exact
    #             elif sub_variant.is_dir():
    #                 final_dir = sub_variant
    #             else:
    #                 # 3) Otherwise, create Pleth_App_Analysis directly under the chosen path.
    #                 final_dir = chosen_path / "Pleth_App_Analysis"
    #                 try:
    #                     final_dir.mkdir(parents=True, exist_ok=True)
    #                 except Exception as e:
    #                     QMessageBox.critical(self, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
    #                     return
    #         # remember picker root for next time
    #         self.settings.setValue("save_root", str(chosen_path))
    #     else:
    #         # Unchecked: default to ABF folder root under Pleth_App_Analysis (create if needed)
    #         root = default_root
    #         final_dir = root / "Pleth_App_Analysis"
    #         try:
    #             final_dir.mkdir(parents=True, exist_ok=True)
    #         except Exception as e:
    #             QMessageBox.critical(self, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
    #             return
    #         self.settings.setValue("save_root", str(root))

    #     # Set base name and meta, then export
    #     self._save_dir = final_dir
    #     self._save_base = suggested
    #     self._save_meta = vals

    #     base_path = self._save_dir / self._save_base
    #     print(f"[save] base path set: {base_path}")
    #     try:
    #         self.statusbar.showMessage(f"Saving to: {base_path}", 4000)
    #     except Exception:
    #         pass

    #     self._export_all_analyzed_data()

    def on_save_analyzed_clicked(self):
        st = self.state
        if not getattr(st, "in_path", None):
            QMessageBox.information(self, "Save analyzed data", "Open an ABF first.")
            return

        # --- Build an auto stim string from current sweep metrics, if available ---
        def _auto_stim_from_metrics() -> str:
            s = max(0, min(getattr(st, "sweep_idx", 0), self._sweep_count()-1))
            m = st.stim_metrics_by_sweep.get(s, {}) if getattr(st, "stim_metrics_by_sweep", None) else {}
            if not m:
                return ""
            def _ri(x):
                try: return int(round(float(x)))
                except Exception: return None

            f = _ri(m.get("freq_hz"))
            d = _ri(m.get("duration_s"))
            pw_s = m.get("pulse_width_s")
            n_pulses = m.get("n_pulses", None)

            if f is not None and d is not None and pw_s is not None:
                pw_str = f"{_ri(pw_s)}s" if pw_s >= 1.0 else f"{_ri(pw_s * 1000)}ms"
                return f"{f}Hz{d}s{pw_str}"

            if pw_s is not None and (n_pulses == 1 or f is None):
                return f"{_ri(pw_s)}sPulse" if pw_s >= 1.0 else f"{_ri(pw_s * 1000)}msPulse"
            return ""

        abf_stem = st.in_path.stem
        chan = st.analyze_chan or ""
        auto_stim = _auto_stim_from_metrics()

        # --- Name builder dialog (with auto stim suggestion) ---
        dlg = self.SaveMetaDialog(abf_name=abf_stem, channel=chan, parent=self, auto_stim=auto_stim)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        vals = dlg.values()
        suggested = self._sanitize_token(vals["preview"]) or "analysis"
        want_picker = bool(vals.get("choose_dir", False))

        target_exact = "Pleth_App_analysis"          # <- required folder name (lowercase 'a')
        target_lower = target_exact.lower()

        def _nearest_analysis_ancestor(p: Path) -> Path | None:
            # Return the closest ancestor (including self) named Pleth_App_analysis (case-insensitive)
            for cand in [p] + list(p.parents):
                if cand.name.lower() == target_lower:
                    return cand
            return None

        if want_picker:
            # User chooses a folder; we keep your previous smart logic here.
            default_root = Path(self.settings.value("save_root", str(st.in_path.parent)))
            chosen = QFileDialog.getExistingDirectory(
                self,
                "Choose a folder (files may go into an existing 'Pleth_App_analysis' here)",
                str(default_root)
            )
            if not chosen:
                return
            chosen_path = Path(chosen)

            # 1) If chosen folder is inside an ancestor named Pleth_App_analysis → save THERE (the ancestor)
            anc = _nearest_analysis_ancestor(chosen_path)
            if anc is not None:
                final_dir = anc
            else:
                # 2) If the chosen folder already contains 'Pleth_App_analysis' or 'Pleth_App_Analysis' subfolder → use it
                sub_exact   = chosen_path / "Pleth_App_analysis"
                sub_variant = chosen_path / "Pleth_App_Analysis"
                if sub_exact.is_dir():
                    final_dir = sub_exact
                elif sub_variant.is_dir():
                    final_dir = sub_variant
                else:
                    # 3) Otherwise, create Pleth_App_analysis directly under the chosen folder
                    final_dir = chosen_path / target_exact
                    try:
                        final_dir.mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        QMessageBox.critical(self, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
                        return

            # Remember the last *picker* root only when the picker is used
            self.settings.setValue("save_root", str(chosen_path))

        else:
            # UNCHECKED: Always use the CURRENT ABF DIRECTORY (not a remembered one)
            base_root = st.in_path.parent if getattr(st, "in_path", None) else Path.cwd()
            final_dir = base_root / target_exact
            try:
                final_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
                return
            # IMPORTANT: Do NOT overwrite 'save_root' here — we don't want to "remember" anything for the unchecked case.

        # Set base name + meta, then export
        self._save_dir = final_dir
        self._save_base = suggested
        self._save_meta = vals

        base_path = self._save_dir / self._save_base
        print(f"[save] base path set: {base_path}")
        try:
            self.statusbar.showMessage(f"Saving to: {base_path}", 4000)
        except Exception:
            pass

        self._export_all_analyzed_data()


    # metrics we won't include in CSV exports
    _EXCLUDE_FOR_CSV = {"d1", "d2"}

    def _metric_keys_in_order(self):
        """Return metric keys in the UI order (from metrics.METRIC_SPECS)."""
        return [key for (_, key) in metrics.METRIC_SPECS]

    def _compute_metric_trace(self, key, t, y, sr_hz, peaks, breaths):
        """
        Call the metric function, passing expoffs if it exists.
        Falls back to legacy signature when needed.
        """
        fn = metrics.METRICS[key]
        on  = breaths.get("onsets")   if breaths else None
        off = breaths.get("offsets")  if breaths else None
        exm = breaths.get("expmins")  if breaths else None
        exo = breaths.get("expoffs")  if breaths else None
        try:
            return fn(t, y, sr_hz, peaks, on, off, exm, exo)  # new signature
        except TypeError:
            return fn(t, y, sr_hz, peaks, on, off, exm)       # legacy signature

    def _get_stim_masks(self, s: int):
        """
        Build (baseline_mask, stim_mask, post_mask) boolean arrays over st.t for sweep s.
        Uses union of all stim spans for 'stim'.
        """
        st = self.state
        t = st.t
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if not spans:
            # no stim: whole trace is baseline; stim/post empty
            B = np.ones_like(t, dtype=bool)
            Z = np.zeros_like(t, dtype=bool)
            return B, Z, Z

        starts = np.array([a for (a, _) in spans], dtype=float)
        ends   = np.array([b for (_, b) in spans], dtype=float)
        t0 = np.min(starts)
        t1 = np.max(ends)

        stim_mask = np.zeros_like(t, dtype=bool)
        for (a, b) in spans:
            stim_mask |= (t >= a) & (t <= b)

        baseline_mask = t < t0
        post_mask     = t > t1
        return baseline_mask, stim_mask, post_mask

    def _nanmean_sem(self, X: np.ndarray, axis: int = 0):
        """Return (nanmean, nansem) along axis; SEM uses ddof=1 where n>=2 else NaN."""
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(X, axis=axis)
            n    = np.sum(np.isfinite(X), axis=axis)
            # std with ddof=1; guard n<2
            std  = np.nanstd(X, axis=axis, ddof=1)
            sem  = np.where(n >= 2, std / np.sqrt(n), np.nan)
        return mean, sem


    # def _export_all_analyzed_data(self):
    #     """
    #     Save a smaller (downsampled) NPZ bundle and two CSVs:
    #     (1) *_bundle.npz : t, downsampled Y_proc and y2_* traces (time axis decimated),
    #                         plus event indices AND event times per sweep,
    #                         plus stim spans, plus meta (including ds_step and ds_sr_hz).
    #     (2) *_means_by_time.csv : time-normalized (laser onset = 0) and downsampled,
    #                                 mean & SEM across sweeps for each continuous metric (excl. d1/d2).
    #     (3) *_baseline_stim_post.csv : same as before (small), computed from full-res traces
    #                                     (no need to downsample since it’s tiny).
    #     """
    #     st = self.state
    #     if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
    #         QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
    #         return
    #     if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
    #         QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
    #         return

    #     # ----------------------------
    #     # Tunables for downsampling
    #     # ----------------------------
    #     DS_TARGET_HZ = 50.0     # export at ~50 Hz (adjust as you like)
    #     CSV_MAX_ROWS = 20000     # safety cap for rows in the per-time CSV

    #     # Pick a decimation step: by target Hz and by row cap
    #     dt = 1.0 / float(st.sr_hz)
    #     step_hz = max(1, int(round(st.sr_hz / DS_TARGET_HZ)))
    #     # cap by rows (use the min across planned steps so we never exceed CSV_MAX_ROWS)
    #     N_full = len(st.t)
    #     step_rows = max(1, int(np.ceil(N_full / CSV_MAX_ROWS)))
    #     ds_step = max(1, step_hz, step_rows)
    #     ds_sr_hz = float(st.sr_hz) / float(ds_step)

    #     # ----------------------------
    #     # Collect data for ALL sweeps
    #     # ----------------------------
    #     any_ch = next(iter(st.sweeps.values()))
    #     n_sweeps = any_ch.shape[1]
    #     N = len(st.t)

    #     # Processed pleth (what the user sees)
    #     Y_proc = np.empty((N, n_sweeps), dtype=float)
    #     Y_proc[:] = np.nan

    #     # Per-sweep peaks & breaths
    #     peaks_by_sweep = []
    #     on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], []

    #     # Metric traces (full-res first, then we’ll decimate for saving)
    #     all_keys = self._metric_keys_in_order()
    #     y2_by_key_full = {k: np.full((N, n_sweeps), np.nan, dtype=float) for k in all_keys}

    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         Y_proc[:, s] = y_proc

    #         pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         breaths = st.breath_by_sweep.get(s, None)
    #         if breaths is None and pks.size:
    #             breaths = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = breaths
    #         if breaths is None:
    #             breaths = {
    #                 "onsets": np.array([], dtype=int),
    #                 "offsets": np.array([], dtype=int),
    #                 "expmins": np.array([], dtype=int),
    #                 "expoffs": np.array([], dtype=int),
    #             }

    #         peaks_by_sweep.append(pks)
    #         on_by_sweep.append(np.asarray(breaths.get("onsets",  []), dtype=int))
    #         off_by_sweep.append(np.asarray(breaths.get("offsets", []), dtype=int))
    #         exm_by_sweep.append(np.asarray(breaths.get("expmins", []), dtype=int))
    #         exo_by_sweep.append(np.asarray(breaths.get("expoffs", []), dtype=int))

    #         # Compute every metric (full-res), we’ll decimate later
    #         for k in all_keys:
    #             y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, breaths)
    #             if y2 is not None and len(y2) == N:
    #                 y2_by_key_full[k][:, s] = y2

    #     # ----------------------------
    #     # Build downsampled arrays
    #     # ----------------------------
    #     idx_ds = np.arange(0, N, ds_step, dtype=int)
    #     t_ds = st.t[idx_ds]
    #     Y_proc_ds = Y_proc[idx_ds, :]

    #     y2_by_key_ds = {k: y2_by_key_full[k][idx_ds, :] for k in all_keys}

    #     # ----------------------------
    #     # (1) NPZ bundle (downsampled traces)
    #     # ----------------------------
    #     base = self._save_dir / self._save_base
    #     npz_path = base.with_name(base.name + "_bundle.npz")

    #     # Stim spans (times) per sweep
    #     stim_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

    #     # Ragged arrays → object arrays (indices)
    #     peaks_obj = np.array(peaks_by_sweep, dtype=object)
    #     on_obj    = np.array(on_by_sweep,    dtype=object)
    #     off_obj   = np.array(off_by_sweep,   dtype=object)
    #     exm_obj   = np.array(exm_by_sweep,   dtype=object)
    #     exo_obj   = np.array(exo_by_sweep,   dtype=object)

    #     # Also store event TIMES (so they align with t_ds easily even after decimation)
    #     def _inds_to_times(inds):
    #         if inds is None or len(inds) == 0:
    #             return np.array([], dtype=float)
    #         return st.t[np.asarray(inds, dtype=int)]

    #     on_times_obj  = np.empty(n_sweeps, dtype=object)
    #     off_times_obj = np.empty(n_sweeps, dtype=object)
    #     exm_times_obj = np.empty(n_sweeps, dtype=object)
    #     exo_times_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         on_times_obj[s]  = _inds_to_times(on_by_sweep[s])
    #         off_times_obj[s] = _inds_to_times(off_by_sweep[s])
    #         exm_times_obj[s] = _inds_to_times(exm_by_sweep[s])
    #         exo_times_obj[s] = _inds_to_times(exo_by_sweep[s])

    #     # flatten downsampled y2 dict into kwargs (y2_<key>=downsampled array)
    #     y2_kwargs_ds = {f"y2_{k}": y2_by_key_ds[k] for k in all_keys}

    #     meta = {
    #         "analyze_channel": st.analyze_chan,
    #         "sr_hz": float(st.sr_hz),
    #         "n_sweeps": int(n_sweeps),
    #         "abf_path": str(getattr(st, "in_path", "")),
    #         "ui_meta": getattr(self, "_save_meta", {}),
    #         "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
    #         "downsample_step": int(ds_step),
    #         "downsample_sr_hz": float(ds_sr_hz),
    #         "notes": "t and Y_proc and y2_* are downsampled by decimation; event indices are at full-res; event *_times are provided.",
    #     }

    #     np.savez_compressed(
    #         npz_path,
    #         t=t_ds,                         # downsampled time
    #         Y_proc=Y_proc_ds,               # downsampled processed pleth
    #         peaks_by_sweep=peaks_obj,       # indices (full-res)
    #         onsets_by_sweep=on_obj,
    #         offsets_by_sweep=off_obj,
    #         expmins_by_sweep=exm_obj,
    #         expoffs_by_sweep=exo_obj,
    #         onsets_times_by_sweep=on_times_obj,    # times (helpful with t_ds)
    #         offsets_times_by_sweep=off_times_obj,
    #         expmins_times_by_sweep=exm_times_obj,
    #         expoffs_times_by_sweep=exo_times_obj,
    #         stim_spans_by_sweep=stim_obj,   # times
    #         meta_json=json.dumps(meta),
    #         **y2_kwargs_ds,                 # downsampled y2 traces
    #     )

    #     # --------------------------------------------------------
    #     # (2) Per-time CSV: mean & SEM across sweeps (downsampled)
    #     #     Time normalized to laser onset (0 at first stim start),
    #     #     aligned to a reference sweep by integer-sample shifting.
    #     # --------------------------------------------------------
    #     csv_time_path = base.with_name(base.name + "_means_by_time.csv")
    #     keys_for_csv = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

    #     # compute first laser onset time per sweep (seconds)
    #     def _first_stim_start_for_sweep(s):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         if spans:
    #             return float(spans[0][0])
    #         return float(st.t[0])  # fallback to sweep start if no stim

    #     t0_by_sweep = np.array([_first_stim_start_for_sweep(s) for s in range(n_sweeps)], dtype=float)

    #     # reference sweep (0) → its zero is t0_ref; we align all to this by integer shifting
    #     t0_ref = float(t0_by_sweep[0])
    #     # number of decimated-sample shifts per sweep relative to reference
    #     # (we use decimated grid, so convert seconds to decimated steps)
    #     shift_ds_by_sweep = np.round((t0_by_sweep - t0_ref) * ds_sr_hz).astype(int)

    #     # helper to shift a (T,S) matrix along time axis 0 by integer steps per column, NaN pad
    #     def _shift_cols_nanpad(A, shifts):
    #         T, S = A.shape
    #         out = np.full_like(A, np.nan)
    #         for s in range(S):
    #             sh = int(shifts[s])
    #             if sh == 0:
    #                 out[:, s] = A[:, s]
    #             elif sh > 0:
    #                 # move forward right; earliest rows become NaN
    #                 if sh < T:
    #                     out[sh:, s] = A[:T - sh, s]
    #             else:
    #                 k = -sh
    #                 if k < T:
    #                     out[:T - k, s] = A[k:, s]
    #         return out

    #     # build aligned (downsampled) y2 stacks for CSV means
    #     aligned_y2_ds = {k: _shift_cols_nanpad(y2_by_key_ds[k], shift_ds_by_sweep) for k in keys_for_csv}

    #     # normalized time for the reference sweep (downsampled)
    #     t_norm_ref_ds = t_ds - t0_ref

    #     # Write CSV
    #     header = ["t"] + [f"{k}_mean" for k in keys_for_csv] + [f"{k}_sem" for k in keys_for_csv]
    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_time_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(header)

    #             T_ds = len(t_norm_ref_ds)
    #             FLUSH_EVERY = 2000
    #             for i in range(T_ds):
    #                 row = [f"{t_norm_ref_ds[i]:.9f}"]
    #                 # means for all metrics
    #                 for k in keys_for_csv:
    #                     col = aligned_y2_ds[k][i, :]
    #                     m, _ = self._mean_sem_1d(col)
    #                     row.append(f"{m:.9g}")
    #                 # sem for all metrics
    #                 for k in keys_for_csv:
    #                     col = aligned_y2_ds[k][i, :]
    #                     _, s = self._mean_sem_1d(col)
    #                     row.append(f"{s:.9g}")
    #                 w.writerow(row)
    #                 if (i % FLUSH_EVERY) == 0:
    #                     QApplication.processEvents()
    #     finally:
    #         self.unsetCursor()

    #     # --------------------------------------------------------
    #     # (3) Window summary CSV: baseline/stim/post (unchanged)
    #     #     Keep as before (small file; compute from full-res y2 traces).
    #     # --------------------------------------------------------
    #     csv_win_path = base.with_name(base.name + "_baseline_stim_post.csv")
    #     header2 = ["metric",
    #             "baseline_mean", "baseline_sem",
    #             "stim_mean",     "stim_sem",
    #             "post_mean",     "post_sem"]
    #     rows2 = []

    #     for k in keys_for_csv:
    #         baseline_vals, stim_vals, post_vals = [], [], []
    #         for s in range(n_sweeps):
    #             B, S, P = self._get_stim_masks(s)
    #             y2 = y2_by_key_full[k][:, s]
    #             if np.any(B):
    #                 mB, _ = self._mean_sem_1d(y2[B])
    #                 if np.isfinite(mB): baseline_vals.append(mB)
    #             if np.any(S):
    #                 mS, _ = self._mean_sem_1d(y2[S])
    #                 if np.isfinite(mS): stim_vals.append(mS)
    #             if np.any(P):
    #                 mP, _ = self._mean_sem_1d(y2[P])
    #                 if np.isfinite(mP): post_vals.append(mP)

    #         bm, bsem = self._mean_sem_1d(np.asarray(baseline_vals, dtype=float)) if baseline_vals else (np.nan, np.nan)
    #         sm, ssem = self._mean_sem_1d(np.asarray(stim_vals, dtype=float))     if stim_vals     else (np.nan, np.nan)
    #         pm, psem = self._mean_sem_1d(np.asarray(post_vals, dtype=float))     if post_vals     else (np.nan, np.nan)

    #         rows2.append([k,
    #                     f"{bm:.9g}", f"{bsem:.9g}",
    #                     f"{sm:.9g}", f"{ssem:.9g}",
    #                     f"{pm:.9g}", f"{psem:.9g}"])

    #     with open(csv_win_path, "w", newline="") as f:
    #         w = csv.writer(f)
    #         w.writerow(header2)
    #         w.writerows(rows2)

    #     # ----------------------------
    #     # Done
    #     # ----------------------------
    #     msg = f"Saved (downsampled):\n- {npz_path.name}\n- {csv_time_path.name}\n- {csv_win_path.name}"
    #     print("[save]", msg)
    #     try:
    #         self.statusbar.showMessage(msg, 6000)
    #     except Exception:
    #         pass


    # def _export_all_analyzed_data(self):
    #     """
    #     Save two files:
    #     1) <base>_bundle.npz  (downsampled Y_proc + all y2 traces, peaks/breaths, stim spans, meta)
    #     2) <base>_means_by_time.csv  (downsampled time-normalized t, per-time mean & SEM across sweeps)

    #     Notes:
    #     - CSV time column is normalized to the first available stim onset (per sweep). For the single
    #     't' column we write the normalized time from the first sweep that has a stim; if none have a
    #     stim we write raw time (starting at 0).
    #     - Everything is downsampled to DS_TARGET_HZ for size/perf.
    #     """
    #     st = self.state
    #     if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
    #         QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
    #         return
    #     if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
    #         QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
    #         return

    #     # ---------- knobs ----------
    #     DS_TARGET_HZ = 50.0          # export sampling rate for NPZ + CSV
    #     CSV_FLUSH_EVERY = 2000       # keep UI snappy while writing

    #     # ---------- basics ----------
    #     any_ch = next(iter(st.sweeps.values()))
    #     n_sweeps = any_ch.shape[1]
    #     N = len(st.t)

    #     # Downsample index (common across sweeps)
    #     ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
    #     ds_idx = np.arange(0, N, ds_step, dtype=int)
    #     M = len(ds_idx)

    #     # We'll keep the raw time grid (for NPZ) and also build a "CSV time" normalized view.
    #     t_ds_raw = st.t[ds_idx]  # unnormalized, for NPZ

    #     # Helper: first stim onset (seconds) for a given sweep; None if no stim in that sweep
    #     def _first_stim_start_for_sweep(s: int) -> float | None:
    #         if not st.stim_chan:
    #             return None
    #         spans = st.stim_spans_by_sweep.get(s, [])
    #         if not spans:
    #             return None
    #         return float(spans[0][0])

    #     # Find a reference sweep (for CSV time column) that actually has a stim
    #     csv_t0 = None
    #     for s_ref in range(n_sweeps):
    #         csv_t0 = _first_stim_start_for_sweep(s_ref)
    #         if csv_t0 is not None:
    #             break
    #     # If *no* sweep has stim, leave csv_t0 = 0
    #     if csv_t0 is None:
    #         csv_t0 = 0.0
    #     t_ds_csv = (st.t - csv_t0)[ds_idx]  # this is the 't' column we write to CSV

    #     # ---------- containers ----------
    #     # Downsampled pleth (what the user sees), shape (M, S)
    #     Y_proc_ds = np.full((M, n_sweeps), np.nan, dtype=float)

    #     # Per-sweep peaks & breaths (ragged → object arrays for NPZ)
    #     peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []

    #     # Compute & collect all metric traces (full res → downsample for NPZ/CSV)
    #     all_keys = self._metric_keys_in_order()
    #     y2_full_by_key = {k: np.full((N, n_sweeps), np.nan, dtype=float) for k in all_keys}
    #     y2_ds_by_key   = {k: np.full((M, n_sweeps), np.nan, dtype=float) for k in all_keys}

    #     # ---------- fill per-sweep ----------
    #     for s in range(n_sweeps):
    #         # Processed pleth (same as plot)
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         Y_proc_ds[:, s] = y_proc[ds_idx]

    #         # Peaks / breaths
    #         pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         breaths = st.breath_by_sweep.get(s, None)
    #         if breaths is None and pks.size:
    #             # compute breaths if missing
    #             breaths = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = breaths
    #         if breaths is None:
    #             breaths = {
    #                 "onsets":  np.array([], dtype=int),
    #                 "offsets": np.array([], dtype=int),
    #                 "expmins": np.array([], dtype=int),
    #                 "expoffs": np.array([], dtype=int),
    #             }

    #         peaks_by_sweep.append(pks)
    #         on_by_sweep.append(np.asarray(breaths.get("onsets",  []), dtype=int))
    #         off_by_sweep.append(np.asarray(breaths.get("offsets", []), dtype=int))
    #         exm_by_sweep.append(np.asarray(breaths.get("expmins", []), dtype=int))
    #         exo_by_sweep.append(np.asarray(breaths.get("expoffs", []), dtype=int))

    #         # y2 metrics (compute full, then downsample)
    #         for k in all_keys:
    #             y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, breaths)
    #             if y2 is not None and len(y2) == N:
    #                 y2_full_by_key[k][:, s] = y2
    #                 y2_ds_by_key[k][:, s]   = y2[ds_idx]

    #     # ---------- (1) NPZ bundle (downsampled) ----------
    #     base = self._save_dir / self._save_base
    #     npz_path = base.with_name(base.name + "_bundle.npz")

    #     # Stim spans per sweep (object array)
    #     stim_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

    #     # Ragged arrays → object arrays
    #     peaks_obj = np.array(peaks_by_sweep, dtype=object)
    #     on_obj    = np.array(on_by_sweep,  dtype=object)
    #     off_obj   = np.array(off_by_sweep, dtype=object)
    #     exm_obj   = np.array(exm_by_sweep, dtype=object)
    #     exo_obj   = np.array(exo_by_sweep, dtype=object)

    #     # y2 downsampled dict → kwargs
    #     y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}

    #     meta = {
    #         "analyze_channel": st.analyze_chan,
    #         "sr_hz": float(st.sr_hz),
    #         "n_sweeps": int(n_sweeps),
    #         "abf_path": str(getattr(st, "in_path", "")),
    #         "ui_meta": getattr(self, "_save_meta", {}),
    #         "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
    #         "ds_target_hz": float(DS_TARGET_HZ),
    #         "ds_step": int(ds_step),
    #     }

    #     np.savez_compressed(
    #         npz_path,
    #         t_ds=t_ds_raw,                 # unnormalized downsampled time grid
    #         Y_proc_ds=Y_proc_ds,           # (M, S)
    #         peaks_by_sweep=peaks_obj,
    #         onsets_by_sweep=on_obj,
    #         offsets_by_sweep=off_obj,
    #         expmins_by_sweep=exm_obj,
    #         expoffs_by_sweep=exo_obj,
    #         stim_spans_by_sweep=stim_obj,
    #         meta_json=json.dumps(meta),
    #         **y2_kwargs_ds,
    #     )

    #     # ---------- (2) Per-time CSV: mean & SEM across sweeps (downsampled, time-normalized) ----------
    #     csv_time_path = base.with_name(base.name + "_means_by_time.csv")
    #     keys_for_csv = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

    #     # Header
    #     header = ["t"]
    #     for k in keys_for_csv:
    #         header += [f"{k}_mean", f"{k}_sem"]

    #     # Busy cursor while writing
    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_time_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(header)

    #             for i in range(M):
    #                 row = [f"{t_ds_csv[i]:.9f}"]
    #                 for k in keys_for_csv:
    #                     col = y2_ds_by_key[k][i, :]  # values at this (downsampled) time across sweeps
    #                     m, sem = self._mean_sem_1d(col)  # finite-only mean/sem
    #                     row += [f"{m:.9g}", f"{sem:.9g}"]
    #                 w.writerow(row)

    #                 if (i % CSV_FLUSH_EVERY) == 0:
    #                     QApplication.processEvents()  # keep UI responsive
    #     finally:
    #         self.unsetCursor()

    #     # ---------- done ----------
    #     msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}"
    #     print("[save]", msg)
    #     try:
    #         self.statusbar.showMessage(msg, 6000)
    #     except Exception:
    #         pass

    # def _export_all_analyzed_data(self):
    #     """
    #     Save two files:
    #     1) <base>_bundle.npz  (downsampled Y_proc + y2 traces, peaks/breaths, stim spans, meta)
    #     2) <base>_means_by_time.csv

    #     CSV layout per row (time-sample):
    #     t, [for each metric k: (optional) k_s1..k_sN, k_mean, k_sem]

    #     Toggle per-trace columns via:
    #         self._csv_include_traces = True/False
    #     """
    #     st = self.state
    #     if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
    #         QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
    #         return
    #     if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
    #         QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
    #         return

    #     import numpy as np, csv, json
    #     from PyQt6.QtCore import Qt
    #     from PyQt6.QtWidgets import QApplication

    #     # ---------- knobs ----------
    #     DS_TARGET_HZ = 50.0                 # export sampling rate for NPZ + CSV
    #     CSV_FLUSH_EVERY = 2000              # keep UI snappy while writing
    #     INCLUDE_TRACES = bool(getattr(self, "_csv_include_traces", True)) #toggle per-sweep traces in CSV

    #     # ---------- basics ----------
    #     any_ch = next(iter(st.sweeps.values()))
    #     n_sweeps = any_ch.shape[1]
    #     N = len(st.t)

    #     # Common downsample index
    #     ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
    #     ds_idx = np.arange(0, N, ds_step, dtype=int)
    #     M = len(ds_idx)

    #     # Time for NPZ (raw) and for CSV (normalized to first stim onset across any sweep)
    #     t_ds_raw = st.t[ds_idx]

    #     def _first_stim_start_for_sweep(s: int):
    #         if not st.stim_chan:
    #             return None
    #         spans = st.stim_spans_by_sweep.get(s, [])
    #         if not spans:
    #             return None
    #         return float(spans[0][0])

    #     csv_t0 = None
    #     for s_ref in range(n_sweeps):
    #         csv_t0 = _first_stim_start_for_sweep(s_ref)
    #         if csv_t0 is not None:
    #             break
    #     if csv_t0 is None:
    #         csv_t0 = 0.0
    #     t_ds_csv = (st.t - csv_t0)[ds_idx]

    #     # ---------- containers ----------
    #     # Downsampled pleth (M, S)
    #     Y_proc_ds = np.full((M, n_sweeps), np.nan, dtype=float)

    #     # Per-sweep peaks & breaths (ragged -> object arrays for NPZ)
    #     peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []

    #     # Metrics (full then downsampled)
    #     all_keys = self._metric_keys_in_order()
    #     y2_ds_by_key = {k: np.full((M, n_sweeps), np.nan, dtype=float) for k in all_keys}

    #     # ---------- fill per-sweep ----------
    #     for s in range(n_sweeps):
    #         # processed pleth
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         Y_proc_ds[:, s] = y_proc[ds_idx]

    #         # peaks & breaths
    #         pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         breaths = st.breath_by_sweep.get(s, None)
    #         if breaths is None and pks.size:
    #             breaths = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = breaths
    #         if breaths is None:
    #             breaths = {
    #                 "onsets":  np.array([], dtype=int),
    #                 "offsets": np.array([], dtype=int),
    #                 "expmins": np.array([], dtype=int),
    #                 "expoffs": np.array([], dtype=int),
    #             }

    #         peaks_by_sweep.append(pks)
    #         on_by_sweep.append(np.asarray(breaths.get("onsets",  []), dtype=int))
    #         off_by_sweep.append(np.asarray(breaths.get("offsets", []), dtype=int))
    #         exm_by_sweep.append(np.asarray(breaths.get("expmins", []), dtype=int))
    #         exo_by_sweep.append(np.asarray(breaths.get("expoffs", []), dtype=int))

    #         # metrics -> downsample for export
    #         for k in all_keys:
    #             y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, breaths)
    #             if y2 is not None and len(y2) == N:
    #                 y2_ds_by_key[k][:, s] = y2[ds_idx]

    #     # ---------- (1) NPZ bundle (downsampled) ----------
    #     base = self._save_dir / self._save_base
    #     npz_path = base.with_name(base.name + "_bundle.npz")

    #     stim_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

    #     peaks_obj = np.array(peaks_by_sweep, dtype=object)
    #     on_obj    = np.array(on_by_sweep,  dtype=object)
    #     off_obj   = np.array(off_by_sweep, dtype=object)
    #     exm_obj   = np.array(exm_by_sweep, dtype=object)
    #     exo_obj   = np.array(exo_by_sweep, dtype=object)

    #     y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}

    #     meta = {
    #         "analyze_channel": st.analyze_chan,
    #         "sr_hz": float(st.sr_hz),
    #         "n_sweeps": int(n_sweeps),
    #         "abf_path": str(getattr(st, "in_path", "")),
    #         "ui_meta": getattr(self, "_save_meta", {}),
    #         "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
    #         "ds_target_hz": float(DS_TARGET_HZ),
    #         "ds_step": int(ds_step),
    #         "csv_time_zero": float(csv_t0),
    #         "csv_includes_traces": bool(INCLUDE_TRACES),
    #     }

    #     np.savez_compressed(
    #         npz_path,
    #         t_ds=t_ds_raw,
    #         Y_proc_ds=Y_proc_ds,
    #         peaks_by_sweep=peaks_obj,
    #         onsets_by_sweep=on_obj,
    #         offsets_by_sweep=off_obj,
    #         expmins_by_sweep=exm_obj,
    #         expoffs_by_sweep=exo_obj,
    #         stim_spans_by_sweep=stim_obj,
    #         meta_json=json.dumps(meta),
    #         **y2_kwargs_ds,
    #     )

    #     # ---------- (2) Per-time CSV: each sweep then mean & SEM (downsampled) ----------
    #     csv_time_path = base.with_name(base.name + "_means_by_time.csv")
    #     keys_for_csv = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

    #     # Header
    #     header = ["t"]
    #     for k in keys_for_csv:
    #         if INCLUDE_TRACES:
    #             header += [f"{k}_s{j+1}" for j in range(n_sweeps)]
    #         header += [f"{k}_mean", f"{k}_sem"]

    #     # Busy cursor while writing
    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_time_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(header)

    #             for i in range(M):
    #                 row = [f"{t_ds_csv[i]:.9f}"]
    #                 for k in keys_for_csv:
    #                     col = y2_ds_by_key[k][i, :]  # values at this time across sweeps
    #                     if INCLUDE_TRACES:
    #                         row += [f"{v:.9g}" for v in col]
    #                     m, sem = self._mean_sem_1d(col)  # finite-only mean/sem
    #                     row += [f"{m:.9g}", f"{sem:.9g}"]
    #                 w.writerow(row)

    #                 if (i % CSV_FLUSH_EVERY) == 0:
    #                     QApplication.processEvents()
    #     finally:
    #         self.unsetCursor()

    #     # ---------- done ----------
    #     msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}"
    #     print("[save]", msg)
    #     try:
    #         self.statusbar.showMessage(msg, 6000)
    #     except Exception:
    #         pass


    # def _export_all_analyzed_data(self):
    #     """
    #     Save three files:
    #     1) <base>_bundle.npz          (downsampled Y_proc + y2 traces, peaks/breaths, stim spans, meta)
    #     2) <base>_means_by_time.csv   (downsampled per-time means/SEMs, optional per-sweep traces)
    #     3) <base>_per_breath.csv      (one row per breath with per-breath metrics and window-tagged columns)
    #     """
    #     st = self.state
    #     if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
    #         QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
    #         return
    #     if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
    #         QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
    #         return

    #     import numpy as np, csv, json
    #     from PyQt6.QtCore import Qt
    #     from PyQt6.QtWidgets import QApplication

    #     # ---------- knobs ----------
    #     DS_TARGET_HZ = 50.0                 # export sampling rate for NPZ + time CSV
    #     CSV_FLUSH_EVERY = 2000              # keep UI snappy while writing
    #     INCLUDE_TRACES = bool(getattr(self, "_csv_include_traces", True))  # toggle per-sweep traces in time CSV

    #     # ---------- basics ----------
    #     any_ch = next(iter(st.sweeps.values()))
    #     n_sweeps = any_ch.shape[1]
    #     N = len(st.t)

    #     # Common downsample index
    #     ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
    #     ds_idx = np.arange(0, N, ds_step, dtype=int)
    #     M = len(ds_idx)

    #     # Time for NPZ (raw) and for CSV (normalized to first stim onset across any sweep)
    #     t_ds_raw = st.t[ds_idx]

    #     def _first_stim_start_for_sweep(s: int):
    #         if not st.stim_chan:
    #             return None
    #         spans = st.stim_spans_by_sweep.get(s, [])
    #         if not spans:
    #             return None
    #         return float(spans[0][0])

    #     csv_t0 = None
    #     for s_ref in range(n_sweeps):
    #         csv_t0 = _first_stim_start_for_sweep(s_ref)
    #         if csv_t0 is not None:
    #             break
    #     if csv_t0 is None:
    #         csv_t0 = 0.0
    #     t_ds_csv = (st.t - csv_t0)[ds_idx]

    #     # ---------- containers ----------
    #     # Downsampled pleth (M, S)
    #     Y_proc_ds = np.full((M, n_sweeps), np.nan, dtype=float)

    #     # Per-sweep peaks & breaths (ragged -> object arrays for NPZ)
    #     peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []

    #     # Metrics (downsampled for NPZ/time CSV)
    #     all_keys = self._metric_keys_in_order()
    #     y2_ds_by_key = {k: np.full((M, n_sweeps), np.nan, dtype=float) for k in all_keys}

    #     # ---------- fill per-sweep ----------
    #     for s in range(n_sweeps):
    #         # processed pleth
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         Y_proc_ds[:, s] = y_proc[ds_idx]

    #         # peaks & breaths
    #         pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         breaths = st.breath_by_sweep.get(s, None)
    #         if breaths is None and pks.size:
    #             breaths = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = breaths
    #         if breaths is None:
    #             breaths = {
    #                 "onsets":  np.array([], dtype=int),
    #                 "offsets": np.array([], dtype=int),
    #                 "expmins": np.array([], dtype=int),
    #                 "expoffs": np.array([], dtype=int),
    #             }

    #         peaks_by_sweep.append(pks)
    #         on_by_sweep.append(np.asarray(breaths.get("onsets",  []), dtype=int))
    #         off_by_sweep.append(np.asarray(breaths.get("offsets", []), dtype=int))
    #         exm_by_sweep.append(np.asarray(breaths.get("expmins", []), dtype=int))
    #         exo_by_sweep.append(np.asarray(breaths.get("expoffs", []), dtype=int))

    #         # metrics -> downsample for export
    #         for k in all_keys:
    #             y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, breaths)
    #             if y2 is not None and len(y2) == N:
    #                 y2_ds_by_key[k][:, s] = y2[ds_idx]

    #     # ---------- (1) NPZ bundle (downsampled) ----------
    #     base = self._save_dir / self._save_base
    #     npz_path = base.with_name(base.name + "_bundle.npz")

    #     stim_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

    #     peaks_obj = np.array(peaks_by_sweep, dtype=object)
    #     on_obj    = np.array(on_by_sweep,  dtype=object)
    #     off_obj   = np.array(off_by_sweep, dtype=object)
    #     exm_obj   = np.array(exm_by_sweep, dtype=object)
    #     exo_obj   = np.array(exo_by_sweep, dtype=object)

    #     y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}

    #     meta = {
    #         "analyze_channel": st.analyze_chan,
    #         "sr_hz": float(st.sr_hz),
    #         "n_sweeps": int(n_sweeps),
    #         "abf_path": str(getattr(st, "in_path", "")),
    #         "ui_meta": getattr(self, "_save_meta", {}),
    #         "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
    #         "ds_target_hz": float(DS_TARGET_HZ),
    #         "ds_step": int(ds_step),
    #         "csv_time_zero": float(csv_t0),
    #         "csv_includes_traces": bool(INCLUDE_TRACES),
    #     }

    #     np.savez_compressed(
    #         npz_path,
    #         t_ds=t_ds_raw,
    #         Y_proc_ds=Y_proc_ds,
    #         peaks_by_sweep=peaks_obj,
    #         onsets_by_sweep=on_obj,
    #         offsets_by_sweep=off_obj,
    #         expmins_by_sweep=exm_obj,
    #         expoffs_by_sweep=exo_obj,
    #         stim_spans_by_sweep=stim_obj,
    #         meta_json=json.dumps(meta),
    #         **y2_kwargs_ds,
    #     )

    #     # ---------- (2) Per-time CSV: each sweep then mean & SEM (downsampled) ----------
    #     csv_time_path = base.with_name(base.name + "_means_by_time.csv")
    #     keys_for_csv = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

    #     # Header
    #     header = ["t"]
    #     for k in keys_for_csv:
    #         if INCLUDE_TRACES:
    #             header += [f"{k}_s{j+1}" for j in range(n_sweeps)]
    #         header += [f"{k}_mean", f"{k}_sem"]

    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_time_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(header)

    #             for i in range(M):
    #                 row = [f"{t_ds_csv[i]:.9f}"]
    #                 for k in keys_for_csv:
    #                     col = y2_ds_by_key[k][i, :]  # values at this time across sweeps
    #                     if INCLUDE_TRACES:
    #                         row += [f"{v:.9g}" for v in col]
    #                     m, sem = self._mean_sem_1d(col)  # finite-only mean/sem
    #                     row += [f"{m:.9g}", f"{sem:.9g}"]
    #                 w.writerow(row)

    #                 if (i % CSV_FLUSH_EVERY) == 0:
    #                     QApplication.processEvents()
    #     finally:
    #         self.unsetCursor()

    #     # ---------- (3) Per-breath CSV ----------
    #     # Helper: format numbers, leave blanks for NaN
    #     def _fmt(v):
    #         try:
    #             if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
    #                 return ""
    #             return f"{float(v):.9g}"
    #         except Exception:
    #             return ""

    #     # Do we have any stim at all?
    #     any_stim_any_sweep = any(len(st.stim_spans_by_sweep.get(s, [])) > 0 for s in range(n_sweeps)) if st.stim_chan else False

    #     csv_breath_path = base.with_name(base.name + "_per_breath.csv")

    #     # Build header
    #     breath_header = ["sweep", "breath", "t", "region"]  # t is normalized to csv_t0
    #     breath_header += [k for k in keys_for_csv]
    #     if any_stim_any_sweep:
    #         for k in keys_for_csv:
    #             breath_header += [f"{k}_baseline", f"{k}_stim", f"{k}_post"]

    #     # Write rows
    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_breath_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(breath_header)

    #             for s in range(n_sweeps):
    #                 # processed y + events
    #                 y_proc = self._get_processed_for(st.analyze_chan, s)
    #                 pks = np.asarray(peaks_by_sweep[s], dtype=int)
    #                 breaths = st.breath_by_sweep.get(s, None)
    #                 if breaths is None and pks.size:
    #                     breaths = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #                 if breaths is None:
    #                     continue

    #                 on = np.asarray(breaths.get("onsets", []), dtype=int)
    #                 if on.size == 0:
    #                     continue

    #                 # per-sweep stim windows (for region tagging)
    #                 spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #                 t0_s = spans[0][0] if spans else None
    #                 t1_s = spans[-1][1] if spans else None

    #                 # compute full-res metric traces for this sweep once (for sampling at onsets)
    #                 y2_full_by_key = {}
    #                 for k in keys_for_csv:
    #                     try:
    #                         y2_full_by_key[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, breaths)
    #                     except Exception:
    #                         y2_full_by_key[k] = None

    #                 for i_b, idx_on in enumerate(on):
    #                     if not (0 <= idx_on < N):
    #                         continue
    #                     t_abs = float(st.t[idx_on])
    #                     t_rel = t_abs - float(csv_t0)

    #                     # region (per sweep)
    #                     if spans:
    #                         in_stim = any((t_abs >= a) and (t_abs <= b) for (a, b) in spans)
    #                         if t0_s is not None and t_abs < t0_s:
    #                             region = "Baseline"
    #                         elif in_stim:
    #                             region = "Stim"
    #                         elif t1_s is not None and t_abs > t1_s:
    #                             region = "Post"
    #                         else:
    #                             region = "None"
    #                     else:
    #                         region = "None"

    #                     row = [str(s+1), str(i_b+1), _fmt(t_rel), region]

    #                     # overall metric value at this breath (sample at onset)
    #                     vals_overall = []
    #                     for k in keys_for_csv:
    #                         arr = y2_full_by_key.get(k, None)
    #                         v = (arr[idx_on] if (arr is not None and len(arr) == N) else np.nan)
    #                         vals_overall.append(v)
    #                         row.append(_fmt(v))

    #                     # window-tagged columns (one non-empty per row, if any stim exists anywhere)
    #                     if any_stim_any_sweep:
    #                         for j, k in enumerate(keys_for_csv):
    #                             v = vals_overall[j]
    #                             if region == "Baseline":
    #                                 row += [_fmt(v), "", ""]
    #                             elif region == "Stim":
    #                                 row += ["", _fmt(v), ""]
    #                             elif region == "Post":
    #                                 row += ["", "", _fmt(v)]
    #                             else:
    #                                 row += ["", "", ""]

    #                     w.writerow(row)

    #                 if (s % 3) == 0:
    #                     QApplication.processEvents()
    #     finally:
    #         self.unsetCursor()

    #     # ---------- done ----------
    #     msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}\n- {csv_breath_path.name}"
    #     print("[save]", msg)
    #     try:
    #         self.statusbar.showMessage(msg, 6000)
    #     except Exception:
    #         pass


    # def _export_all_analyzed_data(self):
    #     """
    #     Save two files:
    #     1) <base>_bundle.npz  (downsampled Y_proc + y2 traces, peaks/breaths, stim spans, meta)
    #     2) <base>_means_by_time.csv  (downsampled time-series: per-sweep (optional), mean, sem)
    #     3) <base>_breaths.csv  (WIDE per-breath table: ALL | BASELINE | STIM | POST blocks)
    #     """
    #     st = self.state
    #     if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
    #         QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
    #         return
    #     if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
    #         QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
    #         return

    #     import numpy as np, csv, json
    #     from PyQt6.QtCore import Qt
    #     from PyQt6.QtWidgets import QApplication

    #     # ---------- knobs ----------
    #     DS_TARGET_HZ      = 50.0
    #     CSV_FLUSH_EVERY   = 2000
    #     INCLUDE_TRACES    = bool(getattr(self, "_csv_include_traces", True))

    #     # ---------- basics ----------
    #     any_ch   = next(iter(st.sweeps.values()))
    #     n_sweeps = any_ch.shape[1]
    #     N        = len(st.t)

    #     # Downsample index used for NPZ + time CSV
    #     ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
    #     ds_idx  = np.arange(0, N, ds_step, dtype=int)
    #     M       = len(ds_idx)

    #     # Global stim zero and duration (union across ALL sweeps)
    #     global_s0, global_s1 = None, None
    #     if st.stim_chan:
    #         for s in range(n_sweeps):
    #             spans = st.stim_spans_by_sweep.get(s, [])
    #             if spans:
    #                 starts = [a for (a, _) in spans]
    #                 ends   = [b for (_, b) in spans]
    #                 m0 = float(min(starts))
    #                 m1 = float(max(ends))
    #                 global_s0 = m0 if global_s0 is None else min(global_s0, m0)
    #                 global_s1 = m1 if global_s1 is None else max(global_s1, m1)
    #     have_global_stim = (global_s0 is not None and global_s1 is not None)
    #     global_dur = (global_s1 - global_s0) if have_global_stim else None

    #     # Time for NPZ (raw) and for CSV (normalized to global_s0 if present)
    #     t_ds_raw = st.t[ds_idx]
    #     csv_t0   = (global_s0 if have_global_stim else 0.0)
    #     t_ds_csv = (st.t - csv_t0)[ds_idx]

    #     # ---------- containers ----------
    #     # Downsampled pleth (M, S)
    #     Y_proc_ds = np.full((M, n_sweeps), np.nan, dtype=float)

    #     # Per-sweep peaks & breaths (ragged -> object arrays for NPZ)
    #     peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []

    #     # Metrics (downsampled time-traces)
    #     all_keys     = self._metric_keys_in_order()
    #     y2_ds_by_key = {k: np.full((M, n_sweeps), np.nan, dtype=float) for k in all_keys}

    #     # ---------- fill per-sweep ----------
    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         Y_proc_ds[:, s] = y_proc[ds_idx]

    #         pks     = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         br      = st.breath_by_sweep.get(s, None)
    #         if br is None and pks.size:
    #             br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = br
    #         if br is None:
    #             br = {
    #                 "onsets":  np.array([], dtype=int),
    #                 "offsets": np.array([], dtype=int),
    #                 "expmins": np.array([], dtype=int),
    #                 "expoffs": np.array([], dtype=int),
    #             }

    #         peaks_by_sweep.append(pks)
    #         on_by_sweep.append(np.asarray(br.get("onsets",  []), dtype=int))
    #         off_by_sweep.append(np.asarray(br.get("offsets", []), dtype=int))
    #         exm_by_sweep.append(np.asarray(br.get("expmins", []), dtype=int))
    #         exo_by_sweep.append(np.asarray(br.get("expoffs", []), dtype=int))

    #         for k in all_keys:
    #             y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
    #             if y2 is not None and len(y2) == N:
    #                 y2_ds_by_key[k][:, s] = y2[ds_idx]

    #     # ---------- (1) NPZ bundle (downsampled) ----------
    #     base     = self._save_dir / self._save_base
    #     npz_path = base.with_name(base.name + "_bundle.npz")

    #     stim_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

    #     peaks_obj = np.array(peaks_by_sweep, dtype=object)
    #     on_obj    = np.array(on_by_sweep,  dtype=object)
    #     off_obj   = np.array(off_by_sweep, dtype=object)
    #     exm_obj   = np.array(exm_by_sweep, dtype=object)
    #     exo_obj   = np.array(exo_by_sweep, dtype=object)

    #     y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}

    #     meta = {
    #         "analyze_channel": st.analyze_chan,
    #         "sr_hz": float(st.sr_hz),
    #         "n_sweeps": int(n_sweeps),
    #         "abf_path": str(getattr(st, "in_path", "")),
    #         "ui_meta": getattr(self, "_save_meta", {}),
    #         "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
    #         "ds_target_hz": float(DS_TARGET_HZ),
    #         "ds_step": int(ds_step),
    #         "csv_time_zero": float(csv_t0),
    #         "csv_includes_traces": bool(INCLUDE_TRACES),
    #     }

    #     np.savez_compressed(
    #         npz_path,
    #         t_ds=t_ds_raw,
    #         Y_proc_ds=Y_proc_ds,
    #         peaks_by_sweep=peaks_obj,
    #         onsets_by_sweep=on_obj,
    #         offsets_by_sweep=off_obj,
    #         expmins_by_sweep=exm_obj,
    #         expoffs_by_sweep=exo_obj,
    #         stim_spans_by_sweep=stim_obj,
    #         meta_json=json.dumps(meta),
    #         **y2_kwargs_ds,
    #     )

    #     # ---------- (2) Per-time CSV ----------
    #     csv_time_path = base.with_name(base.name + "_means_by_time.csv")
    #     keys_for_csv  = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

    #     header = ["t"]
    #     for k in keys_for_csv:
    #         if INCLUDE_TRACES:
    #             header += [f"{k}_s{j+1}" for j in range(n_sweeps)]
    #         header += [f"{k}_mean", f"{k}_sem"]

    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_time_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(header)

    #             for i in range(M):
    #                 row = [f"{t_ds_csv[i]:.9f}"]
    #                 for k in keys_for_csv:
    #                     col = y2_ds_by_key[k][i, :]
    #                     if INCLUDE_TRACES:
    #                         row += [f"{v:.9g}" for v in col]
    #                     m, sem = self._mean_sem_1d(col)
    #                     row += [f"{m:.9g}", f"{sem:.9g}"]
    #                 w.writerow(row)
    #                 if (i % CSV_FLUSH_EVERY) == 0:
    #                     QApplication.processEvents()
    #     finally:
    #         self.unsetCursor()

    #     # ---------- (3) Per-breath CSV (WIDE): ALL | BASELINE | STIM | POST ----------
    #     breaths_path = base.with_name(base.name + "_breaths.csv")

    #     # Columns per block (NO extra 'if_baseline' metric; 'if' only)
    #     BREATH_COLS = [
    #         "sweep", "breath", "t", "region",
    #         "if", "amp_insp", "amp_exp", "area_insp", "area_exp",
    #         "ti", "te", "vent_proxy",
    #     ]

    #     def _headers_for_block(suffix: str | None) -> list[str]:
    #         if not suffix:
    #             return BREATH_COLS[:]
    #         return [f"{c}_{suffix}" for c in BREATH_COLS]

    #     rows_all, rows_bl, rows_st, rows_po = [], [], [], []

    #     # Metrics needed for breathwise table
    #     need_keys = ["if", "amp_insp", "amp_exp", "area_insp", "area_exp", "ti", "te", "vent_proxy"]

    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         pks    = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         br     = st.breath_by_sweep.get(s, None)
    #         if br is None and pks.size:
    #             br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = br
    #         if br is None:
    #             br = {"onsets": np.array([], dtype=int)}

    #         on = np.asarray(br.get("onsets", []), dtype=int)
    #         if on.size < 2:
    #             continue

    #         # Midpoint per breath
    #         mids = (on[:-1] + on[1:]) // 2

    #         # Cache stepwise traces for the metrics we need
    #         traces = {}
    #         for k in need_keys:
    #             if k in metrics.METRICS:
    #                 try:
    #                     traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
    #                 except TypeError:
    #                     traces[k] = None
    #             else:
    #                 traces[k] = None

    #         for i, idx in enumerate(mids, start=1):
    #             # Time relative to global stim start if present (else absolute)
    #             t_rel = float(st.t[int(idx)] - (global_s0 if have_global_stim else 0.0))

    #             # ---- ALL row ----
    #             row_all = [
    #                 str(s + 1),
    #                 str(i),
    #                 f"{t_rel:.9g}",
    #                 "all",
    #             ]
    #             for k in need_keys:
    #                 v = np.nan
    #                 arr = traces.get(k, None)
    #                 if arr is not None and len(arr) == N:
    #                     v = arr[int(idx)]
    #                 row_all.append(f"{v:.9g}" if np.isfinite(v) else "")
    #             rows_all.append(row_all)

    #             # ---- Region rows (use GLOBAL window so all sweeps are included) ----
    #             if not have_global_stim:
    #                 continue

    #             if t_rel < 0:
    #                 tgt_list = rows_bl; region = "Baseline"
    #             elif 0.0 <= t_rel <= global_dur:
    #                 tgt_list = rows_st; region = "Stim"
    #             else:
    #                 tgt_list = rows_po; region = "Post"

    #             row_reg = [
    #                 str(s + 1),
    #                 str(i),
    #                 f"{t_rel:.9g}",
    #                 region,
    #             ]
    #             for k in need_keys:
    #                 v = np.nan
    #                 arr = traces.get(k, None)
    #                 if arr is not None and len(arr) == N:
    #                     v = arr[int(idx)]
    #                 row_reg.append(f"{v:.9g}" if np.isfinite(v) else "")
    #             tgt_list.append(row_reg)

    #     # Compose wide table with blank separators; pad shorter blocks with empties
    #     def _pad_row(row, want_len):
    #         if row is None:
    #             return [""] * want_len
    #         if len(row) < want_len:
    #             return row + [""] * (want_len - len(row))
    #         return row

    #     headers_all = _headers_for_block(None)
    #     headers_bl  = _headers_for_block("baseline")
    #     headers_st  = _headers_for_block("stim")
    #     headers_po  = _headers_for_block("post")

    #     have_stim_blocks = have_global_stim and (len(rows_bl) + len(rows_st) + len(rows_po) > 0)

    #     with open(breaths_path, "w", newline="") as f:
    #         w = csv.writer(f)

    #         if not have_stim_blocks:
    #             w.writerow(headers_all)
    #             for r in rows_all:
    #                 w.writerow(r)
    #         else:
    #             full_header = headers_all + [""] + headers_bl + [""] + headers_st + [""] + headers_po
    #             w.writerow(full_header)

    #             L  = max(len(rows_all), len(rows_bl), len(rows_st), len(rows_po))
    #             LA = len(headers_all); LB = len(headers_bl); LS = len(headers_st); LP = len(headers_po)

    #             for i in range(L):
    #                 ra = rows_all[i] if i < len(rows_all) else None
    #                 rb = rows_bl[i]  if i < len(rows_bl)  else None
    #                 rs = rows_st[i]  if i < len(rows_st)  else None
    #                 rp = rows_po[i]  if i < len(rows_po)  else None

    #                 row = (
    #                     _pad_row(ra, LA) + [""] +
    #                     _pad_row(rb, LB) + [""] +
    #                     _pad_row(rs, LS) + [""] +
    #                     _pad_row(rp, LP)
    #                 )
    #                 w.writerow(row)

    #     # ---------- done ----------
    #     msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}\n- {breaths_path.name}"
    #     print("[save]", msg)
    #     try:
    #         self.statusbar.showMessage(msg, 6000)
    #     except Exception:
    #         pass

    # def _export_all_analyzed_data(self):
    #     """
    #     Save:
    #     1) <base>_bundle.npz  (downsampled Y_proc + y2 traces, peaks/breaths, stim spans, meta)
    #     2) <base>_means_by_time.csv  (downsampled time-series: per-sweep (optional), mean, sem)
    #     3) <base>_breaths.csv  (WIDE per-breath table: ALL | BASELINE | STIM | POST blocks)
    #     4) <base>_summary.pdf  (3-column figure per metric: traces | mean±SEM | histograms)
    #     """
    #     st = self.state
    #     if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
    #         QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
    #         return
    #     if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
    #         QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
    #         return

    #     import numpy as np, csv, json
    #     from PyQt6.QtCore import Qt
    #     from PyQt6.QtWidgets import QApplication

    #     # ---------- knobs ----------
    #     DS_TARGET_HZ      = 50.0
    #     CSV_FLUSH_EVERY   = 2000
    #     INCLUDE_TRACES    = bool(getattr(self, "_csv_include_traces", True))

    #     # ---------- basics ----------
    #     any_ch   = next(iter(st.sweeps.values()))
    #     n_sweeps = any_ch.shape[1]
    #     N        = len(st.t)

    #     # Downsample index used for NPZ + time CSV
    #     ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
    #     ds_idx  = np.arange(0, N, ds_step, dtype=int)
    #     M       = len(ds_idx)

    #     # Global stim zero and duration (union across ALL sweeps)
    #     global_s0, global_s1 = None, None
    #     if st.stim_chan:
    #         for s in range(n_sweeps):
    #             spans = st.stim_spans_by_sweep.get(s, [])
    #             if spans:
    #                 starts = [a for (a, _) in spans]
    #                 ends   = [b for (_, b) in spans]
    #                 m0 = float(min(starts))
    #                 m1 = float(max(ends))
    #                 global_s0 = m0 if global_s0 is None else min(global_s0, m0)
    #                 global_s1 = m1 if global_s1 is None else max(global_s1, m1)
    #     have_global_stim = (global_s0 is not None and global_s1 is not None)
    #     global_dur = (global_s1 - global_s0) if have_global_stim else None

    #     # Time for NPZ (raw) and for CSV (normalized to global_s0 if present)
    #     t_ds_raw = st.t[ds_idx]
    #     csv_t0   = (global_s0 if have_global_stim else 0.0)
    #     t_ds_csv = (st.t - csv_t0)[ds_idx]

    #     # ---------- containers ----------
    #     # Downsampled pleth (M, S)
    #     Y_proc_ds = np.full((M, n_sweeps), np.nan, dtype=float)

    #     # Per-sweep peaks & breaths (ragged -> object arrays for NPZ)
    #     peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []

    #     # Metrics (downsampled time-traces)
    #     all_keys     = self._metric_keys_in_order()
    #     y2_ds_by_key = {k: np.full((M, n_sweeps), np.nan, dtype=float) for k in all_keys}

    #     # NEW: keep full-res processed pleth for CTA panels
    #     Y_full_by_sweep = []

    #     # ---------- fill per-sweep ----------
    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         Y_full_by_sweep.append(y_proc)
    #         Y_proc_ds[:, s] = y_proc[ds_idx]

    #         pks     = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         br      = st.breath_by_sweep.get(s, None)
    #         if br is None and pks.size:
    #             br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = br
    #         if br is None:
    #             br = {
    #                 "onsets":  np.array([], dtype=int),
    #                 "offsets": np.array([], dtype=int),
    #                 "expmins": np.array([], dtype=int),
    #                 "expoffs": np.array([], dtype=int),
    #             }

    #         peaks_by_sweep.append(pks)
    #         on_by_sweep.append(np.asarray(br.get("onsets",  []), dtype=int))
    #         off_by_sweep.append(np.asarray(br.get("offsets", []), dtype=int))
    #         exm_by_sweep.append(np.asarray(br.get("expmins", []), dtype=int))
    #         exo_by_sweep.append(np.asarray(br.get("expoffs", []), dtype=int))

    #         for k in all_keys:
    #             y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
    #             if y2 is not None and len(y2) == N:
    #                 y2_ds_by_key[k][:, s] = y2[ds_idx]

    #     # ---------- (1) NPZ bundle (downsampled) ----------
    #     base     = self._save_dir / self._save_base
    #     npz_path = base.with_name(base.name + "_bundle.npz")

    #     stim_obj = np.empty(n_sweeps, dtype=object)
    #     for s in range(n_sweeps):
    #         spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
    #         stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

    #     peaks_obj = np.array(peaks_by_sweep, dtype=object)
    #     on_obj    = np.array(on_by_sweep,  dtype=object)
    #     off_obj   = np.array(off_by_sweep, dtype=object)
    #     exm_obj   = np.array(exm_by_sweep, dtype=object)
    #     exo_obj   = np.array(exo_by_sweep, dtype=object)

    #     y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}

    #     meta = {
    #         "analyze_channel": st.analyze_chan,
    #         "sr_hz": float(st.sr_hz),
    #         "n_sweeps": int(n_sweeps),
    #         "abf_path": str(getattr(st, "in_path", "")),
    #         "ui_meta": getattr(self, "_save_meta", {}),
    #         "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
    #         "ds_target_hz": float(DS_TARGET_HZ),
    #         "ds_step": int(ds_step),
    #         "csv_time_zero": float(csv_t0),
    #         "csv_includes_traces": bool(INCLUDE_TRACES),
    #     }

    #     np.savez_compressed(
    #         npz_path,
    #         t_ds=t_ds_raw,
    #         Y_proc_ds=Y_proc_ds,
    #         peaks_by_sweep=peaks_obj,
    #         onsets_by_sweep=on_obj,
    #         offsets_by_sweep=off_obj,
    #         expmins_by_sweep=exm_obj,
    #         expoffs_by_sweep=exo_obj,
    #         stim_spans_by_sweep=stim_obj,
    #         meta_json=json.dumps(meta),
    #         **y2_kwargs_ds,
    #     )

    #     # ---------- (2) Per-time CSV ----------
    #     csv_time_path = base.with_name(base.name + "_means_by_time.csv")
    #     keys_for_csv  = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

    #     header = ["t"]
    #     for k in keys_for_csv:
    #         if INCLUDE_TRACES:
    #             header += [f"{k}_s{j+1}" for j in range(n_sweeps)]
    #         header += [f"{k}_mean", f"{k}_sem"]

    #     self.setCursor(Qt.CursorShape.WaitCursor)
    #     try:
    #         with open(csv_time_path, "w", newline="") as f:
    #             w = csv.writer(f)
    #             w.writerow(header)

    #             for i in range(M):
    #                 row = [f"{t_ds_csv[i]:.9f}"]
    #                 for k in keys_for_csv:
    #                     col = y2_ds_by_key[k][i, :]
    #                     if INCLUDE_TRACES:
    #                         row += [f"{v:.9g}" for v in col]
    #                     m, sem = self._mean_sem_1d(col)
    #                     row += [f"{m:.9g}", f"{sem:.9g}"]
    #                 w.writerow(row)
    #                 if (i % CSV_FLUSH_EVERY) == 0:
    #                     QApplication.processEvents()
    #     finally:
    #         self.unsetCursor()

    #     # ---------- (3) Per-breath CSV (WIDE): ALL | BASELINE | STIM | POST ----------
    #     breaths_path = base.with_name(base.name + "_breaths.csv")

    #     BREATH_COLS = [
    #         "sweep", "breath", "t", "region",
    #         "if", "amp_insp", "amp_exp", "area_insp", "area_exp",
    #         "ti", "te", "vent_proxy",
    #     ]
    #     def _headers_for_block(suffix: str | None) -> list[str]:
    #         if not suffix: return BREATH_COLS[:]
    #         return [f"{c}_{suffix}" for c in BREATH_COLS]

    #     rows_all, rows_bl, rows_st, rows_po = [], [], [], []
    #     need_keys = ["if", "amp_insp", "amp_exp", "area_insp", "area_exp", "ti", "te", "vent_proxy"]

    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         pks    = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         br     = st.breath_by_sweep.get(s, None)
    #         if br is None and pks.size:
    #             br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             st.breath_by_sweep[s] = br
    #         if br is None:
    #             br = {"onsets": np.array([], dtype=int)}

    #         on = np.asarray(br.get("onsets", []), dtype=int)
    #         if on.size < 2:
    #             continue

    #         mids = (on[:-1] + on[1:]) // 2

    #         traces = {}
    #         for k in need_keys:
    #             if k in metrics.METRICS:
    #                 traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
    #             else:
    #                 traces[k] = None

    #         for i, idx in enumerate(mids, start=1):
    #             t_rel = float(st.t[int(idx)] - (global_s0 if have_global_stim else 0.0))

    #             row_all = [str(s + 1), str(i), f"{t_rel:.9g}", "all"]
    #             for k in need_keys:
    #                 v = np.nan
    #                 arr = traces.get(k, None)
    #                 if arr is not None and len(arr) == N:
    #                     v = arr[int(idx)]
    #                 row_all.append(f"{v:.9g}" if np.isfinite(v) else "")
    #             rows_all.append(row_all)

    #             if have_global_stim:
    #                 if t_rel < 0:
    #                     tgt_list = rows_bl; region = "Baseline"
    #                 elif 0.0 <= t_rel <= global_dur:
    #                     tgt_list = rows_st; region = "Stim"
    #                 else:
    #                     tgt_list = rows_po; region = "Post"

    #                 row_reg = [str(s + 1), str(i), f"{t_rel:.9g}", region]
    #                 for k in need_keys:
    #                     v = np.nan
    #                     arr = traces.get(k, None)
    #                     if arr is not None and len(arr) == N:
    #                         v = arr[int(idx)]
    #                     row_reg.append(f"{v:.9g}" if np.isfinite(v) else "")
    #                 tgt_list.append(row_reg)

    #     def _pad_row(row, want_len):
    #         if row is None: return [""] * want_len
    #         if len(row) < want_len: return row + [""] * (want_len - len(row))
    #         return row

    #     headers_all = _headers_for_block(None)
    #     headers_bl  = _headers_for_block("baseline")
    #     headers_st  = _headers_for_block("stim")
    #     headers_po  = _headers_for_block("post")

    #     have_stim_blocks = have_global_stim and (len(rows_bl) + len(rows_st) + len(rows_po) > 0)

    #     with open(breaths_path, "w", newline="") as f:
    #         w = csv.writer(f)
    #         if not have_stim_blocks:
    #             w.writerow(headers_all)
    #             for r in rows_all: w.writerow(r)
    #         else:
    #             full_header = headers_all + [""] + headers_bl + [""] + headers_st + [""] + headers_po
    #             w.writerow(full_header)
    #             L  = max(len(rows_all), len(rows_bl), len(rows_st), len(rows_po))
    #             LA = len(headers_all); LB = len(headers_bl); LS = len(headers_st); LP = len(headers_po)
    #             for i in range(L):
    #                 ra = rows_all[i] if i < len(rows_all) else None
    #                 rb = rows_bl[i]  if i < len(rows_bl)  else None
    #                 rs = rows_st[i]  if i < len(rows_st)  else None
    #                 rp = rows_po[i]  if i < len(rows_po)  else None
    #                 row = (
    #                     _pad_row(ra, LA) + [""] +
    #                     _pad_row(rb, LB) + [""] +
    #                     _pad_row(rs, LS) + [""] +
    #                     _pad_row(rp, LP)
    #                 )
    #                 w.writerow(row)

    #     # ---------- (4) Summary PDF ----------
    #     label_by_key = {key: label for (label, key) in metrics.METRIC_SPECS if key in keys_for_csv}
    #     pdf_path = base.with_name(base.name + "_summary.pdf")
    #     try:
    #         self._save_metrics_summary_pdf(
    #             out_path=pdf_path,
    #             t_ds_csv=t_ds_csv,
    #             y2_ds_by_key=y2_ds_by_key,
    #             keys_for_csv=keys_for_csv,
    #             label_by_key=label_by_key,
    #             stim_zero=(global_s0 if have_global_stim else None),
    #             stim_dur=(global_dur if have_global_stim else None),
    #         )
    #         # self._save_summary_pdf(
    #         # base,
    #         # t_ds_csv=t_ds_csv,
    #         # y2_ds_by_key=y2_ds_by_key,
    #         # keys_for_csv=keys_for_csv,
    #         # stim_spans_by_sweep=st.stim_spans_by_sweep,
    #         # Y_proc_full_by_sweep=Y_full_by_sweep,      # NEW
    #         # peaks_by_sweep=peaks_by_sweep,             # you already have this
    #         # sr_hz=st.sr_hz,                            # NEW
    #         # csv_time_zero=csv_t0,                        # (optional, unchanged)
    #         # cta_half_width_s=0.2,
    #         # )

    #     except Exception as e:
    #         print(f"[save][summary-pdf] skipped: {e}")

    #     # ---------- done ----------
    #     msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}\n- {breaths_path.name}\n- {pdf_path.name}"
    #     print("[save]", msg)
    #     try:
    #         self.statusbar.showMessage(msg, 6000)
    #     except Exception:
    #         pass

    def _export_all_analyzed_data(self):
        """
        Save:
        1) <base>_bundle.npz  (downsampled Y_proc + y2 traces, peaks/breaths, stim spans, meta)
        2) <base>_means_by_time.csv  (downsampled time-series: per-sweep (optional), mean, sem)
            + appended block of normalized per-sweep traces/mean/sem (suffix *_norm)
        3) <base>_breaths.csv  (WIDE per-breath table: ALL | BASELINE | STIM | POST blocks)
            + appended duplicate blocks with normalized values (all headers suffixed *_norm)
        4) <base>_summary.pdf  (figure; unchanged here)
        """
        st = self.state
        if not getattr(self, "_save_dir", None) or not getattr(self, "_save_base", None):
            QMessageBox.warning(self, "Save analyzed data", "Choose a save location/name first.")
            return
        if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            QMessageBox.warning(self, "Save analyzed data", "No analyzed data available.")
            return

        import numpy as np, csv, json
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        # ---------- knobs ----------
        DS_TARGET_HZ    = 50.0
        CSV_FLUSH_EVERY = 2000
        INCLUDE_TRACES  = bool(getattr(self, "_csv_include_traces", True))
        # normalization window (seconds before t=0); override via self._norm_window_s if you like
        NORM_BASELINE_WINDOW_S = float(getattr(self, "_norm_window_s", 10.0))
        EPS_BASE = 1e-12  # avoid divide-by-zero

        # ---------- basics ----------
        any_ch   = next(iter(st.sweeps.values()))
        n_sweeps = any_ch.shape[1]
        N        = len(st.t)

        # Downsample index used for NPZ + time CSV
        ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
        ds_idx  = np.arange(0, N, ds_step, dtype=int)
        M       = len(ds_idx)

        # Global stim zero and duration (union across ALL sweeps)
        global_s0, global_s1 = None, None
        if st.stim_chan:
            for s in range(n_sweeps):
                spans = st.stim_spans_by_sweep.get(s, [])
                if spans:
                    starts = [a for (a, _) in spans]
                    ends   = [b for (_, b) in spans]
                    m0 = float(min(starts))
                    m1 = float(max(ends))
                    global_s0 = m0 if global_s0 is None else min(global_s0, m0)
                    global_s1 = m1 if global_s1 is None else max(global_s1, m1)
        have_global_stim = (global_s0 is not None and global_s1 is not None)
        global_dur = (global_s1 - global_s0) if have_global_stim else None

        # Time for NPZ (raw) and for CSV (normalized to global_s0 if present)
        t_ds_raw = st.t[ds_idx]
        csv_t0   = (global_s0 if have_global_stim else 0.0)
        t_ds_csv = (st.t - csv_t0)[ds_idx]

        # ---------- containers ----------
        Y_proc_ds = np.full((M, n_sweeps), np.nan, dtype=float)
        peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []

        all_keys     = self._metric_keys_in_order()
        y2_ds_by_key = {k: np.full((M, n_sweeps), np.nan, dtype=float) for k in all_keys}

        # Keep full-res processed pleth (for other uses)
        Y_full_by_sweep = []

        # ---------- fill per-sweep ----------
        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            Y_full_by_sweep.append(y_proc)
            Y_proc_ds[:, s] = y_proc[ds_idx]

            pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
            br  = st.breath_by_sweep.get(s, None)
            if br is None and pks.size:
                br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                st.breath_by_sweep[s] = br
            if br is None:
                br = {
                    "onsets":  np.array([], dtype=int),
                    "offsets": np.array([], dtype=int),
                    "expmins": np.array([], dtype=int),
                    "expoffs": np.array([], dtype=int),
                }

            peaks_by_sweep.append(pks)
            on_by_sweep.append(np.asarray(br.get("onsets",  []), dtype=int))
            off_by_sweep.append(np.asarray(br.get("offsets", []), dtype=int))
            exm_by_sweep.append(np.asarray(br.get("expmins", []), dtype=int))
            exo_by_sweep.append(np.asarray(br.get("expoffs", []), dtype=int))

            for k in all_keys:
                y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
                if y2 is not None and len(y2) == N:
                    y2_ds_by_key[k][:, s] = y2[ds_idx]

        # ---------- (1) NPZ bundle (downsampled) ----------
        base     = self._save_dir / self._save_base
        npz_path = base.with_name(base.name + "_bundle.npz")

        stim_obj = np.empty(n_sweeps, dtype=object)
        for s in range(n_sweeps):
            spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
            stim_obj[s] = np.array(spans, dtype=float) if spans else np.array([], dtype=float).reshape(0, 2)

        peaks_obj = np.array(peaks_by_sweep, dtype=object)
        on_obj    = np.array(on_by_sweep,  dtype=object)
        off_obj   = np.array(off_by_sweep, dtype=object)
        exm_obj   = np.array(exm_by_sweep, dtype=object)
        exo_obj   = np.array(exo_by_sweep, dtype=object)

        y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in all_keys}

        meta = {
            "analyze_channel": st.analyze_chan,
            "sr_hz": float(st.sr_hz),
            "n_sweeps": int(n_sweeps),
            "abf_path": str(getattr(st, "in_path", "")),
            "ui_meta": getattr(self, "_save_meta", {}),
            "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
            "ds_target_hz": float(DS_TARGET_HZ),
            "ds_step": int(ds_step),
            "csv_time_zero": float(csv_t0),
            "csv_includes_traces": bool(INCLUDE_TRACES),
            "norm_window_s": float(NORM_BASELINE_WINDOW_S),
        }

        np.savez_compressed(
            npz_path,
            t_ds=t_ds_raw,
            Y_proc_ds=Y_proc_ds,
            peaks_by_sweep=peaks_obj,
            onsets_by_sweep=on_obj,
            offsets_by_sweep=off_obj,
            expmins_by_sweep=exm_obj,
            expoffs_by_sweep=exo_obj,
            stim_spans_by_sweep=stim_obj,
            meta_json=json.dumps(meta),
            **y2_kwargs_ds,
        )

        # ---------- helpers for normalization ----------
        def _per_sweep_baseline_for_time(A_ds: np.ndarray) -> np.ndarray:
            """
            A_ds: (M,S) downsampled metric matrix.
            Returns b[S]: mean over last NORM_BASELINE_WINDOW_S before 0; fallback to first W after 0.
            """
            b = np.full((A_ds.shape[1],), np.nan, dtype=float)
            mask_pre  = (t_ds_csv >= -NORM_BASELINE_WINDOW_S) & (t_ds_csv < 0.0)
            mask_post = (t_ds_csv >= 0.0) & (t_ds_csv <=  NORM_BASELINE_WINDOW_S)
            for s in range(A_ds.shape[1]):
                col = A_ds[:, s]
                vals = col[mask_pre]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    vals = col[mask_post]
                    vals = vals[np.isfinite(vals)]
                if vals.size:
                    b[s] = float(np.mean(vals))
            return b

        def _normalize_matrix_by_baseline(A_ds: np.ndarray, b: np.ndarray) -> np.ndarray:
            out = np.full_like(A_ds, np.nan)
            for s in range(A_ds.shape[1]):
                bs = b[s]
                if np.isfinite(bs) and abs(bs) > EPS_BASE:
                    out[:, s] = A_ds[:, s] / bs
            return out

        # ---------- (2) Per-time CSV (raw + normalized appended) ----------
        csv_time_path = base.with_name(base.name + "_means_by_time.csv")
        keys_for_csv  = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]

        # Build normalized stacks per metric
        y2_ds_by_key_norm = {}
        baseline_by_key   = {}
        for k in keys_for_csv:
            b = _per_sweep_baseline_for_time(y2_ds_by_key[k])
            baseline_by_key[k] = b
            y2_ds_by_key_norm[k] = _normalize_matrix_by_baseline(y2_ds_by_key[k], b)

        # headers: raw first (unchanged), then the same pattern with *_norm suffix
        header = ["t"]
        for k in keys_for_csv:
            if INCLUDE_TRACES:
                header += [f"{k}_s{j+1}" for j in range(n_sweeps)]
            header += [f"{k}_mean", f"{k}_sem"]

        # normalized headers appended
        for k in keys_for_csv:
            if INCLUDE_TRACES:
                header += [f"{k}_norm_s{j+1}" for j in range(n_sweeps)]
            header += [f"{k}_norm_mean", f"{k}_norm_sem"]

        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            with open(csv_time_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)

                for i in range(M):
                    row = [f"{t_ds_csv[i]:.9f}"]

                    # RAW block
                    for k in keys_for_csv:
                        col = y2_ds_by_key[k][i, :]
                        if INCLUDE_TRACES:
                            row += [f"{v:.9g}" if np.isfinite(v) else "" for v in col]
                        m, sem = self._mean_sem_1d(col)
                        row += [f"{m:.9g}", f"{sem:.9g}"]

                    # NORMALIZED block
                    for k in keys_for_csv:
                        colN = y2_ds_by_key_norm[k][i, :]
                        if INCLUDE_TRACES:
                            row += [f"{v:.9g}" if np.isfinite(v) else "" for v in colN]
                        mN, semN = self._mean_sem_1d(colN)
                        row += [f"{mN:.9g}", f"{semN:.9g}"]

                    w.writerow(row)
                    if (i % CSV_FLUSH_EVERY) == 0:
                        QApplication.processEvents()
        finally:
            self.unsetCursor()

        # ---------- (3) Per-breath CSV (WIDE) ----------
        breaths_path = base.with_name(base.name + "_breaths.csv")

        BREATH_COLS = [
            "sweep", "breath", "t", "region",
            "if", "amp_insp", "amp_exp", "area_insp", "area_exp",
            "ti", "te", "vent_proxy",
        ]
        def _headers_for_block(suffix: str | None) -> list[str]:
            if not suffix: return BREATH_COLS[:]
            return [f"{c}_{suffix}" for c in BREATH_COLS]

        def _headers_for_block_norm(suffix: str | None) -> list[str]:
            # duplicate the same block but suffix *all* column names with _norm
            base = _headers_for_block(suffix)
            return [h + "_norm" for h in base]

        rows_all, rows_bl, rows_st, rows_po = [], [], [], []
        rows_all_N, rows_bl_N, rows_st_N, rows_po_N = [], [], [], []

        need_keys = ["if", "amp_insp", "amp_exp", "area_insp", "area_exp", "ti", "te", "vent_proxy"]

        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            pks    = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
            br     = st.breath_by_sweep.get(s, None)
            if br is None and pks.size:
                br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                st.breath_by_sweep[s] = br
            if br is None:
                br = {"onsets": np.array([], dtype=int)}

            on = np.asarray(br.get("onsets", []), dtype=int)
            if on.size < 2:
                continue

            mids = (on[:-1] + on[1:]) // 2

            traces = {}
            for k in need_keys:
                if k in metrics.METRICS:
                    traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
                else:
                    traces[k] = None

            # Per-sweep breath-based baselines for normalization (use breath midpoints)
            # Window: last W seconds before t=0 (fallback: first W seconds after 0)
            # Compute t_rel for each breath midpoint once:
            t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)
            mask_pre_b  = (t_rel_all >= -NORM_BASELINE_WINDOW_S) & (t_rel_all < 0.0)
            mask_post_b = (t_rel_all >=  0.0) & (t_rel_all <= NORM_BASELINE_WINDOW_S)

            b_by_k = {}
            for k in need_keys:
                arr = traces.get(k, None)
                if arr is None or len(arr) != N:
                    b_by_k[k] = np.nan
                    continue
                vals = arr[mids[mask_pre_b]]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    vals = arr[mids[mask_post_b]]
                    vals = vals[np.isfinite(vals)]
                b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan

            for i, idx in enumerate(mids, start=1):
                t_rel = float(st.t[int(idx)] - (global_s0 if have_global_stim else 0.0))

                # ----- RAW: ALL
                row_all = [str(s + 1), str(i), f"{t_rel:.9g}", "all"]
                for k in need_keys:
                    v = np.nan
                    arr = traces.get(k, None)
                    if arr is not None and len(arr) == N:
                        v = arr[int(idx)]
                    row_all.append(f"{v:.9g}" if np.isfinite(v) else "")
                rows_all.append(row_all)

                # ----- NORM: ALL (duplicate id columns + normalized metrics)
                row_allN = [str(s + 1), str(i), f"{t_rel:.9g}", "all"]
                for k in need_keys:
                    v = np.nan
                    arr = traces.get(k, None)
                    if arr is not None and len(arr) == N:
                        v = arr[int(idx)]
                    b = b_by_k.get(k, np.nan)
                    vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                    row_allN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
                rows_all_N.append(row_allN)

                if have_global_stim:
                    if t_rel < 0:
                        tgt_list = rows_bl; tgt_listN = rows_bl_N; region = "Baseline"
                    elif 0.0 <= t_rel <= global_dur:
                        tgt_list = rows_st; tgt_listN = rows_st_N; region = "Stim"
                    else:
                        tgt_list = rows_po; tgt_listN = rows_po_N; region = "Post"

                    # RAW regional row
                    row_reg = [str(s + 1), str(i), f"{t_rel:.9g}", region]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        row_reg.append(f"{v:.9g}" if np.isfinite(v) else "")
                    tgt_list.append(row_reg)

                    # NORM regional row
                    row_regN = [str(s + 1), str(i), f"{t_rel:.9g}", region]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        b = b_by_k.get(k, np.nan)
                        vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                        row_regN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
                    tgt_listN.append(row_regN)

        def _pad_row(row, want_len):
            if row is None: return [""] * want_len
            if len(row) < want_len: return row + [""] * (want_len - len(row))
            return row

        headers_all = _headers_for_block(None)
        headers_bl  = _headers_for_block("baseline")
        headers_st  = _headers_for_block("stim")
        headers_po  = _headers_for_block("post")

        headers_allN = _headers_for_block_norm(None)
        headers_blN  = _headers_for_block_norm("baseline")
        headers_stN  = _headers_for_block_norm("stim")
        headers_poN  = _headers_for_block_norm("post")

        have_stim_blocks = have_global_stim and (len(rows_bl) + len(rows_st) + len(rows_po) > 0)

        with open(breaths_path, "w", newline="") as f:
            w = csv.writer(f)
            if not have_stim_blocks:
                # RAW + NORM (ALL only)
                full_header = headers_all + [""] + headers_allN
                w.writerow(full_header)
                L = max(len(rows_all), len(rows_all_N))
                LA = len(headers_all); LAN = len(headers_allN)
                for i in range(L):
                    ra  = rows_all[i]   if i < len(rows_all)   else None
                    raN = rows_all_N[i] if i < len(rows_all_N) else None
                    row = _pad_row(ra, LA) + [""] + _pad_row(raN, LAN)
                    w.writerow(row)
            else:
                # RAW blocks, then NORM blocks
                full_header = (
                    headers_all + [""] + headers_bl + [""] + headers_st + [""] + headers_po + [""] +
                    headers_allN + [""] + headers_blN + [""] + headers_stN + [""] + headers_poN
                )
                w.writerow(full_header)

                L = max(
                    len(rows_all), len(rows_bl), len(rows_st), len(rows_po),
                    len(rows_all_N), len(rows_bl_N), len(rows_st_N), len(rows_po_N),
                )
                LA = len(headers_all); LB = len(headers_bl); LS = len(headers_st); LP = len(headers_po)
                LAN = len(headers_allN); LBN = len(headers_blN); LSN = len(headers_stN); LPN = len(headers_poN)

                for i in range(L):
                    ra  = rows_all[i]   if i < len(rows_all)   else None
                    rb  = rows_bl[i]    if i < len(rows_bl)    else None
                    rs  = rows_st[i]    if i < len(rows_st)    else None
                    rp  = rows_po[i]    if i < len(rows_po)    else None
                    raN = rows_all_N[i] if i < len(rows_all_N) else None
                    rbN = rows_bl_N[i]  if i < len(rows_bl_N)  else None
                    rsN = rows_st_N[i]  if i < len(rows_st_N)  else None
                    rpN = rows_po_N[i]  if i < len(rows_po_N)  else None

                    row = (
                        _pad_row(ra, LA) + [""] +
                        _pad_row(rb, LB) + [""] +
                        _pad_row(rs, LS) + [""] +
                        _pad_row(rp, LP) + [""] +
                        _pad_row(raN, LAN) + [""] +
                        _pad_row(rbN, LBN) + [""] +
                        _pad_row(rsN, LSN) + [""] +
                        _pad_row(rpN, LPN)
                    )
                    w.writerow(row)

        # ---------- (4) Summary PDF ----------
        keys_for_timeplots = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]
        label_by_key = {key: label for (label, key) in metrics.METRIC_SPECS if key in keys_for_timeplots}
        pdf_path = base.with_name(base.name + "_summary.pdf")
        try:
            self._save_metrics_summary_pdf(
                out_path=pdf_path,
                t_ds_csv=t_ds_csv,
                y2_ds_by_key=y2_ds_by_key,
                keys_for_csv=keys_for_timeplots,
                label_by_key=label_by_key,
                stim_zero=(global_s0 if have_global_stim else None),
                stim_dur=(global_dur if have_global_stim else None),
            )
        except Exception as e:
            print(f"[save][summary-pdf] skipped: {e}")

        # ---------- done ----------
        msg = f"Saved:\n- {npz_path.name}\n- {csv_time_path.name}\n- {breaths_path.name}\n- {pdf_path.name}"
        print("[save]", msg)
        try:
            self.statusbar.showMessage(msg, 6000)
        except Exception:
            pass


    # def _save_metrics_summary_pdf(
    #     self,
    #     out_path,
    #     t_ds_csv: np.ndarray,
    #     y2_ds_by_key: dict,
    #     keys_for_csv: list[str],
    #     label_by_key: dict[str, str],
    #     stim_zero: float | None,
    #     stim_dur: float | None,
    # ):
    #     """
    #     Build a PDF: rows = metrics, cols = [all sweeps | mean±SEM | histograms for ALL/Baseline/Stim/Post].
    #     Uses downsampled time series for columns 1–2, and per-breath values for histograms.
    #     """
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     st = self.state
    #     n_sweeps = next(iter(y2_ds_by_key.values())).shape[1] if y2_ds_by_key else 0
    #     M = len(t_ds_csv)
    #     have_stim = (stim_zero is not None and stim_dur is not None)

    #     # Collect per-breath values by metric + region for histograms
    #     hist_vals = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
    #     need_keys = set(keys_for_csv)

    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         pks    = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         br     = self.state.breath_by_sweep.get(s, None)
    #         if br is None and pks.size:
    #             br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             self.state.breath_by_sweep[s] = br
    #         if br is None:
    #             br = {"onsets": np.array([], dtype=int)}

    #         on = np.asarray(br.get("onsets", []), dtype=int)
    #         if on.size < 2:
    #             continue
    #         mids = (on[:-1] + on[1:]) // 2

    #         # Precompute all needed metric traces for this sweep
    #         traces = {}
    #         for k in need_keys:
    #             try:
    #                 traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
    #             except TypeError:
    #                 traces[k] = None

    #         for idx in mids:
    #             t_rel = float(st.t[int(idx)] - (stim_zero or 0.0))
    #             for k in need_keys:
    #                 arr = traces.get(k, None)
    #                 if arr is None or len(arr) != len(st.t):
    #                     continue
    #                 v = float(arr[int(idx)])
    #                 if not np.isfinite(v):
    #                     continue
    #                 hist_vals[k]["all"].append(v)
    #                 if have_stim:
    #                     if t_rel < 0:
    #                         hist_vals[k]["baseline"].append(v)
    #                     elif 0.0 <= t_rel <= stim_dur:
    #                         hist_vals[k]["stim"].append(v)
    #                     else:
    #                         hist_vals[k]["post"].append(v)

    #     # Figure grid
    #     nrows = max(1, len(keys_for_csv))
    #     fig_w = 13
    #     fig_h = max(4.0, 2.8 * nrows)
    #     fig, axes = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
    #     plt.subplots_adjust(hspace=0.6, wspace=0.25)

    #     for r, k in enumerate(keys_for_csv):
    #         label = label_by_key.get(k, k)

    #         # --- col 1: all sweeps overlaid ---
    #         ax1 = axes[r, 0]
    #         Y = y2_ds_by_key.get(k, None)
    #         if Y is not None and Y.shape[0] == M:
    #             for s in range(n_sweeps):
    #                 ax1.plot(t_ds_csv, Y[:, s], lw=0.8, alpha=0.45)
    #         if have_stim:
    #             ax1.axvspan(0.0, stim_dur, color="#4c78a8", alpha=0.12)
    #             ax1.axvline(0.0, color="#4c78a8", lw=1.0, alpha=0.6)
    #             ax1.axvline(stim_dur, color="#4c78a8", lw=1.0, alpha=0.6, ls="--")
    #         ax1.set_title(f"{label} — all sweeps")
    #         if r == nrows - 1:
    #             ax1.set_xlabel("Time (s, rel. stim onset)")

    #         # --- col 2: mean ± SEM ---
    #         ax2 = axes[r, 1]
    #         if Y is not None and Y.shape[0] == M:
    #             with np.errstate(invalid="ignore"):
    #                 mean = np.nanmean(Y, axis=1)
    #                 n    = np.sum(np.isfinite(Y), axis=1)
    #                 std  = np.nanstd(Y, axis=1, ddof=1)
    #                 sem  = np.where(n >= 2, std / np.sqrt(n), np.nan)
    #             ax2.plot(t_ds_csv, mean, lw=1.8)
    #             ax2.fill_between(t_ds_csv, mean - sem, mean + sem, alpha=0.25, linewidth=0)
    #         if have_stim:
    #             ax2.axvspan(0.0, stim_dur, color="#4c78a8", alpha=0.12)
    #             ax2.axvline(0.0, color="#4c78a8", lw=1.0, alpha=0.6)
    #             ax2.axvline(stim_dur, color="#4c78a8", lw=1.0, alpha=0.6, ls="--")
    #         ax2.set_title(f"{label} — mean ± SEM")
    #         if r == nrows - 1:
    #             ax2.set_xlabel("Time (s, rel. stim onset)")

    #         # --- col 3: histograms (normalized) ---
    #         ax3 = axes[r, 2]
    #         vals_all = np.asarray(hist_vals[k]["all"], dtype=float)
    #         if vals_all.size:
    #             bins = "auto"
    #             ax3.hist(vals_all, bins=bins, density=True, alpha=0.45, label="All")
    #         if have_stim:
    #             for name, disp, col in [
    #                 ("baseline", "Baseline", "#72b7b2"),
    #                 ("stim",     "Stim",     "#e45756"),
    #                 ("post",     "Post",     "#f58518"),
    #             ]:
    #                 arr = np.asarray(hist_vals[k][name], dtype=float)
    #                 if arr.size:
    #                     ax3.hist(arr, bins="auto", density=True, alpha=0.45, label=disp)
    #         ax3.set_title(f"{label} — distribution")
    #         ax3.set_ylabel("Density")
    #         ax3.legend(loc="best", fontsize=8)

    #     fig.suptitle("PlethApp summary", y=0.995, fontsize=12)
    #     fig.tight_layout()
    #     fig.savefig(out_path, dpi=150)
    #     plt.close(fig)


    def _mean_sem_1d(self, arr: np.ndarray):
        """Finite-only mean and SEM (ddof=1) for a 1D array. Returns (mean, sem).
        If no finite values -> (nan, nan). If only 1 finite value -> (mean, nan)."""
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        n = int(finite.sum())
        if n == 0:
            return (np.nan, np.nan)
        vals = arr[finite]
        m = float(np.mean(vals))
        if n >= 2:
            s = float(np.std(vals, ddof=1))
            sem = s / np.sqrt(n)
        else:
            sem = np.nan
        return (m, sem)

    # def _save_metrics_summary_pdf(
    #     self,
    #     out_path,
    #     t_ds_csv: np.ndarray,
    #     y2_ds_by_key: dict,
    #     keys_for_csv: list[str],
    #     label_by_key: dict[str, str],
    #     stim_zero: float | None,
    #     stim_dur: float | None,
    # ):
    #     """
    #     Build a PDF: rows = metrics, cols = [all sweeps | mean±SEM | histograms as line curves].
    #     Uses downsampled time series for columns 1–2, and per-breath values for histograms.
    #     """
    #     import numpy as np
    #     import matplotlib.pyplot as plt

    #     st = self.state
    #     n_sweeps = next(iter(y2_ds_by_key.values())).shape[1] if y2_ds_by_key else 0
    #     M = len(t_ds_csv)
    #     have_stim = (stim_zero is not None and stim_dur is not None)

    #     # Collect per-breath values by metric + region for histograms
    #     hist_vals = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
    #     need_keys = set(keys_for_csv)

    #     for s in range(n_sweeps):
    #         y_proc = self._get_processed_for(st.analyze_chan, s)
    #         pks    = np.asarray(self.state.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
    #         br     = self.state.breath_by_sweep.get(s, None)
    #         if br is None and pks.size:
    #             br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
    #             self.state.breath_by_sweep[s] = br
    #         if br is None:
    #             br = {"onsets": np.array([], dtype=int)}

    #         on = np.asarray(br.get("onsets", []), dtype=int)
    #         if on.size < 2:
    #             continue
    #         mids = (on[:-1] + on[1:]) // 2

    #         # Precompute needed metric traces for this sweep
    #         traces = {}
    #         for k in need_keys:
    #             try:
    #                 traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
    #             except TypeError:
    #                 traces[k] = None

    #         for idx in mids:
    #             t_rel = float(st.t[int(idx)] - (stim_zero or 0.0))
    #             for k in need_keys:
    #                 arr = traces.get(k, None)
    #                 if arr is None or len(arr) != len(st.t):
    #                     continue
    #                 v = float(arr[int(idx)])
    #                 if not np.isfinite(v):
    #                     continue
    #                 hist_vals[k]["all"].append(v)
    #                 if have_stim:
    #                     if t_rel < 0:
    #                         hist_vals[k]["baseline"].append(v)
    #                     elif 0.0 <= t_rel <= stim_dur:
    #                         hist_vals[k]["stim"].append(v)
    #                     else:
    #                         hist_vals[k]["post"].append(v)

    #     # Figure grid
    #     nrows = max(1, len(keys_for_csv))
    #     fig_w = 13
    #     fig_h = max(4.0, 2.8 * nrows)
    #     fig, axes = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
    #     plt.subplots_adjust(hspace=0.6, wspace=0.25)

    #     for r, k in enumerate(keys_for_csv):
    #         label = label_by_key.get(k, k)

    #         # --- col 1: all sweeps overlaid ---
    #         ax1 = axes[r, 0]
    #         Y = y2_ds_by_key.get(k, None)
    #         if Y is not None and Y.shape[0] == M:
    #             for s in range(n_sweeps):
    #                 ax1.plot(t_ds_csv, Y[:, s], lw=0.8, alpha=0.45)
    #         if have_stim:
    #             ax1.axvspan(0.0, stim_dur, color="#4c78a8", alpha=0.12)
    #             ax1.axvline(0.0, color="#4c78a8", lw=1.0, alpha=0.6)
    #             ax1.axvline(stim_dur, color="#4c78a8", lw=1.0, alpha=0.6, ls="--")
    #         ax1.set_title(f"{label} — all sweeps")
    #         if r == nrows - 1:
    #             ax1.set_xlabel("Time (s, rel. stim onset)")

    #         # --- col 2: mean ± SEM ---
    #         ax2 = axes[r, 1]
    #         if Y is not None and Y.shape[0] == M:
    #             with np.errstate(invalid="ignore"):
    #                 mean = np.nanmean(Y, axis=1)
    #                 n    = np.sum(np.isfinite(Y), axis=1)
    #                 std  = np.nanstd(Y, axis=1, ddof=1)
    #                 sem  = np.where(n >= 2, std / np.sqrt(n), np.nan)
    #             ax2.plot(t_ds_csv, mean, lw=1.8)
    #             ax2.fill_between(t_ds_csv, mean - sem, mean + sem, alpha=0.25, linewidth=0)
    #         if have_stim:
    #             ax2.axvspan(0.0, stim_dur, color="#4c78a8", alpha=0.12)
    #             ax2.axvline(0.0, color="#4c78a8", lw=1.0, alpha=0.6)
    #             ax2.axvline(stim_dur, color="#4c78a8", lw=1.0, alpha=0.6, ls="--")
    #         ax2.set_title(f"{label} — mean ± SEM")
    #         if r == nrows - 1:
    #             ax2.set_xlabel("Time (s, rel. stim onset)")

    #         # --- col 3: line histograms (normalized densities) ---
    #         ax3 = axes[r, 2]

    #         # Build common bin edges across all groups for this metric
    #         all_groups = []
    #         for nm in ("all", "baseline", "stim", "post"):
    #             vals = np.asarray(hist_vals[k][nm], dtype=float)
    #             if vals.size:
    #                 all_groups.append(vals)
    #         if len(all_groups):
    #             # Use combined data to choose bins, then plot each group as a line
    #             combined = np.concatenate(all_groups)
    #             edges = np.histogram_bin_edges(combined, bins="auto")
    #             centers = 0.5 * (edges[:-1] + edges[1:])

    #             def _plot_line(vals, lbl, style_kw):
    #                 vals = np.asarray(vals, dtype=float)
    #                 if vals.size == 0:
    #                     return
    #                 dens, _ = np.histogram(vals, bins=edges, density=True)
    #                 ax3.plot(centers, dens, **style_kw, label=lbl)

    #             # All breaths
    #             _plot_line(hist_vals[k]["all"], "All", dict(lw=1.8))

    #             if have_stim:
    #                 _plot_line(hist_vals[k]["baseline"], "Baseline", dict(lw=1.6))
    #                 _plot_line(hist_vals[k]["stim"],     "Stim",     dict(lw=1.6, ls="--"))
    #                 _plot_line(hist_vals[k]["post"],     "Post",     dict(lw=1.6, ls=":"))

    #         ax3.set_title(f"{label} — distribution (density)")
    #         ax3.set_ylabel("Density")
    #         ax3.legend(loc="best", fontsize=8)

    #     fig.suptitle("PlethApp summary", y=0.995, fontsize=12)
    #     fig.tight_layout()
    #     fig.savefig(out_path, dpi=150)
    #     plt.close(fig)

    def _save_metrics_summary_pdf(
        self,
        out_path,
        t_ds_csv: np.ndarray,
        y2_ds_by_key: dict,
        keys_for_csv: list[str],
        label_by_key: dict[str, str],
        stim_zero: float | None,
        stim_dur: float | None,
    ):
        """
        Build a two-page PDF:
        • Page 1: rows = metrics, cols = [all sweeps | mean±SEM | histograms] using RAW data
        • Page 2: same layout, using NORMALIZED data (per sweep, per metric baseline)

        Normalization baseline per sweep:
        - mean over the last W seconds before t=0 (W = self._norm_window_s, default 10.0)
        - fallback to first W seconds after 0 if no pre-stim samples exist
        - value_norm = value / baseline; unstable divisions → NaN
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        st = self.state
        n_sweeps = next(iter(y2_ds_by_key.values())).shape[1] if y2_ds_by_key else 0
        M = len(t_ds_csv)
        have_stim = (stim_zero is not None and stim_dur is not None)

        # --- normalization knobs (match exporter) ---
        NORM_BASELINE_WINDOW_S = float(getattr(self, "_norm_window_s", 10.0))
        EPS_BASE = 1e-12

        # ---------- Helpers ----------
        def _per_sweep_baseline_time(A_ds: np.ndarray) -> np.ndarray:
            """Baseline per sweep (time-series): mean over last W sec before 0; fallback to first W sec after 0."""
            b = np.full((A_ds.shape[1],), np.nan, dtype=float)
            mask_pre  = (t_ds_csv >= -NORM_BASELINE_WINDOW_S) & (t_ds_csv < 0.0)
            mask_post = (t_ds_csv >=  0.0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)
            for s in range(A_ds.shape[1]):
                col = A_ds[:, s]
                vals = col[mask_pre]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    vals = col[mask_post]
                    vals = vals[np.isfinite(vals)]
                if vals.size:
                    b[s] = float(np.mean(vals))
            return b

        def _normalize_matrix(A_ds: np.ndarray, b: np.ndarray) -> np.ndarray:
            out = np.full_like(A_ds, np.nan)
            for s in range(A_ds.shape[1]):
                bs = b[s]
                if np.isfinite(bs) and abs(bs) > EPS_BASE:
                    out[:, s] = A_ds[:, s] / bs
            return out

        def _build_hist_vals_raw_and_norm():
            """Collect per-breath RAW and NORM values for histograms by metric/region."""
            hist_raw = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            hist_nrm = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            need_keys = set(keys_for_csv)

            for s in range(n_sweeps):
                y_proc = self._get_processed_for(st.analyze_chan, s)
                pks    = np.asarray(self.state.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br     = self.state.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    self.state.breath_by_sweep[s] = br
                if br is None:
                    br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue
                mids = (on[:-1] + on[1:]) // 2

                # Precompute metric traces for this sweep
                traces = {}
                for k in need_keys:
                    try:
                        traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br)
                    except TypeError:
                        traces[k] = None

                # Per-sweep breath-based baselines (use breath midpoints; match exporter)
                t_rel_all = (st.t[mids] - (stim_zero or 0.0)).astype(float)
                mask_pre_b  = (t_rel_all >= -NORM_BASELINE_WINDOW_S) & (t_rel_all < 0.0)
                mask_post_b = (t_rel_all >=  0.0) & (t_rel_all <= NORM_BASELINE_WINDOW_S)

                b_by_k = {}
                for k in need_keys:
                    arr = traces.get(k, None)
                    if arr is None or len(arr) != len(st.t):
                        b_by_k[k] = np.nan
                        continue
                    vals = arr[mids[mask_pre_b]]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = arr[mids[mask_post_b]]
                        vals = vals[np.isfinite(vals)]
                    b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan

                # Fill raw + normalized buckets
                for idx, t_rel in zip(mids, t_rel_all):
                    for k in need_keys:
                        arr = traces.get(k, None)
                        if arr is None or len(arr) != len(st.t):
                            continue
                        v = float(arr[int(idx)])
                        if not np.isfinite(v):
                            continue
                        # raw
                        hist_raw[k]["all"].append(v)
                        # norm
                        b = b_by_k.get(k, np.nan)
                        vn = (v / b) if (np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                        if np.isfinite(vn):
                            hist_nrm[k]["all"].append(vn)

                        if have_stim:
                            if t_rel < 0:
                                hist_raw[k]["baseline"].append(v)
                                if np.isfinite(vn): hist_nrm[k]["baseline"].append(vn)
                            elif 0.0 <= t_rel <= stim_dur:
                                hist_raw[k]["stim"].append(v)
                                if np.isfinite(vn): hist_nrm[k]["stim"].append(vn)
                            else:
                                hist_raw[k]["post"].append(v)
                                if np.isfinite(vn): hist_nrm[k]["post"].append(vn)

            return hist_raw, hist_nrm

        def _plot_grid(fig, axes, Y_by_key, hist_vals, title_suffix):
            """Render one page (grid) given series & histogram data dicts."""
            nrows = max(1, len(keys_for_csv))
            for r, k in enumerate(keys_for_csv):
                label = label_by_key.get(k, k)

                # --- col 1: all sweeps overlaid ---
                ax1 = axes[r, 0]
                Y = Y_by_key.get(k, None)
                if Y is not None and Y.shape[0] == M:
                    for s in range(n_sweeps):
                        ax1.plot(t_ds_csv, Y[:, s], lw=0.8, alpha=0.45)
                if have_stim:
                    ax1.axvspan(0.0, stim_dur, color="#4c78a8", alpha=0.12)
                    ax1.axvline(0.0, color="#4c78a8", lw=1.0, alpha=0.6)
                    ax1.axvline(stim_dur, color="#4c78a8", lw=1.0, alpha=0.6, ls="--")
                ax1.set_title(f"{label} — all sweeps{title_suffix}")
                if r == nrows - 1:
                    ax1.set_xlabel("Time (s, rel. stim onset)")

                # --- col 2: mean ± SEM ---
                ax2 = axes[r, 1]
                if Y is not None and Y.shape[0] == M:
                    with np.errstate(invalid="ignore"):
                        mean = np.nanmean(Y, axis=1)
                        n    = np.sum(np.isfinite(Y), axis=1)
                        std  = np.nanstd(Y, axis=1, ddof=1)
                        sem  = np.where(n >= 2, std / np.sqrt(n), np.nan)
                    ax2.plot(t_ds_csv, mean, lw=1.8)
                    ax2.fill_between(t_ds_csv, mean - sem, mean + sem, alpha=0.25, linewidth=0)
                if have_stim:
                    ax2.axvspan(0.0, stim_dur, color="#4c78a8", alpha=0.12)
                    ax2.axvline(0.0, color="#4c78a8", lw=1.0, alpha=0.6)
                    ax2.axvline(stim_dur, color="#4c78a8", lw=1.0, alpha=0.6, ls="--")
                ax2.set_title(f"{label} — mean ± SEM{title_suffix}")
                if r == nrows - 1:
                    ax2.set_xlabel("Time (s, rel. stim onset)")

                # --- col 3: line histograms (density) ---
                ax3 = axes[r, 2]
                # Build common bin edges across groups for this metric
                groups = []
                for nm in ("all", "baseline", "stim", "post"):
                    vals = np.asarray(hist_vals[k][nm], dtype=float)
                    if vals.size:
                        groups.append(vals)
                if len(groups):
                    combined = np.concatenate(groups)
                    edges = np.histogram_bin_edges(combined, bins="auto")
                    centers = 0.5 * (edges[:-1] + edges[1:])

                    def _plot_line(vals, lbl, style_kw):
                        vals = np.asarray(vals, dtype=float)
                        if vals.size == 0:
                            return
                        dens, _ = np.histogram(vals, bins=edges, density=True)
                        ax3.plot(centers, dens, **style_kw, label=lbl)

                    _plot_line(hist_vals[k]["all"], "All", dict(lw=1.8))
                    if have_stim:
                        _plot_line(hist_vals[k]["baseline"], "Baseline", dict(lw=1.6))
                        _plot_line(hist_vals[k]["stim"],     "Stim",     dict(lw=1.6, ls="--"))
                        _plot_line(hist_vals[k]["post"],     "Post",     dict(lw=1.6, ls=":"))

                ax3.set_title(f"{label} — distribution (density){title_suffix}")
                ax3.set_ylabel("Density")
                if len(ax3.lines):
                    ax3.legend(loc="best", fontsize=8)

            fig.tight_layout()

        # ---------- Prepare both RAW and NORMALIZED datasets ----------
        # RAW time-series already provided: y2_ds_by_key

        # NORMALIZED time-series per metric
        y2_ds_by_key_norm = {}
        for k in keys_for_csv:
            Y = y2_ds_by_key.get(k, None)
            if Y is None or not Y.size:
                y2_ds_by_key_norm[k] = None
                continue
            b = _per_sweep_baseline_time(Y)
            y2_ds_by_key_norm[k] = _normalize_matrix(Y, b)

        # RAW + NORMALIZED histogram values
        hist_vals_raw, hist_vals_norm = _build_hist_vals_raw_and_norm()

        # ---------- Create two-page PDF ----------
        nrows = max(1, len(keys_for_csv))
        fig_w = 13
        fig_h = max(4.0, 2.8 * nrows)

        with PdfPages(out_path) as pdf:
            # Page 1 — RAW
            fig1, axes1 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
            plt.subplots_adjust(hspace=0.6, wspace=0.25)
            _plot_grid(fig1, axes1, y2_ds_by_key, hist_vals_raw, title_suffix="")
            fig1.suptitle("PlethApp summary — raw", y=0.995, fontsize=12)
            pdf.savefig(fig1, dpi=150)
            plt.close(fig1)

            # Page 2 — NORMALIZED
            fig2, axes2 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
            plt.subplots_adjust(hspace=0.6, wspace=0.25)
            _plot_grid(fig2, axes2, y2_ds_by_key_norm, hist_vals_norm, title_suffix=" (norm)")
            fig2.suptitle("PlethApp summary — normalized", y=0.995, fontsize=12)
            pdf.savefig(fig2, dpi=150)
            plt.close(fig2)


    # def _save_summary_pdf(
    #     self,
    #     base_path,
    #     t_ds_csv: np.ndarray,
    #     y2_ds_by_key: dict[str, np.ndarray],
    #     keys_for_csv: list[str],
    #     stim_spans_by_sweep: dict[int, list[tuple[float, float]]],
    #     # NEW: full-res processed pleth & peaks for CTA row
    #     Y_proc_full_by_sweep: list[np.ndarray],
    #     peaks_by_sweep: list[np.ndarray],
    #     sr_hz: float,
    #     csv_time_zero: float | None = None,
    #     hist_bins: int = 50,
    #     cta_half_width_s: float = 0.6,  # time window around peak for CTA panels
    # ):
    #     """
    #     Builds a summary PDF with:
    #     Row 0 (NEW): cycle-triggered overlays centered on inspiratory peak
    #                 [baseline | stim | post], each with all breaths (faint) + mean±SEM.
    #     Rows 1..:   for each metric key:
    #                 - Col 1: per-sweep time traces
    #                 - Col 2: mean ± SEM across sweeps
    #                 - Col 3: hist curves (All, Baseline, Stim, Post)

    #     Notes
    #     -----
    #     - CTA uses the *full-resolution* processed pleth per sweep and the peak indices.
    #     - Baseline/Stim/Post membership for a breath is determined by the time of the peak
    #     (pk index looked up in the sweep’s masks from _get_stim_masks).
    #     """
    #     import matplotlib.pyplot as plt
    #     from matplotlib.backends.backend_pdf import PdfPages

    #     # ---------- helpers ----------
    #     def _finite_mean_sem(X: np.ndarray, axis: int = 0):
    #         with np.errstate(invalid="ignore"):
    #             m = np.nanmean(X, axis=axis)
    #             n = np.sum(np.isfinite(X), axis=axis)
    #             s = np.nanstd(X, axis=axis, ddof=1)
    #             sem = np.where(n >= 2, s / np.sqrt(n), np.nan)
    #         return m, sem

    #     def _stack_cta_segments():
    #         """
    #         Collect segments around each peak: [-W..+W] seconds, all sweeps.
    #         Split into baseline/stim/post according to pk time mask in that sweep.
    #         Return (tau, segs_baseline, segs_stim, segs_post) where each segs_* is [ [y_seg], ... ]
    #         and all segments have equal length (2*half_n+1).
    #         """
    #         half_n = max(1, int(round(cta_half_width_s * float(sr_hz)))) if sr_hz else 25
    #         tau = (np.arange(-half_n, half_n + 1, dtype=int) / float(sr_hz)) if sr_hz else np.arange(-half_n, half_n + 1)

    #         segs_B, segs_S, segs_P = [], [], []
    #         vmin, vmax = np.inf, -np.inf

    #         N = len(self.state.t)  # length of each sweep’s time base
    #         for s, y in enumerate(Y_proc_full_by_sweep):
    #             if y is None or len(y) != N:
    #                 continue
    #             pks = np.asarray(peaks_by_sweep[s], dtype=int) if s < len(peaks_by_sweep) else np.array([], dtype=int)
    #             if pks.size == 0:
    #                 continue

    #             # masks over the common time base for this sweep
    #             B, S, P = self._get_stim_masks(s)  # booleans length N

    #             for pk in pks:
    #                 i0 = int(pk) - half_n
    #                 i1 = int(pk) + half_n
    #                 if i0 < 0 or i1 >= N:
    #                     continue
    #                 seg = y[i0:i1 + 1]
    #                 if len(seg) != (2 * half_n + 1):
    #                     continue

    #                 # classify by pk’s region
    #                 if B[pk]:
    #                     segs_B.append(seg)
    #                 elif S[pk]:
    #                     segs_S.append(seg)
    #                 elif P[pk]:
    #                     segs_P.append(seg)

    #                 # y-lims agg
    #                 vmin = min(vmin, np.nanmin(seg))
    #                 vmax = max(vmax, np.nanmax(seg))

    #         # if nothing collected, set sane y-lims
    #         if not (segs_B or segs_S or segs_P):
    #             vmin, vmax = -1.0, 1.0

    #         # soft padding for y-lims
    #         pad = 0.05 * max(1e-6, (vmax - vmin))
    #         return tau, segs_B, segs_S, segs_P, (vmin - pad, vmax + pad)



    #     def _plot_cta_panel(ax, tau, segs, title: str, ylim):
    #         ax.set_title(title)
    #         if not segs:
    #             ax.text(0.5, 0.5, "no breaths", ha="center", va="center", transform=ax.transAxes)
    #             ax.axvline(0, ls="--", lw=0.8)
    #             ax.set_xlim(tau[0], tau[-1])
    #             ax.set_ylim(*ylim)
    #             return
    #         A = np.vstack(segs)  # (n_breaths, T)
    #         # faint individual
    #         ax.plot(tau, A.T, alpha=0.15, linewidth=0.7)
    #         # mean ± sem
    #         m, sem = _finite_mean_sem(A, axis=0)
    #         ax.plot(tau, m, linewidth=2)
    #         ax.fill_between(tau, m - sem, m + sem, alpha=0.2, linewidth=0)
    #         ax.axvline(0, ls="--", lw=0.8)
    #         ax.set_xlim(tau[0], tau[-1])
    #         ax.set_ylim(*ylim)

    #     def _hist_curves(ax, arrays, labels):
    #         """
    #         arrays: list of 1D arrays (finite only will be used)
    #         labels: same length as arrays
    #         Draw as line curves (KDE-free): use common bins across all.
    #         """
    #         clean = [a[np.isfinite(a)] for a in arrays]
    #         if not any(len(a) for a in clean):
    #             ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    #             return
    #         allv = np.concatenate([a for a in clean if len(a)])
    #         if np.allclose(np.nanmin(allv), np.nanmax(allv)):
    #             # constant -> single vertical line
    #             x = np.array([allv[0], allv[0]])
    #             y = np.array([0.0, 1.0])
    #             for i, (a, lab) in enumerate(zip(clean, labels)):
    #                 if len(a):
    #                     ax.plot(x, y, label=lab)
    #             ax.set_ylim(0, 1.05)
    #             return

    #         bins = np.linspace(np.nanmin(allv), np.nanmax(allv), hist_bins + 1)
    #         centers = 0.5 * (bins[:-1] + bins[1:])
    #         for a, lab in zip(clean, labels):
    #             if not len(a):
    #                 continue
    #             h, _ = np.histogram(a, bins=bins, density=True)
    #             ax.plot(centers, h, label=lab)
    #         ax.legend(fontsize=8, ncols=2)

    #     # ---------- figure scaffolding ----------
    #     n_metrics = len(keys_for_csv)
    #     ncols = 3
    #     nrows = n_metrics + 1  # +1 for CTA row at top

    #     fig, axes = plt.subplots(
    #         nrows=nrows, ncols=ncols,
    #         figsize=(ncols * 4.0, nrows * 2.8),
    #         sharex=False, squeeze=False
    #     )

    #     # ---------- CTA row (row 0) ----------
    #     tau, segs_B, segs_S, segs_P, ylims_cta = _stack_cta_segments()
    #     titles_cta = [
    #         "Breath-centered (Baseline)",
    #         "Breath-centered (Stim)",
    #         "Breath-centered (Post)",
    #     ]
    #     for ax, segs, ttl in zip(axes[0], [segs_B, segs_S, segs_P], titles_cta):
    #         _plot_cta_panel(ax, tau, segs, ttl, ylims_cta)
    #     axes[0, 0].set_ylabel("Pleth (a.u.)")
    #     for j in range(3):
    #         axes[0, j].set_xlabel("Time from peak (s)")

    #     # ---------- Metric rows (row 1..nrows-1) ----------
    #     # Helper: masks (by sample index along the *CSV time base*)
    #     # We’ll build baseline/stim/post masks for each sweep on the original time base,
    #     # then filter metric values by those masks’ indices.
    #     def _sweep_masks_indices(s: int):
    #         # Booleans over original time base
    #         B, S, P = self._get_stim_masks(s)
    #         return np.where(B)[0], np.where(S)[0], np.where(P)[0]

    #     # Precompute masks per sweep
    #     masks_by_sweep = [ _sweep_masks_indices(s) for s in range(len(Y_proc_full_by_sweep)) ]

    #     for r in range(1, nrows):
    #         k = keys_for_csv[r - 1]  # metric key for this row

    #         # ---- Column 0: per-sweep time traces
    #         ax0 = axes[r, 0]
    #         for s in range(y2_ds_by_key[k].shape[1]):
    #             ax0.plot(t_ds_csv, y2_ds_by_key[k][:, s], alpha=0.5, linewidth=0.9)
    #         ax0.set_title(f"{k}: sweeps")
    #         ax0.set_ylabel(k)

    #         # ---- Column 1: mean ± SEM
    #         ax1 = axes[r, 1]
    #         Y = y2_ds_by_key[k]  # (T_ds, S)
    #         m, sem = _finite_mean_sem(Y, axis=1)
    #         ax1.plot(t_ds_csv, m, linewidth=2)
    #         ax1.fill_between(t_ds_csv, m - sem, m + sem, alpha=0.25, linewidth=0)
    #         ax1.set_title(f"{k}: mean ± SEM")

    #         # ---- Column 2: histogram curves (All / Baseline / Stim / Post)
    #         ax2 = axes[r, 2]

    #         # Build per-breath-like pools by *sample-wise* selection:
    #         # concatenate values across sweeps in each region using the masks.
    #         # (This mirrors the previous version you approved.)
    #         vals_all = []
    #         vals_B   = []
    #         vals_S   = []
    #         vals_P   = []
    #         N = len(self.state.t)
    #         # Map downsample indices (t_ds_csv) back to the original index (nearest)
    #         # so we can use the same masks; for hist we can take all samples (no DS).
    #         # Simpler: just use the downsampled values directly (already fine for hist).
    #         # BUT we still want region separation; build boolean DS masks by checking
    #         # the *nearest original* index’s region.
    #         idx_map = np.clip(np.searchsorted(self.state.t, t_ds_csv), 0, N - 1)

    #         for s in range(y2_ds_by_key[k].shape[1]):
    #             B_idx, S_idx, P_idx = masks_by_sweep[s]
    #             B_mask_ds = np.isin(idx_map, B_idx)
    #             S_mask_ds = np.isin(idx_map, S_idx)
    #             P_mask_ds = np.isin(idx_map, P_idx)

    #             col = y2_ds_by_key[k][:, s]
    #             vals_all.append(col[np.isfinite(col)])
    #             vals_B.append(col[B_mask_ds & np.isfinite(col)])
    #             vals_S.append(col[S_mask_ds & np.isfinite(col)])
    #             vals_P.append(col[P_mask_ds & np.isfinite(col)])

    #         arr_all = np.concatenate([v for v in vals_all if len(v)]) if any(len(v) for v in vals_all) else np.array([])
    #         arr_B   = np.concatenate([v for v in vals_B   if len(v)]) if any(len(v) for v in vals_B)   else np.array([])
    #         arr_S   = np.concatenate([v for v in vals_S   if len(v)]) if any(len(v) for v in vals_S)   else np.array([])
    #         arr_P   = np.concatenate([v for v in vals_P   if len(v)]) if any(len(v) for v in vals_P)   else np.array([])

    #         _hist_curves(ax2, [arr_all, arr_B, arr_S, arr_P], ["All", "Baseline", "Stim", "Post"])
    #         ax2.set_title(f"{k}: distribution")
    #         ax2.set_xlabel(k)

    #     # ---------- save ----------
    #     for j in range(ncols):
    #         axes[-1, j].set_xlabel("time (s)")  # bottom row x-labels

    #     fig.tight_layout()
    #     pdf_path = base_path.with_name(base_path.name + "_summary.pdf")
    #     fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    #     plt.close(fig)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
