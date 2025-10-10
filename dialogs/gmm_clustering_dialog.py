"""
GMM Clustering Dialog for PlethApp.

This dialog provides an interface for Gaussian Mixture Model (GMM) clustering
of breath patterns to identify eupnea and sniffing behaviors.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QScrollArea, QWidget, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core import metrics, filters


class GMMClusteringDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton, QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QSizePolicy
        import numpy as np

        super().__init__(parent)
        self.setWindowTitle("Eupnea/Sniffing Detection")
        self.resize(1200, 800)

        self.main_window = main_window
        self.cluster_labels = None  # Will store cluster assignments
        self.gmm_model = None
        self.feature_data = None  # Per-breath feature matrix
        self.breath_cycles = []  # List of (sweep_idx, breath_idx) tuples
        self.sniffing_cluster_id = None  # Which cluster represents sniffing
        self.cluster_colors = {}  # Map cluster_id -> color (purple for sniffing, green for eupnea)
        self.had_quality_warning = False  # Track if clustering quality was questionable

        # Main layout
        main_layout = QHBoxLayout(self)

        # Left panel: Controls
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)

        # Help/Info button at top
        help_layout = QHBoxLayout()
        help_btn = QPushButton("ℹ️ What is GMM Clustering?")
        help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        help_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: none;
                background: transparent;
                text-decoration: underline;
                color: #4A9EFF;
                font-size: 10pt;
            }
            QPushButton:hover {
                color: #6BB6FF;
            }
        """)
        help_btn.clicked.connect(self.show_help)
        help_layout.addWidget(help_btn)
        help_layout.addStretch()
        left_panel.addLayout(help_layout)

        # Feature selection group
        feature_group = QGroupBox("Select Features for Clustering")
        feature_layout = QVBoxLayout()

        self.feature_checkboxes = {}
        available_features = ["if", "ti", "te", "amp_insp", "amp_exp", "area_insp", "area_exp", "max_dinsp", "max_dexp"]
        default_features = ["if", "ti", "te", "amp_insp", "amp_exp", "max_dinsp", "max_dexp"]  # Good defaults for eupnea/sniffing separation

        for feature in available_features:
            cb = QCheckBox(feature)
            cb.setChecked(feature in default_features)
            self.feature_checkboxes[feature] = cb
            feature_layout.addWidget(cb)

        feature_group.setLayout(feature_layout)
        left_panel.addWidget(feature_group)

        # Number of clusters
        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(QLabel("Number of Clusters:"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 10)
        self.n_clusters_spin.setValue(2)  # Start with 2 for eupnea/sniffing
        cluster_layout.addWidget(self.n_clusters_spin)
        cluster_layout.addStretch()
        left_panel.addLayout(cluster_layout)

        # Run GMM button
        self.run_gmm_btn = QPushButton("Run GMM Clustering")
        self.run_gmm_btn.clicked.connect(self.on_run_gmm)
        self.run_gmm_btn.setMinimumHeight(40)
        left_panel.addWidget(self.run_gmm_btn)

        # Results table
        results_label = QLabel("Cluster Statistics:")
        results_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        left_panel.addWidget(results_label)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Cluster", "Count", "Percentage", "Avg Confidence"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        left_panel.addWidget(self.results_table)

        # Status label
        self.status_label = QLabel("Select features and click 'Run GMM Clustering' to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        left_panel.addWidget(self.status_label)

        # Info label about what this dialog does
        info_label = QLabel("Note: Sniffing breaths will be marked with purple background, eupnea with green overlay.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 9pt; font-style: italic;")
        left_panel.addWidget(info_label)

        # Waveform plotting controls
        waveform_group = QGroupBox("Waveform Visualization")
        waveform_layout = QVBoxLayout()

        self.plot_waveforms_cb = QCheckBox("Plot Mean Waveforms")
        self.plot_waveforms_cb.setChecked(True)  # Enabled by default
        self.plot_waveforms_cb.setToolTip("Enable to plot mean ± SEM waveforms (can be slow with many breaths)")
        waveform_layout.addWidget(self.plot_waveforms_cb)

        n_breaths_layout = QHBoxLayout()
        n_breaths_layout.addWidget(QLabel("Max breaths per group:"))
        self.n_breaths_spin = QSpinBox()
        self.n_breaths_spin.setRange(10, 500)
        self.n_breaths_spin.setValue(25)
        self.n_breaths_spin.setToolTip("Number of breaths to include from each group (eupnea/sniffing)")
        n_breaths_layout.addWidget(self.n_breaths_spin)
        n_breaths_layout.addStretch()
        waveform_layout.addLayout(n_breaths_layout)

        waveform_group.setLayout(waveform_layout)
        left_panel.addWidget(waveform_group)

        # Eupnea detection mode selection (moved to bottom)
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup, QDoubleSpinBox
        mode_group = QGroupBox("Eupnea Detection Method")
        mode_layout = QVBoxLayout()

        self.detection_mode_button_group = QButtonGroup(self)
        self.gmm_mode_radio = QRadioButton("GMM-Based (Automatic)")
        self.gmm_mode_radio.setToolTip("Use GMM clustering to identify eupnea automatically.\n"
                                       "Breaths NOT classified as sniffing are marked as eupnea.")
        self.freq_mode_radio = QRadioButton("Frequency-Based (Manual Threshold)")
        self.freq_mode_radio.setToolTip("Use manual frequency threshold to identify eupnea.\n"
                                        "Breaths below frequency threshold are marked as eupnea.")

        self.detection_mode_button_group.addButton(self.gmm_mode_radio, 0)
        self.detection_mode_button_group.addButton(self.freq_mode_radio, 1)

        # Set initial selection based on main window's current mode
        if hasattr(main_window, 'eupnea_detection_mode') and main_window.eupnea_detection_mode == "frequency":
            self.freq_mode_radio.setChecked(True)
        else:
            self.gmm_mode_radio.setChecked(True)

        mode_layout.addWidget(self.gmm_mode_radio)
        gmm_info = QLabel("  → Automatic classification from clustering")
        gmm_info.setStyleSheet("color: #888; font-size: 9pt; margin-left: 20px;")
        mode_layout.addWidget(gmm_info)

        mode_layout.addWidget(self.freq_mode_radio)

        # Frequency-based parameters (enabled/disabled based on mode selection)
        freq_params_layout = QVBoxLayout()
        freq_params_layout.setContentsMargins(30, 5, 0, 0)  # Indent for visual grouping

        freq_thresh_layout = QHBoxLayout()
        freq_thresh_layout.addWidget(QLabel("Frequency Threshold (Hz):"))
        self.freq_threshold_spin = QDoubleSpinBox()
        self.freq_threshold_spin.setRange(0.1, 20.0)
        self.freq_threshold_spin.setValue(getattr(main_window, 'eupnea_freq_threshold', 5.0))
        self.freq_threshold_spin.setDecimals(1)
        self.freq_threshold_spin.setSingleStep(0.5)
        self.freq_threshold_spin.setToolTip("Breathing below this frequency is considered eupneic")
        freq_thresh_layout.addWidget(self.freq_threshold_spin)
        freq_thresh_layout.addStretch()
        freq_params_layout.addLayout(freq_thresh_layout)

        min_dur_layout = QHBoxLayout()
        min_dur_layout.addWidget(QLabel("Minimum Duration (s):"))
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.5, 10.0)
        self.min_duration_spin.setValue(main_window.eupnea_min_duration if hasattr(main_window, 'eupnea_min_duration') else 2.0)
        self.min_duration_spin.setDecimals(1)
        self.min_duration_spin.setSingleStep(0.5)
        self.min_duration_spin.setToolTip("Region must sustain criteria for at least this long")
        min_dur_layout.addWidget(self.min_duration_spin)
        min_dur_layout.addStretch()
        freq_params_layout.addLayout(min_dur_layout)

        mode_layout.addLayout(freq_params_layout)

        mode_group.setLayout(mode_layout)
        left_panel.addWidget(mode_group)

        # Connect mode radio buttons to auto-apply changes
        self.gmm_mode_radio.toggled.connect(self._on_detection_mode_changed)
        self.freq_mode_radio.toggled.connect(self._on_detection_mode_changed)

        # Connect frequency parameters to auto-apply changes
        self.freq_threshold_spin.valueChanged.connect(self._on_frequency_params_changed)
        self.min_duration_spin.valueChanged.connect(self._on_frequency_params_changed)

        # Close button
        button_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(35)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        left_panel.addLayout(button_layout)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # Right panel: Scrollable Visualizations
        from PyQt6.QtWidgets import QScrollArea, QWidget
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # No horizontal scroll
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Enable mouse wheel scrolling

        # Container widget for plots
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure for visualizations (will contain both histograms and scatter plots)
        # Fit width to window, tall for scrolling
        self.figure = Figure(figsize=(9, 25))  # Narrower to fit width, tall for vertical scroll
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(2000)  # Tall enough to trigger vertical scrolling

        # Install event filter to forward mouse wheel events to scroll area
        self.canvas.installEventFilter(self)

        plot_layout.addWidget(self.canvas)

        self.scroll_area.setWidget(plot_container)
        main_layout.addWidget(self.scroll_area, 2)

        # Auto-run GMM visualization if results already exist (from automatic clustering)
        # This happens after peak detection, so the dialog can immediately show results
        if hasattr(main_window.state, 'gmm_sniff_probabilities') and main_window.state.gmm_sniff_probabilities:
            print("[gmm-dialog] Found existing GMM results, auto-loading visualization...")
            self._load_existing_gmm_results()

    def _load_existing_gmm_results(self):
        """Load and display existing GMM results that were computed automatically."""
        import numpy as np

        try:
            # Check if main window has cached GMM results
            if (hasattr(self.main_window, '_cached_gmm_results') and
                self.main_window._cached_gmm_results is not None):

                print("[gmm-dialog] Loading cached GMM results (fast path)...")

                # Unpack cached results
                cached = self.main_window._cached_gmm_results
                self.cluster_labels = cached['cluster_labels']
                self.cluster_probabilities = cached['cluster_probabilities']
                self.feature_data = cached['feature_matrix']
                self.breath_cycles = cached['breath_cycles']
                self.sniffing_cluster_id = cached['sniffing_cluster_id']
                selected_features = cached['feature_keys']

                # Assign colors based on cached sniffing cluster
                for cluster_id in np.unique(self.cluster_labels):
                    if cluster_id == self.sniffing_cluster_id:
                        self.cluster_colors[cluster_id] = 'purple'
                    else:
                        self.cluster_colors[cluster_id] = 'green'

                # Update results table
                self._update_results_table()

                # Plot results (only plotting, no re-computation)
                self._plot_clusters(self.feature_data, self.cluster_labels, selected_features)

                self.status_label.setText(f"✓ Loaded cached results: {len(self.feature_data)} breaths classified into {len(np.unique(self.cluster_labels))} clusters.")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")

            else:
                print("[gmm-dialog] No cached results found, user needs to run GMM manually")
                self.status_label.setText("No existing results. Click 'Run GMM Clustering' to analyze breaths.")
                self.status_label.setStyleSheet("color: #888; font-style: italic;")

            # Scroll to top
            self.scroll_area.verticalScrollBar().setValue(0)

        except Exception as e:
            print(f"[gmm-dialog] Error loading existing results: {e}")
            import traceback
            traceback.print_exc()

    def eventFilter(self, obj, event):
        """Forward wheel events from canvas to scroll area."""
        from PyQt6.QtCore import QEvent
        if obj == self.canvas and event.type() == QEvent.Type.Wheel:
            # Forward wheel event to scroll area
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - event.angleDelta().y() // 2
            )
            return True
        return super().eventFilter(obj, event)

    def _on_detection_mode_changed(self):
        """Enable/disable frequency parameters and auto-apply detection mode change."""
        gmm_mode = self.gmm_mode_radio.isChecked()
        self.freq_threshold_spin.setEnabled(not gmm_mode)
        self.min_duration_spin.setEnabled(not gmm_mode)

        # Auto-apply detection mode change to main window
        selected_mode = "gmm" if gmm_mode else "frequency"
        if self.main_window.eupnea_detection_mode != selected_mode:
            self.main_window.eupnea_detection_mode = selected_mode
            print(f"[gmm-dialog] Auto-applied eupnea detection mode: {selected_mode}")

            # Update main window's eupnea regions and redraw
            self.main_window.redraw_main_plot()

    def _on_frequency_params_changed(self):
        """Auto-apply frequency parameter changes to main window."""
        # Only apply if in frequency mode
        if self.freq_mode_radio.isChecked():
            # Update main window parameters
            self.main_window.eupnea_freq_threshold = self.freq_threshold_spin.value()
            self.main_window.eupnea_min_duration = self.min_duration_spin.value()

            print(f"[gmm-dialog] Auto-applied frequency params: thresh={self.main_window.eupnea_freq_threshold} Hz, "
                  f"min_dur={self.main_window.eupnea_min_duration} s")

            # Redraw to apply new parameters
            self.main_window.redraw_main_plot()

    def on_run_gmm(self):
        """Run GMM clustering on breath metrics."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Get selected features
        selected_features = [f for f, cb in self.feature_checkboxes.items() if cb.isChecked()]
        if len(selected_features) < 2:
            self.status_label.setText("Error: Please select at least 2 features.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return

        # Collect breath metrics from all analyzed sweeps
        self.status_label.setText("Collecting breath metrics...")
        self.status_label.setStyleSheet("color: blue;")

        try:
            feature_matrix, breath_cycles = self._collect_breath_features(selected_features)

            if len(feature_matrix) < self.n_clusters_spin.value():
                self.status_label.setText(f"Error: Not enough breaths ({len(feature_matrix)}) for {self.n_clusters_spin.value()} clusters.")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return

            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Fit GMM
            n_clusters = self.n_clusters_spin.value()
            self.status_label.setText(f"Fitting GMM with {n_clusters} clusters...")

            self.gmm_model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
            self.cluster_labels = self.gmm_model.fit_predict(feature_matrix_scaled)
            self.cluster_probabilities = self.gmm_model.predict_proba(feature_matrix_scaled)
            self.feature_data = feature_matrix
            self.breath_cycles = breath_cycles

            # Check clustering quality
            quality_warning = self._check_clustering_quality(feature_matrix_scaled, selected_features)

            # Identify sniffing cluster and assign colors
            self._identify_sniffing_cluster(feature_matrix, selected_features, quality_warning)

            # Update results table
            self._update_results_table()

            # Plot results (histograms + scatter plots)
            self._plot_clusters(feature_matrix, self.cluster_labels, selected_features)

            self.status_label.setText(f"✓ Clustering complete: {len(feature_matrix)} breaths classified into {n_clusters} clusters.")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            # Auto-apply results to main plot
            self._auto_apply_gmm_results()

            # Scroll to top after plotting
            self.scroll_area.verticalScrollBar().setValue(0)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            import traceback
            traceback.print_exc()

    def _check_clustering_quality(self, feature_matrix_scaled, feature_keys):
        """Check if clustering is meaningful (not just splitting homogeneous data).

        Returns warning message if clustering quality is poor, None otherwise.
        """
        import numpy as np
        from sklearn.metrics import silhouette_score

        # Silhouette score measures how well-separated clusters are
        # Range: -1 (poor) to +1 (excellent), ~0 means overlapping clusters
        if len(np.unique(self.cluster_labels)) > 1:
            silhouette = silhouette_score(feature_matrix_scaled, self.cluster_labels)
        else:
            silhouette = -1

        print(f"[gmm-clustering] Silhouette score: {silhouette:.3f}")

        # Warn if clusters are poorly separated
        if silhouette < 0.25:
            return (
                f"⚠️ Low cluster separation (silhouette={silhouette:.3f}).\n\n"
                "This suggests the breathing patterns are very similar.\n"
                "The identified 'sniffing' cluster may just be normal variation,\n"
                "not distinct sniffing behavior.\n\n"
                "Consider: (1) using fewer clusters, or (2) this data may not\n"
                "contain distinct sniffing bouts."
            )

        return None

    def _identify_sniffing_cluster(self, feature_matrix, feature_keys, quality_warning=None):
        """Identify which cluster represents sniffing and assign colors.

        Sniffing characteristics:
        - Highest mean instantaneous frequency (if)
        - Lowest mean inspiratory time (ti)

        Args:
            quality_warning: Warning message from quality check (if any)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels)

        # Get indices of IF and Ti features
        if_idx = feature_keys.index('if') if 'if' in feature_keys else None
        ti_idx = feature_keys.index('ti') if 'ti' in feature_keys else None

        if if_idx is None and ti_idx is None:
            # Can't identify sniffing without IF or Ti, use default colors
            print("[gmm-clustering] Warning: Cannot identify sniffing cluster without 'if' or 'ti' features. Using default colors.")
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            self.cluster_colors = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_labels)}
            self.sniffing_cluster_id = None
            return

        # Compute mean IF and Ti for each cluster
        cluster_stats = {}
        for cluster_id in unique_labels:
            mask = self.cluster_labels == cluster_id
            stats = {}

            if if_idx is not None:
                stats['mean_if'] = np.mean(feature_matrix[mask, if_idx])
            if ti_idx is not None:
                stats['mean_ti'] = np.mean(feature_matrix[mask, ti_idx])

            cluster_stats[cluster_id] = stats

        # Identify sniffing: highest IF and/or lowest Ti
        # Score = (normalized IF rank) + (normalized Ti inverse rank)
        cluster_scores = {}
        for cluster_id in unique_labels:
            score = 0

            if if_idx is not None:
                # Higher IF = more likely sniffing
                if_vals = [cluster_stats[c]['mean_if'] for c in unique_labels]
                if_rank = sorted(if_vals).index(cluster_stats[cluster_id]['mean_if'])
                score += if_rank / (n_clusters - 1) if n_clusters > 1 else 0

            if ti_idx is not None:
                # Lower Ti = more likely sniffing
                ti_vals = [cluster_stats[c]['mean_ti'] for c in unique_labels]
                ti_rank = sorted(ti_vals, reverse=True).index(cluster_stats[cluster_id]['mean_ti'])
                score += ti_rank / (n_clusters - 1) if n_clusters > 1 else 0

            cluster_scores[cluster_id] = score

        # Cluster with highest score is sniffing
        self.sniffing_cluster_id = max(cluster_scores, key=cluster_scores.get)

        # Validate that the identified cluster actually looks like sniffing
        # Sniffing typically has IF > 5-6 Hz, eupnea is usually 2-4 Hz
        sniff_stats = cluster_stats[self.sniffing_cluster_id]
        if if_idx is not None:
            sniff_if = sniff_stats['mean_if']
            # Check if "sniffing" cluster actually has high enough frequency
            if sniff_if < 5.0:
                # This doesn't look like real sniffing
                physiological_warning = (
                    f"⚠️ The identified 'sniffing' cluster has low mean IF ({sniff_if:.2f} Hz).\n\n"
                    "True sniffing typically has IF > 5-6 Hz.\n"
                    "This may just be variation in normal breathing, not distinct sniffing.\n\n"
                    "The cluster separation may not represent meaningful patterns."
                )
                if quality_warning:
                    quality_warning = quality_warning + "\n\n" + physiological_warning
                else:
                    quality_warning = physiological_warning

        # Assign colors: purple for sniffing, green for others (if 2 clusters), otherwise tab10
        if n_clusters == 2:
            # Simple case: sniffing = purple, other = green
            for cluster_id in unique_labels:
                if cluster_id == self.sniffing_cluster_id:
                    self.cluster_colors[cluster_id] = 'purple'
                else:
                    self.cluster_colors[cluster_id] = 'green'
        else:
            # Multiple clusters: sniffing = purple, others = tab10 colors (avoiding purple)
            tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
            color_idx = 0
            for cluster_id in unique_labels:
                if cluster_id == self.sniffing_cluster_id:
                    self.cluster_colors[cluster_id] = 'purple'
                else:
                    # Use green as first alternative color, then other tab10 colors
                    if color_idx == 0:
                        self.cluster_colors[cluster_id] = 'green'
                    else:
                        # Skip purple-ish colors in tab10 (index 4)
                        skip_indices = [4]
                        actual_idx = color_idx
                        while actual_idx in skip_indices and actual_idx < 10:
                            actual_idx += 1
                        self.cluster_colors[cluster_id] = tab10_colors[actual_idx % 10]
                    color_idx += 1

        print(f"[gmm-clustering] Identified cluster {self.sniffing_cluster_id} as sniffing")
        for cluster_id in unique_labels:
            stats_str = ", ".join([f"{k}={v:.3f}" for k, v in cluster_stats[cluster_id].items()])
            print(f"  Cluster {cluster_id}: {stats_str}, color={self.cluster_colors[cluster_id]}")

        # Show warning if clustering quality is questionable
        if quality_warning:
            self.had_quality_warning = True
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Clustering Quality Warning")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(quality_warning)
            msg.setInformativeText("You can still explore the results, but be cautious about applying them to the plot.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        else:
            self.had_quality_warning = False

    def _collect_breath_features(self, feature_keys):
        """Collect per-breath features from all analyzed sweeps."""
        import numpy as np
        from core import metrics, filters

        feature_matrix = []
        breath_cycles = []

        # Iterate through all analyzed sweeps
        st = self.main_window.state

        for sweep_idx in sorted(st.breath_by_sweep.keys()):
            breath_data = st.breath_by_sweep[sweep_idx]

            # Get peaks from peaks_by_sweep (they're stored separately)
            if sweep_idx not in st.peaks_by_sweep:
                continue

            peaks = st.peaks_by_sweep[sweep_idx]

            # Get time, signal, and event indices for this sweep
            t = st.t
            y_raw = st.sweeps[st.analyze_chan][:, sweep_idx]

            # Apply filters to get processed signal (replicate _current_trace logic)
            y = filters.apply_all_1d(
                y_raw, st.sr_hz,
                st.use_low, st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert,
                order=self.main_window.filter_order
            )

            # Apply notch filter if configured
            if self.main_window.notch_filter_lower is not None and self.main_window.notch_filter_upper is not None:
                y = self.main_window._apply_notch_filter(y, st.sr_hz,
                                                          self.main_window.notch_filter_lower,
                                                          self.main_window.notch_filter_upper)

            # Apply z-score normalization if enabled (using global statistics)
            if self.main_window.use_zscore_normalization:
                # Compute global stats if not cached
                if self.main_window.zscore_global_mean is None or self.main_window.zscore_global_std is None:
                    self.main_window.zscore_global_mean, self.main_window.zscore_global_std = self.main_window._compute_global_zscore_stats()
                y = filters.zscore_normalize(y, self.main_window.zscore_global_mean, self.main_window.zscore_global_std)

            # Get breath event indices
            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            expmins = breath_data.get('expmins', np.array([]))
            expoffs = breath_data.get('expoffs', np.array([]))

            # Skip if no breath events detected
            if len(onsets) == 0:
                continue

            # Compute metrics for this sweep
            metrics_dict = {}
            for feature_key in feature_keys:
                if feature_key in metrics.METRICS:
                    metric_arr = metrics.METRICS[feature_key](
                        t, y, st.sr_hz, peaks, onsets, offsets, expmins, expoffs
                    )
                    metrics_dict[feature_key] = metric_arr

            # Extract per-breath values (one value per breath cycle)
            n_breaths = len(onsets)
            for breath_idx in range(n_breaths):
                start = int(onsets[breath_idx])
                breath_features = []

                # Extract metric value for this breath
                valid_breath = True
                for feature_key in feature_keys:
                    if feature_key not in metrics_dict:
                        valid_breath = False
                        break

                    metric_arr = metrics_dict[feature_key]
                    if start < len(metric_arr):
                        val = metric_arr[start]
                        if np.isnan(val) or not np.isfinite(val):
                            valid_breath = False
                            break
                        breath_features.append(val)
                    else:
                        valid_breath = False
                        break

                if valid_breath and len(breath_features) == len(feature_keys):
                    feature_matrix.append(breath_features)
                    breath_cycles.append((sweep_idx, breath_idx))

        return np.array(feature_matrix), breath_cycles

    def _update_results_table(self):
        """Update the results table with cluster statistics."""
        import numpy as np
        from PyQt6.QtWidgets import QTableWidgetItem
        from PyQt6.QtGui import QColor

        unique_clusters = np.unique(self.cluster_labels)
        self.results_table.setRowCount(len(unique_clusters))

        for i, cluster_id in enumerate(unique_clusters):
            count = np.sum(self.cluster_labels == cluster_id)
            percentage = 100.0 * count / len(self.cluster_labels)

            # Calculate average confidence (probability of assigned cluster)
            mask = self.cluster_labels == cluster_id
            if hasattr(self, 'cluster_probabilities') and self.cluster_probabilities is not None:
                avg_confidence = np.mean(self.cluster_probabilities[mask, cluster_id])
            else:
                avg_confidence = 1.0  # Perfect confidence if no probabilities

            # Label with pattern type if identified
            if cluster_id == self.sniffing_cluster_id:
                label = f"Cluster {cluster_id} (Sniffing)"
            else:
                label = f"Cluster {cluster_id}"

            cluster_item = QTableWidgetItem(label)
            count_item = QTableWidgetItem(str(count))
            pct_item = QTableWidgetItem(f"{percentage:.1f}%")
            conf_item = QTableWidgetItem(f"{avg_confidence:.3f}")

            # Color-code the row
            if cluster_id in self.cluster_colors:
                color = self.cluster_colors[cluster_id]
                if isinstance(color, str):
                    # Named color
                    bg_color = QColor(color)
                else:
                    # RGBA tuple from colormap
                    bg_color = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255))

                bg_color.setAlpha(80)  # Make it semi-transparent
                cluster_item.setBackground(bg_color)
                count_item.setBackground(bg_color)
                pct_item.setBackground(bg_color)
                conf_item.setBackground(bg_color)

            self.results_table.setItem(i, 0, cluster_item)
            self.results_table.setItem(i, 1, count_item)
            self.results_table.setItem(i, 2, pct_item)
            self.results_table.setItem(i, 3, conf_item)

    def _plot_waveform_overlays(self, total_rows, max_cols, labels, colors):
        """Plot waveform overlays: Column 1 = mean +/- SEM waveforms, Column 2 = mean trajectory line plot."""
        import numpy as np
        from core import filters

        if not self.main_window or not hasattr(self.main_window, 'state'):
            return

        st = self.main_window.state

        # Get current processed signal (with all filters applied)
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return

        # Check if waveform plotting is enabled
        plot_waveforms = self.plot_waveforms_cb.isChecked()
        max_breaths = self.n_breaths_spin.value()

        # Collect waveform cutouts for each cluster (now restricted to onset -> expoff)
        eupnea_waveforms = []  # List of (time_array, signal_array) tuples
        sniffing_waveforms = []

        # Also collect signal and derivative for mean trajectory plot
        eupnea_trajectories = []  # List of (signal_array, derivative_array) tuples
        sniffing_trajectories = []

        # Only collect data if plotting is enabled
        if plot_waveforms:
            for breath_idx, (sweep_idx, breath_i) in enumerate(self.breath_cycles):
                # Early exit if both groups have enough breaths
                if len(eupnea_waveforms) >= max_breaths and len(sniffing_waveforms) >= max_breaths:
                    break

                cluster_id = labels[breath_idx]

                # Skip if this cluster already has enough breaths
                if cluster_id == self.sniffing_cluster_id:
                    if len(sniffing_waveforms) >= max_breaths:
                        continue
                else:
                    if len(eupnea_waveforms) >= max_breaths:
                        continue

                # Get breath event data for this sweep
                breath_data = st.breath_by_sweep.get(sweep_idx)
                if breath_data is None:
                    continue

                onsets = breath_data.get('onsets', None)
                expoffs = breath_data.get('expoffs', None)
                if onsets is None or expoffs is None or len(onsets) == 0 or len(expoffs) == 0:
                    continue

                if breath_i >= len(onsets) or breath_i >= len(expoffs):
                    continue

                # Get processed signal for this sweep
                Y = st.sweeps[st.analyze_chan]
                y_raw = Y[:, sweep_idx]

                # Apply all filters (same as main plot)
                y = filters.apply_all_1d(
                    y_raw,
                    sr_hz=st.sr_hz,
                    use_low=st.use_low,
                    low_hz=st.low_hz,
                    use_high=st.use_high,
                    high_hz=st.high_hz,
                    use_mean=st.use_mean_sub,
                    mean_param=st.mean_val,
                    use_inv=st.use_invert,
                    order=self.main_window.filter_order
                )

                # Apply notch filter if configured
                if self.main_window.notch_filter_lower is not None and self.main_window.notch_filter_upper is not None:
                    y = self.main_window._apply_notch_filter(y, st.sr_hz)

                # Apply z-score if enabled
                if self.main_window.use_zscore_normalization:
                    if self.main_window.zscore_global_mean is None or self.main_window.zscore_global_std is None:
                        self.main_window.zscore_global_mean, self.main_window.zscore_global_std = self.main_window._compute_global_zscore_stats()
                    y = filters.zscore_normalize(y, self.main_window.zscore_global_mean, self.main_window.zscore_global_std)

                t = st.t

                # Extract waveform from inspiratory onset to expiratory offset
                start_idx = int(onsets[breath_i])
                end_idx = int(expoffs[breath_i])

                # Validate indices
                if start_idx < 0 or end_idx >= len(y) or start_idx >= end_idx:
                    continue

                # Extract waveform segment (onset -> expoff)
                waveform_t = t[start_idx:end_idx+1] - t[start_idx]  # Time relative to onset
                waveform_y = y[start_idx:end_idx+1]

                # Compute first derivative for this segment
                dt = 1.0 / st.sr_hz
                waveform_dy = np.gradient(waveform_y, dt)

                # Add to appropriate cluster (limit to max_breaths)
                if cluster_id == self.sniffing_cluster_id:
                    if len(sniffing_waveforms) < max_breaths:
                        sniffing_waveforms.append((waveform_t, waveform_y))
                        sniffing_trajectories.append((waveform_y, waveform_dy))
                else:
                    if len(eupnea_waveforms) < max_breaths:
                        eupnea_waveforms.append((waveform_t, waveform_y))
                        eupnea_trajectories.append((waveform_y, waveform_dy))

        # ========================================
        # Column 1: Mean +/- SEM Waveforms (optional)
        # ========================================
        if plot_waveforms:
            ax1 = self.figure.add_subplot(total_rows, max_cols, 1)

            # Helper function to pad waveforms to longest duration
            def align_and_compute_stats(waveforms):
                """Pad waveforms to longest duration, then compute mean +/- SEM."""
                if len(waveforms) == 0:
                    return None, None, None

                # Find longest duration
                max_len = max(len(wf_t) for wf_t, wf_y in waveforms)

                # Pad all waveforms to max_len with NaN
                padded_signals = []
                for wf_t, wf_y in waveforms:
                    if len(wf_y) < max_len:
                        # Pad with NaN
                        padded = np.full(max_len, np.nan)
                        padded[:len(wf_y)] = wf_y
                        padded_signals.append(padded)
                    else:
                        padded_signals.append(wf_y)

                # Stack and compute statistics (ignoring NaN)
                wf_matrix = np.vstack(padded_signals)
                mean_wf = np.nanmean(wf_matrix, axis=0)
                sem_wf = np.nanstd(wf_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(wf_matrix), axis=0))

                # Create time axis matching the longest breath
                dt = 1.0 / st.sr_hz
                t_common = np.arange(max_len) * dt

                return t_common, mean_wf, sem_wf

            # Plot eupnea mean +/- SEM (green)
            if eupnea_waveforms:
                t_eup, mean_eup, sem_eup = align_and_compute_stats(eupnea_waveforms)
                if t_eup is not None:
                    ax1.plot(t_eup, mean_eup, color='green', linewidth=2.5,
                            label=f'Eupnea (n={len(eupnea_waveforms)})', alpha=0.9)
                    ax1.fill_between(t_eup, mean_eup - sem_eup, mean_eup + sem_eup,
                                    color='green', alpha=0.25, linewidth=0)

            # Plot sniffing mean +/- SEM (purple)
            if sniffing_waveforms:
                t_snf, mean_snf, sem_snf = align_and_compute_stats(sniffing_waveforms)
                if t_snf is not None:
                    ax1.plot(t_snf, mean_snf, color='purple', linewidth=2.5,
                            label=f'Sniffing (n={len(sniffing_waveforms)})', alpha=0.9)
                    ax1.fill_between(t_snf, mean_snf - sem_snf, mean_snf + sem_snf,
                                    color='purple', alpha=0.25, linewidth=0)

            ax1.set_xlabel('Time from Onset (s)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Signal Amplitude', fontsize=11, fontweight='bold')
            ax1.set_title('Mean +/- SEM Breath Waveforms', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            # Placeholder when waveforms disabled
            ax1 = self.figure.add_subplot(total_rows, max_cols, 1)
            ax1.text(0.5, 0.5, 'Waveform plotting disabled\n(Enable checkbox to view)',
                    ha='center', va='center', fontsize=12, color='gray')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')

        # ========================================
        # Column 2: Mean Trajectory (Signal vs Derivative) - optional
        # ========================================
        if plot_waveforms:
            ax2 = self.figure.add_subplot(total_rows, max_cols, 2)

            # Helper function to compute mean trajectory with SEM
            def compute_mean_trajectory_with_sem(trajectories):
                """Pad trajectories to longest, then compute mean +/- SEM."""
                if len(trajectories) == 0:
                    return None, None, None, None

                # Find longest trajectory
                max_len = max(len(sig) for sig, deriv in trajectories)

                # Pad all trajectories to max_len with NaN
                padded_sigs = []
                padded_derivs = []
                for sig, deriv in trajectories:
                    if len(sig) < max_len:
                        padded_sig = np.full(max_len, np.nan)
                        padded_deriv = np.full(max_len, np.nan)
                        padded_sig[:len(sig)] = sig
                        padded_deriv[:len(deriv)] = deriv
                        padded_sigs.append(padded_sig)
                        padded_derivs.append(padded_deriv)
                    else:
                        padded_sigs.append(sig)
                        padded_derivs.append(deriv)

                # Stack and compute mean +/- SEM (ignoring NaN)
                sig_matrix = np.vstack(padded_sigs)
                deriv_matrix = np.vstack(padded_derivs)
                mean_sig = np.nanmean(sig_matrix, axis=0)
                mean_deriv = np.nanmean(deriv_matrix, axis=0)
                sem_sig = np.nanstd(sig_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(sig_matrix), axis=0))
                sem_deriv = np.nanstd(deriv_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(deriv_matrix), axis=0))

                return mean_sig, mean_deriv, sem_sig, sem_deriv

            # Plot mean trajectories as lines with SEM shading
            if eupnea_trajectories:
                mean_sig_eup, mean_deriv_eup, sem_sig_eup, sem_deriv_eup = compute_mean_trajectory_with_sem(eupnea_trajectories)
                if mean_sig_eup is not None:
                    # Plot mean trajectory line
                    ax2.plot(mean_sig_eup, mean_deriv_eup, color='green', linewidth=2.5,
                            label=f'Eupnea Mean Trajectory', alpha=0.9, marker='o', markersize=2, markevery=5)
                    # Add SEM as error bounds (create polygon for shaded region)
                    # Upper bound: (sig + sem_sig, deriv + sem_deriv)
                    # Lower bound: (sig - sem_sig, deriv - sem_deriv)
                    sig_upper = mean_sig_eup + sem_sig_eup
                    sig_lower = mean_sig_eup - sem_sig_eup
                    deriv_upper = mean_deriv_eup + sem_deriv_eup
                    deriv_lower = mean_deriv_eup - sem_deriv_eup
                    # Create polygon vertices (forward path on upper, backward on lower)
                    verts_x = np.concatenate([sig_upper, sig_lower[::-1]])
                    verts_y = np.concatenate([deriv_upper, deriv_lower[::-1]])
                    ax2.fill(verts_x, verts_y, color='green', alpha=0.15, linewidth=0)

            if sniffing_trajectories:
                mean_sig_snf, mean_deriv_snf, sem_sig_snf, sem_deriv_snf = compute_mean_trajectory_with_sem(sniffing_trajectories)
                if mean_sig_snf is not None:
                    # Plot mean trajectory line
                    ax2.plot(mean_sig_snf, mean_deriv_snf, color='purple', linewidth=2.5,
                            label=f'Sniffing Mean Trajectory', alpha=0.9, marker='o', markersize=2, markevery=5)
                    # Add SEM shading
                    sig_upper = mean_sig_snf + sem_sig_snf
                    sig_lower = mean_sig_snf - sem_sig_snf
                    deriv_upper = mean_deriv_snf + sem_deriv_snf
                    deriv_lower = mean_deriv_snf - sem_deriv_snf
                    verts_x = np.concatenate([sig_upper, sig_lower[::-1]])
                    verts_y = np.concatenate([deriv_upper, deriv_lower[::-1]])
                    ax2.fill(verts_x, verts_y, color='purple', alpha=0.15, linewidth=0)

            ax2.set_xlabel('Signal Amplitude', fontsize=11, fontweight='bold')
            ax2.set_ylabel('First Derivative (dy/dt)', fontsize=11, fontweight='bold')
            ax2.set_title('Mean Trajectory (Signal vs Derivative)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            # Placeholder when trajectories disabled
            ax2 = self.figure.add_subplot(total_rows, max_cols, 2)
            ax2.text(0.5, 0.5, 'Trajectory plotting disabled\n(Enable checkbox to view)',
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')

    def _plot_clusters(self, feature_matrix, labels, feature_keys):
        """Plot histograms for each feature + 2D scatter plots of clusters + waveform overlays."""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        self.figure.clear()

        n_features = len(feature_keys)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Use pre-assigned colors (purple for sniffing, green for eupnea, etc.)
        colors = self.cluster_colors

        # Calculate layout: NEW WAVEFORM ROW at top, then histograms in 2 columns, scatter plots below
        waveform_rows = 1  # Add 1 row at top for waveform overlays
        hist_cols = 2
        hist_rows = int(np.ceil(n_features / hist_cols))

        n_scatter_plots = min(6, (n_features * (n_features - 1)) // 2)  # Max 6 scatter plots

        if n_scatter_plots > 0:
            scatter_rows = int(np.ceil(np.sqrt(n_scatter_plots)))
            scatter_cols = int(np.ceil(n_scatter_plots / scatter_rows))
        else:
            scatter_rows = 0
            scatter_cols = 2

        total_rows = waveform_rows + hist_rows + scatter_rows
        max_cols = 2  # Fixed 2 columns for cleaner layout

        # Fixed number of bins for all histograms
        n_bins = 50

        # ========================================
        # Section 0: NEW WAVEFORM OVERLAYS (Top Row)
        # ========================================
        self._plot_waveform_overlays(total_rows, max_cols, labels, colors)

        # ========================================
        # Section 1: Feature Histograms (as line plots)
        # ========================================
        for feat_idx, feature_key in enumerate(feature_keys):
            row = waveform_rows + (feat_idx // hist_cols)  # Offset by waveform_rows
            col = feat_idx % hist_cols
            ax = self.figure.add_subplot(total_rows, max_cols, row * max_cols + col + 1)

            feature_data = feature_matrix[:, feat_idx]

            # Calculate histogram bins (same for all overlays)
            data_min = np.min(feature_data)
            data_max = np.max(feature_data)
            bins = np.linspace(data_min, data_max, n_bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot overall distribution (thicker gray line)
            counts_all, _ = np.histogram(feature_data, bins=bins)
            ax.plot(bin_centers, counts_all, color='gray', linewidth=2.5,
                   label='All Breaths', alpha=0.7)

            # Overlay each cluster's distribution
            for cluster_id in unique_labels:
                mask = labels == cluster_id
                cluster_data = feature_data[mask]
                counts_cluster, _ = np.histogram(cluster_data, bins=bins)

                # Label with pattern type if identified
                if cluster_id == self.sniffing_cluster_id:
                    cluster_label = f'Cluster {cluster_id} (Sniffing)'
                else:
                    cluster_label = f'Cluster {cluster_id}'

                ax.plot(bin_centers, counts_cluster, color=colors[cluster_id],
                       linewidth=2, label=cluster_label, alpha=0.8)

            ax.set_xlabel(feature_key, fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{feature_key} Distribution', fontsize=11, fontweight='bold')

        # ========================================
        # Section 2: 2D Scatter Plots
        # ========================================
        if n_scatter_plots > 0:
            plot_idx = 0

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if plot_idx >= n_scatter_plots:
                        break

                    # Calculate position in grid (below histograms, offset by waveform_rows)
                    row = waveform_rows + hist_rows + (plot_idx // scatter_cols)
                    col = plot_idx % scatter_cols
                    ax = self.figure.add_subplot(total_rows, max_cols, row * max_cols + col + 1)

                    # Plot each cluster with different color
                    for cluster_id in unique_labels:
                        mask = labels == cluster_id

                        # Label with pattern type if identified
                        if cluster_id == self.sniffing_cluster_id:
                            cluster_label = f"Cluster {cluster_id} (Sniffing)"
                        else:
                            cluster_label = f"Cluster {cluster_id}"

                        ax.scatter(feature_matrix[mask, i], feature_matrix[mask, j],
                                 c=[colors[cluster_id]], label=cluster_label,
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

                    ax.set_xlabel(feature_keys[i], fontsize=10)
                    ax.set_ylabel(feature_keys[j], fontsize=10)
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f'{feature_keys[i]} vs {feature_keys[j]}', fontsize=10, fontweight='bold')

                    plot_idx += 1

                if plot_idx >= n_scatter_plots:
                    break

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def show_help(self):
        """Display help dialog explaining GMM clustering."""
        from PyQt6.QtWidgets import QMessageBox

        help_text = """
<b>Gaussian Mixture Model (GMM) Breath Clustering</b>

<p><b>What is it?</b><br>
GMM is an unsupervised machine learning technique that automatically identifies distinct breathing patterns by grouping breaths with similar characteristics.</p>

<p><b>How it works:</b><br>
1. <b>Select Features:</b> Choose which breath metrics to use for clustering (e.g., frequency, timing, amplitude)<br>
2. <b>Set Number of Clusters:</b> Specify how many breathing patterns to identify (2-10)<br>
3. <b>Run GMM:</b> The algorithm analyzes all breaths from analyzed sweeps and groups them into clusters<br>
4. <b>View Results:</b> Examine cluster statistics and 2D scatter plots showing pattern separation</p>

<p><b>Typical Breathing Patterns (4 clusters recommended):</b><br>
• <b>Normal Breathing (Eupnea):</b> Regular frequency and amplitude<br>
• <b>Apnea/Pauses:</b> Extended inter-breath intervals<br>
• <b>Sighs:</b> Large amplitude, longer duration<br>
• <b>Sniffing:</b> High frequency, short duration, small amplitude</p>

<p><b>Tips:</b><br>
• Start with default features (IF, Ti, Amp Insp) and 4 clusters<br>
• Use at least 2-3 features for meaningful separation<br>
• More analyzed sweeps = better clustering results<br>
• Experiment with different feature combinations to find optimal separation</p>

<p><b>Recommended Settings:</b><br>
• <b>For basic pattern detection:</b> IF, Ti, Amp Insp (3 features, 4 clusters)<br>
• <b>For detailed analysis:</b> All 7 features, 5-6 clusters<br>
• <b>For sigh detection:</b> Include Amp Insp, Area Insp (high amplitude patterns)</p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("GMM Clustering Help")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _auto_apply_gmm_results(self):
        """Automatically apply GMM clustering results to main plot (called after successful GMM run)."""
        import numpy as np

        if self.sniffing_cluster_id is None:
            print("[gmm-dialog] Cannot auto-apply: no sniffing cluster identified")
            return

        # Update eupnea detection mode and parameters based on user selection
        selected_mode = "gmm" if self.gmm_mode_radio.isChecked() else "frequency"
        self.main_window.eupnea_detection_mode = selected_mode

        # Save frequency-based parameters (used when mode is "frequency")
        if not hasattr(self.main_window, 'eupnea_freq_threshold'):
            self.main_window.eupnea_freq_threshold = 5.0
        self.main_window.eupnea_freq_threshold = self.freq_threshold_spin.value()
        self.main_window.eupnea_min_duration = self.min_duration_spin.value()

        print(f"[gmm-dialog] Auto-applied eupnea detection mode: {selected_mode}")
        print(f"[gmm-dialog] Frequency threshold: {self.main_window.eupnea_freq_threshold} Hz, Min duration: {self.main_window.eupnea_min_duration} s")

        # Store GMM probabilities for each breath (needed for GMM-based eupnea detection)
        # Format: {sweep_idx: {breath_idx: sniffing_probability}}
        if not hasattr(self.main_window.state, 'gmm_sniff_probabilities'):
            self.main_window.state.gmm_sniff_probabilities = {}

        for i, (sweep_idx, breath_idx) in enumerate(self.breath_cycles):
            if sweep_idx not in self.main_window.state.gmm_sniff_probabilities:
                self.main_window.state.gmm_sniff_probabilities[sweep_idx] = {}

            # Get probability of being in sniffing cluster
            sniff_prob = self.cluster_probabilities[i, self.sniffing_cluster_id]
            self.main_window.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

        # Collect all sniffing breaths
        sniffing_regions_by_sweep = {}  # sweep_idx -> list of (start_time, end_time)

        for i, (sweep_idx, breath_idx) in enumerate(self.breath_cycles):
            cluster_id = self.cluster_labels[i]

            if cluster_id == self.sniffing_cluster_id:
                # This breath is sniffing - get its time range
                breath_data = self.main_window.state.breath_by_sweep.get(sweep_idx)
                if breath_data is None:
                    continue

                onsets = breath_data.get('onsets', np.array([]))
                offsets = breath_data.get('offsets', np.array([]))

                if breath_idx >= len(onsets):
                    continue

                # Get time range for this breath
                t = self.main_window.state.t
                start_time = t[int(onsets[breath_idx])]

                # Offset for this breath (use expiratory offset if available)
                if breath_idx < len(offsets):
                    end_idx = int(offsets[breath_idx])
                else:
                    # Fallback: use next onset or end of trace
                    if breath_idx + 1 < len(onsets):
                        end_idx = int(onsets[breath_idx + 1])
                    else:
                        end_idx = len(t) - 1

                end_time = t[end_idx]

                # Add to sniffing regions for this sweep
                if sweep_idx not in sniffing_regions_by_sweep:
                    sniffing_regions_by_sweep[sweep_idx] = []
                sniffing_regions_by_sweep[sweep_idx].append((start_time, end_time))

        # Apply sniffing regions to main window state
        total_regions = 0
        for sweep_idx, regions in sniffing_regions_by_sweep.items():
            if sweep_idx not in self.main_window.state.sniff_regions_by_sweep:
                self.main_window.state.sniff_regions_by_sweep[sweep_idx] = []

            # Add all regions
            self.main_window.state.sniff_regions_by_sweep[sweep_idx].extend(regions)

            # Merge overlapping/adjacent regions
            self.main_window._merge_sniff_regions(sweep_idx)

            total_regions += len(self.main_window.state.sniff_regions_by_sweep[sweep_idx])

        # Redraw main plot to show sniffing regions AND eupnea (based on selected mode)
        self.main_window.redraw_main_plot()

        # Log results
        n_sniffing_breaths = np.sum(self.cluster_labels == self.sniffing_cluster_id)
        mode_msg = "GMM-based" if selected_mode == "gmm" else "Frequency-based"
        print(f"[gmm-dialog] Auto-applied {n_sniffing_breaths} sniffing breaths across {len(sniffing_regions_by_sweep)} sweep(s)")
        print(f"[gmm-dialog] Created {total_regions} merged sniffing regions")
        print(f"[gmm-dialog] Eupnea detection mode: {mode_msg}")

    def on_apply_to_plot(self):
        """Apply GMM clustering results to main plot by marking sniffing breaths."""
        from PyQt6.QtWidgets import QMessageBox
        import numpy as np

        if self.sniffing_cluster_id is None:
            QMessageBox.warning(
                self,
                "No Sniffing Cluster",
                "Cannot identify sniffing cluster. Make sure you include 'if' or 'ti' features."
            )
            return

        # Warn again if clustering quality was poor
        if self.had_quality_warning:
            reply = QMessageBox.question(
                self,
                "Apply Despite Quality Warning?",
                "The clustering quality was flagged as questionable.\n\n"
                "The identified 'sniffing' breaths may just be normal variation,\n"
                "not true sniffing behavior.\n\n"
                "Are you sure you want to apply these results to the plot?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Update eupnea detection mode and parameters based on user selection
        selected_mode = "gmm" if self.gmm_mode_radio.isChecked() else "frequency"
        self.main_window.eupnea_detection_mode = selected_mode

        # Save frequency-based parameters (used when mode is "frequency")
        if not hasattr(self.main_window, 'eupnea_freq_threshold'):
            self.main_window.eupnea_freq_threshold = 5.0
        self.main_window.eupnea_freq_threshold = self.freq_threshold_spin.value()
        self.main_window.eupnea_min_duration = self.min_duration_spin.value()

        print(f"[gmm-clustering] Updated eupnea detection mode to: {selected_mode}")
        print(f"[gmm-clustering] Frequency threshold: {self.main_window.eupnea_freq_threshold} Hz, Min duration: {self.main_window.eupnea_min_duration} s")

        # Store GMM probabilities for each breath (needed for GMM-based eupnea detection)
        # Format: {sweep_idx: {breath_idx: sniffing_probability}}
        if not hasattr(self.main_window.state, 'gmm_sniff_probabilities'):
            self.main_window.state.gmm_sniff_probabilities = {}

        for i, (sweep_idx, breath_idx) in enumerate(self.breath_cycles):
            if sweep_idx not in self.main_window.state.gmm_sniff_probabilities:
                self.main_window.state.gmm_sniff_probabilities[sweep_idx] = {}

            # Get probability of being in sniffing cluster
            sniff_prob = self.cluster_probabilities[i, self.sniffing_cluster_id]
            self.main_window.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

        # Collect all sniffing breaths
        sniffing_regions_by_sweep = {}  # sweep_idx -> list of (start_time, end_time)

        for i, (sweep_idx, breath_idx) in enumerate(self.breath_cycles):
            cluster_id = self.cluster_labels[i]

            if cluster_id == self.sniffing_cluster_id:
                # This breath is sniffing - get its time range
                breath_data = self.main_window.state.breath_by_sweep.get(sweep_idx)
                if breath_data is None:
                    continue

                onsets = breath_data.get('onsets', np.array([]))
                offsets = breath_data.get('offsets', np.array([]))

                if breath_idx >= len(onsets):
                    continue

                # Get time range for this breath
                t = self.main_window.state.t
                start_time = t[int(onsets[breath_idx])]

                # Offset for this breath (use expiratory offset if available)
                if breath_idx < len(offsets):
                    end_idx = int(offsets[breath_idx])
                else:
                    # Fallback: use next onset or end of trace
                    if breath_idx + 1 < len(onsets):
                        end_idx = int(onsets[breath_idx + 1])
                    else:
                        end_idx = len(t) - 1

                end_time = t[end_idx]

                # Add to sniffing regions for this sweep
                if sweep_idx not in sniffing_regions_by_sweep:
                    sniffing_regions_by_sweep[sweep_idx] = []
                sniffing_regions_by_sweep[sweep_idx].append((start_time, end_time))

        # Apply sniffing regions to main window state
        total_regions = 0
        for sweep_idx, regions in sniffing_regions_by_sweep.items():
            if sweep_idx not in self.main_window.state.sniff_regions_by_sweep:
                self.main_window.state.sniff_regions_by_sweep[sweep_idx] = []

            # Add all regions
            self.main_window.state.sniff_regions_by_sweep[sweep_idx].extend(regions)

            # Merge overlapping/adjacent regions
            self.main_window._merge_sniff_regions(sweep_idx)

            total_regions += len(self.main_window.state.sniff_regions_by_sweep[sweep_idx])

        # Redraw main plot to show sniffing regions AND eupnea (based on selected mode)
        self.main_window.redraw_main_plot()

        # Show success message
        n_sniffing_breaths = np.sum(self.cluster_labels == self.sniffing_cluster_id)
        mode_msg = "GMM-based" if selected_mode == "gmm" else "Frequency-based"
        QMessageBox.information(
            self,
            "Applied to Plot",
            f"Marked {n_sniffing_breaths} sniffing breaths across {len(sniffing_regions_by_sweep)} sweep(s).\n\n"
            f"After merging adjacent regions: {total_regions} total sniffing region(s).\n\n"
            f"Purple background regions are now visible on the main plot.\n\n"
            f"Eupnea detection mode: {mode_msg}"
        )

        print(f"[gmm-clustering] Applied {n_sniffing_breaths} sniffing breaths to main plot")
        print(f"[gmm-clustering] Created {total_regions} merged sniffing regions across {len(sniffing_regions_by_sweep)} sweeps")

