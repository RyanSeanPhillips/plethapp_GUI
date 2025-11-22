"""
Analysis Options Dialog

Multi-tabbed dialog consolidating:
- Peak Detection (Auto-Threshold)
- Breath Classification (GMM)
- Outlier Detection
- ML Settings
"""

from PyQt6.QtWidgets import QDialog, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.uic import loadUi
from pathlib import Path


class AnalysisOptionsDialog(QDialog):
    """Multi-tabbed dialog for all analysis configuration options."""

    def __init__(self, state, parent=None):
        super().__init__(parent)

        self.state = state
        self.parent_window = parent

        # Load the UI file
        ui_path = Path(__file__).parent.parent / "ui" / "analysis_options_dialog.ui"
        loadUi(ui_path, self)

        # Enable resizing
        self.setMinimumSize(1000, 700)  # Set minimum size but allow resizing
        self.setSizeGripEnabled(True)  # Show resize grip in corner

        # Load settings (position, size, and training data path)
        if parent and hasattr(parent, 'settings'):
            self.settings = parent.settings
            # Restore window position
            if self.settings.contains("analysis_dialog_pos"):
                pos = self.settings.value("analysis_dialog_pos")
                self.move(pos)
            # Restore window size
            if self.settings.contains("analysis_dialog_size"):
                size = self.settings.value("analysis_dialog_size")
                self.resize(size)
            # Restore training data path
            self.last_training_data_dir = self.settings.value("ml_training_data_path", "")
            # Restore models directory path
            self.last_models_dir = self.settings.value("ml_models_path", "")
        else:
            self.settings = None
            self.last_training_data_dir = ""
            self.last_models_dir = ""

        # Connect tab change signal to refresh tabs BEFORE initializing tabs
        self.tabWidget.currentChanged.connect(self._on_tab_changed)
        print(f"[AnalysisOptions] Connected tab change signal. Current tab: {self.tabWidget.currentIndex()}")

        # Initialize sub-dialogs
        self._init_peak_detection_tab()
        self._init_breath_classification_tab()
        self._init_outlier_detection_tab()
        self._init_ml_settings_tab()
        print(f"[AnalysisOptions] All tabs initialized")

    def set_active_tab(self, tab_name):
        """
        Switch to a specific tab by name.

        Args:
            tab_name (str): One of 'peak_detection', 'breath_classification', 'outlier_detection', 'ml_training'
        """
        tab_indices = {
            'peak_detection': 0,
            'auto_threshold': 0,  # Alias
            'breath_classification': 1,
            'eup_sniff': 1,  # Alias
            'gmm': 1,  # Alias
            'outlier_detection': 2,
            'outliers': 2,  # Alias
            'ml_training': 3,
            'ml': 3  # Alias
        }

        index = tab_indices.get(tab_name.lower())
        if index is not None:
            self.tabWidget.setCurrentIndex(index)
        else:
            print(f"[AnalysisOptionsDialog] Warning: Unknown tab name '{tab_name}'")

    def _on_tab_changed(self, index):
        """Refresh tab content when user switches tabs."""
        print(f"[AnalysisOptions] Tab changed to index {index}")
        try:
            if index == 0:
                # Peak Detection tab
                print("[AnalysisOptions] Refreshing Peak Detection tab...")
                self._refresh_peak_detection_tab()
            elif index == 1:
                # Breath Classification tab
                print("[AnalysisOptions] Refreshing Breath Classification tab...")
                self._refresh_breath_classification_tab()
            # Note: Outlier Detection (tab 2) and ML Settings (tab 3) don't need refreshing
        except Exception as e:
            print(f"[AnalysisOptions] ERROR in _on_tab_changed: {e}")
            import traceback
            traceback.print_exc()

    def showEvent(self, event):
        """Refresh tabs when dialog is shown."""
        super().showEvent(event)
        # Refresh the currently active tab
        current_index = self.tabWidget.currentIndex()
        print(f"[AnalysisOptions] showEvent - current tab index: {current_index}")
        if current_index == 0:
            self._refresh_peak_detection_tab()
        elif current_index == 1:
            self._refresh_breath_classification_tab()
        # Note: ML settings tab doesn't need refreshing

    def closeEvent(self, event):
        """Save position and size when dialog is closed."""
        if self.settings:
            self.settings.setValue("analysis_dialog_pos", self.pos())
            self.settings.setValue("analysis_dialog_size", self.size())
        super().closeEvent(event)

    def _init_peak_detection_tab(self):
        """Initialize Peak Detection tab with ProminenceThresholdDialog content."""
        self._refresh_peak_detection_tab()

    def _refresh_peak_detection_tab(self):
        """Refresh the Peak Detection tab with current data."""
        from dialogs.prominence_threshold_dialog import ProminenceThresholdDialog
        from PyQt6.QtWidgets import QLabel, QApplication
        import numpy as np

        print("[AnalysisOptions] _refresh_peak_detection_tab() called")

        container = self.peak_detection_container

        # Get or create layout
        layout = container.layout()
        if layout is None:
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            print("[AnalysisOptions] Created new layout for peak detection container")

        # Check if we have data
        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            # Clear layout and show placeholder
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            QApplication.processEvents()

            # Show placeholder if no channel selected
            if not st.sweeps:
                label = QLabel("Please load a data file first.")
            else:
                label = QLabel("Please select a channel from the 'Analyze Channel' dropdown in the main window.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("padding: 20px;")
            layout.addWidget(label)
            print("[AnalysisOptions] Peak detection tab: No data loaded")
            return

        # Check if we already created a dialog for this tab (don't recreate on tab switches)
        if hasattr(self, 'prominence_dialog') and self.prominence_dialog is not None:
            # Dialog already exists in this Analysis Options dialog
            try:
                # Verify it's still valid (not deleted)
                _ = self.prominence_dialog.isVisible()
                print("[AnalysisOptions] Reusing existing prominence dialog (already in layout)")
                return
            except RuntimeError:
                # Dialog was deleted, create new one
                print("[AnalysisOptions] Previous dialog was deleted, creating new one")
                self.prominence_dialog = None

        # Clear existing widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        QApplication.processEvents()

        # Check if we have the cached peak heights from auto-detect
        if hasattr(self.parent_window, 'all_peak_heights') and self.parent_window.all_peak_heights is not None:
            print(f"[AnalysisOptions] Creating prominence dialog using cached peak heights ({len(self.parent_window.all_peak_heights)} peaks)")

            # Concatenate all sweeps (needed for ProminenceThresholdDialog)
            all_sweeps_data = []
            n_sweeps = st.sweeps[st.analyze_chan].shape[1]
            for sweep_idx in range(n_sweeps):
                if sweep_idx in st.omitted_sweeps:
                    continue
                y_sweep = self.parent_window._get_processed_for(st.analyze_chan, sweep_idx)
                all_sweeps_data.append(y_sweep)

            if not all_sweeps_data:
                label = QLabel("All sweeps are omitted.")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
                return

            y_data = np.concatenate(all_sweeps_data)

            # Get current prominence
            try:
                current_prom = self.parent_window.PeakPromValueSpinBox.value() if self.parent_window.PeakPromValueSpinBox.value() > 0 else None
            except:
                current_prom = None

            # Create dialog - it will detect peaks but use cached data if available
            self.prominence_dialog = ProminenceThresholdDialog(
                parent=container,  # Set container as parent for proper embedding
                y_data=y_data,
                sr_hz=st.sr_hz,
                current_prom=current_prom,
                current_min_dist=self.parent_window.peak_min_dist
            )

            # Hide the dialog's window frame (we just want the content)
            self.prominence_dialog.setWindowFlags(self.prominence_dialog.windowFlags() & ~0x00000001)

            layout.addWidget(self.prominence_dialog)
            print("[AnalysisOptions] Created new prominence dialog with cached peak data")
        else:
            # No cached data available - show message
            print("[AnalysisOptions] No cached peak detection data available")
            label = QLabel("Please run peak detection first (it will auto-run when you select a channel).")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("padding: 20px;")
            layout.addWidget(label)
            return

        # Hide OK/Cancel buttons (not needed when embedded)
        # Find the button box (last widget in main layout)
        if hasattr(self, 'prominence_dialog') and self.prominence_dialog is not None:
            main_layout = self.prominence_dialog.layout()
            if main_layout and main_layout.count() > 0:
                # The button box is the last item in the main layout
                last_item = main_layout.itemAt(main_layout.count() - 1)
                if last_item and last_item.widget():
                    button_box = last_item.widget()
                    button_box.hide()

            # Add action buttons to the left panel (after histogram controls)
            self._add_action_buttons_to_left_panel()

    def _add_action_buttons_to_left_panel(self):
        """Add Apply Threshold and Detect Peaks buttons to the left panel of the prominence dialog."""
        if not hasattr(self, 'prominence_dialog'):
            return

        try:
            from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QGroupBox

            # Find the left panel in the prominence dialog
            # The dialog has a main layout with content_layout (HBoxLayout)
            # content_layout has left_panel and right_panel
            main_layout = self.prominence_dialog.layout()
            if not main_layout or main_layout.count() < 2:
                return

            # Skip title and description, find content_layout (HBoxLayout)
            content_layout = None
            for i in range(main_layout.count()):
                item = main_layout.itemAt(i)
                if item and item.layout() and hasattr(item.layout(), 'count'):
                    # Check if this is the HBoxLayout with left and right panels
                    layout_item = item.layout()
                    if layout_item.count() >= 2:
                        content_layout = layout_item
                        break

            if not content_layout:
                return

            # Get the left panel (first item in content_layout)
            left_panel_item = content_layout.itemAt(0)
            if not left_panel_item or not left_panel_item.widget():
                return

            left_panel = left_panel_item.widget()
            left_layout = left_panel.layout()
            if not left_layout:
                return

            # Create a group box for the action buttons
            action_group = QGroupBox("Actions")
            action_layout = QVBoxLayout()
            action_layout.setSpacing(10)

            # Apply Threshold button (updates plot threshold line)
            self.apply_threshold_btn = QPushButton("Apply Threshold to Plot")
            self.apply_threshold_btn.setMinimumHeight(35)
            self.apply_threshold_btn.setMaximumWidth(350)  # Match left panel width
            self.apply_threshold_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d2d;
                    color: #d4d4d4;
                    border: 1px solid #3e3e42;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 8px 12px;
                }
                QPushButton:hover {
                    background-color: #3e3e42;
                }
                QPushButton:pressed {
                    background-color: #505050;
                }
            """)
            self.apply_threshold_btn.clicked.connect(self._apply_threshold_to_plot)
            action_layout.addWidget(self.apply_threshold_btn)

            # Detect Peaks button (runs full peak detection)
            self.detect_peaks_btn = QPushButton("Detect Peaks")
            self.detect_peaks_btn.setMinimumHeight(35)
            self.detect_peaks_btn.setMaximumWidth(350)  # Match left panel width
            self.detect_peaks_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0e639c;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    padding: 8px 12px;
                }
                QPushButton:hover {
                    background-color: #1177bb;
                }
                QPushButton:pressed {
                    background-color: #0d5689;
                }
            """)
            self.detect_peaks_btn.clicked.connect(self._detect_peaks)
            action_layout.addWidget(self.detect_peaks_btn)

            action_group.setLayout(action_layout)

            # Insert the action group before the stretch at the end of left_layout
            # Find the stretch item (should be last)
            stretch_index = -1
            for i in range(left_layout.count()):
                item = left_layout.itemAt(i)
                if item and item.spacerItem():
                    stretch_index = i
                    break

            if stretch_index >= 0:
                # Insert before the stretch
                left_layout.insertWidget(stretch_index, action_group)
            else:
                # No stretch found, just add at the end
                left_layout.addWidget(action_group)

            print("[analysis-options] Added action buttons to left panel")

        except Exception as e:
            print(f"[analysis-options] Error adding action buttons: {e}")

    def _init_breath_classification_tab(self):
        """Initialize Breath Classification tab with GMMClusteringDialog content."""
        self._refresh_breath_classification_tab()

    def _refresh_breath_classification_tab(self):
        """Refresh the Breath Classification (GMM) tab with current data."""
        print("[AnalysisOptions] _refresh_breath_classification_tab() called - ENTRY")

        from dialogs.gmm_clustering_dialog import GMMClusteringDialog
        from PyQt6.QtWidgets import QLabel, QApplication, QSizePolicy
        from PyQt6.QtCore import Qt

        print("[AnalysisOptions] Imports complete")

        container = self.breath_classification_container
        print(f"[AnalysisOptions] Container widget: {container}, visible: {container.isVisible()}, size: {container.size()}")

        # Get or create layout
        layout = container.layout()
        if layout is None:
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            print("[AnalysisOptions] Created new layout for container")
        else:
            # Clear existing widgets
            print(f"[AnalysisOptions] Clearing existing layout (has {layout.count()} items)")
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            # Process events to ensure widgets are actually deleted
            QApplication.processEvents()

        # Check if we have data
        st = self.state
        print(f"[AnalysisOptions] State check: analyze_chan={st.analyze_chan}, has sweeps={bool(st.sweeps)}")

        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            # Show placeholder if no channel selected
            if not st.sweeps:
                label = QLabel("Please load a data file first.")
            else:
                label = QLabel("Please select a channel from the 'Analyze Channel' dropdown in the main window.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("padding: 20px;")
            layout.addWidget(label)
            return

        # Check if peaks have been detected (check both old and new peak storage)
        has_peaks = (hasattr(st, 'peaks_by_sweep') and st.peaks_by_sweep and len(st.peaks_by_sweep) > 0) or \
                    (hasattr(st, 'all_peaks_by_sweep') and st.all_peaks_by_sweep and len(st.all_peaks_by_sweep) > 0)

        if not has_peaks:
            label = QLabel("Please detect peaks first using 'Find Peaks & Events' in the main window.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet("padding: 20px;")
            layout.addWidget(label)
            print("[AnalysisOptions] Breath classification tab: No peaks detected yet")
            return

        # Create the GMM dialog
        print("[AnalysisOptions] Creating GMM dialog for breath classification tab...")

        # Debug: Check if GMM results exist in state
        if hasattr(st, 'gmm_sniff_probabilities'):
            print(f"[AnalysisOptions] State has gmm_sniff_probabilities: {bool(st.gmm_sniff_probabilities)}")
            if st.gmm_sniff_probabilities:
                print(f"[AnalysisOptions] GMM probabilities count: {len(st.gmm_sniff_probabilities)}")
        else:
            print("[AnalysisOptions] State does NOT have gmm_sniff_probabilities attribute")

        try:
            from PyQt6.QtCore import Qt

            # CRITICAL: Must create with NO parent initially, set flags, THEN parent
            self.gmm_dialog = GMMClusteringDialog(parent=None, main_window=self.parent_window)
            print(f"[AnalysisOptions] GMM dialog created. Original flags: {self.gmm_dialog.windowFlags()}")

            # Set widget flag to prevent it from being a window
            # Must be done BEFORE setParent() or it won't work
            self.gmm_dialog.setWindowFlag(Qt.WindowType.Widget, True)
            self.gmm_dialog.setWindowFlag(Qt.WindowType.Window, False)
            self.gmm_dialog.setWindowFlag(Qt.WindowType.Dialog, False)
            print(f"[AnalysisOptions] Updated flags: {self.gmm_dialog.windowFlags()}")

            # Set parent AFTER flags are configured
            self.gmm_dialog.setParent(container)
            print(f"[AnalysisOptions] Set parent to container. Parent: {self.gmm_dialog.parent()}")

            # Add to layout
            layout.addWidget(self.gmm_dialog)
            self.gmm_dialog.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            print(f"[AnalysisOptions] GMM dialog added to layout. Visible: {self.gmm_dialog.isVisible()}")
        except Exception as e:
            print(f"[AnalysisOptions] ERROR creating GMM dialog: {e}")
            import traceback
            traceback.print_exc()
            label = QLabel(f"Error creating GMM dialog: {e}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            layout.addWidget(label)

    def _init_outlier_detection_tab(self):
        """Initialize Outlier Detection tab with OutlierMetricsDialog content."""
        from dialogs.outlier_metrics_dialog import OutlierMetricsDialog
        from core.metrics import METRICS

        container = self.outlier_detection_container
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Get available metrics from core.metrics
        # Filter to only numeric metrics (exclude region detection functions)
        numeric_metrics = {k: v for k, v in METRICS.items()
                          if k not in ["eupnic", "apnea", "regularity"]}

        # Get currently selected metrics from parent window
        selected_metrics = []
        if self.parent_window and hasattr(self.parent_window, 'outlier_metrics'):
            selected_metrics = self.parent_window.outlier_metrics

        # Create the outlier metrics dialog with data
        self.outlier_dialog = OutlierMetricsDialog(
            parent=None,
            available_metrics=list(numeric_metrics.keys()),
            selected_metrics=selected_metrics
        )

        # Hide the dialog's window frame (we just want the content)
        self.outlier_dialog.setWindowFlags(self.outlier_dialog.windowFlags() & ~0x00000001)

        # Hide OK/Cancel buttons (not needed when embedded)
        # Find and hide the button layout at the bottom
        main_layout = self.outlier_dialog.layout()
        if main_layout and main_layout.count() > 0:
            # The button layout is the last item in the main layout
            last_item = main_layout.itemAt(main_layout.count() - 1)
            if last_item and last_item.layout():
                button_layout = last_item.layout()
                # Hide all widgets in the button layout
                for i in range(button_layout.count()):
                    item = button_layout.itemAt(i)
                    if item and item.widget():
                        item.widget().hide()

        # Add it to the container
        layout.addWidget(self.outlier_dialog)

    def _init_ml_settings_tab(self):
        """Initialize ML Settings tab with redesigned interface."""
        from PyQt6.QtWidgets import (
            QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
            QLineEdit, QGroupBox, QScrollArea, QWidget, QProgressBar
        )

        # Get or create layout for the tab
        tab = self.tab_ml_settings

        # Clear existing layout
        if tab.layout():
            old_layout = tab.layout()
            while old_layout.count():
                child = old_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            from PyQt6.QtWidgets import QWidget
            QWidget().setLayout(old_layout)  # Take ownership to delete

        # Create main scroll area for entire tab
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        main_scroll.setStyleSheet("background-color: #1e1e1e; border: none;")

        # Create container widget for scrollable content
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # === Load Models Section (at top) ===
        load_models_group = QGroupBox("Loaded ML Models")
        load_models_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3e3e42;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        load_models_layout = QHBoxLayout()

        # Model status display
        self.loaded_models_status = QLabel("No models loaded")
        self.loaded_models_status.setStyleSheet("color: #888; font-style: italic;")
        self.loaded_models_status.setWordWrap(True)
        load_models_layout.addWidget(self.loaded_models_status, 1)

        # Load Models button
        btn_load_models = QPushButton("Load Models...")
        btn_load_models.setMinimumHeight(40)
        btn_load_models.setMinimumWidth(150)
        btn_load_models.setStyleSheet("""
            QPushButton {
                background-color: #094771;
                color: #ffffff;
                border: 2px solid #0a5a8a;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #0a5a8a;
                border: 2px solid #0c6ea3;
            }
            QPushButton:pressed {
                background-color: #0c6ea3;
            }
        """)
        btn_load_models.clicked.connect(self._load_ml_models)
        load_models_layout.addWidget(btn_load_models)

        load_models_group.setLayout(load_models_layout)
        layout.addWidget(load_models_group)

        # === Top Row: Training Data Location (left) and Model Training (right) ===
        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(10)

        # === Training Data Path Section (2/3 width) ===
        path_group = QGroupBox("Training Data Location")
        path_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3e3e42;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        path_group_layout = QVBoxLayout()

        # Path selection row
        path_row_layout = QHBoxLayout()
        self.training_data_path_edit = QLineEdit()
        self.training_data_path_edit.setPlaceholderText("Select training data directory...")
        self.training_data_path_edit.setReadOnly(True)

        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_training_data)
        btn_browse.setMinimumWidth(100)
        btn_browse.setMinimumHeight(32)
        btn_browse.setStyleSheet("""
            QPushButton {
                background-color: #094771;
                color: #ffffff;
                border: 2px solid #0a5a8a;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #0a5a8a;
                border: 2px solid #0c6ea3;
            }
            QPushButton:pressed {
                background-color: #0c6ea3;
            }
        """)

        path_row_layout.addWidget(QLabel("Path:"))
        path_row_layout.addWidget(self.training_data_path_edit, 1)
        path_row_layout.addWidget(btn_browse)
        path_group_layout.addLayout(path_row_layout)

        # File list with metadata
        from PyQt6.QtWidgets import QTextEdit, QCheckBox
        file_list_label = QLabel("Training files found:")
        file_list_label.setStyleSheet("margin-top: 10px;")
        path_group_layout.addWidget(file_list_label)

        self.file_list_display = QTextEdit()
        self.file_list_display.setReadOnly(True)
        self.file_list_display.setMaximumHeight(100)
        self.file_list_display.setStyleSheet("background-color: #2d2d30; color: #d4d4d4; font-family: monospace; font-size: 9pt; border: 1px solid #3e3e42;")
        self.file_list_display.setPlaceholderText("Browse to a directory to see available files...")
        path_group_layout.addWidget(self.file_list_display)

        # Deduplication checkbox
        self.deduplicate_checkbox = QCheckBox("Remove duplicate files (by source filename)")
        self.deduplicate_checkbox.setChecked(True)
        self.deduplicate_checkbox.setStyleSheet("color: #d4d4d4;")
        self.deduplicate_checkbox.setToolTip("If multiple training files have the same source file, only keep the most recent one")
        path_group_layout.addWidget(self.deduplicate_checkbox)

        path_group.setLayout(path_group_layout)
        top_row_layout.addWidget(path_group, 2)  # 2/3 width

        # Auto-load saved path and scan for files
        if self.last_training_data_dir and Path(self.last_training_data_dir).exists():
            self.training_data_path_edit.setText(self.last_training_data_dir)
            # Trigger file scan
            self._scan_training_files(self.last_training_data_dir)

        # === Training Control Section (1/3 width) ===
        control_group = QGroupBox("Model Training")
        control_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3e3e42;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        control_layout = QVBoxLayout()

        # Retrain All button
        btn_retrain_all = QPushButton("Train All Models\n(9 total: 3 models × 3 algorithms)")
        btn_retrain_all.clicked.connect(self._retrain_all_models)
        btn_retrain_all.setMinimumHeight(60)
        btn_retrain_all.setStyleSheet("""
            QPushButton {
                background-color: #4ec9b0;
                color: #1e1e1e;
                border: 2px solid #5fd9c0;
                padding: 10px 16px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #5fd9c0;
                border: 2px solid #70e9d0;
            }
            QPushButton:pressed {
                background-color: #3db89a;
            }
        """)
        control_layout.addWidget(btn_retrain_all)

        # Progress bar
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        self.training_progress.setTextVisible(True)
        self.training_progress.setStyleSheet("""
            QProgressBar {
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                border: 1px solid #3e3e42;
            }
            QProgressBar::chunk {
                background-color: #4ec9b0;
            }
        """)
        control_layout.addWidget(self.training_progress)

        # Status label
        self.training_status_label = QLabel("Ready to train")
        self.training_status_label.setStyleSheet("color: #4ec9b0; font-style: italic;")
        control_layout.addWidget(self.training_status_label)

        # Save All Models button (initially hidden)
        self.save_all_models_btn = QPushButton("Save All Models...")
        self.save_all_models_btn.setMinimumHeight(40)
        self.save_all_models_btn.setVisible(False)
        self.save_all_models_btn.setStyleSheet("""
            QPushButton {
                background-color: #094771;
                color: #ffffff;
                border: 2px solid #0a5a8a;
                padding: 8px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #0a5a8a;
                border: 2px solid #0c6ea3;
            }
            QPushButton:pressed {
                background-color: #0c6ea3;
            }
        """)
        self.save_all_models_btn.clicked.connect(self._save_all_models)
        control_layout.addWidget(self.save_all_models_btn)

        control_layout.addStretch()  # Push content to top
        control_group.setLayout(control_layout)
        top_row_layout.addWidget(control_group, 1)  # 1/3 width

        layout.addLayout(top_row_layout)

        # === Model Comparison Statistics Section ===
        self.comparison_group = QGroupBox("Model Comparison Statistics")
        self.comparison_group.setMinimumHeight(400)  # Increased height to show full content
        self.comparison_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3e3e42;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        self.comparison_layout = QVBoxLayout()
        comparison_placeholder = QLabel("Train models to see comparison statistics")
        comparison_placeholder.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
        comparison_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.comparison_layout.addWidget(comparison_placeholder)
        self.comparison_group.setLayout(self.comparison_layout)
        layout.addWidget(self.comparison_group)

        # === Results Section ===
        results_group = QGroupBox("Training Results")
        results_group.setMinimumHeight(1200)  # Increased to avoid scrolling
        results_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3e3e42;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
        """)
        results_layout = QVBoxLayout()

        # Direct widget (no internal scroll since tab itself scrolls)
        self.results_widget = QWidget()
        self.results_widget.setStyleSheet("background-color: #1e1e1e;")
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setSpacing(10)

        # Initial message
        initial_label = QLabel("No training results yet. Select training data and click 'Train All Models'.")
        initial_label.setStyleSheet("color: #888; font-style: italic; padding: 20px;")
        initial_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_layout.addWidget(initial_label)

        results_layout.addWidget(self.results_widget)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Set scroll content and add to tab
        main_scroll.setWidget(scroll_content)

        # Add scroll area to tab
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(main_scroll)

        # Set the last training data directory if we have it
        if self.last_training_data_dir:
            self.training_data_path_edit.setText(self.last_training_data_dir)

        # Auto-load ML models from last used directory (if available)
        self._auto_load_ml_models()

    def _auto_load_ml_models(self):
        """Silently load ML models from last used directory on startup."""
        from pathlib import Path
        import core.ml_prediction as ml_prediction

        # Only auto-load if we have a saved path
        if not self.last_models_dir:
            return

        models_path = Path(self.last_models_dir)

        # Check if directory still exists
        if not models_path.exists() or not models_path.is_dir():
            print(f"[Auto-load] Saved models directory no longer exists: {models_path}")
            return

        try:
            # Look for model files
            model_files = list(models_path.glob("model*.pkl"))

            if not model_files:
                print(f"[Auto-load] No model files found in: {models_path}")
                return

            # Load all models
            loaded_models = {}
            for model_file in model_files:
                try:
                    model, metadata = ml_prediction.load_model(model_file)
                    model_key = model_file.stem
                    loaded_models[model_key] = {
                        'model': model,
                        'metadata': metadata,
                        'path': str(model_file)
                    }
                except Exception as e:
                    print(f"[Auto-load] Warning: Failed to load {model_file.name}: {e}")

            if loaded_models:
                # Store in state
                self.state.loaded_ml_models = loaded_models
                # Update status display
                self._update_loaded_models_status()
                print(f"[Auto-load] Successfully loaded {len(loaded_models)} models from {models_path}")
                # Show status message to user
                if hasattr(self, 'parent_window') and self.parent_window:
                    self.parent_window.statusBar().showMessage(f"✓ Auto-loaded {len(loaded_models)} ML models", 3000)
                    # Update dropdown states in main window
                    if hasattr(self.parent_window, '_update_classifier_dropdowns'):
                        self.parent_window._update_classifier_dropdowns()
                    # Auto-trigger peak detection if peaks have already been detected
                    if hasattr(self.parent_window, '_auto_rerun_peak_detection_if_needed'):
                        self.parent_window._auto_rerun_peak_detection_if_needed()
            else:
                print(f"[Auto-load] Failed to load any models from {models_path}")

        except Exception as e:
            print(f"[Auto-load] Error loading models: {e}")

    def _apply_threshold_to_plot(self):
        """Apply threshold from dialog to update the threshold line on main plot."""
        if not hasattr(self, 'prominence_dialog') or not self.parent_window:
            return

        try:
            values = self.prominence_dialog.get_values()
            prominence = values['prominence']
            min_dist = values['min_dist']

            # Update parent window's spinbox (this triggers plot update via valueChanged signal)
            if hasattr(self.parent_window, 'PeakPromValueSpinBox'):
                self.parent_window.PeakPromValueSpinBox.setValue(prominence)

            # Update min_dist
            if hasattr(self.parent_window, 'peak_min_dist'):
                self.parent_window.peak_min_dist = min_dist

            print(f"[analysis-options] Applied threshold to plot: {prominence:.4f}, min_dist: {min_dist}")

        except Exception as e:
            print(f"[analysis-options] Error applying threshold: {e}")

    def _detect_peaks(self):
        """Apply threshold and trigger peak detection, then refresh GMM tab."""
        if not hasattr(self, 'prominence_dialog') or not self.parent_window:
            return

        try:
            values = self.prominence_dialog.get_values()
            prominence = values['prominence']
            min_dist = values['min_dist']

            # Update parent window's spinbox
            if hasattr(self.parent_window, 'PeakPromValueSpinBox'):
                self.parent_window.PeakPromValueSpinBox.setValue(prominence)

            # Update min_dist
            if hasattr(self.parent_window, 'peak_min_dist'):
                self.parent_window.peak_min_dist = min_dist

            # Trigger peak detection
            if hasattr(self.parent_window, '_apply_peak_detection'):
                self.parent_window._apply_peak_detection()

            print(f"[analysis-options] Detected peaks with threshold: {prominence:.4f}, min_dist: {min_dist}")

            # Refresh the GMM/Breath Classification tab with new peak data
            self._refresh_breath_classification_tab()

        except Exception as e:
            print(f"[analysis-options] Error detecting peaks: {e}")

    # OLD VERSION - COMMENTED OUT - DUPLICATE OF LINE 337
    # def _refresh_breath_classification_tab(self):
    #     """Refresh the GMM/Breath Classification tab after peak detection runs."""
    #     if not hasattr(self, 'gmm_dialog'):
    #         return
    #
    #     try:
    #         from PyQt6.QtWidgets import QApplication
    #         from dialogs.gmm_clustering_dialog import GMMClusteringDialog
    #
    #         # Get the container and layout
    #         container = self.breath_classification_container
    #         layout = container.layout()
    #
    #         # Remove old GMM dialog
    #         if hasattr(self, 'gmm_dialog'):
    #             layout.removeWidget(self.gmm_dialog)
    #             self.gmm_dialog.deleteLater()
    #             self.gmm_dialog = None
    #
    #         # Create new GMM dialog with updated peak data
    #         self.gmm_dialog = GMMClusteringDialog(parent=None, main_window=self.parent_window)
    #
    #         # Hide window frame
    #         self.gmm_dialog.setWindowFlags(self.gmm_dialog.windowFlags() & ~0x00000001)
    #
    #         # Add back to container
    #         layout.addWidget(self.gmm_dialog)
    #
    #         # Process events to ensure UI updates
    #         QApplication.processEvents()
    #
    #         print("[analysis-options] Refreshed GMM/Breath Classification tab with new peak data")
    #
    #     except Exception as e:
    #         print(f"[analysis-options] Error refreshing breath classification tab: {e}")

    def accept(self):
        """Save changes when dialog is accepted (OK button clicked)."""
        # Save threshold from prominence dialog
        if hasattr(self, 'prominence_dialog') and self.parent_window:
            values = self.prominence_dialog.get_values()
            prominence = values['prominence']
            min_dist = values['min_dist']

            # Update parent window's spinbox
            if hasattr(self.parent_window, 'PeakPromValueSpinBox'):
                self.parent_window.PeakPromValueSpinBox.setValue(prominence)

            # Update min_dist if it changed
            if hasattr(self.parent_window, 'peak_min_dist'):
                self.parent_window.peak_min_dist = min_dist

            print(f"[analysis-options] Updated prominence threshold: {prominence:.4f}, min_dist: {min_dist}")

        # Save outlier metrics selection back to parent window
        if hasattr(self, 'outlier_dialog') and self.parent_window:
            selected = self.outlier_dialog.get_selected_metrics()
            self.parent_window.outlier_metrics = selected
            print(f"[analysis-options] Updated outlier detection metrics: {selected}")

        # Call parent accept to close dialog
        super().accept()

    def _update_ml_status(self):
        """Update ML model status displays."""
        # Check if models are loaded in state
        if hasattr(self.state, 'ml_model1') and self.state.ml_model1 is not None:
            self.label_model1_status_value.setText("✓ Trained")
            self.label_model1_status_value.setStyleSheet("color: #4ec9b0;")  # Green
        else:
            self.label_model1_status_value.setText("Not trained")
            self.label_model1_status_value.setStyleSheet("color: #ce9178;")  # Orange

        if hasattr(self.state, 'ml_model2') and self.state.ml_model2 is not None:
            self.label_model2_status_value.setText("✓ Trained")
            self.label_model2_status_value.setStyleSheet("color: #4ec9b0;")  # Green
        else:
            self.label_model2_status_value.setText("Not trained")
            self.label_model2_status_value.setStyleSheet("color: #ce9178;")  # Orange

        # Update manual corrections count
        if hasattr(self.state, 'peak_data') and self.state.peak_data is not None:
            manual_count = (self.state.peak_data['label_source'] == 'manual').sum()
            self.label_corrections_value.setText(str(manual_count))
        else:
            self.label_corrections_value.setText("0")

    def _browse_training_data(self):
        """Browse for training data directory and scan for files."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        import numpy as np

        # Use last selected directory if available
        initial_dir = self.last_training_data_dir if self.last_training_data_dir else str(Path.home())

        data_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Training Data Directory",
            initial_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if data_dir:
            self.last_training_data_dir = data_dir
            self.training_data_path_edit.setText(data_dir)
            # Save to settings
            if self.settings:
                self.settings.setValue("ml_training_data_path", data_dir)
            # Scan directory for files
            self._scan_training_files(data_dir)

    def _scan_training_files(self, data_dir):
        """Scan directory for .npz training files and display info."""
        import numpy as np
        from pathlib import Path

        data_path = Path(data_dir)
        npz_files = list(data_path.glob("*.npz"))

        if not npz_files:
            self.file_list_display.setPlainText("No .npz training files found in this directory.")
            return

        # Load metadata from each file
        file_info = []
        for npz_file in sorted(npz_files):
            try:
                data = np.load(npz_file, allow_pickle=True)
                source = str(data.get('source_file', 'unknown'))
                n_peaks = int(data.get('n_all_peaks', 0))
                n_breaths = int(data.get('n_breaths_only', 0))
                file_info.append({
                    'filename': npz_file.name,
                    'source': source,
                    'n_peaks': n_peaks,
                    'n_breaths': n_breaths
                })
            except Exception as e:
                print(f"Error reading {npz_file.name}: {e}")
                continue

        # Display file list
        display_lines = [f"Found {len(file_info)} training files:\n"]
        for info in file_info:
            display_lines.append(
                f"  {info['filename'][:60]:<60}  "
                f"({info['n_peaks']} peaks, {info['n_breaths']} breaths)"
            )

        self.file_list_display.setPlainText("\n".join(display_lines))

    def _retrain_all_models(self):
        """Train all models (Model 1, 2, and 3) with all 3 algorithm types."""
        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication
        from pathlib import Path
        import core.ml_training as ml_training

        # Check if training data path is set
        data_dir = self.training_data_path_edit.text()
        if not data_dir:
            self._show_error_dialog(
                "No Training Data",
                "Please select a training data directory first using the Browse button."
            )
            return

        if not Path(data_dir).exists():
            self._show_error_dialog(
                "Invalid Path",
                f"The selected directory does not exist:\n{data_dir}"
            )
            return

        # Clear previous results
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Show progress bar
        self.training_progress.setVisible(True)
        self.training_progress.setValue(0)
        self.training_progress.setMaximum(9)  # 9 total: 3 algorithms × 3 models

        # Model types to train
        model_types = [('rf', 'Random Forest'), ('xgboost', 'XGBoost'), ('mlp', 'MLP')]
        all_results = []

        try:
            # Get deduplication setting
            deduplicate = self.deduplicate_checkbox.isChecked()

            # ========= Load all data first =========
            self.training_status_label.setText("Loading training data for all models...")
            self.training_status_label.setStyleSheet("color: #4ec9b0; font-style: italic;")
            QApplication.processEvents()

            # Load Model 1 data
            X1, y1, dataset_type1, baseline1, baseline_recall1 = ml_training.load_training_data_from_directory(
                Path(data_dir),
                deduplicate=deduplicate,
                model_number=1
            )

            # Load Model 2 data
            X2, y2, dataset_type2, baseline2, baseline_recall2 = ml_training.load_training_data_from_directory(
                Path(data_dir),
                deduplicate=deduplicate,
                model_number=2
            )

            # Load Model 3 data
            X3, y3, dataset_type3, baseline3, baseline_recall3 = ml_training.load_training_data_from_directory(
                Path(data_dir),
                deduplicate=deduplicate,
                model_number=3
            )

            # ========= Train all models in parallel =========
            self.training_status_label.setText("Training all 9 models in parallel...")
            QApplication.processEvents()

            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing

            # Prepare all training jobs
            training_jobs = []

            # Model 1 jobs
            for model_type, model_name_suffix in model_types:
                training_jobs.append({
                    'model_num': 'model1',
                    'model_type': model_type,
                    'model_name': f"Model 1: Breath vs Noise ({model_name_suffix})",
                    'X': X1,
                    'y': y1,
                    'baseline_accuracy': baseline1,
                    'baseline_recall': baseline_recall1
                })

            # Model 2 jobs
            for model_type, model_name_suffix in model_types:
                training_jobs.append({
                    'model_num': 'model2',
                    'model_type': model_type,
                    'model_name': f"Model 2: Sigh vs Normal ({model_name_suffix})",
                    'X': X2,
                    'y': y2,
                    'baseline_accuracy': baseline2,
                    'baseline_recall': baseline_recall2
                })

            # Model 3 jobs
            for model_type, model_name_suffix in model_types:
                training_jobs.append({
                    'model_num': 'model3',
                    'model_type': model_type,
                    'model_name': f"Model 3: Eupnea vs Sniffing ({model_name_suffix})",
                    'X': X3,
                    'y': y3,
                    'baseline_accuracy': baseline3,
                    'baseline_recall': baseline_recall3
                })

            # Train in parallel using all available CPU cores
            n_workers = min(multiprocessing.cpu_count(), 9)  # Use all cores, max 9
            completed_count = 0

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(
                        ml_training.train_model,
                        job['X'], job['y'],
                        model_type=job['model_type'],
                        model_name=job['model_name'],
                        baseline_accuracy=job['baseline_accuracy'],
                        baseline_recall=job['baseline_recall'],
                        generate_plots=True
                    ): job
                    for job in training_jobs
                }

                # Collect results as they complete
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        all_results.append((job['model_num'], job['model_type'], result))

                        # Update progress
                        completed_count += 1
                        self.training_progress.setValue(completed_count)
                        self.training_status_label.setText(
                            f"Training models in parallel... ({completed_count}/9 complete)"
                        )
                        QApplication.processEvents()

                    except Exception as e:
                        print(f"Training failed for {job['model_name']}: {e}")
                        import traceback
                        traceback.print_exc()

            # Training complete
            self.training_status_label.setText(f"Training complete! Trained {len(all_results)} models.")
            self.training_status_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")

            # Store results for saving
            self.last_training_results = all_results

            # Display results
            self._display_training_results(all_results)

            # Show save button
            self.save_all_models_btn.setVisible(True)

            # Hide progress bar after 2 seconds
            QTimer.singleShot(2000, lambda: self.training_progress.setVisible(False))

        except Exception as e:
            import traceback
            self.training_progress.setVisible(False)
            self.training_status_label.setText("Training failed!")
            self.training_status_label.setStyleSheet("color: #ce9178; font-weight: bold;")

            error_details = f"Failed to train models:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self._show_error_dialog("Training Error", error_details)

    def _display_training_results(self, results):
        """Display training results in tabbed interface with comparison tables."""
        from PyQt6.QtWidgets import (
            QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QWidget,
            QTabWidget, QScrollArea
        )
        from PyQt6.QtGui import QPixmap

        # Clear existing comparison layout
        while self.comparison_layout.count():
            child = self.comparison_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Group results by model number
        model1_results = [(mt, r) for mn, mt, r in results if mn == 'model1']
        model2_results = [(mt, r) for mn, mt, r in results if mn == 'model2']
        model3_results = [(mt, r) for mn, mt, r in results if mn == 'model3']

        # === Create Comparison Tables Side-by-Side ===
        comparison_container = QWidget()
        comparison_container.setStyleSheet("background-color: #1e1e1e;")
        comparison_layout = QHBoxLayout(comparison_container)
        comparison_layout.setSpacing(15)

        # Model 1 comparison table
        if model1_results:
            model1_table = self._create_comparison_table("Model 1: Breath vs Noise", model1_results)
            comparison_layout.addWidget(model1_table)

        # Model 2 comparison table
        if model2_results:
            model2_table = self._create_comparison_table("Model 2: Sigh vs Normal", model2_results)
            comparison_layout.addWidget(model2_table)

        # Model 3 comparison table
        if model3_results:
            model3_table = self._create_comparison_table("Model 3: Eupnea vs Sniffing", model3_results)
            comparison_layout.addWidget(model3_table)

        self.comparison_layout.addWidget(comparison_container)

        # === Create Tabbed Results Display ===
        # Clear existing results
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create tab widget with dark theme
        results_tabs = QTabWidget()
        results_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #3e3e42;
            }
        """)

        # Model 1 tab
        if model1_results:
            model1_tab = self._create_model_results_tab(model1_results, model_number=1)
            results_tabs.addTab(model1_tab, "Model 1: Breath vs Noise")

        # Model 2 tab
        if model2_results:
            model2_tab = self._create_model_results_tab(model2_results, model_number=2)
            results_tabs.addTab(model2_tab, "Model 2: Sigh vs Normal")

        # Model 3 tab
        if model3_results:
            model3_tab = self._create_model_results_tab(model3_results, model_number=3)
            results_tabs.addTab(model3_tab, "Model 3: Eupnea vs Sniffing")

        self.results_layout.addWidget(results_tabs)

    def _create_comparison_table(self, title, model_results):
        """Create a comparison table for a single model with class distribution and detailed metrics."""
        from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout

        widget = QWidget()
        widget.setStyleSheet("background-color: #1e1e1e;")
        layout = QVBoxLayout(widget)

        # Get first result for class information
        first_result = model_results[0][1]

        # Table HTML
        html = f"<h3 style='margin-bottom: 10px; color: #4ec9b0;'>{title}</h3>"

        # Determine descriptive labels based on model type
        # Infer from title which model this is
        class_labels_map = {}
        if "Breath vs Noise" in title or "Model 1" in title:
            class_labels_map = {"0": "Noise", "1": "Breath"}
        elif "Sigh" in title or "Model 2" in title:
            class_labels_map = {"0": "Normal", "1": "Sigh"}
        elif "Eupnea" in title or "Model 3" in title:
            class_labels_map = {"0": "Eupnea", "1": "Sniffing"}

        # Class distribution section with descriptive labels
        if first_result.class_distribution:
            total = sum(first_result.class_distribution.values())
            html += "<p style='margin: 5px 0; color: #d4d4d4; font-size: 10pt;'>"

            # Create descriptive summary
            class_summaries = []
            for class_name in sorted(first_result.class_distribution.keys()):
                count = first_result.class_distribution[class_name]
                pct = 100 * count / total if total > 0 else 0
                desc_name = class_labels_map.get(str(class_name), str(class_name))
                class_summaries.append(f"<b>{desc_name}:</b> {count} ({pct:.1f}%)")

            html += " | ".join(class_summaries)
            html += f" | <b>Total:</b> {total}"
            html += "</p>"

        # Main comparison table
        html += "<table border='1' cellspacing='0' cellpadding='6' style='border-collapse: collapse; font-size: 9pt;'>"
        html += "<tr style='background-color: #2d2d30;'>"
        html += "<th>Method</th><th>Time (s)</th><th>Test Acc</th>"
        html += "<th title='Cross-validation: mean ± std across 5 folds'>CV Score</th>"

        # Add per-class recall columns with descriptive names
        for class_name in sorted(first_result.class_labels):
            desc_name = class_labels_map.get(str(class_name), str(class_name))
            html += f"<th title='Fraction of {desc_name} samples correctly identified'>{desc_name}<br/>Recall</th>"

        html += "</tr>"

        # Add threshold baseline row if available (Model 1 only)
        if first_result.baseline_accuracy is not None:
            html += "<tr style='background-color: #2a2a2a;'>"
            html += "<td><i>Threshold</i></td>"
            html += f"<td>-</td>"
            html += f"<td>{first_result.baseline_accuracy:.3f}</td>"
            html += "<td>-</td>"
            # Add baseline per-class recall if available
            if first_result.baseline_recall:
                for class_name in sorted(first_result.class_labels):
                    recall = first_result.baseline_recall.get(str(class_name), 0)
                    html += f"<td>{recall:.3f}</td>"
            else:
                for _ in first_result.class_labels:
                    html += "<td>-</td>"
            html += "</tr>"

        best_acc = max(r.test_accuracy for _, r in model_results)

        for model_type, result in model_results:
            # Highlight best accuracy
            acc_style = "color: #4ec9b0; font-weight: bold;" if result.test_accuracy == best_acc else ""

            # Training time
            time_str = f"{result.training_time_seconds:.2f}" if result.training_time_seconds else "-"

            html += f"<tr>"
            html += f"<td><b>{model_type.upper()}</b></td>"
            html += f"<td>{time_str}</td>"
            html += f"<td style='{acc_style}'>{result.test_accuracy:.3f}</td>"
            html += f"<td>{result.cv_mean:.3f} ± {result.cv_std:.3f}</td>"

            # Per-class recall (sorted by class name)
            for class_name in sorted(result.class_labels):
                recall = result.recall.get(str(class_name), 0)
                # Highlight low recall for minority classes
                recall_style = ""
                if result.class_distribution:
                    class_count = result.class_distribution.get(str(class_name), 0)
                    total = sum(result.class_distribution.values())
                    if class_count / total < 0.3 and recall < 0.7:  # Minority class with low recall
                        recall_style = "color: #ce9178;"  # Orange warning
                    elif recall > 0.9:
                        recall_style = "color: #4ec9b0;"  # Green for excellent
                html += f"<td style='{recall_style}'>{recall:.3f}</td>"

            html += "</tr>"

        html += "</table>"

        # Best model summary with improvement info
        best_result = max(model_results, key=lambda x: x[1].test_accuracy)
        html += f"<p style='margin-top: 10px; color: #4ec9b0; font-size: 10pt;'>"
        html += f"<b>✓ Best Model: {best_result[0].upper()}</b> - Test Accuracy: {best_result[1].test_accuracy:.2%}"

        # Show improvement vs baseline (Model 1 only)
        if best_result[1].baseline_accuracy is not None:
            improvement = best_result[1].test_accuracy - best_result[1].baseline_accuracy
            html += f"<br/><span style='color: #d4d4d4;'>vs Threshold ({best_result[1].baseline_accuracy:.1%}): "
            html += f"+{improvement:.1%} improvement"
            if best_result[1].error_reduction_pct is not None:
                html += f" ({best_result[1].error_reduction_pct:.1f}% fewer errors)"
            html += "</span>"
        html += "</p>"

        label = QLabel(html)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch()

        return widget

    def _create_model_results_tab(self, model_results, model_number: int):
        """Create a tab showing 3-column results for a model."""
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QScrollArea, QPushButton
        from PyQt6.QtGui import QPixmap

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #1e1e1e; border: none;")

        # Container widget
        container = QWidget()
        container.setStyleSheet("background-color: #1e1e1e;")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Grid for 3 columns
        grid = QGridLayout()
        grid.setSpacing(15)  # Wider spacing between columns

        model_dict = {mt: r for mt, r in model_results}

        for col_idx, model_type in enumerate(['rf', 'xgboost', 'mlp']):
            if model_type not in model_dict:
                continue

            result = model_dict[model_type]

            # Column widget
            col_widget = QWidget()
            col_widget.setStyleSheet("background-color: #1e1e1e;")
            col_layout = QVBoxLayout(col_widget)
            col_layout.setSpacing(5)
            col_layout.setContentsMargins(5, 5, 5, 5)

            # Header
            header = QLabel(f"<h3>{model_type.upper()}</h3>")
            header.setStyleSheet("color: #4ec9b0;")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col_layout.addWidget(header)

            # Metrics (compact)
            metrics_html = f"""
            <table cellspacing='2' style='font-size: 8pt;'>
            <tr><td><b>Test Acc:</b></td><td style='color: #4ec9b0;'>{result.test_accuracy:.2%}</td></tr>
            <tr><td><b>CV Score:</b></td><td>{result.cv_mean:.2%} ± {result.cv_std:.2%}</td></tr>
            <tr><td><b>Samples:</b></td><td>{result.n_train} / {result.n_test}</td></tr>
            </table>
            """
            col_layout.addWidget(QLabel(metrics_html))

            # Save Model button
            save_btn = QPushButton("Save Model...")
            save_btn.setMinimumWidth(100)
            save_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d30;
                    color: #cccccc;
                    border: 1px solid #3e3e42;
                    padding: 6px 12px;
                    border-radius: 3px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #094771;
                }
                QPushButton:pressed {
                    background-color: #0a5a8a;
                }
            """)
            # Use lambda with default argument to capture current result and model_number
            save_btn.clicked.connect(
                lambda checked=False, r=result, m=model_number: self._save_trained_model(r, m)
            )
            col_layout.addWidget(save_btn)

            # Feature importance (wider to fill column)
            if result.feature_importance_plot:
                col_layout.addWidget(QLabel("<b style='font-size: 8pt;'>Feature Importance</b>"))
                pixmap = QPixmap()
                pixmap.loadFromData(result.feature_importance_plot)
                img_label = QLabel()
                img_label.setPixmap(pixmap.scaledToWidth(380, Qt.TransformationMode.SmoothTransformation))
                img_label.setCursor(Qt.CursorShape.PointingHandCursor)
                img_label.mousePressEvent = lambda event, data=result.feature_importance_plot, title=f"{model_type.upper()} - Feature Importance": self._show_plot_viewer(data, title)
                col_layout.addWidget(img_label)

            # Confusion matrix (wider to fill column)
            if result.confusion_matrix_plot:
                col_layout.addWidget(QLabel("<b style='font-size: 8pt;'>Confusion Matrix</b>"))
                pixmap = QPixmap()
                pixmap.loadFromData(result.confusion_matrix_plot)
                img_label = QLabel()
                img_label.setPixmap(pixmap.scaledToWidth(360, Qt.TransformationMode.SmoothTransformation))
                img_label.setCursor(Qt.CursorShape.PointingHandCursor)
                img_label.mousePressEvent = lambda event, data=result.confusion_matrix_plot, title=f"{model_type.upper()} - Confusion Matrix": self._show_plot_viewer(data, title)
                col_layout.addWidget(img_label)

            # Learning curve (wider to fill column)
            if result.learning_curve_plot:
                col_layout.addWidget(QLabel("<b style='font-size: 8pt;'>Learning Curve</b>"))
                pixmap = QPixmap()
                pixmap.loadFromData(result.learning_curve_plot)
                img_label = QLabel()
                img_label.setPixmap(pixmap.scaledToWidth(380, Qt.TransformationMode.SmoothTransformation))
                img_label.setCursor(Qt.CursorShape.PointingHandCursor)
                img_label.mousePressEvent = lambda event, data=result.learning_curve_plot, title=f"{model_type.upper()} - Learning Curve": self._show_plot_viewer(data, title)
                col_layout.addWidget(img_label)

            col_layout.addStretch()
            grid.addWidget(col_widget, 0, col_idx)

        main_layout.addLayout(grid)
        scroll.setWidget(container)

        return scroll

    def _load_model1(self):
        """Load Model 1 (Peak Detection) from file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import pickle

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Peak Detection Model",
            "",
            "Pickle Files (*.pkl);;All Files (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Store in state
                self.state.ml_model1 = model_data['model']
                self.state.ml_model1_type = model_data.get('model_type', 'unknown')
                self.state.ml_model1_scaler = model_data.get('scaler', None)

                # Update status
                self._update_ml_status()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Model 1 loaded successfully!\nType: {self.state.ml_model1_type}"
                )
            except Exception as e:
                import traceback
                error_details = f"Failed to load model:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                self._show_error_dialog("Error", error_details)

    def _retrain_model1(self):
        """Retrain Model 1 (Breath vs Noise) using training data."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
        from PyQt6.QtCore import Qt
        from pathlib import Path
        import core.ml_training as ml_training
        from dialogs.training_results_dialog import TrainingResultsDialog

        # Ask user to select training data directory
        # Use last selected directory if available, otherwise use home directory
        initial_dir = self.last_training_data_dir if self.last_training_data_dir else str(Path.home())

        data_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Training Data Directory",
            initial_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if not data_dir:
            return

        # Remember this directory for next time
        self.last_training_data_dir = data_dir

        # Get selected model type
        model_type_text = self.comboBox_model1_type.currentText()
        model_type_map = {
            'Random Forest': 'rf',
            'XGBoost': 'xgboost',
            'MLP (Neural Network)': 'mlp'
        }
        model_type = model_type_map.get(model_type_text, 'rf')

        # Show progress dialog
        progress = QProgressDialog("Loading training data...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Training Model 1")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        try:
            # Load training data
            X, y, dataset_type = ml_training.load_training_data_from_directory(Path(data_dir))

            if dataset_type != 'all_peaks':
                QMessageBox.warning(
                    self,
                    "Wrong Dataset Type",
                    f"Model 1 requires 'all_peaks' dataset, but found '{dataset_type}'.\n"
                    "Please select a directory with breath vs noise training data."
                )
                progress.close()
                return

            progress.setLabelText(f"Training {model_type_text} model...")
            progress.setValue(0)

            # Train model
            result = ml_training.train_model(
                X, y,
                model_type=model_type,
                model_name="Model 1: Breath vs Noise",
                test_size=0.2,
                random_state=42,
                generate_plots=True
            )

            progress.close()

            # Show results dialog
            results_dialog = TrainingResultsDialog(result, parent=self)

            # Connect save button
            results_dialog.save_model_btn.clicked.connect(
                lambda: self._save_trained_model(result, model_number=1)
            )

            if results_dialog.exec() == QMessageBox.DialogCode.Accepted:
                # Ask if user wants to use this model
                reply = QMessageBox.question(
                    self,
                    "Use Trained Model?",
                    f"Would you like to use this model for breath detection?\n\n"
                    f"Test Accuracy: {result.test_accuracy:.1%}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Store in state
                    self.state.ml_model1 = result.model
                    self.state.ml_model1_type = result.model_type
                    self.state.ml_model1_metrics = {
                        'test_accuracy': result.test_accuracy,
                        'cv_mean': result.cv_mean,
                        'feature_importance': result.feature_importance
                    }

                    # Update status
                    self._update_ml_status()

                    QMessageBox.information(
                        self,
                        "Success",
                        "Model 1 is now active and will be used for breath detection."
                    )

        except Exception as e:
            progress.close()
            import traceback
            error_details = f"Failed to train model:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self._show_error_dialog("Training Error", error_details)

    def _load_model2(self):
        """Load Model 2 (Breath Classification) from file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import pickle

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Breath Classification Model",
            "",
            "Pickle Files (*.pkl);;All Files (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Store in state
                self.state.ml_model2 = model_data['model']
                self.state.ml_model2_type = model_data.get('model_type', 'unknown')
                self.state.ml_model2_scaler = model_data.get('scaler', None)

                # Update status
                self._update_ml_status()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Model 2 loaded successfully!\nType: {self.state.ml_model2_type}"
                )
            except Exception as e:
                import traceback
                error_details = f"Failed to load model:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                self._show_error_dialog("Error", error_details)

    def _retrain_model2(self):
        """Retrain Model 2 (Breath Classification: Eupnea/Sniff/Sigh) using training data."""
        from PyQt6.QtWidgets import QMessageBox

        # NOTE: This requires breaths_only dataset with breath type labels
        # The export format needs to be updated to include breath type labels from GMM
        QMessageBox.information(
            self,
            "Coming Soon",
            "Model 2 training requires breath-type labeled data.\n\n"
            "This feature will be available once the export format is updated to include:\n"
            "- Eupnea/Sniffing labels from GMM clustering\n"
            "- Sigh detection labels\n\n"
            "For now, Model 1 (Breath vs Noise) is fully functional."
        )

    def _export_training_data(self):
        """Export current training data for ML model training."""
        if self.parent_window and hasattr(self.parent_window, '_export_ml_training_data'):
            self.parent_window._export_ml_training_data()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Info",
                "Please use File → Export ML Training Data from the main window."
            )

    def _import_training_data(self):
        """Import training data from NPZ file."""
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Coming Soon",
            "Training data import will be implemented in the next update.\n\n"
            "This will allow you to load previously exported training datasets."
        )

    def _save_trained_model(self, result, model_number: int):
        """Save trained model to file with metadata."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import core.ml_training as ml_training

        # Suggest filename
        model_name = f"model{model_number}_{result.model_type}_{result.test_accuracy:.0%}.pkl"
        default_path = str(Path.home() / model_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save Model {model_number}",
            default_path,
            "Pickle Files (*.pkl);;All Files (*.*)"
        )

        if file_path:
            try:
                # Package metadata for later use
                metadata = {
                    'model_type': result.model_type,
                    'model_number': model_number,
                    'feature_names': result.feature_names,
                    'test_accuracy': result.test_accuracy,
                    'cv_mean': result.cv_mean,
                    'cv_std': result.cv_std,
                    'n_features': result.n_features,
                    'class_labels': result.class_labels
                }

                ml_training.save_model(result.model, Path(file_path), metadata=metadata)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Model saved successfully to:\n{file_path}\n\n"
                    f"Model: {result.model_name}\n"
                    f"Type: {result.model_type.upper()}\n"
                    f"Accuracy: {result.test_accuracy:.1%}"
                )
            except Exception as e:
                self._show_error_dialog("Save Error", f"Failed to save model:\n\n{str(e)}")

    def _load_ml_models(self):
        """Load ML models from a directory."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import core.ml_prediction as ml_prediction
        import pandas as pd
        import numpy as np

        # Ask user to select directory containing models
        # Use last used directory if available, otherwise suggest default
        if self.last_models_dir and Path(self.last_models_dir).exists():
            suggested_dir = self.last_models_dir
        else:
            suggested_dir = str(Path.home() / "PhysioMetrics_Models")

        models_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory Containing ML Models",
            suggested_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if not models_dir:
            return

        models_path = Path(models_dir)

        # Save this directory for next time
        self.last_models_dir = str(models_path)
        if self.settings:
            self.settings.setValue("ml_models_path", self.last_models_dir)

        try:
            # Look for model files
            model_files = list(models_path.glob("model*.pkl"))

            if not model_files:
                QMessageBox.warning(
                    self,
                    "No Models Found",
                    f"No model files (model*.pkl) found in:\n{models_path}"
                )
                return

            # Load all models
            loaded_models = {}
            for model_file in model_files:
                try:
                    model, metadata = ml_prediction.load_model(model_file)

                    # Extract key from filename (e.g., "model1_xgboost.pkl" -> "model1_xgboost")
                    model_key = model_file.stem

                    loaded_models[model_key] = {
                        'model': model,
                        'metadata': metadata,
                        'path': str(model_file)
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {model_file.name}: {e}")

            if not loaded_models:
                QMessageBox.warning(
                    self,
                    "Load Failed",
                    "Failed to load any models from the selected directory."
                )
                return

            # Store in state
            self.state.loaded_ml_models = loaded_models

            # Update status display
            self._update_loaded_models_status()

            # Reconstruct training results from loaded model metadata
            # Group models by model number and algorithm
            training_results = []
            for model_key, model_data in loaded_models.items():
                metadata = model_data['metadata']

                # Parse model_key to get model number and algorithm
                # e.g., "model1_xgboost" -> model_num=1, algorithm="xgboost"
                parts = model_key.split('_')
                if len(parts) >= 2:
                    model_name_part = parts[0]  # "model1"
                    algorithm = '_'.join(parts[1:])  # "xgboost" (or "mlp", "rf")

                    # Reconstruct TrainingResult from metadata (including plot data)
                    from core.ml_training import TrainingResult
                    result = TrainingResult(
                        model=None,  # Don't include actual model in results display
                        model_type=metadata.get('model_type', algorithm),
                        model_name=metadata.get('model_name', f'Model {model_name_part[-1]}'),
                        feature_names=metadata.get('feature_names', []),
                        feature_importance=pd.DataFrame(),  # Empty, no plot available
                        test_accuracy=metadata.get('test_accuracy', 0),
                        train_accuracy=metadata.get('train_accuracy', 0),
                        cv_mean=metadata.get('cv_mean', 0),
                        cv_std=metadata.get('cv_std', 0),
                        n_train=metadata.get('n_train', 0),
                        n_test=metadata.get('n_test', 0),
                        n_features=metadata.get('n_features', 0),
                        class_labels=metadata.get('class_labels', []),
                        precision=metadata.get('precision', {}),
                        recall=metadata.get('recall', {}),
                        f1_score=metadata.get('f1_score', {}),
                        confusion_matrix=metadata.get('confusion_matrix', np.array([])),
                        feature_importance_plot=metadata.get('feature_importance_plot'),
                        confusion_matrix_plot=metadata.get('confusion_matrix_plot'),
                        learning_curve_plot=metadata.get('learning_curve_plot'),
                        baseline_accuracy=metadata.get('baseline_accuracy'),
                        accuracy_improvement=metadata.get('accuracy_improvement'),
                        error_reduction_pct=metadata.get('error_reduction_pct'),
                        baseline_recall=metadata.get('baseline_recall'),
                        class_distribution=metadata.get('class_distribution'),
                        training_time_seconds=metadata.get('training_time_seconds'),
                        is_converged=metadata.get('is_converged'),
                        needs_more_data=metadata.get('needs_more_data')
                    )

                    training_results.append((model_name_part, algorithm, result))

            # Display results if we have any
            if training_results:
                self._display_training_results(training_results)

            # Update dropdown states in main window
            if hasattr(self, 'parent_window') and self.parent_window:
                if hasattr(self.parent_window, '_update_classifier_dropdowns'):
                    self.parent_window._update_classifier_dropdowns()

                # Auto-trigger peak detection if peaks have already been detected
                # This updates predictions with the newly loaded models
                if hasattr(self.parent_window, '_auto_rerun_peak_detection_if_needed'):
                    self.parent_window._auto_rerun_peak_detection_if_needed()

            # Show success message
            QMessageBox.information(
                self,
                "Models Loaded",
                f"Successfully loaded {len(loaded_models)} model(s):\n\n" +
                ml_prediction.get_model_summary(loaded_models) +
                "\n\nModel statistics, comparison tables, and training plots are now displayed below."
            )

        except Exception as e:
            import traceback
            self._show_error_dialog(
                "Load Error",
                f"Failed to load models:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )

    def _update_loaded_models_status(self):
        """Update the loaded models status label."""
        import core.ml_prediction as ml_prediction

        if not self.state.loaded_ml_models:
            self.loaded_models_status.setText("No models loaded")
            self.loaded_models_status.setStyleSheet("color: #888; font-style: italic;")
        else:
            summary = ml_prediction.get_model_summary(self.state.loaded_ml_models)
            self.loaded_models_status.setText(
                f"<span style='color: #4ec9b0;'><b>✓ Loaded:</b></span><br/>" +
                summary.replace('\n', '<br/>')
            )
            self.loaded_models_status.setStyleSheet("color: #d4d4d4;")

    def _save_all_models(self):
        """Save all 9 trained models (3 models × 3 algorithms) to a directory for comparison analysis."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import core.ml_training as ml_training
        from datetime import datetime

        if not hasattr(self, 'last_training_results') or not self.last_training_results:
            QMessageBox.warning(
                self,
                "No Models",
                "No trained models available to save. Please train models first."
            )
            return

        # Ask user to select a directory
        # Use last used directory if available, otherwise suggest default
        if self.last_models_dir and Path(self.last_models_dir).exists():
            suggested_dir = self.last_models_dir
        else:
            suggested_dir = str(Path.home() / "PhysioMetrics_Models")

        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save All Models",
            suggested_dir,
            QFileDialog.Option.ShowDirsOnly
        )

        if not save_dir:
            return

        save_dir_path = Path(save_dir)

        # Save this directory for next time (before potential backup subdirectory creation)
        self.last_models_dir = str(save_dir_path)
        if self.settings:
            self.settings.setValue("ml_models_path", self.last_models_dir)

        # Check if models already exist in this directory
        existing_models = list(save_dir_path.glob("model*.pkl"))
        if existing_models:
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Models?",
                f"Found {len(existing_models)} existing model(s) in:\n{save_dir_path}\n\n"
                "Options:\n"
                "  • Yes: Overwrite existing models (recommended for regular updates)\n"
                "  • No: Create timestamped backup folder\n"
                "  • Cancel: Don't save",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Cancel:
                return
            elif reply == QMessageBox.StandardButton.No:
                # Create timestamped backup subdirectory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir_path = save_dir_path / f"backup_{timestamp}"
                save_dir_path.mkdir(parents=True, exist_ok=True)

        # Use the selected directory directly (no timestamp unless user chose backup)

        try:
            saved_models = []
            model_info_list = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save ALL models (all 9: 3 model numbers × 3 algorithms)
            for model_name, model_type, result in self.last_training_results:
                # Extract model number from model_name string
                if 'model1' in model_name:
                    model_num = 1
                elif 'model2' in model_name:
                    model_num = 2
                elif 'model3' in model_name:
                    model_num = 3
                else:
                    continue

                # Create filename (simpler - no accuracy in filename, easier to use programmatically)
                filename = f"model{model_num}_{model_type}.pkl"
                file_path = save_dir_path / filename

                # Package metadata (include ALL TrainingResult fields for reconstruction)
                metadata = {
                    'model_type': result.model_type,
                    'model_number': model_num,
                    'model_name': result.model_name,
                    'feature_names': result.feature_names,
                    'test_accuracy': result.test_accuracy,
                    'train_accuracy': result.train_accuracy,
                    'cv_mean': result.cv_mean,
                    'cv_std': result.cv_std,
                    'n_train': result.n_train,
                    'n_test': result.n_test,
                    'n_features': result.n_features,
                    'class_labels': result.class_labels,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'confusion_matrix': result.confusion_matrix,
                    'baseline_accuracy': result.baseline_accuracy,
                    'accuracy_improvement': result.accuracy_improvement,
                    'error_reduction_pct': result.error_reduction_pct,
                    'baseline_recall': result.baseline_recall,
                    'class_distribution': result.class_distribution,
                    'training_time_seconds': result.training_time_seconds,
                    'is_converged': result.is_converged,
                    'needs_more_data': result.needs_more_data,
                    'saved_timestamp': timestamp,
                    # Plot data (PNG bytes)
                    'feature_importance_plot': result.feature_importance_plot,
                    'confusion_matrix_plot': result.confusion_matrix_plot,
                    'learning_curve_plot': result.learning_curve_plot
                }

                # Save model
                ml_training.save_model(result.model, file_path, metadata=metadata)
                saved_models.append(filename)
                model_info_list.append(
                    f"  - {filename}: {result.model_name.split(':')[1].strip()} - "
                    f"{result.test_accuracy:.1%} accuracy (CV: {result.cv_mean:.1%})"
                )

            # Create README file with model information
            readme_path = save_dir_path / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(f"PhysioMetrics ML Models - Complete Set\n")
                f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 70 + "\n\n")
                f.write(f"Total Models: {len(saved_models)}\n\n")
                f.write("All Models:\n")
                for model_info in model_info_list:
                    f.write(f"{model_info}\n")
                f.write("\n")
                f.write("Recommended Usage:\n")
                f.write("  - Primary: Use XGBoost models (typically highest accuracy)\n")
                f.write("  - Active Learning: Compare all 3 algorithms to find disagreements\n")
                f.write("  - Edge Cases: Review samples where models disagree\n")
                f.write("\n")
                f.write("Model Architecture:\n")
                f.write("  - Model 1: Breath vs Noise detection (binary classification)\n")
                f.write("  - Model 2: Sigh vs Normal breath classification (binary)\n")
                f.write("  - Model 3: Eupnea vs Sniffing classification (binary)\n")
                f.write("\n")
                f.write("Algorithms:\n")
                f.write("  - RF: Random Forest (ensemble of decision trees)\n")
                f.write("  - XGBoost: Gradient Boosting (usually best performance)\n")
                f.write("  - MLP: Multi-Layer Perceptron (neural network)\n")
                f.write("\n")
                f.write("Future: Consider deep learning models trained on raw waveforms\n")
                f.write("        if feature-based models plateau in accuracy.\n")

            QMessageBox.information(
                self,
                "Success",
                f"Saved all {len(saved_models)} models to:\n{save_dir_path}\n\n"
                f"Recommendation:\n"
                f"  • Use XGBoost models for production\n"
                f"  • Keep all 9 for active learning edge case review"
            )

        except Exception as e:
            import traceback
            self._show_error_dialog(
                "Save Error",
                f"Failed to save models:\n\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )

    def _show_plot_viewer(self, plot_bytes: bytes, title: str):
        """Show a plot in a full-size viewer dialog with matplotlib navigation toolbar."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QHBoxLayout
        from PyQt6.QtCore import Qt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure
        from PIL import Image
        import io

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(1200, 800)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QPushButton {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QToolBar {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                spacing: 3px;
            }
            QToolButton {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 4px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #3e3e42;
            }
            QToolButton:pressed {
                background-color: #094771;
            }
        """)

        layout = QVBoxLayout(dialog)

        # Load image from bytes
        img = Image.open(io.BytesIO(plot_bytes))

        # Create matplotlib figure and display image
        fig = Figure(figsize=(12, 8), facecolor='#1e1e1e')
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis('off')  # Hide axes for image display
        fig.tight_layout()

        # Create canvas and toolbar
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background-color: #1e1e1e;")
        toolbar = NavigationToolbar(canvas, dialog)

        # Add widgets to layout
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        dialog.exec()

    def _show_error_dialog(self, title: str, message: str):
        """Show error dialog with copyable text."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QLabel
        from PyQt6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)

        # Error icon and title
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("color: #f48771; font-size: 14px; padding: 10px;")
        layout.addWidget(title_label)

        # Copyable text area
        text_edit = QTextEdit()
        text_edit.setPlainText(message)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(text_edit)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 8px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
            }
        """)
        layout.addWidget(close_btn)

        dialog.exec()
