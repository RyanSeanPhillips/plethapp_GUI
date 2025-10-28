"""
First-launch dialog for PlethApp.

Shows welcome message and telemetry opt-in/opt-out on first run.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QTextBrowser, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from version_info import VERSION_STRING


class FirstLaunchDialog(QDialog):
    """
    Welcome dialog shown on first launch.

    Explains telemetry and lets user opt out.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to PlethApp")
        self.setModal(True)
        self.resize(600, 500)

        # Default values (opt-out model - checkboxes pre-checked)
        self.telemetry_enabled = True
        self.crash_reports_enabled = True

        self._setup_ui()
        self._apply_dark_theme()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header = QLabel(f"<h2 style='color: #2a7fff;'>Welcome to PlethApp {VERSION_STRING}</h2>")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Welcome message
        welcome_text = QLabel(
            "<p style='font-size: 11pt;'>"
            "Thank you for using PlethApp! This tool was created to help researchers "
            "analyze respiratory signals with advanced breath detection and pattern analysis."
            "</p>"
        )
        welcome_text.setWordWrap(True)
        layout.addWidget(welcome_text)

        # Telemetry explanation group
        telemetry_group = QGroupBox("Help Improve PlethApp")
        telemetry_layout = QVBoxLayout()

        # Intro text
        intro_label = QLabel(
            "<p>Share anonymous usage statistics to help improve PlethApp. "
            "Your data is completely anonymous and helps prioritize features and fix bugs.</p>"
        )
        intro_label.setWordWrap(True)
        telemetry_layout.addWidget(intro_label)

        # Telemetry checkbox
        self.telemetry_checkbox = QCheckBox("Share anonymous usage data")
        self.telemetry_checkbox.setChecked(True)  # Opt-out model
        self.telemetry_checkbox.setStyleSheet("font-size: 11pt; font-weight: bold;")
        telemetry_layout.addWidget(self.telemetry_checkbox)

        # What's collected
        collected_label = QLabel(
            "<ul style='margin-top: 5px; margin-left: 20px;'>"
            "<li>Number of files analyzed (no file names or paths)</li>"
            "<li>Features used (GMM clustering, manual editing, etc.)</li>"
            "<li>App version, OS, and Python version</li>"
            "<li>Anonymous user ID (random UUID)</li>"
            "</ul>"
        )
        telemetry_layout.addWidget(collected_label)

        # Crash reports checkbox
        self.crash_reports_checkbox = QCheckBox("Send crash reports")
        self.crash_reports_checkbox.setChecked(True)  # Opt-out model
        self.crash_reports_checkbox.setStyleSheet("font-size: 11pt; font-weight: bold;")
        telemetry_layout.addWidget(self.crash_reports_checkbox)

        crash_label = QLabel(
            "<ul style='margin-top: 5px; margin-left: 20px;'>"
            "<li>Error stack traces to help fix bugs</li>"
            "<li>No personal or experimental data</li>"
            "</ul>"
        )
        telemetry_layout.addWidget(crash_label)

        # What's NOT collected
        not_collected_label = QLabel(
            "<p style='color: #FFD700; font-weight: bold; margin-top: 10px;'>What we NEVER collect:</p>"
            "<ul style='margin-left: 20px;'>"
            "<li>File names, paths, or directory structure</li>"
            "<li>Animal metadata (strain, virus, injection site)</li>"
            "<li>Actual breathing data (frequencies, amplitudes, metrics)</li>"
            "<li>Your name, email, or institution</li>"
            "</ul>"
        )
        not_collected_label.setWordWrap(True)
        telemetry_layout.addWidget(not_collected_label)

        # Learn more button
        learn_more_btn = QPushButton("Learn More...")
        learn_more_btn.clicked.connect(self._show_details)
        learn_more_btn.setMaximumWidth(150)
        telemetry_layout.addWidget(learn_more_btn)

        telemetry_group.setLayout(telemetry_layout)
        layout.addWidget(telemetry_group)

        # Note about changing preferences
        note_label = QLabel(
            "<p style='font-size: 9pt; color: #888;'>"
            "You can change these preferences anytime in Help → About."
            "</p>"
        )
        note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note_label)

        # Spacer
        layout.addStretch()

        # Continue button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        continue_btn = QPushButton("Continue")
        continue_btn.setDefault(True)
        continue_btn.clicked.connect(self._on_continue)
        continue_btn.setMinimumWidth(120)
        continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a7fff;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4090ff;
            }
        """)
        button_layout.addWidget(continue_btn)

        layout.addLayout(button_layout)

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QGroupBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #2a7fff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2a2a2a;
                border: 2px solid #555;
            }
            QCheckBox::indicator:checked {
                background-color: #2a7fff;
                border: 2px solid #2a7fff;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)

    def _on_continue(self):
        """Handle Continue button click."""
        # Save checkbox states
        self.telemetry_enabled = self.telemetry_checkbox.isChecked()
        self.crash_reports_enabled = self.crash_reports_checkbox.isChecked()

        # Close dialog with accept
        self.accept()

    def _show_details(self):
        """Show detailed information about telemetry."""
        details_dialog = QDialog(self)
        details_dialog.setWindowTitle("Telemetry Details")
        details_dialog.resize(500, 400)

        layout = QVBoxLayout(details_dialog)

        # Text browser with details
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml("""
            <h3 style="color: #2a7fff;">What Data is Collected?</h3>

            <h4>Usage Data</h4>
            <p>When you use PlethApp, we collect anonymous usage statistics:</p>
            <ul>
                <li><b>File types:</b> Whether you load ABF, SMRX, or EDF files (not file names)</li>
                <li><b>Features used:</b> Which tools you use (GMM, manual editing, spectral analysis)</li>
                <li><b>Export types:</b> What you export (PDF, CSV, NPZ)</li>
                <li><b>Session duration:</b> How long you use the app</li>
                <li><b>System info:</b> OS, Python version, PlethApp version</li>
            </ul>

            <h4>Crash Reports</h4>
            <p>If the app crashes, we collect:</p>
            <ul>
                <li><b>Error messages:</b> What went wrong</li>
                <li><b>Stack traces:</b> Where the error occurred in the code</li>
                <li><b>No data:</b> Your files and data are NEVER included</li>
            </ul>

            <h4>Anonymous User ID</h4>
            <p>A random UUID (like "a3f2e8c9-4b7d-...") is generated once and stored on your computer.
            This lets us count unique users without knowing who you are.</p>

            <h4>Why Opt-Out by Default?</h4>
            <p>PlethApp uses an "opt-out" model (checkboxes pre-checked) because:</p>
            <ul>
                <li>This is academic research software (not commercial spyware)</li>
                <li>All data is completely anonymous</li>
                <li>Usage statistics help prioritize bug fixes and features</li>
                <li>You can easily disable it anytime</li>
            </ul>

            <p style="color: #FFD700; font-weight: bold;">You have complete control.</p>
            <p>You can disable telemetry at any time in Help → About.</p>

            <h4>Precedents</h4>
            <p>Many trusted tools use opt-out telemetry:</p>
            <ul>
                <li>VS Code (Microsoft)</li>
                <li>Anaconda Navigator</li>
                <li>JupyterLab extensions</li>
            </ul>
        """)
        layout.addWidget(browser)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(details_dialog.close)
        layout.addWidget(close_btn)

        # Apply dark theme
        details_dialog.setStyleSheet(self.styleSheet())

        details_dialog.exec()

    def get_preferences(self):
        """
        Get user's telemetry preferences.

        Returns:
            tuple: (telemetry_enabled, crash_reports_enabled)
        """
        return (self.telemetry_enabled, self.crash_reports_enabled)
