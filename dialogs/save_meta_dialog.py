"""
Save Metadata Dialog for PlethApp.

This dialog provides a structured interface for collecting metadata about analyzed
breath data before saving, including experimental details like mouse strain, virus,
location, stimulation parameters, and animal information.
"""

import re
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QLabel,
    QCheckBox, QDialogButtonBox, QCompleter
)
from PyQt6.QtCore import Qt as QtCore_Qt


class SaveMetaDialog(QDialog):
    def __init__(self, abf_name: str, channel: str, parent=None, auto_stim: str = "", history: dict = None):
        super().__init__(parent)
        self.setWindowTitle("Save analyzed data — name builder")

        self._abf_name = abf_name
        self._channel = channel
        self._history = history or {}

        lay = QFormLayout(self)

        # Mouse Strain with autocomplete
        self.le_strain = QLineEdit(self)
        self.le_strain.setPlaceholderText("e.g., VgatCre")
        if self._history.get('strain'):
            completer = QCompleter(self._history['strain'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_strain.setCompleter(completer)
        lay.addRow("Mouse Strain:", self.le_strain)

        # Virus with autocomplete
        self.le_virus = QLineEdit(self)
        self.le_virus.setPlaceholderText("e.g., ConFoff-ChR2")
        if self._history.get('virus'):
            completer = QCompleter(self._history['virus'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_virus.setCompleter(completer)
        lay.addRow("Virus:", self.le_virus)

        # Location with autocomplete
        self.le_location = QLineEdit(self)
        self.le_location.setPlaceholderText("e.g., preBotC or RTN")
        if self._history.get('location'):
            completer = QCompleter(self._history['location'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_location.setCompleter(completer)
        lay.addRow("Location:", self.le_location)

        # Stimulation type (can be auto-populated) with autocomplete
        self.le_stim = QLineEdit(self)
        self.le_stim.setPlaceholderText("e.g., 20Hz10s15ms or 15msPulse")
        if auto_stim:
            self.le_stim.setText(auto_stim)
        if self._history.get('stim'):
            completer = QCompleter(self._history['stim'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_stim.setCompleter(completer)
        lay.addRow("Stimulation type:", self.le_stim)

        # Laser power with autocomplete
        self.le_power = QLineEdit(self)
        self.le_power.setPlaceholderText("e.g., 8mW")
        if self._history.get('power'):
            completer = QCompleter(self._history['power'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_power.setCompleter(completer)
        lay.addRow("Laser power:", self.le_power)

        self.cb_sex = QComboBox(self)
        self.cb_sex.addItems(["", "M", "F", "Unknown"])
        lay.addRow("Sex:", self.cb_sex)

        # Animal ID with autocomplete
        self.le_animal = QLineEdit(self)
        self.le_animal.setPlaceholderText("e.g., 25121004")
        if self._history.get('animal'):
            completer = QCompleter(self._history['animal'], self)
            completer.setCaseSensitivity(QtCore_Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.InlineCompletion)
            self.le_animal.setCompleter(completer)
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
        self.le_location.textChanged.connect(self._update_preview)
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
        location = self.le_location.text().strip()

        stim   = self.le_stim.text().strip()
        power  = self.le_power.text().strip()
        sex    = self.cb_sex.currentText().strip()
        animal = self.le_animal.text().strip()
        abf    = self._abf_name
        ch     = self._channel

        # Sanitize for filename
        strain_s = self._san(strain)
        virus_s  = self._san(virus)
        location_s = self._san(location)
        stim_s   = self._san(stim)
        power_s  = self._san(power)
        sex_s    = self._san(sex)
        animal_s = self._san(animal)
        abf_s    = self._san(abf)
        ch_s     = self._san(ch)

        # STANDARD ORDER:
        # Strain_Virus_Location_Sex_Animal_Stim_Power_ABF_Channel
        parts = [p for p in (strain_s, virus_s, location_s, sex_s, animal_s, stim_s, power_s, abf_s, ch_s) if p]
        preview = "_".join(parts) if parts else "analysis"
        self.lbl_preview.setText(preview)

    def values(self) -> dict:
        return {
            "strain": self.le_strain.text().strip(),
            "virus":  self.le_virus.text().strip(),
            "location": self.le_location.text().strip(),
            "stim":   self.le_stim.text().strip(),
            "power":  self.le_power.text().strip(),
            "sex":    self.cb_sex.currentText().strip(),
            "animal": self.le_animal.text().strip(),
            "abf":    self._abf_name,
            "chan":   self._channel,
            "preview": self.lbl_preview.text().strip(),
            "choose_dir": bool(self.cb_choose_dir.isChecked()),
        }
