#!/usr/bin/env python3
"""
Path: MapGenerator/gui/widgets/parameter_slider.py
__init__.py existiert in "widgets"

Wiederverwendbare ParameterSlider-Komponente
Eliminiert Code-Duplikation zwischen allen Tabs
"""

from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QSlider, QSpinBox, QDoubleSpinBox
from PyQt5.QtCore import Qt, pyqtSignal


class ParameterSlider(QWidget):
    """
    Funktionsweise: Einheitlicher Parameter-Slider für alle Tabs
    - Kombiniert Label, Slider und SpinBox in einem Widget
    - Unterstützt Integer und Decimal-Werte
    - Automatische Synchronisation zwischen Slider und SpinBox
    - Emittiert valueChanged Signal bei Änderungen
    """

    # Signal wird ausgesendet wenn sich der Wert ändert
    valueChanged = pyqtSignal()

    def __init__(self, label_text, min_val, max_val, default_val, decimals=0, suffix=""):
        """
        Args:
            label_text (str): Anzeigename des Parameters
            min_val: Minimaler Wert
            max_val: Maximaler Wert
            default_val: Startwert
            decimals (int): Anzahl Dezimalstellen (0 = Integer)
            suffix (str): Einheit/Suffix für SpinBox (z.B. "°C", "%")
        """
        super().__init__()
        self.decimals = decimals
        self.suffix = suffix
        self.init_ui(label_text, min_val, max_val, default_val)

    def init_ui(self, label_text, min_val, max_val, default_val):
        """Erstellt das UI-Layout mit Label, Slider und SpinBox"""
        layout = QGridLayout()

        # Label
        self.label = QLabel(label_text)
        layout.addWidget(self.label, 0, 0, 1, 2)

        # Slider - arbeitet immer mit Integer-Werten
        self.slider = QSlider(Qt.Horizontal)
        if self.decimals > 0:
            # Für Dezimalwerte: Slider-Werte mit 10^decimals multiplizieren
            self.slider.setMinimum(int(min_val * (10 ** self.decimals)))
            self.slider.setMaximum(int(max_val * (10 ** self.decimals)))
            self.slider.setValue(int(default_val * (10 ** self.decimals)))
        else:
            self.slider.setMinimum(min_val)
            self.slider.setMaximum(max_val)
            self.slider.setValue(default_val)

        layout.addWidget(self.slider, 1, 0)

        # SpinBox - Integer oder Double je nach decimals
        if self.decimals > 0:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
            self.spinbox.setValue(default_val)
            self.spinbox.setDecimals(self.decimals)
            self.spinbox.setSingleStep(0.1 if self.decimals == 1 else 0.01)
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
            self.spinbox.setValue(default_val)

        # Suffix hinzufügen wenn angegeben
        if self.suffix:
            self.spinbox.setSuffix(f" {self.suffix}")

        self.spinbox.setMaximumWidth(100)
        layout.addWidget(self.spinbox, 1, 1)

        # Signal-Verbindungen
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)

        self.setLayout(layout)

    def on_slider_changed(self, value):
        """
        Funktionsweise: Wird aufgerufen wenn Slider bewegt wird
        - Konvertiert Slider-Wert zu SpinBox-Wert
        - Emittiert valueChanged Signal
        """
        if self.decimals > 0:
            spinbox_value = value / (10 ** self.decimals)
            self.spinbox.blockSignals(True)  # Verhindert Endlos-Loop
            self.spinbox.setValue(spinbox_value)
            self.spinbox.blockSignals(False)
        else:
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(value)
            self.spinbox.blockSignals(False)

        self.valueChanged.emit()

    def on_spinbox_changed(self, value):
        """
        Funktionsweise: Wird aufgerufen wenn SpinBox geändert wird
        - Konvertiert SpinBox-Wert zu Slider-Wert
        - Emittiert valueChanged Signal
        """
        if self.decimals > 0:
            slider_value = int(value * (10 ** self.decimals))
            self.slider.blockSignals(True)  # Verhindert Endlos-Loop
            self.slider.setValue(slider_value)
            self.slider.blockSignals(False)
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(value)
            self.slider.blockSignals(False)

        self.valueChanged.emit()

    def get_value(self):
        """
        Funktionsweise: Gibt den aktuellen Wert zurück
        Returns:
            float oder int: Aktueller Parameter-Wert
        """
        if self.decimals > 0:
            return self.spinbox.value()
        return self.slider.value()

    def set_value(self, value):
        """
        Funktionsweise: Setzt einen neuen Wert (programmatisch)
        - Aktualisiert sowohl Slider als auch SpinBox
        - Emittiert KEIN valueChanged Signal (blockiert)
        """
        # Blockiere Signale während der Aktualisierung
        self.slider.blockSignals(True)
        self.spinbox.blockSignals(True)

        if self.decimals > 0:
            self.slider.setValue(int(value * (10 ** self.decimals)))
            self.spinbox.setValue(value)
        else:
            self.slider.setValue(value)
            self.spinbox.setValue(value)

        # Signale wieder aktivieren
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)

    def set_enabled(self, enabled):
        """Aktiviert/deaktiviert den gesamten Parameter-Slider"""
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)
        self.label.setEnabled(enabled)