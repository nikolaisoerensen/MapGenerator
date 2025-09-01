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

    def __init__(self, label_text, min_val, max_val, default_val, decimals=0, suffix="", step=1):
        """
        Args:
            label_text (str): Anzeigename des Parameters
            min_val: Minimaler Wert
            max_val: Maximaler Wert
            default_val: Startwert
            decimals (int): Anzahl Dezimalstellen (0 = Integer)
            suffix (str): Einheit/Suffix für SpinBox (z.B. "°C", "%")
            step: Schrittweite (None = automatisch)
        """
        super().__init__()
        self.decimals = decimals
        self.step = step
        self.suffix = suffix

        if step in (None, 1):
            if decimals > 0:
                self.step = 0.1 if decimals == 1 else 0.01  # Decimal default
            else:
                self.step = 1  # Integer default
        else:
            self.step = step
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
            self.slider_scale = 10 ** self.decimals
            self.slider.setMinimum(int(min_val * (10 ** self.decimals)))
            self.slider.setMaximum(int(max_val * (10 ** self.decimals)))
            self.slider.setValue(int(default_val * (10 ** self.decimals)))
            # Step für Decimal-Slider
            step_scaled = int(self.step * self.slider_scale)
            if step_scaled > 1:
                self.slider.setSingleStep(step_scaled)
                self.slider.setPageStep(step_scaled * 5)
        else:
            # Integer-Slider mit Step
            self.slider_scale = 1

            # WICHTIG: Min/Max an Steps anpassen
            adjusted_min = self._round_to_step(min_val, self.step)
            adjusted_max = self._round_to_step(max_val, self.step)
            adjusted_default = self._round_to_step(default_val, self.step)

            # Slider arbeitet in Step-Einheiten
            self.slider.setMinimum(int(adjusted_min // self.step))
            self.slider.setMaximum(int(adjusted_max // self.step))
            self.slider.setValue(int(adjusted_default // self.step))

            self.slider.setSingleStep(1)  # 1 Step im Slider = 1 * self.step im Wert
            self.slider.setPageStep(5)

        layout.addWidget(self.slider, 1, 0)

        # SpinBox - Integer oder Double je nach decimals
        if self.decimals > 0:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
            self.spinbox.setValue(default_val)
            self.spinbox.setDecimals(self.decimals)
            self.spinbox.setSingleStep(self.step)
        else:
            self.spinbox = QSpinBox()
            # Min/Max an Steps anpassen
            adjusted_min = self._round_to_step(min_val, self.step)
            adjusted_max = self._round_to_step(max_val, self.step)
            adjusted_default = self._round_to_step(default_val, self.step)

            self.spinbox.setMinimum(int(adjusted_min))
            self.spinbox.setMaximum(int(adjusted_max))
            self.spinbox.setValue(int(adjusted_default))
            self.spinbox.setSingleStep(int(self.step))

        # Suffix hinzufügen wenn angegeben
        if self.suffix:
            self.spinbox.setSuffix(f" {self.suffix}")

        self.spinbox.setMaximumWidth(100)
        layout.addWidget(self.spinbox, 1, 1)

        # Signal-Verbindungen
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)

        self.setLayout(layout)

    def _round_to_step(self, value, step):
        """Rundet Wert auf nächsten Step"""
        if not step:
            return value
        return round(value / step) * step

    def on_slider_changed(self, value):
        """
        Funktionsweise: Wird aufgerufen wenn Slider bewegt wird
        - Konvertiert Slider-Wert zu SpinBox-Wert
        - Emittiert valueChanged Signal
        """
        if self.decimals > 0:
            # Decimal-Modus
            spinbox_value = value / self.slider_scale
        else:
            # Integer-Modus mit Steps
            spinbox_value = value * self.step

        self.spinbox.blockSignals(True)
        self.spinbox.setValue(spinbox_value)
        self.spinbox.blockSignals(False)

        self.valueChanged.emit()

    def on_spinbox_changed(self, value):
        """
        Funktionsweise: Wird aufgerufen wenn SpinBox geändert wird
        - Konvertiert SpinBox-Wert zu Slider-Wert
        - Emittiert valueChanged Signal
        """
        if self.spinbox.signalsBlocked():
            return

        if self.decimals > 0:
            # Decimal-Modus
            slider_value = int(value * self.slider_scale)
        else:
            # Integer-Modus mit Steps
            # Stelle sicher dass Wert an Step angepasst ist
            rounded_value = self._round_to_step(value, self.step)
            if rounded_value != value:
                self.spinbox.blockSignals(True)
                self.spinbox.setValue(int(rounded_value))
                self.spinbox.blockSignals(False)
                value = rounded_value

            slider_value = int(value // self.step)

        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
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
        else:
            # Stelle sicher dass Wert an Step angepasst ist
            value = self.spinbox.value()
            return self._round_to_step(value, self.step)

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
            # Decimal-Modus
            self.slider.setValue(int(value * self.slider_scale))
            self.spinbox.setValue(value)
        else:
            # Integer-Modus mit Steps
            rounded_value = self._round_to_step(value, self.step)
            self.slider.setValue(int(rounded_value // self.step))
            self.spinbox.setValue(int(rounded_value))

        # Signale wieder aktivieren
        self.slider.blockSignals(False)
        self.spinbox.blockSignals(False)

    def set_enabled(self, enabled):
        """Aktiviert/deaktiviert den gesamten Parameter-Slider"""
        self.slider.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)
        self.label.setEnabled(enabled)