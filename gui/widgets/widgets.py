"""
Path: gui/widgets/widgets.py

Funktionsweise: Erweiterte wiederverwendbare UI-Komponenten für Manual-Only System
- BaseButton mit konfigurierbarem Styling
- ParameterSlider mit value_default.py Integration und No-Wheel-Events
- StatusIndicator für Input-Dependencies
- ProgressBar für LOD-Progression
- RandomSeedButton für Seed-Parameter
- BiomeLegendDialog für Biome-Anzeige
- Memory-Management und Thread-Safety Verbesserungen
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from typing import Any, Dict, List, Optional
import random

from gui.config.gui_default import ColorSchemes


class BaseButton(QPushButton):
    """
    Funktionsweise: Erweiterte Button-Klasse mit konfigurierbarem Styling
    Aufgabe: Einheitliches Button-Design mit Hover-Effekten und Loading-States
    Kommunikation: Standard Qt-Signals mit erweiterten Status-Features
    """

    def __init__(self, text: str, button_type: str = "primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.is_loading = False
        self.original_text = text
        self.setup_styling()

    def setup_styling(self):
        """
        Funktionsweise: Setzt Button-Styling basierend auf Typ
        Aufgabe: Lädt Farben aus gui_default.py und wendet Styling an
        """
        # Standard-Farben (würden normalerweise aus gui_default.py kommen)
        styles = {
            "primary": {
                "color": "#488852",
                "hover": "#5e8964",
                "disabled": "#7f8c8d"
            },
            "secondary": {
                "color": "#487188",
                "hover": "#5e7a89",
                "disabled": "#7f8c8d"
            },
            "warning": {
                "color": "#884858",
                "hover": "#895e69",
                "disabled": "#7f8c8d"
            },
            "danger": {
                "color": "#e74c3c",
                "hover": "#c0392b",
                "disabled": "#7f8c8d"
            }
        }

        style = styles.get(self.button_type, styles["primary"])

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {style["color"]};
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 30px;
            }}
            QPushButton:hover {{
                background-color: {style["hover"]};
            }}
            QPushButton:pressed {{
                background-color: {style["hover"]};
                padding-top: 9px;
                padding-left: 17px;
            }}
            QPushButton:disabled {{
                background-color: {style["disabled"]};
                color: #bdc3c7;
            }}
        """)

    def set_loading(self, loading: bool):
        """
        Funktionsweise: Setzt Loading-Status mit Spinner
        Parameter: loading (bool)
        Aufgabe: Verhindert User-Interaction während Loading
        """
        self.is_loading = loading
        if loading:
            self.setText("⟳ Loading...")
            self.setEnabled(False)
        else:
            self.setText(self.original_text)
            self.setEnabled(True)


class RandomSeedButton(QPushButton):
    """
    Funktionsweise: Button für Random-Seed-Generation
    Aufgabe: Generiert zufällige Seeds für alle Generator-Parameter
    Kommunikation: seed_generated Signal mit neuem Seed-Wert
    Verwendung: Alle Generator-Tabs für seed Parameter
    """

    seed_generated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("🎲", parent)
        self.setMaximumWidth(30)
        self.setMaximumHeight(20)
        self.setToolTip("Generate random seed")
        self.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c5985;
            }
        """)
        self.clicked.connect(self.generate_random_seed)

    @pyqtSlot()
    def generate_random_seed(self):
        """
        Funktionsweise: Generiert und emittiert random seed
        Aufgabe: Erstellt zufälligen Seed-Wert im Standard-Range
        """
        new_seed = random.randint(0, 999999)
        self.seed_generated.emit(new_seed)


class ParameterSlider(QWidget):
    """
    Funktionsweise: Erweiterte Slider-Klasse mit value_default.py Integration
    Aufgabe: Parameter-Input mit Label, Value-Display, Validation
    Kommunikation: valueChanged Signal mit Value und Validation-Status
    Features: No-Wheel-Events (nur bei Focus), Reset-Button, Direct-Input
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default_val: float, step: float = 1.0, suffix: str = "",
                 description: str = "", parent=None):
        super().__init__(parent)

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.suffix = suffix
        self.description = description
        self.current_value = default_val
        self.default_value = default_val

        self.setup_ui(label)
        self.setValue(default_val)

    def setup_ui(self, label: str):
        """
        Funktionsweise: Erstellt UI für Parameter-Slider
        Aufgabe: Label, Slider, Value-Input, Reset-Button
        """
        layout = QVBoxLayout()
        layout.setSpacing(4)

        # Label mit Value-Display
        label_layout = QHBoxLayout()
        self.label = QLabel(label)
        self.label.setStyleSheet("font-size: 11px; font-weight: bold;")

        self.value_display = QLabel(f"{self.current_value}{self.suffix}")
        self.value_display.setAlignment(Qt.AlignRight)
        self.value_display.setMinimumWidth(60)
        self.value_display.setStyleSheet("font-size: 11px; color: #3498db;")

        label_layout.addWidget(self.label)
        label_layout.addWidget(self.value_display)
        layout.addLayout(label_layout)

        # Slider mit Input-Field
        slider_layout = QHBoxLayout()

        # Hauptslider mit No-Wheel-Events
        self.slider = NoWheelSlider(Qt.Horizontal)
        self.slider.setMinimum(int(self.min_val / self.step))
        self.slider.setMaximum(int(self.max_val / self.step))
        self.slider.valueChanged.connect(self.on_slider_changed)

        # Direct Input Field
        self.input_field = QLineEdit()
        self.input_field.setMaximumWidth(60)
        self.input_field.setMaximumHeight(20)
        self.input_field.setStyleSheet("font-size: 11px;")
        self.input_field.returnPressed.connect(self.on_input_changed)
        self.input_field.editingFinished.connect(self.on_input_changed)

        # Reset Button
        self.reset_button = QPushButton("↺")
        self.reset_button.setMaximumWidth(20)
        self.reset_button.setMaximumHeight(20)
        self.reset_button.setStyleSheet("font-size: 10px;")
        self.reset_button.setToolTip("Reset to default")
        self.reset_button.clicked.connect(self.reset_to_default)

        # Info-Button (hinter dem Reset-Button, siehe Nutzer-Anfrage): reine
        # Hover-Tooltip-Anzeige über Qt's eingebauten Tooltip-Mechanismus (kein
        # eigenes Popup nötig) - erklärt, was der Parameter fachlich bewirkt
        # (z.B. "Manning Coefficient" ist ohne Erklärung nicht selbsterklärend).
        # description kommt aus dem "description"-Key des jeweiligen
        # value_default.py-Parameter-Dicts (siehe get_parameter_config()) - bei
        # fehlendem Text ein generischer Platzhalter statt eines leeren Tooltips.
        self.info_button = QPushButton("ⓘ")
        self.info_button.setMaximumWidth(20)
        self.info_button.setMaximumHeight(20)
        self.info_button.setStyleSheet("font-size: 10px; color: #3498db;")
        self.info_button.setToolTip(self.description or "Keine Beschreibung verfügbar")
        self.info_button.setCursor(Qt.WhatsThisCursor)
        self.info_button.setFocusPolicy(Qt.NoFocus)

        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.input_field)
        slider_layout.addWidget(self.reset_button)
        slider_layout.addWidget(self.info_button)
        layout.addLayout(slider_layout)

        self.setLayout(layout)

    def setValue(self, value: float):
        """
        Funktionsweise: Setzt Slider-Wert mit Validation
        Parameter: value (float)
        """
        # Clamp zu Min/Max
        value = max(self.min_val, min(self.max_val, value))

        # Auf Step runden
        value = round(value / self.step) * self.step

        self.current_value = value

        # UI aktualisieren
        slider_value = int(value / self.step)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)

        self.input_field.setText(f"{value:.3f}".rstrip('0').rstrip('.'))
        self.value_display.setText(f"{value:.3f}".rstrip('0').rstrip('.') + self.suffix)

    def getValue(self) -> float:
        """Return: Aktueller Slider-Wert"""
        return self.current_value

    @pyqtSlot(int)
    def on_slider_changed(self, slider_value: int):
        """Slot für Slider-Änderungen"""
        value = slider_value * self.step
        self.setValue(value)
        self.valueChanged.emit(value)

    @pyqtSlot()
    def on_input_changed(self):
        """Slot für Direct-Input Änderungen"""
        try:
            value = float(self.input_field.text())
            self.setValue(value)
            self.valueChanged.emit(value)
        except ValueError:
            # Invalid input - reset to current value
            self.setValue(self.current_value)

    @pyqtSlot()
    def reset_to_default(self):
        """Reset zu Default-Wert"""
        self.setValue(self.default_value)
        self.valueChanged.emit(self.default_value)


class NoWheelSlider(QSlider):
    """QSlider die Mausrad-Events nur bei Focus akzeptiert"""

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class NoWheelComboBox(QComboBox):
    """QComboBox die Mausrad-Events nur bei Focus akzeptiert"""

    def wheelEvent(self, event):
        if not self.hasFocus():
            event.ignore()
        else:
            super().wheelEvent(event)


class StatusIndicator(QWidget):
    """
    Funktionsweise: Status-Anzeige für Dependencies und Validation
    Aufgabe: Visueller Status mit Icon, Text und Tooltip
    Kommunikation: Keine Signals - nur Display
    """

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.current_status = "unknown"
        self.setup_ui()

    # Feste Breite für den Status-Text - ohne das bestimmte die aktuelle
    # Textlänge (z.B. "Generating LOD 1/6 (25%)" vs. "Generation completed
    # (LOD 6)") den sizeHint() des QLabels, was den umgebenden Layouts
    # (Parameter-Slider im selben Panel) bei jedem Status-Wechsel eine neue
    # Breite aufzwang. Text wird bei Bedarf mit "…" gekürzt, volle Nachricht
    # bleibt über den Tooltip erreichbar.
    _STATUS_TEXT_WIDTH = 200

    def setup_ui(self):
        """Erstellt UI für Status-Indicator"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Status Icon
        self.status_icon = QLabel("◯")
        self.status_icon.setFixedSize(16, 16)
        self.status_icon.setAlignment(Qt.AlignCenter)

        # Status Text
        self.status_text = QLabel(self.label_text)
        self.status_text.setFixedWidth(self._STATUS_TEXT_WIDTH)
        self.status_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        layout.addWidget(self.status_icon)
        layout.addWidget(self.status_text)
        layout.addStretch()

        self.setLayout(layout)

        # Initial Status
        self.set_unknown()

    def _set_status_text(self, full_text: str):
        """Setzt den Status-Text elidiert auf die feste Breite, volle Nachricht per Tooltip."""
        metrics = QFontMetrics(self.status_text.font())
        elided = metrics.elidedText(full_text, Qt.ElideRight, self._STATUS_TEXT_WIDTH)
        self.status_text.setText(elided)

    def set_success(self, message: str = ""):
        """Setzt Success-Status"""
        self.current_status = "success"
        self.status_icon.setText("✓")
        self.status_icon.setStyleSheet("color: #27ae60; font-weight: bold;")

        if message:
            self._set_status_text(f"{self.label_text}: {message}")
            self.setToolTip(message)
        else:
            self._set_status_text(f"{self.label_text}: OK")

    def set_warning(self, message: str):
        """Setzt Warning-Status"""
        self.current_status = "warning"
        self.status_icon.setText("⚠")
        self.status_icon.setStyleSheet("color: #f39c12; font-weight: bold;")
        self._set_status_text(f"{self.label_text}: {message}")
        self.setToolTip(message)

    def set_error(self, message: str):
        """Setzt Error-Status"""
        self.current_status = "error"
        self.status_icon.setText("✗")
        self.status_icon.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self._set_status_text(f"{self.label_text}: {message}")
        self.setToolTip(message)

    def set_unknown(self):
        """Setzt Unknown-Status"""
        self.current_status = "unknown"
        self.status_icon.setText("◯")
        self.status_icon.setStyleSheet("color: #7f8c8d;")
        self._set_status_text(f"{self.label_text}: Unknown")
        self.setToolTip("Status unknown")

    def set_pending(self, message: str = ""):
        """Setzt Pending-Status für laufende Vorgänge"""
        self.current_status = "pending"
        self.status_icon.setText("…")
        self.status_icon.setStyleSheet("color: #3498db; font-weight: bold;")

        if message:
            self._set_status_text(f"{self.label_text}: {message}")
            self.setToolTip(message)
        else:
            self._set_status_text(f"{self.label_text}: Pending")


class ProgressBar(QWidget):
    """
    Funktionsweise: Erweiterte Progress-Bar für LOD-Progression
    Aufgabe: Zeigt Fortschritt durch Generator-Pipeline
    Kommunikation: Keine Signals - nur Display
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Progress-Bar"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Progress Label
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        layout.addWidget(self.progress_label)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(20)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-size: 10px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Detail Label
        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("font-size: 9px; color: #7f8c8d;")
        layout.addWidget(self.detail_label)

        self.setLayout(layout)

    def set_progress(self, value: int, text: str = "", detail: str = ""):
        """
        Funktionsweise: Setzt Progress-Wert und Text
        Parameter: value (0-100), text (Haupttext), detail (Detailtext)
        """
        self.progress_bar.setValue(value)

        if text:
            self.progress_label.setText(text)

        if detail:
            self.detail_label.setText(detail)

    def set_lod_progress(self, current_lod: int, max_lod: int, phase: str):
        """
        Funktionsweise: Setzt LOD-spezifischen Progress
        Parameter: current_lod, max_lod, phase (z.B. "Heightmap Generation")
        """
        progress = int((current_lod / max_lod) * 100)
        self.set_progress(
            progress,
            f"LOD {current_lod}/{max_lod}",
            phase
        )

    def reset(self):
        """Reset zu Initial State"""
        self.set_progress(0, "Ready", "")


class DisplayWrapper:
    """
    Funktionsweise: Wrapper für Display-Objekte mit Fallback-Handling
    Aufgabe: Einheitliche API auch für Fallback-Labels
    Verwendung: In base_tab.py für 2D/3D Display-Management
    """
    def __init__(self, display_object):
        self.display = display_object
        self.is_fallback = isinstance(display_object, QLabel)
        self.is_active = True

    def update_display(self, data, layer_type):
        """
        Funktionsweise: Universal Display-Update mit Fallback-Support
        Parameter: data (numpy array), layer_type (str)
        """
        if self.is_fallback:
            # Fallback-Label mit Data-Info aktualisieren
            data_info = f"Size: {data.shape}" if hasattr(data, 'shape') else 'Unknown'
            self.display.setText(f"Display Fallback\n{layer_type} data available\n{data_info}")
        elif hasattr(self.display, 'update_display'):
            self.display.update_display(data, layer_type)
        elif hasattr(self.display, 'update_heightmap'):
            # Fallback für 3D Display
            self.display.update_heightmap(data, layer_type)

    def set_active(self, active: bool):
        """
        Funktionsweise: Setzt Active-State für Memory-Management
        Parameter: active (bool)
        Aufgabe: Cleanup inaktiver Displays
        """
        self.is_active = active
        if hasattr(self.display, 'set_active'):
            self.display.set_active(active)
        elif hasattr(self.display, 'cleanup_textures') and not active:
            # Cleanup bei Deactivation
            self.display.cleanup_textures()

    def cleanup_resources(self):
        """
        Funktionsweise: Resource-Cleanup für Display-Wrapper
        """
        if hasattr(self.display, 'cleanup_resources'):
            self.display.cleanup_resources()


class DependencyResolver:
    """
    Funktionsweise: Robuste Dependency-Resolution für Cross-Tab Dependencies
    Aufgabe: Automatic Retry, Dependency-Chain-Validation, Fallback-Handling
    Verwendung: In Tabs mit komplexen Dependencies (Water, Biome, Settlement)
    """
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.retry_attempts = {}
        self.max_retries = 3

    def resolve_dependencies(self, generator_type: str, required_deps: List[str]) -> tuple[bool, List[str]]:
        """
        Funktionsweise: Versucht Dependencies zu resolven mit Retry-Logic
        Parameter: generator_type (str), required_deps (list)
        Return: (success, missing_dependencies)
        """
        missing = []

        for dep in required_deps:
            if not self.data_manager.has_data(dep):
                # Retry-Logic für fehlende Dependencies
                retry_key = f"{generator_type}_{dep}"

                if retry_key not in self.retry_attempts:
                    self.retry_attempts[retry_key] = 0

                if self.retry_attempts[retry_key] < self.max_retries:
                    self.retry_attempts[retry_key] += 1
                    # Note: Actual re-generation würde hier getriggert werden
                    # Das ist implementation-specific für jeden Generator

                missing.append(dep)
            else:
                # Reset retry counter bei Success
                retry_key = f"{generator_type}_{dep}"
                if retry_key in self.retry_attempts:
                    del self.retry_attempts[retry_key]

        return len(missing) == 0, missing

    def reset_retries(self, generator_type: str = None):
        """
        Funktionsweise: Reset Retry-Counters
        Parameter: generator_type (optional) - None für alle
        """
        if generator_type:
            keys_to_remove = [k for k in self.retry_attempts.keys() if k.startswith(generator_type)]
            for key in keys_to_remove:
                del self.retry_attempts[key]
        else:
            self.retry_attempts.clear()


class BiomeLegendDialog(QDialog):
    """
    Funktionsweise: Dialog für Biome-Legend mit allen 26 Biome-Typen
    Aufgabe: Übersichtliche Darstellung aller Base- und Super-Biomes
    Einsatz: Biome-Tab Legend-Button
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Biome Legend")
        self.setModal(True)
        self.resize(600, 800)
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Biome-Legend"""
        layout = QVBoxLayout()

        # Scroll Area für viele Biomes
        scroll = QScrollArea()
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        # Base Biomes
        base_group = QGroupBox("Base Biomes (15 Types)")
        base_layout = QGridLayout()

        base_biomes = ColorSchemes.BIOME_COLOR_TABLE[:15]

        for i, (name, color) in enumerate(base_biomes):
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

            name_label = QLabel(name)

            base_layout.addWidget(color_box, i, 0)
            base_layout.addWidget(name_label, i, 1)

        base_group.setLayout(base_layout)
        content_layout.addWidget(base_group)

        # Super Biomes
        super_group = QGroupBox("Super Biomes (11 Types)")
        super_layout = QGridLayout()

        super_biomes = ColorSchemes.BIOME_COLOR_TABLE[15:]

        for i, (name, color) in enumerate(super_biomes):
            color_box = QLabel()
            color_box.setFixedSize(20, 20)
            color_box.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

            name_label = QLabel(name)

            super_layout.addWidget(color_box, i, 0)
            super_layout.addWidget(name_label, i, 1)

        super_group.setLayout(super_layout)
        content_layout.addWidget(super_group)

        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)

        layout.addWidget(scroll)

        # Close Button
        close_button = BaseButton("Close", "secondary")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

        self.setLayout(layout)

class GradientBackgroundWidget(QWidget):
    """
    Funktionsweise: Widget mit Gradient-Background fÃ¼r Main Menu
    Aufgabe: Visuell ansprechender Hintergrund
    """

    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        """Override fÃ¼r Custom Background Painting"""
        painter = QPainter(self)

        # Gradient von oben nach unten
        gradient = QLinearGradient(0, 0, 0, self.height())

        gradient.setColorAt(0, QColor("#f8f9fa"))  # Helles Grau oben
        gradient.setColorAt(0.3, QColor("#e9ecef"))  # Mittleres Grau
        gradient.setColorAt(0.7, QColor("#dee2e6"))  # Etwas dunkler
        gradient.setColorAt(1, QColor("#ced4da"))  # Dunkelgrau unten

        # Gradient anwenden
        painter.fillRect(self.rect(), gradient)

        # Optional: Subtle Pattern Overlay
        self.draw_pattern_overlay(painter)

    def draw_pattern_overlay(self, painter: QPainter):
        """
        Funktionsweise: Zeichnet subtiles Pattern-Overlay
        Parameter: painter (QPainter)
        """

        # Sehr subtiles Raster-Pattern
        painter.setPen(QPen(QColor(255, 255, 255, 10), 1))  # Fast transparent

        # Vertikale Linien
        for x in range(0, self.width(), 50):
            painter.drawLine(x, 0, x, self.height())

        # Horizontale Linien
        for y in range(0, self.height(), 50):
            painter.drawLine(0, y, self.width(), y)

class NavigationPanel(QGroupBox):
    """
    Funktionsweise: Navigation Panel für Tab-Wechsel und Workflow-Navigation
    Aufgabe: Previous/Next Buttons, Workflow-Progress
    Kommunikation: Signals für Navigation-Requests
    """

    navigation_requested = pyqtSignal(str)  # (target_tab)

    def __init__(self, navigation_manager, parent=None):
        super().__init__("Navigation", parent)
        self.navigation_manager = navigation_manager
        self.current_tab = None
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Navigation Panel"""
        layout = QVBoxLayout()

        # Previous/Next Navigation
        nav_layout = QHBoxLayout()

        self.prev_button = BaseButton("← Previous", "secondary")
        self.prev_button.clicked.connect(self.go_previous)
        nav_layout.addWidget(self.prev_button)

        self.next_button = BaseButton("Next →", "primary")
        self.next_button.clicked.connect(self.go_next)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)
        self.setLayout(layout)

    def set_current_tab(self, tab_name: str):
        """
        Funktionsweise: Setzt aktuellen Tab und aktualisiert Button-States
        Parameter: tab_name (str)
        """
        self.current_tab = tab_name
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        """Aktualisiert Previous/Next Button States"""
        if not self.navigation_manager:
            return

        # Tab-Reihenfolge
        tab_order = ["terrain", "geology", "weather", "water", "biome", "settlement", "overview"]

        if self.current_tab in tab_order:
            current_index = tab_order.index(self.current_tab)

            # Previous Button
            self.prev_button.setEnabled(current_index > 0)

            # Next Button
            self.next_button.setEnabled(current_index < len(tab_order) - 1)

    @pyqtSlot()
    def go_previous(self):
        """Navigation zu Previous Tab"""
        tab_order = ["terrain", "geology", "weather", "water", "biome", "settlement", "overview"]

        if self.current_tab in tab_order:
            current_index = tab_order.index(self.current_tab)
            if current_index > 0:
                target_tab = tab_order[current_index - 1]
                self.navigation_requested.emit(target_tab)

    @pyqtSlot()
    def go_next(self):
        """Navigation zu Next Tab"""
        tab_order = ["terrain", "geology", "weather", "water", "biome", "settlement", "overview"]

        if self.current_tab in tab_order:
            current_index = tab_order.index(self.current_tab)
            if current_index < len(tab_order) - 1:
                target_tab = tab_order[current_index + 1]
                self.navigation_requested.emit(target_tab)
