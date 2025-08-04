"""
Path: gui/widgets/widgets.py

Funktionsweise: Erweiterte wiederverwendbare UI-Komponenten - ERWEITERT
- BaseButton mit konfigurierbarem Styling erweitert
- ParameterSlider mit value_default.py Integration erweitert
- StatusIndicator für Input-Dependencies erweitert
- ProgressBar für Tab-Navigation erweitert
- AutoSimulationPanel für alle Tabs erweitert - PUBLIC API HINZUGEFÜGT
- RandomSeedButton für Terrain-Parameter hinzugefügt - NEU
- Neue Widgets für komplexe Datenstrukturen
- Memory-Management und Thread-Safety Verbesserungen
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from typing import Any, Dict, List, Optional


class BaseButton(QPushButton):
    """
    Funktionsweise: Erweiterte Button-Klasse mit konfigurierbarem Styling
    Aufgabe: Einheitliches Button-Design mit Hover-Effekten und Status-Integration
    Kommunikation: Standard Qt-Signals mit erweiterten Status-Features
    """

    def __init__(self, text: str, button_type: str = "primary", parent=None):
        super().__init__(text, parent)
        self.button_type = button_type
        self.is_loading = False
        self.original_text = text  # Für Loading-State Recovery
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
        Funktionsweise: Setzt Loading-Status mit Spinner - VERBESSERT
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
    Funktionsweise: Button für Random-Seed-Generation - NEU
    Aufgabe: Generiert zufällige Seeds für map_seed Parameter
    Kommunikation: seed_generated Signal mit neuem Seed-Wert
    Verwendung: Terrain-Tab für map_seed Parameter
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
        Aufgabe: Erstellt zufälligen Seed-Wert im map_seed Range
        """
        import random
        new_seed = random.randint(0, 999999)
        self.seed_generated.emit(new_seed)


class ParameterSlider(QWidget):
    """
    Funktionsweise: Erweiterte Slider-Klasse mit value_default.py Integration
    Aufgabe: Parameter-Input mit Label, Value-Display, Validation
    Kommunikation: valueChanged Signal mit Value und Validation-Status
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, min_val: float, max_val: float,
                 default_val: float, step: float = 1.0, suffix: str = "", parent=None):
        super().__init__(parent)

        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.suffix = suffix
        self.current_value = default_val

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

        # Hauptslider
        self.slider = QSlider(Qt.Horizontal)
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

        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.input_field)
        slider_layout.addWidget(self.reset_button)
        layout.addLayout(slider_layout)

        self.setLayout(layout)

        # Store default für Reset
        self.default_value = self.current_value

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


class StatusIndicator(QWidget):
    """
    Funktionsweise: Erweiterte Status-Anzeige für Dependencies und Validation
    Aufgabe: Visueller Status mit Icon, Text und Tooltip
    Kommunikation: Keine Signals - nur Display
    """

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.label_text = label
        self.current_status = "unknown"
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Status-Indicator"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Status Icon
        self.status_icon = QLabel("●")
        self.status_icon.setFixedSize(16, 16)
        self.status_icon.setAlignment(Qt.AlignCenter)

        # Status Text
        self.status_text = QLabel(self.label_text)

        layout.addWidget(self.status_icon)
        layout.addWidget(self.status_text)
        layout.addStretch()

        self.setLayout(layout)

        # Initial Status
        self.set_unknown()

    def set_success(self, message: str = ""):
        """Setzt Success-Status"""
        self.current_status = "success"
        self.status_icon.setText("✓")
        self.status_icon.setStyleSheet("color: #27ae60; font-weight: bold;")

        if message:
            self.status_text.setText(f"{self.label_text}: {message}")
            self.setToolTip(message)
        else:
            self.status_text.setText(f"{self.label_text}: OK")

    def set_warning(self, message: str):
        """Setzt Warning-Status"""
        self.current_status = "warning"
        self.status_icon.setText("⚠")
        self.status_icon.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.status_text.setText(f"{self.label_text}: {message}")
        self.setToolTip(message)

    def set_error(self, message: str):
        """Setzt Error-Status"""
        self.current_status = "error"
        self.status_icon.setText("✗")
        self.status_icon.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.status_text.setText(f"{self.label_text}: {message}")
        self.setToolTip(message)

    def set_unknown(self):
        """Setzt Unknown-Status"""
        self.current_status = "unknown"
        self.status_icon.setText("●")
        self.status_icon.setStyleSheet("color: #7f8c8d;")
        self.status_text.setText(f"{self.label_text}: Unknown")
        self.setToolTip("Status unknown")


class AutoSimulationPanel(QGroupBox):
    """
    Funktionsweise: Erweiterte Auto-Simulation Controls für alle Tabs - PUBLIC API ERWEITERT
    Aufgabe: Auto-Update Toggle, Manual Generation, Performance-Settings
    Kommunikation: Signals für Auto-Toggle und Manual-Generation
    API-ERWEITERUNG: Public Methods für External Control (Fix für Problem 9)
    """

    auto_simulation_toggled = pyqtSignal(bool)
    manual_generation_requested = pyqtSignal()
    performance_level_changed = pyqtSignal(str)

    def __init__(self, generator_name: str, parent=None):
        super().__init__(f"{generator_name} Generation", parent)
        self.generator_name = generator_name
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Auto-Simulation Panel"""
        layout = QVBoxLayout()

        # Auto-Simulation Toggle
        self.auto_checkbox = QCheckBox("Auto Update")
        self.auto_checkbox.setToolTip("Automatically regenerate when parameters change")
        self.auto_checkbox.toggled.connect(self.auto_simulation_toggled.emit)
        layout.addWidget(self.auto_checkbox)

        # Manual Generation Button
        self.manual_button = BaseButton(f"Generate {self.generator_name}", "primary")
        self.manual_button.clicked.connect(self.manual_generation_requested.emit)
        layout.addWidget(self.manual_button)

        # Performance Level Selection
        perf_layout = QHBoxLayout()
        perf_layout.addWidget(QLabel("Quality:"))

        self.performance_combo = QComboBox()
        self.performance_combo.addItems(["Live (64x64)", "Preview (256x256)", "Final (512x512)"])
        self.performance_combo.setCurrentIndex(1)  # Preview als Default
        self.performance_combo.currentTextChanged.connect(self.on_performance_changed)
        perf_layout.addWidget(self.performance_combo)

        layout.addLayout(perf_layout)

        # Generation Status
        self.generation_status = StatusIndicator("Generation Status")
        layout.addWidget(self.generation_status)

        self.setLayout(layout)

    @pyqtSlot(str)
    def on_performance_changed(self, text: str):
        """Slot für Performance-Level Änderungen"""
        # Extract performance level from combo text
        if "Live" in text:
            level = "live"
        elif "Preview" in text:
            level = "preview"
        elif "Final" in text:
            level = "final"
        else:
            level = "preview"

        self.performance_level_changed.emit(level)

    def set_generation_status(self, status: str, message: str = ""):
        """
        Funktionsweise: Setzt Generation-Status
        Parameter: status ("success", "warning", "error", "progress"), message (str)
        """
        if status == "success":
            self.generation_status.set_success(message or "Generation completed")
        elif status == "warning":
            self.generation_status.set_warning(message)
        elif status == "error":
            self.generation_status.set_error(message)
        elif status == "progress":
            self.generation_status.set_warning(message or "Generation in progress...")
        else:
            self.generation_status.set_unknown()

    def set_generation_in_progress(self, in_progress: bool):
        """
        Funktionsweise: Setzt Loading-Status für Generation
        Parameter: in_progress (bool)
        """
        self.manual_button.set_loading(in_progress)
        if in_progress:
            self.generation_status.set_warning("Generation in progress...")

    # PUBLIC API für External Control - NEUE METHODEN (Fix für Problem 9):

    def set_manual_button_enabled(self, enabled: bool):
        """
        Funktionsweise: Public API für Manual-Button State
        Parameter: enabled (bool)
        Aufgabe: Ermöglicht External Control des Manual-Buttons
        """
        if not self.manual_button.is_loading:  # Nur ändern wenn nicht Loading
            self.manual_button.setEnabled(enabled)

    def get_manual_button_enabled(self) -> bool:
        """
        Funktionsweise: Public API für Manual-Button State Check
        Return: Manual-Button enabled state (bool)
        """
        return self.manual_button.isEnabled()

    def set_auto_simulation_enabled(self, enabled: bool):
        """
        Funktionsweise: Public API für Auto-Simulation State
        Parameter: enabled (bool)
        """
        self.auto_checkbox.setChecked(enabled)

    def get_auto_simulation_enabled(self) -> bool:
        """
        Funktionsweise: Public API für Auto-Simulation State Check
        Return: Auto-Simulation enabled state (bool)
        """
        return self.auto_checkbox.isChecked()


class ParameterUpdateManager:
    """
    Funktionsweise: Centralized Parameter-Update Management - NEU (Fix für Problem 7)
    Aufgabe: Verhindert Race-Conditions zwischen Validation und Generation
    Verwendung: In allen Tab-Klassen für koordinierte Parameter-Updates
    """
    def __init__(self, tab):
        self.tab = tab
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.process_updates)

        self.pending_validation = False
        self.pending_generation = False

    def request_validation(self):
        """
        Funktionsweise: Request Parameter-Validation mit Debouncing
        Aufgabe: Verhindert excessive Validation-Calls
        """
        self.pending_validation = True
        self.update_timer.start(300)  # 300ms Delay für Validation

    def request_generation(self):
        """
        Funktionsweise: Request Auto-Generation mit Debouncing
        Aufgabe: Verhindert excessive Generation-Calls
        """
        self.pending_generation = True
        self.update_timer.start(1000)  # 1000ms Delay für Generation

    def process_updates(self):
        """
        Funktionsweise: Verarbeitet pending Updates in korrekter Reihenfolge
        Aufgabe: Validation IMMER vor Generation
        """
        if self.pending_validation:
            if hasattr(self.tab, 'validate_current_parameters'):
                self.tab.validate_current_parameters()
            self.pending_validation = False

        if self.pending_generation and hasattr(self.tab, 'auto_simulation_enabled'):
            if self.tab.auto_simulation_enabled and not getattr(self.tab, 'generation_in_progress', False):
                if hasattr(self.tab, 'generate'):
                    self.tab.generate()
            self.pending_generation = False


class DisplayWrapper:
    """
    Funktionsweise: Wrapper für Display-Objekte mit Fallback-Handling - NEU (Fix für Problem 5)
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
        Aufgabe: Cleanup inaktiver Displays (Fix für Problem 11)
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
    Funktionsweise: Robust Dependency-Resolution für Cross-Tab Dependencies - NEU (Fix für Problem 20)
    Aufgabe: Automatic Retry, Dependency-Chain-Validation, Fallback-Handling
    Verwendung: In Tabs mit komplexen Dependencies (Geology, Water, etc.)
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


class ProgressBar(QWidget):
    """
    Funktionsweise: Erweiterte Progress-Bar für Tab-Navigation
    Aufgabe: Zeigt Fortschritt durch Generator-Pipeline
    Kommunikation: Keine Signals - nur Display
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tab_order = ["terrain", "geology", "settlement", "weather", "water", "biome"]
        self.completed_tabs = set()
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Progress-Bar"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.tab_indicators = {}

        for i, tab_name in enumerate(self.tab_order):
            # Tab-Indicator
            indicator = QLabel(tab_name.title())
            indicator.setAlignment(Qt.AlignCenter)
            indicator.setMinimumWidth(80)
            indicator.setStyleSheet("""
                QLabel {
                    border: 2px solid #bdc3c7;
                    border-radius: 4px;
                    padding: 4px;
                    background-color: #ecf0f1;
                    color: #7f8c8d;
                }
            """)

            self.tab_indicators[tab_name] = indicator
            layout.addWidget(indicator)

            # Pfeil zwischen Tabs (außer beim letzten)
            if i < len(self.tab_order) - 1:
                arrow = QLabel("→")
                arrow.setAlignment(Qt.AlignCenter)
                arrow.setStyleSheet("color: #bdc3c7; font-size: 16px; font-weight: bold;")
                layout.addWidget(arrow)

        self.setLayout(layout)

    def set_tab_completed(self, tab_name: str):
        """
        Funktionsweise: Markiert Tab als completed
        Parameter: tab_name (str)
        """
        if tab_name in self.tab_indicators:
            self.completed_tabs.add(tab_name)
            indicator = self.tab_indicators[tab_name]
            indicator.setStyleSheet("""
                QLabel {
                    border: 2px solid #27ae60;
                    border-radius: 4px;
                    padding: 4px;
                    background-color: #d5f4e6;
                    color: #27ae60;
                    font-weight: bold;
                }
            """)

    def set_tab_active(self, tab_name: str):
        """
        Funktionsweise: Markiert Tab als aktiv
        Parameter: tab_name (str)
        """
        # Alle Tabs zurücksetzen
        for name, indicator in self.tab_indicators.items():
            if name in self.completed_tabs:
                # Completed Style beibehalten
                continue
            elif name == tab_name:
                # Active Style
                indicator.setStyleSheet("""
                    QLabel {
                        border: 2px solid #3498db;
                        border-radius: 4px;
                        padding: 4px;
                        background-color: #d6eaf8;
                        color: #3498db;
                        font-weight: bold;
                    }
                """)
            else:
                # Default Style
                indicator.setStyleSheet("""
                    QLabel {
                        border: 2px solid #bdc3c7;
                        border-radius: 4px;
                        padding: 4px;
                        background-color: #ecf0f1;
                        color: #7f8c8d;
                    }
                """)


class MultiDependencyStatusWidget(QGroupBox):
    """
    Funktionsweise: Widget für detaillierte Dependency-Status Anzeige
    Aufgabe: Zeigt Status aller Required Dependencies einzeln an
    Wiederverwendung: Für Water-Tab und andere komplexe Dependencies
    """

    def __init__(self, required_dependencies: List[str], title: str = "Dependencies", parent=None):
        super().__init__(title, parent)
        self.required_dependencies = required_dependencies
        self.dependency_indicators = {}
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Dependency-Status"""
        layout = QVBoxLayout()

        # Status Indicator für jede Dependency
        for dependency in self.required_dependencies:
            display_name = dependency.replace("_", " ").title()
            indicator = StatusIndicator(display_name)
            indicator.set_warning("Not available")
            self.dependency_indicators[dependency] = indicator
            layout.addWidget(indicator)

        # Overall Status
        self.overall_status = StatusIndicator("Overall Status")
        self.overall_status.set_error("Dependencies missing")
        layout.addWidget(self.overall_status)

        self.setLayout(layout)

    def update_dependency_status(self, is_complete: bool, missing: List[str]):
        """
        Funktionsweise: Aktualisiert Status aller Dependencies
        Parameter: is_complete (bool), missing (list of dependency names)
        """
        # Individual Dependencies aktualisieren
        for dependency in self.required_dependencies:
            indicator = self.dependency_indicators[dependency]
            if dependency in missing:
                indicator.set_warning("Missing")
            else:
                indicator.set_success("Available")

        # Overall Status
        if is_complete:
            self.overall_status.set_success("All dependencies available")
        else:
            self.overall_status.set_error(f"Missing: {len(missing)} dependencies")

    def set_error(self, message: str):
        """Setzt Error-Status für Overall Status"""
        self.overall_status.set_error(message)


class WorldStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Widget für umfassende Welt-Statistiken
    Aufgabe: Zeigt Zusammenfassung aller Generator-Outputs
    Einsatz: Biome-Tab für finale Welt-Übersicht
    """

    def __init__(self, parent=None):
        super().__init__("World Statistics", parent)
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für World-Statistics"""
        layout = QVBoxLayout()

        # Terrain Statistics
        terrain_group = QGroupBox("Terrain")
        terrain_layout = QVBoxLayout()
        self.terrain_stats = QLabel("Elevation: - | Slope: -")
        terrain_layout.addWidget(self.terrain_stats)
        terrain_group.setLayout(terrain_layout)
        layout.addWidget(terrain_group)

        # Climate Statistics
        climate_group = QGroupBox("Climate")
        climate_layout = QVBoxLayout()
        self.climate_stats = QLabel("Temperature: - | Precipitation: -")
        climate_layout.addWidget(self.climate_stats)
        climate_group.setLayout(climate_layout)
        layout.addWidget(climate_group)

        # Water Statistics
        water_group = QGroupBox("Hydrology")
        water_layout = QVBoxLayout()
        self.water_stats = QLabel("Water Coverage: - | Rivers: -")
        water_layout.addWidget(self.water_stats)
        water_group.setLayout(water_layout)
        layout.addWidget(water_group)

        # Biome Statistics
        biome_group = QGroupBox("Biomes")
        biome_layout = QVBoxLayout()
        self.biome_stats = QLabel("Biome Types: - | Diversity: -")
        biome_layout.addWidget(self.biome_stats)
        biome_group.setLayout(biome_layout)
        layout.addWidget(biome_group)

        self.setLayout(layout)

    def update_world_statistics(self, inputs: Dict[str, np.ndarray],
                                biome_map: np.ndarray, super_biome_mask: np.ndarray):
        """
        Funktionsweise: Berechnet und zeigt umfassende Welt-Statistiken
        Parameter: inputs (dict), biome_map, super_biome_mask (numpy arrays)
        """
        # Terrain Statistics
        heightmap = inputs.get("heightmap")
        if heightmap is not None:
            elev_min, elev_max = np.min(heightmap), np.max(heightmap)
            self.terrain_stats.setText(f"Elevation: {elev_min:.0f}m - {elev_max:.0f}m")

        # Climate Statistics
        temp_map = inputs.get("temp_map")
        if temp_map is not None:
            temp_min, temp_max = np.min(temp_map), np.max(temp_map)
            self.climate_stats.setText(f"Temperature: {temp_min:.1f}°C - {temp_max:.1f}°C")

        # Water Statistics
        soil_moist_map = inputs.get("soil_moist_map")
        if soil_moist_map is not None:
            water_coverage = np.sum(soil_moist_map > 0.5) / soil_moist_map.size * 100
            self.water_stats.setText(f"Water Coverage: {water_coverage:.1f}%")

        # Biome Statistics
        unique_biomes = len(np.unique(biome_map))
        diversity_index = self.calculate_shannon_diversity(biome_map)
        self.biome_stats.setText(f"Biome Types: {unique_biomes} | Diversity: {diversity_index:.2f}")

    def calculate_shannon_diversity(self, biome_map: np.ndarray) -> float:
        """
        Funktionsweise: Berechnet Shannon-Diversity Index für Biom-Verteilung
        Parameter: biome_map (numpy array)
        Return: Shannon-Diversity Index (float)
        """
        unique, counts = np.unique(biome_map, return_counts=True)
        proportions = counts / counts.sum()

        # Shannon-Entropy: H = -Σ(p_i * log(p_i))
        shannon_entropy = -np.sum(proportions * np.log(proportions + 1e-10))  # +epsilon für log(0)
        return shannon_entropy


class WorldExportWidget(QGroupBox):
    """
    Funktionsweise: Widget für World-Export Controls
    Aufgabe: Export-Format Selection, Optionen, Export-Trigger
    Kommunikation: export_requested Signal mit Format und Optionen
    """

    export_requested = pyqtSignal(str, dict)  # (format, options)

    def __init__(self, parent=None):
        super().__init__("World Export", parent)
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Export-Controls"""
        layout = QVBoxLayout()

        # Export Format Selection
        format_group = QGroupBox("Export Format")
        format_layout = QVBoxLayout()

        self.format_radio_group = QButtonGroup()

        self.png_radio = QRadioButton("PNG Maps (All Layers)")
        self.png_radio.setChecked(True)
        self.format_radio_group.addButton(self.png_radio, 0)
        format_layout.addWidget(self.png_radio)

        self.json_radio = QRadioButton("JSON Data (Complete World)")
        self.format_radio_group.addButton(self.json_radio, 1)
        format_layout.addWidget(self.json_radio)

        self.obj_radio = QRadioButton("3D Terrain (OBJ)")
        self.format_radio_group.addButton(self.obj_radio, 2)
        format_layout.addWidget(self.obj_radio)

        format_group.setLayout(format_layout)
        layout.addWidget(format_group)

        # Export Options
        options_group = QGroupBox("Export Options")
        options_layout = QVBoxLayout()

        # Directory/File Selection
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit("world_export")
        self.browse_button = BaseButton("Browse", "secondary")
        self.browse_button.clicked.connect(self.browse_export_location)
        file_layout.addWidget(QLabel("Location:"))
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.browse_button)
        options_layout.addLayout(file_layout)

        # Quality Settings
        quality_layout = QHBoxLayout()
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["High (300 DPI)", "Medium (150 DPI)", "Low (72 DPI)"])
        quality_layout.addWidget(QLabel("Quality:"))
        quality_layout.addWidget(self.quality_combo)
        options_layout.addLayout(quality_layout)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Export Button
        self.export_button = BaseButton("Export World", "primary")
        self.export_button.clicked.connect(self.trigger_export)
        layout.addWidget(self.export_button)

        # Export Status
        self.export_status = StatusIndicator("Export Status")
        layout.addWidget(self.export_status)

        self.setLayout(layout)

    @pyqtSlot()
    def browse_export_location(self):
        """Browse für Export-Location"""
        current_format = self.format_radio_group.checkedId()

        if current_format == 0:  # PNG
            directory = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if directory:
                self.file_path.setText(directory)
        else:  # JSON or OBJ
            if current_format == 1:  # JSON
                filename, _ = QFileDialog.getSaveFileName(self, "Save World Data",
                                                          "world_data.json", "JSON Files (*.json)")
            else:  # OBJ
                filename, _ = QFileDialog.getSaveFileName(self, "Save 3D Terrain",
                                                          "terrain.obj", "OBJ Files (*.obj)")
            if filename:
                self.file_path.setText(filename)

    @pyqtSlot()
    def trigger_export(self):
        """Triggert Export mit aktuellen Einstellungen"""
        current_format = self.format_radio_group.checkedId()

        # Format bestimmen
        format_map = {0: "png", 1: "json", 2: "obj"}
        export_format = format_map[current_format]

        # Export-Optionen sammeln
        options = {
            "export_location": self.file_path.text(),
            "quality": self.quality_combo.currentText(),
            "dpi": self.get_dpi_from_quality()
        }

        if export_format == "png":
            options["export_directory"] = options["export_location"]
        else:
            options["export_file"] = options["export_location"]

        # Validation
        if not options["export_location"]:
            self.export_status.set_error("Please select export location")
            return

        # Export triggern
        self.export_status.set_warning("Export in progress...")
        self.export_button.set_loading(True)

        self.export_requested.emit(export_format, options)

    def get_dpi_from_quality(self) -> int:
        """Extrahiert DPI aus Quality-Selection"""
        quality_text = self.quality_combo.currentText()
        if "300" in quality_text:
            return 300
        elif "150" in quality_text:
            return 150
        else:
            return 72

    def set_export_complete(self, success: bool, message: str = ""):
        """
        Funktionsweise: Setzt Export-Complete Status
        Parameter: success (bool), message (str)
        """
        self.export_button.set_loading(False)

        if success:
            self.export_status.set_success(message or "Export completed successfully")
        else:
            self.export_status.set_error(message or "Export failed")


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

        base_biomes = [
            ("Ice Cap", "#f8f9fa"),
            ("Tundra", "#e9ecef"),
            ("Taiga", "#228b22"),
            ("Grassland", "#90ee90"),
            ("Temperate Forest", "#006400"),
            ("Mediterranean", "#9acd32"),
            ("Desert", "#daa520"),
            ("Semi Arid", "#d2691e"),
            ("Tropical Rainforest", "#008000"),
            ("Tropical Seasonal", "#32cd32"),
            ("Savanna", "#bdb76b"),
            ("Montane Forest", "#2e8b57"),
            ("Swamp", "#556b2f"),
            ("Coastal Dunes", "#f4a460"),
            ("Badlands", "#a0522d")
        ]

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

        super_biomes = [
            ("Ocean", "#0077be"),
            ("Lake", "#4da6ff"),
            ("Grand River", "#0066cc"),
            ("River", "#3399ff"),
            ("Creek", "#66b3ff"),
            ("Cliff", "#696969"),
            ("Beach", "#f5deb3"),
            ("Lake Edge", "#87ceeb"),
            ("River Bank", "#98fb98"),
            ("Snow Level", "#fffafa"),
            ("Alpine Level", "#d3d3d3")
        ]

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
    Funktionsweise: Widget mit Gradient-Background für Main Menu
    Aufgabe: Visuell ansprechender Hintergrund
    """

    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        """Override für Custom Background Painting"""
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
    Aufgabe: Previous/Next Buttons, Tab-Jump, Workflow-Progress
    Kommunikation: Signals für Navigation-Requests
    """

    navigation_requested = pyqtSignal(str)  # (target_tab)

    def __init__(self, navigation_manager, show_tab_buttons=True, parent=None):
        super().__init__("Navigation", parent)
        self.navigation_manager = navigation_manager
        self.show_tab_buttons = show_tab_buttons
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

        # Tab-Jump Buttons (optional)
        if self.show_tab_buttons:
            jump_group = QGroupBox("Jump To")
            jump_layout = QGridLayout()

            self.tab_buttons = {}
            tabs = ["terrain", "geology", "weather", "water", "biome", "settlement", "overview"]

            for i, tab_name in enumerate(tabs):
                button = BaseButton(tab_name.title(), "secondary")
                button.clicked.connect(lambda checked, tab=tab_name: self.jump_to_tab(tab))
                self.tab_buttons[tab_name] = button

                row, col = divmod(i, 2)
                jump_layout.addWidget(button, row, col)

            jump_group.setLayout(jump_layout)
            layout.addWidget(jump_group)

        self.setLayout(layout)

    def set_current_tab(self, tab_name: str):
        """
        Funktionsweise: Setzt aktuellen Tab und aktualisiert Button-States
        Parameter: tab_name (str)
        """
        self.current_tab = tab_name
        self.update_navigation_buttons()
        self.update_tab_buttons()

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

    def update_tab_buttons(self):
        """Aktualisiert Tab-Jump Button States"""
        if not self.show_tab_buttons:
            return

        for tab_name, button in self.tab_buttons.items():
            if tab_name == self.current_tab:
                button.button_type = "primary"
                button.setup_styling()
            else:
                button.button_type = "secondary"
                button.setup_styling()

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

    def jump_to_tab(self, tab_name: str):
        """Jump zu spezifischem Tab"""
        self.navigation_requested.emit(tab_name)