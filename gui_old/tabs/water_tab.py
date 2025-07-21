#!/usr/bin/env python3
"""
Path: MapGenerator/gui/tabs/water_tab.py
__init__.py existiert in "tabs"

World Generator GUI - Water Tab (Vollständig Refactored)
Tab 5: Flüsse, Seen und Wasserflächen
Alle Verbesserungen aus Schritt 1-3 implementiert
ANGEPASST: Verwendet jetzt WaterDualCanvas statt MapCanvas
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QPushButton,
                             QFrame, QSpacerItem, QSizePolicy,
                             QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt

from gui_old.widgets.parameter_slider import ParameterSlider
from gui_old.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui_old.widgets.map_canvas import WaterDualCanvas
from gui_old.utils.performance_utils import debounced_method, performance_tracked
from gui_old.managers.parameter_manager import WorldParameterManager


class WaterControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Water Control Panel mit allen Verbesserungen
    - Neue ParameterSlider (Schritt 1)
    - WorldParameterManager Integration (Schritt 2)
    - Performance-Optimierung mit Debouncing (Schritt 3)
    - Wasser-Coverage Berechnung und Validierung
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas

        # Verwende neuen WorldParameterManager
        self.world_manager = WorldParameterManager()
        self.water_manager = self.world_manager.water

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Flüsse & Wasserflächen")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #1e40af;")
        layout.addWidget(title)

        # Core Water Status
        try:
            from core.water_generator import RiverSimulator
            status_text = "✓ Core Water: Aktiv (Flow + 3D Systems)"
            status_color = "#27ae60"
        except ImportError:
            status_text = "⚠ Core Water: Fallback"
            status_color = "#e74c3c"

        status_label = QLabel(status_text)
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; margin: 5px;")
        layout.addWidget(status_label)

        # Input Status
        self.input_status_label = QLabel("Warte auf Terrain + Weather Daten...")
        self.input_status_label.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                border: 2px solid #ffeaa7;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.input_status_label)

        # Auto-Simulation Control
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_water_body_parameters(layout)
        self.setup_erosion_parameters(layout)

        # Wasser-Coverage Info
        self.setup_water_coverage_info(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation mit Mixin
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="← Weather", next_text="Biome →")

        self.setLayout(layout)

        # Initial preview
        self.update_preview()

    def setup_simulation_controls(self, layout):
        """Erstellt Auto-Simulation Controls"""
        sim_control_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Auto-Simulation")
        self.auto_simulate_checkbox.setChecked(
            WorldParameterManager().ui_state.get_auto_simulate()
        )
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Simulieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.simulate_now_btn.clicked.connect(self.update_preview())
        self.simulate_now_btn.setEnabled(not self.world_manager.ui_state.get_auto_simulate())

        sim_control_widget = QWidget()
        sim_control_layout.addWidget(self.auto_simulate_checkbox)
        sim_control_layout.addWidget(self.simulate_now_btn)
        sim_control_widget.setLayout(sim_control_layout)
        layout.addWidget(sim_control_widget)

    def setup_water_body_parameters(self, layout):
        """Erstellt Wasserkörper-Parameter"""
        water_group = QGroupBox("Wasserkörper")
        water_layout = QVBoxLayout()

        params = self.water_manager.get_parameters()

        self.lake_fill_slider = ParameterSlider("Seefüllung", 0, 100,
                                                params['lake_fill'], suffix="%")
        self.sea_level_slider = ParameterSlider("Meeresspiegel", 0, 50,
                                                params['sea_level'], suffix="%")

        water_sliders = [self.lake_fill_slider, self.sea_level_slider]

        for slider in water_sliders:
            water_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        water_group.setLayout(water_layout)
        layout.addWidget(water_group)

    def setup_erosion_parameters(self, layout):
        """Erstellt Tal-Erosion Parameter"""
        erosion_group = QGroupBox("Tal-Erosion")
        erosion_layout = QVBoxLayout()

        params = self.water_manager.get_parameters()

        self.sediment_slider = ParameterSlider("Sedimentmenge", 0, 100,
                                               params['sediment_amount'], suffix="%")
        self.water_speed_slider = ParameterSlider("Wassergeschwindigkeit", 1.0, 20.0,
                                                  params['water_speed'], decimals=1, suffix="m/s")
        self.rock_dependency_slider = ParameterSlider("Gesteinsabhängigkeit", 0, 100,
                                                      params['rock_dependency'], suffix="%")

        erosion_sliders = [self.sediment_slider, self.water_speed_slider,
                           self.rock_dependency_slider]

        for slider in erosion_sliders:
            erosion_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        erosion_group.setLayout(erosion_layout)
        layout.addWidget(erosion_group)

    def setup_water_coverage_info(self, layout):
        """Erstellt Wasser-Coverage Informations-Panel"""
        coverage_group = QGroupBox("Wasser-Coverage")
        coverage_layout = QVBoxLayout()

        self.coverage_label = QLabel("Berechne...")
        self.coverage_label.setStyleSheet("""
            QLabel {
                background-color: #e8f6ff;
                border: 2px solid #1e40af;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        self.coverage_label.setAlignment(Qt.AlignCenter)

        coverage_layout.addWidget(self.coverage_label)
        coverage_group.setLayout(coverage_layout)
        layout.addWidget(coverage_group)

    def on_auto_simulate_changed(self, state):
        """Auto-Simulation Checkbox geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(250)  # Debouncing für Water-Berechnungen
    def on_parameter_changed(self):
        """Parameter wurden geändert - mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

        # Aktualisiere Water-Coverage sofort (ohne Debouncing)
        self.update_water_coverage()


    def simulate_now(self):
        """Manuelle Simulation"""
        self.update_preview()

    @performance_tracked("Water_Preview_Update")
    def update_preview(self):
        """
        Funktionsweise: Aktualisiert die Wasser-Kartenvorschau
        - Performance-optimiert für komplexe Wassersystem-Berechnungen
        - Validiert Wasser-Parameter
        """
        params = self.get_parameters()

        # Validiere und speichere Parameter
        self.water_manager.set_parameters(params)

        # Input Status prüfen
        missing_inputs = []
        if not self._has_heightmap_input():
            missing_inputs.append("Heightmap")
        if not self._has_weather_input():
            missing_inputs.append("Weather")

        if missing_inputs:
            self.input_status_label.setText(f"⚠ Fehlende Daten: {', '.join(missing_inputs)}")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    border: 2px solid #dc3545;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)
        else:
            self.input_status_label.setText("✓ Terrain + Weather verfügbar - 3D Water Systems aktiv")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 2px solid #28a745;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)

        # Aktualisiere Water-Coverage
        self.update_water_coverage()

        # Aktualisiere Karte (mit automatischem Debouncing)
        self.map_canvas.update_map(**params)

    def _has_heightmap_input(self):
        """Prüft ob Heightmap verfügbar ist"""
        return hasattr(self.map_canvas, 'input_data') and 'heightmap' in self.map_canvas.input_data

    def _has_weather_input(self):
        """Prüft ob Weather-Daten verfügbar sind"""
        return hasattr(self.map_canvas, 'input_data') and 'weather' in self.map_canvas.input_data

    def update_water_coverage(self):
        """
        Funktionsweise: Aktualisiert Water-Coverage Berechnung in Echtzeit
        - Berechnet geschätzten Wasser-Anteil der Karte
        - Visuelles Feedback für realistische Werte
        """
        params = self.get_parameters()
        self.water_manager.set_parameters(params)

        # Hole Water-Coverage vom Manager
        coverage = self.water_manager.calculate_water_coverage()

        # Bewerte Coverage für Feedback
        if coverage < 20:
            coverage_type = "Trocken"
            color = "#e67e22"
        elif coverage < 40:
            coverage_type = "Moderat"
            color = "#f39c12"
        elif coverage < 60:
            coverage_type = "Ausgewogen"
            color = "#3498db"
        else:
            coverage_type = "Sehr Feucht"
            color = "#2980b9"

        self.coverage_label.setText(f"{coverage_type}\n{coverage:.1f}% Wasser")
        self.coverage_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color}20;
                border: 2px solid {color};
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
            }}
        """)

    def get_parameters(self):
        """
        Funktionsweise: Sammelt alle Water Parameter
        - Verwendet neue ParameterSlider API
        - Robuste Parameter-Sammlung
        """
        return {
            'lake_fill': self.lake_fill_slider.get_value(),
            'sea_level': self.sea_level_slider.get_value(),
            'sediment_amount': self.sediment_slider.get_value(),
            'water_speed': self.water_speed_slider.get_value(),
            'rock_dependency': self.rock_dependency_slider.get_value()
        }

    # Navigation Methoden (von NavigationMixin erforderlich)
    def next_menu(self):
        """Wechselt zum nächsten Tab (Biome)"""
        params = self.get_parameters()
        self.water_manager.set_parameters(params)

        next_tab = TabNavigationHelper.get_next_tab('WaterWindow')
        if next_tab:
            self.navigate_to_tab(next_tab[0], next_tab[1])

    def prev_menu(self):
        """Wechselt zum vorherigen Tab (Weather)"""
        params = self.get_parameters()
        self.water_manager.set_parameters(params)

        prev_tab = TabNavigationHelper.get_prev_tab('WaterWindow')
        if prev_tab:
            self.navigate_to_tab(prev_tab[0], prev_tab[1])

    def quick_generate(self):
        """Schnellgenerierung mit Water-Coverage Info"""
        params = self.get_parameters()
        self.water_manager.set_parameters(params)
        coverage = self.water_manager.calculate_water_coverage()


class WaterWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für Water-Tab
    - Verwendet optimierte WaterDualCanvas (GEÄNDERT)
    - Robuste Initialisierung für komplexes Wassersystem
    """

    def __init__(self):
        super().__init__()
        self.init_ui()


    def init_ui(self):
        self.setWindowTitle("World Generator - Flüsse & 3D Wassersysteme")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Linke Seite - WaterDualCanvas (70%) - GEÄNDERT
        self.map_canvas = WaterDualCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Controls (30%)
        self.control_panel = WaterControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Styling für Water-spezifische Elemente
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f8ff, stop:1 #e6f3ff);
            }
            QLabel {
                color: #333;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: rgba(255, 255, 255, 0.9);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            /* Spezielle Styles für Water Parameter */
            QGroupBox[title="Wasserkörper"] {
                border-color: #1e40af;
                background-color: rgba(30, 64, 175, 0.05);
            }
            QGroupBox[title="Tal-Erosion"] {
                border-color: #92400e;
                background-color: rgba(146, 64, 14, 0.05);
            }
            QGroupBox[title="Wasser-Coverage"] {
                border-color: #3b82f6;
                background-color: rgba(59, 130, 246, 0.05);
            }
        """)

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        if hasattr(self, 'map_canvas'):
            self.map_canvas.cleanup()
            super().closeEvent(event)