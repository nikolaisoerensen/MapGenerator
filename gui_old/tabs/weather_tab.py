#!/usr/bin/env python3
"""
Weather Tab - 3D Wind Vectors über Terrain
VERVOLLSTÄNDIGT - Ersetzt die abgebrochene weather_tab.py
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QSpacerItem, QSizePolicy,
    QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt

from gui_old.widgets.map_canvas import WeatherDualCanvas
from gui_old.widgets.parameter_slider import ParameterSlider
from gui_old.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui_old.utils.performance_utils import debounced_method, performance_tracked
from gui_old.managers.parameter_manager import WorldParameterManager


class WeatherControlPanel(QWidget, NavigationMixin):
    """Weather Control Panel mit 3D Wind Vectors"""

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.world_manager = WorldParameterManager()
        self.weather_manager = self.world_manager.weather

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Weather & 3D Wind Fields")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #e67e22;")
        layout.addWidget(title)

        # Core Weather Status
        try:
            from core.weather_generator import RainGenerator, TemperatureGenerator
            status_text = "✓ Core Weather: Aktiv (Hillshade + Wind)"
            status_color = "#27ae60"
        except ImportError:
            status_text = "⚠ Core Weather: Fallback"
            status_color = "#e74c3c"

        status_label = QLabel(status_text)
        status_label.setStyleSheet(f"color: {status_color}; font-weight: bold; margin: 5px;")
        layout.addWidget(status_label)

        # Input Status
        self.input_status_label = QLabel("Warte auf Terrain-Daten...")
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

        # Auto-Simulation
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_precipitation_parameters(layout)
        self.setup_wind_parameters(layout)
        self.setup_temperature_parameters(layout)

        # Klima-Klassifikation
        self.setup_climate_info(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="← Settlement", next_text="Water →")

        self.setLayout(layout)

    def setup_simulation_controls(self, layout):
        """Auto-Simulation Controls"""
        sim_control_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Auto-Simulation")
        self.auto_simulate_checkbox.setChecked(
            WorldParameterManager().ui_state.get_auto_simulate()
        )
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Simulieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.simulate_now_btn.clicked.connect(self.simulate_now)
        self.simulate_now_btn.setEnabled(not self.world_manager.ui_state.get_auto_simulate())

        sim_control_widget = QWidget()
        sim_control_layout.addWidget(self.auto_simulate_checkbox)
        sim_control_layout.addWidget(self.simulate_now_btn)
        sim_control_widget.setLayout(sim_control_layout)
        layout.addWidget(sim_control_widget)

    def setup_precipitation_parameters(self, layout):
        """Niederschlag-Parameter"""
        rain_group = QGroupBox("Niederschlagssystem")
        rain_layout = QVBoxLayout()

        params = self.weather_manager.get_parameters()

        self.max_humidity_slider = ParameterSlider("Max. Luftfeuchtigkeit", 30, 100,
                                                   params['max_humidity'], suffix="%")
        self.rain_amount_slider = ParameterSlider("Regenmenge", 1.0, 10.0,
                                                  params['rain_amount'], decimals=1)
        self.evaporation_slider = ParameterSlider("Verdunstung", 0.0, 5.0,
                                                  params['evaporation'], decimals=1)

        rain_sliders = [self.max_humidity_slider, self.rain_amount_slider, self.evaporation_slider]

        for slider in rain_sliders:
            rain_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        rain_group.setLayout(rain_layout)
        layout.addWidget(rain_group)

    def setup_wind_parameters(self, layout):
        """Wind-Parameter für 3D Vektoren"""
        wind_group = QGroupBox("Windsystem (3D Vektoren)")
        wind_layout = QVBoxLayout()

        params = self.weather_manager.get_parameters()

        self.wind_speed_slider = ParameterSlider("Windgeschwindigkeit", 0.0, 20.0,
                                                 params['wind_speed'], decimals=1, suffix="m/s")
        self.wind_terrain_slider = ParameterSlider("Wind-Terrain Einfluss", 0.0, 10.0,
                                                   params['wind_terrain_influence'], decimals=1)

        # Neue 3D Wind Parameter
        self.wind_height_slider = ParameterSlider("Wind-Höhe über Terrain", 2.0, 20.0, 5.0, decimals=1, suffix="m")
        self.wind_density_slider = ParameterSlider("Vektor-Dichte", 0.5, 3.0, 1.0, decimals=1)

        wind_sliders = [self.wind_speed_slider, self.wind_terrain_slider,
                        self.wind_height_slider, self.wind_density_slider]

        for slider in wind_sliders:
            wind_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        wind_group.setLayout(wind_layout)
        layout.addWidget(wind_group)

    def setup_temperature_parameters(self, layout):
        """Temperatur-Parameter"""
        temp_group = QGroupBox("Temperatursystem")
        temp_layout = QVBoxLayout()

        params = self.weather_manager.get_parameters()

        self.avg_temperature_slider = ParameterSlider("Durchschnittstemperatur", -10, 40,
                                                      params['avg_temperature'], suffix="°C")

        temp_layout.addWidget(self.avg_temperature_slider)
        self.avg_temperature_slider.valueChanged.connect(self.on_parameter_changed)

        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)

    def setup_climate_info(self, layout):
        """Klima-Klassifikations Panel"""
        climate_group = QGroupBox("Live Klima-Klassifikation")
        climate_layout = QVBoxLayout()

        self.climate_label = QLabel("Gemäßigt")
        self.climate_label.setStyleSheet("""
            QLabel {
                background-color: #e8f4f8;
                border: 2px solid #3498db;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        self.climate_label.setAlignment(Qt.AlignCenter)

        climate_layout.addWidget(self.climate_label)
        climate_group.setLayout(climate_layout)
        layout.addWidget(climate_group)

    def on_auto_simulate_changed(self, state):
        """Auto-Simulation geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(300)  # Längeres Debouncing für Weather
    def on_parameter_changed(self):
        """Parameter geändert mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

        # Klima-Klassifikation sofort aktualisieren
        self.update_climate_classification()

    @performance_tracked("Weather_Preview_Update")
    def update_preview(self):
        """Aktualisiert Weather Preview"""
        params = self.get_parameters()

        # Parameter validieren und speichern
        self.weather_manager.set_parameters(params)

            # Input Status prüfen
        if not self._has_terrain_input():
            self.input_status_label.setText("⚠ Keine Terrain-Daten für Hillshade verfügbar")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    border: 2px solid #dc3545;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)
            # Kann trotzdem fortfahren mit Fallback

        else:
            # Update Input Status
            self.input_status_label.setText("✓ Terrain-Daten verfügbar - Hillshade + 3D Wind aktiv")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 2px solid #28a745;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)

        # Canvas Update
        self.map_canvas.update_map(**params)

    def _has_terrain_input(self):
        """Prüft ob Terrain-Input verfügbar ist"""
        return hasattr(self.map_canvas, 'input_data') and 'heightmap' in self.map_canvas.input_data

    def update_climate_classification(self):
        """Aktualisiert Klima-Klassifikation in Echtzeit"""
        params = self.get_parameters()
        climate = self.weather_manager.get_climate_classification()

        # Klima-Farben
        climate_colors = {
            'Tropisch': '#e74c3c',
            'Subtropisch': '#f39c12',
            'Gemäßigt': '#27ae60',
            'Kontinental': '#8e44ad',
            'Polar': '#3498db',
            'Arid': '#f1c40f',
            'Mediterran': '#e67e22'
        }

        color = climate_colors.get(climate, '#95a5a6')

        self.climate_label.setText(climate)
        self.climate_label.setStyleSheet(f"""
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
        """Sammelt alle Weather Parameter"""
        return {
            'max_humidity': self.max_humidity_slider.get_value(),
            'rain_amount': self.rain_amount_slider.get_value(),
            'evaporation': self.evaporation_slider.get_value(),
            'wind_speed': self.wind_speed_slider.get_value(),
            'wind_terrain_influence': self.wind_terrain_slider.get_value(),
            'avg_temperature': self.avg_temperature_slider.get_value(),
            # Neue 3D Wind Parameter
            'wind_height': self.wind_height_slider.get_value(),
            'wind_density': self.wind_density_slider.get_value()
        }

    # Navigation Methods
    def next_menu(self):
        """Wechselt zu Water Tab"""
        params = self.get_parameters()
        self.weather_manager.set_parameters(params)

        next_tab = TabNavigationHelper.get_next_tab('WeatherWindow')
        if next_tab:
            self.navigate_to_tab(next_tab[0], next_tab[1])

    def prev_menu(self):
        """Zurück zu Settlement Tab"""
        params = self.get_parameters()
        self.weather_manager.set_parameters(params)

        prev_tab = TabNavigationHelper.get_prev_tab('WeatherWindow')
        if prev_tab:
            self.navigate_to_tab(prev_tab[0], prev_tab[1])

    def quick_generate(self):
        """Schnellgenerierung"""
        params = self.get_parameters()
        self.weather_manager.set_parameters(params)


class WeatherWindow(QMainWindow):
    """Weather Tab Hauptfenster"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Weather & 3D Wind Fields")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Linke Seite: 3D Weather Canvas (70%)
        self.map_canvas = WeatherDualCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite: Controls (30%)
        self.control_panel = WeatherControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Weather-spezifisches Styling
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
            }
            QGroupBox[title*="Niederschlag"] {
                border-color: #3498db;
                background-color: rgba(52, 152, 219, 0.05);
            }
            QGroupBox[title*="Wind"] {
                border-color: #e67e22;
                background-color: rgba(230, 126, 34, 0.05);
            }
            QGroupBox[title*="Temperatur"] {
                border-color: #e74c3c;
                background-color: rgba(231, 76, 60, 0.05);
            }
            QGroupBox[title*="Klima"] {
                border-color: #27ae60;
                background-color: rgba(39, 174, 96, 0.05);
            }
        """)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        if hasattr(self, 'map_canvas'):
            self.map_canvas.cleanup()
        super().closeEvent(event)