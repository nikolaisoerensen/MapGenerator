#!/usr/bin/env python3
"""
Path: MapGenerator/gui/tabs/weather_tab.py
__init__.py existiert in "tabs"

World Generator GUI - Weather Tab (Vollständig Refactored)
Tab 4: Temperatur und Regensystem
Alle Verbesserungen aus Schritt 1-3 implementiert
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QPushButton,
                             QFrame, QSpacerItem, QSizePolicy,
                             QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt

from gui.widgets.parameter_slider import ParameterSlider
from gui.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui.utils.error_handler import ErrorHandler, safe_execute, TabErrorContext
from gui.widgets.map_canvas import MultiPlotCanvas
from gui.utils.performance_utils import debounced_method, performance_tracked
try:
    from gui.managers.parameter_manager import WorldParameterManager
except ImportError:
    from gui.world_state import WorldState as WorldParameterManager

class WeatherMapCanvas(MultiPlotCanvas):
    """
    Funktionsweise: Weather Map Canvas mit Multi-Plot Optimierung
    - Erbt von OptimizedMultiPlotCanvas für 2 Subplots
    - Performance-optimiert für komplexe Wetter-Visualisierung
    - Intelligente Colorbar-Verwaltung
    """

    def __init__(self):
        super().__init__('weather_maps', subplot_config=(1, 2),
                         figsize=(14, 6), titles=['Temperatur', 'Niederschlag'])
        self.temp_ax = None
        self.rain_ax = None

    def setup_axes(self):
        """Überschreibt setup_axes für Weather-spezifische Konfiguration"""
        super().setup_axes()
        if len(self.axes) >= 2:
            self.temp_ax = self.axes[0]
            self.rain_ax = self.axes[1]

            # Spezielle Konfiguration für Wetter-Achsen
            for ax in self.axes:
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_xlabel('X (West ← → Ost)')
                ax.set_ylabel('Y (Süd ← → Nord)')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)

    @performance_tracked("Weather_Rendering")
    def _render_map(self, max_humidity, rain_amount, evaporation,
                     wind_speed, wind_terrain_influence, avg_temperature):
        """
        Funktionsweise: Rendert beide Wetter-Karten mit Error Handling
        - Temperatur- und Niederschlagskarte parallel
        - Robuste Fehlerbehandlung für komplexe Berechnungen
        """
        with TabErrorContext('Weather', 'Dual Map Rendering'):
            self.clear_and_setup()

            if not self.temp_ax or not self.rain_ax:
                self.error_handler.logger.error("Weather axes nicht korrekt initialisiert")
                return

            # Koordinaten-Mesh generieren
            x = np.linspace(0, 100, 50)
            y = np.linspace(0, 100, 50)
            X, Y = np.meshgrid(x, y)

            # === TEMPERATUR-KARTE ===
            self._render_temperature_map(X, Y, avg_temperature, wind_speed, wind_terrain_influence)

            # === NIEDERSCHLAGS-KARTE ===
            self._render_precipitation_map(X, Y, max_humidity, rain_amount, evaporation,
                                           wind_speed, wind_terrain_influence)

            # Layout optimieren
            self.figure.tight_layout()
            self.draw()

    def _render_temperature_map(self, X, Y, avg_temperature, wind_speed, wind_terrain_influence):
        """
        Funktionsweise: Rendert Temperatur-Karte
        - Nord-Süd Temperaturgradient
        - Höhen- und Wind-Einflüsse
        """
        try:
            # Basis-Temperatur mit Nord-Süd Gradient
            base_temp = avg_temperature / 30.0  # Normalisiert auf 0-1
            lat_gradient = np.linspace(0.3, 1.0, 50)  # Kälter im Norden
            temp_map = np.zeros_like(X)

            for i in range(50):
                temp_map[i, :] = base_temp * lat_gradient[i]

            # Höhenbasierte Abkühlung (simuliert)
            height_effect = np.sin(X * 0.1) * np.cos(Y * 0.1) * 0.3
            temp_map -= height_effect

            # Wind-Einfluss auf Temperatur
            wind_effect = np.sin(X * 0.05 + wind_speed * 0.1) * wind_terrain_influence * 0.02
            temp_map += wind_effect

            # Temperatur-Plot mit robuster Colorbar
            temp_contour = self.temp_ax.contourf(X, Y, temp_map, levels=15, cmap='RdYlBu_r', alpha=0.8)
            self.temp_ax.contour(X, Y, temp_map, levels=10, colors='black', alpha=0.4, linewidths=0.5)

            # Temperatur-spezifische Labels
            temp_range = f"{avg_temperature - 15:.0f}°C bis {avg_temperature + 15:.0f}°C"
            self.temp_ax.set_title(f'Temperatur ({temp_range})')

            # Colorbar mit Fehlerbehandlung
            try:
                cbar = self.figure.colorbar(temp_contour, ax=self.temp_ax, fraction=0.046, pad=0.04)
                cbar.set_label('Relative Temperatur', rotation=270, labelpad=15)
            except Exception as cb_error:
                self.error_handler.logger.warning(f"Temperatur Colorbar Fehler: {cb_error}")

        except Exception as e:
            self.error_handler.logger.error(f"Temperatur-Rendering Fehler: {e}")
            self.temp_ax.text(0.5, 0.5, 'Temperatur\n(Rendering-Fehler)',
                              transform=self.temp_ax.transAxes, ha='center', va='center')

    def _render_precipitation_map(self, X, Y, max_humidity, rain_amount, evaporation,
                                  wind_speed, wind_terrain_influence):
        """
        Funktionsweise: Rendert Niederschlags-Karte
        - Orographische Niederschläge
        - Wind-Transport und Verdunstung
        - Wind-Vektoren
        """
        try:
            # Orographische Niederschläge (Windrichtung West-Ost)
            oro_rain = np.zeros_like(X)

            for i in range(50):
                # Feuchte Luft von Westen
                moisture = max_humidity * (1.0 - i / 50.0 * 0.7) / 100.0

                # Terrain-Einfluss (Steigungsregen)
                terrain_lift = np.sin(Y[:, i] * 0.08) * wind_terrain_influence / 10.0
                oro_rain[:, i] = moisture * (0.5 + terrain_lift) * rain_amount / 100.0

            # Wind-Transport
            wind_transport = np.sin(X * 0.03 + wind_speed * 0.05) * np.cos(Y * 0.04)
            oro_rain += wind_transport * rain_amount * 0.003

            # Verdunstungs-Einfluss
            evap_effect = np.sin(X * 0.15) * np.cos(Y * 0.15) * evaporation / 10.0
            oro_rain -= evap_effect * 0.02
            oro_rain = np.clip(oro_rain, 0, None)

            # Niederschlag-Plot
            rain_contour = self.rain_ax.contourf(X, Y, oro_rain, levels=15, cmap='Blues', alpha=0.8)
            self.rain_ax.contour(X, Y, oro_rain, levels=8, colors='darkblue', alpha=0.6, linewidths=0.5)

            # Windvektoren hinzufügen
            self._add_wind_vectors(wind_speed, wind_terrain_influence)

            # Niederschlag-spezifische Labels
            climate = self._get_climate_info(max_humidity, rain_amount)
            self.rain_ax.set_title(f'Niederschlag ({climate})')

            # Colorbar mit Fehlerbehandlung
            try:
                cbar = self.figure.colorbar(rain_contour, ax=self.rain_ax, fraction=0.046, pad=0.04)
                cbar.set_label('Niederschlag (rel.)', rotation=270, labelpad=15)
            except Exception as cb_error:
                self.error_handler.logger.warning(f"Niederschlag Colorbar Fehler: {cb_error}")

        except Exception as e:
            self.error_handler.logger.error(f"Niederschlag-Rendering Fehler: {e}")
            self.rain_ax.text(0.5, 0.5, 'Niederschlag\n(Rendering-Fehler)',
                              transform=self.rain_ax.transAxes, ha='center', va='center')

    def _add_wind_vectors(self, wind_speed, wind_terrain_influence):
        """Fügt Wind-Vektoren zur Niederschlagskarte hinzu"""
        try:
            step = 8
            X_wind = np.arange(0, 100, step)
            Y_wind = np.arange(0, 100, step)
            X_wind, Y_wind = np.meshgrid(X_wind, Y_wind)

            # Wind-Richtung und -Stärke
            wind_x = np.cos(wind_speed * 0.1) * wind_speed / 100.0
            wind_y = np.sin(wind_speed * 0.1) * wind_speed / 200.0

            U_wind = np.full_like(X_wind, wind_x)
            V_wind = np.full_like(Y_wind, wind_y)

            self.rain_ax.quiver(X_wind, Y_wind, U_wind, V_wind,
                                alpha=0.7, color='darkgreen', scale=0.1, width=0.003)
        except Exception as e:
            self.error_handler.logger.warning(f"Wind-Vektoren Fehler: {e}")

    def _get_climate_info(self, max_humidity, rain_amount):
        """Gibt Klima-Information basierend auf Parametern zurück"""
        if max_humidity > 80 and rain_amount > 7:
            return "Sehr feucht"
        elif max_humidity > 60 and rain_amount > 5:
            return "Gemäßigt"
        elif max_humidity < 40 or rain_amount < 3:
            return "Trocken"
        else:
            return "Moderat"


class WeatherControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Weather Control Panel mit allen Verbesserungen
    - Neue ParameterSlider (Schritt 1)
    - WorldParameterManager Integration (Schritt 2)
    - Performance-Optimierung mit Debouncing (Schritt 3)
    - Klima-Klassifizierung und intelligente Validierung
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.error_handler = ErrorHandler()

        # Verwende neuen WorldParameterManager
        self.world_manager = WorldParameterManager()
        self.weather_manager = self.world_manager.weather

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Temperatur & Regen")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Auto-Simulation Control
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_precipitation_parameters(layout)
        self.setup_wind_parameters(layout)
        self.setup_temperature_parameters(layout)

        # Klima-Information
        self.setup_climate_info(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation mit Mixin
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="Zurück", next_text="Weiter")

        self.setLayout(layout)

        # Initial preview
        self.update_preview()

    def setup_simulation_controls(self, layout):
        """Erstellt Auto-Simulation Controls"""
        sim_control_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Automat. Simulieren")
        self.auto_simulate_checkbox.setChecked(self.world_manager.ui_state.get_auto_simulate())
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
        """Erstellt Niederschlag-Parameter"""
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
        """Erstellt Wind-Parameter"""
        wind_group = QGroupBox("Windsystem")
        wind_layout = QVBoxLayout()

        params = self.weather_manager.get_parameters()

        self.wind_speed_slider = ParameterSlider("Windgeschwindigkeit", 0.0, 20.0,
                                                 params['wind_speed'], decimals=1, suffix="m/s")
        self.wind_terrain_slider = ParameterSlider("Wind-Terrain Einfluss", 0.0, 10.0,
                                                   params['wind_terrain_influence'], decimals=1)

        wind_sliders = [self.wind_speed_slider, self.wind_terrain_slider]

        for slider in wind_sliders:
            wind_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        wind_group.setLayout(wind_layout)
        layout.addWidget(wind_group)

    def setup_temperature_parameters(self, layout):
        """Erstellt Temperatur-Parameter"""
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
        """Erstellt Klima-Informations-Panel"""
        climate_group = QGroupBox("Klima-Klassifikation")
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
        """Auto-Simulation Checkbox geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.error_handler.logger.info("Weather Auto-Simulation aktiviert")
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.error_handler.logger.info("Weather Auto-Simulation deaktiviert")
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(300)  # Längeres Debouncing für komplexe Weather-Berechnungen
    def on_parameter_changed(self):
        """Parameter wurden geändert - mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

        # Aktualisiere Klima-Klassifikation sofort (ohne Debouncing)
        self.update_climate_classification()

    def simulate_now(self):
        """Manuelle Simulation"""
        self.error_handler.logger.info("Weather Simulation gestartet!")
        self.update_preview()

    @performance_tracked("Weather_Preview_Update")
    def update_preview(self):
        """
        Funktionsweise: Aktualisiert die Wetter-Kartenvorschau
        - Performance-optimiert für Dual-Map Rendering
        - Validiert komplexe Wetter-Parameter
        """
        with TabErrorContext('Weather', 'Preview Update'):
            params = self.get_parameters()

            # Validiere und speichere Parameter
            self.weather_manager.set_parameters(params)

            # Aktualisiere Klima-Klassifikation
            self.update_climate_classification()

            # Aktualisiere Karten (mit automatischem Debouncing)
            self.map_canvas.update_map(**params)

    def update_climate_classification(self):
        """
        Funktionsweise: Aktualisiert Klima-Klassifikation in Echtzeit
        - Basiert auf aktuellen Parametern
        - Visuelles Feedback für User
        """
        try:
            params = self.get_parameters()
            self.weather_manager.set_parameters(params)

            # Hole Klima-Klassifikation vom Manager
            climate = self.weather_manager.get_climate_classification()

            # Aktualisiere Label mit passender Farbe
            climate_colors = {
                "Tropisch": "#e74c3c",
                "Gemäßigt": "#27ae60",
                "Kalt": "#3498db",
                "Trocken": "#f39c12"
            }

            color = climate_colors.get(climate, "#95a5a6")
            self.climate_label.setText(f"🌡️ {climate}")
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

        except Exception as e:
            self.error_handler.logger.warning(f"Klima-Klassifikation Update Fehler: {e}")

    def get_parameters(self):
        """
        Funktionsweise: Sammelt alle Weather Parameter
        - Verwendet neue ParameterSlider API
        - Robuste Parameter-Sammlung
        """
        try:
            return {
                'max_humidity': self.max_humidity_slider.get_value(),
                'rain_amount': self.rain_amount_slider.get_value(),
                'evaporation': self.evaporation_slider.get_value(),
                'wind_speed': self.wind_speed_slider.get_value(),
                'wind_terrain_influence': self.wind_terrain_slider.get_value(),
                'avg_temperature': self.avg_temperature_slider.get_value()
            }
        except Exception as e:
            self.error_handler.handle_parameter_error('Weather', 'parameter_collection', e)
            return self.weather_manager.get_parameters()

    # Navigation Methoden (von NavigationMixin erforderlich)
    def next_menu(self):
        """Wechselt zum nächsten Tab (Water)"""
        try:
            params = self.get_parameters()
            self.weather_manager.set_parameters(params)
            self.error_handler.logger.info("Weather Parameter gespeichert")

            next_tab = TabNavigationHelper.get_next_tab('WeatherWindow')
            if next_tab:
                self.navigate_to_tab(next_tab[0], next_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Weather', 'Water', e)

    def prev_menu(self):
        """Wechselt zum vorherigen Tab (Settlement)"""
        try:
            params = self.get_parameters()
            self.weather_manager.set_parameters(params)

            prev_tab = TabNavigationHelper.get_prev_tab('WeatherWindow')
            if prev_tab:
                self.navigate_to_tab(prev_tab[0], prev_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Weather', 'Settlement', e)

    def quick_generate(self):
        """Schnellgenerierung mit Klima-Info"""
        params = self.get_parameters()
        self.weather_manager.set_parameters(params)
        climate = self.weather_manager.get_climate_classification()
        self.error_handler.logger.info(f"Weather Schnellgenerierung: {climate} - {params}")


class WeatherWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für Weather-Tab
    - Verwendet optimierte Multi-Plot Canvas
    - Erweiterte Fenster-Konfiguration für Dual-Maps
    """

    def __init__(self):
        super().__init__()
        self.error_handler = ErrorHandler()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Temperatur & Niederschlag")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Linke Seite - Optimierte Multi-Plot Karte (75%)
        self.map_canvas = WeatherMapCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Controls (25%)
        self.control_panel = WeatherControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Erweiterte Styling für Weather
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            /* Spezielle Styles für Weather Parameter */
            QGroupBox[title="Niederschlagssystem"] {
                border-color: #3498db;
            }
            QGroupBox[title="Windsystem"] {
                border-color: #27ae60;
            }
            QGroupBox[title="Temperatursystem"] {
                border-color: #e74c3c;
            }
            QGroupBox[title="Klima-Klassifikation"] {
                border-color: #9b59b6;
                background-color: #f8f9fa;
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