"""
Path: gui/tabs/weather_tab.py

WeatherTab implementiert die Weather-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den WeatherSystemGenerator aus core/weather_generator.py. Als von
Terrain abhängiger Generator (heightmap_combined, shadowmap) liefert er wind_map, temp_map,
precip_map und humid_map für Water und alle nachgelagerten Systeme.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QLabel
)
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from gui.config.value_default import get_parameter_config


class WeatherTab(BaseMapTab):
    """
    Weather-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert wind_map/temp_map/precip_map/humid_map Generation auf Basis der
    Terrain-Daten (heightmap_combined, shadowmap).
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "weather"
        self.required_dependencies = ["heightmap", "shadowmap"]

        # Weather-spezifische Attribute (vor super(), da create_parameter_controls
        # und create_visualization_controls während BaseMapTab.setup_ui() darauf
        # zugreifen und sie befüllen)
        self.parameter_sliders = {}
        self.climate_stats = None
        self.dependency_status = None
        self.gpu_status = None
        self.display_mode_group = None
        self.current_display_mode = "height"

        self.logger = logging.getLogger("WeatherTab")

        # Manager-Integration
        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )

        # Registrierung beim ParameterManager: parameter_sliders sind durch
        # create_parameter_controls() (innerhalb von super().__init__()) bereits
        # befüllt, get_current_parameters() liefert damit sofort die Default-Werte
        # als Startwert des Caches.
        if self.parameter_manager:
            self.parameter_manager.register_tab(self.generator_type, self)

        # Initialer Dependency-Check für die Status-Anzeige
        self.check_input_dependencies()

        self.logger.info("WeatherTab initialized")

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls für Weather-Generation.
        Implementiert Required-Method von BaseMapTab.
        Stellt bei abgetrenntem Panel-Layout ein neues Layout wieder her,
        damit die Parameter-Erstellung nie leer abbricht.
        """
        if not self.control_panel:
            self.logger.error("Parameter creation skipped: control_panel is None")
            return

        if self.control_panel.layout() is None:
            repaired_layout = QVBoxLayout()
            repaired_layout.setContentsMargins(5, 5, 5, 5)
            repaired_layout.setSpacing(10)
            self.control_panel.setLayout(repaired_layout)
            self.control_panel_content_layout = repaired_layout
            self.logger.info("Control panel layout was detached - reinstalled")

        try:
            self._create_temperature_parameters()
            self._create_wind_parameters()
            self._create_dependency_status()
            self._create_gpu_status()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_temperature_parameters(self):
        """Erstellt Temperature-System Parameter Controls"""
        temp_group = QGroupBox("Temperature System")
        temp_group.setFont(QFont("Arial", 10, QFont.Bold))
        temp_layout = QVBoxLayout()

        for param_key in ("air_temp_entry", "solar_power", "altitude_cooling"):
            config = get_parameter_config("weather", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 1),
                suffix=config.get("suffix", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            temp_layout.addWidget(slider)

        temp_group.setLayout(temp_layout)
        self.control_panel.layout().addWidget(temp_group)

    def _create_wind_parameters(self):
        """Erstellt Wind-System Parameter Controls"""
        wind_group = QGroupBox("Wind System")
        wind_group.setFont(QFont("Arial", 10, QFont.Bold))
        wind_layout = QVBoxLayout()

        for param_key in ("thermic_effect", "wind_speed_factor", "terrain_factor"):
            config = get_parameter_config("weather", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 0.1),
                suffix=config.get("suffix", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            wind_layout.addWidget(slider)

        wind_group.setLayout(wind_layout)
        self.control_panel.layout().addWidget(wind_group)

    def _create_dependency_status(self):
        """Erstellt Dependency-Status-Anzeige (Verfügbarkeit der Terrain-Inputs)"""
        self.dependency_status = StatusIndicator("Weather Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def _create_gpu_status(self):
        """
        Erstellt GPU-Status-Anzeige. Weather-CFD-Simulation läuft aktuell durchgehend
        über den CPU-Fallback (ShaderManager bietet keine Weather-spezifischen
        GPU-Compute-Shader an) - die Anzeige spiegelt das ehrlich wider.
        """
        self.gpu_status = StatusIndicator("GPU Status")
        if self.shader_manager and getattr(self.shader_manager, 'gpu_available', False):
            self.gpu_status.set_success("GPU available")
        else:
            self.gpu_status.set_warning("Using CPU fallback")
        self.control_panel.layout().addWidget(self.gpu_status)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Climate-Statistics (Parameter-Preview + Climate-Classification + Ergebnissen).
        """
        self.climate_stats = ClimateStatisticsWidget()
        layout.addWidget(self.climate_stats)

    def create_visualization_controls(self):
        """
        Erstellt Weather-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """Erstellt Height/Temperature/Precipitation/Humidity/Wind Display Mode Controls"""
        layout = QHBoxLayout()

        self.display_mode_group = QButtonGroup()

        modes = [
            ("height", "Height", 0),
            ("temp_map", "Temperature", 1),
            ("precip_map", "Precipitation", 2),
            ("humid_map", "Humidity", 3),
            ("wind_map", "Wind", 4),
        ]

        for mode_key, label, button_id in modes:
            radio = QRadioButton(label)
            if button_id == 0:
                radio.setChecked(True)
            radio.toggled.connect(lambda checked, key=mode_key: self._on_display_mode_changed(key, checked))
            self.display_mode_group.addButton(radio, button_id)
            layout.addWidget(radio)

        return layout

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            if self.climate_stats:
                self.climate_stats.update_parameter_preview(self.get_current_parameters())

            self.logger.debug(f"Parameter changed: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

    def _on_display_mode_changed(self, mode: str, checked: bool):
        """Handler für Display Mode Changes"""
        if checked:
            self.current_display_mode = mode
            self.update_display_mode()
            self.logger.debug(f"Display mode changed to: {mode}")

    # =============================================================================
    # DISPLAY UPDATE SYSTEM
    # =============================================================================

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab Display-Update für Weather-spezifische Modi.
        Implementiert Height/Temperature/Precipitation/Humidity/Wind Display-Switching.
        """
        try:
            if not self.data_lod_manager:
                return

            current_display = self.get_current_display()
            if not current_display:
                return

            if self.current_display_mode == "height":
                data = self.data_lod_manager.get_terrain_data("heightmap")
                data_type = "heightmap"
                display_data = data
            elif self.current_display_mode in ("temp_map", "precip_map", "humid_map", "wind_map"):
                data = self.data_lod_manager.get_weather_data(self.current_display_mode)
                data_type = self.current_display_mode
                if data_type == "wind_map" and data is not None and hasattr(data, 'shape') and len(data.shape) == 3:
                    # wind_map ist (H,W,2) u/v-Windkomponenten - MapDisplay2D
                    # kann nur echte 2D-Bilder zeichnen, daher auf die
                    # Windgeschwindigkeits-Magnitude reduzieren.
                    display_data = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2).astype(np.float32)
                else:
                    display_data = data
            else:
                return

            if data is not None and display_data is not None and hasattr(current_display, 'update_display'):
                display_id = f"WeatherTab_{self.current_view}_{data_type}"

                if hasattr(self.data_lod_manager, 'display_update_manager'):
                    needs_update = self.data_lod_manager.display_update_manager.needs_update(
                        display_id, data, data_type
                    )

                    if needs_update:
                        self._push_data_to_current_display(display_data, data_type)
                        self.data_lod_manager.display_update_manager.mark_updated(
                            display_id, data, data_type
                        )
                else:
                    self._push_data_to_current_display(display_data, data_type)

        except Exception as e:
            self.logger.debug(f"Weather display mode update failed: {e}")

    # =============================================================================
    # GENERATION
    # =============================================================================

    def generate(self):
        """
        Überschreibt BaseMapTab: Dependency-Check vor der eigentlichen Generation.
        Wird ausschließlich über den globalen [GENERIEREN]-Button im Shell-Footer
        ausgelöst (kein eigener Berechnen-Button mehr im Parameter-Panel).
        """
        try:
            if not self.check_input_dependencies():
                self.logger.warning("Input dependencies (heightmap/shadowmap) not met")
                return

            super().generate()

            self.logger.info("Weather generation requested")

        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """
        Überschreibt BaseMapTab Completion Handler für Weather-spezifische Completion.
        Aktualisiert Climate-Statistics nach erfolgreicher Generation.
        """
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type != self.generator_type:
            return

        try:
            if success:
                self.update_display_mode()

                if self.climate_stats:
                    results = {
                        "temp_map": self.data_lod_manager.get_weather_data("temp_map"),
                        "wind_map": self.data_lod_manager.get_weather_data("wind_map"),
                        "humid_map": self.data_lod_manager.get_weather_data("humid_map"),
                        "precip_map": self.data_lod_manager.get_weather_data("precip_map"),
                    }
                    if any(value is not None for value in results.values()):
                        self.climate_stats.update_generation_statistics(results)

            super().on_generation_completed(result_id, result_data)

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER SYNCHRONISATION
    # =============================================================================

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Werte aller Weather-Parameter-Slider.
        Wird vom ParameterManager als zentrale Quelle für die Weather-Parameter
        genutzt (register_tab()/get_tab_parameters() rufen diese Methode auf,
        siehe gui/OldManagers/parameter_manager.py).
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    def update_parameter_ui(self, param_name: str, value):
        """
        Überschreibt BaseMapTab Parameter-UI Update für Weather-Parameter.
        Synchronisiert UI-Controls mit ParameterManager-Updates.
        """
        try:
            if param_name in self.parameter_sliders:
                slider = self.parameter_sliders[param_name]
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

                self.logger.debug(f"Parameter UI updated: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter UI update failed: {e}")

    # =============================================================================
    # DEPENDENCY SYSTEM
    # =============================================================================

    def check_input_dependencies(self) -> bool:
        """
        Überschreibt BaseMapTab Dependency Check.
        Prüft ob die Terrain-Inputs (heightmap, shadowmap) verfügbar sind.
        """
        try:
            heightmap = self.data_lod_manager.get_terrain_data("heightmap")
            shadowmap = self.data_lod_manager.get_terrain_data("shadowmap")

            dependencies_met = heightmap is not None and shadowmap is not None

            if self.dependency_status:
                if dependencies_met:
                    self.dependency_status.set_success("Terrain inputs available")
                else:
                    missing = []
                    if heightmap is None:
                        missing.append("heightmap")
                    if shadowmap is None:
                        missing.append("shadowmap")
                    self.dependency_status.set_warning(f"Missing terrain data: {', '.join(missing)}")

            return dependencies_met

        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False

    # =============================================================================
    # RESOURCE MANAGEMENT
    # =============================================================================

    def cleanup_resources(self):
        """
        Erweitert BaseMapTab Cleanup für Weather-spezifische Resources.
        """
        try:
            self.logger.debug("Cleaning up weather-specific resources")

            self.parameter_sliders.clear()

            super().cleanup_resources()

        except Exception as e:
            self.logger.error(f"Weather cleanup failed: {e}")


class ClimateStatisticsWidget(QGroupBox):
    """
    Widget für Climate-Statistiken und Parameter-Preview.
    Zeigt Weather-Parameter, Climate-Classification und Generation-Results.
    """

    def __init__(self):
        super().__init__("Climate Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Climate-Statistiken"""
        layout = QVBoxLayout()

        preview_group = QGroupBox("Climate Parameters")
        preview_layout = QVBoxLayout()

        self.base_temp_label = QLabel("Base Temperature: 15°C")
        self.solar_power_label = QLabel("Solar Power: 20°C")
        self.altitude_cooling_label = QLabel("Altitude Cooling: 6°C/100m")
        self.wind_factor_label = QLabel("Wind Factor: 1.0")

        preview_layout.addWidget(self.base_temp_label)
        preview_layout.addWidget(self.solar_power_label)
        preview_layout.addWidget(self.altitude_cooling_label)
        preview_layout.addWidget(self.wind_factor_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        classification_group = QGroupBox("Climate Classification")
        classification_layout = QVBoxLayout()

        self.climate_type_label = QLabel("Dominant Climate: Not calculated")
        self.temp_range_label = QLabel("Temperature Range: -")
        self.precip_total_label = QLabel("Total Precipitation: -")

        classification_layout.addWidget(self.climate_type_label)
        classification_layout.addWidget(self.temp_range_label)
        classification_layout.addWidget(self.precip_total_label)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group)

        results_group = QGroupBox("Generation Results")
        results_layout = QVBoxLayout()

        self.orographic_effect_label = QLabel("Orographic Effect: -")
        self.wind_strength_label = QLabel("Avg Wind Strength: -")
        self.humidity_level_label = QLabel("Avg Humidity: -")

        results_layout.addWidget(self.orographic_effect_label)
        results_layout.addWidget(self.wind_strength_label)
        results_layout.addWidget(self.humidity_level_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def update_parameter_preview(self, parameters: dict):
        """Aktualisiert Parameter-Preview"""
        base_temp = parameters.get("air_temp_entry", 15)
        solar = parameters.get("solar_power", 20)
        altitude = parameters.get("altitude_cooling", 6)
        wind = parameters.get("wind_speed_factor", 1.0)

        self.base_temp_label.setText(f"Base Temperature: {base_temp}°C")
        self.solar_power_label.setText(f"Solar Power: {solar}°C")
        self.altitude_cooling_label.setText(f"Altitude Cooling: {altitude}°C/100m")
        self.wind_factor_label.setText(f"Wind Factor: {wind:.1f}")

    def update_generation_statistics(self, results: dict):
        """
        Aktualisiert Statistiken nach abgeschlossener Generation.
        Parameter: results (dict mit temp_map/wind_map/humid_map/precip_map)
        """
        temp_map = results.get("temp_map")
        wind_map = results.get("wind_map")
        humid_map = results.get("humid_map")
        precip_map = results.get("precip_map")

        if temp_map is not None:
            temp_min, temp_max = np.min(temp_map), np.max(temp_map)
            self.temp_range_label.setText(f"Temperature Range: {temp_min:.1f}°C - {temp_max:.1f}°C")

            avg_temp = np.mean(temp_map)
            if avg_temp < 0:
                climate_type = "Arctic"
            elif avg_temp < 10:
                climate_type = "Subarctic"
            elif avg_temp < 20:
                climate_type = "Temperate"
            else:
                climate_type = "Subtropical"

            self.climate_type_label.setText(f"Dominant Climate: {climate_type}")

        if precip_map is not None:
            total_precip = np.sum(precip_map)
            self.precip_total_label.setText(f"Total Precipitation: {total_precip:.1f} gH2O/m²")

        if wind_map is not None:
            if len(wind_map.shape) == 3:
                wind_strength = np.sqrt(wind_map[:, :, 0] ** 2 + wind_map[:, :, 1] ** 2)
                avg_wind = np.mean(wind_strength)
                self.wind_strength_label.setText(f"Avg Wind Strength: {avg_wind:.2f} m/s")

        if humid_map is not None:
            avg_humidity = np.mean(humid_map)
            self.humidity_level_label.setText(f"Avg Humidity: {avg_humidity:.1f} gH2O/m³")

        if temp_map is not None and wind_map is not None:
            temp_grad = np.gradient(temp_map)
            orographic_strength = np.mean(np.sqrt(temp_grad[0] ** 2 + temp_grad[1] ** 2))
            self.orographic_effect_label.setText(f"Orographic Effect: {orographic_strength:.3f}°C/pixel")
