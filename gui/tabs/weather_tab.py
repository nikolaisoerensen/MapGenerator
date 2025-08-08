"""
Path: gui/tabs/weather_tab.py

Funktionsweise: Wetter-System mit 3D Wind-Visualization und vollständiger Core-Integration
- Climate-Modeling basierend auf Terrain (Orographic Effects)
- 3D Wind-Vector Display über Heightmap
- Live Climate-Classification Display
- Temperature/Precipitation Field Generation mit atmosphärischen Effekten
- GPU-Shader Integration für Berg-Wind-Simulation
- Input: heightmap, shade_map, soil_moist_map
- Output: wind_map, temp_map, precip_map, humid_map
"""

import numpy as np
import logging

from PyQt5.QtCore import QTimer, pyqtSlot, QMetaObject, Q_ARG, Qt
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QButtonGroup, QRadioButton, QCheckBox, QLabel, QProgressBar, \
    QHBoxLayout, QComboBox

from .base_tab import BaseMapTab
from gui.config.value_default import WEATHER, get_parameter_config, validate_parameter_set, VALIDATION_RULES
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton, MultiDependencyStatusWidget
from core.weather_generator import (
    WeatherSystemGenerator, TemperatureCalculator, WindFieldSimulator,
    PrecipitationSystem, AtmosphericMoistureManager
)

def get_weather_error_decorators():
    """
    Funktionsweise: Lazy Loading von Weather Tab Error Decorators
    Aufgabe: Lädt Core-Generation, GPU-Shader und Dependency Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import core_generation_handler, gpu_shader_handler, dependency_handler
        return core_generation_handler, gpu_shader_handler, dependency_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

core_generation_handler, gpu_shader_handler, dependency_handler = get_weather_error_decorators()

class WeatherTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für dynamisches Wetter- und Feuchtigkeitssystem
    Aufgabe: Koordiniert alle Weather-Core-Module und GPU-Shader Integration
    Input: heightmap, shade_map, soil_moist_map für orographic effects
    Output: wind_map, temp_map, precip_map, humid_map für nachfolgende Generatoren
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)
        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.weather_system = WeatherSystemGenerator()
        self.temperature_calculator = TemperatureCalculator()
        self.wind_simulator = WindFieldSimulator()
        self.precipitation_system = PrecipitationSystem()
        self.moisture_manager = AtmosphericMoistureManager()

        # Parameter und State
        self.current_parameters = {}
        self.weather_simulation_active = False

        # Setup UI
        self.setup_weather_ui()
        self.setup_dependency_checking()
        self.setup_shader_integration()
        self.setup_orchestrator_integration()

        # Initial Load
        self.load_default_parameters()
        self.check_input_dependencies()

    def setup_weather_ui(self):
        """
        Funktionsweise: Erstellt komplette UI für Weather-System
        Aufgabe: Parameter, Climate-Preview, 3D-Wind-Visualization, Performance-Monitor
        """
        # Parameter Panel
        self.logger.info("DEBUG: Creating parameter panel...")
        self.parameter_panel = self.create_weather_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)
        self.logger.info("DEBUG: Parameter panel added successfully")

        # LOD Control Panel
        self.logger.info("DEBUG: Creating LOD control panel...")
        self.lod_control_panel = self.create_lod_control_panel()
        self.control_panel.layout().addWidget(self.lod_control_panel)
        self.logger.info("DEBUG: LOD control panel added successfully")

        # Climate Statistics
        self.logger.info("DEBUG: Creating climate stats...")
        self.climate_stats = ClimateStatisticsWidget()
        self.control_panel.layout().addWidget(self.climate_stats)
        self.logger.info("DEBUG: Climate stats added successfully")

        # Visualization Controls
        self.logger.info("DEBUG: Creating visualization controls...")
        self.visualization_controls = self.create_weather_visualization_controls()
        self.control_panel.layout().addWidget(self.visualization_controls)
        self.logger.info("DEBUG: Visualization controls added successfully")

        # Weather Shader Performance
        self.logger.info("DEBUG: Creating shader performance widget...")
        self.shader_performance = WeatherShaderPerformanceWidget(self.shader_manager)
        self.control_panel.layout().addWidget(self.shader_performance)
        self.logger.info("DEBUG: Shader performance widget added successfully")

    def create_weather_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Weather-Parametern
        Aufgabe: Alle 6 Parameter aus value_default.WEATHER strukturiert
        Return: QGroupBox mit Parameter-Slidern
        """
        panel = QGroupBox("Weather Parameters")
        layout = QVBoxLayout()

        self.parameter_sliders = {}

        # Temperature Parameters
        temp_group = QGroupBox("Temperature System")
        temp_layout = QVBoxLayout()

        temp_params = ["air_temp_entry", "solar_power", "altitude_cooling"]
        for param_name in temp_params:
            param_config = get_parameter_config("weather", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 1),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            temp_layout.addWidget(slider)

        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)

        # Wind System Parameters
        wind_group = QGroupBox("Wind System")
        wind_layout = QVBoxLayout()

        wind_params = ["thermic_effect", "wind_speed_factor", "terrain_factor"]
        for param_name in wind_params:
            param_config = get_parameter_config("weather", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 0.1),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            wind_layout.addWidget(slider)

        wind_group.setLayout(wind_layout)
        layout.addWidget(wind_group)

        panel.setLayout(layout)
        return panel

    def create_weather_visualization_controls(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Controls für Weather-Visualization
        Aufgabe: Switcher zwischen allen 4 Weather-Output-Maps mit 3D-Features
        Return: QGroupBox mit Visualization-Controls
        """
        widget = super().create_visualization_controls()
        layout = widget.layout()

        # Weather-spezifische Modi hinzufügen
        self.temperature_radio = QRadioButton("Temperature")
        self.temperature_radio.setStyleSheet("font-size: 11px;")
        self.temperature_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.temperature_radio, 1)
        layout.insertWidget(1, self.temperature_radio)

        self.precipitation_radio = QRadioButton("Precipitation")
        self.precipitation_radio.setStyleSheet("font-size: 11px;")
        self.precipitation_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.precipitation_radio, 2)
        layout.insertWidget(2, self.precipitation_radio)

        self.humidity_radio = QRadioButton("Humidity")
        self.humidity_radio.setStyleSheet("font-size: 11px;")
        self.humidity_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.humidity_radio, 3)
        layout.insertWidget(3, self.humidity_radio)

        self.wind_field_radio = QRadioButton("Wind Field")
        self.wind_field_radio.setStyleSheet("font-size: 11px;")
        self.wind_field_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.wind_field_radio, 4)
        layout.insertWidget(4, self.wind_field_radio)

        # 3D Controls auch hier
        self.terrain_3d_checkbox = QCheckBox("3D Terrain")
        self.terrain_3d_checkbox.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.terrain_3d_checkbox)

        return widget

    def create_lod_control_panel(self) -> QGroupBox:
        """Erstellt LOD-Control Panel für Weather Quality Selection"""
        panel = QGroupBox("Quality / LOD Control")
        layout = QVBoxLayout()

        # Target-LOD Selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Quality:"))

        self.target_lod_combo = QComboBox()
        self.target_lod_combo.addItems(["LOD64 (Fast Preview)", "LOD128 (Medium)", "LOD256 (High)", "FINAL (Best)"])
        self.target_lod_combo.setCurrentIndex(3)
        target_layout.addWidget(self.target_lod_combo)

        layout.addLayout(target_layout)

        # Generation Progress
        self.generation_progress = QProgressBar()
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        self.generation_progress.setTextVisible(True)
        self.generation_progress.setFormat("Ready")
        layout.addWidget(self.generation_progress)

        panel.setLayout(layout)
        return panel

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking - SHADEMAP-FIX
        Aufgabe: MINIMAL-FIX für fehlende Shademap-Dependency
        """
        self.required_dependencies = ["heightmap", "shadowmap"]

        # Optional soil_moist_map (kann fehlen)
        if self.data_manager.get_water_data("soil_moist_map") is not None:
            self.required_dependencies.append("soil_moist_map")

        self.dependency_status = MultiDependencyStatusWidget(
            self.required_dependencies, "Weather Dependencies"
        )
        self.control_panel.layout().addWidget(self.dependency_status)

        self.data_manager.data_updated.connect(self.on_data_updated)

    def setup_shader_integration(self):
        """
        Funktionsweise: Setup für GPU-Shader Integration
        Aufgabe: Konfiguriert alle 8 Weather-Shader für Berg-Wind-Simulation
        """
        # Shader Performance Monitoring
        self.shader_performance_timer = QTimer()
        self.shader_performance_timer.timeout.connect(self.monitor_shader_performance)

        # GPU Verfügbarkeit prüfen
        if self.shader_manager:
            self.gpu_available = self.shader_manager.check_gpu_support()
        else:
            self.gpu_available = False

        if not self.gpu_available:
            self.logger.warning("GPU not available - using CPU fallback for weather simulation")

        # Weather-Shader laden
        self.weather_shaders_loaded = False
        if self.gpu_available:
            self.load_weather_shaders()

    def load_weather_shaders(self):
        """
        Funktionsweise: Lädt alle 8 Weather-Shader für Berg-Wind-Simulation
        Aufgabe: Initialisiert GPU-Compute-Pipeline für Weather-System
        """
        try:
            # Alle 8 Weather-Shader aus Core-Dokumentation
            shader_names = [
                "temperatureCalculation",
                "windFieldGeneration",
                "thermalConvection",
                "moistureTransport",
                "precipitationCalculation",
                "orographicEffects",
                "weatherIntegration",
                "boundaryConditions"
            ]

            for shader_name in shader_names:
                success = self.shader_manager.load_weather_shader(shader_name)
                if not success:
                    self.logger.warning(f"Failed to load {shader_name} shader")

            self.weather_shaders_loaded = True
            self.logger.info("Weather shaders loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load weather shaders: {e}")
            self.weather_shaders_loaded = False

    def load_default_parameters(self):
        """Lädt Default-Parameter"""
        for param_name, slider in self.parameter_sliders.items():
            param_config = get_parameter_config("weather", param_name)
            slider.setValue(param_config["default"])

        self.current_parameters = self.get_current_parameters()

    def get_current_parameters(self) -> dict:
        """Sammelt aktuelle Parameter für Core-Generator"""
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    @pyqtSlot()
    def on_parameter_changed(self):
        """Slot für Parameter-Änderungen"""
        self.current_parameters = self.get_current_parameters()

        # Climate Statistics Preview aktualisieren
        self.climate_stats.update_parameter_preview(self.current_parameters)

        # Auto-Simulation triggern
        if self.auto_simulation_enabled and not self.weather_simulation_active:
            self.auto_simulation_timer.start(1200)  # 1.2s für Weather-Berechnung

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """Slot für Data-Updates von anderen Generatoren"""
        if data_key in self.required_dependencies:
            self.check_input_dependencies()

    def check_input_dependencies(self):
        """
        Funktionsweise: Prüft ob alle Required Dependencies verfügbar sind
        Aufgabe: Aktiviert/Deaktiviert Generation Button und zeigt Status
        """
        is_complete, missing = self.data_manager.check_dependencies("weather", self.required_dependencies)

        self.dependency_status.update_dependency_status(is_complete, missing)

        # FIX: Nutze Public API für Button-Control statt direkten Zugriff
        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_manual_button_enabled(is_complete)

        return is_complete

    def generate(self):
        """
        Funktionsweise: Hauptmethode für Weather-Generation mit Orchestrator Integration
        Aufgabe: Startet Weather-Generation über GenerationOrchestrator mit Target-LOD
        """
        if not self.generation_orchestrator:
            self.logger.error("No GenerationOrchestrator available")
            return

        if self.generation_in_progress:
            self.logger.info("Generation already in progress, ignoring request")
            return

        try:
            self.logger.info(f"Starting generation with target LOD: {self.target_lod}")
            self.start_generation_timing()
            self.generation_in_progress = True

            request_id = self.generation_orchestrator.request_generation(
                generator_type="weather",
                parameters=self.current_parameters.copy(),
                target_lod=self.target_lod,
                source_tab="weather",
                priority=10
            )

            if request_id:
                self.logger.info(f"Generation requested: {request_id}")
            else:
                raise Exception("Failed to request generation")

        except Exception as e:
            self.generation_in_progress = False
            self.handle_generation_error(e)

    @pyqtSlot(str)
    def _update_ui_generation_started(self, lod_level: str):
        """UI-Update für Generation-Start in Main-Thread"""
        self.generation_progress.setValue(0)
        self.generation_progress.setFormat(f"Generating {lod_level}...")

        if self.gpu_available and self.weather_shaders_loaded:
            self.shader_performance_timer.start(500)

    @pyqtSlot(str, bool)
    def _update_ui_generation_completed(self, lod_level: str, success: bool):
        """UI-Update für Generation-Completion in Main-Thread"""
        if success:
            self.available_lods.add(lod_level)
            self.generation_progress.setValue(100)
            self.generation_progress.setFormat(f"{lod_level} Complete")

            self.update_weather_display()

            integrated_results = {
                "temp_map": self.data_manager.get_weather_data("temp_map"),
                "wind_map": self.data_manager.get_weather_data("wind_map"),
                "humid_map": self.data_manager.get_weather_data("humid_map"),
                "precip_map": self.data_manager.get_weather_data("precip_map")
            }
            if any(integrated_results.values()):
                self.climate_stats.update_generation_statistics(integrated_results)
                self.shader_performance.update_shader_statistics()
        else:
            self.generation_progress.setFormat(f"{lod_level} Failed")

        if lod_level == self.target_lod:
            self.generation_in_progress = False
            self.weather_simulation_active = False
            self.end_generation_timing(success)

            if self.gpu_available:
                self.shader_performance_timer.stop()

    @pyqtSlot(str, int)
    def _update_ui_generation_progress(self, lod_level: str, progress_percent: int):
        """UI-Update für Generation-Progress in Main-Thread"""
        self.generation_progress.setValue(progress_percent)
        self.generation_progress.setFormat(f"{lod_level} - {progress_percent}%")

    def collect_input_data(self) -> dict:
        """
        Funktionsweise: Input-Sammlung - SHADEMAP-KEY-FIX
        Aufgabe: Korrekte Shademap-Referenz
        """
        inputs = {}

        # Terrain Inputs
        inputs["heightmap"] = self.data_manager.get_terrain_data("heightmap")
        inputs["shadowmap"] = self.data_manager.get_terrain_data("shadowmap")  # Korrigiert

        # Water Input (Fallback wenn nicht verfügbar)
        inputs["soil_moist_map"] = self.data_manager.get_water_data("soil_moist_map")

        # Validation für Required Inputs
        required_inputs = ["heightmap", "shadowmap"]
        for key in required_inputs:
            if inputs[key] is None:
                raise ValueError(f"Required input '{key}' not available")

        # Fallback für soil_moist_map
        if inputs["soil_moist_map"] is None:
            self.logger.warning("Soil moisture map not available - using uniform base moisture")
            inputs["soil_moist_map"] = np.full_like(inputs["heightmap"], 0.3)

        return inputs

    def save_weather_results(self, results: dict, params: dict):
        """
        Funktionsweise: Speichert alle Weather-System Results im DataManager
        Parameter: results (dict mit allen Weather-Maps), params (dict)
        """
        # Weather Results speichern
        self.data_manager.set_weather_data("temp_map", results["temp_map"], params)
        self.data_manager.set_weather_data("wind_map", results["wind_map"], params)
        self.data_manager.set_weather_data("humid_map", results["humid_map"], params)
        self.data_manager.set_weather_data("precip_map", results["precip_map"], params)

    def update_weather_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt verschiedene Weather-Maps mit 3D-Overlays
        """
        current_mode = self.display_mode.checkedId()

        if current_mode == 0:  # Temperature Map
            temp_map = self.data_manager.get_weather_data("temp_map")
            if temp_map is not None:
                self.map_display.display_temperature_map(temp_map)

        elif current_mode == 1:  # Precipitation Map
            precip_map = self.data_manager.get_weather_data("precip_map")
            if precip_map is not None:
                self.map_display.display_precipitation_map(precip_map)

        elif current_mode == 2:  # Humidity Map
            humid_map = self.data_manager.get_weather_data("humid_map")
            if humid_map is not None:
                self.map_display.display_humidity_map(humid_map)

        elif current_mode == 3:  # Wind Field
            wind_map = self.data_manager.get_weather_data("wind_map")
            if wind_map is not None:
                self.map_display.display_wind_field(wind_map)

        # Overlays anwenden
        self.apply_weather_overlays()

    def apply_weather_overlays(self):
        """
        Funktionsweise: Wendet alle aktivierten Weather-Overlays an
        Aufgabe: 3D Terrain, Contours, Orographic Effects, 3D Wind Vectors
        """
        # 3D Terrain Overlay
        if self.terrain_3d_cb.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.overlay_3d_terrain(heightmap)

        # Contour Lines
        if self.contour_lines_cb.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.overlay_elevation_contours(heightmap)

        # Orographic Effects Highlighting
        if self.orographic_effects_cb.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            wind_map = self.data_manager.get_weather_data("wind_map")
            if heightmap is not None and wind_map is not None:
                self.map_display.highlight_orographic_effects(heightmap, wind_map)

        # 3D Wind Vectors
        if self.wind_vectors_3d_cb.isChecked():
            wind_map = self.data_manager.get_weather_data("wind_map")
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if wind_map is not None and heightmap is not None:
                density = int(self.wind_density_slider.getValue())
                scale = self.wind_scale_slider.getValue()
                self.map_display.overlay_3d_wind_vectors(wind_map, heightmap, density, scale)

    def update_display_mode(self):
        """Funktionsweise: Weather Display-Mode Handler """
        current_mode = self.display_mode.checkedId()

        # FIX 1: get_current_display() verwenden statt self.map_display
        current_display = self.get_current_display()
        if not current_display:
            return

        try:
            if current_mode == 0:  # Temperature Map
                temp_map = self.data_manager.get_weather_data("temp_map")
                if temp_map is not None:
                    current_display.update_display(temp_map, "temp_map")
            # ... weitere Modi
        except Exception as e:
            self.logger.error(f"Weather display update failed: {e}")

    @pyqtSlot(bool)
    def toggle_3d_terrain(self, enabled: bool):
        """Toggle für 3D Terrain Overlay"""
        self.update_weather_display()

    @pyqtSlot(bool)
    def toggle_contour_lines(self, enabled: bool):
        """Toggle für Contour Lines Overlay"""
        self.update_weather_display()

    @pyqtSlot(bool)
    def toggle_orographic_effects(self, enabled: bool):
        """Toggle für Orographic Effects Highlighting"""
        self.update_weather_display()

    @pyqtSlot(bool)
    def toggle_3d_wind_vectors(self, enabled: bool):
        """Toggle für 3D Wind Vectors"""
        self.update_weather_display()

    @pyqtSlot(float)
    def update_wind_vector_density(self, density: float):
        """Update Wind Vector Density"""
        if self.wind_vectors_3d_cb.isChecked():
            self.update_weather_display()

    @pyqtSlot(float)
    def update_wind_vector_scale(self, scale: float):
        """Update Wind Vector Scale"""
        if self.wind_vectors_3d_cb.isChecked():
            self.update_weather_display()

    @pyqtSlot()
    def monitor_shader_performance(self):
        """
        Funktionsweise: Überwacht GPU-Shader Performance während Weather-Generation
        Aufgabe: Updated Shader-Performance Widget mit aktuellen Metriken
        """
        if self.weather_simulation_active and self.gpu_available:
            performance_metrics = self.shader_manager.get_weather_shader_performance()
            self.shader_performance.update_performance_metrics(performance_metrics)

class ClimateStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Widget für Climate-Statistiken und Parameter-Preview
    Aufgabe: Zeigt Weather-Parameter, Climate-Classification, Generation-Results
    """

    def __init__(self):
        super().__init__("Climate Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Climate-Statistiken"""
        layout = QVBoxLayout()

        # Parameter Preview
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

        # Climate Classification Preview
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

        # Generation Results
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
        Funktionsweise: Aktualisiert Statistiken nach Weather-Generation
        Parameter: results (dict mit weather maps)
        """
        temp_map = results.get("temp_map")
        wind_map = results.get("wind_map")
        humid_map = results.get("humid_map")
        precip_map = results.get("precip_map")

        if temp_map is not None:
            temp_min, temp_max = np.min(temp_map), np.max(temp_map)
            self.temp_range_label.setText(f"Temperature Range: {temp_min:.1f}°C - {temp_max:.1f}°C")

            # Climate Classification basierend auf Temperature
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
            # Wind strength berechnen (Magnitude der Vektoren)
            if len(wind_map.shape) == 3:  # Wind vectors [x, y, z]
                wind_strength = np.sqrt(wind_map[:, :, 0] ** 2 + wind_map[:, :, 1] ** 2)
                avg_wind = np.mean(wind_strength)
                self.wind_strength_label.setText(f"Avg Wind Strength: {avg_wind:.2f} m/s")

        if humid_map is not None:
            avg_humidity = np.mean(humid_map)
            self.humidity_level_label.setText(f"Avg Humidity: {avg_humidity:.1f} gH2O/m³")

        # Orographic Effect berechnen (vereinfacht)
        if temp_map is not None and wind_map is not None:
            # Gradient der Temperatur als Proxy für orographic effects
            temp_grad = np.gradient(temp_map)
            orographic_strength = np.mean(np.sqrt(temp_grad[0] ** 2 + temp_grad[1] ** 2))
            self.orographic_effect_label.setText(f"Orographic Effect: {orographic_strength:.3f}°C/pixel")


class WeatherShaderPerformanceWidget(QGroupBox):
    """
    Funktionsweise: Widget für Weather-Shader Performance Monitoring
    Aufgabe: Zeigt Performance-Metriken aller 8 Weather-Shader
    """

    def __init__(self, shader_manager):
        super().__init__("Weather Shader Performance")
        self.shader_manager = shader_manager
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Shader-Performance Display"""
        layout = QVBoxLayout()

        self.gpu_status = StatusIndicator("GPU Status")
        if self.shader_manager and self.shader_manager.check_gpu_support():  # ← Null-Check hinzufügen
            self.gpu_status.set_success("GPU Available")
        else:
            self.gpu_status.set_warning("Using CPU Fallback")
        layout.addWidget(self.gpu_status)

        # Shader Loading Status
        self.shader_loading_status = StatusIndicator("Weather Shaders")
        self.shader_loading_status.set_unknown()
        layout.addWidget(self.shader_loading_status)

        # Performance Metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout()

        self.total_frame_time_label = QLabel("Total Frame Time: - ms")
        self.gpu_memory_usage_label = QLabel("GPU Memory: - MB")
        self.active_shaders_label = QLabel("Active Shaders: 0/8")

        metrics_layout.addWidget(self.total_frame_time_label)
        metrics_layout.addWidget(self.gpu_memory_usage_label)
        metrics_layout.addWidget(self.active_shaders_label)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        # Individual Shader Performance
        shader_group = QGroupBox("Individual Shader Performance")
        shader_layout = QVBoxLayout()

        # Top 3 Performance-kritische Shader anzeigen
        self.temp_calc_perf = QLabel("Temperature Calc: - ms")
        self.wind_sim_perf = QLabel("Wind Simulation: - ms")
        self.precip_calc_perf = QLabel("Precipitation Calc: - ms")

        shader_layout.addWidget(self.temp_calc_perf)
        shader_layout.addWidget(self.wind_sim_perf)
        shader_layout.addWidget(self.precip_calc_perf)

        shader_group.setLayout(shader_layout)
        layout.addWidget(shader_group)

        # Performance Rating
        self.performance_rating = QProgressBar()
        self.performance_rating.setRange(0, 100)
        self.performance_rating.setValue(100)
        layout.addWidget(QLabel("Overall Performance:"))
        layout.addWidget(self.performance_rating)

        self.setLayout(layout)

    def update_performance_metrics(self, metrics: dict):
        """
        Funktionsweise: Aktualisiert Performance-Metrics von Shader-Manager
        Parameter: metrics (dict mit performance data)
        """
        # Total Frame Time
        total_time = metrics.get("total_frame_time_ms", 0)
        self.total_frame_time_label.setText(f"Total Frame Time: {total_time:.1f} ms")

        # GPU Memory
        gpu_memory = metrics.get("gpu_memory_mb", 0)
        self.gpu_memory_usage_label.setText(f"GPU Memory: {gpu_memory:.0f} MB")

        # Active Shaders
        active_shaders = metrics.get("active_weather_shaders", 0)
        self.active_shaders_label.setText(f"Active Shaders: {active_shaders}/8")

        # Individual Shader Performance
        shader_times = metrics.get("individual_shader_times", {})
        temp_time = shader_times.get("temperatureCalculation", 0)
        wind_time = shader_times.get("windFieldGeneration", 0)
        precip_time = shader_times.get("precipitationCalculation", 0)

        self.temp_calc_perf.setText(f"Temperature Calc: {temp_time:.2f} ms")
        self.wind_sim_perf.setText(f"Wind Simulation: {wind_time:.2f} ms")
        self.precip_calc_perf.setText(f"Precipitation Calc: {precip_time:.2f} ms")

        # Performance Rating (basierend auf Frame Time)
        # Target: 60fps = 16.67ms per frame für Real-time
        # Weather ist nicht real-time, aber < 100ms ist gut
        if total_time < 50:
            performance_pct = 100
            color = "#27ae60"  # Green
        elif total_time < 100:
            performance_pct = 80
            color = "#f39c12"  # Orange
        elif total_time < 200:
            performance_pct = 60
            color = "#e67e22"  # Dark Orange
        else:
            performance_pct = 40
            color = "#e74c3c"  # Red

        self.performance_rating.setValue(performance_pct)
        self.performance_rating.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")

    def update_shader_statistics(self):
        """
        Funktionsweise: Aktualisiert Shader-Loading Status
        Wird nach erfolgreicher Generation aufgerufen
        """
        if self.shader_manager.check_gpu_support():
            weather_shaders_count = self.shader_manager.get_loaded_weather_shaders_count()
            if weather_shaders_count >= 8:
                self.shader_loading_status.set_success(f"All {weather_shaders_count}/8 shaders loaded")
            elif weather_shaders_count > 0:
                self.shader_loading_status.set_warning(f"Partial loading: {weather_shaders_count}/8 shaders")
            else:
                self.shader_loading_status.set_error("No weather shaders loaded")
        else:
            self.shader_loading_status.set_warning("CPU fallback active")