"""
Path: gui/tabs/water_tab.py

Funktionsweise: Wassersystem mit River-Networks, Erosion und vollständiger Core-Integration
- Input: heightmap, slopemap, hardness_map, rock_map, precip_map, temp_map, wind_map, humid_map
- River-Generation durch Flow-Accumulation mit Jump Flooding Algorithm
- Lake-Placement und Water-Table Simulation
- Erosion-Simulation modifiziert Heightmap und Rock-Map mit Massenerhaltung
- Alle 8 Shader für GPU-accelerated Hydrologie-Simulation
- Output: 10 verschiedene Water-Maps + ocean_outflow
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import WATER, get_parameter_config, validate_parameter_set, VALIDATION_RULES
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton
from core.water_generator import (
    HydrologySystemGenerator, LakeDetectionSystem, FlowNetworkBuilder,
    ManningFlowCalculator, ErosionSedimentationSystem, SoilMoistureCalculator,
    EvaporationCalculator
)

def get_water_error_decorators():
    """
    Funktionsweise: Lazy Loading von Water Tab Error Decorators
    Aufgabe: Lädt Memory-Critical, GPU-Shader und Core-Generation Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import memory_critical_handler, gpu_shader_handler, core_generation_handler
        return memory_critical_handler, gpu_shader_handler, core_generation_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

memory_critical_handler, gpu_shader_handler, core_generation_handler = get_water_error_decorators()

class WaterTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für komplettes Hydrologie-System
    Aufgabe: Koordiniert alle Water-Core-Module und GPU-Shader Integration
    Input: Alle Dependencies von terrain/geology/weather
    Output: Komplettes Wassersystem mit Erosion und Sedimentation
    """

    def __init__(self, data_manager, navigation_manager, shader_manager):
        super().__init__(data_manager, navigation_manager, shader_manager)
        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.hydrology_system = HydrologySystemGenerator()
        self.lake_detection = LakeDetectionSystem()
        self.flow_network = FlowNetworkBuilder()
        self.manning_calculator = ManningFlowCalculator()
        self.erosion_system = ErosionSedimentationSystem()
        self.soil_moisture = SoilMoistureCalculator()
        self.evaporation = EvaporationCalculator()

        # Parameter und State
        self.current_parameters = {}
        self.generation_in_progress = False

        # Setup UI
        self.setup_water_ui()
        self.setup_dependency_checking()
        self.setup_shader_integration()

        # Initial Load
        self.load_default_parameters()
        self.check_input_dependencies()

    def setup_water_ui(self):
        """
        Funktionsweise: Erstellt komplette UI für Water-System
        Aufgabe: Parameter, Visualization, Statistics, Shader-Controls
        """
        # Parameter Panel
        self.parameter_panel = self.create_water_parameter_panel()
        self.control_panel.addWidget(self.parameter_panel)

        # Hydrologie Statistics
        self.hydrology_stats = HydrologyStatisticsWidget()
        self.control_panel.addWidget(self.hydrology_stats)

        # Visualization Controls
        self.visualization_controls = self.create_water_visualization_controls()
        self.control_panel.addWidget(self.visualization_controls)

        # Shader Performance Panel
        self.shader_controls = ShaderPerformancePanel(self.shader_manager)
        self.control_panel.addWidget(self.shader_controls)

        # Dependencies und Navigation
        self.setup_input_status()
        self.setup_navigation()

    def create_water_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Water-Parametern
        Aufgabe: Alle 8 Parameter aus value_default.WATER
        Return: QGroupBox mit strukturierten Parameter-Slidern
        """
        panel = QGroupBox("Hydrology Parameters")
        layout = QVBoxLayout()

        self.parameter_sliders = {}

        # Lake Detection Parameters
        lake_group = QGroupBox("Lake Detection")
        lake_layout = QVBoxLayout()

        lake_params = ["lake_volume_threshold", "rain_threshold"]
        for param_name in lake_params:
            param_config = get_parameter_config("water", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 0.01),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            lake_layout.addWidget(slider)

        lake_group.setLayout(lake_layout)
        layout.addWidget(lake_group)

        # Flow Calculation Parameters
        flow_group = QGroupBox("Flow Dynamics")
        flow_layout = QVBoxLayout()

        flow_params = ["manning_coefficient", "diffusion_radius"]
        for param_name in flow_params:
            param_config = get_parameter_config("water", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 0.001),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            flow_layout.addWidget(slider)

        flow_group.setLayout(flow_layout)
        layout.addWidget(flow_group)

        # Erosion/Sedimentation Parameters
        erosion_group = QGroupBox("Erosion & Sedimentation")
        erosion_layout = QVBoxLayout()

        erosion_params = ["erosion_strength", "sediment_capacity_factor", "settling_velocity"]
        for param_name in erosion_params:
            param_config = get_parameter_config("water", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 0.01),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            erosion_layout.addWidget(slider)

        erosion_group.setLayout(erosion_layout)
        layout.addWidget(erosion_group)

        # Evaporation Parameters
        evap_group = QGroupBox("Evaporation")
        evap_layout = QVBoxLayout()

        evap_param = "evaporation_base_rate"
        param_config = get_parameter_config("water", evap_param)

        slider = ParameterSlider(
            label=evap_param.replace("_", " ").title(),
            min_val=param_config["min"],
            max_val=param_config["max"],
            default_val=param_config["default"],
            step=param_config.get("step", 0.0001),
            suffix=param_config.get("suffix", "")
        )

        slider.valueChanged.connect(self.on_parameter_changed)
        self.parameter_sliders[evap_param] = slider
        evap_layout.addWidget(slider)

        evap_group.setLayout(evap_layout)
        layout.addWidget(evap_group)

        panel.setLayout(layout)
        return panel

    def create_water_visualization_controls(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Controls für Water-Visualization
        Aufgabe: Switcher zwischen allen 11 Water-Output-Maps
        Return: QGroupBox mit Visualization-Controls
        """
        panel = QGroupBox("Water Visualization")
        layout = QVBoxLayout()

        # Display Mode Selection
        self.display_mode = QButtonGroup()

        # Alle Water-Output-Maps als Radio-Buttons
        water_outputs = [
            ("Water Depth", "water_map"),
            ("Flow Volume", "flow_map"),
            ("Flow Speed", "flow_speed"),
            ("Cross Section", "cross_section"),
            ("Soil Moisture", "soil_moist_map"),
            ("Erosion Rate", "erosion_map"),
            ("Sedimentation", "sedimentation_map"),
            ("Updated Rock Map", "rock_map_updated"),
            ("Evaporation", "evaporation_map"),
            ("Water Biomes", "water_biomes_map")
        ]

        for i, (display_name, data_key) in enumerate(water_outputs):
            radio = QRadioButton(display_name)
            radio.setProperty("data_key", data_key)
            radio.toggled.connect(self.update_display_mode)
            self.display_mode.addButton(radio, i)
            layout.addWidget(radio)

            if i == 0:  # Water Depth als Default
                radio.setChecked(True)

        # River Network Overlay
        self.river_overlay_checkbox = QCheckBox("Show River Network")
        self.river_overlay_checkbox.toggled.connect(self.toggle_river_overlay)
        layout.addWidget(self.river_overlay_checkbox)

        # 3D Terrain Overlay
        self.terrain_3d_checkbox = QCheckBox("Show 3D Terrain")
        self.terrain_3d_checkbox.toggled.connect(self.toggle_3d_terrain)
        layout.addWidget(self.terrain_3d_checkbox)

        # Ocean Outflow Display
        self.ocean_outflow_label = QLabel("Ocean Outflow: - m³/s")
        layout.addWidget(self.ocean_outflow_label)

        panel.setLayout(layout)
        return panel

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für komplexe Input-Dependency Checking
        Aufgabe: Überwacht alle Required Dependencies (8 verschiedene Maps)
        """
        # Required Dependencies für Water-System
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["water"]

        # Dependency Status Widget mit Details
        self.dependency_status = MultiDependencyStatusWidget(self.required_dependencies)
        self.control_panel.addWidget(self.dependency_status)

        # Data Manager Signals
        self.data_manager.data_updated.connect(self.on_data_updated)
        self.data_manager.dependency_changed.connect(self.on_dependency_changed)

    def setup_shader_integration(self):
        """
        Funktionsweise: Setup für GPU-Shader Integration
        Aufgabe: Konfiguriert alle 8 Water-Shader für optimale Performance
        """
        # Shader Performance Monitoring
        self.shader_performance_timer = QTimer()
        self.shader_performance_timer.timeout.connect(self.monitor_shader_performance)

        # GPU Fallback Detection
        self.gpu_available = self.shader_manager.check_gpu_support()
        if not self.gpu_available:
            self.logger.warning("GPU not available - using CPU fallback")

    def load_default_parameters(self):
        """Lädt Default-Parameter"""
        for param_name, slider in self.parameter_sliders.items():
            param_config = get_parameter_config("water", param_name)
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

        # Hydrologie Statistics Preview aktualisieren
        self.hydrology_stats.update_parameter_preview(self.current_parameters)

        # Auto-Simulation triggern
        if self.auto_simulation_enabled and not self.generation_in_progress:
            self.auto_simulation_timer.start(1500)  # 1.5s für komplexe Berechnung

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """Slot für Data-Updates von anderen Generatoren"""
        if data_key in self.required_dependencies:
            self.check_input_dependencies()

    @pyqtSlot(str, list)
    def on_dependency_changed(self, generator_type: str, missing: list):
        """Slot für Dependency-Änderungen"""
        if generator_type == "water":
            self.dependency_status.update_missing_dependencies(missing)

    def check_input_dependencies(self):
        """
        Funktionsweise: Prüft alle Required Dependencies für Water-System
        Aufgabe: Aktiviert/Deaktiviert Generation basierend auf verfügbaren Inputs
        """
        is_complete, missing = self.data_manager.check_dependencies("water", self.required_dependencies)

        self.dependency_status.update_dependency_status(is_complete, missing)
        self.manual_generate_button.setEnabled(is_complete)

        return is_complete

    @memory_critical_handler("water_generation")
    def generate_water_system(self):
        """
        Funktionsweise: Hauptmethode für komplette Water-System Generation
        Aufgabe: Koordiniert alle 7 Core-Module und 8 Shader für Hydrologie-Simulation
        """
        try:
            # Dependencies prüfen
            if not self.check_input_dependencies():
                self.logger.warning("Cannot generate water system - missing dependencies")
                return

            self.generation_in_progress = True
            self.logger.info("Starting water system generation...")

            # Timing für Performance-Messung starten
            self.start_generation_timing()

            # Alle Input-Daten sammeln
            inputs = self.collect_input_data()
            params = self.current_parameters.copy()

            # GPU-Performance Monitoring starten
            if self.gpu_available:
                self.shader_performance_timer.start(1000)

            # 1. Lake Detection (Jump Flooding Algorithm)
            self.logger.info("Step 1: Lake Detection")
            lake_map = self.lake_detection.detect_local_minima(
                heightmap=inputs["heightmap"],
                lake_volume_threshold=params["lake_volume_threshold"]
            )

            # 2. Flow Network Building (Steepest Descent + Upstream Accumulation)
            self.logger.info("Step 2: Flow Network Building")
            flow_data = self.flow_network.calculate_steepest_descent(
                heightmap=inputs["heightmap"],
                precipitation=inputs["precip_map"],
                rain_threshold=params["rain_threshold"]
            )

            # 3. Manning Flow Calculation
            self.logger.info("Step 3: Manning Flow Calculation")
            flow_results = self.manning_calculator.solve_manning_equation(
                flow_map=flow_data["flow_map"],
                heightmap=inputs["heightmap"],
                manning_coefficient=params["manning_coefficient"]
            )

            # 4. Soil Moisture Calculation (Gaussian Diffusion)
            self.logger.info("Step 4: Soil Moisture Calculation")
            soil_moisture_map = self.soil_moisture.apply_gaussian_diffusion(
                water_map=flow_results["water_map"],
                diffusion_radius=params["diffusion_radius"]
            )

            # 5. Erosion & Sedimentation System (Stream Power + Hjulström)
            self.logger.info("Step 5: Erosion & Sedimentation")
            erosion_results = self.erosion_system.calculate_stream_power(
                flow_speed=flow_results["flow_speed"],
                water_depth=flow_results["water_map"],
                hardness_map=inputs["hardness_map"],
                rock_map=inputs["rock_map"],
                erosion_strength=params["erosion_strength"],
                sediment_capacity_factor=params["sediment_capacity_factor"],
                settling_velocity=params["settling_velocity"]
            )

            # 6. Evaporation Calculation (Atmospheric Conditions)
            self.logger.info("Step 6: Evaporation Calculation")
            evaporation_map = self.evaporation.calculate_atmospheric_evaporation(
                water_map=flow_results["water_map"],
                temp_map=inputs["temp_map"],
                humid_map=inputs["humid_map"],
                wind_map=inputs["wind_map"],
                evaporation_base_rate=params["evaporation_base_rate"]
            )

            # 7. Water Biomes Classification
            self.logger.info("Step 7: Water Biomes Classification")
            water_biomes_map = self.flow_network.classify_water_bodies(
                flow_map=flow_data["flow_map"],
                lake_map=lake_map
            )

            # Alle Results im DataManager speichern
            self.save_all_water_results(
                flow_results, erosion_results, soil_moisture_map,
                evaporation_map, water_biomes_map, params
            )

            # Display und Statistics aktualisieren
            self.update_water_display()
            self.hydrology_stats.update_generation_statistics(flow_results, erosion_results)

            # Ocean Outflow anzeigen
            ocean_outflow = flow_data.get("ocean_outflow", 0.0)
            self.ocean_outflow_label.setText(f"Ocean Outflow: {ocean_outflow:.2f} m³/s")

            # Timing beenden
            self.end_generation_timing(True)

            self.logger.info("Water system generation completed successfully")


        except Exception as e:
            self.handle_generation_error(e)
            self.end_generation_timing(False, str(e))
            raise  # Re-raise für Error Handler


        finally:
            self.generation_in_progress = False
            if self.gpu_available:
                self.shader_performance_timer.stop()

    def collect_input_data(self) -> dict:
        """
        Funktionsweise: Sammelt alle Required Input-Daten von DataManager
        Return: dict mit allen benötigten Arrays für Water-Generation
        """
        inputs = {}

        # Terrain Inputs
        inputs["heightmap"] = self.data_manager.get_terrain_data("heightmap")
        inputs["slopemap"] = self.data_manager.get_terrain_data("slopemap")

        # Geology Inputs
        inputs["hardness_map"] = self.data_manager.get_geology_data("hardness_map")
        inputs["rock_map"] = self.data_manager.get_geology_data("rock_map")

        # Weather Inputs
        inputs["precip_map"] = self.data_manager.get_weather_data("precip_map")
        inputs["temp_map"] = self.data_manager.get_weather_data("temp_map")
        inputs["wind_map"] = self.data_manager.get_weather_data("wind_map")
        inputs["humid_map"] = self.data_manager.get_weather_data("humid_map")

        # Validation
        for key, data in inputs.items():
            if data is None:
                raise ValueError(f"Required input '{key}' not available")

        return inputs

    def save_all_water_results(self, flow_results: dict, erosion_results: dict,
                               soil_moisture_map: np.ndarray, evaporation_map: np.ndarray,
                               water_biomes_map: np.ndarray, params: dict):
        """
        Funktionsweise: Speichert alle Water-System Results im DataManager
        Parameter: Alle Result-Dictionaries und Arrays von Core-Generatoren
        """
        # Flow Results
        self.data_manager.set_water_data("water_map", flow_results["water_map"], params)
        self.data_manager.set_water_data("flow_map", flow_results["flow_map"], params)
        self.data_manager.set_water_data("flow_speed", flow_results["flow_speed"], params)
        self.data_manager.set_water_data("cross_section", flow_results["cross_section"], params)

        # Erosion Results
        self.data_manager.set_water_data("erosion_map", erosion_results["erosion_map"], params)
        self.data_manager.set_water_data("sedimentation_map", erosion_results["sedimentation_map"], params)
        self.data_manager.set_water_data("rock_map_updated", erosion_results["rock_map_updated"], params)

        # Additional Results
        self.data_manager.set_water_data("soil_moist_map", soil_moisture_map, params)
        self.data_manager.set_water_data("evaporation_map", evaporation_map, params)
        self.data_manager.set_water_data("water_biomes_map", water_biomes_map, params)

        # Ocean Outflow (Scalar)
        ocean_outflow = flow_results.get("ocean_outflow", 0.0)
        self.data_manager.set_water_data("ocean_outflow", ocean_outflow, params)

    @gpu_shader_handler("water_display")
    def update_water_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt ausgewählte Water-Map mit optionalen Overlays
        """
        current_mode = self.display_mode.checkedId()
        current_radio = self.display_mode.button(current_mode)

        if current_radio is None:
            return

        data_key = current_radio.property("data_key")
        water_data = self.data_manager.get_water_data(data_key)

        if water_data is None:
            return

        # Spezielle Behandlung für verschiedene Datentypen
        if data_key == "water_map":
            self.map_display.display_water_depth(water_data)
        elif data_key == "flow_map":
            self.map_display.display_flow_volume(water_data)
        elif data_key == "flow_speed":
            self.map_display.display_flow_speed(water_data)
        elif data_key == "cross_section":
            self.map_display.display_cross_section(water_data)
        elif data_key == "soil_moist_map":
            self.map_display.display_soil_moisture(water_data)
        elif data_key == "erosion_map":
            self.map_display.display_erosion_rate(water_data)
        elif data_key == "sedimentation_map":
            self.map_display.display_sedimentation_rate(water_data)
        elif data_key == "rock_map_updated":
            self.map_display.display_rock_map_rgb(water_data)
        elif data_key == "evaporation_map":
            self.map_display.display_evaporation_rate(water_data)
        elif data_key == "water_biomes_map":
            self.map_display.display_water_biomes(water_data)

        # River Network Overlay
        if self.river_overlay_checkbox.isChecked():
            flow_map = self.data_manager.get_water_data("flow_map")
            if flow_map is not None:
                self.map_display.overlay_river_network(flow_map)

        # 3D Terrain Overlay
        if self.terrain_3d_checkbox.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.overlay_3d_terrain(heightmap)

    @pyqtSlot()
    def update_display_mode(self):
        """Slot für Visualization-Mode Änderungen"""
        self.update_water_display()

    @pyqtSlot(bool)
    def toggle_river_overlay(self, enabled: bool):
        """Toggle für River Network Overlay"""
        self.update_water_display()

    @pyqtSlot(bool)
    def toggle_3d_terrain(self, enabled: bool):
        """Toggle für 3D Terrain Overlay"""
        self.update_water_display()

    @pyqtSlot()
    def monitor_shader_performance(self):
        """
        Funktionsweise: Überwacht GPU-Shader Performance während Generation
        Aufgabe: Updated Shader-Performance Panel mit aktuellen Metriken
        """
        if self.generation_in_progress and self.gpu_available:
            performance_metrics = self.shader_manager.get_performance_metrics()
            self.shader_controls.update_performance_display(performance_metrics)

    # Override BaseMapTab method
    def generate_terrain(self):
        """Override für Water-spezifische Generation"""
        self.generate_water_system()


class HydrologyStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Widget für umfassende Hydrologie-Statistiken
    Aufgabe: Zeigt Parameter-Preview, Generation-Statistics, Performance-Metriken
    """

    def __init__(self):
        super().__init__("Hydrology Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Hydrologie-Statistiken"""
        layout = QVBoxLayout()

        # Parameter Preview
        preview_group = QGroupBox("Parameter Preview")
        preview_layout = QVBoxLayout()

        self.lake_threshold_label = QLabel("Lake Threshold: 0.1m")
        self.erosion_strength_label = QLabel("Erosion Strength: 1.0")
        self.manning_coeff_label = QLabel("Manning Coefficient: 0.03")

        preview_layout.addWidget(self.lake_threshold_label)
        preview_layout.addWidget(self.erosion_strength_label)
        preview_layout.addWidget(self.manning_coeff_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Generation Statistics
        stats_group = QGroupBox("Generation Results")
        stats_layout = QVBoxLayout()

        self.water_coverage_label = QLabel("Water Coverage: -")
        self.avg_flow_speed_label = QLabel("Avg Flow Speed: -")
        self.total_erosion_label = QLabel("Total Erosion: -")
        self.mass_conservation_label = QLabel("Mass Conservation: -")

        stats_layout.addWidget(self.water_coverage_label)
        stats_layout.addWidget(self.avg_flow_speed_label)
        stats_layout.addWidget(self.total_erosion_label)
        stats_layout.addWidget(self.mass_conservation_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        self.setLayout(layout)

    def update_parameter_preview(self, parameters: dict):
        """Aktualisiert Parameter-Preview"""
        self.lake_threshold_label.setText(f"Lake Threshold: {parameters.get('lake_volume_threshold', 0.1):.3f}m")
        self.erosion_strength_label.setText(f"Erosion Strength: {parameters.get('erosion_strength', 1.0):.1f}")
        self.manning_coeff_label.setText(f"Manning Coefficient: {parameters.get('manning_coefficient', 0.03):.3f}")

    def update_generation_statistics(self, flow_results: dict, erosion_results: dict):
        """
        Funktionsweise: Aktualisiert Statistiken nach Generation
        Parameter: flow_results, erosion_results (dict von Core-Generatoren)
        """
        # Water Coverage berechnen
        water_map = flow_results.get("water_map")
        if water_map is not None:
            water_pixels = np.sum(water_map > 0.01)  # > 1cm Wassertiefe
            total_pixels = water_map.shape[0] * water_map.shape[1]
            water_coverage = (water_pixels / total_pixels) * 100
            self.water_coverage_label.setText(f"Water Coverage: {water_coverage:.1f}%")

        # Average Flow Speed
        flow_speed = flow_results.get("flow_speed")
        if flow_speed is not None:
            avg_speed = np.mean(flow_speed[flow_speed > 0])
            self.avg_flow_speed_label.setText(f"Avg Flow Speed: {avg_speed:.2f} m/s")

        # Total Erosion
        erosion_map = erosion_results.get("erosion_map")
        if erosion_map is not None:
            total_erosion = np.sum(erosion_map)
            self.total_erosion_label.setText(f"Total Erosion: {total_erosion:.1f} m/Jahr")

        # Mass Conservation Check
        rock_map_updated = erosion_results.get("rock_map_updated")
        if rock_map_updated is not None:
            mass_sums = np.sum(rock_map_updated, axis=2)
            mass_conserved = np.allclose(mass_sums, 255, atol=1)
            status = "✓ Conserved" if mass_conserved else "✗ Violated"
            self.mass_conservation_label.setText(f"Mass Conservation: {status}")


class MultiDependencyStatusWidget(QGroupBox):
    """
    Funktionsweise: Widget für detaillierte Dependency-Status Anzeige
    Aufgabe: Zeigt Status aller 8 Required Dependencies einzeln an
    """

    def __init__(self, required_dependencies: list):
        super().__init__("Input Dependencies")
        self.required_dependencies = required_dependencies
        self.dependency_indicators = {}
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Dependency-Status"""
        layout = QVBoxLayout()

        # Status Indicator für jede Dependency
        for dependency in self.required_dependencies:
            indicator = StatusIndicator(dependency.replace("_", " ").title())
            indicator.set_warning("Not available")
            self.dependency_indicators[dependency] = indicator
            layout.addWidget(indicator)

        # Overall Status
        self.overall_status = StatusIndicator("Overall Status")
        self.overall_status.set_error("Dependencies missing")
        layout.addWidget(self.overall_status)

        self.setLayout(layout)

    def update_dependency_status(self, is_complete: bool, missing: list):
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

    def update_missing_dependencies(self, missing: list):
        """Update nur für fehlende Dependencies"""
        self.update_dependency_status(len(missing) == 0, missing)

    def set_error(self, message: str):
        """Setzt Error-Status für Overall Status"""
        self.overall_status.set_error(message)


class ShaderPerformancePanel(QGroupBox):
    """
    Funktionsweise: Widget für GPU-Shader Performance Monitoring
    Aufgabe: Zeigt Performance-Metriken aller 8 Water-Shader an
    """

    def __init__(self, shader_manager):
        super().__init__("Shader Performance")
        self.shader_manager = shader_manager
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Shader-Performance Display"""
        layout = QVBoxLayout()

        # GPU Status
        self.gpu_status = StatusIndicator("GPU Status")
        if self.shader_manager.check_gpu_support():
            self.gpu_status.set_success("GPU Available")
        else:
            self.gpu_status.set_warning("Using CPU Fallback")
        layout.addWidget(self.gpu_status)

        # Performance Metrics
        self.frame_time_label = QLabel("Frame Time: - ms")
        self.memory_usage_label = QLabel("GPU Memory: - MB")
        self.shader_count_label = QLabel("Active Shaders: 0/8")

        layout.addWidget(self.frame_time_label)
        layout.addWidget(self.memory_usage_label)
        layout.addWidget(self.shader_count_label)

        # Performance Progress Bar
        self.performance_bar = QProgressBar()
        self.performance_bar.setRange(0, 100)
        self.performance_bar.setValue(100)
        layout.addWidget(QLabel("Performance:"))
        layout.addWidget(self.performance_bar)

        self.setLayout(layout)

    def update_performance_display(self, metrics: dict):
        """
        Funktionsweise: Aktualisiert Performance-Display mit aktuellen Metriken
        Parameter: metrics (dict mit performance data vom shader_manager)
        """
        # Frame Time
        frame_time = metrics.get("frame_time_ms", 0)
        self.frame_time_label.setText(f"Frame Time: {frame_time:.1f} ms")

        # GPU Memory Usage
        memory_usage = metrics.get("gpu_memory_mb", 0)
        self.memory_usage_label.setText(f"GPU Memory: {memory_usage:.0f} MB")

        # Active Shaders
        active_shaders = metrics.get("active_shaders", 0)
        self.shader_count_label.setText(f"Active Shaders: {active_shaders}/8")

        # Performance Percentage (basierend auf Frame Time)
        performance_pct = max(0, min(100, 100 - (frame_time - 16)))  # 60fps = 16ms target
        self.performance_bar.setValue(int(performance_pct))

        # Performance Bar Color
        if performance_pct >= 80:
            self.performance_bar.setStyleSheet("QProgressBar::chunk { background-color: #27ae60; }")
        elif performance_pct >= 50:
            self.performance_bar.setStyleSheet("QProgressBar::chunk { background-color: #f39c12; }")
        else:
            self.performance_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")