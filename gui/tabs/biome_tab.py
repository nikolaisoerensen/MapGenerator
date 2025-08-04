"""
Path: gui/tabs/biome_tab.py

Funktionsweise: Finale Biome-Klassifizierung mit vollständiger Core-Integration
- Input: heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map
- Whittaker-Biome Classification mit 15 Base-Biomes + 11 Super-Biomes
- 2x2 Supersampling mit diskretisierter Zufalls-Rotation
- Integration aller Systeme in finale Welt-Darstellung
- Export-Funktionalität für komplette Welt
- Gauß-basierte Klassifikation mit konfigurierbaren Gewichtungen
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import BIOME, get_parameter_config, validate_parameter_set, VALIDATION_RULES
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton, WorldExportWidget, WorldStatisticsWidget, \
    MultiDependencyStatusWidget, BiomeLegendDialog
from core.biome_generator import (
    BiomeClassificationSystem, BaseBiomeClassifier, SuperBiomeOverrideSystem,
    SupersamplingManager, ProximityBiomeCalculator
)


class BiomeTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für finale Biome-Classification mit allen Features
    Aufgabe: Koordiniert Base-Biomes, Super-Biomes, Supersampling und Export
    Input: Alle Generator-Outputs für vollständige Welt-Integration
    Output: biome_map, biome_map_super, super_biome_mask für finale Darstellung
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)
        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.biome_classification = BiomeClassificationSystem()
        self.base_classifier = BaseBiomeClassifier()
        self.super_biome_system = SuperBiomeOverrideSystem()
        self.supersampling_manager = SupersamplingManager()
        self.proximity_calculator = ProximityBiomeCalculator()

        # Parameter und State
        self.current_parameters = {}
        self.biome_generation_complete = False

        # Setup UI
        self.setup_biome_ui()
        self.setup_dependency_checking()

        # Initial Load
        self.load_default_parameters()
        self.check_input_dependencies()

    @memory_critical_handler("biome_generation")
    def generate(self):
        """
        Funktionsweise: Hauptmethode für Biome-Generation mit Orchestrator Integration
        Aufgabe: Startet Biome-Generation über GenerationOrchestrator mit Target-LOD
        """
        if not self.generation_orchestrator:
            self.logger.error("No GenerationOrchestrator available")
            self.handle_generation_error(Exception("GenerationOrchestrator not available"))
            return

        if self.generation_in_progress:
            self.logger.info("Generation already in progress, ignoring request")
            return

        if not self.check_input_dependencies():
            self.logger.warning("Cannot generate biome system - missing dependencies")
            return

        try:
            self.logger.info(f"Starting biome generation with target LOD: {self.target_lod}")

            self.start_generation_timing()
            self.generation_in_progress = True

            request_id = self.generation_orchestrator.request_generation(
                generator_type="biome",
                parameters=self.current_parameters.copy(),
                target_lod=self.target_lod,
                source_tab="biome",
                priority=10
            )

            if request_id:
                self.logger.info(f"Biome generation requested: {request_id}")
            else:
                raise Exception("Failed to request generation from orchestrator")

        except Exception as e:
            self.generation_in_progress = False
            self.handle_generation_error(e)
            raise

    def setup_biome_ui(self):
        """
        Funktionsweise: Erstellt komplette UI für Biome-System
        Aufgabe: Parameter, Biome-Preview, World-Overview, Export-Controls
        """
        # Parameter Panel
        self.parameter_panel = self.create_biome_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)

        # Biome Classification Widget
        self.biome_classification_widget = BiomeClassificationWidget()
        self.control_panel.layout().addWidget(self.biome_classification_widget)

        # Visualization Controls
        self.visualization_controls = self.create_biome_visualization_controls()
        self.control_panel.layout().addWidget(self.visualization_controls)

        # World Statistics
        self.world_stats = WorldStatisticsWidget()
        self.control_panel.layout().addWidget(self.world_stats)

        # Export Controls
        self.export_controls = WorldExportWidget()
        self.export_controls.export_requested.connect(self.export_world_data)
        self.control_panel.layout().addWidget(self.export_controls)

        # Dependencies und Navigation
        self.setup_input_status()
        self.setup_navigation_panel()

    def create_biome_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Biome-Parametern
        Aufgabe: Alle 8 Parameter aus value_default.BIOME
        Return: QGroupBox mit strukturierten Parameter-Slidern
        """
        panel = QGroupBox("Biome Parameters")
        layout = QVBoxLayout()

        self.parameter_sliders = {}

        # Climate Weighting Parameters
        climate_group = QGroupBox("Climate Weighting")
        climate_layout = QVBoxLayout()

        climate_params = ["biome_wetness_factor", "biome_temp_factor"]
        for param_name in climate_params:
            param_config = get_parameter_config("biome", param_name)

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
            climate_layout.addWidget(slider)

        climate_group.setLayout(climate_layout)
        layout.addWidget(climate_group)

        # Sea Level and Bank Parameters
        water_group = QGroupBox("Water Features")
        water_layout = QVBoxLayout()

        water_params = ["sea_level", "bank_width"]
        for param_name in water_params:
            param_config = get_parameter_config("biome", param_name)

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
            water_layout.addWidget(slider)

        water_group.setLayout(water_layout)
        layout.addWidget(water_group)

        # Elevation Zones
        elevation_group = QGroupBox("Elevation Zones")
        elevation_layout = QVBoxLayout()

        elevation_params = ["alpine_level", "snow_level", "cliff_slope"]
        for param_name in elevation_params:
            param_config = get_parameter_config("biome", param_name)

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
            elevation_layout.addWidget(slider)

        elevation_group.setLayout(elevation_layout)
        layout.addWidget(elevation_group)

        # Edge Softness (Global Super-Biome Control)
        softness_group = QGroupBox("Transition Softness")
        softness_layout = QVBoxLayout()

        param_config = get_parameter_config("biome", "edge_softness")
        slider = ParameterSlider(
            label="Edge Softness (Global)",
            min_val=param_config["min"],
            max_val=param_config["max"],
            default_val=param_config["default"],
            step=param_config.get("step", 0.1),
            suffix=param_config.get("suffix", "")
        )

        slider.valueChanged.connect(self.on_parameter_changed)
        self.parameter_sliders["edge_softness"] = slider
        softness_layout.addWidget(slider)

        softness_group.setLayout(softness_layout)
        layout.addWidget(softness_group)

        panel.setLayout(layout)
        return panel

    def create_biome_visualization_controls(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Controls für Biome-Visualization
        Aufgabe: Switcher zwischen biome_map, biome_map_super, super_biome_mask
        Return: QGroupBox mit Visualization-Controls
        """
        panel = QGroupBox("Biome Visualization")
        layout = QVBoxLayout()

        # Display Mode Selection
        self.display_mode = QButtonGroup()

        self.base_biomes_radio = QRadioButton("Base Biomes")
        self.base_biomes_radio.setChecked(True)
        self.base_biomes_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.base_biomes_radio, 0)
        layout.addWidget(self.base_biomes_radio)

        self.super_biomes_radio = QRadioButton("Super Biomes (2x2 Supersampled)")
        self.super_biomes_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.super_biomes_radio, 1)
        layout.addWidget(self.super_biomes_radio)

        self.override_mask_radio = QRadioButton("Super-Biome Override Mask")
        self.override_mask_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.override_mask_radio, 2)
        layout.addWidget(self.override_mask_radio)

        # Overlay Controls
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout()

        self.settlements_overlay = QCheckBox("Show Settlements")
        self.settlements_overlay.toggled.connect(self.toggle_settlements_overlay)
        overlay_layout.addWidget(self.settlements_overlay)

        self.rivers_overlay = QCheckBox("Show Rivers")
        self.rivers_overlay.toggled.connect(self.toggle_rivers_overlay)
        overlay_layout.addWidget(self.rivers_overlay)

        self.elevation_contours = QCheckBox("Show Elevation Contours")
        self.elevation_contours.toggled.connect(self.toggle_elevation_contours)
        overlay_layout.addWidget(self.elevation_contours)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        # Biome Legend Toggle
        self.show_legend_button = BaseButton("Show Biome Legend")
        self.show_legend_button.clicked.connect(self.show_biome_legend)
        layout.addWidget(self.show_legend_button)

        panel.setLayout(layout)
        return panel

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking
        Aufgabe: Überwacht alle Required Dependencies für Biome-System
        """
        # Required Dependencies für Biome-System
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["biome"]

        # Dependency Status Widget
        self.dependency_status = MultiDependencyStatusWidget(self.required_dependencies)
        self.control_panel.layout().addWidget(self.dependency_status)

        # Data Manager Signals
        self.data_manager.data_updated.connect(self.on_data_updated)

    def load_default_parameters(self):
        """Lädt Default-Parameter"""
        for param_name, slider in self.parameter_sliders.items():
            param_config = get_parameter_config("biome", param_name)
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

        # Parameter Validation
        is_valid, warnings, errors = validate_parameter_set("biome", self.current_parameters)

        if errors:
            self.biome_classification_widget.show_validation_errors(errors)
        elif warnings:
            self.biome_classification_widget.show_validation_warnings(warnings)
        else:
            self.biome_classification_widget.clear_validation_messages()

        # Biome Classification Preview aktualisieren
        self.biome_classification_widget.update_parameter_preview(self.current_parameters)

        # Auto-Simulation triggern
        if self.auto_simulation_enabled:
            self.auto_simulation_timer.start(1000)

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """Slot für Data-Updates von anderen Generatoren"""
        if data_key in self.required_dependencies:
            self.check_input_dependencies()

    def check_input_dependencies(self):
        """
        Funktionsweise: Prüft alle Required Dependencies für Biome-System
        Aufgabe: Aktiviert/Deaktiviert Generation basierend auf verfügbaren Inputs
        """
        is_complete, missing = self.data_manager.check_dependencies("biome", self.required_dependencies)

        self.dependency_status.update_dependency_status(is_complete, missing)
        self.manual_generate_button.setEnabled(is_complete)

        # Export nur aktivieren wenn Biome-Generation complete
        self.export_controls.setEnabled(is_complete and self.biome_generation_complete)

        return is_complete

    def collect_input_data(self) -> dict:
        """
        Funktionsweise: Sammelt alle Required Input-Daten von DataManager
        Return: dict mit allen benötigten Arrays für Biome-Generation
        """
        inputs = {}

        # Terrain Inputs
        inputs["heightmap"] = self.data_manager.get_terrain_data("heightmap")
        inputs["slopemap"] = self.data_manager.get_terrain_data("slopemap")

        # Weather Inputs
        inputs["temp_map"] = self.data_manager.get_weather_data("temp_map")

        # Water Inputs
        inputs["soil_moist_map"] = self.data_manager.get_water_data("soil_moist_map")
        inputs["water_biomes_map"] = self.data_manager.get_water_data("water_biomes_map")

        # Validation
        for key, data in inputs.items():
            if data is None:
                raise ValueError(f"Required input '{key}' not available")

        return inputs

    def update_biome_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt biome_map, biome_map_super oder super_biome_mask mit Overlays
        """
        current_mode = self.display_mode.checkedId()

        if current_mode == 0:  # Base Biomes
            biome_map = self.data_manager.get_biome_data("biome_map")
            if biome_map is not None:
                self.map_display.display_base_biomes(biome_map)

        elif current_mode == 1:  # Super Biomes (2x2 Supersampled)
            biome_map_super = self.data_manager.get_biome_data("biome_map_super")
            if biome_map_super is not None:
                self.map_display.display_super_biomes(biome_map_super)

        elif current_mode == 2:  # Super-Biome Override Mask
            super_biome_mask = self.data_manager.get_biome_data("super_biome_mask")
            if super_biome_mask is not None:
                self.map_display.display_super_biome_mask(super_biome_mask)

        # Overlays anwenden
        self.apply_overlays()

    def apply_overlays(self):
        """
        Funktionsweise: Wendet alle aktivierten Overlays auf Display an
        Aufgabe: Settlements, Rivers, Elevation Contours basierend auf Checkboxes
        """
        # Settlements Overlay
        if self.settlements_overlay.isChecked():
            settlement_list = self.data_manager.get_settlement_data("settlement_list")
            landmark_list = self.data_manager.get_settlement_data("landmark_list")
            if settlement_list is not None:
                self.map_display.overlay_settlements(settlement_list, landmark_list)

        # Rivers Overlay
        if self.rivers_overlay.isChecked():
            flow_map = self.data_manager.get_water_data("flow_map")
            if flow_map is not None:
                self.map_display.overlay_river_network(flow_map)

        # Elevation Contours Overlay
        if self.elevation_contours.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.overlay_elevation_contours(heightmap)

    @pyqtSlot()
    def update_display_mode(self):
        """Slot für Visualization-Mode Änderungen"""
        self.update_biome_display()

    @pyqtSlot(bool)
    def toggle_settlements_overlay(self, enabled: bool):
        """Toggle für Settlements Overlay"""
        self.update_biome_display()

    @pyqtSlot(bool)
    def toggle_rivers_overlay(self, enabled: bool):
        """Toggle für Rivers Overlay"""
        self.update_biome_display()

    @pyqtSlot(bool)
    def toggle_elevation_contours(self, enabled: bool):
        """Toggle für Elevation Contours Overlay"""
        self.update_biome_display()

    @pyqtSlot()
    def show_biome_legend(self):
        """
        Funktionsweise: Zeigt Biome-Legend Dialog mit allen 26 Biome-Typen
        Aufgabe: Übersichtliche Darstellung aller Base- und Super-Biomes
        """
        legend_dialog = BiomeLegendDialog(self)
        legend_dialog.exec_()

    @pyqtSlot(str, dict)
    def export_world_data(self, export_format: str, export_options: dict):
        """
        Funktionsweise: Exportiert komplette Welt-Daten in verschiedene Formate
        Parameter: export_format ("png", "json", "obj"), export_options (dict)
        """
        try:
            self.logger.info(f"Starting world export in format: {export_format}")

            # Alle verfügbaren Daten sammeln
            world_data = self.collect_all_world_data()

            if export_format == "png":
                self.export_png_maps(world_data, export_options)
            elif export_format == "json":
                self.export_json_data(world_data, export_options)
            elif export_format == "obj":
                self.export_3d_terrain(world_data, export_options)

            self.logger.info("World export completed successfully")

        except Exception as e:
            self.logger.error(f"World export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")

    def collect_all_world_data(self) -> dict:
        """
        Funktionsweise: Sammelt alle verfügbaren Welt-Daten für Export
        Return: dict mit allen Generator-Outputs
        """
        world_data = {
            "terrain": {
                "heightmap": self.data_manager.get_terrain_data("heightmap"),
                "slopemap": self.data_manager.get_terrain_data("slopemap"),
                "shademap": self.data_manager.get_terrain_data("shademap")
            },
            "geology": {
                "rock_map": self.data_manager.get_geology_data("rock_map"),
                "hardness_map": self.data_manager.get_geology_data("hardness_map")
            },
            "settlements": {
                "settlement_list": self.data_manager.get_settlement_data("settlement_list"),
                "landmark_list": self.data_manager.get_settlement_data("landmark_list"),
                "civ_map": self.data_manager.get_settlement_data("civ_map")
            },
            "weather": {
                "temp_map": self.data_manager.get_weather_data("temp_map"),
                "precip_map": self.data_manager.get_weather_data("precip_map"),
                "wind_map": self.data_manager.get_weather_data("wind_map")
            },
            "water": {
                "water_map": self.data_manager.get_water_data("water_map"),
                "flow_map": self.data_manager.get_water_data("flow_map"),
                "soil_moist_map": self.data_manager.get_water_data("soil_moist_map")
            },
            "biomes": {
                "biome_map": self.data_manager.get_biome_data("biome_map"),
                "biome_map_super": self.data_manager.get_biome_data("biome_map_super"),
                "super_biome_mask": self.data_manager.get_biome_data("super_biome_mask")
            },
            "parameters": {
                "terrain": self.get_all_parameters("terrain"),
                "geology": self.get_all_parameters("geology"),
                "settlement": self.get_all_parameters("settlement"),
                "weather": self.get_all_parameters("weather"),
                "water": self.get_all_parameters("water"),
                "biome": self.current_parameters
            }
        }

        return world_data

    def get_all_parameters(self, generator_type: str) -> dict:
        """
        Funktionsweise: Holt Parameter von anderen Tabs für Export
        Parameter: generator_type (str)
        Return: dict mit Parametern oder leerer dict
        """
        # Hier würde normalerweise auf andere Tabs zugegriffen werden
        # Für jetzt leerer dict als Fallback
        return {}

    def export_png_maps(self, world_data: dict, options: dict):
        """Exportiert alle Maps als PNG-Dateien"""
        import os
        from matplotlib import pyplot as plt

        export_dir = options.get("export_directory", ".")

        for category, maps in world_data.items():
            if category == "parameters":
                continue

            category_dir = os.path.join(export_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            for map_name, map_data in maps.items():
                if map_data is not None and isinstance(map_data, np.ndarray):
                    plt.figure(figsize=(10, 10))
                    if len(map_data.shape) == 3:  # RGB Map
                        plt.imshow(map_data)
                    else:  # 2D Map
                        plt.imshow(map_data, cmap='viridis')
                    plt.title(f"{category.title()} - {map_name.replace('_', ' ').title()}")
                    plt.colorbar()
                    plt.savefig(os.path.join(category_dir, f"{map_name}.png"), dpi=300, bbox_inches='tight')
                    plt.close()

    def export_json_data(self, world_data: dict, options: dict):
        """Exportiert alle Daten als JSON"""
        import json
        import os

        export_file = options.get("export_file", "world_data.json")

        # Numpy Arrays zu Listen konvertieren für JSON
        json_data = {}
        for category, data in world_data.items():
            json_data[category] = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[category][key] = value.tolist()
                else:
                    json_data[category][key] = value

        with open(export_file, 'w') as f:
            json.dump(json_data, f, indent=2)

    def export_3d_terrain(self, world_data: dict, options: dict):
        """Exportiert 3D-Terrain als OBJ-Datei"""
        heightmap = world_data["terrain"]["heightmap"]
        if heightmap is None:
            raise ValueError("Heightmap not available for 3D export")

        export_file = options.get("export_file", "terrain.obj")

        # Vereinfachte OBJ-Export Implementation
        with open(export_file, 'w') as f:
            f.write("# Generated by Map Generator\n")

            # Vertices
            height, width = heightmap.shape
            for y in range(height):
                for x in range(width):
                    z = heightmap[y, x]
                    f.write(f"v {x} {z} {y}\n")

            # Faces (Triangles)
            for y in range(height - 1):
                for x in range(width - 1):
                    # Indices (1-based for OBJ)
                    v1 = y * width + x + 1
                    v2 = y * width + (x + 1) + 1
                    v3 = (y + 1) * width + x + 1
                    v4 = (y + 1) * width + (x + 1) + 1

                    # Two triangles per quad
                    f.write(f"f {v1} {v2} {v3}\n")
                    f.write(f"f {v2} {v4} {v3}\n")

class BiomeClassificationWidget(QGroupBox):
    """
    Funktionsweise: Widget für Biome-Classification Status und Preview
    Aufgabe: Zeigt Parameter-Preview, Validation-Messages, Biome-Verteilung
    """

    def __init__(self):
        super().__init__("Biome Classification")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Biome-Classification"""
        layout = QVBoxLayout()

        # Parameter Preview
        self.parameter_preview = QLabel("Parameters: Default values")
        layout.addWidget(self.parameter_preview)

        # Validation Messages
        self.validation_status = StatusIndicator("Parameter Validation")
        self.validation_status.set_success("Parameters valid")
        layout.addWidget(self.validation_status)

        # Biome Distribution (wird nach Generation gefüllt)
        self.biome_distribution = QLabel("Biome Distribution: Not generated")
        layout.addWidget(self.biome_distribution)

        # Super-Biome Statistics
        self.super_biome_stats = QLabel("Super-Biome Override: Not generated")
        layout.addWidget(self.super_biome_stats)

        self.setLayout(layout)

    def update_parameter_preview(self, parameters: dict):
        """Aktualisiert Parameter-Preview"""
        preview_text = f"Wetness: {parameters.get('biome_wetness_factor', 1.0):.1f}, "
        preview_text += f"Temp: {parameters.get('biome_temp_factor', 1.0):.1f}, "
        preview_text += f"Softness: {parameters.get('edge_softness', 1.0):.1f}"
        self.parameter_preview.setText(f"Parameters: {preview_text}")

    def show_validation_errors(self, errors: list):
        """Zeigt Validation-Errors"""
        self.validation_status.set_error(f"Errors: {'; '.join(errors)}")

    def show_validation_warnings(self, warnings: list):
        """Zeigt Validation-Warnings"""
        self.validation_status.set_warning(f"Warnings: {'; '.join(warnings)}")

    def clear_validation_messages(self):
        """Löscht Validation-Messages"""
        self.validation_status.set_success("Parameters valid")

    def update_classification_statistics(self, biome_map: np.ndarray, super_biome_mask: np.ndarray):
        """
        Funktionsweise: Aktualisiert Biome-Statistiken nach Generation
        Parameter: biome_map, super_biome_mask (numpy arrays)
        """
        # Base-Biome Distribution
        unique_biomes, counts = np.unique(biome_map, return_counts=True)
        total_pixels = biome_map.shape[0] * biome_map.shape[1]

        most_common_biome_idx = np.argmax(counts)
        most_common_biome = unique_biomes[most_common_biome_idx]
        most_common_pct = (counts[most_common_biome_idx] / total_pixels) * 100

        self.biome_distribution.setText(
            f"Biome Distribution: {len(unique_biomes)} types, "
            f"most common: Biome #{most_common_biome} ({most_common_pct:.1f}%)"
        )

        # Super-Biome Override Statistics
        override_pixels = np.sum(super_biome_mask > 0)
        override_pct = (override_pixels / total_pixels) * 100

        self.super_biome_stats.setText(
            f"Super-Biome Override: {override_pct:.1f}% of map overridden"
        )