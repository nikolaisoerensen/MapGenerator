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
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton, BiomeLegendDialog
from core.biome_generator import (
    BiomeClassificationSystem, BaseBiomeClassifier, SuperBiomeOverrideSystem,
    SupersamplingManager
)


def get_biome_error_decorators():
    """
    Funktionsweise: Lazy Loading von Biome Tab Error Decorators
    Aufgabe: Lädt Memory-Critical, GPU-Shader und Core-Generation Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.utils.error_handler import memory_critical_handler, gpu_shader_handler, core_generation_handler
        return memory_critical_handler, gpu_shader_handler, core_generation_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

memory_critical_handler, gpu_shader_handler, core_generation_handler = get_biome_error_decorators()

class BiomeTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für finale Biome-Classification mit allen Features
    Aufgabe: Koordiniert Base-Biomes, Super-Biomes, Supersampling und Export
    Input: Alle Generator-Outputs für vollständige Welt-Integration
    Output: biome_map, biome_map_super, super_biome_mask für finale Darstellung
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):
        self.generator_type = "biome"

        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )
        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.biome_classification = BiomeClassificationSystem()
        self.base_classifier = BaseBiomeClassifier()
        self.super_biome_system = SuperBiomeOverrideSystem()
        self.supersampling_manager = SupersamplingManager()

        # Parameter und State
        self.current_parameters = {}
        self.biome_generation_complete = False
        self.generation_in_progress = False

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
            # target_lod nicht selbst gesetzt - wie bei den anderen Tabs
            # bestimmt der Orchestrator es aus map_size/aktuellem Terrain-LOD
            # (request_generation() mit target_lod=None), siehe
            # GenerationOrchestrator.request_generation().
            self.logger.info("Starting biome generation")

            self.start_generation_timing()
            self.generation_in_progress = True

            # Frisch von den Slidern lesen statt des selbst gepflegten
            # self.current_parameters-Caches: der wird nur bei manueller
            # Slider-Interaktion aktualisiert und startet leer (siehe
            # SettlementTab.generate() für dasselbe Problem).
            self.current_parameters = self.get_current_parameters()
            request_id = self.generation_orchestrator.request_generation(
                generator_type="biome",
                parameters=self.current_parameters.copy(),
                target_lod=None,
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
        Aufgabe: Parameter, Biome-Preview
        Statistics stecken im Statistics-Tab (siehe create_statistics_controls()),
        Display-Mode/Overlays in der Viewport-Toolbar (siehe create_visualization_controls()) -
        beide nicht mehr im Parameter-Panel, wie bei TerrainTab.
        """
        # Parameter Panel
        self.parameter_panel = self.create_biome_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)

        # Biome Classification Widget
        self.biome_classification_widget = BiomeClassificationWidget()
        self.control_panel.layout().addWidget(self.biome_classification_widget)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Biome-Statistics statt sie im Parameter-Panel unterzubringen.
        """
        self.statistics_group = self._create_statistics_display()
        layout.addWidget(self.statistics_group)

    def _create_statistics_display(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Anzeige für Biome-eigene Statistiken
        Aufgabe: Zeigt Verteilung der Base-Biomes, Super-Biome-Anteil und Diversitaet
        """
        group = QGroupBox("Biome Statistics")
        layout = QVBoxLayout()

        self.biome_distribution_label = QLabel("Biome Distribution: -")
        layout.addWidget(self.biome_distribution_label)

        self.super_biome_label = QLabel("Super-Biome Coverage: -")
        layout.addWidget(self.super_biome_label)

        self.biome_diversity_label = QLabel("Biome Diversity: -")
        layout.addWidget(self.biome_diversity_label)

        group.setLayout(layout)
        return group

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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
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
            suffix=param_config.get("suffix", ""),
            description=param_config.get("description", "")
        )

        slider.valueChanged.connect(self.on_parameter_changed)
        self.parameter_sliders["edge_softness"] = slider
        softness_layout.addWidget(slider)

        softness_group.setLayout(softness_layout)
        layout.addWidget(softness_group)

        panel.setLayout(layout)

        return panel

    def create_visualization_controls(self):
        """
        Überschreibt BaseMapTab: Display-Mode-Radios, Overlay-Checkboxes und
        Legend-Button sitzen wie bei TerrainTab in der Viewport-Toolbar
        (Spalte 2), nicht mehr im Parameter-Panel.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        controls_layout.addLayout(self._create_biome_display_mode_controls())
        controls_layout.addWidget(self._create_vertical_separator())
        controls_layout.addLayout(self._create_biome_overlay_controls())
        controls_layout.addWidget(self._create_vertical_separator())

        self.show_legend_button = BaseButton("Show Biome Legend")
        self.show_legend_button.clicked.connect(self.show_biome_legend)
        controls_layout.addWidget(self.show_legend_button)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_biome_display_mode_controls(self) -> QHBoxLayout:
        """Erstellt Switcher zwischen biome_map, biome_map_super, super_biome_mask"""
        layout = QHBoxLayout()

        self.display_mode = QButtonGroup()

        self.base_biomes_radio = QRadioButton("Base Biomes")
        self.base_biomes_radio.setChecked(True)
        self.base_biomes_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.base_biomes_radio, 0)
        layout.addWidget(self.base_biomes_radio)

        self.super_biomes_radio = QRadioButton("Super Biomes")
        self.super_biomes_radio.setToolTip("Super Biomes (2x2 Supersampled)")
        self.super_biomes_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.super_biomes_radio, 1)
        layout.addWidget(self.super_biomes_radio)

        self.override_mask_radio = QRadioButton("Override Mask")
        self.override_mask_radio.setToolTip("Super-Biome Override Mask")
        self.override_mask_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.override_mask_radio, 2)
        layout.addWidget(self.override_mask_radio)

        return layout

    def _create_biome_overlay_controls(self) -> QHBoxLayout:
        """
        Erstellt Overlay-Checkboxes (Settlements/Rivers). Elevation-Contours
        gibt es hier bewusst NICHT mehr lokal - das ist jetzt die globale
        Shell-Checkbox (Spalte 2, siehe BaseMapTab.set_contour_overlay()),
        die seit der Contour-Generalisierung auch auf Biome-Layern wirkt.
        """
        layout = QHBoxLayout()

        self.settlements_overlay = QCheckBox("Settlements")
        self.settlements_overlay.toggled.connect(self.toggle_settlements_overlay)
        layout.addWidget(self.settlements_overlay)

        self.rivers_overlay = QCheckBox("Rivers")
        self.rivers_overlay.toggled.connect(self.toggle_rivers_overlay)
        layout.addWidget(self.rivers_overlay)

        return layout

    def _create_vertical_separator(self) -> QWidget:
        separator = QWidget()
        separator.setFixedWidth(1)
        separator.setStyleSheet("background-color: #bdc3c7;")
        return separator

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking
        Aufgabe: Überwacht alle Required Dependencies für Biome-System
        """
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["biome"]

        self.dependency_status = StatusIndicator("Biome Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

        self.data_lod_manager.data_updated.connect(self.on_data_updated)

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
        is_complete, missing = self.data_lod_manager.check_dependencies("biome", self.required_dependencies)

        if is_complete:
            self.dependency_status.set_success("All dependencies available")
        else:
            self.dependency_status.set_warning(f"Missing: {', '.join(missing)}")

        return is_complete

    def collect_input_data(self) -> dict:
        """
        Funktionsweise: Sammelt alle Required Input-Daten von DataManager
        Return: dict mit allen benötigten Arrays für Biome-Generation
        """
        inputs = {}

        # Terrain Inputs
        inputs["heightmap"] = self.data_lod_manager.get_terrain_data("heightmap")
        inputs["slopemap"] = self.data_lod_manager.get_terrain_data("slopemap")

        # Weather Inputs
        inputs["temp_map"] = self.data_lod_manager.get_weather_data("temp_map")

        # Water Inputs
        inputs["soil_moist_map"] = self.data_lod_manager.get_water_data("soil_moist_map")
        inputs["water_biomes_map"] = self.data_lod_manager.get_water_data("water_biomes_map")

        # Validation
        for key, data in inputs.items():
            if data is None:
                raise ValueError(f"Required input '{key}' not available")

        return inputs

    def update_biome_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt biome_map, biome_map_super oder super_biome_mask mit Overlays

        Nutzt wie die anderen Tabs get_current_display()/_push_data_to_current_display()
        statt eines nie zugewiesenen self.map_display (siehe MEMORY: Biome/Settlement
        hatten dadurch nie funktionierendes Rendering, weder 2D noch 3D).
        """
        current_mode = self.display_mode.checkedId()

        if current_mode == 0:  # Base Biomes
            biome_map = self.data_lod_manager.get_biome_data("biome_map")
            if biome_map is not None:
                self._push_data_to_current_display(biome_map, "biome_map")

        elif current_mode == 1:  # Super Biomes (2x2 Supersampled)
            biome_map_super = self.data_lod_manager.get_biome_data("biome_map_super")
            if biome_map_super is not None:
                self._push_data_to_current_display(biome_map_super, "biome_map")

        elif current_mode == 2:  # Super-Biome Override Mask
            super_biome_mask = self.data_lod_manager.get_biome_data("super_biome_mask")
            if super_biome_mask is not None:
                self._push_data_to_current_display(super_biome_mask, "biome_map")

        # Overlays anwenden
        self.apply_overlays()

    def apply_overlays(self):
        """
        Funktionsweise: Wendet alle aktivierten Overlays auf Display an
        Aufgabe: Settlements, Rivers basierend auf Checkboxes (Elevation
        Contours läuft über die globale Shell-Checkbox, siehe set_contour_overlay())
        """
        current_display = self.get_current_display()
        if not current_display or self.current_view != "2d":
            return
        display = current_display.display

        # Settlements Overlay
        if self.settlements_overlay.isChecked() and hasattr(display, 'overlay_settlements'):
            settlement_list = self.data_lod_manager.get_settlement_data("settlement_list")
            landmark_list = self.data_lod_manager.get_settlement_data("landmark_list")
            if settlement_list is not None:
                display.overlay_settlements(settlement_list, landmark_list)

        # Rivers Overlay
        if self.rivers_overlay.isChecked() and hasattr(display, 'overlay_river_network'):
            flow_map = self.data_lod_manager.get_water_data("flow_map")
            if flow_map is not None:
                display.overlay_river_network(flow_map)

    @pyqtSlot()
    def update_display_mode(self):
        """
        Slot für Visualization-Mode Änderungen.
        update_biome_display() ruft self.map_display auf, das nie zugewiesen
        wird (die realen Render-Methoden display_base_biomes()/overlay_*()
        existieren auch nicht auf MapDisplay2D) - Biome-2D/3D-Rendering ist
        noch nicht implementiert. Bis dahin hier abfangen statt hart zu crashen.
        """
        try:
            self.update_biome_display()
        except AttributeError as e:
            self.logger.debug(f"Biome display rendering not yet implemented: {e}")

    @pyqtSlot(bool)
    def toggle_settlements_overlay(self, enabled: bool):
        """Toggle für Settlements Overlay"""
        self.update_biome_display()

    @pyqtSlot(bool)
    def toggle_rivers_overlay(self, enabled: bool):
        """Toggle für Rivers Overlay"""
        self.update_biome_display()

    @pyqtSlot()
    def show_biome_legend(self):
        """
        Funktionsweise: Zeigt Biome-Legend Dialog mit allen 26 Biome-Typen
        Aufgabe: Übersichtliche Darstellung aller Base- und Super-Biomes
        """
        legend_dialog = BiomeLegendDialog(self)
        legend_dialog.exec_()

    def get_all_parameters(self, generator_type: str) -> dict:
        """
        Funktionsweise: Holt Parameter von anderen Tabs für Export
        Parameter: generator_type (str)
        Return: dict mit Parametern oder leerer dict
        """
        # Hier würde normalerweise auf andere Tabs zugegriffen werden
        # Für jetzt leerer dict als Fallback
        return {}

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