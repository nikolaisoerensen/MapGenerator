"""
Path: gui/tabs/settlement_tab.py

Funktionsweise: Settlement-Editor mit terrain_tab-ähnlicher UI-Struktur und vollständiger Core-Integration
- Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
- UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Navigation (fixiert unten)
- GenerationOrchestrator Integration mit StandardOrchestratorHandler
- Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
- Live Settlement-Preview und 3D-Visualization mit Terrain-Integration
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import SETTLEMENT, get_parameter_config, validate_parameter_set, VALIDATION_RULES
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton, MultiDependencyStatusWidget
from gui.managers.orchestrator_manager import StandardOrchestratorHandler, OrchestratorRequestBuilder

def get_settlement_error_decorators():
    """
    Funktionsweise: Lazy Loading von Settlement Tab Error Decorators
    Aufgabe: Lädt Core-Generation, Dependency und UI-Navigation Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import core_generation_handler, dependency_handler, ui_navigation_handler
        return core_generation_handler, dependency_handler, ui_navigation_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

core_generation_handler, dependency_handler, ui_navigation_handler = get_settlement_error_decorators()

class SettlementTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung mit BaseGenerator-Integration
    Aufgabe: Koordiniert alle Settlement-Core-Module, 3D-Visualization und GenerationOrchestrator-Integration
    Input: heightmap, slopemap, water_map für Terrain-Suitability
    Output: SettlementData mit allen Settlement-Komponenten
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)
        self.logger = logging.getLogger(__name__)

        # GenerationOrchestrator Integration
        self.orchestrator_handler = None
        self.setup_orchestrator_integration()

        # Parameter und State
        self.current_parameters = {}
        self.settlement_generation_complete = False

        # Setup UI
        self.setup_settlement_ui()
        self.setup_dependency_checking()

        # Initial Load
        self.load_default_parameters()
        self.check_input_dependencies()

    def setup_orchestrator_integration(self):
        """
        Funktionsweise: Setup für GenerationOrchestrator Integration mit StandardOrchestratorHandler
        Aufgabe: Eliminiert Code-Duplikation durch wiederverwendbare Orchestrator-Integration
        """
        if self.generation_orchestrator:
            # UI-Update-Methods für Settlement-spezifische Updates
            ui_update_methods = {
                'update_system_status': self.update_system_status_display,
                'update_settlement_statistics': self.update_settlement_statistics,
                'update_generation_progress': self.update_generation_progress
            }

            self.orchestrator_handler = StandardOrchestratorHandler(
                self.generation_orchestrator,
                ui_update_methods,
                self.logger
            )

            # Signal-Connections für Settlement-Tab
            self.orchestrator_handler.generation_completed.connect(self.on_settlement_generation_completed)
            self.orchestrator_handler.lod_progression_completed.connect(self.on_lod_progression_completed)

    def generate(self):
        """
        Funktionsweise: Hauptmethode für Settlement-Generation mit Orchestrator Integration
        Aufgabe: Startet Settlement-Generation über GenerationOrchestrator mit Target-LOD
        """
        if not self.generation_orchestrator:
            self.logger.error("No GenerationOrchestrator available")
            self.handle_generation_error(Exception("GenerationOrchestrator not available"))
            return

        if self.generation_in_progress:
            self.logger.info("Generation already in progress, ignoring request")
            return

        if not self.check_input_dependencies():
            self.logger.warning("Cannot generate settlement system - missing dependencies")
            return

        try:
            self.logger.info(f"Starting settlement generation with target LOD: {self.target_lod}")

            self.start_generation_timing()
            self.generation_in_progress = True

            # OrchestratorRequestBuilder für typ-sichere Request-Erstellung
            request_builder = OrchestratorRequestBuilder()
            request = request_builder.build_settlement_request(
                parameters=self.current_parameters.copy(),
                target_lod=self.target_lod,
                source_tab="settlement",
                priority=10
            )

            request_id = self.generation_orchestrator.request_generation(request)

            if request_id:
                self.logger.info(f"Settlement generation requested: {request_id}")
                self.update_system_status_display("queued", "Settlement generation queued...")
            else:
                raise Exception("Failed to request generation from orchestrator")

        except Exception as e:
            self.generation_in_progress = False
            self.handle_generation_error(e)
            raise

    def setup_settlement_ui(self):
        """
        Funktionsweise: Erstellt komplette UI für Settlement-System mit terrain_tab-ähnlicher Struktur
        Aufgabe: Generate Button → System Status → Parameter → Statistics → Navigation (fixiert unten)
        """
        # Generate Button (oben)
        self.manual_generate_button = BaseButton("Generate Settlement System", "primary")
        self.manual_generate_button.clicked.connect(self.generate)
        self.control_panel.layout().addWidget(self.manual_generate_button)

        # System Status Widget
        self.system_status_widget = SettlementSystemStatusWidget()
        self.control_panel.layout().addWidget(self.system_status_widget)

        # Parameter Panel
        self.parameter_panel = self.create_settlement_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)

        # Settlement Statistics
        self.settlement_stats = SettlementStatisticsWidget()
        self.control_panel.layout().addWidget(self.settlement_stats)

        # Visualization Controls
        self.visualization_controls = self.create_settlement_visualization_controls()
        self.control_panel.layout().addWidget(self.visualization_controls)

        # Dependencies und Navigation (wird von base_tab hinzugefügt)
        self.setup_input_status()

    def create_settlement_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Settlement-Parametern
        Aufgabe: Alle 9 Parameter aus value_default.SETTLEMENT strukturiert organisiert
        Return: QGroupBox mit Parameter-Slidern
        """
        panel = QGroupBox("Settlement Parameters")
        layout = QVBoxLayout()

        self.parameter_sliders = {}

        # Location Count Parameters
        locations_group = QGroupBox("Location Counts")
        locations_layout = QVBoxLayout()

        location_params = ["settlements", "landmarks", "roadsites", "plotnodes"]
        for param_name in location_params:
            param_config = get_parameter_config("settlement", param_name)

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
            locations_layout.addWidget(slider)

        locations_group.setLayout(locations_layout)
        layout.addWidget(locations_group)

        # Influence and Terrain Parameters
        influence_group = QGroupBox("Civilization Influence")
        influence_layout = QVBoxLayout()

        influence_params = ["civ_influence_decay", "terrain_factor_villages"]
        for param_name in influence_params:
            param_config = get_parameter_config("settlement", param_name)

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
            influence_layout.addWidget(slider)

        influence_group.setLayout(influence_layout)
        layout.addWidget(influence_group)

        # Road Network Parameters
        road_group = QGroupBox("Road Network")
        road_layout = QVBoxLayout()

        road_params = ["road_slope_to_distance_ratio"]
        for param_name in road_params:
            param_config = get_parameter_config("settlement", param_name)

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
            road_layout.addWidget(slider)

        road_group.setLayout(road_layout)
        layout.addWidget(road_group)

        # Wilderness and Plot Parameters
        misc_group = QGroupBox("Wilderness & Plots")
        misc_layout = QVBoxLayout()

        misc_params = ["landmark_wilderness", "plotsize"]
        for param_name in misc_params:
            param_config = get_parameter_config("settlement", param_name)

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
            misc_layout.addWidget(slider)

        misc_group.setLayout(misc_layout)
        layout.addWidget(misc_group)

        panel.setLayout(layout)
        return panel

    def create_settlement_visualization_controls(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Controls für Settlement-Visualization
        Aufgabe: Switcher zwischen verschiedenen Settlement-Darstellungen
        Return: QGroupBox mit Visualization-Controls
        """
        panel = QGroupBox("Settlement Visualization")
        layout = QVBoxLayout()

        # Display Mode Selection
        self.display_mode = QButtonGroup()

        self.suitability_radio = QRadioButton("Terrain Suitability")
        self.suitability_radio.setChecked(True)
        self.suitability_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.suitability_radio, 0)
        layout.addWidget(self.suitability_radio)

        self.settlements_radio = QRadioButton("Settlements & Landmarks")
        self.settlements_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.settlements_radio, 1)
        layout.addWidget(self.settlements_radio)

        self.road_network_radio = QRadioButton("Road Network")
        self.road_network_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.road_network_radio, 2)
        layout.addWidget(self.road_network_radio)

        self.civ_map_radio = QRadioButton("Civilization Map")
        self.civ_map_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.civ_map_radio, 3)
        layout.addWidget(self.civ_map_radio)

        self.plot_map_radio = QRadioButton("Plot Boundaries")
        self.plot_map_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.plot_map_radio, 4)
        layout.addWidget(self.plot_map_radio)

        # Settlement Type Filters
        filter_group = QGroupBox("Settlement Filters")
        filter_layout = QVBoxLayout()

        self.show_settlements_cb = QCheckBox("Show Settlements")
        self.show_settlements_cb.setChecked(True)
        self.show_settlements_cb.toggled.connect(self.update_display_mode)
        filter_layout.addWidget(self.show_settlements_cb)

        self.show_landmarks_cb = QCheckBox("Show Landmarks")
        self.show_landmarks_cb.setChecked(True)
        self.show_landmarks_cb.toggled.connect(self.update_display_mode)
        filter_layout.addWidget(self.show_landmarks_cb)

        self.show_roadsites_cb = QCheckBox("Show Roadsites")
        self.show_roadsites_cb.setChecked(True)
        self.show_roadsites_cb.toggled.connect(self.update_display_mode)
        filter_layout.addWidget(self.show_roadsites_cb)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # 3D Controls
        d3_group = QGroupBox("3D Visualization")
        d3_layout = QVBoxLayout()

        self.terrain_3d_cb = QCheckBox("Show 3D Terrain")
        self.terrain_3d_cb.toggled.connect(self.toggle_3d_terrain)
        d3_layout.addWidget(self.terrain_3d_cb)

        self.settlement_markers_3d_cb = QCheckBox("3D Settlement Markers")
        self.settlement_markers_3d_cb.toggled.connect(self.toggle_3d_markers)
        d3_layout.addWidget(self.settlement_markers_3d_cb)

        d3_group.setLayout(d3_layout)
        layout.addWidget(d3_group)

        panel.setLayout(layout)
        return panel

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking
        Aufgabe: Überwacht Required Dependencies für Settlement-System
        """
        # Required Dependencies für Settlement-System
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["settlement"]

        # Dependency Status Widget
        self.dependency_status = MultiDependencyStatusWidget(
            self.required_dependencies, "Settlement Dependencies"
        )
        self.control_panel.layout().addWidget(self.dependency_status)

        # Data Manager Signals
        self.data_manager.data_updated.connect(self.on_data_updated)

    def setup_input_status(self):
        """
        Funktionsweise: Setup für Input-Status-Anzeige spezifisch für Settlement
        Aufgabe: Erweitert die Dependency-Checking um Settlement-spezifische Status-Infos
        """
        # Input Status Panel für zusätzliche Settlement-spezifische Informationen
        input_status_group = QGroupBox("Input Status")
        input_layout = QVBoxLayout()

        # Terrain Suitability Status
        self.terrain_status = StatusIndicator("Terrain Quality")
        self.terrain_status.set_unknown()
        input_layout.addWidget(self.terrain_status)

        # Water Proximity Status (optional)
        self.water_status = StatusIndicator("Water Proximity")
        self.water_status.set_unknown()
        input_layout.addWidget(self.water_status)

        # Settlement Placement Viability
        self.placement_status = StatusIndicator("Placement Viability")
        self.placement_status.set_unknown()
        input_layout.addWidget(self.placement_status)

        input_status_group.setLayout(input_layout)
        self.control_panel.layout().addWidget(input_status_group)

        # Update Status basierend auf verfügbaren Daten
        self.update_input_status()

    def update_system_status_display(self, status: str, message: str = ""):
        """
        Funktionsweise: Aktualisiert System Status Display für alle Settlement-Schritte
        Aufgabe: Zeigt Progress für alle 7 calculate-Phasen mit LOD-Info und Validity-State
        Parameter: status (str), message (str) - Aktueller Status und Detail-Message
        """
        if hasattr(self, 'system_status_widget'):
            self.system_status_widget.update_status(status, message)

            # Settlement-spezifische Status-Updates
            if status == "generating":
                self.system_status_widget.update_step_progress(message)
            elif status == "completed":
                # Alle Schritte als completed markieren
                steps = ["terrain_suitability", "settlements", "road_network", "roadsites",
                        "civilization_mapping", "landmarks", "plots"]
                for step in steps:
                    self.system_status_widget.mark_step_completed(step)

    def update_settlement_statistics(self):
        """
        Funktionsweise: Aktualisiert Settlement-Statistiken nach Generation
        Aufgabe: Zeigt Generation-Results in Statistics-Widget
        """
        settlement_data = self.data_manager.get_settlement_data("settlement_data_complete")
        if settlement_data and hasattr(self, 'settlement_stats'):
            self.settlement_stats.update_generation_statistics(
                settlement_data.settlement_list,
                settlement_data.landmark_list,
                settlement_data.roadsite_list,
                settlement_data.plot_map,
                settlement_data.civ_map
            )

    def update_generation_progress(self, progress: int, message: str):
        """
        Funktionsweise: Aktualisiert Progress Bar für Settlement-Generation
        Aufgabe: Zeigt detaillierten Progress für alle calculate-Schritte
        Parameter: progress (int), message (str) - Progress-Prozent und Detail-Message
        """
        if hasattr(self, 'system_status_widget'):
            self.system_status_widget.update_progress(progress, message)

    @pyqtSlot(str, dict)
    def on_settlement_generation_completed(self, result_id: str, result_data: dict):
        """
        Funktionsweise: Slot für Settlement-Generation Completion
        Aufgabe: Verarbeitet Settlement-Results und aktualisiert UI
        Parameter: result_id (str), result_data (dict) - Result-ID und Settlement-Daten
        """
        try:
            self.generation_in_progress = False
            self.settlement_generation_complete = True

            self.logger.info(f"Settlement generation completed: {result_id}")

            # Settlement-Results verarbeiten
            settlement_data = result_data.get("settlement_data")
            if settlement_data:
                # Statistics aktualisieren
                self.update_settlement_statistics()

                # Display aktualisieren
                self.update_settlement_display()

                # System Status als completed setzen
                self.update_system_status_display("completed", "Settlement generation completed successfully")

            self.end_generation_timing()

        except Exception as e:
            self.logger.error(f"Error processing settlement generation completion: {e}")
            self.handle_generation_error(e)

    @pyqtSlot(str, str)
    def on_lod_progression_completed(self, result_id: str, lod_level: str):
        """
        Funktionsweise: Slot für LOD-Progression Updates
        Aufgabe: Aktualisiert Display nach jedem LOD-Level
        Parameter: result_id (str), lod_level (str) - Result-ID und erreichtes LOD-Level
        """
        try:
            self.logger.info(f"Settlement LOD progression: {lod_level}")

            # System Status mit LOD-Info aktualisieren
            self.update_system_status_display("generating", f"Completed LOD {lod_level}")

            # Display mit bestem verfügbarem LOD aktualisieren
            self.update_settlement_display()

        except Exception as e:
            self.logger.error(f"Error processing LOD progression: {e}")

    def load_default_parameters(self):
        """Lädt Default-Parameter"""
        for param_name, slider in self.parameter_sliders.items():
            param_config = get_parameter_config("settlement", param_name)
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

        # Settlement Statistics Preview aktualisieren
        self.settlement_stats.update_parameter_preview(self.current_parameters)

        # Auto-Simulation triggern
        if self.auto_simulation_enabled:
            self.auto_simulation_timer.start(1000)

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """Slot für Data-Updates von anderen Generatoren"""
        if data_key in self.required_dependencies:
            self.check_input_dependencies()
            self.update_input_status()

    def check_input_dependencies(self):
        """
        Funktionsweise: Prüft alle Required Dependencies für Settlement-System
        Aufgabe: Aktiviert/Deaktiviert Generation basierend auf verfügbaren Inputs
        """
        is_complete, missing = self.data_manager.check_dependencies("settlement", self.required_dependencies)

        self.dependency_status.update_dependency_status(is_complete, missing)
        self.manual_generate_button.setEnabled(is_complete)

        return is_complete

    def update_input_status(self):
        """
        Funktionsweise: Aktualisiert Settlement-spezifische Input-Status
        Aufgabe: Zeigt Qualität der Terrain-Daten für Settlement-Placement
        """
        try:
            # Terrain Quality Check
            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")

            if heightmap is not None and slopemap is not None:
                # Analysiere Terrain-Qualität für Settlements
                flat_areas = np.sum(slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2 < 0.1) / heightmap.size
                if flat_areas > 0.3:  # > 30% flache Bereiche
                    self.terrain_status.set_success(f"Good ({flat_areas:.1%} suitable)")
                elif flat_areas > 0.1:  # > 10% flache Bereiche
                    self.terrain_status.set_warning(f"Limited ({flat_areas:.1%} suitable)")
                else:
                    self.terrain_status.set_error(f"Poor ({flat_areas:.1%} suitable)")
            else:
                self.terrain_status.set_error("Missing terrain data")

            # Water Proximity Check
            water_map = self.data_manager.get_water_data("water_map")
            if water_map is not None:
                water_coverage = np.sum(water_map > 0.01) / water_map.size
                if water_coverage > 0.05:  # > 5% Wasser
                    self.water_status.set_success(f"Available ({water_coverage:.1%})")
                else:
                    self.water_status.set_warning(f"Limited ({water_coverage:.1%})")
            else:
                self.water_status.set_warning("No water data - using defaults")

            # Placement Viability (kombiniert)
            if heightmap is not None and slopemap is not None:
                if flat_areas > 0.2:
                    self.placement_status.set_success("Excellent placement conditions")
                elif flat_areas > 0.1:
                    self.placement_status.set_warning("Moderate placement conditions")
                else:
                    self.placement_status.set_error("Difficult placement conditions")
            else:
                self.placement_status.set_error("Cannot assess - missing data")

        except Exception as e:
            self.logger.warning(f"Error updating input status: {e}")
            self.terrain_status.set_error("Status check failed")
            self.water_status.set_error("Status check failed")
            self.placement_status.set_error("Status check failed")

    def update_settlement_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt verschiedene Settlement-Darstellungen mit Filtern
        """
        current_mode = self.display_mode.checkedId()

        if current_mode == 0:  # Terrain Suitability
            suitability_map = self.data_manager.get_settlement_data("combined_suitability_map")
            if suitability_map is not None:
                self.map_display.display_terrain_suitability(suitability_map)

        elif current_mode == 1:  # Settlements & Landmarks
            settlements = self.data_manager.get_settlement_data("settlement_list")
            landmarks = self.data_manager.get_settlement_data("landmark_list")
            roadsites = self.data_manager.get_settlement_data("roadsite_list")

            if settlements is not None:
                # Filter basierend auf Checkboxes
                display_settlements = settlements if self.show_settlements_cb.isChecked() else []
                display_landmarks = landmarks if self.show_landmarks_cb.isChecked() else []
                display_roadsites = roadsites if self.show_roadsites_cb.isChecked() else []

                self.map_display.display_settlements(
                    display_settlements, display_landmarks, display_roadsites
                )

        elif current_mode == 2:  # Road Network
            roads = self.data_manager.get_settlement_data("roads")
            if roads is not None:
                self.map_display.display_road_network(roads)

        elif current_mode == 3:  # Civilization Map
            civ_map = self.data_manager.get_settlement_data("civ_map")
            if civ_map is not None:
                self.map_display.display_civilization_map(civ_map)

        elif current_mode == 4:  # Plot Boundaries
            plot_map = self.data_manager.get_settlement_data("plot_map")
            if plot_map is not None:
                self.map_display.display_plot_boundaries(plot_map)

        # 3D Overlays
        self.apply_3d_overlays()

    def apply_3d_overlays(self):
        """
        Funktionsweise: Wendet 3D-Overlays basierend auf Checkboxes an
        Aufgabe: 3D Terrain und Settlement-Markers
        """
        # 3D Terrain Overlay
        if self.terrain_3d_cb.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.overlay_3d_terrain(heightmap)

        # 3D Settlement Markers
        if self.settlement_markers_3d_cb.isChecked():
            settlements = self.data_manager.get_settlement_data("settlement_list")
            landmarks = self.data_manager.get_settlement_data("landmark_list")
            if settlements is not None:
                self.map_display.overlay_3d_settlement_markers(settlements, landmarks)

    @pyqtSlot()
    def update_display_mode(self):
        """Slot für Visualization-Mode Änderungen"""
        self.update_settlement_display()

    @pyqtSlot(bool)
    def toggle_3d_terrain(self, enabled: bool):
        """Toggle für 3D Terrain Overlay"""
        self.update_settlement_display()

    @pyqtSlot(bool)
    def toggle_3d_markers(self, enabled: bool):
        """Toggle für 3D Settlement Markers"""
        self.update_settlement_display()


class SettlementSystemStatusWidget(QGroupBox):
    """
    Funktionsweise: System Status Widget für alle Settlement-Berechnungsschritte
    Aufgabe: Zeigt LOD-Level, Step-Status und Generation-Progress für alle 7 calculate-Phasen
    """

    def __init__(self):
        super().__init__("Settlement System Status")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Settlement System Status"""
        layout = QVBoxLayout()

        # LOD Level Display
        lod_group = QGroupBox("LOD Level & Validity")
        lod_layout = QVBoxLayout()

        self.lod_level_label = QLabel("Current LOD: LOD64")
        self.validity_state_label = QLabel("Validity: Unknown")

        lod_layout.addWidget(self.lod_level_label)
        lod_layout.addWidget(self.validity_state_label)

        lod_group.setLayout(lod_layout)
        layout.addWidget(lod_group)

        # Step Status für alle 7 calculate-Phasen
        steps_group = QGroupBox("Generation Steps")
        steps_layout = QVBoxLayout()

        self.step_indicators = {}
        steps = [
            ("terrain_suitability", "Terrain Suitability"),
            ("settlements", "Settlement Placement"),
            ("road_network", "Road Network"),
            ("roadsites", "Roadsite Placement"),
            ("civilization_mapping", "Civilization Mapping"),
            ("landmarks", "Landmark Placement"),
            ("plots", "Plot Generation")
        ]

        for step_key, step_name in steps:
            indicator = StatusIndicator(step_name)
            indicator.set_unknown()
            self.step_indicators[step_key] = indicator
            steps_layout.addWidget(indicator)

        steps_group.setLayout(steps_layout)
        layout.addWidget(steps_group)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status Message
        self.status_message_label = QLabel("Ready for generation")
        layout.addWidget(self.status_message_label)

        self.setLayout(layout)

    def update_status(self, status: str, message: str = ""):
        """
        Funktionsweise: Aktualisiert System Status
        Parameter: status (str), message (str) - Status und Message
        """
        self.status_message_label.setText(message or f"Status: {status}")

        if status == "generating":
            self.progress_bar.setVisible(True)
        elif status in ["completed", "failed"]:
            self.progress_bar.setVisible(False)

    def update_step_progress(self, step_message: str):
        """
        Funktionsweise: Aktualisiert Schritt-Progress basierend auf Message
        Parameter: step_message (str) - Detail-Message mit Schritt-Info
        """
        # Parse step aus message und aktualisiere entsprechenden Indicator
        if "terrain suitability" in step_message.lower():
            self.step_indicators["terrain_suitability"].set_generating()
        elif "settlement placement" in step_message.lower():
            self.step_indicators["settlements"].set_generating()
        elif "road network" in step_message.lower():
            self.step_indicators["road_network"].set_generating()
        elif "roadsite" in step_message.lower():
            self.step_indicators["roadsites"].set_generating()
        elif "civilization" in step_message.lower():
            self.step_indicators["civilization_mapping"].set_generating()
        elif "landmark" in step_message.lower():
            self.step_indicators["landmarks"].set_generating()
        elif "plot" in step_message.lower():
            self.step_indicators["plots"].set_generating()

    def mark_step_completed(self, step_key: str):
        """
        Funktionsweise: Markiert einzelnen Schritt als completed
        Parameter: step_key (str) - Schritt-Schlüssel
        """
        if step_key in self.step_indicators:
            self.step_indicators[step_key].set_success("Completed")

    def update_progress(self, progress: int, message: str):
        """
        Funktionsweise: Aktualisiert Progress Bar
        Parameter: progress (int), message (str) - Progress-Prozent und Message
        """
        self.progress_bar.setValue(progress)
        self.status_message_label.setText(message)


class SettlementStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Widget für Settlement-Statistiken und Parameter-Preview
    Aufgabe: Zeigt Settlement-Counts, Suitability-Stats, Generation-Results
    """

    def __init__(self):
        super().__init__("Settlement Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Settlement-Statistiken"""
        layout = QVBoxLayout()

        # Parameter Preview
        preview_group = QGroupBox("Parameter Preview")
        preview_layout = QVBoxLayout()

        self.settlement_count_label = QLabel("Settlements: 8")
        self.landmark_count_label = QLabel("Landmarks: 5")
        self.roadsite_count_label = QLabel("Roadsites: 15")
        self.plotnode_count_label = QLabel("Plot Nodes: 200")

        preview_layout.addWidget(self.settlement_count_label)
        preview_layout.addWidget(self.landmark_count_label)
        preview_layout.addWidget(self.roadsite_count_label)
        preview_layout.addWidget(self.plotnode_count_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Generation Results
        results_group = QGroupBox("Generation Results")
        results_layout = QVBoxLayout()

        self.actual_settlements_label = QLabel("Generated Settlements: -")
        self.road_length_label = QLabel("Total Road Length: -")
        self.plot_count_label = QLabel("Created Plots: -")
        self.avg_suitability_label = QLabel("Avg Suitability: -")

        results_layout.addWidget(self.actual_settlements_label)
        results_layout.addWidget(self.road_length_label)
        results_layout.addWidget(self.plot_count_label)
        results_layout.addWidget(self.avg_suitability_label)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        self.setLayout(layout)

    def update_parameter_preview(self, parameters: dict):
        """Aktualisiert Parameter-Preview"""
        self.settlement_count_label.setText(f"Settlements: {int(parameters.get('settlements', 8))}")
        self.landmark_count_label.setText(f"Landmarks: {int(parameters.get('landmarks', 5))}")
        self.roadsite_count_label.setText(f"Roadsites: {int(parameters.get('roadsites', 15))}")
        self.plotnode_count_label.setText(f"Plot Nodes: {int(parameters.get('plotnodes', 200))}")

    def update_generation_statistics(self, settlements: list, landmarks: list, roadsites: list,
                                     plot_map: np.ndarray, civ_map: np.ndarray):
        """
        Funktionsweise: Aktualisiert Statistiken nach Generation
        Parameter: settlements, landmarks, roadsites (lists), plot_map, civ_map (arrays)
        """
        # Actual Generated Counts
        self.actual_settlements_label.setText(f"Generated Settlements: {len(settlements)}")

        # Road Length (würde normalerweise aus road_network berechnet)
        self.road_length_label.setText("Total Road Length: Calculated from network")

        # Plot Count
        unique_plots = len(np.unique(plot_map)) - 1  # -1 für background
        self.plot_count_label.setText(f"Created Plots: {unique_plots}")

        # Average Civilization Value
        avg_civ = np.mean(civ_map[civ_map > 0])  # Nur non-zero Bereiche
        self.avg_suitability_label.setText(f"Avg Civilization: {avg_civ:.2f}")


class CivilizationInfluenceWidget(QGroupBox):
    """
    Funktionsweise: Widget für Civilization-Influence Monitoring
    Aufgabe: Zeigt Influence-Parameter, Decay-Preview, Civilization-Statistics
    """

    def __init__(self):
        super().__init__("Civilization Influence")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Civilization-Influence Display"""
        layout = QVBoxLayout()

        # Influence Parameters Preview
        params_group = QGroupBox("Influence Parameters")
        params_layout = QVBoxLayout()

        self.decay_factor_label = QLabel("Decay Factor: 0.8")
        self.terrain_factor_label = QLabel("Terrain Factor: 1.0")

        params_layout.addWidget(self.decay_factor_label)
        params_layout.addWidget(self.terrain_factor_label)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Influence Statistics
        stats_group = QGroupBox("Influence Statistics")
        stats_layout = QVBoxLayout()

        self.civilized_area_label = QLabel("Civilized Area: -")
        self.wilderness_area_label = QLabel("Wilderness Area: -")
        self.max_influence_label = QLabel("Max Influence: -")

        stats_layout.addWidget(self.civilized_area_label)
        stats_layout.addWidget(self.wilderness_area_label)
        stats_layout.addWidget(self.max_influence_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        self.setLayout(layout)

    def update_influence_preview(self, parameters: dict):
        """Aktualisiert Influence-Parameter Preview"""
        decay = parameters.get("civ_influence_decay", 0.8)
        terrain = parameters.get("terrain_factor_villages", 1.0)

        self.decay_factor_label.setText(f"Decay Factor: {decay:.1f}")
        self.terrain_factor_label.setText(f"Terrain Factor: {terrain:.1f}")

    def update_influence_statistics(self, civ_map: np.ndarray):
        """
        Funktionsweise: Aktualisiert Influence-Statistiken nach Generation
        Parameter: civ_map (numpy array mit Civilization-Werten)
        """
        # Civilized vs Wilderness Area
        total_pixels = civ_map.shape[0] * civ_map.shape[1]
        civilized_pixels = np.sum(civ_map > 0.2)  # Threshold für "civilized"
        wilderness_pixels = total_pixels - civilized_pixels

        civilized_pct = (civilized_pixels / total_pixels) * 100
        wilderness_pct = (wilderness_pixels / total_pixels) * 100

        self.civilized_area_label.setText(f"Civilized Area: {civilized_pct:.1f}%")
        self.wilderness_area_label.setText(f"Wilderness Area: {wilderness_pct:.1f}%")

        # Max Influence
        max_influence = np.max(civ_map)
        self.max_influence_label.setText(f"Max Influence: {max_influence:.2f}")