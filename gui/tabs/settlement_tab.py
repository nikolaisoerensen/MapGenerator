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
from gui.widgets.widgets import ParameterSlider, StatusIndicator

def get_settlement_error_decorators():
    """
    Funktionsweise: Lazy Loading von Settlement Tab Error Decorators
    Aufgabe: Lädt Core-Generation, Dependency und UI-Navigation Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.utils.error_handler import core_generation_handler, dependency_handler, ui_navigation_handler
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

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):
        self.generator_type = "settlement"

        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )
        self.logger = logging.getLogger(__name__)

        # GenerationOrchestrator Integration
        self.setup_orchestrator_integration()

        # Parameter und State
        self.current_parameters = {}
        self.settlement_generation_complete = False
        self.generation_in_progress = False

        # Setup UI
        self.setup_settlement_ui()
        self.setup_dependency_checking()

        # Initial Load
        self.load_default_parameters()
        self.check_input_dependencies()

    def setup_orchestrator_integration(self):
        """
        Funktionsweise: Verbindet Settlement-spezifische Slots direkt mit dem GenerationOrchestrator
        Aufgabe: Signal-Anbindung für Generation-Completion und LOD-Progression
        """
        if self.generation_orchestrator:
            self.generation_orchestrator.generation_completed.connect(self.on_settlement_generation_completed)
            self.generation_orchestrator.lod_progression_completed.connect(self.on_lod_progression_completed)

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
            # target_lod nicht selbst gesetzt - wie bei den anderen Tabs
            # bestimmt der Orchestrator es aus map_size/aktuellem Terrain-LOD
            # (request_generation() mit target_lod=None).
            self.logger.info("Starting settlement generation")

            self.start_generation_timing()
            self.generation_in_progress = True

            # Frisch von den Slidern lesen statt des selbst gepflegten
            # self.current_parameters-Caches: der wird nur bei manueller
            # Slider-Interaktion aktualisiert und startet leer, wodurch ohne
            # UI-Interaktion Pflichtparameter wie "settlements" fehlten
            # (core/settlement_generator.py griff direkt über
            # parameters['settlements'] zu, ohne Fallback -> KeyError).
            self.current_parameters = self.get_current_parameters()
            request_id = self.generation_orchestrator.request_generation(
                generator_type="settlement",
                parameters=self.current_parameters.copy(),
                target_lod=None,
                source_tab="settlement",
                priority=10
            )

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
        Aufgabe: System Status → Parameter → Navigation (fixiert unten)
        Kein eigener Generate-Button mehr (globaler [GENERIEREN]-Button im
        Shell-Footer übernimmt das). Statistics stecken im Statistics-Tab,
        Display-Mode/Filter/3D-Controls in der Viewport-Toolbar (siehe
        create_statistics_controls()/create_visualization_controls()) -
        beide nicht mehr im Parameter-Panel, wie bei TerrainTab.
        Kein eigenes LOD/Generation-Steps-Status-Widget mehr zwischen der
        Pipeline-Status-Spalte und den Parameter-Slidern (Ticket #6 in
        docs/backlog.md) - Vorbild GeologyTab hat dafuer ebenfalls kein
        eigenes Widget, nur die Parameter-Gruppen direkt gefolgt von
        dependency_status.
        """
        # Parameter Panel
        self.parameter_panel = self.create_settlement_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)

        # Dependencies und Navigation (wird von base_tab hinzugefügt)
        self.setup_input_status()

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Settlement-Statistics statt sie im Parameter-Panel unterzubringen.
        """
        self.settlement_stats = SettlementStatisticsWidget()
        layout.addWidget(self.settlement_stats)

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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
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
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            misc_layout.addWidget(slider)

        misc_group.setLayout(misc_layout)
        layout.addWidget(misc_group)

        panel.setLayout(layout)
        return panel

    def create_visualization_controls(self):
        """
        Überschreibt BaseMapTab: Display-Mode-Radios und Filter-/3D-Checkboxes
        sitzen wie bei TerrainTab in der Viewport-Toolbar (Spalte 2), nicht
        mehr im Parameter-Panel.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        controls_layout.addLayout(self._create_settlement_display_mode_controls())
        controls_layout.addWidget(self._create_vertical_separator())
        controls_layout.addLayout(self._create_settlement_filter_controls())
        controls_layout.addWidget(self._create_vertical_separator())
        controls_layout.addLayout(self._create_settlement_3d_controls())

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_settlement_display_mode_controls(self) -> QHBoxLayout:
        """
        Erstellt Switcher zwischen den Settlement-Basis-Layern. Settlements/
        Landmarks/Roadsites/Roads/City Boundary sind bewusst KEINE eigenen
        Radio-Modi mehr, sondern Overlay-Checkboxen (siehe
        _create_settlement_filter_controls()) - Nutzer-Vorgabe: "ich will ja
        Plots und Straßen und alles gleichzeitig sehen können" statt zwischen
        sich gegenseitig ausschließenden Ansichten wechseln zu müssen.
        """
        layout = QHBoxLayout()

        self.display_mode = QButtonGroup()

        self.suitability_radio = QRadioButton("Terrain Suitability")
        self.suitability_radio.setChecked(True)
        self.suitability_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.suitability_radio, 0)
        layout.addWidget(self.suitability_radio)

        self.civ_map_radio = QRadioButton("Civilization Map")
        self.civ_map_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.civ_map_radio, 1)
        layout.addWidget(self.civ_map_radio)

        self.plot_map_radio = QRadioButton("Plot Boundaries")
        self.plot_map_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.plot_map_radio, 2)
        layout.addWidget(self.plot_map_radio)

        self.landscape_voronoi_radio = QRadioButton("Landscape Voronoi")
        self.landscape_voronoi_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.landscape_voronoi_radio, 3)
        layout.addWidget(self.landscape_voronoi_radio)

        self.city_blocks_radio = QRadioButton("City Blocks")
        self.city_blocks_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.city_blocks_radio, 4)
        layout.addWidget(self.city_blocks_radio)

        return layout

    def _create_settlement_filter_controls(self) -> QHBoxLayout:
        """
        Erstellt Overlay-Checkboxen - kombinierbar mit JEDEM Basis-Layer-Radio
        (siehe update_settlement_display()), nicht auf einen Modus beschränkt.
        """
        layout = QHBoxLayout()

        self.show_settlements_cb = QCheckBox("Settlements")
        self.show_settlements_cb.setChecked(True)
        self.show_settlements_cb.toggled.connect(self.update_display_mode)
        layout.addWidget(self.show_settlements_cb)

        self.show_landmarks_cb = QCheckBox("Landmarks")
        self.show_landmarks_cb.setChecked(True)
        self.show_landmarks_cb.toggled.connect(self.update_display_mode)
        layout.addWidget(self.show_landmarks_cb)

        self.show_roadsites_cb = QCheckBox("Roadsites")
        self.show_roadsites_cb.setChecked(True)
        self.show_roadsites_cb.toggled.connect(self.update_display_mode)
        layout.addWidget(self.show_roadsites_cb)

        self.show_roads_cb = QCheckBox("Roads")
        self.show_roads_cb.setChecked(True)
        self.show_roads_cb.toggled.connect(self.update_display_mode)
        layout.addWidget(self.show_roads_cb)

        self.show_city_boundary_cb = QCheckBox("City Boundary")
        self.show_city_boundary_cb.setChecked(True)
        self.show_city_boundary_cb.toggled.connect(self.update_display_mode)
        layout.addWidget(self.show_city_boundary_cb)

        return layout

    def _create_settlement_3d_controls(self) -> QHBoxLayout:
        """Erstellt 3D-Overlay-Checkboxes (aktuell ohne Effekt, siehe apply_3d_overlays())"""
        layout = QHBoxLayout()

        self.terrain_3d_cb = QCheckBox("3D Terrain")
        self.terrain_3d_cb.toggled.connect(self.toggle_3d_terrain)
        layout.addWidget(self.terrain_3d_cb)

        self.settlement_markers_3d_cb = QCheckBox("3D Settlement Markers")
        self.settlement_markers_3d_cb.toggled.connect(self.toggle_3d_markers)
        layout.addWidget(self.settlement_markers_3d_cb)

        return layout

    def _create_vertical_separator(self) -> QWidget:
        separator = QWidget()
        separator.setFixedWidth(1)
        separator.setStyleSheet("background-color: #bdc3c7;")
        return separator

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking
        Aufgabe: Überwacht Required Dependencies für Settlement-System
        """
        # Required Dependencies für Settlement-System
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["settlement"]

        # Dependency Status Widget
        self.dependency_status = StatusIndicator("Settlement Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

        # Data Manager Signals
        self.data_lod_manager.data_updated.connect(self.on_data_updated)

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
        No-Op-Hook: das detaillierte Multi-Step-Status-Widget wurde entfernt
        (Ticket #6 in docs/backlog.md, Vorbild GeologyTab - keine eigene
        Status-Anzeige zwischen Pipeline-Status-Spalte und Parameter-Slidern).
        Bleibt bestehen, da mehrere Call-Sites in dieser Datei ihn weiterhin
        aufrufen; die allgemeine Pipeline-Status-Spalte deckt den Fortschritt
        bereits ab.
        """
        pass

    def update_settlement_statistics(self):
        """
        Funktionsweise: Aktualisiert Settlement-Statistiken nach Generation
        Aufgabe: Zeigt Generation-Results in Statistics-Widget
        """
        settlement_data = self.data_lod_manager.get_settlement_data("settlement_data_complete")
        if settlement_data and hasattr(self, 'settlement_stats'):
            self.settlement_stats.update_generation_statistics(
                settlement_data.settlement_list,
                settlement_data.landmark_list,
                settlement_data.roadsite_list,
                settlement_data.plot_map,
                settlement_data.civ_map
            )

    def update_generation_progress(self, progress: int, message: str):
        """No-Op-Hook (siehe update_system_status_display() - Status-Widget entfernt)."""
        pass

    @pyqtSlot(str, dict)
    def on_settlement_generation_completed(self, result_id: str, result_data: dict):
        """
        Funktionsweise: Slot für Settlement-Generation Completion
        Aufgabe: Verarbeitet Settlement-Results und aktualisiert UI
        Parameter: result_id (str), result_data (dict) - Result-ID und Settlement-Daten
        """
        if result_data.get("generator_type") != "settlement":
            return

        try:
            self.generation_in_progress = False
            self.settlement_generation_complete = True

            self.logger.info(f"Settlement generation completed: {result_id}")

            # Settlement-Results verarbeiten. emit_final_completion_signal() liefert
            # die Ergebnisse unter "data" (siehe
            # GenerationOrchestrator.get_generator_data_from_data_lod_manager()),
            # nicht unter "settlement_data" - der alte Key existierte nie, wodurch
            # dieser Block trotz erfolgreicher Generation nie ausgeführt wurde.
            settlement_data = result_data.get("data")
            if result_data.get("success") and settlement_data:
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

    @pyqtSlot(str, int)
    def on_lod_progression_completed(self, result_id: str, lod_level: int):
        """
        Funktionsweise: Slot für LOD-Progression Updates
        Aufgabe: Aktualisiert Display nach jedem LOD-Level
        Parameter: result_id (str), lod_level (int) - Result-ID und erreichtes LOD-Level
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
        is_complete, missing = self.data_lod_manager.check_dependencies("settlement", self.required_dependencies)

        if is_complete:
            self.dependency_status.set_success("All dependencies available")
        else:
            self.dependency_status.set_warning(f"Missing: {', '.join(missing)}")

        return is_complete

    def update_input_status(self):
        """
        Funktionsweise: Aktualisiert Settlement-spezifische Input-Status
        Aufgabe: Zeigt Qualität der Terrain-Daten für Settlement-Placement
        """
        try:
            # Terrain Quality Check (kombiniert - reflektiert das tatsächliche
            # Endgelände nach Erosion/Sedimentation, nicht die Rohausgabe)
            heightmap = self.data_lod_manager.get_terrain_data_combined("heightmap")
            slopemap = self.data_lod_manager.get_terrain_data("slopemap")

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
            water_map = self.data_lod_manager.get_water_data("water_map")
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
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Basis-Layer
        (Radio) und den unabhängig davon zuschaltbaren Overlay-Checkboxen
        (Settlements/Landmarks/Roadsites/Roads/City Boundary)
        Aufgabe: Basis-Layer und Overlays sind entkoppelt, damit z.B. Plots und
        Straßen und Settlements gleichzeitig sichtbar sein können statt
        zwischen exklusiven Modi wechseln zu müssen (Nutzer-Vorgabe)

        Nutzt wie die anderen Tabs get_current_display()/_push_data_to_current_display()
        statt eines nie zugewiesenen self.map_display.
        """
        current_mode = self.display_mode.checkedId()

        if current_mode == 0:  # Terrain Suitability
            suitability_map = self.data_lod_manager.get_settlement_data("combined_suitability_map")
            if suitability_map is not None:
                self._push_data_to_current_display(suitability_map, "suitability_map")

        elif current_mode == 1:  # Civilization Map
            civ_map = self.data_lod_manager.get_settlement_data("civ_map")
            if civ_map is not None:
                self._push_data_to_current_display(civ_map, "civ_map")

        elif current_mode == 2:  # Plot Boundaries
            plot_map = self.data_lod_manager.get_settlement_data("plot_map")
            if plot_map is not None:
                self._push_data_to_current_display(plot_map, "plot_map")

        elif current_mode == 3:  # Landscape Voronoi (settlement.landscape_voronoi, #37)
            voronoi_cell_map = self.data_lod_manager.get_settlement_data("voronoi_cell_map")
            if voronoi_cell_map is not None:
                self._push_data_to_current_display(voronoi_cell_map, "voronoi_cell_map")

        elif current_mode == 4:  # City Blocks (settlement.city_blocks, #36)
            house_parcel_map = self.data_lod_manager.get_settlement_data("house_parcel_map")
            if house_parcel_map is not None:
                self._push_data_to_current_display(house_parcel_map, "house_parcel_map")

        self._apply_settlement_overlays()

        # 3D Overlays
        self.apply_3d_overlays()

    def _apply_settlement_overlays(self):
        """
        Zeichnet die ueber Checkboxen zuschaltbaren Overlays (Settlements/
        Landmarks/Roadsites/Roads/City Boundary) auf den aktuell angezeigten
        Basis-Layer - unabhaengig davon, welcher Radio-Modus aktiv ist.
        """
        current_display = self.get_current_display()
        if not current_display or self.current_view != "2d":
            return
        display = current_display.display

        if hasattr(display, 'overlay_settlements') and (
                self.show_settlements_cb.isChecked() or self.show_landmarks_cb.isChecked()
                or self.show_roadsites_cb.isChecked()):
            settlements = self.data_lod_manager.get_settlement_data("settlement_list")
            landmarks = self.data_lod_manager.get_settlement_data("landmark_list")
            roadsites = self.data_lod_manager.get_settlement_data("roadsite_list")

            display_settlements = settlements if self.show_settlements_cb.isChecked() else []
            display_landmarks = landmarks if self.show_landmarks_cb.isChecked() else []
            display_roadsites = roadsites if self.show_roadsites_cb.isChecked() else []

            display.overlay_settlements(display_settlements, display_landmarks, display_roadsites)

        if hasattr(display, 'overlay_roads') and self.show_roads_cb.isChecked():
            # Drei Farben für die drei Road-Kategorien (siehe
            # calculate_road_network()/calculate_landmark_roads()/
            # calculate_outer_connections() in core/settlement_generator.py)
            roads = self.data_lod_manager.get_settlement_data("roads")
            landmark_roads = self.data_lod_manager.get_settlement_data("landmark_roads")
            outer_roads = self.data_lod_manager.get_settlement_data("outer_roads")
            if roads:
                display.overlay_roads(roads, color='darkorange')
            if landmark_roads:
                display.overlay_roads(landmark_roads, color='gold', linewidth=1.0)
            if outer_roads:
                display.overlay_roads(outer_roads, color='dimgray', linewidth=1.0)

        if hasattr(display, 'overlay_street_mask') and self.city_blocks_radio.isChecked():
            street_mask = self.data_lod_manager.get_settlement_data("street_mask")
            if street_mask is not None:
                display.overlay_street_mask(street_mask)

        if hasattr(display, 'overlay_city_boundary_contour') and self.show_city_boundary_cb.isChecked():
            city_mask = self.data_lod_manager.get_settlement_data("city_mask")
            if city_mask is not None:
                display.overlay_city_boundary_contour(city_mask)

    def apply_3d_overlays(self):
        """
        Funktionsweise: 3D-Overlays (Terrain/Settlement-Marker) für Settlement
        Aufgabe: Aktuell bewusst kein Effekt - der 3D-Ausbau wurde auf den
        Basis-Routing-Fix reduziert (siehe [[project-ui-redesign]]), echte
        Overlay-Texturen/Marker-Geometrie in MapDisplay3D existieren noch nicht.
        Die Checkboxen bleiben bedienbar, lösen aber (noch) nichts aus.
        """
        pass

    @pyqtSlot()
    def update_display_mode(self):
        """
        Slot für Visualization-Mode Änderungen.
        update_settlement_display() ruft self.map_display auf, das nie
        zugewiesen wird (die realen Render-Methoden display_settlements()/
        overlay_3d_terrain()/etc. existieren auch nicht auf MapDisplay2D) -
        Settlement-2D/3D-Rendering ist noch nicht implementiert. Bis dahin
        hier abfangen statt hart zu crashen.
        """
        try:
            self.update_settlement_display()
        except AttributeError as e:
            self.logger.debug(f"Settlement display rendering not yet implemented: {e}")

    @pyqtSlot(bool)
    def toggle_3d_terrain(self, enabled: bool):
        """Toggle für 3D Terrain Overlay"""
        self.update_settlement_display()

    @pyqtSlot(bool)
    def toggle_3d_markers(self, enabled: bool):
        """Toggle für 3D Settlement Markers"""
        self.update_settlement_display()


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