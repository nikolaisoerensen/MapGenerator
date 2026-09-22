"""
Path: gui/tabs/settlement_tab.py

Funktionsweise: Settlement-Editor mit terrain_tab-ähnlicher UI-Struktur und vollständiger Core-Integration
- Erbt von BaseMapTab für gemeinsame Features (70/30 Layout, Navigation, etc.)
- UI-Struktur: Generate Button → LOD/Status-Display → Parameter-Panel → Statistics → Navigation (fixiert unten)
- GenerationOrchestrator Integration mit StandardOrchestratorHandler
- Real-time Status-Display für alle calculate-Schritte mit Progress und Validity-State
- Live Settlement-Preview und 3D-Visualization mit Terrain-Integration
"""

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab, Overlay
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
            # Live-Fortschritt während der Plot-Physik-Konvergenz (bis zu 100
            # Iterationen) - siehe [[project-settlement-plot-physics-rebuild]] Teil F.
            self.generation_orchestrator.settlement_plot_live_update.connect(self.on_settlement_plot_live_update)

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

    def create_parameter_controls(self):
        """
        No-Op-Override: SettlementTab baut sein Parameter-Panel über
        setup_settlement_ui()/create_settlement_parameter_panel() statt über
        diesen Basisklassen-Hook (architektonische Abweichung, kein
        fehlendes Feature) - unterdrückt die sonst bei jedem Tab-Start
        geloggte "should implement create_parameter_controls()"-Warnung aus
        BaseMapTab.
        """
        pass

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

        # Anzeige des angeklickten Objekts (docs/OFFENE_PUNKTE.md 6.29).
        # Gehoert in die Statistik-Spalte, nicht ins Parameter-Panel: es ist
        # eine Ausgabe, kein Regler.
        auswahl_box = QGroupBox("Auswahl (3D)")
        auswahl_layout = QVBoxLayout()
        self.auswahl_anzeige = QLabel("Nichts ausgewählt.<br>Linksklick auf Ort oder Weg.")
        self.auswahl_anzeige.setWordWrap(True)
        self.auswahl_anzeige.setTextFormat(Qt.TextFormat.RichText)
        auswahl_layout.addWidget(self.auswahl_anzeige)
        auswahl_box.setLayout(auswahl_layout)
        layout.addWidget(auswahl_box)

        # Signal erst hier verbinden - das 3D-Widget steht zu diesem Zeitpunkt
        # bereits (BaseMapTab.create_ui laeuft davor).
        if self.map_display_3d is not None and hasattr(
                self.map_display_3d.display, "objekt_gewaehlt"):
            self.map_display_3d.display.objekt_gewaehlt.connect(self._on_objekt_gewaehlt)

    def _add_slider_group(self, layout: QVBoxLayout, title: str, param_names: list,
                           default_step: float = 0.1) -> QGroupBox:
        """
        Baut eine QGroupBox mit einem ParameterSlider je param_name (identisches
        Muster wie die bisherigen Einzel-Gruppen unten) und hängt sie an layout.
        Extrahiert, weil das Panel durch die Physics-Lab-Parität (siehe
        [[project-settlement-physics-lab-parity]]) auf 8 Gruppen/25 Slider
        gewachsen ist - vorher war die Duplizierung bei 5 Gruppen noch
        überschaubar.
        """
        group = QGroupBox(title)
        group_layout = QVBoxLayout()
        for param_name in param_names:
            param_config = get_parameter_config("settlement", param_name)
            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", default_step),
                suffix=param_config.get("suffix", ""),
                description=param_config.get("description", "")
            )
            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            group_layout.addWidget(slider)
        group.setLayout(group_layout)
        layout.addWidget(group)
        return group

    def create_settlement_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Settlement-Parametern
        Aufgabe: Alle GUI-exponierten Parameter aus value_default.SETTLEMENT
        strukturiert organisiert, angeglichen an tools/biome_lab's Plot Physics
        Lab (siehe [[project-settlement-physics-lab-parity]]):
        Grundgröße (Plot Nodes, Stadtgröße, Civ-Decay) oben, dann die
        bisherigen Gruppen, eine neue "Plot Physics - Advanced"-Gruppe mit den
        14 Live-Physik-Reglern des Lab, und eine "Forces"-Checkbox-Gruppe ganz
        unten.
        Return: QGroupBox mit Parameter-Slidern
        """
        panel = QGroupBox("Settlement Parameters")
        layout = QVBoxLayout()

        self.parameter_sliders = {}

        # Grundgröße - ganz oben (Nutzer-Vorgabe: Plot-Anzahl, Stadtgröße und
        # Civ-Decay sollen als Erstes einstellbar sein, bevor man sich in die
        # Detail-Parameter vertieft).
        self._add_slider_group(layout, "Grundgröße", ["plotnodes", "city_size", "civ_influence_decay"])

        # Location Count Parameters (plotnodes/civ_influence_decay jetzt oben)
        self._add_slider_group(layout, "Location Counts", ["settlements", "landmarks", "roadsites"])

        # Influence and Terrain Parameters (civ_influence_decay jetzt oben)
        self._add_slider_group(layout, "Civilization Influence", ["terrain_factor_villages"])

        # Road Network Parameters
        self._add_slider_group(layout, "Road Network", ["road_slope_to_distance_ratio"])

        # Wilderness Parameters
        self._add_slider_group(layout, "Wilderness", ["landmark_wilderness"])

        # Plot Physics Parameters (PlotPhysicsSystem, siehe
        # [[project-settlement-plot-physics-rebuild]] Teil A-D) - Grundabstand,
        # Verdichtung zur Stadtmitte und Steigungs-"Baukosten" der
        # Feder-Masse-Simulation, die die Grundstücks-/Straßen-Geometrie erzeugt.
        self._add_slider_group(
            layout, "Plot Physics",
            ["plot_base_spacing", "plot_civ_spacing_factor", "plot_height_cost_factor"])

        # Plot Physics - Advanced: die 14 Live-Physik-Regler aus
        # tools/biome_lab/ui.py (Reihenfolge übernommen), bisher in
        # PlotPhysicsSystem hardcodiert ohne UI-Slider - siehe
        # [[project-settlement-physics-lab-parity]].
        self._add_slider_group(layout, "Plot Physics — Advanced", [
            "core_plotnode_spring_stiffness", "plotnode_plotnode_spring_stiffness",
            "pressure_strength", "core_mass", "plot_node_mass",
            "plot_node_repulsion_strength", "damping", "plot_gravity_strength",
            "plot_city_repulsion_strength", "plot_tier_factor", "potential_strength",
        ])

        # Forces - Kraft-Schalter ganz unten (Nutzer-Vorgabe): anders als im
        # Lab (Default alles AUS, zum einzelnen Isolieren von Kräften beim
        # Debuggen) startet Production mit allem AN, damit die Simulation
        # sofort funktioniert - die Checkboxen sind zum Experimentieren da,
        # nicht als Pflicht-Setup.
        forces_group = QGroupBox("Forces")
        forces_layout = QVBoxLayout()
        self.force_checkboxes = {}
        force_labels = {
            "enable_core_plotnode_spring": "Core ↔ PlotNode Feder",
            "enable_plotnode_plotnode_spring": "PlotNode ↔ PlotNode Feder",
            "enable_pressure": "Innendruck (Flächenerhalt)",
            "enable_plot_node_repulsion": "PlotNode-Abstoßung",
            "enable_field_cores": "Potentialfeld auf Kerne",
            "enable_field_plotnodes": "Potentialfeld auf PlotNodes",
            "enable_core_cell_containment": "Kern-Zellen-Eingrenzung",
            "enable_wilderness_containment": "Wildnis-Eingrenzung",
        }
        for param_name, label in force_labels.items():
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self.on_parameter_changed)
            self.force_checkboxes[param_name] = checkbox
            forces_layout.addWidget(checkbox)
        forces_group.setLayout(forces_layout)
        layout.addWidget(forces_group)

        # Hinweise/Legende - statisches Äquivalent zur Farb-/Marker-Legende
        # aus tools/biome_lab/ui.py, mit den tatsächlichen Production-Farben
        # (siehe map_display_2d.py's PLOT_CORE_COLOR_BY_TYPE/
        # PLOT_NODE_COLOR_BY_TYPE) statt dem Lab-Text 1:1 zu kopieren.
        hints_label = QLabel(
            "Hinweise: Magenta = Wildnis-/Zivilisationskontur · Gold = "
            "Stadtgrenze · Blau/Grün/Rot = Standard-/Wildnis-/Stadt-Kerne · "
            "Straßenfarbe (hell-orange → dunkelrot) = durchschnittlicher "
            "Verkehr über den gesamten Physik-Lauf."
        )
        hints_label.setWordWrap(True)
        hints_label.setStyleSheet("font-size: 10px; color: #666; padding-top: 6px;")
        layout.addWidget(hints_label)

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

        controls_layout.addLayout(self._create_settlement_overlay_toggle_controls())
        controls_layout.addWidget(self._create_vertical_separator())
        controls_layout.addLayout(self._create_settlement_filter_controls())

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_settlement_overlay_toggle_controls(self) -> QHBoxLayout:
        """
        Kein exklusiver Basis-Layer-Umschalter mehr (die frühere Terrain-
        Suitability/Civilization-Map/Plot-Boundaries-Radiogruppe ist entfallen,
        siehe [[project-settlement-physics-lab-parity]]) - Basis ist jetzt
        immer die Heightmap, wie bei den anderen Tabs (Nutzer-Vorgabe: "sonst
        sieht man einfach so wie in anderen tabs nur die heightmap
        (combined)"). Plot-Kerne/-Nodes/-Kanten/Wildnis-/Stadtgrenze werden
        immer als Overlay gezeichnet (siehe update_settlement_display()),
        nicht mehr an einen Radio-Modus gekoppelt.

        Civ-Value und Potential-Field standen bis 2026-08-13 ebenfalls hier -
        nach der Sichtpruefung vom Nutzer entfernt ("Civ-Value und Potential
        Field sind NUR regional. koennen hier entfernt und bei regional
        hinzugefuegt werden"), sie sind jetzt in SettlementRegionalTab. Beides
        sind FLAECHIGE Felder, und dieser Reiter zeigt bewusst nur
        Punkt-/Linienhaftes (siehe _create_settlement_filter_controls()).
        """
        layout = QHBoxLayout()

        # Regionsfaerbung + weisse Grenzen (2026-08-11, docs/OFFENE_PUNKTE.md
        # 6.1) - AUS per Default ("nur wenn man Regionen ausgewaehlt"), in
        # zurueckhaltendem Ton (alpha 0.30, siehe _apply_settlement_overlays()),
        # damit Staedte/Strassen im Vordergrund bleiben.
        self.regions_overlay_cb = QCheckBox("Regionen")
        self.regions_overlay_cb.toggled.connect(self.update_display_mode)
        layout.addWidget(self.regions_overlay_cb)

        return layout

    def _create_settlement_filter_controls(self) -> QHBoxLayout:
        """
        Erstellt Overlay-Checkboxen - kombinierbar mit JEDEM Basis-Layer-Radio
        (siehe update_settlement_display()), nicht auf einen Modus beschränkt.

        NUR PUNKT-/LINIENHAFTE ÜBERSICHT, NICHTS REGIONALES (2026-08-10,
        Nutzer-Vorgabe: "stadtgrenzen und die nodes und alles sollten nur
        regional erscheinen, im globalen bereich ja nur städte,
        verbindungsstraßen und landmarks, roadsites, aber nur als punkte
        jeweils immer"). Globaler Reiter zeigt deshalb NUR Städte/Landmarks/
        Roadsites als Punkte plus die Verbindungsstraßen als Linien - "City
        Boundary" und das Plot-/Node-Feingewebe (PlotPhysicsSystem) sind nach
        SettlementRegionalTab gewandert, wo eine einzelne Region ohnehin groß
        genug im Bild steht, dass diese Detailebene etwas nuetzt.

        `show_roads_cb` ERSETZT die frueher entfernte "Roads"-Checkbox (siehe
        [[project-settlement-physics-lab-parity]]): DAMALS war `roads` die
        alte, straßengerade Pfadfindung von vor dem PlotPhysicsSystem-Umbau
        ("das ist alles noch alter Kram"). Seit 2026-08-10
        (docs/spezifikation/14_SIEDLUNGEN.md Abschnitt 5) ist es ein echtes Gabriel-Graph/
        Kostenfeld/Bereitschafts-Netz - kein Grund mehr, es zu verstecken.
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

    @pyqtSlot(object)
    def on_settlement_plot_live_update(self, snapshot: dict):
        """
        Funktionsweise: Slot für Live-Fortschritt der Plot-Physik-Konvergenz
        (siehe [[project-settlement-plot-physics-rebuild]] Teil F) - zeichnet
        den noch nicht konvergierten Zwischenzustand nach, solange der
        Settlement-Tab in 2D sichtbar ist (Plot-Physik-Overlay ist seit
        [[project-settlement-physics-lab-parity]] immer aktiv, kein
        exklusiver Modus mehr).
        Aufgabe: Analog zu draw.py im ursprünglichen Physics Lab, nur im
        echten Tool statt im Sandbox-Fenster.
        """
        try:
            if self.current_view != "2d":
                return
            current_display = self.get_current_display()
            if not current_display:
                return
            display = current_display.display
            if hasattr(display, 'draw_plot_physics_snapshot'):
                display.draw_plot_physics_snapshot(snapshot)
        except Exception as e:
            self.logger.debug(f"Settlement plot live update failed: {e}")

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
        for param_name, checkbox in getattr(self, "force_checkboxes", {}).items():
            parameters[param_name] = checkbox.isChecked()
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
        Funktionsweise: Basis-Layer ist immer die Heightmap (wie bei den
        anderen Tabs, siehe [[project-settlement-physics-lab-parity]] -
        Nutzer-Vorgabe: "sonst sieht man einfach so wie in anderen tabs nur
        die heightmap (combined)").

        KEIN Plot-/Node-Feingewebe MEHR HIER (2026-08-10, Nutzer-Vorgabe:
        "stadtgrenzen und die nodes und alles sollten nur regional
        erscheinen, im globalen bereich ja nur städte, verbindungsstraßen und
        landmarks, roadsites, aber nur als punkte jeweils immer"). Das
        `overlay_plot_boundaries()`-Aufrufziel (PlotPhysicsSystem-Kerne/
        -Nodes/-Kanten/Wildnisgrenze) ist nach SettlementRegionalTab
        gewandert - auf der Weltkarte waeren tausende Hausparzellen ohnehin
        nur Pixelmatsch, die Regionsansicht zoomt weit genug, dass sie
        wirklich etwas zeigen.
        Aufgabe: Punkt-/Linien-Overlays (Settlements/Landmarks/Roadsites/
        Roads per Checkbox, Civ-Value/Potential-Field per Checkbox) bleiben
        unabhängig vom Basis-Layer zuschaltbar, siehe _apply_settlement_overlays().

        Nutzt wie die anderen Tabs get_current_display()/_push_data_to_current_display()
        statt eines nie zugewiesenen self.map_display.
        """
        heightmap = self.data_lod_manager.get_terrain_data("heightmap")
        if heightmap is not None:
            self._push_data_to_current_display(heightmap, "heightmap")

        self._apply_settlement_overlays()

        # 3D Overlays (Wegbaender/Plots/Auswahlobjekte - noch nicht im Register)
        self.apply_3d_overlays()

        # Siedlungspunkte (Staedte/Landmarken/Roadsites) UEBER DAS REGISTER
        # (Ticket #10, docs/spezifikation/15_ANZEIGE.md): ersetzt die bisherige
        # Verdopplung aus `overlay_settlements()` in _apply_settlement_overlays()
        # (nur 2D, current_view-gated) und der `uebersicht`-Textur in
        # apply_3d_overlays() (nur 3D, ohne Fingerabdruck-Cache) durch einen
        # gemeinsamen Aufruf - derselbe Weg, den BiomeTab.apply_overlays()
        # bereits benutzt.
        settlements = self.data_lod_manager.get_settlement_data("settlement_list")
        landmarks = self.data_lod_manager.get_settlement_data("landmark_list")
        roadsites = self.data_lod_manager.get_settlement_data("roadsite_list")
        display_settlements = settlements if self.show_settlements_cb.isChecked() else []
        display_landmarks = landmarks if self.show_landmarks_cb.isChecked() else []
        display_roadsites = roadsites if self.show_roadsites_cb.isChecked() else []
        siedlungen_sichtbar = bool(display_settlements or display_landmarks or display_roadsites)
        self._push_overlays([
            Overlay("siedlungen", sichtbar=siedlungen_sichtbar,
                    daten=(display_settlements, display_landmarks, display_roadsites)),
        ])

    def _apply_settlement_overlays(self):
        """
        Zeichnet die ueber Checkboxen zuschaltbaren Overlays (Settlements/
        Landmarks/Roadsites als Punkte, Roads als Linien) auf den aktuell
        angezeigten Basis-Layer - unabhaengig davon, welcher Radio-Modus
        aktiv ist. Bewusst NUR die weltkartentaugliche Uebersicht (siehe
        update_settlement_display()); City Boundary und das Plot-/Node-
        Feingewebe zeigt SettlementRegionalTab.
        """
        current_display = self.get_current_display()
        if not current_display or self.current_view != "2d":
            return
        display = current_display.display

        # Grenzen der neun Regionalkarten (docs/OFFENE_PUNKTE.md 6.2/5.15):
        # Global UND Regional, nicht der Terrain-Reiter - dort laege es neben
        # den Kulturfarben. Seit 2026-08-13 werden die TATSAECHLICHEN
        # Vieleck-Grenzen gezeichnet, wenn die Zerlegung vorliegt; sonst
        # weiterhin das alte gerade 3x3-Raster. Ein gerades Gitter neben einer
        # Vieleck-Zerlegung waere eine zweite, falsche Neunerteilung.
        if hasattr(display, 'overlay_region_grid'):
            heightmap = self.data_lod_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                display.overlay_region_grid(
                    heightmap.shape[0],
                    spielkarte=self.data_lod_manager.get_terrain_data("spielkarte"))

        # Siedlungspunkte (Settlements/Landmarks/Roadsites) laufen seit Ticket
        # #10 NICHT mehr hier, sondern gemeinsam mit dem 3D-Weg ueber
        # self._push_overlays() in update_settlement_display() - siehe dort.

        # Verbindungsstrassen (docs/spezifikation/14_SIEDLUNGEN.md Abschnitt 5): Landwege
        # durchgezogen orange, Seewege gestrichelt in eigenem Blau (§4.4
        # "anders gezeichnet - gestrichelt, in einem eigenen Blau").
        if hasattr(display, 'overlay_roads') and self.show_roads_cb.isChecked():
            roads = self.data_lod_manager.get_settlement_data("roads")
            sea_roads = self.data_lod_manager.get_settlement_data("sea_roads")
            if roads:
                display.overlay_roads(roads, color='darkorange')
            if sea_roads:
                display.overlay_roads(sea_roads, color='royalblue', linestyle='--')

        # Civ-Value/Potenzialfeld: 2026-08-13 nach SettlementRegionalTab
        # gewandert (Nutzer-Vorgabe nach der Sichtpruefung: "Civ-Value und
        # Potential Field sind NUR regional. koennen hier entfernt und bei
        # regional hinzugefuegt werden") - dieselbe Trennung wie schon bei
        # Stadtgrenze/Plot-Feingewebe, siehe
        # _create_settlement_filter_controls()-Docstring.

        # Regionen (docs/OFFENE_PUNKTE.md 6.1) - Deckkraft 2026-08-13 von 0.25
        # auf 0.40 angehoben (Nutzer nach der Sichtpruefung: "etwas zu subtil,
        # mach mal 20% mehr", nach dem ersten Versuch mit 0.30 dann
        # ausdruecklich "nein mach mal 0.4"). Bleibt unter dem Wert des
        # Terrain-Reiters (0.55), damit Staedte/Strassen im Vordergrund
        # bleiben.
        if hasattr(display, 'overlay_regions') and self.regions_overlay_cb.isChecked():
            heightmap = self.data_lod_manager.get_terrain_data("heightmap")
            region_map = self.data_lod_manager.get_terrain_data("region_map")
            if heightmap is not None and region_map is not None:
                display.overlay_regions(region_map, heightmap, alpha=0.40)

    def apply_3d_overlays(self):
        """
        KEIN Plot-/Node-Skin mehr hier (2026-08-10, Nutzer-Vorgabe siehe
        _create_settlement_filter_controls()-Docstring) - die Rasterisierung
        von PlotPhysicsSystem als 3D-Textur ist nach
        SettlementRegionalTab.apply_3d_overlays() gewandert. `self.map_display_3d`
        gehoert diesem Tab exklusiv (jeder BaseMapTab bekommt sein eigenes
        3D-Widget, siehe base_tab.py create_ui()) - ein frueher hier gesetzter
        Plot-Layer wuerde also nicht von selbst verschwinden, wenn er nicht
        mehr gefuellt wird. Explizit ausblenden statt nur "nicht mehr fuellen".

        Die Siedlungspunkte selbst (Staedte/Landmarken/Roadsites als
        RGBA-Skin auf dem Gelaende) baut seit Ticket #10 nicht mehr diese
        Methode, sondern das gemeinsame Overlay-Register ueber
        self._push_overlays() (update_settlement_display()) - derselbe Weg,
        den auch BiomeTab benutzt. Hier bleiben nur die Wege ("wegbaender",
        echte Bandgeometrie statt Textur, docs/OFFENE_PUNKTE.md 6.28 - noch
        nicht im Register) und die anklickbaren Auswahlobjekte.

        Die Auswahl folgt DENSELBEN Checkboxen wie die 2D-Ansicht - was in 2D
        ausgeschaltet ist, fehlt auch auf dem Skin.
        """
        if not self.map_display_3d or not hasattr(self.map_display_3d.display, 'set_layer_visibility'):
            return

        display_3d = self.map_display_3d.display
        display_3d.set_layer_visibility("settlement", "plots", False)

        if not hasattr(display_3d, 'update_overlay_data'):
            return

        heightmap = self.data_lod_manager.get_terrain_data("heightmap")
        if heightmap is None:
            return

        # Die Siedlungspunkte-Textur ("uebersicht") baut seit Ticket #10 nicht
        # mehr diese Methode, sondern self._push_overlays() in
        # update_settlement_display() (Overlay "siedlungen", mit
        # Fingerabdruck-Cache aus base_tab.py._siedlungen_3d). Die Variablen
        # settlements/landmarks/roadsites werden hier trotzdem gebraucht - fuer
        # die anklickbaren Auswahlobjekte weiter unten.
        settlements = (self.data_lod_manager.get_settlement_data("settlement_list") or []
                       if self.show_settlements_cb.isChecked() else [])
        landmarks = (self.data_lod_manager.get_settlement_data("landmark_list") or []
                     if self.show_landmarks_cb.isChecked() else [])
        roadsites = (self.data_lod_manager.get_settlement_data("roadsite_list") or []
                     if self.show_roadsites_cb.isChecked() else [])
        if self.show_roads_cb.isChecked():
            roads = self.data_lod_manager.get_settlement_data("roads") or []
            sea_roads = self.data_lod_manager.get_settlement_data("sea_roads") or []
        else:
            roads, sea_roads = [], []

        hat_wege = bool(roads or sea_roads)
        if hat_wege:
            display_3d.update_overlay_data(
                "settlement", "wegbaender",
                {"wege": roads, "seewege": sea_roads})
        display_3d.set_layer_visibility("settlement", "wegbaender", hat_wege)

        # ANKLICKBARE OBJEKTE hinterlegen (docs/OFFENE_PUNKTE.md 6.29).
        # Die Kennung ist ein fertiges dict - der Reiter kennt die Bedeutung
        # der Felder, das Anzeige-Widget muss sie nicht kennen.
        if hasattr(display_3d, "setze_auswahlobjekte"):
            welt_km = float(self.data_lod_manager.get_map_distance_km())
            groesse = heightmap.shape[0]
            auswahl_orte = []
            for gruppe, art in ((settlements, "Siedlung"),
                                (landmarks, "Landmark"),
                                (roadsites, "Roadsite")):
                for ort in gruppe:
                    auswahl_orte.append((
                        float(getattr(ort, "x", 0.0)), float(getattr(ort, "y", 0.0)),
                        self._auswahl_beschreibung(ort, art)))

            from gui.widgets.karten_auswahl import weglaenge_km
            auswahl_wege = []
            for liste, art in ((roads, "Landweg"), (sea_roads, "Seeweg")):
                for nummer, pfad in enumerate(liste or [], start=1):
                    punkte = [(p[0], p[1]) for p in pfad]
                    auswahl_wege.append((punkte, {
                        "art": art,
                        "titel": f"{art} {nummer}",
                        "zeilen": [f"Länge {weglaenge_km(punkte, welt_km, groesse):.1f} km",
                                   f"{len(punkte)} Stützpunkte"],
                    }))
            display_3d.setze_auswahlobjekte(auswahl_orte, auswahl_wege)

    def _auswahl_beschreibung(self, ort, art):
        """
        Was beim Anklicken eines Ortes angezeigt wird (docs/OFFENE_PUNKTE.md
        6.29). Nur Felder, die tatsaechlich belegt SIND - eine Zeile
        "Einwohner: 0" waere schlechter als gar keine.
        """
        from core.settlement_generator import STADTTYPEN
        zeilen = []
        # Stadttyp NUR bei Siedlungen und NUR, wenn er etwas aussagt.
        # `Location.settlement_type` traegt auch bei Landmarks und Roadsites
        # den Vorgabewert "sonstige" - der wuerde dort als "Ort" erscheinen,
        # obwohl diese Objekte gar keinen Stadttyp haben (gemessen beim
        # Bauen: ein Steinkreis wurde als "Ort" beschriftet). "sonstige" ist
        # zudem der Auffangtyp und auch bei echten Siedlungen keine Auskunft.
        typ = getattr(ort, "settlement_type", "") or ""
        if art == "Siedlung" and typ and typ != "sonstige":
            zeilen.append(STADTTYPEN.get(typ, {}).get("name", typ))
        if getattr(ort, "culture", ""):
            zeilen.append(f"Kultur: {ort.culture}")
        if getattr(ort, "rank", ""):
            zeilen.append(f"Rang: {ort.rank}")
        if getattr(ort, "house_count", 0):
            zeilen.append(f"{ort.house_count} Häuser")
        eigenschaften = getattr(ort, "properties", None) or {}
        for schluessel, beschriftung in (("landmark_type", ""),
                                          ("roadsite_type", ""),
                                          ("kategorie", "Lage")):
            wert = eigenschaften.get(schluessel)
            if wert:
                zeilen.append(f"{beschriftung}: {wert}" if beschriftung else str(wert))
        titel = (eigenschaften.get("landmark_type")
                 or eigenschaften.get("roadsite_type")
                 or f"{art} {getattr(ort, 'location_id', '')}".strip())
        return {"art": art, "titel": titel, "zeilen": zeilen,
                "id": getattr(ort, "location_id", None)}

    def _on_objekt_gewaehlt(self, treffer):
        """Zeigt das angeklickte Objekt an - oder leert die Anzeige beim Klick
        ins Leere. Kein Fehlerfall: nichts zu treffen ist ein gueltiges
        Ergebnis."""
        if not hasattr(self, "auswahl_anzeige") or self.auswahl_anzeige is None:
            return
        if not treffer:
            # <br> statt \n: das Feld steht auf RichText (siehe
            # create_statistics_controls), dort ist \n kein Zeilenumbruch.
            self.auswahl_anzeige.setText("Nichts ausgewählt.<br>"
                                          "Linksklick auf Ort oder Weg.")
            return
        kennung = treffer.get("kennung") or {}
        zeilen = [f"<b>{kennung.get('titel', '?')}</b>",
                  f"<i>{kennung.get('art', '')}</i>"]
        zeilen.extend(kennung.get("zeilen", []))
        self.auswahl_anzeige.setText("<br>".join(str(z) for z in zeilen))

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
