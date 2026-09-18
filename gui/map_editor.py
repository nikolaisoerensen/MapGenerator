"""
Path: gui/map_editor_window.py

MapEditor Main Window with Enhanced Tab Management
=================================================

Professional map editor providing tabbed interface for all generator types
with integrated GenerationOrchestrator coordination, comprehensive error
handling, and optimized resource management.

Features:
- Dynamic tab loading with intelligent fallback handling
- Real-time generation status monitoring
- Integrated toolbar and menu system
- Professional error recovery and user feedback
- Memory-efficient display management
- Cross-tab communication and dependency tracking

Architecture:
- TabWidget container for 8 generator tabs
- NavigationManager integration for seamless tab flow
- DataManager coordination for cross-tab data sharing
- GenerationOrchestrator integration for centralized generation control
- Comprehensive status monitoring and progress tracking
"""

# TODO: Exit aus Map Editor muss sauber gemacht werden mit Cleanup.
# TODO: Exit aus Map Editor vorher Signal jetzt direkt: was ist besser? Ich habe schließlich laufende Berechnungen.

from PyQt6.QtWidgets import QMainWindow, QApplication, QTabWidget, QTabBar, QStackedWidget, QMenu, QLabel, \
    QComboBox, QCheckBox, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QSplitter, \
    QRadioButton, QButtonGroup, QInputDialog
from PyQt6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PyQt6.QtCore import QTimer, Qt, pyqtSlot
import logging
import os
from typing import Optional

from gui.config.gui_default import WindowSettings, EditorConstants
from managers.data_lod_manager import DataLODManager
from managers.generation_orchestrator import GenerationOrchestrator, GeneratorType
from core.welt_io import welt_backen, welt_laden, WeltBackenFehler, WeltLadenFehler
from managers.navigation_manager import NavigationManager
from gui.widgets.widgets import ParameterSlider
from managers.parameter_manager import ParameterManager
from managers.shader_manager import ShaderManager
from gui.widgets.widgets import BaseButton, StatusIndicator, ProgressBar
from gui.widgets.pipeline_status_panel import PipelineStatusPanel
from gui.utils.progress_weighting import WeightedProgressCalculator


# Professional tab imports with comprehensive error handling
def _import_tab_safely(module_path: str, class_name: str) -> tuple[bool, Optional[type], str]:
    """
    Safe tab import with detailed error classification
    ================================================

    Attempts to import tab class and provides detailed feedback
    on the type of failure for appropriate error handling.
    Import failures are logged with the original exception so that
    missing classes or broken imports are visible at startup.

    Args:
        module_path: Python module path to import from
        class_name: Class name to import

    Returns:
        Tuple of (import_success, class_object, error_type)
        error_type: "import_failed", "class_missing", "instantiation_failed", or "success"
    """
    logger = logging.getLogger(__name__)

    try:
        module = __import__(module_path, fromlist=[class_name])
        if not hasattr(module, class_name):
            logger.warning(f"Tab import: {module_path} loaded, but class {class_name} is missing")
            return False, None, "class_missing"

        tab_class = getattr(module, class_name)

        if not callable(tab_class):
            logger.warning(f"Tab import: {module_path}.{class_name} is not callable")
            return False, None, "class_missing"

        return True, tab_class, "success"

    except ImportError as e:
        logger.warning(f"Tab import failed for {module_path}.{class_name}: {e}")
        return False, None, "import_failed"
    except Exception as e:
        logger.warning(f"Tab import error for {module_path}.{class_name}: {e}")
        return False, None, "instantiation_failed"


# Import available tabs
TERRAIN_AVAILABLE, TerrainTab, terrain_error = _import_tab_safely("gui.tabs.terrain_tab", "TerrainTab")
GEOLOGY_AVAILABLE, GeologyTab, geology_error = _import_tab_safely("gui.tabs.geology_tab", "GeologyTab")
EROSION_AVAILABLE, ErosionTab, erosion_error = _import_tab_safely("gui.tabs.erosion_tab", "ErosionTab")
WEATHER_AVAILABLE, WeatherTab, weather_error = _import_tab_safely("gui.tabs.weather_tab", "WeatherTab")
WATER_AVAILABLE, WaterTab, water_error = _import_tab_safely("gui.tabs.water_tab", "WaterTab")
BIOME_AVAILABLE, BiomeTab, biome_error = _import_tab_safely("gui.tabs.biome_tab", "BiomeTab")
SETTLEMENT_AVAILABLE, SettlementTab, settlement_error = _import_tab_safely("gui.tabs.settlement_tab", "SettlementTab")
OVERVIEW_AVAILABLE, OverviewTab, overview_error = _import_tab_safely("gui.tabs.overview_tab", "OverviewTab")
# Seit 2026-08-05: eigener Reiter fuer das Flussnetz (steht hinter Terrain,
# weil es dessen Heightmap formt) und ein zweiter Siedlungsreiter, der auf
# eine der neun Regionen zoomt.
REGION_AVAILABLE, RegionTab, region_error = _import_tab_safely(
    "gui.tabs.region_tab", "RegionTab")
KONTINENT_AVAILABLE, KontinentTab, kontinent_error = _import_tab_safely(
    "gui.tabs.kontinent_tab", "KontinentTab")
RIVER_AVAILABLE, RiverTab, river_error = _import_tab_safely("gui.tabs.river_tab", "RiverTab")
SETTLEMENT_REGIONAL_AVAILABLE, SettlementRegionalTab, settlement_regional_error = _import_tab_safely(
    "gui.tabs.settlement_regional_tab", "SettlementRegionalTab")


class MapEditorWindow(QMainWindow):
    """
    Professional Map Editor Window with Integrated Generator Management
    ==================================================================

    Main container for all map generation tabs providing unified interface
    for terrain, geology, weather, water, biome, settlement, and overview
    generation. Integrates with GenerationOrchestrator for coordinated
    generation workflows and provides comprehensive status monitoring.

    Key Features:
    - Dynamic tab loading with intelligent error handling
    - Real-time generation progress monitoring
    - Cross-tab dependency management
    - Professional menu and toolbar integration
    - Memory-efficient resource management
    - Comprehensive error recovery
    """

    # Tab-Schluessel (siehe tab_order/tab_configs in _setup_tabs()), die keine
    # eigene generate()-Methode haben und daher nicht ueber den globalen
    # Footer-Knopf generieren koennen: Region- und Kontinent-Tab dienen der
    # Voransicht/Parametrierung, die eigentliche Generierung beginnt erst ab
    # dem Terrain-Tab. Fuer genau diese beiden zeigt der Footer-Knopf
    # "WEITER" (danger/rot) statt "GENERIEREN" (primary/gruen) und springt
    # zum naechsten Tab, statt zu versuchen zu generieren - siehe
    # _update_footer_button_for_tab()/_on_footer_button_clicked().
    WEITER_TAB_KEYS = frozenset({"region", "kontinent"})

    def __init__(self, main_menu=None):
        # Kein Qt-Parent: ein Top-Level-Fenster MIT Parent bekommt unter Windows
        # keinen eigenen Taskbar-Eintrag (besonders wenn der Owner nur versteckt,
        # nicht geschlossen ist - siehe main_menu.py._on_map_editor_button_clicked).
        # main_menu wird separat gehalten, nur um beim Schließen dorthin
        # zurückzukehren (_return_to_main_menu/closeEvent), nicht für Qt-Ownership.
        super().__init__(None)
        self._main_menu = main_menu

        self.logger = logging.getLogger(__name__)

        # Manager einzeln initialisieren
        self.data_lod_manager = None
        self.shader_manager = None
        self.parameter_manager = None
        self.navigation_manager = None
        self.generation_orchestrator = None

        self._setup_managers()
        self._check_managers()

        # UI components
        # main_tab_bar ist der reine Haupt-Tab-Selektor (Overview/Terrain/...).
        # Spalte 2 (viewport_stack) und Spalte 3 (parameter_stack/statistics_stack)
        # sind eigene QStackedWidgets, die synchron zum main_tab_bar umschalten.
        # Spalte 1 (pipeline_status_panel) ist davon unabhängig und bleibt konstant.
        self.main_tab_bar = None
        self.viewport_stack = None
        self.parameter_stack = None
        self.statistics_stack = None
        self.side_tab_widget = None
        # lowercase tab names, Index == main_tab_bar/stack index.
        #
        # TICKET #56: wird NICHT aus managers.navigation_manager.GENERATOR_TAB_ORDER
        # gespeist, obwohl das auf den ersten Blick wie dieselbe Liste aussieht.
        # Es ist eine ANDERE Menge: diese hier ist die vollstaendige Reihenfolge
        # der main_tab_bar/viewport_stack-Spalte, wie sie unten in
        # _setup_tabs()/tab_configs steht (region, kontinent, terrain, rivers,
        # geology, erosion, weather, water, biome, settlement,
        # settlement_regional, overview) und wird dynamisch nur aus den
        # tatsaechlich erfolgreich erzeugten Tabs befuellt (_add_successful_tab/
        # _add_error_tab). GENERATOR_TAB_ORDER ist dagegen nur die Teilmenge der
        # linearen Generator-Pipeline fuer die Previous/Next-Navigation
        # (NavigationManager/NavigationPanel) - ohne region/kontinent/rivers/
        # settlement_regional, dafuer mit vorangestelltem "main_menu". Beide
        # zusammenzulegen wuerde entweder Reiter aus der Tab-Leiste werfen oder
        # NavigationManager fremde Reiter unterschieben - das Ticket verlangt
        # ausdruecklich, dass die sichtbare Reihenfolge unveraendert bleibt.
        self.tab_order = []
        self.tabs = {}
        # Fertigkeits-Zustand je Generator, gespiegelt in der Tab-Beschriftung.
        # Siehe _set_tab_state() - das ist der Ersatz fuer die grobe
        # LOD-Vorschau: man sieht auf einen Blick, welcher Reiter schon etwas
        # zu zeigen hat, waehrend der naechste noch rechnet.
        self.tab_states = {}
        # Globale 2D/3D-Präferenz, tabübergreifend (User-Report: "beim Wechsel des
        # Tabs ist man vom 3D Modus wieder im 2D Modus") - jeder Tab hat zwar sein
        # eigenes current_view, aber ein frisch angezeigter Tab, der nie manuell
        # umgeschaltet wurde, soll trotzdem den zuletzt GEWÄHLTEN Modus zeigen statt
        # immer bei seinem eigenen 2D-Default zu bleiben. Aktualisiert via
        # BaseMapTab.view_switched-Signal (siehe _register_tab_view_signal()),
        # angewendet in _on_tab_changed().
        self.global_view_mode = "2d"
        self.pipeline_status_panel = None
        self.footer_progress_bar = None
        self.generate_button = None

        # Generation monitoring
        self.tab_generation_status = {}  # tab_name -> lod_status mapping
        self.active_generations = set()  # Set of active generation keys
        self.progress_calculator = WeightedProgressCalculator()

        # Status monitoring
        self.status_update_timer = QTimer()
        self.status_update_timer.timeout.connect(self._update_status)

        # Initialize window
        self._setup_window()
        self._setup_ui()
        self._setup_tabs()
        self._setup_signals()

        # main_tab_bar.currentChanged wird erst nach _setup_tabs() verbunden,
        # daher feuert es für den initial aktiven Tab (Index 0) nicht - einmalig
        # manuell nachholen, damit der Default-Render-Modus beim Start sichtbar ist.
        self._on_tab_changed(self.main_tab_bar.currentIndex())

        # Start status monitoring
        self.status_update_timer.start(EditorConstants.STATUS_UPDATE_INTERVAL_MS)

        # KEIN Auto-Start mehr (2026-07-28). Hier stand bis dahin
        # `QTimer.singleShot(0, self._auto_start_generation)`, was beim Öffnen
        # des Editors sofort die gesamte Pipeline mit Default-Parametern
        # anwarf.
        #
        # Das war sinnvoll, solange die LOD-Leiter nach wenigen Sekunden eine
        # grobe Vorschau lieferte. Inzwischen rechnet die Erosion nur noch am
        # finalen LOD und braucht dort zweistellige Sekunden - der Nutzer
        # wartete also auf einen vollständigen Lauf, den er gar nicht bestellt
        # hatte, und musste ihn abbrechen, um überhaupt einen Parameter
        # einstellen zu können.
        #
        # Der Lauf startet jetzt ausschliesslich über [GENERIEREN] (bzw. Enter)
        # oder "Regenerate All" (Ctrl+R). `_auto_start_generation()` bleibt als
        # Methode bestehen - sie ist der Pfad, den beide benutzen.
        self._mark_all_tabs_pending()

        self.logger.info("MapEditor window initialized successfully")

    def _setup_window(self):
        """
        Configure main window properties and layout
        ==========================================

        Sets window dimensions, minimum sizes, and positioning
        based on configuration from gui_default.py for
        consistent application appearance.
        """
        self.setWindowTitle("MapGenerator - Professional Map Editor")

        # Load settings from configuration
        # (Abfrage nach Bildschirmgröße wird nicht gemacht, kann Fehler verursachen)
        try:
            settings = WindowSettings.MAP_EDITOR
        except AttributeError:
            self.logger.error("WindowSettings.MAP_EDITOR not found. Fallback settings.")
            settings = {"width": 1500, "height": 1000, "min_width": 1200, "min_height": 800}

        self.resize(settings.get("width"), settings.get("height"))
        self.setMinimumSize(settings.get("min_width"), settings.get("min_height"))

        # Center window on screen
        self._center_window()

    def _center_window(self):
        """Center window on primary display"""
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.geometry()
        center_x = (screen_geometry.width() - window_geometry.width()) // 2
        center_y = (screen_geometry.height() - window_geometry.height()) // 2
        self.move(center_x, center_y)

    def _setup_ui(self):
        """
        Create main user interface layout
        ================================

        Constructs the primary UI including tab widget, menu bar,
        toolbar, and status bar with professional styling and
        intuitive navigation structure.
        """
        # Haupt-Tab-Selektor (Overview/Terrain/...). Reiner Selektor ohne eigene
        # Inhalts-Seiten - Spalte 2/3 stecken in separaten QStackedWidgets, die
        # synchron zum main_tab_bar umgeschaltet werden. So bleibt Spalte 1
        # (Pipeline-Status) unabhängig vom Haupt-Tab sichtbar.
        self.main_tab_bar = QTabBar()
        self.main_tab_bar.setShape(QTabBar.Shape.RoundedNorth)
        self.main_tab_bar.setExpanding(False)

        # Spalte 1 (fix): globaler Pipeline-Status, bleibt über alle Haupt-Tabs
        # hinweg sichtbar.
        self.pipeline_status_panel = PipelineStatusPanel()
        self.pipeline_status_panel.setMinimumWidth(190)
        self.pipeline_status_panel.setMaximumWidth(190)

        # Spalte 2 (flexibel): Viewport der Tabs, per BaseMapTab.viewport_widget befüllt,
        # darunter die permanente globale Checkbox-Zeile (Contour Lines), die
        # unabhängig vom Haupt-Tab immer sichtbar ist, plus die Geology-Cross-
        # Section-Regler (nur sichtbar, wenn der Geology-Tab aktiv ist - siehe
        # _on_tab_changed()). Die frühere "Shadows (Shading)"-Checkbox wurde
        # auf Nutzer-Wunsch entfernt (UI-Aufräumung Teil 2) - shadows_enabled
        # bleibt in MapDisplay3D dauerhaft bei seinem Default True.
        self.viewport_stack = QStackedWidget()

        self.contour_checkbox = QCheckBox("Contour Lines (Höhenlinien)")
        # Nutzer-Vorgabe 2026-07-24: Default AN (vorher unchecked) - gilt nur
        # für den Start-Zustand, kein automatisches Wiederanschalten, falls
        # der Nutzer es manuell ausschaltet (reiner Default-Wert, keine
        # erzwungene Rückstellung an anderer Stelle).
        self.contour_checkbox.setChecked(True)
        self.contour_checkbox.toggled.connect(self._on_global_contour_toggled)

        # Cross-Section-Regler (Achse + Position) - nur für den Geology-Tab
        # relevant, direkt neben Contour Lines platziert (Nutzer-Vorgabe:
        # "unterhalb der Hauptgrafik, direkt neben die Option Contour Lines").
        self.cross_section_x_radio = QRadioButton("Cut along X")
        self.cross_section_x_radio.setChecked(True)
        self.cross_section_y_radio = QRadioButton("Cut along Y")
        self.cross_section_axis_group = QButtonGroup(self)
        self.cross_section_axis_group.addButton(self.cross_section_x_radio, 0)
        self.cross_section_axis_group.addButton(self.cross_section_y_radio, 1)
        self.cross_section_x_radio.toggled.connect(
            lambda checked: self._on_global_cross_section_axis_changed("x", checked))
        self.cross_section_y_radio.toggled.connect(
            lambda checked: self._on_global_cross_section_axis_changed("y", checked))

        self.cross_section_position_slider = ParameterSlider(
            label="Cross-Section Position", min_val=0.0, max_val=1.0, default_val=0.5, step=0.01,
            description="Position des geologischen Schnitts entlang der jeweils "
                         "anderen Achse (0=Rand, 1=gegenüberliegender Rand)."
        )
        self.cross_section_position_slider.valueChanged.connect(self._on_global_cross_section_position_changed)

        self.cross_section_widgets = [
            self.cross_section_x_radio, self.cross_section_y_radio, self.cross_section_position_slider,
        ]

        global_overlay_row = QWidget()
        global_overlay_layout = QHBoxLayout(global_overlay_row)
        global_overlay_layout.setContentsMargins(10, 4, 10, 4)
        global_overlay_layout.addWidget(self.contour_checkbox)
        global_overlay_layout.addWidget(self.cross_section_x_radio)
        global_overlay_layout.addWidget(self.cross_section_y_radio)
        global_overlay_layout.addWidget(self.cross_section_position_slider)
        global_overlay_layout.addStretch()

        # Standardmäßig ausgeblendet, bis der Geology-Tab aktiv wird (siehe
        # _on_tab_changed()) - beim allerersten Tab (Terrain) ist noch kein
        # currentChanged-Signal gefeuert worden.
        for widget in self.cross_section_widgets:
            widget.setVisible(False)

        self.center_widget = QWidget()
        center_layout = QVBoxLayout(self.center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.addWidget(self.viewport_stack, 1)
        center_layout.addWidget(global_overlay_row)

        # Spalte 3 (fix): Parameter/Statistics-Tabs. Bleibt als Chrome konstant,
        # nur der Inhalt der beiden inneren Stacks schaltet je Haupt-Tab um.
        self.parameter_stack = QStackedWidget()
        self.statistics_stack = QStackedWidget()

        self.side_tab_widget = QTabWidget()
        self.side_tab_widget.setDocumentMode(True)
        self.side_tab_widget.addTab(self.parameter_stack, "Parameter")
        self.side_tab_widget.addTab(self.statistics_stack, "Statistics")
        self.side_tab_widget.setMinimumWidth(380)
        self.side_tab_widget.setMaximumWidth(380)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.pipeline_status_panel)
        self.main_splitter.addWidget(self.center_widget)
        self.main_splitter.addWidget(self.side_tab_widget)
        # Nur Spalte 2 wächst bei Ultrawide-Auflösungen; Spalte 1 und 3 bleiben fix.
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.main_tab_bar)
        central_layout.addWidget(self.main_splitter, 1)
        central_layout.addWidget(self._create_footer_bar())

        self.setCentralWidget(central_widget)

        # Create UI components
        self._create_menu_bar()
        self._create_status_bar()

    def _create_footer_bar(self) -> QWidget:
        """
        Fußzeile: gewichteter Ladebalken (29 Klassen x LOD-Kosten, siehe
        WeightedProgressCalculator) links, permanenter Knopf rechts.

        Der Knopf ist EIN globales Widget fuer alle Tabs (nicht pro Tab), und
        sein Verhalten haengt vom aktiven Tab ab: auf Region/Kontinent (siehe
        WEITER_TAB_KEYS) zeigt er "WEITER" (danger/rot) und springt zum
        naechsten Tab, weil diese beiden kein generate() haben; auf allen
        anderen Tabs zeigt er "GENERIEREN" (primary/gruen) wie bisher.
        _on_footer_button_clicked() ist der gemeinsame Klick-Handler, der
        anhand des aktiven Tabs entscheidet; _update_footer_button_for_tab()
        (aus _on_tab_changed() aufgerufen) haelt Text/Farbe synchron.
        """
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 6)

        self.footer_progress_bar = ProgressBar()
        footer_layout.addWidget(self.footer_progress_bar, 1)

        self.generate_button = BaseButton("GENERIEREN", "primary")
        self.generate_button.clicked.connect(self._on_footer_button_clicked)

        # LEERTASTE FREIGEBEN (2026-07-28). Es gab nie einen Space-Shortcut im
        # Projekt - die Leertaste generierte, weil Qt damit den FOKUSSIERTEN
        # QPushButton ausloest, und das war nach jedem Klick dieser hier. Die
        # 3D-Ansicht braucht die Leertaste aber zum Aufsteigen im Flugmodus.
        #
        # NoFocus nimmt der Leertaste den Empfaenger; Enter/Return kommt
        # darunter als ausdruecklicher Shortcut zurueck, damit die Tastatur-
        # Bedienung nicht ersatzlos verschwindet.
        self.generate_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        footer_layout.addWidget(self.generate_button)

        for key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(self._on_footer_button_clicked)

        return footer

    def _on_footer_button_clicked(self):
        """
        Gemeinsamer Klick-Handler des globalen Footer-Knopfes (Maus-Klick UND
        Enter/Return-Shortcut, siehe _create_footer_bar()). Verzweigt anhand
        des aktuell aktiven Tabs: WEITER_TAB_KEYS (Region/Kontinent, kein
        generate()) springt zum naechsten Tab, alle anderen generieren wie
        bisher ueber _generate_current_tab().
        """
        current_index = self.main_tab_bar.currentIndex()
        if 0 <= current_index < len(self.tab_order) and \
                self.tab_order[current_index] in self.WEITER_TAB_KEYS:
            self._advance_to_next_tab()
        else:
            self._generate_current_tab()

    def _advance_to_next_tab(self):
        """
        Springt vom aktuellen zum naechsten Tab in main_tab_bar (WEITER-Knopf
        auf Region/Kontinent). Ist der aktuelle Tab bereits der letzte, tut
        dies bewusst nichts (statt auf einen ungueltigen Index zu springen) -
        das kann WEITER_TAB_KEYS heute nicht treffen, weil Region und
        Kontinent in tab_configs (_setup_tabs()) vor Terrain stehen und damit
        nie die letzten Tabs sind, aber die Pruefung haelt die Funktion auch
        dann sicher, wenn sich die Tab-Reihenfolge kuenftig aendert.
        """
        current_index = self.main_tab_bar.currentIndex()
        next_index = current_index + 1
        if next_index < self.main_tab_bar.count():
            self.main_tab_bar.setCurrentIndex(next_index)
        else:
            self.logger.warning(
                "WEITER-Knopf: kein naechster Tab nach Index %s vorhanden",
                current_index)

    def _update_footer_button_for_tab(self, tab_name_lower: str):
        """
        Haelt Beschriftung/Farbe des globalen Footer-Knopfes synchron zum
        aktiven Tab - aufgerufen aus _on_tab_changed(). Siehe WEITER_TAB_KEYS
        und _on_footer_button_clicked() fuer die zugehoerige Verzweigung.
        """
        if not self.generate_button:
            return
        if tab_name_lower in self.WEITER_TAB_KEYS:
            self.generate_button.set_label("WEITER", "danger")
        else:
            self.generate_button.set_label("GENERIEREN", "primary")

    def _refresh_footer_progress(self):
        """
        Aktualisiert den Ladebalken direkt aus dem CalculatorDispatcher's
        completed_lod/target_lod (ein Eintrag pro Calculator-Knoten über ALLE
        6 Generatoren, siehe [[project-pipeline-progress-bar-calibration]]).

        Ersetzt die vorherige WeightedProgressCalculator/tab_generation_status-
        Kombination, die strukturell nie 100% erreichen konnte: deren Zähler
        (tab_generation_status) wurde nur EINMAL pro Generator beim finalen
        Ziel-LOD gesetzt (nie für Zwischen-LODs 1..target_lod-1), während der
        Nenner (progress_calculator.total_cost) alle LOD-Stufen gewichtet
        aufsummierte - selbst wenn alle 6 Generatoren fertig waren, blieb der
        Balken bei ~50% stehen. Zusätzlich las der Code ein nie existierendes
        self.toolbar_lod_combo (immer AttributeError->Fallback auf einen
        hartkodierten Default), unabhängig vom tatsächlichen Ziel-LOD der
        laufenden Anfrage.

        Der CalculatorDispatcher (managers/calculator_graph.py) führt
        bereits pro Knoten (Terrain/Geology/Weather/Water/Biome/Settlement,
        alle 34 Knoten) completed_lod/target_lod - ein einfaches Verhältnis
        über alle Knoten hinweg ist exakt "wie viele LOD-Runden von allen
        insgesamt nötigen sind fertig", erreicht 100% erst wenn wirklich JEDER
        Knoten (inkl. Settlement, das als letztes läuft) sein Ziel-LOD
        vollständig erreicht hat, und aktualisiert sich live pro Knoten statt
        nur einmal pro Generator (siehe _on_calculator_status_changed).
        """
        if not self.footer_progress_bar:
            return

        dispatcher = getattr(self.generation_orchestrator, "calculator_dispatcher", None)
        if dispatcher is None:
            return

        total = sum(dispatcher.target_lod.values())
        if total <= 0:
            self.footer_progress_bar.set_progress(0, "Pipeline: 0%", "0 / 0 LOD-Runden")
            return

        done = sum(
            min(dispatcher.completed_lod.get(cid, 0), target)
            for cid, target in dispatcher.target_lod.items()
        )
        percent = int(round(done / total * 100))
        self.footer_progress_bar.set_progress(
            percent, f"Pipeline: {percent}%", f"{done} / {total} LOD-Runden (alle Calculator-Knoten)"
        )

    def _setup_managers(self):
        try:
            # Manager einzeln erstellen und testen
            self.logger.info("Creating DataLODManager...")
            self.data_lod_manager = DataLODManager()

            self.logger.info("Creating ShaderManager...")
            self.shader_manager = ShaderManager()

            self.logger.info("Creating ParameterManager...")
            self.parameter_manager = ParameterManager()

            self.logger.info("Creating NavigationManager...")
            self.navigation_manager = NavigationManager(data_lod_manager=self.data_lod_manager)

            self.logger.info("Creating GenerationOrchestrator...")
            # Denselben ShaderManager wie die 3D-Ansicht übergeben - vorher hatte der
            # Orchestrator gar keinen shader_manager-Parameter, jeder Generator lief
            # dadurch immer auf CPU-Fallback (siehe GPU-Fundament-Fix dieser Session).
            self.generation_orchestrator = GenerationOrchestrator(
                data_lod_manager=self.data_lod_manager, shader_manager=self.shader_manager)

        except Exception as e:
            self.logger.error(f"Manager creation failed: {e}")
            import traceback
            traceback.print_exc()
            return

    def _check_managers(self):
        if self.data_lod_manager is None:
            self.logger.error("DEBUG: MapEditor received None as data_lod_manager!")
        else:
            self.logger.info(f"DEBUG: MapEditor received data_lod_manager: {type(self.data_lod_manager)}")

        if self.parameter_manager is None:
            self.logger.error("DEBUG: MapEditor received None as parameter_manager!")
        else:
            self.logger.info(f"DEBUG: MapEditor received parameter_manager: {type(self.parameter_manager)}")

        if self.shader_manager is None:
            self.logger.error("DEBUG: MapEditor received None as shader_manager!")
        else:
            self.logger.info(f"DEBUG: MapEditor received shader_manager: {type(self.shader_manager)}")

        if self.navigation_manager is None:
            self.logger.error("DEBUG: MapEditor received None as navigation_manager!")
        else:
            self.logger.info(f"DEBUG: MapEditor received navigation_manager: {type(self.navigation_manager)}")

        if self.generation_orchestrator is None:
            self.logger.error("DEBUG: MapEditor received None as generation_orchestrator!")
        else:
            self.logger.info(f"DEBUG: MapEditor received orchestrator: {type(self.generation_orchestrator)}")


    def _create_menu_bar(self):
        """
        Create comprehensive menu bar with all editor functions
        ======================================================

        Builds professional menu structure including File, Generation,
        View, and Help menus with keyboard shortcuts and proper
        organization of functionality.
        """
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')

        file_actions = [
            ("&New World", "Ctrl+N", self._new_world),
            ("&Open World", "Ctrl+O", self._open_world),
            ("&Save World", "Ctrl+S", self._save_world),
            ("separator", None, None),
            ("&Export World", "Ctrl+E", self._export_world),
            ("separator", None, None),
            ("&Return to Main Menu", None, self._return_to_main_menu)
        ]

        self._add_menu_actions(file_menu, file_actions)

        # Generation Menu (only if orchestrator available)
        if self.generation_orchestrator:
            generation_menu = menubar.addMenu('&Generation')

            generation_actions = [
                ("&Regenerate All", "Ctrl+R", self._regenerate_all_generators),
                ("&Stop All Generation", "Ctrl+Shift+S", self._stop_all_generation),
                ("separator", None, None),
            ]

            self._add_menu_actions(generation_menu, generation_actions)

        # View Menu
        view_menu = menubar.addMenu('&View')
        view_actions = [
            ("&Fullscreen", "F11", self._toggle_fullscreen),
            ("separator", None, None),
            ("&Reset Tab Layout", None, self._reset_tab_layout),
            ("&Refresh All Displays", "F5", self._refresh_all_displays)
        ]
        self._add_menu_actions(view_menu, view_actions)

        # Help Menu
        help_menu = menubar.addMenu('&Help')
        help_actions = [
            ("&About MapGenerator", None, self._show_about),
            ("&Keyboard Shortcuts", "F1", self._show_shortcuts),
            ("&Report Issue", None, self._report_issue)
        ]
        self._add_menu_actions(help_menu, help_actions)

    def _add_menu_actions(self, menu: QMenu, actions: list):
        """
        Helper method to add actions to menu with consistent formatting
        ==============================================================

        Args:
            menu: QMenu to add actions to
            actions: List of (name, shortcut, callback) tuples
        """
        for action_data in actions:
            if action_data[0] == "separator":
                menu.addSeparator()
                continue

            name, shortcut, callback = action_data
            action = QAction(name, self)

            if shortcut:
                action.setShortcut(shortcut)
            if callback:
                action.triggered.connect(callback)

            menu.addAction(action)

    def _create_status_bar(self):
        """
        Create comprehensive status bar with generation monitoring
        =========================================================

        Builds status bar showing current tab, active generations,
        memory usage, and overall system status with real-time updates.
        """
        statusbar = self.statusBar()

        # Current tab indicator
        self.current_tab_label = QLabel("Current: Loading...")
        statusbar.addWidget(self.current_tab_label)

        statusbar.addPermanentWidget(QLabel(" | "))

        # Active generations indicator (if orchestrator available)
        if self.generation_orchestrator:
            self.active_generations_label = QLabel("Generations: 0 active")
            statusbar.addPermanentWidget(self.active_generations_label)
            statusbar.addPermanentWidget(QLabel(" | "))

        # Memory usage indicator
        self.memory_label = QLabel("Memory: 0 MB")
        statusbar.addPermanentWidget(self.memory_label)

        statusbar.addPermanentWidget(QLabel(" | "))

        # Overall status indicator
        self.status_indicator = StatusIndicator("System")
        self.status_indicator.set_success("Ready")
        statusbar.addPermanentWidget(self.status_indicator)

    def _setup_tabs(self):
        """
        Initialize all generator tabs with intelligent error handling
        ============================================================

        Creates instances of all available generator tabs with
        comprehensive error handling and appropriate fallbacks
        for missing or failed tab implementations.
        """
        # SCHLUESSEL UND BESCHRIFTUNG SIND GETRENNT - seit 2026-08-05.
        #
        # Bis dahin war die Beschriftung zugleich der Schluessel: `self.tabs`,
        # `tab_order` und der 3D-renderMode lasen alle `tab_name.lower()`. Eine
        # deutsche Beschriftung haette damit jeden dieser Nachschlaege gebrochen
        # - "geologie" findet keinen Generator namens "geology".
        #
        # Jetzt: (schluessel, beschriftung, klasse, ...). Der Schluessel bleibt
        # englisch und unveraendert, die Beschriftung ist frei.
        #
        # Reihenfolge nach Vorgabe des Nutzers vom 2026-08-05 und zugleich nach
        # der Pipeline (siehe CALCULATOR_GRAPH): das Flussnetz kommt direkt nach
        # dem Terrain, weil es dessen Heightmap formt, und die Siedlungen
        # zerfallen in eine globale und eine regionale Ansicht.
        tab_configs = [
            # DER REGIONSREITER STEHT VORN (2026-08-26). Nutzerentwurf:
            # *"als ziel wuerde ich vorschlagen, dass wir am anfang eine
            # regionen-ansicht haben ... dort stellt man die optik jeder
            # region ein. dann gehts in kontinent sicht."* Die Reihenfolge
            # der Reiter IST der Arbeitsablauf, und sie ordnet ihn zugleich
            # nach den Kosten: der Regionsreiter rechnet in 0.1-1.2 s, die
            # Vollkarte in 40 s.
            ("region", "Regionen", RegionTab, REGION_AVAILABLE, region_error),
            ("kontinent", "Kontinent", KontinentTab,
             KONTINENT_AVAILABLE, kontinent_error),
            ("terrain", "Terrain", TerrainTab, TERRAIN_AVAILABLE, terrain_error),
            ("rivers", "Flussnetzwerk", RiverTab, RIVER_AVAILABLE, river_error),
            ("geology", "Geologie", GeologyTab, GEOLOGY_AVAILABLE, geology_error),
            ("erosion", "Erosion", ErosionTab, EROSION_AVAILABLE, erosion_error),
            ("weather", "Wetter", WeatherTab, WEATHER_AVAILABLE, weather_error),
            ("water", "Wasser", WaterTab, WATER_AVAILABLE, water_error),
            ("biome", "Biome", BiomeTab, BIOME_AVAILABLE, biome_error),
            ("settlement", "Siedlungen (Global)", SettlementTab,
             SETTLEMENT_AVAILABLE, settlement_error),
            ("settlement_regional", "Siedlungen (Regional)", SettlementRegionalTab,
             SETTLEMENT_REGIONAL_AVAILABLE, settlement_regional_error),
            ("overview", "Overview", OverviewTab, OVERVIEW_AVAILABLE, overview_error)
        ]

        for tab_key, tab_name, tab_class, available, error_type in tab_configs:
            try:
                self.logger.info(f"DEBUG: Starting creation of {tab_name} tab...")

                if available and tab_class:
                    # Attempt to create real tab instance
                    tab_instance = self._create_tab_instance(tab_class, tab_name)
                    if tab_instance:
                        self._add_successful_tab(tab_key, tab_name, tab_instance)
                        self.logger.info(f"DEBUG: {tab_name} tab completed successfully")
                    else:
                        self._add_error_tab(tab_key, tab_name, "instantiation_failed")
                else:
                    # Create appropriate error tab based on failure type
                    self._add_error_tab(tab_key, tab_name, error_type)

                self.logger.info(f"DEBUG: {tab_name} tab processing finished")

            except Exception as e:
                self.logger.error(f"Unexpected error creating {tab_name} tab: {e}")
                import traceback
                traceback.print_exc()
                self._add_error_tab(tab_key, tab_name, "unexpected_error")

    def _create_tab_instance(self, tab_class: type, tab_name: str) -> Optional[QWidget]:
        """
        Safely create tab instance with comprehensive error handling
        ===========================================================

        Args:
            tab_class: Tab class to instantiate
            tab_name: Name of tab for logging

        Returns:
            Tab instance or None if creation failed
        """
        try:
            self.logger.debug(f"Creating {tab_name} tab instance")

            tab_instance = tab_class(
                data_lod_manager=self.data_lod_manager,
                parameter_manager=self.parameter_manager,
                navigation_manager=self.navigation_manager,
                shader_manager=self.shader_manager,
                generation_orchestrator=self.generation_orchestrator
            )

            self.logger.info(f"Successfully created {tab_name} tab")
            return tab_instance

        except Exception as e:
            self.logger.error(f"Failed to instantiate {tab_name} tab: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_successful_tab(self, tab_key: str, tab_name: str, tab_instance: QWidget):
        """
        Add successfully created tab to interface
        ========================================

        Args:
            tab_name: Name of the tab
            tab_instance: Created tab widget instance
        """

        self.logger.info(f"Adding {tab_name} to shell...")

        try:
            index = self.main_tab_bar.addTab(tab_name)
            self.viewport_stack.addWidget(tab_instance.viewport_widget)
            self.parameter_stack.addWidget(tab_instance.parameter_widget)
            self.statistics_stack.addWidget(tab_instance.statistics_widget)
            self.tab_order.append(tab_key)
            self.logger.info(f"Tab added to shell at index: {index}")

            self.tabs[tab_key] = tab_instance
            self.tab_generation_status[tab_key] = {}

            if hasattr(tab_instance, 'view_switched'):
                tab_instance.view_switched.connect(self._on_tab_view_switched)

            # DEBUG: Signal-Verbindung prüfen
            self.logger.info(f"DEBUG: Checking for generation_completed signal...")
            if hasattr(tab_instance, 'generation_completed'):
                self.logger.info(f"DEBUG: generation_completed signal found, connecting...")
                tab_instance.generation_completed.connect(self._on_tab_generation_completed)
                self.logger.info(f"DEBUG: Signal connected successfully")
            else:
                self.logger.info(f"DEBUG: No generation_completed signal found")

        except Exception as e:
            self.logger.error(f"DEBUG: Error in _add_successful_tab: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _add_error_tab(self, tab_key: str, tab_name: str, error_type: str):
        """
        Create and add error tab with appropriate messaging
        ==================================================

        Args:
            tab_name: Name of the failed tab
            error_type: Type of error that occurred
        """
        error_tab = self._create_error_tab(tab_name, error_type)

        self.main_tab_bar.addTab(tab_name)
        self.viewport_stack.addWidget(error_tab)
        self.parameter_stack.addWidget(self._create_placeholder_widget("Not available"))
        self.statistics_stack.addWidget(self._create_placeholder_widget("Not available"))
        self.tab_order.append(tab_key)

        self.tabs[tab_key] = error_tab

    def _create_placeholder_widget(self, text: str) -> QWidget:
        """Kleiner Platzhalter für Spalte 3, wenn ein Tab nicht geladen werden konnte."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #7f8c8d; padding: 20px;")
        layout.addWidget(label)
        widget.setLayout(layout)
        return widget

    def _create_error_tab(self, tab_name: str, error_type: str) -> QWidget:
        """
        Create informative error tab based on failure type
        =================================================

        Args:
            tab_name: Name of the failed tab
            error_type: Specific type of failure

        Returns:
            QWidget with appropriate error messaging
        """
        error_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Error-specific messaging
        if error_type == "import_failed":
            title = f"{tab_name} Generator"
            message = f"Tab implementation not found\n\nThe {tab_name.lower()}_tab.py file could not be loaded.\nThis feature will be available in a future version."
            icon = "⚠️"
            color = "#f39c12"  # Orange

        elif error_type == "instantiation_failed":
            title = f"{tab_name} Generator"
            message = f"Connection to tab files failed\n\nThe {tab_name} tab could not be initialized.\nPlease check the console for detailed error information."
            icon = "❌"
            color = "#e74c3c"  # Red

        elif error_type == "class_missing":
            title = f"{tab_name} Generator"
            message = f"Tab class definition missing\n\nThe {tab_name}Tab class was not found in the module.\nPlease verify the implementation."
            icon = "❌"
            color = "#e74c3c"  # Red

        else:  # unexpected_error or unknown
            title = f"{tab_name} Generator"
            message = f"Unexpected error occurred\n\nAn unexpected error prevented the {tab_name} tab from loading.\nPlease check logs for details."
            icon = "❌"
            color = "#e74c3c"  # Red

        # Title with icon
        title_label = QLabel(f"{icon} {title}")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color}; margin: 20px;")
        layout.addWidget(title_label)

        # Error message
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("font-size: 14px; color: #7f8c8d; line-height: 1.6; margin: 20px;")
        layout.addWidget(message_label)

        # Action button (disabled for errors)
        if error_type == "import_failed":
            button_text = f"{tab_name} (Coming Soon)"
            button_style = "secondary"
        else:
            button_text = f"Retry {tab_name} (Disabled)"
            button_style = "danger"

        action_button = BaseButton(button_text, button_style)
        action_button.setEnabled(False)
        layout.addWidget(action_button)

        error_widget.setLayout(layout)
        return error_widget

    def _setup_signals(self):
        """
        Connect all signal-slot relationships
        ====================================

        Establishes communication between tab widget, navigation
        manager, and data manager for coordinated operation.
        """
        # Main tab bar signals
        self.main_tab_bar.currentChanged.connect(self._on_tab_changed)

        # Navigation manager signals
        if self.navigation_manager:
            self.navigation_manager.tab_changed.connect(self._on_navigation_requested)

        # Data manager signals
        if self.data_lod_manager:
            self.data_lod_manager.data_updated.connect(self._on_data_updated)

        # Orchestrator signals - treiben die Pipeline-Status-Spalte und den
        # Footer-Fortschrittsbalken an. Diese Verbindung fehlte bisher komplett
        # (nur der Disconnect-Cleanup in _disconnect_orchestrator_signals()
        # existierte), wodurch die Pipeline-Status-Spalte dauerhaft auf
        # "Unknown" stehen blieb, egal was im Hintergrund generiert wurde.
        if self.generation_orchestrator:
            self.generation_orchestrator.generation_started.connect(self._on_generation_started)
            self.generation_orchestrator.generation_completed.connect(self._on_generation_completed)
            self.generation_orchestrator.generation_progress.connect(self._on_generation_progress)
            self.generation_orchestrator.batch_generation_completed.connect(self._on_batch_generation_completed)
            self.generation_orchestrator.dependency_invalidated.connect(self._on_dependency_invalidated)
            self.generation_orchestrator.calculator_status_changed.connect(self._on_calculator_status_changed)

    def activate_tab(self, tab_name: str) -> bool:
        """
        Programmatically activate specified tab
        ======================================

        Args:
            tab_name: Name of tab to activate

        Returns:
            True if tab was successfully activated
        """
        tab_name_lower = tab_name.lower()

        if tab_name_lower in self.tab_order:
            index = self.tab_order.index(tab_name_lower)
            self.main_tab_bar.setCurrentIndex(index)
            self.logger.info(f"Activated {tab_name} tab")
            return True

        self.logger.warning(f"Tab {tab_name} not found for activation")
        return False

    def navigate_to_tab(self, tab_name: str) -> bool:
        """
        Navigate to specified tab (NavigationManager interface)
        ======================================================

        Args:
            tab_name: Target tab name

        Returns:
            True if navigation successful
        """
        return self.activate_tab(tab_name)

    # Signal Handlers

    @pyqtSlot(int)
    def _on_tab_changed(self, index: int):
        """
        Handle tab change events
        =======================

        Args:
            index: Index of newly active tab
        """
        if 0 <= index < len(self.tab_order):
            tab_name_lower = self.tab_order[index]
            tab_text = self.main_tab_bar.tabText(index)

            # Spalte 2/3 synchron zum Haupt-Tab umschalten. Spalte 1
            # (Pipeline-Status) bleibt unabhängig davon unverändert sichtbar.
            self.viewport_stack.setCurrentIndex(index)
            self.parameter_stack.setCurrentIndex(index)
            self.statistics_stack.setCurrentIndex(index)

            self.current_tab_label.setText(f"Current: {tab_text}")

            # Footer-Knopf (GENERIEREN/WEITER) auf den neu aktiven Tab
            # umschalten - siehe WEITER_TAB_KEYS/_update_footer_button_for_tab().
            self._update_footer_button_for_tab(tab_name_lower)

            # Update navigation manager
            if self.navigation_manager:
                self.navigation_manager.current_tab = tab_name_lower

            tab_instance = self.tabs.get(tab_name_lower)

            self._update_cross_section_visibility(tab_name_lower)

            # Globale Contour-Checkbox/Cross-Section-Regler gelten tab-
            # übergreifend deklariert - beim Wechsel auf den neu aktiven Tab
            # anwenden, damit dessen Display den aktuellen globalen Zustand
            # übernimmt.
            self._apply_global_overlays_to_active_tab()

            # Globale 2D/3D-Präferenz auf den neu aktiven Tab anwenden, falls er
            # noch nicht im gewünschten Modus ist (siehe global_view_mode oben).
            if tab_instance and hasattr(tab_instance, 'current_view') and hasattr(tab_instance, 'switch_view'):
                if tab_instance.current_view != self.global_view_mode:
                    tab_instance.switch_view(self.global_view_mode)

            # Erzwingt einen Display-Refresh mit dem aktuell gewählten Render-Modus.
            # Ohne das bleibt der Viewport leer, bis der Nutzer den Modus manuell
            # umschaltet - der Default-Radio (z.B. "Height") feuert beim Erstellen
            # kein toggled-Signal, weil setChecked(True) vor dem connect() passiert.
            # Per QTimer.singleShot(0, ...) statt synchron aufgerufen (Nutzer-
            # Beobachtung: im 3D-Modus zeigt ein Tab-Wechsel weiterhin die alten
            # Daten, bis man zusätzlich manuell ein Anzeige-Radio anklickt) -
            # setCurrentIndex()/switch_view() direkt darüber ändern den sichtbaren
            # Widget-Stack bzw. bauen ggf. den 3D-Viewport neu auf; ein SOFORT im
            # selben Slot-Durchlauf folgender update_heightmap()/makeCurrent()-
            # Aufruf kann auf einem OpenGL-Widget landen, dessen Sichtbarkeits-/
            # Kontext-Wechsel durch Qt noch nicht vollständig verarbeitet wurde.
            # Ein Event-Loop-Tick Verzögerung (0ms) reicht, damit das zuverlässig
            # nach dem tatsächlichen Umschalten läuft - gleiches Muster wie
            # QTimer.singleShot(0, self._auto_start_generation) oben.
            if tab_instance and hasattr(tab_instance, 'update_display_mode'):
                QTimer.singleShot(0, tab_instance.update_display_mode)

            self.logger.debug(f"Tab changed to: {tab_text}")

    @pyqtSlot(str)
    def _on_tab_view_switched(self, view_type: str):
        """
        Hält die globale 2D/3D-Präferenz aktuell, sobald der Nutzer auf
        IRGENDEINEM Tab manuell zwischen 2D/3D umschaltet (BaseMapTab.
        view_switched-Signal, siehe _add_successful_tab()). _on_tab_changed()
        wendet global_view_mode dann auf den jeweils neu aktivierten Tab an.
        """
        self.global_view_mode = view_type
        index = self.main_tab_bar.currentIndex()
        if 0 <= index < len(self.tab_order):
            self._update_cross_section_visibility(self.tab_order[index])

    def _update_cross_section_visibility(self, tab_name_lower: str):
        """
        Cross-Section-Regler (Shell-Zeile Achse/Position + Geology-Tabs
        eigenes "Cross-Section"-Radio) sind nur im 2D-Modus des Geology-Tabs
        sinnvoll - ein Schnitt lässt sich im 3D-Modus nicht darstellen (Nutzer-
        Vorgabe). Deckt sowohl Tab-Wechsel (_on_tab_changed) als auch
        manuelles 2D/3D-Umschalten innerhalb des bereits aktiven Geology-Tabs
        (_on_tab_view_switched) ab.
        """
        is_geology = tab_name_lower == "geology"
        visible = is_geology and self.global_view_mode == "2d"
        for widget in self.cross_section_widgets:
            widget.setVisible(visible)

        geology_tab = self.tabs.get("geology")
        if geology_tab and hasattr(geology_tab, "cross_section_mode_radio"):
            geology_tab.cross_section_mode_radio.setVisible(visible)
            # Cross-Section kann im 3D-Modus nicht dargestellt werden - beim
            # Wechsel dorthin automatisch auf "Height" zurückfallen, statt
            # eine tote Auswahl (Radio unsichtbar, aber weiterhin aktiv) zu
            # hinterlassen.
            if not visible and getattr(geology_tab, "current_display_mode", None) == "cross_section":
                height_button = geology_tab.display_mode_group.button(0)
                if height_button:
                    height_button.setChecked(True)

    def _get_active_tab_instance(self):
        """Liefert die BaseMapTab-Instanz des aktuell im main_tab_bar aktiven Tabs."""
        index = self.main_tab_bar.currentIndex()
        if 0 <= index < len(self.tab_order):
            return self.tabs.get(self.tab_order[index])
        return None

    def _apply_global_overlays_to_active_tab(self):
        """Wendet den aktuellen Zustand der globalen Checkboxen/Regler auf den aktiven Tab an."""
        tab_instance = self._get_active_tab_instance()
        if not tab_instance:
            return
        if hasattr(tab_instance, 'set_contour_overlay'):
            tab_instance.set_contour_overlay(self.contour_checkbox.isChecked())
        if hasattr(tab_instance, 'set_cross_section_axis'):
            tab_instance.set_cross_section_axis("x" if self.cross_section_x_radio.isChecked() else "y")
        if hasattr(tab_instance, 'set_cross_section_position'):
            tab_instance.set_cross_section_position(self.cross_section_position_slider.getValue())

    @pyqtSlot(bool)
    def _on_global_contour_toggled(self, checked: bool):
        """Globales Contour-Lines-Toggle (Spalte 2) - wirkt auf den aktiven Tab."""
        tab_instance = self._get_active_tab_instance()
        if tab_instance and hasattr(tab_instance, 'set_contour_overlay'):
            tab_instance.set_contour_overlay(checked)

    def _on_global_cross_section_axis_changed(self, axis: str, checked: bool):
        """Cross-Section-Achsen-Radio (Spalte 2, nur bei aktivem Geology-Tab
        sichtbar) - wirkt auf den aktiven Tab (immer Geology, siehe
        _on_tab_changed())."""
        if not checked:
            return
        tab_instance = self._get_active_tab_instance()
        if tab_instance and hasattr(tab_instance, 'set_cross_section_axis'):
            tab_instance.set_cross_section_axis(axis)

    def _on_global_cross_section_position_changed(self, value: float):
        """Cross-Section-Positions-Slider (Spalte 2, nur bei aktivem Geology-
        Tab sichtbar) - wirkt auf den aktiven Tab."""
        tab_instance = self._get_active_tab_instance()
        if tab_instance and hasattr(tab_instance, 'set_cross_section_position'):
            tab_instance.set_cross_section_position(value)

    @pyqtSlot(str, str)
    def _on_navigation_requested(self, from_tab: str, to_tab: str):
        """Handle navigation requests from NavigationManager"""
        self.navigate_to_tab(to_tab)

    @pyqtSlot(str, str)
    def _on_data_updated(self, generator_type: str, data_key: str):
        """Handle data update notifications"""
        self.logger.debug(f"Data updated: {generator_type}.{data_key}")

    # Generation Event Handlers
    #
    # Signaturen richten sich exakt nach GenerationOrchestrator's echten
    # pyqtSignal-Deklarationen (managers/generation_orchestrator.py):
    #   generation_started    = pyqtSignal(str, int)   # (generator_type, lod_level)
    #   generation_completed  = pyqtSignal(str, dict)  # (request_id, result_data)
    #   generation_progress   = pyqtSignal(int, str)   # (progress, message) - global, kein generator_type
    # Vorherige Versionen dieser Handler hatten davon abweichende Signaturen
    # UND waren nie mit den Orchestrator-Signalen verbunden (siehe
    # _setup_signals()) - die Pipeline-Status-Spalte blieb dadurch dauerhaft
    # auf "Unknown" stehen.

    # --- Tab-Zustandsanzeige -------------------------------------------------
    #
    # Ersatz fuer die grobe LOD-Vorschau: statt dass alle Generatoren
    # gleichzeitig ein unscharfes Bild zeigen, faerbt sich ein Reiter ein,
    # sobald SEIN Generator fertig ist. Man kann sich dort umsehen, waehrend
    # der naechste rechnet - und sieht, dass es sich lohnt hinzuschauen.
    #
    # Die Farbe sitzt auf dem Reiter-Text (QTabBar.setTabTextColor) statt in
    # einem Stylesheet: ein Stylesheet auf der QTabBar wuerde das komplette
    # Aussehen der Leiste uebernehmen und die vorhandene Gestaltung
    # ueberschreiben, die Textfarbe wirkt gezielt auf genau einen Reiter.
    TAB_STATE_COLOURS = {
        "pending": None,          # Vorgabefarbe des Themes - noch nichts gerechnet
        "running": "#c8922b",     # rechnet gerade
        "ready": "#5fb96b",       # fertig, es gibt etwas zu sehen
        "stale": "#8a8a8a",       # war fertig, ist durch eine Aenderung ungueltig
        "failed": "#c4564b",
    }

    def _set_tab_state(self, generator_type: str, state: str):
        """
        Funktionsweise: Faerbt den Reiter eines Generators nach seinem Zustand
        Parameter: generator_type (klein geschrieben), state - Schluessel aus
            TAB_STATE_COLOURS
        """
        if not generator_type or self.main_tab_bar is None:
            return
        if generator_type not in self.tab_order:
            return

        self.tab_states[generator_type] = state
        index = self.tab_order.index(generator_type)
        colour = self.TAB_STATE_COLOURS.get(state)
        # QColor() ohne Argument = "ungueltig" und bedeutet fuer
        # setTabTextColor ausdruecklich "nimm die Vorgabe des Themes".
        self.main_tab_bar.setTabTextColor(
            index, QColor(colour) if colour else QColor())

    def _mark_all_tabs_pending(self):
        """Setzt alle Generator-Reiter auf 'noch nichts gerechnet'.

        Wird beim Oeffnen des Editors aufgerufen, seit dort nicht mehr
        automatisch generiert wird - sonst saehen die Reiter aus, als waere
        bereits etwas fertig.
        """
        for generator_enum in GeneratorType:
            self._set_tab_state(generator_enum.value, "pending")

    @pyqtSlot(str, int)
    def _on_generation_started(self, generator_type: str, lod_level: int):
        """Handle generation start events (ein LOD-Level beginnt)"""
        if not generator_type:
            return

        self._set_tab_state(generator_type, "running")

        generation_key = f"{generator_type}_{lod_level}"
        self.active_generations.add(generation_key)

        if hasattr(self, 'active_generations_label'):
            self.active_generations_label.setText(f"Generations: {len(self.active_generations)} active")

        self.status_indicator.set_warning(f"Generating {generator_type} LOD {lod_level}")
        self.logger.debug(f"Generation started: {generation_key}")

    @pyqtSlot(str, dict)
    def _on_generation_completed(self, request_id: str, result_data: dict):
        """
        Handle final generation completion events (komplette Target-LOD-
        Progression eines Requests abgeschlossen, nicht pro Zwischen-LOD).
        """
        generator_type = result_data.get("generator_type", "")
        lod_level = result_data.get("lod_level")
        success = result_data.get("success", False)

        if not generator_type:
            return

        self._set_tab_state(generator_type, "ready" if success else "failed")

        generation_key = f"{generator_type}_{lod_level}"
        self.active_generations.discard(generation_key)

        if generator_type in self.tab_generation_status and lod_level is not None:
            self.tab_generation_status[generator_type][lod_level] = "success" if success else "failed"

        if hasattr(self, 'active_generations_label'):
            self.active_generations_label.setText(f"Generations: {len(self.active_generations)} active")

        if success:
            if len(self.active_generations) == 0:
                self.status_indicator.set_success("All generations complete")
            else:
                self.status_indicator.set_success(f"{generator_type} LOD {lod_level} complete")
        else:
            self.status_indicator.set_error(f"{generator_type} LOD {lod_level} failed")

        self._refresh_footer_progress()
        self.logger.info(f"Generation completed: {generation_key} success={success}")

    @pyqtSlot(int, str)
    def _on_generation_progress(self, progress_percent: int, message: str):
        """Handle generation progress updates (global, ohne generator_type/lod_level)"""
        self.status_indicator.set_warning(f"{message} ({progress_percent}%)")

    @pyqtSlot(dict)
    def _on_calculator_status_changed(self, snapshot: dict):
        """
        Treibt die PipelineStatusPanel-Spalte granular über alle 34 Calculator-
        Knoten an (siehe GenerationOrchestrator.calculator_status_changed).
        Ersetzt die bisherigen verstreuten set_calculating()/set_finished()/
        set_failed()/set_queued()-Aufrufe an einzelnen Stellen (die nur die
        6 Generatoren kannten und ein neues, höheres Ziel-LOD nicht sofort
        sichtbar machten) durch einen einzigen reaktiven Snapshot-Apply.
        """
        if self.pipeline_status_panel:
            self.pipeline_status_panel.apply_snapshot(snapshot)
        self._refresh_footer_progress()

    @pyqtSlot(bool, str)
    def _on_batch_generation_completed(self, success: bool, summary_message: str):
        """Handle batch generation completion"""
        if success:
            self.status_indicator.set_success(f"Batch complete: {summary_message}")
        else:
            self.status_indicator.set_error(f"Batch failed: {summary_message}")

    @pyqtSlot(str, list)
    def _on_dependency_invalidated(self, generator_type: str, affected_generators: list):
        """
        Handle dependency invalidation events
        ====================================

        Args:
            generator_type: Generator that triggered invalidation
            affected_generators: List of affected generator types
        """
        if not generator_type or not isinstance(affected_generators, list):
            return

        # Notify affected tabs
        for affected_type in affected_generators:
            # Was invalidiert wurde, ist nicht mehr aktuell - der Reiter darf
            # nicht weiter als "fertig" leuchten. Ein Reiter, der noch nie
            # gerechnet hat, bleibt auf "pending" statt auf "veraltet"
            # zurueckzufallen.
            if self.tab_states.get(affected_type) in ("ready", "running", "failed"):
                self._set_tab_state(affected_type, "stale")

            if affected_type in self.tabs:
                tab_instance = self.tabs[affected_type]
                if hasattr(tab_instance, 'on_dependency_invalidated'):
                    tab_instance.on_dependency_invalidated(generator_type)

        self.logger.info(f"Dependencies invalidated: {generator_type} → {affected_generators}")

    @pyqtSlot(str, bool)
    def _on_tab_generation_completed(self, generator_type: str, success: bool):
        """Handle generation completion from individual tabs"""
        self.logger.debug(f"Tab generation completed: {generator_type} success={success}")

    # Status and Resource Management

    def _update_status(self):
        """
        Periodic status update for monitoring and optimization
        ====================================================

        Updates memory usage display and performs maintenance
        tasks. Optimized to only run when necessary.
        """
        try:
            # Update memory usage
            if self.data_lod_manager:
                try:
                    memory_usage = self.data_lod_manager.get_memory_usage()
                    total_memory = sum(memory_usage.values()) if memory_usage else 0
                    self.memory_label.setText(f"Memory: {total_memory:.1f} MB")
                except Exception as e:
                    self.memory_label.setText("Memory: Error")
                    self.logger.debug(f"Memory usage check failed: {e}")

            # Update active generations count
            if hasattr(self, 'active_generations_label'):
                count = len(self.active_generations)
                self.active_generations_label.setText(f"Generations: {count} active")

        except Exception as e:
            self.logger.warning(f"Status update failed: {e}")

    # Menu Action Handlers

    def _new_world(self):
        """Create new world with user confirmation"""
        reply = QMessageBox.question(
            self, "New World",
            "This will clear all current data and reset all generators. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Clear data + Orchestrator-LOD-Fortschritt fuer einen echten
                # Neustart (siehe _clear_and_reset_all_generators()-Docstring -
                # bisher fehlte hier reset_lod_status(), wodurch die naechste
                # Generierung mit unveraenderten Parametern faelschlich auf
                # dem ALTEN, eigentlich geloeschten LOD-Stand aufbaute).
                self._clear_and_reset_all_generators()

                # Reset all tabs
                for tab_name, tab_instance in self.tabs.items():
                    if hasattr(tab_instance, 'reset_state'):
                        tab_instance.reset_state()

                # Clear generation tracking
                self.active_generations.clear()
                self.tab_generation_status.clear()

                self.status_indicator.set_success("New world created")
                self.logger.info("New world created successfully")

            except Exception as e:
                self.logger.error(f"New world creation failed: {e}")
                QMessageBox.critical(self, "Error", f"Failed to create new world: {str(e)}")

    def _open_world(self):
        """
        Laedt eine zuvor mit welt_backen() geschriebene Welt (Ticket #40).

        Eine Welt ist laut core/welt_io.py ein ganzer ORDNER (welt_manifest.json
        + zustand/ + godot/ + vorschau/), kein einzelner Dateiname - deshalb
        Ordnerauswahl statt Dateiauswahl, anders als der bisherige Attrappen-
        Dialog mit JSON-Dateifilter.

        Der Ordner wird VOR dem eigentlichen welt_laden() knapp auf ein
        vorhandenes welt_manifest.json geprueft. Grund: welt_laden() selbst
        schreibt Kategorie fuer Kategorie in data_lod_manager und wuerde bei
        einem Fehler in einer spaeteren Kategorie den zuvor schon geleerten
        Zustand nicht zuruecksetzen - ein falsch gewaehlter Ordner (Tippfehler,
        kein Welt-Ordner) soll deshalb NICHT erst den aktuellen Arbeitsstand
        wegwerfen, bevor der Fehler bemerkt wird.
        """
        ordner = QFileDialog.getExistingDirectory(
            self, "Welt oeffnen - Ordner waehlen", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if not ordner:
            return

        manifest_pfad = os.path.join(ordner, "welt_manifest.json")
        if not os.path.isfile(manifest_pfad):
            QMessageBox.critical(
                self, "Welt oeffnen fehlgeschlagen",
                f"'{ordner}' enthaelt kein welt_manifest.json - das ist "
                f"kein mit 'Welt speichern' erzeugter Welt-Ordner."
            )
            return

        try:
            # Alten Zustand + Orchestrator-LOD-Fortschritt wegwerfen, BEVOR
            # die geladene Welt geschrieben wird - sonst mischen sich Reste
            # der vorherigen Sitzung (z.B. eine Kategorie, die die geladene
            # Welt gar nicht enthaelt) mit den frisch geladenen Daten. Siehe
            # _clear_and_reset_all_generators()-Docstring.
            self._clear_and_reset_all_generators()
            welt_laden(ordner, self.data_lod_manager, self.parameter_manager)
        except WeltLadenFehler as e:
            QMessageBox.critical(self, "Welt oeffnen fehlgeschlagen", str(e))
            self.logger.error(f"World load failed: {e}")
            return
        except Exception as e:
            QMessageBox.critical(
                self, "Welt oeffnen fehlgeschlagen", f"Unerwarteter Fehler: {e}")
            self.logger.error(f"World load failed unexpectedly: {e}")
            return

        # Geladenen Zustand SICHTBAR machen: jeder Tab holt sich seine Daten
        # ueber update_display_mode() aus dem DataLODManager zurueck - dieselbe
        # Methode fuer 2D UND 3D (siehe STEHENDE REGEL in CLAUDE.md), kein
        # Sonderfall fuer den Lade-Weg noetig. _clear_and_reset_all_generators()
        # hat den Display-Cache bereits geleert, der Dirty-Check erkennt die
        # neuen Daten also zuverlaessig als Aenderung.
        for tab_instance in self.tabs.values():
            if hasattr(tab_instance, 'update_display_mode'):
                try:
                    tab_instance.update_display_mode()
                except Exception as e:
                    self.logger.debug(f"Display refresh after world load failed for a tab: {e}")

        self.active_generations.clear()
        self.tab_generation_status.clear()

        self.status_indicator.set_success("Welt geladen")
        self.logger.info(f"World loaded from: {ordner}")
        QMessageBox.information(
            self, "Welt geoeffnet", f"Welt erfolgreich geladen aus:\n{ordner}")

    def _save_world(self):
        """
        Speichert die aktuelle Welt nach welt_backen() (Ticket #40).

        Zielordner statt Zieldateiname: welt_backen() schreibt einen ganzen
        Ordner (welt_manifest.json, zustand/, godot/, vorschau/ - siehe
        core/welt_io.py-Moduldocstring), keine einzelne Datei. Der native
        Ordnerdialog erlaubt es dem Nutzer, darin auch einen neuen Ordner
        anzulegen.
        """
        ordner = QFileDialog.getExistingDirectory(
            self, "Welt speichern - Zielordner waehlen", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if not ordner:
            return

        try:
            welt_backen(ordner, self.data_lod_manager, self.parameter_manager)
        except WeltBackenFehler as e:
            QMessageBox.critical(self, "Welt speichern fehlgeschlagen", str(e))
            self.logger.error(f"World save failed: {e}")
            return
        except Exception as e:
            QMessageBox.critical(
                self, "Welt speichern fehlgeschlagen", f"Unerwarteter Fehler: {e}")
            self.logger.error(f"World save failed unexpectedly: {e}")
            return

        self.status_indicator.set_success("Welt gespeichert")
        self.logger.info(f"World saved to: {ordner}")
        QMessageBox.information(
            self, "Welt gespeichert", f"Welt erfolgreich gespeichert nach:\n{ordner}")

    def _export_world(self):
        """
        Exportiert die Welt als Godot/Terrain3D-Layer (Ticket #40).

        Verdrahtet auf denselben export_all_layers()-Pfad (gui/utils/
        map_export.py), den sowohl welt_backen() fuer seinen godot/-
        Unterordner als auch OverviewTab.export_layers_to_disk() bereits
        benutzen - siehe core/welt_io.py-Moduldocstring: es soll KEINEN
        zweiten, eigenen Export-Pfad geben. Ordner + Dateinamen-Praefix
        werden hier abgefragt, weil ein Menuepunkt (anders als der
        dauerhafte Reiter in Overview) keinen eigenen Platz fuer ein
        Formularfeld hat.
        """
        ordner = QFileDialog.getExistingDirectory(
            self, "Welt exportieren - Zielordner waehlen", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if not ordner:
            return

        vorschlag = "Mapseed_xxxxxx"
        try:
            if self.parameter_manager:
                seed = self.parameter_manager.get_tab_parameters("terrain").get("map_seed")
                if seed is not None:
                    vorschlag = f"Mapseed_{seed}"
        except Exception as e:
            self.logger.debug(f"Dateinamen-Vorschlag fuer Export nicht ermittelbar: {e}")

        filename_prefix, ok = QInputDialog.getText(
            self, "Welt exportieren", "Name des Export-Unterordners:", text=vorschlag)
        if not ok or not filename_prefix.strip():
            return

        try:
            from gui.utils.map_export import export_all_layers
            success, message, output_dir = export_all_layers(
                self.data_lod_manager, self.parameter_manager, ordner, filename_prefix.strip())
        except Exception as e:
            QMessageBox.critical(self, "Export fehlgeschlagen", f"Unerwarteter Fehler: {e}")
            self.logger.error(f"World export failed unexpectedly: {e}")
            return

        if success:
            self.status_indicator.set_success("Welt exportiert")
            self.logger.info(f"World exported to: {output_dir}")
            QMessageBox.information(
                self, "Export abgeschlossen", f"{message}\n\nZiel: {output_dir}")
        else:
            self.logger.warning(f"World export incomplete: {message}")
            QMessageBox.warning(self, "Export unvollstaendig", message)

    def _export_current_png(self):
        """Export current tab view as PNG"""
        current_index = self.main_tab_bar.currentIndex()
        current_tab = None
        if 0 <= current_index < len(self.tab_order):
            current_tab = self.tabs.get(self.tab_order[current_index])

        if hasattr(current_tab, 'export_current_view'):
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Current View", "map_view.png",
                "PNG Files (*.png);;All Files (*)"
            )

            if filename:
                try:
                    success = current_tab.export_current_view(filename)
                    if success:
                        QMessageBox.information(self, "Export Complete", f"View exported successfully to {filename}")
                    else:
                        QMessageBox.warning(self, "Export Failed", "Failed to export current view")
                except Exception as e:
                    QMessageBox.critical(self, "Export Error", f"Export failed: {str(e)}")
        else:
            QMessageBox.information(
                self, "Export Not Available",
                "Current tab does not support view export"
            )

    def _return_to_main_menu(self):
        try:
            # Clear data + Orchestrator-LOD-Fortschritt (siehe
            # _clear_and_reset_all_generators()-Docstring).
            self._clear_and_reset_all_generators()

            # Reset all tabs
            for tab_name, tab_instance in self.tabs.items():
                if hasattr(tab_instance, 'reset_state'):
                    tab_instance.reset_state()

            # Clear generation tracking
            self.active_generations.clear()
            self.tab_generation_status.clear()

            self.status_indicator.set_success("New world created")
            self.logger.info("Returning to Main Menu...")

            self.close()

            if self._main_menu:
                self._main_menu.show()

            self.logger.info("Success returning to Main Menu.")

        except Exception as e:
            self.logger.error(f"Failed to return to Main Menu: {e}")


    # Generation Control Methods

    def _auto_start_generation(self):
        """
        Tracker #16 Task 8: startet automatisch alle 6 Generatoren mit ihren
        Default-Parametern, sobald der Map Editor öffnet. Geht bewusst NICHT
        über tab_instance.generate() - das würde bei Geology/Weather/Water/
        Biome/Settlement sofort an check_input_dependencies() scheitern, da
        Terrain zu diesem Zeitpunkt (alle 6 Requests praktisch gleichzeitig)
        noch nichts geliefert hat. Stattdessen direkter Aufruf von
        request_generation() wie schon in _regenerate_all_generators(), nur
        mit den echten Default-Parametern jedes Tabs statt einem leeren dict.
        Der globale CalculatorDispatcher (siehe generation_orchestrator.py,
        Tracker #16) sorgt dafür, dass kein Generator seinen tatsächlichen
        Abhängigkeiten vorauseilt, auch wenn alle 6 Requests gleichzeitig
        gestellt werden - echtes Lockstep auf Calculator-Knoten-Ebene, nicht
        nur auf Generator-Ebene.
        """
        if not self.generation_orchestrator:
            return

        # Liste AUS GeneratorType abgeleitet, nicht fest verdrahtet.
        #
        # Sie war es bis 2026-07-28, und der neue Erosion-Generator fehlte
        # darin. Folge: erosion.hydraulic blieb auf Ziel-LOD 0 und wurde nie
        # ausgefuehrt - und weil weather.temperature seit dem Umbau darauf
        # wartet, stand die GESAMTE Pipeline nach Terrain und Geology still
        # (beobachtet als "13 / 111 LOD-Runden, 12%, haengt"). Ein fehlender
        # Eintrag in einer handgepflegten Liste darf keinen Deadlock
        # erzeugen koennen.
        for generator_enum in GeneratorType:
            generator_type = generator_enum.value
            tab_instance = self.tabs.get(generator_type)
            if not tab_instance or not hasattr(tab_instance, 'get_current_parameters'):
                continue

            try:
                parameters = tab_instance.get_current_parameters()
            except Exception as e:
                self.logger.warning(f"Auto-Start: Default-Parameter für {generator_type} nicht lesbar: {e}")
                parameters = {}

            self.generation_orchestrator.request_generation(
                generator_type=generator_type,
                parameters=parameters,
                target_lod=None,
                source_tab="auto_start",
            )

        self.logger.info("Auto-Start: alle %d Generatoren mit Default-Parametern angefragt",
                         len(GeneratorType))

    def _prime_all_generator_parameters(self):
        """
        Reicht jedem Generator die aktuellen Parameter SEINES Tabs durch, ohne
        eine Generierung anzufragen.

        Die Liste kommt aus GeneratorType, nicht von Hand - dieselbe Regel wie
        beim Auto-Start und beim Dependency-Tree. Ein vergessener Eintrag hier
        faellt sonst erst auf, wenn der betroffene Generator mitten im Lauf
        ueber einen fehlenden Parameter stolpert.
        """
        if not self.generation_orchestrator:
            return

        for generator_enum in GeneratorType:
            generator_type = generator_enum.value
            tab_instance = self.tabs.get(generator_type)
            if not tab_instance or not hasattr(tab_instance, 'get_current_parameters'):
                continue
            try:
                parameters = tab_instance.get_current_parameters()
            except Exception as error:  # noqa: BLE001 - Tab-Fehler darf den Lauf nicht kippen
                self.logger.warning(
                    "Parameter von %s nicht lesbar: %s", generator_type, error)
                continue
            self.generation_orchestrator.prime_generator_parameters(
                generator_type, parameters)

    def _generate_current_tab(self):
        """Generate content for currently active tab"""
        if not self.generation_orchestrator:
            QMessageBox.warning(self, "Generation Unavailable", "No GenerationOrchestrator available")
            return

        # ALLE Generatoren mit den aktuellen Tab-Parametern versorgen, bevor
        # irgendetwas laeuft. Ein Klick im Terrain-Tab fragt nur Terrain an,
        # zieht aber ueber die Abhaengigkeiten die gesamte Pipeline nach - und
        # deren Generatoren bekommen von request_generation() keine Parameter.
        #
        # Solange der Auto-Start beim Programmstart alle sieben einzeln
        # anfragte, war das gedeckt. Ohne ihn stieg das Wetter mit
        # KeyError 'altitude_cooling' aus (Nutzer-Log 2026-07-28).
        self._prime_all_generator_parameters()

        current_index = self.main_tab_bar.currentIndex()
        if 0 <= current_index < len(self.tab_order):
            tab_name = self.tab_order[current_index]
            tab_instance = self.tabs.get(tab_name)

            if tab_instance and hasattr(tab_instance, 'generate'):
                try:
                    tab_instance.generate()
                    self.logger.info(f"Generation triggered for tab: {tab_name}")
                except Exception as e:
                    self.logger.error(f"Generation failed for {tab_name}: {e}")
                    QMessageBox.critical(self, "Generation Error", f"Failed to generate {tab_name}: {str(e)}")
            else:
                QMessageBox.information(self, "Generation Unavailable", f"Cannot generate for {tab_name} tab")

    def _generate_all_maps(self):
        """Generate all maps in dependency order"""
        if not self.generation_orchestrator:
            QMessageBox.warning(self, "Generation Unavailable", "No GenerationOrchestrator available")
            return

        reply = QMessageBox.question(
            self, "Generate All Maps",
            "This will regenerate all maps in sequence. This may take several minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._regenerate_all_generators()
            except Exception as e:
                QMessageBox.critical(self, "Generation Error", f"Failed to start generation sequence: {str(e)}")

    def _clear_and_reset_all_generators(self):
        """
        Loescht Calculator-/Domain-Storage (data_lod_manager.clear_all_data())
        UND setzt den Orchestrator-seitigen LOD-Fortschritt fuer ALLE 6
        Generatoren zurueck (generation_orchestrator.reset_lod_status()) -
        Bug-Report 2026-07-25 ("nach Reset haeuft sich Erosion/Sedimentation
        immer weiter an"): clear_all_data() allein loescht nur die Daten,
        laesst aber CalculatorDispatcher.completed_lod/target_lod (siehe
        calculator_graph.py) unberuehrt. Klickt der Nutzer danach mit
        UNVERAENDERTEN Parametern auf Generieren, wertet der self_changed-
        Vergleich in request_generation() das als "nichts geaendert" und
        ueberspringt reset_lod_status() erneut - der Dispatcher haelt
        laengst geloeschte Knoten (z.B. water.erosion_sedimentation) faelsch-
        lich fuer bereits fertig auf dem ALTEN LOD-Stand.

        Gemeinsamer Helfer statt Duplikat: dieselbe Luecke wurde zuerst in
        _regenerate_all_generators() gefunden/gefixt, dann UNABHAENGIG davon
        erneut in _new_world()/_return_to_main_menu() entdeckt (beide
        riefen clear_all_data() aber nie reset_lod_status() auf) - ein
        gemeinsamer Helfer verhindert eine vierte Kopie derselben Luecke."""
        if self.data_lod_manager:
            self.data_lod_manager.clear_all_data()
        if self.generation_orchestrator:
            # Wie beim Auto-Start aus GeneratorType abgeleitet - ein
            # vergessener Generator wuerde hier seinen LOD-Fortschritt
            # behalten und beim naechsten Lauf gar nicht mehr rechnen.
            for generator_enum in GeneratorType:
                self.generation_orchestrator.reset_lod_status(generator_enum)

    def _regenerate_all_generators(self):
        """Regenerate all generators in proper dependency order"""
        if not self.generation_orchestrator:
            return

        try:
            # Clear all data + Orchestrator-LOD-Fortschritt fuer einen
            # echten Neustart (siehe _clear_and_reset_all_generators()).
            self._clear_and_reset_all_generators()

            # Get generation sequence from navigation manager
            if self.navigation_manager and hasattr(self.navigation_manager, 'tab_order'):
                generator_sequence = [tab for tab in self.navigation_manager.tab_order[2:] if tab in self.tabs]
            else:
                generator_sequence = [g.value for g in GeneratorType]

            target_lod = self.toolbar_lod_combo.currentText() if hasattr(self, 'toolbar_lod_combo') else "FINAL"

            # Queue all generators
            for generator_type in generator_sequence:
                if generator_type in self.tabs:
                    self.generation_orchestrator.request_generation(
                        generator_type=generator_type,
                        parameters={},
                        target_lod=target_lod,
                        source_tab="regenerate_all",
                        priority=10
                    )

            self.logger.info(f"Regeneration sequence started for {len(generator_sequence)} generators")

        except Exception as e:
            self.logger.error(f"Regeneration sequence failed: {e}")
            raise

    def _stop_all_generation(self):
        """Stop all active generation processes"""
        if not self.generation_orchestrator:
            return

        if hasattr(self.generation_orchestrator, 'stop_all_generation'):
            self.generation_orchestrator.stop_all_generation()
            self.active_generations.clear()
            self.status_indicator.set_warning("All generation stopped")
            self.logger.info("All generation stopped by user request")

    @pyqtSlot(str)
    def _on_toolbar_lod_changed(self, lod_level: str):
        """Handle toolbar LOD selection changes"""
        self._set_global_target_lod(lod_level)

    # View and Interface Methods

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _reset_tab_layout(self):
        """Reset all tab layouts to default"""
        for tab_instance in self.tabs.values():
            if hasattr(tab_instance, 'reset_layout'):
                tab_instance.reset_layout()
        self.logger.info("Tab layouts reset to defaults")

    def _refresh_all_displays(self):
        """Refresh all tab displays with current data"""
        for tab_instance in self.tabs.values():
            if hasattr(tab_instance, 'update_display_mode'):
                try:
                    tab_instance.update_display_mode()
                except Exception as e:
                    self.logger.warning(f"Failed to refresh tab display: {e}")
        self.logger.info("All displays refreshed")

    # Help and Information Methods

    def _show_about(self):
        """Display application about dialog"""
        QMessageBox.about(
            self, "About MapGenerator",
            "MapGenerator Professional v1.0\n\n"
            "Advanced Terrain & World Generation Suite\n"
            "Built with PyQt6 and optimized algorithms\n\n"
            "Features:\n"
            "• Multi-LOD terrain generation\n"
            "• Integrated geology and climate modeling\n"
            "• Real-time 2D/3D visualization\n"
            "• Professional export capabilities\n\n"
            "© 2024 MapGenerator Development Team"
        )

    def _show_shortcuts(self):
        """Display keyboard shortcuts help"""
        shortcuts_text = """
        Keyboard Shortcuts:
        
        File Operations:
        Ctrl+N    - New World
        Ctrl+O    - Open World
        Ctrl+S    - Save World
        Ctrl+E    - Export World
        
        Generation:
        Ctrl+R    - Regenerate All
        Ctrl+Shift+S - Stop All Generation
        
        View:
        F11       - Toggle Fullscreen
        F5        - Refresh All Displays
        F1        - Show This Help
        
        Navigation:
        Tab       - Next Tab
        Shift+Tab - Previous Tab
        """

        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)

    def _report_issue(self):
        """Handle issue reporting"""
        QMessageBox.information(
            self, "Report Issue",
            "Issue reporting functionality will be available in future version.\n\n"
            "For now, please check the console output for detailed error information."
        )

    # Resource Management and Cleanup

    def closeEvent(self, event):
        """
        Handle window close event with comprehensive cleanup
        ==================================================

        Performs thorough resource cleanup before closing the
        editor window, without asking for confirmation.

        Args:
            event: QCloseEvent to accept or ignore
        """
        try:
            self._perform_cleanup()
            if self._main_menu:
                self._main_menu.show()
            event.accept()
            self.logger.info("MapEditor closed successfully")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            event.accept()  # Close anyway to prevent hanging

    def _perform_cleanup(self):
        """
        Comprehensive resource cleanup
        =============================

        Performs thorough cleanup of all resources including
        timers, generation processes, tab resources, and
        signal disconnections.
        """
        # Stop status timer
        if self.status_update_timer.isActive():
            self.status_update_timer.stop()

        # Stop all active generations
        if self.generation_orchestrator and hasattr(self.generation_orchestrator, 'stop_all_generation'):
            try:
                self.generation_orchestrator.stop_all_generation()
            except Exception as e:
                self.logger.warning(f"Error stopping generations: {e}")

        # Cleanup all tabs
        for tab_name, tab_instance in self.tabs.items():
            try:
                if hasattr(tab_instance, 'cleanup_resources'):
                    tab_instance.cleanup_resources()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup tab {tab_name}: {e}")

        # Disconnect orchestrator signals safely
        if self.generation_orchestrator:
            self._disconnect_orchestrator_signals()

        # GPU resource cleanup
        if self.shader_manager and hasattr(self.shader_manager, 'cleanup_all_resources'):
            try:
                self.shader_manager.cleanup_all_resources()
            except Exception as e:
                self.logger.warning(f"GPU cleanup failed: {e}")

    def _disconnect_orchestrator_signals(self):
        """
        Safely disconnect all orchestrator signals
        ==========================================

        Prevents memory leaks by properly disconnecting all
        signal-slot connections with error handling.
        """
        signals_to_disconnect = [
            (self.generation_orchestrator.generation_started, self._on_generation_started),
            (self.generation_orchestrator.generation_completed, self._on_generation_completed),
            (self.generation_orchestrator.generation_progress, self._on_generation_progress),
            (self.generation_orchestrator.batch_generation_completed, self._on_batch_generation_completed),
            (self.generation_orchestrator.dependency_invalidated, self._on_dependency_invalidated),
            (self.generation_orchestrator.calculator_status_changed, self._on_calculator_status_changed)
        ]

        for signal, slot in signals_to_disconnect:
            try:
                signal.disconnect(slot)
                self.logger.debug(f"Disconnected orchestrator signal: {signal}")
            except (TypeError, RuntimeError):
                # Signal was not connected or already disconnected
                self.logger.debug(f"Signal already disconnected: {signal}")

    # Debug and Development Methods

    def get_current_status_summary(self) -> dict:
        """
        Generate comprehensive status summary for debugging
        =================================================

        Collects detailed status information from all components
        for diagnostic and development purposes.

        Returns:
            Dictionary containing comprehensive status information
        """
        try:
            return {
                "window_info": {
                    "current_tab": self.main_tab_bar.tabText(
                        self.main_tab_bar.currentIndex()) if self.main_tab_bar.currentIndex() >= 0 else "None",
                    "total_tabs": self.main_tab_bar.count(),
                    "window_size": f"{self.width()}x{self.height()}"
                },
                "generation_status": {
                    "active_generations": list(self.active_generations),
                    "tab_generation_status": self.tab_generation_status,
                    "orchestrator_available": self.generation_orchestrator is not None
                },
                "resource_status": {
                    "memory_usage": self.data_lod_manager.get_memory_usage() if self.data_lod_manager else {},
                    "tabs_loaded": list(self.tabs.keys()),
                    "shader_manager_available": self.shader_manager is not None
                },
                "ui_status": {
                    "status_timer_active": self.status_update_timer.isActive(),
                    "fullscreen": self.isFullScreen()
                }
            }
        except Exception as e:
            self.logger.warning(f"Status summary generation failed: {e}")
            return {"error": str(e)}

    def force_refresh_all_tabs(self):
        """
        Force refresh of all tabs (development/debug method)
        ===================================================

        Forces complete refresh of all tab content and displays.
        Useful for development and troubleshooting.
        """
        self.logger.info("Force refreshing all tabs")

        for tab_name, tab_instance in self.tabs.items():
            try:
                # Force data reload if available
                if hasattr(tab_instance, 'force_data_reload'):
                    tab_instance.force_data_reload()

                # Force display update
                if hasattr(tab_instance, 'update_display_mode'):
                    tab_instance.update_display_mode()

                self.logger.debug(f"Refreshed tab: {tab_name}")

            except Exception as e:
                self.logger.warning(f"Failed to refresh tab {tab_name}: {e}")

    def export_debug_information(self) -> str:
        """
        Export comprehensive debug information
        =====================================

        Generates detailed debug report including status,
        errors, and configuration for troubleshooting.

        Returns:
            Formatted debug information string
        """
        import json
        import datetime

        try:
            debug_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "application": "MapGenerator MapEditor",
                "version": "1.0.0",
                "status_summary": self.get_current_status_summary(),
                "error_log": [],  # Could be expanded to include recent errors
                "configuration": {
                    "window_settings": WindowSettings.MAP_EDITOR,
                    "constants": {
                        "status_update_interval": EditorConstants.STATUS_UPDATE_INTERVAL_MS,
                        "generation_timeout": EditorConstants.GENERATION_TIMEOUT_MS
                    }
                }
            }

            return json.dumps(debug_info, indent=2)

        except Exception as e:
            return f"Debug information export failed: {str(e)}"
