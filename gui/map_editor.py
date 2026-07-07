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

from PyQt5.QtWidgets import QMainWindow, QApplication, QTabWidget, QTabBar, QStackedWidget, QMenu, QAction, QLabel, \
    QComboBox, QCheckBox, QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QSplitter
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
import logging
from typing import Optional

from gui.config.gui_default import WindowSettings, EditorConstants
from gui.OldManagers.data_lod_manager import DataLODManager
from gui.OldManagers.generation_orchestrator import GenerationOrchestrator
from gui.OldManagers.navigation_manager import NavigationManager
from gui.OldManagers.parameter_manager import ParameterManager
from gui.OldManagers.shader_manager import ShaderManager
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
WEATHER_AVAILABLE, WeatherTab, weather_error = _import_tab_safely("gui.tabs.weather_tab", "WeatherTab")
WATER_AVAILABLE, WaterTab, water_error = _import_tab_safely("gui.tabs.water_tab", "WaterTab")
BIOME_AVAILABLE, BiomeTab, biome_error = _import_tab_safely("gui.tabs.biome_tab", "BiomeTab")
SETTLEMENT_AVAILABLE, SettlementTab, settlement_error = _import_tab_safely("gui.tabs.settlement_tab", "SettlementTab")
OVERVIEW_AVAILABLE, OverviewTab, overview_error = _import_tab_safely("gui.tabs.overview_tab", "OverviewTab")


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

    def __init__(self, parent=None):
        super().__init__(parent)

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
        self.tab_order = []  # lowercase tab names, Index == main_tab_bar/stack index
        self.tabs = {}
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
        screen_geometry = QApplication.desktop().screenGeometry()
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
        self.main_tab_bar.setShape(QTabBar.RoundedNorth)
        self.main_tab_bar.setExpanding(False)

        # Spalte 1 (fix): globaler Pipeline-Status, bleibt über alle Haupt-Tabs
        # hinweg sichtbar.
        self.pipeline_status_panel = PipelineStatusPanel()
        self.pipeline_status_panel.setMinimumWidth(190)
        self.pipeline_status_panel.setMaximumWidth(190)

        # Spalte 2 (flexibel): Viewport der Tabs, per BaseMapTab.viewport_widget befüllt,
        # darunter die zwei permanenten globalen Checkboxen (Contour Lines/Shadows),
        # die unabhängig vom Haupt-Tab immer sichtbar sind.
        self.viewport_stack = QStackedWidget()

        self.contour_checkbox = QCheckBox("Contour Lines (Höhenlinien)")
        self.shadow_checkbox = QCheckBox("Shadows (Shading)")
        self.contour_checkbox.toggled.connect(self._on_global_contour_toggled)
        self.shadow_checkbox.toggled.connect(self._on_global_shadow_toggled)

        global_overlay_row = QWidget()
        global_overlay_layout = QHBoxLayout(global_overlay_row)
        global_overlay_layout.setContentsMargins(10, 4, 10, 4)
        global_overlay_layout.addWidget(self.contour_checkbox)
        global_overlay_layout.addWidget(self.shadow_checkbox)
        global_overlay_layout.addStretch()

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

        self.main_splitter = QSplitter(Qt.Horizontal)
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
        WeightedProgressCalculator) links, permanenter [GENERIEREN]-Button rechts.
        """
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(10, 6, 10, 6)

        self.footer_progress_bar = ProgressBar()
        footer_layout.addWidget(self.footer_progress_bar, 1)

        self.generate_button = BaseButton("GENERIEREN", "primary")
        self.generate_button.clicked.connect(self._generate_current_tab)
        footer_layout.addWidget(self.generate_button)

        return footer

    def _refresh_footer_progress(self):
        """Aktualisiert den gewichteten Ladebalken aus tab_generation_status."""
        if not self.footer_progress_bar:
            return

        target_lod_text = self.toolbar_lod_combo.currentText() if hasattr(self, 'toolbar_lod_combo') else None
        try:
            target_lod = int(target_lod_text)
        except (TypeError, ValueError):
            target_lod = self.progress_calculator.max_lod

        percent, done, total = self.progress_calculator.progress_percent(
            self.tab_generation_status, target_lod
        )
        self.footer_progress_bar.set_progress(
            percent, f"Pipeline: {percent}%", f"{done:.1f} / {total:.1f} gewichtete Kosten"
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
            self.generation_orchestrator = GenerationOrchestrator(data_lod_manager=self.data_lod_manager)

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
        # Tab configuration with availability status - Terrain startet zuerst,
        # Overview steht als Abschluss-Übersicht am Ende.
        tab_configs = [
            ("Terrain", TerrainTab, TERRAIN_AVAILABLE, terrain_error),
            ("Geology", GeologyTab, GEOLOGY_AVAILABLE, geology_error),
            ("Weather", WeatherTab, WEATHER_AVAILABLE, weather_error),
            ("Water", WaterTab, WATER_AVAILABLE, water_error),
            ("Biome", BiomeTab, BIOME_AVAILABLE, biome_error),
            ("Settlement", SettlementTab, SETTLEMENT_AVAILABLE, settlement_error),
            ("Overview", OverviewTab, OVERVIEW_AVAILABLE, overview_error)
        ]

        for tab_name, tab_class, available, error_type in tab_configs:
            try:
                self.logger.info(f"DEBUG: Starting creation of {tab_name} tab...")

                if available and tab_class:
                    # Attempt to create real tab instance
                    tab_instance = self._create_tab_instance(tab_class, tab_name)
                    if tab_instance:
                        self._add_successful_tab(tab_name, tab_instance)
                        self.logger.info(f"DEBUG: {tab_name} tab completed successfully")
                    else:
                        self._add_error_tab(tab_name, "instantiation_failed")
                else:
                    # Create appropriate error tab based on failure type
                    self._add_error_tab(tab_name, error_type)

                self.logger.info(f"DEBUG: {tab_name} tab processing finished")

            except Exception as e:
                self.logger.error(f"Unexpected error creating {tab_name} tab: {e}")
                import traceback
                traceback.print_exc()
                self._add_error_tab(tab_name, "unexpected_error")

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

    def _add_successful_tab(self, tab_name: str, tab_instance: QWidget):
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
            self.tab_order.append(tab_name.lower())
            self.logger.info(f"Tab added to shell at index: {index}")

            self.tabs[tab_name.lower()] = tab_instance
            self.tab_generation_status[tab_name.lower()] = {}

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

    def _add_error_tab(self, tab_name: str, error_type: str):
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
        self.tab_order.append(tab_name.lower())

        self.tabs[tab_name.lower()] = error_tab

    def _create_placeholder_widget(self, text: str) -> QWidget:
        """Kleiner Platzhalter für Spalte 3, wenn ein Tab nicht geladen werden konnte."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
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
        layout.setAlignment(Qt.AlignCenter)

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
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color}; margin: 20px;")
        layout.addWidget(title_label)

        # Error message
        message_label = QLabel(message)
        message_label.setAlignment(Qt.AlignCenter)
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

            # Update navigation manager
            if self.navigation_manager:
                self.navigation_manager.current_tab = tab_name_lower

            # Globale Contour/Shadow-Checkboxen gelten tab-übergreifend - beim
            # Wechsel auf den neu aktiven Tab anwenden, damit dessen Display
            # den aktuellen globalen Zustand übernimmt.
            self._apply_global_overlays_to_active_tab()

            # Erzwingt einen Display-Refresh mit dem aktuell gewählten Render-Modus.
            # Ohne das bleibt der Viewport leer, bis der Nutzer den Modus manuell
            # umschaltet - der Default-Radio (z.B. "Height") feuert beim Erstellen
            # kein toggled-Signal, weil setChecked(True) vor dem connect() passiert.
            tab_instance = self.tabs.get(tab_name_lower)
            if tab_instance and hasattr(tab_instance, 'update_display_mode'):
                tab_instance.update_display_mode()

            self.logger.debug(f"Tab changed to: {tab_text}")

    def _get_active_tab_instance(self):
        """Liefert die BaseMapTab-Instanz des aktuell im main_tab_bar aktiven Tabs."""
        index = self.main_tab_bar.currentIndex()
        if 0 <= index < len(self.tab_order):
            return self.tabs.get(self.tab_order[index])
        return None

    def _apply_global_overlays_to_active_tab(self):
        """Wendet den aktuellen Zustand der globalen Checkboxen auf den aktiven Tab an."""
        tab_instance = self._get_active_tab_instance()
        if not tab_instance:
            return
        if hasattr(tab_instance, 'set_contour_overlay'):
            tab_instance.set_contour_overlay(self.contour_checkbox.isChecked())
        if hasattr(tab_instance, 'set_shadow_overlay'):
            tab_instance.set_shadow_overlay(self.shadow_checkbox.isChecked())

    @pyqtSlot(bool)
    def _on_global_contour_toggled(self, checked: bool):
        """Globales Contour-Lines-Toggle (Spalte 2) - wirkt auf den aktiven Tab."""
        tab_instance = self._get_active_tab_instance()
        if tab_instance and hasattr(tab_instance, 'set_contour_overlay'):
            tab_instance.set_contour_overlay(checked)

    @pyqtSlot(bool)
    def _on_global_shadow_toggled(self, checked: bool):
        """Globales Shadows-Toggle (Spalte 2) - wirkt auf den aktiven Tab."""
        tab_instance = self._get_active_tab_instance()
        if tab_instance and hasattr(tab_instance, 'set_shadow_overlay'):
            tab_instance.set_shadow_overlay(checked)

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
    # pyqtSignal-Deklarationen (gui/OldManagers/generation_orchestrator.py):
    #   generation_started    = pyqtSignal(str, int)   # (generator_type, lod_level)
    #   generation_completed  = pyqtSignal(str, dict)  # (request_id, result_data)
    #   generation_progress   = pyqtSignal(int, str)   # (progress, message) - global, kein generator_type
    # Vorherige Versionen dieser Handler hatten davon abweichende Signaturen
    # UND waren nie mit den Orchestrator-Signalen verbunden (siehe
    # _setup_signals()) - die Pipeline-Status-Spalte blieb dadurch dauerhaft
    # auf "Unknown" stehen.

    @pyqtSlot(str, int)
    def _on_generation_started(self, generator_type: str, lod_level: int):
        """Handle generation start events (ein LOD-Level beginnt)"""
        if not generator_type:
            return

        generation_key = f"{generator_type}_{lod_level}"
        self.active_generations.add(generation_key)

        if hasattr(self, 'active_generations_label'):
            self.active_generations_label.setText(f"Generations: {len(self.active_generations)} active")

        self.status_indicator.set_warning(f"Generating {generator_type} LOD {lod_level}")
        if self.pipeline_status_panel:
            self.pipeline_status_panel.set_calculating(generator_type, lod_level)
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
            if self.pipeline_status_panel:
                self.pipeline_status_panel.set_finished(generator_type, lod_level)
        else:
            self.status_indicator.set_error(f"{generator_type} LOD {lod_level} failed")
            if self.pipeline_status_panel:
                self.pipeline_status_panel.set_failed(generator_type, f"LOD {lod_level} failed")

        self._refresh_footer_progress()
        self.logger.info(f"Generation completed: {generation_key} success={success}")

    @pyqtSlot(int, str)
    def _on_generation_progress(self, progress_percent: int, message: str):
        """Handle generation progress updates (global, ohne generator_type/lod_level)"""
        self.status_indicator.set_warning(f"{message} ({progress_percent}%)")

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
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Clear data manager
                if self.data_lod_manager:
                    self.data_lod_manager.clear_all_data()

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
        """Open saved world data"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open World", "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                # TODO: Implement world loading
                QMessageBox.information(self, "Open World", "World loading will be implemented in future version")
                self.logger.info(f"World open requested: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open world: {str(e)}")

    def _save_world(self):
        """Save current world data"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save World", "world.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            try:
                # TODO: Implement world saving
                QMessageBox.information(self, "Save World", "World saving will be implemented in future version")
                self.logger.info(f"World save requested: {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save world: {str(e)}")

    def _export_world(self):
        """Export complete world data"""
        try:
            # TODO: Implement comprehensive world export
            QMessageBox.information(self, "Export World",
                                    "Comprehensive world export will be implemented in future version")
            self.logger.info("World export requested")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export world: {str(e)}")

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
            # Clear data manager
            if self.data_lod_manager:
                self.data_lod_manager.clear_all_data()

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

            if self.parent():
                self.parent().show()

            self.logger.info("Success returning to Main Menu.")

        except Exception as e:
            self.logger.error(f"Failed to return to Main Menu: {e}")


    # Generation Control Methods

    def _generate_current_tab(self):
        """Generate content for currently active tab"""
        if not self.generation_orchestrator:
            QMessageBox.warning(self, "Generation Unavailable", "No GenerationOrchestrator available")
            return

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
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                self._regenerate_all_generators()
            except Exception as e:
                QMessageBox.critical(self, "Generation Error", f"Failed to start generation sequence: {str(e)}")

    def _regenerate_all_generators(self):
        """Regenerate all generators in proper dependency order"""
        if not self.generation_orchestrator:
            return

        try:
            # Clear all data for fresh start
            if self.data_lod_manager:
                self.data_lod_manager.clear_all_data()

            # Get generation sequence from navigation manager
            if self.navigation_manager and hasattr(self.navigation_manager, 'tab_order'):
                generator_sequence = [tab for tab in self.navigation_manager.tab_order[2:] if tab in self.tabs]
            else:
                generator_sequence = ["terrain", "geology", "weather", "water", "biome", "settlement"]

            target_lod = self.toolbar_lod_combo.currentText() if hasattr(self, 'toolbar_lod_combo') else "FINAL"

            # Queue all generators
            for generator_type in generator_sequence:
                if generator_type in self.tabs:
                    if self.pipeline_status_panel:
                        self.pipeline_status_panel.set_queued(generator_type)
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
            "Built with PyQt5 and optimized algorithms\n\n"
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
            if self.parent():
                self.parent().show()
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
            (self.generation_orchestrator.dependency_invalidated, self._on_dependency_invalidated)
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
