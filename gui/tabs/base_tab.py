"""
Path: gui/tabs/base_tab.py

Funktionsweise: Basis-Klasse für alle Map-Editor Tabs - VOLLSTÄNDIG REFACTORED
- Standardisiertes 70/30 Layout (Canvas links, Controls rechts) mit dynamischer Größenberechnung
- Gemeinsame Auto-Simulation Controls für alle Tabs
- Input-Status Display (verfügbare Dependencies)
- Observer-Pattern für Cross-Tab Updates
- 2D/3D Toggle-Navigation mit fixiertem Navigation-Panel
- Performance-optimiertes Debouncing-System
- Error-Handling und Resource-Management
- REFACTORED: Modulare Architektur, Resource-Tracking, Display-Change-Detection,
             Standard-Orchestrator-Handler, Explizite Imports, Konstanten-System
"""

from PyQt5.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QTimer, QMetaObject, Q_ARG, QDateTime
)
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QFrame, QSizePolicy, QLabel, QVBoxLayout,
    QScrollArea, QSplitter, QMessageBox, QGroupBox, QTextEdit,
    QStackedWidget, QRadioButton, QButtonGroup, QComboBox, QProgressBar
)

import logging
import time
import hashlib
import weakref
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass

from gui.widgets.widgets import BaseButton, AutoSimulationPanel, ProgressBar, StatusIndicator, DisplayWrapper, \
    NavigationPanel

# Map Display Import mit Fallback - ERWEITERT für 2D+3D
try:
    from gui.widgets.map_display_2d import MapDisplay2D
    MAP_DISPLAY_2D_AVAILABLE = True
except ImportError:
    MAP_DISPLAY_2D_AVAILABLE = False
    print("Warning: MapDisplay2D not available")

try:
    from gui.widgets.map_display_3d import MapDisplay3DWidget
    MAP_DISPLAY_3D_AVAILABLE = True
except ImportError:
    MAP_DISPLAY_3D_AVAILABLE = False
    print("Warning: MapDisplay3DWidget not available")

def get_base_error_decorators():
    """
    Funktionsweise: Zentrale Error-Decorators für alle BaseMapTab-Methoden
    Aufgabe: Lädt ALLE benötigten Decorators für Tab-Funktionalität
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import (
            core_generation_handler,
            cleanup_handler,
            ui_navigation_handler,
            memory_critical_handler,
            ui_update_handler,
            dependency_handler
        )
        return (
            core_generation_handler,
            cleanup_handler,
            ui_navigation_handler,
            memory_critical_handler,
            ui_update_handler,
            dependency_handler
        )
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator, noop_decorator, noop_decorator, noop_decorator

(core_generation_handler, cleanup_handler, ui_navigation_handler,
 memory_critical_handler, ui_update_handler, dependency_handler) = get_base_error_decorators()


@dataclass
class LayoutConstants:
    """Konstanten für Layout-Berechnungen"""
    MIN_CONTROL_PANEL_WIDTH: int = 280
    MAX_CONTROL_PANEL_WIDTH: int = 450
    CANVAS_RATIO: float = 0.7
    CONTROL_RATIO: float = 0.3
    SCROLLBAR_MARGIN: int = 25
    CONTENT_MARGIN: int = 20
    SPLITTER_SPACING: int = 10
    CONTAINER_MARGINS: int = 5


class ResourceTracker:
    """
    Funktionsweise: Systematisches Resource-Management für Memory-Leak-Prevention
    Aufgabe: Verfolgt alle erstellten Ressourcen und ermöglicht systematisches Cleanup
    """

    def __init__(self):
        self.tracked_resources = {}  # {resource_id: WeakReference}
        self.resource_types = {}     # {resource_id: resource_type}
        self.creation_timestamps = {} # {resource_id: timestamp}
        self.cleanup_functions = {}  # {resource_id: cleanup_func}
        self._next_id = 0

    def register_resource(self, resource, resource_type: str, cleanup_func=None) -> str:
        """Registriert Ressource für Tracking"""
        resource_id = f"{resource_type}_{self._next_id}"
        self._next_id += 1

        self.tracked_resources[resource_id] = weakref.ref(resource)
        self.resource_types[resource_id] = resource_type
        self.creation_timestamps[resource_id] = time.time()

        if cleanup_func:
            self.cleanup_functions[resource_id] = cleanup_func

        return resource_id

    def cleanup_by_type(self, resource_type: str):
        """Cleaned alle Ressourcen eines bestimmten Typs"""
        to_cleanup = [
            rid for rid, rtype in self.resource_types.items()
            if rtype == resource_type
        ]

        for resource_id in to_cleanup:
            self._cleanup_resource(resource_id)

    def cleanup_by_age(self, max_age_seconds: float):
        """Cleaned alte Ressourcen basierend auf Alter"""
        current_time = time.time()
        to_cleanup = [
            rid for rid, timestamp in self.creation_timestamps.items()
            if current_time - timestamp > max_age_seconds
        ]

        for resource_id in to_cleanup:
            self._cleanup_resource(resource_id)

    def force_cleanup_all(self):
        """Emergency cleanup aller Ressourcen"""
        for resource_id in list(self.tracked_resources.keys()):
            self._cleanup_resource(resource_id)

    def _cleanup_resource(self, resource_id: str):
        """Cleaned einzelne Ressource"""
        if resource_id in self.cleanup_functions:
            try:
                self.cleanup_functions[resource_id]()
            except Exception as e:
                logging.getLogger(__name__).warning(f"Cleanup function failed for {resource_id}: {e}")

        # Remove from tracking
        self.tracked_resources.pop(resource_id, None)
        self.resource_types.pop(resource_id, None)
        self.creation_timestamps.pop(resource_id, None)
        self.cleanup_functions.pop(resource_id, None)

    def get_memory_usage(self) -> Dict[str, int]:
        """Gibt Memory-Usage pro Resource-Type zurück"""
        usage = {}
        for resource_id, resource_ref in self.tracked_resources.items():
            resource = resource_ref()
            if resource is not None:
                resource_type = self.resource_types[resource_id]
                if hasattr(resource, 'nbytes'):
                    usage[resource_type] = usage.get(resource_type, 0) + resource.nbytes
                elif hasattr(resource, '__sizeof__'):
                    usage[resource_type] = usage.get(resource_type, 0) + resource.__sizeof__()
        return usage


class DisplayUpdateManager:
    """
    Funktionsweise: Change-Detection für Display-Updates um unnötige Re-Renderings zu vermeiden
    Aufgabe: Prüft ob Display-Update wirklich nötig ist basierend auf Data-Hash
    """

    def __init__(self):
        self.last_display_hashes = {}  # {display_id: data_hash}
        self.pending_updates = set()   # Set von display_ids

    def needs_update(self, display_id: str, data, layer_type: str, display_mode: str = "default") -> bool:
        """Prüft ob Display-Update wirklich nötig ist"""
        current_hash = self._calculate_hash(data, layer_type, display_mode)
        last_hash = self.last_display_hashes.get(display_id)
        return current_hash != last_hash

    def mark_updated(self, display_id: str, data, layer_type: str, display_mode: str = "default"):
        """Markiert Display als updated"""
        current_hash = self._calculate_hash(data, layer_type, display_mode)
        self.last_display_hashes[display_id] = current_hash
        self.pending_updates.discard(display_id)

    def _calculate_hash(self, data, layer_type: str, display_mode: str) -> str:
        """Berechnet Hash für Change-Detection"""
        hash_input = f"{layer_type}_{display_mode}"

        if data is not None:
            if hasattr(data, 'data'):
                hash_input += f"_{hash(data.data.tobytes()) if hasattr(data.data, 'tobytes') else hash(str(data.data))}"
            elif hasattr(data, 'tobytes'):
                hash_input += f"_{hash(data.tobytes())}"
            else:
                hash_input += f"_{hash(str(data))}"

        return hashlib.md5(hash_input.encode()).hexdigest()


class BaseMapTab(QWidget):
    """
    Funktionsweise: Basis-Klasse für alle Map-Editor Tabs - VOLLSTÄNDIG REFACTORED
    Aufgabe: Stellt modulare Standard-Funktionalität und Layout bereit
    Kommunikation: Signals für data_updated, parameter_changed, validation_status_changed
    REFACTORED: Modulare Architektur, Resource-Tracking, Standard-Handler
    """

    # Signals für Cross-Tab Communication
    data_updated = pyqtSignal(str, str)  # (generator_type, data_key)
    parameter_changed = pyqtSignal(str, dict)  # (generator_type, parameters)
    validation_status_changed = pyqtSignal(str, bool, list)  # (generator_type, is_valid, messages)
    generation_completed = pyqtSignal(str, bool)  # (generator_type, success)

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        super().__init__()

        # Attribute ZUERST initialisieren
        self.data_manager = data_manager
        self.navigation_manager = navigation_manager
        self.shader_manager = shader_manager
        self.generation_orchestrator = generation_orchestrator

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Layout-Konstanten
        self.layout_constants = LayoutConstants()

        # Resource-Management
        self.resource_tracker = ResourceTracker()
        self.display_manager = DisplayUpdateManager()

        # Auto-Simulation State
        self.auto_simulation_enabled = False
        self.auto_simulation_timer = QTimer()
        self.auto_simulation_timer.setSingleShot(True)
        self.auto_simulation_timer.timeout.connect(self.on_auto_simulation_triggered)

        self.performance_level = "preview"  # Default performance level
        self.manual_generate_button = None

        # Generation State
        self.generation_start_time = None
        self.generation_in_progress = False
        self.target_lod = "FINAL"
        self.available_lods = set()

        # UI Components (werden von setup_ui gesetzt)
        self.canvas_container = None
        self.control_panel = None
        self.control_widget = None

        # 2D/3D Display Components
        self.map_display_2d = None
        self.map_display_3d = None
        self.display_stack = None
        self.current_view = "2d"

        # Control Panel Components
        self.auto_simulation_panel = None
        self.navigation_panel = None
        self.error_status = None

        # Splitter für Layout-State-Management
        self.splitter = None

        self.tab_generator_type = None

        # UI Setup mit modularer Reihenfolge
        self.setup_ui()
        self.setup_signals()

    def setup_ui(self):
        """
        Funktionsweise: Hauptmethode für UI-Setup - REFACTORED in modulare Struktur
        Aufgabe: Koordiniert alle UI-Setup-Schritte
        """
        self.logger.debug("setup_ui() started")

        self.setup_layout_structure()
        self.setup_display_stack()
        self.setup_control_panel()
        self.setup_navigation_panel()
        self.setup_auto_simulation()
        self.setup_error_handling()

        self.logger.debug("setup_ui() completed successfully")

    def setup_layout_structure(self):
        """
        Funktionsweise: Erstellt Basis-Layout-Struktur - MODULAR
        Aufgabe: 70/30 Layout mit Splitter und dynamischer Größenberechnung
        """
        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )
        main_layout.setSpacing(self.layout_constants.SPLITTER_SPACING)

        # Canvas Container (70%)
        self.canvas_container = QFrame()
        self.canvas_container.setFrameStyle(QFrame.StyledPanel)
        self.canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Control Widget (30%)
        optimal_width = self.calculate_control_panel_width()
        self.control_widget = QFrame()
        self.control_widget.setFrameStyle(QFrame.StyledPanel)
        self.control_widget.setMaximumWidth(optimal_width)
        self.control_widget.setMinimumWidth(min(self.layout_constants.MIN_CONTROL_PANEL_WIDTH, optimal_width))
        self.control_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Splitter für resizable Layout
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.canvas_container)
        self.splitter.addWidget(self.control_widget)

        # 70/30 initial ratio basierend auf Konstanten
        canvas_width = int(1000 * self.layout_constants.CANVAS_RATIO)
        control_width = int(1000 * self.layout_constants.CONTROL_RATIO)
        self.splitter.setSizes([canvas_width, control_width])

        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.splitterMoved.connect(self.save_splitter_state)

        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    def setup_display_stack(self):
        """
        Funktionsweise: Erstellt 2D/3D Display-Stack - MODULAR
        Aufgabe: View-Toggle und Display-Erstellung mit Resource-Tracking
        """
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )

        # View Toggle Buttons
        view_toggle_layout = QHBoxLayout()

        self.view_2d_button = BaseButton("2D View", "primary")
        self.view_2d_button.clicked.connect(lambda: self.switch_view("2d"))
        view_toggle_layout.addWidget(self.view_2d_button)

        self.view_3d_button = BaseButton("3D View", "secondary")
        self.view_3d_button.clicked.connect(lambda: self.switch_view("3d"))
        view_toggle_layout.addWidget(self.view_3d_button)

        view_toggle_layout.addStretch()

        # Visualization Controls
        self.visualization_controls = self.create_visualization_controls()
        view_toggle_layout.addWidget(self.visualization_controls)

        canvas_layout.addLayout(view_toggle_layout)

        # Stacked Widget für 2D/3D Display
        self.display_stack = QStackedWidget()

        # 2D Display mit Resource-Tracking
        self._create_2d_display()

        # 3D Display mit Resource-Tracking
        self._create_3d_display()

        canvas_layout.addWidget(self.display_stack)
        self.canvas_container.setLayout(canvas_layout)

    def setup_standard_orchestrator_handlers(self, generator_type: str):
        """
        Funktionsweise: Einfache Orchestrator-Integration ohne Handler-Klasse - VEREINFACHT
        Aufgabe: Direkte Signal-Verbindungen für bessere Debugging-Fähigkeiten
        Parameter: generator_type (str) - Type des Generators für diesen Tab
        """
        if not self.generation_orchestrator:
            self.logger.warning("No GenerationOrchestrator available for signal setup")
            return

        self.tab_generator_type = generator_type  # Speichern für Filter-Logic

        try:
            # Direkte Signal-Verbindungen
            self.generation_orchestrator.generation_started.connect(self.on_orchestrator_generation_started)
            self.generation_orchestrator.generation_completed.connect(self.on_orchestrator_generation_completed)
            self.generation_orchestrator.generation_progress.connect(self.on_orchestrator_generation_progress)

            # LOD Progression falls verfügbar
            if hasattr(self.generation_orchestrator, 'lod_progression_completed'):
                self.generation_orchestrator.lod_progression_completed.connect(self.on_lod_progression_completed)

            self.logger.debug(f"Orchestrator signals connected for {generator_type}")

        except Exception as e:
            self.logger.error(f"Failed to connect orchestrator signals: {e}")

    def _create_2d_display(self):
        """Erstellt 2D Display mit Resource-Tracking"""
        try:
            if MAP_DISPLAY_2D_AVAILABLE:
                display_2d = MapDisplay2D()
                self.map_display_2d = DisplayWrapper(display_2d)
                self.display_stack.addWidget(self.map_display_2d.display)

                # Resource-Tracking
                self.resource_tracker.register_resource(
                    display_2d, "2d_display",
                    cleanup_func=lambda: self.map_display_2d.cleanup_resources()
                )

                self.logger.debug("MapDisplay2D created successfully")
            else:
                fallback_2d = QLabel("2D Display\n(MapDisplay2D not available)")
                fallback_2d.setAlignment(Qt.AlignCenter)
                fallback_2d.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7;")
                self.map_display_2d = DisplayWrapper(fallback_2d)
                self.display_stack.addWidget(self.map_display_2d.display)
                self.logger.debug("2D fallback created")

        except Exception as e:
            self.logger.error(f"2D Display creation failed: {e}")
            fallback_2d = QLabel("2D Display\n(Error loading)")
            fallback_2d.setAlignment(Qt.AlignCenter)
            fallback_2d.setStyleSheet("background-color: #ecf0f1; border: 1px solid #bdc3c7;")
            self.map_display_2d = DisplayWrapper(fallback_2d)
            self.display_stack.addWidget(self.map_display_2d.display)

    def _create_3d_display(self):
        """Erstellt 3D Display mit Resource-Tracking"""
        try:
            if MAP_DISPLAY_3D_AVAILABLE:
                display_3d = MapDisplay3DWidget()
                self.map_display_3d = DisplayWrapper(display_3d)
                self.display_stack.addWidget(self.map_display_3d.display)

                # Resource-Tracking
                self.resource_tracker.register_resource(
                    display_3d, "3d_display",
                    cleanup_func=lambda: self.map_display_3d.cleanup_resources()
                )

                self.logger.debug("MapDisplay3DWidget created successfully")
            else:
                fallback_3d = QLabel("3D Display\n(MapDisplay3DWidget not available)")
                fallback_3d.setAlignment(Qt.AlignCenter)
                fallback_3d.setStyleSheet("background-color: #f8f9fa; border: 1px solid #bdc3c7;")
                self.map_display_3d = DisplayWrapper(fallback_3d)
                self.display_stack.addWidget(self.map_display_3d.display)
                self.logger.debug("3D fallback created")

        except Exception as e:
            self.logger.error(f"3D Display creation failed: {e}")
            fallback_3d = QLabel("3D Display\n(Error loading)")
            fallback_3d.setAlignment(Qt.AlignCenter)
            fallback_3d.setStyleSheet("background-color: #f8f9fa; border: 1px solid #bdc3c7;")
            self.map_display_3d = DisplayWrapper(fallback_3d)
            self.display_stack.addWidget(self.map_display_3d.display)

    def setup_control_panel(self):
        """
        Funktionsweise: Erstellt Control Panel mit Scroll-Area - MODULAR
        Aufgabe: Scrollable Parameter-Controls
        """
        control_main_layout = QVBoxLayout()
        control_main_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )

        # Scroll Area für Parameter
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Control Panel Content (scrollable)
        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )
        control_layout.setSpacing(10)
        self.control_panel.setLayout(control_layout)

        scroll_area.setWidget(self.control_panel)
        control_main_layout.addWidget(scroll_area)

        self.control_widget.setLayout(control_main_layout)

    def setup_navigation_panel(self):
        """
        Funktionsweise: Erstellt Navigation Panel - MODULAR
        Aufgabe: Fixed Navigation (nicht scrollbar)
        """
        self.navigation_panel = NavigationPanel(self.navigation_manager, show_tab_buttons=False)

        # Navigation direkt zum Control Widget hinzufügen (nicht scrollbar)
        if self.control_widget.layout():
            self.control_widget.layout().addWidget(self.navigation_panel)

    def setup_auto_simulation(self):
        """
        Funktionsweise: Erstellt Auto-Simulation Panel - MODULAR
        Aufgabe: Auto-Update Toggle, Manual Generation, Performance-Settings
        """
        generator_name = self.__class__.__name__.replace("Tab", "")
        self.auto_simulation_panel = AutoSimulationPanel(generator_name)

        # Signals verbinden
        self.auto_simulation_panel.auto_simulation_toggled.connect(self.toggle_auto_simulation)
        self.auto_simulation_panel.manual_generation_requested.connect(self.generate)
        self.auto_simulation_panel.performance_level_changed.connect(self.set_performance_level)

        # Panel zum Control Panel hinzufügen
        self.control_panel.layout().addWidget(self.auto_simulation_panel)

    @ui_update_handler("auto_simulation")
    def toggle_auto_simulation(self, enabled: bool):
        """
        Funktionsweise: Toggle für Auto-Simulation - HINZUGEFÜGT
        Parameter: enabled (bool) - Auto-Simulation aktiviert/deaktiviert
        Aufgabe: Aktiviert/Deaktiviert automatische Generation bei Parameter-Änderungen
        """
        self.auto_simulation_enabled = enabled
        self.logger.debug(f"Auto-simulation {'enabled' if enabled else 'disabled'}")

        # Timer stoppen wenn deaktiviert
        if not enabled and hasattr(self, 'auto_simulation_timer'):
            self.auto_simulation_timer.stop()

    @ui_update_handler("performance_level")
    def set_performance_level(self, level: str):
        """
        Funktionsweise: Setzt Performance-Level für Generation - HINZUGEFÜGT
        Parameter: level (str) - Performance-Level ("preview", "medium", "high")
        Aufgabe: Konfiguriert Generation-Performance
        """
        self.performance_level = level
        self.logger.debug(f"Performance level set to: {level}")

        # Optional: Performance-abhängige Konfiguration
        if level == "preview":
            # Schnelle Preview-Generierung
            pass
        elif level == "medium":
            # Mittlere Qualität
            pass
        elif level == "high":
            # Hohe Qualität
            pass

    @ui_update_handler("external_data")
    def on_external_data_updated(self, generator_type: str, data_key: str):
        """
        Funktionsweise: Standard-Handler für externe Daten-Updates - STANDARD-IMPLEMENTIERUNG
        Parameter: generator_type (str), data_key (str)
        Aufgabe: Reagiert auf Daten-Updates von anderen Generatoren
        """
        self.logger.debug(f"External data updated: {generator_type}.{data_key}")

        # Standard-Verhalten: Prüfe ob Update für diesen Tab relevant ist
        if hasattr(self, 'required_dependencies'):
            if data_key in self.required_dependencies:
                # Dependency-Check triggern wenn vorhanden
                if hasattr(self, 'check_input_dependencies'):
                    self.check_input_dependencies()

                # Display-Update triggern wenn Auto-Simulation aktiv
                if self.auto_simulation_enabled and hasattr(self, 'update_display_mode'):
                    self.update_display_mode()

    @ui_update_handler("cache_invalidated")
    def on_cache_invalidated(self, generator_type: str):
        """
        Funktionsweise: Standard-Handler für Cache-Invalidierung - STANDARD-IMPLEMENTIERUNG
        Parameter: generator_type (str)
        Aufgabe: Reagiert auf Cache-Invalidierung von anderen Generatoren
        """
        self.logger.debug(f"Cache invalidated for: {generator_type}")

        # Standard-Verhalten: Display-Update wenn betroffen
        if hasattr(self, 'update_display_mode') and not self.generation_in_progress:
            try:
                self.update_display_mode()
            except Exception as e:
                self.logger.debug(f"Display update after cache invalidation failed: {e}")

    @pyqtSlot(str, str)
    def on_orchestrator_generation_started(self, generator_type: str, lod_level: str):
        """
        Funktionsweise: Handler für Generation-Start vom Orchestrator - VEREINFACHT
        Parameter: generator_type, lod_level
        """
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        try:
            QMetaObject.invokeMethod(self, "_update_ui_generation_started",
                                     Qt.QueuedConnection, Q_ARG(str, lod_level))
        except Exception as e:
            self.logger.error(f"Failed to update UI for generation started: {e}")

    @pyqtSlot(str, str, bool)
    def on_orchestrator_generation_completed(self, generator_type: str, lod_level: str, success: bool):
        """
        Funktionsweise: Handler für Generation-Completion vom Orchestrator - VEREINFACHT
        Parameter: generator_type, lod_level, success
        """
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        try:
            QMetaObject.invokeMethod(self, "_update_ui_generation_completed",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, lod_level), Q_ARG(bool, success))
        except Exception as e:
            self.logger.error(f"Failed to update UI for generation completed: {e}")

    @pyqtSlot(str, str, int, str)
    def on_orchestrator_generation_progress(self, generator_type: str, lod_level: str, progress_percent: int,
                                            detail: str):
        """
        Funktionsweise: Handler für Generation-Progress vom Orchestrator - VEREINFACHT
        Parameter: generator_type, lod_level, progress_percent, detail
        """
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        try:
            QMetaObject.invokeMethod(self, "_update_ui_generation_progress",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, lod_level),
                                     Q_ARG(int, progress_percent),
                                     Q_ARG(str, detail))
        except Exception as e:
            self.logger.error(f"Failed to update UI for generation progress: {e}")

    @pyqtSlot(str, str)
    def on_lod_progression_completed(self, generator_type: str, final_lod: str):
        """
        Funktionsweise: Handler für LOD-Progression-Completion - VEREINFACHT
        Parameter: generator_type, final_lod
        """
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        try:
            QMetaObject.invokeMethod(self, "_update_ui_lod_progression_completed",
                                     Qt.QueuedConnection, Q_ARG(str, final_lod))
        except Exception as e:
            self.logger.error(f"Failed to update UI for LOD progression: {e}")

    @memory_critical_handler("generation")
    @core_generation_handler("tab_generation")
    def generate(self):
        """
        Funktionsweise: Standard-Fallback für Generation - STANDARD-IMPLEMENTIERUNG
        Aufgabe: Wird von Tabs überschrieben, die eigene Generation haben
        """
        # Check ob Tab eigene Generation hat
        tab_name = self.__class__.__name__.replace("Tab", "").lower()

        if hasattr(self, f'generate_{tab_name}_system'):
            # Tab hat eigene Generator-Methode
            generator_method = getattr(self, f'generate_{tab_name}_system')
            try:
                generator_method()
            except Exception as e:
                self.logger.error(f"Generation failed in {tab_name}: {e}")
                self.handle_generation_error(e)
        else:
            # Kein Generator verfügbar - Log Info
            self.logger.info(f"{self.__class__.__name__} has no generation capability")

            # Update Auto-Simulation Panel Status
            if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                self.auto_simulation_panel.set_generation_status("info", "No generation available for this tab")

    @pyqtSlot()
    def on_auto_simulation_triggered(self):
        """
        Standard Auto-Simulation Handler - kann von Sub-Classes überschrieben werden
        """
        if hasattr(self, 'generate') and not getattr(self, 'generation_in_progress', False):
            try:
                self.generate()
            except Exception as e:
                self.logger.error(f"Auto-simulation failed: {e}")

    def save_splitter_state(self):
        """
        Funktionsweise: Speichert Splitter-Position für Layout-Persistenz
        Aufgabe: Wird bei splitterMoved Signal aufgerufen
        """
        if hasattr(self, 'splitter') and self.splitter:
            sizes = self.splitter.sizes()
            # Optional: Speichere in Settings oder einfach als Instanz-Variable
            self.splitter_sizes = sizes
            self.logger.debug(f"Splitter state saved: {sizes}")

    def restore_splitter_state(self):
        """
        Funktionsweise: Stellt gespeicherte Splitter-Position wieder her
        Aufgabe: Kann beim Tab-Wechsel aufgerufen werden
        """
        if hasattr(self, 'splitter') and hasattr(self, 'splitter_sizes'):
            self.splitter.setSizes(self.splitter_sizes)
            self.logger.debug(f"Splitter state restored: {self.splitter_sizes}")

    @memory_critical_handler("timing")
    def start_generation_timing(self):
        """
        Funktionsweise: Startet Performance-Timing für Generation
        Aufgabe: Wird von generate() Methoden aufgerufen
        """
        import time
        self.generation_start_time = time.time()

    @memory_critical_handler("timing")
    def end_generation_timing(self, success: bool, error_message: str = ""):
        """
        Funktionsweise: Beendet Performance-Timing und loggt Results
        Parameter: success (bool), error_message (str)
        """
        if hasattr(self, 'generation_start_time') and self.generation_start_time:
            duration = time.time() - self.generation_start_time

            if success:
                self.logger.info(f"Generation completed successfully in {duration:.2f}s")
            else:
                self.logger.error(f"Generation failed after {duration:.2f}s: {error_message}")

            self.generation_start_time = None

    @ui_update_handler("error_handling")
    def handle_generation_error(self, exception: Exception):
        """
        Funktionsweise: Centralized Error-Handling für alle Generationen
        Parameter: exception (Exception)
        """
        error_msg = str(exception)
        self.logger.error(f"Generation error in {self.__class__.__name__}: {error_msg}")

        # Update UI Status
        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_status("error", error_msg)

    def setup_error_handling(self):
        """
        Funktionsweise: Setup für Error-Handling - MODULAR
        Aufgabe: Status-Anzeige und Error-Recovery
        """
        self.error_status = StatusIndicator("System Status")
        self.error_status.set_success("Ready")
        self.control_panel.layout().addWidget(self.error_status)

    def setup_signals(self):
        """
        Funktionsweise: Verbindet alle Signal-Slot Verbindungen - MODULAR
        Aufgabe: Cross-Tab Communication und Data-Updates
        """
        if self.data_manager:
            self.data_manager.data_updated.connect(self.on_external_data_updated)
            self.data_manager.cache_invalidated.connect(self.on_cache_invalidated)

    def create_standard_lod_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Standard-LOD-Control Panel für alle Generator-Tabs
        Aufgabe: Wiederverwendbares LOD-Control-Panel
        Return: QGroupBox mit LOD-Controls
        """
        panel = QGroupBox("Quality / LOD Control")
        layout = QVBoxLayout()

        # Target-LOD Selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Quality:"))

        self.target_lod_combo = QComboBox()
        self.target_lod_combo.addItems(["LOD64 (Fast Preview)", "LOD128 (Medium)", "LOD256 (High)", "FINAL (Best)"])
        self.target_lod_combo.setCurrentIndex(3)
        self.target_lod_combo.currentTextChanged.connect(self.on_target_lod_changed)
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

    def calculate_control_panel_width(self) -> int:
        """
        Funktionsweise: Berechnet optimale Control-Panel-Breite - KONSTANTEN
        Aufgabe: Dynamische Größenberechnung basierend auf Layout-Konstanten
        Return: int - Optimale Control-Panel-Breite
        """
        total_width = self.width() if self.width() > 0 else 1200
        target_width = int(total_width * self.layout_constants.CONTROL_RATIO)

        optimal_width = max(
            self.layout_constants.MIN_CONTROL_PANEL_WIDTH,
            min(
                self.layout_constants.MAX_CONTROL_PANEL_WIDTH,
                target_width - self.layout_constants.SCROLLBAR_MARGIN - self.layout_constants.CONTENT_MARGIN
            )
        )

        return optimal_width

    def create_visualization_controls(self):
        """
        Funktionsweise: Erstellt Standard-Visualization-Controls - DEFAULT IMPLEMENTATION
        Aufgabe: Basis-Controls für Heightmap-Display (von Sub-Classes erweitert)
        Return: QWidget mit Base-Visualization-Controls
        """
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Display Mode Button Group
        self.display_mode = QButtonGroup()

        # Standard Heightmap Mode
        self.heightmap_radio = QRadioButton("Height")
        self.heightmap_radio.setStyleSheet("font-size: 11px;")
        self.heightmap_radio.setChecked(True)
        self.heightmap_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.heightmap_radio, 0)
        layout.addWidget(self.heightmap_radio)

        widget.setLayout(layout)
        return widget

    @ui_update_handler("display_update")
    def update_display_mode(self):
        """
        Funktionsweise: Default Display-Mode Handler - MUSS von Sub-Classes überschrieben werden
        Aufgabe: Default-Implementation für Heightmap-Display
        """
        heightmap = self.data_manager.get_terrain_data("heightmap") if self.data_manager else None
        if heightmap is not None:
            current_display = self.get_current_display()
            if current_display:
                self._render_current_mode(0, current_display, heightmap, "heightmap")

    @dependency_handler("dependency_check")
    def check_input_dependencies(self):
        """Standard-Dependency-Check mit Error-Handling - ÜBERSCHREIBBAR"""
        if hasattr(self, 'required_dependencies') and hasattr(self, 'data_manager'):
            if self.data_manager:
                is_complete, missing = self.data_manager.check_dependencies(
                    self.__class__.__name__.replace("Tab", "").lower(),
                    self.required_dependencies
                )

                if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                    self.auto_simulation_panel.set_manual_button_enabled(is_complete)

                return is_complete
        return True

    @cleanup_handler("resource_cleanup")
    def cleanup_resources(self):
        """Resource-Cleanup mit Error-Protection - ERWEITERT"""

        if hasattr(self, 'generation_orchestrator') and self.generation_orchestrator:
            try:
                # Alle Orchestrator-Signals disconnecten
                self.generation_orchestrator.generation_started.disconnect(self.on_orchestrator_generation_started)
                self.generation_orchestrator.generation_completed.disconnect(self.on_orchestrator_generation_completed)
                self.generation_orchestrator.generation_progress.disconnect(self.on_orchestrator_generation_progress)

                if hasattr(self.generation_orchestrator, 'lod_progression_completed'):
                    self.generation_orchestrator.lod_progression_completed.disconnect(self.on_lod_progression_completed)

                self.logger.debug("Orchestrator signals disconnected successfully")

            except (TypeError, RuntimeError):
                # Signals bereits disconnected oder Orchestrator bereits deleted
                pass

        if hasattr(self, 'resource_tracker'):
            self.resource_tracker.force_cleanup_all()

        if hasattr(self, 'parameter_manager'):
            self.parameter_manager.cleanup()

        if hasattr(self, 'display_manager'):
            self.display_manager.last_display_hashes.clear()

    def _render_current_mode(self, mode_id: int, display, data, layer_type: str):
        """
        Funktionsweise: Zentraler Display-Renderer mit Change-Detection
        Aufgabe: Eliminiert doppelte Display-Update-Logik zwischen Tabs
        Parameter: mode_id, display, data, layer_type
        """
        display_id = f"{self.__class__.__name__}_{self.current_view}_{mode_id}"

        if self.display_manager.needs_update(display_id, data, layer_type):
            if hasattr(display, 'cleanup_textures'):
                display.cleanup_textures()

            display.update_display(data, layer_type)
            self.display_manager.mark_updated(display_id, data, layer_type)

            # Force Garbage Collection für große Updates
            if hasattr(data, 'nbytes') and data.nbytes > 50_000_000:
                import gc
                gc.collect()

    @pyqtSlot(str)
    def switch_view(self, view_type: str):
        """
        Funktionsweise: Wechselt zwischen 2D und 3D Ansicht mit Memory-Management
        Parameter: view_type ("2d" oder "3d")
        """
        old_display = self.get_current_display()
        if old_display:
            old_display.set_active(False)

        self.current_view = view_type

        if view_type == "2d":
            self.display_stack.setCurrentIndex(0)
            self.view_2d_button.button_type = "primary"
            self.view_3d_button.button_type = "secondary"
        else:
            self.display_stack.setCurrentIndex(1)
            self.view_2d_button.button_type = "secondary"
            self.view_3d_button.button_type = "primary"

        self.view_2d_button.setup_styling()
        self.view_3d_button.setup_styling()

        new_display = self.get_current_display()
        if new_display:
            new_display.set_active(True)

        self.update_display_mode()
        self.logger.debug(f"Switched to {view_type.upper()} view")

    def get_current_display(self):
        """
        Funktionsweise: Gibt aktuell aktives Display-Widget zurück
        Return: DisplayWrapper (MapDisplay2D oder MapDisplay3DWidget oder Fallback)
        """
        if self.current_view == "2d":
            return self.map_display_2d
        else:
            return self.map_display_3d

    @pyqtSlot(str)
    def on_target_lod_changed(self, combo_text: str):
        """
        Funktionsweise: Standard LOD-Änderungs-Handler
        Parameter: combo_text (str) - ComboBox-Text
        """
        if "LOD64" in combo_text:
            self.target_lod = "LOD64"
        elif "LOD128" in combo_text:
            self.target_lod = "LOD128"
        elif "LOD256" in combo_text:
            self.target_lod = "LOD256"
        elif "FINAL" in combo_text:
            self.target_lod = "FINAL"

        self.logger.info(f"Target LOD changed to: {self.target_lod}")

    # STANDARD ORCHESTRATOR UI-UPDATE METHODS (für Sub-Classes):

    @pyqtSlot(str)
    def _update_ui_generation_started(self, lod_level: str):
        """UI-Update für Generation-Start in Main-Thread - STANDARD"""
        if hasattr(self, 'generation_progress'):
            self.generation_progress.setValue(0)
            self.generation_progress.setFormat(f"Generating {lod_level}...")

    @pyqtSlot(str, bool)
    def _update_ui_generation_completed(self, lod_level: str, success: bool):
        """UI-Update für Generation-Completion in Main-Thread - STANDARD"""
        if success:
            self.available_lods.add(lod_level)
            if hasattr(self, 'generation_progress'):
                self.generation_progress.setValue(100)
                self.generation_progress.setFormat(f"{lod_level} Complete")
        else:
            if hasattr(self, 'generation_progress'):
                self.generation_progress.setFormat(f"{lod_level} Failed")

        if lod_level == self.target_lod:
            self.generation_in_progress = False
            self.end_generation_timing(success)

    @pyqtSlot(str, int, str)
    def _update_ui_generation_progress(self, lod_level: str, progress_percent: int, detail: str):
        """UI-Update für Generation-Progress in Main-Thread - STANDARD"""
        if hasattr(self, 'generation_progress'):
            self.generation_progress.setValue(progress_percent)
            self.generation_progress.setFormat(f"{lod_level} - {progress_percent}%")

    @pyqtSlot(str)
    def _update_ui_lod_progression_completed(self, final_lod: str):
        """UI-Update für LOD-Progression-Completion in Main-Thread - STANDARD"""
        self.generation_in_progress = False
        if hasattr(self, 'generation_progress'):
            self.generation_progress.setValue(100)
            self.generation_progress.setFormat("Complete")

        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_in_progress(False)