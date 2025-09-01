"""
Path: gui/tabs/base_tab.py

BaseMapTab ist die fundamentale Basis-Klasse für alle spezialisierten Map-Editor Tabs.
Sie implementiert ein standardisiertes 70/30 Layout-System, Cross-Tab-Kommunikation
und robustes UI-Management mit klarer Trennung zwischen UI-Layer und Business-Logic.

Kernverantwortlichkeiten:
- Standardisiertes 70/30 Layout mit QSplitter (Canvas links, Controls rechts)
- 2D/3D Display-Stack mit Toggle-Controls und Fallback-Management
- Parameter-UI-Controls als Proxy zum ParameterManager
- Display-Update-Koordination ohne eigene Business-Logic

Manager-Integration als Proxy:
- ParameterManager: UI-Parameter ↔ Storage-Parameter Synchronisation
- GenerationOrchestrator: Generation-Requests mit Parameter-Weiterleitung
- DataLODManager: Display-Data-Retrieval für UI-Updates
- NavigationManager: Tab-Navigation und Window-Geometry

Extensibility für Sub-Classes:
- Required: generator_type, required_dependencies, create_parameter_controls()
- Optional: create_visualization_controls(), update_display_mode(), check_input_dependencies()
"""

from PyQt5.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QTimer
)
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QFrame, QSizePolicy, QLabel, QVBoxLayout,
    QScrollArea, QSplitter, QGroupBox, QStackedWidget, QRadioButton,
    QButtonGroup, QProgressBar
)

import logging
import time
import hashlib
import weakref
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from gui.widgets.widgets import BaseButton, StatusIndicator, DisplayWrapper, NavigationPanel

# Map Display Import mit Fallback
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

def get_error_handler():
    """Zentrale Error-Handler Import mit Fallback"""
    try:
        from gui.utils.error_handler import error_handler
        return error_handler
    except ImportError:
        def noop_decorator(func):
            return func
        return noop_decorator

error_handler = get_error_handler()


class BaseMapTab(QWidget):
    """
    Basis-Klasse für alle Map-Editor Tabs mit einheitlicher Manager-Integration
    und standardisiertem 70/30 Layout-System.
    """

    # Outgoing Signals (UI → Managers)
    generate_requested = pyqtSignal(str)  # generator_type
    parameter_ui_changed = pyqtSignal(str, str, object)  # generator_type, param_name, value
    display_mode_changed = pyqtSignal(str)  # display_mode

    def __init__(self, data_lod_manager=None, parameter_manager=None,
                 navigation_manager=None, shader_manager=None, generation_orchestrator=None):
        super().__init__()

        # Manager-Referenzen (können None sein für Fallback-Verhalten)
        self.data_lod_manager = data_lod_manager
        self.parameter_manager = parameter_manager
        self.navigation_manager = navigation_manager
        self.generation_orchestrator = generation_orchestrator

        # Core Attributes - müssen von Sub-Classes gesetzt werden
        self.generator_type = getattr(self, 'generator_type', 'unknown')
        self.required_dependencies = getattr(self, 'required_dependencies', [])

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Layout-System
        self.layout_config = LayoutConfiguration()

        # UI Components (werden in setup_ui() initialisiert)
        self.splitter = None
        self.canvas_container = None
        self.control_panel = None
        self.control_widget = None
        self.navigation_panel = None
        self.status_display = None

        # Display-System
        self.display_stack = None
        self.map_display_2d = None
        self.map_display_3d = None
        self.current_view = "2d"
        self.view_2d_button = None
        self.view_3d_button = None
        self.visualization_controls = None

        # Generation-System (vereinfacht)
        self.generation_active = False

        # Setup-Sequenz
        self.setup_ui()
        self.setup_manager_connections()

    def setup_ui(self):
        """Hauptmethode für UI-Setup mit modularer Struktur"""
        self.logger.debug("Setting up UI components")

        try:
            self._create_main_layout()
            self._create_canvas_area()
            self._create_control_panel()
            self._create_navigation_panel()
            self._create_status_display()

            # Sub-Class spezifische Controls
            self.create_parameter_controls()

            self.logger.debug("UI setup completed successfully")

        except Exception as e:
            self.logger.error(f"UI setup failed: {e}")
            self._create_fallback_ui(str(e))

    def _create_main_layout(self):
        """Erstellt 70/30 Splitter-Layout"""
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Canvas Container (70%)
        self.canvas_container = QFrame()
        self.canvas_container.setFrameStyle(QFrame.StyledPanel)
        self.canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Control Widget (30%)
        control_width = self._calculate_optimal_control_width()
        self.control_widget = QFrame()
        self.control_widget.setFrameStyle(QFrame.StyledPanel)
        self.control_widget.setFixedWidth(control_width)
        self.control_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Splitter für resizable Layout
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.canvas_container)
        self.splitter.addWidget(self.control_widget)
        self.splitter.setSizes([700, 300])  # 70/30 ratio
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    def _create_canvas_area(self):
        """Erstellt Canvas mit 2D/3D Display-Stack"""
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        # View Toggle Buttons
        button_layout = QHBoxLayout()

        self.view_2d_button = BaseButton("2D View", "primary")
        self.view_2d_button.clicked.connect(lambda: self.switch_view("2d"))
        button_layout.addWidget(self.view_2d_button)

        self.view_3d_button = BaseButton("3D View", "secondary")
        self.view_3d_button.clicked.connect(lambda: self.switch_view("3d"))
        button_layout.addWidget(self.view_3d_button)

        button_layout.addStretch()

        # Visualization Controls (überschreibbar von Sub-Classes)
        self.visualization_controls = self.create_visualization_controls()
        if self.visualization_controls:
            button_layout.addWidget(self.visualization_controls)

        canvas_layout.addLayout(button_layout)

        # Display Stack
        self.display_stack = QStackedWidget()
        self._create_display_widgets()
        canvas_layout.addWidget(self.display_stack)

        self.canvas_container.setLayout(canvas_layout)

    def _create_display_widgets(self):
        """Erstellt 2D und 3D Display-Widgets mit Fallback"""
        # 2D Display
        if MAP_DISPLAY_2D_AVAILABLE:
            try:
                display_2d = MapDisplay2D()
                self.map_display_2d = DisplayWrapper(display_2d)
            except Exception as e:
                self.logger.error(f"2D Display creation failed: {e}")
                self.map_display_2d = self._create_fallback_display("2D Display\n(Creation failed)")
        else:
            self.map_display_2d = self._create_fallback_display("2D Display\n(Not available)")

        self.display_stack.addWidget(self.map_display_2d.display)

        # 3D Display
        if MAP_DISPLAY_3D_AVAILABLE:
            try:
                display_3d = MapDisplay3DWidget()
                self.map_display_3d = DisplayWrapper(display_3d)
            except Exception as e:
                self.logger.error(f"3D Display creation failed: {e}")
                self.map_display_3d = self._create_fallback_display("3D Display\n(Creation failed)")
        else:
            self.map_display_3d = self._create_fallback_display("3D Display\n(Not available)")

        self.display_stack.addWidget(self.map_display_3d.display)

    def _create_fallback_display(self, text: str) -> DisplayWrapper:
        """Erstellt Fallback-Display für fehlerhafte Display-Creation"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; color: #6c757d;")
        return DisplayWrapper(label)

    def _create_control_panel(self):
        """Erstellt scrollbares Control Panel"""
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(5, 5, 5, 5)

        # Scrollable Area für Parameter
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Control Panel Content
        self.control_panel = QWidget()
        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(5, 5, 5, 5)
        panel_layout.setSpacing(10)
        self.control_panel.setLayout(panel_layout)

        scroll_area.setWidget(self.control_panel)
        control_layout.addWidget(scroll_area)

        self.control_widget.setLayout(control_layout)

    def _create_navigation_panel(self):
        """Erstellt Navigation Panel (nicht scrollbar)"""
        try:
            if self.navigation_manager:
                self.navigation_panel = NavigationPanel(self.navigation_manager)
                self.control_widget.layout().addWidget(self.navigation_panel)
        except Exception as e:
            self.logger.error(f"Navigation panel creation failed: {e}")

    def _create_status_display(self):
        """Erstellt Status-Display"""
        try:
            self.status_display = StatusIndicator("System Status")
            self.status_display.set_success("Ready")

            if self.control_panel and self.control_panel.layout():
                self.control_panel.layout().addWidget(self.status_display)

        except Exception as e:
            self.logger.error(f"Status display creation failed: {e}")

    def _create_fallback_ui(self, error_message: str):
        """Erstellt minimale Fallback-UI bei Setup-Fehlern"""
        try:
            fallback_layout = QVBoxLayout()
            error_label = QLabel(f"UI Setup Error:\n{error_message}")
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: red; padding: 20px; font-size: 12px;")
            fallback_layout.addWidget(error_label)
            self.setLayout(fallback_layout)
        except Exception:
            pass  # Auch Fallback fehlgeschlagen

    def setup_manager_connections(self):
        """Verbindet Signals mit verfügbaren Managern"""
        try:
            # ParameterManager Connections
            if self.parameter_manager:
                self.parameter_manager.parameter_changed.connect(self.on_parameter_changed)
                self.parameter_ui_changed.connect(self.parameter_manager.set_parameter)

            # DataLODManager Connections
            if self.data_lod_manager:
                self.data_lod_manager.data_updated.connect(self.on_data_updated)

            # GenerationOrchestrator Connections
            if self.generation_orchestrator:
                self.generate_requested.connect(self.generation_orchestrator.request_generation)
                self.generation_orchestrator.generation_started.connect(self.on_generation_started)
                self.generation_orchestrator.generation_completed.connect(self.on_generation_completed)
                self.generation_orchestrator.generation_progress.connect(self.on_generation_progress)

            self.logger.debug("Manager connections established")

        except Exception as e:
            self.logger.error(f"Manager connection setup failed: {e}")

    def _calculate_optimal_control_width(self) -> int:
        """Berechnet optimale Breite für Control Panel"""
        total_width = self.width() if self.width() > 0 else 1200
        target_width = int(total_width * 0.3)  # 30%

        return max(
            self.layout_config.MIN_CONTROL_WIDTH,
            min(self.layout_config.MAX_CONTROL_WIDTH, target_width - 40)
        )

    # =============================================================================
    # DISPLAY MANAGEMENT
    # =============================================================================

    @pyqtSlot(str)
    def switch_view(self, view_type: str):
        """Wechselt zwischen 2D und 3D Ansicht"""
        # Altes Display deaktivieren
        current_display = self.get_current_display()
        if current_display and hasattr(current_display, 'set_active'):
            try:
                current_display.set_active(False)
            except Exception as e:
                self.logger.debug(f"Display deactivation warning: {e}")

        # View umschalten
        self.current_view = view_type

        if view_type == "2d":
            self.display_stack.setCurrentIndex(0)
            self._update_button_styles("2d")
        else:
            self.display_stack.setCurrentIndex(1)
            self._update_button_styles("3d")

        # Neues Display aktivieren
        new_display = self.get_current_display()
        if new_display and hasattr(new_display, 'set_active'):
            try:
                new_display.set_active(True)
            except Exception as e:
                self.logger.debug(f"Display activation warning: {e}")

        # Display aktualisieren
        self.update_display_mode()
        self.logger.debug(f"Switched to {view_type} view")

    def _update_button_styles(self, active_view: str):
        """Aktualisiert Button-Styles für aktive View"""
        try:
            if active_view == "2d":
                self.view_2d_button.button_type = "primary"
                self.view_3d_button.button_type = "secondary"
            else:
                self.view_2d_button.button_type = "secondary"
                self.view_3d_button.button_type = "primary"

            # Styling neu anwenden
            for button in [self.view_2d_button, self.view_3d_button]:
                if hasattr(button, 'setup_styling'):
                    button.setup_styling()

        except Exception as e:
            self.logger.debug(f"Button style update failed: {e}")

    def get_current_display(self):
        """Gibt aktuell aktives Display zurück"""
        try:
            return self.map_display_2d if self.current_view == "2d" else self.map_display_3d
        except Exception as e:
            self.logger.debug(f"Get current display failed: {e}")
            return None

    @error_handler
    def update_display_mode(self):
        """Display-Update über DataLODManager"""
        try:
            if self.data_lod_manager:
                # Hole beste verfügbare Daten vom DataLODManager
                heightmap = self.data_lod_manager.get_terrain_data("heightmap")
                if heightmap is not None:
                    current_display = self.get_current_display()
                    if current_display and hasattr(current_display, 'update_display'):
                        # Nutze DataLODManager's Change-Detection
                        display_id = f"{self.__class__.__name__}_{self.current_view}_heightmap"
                        if self.data_lod_manager.display_update_manager.needs_update(display_id, heightmap, "heightmap"):
                            current_display.update_display(heightmap, "heightmap")
                            self.data_lod_manager.display_update_manager.mark_updated(display_id, heightmap, "heightmap")

        except Exception as e:
            self.logger.debug(f"Display mode update failed: {e}")

    # =============================================================================
    # GENERATION SYSTEM (nur UI-Proxy)
    # =============================================================================

    @error_handler
    def generate(self):
        """Generation-Request an GenerationOrchestrator (nur Signal)"""
        if self.generation_active:
            self.logger.warning("Generation already active")
            return

        self.generation_active = True

        try:
            # Status-Update
            if self.status_display:
                self.status_display.set_pending("Generation requested...")

            # Signal an GenerationOrchestrator (keine eigene Logic)
            self.generate_requested.emit(self.generator_type)

        except Exception as e:
            self.generation_active = False
            self.logger.error(f"Generation request failed: {e}")
            if self.status_display:
                self.status_display.set_error(f"Generation failed: {e}")

    # =============================================================================
    # SIGNAL HANDLERS (vereinfacht)
    # =============================================================================

    @pyqtSlot(str, str, object)
    def on_parameter_changed(self, generator_type: str, param_name: str, value):
        """Handler für Parameter-Änderungen vom ParameterManager"""
        if generator_type == self.generator_type:
            self.update_parameter_ui(param_name, value)

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """Handler für Data-Updates vom DataLODManager"""
        try:
            # Dependency-Check
            if hasattr(self, 'required_dependencies') and data_key in self.required_dependencies:
                self.check_input_dependencies()

            # Display-Update bei relevanten Daten
            if generator_type == self.generator_type or data_key in getattr(self, 'display_data_keys', []):
                self.update_display_mode()

        except Exception as e:
            self.logger.debug(f"Data update handling failed: {e}")

    @pyqtSlot(str)
    def on_generation_started(self, generator_type: str):
        """Handler für Generation-Start"""
        if generator_type == self.generator_type:
            if self.status_display:
                self.status_display.set_pending("Generation in progress...")

    @pyqtSlot(str, bool)
    def on_generation_completed(self, generator_type: str, success: bool):
        """Handler für Generation-Completion"""
        if generator_type == self.generator_type:
            self.generation_active = False

            if self.status_display:
                if success:
                    self.status_display.set_success("Generation completed")
                else:
                    self.status_display.set_error("Generation failed")

    @pyqtSlot(str, int)
    def on_generation_progress(self, generator_type: str, progress: int):
        """Handler für Generation-Progress"""
        if generator_type == self.generator_type:
            if self.status_display:
                self.status_display.set_pending(f"Generating... {progress}%")

    # =============================================================================
    # EXTENSIBILITY INTERFACE FÜR SUB-CLASSES
    # =============================================================================

    def create_parameter_controls(self):
        """MUSS von Sub-Classes implementiert werden - erstellt Parameter-UI-Controls"""
        self.logger.warning(f"{self.__class__.__name__} should implement create_parameter_controls()")

    def create_visualization_controls(self):
        """KANN von Sub-Classes überschrieben werden - erstellt Display-Mode-Controls"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Standard Heightmap Mode
        self.display_mode_group = QButtonGroup()

        height_radio = QRadioButton("Height")
        height_radio.setChecked(True)
        height_radio.toggled.connect(self.update_display_mode)
        self.display_mode_group.addButton(height_radio, 0)
        layout.addWidget(height_radio)

        widget.setLayout(layout)
        return widget

    def check_input_dependencies(self) -> bool:
        """KANN von Sub-Classes überschrieben werden - prüft Input-Dependencies"""
        try:
            if self.data_lod_manager and hasattr(self, 'required_dependencies'):
                # Delegiert an DataLODManager's Dependency-System
                return self.data_lod_manager.check_dependencies(self.generator_type, 1)
        except Exception as e:
            self.logger.debug(f"Dependency check failed: {e}")

        return True  # Fallback: Dependencies erfüllt

    def update_parameter_ui(self, param_name: str, value):
        """KANN von Sub-Classes überschrieben werden - aktualisiert Parameter-UI"""
        self.logger.debug(f"Parameter UI update: {param_name} = {value}")

    # =============================================================================
    # RESOURCE MANAGEMENT (delegiert an DataLODManager)
    # =============================================================================

    @error_handler
    def cleanup_resources(self):
        """Systematische Resource-Cleanup über DataLODManager"""
        self.logger.debug("Cleaning up resources")

        try:
            # Signal-Disconnections
            self._disconnect_all_signals()

            # Delegiere Cleanup an DataLODManager
            if self.data_lod_manager and hasattr(self.data_lod_manager, 'resource_tracker'):
                self.data_lod_manager.resource_tracker.cleanup_resources()

            # Generation-State Reset
            self.generation_active = False

            self.logger.debug("Resource cleanup completed")

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")

    def reset_data(self):
        """Setzt alle generierten Daten zurück über DataLODManager"""
        try:
            if self.data_lod_manager:
                # Delegiere an DataLODManager
                self.data_lod_manager.clear_generator_data(self.generator_type)

            # Status zurücksetzen
            if self.status_display:
                self.status_display.set_success("Ready")

            self.logger.info(f"Data reset completed for {self.generator_type}")

        except Exception as e:
            self.logger.error(f"Data reset failed: {e}")

    def _disconnect_all_signals(self):
        """Disconnected alle Signal-Verbindungen sicher"""
        managers_and_signals = [
            (self.parameter_manager, ['parameter_changed']),
            (self.data_lod_manager, ['data_updated']),
            (self.generation_orchestrator, ['generation_started', 'generation_completed', 'generation_progress'])
        ]

        for manager, signals in managers_and_signals:
            if manager:
                for signal_name in signals:
                    try:
                        signal = getattr(manager, signal_name, None)
                        if signal:
                            signal.disconnect()
                    except (TypeError, RuntimeError):
                        pass  # Signal bereits disconnected


# =============================================================================
# HELPER CLASSES
# =============================================================================

@dataclass
class LayoutConfiguration:
    """Konfiguration für Layout-Parameter"""
    MIN_CONTROL_WIDTH: int = 280
    MAX_CONTROL_WIDTH: int = 400
    CANVAS_RATIO: float = 0.7
    CONTROL_RATIO: float = 0.3