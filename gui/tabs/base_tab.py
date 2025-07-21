"""
Path: gui/tabs/base_tab.py

Funktionsweise: Basis-Klasse für alle Map-Editor Tabs
- Standardisiertes 70/30 Layout (Canvas links, Controls rechts)
- Gemeinsame Auto-Simulation Controls für alle Tabs
- Input-Status Display (verfügbare Dependencies)
- Observer-Pattern für Cross-Tab Updates
- Einheitliche Navigation (Prev/Next Buttons)
- Performance-optimiertes Debouncing-System
- Error-Handling und Resource-Management
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging
from typing import Optional, List, Dict, Any

from gui.widgets.widgets import BaseButton, AutoSimulationPanel, ProgressBar, StatusIndicator
from gui.widgets.map_display_2d import MapDisplay2D

def get_base_error_decorators():
    """
    Funktionsweise: Lazy Loading von Base Tab Error Decorators
    Aufgabe: Lädt Core-Generation und Cleanup Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import core_generation_handler, cleanup_handler, ui_navigation_handler
        return core_generation_handler, cleanup_handler, ui_navigation_handler
    except ImportError:
        # No-op Fallback
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

# Decorators laden
core_generation_handler, cleanup_handler, ui_navigation_handler = get_base_error_decorators()

class BaseMapTab(QWidget):
    """
    Funktionsweise: Basis-Klasse für alle Map-Editor Tabs mit gemeinsamen Features
    Aufgabe: Stellt Standard-Layout und gemeinsame Funktionalität bereit
    Kommunikation: Signals für data_updated, parameter_changed, validation_status_changed
    """

    # Signals für Cross-Tab Communication
    data_updated = pyqtSignal(str, str)  # (generator_type, data_key)
    parameter_changed = pyqtSignal(str, dict)  # (generator_type, parameters)
    validation_status_changed = pyqtSignal(str, bool, list)  # (generator_type, is_valid, messages)
    generation_completed = pyqtSignal(str, bool)  # (generator_type, success)

    def __init__(self, data_manager, navigation_manager, shader_manager):
        super().__init__()

        # Core Manager References
        self.data_manager = data_manager
        self.navigation_manager = navigation_manager
        self.shader_manager = shader_manager

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Auto-Simulation State
        self.auto_simulation_enabled = False
        self.auto_simulation_timer = QTimer()
        self.auto_simulation_timer.setSingleShot(True)
        self.auto_simulation_timer.timeout.connect(self.on_auto_simulation_triggered)

        # Performance Monitoring
        self.performance_timer = QTimer()
        self.generation_start_time = None

        # UI Components (werden von Sub-Classes gesetzt)
        self.canvas_container = None
        self.control_panel = None
        self.map_display = None
        self.auto_simulation_panel = None
        self.navigation_panel = None
        self.input_status_panel = None

        # Setup Common UI
        self.setup_common_ui()
        self.setup_signals()

        # Error Handling
        self.setup_error_handling()

    @ui_navigation_handler
    def setup_common_ui(self):
        """
        Funktionsweise: Erstellt standardisiertes 70/30 Layout für alle Tabs
        Aufgabe: Canvas links (70%), Control Panel rechts (30%) mit Error-Protection
        Besonderheit: Error Handler schützt vor UI-Setup Fehlern
        """
        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # Canvas Container (70%)
        self.canvas_container = QFrame()
        self.canvas_container.setFrameStyle(QFrame.StyledPanel)
        self.canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Map Display Widget
        self.map_display = MapDisplay2D()
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.addWidget(self.map_display)
        self.canvas_container.setLayout(canvas_layout)

        # Control Panel Container (30%)
        control_widget = QFrame()
        control_widget.setFrameStyle(QFrame.StyledPanel)
        control_widget.setMaximumWidth(400)
        control_widget.setMinimumWidth(350)
        control_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Control Panel ist QVBoxLayout für Sub-Classes
        self.control_panel = QVBoxLayout()
        self.control_panel.setContentsMargins(10, 10, 10, 10)
        self.control_panel.setSpacing(10)

        # Scroll Area für Control Panel (falls zu viele Controls)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarNever)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_content.setLayout(self.control_panel)
        scroll_area.setWidget(scroll_content)

        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(scroll_area)
        control_widget.setLayout(control_layout)

        # Splitter für resizable Layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas_container)
        splitter.addWidget(control_widget)
        splitter.setSizes([700, 300])  # 70/30 initial ratio
        splitter.setCollapsible(0, False)  # Canvas nicht kollabierbar
        splitter.setCollapsible(1, False)  # Control Panel nicht kollabierbar

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # Setup Common Panels
        self.setup_auto_simulation()

    def setup_auto_simulation(self):
        """
        Funktionsweise: Erstellt Auto-Simulation Panel für alle Tabs
        Aufgabe: Auto-Update Toggle, Manual Generation, Performance-Settings
        """
        generator_name = self.__class__.__name__.replace("Tab", "")
        self.auto_simulation_panel = AutoSimulationPanel(generator_name)

        # Signals verbinden
        self.auto_simulation_panel.auto_simulation_toggled.connect(self.toggle_auto_simulation)
        self.auto_simulation_panel.manual_generation_requested.connect(self.generate_terrain)
        self.auto_simulation_panel.performance_level_changed.connect(self.set_performance_level)

        # Panel zum Control Panel hinzufügen
        self.control_panel.addWidget(self.auto_simulation_panel)

    def setup_input_status(self):
        """
        Funktionsweise: Erstellt Input-Status Panel für Dependencies
        Aufgabe: Zeigt verfügbare Input-Daten von anderen Generatoren
        """
        self.input_status_panel = InputStatusPanel()
        self.control_panel.addWidget(self.input_status_panel)

        # Manual Generate Button Reference für Enable/Disable
        self.manual_generate_button = self.auto_simulation_panel.manual_button

    def setup_navigation(self):
        """
        Funktionsweise: Erstellt Navigation Panel mit Prev/Next Buttons
        Aufgabe: Tab-Navigation über NavigationManager
        """
        self.navigation_panel = NavigationPanel(self.navigation_manager)

        # Navigation Panel am Ende des Control Panels hinzufügen
        self.control_panel.addStretch()  # Push navigation to bottom
        self.control_panel.addWidget(self.navigation_panel)

    def setup_signals(self):
        """
        Funktionsweise: Verbindet alle Signal-Slot Verbindungen
        Aufgabe: Cross-Tab Communication und Data-Updates
        """
        # Data Manager Signals
        if self.data_manager:
            self.data_manager.data_updated.connect(self.on_external_data_updated)
            self.data_manager.cache_invalidated.connect(self.on_cache_invalidated)

        # Performance Monitoring
        self.performance_timer.timeout.connect(self.update_performance_metrics)

    def setup_error_handling(self):
        """
        Funktionsweise: Setup für Error-Handling und Recovery
        Aufgabe: Graceful Error-Handling mit User-Feedback
        """
        # Error Status Display
        self.error_status = StatusIndicator("System Status")
        self.error_status.set_success("Ready")
        self.control_panel.addWidget(self.error_status)

    @pyqtSlot(bool)
    def toggle_auto_simulation(self, enabled: bool):
        """
        Funktionsweise: Toggle Auto-Simulation on/off
        Parameter: enabled (bool)
        """
        self.auto_simulation_enabled = enabled
        self.logger.debug(f"Auto-simulation {'enabled' if enabled else 'disabled'}")

        if enabled:
            # Start Performance Monitoring
            self.performance_timer.start(1000)  # Every second
        else:
            self.performance_timer.stop()

    @pyqtSlot(str)
    def set_performance_level(self, level: str):
        """
        Funktionsweise: Setzt Performance-Level im DataManager
        Parameter: level ("live", "preview", "final")
        """
        if self.data_manager:
            self.data_manager.set_lod_level(level)
            self.logger.info(f"Performance level set to: {level}")

    @pyqtSlot()
    def on_auto_simulation_triggered(self):
        """
        Funktionsweise: Slot für Auto-Simulation Timer
        Aufgabe: Triggert Generation wenn Auto-Simulation aktiviert
        """
        if self.auto_simulation_enabled:
            self.logger.debug("Auto-simulation triggered")
            self.generate_terrain()

    @pyqtSlot(str, str)
    def on_external_data_updated(self, generator_type: str, data_key: str):
        """
        Funktionsweise: Slot für externe Data-Updates
        Parameter: generator_type (str), data_key (str)
        """
        # Sub-Classes können diesen Slot überschreiben für spezifische Reaktionen
        self.logger.debug(f"External data updated: {generator_type}.{data_key}")

        # Input Status aktualisieren
        if hasattr(self, 'input_status_panel'):
            self.input_status_panel.update_available_data(generator_type, data_key)

    @pyqtSlot(str)
    def on_cache_invalidated(self, generator_type: str):
        """
        Funktionsweise: Slot für Cache-Invalidation
        Parameter: generator_type (str)
        """
        self.logger.info(f"Cache invalidated for: {generator_type}")

        # Auto-Regeneration triggern wenn dieser Generator betroffen ist
        current_generator = self.__class__.__name__.replace("Tab", "").lower()
        if generator_type == current_generator and self.auto_simulation_enabled:
            self.auto_simulation_timer.start(2000)  # 2s delay für Cache-Invalidation

    @pyqtSlot()
    def update_performance_metrics(self):
        """
        Funktionsweise: Aktualisiert Performance-Metriken
        Aufgabe: Zeigt Generation-Zeit und Memory-Usage
        """
        if hasattr(self, 'auto_simulation_panel'):
            # Memory Usage vom DataManager holen
            memory_usage = self.data_manager.get_memory_usage() if self.data_manager else {}

            # Performance-Metriken an Auto-Simulation Panel weitergeben
            # (würde normalerweise detailliertere Metriken enthalten)
            pass

    def start_generation_timing(self):
        """
        Funktionsweise: Startet Generation-Timer für Performance-Messung
        """
        import time
        self.generation_start_time = time.time()

        if hasattr(self, 'auto_simulation_panel'):
            self.auto_simulation_panel.set_generation_in_progress(True)

    def end_generation_timing(self, success: bool, error_message: str = ""):
        """
        Funktionsweise: Beendet Generation-Timer und zeigt Ergebnis
        Parameter: success (bool), error_message (str)
        """
        generation_time = None
        if self.generation_start_time:
            import time
            generation_time = time.time() - self.generation_start_time
            self.generation_start_time = None

        if hasattr(self, 'auto_simulation_panel'):
            self.auto_simulation_panel.set_generation_in_progress(False)

            if success:
                time_text = f"completed in {generation_time:.2f}s" if generation_time else "completed"
                self.auto_simulation_panel.set_generation_status("success", time_text)
                self.error_status.set_success("Generation successful")
            else:
                self.auto_simulation_panel.set_generation_status("error", error_message)
                self.error_status.set_error(f"Generation failed: {error_message}")

        # Signal emittieren
        generator_type = self.__class__.__name__.replace("Tab", "").lower()
        self.generation_completed.emit(generator_type, success)

    def handle_generation_error(self, error: Exception):
        """
        Funktionsweise: Zentrale Error-Behandlung für Generation-Fehler
        Parameter: error (Exception)
        """
        error_message = str(error)
        self.logger.error(f"Generation error: {error_message}")

        # Error-Status setzen
        self.end_generation_timing(False, error_message)

        # Error-Dialog anzeigen (optional)
        if hasattr(error, '__cause__') and error.__cause__:
            detailed_message = f"{error_message}\n\nCause: {error.__cause__}"
        else:
            detailed_message = error_message

        # Critical Errors als Dialog anzeigen
        if "critical" in error_message.lower() or "fatal" in error_message.lower():
            QMessageBox.critical(self, "Critical Error", detailed_message)

    def validate_inputs(self) -> tuple[bool, List[str]]:
        """
        Funktionsweise: Validiert Input-Dependencies vor Generation
        Return: (is_valid: bool, missing_dependencies: List[str])
        """
        # Default Implementation - Sub-Classes überschreiben für spezifische Validation
        return True, []

    @cleanup_handler
    def cleanup_resources(self):
        """
        Funktionsweise: Cleanup-Methode für Resource-Management mit Error-Protection
        Aufgabe: Wird beim Tab-Wechsel oder Schließen aufgerufen
        Besonderheit: Error Handler verhindert Resource-Leaks bei Cleanup-Fehlern
        """
        # Timer stoppen
        if self.auto_simulation_timer.isActive():
            self.auto_simulation_timer.stop()

        if self.performance_timer.isActive():
            self.performance_timer.stop()

        # GPU-Resources freigeben (falls verwendet)
        if hasattr(self, 'shader_manager') and self.shader_manager:
            self.shader_manager.cleanup_tab_resources(self.__class__.__name__)

        self.logger.debug("Resources cleaned up")

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt aktuelle Parameter-Werte
        Return: dict mit aktuellen Parametern
        Muss von Sub-Classes überschrieben werden
        """
        return {}

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Funktionsweise: Setzt Parameter-Werte
        Parameter: parameters (dict)
        Muss von Sub-Classes überschrieben werden
        """
        pass

    def export_current_view(self, filename: str) -> bool:
        """
        Funktionsweise: Exportiert aktuelle Map-Ansicht
        Parameter: filename (str)
        Return: Success (bool)
        """
        try:
            if self.map_display:
                return self.map_display.save_current_view(filename)
            return False
        except Exception as e:
            self.logger.error(f"Failed to export view: {e}")
            return False

    @core_generation_handler("base_terrain")
    def generate_terrain(self):
        """
        Funktionsweise: Hauptmethode für Generation - MUSS von Sub-Classes überschrieben werden
        Aufgabe: Implementiert Generator-spezifische Logic mit Error-Protection
        Besonderheit: Error Handler schützt vor Generation-Fehlern und Memory-Issues
        """
        raise NotImplementedError("Sub-classes must implement generate_terrain()")

    def closeEvent(self, event):
        """Override für Cleanup beim Schließen"""
        self.cleanup_resources()
        super().closeEvent(event)


class InputStatusPanel(QGroupBox):
    """
    Funktionsweise: Panel für Input-Status Display
    Aufgabe: Zeigt verfügbare Dependencies von anderen Generatoren
    """

    def __init__(self):
        super().__init__("Input Status")
        self.available_data = {}
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Input-Status"""
        layout = QVBoxLayout()

        # Available Data Summary
        self.data_summary = QLabel("Available Data: Checking...")
        layout.addWidget(self.data_summary)

        # Detailed Status (würde normalerweise expandable sein)
        self.detailed_status = QTextEdit()
        self.detailed_status.setMaximumHeight(100)
        self.detailed_status.setReadOnly(True)
        layout.addWidget(self.detailed_status)

        self.setLayout(layout)
        self.update_display()

    def update_available_data(self, generator_type: str, data_key: str):
        """
        Funktionsweise: Aktualisiert verfügbare Daten
        Parameter: generator_type (str), data_key (str)
        """
        if generator_type not in self.available_data:
            self.available_data[generator_type] = []

        if data_key not in self.available_data[generator_type]:
            self.available_data[generator_type].append(data_key)

        self.update_display()

    def update_display(self):
        """Aktualisiert Display mit verfügbaren Daten"""
        total_items = sum(len(items) for items in self.available_data.values())

        if total_items == 0:
            self.data_summary.setText("Available Data: None")
            self.detailed_status.setText("No data available from other generators.")
        else:
            self.data_summary.setText(f"Available Data: {total_items} items from {len(self.available_data)} generators")

            # Detailed Status
            details = []
            for generator, items in self.available_data.items():
                details.append(f"{generator.title()}: {', '.join(items)}")

            self.detailed_status.setText('\n'.join(details))


class NavigationPanel(QGroupBox):
    """
    Funktionsweise: Panel für Tab-Navigation
    Aufgabe: Prev/Next Navigation über NavigationManager
    """

    def __init__(self, navigation_manager):
        super().__init__("Navigation")
        self.navigation_manager = navigation_manager
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Navigation"""
        layout = QHBoxLayout()

        # Previous Button
        self.prev_button = BaseButton("← Previous", "secondary")
        self.prev_button.clicked.connect(self.go_previous)
        layout.addWidget(self.prev_button)

        # Next Button
        self.next_button = BaseButton("Next →", "primary")
        self.next_button.clicked.connect(self.go_next)
        layout.addWidget(self.next_button)

        # Progress Indicator
        self.progress_bar = ProgressBar()

        # Vertical Layout für Buttons + Progress
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)

        # Initial Button States aktualisieren
        self.update_navigation_state()

    @pyqtSlot()
    def go_previous(self):
        """Navigate zu Previous Tab"""
        if self.navigation_manager:
            success = self.navigation_manager.navigate_previous()
            if not success:
                self.prev_button.setEnabled(False)

    @pyqtSlot()
    def go_next(self):
        """Navigate zu Next Tab"""
        if self.navigation_manager:
            success = self.navigation_manager.navigate_next()
            if not success:
                self.next_button.setEnabled(False)

    def update_navigation_state(self):
        """
        Funktionsweise: Aktualisiert Navigation-State basierend auf aktueller Position
        """
        if not self.navigation_manager:
            return

        # Button States (würde normalerweise vom NavigationManager kommen)
        current_tab = self.navigation_manager.get_current_tab_name()

        # Previous Button
        self.prev_button.setEnabled(self.navigation_manager.can_navigate_previous())

        # Next Button
        self.next_button.setEnabled(self.navigation_manager.can_navigate_next())

        # Progress Bar
        self.progress_bar.set_tab_active(current_tab)

    def mark_tab_completed(self, tab_name: str):
        """Markiert Tab als completed in Progress Bar"""
        self.progress_bar.set_tab_completed(tab_name)