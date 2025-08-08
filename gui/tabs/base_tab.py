"""
Path: gui/tabs/base_tab.py

=========================================================================================
BASEMAPTAB
=========================================================================================

ARCHITECTURE OVERVIEW:
----------------------

BaseMapTab ist die fundamentale Basis-Klasse für alle spezialisierten Map-Editor Tabs.
Sie implementiert ein standardisiertes 70/30 Layout-System, Cross-Tab-Kommunikation,
automatische Generation-Workflows und robustes Resource-Management.

DESIGN PRINCIPLES:
------------------
1. MODULARE ARCHITEKTUR: Jede Funktionalität ist in separate, testbare Methoden aufgeteilt
2. RESOURCE SAFETY: Systematisches Tracking und Cleanup aller UI-Ressourcen
3. ERROR RESILIENCE: Jede kritische Operation hat Exception-Handling
4. SIGNAL ORCHESTRATION: Standardisierte Cross-Tab-Kommunikation über Signals
5. MEMORY EFFICIENCY: Change-Detection verhindert unnötige Display-Updates
6. EXTENSIBILITY: Sub-Classes können jede Funktionalität selektiv überschreiben

CORE COMPONENTS:
----------------

┌─────────────────┬─────────────────────────────────────────────────────────────┐
│ COMPONENT       │ RESPONSIBILITY                                              │
├─────────────────┼─────────────────────────────────────────────────────────────┤
│ Layout System   │ 70/30 Canvas/Control Split mit dynamischer Größenanpassung  │
│ Display Stack   │ 2D/3D View-Toggle mit Fallback-Management                   │
│ Auto-Simulation │ Parameter-Change-Detection und Auto-Generation              │
│ Orchestrator    │ Integration mit GenerationOrchestrator für LOD-Progression  │
│ Resource Track  │ Memory-Leak-Prevention durch systematisches Cleanup         │
│ Signal Hub      │ Cross-Tab-Communication und Data-Change-Notifications       │
└─────────────────┴─────────────────────────────────────────────────────────────┘

SIGNAL ARCHITECTURE:
--------------------

OUTGOING SIGNALS (BaseMapTab → Other Components):
• data_updated(generator_type: str, data_key: str)
  - Emittiert wenn Tab neue Daten generiert hat
  - Andere Tabs können darauf reagieren und ihre Dependencies prüfen

• parameter_changed(generator_type: str, parameters: dict)
  - Emittiert bei Parameter-Änderungen
  - Triggert Auto-Simulation in anderen Tabs falls aktiviert

• validation_status_changed(generator_type: str, is_valid: bool, messages: list)
  - Emittiert bei Dependency-Status-Änderungen
  - Navigation-Manager kann Tab-Verfügbarkeit entsprechend anpassen

• generation_completed(generator_type: str, success: bool)
  - Emittiert nach Abschluss einer Generation
  - Ermöglicht Chain-Generationen zwischen Tabs

INCOMING SIGNALS (Other Components → BaseMapTab):
• data_manager.data_updated → on_external_data_updated()
  - Reagiert auf Daten-Updates von anderen Generatoren
  - Prüft Dependencies und triggert Display-Updates

• data_manager.cache_invalidated → on_cache_invalidated()
  - Reagiert auf Cache-Invalidierung
  - Refresht Display falls nötig

• generation_orchestrator.* → _on_generation_*()
  - Empfängt LOD-Progression-Updates vom Orchestrator
  - Aktualisiert UI-Status und Progress-Anzeigen

UI ARCHITECTURE:
----------------

MAIN LAYOUT STRUCTURE:
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ BaseMapTab (QWidget)                                                                  │
│ ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│ │ QHBoxLayout (main_layout)                                                         │ │
│ │ ┌───────────────────────────┬───────────────────────────────────────────────────┐ │ │
│ │ │ Canvas Container (70%)    │ Control Widget (30%)                              │ │ │
│ │ │ ┌───────────────────────┐ │ ┌───────────────────────────────────────────────┐ │ │ │
│ │ │ │ View Toggle Buttons   │ │ │ QVBoxLayout                                   │ │ │ │
│ │ │ └───────────────────────┘ │ │ ┌───────────────────────────────────────┐     │ │ │ │
│ │ │ ┌───────────────────────┐ │ │ │ QScrollArea                           │     │ │ │ │
│ │ │ │ QStackedWidget        │ │ │ │ ┌───────────────────────────────────┐ │     │ │ │ │
│ │ │ │ ├─ 2D Display         │ │ │ │ │ Control Panel (QWidget)           │ │     │ │ │ │
│ │ │ │ └─ 3D Display         │ │ │ │ │ ┌─ Parameter Controls (Sub-Class) │ │     │ │ │ │
│ │ │ └───────────────────────┘ │ │ │ │ ├─ Auto-Simulation Panel          │ │     │ │ │ │
│ │ └───────────────────────────┘ │ │ │ └─ Error Status Display           │ │     │ │ │ │
│ │                               │ │ └───────────────────────────────────┘ │     │ │ │ │
│ │                               │ └───────────────────────────────────────┘     │ │ │ │
│ │                               │ ┌───────────────────────────────────────────┐ │ │ │ │
│ │                               │ │ Navigation Panel (Fixed, not scrollable)  │ │ │ │ │
│ │                               │ └───────────────────────────────────────────┘ │ │ │ │
│ │                               └───────────────────────────────────────────────┘ │ │ │
│ └─────────────────────────────────────────────────────────────────────────────────┘ │ │
│                                                                                     │ │
│ QSplitter (Canvas ↔ Control resizable, 70/30 ratio maintained)                      │ │
└─────────────────────────────────────────────────────────────────────────────────────┘ │
                                                                                        │
└───────────────────────────────────────────────────────────────────────────────────────┘

CONTROL PANEL COMPOSITION:
┌─────────────────────────────────────┐
│ QScrollArea (scrollable content)    │
│ ┌─────────────────────────────────┐ │
│ │ Control Panel (QWidget)         │ │  ← Sub-Classes add content here
│ │ ┌─ Custom Parameter Controls    │ │  ← Via control_panel.layout()
│ │ ├─ Auto-Simulation Panel        │ │  ← Standardized across all tabs
│ │ └─ Error Status Display         │ │  ← Shows dependency/generation status
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│ Navigation Panel (fixed)            │  ← Always visible, not scrollable
└─────────────────────────────────────┘

LIFECYCLE MANAGEMENT:
---------------------

INITIALIZATION SEQUENCE:
1. __init__(): Core attribute setup, dependency injection
2. setup_ui(): UI structure creation with error handling
   ├─ setup_layout_structure(): 70/30 split, splitter configuration
   ├─ setup_display_stack(): 2D/3D views with fallback handling
   ├─ setup_control_panel(): Scrollable parameter area
   ├─ setup_navigation_panel(): Fixed navigation (not scrollable)
   ├─ setup_auto_simulation(): Auto-generation controls
   └─ setup_error_handling(): Status display and error recovery
3. setup_signals(): Cross-tab communication setup
4. setup_standard_orchestrator_handlers(): LOD-progression integration

CLEANUP SEQUENCE:
1. cleanup_resources(): Systematic resource deallocation
   ├─ _safe_disconnect_orchestrator_signals(): Exception-safe disconnects
   ├─ _safe_disconnect_data_manager_signals(): Exception-safe disconnects
   ├─ resource_tracker.force_cleanup_all(): Memory leak prevention
   ├─ parameter_manager.cleanup(): Parameter state cleanup
   └─ Force garbage collection for large arrays

DATA RESET SEQUENCE (NEW):
1. reset_all_data(): Complete data reset while keeping parameters
   ├─ data_manager.clear_all_data(): Clear all generated data
   ├─ _reset_display_content(): Clear display content
   ├─ Generation state reset: Stop timers, reset flags
   └─ UI status reset: Ready state, clear error messages

GENERATION WORKFLOW:
--------------------

STANDARD GENERATION FLOW:
1. User Action/Auto-Trigger → generate()
2. State validation and reset (prevents double-generation)
3. Check for tab-specific generator method (generate_[tab_name]_system)
4. Start performance timing
5. Execute generation with error handling
6. Update UI status and progress (via orchestrator signals)
7. End timing and cleanup state

AUTO-SIMULATION FLOW:
1. Parameter change detected → auto_simulation_timer starts
2. Timer expires → on_auto_simulation_triggered()
3. For auto-generation tabs (geology+): Check dependencies
4. If dependencies satisfied: Trigger generation
5. Update dependent tabs via signals

AUTO-TAB-GENERATION (NEW FEATURE):
- Tabs: Geology, Settlement, Weather, Water, Biome
- Behavior: Automatic generation on tab switch OR manual generate click
- Dependency: Only executes if required dependencies are available
- Integration: check_input_dependencies() → generate() → signal emission

DISPLAY SYSTEM:
---------------

VIEW SWITCHING ARCHITECTURE:
• 2D/3D Toggle: QStackedWidget with DisplayWrapper abstraction
• Fallback System: Graceful degradation when display classes unavailable
• State Management: Active/inactive display coordination
• Memory Management: Texture cleanup on view switches

DISPLAY UPDATE OPTIMIZATION:
• Change Detection: Hash-based comparison prevents unnecessary updates
• Resource Tracking: Systematic cleanup of display resources
• Memory Management: Garbage collection for large array updates
• Error Resilience: Failed updates don't crash the UI

RENDERING PIPELINE:
1. update_display_mode() → _render_current_mode()
2. Change detection via DisplayUpdateManager
3. Cleanup old textures if update needed
4. Apply new data to display
5. Mark as updated in cache
6. Force GC for large arrays (>50MB)

EXTENSIBILITY FOR SUB-CLASSES:
------------------------------

REQUIRED IMPLEMENTATIONS:
• generate_[tab_name]_system(): Core generation logic
• required_dependencies: List[str]: Input dependency specification
• create_parameter_controls(): UI controls for generator parameters

OPTIONAL OVERRIDES:
• create_visualization_controls(): Custom display mode buttons
• update_display_mode(): Custom display rendering logic
• check_input_dependencies(): Custom dependency validation
• on_external_data_updated(): Custom cross-tab data handling

STANDARD INTEGRATION POINTS:
• control_panel.layout(): Add custom parameter controls
• setup_standard_orchestrator_handlers(generator_type): LOD integration
• Signal connections: Connect to data_updated, parameter_changed etc.

PARAMETER INTEGRATION:
• parameter_manager: Handles slider persistence and change detection
• Auto-simulation: Automatic triggering on parameter changes
• Validation: Dependency checking and UI state updates

ERROR HANDLING STRATEGY:
------------------------

DEFENSIVE PROGRAMMING:
• Every critical operation wrapped in try/except
• Graceful degradation: Continue operation even if non-critical parts fail
• Resource safety: Cleanup guaranteed even on exceptions
• User feedback: Clear error messages in UI status displays

ERROR CATEGORIES:
• Setup Errors: UI creation failures → Minimal fallback UI
• Display Errors: Rendering failures → Continue with degraded display
• Generation Errors: Algorithm failures → Clear error status, enable retry
• Resource Errors: Memory/cleanup failures → Force cleanup and logging

MEMORY MANAGEMENT:
------------------

RESOURCE TRACKING SYSTEM:
• ResourceTracker: Systematic registration and cleanup of all resources
• Weak references: Prevent circular dependencies
• Cleanup functions: Custom cleanup logic per resource type
• Age-based cleanup: Automatic cleanup of old resources

MEMORY OPTIMIZATION:
• Display change detection: Prevents unnecessary texture uploads
• Large array handling: Force garbage collection for >50MB arrays
• Resource pooling: Reuse display resources where possible
• Exception safety: Cleanup guaranteed even on errors

PERFORMANCE CONSIDERATIONS:
---------------------------

UI RESPONSIVENESS:
• Auto-simulation debouncing: Prevents excessive generation triggers
• Progressive loading: LOD-based generation with incremental updates
• Background processing: Non-blocking generation workflows
• Change detection: Skip rendering if data hasn't changed

MEMORY EFFICIENCY:
• Reference sharing: numpy arrays shared between components (no copies)
• Lazy loading: Resources created only when needed
• Systematic cleanup: Prevent memory leaks through resource tracking
• Garbage collection: Forced GC for large operations

THREAD SAFETY:
• Signal-based communication: Qt's thread-safe signal system
• UI updates: All UI changes in main thread via pyqtSlot
• Resource access: Coordinated through data_manager
• State synchronization: Generation state managed centrally

=========================================================================================
USAGE EXAMPLES FOR SUB-CLASSES:
=========================================================================================

MINIMAL TAB IMPLEMENTATION:
```python
class TerrainTab(BaseMapTab):
    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator):
        # Required dependencies for this tab
        self.required_dependencies = []  # No dependencies for terrain (base generator)

        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)

        # Setup orchestrator integration
        self.setup_standard_orchestrator_handlers("terrain")

        # Create parameter controls
        self.create_parameter_controls()

    def create_parameter_controls(self):
        # Add custom parameter sliders/controls to self.control_panel.layout()
        pass

    def generate_terrain_system(self):
        # Core generation logic
        # Should emit self.data_updated.emit("terrain", "heightmap") when done
        pass
```

ADVANCED TAB WITH CUSTOM DISPLAYS:
```python
class GeologyTab(BaseMapTab):
    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator):
        self.required_dependencies = ["heightmap", "slopemap"]  # Requires terrain data

        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)
        self.setup_standard_orchestrator_handlers("geology")
        self.create_parameter_controls()

    def create_visualization_controls(self):
        # Override to add custom display modes (rock_map, hardness_map)
        widget = super().create_visualization_controls()

        self.rock_radio = QRadioButton("Rock Types")
        self.hardness_radio = QRadioButton("Hardness")

        # Add to layout and connect signals
        return widget

    def update_display_mode(self):
        # Custom rendering logic for geology-specific displays
        if self.rock_radio.isChecked():
            rock_data = self.data_manager.get_geology_data("rock_map")
            current_display = self.get_current_display()
            if rock_data is not None and current_display:
                self._render_current_mode(1, current_display, rock_data, "rock_types")
        # ... etc

    def generate_geology_system(self):
        # Geology generation logic
        # Emits data_updated signals for rock_map, hardness_map etc.
        pass
```

INTEGRATION CHECKLIST FOR NEW TABS:
• ✓ Define required_dependencies list
• ✓ Call setup_standard_orchestrator_handlers(generator_type)
• ✓ Implement generate_[tab_name]_system() method
• ✓ Create parameter controls and add to control_panel.layout()
• ✓ Override visualization controls if custom display modes needed
• ✓ Override update_display_mode() if custom rendering needed
• ✓ Emit data_updated signals when generation completes
• ✓ Handle parameter changes and auto-simulation appropriately

=========================================================================================

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


class BaseMapTab(QWidget):
    """
    Funktionsweise: Basis-Klasse für alle Map-Editor Tabs - TEIL 1 REFACTORED
    Aufgabe: Core Setup, Layout-Management und Resource-Lifecycle
    Kommunikation: Signals für data_updated, parameter_changed, validation_status_changed
    ÄNDERUNGEN: Obsolete LOD/Performance-Attribute entfernt, Data-Reset hinzugefügt
    """

    # Signals für Cross-Tab Communication (unverändert)
    data_updated = pyqtSignal(str, str)  # (generator_type, data_key)
    parameter_changed = pyqtSignal(str, dict)  # (generator_type, parameters)
    validation_status_changed = pyqtSignal(str, bool, list)  # (generator_type, is_valid, messages)
    generation_completed = pyqtSignal(str, bool)

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        super().__init__()

        # KRITISCHE Attribute ZUERST initialisieren
        self.data_manager = data_manager
        self.navigation_manager = navigation_manager
        self.shader_manager = shader_manager
        self.generation_orchestrator = generation_orchestrator

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Layout-Konstanten
        self.layout_constants = LayoutConstants()

        # Resource-Management - ERWEITERT
        self.resource_tracker = ResourceTracker()
        self.display_manager = DisplayUpdateManager()

        # Auto-Simulation State - VEREINFACHT
        self.auto_simulation_enabled = False
        self.auto_simulation_timer = QTimer()
        self.auto_simulation_timer.setSingleShot(True)
        self.auto_simulation_timer.timeout.connect(self.on_auto_simulation_triggered)

        # Generation State - VEREINFACHT (LOD-Management in data_manager)
        self.generation_start_time = None
        self.generation_in_progress = False
        self.tab_generator_type = None

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

        # UI Setup mit modularer Reihenfolge
        self.setup_ui()
        self.setup_signals()

    def setup_ui(self):
        """
        Funktionsweise: Hauptmethode für UI-Setup - REFACTORED in modulare Struktur
        Aufgabe: Koordiniert alle UI-Setup-Schritte mit Error-Handling
        ÄNDERUNGEN: Konsistente Resource-Tracker-Integration
        """
        self.logger.debug("setup_ui() started")

        try:
            self.setup_layout_structure()
            self.setup_display_stack()
            self.setup_control_panel()
            self.setup_navigation_panel()
            self.setup_auto_simulation()
            self.setup_error_handling()

            self.logger.debug("setup_ui() completed successfully")

        except Exception as e:
            self.logger.error(f"UI setup failed: {e}")
            self.handle_setup_error(e)

    def setup_layout_structure(self):
        """
        Funktionsweise: Erstellt Basis-Layout-Struktur - MODULAR
        Aufgabe: 70/30 Layout mit Splitter und dynamischer Größenberechnung
        ÄNDERUNGEN: Konsistente Konstanten-Nutzung, Resource-Tracking
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

        # Control Widget (30%) - VERBESSERTE Größenberechnung
        optimal_width = self.calculate_control_panel_width()
        self.control_widget = QFrame()
        self.control_widget.setFrameStyle(QFrame.StyledPanel)
        self.control_widget.setMaximumWidth(optimal_width)
        self.control_widget.setMinimumWidth(
            min(self.layout_constants.MIN_CONTROL_PANEL_WIDTH, optimal_width)
        )
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

        # Resource-Tracking für Splitter
        self.resource_tracker.register_resource(
            self.splitter, "layout_splitter"
        )

        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    def setup_control_panel(self):
        """
        Funktionsweise: Erstellt Control Panel mit Scroll-Area - MODULAR
        Aufgabe: Scrollable Parameter-Controls mit konsistenten Margins
        ÄNDERUNGEN: Konsistente Layout-Konstanten-Nutzung
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

        # Resource-Tracking für Scroll-Area
        self.resource_tracker.register_resource(
            scroll_area, "control_scroll_area"
        )

    @cleanup_handler("resource_cleanup")
    def cleanup_resources(self):
        """
        Funktionsweise: ERWEITERTE Resource-Cleanup mit Complete Data Reset
        Aufgabe: Systematisches Cleanup aller Ressourcen + Optional Data-Reset
        ÄNDERUNGEN: Force-Cleanup, Exception-sicherer Disconnect, Memory-Management
        """
        self.logger.debug("Starting comprehensive resource cleanup")

        # 1. Signal-Disconnects (Exception-sicher)
        self._safe_disconnect_orchestrator_signals()
        self._safe_disconnect_data_manager_signals()

        # 2. Timer-Cleanup
        if hasattr(self, 'auto_simulation_timer') and self.auto_simulation_timer:
            self.auto_simulation_timer.stop()
            self.auto_simulation_timer.deleteLater()

        # 3. Resource-Tracker Cleanup
        if hasattr(self, 'resource_tracker'):
            self.resource_tracker.force_cleanup_all()

        # 4. Parameter-Manager Cleanup
        if hasattr(self, 'parameter_manager'):
            try:
                self.parameter_manager.cleanup()
            except Exception as e:
                self.logger.debug(f"Parameter manager cleanup warning: {e}")

        # 5. Display-Manager Cleanup
        if hasattr(self, 'display_manager'):
            self.display_manager.last_display_hashes.clear()

        # 6. Generation State Reset
        self.generation_in_progress = False
        self.generation_start_time = None

        # 7. Force Garbage Collection für große Arrays
        import gc
        gc.collect()

        self.logger.debug("Resource cleanup completed")

    def reset_all_data(self):
        """
        Funktionsweise: NEUE Methode - Complete Data Reset für Hauptmenü-Rückkehr
        Aufgabe: Löscht alle Daten aber behält Parameter (wie Programm-Neustart)
        SICHERHEIT: 100% - Parameter bleiben erhalten, nur Generierungsdaten werden gelöscht
        """
        self.logger.info("Performing complete data reset (keeping parameters)")

        try:
            # 1. Data-Manager Reset
            if self.data_manager:
                self.data_manager.clear_all_data()
                self.logger.debug("Data manager cleared")

            # 2. Display-State Reset
            if hasattr(self, 'display_manager'):
                self.display_manager.last_display_hashes.clear()

            # 3. Generation-State Reset
            self.generation_in_progress = False
            self.generation_start_time = None

            # 4. Auto-Simulation Stop
            if hasattr(self, 'auto_simulation_timer'):
                self.auto_simulation_timer.stop()

            # 5. Display-Content Reset (schwarze/leere Displays)
            self._reset_display_content()

            # 6. UI-Status Reset
            if hasattr(self, 'error_status') and self.error_status:
                self.error_status.set_success("Ready")

            if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                self.auto_simulation_panel.set_generation_status("info", "Data reset - ready for new generation")

            # NOTE: Parameter-Manager wird NICHT resettet (Parameter bleiben erhalten)

            self.logger.info("Complete data reset finished successfully")

        except Exception as e:
            self.logger.error(f"Data reset failed: {e}")
            raise

    def calculate_control_panel_width(self) -> int:
        """
        Funktionsweise: Berechnet optimale Control-Panel-Breite - KONSTANTEN-BASIERT
        Aufgabe: Dynamische Größenberechnung mit konsistenten Layout-Konstanten
        Return: int - Optimale Control-Panel-Breite
        ÄNDERUNGEN: Vollständige Nutzung der Layout-Konstanten
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

        # PRIVATE HELPER-METHODEN für Cleanup

    def _safe_disconnect_orchestrator_signals(self):
        """Exception-sichere Orchestrator Signal-Disconnects"""
        if not hasattr(self, 'generation_orchestrator') or not self.generation_orchestrator:
            return

        signals_to_disconnect = [
            ('generation_started', 'on_orchestrator_generation_started'),
            ('generation_completed', 'on_orchestrator_generation_completed'),
            ('generation_progress', 'on_orchestrator_generation_progress'),
            ('lod_progression_completed', 'on_lod_progression_completed')
        ]

        for signal_name, handler_name in signals_to_disconnect:
            try:
                signal = getattr(self.generation_orchestrator, signal_name, None)
                handler = getattr(self, handler_name, None)
                if signal and handler:
                    signal.disconnect(handler)
            except (TypeError, RuntimeError, AttributeError):
                # Signal bereits disconnected oder Objekt bereits deleted
                pass

        self.logger.debug("Orchestrator signals disconnected safely")

    def _safe_disconnect_data_manager_signals(self):
        """Exception-sichere Data-Manager Signal-Disconnects"""
        if not self.data_manager:
            return

        try:
            self.data_manager.data_updated.disconnect(self.on_external_data_updated)
            self.data_manager.cache_invalidated.disconnect(self.on_cache_invalidated)
        except (TypeError, RuntimeError):
            # Signals bereits disconnected
            pass

        self.logger.debug("Data manager signals disconnected safely")

    def _reset_display_content(self):
        """Setzt Display-Inhalte zurück (schwarze/leere Anzeige)"""
        current_display = self.get_current_display()
        if current_display and hasattr(current_display, 'clear_display'):
            try:
                current_display.clear_display()
            except Exception as e:
                self.logger.debug(f"Display content reset warning: {e}")

    def handle_setup_error(self, exception: Exception):
        """Error-Handler für UI-Setup Probleme"""
        error_msg = f"UI setup failed: {str(exception)}"
        self.logger.error(error_msg)

        # Minimal-UI falls möglich
        try:
            if not hasattr(self, 'layout') or not self.layout():
                minimal_layout = QVBoxLayout()
                error_label = QLabel(f"UI Setup Error:\n{error_msg}")
                error_label.setAlignment(Qt.AlignCenter)
                error_label.setStyleSheet("color: red; padding: 20px;")
                minimal_layout.addWidget(error_label)
                self.setLayout(minimal_layout)
        except Exception:
            pass  # Auch Minimal-UI fehlgeschlagen

    def setup_display_stack(self):
        """
        Funktionsweise: Erstellt 2D/3D Display-Stack - REFACTORED
        Aufgabe: View-Toggle und Display-Erstellung mit verbessertem Resource-Tracking
        ÄNDERUNGEN: Abstrahierte Fallback-Logik, konsistente Error-Behandlung
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

        # Display-Erstellung mit abstrahierter Fallback-Logik
        self._create_2d_display()
        self._create_3d_display()

        canvas_layout.addWidget(self.display_stack)
        self.canvas_container.setLayout(canvas_layout)

        # Resource-Tracking für Display-Stack
        self.resource_tracker.register_resource(
            self.display_stack, "display_stack"
        )

    def _create_2d_display(self):
        """
        Funktionsweise: Erstellt 2D Display mit verbessertem Resource-Tracking
        ÄNDERUNGEN: Abstrahierte Fallback-Logik, konsistente Error-Behandlung
        """
        display_2d = self._create_display_with_fallback(
            display_type="2d",
            available_flag=MAP_DISPLAY_2D_AVAILABLE,
            display_class_creator=lambda: MapDisplay2D(),
            fallback_text="2D Display\n(MapDisplay2D not available)",
            fallback_style="background-color: #ecf0f1; border: 1px solid #bdc3c7;"
        )

        self.map_display_2d = DisplayWrapper(display_2d)
        self.display_stack.addWidget(self.map_display_2d.display)

    def _create_3d_display(self):
        """
        Funktionsweise: Erstellt 3D Display mit verbessertem Resource-Tracking
        ÄNDERUNGEN: Abstrahierte Fallback-Logik, konsistente Error-Behandlung
        """
        display_3d = self._create_display_with_fallback(
            display_type="3d",
            available_flag=MAP_DISPLAY_3D_AVAILABLE,
            display_class_creator=lambda: MapDisplay3DWidget(),
            fallback_text="3D Display\n(MapDisplay3DWidget not available)",
            fallback_style="background-color: #f8f9fa; border: 1px solid #bdc3c7;"
        )

        self.map_display_3d = DisplayWrapper(display_3d)
        self.display_stack.addWidget(self.map_display_3d.display)

    def _create_display_with_fallback(self, display_type: str, available_flag: bool,
                                      display_class_creator, fallback_text: str, fallback_style: str):
        """
        Funktionsweise: NEUE Methode - Abstrahierte Display-Erstellung mit Fallback
        Aufgabe: Eliminiert duplizierte Fallback-Logik zwischen 2D/3D Display-Erstellung
        Parameter: display_type, available_flag, display_class_creator, fallback_text, fallback_style
        Return: Display-Objekt oder Fallback-Label
        """
        try:
            if available_flag:
                display = display_class_creator()

                # Resource-Tracking mit Display-spezifischem Cleanup
                self.resource_tracker.register_resource(
                    display, f"{display_type}_display",
                    cleanup_func=lambda d=display: self._cleanup_display(d)
                )

                self.logger.debug(f"Map Display {display_type.upper()} created successfully")
                return display
            else:
                # Availability-Fallback
                fallback = self._create_fallback_label(fallback_text, fallback_style)
                self.logger.debug(f"{display_type.upper()} fallback created (not available)")
                return fallback

        except Exception as e:
            # Exception-Fallback
            self.logger.error(f"{display_type.upper()} Display creation failed: {e}")
            error_text = f"{display_type.upper()} Display\n(Error loading)"
            fallback = self._create_fallback_label(error_text, fallback_style)
            return fallback

    def _create_fallback_label(self, text: str, style: str) -> QLabel:
        """Erstellt standardisierte Fallback-Labels für Displays"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(style)
        return label

    def _cleanup_display(self, display):
        """Display-spezifisches Cleanup"""
        try:
            if hasattr(display, 'cleanup_resources'):
                display.cleanup_resources()
            elif hasattr(display, 'cleanup_textures'):
                display.cleanup_textures()
        except Exception as e:
            self.logger.debug(f"Display cleanup warning: {e}")

    def setup_navigation_panel(self):
        """
        Funktionsweise: Erstellt Navigation Panel - REFACTORED
        Aufgabe: Fixed Navigation (nicht scrollbar) mit Error-Handling
        ÄNDERUNGEN: Verbesserte Error-Behandlung, Resource-Tracking
        """
        try:
            self.navigation_panel = NavigationPanel(
                self.navigation_manager,
                show_tab_buttons=False
            )

            # Navigation direkt zum Control Widget hinzufügen (nicht scrollbar)
            if self.control_widget.layout():
                self.control_widget.layout().addWidget(self.navigation_panel)

            # Resource-Tracking
            self.resource_tracker.register_resource(
                self.navigation_panel, "navigation_panel"
            )

            self.logger.debug("Navigation panel created successfully")

        except Exception as e:
            self.logger.error(f"Navigation panel setup failed: {e}")
            # Continue without navigation panel

    def setup_auto_simulation(self):
        """
        Funktionsweise: Erstellt Auto-Simulation Panel - REFACTORED
        Aufgabe: Auto-Update Toggle, Manual Generation, verbesserte Status-Anzeige
        ÄNDERUNGEN: Robustere Error-Statusanzeige, bessere Integration
        """
        try:
            generator_name = self.__class__.__name__.replace("Tab", "")
            self.auto_simulation_panel = AutoSimulationPanel(generator_name)

            # Signals verbinden
            self.auto_simulation_panel.auto_simulation_toggled.connect(self.toggle_auto_simulation)
            self.auto_simulation_panel.manual_generation_requested.connect(self.generate)
            # NOTE: performance_level_changed ENTFERNT (obsolet)

            # Panel zum Control Panel hinzufügen
            if self.control_panel and self.control_panel.layout():
                self.control_panel.layout().addWidget(self.auto_simulation_panel)

            # Resource-Tracking
            self.resource_tracker.register_resource(
                self.auto_simulation_panel, "auto_simulation_panel"
            )

            self.logger.debug("Auto-simulation panel created successfully")

        except Exception as e:
            self.logger.error(f"Auto-simulation panel setup failed: {e}")
            # Continue without auto-simulation panel

    def setup_error_handling(self):
        """
        Funktionsweise: Setup für Error-Handling - REFACTORED
        Aufgabe: Status-Anzeige und Error-Recovery mit robuster Integration
        ÄNDERUNGEN: Verbesserte Error-Recovery, konsistente Status-Updates
        """
        try:
            self.error_status = StatusIndicator("System Status")
            self.error_status.set_success("Ready")

            if self.control_panel and self.control_panel.layout():
                self.control_panel.layout().addWidget(self.error_status)

            # Resource-Tracking
            self.resource_tracker.register_resource(
                self.error_status, "error_status"
            )

            self.logger.debug("Error handling setup completed")

        except Exception as e:
            self.logger.error(f"Error handling setup failed: {e}")
            # Continue without error status display

    def create_visualization_controls(self):
        """
        Funktionsweise: Erstellt Standard-Visualization-Controls - DEFAULT IMPLEMENTATION
        Aufgabe: Basis-Controls für Heightmap-Display (von Sub-Classes erweitert)
        Return: QWidget mit Base-Visualization-Controls
        ÄNDERUNGEN: Robustere Widget-Erstellung, Error-Handling
        """
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            # Display Mode Button Group
            self.display_mode = QButtonGroup()

            # Standard Heightmap Mode
            self.heightmap_radio = QRadioButton("Height")
            self.heightmap_radio.setStyleSheet("font-size: 11px;")
            self.heightmap_radio.setChecked(True)
            self.heightmap_radio.toggled.connect(self.update_display_mode)
            self.display_mode.addButton(self.heightmap_radio, 0)
            layout.addWidget(self.heightmap_radio)

        except Exception as e:
            self.logger.error(f"Visualization controls creation failed: {e}")
            # Return empty widget if creation fails

        widget.setLayout(layout)
        return widget

    @pyqtSlot(str)
    def switch_view(self, view_type: str):
        """
        Funktionsweise: Wechselt zwischen 2D und 3D Ansicht - REFACTORED
        Parameter: view_type ("2d" oder "3d")
        ÄNDERUNGEN: Display-State-Corruption behoben, Null-Checks hinzugefügt
        """
        # Sicherer Display-State-Übergang
        old_display = self.get_current_display()
        if old_display and hasattr(old_display, 'set_active'):
            try:
                old_display.set_active(False)
            except Exception as e:
                self.logger.debug(f"Old display deactivation warning: {e}")

        # View-State aktualisieren
        self.current_view = view_type

        # Display-Stack und Button-States aktualisieren
        try:
            if view_type == "2d":
                self.display_stack.setCurrentIndex(0)
                self._update_view_button_states("2d")
            else:
                self.display_stack.setCurrentIndex(1)
                self._update_view_button_states("3d")

            # Neues Display aktivieren
            new_display = self.get_current_display()
            if new_display and hasattr(new_display, 'set_active'):
                try:
                    new_display.set_active(True)
                except Exception as e:
                    self.logger.debug(f"New display activation warning: {e}")

            # Display-Mode Update
            self.update_display_mode()
            self.logger.debug(f"Switched to {view_type.upper()} view")

        except Exception as e:
            self.logger.error(f"View switch to {view_type} failed: {e}")

    def _update_view_button_states(self, active_view: str):
        """Button-States für View-Toggle aktualisieren"""
        try:
            if active_view == "2d":
                self.view_2d_button.button_type = "primary"
                self.view_3d_button.button_type = "secondary"
            else:
                self.view_2d_button.button_type = "secondary"
                self.view_3d_button.button_type = "primary"

            # Styling aktualisieren
            if hasattr(self.view_2d_button, 'setup_styling'):
                self.view_2d_button.setup_styling()
            if hasattr(self.view_3d_button, 'setup_styling'):
                self.view_3d_button.setup_styling()

        except Exception as e:
            self.logger.debug(f"Button state update warning: {e}")

    def get_current_display(self):
        """
        Funktionsweise: Gibt aktuell aktives Display-Widget zurück - REFACTORED
        Return: DisplayWrapper (MapDisplay2D oder MapDisplay3DWidget oder Fallback)
        ÄNDERUNGEN: Null-Checks hinzugefügt, robustere Error-Behandlung
        """
        try:
            if self.current_view == "2d":
                return self.map_display_2d
            else:
                return self.map_display_3d
        except Exception as e:
            self.logger.debug(f"Get current display warning: {e}")
            return None

    def save_splitter_state(self):
        """
        Funktionsweise: Speichert Splitter-Position für Layout-Persistenz
        Aufgabe: Wird bei splitterMoved Signal aufgerufen
        ÄNDERUNGEN: Exception-sichere Implementierung
        """
        try:
            if hasattr(self, 'splitter') and self.splitter:
                sizes = self.splitter.sizes()
                self.splitter_sizes = sizes
                self.logger.debug(f"Splitter state saved: {sizes}")
        except Exception as e:
            self.logger.debug(f"Splitter state save warning: {e}")

    def restore_splitter_state(self):
        """
        Funktionsweise: Stellt gespeicherte Splitter-Position wieder her
        Aufgabe: Kann beim Tab-Wechsel aufgerufen werden
        ÄNDERUNGEN: Exception-sichere Implementierung
        """
        try:
            if hasattr(self, 'splitter') and hasattr(self, 'splitter_sizes'):
                self.splitter.setSizes(self.splitter_sizes)
                self.logger.debug(f"Splitter state restored: {self.splitter_sizes}")
        except Exception as e:
            self.logger.debug(f"Splitter state restore warning: {e}")

    def setup_signals(self):
        """
        Funktionsweise: Verbindet alle Signal-Slot Verbindungen - REFACTORED
        Aufgabe: Cross-Tab Communication und Data-Updates mit Error-Handling
        ÄNDERUNGEN: Exception-sichere Signal-Verbindungen
        """
        try:
            if self.data_manager:
                self.data_manager.data_updated.connect(self.on_external_data_updated)
                self.data_manager.cache_invalidated.connect(self.on_cache_invalidated)
                self.logger.debug("Data manager signals connected")
        except Exception as e:
            self.logger.error(f"Signal setup failed: {e}")

    # =============================================================================
    # TEIL 3: GENERATION & AUTO-SIMULATION MANAGEMENT - REFACTORED
    # =============================================================================

    @ui_update_handler("auto_simulation")
    def toggle_auto_simulation(self, enabled: bool):
        """
        Funktionsweise: Toggle für Auto-Simulation - REFACTORED
        ÄNDERUNGEN: Generation-State-Reset hinzugefügt für double-generation Problem
        """
        self.auto_simulation_enabled = enabled
        self.logger.debug(f"Auto-simulation {'enabled' if enabled else 'disabled'}")

        # Timer stoppen wenn deaktiviert
        if not enabled and hasattr(self, 'auto_simulation_timer'):
            self.auto_simulation_timer.stop()

        # Generation-State reset bei Toggle
        if not enabled:
            self.generation_in_progress = False

    @memory_critical_handler("generation")
    @core_generation_handler("tab_generation")
    def generate(self):
        """
        Funktionsweise: Standard-Generation mit automatischer Target-LOD-Berechnung und Orchestrator-Integration
        Aufgabe: Koordiniert Generation über Orchestrator oder Fallback auf direkte Tab-Generation
        ÄNDERUNGEN: Numerische LODs, CPU-optimierte Parallelisierung, automatische Target-LOD-Berechnung
        """
        # STATE-RESET für double-generation Problem
        if self.generation_in_progress:
            self.logger.warning("Generation already in progress, resetting state")
            self.generation_in_progress = False
            if hasattr(self, 'auto_simulation_timer'):
                self.auto_simulation_timer.stop()

        # Generation-State setzen
        self.generation_in_progress = True
        self.start_generation_timing()

        # Tab-Generator-Type bestimmen
        tab_name = self.__class__.__name__.replace("Tab", "").lower()

        try:
            # ORCHESTRATOR-PATH (Preferred)
            if self.generation_orchestrator and hasattr(self, 'parameter_manager'):
                parameters = self.parameter_manager.get_current_parameters()
                map_size = parameters.get("size", 128)  # Default aus TERRAIN.MAPSIZE

                # Target-LOD basierend auf map_size berechnen
                from gui.managers.data_lod_manager import calculate_max_lod_for_size
                target_lod = calculate_max_lod_for_size(map_size)

                # Generator-Type aus tab_name
                from gui.managers.generation_orchestrator import GeneratorType
                try:
                    generator_type = GeneratorType(tab_name.upper())
                except ValueError:
                    self.logger.error(f"Unknown generator type: {tab_name}")
                    generator_type = GeneratorType.TERRAIN  # Fallback

                # Request über Orchestrator
                from gui.managers.generation_orchestrator import OrchestratorRequestBuilder
                request = OrchestratorRequestBuilder.build_request(
                    parameters=parameters,
                    generator_type=generator_type,
                    target_lod=target_lod,
                    source_tab=tab_name
                )

                request_id = self.generation_orchestrator.request_generation(request)
                self.logger.info(f"Generation requested via Orchestrator: {request_id}")

                # UI-Status Update
                if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                    self.auto_simulation_panel.set_generation_status("pending",
                                                                     f"Generation queued (LOD 1-{target_lod})")

            # DIRECT-PATH (Fallback)
            elif hasattr(self, f'generate_{tab_name}_system'):
                # Tab hat eigene Generator-Methode
                generator_method = getattr(self, f'generate_{tab_name}_system')
                generator_method()

                # UI-Status Update
                if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                    self.auto_simulation_panel.set_generation_status("generating", "Direct generation in progress")

            # NO-GENERATOR-PATH
            else:
                # Kein Generator verfügbar
                self.logger.info(f"{self.__class__.__name__} has no generation capability")
                self.generation_in_progress = False

                if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                    self.auto_simulation_panel.set_generation_status("info", "No generation available for this tab")

        except Exception as e:
            self.generation_in_progress = False
            self.end_generation_timing(False, str(e))
            self.handle_generation_error(e)

    @pyqtSlot()
    def on_auto_simulation_triggered(self):
        """
        Funktionsweise: Auto-Simulation Handler - REFACTORED
        ÄNDERUNGEN: Auto-tab-generation Logik hinzugefügt
        """
        if not getattr(self, 'generation_in_progress', False):
            try:
                # AUTO-TAB-GENERATION: Ab Geology-Tab automatisch generieren
                tab_name = self.__class__.__name__.replace("Tab", "").lower()
                auto_generation_tabs = ['geology', 'settlement', 'weather', 'water', 'biome']

                if tab_name in auto_generation_tabs:
                    # Prüfe Dependencies
                    if hasattr(self, 'check_input_dependencies'):
                        if self.check_input_dependencies():
                            self.generate()
                        else:
                            self.logger.debug(f"Auto-generation skipped for {tab_name} - dependencies missing")
                    else:
                        self.generate()
                else:
                    # Standard Auto-Simulation
                    self.generate()

            except Exception as e:
                self.logger.error(f"Auto-simulation failed: {e}")

    @memory_critical_handler("timing")
    def start_generation_timing(self):
        """Startet Performance-Timing für Generation"""
        import time
        self.generation_start_time = time.time()

    @memory_critical_handler("timing")
    def end_generation_timing(self, success: bool, error_message: str = ""):
        """Beendet Performance-Timing und loggt Results"""
        if hasattr(self, 'generation_start_time') and self.generation_start_time:
            duration = time.time() - self.generation_start_time
            if success:
                self.logger.info(f"Generation completed in {duration:.2f}s")
            else:
                self.logger.error(f"Generation failed after {duration:.2f}s: {error_message}")
            self.generation_start_time = None

    @ui_update_handler("error_handling")
    def handle_generation_error(self, exception: Exception):
        """Centralized Error-Handling für alle Generationen"""
        error_msg = str(exception)
        self.logger.error(f"Generation error: {error_msg}")

        self.generation_in_progress = False
        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_status("error", error_msg)

    # =============================================================================
    # TEIL 4: ORCHESTRATOR INTEGRATION - VEREINFACHT (ohne QMetaObject)
    # =============================================================================

    def setup_standard_orchestrator_handlers(self, generator_type: str):
        """
        Funktionsweise: Vereinfachte Orchestrator-Integration - REFACTORED
        ÄNDERUNGEN: QMetaObject entfernt, direkte Signal-Verbindungen, harmonisierte Signaturen
        """
        if not self.generation_orchestrator:
            self.logger.warning("No GenerationOrchestrator available")
            return

        self.tab_generator_type = generator_type

        try:
            # Direkte Signal-Verbindungen (VEREINFACHT)
            self.generation_orchestrator.generation_started.connect(self._on_generation_started)
            self.generation_orchestrator.generation_completed.connect(self._on_generation_completed)
            self.generation_orchestrator.generation_progress.connect(self._on_generation_progress)

            if hasattr(self.generation_orchestrator, 'lod_progression_completed'):
                self.generation_orchestrator.lod_progression_completed.connect(self._on_lod_progression_completed)

            self.logger.debug(f"Orchestrator signals connected for {generator_type}")

        except Exception as e:
            self.logger.error(f"Failed to connect orchestrator signals: {e}")

    # VEREINFACHTE ORCHESTRATOR HANDLER (ohne QMetaObject)

    @pyqtSlot(str, int)  # HARMONISIERTE SIGNATUR: (generator_type, lod_level_numeric)
    def _on_generation_started(self, generator_type: str, lod_level: int):
        """Vereinfachter Generation-Start Handler"""
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_status("generating", f"Generating LOD {lod_level}...")

    @pyqtSlot(str, int, bool)  # HARMONISIERTE SIGNATUR: (generator_type, lod_level_numeric, success)
    def _on_generation_completed(self, generator_type: str, lod_level: int, success: bool):
        """Vereinfachter Generation-Complete Handler"""
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        if success:
            if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                self.auto_simulation_panel.set_generation_status("success", f"LOD {lod_level} Complete")
        else:
            if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                self.auto_simulation_panel.set_generation_status("error", f"LOD {lod_level} Failed")

        self.generation_in_progress = False
        self.end_generation_timing(success)

    @pyqtSlot(str, int, int,
              str)  # HARMONISIERTE SIGNATUR: (generator_type, lod_level_numeric, progress_percent, detail)
    def _on_generation_progress(self, generator_type: str, lod_level: int, progress_percent: int, detail: str):
        """Vereinfachter Generation-Progress Handler"""
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_progress(progress_percent, f"LOD {lod_level} - {detail}")

    @pyqtSlot(str, int)  # HARMONISIERTE SIGNATUR: (generator_type, final_lod_numeric)
    def _on_lod_progression_completed(self, generator_type: str, final_lod: int):
        """Vereinfachter LOD-Progression Handler"""
        if generator_type != getattr(self, 'tab_generator_type', None):
            return

        self.generation_in_progress = False
        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_status("success", "Generation Complete")

    # =============================================================================
    # TEIL 5: DISPLAY MANAGEMENT & SIGNAL HANDLING - REFACTORED
    # =============================================================================

    @ui_update_handler("display_update")
    def update_display_mode(self):
        """
        Funktionsweise: Default Display-Mode Handler - STANDARD IMPLEMENTATION
        ÄNDERUNGEN: Robustere Error-Behandlung, konsistentes Data-Fetching
        """
        try:
            heightmap = self.data_manager.get_terrain_data("heightmap") if self.data_manager else None
            if heightmap is not None:
                current_display = self.get_current_display()
                if current_display:
                    self._render_current_mode(0, current_display, heightmap, "heightmap")
        except Exception as e:
            self.logger.debug(f"Display mode update failed: {e}")

    def _render_current_mode(self, mode_id: int, display, data, layer_type: str):
        """
        Funktionsweise: Zentraler Display-Renderer mit Change-Detection
        ÄNDERUNGEN: Verbesserte Memory-Management, robustere Error-Behandlung
        """
        try:
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

        except Exception as e:
            self.logger.error(f"Render current mode failed: {e}")

    # SIGNAL HANDLING

    @ui_update_handler("external_data")
    def on_external_data_updated(self, generator_type: str, data_key: str):
        """Standard-Handler für externe Daten-Updates"""
        self.logger.debug(f"External data updated: {generator_type}.{data_key}")

        try:
            # Dependency-Check triggern
            if hasattr(self, 'required_dependencies') and hasattr(self, 'check_input_dependencies'):
                if data_key in self.required_dependencies:
                    self.check_input_dependencies()

            # Display-Update triggern wenn Auto-Simulation aktiv
            if self.auto_simulation_enabled and hasattr(self, 'update_display_mode'):
                self.update_display_mode()

        except Exception as e:
            self.logger.debug(f"External data update handling failed: {e}")

    @ui_update_handler("cache_invalidated")
    def on_cache_invalidated(self, generator_type: str):
        """Standard-Handler für Cache-Invalidierung"""
        self.logger.debug(f"Cache invalidated for: {generator_type}")

        try:
            if hasattr(self, 'update_display_mode') and not self.generation_in_progress:
                self.update_display_mode()
        except Exception as e:
            self.logger.debug(f"Cache invalidation handling failed: {e}")

    @dependency_handler("dependency_check")
    def check_input_dependencies(self):
        """Standard-Dependency-Check - ÜBERSCHREIBBAR"""
        try:
            if hasattr(self, 'required_dependencies') and self.data_manager:
                is_complete, missing = self.data_manager.check_dependencies(
                    self.__class__.__name__.replace("Tab", "").lower(),
                    self.required_dependencies
                )

                if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                    self.auto_simulation_panel.set_manual_button_enabled(is_complete)

                return is_complete
        except Exception as e:
            self.logger.debug(f"Dependency check failed: {e}")

        return True

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