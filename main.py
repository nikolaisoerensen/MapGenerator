#!/usr/bin/env python3
"""
Path: MapGenerator/main.py

MapGenerator Main Application with Central Window Management
============================================================

Core application orchestrating navigation flow: MainMenu → Loading → MapEditor
Provides centralized manager creation, clean navigation chains, and integrated
GenerationOrchestrator for cross-component generator coordination.

Architecture:
- MapGeneratorApp manages all window lifecycles
- Clean navigation without overlapping windows
- Centralized DataManager and NavigationManager
- Integrated GenerationOrchestrator for generator coordination
- Enhanced orchestrator signals for cross-tab communication
- Professional error handling and resource management
"""

import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

# Add project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.tabs.main_menu import MainMenuWindow
from gui.config.gui_default import AppConstants
from gui.tabs.loading_tab import LoadingTab
from gui.map_editor_window import MapEditorWindow
from gui.managers.data_manager import DataManager
from gui.managers.navigation_manager import NavigationManager
from gui.managers.generation_orchestrator import GenerationOrchestrator

class MapGeneratorApp(QObject):
    """
    Central Application Controller for MapGenerator Suite
    ====================================================

    Manages complete application lifecycle including window navigation,
    resource management, and cross-component communication through
    integrated GenerationOrchestrator and manager coordination.

    Navigation Flow:
        MainMenu → LoadingTab → MapEditor → (return to MainMenu)

    Manager Integration:
        - DataManager: Centralized data storage and caching
        - NavigationManager: Tab navigation and state management
        - GenerationOrchestrator: Generator coordination and execution

    Signal Architecture:
        - Clean signal-based communication between windows
        - Orchestrator signals for generation progress tracking
        - Dependency invalidation for cross-tab updates
        - Memory usage monitoring and performance tracking
    """

    def __init__(self):
        super().__init__()

        # Initialize logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Core managers - centralized creation
        self.data_manager = DataManager()
        self.navigation_manager = NavigationManager(data_manager=self.data_manager)
        self.generation_orchestrator = GenerationOrchestrator(data_manager=self.data_manager)

        # Window references
        self.main_menu = None
        self.loading_dialog = None
        self.map_editor = None
        self.current_window = None

        # Application state
        self.is_shutting_down = False

        # Setup manager integrations
        self._setup_manager_integration()
        self._setup_memory_monitoring()

        self.logger.info("MapGeneratorApp initialized successfully")

    def _setup_logging(self):
        """
        Configure application-wide logging with colored output
        ======================================================

        Sets up structured logging with appropriate levels and formatting
        for development and production use. Configures colored console
        output for improved readability.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )

        # Configure colored logging
        try:
            import colorlog
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'yellow',  # Changed from red to yellow
                    'WARNING': 'orange',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            ))

            root_logger = logging.getLogger()
            root_logger.handlers.clear()
            root_logger.addHandler(handler)

        except ImportError:
            # Fallback to standard logging if colorlog not available
            pass

    def _setup_manager_integration(self):
        """
        Configure manager cross-communication and signal routing
        =======================================================

        Establishes signal connections between NavigationManager and
        GenerationOrchestrator for coordinated application behavior.
        Connects orchestrator events to application-level handlers
        for UI updates and state management.
        """
        # Navigation manager signals
        if self.navigation_manager:
            self.navigation_manager.tab_changed.connect(self._on_tab_navigation_requested)

        # Generation orchestrator signals
        if self.generation_orchestrator:
            self.generation_orchestrator.generation_started.connect(self._on_generation_started)
            self.generation_orchestrator.generation_completed.connect(self._on_generation_completed)
            self.generation_orchestrator.lod_progression_started.connect(self._on_lod_progression_started)
            self.generation_orchestrator.lod_progression_completed.connect(self._on_lod_progression_completed)
            self.generation_orchestrator.dependency_invalidated.connect(self._on_dependency_invalidated)

    def _setup_memory_monitoring(self):
        """
        Initialize periodic memory usage monitoring
        ==========================================

        Sets up timer-based memory monitoring for performance tracking
        and resource leak detection. Only runs when windows are active
        to minimize overhead during idle periods.
        """
        self.memory_monitor_timer = QTimer()
        self.memory_monitor_timer.timeout.connect(self._log_memory_usage)
        # Timer started only when needed

    def start_application(self):
        """
        Launch application with MainMenu as entry point
        ==============================================

        Initializes and displays the MainMenu window to begin
        the application workflow. Sets up initial application
        state and begins memory monitoring.
        """
        self.logger.info("Starting MapGenerator application")
        self._show_main_menu()

        # Start memory monitoring
        self.memory_monitor_timer.start(AppConstants.MEMORY_CHECK_INTERVAL_MS)

    def _show_main_menu(self):
        """
        Create and display MainMenu window
        =================================

        Instantiates MainMenu with required manager references
        and connects navigation signals for map editor launch.
        """
        try:
            self.main_menu = MainMenuWindow(
                data_manager=self.data_manager,
                navigation_manager=self.navigation_manager
            )

            # Connect map editor launch signal
            self.main_menu.map_editor_requested.connect(self._start_map_editor)

            self.main_menu.show()
            self.current_window = self.main_menu

            self.logger.info("MainMenu window displayed")

        except Exception as e:
            self.logger.error(f"Failed to create MainMenu: {e}")
            self._handle_critical_error("MainMenu creation failed", e)

    @pyqtSlot()
    def _start_map_editor(self):
        """
        Initiate map editor through loading process
        ==========================================

        Transitions from MainMenu to LoadingTab, then to MapEditor.
        Handles the complete loading workflow with proper error
        recovery and state management.
        """
        if self.is_shutting_down:
            return

        try:
            self.logger.info("Starting map editor transition")

            # Hide MainMenu (preserve for potential return)
            if self.main_menu:
                self.main_menu.hide()

            # Create and display LoadingTab
            self.loading_dialog = LoadingTab(
                data_manager=self.data_manager,
                generation_orchestrator=self.generation_orchestrator,
                parent=None
            )

            # Connect loading completion signals
            self.loading_dialog.loading_completed.connect(self._on_loading_completed)
            self.loading_dialog.loading_cancelled.connect(self._on_loading_cancelled)

            # Execute loading dialog
            self.loading_dialog.exec_()

        except Exception as e:
            self.logger.error(f"Map editor transition failed: {e}")
            self._recover_to_main_menu()

    @pyqtSlot(bool)
    def _on_loading_completed(self, success: bool):
        """
        Handle loading completion with simplified state management
        =========================================================

        Processes LoadingTab completion and transitions to MapEditor
        on success or returns to MainMenu on failure. Uses simplified
        state checking instead of complex flag management.

        Args:
            success: True if all generators loaded successfully
        """
        self.logger.info(f"Loading completed with success={success}")

        # double-call protection
        if hasattr(self, '_loading_completed_called') and self._loading_completed_called:
            self.logger.debug("Loading completion already handled, ignoring duplicate call")
            return
        self._loading_completed_called = True

        try:
            self.loading_dialog.close()
        except Exception as e:
            self.logger.warning(f"Error closing loading dialog: {e}")
        finally:
            self.loading_dialog = None

        if success:
            self._show_map_editor()
        else:
            self.logger.warning("Loading failed, returning to MainMenu")
            self._recover_to_main_menu()

        # Reset flag after delay for next cycle
        QTimer.singleShot(2000, lambda: delattr(self, '_loading_completed_called') if hasattr(self,'_loading_completed_called') else None)

    @pyqtSlot()
    def _on_loading_cancelled(self):
        """
        Handle loading cancellation by user
        ===================================

        Processes user cancellation of loading process and
        returns gracefully to MainMenu with proper cleanup.
        """
        self.logger.info("Loading cancelled by user")

        # Close loading dialog
        if self.loading_dialog:
            try:
                self.loading_dialog.close()
            except Exception as e:
                self.logger.warning(f"Error closing loading dialog: {e}")
            finally:
                self.loading_dialog = None

        self._recover_to_main_menu()

    def _show_map_editor(self):
        """
        Create and display MapEditor window
        ==================================

        Instantiates MapEditor with all required managers and
        activates terrain tab as the starting point. Handles
        creation errors with fallback to MainMenu.
        """
        try:
            self.logger.info("Creating MapEditor window")

            self.map_editor = MapEditorWindow(
                data_manager=self.data_manager,
                navigation_manager=self.navigation_manager,
                generation_orchestrator=self.generation_orchestrator
            )

            # Connect return navigation signal
            self.map_editor.return_to_main_menu.connect(self._return_to_main_menu)

            # Display map editor and activate terrain tab
            self.map_editor.show()
            self.map_editor.activate_tab("terrain")

            self.current_window = self.map_editor
            self.logger.info("MapEditor displayed successfully")

        except Exception as e:
            self.logger.error(f"MapEditor creation failed: {e}")
            import traceback
            traceback.print_exc()
            self._recover_to_main_menu()

    def _recover_to_main_menu(self):
        """
        Fallback recovery to MainMenu on errors
        =======================================

        Safely returns to MainMenu when other windows fail
        to load or encounter critical errors. Ensures
        application remains functional.
        """
        if self.main_menu:
            self.main_menu.show()
            self.current_window = self.main_menu
        else:
            # MainMenu lost - recreate
            self._show_main_menu()

    @pyqtSlot()
    def _return_to_main_menu(self):
        """
        Navigate back from MapEditor to MainMenu
        =======================================

        Handles clean transition from MapEditor back to MainMenu
        with proper resource cleanup and window management.
        """
        self.logger.info("Returning to MainMenu from MapEditor")

        # Close and cleanup MapEditor
        if self.map_editor:
            try:
                self.map_editor.close()
            except Exception as e:
                self.logger.warning(f"Error closing MapEditor: {e}")
            finally:
                self.map_editor = None

        # Show MainMenu
        if self.main_menu:
            self.main_menu.show()
            self.current_window = self.main_menu
        else:
            self._show_main_menu()

    # Navigation and orchestrator signal handlers

    @pyqtSlot(str, str)
    def _on_tab_navigation_requested(self, from_tab: str, to_tab: str):
        """
        Handle tab navigation requests from NavigationManager
        ====================================================

        Routes tab change requests to active MapEditor window
        for processing. Provides centralized navigation control.

        Args:
            from_tab: Source tab identifier
            to_tab: Target tab identifier
        """
        if self.map_editor and hasattr(self.map_editor, 'navigate_to_tab'):
            self.map_editor.navigate_to_tab(to_tab)

    @pyqtSlot(str, str)
    def _on_generation_started(self, generator_type: str, lod_level: str):
        """
        Handle generation start events for application-wide coordination
        ===============================================================

        Processes generation start notifications from orchestrator
        and routes to appropriate UI components for status updates.

        Args:
            generator_type: Type of generator starting
            lod_level: LOD level being generated
        """
        self.logger.debug(f"Generation started: {generator_type} {lod_level}")

        # Route to active window
        if self.map_editor and hasattr(self.map_editor, 'on_generation_started'):
            self.map_editor.on_generation_started(generator_type, lod_level)

    @pyqtSlot(str, str, bool)
    def _on_generation_completed(self, generator_type: str, lod_level: str, success: bool):
        """
        Handle generation completion for UI updates
        ==========================================

        Processes generation completion notifications and updates
        UI components with results. Enables dependent generators
        when dependencies become available.

        Args:
            generator_type: Type of generator completed
            lod_level: LOD level that was generated
            success: Whether generation succeeded
        """
        self.logger.info(f"Generation completed: {generator_type} {lod_level} success={success}")

        # Route to active window
        if self.map_editor and hasattr(self.map_editor, 'on_generation_completed'):
            self.map_editor.on_generation_completed(generator_type, lod_level, success)

    @pyqtSlot(str, str)
    def _on_lod_progression_started(self, generator_type: str, target_lod: str):
        """
        Handle LOD progression start for performance monitoring
        ======================================================

        Tracks start of multi-LOD generation sequences for
        performance analysis and resource management.

        Args:
            generator_type: Generator beginning LOD progression
            target_lod: Final target LOD level
        """
        self.logger.debug(f"LOD progression started: {generator_type} → {target_lod}")

    @pyqtSlot(str, str)
    def _on_lod_progression_completed(self, generator_type: str, final_lod: str):
        """
        Handle LOD progression completion
        ================================

        Processes completion of full LOD progression sequences
        and updates performance metrics.

        Args:
            generator_type: Generator that completed progression
            final_lod: Final LOD level achieved
        """
        self.logger.info(f"LOD progression completed: {generator_type} final: {final_lod}")

    @pyqtSlot(str, list)
    def _on_dependency_invalidated(self, generator_type: str, affected_generators: list):
        """
        Handle dependency invalidation for cross-tab updates
        ===================================================

        Processes cache invalidation events and notifies affected
        components to update their state or regenerate content.

        Args:
            generator_type: Generator that triggered invalidation
            affected_generators: List of affected generator types
        """
        self.logger.info(f"Dependencies invalidated: {generator_type} affected: {affected_generators}")

        # Notify MapEditor for tab updates
        if self.map_editor and hasattr(self.map_editor, 'on_dependency_invalidated'):
            self.map_editor.on_dependency_invalidated(generator_type, affected_generators)

    def _log_memory_usage(self):
        """
        Periodic memory usage logging for performance monitoring
        =======================================================

        Collects memory usage from all managers and logs summary
        for performance analysis. Only runs when windows are active.
        """
        if self.is_shutting_down:
            return

        try:
            memory_summary = self.get_memory_usage_summary()
            self.logger.debug(f"Memory usage: {memory_summary}")
        except Exception as e:
            self.logger.warning(f"Memory usage monitoring failed: {e}")

    def get_memory_usage_summary(self) -> dict:
        """
        Collect comprehensive memory usage from all components
        =====================================================

        Aggregates memory usage statistics from DataManager,
        GenerationOrchestrator, and active windows for monitoring
        and optimization purposes.

        Returns:
            Dictionary containing memory usage by component
        """
        try:
            summary = {
                "data_manager": self.data_manager.get_memory_usage() if self.data_manager else {},
                "generation_orchestrator": (
                    self.generation_orchestrator.get_memory_usage_summary()
                    if self.generation_orchestrator else {}
                ),
                "active_window": self.current_window.__class__.__name__ if self.current_window else "None"
            }
            return summary
        except Exception as e:
            self.logger.warning(f"Memory usage collection failed: {e}")
            return {"error": str(e)}

    def _handle_critical_error(self, context: str, error: Exception):
        """
        Handle critical application errors with graceful degradation
        ===========================================================

        Processes critical errors that threaten application stability
        and attempts graceful recovery or controlled shutdown.

        Args:
            context: Description of operation that failed
            error: Exception that occurred
        """
        self.logger.critical(f"Critical error in {context}: {error}")
        import traceback
        traceback.print_exc()

        # Attempt graceful recovery
        try:
            if self.main_menu:
                self.main_menu.show()
            else:
                QApplication.quit()
        except Exception as recovery_error:
            self.logger.critical(f"Recovery failed: {recovery_error}")
            QApplication.quit()

    def cleanup_and_exit(self):
        """
        Graceful application shutdown with comprehensive resource cleanup
        ================================================================

        Performs orderly shutdown of all components including window
        closure, manager cleanup, and resource deallocation. Ensures
        no resource leaks or dangling processes.
        """
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        self.logger.info("Beginning application shutdown")

        try:
            # Stop timers
            if hasattr(self, 'memory_monitor_timer'):
                self.memory_monitor_timer.stop()

            # Close all windows
            windows_to_close = [
                ("loading_dialog", self.loading_dialog),
                ("map_editor", self.map_editor),
                ("main_menu", self.main_menu)
            ]

            for window_name, window in windows_to_close:
                if window:
                    try:
                        window.close()
                        self.logger.debug(f"Closed {window_name}")
                    except Exception as e:
                        self.logger.warning(f"Error closing {window_name}: {e}")

            # Cleanup managers
            managers_to_cleanup = [
                ("NavigationManager", self.navigation_manager),
                ("GenerationOrchestrator", self.generation_orchestrator),
                ("DataManager", self.data_manager)
            ]

            for manager_name, manager in managers_to_cleanup:
                if manager and hasattr(manager, 'cleanup_resources'):
                    try:
                        manager.cleanup_resources()
                        self.logger.debug(f"Cleaned up {manager_name}")
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up {manager_name}: {e}")

            self.logger.info("Application shutdown completed successfully")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            import traceback
            traceback.print_exc()

        # Force application exit
        QApplication.quit()


def main():
    """
    Application entry point with enhanced error handling
    ===================================================

    Initializes QApplication, creates MapGeneratorApp instance,
    and starts the application event loop with proper exception
    handling and resource management.
    """
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)  # Manual quit control

        # Set application metadata
        app.setApplicationName("MapGenerator")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("MapGenerator Team")

        # Create and start MapGenerator application
        map_generator_app = MapGeneratorApp()

        # Connect shutdown signal
        app.aboutToQuit.connect(map_generator_app.cleanup_and_exit)

        # Start application
        map_generator_app.start_application()

        # Enter event loop
        return app.exec_()

    except Exception as e:
        print(f"Critical startup error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())