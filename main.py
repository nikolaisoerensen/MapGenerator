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

import sys # Für Kommandozeilen Argumente und Python Suchpfade
import logging # Für Logging statt print("..")
from pathlib import Path # moderne Art mit Suchpfaden zu arbeiten

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, QTimer

# Add project directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.main_menu import MainMenuWindow


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
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Nur minimale Initialisierung
        self.main_menu = None
        self.is_shutting_down = False

        self.logger.info("MapGeneratorApp minimal initialization completed")

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
            handlers=[logging.StreamHandler()]
        )

        # Configure colored logging
        try:
            import colorlog
            handler = colorlog.StreamHandler()
            handler.setFormatter(colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'yellow',
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
            self.logger.error("Logger ColorLog Import Error")
            pass

    def show_main_menu(self):
        """
        Create and display MainMenu window
        =================================

        Instantiates MainMenu with required manager references
        and connects navigation signals for map editor launch.
        """
        try:
            self.main_menu = MainMenuWindow()
            self.main_menu.show()
            self.logger.info("MainMenu window displayed")

        except Exception as e:
            self.logger.error(f"Failed to create MainMenu: {e}")
            self._handle_critical_error("MainMenu creation failed", e)

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
        else:
            self.show_main_menu()

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

            if self.main_menu:
                self.main_menu.close()

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
        app.setOrganizationName("SörAppz")

        # Create and start MapGenerator application
        map_generator_app = MapGeneratorApp()

        # Connect shutdown signal
        app.aboutToQuit.connect(map_generator_app.cleanup_and_exit)

        # Start application
        map_generator_app.logger.info("Starting MapGenerator application")
        map_generator_app.show_main_menu()

        # Enter event loop
        return app.exec_()

    except Exception as e:
        print(f"Critical startup error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())