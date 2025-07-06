#!/usr/bin/env python3
"""
Path: MapGenerator/gui/utils/error_handler.py
__init__.py existiert in "utils"

Einheitliches Error Handling und Logging System
Vereinheitlicht Fehlerbehandlung zwischen allen Tabs
"""

import logging
import traceback
import functools
from PyQt5.QtWidgets import QMessageBox, QApplication


class WorldGeneratorLogger:
    """
    Funktionsweise: Zentraler Logger für die gesamte Anwendung
    - Einheitliche Log-Formate und -Level
    - Automatische Fehler-Kategorisierung
    - GUI-Benachrichtigungen bei kritischen Fehlern
    """
    import os
    DEBUG_MODE = True

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        """Konfiguriert den Logger mit einheitlichen Einstellungen"""
        cls._logger = logging.getLogger('WorldGenerator')
        cls._logger.setLevel(logging.DEBUG)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)

        # Handler hinzufügen (nur einmal)
        if not cls._logger.handlers:
            cls._logger.addHandler(console_handler)

    @classmethod
    def get_logger(cls):
        """Gibt den konfigurierten Logger zurück"""
        if cls._logger is None:
            cls._setup_logger()
        return cls._logger


class ErrorHandler:
    """
    Funktionsweise: Zentraler Error Handler mit GUI-Integration
    - Kategorisiert Fehler nach Schwere
    - Zeigt Benutzer-freundliche Fehlermeldungen
    - Loggt technische Details für Debugging
    """

    def __init__(self):
        self.logger = WorldGeneratorLogger.get_logger()

    def handle_import_error(self, module_name, fallback_action=None):
        """
        Funktionsweise: Behandelt Import-Fehler einheitlich
        Args:
            module_name (str): Name des fehlgeschlagenen Moduls
            fallback_action (callable): Fallback-Funktion
        """
        self.logger.warning(f"Modul '{module_name}' konnte nicht importiert werden")

        if fallback_action:
            self.logger.info(f"Verwende Fallback für {module_name}")
            return fallback_action()

        return None

    def handle_tab_navigation_error(self, source_tab, target_tab, error):
        """
        Funktionsweise: Behandelt Tab-Navigation Fehler
        - Loggt Details für Debugging
        - Zeigt Benutzer-Warnung
        - Bleibt im aktuellen Tab
        """
        error_msg = f"Fehler beim Wechsel von {source_tab} zu {target_tab}: {error}"
        self.logger.error(error_msg)
        self.logger.debug(traceback.format_exc())

        # GUI-Benachrichtigung
        self.show_warning("Navigation Fehler",
                          f"Konnte nicht zu {target_tab} wechseln.\nBleibe im aktuellen Tab.")

    def handle_parameter_error(self, tab_name, param_name, error, corrected_value=None):
        """
        Funktionsweise: Behandelt Parameter-Validierungsfehler
        - Korrigiert automatisch wenn möglich
        - Benachrichtigt Benutzer über Korrekturen
        """
        if corrected_value is not None:
            self.logger.warning(f"{tab_name}: Parameter {param_name} korrigiert zu {corrected_value}")
            return corrected_value
        else:
            self.logger.error(f"{tab_name}: Parameter {param_name} Fehler: {error}")
            self.show_error("Parameter Fehler", f"Ungültiger Wert für {param_name}: {error}")
            return None

    def handle_map_rendering_error(self, tab_name, error):
        """
        Funktionsweise: Behandelt Karten-Rendering Fehler
        - Zeigt Fallback-Karte
        - Loggt Details für Debugging
        """
        self.logger.error(f"{tab_name}: Karten-Rendering Fehler: {error}")
        self.logger.debug(traceback.format_exc())

        # Zeige vereinfachte Fehlermeldung
        self.show_info("Karten-Update",
                       f"Karte konnte nicht aktualisiert werden.\nVerwende Standard-Ansicht.")

    def handle_worldstate_error(self, operation, error):
        """
        Funktionsweise: Behandelt WorldState Fehler
        - Parameter-Speicherung/Laden
        - Fallback zu Standard-Werten
        """
        self.logger.error(f"WorldState {operation} Fehler: {error}")
        self.show_warning("Daten-Fehler",
                          f"Problem beim {operation} der Parameter.\nVerwende Standard-Werte.")

    def show_error(self, title, message):
        """Zeigt kritische Fehlermeldung"""
        if QApplication.instance():
            QMessageBox.critical(None, title, message)

    def show_warning(self, title, message):
        """Zeigt Warnung"""
        if QApplication.instance():
            QMessageBox.warning(None, title, message)

    def show_info(self, title, message):
        """Zeigt Information"""
        if QApplication.instance():
            QMessageBox.information(None, title, message)


def safe_execute(error_handler_method=None, fallback_return=None, show_gui_error=True):
    """
    Funktionsweise: Decorator für sichere Ausführung von Methoden
    - Fängt alle Exceptions ab
    - Loggt automatisch
    - Ruft spezifischen Error Handler auf

    Args:
        error_handler_method (str): Name der ErrorHandler Methode
        fallback_return: Rückgabe-Wert bei Fehler
        show_gui_error (bool): Zeige GUI-Fehlermeldung
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if WorldGeneratorLogger.DEBUG_MODE:
                return func(*args, **kwargs)

            # PRODUCTION MODE: Fange Exceptions ab
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()

                # Standard-Logging
                error_handler.logger.error(f"Fehler in {func.__name__}: {e}")
                error_handler.logger.debug(traceback.format_exc())

                # Spezifischer Error Handler
                if error_handler_method and hasattr(error_handler, error_handler_method):
                    handler_func = getattr(error_handler, error_handler_method)
                    handler_func(func.__name__, e)
                elif show_gui_error:
                    error_handler.show_error("Unerwarteter Fehler",
                                             f"Fehler in {func.__name__}:\n{str(e)}")

                return fallback_return

        return wrapper

    return decorator


def safe_import(module_path, fallback=None):
    """
    Funktionsweise: Sicherer Import mit automatischem Fallback
    Args:
        module_path (str): Import-Pfad des Moduls
        fallback: Fallback-Wert bei Import-Fehler
    Returns:
        Das importierte Modul oder Fallback
    """
    try:
        # Dynamischer Import
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])
        return module
    except ImportError as e:
        error_handler = ErrorHandler()
        error_handler.handle_import_error(module_path)
        return fallback


class TabErrorContext:
    """
    Funktionsweise: Context Manager für Tab-spezifische Fehlerbehandlung
    - Automatisches Error Handling für gesamte Tab-Operationen
    - Cleanup bei Fehlern
    """

    def __init__(self, tab_name, operation_name):
        self.tab_name = tab_name
        self.operation_name = operation_name
        self.error_handler = ErrorHandler()

    def __enter__(self):
        self.error_handler.logger.debug(f"{self.tab_name}: Starte {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.logger.error(
                f"{self.tab_name}: Fehler in {self.operation_name}: {exc_val}"
            )
            self.error_handler.show_warning(
                f"{self.tab_name} Fehler",
                f"Problem bei {self.operation_name}.\nOperation wurde abgebrochen."
            )
            return True  # Exception behandelt
        else:
            self.error_handler.logger.debug(f"{self.tab_name}: {self.operation_name} erfolgreich")