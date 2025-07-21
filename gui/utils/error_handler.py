"""
Path: gui/error_handler.py

Funktionsweise: Zentrale Error-Behandlung für alle Map-Generator Komponenten
- Paralleles Error Handling zu PyCharm ohne Unterdrückung von Exceptions
- Konfigurierbares Logging (Konsole, Datei, beide)
- Spezialisierte Handler für alle kritischen Generator-Operationen
- Ein/Aus-Schalter für flexibles Error-Management
- Performance-optimiertes Decorator-System
- Automatische Error-Statistiken und Reporting

Kommunikationskanäle:
- Input: Exception-Objekte von allen Generator-Komponenten
- Output: Strukturierte Error-Logs mit Kontext-Informationen
- Config: Globale Schalter für Error-Handler Aktivierung
- Integration: Decorator-basierte Einbindung in bestehenden Code

Struktur der 67 kritischen Positionen (zusammengefasst in 12 Kategorien):

1. CORE GENERATOR OPERATIONS (18 Positionen)
   - Alle generate_*() Methoden in terrain/geology/settlement/weather/water/biome_tab
   - Heightmap/Slopemap/Shadowmap Generierung
   - Spezialisierte Berechnungen (Erosion, Wind, Biome-Classification)

2. DATA MANAGEMENT OPERATIONS (12 Positionen)
   - set_*_data() / get_*_data() Operationen in data_manager
   - Cache-Management und Validation
   - Dependency-Checking zwischen Generatoren
   - Memory-Management für große Arrays

3. GPU/SHADER OPERATIONS (8 Positionen)
   - OpenGL-Context und Shader-Compilation
   - GPU-Buffer-Management und Texture-Operations
   - Compute-Shader Ausführung
   - GPU-Memory Allocation/Cleanup

4. UI/NAVIGATION OPERATIONS (6 Positionen)
   - Tab-Navigation und Window-Management
   - Parameter-UI Setup und Validation
   - Display-Updates und Visualization
   - Event-Handling (Mouse, Keyboard)

5. EXPORT/IMPORT OPERATIONS (5 Positionen)
   - World-Export (PNG, JSON, OBJ)
   - Parameter-Serialization
   - File I/O Operations
   - Data-Format Conversion

6. INITIALIZATION OPERATIONS (4 Positionen)
   - Core-Generator Initialization
   - Manager-Setup (Data, Navigation, Shader)
   - UI-Component Initialization
   - Module-Import Validation

7. PARAMETER OPERATIONS (4 Positionen)
   - Parameter-Loading und Validation
   - Cross-Parameter Constraints
   - Default-Value Management
   - Parameter-Change Processing

8. STATISTICS/CALCULATION OPERATIONS (3 Positionen)
   - Statistics-Berechnung für alle Generatoren
   - Performance-Monitoring
   - World-Completeness Analysis

9. 3D RENDERING OPERATIONS (3 Positionen)
   - 3D-Mesh Generation
   - OpenGL-Rendering Pipeline
   - 3D-Interaction Handling

10. CLEANUP/RESOURCE OPERATIONS (2 Positionen)
    - Resource-Cleanup bei Tab-Wechsel
    - Application-Shutdown Handling

11. CACHE OPERATIONS (2 Positionen)
    - Cache-Invalidation und Validation
    - Timestamp-Management

12. DEPENDENCY OPERATIONS (4 Positionen)
    - Dependency-Validation zwischen Tabs
    - Input-Status Monitoring
    - Missing-Dependency Handling
    - Circular-Dependency Detection

Error Handler Methoden (12 Hauptkategorien mit Untermethoden):
"""

import logging
import traceback
import functools
import datetime
import sys
import psutil  # Für Memory-Diagnostik
import gc      # Für Garbage Collection Info
from typing import Any, Callable, Optional, Dict, List, Union
from pathlib import Path

# =============================================================================
# GLOBALE KONFIGURATION - HIER EIN/AUSSCHALTEN
# =============================================================================

# Hauptschalter - True = Error Handler aktiv, False = nur PyCharm
ERROR_HANDLER_ENABLED = True

# Ausgabe-Konfiguration
ERROR_OUTPUT_CONSOLE = True  # Ausgabe in Konsole
ERROR_OUTPUT_FILE = True  # Ausgabe in Datei
ERROR_LOG_FILE = "map_generator_errors.log"  # Dateiname für Error-Log

# Kategorien-spezifische Aktivierung
ERROR_CATEGORIES = {
    "core_generation": True,  # Core Generator Operations
    "data_management": True,  # Data Manager Operations
    "gpu_shader": True,  # GPU/Shader Operations
    "ui_navigation": True,  # UI/Navigation Operations
    "export_import": True,  # Export/Import Operations
    "initialization": True,  # Initialization Operations
    "parameters": True,  # Parameter Operations
    "statistics": True,  # Statistics/Calculation Operations
    "rendering_3d": True,  # 3D Rendering Operations
    "cleanup": True,  # Cleanup/Resource Operations
    "cache": True,  # Cache Operations
    "dependencies": True  # Dependency Operations
}

# Log-Level Konfiguration
ERROR_LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
SHOW_FULL_TRACEBACK = True  # Vollständigen Traceback anzeigen
SHOW_FUNCTION_ARGS = True  # Funktionsargumente im Error-Log
SHOW_ERROR_STATISTICS = True  # Error-Statistiken sammeln

# Performance-Optimierung
MAX_ERRORS_PER_FUNCTION = 10  # Max Errors pro Funktion loggen
ERROR_RATE_LIMITING = True  # Rate-Limiting für wiederkehrende Errors


# =============================================================================
# SPEZIALISIERTE ERROR HANDLER KLASSEN
# =============================================================================

class ErrorStatistics:
    """
    Funktionsweise: Sammelt und verwaltet Error-Statistiken
    Aufgabe: Performance-optimierte Statistik-Sammlung für alle Error-Kategorien
    """

    def __init__(self):
        self.total_errors = 0
        self.errors_by_category = {cat: 0 for cat in ERROR_CATEGORIES.keys()}
        self.errors_by_function = {}
        self.critical_errors = []
        self.error_timeline = []

    def record_error(self, category: str, function_name: str, error_type: str, severity: str):
        """
        Funktionsweise: Protokolliert Errors mit Timestamp und Kontext
        Aufgabe: Sammelt Error-Statistiken mit Memory-Error Spezialbehandlung
        Besonderheit: MemoryErrors werden NIEMALS rate-limited (kritisch für Debugging)
        """
        self.total_errors += 1
        self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1
        self.errors_by_function[function_name] = self.errors_by_function.get(function_name, 0) + 1

        if severity == "CRITICAL":
            self.critical_errors.append({
                'function': function_name,
                'error_type': error_type,
                'timestamp': datetime.datetime.now(),
                'category': category
            })

        # Memory-kritische Errors niemals rate-limitieren (NEUE LOGIK)
        if error_type == "MemoryError":
            return True  # Immer loggen - Memory-Errors sind kritisch

        if ERROR_RATE_LIMITING:
            # Rate limiting - nur erste N Errors pro Funktion
            if self.errors_by_function[function_name] > MAX_ERRORS_PER_FUNCTION:
                return False  # Nicht loggen

        return True  # Loggen


class MapGeneratorErrorHandler:
    """
    Funktionsweise: Zentrale Error-Handler Klasse für Map-Generator
    Aufgabe: Koordiniert alle Error-Handling Operationen mit Kategorie-Support
    """

    def __init__(self):
        self.logger = None
        self.statistics = ErrorStatistics()
        self.error_contexts = {}

        if ERROR_HANDLER_ENABLED:
            self._setup_logging()

    def _setup_logging(self):
        """
        Funktionsweise: Konfiguriert erweiterte Logging für Map-Generator
        Aufgabe: Multi-Handler Setup mit Formatierung und Kategorisierung
        """
        self.logger = logging.getLogger('MapGeneratorErrorHandler')
        self.logger.setLevel(getattr(logging, ERROR_LOG_LEVEL))

        # Verhindere Duplikate
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Erweiterte Formatierung
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | [%(category)s] %(funcName)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Konsole-Handler mit Farben
        if ERROR_OUTPUT_CONSOLE:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Datei-Handler mit Rotation
        if ERROR_OUTPUT_FILE:
            file_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # =============================================================================
    # 1. CORE GENERATOR ERROR HANDLERS
    # =============================================================================

    def handle_core_generation_error(self, func_name: str, error: Exception,
                                     generator_type: str, args: tuple, kwargs: dict):
        """
        Funktionsweise: Behandelt Fehler in allen Core-Generator Operationen
        Aufgabe: Spezialisierte Behandlung für terrain/geology/settlement/weather/water/biome
        Kategorie: core_generation
        """
        if not self._should_handle_category("core_generation"):
            return

        severity = self._determine_severity(error, "core_generation")

        if not self.statistics.record_error("core_generation", func_name, type(error).__name__, severity):
            return  # Rate limited

        error_msg = f"CORE GENERATION ERROR in {generator_type.upper()}"
        error_msg += f"\nGenerator: {generator_type}"
        error_msg += f"\nFunction: {func_name}"
        error_msg += f"\nError Type: {type(error).__name__}"
        error_msg += f"\nError Message: {str(error)}"

        # Generator-spezifische Kontextinformationen
        if generator_type == "terrain":
            error_msg += self._get_terrain_context(kwargs)
        elif generator_type == "water":
            error_msg += self._get_water_context(kwargs)
        elif generator_type == "settlement":
            error_msg += self._get_settlement_context(kwargs)

        if SHOW_FUNCTION_ARGS and (args or kwargs):
            error_msg += f"\nParameters: args={args}, kwargs={kwargs}"

        if SHOW_FULL_TRACEBACK:
            error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"

        # Log mit extra Kategorie-Info
        extra = {'category': 'CORE_GEN', 'funcName': func_name}
        self.logger.error(error_msg, extra=extra)

    def handle_heightmap_generation_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Spezialisiert für Heightmap-Generation Fehler"""
        self.handle_core_generation_error(func_name, error, "terrain", args, kwargs)

    def handle_erosion_system_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Spezialisiert für Erosion-System Fehler"""
        self.handle_core_generation_error(func_name, error, "water", args, kwargs)

    # =============================================================================
    # 2. DATA MANAGEMENT ERROR HANDLERS
    # =============================================================================

    def handle_data_management_error(self, func_name: str, error: Exception,
                                     operation_type: str, args: tuple, kwargs: dict):
        """
        Funktionsweise: Behandelt alle Data-Manager Operationen
        Aufgabe: set_data, get_data, cache, dependencies, memory
        Kategorie: data_management
        """
        if not self._should_handle_category("data_management"):
            return

        severity = self._determine_severity(error, "data_management")

        if not self.statistics.record_error("data_management", func_name, type(error).__name__, severity):
            return

        error_msg = f"DATA MANAGEMENT ERROR"
        error_msg += f"\nOperation: {operation_type}"
        error_msg += f"\nFunction: {func_name}"
        error_msg += f"\nError: {type(error).__name__}: {str(error)}"

        # Data-spezifische Kontextinformationen
        if "data_key" in kwargs:
            error_msg += f"\nData Key: {kwargs['data_key']}"
        if "generator_type" in kwargs:
            error_msg += f"\nGenerator Type: {kwargs['generator_type']}"

        extra = {'category': 'DATA_MGR', 'funcName': func_name}
        self.logger.error(error_msg, extra=extra)

    def handle_missing_key_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Spezialisiert für fehlende Data-Keys"""
        self.handle_data_management_error(func_name, error, "get_data", args, kwargs)

    def handle_cache_validation_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Spezialisiert für Cache-Validation Fehler"""
        self.handle_data_management_error(func_name, error, "cache_validation", args, kwargs)

    # =============================================================================
    # 3. GPU/SHADER ERROR HANDLERS
    # =============================================================================

    def handle_gpu_shader_error(self, func_name: str, error: Exception,
                                operation_type: str, args: tuple, kwargs: dict):
        """
        Funktionsweise: Behandelt alle GPU/Shader Operationen
        Aufgabe: OpenGL, Shader-Compilation, GPU-Memory, Compute-Execution
        Kategorie: gpu_shader
        """
        if not self._should_handle_category("gpu_shader"):
            return

        # GPU-Fehler sind oft kritisch
        severity = "CRITICAL" if "gpu" in str(error).lower() else self._determine_severity(error, "gpu_shader")

        if not self.statistics.record_error("gpu_shader", func_name, type(error).__name__, severity):
            return

        error_msg = f"GPU/SHADER ERROR"
        error_msg += f"\nOperation: {operation_type}"
        error_msg += f"\nFunction: {func_name}"
        error_msg += f"\nError: {type(error).__name__}: {str(error)}"

        # GPU-spezifische Diagnostik
        error_msg += self._get_gpu_diagnostics()

        extra = {'category': 'GPU_SHADER', 'funcName': func_name}
        self.logger.critical(error_msg, extra=extra) if severity == "CRITICAL" else self.logger.error(error_msg,
                                                                                                      extra=extra)

    def handle_shader_compilation_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Spezialisiert für Shader-Compilation Fehler"""
        self.handle_gpu_shader_error(func_name, error, "shader_compilation", args, kwargs)

    def handle_compute_execution_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Spezialisiert für Compute-Shader Execution Fehler"""
        self.handle_gpu_shader_error(func_name, error, "compute_execution", args, kwargs)

    # =============================================================================
    # 4. UI/NAVIGATION ERROR HANDLERS
    # =============================================================================

    def handle_ui_navigation_error(self, func_name: str, error: Exception,
                                   operation_type: str, args: tuple, kwargs: dict):
        """
        Funktionsweise: Behandelt UI und Navigation Fehler
        Aufgabe: Tab-Navigation, UI-Setup, Display-Updates, Event-Handling
        Kategorie: ui_navigation
        """
        if not self._should_handle_category("ui_navigation"):
            return

        severity = self._determine_severity(error, "ui_navigation")

        if not self.statistics.record_error("ui_navigation", func_name, type(error).__name__, severity):
            return

        error_msg = f"UI/NAVIGATION ERROR"
        error_msg += f"\nOperation: {operation_type}"
        error_msg += f"\nFunction: {func_name}"
        error_msg += f"\nError: {type(error).__name__}: {str(error)}"

        extra = {'category': 'UI_NAV', 'funcName': func_name}
        self.logger.warning(error_msg, extra=extra)

    # =============================================================================
    # 5. EXPORT/IMPORT ERROR HANDLERS
    # =============================================================================

    def handle_export_import_error(self, func_name: str, error: Exception,
                                   operation_type: str, args: tuple, kwargs: dict):
        """
        Funktionsweise: Behandelt Export/Import Operationen
        Aufgabe: World-Export, Parameter-Serialization, File-I/O
        Kategorie: export_import
        """
        if not self._should_handle_category("export_import"):
            return

        severity = self._determine_severity(error, "export_import")

        if not self.statistics.record_error("export_import", func_name, type(error).__name__, severity):
            return

        error_msg = f"EXPORT/IMPORT ERROR"
        error_msg += f"\nOperation: {operation_type}"
        error_msg += f"\nFunction: {func_name}"
        error_msg += f"\nError: {type(error).__name__}: {str(error)}"

        # Export-spezifische Informationen
        if "export_format" in kwargs:
            error_msg += f"\nExport Format: {kwargs['export_format']}"
        if "filename" in kwargs:
            error_msg += f"\nFilename: {kwargs['filename']}"

        extra = {'category': 'EXPORT', 'funcName': func_name}
        self.logger.error(error_msg, extra=extra)

    # =============================================================================
    # 6-12. WEITERE SPEZIALISIERTE HANDLER (kompakt)
    # =============================================================================

    def handle_initialization_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Initialization Operations"""
        self._handle_categorized_error("initialization", "INIT", func_name, error, args, kwargs)

    def handle_parameter_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Parameter Operations"""
        self._handle_categorized_error("parameters", "PARAM", func_name, error, args, kwargs)

    def handle_statistics_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Statistics/Calculation Operations"""
        self._handle_categorized_error("statistics", "STATS", func_name, error, args, kwargs)

    def handle_rendering_3d_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """3D Rendering Operations"""
        self._handle_categorized_error("rendering_3d", "3D_RENDER", func_name, error, args, kwargs)

    def handle_cleanup_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Cleanup/Resource Operations"""
        self._handle_categorized_error("cleanup", "CLEANUP", func_name, error, args, kwargs)

    def handle_cache_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Cache Operations"""
        self._handle_categorized_error("cache", "CACHE", func_name, error, args, kwargs)

    def handle_dependency_error(self, func_name: str, error: Exception, args: tuple, kwargs: dict):
        """Dependency Operations"""
        self._handle_categorized_error("dependencies", "DEPENDENCY", func_name, error, args, kwargs)

    def handle_memory_critical_error(self, func_name: str, error: MemoryError,
                                     operation_type: str, args: tuple, kwargs: dict):
        """
        Funktionsweise: Spezialisierte Behandlung für kritische Memory-Errors
        Aufgabe: Sofortige Protokollierung ohne Rate-Limiting + detaillierte Memory-Diagnostik
        Besonderheit: Sammelt umfassende Speicher-Informationen für PyCharm Debugging
        Return: None (Error wird immer weitergegeben für PyCharm)
        """
        # Memory-Errors sind IMMER kritisch - kein Category-Check
        severity = "CRITICAL"

        # Forciere Logging - Memory-Errors dürfen nie unterdrückt werden
        self.statistics.record_error("memory_critical", func_name, "MemoryError", severity)

        error_msg = f"MEMORY CRITICAL ERROR"
        error_msg += f"\nOperation: {operation_type}"
        error_msg += f"\nFunction: {func_name}"
        error_msg += f"\nError: MemoryError: {str(error)}"

        # Detaillierte Memory-Diagnostik sammeln
        error_msg += self._get_memory_diagnostics()

        # Garbage Collection Info
        error_msg += self._get_gc_diagnostics()

        # Funktionsparameter (falls relevant für Memory-Debug)
        if SHOW_FUNCTION_ARGS and (args or kwargs):
            error_msg += f"\nParameters: args={str(args)[:200]}..." if len(
                str(args)) > 200 else f"\nParameters: args={args}"
            error_msg += f", kwargs={str(kwargs)[:200]}..." if len(str(kwargs)) > 200 else f", kwargs={kwargs}"

        if SHOW_FULL_TRACEBACK:
            error_msg += f"\n\nFull Traceback:\n{traceback.format_exc()}"

        # CRITICAL Level für Memory-Errors
        extra = {'category': 'MEMORY_CRITICAL', 'funcName': func_name}
        self.logger.critical(error_msg, extra=extra)

    # =============================================================================
    # HELPER METHODEN
    # =============================================================================

    def _handle_categorized_error(self, category: str, log_prefix: str, func_name: str,
                                  error: Exception, args: tuple, kwargs: dict):
        """Generic handler für kategorisierte Errors"""
        if not self._should_handle_category(category):
            return

        severity = self._determine_severity(error, category)

        if not self.statistics.record_error(category, func_name, type(error).__name__, severity):
            return

        error_msg = f"{log_prefix} ERROR in {func_name}: {type(error).__name__}: {str(error)}"

        extra = {'category': log_prefix, 'funcName': func_name}
        getattr(self.logger, severity.lower())(error_msg, extra=extra)

    def _should_handle_category(self, category: str) -> bool:
        """Prüft ob Kategorie aktiviert ist"""
        return ERROR_HANDLER_ENABLED and ERROR_CATEGORIES.get(category, True)

    def _determine_severity(self, error: Exception, category: str) -> str:
        """
        Funktionsweise: Bestimmt Error-Severity basierend auf Error-Typ und Kategorie
        Aufgabe: Klassifiziert Errors für angemessene Logging-Level
        Besonderheit: Memory-Errors sind IMMER CRITICAL, unabhängig von Kategorie
        """
        # Memory-Errors sind IMMER kritisch (NEUE PRIORISIERUNG)
        if isinstance(error, MemoryError):
            return "CRITICAL"
        elif isinstance(error, (SystemError, OSError)) and "memory" in str(error).lower():
            return "CRITICAL"
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            return "CRITICAL"
        elif category in ["core_generation", "gpu_shader", "export_import"]:
            return "ERROR"
        else:
            return "WARNING"

    def _get_terrain_context(self, kwargs: dict) -> str:
        """Terrain-spezifische Kontextinformationen"""
        context = ""
        if "map_size" in kwargs:
            context += f"\nMap Size: {kwargs['map_size']}"
        if "amplitude" in kwargs:
            context += f"\nAmplitude: {kwargs['amplitude']}"
        return context

    def _get_water_context(self, kwargs: dict) -> str:
        """Water-spezifische Kontextinformationen"""
        context = ""
        if "erosion_strength" in kwargs:
            context += f"\nErosion Strength: {kwargs['erosion_strength']}"
        return context

    def _get_settlement_context(self, kwargs: dict) -> str:
        """Settlement-spezifische Kontextinformationen"""
        context = ""
        if "num_settlements" in kwargs:
            context += f"\nNumber of Settlements: {kwargs['num_settlements']}"
        return context

    def _get_gpu_diagnostics(self) -> str:
        """
        Funktionsweise: Sammelt GPU-Diagnostik mit spezifischen OpenGL-Error-Codes
        Aufgabe: OpenGL-Status, Error-Codes, GPU-Memory wenn verfügbar
        Besonderheit: Kein Import-Overhead - prüft ob OpenGL bereits geladen ist
        """
        try:
            # Prüfe ob OpenGL bereits importiert ist (kein neuer Import)
            if 'OpenGL.GL' in sys.modules:
                gl = sys.modules['OpenGL.GL']
                error_code = gl.glGetError()

                if error_code != gl.GL_NO_ERROR:
                    # Spezifische OpenGL Error-Codes
                    error_names = {
                        gl.GL_INVALID_ENUM: "GL_INVALID_ENUM",
                        gl.GL_INVALID_VALUE: "GL_INVALID_VALUE",
                        gl.GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
                        gl.GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
                        gl.GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION"
                    }
                    error_name = error_names.get(error_code, f"UNKNOWN_ERROR_{error_code}")
                    return f"\nGPU Error: {error_name} (Code: {error_code})"
                else:
                    return f"\nGPU Status: Active, No OpenGL Errors"
            else:
                return f"\nGPU Status: OpenGL not loaded in current session"

        except Exception as e:
            return f"\nGPU Diagnostics Error: {type(e).__name__}: {str(e)}"

    def _get_memory_diagnostics(self) -> str:
        """
        Funktionsweise: Sammelt detaillierte Memory-Diagnostik für Error-Debugging
        Aufgabe: Speicher-Status, verfügbarer RAM, Process-Memory für PyCharm Analysis
        Return: Formatierter String mit Memory-Informationen
        """
        try:
            # System Memory Info
            memory = psutil.virtual_memory()
            process = psutil.Process()

            diagnostics = f"\n--- MEMORY DIAGNOSTICS ---"
            diagnostics += f"\nSystem RAM Total: {memory.total / (1024 ** 3):.2f} GB"
            diagnostics += f"\nSystem RAM Available: {memory.available / (1024 ** 3):.2f} GB"
            diagnostics += f"\nSystem RAM Used: {memory.percent:.1f}%"

            # Process-spezifische Memory Info
            process_memory = process.memory_info()
            diagnostics += f"\nProcess Memory RSS: {process_memory.rss / (1024 ** 2):.2f} MB"
            diagnostics += f"\nProcess Memory VMS: {process_memory.vms / (1024 ** 2):.2f} MB"

            # Memory-kritische Schwellenwerte
            if memory.percent > 90:
                diagnostics += f"\n⚠️  CRITICAL: System memory usage > 90%"
            if process_memory.rss > (2 * 1024 ** 3):  # 2GB
                diagnostics += f"\n⚠️  CRITICAL: Process memory > 2GB"

            return diagnostics

        except Exception as e:
            return f"\nMemory Diagnostics Error: {type(e).__name__}: {str(e)}"

    def _get_gc_diagnostics(self) -> str:
        """
        Funktionsweise: Sammelt Garbage Collection Informationen für Memory-Error Analysis
        Aufgabe: GC-Statistiken, uncollectable Objects, Generation-Counts
        Return: Formatierter String mit GC-Informationen
        """
        try:
            # Garbage Collection Stats
            gc_stats = gc.get_stats()
            gc_counts = gc.get_count()

            diagnostics = f"\n--- GARBAGE COLLECTION DIAGNOSTICS ---"
            diagnostics += f"\nGC Generation Counts: {gc_counts}"

            # Uncollectable objects (Memory-Leaks)
            uncollectable = len(gc.garbage)
            if uncollectable > 0:
                diagnostics += f"\n⚠️  CRITICAL: {uncollectable} uncollectable objects detected"

            # GC-Threshold Info
            thresholds = gc.get_threshold()
            diagnostics += f"\nGC Thresholds: {thresholds}"

            # Force GC und Berichte Ergebnis
            collected = gc.collect()
            if collected > 0:
                diagnostics += f"\nGC Run: Collected {collected} objects"

            return diagnostics

        except Exception as e:
            return f"\nGC Diagnostics Error: {type(e).__name__}: {str(e)}"

    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Funktionsweise: Gibt umfassende Error-Statistiken zurück
        Return: Dict mit allen Error-Statistiken
        """
        return {
            'total_errors': self.statistics.total_errors,
            'errors_by_category': self.statistics.errors_by_category,
            'errors_by_function': self.statistics.errors_by_function,
            'critical_errors': len(self.statistics.critical_errors),
            'most_problematic_category': max(self.statistics.errors_by_category.items(),
                                             key=lambda x: x[1]) if self.statistics.errors_by_category else None,
            'most_problematic_function': max(self.statistics.errors_by_function.items(),
                                             key=lambda x: x[1]) if self.statistics.errors_by_function else None
        }


# =============================================================================
# GLOBALE ERROR HANDLER INSTANZ
# =============================================================================

_error_handler = MapGeneratorErrorHandler()


# =============================================================================
# DECORATOR FUNKTIONEN (Kategorisiert)
# =============================================================================

def core_generation_handler(generator_type: str):
    """Decorator für Core-Generation Funktionen"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ERROR_HANDLER_ENABLED:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _error_handler.handle_core_generation_error(func.__name__, e, generator_type, args, kwargs)
                raise e

        return wrapper

    return decorator


def data_management_handler(operation_type: str):
    """Decorator für Data-Management Funktionen"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ERROR_HANDLER_ENABLED:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _error_handler.handle_data_management_error(func.__name__, e, operation_type, args, kwargs)
                raise e

        return wrapper

    return decorator


def gpu_shader_handler(operation_type: str):
    """Decorator für GPU/Shader Funktionen"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ERROR_HANDLER_ENABLED:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                _error_handler.handle_gpu_shader_error(func.__name__, e, operation_type, args, kwargs)
                raise e

        return wrapper

    return decorator


# Weitere Decorator für andere Kategorien...
def ui_navigation_handler(func: Callable) -> Callable:
    """Decorator für UI/Navigation Funktionen"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ERROR_HANDLER_ENABLED:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _error_handler.handle_ui_navigation_error(func.__name__, e, "ui_operation", args, kwargs)
            raise e

    return wrapper


def export_import_handler(func: Callable) -> Callable:
    """Decorator für Export/Import Funktionen"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not ERROR_HANDLER_ENABLED:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _error_handler.handle_export_import_error(func.__name__, e, "export_import", args, kwargs)
            raise e

    return wrapper

def memory_critical_handler(operation_type: str = "memory_operation"):
    """
    Funktionsweise: Decorator für Memory-kritische Funktionen
    Aufgabe: Automatische Memory-Error Behandlung mit Spezial-Logging
    Verwendung: @memory_critical_handler("array_processing") für große Array-Operationen
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ERROR_HANDLER_ENABLED:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except MemoryError as e:
                _error_handler.handle_memory_critical_error(func.__name__, e, operation_type, args, kwargs)
                raise e  # Immer weiterleiten für PyCharm Debugging
            except Exception as e:
                # Andere Errors normal behandeln
                if "memory" in str(e).lower():
                    _error_handler.handle_memory_critical_error(func.__name__, MemoryError(str(e)), operation_type, args, kwargs)
                raise e
        return wrapper
    return decorator


# =============================================================================
# UTILITY FUNKTIONEN
# =============================================================================

def toggle_error_handler(enabled: bool):
    """Ein/Ausschalten des Error Handlers"""
    global ERROR_HANDLER_ENABLED
    ERROR_HANDLER_ENABLED = enabled
    print(f"Map Generator Error Handler {'aktiviert' if enabled else 'deaktiviert'}")


def toggle_error_category(category: str, enabled: bool):
    """Ein/Ausschalten einzelner Error-Kategorien"""
    if category in ERROR_CATEGORIES:
        ERROR_CATEGORIES[category] = enabled
        print(f"Error Category '{category}' {'aktiviert' if enabled else 'deaktiviert'}")


def get_error_statistics():
    """Gibt aktuelle Error-Statistiken zurück"""
    return _error_handler.get_statistics_summary()


def print_error_summary():
    """Druckt zusammengefasste Error-Statistiken"""
    stats = get_error_statistics()

    print("\n" + "=" * 60)
    print("MAP GENERATOR ERROR HANDLER STATISTICS")
    print("=" * 60)
    print(f"Total Errors: {stats['total_errors']}")
    print(f"Critical Errors: {stats['critical_errors']}")
    print(f"Error Handler Status: {'Aktiv' if ERROR_HANDLER_ENABLED else 'Inaktiv'}")

    print("\nErrors by Category:")
    for category, count in stats['errors_by_category'].items():
        if count > 0:
            status = "ON" if ERROR_CATEGORIES[category] else "OFF"
            print(f"  {category}: {count} errors ({status})")

    if stats['most_problematic_category']:
        cat, count = stats['most_problematic_category']
        print(f"\nMost Problematic Category: {cat} ({count} errors)")

    if stats['most_problematic_function']:
        func, count = stats['most_problematic_function']
        print(f"Most Problematic Function: {func} ({count} errors)")

    print("=" * 60)


# =============================================================================
# QUICK USAGE BEISPIELE
# =============================================================================

if __name__ == "__main__":
    # Beispiele für die Verwendung

    @core_generation_handler("terrain")
    def example_terrain_generation():
        """Beispiel Terrain-Generation mit Error Handler"""
        raise ValueError("Beispiel Terrain-Fehler")


    @data_management_handler("set_data")
    def example_data_operation():
        """Beispiel Data-Operation mit Error Handler"""
        raise KeyError("Beispiel Data-Key Fehler")


    print("Error Handler Test...")
    print(f"Status: {'Aktiv' if ERROR_HANDLER_ENABLED else 'Inaktiv'}")

    # Test verschiedene Error-Kategorien
    try:
        example_terrain_generation()
    except ValueError:
        pass

    try:
        example_data_operation()
    except KeyError:
        pass

    # Statistiken anzeigen
    print_error_summary()

    # Kategorie ausschalten und testen
    toggle_error_category("core_generation", False)

    try:
        example_terrain_generation()
    except ValueError:
        pass

    print_error_summary()