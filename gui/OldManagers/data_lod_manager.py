"""
Path: gui/managers/data_lod_manager.py

Funktionsweise: Zentrale Datenverwaltung mit numerischem LOD-System, Signal-basierter Kommunikation und integriertem Resource-Management
- Numerische LOD-Levels (1, 2, 3, ...) statt String-basiert
- Map-size-proportionale Skalierung für alle Generatoren
- LODCommunicationHub für Tab-übergreifende Koordination
- Memory-effiziente Array-Referenzen ohne Kopien
- Automatische Dependency-Checking auf gleicher LOD-Stufe
- Signal-basierte Status-Updates für UI-Integration
- Integriertes Resource-Management (ResourceTracker, DisplayUpdateManager)
- Memory-Leak-Prevention und systematisches Cleanup
- ERWEITERT: Data-Key-Level Generator Integration und Automatisierung

INTEGRIERTE FEATURES:
- DataManager: Numerisches LOD-System, Generator-Daten-Speicherung, Cache-Management
- LODCommunicationHub: Signal-basierte Tab-Kommunikation, Dependency-Tracking
- ResourceTracker: Systematisches Resource-Management, Memory-Leak-Prevention
- DisplayUpdateManager: Change-Detection für Display-Updates, Performance-Optimierung
- DataKeyGeneratorSystem: Automatische Data-Key-Generierung, Generator-Registry, Signal-Integration
"""

import numpy as np
import weakref
import time
import hashlib
import logging
import threading
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Tuple, Callable, Set
from dataclasses import dataclass, field
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer
from abc import abstractmethod
from enum import Enum
import queue

from gui.config.value_default import TERRAIN


def get_data_manager_error_decorators():
    """Lazy Loading von Data Manager Error Decorators"""
    try:
        from gui.utils.error_handler import data_management_handler, memory_critical_handler
        return data_management_handler, memory_critical_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        return noop_decorator, noop_decorator


data_management_handler, memory_critical_handler = get_data_manager_error_decorators()


# =============================================================================
# LOD SYSTEM CORE FUNCTIONS UND ENUMS
# =============================================================================

def calculate_lod_size(lod_level: int, map_size_min: int = None) -> int:
    """
    Berechnet LOD-Size aus LOD-Level
    lod_size = map_size_min * 2^(lod_level-1)
    """
    if map_size_min is None:
        map_size_min = TERRAIN.MAPSIZEMIN
    return map_size_min * (2 ** (lod_level - 1))


def calculate_max_lod_for_size(target_map_size: int, map_size_min: int = None) -> int:
    """
    Berechnet maximale LOD für gegebene Map-Size.
    Zählt Verdopplungsschritte direkt mit (statt int(log2(...)) - das rundet bei
    Nicht-Zweierpotenz-Zielgrößen wie 96 oder 288 immer ab und lässt die letzte,
    von get_map_size_for_lod() ohnehin auf target_map_size geklemmte Stufe weg,
    wodurch die tatsächlich eingestellte map_size nie erreicht wurde.
    """
    if map_size_min is None:
        map_size_min = TERRAIN.MAPSIZEMIN
    lod = 1
    size = map_size_min
    while size < target_map_size:
        size *= 2
        lod += 1
    return lod


class GenerationMode(Enum):
    """Generierungsmodi für Data-Keys"""
    MANUAL = "manual"
    AUTO = "auto"
    DEPENDENCY_TRIGGERED = "dependency-triggered"
    RECOVERY = "recovery"


class GenerationPriority(Enum):
    """Prioritäten für Generator-Queue"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# =============================================================================
# DATACLASSES FÜR STATUS UND KONFIGURATION
# =============================================================================

@dataclass
class TabStatus:
    """Status-Information für einen Generator-Tab"""
    tab: str
    lod_level: int
    lod_size: int
    lod_status: str  # "idle", "pending", "success", "failure"
    progress_percent: int = 0
    error_message: str = ""
    available_data_keys: List[str] = field(default_factory=list)


@dataclass
class DataStatus:
    """Status-Information für ein Dataset - ERWEITERT"""
    data: str
    lod_level: int
    lod_size: int
    lod_status: str  # "idle", "pending", "success", "failure"
    progress_percent: int = 0
    error_message: str = ""
    available_data_keys: List[str] = field(default_factory=list)
    # NEUE FELDER für erweiterte Data-Key-Verwaltung
    dependency_status: Dict[str, str] = field(default_factory=dict)  # {dependency: status}
    generation_mode: str = "manual"  # "auto", "manual", "dependency-triggered"
    last_generation_time: float = 0.0
    generation_duration: float = 0.0
    parameter_hash: str = ""
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class GenerationTask:
    """Task für Generator-Queue"""
    data_key: str
    lod_level: int
    parameters: Dict[str, Any]
    priority: GenerationPriority
    mode: GenerationMode
    created_time: float = field(default_factory=time.time)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout_seconds: float = 300.0  # 5 minutes default

    def __lt__(self, other):
        """Für Priority Queue Sortierung"""
        return self.priority.value > other.priority.value


@dataclass
class LODConfig:
    """LOD-Konfiguration für Map-size-proportionale Skalierung"""
    minimal_map_size: int  # z.B. 32 aus TerrainConstants
    target_map_size: int  # z.B. 288 aus Parameter
    max_terrain_lod: int = 7  # Maximale Sonnenstände

    def get_map_size_for_lod(self, lod_level: int) -> int:
        """
        Berechnet Map-Size für LOD-Level
        LOD 1=32, LOD 2=64, LOD 3=128, LOD 4=256, LOD 5+=target_map_size
        """
        if lod_level <= 0:
            return self.minimal_map_size

        calculated_size = self.minimal_map_size * (2 ** (lod_level - 1))

        # Bei Überschreitung der target_map_size: target_map_size verwenden
        if calculated_size > self.target_map_size:
            return self.target_map_size

        return calculated_size

    def get_max_lod_for_map_size(self) -> int:
        """Berechnet maximale sinnvolle LOD für target_map_size (siehe calculate_max_lod_for_size)"""
        return calculate_max_lod_for_size(self.target_map_size, self.minimal_map_size)


@dataclass
class ResourceInfo:
    """Information über eine tracked Resource"""
    resource_id: str
    resource_type: str
    creation_time: float
    cleanup_func: Optional[Callable]
    estimated_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DEPENDENCY MATRIZEN UND KONFIGURATION
# =============================================================================

# Tab-Dependencies: Data-abhängig, gleiche LOD-Stufe
TAB_DEPENDENCY_MATRIX = {
    "terrain": ["heightmap_combined", "slopemap", "shadowmap"],
    "geology": ["rock_map", "hardness_map"],
    "weather": ["wind_map", "temp_map", "precip_map", "humid_map"],
    "water": ["flow_map", "flow_speed", "cross_section", "soil_moist_map", "erosion_map", "sedimentation_map",
              "evaporation_map", "ocean_outflow", "water_biomes_map"],
    "biome": ["biome_map", "biome_map_super", "super_biome_mask", "climate_classification", "biome_statistics"],
    "settlement": ["settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map"]
}

# Data-Dependencies: Data-abhängig, gleiche LOD-Stufe
DATA_DEPENDENCY_MATRIX = {
    "heightmap": [],
    "heightmap_combined": ["heightmap"],
    "slopemap": ["heightmap_combined"],
    "shadowmap": ["heightmap_combined"],
    "rock_map": ["heightmap_combined"],
    "hardness_map": ["rock_map"],
    "wind_map": ["heightmap_combined"],
    "temp_map": ["heightmap_combined"],
    "precip_map": ["heightmap_combined"],
    "humid_map": ["heightmap_combined"],
    "water_map": ["heightmap_combined"],
    "flow_map": ["heightmap_combined"],
    "flow_speed": ["heightmap_combined"],
    "cross_section": ["heightmap_combined"],
    "soil_moist_map": ["heightmap_combined"],
    "erosion_map": ["heightmap_combined"],
    "sedimentation_map": ["heightmap_combined"],
    "evaporation_map": ["heightmap_combined"],
    "ocean_outflow": ["heightmap_combined"],
    "water_biomes_map": ["heightmap_combined"],
    "biome_map": ["heightmap_combined"],
    "biome_map_super": ["heightmap_combined"],
    "super_biome_mask": ["heightmap_combined"],
    "climate_classification": ["heightmap_combined"],
    "biome_statistics": ["heightmap_combined"],
    "settlement_list": ["heightmap_combined"],
    "landmark_list": ["heightmap_combined"],
    "roadsite_list": ["heightmap_combined"],
    "plot_map": ["heightmap_combined"],
    "civ_map": ["heightmap_combined"]
}

# NEUE KONFIGURATION: Auto-Generation für Data-Keys
AUTO_GENERATION_DATA_KEYS = {
    # Terrain-Chain: heightmap → heightmap_combined → slopemap, shadowmap
    "heightmap_combined": {"auto": True, "priority": GenerationPriority.HIGH},
    "slopemap": {"auto": True, "priority": GenerationPriority.NORMAL},
    "shadowmap": {"auto": True, "priority": GenerationPriority.NORMAL},

    # Geology-Chain: heightmap_combined → rock_map → hardness_map
    "rock_map": {"auto": True, "priority": GenerationPriority.NORMAL},
    "hardness_map": {"auto": True, "priority": GenerationPriority.LOW},

    # Weather: Alle abhängig von heightmap_combined
    "wind_map": {"auto": True, "priority": GenerationPriority.NORMAL},
    "temp_map": {"auto": True, "priority": GenerationPriority.NORMAL},
    "precip_map": {"auto": True, "priority": GenerationPriority.NORMAL},
    "humid_map": {"auto": True, "priority": GenerationPriority.LOW},

    # Water: Alle abhängig von heightmap_combined
    "flow_map": {"auto": True, "priority": GenerationPriority.NORMAL},
    "flow_speed": {"auto": True, "priority": GenerationPriority.LOW},
    "cross_section": {"auto": True, "priority": GenerationPriority.LOW},

    # Biome: Abhängig von heightmap_combined
    "biome_map": {"auto": True, "priority": GenerationPriority.NORMAL},
    "biome_map_super": {"auto": True, "priority": GenerationPriority.LOW},
}

# Data-Key zu Tab-Mapping
DATA_KEY_TO_TAB_MAPPING = {
    "heightmap": "terrain",
    "heightmap_combined": "terrain",
    "slopemap": "terrain",
    "shadowmap": "terrain",
    "rock_map": "geology",
    "hardness_map": "geology",
    "wind_map": "weather",
    "temp_map": "weather",
    "precip_map": "weather",
    "humid_map": "weather",
    "flow_map": "water",
    "flow_speed": "water",
    "cross_section": "water",
    "soil_moist_map": "water",
    "erosion_map": "water",
    "sedimentation_map": "water",
    "evaporation_map": "water",
    "ocean_outflow": "water",
    "water_biomes_map": "water",
    "biome_map": "biome",
    "biome_map_super": "biome",
    "super_biome_mask": "biome",
    "climate_classification": "biome",
    "biome_statistics": "biome",
    "settlement_list": "settlement",
    "landmark_list": "settlement",
    "roadsite_list": "settlement",
    "plot_map": "settlement",
    "civ_map": "settlement"
}


# =============================================================================
# DATA-KEY GENERATOR SYSTEM
# =============================================================================

class DataKeyGeneratorBase(QObject):
    """
    Basisklasse für Data-Key-Generatoren - PUNKT 1 LÖSUNG
    Definiert Standard-Interface für alle Data-Key-spezifischen Generatoren
    Ermöglicht einheitliche Integration in das LOD-System mit Signal-Handling

    Diese Klasse löst Punkt 1: "Data-Key Generator Integration"
    - Einheitliches Interface für alle Data-Key-Generatoren
    - Automatisches Signal-Handling für LOD-Events
    - Parameter-Validation und Error-Handling
    - Progress-Reporting während Generierung
    - Resource-Management Integration
    """

    # Standard-Signale für alle Data-Key-Generatoren
    data_lod_started = pyqtSignal(str, int, int)  # (data_key, lod_level, lod_size)
    data_lod_progress = pyqtSignal(str, int, int)  # (data_key, lod_level, progress_percent)
    data_lod_completed = pyqtSignal(str, int, bool)  # (data_key, lod_level, success)
    data_lod_failed = pyqtSignal(str, int, str)  # (data_key, lod_level, error_message)

    def __init__(self, data_key: str):
        super().__init__()
        self.data_key = data_key
        self.logger = logging.getLogger(f"{__name__}.{data_key}")
        self.generation_start_time = 0.0
        self._current_parameters = {}
        self._generation_active = False
        self._current_lod_level = 0

    @abstractmethod
    def generate_data_key(self, lod_level: int, lod_size: int, parameters: Dict[str, Any]) -> Any:
        """
        Generiert spezifischen Data-Key für LOD-Level
        Muss von Subklassen implementiert werden

        Parameter:
            lod_level: LOD-Level (1, 2, 3, ...)
            lod_size: Map-Größe für dieses LOD-Level
            parameters: Generator-Parameter

        Return:
            Generierte Daten (numpy array, object, etc.)

        Raises:
            Exception: Bei Generierungsfehlern
        """
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validiert Parameter für Data-Key-Generierung
        Muss von Subklassen implementiert werden

        Parameter:
            parameters: Zu validierende Parameter

        Return:
            (valid: bool, error_message: str)
        """
        pass

    def execute_generation(self, lod_level: int, parameters: Dict[str, Any]) -> bool:
        """
        Führt vollständige Data-Key-Generierung mit Signal-Handling aus
        Diese Methode implementiert den kompletten Generation-Workflow

        Parameter:
            lod_level: LOD-Level für Generierung
            parameters: Generator-Parameter

        Return:
            bool - Generierung erfolgreich
        """
        if self._generation_active:
            self.logger.warning(f"Generation already active for {self.data_key}")
            return False

        try:
            # Parameter validieren
            valid, error_msg = self.validate_parameters(parameters)
            if not valid:
                error_message = f"Parameter validation failed: {error_msg}"
                self.logger.error(f"Generation failed for {self.data_key} LOD {lod_level}: {error_message}")
                self.data_lod_failed.emit(self.data_key, lod_level, error_message)
                return False

            # LOD-Size berechnen
            lod_size = calculate_lod_size(lod_level)

            # Generation initialisieren
            self._generation_active = True
            self._current_lod_level = lod_level
            self.generation_start_time = time.time()
            self._current_parameters = parameters.copy()

            self.logger.info(f"Starting generation for {self.data_key} LOD {lod_level} (size: {lod_size})")

            # Start-Signal emittieren
            self.data_lod_started.emit(self.data_key, lod_level, lod_size)

            # Initial Progress-Report
            self.data_lod_progress.emit(self.data_key, lod_level, 0)

            # Actual generation durch Subklasse
            result = self.generate_data_key(lod_level, lod_size, parameters)

            # Erfolg verarbeiten
            duration = time.time() - self.generation_start_time
            self.logger.info(f"Generation completed for {self.data_key} LOD {lod_level} in {duration:.2f}s")

            # Final Progress und Success-Signal
            self.data_lod_progress.emit(self.data_key, lod_level, 100)
            self.data_lod_completed.emit(self.data_key, lod_level, True)

            self._generation_active = False
            return True

        except Exception as e:
            # Fehler-Handling
            duration = time.time() - self.generation_start_time
            error_msg = f"Generation failed after {duration:.2f}s: {str(e)}"

            self.logger.error(f"Generation failed for {self.data_key} LOD {lod_level}: {e}")
            self.data_lod_failed.emit(self.data_key, lod_level, error_msg)

            self._generation_active = False
            return False

    def report_progress(self, progress_percent: int):
        """
        Helper-Methode für Progress-Reporting während Generierung
        Kann von generate_data_key() aufgerufen werden

        Parameter:
            progress_percent: Fortschritt in Prozent (0-100)
        """
        if self._generation_active:
            progress_percent = max(0, min(100, progress_percent))  # Clamp to 0-100
            self.data_lod_progress.emit(self.data_key, self._current_lod_level, progress_percent)

    def get_estimated_duration(self, lod_level: int, lod_size: int) -> float:
        """
        Schätzt Generierungsdauer für Data-Key
        Kann von Subklassen überschrieben werden für spezifische Schätzungen

        Parameter:
            lod_level: LOD-Level
            lod_size: Map-Größe

        Return:
            Geschätzte Dauer in Sekunden
        """
        # Default-Implementierung basierend auf LOD-Size
        base_time = 1.0  # 1 Sekunde für LOD 1
        complexity_factor = (lod_size / 32) ** 0.5  # Quadratisch mit Größe
        return base_time * complexity_factor

    def get_required_dependencies(self) -> List[str]:
        """
        Gibt erforderliche Dependencies für diesen Data-Key zurück

        Return:
            Liste von Data-Key-Namen die als Dependencies benötigt werden
        """
        return DATA_DEPENDENCY_MATRIX.get(self.data_key, [])

    def supports_incremental_generation(self) -> bool:
        """
        Gibt an ob Generator inkrementelle Generierung unterstützt
        Kann von Subklassen überschrieben werden

        Return:
            bool - Unterstützt inkrementelle Generierung
        """
        return False

    def get_memory_requirements(self, lod_level: int, lod_size: int) -> int:
        """
        Schätzt Memory-Requirements für Generierung
        Kann von Subklassen überschrieben werden für genauere Schätzungen

        Parameter:
            lod_level: LOD-Level
            lod_size: Map-Größe

        Return:
            Geschätzte Memory-Usage in Bytes
        """
        # Default: Annahme von 4 Bytes pro Pixel (float32) plus Overhead
        base_memory = lod_size * lod_size * 4  # Hauptdaten
        overhead = base_memory * 0.5  # 50% Overhead für temporäre Daten
        return int(base_memory + overhead)

    def get_generator_metadata(self) -> Dict[str, Any]:
        """
        Gibt Metadata über den Generator zurück
        Kann von Subklassen erweitert werden

        Return:
            Dictionary mit Generator-Metadaten
        """
        return {
            "data_key": self.data_key,
            "generator_class": self.__class__.__name__,
            "supports_incremental": self.supports_incremental_generation(),
            "dependencies": self.get_required_dependencies(),
            "is_active": self._generation_active
        }

    def cleanup_resources(self):
        """
        Cleanup-Methode für Generator-spezifische Ressourcen
        Sollte von Subklassen erweitert werden wenn nötig
        """
        self._generation_active = False
        self._current_parameters.clear()
        self._current_lod_level = 0
        self.logger.debug(f"Cleaned up resources for {self.data_key}")

    def is_generation_active(self) -> bool:
        """
        Gibt an ob Generation aktuell läuft

        Return:
            bool - Generation ist aktiv
        """
        return self._generation_active

    def get_current_lod_level(self) -> int:
        """
        Gibt aktuelles LOD-Level zurück (falls Generation aktiv)

        Return:
            int - Aktuelles LOD-Level oder 0 wenn inaktiv
        """
        return self._current_lod_level if self._generation_active else 0

    def abort_generation(self, reason: str = "User abort"):
        """
        Bricht laufende Generierung ab

        Parameter:
            reason: Grund für Abbruch
        """
        if self._generation_active:
            self.logger.warning(f"Aborting generation for {self.data_key}: {reason}")
            self.data_lod_failed.emit(self.data_key, self._current_lod_level, f"Aborted: {reason}")
            self._generation_active = False
            self._current_lod_level = 0

class DataKeyGeneratorRegistry(QObject):
    """
    Registry für alle Data-Key-Generatoren - PUNKT 1 FORTSETZUNG
    Verwaltet Generator-Instanzen und ermöglicht dynamische Registrierung
    Zentrale Anlaufstelle für Data-Key-Generator-Management

    Diese Klasse löst Punkt 1: "Data-Key Generator Integration" (Fortsetzung)
    - Zentrale Registry für alle Data-Key-Generatoren
    - Dynamische Registrierung und Deregistrierung
    - Validation aller Generatoren
    - Thread-sichere Verwaltung
    - Signal-Integration für Registry-Events
    """

    # Signale für Registry-Events
    generator_registered = pyqtSignal(str, str)  # (data_key, generator_type)
    generator_unregistered = pyqtSignal(str)  # (data_key)
    registry_validation_completed = pyqtSignal(dict)  # validation_results

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Generator-Registry
        self._generators: Dict[str, DataKeyGeneratorBase] = {}
        self._generator_types: Dict[str, str] = {}  # {data_key: generator_class_name}
        self._generator_metadata: Dict[str, Dict[str, Any]] = {}

        # Thread-Safety
        self._registry_lock = threading.RLock()

    def register_generator(self, generator: DataKeyGeneratorBase,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Registriert Data-Key-Generator
        Parameter: generator, metadata
        Return: bool - Registrierung erfolgreich
        """
        with self._registry_lock:
            try:
                data_key = generator.data_key

                if data_key in self._generators:
                    self.logger.warning(f"Generator for '{data_key}' already registered, replacing")
                    # Cleanup des alten Generators
                    old_generator = self._generators[data_key]
                    old_generator.cleanup_resources()

                # Generator registrieren
                self._generators[data_key] = generator
                self._generator_types[data_key] = generator.__class__.__name__
                self._generator_metadata[data_key] = metadata or {}

                self.logger.info(f"Registered generator for data key '{data_key}': {generator.__class__.__name__}")
                self.generator_registered.emit(data_key, generator.__class__.__name__)

                return True

            except Exception as e:
                self.logger.error(f"Failed to register generator for '{generator.data_key}': {e}")
                return False

    def unregister_generator(self, data_key: str) -> bool:
        """
        Deregistriert Data-Key-Generator
        Parameter: data_key
        Return: bool - Deregistrierung erfolgreich
        """
        with self._registry_lock:
            if data_key not in self._generators:
                self.logger.warning(f"No generator registered for '{data_key}'")
                return False

            try:
                generator = self._generators[data_key]
                generator.cleanup_resources()

                del self._generators[data_key]
                del self._generator_types[data_key]
                del self._generator_metadata[data_key]

                self.logger.info(f"Unregistered generator for data key '{data_key}'")
                self.generator_unregistered.emit(data_key)

                return True

            except Exception as e:
                self.logger.error(f"Failed to unregister generator for '{data_key}': {e}")
                return False

    def get_generator(self, data_key: str) -> Optional[DataKeyGeneratorBase]:
        """
        Gibt Generator für Data-Key zurück
        Parameter: data_key
        Return: Generator-Instanz oder None
        """
        with self._registry_lock:
            return self._generators.get(data_key)

    def has_generator(self, data_key: str) -> bool:
        """
        Prüft ob Generator für Data-Key registriert ist
        Parameter: data_key
        Return: bool
        """
        with self._registry_lock:
            return data_key in self._generators

    def get_all_data_keys(self) -> List[str]:
        """
        Gibt alle registrierten Data-Keys zurück
        Return: Liste von Data-Key-Namen
        """
        with self._registry_lock:
            return list(self._generators.keys())

    def get_generators_for_tab(self, tab_name: str) -> List[DataKeyGeneratorBase]:
        """
        Gibt alle Generatoren für einen Tab zurück
        Parameter: tab_name
        Return: Liste von Generator-Instanzen
        """
        with self._registry_lock:
            tab_data_keys = TAB_DEPENDENCY_MATRIX.get(tab_name, [])
            generators = []

            for data_key in tab_data_keys:
                if data_key in self._generators:
                    generators.append(self._generators[data_key])

            return generators

    def validate_all_generators(self) -> Dict[str, bool]:
        """
        Validiert alle registrierten Generatoren
        Return: {data_key: is_valid}
        """
        validation_results = {}

        with self._registry_lock:
            for data_key, generator in self._generators.items():
                try:
                    # Basic validation - kann von Subklassen erweitert werden
                    test_params = {"lod_level": 1, "map_size": 32}
                    valid, error_msg = generator.validate_parameters(test_params)
                    validation_results[data_key] = valid

                    if not valid:
                        self.logger.warning(f"Generator validation failed for '{data_key}': {error_msg}")

                except Exception as e:
                    validation_results[data_key] = False
                    self.logger.error(f"Generator validation error for '{data_key}': {e}")

        self.registry_validation_completed.emit(validation_results)
        return validation_results

    def cleanup_all_generators(self):
        """
        Cleanup aller registrierten Generatoren
        """
        with self._registry_lock:
            for data_key in list(self._generators.keys()):
                self.unregister_generator(data_key)

            self.logger.info("All generators cleaned up")

    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Gibt Registry-Statistiken zurück
        Return: Dictionary mit Statistiken
        """
        with self._registry_lock:
            active_generators = sum(1 for gen in self._generators.values() if gen.is_generation_active())

            return {
                "total_generators": len(self._generators),
                "active_generations": active_generators,
                "generator_types": dict(self._generator_types),
                "registered_data_keys": list(self._generators.keys()),
                "generators_by_tab": {
                    tab: [dk for dk in data_keys if dk in self._generators]
                    for tab, data_keys in TAB_DEPENDENCY_MATRIX.items()
                }
            }

# =============================================================================
# GENERATOR QUEUE - PUNKT 3 LÖSUNG
# =============================================================================

class GeneratorQueue(QObject):
    """
    Priority-Queue für Data-Key-Generierung - PUNKT 3 LÖSUNG
    Verwaltet Generation-Tasks mit Prioritäten und Dependencies
    Implementiert Task-Scheduling und Conflict-Resolution

    Diese Klasse löst Punkt 3: "Automatische Data-Generierung implementieren"
    - Priority-Queue für Generator-Tasks
    - Dependency-Resolution vor Task-Start
    - Concurrent Task-Limiting
    - Timeout-Handling
    - Task-Retry-Mechanismus
    """

    # Signale für Queue-Events
    task_queued = pyqtSignal(str, int, str)  # (data_key, lod_level, priority)
    task_started = pyqtSignal(str, int)  # (data_key, lod_level)
    task_completed = pyqtSignal(str, int, bool)  # (data_key, lod_level, success)
    task_timeout = pyqtSignal(str, int, float)  # (data_key, lod_level, timeout_seconds)
    queue_empty = pyqtSignal()

    def __init__(self, max_concurrent_tasks: int = 3):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Queue-Management
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._active_tasks: Dict[str, GenerationTask] = {}  # {task_id: task}
        self._completed_tasks: Dict[str, GenerationTask] = {}
        self._failed_tasks: Dict[str, GenerationTask] = {}

        # Concurrent Task-Limiting
        self.max_concurrent_tasks = max_concurrent_tasks
        self._current_task_count = 0

        # Thread-Safety
        self._queue_lock = threading.RLock()

        # Dependency-Check-Callback
        self._check_dependencies_callback: Optional[Callable[[GenerationTask], bool]] = None

        # Task-Processing Timer
        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self._process_queue)
        self.processing_timer.start(1000)  # Check every second

        # Statistics
        self._total_tasks_processed = 0
        self._total_failed_tasks = 0

    def enqueue_task(self, task: GenerationTask) -> str:
        """
        Fügt Task zur Queue hinzu
        Parameter: task
        Return: task_id
        """
        with self._queue_lock:
            task_id = f"{task.data_key}_lod_{task.lod_level}_{int(task.created_time)}"

            # Prüfe ob bereits in Queue oder aktiv
            if self._is_task_active_or_queued(task.data_key, task.lod_level):
                self.logger.debug(f"Task for {task.data_key} LOD {task.lod_level} already active/queued")
                return task_id

            # Task in Priority Queue einreihen
            priority_tuple = (task.priority.value, task.created_time, task_id, task)
            self._task_queue.put(priority_tuple)

            self.logger.debug(
                f"Enqueued task: {task.data_key} LOD {task.lod_level} (Priority: {task.priority.name})")
            self.task_queued.emit(task.data_key, task.lod_level, task.priority.name)

            return task_id

    def _is_task_active_or_queued(self, data_key: str, lod_level: int) -> bool:
        """
        Prüft ob Task bereits aktiv oder in Queue
        Parameter: data_key, lod_level
        Return: bool
        """
        # Prüfe aktive Tasks
        for task in self._active_tasks.values():
            if task.data_key == data_key and task.lod_level == lod_level:
                return True

        # Prüfe Queue - aufwendiger aber notwendig
        with self._task_queue.mutex:
            for priority_tuple in self._task_queue.queue:
                task = priority_tuple[3]
                if task.data_key == data_key and task.lod_level == lod_level:
                    return True

        return False

    def _process_queue(self):
        """
        Verarbeitet Queue und startet neue Tasks
        """
        if not getattr(self, "_first_tick_logged", False):
            self._first_tick_logged = True
            self.logger.info("GeneratorQueue processing timer: first tick")

        with self._queue_lock:
            try:
                # Entferne abgeschlossene Tasks aus aktiver Liste
                completed_task_ids = []
                for task_id, task in self._active_tasks.items():
                    # Timeout-Check
                    if time.time() - task.created_time > task.timeout_seconds:
                        self.logger.warning(f"Task {task_id} timed out after {task.timeout_seconds}s")
                        completed_task_ids.append(task_id)
                        self._failed_tasks[task_id] = task
                        self.task_timeout.emit(task.data_key, task.lod_level, task.timeout_seconds)

                for task_id in completed_task_ids:
                    del self._active_tasks[task_id]
                    self._current_task_count -= 1

                # Starte neue Tasks falls Kapazität verfügbar
                while (self._current_task_count < self.max_concurrent_tasks and
                       not self._task_queue.empty()):

                    try:
                        priority_tuple = self._task_queue.get_nowait()
                        task_id = priority_tuple[2]
                        task = priority_tuple[3]

                        # Prüfe Dependencies vor Start
                        if self._check_dependencies_callback and not self._check_dependencies_callback(task):
                            # Task zurück in Queue - niedrigere Priorität
                            delayed_task = GenerationTask(
                                data_key=task.data_key,
                                lod_level=task.lod_level,
                                parameters=task.parameters,
                                priority=GenerationPriority.LOW,
                                mode=task.mode,
                                dependencies=task.dependencies,
                                retry_count=task.retry_count + 1
                            )

                            if delayed_task.retry_count < 5:  # Max 5 Retries
                                self.enqueue_task(delayed_task)
                            else:
                                self.logger.error(f"Task {task_id} failed due to unresolved dependencies")
                                self._failed_tasks[task_id] = task
                                self._total_failed_tasks += 1
                        else:
                            self._start_task(task_id, task)

                    except queue.Empty:
                        break  # Queue empty

                # Emit queue empty signal wenn nötig
                if (self._task_queue.empty() and
                        self._current_task_count == 0):
                    self.queue_empty.emit()

            except Exception as e:
                self.logger.error(f"Queue processing error: {e}")

    def _start_task(self, task_id: str, task: GenerationTask):
        """
        Startet Task-Ausführung
        Parameter: task_id, task
        """
        self._active_tasks[task_id] = task
        self._current_task_count += 1

        self.logger.info(f"Starting task: {task.data_key} LOD {task.lod_level}")
        self.task_started.emit(task.data_key, task.lod_level)

    def mark_task_completed(self, data_key: str, lod_level: int, success: bool):
        """
        Markiert Task als abgeschlossen
        Parameter: data_key, lod_level, success
        """
        with self._queue_lock:
            # Finde entsprechenden Task
            task_id_to_remove = None
            for task_id, task in self._active_tasks.items():
                if task.data_key == data_key and task.lod_level == lod_level:
                    task_id_to_remove = task_id
                    break

            if task_id_to_remove:
                task = self._active_tasks[task_id_to_remove]
                del self._active_tasks[task_id_to_remove]
                self._current_task_count -= 1
                self._total_tasks_processed += 1

                if success:
                    self._completed_tasks[task_id_to_remove] = task
                else:
                    self._failed_tasks[task_id_to_remove] = task
                    self._total_failed_tasks += 1

                self.task_completed.emit(data_key, lod_level, success)

    def set_dependency_check_callback(self, callback: Callable[[GenerationTask], bool]):
        """
        Setzt Callback für Dependency-Checking
        Parameter: callback - Funktion die Task prüft und bool zurückgibt
        """
        self._check_dependencies_callback = callback

    def clear_queue(self):
        """
        Leert die komplette Queue
        """
        with self._queue_lock:
            # Queue leeren
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except queue.Empty:
                    break

            # Aktive Tasks abbrechen
            for task_id in list(self._active_tasks.keys()):
                task = self._active_tasks[task_id]
                self._failed_tasks[task_id] = task
                del self._active_tasks[task_id]

            self._current_task_count = 0
            self.logger.info("Queue cleared")

    def get_queue_statistics(self) -> Dict[str, Any]:
        """
        Gibt Queue-Statistiken zurück
        Return: Dictionary mit Queue-Statistiken
        """
        with self._queue_lock:
            return {
                "queued_tasks": self._task_queue.qsize(),
                "active_tasks": len(self._active_tasks),
                "completed_tasks": len(self._completed_tasks),
                "failed_tasks": len(self._failed_tasks),
                "current_task_count": self._current_task_count,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "total_tasks_processed": self._total_tasks_processed,
                "total_failed_tasks": self._total_failed_tasks,
                "success_rate": (
                                        (self._total_tasks_processed - self._total_failed_tasks) / max(1,
                                                                                                       self._total_tasks_processed)
                                ) * 100
            }

class ResourceTracker(QObject):
    """
    Funktionsweise: Systematisches Resource-Management für Memory-Leak-Prevention
    Aufgabe: Verfolgt alle erstellten Ressourcen und ermöglicht systematisches Cleanup
    Features: WeakReference-Tracking, automatisches Cleanup bei GC, Resource-Type-Management

    Diese Klasse löst Punkt 7: "Memory-Management für Data-Level erweitern"
    - Data-Key-spezifische Memory-Schwellwerte
    - LRU-Cache für Resource-Cleanup
    - Memory-Pressure-Detection
    - Automatisches Resource-Cleanup
    - WeakReference-basiertes Tracking
    """

    # Signals für Resource-Management
    resource_registered = pyqtSignal(str, str)  # (resource_id, resource_type)
    resource_cleaned = pyqtSignal(str, str)  # (resource_id, resource_type)
    memory_warning = pyqtSignal(int, int)  # (current_mb, threshold_mb)
    memory_pressure = pyqtSignal(str, int)  # (data_key, memory_mb)

    def __init__(self, memory_threshold_mb: int = 500):
        super().__init__()

        # Thread-Safety
        self._lock = threading.RLock()

        # Datenstrukturen
        self._tracked_resources = {}  # {resource_id: WeakReference}
        self._resource_info = {}  # {resource_id: ResourceInfo}
        self._type_registry = defaultdict(set)  # {resource_type: {resource_ids}}

        # NEUE ERWEITERTE STRUKTUREN
        self._data_key_resources = defaultdict(set)  # {data_key: {resource_ids}}
        self._data_key_memory_thresholds = {}  # {data_key: threshold_mb}
        self._lru_cache = deque()  # LRU für Resource-Cleanup
        self._memory_pressure_history = defaultdict(list)  # {data_key: [memory_values]}

        # ID-Management
        self._next_id = 0
        self.memory_threshold_mb = memory_threshold_mb

        # State Management
        self._is_shutting_down = False

        # Timer
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._automatic_cleanup)
        self.cleanup_timer.start(30000)  # 30 seconds

        self.logger = logging.getLogger(__name__)

    def register_resource(self, resource: Any, resource_type: str,
                          cleanup_func: Optional[Callable] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          data_key: Optional[str] = None) -> str:
        """
        ERWEITERT: Registriert Ressource für Tracking mit Data-Key-Zuordnung
        Parameter: resource, resource_type, cleanup_func, metadata, data_key
        Return: resource_id für Referenzierung
        """
        if self._is_shutting_down:
            raise RuntimeError("ResourceTracker is shutting down")

        with self._lock:
            try:
                # ID generieren
                temp_resource_id = f"{resource_type}_{self._next_id}"

                # Resource-Größe schätzen
                estimated_size = self._estimate_resource_size(resource)

                # ResourceInfo erstellen
                info = ResourceInfo(
                    resource_id=temp_resource_id,
                    resource_type=resource_type,
                    creation_time=time.time(),
                    cleanup_func=cleanup_func,
                    estimated_size=estimated_size,
                    metadata=metadata or {}
                )

                # WeakReference erstellen
                weak_ref = weakref.ref(
                    resource,
                    self._make_cleanup_callback(temp_resource_id)
                )

                # Alles erfolgreich → committen
                resource_id = temp_resource_id
                self._next_id += 1

                # Atomic registration
                self._tracked_resources[resource_id] = weak_ref
                self._resource_info[resource_id] = info
                self._type_registry[resource_type].add(resource_id)

                # NEUE: Data-Key-Zuordnung
                if data_key:
                    self._data_key_resources[data_key].add(resource_id)
                    info.metadata['data_key'] = data_key

                # LRU-Cache aktualisieren
                self._lru_cache.append(resource_id)

                self.logger.debug(f"Registered resource: {resource_id} ({resource_type}, {estimated_size} bytes)")
                self.resource_registered.emit(resource_id, resource_type)

            except Exception as e:
                self.logger.error(f"Resource registration failed: {e}")
                raise

        # Memory-Check außerhalb Lock
        try:
            self._check_memory_usage()
            if data_key:
                self._check_data_key_memory_pressure(data_key)
        except Exception as e:
            self.logger.warning(f"Memory check after registration failed: {e}")

        return resource_id

    def set_data_key_memory_threshold(self, data_key: str, threshold_mb: int):
        """
        NEUE METHODE: Setzt Data-Key-spezifische Memory-Schwellwerte
        Parameter: data_key, threshold_mb
        """
        with self._lock:
            self._data_key_memory_thresholds[data_key] = threshold_mb
            self.logger.info(f"Set memory threshold for '{data_key}': {threshold_mb}MB")

    def cleanup_data_resources(self, data_key: str) -> int:
        """
        NEUE METHODE: Gezieltes Data-Key-Cleanup
        Parameter: data_key
        Return: Anzahl bereinigte Ressourcen
        """
        with self._lock:
            resource_ids = list(self._data_key_resources.get(data_key, set()))

        cleaned_count = 0
        for resource_id in resource_ids:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1

        self.logger.info(f"Cleaned {cleaned_count} resources for data key '{data_key}'")
        return cleaned_count

    def cleanup_by_type(self, generator_type: str) -> int:
        """
        NEUE METHODE: Gezieltes Cleanup aller Ressourcen eines Generators anhand
        des resource_type-Prefixes (Ressourcen werden beim Registrieren als
        f"{generator_type}_{data_key}" getypt, siehe set_terrain_data_lod etc.)
        Parameter: generator_type
        Return: Anzahl bereinigte Ressourcen
        """
        prefix = f"{generator_type}_"

        with self._lock:
            matching_types = [rtype for rtype in self._type_registry if rtype.startswith(prefix)]
            resource_ids = []
            for rtype in matching_types:
                resource_ids.extend(self._type_registry[rtype])

        cleaned_count = 0
        for resource_id in resource_ids:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1

        self.logger.info(f"Cleaned {cleaned_count} resources for generator type '{generator_type}'")
        return cleaned_count

    def get_data_key_memory_usage(self, data_key: str) -> int:
        """
        NEUE METHODE: Memory-Usage für spezifischen Data-Key
        Parameter: data_key
        Return: Memory-Usage in Bytes
        """
        with self._lock:
            total_memory = 0
            resource_ids = self._data_key_resources.get(data_key, set())

            for resource_id in resource_ids:
                if resource_id in self._resource_info:
                    # Prüfe ob Resource noch lebt
                    weak_ref = self._tracked_resources.get(resource_id)
                    if weak_ref and weak_ref() is not None:
                        total_memory += self._resource_info[resource_id].estimated_size

            return total_memory

    def _check_data_key_memory_pressure(self, data_key: str):
        """
        NEUE METHODE: Memory-Pressure-Detection für Data-Key
        Parameter: data_key
        """
        current_memory_bytes = self.get_data_key_memory_usage(data_key)
        current_memory_mb = current_memory_bytes / (1024 * 1024)

        # History aktualisieren
        self._memory_pressure_history[data_key].append(current_memory_mb)
        if len(self._memory_pressure_history[data_key]) > 10:
            self._memory_pressure_history[data_key].pop(0)

        # Threshold prüfen
        threshold = self._data_key_memory_thresholds.get(data_key, 100)  # Default 100MB
        if current_memory_mb > threshold:
            self.logger.warning(f"Memory pressure for '{data_key}': {current_memory_mb:.1f}MB > {threshold}MB")
            self.memory_pressure.emit(data_key, int(current_memory_mb))

    def cleanup_lru_resources(self, target_count: int) -> int:
        """
        NEUE METHODE: LRU-basiertes Resource-Cleanup
        Parameter: target_count - Anzahl zu bereinigender Ressourcen
        Return: Anzahl tatsächlich bereinigte Ressourcen
        """
        cleaned_count = 0

        with self._lock:
            # Oldest resources from LRU
            cleanup_candidates = list(self._lru_cache)[:target_count]

        for resource_id in cleanup_candidates:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1
                with self._lock:
                    if resource_id in self._lru_cache:
                        self._lru_cache.remove(resource_id)

        self.logger.info(f"LRU cleanup: {cleaned_count} resources cleaned")
        return cleaned_count

    def _make_cleanup_callback(self, resource_id: str):
        """
        Erstellt thread-sicheren Cleanup-Callback
        Parameter: resource_id
        Return: Callback-Funktion
        """

        def cleanup_callback(weak_ref):
            if self._is_shutting_down:
                return

            try:
                with self._lock:
                    if resource_id in self._resource_info:
                        self.logger.debug(f"Resource garbage collected: {resource_id}")
                        self._cleanup_resource_unsafe(resource_id)
            except Exception as e:
                self.logger.warning(f"WeakRef callback failed for {resource_id}: {e}")

        return cleanup_callback

    def _cleanup_resource(self, resource_id: str) -> bool:
        """
        Thread-sichere einzelne Resource-Cleanup
        Parameter: resource_id
        Return: bool - Cleanup erfolgreich
        """
        with self._lock:
            return self._cleanup_resource_unsafe(resource_id)

    def _cleanup_resource_unsafe(self, resource_id: str) -> bool:
        """
        Interne Cleanup-Methode (Lock muss bereits gehalten werden)
        Parameter: resource_id
        Return: bool - Cleanup erfolgreich
        """
        if resource_id not in self._resource_info:
            return False

        info = self._resource_info[resource_id]

        # Cleanup-Funktion ausführen
        if info.cleanup_func:
            try:
                info.cleanup_func()
            except Exception as e:
                self.logger.warning(f"Cleanup function failed for {resource_id}: {e}")

        # Aus allen Registries entfernen
        self._tracked_resources.pop(resource_id, None)
        self._resource_info.pop(resource_id, None)
        self._type_registry[info.resource_type].discard(resource_id)

        # Data-Key-Zuordnung entfernen
        data_key = info.metadata.get('data_key')
        if data_key:
            self._data_key_resources[data_key].discard(resource_id)

        self.logger.debug(f"Cleaned resource: {resource_id}")
        self.resource_cleaned.emit(resource_id, info.resource_type)

        return True

    def _automatic_cleanup(self):
        """
        Automatische Cleanup-Routine
        """
        if self._is_shutting_down:
            return

        try:
            # Dead references cleanup
            with self._lock:
                resource_items = list(self._tracked_resources.items())

            dead_refs = []
            for resource_id, weak_ref in resource_items:
                try:
                    if weak_ref() is None:
                        dead_refs.append(resource_id)
                except Exception:
                    dead_refs.append(resource_id)

            for resource_id in dead_refs:
                self._cleanup_resource(resource_id)

            # Memory-Check
            self._check_memory_usage()

            # Memory-Pressure-basiertes Cleanup
            total_memory = sum(self.get_memory_usage().values())
            if total_memory > self.memory_threshold_mb:
                # LRU-Cleanup der ältesten 10% Resources
                with self._lock:
                    total_resources = len(self._tracked_resources)
                cleanup_count = max(1, total_resources // 10)
                self.cleanup_lru_resources(cleanup_count)

        except Exception as e:
            self.logger.error(f"Automatic cleanup failed: {e}")

    def _check_memory_usage(self):
        """
        Memory-Usage-Prüfung
        """
        try:
            memory_usage = self.get_memory_usage()
            total_memory_bytes = sum(memory_usage.values())
            total_memory_mb = total_memory_bytes / (1024 * 1024)

            if total_memory_mb > self.memory_threshold_mb:
                self.logger.warning(f"Memory usage: {total_memory_mb:.1f}MB > {self.memory_threshold_mb}MB")
                self.memory_warning.emit(int(total_memory_mb), self.memory_threshold_mb)

        except Exception as e:
            self.logger.warning(f"Memory usage check failed: {e}")

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Thread-sichere Memory-Usage-Berechnung
        Return: {resource_type: memory_bytes}
        """
        with self._lock:
            usage = defaultdict(int)
            resource_info_items = list(self._resource_info.items())

        for resource_id, info in resource_info_items:
            weak_ref = None
            with self._lock:
                weak_ref = self._tracked_resources.get(resource_id)

            if weak_ref and weak_ref() is not None:
                usage[info.resource_type] += info.estimated_size

        return dict(usage)

    def get_resource_statistics(self) -> Dict[str, Any]:
        """
        Umfassende Resource-Statistiken
        Return: Dictionary mit Statistiken
        """
        with self._lock:
            total_resources = len(self._tracked_resources)

            alive_count = 0
            for weak_ref in self._tracked_resources.values():
                try:
                    if weak_ref() is not None:
                        alive_count += 1
                except Exception:
                    pass

            dead_count = total_resources - alive_count
            memory_usage = self.get_memory_usage()
            total_memory_mb = sum(memory_usage.values()) / (1024 * 1024)

            # Data-Key-spezifische Statistiken
            data_key_stats = {}
            for data_key in self._data_key_resources:
                data_key_memory = self.get_data_key_memory_usage(data_key) / (1024 * 1024)
                data_key_stats[data_key] = {
                    'memory_mb': data_key_memory,
                    'resource_count': len(self._data_key_resources[data_key]),
                    'threshold_mb': self._data_key_memory_thresholds.get(data_key, 100)
                }

        return {
            'total_resources': total_resources,
            'alive_resources': alive_count,
            'dead_references': dead_count,
            'total_memory_mb': total_memory_mb,
            'memory_threshold_mb': self.memory_threshold_mb,
            'data_key_statistics': data_key_stats,
            'lru_cache_size': len(self._lru_cache),
            'is_shutting_down': self._is_shutting_down
        }

    def _estimate_resource_size(self, resource: Any) -> int:
        """
        Sichere Resource-Größenschätzung
        Parameter: resource
        Return: Geschätzte Größe in Bytes
        """
        try:
            if hasattr(resource, 'nbytes'):
                return resource.nbytes
            elif hasattr(resource, '__sizeof__'):
                base_size = resource.__sizeof__()
                # Für Container: begrenzte Sampling-Schätzung
                if isinstance(resource, (list, tuple)) and len(resource) > 100:
                    sample_size = sum(
                        self._estimate_resource_size(item)
                        for item in resource[:100]
                    )
                    estimated_total = (sample_size * len(resource)) // 100
                    return base_size + estimated_total
                return base_size
            else:
                return 1024  # 1KB conservative estimate
        except Exception:
            return 1024

    def cleanup_resources(self):
        """
        Vollständiges Cleanup für Shutdown
        """
        self.logger.info("Starting ResourceTracker shutdown")

        with self._lock:
            self._is_shutting_down = True

        if hasattr(self, 'cleanup_timer') and self.cleanup_timer.isActive():
            self.cleanup_timer.stop()

        # Alle Resources cleanup
        with self._lock:
            resource_ids = list(self._tracked_resources.keys())

        for resource_id in resource_ids:
            self._cleanup_resource(resource_id)

        with self._lock:
            self._tracked_resources.clear()
            self._resource_info.clear()
            self._type_registry.clear()
            self._data_key_resources.clear()
            self._lru_cache.clear()

        self.logger.info("ResourceTracker shutdown completed")


# =============================================================================
# DISPLAY UPDATE MANAGER - PERFORMANCE OPTIMIERUNG
# =============================================================================

class DisplayUpdateManager(QObject):
    """
    Funktionsweise: Change-Detection für Display-Updates - INTEGRIERT
    Aufgabe: Prüft ob Display-Update wirklich nötig ist basierend auf Data-Hash und Display-Mode
    Features: Multi-Level-Hashing, Pending-Updates-Management, Performance-Optimierung

    Diese Klasse optimiert Display-Performance durch intelligente Change-Detection
    - Hash-basierte Change-Detection für Arrays
    - Sample-basiertes Hashing für große Datasets
    - Pending-Updates-Management
    - Performance-Metriken und Caching
    """

    # Signals für Display-Management
    update_skipped = pyqtSignal(str, str)  # (display_id, reason)
    update_performed = pyqtSignal(str, str, float)  # (display_id, layer_type, hash_time)

    def __init__(self):
        super().__init__()

        self.last_display_hashes = {}  # {display_id: data_hash}
        self.last_update_times = {}  # {display_id: timestamp}
        self.pending_updates = set()  # Set von display_ids
        self.hash_cache = {}  # {data_hash: calculation_time}

        self.logger = logging.getLogger(__name__)

    def needs_update(self, display_id: str, data: Any, layer_type: str,
                     display_mode: str = "default", force_update: bool = False) -> bool:
        """
        Funktionsweise: Prüft ob Display-Update wirklich nötig ist
        Parameter: display_id, data, layer_type, display_mode, force_update
        Return: bool - Update nötig
        """
        if force_update:
            return True

        # Pending-Update-Check
        if display_id in self.pending_updates:
            return False

        # Hash-basierte Change-Detection
        start_time = time.time()
        current_hash = self._calculate_hash(data, layer_type, display_mode)
        hash_time = time.time() - start_time

        last_hash = self.last_display_hashes.get(display_id)
        needs_update = current_hash != last_hash

        if needs_update:
            self.logger.debug(f"Display update needed for {display_id} (hash changed)")
            self.update_performed.emit(display_id, layer_type, hash_time)
        else:
            self.logger.debug(f"Display update skipped for {display_id} (hash unchanged)")
            self.update_skipped.emit(display_id, "hash_unchanged")

        return needs_update

    def mark_updated(self, display_id: str, data: Any, layer_type: str,
                     display_mode: str = "default"):
        """
        Funktionsweise: Markiert Display als updated
        Parameter: display_id, data, layer_type, display_mode
        """
        current_hash = self._calculate_hash(data, layer_type, display_mode)
        self.last_display_hashes[display_id] = current_hash
        self.last_update_times[display_id] = time.time()
        self.pending_updates.discard(display_id)

    def mark_pending(self, display_id: str):
        """
        Funktionsweise: Markiert Display als pending update
        Parameter: display_id
        """
        self.pending_updates.add(display_id)

    def clear_display_cache(self, display_id: str):
        """
        Funktionsweise: Löscht Cache für spezifisches Display
        Parameter: display_id
        """
        self.last_display_hashes.pop(display_id, None)
        self.last_update_times.pop(display_id, None)
        self.pending_updates.discard(display_id)

    def clear_all_cache(self):
        """
        Funktionsweise: Löscht kompletten Display-Cache
        """
        self.last_display_hashes.clear()
        self.last_update_times.clear()
        self.pending_updates.clear()
        self.hash_cache.clear()

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt Cache-Statistiken für Performance-Monitoring
        Return: dict mit Cache-Stats
        """
        current_time = time.time()

        return {
            'cached_displays': len(self.last_display_hashes),
            'pending_updates': len(self.pending_updates),
            'hash_cache_size': len(self.hash_cache),
            'average_hash_time': (
                sum(self.hash_cache.values()) / len(self.hash_cache)
                if self.hash_cache else 0
            ),
            'oldest_cache_entry': (
                current_time - min(self.last_update_times.values())
                if self.last_update_times else 0
            )
        }

    def _calculate_hash(self, data: Any, layer_type: str, display_mode: str) -> str:
        """
        Funktionsweise: Berechnet Hash für Change-Detection mit Caching
        Parameter: data, layer_type, display_mode
        Return: Hash-String
        """
        # Hash-Input zusammenstellen
        hash_input = f"{layer_type}_{display_mode}"

        if data is not None:
            data_hash = self._hash_data(data)
            hash_input += f"_{data_hash}"

        # Hash berechnen
        full_hash = hashlib.md5(hash_input.encode()).hexdigest()

        return full_hash

    def _hash_data(self, data: Any) -> str:
        """
        Funktionsweise: Erstellt Hash für verschiedene Datentypen
        Parameter: data
        Return: Data-Hash-String
        """
        try:
            if isinstance(data, np.ndarray):
                # Für große Arrays: Sample-based Hashing für Performance
                if data.size > 1_000_000:  # > 1M elements
                    # Sample every nth element für große Arrays
                    step = max(1, data.size // 10000)  # Max 10k samples
                    sample = data.flat[::step]
                    return hashlib.md5(sample.tobytes()).hexdigest()
                else:
                    return hashlib.md5(data.tobytes()).hexdigest()

            elif hasattr(data, 'data') and hasattr(data.data, 'tobytes'):
                return hashlib.md5(data.data.tobytes()).hexdigest()

            elif hasattr(data, 'tobytes'):
                return hashlib.md5(data.tobytes()).hexdigest()

            else:
                return hashlib.md5(str(data).encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Data hashing failed: {e}")
            return hashlib.md5(str(id(data)).encode()).hexdigest()  # Fallback to object id


# =============================================================================
# LOD COMMUNICATION HUB
# =============================================================================

class LODCommunicationHub(QObject):
    """
    Funktionsweise: Zentraler Communication-Hub für LOD-Status zwischen Data-Keys und Tabs - ERWEITERT
    Aufgabe: Signal-basierte Koordination, Status-Cache, Dependency-Checking für zweischaliges System
    Implementation: Data-Level und Tab-Level mit gemeinsamen Methoden

    Zweischaliges System:
    - Data-Level: Einzelne Datenkomponenten (heightmap, slopemap, etc.)
    - Tab-Level: Komplette Tab-Gruppen (terrain, geology, etc.) - werden automatisch aktiviert wenn alle Data verfügbar

    ERWEITERT:
    - Punkt 2: Signal-Verbindungen für Data-Level erweitern
    - Punkt 4: Error-Handling zwischen Data/Tab-Level harmonisieren
    - Punkt 5: Data-Key-spezifische Cache-Invalidierung
    - Punkt 6: Status-Management verfeinern
    - Punkt 8: LOD-Dependency-Matrix erweitern
    """

    # === SIGNALS FÜR DATA-LEVEL ===
    data_status_updated = pyqtSignal(str, dict)  # (data_key, complete_status_dict)
    data_dependencies_satisfied = pyqtSignal(str, int)  # (data_key, lod_level)
    data_generation_ready = pyqtSignal(str, int)  # (data_key, lod_level) für Auto-Data-Generation

    # === SIGNALS FÜR TAB-LEVEL ===
    tab_status_updated = pyqtSignal(str, dict)  # (tab, complete_status_dict)
    tab_dependencies_satisfied = pyqtSignal(str, int)  # (tab, lod_level)
    tab_generation_ready = pyqtSignal(str, int)  # (tab, lod_level) für Auto-Tab-Generation

    # === GLOBALE OVERVIEW SIGNALS ===
    all_statuses = pyqtSignal(dict)  # Complete overview (data + tabs)
    all_data_status = pyqtSignal(dict)  # Nur Data-Keys
    all_tabs_status = pyqtSignal(dict)  # Nur Tabs

    # === NEUE SIGNALS FÜR ERWEITERTE FUNKTIONALITÄT ===
    dependency_registered = pyqtSignal(str, str, list)  # (item_type, item_name, dependencies)
    dependency_failed = pyqtSignal(str, str, str)  # (item_type, item_name, failed_dependency)
    generation_timeout = pyqtSignal(str, int, float)  # (data_key, lod_level, timeout_seconds)
    cache_invalidated = pyqtSignal(str, int)  # (data_key, lod_level)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # === DATENSTRUKTUREN FÜR ZWEISCHALIGES SYSTEM ===
        # Data-Level Status
        self.datakey_statuses: Dict[str, DataStatus] = {}
        # Tab-Level Status
        self.tab_statuses: Dict[str, TabStatus] = {}

        # LOD-Konfiguration
        self.lod_config: Optional[LODConfig] = None

        # === NEUE ERWEITERTE DATENSTRUKTUREN - PUNKT 8 ===
        # Dynamische Dependency-Registration
        self._dynamic_data_dependencies: Dict[str, List[str]] = {}
        self._dynamic_tab_dependencies: Dict[str, List[str]] = {}

        # Cross-LOD Dependencies (Data aus LOD N-1 für LOD N)
        self._cross_lod_dependencies: Dict[str, Dict[int, List[str]]] = {}

        # Conditional Dependencies (Parameter-abhängig)
        self._conditional_dependencies: Dict[str, List[Callable]] = {}

        # Error-Propagation-Configuration - PUNKT 4
        self._critical_data_keys: Set[str] = {
            "heightmap", "heightmap_combined"  # Diese Data-Keys sind kritisch für Tabs
        }

        # Cache-Invalidation-Tracking - PUNKT 5
        self._data_cache_timestamps: Dict[str, Dict[int, float]] = {}  # {data_key: {lod: timestamp}}

        # Thread-Safety für erweiterte Features
        self._dependency_lock = threading.RLock()

        # === INITIALISIERUNG DATA-LEVEL ===
        # Initialisiere alle DataKeys mit Idle-Status
        for datakey in DATA_DEPENDENCY_MATRIX.keys():
            self.datakey_statuses[datakey] = DataStatus(
                data=datakey, lod_level=0, lod_size=0, lod_status="idle"
            )

        # === INITIALISIERUNG TAB-LEVEL ===
        # Initialisiere alle bekannten Tabs mit Idle-Status
        for tab in TAB_DEPENDENCY_MATRIX.keys():
            self.tab_statuses[tab] = TabStatus(
                tab=tab, lod_level=0, lod_size=0, lod_status="idle"
            )

    def set_lod_config(self, minimal_map_size: int, target_map_size: int):
        """Setzt LOD-Konfiguration für Data und Tabs"""
        self.lod_config = LODConfig(minimal_map_size, target_map_size)
        self.logger.info(f"LOD config set: {minimal_map_size} → {target_map_size}")

        # === DATA-STATUS AKTUALISIEREN ===
        for datakey_name in self.datakey_statuses:
            status = self.datakey_statuses[datakey_name]
            if status.lod_level > 0:
                status.lod_size = self.lod_config.get_map_size_for_lod(status.lod_level)

        # === TAB-STATUS AKTUALISIEREN ===
        for tab_name in self.tab_statuses:
            status = self.tab_statuses[tab_name]
            if status.lod_level > 0:
                status.lod_size = self.lod_config.get_map_size_for_lod(status.lod_level)

        # === GLOBALE STATUS-UPDATES ===
        self._emit_all_status_updates()

    # =============================================================================
    # ERWEITERTE DEPENDENCY-MANAGEMENT METHODEN - PUNKT 8
    # =============================================================================

    def register_dynamic_dependency(self, item_name: str, item_type: str, dependencies: List[str]):
        """
        NEUE METHODE - PUNKT 8: Registriert dynamische Dependencies zur Laufzeit
        Parameter: item_name, item_type ("data" oder "tab"), dependencies
        """
        with self._dependency_lock:
            if item_type == "data":
                self._dynamic_data_dependencies[item_name] = dependencies
            elif item_type == "tab":
                self._dynamic_tab_dependencies[item_name] = dependencies

            self.logger.info(f"Registered dynamic {item_type} dependencies for '{item_name}': {dependencies}")
            self.dependency_registered.emit(item_type, item_name, dependencies)

    def register_cross_lod_dependency(self, data_key: str, lod_level: int, dependencies: List[str]):
        """
        NEUE METHODE - PUNKT 8: Registriert Cross-LOD-Dependencies (Data aus niedrigerem LOD)
        Parameter: data_key, lod_level, dependencies (aus LOD N-1)
        """
        with self._dependency_lock:
            if data_key not in self._cross_lod_dependencies:
                self._cross_lod_dependencies[data_key] = {}

            self._cross_lod_dependencies[data_key][lod_level] = dependencies

            self.logger.info(f"Registered cross-LOD dependency for '{data_key}' LOD {lod_level}: {dependencies}")

    def register_conditional_dependency(self, item_name: str, condition_func: Callable[[Dict[str, Any]], List[str]]):
        """
        NEUE METHODE - PUNKT 8: Registriert conditional Dependencies basierend auf Parametern
        Parameter: item_name, condition_func (Parameter → Dependencies)
        """
        with self._dependency_lock:
            if item_name not in self._conditional_dependencies:
                self._conditional_dependencies[item_name] = []

            self._conditional_dependencies[item_name].append(condition_func)

            self.logger.info(f"Registered conditional dependency for '{item_name}'")

    def get_effective_dependencies(self, item_name: str, item_type: str,
                                   lod_level: int = 1, parameters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        NEUE METHODE - PUNKT 8: Berechnet effektive Dependencies unter Berücksichtigung aller Arten
        Parameter: item_name, item_type, lod_level, parameters
        Return: Kombinierte Liste aller Dependencies
        """
        with self._dependency_lock:
            dependencies = []

            # 1. Standard-Dependencies
            if item_type == "data":
                dependencies.extend(DATA_DEPENDENCY_MATRIX.get(item_name, []))
                dependencies.extend(self._dynamic_data_dependencies.get(item_name, []))
            else:  # tab
                dependencies.extend(TAB_DEPENDENCY_MATRIX.get(item_name, []))
                dependencies.extend(self._dynamic_tab_dependencies.get(item_name, []))

            # 2. Cross-LOD-Dependencies
            if item_type == "data" and item_name in self._cross_lod_dependencies:
                cross_deps = self._cross_lod_dependencies[item_name].get(lod_level, [])
                dependencies.extend(cross_deps)

            # 3. Conditional Dependencies
            if item_name in self._conditional_dependencies and parameters:
                for condition_func in self._conditional_dependencies[item_name]:
                    try:
                        conditional_deps = condition_func(parameters)
                        dependencies.extend(conditional_deps)
                    except Exception as e:
                        self.logger.warning(f"Conditional dependency evaluation failed for '{item_name}': {e}")

            # Duplikate entfernen
            return list(set(dependencies))

    def validate_dependency_graph(self, item_type: str) -> Tuple[bool, List[str]]:
        """
        NEUE METHODE - PUNKT 8: Validiert Dependency-Graph auf Circular-Dependencies
        Parameter: item_type ("data" oder "tab")
        Return: (valid: bool, circular_dependencies: List[str])
        """
        dependency_matrix = (DATA_DEPENDENCY_MATRIX if item_type == "data"
                             else TAB_DEPENDENCY_MATRIX)

        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            dependencies = self.get_effective_dependencies(node, item_type)
            for dep in dependencies:
                if dep not in dependency_matrix:
                    continue

                if dep not in visited:
                    result = has_cycle(dep, visited, rec_stack, path)
                    if result:
                        return result
                elif dep in rec_stack:
                    # Circular dependency gefunden
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    return cycle

            rec_stack.remove(node)
            path.pop()
            return False

        visited = set()
        circular_deps = []

        for item in dependency_matrix:
            if item not in visited:
                result = has_cycle(item, visited, set(), [])
                if result and isinstance(result, list):
                    circular_deps.extend(result)

        is_valid = len(circular_deps) == 0
        return is_valid, circular_deps

    # =============================================================================
    # ERWEITERTE ERROR-HANDLING UND PROPAGATION
    # =============================================================================

    def set_critical_data_keys(self, critical_keys: Set[str]):
        """
        Setzt kritische Data-Keys die Tab-Failure verursachen
        Parameter: critical_keys - Set von Data-Key-Namen
        """
        self._critical_data_keys = critical_keys
        self.logger.info(f"Set critical data keys: {critical_keys}")

    def _propagate_data_error_to_tab(self, data_key: str, lod_level: int, error_message: str):
        """
        Propagiert Data-Key-Fehler zu übergeordnetem Tab
        Parameter: data_key, lod_level, error_message
        """
        # Finde übergeordneten Tab
        parent_tab = DATA_KEY_TO_TAB_MAPPING.get(data_key)
        if not parent_tab or parent_tab not in self.tab_statuses:
            return

        tab_status = self.tab_statuses[parent_tab]

        # Prüfe ob Data-Key kritisch ist
        if data_key in self._critical_data_keys:
            # Kritischer Fehler → Tab als fehlgeschlagen markieren
            tab_status.lod_status = "failure"
            tab_status.error_message = f"Critical dependency '{data_key}' failed: {error_message}"
            tab_status.progress_percent = 0

            self.logger.error(f"Tab '{parent_tab}' failed due to critical data key '{data_key}': {error_message}")
            self._emit_status_update(parent_tab, "tab")
        else:
            # Nicht-kritischer Fehler → Partial-Failure-Handling
            if tab_status.lod_status == "pending":
                # Tab läuft noch → Error nur loggen
                self.logger.warning(f"Non-critical data key '{data_key}' failed in tab '{parent_tab}': {error_message}")

            # Error-Aggregation für Tab-Level-Display
            if not tab_status.error_message:
                tab_status.error_message = f"Partial failures: {data_key}"
            else:
                tab_status.error_message += f", {data_key}"

    def _handle_data_recovery(self, data_key: str, lod_level: int, max_retries: int = 3):
        """
        Implementiert Recovery-Mechanismen für fehlgeschlagene Data-Keys
        Parameter: data_key, lod_level, max_retries
        """
        if data_key not in self.datakey_statuses:
            return

        status = self.datakey_statuses[data_key]

        if status.retry_count < max_retries:
            status.retry_count += 1
            status.lod_status = "idle"  # Reset für Retry
            status.generation_mode = GenerationMode.RECOVERY.value

            self.logger.info(
                f"Scheduling recovery for '{data_key}' LOD {lod_level} (attempt {status.retry_count}/{max_retries})")

            # Recovery-Generation mit niedriger Priorität
            self.data_generation_ready.emit(data_key, lod_level)
        else:
            self.logger.error(f"Recovery failed for '{data_key}' LOD {lod_level} after {max_retries} attempts")

            # Permanenter Failure → Cleanup dependent items
            self._cleanup_dependent_items_on_failure(data_key, lod_level)

    def _cleanup_dependent_items_on_failure(self, failed_data_key: str, lod_level: int):
        """
        Cleanup abhängiger Items bei permanentem Failure
        Parameter: failed_data_key, lod_level
        """
        # Finde alle abhängigen Data-Keys
        dependent_keys = []
        for data_key, deps in DATA_DEPENDENCY_MATRIX.items():
            if failed_data_key in deps:
                dependent_keys.append(data_key)

        # Dynamische Dependencies prüfen
        for data_key, deps in self._dynamic_data_dependencies.items():
            if failed_data_key in deps:
                dependent_keys.append(data_key)

        # Markiere abhängige Data-Keys als "blocked"
        for dep_key in dependent_keys:
            if dep_key in self.datakey_statuses:
                dep_status = self.datakey_statuses[dep_key]
                if dep_status.lod_level <= lod_level:
                    dep_status.lod_status = "failure"
                    dep_status.error_message = f"Dependency '{failed_data_key}' permanently failed"

                    self.logger.warning(f"Marking '{dep_key}' as failed due to dependency failure")
                    self._emit_status_update(dep_key, "data")

    # =============================================================================
    # CACHE-INVALIDIERUNG
    # =============================================================================

    def invalidate_data_cache(self, data_key: str, lod_level: Optional[int] = None,
                              cascade: bool = True):
        """
        Data-Key-spezifische Cache-Invalidierung
        Parameter: data_key, lod_level (None = alle LODs), cascade (Dependencies invalidieren)
        """
        # Status auf "idle" setzen
        if data_key in self.datakey_statuses:
            status = self.datakey_statuses[data_key]
            if lod_level is None or status.lod_level == lod_level:
                status.lod_status = "idle"
                status.progress_percent = 0
                status.error_message = ""
                status.available_data_keys.clear()
                status.parameter_hash = ""  # PUNKT 6: Parameter-Hash zurücksetzen

                self._emit_status_update(data_key, "data")

        # Cache-Timestamps löschen
        if data_key in self._data_cache_timestamps:
            if lod_level is None:
                del self._data_cache_timestamps[data_key]
            else:
                self._data_cache_timestamps[data_key].pop(lod_level, None)

        self.logger.info(f"Invalidated cache for '{data_key}'" +
                         (f" LOD {lod_level}" if lod_level else " (all LODs)"))
        self.cache_invalidated.emit(data_key, lod_level or -1)

        # Cascade-Invalidation für abhängige Data-Keys
        if cascade:
            self._cascade_invalidate_dependent_data(data_key, lod_level)

    def _cascade_invalidate_dependent_data(self, invalidated_key: str, lod_level: Optional[int]):
        """
        Kaskadiert Cache-Invalidierung auf abhängige Data-Keys
        Parameter: invalidated_key, lod_level
        """
        dependent_keys = []

        # Finde abhängige Data-Keys aus Standard-Matrix
        for data_key, deps in DATA_DEPENDENCY_MATRIX.items():
            if invalidated_key in deps:
                dependent_keys.append(data_key)

        # Dynamische Dependencies prüfen
        for data_key, deps in self._dynamic_data_dependencies.items():
            if invalidated_key in deps:
                dependent_keys.append(data_key)

        # Abhängige Keys invalidieren (ohne weitere Kaskadierung)
        for dep_key in dependent_keys:
            if dep_key != invalidated_key:  # Infinite Loop verhindern
                self.invalidate_data_cache(dep_key, lod_level, cascade=False)
                self.logger.debug(f"Cascade invalidated '{dep_key}' due to '{invalidated_key}'")

    # =============================================================================
    # DATA-LEVEL EVENT HANDLERS - PUNKT 2 & 6 LÖSUNG
    # =============================================================================

    @pyqtSlot(str, int, int)
    def on_data_lod_started(self, data_key: str, lod_level: int, lod_size: int):
        """
        Slot: Data-Key hat LOD-Generation gestartet - PUNKT 2 & 6 ERWEITERT
        Implementiert verfeinerte Status-Management und Signal-Verbindungen
        """
        if data_key not in self.datakey_statuses:
            self.datakey_statuses[data_key] = DataStatus(data=data_key, lod_level=0, lod_size=0, lod_status="idle")

        status = self.datakey_statuses[data_key]

        # PUNKT 6: Verfeinerte Status-Transitions
        if status.lod_status not in ["idle", "failure"]:
            self.logger.warning(f"Invalid status transition for '{data_key}': {status.lod_status} → pending")

        # Status aktualisieren
        status.lod_level = lod_level
        status.lod_size = lod_size
        status.lod_status = "pending"
        status.progress_percent = 0
        status.error_message = ""
        status.last_generation_time = time.time()
        status.generation_mode = GenerationMode.AUTO.value if data_key in AUTO_GENERATION_DATA_KEYS else GenerationMode.MANUAL.value

        # PUNKT 6: Dependency-Status aktualisieren
        dependencies = self.get_effective_dependencies(data_key, "data", lod_level)
        status.dependency_status.clear()
        for dep in dependencies:
            if dep in self.datakey_statuses:
                dep_status = self.datakey_statuses[dep]
                status.dependency_status[dep] = dep_status.lod_status
            else:
                status.dependency_status[dep] = "unknown"

        self.logger.debug(
            f"Data '{data_key}' LOD {lod_level} started (size: {lod_size}, mode: {status.generation_mode})")
        self._emit_status_update(data_key, "data")

    @pyqtSlot(str, int, int)
    def on_data_lod_progress(self, data_key: str, lod_level: int, progress_percent: int):
        """
        Slot: Data-Key LOD-Generation Progress Update - PUNKT 6 ERWEITERT
        """
        if data_key in self.datakey_statuses:
            status = self.datakey_statuses[data_key]
            if status.lod_level == lod_level and status.lod_status == "pending":
                # PUNKT 6: Progress-Validation
                progress_percent = max(0, min(100, progress_percent))
                status.progress_percent = progress_percent
                self._emit_status_update(data_key, "data")

    @pyqtSlot(str, int, bool)
    def on_data_lod_completed(self, data_key: str, lod_level: int, success: bool):
        """
        Slot: Data-Key LOD-Generation abgeschlossen - PUNKT 2, 4, 6 ERWEITERT
        """
        if data_key not in self.datakey_statuses:
            return

        status = self.datakey_statuses[data_key]

        # PUNKT 6: Status-Validation
        if status.lod_status != "pending":
            self.logger.warning(f"Unexpected completion for '{data_key}' with status '{status.lod_status}'")

        status.lod_level = lod_level
        status.lod_status = "success" if success else "failure"
        status.progress_percent = 100 if success else 0
        status.generation_duration = time.time() - status.last_generation_time

        if success:
            status.available_data_keys = [data_key]
            status.error_message = ""
            status.retry_count = 0  # Reset bei Erfolg

            # PUNKT 5: Cache-Timestamp aktualisieren
            if data_key not in self._data_cache_timestamps:
                self._data_cache_timestamps[data_key] = {}
            self._data_cache_timestamps[data_key][lod_level] = time.time()
        else:
            # PUNKT 4: Fehler-Handling mit Recovery
            self._handle_data_recovery(data_key, lod_level)

        self.logger.debug(
            f"Data '{data_key}' LOD {lod_level} completed: {success} (duration: {status.generation_duration:.2f}s)")
        self._emit_status_update(data_key, "data")

        if success:
            # PUNKT 2: Erweiterte Dependency-Checking
            self._check_dependent_items(data_key, lod_level, "data")
            # Tab-Completion-Checking
            self._check_tab_completion_from_data(data_key, lod_level)

    @pyqtSlot(str, int, str)
    def on_data_lod_failed(self, data_key: str, lod_level: int, error_message: str):
        """
        Slot: Data-Key LOD-Generation fehlgeschlagen - PUNKT 4 & 6 ERWEITERT
        """
        if data_key in self.datakey_statuses:
            status = self.datakey_statuses[data_key]
            status.lod_level = lod_level
            status.lod_status = "failure"
            status.progress_percent = 0
            status.error_message = error_message
            status.generation_duration = time.time() - status.last_generation_time

            self.logger.error(f"Data '{data_key}' LOD {lod_level} failed: {error_message}")
            self._emit_status_update(data_key, "data")

            # PUNKT 4: Error-Propagation
            self._propagate_data_error_to_tab(data_key, lod_level, error_message)

            # Dependent items benachrichtigen
            self.dependency_failed.emit("data", data_key, error_message)

    # =============================================================================
    # TAB-LEVEL EVENT HANDLERS - BESTEHEND ABER ERWEITERT
    # =============================================================================

    @pyqtSlot(str, int, int)
    def on_tab_lod_started(self, tab: str, lod_level: int, lod_size: int):
        """Slot: Tab hat LOD-Generation gestartet - PUNKT 6 ERWEITERT"""
        if tab not in self.tab_statuses:
            self.tab_statuses[tab] = TabStatus(tab=tab, lod_level=0, lod_size=0, lod_status="idle")

        status = self.tab_statuses[tab]

        # PUNKT 6: Status-Transition-Validation
        if status.lod_status not in ["idle", "failure"]:
            self.logger.warning(f"Invalid tab status transition for '{tab}': {status.lod_status} → pending")

        status.lod_level = lod_level
        status.lod_size = lod_size
        status.lod_status = "pending"
        status.progress_percent = 0
        status.error_message = ""

        self.logger.debug(f"Tab '{tab}' LOD {lod_level} started (size: {lod_size})")
        self._emit_status_update(tab, "tab")

    @pyqtSlot(str, int, int)
    def on_tab_lod_progress(self, tab: str, lod_level: int, progress_percent: int):
        """Slot: Tab LOD-Generation Progress Update - PUNKT 6 ERWEITERT"""
        if tab in self.tab_statuses:
            status = self.tab_statuses[tab]
            if status.lod_level == lod_level and status.lod_status == "pending":
                # PUNKT 6: Progress-Validation
                progress_percent = max(0, min(100, progress_percent))
                status.progress_percent = progress_percent
                self._emit_status_update(tab, "tab")

    @pyqtSlot(str, int, bool, list)
    def on_tab_lod_completed(self, tab: str, lod_level: int, success: bool, data_keys: list):
        """Slot: Tab LOD-Generation abgeschlossen - PUNKT 6 ERWEITERT"""
        if tab not in self.tab_statuses:
            return

        status = self.tab_statuses[tab]

        # PUNKT 6: Status-Validation
        if status.lod_status != "pending":
            self.logger.warning(f"Unexpected tab completion for '{tab}' with status '{status.lod_status}'")

        status.lod_level = lod_level
        status.lod_status = "success" if success else "failure"
        status.progress_percent = 100 if success else 0

        if success:
            status.available_data_keys = data_keys
            status.error_message = ""

        self.logger.debug(f"Tab '{tab}' LOD {lod_level} completed: {success}")
        self._emit_status_update(tab, "tab")

        if success:
            self._check_dependent_items(tab, lod_level, "tab")

    @pyqtSlot(str, int, str)
    def on_tab_lod_failed(self, tab: str, lod_level: int, error_message: str):
        """Slot: Tab LOD-Generation fehlgeschlagen - PUNKT 6 ERWEITERT"""
        if tab in self.tab_statuses:
            status = self.tab_statuses[tab]
            status.lod_level = lod_level
            status.lod_status = "failure"
            status.progress_percent = 0
            status.error_message = error_message

            self.logger.error(f"Tab '{tab}' LOD {lod_level} failed: {error_message}")
            self._emit_status_update(tab, "tab")

    # =============================================================================
    # ERWEITERTE SIGNAL-VERBINDUNGEN - PUNKT 2 LÖSUNG
    # =============================================================================

    def connect_data_to_hub(self, data_key: str, generator: DataKeyGeneratorBase):
        """
        NEUE METHODE - PUNKT 2: Verbindet Data-Key-Generator-Signale mit Hub
        Parameter: data_key, generator (DataKeyGeneratorBase instance)
        """
        try:
            # Generator → Hub Signal-Verbindungen
            generator.data_lod_started.connect(self.on_data_lod_started)
            generator.data_lod_progress.connect(self.on_data_lod_progress)
            generator.data_lod_completed.connect(self.on_data_lod_completed)
            generator.data_lod_failed.connect(self.on_data_lod_failed)

            # Hub → Generator Signal-Verbindungen (falls Generator Slots hat)
            if hasattr(generator, 'on_dependencies_satisfied'):
                self.data_dependencies_satisfied.connect(generator.on_dependencies_satisfied)
            if hasattr(generator, 'on_generation_ready'):
                self.data_generation_ready.connect(generator.on_generation_ready)

            self.logger.debug(f"Data-Key '{data_key}' connected to LOD communication hub")

        except Exception as e:
            self.logger.error(f"Failed to connect data-key '{data_key}' to LOD hub: {e}")

    def disconnect_data_from_hub(self, data_key: str, generator: DataKeyGeneratorBase):
        """
        NEUE METHODE - PUNKT 2: Trennt Data-Key-Generator-Signale vom Hub
        Parameter: data_key, generator
        """
        try:
            # Signal-Verbindungen trennen
            generator.data_lod_started.disconnect(self.on_data_lod_started)
            generator.data_lod_progress.disconnect(self.on_data_lod_progress)
            generator.data_lod_completed.disconnect(self.on_data_lod_completed)
            generator.data_lod_failed.disconnect(self.on_data_lod_failed)

            if hasattr(generator, 'on_dependencies_satisfied'):
                self.data_dependencies_satisfied.disconnect(generator.on_dependencies_satisfied)
            if hasattr(generator, 'on_generation_ready'):
                self.data_generation_ready.disconnect(generator.on_generation_ready)

            self.logger.debug(f"Data-Key '{data_key}' disconnected from LOD communication hub")

        except Exception as e:
            self.logger.error(f"Failed to disconnect data-key '{data_key}' from LOD hub: {e}")

    def connect_tab_to_hub(self, tab_name: str, tab_widget):
        """
        BESTEHENDE METHODE - ERWEITERT: Verbindet Tab-Signals mit LOD-Hub
        Parameter: tab_name, tab_widget (BaseMapTab instance)
        """
        try:
            # Tab → Hub Signal-Verbindungen
            if hasattr(tab_widget, 'lod_started'):
                tab_widget.lod_started.connect(self.on_tab_lod_started)
            if hasattr(tab_widget, 'lod_progress'):
                tab_widget.lod_progress.connect(self.on_tab_lod_progress)
            if hasattr(tab_widget, 'lod_completed'):
                tab_widget.lod_completed.connect(self.on_tab_lod_completed)
            if hasattr(tab_widget, 'lod_failed'):
                tab_widget.lod_failed.connect(self.on_tab_lod_failed)

            # Hub → Tab Signal-Verbindungen
            if hasattr(tab_widget, 'on_dependencies_satisfied'):
                self.tab_dependencies_satisfied.connect(tab_widget.on_dependencies_satisfied)
            if hasattr(tab_widget, 'on_generation_ready'):
                self.tab_generation_ready.connect(tab_widget.on_generation_ready)

            self.logger.debug(f"Tab '{tab_name}' connected to LOD communication hub")

        except Exception as e:
            self.logger.error(f"Failed to connect tab '{tab_name}' to LOD hub: {e}")

    # =============================================================================
    # GENERISCHE STATUS-UPDATE UND DEPENDENCY-CHECKING METHODEN
    # =============================================================================

    def _emit_status_update(self, item_name: str, item_type: str):
        """Emittiert Status-Update für spezifischen Data-Key oder Tab"""
        if item_type == "data" and item_name in self.datakey_statuses:
            status_dict = self._status_to_dict(self.datakey_statuses[item_name], "data")
            self.data_status_updated.emit(item_name, status_dict)
        elif item_type == "tab" and item_name in self.tab_statuses:
            status_dict = self._status_to_dict(self.tab_statuses[item_name], "tab")
            self.tab_status_updated.emit(item_name, status_dict)

        # Globale Status-Updates
        self._emit_all_status_updates()

    def _emit_all_status_updates(self):
        """Emittiert alle globalen Status-Updates"""
        self.all_data_status.emit(self._get_all_data_statuses_dict())
        self.all_tabs_status.emit(self._get_all_tab_statuses_dict())
        self.all_statuses.emit(self._get_complete_status_dict())

    def _check_dependent_items(self, completed_item: str, completed_lod: int, item_type: str):
        """
        Prüft welche Items jetzt ihre Dependencies erfüllt haben - PUNKT 8 ERWEITERT
        Verwendet erweiterte Dependency-Resolution
        """
        if item_type == "data":
            dependency_matrix = DATA_DEPENDENCY_MATRIX
            status_dict = self.datakey_statuses
            satisfied_signal = self.data_dependencies_satisfied
            ready_signal = self.data_generation_ready
            auto_items = [key for key, config in AUTO_GENERATION_DATA_KEYS.items() if config.get("auto", False)]
        else:  # tab
            dependency_matrix = TAB_DEPENDENCY_MATRIX
            status_dict = self.tab_statuses
            satisfied_signal = self.tab_dependencies_satisfied
            ready_signal = self.tab_generation_ready
            auto_items = ['geology', 'weather', 'water', 'settlement', 'biome']

        # Prüfe alle Items die den completed_item als Dependency haben
        all_items = set(dependency_matrix.keys())
        all_items.update(
            self._dynamic_data_dependencies.keys() if item_type == "data" else self._dynamic_tab_dependencies.keys())

        for item in all_items:
            effective_deps = self.get_effective_dependencies(item, item_type, completed_lod)

            if completed_item in effective_deps:
                if self.are_dependencies_satisfied(item, completed_lod, item_type):
                    self.logger.debug(f"{item_type.title()} '{item}' dependencies satisfied for LOD {completed_lod}")
                    satisfied_signal.emit(item, completed_lod)

                    if item in auto_items:
                        ready_signal.emit(item, completed_lod)

    def _check_tab_completion_from_data(self, completed_data_key: str, completed_lod: int):
        """
        Prüft ob ein Tab durch Data-Completion aktiviert werden kann - PUNKT 8 ERWEITERT
        """
        # Finde alle Tabs die diesen Data-Key benötigen
        for tab, required_data_keys in TAB_DEPENDENCY_MATRIX.items():
            if completed_data_key in required_data_keys:
                if self.check_tab_completion(tab, completed_lod):
                    # Tab ist komplett - aktualisiere Status
                    if tab in self.tab_statuses:
                        status = self.tab_statuses[tab]
                        if status.lod_level < completed_lod:
                            status.lod_level = completed_lod
                            status.lod_size = self.get_lod_size_for_level(completed_lod)
                            status.lod_status = "success"
                            status.progress_percent = 100
                            status.available_data_keys = required_data_keys.copy()

                            self.logger.info(f"Tab '{tab}' automatically completed for LOD {completed_lod}")
                            self._emit_status_update(tab, "tab")

                            # Tab-Dependencies prüfen
                            self._check_dependent_items(tab, completed_lod, "tab")

    def check_tab_completion(self, tab: str, lod_level: int) -> bool:
        """Prüft ob alle Data-Keys eines Tabs für das LOD verfügbar sind"""
        required_data_keys = TAB_DEPENDENCY_MATRIX.get(tab, [])
        for data_key in required_data_keys:
            if not self._is_data_available(data_key, lod_level):
                return False
        return True

    def _is_data_available(self, data_key: str, lod_level: int) -> bool:
        """Prüft ob ein Data-Key für LOD-Level verfügbar ist"""
        if data_key in self.datakey_statuses:
            status = self.datakey_statuses[data_key]
            return status.lod_level >= lod_level and status.lod_status == "success"
        return False

    def are_dependencies_satisfied(self, item: str, lod_level: int, item_type: str) -> bool:
        """
        Prüft ob alle Dependencies für Item auf LOD-Level erfüllt sind - PUNKT 8 ERWEITERT
        Verwendet erweiterte Dependency-Resolution
        """
        dependencies = self.get_effective_dependencies(item, item_type, lod_level)

        if item_type == "data":
            status_dict = self.datakey_statuses
        else:  # tab
            status_dict = self.tab_statuses

        for dep_item in dependencies:
            if dep_item not in status_dict:
                return False

            dep_status = status_dict[dep_item]
            if dep_status.lod_level < lod_level or dep_status.lod_status != "success":
                return False

        return True

    # =============================================================================
    # STATUS-ABFRAGE UND UTILITY METHODEN
    # =============================================================================

    def get_data_status(self, data_key: str) -> Optional[Dict[str, Any]]:
        """Gibt Status für spezifischen Data-Key zurück"""
        if data_key in self.datakey_statuses:
            return self._status_to_dict(self.datakey_statuses[data_key], "data")
        return None

    def get_tab_status(self, tab: str) -> Optional[Dict[str, Any]]:
        """Gibt Status für spezifischen Tab zurück"""
        if tab in self.tab_statuses:
            return self._status_to_dict(self.tab_statuses[tab], "tab")
        return None

    def get_lod_size_for_level(self, lod_level: int) -> int:
        """Gibt Map-Size für LOD-Level zurück"""
        if self.lod_config:
            return self.lod_config.get_map_size_for_lod(lod_level)
        return 64

    def get_max_available_lod_for_data(self, data_key: str) -> int:
        """Gibt höchste verfügbare LOD für Data-Key zurück"""
        return self._get_max_available_lod(data_key, "data")

    def get_max_available_lod_for_tab(self, tab: str) -> int:
        """Gibt höchste verfügbare LOD für Tab zurück"""
        return self._get_max_available_lod(tab, "tab")

    def _get_max_available_lod(self, item: str, item_type: str) -> int:
        """Generische Max-LOD-Berechnung mit erweiterten Dependencies"""
        dependencies = self.get_effective_dependencies(item, item_type)

        if item_type == "data":
            status_dict = self.datakey_statuses
        else:  # tab
            status_dict = self.tab_statuses

        if not dependencies:
            return self.lod_config.max_terrain_lod if self.lod_config else 7

        min_dep_lod = float('inf')
        for dep_item in dependencies:
            if dep_item in status_dict:
                dep_status = status_dict[dep_item]
                if dep_status.lod_status == "success":
                    min_dep_lod = min(min_dep_lod, dep_status.lod_level)
                else:
                    return 0
            else:
                return 0

        return int(min_dep_lod) if min_dep_lod != float('inf') else 0

    # =============================================================================
    # DICTIONARY-KONVERTIERUNG - PUNKT 6 ERWEITERT
    # =============================================================================

    def _get_all_data_statuses_dict(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle Data-Status als Dictionary zurück"""
        return {
            data_key: self._status_to_dict(status, "data")
            for data_key, status in self.datakey_statuses.items()
        }

    def _get_all_tab_statuses_dict(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle Tab-Status als Dictionary zurück"""
        return {
            tab: self._status_to_dict(status, "tab")
            for tab, status in self.tab_statuses.items()
        }

    def _get_complete_status_dict(self) -> Dict[str, Any]:
        """Gibt kompletten Status zurück"""
        return {
            "data": self._get_all_data_statuses_dict(),
            "tabs": self._get_all_tab_statuses_dict()
        }

    def _status_to_dict(self, status_obj, status_type: str) -> Dict[str, Any]:
        """
        Generische Status-zu-Dictionary Konvertierung - PUNKT 6 ERWEITERT
        Enthält alle erweiterten Felder für detaillierte Status-Information
        """
        if status_type == "data":
            base_dict = {
                "data": status_obj.data,
                "lod_level": status_obj.lod_level,
                "lod_size": status_obj.lod_size,
                "lod_status": status_obj.lod_status,
                "progress_percent": status_obj.progress_percent,
                "error_message": status_obj.error_message,
                "available_data_keys": status_obj.available_data_keys.copy()
            }
            # PUNKT 6: Erweiterte Felder hinzufügen
            base_dict.update({
                "dependency_status": status_obj.dependency_status.copy(),
                "generation_mode": status_obj.generation_mode,
                "last_generation_time": status_obj.last_generation_time,
                "generation_duration": status_obj.generation_duration,
                "parameter_hash": status_obj.parameter_hash,
                "retry_count": status_obj.retry_count,
                "max_retries": status_obj.max_retries
            })
            return base_dict
        else:  # tab
            return {
                "tab": status_obj.tab,
                "lod_level": status_obj.lod_level,
                "lod_size": status_obj.lod_size,
                "lod_status": status_obj.lod_status,
                "progress_percent": status_obj.progress_percent,
                "error_message": status_obj.error_message,
                "available_data_keys": status_obj.available_data_keys.copy()
            }

    # =============================================================================
    # EXTENDED UTILITY METHODS - PUNKT 6
    # =============================================================================

    def get_hub_statistics(self) -> Dict[str, Any]:
        """
        NEUE METHODE - PUNKT 6: Umfassende Hub-Statistiken
        Return: Dictionary mit detaillierten Hub-Statistiken
        """
        data_stats = {
            "total_data_keys": len(self.datakey_statuses),
            "idle_data_keys": len([s for s in self.datakey_statuses.values() if s.lod_status == "idle"]),
            "pending_data_keys": len([s for s in self.datakey_statuses.values() if s.lod_status == "pending"]),
            "success_data_keys": len([s for s in self.datakey_statuses.values() if s.lod_status == "success"]),
            "failed_data_keys": len([s for s in self.datakey_statuses.values() if s.lod_status == "failure"]),
        }

        tab_stats = {
            "total_tabs": len(self.tab_statuses),
            "idle_tabs": len([s for s in self.tab_statuses.values() if s.lod_status == "idle"]),
            "pending_tabs": len([s for s in self.tab_statuses.values() if s.lod_status == "pending"]),
            "success_tabs": len([s for s in self.tab_statuses.values() if s.lod_status == "success"]),
            "failed_tabs": len([s for s in self.tab_statuses.values() if s.lod_status == "failure"]),
        }

        dependency_stats = {
            "dynamic_data_dependencies": len(self._dynamic_data_dependencies),
            "dynamic_tab_dependencies": len(self._dynamic_tab_dependencies),
            "cross_lod_dependencies": len(self._cross_lod_dependencies),
            "conditional_dependencies": len(self._conditional_dependencies),
            "critical_data_keys": len(self._critical_data_keys)
        }

        return {
            "data_statistics": data_stats,
            "tab_statistics": tab_stats,
            "dependency_statistics": dependency_stats,
            "cache_timestamps": len(self._data_cache_timestamps),
            "lod_config_set": self.lod_config is not None
        }

    def reset_hub_state(self):
        """
        NEUE METHODE - PUNKT 6: Setzt Hub-Status zurück (für Testing/Reset)
        """
        self.logger.info("Resetting LOD Communication Hub state")

        # Alle Data-Status zurücksetzen
        for data_key in self.datakey_statuses:
            status = self.datakey_statuses[data_key]
            status.lod_level = 0
            status.lod_size = 0
            status.lod_status = "idle"
            status.progress_percent = 0
            status.error_message = ""
            status.available_data_keys.clear()
            status.dependency_status.clear()
            status.generation_mode = "manual"
            status.last_generation_time = 0.0
            status.generation_duration = 0.0
            status.parameter_hash = ""
            status.retry_count = 0

        # Alle Tab-Status zurücksetzen
        for tab in self.tab_statuses:
            status = self.tab_statuses[tab]
            status.lod_level = 0
            status.lod_size = 0
            status.lod_status = "idle"
            status.progress_percent = 0
            status.error_message = ""
            status.available_data_keys.clear()

        # Cache zurücksetzen
        self._data_cache_timestamps.clear()

        # Status-Updates emittieren
        self._emit_all_status_updates()

        self.logger.info("LOD Communication Hub state reset completed")


# =============================================================================
# MAIN DATA LOD MANAGER - VOLLSTÄNDIG INTEGRIERT
# =============================================================================

class DataLODManager(QObject):
    """
    Funktionsweise: Zentrale Klasse für Datenverwaltung - KOMPLETT INTEGRIERT
    Aufgabe: Speichert und verwaltet alle Arrays zwischen Tabs, LOD-basiertes Cache-Management
    Kommunikation: Signals über LODCommunicationHub, Memory-Management für große Arrays

    Diese Klasse integriert alle Komponenten und löst die verbleibenden Punkte:
    - Punkt 3: Automatische Data-Generierung implementieren (mit GeneratorQueue)
    - Punkt 7: Memory-Management für Data-Level erweitern (integriert)
    - Punkt 9: Factory und Registry-Pattern (integriert)
    - Punkt 10: Data-Key-spezifische Utils (integriert)

    INTEGRIERTE KOMPONENTEN:
    - LODCommunicationHub: Signal-basierte Koordination
    - ResourceTracker: Memory-Management
    - DisplayUpdateManager: Display-Optimierung
    - DataKeyGeneratorRegistry: Generator-Management
    - GeneratorQueue: Automatische Generierung
    """

    # Legacy-Signals für Backward-Compatibility
    data_updated = pyqtSignal(str, str)  # (generator_type, data_key)
    cache_invalidated = pyqtSignal(str)  # (generator_type)

    # Neue integrierte Signals
    lod_data_stored = pyqtSignal(str, int, list)  # (generator_type, lod_level, data_keys)
    auto_generation_triggered = pyqtSignal(str, int, str)  # (data_key, lod_level, trigger_reason)
    memory_optimization_performed = pyqtSignal(str, int, int)  # (action, resources_cleaned, memory_freed_mb)

    def __init__(self, memory_threshold_mb: int = 500):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # === INTEGRIERTE KOMPONENTEN INITIALISIERUNG ===
        self.lod_hub = LODCommunicationHub()
        self.resource_tracker = ResourceTracker(memory_threshold_mb)
        self.display_manager = DisplayUpdateManager()
        self.generator_registry = DataKeyGeneratorRegistry()
        self.generator_queue = GeneratorQueue(max_concurrent_tasks=3)

        # === SIGNAL-VERBINDUNGEN ZWISCHEN KOMPONENTEN ===
        self._setup_integrated_signals()

        # === HAUPT-DATENSTRUKTUREN FÜR ALLE GENERATOR-OUTPUTS (LOD-ERWEITERT) ===
        self._terrain_data = {}  # {f"lod_{level}_{data_key}": data}
        self._geology_data = {}
        self._settlement_data = {}
        self._weather_data = {}
        self._water_data = {}
        self._biome_data = {}

        # === FEINGRANULARER CALCULATOR-STORAGE (LOD-Lockstep-Umbau, Tracker #16) ===
        # Speichert Outputs einzelner Calculator-Knoten (siehe
        # gui/OldManagers/calculator_graph.py CALCULATOR_GRAPH), nicht nur die finalen
        # Domain-Objekte (TerrainData/GeologyData/...). Grundlage für den globalen
        # CalculatorDispatcher: Calculator-Knoten werden künftig einzeln dispatcht statt
        # nur als Teil eines ganzen Generator-Laufs und müssen ihren Zwischenzustand
        # deshalb selbstständig persistieren, damit ein abhängiger Knoten, der erst in
        # einer späteren Runde dispatcht wird, darauf zugreifen kann.
        self._calculator_data = {}  # {f"lod_{level}_{calculator_id}_{output_key}": value}
        self._current_calculator_lods = {}  # {calculator_id: höchstes abgeschlossenes LOD}

        # === CACHE-MANAGEMENT (LOD-ERWEITERT) ===
        self._cache_timestamps = {}  # {f"{generator}_{lod}_{key}": timestamp}
        self._parameter_hashes = {}  # {f"{generator}_{lod}": param_hash}

        # === CURRENT LOD-LEVELS FÜR JEDEN GENERATOR ===
        self._current_lods = {
            "terrain": 0, "geology": 0, "settlement": 0,
            "weather": 0, "water": 0, "biome": 0
        }

        # === AUTOMATISCHE GENERIERUNG - PUNKT 3 ===
        self._auto_generation_enabled = True
        self._generation_in_progress = set()  # {data_key} aktuell in Generierung

        # Memory-Management Timer für große Arrays
        self.memory_cleanup_timer = QTimer()
        self.memory_cleanup_timer.timeout.connect(self._periodic_memory_cleanup)
        self.memory_cleanup_timer.start(60000)  # 60 seconds

    def _setup_integrated_signals(self):
        """
        Setzt Signal-Verbindungen zwischen allen integrierten Komponenten auf
        PUNKT 3: Automatische Data-Generierung implementieren
        """
        # LOD Hub → DataLODManager für automatische Generierung
        self.lod_hub.data_generation_ready.connect(self._on_data_generation_ready)
        self.lod_hub.data_dependencies_satisfied.connect(self._on_data_dependencies_satisfied)

        # Generator Queue → DataLODManager
        self.generator_queue.task_started.connect(self._on_generation_task_started)
        self.generator_queue.task_completed.connect(self._on_generation_task_completed)
        self.generator_queue.task_timeout.connect(self._on_generation_task_timeout)

        # ResourceTracker → DataLODManager für Memory-Management
        self.resource_tracker.memory_warning.connect(self._on_memory_warning)
        self.resource_tracker.memory_pressure.connect(self._on_memory_pressure)

        # DisplayUpdateManager → DataLODManager für Performance-Feedback
        self.display_manager.update_performed.connect(self._on_display_update_performed)

        # Generator Registry → DataLODManager
        self.generator_registry.generator_registered.connect(self._on_generator_registered)
        self.generator_registry.generator_unregistered.connect(self._on_generator_unregistered)

        # Queue Dependency-Check-Callback setzen
        self.generator_queue.set_dependency_check_callback(self._check_task_dependencies)

    # =============================================================================
    # AUTOMATISCHE DATA-GENERIERUNG - PUNKT 3 LÖSUNG
    # =============================================================================

    @pyqtSlot(str, int)
    def _on_data_generation_ready(self, data_key: str, lod_level: int):
        """
        Slot: Data-Key ist bereit für automatische Generierung - PUNKT 3
        """
        if not self._auto_generation_enabled:
            return

        if data_key in self._generation_in_progress:
            self.logger.debug(f"Data-Key '{data_key}' already in generation queue")
            return

        # Prüfe ob Auto-Generation für diesen Data-Key konfiguriert ist
        auto_config = AUTO_GENERATION_DATA_KEYS.get(data_key)
        if not auto_config or not auto_config.get("auto", False):
            return

        # Prüfe ob Generator verfügbar ist
        generator = self.generator_registry.get_generator(data_key)
        if not generator:
            self.logger.warning(f"No generator registered for auto-generation of '{data_key}'")
            return

        # Task für Queue erstellen
        task = GenerationTask(
            data_key=data_key,
            lod_level=lod_level,
            parameters=self._get_default_parameters_for_data_key(data_key, lod_level),
            priority=auto_config.get("priority", GenerationPriority.NORMAL),
            mode=GenerationMode.AUTO,
            dependencies=generator.get_required_dependencies()
        )

        # Task einreihen
        task_id = self.generator_queue.enqueue_task(task)
        self._generation_in_progress.add(data_key)

        self.logger.info(f"Enqueued auto-generation for '{data_key}' LOD {lod_level}")
        self.auto_generation_triggered.emit(data_key, lod_level, "dependencies_satisfied")

    @pyqtSlot(str, int)
    def _on_data_dependencies_satisfied(self, data_key: str, lod_level: int):
        """
        Slot: Dependencies für Data-Key wurden erfüllt
        """
        self.logger.debug(f"Dependencies satisfied for '{data_key}' LOD {lod_level}")
        # Die eigentliche Generierung wird über data_generation_ready getriggert

    @pyqtSlot(str, int)
    def _on_generation_task_started(self, data_key: str, lod_level: int):
        """
        Slot: Generator-Task wurde gestartet
        """
        self.logger.info(f"Generation task started for '{data_key}' LOD {lod_level}")

        # Generator holen und Generierung starten
        generator = self.generator_registry.get_generator(data_key)
        if generator:
            # Generierung in separatem Thread starten (vereinfacht)
            # In realer Implementierung würde hier QThread verwendet
            parameters = self._get_default_parameters_for_data_key(data_key, lod_level)

            # Generator direkt ausführen (für Demo)
            success = generator.execute_generation(lod_level, parameters)

            # Task als abgeschlossen markieren
            self.generator_queue.mark_task_completed(data_key, lod_level, success)

    @pyqtSlot(str, int, bool)
    def _on_generation_task_completed(self, data_key: str, lod_level: int, success: bool):
        """
        Slot: Generator-Task wurde abgeschlossen
        """
        self._generation_in_progress.discard(data_key)

        if success:
            self.logger.info(f"Auto-generation completed successfully for '{data_key}' LOD {lod_level}")
        else:
            self.logger.error(f"Auto-generation failed for '{data_key}' LOD {lod_level}")

    @pyqtSlot(str, int, float)
    def _on_generation_task_timeout(self, data_key: str, lod_level: int, timeout_seconds: float):
        """
        Slot: Generator-Task hat Timeout erreicht
        """
        self._generation_in_progress.discard(data_key)
        self.logger.error(f"Auto-generation timed out for '{data_key}' LOD {lod_level} after {timeout_seconds}s")

    def _check_task_dependencies(self, task: GenerationTask) -> bool:
        """
        Callback: Prüft ob Task-Dependencies erfüllt sind - PUNKT 3
        """
        return self.lod_hub.are_data_dependencies_satisfied(task.data_key, task.lod_level)

    def _get_default_parameters_for_data_key(self, data_key: str, lod_level: int) -> Dict[str, Any]:
        """
        Generiert Default-Parameter für Data-Key-Generierung
        """
        map_size = self.get_map_size_for_lod(lod_level)

        return {
            "lod_level": lod_level,
            "map_size": map_size,
            "data_key": data_key,
            # Weitere default Parameter können hier hinzugefügt werden
        }

    def set_auto_generation_enabled(self, enabled: bool):
        """
        Aktiviert/Deaktiviert automatische Generierung
        """
        self._auto_generation_enabled = enabled
        self.logger.info(f"Auto-generation {'enabled' if enabled else 'disabled'}")

    def trigger_manual_generation(self, data_key: str, lod_level: int, parameters: Dict[str, Any],
                                  priority: GenerationPriority = GenerationPriority.HIGH) -> bool:
        """
        Triggert manuelle Generierung für Data-Key
        """
        generator = self.generator_registry.get_generator(data_key)
        if not generator:
            self.logger.error(f"No generator registered for '{data_key}'")
            return False

        task = GenerationTask(
            data_key=data_key,
            lod_level=lod_level,
            parameters=parameters,
            priority=priority,
            mode=GenerationMode.MANUAL,
            dependencies=generator.get_required_dependencies()
        )

        task_id = self.generator_queue.enqueue_task(task)
        self._generation_in_progress.add(data_key)

        self.logger.info(f"Triggered manual generation for '{data_key}' LOD {lod_level}")
        self.auto_generation_triggered.emit(data_key, lod_level, "manual_trigger")

        return True

    # =============================================================================
    # LOD CONFIG UND CORE METHODS
    # =============================================================================

    def set_lod_config(self, minimal_map_size: int, target_map_size: int):
        """Setzt LOD-Konfiguration über Hub"""
        self.lod_hub.set_lod_config(minimal_map_size, target_map_size)

    def get_current_lod_level(self, generator_type: str) -> int:
        """Gibt aktuelles LOD-Level für Generator zurück"""
        return self._current_lods.get(generator_type, 0)

    def get_map_size_for_lod(self, lod_level: int) -> int:
        """Gibt Map-Size für LOD-Level zurück"""
        return self.lod_hub.get_lod_size_for_level(lod_level)

    # =============================================================================
    # GENERATOR INTEGRATION - PUNKT 9 LÖSUNG
    # =============================================================================

    def register_data_generator(self, generator: DataKeyGeneratorBase,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        PUNKT 9: Registriert Data-Key-Generator im System
        Parameter: generator, metadata
        Return: bool - Registrierung erfolgreich
        """
        # Generator im Registry registrieren
        success = self.generator_registry.register_generator(generator, metadata)

        if success:
            # Signal-Verbindungen zwischen Generator und Hub herstellen
            self.lod_hub.connect_data_to_hub(generator.data_key, generator)

            self.logger.info(f"Successfully registered and connected generator for '{generator.data_key}'")

        return success

    def unregister_data_generator(self, data_key: str) -> bool:
        """
        PUNKT 9: Deregistriert Data-Key-Generator
        Parameter: data_key
        Return: bool - Deregistrierung erfolgreich
        """
        generator = self.generator_registry.get_generator(data_key)
        if generator:
            # Signal-Verbindungen trennen
            self.lod_hub.disconnect_data_from_hub(data_key, generator)

        # Generator deregistrieren
        return self.generator_registry.unregister_generator(data_key)

    @pyqtSlot(str, str)
    def _on_generator_registered(self, data_key: str, generator_type: str):
        """Slot: Generator wurde registriert"""
        self.logger.debug(f"Generator registered event: {data_key} ({generator_type})")

    @pyqtSlot(str)
    def _on_generator_unregistered(self, data_key: str):
        """Slot: Generator wurde deregistriert"""
        self.logger.debug(f"Generator unregistered event: {data_key}")
        # Cleanup laufender Generierungen
        self._generation_in_progress.discard(data_key)

    def get_generator_registry(self) -> DataKeyGeneratorRegistry:
        """Gibt Generator-Registry zurück für externe Zugriffe"""
        return self.generator_registry

    def _validate_lod_input(self, generator: str, data_key: str, data, lod_level) -> bool:
        """
        Funktionsweise: Prüft die Eingabe eines Generator-Setters an der Grenze zum Datenspeicher
        Aufgabe: Stellt sicher, dass nur gültige Arrays mit gültigem LOD gespeichert werden.
                 Bei Verstoß wird der Vorgang protokolliert und abgelehnt, ohne eine Exception
                 zu werfen (kein Absturz in Qt-Slots oder Worker-Threads)
        Parameter:
            generator - Generator-Name für Logging ("terrain", "geology", ...)
            data_key  - betroffener Data-Key
            data      - zu speicherndes Array
            lod_level - Ziel-LOD-Level
        Return: True wenn gültig, sonst False
        """
        if not isinstance(lod_level, int) or lod_level < 1:
            self.logger.warning(f"{generator} '{data_key}': ungültiges LOD-Level {lod_level}, wird nicht gespeichert")
            return False
        if not isinstance(data, np.ndarray):
            self.logger.warning(
                f"{generator} '{data_key}': kein numpy-Array (Typ {type(data).__name__}), wird nicht gespeichert")
            return False
        if data.size == 0:
            self.logger.warning(f"{generator} '{data_key}': leeres Array, wird nicht gespeichert")
            return False
        return True

    def _get_data_lod(self, generator: str, store: dict, data_key: str, lod_level):
        """
        Funktionsweise: Gemeinsame LOD-Auflösung für alle Generator-Getter
        Aufgabe: Löst das Ziel-LOD auf und gibt die gespeicherten Daten zurück. Fehlt das
                 gewünschte LOD, wird das beste verfügbare LOD darunter geliefert; ist gar nichts
                 vorhanden, wird None mit DEBUG-Log zurückgegeben statt eines nie existierenden
                 lod_0-Schlüssels
        Parameter:
            generator - Generator-Name für current_lods und Logging ("terrain", ...)
            store     - zugehöriges Daten-Dict (self._terrain_data, self._geology_data, ...)
            data_key  - gesuchter Data-Key
            lod_level - gewünschtes LOD oder None für das höchste verfügbare
        Return: Gespeicherte Daten oder None
        """
        if lod_level is None:
            lod_level = self._current_lods.get(generator, 0)

        if lod_level < 1:
            self.logger.debug(f"{generator} '{data_key}': kein Daten-LOD verfügbar")
            return None

        for candidate in range(lod_level, 0, -1):
            candidate_key = f"lod_{candidate}_{data_key}"
            if candidate_key in store:
                if candidate != lod_level:
                    self.logger.debug(
                        f"{generator} '{data_key}': LOD {lod_level} nicht vorhanden, nutze LOD {candidate}")
                return store[candidate_key]

        self.logger.debug(f"{generator} '{data_key}': kein Daten-LOD verfügbar")
        return None

    def _set_data_lod(self, generator: str, store: dict, data_key: str, data, lod_level: int,
                      parameters: Dict[str, Any], require_array: bool = True):
        """
        Funktionsweise: Generischer Ablage-Pfad für ein einzelnes Daten-Produkt eines Generators
        Aufgabe: Legt ein Daten-Produkt unter seinem LOD-Key im zugehörigen Store ab und aktualisiert
                 das aktuelle LOD. Array-Produkte durchlaufen die volle Eingangs-Validierung;
                 Nicht-Array-Produkte (Listen, Dicts, Skalare) werden nur auf Vorhandensein und
                 gültiges LOD geprüft.
        Parameter:
            generator     - Generator-Name für current_lods und Logging ("geology", ...)
            store         - zugehöriges Daten-Dict (self._geology_data, ...)
            data_key      - Data-Key des Produkts
            data          - abzulegendes Produkt
            lod_level     - Ziel-LOD-Level
            parameters    - verwendete Parameter (Cache-Metadaten)
            require_array - True erzwingt die np.ndarray-Validierung, False erlaubt Nicht-Arrays
        """
        if require_array:
            if not self._validate_lod_input(generator, data_key, data, lod_level):
                return
        else:
            if not isinstance(lod_level, int) or lod_level < 1:
                self.logger.warning(
                    f"{generator} '{data_key}': ungültiges LOD-Level {lod_level}, wird nicht gespeichert")
                return
            if data is None:
                self.logger.warning(f"{generator} '{data_key}': None, wird nicht gespeichert")
                return

        lod_key = f"lod_{lod_level}_{data_key}"
        store[lod_key] = data
        self._current_lods[generator] = max(self._current_lods.get(generator, 0), lod_level)

    # =============================================================================
    # FEINGRANULARER CALCULATOR-STORAGE (LOD-Lockstep-Umbau, Tracker #16)
    # =============================================================================

    def set_calculator_output(self, calculator_id: str, lod_level: int, outputs: Dict[str, Any]):
        """
        Speichert die Output(s) eines einzelnen Calculator-Knotens (siehe
        gui/OldManagers/calculator_graph.py CALCULATOR_GRAPH) für ein LOD-Level.
        Parameter:
            calculator_id - z.B. "geology.tectonic_deformation"
            lod_level     - Ziel-LOD-Level
            outputs       - dict output_key -> value (mehrere Output-Keys pro Knoten
                            möglich, siehe CalculatorSpec.output_keys). None-Werte werden
                            übersprungen statt gespeichert.
        Keine np.ndarray-Validierung wie bei _validate_lod_input(): Calculator-Outputs
        können auch Nicht-Array-Typen sein (z.B. Location-Listen bei Settlement).
        """
        if not isinstance(lod_level, int) or lod_level < 1:
            self.logger.warning(
                f"calculator '{calculator_id}': ungültiges LOD-Level {lod_level}, wird nicht gespeichert")
            return

        stored_any = False
        for output_key, value in outputs.items():
            if value is None:
                continue
            lod_key = f"lod_{lod_level}_{calculator_id}_{output_key}"
            self._calculator_data[lod_key] = value
            stored_any = True

        if stored_any:
            self._current_calculator_lods[calculator_id] = max(
                self._current_calculator_lods.get(calculator_id, 0), lod_level)

    def get_calculator_output(self, calculator_id: str, output_key: str, lod_level: int = None):
        """
        Holt einen einzelnen Output-Key eines Calculator-Knotens - bestes verfügbares
        LOD <= lod_level als Fallback (None = insgesamt bestes verfügbares LOD für
        diesen Knoten), analog zu get_terrain_data_lod()/_get_data_lod().
        Return: Gespeicherter Wert oder None
        """
        if lod_level is None:
            lod_level = self._current_calculator_lods.get(calculator_id, 0)

        if lod_level < 1:
            return None

        for candidate in range(lod_level, 0, -1):
            candidate_key = f"lod_{candidate}_{calculator_id}_{output_key}"
            if candidate_key in self._calculator_data:
                return self._calculator_data[candidate_key]

        return None

    def get_calculator_completed_lod(self, calculator_id: str) -> int:
        """
        Höchstes abgeschlossenes LOD für diesen Calculator-Knoten (0 = noch keins) -
        Grundlage für die Runden-Bereitschaftsprüfung im CalculatorDispatcher.
        """
        return self._current_calculator_lods.get(calculator_id, 0)

    # =============================================================================
    # TERRAIN DATA MANAGEMENT - LOD-ERWEITERT (BESTEHEND)
    # =============================================================================

    @data_management_handler("terrain_data")
    def set_terrain_data_lod(self, data_key: str, data: np.ndarray, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert Terrain-Arrays mit LOD-Level und Resource-Tracking
        Prüft Typ, Inhalt und LOD-Level am Eingang; ungültige Daten werden abgelehnt statt gespeichert
        """
        if not self._validate_lod_input("terrain", data_key, data, lod_level):
            return

        lod_key = f"lod_{lod_level}_{data_key}"
        self._terrain_data[lod_key] = data
        self._current_lods["terrain"] = max(self._current_lods["terrain"], lod_level)

        # Resource-Tracking für große Arrays
        if isinstance(data, np.ndarray) and data.nbytes > 10_000_000:  # >10MB
            resource_id = self.resource_tracker.register_resource(
                data, f"terrain_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key},
                data_key=data_key
            )

        self._update_cache_timestamp("terrain", lod_level, data_key, parameters)

        # Emit signals
        self.data_updated.emit("terrain", data_key)
        self.lod_data_stored.emit("terrain", lod_level, [data_key])

        # LOD Hub benachrichtigen
        self.lod_hub.on_data_lod_completed(data_key, lod_level, True)

        self.logger.debug(f"Terrain data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")

    def set_terrain_data_complete_lod(self, terrain_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert komplettes TerrainData-Objekt mit LOD-Level
        """
        # TerrainData-Objekt komplett speichern
        lod_key = f"lod_{lod_level}_terrain_data_object"
        self._terrain_data[lod_key] = terrain_data

        # Resource-Tracking
        resource_id = self.resource_tracker.register_resource(
            terrain_data, "terrain_complete",
            metadata={"lod_level": lod_level},
            data_key="terrain_complete"
        )

        # Einzelne Arrays für Legacy-Kompatibilität extrahieren
        data_keys = []
        if terrain_data.heightmap is not None:
            self.set_terrain_data_lod("heightmap", terrain_data.heightmap, lod_level, parameters)
            data_keys.append("heightmap")
        if terrain_data.slopemap is not None:
            self.set_terrain_data_lod("slopemap", terrain_data.slopemap, lod_level, parameters)
            data_keys.append("slopemap")
        if terrain_data.shadowmap is not None:
            self.set_terrain_data_lod("shadowmap", terrain_data.shadowmap, lod_level, parameters)
            data_keys.append("shadowmap")

        # Cache und Metadaten
        self._update_cache_timestamp("terrain", lod_level, "complete", parameters)
        self._current_lods["terrain"] = max(self._current_lods["terrain"], lod_level)

        # LOD-spezifische Metadaten
        self._terrain_data[f"lod_{lod_level}_actual_size"] = terrain_data.actual_size
        self._terrain_data[f"lod_{lod_level}_calculated_sun_angles"] = getattr(terrain_data, 'calculated_sun_angles',
                                                                               [])

        data_keys.append("complete")
        self.lod_data_stored.emit("terrain", lod_level, data_keys)
        self.data_updated.emit("terrain", "complete")

        self.logger.debug(f"Complete terrain data updated for LOD {lod_level}, size: {terrain_data.actual_size}")

    def get_terrain_data_lod(self, data_key: str, lod_level: int = None):
        """
        Holt Terrain-Daten für spezifisches LOD-Level
        Fehlt das gewünschte LOD, wird das beste verfügbare darunter geliefert; ist nichts vorhanden,
        wird None mit DEBUG-Log zurückgegeben statt eines nie existierenden lod_0-Schlüssels
        """
        if data_key == "complete":
            lod = lod_level if lod_level is not None else self._current_lods.get("terrain", 0)
            if lod < 1:
                self.logger.debug("terrain 'complete': kein Daten-LOD verfügbar")
                return None
            return self._terrain_data.get(f"lod_{lod}_terrain_data_object")

        return self._get_data_lod("terrain", self._terrain_data, data_key, lod_level)

    def get_terrain_data(self, data_key: str = None):
        """Legacy-Methode: Gibt höchstes verfügbares LOD zurück"""
        if data_key is None:
            # Gib alle Terrain-Daten zurück
            current_lod = self._current_lods.get("terrain", 0)
            result = {}
            for key, data in self._terrain_data.items():
                if key.startswith(f"lod_{current_lod}_"):
                    clean_key = key.replace(f"lod_{current_lod}_", "")
                    result[clean_key] = data
            return result
        else:
            return self.get_terrain_data_lod(data_key)

    def get_terrain_data_combined(self, data_key: str, lod_level: int = None):
        """
        Funktionsweise: Zugriff auf die kombinierte Heightmap (Terrain plus Geology-Tektonik
                 plus Water-Erosion/-Sedimentation)
        Aufgabe: Liefert die Datenbasis, die alle nachgelagerten Generatoren als heightmap_combined
                 erwarten: base_heightmap + geology_height_delta - erosion_map + sedimentation_map
                 (erosion_map/sedimentation_map sind laut water_generator.py positive Magnituden -
                 Erosion trägt Material ab, Sedimentation lagert es an; geology_height_delta ist
                 vorzeichenbehaftet, siehe core/geology_generator.py:_apply_tectonic_deformation).
                 Jeder der drei Anteile wird unabhängig von den anderen angewendet, wenn verfügbar
                 (z.B. Geology fertig, Water noch nicht). Ohne jede Zusatz-Daten (vor dem ersten
                 Geology-/Water-Lauf) entspricht das Ergebnis der reinen Terrain-Heightmap. Anteile
                 in falscher Auflösung (jeweiliger Generator noch nicht auf demselben LOD) werden
                 übersprungen statt mit falscher Form zu verrechnen.
        Parameter:
            data_key  - Terrain-Data-Key, üblicherweise "heightmap"
            lod_level - gewünschtes LOD oder None für das höchste verfügbare
        Return: Kombinierte Terrain-Daten oder None
        """
        base = self.get_terrain_data_lod(data_key, lod_level)

        if data_key != "heightmap" or base is None:
            return base

        combined = base.copy()

        geology_height_delta = self.get_geology_data_lod("height_delta", lod_level)
        if geology_height_delta is not None and geology_height_delta.shape == base.shape:
            combined = combined + geology_height_delta

        erosion_map = self.get_water_data_lod("erosion_map", lod_level)
        if erosion_map is not None and erosion_map.shape == base.shape:
            combined = combined - erosion_map

        sedimentation_map = self.get_water_data_lod("sedimentation_map", lod_level)
        if sedimentation_map is not None and sedimentation_map.shape == base.shape:
            combined = combined + sedimentation_map

        return combined

    def get_calculator_combined_heightmap(self, lod_level: int):
        """
        Funktionsweise: Wie get_terrain_data_combined("heightmap", ...), liest die drei
                 Anteile (Terrain-Basis, Geology-Tektonik, Water-Erosion/-Sedimentation)
                 aber direkt aus dem feingranularen Calculator-Storage
                 (terrain.redistribution/geology.tectonic_deformation/
                 water.erosion_sedimentation) statt aus dem Domain-Level-Storage
                 (_terrain_data/_geology_data/_water_data).
        Aufgabe: Domain-Level-Storage wird erst befüllt, wenn ALLE Calculator-Knoten
                 EINES Generators ein LOD abgeschlossen haben und der Orchestrator
                 assemble_*_data() aufgerufen hat (_maybe_assemble_generator) - das
                 passiert oft deutlich SPÄTER als der einzelne Calculator-Knoten
                 terrain.redistribution, von dem z.B. geology.classify_elevation laut
                 CALCULATOR_GRAPH bereits abhängen darf. _calc_*-Methoden anderer
                 Generatoren, die "heightmap_combined" als Input brauchen, müssen
                 deshalb diese Methode statt get_terrain_data_combined() verwenden,
                 sonst schlagen sie fehl, obwohl ihre einzige echte Abhängigkeit
                 (terrain.redistribution) längst fertig ist.
        Parameter:
            lod_level - gewünschtes LOD (best verfügbares darunter als Fallback,
                        analog get_calculator_output())
        Return: Kombinierte Heightmap oder None, wenn terrain.redistribution noch
                 nichts für dieses/ein niedrigeres LOD geliefert hat
        """
        base = self.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        if base is None:
            return None

        combined = base.copy()

        geology_height_delta = self.get_calculator_output(
            "geology.tectonic_deformation", "height_delta", lod_level)
        if geology_height_delta is not None and geology_height_delta.shape == base.shape:
            combined = combined + geology_height_delta

        erosion_map = self.get_calculator_output("water.erosion_sedimentation", "erosion_map", lod_level)
        if erosion_map is not None and erosion_map.shape == base.shape:
            combined = combined - erosion_map

        sedimentation_map = self.get_calculator_output(
            "water.erosion_sedimentation", "sedimentation_map", lod_level)
        if sedimentation_map is not None and sedimentation_map.shape == base.shape:
            combined = combined + sedimentation_map

        return combined

    # =============================================================================
    # GEOLOGY DATA MANAGEMENT - LOD-ERWEITERT (VEREINFACHT)
    # =============================================================================

    @data_management_handler("geology_data")
    def set_geology_data_lod(self, data_key: str, data: np.ndarray, lod_level: int, parameters: Dict[str, Any]):
        """Speichert Geology-Generator Output mit LOD"""
        if not self._validate_lod_input("geology", data_key, data, lod_level):
            return

        lod_key = f"lod_{lod_level}_{data_key}"
        self._geology_data[lod_key] = data
        self._current_lods["geology"] = max(self._current_lods["geology"], lod_level)

        # Resource-Tracking
        if isinstance(data, np.ndarray) and data.nbytes > 10_000_000:
            self.resource_tracker.register_resource(
                data, f"geology_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key},
                data_key=data_key
            )

        self._update_cache_timestamp("geology", lod_level, data_key, parameters)
        self.data_updated.emit("geology", data_key)
        self.lod_data_stored.emit("geology", lod_level, [data_key])

        self.logger.debug(f"Geology data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")

    def set_geology_data_complete_lod(self, geology_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert komplettes GeologyData-Objekt mit LOD-Level und zerlegt es in seine Data-Keys
        """
        self._geology_data[f"lod_{lod_level}_geology_data_object"] = geology_data

        data_keys = []
        for key in ("rock_map", "hardness_map", "height_delta"):
            value = getattr(geology_data, key, None)
            if value is not None:
                self._set_data_lod("geology", self._geology_data, key, value, lod_level, parameters)
                data_keys.append(key)

        self._update_cache_timestamp("geology", lod_level, "complete", parameters)
        data_keys.append("complete")
        self.lod_data_stored.emit("geology", lod_level, data_keys)
        self.data_updated.emit("geology", "complete")

    def set_weather_data_complete_lod(self, weather_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert komplettes WeatherData-Objekt mit LOD-Level und zerlegt es in seine Data-Keys
        """
        self._weather_data[f"lod_{lod_level}_weather_data_object"] = weather_data

        data_keys = []
        for key in ("wind_map", "temp_map", "precip_map", "humid_map"):
            value = getattr(weather_data, key, None)
            if value is not None:
                self._set_data_lod("weather", self._weather_data, key, value, lod_level, parameters)
                data_keys.append(key)

        # Saisonale Monats-Listen (je 6 np.ndarray) - Nicht-Array-Produkte,
        # analog zum bestehenden ocean_outflow/biome_statistics-Muster
        # (require_array=False), für die animierte Weather-Tab-Anzeige.
        for key in ("wind_map_monthly", "temp_map_monthly", "precip_map_monthly", "humid_map_monthly"):
            value = getattr(weather_data, key, None)
            if value:
                self._set_data_lod("weather", self._weather_data, key, value, lod_level,
                                   parameters, require_array=False)
                data_keys.append(key)

        self._update_cache_timestamp("weather", lod_level, "complete", parameters)
        data_keys.append("complete")
        self.lod_data_stored.emit("weather", lod_level, data_keys)
        self.data_updated.emit("weather", "complete")

    def set_water_data_complete_lod(self, water_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert komplettes WaterData-Objekt mit LOD-Level und zerlegt es in seine Data-Keys
        """
        self._water_data[f"lod_{lod_level}_water_data_object"] = water_data

        data_keys = []
        for key in ("water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
                    "erosion_map", "sedimentation_map", "evaporation_map", "water_biomes_map"):
            value = getattr(water_data, key, None)
            if value is not None:
                self._set_data_lod("water", self._water_data, key, value, lod_level, parameters)
                data_keys.append(key)

        # Nicht-Array-Produkt: ocean_outflow ist ein Skalar
        if getattr(water_data, "ocean_outflow", None) is not None:
            self._set_data_lod("water", self._water_data, "ocean_outflow", water_data.ocean_outflow,
                               lod_level, parameters, require_array=False)
            data_keys.append("ocean_outflow")

        self._update_cache_timestamp("water", lod_level, "complete", parameters)
        data_keys.append("complete")
        self.lod_data_stored.emit("water", lod_level, data_keys)
        self.data_updated.emit("water", "complete")

    def set_biome_data_complete_lod(self, biome_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert komplettes BiomeData-Objekt mit LOD-Level und zerlegt es in seine Data-Keys
        """
        self._biome_data[f"lod_{lod_level}_biome_data_object"] = biome_data

        data_keys = []
        for key in ("biome_map", "biome_map_super", "super_biome_mask", "climate_classification"):
            value = getattr(biome_data, key, None)
            if value is not None:
                self._set_data_lod("biome", self._biome_data, key, value, lod_level, parameters)
                data_keys.append(key)

        # Nicht-Array-Produkt: biome_statistics ist ein Dict
        if getattr(biome_data, "biome_statistics", None) is not None:
            self._set_data_lod("biome", self._biome_data, "biome_statistics", biome_data.biome_statistics,
                               lod_level, parameters, require_array=False)
            data_keys.append("biome_statistics")

        self._update_cache_timestamp("biome", lod_level, "complete", parameters)
        data_keys.append("complete")
        self.lod_data_stored.emit("biome", lod_level, data_keys)
        self.data_updated.emit("biome", "complete")

    def set_settlement_data_complete_lod(self, settlement_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Speichert komplettes SettlementData-Objekt mit LOD-Level und zerlegt es in seine Data-Keys
        """
        self._settlement_data[f"lod_{lod_level}_settlement_data_object"] = settlement_data

        data_keys = []
        # city_mask/voronoi_cell_map/street_mask/house_parcel_map: neue Arrays aus
        # dem Settlement-Rework (#35-#37, siehe docs/backlog.md Ticket #4) -
        # ohne diese Ergänzung bleiben SettlementTab.update_settlement_display()s
        # neue Display-Modi (City Boundary/Landscape Voronoi/City Blocks) leer,
        # exakt derselbe Bug-Typ wie zuvor bei plot_map/civ_map (siehe
        # docs/generation_pipeline_dependencies.md, Punkt 9 der Bugliste).
        for key in ("plot_map", "civ_map", "combined_suitability_map",
                    "city_mask", "voronoi_cell_map", "street_mask", "house_parcel_map"):
            value = getattr(settlement_data, key, None)
            if value is not None:
                self._set_data_lod("settlement", self._settlement_data, key, value, lod_level, parameters)
                data_keys.append(key)

        # Nicht-Array-Produkte: Location-Listen und Straßen/Plot-Strukturen
        for key in ("settlement_list", "landmark_list", "roadsite_list", "roads", "plots", "plot_nodes",
                    "landmark_roads", "outer_roads", "plot_edges"):
            value = getattr(settlement_data, key, None)
            if value:
                self._set_data_lod("settlement", self._settlement_data, key, value, lod_level,
                                   parameters, require_array=False)
                data_keys.append(key)

        self._update_cache_timestamp("settlement", lod_level, "complete", parameters)
        data_keys.append("complete")
        self.lod_data_stored.emit("settlement", lod_level, data_keys)
        self.data_updated.emit("settlement", "complete")

    def get_geology_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Geology-Daten für LOD-Level zurück (bestes verfügbares LOD als Fallback)"""
        return self._get_data_lod("geology", self._geology_data, data_key, lod_level)

    def get_geology_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode"""
        return self.get_geology_data_lod(data_key)

    # =============================================================================
    # REST DATA MANAGEMENT - LOD-ERWEITERT (PRÜFEN OB SPÄTER SETTER HIER HER GEBRACHT WERDEN)
    # =============================================================================

    def get_weather_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Weather-Daten für LOD-Level zurück (bestes verfügbares LOD als Fallback)"""
        return self._get_data_lod("weather", self._weather_data, data_key, lod_level)

    def get_weather_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode"""
        return self.get_weather_data_lod(data_key)

    def get_water_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Water-Daten für LOD-Level zurück (bestes verfügbares LOD als Fallback)"""
        return self._get_data_lod("water", self._water_data, data_key, lod_level)

    def get_water_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode"""
        return self.get_water_data_lod(data_key)

    def get_biome_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Biome-Daten für LOD-Level zurück (bestes verfügbares LOD als Fallback)"""
        return self._get_data_lod("biome", self._biome_data, data_key, lod_level)

    def get_biome_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode"""
        return self.get_biome_data_lod(data_key)

    def get_settlement_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Settlement-Daten für LOD-Level zurück (bestes verfügbares LOD als Fallback)"""
        return self._get_data_lod("settlement", self._settlement_data, data_key, lod_level)

    def get_settlement_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode"""
        return self.get_settlement_data_lod(data_key)

    def check_dependencies(self, generator_type: str, required_dependencies: list) -> tuple:
        """
        Funktionsweise: Prüft Verfügbarkeit aller Required Dependencies eines Generators
        Aufgabe: Gegenstück zu check_input_dependencies() der Tabs
        Parameter: generator_type (str), required_dependencies (list of data_keys)
        Return: (is_complete: bool, missing: list of data_keys)
        """
        getters = [
            self.get_terrain_data, self.get_geology_data, self.get_weather_data,
            self.get_water_data, self.get_biome_data, self.get_settlement_data
        ]

        missing = []
        for data_key in required_dependencies:
            available = False
            for getter in getters:
                try:
                    if getter(data_key) is not None:
                        available = True
                        break
                except Exception:
                    continue
            if not available:
                missing.append(data_key)

        return len(missing) == 0, missing

    # =============================================================================
    # MEMORY-MANAGEMENT INTEGRATION - PUNKT 7 ERWEITERT
    # =============================================================================

    @pyqtSlot(int, int)
    def _on_memory_warning(self, current_mb: int, threshold_mb: int):
        """Callback für Memory-Warnings vom Resource-Tracker"""
        self.logger.warning(f"Memory warning triggered: {current_mb}MB > {threshold_mb}MB")

        # Automatisches Cleanup bei Memory-Warnings
        cleaned_resources = self.cleanup_old_lod_resources(max_age_hours=1.0)
        memory_freed = self.optimize_display_cache()

        self.memory_optimization_performed.emit("memory_warning_cleanup", cleaned_resources, memory_freed)

    @pyqtSlot(str, int)
    def _on_memory_pressure(self, data_key: str, memory_mb: int):
        """Callback für Data-Key-spezifische Memory-Pressure"""
        self.logger.warning(f"Memory pressure for '{data_key}': {memory_mb}MB")

        # Data-Key-spezifisches Cleanup
        cleaned_count = self.resource_tracker.cleanup_data_resources(data_key)

        if cleaned_count > 0:
            self.memory_optimization_performed.emit(f"data_key_cleanup_{data_key}", cleaned_count, memory_mb)

    @pyqtSlot(str, str, float)
    def _on_display_update_performed(self, display_id: str, layer_type: str, hash_time: float):
        """Callback für Display-Updates vom Display-Manager"""
        if hash_time > 0.1:  # Slow hash calculation
            self.logger.debug(f"Slow display hash calculation for {display_id}: {hash_time:.3f}s")

    def cleanup_old_lod_resources(self, max_age_hours: float = 2.0) -> int:
        """
        ERWEITERT: Cleaned alte LOD-Ressourcen für Memory-Management
        """
        max_age_seconds = max_age_hours * 3600
        cleaned_count = self.resource_tracker.cleanup_by_age(max_age_seconds)

        if cleaned_count > 0:
            self.logger.info(f"Cleaned {cleaned_count} old LOD resources (>{max_age_hours}h)")

        return cleaned_count

    def optimize_display_cache(self) -> int:
        """
        ERWEITERT: Optimiert Display-Cache für bessere Performance
        Return: Geschätzte freigegebene Memory in MB
        """
        # Entferne alte Display-Cache-Einträge
        current_time = time.time()
        old_displays = [
            display_id for display_id, last_update in self.display_manager.last_update_times.items()
            if current_time - last_update > 1800  # 30 minutes
        ]

        for display_id in old_displays:
            self.display_manager.clear_display_cache(display_id)

        # Geschätzte Memory-Ersparnis (vereinfacht)
        memory_freed_mb = len(old_displays) * 5  # ~5MB pro Display geschätzt

        if old_displays:
            self.logger.debug(f"Cleaned {len(old_displays)} old display cache entries")

        return memory_freed_mb

    def _periodic_memory_cleanup(self):
        """Periodisches Memory-Cleanup für große Arrays"""
        try:
            # Statistiken sammeln
            stats = self.get_integrated_statistics()
            total_memory = sum(stats["data_manager"]["memory_usage_mb"].values())

            # Cleanup-Schwellwerte
            if total_memory > 1000:  # >1GB total
                cleaned = self.cleanup_old_lod_resources(max_age_hours=1.5)
                if cleaned > 0:
                    self.memory_optimization_performed.emit("periodic_cleanup", cleaned, int(total_memory * 0.1))

            if total_memory > 2000:  # >2GB total - aggressive cleanup
                cleaned = self.resource_tracker.cleanup_lru_resources(10)
                if cleaned > 0:
                    self.memory_optimization_performed.emit("aggressive_cleanup", cleaned, int(total_memory * 0.2))

            # Display-Cache-Optimierung
            memory_freed = self.optimize_display_cache()
            if memory_freed > 0:
                self.memory_optimization_performed.emit("display_cache_cleanup", 1, memory_freed)

            # Force Garbage Collection bei hoher Memory-Usage
            if total_memory > 1500:  # >1.5GB
                import gc
                gc.collect()

        except Exception as e:
            self.logger.error(f"Periodic memory cleanup failed: {e}")

    # =============================================================================
    # CACHE MANAGEMENT UND UTILITIES - PUNKT 10
    # =============================================================================

    def _update_cache_timestamp(self, generator_type: str, lod_level: int, data_key: str,
                                parameters: Dict[str, Any]):
        """
        Aktualisiert Cache-Timestamp und Parameter-Hash für LOD-spezifische Invalidation
        """
        import time
        import hashlib

        timestamp = time.time()
        param_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()

        cache_key = f"{generator_type}_lod_{lod_level}_{data_key}"
        self._cache_timestamps[cache_key] = timestamp
        self._parameter_hashes[f"{generator_type}_lod_{lod_level}"] = param_hash

    def is_cache_valid_lod(self, generator_type: str, lod_level: int, data_key: str,
                           parameters: Dict[str, Any]) -> bool:
        """
        Prüft ob Cache für LOD-spezifische Parameter noch valid ist
        """
        import hashlib

        cache_key = f"{generator_type}_lod_{lod_level}"

        if cache_key not in self._parameter_hashes:
            return False

        current_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()
        return self._parameter_hashes[cache_key] == current_hash

    def invalidate_cache_lod(self, generator_type: str, lod_level: int = None):
        """
        Invalidiert Cache für spezifisches LOD-Level oder alle LODs
        """
        if lod_level is None:
            # Alle LODs für Generator invalidieren
            keys_to_remove = [key for key in self._cache_timestamps.keys()
                              if key.startswith(f"{generator_type}_lod_")]
            param_keys_to_remove = [key for key in self._parameter_hashes.keys()
                                    if key.startswith(f"{generator_type}_lod_")]
        else:
            # Nur spezifisches LOD invalidieren
            keys_to_remove = [key for key in self._cache_timestamps.keys()
                              if key.startswith(f"{generator_type}_lod_{lod_level}_")]
            param_keys_to_remove = [f"{generator_type}_lod_{lod_level}"]

        # Cache-Einträge entfernen
        for key in keys_to_remove:
            del self._cache_timestamps[key]
        for key in param_keys_to_remove:
            if key in self._parameter_hashes:
                del self._parameter_hashes[key]

        # Resource-Cleanup bei vollständiger Invalidierung
        if lod_level is None:
            self.resource_tracker.cleanup_by_type(generator_type)

            # Daten löschen - alle 6 Generatoren, nicht nur terrain/geology.
            # Fehlte vorher für weather/water/biome/settlement: deren
            # _current_lods blieb auf dem alten (bereits erreichten) Wert
            # stehen, wodurch create_lod_sequence() bei jeder erneuten
            # Generation current_lod >= target_lod sah und NICHTS tat - der
            # Grund, warum ein zweiter Generate-Klick wirkungslos blieb.
            generator_data_map = {
                "terrain": self._terrain_data,
                "geology": self._geology_data,
                "weather": self._weather_data,
                "water": self._water_data,
                "biome": self._biome_data,
                "settlement": self._settlement_data,
            }
            data_dict = generator_data_map.get(generator_type)
            if data_dict is not None:
                data_dict.clear()
                self._current_lods[generator_type] = 0

        self.cache_invalidated.emit(generator_type)
        self.logger.info(f"Cache invalidated for {generator_type}" +
                         (f" LOD {lod_level}" if lod_level is not None else " (all LODs)"))

    # =============================================================================
    # UTILITY METHODS - PUNKT 10 LÖSUNG
    # =============================================================================

    def has_data_lod(self, data_key: str, lod_level: int) -> bool:
        """
        PUNKT 10: Prüft ob spezifischer Daten-Key für LOD-Level verfügbar ist
        """
        lod_key_pattern = f"lod_{lod_level}_{data_key}"

        all_data_dicts = [
            self._terrain_data,
            self._geology_data,
            self._settlement_data,
            self._weather_data,
            self._water_data,
            self._biome_data
        ]

        for data_dict in all_data_dicts:
            if lod_key_pattern in data_dict and data_dict[lod_key_pattern] is not None:
                return True

        return False

    def has_data(self, data_key: str) -> bool:
        """Legacy-Methode: Prüft höchstes verfügbares LOD"""
        for lod_level in range(10, 0, -1):  # Prüfe LOD 10 bis 1
            if self.has_data_lod(data_key, lod_level):
                return True
        return False

    def get_memory_usage_by_lod(self) -> Dict[str, Dict[str, float]]:
        """
        PUNKT 10: Berechnet Memory-Usage aufgeschlüsselt nach Generator und LOD
        """

        def calculate_lod_memory(data_dict):
            lod_memory = {}
            for key, data in data_dict.items():
                if key.startswith("lod_") and isinstance(data, np.ndarray):
                    try:
                        lod_part = key.split("_")[1]
                        lod_level = f"LOD_{lod_part}"
                        if lod_level not in lod_memory:
                            lod_memory[lod_level] = 0
                        lod_memory[lod_level] += data.nbytes / (1024 * 1024)  # Convert to MB
                    except (IndexError, ValueError):
                        continue
            return lod_memory

        return {
            "terrain": calculate_lod_memory(self._terrain_data),
            "geology": calculate_lod_memory(self._geology_data),
            "settlement": calculate_lod_memory(self._settlement_data),
            "weather": calculate_lod_memory(self._weather_data),
            "water": calculate_lod_memory(self._water_data),
            "biome": calculate_lod_memory(self._biome_data)
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Legacy-Methode: Berechnet Memory-Usage aller gespeicherten Arrays
        """

        def calculate_array_memory(data_dict):
            total_memory = 0
            for data in data_dict.values():
                if isinstance(data, np.ndarray):
                    total_memory += data.nbytes
            return total_memory / (1024 * 1024)  # Convert to MB

        # Kombiniere Data-Manager-Memory mit Resource-Tracker-Memory
        data_memory = {
            "terrain": calculate_array_memory(self._terrain_data),
            "geology": calculate_array_memory(self._geology_data),
            "settlement": calculate_array_memory(self._settlement_data),
            "weather": calculate_array_memory(self._weather_data),
            "water": calculate_array_memory(self._water_data),
            "biome": calculate_array_memory(self._biome_data)
        }

        # Resource-Tracker-Memory hinzufügen
        resource_memory = self.resource_tracker.get_memory_usage()
        for resource_type, memory_bytes in resource_memory.items():
            memory_mb = memory_bytes / (1024 * 1024)
            if resource_type.startswith("terrain"):
                data_memory["terrain"] += memory_mb
            elif resource_type.startswith("geology"):
                data_memory["geology"] += memory_mb
            elif resource_type.startswith("weather"):
                data_memory["weather"] += memory_mb
            elif resource_type.startswith("water"):
                data_memory["water"] += memory_mb
            elif resource_type.startswith("biome"):
                data_memory["biome"] += memory_mb
            elif resource_type.startswith("settlement"):
                data_memory["settlement"] += memory_mb

        return data_memory

    def get_integrated_statistics(self) -> Dict[str, Any]:
        """
        PUNKT 10: Sammelt alle integrierten Statistics für Monitoring
        """
        return {
            "data_manager": {
                "current_lods": self._current_lods,
                "total_data_objects": sum(len(d) for d in [
                    self._terrain_data, self._geology_data, self._settlement_data,
                    self._weather_data, self._water_data, self._biome_data
                ]),
                "cache_entries": len(self._cache_timestamps),
                "memory_usage_mb": self.get_memory_usage(),
                "auto_generation_enabled": self._auto_generation_enabled,
                "active_generations": len(self._generation_in_progress)
            },
            "resource_tracker": self.resource_tracker.get_resource_statistics(),
            "display_manager": self.display_manager.get_cache_statistics(),
            "generator_registry": self.generator_registry.get_registry_statistics(),
            "generator_queue": self.generator_queue.get_queue_statistics(),
            "lod_hub": self.lod_hub.get_hub_statistics()
        }

    def export_data_summary_lod(self) -> Dict[str, Any]:
        """
        PUNKT 10: Erstellt LOD-erweiterte Zusammenfassung
        """

        def extract_lod_info(data_dict):
            lod_data = {}
            for key, data in data_dict.items():
                if key.startswith("lod_"):
                    try:
                        parts = key.split("_", 2)
                        lod_level = int(parts[1])
                        data_key = parts[2]

                        if lod_level not in lod_data:
                            lod_data[lod_level] = {}

                        if isinstance(data, np.ndarray):
                            lod_data[lod_level][data_key] = {
                                "shape": data.shape,
                                "dtype": str(data.dtype),
                                "memory_mb": data.nbytes / (1024 * 1024)
                            }
                        else:
                            lod_data[lod_level][data_key] = {
                                "type": type(data).__name__,
                                "value": str(data) if not isinstance(data, (list,
                                                                            dict)) else f"{type(data).__name__}({len(data)})"
                            }
                    except (IndexError, ValueError):
                        continue
            return lod_data

        summary = {
            "terrain": extract_lod_info(self._terrain_data),
            "geology": extract_lod_info(self._geology_data),
            "settlement": extract_lod_info(self._settlement_data),
            "weather": extract_lod_info(self._weather_data),
            "water": extract_lod_info(self._water_data),
            "biome": extract_lod_info(self._biome_data),
            "current_lods": self._current_lods.copy(),
            "memory_usage_by_lod": self.get_memory_usage_by_lod(),
            "total_memory_usage_mb": self.get_memory_usage(),
            "integrated_statistics": self.get_integrated_statistics()
        }

        # LOD-Konfiguration hinzufügen
        if self.lod_hub.lod_config:
            summary["lod_config"] = {
                "minimal_map_size": self.lod_hub.lod_config.minimal_map_size,
                "target_map_size": self.lod_hub.lod_config.target_map_size,
                "max_terrain_lod": self.lod_hub.lod_config.max_terrain_lod
            }

        return summary

    def clear_all_data(self):
        """
        Funktionsweise: Löscht alle Daten und Cache - ERWEITERT mit Resource-Management
        """
        # Stoppe automatische Generierung
        self._auto_generation_enabled = False
        self._generation_in_progress.clear()

        # Queue leeren
        self.generator_queue.clear_queue()

        # Resource-Cleanup ZUERST
        self.resource_tracker.cleanup_resources()
        self.display_manager.clear_all_cache()

        # Daten löschen
        self._terrain_data.clear()
        self._geology_data.clear()
        self._settlement_data.clear()
        self._weather_data.clear()
        self._water_data.clear()
        self._biome_data.clear()

        # Cache löschen
        self._cache_timestamps.clear()
        self._parameter_hashes.clear()

        # LOD-Levels zurücksetzen
        for generator in self._current_lods:
            self._current_lods[generator] = 0

        # LOD-Hub Status zurücksetzen
        self.lod_hub.reset_hub_state()

        # Automatische Generierung wieder aktivieren
        self._auto_generation_enabled = True

        # Force Garbage Collection nach großem Cleanup
        import gc
        gc.collect()

        self.logger.info("All data, LOD status, and resources cleared")

    # =============================================================================
    # INTEGRATION METHODS - PUNKT 9 FORTSETZUNG
    # =============================================================================

    def get_lod_hub(self) -> LODCommunicationHub:
        """Gibt LOD Communication Hub zurück für externe Integration"""
        return self.lod_hub

    def get_resource_tracker(self) -> ResourceTracker:
        """Gibt Resource Tracker zurück für externe Integration"""
        return self.resource_tracker

    def get_display_manager(self) -> DisplayUpdateManager:
        """Gibt Display Update Manager zurück für externe Integration"""
        return self.display_manager

    def connect_tab_to_hub(self, tab_name: str, tab_widget):
        """
        Funktionsweise: Verbindet Tab-Signals mit LOD-Hub
        """
        self.lod_hub.connect_tab_to_hub(tab_name, tab_widget)

    def cleanup_resources(self):
        """
        Funktionsweise: Complete Cleanup für App-Shutdown - INTEGRIERT
        """
        self.logger.info("Starting comprehensive DataLODManager cleanup")

        # Timer stoppen
        if hasattr(self, 'memory_cleanup_timer'):
            self.memory_cleanup_timer.stop()

        # Generator-Queue cleanup
        self.generator_queue.clear_queue()
        if hasattr(self.generator_queue, 'processing_timer'):
            self.generator_queue.processing_timer.stop()

        # Alle Generatoren cleanup
        self.generator_registry.cleanup_all_generators()

        # Resource-Cleanup
        self.resource_tracker.cleanup_resources()

        # Display-Cache-Cleanup
        self.display_manager.clear_all_cache()

        # Alle Daten löschen
        self.clear_all_data()

        self.logger.info("DataLODManager cleanup completed")


# =============================================================================
# UTILITY FUNCTIONS UND FACTORY
# =============================================================================

def create_integrated_data_lod_manager(memory_threshold_mb: int = 500) -> DataLODManager:
    """
    PUNKT 9: Factory für DataLODManager mit konfigurierten Komponenten
    Parameter: memory_threshold_mb
    Return: Vollständig konfigurierter DataLODManager
    """
    manager = DataLODManager(memory_threshold_mb)

    # LOD-Config mit Standard-Werten setzen falls nicht anders spezifiziert
    manager.set_lod_config(minimal_map_size=32, target_map_size=512)

    # Data-Key-spezifische Memory-Thresholds setzen
    manager.resource_tracker.set_data_key_memory_threshold("heightmap", 200)  # 200MB
    manager.resource_tracker.set_data_key_memory_threshold("terrain_complete", 300)  # 300MB
    manager.resource_tracker.set_data_key_memory_threshold("biome_map_super", 500)  # 500MB für super-sampling

    return manager


def validate_data_lod_system(manager: DataLODManager) -> Dict[str, Any]:
    """
    PUNKT 10: Validiert das komplette Data-LOD-System
    Parameter: manager
    Return: Validation-Results
    """
    validation_results = {
        "system_healthy": True,
        "issues": [],
        "warnings": [],
        "statistics": {}
    }

    try:
        # Generator-Registry validieren
        generator_validation = manager.generator_registry.validate_all_generators()
        failed_generators = [k for k, v in generator_validation.items() if not v]

        if failed_generators:
            validation_results["issues"].append(f"Failed generator validation: {failed_generators}")
            validation_results["system_healthy"] = False

        # Dependency-Graph validieren
        data_valid, data_cycles = manager.lod_hub.validate_dependency_graph("data")
        tab_valid, tab_cycles = manager.lod_hub.validate_dependency_graph("tab")

        if not data_valid:
            validation_results["issues"].append(f"Circular data dependencies: {data_cycles}")
            validation_results["system_healthy"] = False

        if not tab_valid:
            validation_results["issues"].append(f"Circular tab dependencies: {tab_cycles}")
            validation_results["system_healthy"] = False

        # Memory-Usage prüfen
        memory_stats = manager.get_memory_usage()
        total_memory = sum(memory_stats.values())

        if total_memory > 2000:  # >2GB
            validation_results["warnings"].append(f"High memory usage: {total_memory:.1f}MB")

        # Queue-Status prüfen
        queue_stats = manager.generator_queue.get_queue_statistics()
        if queue_stats["failed_tasks"] > queue_stats["completed_tasks"] * 0.1:  # >10% failure rate
            validation_results["warnings"].append(
                f"High task failure rate: {queue_stats['failed_tasks']}/{queue_stats['completed_tasks']}")

        # Statistiken sammeln
        validation_results["statistics"] = manager.get_integrated_statistics()

    except Exception as e:
        validation_results["system_healthy"] = False
        validation_results["issues"].append(f"Validation error: {str(e)}")

    return validation_results