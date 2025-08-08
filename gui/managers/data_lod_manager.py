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

INTEGRIERTE FEATURES:
- DataManager: Numerisches LOD-System, Generator-Daten-Speicherung, Cache-Management
- LODCommunicationHub: Signal-basierte Tab-Kommunikation, Dependency-Tracking
- ResourceTracker: Systematisches Resource-Management, Memory-Leak-Prevention
- DisplayUpdateManager: Change-Detection für Display-Updates, Performance-Optimierung
"""

import numpy as np
import weakref
import time
import hashlib
import logging
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer

from gui.config.value_default import TERRAIN


def get_data_manager_error_decorators():
    """Lazy Loading von Data Manager Error Decorators"""
    try:
        from gui.error_handler import data_management_handler, memory_critical_handler
        return data_management_handler, memory_critical_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator

data_management_handler, memory_critical_handler = get_data_manager_error_decorators()

# =============================================================================
# LOD SYSTEM CORE CLASSES
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
    """Berechnet maximale LOD für gegebene Map-Size"""
    if map_size_min is None:
        map_size_min = TERRAIN.MAPSIZEMIN
    import math
    return int(math.log2(target_map_size / map_size_min)) + 1

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
class LODConfig:
    """LOD-Konfiguration für Map-size-proportionale Skalierung"""
    minimal_map_size: int  # z.B. 32 aus TerrainConstants
    target_map_size: int   # z.B. 288 aus Parameter
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
        """Berechnet maximale sinnvolle LOD für target_map_size"""
        import math
        max_doubling_lod = int(math.log2(self.target_map_size / self.minimal_map_size)) + 1
        return max_doubling_lod

# Tab-Dependencies: Nur Tab-abhängig, gleiche LOD-Stufe
DEPENDENCY_MATRIX = {
    "terrain": [],                           # Keine Dependencies  
    "geology": ["terrain"],                  # Gleiche LOD-Stufe
    "weather": ["terrain"],                  # Gleiche LOD-Stufe
    "water": ["terrain"],                    # Gleiche LOD-Stufe  
    "biome": ["weather", "water"],          # Gleiche LOD-Stufe
    "settlement": ["terrain", "geology"]     # Gleiche LOD-Stufe
}

# =============================================================================
# RESOURCE MANAGEMENT CLASSES (INTEGRATED)
# =============================================================================

@dataclass
class ResourceInfo:
    """Information über eine tracked Resource"""
    resource_id: str
    resource_type: str
    creation_time: float
    cleanup_func: Optional[Callable]
    estimated_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResourceTracker(QObject):
    """
    Funktionsweise: Systematisches Resource-Management für Memory-Leak-Prevention - INTEGRIERT
    Aufgabe: Verfolgt alle erstellten Ressourcen und ermöglicht systematisches Cleanup
    Features: WeakReference-Tracking, automatisches Cleanup bei GC, Resource-Type-Management
    """

    # Signals für Resource-Management
    resource_registered = pyqtSignal(str, str)  # (resource_id, resource_type)
    resource_cleaned = pyqtSignal(str, str)  # (resource_id, resource_type)
    memory_warning = pyqtSignal(int, int)  # (current_mb, threshold_mb)

    def __init__(self, memory_threshold_mb: int = 500):
        super().__init__()

        self.tracked_resources = {}  # {resource_id: WeakReference}
        self.resource_info = {}  # {resource_id: ResourceInfo}
        self.type_registry = defaultdict(set)  # {resource_type: {resource_ids}}

        self._next_id = 0
        self.memory_threshold_mb = memory_threshold_mb

        # Automatic cleanup timer
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._automatic_cleanup)
        self.cleanup_timer.start(30000)  # 30 seconds

        self.logger = logging.getLogger(__name__)

    def register_resource(self, resource: Any, resource_type: str,
                          cleanup_func: Optional[Callable] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Funktionsweise: Registriert Ressource für Tracking
        Parameter: resource, resource_type, cleanup_func, metadata
        Return: resource_id für Referenzierung
        """
        resource_id = f"{resource_type}_{self._next_id}"
        self._next_id += 1

        # WeakReference mit Callback für automatisches Cleanup
        weak_ref = weakref.ref(resource, lambda ref: self._on_resource_deleted(resource_id))

        # Resource-Info erstellen
        estimated_size = self._estimate_resource_size(resource)
        info = ResourceInfo(
            resource_id=resource_id,
            resource_type=resource_type,
            creation_time=time.time(),
            cleanup_func=cleanup_func,
            estimated_size=estimated_size,
            metadata=metadata or {}
        )

        # Registrierung
        self.tracked_resources[resource_id] = weak_ref
        self.resource_info[resource_id] = info
        self.type_registry[resource_type].add(resource_id)

        self.logger.debug(f"Registered resource: {resource_id} ({resource_type}, {estimated_size} bytes)")
        self.resource_registered.emit(resource_id, resource_type)

        # Memory-Check
        self._check_memory_usage()

        return resource_id

    def cleanup_by_type(self, resource_type: str) -> int:
        """
        Funktionsweise: Cleaned alle Ressourcen eines bestimmten Typs
        Parameter: resource_type
        Return: Anzahl gecleanter Ressourcen
        """
        resource_ids = list(self.type_registry.get(resource_type, set()))
        cleaned_count = 0

        for resource_id in resource_ids:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1

        self.logger.info(f"Cleaned {cleaned_count} resources of type {resource_type}")
        return cleaned_count

    def cleanup_by_age(self, max_age_seconds: float) -> int:
        """
        Funktionsweise: Cleaned alte Ressourcen basierend auf Alter
        Parameter: max_age_seconds
        Return: Anzahl gecleanter Ressourcen
        """
        current_time = time.time()
        old_resources = [
            resource_id for resource_id, info in self.resource_info.items()
            if current_time - info.creation_time > max_age_seconds
        ]

        cleaned_count = 0
        for resource_id in old_resources:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1

        self.logger.info(f"Cleaned {cleaned_count} old resources (>{max_age_seconds}s)")
        return cleaned_count

    def force_cleanup_all(self) -> int:
        """
        Funktionsweise: Emergency cleanup aller Ressourcen
        Return: Anzahl gecleanter Ressourcen
        """
        resource_ids = list(self.tracked_resources.keys())
        cleaned_count = 0

        for resource_id in resource_ids:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1

        self.logger.warning(f"Force cleanup: {cleaned_count} resources cleaned")
        return cleaned_count

    def get_memory_usage(self) -> Dict[str, int]:
        """
        Funktionsweise: Gibt Memory-Usage pro Resource-Type zurück
        Return: dict {resource_type: bytes}
        """
        usage = defaultdict(int)

        for resource_id, resource_ref in self.tracked_resources.items():
            resource = resource_ref()
            if resource is not None and resource_id in self.resource_info:
                info = self.resource_info[resource_id]
                usage[info.resource_type] += info.estimated_size

        return dict(usage)

    def get_resource_statistics(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt detaillierte Resource-Statistiken
        Return: dict mit Resource-Stats
        """
        stats = {
            'total_resources': len(self.tracked_resources),
            'alive_resources': sum(1 for ref in self.tracked_resources.values() if ref() is not None),
            'dead_references': sum(1 for ref in self.tracked_resources.values() if ref() is None),
            'total_memory_mb': sum(self.get_memory_usage().values()) / (1024 * 1024),
            'types': {},
            'oldest_resource_age': 0
        }

        # Per-Type Statistics
        for resource_type, resource_ids in self.type_registry.items():
            alive_count = sum(
                1 for rid in resource_ids
                if rid in self.tracked_resources and self.tracked_resources[rid]() is not None
            )
            total_size = sum(
                self.resource_info[rid].estimated_size
                for rid in resource_ids
                if rid in self.resource_info
            )

            stats['types'][resource_type] = {
                'count': alive_count,
                'total_size_mb': total_size / (1024 * 1024)
            }

        # Oldest Resource Age
        if self.resource_info:
            current_time = time.time()
            oldest_time = min(info.creation_time for info in self.resource_info.values())
            stats['oldest_resource_age'] = current_time - oldest_time

        return stats

    def _cleanup_resource(self, resource_id: str) -> bool:
        """
        Funktionsweise: Cleaned einzelne Ressource
        Parameter: resource_id
        Return: bool - Cleanup erfolgreich
        """
        if resource_id not in self.resource_info:
            return False

        info = self.resource_info[resource_id]

        # Cleanup-Funktion ausführen
        if info.cleanup_func:
            try:
                info.cleanup_func()
            except Exception as e:
                self.logger.warning(f"Cleanup function failed for {resource_id}: {e}")

        # Aus Registries entfernen
        self.tracked_resources.pop(resource_id, None)
        self.resource_info.pop(resource_id, None)
        self.type_registry[info.resource_type].discard(resource_id)

        self.logger.debug(f"Cleaned resource: {resource_id}")
        self.resource_cleaned.emit(resource_id, info.resource_type)

        return True

    def _on_resource_deleted(self, resource_id: str):
        """
        Funktionsweise: Callback für automatisches Cleanup bei Garbage Collection
        Parameter: resource_id
        """
        if resource_id in self.resource_info:
            self.logger.debug(f"Resource garbage collected: {resource_id}")
            self._cleanup_resource(resource_id)

    def _automatic_cleanup(self):
        """
        Funktionsweise: Automatisches Cleanup für Timer-basierte Wartung
        """
        # Dead references cleanup
        dead_refs = [
            resource_id for resource_id, ref in self.tracked_resources.items()
            if ref() is None
        ]

        for resource_id in dead_refs:
            self._cleanup_resource(resource_id)

        # Memory-Check
        self._check_memory_usage()

    def _check_memory_usage(self):
        """
        Funktionsweise: Prüft Memory-Usage und emittiert Warnings
        """
        total_memory_bytes = sum(self.get_memory_usage().values())
        total_memory_mb = total_memory_bytes / (1024 * 1024)

        if total_memory_mb > self.memory_threshold_mb:
            self.logger.warning(f"Memory usage: {total_memory_mb:.1f}MB > {self.memory_threshold_mb}MB")
            self.memory_warning.emit(int(total_memory_mb), self.memory_threshold_mb)

    def _estimate_resource_size(self, resource: Any) -> int:
        """
        Funktionsweise: Schätzt Ressourcen-Größe für Memory-Tracking
        Parameter: resource
        Return: Geschätzte Größe in Bytes
        """
        try:
            # NumPy Arrays
            if isinstance(resource, np.ndarray):
                return resource.nbytes

            # Qt Objects mit size hint
            if hasattr(resource, 'size') and hasattr(resource.size(), 'width'):
                size = resource.size()
                # Annahme: 4 bytes per pixel für RGBA
                return size.width() * size.height() * 4

            # Generic Python Objects
            if hasattr(resource, '__sizeof__'):
                return resource.__sizeof__()

            # Fallback für Listen/Tuples
            if isinstance(resource, (list, tuple)):
                return sum(self._estimate_resource_size(item) for item in resource[:100])  # Sample first 100

            # Default fallback
            return 1024  # 1KB default

        except Exception:
            return 1024  # Fallback bei Estimation-Fehlern

class DisplayUpdateManager(QObject):
    """
    Funktionsweise: Change-Detection für Display-Updates - INTEGRIERT
    Aufgabe: Prüft ob Display-Update wirklich nötig ist basierend auf Data-Hash und Display-Mode
    Features: Multi-Level-Hashing, Pending-Updates-Management, Performance-Optimierung
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
    Funktionsweise: Zentraler Communication-Hub für LOD-Status zwischen Tabs
    Aufgabe: Signal-basierte Koordination, Status-Cache, Dependency-Checking
    Implementation: Option B - Signal-basiert mit zentralem Status-Cache
    """
    
    # Outgoing Signals für UI/Navigation
    tab_status_updated = pyqtSignal(str, dict)      # (tab, complete_status_dict)
    dependencies_satisfied = pyqtSignal(str, int)   # (tab, lod_level) 
    all_tabs_status = pyqtSignal(dict)             # Complete overview
    auto_generation_ready = pyqtSignal(str, int)   # (tab, lod_level) für Auto-Tab-Generation
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Status-Cache für alle Tabs
        self.tab_statuses: Dict[str, TabStatus] = {}
        
        # LOD-Konfiguration
        self.lod_config: Optional[LODConfig] = None
        
        # Initialisiere alle bekannten Tabs mit Idle-Status
        for tab in DEPENDENCY_MATRIX.keys():
            self.tab_statuses[tab] = TabStatus(
                tab=tab, lod_level=0, lod_size=0, lod_status="idle"
            )
    
    def set_lod_config(self, minimal_map_size: int, target_map_size: int):
        """Setzt LOD-Konfiguration für alle Tabs"""
        self.lod_config = LODConfig(minimal_map_size, target_map_size)
        self.logger.info(f"LOD config set: {minimal_map_size} → {target_map_size}")
        
        # Aktualisiere alle Tab-Status mit neuen LOD-Sizes
        for tab_name in self.tab_statuses:
            status = self.tab_statuses[tab_name]
            if status.lod_level > 0:
                status.lod_size = self.lod_config.get_map_size_for_lod(status.lod_level)
        
        self.all_tabs_status.emit(self._get_all_statuses_dict())
    
    @pyqtSlot(str, int, int)
    def on_tab_lod_started(self, tab: str, lod_level: int, lod_size: int):
        """Slot: Tab hat LOD-Generation gestartet"""
        if tab not in self.tab_statuses:
            self.tab_statuses[tab] = TabStatus(tab=tab, lod_level=0, lod_size=0, lod_status="idle")
        
        status = self.tab_statuses[tab]
        status.lod_level = lod_level
        status.lod_size = lod_size
        status.lod_status = "pending"
        status.progress_percent = 0
        status.error_message = ""
        
        self.logger.debug(f"{tab} LOD {lod_level} started (size: {lod_size})")
        self._emit_status_update(tab)
    
    @pyqtSlot(str, int, int)
    def on_tab_lod_progress(self, tab: str, lod_level: int, progress_percent: int):
        """Slot: Tab LOD-Generation Progress Update"""
        if tab in self.tab_statuses:
            status = self.tab_statuses[tab]
            if status.lod_level == lod_level:
                status.progress_percent = progress_percent
                self._emit_status_update(tab)
    
    @pyqtSlot(str, int, bool, list)
    def on_tab_lod_completed(self, tab: str, lod_level: int, success: bool, data_keys: list):
        """Slot: Tab LOD-Generation abgeschlossen"""
        if tab not in self.tab_statuses:
            return
            
        status = self.tab_statuses[tab]
        status.lod_level = lod_level
        status.lod_status = "success" if success else "failure"
        status.progress_percent = 100 if success else 0
        
        if success:
            status.available_data_keys = data_keys
            status.error_message = ""
        
        self.logger.debug(f"{tab} LOD {lod_level} completed: {success}")
        self._emit_status_update(tab)
        
        # Prüfe Dependencies für andere Tabs
        if success:
            self._check_dependent_tabs(tab, lod_level)
    
    @pyqtSlot(str, int, str)
    def on_tab_lod_failed(self, tab: str, lod_level: int, error_message: str):
        """Slot: Tab LOD-Generation fehlgeschlagen"""
        if tab in self.tab_statuses:
            status = self.tab_statuses[tab]
            status.lod_level = lod_level
            status.lod_status = "failure"
            status.progress_percent = 0
            status.error_message = error_message
            
            self.logger.error(f"{tab} LOD {lod_level} failed: {error_message}")
            self._emit_status_update(tab)
    
    def _emit_status_update(self, tab: str):
        """Emittiert Status-Update für spezifischen Tab"""
        if tab in self.tab_statuses:
            status_dict = self._tab_status_to_dict(self.tab_statuses[tab])
            self.tab_status_updated.emit(tab, status_dict)
            self.all_tabs_status.emit(self._get_all_statuses_dict())
    
    def _check_dependent_tabs(self, completed_tab: str, completed_lod: int):
        """Prüft welche Tabs jetzt ihre Dependencies erfüllt haben"""
        for tab, dependencies in DEPENDENCY_MATRIX.items():
            if completed_tab in dependencies:
                if self.are_dependencies_satisfied(tab, completed_lod):
                    self.logger.debug(f"{tab} dependencies satisfied for LOD {completed_lod}")
                    self.dependencies_satisfied.emit(tab, completed_lod)
                    
                    # Auto-Generation-Signal für Tabs die automatisch starten sollen
                    auto_generation_tabs = ['geology', 'weather', 'water', 'settlement', 'biome']
                    if tab in auto_generation_tabs:
                        self.auto_generation_ready.emit(tab, completed_lod)
    
    def get_tab_status(self, tab: str) -> Optional[Dict[str, Any]]:
        """Gibt Status für spezifischen Tab zurück"""
        if tab in self.tab_statuses:
            return self._tab_status_to_dict(self.tab_statuses[tab])
        return None
    
    def get_lod_size_for_level(self, lod_level: int) -> int:
        """Gibt Map-Size für LOD-Level zurück"""
        if self.lod_config:
            return self.lod_config.get_map_size_for_lod(lod_level)
        return 64  # Fallback
    
    def are_dependencies_satisfied(self, tab: str, lod_level: int) -> bool:
        """Prüft ob alle Dependencies für Tab auf LOD-Level erfüllt sind"""
        if tab not in DEPENDENCY_MATRIX:
            return True
            
        dependencies = DEPENDENCY_MATRIX[tab]
        
        for dep_tab in dependencies:
            if dep_tab not in self.tab_statuses:
                return False
                
            dep_status = self.tab_statuses[dep_tab]
            
            # Dependency muss mindestens die gleiche LOD-Stufe erfolgreich abgeschlossen haben
            if dep_status.lod_level < lod_level or dep_status.lod_status != "success":
                return False
        
        return True

    def get_max_available_lod(self, tab: str) -> int:
        """Gibt höchste verfügbare LOD für Tab zurück basierend auf Dependencies"""
        if tab not in DEPENDENCY_MATRIX:
            return 1

        dependencies = DEPENDENCY_MATRIX[tab]
        if not dependencies:
            # Terrain hat keine Dependencies
            return self.lod_config.max_terrain_lod if self.lod_config else 7

        min_dep_lod = float('inf')
        for dep_tab in dependencies:
            if dep_tab in self.tab_statuses:
                dep_status = self.tab_statuses[dep_tab]
                if dep_status.lod_status == "success":
                    min_dep_lod = min(min_dep_lod, dep_status.lod_level)
                else:
                    return 0  # Dependency nicht erfüllt
            else:
                return 0  # Dependency nicht verfügbar

        return int(min_dep_lod) if min_dep_lod != float('inf') else 0

    def _get_all_statuses_dict(self) -> Dict[str, Dict[str, Any]]:
        """Gibt alle Tab-Status als Dictionary zurück"""
        return {
            tab: self._tab_status_to_dict(status)
            for tab, status in self.tab_statuses.items()
        }

    def _tab_status_to_dict(self, status: TabStatus) -> Dict[str, Any]:
        """Konvertiert TabStatus zu Dictionary"""
        return {
            "tab": status.tab,
            "lod_level": status.lod_level,
            "lod_size": status.lod_size,
            "lod_status": status.lod_status,
            "progress_percent": status.progress_percent,
            "error_message": status.error_message,
            "available_data_keys": status.available_data_keys.copy()
        }

# =============================================================================
# MAIN DATA LOD MANAGER
# =============================================================================

class DataLODManager(QObject):
    """
    Funktionsweise: Zentrale Klasse für Datenverwaltung - KOMPLETT INTEGRIERT
    Aufgabe: Speichert und verwaltet alle Arrays zwischen Tabs, LOD-basiertes Cache-Management
    Kommunikation: Signals über LODCommunicationHub, Memory-Management für große Arrays
    Integriert: ResourceTracker, DisplayUpdateManager, LODCommunicationHub
    """

    # Signals für Data-Updates (an LODCommunicationHub weitergeleitet)
    data_updated = pyqtSignal(str, str)  # (generator_type, data_key) - LEGACY
    cache_invalidated = pyqtSignal(str)  # (generator_type) - LEGACY

    # Neue LOD-basierte Signals
    lod_data_stored = pyqtSignal(str, int, list)  # (generator_type, lod_level, data_keys)

    def __init__(self, memory_threshold_mb: int = 500):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # INTEGRIERTE KOMPONENTEN
        self.lod_hub = LODCommunicationHub()
        self.resource_tracker = ResourceTracker(memory_threshold_mb)
        self.display_manager = DisplayUpdateManager()

        # Signal-Verbindungen zwischen integrierten Komponenten
        self.resource_tracker.memory_warning.connect(self._on_memory_warning)
        self.display_manager.update_performed.connect(self._on_display_update_performed)

        # Haupt-Datenstrukturen für alle Generator-Outputs (LOD-erweitert)
        self._terrain_data = {}  # {f"lod_{level}_{data_key}": data}
        self._geology_data = {}
        self._settlement_data = {}
        self._weather_data = {}
        self._water_data = {}
        self._biome_data = {}

        # Cache-Management (LOD-erweitert)
        self._cache_timestamps = {}  # {f"{generator}_{lod}_{key}": timestamp}
        self._parameter_hashes = {}  # {f"{generator}_{lod}": param_hash}

        # Current LOD-Levels für jeden Generator
        self._current_lods = {
            "terrain": 0, "geology": 0, "settlement": 0,
            "weather": 0, "water": 0, "biome": 0
        }

        # Memory-Management Timer für große Arrays
        self.memory_cleanup_timer = QTimer()
        self.memory_cleanup_timer.timeout.connect(self._periodic_memory_cleanup)
        self.memory_cleanup_timer.start(60000)  # 60 seconds

    # =============================================================================
    # LOD CONFIG AND CORE METHODS
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
    # TERRAIN DATA MANAGEMENT - LOD-ERWEITERT
    # =============================================================================

    @data_management_handler("terrain_data")
    def set_terrain_data_lod(self, data_key: str, data: np.ndarray, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Terrain-Arrays mit LOD-Level und Resource-Tracking
        Parameter: data_key ("heightmap", "slopemap", "shadowmap"), data, lod_level, parameters
        """
        lod_key = f"lod_{lod_level}_{data_key}"
        self._terrain_data[lod_key] = data
        self._current_lods["terrain"] = max(self._current_lods["terrain"], lod_level)

        # Resource-Tracking für große Arrays
        if isinstance(data, np.ndarray) and data.nbytes > 10_000_000:  # >10MB
            resource_id = self.resource_tracker.register_resource(
                data, f"terrain_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key}
            )

        self._update_cache_timestamp("terrain", lod_level, data_key, parameters)

        # Emit both legacy and new signals
        self.data_updated.emit("terrain", data_key)
        self.lod_data_stored.emit("terrain", lod_level, [data_key])

        self.logger.debug(f"Terrain data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")

    def set_terrain_data_complete_lod(self, terrain_data, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert komplettes TerrainData-Objekt mit LOD-Level und Resource-Tracking
        Parameter: terrain_data (TerrainData-Objekt), lod_level, parameters
        """
        # TerrainData-Objekt komplett speichern
        lod_key = f"lod_{lod_level}_terrain_data_object"
        self._terrain_data[lod_key] = terrain_data

        # Resource-Tracking für TerrainData-Objekt
        resource_id = self.resource_tracker.register_resource(
            terrain_data, "terrain_complete",
            metadata={"lod_level": lod_level}
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
        self._terrain_data[f"lod_{lod_level}_calculated_sun_angles"] = getattr(terrain_data,
                                                                               'calculated_sun_angles', [])

        data_keys.append("complete")
        self.lod_data_stored.emit("terrain", lod_level, data_keys)
        self.data_updated.emit("terrain", "complete")  # Legacy

        self.logger.debug(f"Complete terrain data updated for LOD {lod_level}, size: {terrain_data.actual_size}")

    def get_terrain_data_lod(self, data_key: str, lod_level: int = None):
        """
        Funktionsweise: Holt Terrain-Daten für spezifisches LOD-Level
        Parameter: data_key, lod_level (None = höchstes verfügbares LOD)
        Return: numpy array, TerrainData-Objekt oder None
        """
        if lod_level is None:
            lod_level = self._current_lods.get("terrain", 0)

        if data_key == "complete":
            lod_key = f"lod_{lod_level}_terrain_data_object"
            return self._terrain_data.get(lod_key)
        else:
            lod_key = f"lod_{lod_level}_{data_key}"
            return self._terrain_data.get(lod_key)

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

    def has_terrain_lod(self, lod_level: str) -> bool:
        """Legacy-Kompatibilität für String-LOD-Level"""
        # Konvertiere String-LOD zu numerisch
        lod_mapping = {"LOD64": 1, "LOD128": 2, "LOD256": 3, "FINAL": 4}
        numeric_lod = lod_mapping.get(lod_level, 0)

        current_lod = self._current_lods.get("terrain", 0)
        return current_lod >= numeric_lod

    # =============================================================================
    # GEOLOGY DATA MANAGEMENT - LOD-ERWEITERT
    # =============================================================================

    @data_management_handler("geology_data")
    def set_geology_data_lod(self, data_key: str, data: np.ndarray, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Geology-Generator Output mit LOD und Resource-Tracking
        Parameter: data_key ("rock_map", "hardness_map"), data, lod_level, parameters
        """
        lod_key = f"lod_{lod_level}_{data_key}"
        self._geology_data[lod_key] = data
        self._current_lods["geology"] = max(self._current_lods["geology"], lod_level)

        # Resource-Tracking für große Arrays
        if isinstance(data, np.ndarray) and data.nbytes > 10_000_000:  # >10MB
            self.resource_tracker.register_resource(
                data, f"geology_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key}
            )

        self._update_cache_timestamp("geology", lod_level, data_key, parameters)
        self.data_updated.emit("geology", data_key)
        self.lod_data_stored.emit("geology", lod_level, [data_key])

        self.logger.debug(f"Geology data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")

    def get_geology_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Geology-Daten für LOD-Level zurück"""
        if lod_level is None:
            lod_level = self._current_lods.get("geology", 0)

        lod_key = f"lod_{lod_level}_{data_key}"
        return self._geology_data.get(lod_key)

    def get_geology_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode für höchstes verfügbares LOD"""
        return self.get_geology_data_lod(data_key)

    # =============================================================================
    # WEATHER DATA MANAGEMENT - CFD-SPEZIAL
    # =============================================================================

    @data_management_handler("weather_data")
    def set_weather_data_lod(self, data_key: str, data: np.ndarray, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Weather-Generator Output (CFD-basiert) mit Resource-Tracking
        Parameter: data_key ("wind_map", "temp_map", "precip_map", "humid_map"), data, lod_level, parameters
        Besonderheit: CFD-Zellen halbieren sich pro LOD, Z-Achse bleibt konstant (3 Schichten)
        """
        lod_key = f"lod_{lod_level}_{data_key}"
        self._weather_data[lod_key] = data
        self._current_lods["weather"] = max(self._current_lods["weather"], lod_level)

        # Resource-Tracking für CFD-Daten (können groß sein)
        if isinstance(data, np.ndarray) and data.nbytes > 5_000_000:  # >5MB
            self.resource_tracker.register_resource(
                data, f"weather_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key, "cfd_data": True}
            )

        self._update_cache_timestamp("weather", lod_level, data_key, parameters)
        self.data_updated.emit("weather", data_key)
        self.lod_data_stored.emit("weather", lod_level, [data_key])

        # Spezielle Validierung für CFD-Daten
        if data.ndim == 3:
            self.logger.debug(
                f"Weather CFD data '{data_key}' updated for LOD {lod_level}, shape: {data.shape} (X,Y,Z)")
        else:
            self.logger.debug(f"Weather data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")

    def get_weather_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Weather-Daten für LOD-Level zurück"""
        if lod_level is None:
            lod_level = self._current_lods.get("weather", 0)

        lod_key = f"lod_{lod_level}_{data_key}"
        return self._weather_data.get(lod_key)

    def get_weather_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode für höchstes verfügbares LOD"""
        return self.get_weather_data_lod(data_key)

    # =============================================================================
    # WATER DATA MANAGEMENT - LOD-ERWEITERT
    # =============================================================================

    @data_management_handler("water_data")
    def set_water_data_lod(self, data_key: str, data: Any, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Water-Generator Output mit LOD und Resource-Tracking
        Parameter: data_key ("water_map", "flow_map", "erosion_map", etc.), data, lod_level, parameters
        Hinweis: Proportionalität mit map-size noch nicht final geprüft
        """
        lod_key = f"lod_{lod_level}_{data_key}"
        self._water_data[lod_key] = data
        self._current_lods["water"] = max(self._current_lods["water"], lod_level)

        # Resource-Tracking für große Arrays
        if isinstance(data, np.ndarray) and data.nbytes > 10_000_000:  # >10MB
            self.resource_tracker.register_resource(
                data, f"water_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key}
            )

        self._update_cache_timestamp("water", lod_level, data_key, parameters)
        self.data_updated.emit("water", data_key)
        self.lod_data_stored.emit("water", lod_level, [data_key])

        if isinstance(data, np.ndarray):
            self.logger.debug(f"Water data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")
        else:
            self.logger.debug(f"Water data '{data_key}' updated for LOD {lod_level}, value: {data}")

    def get_water_data_lod(self, data_key: str, lod_level: int = None) -> Any:
        """Gibt Water-Daten für LOD-Level zurück"""
        if lod_level is None:
            lod_level = self._current_lods.get("water", 0)

        lod_key = f"lod_{lod_level}_{data_key}"
        return self._water_data.get(lod_key)

    def get_water_data(self, data_key: str) -> Any:
        """Legacy-Methode für höchstes verfügbares LOD"""
        return self.get_water_data_lod(data_key)

    # =============================================================================
    # BIOME DATA MANAGEMENT - SUPER-SAMPLING
    # =============================================================================

    @data_management_handler("biome_data")
    def set_biome_data_lod(self, data_key: str, data: np.ndarray, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Biome-Generator Output mit LOD und Resource-Tracking
        Parameter: data_key ("biome_map", "biome_map_super", "super_biome_mask"), data, lod_level, parameters
        Besonderheit: Super-sampling (4px pro 1px) für einige Outputs
        """
        lod_key = f"lod_{lod_level}_{data_key}"
        self._biome_data[lod_key] = data
        self._current_lods["biome"] = max(self._current_lods["biome"], lod_level)

        # Resource-Tracking für Super-sampled Daten (können sehr groß sein)
        if isinstance(data, np.ndarray):
            # Super-sampled Daten haben 4x die Pixel-Anzahl
            threshold = 5_000_000 if "super" not in data_key else 20_000_000  # 20MB für super-sampled
            if data.nbytes > threshold:
                self.resource_tracker.register_resource(
                    data, f"biome_{data_key}",
                    metadata={"lod_level": lod_level, "data_key": data_key, "super_sampled": "super" in data_key}
                )

        self._update_cache_timestamp("biome", lod_level, data_key, parameters)
        self.data_updated.emit("biome", data_key)
        self.lod_data_stored.emit("biome", lod_level, [data_key])

        # Spezielle Validierung für Super-sampled Daten
        if "super" in data_key:
            self.logger.debug(
                f"Biome super-sampled data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")
        else:
            self.logger.debug(f"Biome data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")

    def get_biome_data_lod(self, data_key: str, lod_level: int = None) -> Optional[np.ndarray]:
        """Gibt Biome-Daten für LOD-Level zurück"""
        if lod_level is None:
            lod_level = self._current_lods.get("biome", 0)

        lod_key = f"lod_{lod_level}_{data_key}"
        return self._biome_data.get(lod_key)

    def get_biome_data(self, data_key: str) -> Optional[np.ndarray]:
        """Legacy-Methode für höchstes verfügbares LOD"""
        return self.get_biome_data_lod(data_key)

    # =============================================================================
    # SETTLEMENT DATA MANAGEMENT - NODE-BASIERT
    # =============================================================================

    @data_management_handler("settlement_data")
    def set_settlement_data_lod(self, data_key: str, data: Any, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Settlement-Generator Output mit LOD und Resource-Tracking
        Parameter: data_key ("settlement_list", "landmark_list", "plot_map", "civ_map"), data, lod_level, parameters
        Besonderheit: Skaliert mit map-size + Node-Anzahl (Implementierung noch zu klären)
        """
        lod_key = f"lod_{lod_level}_{data_key}"
        self._settlement_data[lod_key] = data
        self._current_lods["settlement"] = max(self._current_lods["settlement"], lod_level)

        # Resource-Tracking für große Settlement-Daten
        if isinstance(data, np.ndarray) and data.nbytes > 10_000_000:  # >10MB
            self.resource_tracker.register_resource(
                data, f"settlement_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key}
            )
        elif isinstance(data, list) and len(data) > 1000:  # Große Listen
            self.resource_tracker.register_resource(
                data, f"settlement_{data_key}",
                metadata={"lod_level": lod_level, "data_key": data_key, "list_size": len(data)}
            )

        self._update_cache_timestamp("settlement", lod_level, data_key, parameters)
        self.data_updated.emit("settlement", data_key)
        self.lod_data_stored.emit("settlement", lod_level, [data_key])

        if isinstance(data, np.ndarray):
            self.logger.debug(f"Settlement data '{data_key}' updated for LOD {lod_level}, shape: {data.shape}")
        elif isinstance(data, list):
            self.logger.debug(f"Settlement data '{data_key}' updated for LOD {lod_level}, count: {len(data)}")
        else:
            self.logger.debug(f"Settlement data '{data_key}' updated for LOD {lod_level}, type: {type(data)}")

    def get_settlement_data_lod(self, data_key: str, lod_level: int = None) -> Any:
        """Gibt Settlement-Daten für LOD-Level zurück"""
        if lod_level is None:
            lod_level = self._current_lods.get("settlement", 0)

        lod_key = f"lod_{lod_level}_{data_key}"
        return self._settlement_data.get(lod_key)

    def get_settlement_data(self, data_key: str) -> Any:
        """Legacy-Methode für höchstes verfügbares LOD"""
        return self.get_settlement_data_lod(data_key)

    # =============================================================================
    # DEPENDENCY CHECKING - VEREINFACHT
    # =============================================================================

    @memory_critical_handler("dependency_check")
    def check_dependencies_lod(self, generator_type: str, lod_level: int) -> Tuple[bool, List[str]]:
        """
        Funktionsweise: Prüft ob alle Dependencies für Generator auf LOD-Level verfügbar sind
        Parameter: generator_type, lod_level
        Return: (alle_verfügbar: bool, fehlende_dependencies: List[str])
        """
        if generator_type not in DEPENDENCY_MATRIX:
            return True, []

        dependencies = DEPENDENCY_MATRIX[generator_type]
        missing = []

        for dep_generator in dependencies:
            dep_lod = self._current_lods.get(dep_generator, 0)

            # Dependency muss mindestens das gleiche LOD-Level haben
            if dep_lod < lod_level:
                missing.append(f"{dep_generator} (has LOD {dep_lod}, need LOD {lod_level})")

        is_complete = len(missing) == 0
        return is_complete, missing

    def check_dependencies(self, generator_type: str, required_dependencies: List[str]) -> Tuple[bool, List[str]]:
        """
        Legacy-Methode: Kompatibilität mit altem Dependency-System
        Verwendet aktuelles LOD-Level des Generators
        """
        current_lod = self._current_lods.get(generator_type, 1)
        return self.check_dependencies_lod(generator_type, current_lod)

    # =============================================================================
    # CACHE MANAGEMENT - LOD-ERWEITERT
    # =============================================================================

    def _update_cache_timestamp(self, generator_type: str, lod_level: int, data_key: str,
                                parameters: Dict[str, Any]):
        """
        Funktionsweise: Aktualisiert Cache-Timestamp und Parameter-Hash für LOD-spezifische Invalidation
        Parameter: generator_type, lod_level, data_key, parameters
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
        Funktionsweise: Prüft ob Cache für LOD-spezifische Parameter noch valid ist
        Parameter: generator_type, lod_level, data_key, parameters
        Return: bool (Cache ist valid)
        """
        import hashlib

        cache_key = f"{generator_type}_lod_{lod_level}"

        if cache_key not in self._parameter_hashes:
            return False

        current_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()
        return self._parameter_hashes[cache_key] == current_hash

    def invalidate_cache_lod(self, generator_type: str, lod_level: int = None):
        """
        Funktionsweise: Invalidiert Cache für spezifisches LOD-Level oder alle LODs
        Parameter: generator_type, lod_level (None = alle LODs)
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

        # Daten löschen wenn alle LODs invalidiert werden
        if lod_level is None:
            # Resource-Tracking-Cleanup vor Daten-Löschung
            self.resource_tracker.cleanup_by_type(generator_type)

            if generator_type == "terrain":
                self._terrain_data.clear()
                self._current_lods["terrain"] = 0
            elif generator_type == "geology":
                self._geology_data.clear()
                self._current_lods["geology"] = 0
            elif generator_type == "settlement":
                self._settlement_data.clear()
                self._current_lods["settlement"] = 0
            elif generator_type == "weather":
                self._weather_data.clear()
                self._current_lods["weather"] = 0
            elif generator_type == "water":
                self._water_data.clear()
                self._current_lods["water"] = 0
            elif generator_type == "biome":
                self._biome_data.clear()
                self._current_lods["biome"] = 0

        self.cache_invalidated.emit(generator_type)
        self.logger.info(f"Cache invalidated for {generator_type}" +
                         (f" LOD {lod_level}" if lod_level is not None else " (all LODs)"))

    def invalidate_cache(self, generator_type: str):
        """Legacy-Methode: Invalidiert alle LODs"""
        self.invalidate_cache_lod(generator_type, None)

    # =============================================================================
    # UTILITY METHODS - LOD-ERWEITERT & RESOURCE-INTEGRIERT
    # =============================================================================

    def has_data_lod(self, data_key: str, lod_level: int) -> bool:
        """
        Funktionsweise: Prüft ob spezifischer Daten-Key für LOD-Level verfügbar ist
        Parameter: data_key, lod_level
        Return: bool - Daten verfügbar
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
        # Finde höchstes LOD für diesen data_key
        for lod_level in range(10, 0, -1):  # Prüfe LOD 10 bis 1
            if self.has_data_lod(data_key, lod_level):
                return True
        return False

    def get_memory_usage_by_lod(self) -> Dict[str, Dict[str, float]]:
        """
        Funktionsweise: Berechnet Memory-Usage aufgeschlüsselt nach Generator und LOD - INTEGRIERT
        Return: dict mit Memory-Usage in MB für jeden Generator pro LOD
        """

        def calculate_lod_memory(data_dict):
            lod_memory = {}
            for key, data in data_dict.items():
                if key.startswith("lod_") and isinstance(data, np.ndarray):
                    # Extrahiere LOD-Level aus Key
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
        Legacy-Methode: Berechnet Memory-Usage aller gespeicherten Arrays - RESOURCE-TRACKER-INTEGRIERT
        Return: dict mit Memory-Usage in MB für jeden Generator (alle LODs)
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

    def clear_all_data(self):
        """
        Funktionsweise: Löscht alle Daten und Cache - ERWEITERT mit Resource-Management
        Erweitert: Resettet auch LOD-Levels und LOD-Hub-Status, integriertes Resource-Cleanup
        """
        # Resource-Cleanup ZUERST
        self.resource_tracker.force_cleanup_all()
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
        for tab in self.lod_hub.tab_statuses:
            status = self.lod_hub.tab_statuses[tab]
            status.lod_level = 0
            status.lod_size = 0
            status.lod_status = "idle"
            status.progress_percent = 0
            status.error_message = ""
            status.available_data_keys.clear()

        # Emit reset signals
        self.lod_hub.all_tabs_status.emit(self.lod_hub._get_all_statuses_dict())

        # Force Garbage Collection nach großem Cleanup
        import gc
        gc.collect()

        self.logger.info("All data, LOD status, and resources cleared")

    def export_data_summary_lod(self) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt LOD-erweiterte Zusammenfassung - RESOURCE-INTEGRIERT
        Return: dict mit Summary aller Generator-Outputs pro LOD + Resource-Stats
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
            "lod_hub_status": self.lod_hub._get_all_statuses_dict(),

            # INTEGRIERTE RESOURCE-STATISTICS
            "resource_statistics": self.resource_tracker.get_resource_statistics(),
            "display_cache_statistics": self.display_manager.get_cache_statistics()
        }

        # LOD-Konfiguration hinzufügen
        if self.lod_hub.lod_config:
            summary["lod_config"] = {
                "minimal_map_size": self.lod_hub.lod_config.minimal_map_size,
                "target_map_size": self.lod_hub.lod_config.target_map_size,
                "max_terrain_lod": self.lod_hub.lod_config.max_terrain_lod
            }

        return summary

    def export_data_summary(self) -> Dict[str, Any]:
        """Legacy-Methode: Ruft erweiterte LOD-Summary auf"""
        return self.export_data_summary_lod()

    # =============================================================================
    # INTEGRATED RESOURCE MANAGEMENT METHODS
    # =============================================================================

    def cleanup_old_lod_resources(self, max_age_hours: float = 2.0):
        """
        Funktionsweise: Cleaned alte LOD-Ressourcen für Memory-Management
        Parameter: max_age_hours - Maximales Alter in Stunden
        """
        max_age_seconds = max_age_hours * 3600
        cleaned_count = self.resource_tracker.cleanup_by_age(max_age_seconds)

        if cleaned_count > 0:
            self.logger.info(f"Cleaned {cleaned_count} old LOD resources (>{max_age_hours}h)")

        return cleaned_count

    def cleanup_large_resources(self, size_threshold_mb: float = 50.0):
        """
        Funktionsweise: Cleaned große Ressourcen über Threshold für Memory-Management
        Parameter: size_threshold_mb - Größen-Threshold in MB
        """
        cleaned_count = self.resource_tracker.cleanup_by_size_threshold(size_threshold_mb)

        if cleaned_count > 0:
            self.logger.info(f"Cleaned {cleaned_count} large resources (>{size_threshold_mb}MB)")

        return cleaned_count

    def optimize_display_cache(self):
        """
        Funktionsweise: Optimiert Display-Cache für bessere Performance
        """
        # Entferne alte Display-Cache-Einträge
        current_time = time.time()
        old_displays = [
            display_id for display_id, last_update in self.display_manager.last_update_times.items()
            if current_time - last_update > 1800  # 30 minutes
        ]

        for display_id in old_displays:
            self.display_manager.clear_display_cache(display_id)

        if old_displays:
            self.logger.debug(f"Cleaned {len(old_displays)} old display cache entries")

    def get_integrated_statistics(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt alle integrierten Statistics für Monitoring
        Return: Comprehensive Stats dict
        """
        return {
            "data_manager": {
                "current_lods": self._current_lods,
                "total_data_objects": sum(len(d) for d in [
                    self._terrain_data, self._geology_data, self._settlement_data,
                    self._weather_data, self._water_data, self._biome_data
                ]),
                "cache_entries": len(self._cache_timestamps),
                "memory_usage_mb": self.get_memory_usage()
            },
            "resource_tracker": self.resource_tracker.get_resource_statistics(),
            "display_manager": self.display_manager.get_cache_statistics(),
            "lod_hub": {
                "active_tabs": len([s for s in self.lod_hub.tab_statuses.values() if s.lod_status != "idle"]),
                "total_tabs": len(self.lod_hub.tab_statuses),
                "lod_config_set": self.lod_hub.lod_config is not None
            }
        }

    # =============================================================================
    # LOD-HUB INTEGRATION METHODS
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
        Parameter: tab_name, tab_widget (BaseMapTab instance)
        Aufgabe: Automatische Signal-Verbindung für LOD-Kommunikation
        """
        try:
            # Tab → Hub Signal-Verbindungen
            if hasattr(tab_widget, 'lod_started'):
                tab_widget.lod_started.connect(self.lod_hub.on_tab_lod_started)
            if hasattr(tab_widget, 'lod_progress'):
                tab_widget.lod_progress.connect(self.lod_hub.on_tab_lod_progress)
            if hasattr(tab_widget, 'lod_completed'):
                tab_widget.lod_completed.connect(self.lod_hub.on_tab_lod_completed)
            if hasattr(tab_widget, 'lod_failed'):
                tab_widget.lod_failed.connect(self.lod_hub.on_tab_lod_failed)

            # Hub → Tab Signal-Verbindungen
            if hasattr(tab_widget, 'on_dependencies_satisfied'):
                self.lod_hub.dependencies_satisfied.connect(tab_widget.on_dependencies_satisfied)
            if hasattr(tab_widget, 'on_auto_generation_ready'):
                self.lod_hub.auto_generation_ready.connect(tab_widget.on_auto_generation_ready)

            self.logger.debug(f"Tab '{tab_name}' connected to LOD communication hub")

        except Exception as e:
            self.logger.error(f"Failed to connect tab '{tab_name}' to LOD hub: {e}")

    # =============================================================================
    # PRIVATE INTEGRATED METHODS (CALLBACKS)
    # =============================================================================

    def _on_memory_warning(self, current_mb: int, threshold_mb: int):
        """Callback für Memory-Warnings vom Resource-Tracker"""
        self.logger.warning(f"Memory warning triggered: {current_mb}MB > {threshold_mb}MB")

        # Automatisches Cleanup bei Memory-Warnings
        self.cleanup_old_lod_resources(max_age_hours=1.0)  # Aggressive cleanup
        self.optimize_display_cache()

    def _on_display_update_performed(self, display_id: str, layer_type: str, hash_time: float):
        """Callback für Display-Updates vom Display-Manager"""
        if hash_time > 0.1:  # Slow hash calculation
            self.logger.debug(f"Slow display hash calculation for {display_id}: {hash_time:.3f}s")

    def _periodic_memory_cleanup(self):
        """Periodisches Memory-Cleanup für große Arrays"""
        try:
            # Statistiken sammeln
            stats = self.get_integrated_statistics()
            total_memory = stats["data_manager"]["memory_usage_mb"]

            # Cleanup-Schwellwerte
            if sum(total_memory.values()) > 1000:  # >1GB total
                self.cleanup_old_lod_resources(max_age_hours=1.5)

            if sum(total_memory.values()) > 2000:  # >2GB total - aggressive cleanup
                self.cleanup_large_resources(size_threshold_mb=100.0)

            # Display-Cache-Optimierung
            self.optimize_display_cache()

            # Force Garbage Collection bei hoher Memory-Usage
            if sum(total_memory.values()) > 1500:  # >1.5GB
                import gc
                gc.collect()

        except Exception as e:
            self.logger.error(f"Periodic memory cleanup failed: {e}")

    # =============================================================================
    # LEGACY COMPATIBILITY LAYER
    # =============================================================================

    def get_terrain_dependencies(self) -> List[str]:
        """Legacy: Gibt verfügbare Terrain-Outputs zurück"""
        current_lod = self._current_lods.get("terrain", 0)
        if current_lod == 0:
            return []

        deps = []
        for key in self._terrain_data:
            if key.startswith(f"lod_{current_lod}_") and not key.endswith("_terrain_data_object"):
                clean_key = key.replace(f"lod_{current_lod}_", "")
                if clean_key not in ["actual_size", "calculated_sun_angles"]:
                    deps.append(clean_key)

        if f"lod_{current_lod}_terrain_data_object" in self._terrain_data:
            deps.append("terrain_complete")
            deps.append(f"terrain_LOD{current_lod}")

        return deps

    def get_geology_dependencies(self) -> List[str]:
        """Legacy: Gibt verfügbare Geology-Outputs zurück"""
        current_lod = self._current_lods.get("geology", 0)
        if current_lod == 0:
            return []

        deps = []
        for key in self._geology_data:
            if key.startswith(f"lod_{current_lod}_"):
                clean_key = key.replace(f"lod_{current_lod}_", "")
                deps.append(clean_key)
        return deps

    def get_settlement_dependencies(self) -> List[str]:
        """Legacy: Gibt verfügbare Settlement-Outputs zurück"""
        current_lod = self._current_lods.get("settlement", 0)
        if current_lod == 0:
            return []

        deps = []
        for key in self._settlement_data:
            if key.startswith(f"lod_{current_lod}_"):
                clean_key = key.replace(f"lod_{current_lod}_", "")
                deps.append(clean_key)
        return deps

    def get_weather_dependencies(self) -> List[str]:
        """Legacy: Gibt verfügbare Weather-Outputs zurück"""
        current_lod = self._current_lods.get("weather", 0)
        if current_lod == 0:
            return []

        deps = []
        for key in self._weather_data:
            if key.startswith(f"lod_{current_lod}_"):
                clean_key = key.replace(f"lod_{current_lod}_", "")
                deps.append(clean_key)
        return deps

    def get_water_dependencies(self) -> List[str]:
        """Legacy: Gibt verfügbare Water-Outputs zurück"""
        current_lod = self._current_lods.get("water", 0)
        if current_lod == 0:
            return []

        deps = []
        for key in self._water_data:
            if key.startswith(f"lod_{current_lod}_"):
                clean_key = key.replace(f"lod_{current_lod}_", "")
                deps.append(clean_key)
        return deps

    def get_biome_dependencies(self) -> List[str]:
        """Legacy: Gibt verfügbare Biome-Outputs zurück"""
        current_lod = self._current_lods.get("biome", 0)
        if current_lod == 0:
            return []

        deps = []
        for key in self._biome_data:
            if key.startswith(f"lod_{current_lod}_"):
                clean_key = key.replace(f"lod_{current_lod}_", "")
                deps.append(clean_key)
        return deps

    def cleanup_resources(self):
        """
        Funktionsweise: Complete Cleanup für App-Shutdown - INTEGRIERT
        Aufgabe: Systematisches Cleanup aller integrierten Komponenten
        """
        self.logger.info("Starting comprehensive DataLODManager cleanup")

        # Timer stoppen
        if hasattr(self, 'memory_cleanup_timer'):
            self.memory_cleanup_timer.stop()

        # Resource-Cleanup
        self.resource_tracker.force_cleanup_all()

        # Display-Cache-Cleanup
        self.display_manager.clear_all_cache()

        # ResourceTracker-Timer stoppen
        if hasattr(self.resource_tracker, 'cleanup_timer'):
            self.resource_tracker.cleanup_timer.stop()

        # Alle Daten löschen
        self.clear_all_data()

        # Force Garbage Collection
        import gc
        gc.collect()

        self.logger.info("DataLODManager cleanup completed")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_integrated_data_lod_manager(memory_threshold_mb: int = 500) -> DataLODManager:
    """
    Funktionsweise: Factory für DataLODManager mit konfigurierten Komponenten
    Parameter: memory_threshold_mb
    Return: Vollständig konfigurierter DataLODManager
    """
    manager = DataLODManager(memory_threshold_mb)

    # LOD-Config mit Standard-Werten setzen falls nicht anders spezifiziert
    manager.set_lod_config(minimal_map_size=32, target_map_size=512)

    return manager