"""
Path: gui/managers/resource_manager.py

Funktionsweise: Zentrale Verwaltung aller Ressourcen mit automatischem Cleanup und Memory-Leak-Prevention
Aufgabe: Verfolgt GPU-Textures, 3D-Objekte, große Arrays und ermöglicht systematisches Resource-Management
Features: WeakReference-Tracking, Age-based Cleanup, Resource-Type-Management, Memory-Usage-Monitoring
"""

import weakref
import time
import hashlib
import logging
from typing import Dict, Set, Optional, Any, Callable, List
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal


@dataclass
class ResourceInfo:
    """Information über eine tracked Resource"""
    resource_id: str
    resource_type: str
    creation_time: float
    cleanup_func: Optional[Callable]
    estimated_size: int
    metadata: Dict[str, Any]


class ResourceTracker(QObject):
    """
    Funktionsweise: Systematisches Resource-Management für Memory-Leak-Prevention
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

    def cleanup_by_size_threshold(self, size_threshold_mb: float) -> int:
        """
        Funktionsweise: Cleaned große Ressourcen über Größen-Threshold
        Parameter: size_threshold_mb
        Return: Anzahl gecleanter Ressourcen
        """
        size_threshold_bytes = size_threshold_mb * 1024 * 1024
        large_resources = [
            resource_id for resource_id, info in self.resource_info.items()
            if info.estimated_size > size_threshold_bytes
        ]

        cleaned_count = 0
        for resource_id in large_resources:
            if self._cleanup_resource(resource_id):
                cleaned_count += 1

        self.logger.info(f"Cleaned {cleaned_count} large resources (>{size_threshold_mb}MB)")
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
    Funktionsweise: Change-Detection für Display-Updates um unnötige Re-Renderings zu vermeiden
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