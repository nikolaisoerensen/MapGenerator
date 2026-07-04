"""
Path: gui/managers/statistics_manager.py

Funktionsweise: Effizientes Caching und Sharing von Statistics zwischen Tabs um redundante Berechnungen zu vermeiden
Aufgabe: Zentrale Statistics-Berechnung, Cross-Tab-Sharing, Performance-optimiertes Caching
Features: Hash-basiertes Caching, Automatic Invalidation, Statistics-Pipeline, Cross-Tab-Communication
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer


@dataclass
class CacheEntry:
    """Cache-Eintrag für Statistics"""
    statistics: Dict[str, Any]
    data_hash: str
    creation_time: float
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)


class StatisticsCache(QObject):
    """
    Funktionsweise: Hash-basiertes Caching-System für Statistics
    Aufgabe: Vermeidet redundante Berechnungen durch intelligentes Caching
    Features: Data-Hash-Validation, Age-based Expiration, Access-Counting
    """

    # Signals für Cache-Management
    cache_hit = pyqtSignal(str, str)  # (cache_key, stat_type)
    cache_miss = pyqtSignal(str, str)  # (cache_key, stat_type)
    cache_expired = pyqtSignal(str)  # (cache_key)

    def __init__(self, max_age_seconds: float = 300, max_entries: int = 100):
        super().__init__()

        self.cache = {}  # {cache_key: CacheEntry}
        self.max_age = max_age_seconds  # 5 Minutes default
        self.max_entries = max_entries

        # Cleanup timer
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_expired)
        self.cleanup_timer.start(60000)  # 1 minute

        self.logger = logging.getLogger(__name__)

    def get_cached_statistics(self, data: Any, stat_type: str, **kwargs) -> Dict[str, Any]:
        """
        Funktionsweise: Holt cached statistics oder berechnet neu falls nötig
        Parameter: data, stat_type, kwargs (additional parameters)
        Return: Statistics dict
        """
        cache_key = self._generate_cache_key(data, stat_type, kwargs)
        data_hash = self._calculate_data_hash(data)

        # Cache-Hit prüfen
        if cache_key in self.cache:
            entry = self.cache[cache_key]

            # Validierung: Hash und Age
            if (entry.data_hash == data_hash and
                    not self._is_expired(entry) and
                    entry.statistics):

                # Access-Tracking
                entry.access_count += 1
                entry.last_access_time = time.time()

                self.cache_hit.emit(cache_key, stat_type)
                self.logger.debug(f"Cache hit for {stat_type}: {cache_key}")
                return entry.statistics

            else:
                # Cache-Eintrag ist invalid/expired
                del self.cache[cache_key]
                self.cache_expired.emit(cache_key)

        # Cache-Miss: Neu berechnen
        self.cache_miss.emit(cache_key, stat_type)
        self.logger.debug(f"Cache miss for {stat_type}: {cache_key}")

        statistics = self._calculate_statistics(data, stat_type, **kwargs)
        self._store_in_cache(cache_key, statistics, data_hash)

        return statistics

    def invalidate_cache(self, pattern: Optional[str] = None, stat_type: Optional[str] = None):
        """
        Funktionsweise: Invalidiert Cache-Einträge basierend auf Pattern oder Type
        Parameter: pattern (optional), stat_type (optional)
        """
        if pattern is None and stat_type is None:
            # Kompletten Cache leeren
            self.cache.clear()
            self.logger.info("Complete cache invalidated")
            return

        # Pattern-basierte Invalidation
        keys_to_remove = []
        for cache_key in self.cache:
            if pattern and pattern in cache_key:
                keys_to_remove.append(cache_key)
            elif stat_type and stat_type in cache_key:
                keys_to_remove.append(cache_key)

        for key in keys_to_remove:
            del self.cache[key]

        self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt Cache-Performance-Statistiken
        Return: Cache-Stats dict
        """
        current_time = time.time()

        total_access_count = sum(entry.access_count for entry in self.cache.values())
        total_entries = len(self.cache)

        if self.cache:
            avg_age = current_time - sum(entry.creation_time for entry in self.cache.values()) / total_entries
            most_accessed = max(self.cache.values(), key=lambda e: e.access_count)
        else:
            avg_age = 0
            most_accessed = None

        return {
            'total_entries': total_entries,
            'total_access_count': total_access_count,
            'average_age_seconds': avg_age,
            'most_accessed_key': getattr(most_accessed, 'statistics', {}).get('stat_type', 'none'),
            'memory_usage_mb': self._estimate_cache_size() / (1024 * 1024)
        }

    def _generate_cache_key(self, data: Any, stat_type: str, kwargs: Dict[str, Any]) -> str:
        """
        Funktionsweise: Generiert eindeutigen Cache-Key
        Parameter: data, stat_type, kwargs
        Return: Cache-Key string
        """
        # Data-Shape als Teil des Keys
        data_info = ""
        if isinstance(data, np.ndarray):
            data_info = f"shape_{data.shape}_dtype_{data.dtype}"
        elif hasattr(data, 'shape'):
            data_info = f"shape_{data.shape}"
        else:
            data_info = f"type_{type(data).__name__}"

        # Kwargs sortiert für konsistente Keys
        kwargs_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))

        return f"{stat_type}_{data_info}_{kwargs_str}"

    def _calculate_data_hash(self, data: Any) -> str:
        """
        Funktionsweise: Berechnet Hash für Data-Change-Detection
        Parameter: data
        Return: Hash string
        """
        try:
            if isinstance(data, np.ndarray):
                # Sample-based hashing für große Arrays
                if data.size > 100_000:
                    step = max(1, data.size // 1000)
                    sample = data.flat[::step]
                    return hashlib.md5(sample.tobytes()).hexdigest()
                else:
                    return hashlib.md5(data.tobytes()).hexdigest()
            else:
                return hashlib.md5(str(data).encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(id(data)).encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """
        Funktionsweise: Prüft ob Cache-Entry expired ist
        Parameter: entry
        Return: bool - ist expired
        """
        return time.time() - entry.creation_time > self.max_age

    def _store_in_cache(self, cache_key: str, statistics: Dict[str, Any], data_hash: str):
        """
        Funktionsweise: Speichert Statistics in Cache mit LRU-Management
        Parameter: cache_key, statistics, data_hash
        """
        # LRU-Eviction wenn Cache zu groß
        if len(self.cache) >= self.max_entries:
            # Entferne ältesten/am wenigsten genutzten Eintrag
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: (self.cache[k].last_access_time, self.cache[k].access_count)
            )
            del self.cache[oldest_key]

        # Neuen Entry erstellen
        entry = CacheEntry(
            statistics=statistics,
            data_hash=data_hash,
            creation_time=time.time()
        )

        self.cache[cache_key] = entry

    def _cleanup_expired(self):
        """
        Funktionsweise: Automatisches Cleanup für expired entries
        """
        expired_keys = [
            cache_key for cache_key, entry in self.cache.items()
            if self._is_expired(entry)
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")

    def _estimate_cache_size(self) -> int:
        """
        Funktionsweise: Schätzt Cache-Memory-Usage
        Return: Estimated size in bytes
        """
        estimated_size = 0
        for entry in self.cache.values():
            # Statistics dict size estimation
            estimated_size += len(str(entry.statistics)) * 4  # rough estimate
            estimated_size += 200  # overhead per entry
        return estimated_size

    def _calculate_statistics(self, data: Any, stat_type: str, **kwargs) -> Dict[str, Any]:
        """
        Funktionsweise: Berechnet Statistics basierend auf Typ
        Parameter: data, stat_type, kwargs
        Return: Statistics dict
        """
        try:
            if stat_type == "terrain_height":
                return self._calculate_terrain_height_stats(data, **kwargs)
            elif stat_type == "terrain_slope":
                return self._calculate_terrain_slope_stats(data, **kwargs)
            elif stat_type == "biome_distribution":
                return self._calculate_biome_distribution_stats(data, **kwargs)
            elif stat_type == "geology_rock_types":
                return self._calculate_geology_rock_stats(data, **kwargs)
            elif stat_type == "water_flow":
                return self._calculate_water_flow_stats(data, **kwargs)
            elif stat_type == "weather_climate":
                return self._calculate_weather_climate_stats(data, **kwargs)
            else:
                return self._calculate_generic_stats(data, **kwargs)

        except Exception as e:
            self.logger.error(f"Statistics calculation failed for {stat_type}: {e}")
            return {"error": str(e), "stat_type": stat_type}

    def _calculate_statistics(self, data: Any, stat_type: str, **kwargs) -> Dict[str, Any]:
        """
        Funktionsweise: Berechnet Statistics basierend auf Typ
        Parameter: data, stat_type, kwargs
        Return: Statistics dict
        """
        try:
            if stat_type == "terrain_height":
                return self._calculate_terrain_height_stats(data, **kwargs)
            elif stat_type == "terrain_slope":
                return self._calculate_terrain_slope_stats(data, **kwargs)
            elif stat_type == "biome_distribution":
                return self._calculate_biome_distribution_stats(data, **kwargs)
            elif stat_type == "geology_rock_types":
                return self._calculate_geology_rock_stats(data, **kwargs)
            elif stat_type == "water_flow":
                return self._calculate_water_flow_stats(data, **kwargs)
            elif stat_type == "weather_climate":
                return self._calculate_weather_climate_stats(data, **kwargs)
            else:
                return self._calculate_generic_stats(data, **kwargs)

        except Exception as e:
            self.logger.error(f"Statistics calculation failed for {stat_type}: {e}")
            return {"error": str(e), "stat_type": stat_type}

    def _calculate_terrain_height_stats(self, heightmap: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Terrain Heightmap Statistics"""
        return {
            "stat_type": "terrain_height",
            "min": float(np.min(heightmap)),
            "max": float(np.max(heightmap)),
            "mean": float(np.mean(heightmap)),
            "std": float(np.std(heightmap)),
            "median": float(np.median(heightmap)),
            "range": float(np.max(heightmap) - np.min(heightmap)),
            "shape": heightmap.shape
        }

    def _calculate_terrain_slope_stats(self, slopemap: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Terrain Slope Statistics"""
        if len(slopemap.shape) == 3 and slopemap.shape[2] >= 2:
            slope_magnitude = np.sqrt(slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2)
        else:
            slope_magnitude = slopemap

        slope_degrees = np.arctan(slope_magnitude) * 180 / np.pi

        return {
            "stat_type": "terrain_slope",
            "max_slope_degrees": float(np.max(slope_degrees)),
            "mean_slope_degrees": float(np.mean(slope_degrees)),
            "steep_areas_percent": float(np.sum(slope_degrees > 30) / slope_degrees.size * 100),
            "flat_areas_percent": float(np.sum(slope_degrees < 5) / slope_degrees.size * 100)
        }

    def _calculate_biome_distribution_stats(self, biome_map: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Biome Distribution Statistics"""
        unique, counts = np.unique(biome_map, return_counts=True)
        total = len(biome_map.flat)

        distribution = {f"biome_{int(u)}": float(c / total) * 100 for u, c in zip(unique, counts)}

        return {
            "stat_type": "biome_distribution",
            "total_biomes": len(unique),
            "distribution_percent": distribution,
            "most_common_biome": int(unique[np.argmax(counts)]),
            "diversity_index": float(-np.sum((counts / total) * np.log(counts / total + 1e-10)))
        }

    def _calculate_geology_rock_stats(self, rock_map: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Geology Rock Types Statistics"""
        if len(rock_map.shape) == 3 and rock_map.shape[2] == 3:
            # RGB rock map
            total_pixels = rock_map.shape[0] * rock_map.shape[1]
            distribution = np.sum(rock_map, axis=(0, 1)) / (total_pixels * 255) * 100

            return {
                "stat_type": "geology_rock_types",
                "sedimentary_percent": float(distribution[0]),
                "igneous_percent": float(distribution[1]),
                "metamorphic_percent": float(distribution[2]),
                "mass_conservation_valid": bool(np.allclose(np.sum(rock_map, axis=2), 255, atol=1.0))
            }
        else:
            return {"stat_type": "geology_rock_types", "error": "Invalid rock map format"}

    def _calculate_water_flow_stats(self, flow_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Water Flow Statistics"""
        if len(flow_data.shape) == 3:
            flow_magnitude = np.sqrt(np.sum(flow_data ** 2, axis=2))
        else:
            flow_magnitude = flow_data

        return {
            "stat_type": "water_flow",
            "max_flow_rate": float(np.max(flow_magnitude)),
            "mean_flow_rate": float(np.mean(flow_magnitude)),
            "stagnant_areas_percent": float(np.sum(flow_magnitude < 0.01) / flow_magnitude.size * 100)
        }

    def _calculate_weather_climate_stats(self, weather_data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Weather/Climate Statistics"""
        return {
            "stat_type": "weather_climate",
            "min_temp": float(np.min(weather_data)),
            "max_temp": float(np.max(weather_data)),
            "mean_temp": float(np.mean(weather_data)),
            "temp_variance": float(np.var(weather_data))
        }

    def _calculate_generic_stats(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generic Statistics für unbekannte Datentypen"""
        return {
            "stat_type": "generic",
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "shape": data.shape,
            "dtype": str(data.dtype)
        }


class StatisticsCommunicationPipeline(QObject):
    """
    Funktionsweise: Cross-Tab Statistics-Sharing und Communication
    Aufgabe: Ermöglicht Tabs, Statistics von anderen Tabs zu nutzen
    Features: Tab-Registration, Statistics-Broadcasting, Dependency-Management
    """

    # Signals für Statistics-Communication
    statistics_updated = pyqtSignal(str, str, dict)  # (tab_name, stat_type, statistics)
    statistics_requested = pyqtSignal(str, str)  # (requesting_tab, target_tab)

    def __init__(self, statistics_cache: StatisticsCache):
        super().__init__()

        self.registered_tabs = {}  # {tab_name: tab_instance}
        self.statistics_cache = statistics_cache
        self.tab_dependencies = {}  # {tab_name: [dependency_tabs]}
        self.statistics_history = defaultdict(list)  # {tab_name: [statistics]}

        self.logger = logging.getLogger(__name__)

    def register_tab(self, tab_name: str, tab_instance: Any, dependencies: Optional[List[str]] = None):
        """
        Funktionsweise: Registriert Tab für Statistics-Sharing
        Parameter: tab_name, tab_instance, dependencies
        """
        self.registered_tabs[tab_name] = tab_instance

        if dependencies:
            self.tab_dependencies[tab_name] = dependencies

        self.logger.info(f"Registered tab {tab_name} with dependencies: {dependencies}")

    def get_tab_statistics(self, tab_name: str, stat_type: Optional[str] = None,
                           force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Funktionsweise: Holt Statistics von anderem Tab
        Parameter: tab_name, stat_type, force_refresh
        Return: Statistics dict oder None
        """
        if tab_name not in self.registered_tabs:
            self.logger.warning(f"Tab {tab_name} not registered")
            return None

        tab = self.registered_tabs[tab_name]

        # Try cached statistics first (unless force refresh)
        if not force_refresh and hasattr(tab, 'get_cached_statistics'):
            try:
                cached_stats = tab.get_cached_statistics(stat_type)
                if cached_stats:
                    return cached_stats
            except Exception as e:
                self.logger.debug(f"Failed to get cached statistics from {tab_name}: {e}")

        # Calculate current statistics
        if hasattr(tab, 'calculate_current_statistics'):
            try:
                current_stats = tab.calculate_current_statistics(stat_type)
                if current_stats:
                    # Store in history
                    self.statistics_history[tab_name].append({
                        'timestamp': time.time(),
                        'stat_type': stat_type,
                        'statistics': current_stats
                    })

                    # Limit history size
                    if len(self.statistics_history[tab_name]) > 10:
                        self.statistics_history[tab_name] = self.statistics_history[tab_name][-10:]

                    return current_stats

            except Exception as e:
                self.logger.error(f"Failed to calculate statistics for {tab_name}: {e}")

        return None

    def update_cross_tab_statistics(self, source_tab: str, stat_type: str, statistics: Dict[str, Any]):
        """
        Funktionsweise: Teilt Statistics zwischen Tabs
        Parameter: source_tab, stat_type, statistics
        """
        self.statistics_updated.emit(source_tab, stat_type, statistics)

        # Notify dependent tabs
        for tab_name, tab in self.registered_tabs.items():
            if (tab_name != source_tab and
                    hasattr(tab, 'on_external_statistics_updated') and
                    source_tab in self.tab_dependencies.get(tab_name, [])):

                try:
                    tab.on_external_statistics_updated(source_tab, stat_type, statistics)
                except Exception as e:
                    self.logger.error(f"Failed to update {tab_name} with statistics from {source_tab}: {e}")

    def get_dependency_statistics(self, tab_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Funktionsweise: Sammelt Statistics aller Dependencies für einen Tab
        Parameter: tab_name
        Return: dict {dependency_tab: statistics}
        """
        dependency_stats = {}
        dependencies = self.tab_dependencies.get(tab_name, [])

        for dep_tab in dependencies:
            stats = self.get_tab_statistics(dep_tab)
            if stats:
                dependency_stats[dep_tab] = stats

        return dependency_stats

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Funktionsweise: Sammelt Statistics aller registrierten Tabs
        Return: dict {tab_name: statistics}
        """
        all_stats = {}

        for tab_name in self.registered_tabs:
            stats = self.get_tab_statistics(tab_name)
            if stats:
                all_stats[tab_name] = stats

        return all_stats

    def invalidate_tab_statistics(self, tab_name: str):
        """
        Funktionsweise: Invalidiert Statistics für spezifischen Tab
        Parameter: tab_name
        """
        # Cache invalidation
        self.statistics_cache.invalidate_cache(pattern=tab_name)

        # Clear history
        if tab_name in self.statistics_history:
            del self.statistics_history[tab_name]

        self.logger.info(f"Invalidated statistics for {tab_name}")

    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt Summary aller Statistics für Overview
        Return: Summary dict
        """
        summary = {
            'registered_tabs': list(self.registered_tabs.keys()),
            'total_statistics_entries': sum(len(hist) for hist in self.statistics_history.values()),
            'cache_stats': self.statistics_cache.get_cache_statistics(),
            'tab_dependencies': dict(self.tab_dependencies)
        }

        return summary


# Utility Functions für Statistics-Manager

def create_performance_optimized_statistics_pipeline(cache_size: int = 100,
                                                     cache_age: float = 300) -> StatisticsCommunicationPipeline:
    """
    Funktionsweise: Factory für Performance-optimierte Statistics-Pipeline
    Parameter: cache_size, cache_age
    Return: Konfigurierte StatisticsCommunicationPipeline
    """
    cache = StatisticsCache(max_age_seconds=cache_age, max_entries=cache_size)
    pipeline = StatisticsCommunicationPipeline(cache)

    return pipeline


def register_standard_tabs(pipeline: StatisticsCommunicationPipeline, tabs: Dict[str, Any]):
    """
    Funktionsweise: Registriert Standard-Tab-Set mit Dependencies
    Parameter: pipeline, tabs dict
    """
    # Standard Dependencies für Map-Generator
    dependencies = {
        'terrain': [],
        'geology': ['terrain'],
        'water': ['terrain', 'geology'],
        'weather': ['terrain'],
        'biome': ['terrain', 'geology', 'water', 'weather'],
        'settlement': ['terrain', 'geology', 'water', 'biome']
    }

    for tab_name, tab_instance in tabs.items():
        tab_deps = dependencies.get(tab_name, [])
        pipeline.register_tab(tab_name, tab_instance, tab_deps)