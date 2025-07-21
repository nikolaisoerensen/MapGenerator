"""
Path: gui/managers/data_manager.py

Funktionsweise: Zentrale Datenverwaltung für alle Tabs mit erweiterten Datenstrukturen
- Memory-effiziente numpy Array-Referenzen (keine Kopien)
- LOD-System: Live (64x64), Preview (256x256), Final (512x512)
- Dependency-Tracking zwischen Generator-Outputs erweitert
- Cross-Tab Data-Sharing ohne Pickle-Serialisierung
- Automatic Cache-Invalidation bei Parameter-Änderungen
- Unterstützung für alle neuen Core-Generator Outputs
"""

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, Any, Optional, List, Tuple
import logging

def get_data_manager_error_decorators():
    """
    Funktionsweise: Lazy Loading von Data Manager Error Decorators
    Aufgabe: Lädt Data-Management und Memory-Critical Decorators
    Return: Tuple von Decorator-Funktionen
    """
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

class DataManager(QObject):
    """
    Funktionsweise: Zentrale Klasse für Datenverwaltung aller Generator-Outputs
    Aufgabe: Speichert und verwaltet alle Arrays zwischen Tabs, Cache-Management
    Kommunikation: Signals für Data-Updates, Memory-Management für große Arrays
    """

    # Signals für Cross-Tab Communication
    data_updated = pyqtSignal(str, str)  # (generator_type, data_key)
    cache_invalidated = pyqtSignal(str)  # (generator_type)
    dependency_changed = pyqtSignal(str, list)  # (generator_type, missing_dependencies)

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Haupt-Datenstrukturen für alle Generator-Outputs
        self._terrain_data = {}
        self._geology_data = {}
        self._settlement_data = {}
        self._weather_data = {}
        self._water_data = {}
        self._biome_data = {}

        # Cache-Management
        self._cache_timestamps = {}
        self._parameter_hashes = {}

        # LOD-Level Management
        self._current_lod = "preview"  # "live", "preview", "final"
        self._lod_sizes = {
            "live": 64,
            "preview": 256,
            "final": 512
        }

    def set_lod_level(self, lod_level: str):
        """
        Funktionsweise: Setzt aktuelles Level-of-Detail für alle Generatoren
        Aufgabe: Performance-Optimierung durch angepasste Array-Größen
        Parameter: lod_level ("live", "preview", "final")
        """
        if lod_level not in self._lod_sizes:
            raise ValueError(f"Invalid LOD level: {lod_level}")
        self._current_lod = lod_level
        self.logger.info(f"LOD level set to {lod_level} ({self._lod_sizes[lod_level]}x{self._lod_sizes[lod_level]})")

    def get_current_map_size(self) -> int:
        """
        Funktionsweise: Gibt aktuelle Kartengröße basierend auf LOD zurück
        Return: int (Kartengröße in Pixeln)
        """
        return self._lod_sizes[self._current_lod]

    # ===== TERRAIN DATA MANAGEMENT =====
    @data_management_handler("terrain_data")
    def set_terrain_data(self, data_key: str, data: np.ndarray, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert einzelne Terrain-Arrays (Legacy-Kompatibilität)
        Parameter: data_key ("heightmap", "slopemap", "shademap"), data (numpy array), parameters (dict)
        """
        self._terrain_data[data_key] = data
        self._update_cache_timestamp("terrain", data_key, parameters)
        self.data_updated.emit("terrain", data_key)
        self.logger.debug(f"Terrain data '{data_key}' updated, shape: {data.shape}")

    def set_terrain_data_complete(self, terrain_data, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert komplettes TerrainData-Objekt mit LOD-Informationen
        Parameter: terrain_data (TerrainData-Objekt), parameters (dict)
        """
        # TerrainData-Objekt komplett speichern
        self._terrain_data["terrain_data_object"] = terrain_data

        # Einzelne Arrays für Legacy-Kompatibilität extrahieren
        if terrain_data.heightmap is not None:
            self._terrain_data["heightmap"] = terrain_data.heightmap
        if terrain_data.slopemap is not None:
            self._terrain_data["slopemap"] = terrain_data.slopemap
        if terrain_data.shadowmap is not None:
            self._terrain_data["shadowmap"] = terrain_data.shadowmap

        # Cache und Metadaten
        self._update_cache_timestamp("terrain", "complete", parameters)
        self._terrain_data["lod_level"] = terrain_data.lod_level
        self._terrain_data["actual_size"] = terrain_data.actual_size

        self.data_updated.emit("terrain", "complete")
        self.logger.debug(
            f"Complete terrain data updated, LOD: {terrain_data.lod_level}, size: {terrain_data.actual_size}")

    def get_terrain_data(self, data_key: str = None):
        """
        Funktionsweise: Holt Terrain-Daten für andere Generatoren
        Parameter: data_key ("heightmap", "slopemap", "shademap", "complete", None für alles)
        Return: numpy array, TerrainData-Objekt oder dict aller Daten
        """
        if data_key is None:
            return self._terrain_data
        elif data_key == "complete":
            return self._terrain_data.get("terrain_data_object")
        else:
            return self._terrain_data.get(data_key)

    def get_terrain_lod_info(self) -> Dict[str, Any]:
        """
        Funktionsweise: Gibt aktuelle LOD-Informationen des Terrains zurück
        Return: dict mit lod_level, actual_size, available_outputs
        """
        terrain_obj = self._terrain_data.get("terrain_data_object")
        if terrain_obj:
            return {
                "lod_level": terrain_obj.lod_level,
                "actual_size": terrain_obj.actual_size,
                "calculated_sun_angles": getattr(terrain_obj, 'calculated_sun_angles', []),
                "available_outputs": self.get_terrain_dependencies()
            }
        else:
            return {
                "lod_level": self._terrain_data.get("lod_level", "unknown"),
                "actual_size": self._terrain_data.get("actual_size", 0),
                "calculated_sun_angles": [],
                "available_outputs": self.get_terrain_dependencies()
            }

    def has_terrain_lod(self, required_lod: str) -> bool:
        """
        Funktionsweise: Prüft ob Terrain-Daten mindestens in erforderlicher LOD vorhanden sind
        Parameter: required_lod ("LOD64", "LOD128", "LOD256", "FINAL")
        Return: bool (LOD ist verfügbar)
        """
        lod_hierarchy = {"LOD64": 1, "LOD128": 2, "LOD256": 3, "FINAL": 4}

        current_lod = self._terrain_data.get("lod_level", "LOD64")
        if current_lod not in lod_hierarchy or required_lod not in lod_hierarchy:
            return False

        return lod_hierarchy[current_lod] >= lod_hierarchy[required_lod]

    def get_terrain_dependencies(self) -> List[str]:
        """Return: Liste verfügbarer Terrain-Outputs für Dependencies"""
        deps = []
        for key in ["heightmap", "slopemap", "shadowmap"]:
            if key in self._terrain_data:
                deps.append(key)

        # LOD-spezifische Dependencies hinzufügen
        if "terrain_data_object" in self._terrain_data:
            deps.append("terrain_complete")
            lod_info = self.get_terrain_lod_info()
            deps.append(f"terrain_{lod_info['lod_level']}")

        return deps

    # FÜGE diese neue Methode zum LOD-Level Management hinzu (nach set_lod_level):

    def set_lod_level(self, lod_level: str):
        """
        Funktionsweise: Setzt aktuelles Level-of-Detail für alle Generatoren
        Aufgabe: Performance-Optimierung durch angepasste Array-Größen
        Parameter: lod_level ("live", "preview", "final")
        """
        if lod_level not in self._lod_sizes:
            raise ValueError(f"Invalid LOD level: {lod_level}")
        self._current_lod = lod_level
        self.logger.info(f"LOD level set to {lod_level} ({self._lod_sizes[lod_level]}x{self._lod_sizes[lod_level]})")

    def translate_lod_to_terrain(self, data_manager_lod: str) -> str:
        """
        Funktionsweise: Übersetzt DataManager-LOD zu Terrain-Generator-LOD
        Aufgabe: Kompatibilität zwischen verschiedenen LOD-Systemen
        Parameter: data_manager_lod ("live", "preview", "final")
        Return: str - Terrain-Generator LOD ("LOD64", "LOD128", "LOD256", "FINAL")
        """
        lod_translation = {
            "live": "LOD64",
            "preview": "LOD256",
            "final": "FINAL"
        }
        return lod_translation.get(data_manager_lod, "LOD64")

    def translate_terrain_to_lod(self, terrain_lod: str) -> str:
        """
        Funktionsweise: Übersetzt Terrain-Generator-LOD zu DataManager-LOD
        Parameter: terrain_lod ("LOD64", "LOD128", "LOD256", "FINAL")
        Return: str - DataManager LOD ("live", "preview", "final")
        """
        terrain_translation = {
            "LOD64": "live",
            "LOD128": "preview",
            "LOD256": "preview",
            "FINAL": "final"
        }
        return terrain_translation.get(terrain_lod, "live")

    # ===== GEOLOGY DATA MANAGEMENT =====
    def set_geology_data(self, data_key: str, data: np.ndarray, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Geology-Generator Output
        Parameter: data_key ("rock_map", "hardness_map"), data (numpy array), parameters (dict)
        """
        self._geology_data[data_key] = data
        self._update_cache_timestamp("geology", data_key, parameters)
        self.data_updated.emit("geology", data_key)
        self.logger.debug(f"Geology data '{data_key}' updated, shape: {data.shape}")

    def get_geology_data(self, data_key: str) -> Optional[np.ndarray]:
        """
        Parameter: data_key ("rock_map", "hardness_map")
        Return: numpy array oder None
        """
        return self._geology_data.get(data_key)

    def get_geology_dependencies(self) -> List[str]:
        return list(self._geology_data.keys())

    # ===== SETTLEMENT DATA MANAGEMENT =====
    def set_settlement_data(self, data_key: str, data: Any, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Settlement-Generator Output (Listen und Arrays)
        Parameter: data_key ("settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map")
        """
        self._settlement_data[data_key] = data
        self._update_cache_timestamp("settlement", data_key, parameters)
        self.data_updated.emit("settlement", data_key)

        if isinstance(data, np.ndarray):
            self.logger.debug(f"Settlement data '{data_key}' updated, shape: {data.shape}")
        else:
            self.logger.debug(f"Settlement data '{data_key}' updated, type: {type(data)}")

    def get_settlement_data(self, data_key: str) -> Any:
        """
        Parameter: data_key ("settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map")
        """
        return self._settlement_data.get(data_key)

    def get_settlement_dependencies(self) -> List[str]:
        return list(self._settlement_data.keys())

    # ===== WEATHER DATA MANAGEMENT =====
    def set_weather_data(self, data_key: str, data: np.ndarray, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Weather-Generator Output
        Parameter: data_key ("wind_map", "temp_map", "precip_map", "humid_map")
        """
        self._weather_data[data_key] = data
        self._update_cache_timestamp("weather", data_key, parameters)
        self.data_updated.emit("weather", data_key)
        self.logger.debug(f"Weather data '{data_key}' updated, shape: {data.shape}")

    def get_weather_data(self, data_key: str) -> Optional[np.ndarray]:
        """
        Parameter: data_key ("wind_map", "temp_map", "precip_map", "humid_map")
        """
        return self._weather_data.get(data_key)

    def get_weather_dependencies(self) -> List[str]:
        return list(self._weather_data.keys())

    # ===== WATER DATA MANAGEMENT =====
    def set_water_data(self, data_key: str, data: Any, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Water-Generator Output (Arrays und Skalar-Werte)
        Parameter: data_key ("water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
                            "erosion_map", "sedimentation_map", "rock_map_updated", "evaporation_map",
                            "ocean_outflow", "water_biomes_map")
        """
        self._water_data[data_key] = data
        self._update_cache_timestamp("water", data_key, parameters)
        self.data_updated.emit("water", data_key)

        if isinstance(data, np.ndarray):
            self.logger.debug(f"Water data '{data_key}' updated, shape: {data.shape}")
        else:
            self.logger.debug(f"Water data '{data_key}' updated, value: {data}")

    def get_water_data(self, data_key: str) -> Any:
        """
        Parameter: data_key ("water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
                            "erosion_map", "sedimentation_map", "rock_map_updated", "evaporation_map",
                            "ocean_outflow", "water_biomes_map")
        """
        return self._water_data.get(data_key)

    def get_water_dependencies(self) -> List[str]:
        return list(self._water_data.keys())

    # ===== BIOME DATA MANAGEMENT =====
    def set_biome_data(self, data_key: str, data: np.ndarray, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert Biome-Generator Output
        Parameter: data_key ("biome_map", "biome_map_super", "super_biome_mask")
        """
        self._biome_data[data_key] = data
        self._update_cache_timestamp("biome", data_key, parameters)
        self.data_updated.emit("biome", data_key)
        self.logger.debug(f"Biome data '{data_key}' updated, shape: {data.shape}")

    def get_biome_data(self, data_key: str) -> Optional[np.ndarray]:
        """
        Parameter: data_key ("biome_map", "biome_map_super", "super_biome_mask")
        """
        return self._biome_data.get(data_key)

    def get_biome_dependencies(self) -> List[str]:
        return list(self._biome_data.keys())

    # ===== DEPENDENCY CHECKING =====
    @memory_critical_handler("dependency_check")
    def check_dependencies(self, generator_type: str, required_dependencies: List[str]) -> Tuple[bool, List[str]]:
        """
        Funktionsweise: Prüft ob alle Dependencies für Generator verfügbar sind
        Parameter: generator_type (str), required_dependencies (List[str])
        Return: (alle_verfügbar: bool, fehlende_dependencies: List[str])
        """
        available_data = self._get_all_available_data()
        missing = []

        for dependency in required_dependencies:
            if dependency not in available_data:
                missing.append(dependency)

        is_complete = len(missing) == 0

        if not is_complete:
            self.dependency_changed.emit(generator_type, missing)
            self.logger.warning(f"{generator_type} missing dependencies: {missing}")

        return is_complete, missing

    def _get_all_available_data(self) -> List[str]:
        """
        Funktionsweise: Sammelt alle verfügbaren Daten-Keys von allen Generatoren
        Return: Liste aller verfügbaren Daten-Keys
        """
        all_data = []
        all_data.extend(self._terrain_data.keys())
        all_data.extend(self._geology_data.keys())
        all_data.extend(self._settlement_data.keys())
        all_data.extend(self._weather_data.keys())
        all_data.extend(self._water_data.keys())
        all_data.extend(self._biome_data.keys())
        return all_data

    # ===== CACHE MANAGEMENT =====
    def _update_cache_timestamp(self, generator_type: str, data_key: str, parameters: Dict[str, Any]):
        """
        Funktionsweise: Aktualisiert Cache-Timestamp und Parameter-Hash für Invalidation
        Parameter: generator_type (str), data_key (str), parameters (dict)
        """
        import time
        import hashlib

        timestamp = time.time()
        param_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()

        cache_key = f"{generator_type}_{data_key}"
        self._cache_timestamps[cache_key] = timestamp
        self._parameter_hashes[cache_key] = param_hash

    def is_cache_valid(self, generator_type: str, data_key: str, parameters: Dict[str, Any]) -> bool:
        """
        Funktionsweise: Prüft ob Cache für gegebene Parameter noch valid ist
        Parameter: generator_type (str), data_key (str), parameters (dict)
        Return: bool (Cache ist valid)
        """
        import hashlib

        cache_key = f"{generator_type}_{data_key}"

        if cache_key not in self._parameter_hashes:
            return False

        current_hash = hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()
        return self._parameter_hashes[cache_key] == current_hash

    def invalidate_cache(self, generator_type: str):
        """
        Funktionsweise: Invalidiert kompletten Cache für einen Generator-Typ
        Parameter: generator_type (str)
        """
        # Entferne alle Cache-Einträge für diesen Generator
        keys_to_remove = [key for key in self._cache_timestamps.keys() if key.startswith(generator_type)]

        for key in keys_to_remove:
            del self._cache_timestamps[key]
            if key in self._parameter_hashes:
                del self._parameter_hashes[key]

        # Lösche entsprechende Daten
        if generator_type == "terrain":
            self._terrain_data.clear()
        elif generator_type == "geology":
            self._geology_data.clear()
        elif generator_type == "settlement":
            self._settlement_data.clear()
        elif generator_type == "weather":
            self._weather_data.clear()
        elif generator_type == "water":
            self._water_data.clear()
        elif generator_type == "biome":
            self._biome_data.clear()

        self.cache_invalidated.emit(generator_type)
        self.logger.info(f"Cache invalidated for {generator_type}")

    # ===== UTILITY METHODS =====
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Funktionsweise: Berechnet Memory-Usage aller gespeicherten Arrays
        Return: dict mit Memory-Usage in MB für jeden Generator
        """
        def calculate_array_memory(data_dict):
            total_memory = 0
            for data in data_dict.values():
                if isinstance(data, np.ndarray):
                    total_memory += data.nbytes
            return total_memory / (1024 * 1024)  # Convert to MB

        return {
            "terrain": calculate_array_memory(self._terrain_data),
            "geology": calculate_array_memory(self._geology_data),
            "settlement": calculate_array_memory(self._settlement_data),
            "weather": calculate_array_memory(self._weather_data),
            "water": calculate_array_memory(self._water_data),
            "biome": calculate_array_memory(self._biome_data)
        }

    def clear_all_data(self):
        """
        Funktionsweise: Löscht alle Daten und Cache (für Neustart)
        """
        self._terrain_data.clear()
        self._geology_data.clear()
        self._settlement_data.clear()
        self._weather_data.clear()
        self._water_data.clear()
        self._biome_data.clear()

        self._cache_timestamps.clear()
        self._parameter_hashes.clear()

        self.logger.info("All data cleared")

    def export_data_summary(self) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt Zusammenfassung aller verfügbaren Daten für Export
        Return: dict mit Summary aller Generator-Outputs
        """
        summary = {
            "terrain": {
                "available_outputs": list(self._terrain_data.keys()),
                "shapes": {key: data.shape for key, data in self._terrain_data.items()}
            },
            "geology": {
                "available_outputs": list(self._geology_data.keys()),
                "shapes": {key: data.shape for key, data in self._geology_data.items()}
            },
            "settlement": {
                "available_outputs": list(self._settlement_data.keys()),
                "types": {key: type(data).__name__ for key, data in self._settlement_data.items()}
            },
            "weather": {
                "available_outputs": list(self._weather_data.keys()),
                "shapes": {key: data.shape for key, data in self._weather_data.items()}
            },
            "water": {
                "available_outputs": list(self._water_data.keys()),
                "types": {key: type(data).__name__ for key, data in self._water_data.items()}
            },
            "biome": {
                "available_outputs": list(self._biome_data.keys()),
                "shapes": {key: data.shape for key, data in self._biome_data.items()}
            },
            "memory_usage_mb": self.get_memory_usage(),
            "current_lod": self._current_lod,
            "map_size": self.get_current_map_size()
        }
        return summary