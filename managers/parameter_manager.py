"""
Path: gui/managers/parameter_manager.py

Funktionsweise: Zentrale Koordination aller Parameter zwischen Tabs für Export, Reproduzierbarkeit und Cross-Tab-Dependencies
Aufgabe: Parameter-Communication, Export/Import, Preset-Management, Synchronisation zwischen Tabs
Features: Parameter-Hub, Export-Manager, Preset-System, Change-Tracking, Validation
"""

import logging
from typing import Dict, Any, List, Optional, Callable

from PyQt5.QtCore import QObject, pyqtSignal, QTimer

class TERRAIN:
    """Parameter für core/terrain_generator.py"""
    MAPSIZEMIN = 32
    MAPSIZEMAX = 1024

    MAPSIZE = {"min": MAPSIZEMIN, "max": MAPSIZEMAX, "default": 128, "step": 32}
    AMPLITUDE = {"min": 30, "max": 6000.0, "default": 2000.0, "step": 10, "suffix": "m"}
    OCTAVES = {"min": 1, "max": 12, "default": 8, "step": 1}
    FREQUENCY = {"min": 0.001, "max": 0.1, "default": 0.037, "step": 0.001}
    PERSISTENCE = {"min": 0.1, "max": 1.0, "default": 0.68, "step": 0.01}
    LACUNARITY = {"min": 1.1, "max": 4.0, "default": 2.3, "step": 0.1}
    REDISTRIBUTE_POWER = {"min": 0.5, "max": 4.0, "default": 2.5, "step": 0.1}
    MAP_SEED = {"min": 0, "max": 999999, "default": 542595, "step": 1}

class GEOLOGY:
    """Parameter für core/geology_generator.py"""
    SEDIMENTARY_HARDNESS = {"min": 1, "max": 100, "default": 30, "step": 1}
    IGNEOUS_HARDNESS = {"min": 1, "max": 100, "default": 80, "step": 1}
    METAMORPHIC_HARDNESS = {"min": 1, "max": 100, "default": 65, "step": 1}
    RIDGE_WARPING = {"min": 0.0, "max": 2.0, "default": 0.5, "step": 0.1}
    BEVEL_WARPING = {"min": 0.0, "max": 2.0, "default": 0.3, "step": 0.1}
    METAMORPH_FOLIATION = {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.1}
    METAMORPH_FOLDING = {"min": 0.0, "max": 1.0, "default": 0.6, "step": 0.1}
    IGNEOUS_FLOWING = {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.1}

class SETTLEMENT:
    """Parameter für core/settlement_generator.py"""
    SETTLEMENTS = {"min": 1, "max": 5, "default": 3, "step": 1}
    LANDMARKS = {"min": 0, "max": 6, "default": 3, "step": 1}
    ROADSITES = {"min": 0, "max": 6, "default": 3, "step": 1}
    PLOTNODES = {"min": 50, "max": 5000, "default": 1000, "step": 10}
    CIV_INFLUENCE_DECAY = {"min": 0.1, "max": 2.0, "default": 0.8, "step": 0.1}
    TERRAIN_FACTOR_VILLAGES = {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1}
    ROAD_SLOPE_TO_DISTANCE_RATIO = {"min": 0.1, "max": 3.0, "default": 1.5, "step": 0.1}
    LANDMARK_WILDERNESS = {"min": 0.1, "max": 0.8, "default": 0.3, "step": 0.05}
    PLOTSIZE = {"min": 0.5, "max": 5.0, "default": 2.0, "step": 0.1}


class WEATHER:
    """Parameter für core/weather_generator.py"""
    AIR_TEMP_ENTRY = {"min": -30, "max": 40, "default": 15, "step": 1, "suffix": "°C"}
    SOLAR_POWER = {"min": 0, "max": 50, "default": 20, "step": 1, "suffix": "°C"}
    ALTITUDE_COOLING = {"min": 2, "max": 100, "default": 6, "step": 1, "suffix": "°C/km"}
    THERMIC_EFFECT = {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.1}
    WIND_SPEED_FACTOR = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    TERRAIN_FACTOR = {"min": 0.0, "max": 2.0, "default": 1.2, "step": 0.1}


class WATER:
    """Parameter für core/water_generator.py"""
    LAKE_VOLUME_THRESHOLD = {"min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "suffix": "m"}
    RAIN_THRESHOLD = {"min": 1.0, "max": 20.0, "default": 5.0, "step": 0.1, "suffix": "gH2O/m²"}
    MANNING_COEFFICIENT = {"min": 0.01, "max": 0.1, "default": 0.03, "step": 0.005}
    EROSION_STRENGTH = {"min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}
    SEDIMENT_CAPACITY_FACTOR = {"min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01}
    EVAPORATION_BASE_RATE = {"min": 0.0001, "max": 0.01, "default": 0.002, "step": 0.0001, "suffix": "m/Tag"}
    DIFFUSION_RADIUS = {"min": 1.0, "max": 20.0, "default": 5.0, "step": 0.5, "suffix": "Pixel"}
    SETTLING_VELOCITY = {"min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "suffix": "m/s"}


class BIOME:
    """Parameter für core/biome_generator.py"""
    BIOME_WETNESS_FACTOR = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    BIOME_TEMP_FACTOR = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    SEA_LEVEL = {"min": 0, "max": 200, "default": 10, "step": 5, "suffix": "m"}
    BANK_WIDTH = {"min": 1, "max": 20, "default": 3, "step": 1, "suffix": "Pixel"}
    EDGE_SOFTNESS = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    ALPINE_LEVEL = {"min": 500, "max": 3000, "default": 1500, "step": 50, "suffix": "m"}
    SNOW_LEVEL = {"min": 800, "max": 4000, "default": 2000, "step": 50, "suffix": "m"}
    CLIFF_SLOPE = {"min": 30, "max": 80, "default": 60, "step": 1, "suffix": "°"}

class ParameterManager(QObject):

    # Signals für Parameter-Communication
    parameter_changed = pyqtSignal(str, str, object, object)  # (tab, param, old_val, new_val)


    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.defaults = {
            BIOME: "biome",
        }

        self.MapValidated = False           # Erst muss einmal gerechnet werden damit die Daten validiert werden

    def get_parameter(self, tab_name: str, parameter_name: str) -> Dict[str, Any]:
        """
        Funktionsweise: Holt Parameter eines spezifischen Tabs
        Parameter: tab_name
        Return: Parameters dict
        """

        if hasattr(tab, 'get_current_parameters'):
            try:
                params = tab.get_current_parameters()
                self.parameter_cache[tab_name] = params
                return params
            except Exception as e:
                self.logger.error(f"Failed to get parameters from {tab_name}: {e}")

        return self.parameter_cache.get(parameter_name, {})

    def set_parameter(self, tab_name: str, parameters: Dict[str, Any],
                          validate: bool = True, notify_listeners: bool = True):
        """
        Funktionsweise: Setzt Parameter für spezifischen Tab
        Parameter: tab_name, parameters, validate, notify_listeners
        """
        if tab_name not in self.registered_tabs:
            self.logger.warning(f"Tab {tab_name} not registered")
            return False

        tab = self.registered_tabs[tab_name]
        old_params = self.parameter_cache.get(tab_name, {})

        # Validation (optional)
        if validate:
            is_valid, errors = self._validate_parameters(tab_name, parameters)
            if not is_valid:
                self.logger.error(f"Parameter validation failed for {tab_name}: {errors}")
                self.validation_status_changed.emit(tab_name, False, errors)
                return False
            else:
                self.validation_status_changed.emit(tab_name, True, [])

        # Parameter setzen
        if hasattr(tab, 'set_parameters'):
            try:
                tab.set_parameters(parameters)
                self.parameter_cache[tab_name] = parameters

                # Change-Tracking
                self._track_parameter_changes(tab_name, old_params, parameters)

                # Listeners benachrichtigen
                if notify_listeners:
                    self._notify_parameter_change(tab_name, parameters)

                self.tab_parameters_updated.emit(tab_name, parameters)
                return True

            except Exception as e:
                self.logger.error(f"Failed to set parameters for {tab_name}: {e}")
                return False
        else:
            self.logger.warning(f"Tab {tab_name} does not support set_parameters")
            return False

    def _invalidate_map(self, parameter):
         self.mapValidated = False
         self.logger.info(f"Invalidated MapData after changed parameter {parameter}.")

    def _validate_map(self):
         self.mapValidated = True
         self.logger.info(f"Validated MapData after finished calculation.")