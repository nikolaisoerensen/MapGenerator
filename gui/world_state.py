#!/usr/bin/env python3
"""
Path: MapGenerator/gui/world_state.py
__init__.py existiert in "gui"

Globaler World State Manager
Speichert alle Parameter zwischen den Tabs
"""


class WorldState:
    """Singleton für globalen Zustand"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WorldState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Terrain Parameter
        self.terrain_params = {
            'size': 256,
            'height': 100,
            'octaves': 6,
            'frequency': 0.012,
            'persistence': 0.5,
            'lacunarity': 2.0,
            'redistribute_power': 1.3,
            'seed': 42
        }

        # Geology Parameter
        self.geology_params = {
            'rock_types': [],
            'hardness_values': [],
            'ridge_warping': 0.25,
            'bevel_warping': 0.15
        }

        # Settlement Parameter
        self.settlement_params = {
            'villages': 3,
            'landmarks': 2,
            'pubs': 2,
            'connections': 3,
            'village_size': 15,
            'village_influence': 25,
            'landmark_influence': 20
        }

        # Weather Parameter
        self.weather_params = {
            'max_humidity': 70,
            'rain_amount': 5.0,
            'evaporation': 2.0,
            'wind_speed': 8.0,
            'wind_terrain_influence': 4.0,
            'avg_temperature': 15
        }

        # Water Parameter
        self.water_params = {
            'lake_fill': 40,
            'sea_level': 15,
            'sediment_amount': 50,
            'water_speed': 8.0,
            'rock_dependency': 60
        }

        # UI Zustand
        self.auto_simulate = False
        self.window_geometry = None
        self._initialized = True

    def get_terrain_params(self):
        return self.terrain_params.copy()

    def set_terrain_params(self, params):
        self.terrain_params.update(params)

    def get_geology_params(self):
        return self.geology_params.copy()

    def set_geology_params(self, params):
        self.geology_params.update(params)

    def get_settlement_params(self):
        return self.settlement_params.copy()

    def set_settlement_params(self, params):
        self.settlement_params.update(params)

    def get_weather_params(self):
        return self.weather_params.copy()

    def set_weather_params(self, params):
        self.weather_params.update(params)

    def get_water_params(self):
        return self.water_params.copy()

    def set_water_params(self, params):
        self.water_params.update(params)

    def set_auto_simulate(self, value):
        self.auto_simulate = value

    def get_auto_simulate(self):
        return self.auto_simulate

    def set_window_geometry(self, geometry):
        self.window_geometry = geometry

    def get_window_geometry(self):
        return self.window_geometry