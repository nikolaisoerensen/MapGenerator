#!/usr/bin/env python3
"""
Path: MapGenerator/gui/managers/data_pipeline.py

NEUE DATEI - Datenübertragung zwischen Tabs
"""


class WorldDataPipeline:
    """Singleton für Tab-übergreifende Daten"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Daten-Speicher
        self.terrain_heightmap = None
        self.terrain_params = None
        self.geology_data = None
        self.weather_data = None
        self.water_data = None
        self.biome_data = None

        self._initialized = True

    def store_terrain(self, heightmap, params):
        """Terrain Tab speichert hier"""
        self.terrain_heightmap = heightmap
        self.terrain_params = params
        print(f"Pipeline: Terrain gespeichert {heightmap.shape}")

    def get_terrain(self):
        """Geology Tab holt hier"""
        if self.terrain_heightmap is None:
            return None, None
        return self.terrain_heightmap, self.terrain_params

    def store_geology(self, geology_result):
        self.geology_data = geology_result
        print("Pipeline: Geology gespeichert")

    def get_geology(self):
        return self.geology_data

    def store_settlement(self, settlement_result):
        self.settlement_data = settlement_result
        print("Pipeline: Settlement gespeichert")

    def get_settlement(self):
        return self.settlement_data

    def store_weather(self, weather_result):
        self.weather_data = weather_result
        print("Pipeline: Weather gespeichert")

    def get_weather(self):
        return self.weather_data

    def store_water(self, water_result):
        self.water_data = water_result
        print("Pipeline: Water gespeichert")

    def get_water(self):
        return self.water_data

    def store_biome(self, biome_result):
        self.biome_data = biome_result
        print("Pipeline: Biome gespeichert")

    def get_biome(self):
        return self.biome_data