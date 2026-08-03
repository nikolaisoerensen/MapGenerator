"""
Throwaway diagnostic (NOT a smoke test) - runs the EXACT LOD1->LOD2->LOD3
progression the real app performs by default (same DataLODManager/generator
instance across calls, so LOD-inheritance activates exactly like in
production), on a multi-peak terrain with KNOWN elevations, and prints
temperature statistics at each LOD to check whether there's a systematic
cold-drift across LOD transitions (user's hypothesis 2026-07-24: "du wirst
sehen das alles viel zu kalt wird aus irgendeinem Grund").
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.weather_generator import WeatherSystemGenerator
from managers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER


def make_multipeak_heightmap(size):
    """Kontinuierliche Funktion normierter Koordinaten - bei jeder Aufloesung
    dieselbe REALE Landschaft, nur feiner abgetastet (wie echte Terrain-
    Generierung, keine simple Hochskalierung eines Low-Res-Arrays)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    nx, ny = x / (size - 1), y / (size - 1)
    base = 200.0
    peak1 = 3000.0 * np.exp(-(((nx - 0.3) ** 2 + (ny - 0.3) ** 2) / (2 * 0.08 ** 2)))
    peak2 = 2000.0 * np.exp(-(((nx - 0.7) ** 2 + (ny - 0.6) ** 2) / (2 * 0.10 ** 2)))
    valley = -50.0 * np.exp(-(((nx - 0.6) ** 2 + (ny - 0.2) ** 2) / (2 * 0.10 ** 2)))
    return (base + peak1 + peak2 + valley).astype(np.float32)


def main():
    base_params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'solar_power': WEATHER.SOLAR_POWER["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 45.0,
        'map_longitude': 15.0,
    }

    dlm = DataLODManager()
    dlm.set_map_distance_km(20.0)
    geo = WeatherSystemGenerator(map_seed=99, data_lod_manager=dlm)

    print(f"{'LOD':>4} {'size':>5} {'peak1_h':>8} {'peak1_T':>8} {'peak2_h':>8} "
          f"{'peak2_T':>8} {'base_h':>7} {'base_T':>7} {'map_min':>8} {'map_max':>8} {'map_mean':>9}")

    for lod_level in (1, 2, 3, 4):
        size = min(32 * (2 ** (lod_level - 1)), 128)
        heightmap = make_multipeak_heightmap(size)
        shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

        result = geo.calculate_weather_system(heightmap, shadowmap, base_params, lod_level=lod_level)

        nx1, ny1 = int(0.3 * (size - 1)), int(0.3 * (size - 1))
        nx2, ny2 = int(0.7 * (size - 1)), int(0.6 * (size - 1))
        nx0, ny0 = int(0.05 * (size - 1)), int(0.05 * (size - 1))  # Ecke, nahe Basishoehe

        peak1_h = heightmap[ny1, nx1]
        peak2_h = heightmap[ny2, nx2]
        base_h = heightmap[ny0, nx0]
        peak1_T = result.temp_map[ny1, nx1]
        peak2_T = result.temp_map[ny2, nx2]
        base_T = result.temp_map[ny0, nx0]

        print(f"{lod_level:4d} {size:5d} {peak1_h:8.0f} {peak1_T:8.1f} {peak2_h:8.0f} "
              f"{peak2_T:8.1f} {base_h:7.0f} {base_T:7.1f} "
              f"{result.temp_map.min():8.1f} {result.temp_map.max():8.1f} {result.temp_map.mean():9.1f}")


if __name__ == "__main__":
    main()
