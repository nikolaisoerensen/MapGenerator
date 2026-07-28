"""
Throwaway diagnostic - decisive test: does a FRESH (non-inherited) LOD3 run
on the same multi-peak terrain also come out very cold (like the inherited
LOD1->2->3 chain did), or does it stay close to LOD1's fresh result? If
fresh-LOD3 is warm and inherited-LOD3 is cold, the bug is specifically in
the LOD-inheritance mechanism. If fresh-LOD3 is ALSO cold, the bug is
elsewhere (e.g. more atmosphere-loop steps at higher LOD driving more net
cooling through elevation-dependent dynamics on a flat-map-blind mechanism
like thermal pressure coupling).
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.weather_generator import WeatherSystemGenerator
from gui.OldManagers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER


def make_multipeak_heightmap(size):
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    nx, ny = x / (size - 1), y / (size - 1)
    base = 200.0
    peak1 = 3000.0 * np.exp(-(((nx - 0.3) ** 2 + (ny - 0.3) ** 2) / (2 * 0.08 ** 2)))
    peak2 = 2000.0 * np.exp(-(((nx - 0.7) ** 2 + (ny - 0.6) ** 2) / (2 * 0.10 ** 2)))
    valley = -50.0 * np.exp(-(((nx - 0.6) ** 2 + (ny - 0.2) ** 2) / (2 * 0.10 ** 2)))
    return (base + peak1 + peak2 + valley).astype(np.float32)


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

size = 128
heightmap = make_multipeak_heightmap(size)
shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

nx1, ny1 = int(0.3 * (size - 1)), int(0.3 * (size - 1))
nx2, ny2 = int(0.7 * (size - 1)), int(0.6 * (size - 1))
nx0, ny0 = int(0.05 * (size - 1)), int(0.05 * (size - 1))

print("=== FRESH LOD3 (direct jump, no inheritance) ===")
dlm_fresh = DataLODManager()
dlm_fresh.set_map_distance_km(20.0)
geo_fresh = WeatherSystemGenerator(map_seed=99, data_lod_manager=dlm_fresh)
result_fresh = geo_fresh.calculate_weather_system(heightmap, shadowmap, base_params, lod_level=3)
print(f"peak1_T={result_fresh.temp_map[ny1, nx1]:.1f}  peak2_T={result_fresh.temp_map[ny2, nx2]:.1f}  "
      f"base_T={result_fresh.temp_map[ny0, nx0]:.1f}  mean={result_fresh.temp_map.mean():.1f}")

print("\n=== INHERITED LOD3 (via LOD1 -> LOD2 -> LOD3, same as production) ===")
dlm_inherit = DataLODManager()
dlm_inherit.set_map_distance_km(20.0)
geo_inherit = WeatherSystemGenerator(map_seed=99, data_lod_manager=dlm_inherit)
for lod_level in (1, 2, 3):
    lod_size = min(32 * (2 ** (lod_level - 1)), 128)
    hm = make_multipeak_heightmap(lod_size)
    sm = np.full((lod_size, lod_size, 7), 0.6, dtype=np.float32)
    result_inherit = geo_inherit.calculate_weather_system(hm, sm, base_params, lod_level=lod_level)
nx1i, ny1i = int(0.3 * (size - 1)), int(0.3 * (size - 1))
print(f"peak1_T={result_inherit.temp_map[ny1, nx1]:.1f}  peak2_T={result_inherit.temp_map[ny2, nx2]:.1f}  "
      f"base_T={result_inherit.temp_map[ny0, nx0]:.1f}  mean={result_inherit.temp_map.mean():.1f}")
