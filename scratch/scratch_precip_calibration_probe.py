"""
Throwaway diagnostic (NOT a smoke test) - measures the actual precip_map
output range the coupled atmosphere simulation currently produces, across
representative latitude/season combinations, so the precipitation rescale
(Nutzer-Abstimmung 2026-07-23: "annual mm/year equivalent", desert ~50-250,
temperate ~600-1200, rainforest ~2000-3000) can be designed against real
current numbers instead of the stale ~50-max comment in biome_generator.py
(core/biome_generator.py _rescale_precip_moisture_ranges() docstring - that
comment predates this session's coupled 3-layer CFD rework). Prints stats
only, no assertions - delete after use.
"""
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.weather_generator import WeatherSystemGenerator
from core.terrain_generator import ShadowCalculator, generate_seasonal_sun_angles
from managers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER

size = 64
rng = np.random.RandomState(9)
x = np.linspace(0, 5, size)
y = np.linspace(0, 5, size)
X, Y = np.meshgrid(x, y)
heightmap = (600 + 300 * np.sin(X) * np.cos(Y) + 80 * rng.randn(size, size)).astype(np.float32)

base_params = {
    'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
    'solar_power': WEATHER.SOLAR_POWER["default"],
    'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
    'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
    'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
    'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
    'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
    'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
    'map_longitude': 0.0,
}

print(f"{'lat':>5} {'humidity':>9} {'precip_mean':>12} {'precip_max':>11} {'precip_p90':>11}")

for lat in (5.0, 25.0, 45.0, 65.0, 80.0):
    for humidity_offset, hlabel in ((0.0, "dry"), (30.0, "wet")):
        dlm = DataLODManager()
        dlm.set_map_distance_km(15.0)
        geo = WeatherSystemGenerator(map_seed=42, data_lod_manager=dlm)
        params = dict(base_params, map_latitude=lat, air_humidity_entry=humidity_offset)

        calc = ShadowCalculator()
        sun_angles = generate_seasonal_sun_angles(3, lat, 0.0)
        shadowmap = calc.calculate_shadows(heightmap, lod_level=3, sun_angles_override=sun_angles)

        result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
        p = result.precip_map
        print(f"{lat:5.0f} {hlabel:>9} {float(p.mean()):12.2f} {float(p.max()):11.2f} "
              f"{float(np.percentile(p, 90)):11.2f}")
