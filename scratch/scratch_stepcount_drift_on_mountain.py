"""
Throwaway diagnostic - isolates whether the remaining LOD3->LOD4 same-
resolution drift comes from step count alone (25 steps vs 18) on REAL
elevation-varying terrain (unlike the earlier flat-map per-step test, this
engages thermal_pressure_coupling since there's real spatial temperature
variation to create pressure anomalies from). Two completely FRESH
(non-inherited) single-shot calls to _run_coupled_atmosphere_simulation on
the identical mountain heightmap, only n_steps differs (18 vs 25).
"""
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.weather_generator import WeatherSystemGenerator, AtmosphereLayers
from managers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER

size = 128
rng = np.random.RandomState(11)
y, x = np.mgrid[0:size, 0:size].astype(np.float64)
nx, ny = x / (size - 1), y / (size - 1)
heightmap = (300.0 + 2800.0 * np.exp(-(((nx - 0.4) ** 2 + (ny - 0.4) ** 2) / (2 * 0.09 ** 2)))
             + 30.0 * rng.randn(size, size)).astype(np.float32)
shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

dlm = DataLODManager()
dlm.set_map_distance_km(20.0)
geo = WeatherSystemGenerator(map_seed=13, data_lod_manager=dlm)

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
    'month_index': 0,
}
month_params = geo._generate_seasonal_parameters(base_params, 0)
roughness_damping = geo._get_roughness_damping(heightmap.shape, 3)

peak_y, peak_x = int(0.4 * (size - 1)), int(0.4 * (size - 1))

print(f"{'n_steps':>8} {'peak_T':>8} {'mean_T':>8} {'min_T':>8} {'max_T':>8}")
for n_steps in (1, 5, 10, 18, 25, 35, 50):
    result = geo._run_coupled_atmosphere_simulation(
        heightmap, shadowmap, month_params, size, n_steps=n_steps, roughness_damping=roughness_damping)
    ground_temp = result['temp_layers'][AtmosphereLayers.GROUND]
    print(f"{n_steps:8d} {ground_temp[peak_y, peak_x]:8.1f} {ground_temp.mean():8.1f} "
          f"{ground_temp.min():8.1f} {ground_temp.max():8.1f}")
