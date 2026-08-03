"""
Throwaway diagnostic - isolates whether temperature drift comes from the
per-step physics loop itself (independent of LOD-inheritance) by running
_run_coupled_atmosphere_simulation directly on a FLAT heightmap (so the
lapse-rate contribution is a known constant and any drift must come from
the step loop, not from elevation/inheritance arithmetic) with increasing
n_steps, and printing GROUND-layer real temperature at a fixed point after
each step count.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.weather_generator import WeatherSystemGenerator, AtmosphereLayers
from managers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER

size = 48
heightmap = np.full((size, size), 500.0, dtype=np.float32)  # flach, bekannte Hoehe
shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

dlm = DataLODManager()
dlm.set_map_distance_km(20.0)
geo = WeatherSystemGenerator(map_seed=7, data_lod_manager=dlm)

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

climate_temp, _ = geo._climate_baseline(45.0, 0)
lapse = month_params['altitude_cooling'] / 1000.0
expected_ground_temp = climate_temp + month_params.get('air_temp_entry_offset', 0.0) - lapse * (500.0 + 75.0)
print(f"climate_temp(lat=45,month=0) = {climate_temp:.2f}, lapse*altitude(575m) = {lapse*575.0:.2f}")
print(f"Roughly-expected GROUND temp (ignoring solar/noise) ~= {climate_temp - lapse*575.0:.2f}C\n")

roughness_damping = geo._get_roughness_damping(heightmap.shape, 3)
center = size // 2

print(f"{'n_steps':>8} {'ground_T_center':>16} {'ground_T_mean':>15} {'ground_T_min':>13} {'ground_T_max':>13}")
for n_steps in (0, 1, 2, 5, 10, 20, 40):
    result = geo._run_coupled_atmosphere_simulation(
        heightmap, shadowmap, month_params, size, n_steps=n_steps, roughness_damping=roughness_damping)
    ground_temp = result['temp_layers'][AtmosphereLayers.GROUND]
    print(f"{n_steps:8d} {ground_temp[center, center]:16.2f} {ground_temp.mean():15.2f} "
          f"{ground_temp.min():13.2f} {ground_temp.max():13.2f}")
