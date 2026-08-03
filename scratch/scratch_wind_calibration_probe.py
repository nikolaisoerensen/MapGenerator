"""
Throwaway diagnostic (NOT a smoke test) - measures actual wind_map speed
statistics the coupled atmosphere simulation currently produces across a
handful of representative latitude/season combinations, so the Weather
wind-speed calibration (Nutzer-Abstimmung 2026-07-23: typical 2-15 m/s,
extremes up to 25/30/40 m/s GROUND/MID/HIGH, strongest at high latitude in
winter) can be checked against reality before touching any constants.
Prints stats only, no pass/fail assertions - delete after use.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

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

print(f"{'lat':>5} {'month':>6} {'ground_mean':>12} {'ground_max':>11} "
      f"{'mid_mean':>9} {'mid_max':>8} {'high_mean':>10} {'high_max':>9}")

for lat in (5.0, 25.0, 45.0, 65.0, 80.0):
    for month_index, label in ((0, "JanFeb"), (3, "JulAug")):
        dlm = DataLODManager()
        dlm.set_map_distance_km(15.0)
        geo = WeatherSystemGenerator(map_seed=42, data_lod_manager=dlm)
        params = dict(base_params, map_latitude=lat)

        calc = ShadowCalculator()
        sun_angles = generate_seasonal_sun_angles(month_index, lat, 0.0)
        shadowmap = calc.calculate_shadows(heightmap, lod_level=3, sun_angles_override=sun_angles)

        month_params = geo._generate_seasonal_parameters(params, month_index)
        atmosphere_steps = geo._get_atmosphere_loop_steps(3)
        result = geo._run_coupled_atmosphere_simulation(
            heightmap, shadowmap, month_params, size, n_steps=atmosphere_steps)

        wind_layers = result['wind_layers']  # (3,H,W,2)
        mags = np.hypot(wind_layers[:, :, :, 0], wind_layers[:, :, :, 1])
        g_mean, g_max = float(mags[0].mean()), float(mags[0].max())
        m_mean, m_max = float(mags[1].mean()), float(mags[1].max())
        h_mean, h_max = float(mags[2].mean()), float(mags[2].max())
        print(f"{lat:5.0f} {label:>6} {g_mean:12.2f} {g_max:11.2f} "
              f"{m_mean:9.2f} {m_max:8.2f} {h_mean:10.2f} {h_max:9.2f}")
