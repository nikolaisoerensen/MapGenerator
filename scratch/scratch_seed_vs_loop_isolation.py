"""
Throwaway diagnostic - isolates whether the residual LOD-inheritance drift
comes from the SEEDING arithmetic itself (wrong immediately at n_steps=0,
before any simulation runs) or from the LOOP treating an inherited starting
state differently than a fresh one (seed looks fine at n_steps=0 but
diverges only after steps run). Constructs a real initial_state exactly
like _calc_temperature would (by first running a real LOD3 pass), then
compares fresh vs inherited GROUND peak temperature at n_steps=0 and at the
LOD4 step count (25).
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

# Echter LOD3-Lauf (fresh), liefert temp_layers/wind_layers/humid_layers
# exakt wie _calc_temperature sie fuer die naechste Runde speichern wuerde.
result_lod3 = geo._run_coupled_atmosphere_simulation(
    heightmap, shadowmap, month_params, size, n_steps=18, roughness_damping=roughness_damping)
print(f"LOD3 (18 steps, fresh) peak_T = {result_lod3['temp_layers'][AtmosphereLayers.GROUND][peak_y, peak_x]:.1f}")

initial_state = {
    'temp_layers': result_lod3['temp_layers'],
    'wind_layers': result_lod3['wind_layers'],
    'humid_layers': result_lod3['humid_layers'],
}

print(f"\n{'n_steps':>8} {'fresh_peak_T':>13} {'inherited_peak_T':>17} {'diff':>7}")
for n_steps in (0, 1, 5, 10, 18, 25):
    result_fresh = geo._run_coupled_atmosphere_simulation(
        heightmap, shadowmap, month_params, size, n_steps=n_steps, roughness_damping=roughness_damping)
    result_inherited = geo._run_coupled_atmosphere_simulation(
        heightmap, shadowmap, month_params, size, n_steps=n_steps, roughness_damping=roughness_damping,
        initial_state=initial_state)
    fresh_t = result_fresh['temp_layers'][AtmosphereLayers.GROUND][peak_y, peak_x]
    inherited_t = result_inherited['temp_layers'][AtmosphereLayers.GROUND][peak_y, peak_x]
    print(f"{n_steps:8d} {fresh_t:13.1f} {inherited_t:17.1f} {inherited_t - fresh_t:7.1f}")
