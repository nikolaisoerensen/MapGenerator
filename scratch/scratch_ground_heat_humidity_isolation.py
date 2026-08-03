"""
Throwaway diagnostic (NOT a smoke test) - isolates the latent-heat
contribution to temp_map from the new sensible (ground-air convective)
heat-transfer term, per the plan "Weather: Bodentemperatur-Modell +
konvektiver Waermeuebergang" section 8 / user request. Runs the same
scenario twice, once with air_humidity_entry pushed to its slider minimum
(as close to 0% base humidity as the latitude allows - the combined value
climate_humid_fraction*100 + offset is clipped to [0,100], so equatorial
latitudes with a high climate_humid_fraction can't reach exactly 0% even
at the full -50 offset) and once at default humidity, diffs the resulting
temp_map. The remaining diff after removing the humidity-driven component
is attributable to latent heat (condensation/evaporation terms in
_run_coupled_atmosphere_simulation) - informs a future manual retune of
LATENT_HEAT_COEFFICIENT (core/weather_generator.py, currently 0.1,
documented as "empirisch, noch nicht abgestimmt"). Prints stats only, no
assertions - delete after use.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.weather_generator import WeatherSystemGenerator
from managers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER

size = 48
rng = np.random.RandomState(5)
x = np.linspace(0, 5, size)
y = np.linspace(0, 5, size)
X, Y = np.meshgrid(x, y)
heightmap = (300 + 200 * np.sin(X) * np.cos(Y) + 40 * rng.randn(size, size)).astype(np.float32)
shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

base_params = {
    'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
    'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
    'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
    'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
    'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
    'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
    'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
    'map_longitude': 0.0,
}

print(f"{'lat':>5} {'humid_offset':>12} {'achieved_humid%':>16} {'temp_mean':>10} {'temp_vs_dry_delta':>18}")

for lat in (5.0, 30.0, 55.0, 80.0):
    results = {}
    achieved_humid = {}
    for label, humid_offset in (("dry", WEATHER.AIR_HUMIDITY_ENTRY["min"]), ("wet", 0.0)):
        dlm = DataLODManager()
        dlm.set_map_distance_km(15.0)
        geo = WeatherSystemGenerator(map_seed=31, data_lod_manager=dlm)
        params = dict(base_params, map_latitude=lat, air_humidity_entry=humid_offset)
        result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
        results[label] = result
        climate_temp, climate_humid_fraction = geo._climate_baseline(lat, 0)
        achieved_humid[label] = float(np.clip(climate_humid_fraction * 100.0 + humid_offset, 0.0, 100.0))

    dry_mean = float(results["dry"].temp_map.mean())
    wet_mean = float(results["wet"].temp_map.mean())
    delta = wet_mean - dry_mean
    print(f"{lat:5.0f} {'min (dry)':>12} {achieved_humid['dry']:16.1f} {dry_mean:10.2f} {'-':>18}")
    print(f"{lat:5.0f} {'0 (wet)':>12} {achieved_humid['wet']:16.1f} {wet_mean:10.2f} {delta:18.2f}")

print("\nDelta = latent-heat-driven temp shift between the two humidity regimes at each")
print("latitude (positive = wetter run is warmer, i.e. condensation warming dominates;")
print("negative = evaporative cooling dominates). Use this magnitude to judge whether")
print("LATENT_HEAT_COEFFICIENT (currently 0.1) needs retuning relative to the new")
print("sensible-heat term's magnitude.")
