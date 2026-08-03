"""
Throwaway diagnostic (NOT a smoke test) - measures the new stream-power
erosion formula's shear_stress/erosion_rate magnitude after replacing the
old velocity-derived water_depth/slope placeholders with real values (Manning
channel depth + terrain.slope), see core/water_generator.py
ErosionSedimentationSystem._calculate_stream_power_erosion(). Goal: retune
EROSION_RATE_SCALE (currently inherited unchanged at 1e-6 from the old,
now-replaced formula) so erosion_rate lands in a similar few-cm-to-tens-of-cm
-per-generation-pass ballpark as originally intended (min(...,0.1) cap).
Prints stats only, no assertions - delete after use.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.water_generator import ManningFlowCalculator, ErosionSedimentationSystem, FlowNetworkBuilder, \
    LakeDetectionSystem
from managers.data_lod_manager import DataLODManager

size = 96
rng = np.random.RandomState(11)
x = np.linspace(0, 6, size)
y = np.linspace(0, 6, size)
X, Y = np.meshgrid(x, y)
heightmap = (200.0 + 800.0 * np.exp(-((X - 3) ** 2 + (Y - 3) ** 2) / 4.0)
             + 40.0 * rng.randn(size, size)).astype(np.float32)

# Reale slopemap (zentrale Differenzen, wie terrain_generator._calculate_slopes_vectorized)
spacing = 10000.0 / size
slopemap = np.zeros((size, size, 2), dtype=np.float32)
slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5 / spacing
slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5 / spacing

precip_map = np.full((size, size), 3.0, dtype=np.float32)
hardness_map = np.full((size, size), 50.0, dtype=np.float32)

dlm = DataLODManager()
dlm.set_map_distance_km(10.0)

lake_system = LakeDetectionSystem(lake_volume_threshold=0.02)
lake_map, _ = lake_system.detect_lakes(heightmap, {})

flow_builder = FlowNetworkBuilder(rain_threshold=0.2, river_abundance=0.3)
from core.water_generator import compute_full_watershed
full_basin_map, _ = compute_full_watershed(heightmap, smoothing_sigma=0.0)
flow_accumulation, water_biomes_map = flow_builder.build_flow_network(
    heightmap, precip_map, lake_map, full_basin_map, {}, {"flow": 60})

manning = ManningFlowCalculator(manning_coefficient=0.03)
flow_speed, cross_section, water_depth = manning.calculate_flow_properties(
    flow_accumulation, slopemap, heightmap, {"stream_threshold": 2.0}, {"manning": 8})

print(f"flow_accumulation: mean={flow_accumulation.mean():.2f} max={flow_accumulation.max():.2f} "
      f"p90={np.percentile(flow_accumulation, 90):.2f}")
wet = flow_speed > 0.1
print(f"wet pixels: {wet.sum()} / {size * size} ({100 * wet.sum() / (size * size):.1f}%)")
if wet.sum() > 0:
    print(f"flow_speed (wet): mean={flow_speed[wet].mean():.3f} max={flow_speed[wet].max():.3f}")
    print(f"water_depth (wet): mean={water_depth[wet].mean():.3f} max={water_depth[wet].max():.3f}")
    slope_mag = np.sqrt(slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2)
    print(f"slope magnitude (wet): mean={slope_mag[wet].mean():.4f} max={slope_mag[wet].max():.4f}")

    rho_water, gravity = 1000.0, 9.81
    shear = rho_water * gravity * water_depth * np.maximum(0.001, slope_mag)
    critical = hardness_map * 10.0
    print(f"shear_stress (wet): mean={shear[wet].mean():.1f} max={shear[wet].max():.1f} "
          f"p90={np.percentile(shear[wet], 90):.1f}")
    print(f"critical_shear: {critical[0, 0]:.1f} (constant, hardness=50)")
    exceeds = wet & (shear > critical)
    print(f"pixels exceeding critical shear: {exceeds.sum()} ({100 * exceeds.sum() / (size * size):.2f}%)")

erosion_system = ErosionSedimentationSystem(erosion_strength=2.5, sediment_capacity_factor=0.0001,
                                             settling_velocity=0.1)
for scale in (1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
    erosion_system.EROSION_RATE_SCALE = scale
    erosion_map = erosion_system._calculate_stream_power_erosion(
        flow_accumulation, flow_speed, slopemap, water_depth, hardness_map)
    nonzero = erosion_map > 0
    capped = erosion_map >= 0.0999
    if nonzero.sum() > 0:
        print(f"scale={scale:.0e}: nonzero={nonzero.sum():5d} mean={erosion_map[nonzero].mean():.5f} "
              f"max={erosion_map.max():.5f} capped_at_0.1={capped.sum()}")
    else:
        print(f"scale={scale:.0e}: nonzero=0 (nothing erodes)")
