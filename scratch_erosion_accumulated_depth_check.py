"""
Throwaway diagnostic (NOT a smoke test) - misst, wieviele Meter Erosion/
Sedimentation TATSAECHLICH ueber eine volle Generierung (alle LOD-Stufen,
default map_size=128 -> LOD1=32px, LOD2=64px, LOD3=128px, siehe
HydrologySystemGenerator._get_lod_size()) akkumulieren, im Vergleich zur
Gelaende-Hoehenspanne. Nutzer-Frage: "warum sehe ich keine eingeschnittenen
Taeler/flachen Ebenen - zu wenige Iterationen?" Prueft direkt, ob die
akkumulierte Tiefe ueberhaupt gross genug waere, um bei typischen
Gelaende-Hoehen (hunderte Meter) sichtbar zu sein - unabhaengig von der
Iterationszahl fuer Sediment-Transport (die nur die RAEUMLICHE VERTEILUNG,
nicht die GESAMTMENGE der Erosion pro Durchlauf steuert).
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.water_generator import (
    LakeDetectionSystem, FlowNetworkBuilder, ManningFlowCalculator, ErosionSedimentationSystem,
    compute_full_watershed,
)

LOD_SIZES = [32, 64, 128]
LOD_ITERATIONS = [
    {"flow": 50, "sediment": 3, "manning": 5},
    {"flow": 100, "sediment": 5, "manning": 10},
    {"flow": 200, "sediment": 7, "manning": 15},
]


def make_terrain(size, seed=21):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 6, size)
    y = np.linspace(0, 6, size)
    X, Y = np.meshgrid(x, y)
    heightmap = (200.0 + 700.0 * np.exp(-((X - 3) ** 2 + (Y - 3) ** 2) / 3.0)
                 + 30.0 * rng.randn(size, size)).astype(np.float32)
    return heightmap


def slopemap_from_heightmap(heightmap, world_size_km=10.0):
    size = heightmap.shape[0]
    spacing = world_size_km * 1000.0 / size
    slopemap = np.zeros((size, size, 2), dtype=np.float32)
    slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5 / spacing
    slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5 / spacing
    return slopemap


accumulated_erosion = None
accumulated_sedimentation = None

lake_system = LakeDetectionSystem(lake_volume_threshold=0.02)
flow_builder = FlowNetworkBuilder(rain_threshold=0.2, river_abundance=0.3)
manning = ManningFlowCalculator(manning_coefficient=0.03)
erosion_system = ErosionSedimentationSystem(erosion_strength=2.5, sediment_capacity_factor=0.0001,
                                             settling_velocity=0.1)

for lod_index, size in enumerate(LOD_SIZES):
    heightmap = make_terrain(size)
    precip_map = np.full((size, size), 3.0, dtype=np.float32)
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)
    slopemap = slopemap_from_heightmap(heightmap)
    lod_iterations = LOD_ITERATIONS[lod_index]

    lake_map, _ = lake_system.detect_lakes(heightmap, {})
    full_basin_map, _ = compute_full_watershed(heightmap, smoothing_sigma=0.0)
    flow_accumulation, _ = flow_builder.build_flow_network(
        heightmap, precip_map, lake_map, full_basin_map, {}, lod_iterations)
    flow_speed, cross_section, water_depth = manning.calculate_flow_properties(
        flow_accumulation, slopemap, heightmap, {"stream_threshold": 2.0}, lod_iterations)
    flow_directions = flow_builder._calculate_steepest_descent(heightmap)
    flow_directions = flow_builder._redirect_basin_flow_to_spill(heightmap, full_basin_map, flow_directions)

    erosion_map, sedimentation_map = erosion_system.simulate_erosion_sedimentation(
        flow_accumulation, flow_speed, flow_directions, hardness_map, {}, lod_iterations,
        heightmap=heightmap, slopemap=slopemap, water_depth=water_depth)

    # Wie _calc_erosion_sedimentation(): auf die Zielgroesse resamplen und
    # ADDITIV mit dem Vorlauf akkumulieren.
    if accumulated_erosion is not None:
        prev_e = np.array([[accumulated_erosion[int(yy * accumulated_erosion.shape[0] / size),
                                                  int(xx * accumulated_erosion.shape[1] / size)]
                             for xx in range(size)] for yy in range(size)], dtype=np.float32)
        prev_s = np.array([[accumulated_sedimentation[int(yy * accumulated_sedimentation.shape[0] / size),
                                                        int(xx * accumulated_sedimentation.shape[1] / size)]
                             for xx in range(size)] for yy in range(size)], dtype=np.float32)
        accumulated_erosion = erosion_map + prev_e
        accumulated_sedimentation = sedimentation_map + prev_s
    else:
        accumulated_erosion = erosion_map.copy()
        accumulated_sedimentation = sedimentation_map.copy()

    print(f"--- LOD Stufe {lod_index + 1} ({size}px) ---")
    print(f"  Terrain-Hoehenspanne: {heightmap.min():.1f}m - {heightmap.max():.1f}m "
          f"(Relief {heightmap.max() - heightmap.min():.1f}m)")
    print(f"  diese Stufe: erosion mean={erosion_map.mean():.5f}m max={erosion_map.max():.5f}m, "
          f"sedimentation mean={sedimentation_map.mean():.5f}m max={sedimentation_map.max():.5f}m")
    print(f"  AKKUMULIERT bis hier: erosion max={accumulated_erosion.max():.5f}m "
          f"({100 * accumulated_erosion.max() / (heightmap.max() - heightmap.min()):.3f}% des Reliefs), "
          f"sedimentation max={accumulated_sedimentation.max():.5f}m")

print("\n=== FAZIT ===")
final_relief = float(heightmap.max() - heightmap.min())
print(f"Nach 3 vollen LOD-Durchlaeufen (32->64->128px, Default map_size): "
      f"max. akkumulierte Erosion = {accumulated_erosion.max():.4f}m")
print(f"Terrain-Relief in diesem Szenario: {final_relief:.1f}m")
print(f"Verhaeltnis: {100 * accumulated_erosion.max() / final_relief:.3f}% - "
      f"das ist der Anteil des Gelaendes, der ueberhaupt sichtbar 'eingeschnitten' werden koennte.")
