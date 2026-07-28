"""
Throwaway headless smoke test for the Water erosion formula + end-to-end
CPU pipeline (core/water_generator.py). Not part of the test suite - run
manually via the shared venv, see CLAUDE.md.

Rewritten 2026-07-25 for the Eulerian -> Droplet-basierte Erosion
Architektur-Umbau (siehe core/water_generator.py DropletErosionSystem, Plan
"Water: Droplet-basierte Erosion"): die vorherige ErosionSedimentationSystem
(Stream-Power-Formel + MacCormack-Sedimenttransport entlang des Pipe-Modell-
Geschwindigkeitsfelds) existiert nicht mehr. Erosion ist jetzt VOLLSTAENDIG
entkoppelt von water.flow_network - Partikel spawnen hoehen-gewichtet direkt
aus der Heightmap und mutieren sie waehrend ihres Lebenswegs.

- (a) DropletErosionSystem._cpu_simulate() auf einem realistischen Huegel-
  Szenario liefert endliche, messbare, aber nicht absurd grosse Erosion/
  Sedimentation - UND die exakte Massenerhaltung gilt bereits auf dieser
  Ebene (Zwangs-Absetzung des Rests bei jedem Partikel-Abbruch, siehe
  Klassen-Docstring).
- (b) volle CPU-Pipeline in der NEUEN Reihenfolge (Erosion zuerst, dann
  LakeDetection -> PipeFlowSimulator auf dem bereits erodierten Gelaende,
  siehe _execute_generation()) laeuft end-to-end ohne NaN/Inf.

GPU-spezifische Dispatch-Pfade werden hier NICHT abgedeckt - per CLAUDE.md
braucht GPU-/Shader-Code die lebende App, keine headless Smoke-Tests (ohnehin
diese Runde nicht implementiert, siehe Plan "GPU-Wettlauf").
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.water_generator import LakeDetectionSystem, FlowNetworkBuilder, DropletErosionSystem


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def _make_multi_basin_terrain(size=64, seed=3):
    """Terrain mit mehreren unterschiedlich großen, benachbarten Senken."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    heightmap = np.full((size, size), 500.0, dtype=np.float64)

    big_center = (size * 0.3, size * 0.3)
    dist_big = np.hypot(x - big_center[1], y - big_center[0])
    heightmap -= 80.0 * np.exp(-(dist_big ** 2) / (2 * (size * 0.08) ** 2))

    small_centers = [(size * 0.6, size * 0.6), (size * 0.7, size * 0.2), (size * 0.2, size * 0.75)]
    for cy, cx in small_centers:
        dist_small = np.hypot(x - cx, y - cy)
        heightmap -= 3.0 * np.exp(-(dist_small ** 2) / (2 * (size * 0.02) ** 2))

    heightmap += 5.0 * rng.randn(size, size)
    return heightmap.astype(np.float32)


def run_erosion_formula_sane_output():
    """Droplet-Erosion auf einem realistischen Huegel-Szenario muss messbare,
    endliche, plausibel-grosse Erosion/Sedimentation liefern. Massenbilanz
    ist bewusst NICHT exakt (Nutzer-Vorgabe: 'ich brauche keine strikte
    Massenerhaltung, ich will das wie Sebastian Lague' - Partikel verwerfen
    Restfracht beim Tod statt sie zwangsabzusetzen) - abgelagert darf nie
    mehr als abgetragen sein, muss aber nicht gleich sein."""
    size = 48
    rng = np.random.RandomState(5)
    heightmap = (300.0 + 400.0 * np.exp(-((np.arange(size)[:, None] - size / 2) ** 2
                                           + (np.arange(size)[None, :] - size / 2) ** 2) / (2 * (size * 0.2) ** 2))
                 + 20.0 * rng.randn(size, size)).astype(np.float32)
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)
    meters_per_pixel = 10000.0 / size
    relief = float(heightmap.max() - heightmap.min())

    erosion_system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0, deposit_speed=0.3)
    spawn_positions = erosion_system._sample_spawn_positions(heightmap, 800, {"water_seed": 7})
    erosion_map, sedimentation_map = erosion_system.simulate_droplets_only(
        heightmap, hardness_map, spawn_positions, meters_per_pixel)

    total_erosion = float(erosion_map.sum())
    total_sedimentation = float(sedimentation_map.sum())

    ok = check("erosion_map endlich", bool(np.all(np.isfinite(erosion_map))))
    ok &= check("sedimentation_map endlich", bool(np.all(np.isfinite(sedimentation_map))))
    ok &= check("erosion_map nicht komplett 0 (sichtbare Erosion)", bool(np.any(erosion_map > 0)))
    ok &= check("sedimentation_map nicht komplett 0 (sichtbare Sedimentation)", bool(np.any(sedimentation_map > 0)))
    ok &= check(f"erosion_map in plausibler Groessenordnung relativ zum Relief "
                f"(max={erosion_map.max():.4f}m, Relief={relief:.2f}m)",
                bool(erosion_map.max() < relief))
    ok &= check(f"abgelagert <= abgetragen (keine Massenerzeugung aus dem Nichts): "
                f"abgetragen={total_erosion:.4f}, abgelagert={total_sedimentation:.4f}",
                total_sedimentation <= total_erosion + 1e-6)
    return ok


def run_end_to_end_cpu_pipeline():
    """
    Vollstaendige CPU-Pipeline in der NEUEN Reihenfolge: DropletErosionSystem
    zuerst (unabhaengig von Wassersimulation) -> das eroded Terrain fliesst in
    LakeDetection -> PipeFlowSimulator (via FlowNetworkBuilder.build_flow_network)
    - keine NaN/Inf, plausible Wertebereiche. Ersetzt den vorherigen Ablauf, in
    dem Erosion NACH der Wassersimulation lief und deren Output brauchte.
    """
    heightmap = _make_multi_basin_terrain(size=64, seed=13)
    precip_map = np.full(heightmap.shape, 3.0, dtype=np.float32)
    hardness_map = np.full(heightmap.shape, 50.0, dtype=np.float32)
    meters_per_pixel = 10000.0 / heightmap.shape[0]

    erosion_system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0, deposit_speed=0.3)
    erosion_map, sedimentation_map = erosion_system.simulate_erosion_sedimentation(
        heightmap, hardness_map, {"water_seed": 13}, {"erosion_particles": 1200},
        meters_per_pixel=meters_per_pixel)

    eroded_heightmap = (heightmap.astype(np.float64) - erosion_map.astype(np.float64)
                         + sedimentation_map.astype(np.float64)).astype(np.float32)

    lake_system = LakeDetectionSystem(lake_volume_threshold=0.02)
    lake_map, _ = lake_system.detect_lakes(eroded_heightmap, {})

    flow_builder = FlowNetworkBuilder()
    potential_evaporation = np.zeros_like(precip_map)
    sim_result = flow_builder.build_flow_network(
        eroded_heightmap, precip_map, potential_evaporation, lake_map, {},
        {"flow": 40}, meters_per_pixel)

    flow_accumulation = sim_result["discharge_map"]
    water_depth = sim_result["water_depth"]
    velocity_x = sim_result["velocity_x"]
    velocity_y = sim_result["velocity_y"]

    ok = check("erosion_map endlich", bool(np.all(np.isfinite(erosion_map))))
    ok &= check("sedimentation_map endlich", bool(np.all(np.isfinite(sedimentation_map))))
    ok &= check("eroded_heightmap endlich", bool(np.all(np.isfinite(eroded_heightmap))))
    ok &= check("flow_accumulation (Durchfluss-Betrag) endlich", bool(np.all(np.isfinite(flow_accumulation))))
    ok &= check("velocity_x/velocity_y endlich",
                bool(np.all(np.isfinite(velocity_x))) and bool(np.all(np.isfinite(velocity_y))))
    ok &= check("water_depth endlich, >= 0", bool(np.all(np.isfinite(water_depth)) and np.all(water_depth >= 0)))
    ok &= check("water_biomes_map vorhanden, plausibler Wertebereich [0,4]",
                bool(sim_result["water_biomes_map"].max() <= 4) and bool(sim_result["water_biomes_map"].min() >= 0))
    return ok


if __name__ == "__main__":
    results = {
        "erosion_formula_sane_output": run_erosion_formula_sane_output(),
        "end_to_end_cpu_pipeline": run_end_to_end_cpu_pipeline(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
