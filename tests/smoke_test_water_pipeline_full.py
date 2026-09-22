"""
Throwaway headless smoke test for the FULL water calculator-node pipeline
(core/water_generator.py HydrologySystemGenerator._execute_generation(), via
calculate_hydrology()) after the D8 -> virtual-pipe hydraulic model rework.
Not part of the test suite - run manually via the shared venv, see CLAUDE.md.

Unlike smoke_test_water_pipe_flow.py/smoke_test_water_drainage_erosion.py/
smoke_test_water_edge_sediment.py (which call the CPU classes directly),
this exercises the actual node-by-node DataLODManager-backed path the live
app uses (_calc_lake_detection -> _calc_flow_network -> _calc_manning_flow ->
_calc_erosion_sedimentation -> _calc_soil_moisture -> _calc_evaporation ->
assemble_water_data) - catches wiring bugs (wrong calculator_id/key lookups,
missing outputs) that direct class-level tests can't see.
"""
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.water_generator import HydrologySystemGenerator, WaterData
from managers.data_lod_manager import DataLODManager


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def _make_terrain(size=48, seed=9):
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    heightmap = (300.0 + 400.0 * np.exp(-((x - size * 0.4) ** 2 + (y - size * 0.4) ** 2)
                                         / (2 * (size * 0.15) ** 2))
                 + 20.0 * rng.randn(size, size)).astype(np.float32)
    return heightmap


def run_full_calculator_pipeline_two_lods():
    """Zwei aufeinanderfolgende LOD-Durchlaeufe (prueft insbesondere die
    depth_state/flux_state-Weiterreichung zwischen LODs, siehe
    _calc_flow_network()-Docstring, und die kumulative Erosion/Sedimentation-
    Akkumulation)."""
    size = 48
    heightmap = _make_terrain(size=size)
    slopemap = np.zeros((size, size, 2), dtype=np.float32)
    spacing = 10000.0 / size
    slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5 / spacing
    slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5 / spacing
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)
    rock_map = np.zeros((size, size, 3), dtype=np.uint8)
    precip_map = np.full((size, size), 4.0, dtype=np.float32)
    temp_map = np.full((size, size), 15.0, dtype=np.float32)
    wind_map = np.zeros((size, size, 2), dtype=np.float32)
    humid_map = np.full((size, size), 50.0, dtype=np.float32)

    dependencies = {
        'heightmap': heightmap, 'slopemap': slopemap, 'hardness_map': hardness_map,
        'rock_map': rock_map, 'precip_map': precip_map, 'temp_map': temp_map,
        'wind_map': wind_map, 'humid_map': humid_map,
    }
    parameters = {
        'lake_volume_threshold': 5000.0,
        'river_abundance': 0.10, 'erosion_strength': 2.5,
        'sediment_capacity_factor': 0.0001, 'evaporation_base_rate': 0.002,
        'diffusion_radius': 2.0, 'settling_velocity': 0.1,
        'thermal_erosion_strength': 1.0,
    }

    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    generator = HydrologySystemGenerator(map_seed=7, data_lod_manager=dlm)

    ok = True
    water_data_1 = generator._execute_generation(1, dependencies, parameters)
    ok &= check("LOD1: WaterData-Objekt zurueckgegeben", isinstance(water_data_1, WaterData))
    ok &= check("LOD1: water_map endlich", bool(np.all(np.isfinite(water_data_1.water_map))))
    # Erosion-Umbau 2026-07-28: erosion_map/sedimentation_map gehoeren zum
    # eigenen Erosion-Generator (core/erosion_generator.py) und werden dort
    # von smoke_test_erosion_field.py geprueft. WaterData fuehrt sie nicht mehr.
    ok &= check("LOD1: soil_moist_map endlich",
                bool(np.all(np.isfinite(water_data_1.soil_moist_map))))
    ok &= check("LOD1: soil_moist_map im Bereich [0,100]",
                bool(water_data_1.soil_moist_map.min() >= 0) and bool(water_data_1.soil_moist_map.max() <= 100.001))
    ok &= check("LOD1: water_biomes_map im Bereich [0,4]",
                bool(water_data_1.water_biomes_map.min() >= 0) and bool(water_data_1.water_biomes_map.max() <= 4))
    ok &= check("LOD1: ocean_outflow endlich, >= 0",
                np.isfinite(water_data_1.ocean_outflow) and water_data_1.ocean_outflow >= 0)

    water_data_2 = generator._execute_generation(2, dependencies, parameters)
    ok &= check("LOD2: WaterData-Objekt zurueckgegeben", isinstance(water_data_2, WaterData))
    ok &= check("LOD2: flow_map endlich", bool(np.all(np.isfinite(water_data_2.flow_map))))
    # Kein Kumulations-Test mehr: Erosion rechnet seit 2026-07-27 NUR in der
    # letzten LOD-Runde (HydrologySystemGenerator._is_final_lod()), frühere
    # Runden liefern bewusst Nullkarten. Ohne gesetztes Ziel-LOD leitet
    # _is_final_lod() es aus der Heightmap-Groesse ab - bei dieser 48px-Karte
    # ist das LOD 1, weshalb LOD1 hier erodiert und LOD2 nicht.
    ok &= check("LOD2: soil_moist_map endlich und nicht-negativ",
                bool(np.all(np.isfinite(water_data_2.soil_moist_map)))
                and float(water_data_2.soil_moist_map.min()) >= 0.0)
    ok &= check("LOD2: water_biomes_map im Bereich [0,4]",
                bool(water_data_2.water_biomes_map.min() >= 0) and bool(water_data_2.water_biomes_map.max() <= 4))
    return ok


if __name__ == "__main__":
    results = {
        "full_calculator_pipeline_two_lods": run_full_calculator_pipeline_two_lods(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
