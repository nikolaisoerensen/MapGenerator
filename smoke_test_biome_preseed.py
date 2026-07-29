"""
Throwaway headless smoke test for the Biome-Preseed + biom-abhängige
Boden-Feuchte-Kapazität-Änderungen (core/biome_generator.py,
core/water_generator.py, gui/OldManagers/calculator_graph.py). Not part of
the test suite - run manually via the shared venv, see CLAUDE.md.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.biome_generator import BiomeClassificationSystem
from core.water_generator import HydrologySystemGenerator, _BIOME_MOISTURE_CAPACITY, _BIOME_EVAPORATION_FACTOR
from gui.OldManagers.calculator_graph import CALCULATOR_GRAPH
from gui.OldManagers.data_lod_manager import DataLODManager


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def run_calculator_graph_sanity():
    """biome.preseed_hint muss registriert sein und zyklenfrei bleiben,
    water.soil_moisture muss auch von weather.temperature abhaengen.

    Die Zusicherung lautete bis 2026-07-28 woertlich "nur terrain.*". Das war
    eine zu enge Formulierung des eigentlichen Ziels: der Vorab-Schaetzwert
    darf nicht von water.* oder biome.* abhaengen, sonst entsteht genau der
    Zyklus, den er aufloesen soll (Bodenfeuchte braucht den Biomtyp, Biomtyp
    braucht die Bodenfeuchte). erosion.slope ist unbedenklich - die Erosion
    laeuft vor Water und Biome - und fachlich noetig, weil der Schaetzwert die
    Steilheit bewertet und die erst durch die Erosion entsteht.
    """
    ok = check("biome.preseed_hint im CALCULATOR_GRAPH registriert",
               "biome.preseed_hint" in CALCULATOR_GRAPH)
    preseed_deps = set(CALCULATOR_GRAPH["biome.preseed_hint"].depends_on)
    ok &= check(f"preseed_hint deps={preseed_deps} - kein water./biome.-Zyklus",
                not any(dep.startswith(("water.", "biome."))
                        for dep in preseed_deps))
    ok &= check("preseed_hint benutzt den Slope NACH der Erosion",
                "erosion.slope" in preseed_deps)
    soil_deps = set(CALCULATOR_GRAPH["water.soil_moisture"].depends_on)
    # water.manning_flow ergaenzt (2026-07-27): water.soil_moisture liest die
    # GEMALTE Klassifikation von dort und die Zentrallinie von
    # water.flow_network - siehe core/water_generator.py._calc_manning_flow().
    # terrain.shadow ergaenzt (2026-07-27): die Besonnung steuert, wie stark
    # ein Hang austrocknet - verschattete Nordhaenge bleiben feuchter, siehe
    # core/water_generator.py._apply_biome_soil_drying().
    ok &= check(f"water.soil_moisture deps={soil_deps} (inkl. weather.temperature, "
                f"beider Wasser-Klassifikationen und der Besonnung)",
                soil_deps == {"water.flow_network", "water.manning_flow",
                              "weather.temperature", "terrain.shadow"})
    return ok


def run_preseed_hint_ridge_test():
    """Nord-Sued-Ridge: Suedhang muss trockeneres Biom-Index bekommen als
    Nordhang (row height-1 = Norden, verifiziert diese Session)."""
    size = 48
    y = np.arange(size)[:, None] * np.ones((1, size))
    heightmap = (400.0 - np.abs(y - size / 2) * 8.0).astype(np.float32)
    grad_y, grad_x = np.gradient(heightmap)
    slopemap = np.stack([grad_x, grad_y], axis=-1).astype(np.float32)

    dlm = DataLODManager()
    dlm.set_map_latitude(48.0)
    dlm.set_calculator_output("terrain.redistribution", 3, {"heightmap": heightmap})
    # erosion.slope, nicht terrain.slope: von dort liest _calc_preseed_hint()
    # seit 2026-07-28 (Hangneigung auf dem erodierten Gelaende).
    dlm.set_calculator_output("erosion.slope", 3, {"slopemap": slopemap})

    biome = BiomeClassificationSystem(data_lod_manager=dlm)
    biome._calc_preseed_hint("biome.preseed_hint", 3)
    preseed = dlm.get_calculator_output("biome.preseed_hint", "preseed_biome_map", 3)

    south_id = int(preseed[size // 4, size // 2])
    north_id = int(preseed[3 * size // 4, size // 2])
    south_capacity = _BIOME_MOISTURE_CAPACITY[south_id]
    north_capacity = _BIOME_MOISTURE_CAPACITY[north_id]
    return check(f"Suedhang-Biom (id={south_id}, Kapazitaet={south_capacity}) trockener als "
                 f"Nordhang-Biom (id={north_id}, Kapazitaet={north_capacity})",
                 south_capacity < north_capacity)


def run_soil_drying_desert_vs_swamp():
    """Bei identischem Wasserangebot und Waerme muss Wueste staerker
    austrocknen und niedriger gedeckelt sein als Sumpf."""
    size = 16
    dlm = DataLODManager()
    water = HydrologySystemGenerator(data_lod_manager=dlm)

    soil_moist = np.full((size, size), 60.0, dtype=np.float32)
    temp_map = np.full((size, size), 30.0, dtype=np.float32)
    water_mask = np.zeros((size, size), dtype=np.uint8)

    desert_biome = np.full((size, size), 6, dtype=np.uint8)
    swamp_biome = np.full((size, size), 12, dtype=np.uint8)

    dried_desert = water._apply_biome_soil_drying(soil_moist.copy(), temp_map, desert_biome, water_mask)
    dried_swamp = water._apply_biome_soil_drying(soil_moist.copy(), temp_map, swamp_biome, water_mask)

    ok = check(f"Wueste ({dried_desert.mean():.1f}) trockener als Sumpf ({dried_swamp.mean():.1f})",
               dried_desert.mean() < dried_swamp.mean())
    ok &= check(f"Wueste unter ihrer Kapazitaet (max={dried_desert.max():.1f} <= 20.0)",
                dried_desert.max() <= 20.0 + 1e-3)
    return ok


def run_water_tiles_dont_dry():
    """Echte Wasser-Kacheln (water_mask_source > 0) duerfen durch den
    Trocknungs-Term NICHT reduziert werden - nur die diffuse Umgebung."""
    size = 8
    dlm = DataLODManager()
    water = HydrologySystemGenerator(data_lod_manager=dlm)

    soil_moist = np.full((size, size), 100.0, dtype=np.float32)
    temp_map = np.full((size, size), 40.0, dtype=np.float32)  # extreme Hitze
    water_mask = np.ones((size, size), dtype=np.uint8)  # alles "echtes Wasser"
    desert_biome = np.full((size, size), 6, dtype=np.uint8)

    result = water._apply_biome_soil_drying(soil_moist.copy(), temp_map, desert_biome, water_mask)
    return check(f"Wasser-Kacheln bleiben bei 100 trotz Wueste+Hitze (min={result.min():.1f})",
                 bool(np.all(result >= 99.9)))


def run_calculator_node_integration():
    """_calc_soil_moisture() end-to-end mit preseed_hint als Biom-Quelle -
    keine Exceptions, Kapazitaets-Clip greift."""
    size = 16
    dlm = DataLODManager()
    dlm.set_map_latitude(48.0)

    heightmap = np.full((size, size), 100.0, dtype=np.float32)
    flow_acc = np.zeros((size, size), dtype=np.float32)
    water_biomes = np.zeros((size, size), dtype=np.uint8)
    temp_map = np.full((size, size), 30.0, dtype=np.float32)
    preseed = np.full((size, size), 6, dtype=np.uint8)  # desert

    dlm.set_calculator_output("terrain.redistribution", 1, {"heightmap": heightmap})
    # water.flow_network liefert die ZENTRALLINIE, water.manning_flow die
    # GEMALTE Fassung - _calc_soil_moisture liest beide (siehe dortigen
    # Docstring). Im Test dieselbe Karte fuer beide, die Unterscheidung ist
    # hier nicht der Prueffokus.
    dlm.set_calculator_output("water.flow_network", 1,
                               {"flow_accumulation": flow_acc, "water_biomes_map": water_biomes})
    dlm.set_calculator_output("water.manning_flow", 1, {"water_biomes_map": water_biomes})
    dlm.set_calculator_output("weather.temperature", 1, {"temp_map": temp_map})
    dlm.set_calculator_output("biome.preseed_hint", 1, {"preseed_biome_map": preseed})

    water = HydrologySystemGenerator(data_lod_manager=dlm)
    water.set_active_parameters({})
    try:
        water._calc_soil_moisture("water.soil_moisture", 1)
    except Exception as e:
        print(f"[FAIL] _calc_soil_moisture raised: {e}")
        return False

    result = dlm.get_calculator_output("water.soil_moisture", "soil_moist_map", 1)
    return check(f"soil_moist_map produziert, max={result.max():.1f} <= 20.0 (Wueste-Kapazitaet)",
                 result is not None and result.max() <= 20.0 + 1e-3)


if __name__ == "__main__":
    results = {
        "calculator_graph_sanity": run_calculator_graph_sanity(),
        "preseed_hint_ridge_test": run_preseed_hint_ridge_test(),
        "soil_drying_desert_vs_swamp": run_soil_drying_desert_vs_swamp(),
        "water_tiles_dont_dry": run_water_tiles_dont_dry(),
        "calculator_node_integration": run_calculator_node_integration(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
