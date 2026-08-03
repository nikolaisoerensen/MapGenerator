"""
Path: tests/smoke_test_erosion_hauptschalter.py

Prueft den Hauptschalter value_default.EROSION_AKTIV (eingefuehrt 2026-07-30,
siehe SPEZIFIKATION §8): steht er auf False, muss erosion.hydraulic fuer JEDES
LOD Nullkarten liefern und das Gelaende exakt unveraendert lassen.

Drei Zusicherungen, und die dritte ist die eigentliche - SPEZIFIKATION §5.1.4:
eine Zusicherung, die auch OHNE die Aenderung haelt, prueft nichts.

  1. mit EROSION_AKTIV=False sind alle sieben Karten exakt null
  2. das kombinierte Gelaende ist bitgleich mit dem unerodierten Terrain
  3. GEGENPROBE mit EROSION_AKTIV=True: derselbe Aufruf veraendert das
     Gelaende messbar. Ohne diesen Teil koennte der Test auch dann gruen sein,
     wenn die Erosion aus einem voelig anderen Grund nichts tut.

Aufruf:
    .venv\\Scripts\\python.exe tests/smoke_test_erosion_hauptschalter.py
"""

import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

SIZE = 128
LOD = 3          # 32 -> 64 -> 128


def _aufbau():
    """Terrain + Haerte in einen DataLODManager, wie die echte Pipeline es tut."""
    from managers.data_lod_manager import DataLODManager
    from gui.config.value_default import TERRAIN, EROSION
    from core.terrain_generator import BaseTerrainGenerator
    from core.erosion_generator import ErosionSystemGenerator

    manager = DataLODManager()
    manager.set_map_distance_km(TERRAIN.MAP_DISTANCE_KM["default"])

    parameters = {key.lower(): getattr(TERRAIN, key)["default"] for key in
                  ("AMPLITUDE", "OCTAVES", "FEATURE_SIZE_M", "PERSISTENCE",
                   "LACUNARITY", "REDISTRIBUTE_POWER", "MAP_SEED")}
    parameters["map_size"] = SIZE
    parameters["map_seed"] = 20260730

    terrain = BaseTerrainGenerator(data_lod_manager=manager)
    terrain.set_active_parameters(parameters)
    for node in ("terrain.noise", "terrain.redistribution"):
        manager.set_calculator_target_lod(node, LOD)
    terrain._calc_noise("terrain.noise", LOD)
    terrain._calc_redistribution("terrain.redistribution", LOD)

    heightmap = manager.get_calculator_output(
        "terrain.redistribution", "heightmap", LOD)
    assert heightmap.shape == (SIZE, SIZE), (
        "Werkzeug liefert %s statt %dx%d - §5.2, build_terrain(512) gab schon "
        "einmal still ein 256er Array zurueck" % (heightmap.shape, SIZE, SIZE))

    # Haerte gleichmaessig: hier wird die Erosion geprueft, nicht die Geologie.
    manager.set_calculator_output(
        "geology.hardness", LOD,
        {"hardness_map": np.full((SIZE, SIZE), 0.5, dtype=np.float32)})

    erosion_parameter = {key.lower(): getattr(EROSION, key)["default"]
                         for key in dir(EROSION) if key.isupper()}
    erosion_parameter["thermal_variant"] = "gather"
    # Kurz halten - geprueft wird OB die Erosion wirkt, nicht wie gut.
    erosion_parameter["max_steps"] = 300

    generator = ErosionSystemGenerator(data_lod_manager=manager)
    generator.set_active_parameters(erosion_parameter)
    manager.set_calculator_target_lod("erosion.hydraulic", LOD)
    return manager, generator, heightmap


def lauf():
    import gui.config.value_default as vd
    from core.erosion_generator import ErosionSystemGenerator

    original = vd.EROSION_AKTIV
    fehler = []
    try:
        # ---------- 1 + 2: Schalter AUS ----------
        vd.EROSION_AKTIV = False
        manager, generator, terrain_hoehe = _aufbau()
        generator._calc_hydraulic("erosion.hydraulic", LOD)

        for key in ErosionSystemGenerator.EROSION_DATA_KEYS:
            karte = manager.get_calculator_output("erosion.hydraulic", key, LOD)
            if karte is None:
                fehler.append("Karte %s fehlt ganz" % key)
            elif karte.shape != (SIZE, SIZE):
                fehler.append("Karte %s hat Form %s statt (%d,%d)"
                              % (key, karte.shape, SIZE, SIZE))
            elif np.any(karte != 0.0):
                fehler.append("Karte %s ist nicht null (max %.6g)"
                              % (key, np.abs(karte).max()))
        print("1. sieben Nullkarten in richtiger Form ......... %s"
              % ("ok" if not fehler else "FEHLER"))

        kombiniert = manager.get_calculator_combined_heightmap(LOD)
        abweichung = float(np.abs(kombiniert - terrain_hoehe).max())
        if abweichung != 0.0:
            fehler.append("Gelaende weicht um %.6g m ab, erwartet 0" % abweichung)
        print("2. Gelaende bitgleich mit dem unerodierten ..... %s (max %.3g m)"
              % ("ok" if abweichung == 0.0 else "FEHLER", abweichung))

        # ---------- 3: GEGENPROBE, Schalter AN ----------
        vd.EROSION_AKTIV = True
        manager2, generator2, terrain_hoehe2 = _aufbau()
        generator2._calc_hydraulic("erosion.hydraulic", LOD)
        kombiniert2 = manager2.get_calculator_combined_heightmap(LOD)
        wirkung = float(np.abs(kombiniert2 - terrain_hoehe2).max())
        if wirkung <= 0.0:
            fehler.append(
                "GEGENPROBE: mit EROSION_AKTIV=True aendert sich das Gelaende "
                "auch nicht - der Test oben belegt dann nichts")
        print("3. Gegenprobe: mit True wirkt die Erosion ...... %s (max %.1f m)"
              % ("ok" if wirkung > 0.0 else "FEHLER", wirkung))
    finally:
        vd.EROSION_AKTIV = original

    print()
    if fehler:
        for f in fehler:
            print("  FEHLER: %s" % f)
        return 1
    print("Alle drei Zusicherungen erfuellt. EROSION_AKTIV steht auf %s."
          % original)
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
