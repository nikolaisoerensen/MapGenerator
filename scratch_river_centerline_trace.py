"""
Throwaway headless smoke check for MapDisplay2D._trace_river_centerlines()
(gui/widgets/map_display_2d.py) - isolierte Logikpruefung ohne Qt-Instanz,
da die Methode nur numpy-Arrays liest/self nicht sonst benoetigt.
Nicht Teil der Test-Suite, siehe CLAUDE.md smoke-test Konvention.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from gui.widgets.map_display_2d import MapDisplay2D


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


class _Stub:
    RIVER_CLASSES = MapDisplay2D.RIVER_CLASSES
    RIVER_MIN_CHAIN_LENGTH = MapDisplay2D.RIVER_MIN_CHAIN_LENGTH
    _trace_river_centerlines = MapDisplay2D._trace_river_centerlines


def run_diagonal_river_traced_as_single_chain():
    size = 10
    water_biomes = np.zeros((size, size), dtype=np.int32)
    heightmap = np.zeros((size, size), dtype=np.float32)
    # Diagonale Kette von (0,0) [hoch] nach (7,7) [niedrig] -> ein einziger
    # zusammenhaengender Fluss-Pfad, klassisches 45°-Blockmuster.
    for i in range(8):
        water_biomes[i, i] = 2  # river
        heightmap[i, i] = 100.0 - i * 10.0

    stub = _Stub()
    paths = stub._trace_river_centerlines(water_biomes, heightmap)

    ok = check(f"genau ein Pfad gefunden (gefunden: {len(paths)})", len(paths) == 1)
    if paths:
        rows, cols = paths[0]
        ok &= check(f"Pfadlaenge = 8 Pixel (gefunden: {len(rows)})", len(rows) == 8)
        ok &= check("Pfad beginnt an der Quelle (0,0)", rows[0] == 0 and cols[0] == 0)
        ok &= check("Pfad endet am tiefsten Punkt (7,7)", rows[-1] == 7 and cols[-1] == 7)
    return ok


def run_river_stops_at_lake():
    size = 10
    water_biomes = np.zeros((size, size), dtype=np.int32)
    heightmap = np.zeros((size, size), dtype=np.float32)
    for i in range(5):
        water_biomes[i, i] = 2
        heightmap[i, i] = 100.0 - i * 10.0
    water_biomes[5, 5] = 4  # lake
    heightmap[5, 5] = 40.0
    # See-Randzelle, tiefer als der See selbst, damit die Kette nicht durch
    # den gesamten (flachen) See weiterverfolgt wird.
    water_biomes[6, 6] = 4
    heightmap[6, 6] = 5.0

    stub = _Stub()
    paths = stub._trace_river_centerlines(water_biomes, heightmap)

    ok = check(f"genau ein Pfad gefunden (gefunden: {len(paths)})", len(paths) == 1)
    if paths:
        rows, cols = paths[0]
        ok &= check(f"Kette endet an der ersten Seezelle (5,5), Laenge=6 "
                    f"(gefunden: letzter Punkt=({rows[-1]},{cols[-1]}), Laenge={len(rows)})",
                    rows[-1] == 5 and cols[-1] == 5 and len(rows) == 6)
    return ok


def run_two_tributaries_merge():
    size = 10
    water_biomes = np.zeros((size, size), dtype=np.int32)
    heightmap = np.full((size, size), 200.0, dtype=np.float32)
    # Zwei Quellen, die in einer gemeinsamen Fluss-Zelle zusammenlaufen.
    water_biomes[0, 2] = 2
    heightmap[0, 2] = 100.0
    water_biomes[1, 3] = 2
    heightmap[1, 3] = 90.0
    water_biomes[0, 4] = 2
    heightmap[0, 4] = 100.0
    for i, (r, c, h) in enumerate([(2, 4, 70.0), (3, 5, 60.0), (4, 6, 50.0)]):
        water_biomes[r, c] = 2
        heightmap[r, c] = h

    stub = _Stub()
    paths = stub._trace_river_centerlines(water_biomes, heightmap, min_length=1)

    ok = check(f"zwei Quell-Zuflüsse erkannt, die in den Hauptstrom münden "
               f"(gefunden: {len(paths)} Pfade)", len(paths) == 2)
    ok &= check("kein NaN/leere Arrays", all(
        np.all(np.isfinite(r)) and np.all(np.isfinite(c)) for r, c in paths))
    return ok


if __name__ == "__main__":
    results = {
        "diagonal_river_traced_as_single_chain": run_diagonal_river_traced_as_single_chain(),
        "river_stops_at_lake": run_river_stops_at_lake(),
        "two_tributaries_merge": run_two_tributaries_merge(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
