"""
Path: tests/smoke_test_seenflaeche_messung.py

Ticket #32: `docs/TESTBERICHT.md` meldete eine Seenflaeche von 0,0 % gegen
ein Ziel groesser null. Dieser Test erhebt den IST-Wert neu, auf echten
Kartengroessen (256/512/1024 px, 90_MESSPROTOKOLLE.md §10 - nicht 129/257,
das war schon einmal die Ursache dafuer, dass zehn gruene Tests eine tote
Funktion verdeckt haben, siehe CLAUDE.md).

Gemessen wird nur die fuer den Seenanteil noetige Teilkette (terrain ->
geology-Kette -> erosion.hydraulic -> water.lake_detection), NICHT die volle
39-Knoten-Pipeline - weather/water.flow_network/biome/settlement haengen
nicht an water.lake_detection und wuerden nur Rechenzeit kosten.

Ergebnis ist eine MESSUNG, keine Korrektur (Ticket-Vorgabe): der Test hat
keine Zusicherung gegen ein Zielband, sondern druckt die Zahlen aus, die
dann von Hand ins Ticket und in docs/TESTBERICHT.md uebernommen werden.

DER IRRWEG, DER ZU DER 0,0-%-MELDUNG GEFUEHRT HAT - NICHT WIEDERHOLEN
=====================================================================
Die urspruengliche Messung fragte `(seegrad > 0) & (heightmap > 0)` ab,
also "Seegrad ueber null UND Land". Diese Bedingung ist nicht knapp
danebengegangen, sie ist LEER - und zwar fuer jede Karte, jeden Seed und
jede Groesse. Deshalb kam immer exakt 0,0 % heraus, was wie ein echter
Befund aussah und als solcher in docs/TESTBERICHT.md landete.

Der Grund steht in einer einzigen Zeile, core/terrain_weltkarte.py:

    seegrad_roh = np.where(maske, 0, grad[etikett]).astype(np.int16)

`maske` ist die LANDmaske. Auf Land wird `seegrad` also per Konstruktion
auf 0 gesetzt; `seegrad > 0` kann nur auf Wasser wahr sein. Die beiden
Haelften der Und-Verknuepfung schliessen einander damit aus.

Dazu kommt, dass `seegrad` das Gesuchte ohnehin nicht kennt: es ist die
Zahl der Ringschritte von der naechsten landberuehrenden Zelle nach
aussen (Breitensuche ueber den Voronoi-Nachbarschaftsgraphen, ebd.), und
sie laeuft ueber ALLES Wasser - offenes Meer eingeschlossen. Ein
Binnensee ist daran nicht zu erkennen. `seegrad` beschreibt, wie weit
draussen ein Wasserpixel liegt, nicht, ob es zu einem See gehoert.

Zustaendig ist `water.lake_detection`: dessen `lake_map` traegt je
Seepixel eine Seenummer und -1 ueberall sonst. Das Kriterium unten
(`(lake_map >= 0) & land`) ist deshalb das richtige - `>= 0`, nicht
`> 0`, weil die Nummerierung bei null anfaengt und ein `> 0` still den
ersten See verschluckt.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_seenflaeche_messung.py
"""

import sys
import time

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


KM = 15.0

# _lod_level_to_size() (core/terrain_generator.py) verdoppelt ab 32 px je
# LOD-Stufe (LOD1=32 ... LOD4=256, LOD5=512, LOD6=1024) und deckelt auf
# map_size. Um wirklich bei 256/512/1024 px zu rechnen (nicht nur bis dahin
# GEDECKELT), muss LOD passend zur Zielgroesse gewaehlt werden.
LOD_FUER_GROESSE = {256: 4, 512: 5, 1024: 6}

# Nur der Teilgraph, der water.lake_detection speist (keine 39 Knoten - siehe
# Modul-Docstring). Reihenfolge von Hand aus managers/calculator_graph.py
# abgeschrieben (depends_on-Ketten von terrain.noise bis water.lake_detection).
TEILGRAPH = [
    "terrain.noise", "terrain.redistribution", "terrain.slope",
    "geology.layer_thickness", "geology.tectonic_displacement",
    "geology.outcrop", "geology.intrusions", "geology.sediment_overlay",
    "geology.metamorphic_overprint", "geology.hardness",
    "erosion.hydraulic",
    "water.lake_detection",
]


def _parameter(size, seed):
    import gui.config.value_default as vd

    parameter = {"map_size": size, "map_distance_km": KM, "map_seed": seed}
    for klassenname in ("TERRAIN", "GEOLOGY", "WEATHER", "EROSION", "WATER",
                        "BIOME", "EROSION_FILTER", "RIVER_NETWORK"):
        klasse = getattr(vd, klassenname, None)
        if klasse is None:
            continue
        praefix = {"EROSION_FILTER": "erosion_filter_",
                   "RIVER_NETWORK": "river_"}.get(klassenname, "")
        for name in dir(klasse):
            if not name.isupper():
                continue
            wert = getattr(klasse, name)
            if isinstance(wert, dict) and "default" in wert:
                parameter.setdefault(praefix + name.lower(), wert["default"])
    parameter["thermal_variant"] = "gather"
    parameter["max_steps"] = 200
    return parameter


def _erheben(size, seed):
    from core.terrain_generator import BaseTerrainGenerator
    from core.geology_generator import GeologySystemGenerator
    from core.erosion_generator import ErosionSystemGenerator
    from core.water_generator import HydrologySystemGenerator
    from managers.data_lod_manager import DataLODManager

    lod = LOD_FUER_GROESSE[size]
    manager = DataLODManager()
    manager.set_map_distance_km(KM)
    manager.set_map_seed(seed)
    parameter = _parameter(size, seed)

    gemeinsam = dict(shader_manager=None, data_lod_manager=manager)
    generatoren = {
        "terrain": BaseTerrainGenerator(map_seed=seed, **gemeinsam),
        "geology": GeologySystemGenerator(**gemeinsam),
        "erosion": ErosionSystemGenerator(**gemeinsam),
        "water": HydrologySystemGenerator(**gemeinsam),
    }
    for knoten in TEILGRAPH:
        manager.set_calculator_target_lod(knoten, lod)
    for generator in generatoren.values():
        if hasattr(generator, "set_active_parameters"):
            generator.set_active_parameters(parameter)

    from managers.calculator_graph import CALCULATOR_GRAPH
    for knoten in TEILGRAPH:
        spec = CALCULATOR_GRAPH[knoten]
        generator = generatoren[spec.generator]
        methode = getattr(generator, "_calc_" + knoten.split(".", 1)[1])
        methode(knoten, lod)

    heightmap = manager.get_calculator_output("terrain.redistribution", "heightmap", lod)
    region_map = manager.get_calculator_output("terrain.redistribution", "region_map", lod)
    lake_map = manager.get_calculator_output("water.lake_detection", "lake_map", lod)
    return heightmap, region_map, lake_map


def _regionsnamen():
    from core.terrain_weltkarte import REGIONEN
    return [r["name"] for zeile in REGIONEN for r in zeile]


def lauf():
    _qt()
    regionsnamen = _regionsnamen()
    grössen = (256, 512, 1024)
    seeds = (20260804, 12345, 4242)

    print("Ticket #32 - Seenflaechen-Messung (neu erhoben)\n")
    print("%-6s %-10s %8s %8s %10s %s" % (
        "px", "Seed", "Land-%", "See-%", "Anzahl", "See je Region (Anteil an Landflaeche)"))
    print("-" * 100)

    gesamt = []
    for size in grössen:
        for seed in seeds:
            t0 = time.time()
            heightmap, region_map, lake_map = _erheben(size, seed)
            dauer = time.time() - t0

            land = heightmap > 0
            land_px = int(land.sum())
            see = (lake_map >= 0) & land
            see_px = int(see.sum())
            see_anteil = 100.0 * see_px / land_px if land_px else 0.0
            ids = np.unique(lake_map[see])
            anzahl = int(len(ids))

            je_region = []
            for i, name in enumerate(regionsnamen):
                r_land = land & (region_map == i)
                r_land_px = int(r_land.sum())
                if r_land_px == 0:
                    continue
                r_see_px = int((see & (region_map == i)).sum())
                r_anteil = 100.0 * r_see_px / r_land_px
                je_region.append((name, r_anteil, r_see_px))
            gesamt.append((size, seed, land_px, see_anteil, anzahl, je_region, dauer))

            region_text = ", ".join("%s %.1f%%" % (n, a) for n, a, _ in je_region)
            print("%-6d %-10d %8.2f %8.3f %10d %s  [%.1fs]" % (
                size, seed, 100.0 * land_px / (size * size), see_anteil, anzahl,
                region_text, dauer))

    print()
    print("Regionen mit 0 Seen (ueber alle Groessen/Seeds gemeinsam betrachtet):")
    nullregionen = set(regionsnamen)
    for _, _, _, _, _, je_region, _ in gesamt:
        for name, _, see_px in je_region:
            if see_px > 0:
                nullregionen.discard(name)
    print("   " + (", ".join(sorted(nullregionen)) if nullregionen else "keine"))

    print()
    print("Seenflaeche gemittelt je Kartengroesse (Auflösungsabhaengigkeit, "
          "nicht Teil der Abnahmekriterien, aber auffaellig):")
    for size in grössen:
        werte = [a for s, _, _, a, _, _, _ in gesamt if s == size]
        print("   %5d px: Mittel %.3f%%" % (size, sum(werte) / len(werte)))

    print()
    alte_werte = [a for _, _, _, a, _, _, _ in gesamt]
    print("Alter gemeldeter Wert (docs/TESTBERICHT.md): 0,0%")
    print("Neu gemessen: Minimum %.3f%%, Maximum %.3f%%, Mittel %.3f%%"
          % (min(alte_werte), max(alte_werte), sum(alte_werte) / len(alte_werte)))
    if min(alte_werte) > 0.0:
        print("-> 0,0% trifft NICHT mehr zu, ueberall ist die Seenflaeche > 0.")
    else:
        print("-> mindestens eine Messung liefert weiterhin 0,0%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
