"""
Path: tests/smoke_test_pipeline_outputs.py

Faehrt die GESAMTE Pipeline - alle 39 Knoten des CALCULATOR_GRAPH, alle sieben
Generatoren - und prueft jeden einzelnen deklarierten Output.

WARUM ES DIESEN TEST GIBT. Am 2026-07-30 meldete der Nutzer, dass viele
Anzeige-Schalter nichts mehr zeigen: Slope, fast alle Geology-Schalter, alle
Erosion-Schalter, fast alle Water-Schalter. Kein Test haette das gefunden - die
vorhandenen pruefen je einen Generator, und die Anzeige liest Outputs, die
dabei gar nicht entstehen.

Geprueft wird je Output eine von vier Lagen:

    FEHLT       der Knoten hat ihn nie geschrieben
    NUR NULL    vorhanden, aber ueberall exakt 0 - die Anzeige bleibt leer
    KONSTANT    vorhanden, aber ueberall derselbe Wert
    OK          enthaelt echte Werte

"NUR NULL" ist nicht automatisch ein Fehler: erosion.* liefert absichtlich
Nullkarten, solange EROSION_AKTIV auf False steht (§8). Dasselbe gilt fuer
geology.intrusions/height_delta - die Nullkarte ist dort Absicht, nicht
Ausfall (Nutzer-Vorgabe "Stoerungen greifen nicht in das Terrain ein",
siehe core/geology_generator.py Modul-Docstring Zeilen 15-25 und
_calc_intrusions() Zeilen 1222-1239). Der Test nennt das deshalb getrennt.

Ebenso settlement.pathfinding/sea_roads: eine leere Liste ist auf DIESER
Testkarte (SIZE/KM/SEED oben) korrekt, kein Ausfall (Ticket #80). Ein
Seeweg entsteht nur, wenn ein Kulturpaar KEINEN endlichen Landweg hat
(calculate_road_network(), §4.4-Zweig). Nachgemessen: bei diesem Seed
liegen alle 40 Siedlungen auf derselben, 5326 px grossen Hauptlandmasse
(daneben nur neun Kleinstinseln mit 5-9 px, siedlungsfrei) - jedes der
neun Kulturpaare findet also einen Landweg, 0 Seewege ist damit die
richtige Zahl fuer diese Karte, nicht ein Zeichen fehlender Berechnung.

ZWEI DURCHGAENGE, das ist der zweite Zweck:

    mit ShaderManager    der GPU-Pfad, wie die App ihn nimmt
    ohne ShaderManager   der reine CPU-Pfad (jeder Generator hat einen)

Unterscheiden sich die Ergebnisse in der FORM oder faellt ein Output nur in
einem der beiden Durchgaenge aus, ist die Paritaet verletzt (SPEZIFIKATION
§4.1: eine Aenderung an einem Pfad ist erst fertig, wenn der andere mitgezogen
ist).

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_pipeline_outputs.py
"""

import sys
import traceback

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

# Die Qt-Anwendung MUSS modulweit gehalten werden. Als lokale Variable raeumt
# Python sie ab, waehrend der GL-Kontext noch lebt - Segfault ohne Meldung
# (derselbe Fall wie in tools/drainage_lab.py, siehe dortigen Kommentar).
_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


# 2026-08-05 von 64 auf 128 px angehoben.
#
# Bei 64 px deckt ein Pixel 333 m ab. Die Siedlungen liegen dann so dicht
# beieinander, dass jede Strasse kuerzer als drei Punkte ist -
# calculate_roadsites() ueberspringt solche Strassen, und der Test meldete
# roadsite_list als leer. Das ist kein Fehler im Programm, sondern eine Karte,
# die zu grob fuer Siedlungswege ist. Bei 128 px und darueber erscheint die
# Liste. Nachgewiesen im Vergleichslauf mit und ohne geklemmte Hoehen.
SIZE = 128
LOD = 3
KM = 15.0
SEED = 20260730


def _parameter():
    """Ein Parametersatz aus allen Vorgaben von value_default."""
    import gui.config.value_default as vd

    parameter = {"map_size": SIZE, "map_distance_km": KM, "map_seed": SEED}
    for klassenname in ("TERRAIN", "GEOLOGY", "WEATHER", "EROSION", "WATER",
                        "BIOME", "SETTLEMENT", "EROSION_FILTER",
                        "RIVER_NETWORK"):
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


def _generatoren(manager, shader_manager):
    from core.terrain_generator import BaseTerrainGenerator
    from core.geology_generator import GeologySystemGenerator
    from core.erosion_generator import ErosionSystemGenerator
    from core.weather_generator import WeatherSystemGenerator
    from core.water_generator import HydrologySystemGenerator
    from core.biome_generator import BiomeClassificationSystem
    from core.settlement_generator import SettlementGenerator

    gemeinsam = dict(shader_manager=shader_manager, data_lod_manager=manager)
    return {
        "terrain": BaseTerrainGenerator(map_seed=SEED, **gemeinsam),
        "geology": GeologySystemGenerator(**gemeinsam),
        "erosion": ErosionSystemGenerator(**gemeinsam),
        "weather": WeatherSystemGenerator(**gemeinsam),
        "water": HydrologySystemGenerator(**gemeinsam),
        "biome": BiomeClassificationSystem(**gemeinsam),
        "settlement": SettlementGenerator(**gemeinsam),
    }


def _reihenfolge():
    """Topologische Reihenfolge aus dem Graphen - nicht von Hand gefuehrt (§4.5)."""
    from managers.calculator_graph import CALCULATOR_GRAPH

    offen = dict(CALCULATOR_GRAPH)
    fertig = []
    erledigt = set()
    while offen:
        bereit = [k for k, s in offen.items()
                  if all(d in erledigt or d not in CALCULATOR_GRAPH
                         for d in s.depends_on)]
        if not bereit:
            # Zyklus oder fehlende Kante - die restlichen hinten anhaengen.
            fertig.extend(sorted(offen))
            break
        for k in sorted(bereit):
            fertig.append(k)
            erledigt.add(k)
            offen.pop(k)
    return fertig


def _lage(wert):
    """FEHLT / NUR NULL / KONSTANT / OK / NICHT-ENDLICH."""
    if wert is None:
        return "FEHLT"
    # Nicht jeder Output ist ein Feld - Settlement liefert z.B. Listen von
    # Orten unterschiedlicher Laenge. Die zaehlen als vorhanden, sobald sie
    # nicht leer sind.
    if isinstance(wert, (list, tuple, dict, set)):
        return "OK" if len(wert) else "NUR NULL"
    if isinstance(wert, (int, float, bool, np.integer, np.floating)):
        return "NUR NULL" if float(wert) == 0.0 else "OK"
    try:
        feld = np.asarray(wert)
    except Exception:
        return "OK"
    if feld.dtype == object or feld.size == 0:
        return "OK" if feld.size else "FEHLT"
    if not np.issubdtype(feld.dtype, np.number):
        return "OK"
    if not np.all(np.isfinite(feld)):
        return "NICHT-ENDLICH"
    if np.all(feld == 0):
        return "NUR NULL"
    if float(feld.max() - feld.min()) == 0.0:
        return "KONSTANT"
    return "OK"


def _durchlauf(mit_gpu):
    """Fahre alle Knoten und liefere {output_schluessel: (lage, form)}."""
    from managers.data_lod_manager import DataLODManager
    from managers.calculator_graph import CALCULATOR_GRAPH

    shader_manager = None
    if mit_gpu:
        from managers.shader_manager import ShaderManager
        shader_manager = ShaderManager()

    manager = DataLODManager()
    manager.set_map_distance_km(KM)
    manager.set_map_seed(SEED)
    parameter = _parameter()
    generatoren = _generatoren(manager, shader_manager)

    for knoten in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(knoten, LOD)
    for generator in generatoren.values():
        if hasattr(generator, "set_active_parameters"):
            generator.set_active_parameters(parameter)

    ergebnis = {}
    abbrueche = []
    for knoten in _reihenfolge():
        spec = CALCULATOR_GRAPH[knoten]
        generator = generatoren.get(spec.generator)
        methode = getattr(generator, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            abbrueche.append("%s: keine Methode _calc_%s"
                             % (knoten, knoten.split(".", 1)[1]))
            continue
        try:
            methode(knoten, LOD)
        except Exception as fehler:
            abbrueche.append("%s: %s: %s"
                             % (knoten, type(fehler).__name__,
                                str(fehler).splitlines()[0][:90]))
        for schluessel in spec.output_keys:
            wert = manager.get_calculator_output(knoten, schluessel, LOD)
            form = getattr(wert, "shape", None)
            ergebnis["%s / %s" % (knoten, schluessel)] = (_lage(wert), form)
    return ergebnis, abbrueche


def lauf():
    _qt()

    print("Pipeline %d px, LOD %d, %.0f km\n" % (SIZE, LOD, KM))
    gpu, gpu_abbrueche = _durchlauf(True)
    cpu, cpu_abbrueche = _durchlauf(False)

    fehler = []
    zaehler = {}
    print("%-46s %-14s %-14s" % ("Knoten / Output", "GPU-Pfad", "CPU-Pfad"))
    print("-" * 78)
    for schluessel in gpu:
        lage_g, form_g = gpu[schluessel]
        lage_c, form_c = cpu.get(schluessel, ("FEHLT", None))
        zaehler[lage_g] = zaehler.get(lage_g, 0) + 1
        marke = ""
        if lage_g != lage_c or form_g != form_c:
            marke = "  <- PFADE VERSCHIEDEN"
            fehler.append("%s: GPU %s%s, CPU %s%s"
                          % (schluessel, lage_g, form_g or "", lage_c,
                             form_c or ""))
        if lage_g != "OK" or marke:
            print("%-46s %-14s %-14s%s" % (schluessel[:46], lage_g, lage_c, marke))

    print()
    print("Zusammenfassung ueber %d Outputs:" % len(gpu))
    for lage in ("OK", "NUR NULL", "KONSTANT", "FEHLT", "NICHT-ENDLICH"):
        if zaehler.get(lage):
            print("   %-14s %d" % (lage, zaehler[lage]))

    for name, liste in (("GPU", gpu_abbrueche), ("CPU", cpu_abbrueche)):
        if liste:
            print()
            print("Abbrueche im %s-Durchlauf:" % name)
            for eintrag in liste:
                print("   %s" % eintrag)
                fehler.append("%s-Durchlauf: %s" % (name, eintrag))

    # NUR NULL ist bei erosion.* erwartet, solange der Hauptschalter aus ist,
    # bei geology.intrusions/height_delta immer (Ticket #79: die Nullkarte
    # ist dort Absicht, kein Ausfall), und bei settlement.pathfinding/
    # sea_roads auf dieser Testkarte (Ticket #80: kein Kulturpaar braucht
    # hier einen Seeweg, siehe Modul-Docstring oben).
    ERWARTET_NULL = ("geology.intrusions / height_delta",
                      "settlement.pathfinding / sea_roads")
    unerwartet_null = [s for s, (l, _) in gpu.items()
                       if l in ("FEHLT", "NICHT-ENDLICH")
                       or (l == "NUR NULL" and not s.startswith("erosion.")
                           and s not in ERWARTET_NULL)]
    if unerwartet_null:
        print()
        print("Outputs ohne Daten (Anzeige bleibt leer):")
        for s in unerwartet_null:
            print("   %s" % s)

    print()
    if fehler or unerwartet_null:
        print("NICHT IN ORDNUNG - %d Befunde" % (len(fehler) + len(unerwartet_null)))
        return 1
    print("Alle %d Outputs liefern Daten, GPU- und CPU-Pfad stimmen ueberein."
          % len(gpu))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(lauf())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
