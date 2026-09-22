"""
Path: tests/smoke_test_pipeline_outputs.py

Faehrt die GESAMTE Pipeline - alle 38 Knoten des CALCULATOR_GRAPH, alle sieben
Generatoren - und prueft jeden einzelnen deklarierten Output.

WARUM ES DIESEN TEST GIBT. Am 2026-07-30 meldete der Nutzer, dass viele
Anzeige-Schalter nichts mehr zeigen: Slope, fast alle Geology-Schalter, alle
Erosion-Schalter, fast alle Water-Schalter. Kein Test haette das gefunden - die
vorhandenen pruefen je einen Generator, und die Anzeige liest Outputs, die
dabei gar nicht entstehen.

Geprueft wird je Output eine von fuenf Lagen:

    FEHLT                        der Knoten hat ihn nie geschrieben
    NUR NULL                     vorhanden, aber ueberall exakt 0 - die Anzeige bleibt leer
    KONSTANT                     vorhanden, aber ueberall derselbe Wert
    OK                           enthaelt echte Werte
    NICHT-ENDLICH                enthaelt NaN/Inf
    TEILWEISE, NaN dokumentiert  Sonderfall von NICHT-ENDLICH, siehe unten

"NUR NULL" ist nicht automatisch ein Fehler: erosion.* liefert absichtlich
Nullkarten, solange EROSION_AKTIV auf False steht (§8). Dasselbe gilt fuer
geology.intrusions/height_delta - die Nullkarte ist dort Absicht, nicht
Ausfall (Nutzer-Vorgabe "Stoerungen greifen nicht in das Terrain ein",
siehe core/geology_generator.py Modul-Docstring Zeilen 15-25 und
_calc_intrusions() Zeilen 1222-1239). Der Test nennt das deshalb getrennt
(Ticket #79).

Ebenso settlement.pathfinding/sea_roads: eine leere Liste ist auf DIESER
Testkarte (SIZE/KM/SEED oben) korrekt, kein Ausfall (Ticket #80). Ein
Seeweg entsteht nur, wenn ein Kulturpaar KEINEN endlichen Landweg hat
(calculate_road_network(), §4.4-Zweig). Nachgemessen: bei diesem Seed
liegen alle 40 Siedlungen auf derselben, 5326 px grossen Hauptlandmasse
(daneben nur neun Kleinstinseln mit 5-9 px, siedlungsfrei) - jedes der
neun Kulturpaare findet also einen Landweg, 0 Seewege ist damit die
richtige Zahl fuer diese Karte, nicht ein Zeichen fehlender Berechnung.

"NICHT-ENDLICH" (NaN/Inf im Feld) ist ebenfalls nicht automatisch ein
Fehler: manche Felder benutzen NaN als dokumentiertes "nicht zutreffend"-
Sentinel fuer die Anzeige, kein Rechenfehler. terrain.redistribution/
hinterland_height ist so ein Fall - core/terrain_weltkarte.py,
kuestengebiete(), belegt die Hoehenkarte mit
"np.full(H.shape, np.nan, ...)" und ueberschreibt nur Landflaechen mit
zugeordnetem Kuestengebiet; auf See und im alpinen Sonderfall bleibt NaN
stehen ("DIE HOEHENFAKTOR-KARTE (2026-08-26)": "Auf See und im alpinen
Sonderfall NaN, damit die Anzeige dort nichts einfaerbt."). Ein Output, der
namentlich in NAN_SENTINEL_DOKUMENTIERT steht, bekommt deshalb statt
"NICHT-ENDLICH" die eigene, nicht-fehlerhafte Lage
"TEILWEISE, NaN dokumentiert" (Ticket #71). Auf dieser Karte wird der
Output derzeit noch nicht erzeugt (hinterland_height kommt erst mit
Ticket #64 in den CALCULATOR_GRAPH), die Ausnahme greift also aktuell
nicht sichtbar - sie ist trotzdem hier eingetragen, damit der Fehlalarm
nicht erneut auftritt, sobald #64 gemerged ist.

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

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

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


# terrain.redistribution/hinterland_height: core/terrain_weltkarte.py,
# Funktion kuestengebiete() ("hoehen_karte = np.full(H.shape, np.nan, ...)",
# Kommentar "DIE HOEHENFAKTOR-KARTE (2026-08-26)"): "Auf See und im alpinen
# Sonderfall NaN, damit die Anzeige dort nichts einfaerbt." NaN ist hier ein
# absichtliches "nicht zutreffend"-Sentinel fuer See- und Alpin-Sonderfall,
# kein Rechenfehler (Ticket #71). Neue Eintraege hier nur mit Verweis auf
# die Stelle, die das Verhalten begruendet - sonst wird aus der
# Ausnahmeliste ein Freibrief.
NAN_SENTINEL_DOKUMENTIERT = (
    "terrain.redistribution / hinterland_height",
)

# Ersatz-Lage fuer Felder aus NAN_SENTINEL_DOKUMENTIERT: statt des
# generischen Fehlerbefunds "NICHT-ENDLICH" ein eigener, nicht-fehlerhafter
# Status.
LAGE_NAN_DOKUMENTIERT = "TEILWEISE, NaN dokumentiert"


def _lage(wert, schluessel=None):
    """FEHLT / NUR NULL / KONSTANT / OK / NICHT-ENDLICH.

    `schluessel` ("Knoten / Output", wie in `ergebnis` verwendet) ist
    optional und wird nur gebraucht, um NAN_SENTINEL_DOKUMENTIERT zu
    erkennen: ein Feld, das dort namentlich gelistet ist, bekommt statt
    "NICHT-ENDLICH" den eigenen, nicht-fehlerhaften Status
    LAGE_NAN_DOKUMENTIERT (Ticket #71).
    """
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
        if schluessel in NAN_SENTINEL_DOKUMENTIERT:
            return LAGE_NAN_DOKUMENTIERT
        return "NICHT-ENDLICH"
    if np.all(feld == 0):
        return "NUR NULL"
    if float(feld.max() - feld.min()) == 0.0:
        return "KONSTANT"
    return "OK"


def _durchlauf(mit_gpu):
    """Fahre alle Knoten und liefere {output_schluessel: (lage, form)}.

    Als drittes Element auch {knoten: {undeklarierte Keys}} - was
    set_calculator_output() tatsaechlich gespeichert bekommen hat, aber in
    CalculatorSpec.output_keys nicht steht. Gefunden bei Ticket #64: acht
    erklaerte gegen achtzehn tatsaechliche Ausgaben bei terrain.redistribution,
    dieselbe Luecke unerkannt bei sechs weiteren Knoten. output_keys ist keine
    Attrappe - gui/widgets/pipeline_status_panel.py baut daraus sein Anzeige-
    Etikett, managers/generation_orchestrator.py sein Prozess-Log, und dieser
    Test selbst prueft nur die hier gelisteten Keys auf GPU/CPU-Paritaet. Eine
    unvollstaendige Liste heisst also: Etikett, Log UND Paritaetspruefung
    fehlen fuer genau die nicht gelisteten Felder.
    """
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

    undeklariert = {}
    _original_set = DataLODManager.set_calculator_output

    def _set_hook(self, calculator_id, lod_level, outputs):
        deklariert = set(CALCULATOR_GRAPH[calculator_id].output_keys)
        neu = {k for k, v in outputs.items() if v is not None} - deklariert
        if neu:
            undeklariert.setdefault(calculator_id, set()).update(neu)
        return _original_set(self, calculator_id, lod_level, outputs)

    ergebnis = {}
    abbrueche = []
    DataLODManager.set_calculator_output = _set_hook
    try:
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
                voller_schluessel = "%s / %s" % (knoten, schluessel)
                ergebnis[voller_schluessel] = (
                    _lage(wert, voller_schluessel), form)
    finally:
        DataLODManager.set_calculator_output = _original_set
    return ergebnis, abbrueche, undeklariert


def lauf():
    _qt()

    print("Pipeline %d px, LOD %d, %.0f km\n" % (SIZE, LOD, KM))
    gpu, gpu_abbrueche, gpu_undeklariert = _durchlauf(True)
    cpu, cpu_abbrueche, cpu_undeklariert = _durchlauf(False)

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
    for lage in ("OK", "NUR NULL", "KONSTANT", "FEHLT", "NICHT-ENDLICH",
                 LAGE_NAN_DOKUMENTIERT):
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
    # NICHT-ENDLICH bleibt hier ein Fehlerbefund - Outputs aus
    # NAN_SENTINEL_DOKUMENTIERT tragen wegen _lage() bereits die eigene Lage
    # LAGE_NAN_DOKUMENTIERT statt "NICHT-ENDLICH" (Ticket #71) und fallen
    # deshalb gar nicht erst unter diese Bedingung.
    unerwartet_null = [s for s, (l, _) in gpu.items()
                       if l in ("FEHLT", "NICHT-ENDLICH")
                       or (l == "NUR NULL" and not s.startswith("erosion.")
                           and s not in ERWARTET_NULL)]
    if unerwartet_null:
        print()
        print("Outputs ohne Daten (Anzeige bleibt leer):")
        for s in unerwartet_null:
            print("   %s" % s)

    undeklariert = {}
    for quelle in (gpu_undeklariert, cpu_undeklariert):
        for knoten, keys in quelle.items():
            undeklariert.setdefault(knoten, set()).update(keys)
    if undeklariert:
        print()
        print("Knoten liefern Outputs, die CalculatorSpec.output_keys nicht kennt:")
        for knoten in sorted(undeklariert):
            eintrag = "%s: %s" % (knoten, ", ".join(sorted(undeklariert[knoten])))
            print("   %s" % eintrag)
            fehler.append("undeklarierter Output bei " + eintrag)

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
