"""
Path: tests/smoke_test_wasserbilanz_toleranz.py

Ticket: https://github.com/nikolaisoerensen/MapGenerator/issues/70
"Wasserbilanz mit Toleranzband von zehn Prozent zusichern"

WAS DIESER TEST PRUEFT. Ueber einen VOLLSTAENDIGEN Erzeugungslauf (Terrain ->
Weather -> Hydrology, bei 512 px, mit festem Seed) muss gelten:

    Niederschlag (Summe ueber die Karte)
        = Randabfluss + Verdunstung + Zuwachs im Wasserspeicher   (+/- Band)

Das Toleranzband selbst steht NICHT hier im Code, sondern versioniert in
tests/toleranzen.toml (`[wasserbilanz] toleranz_prozent`) - Nutzer-Entscheidung
zu Ticket #18: "Ja, mit einem Toleranzband von zehn Prozent. Wichtig ist nur,
dass es nicht ausufert." Der Dateiname ist bewusst NICHT "bandgrenzen*.toml":
genau dieses Muster ist in nachtbetrieb/sperrliste.toml hart gesperrt, und
"Toleranz" sollte mit "Bandgrenze" nicht verwechselbar sein.

EICHRANG, NICHT WAECHTERRANG. Dieser Test faehrt die komplette Pipeline bei
512 px echt durch (siehe Laufzeit unten) - er gehoert damit zu den Tests, die
NICHT bei jedem Commit laufen, sondern nachts oder von Hand
(`.venv/Scripts/python.exe tests/smoke_test_wasserbilanz_toleranz.py`), genau
wie `tests/smoke_test_regionen_welt.py`.

WOHER DIE RECHNUNG STAMMT. `tools/weather_lab.py` (`water_budget()`) hat diese
Bilanz zuerst gebaut und dabei einen lehrreichen Fehler gemacht: einen
Vergleich von Niederschlag in mm/JAHR gegen Randabfluss in m³ ueber die
SIMULATIONSDAUER (`PipeFlowSimulator.PIPE_TIME_SCALE_S` = 1800 s) - zwei
verschiedene Uhren, keine Bilanz, +3517 % Scheinabweichung. Richtig ist die
Buchfuehrung ueber genau die 1800 s, die das Pipe-Modell tatsaechlich simuliert
- exakt das macht diese Funktion hier, mit denselben importierten Konstanten
(`PipeFlowSimulator.PIPE_TIME_SCALE_S`/`RAIN_TO_DEPTH_RATE`), damit die Bilanz
nicht still auseinanderlaeuft, wenn dort jemand kalibriert.

Die Rechnung ist bewusst NICHT aus tools/weather_lab.py importiert, sondern
hier noch einmal aufgeschrieben: dieses Skript laeuft aus einem Git-Worktree,
und tools/weather_lab.py setzt beim Import selbst einen `sys.path`-Eintrag auf
den HAUPT-Checkout (fester String, kein `__file__`-relativer Pfad) - ein
`import tools.weather_lab` wuerde diesen Eintrag vor den Worktree-Pfad
schieben und nachfolgende `core`/`managers`-Importe leise auf den falschen
Checkout umleiten (siehe CLAUDE.md, Abschnitt "Git worktrees"). Zwei Kopien
derselben Rechnung sind unschoen, aber ein Test, der im Worktree den falschen
Code misst, ist schlimmer.

GEMESSENER STAND (2026-09-16, dieser Nachtlauf, Seed 424242, 512x512,
map_distance_km=15.0, FEEDBACK_PASSES=1 - der Default aus
managers/calculator_graph.py):

    Niederschlag            6.54e+09 m3
    Randabfluss + Verdunstung + Speicher   8.81e+09 m3
    ABWEICHUNG               +34.6 %

Das liegt WEIT ausserhalb des 10-Prozent-Bandes - der Test ist damit heute
ROT, und das ist nach Ticket #70 ausdruecklich zulaessig ("Falls er ... ROT
ist: das ist... in Ordnung, WENN er klar als bekannter Befund mit Frist
markiert ist, nicht wegerklaert"). DIES IST DIE MARKIERUNG. Eine tatsaechliche
Korrektur der Bilanz selbst ist NICHT Teil von Ticket #70 (das baut nur die
Zusicherung) und braucht ein eigenes Folge-Ticket.

WICHTIGER GEGENBEFUND ZUR DOKUMENTIERTEN +10,7-%-ZAHL. `docs/archiv/2026-07-29_SPEZIFIKATION.md`
§3.6 nennt "+10.7 % ungeklärt" als Stand der Wasserbilanz. Bei ECHTEN 512 px
(map_size=512 UND LOD passend zu 512 px, siehe unten) reproduziert sich diese
Zahl NICHT - gemessen wurden +34.6 %, stabil ueber zwei Wiederholungen
(+34.63 % / +34.62 %). Ein erster Messversuch mit map_size=512 aber LOD=3
(tatsaechliche Rechengroesse dabei nur 128 px, siehe
`managers/data_lod_manager.calculate_lod_size`) ergab +9.15 % - deutlich naeher
an einem gruenen Ergebnis, aber eben NICHT bei 512 px gerechnet. Vermutung:
die dokumentierte +10,7-%-Zahl in der Spezifikation stammt aus einem Lauf bei
kleinerer tatsaechlicher Kartengroesse. Das hier ist keine Erklaerung, die
wegerklaert - es ist eine Messung, die einer frueheren, nicht mehr
nachvollziehbaren Messung widerspricht. Nach der Projektregel "sind mehrere
Pruefungen gruen/eine Zahl dokumentiert und das Ergebnis weicht trotzdem ab,
zwei Enden der Kette gegeneinander messen" wird das hier so stehen gelassen,
nicht angepasst, bis jemand am Tag entscheidet, welche Messung die
massgebliche ist.

DIE LOD-FALLE, extra genannt, weil sie leicht wieder zuschlaegt: `map_size`
alleine reicht nicht - der DataLODManager rechnet bei gegebenem LOD-Level
`lod_size = MAPSIZEMIN * 2^(lod-1)` (`MAPSIZEMIN` = 32 laut
`gui/config/value_default.py`). Nur wer zusaetzlich
`calculate_max_lod_for_size(512)` (liefert 5) als LOD-Level uebergibt, rechnet
tatsaechlich auf einem 512x512-Feld - sonst rechnet die Pipeline leise auf
einer kleineren Karte, ohne Fehlermeldung, mit einem plausibel aussehenden
Ergebnis (siehe oben, +9.15 % statt +34.6 %).

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_wasserbilanz_toleranz.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Fester Seed, damit zwei Laeufe vergleichbar sind (TERRAIN.MAP_SEED wuerfelt
# sonst bei jedem Start neu) - derselbe Wert wie tools/weather_lab.py LAB_SEED,
# aus derselben Begruendung (Erosion-Labor).
SEED = 424242
SIZE = 512
KM = 15.0

# Die Qt-Anwendung MUSS modulweit gehalten werden - als lokale Variable raeumt
# Python sie am Funktionsende ab, waehrend der GPU-Worker noch seinen
# OpenGL-Kontext haelt (Segfault ohne Meldung, siehe CLAUDE.md).
_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


def _defaults(generator_name, map_size, map_distance_km):
    """Die echten Default-Parameter eines Generators aus value_default.py."""
    from gui.config import value_default as vd

    klassen = {"terrain": vd.TERRAIN, "geology": vd.GEOLOGY, "erosion": vd.EROSION,
               "weather": vd.WEATHER, "water": vd.WATER, "biome": vd.BIOME,
               "settlement": vd.SETTLEMENT}
    klasse = klassen[generator_name]
    werte = {}
    for name in dir(klasse):
        if not name.isupper():
            continue
        eintrag = getattr(klasse, name)
        werte[name.lower()] = (eintrag["default"]
                               if isinstance(eintrag, dict) and "default" in eintrag
                               else eintrag)
    werte.update({"map_size": map_size, "map_seed": SEED,
                  "map_distance_km": map_distance_km})
    return werte


def _run_pipeline(map_size, map_distance_km, lod):
    """
    Faehrt die vollstaendige Pipeline (alle Generatoren, ueber den echten
    CalculatorDispatcher) einmal durch und liefert den DataLODManager, aus dem
    sich die Bilanz auslesen laesst.

    Nutzt den ECHTEN Dispatcher und die echten Generatoren - nicht eine
    nachgebaute Reihenfolge, sonst misst dieser Test etwas anderes als die
    laufende App. Vorbild: tools/weather_lab.py:run_pipeline().
    """
    _qt()

    from managers.data_lod_manager import DataLODManager
    from managers.generation_orchestrator import (
        GenerationOrchestrator, GeneratorType)

    manager = DataLODManager()
    manager.set_map_distance_km(map_distance_km)
    orchestrator = GenerationOrchestrator(data_lod_manager=manager)

    for generator in GeneratorType:
        orchestrator.prime_generator_parameters(
            generator.value, _defaults(generator.value, map_size, map_distance_km))

    from managers.calculator_graph import CalculatorDispatcher
    dispatcher = CalculatorDispatcher(orchestrator._build_calculator_executors())
    orchestrator.calculator_dispatcher = dispatcher
    for generator in GeneratorType:
        dispatcher.request(generator.value, lod)
        for cid in orchestrator._calculator_ids_for(generator.value):
            manager.set_calculator_target_lod(cid, lod)

    while not dispatcher.is_fully_done():
        bereit = dispatcher.get_next_ready_batch()
        if not bereit:
            break
        for cid in bereit:
            orchestrator._run_calculator_sync(cid, dispatcher.current_round)
            dispatcher.mark_completed(cid, dispatcher.current_round)

    return manager


def _lies(manager, calculator_id, key, lod):
    wert = manager.get_calculator_output(calculator_id, key, lod)
    return np.asarray(wert, dtype=np.float64) if isinstance(wert, np.ndarray) else wert


def wasserbilanz(manager, lod, map_distance_km):
    """
    Niederschlag ueber die Simulationsdauer gegen Randabfluss + Verdunstung +
    Zuwachs im Speicher - alles auf denselben Nenner (m3, siehe Docstring oben
    zur Zeitbasis-Falle).
    """
    from core.water_generator import PipeFlowSimulator

    precip = _lies(manager, "weather.precipitation", "precip_map", lod)
    if precip is None:
        raise ValueError("precip_map fehlt - lief die Pipeline durch?")

    groesse = precip.shape[0]
    zellflaeche = (map_distance_km * 1000.0 / groesse) ** 2
    kartenflaeche = zellflaeche * precip.size

    dauer_s = PipeFlowSimulator.PIPE_TIME_SCALE_S
    regen_m = float(precip.sum()) * PipeFlowSimulator.RAIN_TO_DEPTH_RATE * dauer_s
    regen_m3 = regen_m * zellflaeche

    randabfluss = float(_lies(manager, "water.flow_network", "ocean_outflow", lod) or 0.0)
    verdunstung = float(_lies(manager, "water.flow_network", "evaporated_volume", lod) or 0.0)

    tiefe = _lies(manager, "water.flow_network", "water_depth", lod)
    speicher_m3 = float(tiefe.sum()) * zellflaeche if tiefe is not None else 0.0

    raus_m3 = randabfluss + verdunstung + speicher_m3
    abweichung = 100.0 * (raus_m3 - regen_m3) / regen_m3 if regen_m3 else float("nan")

    return {
        "kartengroesse_px": groesse,
        "kartenflaeche_km2": kartenflaeche / 1e6,
        "dauer_s": dauer_s,
        "regen_m3": regen_m3,
        "randabfluss_m3": randabfluss,
        "verdunstung_m3": verdunstung,
        "speicher_m3": speicher_m3,
        "raus_m3": raus_m3,
        "abweichung_prozent": abweichung,
    }


def _lade_toleranz_prozent():
    """Das Toleranzband aus tests/toleranzen.toml - Nutzer-Entscheidung zu
    Ticket #18, versioniert im Repo statt im Testcode (Ticket #70)."""
    import tomllib

    pfad = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toleranzen.toml")
    with open(pfad, "rb") as datei:
        daten = tomllib.load(datei)
    return float(daten["wasserbilanz"]["toleranz_prozent"])


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def run_wasserbilanz_im_toleranzband():
    from managers.data_lod_manager import calculate_max_lod_for_size

    lod = calculate_max_lod_for_size(SIZE)
    toleranz = _lade_toleranz_prozent()

    t0 = time.time()
    manager = _run_pipeline(map_size=SIZE, map_distance_km=KM, lod=lod)
    bilanz = wasserbilanz(manager, lod, map_distance_km=KM)
    dauer = time.time() - t0

    print("Pipeline %dx%d px (LOD %d), Seed %d, %.1f km, %.1f s Laufzeit"
          % (bilanz["kartengroesse_px"], bilanz["kartengroesse_px"], lod, SEED, KM, dauer))
    print("  Niederschlag              %12.4e m3" % bilanz["regen_m3"])
    print("  Randabfluss               %12.4e m3" % bilanz["randabfluss_m3"])
    print("  Verdunstung               %12.4e m3" % bilanz["verdunstung_m3"])
    print("  im Speicher geblieben     %12.4e m3" % bilanz["speicher_m3"])
    print("  Summe RAUS                %12.4e m3" % bilanz["raus_m3"])
    print("  Abweichung                %+11.2f %%" % bilanz["abweichung_prozent"])
    print("  Toleranzband (toleranzen.toml)  +/- %.1f %%" % toleranz)

    ok = True
    ok &= check("Karte tatsaechlich bei %d px gerechnet (nicht durch LOD verkleinert)"
                % SIZE, bilanz["kartengroesse_px"] == SIZE)
    ok &= check("Niederschlag > 0 (Bilanz waere sonst undefiniert)",
                bilanz["regen_m3"] > 0.0)
    ok &= check("Abweichung liegt im Toleranzband von +/- %.1f %%" % toleranz,
                abs(bilanz["abweichung_prozent"]) <= toleranz)
    return ok


if __name__ == "__main__":
    ok = run_wasserbilanz_im_toleranzband()
    print("\n=== SUMMARY ===")
    print("wasserbilanz_im_toleranzband: %s" % ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)
