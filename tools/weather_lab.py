"""
Path: tools/weather_lab.py

WERKZEUG, kein Test. Fährt den Rückkopplungs-Kreis Weather -> Water -> Biome
und misst, ob die Wasserbilanz aufgeht und ob der Kreis konvergiert.

WARUM ES DAS BRAUCHT: der Kreis läuft heute weg - über fünf Durchgänge trocknet
die Karte monoton aus (Niederschlag 14.34 -> 0.34), die Änderung je Durchgang
liegt am Ende noch über 50 %. Bevor irgendeine Kopplung angefasst wird, muss
sichtbar sein, WO das Wasser bleibt. Genau diese Reihenfolge hat die Erosion
gerettet: dort waren 69 % des abgetragenen Materials verschwunden, ohne dass es
jemandem aufgefallen wäre - erst eine Bilanz je Pass hat es gezeigt.

Es gibt bereits einen `water_mass_balance`-Test (smoke_test_weather_climatology),
der grün ist. Der deckt aber nur die Atmosphäre ab: Kondensation muss die Menge
exakt aus q entfernen. Was danach passiert - Regen fällt, versickert, fliesst ab,
verdunstet wieder - ist darin nicht enthalten. Diese Lücke schliesst die Bilanz
hier.

EINHEITEN: alles in m³ pro Karte und Jahr. Die Quellen liefern gemischt:
    precip_map        mm/Jahr je Zelle   (weather_generator.py:194)
    water_depth       m Wassersäule
    soil_moist_map    Prozent
    ocean_outflow     m³ über den Durchlauf   (water_generator.py:129)
    evaporated_volume m³ über den Durchlauf
Eine Bilanz aus gemischten Einheiten ist keine Bilanz - deshalb rechnet
`_als_kubikmeter()` alles auf denselben Nenner.

Aufruf:
    .venv\\Scripts\\python.exe tools/weather_lab.py bilanz
    .venv\\Scripts\\python.exe tools/weather_lab.py konvergenz
"""

import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lab_output", "weather_lab")

# Fester Seed - dieselbe Begründung wie im Erosion-Labor: TERRAIN.MAP_SEED wird
# bei jedem Programmstart neu gewürfelt, und zwei Messreihen auf verschiedenen
# Karten sind nicht vergleichbar.
LAB_SEED = 424242


# =============================================================================
# AUFBAU
# =============================================================================

# Die QGuiApplication MUSS in einem modulweiten Namen festgehalten werden.
# Wird sie nur lokal erzeugt, raeumt Python sie am Funktionsende ab, waehrend
# der GPU-Worker noch seinen OpenGL-Kontext haelt - das Ergebnis ist ein
# Segmentation Fault ohne jede Meldung (genau so beim Bauen passiert: exit=139,
# null Zeilen Ausgabe).
_QT_APP = None


def _ensure_qt_application():
    """Offscreen-Qt-Kontext fuer den GPU-Pfad, ohne Fenster (siehe CLAUDE.md)."""
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
    werte.update({"map_size": map_size, "map_seed": LAB_SEED,
                  "map_distance_km": map_distance_km})
    return werte


def run_pipeline(map_size=128, map_distance_km=15.0, feedback_passes=None,
                 lod=3):
    """
    Fährt die vollständige Pipeline einmal durch und gibt den DataLODManager
    zurück, aus dem sich anschliessend alles auslesen lässt.

    Nutzt den ECHTEN Dispatcher und die echten Generatoren - nicht eine
    nachgebaute Reihenfolge. Sonst misst das Werkzeug etwas anderes als die App.
    """
    _ensure_qt_application()

    import managers.calculator_graph as graph_module
    from managers.calculator_graph import CalculatorDispatcher
    from managers.data_lod_manager import DataLODManager
    from managers.generation_orchestrator import (
        GenerationOrchestrator, GeneratorType)

    vorher = graph_module.FEEDBACK_PASSES
    if feedback_passes is not None:
        graph_module.FEEDBACK_PASSES = feedback_passes

    try:
        manager = DataLODManager()
        manager.set_map_distance_km(map_distance_km)
        orchestrator = GenerationOrchestrator(data_lod_manager=manager)

        for generator in GeneratorType:
            orchestrator.prime_generator_parameters(
                generator.value, _defaults(generator.value, map_size, map_distance_km))

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

        return manager, lod
    finally:
        graph_module.FEEDBACK_PASSES = vorher


# =============================================================================
# WASSERBILANZ
# =============================================================================

def _lies(manager, calculator_id, key, lod):
    wert = manager.get_calculator_output(calculator_id, key, lod)
    return np.asarray(wert, dtype=np.float64) if isinstance(wert, np.ndarray) else wert


def water_budget(manager, lod, map_distance_km=15.0):
    """
    Die Wasserbilanz einer Karte - in der ZEITBASIS DER SIMULATION.

    DER ERSTE ANLAUF WAR FALSCH, und der Fehler ist lehrreich genug, um ihn
    hier zu behalten: ich habe den Niederschlag (mm pro JAHR, siehe
    weather_generator.py:194) gegen den Randabfluss (m³ ueber den Durchlauf)
    gerechnet und kam auf +3517 % Abweichung. Das sah nach einem gewaltigen
    Leck aus, war aber nur ein Vergleich zweier verschiedener Zeitbasen:
    das Pipe-Modell simuliert PIPE_TIME_SCALE_S = 1800 Sekunden, keine
    Jahre. Eine Bilanz aus zwei Uhren ist keine Bilanz.

    Richtig ist die Buchfuehrung ueber den Lauf:

        Regen ueber die Simulationsdauer
            = Randabfluss + Verdunstung + Zuwachs im Speicher

    Der Speicher gehoert hinein, weil die Karte trocken startet: alles, was am
    Ende noch als stehendes Wasser dasteht, ist weder abgeflossen noch
    verdunstet - es liegt noch da. Ohne diesen Posten fehlt der groesste
    Einzelbetrag.

    ZWEITE ZEITBASIS, getrennt ausgewiesen: die Klimatologie in mm/Jahr. Sie
    ist die Groesse, gegen die spaeter die Referenztabelle prueft, und sie ist
    NICHT mit den Volumina oben verrechenbar.
    """
    from core.water_generator import PipeFlowSimulator

    precip = _lies(manager, "weather.precipitation", "precip_map", lod)
    if precip is None:
        raise ValueError("precip_map fehlt - lief die Pipeline durch?")

    groesse = precip.shape[0]
    zellflaeche = (map_distance_km * 1000.0 / groesse) ** 2
    kartenflaeche = zellflaeche * precip.size

    # Genau die Umrechnung, die das Pipe-Modell selbst benutzt - importiert
    # statt nachgebaut, damit die Bilanz nicht still auseinanderlaeuft, wenn
    # dort jemand kalibriert.
    dauer_s = PipeFlowSimulator.PIPE_TIME_SCALE_S
    regen_m = float(precip.sum()) * PipeFlowSimulator.RAIN_TO_DEPTH_RATE * dauer_s
    regen_m3 = regen_m * zellflaeche

    randabfluss = float(_lies(manager, "water.flow_network", "ocean_outflow", lod) or 0.0)
    verdunstung = float(_lies(manager, "water.flow_network", "evaporated_volume", lod) or 0.0)

    tiefe = _lies(manager, "water.flow_network", "water_depth", lod)
    speicher_m3 = float(tiefe.sum()) * zellflaeche if tiefe is not None else 0.0

    raus = randabfluss + verdunstung + speicher_m3

    boden = _lies(manager, "water.soil_moisture", "soil_moist_map", lod)

    return {
        "kartenflaeche_km2": kartenflaeche / 1e6,
        "dauer_s": dauer_s,
        "regen_m3": regen_m3,
        "randabfluss_m3": randabfluss,
        "verdunstung_m3": verdunstung,
        "speicher_m3": speicher_m3,
        "raus_m3": raus,
        "abweichung_prozent": 100.0 * (raus - regen_m3) / regen_m3 if regen_m3 else float("nan"),
        # zweite Zeitbasis, bewusst getrennt
        "regen_mm_jahr": float(precip.mean()),
        "bodenfeuchte_prozent": float(boden.mean()) if boden is not None else float("nan"),
    }


def print_budget(bilanz, titel=""):
    print()
    print("--- Wasserbilanz {} ---".format(titel))
    print("  Karte {:.0f} km2, Simulationsdauer {:.0f} s".format(
        bilanz["kartenflaeche_km2"], bilanz["dauer_s"]))
    print("  REIN   Niederschlag        {:12.4e} m3".format(bilanz["regen_m3"]))
    print("  RAUS   Randabfluss         {:12.4e} m3".format(bilanz["randabfluss_m3"]))
    print("         Verdunstung         {:12.4e} m3".format(bilanz["verdunstung_m3"]))
    print("         im Speicher geblieben {:10.4e} m3".format(bilanz["speicher_m3"]))
    print("         Summe               {:12.4e} m3".format(bilanz["raus_m3"]))
    print("  ABWEICHUNG                 {:+11.2f} %".format(bilanz["abweichung_prozent"]))
    print("  --- andere Zeitbasis, nicht verrechenbar ---")
    print("  Niederschlag  {:8.1f} mm/Jahr     Bodenfeuchte {:6.1f} %".format(
        bilanz["regen_mm_jahr"], bilanz["bodenfeuchte_prozent"]))


# =============================================================================
# ABLAEUFE
# =============================================================================

def lauf_bilanz():
    """Stufe 1: die Bilanz einmal aufschreiben, ohne etwas zu aendern."""
    manager, lod = run_pipeline(feedback_passes=1)
    bilanz = water_budget(manager, lod)
    print_budget(bilanz, "(FEEDBACK_PASSES=1, heutiger Stand)")
    return 0


def lauf_konvergenz():
    """Wie entwickeln sich Bilanz und Karten ueber mehrere Durchgaenge?"""
    print("%-8s %10s %12s %12s %12s %10s"
          % ("Passes", "Regen mm/a", "Randabfl m3", "Verdunst m3", "Speicher m3", "Abw %"))
    for passes in (1, 2, 3):
        manager, lod = run_pipeline(feedback_passes=passes)
        b = water_budget(manager, lod)
        print("%-8d %10.2f %12.3e %12.3e %12.3e %9.2f%%"
              % (passes, b["regen_mm_jahr"], b["randabfluss_m3"], b["verdunstung_m3"],
                 b["speicher_m3"], b["abweichung_prozent"]))
    return 0



# =============================================================================
# KLIMA UEBER DIE BREITENGRADE
# =============================================================================

# Referenzwerte fuer FLACHES Gelaende auf Meereshoehe, grob nach realen
# Klimadaten. Sie sind ausdruecklich ein VORSCHLAG zur Abstimmung - genau so
# ist die Temperatur-Klimatologie im Projekt entstanden
# (_TEMP_CLIMATOLOGY_TABLE, "Nutzer-Abstimmung 2026-07-23").
#
# Der Niederschlag ist der schwierigere Teil: absolute Werte darf man mit einem
# Faktor skalieren, aber die VERHAELTNISSE zwischen den Breiten muessen stimmen
# - der Wuestenguertel um 20-30 Grad ist trockener als Aequator UND als die
# Westwindzone bei 50 Grad. Diese Doppel-Struktur ist das eigentliche Ziel.
KLIMA_REFERENZ = {
    #        Jahresmittel  waermster  kaeltester  Niederschlag mm/a
    0:      (26.0,        27.0,      25.0,       2200),
    10:     (27.0,        29.0,      25.0,       1500),
    20:     (25.0,        31.0,      18.0,        400),   # Wuestenguertel
    30:     (20.0,        30.0,      10.0,        450),
    40:     (14.0,        25.0,       2.0,        700),
    50:     ( 9.0,        19.0,      -2.0,        750),   # Westwindzone
    60:     ( 3.0,        15.0,     -10.0,        500),
    70:     (-5.0,         8.0,     -20.0,        300),
}


def climate_sweep(map_size=128, breiten=(0, 10, 20, 30, 40, 50, 60), flach=False):
    """
    Misst, was die Simulation je Breitengrad tatsaechlich AUSGIBT.

    Die vorhandene Zusicherung climatology_reference_table_match prueft
    _climate_baseline(), also die Eingangs-Klimatologie. Ob das SIMULIERTE
    temp_map am Ende dort landet, prueft nichts - genau diese Luecke ist der
    Grund, warum die Temperaturen "irgendwie nicht stimmen".

    `flach=True` nimmt eine ebene Karte auf 100 m: dann faellt die
    Hoehen-Abkuehlung als Stoergroesse weg und die Zahlen sind direkt mit
    Meereshoehen-Klimadaten vergleichbar.
    """
    import numpy as np
    from core.weather_generator import WeatherSystemGenerator
    from core.terrain_generator import ShadowCalculator, generate_seasonal_sun_angles
    from managers.data_lod_manager import DataLODManager
    from gui.config import value_default as vd

    _ensure_qt_application()
    ergebnisse = {}

    for breite in breiten:
        heightmap = np.full((map_size, map_size), 100.0, dtype=np.float32)
        if not flach:
            y = np.arange(map_size)[:, None] * np.ones((1, map_size))
            heightmap = (100.0 + 1500.0 * np.exp(
                -((y - map_size / 2) ** 2) / (2 * (map_size / 6.0) ** 2))).astype(np.float32)

        shadowmap = ShadowCalculator().calculate_shadows(
            heightmap, lod_level=3,
            sun_angles_override=generate_seasonal_sun_angles(1, float(breite), 0.0))

        manager = DataLODManager()
        manager.set_map_distance_km(15.0)

        # DAS VORAB-BIOM MITRECHNEN. Ohne diesen Block ruft der Sweep
        # calculate_weather_system direkt auf, umgeht damit den Graphen - und
        # weather faellt still auf seinen Platzhalter zurueck, weil kein
        # Pre-Biome im Speicher liegt. Der Sweep misst dann genau die
        # Aenderung nicht, um die es geht (2026-07-29 prompt passiert: die
        # Niederschlagszahlen bewegten sich nach der Umstellung um weniger als
        # 1 %, weil der neue Pfad gar nicht betreten wurde).
        from core.terrain_generator import SlopeCalculator
        from core.biome_generator import BiomeClassificationSystem

        manager.set_calculator_output("terrain.redistribution", 3,
                                      {"heightmap": heightmap})
        manager.set_calculator_output("erosion.slope", 3, {"slopemap":
            SlopeCalculator().calculate_slopes(heightmap, {"map_size": map_size})})
        biome = BiomeClassificationSystem(data_lod_manager=manager)
        # Der Breitengrad kommt im Preseed aus dem MANAGER, nicht aus den
        # Parametern (get_map_latitude) - ihn nur in set_active_parameters zu
        # stecken hat ihn nie erreicht.
        manager.set_map_latitude(float(breite))
        biome.set_active_parameters({"map_size": map_size})
        biome._calc_preseed_hint("biome.preseed_hint", 3)

        generator = WeatherSystemGenerator(map_seed=LAB_SEED, data_lod_manager=manager)
        parameter = _defaults("weather", map_size, 15.0)
        parameter["map_latitude"] = float(breite)
        parameter["map_longitude"] = 0.0

        daten = generator.calculate_weather_system(
            heightmap, shadowmap, parameter, lod_level=3)

        monatlich = manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", 3)
        if monatlich is not None:
            mittel_je_monat = [float(np.mean(m)) for m in monatlich]
        else:
            mittel_je_monat = [float(np.mean(daten.temp_map))] * 6

        ergebnisse[breite] = {
            "jahresmittel": float(np.mean(mittel_je_monat)),
            "waermster": max(mittel_je_monat),
            "kaeltester": min(mittel_je_monat),
            "niederschlag": float(np.mean(daten.precip_map)),
            "je_monat": mittel_je_monat,
        }
    return ergebnisse


def lauf_klima():
    """Was die Simulation je Breitengrad liefert, gegen die Referenz."""
    ergebnisse = climate_sweep()
    print()
    print("%-8s %-26s %-26s" % ("", "SIMULIERT", "REFERENZ"))
    print("%-8s %8s %8s %8s %8s %8s %8s %8s"
          % ("Breite", "Mittel", "max", "min", "Mittel", "max", "min", "Regen"))
    for breite, wert in sorted(ergebnisse.items()):
        ref = KLIMA_REFERENZ.get(breite)
        if ref:
            print("%-8d %8.1f %8.1f %8.1f %8.1f %8.1f %8.1f %8.0f"
                  % (breite, wert["jahresmittel"], wert["waermster"], wert["kaeltester"],
                     ref[0], ref[1], ref[2], ref[3]))
        else:
            print("%-8d %8.1f %8.1f %8.1f" % (breite, wert["jahresmittel"],
                                              wert["waermster"], wert["kaeltester"]))
    print()
    print("Niederschlag simuliert (mm/Jahr, Kartenmittel):")
    for breite, wert in sorted(ergebnisse.items()):
        ref = KLIMA_REFERENZ.get(breite)
        print("   %3d Grad: %8.2f    Referenz %5s"
              % (breite, wert["niederschlag"], ref[3] if ref else "?"))
    return 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else "bilanz"
    return {"bilanz": lauf_bilanz, "konvergenz": lauf_konvergenz,
            "klima": lauf_klima}[name]()


if __name__ == "__main__":
    sys.exit(main())
