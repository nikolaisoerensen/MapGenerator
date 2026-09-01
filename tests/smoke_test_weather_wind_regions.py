"""
Wind trifft die Regionsziele aus SPEZIFIKATION.md §3.5.

ANLASS (2026-08-11). Nutzer bestaetigte Windziele je Region (Referenzorte wie
bei Temperatur/Niederschlag: Cork, Bergen, Wologda, ...), danach Auftrag:
"kleineren, sicheren Teil jetzt" - Schrittzahl/Umbau der 3-Schicht-Simulation
selbst (1.5) bleibt separat, aber die Windwerte sollen die Regionsziele
treffen (Teil von 1.6/3.5).

VORHER: `wind_speed_factor` war ein einziger globaler Regler - alle neun
Regionen lagen faktisch gleich (gemessen 5.1-5.8 m/s ueberall), obwohl die
Zielspanne von 2.2 (Nevadin) bis 4.5 m/s (mehrere Kuestenregionen) reicht.

WIE DAS TRIFFT: `_wind_regional_faktor()` normiert das REGIONALE MITTEL der
Windgeschwindigkeit direkt auf `wind_ziel_map` (core/terrain_weltkarte.py
REGIONEN.wind_mittel_ms) - dasselbe "direkt auf den Zielwert normieren"-
Prinzip wie beim Niederschlag. EIN erster Versuch, den globalen
`wind_speed_factor`-Regler raeumlich zu variieren (an der Druckgradient-
Antriebskraft), zeigte praktisch KEINEN Effekt (0.16 gegen 0.71 Faktor:
1.871 gegen 1.872 m/s) - der Druckgradient ist nur einer von mehreren
additiven Antrieben und dominiert die Endgeschwindigkeit nicht. Die direkte
Normierung des ERGEBNISSES wirkt dagegen garantiert.

LUV/LEE (SPEZIFIKATION.md §3.5, 1.5-2x Kontrast): ein multiplikativer Term
(`_wind_luv_lee_faktor`) ist eingebaut, aber NICHT verifizierbar als
verlaesslicher Gruppenkontrast - gemessen im Nevadin: die vorhandene
Simulation hat selbst schon eine terraingetriebene Windstruktur, die mit
diesem einfachen Hangneigungs-Ansatz ANTIKORRELIERT (-0.53 gemessen), nicht
neutral. Deshalb hier NICHT geprueft; nur die REGIONSMITTEL-Zusicherung.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    fehler = []
    from PyQt6.QtGui import QGuiApplication
    _app = QGuiApplication.instance() or QGuiApplication([])

    import core.terrain_weltkarte as rw
    import smoke_test_pipeline_outputs as sp
    from managers.calculator_graph import CALCULATOR_GRAPH
    from managers.data_lod_manager import DataLODManager

    SIZE, LOD, SEED = 256, 6, 20260804

    manager = DataLODManager()
    manager.set_map_seed(SEED)
    manager.set_map_distance_km(rw.WELT_KM)
    par = dict(sp._parameter())
    par["map_size"] = SIZE
    par["map_seed"] = SEED
    par["map_distance_km"] = rw.WELT_KM
    gen = sp._generatoren(manager, None)
    for k in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(k, LOD)
    for g in gen.values():
        if hasattr(g, "set_active_parameters"):
            g.set_active_parameters(par)

    for knoten in sp._reihenfolge():
        if knoten.startswith(("settlement.", "biome.", "erosion.", "water.", "geology.")):
            continue
        spec = CALCULATOR_GRAPH[knoten]
        erzeuger = gen.get(spec.generator)
        methode = getattr(erzeuger, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            continue
        methode(knoten, LOD)
        if knoten == "weather.temperature":
            break

    wind_map = manager.get_calculator_output("weather.wind", "wind_map", LOD)
    region_map = manager.get_calculator_output("terrain.redistribution", "region_map", LOD)
    heightmap = manager.get_calculator_output("terrain.redistribution", "heightmap", LOD)
    if wind_map is None or region_map is None or heightmap is None:
        print("NICHT IN ORDNUNG: wind_map/region_map/heightmap nicht verfuegbar")
        return 1

    speed = np.hypot(wind_map[..., 0], wind_map[..., 1])
    land = heightmap > 0.0
    namen = [r["name"] for _z, _s, r in rw.alle_regionen()]
    ziele = {r["name"]: r["wind_mittel_ms"] for _z, _s, r in rw.alle_regionen()}

    print("1. Regionsmittel gegen SPEZIFIKATION.md §3.5")
    print("%-20s %8s %8s %10s" % ("Region", "ist", "ziel", "Abweichung"))
    TOLERANZ_MS = 0.6
    for i, name in enumerate(namen):
        maske = (region_map == i) & land
        if maske.sum() < 10:
            continue
        ist = float(speed[maske].mean())
        ziel = ziele[name]
        abweichung = ist - ziel
        print("%-20s %8.2f %8.2f %+10.2f" % (name, ist, ziel, abweichung))
        if abs(abweichung) > TOLERANZ_MS:
            fehler.append("%s: %.2f m/s weicht mehr als %.1f m/s vom Ziel %.2f ab"
                          % (name, ist, TOLERANZ_MS, ziel))

    print("")
    print("2. Reihenfolge stimmt (Nevadin am ruhigsten, Kuesten am windigsten)")
    mittel_je_region = {}
    for i, name in enumerate(namen):
        maske = (region_map == i) & land
        if maske.sum() >= 10:
            mittel_je_region[name] = float(speed[maske].mean())
    if "Nevadin" in mittel_je_region:
        ruhigste = min(mittel_je_region, key=mittel_je_region.get)
        print("   Ruhigste Region: %s (%.2f m/s)" % (ruhigste, mittel_je_region[ruhigste]))
        if ruhigste != "Nevadin":
            fehler.append("Nevadin ist nicht die windaermste Region (das ist %s)" % ruhigste)

    print("")
    print("3. Alter Nicht-Weltkarten-Pfad bleibt unveraendert (kein Absturz ohne wind_ziel_map)")
    leerer_manager = DataLODManager()
    leerer_manager.set_map_seed(SEED)
    leere_gen = sp._generatoren(leerer_manager, None)
    wg_leer = leere_gen.get(CALCULATOR_GRAPH["weather.temperature"].generator)
    dummy_wind = np.ones((32, 32, 2), dtype=np.float32)
    faktor = wg_leer._wind_regional_faktor(dummy_wind, lod_level=LOD)  # nie terrain.redistribution gerechnet
    if faktor is not None:
        fehler.append("wind_regional_faktor liefert etwas ohne je gerechnetes "
                      "terrain.redistribution (sollte None sein)")
    else:
        print("   ok - None ohne verfuegbare wind_ziel_map/region_map")

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
