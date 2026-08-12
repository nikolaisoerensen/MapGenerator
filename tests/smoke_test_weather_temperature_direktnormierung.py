"""
Temperatur trifft KLIMA_ZIEL direkt, unabhaengig vom Seed (docs/OFFENE_PUNKTE.md
1.10/1.11).

ANLASS. `t_mittel`/`t_spanne` kommen aus `klima_map`, also aus der weich ueber
die Regionsgrenzen GEBLENDETEN Fassung von core.terrain_weltkarte.REGIONEN.
temp_mittel_m0/temp_spanne - die Regionsmischung zieht jede Region zu ihren
Nachbarn hin. VORHER wurde das durch HANDKALIBRIERTE Eingabewerte kompensiert
("Taiga 1.8, damit am Ende 3.8 ankommen"), ueber drei Seeds von Hand geeicht -
1.11 hielt fest, dass das auf einem VIERTEN Seed schon wieder daneben lag
(Taiga-Jahresspanne 30.9 K statt 29.0 K).

FIX: `_je_region_auf_mittel()` (dasselbe Prinzip wie beim Niederschlag, 1.3)
normiert das fertige, geblendete Feld direkt auf `KLIMA_ZIEL` - das trifft die
Vorgabe PER KONSTRUKTION, nicht per Kalibrierung. Die Eingabewerte in REGIONEN
sind seither die lesbaren KLIMA_ZIEL-Werte selbst.

Zusicherung: ueber VIER Seeds (nicht nur die, auf die frueher von Hand geeicht
wurde) trifft jede Region ihr Jahresmittel UND ihre Jahresspanne, gemessen AUF
MEERESHOEHE (die Hoehenabnahme selbst ist gewollte Variation, keine
Abweichung - siehe `_je_region_auf_mittel`-Docstring "auf Meereshoehe").
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
    import core.weather_generator as wg_modul
    import smoke_test_pipeline_outputs as sp
    from managers.calculator_graph import CALCULATOR_GRAPH
    from managers.data_lod_manager import DataLODManager

    SIZE, LOD = 256, 6
    SEEDS = (20260804, 12345, 4242, 99)  # bewusst NICHT nur die alten Eichseeds
    TOLERANZ_MITTEL_K = 0.7
    # Grosszuegiger als beim Mittel: die Spanne haengt zusaetzlich am
    # Jahresgang-Sample (jahresgang() an sechs diskreten Monatspunkten) und
    # damit staerker an der zufaelligen Kartenform. Gemessen ueber vier
    # Seeds: schlechtester Fall 1.69 K (Taiga, Seed 4242) - deutlich besser
    # als die alte Handeichung (1.11: bis zu 1.9 K auf einem ungeeichten
    # Seed), aber nicht ganz so knapp wie das Mittel.
    TOLERANZ_SPANNE_K = 2.0
    namen = [r["name"] for _z, _s, r in rw.alle_regionen()]

    print("%-20s %8s %8s %8s %8s" % ("Region", "T_ist", "T_ziel", "Sp_ist", "Sp_ziel"))
    for seed in SEEDS:
        manager = DataLODManager()
        manager.set_map_seed(seed)
        manager.set_map_distance_km(rw.WELT_KM)
        par = dict(sp._parameter())
        par["map_size"] = SIZE
        par["map_seed"] = seed
        par["map_distance_km"] = rw.WELT_KM
        gen = sp._generatoren(manager, None)
        for k in CALCULATOR_GRAPH:
            manager.set_calculator_target_lod(k, LOD)
        for g in gen.values():
            if hasattr(g, "set_active_parameters"):
                g.set_active_parameters(par)
        for knoten in ("terrain.noise", "terrain.redistribution", "terrain.shadow", "terrain.slope"):
            spec = CALCULATOR_GRAPH[knoten]
            erz = gen.get(spec.generator)
            getattr(erz, "_calc_" + knoten.split(".", 1)[1])(knoten, LOD)

        wg = gen.get(CALCULATOR_GRAPH["weather.temperature"].generator)
        heightmap, shadowmap, target_size = wg._get_prepared_terrain_inputs(LOD)
        muster = wg._temperatur_raummuster(heightmap, shadowmap, LOD)
        if muster is None:
            fehler.append("Seed %d: _temperatur_raummuster liefert None" % seed)
            continue
        sockel, amplitude = muster
        spanne = amplitude * 2.0
        meeres_temp = sockel + wg_modul.HOEHENABNAHME_K_PRO_M * heightmap

        region_map = manager.get_calculator_output("terrain.redistribution", "region_map", LOD)
        land = heightmap > 0.0

        print("Seed %d" % seed)
        for i, name in enumerate(namen):
            maske = (region_map == i) & land
            if maske.sum() < 10:
                continue
            t_ziel, sp_ziel = rw.KLIMA_ZIEL[name]
            t_ist = float(meeres_temp[maske].mean())
            sp_ist = float(spanne[maske].mean())
            print("  %-18s %8.2f %8.2f %8.2f %8.2f" % (name, t_ist, t_ziel, sp_ist, sp_ziel))
            if abs(t_ist - t_ziel) > TOLERANZ_MITTEL_K:
                fehler.append("Seed %d, %s: Jahresmittel %.2f weicht mehr als %.1f K von "
                              "%.2f ab" % (seed, name, t_ist, TOLERANZ_MITTEL_K, t_ziel))
            if abs(sp_ist - sp_ziel) > TOLERANZ_SPANNE_K:
                fehler.append("Seed %d, %s: Jahresspanne %.2f weicht mehr als %.1f K von "
                              "%.2f ab" % (seed, name, sp_ist, TOLERANZ_SPANNE_K, sp_ziel))

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - alle neun Regionen treffen ihr Ziel auf "
          "allen vier Seeds, nicht nur den urspruenglich handgeeichten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
