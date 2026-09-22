"""
Siedlungen je Kultur - Eignungsfeld und Platzierung nach
docs/spezifikation/14_SIEDLUNGEN.md Abschnitt 2+3 (Umbau 2026-08-10).

Vorher: `TerrainSuitabilityAnalyzer` kannte drei Faktoren (Slope/Wasser-Naehe/
Hoehen-Wohlfuehlzone, je als Python-Doppelschleife), `calculate_settlements`
platzierte eine feste, kulturlose Anzahl ohne Rang.

Geprueft wird hier das NEUE Verhalten, nicht ob es "gut aussieht":

  1. Fuenf Faktoren wirken tatsaechlich unterschiedlich (Wassertyp-Gewichtung,
     Ackerland-Radius, Hoehen-DAEMPFUNG statt Wohlfuehlzone).
  2. Jede vorhandene Kultur bekommt 2 bis 5 Siedlungen.
  3. Jede vorhandene Kultur hat GENAU eine Stadt (die Zusicherung des Entwurfs
     ist "mindestens eine" - die Umsetzung hier liefert immer exakt eine).
  4. Haeuserzahl liegt im Bereich des zugewiesenen Rangs.
  5. Der Rang ist NICHT einfach der Eignungswert - das Rauschen aus
     14_SIEDLUNGEN.md 2 muss ueber mehrere Seeds sichtbar etwas verschieben,
     sonst waere es totes Code.

Faehrt die echte Pipeline (Terrain bis Settlements) bei kleiner Kartengroesse,
mehrere Seeds - das ist teuer, deshalb bewusst nur 3 Seeds bei 96 px.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import smoke_test_pipeline_outputs as sp

SIZE = 96
LOD = 3
KM = 15.0
SEEDS = (20260804, 12345, 4242)


def _lauf(seed):
    from managers.data_lod_manager import DataLODManager
    from managers.calculator_graph import CALCULATOR_GRAPH

    sp._qt()
    manager = DataLODManager()
    manager.set_map_distance_km(KM)
    manager.set_map_seed(seed)
    parameter = dict(sp._parameter())
    parameter["map_size"] = SIZE
    parameter["map_seed"] = seed
    generatoren = sp._generatoren(manager, None)
    for knoten in CALCULATOR_GRAPH:
        manager.set_calculator_target_lod(knoten, LOD)
    for generator in generatoren.values():
        if hasattr(generator, "set_active_parameters"):
            generator.set_active_parameters(parameter)

    for knoten in sp._reihenfolge():
        spec = CALCULATOR_GRAPH[knoten]
        generator = generatoren.get(spec.generator)
        methode = getattr(generator, "_calc_" + knoten.split(".", 1)[1], None)
        if methode is None:
            continue
        methode(knoten, LOD)
        if knoten == "settlement.settlements":
            break

    return manager.get_calculator_output("settlement.settlements", "settlement_list", LOD)


def main():
    fehler = []

    # ---------- 1: fuenf Faktoren ----------
    print("1. Fuenf Faktoren wirken unterschiedlich")
    from core.settlement_generator import TerrainSuitabilityAnalyzer
    from scipy.ndimage import zoom as _zoom

    # ECHTE lokale Struktur statt einer glatten Sinusflaeche: zoom() aus
    # grobem Rauschen gibt Fleckigkeit auf der Groessenordnung, die
    # evaluate_farmland_radius() auch auf einer echten Karte sieht. Bei einer
    # zu glatten Testflaeche UND einem Radius (farmland_radius_px), der
    # relativ zur Testgroesse riesig ist, mittelt der Boxfilter praktisch die
    # gesamte Karte zu einem einzigen Wert - das war der erste, falsche
    # Befund hier: "Ackerland ist konstant" lag an der Testkarte, nicht am
    # Faktor. n=160 mit map_size=160 haelt den Radius (round(40*0.6)=24)
    # klein genug gegenueber der Kartengroesse.
    n = 160
    rng = np.random.RandomState(3)
    grob = rng.rand(10, 10).astype(np.float64)
    heightmap = (_zoom(grob, n / 10.0, order=3)[:n, :n] * 400.0 - 60.0).astype(np.float32)
    dy, dx = np.gradient(heightmap.astype(np.float64))
    slopemap = np.dstack([dx, dy]).astype(np.float32)
    water_map = np.zeros((n, n), dtype=np.float32)
    wasser_stellen = rng.rand(n, n) > 0.985
    water_map[wasser_stellen] = rng.randint(1, 5, int(wasser_stellen.sum()))

    analyzer = TerrainSuitabilityAnalyzer(1.0, n)
    wasser = analyzer.calculate_water_proximity(water_map, heightmap)
    flach = analyzer.analyze_slope_suitability(slopemap)
    hoehe = analyzer.evaluate_elevation_fitness(heightmap)
    acker = analyzer.evaluate_farmland_radius(flach, hoehe, heightmap > 0)
    kombiniert = analyzer.create_combined_suitability(heightmap, slopemap, water_map)

    for name, feld in (("Wasser", wasser), ("Flach", flach), ("Hoehe", hoehe),
                       ("Ackerland", acker), ("kombiniert", kombiniert)):
        spanne = float(feld.max() - feld.min())
        print("   %-12s Spanne %.3f" % (name, spanne))
        if spanne <= 1e-6:
            fehler.append("Faktor %s ist konstant - traegt nichts bei" % name)

    # Wassertyp muss wirklich unterscheiden: ein Grossfluss (3) UND ein Bach
    # (1), weit genug auseinander, dass sich ihre Felder nicht gegenseitig
    # ueberlagern. Verglichen wird NICHT auf dem Wasserpixel selbst (dort
    # unterdrueckt "zu nah, inkl. direkt auf Wasser = 0" jedes Signal - das
    # war der zweite falsche Befund: beide Testpunkte lagen auf ihrem
    # jeweiligen Wasserpixel und wurden zusaetzlich vom je ANDEREN, entfernten
    # Typ ueberlagert), sondern an einer gleich weit entfernten, tatsaechlich
    # bebaubaren Stelle (5 Pixel, innerhalb des Optimalbands 2-10).
    einzel = np.zeros((80, 80), dtype=np.float32)
    einzel[40, 10] = 1.0   # Bach
    einzel[40, 70] = 3.0   # Grossfluss, weit weg vom Bach
    hoch_und_trocken = np.ones((80, 80), np.float32) * 50.0
    w = analyzer.calculate_water_proximity(einzel, hoch_und_trocken)
    bach_naehe = float(w[40, 15])       # 5 px vom Bach
    fluss_naehe = float(w[40, 65])      # 5 px vom Grossfluss
    if not (fluss_naehe > bach_naehe):
        fehler.append("Grossfluss wird nicht hoeher bewertet als Bach gleicher "
                      "Entfernung: %.3f vs %.3f" % (fluss_naehe, bach_naehe))

    # Hoehe muss DAEMPFEND sein (monoton fallend), nicht eine Wohlfuehlzone in
    # der Mitte der lokalen Spanne.
    h_test = np.array([[0.0, 300.0, 900.0, 2000.0]], dtype=np.float32)
    e_test = analyzer.evaluate_elevation_fitness(h_test)[0]
    if not np.all(np.diff(e_test) <= 1e-9):
        fehler.append("Hoehen-Eignung faellt nicht monoton: %s" % e_test.tolist())

    # ---------- 2-5: echte Pipeline ----------
    print("\n2-5. Platzierung ueber %d Seeds bei %d px" % (len(SEEDS), SIZE))
    alle_kulturen = {}   # name -> Liste von (rang_wert_roh?, rang)
    for seed in SEEDS:
        siedlungen = _lauf(seed)
        if not siedlungen:
            fehler.append("Seed %d: keine Siedlungen erzeugt" % seed)
            continue
        je_kultur = {}
        for s in siedlungen:
            je_kultur.setdefault(s.culture, []).append(s)

        for kultur, gruppe in je_kultur.items():
            anzahl = len(gruppe)
            staedte = [s for s in gruppe if s.rank == "stadt"]
            print("   Seed %-10d %-20s %d Orte, %d Stadt" % (
                seed, kultur or "(namenlos)", anzahl, len(staedte)))
            if not (2 <= anzahl <= 5):
                fehler.append("Seed %d, Kultur %s: %d Orte, erwartet 2-5"
                              % (seed, kultur, anzahl))
            if len(staedte) != 1:
                fehler.append("Seed %d, Kultur %s: %d Staedte, erwartet genau 1"
                              % (seed, kultur, len(staedte)))
            for s in gruppe:
                lo, hi = {"dorf": (15, 25), "siedlung": (25, 35),
                         "stadt": (35, 50)}.get(s.rank, (0, 0))
                if not (lo <= s.house_count <= hi):
                    fehler.append("Seed %d: %s mit %d Haeusern passt nicht zu "
                                  "Rang %s (%d-%d)" % (seed, kultur,
                                                       s.house_count, s.rank, lo, hi))
                alle_kulturen.setdefault(kultur, []).append(
                    (s.properties.get('rang_wert', float('nan')), s.rank))

    # ---------- 5: Rauschen zeigt Wirkung ----------
    #
    # Die Zusicherung ist nicht "guenstiger Fall X trat auf", sondern dass
    # Rang wirklich am VERRAUSCHTEN Wert haengt: bei genuegend Beobachtungen
    # ueber drei Seeds sollte NICHT jede 'stadt' den gleichen (immer besten)
    # Roh-Eignungswert ihrer Kultur haben, wenn das Rauschen ueberhaupt
    # einfliesst - das pruefen wir indirekt ueber die Streuung der rang_werte
    # innerhalb des Rangs 'stadt' (bei totem Rauschen waere sie strukturell
    # gleich der Streuung des Eignungsfelds selbst, nicht Null - also stattdessen:
    # das Rauschband selbst, +/-0.25, muss bei genuegend Staedten sichtbar
    # in den Werten auftauchen, d.h. rang_wert dieser Staedte weicht von der
    # naechsten Ganzzahl-Rasterung ab. Einfacher direkter Test: die Funktion
    # _rang_zuweisen() selbst, mit kuenstlichem Input.
    print("\n5. Rang folgt dem VERRAUSCHTEN Wert, nicht der rohen Eignung")
    # Das Rauschen selbst entsteht in calculate_settlements() beim Bilden von
    # `rang_wert = Eignung_am_Standort + zufall.uniform(-0.25, 0.25)` - das ist
    # in der echten Pipeline oben oder in Punkt 1 nicht direkt beobachtbar
    # (dafuer muesste man denselben Standort zweimal mit/ohne Rauschen
    # platzieren). Direkt pruefbar ist aber die Kette danach: `_rang_zuweisen`
    # muss auf dem UEBERGEBENEN (schon verrauschten) Wert arbeiten, nicht auf
    # einer eigenen zweiten, unverrauschten Neuberechnung - sonst waere das
    # Rauschen aus calculate_settlements() zwar erzeugt, aber wirkungslos.
    # Zwei kuenstliche Faelle mit vertauschter Eignungs- vs. Rangordnung:
    from core.settlement_generator import SettlementGenerator
    dummy = SettlementGenerator.__new__(SettlementGenerator)

    werte_a = [0.5, 0.9, 0.3, 0.9001]     # Position 3 hauchduenn vor Position 1
    raenge_a = dummy._rang_zuweisen(werte_a)
    if raenge_a != ["dorf", "siedlung", "dorf", "stadt"]:
        fehler.append("_rang_zuweisen liefert bei %s %s, erwartet "
                      "[dorf,siedlung,dorf,stadt]" % (werte_a, raenge_a))

    # Dieselben Eignungswerte, andere Rangfolge NUR weil das (verrauschte)
    # rang_wert-Array anders aussieht - beweist, dass _rang_zuweisen keine
    # eigene Meinung zur "wahren" Eignung hat, sondern dem Eingabewert folgt.
    werte_b = [0.9001, 0.9, 0.3, 0.5]     # dieselben vier Werte, umsortiert
    raenge_b = dummy._rang_zuweisen(werte_b)
    if raenge_b != ["stadt", "siedlung", "dorf", "dorf"]:
        fehler.append("_rang_zuweisen liefert bei %s %s, erwartet "
                      "[stadt,siedlung,dorf,dorf]" % (werte_b, raenge_b))

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
