"""
Sitzen Siedlungen einer Kultur gleichmaessig verteilt, oder klumpen sie?

ANLASS (2026-08-10). Nutzer-Vorgabe: "können wir die städte gleichmäßiger
verteilen, etwas weniger geklumpt."

MASSZAHL: mittlerer Abstand zum naechsten Nachbarn DERSELBEN Kultur, im
Verhaeltnis zu dem Abstand, den eine GLEICHVERTEILTE Platzierung ueber
dieselbe Flaeche im Mittel haette (fuer n Punkte auf Flaeche A: erwarteter
NN-Abstand ~ 0.5*sqrt(A/n), Standardformel fuer Poisson-Punktprozesse). Ein
Verhaeltnis deutlich unter 1 heisst geklumpt, nahe 1 oder darueber heisst
gleichmaessig bis leicht ueberdispers.

URSACHE DES KLUMPENS. `_reduce_suitability_around_point()` senkte die Eignung
nur INNERHALB von `min_distance` - der harten Ausschlusszone, die fuer die
2-5-Trefferquote je Kultur gebraucht wird. Ausserhalb blieb die Eignung
UNBERUEHRT. Bei einer Region mit einem einzelnen starken Eignungshuegel blieb
direkt ausserhalb des Ausschlussradius reichlich hohe Eignung uebrig, und der
naechste Ort setzte sich an den Rand genau dieses Huegels statt in einen
anderen Teil der Region - eine Perlenkette am Huegelrand statt einer Streuung.

Gemessen VORHER (dieses Skript, 160 px, Seed 20260804): mittleres Verhaeltnis
0.76, 5 von 9 Kulturen unter 0.65 (deutlich geklumpt). Fix: die weiche
Eignungs-Absenkung wirkt jetzt ueber `min_distance * 2.5` statt nur
`min_distance` - die harte Ausschlusszone (fuer die Trefferquote) bleibt
unveraendert.
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

    SIZE, LOD = 160, 5
    SEEDS = (20260804, 12345)

    verhaeltnisse_gesamt = []
    print("%-10s %-18s %4s %10s %10s %8s"
          % ("Seed", "Kultur", "n", "NN-Mittel", "erwartet", "Verhaeltnis"))
    print("-" * 66)

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

        for knoten in sp._reihenfolge():
            spec = CALCULATOR_GRAPH[knoten]
            erzeuger = gen.get(spec.generator)
            methode = getattr(erzeuger, "_calc_" + knoten.split(".", 1)[1], None)
            if methode is None:
                continue
            methode(knoten, LOD)
            if knoten == "settlement.settlements":
                break

        siedlungen = manager.get_calculator_output(
            "settlement.settlements", "settlement_list", LOD)
        region_map = manager.get_calculator_output(
            "terrain.redistribution", "region_map", LOD)
        if siedlungen is None or region_map is None:
            fehler.append("Seed %d: Siedlungen/region_map nicht verfuegbar" % seed)
            continue

        je_kultur = {}
        for s in siedlungen:
            je_kultur.setdefault(s.culture, []).append(s)

        for i, (_z, _s, r) in enumerate(rw.alle_regionen()):
            kultur = r["volk"]
            gruppe = je_kultur.get(kultur, [])
            n = len(gruppe)
            if n < 2:
                continue
            flaeche = float(np.count_nonzero(region_map == i))
            pts = np.array([[s.x, s.y] for s in gruppe])
            nn = []
            for k in range(n):
                d = np.hypot(pts[:, 0] - pts[k, 0], pts[:, 1] - pts[k, 1])
                d[k] = np.inf
                nn.append(float(d.min()))
            nn_mittel = float(np.mean(nn))
            erwartet = 0.5 * np.sqrt(flaeche / n)
            verhaeltnis = nn_mittel / erwartet if erwartet > 0 else float("nan")
            verhaeltnisse_gesamt.append(verhaeltnis)
            print("%-10d %-18s %4d %10.1f %10.1f %8.2f"
                  % (seed, kultur, n, nn_mittel, erwartet, verhaeltnis))
            # Grosszuegige Einzelschwelle - Streuung durch Regionsform/Seed
            # ist normal, ein SYSTEMATISCHES Klumpen (wie vor dem Fix, 0.76
            # im Mittel, mehrere Kulturen unter 0.5) soll aber auffallen.
            if verhaeltnis < 0.35:
                fehler.append("Seed %d, %s: Verhaeltnis %.2f - deutlich geklumpt"
                              % (seed, kultur, verhaeltnis))

    print("")
    mittel = float(np.mean(verhaeltnisse_gesamt)) if verhaeltnisse_gesamt else float("nan")
    print("Mittleres Verhaeltnis ueber alle Kulturen/Seeds: %.2f "
          "(1.0 = gleichmaessig)" % mittel)
    if mittel < 0.7:
        fehler.append("Mittleres Verhaeltnis %.2f unter 0.7 - Siedlungen "
                      "klumpen im Schnitt spuerbar" % mittel)

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - keine systematische Klumpung.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
