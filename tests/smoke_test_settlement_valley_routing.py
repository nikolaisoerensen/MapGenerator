"""
Folgen Strassen wirklich dem Tal/Pass, oder schneiden sie ueber den Grat?

ANLASS (2026-08-10). Nutzer-Vorgabe: "straßen sollen entsprechend des plans
ja möglichst realistisch durch die täler meandern."

ZWEI GETRENNTE FRAGEN, ZWEI GETRENNTE BEFUNDE:

  1. Folgt die ROHE A*-Route dem Kostenfeld (weicht einem teuren Grat zugunsten
     eines guenstigen Passes aus)? JA, das Kostenfeld selbst war nie das
     Problem.
  2. Verfaelscht apply_spline_smoothing() das wieder, indem es die Route
     geometrisch glaettet, ohne das Gelaende zu kennen? NEIN, gemessen ueber
     mehrere Faelle liegt die geglaettete Route nur wenige Meter ueber der
     rohen - kein Corner-Cutting durch den Grat.

DAS EIGENTLICHE PROBLEM WAR EIN DRITTES: das A*-Suchbudget
(PathfindingSystem.max_search_nodes) stammte aus der Zeit vor dem
Kostenfeld-Umbau (§4.1) und war fuer das neue, viel schaerfere Kostenfeld
(Hangkosten QUADRATISCH, Wasser bis 25x, ganze Bereiche unendlich teuer) zu
knapp bemessen. Ergebnis: die Suche brach ab, bevor sie den Umweg zum Pass
fand, und fiel auf den nutzlosen Geradlinien-Fallback zurueck - der dank des
in dieser Session behobenen Fallback-Kosten-Bugs (siehe
smoke_test_settlement_sites.py) wenigstens nicht mehr faelschlich als
guenstige Route durchging, aber eben auch keine Route mehr lieferte.

Gemessen an einem synthetischen Grat mit einem einzigen Pass: das noetige
Budget wuchs ungefaehr mit dem QUADRAT der Kantenlaenge (100px->10000,
256px->51200, 512px->200000) - PathfindingSystem.__init__ setzt das Budget
seitdem auf rund 1.3*Kantenlaenge^2 (gedeckelt), siehe dortiger Kommentar.

Dieser Test baut ein Gelaende mit einem klaren Grat und einem einzigen Pass,
routet eine Strasse hindurch und prueft, dass sowohl die rohe als auch die
geglaettete Route unter der Grathoehe bleiben - bei mehreren Kartengroessen,
damit die Budget-Skalierung selbst mitgeprueft ist.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _grat_mit_pass(m):
    """Ein glatter Grat quer zur Karte mit einer einzigen Senke (Pass)."""
    yy, xx = np.mgrid[0:m, 0:m].astype(np.float64)
    cx = m / 2.0
    breite = m * 0.1
    grat = 350.0 * np.exp(-((xx - cx) ** 2) / (2 * breite ** 2))
    pass_senke = 320.0 * np.exp(-((xx - cx) ** 2) / (2 * breite ** 2)) \
                * np.exp(-((yy - m * 0.25) ** 2) / (2 * breite ** 2))
    heightmap = (20.0 + grat - pass_senke).astype(np.float32)
    dy, dx = np.gradient(heightmap.astype(np.float64))
    slopemap = np.dstack([dx, dy]).astype(np.float32)
    return heightmap, slopemap


def main():
    fehler = []
    from core.settlement_generator import PathfindingSystem, bau_kostenfeld

    print("%-6s %-10s %-8s %-10s %-12s %s"
          % ("Groesse", "Erreicht", "Laenge", "Max Hoehe", "Zeit", "Ergebnis"))
    print("-" * 68)
    import time

    for m in (100, 256, 512):
        heightmap, slopemap = _grat_mit_pass(m)
        grat_hoehe_fern = float(heightmap[int(m * 0.7), int(m * 0.5)])
        pass_hoehe = float(heightmap[int(m * 0.25), int(m * 0.5)])

        kostenfeld = bau_kostenfeld(heightmap, slopemap, 1.5)
        pf = PathfindingSystem(kostenfeld, m)

        t0 = time.perf_counter()
        roh_pfad, erreicht = pf.find_least_resistance_path(
            (m * 0.2, m * 0.7), (m * 0.8, m * 0.7), None)
        dauer = time.perf_counter() - t0
        geglaettet = pf.apply_spline_smoothing(roh_pfad, smoothing_factor=3)

        def max_hoehe(pfad):
            return max(heightmap[int(np.clip(round(y), 0, m - 1)),
                                 int(np.clip(round(x), 0, m - 1))]
                       for x, y in pfad)

        roh_max = float(max_hoehe(roh_pfad)) if erreicht else float('nan')
        geglaettet_max = float(max_hoehe(geglaettet)) if erreicht else float('nan')
        # Grosszuegige Schwelle (deutlich unter dem Grat ~350-370, deutlich
        # ueber dem Pass ~20-50) - es geht darum, GRAT von PASS zu
        # unterscheiden, nicht um ein exaktes Hoehenlimit.
        schwelle = (grat_hoehe_fern + pass_hoehe) / 2.0

        ok = erreicht and roh_max < schwelle and geglaettet_max < schwelle
        print("%-6d %-10s %-8d %-10.1f %-12.3f %s"
              % (m, erreicht, len(geglaettet) if erreicht else 0,
                 geglaettet_max if erreicht else -1, dauer,
                 "ok" if ok else "DANEBEN"))

        if not erreicht:
            fehler.append("%dpx: keine Route gefunden (Suchbudget zu knapp? "
                          "PathfindingSystem.max_search_nodes=%d)"
                          % (m, pf.max_search_nodes))
        elif roh_max >= schwelle:
            fehler.append("%dpx: rohe Route quert den Grat (%.1f >= Schwelle %.1f) "
                          "- das Kostenfeld/A* weicht dem Grat nicht aus"
                          % (m, roh_max, schwelle))
        elif geglaettet_max >= schwelle:
            fehler.append("%dpx: geglaettete Route quert den Grat (%.1f >= "
                          "Schwelle %.1f), obwohl die rohe Route (%.1f) das "
                          "nicht tat - apply_spline_smoothing() schneidet "
                          "Ecken durch das Gelaende" % (m, geglaettet_max, schwelle, roh_max))

    print("")
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for f in fehler:
            print("   " + f)
        return 1
    print("Alle Zusicherungen erfuellt - Strassen weichen dem Grat aus und "
          "bleiben das auch nach dem Glaetten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
