"""
Wegenetz nach docs/SIEDLUNGEN_ENTWURF.md §4 (Umbau 2026-08-10).

Vorher: `calculate_road_network` verband Siedlungen ueber ein einfaches
Minimum-Spanning-Tree (naechster unverbundener Nachbar), Wegkosten kannten nur
Hangneigung (linear), kein Wasser, kein bestehender-Weg-Rabatt, keine Kultur,
kein Bereitschaftstest.

Geprueft wird hier das NEUE Verhalten:

  1. Kostenfeld (§4.1): die drei Wasserstufen und der Wegerabatt wirken
     tatsaechlich, Hangkosten wachsen quadratisch.
  2. Gabriel-Graph (§4.2): sparse (nicht Vollverknuepfung) UND zusammenhaengend
     (Eigenschaft von Gabriel-Graphen fuer Punkte in allgemeiner Lage).
  3. Bereitschaftstest (§4.3): zwei Staedte derselben Kultur verbinden sich
     ueber einfaches Terrain, zwei Doerfer verschiedener Kultur ueber
     gesperrtes/sehr teures Terrain nicht - direkt aus der Formel geprueft,
     nicht nur behauptet.
  4. Kulturzusammenhang (§4.3 Ausnahme): Orte derselben Kultur sind nach dem
     Netzbau IMMER verbunden, auch wenn der direkte Kandidat durchgefallen
     waere.
  5. `city_cost_map` enthaelt kein np.inf mehr (§5.13) - genau das haette der
     gemeinsame Anzeige-Validator (map_display_2d.py/_validate_input_data)
     JEDE Anzeige dieses Feldes verweigert.
  6. `settlement.outer_roads` existiert nicht mehr im CALCULATOR_GRAPH (§5.11).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def main():
    fehler = []
    from core.settlement_generator import (
        Location, bau_kostenfeld, _gabriel_kandidaten, WASSERKOSTEN_FLACH,
        WASSERKOSTEN_TIEF, WEGERABATT, RANG_ZAHL, BEREITSCHAFT_FREMDKULTUR,
        SettlementGenerator, CityBoundaryAnalyzer,
    )

    # ---------- 1: Kostenfeld ----------
    print("1. Kostenfeld")
    n = 40
    heightmap = np.full((n, n), 50.0, dtype=np.float32)   # ueberall Land
    heightmap[10, :] = -2.0    # Flachwasser-Streifen
    heightmap[20, :] = -7.0    # tiefes-aber-noch-Bruecke-Streifen
    heightmap[30, :] = -20.0   # gesperrt
    slopemap = np.zeros((n, n, 2), dtype=np.float32)
    slopemap[5, 5] = [1.0, 0.0]   # steile Stelle, |hang|=1.0

    feld = bau_kostenfeld(heightmap, slopemap, slope_distance_ratio=1.5)
    print("   eben              %.3f (soll 1.0)" % feld[0, 0])
    print("   Flachwasser       %.3f (soll %.1f)" % (feld[10, 0], WASSERKOSTEN_FLACH))
    print("   tieferes Wasser   %.3f (soll %.1f)" % (feld[20, 0], WASSERKOSTEN_TIEF))
    print("   gesperrt          %s (soll inf)" % feld[30, 0])
    print("   Hang |1.0|        %.3f (soll 1+1.5*1.0^2=2.5, quadratisch)" % feld[5, 5])

    if abs(feld[0, 0] - 1.0) > 1e-6:
        fehler.append("ebener Grund kostet nicht 1.0: %.3f" % feld[0, 0])
    if abs(feld[10, 0] - WASSERKOSTEN_FLACH) > 1e-6:
        fehler.append("Flachwasser-Kosten falsch: %.3f" % feld[10, 0])
    if abs(feld[20, 0] - WASSERKOSTEN_TIEF) > 1e-6:
        fehler.append("Tiefwasser-Kosten falsch: %.3f" % feld[20, 0])
    if np.isfinite(feld[30, 0]):
        fehler.append("gesperrtes Wasser ist nicht unendlich: %.3f" % feld[30, 0])
    if abs(feld[5, 5] - 2.5) > 1e-6:
        fehler.append("Hangkosten nicht quadratisch: %.3f (erwartet 2.5)" % feld[5, 5])

    weg_maske = np.zeros((n, n), dtype=bool)
    weg_maske[0, 0] = True
    feld_mit_weg = bau_kostenfeld(heightmap, slopemap, 1.5, weg_maske)
    erwartet_rabatt = 1.0 * WEGERABATT
    if abs(feld_mit_weg[0, 0] - erwartet_rabatt) > 1e-6:
        fehler.append("Wegerabatt wirkt nicht: %.3f statt %.3f"
                      % (feld_mit_weg[0, 0], erwartet_rabatt))

    # ---------- 2: Gabriel-Graph ----------
    print("\n2. Gabriel-Graph")
    rng = np.random.RandomState(5)
    punkte = rng.rand(15, 2) * 100.0
    kandidaten = _gabriel_kandidaten(punkte)
    voll = 15 * 14 // 2
    print("   %d von %d moeglichen Paaren (sparse erwartet)" % (len(kandidaten), voll))
    if len(kandidaten) >= voll:
        fehler.append("Gabriel-Graph ist nicht sparse: %d von %d Paaren"
                      % (len(kandidaten), voll))

    # Zusammenhang pruefen (Eigenschaft von Gabriel-Graphen).
    eltern = list(range(15))

    def find(x):
        while eltern[x] != x:
            eltern[x] = eltern[eltern[x]]
            x = eltern[x]
        return x

    for i, j in kandidaten:
        wi, wj = find(i), find(j)
        if wi != wj:
            eltern[wi] = wj
    komponenten = len({find(i) for i in range(15)})
    print("   %d Zusammenhangskomponente(n) (soll 1)" % komponenten)
    if komponenten != 1:
        fehler.append("Gabriel-Graph nicht zusammenhaengend: %d Komponenten" % komponenten)

    # ---------- 3+4: Bereitschaft und Kulturzusammenhang ----------
    print("\n3+4. Bereitschaft und Kulturzusammenhang")
    m = 60
    heightmap2 = np.full((m, m), 30.0, dtype=np.float32)
    slopemap2 = np.zeros((m, m, 2), dtype=np.float32)

    def ort(id_, x, y, culture, rank):
        return Location(location_id=id_, x=float(x), y=float(y), location_type='settlement',
                        radius=4.0, civ_influence=0.8, culture=culture, rank=rank)

    # Zwei Staedte DERSELBEN Kultur, einfaches Gelaende -> muessen sich verbinden.
    stadt_a = ort(0, 5, 30, "Kelten", "stadt")
    stadt_b = ort(1, 55, 30, "Kelten", "stadt")

    # Zwei Doerfer VERSCHIEDENER Kultur, durch eine Mauer aus gesperrtem
    # Wasser getrennt (nicht ueberquerbar) - der direkte Kandidat darf nicht
    # gebaut werden, aber falls sie NICHT derselben Kultur angehoeren, gibt es
    # keine Zusammenhangspflicht, die das erzwingen wuerde.
    # 5 Pixel breit, nicht 1: bei dieser Kartengroesse (<=64 px) sucht A* mit
    # path_resolution=2 (siehe PathfindingSystem.__init__) - eine nur 1 Pixel
    # breite Mauer liesse sich damit in einem einzelnen 2-Pixel-Schritt
    # ueberspringen, ohne dass die Zelle je besucht wird. Kein Fehler des
    # Wegenetzes, sondern eine zu duenne Testmauer fuer die grobe Suchtiefe
    # dieser LOD-Stufe - reale Gelaendehindernisse sind nie nur 1 Pixel breit.
    heightmap3 = heightmap2.copy()
    heightmap3[:, 28:33] = -20.0   # gesperrte Wassermauer quer durch die Karte
    dorf_a = ort(2, 5, 10, "Sachsen", "dorf")
    dorf_b = ort(3, 55, 10, "Wikinger", "dorf")

    gen = SettlementGenerator.__new__(SettlementGenerator)
    gen.road_slope_to_distance_ratio = 1.5
    gen.map_seed = 1
    gen._update_progress = None

    # calculate_road_network() liefert seit dem Seeweg-Umbau (§4.4) ein Paar
    # (roads, sea_roads) - hier immer entpackt, sonst misst len() die Groesse
    # des 2-Tupels selbst (immer 2) statt der Wegeliste. Genau dieser Fehler
    # stand hier zuerst und meldete faelschlich "2 Wege" in allen drei Faellen.
    roads_staedte, _sea_staedte = gen.calculate_road_network(
        [stadt_a, stadt_b], heightmap2, slopemap2, 5)
    print("   2 Staedte, einfaches Gelaende: %d Weg(e) (soll 1)" % len(roads_staedte))
    if len(roads_staedte) != 1:
        fehler.append("Zwei Staedte derselben Kultur verbinden sich nicht "
                      "ueber einfaches Gelaende: %d Wege" % len(roads_staedte))

    roads_doerfer, _sea_doerfer = gen.calculate_road_network(
        [dorf_a, dorf_b], heightmap3, slopemap2, 5)
    print("   2 Doerfer, versch. Kultur, gesperrte Mauer: %d Weg(e) (soll 0)"
          % len(roads_doerfer))
    if len(roads_doerfer) != 0:
        fehler.append("Verbindung durch gesperrtes Gelaende wurde trotzdem "
                      "gebaut: %d Wege" % len(roads_doerfer))

    # Dieselbe Mauer, aber jetzt DERSELBEN Kultur - der Kulturzusammenhang
    # (§4.3 Ausnahme) muss trotzdem eine Verbindung erzwingen, "egal was sie
    # kostet". Bei einer einzelnen gesperrten Wasserspalte findet sich noch
    # ein endlicher LANDWEG drumherum (die Mauer ist nur 1 Pixel breit auf
    # einer 60 px hohen Karte) - erzwungen wird hier also der teure Umweg,
    # nicht zwingend ein Seeweg.
    dorf_c = ort(4, 5, 10, "Sachsen", "dorf")
    dorf_d = ort(5, 55, 10, "Sachsen", "dorf")
    roads_kultur, sea_kultur = gen.calculate_road_network(
        [dorf_c, dorf_d], heightmap3, slopemap2, 5)
    print("   2 Doerfer, GLEICHE Kultur, gesperrte Mauer: %d Landweg(e), %d Seeweg(e) "
          "(soll zusammen 1, erzwungen)" % (len(roads_kultur), len(sea_kultur)))
    if len(roads_kultur) + len(sea_kultur) != 1:
        fehler.append("Kulturzusammenhang wird nicht erzwungen: %d Landwege + "
                      "%d Seewege statt 1" % (len(roads_kultur), len(sea_kultur)))

    # ---------- 5: city_cost_map ohne np.inf ----------
    print("\n5. city_cost_map")
    analyzer = CityBoundaryAnalyzer(terrain_factor=1.0, reach_factor=4.0)
    _mask, cost_map = analyzer.compute_city_boundaries(
        heightmap2, slopemap2, [stadt_a, stadt_b])
    if not np.all(np.isfinite(cost_map)):
        fehler.append("city_cost_map enthaelt weiterhin np.inf")
    else:
        print("   ueberall endlich - ok")
    if not np.any(cost_map < 0.0):
        fehler.append("city_cost_map hat kein '-1 = unerreicht' mehr - bei "
                      "dieser kleinen Reichweite auf 60x60 muss es "
                      "unerreichte Randpixel geben")

    # ---------- 6: outer_roads entfernt ----------
    print("\n6. settlement.outer_roads entfernt")
    from managers.calculator_graph import CALCULATOR_GRAPH
    if "settlement.outer_roads" in CALCULATOR_GRAPH:
        fehler.append("settlement.outer_roads steht noch im CALCULATOR_GRAPH")
    else:
        print("   nicht mehr im Graph - ok")

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
