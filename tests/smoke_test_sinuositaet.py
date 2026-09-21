"""
Path: tests/smoke_test_sinuositaet.py

Prueft `core/fluss_sinuositaet.py` (Ticket #33).

Vier Gruppen, aufsteigend vom Rechenkern zur echten Karte:

  1. `sinuositaet_pfad` an zwei Handfaellen mit bekanntem Ergebnis (gerader
     Lauf = 1.0, Zickzack = sqrt(2)).
  2. `fluss_segmente` an einem von Hand gebauten kleinen Baum - prueft, dass
     die Zerlegung in "Fluesse" (Ketten gleicher Ordnung) stimmt.
  3. `sinuositaet_je_fluss` Ende-zu-Ende an demselben Handbaum.
  4. Ein echter Lauf durch `core.terrain_weltfluesse.flussnetz()` auf einer
     kleinen Karte - keine Zahlenpruefung, nur: stuerzt nicht ab, jede
     Sinuositaet ist >= 1.0 (eine Lauflaenge kann nie kuerzer sein als die
     Luftlinie).

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_sinuositaet.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 20260804


def gerader_lauf_ist_eins():
    """1a. Eine Gerade hat Sinuositaet 1.0 - die Definition selbst."""
    from core.fluss_sinuositaet import sinuositaet_pfad

    fehler = []
    faelle = [
        ("5 Punkte auf einer Achse", np.array([[0, 0], [1, 0], [2, 0],
                                               [3, 0], [4, 0]], dtype=float)),
        ("2 Punkte, diagonal", np.array([[0, 0], [10, 10]], dtype=float)),
        ("unregelmaessig verteilte Punkte auf derselben Geraden",
         np.array([[0, 0], [0.3, 0.3], [1.0, 1.0], [4.0, 4.0]], dtype=float)),
    ]
    for name, punkte in faelle:
        sin = sinuositaet_pfad(punkte)
        ok = abs(sin - 1.0) < 1e-9
        print(f"[{'OK' if ok else 'FEHLER'}] {name}: Sinuositaet {sin:.6f} "
              f"(erwartet 1.0)")
        if not ok:
            fehler.append(f"{name}: {sin} statt 1.0")
    return fehler


def bekannter_zickzack():
    """1b. Ein Saegezahn mit von Hand ausgerechnetem Sollwert."""
    from core.fluss_sinuositaet import sinuositaet_pfad

    fehler = []
    # (0,0)-(1,1)-(2,0)-(3,1)-(4,0): 4 Diagonalstuecke der Laenge sqrt(2),
    # Lauflaenge 4*sqrt(2). Luftlinie (0,0)->(4,0) = 4. Sinuositaet sqrt(2).
    punkte = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]], dtype=float)
    sin = sinuositaet_pfad(punkte)
    erwartet = np.sqrt(2.0)
    ok = abs(sin - erwartet) < 1e-9
    print(f"[{'OK' if ok else 'FEHLER'}] Saegezahn 4x sqrt(2): "
          f"Sinuositaet {sin:.6f} (erwartet {erwartet:.6f})")
    if not ok:
        fehler.append(f"Saegezahn: {sin} statt {erwartet}")

    # Rechteckiger Maeander: (0,0)-(0,1)-(1,1)-(1,2)-(2,2). Lauflaenge = 4
    # (vier Achsenstuecke der Laenge 1), Luftlinie = sqrt(2^2+2^2)=2*sqrt(2).
    # Sinuositaet = 4 / (2*sqrt(2)) = sqrt(2).
    punkte2 = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]], dtype=float)
    sin2 = sinuositaet_pfad(punkte2)
    ok2 = abs(sin2 - erwartet) < 1e-9
    print(f"[{'OK' if ok2 else 'FEHLER'}] Treppen-Maeander: "
          f"Sinuositaet {sin2:.6f} (erwartet {erwartet:.6f})")
    if not ok2:
        fehler.append(f"Treppen-Maeander: {sin2} statt {erwartet}")
    return fehler


def entartete_faelle():
    """1c. Randfaelle: zu kurz oder Anfang=Ende - nan statt vorgetaeuschter 1.0."""
    from core.fluss_sinuositaet import sinuositaet_pfad

    fehler = []
    faelle = [
        ("ein einziger Punkt", np.array([[5.0, 5.0]])),
        ("kein Punkt", np.zeros((0, 2))),
        ("Anfang = Ende (Ring)", np.array([[0, 0], [1, 1], [0, 0]],
                                          dtype=float)),
    ]
    for name, punkte in faelle:
        sin = sinuositaet_pfad(punkte)
        ok = np.isnan(sin)
        print(f"[{'OK' if ok else 'FEHLER'}] {name}: {sin} (erwartet nan, "
              f"nicht still 1.0)")
        if not ok:
            fehler.append(f"{name}: {sin} statt nan")
    return fehler


def _handbaum():
    """
    Ein kleiner Baum von Hand, 7 Knoten, gebaut wie `strahler_order` ihn
    liefern wuerde:

        0 (Quelle, Ordnung 1) --\
                                  >-- 2 (Ordnung 2, Bump: zwei Ordnung-1 treffen)
        1 (Quelle, Ordnung 1) --/
                                       \
        2 --- 3 (Ordnung 2, Fortsetzung) --- 4 (Muendung eines Nebenbachs,
                                                  Ordnung 1) trifft NICHT auf 3,
                                                  sondern:
        5 (Quelle, Ordnung 1) -- 4 (Ordnung 1, Fortsetzung) -- 3
                                       (an 3 treffen Ordnung 2 und Ordnung 1 -
                                        KEIN Bump, weil ungleich: 3 bleibt 2)
        3 --- 6 (Auslass, eltern=-1, Ordnung bleibt 2)

    Erwartete Fluesse (Ketten gleicher Ordnung):
      A: [0, 2]         Ordnung 1  (Quelle 0 bis zum Zusammenfluss bei 2)
      B: [1, 2]         Ordnung 1  (Quelle 1 bis zum Zusammenfluss bei 2)
      C: [5, 4, 3]      Ordnung 1  (Nebenbach 5 ueber 4 bis Muendung bei 3)
      D: [2, 3, 6]      Ordnung 2  (Hauptstrom von der Bump-Stelle 2 bis zum
                                     Auslass 6)
    """
    # Knoten 0..6, Positionen frei erfunden, Baum ueber `eltern`.
    punkte = np.array([
        [0.0, 0.0],   # 0
        [0.0, 2.0],   # 1
        [1.0, 1.0],   # 2
        [3.0, 1.0],   # 3
        [5.0, 1.0],   # 4
        [6.0, 3.0],   # 5
        [4.0, 1.0],   # 6 (Auslass, knapp hinter 3)
    ])
    eltern = np.array([2, 2, 3, 6, 3, 4, -1], dtype=np.int64)
    # `strahler_order` iteriert intern ueber `reihenfolge[::-1]` und braucht
    # DORT die Kinder-vor-Eltern-Reihenfolge [0,1,5,4,2,3,6] - `reihenfolge`
    # selbst ist also deren Umkehrung.
    reihenfolge = np.array([6, 3, 2, 4, 5, 1, 0])
    return punkte, eltern, reihenfolge


def segmentierung_stimmt():
    """2. `fluss_segmente` auf dem Handbaum."""
    import core.terrain_river_network as rn
    from core.fluss_sinuositaet import fluss_segmente

    fehler = []
    punkte, eltern, reihenfolge = _handbaum()
    order = rn.strahler_order(eltern, reihenfolge)
    print("Ordnung je Knoten:", dict(enumerate(order.tolist())))

    segmente = fluss_segmente(eltern, order)
    ketten = sorted(tuple(k) for k in segmente)
    erwartet = sorted([(0, 2), (1, 2), (5, 4, 3), (2, 3, 6)])
    ok = ketten == erwartet
    print(f"[{'OK' if ok else 'FEHLER'}] Ketten: {ketten}")
    print(f"        erwartet: {erwartet}")
    if not ok:
        fehler.append(f"Segmentierung weicht ab: {ketten} statt {erwartet}")
    return fehler


def sinuositaet_je_fluss_stimmt():
    """3. `sinuositaet_je_fluss` Ende-zu-Ende auf dem Handbaum."""
    import core.terrain_river_network as rn
    from core.fluss_sinuositaet import sinuositaet_je_fluss

    fehler = []
    punkte, eltern, reihenfolge = _handbaum()
    order = rn.strahler_order(eltern, reihenfolge)

    ergebnis = sinuositaet_je_fluss(punkte, eltern, order, mpp=1.0)
    nach_kette = {}
    for r in ergebnis:
        nach_kette[tuple(r["knoten"])] = r
    print("Gefundene Fluesse:")
    for k, r in nach_kette.items():
        print(f"  {k}: Ordnung {r['order']}, Sinuositaet {r['sinuositaet']:.4f}, "
              f"Laenge {r['laenge_m']:.4f}")

    def erwarte(kette, order_soll):
        if kette not in nach_kette:
            fehler.append(f"Kette {kette} fehlt im Ergebnis")
            return
        r = nach_kette[kette]
        if r["order"] != order_soll:
            fehler.append(f"{kette}: Ordnung {r['order']} statt {order_soll}")
        # Sinuositaet von Hand: Kette 2->3->6 z.B. hat einen Knick.
        pkt = punkte[list(kette)]
        lauf = float(np.sum(np.linalg.norm(np.diff(pkt, axis=0), axis=1)))
        luft = float(np.linalg.norm(pkt[-1] - pkt[0]))
        soll = lauf / luft
        ok = abs(r["sinuositaet"] - soll) < 1e-9
        print(f"  [{'OK' if ok else 'FEHLER'}] {kette}: {r['sinuositaet']:.6f} "
              f"(erwartet {soll:.6f})")
        if not ok:
            fehler.append(f"{kette}: Sinuositaet {r['sinuositaet']} statt {soll}")

    erwarte((0, 2), 1)
    erwarte((1, 2), 1)
    erwarte((5, 4, 3), 1)
    erwarte((2, 3, 6), 2)

    if len(ergebnis) != 4:
        fehler.append(f"{len(ergebnis)} Fluesse gefunden, erwartet 4")
    return fehler


def echte_karte_stuerzt_nicht_ab():
    """
    4. Gegenprobe an einer echten, kleinen Weltkarte: `flussnetz()` ->
    Strahler-Ordnung -> Sinuositaet je Fluss. Keine Sollwerte, nur:

      * kein Absturz,
      * jede Sinuositaet ist endlich und >= 1.0 (die Lauflaenge kann per
        Definition nie kuerzer als die Luftlinie sein - waere ein Fluss
        darunter, waere die Extraktion selbst falsch, nicht nur ungenau),
      * es werden ueberhaupt Fluesse gefunden.
    """
    import core.terrain_river_network as rn
    from core.terrain_weltkarte import weltfeld
    from core.terrain_weltfluesse import flussnetz
    from core.fluss_sinuositaet import sinuositaet_je_fluss

    fehler = []
    size = 128
    H, felder = weltfeld(size, SEED)
    H = np.asarray(H, dtype=np.float64)
    netz = flussnetz(H, SEED, region_map=felder.get("regionen"),
                     niederschlag_mm=felder.get("niederschlag_mm"))
    if netz is None:
        fehler.append("flussnetz() gab None zurueck - Karte zu klein fuer den Test?")
        return fehler

    order = rn.strahler_order(netz["eltern"], netz["reihenfolge"])
    mpp = 21300.0 / size
    ergebnis = sinuositaet_je_fluss(netz["punkte"], netz["eltern"], order,
                                    mpp=mpp, region_map=felder.get("regionen"))

    n = len(ergebnis)
    unter_eins = [r for r in ergebnis if r["sinuositaet"] < 1.0 - 1e-9]
    unendlich = [r for r in ergebnis if not np.isfinite(r["sinuositaet"])]
    ok = n > 0 and not unter_eins and not unendlich
    print(f"[{'OK' if ok else 'FEHLER'}] {size}px: {n} Fluesse gefunden, "
          f"{len(unter_eins)} unter 1.0, {len(unendlich)} nicht endlich")
    if n > 0:
        werte = np.array([r["sinuositaet"] for r in ergebnis])
        print(f"        Median {np.median(werte):.3f}, "
              f"Min {werte.min():.3f}, Max {werte.max():.3f}")
    if not ok:
        if n == 0:
            fehler.append("keine Fluesse gefunden")
        if unter_eins:
            fehler.append(f"{len(unter_eins)} Fluesse mit Sinuositaet < 1.0 "
                          f"- das kann nicht sein, Extraktion ist falsch")
        if unendlich:
            fehler.append(f"{len(unendlich)} Fluesse mit nicht-endlicher Sinuositaet")
    return fehler


def lauf():
    gruppen = [
        ("gerader_lauf_ist_eins", gerader_lauf_ist_eins),
        ("bekannter_zickzack", bekannter_zickzack),
        ("entartete_faelle", entartete_faelle),
        ("segmentierung_stimmt", segmentierung_stimmt),
        ("sinuositaet_je_fluss_stimmt", sinuositaet_je_fluss_stimmt),
        ("echte_karte_stuerzt_nicht_ab", echte_karte_stuerzt_nicht_ab),
    ]
    ergebnis, alle = {}, []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        f = fn()
        ergebnis[name] = not f
        alle.extend(f)

    print("\n=== SUMMARY ===")
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if alle:
        print(f"\nNICHT IN ORDNUNG - {len(alle)} Befunde:")
        for f in alle:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
