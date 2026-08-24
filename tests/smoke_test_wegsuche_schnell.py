"""
Path: tests/smoke_test_wegsuche_schnell.py

Prueft `core/wegsuche_schnell.py` - den mit numba uebersetzten A*.

DIE KERNZUSICHERUNG ist die erste Gruppe: derselbe Pfad wie der
Python-Pfad, Punkt fuer Punkt. Ein schnellerer Wegsucher, der einen
ANDEREN Weg findet, ist kein Geschwindigkeitsgewinn, sondern eine
Aenderung an der Karte - Strassen liegen dann woanders, Roadsites
ebenfalls, und die Karte ist nicht mehr aus Seed und Reglern
reproduzierbar.

Drei Dinge koennen das brechen, und alle drei sind hier abgedeckt:
Heuristik, Heap-Reihenfolge bei Kostengleichstand, Nachbarreihenfolge.
Die zweite ist die heimtueckischste - sie faellt nur auf, wenn zwei Wege
EXAKT gleich teuer sind, und das kommt auf einem Kostenfeld mit vielen
gleichen Werten (Ebenen!) staendig vor.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_wegsuche_schnell.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.settlement_generator import PathfindingSystem, bau_kostenfeld
from core.wegsuche_schnell import wegsuche, NUMBA_DA

SEED = 20260804


_CACHE = []


def _testfelder():
    """
    Vier Felder, vom entarteten bis zum echten Gelaende.

    Einmal gebaut und gemerkt: fuenf Gruppen fragen dieselbe Liste an, und
    das echte Gelaende darin kostet einen vollen `weltfeld()`-Lauf. Ohne den
    Cache lief dieser Test unnoetig lang - und ein Test, der lange laeuft,
    wird nicht gelaufen.
    """
    if _CACHE:
        return _CACHE
    felder = []

    # 1. Ebene - MAXIMAL viele Kostengleichstaende, der harte Fall fuer die
    #    Heap-Reihenfolge.
    felder.append(("Ebene", np.ones((64, 64)), (2, 2), (60, 60)))

    # 2. Mauer mit Luecke - erzwingt einen Umweg.
    mauer = np.ones((64, 64))
    mauer[10:55, 32] = np.inf
    felder.append(("Mauer mit Luecke", mauer, (5, 5), (60, 60)))

    # 3. Zufallskosten - kaum Gleichstaende, prueft die Arithmetik.
    rng = np.random.default_rng(SEED)
    felder.append(("Zufallskosten", 1.0 + 9.0 * rng.random((96, 96)),
                   (3, 3), (92, 92)))

    # 4. ECHTES GELAENDE in echter Groesse. CLAUDE.md: Tests mit den
    #    wirklichen Eingabegroessen bauen, nicht mit ausgedachten.
    from core.terrain_weltkarte import weltfeld
    H, _f = weltfeld(256, SEED)
    H = np.asarray(H, dtype=np.float32)
    gy, gx = np.gradient(H)
    slope = np.stack([gx, gy], axis=-1).astype(np.float32)
    feld = bau_kostenfeld(H, slope, 1.5)
    land = np.argwhere(H > 50)
    a, b = land[len(land) // 5], land[-len(land) // 5]
    felder.append(("echtes Gelaende 256 px", feld,
                   (int(a[1]), int(a[0])), (int(b[1]), int(b[0]))))
    _CACHE.extend(felder)
    return felder


def gleicher_pfad():
    """1. DIE KERNZUSICHERUNG: Punkt fuer Punkt derselbe Weg."""
    fehler = []
    if not NUMBA_DA:
        print("[FEHLER] numba nicht verfuegbar - der schnelle Pfad laeuft gar nicht")
        return ["numba fehlt"]

    for name, feld, start, ziel in _testfelder():
        n = feld.shape[0]
        pf = PathfindingSystem(feld, n)
        alt = pf._a_stern(start[0], start[1], ziel[0], ziel[1],
                          pf.max_search_nodes, schnell=False)
        neu = wegsuche(feld, start, ziel, pf.max_search_nodes,
                       schritt=pf.path_resolution)

        beide_leer = alt is None and neu is None
        gleich = beide_leer or (
            alt is not None and neu is not None
            and len(alt) == len(neu)
            and all(tuple(p) == tuple(q) for p, q in zip(alt, neu)))
        laenge = len(alt) if alt else 0
        print(f"[{'OK' if gleich else 'FEHLER'}] {name}: "
              f"Python {laenge} Punkte, numba "
              f"{len(neu) if neu else 0} Punkte, identisch {gleich}")
        if not gleich:
            if alt and neu:
                ab = next((i for i, (p, q) in enumerate(zip(alt, neu))
                           if tuple(p) != tuple(q)), min(len(alt), len(neu)))
                fehler.append(f"{name}: weicht ab Punkt {ab} ab "
                              f"({alt[ab] if ab < len(alt) else '-'} gegen "
                              f"{neu[ab] if ab < len(neu) else '-'})")
            else:
                fehler.append(f"{name}: einer der beiden fand keinen Pfad")
    return fehler


def gleiche_kosten():
    """
    2. Gegenprobe ueber die Pfadkosten.

    Selbst wenn zwei Pfade Punkt fuer Punkt gleich sind, koennte die
    Kostenrechnung abweichen. Die Summe ist das, was der
    Bereitschaftstest in calculate_road_network() liest - sie entscheidet,
    OB eine Strasse gebaut wird.
    """
    fehler = []
    if not NUMBA_DA:
        return ["numba fehlt"]
    for name, feld, start, ziel in _testfelder():
        n = feld.shape[0]
        pf = PathfindingSystem(feld, n)
        alt = pf._a_stern(start[0], start[1], ziel[0], ziel[1],
                          pf.max_search_nodes, schnell=False)
        neu = wegsuche(feld, start, ziel, pf.max_search_nodes,
                       schritt=pf.path_resolution)
        if alt is None or neu is None:
            continue
        k_alt = sum(pf.calculate_movement_cost(x, y) for x, y in alt[1:])
        k_neu = sum(pf.calculate_movement_cost(x, y) for x, y in neu[1:])
        ok = abs(k_alt - k_neu) < 1e-9
        print(f"[{'OK' if ok else 'FEHLER'}] {name}: Kosten "
              f"{k_alt:.6f} gegen {k_neu:.6f}")
        if not ok:
            fehler.append(f"{name}: Kosten weichen ab ({k_alt} / {k_neu})")
    return fehler


def schneller():
    """3. Der Zweck der Uebung - und um wieviel."""
    fehler = []
    if not NUMBA_DA:
        return ["numba fehlt"]
    name, feld, start, ziel = _testfelder()[-1]
    n = feld.shape[0]
    pf = PathfindingSystem(feld, n)

    wegsuche(feld, start, ziel, pf.max_search_nodes,
             schritt=pf.path_resolution)          # JIT aufwaermen

    t0 = time.perf_counter()
    pf._a_stern(start[0], start[1], ziel[0], ziel[1], pf.max_search_nodes,
                schnell=False)
    t_alt = time.perf_counter() - t0

    t0 = time.perf_counter()
    wegsuche(feld, start, ziel, pf.max_search_nodes,
             schritt=pf.path_resolution)
    t_neu = time.perf_counter() - t0

    faktor = t_alt / max(t_neu, 1e-9)
    ok = faktor > 3.0
    print(f"[{'OK' if ok else 'FEHLER'}] {name}: Python {t_alt:.3f}s, "
          f"numba {t_neu:.4f}s -> {faktor:.1f}x")
    if not ok:
        fehler.append(f"nur {faktor:.1f}x schneller - unter der Erwartung von 3x")
    return fehler


def randfaelle():
    """4. Start gleich Ziel, gesperrtes Ziel, Ziel ausserhalb."""
    fehler = []
    if not NUMBA_DA:
        return ["numba fehlt"]
    feld = np.ones((32, 32))
    feld[16, :] = np.inf                       # Karte vollstaendig geteilt

    faelle = [
        ("Start gleich Ziel", np.ones((32, 32)), (5, 5), (5, 5), True),
        ("unerreichbar", feld, (5, 5), (5, 25), False),
        ("gesperrtes Ziel", feld, (5, 5), (5, 16), False),
    ]
    for name, f, a, b, soll_gefunden in faelle:
        try:
            p = wegsuche(f, a, b, 100000)
            ok = (p is not None) == soll_gefunden
            print(f"[{'OK' if ok else 'FEHLER'}] {name}: gefunden "
                  f"{p is not None} (erwartet {soll_gefunden})")
            if not ok:
                fehler.append(f"{name}: gefunden {p is not None}")
        except Exception as e:                            # noqa: BLE001
            print(f"[FEHLER] {name}: Absturz - {e}")
            fehler.append(f"{name}: Absturz - {e}")
    return fehler


def gewichtete_heuristik():
    """
    5. h_gewicht > 1 ist schneller und hoechstens h_gewicht-mal teurer.

    Das ist die Zusicherung von Weighted A* - kein Wunsch, sondern ein
    Satz. Wird sie verletzt, ist die Heuristik falsch implementiert.
    """
    fehler = []
    if not NUMBA_DA:
        return ["numba fehlt"]
    name, feld, start, ziel = _testfelder()[-1]
    n = feld.shape[0]
    pf = PathfindingSystem(feld, n)

    p1 = wegsuche(feld, start, ziel, pf.max_search_nodes)
    if p1 is None:
        print("[FEHLER] Grundlauf fand keinen Pfad")
        return ["Grundlauf ohne Pfad"]
    k1 = sum(pf.calculate_movement_cost(x, y) for x, y in p1[1:])

    for w in (1.2, 1.5):
        t0 = time.perf_counter()
        p2 = wegsuche(feld, start, ziel, pf.max_search_nodes, h_gewicht=w)
        t2 = time.perf_counter() - t0
        k2 = sum(pf.calculate_movement_cost(x, y) for x, y in p2[1:])
        schranke = w * k1
        ok = k2 <= schranke * (1 + 1e-9)
        print(f"[{'OK' if ok else 'FEHLER'}] {name} w={w}: Kosten {k2:.1f} "
              f"gegen optimal {k1:.1f} (Schranke {schranke:.1f}), "
              f"{100.0 * (k2 / k1 - 1.0):+.1f} %, {t2:.4f}s")
        if not ok:
            fehler.append(f"w={w}: {k2:.1f} ueber der Schranke {schranke:.1f}")
    return fehler


def lauf():
    gruppen = [
        ("gleicher_pfad", gleicher_pfad),
        ("gleiche_kosten", gleiche_kosten),
        ("schneller", schneller),
        ("randfaelle", randfaelle),
        ("gewichtete_heuristik", gewichtete_heuristik),
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
