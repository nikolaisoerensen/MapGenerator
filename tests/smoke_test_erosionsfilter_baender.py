"""
Path: tests/smoke_test_erosionsfilter_baender.py

Prueft `_erosion_filter_bandweise()` in core/terrain_erosion_filter.py.

DIE KERNZUSICHERUNG ist die Bitgleichheit. Der Filter formt das Gelaende,
und die Regionseichung (smoke_test_regionen_welt) haengt an seiner Wirkung.
Waere die bandweise Auswertung auch nur im letzten Bit anders, waeren alle
neun Regionsziele neu zu eichen - ein Geschwindigkeitsgewinn, der eine
Eichrunde nach sich zieht, ist keiner.

Die Zerlegung ist zulaessig, WEIL erosion_filter() punktweise ist: das
Rauschen haengt allein an (px, py), die Neigungen kommen fertig herein.
Sollte das jemand aendern - etwa einen Nachbarschaftsfilter einbauen -,
faellt es hier auf, und zwar sofort und laut.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_erosionsfilter_baender.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.terrain_erosion_filter as ef

SEED = 20260804
# CLAUDE.md: mit den ECHTEN Kartengroessen pruefen, nicht mit ausgedachten.
GROESSEN = (256, 512, 1024)


_CACHE = {}


def _gelaende(n):
    """
    Kein glattes Kunstfeld - der Filter reagiert auf Grate und Rinnen.

    GEMERKT je Groesse. `weltfeld()` kostet bei 1024 px ueber eine halbe
    Minute, und drei Gruppen fragen dieselbe Groesse an; ohne den Cache lief
    dieser Test ueber zehn Minuten und damit praktisch nie (2026-08-23 selbst
    erlebt: der Lauf schlug scheinbar fehl, weil er in ein Aufrufzeitlimit
    lief, nicht weil etwas kaputt war).
    """
    if n not in _CACHE:
        from core.terrain_weltkarte import weltfeld
        H, _f = weltfeld(n, SEED)
        _CACHE[n] = np.asarray(H, dtype=np.float64)
    return _CACHE[n]


def _einteilig(px, py, h, sx, sy, ft, p, baender=None):
    return ef.erosion_filter(px, py, h, sx, sy, ft, p)


def bitgleich():
    """1. DIE KERNZUSICHERUNG: Band fuer Band dasselbe wie am Stueck."""
    fehler = []
    for n in GROESSEN:
        H = _gelaende(n)
        mpp = 21300.0 / n

        neu = ef.filter_heightmap(H, mpp)
        orig = ef._erosion_filter_bandweise
        ef._erosion_filter_bandweise = _einteilig
        try:
            alt = ef.filter_heightmap(H, mpp)
        finally:
            ef._erosion_filter_bandweise = orig

        schluessel = sorted(set(alt) | set(neu))
        abweichung = {}
        for k in schluessel:
            a, b = np.asarray(alt[k]), np.asarray(neu[k])
            if a.shape != b.shape:
                abweichung[k] = "Form"
            elif not np.array_equal(a, b):
                abweichung[k] = f"max {float(np.max(np.abs(a - b))):.3e}"

        ok = not abweichung
        print(f"[{'OK' if ok else 'FEHLER'}] {n} px: {len(schluessel)} "
              f"Ausgaben, bitgleich {ok}"
              + (f" - abweichend: {abweichung}" if abweichung else ""))
        if not ok:
            fehler.append(f"{n} px: {abweichung}")
    return fehler


def schneller():
    """2. Der Zweck - und um wieviel."""
    fehler = []
    n = 512
    H = _gelaende(n)
    mpp = 21300.0 / n

    t0 = time.perf_counter()
    ef.filter_heightmap(H, mpp)
    t_neu = time.perf_counter() - t0

    orig = ef._erosion_filter_bandweise
    ef._erosion_filter_bandweise = _einteilig
    try:
        t0 = time.perf_counter()
        ef.filter_heightmap(H, mpp)
        t_alt = time.perf_counter() - t0
    finally:
        ef._erosion_filter_bandweise = orig

    faktor = t_alt / max(t_neu, 1e-9)
    ok = faktor > 1.8
    print(f"[{'OK' if ok else 'FEHLER'}] {n} px: einteilig {t_alt:.2f}s, "
          f"bandweise {t_neu:.2f}s -> {faktor:.2f}x")
    if not ok:
        fehler.append(f"nur {faktor:.2f}x - unter der Erwartung von 1.8x")
    return fehler


def keine_nachwirkung():
    """
    3. Nach dem Filter darf numpy nicht langsamer sein als davor.

    DER GRUND, WARUM ES DIESE GRUPPE GIBT: die erste Fassung verteilte die
    Baender ueber 22 Threads. Der Filter wurde 5x schneller - und JEDE
    numpy-Rechnung danach im selben Prozess dauerhaft 2.4x langsamer, weil
    22 Threads gleichzeitig grosse Felder allokierten und den Allokator
    fragmentierten. Im Pipelinelauf stieg `weltfluesse` dadurch von 8.4 s
    auf 31.7 s; der Knoten wurde als Ganzes langsamer, obwohl sein
    teuerster Teilschritt schneller geworden war.

    Diese Gruppe faengt genau diesen Fall - er ist an der Zeit des Filters
    allein NICHT zu sehen.
    """
    fehler = []
    A = np.random.default_rng(SEED).random((1500, 1500))

    def referenz():
        t0 = time.perf_counter()
        for _ in range(4):
            np.fft.fft2(A)
        return time.perf_counter() - t0

    referenz()                                   # einschwingen
    vorher = min(referenz() for _ in range(2))
    H = _gelaende(512)
    for _ in range(3):
        ef.filter_heightmap(H, 41.6)
    nachher = min(referenz() for _ in range(2))

    verhaeltnis = nachher / max(vorher, 1e-9)
    ok = verhaeltnis < 1.4
    print(f"[{'OK' if ok else 'FEHLER'}] Referenzlast vorher {vorher:.2f}s, "
          f"nach 3 Filterlaeufen {nachher:.2f}s -> {verhaeltnis:.2f}x")
    if not ok:
        fehler.append(f"numpy nach dem Filter {verhaeltnis:.2f}x langsamer - "
                      f"der Filter schadet allem, was danach kommt")
    return fehler


def bandzahl_egal():
    """
    3. Das Ergebnis darf nicht von der Bandzahl abhaengen.

    Waere es das, haette die Zerlegung eine Nahtstelle - und Naehte in der
    Gelaendeform sind in diesem Projekt schon mehrfach als sichtbare
    Streifen aufgefallen.
    """
    fehler = []
    n = 256
    H = _gelaende(n)
    achse = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    px, py = np.meshgrid(achse, achse, indexing="xy")
    tief, hoch = float(H.min()), float(H.max())
    normiert = (H - tief) / max(hoch - tief, 1e-9)
    dz, ds = np.gradient(normiert, 1.0 / float(n))
    ft = np.clip((normiert - float(np.median(normiert)))
                 / (2.0 * max(float(np.std(normiert)), 1e-6)), -1.0, 1.0)
    p = dict(ef.ATEF_DEFAULTS)

    grund = ef._erosion_filter_bandweise(px, py, normiert, ds, dz, ft, p,
                                         baender=1)
    for b in (2, 3, 7, 16, 64):
        andere = ef._erosion_filter_bandweise(px, py, normiert, ds, dz, ft, p,
                                              baender=b)
        gleich = all(np.array_equal(np.asarray(grund[k]), np.asarray(andere[k]))
                     for k in grund)
        print(f"[{'OK' if gleich else 'FEHLER'}] {b} Baender: identisch zu "
              f"einem Band: {gleich}")
        if not gleich:
            fehler.append(f"{b} Baender weichen ab")
    return fehler


def randfaelle():
    """4. Groessen, die nicht glatt durch die Bandzahl teilbar sind."""
    fehler = []
    for n in (65, 100, 257):
        try:
            H = np.zeros((n, n), dtype=np.float64)
            H[n // 3:, :] = 500.0
            r = ef.filter_heightmap(H, 21300.0 / n)
            form_ok = all(np.asarray(v).shape in ((n, n), ())
                          for v in r.values())
            print(f"[{'OK' if form_ok else 'FEHLER'}] {n} px "
                  f"(nicht durch die Bandzahl teilbar): Formen stimmen "
                  f"{form_ok}")
            if not form_ok:
                fehler.append(f"{n} px: Ausgabeform falsch")
        except Exception as e:                                # noqa: BLE001
            print(f"[FEHLER] {n} px: Absturz - {e}")
            fehler.append(f"{n} px: Absturz - {e}")
    return fehler


def lauf():
    gruppen = [
        ("bitgleich", bitgleich),
        ("schneller", schneller),
        ("keine_nachwirkung", keine_nachwirkung),
        ("bandzahl_egal", bandzahl_egal),
        ("randfaelle", randfaelle),
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
