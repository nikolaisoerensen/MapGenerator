"""
Path: tests/smoke_test_weltfluesse_vektor.py

Prueft die vektorisierte Stuetzstellen-Schleife in `taeler_eingraben()`
(core/terrain_weltfluesse.py).

WORUM ES GEHT. Die Schleife legte je Kante `schritte` Stuetzstellen und
schrieb an jeder in zwei Felder:

    sohle[y, x]  = wert          # ZUWEISUNG - der letzte gewinnt
    breite[y, x] = max(..., w)   # MAXIMUM   - reihenfolgeunabhaengig

Mehrere Stuetzstellen treffen regelmaessig dasselbe Pixel (bei 3 Stuetz-
stellen je Pixel Streckenlaenge ist das der Normalfall, nicht die
Ausnahme). Die vektorisierte Fassung muss deshalb GENAU dieselbe
Aufloesung solcher Kollisionen liefern:

  * `sohle[ys, xs] = werte` - numpy schreibt die Eintraege der Reihe nach,
    bei doppelten Indizes bleibt der LETZTE stehen. Das ist dasselbe wie
    die Schleife.
  * `np.maximum.at(breite, (ys, xs), w)` - streuende Maximumbildung,
    unabhaengig von der Reihenfolge.

Waere das falsch, laegen Flusssohlen auf anderer Hoehe - und weil
`taeler_eingraben` in die Heightmap schneidet, waere das eine
Gelaendeaenderung, keine Beschleunigung.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_weltfluesse_vektor.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 20260804


def _schleife(pk, z, kanten, size, breite_feld, tiefe_feld, gebiet, anteile):
    """Die urspruengliche Fassung, Stuetzstelle fuer Stuetzstelle."""
    sohle = np.full((size, size), np.nan)
    breite = np.zeros((size, size))
    for nr, (i, e) in enumerate(kanten):
        strecke = float(np.linalg.norm(pk[i] - pk[e]))
        schritte = max(int(strecke * 3.0), 3)
        anteil = anteile[nr]
        for t in np.linspace(0.0, 1.0, schritte):
            p = pk[e] * (1.0 - t) + pk[i] * t
            y = int(np.clip(round(p[0]), 0, size - 1))
            x = int(np.clip(round(p[1]), 0, size - 1))
            w = max(anteil * breite_feld[y, x], 2.5)
            sohle_roh = z[e] * (1.0 - t) + z[i] * t
            tief = min(tiefe_feld[y, x] * gebiet[i] ** 0.30,
                       0.55 * max(sohle_roh, 0.0))
            sohle[y, x] = sohle_roh - tief
            breite[y, x] = max(breite[y, x], w)
    return sohle, breite


def _vektor(pk, z, kanten, size, breite_feld, tiefe_feld, gebiet, anteile):
    """Die Fassung, wie sie in taeler_eingraben() steht."""
    sohle = np.full((size, size), np.nan)
    breite = np.zeros((size, size))
    for nr, (i, e) in enumerate(kanten):
        strecke = float(np.linalg.norm(pk[i] - pk[e]))
        schritte = max(int(strecke * 3.0), 3)
        anteil = anteile[nr]
        t = np.linspace(0.0, 1.0, schritte)
        ps = pk[e][None, :] * (1.0 - t)[:, None] + pk[i][None, :] * t[:, None]
        ys = np.clip(np.round(ps[:, 0]), 0, size - 1).astype(np.int64)
        xs = np.clip(np.round(ps[:, 1]), 0, size - 1).astype(np.int64)
        w = np.maximum(anteil * breite_feld[ys, xs], 2.5)
        sohle_roh = z[e] * (1.0 - t) + z[i] * t
        tief = np.minimum(tiefe_feld[ys, xs] * gebiet[i] ** 0.30,
                          0.55 * np.maximum(sohle_roh, 0.0))
        sohle[ys, xs] = sohle_roh - tief
        np.maximum.at(breite, (ys, xs), w)
    return sohle, breite


def _fall(rng, size, n_punkte, kanten_zahl):
    pk = rng.random((n_punkte, 2)) * (size - 1)
    z = rng.random(n_punkte) * 900.0
    kanten = [(int(a), int(b)) for a, b in
              rng.integers(0, n_punkte, size=(kanten_zahl, 2)) if a != b]
    breite_feld = 1.0 + 8.0 * rng.random((size, size))
    tiefe_feld = 10.0 + 90.0 * rng.random((size, size))
    gebiet = 1.0 + 50.0 * rng.random(n_punkte)
    anteile = 0.22 + 0.78 * rng.random(len(kanten))
    return pk, z, kanten, breite_feld, tiefe_feld, gebiet, anteile


def gleiches_ergebnis():
    """1. DIE KERNZUSICHERUNG: bitgleich, Kollisionen eingeschlossen."""
    fehler = []
    rng = np.random.default_rng(SEED)
    faelle = [
        ("wenige lange Kanten", 128, 40, 60),
        ("viele kurze Kanten", 256, 400, 900),
        # Punkte dicht beieinander -> MAXIMAL viele Pixelkollisionen, also
        # genau der Fall, in dem die Schreibreihenfolge entscheidet.
        ("Punkte dicht gedraengt", 64, 200, 500),
    ]
    for name, size, n_p, n_k in faelle:
        args = _fall(rng, size, n_p, n_k)
        s1, b1 = _schleife(args[0], args[1], args[2], size, *args[3:])
        s2, b2 = _vektor(args[0], args[1], args[2], size, *args[3:])

        s_gleich = np.array_equal(s1, s2, equal_nan=True)
        b_gleich = np.array_equal(b1, b2)
        belegt = int(np.isfinite(s1).sum())
        ok = s_gleich and b_gleich
        print(f"[{'OK' if ok else 'FEHLER'}] {name}: {belegt} Flusspixel, "
              f"sohle bitgleich {s_gleich}, breite bitgleich {b_gleich}")
        if not ok:
            if not s_gleich:
                d = np.abs(np.nan_to_num(s1) - np.nan_to_num(s2))
                fehler.append(f"{name}: sohle weicht ab, max {d.max():.3e}, "
                              f"{int((d > 0).sum())} Pixel")
            if not b_gleich:
                d = np.abs(b1 - b2)
                fehler.append(f"{name}: breite weicht ab, max {d.max():.3e}")
    return fehler


def kollisionen_kommen_wirklich_vor():
    """
    2. Gegenprobe: der gepruefte Fall muss auch auftreten.

    Ohne diese Gruppe koennte Gruppe 1 gruen sein, weil gar keine zwei
    Stuetzstellen dasselbe Pixel treffen - dann wuerde sie die
    Schreibreihenfolge ueberhaupt nicht pruefen. Genau diese Art von
    stillem Nichts-Pruefen ist in diesem Projekt schon vorgekommen
    (CLAUDE.md, adaptives Mesh).
    """
    fehler = []
    rng = np.random.default_rng(SEED)
    args = _fall(rng, 64, 200, 500)
    pk, z, kanten = args[0], args[1], args[2]

    treffer = {}
    for i, e in kanten:
        strecke = float(np.linalg.norm(pk[i] - pk[e]))
        schritte = max(int(strecke * 3.0), 3)
        for t in np.linspace(0.0, 1.0, schritte):
            p = pk[e] * (1.0 - t) + pk[i] * t
            y = int(np.clip(round(p[0]), 0, 63))
            x = int(np.clip(round(p[1]), 0, 63))
            treffer[(y, x)] = treffer.get((y, x), 0) + 1

    mehrfach = sum(1 for v in treffer.values() if v > 1)
    hoechste = max(treffer.values()) if treffer else 0
    ok = mehrfach > 50
    print(f"[{'OK' if ok else 'FEHLER'}] {mehrfach} von {len(treffer)} Pixeln "
          f"mehrfach getroffen, hoechstens {hoechste}x - die Kollisionen "
          f"aus Gruppe 1 sind real")
    if not ok:
        fehler.append(f"nur {mehrfach} Kollisionen - Gruppe 1 prueft die "
                      f"Schreibreihenfolge dann gar nicht")
    return fehler


def schneller():
    """3. Der Zweck - gemessen auf einer echten Karte."""
    fehler = []
    from core.terrain_weltkarte import weltfeld
    from core.terrain_weltfluesse import flussnetz, taeler_eingraben

    H, felder = weltfeld(512, SEED)
    H = np.asarray(H, dtype=np.float64)
    netz = flussnetz(H, SEED)

    t0 = time.perf_counter()
    taeler_eingraben(H, netz, felder)
    t = time.perf_counter() - t0
    ok = t < 4.0
    print(f"[{'OK' if ok else 'FEHLER'}] taeler_eingraben 512 px: {t:.2f}s "
          f"(vor der Vektorisierung 12.6s bei 1024 px)")
    if not ok:
        fehler.append(f"{t:.2f}s - langsamer als erwartet")
    return fehler


def randfaelle():
    """4. Entartete Kanten."""
    fehler = []
    size = 32
    rng = np.random.default_rng(SEED)
    breite_feld = np.ones((size, size))
    tiefe_feld = np.full((size, size), 50.0)

    faelle = [
        ("Kante der Laenge 0", np.array([[5.0, 5.0], [5.0, 5.0]])),
        ("Kante ueber den Rand", np.array([[-40.0, -40.0], [80.0, 80.0]])),
        ("Kante genau auf dem Rand", np.array([[0.0, 0.0], [31.0, 31.0]])),
    ]
    for name, pk in faelle:
        try:
            z = np.array([100.0, 200.0])
            gebiet = np.array([5.0, 5.0])
            s1, b1 = _schleife(pk, z, [(0, 1)], size, breite_feld,
                               tiefe_feld, gebiet, [0.5])
            s2, b2 = _vektor(pk, z, [(0, 1)], size, breite_feld,
                             tiefe_feld, gebiet, [0.5])
            ok = (np.array_equal(s1, s2, equal_nan=True)
                  and np.array_equal(b1, b2))
            print(f"[{'OK' if ok else 'FEHLER'}] {name}: identisch {ok}")
            if not ok:
                fehler.append(f"{name}: weicht ab")
        except Exception as e:                                # noqa: BLE001
            print(f"[FEHLER] {name}: Absturz - {e}")
            fehler.append(f"{name}: Absturz - {e}")
    return fehler


def lauf():
    gruppen = [
        ("gleiches_ergebnis", gleiches_ergebnis),
        ("kollisionen_kommen_wirklich_vor", kollisionen_kommen_wirklich_vor),
        ("schneller", schneller),
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
