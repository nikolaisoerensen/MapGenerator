"""
Path: tests/smoke_test_poisson_punkte.py

Prueft `poisson_points()` in core/terrain_river_network.py nach dem
Umbau vom 2026-08-23 (numpy-Skalarzugriffe raus, Python-Listen rein).

DIE KERNZUSICHERUNG ist der IDENTISCHE PUNKTSATZ. Der Punktsatz ist die
Wurzel des ganzen Flussnetzes: aus ihm entstehen Delaunay-Graph,
Spannbaum und schliesslich die eingegrabenen Taeler. Ein einziger
verschobener Punkt aendert das Gelaende - und damit Hydrologie, Biome und
Siedlungsplaetze.

Zwei Dinge koennen ihn verschieben, und beide sind hier abgedeckt:

1. **Die Zufallsreihenfolge.** Der Generator wird in einer festen Folge
   abgefragt (`random(2)`, dann je Runde `integers` und je Versuch zweimal
   `random`). Jede Zusammenfassung - etwa "alle Winkel auf einmal ziehen" -
   verschiebt den Strom und liefert einen anderen Punktsatz. Genau deshalb
   wurde beim Umbau NUR das geaendert, was zwischen den Zufallsaufrufen
   passiert.

2. **Die Arithmetik.** Skalarrechnung in Python statt in numpy ist
   bitgleich, solange dieselben Operationen in derselben Reihenfolge
   laufen - beides ist IEEE-double. Die Referenzfassung unten rechnet
   noch mit numpy-Arrays und deckt genau diesen Unterschied ab.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_poisson_punkte.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.terrain_river_network import poisson_points

# Die ECHTEN Stufenabstaende des Flussnetzes (STUFEN in
# core/terrain_weltfluesse.py) auf der echten Weltbreite von 21.3 km.
# CLAUDE.md: mit den wirklichen Eingabegroessen pruefen.
WELT_M = 21300.0
ABSTAENDE = (1200.0, 420.0, 150.0)
SEED = 20260804


def _referenz(extent_m, min_distance_m, seed, attempts=30):
    """
    Die Fassung vor dem Umbau, Zeile fuer Zeile: numpy-Gitter,
    numpy-Punkte, ein 2-Element-Array je Versuch.
    """
    rng = np.random.default_rng(seed)
    cell = min_distance_m / np.sqrt(2.0)
    n = int(np.ceil(extent_m / cell)) + 1
    grid = -np.ones((n, n), dtype=np.int64)
    points, active = [], []

    def insert(p):
        points.append(p)
        grid[int(p[0] / cell), int(p[1] / cell)] = len(points) - 1
        active.append(len(points) - 1)

    insert(rng.random(2) * extent_m)

    while active:
        k = int(rng.integers(0, len(active)))
        centre = points[active[k]]
        found = False
        for _ in range(attempts):
            angle = rng.random() * 2.0 * np.pi
            radius = min_distance_m * (1.0 + rng.random())
            p = centre + radius * np.array([np.cos(angle), np.sin(angle)])
            if not (0.0 <= p[0] < extent_m and 0.0 <= p[1] < extent_m):
                continue
            gy, gx = int(p[0] / cell), int(p[1] / cell)
            free = True
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    yy, xx = gy + dy, gx + dx
                    if 0 <= yy < n and 0 <= xx < n and grid[yy, xx] >= 0:
                        q = points[grid[yy, xx]]
                        if ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2
                                < min_distance_m ** 2):
                            free = False
                            break
                if not free:
                    break
            if free:
                insert(p)
                found = True
                break
        if not found:
            active.pop(k)
    return np.array(points)


def bitgleicher_punktsatz():
    """1. DIE KERNZUSICHERUNG."""
    fehler = []
    for d in ABSTAENDE:
        alt = _referenz(WELT_M, d, SEED)
        neu = poisson_points(WELT_M, d, SEED)
        form_ok = alt.shape == neu.shape
        gleich = form_ok and np.array_equal(alt, neu)
        print(f"[{'OK' if gleich else 'FEHLER'}] Abstand {d:>6.0f} m: "
              f"{len(alt)} gegen {len(neu)} Punkte, bitgleich {gleich}")
        if not gleich:
            if not form_ok:
                fehler.append(f"{d} m: {len(alt)} statt {len(neu)} Punkte")
            else:
                d_max = float(np.max(np.abs(alt - neu)))
                anz = int((np.abs(alt - neu) > 0).any(axis=1).sum())
                fehler.append(f"{d} m: {anz} Punkte verschoben, "
                              f"max {d_max:.3e} m")
    return fehler


def andere_seeds():
    """
    2. Nicht nur der eine Seed.

    Ein Umbau, der zufaellig bei einem Seed passt und bei anderen nicht,
    waere schlimmer als einer, der immer danebenliegt - er faellt erst
    auf, wenn jemand den Seed wechselt.
    """
    fehler = []
    for seed in (1, 42, 99, 20260804, 987654321):
        alt = _referenz(WELT_M, 420.0, seed)
        neu = poisson_points(WELT_M, 420.0, seed)
        gleich = alt.shape == neu.shape and np.array_equal(alt, neu)
        print(f"[{'OK' if gleich else 'FEHLER'}] Seed {seed:>10}: "
              f"{len(alt)} Punkte, bitgleich {gleich}")
        if not gleich:
            fehler.append(f"Seed {seed}: weicht ab")
    return fehler


def mindestabstand_eingehalten():
    """
    3. Die eigentliche Zusicherung der Funktion.

    Bitgleichheit sagt nur, dass sich nichts geaendert hat - nicht, dass
    das Ergebnis richtig ist. Ein Poisson-Disk-Satz muss den
    Mindestabstand ueberall einhalten.
    """
    fehler = []
    from scipy.spatial import cKDTree
    for d in ABSTAENDE:
        P = poisson_points(WELT_M, d, SEED)
        baum = cKDTree(P)
        abst, _i = baum.query(P, k=2)
        kleinster = float(abst[:, 1].min())
        ok = kleinster >= d - 1e-9
        print(f"[{'OK' if ok else 'FEHLER'}] Abstand {d:>6.0f} m: "
              f"kleinster Punktabstand {kleinster:.3f} m")
        if not ok:
            fehler.append(f"{d} m: kleinster Abstand nur {kleinster:.3f} m")
    return fehler


def schneller():
    """4. Der Zweck - auf der teuersten Stufe."""
    fehler = []
    d = 150.0                                    # Mikrostufe, 12708 Punkte

    t0 = time.perf_counter()
    neu = poisson_points(WELT_M, d, SEED)
    t_neu = time.perf_counter() - t0

    t0 = time.perf_counter()
    _referenz(WELT_M, d, SEED)
    t_alt = time.perf_counter() - t0

    faktor = t_alt / max(t_neu, 1e-9)
    ok = faktor > 1.8
    print(f"[{'OK' if ok else 'FEHLER'}] Mikrostufe ({len(neu)} Punkte): "
          f"vorher {t_alt:.2f}s, jetzt {t_neu:.2f}s -> {faktor:.2f}x")
    if not ok:
        fehler.append(f"nur {faktor:.2f}x - unter der Erwartung von 1.8x")
    return fehler


def randfaelle():
    """5. Sehr grosse Abstaende und winzige Ausschnitte."""
    fehler = []
    faelle = [
        ("Abstand groesser als die Karte", 1000.0, 5000.0),
        ("genau ein Punkt passt", 1000.0, 900.0),
        ("winziger Ausschnitt", 100.0, 30.0),
    ]
    for name, ext, d in faelle:
        try:
            P = poisson_points(ext, d, SEED)
            alt = _referenz(ext, d, SEED)
            gleich = P.shape == alt.shape and np.array_equal(P, alt)
            print(f"[{'OK' if gleich else 'FEHLER'}] {name}: {len(P)} "
                  f"Punkte, bitgleich {gleich}")
            if not gleich:
                fehler.append(f"{name}: weicht ab")
        except Exception as e:                                # noqa: BLE001
            print(f"[FEHLER] {name}: Absturz - {e}")
            fehler.append(f"{name}: Absturz - {e}")
    return fehler


def lauf():
    gruppen = [
        ("bitgleicher_punktsatz", bitgleicher_punktsatz),
        ("andere_seeds", andere_seeds),
        ("mindestabstand_eingehalten", mindestabstand_eingehalten),
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
