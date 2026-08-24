"""
Path: tests/smoke_test_spielkarten.py

Prueft core/spielkarten.py gegen die Vorgaben des Nutzers vom 2026-08-13
(docs/OFFENE_PUNKTE.md 5.15):

  1. "jeweils gleich viel Landmasse (der erste seegrad an kueste zaehlt auch
     als 50% landmasse)"  -> Massenspanne <= 1.5
  2. dasselbe fuer das REINE Land - sonst kann die halb zaehlende Kuestensee
     eine Karte auffuellen, die kaum Land hat (genau das ist beim Bauen
     passiert: Massenspanne 1.35 bei Landspanne 2.09)
  3. "optimiert um auf einem quadrat gezeigt werden zu koennen, also nicht
     unnoetig lang" -> Seitenverhaeltnis der Bounding-Box begrenzt
  4. "die karten koennen bis zu 50% unterschiedlich gross sein" -> genau die
     1.5 aus (1)/(2), nicht mehr
  5. Determinismus: gleicher Seed -> gleiche Zerlegung

Laeuft mit ECHTEN Kartengroessen (256/512), nicht mit ausgedachten - siehe
CLAUDE.md zur Lehre aus 6.16.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from core.terrain_weltkarte import weltfeld
from core import spielkarten as sk


# Ueber alle geprueften Seeds gemessen liegt das schlechteste
# Seitenverhaeltnis bei 2.11. Die Schranke liegt bewusst leicht darueber:
# sie soll eine ECHTE Verschlechterung fangen (langgezogene Baender), nicht
# bei jeder harmlosen Schwankung der Weltform anschlagen.
MAX_SEITENVERHAELTNIS = 2.4

_SEEDS = (20260804, 12345, 4242, 777001)


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def run_vorgaben():
    ok = True
    print("Zerlegung gegen die Nutzer-Vorgaben (512 px):\n")
    print(f"{'Seed':>9} {'Masse-Sp':>9} {'Land-Sp':>8} {'Seitv.max':>10} "
          f"{'Kern min':>9} {'Runden':>7} {'Zeit':>7}")
    for seed in _SEEDS:
        H, felder = weltfeld(512, seed)
        t0 = time.time()
        erg = sk.zerlegen(H, felder.get("seegrad"), anzahl=9, seed=seed)
        dauer = time.time() - t0
        kz = sk.kennzahlen(erg, region_map=felder.get("regionen"))
        seitv = float(np.nanmax(kz["seitenverhaeltnis"]))
        print(f"{seed:>9} {kz['masse_spanne']:9.2f} {kz['land_spanne']:8.2f} "
              f"{seitv:10.2f} {kz['kern_anteil'].min():9.2f} "
              f"{erg['runden']:7d} {dauer:6.1f}s")

        ok &= check(f"  Seed {seed}: Massenspanne <= {sk.SPANNE_ZIEL}",
                    kz["masse_spanne"] <= sk.SPANNE_ZIEL,
                    f"{kz['masse_spanne']:.2f}")
        ok &= check(f"  Seed {seed}: Landspanne <= {sk.SPANNE_ZIEL}",
                    kz["land_spanne"] <= sk.SPANNE_ZIEL,
                    f"{kz['land_spanne']:.2f}")
        ok &= check(f"  Seed {seed}: kein Seitenverhaeltnis ueber {MAX_SEITENVERHAELTNIS}",
                    seitv <= MAX_SEITENVERHAELTNIS, f"{seitv:.2f}")
        ok &= check(f"  Seed {seed}: jede Karte hat Land",
                    kz["land_je_karte"].min() > 0,
                    f"kleinste {int(kz['land_je_karte'].min())} px")
    return ok


def run_kuestensee_zaehlt_halb():
    """Die Vorgabe 'erster Seegrad zaehlt als 50 % Landmasse' muss im
    Gewichtsfeld tatsaechlich ankommen - und NUR dort, nicht auf offener See."""
    H, felder = weltfeld(256, 20260804)
    grad = np.asarray(felder["seegrad"])
    w = sk.gewichtsfeld(H, grad)
    land = H > 0
    kueste = (H <= 0) & (grad == 1)
    tief = (H <= 0) & (grad > 1)

    ok = True
    ok &= check("Land traegt Gewicht 1.0", np.allclose(w[land], 1.0))
    ok &= check(f"Kuestensee (Seegrad 1) traegt {sk.KUESTENSEE_GEWICHT}",
                np.allclose(w[kueste], sk.KUESTENSEE_GEWICHT) if kueste.any() else True,
                f"{kueste.sum()} Pixel")
    ok &= check("offene See traegt 0", np.allclose(w[tief], 0.0) if tief.any() else True,
                f"{tief.sum()} Pixel")

    # Ohne seegrad darf sich nichts anderes ergeben als reines Land
    w2 = sk.gewichtsfeld(H, None)
    ok &= check("ohne Seegrad zaehlt nur Land",
                np.allclose(w2, land.astype(np.float32)))
    return ok


def run_determinismus():
    H, felder = weltfeld(256, 4242)
    a = sk.zerlegen(H, felder.get("seegrad"), anzahl=9, seed=4242)
    b = sk.zerlegen(H, felder.get("seegrad"), anzahl=9, seed=4242)
    return check("gleicher Seed -> bitgleiche Zerlegung",
                 np.array_equal(a["karte"], b["karte"])
                 and np.allclose(a["saat"], b["saat"]))


def run_saatpunkte_aus_siedlungen():
    """Mit Siedlungen muessen die Saatpunkte AUS den Siedlungen kommen (sonst
    verlaufen die Grenzen nicht zwischen den Staedten) - und der Rueckfall
    muss sich melden, statt still zu greifen."""
    class Ort:
        def __init__(self, x, y, h=20):
            self.x, self.y, self.house_count = x, y, h

    H, felder = weltfeld(256, 12345)
    w = sk.gewichtsfeld(H, felder.get("seegrad"))
    land = np.argwhere(H > 0)
    rng = np.random.RandomState(1)
    auswahl = land[rng.choice(len(land), 40, replace=False)]
    orte = [Ort(float(x), float(y)) for y, x in auswahl]

    saat, aus_siedlungen = sk.saatpunkte(w, 9, 12345, orte)
    ok = check("mit 40 Siedlungen: Saatpunkte kommen aus den Siedlungen",
               aus_siedlungen)
    ok &= check("  9 Saatpunkte geliefert", len(saat) == 9, f"{len(saat)}")

    saat2, aus2 = sk.saatpunkte(w, 9, 12345, orte[:3])
    ok &= check("mit nur 3 Siedlungen: Rueckfall MELDET sich", not aus2)

    saat3, aus3 = sk.saatpunkte(w, 9, 12345, None)
    ok &= check("ohne Siedlungen: Rueckfall meldet sich", not aus3)
    # Rueckfall-Saatpunkte muessen auf LAND liegen (sonst bekommt eine Karte
    # fast nur Kuestensee - gemessener Fehler beim Bauen)
    auf_land = [w[int(round(y)), int(round(x))] >= 1.0 for x, y in saat3]
    ok &= check("  Rueckfall-Saatpunkte liegen alle auf Land",
                all(auf_land), f"{sum(auf_land)}/9")
    return ok


if __name__ == "__main__":
    ergebnisse = {
        "kuestensee_zaehlt_halb": run_kuestensee_zaehlt_halb(),
        "saatpunkte_aus_siedlungen": run_saatpunkte_aus_siedlungen(),
        "determinismus": run_determinismus(),
        "vorgaben": run_vorgaben(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
