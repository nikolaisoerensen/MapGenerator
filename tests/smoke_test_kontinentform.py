"""
Path: tests/smoke_test_kontinentform.py

Der Formregler des Kontinents (docs/AUFRAEUMPLAN.md 4.10).

NUTZERENTWURF 2026-08-26: *"zB laesst sich hier ein eher runder kontinent
erstellen oder aber einer mit vielen armen (also die grundformen als slider,
links rund, mitte laenglich, rechts mit vielen auslaeufern)."*

WAS GEPRUEFT WIRD, und warum gerade das:

  1. **Ohne Regler aendert sich NICHTS.** `form=None` muss den Kontinent
     erzeugen, den es bis zum 2026-08-26 gab. Daran haengt jede Eichung -
     Regionsflaechen, Wasseranteile, `smoke_test_regionen_welt`. Ein
     Formregler, der die Vorgabe verschiebt, entwertet sie alle stillschweigend.
  2. **Die Landflaeche bleibt gleich.** `kontinentform()` sucht ihren
     Schwellwert per Intervallhalbierung auf `KONTINENT_KM^2`. Der Regler
     darf die Gestalt aendern, nicht die Groesse.
  3. **Der Kontinent bleibt EIN Stueck.** Bei "viele Auslaeufer" liegt der
     Verdacht nahe, dass er in Inseln zerfaellt.
  4. **Die Enden tun, was sie versprechen** - gemessen, nicht behauptet:
     links am rundesten, Mitte am laengsten, rechts am zerlapptesten.

EIN ERSTER ENTWURF WAR GENAU DARIN FALSCH. Er interpolierte Kern, Lappen
und Vereinigungshaerte, liess die abgezogenen BUCHTEN aber fest. Gemessen
war die Stellung "rund" damit die UNRUNDESTE von allen (Rundheit 0.53
gegen 0.92 heute): die festen Buchten schnitten tief in eine Landmasse, die
durch die nahen Lappen ohnehin kompakter geworden war. Nur eine Messung der
Rundheit zeigte das - im Bild sah es lange nach "irgendwie unruhig" aus.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_kontinentform.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from scipy import ndimage

import core.terrain_weltkarte as rw

SIZE = 256
SEEDS = (20260804, 20261817, 20262830)
# Wieviel die Landflaeche zwischen den Reglerstellungen schwanken darf.
# Gemessen 37.9 bis 38.3 % - die Intervallhalbierung trifft ihr Ziel nicht
# exakt, weil sie auf einem Pixelraster arbeitet.
FLAECHE_MAX_ABWEICHUNG = 0.015


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def _kennzahlen(maske):
    """Rundheit (isoperimetrisch), Seitenverhaeltnis, Teile, Landanteil."""
    umfang = int((maske & ~ndimage.binary_erosion(maske)).sum())
    flaeche = int(maske.sum())
    ys, xs = np.where(maske)
    breite = xs.max() - xs.min() + 1
    hoehe = ys.max() - ys.min() + 1
    _lab, teile = ndimage.label(maske)
    return (4.0 * np.pi * flaeche / max(umfang * umfang, 1),
            breite / max(hoehe, 1), teile, float(maske.mean()))


def lauf():
    fehler = []

    # --- 1. Ohne Regler aendert sich nichts ---------------------------------
    gleich = all(
        np.array_equal(rw.kontinentform(SIZE, s)[0],
                       rw.kontinentform(SIZE, s, form=None)[0])
        for s in SEEDS)
    fehler += check("ohne Regler bleibt die Vorgabe unveraendert", gleich,
                    "sonst waeren alle Eichungen still verschoben")

    # --- 2-4. Die Reglerstellungen vermessen --------------------------------
    stellungen = (0.0, 0.25, 0.5, 0.75, 1.0)
    werte = {}
    print()
    print(f"       {'form':>5}{'Land':>8}{'Rundheit':>10}{'Seitenverh.':>13}"
          f"{'Teile':>7}")
    for f in stellungen:
        k = np.array([_kennzahlen(rw.kontinentform(SIZE, s, form=f)[0])
                      for s in SEEDS])
        werte[f] = k.mean(axis=0)
        print(f"       {f:>5.2f}{100 * werte[f][3]:>7.1f}%{werte[f][0]:>10.3f}"
              f"{werte[f][1]:>13.2f}{werte[f][2]:>7.1f}")
    print()

    anteile = [werte[f][3] for f in stellungen]
    fehler += check("die Landflaeche bleibt gleich",
                    max(anteile) - min(anteile) < FLAECHE_MAX_ABWEICHUNG,
                    f"{100 * min(anteile):.1f} bis {100 * max(anteile):.1f} %")

    fehler += check("der Kontinent bleibt EIN Stueck",
                    all(werte[f][2] < 1.5 for f in stellungen),
                    ", ".join(f"{f:.2f}: {werte[f][2]:.1f}"
                              for f in stellungen if werte[f][2] >= 1.5))

    rundeste = max(stellungen, key=lambda f: werte[f][0])
    fehler += check("links ist am rundesten", rundeste == 0.0,
                    f"rundeste Stellung ist {rundeste:.2f} "
                    f"({werte[rundeste][0]:.3f}), links {werte[0.0][0]:.3f}")

    laengste = max(stellungen, key=lambda f: werte[f][1])
    fehler += check("die Mitte ist am laengsten", laengste == 0.5,
                    f"laengste Stellung ist {laengste:.2f} "
                    f"({werte[laengste][1]:.2f})")

    zerlapptste = min(stellungen, key=lambda f: werte[f][0])
    fehler += check("rechts ist am zerlapptesten", zerlapptste == 1.0,
                    f"zerlapptste Stellung ist {zerlapptste:.2f} "
                    f"({werte[zerlapptste][0]:.3f})")

    # --- 5. Der Regler kommt durch weltfeld() an ----------------------------
    a, _f = rw.weltfeld(192, SEEDS[0])
    b, _f = rw.weltfeld(192, SEEDS[0], kontinentform_regler=None)
    c, _f = rw.weltfeld(192, SEEDS[0], kontinentform_regler=1.0)
    fehler += check("weltfeld ohne Regler unveraendert",
                    np.array_equal(np.asarray(a), np.asarray(b)))
    fehler += check("weltfeld reicht den Regler durch",
                    not np.array_equal(np.asarray(a), np.asarray(c)))

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("der Formregler tut, was er verspricht")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
