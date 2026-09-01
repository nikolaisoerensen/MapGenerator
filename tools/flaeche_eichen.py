"""
Path: tools/flaeche_eichen.py

`flaeche_soll` je Region so einregeln, dass jede Region ihren Zielwert
an BESIEDELBARER Flaeche erreicht.

NUTZERAUFTRAG 2026-08-24:

  *"lass uns zielwert 0.8 fuer fjordland festlegen, aber dann muessen wir
  noch etwas mehr flaeche bekommen. du kannst ja immer die flaechen
  vergroessert etc. fuer einzelne regionen. insgesamt bekommt halt jede
  region einen bestimmten zielwert der erreicht werden soll. also so und
  so viel region die man besiedeln kann."*

## Warum ein Werkzeug und keine Handarbeit

Die Groesse einer Region und ihr NUTZWERT haengen nicht linear zusammen.
Skerrheim verliert doppelt - erst ein Drittel ans Wasser, dann die Haelfte
des Rests an zu steile Haenge. Mehr Grundflaeche bringt dort weniger als
anderswo, und wieviel weniger, haengt an der Form des Kontinents.

Dazu ist es ein NULLSUMMENSPIEL: der Index misst gegen den Median aller
Regionen. Wer waechst, druckt alle anderen nach unten. Von Hand einen
Wert zu setzen, zu messen und nachzubessern, kostet je Runde eine volle
Weltberechnung - und die Runde danach verschiebt, was die vorige
erreicht hatte.

Der Regelkreis hier macht das in einem Rutsch. Er ist LOGARITHMISCH, aus
demselben Grund wie die Flaecheneichung in `voronoi_regionen()`: ein
linearer Schritt schwingt, weil ein Vorteil multiplikativ auf das Gewicht
wirkt.

## Was er NICHT tut

Er aendert `ZIELWERT` nicht. Welche Region wieviel bekommen SOLL, ist
eine Entscheidung ueber das Zielbild und steht in
`tests/smoke_test_regionen_fairness.py`. Dieses Werkzeug stellt nur ein,
was noetig ist, um sie zu erreichen.

Aufruf:
    .venv/Scripts/python.exe tools/flaeche_eichen.py
    .venv/Scripts/python.exe tools/flaeche_eichen.py --runden 8 --seeds 3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import core.terrain_weltkarte as rw
import tests.smoke_test_regionen_fairness as F

# Wie stark je Runde nachgezogen wird. 1.0 waere der volle Ausgleich und
# schwingt - der Index ist gegen den MEDIAN normiert, also bewegt sich der
# Bezugspunkt mit, sobald eine Region waechst.
DAEMPFUNG = 0.55

# Grenzen fuer `flaeche_soll`. Nach unten, damit eine Region nicht
# verschwindet; nach oben, damit keine den halben Kontinent bekommt.
SOLL_MIN, SOLL_MAX = 0.60, 2.50


def _regionen_dicts():
    return {r["name"]: r for _z, _s, r in rw.alle_regionen()}


def _messen(seeds):
    """Index je Region, ueber mehrere Seeds gemittelt."""
    sammlung = {}
    for seed in seeds:
        for z in F.bewerten(F.messen(seed=seed)):
            sammlung.setdefault(z["name"], []).append(z)
    aus = {}
    for name, liste in sammlung.items():
        aus[name] = {
            "index": float(np.mean([z["index"] for z in liste])),
            "ziel": liste[0]["ziel"],
            "gesamt": float(np.mean([z["gesamt"] for z in liste])),
            "bewohnbar": float(np.mean([z["bewohnbar"] for z in liste])),
        }
    return aus


def eichen(runden, seeds):
    dicts = _regionen_dicts()
    verlauf = []
    for runde in range(runden):
        lage = _messen(seeds)
        fehler = np.array([abs(lage[n]["index"] - lage[n]["ziel"])
                           for n in lage])
        verlauf.append((runde, float(fehler.max()), float(fehler.mean())))
        print(f"\nRunde {runde}: groesste Abweichung {fehler.max():.3f}, "
              f"mittlere {fehler.mean():.3f}")
        print(f"   {'Region':22s}{'soll':>7}{'Index':>8}{'Ziel':>7}"
              f"{'bewohnbar':>11}")
        for name in sorted(lage, key=lambda n: lage[n]["index"]):
            z = lage[name]
            marke = "  <--" if abs(z["index"] - z["ziel"]) > F.TOLERANZ else ""
            print(f"   {name:22s}{dicts[name].get('flaeche_soll', 1.0):7.2f}"
                  f"{z['index']:8.2f}{z['ziel']:7.2f}"
                  f"{z['bewohnbar']:11.0f}{marke}")

        if fehler.max() <= 0.5 * F.TOLERANZ:
            print("\nAlle Regionen deutlich im Rahmen - fertig.")
            break
        if runde == runden - 1:
            break

        for name, z in lage.items():
            ist = max(z["index"], 1e-6)
            # Logarithmischer Schritt: ein Vorteil wirkt multiplikativ.
            faktor = (z["ziel"] / ist) ** DAEMPFUNG
            alt = dicts[name].get("flaeche_soll", 1.0)
            dicts[name]["flaeche_soll"] = float(
                np.clip(alt * faktor, SOLL_MIN, SOLL_MAX))

    print("\n" + "=" * 66)
    print("ERGEBNIS - diese Werte gehoeren nach core/terrain_weltkarte.py:")
    print("=" * 66)
    for _z, _s, r in rw.alle_regionen():
        print(f"   {r['name']:22s} flaeche_soll={r.get('flaeche_soll', 1.0):.2f}")
    print()
    for runde, gross, mittel in verlauf:
        print(f"   Runde {runde}: max {gross:.3f}, mittel {mittel:.3f}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runden", type=int, default=6)
    p.add_argument("--seeds", type=int, default=1,
                   help="wieviele Seeds je Messung gemittelt werden")
    args = p.parse_args()
    seeds = [F.SEED + 1000 * k for k in range(max(1, args.seeds))]
    print(f"Eichung ueber {len(seeds)} Seed(s), {args.runden} Runden.")
    print("Jede Runde ist eine volle Weltberechnung je Seed - das dauert.")
    return eichen(args.runden, seeds)


if __name__ == "__main__":
    sys.exit(main())
