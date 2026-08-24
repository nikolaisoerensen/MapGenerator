"""
Path: tests/smoke_test_regionen_fairness.py

Bekommt jede Region ungefaehr gleich viel WERT?

NUTZERAUFTRAG 2026-08-24:

  *"dann brauchen wir einen test den wir immer abrufen koennen mit dem wir
  bewerten wie gut die flaechen getroffen werden. also dass wir einmal
  vergleichen koennen ob jede region fair verteilt ist. dabei muessen
  flaeche allgemein, flaeche unbewohnbar (sehr steil / hoch / wasser) und
  kuestenlinie verglichen werden. irgendwie muss daraus ein gemeinsamer
  vergleichbarer wert gefunden werden. also bewohnbare flaeche geht zB
  100% ein, unbewohnbar geht mit 25% ein und dann wirkt sich viel
  kuestenlinie nochmal positiv aus (zB 130% bei viel Kuestenlinie).
  Alpenland kann einen etwas kleineren zielwert haben (80% im vergleich zu
  den anderen regionen, weil es einfach normal ist, dass es in den bergen
  weniger platz gibt)."*

## Warum es diesen Test braucht

`smoke_test_regionen_welt.py` prueft, ob jede Region ihre EIGENEN
Zielwerte trifft - Hangneigung, Wasseranteil, Hoehe. Das sagt nichts
darueber, ob die Regionen im VERGLEICH fair dastehen. Eine Region kann
alle ihre Zielwerte treffen und trotzdem doppelt so viel nutzbares Land
haben wie ihre Nachbarin.

## Der Wert

    roh   = bewohnbar_px * 1.00 + unbewohnbar_land_px * 0.25
                                + wasser_px * 0.25
    wert  = roh * kuestenbonus
    index = wert / Median(alle Regionen)

`kuestenbonus` laeuft von 1.00 (keine Kuestenlinie) bis
KUESTENBONUS_MAX (viel Kuestenlinie), linear in der Kuestendichte.

WASSER ZAEHLT MIT 25 %, nicht mit 0. Eine Fjordbucht ist kein verlorener
Raum - sie traegt Fischerei, Wege und Siedlungsplaetze am Ufer. Genau
deshalb steht der Kuestenbonus daneben: Wasser MIT viel Ufer ist mehr
wert als Wasser ohne.

## Was der Test NICHT tut

Er verteilt nichts um. Er misst und meldet. Ob eine Abweichung ein Fehler
ist oder gewollt, entscheidet der Nutzer - deshalb stehen die Zielwerte
oben als Tabelle und nicht im Code verstreut.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionen_fairness.py
"""

import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.terrain_weltkarte as rw

SIZE = 512
SEED = 20260804

# ---------------------------------------------------------------- Schwellen
#
# Es gibt im Projekt KEINE feste Bewohnbarkeitsgrenze - `settlement_generator`
# rechnet mit relativen Eignungen (Perzentile der jeweiligen Region). Fuer
# einen VERGLEICH zwischen Regionen braucht es aber einen gemeinsamen
# Massstab, sonst misst jede Region an sich selbst.
#
# Die Werte hier sind bewusst grob und stehen zur Diskussion:

# Hangneigung, ab der Dauersiedlung unrealistisch wird. In den Alpen wird
# bis rund 30 Grad gesiedelt, Ackerbau endet deutlich frueher; 25 Grad ist
# die Mitte davon.
UNBEWOHNBAR_HANG_GRAD = 25.0

# Hoehe, ab der es ungemuetlich wird. Bezogen auf DIESE Welt, nicht auf die
# Realitaet: das Alpenland reicht bis rund 1030 m, der Rest deutlich
# tiefer. 800 m trennt das Hochgebirge vom besiedelbaren Teil.
UNBEWOHNBAR_HOEHE_M = 800.0

# ---------------------------------------------------------------- Gewichte
BEWOHNBAR_GEWICHT = 1.00
UNBEWOHNBAR_GEWICHT = 0.25          # Nutzervorgabe: "unbewohnbar geht mit 25%"
WASSER_GEWICHT = 0.25               # Wasser ist kein verlorener Raum
KUESTENBONUS_MAX = 1.30             # Nutzervorgabe: "zB 130% bei viel Kueste"

# Ab welcher Kuestendichte (Kuestenpixel je Landpixel) der Bonus voll gilt.
# Gemessen liegt die dichteste Region (Griechische Inseln) bei rund 15 %.
KUESTENDICHTE_VOLL = 0.15

# ------------------------------------------------------------- Zielwerte
#
# 1.0 heisst "so viel Wert wie der Median aller Regionen". Nur das
# Alpenland weicht ab - Nutzervorgabe: *"Alpenland kann einen etwas
# kleineren zielwert haben (80%), weil es einfach normal ist, dass es in
# den bergen weniger platz gibt"*.
ZIELWERT = {
    "Alpenland": 0.80,
    # Nutzerentscheidung 2026-08-24, nachdem der Verteilungsfix echte
    # Fjordwaende auf die Karte gebracht hatte: *"lass uns zielwert 0.8
    # fuer fjordland festlegen, aber dann muessen wir noch etwas mehr
    # flaeche bekommen."*
    #
    # Fjordland verliert DOPPELT - erst rund ein Drittel ans Wasser, dann
    # die Haelfte des Rests an Haenge ueber der Bewohnbarkeitsgrenze. Es
    # ist damit strukturell dem Alpenland aehnlich und bekommt denselben
    # Zielwert. Der Ausgleich laeuft ueber `flaeche_soll` in
    # core/terrain_weltkarte.py - siehe tools/flaeche_eichen.py.
    "Fjordland": 0.80,
}
ZIELWERT_STANDARD = 1.00

# Wieviel eine Region vom Zielwert abweichen darf. 25 % ist grosszuegig -
# der Test soll grobe Schieflagen fangen, nicht Rauschen melden.
TOLERANZ = 0.25


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


def messen(size=SIZE, seed=SEED):
    """Je Region die Rohgroessen. Rueckgabe: Liste von dicts."""
    H, felder = rw.weltfeld(size, seed)
    H = np.asarray(H, dtype=np.float64)
    mpp = rw.WELT_KM * 1000.0 / size

    # NUR INNERHALB DER KONTINENTFORM. Die Voronoi-Regionen decken die ganze
    # Karte ab, auch das offene Meer weit draussen - das gehoert keiner
    # Region als "Flaeche" zu. Ohne diese Maske haetten Randregionen
    # zufaellig mehr "Wasser", je nachdem wie die Form liegt.
    maske, _sdf = rw.kontinentform(size, seed)
    reg = felder["regionen"]

    dy, dx = np.gradient(H, mpp)
    hang_grad = np.degrees(np.arctan(np.hypot(dx, dy)))
    kueste = ndimage.binary_dilation(H <= 0.0, iterations=1) & (H > 0.0)

    zeilen = []
    for i, (_z, _s, r) in enumerate(rw.alle_regionen()):
        m = (reg == i) & maske
        if m.sum() < 50:
            continue
        land = m & (H > 0.0)
        wasser = m & (H <= 0.0)
        zu_steil = land & (hang_grad > UNBEWOHNBAR_HANG_GRAD)
        zu_hoch = land & (H > UNBEWOHNBAR_HOEHE_M)
        unbewohnbar = zu_steil | zu_hoch
        bewohnbar = land & ~unbewohnbar
        kuestenpixel = m & kueste

        zeilen.append({
            "name": r["name"],
            "gesamt": int(m.sum()),
            "land": int(land.sum()),
            "wasser": int(wasser.sum()),
            "bewohnbar": int(bewohnbar.sum()),
            "unbewohnbar": int(unbewohnbar.sum()),
            "zu_steil": int(zu_steil.sum()),
            "zu_hoch": int(zu_hoch.sum()),
            "kueste": int(kuestenpixel.sum()),
        })
    return zeilen


def bewerten(zeilen):
    """Den gemeinsamen Wert je Region und den Index zum Median."""
    for z in zeilen:
        dichte = z["kueste"] / max(z["land"], 1)
        z["kuestendichte"] = dichte
        z["kuestenbonus"] = 1.0 + (KUESTENBONUS_MAX - 1.0) * min(
            dichte / KUESTENDICHTE_VOLL, 1.0)
        z["roh"] = (z["bewohnbar"] * BEWOHNBAR_GEWICHT
                    + z["unbewohnbar"] * UNBEWOHNBAR_GEWICHT
                    + z["wasser"] * WASSER_GEWICHT)
        z["wert"] = z["roh"] * z["kuestenbonus"]

    median = float(np.median([z["wert"] for z in zeilen])) or 1.0
    for z in zeilen:
        z["index"] = z["wert"] / median
        z["ziel"] = ZIELWERT.get(z["name"], ZIELWERT_STANDARD)
        z["abweichung"] = z["index"] - z["ziel"]
    return zeilen


def run_fairness():
    zeilen = bewerten(messen())
    ok = check("genug Regionen gemessen", len(zeilen) >= 8,
               f"{len(zeilen)} Regionen")
    if not ok:
        return False

    print()
    print(f"       {'Region':<21}{'bewohn.':>9}{'unbewohn':>9}{'Wasser':>8}"
          f"{'Kueste':>8}{'Bonus':>7}{'Index':>7}{'Ziel':>6}{'Diff':>7}")
    print("       " + "-" * 82)
    for z in sorted(zeilen, key=lambda x: -x["index"]):
        marke = "  <-- daneben" if abs(z["abweichung"]) > TOLERANZ else ""
        print(f"       {z['name']:<21}{z['bewohnbar']:>9}{z['unbewohnbar']:>9}"
              f"{z['wasser']:>8}{z['kuestendichte']:>7.1%}"
              f"{z['kuestenbonus']:>7.2f}{z['index']:>7.2f}{z['ziel']:>6.2f}"
              f"{z['abweichung']:>+7.2f}{marke}")
    print()

    daneben = [z for z in zeilen if abs(z["abweichung"]) > TOLERANZ]
    ok &= check(
        f"jede Region im Rahmen (+-{TOLERANZ:.0%} vom Zielwert)",
        not daneben,
        ", ".join(f"{z['name']} {z['index']:.2f} statt {z['ziel']:.2f}"
                  for z in daneben) if daneben
        else f"groesste Abweichung {max(abs(z['abweichung']) for z in zeilen):.2f}")

    # DIE SPANNE ist die eigentliche Fairnessfrage: wie weit liegen die
    # beste und die schlechteste Region auseinander?
    werte = [z["index"] / z["ziel"] for z in zeilen]
    spanne = max(werte) / max(min(werte), 1e-9)
    ok &= check("Spanne zwischen bester und schlechtester Region",
                spanne < 2.0,
                f"Faktor {spanne:.2f} "
                f"({max(zeilen, key=lambda z: z['index'] / z['ziel'])['name']} "
                f"gegen "
                f"{min(zeilen, key=lambda z: z['index'] / z['ziel'])['name']})")
    return ok


def run_bestandteile():
    """
    Die Rohgroessen einzeln - damit man SIEHT, woran eine Schieflage liegt.

    Ohne diese Aufschluesselung sagt der Index nur, DASS eine Region
    danebenliegt, nicht warum. Und ohne das Warum ist er nicht zu beheben.
    """
    zeilen = bewerten(messen())
    print()
    print(f"       {'Region':<21}{'Flaeche':>9}{'Land%':>8}{'steil%':>8}"
          f"{'hoch%':>8}{'nutzbar%':>10}")
    print("       " + "-" * 66)
    for z in sorted(zeilen, key=lambda x: -x["bewohnbar"]):
        g = max(z["gesamt"], 1)
        l = max(z["land"], 1)
        print(f"       {z['name']:<21}{z['gesamt']:>9}"
              f"{z['land'] / g:>7.0%}{z['zu_steil'] / l:>8.0%}"
              f"{z['zu_hoch'] / l:>8.0%}{z['bewohnbar'] / g:>9.0%}")
    print()
    # Eine echte Zusicherung, keine reine Ausgabe: eine Region ohne jedes
    # bewohnbare Land waere unbrauchbar, egal was der Index sagt.
    ohne = [z["name"] for z in zeilen if z["bewohnbar"] < 0.05 * z["gesamt"]]
    return check("jede Region hat nennenswert bewohnbares Land",
                 not ohne,
                 ", ".join(ohne) if ohne
                 else f"kleinster Anteil "
                      f"{min(z['bewohnbar'] / max(z['gesamt'], 1) for z in zeilen):.0%}")


def main():
    print("=" * 78)
    print(f"REGIONEN-FAIRNESS  ({SIZE} px, Seed {SEED})")
    print("=" * 78)
    print(f"unbewohnbar ab {UNBEWOHNBAR_HANG_GRAD:.0f} Grad Hang oder "
          f"{UNBEWOHNBAR_HOEHE_M:.0f} m Hoehe")
    print(f"Gewichte: bewohnbar {BEWOHNBAR_GEWICHT:.2f}, unbewohnbar "
          f"{UNBEWOHNBAR_GEWICHT:.2f}, Wasser {WASSER_GEWICHT:.2f}, "
          f"Kuestenbonus bis {KUESTENBONUS_MAX:.2f}")
    print()

    ergebnis = {}
    for titel, fn in (("Fairness", run_fairness),
                      ("Bestandteile", run_bestandteile)):
        print(f"--- {titel} ---")
        ergebnis[titel] = fn()

    print()
    print("=" * 78)
    gut = sum(1 for v in ergebnis.values() if v)
    print(f"{gut}/{len(ergebnis)} Gruppen gruen")
    return 0 if gut == len(ergebnis) else 1


if __name__ == "__main__":
    sys.exit(main())
