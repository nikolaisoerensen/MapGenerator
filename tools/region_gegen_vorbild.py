"""
Path: tools/region_gegen_vorbild.py

Eine erzeugte Region gegen ihr echtes Vorbild halten.

NUTZERAUFTRAG 2026-08-24:

  *"gibt es eine testmoeglichkeit um zB Geiranger region ueber
  Opentopology abzugreifen und mit terrain noise nachzubilden? also wir
  vergleichen die hoehenprofile und bilden das mit den parametern oben nach
  und dann skalieren wir das ganze etwas herunter (weil in unserem spiel ja
  1000 m hoehe in etwa 3000 m entspricht oder sowas aehnliches und weil wir
  in unserer flaeche von fjordland insgesamt dann ja nur 1-2 fjords drauf
  bekommen und das muss von der skalierung etwa passen. aber es geht mehr
  um die formen auch."*

## Was verglichen wird, und warum nicht die Hoehen selbst

Ein direkter Hoehenvergleich waere sinnlos: Geiranger steigt auf ueber
1400 m, das Skerrheim dieser Welt auf rund 500. Das ist gewollt - die
Spielwelt ist gestaucht. Verglichen werden deshalb **Formkennzahlen**,
die von der absoluten Groesse unabhaengig sind:

  Kuestendichte      Kuestenlinie je Landflaeche - das Mass fuer
                     Zerkluftung. Ein Fjord hat viel davon, eine runde
                     Bucht wenig.
  Hangverteilung     Perzentile der Hangneigung. Sagt, ob steile Waende
                     die Ausnahme oder die Regel sind.
  Hoehenverteilung   Perzentile, auf die eigene Spanne normiert. Sagt, ob
                     das Gelaende aus Plateaus oder aus Graten besteht.
  Wasseranteil       Wieviel der Flaeche unter dem Meeresspiegel liegt.
  Formfaktor         Kuestenlaenge / Wurzel(Landflaeche). Dimensionslos -
                     eine Kreisscheibe hat rund 3.5, ein Skerrheim weit
                     mehr.

## Die Skalierung

`--skala` staucht das Vorbild vertikal, bevor verglichen wird
(Nutzervorgabe: 1000 m im Spiel entsprechen etwa 3000 m real, also
Faktor 1/3). Das betrifft NUR die Hangkennzahlen - Kuestendichte und
Formfaktor sind ohnehin masstabsfrei.

DER AUSSCHNITT WIRD MITSKALIERT: das Skerrheim dieser Welt misst rund
9650 Pixel bei 41.6 m, also etwa 16.7 km2. Ein Vorbildausschnitt sollte
in derselben Groessenordnung liegen, sonst vergleicht man eine ganze
Fjordlandschaft mit einem einzelnen Fjord.

Aufruf:
    .venv/Scripts/python.exe tools/region_gegen_vorbild.py Skerrheim geiranger
    .venv/Scripts/python.exe tools/region_gegen_vorbild.py Skerrheim geiranger --skala 0.33
"""

import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import core.terrain_weltkarte as rw
from tools.kuestenlaengsschnitt import STRECKEN, dem_holen

SIZE = 512
SEED = 20260804


def _kennzahlen(H, mpp, name, gebiet=None):
    """
    Die masstabsfreien Formkennzahlen eines Hoehenfelds.

    `gebiet` grenzt die Auswertung ein. Ohne diese Maske war der
    Wasseranteil der erzeugten Region 97.5 % - alles AUSSERHALB der
    Region ist auf -999 gesetzt, damit die Kuestenlinie am Regionsrand
    nicht mitzaehlt, und wurde dann als Wasser mitgezaehlt (gemessen und
    behoben 2026-08-24).
    """
    H = np.asarray(H, dtype=np.float64)
    land = H > 0.0
    if gebiet is None:
        gebiet = np.ones(H.shape, dtype=bool)
    else:
        gebiet = np.asarray(gebiet, dtype=bool)
        land = land & gebiet
    if land.sum() < 50:
        return None

    dy, dx = np.gradient(H, mpp)
    hang = np.degrees(np.arctan(np.hypot(dx, dy)))

    # Kuestenlinie: Landpixel mit Wassernachbar. Mal Pixelbreite ergibt
    # eine Laenge - grob, aber fuer den VERGLEICH zweier Felder mit
    # derselben Methode ausreichend.
    kueste = ndimage.binary_dilation(~land, iterations=1) & land
    kuesten_m = float(kueste.sum()) * mpp
    land_m2 = float(land.sum()) * mpp * mpp
    gebiet_px = max(float(gebiet.sum()), 1.0)

    h_land = H[land]
    spanne = max(float(np.percentile(h_land, 98) - np.percentile(h_land, 2)), 1e-6)
    h_norm = (h_land - float(np.percentile(h_land, 2))) / spanne

    return {
        "name": name,
        "flaeche_km2": land_m2 / 1e6,
        "wasseranteil": float((gebiet & ~land).sum()) / gebiet_px,
        "kuestendichte": float(kueste.sum()) / max(float(land.sum()), 1.0),
        # Dimensionslos: eine Kreisscheibe kommt auf rund 3.5, je
        # zerklufteter desto groesser.
        "formfaktor": kuesten_m / max(np.sqrt(land_m2), 1e-6),
        "hang_p50": float(np.percentile(hang[land], 50)),
        "hang_p90": float(np.percentile(hang[land], 90)),
        "hang_p99": float(np.percentile(hang[land], 99)),
        "hoehe_max": float(h_land.max()),
        # Form der Hoehenverteilung, auf die eigene Spanne normiert -
        # unabhaengig davon, wie hoch das Gebirge wirklich ist.
        "h_p25": float(np.percentile(h_norm, 25)),
        "h_p50": float(np.percentile(h_norm, 50)),
        "h_p75": float(np.percentile(h_norm, 75)),
        # Ab hier ROHDATEN, nicht Kennzahlen - fuer
        # tests/smoke_test_region_vorbild_aehnlichkeit.py (KS-Test auf der
        # vollen Hoehenverteilung, Box-Counting auf der Kuestenlinie). Rein
        # additiv, die CLI oben liest nur die benannten Kennzahlen und bleibt
        # unveraendert.
        "h_norm_werte": h_norm,
        "land_maske": land,
    }


def vorbild(schluessel, skala):
    """Kennzahlen der echten Kueste, vertikal gestaucht."""
    eintrag = STRECKEN.get(schluessel)
    if eintrag is None:
        print(f"Unbekannte Strecke: {schluessel}")
        print("Bekannt:", ", ".join(sorted(STRECKEN)))
        return None
    werte, kopf = dem_holen(eintrag["sued"], eintrag["nord"],
                            eintrag["west"], eintrag["ost"], "COP30")
    mpp = kopf["cellsize"] * 111320.0
    H = np.nan_to_num(np.asarray(werte, dtype=np.float64), nan=-1.0)
    # NUR DIE HOEHEN STAUCHEN, nicht die Flaeche. Die Spielwelt ist
    # vertikal gestaucht, nicht horizontal - ein Fjord ist dort genauso
    # lang, aber weniger hoch.
    return _kennzahlen(H * skala, mpp, f"{eintrag['titel'][:34]} (x{skala:.2f})")


def region(name):
    """Kennzahlen der erzeugten Region."""
    H, felder = rw.weltfeld(SIZE, SEED)
    H = np.asarray(H, dtype=np.float64)
    mpp = rw.WELT_KM * 1000.0 / SIZE
    maske, _sdf = rw.kontinentform(SIZE, SEED)
    reg = felder["regionen"]
    index = None
    for i, (_z, _s, r) in enumerate(rw.alle_regionen()):
        if r["name"] == name:
            index = i
            break
    if index is None:
        print(f"Unbekannte Region: {name}")
        return None
    m = (reg == index) & maske
    # Ausserhalb der Region auf "tiefes Wasser" setzen, damit die
    # Kuestenlinie am Regionsrand nicht mitgezaehlt wird.
    aus = np.where(m, H, -999.0)
    return _kennzahlen(aus, mpp, f"{name} (erzeugt)", gebiet=m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region", help="z.B. Skerrheim")
    p.add_argument("strecke", help="z.B. geiranger")
    p.add_argument("--skala", type=float, default=1.0 / 3.0,
                   help="vertikale Stauchung des Vorbilds (Vorgabe 1/3)")
    args = p.parse_args()

    a = region(args.region)
    b = vorbild(args.strecke, args.skala)
    if a is None or b is None:
        return 1

    print()
    print("=" * 76)
    print(f"{args.region} gegen {args.strecke}")
    print("=" * 76)
    print(f"{'Kennzahl':<24}{'erzeugt':>14}{'Vorbild':>14}{'Verhaeltnis':>14}")
    print("-" * 76)
    zeilen = [
        ("Landflaeche km2", "flaeche_km2", "{:.1f}"),
        ("Wasseranteil", "wasseranteil", "{:.1%}"),
        ("Kuestendichte", "kuestendichte", "{:.1%}"),
        ("Formfaktor", "formfaktor", "{:.1f}"),
        ("Hang Median", "hang_p50", "{:.1f}"),
        ("Hang p90", "hang_p90", "{:.1f}"),
        ("Hang p99", "hang_p99", "{:.1f}"),
        ("Hoehe max m", "hoehe_max", "{:.0f}"),
        ("Hoehe p25 (norm)", "h_p25", "{:.2f}"),
        ("Hoehe p50 (norm)", "h_p50", "{:.2f}"),
        ("Hoehe p75 (norm)", "h_p75", "{:.2f}"),
    ]
    for label, schluessel, fmt in zeilen:
        va, vb = a[schluessel], b[schluessel]
        verh = va / vb if abs(vb) > 1e-9 else float("nan")
        print(f"{label:<24}{fmt.format(va):>14}{fmt.format(vb):>14}"
              f"{verh:>13.2f}x")

    verh_flaeche = a["flaeche_km2"] / max(b["flaeche_km2"], 1e-9)
    if not 0.4 < verh_flaeche < 2.5:
        print()
        print(f"ACHTUNG: die Flaechen unterscheiden sich um Faktor "
              f"{1/verh_flaeche if verh_flaeche < 1 else verh_flaeche:.1f}.")
        print("Kuestendichte und Formfaktor haengen an der Ausschnittsgroesse:")
        print("eine grosse Kachel enthaelt ganze Fjordsysteme, eine kleine nur")
        print("einen Hang. Fuer einen fairen Vergleich sollte der Vorbild-")
        print("ausschnitt aehnlich gross sein wie die erzeugte Region.")

    print()
    print("DEUTUNG - was die Zahlen sagen sollten:")
    print("  Kuestendichte und Formfaktor sind das MASS FUER FJORDE. Liegt")
    print("  die erzeugte Region deutlich darunter, ist die Kueste zu glatt")
    print("  - der Hebel dafuer ist `rauheit` in der Regionentabelle.")
    print("  Die Hang-Perzentile sagen, ob steile Waende Ausnahme oder Regel")
    print("  sind. Die normierten Hoehen-Perzentile sagen, ob das Gelaende")
    print("  aus Plateaus (hohe Werte) oder aus Graten (niedrige) besteht.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
