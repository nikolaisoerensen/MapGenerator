"""
Path: tools/archetyp_profile_messen.py

Kuestenprofile je ARCHETYP, in METERN, jeder aus SEINER EIGENEN Vorbildkueste.

WARUM DIESE FASSUNG (Nutzerbefund 2026-08-24 am Bild "Nebelrode", dann
Auftrag "ich moechte sofort alle profile bekommen"):

Die Vorfassung hatte nur 11 Vorbildstrecken fuer 27 Archetypen. Sie teilte
die Schnitte EINER Region nach h(150 m) sortiert in Drittel und gab jedem
Archetyp eines davon. Das ergab drei Kurven mit praktisch identischer Form,
die sich nur in der Hoehe unterschieden - normiert 0.28/0.26/0.30 nach 50 m
und 0.58/0.57/0.57 nach 100 m. Zwei Fehler auf einmal:

  1. Alle drei kamen aus DERSELBEN Kachel. Fuers Nebelrode war das
     Ruegen - eine Kreidekueste, wo alles steil ist. Eine
     "Ostsee-Flachkueste", die nach 150 m auf 73 m steigt, ist keine.
  2. Sortieren nach h(150 m) und dann h(150 m) messen ist ein
     Zirkelschluss: die Drittel unterscheiden sich zwangslaeufig in genau
     dieser Groesse und in sonst nichts.

Jetzt hat jeder Archetyp seine eigene Kueste (tools/archetyp_vorbilder.py),
und die Formunterschiede sind gemessen statt konstruiert.

DIE MESSUNG SELBST: h(x) ueber der Uferhoehe, an festen Stellen von x = 0
bis PROFIL_TIEFE_M. Kein Abbruchkriterium, keine Normierung, keine
Streckung - passend zu den festen Zonen (0-350 m volles Profil, 350-500 m
Uebergang) und zur Vorgabe *"gleiches hoehen zu tiefen verhaeltnis wie in
echt"*.

Aufruf:
    .venv/Scripts/python.exe tools/archetyp_profile_messen.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.archetyp_vorbilder import VORBILDER
from tools.kuestenlaengsschnitt import (dem_holen, kuestenlinie,
                                        schnitte_auf_normalen)

PROFIL_TIEFE_M = 900.0
STUETZSTELLEN = 19                      # alle 50 m

# DIE SEESEITE (Nutzervorgabe 2026-08-24):
#   *"Wenn x < 0 (Meer): Blende ueber eine Distanz von 200m (x = 0 bis
#   x = -200) sanft mittels Smoothstep von coast_profile(x) zu
#   ocean_profile(x) ueber. Fuer x < -200 gilt 100% ocean_profile(x)."*
#
# Dafuer braucht es das gemessene Profil auch UNTER Wasser. Die Schnitte
# reichen ohnehin INS_MEER_M = 250 m hinaus; bis hierher wurde nur die
# Landseite ausgewertet.
#
# WAS DAS DEM ist UND WAS NICHT: COP30 ist ein OBERFLAECHENmodell und
# setzt offenes Meer auf 0. Was hier als "Tiefe" herauskommt, ist
# deshalb nicht die echte Bathymetrie, sondern der Verlauf der
# Wasserlinie im Modell - nahe Null und leicht negativ dort, wo Fels
# unter die Oberflaeche taucht. Es beschreibt die FORM des Uferansatzes,
# nicht den Meeresboden. Die eigentliche Meerestiefe kommt weiterhin aus
# dem Seegrad-Prozess (_seetiefe_aus_archetyp).
SEE_TIEFE_M = 250.0
SEE_STUETZSTELLEN = 6                   # alle 50 m


def _profil_see(strecke, hoehen, stellen_see):
    """
    h(x) fuer x < 0, ueber der Uferhoehe. `stellen_see` ist POSITIV
    (Abstand ins Meer), das Ergebnis in derselben Ordnung.
    """
    see = strecke <= 0.0
    s, h = strecke[see], np.asarray(hoehen, dtype=np.float64)[see]
    if len(s) < 4:
        return None
    fehlt = ~np.isfinite(h)
    if fehlt.mean() > 0.25:
        return None
    if fehlt.any():
        h = np.interp(s, s[~fehlt], h[~fehlt])
    # Uferhoehe ist der Wert bei x = 0, also der LETZTE der Seeseite.
    ufer = h[-1]
    if -s[0] < stellen_see[-1] * 0.98:
        return None
    return np.interp(-stellen_see[::-1], s, h)[::-1] - ufer


def _profil_in_metern(strecke, hoehen, stellen):
    """h(x) ueber der Uferhoehe. Kein Abbruch, keine Normierung."""
    land = strecke >= 0.0
    s, h = strecke[land], np.asarray(hoehen, dtype=np.float64)[land]
    if len(s) < 8:
        return None
    fehlt = ~np.isfinite(h)
    if fehlt.mean() > 0.25:
        return None
    if fehlt.any():
        h = np.interp(s, s[~fehlt], h[~fehlt])
    if s[-1] < stellen[-1] * 0.98:
        return None
    return np.interp(stellen, s, h) - h[0]


def eine_kueste(v, stellen):
    werte, kopf = dem_holen(v["sued"], v["nord"], v["west"], v["ost"], "COP30")
    linien = kuestenlinie(werte, kopf)
    if not linien:
        return None
    linie = linien[0][1]
    if len(linie) < 3:
        return None
    _s, _b, strecke, profile = schnitte_auf_normalen(werte, kopf, linie)
    stellen_see = np.linspace(0.0, SEE_TIEFE_M, SEE_STUETZSTELLEN)
    gut, gut_see = [], []
    for pr in profile:
        h = _profil_in_metern(strecke, pr, stellen)
        if h is None or not np.isfinite(h).all():
            continue
        hs = _profil_see(strecke, pr, stellen_see)
        if hs is None or not np.isfinite(hs).all():
            continue
        gut.append(h)
        gut_see.append(hs)
    if len(gut) < 6:
        return None
    return np.array(gut), np.array(gut_see)


def main():
    stellen = np.linspace(0.0, PROFIL_TIEFE_M, STUETZSTELLEN)
    tabelle, tabelle_see, zeilen = {}, {}, []
    for name, v in VORBILDER.items():
        try:
            aus = eine_kueste(v, stellen)
        except Exception as e:                                # noqa: BLE001
            print(f"{name:<24} FEHLER: {e}")
            continue
        if aus is None:
            print(f"{name:<24} zu wenige brauchbare Schnitte")
            continue
        P, P_see = aus
        # Seeseite: Median, dann MONOTON FALLEND nach aussen - ein Ufer,
        # das seewaerts wieder ansteigt, waere eine Sandbank und gehoert
        # nicht ins Uferprofil.
        m_see = np.minimum.accumulate(np.median(P_see, axis=0)[::-1])[::-1]
        tabelle_see[name] = m_see
        # Median ueber alle Schnitte, dann monoton: eine Kueste, die
        # landeinwaerts wieder abfaellt, ist ein Tal und gehoert nicht
        # ins Kuestenprofil.
        m = np.maximum.accumulate(np.median(P, axis=0))
        tabelle[name] = m
        zeilen.append((v["region"], name, len(P), m))

    print(f"{'Region':<20}{'Archetyp':<24}{'n':>5}"
          f"{'h(150)':>8}{'h(350)':>8}{'h(500)':>8}{'h(900)':>8}"
          f"{'Steilheit':>11}")
    print("-" * 92)
    letzte = None
    for region, name, n, m in sorted(zeilen):
        if region != letzte:
            print(); letzte = region
        h150 = np.interp(150.0, stellen, m)
        print(f"{region:<20}{name:<24}{n:>5}"
              f"{h150:>6.0f} m{np.interp(350.0, stellen, m):>6.0f} m"
              f"{np.interp(500.0, stellen, m):>6.0f} m{m[-1]:>6.0f} m"
              f"{h150 / 150.0:>11.3f}")

    print()
    print()
    print(f"{'Archetyp':<24}{'h(-50)':>9}{'h(-100)':>9}{'h(-250)':>9}")
    print("-" * 52)
    st_see = np.linspace(0.0, SEE_TIEFE_M, SEE_STUETZSTELLEN)
    for name in sorted(tabelle_see):
        m = tabelle_see[name]
        print(f"{name:<24}{np.interp(50.0, st_see, m):>7.1f} m"
              f"{np.interp(100.0, st_see, m):>7.1f} m{m[-1]:>7.1f} m")

    print()
    print("Als Tabelle fuer core/vektor_kueste.py:")
    print(f"SEE_STELLEN_M = {tuple(float(v) for v in st_see)}")
    print("MESS_SEEPROFIL_M_JE_ARCHETYP = {")
    for name in sorted(tabelle_see):
        v = tabelle_see[name]
        print(f'    "{name}": ({", ".join(f"{q:.2f}" for q in v)}),')
    print("}")
    print(f"PROFIL_STELLEN_M = {tuple(float(v) for v in stellen)}")
    print("MESS_PROFIL_M_JE_ARCHETYP = {")
    for name in sorted(tabelle):
        v = tabelle[name]
        bloecke = ",\n        ".join(", ".join(f"{q:.1f}" for q in v[i:i + 7])
                                     for i in range(0, len(v), 7))
        print(f'    "{name}": (\n        {bloecke}),')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
