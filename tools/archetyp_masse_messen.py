"""
Path: tools/archetyp_masse_messen.py

Distanz UND Hoehe je ARCHETYP messen, nicht je Region.

WARUM (Nutzervorgabe 2026-08-24):

  *"die kuestenprofile sind dadurch nicht gestreckt oder gestaucht sondern
  haben das gleiche hoehen zu tiefen verhaeltnis wie in echt"*

Damit ein Profil sein echtes Hoehen-zu-Tiefen-Verhaeltnis behaelt, muss es
in METERN ausgewertet werden - und dafuer braucht es je Archetyp beide
Masse: ueber welche Strecke er laeuft und auf welche Hoehe.

WAS HEUTE DA IST, UND WAS FEHLT:

  MESS_FORM_JE_ARCHETYP   27 Formen, je ARCHETYP        - vorhanden
  MESS_REICHWEITE_M        9 Distanzen, je REGION       - zu grob
  MESSWERTE_JE_REGION      9 Hoehenbaender, je REGION   - zu grob

Die Formen entstanden, indem die Schnitte einer Vorbildstrecke nach
Steilheit sortiert und auf die Archetypen ihrer Region verteilt wurden -
eine Region hat drei Archetypen unterschiedlichen Charakters, und der
steilste Drittel gehoert zur Steilkueste, der flachste zum Strand.
Distanz und Hoehe wurden dabei NICHT mit aufgeteilt; beide blieben
Regionswerte. Genau das erzeugt die vier bekannten Hoehenausreisser
(smoke_test_kuestenprofiltreue): die Morobora-Hoehen stammen von den
Stockholmer Schaeren (15-30 m) und gelten deshalb auch fuer die
Kola-Steilkueste, die das gar nicht einhalten kann.

Dieses Werkzeug wiederholt dieselbe Aufteilung und misst diesmal ALLE
DREI Groessen je Archetyp.

KEIN API-KONTINGENT NOETIG: `dem_holen()` liest aus `tools/_dem_cache/`,
und die 23 Kacheln der Vorbildstrecken liegen dort bereits.

Aufruf:
    .venv/Scripts/python.exe tools/archetyp_masse_messen.py
    .venv/Scripts/python.exe tools/archetyp_masse_messen.py --bild
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.kuestenlaengsschnitt import (STRECKEN, dem_holen, kuestenlinie,
                                        schnitte_auf_normalen)
from core.terrain_weltkarte import alle_regionen, KUESTEN_ARCHETYPEN

# Ab wann gilt das Kuestenprofil als zu Ende. DASSELBE Kriterium, mit dem
# MESS_REICHWEITE_M bestimmt wurde: der Anstieg ist auf diesen Bruchteil
# seines Hoechstwerts gefallen. Saettigungskriterien ("Hoehe erreicht 95 %
# des Endwerts") sind an echten Kuesten unbrauchbar - reales Gelaende
# erreicht innerhalb von 900 m nie ein Plateau.
ANSTIEG_RESTANTEIL = 0.10

# Glaettungsfenster fuer den Anstieg, in Stuetzstellen. Ohne Glaettung
# entscheidet Messrauschen ueber die Reichweite.
GLAETTUNG = 9


def _reichweite_und_hoehe(strecke, profil):
    """
    (Reichweite in m, Hoehe in m) eines EINZELNEN Schnitts.

    Gerechnet wird ab der Wasserlinie (strecke = 0) landeinwaerts. Die
    Hoehe ist die Hoehe am Ende der Reichweite ueber der Uferhoehe, nicht
    das Maximum ueber die ganze Landseite - sonst misst man das Hinterland
    mit (derselbe Messfehler wie in der ersten Fassung von
    smoke_test_kuestenprofiltreue).
    """
    land = strecke >= 0.0
    s = strecke[land]
    h = np.asarray(profil, dtype=np.float64)[land]
    if len(s) < GLAETTUNG + 2:
        return np.nan, np.nan
    # Einzelne NaN kommen am Kachelrand vor - ueberbruecken statt den
    # ganzen Schnitt zu verwerfen. Mehr als ein Viertel Luecken heisst
    # aber, dass der Schnitt aus der Kachel laeuft.
    fehlt = ~np.isfinite(h)
    if fehlt.mean() > 0.25:
        return np.nan, np.nan
    if fehlt.any():
        h = np.interp(s, s[~fehlt], h[~fehlt])

    ufer = h[0]
    kern = np.ones(GLAETTUNG) / GLAETTUNG
    anstieg = np.convolve(np.gradient(h, s), kern, mode="same")
    hoechst = float(np.nanmax(anstieg[:len(anstieg) // 2]))
    if not np.isfinite(hoechst) or hoechst <= 0.0:
        return np.nan, np.nan

    # Erste Stelle NACH dem Hoechstanstieg, an der er unter die Schwelle faellt.
    ab = int(np.nanargmax(anstieg[:len(anstieg) // 2]))
    unter = np.flatnonzero(anstieg[ab:] < ANSTIEG_RESTANTEIL * hoechst)
    if not len(unter):
        return np.nan, np.nan
    ende = ab + int(unter[0])
    return float(s[ende]), float(h[ende] - ufer)


def _steilheit(strecke, profil):
    """Ein Mass, nach dem die Schnitte auf die Archetypen verteilt werden."""
    r, hoehe = _reichweite_und_hoehe(strecke, profil)
    if not np.isfinite(r) or r <= 0.0:
        return np.nan
    return hoehe / r


def eine_strecke(schluessel, eintrag):
    """Alle Schnitte einer Vorbildstrecke, mit Reichweite/Hoehe/Steilheit."""
    werte, kopf = dem_holen(eintrag["sued"], eintrag["nord"],
                            eintrag["west"], eintrag["ost"], "COP30")
    # kuestenlinie() liefert eine LISTE von (laenge, zug)-Paaren, nach
    # Laenge absteigend - die laengste ist die Hauptkueste der Kachel.
    linien = kuestenlinie(werte, kopf)
    if not linien:
        return None
    linie = linien[0][1]
    if len(linie) < 3:
        return None
    _st, _bogen, strecke, profile = schnitte_auf_normalen(werte, kopf, linie)
    reich = np.empty(len(profile))
    hoehe = np.empty(len(profile))
    for k, p in enumerate(profile):
        reich[k], hoehe[k] = _reichweite_und_hoehe(strecke, p)
    gut = np.isfinite(reich) & np.isfinite(hoehe) & (reich > 0)
    if gut.sum() < 6:
        return None
    return reich[gut], hoehe[gut], (hoehe[gut] / reich[gut])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bild", action="store_true")
    args = p.parse_args()

    # Region -> Archetypen, in der Reihenfolge der Tabelle
    je_region = {}
    for _z, _s, r in alle_regionen():
        je_region[r["name"]] = list(KUESTEN_ARCHETYPEN.get(r["name"], []))

    ergebnis = {}
    print(f"{'Strecke':<16}{'Region':<22}{'Schnitte':>9}"
          f"{'Reichw. Median':>16}{'Hoehe Median':>14}")
    print("-" * 78)
    messungen = {}
    for schluessel, eintrag in STRECKEN.items():
        region = eintrag.get("region", "?")
        if region.startswith("Insel"):
            continue
        try:
            aus = eine_strecke(schluessel, eintrag)
        except Exception as e:                                # noqa: BLE001
            print(f"{schluessel:<16}{region:<22}  FEHLER: {e}")
            continue
        if aus is None:
            print(f"{schluessel:<16}{region:<22}  zu wenige brauchbare Schnitte")
            continue
        reich, hoehe, steil = aus
        messungen.setdefault(region, []).append((reich, hoehe, steil))
        print(f"{schluessel:<16}{region:<22}{len(reich):>9}"
              f"{np.median(reich):>14.0f} m{np.median(hoehe):>12.0f} m")

    print()
    print("=" * 78)
    print("JE ARCHETYP - die Schnitte einer Region nach Steilheit sortiert")
    print("und auf ihre Archetypen verteilt (flachster Teil -> flachster Typ)")
    print("=" * 78)
    print(f"{'Region':<22}{'Archetyp':<24}{'n':>5}"
          f"{'Distanz':>10}{'Hoehe':>9}{'H/D':>8}")
    print("-" * 78)

    for region, teile in sorted(messungen.items()):
        reich = np.concatenate([t[0] for t in teile])
        hoehe = np.concatenate([t[1] for t in teile])
        steil = np.concatenate([t[2] for t in teile])
        typen = je_region.get(region, [])
        if not typen:
            continue
        # Nach Steilheit sortieren, dann in so viele Teile wie Archetypen.
        # Die Archetypen stehen in der Tabelle vom anspruchsvollsten zum
        # flachsten - deshalb wird die Sortierung umgedreht zugeordnet.
        ordnung = np.argsort(steil)
        stuecke = np.array_split(ordnung, len(typen))
        for typ, teil in zip(reversed(typen), stuecke):
            if not len(teil):
                continue
            d = float(np.median(reich[teil]))
            h = float(np.median(hoehe[teil]))
            ergebnis[typ["name"]] = (d, h)
            print(f"{region:<22}{typ['name']:<24}{len(teil):>5}"
                  f"{d:>8.0f} m{h:>7.0f} m{h / max(d, 1e-9):>8.3f}")

    print()
    print("Als Tabelle fuer core/vektor_kueste.py:")
    print("MESS_ARCHETYP_MASSE = {")
    for name, (d, h) in sorted(ergebnis.items()):
        print(f'    "{name}": ({d:.0f}, {h:.0f}),')
    print("}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
