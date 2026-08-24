"""
Path: tools/kuestenkatalog_bauen.py

ALLE VORBILDKUESTEN AM STUECK VERMESSEN - Grundlage fuer den Kuestentyp-
Katalog (Nutzer-Vorgabe 2026-08-18).

Faehrt alle Eintraege aus `kuestenlaengsschnitt.STRECKEN` ab, misst je Ort
die Kennzahlen, die spaeter im Katalog stehen (Zielhoehe, Anstiegsstrecke,
Aenderungsrate laengs), und legt zwei Uebersichtsbilder an: eines fuer die
Festlandskuesten, eines fuer die Inseln.

Die Abrufe sind zwischengespeichert (`tools/_dem_cache/`) - ein zweiter Lauf
kostet kein API-Kontingent.

Aufruf:
    .venv/Scripts/python.exe tools/kuestenkatalog_bauen.py
    .venv/Scripts/python.exe tools/kuestenkatalog_bauen.py --nur inseln
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.kuestenlaengsschnitt import (STRECKEN, kennzahlen, kuestenlinie,
                                        schnitte_auf_normalen)
from tools.kuestenprofil_holen import CACHE, dem_holen

ABSTAND_M = 100.0


def einen_ort(name, eintrag, demtype="COP30"):
    werte, kopf = dem_holen(eintrag["sued"], eintrag["nord"],
                            eintrag["west"], eintrag["ost"], demtype)
    linien = kuestenlinie(werte, kopf, mindestlaenge_m=400.0)
    if not linien:
        print(f"  {name}: KEINE Kuestenlinie gefunden")
        return None
    laenge_m, linie = linien[0]
    stationen, bogen, strecke, profile = schnitte_auf_normalen(
        werte, kopf, linie, ABSTAND_M)
    if len(stationen) < 3:
        print(f"  {name}: zu wenige Stationen ({len(stationen)})")
        return None
    gipfel, breite = kennzahlen(strecke, profile)
    aenderung = np.abs(np.diff(gipfel)) / (ABSTAND_M / 1000.0)
    return {
        "name": name, "titel": eintrag["titel"], "region": eintrag["region"],
        "werte": werte, "kopf": kopf, "linie": linie, "stationen": stationen,
        "bogen": bogen, "strecke": strecke, "profile": profile,
        "gipfel": gipfel, "breite": breite,
        "kuestenlaenge_km": laenge_m / 1000.0,
        "n_zuege": len(linien),
        "aenderung_m_km": float(np.nanmedian(aenderung)),
    }


def _bild(ergebnisse, titel, ziel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(ergebnisse)
    spalten = 3
    zeilen = int(np.ceil(n / spalten))
    fig, achsen = plt.subplots(zeilen, spalten,
                               figsize=(5.4 * spalten, 3.7 * zeilen))
    achsen = np.atleast_1d(achsen).ravel()

    for a, e in zip(achsen, ergebnisse):
        farben = plt.cm.rainbow(np.linspace(0, 1, len(e["profile"])))
        land = e["strecke"] >= 0
        for p, f in zip(e["profile"], farben):
            a.plot(e["strecke"][land], np.clip(p[land], 0, None),
                   color=f, lw=.5, alpha=.6)
        a.set_title(f"{e['region']}  -  {e['name']}\n"
                    f"{e['kuestenlaenge_km']:.1f} km, Gipfel "
                    f"{np.nanmin(e['gipfel']):.0f}-{np.nanmax(e['gipfel']):.0f} m",
                    fontsize=9)
        a.set_xlabel("m ab Wasserlinie", fontsize=8)
        a.set_ylabel("Hoehe (m)", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=.3)
    for a in achsen[n:]:
        a.axis("off")
    fig.suptitle(titel, fontsize=14)
    fig.tight_layout()
    fig.savefig(ziel, dpi=95)
    print(f"\nBild: {ziel}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nur", choices=["festland", "inseln", "alle"],
                   default="alle")
    args = p.parse_args()

    inseln = {k: v for k, v in STRECKEN.items()
              if v["region"].startswith("Insel")}
    festland = {k: v for k, v in STRECKEN.items()
                if not v["region"].startswith("Insel")}

    gruppen = []
    if args.nur in ("festland", "alle"):
        gruppen.append(("Festlandskuesten je Region", festland, "katalog_festland.png"))
    if args.nur in ("inseln", "alle"):
        gruppen.append(("Inseln", inseln, "katalog_inseln.png"))

    zeilen_tabelle = []
    for titel, orte, dateiname in gruppen:
        print(f"\n=== {titel} ({len(orte)} Orte) ===")
        ergebnisse = []
        for name, eintrag in orte.items():
            print(f"\n{name} - {eintrag['titel']}")
            try:
                e = einen_ort(name, eintrag)
            except SystemExit as fehler:
                print(f"  uebersprungen: {fehler}")
                continue
            except Exception as fehler:                      # noqa: BLE001
                print(f"  FEHLER: {fehler}")
                continue
            if e is None:
                continue
            ergebnisse.append(e)
            print(f"  Kueste {e['kuestenlaenge_km']:5.1f} km, "
                  f"{len(e['stationen'])} Stationen, {e['n_zuege']} Zuege")
            print(f"  Gipfel {np.nanmin(e['gipfel']):5.0f} .. "
                  f"{np.nanmax(e['gipfel']):5.0f} m, "
                  f"Anstieg bis 90 %: Median {np.nanmedian(e['breite']):5.0f} m, "
                  f"Aenderung {e['aenderung_m_km']:4.0f} m/km")
            zeilen_tabelle.append(e)
        if ergebnisse:
            _bild(ergebnisse, titel, os.path.join(CACHE, dateiname))

    print("\n\n=== KATALOG-ROHTABELLE ===")
    print(f"{'Ort':<18}{'Region':<24}{'Kueste':>8}{'Gipfel min-max':>17}"
          f"{'Anstieg':>9}{'Aend.':>8}")
    print("-" * 84)
    for e in zeilen_tabelle:
        print(f"{e['name']:<18}{e['region']:<24}"
              f"{e['kuestenlaenge_km']:7.1f}km"
              f"{np.nanmin(e['gipfel']):8.0f}-{np.nanmax(e['gipfel']):<8.0f}"
              f"{np.nanmedian(e['breite']):7.0f}m{e['aenderung_m_km']:7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
