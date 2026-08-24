"""
Path: tools/vorbilder_holen.py

Holt fuer JEDEN der 27 Archetypen seine eigene Vorbildkachel.

Laeuft `tools/archetyp_vorbilder.VORBILDER` ab, ruft die fehlenden bei
OpenTopography ab (vorhandene kommen aus tools/_dem_cache/) und meldet je
Ort, ob eine Kuestenlinie gefunden wurde und wie viele Schnitte brauchbar
sind. Ein Ausschnitt ohne genug Schnitte muss nachgebessert werden - das
steht dann in der Spalte "Befund".

Der API-Schluessel kommt aus OPENTOPOGRAPHY_API_KEY und wird nirgends
ausgegeben.

Aufruf:
    .venv/Scripts/python.exe tools/vorbilder_holen.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.archetyp_vorbilder import VORBILDER
from tools.kuestenlaengsschnitt import dem_holen, kuestenlinie, schnitte_auf_normalen

def pruefen(werte, kopf):
    linien = kuestenlinie(werte, kopf)
    if not linien:
        return 0, 0.0, "KEINE Kuestenlinie"
    laenge_m, linie = linien[0]
    zelle_m = kopf["cellsize"] * 111320.0
    km = laenge_m / 1000.0
    if len(linie) < 3:
        return 0, km, "Kontur zu kurz"
    _s, _b, strecke, profile = schnitte_auf_normalen(werte, kopf, linie)
    land = strecke >= 0
    gut = 0
    for p in profile:
        h = np.asarray(p)[land]
        if np.isfinite(h).mean() > 0.75:
            gut += 1
    befund = ("ok" if gut >= 20 else
              "wenige Schnitte" if gut >= 6 else "ZU WENIGE")
    return gut, km, befund

print(f"{'Archetyp':<24}{'Ort':<40}{'Kueste':>8}{'Schnitte':>9}  Befund")
print("-" * 96)
schlecht = []
for name, v in VORBILDER.items():
    try:
        werte, kopf = dem_holen(v["sued"], v["nord"], v["west"], v["ost"], "COP30")
        gut, km, befund = pruefen(werte, kopf)
    except Exception as e:
        gut, km, befund = 0, 0.0, f"FEHLER: {str(e)[:40]}"
    print(f"{name:<24}{v['titel'][:38]:<40}{km:>6.1f}km{gut:>9}  {befund}")
    if befund != "ok":
        schlecht.append((name, befund))
    if not v.get("vorhanden"):
        time.sleep(1.0)          # hoeflich gegen die API

print()
if schlecht:
    print(f"NACHBESSERN ({len(schlecht)}):")
    for n, b in schlecht:
        print(f"   {n:<24} {b}")
else:
    print("Alle 27 Ausschnitte brauchbar.")
