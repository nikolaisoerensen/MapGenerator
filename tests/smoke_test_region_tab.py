"""
Path: tests/smoke_test_region_tab.py

Der Regionsreiter - baut er sich auf, und verstellt er dabei etwas?

ANLASS: der Reiter (docs/AUFRAEUMPLAN.md 4.10) laesst die neun
Regionsparameter fuer die aktuelle Karte ueberschreiben. **Der Katalog
bleibt die Vorgabe** (Nutzerentscheidung 2026-08-25), und die Tests der
Pipeline messen weiter gegen ihn. Ein Reiter, der schon beim Aufbau eine
Ueberschreibung erzeugt, wuerde diese Zusage still brechen.

GENAU DAS IST PASSIERT. Der erste Entwurf gab Hoehe und Relief eine
Schrittweite von 10 m. Die Katalogwerte liegen aber nicht auf dieser Stufe
(Clonagh 165.3 m, Skerrheim 484.9 m) - die Regler rasteten beim Aufbau
auf 170 bzw. 480, und der Reiter meldete SOFORT eine Ueberschreibung, ohne
dass jemand etwas angefasst hatte.

Dieselbe Fehlerklasse hatte am selben Tag `MAP_DISTANCE_KM` getroffen (21.3
bei Schritt 1.0 rastet auf 21 - das ist die WELTBREITE, gegen die jede
Meterrechnung geeicht ist). `smoke_test_parameter_eindeutig.py` bewacht die
Reglertabellen; dieser Test bewacht den Reiter, dessen Regler NICHT aus der
Tabelle kommen.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_region_tab.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging

import numpy as np
from PyQt6.QtWidgets import QApplication

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])

import core.terrain_weltkarte as rw
from gui.tabs.region_tab import (EROSIONSREGLER, REGIONSREGLER, RegionTab,
                                 VORSCHAU_BREITE, VORSCHAU_HOEHE)

# Wie lange die Vorschau OHNE Kueste hoechstens brauchen darf. Gemessen
# 0.10-0.29 s; darueber ist "live am Regler" nicht mehr ehrlich.
LIVE_MAX_S = 0.8


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    import time

    fehler = []
    reiter = RegionTab()

    fehler += check("alle neun Regionen im Dropdown",
                    reiter.auswahl.count() == 9,
                    f"{reiter.auswahl.count()}")

    # --- 1. Der Aufbau darf NICHTS ueberschreiben ----------------------------
    fehler += check("frisch geoeffnet ohne Ueberschreibung",
                    not reiter.ueberschreibungen,
                    str(reiter.ueberschreibungen))

    # --- 2. Auch ein Durchlauf durch alle Regionen nicht -------------------
    zeiten = []
    for i in range(reiter.auswahl.count()):
        reiter.auswahl.setCurrentIndex(i)
        reiter._takt.stop()
        t = time.time()
        reiter._neu_zeichnen()
        zeiten.append(time.time() - t)
    fehler += check("Durchlauf aller Regionen ohne Ueberschreibung",
                    not reiter.ueberschreibungen,
                    str(reiter.ueberschreibungen))
    fehler += check(f"Vorschau bleibt live (unter {LIVE_MAX_S} s)",
                    max(zeiten) < LIVE_MAX_S,
                    f"langsamste {max(zeiten):.2f} s, Median "
                    f"{float(np.median(zeiten)):.2f} s")

    # --- 3. Die Regler treffen ihren Katalogwert ----------------------------
    daneben = []
    for i in range(reiter.auswahl.count()):
        reiter.auswahl.setCurrentIndex(i)
        name = reiter.auswahl.currentText()
        katalog = reiter._katalog(name)
        for schluessel, _b, _mn, _mx, schritt in REGIONSREGLER:
            ist = float(reiter.regler[schluessel].getValue())
            soll = float(katalog[schluessel])
            if abs(ist - soll) > 0.5 * schritt:
                daneben.append(f"{name}.{schluessel} {ist:g} statt {soll:g}")
    fehler += check("jeder Regler zeigt seinen Katalogwert",
                    not daneben, ", ".join(daneben[:5]))

    # --- 4. Eine echte Aenderung wird gemerkt und laesst sich zuruecknehmen --
    reiter.auswahl.setCurrentIndex(0)
    reiter._takt.stop()
    reiter._neu_zeichnen()
    vorher = reiter._letztes_feld.copy()
    reiter.regler["rauheit"].setValue(0.85)
    reiter._takt.stop()
    reiter._neu_zeichnen()
    fehler += check("echte Aenderung wird gemerkt",
                    "rauheit" in reiter.ueberschreibungen.get(
                        reiter.auswahl.currentText(), {}),
                    str(reiter.ueberschreibungen))
    fehler += check("echte Aenderung wirkt auf das Gelaende",
                    not np.array_equal(vorher, reiter._letztes_feld))
    reiter._zuruecksetzen()
    fehler += check("Zuruecksetzen raeumt auf",
                    not reiter.ueberschreibungen,
                    str(reiter.ueberschreibungen))

    # --- 5. Feldform und Bild ------------------------------------------------
    H = reiter._letztes_feld
    fehler += check("Feld hat die Vorschauform",
                    H is not None
                    and H.shape == (VORSCHAU_HOEHE, VORSCHAU_BREITE),
                    str(None if H is None else H.shape))
    bild = reiter._als_bild(H)
    fehler += check("2D-Bild wird erzeugt",
                    bild.width() == VORSCHAU_BREITE
                    and bild.height() == VORSCHAU_HOEHE,
                    f"{bild.width()}x{bild.height()}")
    fehler += check("3D-Ansicht ist vorhanden",
                    hasattr(reiter.anzeige_3d, "update_heightmap"),
                    "sonst waere die stehende Regel aus CLAUDE.md verletzt")

    # --- 6. Die Kueste laesst sich zuschalten -------------------------------
    reiter.kueste_an.setChecked(True)
    reiter._takt.stop()
    reiter._neu_zeichnen()
    fehler += check("Kuestenhaken veraendert das Gelaende",
                    not np.array_equal(H, reiter._letztes_feld))

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("der Regionsreiter ist in Ordnung")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
