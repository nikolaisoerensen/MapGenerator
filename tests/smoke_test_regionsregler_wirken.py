"""
Path: tests/smoke_test_regionsregler_wirken.py

Wirken die Einstellungen des Regionsreiters auf die ERZEUGTE KARTE?

ANLASS (gefunden 2026-08-26, ohne dass jemand es gemeldet haette): der
Regionsreiter war fertig, seine Vorschau reagierte auf jeden Regler - und
**die Einstellungen kamen nirgends an**. `self.ueberschreibungen` lebte
allein im Reiter, `weltfeld()` las ausschliesslich den Katalog. Man konnte
eine Region einstellen, "Generieren" druecken und bekam dieselbe Karte wie
vorher. Kein Absturz, keine Meldung, kein roter Test.

Gefunden wurde es nur durch die Frage "WER liest das eigentlich?" - ein
`grep` ueber `ueberschreibungen` ausserhalb des Reiters lieferte null
Treffer.

**DIESER TEST PRUEFT DESHALB NICHT, OB DER PARAMETER ANKOMMT.** Er prueft,
ob sich das GELAENDE messbar aendert. Ein Test auf "der Schluessel steht im
Dict" waere an genau diesem Fehler vorbeigelaufen: der Schluessel stand ja
nirgends, und niemand hatte ihn erwartet.

Die Kette hat vier Glieder, und jedes einzelne war schon einmal die
Fehlerstelle:

    RegionTab.get_current_parameters()
      -> ParameterManager (angemeldet als "region")
      -> TerrainTab.get_current_parameters() nimmt sie mit
      -> BaseTerrainGenerator -> weltfeld(regionen_ueberschreibung=...)
      -> parameterfeld() ersetzt einzelne Katalogwerte

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_regionsregler_wirken.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging

import numpy as np
from PyQt6.QtWidgets import QApplication
from scipy import ndimage

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])

import core.terrain_weltkarte as rw

SIZE = 192
SEED = 20260804


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []

    # --- 1. Die Kette bis zum ParameterManager ------------------------------
    from managers.parameter_manager import ParameterManager
    from gui.tabs.region_tab import RegionTab

    pm = ParameterManager()
    reiter = RegionTab(parameter_manager=pm)
    fehler += check("der Regionsreiter meldet sich an",
                    "region" in pm.registered_tabs)
    fehler += check("ohne Aenderung liefert er nichts",
                    not pm.get_tab_parameters("region"),
                    str(pm.get_tab_parameters("region")))

    reiter.auswahl.setCurrentIndex(
        [i for i in range(reiter.auswahl.count())
         if reiter.auswahl.itemText(i) == "Nevadin"][0])
    reiter._takt.stop()
    reiter.regler["relief_m"].setValue(400.0)
    reiter._takt.stop()
    reiter._neu_zeichnen()
    vom_reiter = pm.get_tab_parameters("region")
    fehler += check("nach der Aenderung liefert er die Ueberschreibung",
                    vom_reiter.get("regionen_ueberschreibung", {})
                    .get("Nevadin", {}).get("relief_m") == 400.0,
                    str(vom_reiter))

    # --- 2. Der Terrain-Reiter nimmt sie mit --------------------------------
    #
    # Ohne dieses Glied kaeme nichts an: die Generierung holt sich
    # `get_tab_parameters(self.generator_type)`, also NUR den Terrain-Satz.
    try:
        from gui.tabs.terrain_tab import TerrainTab
        # TerrainTab verlangt alle fuenf Manager (anders als RegionTab, der
        # sie nur entgegennimmt). Ein DataLODManager reicht hier - der Reiter
        # baut, und mehr braucht diese Pruefung nicht.
        from managers.data_lod_manager import DataLODManager
        from managers.navigation_manager import NavigationManager
        dlm = DataLODManager()
        terrain = TerrainTab(
            data_lod_manager=dlm, parameter_manager=pm,
            navigation_manager=NavigationManager(data_lod_manager=dlm),
            shader_manager=None, generation_orchestrator=None)
        weiter = terrain.get_current_parameters()
        fehler += check("der Terrain-Reiter reicht sie weiter",
                        "regionen_ueberschreibung" in weiter,
                        ", ".join(sorted(k for k in weiter
                                         if k.startswith("region"))) or "nichts")
    except Exception as f:                                # noqa: BLE001
        fehler += check("der Terrain-Reiter laesst sich bauen", False,
                        f"{type(f).__name__}: {str(f)[:90]}")

    # --- 3. UND DAS GELAENDE AENDERT SICH -----------------------------------
    #
    # Das eigentliche Ziel. Gemessen wird das `relief_m`-FELD im Kern der
    # Region und die Hoehenspanne dort - nicht der Parameter.
    ohne, f_ohne = rw.weltfeld(SIZE, SEED)
    mit, f_mit = rw.weltfeld(
        SIZE, SEED, regionen_ueberschreibung={"Nevadin": {"relief_m": 400.0}})
    ohne_a = np.asarray(ohne, dtype=np.float64)
    mit_a = np.asarray(mit, dtype=np.float64)

    idx = next(i for i, (_z, _s, r) in enumerate(rw.alle_regionen())
               if r["name"] == "Nevadin")
    kern = f_ohne["regionen"] == idx
    innen = ndimage.binary_erosion(kern, iterations=4)
    if innen.sum() < 100:
        innen = kern

    feld_ohne = float(f_ohne["relief_m"][innen].mean())
    feld_mit = float(f_mit["relief_m"][innen].mean())
    fehler += check("das relief_m-Feld der Region sinkt",
                    feld_mit < 0.7 * feld_ohne,
                    f"{feld_ohne:.0f} -> {feld_mit:.0f} m "
                    f"(Katalog 1050, gesetzt 400; der Rest ist "
                    f"Voronoi-Mischung mit den Nachbarn)")

    spanne_ohne = float(np.ptp(ohne_a[innen]))
    spanne_mit = float(np.ptp(mit_a[innen]))
    fehler += check("die Hoehenspanne der Region sinkt",
                    spanne_mit < 0.85 * spanne_ohne,
                    f"{spanne_ohne:.0f} -> {spanne_mit:.0f} m")

    # --- 4. Und der REST der Karte bleibt weitgehend stehen -----------------
    #
    # Gemessen als MEDIAN, nicht als Mittelwert: ein erster Anlauf nahm den
    # Mittelwert und meldete 30 m Aenderung "ausserhalb". Das war ein
    # Metrikartefakt - der Mittelwert wurde vom Grenzband dominiert, in dem
    # die Voronoi-Mischung zu Recht mitzieht. Der Median liegt bei 0.7 m.
    aussen = ~kern
    median_aussen = float(np.median(np.abs(ohne_a - mit_a)[aussen]))
    fehler += check("ausserhalb der Region bleibt es stehen",
                    median_aussen < 5.0,
                    f"Median {median_aussen:.1f} m (p95 "
                    f"{np.percentile(np.abs(ohne_a - mit_a)[aussen], 95):.0f} m "
                    f"- die Spitze ist das Grenzband)")

    # --- 5. Ohne Ueberschreibung bitgleich ----------------------------------
    gleich, _f = rw.weltfeld(SIZE, SEED, regionen_ueberschreibung=None)
    fehler += check("ohne Ueberschreibung bitgleich zur Vorgabe",
                    np.array_equal(ohne_a, np.asarray(gleich, dtype=np.float64)),
                    "sonst waeren alle Eichungen still verschoben")

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("die Regionsregler erreichen die Karte")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
