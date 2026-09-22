"""
Path: tests/smoke_test_river_tab_anzeigemodi.py

Die drei verbliebenen Ansichten des Fluss-Reiters, nach dem Aufraeumen
2026-09-23.

ANLASS: die vierte Ansicht "Flussnetz (Generationen)" (interner Schluessel
"rivers") und die Checkbox "Baeche (Mikro)" (`self.mikro_checkbox`) wurden
aus gui/tabs/river_tab.py entfernt - nicht mehr gebraucht, seit
"Wassermenge" die Leitansicht ist (Begruendung im Modul-Docstring dort).
Die gleiche Generationsfaerbung lebt unveraendert als eigenstaendiges
Overlay im Biome-Reiter weiter (tests/smoke_test_biome_overlays_3d.py).

WAS HIER GEPRUEFT WIRD:

  1. Es gibt genau drei Radio-Knoepfe, keiner davon heisst "Flussnetz
     (Generationen)".
  2. `mikro_checkbox` existiert nicht mehr als Attribut.
  3. Der interne Schluessel "rivers" taucht in `_display_modes_by_id`
     nirgends mehr auf.
  4. Jede der drei verbliebenen Ansichten (Gelaende, Wassermenge, Ordnung)
     laesst sich ueber `update_display_mode()` aufbauen, ohne eine
     Ausnahme zu werfen - mit echten (wenn auch kleinen) Daten aus dem
     DataLODManager, nicht mit leeren Attrappen (CLAUDE.md: "Tests mit den
     ECHTEN Eingabegroessen bauen").

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_river_tab_anzeigemodi.py
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

MAP_SIZE = 128


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    from managers.data_lod_manager import DataLODManager
    from managers.navigation_manager import NavigationManager
    from managers.parameter_manager import ParameterManager
    from gui.tabs.river_tab import RiverTab

    dlm = DataLODManager()
    for schluessel in ("heightmap", "river_water", "river_order"):
        daten = np.random.default_rng(20260923).random(
            (MAP_SIZE, MAP_SIZE)).astype(np.float64)
        dlm.set_terrain_data_lod(schluessel, daten, lod_level=1, parameters={})

    reiter = RiverTab(
        data_lod_manager=dlm, parameter_manager=ParameterManager(),
        navigation_manager=NavigationManager(data_lod_manager=dlm),
        shader_manager=None, generation_orchestrator=None)

    # --- 1. Genau drei Radio-Knoepfe, kein "Flussnetz (Generationen)" ------
    knoepfe = reiter.display_mode_group.buttons()
    beschriftungen = [k.text() for k in knoepfe]
    fehler += check("es gibt genau drei Ansichts-Knoepfe",
                    len(knoepfe) == 3, f"gefunden: {beschriftungen}")
    fehler += check('"Flussnetz (Generationen)" ist nicht mehr dabei',
                    "Flussnetz (Generationen)" not in beschriftungen)

    # --- 2. Die Checkbox ist weg ---------------------------------------------
    fehler += check("mikro_checkbox existiert nicht mehr",
                    not hasattr(reiter, "mikro_checkbox"))

    # --- 3. Der interne Schluessel "rivers" ist nirgends mehr eingetragen ---
    fehler += check('der Modus-Schluessel "rivers" ist entfernt',
                    "rivers" not in reiter._display_modes_by_id.values(),
                    f"vorhanden: {list(reiter._display_modes_by_id.values())}")

    # --- 4. Jede verbliebene Ansicht baut sich ohne Ausnahme auf -----------
    for nummer, schluessel in sorted(reiter._display_modes_by_id.items()):
        reiter.current_display_mode = schluessel
        try:
            reiter.update_display_mode()
            ok = True
            zusatz = ""
        except Exception as exc:                          # noqa: BLE001
            ok = False
            zusatz = str(exc)[:160]
        fehler += check(f"Ansicht '{schluessel}' baut sich ohne Ausnahme auf",
                        ok, zusatz)

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("die drei verbliebenen Ansichten des Fluss-Reiters sind in Ordnung")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
