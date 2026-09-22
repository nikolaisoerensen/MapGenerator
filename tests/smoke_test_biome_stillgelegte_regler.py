"""
Path: tests/smoke_test_biome_stillgelegte_regler.py

Sind `alpine_level`/`snow_level` im Biome-Reiter gesperrt - und die anderen
sechs Biome-Regler weiterhin frei?

ANLASS. Seit dem Julitemperatur-Umbau vom 2026-08-07 lesen
`_calculate_alpine_level_probabilities()`/`_calculate_snow_level_probabilities()`
(core/biome_generator.py) die festen Modulkonstanten BAUMGRENZE_JULI_C/
FIRN_JULI_C statt `self.alpine_level`/`self.snow_level`. `alpine_level` wird
seither nur noch von der toten Vergleichsmethode
`_alt_alpine_level_probabilities()` gelesen, `snow_level` von keinem Code
mehr - unabhaengig vom Aufrufpfad, weil der GPU-Handler
`shader_manager.request_biome_classification` im Repo gar nicht existiert
und jeder Pfad auf denselben toten CPU-Code zurueckfaellt.

Dieser Test prueft NICHT erneut, ob die Werte tot sind (das ist bereits
durch Lesen des Codes belegt, siehe gui/config/value_default.stillgelegte_regler()).
Er prueft die MECHANIK: dass der bestehende Sperr-Mechanismus
(BaseMapTab._stillgelegte_regler_sperren(), gebaut fuer Terrain/Erosion)
beim Bauen des Biome-Reiters automatisch auch diese zwei Regler sperrt -
und dass dabei kein lebendiger Regler versehentlich mitgesperrt wird.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_biome_stillgelegte_regler.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging

from PyQt6.QtWidgets import QApplication

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


GESPERRT_ERWARTET = ("alpine_level", "snow_level")
FREI_ERWARTET = ("biome_wetness_factor", "biome_temp_factor", "sea_level",
                  "bank_width", "cliff_slope", "edge_softness")


def lauf():
    fehler = []

    from gui.config.value_default import stillgelegte_regler
    gesperrt = stillgelegte_regler()

    # --- 1. Die Begruendungsliste selbst --------------------------------
    for schluessel in GESPERRT_ERWARTET:
        grund = gesperrt.get(schluessel)
        fehler += check(f"'{schluessel}' traegt eine Begruendung in "
                        f"stillgelegte_regler()",
                        isinstance(grund, str) and len(grund) > 30,
                        (grund or "keine")[:70])
    for schluessel in FREI_ERWARTET:
        fehler += check(f"'{schluessel}' steht NICHT in stillgelegte_regler()",
                        schluessel not in gesperrt,
                        str(gesperrt.get(schluessel)))

    # --- 2. Der echte Reiter, echter Sperr-Mechanismus -------------------
    from managers.data_lod_manager import DataLODManager
    from managers.navigation_manager import NavigationManager
    from managers.parameter_manager import ParameterManager
    from gui.tabs.biome_tab import BiomeTab

    dlm = DataLODManager()
    pm = ParameterManager()
    tab = BiomeTab(
        data_lod_manager=dlm, parameter_manager=pm,
        navigation_manager=NavigationManager(data_lod_manager=dlm),
        shader_manager=None, generation_orchestrator=None)

    regler = getattr(tab, "parameter_sliders", {})
    fehler += check("der Biome-Reiter baut alle 8 Regler",
                    len(regler) == 8, ", ".join(sorted(regler)))

    for schluessel in GESPERRT_ERWARTET:
        widget = regler.get(schluessel)
        fehler += check(f"Regler '{schluessel}' existiert im Reiter",
                        widget is not None)
        if widget is None:
            continue
        fehler += check(f"Regler '{schluessel}' ist deaktiviert (setEnabled)",
                        widget.isEnabled() is False)
        fehler += check(f"Regler '{schluessel}' traegt einen Tooltip-Hinweis",
                        bool(widget.toolTip()), widget.toolTip()[:70])
        fehler += check(f"Regler '{schluessel}' kennt seinen Stilllegungsgrund",
                        bool(getattr(widget, "stilllegungsgrund", None)))

    for schluessel in FREI_ERWARTET:
        widget = regler.get(schluessel)
        fehler += check(f"Regler '{schluessel}' existiert im Reiter",
                        widget is not None)
        if widget is None:
            continue
        fehler += check(f"Regler '{schluessel}' bleibt aktiv",
                        widget.isEnabled() is True)

    return fehler


if __name__ == "__main__":
    fehler = lauf()
    if fehler:
        print(f"\n{len(fehler)} Fehlschlag(e).")
        sys.exit(1)
    print("\nAlle Pruefungen bestanden.")
