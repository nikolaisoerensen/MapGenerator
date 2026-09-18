"""
Path: tests/smoke_test_fluss_vorschau.py

Die Live-Vorschau des Flussreiters (docs/AUFRAEUMPLAN.md 4.10).

NUTZERENTWURF 2026-08-26: *"dann kommt flussnetzwerke und auch hier sollte
eine live sicht moeglich sein."*

WAS GEPRUEFT WIRD:

  1. Sie ist AUS, solange niemand sie einschaltet. Ein Reiter, der beim
     Oeffnen anfaengt zu rechnen, macht das Programm beim Start langsam -
     und die Vorschau kostet beim ersten Mal knapp drei Sekunden.
  2. Der Zwischenspeicher haelt. Die fuenf Regler dieses Reiters aendern
     das Grundgelaende NICHT, nur Netz und Taeler. Ohne Zwischenspeicher
     kostete jeder Reglerzug 1.95 statt 0.65 s - der ganze Sinn der
     Vorschau haenge daran.
  3. Ein Reglerzug aendert das Ergebnis wirklich. Eine Vorschau, die
     immer dasselbe zeigt, waere von einer kaputten nicht zu unterscheiden.
  4. Sie geht durch den GEWOEHNLICHEN Anzeigeweg (`_show_data` mit
     "heightmap") und damit ohne Sonderbehandlung durch 2D und 3D. Am
     selben Tag hat ein eigener Anzeigeweg schon einmal dazu gefuehrt, dass
     im 3D lautlos nichts passierte.

GEMESSEN 2026-08-26 (128 px):

    erste Vorschau (mit Grundgelaende)   2.92 s
    danach je Reglerzug                  0.71 - 0.75 s

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_fluss_vorschau.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging
import time

import numpy as np
from PyQt6.QtWidgets import QApplication

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])

# Wieviel ein Reglerzug hoechstens kosten darf, wenn das Grundgelaende
# schon steht. Gemessen 0.71-0.75 s.
REGLERZUG_MAX_S = 1.6


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    from managers.data_lod_manager import DataLODManager
    from managers.navigation_manager import NavigationManager
    from managers.parameter_manager import ParameterManager
    from gui.tabs.river_tab import RiverTab, VORSCHAU_PX

    dlm = DataLODManager()
    reiter = RiverTab(
        data_lod_manager=dlm, parameter_manager=ParameterManager(),
        navigation_manager=NavigationManager(data_lod_manager=dlm),
        shader_manager=None, generation_orchestrator=None)

    fehler += check("die Vorschau ist beim Oeffnen AUS",
                    not reiter.vorschau_an.isChecked(),
                    "sonst rechnete das Programm beim Start knapp 3 s")
    fehler += check("nichts wurde vorgerechnet",
                    reiter._vorschau_basis is None)

    # --- Erster Lauf: Grundgelaende plus Fluesse ----------------------------
    t0 = time.time()
    reiter._vorschau_rechnen()
    erste = time.time() - t0
    fehler += check("die erste Vorschau liefert ein Gelaende",
                    reiter._vorschau_basis is not None,
                    f"{erste:.2f} s")

    basis_vorher = reiter._vorschau_basis[0].copy()

    # --- Zweiter Lauf: das Grundgelaende muss BEHALTEN werden ---------------
    t0 = time.time()
    reiter._vorschau_rechnen()
    zweite = time.time() - t0
    fehler += check("das Grundgelaende wird behalten",
                    np.array_equal(basis_vorher, reiter._vorschau_basis[0]),
                    "sonst waere der Zwischenspeicher wirkungslos")
    fehler += check(f"ein Reglerzug bleibt unter {REGLERZUG_MAX_S} s",
                    zweite < REGLERZUG_MAX_S,
                    f"{zweite:.2f} s (erster Lauf {erste:.2f} s)")

    # --- Ein Reglerzug muss das Ergebnis aendern ----------------------------
    #
    # Geprueft wird das ERGEBNIS, nicht der Parameter - dieselbe Lehre wie
    # bei den Regionsreglern am selben Tag, wo der Parameter sauber gesetzt
    # war und trotzdem nirgends ankam.
    import core.terrain_weltkarte as rw
    from core.terrain_generator import BaseTerrainGenerator

    basis, felder = reiter._vorschau_basis

    def gelaende(breite):
        gen = BaseTerrainGenerator.__new__(BaseTerrainGenerator)
        gen.shader_manager = None
        gen.data_lod_manager = None
        gen.logger = logging.getLogger("vorschau")
        gen._current_parameters = {"river_valley_width": breite}
        H, _m, _o, _g, _w, _fg = gen._weltfluesse(
            basis.copy(), felder, VORSCHAU_PX, reiter._vorschau_seed)
        return np.asarray(H, dtype=np.float64)

    schmal, breit = gelaende(0.2), gelaende(2.0)
    unterschied = float(np.abs(schmal - breit).mean())
    fehler += check("ein Reglerzug aendert das Gelaende",
                    unterschied > 1.0,
                    f"Talbreite 0.2 gegen 2.0: {unterschied:.1f} m im Mittel")

    # --- Der gewoehnliche Anzeigeweg ---------------------------------------
    import inspect
    quelle = inspect.getsource(RiverTab._vorschau_rechnen)
    fehler += check("die Vorschau geht ueber _show_data(..., 'heightmap')",
                    '_show_data(' in quelle and '"heightmap"' in quelle,
                    "ein eigener Anzeigeweg waere im 3D lautlos wirkungslos")

    # --- Ausschalten raeumt auf --------------------------------------------
    reiter.vorschau_an.setChecked(True)
    reiter._vorschau_takt.stop()
    reiter.vorschau_an.setChecked(False)
    fehler += check("Ausschalten verwirft den Zwischenspeicher",
                    reiter._vorschau_basis is None)

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("die Flussvorschau ist in Ordnung")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
