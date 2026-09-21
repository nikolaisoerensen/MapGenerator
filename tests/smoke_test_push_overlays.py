"""
Path: tests/smoke_test_push_overlays.py

Prueft BaseMapTab._push_overlays() und das Overlay-Register aus
gui/tabs/base_tab.py (Ticket #8, docs/SPEC_OVERLAYS.md) - den Nachfolger der
alten `hasattr`-Weichen, an denen viermal derselbe Fehler passierte (siehe
CLAUDE.md "STEHENDE REGEL: was in 2D sichtbar ist, gehoert auch in 3D"):
ein Overlay-Haken wirkte in 2D, aber lautlos nicht in 3D.

WAS HIER GEPRUEFT WIRD (docs/SPEC_OVERLAYS.md, Testing Decisions):

1. Das Register, gegen mitschreibende Attrappen-Anzeigen (kein Qt, kein
   OpenGL) - je angemeldetem Overlay erzeugt sichtbar=True einen Aufruf in
   BEIDEN Attrappen, sichtbar=False raeumt in beiden ab, ein unbekannter
   Name wirft einen Fehler statt still nichts zu tun.
2. `current_view` kommt im neuen Code nicht vor - ein Quelltexttest ueber
   _push_overlays() und die Register-Adapter. Klingt grob, trifft aber genau
   die Ursache aller vier Vorfaelle: jeder davon war eine `current_view`-
   Weiche, die den 3D-Zweig nie erreichte.

NICHT geprueft (siehe SPEC Punkt "Was sich nur am laufenden Programm pruefen
laesst"): ob die RGBA-Textur im 3D an der richtigen Stelle sitzt und lesbar
aussieht - das sieht nur der Nutzer, siehe docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md.
"""
import ast
import inspect
import sys
import textwrap

import numpy as np

sys.path.insert(0, ".")

from gui.tabs.base_tab import (
    BaseMapTab, Overlay, _OVERLAY_REGISTER, _siedlungen_2d, _siedlungen_3d,
    _fluesse_zeichnen,
)


class _FakeWrapper:
    """Minimaler Ersatz fuer DisplayWrapper (gui/widgets/widgets.py) -
    _push_overlays() greift nur auf `.display` zu, alles andere braucht es
    nicht."""

    def __init__(self, display):
        self.display = display


class _FakeSelf:
    """Steht statt eines echten BaseMapTab - _push_overlays() liest nur
    self.map_display_2d/self.map_display_3d, alles andere (QWidget-Aufbau,
    Manager, ...) waere fuer diesen Test unnoetiger Ballast."""

    def __init__(self, display_2d=None, display_3d=None):
        self.map_display_2d = _FakeWrapper(display_2d) if display_2d else None
        self.map_display_3d = _FakeWrapper(display_3d) if display_3d else None


class FakeDisplay2D:
    """Attrappe fuer MapDisplay2D - protokolliert jeden Aufruf statt zu
    zeichnen. Hat ABSICHTLICH kein `clear_river_overlay` (das gibt es auf der
    echten Klasse auch nicht, siehe _fluesse_zeichnen()-Docstring: 2D baut
    sein Bild bei jedem update_display() ohnehin neu auf)."""

    def __init__(self):
        self.calls = []

    def overlay_settlements(self, settlements, landmarks, roadsites):
        self.calls.append(("overlay_settlements", settlements, landmarks, roadsites))

    def overlay_river_generations(self, generation_map, zeige_mikro=False):
        self.calls.append(("overlay_river_generations", generation_map, zeige_mikro))


class FakeDisplay3D:
    """Attrappe fuer MapDisplay3DWidget - `heightmap` steht wie im echten
    Widget bereits, weil update_heightmap() (Teil von
    _push_data_to_current_display()) laut Vorbild-Reihenfolge vor jedem
    Overlay-Push laeuft."""

    def __init__(self, heightmap):
        self.heightmap = heightmap
        self.calls = []
        self.overlay_data = {}
        self.layer_visibility = {}

    def update_overlay_data(self, kategorie, name, rgba):
        self.calls.append(("update_overlay_data", kategorie, name))
        self.overlay_data[(kategorie, name)] = rgba

    def set_layer_visibility(self, kategorie, name, sichtbar):
        self.calls.append(("set_layer_visibility", kategorie, name, sichtbar))
        self.layer_visibility[(kategorie, name)] = sichtbar

    def overlay_river_generations(self, generation_map, zeige_mikro=False):
        self.calls.append(("overlay_river_generations", generation_map, zeige_mikro))

    def clear_river_overlay(self):
        self.calls.append(("clear_river_overlay",))


# Echte Kartengroessen (docs/SPEC_OVERLAYS.md Testing Decisions Punkt 5, die
# Lehre aus dem adaptiven 3D-Netz: eine ausgedachte Groesse testet eine
# ausgedachte Situation).
_ECHTE_KARTENGROESSE = 256


def run_siedlungen_sichtbar():
    heightmap = np.ones((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE), dtype=np.float32)
    d2, d3 = FakeDisplay2D(), FakeDisplay3D(heightmap)
    fake_self = _FakeSelf(d2, d3)

    settlements = [(10, 20)]
    BaseMapTab._push_overlays(fake_self, [
        Overlay("siedlungen", sichtbar=True, daten=(settlements, [], [])),
    ])

    ok = True
    if not any(c[0] == "overlay_settlements" for c in d2.calls):
        print("[FAIL] siedlungen sichtbar=True: 2D bekommt keinen overlay_settlements()-Aufruf")
        ok = False
    if not any(c[0] == "update_overlay_data" for c in d3.calls):
        print("[FAIL] siedlungen sichtbar=True: 3D bekommt keinen update_overlay_data()-Aufruf")
        ok = False
    if d3.layer_visibility.get(("settlement", "uebersicht")) is not True:
        print("[FAIL] siedlungen sichtbar=True: 3D-Layer 'uebersicht' steht nicht auf sichtbar")
        ok = False
    if ok:
        print("[OK] siedlungen sichtbar=True erreicht 2D UND 3D")
    return ok


def run_siedlungen_unsichtbar_raeumt_3d_ab():
    heightmap = np.ones((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE), dtype=np.float32)
    d2, d3 = FakeDisplay2D(), FakeDisplay3D(heightmap)
    fake_self = _FakeSelf(d2, d3)

    BaseMapTab._push_overlays(fake_self, [
        Overlay("siedlungen", sichtbar=False, daten=([], [], [])),
    ])

    ok = True
    if d2.calls:
        print(f"[FAIL] siedlungen sichtbar=False: 2D haette nichts zeichnen sollen, bekam {d2.calls}")
        ok = False
    if d3.layer_visibility.get(("settlement", "uebersicht")) is not False:
        print("[FAIL] siedlungen sichtbar=False: 3D-Layer 'uebersicht' wurde nicht abgeraeumt")
        ok = False
    if any(c[0] == "update_overlay_data" for c in d3.calls):
        print("[FAIL] siedlungen sichtbar=False: 3D haette keine neue Textur bauen sollen")
        ok = False
    if ok:
        print("[OK] siedlungen sichtbar=False raeumt 3D ab, ohne 2D anzufassen")
    return ok


def run_fluesse_sichtbar():
    d2, d3 = FakeDisplay2D(), FakeDisplay3D(np.ones((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE)))
    fake_self = _FakeSelf(d2, d3)

    generation_map = np.zeros((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE), dtype=np.int32)
    BaseMapTab._push_overlays(fake_self, [
        Overlay("fluesse", sichtbar=True, daten=generation_map),
    ])

    ok = True
    for name, d in (("2D", d2), ("3D", d3)):
        if not any(c[0] == "overlay_river_generations" for c in d.calls):
            print(f"[FAIL] fluesse sichtbar=True: {name} bekommt keinen overlay_river_generations()-Aufruf")
            ok = False
    if ok:
        print("[OK] fluesse sichtbar=True erreicht 2D UND 3D mit derselben Quelle")
    return ok


def run_fluesse_unsichtbar_raeumt_3d_ab():
    d2, d3 = FakeDisplay2D(), FakeDisplay3D(np.ones((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE)))
    fake_self = _FakeSelf(d2, d3)

    BaseMapTab._push_overlays(fake_self, [
        Overlay("fluesse", sichtbar=False, daten=None),
    ])

    ok = True
    if d2.calls:
        print(f"[FAIL] fluesse sichtbar=False: 2D haette nichts tun sollen, bekam {d2.calls}")
        ok = False
    if not any(c[0] == "clear_river_overlay" for c in d3.calls):
        print("[FAIL] fluesse sichtbar=False: 3D raeumt die Fluss-Textur nicht ab")
        ok = False
    if ok:
        print("[OK] fluesse sichtbar=False raeumt 3D ab (2D hat kein clear_river_overlay, "
              "zeichnet stattdessen bei jedem update_display() neu)")
    return ok


def run_unbekannter_name_wirft():
    d2, d3 = FakeDisplay2D(), FakeDisplay3D(np.ones((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE)))
    fake_self = _FakeSelf(d2, d3)

    try:
        BaseMapTab._push_overlays(fake_self, [Overlay("nie_registriert", sichtbar=True)])
    except ValueError:
        print("[OK] unbekannter Overlay-Name wirft ValueError statt still nichts zu tun")
        return True
    print("[FAIL] unbekannter Overlay-Name haette einen Fehler werfen muessen")
    return False


def run_register_vollstaendig():
    """Jeder Registereintrag braucht BEIDE Seiten - ein Eintrag mit nur einem
    Adapter waere genau der Fall, den docs/SPEC_OVERLAYS.md Punkt 5 als
    Fehler beim Start verlangt, nicht als spaeteres Nichtstun."""
    ok = True
    for name, eintrag in _OVERLAY_REGISTER.items():
        if "2d" not in eintrag or "3d" not in eintrag:
            print(f"[FAIL] Registereintrag '{name}' hat nicht beide Adapter: {eintrag.keys()}")
            ok = False
    if ok:
        print(f"OK - alle {len(_OVERLAY_REGISTER)} Registereintraege haben 2D- UND 3D-Adapter "
              f"({', '.join(sorted(_OVERLAY_REGISTER))})")
    return ok


def _ohne_docstring(quelltext: str) -> str:
    """Entfernt den Docstring-Kopf einer Funktion vor der current_view-Pruefung.
    Die Docstring von _push_overlays() erklaert ABSICHTLICH in Prosa, warum
    current_view im Code fehlt (siehe base_tab.py) - das Wort taucht dort also
    zurecht auf. Geprueft werden soll nur der eigentliche Funktionskoerper."""
    baum = ast.parse(textwrap.dedent(quelltext))
    funktion = baum.body[0]
    docstring = ast.get_docstring(funktion)
    koerper = funktion.body[1:] if docstring is not None else funktion.body
    if not koerper:
        return ""
    return "\n".join(ast.get_source_segment(quelltext, knoten) or "" for knoten in koerper)


def run_current_view_kommt_nicht_vor():
    """Quelltexttest (docs/SPEC_OVERLAYS.md Testing Decisions Punkt 2): trifft
    grob, aber genau die Ursache aller vier Vorfaelle aus CLAUDE.md - jeder
    davon war eine current_view-Weiche, die den 3D-Zweig nie erreichte."""
    quellen = {
        "_push_overlays": inspect.getsource(BaseMapTab._push_overlays),
        "_siedlungen_2d": inspect.getsource(_siedlungen_2d),
        "_siedlungen_3d": inspect.getsource(_siedlungen_3d),
        "_fluesse_zeichnen": inspect.getsource(_fluesse_zeichnen),
    }
    treffer = [name for name, quelle in quellen.items() if "current_view" in _ohne_docstring(quelle)]
    if treffer:
        print(f"[FAIL] 'current_view' kommt im Funktionskoerper vor bei: {', '.join(treffer)}")
        return False
    print("[OK] 'current_view' kommt im Funktionskoerper von _push_overlays()/dem Register nirgends vor")
    return True


if __name__ == "__main__":
    ergebnisse = {
        "siedlungen_sichtbar": run_siedlungen_sichtbar(),
        "siedlungen_unsichtbar_raeumt_3d_ab": run_siedlungen_unsichtbar_raeumt_3d_ab(),
        "fluesse_sichtbar": run_fluesse_sichtbar(),
        "fluesse_unsichtbar_raeumt_3d_ab": run_fluesse_unsichtbar_raeumt_3d_ab(),
        "unbekannter_name_wirft": run_unbekannter_name_wirft(),
        "register_vollstaendig": run_register_vollstaendig(),
        "current_view_kommt_nicht_vor": run_current_view_kommt_nicht_vor(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
