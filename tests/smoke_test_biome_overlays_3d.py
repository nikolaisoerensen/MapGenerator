"""
Path: tests/smoke_test_biome_overlays_3d.py

Prueft, dass BiomeTab.apply_overlays() Settlements und Flussnetz auch in
der 3D-Ansicht auf das Display bringt.

ANLASS (Ticket #5, github.com/nikolaisoerensen/MapGenerator/issues/5):
`apply_overlays()` stieg bis zum 2026-09-16 in der ERSTEN Zeile aus, wenn
die Ansicht nicht 2D war (`if not current_display or self.current_view !=
"2d": return`). Die 3D-Zweige darunter - fuer Settlements per
`update_overlay_data("settlement", "uebersicht", ...)` und fuer das
Flussnetz per `overlay_river_generations()` - wurden am 2026-08-25
ausdruecklich als Behebung eingebaut, konnten aber wegen dieses fruehen
Ausstiegs NIE ausgefuehrt werden. Vorfall 4 derselben Fehlerklasse (siehe
CLAUDE.md, Abschnitt "was in 2D sichtbar ist, gehoert auch in 3D") und der
dritte, der als behoben verbucht wurde, ohne es zu sein.

WAS HIER GEPRUEFT WIRD und was nicht: dass die Methode bei `current_view
== "3d"` tatsaechlich bis zu den 3D-Aufrufen durchlaeuft und sie mit den
richtigen Argumenten aufruft - das ist reine Python-Logik und headless
pruefbar. Ob die Textur am Bildschirm ERSCHEINT, kann dieser Test NICHT
sagen (CLAUDE.md - OpenGL braucht ein sichtbares Fenster); dafuer steht ein
Eintrag in docs/PRUEFLISTE_LIVE.md.

WIE GEPRUEFT WIRD: `apply_overlays` ist eine gewoehnliche Methode, die nur
ueber `self`-Attribute auf Display, Checkboxes und DataLODManager zugreift.
Statt eines vollen `BiomeTab`-Widgets (das einen ParameterManager, einen
GenerationOrchestrator usw. braucht) genuegt ein duck-typed Fake-Objekt mit
genau diesen Attributen - die UNGEBUNDENE Methode `BiomeTab.apply_overlays`
wird direkt darauf aufgerufen. Das prueft exakt denselben Code, der auch im
echten Tab laeuft.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_biome_overlays_3d.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui.tabs.biome_tab import BiomeTab


class _FakeCheckbox:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _Fake3DDisplay:
    """Nur die 3D-API - KEIN `overlay_settlements`, damit die hasattr-Weiche
    in apply_overlays() zwingend den 3D-Zweig nimmt, genau wie bei der
    echten MapDisplay3DWidget."""

    def __init__(self):
        self.overlay_data_calls = []
        self.visibility_calls = []
        self.river_calls = []
        self.river_cleared = False

    def update_overlay_data(self, tab_type, layer_name, data):
        self.overlay_data_calls.append((tab_type, layer_name, data))

    def set_layer_visibility(self, tab_type, layer_name, visible):
        self.visibility_calls.append((tab_type, layer_name, visible))

    def overlay_river_generations(self, generation, zeige_mikro=False):
        self.river_calls.append((generation, zeige_mikro))

    def clear_river_overlay(self):
        self.river_cleared = True


class _FakeDisplayWrapper:
    def __init__(self, display):
        self.display = display


class _FakeDataLODManager:
    def __init__(self, mit_daten):
        n = 8
        self._settlements = [{"id": 1}] if mit_daten else []
        self._heightmap = np.full((n, n), 50.0)
        self._river_generation = np.ones((n, n)) if mit_daten else None

    def get_settlement_data(self, key):
        return {
            "settlement_list": self._settlements,
            "landmark_list": [],
            "roadsite_list": [],
        }.get(key)

    def get_terrain_data_combined(self, key):
        return self._heightmap if key == "heightmap" else None

    def get_terrain_data(self, key):
        return self._river_generation if key == "river_generation" else None


class _FakeBiomeTab:
    """Duck-typed Stand-in fuer BiomeTab - siehe Moduldocstring."""

    def __init__(self, current_view, settlements_checked, rivers_checked, mit_daten=True):
        self.current_view = current_view
        self._display = _Fake3DDisplay() if current_view == "3d" else None
        self.data_lod_manager = _FakeDataLODManager(mit_daten)
        self.settlements_overlay = _FakeCheckbox(settlements_checked)
        self.rivers_overlay = _FakeCheckbox(rivers_checked)

    def get_current_display(self):
        if self._display is None:
            return None
        return _FakeDisplayWrapper(self._display)


def ausfall_kommt_nicht_zurueck():
    """
    1. DIE EIGENTLICHE ZUSICHERUNG: bei current_view == "3d" muss
    apply_overlays() bis zu den 3D-Aufrufen durchlaufen, statt in der
    ersten Zeile auszusteigen.
    """
    fehler = []
    tab = _FakeBiomeTab("3d", settlements_checked=True, rivers_checked=True)
    BiomeTab.apply_overlays(tab)

    settlement_gesetzt = any(
        call == ("settlement", "uebersicht", call[2]) and call[2] is not None
        for call in tab._display.overlay_data_calls
    )
    print(f"[{'OK' if settlement_gesetzt else 'FEHLER'}] Settlements: "
          f"update_overlay_data('settlement', 'uebersicht', ...) aufgerufen "
          f"({len(tab._display.overlay_data_calls)} Aufruf(e))")
    if not settlement_gesetzt:
        fehler.append("Settlements: update_overlay_data nie aufgerufen - "
                       "apply_overlays() ist vermutlich frueh ausgestiegen")

    settlement_sichtbar = ("settlement", "uebersicht", True) in tab._display.visibility_calls
    print(f"[{'OK' if settlement_sichtbar else 'FEHLER'}] Settlements: "
          f"set_layer_visibility('settlement', 'uebersicht', True) aufgerufen")
    if not settlement_sichtbar:
        fehler.append("Settlements: Sichtbarkeit nie auf True gesetzt")

    fluss_gezeichnet = len(tab._display.river_calls) == 1
    print(f"[{'OK' if fluss_gezeichnet else 'FEHLER'}] Flussnetz: "
          f"overlay_river_generations() aufgerufen "
          f"({len(tab._display.river_calls)} Aufruf(e))")
    if not fluss_gezeichnet:
        fehler.append("Flussnetz: overlay_river_generations nie aufgerufen")

    return fehler


def abwaehlen_entfernt_die_textur():
    """
    2. Ein deaktivierter Haken muss die 3D-Textur ABSCHALTEN, nicht nur
    kein neues Bild liefern - sonst bliebe die alte Textur liegen.
    """
    fehler = []
    tab = _FakeBiomeTab("3d", settlements_checked=False, rivers_checked=False)
    BiomeTab.apply_overlays(tab)

    settlement_ausgeblendet = ("settlement", "uebersicht", False) in tab._display.visibility_calls
    print(f"[{'OK' if settlement_ausgeblendet else 'FEHLER'}] Settlements "
          f"abgewaehlt: set_layer_visibility(..., False) aufgerufen")
    if not settlement_ausgeblendet:
        fehler.append("Settlements: Sichtbarkeit beim Abwaehlen nicht auf "
                       "False gesetzt")

    fluss_entfernt = tab._display.river_cleared
    print(f"[{'OK' if fluss_entfernt else 'FEHLER'}] Flussnetz abgewaehlt: "
          f"clear_river_overlay() aufgerufen")
    if not fluss_entfernt:
        fehler.append("Flussnetz: clear_river_overlay() beim Abwaehlen "
                       "nicht aufgerufen")

    return fehler


def zweite_ansicht_2d_bleibt_unveraendert():
    """
    3. Die 2D-Ansicht darf durch die Behebung nicht beruehrt werden - dort
    gibt es kein `_Fake3DDisplay`, sondern ein Objekt mit `overlay_settlements`,
    das die hasattr-Weiche in den 2D-Zweig lenkt. Hier reicht die Zusicherung,
    dass apply_overlays() bei `current_display is None` weiterhin sauber
    zurueckkehrt (kein Absturz) - das eigentliche 2D-Zeichnen ist bereits
    durch tests/smoke_test_display_2d.py abgedeckt.
    """
    fehler = []
    tab = _FakeBiomeTab("2d", settlements_checked=True, rivers_checked=True)
    tab._display = None
    try:
        BiomeTab.apply_overlays(tab)
        print("[OK] current_display is None: kein Absturz")
    except Exception as e:                                       # noqa: BLE001
        print(f"[FEHLER] current_display is None: Absturz - {e}")
        fehler.append(f"Absturz bei fehlendem Display: {e}")
    return fehler


def lauf():
    gruppen = [
        ("ausfall_kommt_nicht_zurueck", ausfall_kommt_nicht_zurueck),
        ("abwaehlen_entfernt_die_textur", abwaehlen_entfernt_die_textur),
        ("zweite_ansicht_2d_bleibt_unveraendert", zweite_ansicht_2d_bleibt_unveraendert),
    ]
    ergebnis, alle = {}, []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        f = fn()
        ergebnis[name] = not f
        alle.extend(f)

    print("\n=== SUMMARY ===")
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print("\nNICHT geprueft: ob die Textur am Bildschirm erscheint.")
    print("OpenGL braucht dafuer ein sichtbares Fenster (CLAUDE.md).")
    if alle:
        print(f"\nNICHT IN ORDNUNG - {len(alle)} Befunde:")
        for f in alle:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
