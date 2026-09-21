"""
Path: tests/smoke_test_biome_overlays_3d.py

Prueft, dass BiomeTab.apply_overlays() Settlements und Flussnetz auch in
der 3D-Ansicht auf das Display bringt.

ANLASS (Ticket #5, github.com/nikolaisoerensen/MapGenerator/issues/5):
`apply_overlays()` stieg bis zum 2026-09-16 in der ERSTEN Zeile aus, wenn
die Ansicht nicht 2D war (`if not current_display or self.current_view !=
"2d": return`). Die 3D-Zweige darunter wurden am 2026-08-25 ausdruecklich
als Behebung eingebaut, konnten aber wegen dieses fruehen Ausstiegs NIE
ausgefuehrt werden. Vorfall 4 derselben Fehlerklasse (siehe CLAUDE.md,
Abschnitt "was in 2D sichtbar ist, gehoert auch in 3D").

NACHTRAG 2026-09-16 (Code-Review nacht/2026-09-16): Ticket #9 hat
apply_overlays() seither auf `self._push_overlays([...])` umgestellt (siehe
gui/tabs/base_tab.py) - es gibt kein `current_view`/`get_current_display()`
mehr, das Register schickt IMMER an 2D UND 3D gleichzeitig. Die Fake-
Attrappe unten (`_FakeBiomeTab`) bildete noch den ALTEN Vertrag nach und
stuerzte deshalb bei jedem Testlauf mit AttributeError ab, weil
BiomeTab.apply_overlays() ein `self._push_overlays` erwartet, das die
Attrappe nicht hatte - gefunden im Code-Review, hier auf den aktuellen
Vertrag nachgezogen (dieselbe echte `BaseMapTab._push_overlays()` wie in
tests/smoke_test_push_overlays.py, nicht neu nachgebaut).

WAS HIER GEPRUEFT WIRD und was nicht: dass die Methode aus den Checkbox-
und DataLODManager-Zustaenden die richtigen Overlay-Objekte baut und ueber
_push_overlays() TATSAECHLICH bei beiden Anzeigen ankommt - das ist reine
Python-Logik und headless pruefbar. Ob die Textur am Bildschirm ERSCHEINT,
kann dieser Test NICHT sagen (CLAUDE.md - OpenGL braucht ein sichtbares
Fenster); dafuer steht ein Eintrag in docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md. Der
generische Dispatch-Mechanismus selbst (Register, current_view kommt nicht
mehr vor) ist bereits durch tests/smoke_test_push_overlays.py abgedeckt -
dieser Test prueft zusaetzlich, dass BiomeTab.apply_overlays() ihn mit den
RICHTIGEN Daten aus Checkboxes/DataLODManager aufruft.

WIE GEPRUEFT WIRD: `apply_overlays` ist eine gewoehnliche Methode, die nur
ueber `self`-Attribute auf Displays, Checkboxes und DataLODManager
zugreift. Statt eines vollen `BiomeTab`-Widgets (das einen
ParameterManager, einen GenerationOrchestrator usw. braucht) genuegt ein
duck-typed Fake-Objekt mit genau diesen Attributen - die UNGEBUNDENE
Methode `BiomeTab.apply_overlays` wird direkt darauf aufgerufen, und
`_push_overlays` ist die ECHTE `BaseMapTab._push_overlays` (kein Fake) -
das prueft exakt denselben Code, der auch im echten Tab laeuft.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_biome_overlays_3d.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui.tabs.base_tab import BaseMapTab
from gui.tabs.biome_tab import BiomeTab

# Echte Kartengroesse (CLAUDE.md "Gruene Tests koennen eine tote Funktion
# verdecken" - eine ausgedachte Groesse testet eine ausgedachte Situation).
_ECHTE_KARTENGROESSE = 256


class _FakeCheckbox:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeWrapper:
    """Minimaler Ersatz fuer DisplayWrapper - _push_overlays() greift nur
    auf `.display` zu."""

    def __init__(self, display):
        self.display = display


class _FakeDisplay2D:
    def __init__(self):
        self.calls = []

    def overlay_settlements(self, settlements, landmarks, roadsites):
        self.calls.append(("overlay_settlements", settlements, landmarks, roadsites))

    def overlay_river_generations(self, generation_map, zeige_mikro=False):
        self.calls.append(("overlay_river_generations", generation_map, zeige_mikro))


class _FakeDisplay3D:
    """`heightmap` steht wie im echten MapDisplay3DWidget bereits (seit der
    heightmap-Property aus demselben Code-Review), weil update_heightmap()
    laut Vorbild-Reihenfolge vor jedem Overlay-Push laeuft."""

    def __init__(self, heightmap):
        self.heightmap = heightmap
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


class _FakeDataLODManager:
    def __init__(self, mit_daten):
        n = _ECHTE_KARTENGROESSE
        self._settlements = [{"id": 1}] if mit_daten else []
        self._river_generation = np.ones((n, n), dtype=np.int32) if mit_daten else None

    def get_settlement_data(self, key):
        return {
            "settlement_list": self._settlements,
            "landmark_list": [],
            "roadsite_list": [],
        }.get(key)

    def get_terrain_data(self, key):
        return self._river_generation if key == "river_generation" else None


class _FakeBiomeTab:
    """Duck-typed Stand-in fuer BiomeTab - siehe Moduldocstring.

    `_push_overlays` ist die ECHTE BaseMapTab._push_overlays (als
    Klassenattribut gebunden), keine eigene Nachbildung - damit dieser Test
    denselben Dispatch-Code prueft wie tests/smoke_test_push_overlays.py.
    """

    _push_overlays = BaseMapTab._push_overlays

    def __init__(self, settlements_checked, rivers_checked, mit_daten=True):
        heightmap = np.ones((_ECHTE_KARTENGROESSE, _ECHTE_KARTENGROESSE), dtype=np.float32)
        self.map_display_2d = _FakeWrapper(_FakeDisplay2D())
        self.map_display_3d = _FakeWrapper(_FakeDisplay3D(heightmap))
        self.data_lod_manager = _FakeDataLODManager(mit_daten)
        self.settlements_overlay = _FakeCheckbox(settlements_checked)
        self.rivers_overlay = _FakeCheckbox(rivers_checked)


def ausfall_kommt_nicht_zurueck():
    """
    1. DIE EIGENTLICHE ZUSICHERUNG: apply_overlays() muss bis zu den
    3D-Aufrufen durchlaufen (frueher: brach bei current_view != '2d' in
    der ersten Zeile ab).
    """
    fehler = []
    tab = _FakeBiomeTab(settlements_checked=True, rivers_checked=True)
    BiomeTab.apply_overlays(tab)
    d3 = tab.map_display_3d.display

    settlement_gesetzt = any(
        call == ("settlement", "uebersicht", call[2]) and call[2] is not None
        for call in d3.overlay_data_calls
    )
    print(f"[{'OK' if settlement_gesetzt else 'FEHLER'}] Settlements: "
          f"update_overlay_data('settlement', 'uebersicht', ...) aufgerufen "
          f"({len(d3.overlay_data_calls)} Aufruf(e))")
    if not settlement_gesetzt:
        fehler.append("Settlements: update_overlay_data nie aufgerufen - "
                       "apply_overlays() ist vermutlich frueh ausgestiegen")

    settlement_sichtbar = ("settlement", "uebersicht", True) in d3.visibility_calls
    print(f"[{'OK' if settlement_sichtbar else 'FEHLER'}] Settlements: "
          f"set_layer_visibility('settlement', 'uebersicht', True) aufgerufen")
    if not settlement_sichtbar:
        fehler.append("Settlements: Sichtbarkeit nie auf True gesetzt")

    fluss_gezeichnet = len(d3.river_calls) == 1
    print(f"[{'OK' if fluss_gezeichnet else 'FEHLER'}] Flussnetz: "
          f"overlay_river_generations() aufgerufen "
          f"({len(d3.river_calls)} Aufruf(e))")
    if not fluss_gezeichnet:
        fehler.append("Flussnetz: overlay_river_generations nie aufgerufen")

    return fehler


def abwaehlen_entfernt_die_textur():
    """
    2. Ein deaktivierter Haken muss die 3D-Textur ABSCHALTEN, nicht nur
    kein neues Bild liefern - sonst bliebe die alte Textur liegen.
    """
    fehler = []
    tab = _FakeBiomeTab(settlements_checked=False, rivers_checked=False)
    BiomeTab.apply_overlays(tab)
    d3 = tab.map_display_3d.display

    settlement_ausgeblendet = ("settlement", "uebersicht", False) in d3.visibility_calls
    print(f"[{'OK' if settlement_ausgeblendet else 'FEHLER'}] Settlements "
          f"abgewaehlt: set_layer_visibility(..., False) aufgerufen")
    if not settlement_ausgeblendet:
        fehler.append("Settlements: Sichtbarkeit beim Abwaehlen nicht auf "
                       "False gesetzt")

    fluss_entfernt = d3.river_cleared
    print(f"[{'OK' if fluss_entfernt else 'FEHLER'}] Flussnetz abgewaehlt: "
          f"clear_river_overlay() aufgerufen")
    if not fluss_entfernt:
        fehler.append("Flussnetz: clear_river_overlay() beim Abwaehlen "
                       "nicht aufgerufen")

    return fehler


def beide_ansichten_bekommen_dasselbe_overlay():
    """
    3. Ersetzt den frueheren '2D unveraendert bei current_view=3d'-Test:
    seit Ticket #9 gibt es keine current_view-Weiche mehr, _push_overlays()
    schickt IMMER an 2D UND 3D. Die relevante Zusicherung ist jetzt die
    Umkehrung - dass die 2D-Anzeige beim selben apply_overlays()-Aufruf
    GLEICHZEITIG denselben Aufruf bekommt wie 3D, nicht nur irgendwann
    spaeter bei einem Tabwechsel.
    """
    fehler = []
    tab = _FakeBiomeTab(settlements_checked=True, rivers_checked=True)
    BiomeTab.apply_overlays(tab)
    d2 = tab.map_display_2d.display

    settlement_2d = any(c[0] == "overlay_settlements" for c in d2.calls)
    print(f"[{'OK' if settlement_2d else 'FEHLER'}] Settlements: 2D bekommt "
          f"ebenfalls overlay_settlements()")
    if not settlement_2d:
        fehler.append("Settlements: 2D-Anzeige bekam keinen Aufruf")

    fluss_2d = any(c[0] == "overlay_river_generations" for c in d2.calls)
    print(f"[{'OK' if fluss_2d else 'FEHLER'}] Flussnetz: 2D bekommt "
          f"ebenfalls overlay_river_generations()")
    if not fluss_2d:
        fehler.append("Flussnetz: 2D-Anzeige bekam keinen Aufruf")

    return fehler


def lauf():
    gruppen = [
        ("ausfall_kommt_nicht_zurueck", ausfall_kommt_nicht_zurueck),
        ("abwaehlen_entfernt_die_textur", abwaehlen_entfernt_die_textur),
        ("beide_ansichten_bekommen_dasselbe_overlay", beide_ansichten_bekommen_dasselbe_overlay),
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
