"""
Path: tests/smoke_test_display_methoden_existieren.py

Faengt die Fehlerklasse ab, die den Reiter "Siedlungen (Regional)" unbemerkt
lahmgelegt hat (docs/OFFENE_PUNKTE.md 5.15, gefunden 2026-08-13 durch einen
Nutzerbefund, NICHT durch einen Test).

DER FEHLER: settlement_regional_tab.py rief

    if hasattr(ziel, "update_map_data"):     ziel.update_map_data(...)
    elif hasattr(ziel, "update_heightmap"):  ziel.update_heightmap(...)

`ziel` ist in der 2D-Ansicht ein MapDisplay2D - und das hat WEDER
`update_map_data` NOCH `update_heightmap`, sondern `update_display`. Beide
Weichen trafen also nicht zu, die Basiskarte wurde nie gezeichnet, und weil
jede Overlay-Methode mit `if self.current_data is None: return` beginnt,
kehrten anschliessend auch alle Overlays sofort zurueck. Ergebnis: ein
komplett leerer Reiter, ohne Absturz, ohne Logzeile, ohne fehlschlagenden
Test - genau das Muster aus CLAUDE.md ("jeder stille Rueckfall auf einen
Ersatzpfad braucht eine laute Logzeile").

WAS DIESER TEST PRUEFT: dass jeder per `hasattr(...)`-Weiche angesprochene
Methodenname auf MINDESTENS einer der beiden Anzeigeklassen existiert. Ein
Name, den keine von beiden kennt, ist immer ein Fehler - entweder ein
Tippfehler oder eine Methode, die es nie gab. Beides fuehrt zu genau dem
stillen Nichtstun oben.

Bewusst NICHT geprueft: ob die Weiche fuer JEDE Anzeigeart einen Treffer hat.
Manche Aufrufe sind absichtlich nur fuer 3D gedacht (z.B. `update_shademap`).
Der Fehler oben war ein Name, den KEINE Klasse hatte.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")

from gui.widgets.map_display_2d import MapDisplay2D
from gui.widgets.map_display_3d import MapDisplay3D, MapDisplay3DWidget
# DisplayWrapper gehoert dazu: die Tabs halten in `self.map_display_2d`/
# `self.map_display_3d` NICHT die Anzeige selbst, sondern diesen Wrapper
# (siehe base_tab.py create_ui()). Erst `.display` darauf ist die echte
# MapDisplay2D/3D. Beide Ebenen muessen hier bekannt sein, sonst meldet der
# Test Wrapper-eigene Methoden wie `set_active()` faelschlich als fehlend -
# genau das ist beim ersten Lauf dieses Tests passiert.
from gui.widgets.widgets import DisplayWrapper


# Namen, die absichtlich auf anderen Objekten als den Anzeigeklassen geprueft
# werden (Manager, Tabs, Datenobjekte) - keine Display-Methoden.
_KEIN_DISPLAY_NAME = {
    "shape", "ndim", "display", "x", "y", "node_id", "node_location",
    "node_type", "get_current_display", "viewport_widget", "layout",
    "control_panel", "isVisible", "map_display_3d", "map_display_2d",
    "data_lod_manager", "current_display", "generator_type",
    "parameter_manager", "setChecked", "isChecked", "value", "text",
    "_current_parameters", "logger", "canvas", "ax", "figure",
}

_TAB_VERZEICHNIS = Path("gui/tabs")


def sammle_hasattr_namen():
    """Alle `hasattr(<irgendwas>, "name")`-Aufrufe in gui/tabs/*.py."""
    muster = re.compile(r"hasattr\(\s*([A-Za-z_][\w\.\[\]'\"]*)\s*,\s*['\"](\w+)['\"]\s*\)")
    treffer = []
    for pfad in sorted(_TAB_VERZEICHNIS.glob("*.py")):
        text = pfad.read_text(encoding="utf-8")
        for zeilennr, zeile in enumerate(text.split("\n"), 1):
            for objekt, name in muster.findall(zeile):
                treffer.append((pfad.name, zeilennr, objekt, name))
    return treffer


def run():
    bekannt = set()
    for klasse in (MapDisplay2D, MapDisplay3D, MapDisplay3DWidget, DisplayWrapper):
        bekannt |= {n for n in dir(klasse) if not n.startswith("__")}

    treffer = sammle_hasattr_namen()
    print(f"{len(treffer)} hasattr()-Aufrufe in {_TAB_VERZEICHNIS}/ gefunden\n")

    # Nur solche betrachten, die nach einer Anzeige-Methode aussehen: das
    # gepruefte Objekt heisst display/ziel/anzeige/... ODER der Name ist auf
    # einer der Anzeigeklassen bekannt (dann ist es sicher ein Display-Aufruf).
    verdaechtige_objekte = ("display", "ziel", "anzeige", "current_display",
                            "map_display", "self.map_display")

    fehler = []
    geprueft = 0
    for datei, zeile, objekt, name in treffer:
        if name in _KEIN_DISPLAY_NAME:
            continue
        ist_display_objekt = any(v in objekt for v in verdaechtige_objekte)
        if not ist_display_objekt:
            continue
        geprueft += 1
        if name not in bekannt:
            fehler.append((datei, zeile, objekt, name))

    print(f"{geprueft} davon betreffen ein Anzeige-Objekt\n")

    if fehler:
        print("FEHLGESCHLAGEN - Methodennamen, die WEDER MapDisplay2D noch")
        print("MapDisplay3D(Widget) kennen (die hasattr-Weiche trifft nie zu,")
        print("der Code tut still gar nichts):\n")
        for datei, zeile, objekt, name in fehler:
            print(f"  {datei}:{zeile}  hasattr({objekt}, {name!r})")
        return False

    print("OK - jeder auf einem Anzeige-Objekt gepruefte Methodenname")
    print("     existiert auf mindestens einer der Anzeigeklassen.")
    return True


def run_regionaltab_zeichnet_basiskarte():
    """Gezielt: der Reiter, an dem der Fehler auftrat, darf die Basiskarte
    NICHT mehr ueber eine eigene hasattr-Weiche zeichnen, sondern muss den
    gemeinsamen Push-Pfad benutzen."""
    quelle = Path("gui/tabs/settlement_regional_tab.py").read_text(encoding="utf-8")
    ok = True

    if "_push_data_to_current_display" not in quelle:
        print("[FAIL] settlement_regional_tab.py benutzt nicht "
              "_push_data_to_current_display()")
        ok = False
    else:
        print("[OK] settlement_regional_tab.py benutzt den gemeinsamen Push-Pfad")

    for tot in ("ziel.update_map_data", "ziel.update_heightmap"):
        if tot in quelle:
            print(f"[FAIL] alter, nie zutreffender Aufruf noch vorhanden: {tot}")
            ok = False
    if ok:
        print("[OK] keine der beiden alten, nie zutreffenden Weichen mehr da")
    return ok


if __name__ == "__main__":
    ergebnisse = {
        "hasattr_namen_existieren": run(),
        "regionaltab_zeichnet_basiskarte": run_regionaltab_zeichnet_basiskarte(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
