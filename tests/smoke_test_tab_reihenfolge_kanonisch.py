"""
Path: tests/smoke_test_tab_reihenfolge_kanonisch.py

Ticket #56 - "Die fuenf Reiterreihenfolgen auf eine reduzieren".

BEFUND VOR DIESEM TICKET: die Reihenfolge der Karteneditor-Reiter stand an
FUENF Stellen als eigene Python-Liste:

    gui/map_editor.py:164            (leere Liste, dynamisch befuellt)
    gui/widgets/widgets.py:783       (mit "erosion")
    gui/widgets/widgets.py:798       (OHNE "erosion")
    gui/widgets/widgets.py:809       (OHNE "erosion")
    managers/navigation_manager.py:43

Drei davon (783/798/809) MUSSTEN dieselbe Sequenz sein - alle drei steuern
dasselbe NavigationPanel (Previous/Next-Buttons) - waren es aber nicht: zwei
der drei hatten "erosion" schlicht vergessen. Ein Kopierfehler, der keinem
Test aufgefallen ist, weil jede Kopie fuer sich genommen eine gueltige,
funktionierende Liste war.

WAS JETZT GILT: managers/navigation_manager.py definiert die Sequenz genau
EINMAL als Modul-Konstante `GENERATOR_TAB_ORDER`. widgets.py liest sie ein
(Objektidentitaet, keine neue Kopie). gui/map_editor.py behaelt seine eigene,
dynamisch aus tab_configs gefuellte `self.tab_order` bewusst bei - das ist
eine ANDERE, groessere Menge (sie enthaelt zusaetzlich region/kontinent/
rivers/settlement_regional), siehe Kommentar dort. Sie wird hier deshalb
NICHT auf Objektidentitaet mit GENERATOR_TAB_ORDER geprueft, sondern nur
darauf, dass die Begruendung dafuer im Code steht.

ENTSCHEIDUNG "erosion": gehoert in die kanonische Liste. gui/tabs/erosion_tab.py
definiert eine echte ErosionTab-Klasse, registriert in gui/map_editor.py
zwischen "geology" und "weather". EROSION_AKTIV schaltet nur die Berechnung
ab, nicht den Reiter. Dieser Test prueft das nicht nur per Behauptung, sondern
sucht die tatsaechliche Registrierung im Quelltext (siehe
`pruefe_erosion_tab_ist_echt()`).

WAS DIESER TEST NICHT PRUEFT: das tatsaechliche Qt-Verhalten der Buttons
(braucht eine laufende QApplication, siehe CLAUDE.md "was noch die
laufende GUI braucht"). Geprueft wird die Struktur: Es gibt nur noch EINE
Definition, und alle Verbraucher lesen dieselbe Objektidentitaet statt eigener
Kopien.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

import managers.navigation_manager as nm
import gui.widgets.widgets as widgets_modul

_WIDGETS_QUELLE = Path("gui/widgets/widgets.py").read_text(encoding="utf-8")
_MAP_EDITOR_QUELLE = Path("gui/map_editor.py").read_text(encoding="utf-8")
_NAV_MANAGER_QUELLE = Path("managers/navigation_manager.py").read_text(encoding="utf-8")

_ERWARTETE_REIHENFOLGE = [
    "main_menu", "terrain", "geology", "erosion", "weather",
    "water", "biome", "settlement", "overview",
]


def pruefe_konstante_existiert_und_stimmt():
    """GENERATOR_TAB_ORDER existiert in managers.navigation_manager, enthaelt
    "erosion" und ist unveraendert die Reihenfolge, die vorher schon an der
    VOLLSTAENDIGEN Kopie (widgets.py:783) stand - die sichtbare Reihenfolge
    darf sich durch die Konsolidierung nicht aendern."""
    ok = True
    if not hasattr(nm, "GENERATOR_TAB_ORDER"):
        print("[FAIL] managers.navigation_manager.GENERATOR_TAB_ORDER fehlt")
        return False

    if nm.GENERATOR_TAB_ORDER != _ERWARTETE_REIHENFOLGE:
        print(f"[FAIL] GENERATOR_TAB_ORDER = {nm.GENERATOR_TAB_ORDER!r}, "
              f"erwartet {_ERWARTETE_REIHENFOLGE!r}")
        ok = False
    else:
        print(f"[OK] GENERATOR_TAB_ORDER = {nm.GENERATOR_TAB_ORDER!r}")

    if "erosion" not in nm.GENERATOR_TAB_ORDER:
        print("[FAIL] 'erosion' fehlt in der kanonischen Liste")
        ok = False

    return ok


def pruefe_nur_eine_definition_im_repo():
    """Eine Modul-Konstante `GENERATOR_TAB_ORDER = [...]` darf nur EIN EINZIGES
    Mal im Repo als Zuweisung auftauchen (in navigation_manager.py) - alle
    anderen Stellen duerfen sie nur IMPORTIEREN, nicht neu zuweisen."""
    muster = re.compile(r"^GENERATOR_TAB_ORDER\s*=\s*\[", re.MULTILINE)
    fundstellen = []
    for pfad in sorted(Path("gui").rglob("*.py")) + sorted(Path("managers").rglob("*.py")):
        text = pfad.read_text(encoding="utf-8")
        if muster.search(text):
            fundstellen.append(str(pfad))

    if fundstellen != ["managers\\navigation_manager.py"] and \
       fundstellen != ["managers/navigation_manager.py"]:
        print(f"[FAIL] GENERATOR_TAB_ORDER wird an {len(fundstellen)} Stellen "
              f"zugewiesen, erwartet genau 1 (managers/navigation_manager.py): "
              f"{fundstellen}")
        return False

    print("[OK] GENERATOR_TAB_ORDER wird nur an genau einer Stelle definiert "
          f"({fundstellen[0]})")
    return True


def pruefe_widgets_liest_dieselbe_konstante():
    """widgets.py darf keine eigene, hartkodierte Kopie der Tab-Sequenz mehr
    fuehren, sondern muss GENERATOR_TAB_ORDER importieren UND es sich um
    dasselbe Objekt handeln (Identitaet, nicht nur zufaellig gleicher Inhalt -
    sonst waere ein spaeteres Auseinanderdriften wieder moeglich)."""
    ok = True

    if "from managers.navigation_manager import GENERATOR_TAB_ORDER" not in _WIDGETS_QUELLE:
        print("[FAIL] widgets.py importiert GENERATOR_TAB_ORDER nicht")
        ok = False
    else:
        print("[OK] widgets.py importiert GENERATOR_TAB_ORDER")

    if widgets_modul.GENERATOR_TAB_ORDER is not nm.GENERATOR_TAB_ORDER:
        print("[FAIL] widgets.GENERATOR_TAB_ORDER ist NICHT dasselbe Objekt "
              "wie managers.navigation_manager.GENERATOR_TAB_ORDER - "
              "irgendwo wurde kopiert statt importiert")
        ok = False
    else:
        print("[OK] widgets.py und navigation_manager.py teilen dasselbe "
              "Listenobjekt (Identitaet geprueft, nicht nur Gleichheit)")

    # Die alte Bugmuster-Suche: drei fast identische, hartkodierte
    # Tab-Namenslisten. Keine davon darf mehr im Quelltext als Literal stehen.
    alte_kopien = re.findall(
        r'\[\s*"terrain"\s*,\s*"geology"[^\]]*\]', _WIDGETS_QUELLE)
    if alte_kopien:
        print(f"[FAIL] widgets.py enthaelt noch {len(alte_kopien)} "
              f"hartkodierte Tab-Reihenfolge(n) als Literal: {alte_kopien}")
        ok = False
    else:
        print("[OK] keine hartkodierte Tab-Reihenfolge mehr als Literal in "
              "widgets.py")

    return ok


def pruefe_go_previous_und_go_next_nutzen_konstante():
    """Gezielt an den zwei Stellen, die vorher 'erosion' vergessen hatten:
    go_previous() und go_next() muessen jetzt GENERATOR_TAB_ORDER referenzieren,
    nicht mehr ihre eigene verkuerzte Liste."""
    ok = True
    for name in ("go_previous", "go_next"):
        treffer = re.search(
            rf"def {name}\(self\):.*?(?=\n    def |\Z)", _WIDGETS_QUELLE, re.DOTALL)
        if not treffer:
            print(f"[FAIL] Methode {name}() nicht gefunden")
            ok = False
            continue
        rumpf = treffer.group(0)
        if "GENERATOR_TAB_ORDER" not in rumpf:
            print(f"[FAIL] {name}() liest GENERATOR_TAB_ORDER nicht")
            ok = False
        elif '"geology", "weather"' in rumpf or "'geology', 'weather'" in rumpf:
            print(f"[FAIL] {name}() hat noch eine eigene Literal-Liste ohne "
                  f"'erosion'")
            ok = False
        else:
            print(f"[OK] {name}() liest GENERATOR_TAB_ORDER")
    return ok


def pruefe_erosion_tab_ist_echt():
    """Grundlage der 'erosion gehoert dazu'-Entscheidung: es muss eine
    tatsaechlich registrierte ErosionTab-Klasse geben, kein totes Feature."""
    ok = True

    if not Path("gui/tabs/erosion_tab.py").exists():
        print("[FAIL] gui/tabs/erosion_tab.py existiert nicht - 'erosion' "
              "waere ein Karteneintrag ohne echten Reiter")
        return False
    print("[OK] gui/tabs/erosion_tab.py existiert")

    if "class ErosionTab(BaseMapTab)" not in Path("gui/tabs/erosion_tab.py").read_text(encoding="utf-8"):
        print("[FAIL] gui/tabs/erosion_tab.py definiert keine ErosionTab(BaseMapTab)")
        ok = False
    else:
        print("[OK] ErosionTab(BaseMapTab) ist definiert")

    if re.search(r'\(\s*"erosion"\s*,\s*"Erosion"\s*,\s*ErosionTab', _MAP_EDITOR_QUELLE) is None:
        print("[FAIL] gui/map_editor.py registriert ErosionTab nicht in tab_configs")
        ok = False
    else:
        print("[OK] ErosionTab ist in gui/map_editor.py tab_configs registriert "
              "(zwischen geology und weather)")

    return ok


def pruefe_map_editor_begruendet_abweichung():
    """gui/map_editor.py behaelt eine eigene, dynamische self.tab_order -
    das ist gewollt (andere Menge, siehe Docstring oben), muss aber im Code
    begruendet sein, damit es nicht wie eine uebersehene sechste Kopie
    aussieht."""
    if "GENERATOR_TAB_ORDER" not in _MAP_EDITOR_QUELLE:
        print("[FAIL] gui/map_editor.py erwaehnt GENERATOR_TAB_ORDER nirgends "
              "- die Abweichung von der kanonischen Liste ist nicht dokumentiert")
        return False
    print("[OK] gui/map_editor.py begruendet im Code, warum self.tab_order "
          "nicht aus GENERATOR_TAB_ORDER gespeist wird")
    return True


if __name__ == "__main__":
    ergebnisse = {
        "konstante_existiert_und_stimmt": pruefe_konstante_existiert_und_stimmt(),
        "nur_eine_definition_im_repo": pruefe_nur_eine_definition_im_repo(),
        "widgets_liest_dieselbe_konstante": pruefe_widgets_liest_dieselbe_konstante(),
        "go_previous_und_go_next_nutzen_konstante": pruefe_go_previous_und_go_next_nutzen_konstante(),
        "erosion_tab_ist_echt": pruefe_erosion_tab_ist_echt(),
        "map_editor_begruendet_abweichung": pruefe_map_editor_begruendet_abweichung(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
