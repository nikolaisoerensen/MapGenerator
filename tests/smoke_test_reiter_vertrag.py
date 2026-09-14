"""
Path: tests/smoke_test_reiter_vertrag.py

Erfuellt jeder Reiter den Vertrag, den die Shell voraussetzt?

ANLASS (Nutzerbefund 2026-08-26, am laufenden Programm): *"geht nicht und
oben ist der reiter doppelt und jeder naechste reiter ist verschoben (also
regionen(2) ist terrain und terrain ist flussnetzwerk etc."*

DER VERTRAG STEHT IN DER SHELL, NICHT IM REITER. `MapEditorWindow.
_add_successful_tab()` zerlegt jeden Reiter in drei Teile:

    index = self.main_tab_bar.addTab(tab_name)          # (1) Beschriftung
    self.viewport_stack.addWidget(tab.viewport_widget)  # (2) Karte
    self.parameter_stack.addWidget(tab.parameter_widget)
    self.statistics_stack.addWidget(tab.statistics_widget)

Fehlt eines der drei Attribute, wirft Zeile (2) - **nachdem (1) schon
gelaufen ist**. In der Reiterleiste bleibt dann eine Beschriftung ohne
Inhalt stehen, und JEDER FOLGENDE REITER ist um eins verschoben: man klickt
"Terrain" und bekommt das Flussnetzwerk. Zusaetzlich faengt die aeussere
Fehlerbehandlung den Absturz ab und legt einen Fehlerreiter mit demselben
Namen an - der Name steht dann doppelt.

WARUM KEIN VORHANDENER TEST DAS SAH. Der neue Regionsreiter liess sich
headless einwandfrei bauen, und `tests/smoke_test_region_tab.py` lief mit
zwoelf gruenen Pruefungen durch. Geprueft wurde der Reiter FUER SICH - der
Vertrag lebt aber in der Shell. Dieselbe Klasse Fehler wie das
3D-Anzeigeregister am selben Tag: nicht eine fehlende Methode, sondern eine
fehlende Verbindung zwischen zwei Teilen, die einzeln in Ordnung sind.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_reiter_vertrag.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import logging

from PyQt6.QtWidgets import QApplication, QWidget

logging.disable(logging.WARNING)
_APP = QApplication.instance() or QApplication([])

# Die drei Attribute, die `_add_successful_tab()` abgreift.
PFLICHT = ("viewport_widget", "parameter_widget", "statistics_widget")


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    import gui.map_editor as me

    # --- 1. Der Vertrag steht so in der Shell, wie dieser Test annimmt ------
    #
    # Ohne diese Pruefung wuerde der Test stillschweigend veralten, sobald
    # jemand die Shell umbaut - und dann pruefte er drei Attribute, die
    # niemand mehr abfragt.
    import inspect
    quelle = inspect.getsource(me.MapEditorWindow._add_successful_tab)
    unbekannt = [a for a in PFLICHT if f"tab_instance.{a}" not in quelle]
    fehler += check("die Shell greift genau diese drei Attribute ab",
                    not unbekannt,
                    f"in _add_successful_tab nicht gefunden: {unbekannt}")

    # --- 2. Jede Reiterklasse liefert sie -----------------------------------
    #
    # Die Klassen werden aus map_editor gelesen, damit die Liste nicht
    # danebenlaufen kann: derselbe Ort, an dem die Shell sie holt.
    klassen = {}
    for name in dir(me):
        wert = getattr(me, name)
        if (isinstance(wert, type) and name.endswith("Tab")
                and issubclass(wert, QWidget)):
            klassen[name] = wert
    fehler += check("Reiterklassen gefunden", len(klassen) >= 8,
                    f"{len(klassen)}: {', '.join(sorted(klassen))}")

    fehlend = []
    for name, klasse in sorted(klassen.items()):
        try:
            reiter = klasse(
                data_lod_manager=None, parameter_manager=None,
                navigation_manager=None, shader_manager=None,
                generation_orchestrator=None)
        except Exception as f:                            # noqa: BLE001
            # Ein Reiter, der ohne Manager nicht baut, ist hier nicht
            # pruefbar - das ist kein Vertragsbruch, sondern eine Grenze
            # dieses Tests. Wird gemeldet, nicht verschwiegen.
            print(f"     ({name} ohne Manager nicht baubar: "
                  f"{type(f).__name__}) - uebersprungen")
            continue
        luecken = [a for a in PFLICHT if not hasattr(reiter, a)]
        if luecken:
            fehlend.append(f"{name}: {', '.join(luecken)}")
        else:
            for a in PFLICHT:
                if not isinstance(getattr(reiter, a), QWidget):
                    fehlend.append(f"{name}.{a} ist kein QWidget")

    fehler += check("jeder baubare Reiter liefert die drei Widgets",
                    not fehlend, "; ".join(fehlend))

    # --- 3. Die Shell tatsaechlich aufbauen und ABZAEHLEN -------------------
    #
    # DIE STAERKSTE FORM DER PRUEFUNG, und die einzige, die den Befund vom
    # 2026-08-26 direkt gefunden haette: nicht "hat der Reiter das
    # Attribut", sondern "kommen am Ende gleich viele Beschriftungen wie
    # Inhalte heraus".
    #
    # Genau das war kaputt: die Reiterleiste hatte einen Eintrag mehr als
    # der Inhaltsstapel, deshalb zeigte jeder Reiter den Inhalt seines
    # Nachfolgers.
    fenster = me.MapEditorWindow()
    zahlen = {
        "Reiterleiste": fenster.main_tab_bar.count(),
        "Viewport": fenster.viewport_stack.count(),
        "Parameter": fenster.parameter_stack.count(),
        "Statistik": fenster.statistics_stack.count(),
    }
    einig = len(set(zahlen.values())) == 1
    fehler += check("Reiterleiste und Inhaltsstapel sind gleich lang",
                    einig,
                    ", ".join(f"{k} {v}" for k, v in zahlen.items()))

    beschriftungen = [fenster.main_tab_bar.tabText(i)
                      for i in range(fenster.main_tab_bar.count())]
    doppelt = [b for b in set(beschriftungen)
               if beschriftungen.count(b) > 1]
    fehler += check("keine Beschriftung steht doppelt",
                    not doppelt,
                    ", ".join(doppelt) if doppelt else
                    " | ".join(beschriftungen))
    fehler += check("tab_order passt zur Reiterzahl",
                    len(fenster.tab_order) == fenster.main_tab_bar.count(),
                    f"{len(fenster.tab_order)} gegen "
                    f"{fenster.main_tab_bar.count()}")

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("alle geprueften Reiter erfuellen den Shell-Vertrag")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
