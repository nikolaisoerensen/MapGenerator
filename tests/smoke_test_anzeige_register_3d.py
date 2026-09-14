"""
Path: tests/smoke_test_anzeige_register_3d.py

Kommt jeder Anzeigemodus eines Reiters auch im 3D an?

ANLASS (Nutzerbefund 2026-08-26): *"du hast im übrigen gegen die regel
verstossen eine 2D karte zu erstellen ohne den shader für 3D
mitzuschreiben. bei Flussnetzwerk gehen die 3D karten nicht."*

DIE STEHENDE REGEL AUS CLAUDE.md, AN EINER NEUEN STELLE. Bisher hiess der
Fehler "Anzeigemethode gibt es nur auf MapDisplay2D, die hasattr-Weiche
trifft im 3D nie zu". `tests/smoke_test_display_methoden_existieren.py`
faengt das. Hier ist es ein fehlender REGISTEREINTRAG:

    mapped_layer = self._LAYER_NAME_MAP_3D.get(layer_type)

Steht ein Layer dort nicht, ist `mapped_layer` None - es wird nichts ans 3D
gepusht, UND die Sichtbarkeitsschleife direkt danach schaltet alle Layer
des Reiters unsichtbar. Im 3D bleibt blankes Gelaende stehen. Von aussen
ist das von "die Methode fehlt" nicht zu unterscheiden: es passiert
lautlos nichts.

GEFUNDEN WURDE: der ganze Flussreiter fehlte. Er meldet sich mit
`generator_type = "terrain"` an, aber weder `river_water` (2026-08-26 neu)
noch `river_order` (seit jeher) standen in einer der beiden Listen.

WAS HIER GEPRUEFT WIRD:

  1. Jeder Anzeigemodus, den ein Reiter anbietet, hat einen Eintrag in
     `_LAYER_NAME_MAP_3D` - oder steht in NUR_2D unten, mit Begruendung.
  2. Jeder gemappte Name steht auch in `_LAYER_SELECTION_KEYS_3D` seines
     Reitertyps. Ohne das wird er zwar gepusht, aber nie sichtbar
     geschaltet.
  3. Jeder Name aus `_LAYER_SELECTION_KEYS_3D` existiert auf der 3D-Seite
     in `layer_visibility` und `overlay_data`. Fehlt er dort, wirft
     `set_layer_visibility()` ins Leere.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_anzeige_register_3d.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.map_display_3d import MapDisplay3D

# GEPRUEFT WIRD DER GEPUSHTE `layer_type`, NICHT DER MODUSNAME.
#
# Die beiden fallen oft auseinander: der Terrain-Reiter nennt seinen Modus
# "slope", pusht aber `data_type = "slopemap"`, und im Register steht
# "slopemap". Ein erster Anlauf dieses Tests verglich die Modusnamen und
# meldete "terrain.slope fehlt" - der Test hatte unrecht, nicht der Code.
# Deshalb steht hier ausdruecklich das Paar (Modus, gepushter Layer).
NUR_2D = {
    "heightmap": "Die Grundkarte ist im 3D das Netz selbst, kein Overlay.",
    "heightmap_combined": "wie heightmap",
    "spielkarte": "Die neun Vielecke sind eine reine Zerlegungsansicht - als "
                  "Skin auf dem 3D-Netz ohne Nutzen, weil ihre Grenzen gerade "
                  "Schnitte sind, keine Gelaendeform.",
    "__overlay__": "Laeuft nicht ueber das Layer-Register, sondern ueber eine "
                   "eigene Methode, die es auf BEIDEN Anzeigeklassen gibt "
                   "(z.B. overlay_river_generations()).",
}

# Reiter -> (generator_type, ((Modusname, gepushter layer_type), ...)).
# Der Flussreiter meldet sich als "terrain" an - genau deshalb fiel seine
# Luecke nicht auf.
REITER_MODI = {
    "river": ("terrain", (
        ("height", "heightmap"),
        ("river_water", "river_water"),
        ("rivers", "__overlay__"),
        ("river_order", "river_order"),
    )),
    "terrain": ("terrain", (
        ("height", "heightmap"),
        ("combined", "heightmap_combined"),
        ("slope", "slopemap"),
        ("regions", "region_map"),
        ("kuestentypen", "kuesten_archetyp"),
        ("spielkarten", "spielkarte"),
        ("hinterland_height", "hinterland_height"),
        ("voronoi_map", "voronoi_map"),
    )),
}


def check(label, bedingung, zusatz=""):
    print(f"[{'OK' if bedingung else 'FAIL'}] {label}"
          + (f" - {zusatz}" if zusatz else ""))
    return [] if bedingung else [f"{label}{' - ' + zusatz if zusatz else ''}"]


def lauf():
    fehler = []
    karte = BaseMapTab._LAYER_NAME_MAP_3D
    auswahl = BaseMapTab._LAYER_SELECTION_KEYS_3D

    # --- 1. Jeder Modus ist gemappt oder begruendet nur-2D -------------------
    ungemappt = []
    for reiter, (typ, modi) in REITER_MODI.items():
        for modus, layer in modi:
            if layer in NUR_2D:
                continue
            if layer not in karte:
                ungemappt.append(f"{reiter}.{modus} (pusht {layer})")
    fehler += check("jeder Anzeigemodus ist im 3D-Register",
                    not ungemappt,
                    ", ".join(ungemappt) if ungemappt else
                    f"{sum(len(m) for _t, m in REITER_MODI.values())} Modi geprueft, "
                    f"{len(NUR_2D)} begruendete Ausnahmen")

    # --- 2. Gemappte Namen sind auch sichtbar schaltbar -----------------------
    nicht_waehlbar = []
    for reiter, (typ, modi) in REITER_MODI.items():
        for modus, layer in modi:
            ziel = karte.get(layer)
            if ziel is None:
                continue
            if ziel not in auswahl.get(typ, set()):
                nicht_waehlbar.append(f"{reiter}.{modus} -> {typ}.{ziel}")
    fehler += check("jeder gemappte Layer ist im 3D sichtbar schaltbar",
                    not nicht_waehlbar, ", ".join(nicht_waehlbar))

    # --- 3. Die 3D-Seite kennt jeden Namen -----------------------------------
    # Ohne ein echtes GL-Fenster: die Vorgabe-Dicts stehen im Konstruktor,
    # also wird eine Instanz OHNE initializeGL() gebaut.
    zeige = MapDisplay3D.__new__(MapDisplay3D)
    try:
        MapDisplay3D.__init__(zeige)
    except Exception as f:                                # noqa: BLE001
        return check("MapDisplay3D baubar", False, str(f)[:120])

    fehlt_sicht, fehlt_daten = [], []
    for typ, namen in auswahl.items():
        for name in namen:
            if name not in zeige.layer_visibility.get(typ, {}):
                fehlt_sicht.append(f"{typ}.{name}")
            if name not in zeige.overlay_data.get(typ, {}):
                fehlt_daten.append(f"{typ}.{name}")
    fehler += check("jeder waehlbare Layer hat einen Sichtbarkeits-Eintrag",
                    not fehlt_sicht, ", ".join(fehlt_sicht))
    fehler += check("jeder waehlbare Layer hat einen Datenplatz",
                    not fehlt_daten, ", ".join(fehlt_daten))

    # --- 4. UND ER WIRD AUCH GEZEICHNET --------------------------------------
    #
    # Die fuenfte Tabelle, und bis zum 2026-08-27 die einzige ungepruefte:
    # die Verteilung in `paintGL`. Sie ist teils eine Kette aus `if`, teils
    # eine fest geschriebene Liste in `_render_<reiter>_tab()`. Ein Layer
    # kann in ALLEN vier Registern oben stehen und trotzdem nie gezeichnet
    # werden - dann bleibt die 3D-Ansicht leer, ohne Fehlermeldung.
    #
    # Genau so war es bei `evaporation`: in beiden Registern des
    # Wasserreiters, aber nicht in `water_layers`. Damals gefunden, weil
    # dieser Test die Vorgabe-Dicts prueft - die Zeichnung selbst hat
    # niemand geprueft, es war Glueck.
    #
    # DAS VERFAHREN IST EINE HEURISTIK, und das gehoert dazugesagt: geprueft
    # wird, ob der Layername als Zeichenkette im Quelltext der zugehoerigen
    # Zeichenfunktion vorkommt. Ein Name, der dort nur in einem Kommentar
    # steht, wuerde durchgehen. Die Gegenrichtung - ein Name, der FEHLT -
    # ist dafuer sicher, und das ist der Fehler, um den es geht.
    import inspect
    ohne_zeichnung, ohne_funktion = [], []
    for typ, namen in sorted(auswahl.items()):
        fn = getattr(MapDisplay3D, f"_render_{typ}_tab", None)
        if fn is None:
            ohne_funktion.append(typ)
            continue
        quelle = inspect.getsource(fn)
        for name in sorted(namen):
            if f'"{name}"' not in quelle:
                ohne_zeichnung.append(f"{typ}.{name}")
    fehler += check("zu jedem Reiter gibt es eine Zeichenfunktion",
                    not ohne_funktion,
                    ", ".join(f"_render_{t}_tab fehlt" for t in ohne_funktion))
    fehler += check("jeder waehlbare Layer wird in paintGL auch gezeichnet",
                    not ohne_zeichnung,
                    ", ".join(ohne_zeichnung)
                    or f"{sum(len(n) for n in auswahl.values())} Layer "
                       f"ueber {len(auswahl)} Reiter")

    print()
    print("=" * 78)
    if fehler:
        print(f"NICHT IN ORDNUNG - {len(fehler)} Befunde:")
        for f in fehler:
            print(f"   {f}")
        return 1
    print("das 3D-Anzeigeregister ist vollstaendig")
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
