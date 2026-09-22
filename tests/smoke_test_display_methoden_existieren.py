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

Bewusst NICHT geprueft in `run()`: ob die Weiche fuer JEDE Anzeigeart einen
Treffer hat. Manche Aufrufe sind absichtlich nur fuer 3D gedacht (z.B.
`update_shademap`). Der Fehler oben war ein Name, den KEINE Klasse hatte.

ERREICHBARKEIT (Ticket #7, docs/OFFENE_PUNKTE.md 14.2): `run()` und
`run_einseitige_sind_begruendet()` pruefen nur den Methodennamen - nicht, ob
die Funktion, die ihn per hasattr abfragt, ueberhaupt jemals aufgerufen wird.
`run_zweige_sind_erreichbar()` schliesst das: eine Anzeige-Weiche in einer
Funktion, die im ganzen `gui/`-Baum nirgends gerufen wird, ist tote Weiche -
derselbe Fehler wie in 14.5 (vier Aufrufe ins Leere), nur umgekehrt (eine
Funktion mit korrekter Weiche, die selbst niemand ruft).
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
    """Alle `hasattr(<irgendwas>, "name")`-Aufrufe in gui/tabs/*.py, mit dem
    Namen der Funktion, in der die Weiche steht (fuer die Erreichbarkeits-
    pruefung unten - eine leere Zeichenkette, wenn keine `def`-Zeile davor
    gefunden wurde, z.B. auf Modulebene)."""
    muster = re.compile(r"hasattr\(\s*([A-Za-z_][\w\.\[\]'\"]*)\s*,\s*['\"](\w+)['\"]\s*\)")
    def_muster = re.compile(r"^\s*def\s+(\w+)\s*\(")
    treffer = []
    for pfad in sorted(_TAB_VERZEICHNIS.glob("*.py")):
        text = pfad.read_text(encoding="utf-8")
        aktuelle_funktion = ""
        for zeilennr, zeile in enumerate(text.split("\n"), 1):
            def_treffer = def_muster.match(zeile)
            if def_treffer:
                aktuelle_funktion = def_treffer.group(1)
            for objekt, name in muster.findall(zeile):
                treffer.append((pfad.name, zeilennr, objekt, name, aktuelle_funktion))
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
    for datei, zeile, objekt, name, _funktion in treffer:
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


# EINSEITIGE ANZEIGEMETHODEN, absichtlich und begruendet.
#
# WARUM ES DIESES REGISTER GIBT (2026-08-25). Die Pruefung oben verlangt nur,
# dass ein Methodenname auf MINDESTENS EINER der beiden Anzeigeklassen
# existiert. Das ist zu wenig: gibt es ihn nur auf einer, faellt die
# hasattr-Weiche in der ANDEREN Ansicht lautlos aus - kein Fehler, keine
# Warnung, das Overlay fehlt einfach.
#
# Genau so verschwand `overlay_river_network` im Biome-Reiter aus der
# 3D-Ansicht, und genau so war es am 2026-08-24 schon einmal bei
# `overlay_river_generations` (siehe smoke_test_fluss_overlay.py). Zweimal
# dieselbe Fehlerklasse an zwei Stellen.
#
# Jeder Eintrag hier ist eine ENTSCHEIDUNG mit Begruendung. Alles, was
# einseitig ist und NICHT hier steht, laesst den Test fehlschlagen.
# Beide frueheren Eintraege `overlay_river_network` und `objekt_gewaehlt`
# sind seit dem 2026-08-25 nicht mehr noetig: kein Reiter fragt sie noch
# per hasattr ab. Der Test meldet veraltete Eintraege von selbst.
NUR_EINE_ANZEIGE = {
    # --- nur 2D, und das ist richtig so -----------------------------------
    "update_display": "2D-Gegenstueck zu update_heightmap (3D). Welche der "
                      "beiden gerufen wird, entscheidet "
                      "_push_data_to_current_display() in base_tab.py.",
    "draw_plot_physics_snapshot": "Diagnosebild des Plot-Physik-Labors, "
                                  "bewusst nur 2D.",
    "overlay_settlements": "2D-ZWEIG EINES PAARES, kein Loch. Seit Ticket #8-#11 "
                           "(docs/spezifikation/15_ANZEIGE.md) ruft KEIN Reiter mehr direkt "
                           "auf; BaseMapTab._push_overlays() ruft ueber das "
                           "Register \"siedlungen\" fuer 3D stattdessen "
                           "`update_overlay_data(\"settlement\", \"uebersicht\", "
                           "rgba)` (RGBA-Skin). Die Methode selbst gibt es zu "
                           "Recht nur in 2D - matplotlib zeichnet Punkte "
                           "direkt, OpenGL braucht eine Textur.",
    "overlay_roads": "Im 3D sind Wege echte Bandgeometrie "
                     "(gui/widgets/wege_geometrie.py, docs 6.28), keine "
                     "Overlay-Methode.",
    # --- nur 3D, und das ist richtig so -----------------------------------
    "update_heightmap": "3D-Gegenstueck zu update_display, siehe dort.",
    "update_shademap": "Schattenkarte fuer die 3D-Beleuchtung.",
    "update_overlay_data": "Der 3D-Weg fuer JEDES Overlay: RGBA-Textur auf "
                           "das Gelaende. In 2D zeichnet matplotlib direkt.",
    "set_layer_visibility": "Schaltet einzelne 3D-Texturschichten; in 2D "
                            "wird ohnehin bei jeder Aenderung neu gezeichnet.",
    "set_sun_direction": "Es gibt keine Sonne in einer 2D-Karte.",
    "set_world_size_km": "Die 3D-Anzeige braucht den Massstab fuer die "
                         "Hoehenskalierung; 2D zeichnet in Pixeln.",
    "clear_river_overlay": "Im 3D bleibt eine gesetzte Textur liegen, bis sie "
                           "abgeschaltet wird - 2D zeichnet neu und braucht "
                           "das nicht.",
    "setze_auswahlobjekte": "Anklickbare Objekte werden gegen die "
                            "3D-Projektion getroffen (docs 6.29).",
}


# EINSEITIG UND EINE SCHULD - was in 2D geht und in 3D (noch) nicht.
#
# Getrennt von NUR_EINE_ANZEIGE, weil der Unterschied wichtig ist: dort steht,
# was aus gutem Grund nur eine Ansicht hat; hier steht, was fehlt.
#
# STEHENDE REGEL (CLAUDE.md, Nutzervorgabe 2026-08-25): was in 2D sichtbar
# ist, gehoert in derselben Aenderung auch in 3D. Diese Liste ist die
# Restschuld aus der Zeit davor. **Sie darf nur schrumpfen.** Kommt eine neue
# einseitige Anzeigemethode dazu, die weder hier noch in NUR_EINE_ANZEIGE
# steht, schlaegt der Test fehl.
FEHLT_IM_3D = {
    "overlay_regions": "Regionsfarben als Flaeche. In 3D gaebe es sie ueber "
                       "denselben RGBA-Skin wie im Terrain-Reiter.",
    "overlay_region_grid": "Grenzen der neun Spielkarten.",
    "overlay_city_boundary_contour": "Stadtgrenzkontur.",
    "set_contour_reference_heightmap": "Hoehenlinien. Im 3D zeigt das Netz "
                                       "die Form zwar selbst, aber "
                                       "beschriftete Linien sind etwas "
                                       "anderes als eine Silhouette.",
    "overlay_plot_boundaries": "Grundstuecksgewebe des Regionalreiters. "
                               "Streitbar - tausende Parzellen koennten in "
                               "3D Pixelmatsch werden; steht hier, damit die "
                               "Entscheidung bewusst faellt statt zu "
                               "verschwinden.",
}


def run_einseitige_sind_begruendet():
    """
    Jede Anzeigemethode, die es nur auf EINER der beiden Klassen gibt, muss
    in NUR_EINE_ANZEIGE begruendet sein.

    Der Test oben findet nur Namen, die es NIRGENDS gibt. Der haeufigere und
    teurere Fall ist der halbe: die Methode existiert, aber nur in einer
    Ansicht - dann funktioniert das Overlay in 2D und fehlt im 3D, ohne dass
    irgendetwas meldet.
    """
    import gui.widgets.map_display_3d as m3
    from gui.widgets.map_display_2d import MapDisplay2D

    klassen_3d = [getattr(m3, n) for n in dir(m3) if n.startswith("MapDisplay3D")]

    def hat_3d(name):
        return any(hasattr(k, name) for k in klassen_3d)

    namen = {name for _d, _z, _o, name, _f in sammle_hasattr_namen()
             if name not in _KEIN_DISPLAY_NAME}
    einseitig = {}
    for name in sorted(namen):
        in2, in3 = hasattr(MapDisplay2D, name), hat_3d(name)
        if in2 != in3 and (in2 or in3):
            einseitig[name] = "nur 2D" if in2 else "nur 3D"

    print(f"{len(einseitig)} einseitige Anzeigemethoden gefunden")

    # DIE BEIDEN REGISTER DUERFEN SICH NICHT UEBERSCHNEIDEN.
    #
    # Erst falsch gebaut: die fuenf Schuldeintraege standen zusaetzlich in
    # NUR_EINE_ANZEIGE, und weil die Pruefung unten beide akzeptiert, war die
    # Schuldliste reine Zierde - sie herauszunehmen aenderte nichts. Gefunden
    # durch die Gegenprobe (Eintrag entfernen, Test muss fehlschlagen).
    doppelt = sorted(set(NUR_EINE_ANZEIGE) & set(FEHLT_IM_3D))
    if doppelt:
        print("")
        print("FEHLGESCHLAGEN - in BEIDEN Registern, damit wirkungslos:")
        for n in doppelt:
            print(f"  {n}")
        return False

    schuld = sorted(n for n in einseitig if n in FEHLT_IM_3D)
    if schuld:
        print(f"\n[--] OFFENE 3D-SCHULD ({len(schuld)}) - in 2D da, in 3D nicht:")
        for n in schuld:
            print(f"       {n:32s} {FEHLT_IM_3D[n][:60]}")
        print("     Stehende Regel dazu: CLAUDE.md, Abschnitt "
              "'was in 2D sichtbar ist, gehoert auch in 3D'.")

    veraltete_schuld = [n for n in FEHLT_IM_3D if n not in einseitig]
    if veraltete_schuld:
        print(f"[OK] {len(veraltete_schuld)} Schulden getilgt, gehoeren aus "
              f"FEHLT_IM_3D raus: {', '.join(sorted(veraltete_schuld))}")

    fehlend = [n for n in einseitig
               if n not in NUR_EINE_ANZEIGE and n not in FEHLT_IM_3D]
    if fehlend:
        print("")
        print("FEHLGESCHLAGEN - einseitig und NICHT begruendet:")
        print("(die hasattr-Weiche faellt in der anderen Ansicht still aus)")
        print("")
        for n in fehlend:
            print(f"  {n}  ({einseitig[n]})")
        print("")
        print("Entweder auf beiden Anzeigen anbieten (Regel in CLAUDE.md), "
              "oder in NUR_EINE_ANZEIGE begruenden, "
              "oder als Schuld in FEHLT_IM_3D eintragen.")
        return False

    veraltet = [n for n in NUR_EINE_ANZEIGE if n not in einseitig]
    if veraltet:
        print(f"[--] {len(veraltet)} Eintraege im Register sind nicht mehr "
              f"einseitig (koennen raus): {', '.join(sorted(veraltet))}")

    print(f"OK - alle {len(einseitig)} sind namentlich begruendet.")
    for n, seite in sorted(einseitig.items()):
        print(f"     {n:32s} {seite}")
    return True


# FUNKTIONEN, DEREN NAME NIE ALS TEXT-AUFRUF VORKOMMT, WEIL DAS FRAMEWORK SIE
# RUFT (Qt-Ueberschreibungen) - keine tote Funktion, nur ohne sichtbare
# Aufrufstelle im eigenen Quelltext. Wer hier steht, muss begruenden warum.
_RAHMEN_RUFT_SELBST = {
    "paintEvent", "resizeEvent", "showEvent", "closeEvent", "__init__",
}


def _wird_aufgerufen(funktionsname):
    """
    Ob `funktionsname` irgendwo im `gui/`-Baum als Aufruf ODER als Referenz
    (per Namen an eine Registrierungsstelle uebergeben, z.B. Dict-Eintrag in
    _OVERLAY_REGISTER oder Qt-`connect(self.foo)`-Slot) vorkommt - nicht nur
    definiert wird. Reiner Namensaufruf allein (`funktionsname(`) reicht
    nicht: `_siedlungen_2d`/`_siedlungen_3d`/`_fluesse_zeichnen` werden nie
    mit Klammer aufgerufen, sondern als Wert in _OVERLAY_REGISTER abgelegt und
    erst ueber `eintrag["2d"](...)` in _push_overlays() indirekt ausgefuehrt;
    `on_settlement_plot_live_update` haengt nur als `.connect(self.foo)`-Slot
    an einem Signal. Beides ist echte Erreichbarkeit, nur eben indirekt - die
    reine Namensnennung ausserhalb der eigenen `def`-Zeile ist der einzige
    gemeinsame Nenner, den eine Textsuche ohne Call-Graph dafuer pruefen kann.
    Grobe Textsuche (kein AST/Call-Graph), aber genug fuer
    genau die Fehlerklasse aus docs/OFFENE_PUNKTE.md 14.5: der
    Uebersichts-Reiter rief vier Methoden auf, die es nirgends im Programm
    gab - der umgekehrte Fall waere eine Methode, die es gibt, aber die
    NIEMAND ruft. Beides ist derselbe Fehler aus verschiedener Richtung, und
    CLAUDE.md nennt ihn ausdruecklich ("gruene Tests koennen eine tote
    Funktion verdecken", adaptive_terrain_mesh.py 2026-08-12).
    """
    referenz_muster = re.compile(r"\b" + re.escape(funktionsname) + r"\b")
    def_muster = re.compile(r"^\s*def\s+" + re.escape(funktionsname) + r"\s*\(")
    for pfad in Path("gui").rglob("*.py"):
        for zeile in pfad.read_text(encoding="utf-8").split("\n"):
            if def_muster.match(zeile):
                continue
            if referenz_muster.search(zeile):
                return True
    return False


def run_zweige_sind_erreichbar():
    """
    TICKET #7 (docs/OFFENE_PUNKTE.md 14.2): `run()` oben prueft nur, OB eine
    per hasattr angesprochene Anzeigemethode irgendwo existiert. Das faengt
    einen Tippfehler im Methodennamen - aber nicht den Fall, dass die ganze
    Funktion, in der die hasattr-Weiche steht, nie aufgerufen wird. Dann ist
    der Methodenname korrekt, die Anzeigeklasse hat ihn auch, und trotzdem
    erscheint das Overlay nie: die Weiche wird schlicht nie erreicht.

    Ohne diese Pruefung wiederholt sich exakt der Fall aus 14.5
    (`gui/tabs/overview_tab.py` rief vier Methoden auf, die es nirgends gab)
    nur umgekehrt: eine Methode MIT hasattr-Weiche, die selbst nie gerufen
    wird - `run()` allein sieht das nicht, weil sie ja korrekt geschrieben
    ist, nur eben totes Gewebe.
    """
    betroffene_funktionen = {}
    for datei, zeile, objekt, name, funktion in sammle_hasattr_namen():
        if name in _KEIN_DISPLAY_NAME or not funktion:
            continue
        if not any(v in objekt for v in
                   ("display", "ziel", "anzeige", "current_display", "map_display")):
            continue
        betroffene_funktionen.setdefault(funktion, []).append((datei, zeile, name))

    unerreichbar = []
    for funktion, stellen in sorted(betroffene_funktionen.items()):
        if funktion in _RAHMEN_RUFT_SELBST:
            continue
        if not _wird_aufgerufen(funktion):
            unerreichbar.append((funktion, stellen))

    print(f"{len(betroffene_funktionen)} Funktionen mit mindestens einer "
          f"Anzeige-hasattr-Weiche gefunden")

    if unerreichbar:
        print("\nFEHLGESCHLAGEN - diese Funktionen werden NIRGENDS im "
              "gui/-Baum aufgerufen, die hasattr-Weiche darin wird also nie")
        print("erreicht, egal ob der Methodenname stimmt:\n")
        for funktion, stellen in unerreichbar:
            datei, zeile, name = stellen[0]
            print(f"  {funktion}()  ({datei}:{zeile}, hasattr(..., {name!r}))")
        print("\nEntweder verdrahten (aufrufen lassen) oder loeschen "
              "(siehe 14.5: loeschen oder bauen), oder in "
              "_RAHMEN_RUFT_SELBST begruenden, falls das Framework sie ruft.")
        return False

    print("OK - jede Funktion mit einer Anzeige-hasattr-Weiche wird auch "
          "tatsaechlich irgendwo aufgerufen.")
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
        "einseitige_sind_begruendet": run_einseitige_sind_begruendet(),
        "zweige_sind_erreichbar": run_zweige_sind_erreichbar(),
        "regionaltab_zeichnet_basiskarte": run_regionaltab_zeichnet_basiskarte(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
