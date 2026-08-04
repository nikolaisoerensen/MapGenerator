"""
Path: tests/smoke_test_shader_paths.py

Prueft, dass JEDE Shader-Datei, die managers/shader_manager.py per
get_program() anfordert, an der erwarteten Stelle liegt.

WARUM ES DIESEN TEST GIBT. Beim Aufraeumen am 2026-07-30 wanderte
shader_manager.py von gui/OldManagers/ nach managers/, also eine Ebene nach
oben. Die Berechnung von SHADERS_ROOT ging weiterhin ZWEI Ebenen hoch und
zeigte damit auf ein Verzeichnis NEBEN dem Projekt:

    C:\\...\\Projects\\Python\\shaders\\terrain\\noiseGeneration.comp
                            ^^^^^^^^ "MapGenerator" fehlt

Nichts davon war zu sehen. Jede GPU-Operation faengt ihren Fehler ab und
faellt still auf den CPU-Pfad zurueck; im Log stand eine WARNING je Aufruf,
und das Programm lief scheinbar normal weiter. Die gesamte Pipeline rechnete
ohne GPU.

Die Pruefung davor hatte nur IMPORTE getestet - und eine Pfadberechnung aus
__file__ ist beim Import unsichtbar. Genau diese Luecke schliesst dieser Test.

Zwei Zusicherungen:

  1. SHADERS_ROOT zeigt auf das shaders/-Verzeichnis DIESES Projekts.
  2. Jede in get_program("kategorie", "operation") genannte Datei existiert.
     Die Aufrufstellen werden aus dem SYNTAXBAUM gelesen, nicht per
     Textsuche - SPEZIFIKATION §5.2 fuehrt den Fall, in dem ein eigener
     Erklaerkommentar von der Suche gefunden wurde, die den Code pruefen
     sollte.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_shader_paths.py
"""

import ast
import io
import os
import sys

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

PROJEKT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUELLE = os.path.join(PROJEKT, "managers", "shader_manager.py")


def geforderte_shader():
    """
    Alle get_program("kategorie", "operation")-Aufrufe mit festen Zeichenketten,
    aus dem Syntaxbaum gelesen.
    """
    baum = ast.parse(io.open(QUELLE, encoding="utf-8").read(), filename=QUELLE)
    gefunden = []
    for knoten in ast.walk(baum):
        if not isinstance(knoten, ast.Call):
            continue
        ziel = knoten.func
        if not (isinstance(ziel, ast.Attribute) and ziel.attr == "get_program"):
            continue
        args = knoten.args
        if len(args) < 2:
            continue
        if not all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                   for a in args[:2]):
            continue
        gefunden.append((args[0].value, args[1].value, knoten.lineno))
    return sorted(set(gefunden))


def lauf():
    fehler = []

    # ---------- 1: SHADERS_ROOT ----------
    from managers.shader_manager import SHADERS_ROOT
    erwartet = os.path.join(PROJEKT, "shaders")
    wurzel_ok = os.path.normpath(SHADERS_ROOT) == os.path.normpath(erwartet)
    if not wurzel_ok:
        fehler.append("SHADERS_ROOT ist %s, erwartet %s" % (SHADERS_ROOT, erwartet))
    print("1. SHADERS_ROOT zeigt ins Projekt ............... %s"
          % ("ok" if wurzel_ok else "FEHLER"))
    print("   %s" % SHADERS_ROOT)

    # ---------- 2: jede angeforderte Datei ----------
    angefordert = geforderte_shader()
    fehlend = []
    for kategorie, operation, zeile in angefordert:
        pfad = os.path.join(SHADERS_ROOT, kategorie, "%s.comp" % operation)
        if not os.path.isfile(pfad):
            fehlend.append("%s/%s.comp (shader_manager.py:%d)"
                           % (kategorie, operation, zeile))
    if fehlend:
        fehler.append("%d angeforderte Shader fehlen" % len(fehlend))
    print("2. %d angeforderte Shader-Dateien ............... %s"
          % (len(angefordert), "alle da" if not fehlend else "FEHLEN"))
    for f in fehlend:
        print("     fehlt: %s" % f)

    # Uebersicht je Kategorie
    je_kategorie = {}
    for kategorie, _, _ in angefordert:
        je_kategorie[kategorie] = je_kategorie.get(kategorie, 0) + 1
    print("   %s" % ", ".join("%s %d" % (k, n)
                              for k, n in sorted(je_kategorie.items())))

    print()
    if fehler:
        for f in fehler:
            print("  FEHLER: %s" % f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
