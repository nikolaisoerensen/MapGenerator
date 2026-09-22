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
     Textsuche - 03_ARBEITSREGELN.md 1.2 fuehrt den Fall, in dem ein
     eigener Erklaerkommentar von der Suche gefunden wurde, die den Code
     pruefen sollte.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_shader_paths.py
"""

import ast
import io
import os
import sys

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

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

    fehler.extend(anzeige_shader_unabhaengig_vom_arbeitsverzeichnis())
    fehler.extend(wegband_shader_passt_zusammen())

    print()
    if fehler:
        for f in fehler:
            print("  FEHLER: %s" % f)
        return 1
    print("Alle Zusicherungen erfuellt.")
    return 0


# Die sechs Dateien, die MapDisplay3D._load_shaders() anfordert.
ANZEIGE_SHADER = ("terrain.vert", "terrain.frag",
                  "simple.vert", "simple.frag",
                  "wind_vector.vert", "wind_vector.frag")


def anzeige_shader_unabhaengig_vom_arbeitsverzeichnis():
    """
    Dritte Zusicherung: die Shader des 3D-Displays muessen auch dann gefunden
    werden, wenn das Programm NICHT aus dem Projektstamm gestartet wurde.

    WARUM DAS HIER STEHT. `MapDisplay3D._load_shader_from_file()` suchte bis
    zum 2026-08-16 ausschliesslich relativ zum Arbeitsverzeichnis. Beim Start
    von `tools/mesh_werkstatt.py` (Arbeitsverzeichnis also `tools/`) wurde
    keine einzige der sechs Dateien gefunden - und danach zeichnete
    `_prepare_rendering()` ohne Shaderprogramm weiter. `glDrawElements` ohne
    aktives Programm ist im Core-Profile undefiniert; der Prozess starb hart
    mit 0xC0000409, ganz ohne Python-Traceback.

    Es ist dieselbe Lektion wie oben fuer SHADERS_ROOT, nur an einer zweiten
    Stelle - und wieder war sie beim blossen Importieren unsichtbar.
    """
    print("3. Shader des 3D-Displays, aus fremdem Arbeitsverzeichnis")
    fehler = []
    import gui.widgets.map_display_3d as anzeige

    wurzel = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(anzeige.__file__))))
    print("   ermittelte Projektwurzel: %s" % wurzel)
    if not os.path.isdir(os.path.join(wurzel, "shaders", "3d_display")):
        fehler.append("Projektwurzel aus __file__ zeigt nicht auf dieses "
                      "Projekt: %s" % wurzel)
        return fehler

    alt = os.getcwd()
    try:
        # Irgendwo hin, nur nicht in den Projektstamm
        os.chdir(os.path.dirname(wurzel))
        print("   Arbeitsverzeichnis fuer den Test: %s" % os.getcwd())
        for name in ANZEIGE_SHADER:
            pfad = os.path.join(wurzel, "shaders", "3d_display", name)
            if os.path.exists(pfad):
                print("   [OK]   %s" % name)
            else:
                print("   [FEHLT] %s" % name)
                fehler.append("Anzeige-Shader nicht auffindbar: %s" % name)
    finally:
        os.chdir(alt)

    # Und die Suchliste im Code muss den absoluten Pfad wirklich enthalten -
    # sonst stimmt zwar die Rechnung oben, der Code benutzt sie aber nicht.
    import inspect
    quelltext = inspect.getsource(anzeige.MapDisplay3D._load_shader_from_file)
    if "os.path.join(wurzel" not in quelltext:
        fehler.append("_load_shader_from_file sucht nicht vom Projektstamm aus")
    else:
        print("   [OK]   _load_shader_from_file sucht auch vom Projektstamm aus")

    quelltext = inspect.getsource(anzeige.MapDisplay3D._prepare_rendering)
    if "self.shader_program is None:" not in quelltext:
        fehler.append("_prepare_rendering zeichnet moeglicherweise ohne "
                      "Shaderprogramm weiter (Absturzgefahr)")
    else:
        print("   [OK]   ohne Shaderprogramm wird nicht gezeichnet")
    return fehler




def wegband_shader_passt_zusammen():
    """
    Vierte Zusicherung: die Varyings von wegband.vert und wegband.frag
    muessen zusammenpassen, und jedes Vertexattribut braucht einen Platz.

    WARUM STATISCH. Ein Link-Fehler waere zur Laufzeit unsichtbar:
    `wegband_shader_program` bliebe None und `_render_wegbaender()` steigt
    aus - man saehe einfach keine Wege und hielte es fuer "es gibt keine".
    Der Fehler laesst sich hier ohne OpenGL finden, weil er reine
    Textuebereinstimmung ist. Anlass war der Umbau am 2026-08-24, der ein
    drittes Attribut (`deckung`) und ein drittes Varying eingefuehrt hat.
    """
    print("4. wegband.vert und wegband.frag passen zusammen")
    fehler = []
    import re
    wurzel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ordner = os.path.join(wurzel, "shaders", "3d_display")
    try:
        vert = open(os.path.join(ordner, "wegband.vert"), encoding="utf-8").read()
        frag = open(os.path.join(ordner, "wegband.frag"), encoding="utf-8").read()
    except OSError as e:
        return ["wegband-Shader nicht lesbar: %s" % e]

    # NAME als Schluessel, nicht Typ. Erst falsch herum gebaut - `dict()` auf
    # (typ, name)-Paare macht den TYP zum Schluessel, und damit verschwand
    # `FragPos` hinter `Normal`, weil beide vec3 sind. Die Pruefung sah gruen
    # aus und pruefte zwei von drei Varyings.
    ausgaenge = {n: t for t, n in
                 re.findall(r"^out\s+(\w+)\s+(\w+);", vert, re.M)}
    eingaenge = {n: t for t, n in
                 re.findall(r"^in\s+(\w+)\s+(\w+);", frag, re.M)}
    for name, typ in eingaenge.items():
        if name not in ausgaenge:
            fehler.append("wegband.frag liest '%s', wegband.vert schreibt es nicht" % name)
        elif ausgaenge[name] != typ:
            fehler.append("Typ von '%s': vert %s, frag %s"
                          % (name, ausgaenge[name], typ))
        else:
            print("   [OK]   %s %s" % (typ, name))

    plaetze = re.findall(r"layout\s*\(location\s*=\s*(\d+)\)\s*in\s+(\w+)\s+(\w+);", vert)
    belegt = sorted(int(p) for p, _t, _n in plaetze)
    if belegt != list(range(len(belegt))):
        fehler.append("Attributplaetze nicht lueckenlos ab 0: %s" % belegt)
    else:
        print("   [OK]   %d Vertexattribute, Plaetze %s"
              % (len(belegt), belegt))

    # Die Python-Seite muss genauso viele Attribute binden.
    anzeige = open(os.path.join(wurzel, "gui", "widgets", "map_display_3d.py"),
                   encoding="utf-8").read()
    if "stride = 7 * 4" not in anzeige:
        fehler.append("map_display_3d bindet keinen 7-float-Stride fuer die "
                      "Wegbaender - passt nicht zu %d Attributen" % len(belegt))
    else:
        print("   [OK]   map_display_3d bindet 7 float je Vertex")
    return fehler


if __name__ == "__main__":
    raise SystemExit(lauf())
