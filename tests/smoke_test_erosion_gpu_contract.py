"""
Path: tests/smoke_test_erosion_gpu_contract.py

Statischer Vertrag zwischen den Erosions-Shadern (shaders/erosion/*.comp) und
ihrem Dispatcher (shader_manager._dispatch_hydraulic_field).

WOZU: GPU-Code ist in diesem Projekt nicht headless ausfuehrbar (siehe
CLAUDE.md) - ein Fehler faellt erst in der laufenden App auf, und zwar oft
nicht als Absturz, sondern als falsches Bild. Die haeufigste Ursache ist
banal: ein Uniform heisst im Shader anders als im Dispatcher, oder ein
Binding-Index passt nicht. Ein nicht gesetztes Uniform ist in GLSL schlicht 0 -
die Simulation laeuft dann durch und liefert Unsinn.

Genau das laesst sich ohne GL-Kontext pruefen, indem man beide Seiten parst:

  (A) Jedes Uniform, das ein Shader deklariert, wird vom Dispatcher gesetzt -
      und umgekehrt wird kein Uniform gesetzt, das es nicht gibt.
  (B) Jedes Image-Binding, das ein Shader deklariert, wird gebunden - mit
      passendem Index und passendem Format.
  (C) Die Formeln stimmen mit dem CPU-Pfad ueberein, soweit sie sich als
      Konstanten vergleichen lassen.

Das ersetzt keinen Live-Test, faengt aber die Fehlerklasse ab, die man sonst
erst nach einem Vier-Minuten-Lauf am Ergebnis sieht.

Aufruf: .venv\\Scripts\\python.exe tests/smoke_test_erosion_gpu_contract.py
"""

import io
import os
import re
import sys

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

PROJECT = r"C:\Lokale Dateien\Projects\Python\MapGenerator"
SHADER_DIR = os.path.join(PROJECT, "shaders", "erosion")
DISPATCHER = os.path.join(PROJECT, "managers", "shader_manager.py")


def check(label, condition):
    print(("[OK] " if condition else "[FAIL] ") + label)
    return bool(condition)


def read(path):
    return io.open(path, encoding="utf-8").read()


# -----------------------------------------------------------------------------
# Shader-Seite parsen
# -----------------------------------------------------------------------------

UNIFORM_RE = re.compile(r"^\s*uniform\s+\w+\s+(u_\w+)\s*;", re.MULTILINE)
BINDING_RE = re.compile(
    r"layout\s*\(\s*(\w+)\s*,\s*binding\s*=\s*(\d+)\s*\)\s*uniform\s+(?:readonly\s+|writeonly\s+)?[iu]?image2D\s+(\w+)")


def parse_shader(path):
    source = read(path)
    uniforms = set(UNIFORM_RE.findall(source))
    bindings = {int(index): (fmt, name) for fmt, index, name in BINDING_RE.findall(source)}
    return uniforms, bindings, source


# -----------------------------------------------------------------------------
# Dispatcher-Seite parsen
# -----------------------------------------------------------------------------

def parse_dispatcher():
    """
    Zerlegt _dispatch_hydraulic_field in seine Dispatch-Bloecke. Ein Block
    endet mit glDispatchCompute und enthaelt davor die glBindImageTexture-
    Aufrufe, das glUseProgram und die _set_uniforms-Schluessel.
    """
    source = read(DISPATCHER)
    start = source.index("def _dispatch_hydraulic_field(")
    end = source.index("\nDISPATCH_TABLE = {", start)
    body = source[start:end]

    blocks = []
    for segment in body.split("gl.glDispatchCompute")[:-1]:
        match = re.findall(r"gl\.glUseProgram\((\w+)_program\)", segment)
        if not match:
            continue
        program = match[-1]
        # Nur die Binds NACH dem letzten vorherigen Dispatch zaehlen.
        tail = segment[segment.rindex("gl.glUseProgram") - 2000:] \
            if len(segment) > 2000 else segment
        binds = {int(index): fmt for index, fmt in re.findall(
            r"gl\.glBindImageTexture\((\d+),\s*\w+,\s*0,\s*gl\.GL_FALSE,\s*0,\s*"
            r"gl\.GL_(?:READ_ONLY|WRITE_ONLY|READ_WRITE),\s*gl\.GL_(\w+)\)", tail)}
        uniform_block = re.search(r"_set_uniforms\(\w+_program,\s*\{(.*?)\}\)", segment, re.DOTALL)
        uniforms = set(re.findall(r'"(u_\w+)"', uniform_block.group(1))) if uniform_block else set()
        blocks.append({"program": program, "binds": binds, "uniforms": uniforms})
    return blocks


# Zuordnung Dispatcher-Variablenname -> Shader-Datei.
PROGRAM_TO_SHADER = {
    "flux": "pipeFlux.comp",
    "depth": "pipeDepth.comp",
    "erode": "erodeDeposit.comp",
    "transport": "sedimentTransport.comp",
    "thermal_flux": "thermalFlux.comp",
    "thermal_apply": "thermalApply.comp",
    "smooth": "smooth.comp",
}

# GLSL-Layoutformat -> OpenGL-Konstantenname im Dispatcher.
FORMAT_MAP = {"r32f": "R32F", "rg32f": "RG32F", "rgba32f": "RGBA32F", "r32i": "R32I"}


def run_every_shader_is_dispatched():
    """Jede Shader-Datei im Verzeichnis muss auch benutzt werden - eine
    verwaiste Datei ist entweder toter Code oder ein vergessener Pass."""
    on_disk = {name for name in os.listdir(SHADER_DIR) if name.endswith(".comp")}
    referenced = set(PROGRAM_TO_SHADER.values())
    ok = check("keine unbenutzte Shader-Datei (gefunden {})".format(sorted(on_disk - referenced)),
               not (on_disk - referenced))
    ok &= check("keine fehlende Shader-Datei (vermisst {})".format(sorted(referenced - on_disk)),
                not (referenced - on_disk))
    return ok


# GLSL-430-Schluesselwoerter und RESERVIERTE Woerter, die als Bezeichner
# verboten sind. Nicht die vollstaendige Liste der Spezifikation, sondern die
# Teilmenge, die man versehentlich als Variablennamen waehlt.
#
# Der Anlass: eine Zwischengroesse hiess `common`. Das ist reserviert, der
# Shader scheiterte beim Kompilieren - aber erst auf der GPU, also genau dort,
# wo headless niemand hinschaut. In der laufenden App kam nur
# "ERROR: 0:74: 'common' : reserved word", danach fiel der Lauf auf die CPU
# zurueck und lief in den Timeout.
_GLSL_RESERVED = {
    "common", "partition", "active", "asm", "class", "union", "enum", "typedef",
    "template", "this", "resource", "goto", "inline", "noinline", "public",
    "static", "extern", "external", "interface", "long", "short", "half",
    "fixed", "unsigned", "superp", "input", "output", "filter", "sizeof",
    "cast", "namespace", "using", "row_major", "packed", "attribute", "varying",
}

_DECLARATION_RE = re.compile(
    r"(?:float|int|uint|bool|vec[234]|ivec[234]|uvec[234]|bvec[234]|mat[234])\s+(\w+)\s*(?:=|;|\[)")


def run_no_reserved_words():
    """
    Kein Bezeichner in den Shadern darf ein reserviertes GLSL-Wort sein.

    Das laesst sich ohne GL-Kontext pruefen und faengt genau die Fehlerklasse,
    die sonst erst beim Kompilieren auf der GPU auffaellt - und dort in einem
    Fallback verschwindet, statt den Lauf zu stoppen.
    """
    ok = True
    for name in sorted(PROGRAM_TO_SHADER.values()):
        source = read(os.path.join(SHADER_DIR, name))
        declared = set(_DECLARATION_RE.findall(source))
        clashes = declared & _GLSL_RESERVED
        ok &= check("{}: kein reserviertes Wort als Bezeichner (gefunden: {})".format(
            name, sorted(clashes)), not clashes)
    return ok


def run_uniforms_match():
    """
    Jedes deklarierte Uniform wird gesetzt, und keins wird gesetzt, das es
    nicht gibt.

    Die zweite Richtung ist die wichtigere: ein nicht gesetztes Uniform ist in
    GLSL 0. Ein Tippfehler im Dispatcher fuehrt also nicht zu einem Fehler,
    sondern zu einer Simulation, in der z.B. die Erosionsstaerke stillschweigend
    null ist.
    """
    blocks = parse_dispatcher()
    ok = check("alle sieben Passes im Dispatcher gefunden (gefunden {})".format(len(blocks)),
               len(blocks) == 7)

    for block in blocks:
        shader_name = PROGRAM_TO_SHADER.get(block["program"])
        if shader_name is None:
            ok &= check("unbekanntes Programm '{}'".format(block["program"]), False)
            continue
        declared, _, _ = parse_shader(os.path.join(SHADER_DIR, shader_name))
        missing = declared - block["uniforms"]
        extra = block["uniforms"] - declared
        ok &= check("{}: alle deklarierten Uniforms werden gesetzt (fehlen: {})".format(
            shader_name, sorted(missing)), not missing)
        ok &= check("{}: kein Uniform gesetzt, das der Shader nicht kennt (extra: {})".format(
            shader_name, sorted(extra)), not extra)
    return ok


def run_uniform_types_match():
    """
    Jedes als `uniform int` deklarierte Uniform muss vom Dispatcher auch als
    int gesetzt werden - und keins, das float ist, faelschlich als int.

    Der Anlass: shader_manager._INT_UNIFORM_NAMES war eine HANDGEPFLEGTE Liste.
    Das neue `u_variant` fehlte darin, wurde deshalb mit glUniform1f statt
    glUniform1i gesetzt, und der gesamte Erosions-Dispatch brach mit
    GL_INVALID_OPERATION ab. Sichtbar erst in der laufenden App - und die alte
    Liste enthielt umgekehrt ein `u_depth_tests`, das in keinem Shader mehr
    existierte. Eine Kopie driftet in beide Richtungen.

    Die Liste wird inzwischen aus den Shader-Quellen abgeleitet; dieser Test
    haelt fest, dass das auch so bleibt.
    """
    from managers.shader_manager import _INT_UNIFORM_NAMES

    int_pattern = re.compile(r"^\s*uniform\s+int\s+(\w+)\s*;", re.MULTILINE)
    float_pattern = re.compile(r"^\s*uniform\s+float\s+(\w+)\s*;", re.MULTILINE)

    ok = True
    for name in sorted(PROGRAM_TO_SHADER.values()):
        source = read(os.path.join(SHADER_DIR, name))
        declared_int = set(int_pattern.findall(source))
        declared_float = set(float_pattern.findall(source))

        missing = declared_int - _INT_UNIFORM_NAMES
        ok &= check("{}: alle int-Uniforms werden als int gesetzt (fehlen: {})".format(
            name, sorted(missing)), not missing)

        wrong = declared_float & _INT_UNIFORM_NAMES
        ok &= check("{}: kein float-Uniform wird als int gesetzt (betroffen: {})".format(
            name, sorted(wrong)), not wrong)
    return ok


def run_bindings_match():
    """Jedes Image-Binding wird mit passendem Index UND passendem Format gebunden."""
    blocks = parse_dispatcher()
    ok = True
    for block in blocks:
        shader_name = PROGRAM_TO_SHADER.get(block["program"])
        if shader_name is None:
            continue
        _, bindings, _ = parse_shader(os.path.join(SHADER_DIR, shader_name))

        missing = set(bindings) - set(block["binds"])
        extra = set(block["binds"]) - set(bindings)
        ok &= check("{}: alle {} Bindings werden gebunden (fehlen: {})".format(
            shader_name, len(bindings), sorted(missing)), not missing)
        ok &= check("{}: keine ueberzaehligen Bindings (extra: {})".format(
            shader_name, sorted(extra)), not extra)

        for index, (glsl_format, name) in sorted(bindings.items()):
            expected = FORMAT_MAP.get(glsl_format)
            actual = block["binds"].get(index)
            ok &= check("{}: Binding {} ({}) hat Format {} (gebunden: {})".format(
                shader_name, index, name, expected, actual), actual == expected)
    return ok


def run_constants_match_cpu():
    """
    Die Konstanten, die der Dispatcher an die Shader durchreicht, muessen aus
    dem CPU-Modell kommen - nicht als Zahlenliteral daneben stehen.

    Eine zweite Kopie derselben Kalibrierung ist der sicherste Weg, dass CPU-
    und GPU-Pfad nach der ersten Nachjustierung auseinanderlaufen.
    """
    from core.erosion_generator import HydraulicFieldSimulator
    from core.water_generator import ThermalErosionSystem

    source = read(DISPATCHER)
    start = source.index("def _dispatch_hydraulic_field(")
    end = source.index("\nDISPATCH_TABLE = {", start)
    body = source[start:end]

    ok = check("der Dispatcher enthaelt keine eigenen Kalibrierungs-Literale "
               "(alle Werte kommen ueber `inputs`)",
               not re.search(r"=\s*(?:0\.5|1\.0|2\.0|3\.0|8\.0)\s*$", body, re.MULTILINE))

    # Der Aufrufer muss jeden Wert, den der Dispatcher liest, auch mitgeben.
    required = set(re.findall(r'inputs\["(\w+)"\]', body)) | set(
        re.findall(r'inputs\.get\("(\w+)"\)', body))
    # Der Rumpf von _simulate_gpu() - von der Methodendefinition bis zur
    # naechsten Methode auf derselben Einrueckung.
    simulator_source = read(os.path.join(PROJECT, "core", "erosion_generator.py"))
    method_start = simulator_source.index("    def _simulate_gpu(")
    method_end = simulator_source.index("\n    def ", method_start + 10)
    method_body = simulator_source[method_start:method_end]
    # Zwei Schreibweisen zaehlen als "geliefert": als Schluessel im
    # Dict-Literal (`"name": wert`) und als nachtraegliche Zuweisung
    # (`request["name"] = wert`). Die zweite braucht es fuer alles, was sich
    # von Abschnitt zu Abschnitt aendert - chunk_steps und der durchgereichte
    # Zustand.
    provided = set(re.findall(r'"(\w+)":', method_body))
    provided |= set(re.findall(r'\[\s*"(\w+)"\s*\]\s*=', method_body))
    missing = required - provided
    ok &= check("der Simulator liefert jeden vom Dispatcher gelesenen Wert "
                "(fehlen: {})".format(sorted(missing)), not missing)

    # Stichproben: die drei Konstanten, deren Drift am teuersten waere.
    ok &= check("CPU kennt REFERENCE_SPECIFIC_DISCHARGE",
                hasattr(HydraulicFieldSimulator, "REFERENCE_SPECIFIC_DISCHARGE"))
    ok &= check("CPU kennt MIN_SLOPE_FACTOR",
                hasattr(HydraulicFieldSimulator, "MIN_SLOPE_FACTOR"))
    ok &= check("Thermal-Konstanten kommen aus ThermalErosionSystem",
                hasattr(ThermalErosionSystem, "TRANSFER_RATE")
                and hasattr(ThermalErosionSystem, "CAP_RELIEF_FRACTION"))
    return ok


def run_shaders_are_wellformed():
    """Grobe Syntax-Plausibilitaet ohne GL-Kontext: Version, main(), Klammern."""
    ok = True
    for name in sorted(PROGRAM_TO_SHADER.values()):
        source = read(os.path.join(SHADER_DIR, name))
        ok &= check("{}: #version 430".format(name), source.lstrip().startswith("#version 430"))
        ok &= check("{}: hat main()".format(name), "void main()" in source)
        ok &= check("{}: Klammern ausgeglichen".format(name),
                    source.count("{") == source.count("}")
                    and source.count("(") == source.count(")"))
        ok &= check("{}: local_size deklariert".format(name), "local_size_x" in source)
        # Kommentare in GLSL sind //, nicht # - ein einmal gemachter Fehler.
        code_lines = [line for line in source.splitlines()
                      if line.strip().startswith("#") and not line.strip().startswith("#version")]
        ok &= check("{}: keine Python-Kommentare (#) im GLSL".format(name), not code_lines)
    return ok


def main():
    tests = [
        ("every_shader_is_dispatched", run_every_shader_is_dispatched),
        ("shaders_are_wellformed", run_shaders_are_wellformed),
        ("no_reserved_words", run_no_reserved_words),
        ("uniforms_match", run_uniforms_match),
        ("uniform_types_match", run_uniform_types_match),
        ("bindings_match", run_bindings_match),
        ("constants_match_cpu", run_constants_match_cpu),
    ]
    results = {}
    for name, func in tests:
        print("\n=== {} ===".format(name))
        results[name] = func()

    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print("{}: {}".format(name, "PASS" if passed else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
