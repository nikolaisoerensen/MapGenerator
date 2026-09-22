"""
Path: tests/smoke_test_elevation_scale_2d_3d.py

Prueft, dass die Hoehen-Farbskala des 3D-Gelaendemeshs (getTerrainColor() in
shaders/3d_display/terrain.frag, gesetzt in
MapDisplay3D._render_terrain_base()) dieselbe FESTE Spanne benutzt wie die
2D-Ansicht (CanvasSettings.CANVAS_2D["elevation_vmin"/"elevation_vmax"],
gui/config/gui_default.py), statt bei jedem Rendern neu aus
np.max(self.heightmap) zu berechnen.

Hintergrund: bis 2026-09-22 berechnete _render_terrain_base()
    max_height = np.max(self.heightmap) * self.terrain_height_scale
und schickte das als "maxHeight"-Uniform an den Shader. Bei einer flachen
Karte war max_height klein, bei einer sehr hohen Karte gross - die 3D-
Farbskala "atmete" mit der jeweiligen Karte statt fest zu stehen wie in 2D
(map_display_2d.py nutzt bereits die feste elevation_vmin/vmax-Spanne).

Warum das ohne echten GL-Kontext pruefbar ist: hier interessieren nur die
WERTE, die _render_terrain_base() an glUniform1f() uebergibt - keine
tatsaechliche Rasterung/Zeichnung (die braeuchte ein echtes Fenster, siehe
CLAUDE.md "Verifying backend/generator changes"). OpenGL.GL wird deshalb im
Modul map_display_3d komplett durch einen Platzhalter ersetzt, der
glGetUniformLocation/glUniform1f mitschreibt und jeden anderen Aufruf
(VAO/Buffer/Draw-Calls, Konstanten wie GL_TRIANGLES) wirkungslos durchlaesst.
_bind_shadow_texture() wird separat gestubbt, weil sie eigene Textur-Logik
hat, die mit dieser Frage nichts zu tun hat.
"""
import sys

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

import numpy as np
from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from gui.config.gui_default import CanvasSettings  # noqa: E402
from gui.widgets import map_display_3d  # noqa: E402
from gui.widgets.map_display_3d import MapDisplay3D  # noqa: E402


def check(label, condition):
    print("[{}] {}".format("OK" if condition else "FAIL", label))
    return bool(condition)


class _FakeGL:
    """Ersetzt OpenGL.GL fuer diesen Test komplett - kein echter Kontext
    noetig. glGetUniformLocation/glUniform1f werden mitgeschrieben, jeder
    andere Aufruf (VAO/Buffer/Draw-Calls, Konstanten wie GL_TRIANGLES) ist
    ein wirkungsloser Platzhalter, weil nur die Uniform-WERTE interessieren."""

    def __init__(self):
        self.uniform_werte = {}
        self._namen_je_location = {}
        self._naechste_location = 1

    def glGetUniformLocation(self, program, name):
        location = self._naechste_location
        self._naechste_location += 1
        self._namen_je_location[location] = name
        return location

    def glUniform1f(self, location, value):
        name = self._namen_je_location.get(location)
        if name is not None:
            self.uniform_werte[name] = value

    def __getattr__(self, item):
        return lambda *args, **kwargs: None


def _render_und_hole_uniforms(heightmap, terrain_height_scale):
    """Baut ein MapDisplay3D ohne echten GL-Kontext auf (wie
    smoke_test_camera_controls.py es fuer die Kamerarechnung tut), faelscht
    OpenGL.GL im Modul weg und liest danach ab, welche Uniform-Werte
    _render_terrain_base() tatsaechlich gesetzt haette."""
    fake_gl = _FakeGL()
    original_gl = map_display_3d.gl
    map_display_3d.gl = fake_gl
    try:
        display = MapDisplay3D()
        display.heightmap = heightmap
        display.terrain_height_scale = terrain_height_scale
        display.shader_program = 1  # truthy Platzhalter, kein echtes Programm
        display.vertex_buffer = object()
        display.mesh_indices = np.arange(3, dtype=np.uint32)
        display._bind_shadow_texture = lambda: None
        display._render_terrain_base()
        return fake_gl.uniform_werte
    finally:
        map_display_3d.gl = original_gl


def run_max_height_is_fixed_not_map_dependent():
    scale = 0.0037  # realistische Groessenordnung (siehe terrain_height_scale-Berechnung)
    flach = np.full((32, 32), 5.0, dtype=np.float32)            # fast ebene Karte
    hochgebirge = np.full((32, 32), 8000.0, dtype=np.float32)   # sehr hohe Karte

    uniforms_flach = _render_und_hole_uniforms(flach, scale)
    uniforms_hoch = _render_und_hole_uniforms(hochgebirge, scale)

    erwartetes_max = CanvasSettings.CANVAS_2D["elevation_vmax"] * scale
    erwartetes_min = CanvasSettings.CANVAS_2D["elevation_vmin"]

    ok = check("maxHeight bei flacher Karte == fester 2D-Spanne * heightScale "
               "({:.4f})".format(erwartetes_max),
               abs(uniforms_flach.get("maxHeight", float("nan")) - erwartetes_max) < 1e-9)
    ok &= check("maxHeight bei Hochgebirgs-Karte == DIESELBE feste Spanne, "
                "NICHT np.max(heightmap)*scale ({:.4f})".format(erwartetes_max),
                abs(uniforms_hoch.get("maxHeight", float("nan")) - erwartetes_max) < 1e-9)
    ok &= check("maxHeight ist fuer beide Karten identisch ({} == {})".format(
                uniforms_flach.get("maxHeight"), uniforms_hoch.get("maxHeight")),
                uniforms_flach.get("maxHeight") == uniforms_hoch.get("maxHeight"))

    ok &= check("minHeight bei flacher Karte == elevation_vmin ({:.1f})".format(
                erwartetes_min),
                abs(uniforms_flach.get("minHeight", float("nan")) - erwartetes_min) < 1e-9)
    ok &= check("minHeight bei Hochgebirgs-Karte == DIESELBE elevation_vmin ({:.1f})".format(
                erwartetes_min),
                abs(uniforms_hoch.get("minHeight", float("nan")) - erwartetes_min) < 1e-9)
    return ok


def run_shader_source_uses_configurable_min_height():
    """Gegenprobe gegen den Shader-Quelltext: die alte hartkodierte 400.0 fuer
    die Tiefsee-Faerbung darf nicht mehr dastehen, minHeight muss als Uniform
    deklariert und in getTerrainColor() tatsaechlich benutzt sein."""
    frag_path = _os.path.join(_PROJEKTWURZEL, "shaders", "3d_display", "terrain.frag")
    with open(frag_path, "r", encoding="utf-8") as f:
        source = f.read()
    ok = check("terrain.frag deklariert 'uniform float minHeight;'",
                "uniform float minHeight;" in source)
    ok &= check("getTerrainColor() benutzt minHeight in der Tiefsee-Formel",
                "rohHoehe / min(minHeight" in source)
    ok &= check("die alte hartkodierte Tiefsee-Konstante '-rohHoehe / 400.0' ist weg",
                "-rohHoehe / 400.0" not in source)
    return ok


if __name__ == "__main__":
    results = {
        "max_height_is_fixed_not_map_dependent": run_max_height_is_fixed_not_map_dependent(),
        "shader_source_uses_configurable_min_height": run_shader_source_uses_configurable_min_height(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
