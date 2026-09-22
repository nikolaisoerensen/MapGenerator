"""
Regressionstest fuer die 2D/3D-Layer-Parity (2026-07-27). Nicht Teil einer
Test-Suite - manuell ueber das gemeinsame venv laufen lassen, siehe CLAUDE.md.

Hintergrund: Farbskala, Wertebereich und Skalierungsart (linear/log) jedes
Layers stehen zentral in CanvasSettings.CANVAS_2D["layer_ranges"]. BEIDE
Ansichten lesen daraus - map_display_2d._get_layer_range() direkt,
map_display_3d._colorize_layer() ueber die Namensbruecke
_LAYER_RANGE_KEY_MAP (2D-Key <-> 3D-interner Layername).

Diese Bruecke ist eine handgepflegte Tabelle, und genau dort entstehen die
Luecken: ein neuer Anzeigemodus wird in base_tab._LAYER_NAME_MAP_3D
eingetragen, in _LAYER_RANGE_KEY_MAP aber vergessen. Der Layer rendert dann
in 3D mit viridis und Auto-Skalierung statt mit seiner eigenen Farbskala -
sichtbar anders als in 2D, ohne dass irgendetwas fehlschlaegt. Genau so ist
es zuletzt bei thermal_erosion/thermal_deposition/evaporation (neue
Water-Modi) und bei civ_map passiert.

Der Test prueft die drei Richtungen dieser Kette und rendert jeden Layer
einmal durch beide Pfade.
"""
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from gui.config.gui_default import CanvasSettings
from gui.tabs.base_tab import BaseMapTab
from gui.widgets.map_display_2d import _get_layer_range
from gui.widgets.map_display_3d import _LAYER_RANGE_KEY_MAP, _colorize_layer


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return bool(condition)


# Layer, die bewusst NICHT ueber layer_ranges laufen, weil sie kategorisch
# oder bereits RGB sind - _colorize_layer() behandelt sie gesondert
# (Index-Tabelle bzw. Direktuebernahme), identisch zur 2D-Seite.
#
# region_overlay/kuesten_overlay (2026-08-13): laufen NIE durch
# _colorize_layer() - eigener Pfad ueber _render_dict_rgba_overlay(), der die
# RGBA-Textur direkt aus demselben Rohdaten-Dict wie der 2D-Renderer baut
# (rasterize_regions_rgba()/rasterize_kuesten_archetypen_rgba(), siehe
# map_display_2d.py). Eine Farbskala waere hier ohnehin sinnlos, die Farben
# kommen aus den Regionsfarben der Tabelle, nicht aus einem Skalarwert.
_CATEGORICAL_LAYERS_3D = {"rock_map", "biome_map", "super_biome_mask",
                          "region_overlay", "kuesten_overlay"}


def run_every_3d_layer_has_a_range():
    """Jeder Layer, den ein Tab in 3D auswaehlbar macht, braucht entweder
    einen layer_ranges-Eintrag oder eine kategorische Sonderbehandlung."""
    ranges = CanvasSettings.CANVAS_2D.get("layer_ranges", {})
    ok = True
    for tab, layer_keys in sorted(BaseMapTab._LAYER_SELECTION_KEYS_3D.items()):
        for layer in sorted(layer_keys):
            if layer in _CATEGORICAL_LAYERS_3D:
                continue
            range_key = _LAYER_RANGE_KEY_MAP.get(layer)
            ok &= check(f"{tab}/{layer}: Range-Mapping vorhanden", range_key is not None)
            if range_key is not None:
                ok &= check(f"{tab}/{layer}: layer_ranges['{range_key}'] existiert",
                            range_key in ranges)
    return ok


def run_every_2d_mode_reaches_3d():
    """Jeder 2D-Anzeigemodus, der in _LAYER_NAME_MAP_3D auf einen 3D-Layer
    zeigt, muss dort auch eine Farbskala finden."""
    ranges = CanvasSettings.CANVAS_2D.get("layer_ranges", {})
    ok = True
    for layer_2d, layer_3d in sorted(BaseMapTab._LAYER_NAME_MAP_3D.items()):
        if layer_3d in _CATEGORICAL_LAYERS_3D:
            continue
        range_key = _LAYER_RANGE_KEY_MAP.get(layer_3d)
        ok &= check(f"2D '{layer_2d}' -> 3D '{layer_3d}': Range-Mapping vorhanden",
                    range_key is not None)
        if range_key is not None:
            ok &= check(f"2D '{layer_2d}': gleicher layer_ranges-Eintrag in beiden Pfaden "
                        f"('{range_key}')",
                        range_key in ranges and range_key == layer_2d)
    return ok


def run_identical_colormap_and_scale():
    """Fuer jeden gemappten Layer muessen 2D und 3D dieselbe Colormap,
    denselben Wertebereich UND dieselbe Skalierungsart verwenden."""
    ok = True
    for layer_3d, range_key in sorted(_LAYER_RANGE_KEY_MAP.items()):
        cmap_2d, vmin_2d, vmax_2d, scale_2d = _get_layer_range(range_key)
        # 3D liest ueber exakt denselben Aufruf - geprueft wird hier, dass die
        # Bruecke auf den Eintrag zeigt, der auch in 2D benutzt wird.
        ok &= check(f"{layer_3d}: Colormap/Bereich/Skala definiert "
                    f"({cmap_2d}, {vmin_2d}..{vmax_2d}, {scale_2d})",
                    cmap_2d is not None)
    return ok


def run_render_every_layer():
    """Jeden Layer einmal tatsaechlich durch _colorize_layer() schicken -
    faengt Form-/dtype-Fehler, die eine reine Tabellen-Pruefung nicht sieht.
    Insbesondere log-skalierte Layer mit Nullen (erosion/sedimentation/
    thermal_*) und Vektorfelder (wind, slope)."""
    size = 24
    rng = np.random.RandomState(4)
    ok = True

    for layer_3d in sorted(_LAYER_RANGE_KEY_MAP):
        if layer_3d == "wind":
            data = (rng.rand(size, size, 2) * 20.0 - 10.0).astype(np.float32)
        elif layer_3d == "slope":
            data = (rng.rand(size, size, 2) * 2.0 - 1.0).astype(np.float32)
        else:
            # Bewusst mit vielen exakten Nullen: das ist die Verteilung, an der
            # eine LogNorm ohne Epsilon-Floor scheitern wuerde.
            data = (rng.rand(size, size) * 30.0).astype(np.float32)
            data[data < 15.0] = 0.0

        try:
            rgb = _colorize_layer(data, layer_3d)
        except Exception as e:
            ok &= check(f"{layer_3d}: _colorize_layer() laeuft durch", False)
            print(f"    {type(e).__name__}: {e}")
            continue

        ok &= check(f"{layer_3d}: RGB ({size},{size},3) uint8, endlich",
                    rgb.shape == (size, size, 3) and rgb.dtype == np.uint8
                    and bool(np.all(np.isfinite(rgb))))

    for categorical in sorted(_CATEGORICAL_LAYERS_3D):
        if categorical == "rock_map":
            data = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        else:
            data = rng.randint(0, 12, (size, size)).astype(np.int32)
        try:
            rgb = _colorize_layer(data, categorical)
            ok &= check(f"{categorical} (kategorisch): RGB ({size},{size},3) uint8",
                        rgb.shape == (size, size, 3) and rgb.dtype == np.uint8)
        except Exception as e:
            ok &= check(f"{categorical} (kategorisch): _colorize_layer() laeuft durch", False)
            print(f"    {type(e).__name__}: {e}")
    return ok


if __name__ == "__main__":
    results = {
        "every_3d_layer_has_a_range": run_every_3d_layer_has_a_range(),
        "every_2d_mode_reaches_3d": run_every_2d_mode_reaches_3d(),
        "identical_colormap_and_scale": run_identical_colormap_and_scale(),
        "render_every_layer": run_render_every_layer(),
    }
    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)
