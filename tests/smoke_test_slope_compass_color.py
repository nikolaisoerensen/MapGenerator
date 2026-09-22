"""
Throwaway headless smoke test for the compass-direction slope coloring
(gui/widgets/map_display_2d.py compute_slope_compass_rgb(), reused by
map_display_3d.py's _colorize_layer() for the "slope" overlay). Not part of
the test suite - run manually via the shared venv, see CLAUDE.md.
"""
import sys

import numpy as np
from matplotlib.colors import rgb_to_hsv

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from gui.widgets.map_display_2d import compute_slope_compass_rgb


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def run_aspect_direction_sanity():
    """Verifiziert die Kompass-Konvention dieses Projekts (Zeile H-1=Norden,
    Zeile 0=Sueden, Spalte 0=Westen, Spalte W-1=Osten, mehrfach in dieser
    Session verifiziert) gegen compute_slope_compass_rgb(): ein Hang, der
    von Norden nach Sueden abfaellt, muss Hue=180 (Sueden) zeigen; ein Hang,
    der von Osten nach Westen abfaellt, muss Hue=270 (Westen) zeigen."""
    size = 20
    ok = True

    h_north_high = np.zeros((size, size), dtype=np.float64)
    for r in range(size):
        h_north_high[r, :] = r * 10.0
    grad_y, grad_x = np.gradient(h_north_high)
    hue = rgb_to_hsv(compute_slope_compass_rgb(grad_x, grad_y))[size // 2, size // 2, 0] * 360
    ok &= check(f"Norden hoch/Sueden niedrig -> Aspect Sueden (180 Grad), gemessen {hue:.1f}",
                abs(hue - 180.0) < 1.0)

    h_east_high = np.zeros((size, size), dtype=np.float64)
    for c in range(size):
        h_east_high[:, c] = c * 10.0
    grad_y2, grad_x2 = np.gradient(h_east_high)
    hue2 = rgb_to_hsv(compute_slope_compass_rgb(grad_x2, grad_y2))[size // 2, size // 2, 0] * 360
    ok &= check(f"Osten hoch/Westen niedrig -> Aspect Westen (270 Grad), gemessen {hue2:.1f}",
                abs(hue2 - 270.0) < 1.0)

    h_south_high = np.zeros((size, size), dtype=np.float64)
    for r in range(size):
        h_south_high[r, :] = (size - 1 - r) * 10.0
    grad_y3, grad_x3 = np.gradient(h_south_high)
    hue3 = rgb_to_hsv(compute_slope_compass_rgb(grad_x3, grad_y3))[size // 2, size // 2, 0] * 360
    ok &= check(f"Sueden hoch/Norden niedrig -> Aspect Norden (0/360 Grad), gemessen {hue3:.1f}",
                abs(hue3 - 0.0) < 1.0 or abs(hue3 - 360.0) < 1.0)

    return ok


def run_flat_is_white_steep_is_saturated():
    """Flach = weiss (Saettigung 0), steil = voll gesaettigt (Nutzer-Vorgabe:
    'komplett flach ist weiß... komplett knallig... bei einem 90° Hang')."""
    size = 10
    flat = np.zeros((size, size), dtype=np.float64)
    gy_flat, gx_flat = np.gradient(flat)
    rgb_flat = compute_slope_compass_rgb(gx_flat, gy_flat)
    ok = check("Flach ist weiss (RGB nahe [1,1,1])",
               bool(np.allclose(rgb_flat, 1.0, atol=1e-6)))

    # Extrem steiler Gradient (>> tan(89 Grad)) - Saettigung muss nahe 1 sein.
    steep_dz_dx = np.full((size, size), 1000.0)
    steep_dz_dy = np.zeros((size, size))
    hsv_steep = rgb_to_hsv(compute_slope_compass_rgb(steep_dz_dx, steep_dz_dy))
    ok &= check(f"Sehr steiler Hang hat Saettigung nahe 1 (gemessen {hsv_steep[0, 0, 1]:.3f})",
                hsv_steep[0, 0, 1] > 0.95)
    return ok


if __name__ == "__main__":
    results = {
        "aspect_direction_sanity": run_aspect_direction_sanity(),
        "flat_is_white_steep_is_saturated": run_flat_is_white_steep_is_saturated(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
