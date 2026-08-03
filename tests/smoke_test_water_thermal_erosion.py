"""
Throwaway headless smoke test for the thermal erosion / angle-of-repose
mechanic (core/water_generator.py ThermalErosionSystem, "Phase 6"). Not
part of the test suite - run manually via the shared venv, see CLAUDE.md.

Covers:
- (a) V-Kerbtal unter hartem vs. weichem Gestein: weiches Gestein muss nach
  gleich vielen Iterationen messbar breiter/flacher werden als hartes
  Gestein - der konkrete, pruefbare Nachweis fuer "haertere Materialien
  bleiben besser zurueck" UND "V- und U-Taeler je nach Bedingung" in einem
  Test.
- (b) exakte Massenerhaltung (Sigma abgetragen == Sigma abgelagert - hier
  strukturell exakt, da jede transportierte Einheit Gather-only genau einer
  Quelle/einem Ziel zugeordnet ist, siehe ThermalErosionSystem-Docstring -
  keine Renormierung wie beim Semi-Lagrange-Sedimenttransport noetig).
- (c) Deckel-Test: keine Zelle gibt pro Iteration mehr ab, als der
  relief-relative Deckel erlaubt.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.water_generator import ThermalErosionSystem


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def _make_v_notch(size=40, depth=30.0, half_width_px=2):
    """Schmales V-Kerbtal (wie es Fluss-Erosion hinterlaesst) quer ueber
    eine ansonsten flache Hochebene, in x-Richtung verlaufend (Tal-Achse =
    y), damit die Talbreite entlang x gemessen werden kann."""
    heightmap = np.full((size, size), 100.0, dtype=np.float32)
    center_x = size // 2
    for x in range(size):
        dist = abs(x - center_x)
        if dist <= half_width_px * 4:
            heightmap[:, x] -= max(0.0, depth * (1.0 - dist / (half_width_px * 4)))
    return heightmap


def _mean_wall_slope(heightmap, wall_columns):
    """Mittlere |dh/dx| (Pixel-Einheiten) ueber die Tal-Wandspalten, in der
    mittleren Zeile - direktes Signal fuer 'wie steil steht die Talwand
    noch' (robuster als eine Breiten-Messung bei fester Hoehe, die durch
    gleichzeitige Ablagerung im Talboden verzerrt werden kann - der
    Talboden hebt sich, wodurch eine Breite-bei-fester-Hoehe-Messung
    faelschlich SCHMALER erscheinen kann, obwohl die Waende tatsaechlich
    flacher/kollabierter sind)."""
    row = heightmap[heightmap.shape[0] // 2, :].astype(np.float64)
    grad = np.abs(np.diff(row))
    return float(np.mean(grad[wall_columns]))


def run_soft_rock_widens_more_than_hard_rock():
    """Kernnachweis: gleiche Ausgangsform, gleiche Iterationszahl, nur die
    Haerte unterscheidet sich - weiches Gestein muss messbar breiter/flacher
    werden."""
    size = 40
    meters_per_pixel = 10.0
    iterations = 250

    heightmap_hard = _make_v_notch(size=size)
    heightmap_soft = heightmap_hard.copy()
    hardness_hard = np.full((size, size), 95.0, dtype=np.float32)
    hardness_soft = np.full((size, size), 2.0, dtype=np.float32)

    system = ThermalErosionSystem(thermal_strength=2.0)

    erosion_hard, deposition_hard = system._cpu_simulate(
        heightmap_hard, hardness_hard, iterations, meters_per_pixel)
    erosion_soft, deposition_soft = system._cpu_simulate(
        heightmap_soft, hardness_soft, iterations, meters_per_pixel)

    final_hard = heightmap_hard + deposition_hard - erosion_hard
    final_soft = heightmap_soft + deposition_soft - erosion_soft

    # Wand-Steigung statt Breite-bei-fester-Hoehe: robuster, da gleichzeitige
    # Ablagerung im Talboden eine reine Breiten-Messung verzerren kann (der
    # Boden hebt sich, wodurch eine feste Hoehen-Schwelle faelschlich
    # schmaler erscheinen kann, obwohl die Waende tatsaechlich flacher
    # geworden sind - siehe _mean_wall_slope-Docstring).
    center_x = size // 2
    wall_columns = np.arange(center_x - size // 4, center_x + size // 4)
    wall_columns = wall_columns[(wall_columns >= 0) & (wall_columns < size - 1)]

    slope_before = _mean_wall_slope(heightmap_hard, wall_columns)
    slope_hard = _mean_wall_slope(final_hard, wall_columns)
    slope_soft = _mean_wall_slope(final_soft, wall_columns)

    # Erwartungswert statt fester Prozent-Schwelle: weiches Gestein muss bis
    # auf seinen eigenen Böschungswinkel abflachen (und nicht weiter), hartes
    # Gestein darf gar nicht kollabieren, weil seine Schwelle über der
    # Wandsteigung liegt. Bei meters_per_pixel=10 sind das
    # 10*tan(15.45°) = 2.76 m/px für weiches und 10*tan(57.7°) = 15.8 m/px
    # für hartes Gestein - letzteres liegt weit über der Ausgangssteigung von
    # 3.0 m/px, hartes Gestein bleibt also unangetastet.
    #
    # Die frühere feste Schwelle (soft < 0.85 * hard) stammte aus der Zeit des
    # zehnfach höheren Iterations-Deckels (CAP_RELIEF_FRACTION 1% statt 0.1%,
    # gesenkt 2026-07-27 wegen "thermal erosion zerfrisst die hügel sehr
    # stark"). Mit dem gezähmten Deckel bleibt weiches Gestein korrekt bei
    # seinem Böschungswinkel stehen, statt darüber hinaus abzurutschen - der
    # Test prüft jetzt genau das, statt eine an den alten Deckel gebundene
    # Verhältniszahl.
    repose_soft_slope = meters_per_pixel * np.tan(np.radians(
        ThermalErosionSystem.REPOSE_ANGLE_MIN_DEG))

    ok = check(f"weiches Gestein flacht Richtung seines Böschungswinkels ab "
               f"(vorher={slope_before:.2f}, weich={slope_soft:.2f}, "
               f"Zielwinkel entspricht {repose_soft_slope:.2f} m/px)",
               slope_soft < slope_before and slope_soft <= repose_soft_slope * 1.15)
    ok &= check(f"hartes Gestein kollabiert nicht (Schwelle liegt über der Wandsteigung): "
                f"hart={slope_hard:.2f} m/px", slope_hard >= 0.95 * slope_before)
    ok &= check(f"weiches Gestein ist messbar flacher als hartes "
                f"(weich={slope_soft:.2f}, hart={slope_hard:.2f} m/px)",
                slope_soft < slope_hard)
    ok &= check("beide Ergebnisse endlich", bool(np.all(np.isfinite(final_hard)))
                and bool(np.all(np.isfinite(final_soft))))
    return ok


def run_mass_conservation_exact():
    """Massenbilanz muss strukturell exakt sein (Gather-only, siehe Docstring) -
    keine Toleranz-Aufweichung wie beim Semi-Lagrange-Sedimenttransport noetig."""
    size = 32
    rng = np.random.RandomState(31)
    heightmap = (200.0 + 50.0 * rng.randn(size, size)).astype(np.float32)
    hardness_map = (1.0 + 99.0 * rng.rand(size, size)).astype(np.float32)
    meters_per_pixel = 15.0

    system = ThermalErosionSystem(thermal_strength=1.0)

    ok = True
    for iterations in (5, 30, 100):
        erosion_map, deposition_map = system._cpu_simulate(
            heightmap, hardness_map, iterations, meters_per_pixel)
        total_erosion = float(erosion_map.sum())
        total_deposition = float(deposition_map.sum())
        relative_error = abs(total_erosion - total_deposition) / max(total_erosion, 1e-9)

        ok &= check(f"iterations={iterations}: Massenbilanz exakt "
                    f"(abgetragen={total_erosion:.4f}, abgelagert={total_deposition:.4f}, "
                    f"rel. Fehler {100*relative_error:.6f}%)", relative_error < 1e-6)
        ok &= check(f"iterations={iterations}: kein NaN/Inf",
                    bool(np.all(np.isfinite(erosion_map))) and bool(np.all(np.isfinite(deposition_map))))
    return ok


def run_per_cell_cap_respected():
    """Keine Zelle darf pro Iteration mehr abgeben, als der relief-relative
    Deckel erlaubt (explizite Anforderung 'nie mehr abgeben als sie hat')."""
    size = 24
    heightmap = _make_v_notch(size=size, depth=80.0, half_width_px=1)
    hardness_map = np.full((size, size), 1.0, dtype=np.float32)  # weichstes Material, staerkster Antrieb
    meters_per_pixel = 5.0
    iterations = 10

    system = ThermalErosionSystem(thermal_strength=3.0)  # hoher Multiplikator, Deckel muss trotzdem greifen
    relief = float(heightmap.max() - heightmap.min())
    cap_per_step = max(system.CAP_MIN_M, system.CAP_RELIEF_FRACTION * relief)

    erosion_map, _ = system._cpu_simulate(heightmap, hardness_map, iterations, meters_per_pixel)
    max_possible_total = cap_per_step * iterations * 1.0001  # kleine Fliesskomma-Toleranz

    return check(f"kumulierter Abtrag pro Zelle <= Deckel*Iterationen "
                 f"(max gefunden={float(erosion_map.max()):.5f}m, Grenze={max_possible_total:.5f}m)",
                 bool(erosion_map.max() <= max_possible_total))


if __name__ == "__main__":
    results = {
        "soft_rock_widens_more_than_hard_rock": run_soft_rock_widens_more_than_hard_rock(),
        "mass_conservation_exact": run_mass_conservation_exact(),
        "per_cell_cap_respected": run_per_cell_cap_respected(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
