"""
Path: tests/smoke_test_erosion_realismus.py

Realismus-Test fuer den Feld-Erosionssimulator (core/erosion_generator.py,
Klasse HydraulicFieldSimulator, Kernschritt _pass_erode_deposit()).

ANDERS als tests/smoke_test_erosion_field.py (prueft die Zusagen: Massenbilanz,
Buchhaltung, Determinismus) und tests/smoke_test_erosion_quality.py (prueft die
BILDEIGENSCHAFTEN gegen fest eingefrorene Schwellen): dieser Test prueft, ob das
Modell sich wie eine physikalische Landschaft VERHAELT, gegen unabhaengig
begruendete Erwartungen - nicht gegen Schwellen, die aus dem Modell selbst
zurueckgerechnet sind.

`EROSION_AKTIV = False` (siehe docs/OFFENE_PUNKTE.md) schaltet die Kette in der
laufenden App-Pipeline ab. Das betrifft diesen Test NICHT - er spricht
HydraulicFieldSimulator().simulate() direkt an, am oeffentlichen Einstieg des
Simulators, unabhaengig vom Pipeline-Schalter.

Drei Tests:

  A) Slope-Area-Gesetz   Hangneigung ~ Einzugsgebiet^beta. Zielkorridor
                         -0.4 .. -0.7 aus docs/archiv/2026-07-29_SPEZIFIKATION.md §3.2 (dort
                         gemessen: -0.638) - eine externe, nicht aus diesem
                         Test zurueckgerechnete Quelle.
  B) Monotonie           steilere Abschnitte tragen mehr ab als flachere -
                         die direkteste, robusteste der drei Pruefungen.
  C) Sediment-Konkavitaet  Senken (konkav) sammeln Ablagerung, Kuppen (konvex)
                         praktisch nicht.

Aufruf: .venv\\Scripts\\python.exe tests/smoke_test_erosion_realismus.py
"""

import sys

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.erosion_generator import HydraulicFieldSimulator
from tools.erosion_lab import build_terrain, default_parameters, flow_accumulation, slope_area_beta


def check(label, condition):
    print(("[OK] " if condition else "[FAIL] ") + label)
    return bool(condition)


# =============================================================================
# Test A - Slope-Area-Gesetz
# =============================================================================

def run_slope_area_law_matches_target_corridor():
    """
    Echtes Gelaende (tools.erosion_lab.build_terrain, die reale
    Terrain-Pipeline mit den ausgelieferten Defaults), echte Kartengroesse
    (256 px - die von MAX_CPU_RESOLUTION erlaubte Obergrenze ohne GPU),
    ausgelieferte Erosions-Defaults (tools.erosion_lab.default_parameters,
    identisch zu gui/config/value_default.py class EROSION), 4000 Schritte
    mit deaktiviertem Konvergenzabbruch (damit die volle Schrittzahl auch
    wirklich gerechnet wird statt vorzeitig als "fertig" gemeldet zu werden).

    Der Zielkorridor -0.4 .. -0.7 kommt NICHT aus einer Neuberechnung in
    diesem Test, sondern aus docs/archiv/2026-07-29_SPEZIFIKATION.md §3.2 - dort unabhaengig als
    die Standard-Signatur fluvial geformter Landschaften begruendet und mit
    -0.638 als erreichten Wert dokumentiert.
    """
    size = 256
    terrain, meters_per_pixel = build_terrain(size)
    hardness = np.full((size, size), 50.0, dtype=np.float32)

    params = default_parameters()
    params["max_steps"] = 4000
    params["convergence_threshold"] = 1e-8

    result = HydraulicFieldSimulator().simulate(terrain, hardness, params, meters_per_pixel)
    after = terrain.astype(np.float64) - result["erosion_map"] + result["sedimentation_map"]

    accumulation = flow_accumulation(after)
    beta = slope_area_beta(after, accumulation, meters_per_pixel)

    print("    {} Schritte, Relief {:.0f} m, beta = {:.3f}".format(
        result["steps_taken"], float(after.max() - after.min()), beta))

    return check(
        "beta liegt im fluvialen Zielkorridor -0.4 .. -0.7 "
        "(docs/archiv/2026-07-29_SPEZIFIKATION.md Paragraph 3.2, gemessen {:.3f})".format(beta),
        -0.7 <= beta <= -0.4)


# =============================================================================
# Test B - Monotonie: staerkerer Hang -> mehr Abtrag
# =============================================================================

def make_uniform_ramp(size, meters_per_pixel, slope, seed=7):
    """Gleichmaessige schiefe Ebene mit bekannter, konstanter Hangneigung
    (m Hoehenverlust je m Weg) in +y-Richtung, plus minimales Rauschen gegen
    Gitterartefakte durch perfekte Spaltensymmetrie."""
    rows = np.arange(size).reshape(-1, 1).astype(np.float64)
    terrain = 3000.0 - slope * meters_per_pixel * rows
    terrain = np.broadcast_to(terrain, (size, size)).copy()
    rng = np.random.RandomState(seed)
    terrain += rng.uniform(-0.01, 0.01, size=terrain.shape)
    return terrain.astype(np.float32)


def run_erosion_rate_increases_with_slope():
    """
    Sechs UNABHAENGIGE Laeufe, je einer auf einer eigenen, komplett
    gleichmaessigen Rampe mit fester, bekannter Hangneigung. Bewusst separate
    Laeufe statt eines einzigen Gelaendes mit mehreren Baendern: in einem
    gemeinsamen Gelaende waechst die Wassermenge mit der Ent-fernung vom
    oberen Kartenrand (Einzugsgebiet), und dieser Lageeffekt wuerde sich mit
    dem Hangneigungseffekt vermischen. Da hier fuer jede Neigung ein eigener
    Lauf auf gleich grossem Gelaende steht, ist die Lage-Komponente fuer alle
    sechs identisch - einzige Variable bleibt die Neigung.

    Gemessen (128 px, 150 Schritte, mittlere Kartenzeile ausgewertet):

        Neigung   0.05    0.08     0.11     0.14     0.18     0.24
        Abtrag  7.4e-6  9.6e-3  1.69e-2  2.71e-2  4.43e-2  7.76e-2

    Erwartung unabhaengig begruendet: das Stream-Power-Modell traegt umso
    mehr ab, je steiler das Gelaende (mehr Fliessgeschwindigkeit, mehr
    Transportkapazitaet) - eine Grundeigenschaft jedes hydraulischen
    Erosionsmodells, keine aus diesem Testlauf zurueckgerechnete Zahl.
    """
    size = 128
    meters_per_pixel = 40.0
    hardness = np.full((size, size), 50.0, dtype=np.float32)
    slopes = [0.05, 0.08, 0.11, 0.14, 0.18, 0.24]

    mean_erosion = []
    for slope in slopes:
        terrain = make_uniform_ramp(size, meters_per_pixel, slope)
        params = {"max_steps": 150, "rainfall": 2.0, "thermal_strength": 0.0,
                  "smoothing": 0.0, "convergence_threshold": 1e-8}
        result = HydraulicFieldSimulator().simulate(terrain, hardness, params, meters_per_pixel)
        # nur die mittlere Haelfte auswerten - Kartenrand oben (kein Einzugs-
        # gebiet) und unten (offener Rand) aussparen
        band = result["erosion_map"][size // 4: 3 * size // 4, :]
        mean_erosion.append(float(band.mean()))

    print("    Neigung -> mittlerer Abtrag: " + ", ".join(
        "{:.3f}->{:.2e}".format(s, e) for s, e in zip(slopes, mean_erosion)))

    ok = True
    for i in range(1, len(slopes)):
        ok &= check(
            "Abtrag bei Neigung {:.2f} ({:.3e}) > Abtrag bei Neigung {:.2f} ({:.3e})".format(
                slopes[i], mean_erosion[i], slopes[i - 1], mean_erosion[i - 1]),
            mean_erosion[i] > mean_erosion[i - 1])
    return ok


# =============================================================================
# Test C - Sediment-Konkavitaet
# =============================================================================

def make_bumpy_ramp(size, meters_per_pixel, base_slope=0.05, seed=11):
    """
    Schiefe Ebene (damit ueberhaupt Wasser und Fracht durchlaufen) mit sechs
    aufgesetzten Gauss-Kuppen abwechselnd nach oben (Huegel, konvex) und
    unten (Senken, konkav) verschoben. Die Zugehoerigkeit einer Zelle zu
    "Senke" oder "Kuppe" kommt aus der KONSTRUKTIONS-Geometrie (Abstand zum
    Zentrum einer bekannten Kuppe), nicht aus einer im Test berechneten
    Kruemmung - die Masken sind damit unabhaengig vom Simulationsergebnis.
    """
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float64)
    terrain = 3000.0 - base_slope * meters_per_pixel * rows

    rng = np.random.RandomState(seed)
    basin_mask = np.zeros((size, size), dtype=bool)
    hill_mask = np.zeros((size, size), dtype=bool)
    radius = 10.0
    amplitude = 60.0

    centers_y = np.linspace(size * 0.2, size * 0.8, 6)
    kinds = ["basin", "hill"] * 3
    rng.shuffle(kinds)
    for centre_y, kind in zip(centers_y, kinds):
        centre_x = size / 2.0 + rng.uniform(-size * 0.15, size * 0.15)
        distance_sq = (rows - centre_y) ** 2 + (cols - centre_x) ** 2
        bump = amplitude * np.exp(-distance_sq / (2 * radius ** 2))
        if kind == "basin":
            terrain -= bump
            basin_mask |= distance_sq < (radius * 0.6) ** 2
        else:
            terrain += bump
            hill_mask |= distance_sq < (radius * 0.6) ** 2

    terrain += rng.uniform(-0.01, 0.01, size=terrain.shape)
    return terrain.astype(np.float32), basin_mask, hill_mask


def run_deposition_concentrates_in_concave_terrain():
    """
    Erwartung unabhaengig begruendet: Sediment-Ablagerung entsteht, wo die
    Transportkapazitaet unter die mitgefuehrte Fracht faellt - das ist in
    jedem Stream-Power-Modell dort der Fall, wo Wasser sich sammelt und
    verlangsamt (Senken, Talboeden), nicht dort, wo es sich verteilt und
    beschleunigt (Kuppen, Grate). Das ist Modul-Allgemeinwissen ueber
    hydraulische Erosion, keine aus dem Testlauf zurueckgerechnete Zahl.

    Gemessen (128 px, 2000 Schritte): mittlere Ablagerung in den sechs
    Senkenzentren 2.78 m gegen 0.00 m in den sechs Kuppenzentren.
    """
    size = 128
    meters_per_pixel = 40.0
    hardness = np.full((size, size), 50.0, dtype=np.float32)
    terrain, basin_mask, hill_mask = make_bumpy_ramp(size, meters_per_pixel)

    params = {"max_steps": 2000, "rainfall": 3.0, "thermal_strength": 0.0,
              "smoothing": 0.0, "convergence_threshold": 1e-8, "deposition_rate": 0.3}
    result = HydraulicFieldSimulator().simulate(terrain, hardness, params, meters_per_pixel)
    deposition = result["sedimentation_map"]

    basin_mean = float(deposition[basin_mask].mean())
    hill_mean = float(deposition[hill_mask].mean())
    print("    Senken (konkav): {:.4f} m mittlere Ablagerung | "
          "Kuppen (konvex): {:.4f} m".format(basin_mean, hill_mean))

    return check(
        "Ablagerung konzentriert sich signifikant staerker in konkaven Senken "
        "als auf konvexen Kuppen ({:.4f} m gegen {:.4f} m)".format(basin_mean, hill_mean),
        basin_mean > 10.0 * max(hill_mean, 1e-6))


# =============================================================================

def main():
    tests = [
        ("slope_area_law_matches_target_corridor", run_slope_area_law_matches_target_corridor),
        ("erosion_rate_increases_with_slope", run_erosion_rate_increases_with_slope),
        ("deposition_concentrates_in_concave_terrain", run_deposition_concentrates_in_concave_terrain),
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
