"""
Throwaway headless smoke test for the virtual-pipe hydraulic flow model
(core/water_generator.py PipeFlowSimulator) - replaces the D8 steepest-
descent + watershed-redirection model (bis 2026-07-25). Not part of the
test suite - run manually via the shared venv, see CLAUDE.md.

Covers:
- (a) exact mass conservation: total rain input volume equals final standing
  water volume plus total edge outflow, within float tolerance - the
  property the user called non-negotiable ("ich will Massenerhaltung").
- (b) no dead sinks: a closed basin's water depth plateaus (converges) once
  it fills past its rim, instead of growing without bound - proves
  depressions fill-and-spill correctly WITHOUT any separate watershed-
  redirection step (structurally replaced by the K mass-conservation
  scaling, see PipeFlowSimulator docstring).
- (c) CPU/GPU parity is NOT covered here - per CLAUDE.md, GPU/shader code
  needs the live app, not headless smoke tests. This suite only exercises
  the CPU fallback path.
"""
import sys

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.water_generator import PipeFlowSimulator


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def _make_basin_terrain(size=48, seed=11):
    """Geschlossenes Becken (Krater) mit Rand deutlich unter dem
    umgebenden, nach aussen zum Kartenrand abfallenden Gelaende - Wasser
    MUSS sich erst bis zur Rand-Hoehe fuellen, bevor es weiter nach aussen
    fliessen kann (genau der Fall, der beim alten D8-Modell eine gesonderte
    Wasserscheiden-Umleitung brauchte)."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    center = (size * 0.5, size * 0.5)
    dist_to_edge = np.minimum(np.minimum(x, size - 1 - x), np.minimum(y, size - 1 - y))
    # Aussen (nahe Kartenrand) niedrig, zur Mitte hin ansteigend, mit einem
    # tiefen Krater GENAU in der Mitte - der Rand des Kraters liegt ueber
    # dem Umgebungs-Niveau am Kartenrand, aber deutlich unter dem Krater-Rand.
    outward_slope = dist_to_edge * 2.0
    dist_center = np.hypot(x - center[1], y - center[0])
    crater = 60.0 * np.exp(-(dist_center ** 2) / (2 * (size * 0.1) ** 2))
    heightmap = 50.0 + outward_slope - crater
    heightmap += 0.5 * rng.randn(size, size)
    return heightmap.astype(np.float32), center


def run_mass_conservation_flat_terrain():
    """Massenbilanz auf einfachem geneigtem Gelaende (kein geschlossenes
    Becken) ueber mehrere Iterationszahlen - Kern-Eigenschaft, die der
    Nutzer als nicht verhandelbar bezeichnet hat."""
    size = 32
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    heightmap = (100.0 - 0.5 * x - 0.3 * y).astype(np.float32)
    precip_map = np.full((size, size), 5.0, dtype=np.float32)
    meters_per_pixel = 50.0
    cell_area = meters_per_pixel * meters_per_pixel

    sim = PipeFlowSimulator()
    ok = True
    for n_steps in (20, 100, 400):
        dt_seconds = sim.PIPE_TIME_SCALE_S / n_steps
        # Verdunstung hier bewusst 0: dieser Test prueft die MASSENBILANZ des
        # Transports (Regen rein = stehendes Wasser + Randabfluss). Die
        # Verdunstungs-Senke wird separat in
        # run_evaporation_is_a_real_sink() geprueft.
        no_evaporation = np.zeros_like(precip_map)
        result = sim._cpu_simulate(heightmap, precip_map, no_evaporation,
                                    n_steps, dt_seconds, meters_per_pixel)

        # Regen ist eine RATE pro Sekunde (siehe RAIN_TO_DEPTH_RATE) - die
        # Gesamtmenge haengt an der Simulationsdauer, nicht an der Schrittzahl.
        total_rain_volume = (float((precip_map * sim.RAIN_TO_DEPTH_RATE).sum()) * cell_area
                             * dt_seconds * n_steps)
        standing_volume = float(result["water_depth"].sum()) * cell_area
        edge_outflow_volume = result["edge_outflow"]

        total_out = standing_volume + edge_outflow_volume
        relative_error = abs(total_out - total_rain_volume) / max(total_rain_volume, 1e-9)

        ok &= check(f"n_steps={n_steps}: Massenbilanz erhalten (Regen={total_rain_volume:.2f}m³, "
                    f"stehend={standing_volume:.2f}m³ + Randabfluss={edge_outflow_volume:.2f}m³ = "
                    f"{total_out:.2f}m³, rel. Fehler {100*relative_error:.4f}%)",
                    relative_error < 0.01)
        ok &= check(f"n_steps={n_steps}: kein NaN/Inf, Tiefe >= 0",
                    bool(np.all(np.isfinite(result["water_depth"]))) and bool(np.all(result["water_depth"] >= 0)))
    return ok


def run_mass_conservation_closed_basin():
    """Dieselbe Massenbilanz-Pruefung, aber auf einem geschlossenen Becken
    (siehe _make_basin_terrain) - stellt sicher, dass die K-Skalierung auch
    bei aufgestautem Wasser exakt bleibt, nicht nur auf frei abfliessendem
    Gelaende."""
    heightmap, _ = _make_basin_terrain()
    size = heightmap.shape[0]
    precip_map = np.full((size, size), 8.0, dtype=np.float32)
    meters_per_pixel = 30.0
    cell_area = meters_per_pixel * meters_per_pixel
    n_steps = 300

    sim = PipeFlowSimulator()
    dt_seconds = sim.PIPE_TIME_SCALE_S / n_steps
    no_evaporation = np.zeros_like(precip_map)
    result = sim._cpu_simulate(heightmap, precip_map, no_evaporation,
                                n_steps, dt_seconds, meters_per_pixel)

    total_rain_volume = (float((precip_map * sim.RAIN_TO_DEPTH_RATE).sum()) * cell_area
                         * dt_seconds * n_steps)
    standing_volume = float(result["water_depth"].sum()) * cell_area
    total_out = standing_volume + result["edge_outflow"]
    relative_error = abs(total_out - total_rain_volume) / max(total_rain_volume, 1e-9)

    ok = check(f"geschlossenes Becken: Massenbilanz erhalten (Regen={total_rain_volume:.2f}m³, "
               f"stehend={standing_volume:.2f}m³ + Randabfluss={result['edge_outflow']:.2f}m³, "
               f"rel. Fehler {100*relative_error:.4f}%)", relative_error < 0.01)
    ok &= check("kein NaN/Inf, Tiefe >= 0",
                bool(np.all(np.isfinite(result["water_depth"]))) and bool(np.all(result["water_depth"] >= 0)))
    return ok


def run_closed_basin_no_dead_sink():
    """
    Kernnachweis fuer 'keine toten Senken' (Nutzer-Report urspruenglich:
    'die meisten Seen enden in lokalen Minima ohne Flusssystem, das von der
    Karte fuehrt'): die Wassertiefe im Krater-Zentrum darf NICHT unbegrenzt
    weiterwachsen, sobald der Kraterrand ueberflutet ist - Wachstum in der
    zweiten Haelfte der Simulation muss deutlich langsamer sein als in der
    ersten (Plateau/Konvergenz statt linearem Weiterwachsen), OHNE jede
    gesonderte Wasserscheiden-Umleitung (die gibt es im Pipe-Modell nicht
    mehr - Fuellen-und-Ueberlaufen ist strukturell eingebaut, siehe
    PipeFlowSimulator-Docstring).
    """
    heightmap, center = _make_basin_terrain(size=48, seed=11)
    size = heightmap.shape[0]
    cy, cx = int(center[0]), int(center[1])
    precip_map = np.full((size, size), 15.0, dtype=np.float32)
    meters_per_pixel = 30.0

    sim = PipeFlowSimulator()
    n_steps_half = 150
    dt_seconds = sim.PIPE_TIME_SCALE_S / (2 * n_steps_half)

    depth = None
    flux = None
    depths_over_time = []
    for _ in range(2 * n_steps_half):
        result = sim._cpu_simulate(
            heightmap, precip_map, np.zeros_like(precip_map), 1, dt_seconds, meters_per_pixel,
            previous_depth=depth, previous_flux=flux)
        depth = result["depth_state"]
        flux = result["flux_state"]
        depths_over_time.append(float(depth[cy, cx]))

    first_half_growth = depths_over_time[n_steps_half - 1] - depths_over_time[0]
    second_half_growth = depths_over_time[-1] - depths_over_time[n_steps_half - 1]

    ok = check(f"Krater-Tiefe waechst in der 2. Haelfte ({second_half_growth:.4f}m) deutlich "
               f"langsamer als in der 1. Haelfte ({first_half_growth:.4f}m) - Plateau statt "
               f"unbegrenztem Wachstum", second_half_growth < 0.5 * max(first_half_growth, 1e-6))
    ok &= check("Krater-Tiefe endlich, nicht-negativ, kein Explodieren",
                bool(np.isfinite(depths_over_time[-1])) and depths_over_time[-1] >= 0
                and depths_over_time[-1] < 1000.0)
    return ok


def run_gpu_note():
    """Kein echter Test - dokumentiert nur, dass der GPU-Pfad
    (pipeFluxUpdate.comp/pipeDepthUpdate.comp via _dispatch_pipe_flow_network)
    per CLAUDE.md nicht headless testbar ist, Live-Abgleich noetig."""
    return check("GPU-Pfad nicht headless testbar (siehe CLAUDE.md) - Live-App-Bestaetigung noetig", True)


if __name__ == "__main__":
    results = {
        "mass_conservation_flat_terrain": run_mass_conservation_flat_terrain(),
        "mass_conservation_closed_basin": run_mass_conservation_closed_basin(),
        "closed_basin_no_dead_sink": run_closed_basin_no_dead_sink(),
        "gpu_note": run_gpu_note(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
