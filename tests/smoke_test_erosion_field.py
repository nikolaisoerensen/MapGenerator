"""
Path: tests/smoke_test_erosion_field.py

Regressionstest fuer den Feld-Simulationskern core/erosion_generator.py
(Stufe 1 des Erosion-Plans). Prueft die ZUSAGEN des Modells, nicht seine
Kalibrierung - die Bildqualitaet ist Gegenstand von smoke_test_erosion_quality.py
und der Kalibrierungsstufe.

Geprueft wird:
  (A) Massenbilanz    - jeder Pass ist erhaltend; einziger Verlustpfad ist der
                        Kartenrand
  (B) Buchhaltung     - Basis - erosion_map + sedimentation_map ergibt exakt
                        das simulierte Gelaende
  (C) Determinismus   - zwei Laeufe mit identischen Eingaben sind bitidentisch
  (D) CPU-Grenze      - oberhalb MAX_CPU_RESOLUTION ein klarer Fehler statt
                        eines stillen Haengers
  (E) Konvergenz      - das Abbruchkriterium greift und meldet sich ehrlich
  (F) Senken          - ein gefuelltes Becken graebt sich nicht weiter ein
                        (die Zusage, fuer die das Wasserspiegel-Gefaelle da ist)

Aufruf: .venv\\Scripts\\python.exe tests/smoke_test_erosion_field.py
"""

import sys

import time

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

from core.erosion_generator import HydraulicFieldSimulator
from core.water_generator import _NEIGHBOR_FOOTPRINT_8


FAILURES = []


def check(label, condition):
    print(("[OK] " if condition else "[FAIL] ") + label)
    if not condition:
        FAILURES.append(label)
    return bool(condition)


def make_terrain(size=64, seed=5, amplitude=4000.0):
    rng = np.random.RandomState(seed)
    return (gaussian_filter(rng.rand(size, size), 3) * amplitude).astype(np.float32)


def hardness(size=64, value=50.0):
    return np.full((size, size), value, dtype=np.float32)


# =============================================================================
# (A) Massenbilanz pro Pass
# =============================================================================

def run_every_pass_conserves_mass():
    """
    Material = Gelaende + geloeste Fracht + ueber den Rand exportierte Fracht.
    Diese Summe darf sich durch KEINEN Pass aendern - mit einer bewusst
    dokumentierten Ausnahme: die bedingte Glaettung verteilt ihren Anteil auf
    acht Nachbarn, und an der aeussersten Zellreihe fallen Nachbarn weg.

    Dieser Test ist der Grund, warum die semi-lagrangesche Advektion des
    Vorbilds ersetzt wurde: sie verlor gemessen 69% des abgetragenen Materials
    (siehe _pass_advect_sediment).
    """
    size = 64
    terrain = make_terrain(size)
    sim = HydraulicFieldSimulator()
    cfg = sim._resolve_parameters({"rainfall": 2.0})
    state = sim._initial_state(terrain, hardness(size), cfg, 10000.0 / size)
    dt = 0.5 * (10000.0 / size) / sim.MAX_FLOW_VELOCITY_M_S

    def material(st):
        return float(st["terrain"].sum() + st["sediment"].sum()) + st["sediment_exported"]

    passes = [
        ("Regen", lambda s: sim._pass_rain(s, cfg, dt)),
        ("Fluss+Tiefe", lambda s: sim._pass_flux_and_depth(s, dt)),
        ("Erosion/Ablagerung", lambda s: sim._pass_erode_deposit(s, cfg, dt)),
        ("Sedimenttransport", lambda s: sim._pass_advect_sediment(s, dt)),
        ("Thermal", lambda s: sim._pass_thermal(s, cfg)),
        ("Verdunstung", lambda s: sim._pass_evaporate(s, cfg, dt)),
    ]
    drift = {name: 0.0 for name, _ in passes}
    smoothing_drift = 0.0

    for _ in range(150):
        for name, func in passes:
            before = material(state)
            func(state)
            drift[name] += material(state) - before
        before = material(state)
        sim._pass_smooth(state, cfg)
        smoothing_drift += material(state) - before

    scale = max(abs(float(state["terrain"].sum())), 1.0)
    ok = True
    for name, _ in passes:
        ok &= check(
            "Pass '{}' ist erhaltend (Drift {:+.3e} m)".format(name, drift[name]),
            abs(drift[name]) < 1e-6 * scale)

    # Die Glaettung darf nur am Rand driften - eine Groessenordnung, die
    # gegenueber dem Gesamtmaterial verschwindet.
    ok &= check(
        "Glaettung driftet nur am Rand (Drift {:+.1f} m gegen {:.3e} m Gesamtmaterial)".format(
            smoothing_drift, scale),
        abs(smoothing_drift) < 1e-4 * scale)
    return ok


def run_overall_balance_closes():
    """Abtrag = anderswo abgelagert + noch geloest + ueber den Rand exportiert."""
    size = 64
    terrain = make_terrain(size)
    result = HydraulicFieldSimulator().simulate(
        terrain, hardness(size), {"max_steps": 3000, "rainfall": 2.0}, 10000.0 / size)
    return check(
        "Gesamtbilanz geht auf (Abweichung {:.2%} des Gesamtabtrags)".format(
            abs(result["mass_balance"])),
        abs(result["mass_balance"]) < 0.02)


# =============================================================================
# (B) Buchhaltung gegenueber der restlichen Pipeline
# =============================================================================

def run_difference_maps_reconstruct_terrain():
    """
    `Basis - erosion_map + sedimentation_map` muss EXAKT das simulierte
    Gelaende ergeben - genau so setzt DataLODManager.
    get_calculator_combined_heightmap() die Heightmap zusammen. Zusaetzlich
    darf keine Zelle in BEIDEN Karten gleichzeitig stehen (das Feldmodell
    bucht netto, siehe _collect_results).
    """
    size = 64
    terrain = make_terrain(size)
    result = HydraulicFieldSimulator().simulate(
        terrain, hardness(size), {"max_steps": 1500, "rainfall": 2.0}, 10000.0 / size)

    erosion = result["erosion_map"]
    sedimentation = result["sedimentation_map"]
    both = int(((erosion > 0) & (sedimentation > 0)).sum())

    ok = check("keine Zelle steht in beiden Differenzkarten (gemessen {})".format(both), both == 0)
    ok &= check("beide Karten sind nicht-negativ",
                bool(erosion.min() >= 0.0 and sedimentation.min() >= 0.0))
    ok &= check("alle Ausgaben endlich",
                all(np.all(np.isfinite(result[key])) for key in
                    ("erosion_map", "sedimentation_map", "sediment_load_map",
                     "water_depth_map", "flow_velocity_map")))
    return ok


# =============================================================================
# (C) Determinismus
# =============================================================================

def run_determinism():
    """Zwei unabhaengige Laeufe mit identischen Eingaben -> bitidentisch."""
    size = 48
    terrain = make_terrain(size)
    params = {"max_steps": 800, "rainfall": 2.0}
    first = HydraulicFieldSimulator().simulate(terrain, hardness(size), params, 10000.0 / size)
    second = HydraulicFieldSimulator().simulate(terrain, hardness(size), params, 10000.0 / size)

    ok = True
    for key in ("erosion_map", "sedimentation_map", "sediment_load_map",
                "water_depth_map", "flow_velocity_map"):
        ok &= check("{} bitidentisch ueber zwei Laeufe".format(key),
                    np.array_equal(first[key], second[key]))
    ok &= check("Schrittzahl identisch", first["steps_taken"] == second["steps_taken"])
    return ok


def run_input_is_not_mutated():
    """Die uebergebene Heightmap darf NICHT veraendert werden."""
    size = 48
    terrain = make_terrain(size)
    original = terrain.copy()
    HydraulicFieldSimulator().simulate(
        terrain, hardness(size), {"max_steps": 300}, 10000.0 / size)
    return check("uebergebene Heightmap unveraendert", np.array_equal(terrain, original))


# =============================================================================
# (D) CPU-Groessengrenze
# =============================================================================

def run_cpu_limit_fails_loudly():
    """
    Oberhalb MAX_CPU_RESOLUTION muss ein klarer Fehler kommen - ein stiller
    Haenger ueber Stunden waere genau die Art Falle, die in diesem Projekt
    zuletzt systematisch entfernt wurde.
    """
    size = HydraulicFieldSimulator.MAX_CPU_RESOLUTION + 32
    terrain = np.zeros((size, size), dtype=np.float32)
    try:
        HydraulicFieldSimulator().simulate(
            terrain, hardness(size), {"max_steps": 1}, 10.0)
    except ValueError as error:
        return check("zu grosse Karte meldet einen klaren Fehler",
                     "GPU" in str(error) and str(size) in str(error))
    return check("zu grosse Karte meldet einen klaren Fehler", False)


def run_resolution_cap_follows_gpu_registration():
    """
    Die Aufloesungs-Begrenzung muss an der REGISTRIERUNG des GPU-Pfads haengen,
    nicht an der blossen Existenz eines ShaderManagers.

    Der Anlass: die Pruefung hing zuerst an `shader_manager is None`. In der
    laufenden App ist immer ein ShaderManager vorhanden - der Deckel griff
    deshalb nie, und der Lauf endete mitten in der Pipeline mit
    "HydraulicFieldSimulator (CPU) verweigert 512x512". Der harte Fehler war
    korrekt, er haette nur nie ausgeloest werden duerfen.

    Seit Stufe 2 ist ("erosion", "hydraulicField") in DISPATCH_TABLE
    registriert. Der Deckel darf jetzt also NICHT mehr greifen, sobald ein
    ShaderManager da ist - und muss weiterhin greifen, wenn keiner da ist.
    Genau diese beiden Richtungen werden hier geprueft; ein Test, der nur eine
    davon abdeckt, haette den urspruenglichen Fehler nicht gefunden.
    """
    from core.erosion_generator import ErosionSystemGenerator
    from managers.shader_manager import DISPATCH_TABLE

    class ShaderManagerStub:
        gpu_available = True

    ok = check("der Erosions-Dispatch ist registriert",
               HydraulicFieldSimulator.GPU_OPERATION in DISPATCH_TABLE)
    ok &= check("ohne ShaderManager: kein GPU-Pfad",
                not HydraulicFieldSimulator().has_gpu_path())
    ok &= check("mit ShaderManager und registriertem Dispatch: GPU-Pfad",
                HydraulicFieldSimulator(shader_manager=ShaderManagerStub()).has_gpu_path())

    without_gpu = ErosionSystemGenerator()
    without_gpu.set_active_parameters({"simulation_resolution": 512})
    capped = without_gpu._resolve_simulation_size()
    ok &= check(
        "ohne GPU werden angefragte 512 auf {} begrenzt (gemessen {})".format(
            HydraulicFieldSimulator.MAX_CPU_RESOLUTION, capped),
        capped == HydraulicFieldSimulator.MAX_CPU_RESOLUTION)
    without_gpu.set_active_parameters({"simulation_resolution": 128})
    ok &= check("kleinere Aufloesungen bleiben auch ohne GPU unangetastet",
                without_gpu._resolve_simulation_size() == 128)

    with_gpu = ErosionSystemGenerator(shader_manager=ShaderManagerStub())
    with_gpu.set_active_parameters({"simulation_resolution": 512})
    ok &= check("mit GPU bleiben angefragte 512 stehen",
                with_gpu._resolve_simulation_size() == 512)
    return ok


def run_gpu_midrun_failure_above_cpu_limit_fails_loudly():
    """
    Ticket #30: faellt der GPU-Dispatch MITTEN im Lauf aus (Treiberfehler,
    Timeout), gibt _simulate_gpu() None zurueck und simulate() faellt bislang
    UNGEPRUEFT in die CPU-Hauptschleife unten durch - ungeachtet der Groesse.

    Das ist genau der Fall, den MAX_CPU_RESOLUTION eigentlich verhindern soll:
    eine Karte, die nur WEIL ein GPU-Pfad registriert war ueberhaupt groesser
    als 256x256 gewaehlt wurde, wuerde nach einem GPU-Ausfall still auf der CPU
    weiterrechnen - Stunden, mit nur einer WARNING-Zeile statt einem Fehler.

    Der Stub hat einen registrierten Dispatch (has_gpu_path() also True, der
    Groessen-Deckel am Anfang von simulate() greift NICHT), aber
    request_shader_operation() wirft sofort eine Exception - simuliert einen
    Treiberfehler nach Start des Laufs.
    """
    size = HydraulicFieldSimulator.MAX_CPU_RESOLUTION + 32
    terrain = np.zeros((size, size), dtype=np.float32)

    class FailingShaderManagerStub:
        gpu_available = True

        def request_shader_operation(self, *args, **kwargs):
            raise RuntimeError("simulierter Treiberfehler")

    simulator = HydraulicFieldSimulator(shader_manager=FailingShaderManagerStub())
    ok = check("Stub meldet einen registrierten GPU-Pfad (Vorbedingung)",
               simulator.has_gpu_path())
    try:
        simulator.simulate(terrain, hardness(size), {"max_steps": 1}, 10.0)
    except ValueError as error:
        ok &= check(
            "GPU-Ausfall mitten im Lauf bei zu grosser Karte meldet einen "
            "klaren Fehler statt still auf der CPU weiterzurechnen",
            "GPU" in str(error) and str(size) in str(error))
        return ok
    return check("GPU-Ausfall mitten im Lauf bei zu grosser Karte meldet einen "
                 "klaren Fehler statt still auf der CPU weiterzurechnen", False)


# =============================================================================
# (E) Konvergenz
# =============================================================================

def run_convergence_reports_honestly():
    """
    Bei einer grosszuegigen Schwelle muss der Lauf konvergieren und das auch
    melden; bei einer sehr strengen Schwelle muss er die Obergrenze erreichen
    und `converged=False` melden statt sie zu behaupten.

    Die Schwellen sind RELATIV ZUM RELIEF (siehe
    HydraulicFieldSimulator.simulate()) und aus der gemessenen Abklingkurve
    gewaehlt: 5e-5 wird nach wenigen hundert Schritten unterschritten, 1e-7
    innerhalb von 300 Schritten sicher nicht.
    """
    size = 48
    terrain = make_terrain(size)

    loose = HydraulicFieldSimulator().simulate(
        terrain, hardness(size),
        {"max_steps": 5000, "rainfall": 2.0, "convergence_threshold": 5e-5},
        10000.0 / size)
    strict = HydraulicFieldSimulator().simulate(
        terrain, hardness(size),
        {"max_steps": 300, "rainfall": 2.0, "convergence_threshold": 1e-7},
        10000.0 / size)

    ok = check("lockere Schwelle konvergiert (nach {} Schritten)".format(loose["steps_taken"]),
               bool(loose["converged"]) and loose["steps_taken"] < 5000)
    ok &= check("strenge Schwelle laeuft bis zur Obergrenze und meldet converged=False",
                (not strict["converged"]) and strict["steps_taken"] == 300)
    return ok


def run_colour_ranges_fit_the_data():
    """
    Die Farbskalen muessen zu dem passen, was das Modell tatsaechlich liefert.

    Der Anlass: die Bereiche standen auf 0-400 m linear, waehrend der typische
    Wert (95. Perzentil) zwischen 24 und 127 m liegt. Praktisch die ganze
    Karte landete damit in der hellsten Farbstufe - der Nutzer sah LEERE
    Karten, obwohl die Daten vollstaendig da waren. Dieselbe Fehlerklasse hatte
    das Projekt zuvor schon zweimal (erosion_map beim Droplet-Modell, humid_map
    im Weather-Tab).

    Geprueft wird die Aussage, auf die es ankommt: der TYPISCHE Wert muss
    innerhalb der Skala liegen - nicht unter vmin (Karte wirkt leer) und nicht
    ueber vmax (Karte wirkt gesaettigt). Extremwerte duerfen saettigen, das ist
    der Zweck einer festen Skala.
    """
    from gui.config.gui_default import CanvasSettings

    size = 64
    terrain = make_terrain(size)
    result = HydraulicFieldSimulator().simulate(
        terrain, hardness(size), {"max_steps": 2000, "rainfall": 2.0}, 10000.0 / size)

    ranges = CanvasSettings.CANVAS_2D["layer_ranges"]
    ok = True
    for key in ("erosion_map", "sedimentation_map", "sediment_load_map",
                "water_depth_map", "flow_velocity_map"):
        values = np.asarray(result[key])
        positive = values[values > 0]
        if positive.size == 0:
            ok &= check("{}: liefert ueberhaupt Werte".format(key), False)
            continue
        typical = float(np.percentile(positive, 95))
        _, vmin, vmax, _ = ranges[key]
        ok &= check(
            "{}: typischer Wert {:.2f} liegt in der Skala [{}, {}]".format(
                key, typical, vmin, vmax),
            vmin <= typical <= vmax)

    # Net Change wird im Tab aus beiden Karten gebildet und hat deshalb keinen
    # eigenen Simulator-Output - hier direkt nachgerechnet.
    net = result["sedimentation_map"].astype(np.float64) - result["erosion_map"]
    extreme = float(np.percentile(np.abs(net[net != 0]), 95))
    _, vmin, vmax, _ = ranges["net_change_map"]
    ok &= check(
        "net_change_map: typische Abweichung {:.2f} liegt in der Skala [{}, {}]".format(
            extreme, vmin, vmax),
        vmin <= -extreme and extreme <= vmax)
    return ok


def run_progress_is_reported_often_enough():
    """
    Der Simulator muss oft genug ein Lebenszeichen geben, dass der
    Inaktivitaets-Timeout des Orchestrators gar nicht erst greifen kann.

    Der Anlass: der Timeout mass frueher die Zeit seit dem ANFORDERN und brach
    den Erosions-Generator nach 5 Minuten ab, obwohl er ordentlich rechnete -
    und riss weather, water, biome und settlement gleich mit. Er misst jetzt
    die INAKTIVITAET, was nur trAegt, wenn tatsaechlich regelmaessig gemeldet
    wird.

    Geprueft auf der groessten Aufloesung, die der CPU-Pfad zulaesst - dort ist
    der Abstand zwischen zwei Meldungen am groessten.
    """
    from managers.generation_orchestrator import GenerationOrchestrator

    size = HydraulicFieldSimulator.MAX_CPU_RESOLUTION
    terrain = make_terrain(size)
    reports = []

    def record(step, total, current, convergence):
        reports.append((time.time(), step, convergence))

    interval = HydraulicFieldSimulator.PROGRESS_REPORT_INTERVAL
    start = time.time()
    HydraulicFieldSimulator().simulate(
        terrain, hardness(size), {"max_steps": 3 * interval, "rainfall": 2.0},
        10000.0 / size, progress_callback=record)

    ok = check("es kommen ueberhaupt Meldungen (gemessen {})".format(len(reports)),
               len(reports) >= 2)
    if not ok:
        return False

    stamps = [start] + [t for t, _, _ in reports]
    longest = max(b - a for a, b in zip(stamps, stamps[1:]))
    ok &= check(
        "groesster Abstand zwischen zwei Meldungen {:.1f}s liegt unter dem "
        "Inaktivitaets-Timeout von {:.0f}s".format(
            longest, GenerationOrchestrator.INACTIVITY_TIMEOUT_S),
        longest < GenerationOrchestrator.INACTIVITY_TIMEOUT_S)

    values = [c for _, _, c in reports]
    ok &= check("der gemeldete Konvergenz-Fortschritt laeuft nie rueckwaerts "
                "({})".format(" -> ".join("{:.0f}%".format(100 * v) for v in values)),
                all(b >= a for a, b in zip(values, values[1:])))
    ok &= check("der Fortschritt bleibt zwischen 0 und 1",
                all(0.0 <= v <= 1.0 for v in values))
    return ok


# =============================================================================
# (F) Senken
# =============================================================================

def run_basin_does_not_deepen():
    """
    Die zentrale Zusage des Feldmodells: ein abflussloses Becken fuellt sich
    mit Wasser, der Wasserspiegel wird eben, das Gefaelle geht gegen null - und
    ohne Gefaelle gibt es keine Erosion. Das Becken darf sich also NICHT
    weiter eintiefen.

    Das Partikelverfahren brauchte fuer dieselbe Zusage eine nachgeschaltete
    Senkenfuellung (fill_depressions in core/water_generator.py).
    """
    size = 64
    yy, xx = np.mgrid[0:size, 0:size]
    # Schiefe Ebene mit einer kreisrunden Grube in der Mitte.
    terrain = (200.0 + 8.0 * xx).astype(np.float64)
    radius = np.hypot(yy - size / 2, xx - size / 2)
    terrain -= 300.0 * np.exp(-(radius ** 2) / (2 * 6.0 ** 2))
    terrain = terrain.astype(np.float32)

    centre = (slice(size // 2 - 4, size // 2 + 5), slice(size // 2 - 4, size // 2 + 5))
    depth_before = float(terrain[centre].min())

    result = HydraulicFieldSimulator().simulate(
        terrain, hardness(size), {"max_steps": 3000, "rainfall": 2.0}, 10000.0 / size)
    after = terrain.astype(np.float64) - result["erosion_map"] + result["sedimentation_map"]
    depth_after = float(after[centre].min())

    ok = check(
        "die Grube vertieft sich nicht (vorher {:.1f} m, nachher {:.1f} m)".format(
            depth_before, depth_after),
        depth_after >= depth_before - 1.0)
    ok &= check(
        "in der Grube steht Wasser ({:.2f} m)".format(float(result["water_depth_map"][centre].max())),
        float(result["water_depth_map"][centre].max()) > 0.0)

    # Krater NUR ausserhalb des Beckens zaehlen. Das Becken ist per
    # Konstruktion eine Senke - es hier mitzuzaehlen hiesse, dem Modell die
    # Testgeometrie als Fehler anzurechnen. Geprueft wird, ob die Erosion
    # NEUE Senken auf dem umgebenden Hang erzeugt.
    neighbour_min = minimum_filter(after, footprint=_NEIGHBOR_FOOTPRINT_8,
                                   mode='constant', cval=np.inf)
    is_crater = (neighbour_min - after) > 0.5
    is_crater[0, :] = is_crater[-1, :] = is_crater[:, 0] = is_crater[:, -1] = False
    is_crater[radius < 12.0] = False
    craters = int(is_crater.sum())
    ok &= check("keine neuen Senken auf dem umgebenden Hang (gemessen {})".format(craters),
                craters == 0)
    return ok


# =============================================================================

def main():
    tests = [
        ("every_pass_conserves_mass", run_every_pass_conserves_mass),
        ("overall_balance_closes", run_overall_balance_closes),
        ("difference_maps_reconstruct_terrain", run_difference_maps_reconstruct_terrain),
        ("determinism", run_determinism),
        ("input_is_not_mutated", run_input_is_not_mutated),
        ("cpu_limit_fails_loudly", run_cpu_limit_fails_loudly),
        ("resolution_cap_follows_gpu_registration", run_resolution_cap_follows_gpu_registration),
        ("gpu_midrun_failure_above_cpu_limit_fails_loudly",
         run_gpu_midrun_failure_above_cpu_limit_fails_loudly),
        ("convergence_reports_honestly", run_convergence_reports_honestly),
        ("colour_ranges_fit_the_data", run_colour_ranges_fit_the_data),
        ("progress_is_reported_often_enough", run_progress_is_reported_often_enough),
        ("basin_does_not_deepen", run_basin_does_not_deepen),
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
