"""
Path: smoke_test_erosion_gpu_parity.py

Fuehrt den GPU-Pfad der Feld-Erosion WIRKLICH AUS und vergleicht ihn mit der
CPU-Referenz.

WARUM DAS GEHT, obwohl CLAUDE.md "GPU nicht headless testbar" sagt: die Aussage
gilt fuer das RENDERING (MapDisplay3D-Widgets brauchen ein sichtbares Fenster).
Der GPUWorker in shader_manager.py ist dagegen bewusst als OFFSCREEN-Worker
gebaut - eigener QOffscreenSurface, eigener QOpenGLContext, kein Fenster. Es
genuegt eine QGuiApplication, und die laesst sich in einem Skript anlegen.

Der Unterschied ist teuer erkauft: drei GPU-Fehler in Folge fielen erst in der
laufenden App auf, weil dieser Test fehlte -

  1. `common` als Bezeichner (in GLSL reserviert) -> Shader kompilierte nicht
  2. `u_variant` fehlte in _INT_UNIFORM_NAMES -> glUniform1f auf int-Location
  3. Export-Zaehler als Fixed-Point-int32 -> Ueberlauf ab 2147 m Gesamtexport

Nummer 3 haette KEIN statischer Test gefunden: der Shader war syntaktisch und
vertraglich in Ordnung, nur das Ergebnis war falsch. Genau dafuer ist dieser
Test da.

OHNE GPU wird der Test uebersprungen und meldet das laut - "keine GPU" ist eine
Umgebungsbedingung, kein Defekt.

Aufruf: .venv\\Scripts\\python.exe smoke_test_erosion_gpu_parity.py
"""

import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")


def check(label, condition):
    print(("[OK] " if condition else "[FAIL] ") + label)
    return bool(condition)


def make_terrain(size, seed=5, amplitude=4000.0):
    rng = np.random.RandomState(seed)
    return (gaussian_filter(rng.rand(size, size), 3) * amplitude).astype(np.float32)


class StepProbe:
    """
    Erlaubt Laeufe mit WENIGER als 200 Schritten.

    HydraulicFieldSimulator._resolve_parameters() klemmt max_steps auf
    mindestens 200 - sinnvoll fuer echte Laeufe, aber es macht einen
    Einzelschritt-Vergleich unmoeglich. Ohne diese Umgehung verglich ein
    frueherer Versuch fuer 1, 2, 5 und 25 Schritte immer dasselbe Ergebnis
    (naemlich 200 Schritte) und zeigte dadurch eine scheinbare Abweichung von
    40%, die in Wahrheit die chaotische Verstaerkung ueber 200 Schritte war.
    """

    def __init__(self, base_class, shader_manager=None):
        self.simulator = base_class(shader_manager=shader_manager)
        original = self.simulator._resolve_parameters

        def unclamped(parameters):
            cfg = original(parameters)
            cfg["max_steps"] = int(parameters.get("max_steps", 1))
            return cfg

        self.simulator._resolve_parameters = unclamped

    def simulate(self, *args, **kwargs):
        return self.simulator.simulate(*args, **kwargs)


def run_all():
    from PyQt6.QtGui import QGuiApplication
    application = QGuiApplication([])  # noqa: F841 - haelt den GL-Kontext am Leben

    from gui.OldManagers.shader_manager import ShaderManager
    from core.erosion_generator import HydraulicFieldSimulator

    shader_manager = ShaderManager()
    worker = shader_manager._ensure_worker()
    if not worker.gpu_available:
        print("[SKIP] Keine GPU mit OpenGL 4.3 verfuegbar - Test uebersprungen.")
        print("       Das ist eine Umgebungsbedingung, kein Defekt.")
        return True

    size = 64
    terrain = make_terrain(size)
    hardness = np.full((size, size), 50.0, dtype=np.float32)
    meters_per_pixel = 10000.0 / size

    ok = check("der Erosions-Dispatch ist auf der GPU verfuegbar",
               HydraulicFieldSimulator(shader_manager=shader_manager).has_gpu_path())

    # --- (A) Einzelschritt: die Formeln muessen uebereinstimmen -----------
    print("\n=== single_step_parity ===")
    params = {"max_steps": 1, "rainfall": 2.0}
    gpu = StepProbe(HydraulicFieldSimulator, shader_manager).simulate(
        terrain, hardness, params, meters_per_pixel)
    cpu = StepProbe(HydraulicFieldSimulator).simulate(
        terrain, hardness, params, meters_per_pixel)

    for key, tolerance in (("water_depth_map", 1e-4), ("flow_velocity_map", 1e-3),
                           ("erosion_map", 1e-3), ("sedimentation_map", 1e-3)):
        a = gpu[key].astype(np.float64)
        b = cpu[key].astype(np.float64)
        scale = max(abs(b).max(), 1e-9)
        deviation = float(np.abs(a - b).max()) / scale
        ok &= check("{}: GPU und CPU stimmen nach EINEM Schritt ueberein "
                    "(Abweichung {:.2e}, erlaubt < {:.0e})".format(key, deviation, tolerance),
                    deviation < tolerance)

    # --- (B) Langer Lauf: Bilanz und Endlichkeit --------------------------
    print("\n=== long_run_balance ===")
    long_params = {"max_steps": 2000, "rainfall": 2.0}
    gpu_long = HydraulicFieldSimulator(shader_manager=shader_manager).simulate(
        terrain, hardness, long_params, meters_per_pixel)
    cpu_long = HydraulicFieldSimulator().simulate(
        terrain, hardness, long_params, meters_per_pixel)

    print("    GPU: {} Schritte, Export {:.1f} m, Bilanz {:+.3%}".format(
        gpu_long["steps_taken"], gpu_long["sediment_exported"], gpu_long["mass_balance"]))
    print("    CPU: {} Schritte, Export {:.1f} m, Bilanz {:+.3%}".format(
        cpu_long["steps_taken"], cpu_long["sediment_exported"], cpu_long["mass_balance"]))

    ok &= check("die GPU-Massenbilanz geht auf (Abweichung {:.3%})".format(
        abs(gpu_long["mass_balance"])), abs(gpu_long["mass_balance"]) < 0.02)

    # Der Ueberlauf des frueheren Fixed-Point-Zaehlers zeigte sich als
    # NEGATIVER Export - ein Wert, den es physikalisch nicht geben kann.
    ok &= check("der Export ueber den Kartenrand ist positiv (gemessen {:.1f} m)".format(
        gpu_long["sediment_exported"]), gpu_long["sediment_exported"] > 0.0)
    ok &= check("der Export liegt in derselben Groessenordnung wie auf der CPU "
                "({:.1f} gegen {:.1f} m)".format(
                    gpu_long["sediment_exported"], cpu_long["sediment_exported"]),
                0.5 < gpu_long["sediment_exported"] / max(cpu_long["sediment_exported"], 1e-9) < 2.0)

    ok &= check("alle GPU-Ausgaben sind endlich",
                all(np.all(np.isfinite(gpu_long[key])) for key in
                    ("erosion_map", "sedimentation_map", "sediment_load_map",
                     "water_depth_map", "flow_velocity_map")))
    ok &= check("keine Zelle steht in beiden Differenzkarten",
                int(((gpu_long["erosion_map"] > 0)
                     & (gpu_long["sedimentation_map"] > 0)).sum()) == 0)
    return ok


def main():
    print("=== gpu_available ===")
    try:
        passed = run_all()
    except Exception as error:  # pragma: no cover - Diagnose
        import traceback
        traceback.print_exc()
        print("[FAIL] GPU-Test mit Ausnahme abgebrochen: {}".format(error))
        passed = False

    print("\n=== SUMMARY ===")
    print("erosion_gpu_parity: {}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
