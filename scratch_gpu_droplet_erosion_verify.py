"""
Wegwerf-Skript: verifiziert den neuen GPU-Dispatch fuer Droplet-Erosion
(("water", "dropletErosion"), gui/OldManagers/shader_manager.py
_dispatch_droplet_erosion) direkt ueber ShaderManager.request_shader_operation(),
OHNE die volle GUI - GPUWorker nutzt bereits QOffscreenSurface/QOpenGLContext,
das braucht kein sichtbares Fenster. Nicht Teil der offiziellen Smoke-Test-
Suite (GPU-Pfade sind sonst nicht headless testbar, siehe CLAUDE.md) - dieses
Skript nutzt aber denselben Offscreen-Mechanismus, den GPUWorker selbst schon
verwendet, und ist deshalb ausnahmsweise doch headless moeglich.
"""
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from gui.OldManagers.shader_manager import ShaderManager
from core.water_generator import DropletErosionSystem


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    manager = ShaderManager()
    if not manager.gpu_available:
        print("[SKIP] Kein GL 4.3+ Kontext in dieser Umgebung verfuegbar - "
              "Verifikation muss ueber die Live-App erfolgen (siehe CLAUDE.md).")
        return True

    size = 64
    rng = np.random.RandomState(5)
    heightmap = (300.0 + 400.0 * np.exp(-((np.arange(size)[:, None] - size / 2) ** 2
                                           + (np.arange(size)[None, :] - size / 2) ** 2) / (2 * (size * 0.2) ** 2))
                 + 20.0 * rng.randn(size, size)).astype(np.float32)
    hardness_map = np.full((size, size), 50.0, dtype=np.float32)

    erosion_system = DropletErosionSystem(erosion_strength=2.5, sediment_capacity_factor=4.0, deposit_speed=0.3)
    spawn_positions = erosion_system._sample_spawn_positions(heightmap, 2000, {"water_seed": 7})

    result = manager.request_shader_operation(
        "water", "dropletErosion",
        {"heightmap": heightmap, "hardness_map": hardness_map, "spawn_positions": spawn_positions,
         "meters_per_pixel": 10000.0 / size, "erosion_strength": 2.5,
         "capacity_factor": 4.0, "deposit_speed": 0.3},
        {}
    )

    ok = check(f"GPU-Dispatch erfolgreich (result={result.get('reason', 'success')})", result.get("success", False))
    if not ok:
        return False

    erosion_map = result["erosion_map"]
    sedimentation_map = result["sedimentation_map"]

    ok &= check("erosion_map endlich", bool(np.all(np.isfinite(erosion_map))))
    ok &= check("sedimentation_map endlich", bool(np.all(np.isfinite(sedimentation_map))))
    ok &= check("erosion_map nicht komplett 0", bool(np.any(erosion_map > 0)))
    ok &= check("sedimentation_map nicht komplett 0", bool(np.any(sedimentation_map > 0)))

    total_erosion = float(erosion_map.sum())
    total_sedimentation = float(sedimentation_map.sum())
    relative_error = abs(total_erosion - total_sedimentation) / max(total_erosion, 1e-9)
    ok &= check(f"exakte Massenerhaltung (abgetragen={total_erosion:.4f}, "
                f"abgelagert={total_sedimentation:.4f}, rel. Fehler {100*relative_error:.6f}%)",
                relative_error < 1e-6)

    # Strukturplausibilitaet vs. CPU (nicht pixelgleich - siehe dokumentierte
    # Lockstep-Abweichung): gleiche Groessenordnung Gesamterosion.
    cpu_erosion_map, cpu_sedimentation_map = erosion_system._cpu_simulate(
        heightmap, hardness_map, spawn_positions, 10000.0 / size)
    cpu_total = float(cpu_erosion_map.sum())
    ratio = total_erosion / max(cpu_total, 1e-9)
    ok &= check(f"GPU-Gesamterosion in plausibler Groessenordnung zu CPU "
                f"(GPU={total_erosion:.4f}, CPU={cpu_total:.4f}, Verhaeltnis={ratio:.2f})",
                0.2 < ratio < 5.0)

    # Performance-Vergleich bei realistischer LOD4+-Partikelzahl.
    size_big = 512
    rng2 = np.random.RandomState(1)
    y, x = np.mgrid[0:size_big, 0:size_big].astype(np.float64)
    heightmap_big = (300.0 + 400.0 * np.exp(-((x - size_big / 2) ** 2 + (y - size_big / 2) ** 2)
                                             / (2 * (size_big * 0.2) ** 2))
                     + 20.0 * rng2.randn(size_big, size_big)).astype(np.float32)
    hardness_big = np.full((size_big, size_big), 50.0, dtype=np.float32)
    spawn_big = erosion_system._sample_spawn_positions(heightmap_big, 20000, {"water_seed": 1})

    t0 = time.perf_counter()
    result_big = manager.request_shader_operation(
        "water", "dropletErosion",
        {"heightmap": heightmap_big, "hardness_map": hardness_big, "spawn_positions": spawn_big,
         "meters_per_pixel": 10000.0 / size_big, "erosion_strength": 2.5,
         "capacity_factor": 4.0, "deposit_speed": 0.3},
        {}
    )
    t1 = time.perf_counter()
    ok &= check(f"512x512/20000 Partikel GPU-Dispatch erfolgreich, {t1 - t0:.2f}s "
                f"(CPU-Baseline: ~19.2s)", result_big.get("success", False))

    return ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
