"""
Path: tests/smoke_test_water_lake_detection_gpu.py

GPU-Seedetektion (`water.lake_detection`, jumpFloodLakes.comp) WIRKLICH
ausgefuehrt und gegen die CPU-Referenz verglichen.

ANLASS (Pipeline-Audit 2026-08-11, docs/OFFENE_PUNKTE.md 7.1). Befund: die
GPU-`lake_map` war KONSTANT -1 - unabhaengig vom Gelaende fand der GPU-Pfad
niemals auch nur einen See.

URSACHE: `_classify_lake_basins_vectorized()` (managers/shader_manager.py)
verglich die rohe Wassertiefen-Summe (in "Meter-Pixel") direkt gegen
`lake_volume_threshold`, das seit 2026-07-27 ein echtes Volumen in m³ ist
(siehe core/water_generator.py `LakeDetectionSystem._cell_area_m2()` und
`_classify_lake_basins()`, die dort korrekt mit der Zellflaeche
multiplizieren). Bei realistischen `meters_per_pixel`-Werten (z.B. 100 m/px
-> Zellflaeche 10000 m²) fehlte dadurch ein Faktor von 10000x - jedes Becken
verfehlte die Schwelle. Fix: `meters_per_pixel` wird jetzt durchgereicht und
`total_volume = depth_sum * meters_per_pixel**2` vor dem Schwellenvergleich
gebildet, genau wie auf der CPU-Seite.

WARUM DAS GEHT, obwohl CLAUDE.md "GPU nicht headless testbar" sagt: der
GPUWorker ist bewusst als Offscreen-Worker gebaut (eigener
QOffscreenSurface/QOpenGLContext) - siehe smoke_test_erosion_gpu_parity.py
fuer dasselbe Muster. WICHTIG: KEIN `QT_QPA_PLATFORM=offscreen` setzen - das
verhindert einen echten GL-Kontext und `gpu_available` bleibt False (in
dieser Session live beobachtet).

Kein Bit-Identitaets-Anspruch: die Becken-ZUORDNUNG selbst unterscheidet sich
weiterhin bewusst zwischen GPU (Jump-Flooding, "current_height >= seed_height"
plus Luftlinien-Distanz) und CPU (echter Watershed-Transform seit
2026-07-27, siehe `_apply_priority_flood_watershed()`-Docstring in
core/water_generator.py) - das ist ein bekannter, dokumentierter
Algorithmus-Unterschied, kein Ziel dieses Fixes. Die einzige harte
Zusicherung hier: findet die CPU Seen, findet die GPU AUCH welche (nicht
mehr konstant leer).

Aufruf: .venv\\Scripts\\python.exe tests/smoke_test_water_lake_detection_gpu.py
"""
import sys

import numpy as np
from scipy.ndimage import zoom

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)


def check(label, condition):
    print(("[OK] " if condition else "[FAIL] ") + label)
    return bool(condition)


def make_terrain(seed, size=128):
    rng = np.random.RandomState(seed)
    roh = rng.rand(max(4, size // 16), max(4, size // 16)).astype(np.float64)
    return (zoom(roh, 16, order=3)[:size, :size] * 200.0 - 60.0).astype(np.float32)


def main():
    from PyQt6.QtGui import QGuiApplication
    application = QGuiApplication([])  # noqa: F841 - haelt den GL-Kontext am Leben

    from managers.shader_manager import ShaderManager
    from core.water_generator import LakeDetectionSystem

    manager = ShaderManager()
    worker = manager._ensure_worker()
    if not worker.gpu_available:
        print("[SKIP] keine GPU verfuegbar - Umgebungsbedingung, kein Defekt")
        return 0

    checks = []
    meters_per_pixel = 100.0
    volume_threshold = 5000.0

    for seed in (7, 11, 42):
        heightmap = make_terrain(seed)
        lake_sys = LakeDetectionSystem(
            lake_volume_threshold=volume_threshold, shader_manager=manager,
            meters_per_pixel=meters_per_pixel)

        gpu_result = manager.request_shader_operation(
            "water", "jumpFloodLakes",
            {"heightmap": heightmap, "lake_volume_threshold": volume_threshold,
             "meters_per_pixel": meters_per_pixel},
            {})
        checks.append(check(f"seed={seed}: GPU-Dispatch erfolgreich", gpu_result.get("success")))
        if not gpu_result.get("success"):
            continue
        gpu_lake_map = gpu_result["lake_map"]
        gpu_n = len(gpu_result["valid_lakes"])

        cpu_lake_map, cpu_lakes = lake_sys._cpu_lake_detection(heightmap)
        cpu_n = len(cpu_lakes)

        print(f"    seed={seed}: CPU {cpu_n} Seen, GPU {gpu_n} Seen")
        if cpu_n > 0:
            checks.append(check(
                f"seed={seed}: GPU findet ebenfalls Seen, wenn CPU welche findet (nicht mehr konstant -1)",
                gpu_n > 0))
            checks.append(check(
                f"seed={seed}: GPU-lake_map ist nicht ueberall -1",
                bool(np.any(gpu_lake_map >= 0))))

    print("\n=== SUMMARY ===")
    ok = all(checks)
    print("lake_detection_gpu_parity:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
