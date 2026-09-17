"""
Path: tests/smoke_test_erosion_kein_stiller_cpu_ruecksprung.py

Deckt die in Ticket #30 gefundene Luecke ab: scheitert die GPU-Erosion nicht
gleich zu Laufbeginn (dafuer gab es schon MAX_CPU_RESOLUTION als Wache),
sondern ERST WAEHREND eines laufenden Abschnitts (Treiberfehler, Timeout in
GPUWorker.submit), fiel HydraulicFieldSimulator.simulate() bisher ungeprueft
in die volle CPU-Schleife durch - bei 1024 px gemessen bis zu ~2h, ein
stiller Haenger statt eines Fehlers (siehe docs/TESTBERICHT.md, Nachtrag
17.09.2026).

Braucht KEINE echte GPU: has_gpu_path() prueft nur die statische
DISPATCH_TABLE aus managers/shader_manager.py, nicht echte Hardware.
_simulate_gpu() wird hier absichtlich auf einen Fehlschlag gestellt (gibt
None zurueck, wie es der echte Code bei einer Ausnahme oder einem
misslungenen Dispatch auch tut), um beide Seiten der neuen Wache zu pruefen.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_erosion_kein_stiller_cpu_ruecksprung.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.erosion_generator import HydraulicFieldSimulator


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


class _FakeShaderManager:
    """Reicht, damit has_gpu_path() True liefert (shader_manager is not None
    und die Operation steht in der echten, statischen DISPATCH_TABLE)."""


def _simulator_mit_gpu_fehlschlag():
    sim = HydraulicFieldSimulator(shader_manager=_FakeShaderManager())
    sim._simulate_gpu = lambda *a, **k: None  # simuliert einen Laufzeitfehler
    return sim


def run_grosse_aufloesung_wirft_laut():
    sim = _simulator_mit_gpu_fehlschlag()
    size = 300  # > MAX_CPU_RESOLUTION (256)
    terrain = np.full((size, size), 100.0, dtype=np.float32)
    hardness = np.full((size, size), 50.0, dtype=np.float32)

    ok = check("has_gpu_path() ist True (Vorbedingung des Tests)",
                sim.has_gpu_path())

    try:
        sim.simulate(terrain, hardness, {"max_steps": 5}, meters_per_pixel=10.0)
        ok &= check(f"simulate() wirft bei {size}px nach GPU-Fehlschlag "
                    "einen ValueError", False, "kein Fehler geworfen")
    except ValueError as exc:
        ok &= check(f"simulate() wirft bei {size}px nach GPU-Fehlschlag "
                    "einen ValueError", True, str(exc)[:80] + "...")
    return ok


def run_kleine_aufloesung_faellt_weiter_zurueck():
    """Unterhalb MAX_CPU_RESOLUTION bleibt der CPU-Ruecksprung eine legitime
    Notloesung (z.B. die 64px-Testkarten in smoke_test_erosion_gpu_parity.py)
    - dort darf die neue Wache NICHT greifen."""
    sim = _simulator_mit_gpu_fehlschlag()
    size = 64  # <= MAX_CPU_RESOLUTION (256)
    terrain = np.full((size, size), 100.0, dtype=np.float32)
    hardness = np.full((size, size), 50.0, dtype=np.float32)

    try:
        result = sim.simulate(terrain, hardness, {"max_steps": 5},
                              meters_per_pixel=10.0)
        ok = check(f"simulate() faellt bei {size}px nach GPU-Fehlschlag "
                    "weiterhin auf die CPU zurueck (kein Fehler)", True,
                    f"steps_taken={result['steps_taken']}")
    except ValueError as exc:
        ok = check(f"simulate() faellt bei {size}px nach GPU-Fehlschlag "
                    "weiterhin auf die CPU zurueck (kein Fehler)", False, str(exc))
    return ok


def main():
    print("=" * 78)
    print("KEIN STILLER CPU-RUECKSPRUNG BEI GROSSER AUFLOESUNG (Ticket #30)")
    print("=" * 78)
    print()

    ergebnis = {}
    for titel, fn in (
        ("Grosse Aufloesung wirft laut", run_grosse_aufloesung_wirft_laut),
        ("Kleine Aufloesung faellt weiter zurueck", run_kleine_aufloesung_faellt_weiter_zurueck),
    ):
        print(f"--- {titel} ---")
        ergebnis[titel] = fn()
        print()

    print("=" * 78)
    gut = sum(1 for v in ergebnis.values() if v)
    print(f"{gut}/{len(ergebnis)} Gruppen gruen")
    return 0 if gut == len(ergebnis) else 1


if __name__ == "__main__":
    sys.exit(main())
