"""
Path: tests/smoke_test_noise_offset_gpu.py

Prueft den WELTVERSATZ im Noise-Shader gegen die CPU-Referenz.

WARUM ES DIESEN TEST GIBT. Bis 2026-08-04 kannte
shaders/terrain/noiseGeneration.comp keinen Versatz, und
SimplexNoiseGenerator.generate_noise_grid() sprang bei offset != 0 STILL auf
die CPU:

    if self._gpu_available() and offset_x == 0 and offset_y == 0:

Kein Fehler, keine WARNING - nur ein Programm, das ein Vielfaches laenger
rechnet. Genau die Falle, die CLAUDE.md fuer SHADERS_ROOT beschreibt: ein
Modul kann tadellos importieren und trotzdem den falschen Pfad nehmen.

Die Zoom-Pyramide (Makro/Meso/Mikro auf dieselbe Weltstelle) braucht den
Versatz fuer JEDES Fenster ausser dem am Weltursprung. Ohne ihn liefe die
gesamte Gelaendeerzeugung auf der CPU.

Geprueft wird:

    1. Der Shader hat u_offset_x und u_offset_y ueberhaupt (statisch).
    2. Ohne Versatz liefert die GPU dasselbe wie vorher (keine Regression).
    3. Mit Versatz stimmt die GPU mit der CPU-Referenz ueberein.
    4. Ein Versatz von n Pixeln ergibt DIESELBEN Werte wie das um n Pixel
       verschobene Fenster - die Kerneigenschaft, auf der die Pyramide steht.
    5. Der GPU-Pfad wird bei Versatz auch wirklich genommen (kein stiller
       Rueckfall), nachgewiesen ueber die Rechenzeit.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_noise_offset_gpu.py
"""

import os
import re
import sys
import time

import numpy as np

import os as _os
_PROJEKTWURZEL = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _PROJEKTWURZEL)

# Die Qt-Anwendung MUSS modulweit gehalten werden - als lokale Variable raeumt
# Python sie ab, waehrend der GL-Kontext noch lebt (Segfault ohne Meldung).
_QT_APP = None


def _qt():
    global _QT_APP
    from PyQt6.QtGui import QGuiApplication
    if _QT_APP is None:
        _QT_APP = QGuiApplication.instance() or QGuiApplication([])
    return _QT_APP


SIZE = 128
OKTAVEN = 6
FREQ = 0.012
PERS = 0.65
LAC = 2.0
SEED = 20260804


def lauf():
    _qt()
    from managers.shader_manager import ShaderManager, SHADERS_ROOT
    from core.terrain_generator import SimplexNoiseGenerator

    fehler = []

    # ---------- 1 ---------- statisch, vor jedem Dispatch
    quelle = os.path.join(SHADERS_ROOT, "terrain", "noiseGeneration.comp")
    with open(quelle, "r", encoding="utf-8") as datei:
        text = datei.read()
    fehlend = [n for n in ("u_offset_x", "u_offset_y")
               if not re.search(r"uniform\s+float\s+%s\s*;" % n, text)]
    ok = not fehlend
    print("1. Shader deklariert u_offset_x/u_offset_y ... %s"
          % ("ok" if ok else "FEHLER: %s fehlt" % ", ".join(fehlend)))
    if not ok:
        fehler.append("Shader ohne Versatz-Uniforms")
        return _ende(fehler)
    # Der Versatz muss VOR der Frequenz addiert werden - andere Reihenfolge
    # ergibt ein anderes Feld als die CPU.
    if not re.search(r"float\s+x\s*=\s*float\(coord\.x\)\s*\+\s*u_offset_x", text):
        fehler.append("u_offset_x wird nicht vor der Frequenz addiert")

    manager = ShaderManager()
    worker = manager._ensure_worker()
    if not worker.gpu_available:
        print("   GPU nicht verfuegbar - Test kann nichts aussagen.")
        return 2

    def gpu(versatz_x=0.0, versatz_y=0.0):
        return manager.process_noise_generation(
            size=SIZE, octaves=OKTAVEN, frequency=FREQ, persistence=PERS,
            lacunarity=LAC, seed=SEED, offset_x=versatz_x, offset_y=versatz_y)

    def cpu(versatz_x=0.0, versatz_y=0.0):
        gen = SimplexNoiseGenerator()
        gen.set_seed(SEED)
        return gen._generate_cpu_optimized({
            "size": SIZE, "frequency": FREQ, "octaves": OKTAVEN,
            "persistence": PERS, "lacunarity": LAC,
            "offset_x": versatz_x, "offset_y": versatz_y})

    def vergleich(a, b):
        spanne = float(max(a.max() - a.min(), 1e-9))
        return (float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
                float(np.abs(a - b).max()) / spanne)

    # ---------- 2 ---------- keine Regression ohne Versatz
    r, abw = vergleich(gpu(), cpu())
    ok = r > 0.99 and abw < 0.05
    print("2. Ohne Versatz GPU gegen CPU: r=%+.5f, groesste Abweichung %.3f "
          "der Spanne ... %s" % (r, abw, "ok" if ok else "FEHLER"))
    if not ok:
        fehler.append("GPU und CPU weichen schon ohne Versatz ab")

    # ---------- 3 ---------- mit Versatz
    print("3. Mit Versatz GPU gegen CPU:")
    for vx, vy in ((37.0, 0.0), (0.0, 121.0), (2048.0, 2048.0),
                   (-853.5, 1290.25)):
        r, abw = vergleich(gpu(vx, vy), cpu(vx, vy))
        gut = r > 0.99 and abw < 0.05
        print("   Versatz (%9.2f, %9.2f): r=%+.5f, Abweichung %.3f  %s"
              % (vx, vy, r, abw, "ok" if gut else "FEHLER"))
        if not gut:
            fehler.append("Versatz (%.1f, %.1f): r=%.4f, Abweichung %.3f"
                          % (vx, vy, r, abw))

    # ---------- 4 ---------- die Kerneigenschaft der Pyramide
    #
    # Ein Versatz von n Pixeln muss DASSELBE Feld liefern wie der um n Pixel
    # verschobene Ausschnitt. Sonst haengt die Landschaft am Bildausschnitt und
    # die Zoomstufen zeigen verschiedene Welten.
    n = 32
    gross = manager.process_noise_generation(
        size=SIZE, octaves=OKTAVEN, frequency=FREQ, persistence=PERS,
        lacunarity=LAC, seed=SEED)
    verschoben = gpu(float(n), float(n))
    a = gross[n:, n:]
    b = verschoben[:SIZE - n, :SIZE - n]
    r, abw = vergleich(a, b)
    ok = r > 0.999 and abw < 0.02
    print("4. Versatz %d px == derselbe Ausschnitt %d px weiter: "
          "r=%+.5f, Abweichung %.4f ... %s" % (n, n, r, abw,
                                               "ok" if ok else "FEHLER"))
    if not ok:
        fehler.append("Versatz verschiebt das Feld nicht deckungsgleich")

    # ---------- 5 ---------- laeuft es wirklich auf der GPU?
    #
    # Der Rueckgabewert allein beweist nichts: bei einem Fehlschlag liefert
    # process_noise_generation still das CPU-Ergebnis. Die Rechenzeit
    # unterscheidet die beiden Pfade.
    gross_size = 512
    t = time.perf_counter()
    manager.process_noise_generation(
        size=gross_size, octaves=OKTAVEN, frequency=FREQ, persistence=PERS,
        lacunarity=LAC, seed=SEED, offset_x=1234.0, offset_y=5678.0)
    t_gpu = time.perf_counter() - t
    t = time.perf_counter()
    manager._cpu_fallback_noise(gross_size, OKTAVEN, FREQ, PERS, LAC, SEED,
                                offset_x=1234.0, offset_y=5678.0)
    t_cpu = time.perf_counter() - t
    ok = t_gpu < 0.5 * t_cpu
    print("5. Bei Versatz wirklich GPU: %.3f s gegen %.3f s auf der CPU "
          "(%.1f x) ... %s" % (t_gpu, t_cpu, t_cpu / max(t_gpu, 1e-9),
                               "ok" if ok else "FEHLER - stiller Rueckfall?"))
    if not ok:
        fehler.append("GPU-Pfad wird bei Versatz offenbar nicht genommen "
                      "(%.3f s gegen %.3f s)" % (t_gpu, t_cpu))

    return _ende(fehler)


def _ende(fehler):
    print()
    if fehler:
        print("NICHT IN ORDNUNG - %d Befunde:" % len(fehler))
        for eintrag in fehler:
            print("   %s" % eintrag)
        return 1
    print("Alle fuenf Zusicherungen erfuellt - der Weltversatz laeuft auf der GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(lauf())
