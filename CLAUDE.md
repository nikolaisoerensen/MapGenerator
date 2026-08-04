# MapGenerator — Project Notes for Claude Code

## ZUERST LESEN: docs/SPEZIFIKATION.md

Sie enthaelt das Oberziel, die Zielwerte je Komponente und die Invarianten, die
bei JEDER Aenderung geprueft werden (CPU/GPU-Paritaet, Massenbilanz, Zeitbasen,
Reihenfolge im Graph, Anzeige und Skalen, Reglerverhalten).

Sie ist am 2026-07-29 entstanden, weil die Arbeit reaktiv geworden war: jeweils
dem letzten Befund nachlaufend, ohne Zielbild pro Komponente. Ergebnis waren
drei Messungen am falschen Codepfad an einem Tag und Aenderungen, die anderswo
etwas kaputt machten, ohne dass es auffiel. Die Spezifikation ist das
Gegenmittel - vor der Arbeit lesen, nach der Arbeit die Prueflisten abgehen.


## Git worktrees: changes are invisible until merged or tested in-place

This project is frequently worked on via Claude Code sessions that run in an
isolated **git worktree** under `.claude/worktrees/<name>/`, on a branch like
`claude/<name>`. That worktree is a **separate directory and working tree**
from the user's main checkout at the repo root
(`C:\Lokale Dateien\Projects\Python\MapGenerator`, branch `main`).

**Editing files in a worktree does NOT affect the user's main checkout in any
way**, and the running desktop app (`python main.py`) only reflects whatever
directory it was actually launched from. If the user reports "nothing
changed" / "still looks exactly the same" after a fix, **check this before
re-investigating the original bug**:

```
git worktree list
git status --short   # confirm the fix is actually sitting uncommitted here
```

If the user is testing from their normal checkout and the current session's
work is in a worktree, the fix is real but invisible to them until one of:

1. **Test in place (fastest, no merge needed):** run the app directly from
   the worktree directory instead of the main checkout:
   ```
   cd "C:\Lokale Dateien\Projects\Python\MapGenerator\.claude\worktrees\<name>"
   "C:\Lokale Dateien\Projects\Python\MapGenerator\.venv\Scripts\python.exe" main.py
   ```
   Note: the venv lives at the **main checkout root**
   (`...\MapGenerator\.venv`), not per-worktree — worktrees don't get their
   own venv, always invoke the shared one explicitly as shown above.
2. **Commit + PR:** commit the worktree's changes and open a PR against
   `main` for the user to review and merge normally.
3. **Commit + direct merge:** commit on the worktree branch, user pulls/merges
   `claude/<name>` into their main checkout locally without a PR.

Always ask the user which of these they want — don't assume. Default to
suggesting option 1 first when the user just wants to see if a fix works,
since it requires no merge/commit decision at all.

## Verifying backend/generator changes without the live GUI

Most core generator logic (`core/*.py`) can be exercised headlessly via
throwaway smoke-test scripts run through the shared venv — see prior session
memory for established patterns (stubbing `managers.calculator_graph`
if missing on a given branch, building a minimal `FakeScheduler`, driving
`BaseTerrainGenerator` → `WeatherSystemGenerator` → `HydrologySystemGenerator`
→ `BiomeClassificationSystem` end-to-end with real default parameters from
`gui/config/value_default.py`). This validates the actual computation but
**not** the Qt/OpenGL *rendering* — GUI-facing changes (matplotlib colormaps,
Qt widget behavior, the 3D mesh) still need the user to confirm visually
against the live app, per the worktree note above.

### Compute shaders CAN be run headlessly — this was wrong before

The sentence above used to say GLSL shaders needed the live app. That is
**false for compute shaders**, and believing it cost three debugging rounds
(a reserved word as an identifier, an `int` uniform set with `glUniform1f`,
and a fixed-point counter overflowing) — each found only after the user ran
the program.

`GPUWorker` in `managers/shader_manager.py` is deliberately built as an
*offscreen* worker: its own `QOffscreenSurface`, its own `QOpenGLContext`, no
window. All it needs is a `QGuiApplication`, which a script can create:

```python
from PyQt6.QtGui import QGuiApplication
app = QGuiApplication([])                    # no window appears
from managers.shader_manager import ShaderManager
manager = ShaderManager()
worker = manager._ensure_worker()
assert worker.gpu_available                  # verified True on this machine
result = manager.request_shader_operation("erosion", "hydraulicField", inputs, {})
```

Working example: `tests/smoke_test_erosion_gpu_parity.py` — it runs the full GPU
erosion path and compares it against the CPU reference, single-step (tight
tolerance) and over a long run (mass balance). Any new compute shader should
get the same treatment; the static contract check
(`tests/smoke_test_erosion_gpu_contract.py`) catches naming and type mismatches, but
only an actual run catches wrong *results*.

What still needs the live app: anything drawn into a visible widget.


## Abhaengigkeiten: numpy ist bewusst NICHT die neueste Version

`requirements.txt` pinnt `numpy==2.4.6`, obwohl 2.5.1 verfuegbar ist. Das ist
kein vergessenes Update.

`numba` (selbst auf der neuesten Version 0.66.0) verlangt NumPy <= 2.4. Mit
numpy 2.5 schlaegt `import numba` fehl - und **opensimplex faengt das ab** und
benutzt still seinen Attrappen-Dekorator statt des JIT. Es gibt keine
Fehlermeldung, nur ein Programm, das deutlich laenger rechnet. Gemessen am
2026-07-30, `noise2array` 256x256:

| | Zeit |
|---|---|
| numpy 2.5.1, numba tot | 0.3831 s |
| numpy 2.4.6, numba aktiv | 0.0018 s |

Faktor 213 im heissen Pfad; die Rauscherzeugung machte frueher 36 % der
gesamten Pipeline aus.

Vor jedem `pip install --upgrade numpy` also pruefen:

```
.venv/Scripts/python.exe -c "import numba, numpy; print(numpy.__version__)"
```

Schlaegt der Import fehl, ist numpy zu neu und der JIT-Pfad ist weg.
Begruendung und Messwerte ausfuehrlich in `requirements.txt`.


## Beim Verschieben von Dateien: Pfade aus `__file__` mitzaehlen

Am 2026-07-30 wanderte `shader_manager.py` von `gui/OldManagers/` nach
`managers/`, also eine Ebene nach oben. Die Berechnung von `SHADERS_ROOT` ging
weiterhin ZWEI Ebenen hoch und zeigte damit neben das Projekt. Die gesamte
Pipeline rechnete danach ohne GPU.

**Der Fehler war unsichtbar.** Jede GPU-Operation faengt ihren Fehler ab und
faellt still auf den CPU-Pfad zurueck; im Log stand nur eine WARNING je
Aufruf, das Programm lief scheinbar normal weiter. Alle Smoke-Tests blieben
gruen, weil sie den CPU-Pfad benutzen.

Die Pruefung nach dem Umzug hatte nur **Importe** getestet - und eine
Pfadberechnung aus `__file__` ist beim Import unsichtbar. Ein Modul kann
tadellos importieren und trotzdem auf ein Verzeichnis zeigen, das es nicht
gibt.

Nach jedem Verschieben deshalb zusaetzlich:

1. Jede `os.path.dirname(...__file__...)`-Kette auflaesen und pruefen, dass
   das Ziel noch im Projekt liegt.
2. `tests/smoke_test_shader_paths.py` laufen lassen - er prueft `SHADERS_ROOT`
   und die Existenz aller 31 per `get_program()` angeforderten Shader.
3. Eine echte GPU-Operation fahren und dabei auf WARNINGs achten, nicht nur
   auf den Rueckgabewert.
