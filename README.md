# MapGenerator

MapGenerator is a Python desktop application for procedural world generation. It combines terrain, geology, weather, water, biome, and settlement systems behind a PyQt6 GUI with 2D and experimental 3D map views.

The project is under active development. The main editor opens and the generation systems are being wired through a calculator-aware pipeline, but some actions and tab controls are still placeholders.

## Features

- PyQt6 desktop UI with a main menu and map editor.
- Multi-step generation pipeline for terrain, geology, weather, water, biome, and settlements.
- LOD-aware data management and generation orchestration.
- 2D rendering through Matplotlib.
- Experimental 3D terrain rendering through PyOpenGL.
- Pipeline status panel and per-generator parameter tabs.
- Standalone plot/biome physics lab for settlement and topology experiments.

## Requirements

- Python 3.12 or newer.
- `uv` for dependency resolution, locking, and running the app.
- A working Qt environment for the GUI.
- A working OpenGL environment for GPU features and the 3D viewport.

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is intentionally no `requirements.txt`; use `uv sync`/`uv run` from the project metadata instead.

## Setup

From the repository root:

```powershell
uv sync
```

Run the application:

```powershell
uv run python main.py
```

Run the standalone plot/biome physics lab:

```powershell
uv run python tools/plot_physics_lab.py
```

For a headless or CI-style import check, Qt can use the offscreen platform:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run python -c "from PyQt6.QtWidgets import QApplication; app = QApplication([]); from gui.map_editor import MapEditorWindow; MapEditorWindow(); print('map editor constructed ok')"
```

## Project Layout

```text
core/       Generator implementations for terrain, geology, weather, water, biome, and settlement data.
gui/        PyQt6 application windows, tabs, widgets, managers, and display components.
shaders/    GLSL shader assets used by display and generation systems.
tools/      Standalone developer tools, including the plot/biome physics lab.
docs/       Design notes, backlog, and pipeline dependency documentation.
main.py     Application entry point.
```

## Generation Pipeline

The high-level generator order is:

1. Terrain
2. Geology
3. Weather
4. Water
5. Biome
6. Settlement

The detailed calculator-level dependency graph is documented in `docs/generation_pipeline_dependencies.md`.

## Development Notes

- Keep dependency changes in `pyproject.toml`, then run `uv lock`.
- The application is PyQt6-only. Avoid adding PyQt5 imports or Qt5 Matplotlib backends.
- `scikit-image` is required for `skimage.measure` usage in biome/settlement contour logic.
- Generated caches, local virtual environments, profiling output, and exported map artifacts are ignored by `.gitignore`.
- Useful smoke checks:

```powershell
uv run python -m compileall -q main.py gui core tools
$env:QT_QPA_PLATFORM = "offscreen"
uv run python -c "from PyQt6.QtWidgets import QApplication; app = QApplication([]); from gui.map_editor import MapEditorWindow; MapEditorWindow(); print('map editor constructed ok')"
```

## Known Caveats

- Some settings, load/save, export, and report actions are still placeholders.
- Some tabs still use placeholder control creation hooks while the parameter UI is being refactored.
- Headless/offscreen environments may log that the offscreen OpenGL context cannot be activated. That affects GPU/OpenGL acceleration checks, not basic editor construction.
