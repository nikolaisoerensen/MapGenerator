# MapGenerator

MapGenerator is a Python desktop application for procedural world generation. It combines terrain, geology, weather, water, biome, and settlement systems behind a PyQt6 GUI with 2D and 3D map views.

The project is still in active development. Some systems are functional, some UI paths are placeholders, and the pipeline is being moved toward more precise calculator-level dependency handling.

## Features

- PyQt6 desktop UI with a main menu and map editor.
- Multi-step generation pipeline for terrain, geology, weather, water, biome, and settlements.
- LOD-aware data management and generation orchestration.
- 2D rendering through Matplotlib.
- Experimental 3D terrain rendering through PyOpenGL.
- Pipeline status panel and per-generator parameter tabs.

## Requirements

- Python 3.12 or newer.
- `uv` for dependency resolution and running the app.
- A working Qt/OpenGL environment for the GUI and 3D viewport.

## Setup

From the repository root:

```powershell
uv sync
```

Run the application:

```powershell
uv run python main.py
```

If you are running in a headless or CI-like environment, Qt may need an offscreen platform:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
uv run python -c "import main; print('imports ok')"
```

## Project Layout

```text
core/       Generator implementations for terrain, geology, weather, water, biome, and settlement data.
gui/        PyQt6 application windows, tabs, widgets, and manager classes.
shaders/    GLSL shader assets used by display and generation systems.
docs/       Design notes and pipeline dependency documentation.
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

- Project metadata is in `pyproject.toml`.
- Dependencies are locked with `uv.lock`.
- The application currently uses PyQt6. Avoid adding new PyQt5 imports.
- Generated caches, local virtual environments, profiling output, and exported map artifacts are ignored by `.gitignore`.

## Known Caveats

- The GUI is under active refactoring, especially the generation orchestration and dependency model.
- Some export and settings actions are placeholders.
- Importing the full map editor can currently log a missing optional/internal module for the Biome tab: `gui.OldManagers.calculator_graph`. The editor catches tab import failures, but that dependency should be fixed before treating the full pipeline as production-ready.

