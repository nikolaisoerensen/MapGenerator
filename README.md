# MapGenerator

A procedural world/map generator with a PyQt6 desktop GUI. Generates a full
map through six coupled stages — Terrain, Geology, Weather, Water, Biome, and
Settlement — each with its own generator module and its own editor tab,
including live 2D and 3D (OpenGL) previews.

## Features

- **Terrain**: heightmap generation (OpenSimplex noise), erosion feedback
- **Geology**: rock types, hardness, tectonic deformation
- **Weather**: coupled 3-layer atmosphere simulation (wind, temperature,
  humidity, precipitation) with orographic effects
- **Water**: rivers, lakes, flow networks, erosion/sedimentation
- **Biome**: climate-based biome classification
- **Settlement**: settlement/road placement and a physics-based plot/parcel
  simulation (city blocks, wilderness boundaries, traffic-weighted roads)
- Staged preview generation: fast low-resolution passes (128 px) that refine
  automatically up to full resolution (1024 px) as generation continues,
  rather than blocking on the final result
- 2D (matplotlib) and 3D (OpenGL) map views with per-layer overlays
- A standalone "Plot Physics Lab" tool (`tools/biome_lab/`) for iterating on
  the settlement/plot-physics simulation outside the main app

## Requirements

- Python 3.13 (developed/tested on Windows; no Windows-only APIs are used,
  but other platforms are untested)
- See [requirements.txt](requirements.txt) for pinned package versions
  (PyQt6, numpy, scipy, matplotlib, opensimplex, PyOpenGL, shapely,
  scikit-image, pillow, psutil, colorlog)

## Getting Started

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

On launch, the app opens a main menu, then the Map Editor with one tab per
generation stage (Terrain, Geology, Weather, Water, Biome, Settlement,
Overview).

## Project Structure

```
core/           Generator logic for each stage (terrain, geology, weather,
                water, biome, settlement) - no Qt dependencies
gui/            PyQt6 application: main window, tabs, widgets, managers
tools/          Standalone dev tools (e.g. the Plot Physics Lab)
docs/           Design notes, pipeline dependency docs, backlog
```

## Testing

There is no automated test suite yet. Core generator logic is validated via
throwaway headless smoke-test scripts run through the project's virtual
environment (see `CLAUDE.md` for the established pattern); GUI/OpenGL
rendering changes are verified by running the app directly.

## License

No license has been chosen yet for this project.
