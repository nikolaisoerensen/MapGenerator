"""
Path: tools/biome_lab/app.py

Haupteinstiegspunkt: die PlotPhysicsLab-QMainWindow-Klasse und main().

Interaktives Testprogramm fuer die Plot-Physik: Plotkerne (rote Punkte,
best-candidate-verteilt ueber die civ_map), civ-gewichtete Nachbar-Abstossung,
rang-distanz-gewichtete "Gravitation" zu den Staedten, ein separates
Wege-Netz (Splines zwischen Plotnodes = Voronoi-Kreuzungen der Plotkerne,
kontinuierliche YlOrRd-Farbkarte fuer Traffic, fliessende Linienstaerke).

Wildnis-Grenze: Voronoi-Vertices, die nahe an der Wildnisgrenze liegen,
werden auf die Grenzkontur projiziert und bleiben dort "kleben".

Laeuft komplett standalone ohne DataLODManager/GenerationOrchestrator/
CalculatorDispatcher - nur eine einfache Heightmap (reine Terrain-
Rauschgenerierung, Default-Parameter aus value_default.py), 3 Settlements,
civ_map und ein grobes Landschafts-Voronoi als Hintergrund.

Die Klasse selbst ist bewusst als Zusammensetzung von Mixins organisiert
(ein "God-Object" QMainWindow laesst sich nicht sauber auf mehrere Dateien
aufteilen, da PyQt genau eine Widget-Klasse erwartet) -- jedes Mixin deckt
einen fachlichen Bereich ab:
  - scene.SceneMixin     : Terrain/Settlement/Civ-Map-Aufbau ("Anfang")
  - topology.TopologyMixin: Plotkern-Sampling, Voronoi-Netz, Traffic-Graph
  - field.FieldMixin     : Potentialfeld-Berechnung + Sampling
  - physics.PhysicsMixin : Federkraefte, Bewegungsintegration, Tick-Loop
  - traffic.TrafficMixin : Traffic-Zuweisung ueber den Distanzgraphen
  - draw.DrawMixin       : Matplotlib-Rendering + Canvas-Klicks
  - ui.UIMixin           : Qt-Bedienpanel + normierte Slider-Properties

Start: .venv/Scripts/python.exe tools/plot_physics_lab.py
"""
import sys
import time

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

import matplotlib
matplotlib.use("Qt5Agg")

from .logging_setup import logger
from .scene import SceneMixin
from .topology import TopologyMixin
from .field import FieldMixin
from .physics import PhysicsMixin
from .traffic import TrafficMixin
from .draw import DrawMixin
from .ui import UIMixin


class PlotPhysicsLab(QMainWindow, SceneMixin, TopologyMixin, FieldMixin, PhysicsMixin,
                      TrafficMixin, DrawMixin, UIMixin):
    # ------------------------------------------------------------ Konstanten --
    TRAFFIC_RECOMPUTE_INTERVAL = 3  # Traffic-Simulation laeuft nur jeden n-ten Tick
    WILDERNESS_CIV_THRESHOLD = 0.20
    WILDERNESS_MIN_AREA = 50
    SOFTENING = 5.0
    PHYSICS_TIME_STEP = 0.25
    MAX_SPRING_RESULTANT = 4.0
    MAX_DISPLACEMENT_PER_TICK = 2.0
    MAP_EDGE_INSET = 5.0
    PLOT_NODE_FOLLOW_FACTOR = 0.35  # wie stark plot_nodes der Plotkern-Wanderung folgen ("aufrutschen")

    # ------------------------------------------------------- Profiling --
    def _start_timer(self, name):
        if not hasattr(self, '_timers'):
            self._timers = {}
        self._timers[name] = time.perf_counter()

    def _stop_timer(self, name):
        elapsed = (time.perf_counter() - self._timers[name]) * 1000
        if not hasattr(self, '_tick_timings'):
            self._tick_timings = {}
        self._tick_timings[name] = elapsed

    def _perf_text(self):
        if not hasattr(self, '_tick_timings') or not self._tick_timings:
            return "Total: 0.0 ms"
        total = sum(self._tick_timings.values())
        sorted_items = sorted(self._tick_timings.items(), key=lambda kv: -kv[1])
        lines = [f"Total: {total:.1f} ms"]
        for name, t in sorted_items[:3]:
            pct = (t / total * 100) if total > 0 else 0
            lines.append(f"  {name}: {t:.1f} ms ({pct:.0f}%)")
        return " | ".join(lines)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot-Node Physics Lab (Potentialfeld)")
        self.resize(1500, 950)

        self.map_size = 256
        self.iteration = 0
        self.playing = True
        self.next_node_id = 0
        self.boundary_owner = {}
        self.ridge_traffic_history = {}
        self.current_ridge_edges = []
        self.ridge_vertex_positions = None
        self.potential_field = None

        self.settlement_border_margin = 25.0

        # -- Basiswerte mit physikalischer Einheit (px), bleiben unnormiert --
        self.plot_base_spacing = 60.0
        self.wilderness_core_spacing = 80.0

        # -- Logarithmische Regler: 1.0 = neutral (kein Effekt auf Basiswert),
        #    5.0 = 5x staerker, 0.2 = 5x schwaecher, usw. (Basis 5 statt Basis 10) --
        self.norm_civ_spacing_factor = 1.0
        # Default bewusst > 1: ohne staerkere Zentroid-Feder wandern Plotkerne
        # sichtbar aus ihrem Grundstueck heraus, bevor die harte Zell-Grenze
        # (physics._contain_core_in_cell) sie zurueckprojiziert -- mit hoeherer
        # Steifigkeit bleiben sie meist von selbst nahe der Zellmitte.
        self.norm_spacing_spring_stiffness = 3.0
        self.norm_traffic_spring_stiffness = 1.0
        self.norm_gravity_strength = 1.0
        self.norm_city_repulsion_strength = 1.0
        self.norm_height_cost_factor = 1.0
        self.norm_tier_factor = 1.0
        self.norm_potential_strength = 1.0

        # -- Basiswerte, auf die die Normierung bei Multiplikator 1.0 abbildet --
        self._BASE_CIV_SPACING_FACTOR = 8.0
        self._BASE_SPACING_SPRING_STIFFNESS = 0.04
        self._BASE_TRAFFIC_SPRING_STIFFNESS = 0.02
        self._BASE_GRAVITY_STRENGTH = 0.01
        self._BASE_CITY_REPULSION_STRENGTH = 0.5
        self._BASE_HEIGHT_COST_FACTOR = 3.0
        self._BASE_TIER_FACTOR = 1.0

        self.spring_traffic_shrink = 0.002
        self.spring_min_shrink_fraction = 0.4

        self.show_civ_overlay = False
        self.show_potential_overlay = False

        self.plot_nodes_count = 200
        self.civ_influence_decay = 0.8
        self.city_size = 0.5
        self.city_reach_factor = 4.0
        self.civ_influence_range = 0.30
        self.plot_intercity_traffic = 30

        self.spline_wiggle_pct = 20.0
        self.spline_detail = 2.0

        self._wilderness_boundary_polygon = np.empty((0, 2), dtype=np.float64)
        self._wilderness_polygons = []
        self._selected_node_id = None
        self._selected_core_id = None
        self.wilderness_node_ids = set()
        self.map_border_node_ids = set()

        self._build_ui()
        self._regenerate_scene()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(150)


def _global_exception_hook(exc_type, exc_value, exc_traceback):
    logger.critical(
        "UNBEHANDELTE EXCEPTION: %s: %s",
        exc_type.__name__, exc_value,
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _global_exception_hook


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = PlotPhysicsLab()
    window.show()
    exit_code = app.exec_()
    proc = getattr(window, "_voronoi_proc", None)
    conn = getattr(window, "_voronoi_conn", None)
    if proc is not None and proc.is_alive():
        try:
            conn.send(None)
            proc.join(timeout=1.0)
        except (BrokenPipeError, OSError):
            pass
        if proc.is_alive():
            proc.terminate()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
