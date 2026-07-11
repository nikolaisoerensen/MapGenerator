"""
Path: tools/plot_physics_lab.py

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

Start: .venv/Scripts/python.exe tools/plot_physics_lab.py
"""
import sys
import os
import random as random_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import label, gaussian_filter, grey_closing, grey_opening, distance_transform_edt
from skimage import measure
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QCheckBox, QGroupBox, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

import time

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection

from core.terrain_generator import BaseTerrainGenerator
from core.settlement_generator import (
    SettlementGenerator, TerrainSuitabilityAnalyzer, CityBoundaryAnalyzer,
    PlotNodeSystem, PlotNode,
)
from gui.config.value_default import TERRAIN

import logging
import multiprocessing as mp

LOG_DIR = os.path.join(REPO_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("plot_physics_lab")
logger.setLevel(logging.DEBUG)
_file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "plot_physics_lab.log"), encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(_file_handler)


class _FlushingFileHandler(logging.FileHandler):
    """Erzwingt sofortiges Flush + fsync, damit bei einem nativen Crash
    (Qhull STATUS_STACK_BUFFER_OVERRUN) keine Log-Zeilen im Puffer verloren
    gehen."""
    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            os.fsync(self.stream.fileno())
        except (OSError, ValueError):
            pass


logger.removeHandler(_file_handler)
_file_handler = _FlushingFileHandler(
    os.path.join(LOG_DIR, "plot_physics_lab.log"), encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(_file_handler)


def generate_default_heightmap(map_size, seed):
    """Nur der einfache Terrain-Teil: Multi-Octave-Rauschen + Redistribution,
    exakt die TERRAIN-Defaults aus value_default.py, ohne Geology/Water/etc."""
    terrain_gen = BaseTerrainGenerator(map_seed=seed)
    adjusted_frequency = TERRAIN.FREQUENCY["default"] * (64 / map_size)
    noise = terrain_gen.noise_generator.generate_noise_grid(
        size=map_size, frequency=adjusted_frequency, octaves=TERRAIN.OCTAVES["default"],
        persistence=TERRAIN.PERSISTENCE["default"], lacunarity=TERRAIN.LACUNARITY["default"])
    amplitude = TERRAIN.AMPLITUDE["default"]
    heightmap = ((noise + 1.0) * 0.5 * amplitude).astype(np.float32)
    heightmap = terrain_gen._apply_redistribution(heightmap, TERRAIN.REDISTRIBUTE_POWER["default"], amplitude)
    return heightmap


def _voronoi_worker_loop(conn):
    """Laeuft dauerhaft in einem separaten Prozess (einmalig gestartet).
    Wartet in einer Schleife auf Punktmengen und sendet Ergebnisse zurueck.
    Ein nativer Qhull-Crash (Windows STATUS_STACK_BUFFER_OVERRUN) beendet
    nur diesen einen Prozess -- die Haupt-App bleibt am Leben und kann
    per _ensure_voronoi_worker() einen neuen Worker nachstarten."""
    from scipy.spatial import Voronoi
    while True:
        try:
            points = conn.recv()
        except (EOFError, OSError):
            break
        if points is None:  # Shutdown-Signal
            break
        try:
            vor = Voronoi(points)
            conn.send({
                "vertices": vor.vertices,
                "ridge_points": vor.ridge_points,
                "ridge_vertices": vor.ridge_vertices,
                "point_region": vor.point_region,
                "regions": vor.regions,
            })
        except Exception as e:
            conn.send({"error": str(e)})


class PlotPhysicsLab(QMainWindow):
    # ------------------------------------------------------------ Konstanten --
    VORONOI_RECOMPUTE_INTERVAL = 3
    WILDERNESS_CIV_THRESHOLD = 0.20
    WILDERNESS_MIN_AREA = 50
    WILDERNESS_TRAFFIC_WEIGHT = 1.5
    SOFTENING = 5.0
    WILDERNESS_BOUNDARY_CLAMP_DIST = 2.5  # Max Abstand zum Projektionspunkt auf Grenzkontur
    PHYSICS_TIME_STEP = 0.25
    MAX_SPRING_RESULTANT = 4.0
    MAX_DISPLACEMENT_PER_TICK = 2.0
    MAP_EDGE_INSET = 5.0

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

        self._voronoi_cache_tick = 0
        self._cached_ridge_edges = []
        self._cached_vertex_positions = None
        self._cached_predecessors = None
        self._cached_distances = None
        self._cached_boundary_entries = []
        self._cached_boundary_settlement = []
        self._cached_node_entry = {}
        self._cached_num_vertices = 0
        self._cached_core_springs = []

        self.settlement_border_margin = 25.0
        self.settlement_min_distance = 35.0

        # -- Basiswerte mit physikalischer Einheit (px), bleiben unnormiert --
        self.plot_base_spacing = 60.0

        # -- Logarithmische Regler: 1.0 = neutral (kein Effekt auf Basiswert),
        #    5.0 = 5x staerker, 0.2 = 5x schwaecher, usw. (Basis 5 statt Basis 10) --
        self.norm_civ_spacing_factor = 1.0
        self.norm_spacing_spring_stiffness = 1.0
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

        self.plotnodes_count = 200
        self.civ_influence_decay = 0.8
        self.city_size = 0.5
        self.city_reach_factor = 4.0
        self.civ_influence_range = 0.30
        self.plot_intercity_traffic = 30

        self.spline_wiggle_pct = 20.0
        self.spline_detail = 2.0

        self._wilderness_boundary_polygon = np.empty((0, 2), dtype=np.float64)
        self._wilderness_polygons = []

        self._build_ui()
        self._regenerate_scene()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(150)


    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=3)

        panel = QWidget()
        panel.setFixedWidth(450)
        panel_layout = QVBoxLayout(panel)
        layout.addWidget(panel, stretch=0)

        self.status_label = QLabel("Iteration: 0")
        panel_layout.addWidget(self.status_label)

        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self._toggle_play)
        panel_layout.addWidget(self.play_button)

        regen_button = QPushButton("Regenerate Terrain + Settlements")
        regen_button.clicked.connect(self._regenerate_scene)
        panel_layout.addWidget(regen_button)

        reset_button = QPushButton("Reset Plot Nodes (übernimmt Hintergrund-Slider)")
        reset_button.clicked.connect(self._reset_plot_nodes)
        panel_layout.addWidget(reset_button)

        live_group = QGroupBox("Live-Physik (wirkt sofort, jeden Tick)")
        live_form = QFormLayout()
        self._add_slider(live_form, "plot_base_spacing", "Base Spacing (px)", 2, 50, self.plot_base_spacing, scale=10,
                         live=True)
        self._add_log_slider(live_form, "norm_civ_spacing_factor", "Civ Spacing Factor (0.04x-25x)",
                             default=self.norm_civ_spacing_factor)
        self._add_log_slider(live_form, "norm_spacing_spring_stiffness", "Feder: Abstand (0.04x-25x)",
                             default=self.norm_spacing_spring_stiffness)
        self._add_log_slider(live_form, "norm_traffic_spring_stiffness", "Feder: Traffic-Zug (0.04x-25x)",
                             default=self.norm_traffic_spring_stiffness)
        self._add_log_slider(live_form, "norm_gravity_strength", "Gravity Strength (0.04x-25x)",
                             default=self.norm_gravity_strength)
        self._add_log_slider(live_form, "norm_city_repulsion_strength", "Stadtmauer-Gegenkraft (0.04x-25x)",
                             default=self.norm_city_repulsion_strength)
        self._add_log_slider(live_form, "norm_height_cost_factor", "Height Cost Factor (0.04x-25x)",
                             default=self.norm_height_cost_factor)
        self._add_log_slider(live_form, "norm_tier_factor", "Pfad/Weg/Straße Faktor (0.04x-25x)",
                             default=self.norm_tier_factor)
        self._add_log_slider(live_form, "norm_potential_strength", "Potential-Stärke gesamt (0.04x-25x)",
                             default=self.norm_potential_strength)
        live_group.setLayout(live_form)
        panel_layout.addWidget(live_group)

        self.civ_overlay_cb = QCheckBox("Civ-Wert als Heatmap überlagern")
        self.civ_overlay_cb.setChecked(self.show_civ_overlay)
        self.civ_overlay_cb.toggled.connect(self._set_civ_overlay)
        panel_layout.addWidget(self.civ_overlay_cb)

        self.potential_overlay_cb = QCheckBox("Potentialfeld überlagern")
        self.potential_overlay_cb.setChecked(self.show_potential_overlay)
        self.potential_overlay_cb.toggled.connect(self._set_potential_overlay)
        panel_layout.addWidget(self.potential_overlay_cb)

        way_group = QGroupBox("Wege-Darstellung (Splines)")
        way_form = QFormLayout()
        self._add_slider(way_form, "spline_wiggle_pct", "Kurvigkeit % (0=gerade, 70=verschlungen)",
                         0, 70, self.spline_wiggle_pct, scale=1)
        self._add_slider(way_form, "spline_detail", "Detailgrad (Wellenfrequenz)",
                         1, 6, self.spline_detail, scale=10)
        way_group.setLayout(way_form)
        panel_layout.addWidget(way_group)

        bg_group = QGroupBox("Hintergrund (Klick auf 'Reset Plot Nodes' nötig)")
        bg_form = QFormLayout()
        self._add_slider(bg_form, "plotnodes_count", "Plot Nodes (Ziel)", 20, 800, self.plotnodes_count, scale=1)
        self._add_slider(bg_form, "city_size", "Stadtgröße", 0, 1.0, self.city_size, scale=100)
        self._add_slider(bg_form, "civ_influence_decay", "Civ Influence Decay", 0.1, 2, self.civ_influence_decay,
                         scale=100)
        bg_group.setLayout(bg_form)
        panel_layout.addWidget(bg_group)

        legend = QLabel(
            "Wege-Linien: YlOrRd-Farbkarte (gelb-orange-rot), Breite 0.3-3.0 proportional zum Traffic\n"
            "Hellgrau = Traffic unter Schwelle (<20×Faktor)\n"
            "Gold-Kontur = Stadtgrenze (Civ-Wert-Kontour)\n"
            "Sterne = Settlement/Marktplatz\n"
            "Rote Punkte = Plotkern (best-candidate verteilt, erzeugt konstanten Traffic 3-5)\n"
            "Blaue Punkte = Plotnodes (Voronoi-Kreuzungen, an Wildnisgrenze projiziert)\n"
            "Dünne graue Linien = Grundstücksgrenzen (immer gerade)\n"
            "Alle Kräfte wirken über ein vorberechnetes Potentialfeld")
        legend.setWordWrap(True)
        panel_layout.addWidget(legend)
        panel_layout.addStretch(1)

    def _add_log_slider(self, form, attr_name, label, default=1.0, min_exp=-2.0, max_exp=2.0, base=5.0):
        """Logarithmischer Regler: die Slider-Position ist ein Exponent zur 'base'.
        default=1.0 liegt bei Exponent 0 (neutral, kein Effekt auf den Basiswert).
        Bei base=5, max_exp=2 ergibt sich ein Maximalfaktor von 25x, bei min_exp=-2
        ein Minimalfaktor von 0.04x. Damit lassen sich Kraefte sowohl fast
        abschalten als auch stark verstaerken, ohne die Slider-Mitte zu verzerren."""
        RESOLUTION = 100  # Slider-Schritte pro Exponent-Einheit, fuer feine Aufloesung

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(round(min_exp * RESOLUTION)))
        slider.setMaximum(int(round(max_exp * RESOLUTION)))
        default_exp = float(np.log(default) / np.log(base)) if default > 0 else 0.0
        slider.setValue(int(round(default_exp * RESOLUTION)))

        value_label = QLabel(f"{default:.2f}x")
        value_label.setFixedWidth(50)

        def on_change(v):
            exponent = v / RESOLUTION
            real_value = base ** exponent
            value_label.setText(f"{real_value:.2f}x")
            setattr(self, attr_name, real_value)
            if hasattr(self, 'potential_field') and self.potential_field is not None:
                self._compute_potential_field()

        slider.valueChanged.connect(on_change)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        form.addRow(label, row)

    def _add_slider(self, form, attr_name, label, min_val, max_val, default, scale, live=False):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(round(min_val * scale)))
        slider.setMaximum(int(round(max_val * scale)))
        slider.setValue(int(round(default * scale)))

        value_label = QLabel(f"{default:.3g}")
        value_label.setFixedWidth(50)

        def on_change(v):
            real_value = v / scale
            value_label.setText(f"{real_value:.3g}")
            setattr(self, attr_name, real_value)
            if live and hasattr(self, 'potential_field') and self.potential_field is not None:
                self._compute_potential_field()

        slider.valueChanged.connect(on_change)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        form.addRow(label, row)

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_button.setText("Pause" if self.playing else "Play")

    def _set_civ_overlay(self, checked):
        self.show_civ_overlay = checked
        self._redraw()

    def _set_potential_overlay(self, checked):
        self.show_potential_overlay = checked
        self._redraw()

    def _n(self, norm_attr_name, base_value):
        """norm_attr_name enthaelt bereits den fertigen Multiplikator (Log-Slider-
        Ausgabe), daher reine Multiplikation ohne weitere Skalierung."""
        return base_value * getattr(self, norm_attr_name)

    @property
    def plot_civ_spacing_factor(self):
        return self._n("norm_civ_spacing_factor", self._BASE_CIV_SPACING_FACTOR)

    @property
    def spacing_spring_stiffness(self):
        return self._n("norm_spacing_spring_stiffness", self._BASE_SPACING_SPRING_STIFFNESS)

    @property
    def traffic_spring_stiffness(self):
        return self._n("norm_traffic_spring_stiffness", self._BASE_TRAFFIC_SPRING_STIFFNESS)

    @property
    def plot_gravity_strength(self):
        return self._n("norm_gravity_strength", self._BASE_GRAVITY_STRENGTH)

    @property
    def plot_city_repulsion_strength(self):
        return self._n("norm_city_repulsion_strength", self._BASE_CITY_REPULSION_STRENGTH)

    @property
    def plot_height_cost_factor(self):
        return self._n("norm_height_cost_factor", self._BASE_HEIGHT_COST_FACTOR)

    @property
    def plot_tier_factor(self):
        return self._n("norm_tier_factor", self._BASE_TIER_FACTOR)

    def _calculate_contour_levels(self, heightmap, num_levels=10):
        """Berechnet gleichmässig verteilte Hoehenlinien-Stufen zwischen Min und Max,
        ignoriert Bereiche unterhalb 0 (Wasser), damit die Linien nur das Landrelief zeigen."""
        hm_min = float(np.percentile(heightmap, 2))
        hm_max = float(np.max(heightmap))
        if hm_max - hm_min < 1e-6:
            return [hm_min]
        return list(np.linspace(max(hm_min, 0.0), hm_max, num_levels))

    def _draw_contour_lines(self, heightmap):
        """
        Funktionsweise: Zeichnet Hoehenlinien ueber das aktuell aktive Axes.
        Aufgabe: Reine Relief-Darstellung als Ersatz/Ergaenzung zur Terrain-Heatmap,
        damit das Potentialfeld-Overlay ungestoert sichtbar bleibt.
        Parameter: heightmap (numpy.ndarray) - Hoehendaten fuer die Konturlinien
        """
        contour_levels = self._calculate_contour_levels(heightmap)
        contours = self.ax.contour(
            heightmap, levels=contour_levels, colors="dimgray",
            linewidths=0.5, alpha=0.6,
            extent=(0, self.map_size, 0, self.map_size), origin="lower")
        return contours

    # -------------------------------------------------------- Szenen-Setup --
    def _regenerate_scene(self):
        seed = random_mod.randint(1, 999_999)
        self.heightmap = generate_default_heightmap(self.map_size, seed)
        dz_dy, dz_dx = np.gradient(self.heightmap)
        self.slopemap = np.stack([dz_dx, dz_dy], axis=-1).astype(np.float32)
        self.water_map = (self.heightmap < np.percentile(self.heightmap, 8)).astype(np.float32)

        suitability_analyzer = TerrainSuitabilityAnalyzer(terrain_factor_villages=1.0, map_size=self.map_size)
        suitability_map = suitability_analyzer.create_combined_suitability(
            self.heightmap, self.slopemap, self.water_map)

        margin = int(self.settlement_border_margin)
        if margin > 0:
            suitability_map[:margin, :] = 0.0
            suitability_map[-margin:, :] = 0.0
            suitability_map[:, :margin] = 0.0
            suitability_map[:, -margin:] = 0.0

        self.gen = SettlementGenerator(map_seed=seed, shader_manager=None)
        self.gen.settlements = 3
        self.settlements = self.gen.calculate_settlements(suitability_map, self.heightmap, self.map_size)

        self._recompute_background()
        self._reset_plot_nodes()
        print(self.heightmap.min(), self.heightmap.max())

    def _recompute_background(self):
        self.city_reach_factor = 4.0 + self.city_size * 6.0
        self.civ_influence_range = 0.15 + self.city_size * 0.30
        self.plot_intercity_traffic = 20.0 + self.city_size * 40.0

        self.gen.civ_influence_range = self.civ_influence_range
        self.gen.civ_influence_decay = self.civ_influence_decay
        self.gen.city_reach_factor = self.city_reach_factor
        self.gen.terrain_factor_villages = 1.0

        self.civ_map = self.gen.calculate_civilization_mapping(
            self.heightmap, self.slopemap, self.settlements, [], [])

        self.civ_map = gaussian_filter(self.civ_map, sigma=2.5)
        self.civ_map = grey_closing(self.civ_map, size=(7, 7))
        self.civ_map = grey_opening(self.civ_map, size=(5, 5))

        self._remove_civ_islands()  # NEU: verhindert stadtlose civ-Inseln in der Wildnis

        boundary_analyzer = CityBoundaryAnalyzer(
            self.gen.terrain_factor_villages, self.city_reach_factor, shader_manager=None)
        self.city_mask, _ = boundary_analyzer.compute_city_boundaries(
            self.heightmap, self.slopemap, self.settlements)

        # ---- Dynamische Wildnis-Grenzpunkte statt fester Knoten ----
        self._wilderness_boundary_polygon = self._build_wilderness_boundary_points()

        self._compute_potential_field()

    def _remove_civ_islands(self):
        """Entfernt civ-Flaechen, die keine Stadt enthalten (Inseln in der
        Wildnis). Jede verbleibende zusammenhaengende civ-Komponente ist damit
        garantiert mit mindestens einer Stadt verbunden. Getrennte Staedte auf
        verschiedenen Kontinenten bleiben erlaubt, solange jede fuer sich
        mindestens eine Stadt einschliesst."""
        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        labeled, num_features = label(civ_mask)
        if num_features == 0:
            return

        valid_labels = set()
        h, w = labeled.shape
        for settlement in self.settlements:
            sx, sy = int(round(settlement.x)), int(round(settlement.y))
            sx = np.clip(sx, 0, w - 1)
            sy = np.clip(sy, 0, h - 1)
            lbl = labeled[sy, sx]
            if lbl > 0:
                valid_labels.add(int(lbl))
            else:
                y0, y1 = max(0, sy - 5), min(h, sy + 6)
                x0, x1 = max(0, sx - 5), min(w, sx + 6)
                nearby = labeled[y0:y1, x0:x1]
                nearby_labels = nearby[nearby > 0]
                if len(nearby_labels):
                    valid_labels.add(int(nearby_labels[0]))

        island_mask = np.isin(labeled, list(valid_labels), invert=True) & (labeled > 0)
        if np.any(island_mask):
            self.civ_map[island_mask] = self.WILDERNESS_CIV_THRESHOLD * 0.5

    def _build_wilderness_boundary_points(self):
        """Baut ein echtes Polygon der Zivilisationsflaeche via Marching-Squares."""
        mask = (self.civ_map >= self.WILDERNESS_CIV_THRESHOLD).astype(np.float32)
        contours = measure.find_contours(mask, level=0.5)
        polygons = []
        for c in contours:
            pts = np.column_stack([c[:, 1], c[:, 0]])  # (row,col) -> (x,y)
            if len(pts) >= 4:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > self.WILDERNESS_MIN_AREA:
                    polygons.append(poly)
        self._wilderness_polygons = polygons
        if polygons:
            return np.vstack([np.array(p.exterior.coords) for p in polygons])
        return np.empty((0, 2), dtype=np.float64)

    def _add_log_slider(self, form, attr_name, label, default=1.0, min_exp=-2.0, max_exp=2.0, base=5.0):
        """Logarithmischer Regler: Slider-Position ist ein Exponent zur 'base'.
        default=1.0 liegt bei Exponent 0 (Mitte), max_exp gibt z.B. bei base=5, max_exp=2
        einen Maximalfaktor von 25x, min_exp=-2 einen Minimalfaktor von 0.04x."""
        RESOLUTION = 100  # Slider-Schritte pro Exponent-Einheit, fuer feine Aufloesung

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(round(min_exp * RESOLUTION)))
        slider.setMaximum(int(round(max_exp * RESOLUTION)))
        default_exp = np.log(default) / np.log(base)
        slider.setValue(int(round(default_exp * RESOLUTION)))

        value_label = QLabel(f"{default:.2f}x")
        value_label.setFixedWidth(50)

        def on_change(v):
            exponent = v / RESOLUTION
            real_value = base ** exponent
            value_label.setText(f"{real_value:.2f}x")
            setattr(self, attr_name, real_value)
            if hasattr(self, 'potential_field') and self.potential_field is not None:
                self._compute_potential_field()

        slider.valueChanged.connect(on_change)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_label)
        form.addRow(label, row)

    def _draw_potential_quiver(self):
        if self.potential_field is None:
            return
        step = 5
        h, w = self.potential_field.shape[:2]
        yy, xx = np.mgrid[0:h:step, 0:w:step]
        fx = self.potential_field[0:h:step, 0:w:step, 0]
        fy = self.potential_field[0:h:step, 0:w:step, 1]
        magnitude = np.sqrt(fx ** 2 + fy ** 2)

        ARROW_LENGTH_SCALE = 6.0  # kleiner Wert = laengere Pfeile, groesser = kuerzere Pfeile
        ARROW_WIDTH = 0.0025  # Schaftdicke

        self.ax.quiver(
            xx, yy, fx, fy, magnitude,
            cmap="viridis", scale=ARROW_LENGTH_SCALE, scale_units="inches",
            width=ARROW_WIDTH, headwidth=3.0, headlength=5.0, headaxislength=4.5,
            pivot="tail", alpha=0.95, zorder=3, clim=(0.0, 1.2))

    # ------------------------------------------------------ Potentialfeld --
    def _compute_potential_field(self):
        h, w = self.civ_map.shape
        field = np.zeros((h, w, 2), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]

        # 1) Civ-Gradient: Saettigung statt harter Normalisierung, damit in
        #    flachen Bereichen kein Rauschen auf volle Staerke aufgeblasen wird.
        gy, gx = np.gradient(self.civ_map)
        grad_norm = np.sqrt(gx ** 2 + gy ** 2)
        GRAD_SATURATION = 0.05
        grad_strength = 1.0 - np.exp(-grad_norm / GRAD_SATURATION)
        mask = grad_norm > 1e-10
        gx_dir = np.zeros_like(gx)
        gy_dir = np.zeros_like(gy)
        gx_dir[mask] = gx[mask] / grad_norm[mask]
        gy_dir[mask] = gy[mask] / grad_norm[mask]
        civ_strength = np.clip(1.0 - self.civ_map, 0.0, 1.0) * 0.5
        field[:, :, 0] += gx_dir * grad_strength * civ_strength
        field[:, :, 1] += gy_dir * grad_strength * civ_strength

        # 2) Gravitation zur naechsten Stadt -- langsamerer Abfall ueber Distanz,
        # damit auch tief in der Wildnis noch ein spuerbarer Zug zur Stadt besteht.
        if self.settlements:
            eps = self.SOFTENING
            for settlement in self.settlements:
                sx, sy = settlement.x, settlement.y
                dx = xx - sx
                dy = yy - sy
                dist = np.maximum(np.hypot(dx, dy), eps)
                weight = 50.0 / np.sqrt(dist)
                field[:, :, 0] -= (dx / dist) * weight * self.plot_gravity_strength
                field[:, :, 1] -= (dy / dist) * weight * self.plot_gravity_strength

        # 3) Stadtmauer-Abstossung, an der Quelle gekappt (Explosion verhindern)
        PUSH_CAP = 3.0
        wall_spacing = 5.0
        for settlement in self.settlements:
            sx, sy = settlement.x, settlement.y
            dx = xx - sx
            dy = yy - sy
            dist = np.maximum(np.hypot(dx, dy), 1e-6)
            ratio = np.minimum(wall_spacing / dist, 6.0)
            push = np.minimum(self.plot_city_repulsion_strength * ratio ** 3, PUSH_CAP)
            field[:, :, 0] += (dx / dist) * push
            field[:, :, 1] += (dy / dist) * push

        # 4) Wildnis-Repulsion ueber signed distance field: wirkt auch INNERHALB
        #    der Wildnis Richtung Zivilisation, nicht nur an der Grenze.
        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        dist_out = distance_transform_edt(~civ_mask)
        dist_in = distance_transform_edt(civ_mask)
        signed_dist = np.where(civ_mask, dist_in, -dist_out)
        gy_s, gx_s = np.gradient(gaussian_filter(signed_dist, sigma=3.0))
        norm_s = np.sqrt(gx_s ** 2 + gy_s ** 2)
        mask_s = norm_s > 1e-10
        wgx = np.zeros_like(gx_s)
        wgy = np.zeros_like(gy_s)
        wgx[mask_s] = gx_s[mask_s] / norm_s[mask_s]
        wgy[mask_s] = gy_s[mask_s] / norm_s[mask_s]
        wild_scale = 25.0
        push_strength = np.where(
            signed_dist < 0,
            1.0 - np.exp(-np.abs(signed_dist) / wild_scale),
            np.exp(-np.maximum(signed_dist, 0) / wild_scale))
        field[:, :, 0] += wgx * push_strength * 0.6
        field[:, :, 1] += wgy * push_strength * 0.6

        # 5) Kartenrand-Abstossung
        BORDER_MARGIN = 25.0
        BORDER_STRENGTH = 0.4

        def _edge_push(dist_to_edge):
            t = np.clip(1.0 - dist_to_edge / BORDER_MARGIN, 0.0, 1.0)
            return t ** 2 * BORDER_STRENGTH

        field[:, :, 0] += _edge_push(xx.astype(float)) - _edge_push((w - 1 - xx).astype(float))
        field[:, :, 1] += _edge_push(yy.astype(float)) - _edge_push((h - 1 - yy).astype(float))

        # 6) Globaler Multiplikator ueber den neuen Log-Regler
        field *= self.norm_potential_strength

        self.potential_field = field

    def _sample_field(self, x, y):
        if self.potential_field is None:
            return np.zeros(2)
        h, w = self.potential_field.shape[:2]
        x = np.clip(x, 0.5, w - 1.5)
        y = np.clip(y, 0.5, h - 1.5)
        ix, iy = int(x), int(y)
        fx, fy = x - ix, y - iy
        dx, dy = 1 if ix + 1 < w else 0, 1 if iy + 1 < h else 0
        return (
            self.potential_field[iy, ix] * (1 - fx) * (1 - fy) +
            self.potential_field[iy, ix + dx] * fx * (1 - fy) +
            self.potential_field[iy + dy, ix] * (1 - fx) * fy +
            self.potential_field[iy + dy, ix + dx] * fx * fy
        )

    def _civ_at(self, pos):
        x, y = int(pos[0]), int(pos[1])
        h, w = self.civ_map.shape
        if 0 <= y < h and 0 <= x < w:
            return float(self.civ_map[y, x])
        return 0.0

    def _civ_at_continuous(self, pos):
        """Civ-Wert ohne Pixel-Spruenge bilinear abfragen.

        Die diskrete Abfrage in ``_civ_at`` ist fuer Zuordnungen ausreichend,
        erzeugt an einer harten Kollisionsgrenze aber ein treppenfoermiges
        Verhalten. Diese Variante wird deshalb fuer die Randkollision benutzt.
        """
        h, w = self.civ_map.shape
        x = float(np.clip(pos[0], 0.0, w - 1.0))
        y = float(np.clip(pos[1], 0.0, h - 1.0))
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        tx, ty = x - x0, y - y0
        return float(
            self.civ_map[y0, x0] * (1.0 - tx) * (1.0 - ty) +
            self.civ_map[y0, x1] * tx * (1.0 - ty) +
            self.civ_map[y1, x0] * (1.0 - tx) * ty +
            self.civ_map[y1, x1] * tx * ty
        )

    # ------------------------------------------------------- Plotkern-Setup --
    def _best_candidate_sample(self, valid_mask, count, base_spacing, civ_spacing_factor, civ_map, k=15):
        ys, xs = np.nonzero(valid_mask)
        if len(xs) == 0:
            return []
        count = min(count, len(xs))
        chosen = []
        chosen_arr = np.empty((0, 2))
        pool_size = len(xs)
        for _ in range(count):
            idxs = np.random.randint(0, pool_size, size=min(k, pool_size))
            best_pos, best_score = None, float("-inf")
            for idx in idxs:
                x, y = float(xs[idx]), float(ys[idx])
                civ_value = float(civ_map[int(y), int(x)])
                target_spacing = max(base_spacing / (1.0 + civ_spacing_factor * civ_value), 2.5)
                print(f"spacing={target_spacing:.4f}", flush=True)
                if len(chosen_arr):
                    dist = float(np.sqrt(np.min((chosen_arr[:, 0] - x) ** 2 + (chosen_arr[:, 1] - y) ** 2)))
                else:
                    dist = target_spacing * 10.0
                score = dist - target_spacing
                if score > best_score:
                    best_score, best_pos = score, (x, y)
            if best_pos is None:
                continue
            chosen.append(best_pos)
            chosen_arr = np.array(chosen)
        return chosen

    def _generate_city_boundary_nodes(self):
        city_mask = self.city_mask
        inside = city_mask >= 0
        boundary = inside & (
            ~np.roll(inside, 1, axis=0) | ~np.roll(inside, -1, axis=0) |
            ~np.roll(inside, 1, axis=1) | ~np.roll(inside, -1, axis=1)
        )
        ys, xs = np.nonzero(boundary)
        target_count = int(round(3 + self.city_size * 4))
        by_settlement = {}
        for y, x in zip(ys, xs):
            sid = int(city_mask[y, x])
            by_settlement.setdefault(sid, []).append((float(x), float(y)))
        settlement_by_id = {s.location_id: s for s in self.settlements}
        boundary_nodes = []
        owner = {}
        for sid, pts in by_settlement.items():
            settlement = settlement_by_id.get(sid)
            if settlement is None or not pts:
                continue
            angles = np.array([np.arctan2(y - settlement.y, x - settlement.x) for x, y in pts])
            order = np.argsort(angles)
            sorted_pts = [pts[i] for i in order]
            n = len(sorted_pts)
            count = min(target_count, n)
            for k in range(count):
                x, y = sorted_pts[int(round(k * n / count)) % n]
                node = PlotNode(node_id=self.next_node_id, node_location=(x, y),
                                connector_id=[], connector_distance=[], connector_elevation=[],
                                connector_movecost=[], connector_edge_id=[])
                self.next_node_id += 1
                boundary_nodes.append(node)
                owner[node.node_id] = sid
        return boundary_nodes, owner

    def _reset_plot_nodes(self):
        self._recompute_background()
        self.next_node_id = 0
        # Sicherheitsabstand zur Stadtgrenze, verhindert dass Kern-Nodes
        # zu nah an den dichten Boundary-Node-Clustern spawnen (Qhull-Absturz
        # durch numerisch fast-degenerierte Nachbarschaftskonstellation).
        MIN_BUFFER_TO_CITY_PX = 10.0
        city_inside = self.city_mask >= 0
        dist_to_city = distance_transform_edt(~city_inside)
        valid_mask = (self.civ_map >= 0.2) & (self.city_mask < 0) & (dist_to_city >= MIN_BUFFER_TO_CITY_PX)
        positions = self._best_candidate_sample(
            valid_mask, int(self.plotnodes_count),
            self.plot_base_spacing, self.plot_civ_spacing_factor, self.civ_map)
        self.nodes = []
        for x, y in positions:
            node = PlotNode(node_id=self.next_node_id, node_location=(x, y),
                            connector_id=[], connector_distance=[], connector_elevation=[],
                            connector_movecost=[], connector_edge_id=[])
            node.traffic_weight = random_mod.uniform(3.0, 5.0)
            self.next_node_id += 1
            self.nodes.append(node)
        boundary_nodes, self.boundary_owner = self._generate_city_boundary_nodes()
        self.nodes.extend(boundary_nodes)
        self.ridge_traffic_history = {}
        self.current_ridge_edges = []
        self.ridge_vertex_positions = None
        self._voronoi_cache_tick = 0
        self._cached_ridge_edges = []
        self._cached_vertex_positions = None
        self._cached_predecessors = None
        self._cached_distances = None
        self._cached_boundary_entries = []
        self._cached_boundary_settlement = []
        self._cached_node_entry = {}
        self._cached_num_vertices = 0
        self._cached_core_springs = []  # NEU
        self.iteration = 0
        self._redraw()

    # ---------------------------------------------------- Voronoi-Isolation --
    def _ensure_voronoi_worker(self):
        """Startet den persistenten Worker-Prozess, falls er noch nicht laeuft
        oder nach einem Crash neu gestartet werden muss."""
        if getattr(self, "_voronoi_proc", None) is not None and self._voronoi_proc.is_alive():
            return
        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(target=_voronoi_worker_loop, args=(child_conn,), daemon=True)
        proc.start()
        self._voronoi_proc = proc
        self._voronoi_conn = parent_conn
        logger.info("Voronoi-Worker-Prozess gestartet (pid=%s).", proc.pid)

    def _safe_voronoi(self, points, timeout=3.0):
        """Sendet Punkte an den persistenten Worker-Prozess statt jedes Mal
        einen neuen Prozess zu starten. Bei Crash/Timeout wird der Worker
        automatisch neu gestartet, damit das Programm dauerhaft lauffaehig
        bleibt."""
        self._ensure_voronoi_worker()
        try:
            self._voronoi_conn.send(points)
        except (BrokenPipeError, OSError) as e:
            logger.error("Voronoi-Worker-Pipe defekt beim Senden: %s", e)
            self._voronoi_proc = None
            return None

        if not self._voronoi_conn.poll(timeout):
            logger.error(
                "Voronoi-Worker Timeout (>%.1fs). n_points=%d, iteration=%d. "
                "Worker wird neu gestartet.",
                timeout, len(points), self.iteration
            )
            np.save(
                os.path.join(LOG_DIR, f"crash_dump_iter{self.iteration}.npy"),
                points,
            )
            self._voronoi_proc.terminate()
            self._voronoi_proc.join()
            self._voronoi_proc = None
            return None

        if not self._voronoi_proc.is_alive():
            logger.error(
                "Voronoi-Worker abgestuerzt (exitcode=%s). n_points=%d, iteration=%d",
                self._voronoi_proc.exitcode, len(points), self.iteration
            )
            np.save(
                os.path.join(LOG_DIR, f"crash_dump_iter{self.iteration}.npy"),
                points,
            )
            self._voronoi_proc = None
            return None

        try:
            result = self._voronoi_conn.recv()
        except (EOFError, OSError) as e:
            logger.error("Voronoi-Worker-Pipe defekt beim Empfangen: %s", e)
            self._voronoi_proc = None
            return None

        if "error" in result:
            logger.warning("Voronoi lieferte Exception: %s", result["error"])
            return None
        return result

    # -------------------------------------------------------- Tick-Physik --
    def _physics_step(self):
        regular = [n for n in self.nodes if n.node_id not in self.boundary_owner]
        if len(regular) < 3:
            return

        spring_forces = self._apply_spring_forces()

        for node in regular:
            x, y = node.node_location
            field_force = self._sample_field(x, y).copy()

            # Die Summe vieler einzeln gekappter Federn konnte bisher trotzdem
            # sehr gross werden. Den resultierenden Federvektor ebenfalls
            # begrenzen, bevor er mit dem Feld kombiniert wird.
            spring_force = spring_forces.get(node.node_id, np.zeros(2)).copy()
            spring_mag = float(np.hypot(spring_force[0], spring_force[1]))
            if spring_mag > self.MAX_SPRING_RESULTANT:
                spring_force *= self.MAX_SPRING_RESULTANT / spring_mag

            # Expliziter kleiner Zeitschritt verhindert das Ueberschiessen um
            # die Feder-Ruhelage bei hohen Steifigkeiten.
            displacement = (field_force + spring_force) * self.PHYSICS_TIME_STEP

            mag = float(np.hypot(displacement[0], displacement[1]))
            if mag > self.MAX_DISPLACEMENT_PER_TICK:
                displacement = displacement / mag * self.MAX_DISPLACEMENT_PER_TICK

            old_pos = np.array(node.node_location, dtype=float)
            new_pos = old_pos + displacement

            # Der Kartenrand ist eine harte Wand. Eine Feder darf einen Kern
            # bis an die Wand ziehen, aber nicht darueber hinaus.
            new_pos[0] = np.clip(new_pos[0], self.MAP_EDGE_INSET,
                                 float(self.map_size) - self.MAP_EDGE_INSET)
            new_pos[1] = np.clip(new_pos[1], self.MAP_EDGE_INSET,
                                 float(self.map_size) - self.MAP_EDGE_INSET)

            # Wildnisgrenze: richtungsabhaengige Pruefung, damit Knoten in der
            # Wildnis zurueck zur Zivilisation wandern koennen (Potentialfeld).
            new_civ = self._civ_at_continuous(new_pos)
            old_civ = self._civ_at_continuous(old_pos)

            if new_civ < self.WILDERNESS_CIV_THRESHOLD:
                if old_civ >= self.WILDERNESS_CIV_THRESHOLD:
                    # Grenzueberschreitung Civ -> Wild: Bisektion zur Grenze
                    valid = old_pos.copy()
                    invalid = new_pos.copy()
                    for _ in range(12):
                        middle = (valid + invalid) * 0.5
                        if self._civ_at_continuous(middle) >= self.WILDERNESS_CIV_THRESHOLD:
                            valid = middle
                        else:
                            invalid = middle
                    new_pos = valid
                elif new_civ < old_civ:
                    # Bereits in der Wildnis UND Bewegung fuehrt tiefer hinein:
                    # blockieren, damit Knoten nicht endlos weiter abwandern.
                    new_pos = old_pos
                # sonst: new_civ >= old_civ -> Bewegung verbessert die Position
                # (naeher an der Zivilisation) -> erlauben

            node.node_location = (float(new_pos[0]), float(new_pos[1]))

    # ---------------------------------------------------- Wege-Netz & Traffic --
    def _edge_key(self, p1, p2):
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        return (round(mx, 0), round(my, 0))

    def height_at(self, pos):
        x, y = int(round(pos[0])), int(round(pos[1]))
        h, w = self.heightmap.shape
        if 0 <= y < h and 0 <= x < w:
            return float(self.heightmap[y, x])
        return 0.0

    def _sampled_slope(self, p1, p2, length):

        if length < 1e-6:
            return 0.0
        if length <= 12.0:
            return abs(self.height_at(p1) - self.height_at(p2)) / length
        num_segments = max(1, int(np.ceil(length / 10.0)))
        t = np.linspace(0, 1, num_segments + 1)
        xs = p1[0] + (p2[0] - p1[0]) * t
        ys = p1[1] + (p2[1] - p1[1]) * t
        h, w = self.heightmap.shape
        ix = np.clip(xs.round().astype(int), 0, w - 1)
        iy = np.clip(ys.round().astype(int), 0, h - 1)
        heights = self.heightmap[iy, ix]
        cumulative_height_change = float(np.sum(np.abs(np.diff(heights))))
        return cumulative_height_change / length

    def _build_traffic_graph(self):
        if self._voronoi_cache_tick < self.VORONOI_RECOMPUTE_INTERVAL - 1:
            if self._cached_vertex_positions is not None and self._cached_ridge_edges:
                self.ridge_vertex_positions = self._cached_vertex_positions
                self.current_ridge_edges = self._cached_ridge_edges
                return False
        self._voronoi_cache_tick = 0
        points = np.array([n.node_location for n in self.nodes])
        if len(points) < 4:
            self.ridge_vertex_positions = None
            self.current_ridge_edges = []
            self._cached_ridge_edges = []
            self._cached_vertex_positions = None
            self._cached_core_springs = []
            return False

        # Schutz gegen Qhull-Abstuerze (Windows STATUS_STACK_BUFFER_OVERRUN) bei
        # (nahezu) identischen oder exakt kollinearen Punkten (z.B. Boundary-Nodes
        # auf dem Pixelraster der Stadtgrenze): minimaler kontinuierlicher Jitter
        # auf ALLE Punkte, wirkt nur auf diese lokale Kopie, nicht auf die
        # tatsaechlichen node.node_location-Werte.
        rng = np.random.RandomState(self.iteration)
        points = points + rng.uniform(-0.05, 0.05, size=points.shape)

        vor_result = self._safe_voronoi(points, timeout=3.0)
        if vor_result is None:
            self._voronoi_failure_count = getattr(self, "_voronoi_failure_count", 0) + 1
            logger.warning(
                "Voronoi fehlgeschlagen (%d. Mal in Folge). Fallback auf Cache.",
                self._voronoi_failure_count
            )
            if self._voronoi_failure_count >= 3:
                self._min_buffer_to_city_px = getattr(self, "_min_buffer_to_city_px", 10.0) + 2.0
                logger.error(
                    "3+ aufeinanderfolgende Voronoi-Fehler. Buffer-Zone automatisch "
                    "erhoeht auf %.1fpx. Reset empfohlen.",
                    self._min_buffer_to_city_px
                )
            self.ridge_vertex_positions = None
            self.current_ridge_edges = []
            self._cached_ridge_edges = []
            self._cached_vertex_positions = None
            self._cached_core_springs = []
            return False

        self._voronoi_failure_count = 0
        vertex_positions = vor_result["vertices"]
        num_vertices = len(vertex_positions)
        ridge_points = vor_result["ridge_points"]
        ridge_vertices_list = vor_result["ridge_vertices"]
        point_region = vor_result["point_region"]
        regions = vor_result["regions"]

        civ_union = unary_union(self._wilderness_polygons) if self._wilderness_polygons else None

        def clip_segment_to_civ(p1, p2):
            if civ_union is None:
                return [(p1, p2, False)]
            civ1 = self._civ_at(p1)
            civ2 = self._civ_at(p2)
            if civ1 > self.WILDERNESS_CIV_THRESHOLD + 0.05 and civ2 > self.WILDERNESS_CIV_THRESHOLD + 0.05:
                return [(p1, p2, False)]
            try:
                line = LineString([p1, p2])
                inter = line.intersection(civ_union)
                if inter.is_empty:
                    return [(p1, p2, True)]
                if inter.length >= line.length - 1e-6:
                    return [(p1, p2, False)]
                civ_part = inter
                wild_part = line.difference(civ_union)
                result = []
                for geom, is_wild in [(civ_part, False), (wild_part, True)]:
                    if geom.is_empty:
                        continue
                    if geom.geom_type == "LineString":
                        coords = list(geom.coords)
                        result.append((coords[0], coords[-1], is_wild))
                    elif geom.geom_type == "MultiLineString":
                        for g in geom.geoms:
                            coords = list(g.coords)
                            result.append((coords[0], coords[-1], is_wild))
                return result
            except Exception as e:
                logger.warning("clip_segment_to_civ fehlgeschlagen p1=%s p2=%s: %s", p1, p2, e)
                return [(p1, p2, False)]

        map_w, map_h = float(self.map_size), float(self.map_size)
        center = points.mean(axis=0)

        def clip_to_map_border(p_finite, richtung):
            dx, dy = richtung
            t_candidates = []
            if abs(dx) > 1e-9:
                t_candidates.append((0.0 - p_finite[0]) / dx)
                t_candidates.append((map_w - p_finite[0]) / dx)
            if abs(dy) > 1e-9:
                t_candidates.append((0.0 - p_finite[1]) / dy)
                t_candidates.append((map_h - p_finite[1]) / dy)
            t_valid = [t for t in t_candidates if t > 0]
            if not t_valid:
                return None
            t = min(t_valid)
            return p_finite[0] + dx * t, p_finite[1] + dy * t

        ridge_edges = []
        core_springs = []  # NEU: (core_i, core_j, key, dist0)
        for ridge_idx, ridge in enumerate(ridge_vertices_list):
            i, j = ridge
            if i == -1 or j == -1:
                finite_idx = j if i == -1 else i
                p_finite = tuple(vertex_positions[finite_idx])
                p1_idx, p2_idx = ridge_points[ridge_idx]
                midpoint = (points[p1_idx] + points[p2_idx]) / 2.0
                direction = midpoint - center
                norm = np.hypot(direction[0], direction[1])
                if norm < 1e-9:
                    continue
                direction = direction / norm
                edge_dir = points[p2_idx] - points[p1_idx]
                perp = np.array([-edge_dir[1], edge_dir[0]])
                perp_norm = np.hypot(perp[0], perp[1])
                if perp_norm < 1e-9:
                    continue
                perp = perp / perp_norm
                if np.dot(perp, direction) < 0:
                    perp = -perp
                far_point = clip_to_map_border(p_finite, perp)
                if far_point is None:
                    continue
                p1, p2 = p_finite, far_point
            else:
                p1, p2 = tuple(vertex_positions[i]), tuple(vertex_positions[j])
                # NEU: Core-Feder fuer diese finite Ridge registrieren
                core_i, core_j = ridge_points[ridge_idx]
                dist0 = float(np.hypot(points[core_j, 0] - points[core_i, 0],
                                       points[core_j, 1] - points[core_i, 1]))
                if dist0 > 1e-6:
                    key = self._edge_key(p1, p2)
                    core_springs.append((int(core_i), int(core_j), key, dist0))

            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if length < 1e-6:
                continue
            for seg_p1, seg_p2, is_wild in clip_segment_to_civ(p1, p2):
                seg_len = float(np.hypot(seg_p2[0] - seg_p1[0], seg_p2[1] - seg_p1[1]))
                if seg_len < 1e-6:
                    continue
                raw_slope = self._sampled_slope(seg_p1, seg_p2, seg_len)
                normalized_slope = min(1.0, raw_slope / 30.0)
                cost = seg_len * (1.0 + self.plot_height_cost_factor * normalized_slope)
                ridge_edges.append((i, j, seg_p1, seg_p2, cost))

        tree = cKDTree(vertex_positions)
        node_entry = {}
        entry_edges = []
        for idx, node in enumerate(self.nodes):
            try:
                region_idx = point_region[idx]
                region = regions[region_idx]
                finite_vertices = [v for v in region if v >= 0]
                node_pos = np.array(node.node_location)
                if finite_vertices:
                    cand_positions = vertex_positions[finite_vertices]
                    dists = np.hypot(cand_positions[:, 0] - node_pos[0], cand_positions[:, 1] - node_pos[1])
                    best_local = int(np.argmin(dists))
                    vertex_idx = finite_vertices[best_local]
                    dist = float(dists[best_local])
                else:
                    dist, vertex_idx = tree.query(node.node_location)
                    vertex_idx = int(vertex_idx)
                entry_idx = num_vertices + idx
                node_entry[node.node_id] = entry_idx
                entry_edges.append((entry_idx, vertex_idx, max(float(dist), 1e-3)))
            except Exception as e:
                logger.warning("Fehler bei node %s (idx=%d): %s", node.node_id, idx, e)
                continue

        total_graph_nodes = num_vertices + len(self.nodes)
        rows, cols, costs = [], [], []
        for i, j, p1, p2, cost in ridge_edges:
            if i == -1 or j == -1:
                continue
            rows += [i, j]
            cols += [j, i]
            costs += [cost, cost]
        for entry_idx, vertex_idx, cost in entry_edges:
            rows += [entry_idx, vertex_idx]
            cols += [vertex_idx, entry_idx]
            costs += [cost, cost]

        boundary_entries = []
        boundary_settlement = []
        for nid, sid in self.boundary_owner.items():
            if nid in node_entry:
                boundary_entries.append(node_entry[nid])
                boundary_settlement.append(sid)
        if not boundary_entries:
            self.ridge_vertex_positions = None
            self.current_ridge_edges = []
            self._cached_ridge_edges = []
            self._cached_vertex_positions = None
            self._cached_core_springs = []
            return False

        try:
            graph = csr_matrix((costs, (rows, cols)), shape=(total_graph_nodes, total_graph_nodes))
            distances, predecessors = dijkstra(graph, indices=boundary_entries, return_predecessors=True)
        except Exception as e:
            logger.error("Dijkstra fehlgeschlagen: total_graph_nodes=%d: %s", total_graph_nodes, e)
            self.ridge_vertex_positions = None
            self.current_ridge_edges = []
            self._cached_ridge_edges = []
            self._cached_vertex_positions = None
            self._cached_core_springs = []
            return False

        self._cached_vertex_positions = vertex_positions
        self._cached_ridge_edges = ridge_edges
        self._cached_num_vertices = num_vertices
        self._cached_boundary_entries = boundary_entries
        self._cached_boundary_settlement = boundary_settlement
        self._cached_node_entry = node_entry
        self._cached_predecessors = predecessors
        self._cached_distances = distances
        self._cached_core_springs = core_springs  # NEU
        self.ridge_vertex_positions = vertex_positions
        self.current_ridge_edges = ridge_edges
        return True

    def _apply_spring_forces(self):
        """Zwei Federn pro Delaunay-Nachbarpaar: Spacing-Feder (ersetzt Repulsion,
        Ruhelaenge = lokaler civ-abhaengiger Zielabstand) und Traffic-Feder
        (zusaetzliche Kraft, die staerker befahrene Verbindungen zusammenzieht).
        Force-Clamp verhindert, dass ein bei hoher Node-Dichte unerreichbares
        target_spacing zu explosionsartigen Kraeften fuehrt, die alle Nodes
        an den Kartenrand drueckt (Kollinearitaet -> Qhull-Absturz)."""
        forces = {}
        if not self._cached_core_springs:
            return forces
        MIN_TARGET_SPACING = 2.0
        MAX_SPRING_FORCE = 5.0  # Obergrenze pro einzelner Federkraft, unabhaengig vom Delta
        for core_i, core_j, key, dist0 in self._cached_core_springs:
            if core_i >= len(self.nodes) or core_j >= len(self.nodes):
                continue
            node_i = self.nodes[core_i]
            node_j = self.nodes[core_j]
            if node_i.node_id in self.boundary_owner or node_j.node_id in self.boundary_owner:
                continue
            pos_i = np.array(node_i.node_location)
            pos_j = np.array(node_j.node_location)
            delta_vec = pos_j - pos_i
            dist = float(np.hypot(delta_vec[0], delta_vec[1]))
            if dist < 1e-6:
                continue
            direction = delta_vec / dist

            mid = (pos_i + pos_j) * 0.5
            civ_here = self._civ_at((mid[0], mid[1]))
            target_spacing = max(
                self.plot_base_spacing / (1.0 + self.plot_civ_spacing_factor * civ_here),
                MIN_TARGET_SPACING
            )

            spacing_delta = dist - target_spacing
            spacing_mag = np.clip(self.spacing_spring_stiffness * spacing_delta, -MAX_SPRING_FORCE, MAX_SPRING_FORCE)
            spacing_force = direction * spacing_mag

            traffic = self.ridge_traffic_history.get(key, 0.0)
            shrink = 1.0 / (1.0 + self.spring_traffic_shrink * traffic)
            traffic_rest_length = max(target_spacing * shrink, target_spacing * self.spring_min_shrink_fraction)
            traffic_delta = dist - traffic_rest_length
            traffic_mag = np.clip(self.traffic_spring_stiffness * traffic_delta, -MAX_SPRING_FORCE, MAX_SPRING_FORCE)
            traffic_force = direction * traffic_mag

            f = spacing_force + traffic_force
            forces[node_i.node_id] = forces.get(node_i.node_id, np.zeros(2)) + f
            forces[node_j.node_id] = forces.get(node_j.node_id, np.zeros(2)) - f
        return forces

    def _simulate_traffic(self):
        self._build_traffic_graph()
        vertex_positions = self._cached_vertex_positions
        ridge_edges = self._cached_ridge_edges
        num_vertices = self._cached_num_vertices
        predecessors = self._cached_predecessors
        distances = self._cached_distances
        boundary_entries = self._cached_boundary_entries
        boundary_settlement = self._cached_boundary_settlement
        node_entry = self._cached_node_entry
        path_cache = {}
        if vertex_positions is None or not ridge_edges:
            self.ridge_traffic_history = {}
            return

        def trace_and_add(pred_row, source_entry, target_entry, amount, contrib):
            cache_key = (id(pred_row), source_entry, target_entry)
            cached_keys = path_cache.get(cache_key)
            if cached_keys is None:
                cached_keys = []
                current = target_entry
                while current != source_entry and current >= 0:
                    prev = pred_row[current]
                    if prev < 0:
                        break
                    if current < num_vertices and prev < num_vertices:
                        cached_keys.append(self._edge_key(vertex_positions[current], vertex_positions[prev]))
                    current = prev
                path_cache[cache_key] = cached_keys
            for key in cached_keys:
                contrib[key] = contrib.get(key, 0.0) + amount

        city_boundary_indices = {}
        for row, (sid, entry) in enumerate(zip(boundary_settlement, boundary_entries)):
            city_boundary_indices.setdefault(sid, []).append((row, entry))

        fresh_contrib = {}
        for node in self.nodes:
            if node.node_id in self.boundary_owner:
                continue
            entry_idx = node_entry.get(node.node_id)
            if entry_idx is None:
                continue
            node_distances = distances[:, entry_idx]
            per_settlement_best = {}
            for sid, rows in city_boundary_indices.items():
                best = None
                for row, _ in rows:
                    d = node_distances[row]
                    if not np.isfinite(d):
                        continue
                    if best is None or d < best[0]:
                        best = (d, row)
                if best is not None:
                    per_settlement_best[sid] = best
            if not per_settlement_best:
                continue
            ranked = sorted(per_settlement_best.items(), key=lambda kv: kv[1][0])
            weights = self._halving_weights(len(ranked))
            traffic_weight = getattr(node, "traffic_weight", 4.0)
            for rank, (sid, (d, row)) in enumerate(ranked):
                amount = weights[rank] * traffic_weight
                trace_and_add(predecessors[row], boundary_entries[row], entry_idx, amount, fresh_contrib)

        settlement_ids = [s.location_id for s in self.settlements]
        if len(settlement_ids) > 1:
            for sid_from in settlement_ids:
                own_rows = [row for row, sid in enumerate(boundary_settlement) if sid == sid_from]
                if not own_rows:
                    continue
                other_best = {}
                for sid_to in settlement_ids:
                    if sid_to == sid_from:
                        continue
                    cols_to = [row for row, sid in enumerate(boundary_settlement) if sid == sid_to]
                    best = None
                    for r_from in own_rows:
                        for c_row in cols_to:
                            target_entry = boundary_entries[c_row]
                            d = distances[r_from, target_entry]
                            if not np.isfinite(d):
                                continue
                            if best is None or d < best[0]:
                                best = (d, r_from, target_entry)
                    if best is not None:
                        other_best[sid_to] = best
                if not other_best:
                    continue
                ranked = sorted(other_best.items(), key=lambda kv: kv[1][0])
                weights = PlotNodeSystem._rank_distance_weights(len(ranked))
                for rank, (sid_to, (d, r_from, target_entry)) in enumerate(ranked):
                    amount = weights[rank] * self.plot_intercity_traffic
                    trace_and_add(predecessors[r_from], boundary_entries[r_from], target_entry, amount, fresh_contrib)

        TRAFFIC_DECAY = 0.15
        KEEP_FACTOR = 1.0 - TRAFFIC_DECAY
        new_history = {}
        for i, j, p1, p2, cost in ridge_edges:
            key = self._edge_key(p1, p2)
            old = self.ridge_traffic_history.get(key, 0.0)
            new_history[key] = old * KEEP_FACTOR + fresh_contrib.get(key, 0.0)
        self.ridge_traffic_history = new_history

    def _halving_weights(self, n):
        """50%, 25%, 12.5%, ... - letzter Rang bekommt den Rest."""
        if n <= 0:
            return []
        weights = []
        remaining = 1.0
        for rank in range(n - 1):
            w = remaining * 0.5
            weights.append(w)
            remaining -= w
        weights.append(remaining)  # letzter Rang bekommt den Rest exakt
        return weights

    def _tick(self):
        if not self.playing:
            return
        self.iteration += 1
        self._tick_timings = {}
        self._start_timer("physics_step")
        self._voronoi_cache_tick += 1
        self._physics_step()
        self._stop_timer("physics_step")

        self._start_timer("simulate_traffic")
        self._simulate_traffic()
        self._stop_timer("simulate_traffic")

        self._start_timer("redraw")
        self._redraw()
        self._stop_timer("redraw")

    # ------------------------------------------------------------- Zeichnen --
    def _wiggly_path(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        length = float(np.hypot(x2 - x1, y2 - y1))
        max_fraction = self.spline_wiggle_pct / 100.0
        if length < 1e-6 or max_fraction <= 0:
            return np.array([x1, x2]), np.array([y1, y2])
        steps = max(6, int(length))
        t = np.linspace(0, 1, steps)
        xs_straight = x1 + (x2 - x1) * t
        ys_straight = y1 + (y2 - y1) * t
        h, w = self.heightmap.shape
        ix = np.clip(xs_straight.round().astype(int), 0, w - 1)
        iy = np.clip(ys_straight.round().astype(int), 0, h - 1)
        heights = self.heightmap[iy, ix]
        height_cost = float(np.sum(np.abs(np.diff(heights))))
        avg_slope = height_cost / length
        normalized_slope = min(1.0, avg_slope / 30.0)
        wiggle_fraction = max_fraction * normalized_slope
        amplitude = wiggle_fraction * length
        if amplitude < 0.5:
            return xs_straight, ys_straight
        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        perp_x, perp_y = -dy, dx
        envelope = np.sin(t * np.pi)
        wave = np.sin(t * np.pi * self.spline_detail)
        offset = amplitude * envelope * wave
        return xs_straight + perp_x * offset, ys_straight + perp_y * offset

    def _redraw(self):
        MIN_TRAFFIC = 20.0 * self.plot_tier_factor
        TRAFFIC_MAX = 300.0 * self.plot_tier_factor

        self.ax.clear()

        # Terrain-Basisfarben, ab Gruen (Wasseranteil der 'terrain'-cmap ausblenden)
        hmin = float(np.percentile(self.heightmap, 2))
        hmax = float(np.max(self.heightmap))
        # 'terrain' cmap: Blau=Wasser (~0-0.3), Gruen ab ~0.35 im Colormap-Bereich
        # deshalb vmin nach unten schieben, damit hmin schon im gruenen Bereich der cmap liegt
        terrain_vmin = hmin - 0.55 * (hmax - hmin)
        self.ax.imshow(
            self.heightmap, cmap='terrain', origin='lower',
            vmin=terrain_vmin, vmax=hmax,
            extent=(0, self.map_size, 0, self.map_size), zorder=0
        )

        if self.show_potential_overlay and self.potential_field is not None:
            self._draw_potential_quiver()

        if self.show_civ_overlay:
            self.ax.imshow(self.civ_map, cmap="hot", origin="lower", alpha=0.5, vmin=0.0, vmax=1.0,
                           extent=(0, self.map_size, 0, self.map_size), zorder=1)

        if self.show_potential_overlay and self.potential_field is not None:
            magnitude = np.sqrt(self.potential_field[:, :, 0] ** 2 + self.potential_field[:, :, 1] ** 2)
            knee = np.percentile(magnitude, 95) if np.max(magnitude) > 0 else 1.0
            knee = max(knee, 1e-9)
            normalized = magnitude / knee
            magnitude_display = np.where(
                normalized <= 1.0,
                normalized,
                1.0 + np.tanh(normalized - 1.0)
            ) / 2.0
            self.ax.imshow(magnitude_display, cmap="viridis", origin="lower", alpha=0.4,
                           extent=(0, self.map_size, 0, self.map_size), zorder=1)

        if self.show_potential_overlay and self.potential_field is not None:
            magnitude = np.sqrt(self.potential_field[:, :, 0] ** 2 + self.potential_field[:, :, 1] ** 2)
            knee = np.percentile(magnitude, 95) if np.max(magnitude) > 0 else 1.0
            knee = max(knee, 1e-9)
            normalized = magnitude / knee
            magnitude_display = np.where(
                normalized <= 1.0,
                normalized,
                1.0 + np.tanh(normalized - 1.0)
            ) / 2.0
            self.ax.imshow(magnitude_display, cmap="viridis", origin="lower", alpha=0.4,
                           extent=(0, self.map_size, 0, self.map_size), zorder=1)

        inside = (self.city_mask >= 0).astype(float)
        if np.any(inside):
            self.ax.contour(inside, levels=[0.5], colors="gold", linewidths=2.0,
                             extent=(0, self.map_size, 0, self.map_size))

        wilderness = (self.civ_map < self.WILDERNESS_CIV_THRESHOLD).astype(float)
        if np.any(wilderness):
            self.ax.contour(wilderness, levels=[0.5], colors="cyan", linewidths=0.8, alpha=0.4,
                             extent=(0, self.map_size, 0, self.map_size))

        polygon_segments = [(p1, p2) for (i, j, p1, p2, cost) in self.current_ridge_edges]
        if polygon_segments:
            self.ax.add_collection(LineCollection(
                polygon_segments, colors="dimgray", linewidths=0.5, alpha=0.5, zorder=2))

        # --- Kontinuierliche Wege-Darstellung (YlOrRd + fliessende Breite) ---
        strasse_count = weg_count = pfad_count = 0
        total_traffic = 0.0
        active_segs, active_norms, active_widths = [], [], []
        unused_segs = []
        for i, j, p1, p2, cost in self.current_ridge_edges:
            key = self._edge_key(p1, p2)
            traffic = self.ridge_traffic_history.get(key, 0.0)
            xs, ys = self._wiggly_path(tuple(p1), tuple(p2))
            seg = np.column_stack([xs, ys])
            if traffic >= MIN_TRAFFIC:
                active_segs.append(seg)
                norm = np.clip(np.log10(1.0 + traffic) / np.log10(1.0 + TRAFFIC_MAX), 0.0, 1.0)
                active_norms.append(norm)
                width = 0.3 + norm * 2.7
                active_widths.append(width)
                total_traffic += traffic
                if traffic >= 170.0 * self.plot_tier_factor:
                    strasse_count += 1
                elif traffic >= 90.0 * self.plot_tier_factor:
                    weg_count += 1
                else:
                    pfad_count += 1
            else:
                unused_segs.append(seg)

        if active_segs:
            coll = LineCollection(active_segs, array=np.array(active_norms), cmap="YlOrRd",
                                  linewidths=active_widths, zorder=4)
            self.ax.add_collection(coll)
        if unused_segs:
            self.ax.add_collection(LineCollection(unused_segs, colors="lightgray", linewidths=0.35, zorder=4))

        if self.ridge_vertex_positions is not None and len(self.ridge_vertex_positions) > 0:
            self.ax.scatter(self.ridge_vertex_positions[:, 0], self.ridge_vertex_positions[:, 1],
                            s=6, c="blue", zorder=5, alpha=0.7)

        regular_nodes = [n for n in self.nodes if n.node_id not in self.boundary_owner]
        if regular_nodes:
            xs = [n.node_location[0] for n in regular_nodes]
            ys = [n.node_location[1] for n in regular_nodes]
            self.ax.scatter(xs, ys, s=14, c="red", zorder=6)

        boundary_nodes = [n for n in self.nodes if n.node_id in self.boundary_owner]
        if boundary_nodes:
            xs = [n.node_location[0] for n in boundary_nodes]
            ys = [n.node_location[1] for n in boundary_nodes]
            self.ax.scatter(xs, ys, s=25, c="gold", marker="^", zorder=6, edgecolors="black", linewidths=0.4)

        sx = [s.x for s in self.settlements]
        sy = [s.y for s in self.settlements]
        self.ax.scatter(sx, sy, s=220, c="crimson", marker="*", zorder=7, edgecolors="black")

        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"Iteration {self.iteration}  |  Plotkerne: {len(regular_nodes)}  |  "
            f"Straße: {strasse_count}  Weg: {weg_count}  Pfad: {pfad_count}  Total Traffic: {total_traffic:.1f}")
        self.status_label.setText(f"Iteration: {self.iteration} | {self._perf_text()}")
        self.canvas.draw_idle()

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
