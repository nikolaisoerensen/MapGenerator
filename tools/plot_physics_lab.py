"""
Path: tools/plot_physics_lab.py

Interaktives Testprogramm fuer die Plot-Physik: Plotkerne (rote Punkte,
best-candidate-verteilt ueber die civ_map), civ-gewichtete Nachbar-Abstossung,
rang-distanz-gewichtete "Gravitation" zu den Staedten, ein separates
Wege-Netz (Splines zwischen Plotnodes = Voronoi-Kreuzungen der Plotkerne,
3-stufige Klassifizierung Pfad/Weg/Strasse mit Traffic-Abklingen pro Tick).

Laeuft komplett standalone ohne DataLODManager/GenerationOrchestrator/
CalculatorDispatcher - nur eine einfache Heightmap (reine Terrain-
Rauschgenerierung, Default-Parameter aus value_default.py), 3 Settlements,
civ_map und ein grobes Landschafts-Voronoi als Hintergrund.

Alles hier ist bewusst NUR im Lab-Tool implementiert (nicht in
core/settlement_generator.py) - explorativ, wie vom User gewuenscht, bevor
entschieden wird, ob/wie Teile davon in die echte Pipeline uebernommen
werden. Einzige Ausnahme bleibt der civ_map-Steigungs-Bugfix in
CivilizationInfluenceMapper (Runde 6, echter Core-Fix).

Start: .venv/Scripts/python.exe tools/plot_physics_lab.py
"""
import sys
import os
import random as random_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QCheckBox, QGroupBox, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

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


class PlotPhysicsLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot-Node Physics Lab")
        self.resize(1500, 950)

        self.map_size = 256
        self.iteration = 0
        self.playing = True
        self.next_node_id = 0
        self.boundary_owner = {}  # node_id -> settlement_id (Grenz-Nodes, siehe _generate_city_boundary_nodes())
        self.ridge_traffic_history = {}  # edge_key -> traffic (persistiert, 50%/Tick Abklingen)
        self.current_ridge_edges = []  # [(i, j, p1, p2, cost), ...] des aktuellen Ticks
        self.ridge_vertex_positions = None  # Plotnodes (Voronoi-Vertices von Plotkerne+Grenz-Nodes)

        # Mindestabstände bei der Settlement-Platzierung (Nutzer-Vorgabe)
        self.settlement_border_margin = 25.0
        self.settlement_min_distance = 35.0

        # Live-Physik-Parameter
        self.plot_base_spacing = 8.0
        self.plot_civ_spacing_factor = 3.0
        self.plot_repulsion_strength = 1.0  # Staerke der Nachbar-"Antigravitation" (Regel 2)
        self.plot_gravity_strength = 0.01  # Staerke der Stadt-"Gravitation" (Regel 3)
        self.plot_city_repulsion_strength = 0.5  # Kurzreichweitige Stadtmauer-Gegenkraft (Regel 1)
        self.plot_height_cost_factor = 2.0
        self.plot_tier_factor = 1.0  # Multiplikator auf die Pfad/Weg/Strasse-Basiswerte 20/40/60 (Regel 8)

        self.show_civ_overlay = False  # Debug-Overlay: civ_map als Heatmap (Regel 3)

        # Hintergrund-Parameter (brauchen "Reset Plot Nodes" zum Uebernehmen)
        self.plotnodes_count = 200
        self.civ_influence_decay = 0.8
        # "Stadtgröße" (0..1) steuert city_reach_factor, civ_influence_range,
        # plot_intercity_traffic UND die Anzahl Grenz-Wegknoten (3-7) gemeinsam
        # (Nutzer-Vorgabe) - siehe _recompute_background().
        self.city_size = 0.5
        self.city_reach_factor = 4.0
        self.civ_influence_range = 0.30
        self.plot_intercity_traffic = 30  # 20-60, aus city_size abgeleitet

        # Wege-Darstellung (Regel 8): zwei getrennte Spline-Regler
        self.spline_wiggle_pct = 0.0  # Kurvigkeit (Amplitude)
        self.spline_detail = 3.0  # Detailgrad (Wellenfrequenz)

        self._build_ui()
        self._regenerate_scene()

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(400)

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
        panel.setFixedWidth(360)
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
        self._add_slider(live_form, "plot_base_spacing", "Base Spacing (px)", 2, 30, self.plot_base_spacing, scale=10)
        self._add_slider(live_form, "plot_civ_spacing_factor", "Civ Spacing Factor", 0, 10, self.plot_civ_spacing_factor, scale=10)
        self._add_slider(live_form, "plot_repulsion_strength", "Repulsion Strength (Antigravitation)", 0, 3.0, self.plot_repulsion_strength, scale=100)
        self._add_slider(live_form, "plot_gravity_strength", "Gravity Strength", 0, 0.1, self.plot_gravity_strength, scale=100)
        self._add_slider(live_form, "plot_city_repulsion_strength", "Stadtmauer-Gegenkraft", 0, 5.0, self.plot_city_repulsion_strength, scale=100)
        self._add_slider(live_form, "plot_height_cost_factor", "Height Cost Factor", 0, 10, self.plot_height_cost_factor, scale=10)
        self._add_slider(live_form, "plot_tier_factor", "Pfad/Weg/Straße Faktor (0.25-5, 1=20/40/60)", 0.25, 5.0, self.plot_tier_factor, scale=100)
        live_group.setLayout(live_form)
        panel_layout.addWidget(live_group)

        self.civ_overlay_cb = QCheckBox("Civ-Wert als Heatmap überlagern")
        self.civ_overlay_cb.setChecked(self.show_civ_overlay)
        self.civ_overlay_cb.toggled.connect(self._set_civ_overlay)
        panel_layout.addWidget(self.civ_overlay_cb)

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
        self._add_slider(bg_form, "civ_influence_decay", "Civ Influence Decay", 0.1, 2, self.civ_influence_decay, scale=100)
        bg_group.setLayout(bg_form)
        panel_layout.addWidget(bg_group)

        legend = QLabel(
            "Wege-Linien: Rot = Straße, Orange = Weg, Goldgelb = Pfad, Hellgrau = ungenutzt\n"
            "(Schwellwerte 20/40/60 × Faktor-Slider, Traffic klingt pro Tick um 50% ab)\n"
            "Gold-Kontur = Stadtgrenze (Civ-Wert-Kontour, breitet sich über flache Gebiete "
            "weiter aus als über Steigungen)\n"
            "Sterne = Settlement/Marktplatz (Ziel der Wegknoten an der Stadtgrenze)\n"
            "Kleine gelbe Dreiecke = Grenz-Wegknoten (3-7 je Stadt, abhängig von Stadtgröße) - "
            "ihr Erreichen zählt als Erreichen des Marktplatzes\n"
            "Rote Punkte = Plotkern (best-candidate verteilt, erzeugt konstanten Traffic 3-5)\n"
            "Blaue Punkte = Plotnodes (Voronoi-Kreuzungen, 3-5 je Plot) - Familienverkehr "
            "entsteht an einem zufällig gewählten eigenen Plotnode\n"
            "Dünne graue Linien = Grundstücksgrenzen (immer gerade)\n"
            "Rang-Distanz-Verteilung auf alle erreichbaren Städte (50%/25%/12.5%/...) für "
            "Plotkern-Traffic, Intercity-Traffic UND Gravitation"
        )
        legend.setWordWrap(True)
        panel_layout.addWidget(legend)
        panel_layout.addStretch(1)

    def _add_slider(self, form, attr_name, label, min_val, max_val, default, scale):
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

    # ------------------------------------------------------------ Szene neu --
    def _regenerate_scene(self):
        seed = random_mod.randint(1, 999_999)
        self.heightmap = generate_default_heightmap(self.map_size, seed)
        dz_dy, dz_dx = np.gradient(self.heightmap)
        self.slopemap = np.stack([dz_dx, dz_dy], axis=-1).astype(np.float32)
        self.water_map = (self.heightmap < np.percentile(self.heightmap, 8)).astype(np.float32)

        suitability_analyzer = TerrainSuitabilityAnalyzer(terrain_factor_villages=1.0, map_size=self.map_size)
        suitability_map = suitability_analyzer.create_combined_suitability(
            self.heightmap, self.slopemap, self.water_map)

        # Mindestabstand zum Kartenrand (Nutzer-Vorgabe, Default 25px): Rand-
        # Streifen für die Settlement-Platzierung unattraktiv machen, statt
        # calculate_settlements() selbst zu ändern (dort gibt es aktuell keinen
        # Rand-Parameter). Der Mindestabstand Stadt-zu-Stadt (35px) ist bei
        # map_size 256 mit 3 Settlements schon durch die eingebaute
        # min_distance-Formel (~90px) automatisch erfüllt.
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

    def _recompute_background(self):
        """civ_map und Stadtgrenze - übernimmt die aktuellen Hintergrund-Slider.

        "Stadtgröße" (city_size, 0..1) leitet vier zusammengehörige Werte
        gemeinsam ab (Nutzer-Vorgabe):
        - city_reach_factor: 2.0 (klein) bis 8.0 (groß)
        - civ_influence_range: 0.15 bis 0.45 (Bruchteil der Kartendiagonale)
        - plot_intercity_traffic: 20 bis 60 Personen pro Richtung
        - Grenz-Wegknoten pro Stadt: 3 bis 7 (siehe _generate_city_boundary_nodes())
        """
        self.city_reach_factor = 2.0 + self.city_size * 6.0
        self.civ_influence_range = 0.15 + self.city_size * 0.30
        self.plot_intercity_traffic = 20.0 + self.city_size * 40.0

        self.gen.civ_influence_range = self.civ_influence_range
        self.gen.civ_influence_decay = self.civ_influence_decay
        self.gen.city_reach_factor = self.city_reach_factor
        self.gen.terrain_factor_villages = 1.0

        self.civ_map = self.gen.calculate_civilization_mapping(
            self.heightmap, self.slopemap, self.settlements, [], [])

        boundary_analyzer = CityBoundaryAnalyzer(
            self.gen.terrain_factor_villages, self.city_reach_factor, shader_manager=None)
        self.city_mask, _ = boundary_analyzer.compute_city_boundaries(
            self.heightmap, self.slopemap, self.settlements)

    def _civ_at(self, pos):
        x, y = int(pos[0]), int(pos[1])
        h, w = self.civ_map.shape
        if 0 <= y < h and 0 <= x < w:
            return float(self.civ_map[y, x])
        return 0.0

    # ------------------------------------------------------- Plotkern-Setup --
    def _best_candidate_sample(self, valid_mask, count, base_spacing, civ_spacing_factor, civ_map, k=15):
        """
        Best-Candidate-Sampling (Mitchell): pro Punkt werden k zufällige
        Kandidaten aus der gültigen Fläche gezogen, gewählt wird der, dessen
        (Abstand zum nächsten bereits gewählten Punkt / lokaler civ-abhängiger
        Zielabstand) am größten ist - deutlich gleichmäßigere Verteilung als
        hartes Reject-Sampling bei ähnlichen Kosten (Nutzer-Vorgabe Regel 1).
        Die civ_map ist bereits terrain-steigungs-gewichtet (Runde 6 Core-Fix),
        dadurch fließt der Terrain-Einfluss automatisch mit ein.
        """
        ys, xs = np.nonzero(valid_mask)
        if len(xs) == 0:
            return []
        count = min(count, len(xs))
        chosen = []
        chosen_arr = np.empty((0, 2))
        pool_size = len(xs)
        for _ in range(count):
            idxs = np.random.randint(0, pool_size, size=min(k, pool_size))
            best_pos, best_score = None, -1.0
            for idx in idxs:
                x, y = float(xs[idx]), float(ys[idx])
                civ_value = float(civ_map[int(y), int(x)])
                target_spacing = max(base_spacing / (1.0 + civ_spacing_factor * civ_value), 1e-3)
                if len(chosen_arr):
                    dist = float(np.sqrt(np.min((chosen_arr[:, 0] - x) ** 2 + (chosen_arr[:, 1] - y) ** 2)))
                else:
                    dist = target_spacing * 10.0
                score = dist / target_spacing
                if score > best_score:
                    best_score, best_pos = score, (x, y)
            chosen.append(best_pos)
            chosen_arr = np.array(chosen)
        return chosen

    def _generate_city_boundary_nodes(self):
        """
        Keine Plot-Nodes im Stadt-Inneren, stattdessen 3-7 Wegknoten (abhängig
        von Stadtgröße) gleichmäßig verteilt entlang der Stadtgrenze -
        Erreichen eines dieser Knoten zählt (aus Traffic-/Dijkstra-Sicht) als
        Erreichen des Marktplatzes (Regel 5) - kein separater virtueller
        Marktplatz-Node mehr nötig.
        Returns: (List[PlotNode], Dict[node_id, settlement_id])
        """
        city_mask = self.city_mask
        inside = city_mask >= 0
        boundary = inside & (
            ~np.roll(inside, 1, axis=0) | ~np.roll(inside, -1, axis=0) |
            ~np.roll(inside, 1, axis=1) | ~np.roll(inside, -1, axis=1)
        )
        ys, xs = np.nonzero(boundary)
        target_count = int(round(3 + self.city_size * 4))  # 3..7 je Stadt

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
                node = PlotNode(
                    node_id=self.next_node_id, node_location=(x, y),
                    connector_id=[], connector_distance=[], connector_elevation=[],
                    connector_movecost=[], connector_edge_id=[]
                )
                self.next_node_id += 1
                boundary_nodes.append(node)
                owner[node.node_id] = sid
        return boundary_nodes, owner

    def _reset_plot_nodes(self):
        """Übernimmt alle Hintergrund-Slider und würfelt die Plotkerne neu."""
        self._recompute_background()
        self.next_node_id = 0

        # Stadt-Inneres komplett von Plotkernen ausschließen (Regel 5).
        valid_mask = (self.civ_map >= 0.2) & (self.city_mask < 0)
        positions = self._best_candidate_sample(
            valid_mask, int(self.plotnodes_count),
            self.plot_base_spacing, self.plot_civ_spacing_factor, self.civ_map)

        self.nodes = []
        for x, y in positions:
            node = PlotNode(
                node_id=self.next_node_id, node_location=(x, y),
                connector_id=[], connector_distance=[], connector_elevation=[],
                connector_movecost=[], connector_edge_id=[]
            )
            node.traffic_weight = random_mod.uniform(3.0, 5.0)  # Regel 7: fest, bleibt über Zeit konstant
            self.next_node_id += 1
            self.nodes.append(node)

        boundary_nodes, self.boundary_owner = self._generate_city_boundary_nodes()
        self.nodes.extend(boundary_nodes)

        self.ridge_traffic_history = {}
        self.current_ridge_edges = []
        self.ridge_vertex_positions = None
        self.iteration = 0
        self._redraw()

    # -------------------------------------------------------- Tick-Physik --
    def _physics_step(self):
        """
        Bewegt die Plotkerne: civ-gewichtete Nachbar-"Antigravitation" (Regel 2)
        plus rang-distanz-gewichtete Stadt-"Gravitation" (Regel 3) plus eine
        kurzreichweitige Stadtmauer-Gegenkraft, die viel schneller waechst als
        die Gravitation je naeher man der Stadt kommt (Regel 1, verhindert
        dass Plotkerne unabhaengig von der Gravity-Slider-Staerke in die Stadt
        hineingezogen werden). Ersetzt die alte Traffic-Anziehung vollstaendig
        (Regel 6). Grenz-Wegknoten bleiben fix (repraesentieren die Stadtgrenze).
        """
        regular = [n for n in self.nodes if n.node_id not in self.boundary_owner]
        if len(regular) < 3:
            return

        positions = {n.node_id: np.array(n.node_location, dtype=np.float64) for n in regular}
        displacement = {n.node_id: np.zeros(2) for n in regular}

        points = np.array([n.node_location for n in regular])
        id_by_index = [n.node_id for n in regular]
        try:
            tri = Delaunay(points)
        except Exception:
            tri = None

        if tri is not None:
            seen_pairs = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = int(simplex[i]), int(simplex[j])
                        pair = (min(a, b), max(a, b))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        nid_a, nid_b = id_by_index[a], id_by_index[b]
                        pos_a, pos_b = positions[nid_a], positions[nid_b]
                        delta = pos_b - pos_a
                        dist = float(np.hypot(delta[0], delta[1]))
                        if dist < 1e-6:
                            continue
                        civ_avg = 0.5 * (self._civ_at(pos_a) + self._civ_at(pos_b))
                        min_spacing = self.plot_base_spacing / (1.0 + self.plot_civ_spacing_factor * civ_avg)
                        # Antigravitation statt linearer Kraft (Regel 2): waechst
                        # als Potenz der Naehe, sonst verliert eine lineare Kraft
                        # gegen die Stadt-Gravitation. Ratio gecappt, damit eine
                        # Annaeherung an dist->0 nicht zu einem Sprung ins
                        # Unendliche fuehrt.
                        ratio = min(min_spacing / max(dist, 1e-6), 6.0)
                        push = self.plot_repulsion_strength * ratio ** 3
                        direction = delta / dist
                        displacement[nid_a] -= direction * push * 0.5
                        displacement[nid_b] += direction * push * 0.5

        # Gravitation: jeder Plotkern wird von ALLEN erreichbaren Städten
        # angezogen, gewichtet nach derselben Rang-Distanz-Formel wie beim
        # Traffic (nächste Stadt = volle Kraft, zweitnächste = halb, ...).
        if self.settlements:
            settlement_positions = np.array([(s.x, s.y) for s in self.settlements])
            weights_by_rank = PlotNodeSystem._rank_distance_weights(len(self.settlements))
            softening = 5.0
            for node in regular:
                pos = positions[node.node_id]
                deltas = settlement_positions - pos
                dists = np.hypot(deltas[:, 0], deltas[:, 1])
                order = np.argsort(dists)
                pull = np.zeros(2)
                for rank, idx in enumerate(order):
                    dist = max(float(dists[idx]), softening)
                    direction = deltas[idx] / max(float(dists[idx]), 1e-6)
                    pull += direction * weights_by_rank[rank] * self.plot_gravity_strength * (50.0 / dist)
                # Stadtmauer-Gegenkraft (Regel 1): waechst als Potenz der Naehe,
                # dominiert die Gravitation deutlich vor Erreichen von etwa
                # plot_base_spacing Abstand zur Stadt - unabhaengig davon, wie
                # hoch Gravity Strength eingestellt ist.
                for idx in range(len(self.settlements)):
                    dist = max(float(dists[idx]), 1e-6)
                    ratio = min(self.plot_base_spacing / dist, 6.0)
                    direction = deltas[idx] / dist
                    pull -= direction * self.plot_city_repulsion_strength * ratio ** 3
                displacement[node.node_id] += pull

        height, width = self.heightmap.shape
        for node in regular:
            new_pos = positions[node.node_id] + displacement[node.node_id]
            new_pos[0] = np.clip(new_pos[0], 0, width - 1)
            new_pos[1] = np.clip(new_pos[1], 0, height - 1)
            node.node_location = (float(new_pos[0]), float(new_pos[1]))

    def _edge_key(self, p1, p2):
        """Grob-stabile Kanten-Identitaet ueber Ticks hinweg (Voronoi-Vertex-
        Indizes sind zwischen zwei Berechnungen NICHT stabil, da sich die
        Plotkern-Positionen jeden Tick leicht verschieben) - gerundete
        Kanten-Mittelpunkt-Position reicht fuer die "grob"-Anforderung des
        Traffic-Abklingens (Regel 8)."""
        mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        return (round(mx), round(my))

    def _simulate_traffic(self):
        """
        Baut das Wege-Netz (Voronoi-Kanten zwischen den Plotnode-Kreuzungen
        von Plotkernen+Grenzknoten) und simuliert zwei Traffic-Quellen:
        1) Plotkern-Traffic (fester traffic_weight 3-5, Regel 7), rang-
           distanz-verteilt auf alle über Grenzknoten erreichbaren Städte.
        2) Intercity-Traffic (plot_intercity_traffic, Regel 5), rang-distanz-
           verteilt von jeder Stadt zu allen anderen.
        Traffic ist über Ticks persistent mit 50%-Abklingen pro Tick
        (Regel 8) statt bei jedem Tick komplett neu berechnet zu werden.
        """
        points = np.array([n.node_location for n in self.nodes])
        if len(points) < 4:
            self.ridge_vertex_positions = None
            self.current_ridge_edges = []
            return
        try:
            vor = Voronoi(points)
        except Exception:
            self.ridge_vertex_positions = None
            self.current_ridge_edges = []
            return

        vertex_positions = vor.vertices
        num_vertices = len(vertex_positions)

        def avg_slope(p1, p2):
            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if length < 1e-6:
                return 0.0
            steps = max(1, int(round(length)))
            xs = np.linspace(p1[0], p2[0], steps + 1)
            ys = np.linspace(p1[1], p2[1], steps + 1)
            h, w = self.heightmap.shape
            ix = np.clip(xs.round().astype(int), 0, w - 1)
            iy = np.clip(ys.round().astype(int), 0, h - 1)
            heights = self.heightmap[iy, ix]
            return float(np.sum(np.abs(np.diff(heights)))) / length

        ridge_edges = []  # (i, j, p1, p2, cost)
        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue
            i, j = ridge
            p1, p2 = vertex_positions[i], vertex_positions[j]
            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if length < 1e-6:
                continue
            cost = length * (1.0 + self.plot_height_cost_factor * avg_slope(p1, p2))
            ridge_edges.append((i, j, p1, p2, cost))

        self.current_ridge_edges = ridge_edges
        self.ridge_vertex_positions = vertex_positions

        if not ridge_edges:
            self.ridge_traffic_history = {}
            return

        # Eingangskante jedes Nodes (Plotkern ODER Grenzknoten) zu einem
        # zufälligen eigenen Voronoi-Vertex (Nutzer-Vorgabe: "in welchem ist
        # erstmal zufällig") - nur bei unbeschränkten Randzellen bleibt der
        # nächstgelegene Vertex als Fallback.
        tree = cKDTree(vertex_positions)
        node_entry = {}
        entry_edges = []
        for idx, node in enumerate(self.nodes):
            region_idx = vor.point_region[idx]
            region = vor.regions[region_idx]
            finite_vertices = [v for v in region if v >= 0]
            if finite_vertices:
                vertex_idx = random_mod.choice(finite_vertices)
                dist = float(np.hypot(*(vertex_positions[vertex_idx] - np.array(node.node_location))))
            else:
                dist, vertex_idx = tree.query(node.node_location)
                vertex_idx = int(vertex_idx)
            entry_idx = num_vertices + idx
            node_entry[node.node_id] = entry_idx
            entry_edges.append((entry_idx, vertex_idx, max(float(dist), 1e-3)))

        total_graph_nodes = num_vertices + len(self.nodes)
        rows, cols, costs = [], [], []
        for i, j, p1, p2, cost in ridge_edges:
            rows += [i, j]
            cols += [j, i]
            costs += [cost, cost]
        for entry_idx, vertex_idx, cost in entry_edges:
            rows += [entry_idx, vertex_idx]
            cols += [vertex_idx, entry_idx]
            costs += [cost, cost]
        graph = csr_matrix((costs, (rows, cols)), shape=(total_graph_nodes, total_graph_nodes))

        boundary_entries = [node_entry[nid] for nid in self.boundary_owner if nid in node_entry]
        boundary_settlement = [self.boundary_owner[nid] for nid in self.boundary_owner if nid in node_entry]
        if not boundary_entries:
            self.ridge_traffic_history = {}
            return

        distances, predecessors = dijkstra(graph, indices=boundary_entries, return_predecessors=True)

        def trace_and_add(pred_row, source_entry, target_entry, amount, contrib):
            current = target_entry
            while current != source_entry and current >= 0:
                prev = pred_row[current]
                if prev < 0:
                    break
                if current < num_vertices and prev < num_vertices:
                    key = self._edge_key(vertex_positions[current], vertex_positions[prev])
                    contrib[key] = contrib.get(key, 0.0) + amount
                current = prev

        fresh_contrib = {}

        # --- Plotkern-Traffic: fester traffic_weight (3-5), rang-distanz auf alle Städte ---
        for node in self.nodes:
            if node.node_id in self.boundary_owner:
                continue
            entry_idx = node_entry[node.node_id]
            node_distances = distances[:, entry_idx]
            per_settlement_best = {}
            for row, sid in enumerate(boundary_settlement):
                d = node_distances[row]
                if not np.isfinite(d):
                    continue
                if sid not in per_settlement_best or d < per_settlement_best[sid][0]:
                    per_settlement_best[sid] = (d, row)
            if not per_settlement_best:
                continue
            ranked = sorted(per_settlement_best.items(), key=lambda kv: kv[1][0])
            weights = PlotNodeSystem._rank_distance_weights(len(ranked))
            traffic_weight = getattr(node, "traffic_weight", 4.0)
            for rank, (sid, (d, row)) in enumerate(ranked):
                amount = weights[rank] * traffic_weight
                trace_and_add(predecessors[row], boundary_entries[row], entry_idx, amount, fresh_contrib)

        # --- Intercity-Traffic: jede Stadt schickt plot_intercity_traffic zu
        # den anderen, rang-distanz-gewichtet (Regel 5) ---
        settlement_ids = [s.location_id for s in self.settlements]
        if len(settlement_ids) > 1:
            for sid_from in settlement_ids:
                own_rows = [r for r, sid in enumerate(boundary_settlement) if sid == sid_from]
                if not own_rows:
                    continue
                other_best = {}
                for sid_to in settlement_ids:
                    if sid_to == sid_from:
                        continue
                    cols_to = [c for c, sid in enumerate(boundary_settlement) if sid == sid_to]
                    best = None
                    for r_from in own_rows:
                        for c_to in cols_to:
                            target_entry = boundary_entries[c_to]
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

        new_history = {}
        for i, j, p1, p2, cost in ridge_edges:
            key = self._edge_key(p1, p2)
            old = self.ridge_traffic_history.get(key, 0.0)
            new_history[key] = 0.5 * old + fresh_contrib.get(key, 0.0)
        self.ridge_traffic_history = new_history

    def _tick(self):
        if not self.playing:
            return
        self.iteration += 1
        self._physics_step()
        self._simulate_traffic()
        self._redraw()

    # ------------------------------------------------------------- Zeichnen --
    def _wiggly_path(self, p1, p2):
        """
        Spline-artige, wellige Verbindung statt einer Geraden - Amplitude
        (Kurvigkeit-Slider) skaliert mit der mittleren Steigung entlang der
        Strecke, Wellenfrequenz separat über den Detailgrad-Slider (Regel 8).
        Flaches Terrain bleibt auch bei hohem Kurvigkeit-Wert nahezu gerade.
        Gilt NUR für das Wege-Netz - Grundstücksgrenzen bleiben immer gerade
        (Regel 4, siehe current_ridge_edges-Rendering in _redraw()).
        """
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

        normalized_slope = min(1.0, avg_slope / 30.0)  # empirisch fürs Demo-Tool skaliert
        wiggle_fraction = max_fraction * normalized_slope
        amplitude = wiggle_fraction * length
        if amplitude < 0.5:
            return xs_straight, ys_straight

        dx, dy = (x2 - x1) / length, (y2 - y1) / length
        perp_x, perp_y = -dy, dx
        envelope = np.sin(t * np.pi)  # 0 an beiden Enden - Endpunkte bleiben fix
        wave = np.sin(t * np.pi * self.spline_detail)
        offset = amplitude * envelope * wave
        return xs_straight + perp_x * offset, ys_straight + perp_y * offset

    def _classify(self, traffic):
        path_t = 20.0 * self.plot_tier_factor
        weg_t = 40.0 * self.plot_tier_factor
        strasse_t = 60.0 * self.plot_tier_factor
        if traffic >= strasse_t:
            return "red", 2.6
        elif traffic >= weg_t:
            return "orange", 1.6
        elif traffic >= path_t:
            return "goldenrod", 0.9
        return "lightgray", 0.35

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.heightmap, cmap="terrain", origin="lower", alpha=0.5,
                       extent=(0, self.map_size, 0, self.map_size))

        if self.show_civ_overlay:
            self.ax.imshow(self.civ_map, cmap="hot", origin="lower", alpha=0.5, vmin=0.0, vmax=1.0,
                           extent=(0, self.map_size, 0, self.map_size), zorder=1)

        inside = (self.city_mask >= 0).astype(float)
        if np.any(inside):
            self.ax.contour(inside, levels=[0.5], colors="gold", linewidths=2.0,
                             extent=(0, self.map_size, 0, self.map_size))

        # Grundstücks-Polygone: immer gerade Voronoi-Kanten, keine Splines (Regel 4)
        polygon_segments = [(p1, p2) for (i, j, p1, p2, cost) in self.current_ridge_edges]
        if polygon_segments:
            self.ax.add_collection(LineCollection(
                polygon_segments, colors="dimgray", linewidths=0.5, alpha=0.5, zorder=2))

        strasse_count = weg_count = pfad_count = 0
        total_traffic = 0.0
        for i, j, p1, p2, cost in self.current_ridge_edges:
            key = self._edge_key(p1, p2)
            traffic = self.ridge_traffic_history.get(key, 0.0)
            color, width = self._classify(traffic)
            if color != "lightgray":
                total_traffic += traffic
                strasse_count += color == "red"
                weg_count += color == "orange"
                pfad_count += color == "goldenrod"
            xs, ys = self._wiggly_path(tuple(p1), tuple(p2))
            self.ax.plot(xs, ys, color=color, linewidth=width, zorder=4)

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
        self.status_label.setText(f"Iteration: {self.iteration}")

        self.canvas.draw_idle()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = PlotPhysicsLab()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
