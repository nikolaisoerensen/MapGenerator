"""
Path: tools/biome_lab/ui.py

Qt-Bedienpanel: Slider/Checkboxen/Buttons und die daraus abgeleiteten,
normierten Physik-Parameter (Properties wie plot_gravity_strength). "Live"-
Slider loesen sofort eine Potentialfeld-Neuberechnung aus (siehe field.py),
die "Hintergrund"-Slider wirken erst nach einem Klick auf
"Reset Plot Nodes" (siehe topology.py::_reset_plot_nodes).
"""
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QCheckBox, QGroupBox, QFormLayout, QSizePolicy
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class UIMixin:
    """Bedienpanel-Aufbau + normierte Slider-Properties. Wird von
    PlotPhysicsLab (app.py) zusammen mit den uebrigen Mixins eingebunden."""

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=3)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

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
                          0, 70, self.spline_wiggle_pct, scale=1, invalidate_wiggle_cache=True)
        self._add_slider(way_form, "spline_detail", "Detailgrad (Wellenfrequenz)",
                          1, 6, self.spline_detail, scale=10, invalidate_wiggle_cache=True)
        way_group.setLayout(way_form)
        panel_layout.addWidget(way_group)

        bg_group = QGroupBox("Hintergrund (Klick auf 'Reset Plot Nodes' nötig)")
        bg_form = QFormLayout()
        self._add_slider(bg_form, "plot_nodes_count", "Plot Nodes (Ziel)", 20, 800, self.plot_nodes_count, scale=1)
        self._add_slider(bg_form, "city_size", "Stadtgröße", 0, 1.0, self.city_size, scale=100)
        self._add_slider(bg_form, "civ_influence_decay", "Civ Influence Decay", 0.1, 2, self.civ_influence_decay,
                          scale=100)
        self._add_slider(bg_form, "wilderness_core_spacing", "Wildniskern-Abstand (px)", 20, 200,
                          self.wilderness_core_spacing, scale=1)
        bg_group.setLayout(bg_form)
        panel_layout.addWidget(bg_group)

        legend = QLabel(
            "Wege-Linien: YlOrRd-Farbkarte (gelb-orange-rot), Breite 0.3-3.0 proportional zum Traffic\n"
            "Hellgrau = Traffic unter Schwelle (<20×Faktor)\n"
            "Gold-Kontur = Stadtgrenze (Civ-Wert-Kontour)\n"
            "Sterne = Settlement/Marktplatz\n"
            "Rote Punkte = Plotkern (best-candidate verteilt, erzeugt konstanten Traffic 3-5)\n"
            "Grüne Punkte = Wildniskern (keine Feder-/Feldphysik, sitzt mittig in seiner Zelle,\n"
            "  erzeugt wenig Traffic 1-2 -> Wege durch die Wildnis)\n"
            "Blaue Punkte = Plotnodes (Voronoi-Kreuzungen, ziv-nahe an Wildnisgrenze geklebt)\n"
            "Dünne graue Linien = Grundstücksgrenzen (immer gerade)\n"
            "Gold-Dreiecke = Stadtgrenz-Nodes (gleiten auf Stadtkontur)\n"
            "Cyan-Kreise = Wildnisgrenz-Nodes (gleiten auf civ-Kontur)\n"
            "Graue Quadrate = Kartenrand-Nodes (gleiten auf Rand-Rechteck)\n"
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

    def _add_slider(self, form, attr_name, label, min_val, max_val, default, scale, live=False,
                     invalidate_wiggle_cache=False):
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
            if invalidate_wiggle_cache:
                self._wiggly_path_cache = {}

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
        self._mark_overlay_dirty()
        self._redraw()

    def _set_potential_overlay(self, checked):
        self.show_potential_overlay = checked
        self._mark_overlay_dirty()
        self._redraw()

    # -------------------------------------------- Normierte Physik-Parameter --
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
