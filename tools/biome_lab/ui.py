"""
Path: tools/biome_lab/ui.py

Qt-Bedienpanel: Slider/Checkboxen/Buttons und die daraus abgeleiteten,
normierten Physik-Parameter (Properties wie plot_gravity_strength). "Live"-
Slider loesen sofort eine Potentialfeld-Neuberechnung aus (siehe field.py),
die "Hintergrund"-Slider wirken erst nach einem Klick auf
"Reset Plot Nodes" (siehe topology.py::_reset_plot_nodes).
"""
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QCheckBox, QGroupBox, QFormLayout, QSizePolicy, QScrollArea, QPlainTextEdit
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=3)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # QScrollArea statt Panel direkt ins Layout: die Anzahl der Regler/
        # Checkboxen ist mittlerweile zu gross fuer eine feste Fensterhoehe --
        # ohne Scroll-Wrapper quetscht Qt den Inhalt (ueberlappender Text)
        # statt ihn abzuschneiden/scrollbar zu machen.
        panel = QWidget()
        panel.setFixedWidth(450)
        panel_layout = QVBoxLayout(panel)

        scroll_area = QScrollArea()
        scroll_area.setWidget(panel)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFixedWidth(470)
        layout.addWidget(scroll_area, stretch=0)

        self.status_label = QLabel("Iteration: 0")
        panel_layout.addWidget(self.status_label)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        panel_layout.addWidget(self.play_button)

        regen_button = QPushButton("Regenerate Terrain + Settlements")
        # Bug-Fix: QPushButton.clicked emittiert IMMER ein bool "checked"-Arg.
        # Direkt verbunden (`.connect(self._regenerate_scene)`) landet dieses
        # bool in `seed` (scene._regenerate_scene(self, seed=None)), IMMER
        # als False -- "if seed is None" greift dann nie, jeder Klick nutzt
        # map_seed=False (==0), die Karte war deshalb nie wirklich random.
        # Lambda kappt das Signal-Argument, damit seed=None (echtes Zufalls-
        # Seed via random_mod.randint) tatsaechlich ankommt.
        regen_button.clicked.connect(lambda: self._regenerate_scene())
        panel_layout.addWidget(regen_button)

        reset_button = QPushButton("Reset Plot Nodes (übernimmt Hintergrund-Slider)")
        reset_button.clicked.connect(self._reset_plot_nodes)
        panel_layout.addWidget(reset_button)

        # Schritt-Log: zeigt die Diagnostik-Ausgaben der 8 Generierungs-
        # Schritte (siehe topology._gen_step_1.._gen_step_8/_log_step) direkt
        # in der App, damit sich der Schritt, der einen Fehler einfuehrt,
        # ohne Logdatei-Wechsel eingrenzen laesst.
        step_log_label = QLabel("Generierungs-Schritte (Play = naechster Schritt, solange Regenerate/Reset noch nicht durchgelaufen ist):")
        step_log_label.setWordWrap(True)
        panel_layout.addWidget(step_log_label)
        self.step_log_widget = QPlainTextEdit()
        self.step_log_widget.setReadOnly(True)
        self.step_log_widget.setFixedHeight(160)
        panel_layout.addWidget(self.step_log_widget)

        forces_group = QGroupBox("Kräfte an/aus (zum Isolieren einzeln zuschaltbar, alle Default AUS)")
        forces_form = QVBoxLayout()
        self._add_force_checkbox(forces_form, "enable_core_plotnode_spring", "Feder (a): Core↔PlotNode")
        self._add_force_checkbox(forces_form, "enable_plotnode_plotnode_spring", "Feder (b): PlotNode↔PlotNode")
        self._add_force_checkbox(forces_form, "enable_pressure", "Druck (f): Flächenerhalt")
        self._add_force_checkbox(forces_form, "enable_plot_node_repulsion", "Kollisionsabstoßung PlotNodes")
        self._add_force_checkbox(forces_form, "enable_field_cores", "Potentialfeld auf Kerne (c)")
        self._add_force_checkbox(forces_form, "enable_field_plotnodes", "Potentialfeld auf PlotNodes")
        self._add_force_checkbox(forces_form, "enable_core_cell_containment", "Harte Zellgrenze (Kerne)")
        self._add_force_checkbox(forces_form, "enable_wilderness_containment", "Wildnisgrenze (PlotNodes, weiche Feder)")
        forces_group.setLayout(forces_form)
        panel_layout.addWidget(forces_group)

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
        self._add_log_slider(live_form, "norm_pressure_strength", "Druck: Flächenerhalt (0.04x-25x)",
                              default=self.norm_pressure_strength)
        self._add_log_slider(live_form, "norm_core_mass", "Masse: Kern (0.04x-25x)",
                              default=self.norm_core_mass)
        self._add_log_slider(live_form, "norm_plot_node_mass", "Masse: PlotNode (0.04x-25x)",
                              default=self.norm_plot_node_mass)
        self._add_log_slider(live_form, "norm_plot_node_repulsion_strength", "PlotNode-Kollisionsabstand (0.04x-25x)",
                              default=self.norm_plot_node_repulsion_strength)
        self._add_slider(live_form, "damping", "Dämpfung (0=sofort still, 1=keine)", 0.0, 1.0, self.damping,
                          scale=100)
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

        slider = QSlider(Qt.Orientation.Horizontal)
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
        slider = QSlider(Qt.Orientation.Horizontal)
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

    def _add_force_checkbox(self, layout, attr_name, label):
        """Checkbox fuer einen einzelnen Kraft-/Mechanismus-Schalter (siehe
        app.py enable_*-Attribute) -- direkt in ein QVBoxLayout, kein
        Label/Value-Paar wie bei den Slidern noetig."""
        checkbox = QCheckBox(label)
        checkbox.setChecked(bool(getattr(self, attr_name)))

        def on_toggle(checked):
            setattr(self, attr_name, bool(checked))

        checkbox.toggled.connect(on_toggle)
        layout.addWidget(checkbox)

    def _toggle_play(self):
        # Solange eine Schritt-fuer-Schritt-Generierung ansteht (siehe
        # topology._start_step_through_generation), wird der Play-Button
        # umgewidmet: ein Klick fuehrt genau den naechsten der 8 Schritte
        # aus, statt die Physik-Ticks zu (de-)pausieren -- erst nach dem
        # letzten Schritt wirkt der Button wieder normal als Play/Pause.
        pending = getattr(self, "_pending_generation_steps", None) or []
        step_idx = getattr(self, "_generation_step_index", 0)
        if step_idx < len(pending):
            self._advance_generation_step()
            self._update_play_button_label()
            return

        self.playing = not self.playing
        self._update_play_button_label()

    def _update_play_button_label(self):
        """Zeigt entweder 'Naechster Schritt (N/8)' (waehrend eine
        Schritt-fuer-Schritt-Generierung ansteht) oder normal Play/Pause."""
        pending = getattr(self, "_pending_generation_steps", None) or []
        step_idx = getattr(self, "_generation_step_index", 0)
        if step_idx < len(pending):
            self.play_button.setText(f"Nächster Schritt ({step_idx + 1}/{len(pending)})")
        else:
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
    def core_plotnode_spring_stiffness(self):
        """Feder: Abstand -- Core<->PlotNode-Federn (a), civ-abhaengige Ruhelaenge."""
        return self._n("norm_spacing_spring_stiffness", self._BASE_SPACING_SPRING_STIFFNESS)

    @property
    def plotnode_plotnode_spring_stiffness(self):
        """Feder: Traffic-Zug -- PlotNode<->PlotNode-Federn (b), civ+traffic-
        abhaengige Ruhelaenge."""
        return self._n("norm_traffic_spring_stiffness", self._BASE_TRAFFIC_SPRING_STIFFNESS)

    @property
    def pressure_strength(self):
        """Druck: Flaechenerhalt -- Innendruck je Kern-Zelle (f), nur nach aussen."""
        return self._n("norm_pressure_strength", self._BASE_PRESSURE_STRENGTH)

    @property
    def core_mass(self):
        return self._n("norm_core_mass", self._BASE_CORE_MASS)

    @property
    def plot_node_mass(self):
        return self._n("norm_plot_node_mass", self._BASE_PLOT_NODE_MASS)

    @property
    def plot_node_repulsion_strength(self):
        """Kurzreichweitige Kollisionsabstossung zwischen NAHEN plot_nodes,
        unabhaengig von direkter Netz-Adjazenz (Abschnitt D)."""
        return self._n("norm_plot_node_repulsion_strength", self._BASE_PLOT_NODE_REPULSION_STRENGTH)

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
