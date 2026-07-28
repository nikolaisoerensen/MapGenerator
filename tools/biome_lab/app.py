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
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer

import matplotlib
matplotlib.use("QtAgg")

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
    # Deutlich kleiner als WILDERNESS_MIN_AREA: Staedte koennen legitim nur
    # 30-50px Flaeche haben (city_reach_factor bei niedrigem city_size), ein
    # mit WILDERNESS_MIN_AREA geteiltes Filterkriterium warf solche echten,
    # kleinen Stadtkonturen komplett weg -- die betroffene Siedlung bekam
    # dadurch nie Stadttore (siehe scene._build_city_boundary_polygons).
    CITY_MIN_AREA = 5.0
    SOFTENING = 5.0
    PHYSICS_TIME_STEP = 0.25
    MAX_SPRING_RESULTANT = 4.0
    MAX_DISPLACEMENT_PER_TICK = 2.0  # nur noch Geschwindigkeits-Deckel (siehe physics._physics_step: max_speed)
    # Kartenrand-Inset fuer die Voronoi-Klip-Box (siehe topology._build_voronoi_mesh)
    # UND fuer die Seed-Punkt-Ausschlusszone (siehe _gen_step_2_plot_cores) --
    # EIN gemeinsamer Wert seit dem Umbau auf ein einziges geklipptes Voronoi:
    # Kartenrand-Nodes entstehen jetzt direkt am Klip-Rand, es gibt keinen
    # getrennt generierten Kartenrand-Ring mehr, der einen eigenen (frueher
    # engeren) Abstand gebraucht haette.
    PLOT_CORE_EDGE_MARGIN = 5.0

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
        # Bewusst False: der Nutzer soll die Generierung Schritt fuer Schritt
        # durchklicken koennen (siehe _gen_step_*/_advance_generation_step in
        # topology.py), statt dass die Karte sofort komplett generiert UND
        # die Physik sofort zu ticken beginnt.
        self.playing = False
        self._pending_generation_steps = []
        self._generation_step_index = 0
        self.next_node_id = 0
        self.boundary_owner = {}
        self.ridge_traffic_history = {}
        self.ridge_traffic_shrink_ema = {}
        self.current_ridge_edges = []
        self.ridge_vertex_positions = None
        self.potential_field = None

        self.settlement_border_margin = 25.0

        # -- Basiswerte mit physikalischer Einheit (px), bleiben unnormiert --
        self.plot_base_spacing = 60.0

        # -- Logarithmische Regler: 1.0 = neutral (kein Effekt auf Basiswert),
        #    5.0 = 5x staerker, 0.2 = 5x schwaecher, usw. (Basis 5 statt Basis 10) --
        self.norm_civ_spacing_factor = 1.0
        # norm_spacing_spring_stiffness/norm_traffic_spring_stiffness: physikalischer
        # Rebuild -- keine Feder hat mehr rest_length=0 (Core<->PlotNode/PlotNode<->
        # PlotNode haben jetzt echte, civ-abhaengige Ruhelaengen > 0, siehe
        # topology._spring_rest_length), daher braucht es keinen kuenstlich erhoehten
        # Default mehr wie in der vorherigen (Zentroid-Feder-)Version. Start bewusst
        # "steif" (siehe Nutzer-Vorgabe): hohe Basis-Steifigkeit + hohe Daempfung,
        # zum Testen schrittweise weicher stellen.
        self.norm_spacing_spring_stiffness = 1.0   # Feder: Abstand -> Core<->PlotNode (a)
        self.norm_traffic_spring_stiffness = 1.0   # Feder: Traffic-Zug -> PlotNode<->PlotNode (b)
        self.norm_pressure_strength = 1.0          # Druck: Flaechenerhalt (f)
        self.norm_core_mass = 1.0                  # Masse: Kern (d)
        self.norm_plot_node_mass = 1.0             # Masse: PlotNode (d)
        self.norm_plot_node_repulsion_strength = 1.0  # PlotNode-Kollisionsabstossung (Abschnitt D)
        self.norm_gravity_strength = 1.0
        self.norm_city_repulsion_strength = 1.0
        self.norm_height_cost_factor = 1.0
        self.norm_tier_factor = 1.0
        self.norm_potential_strength = 1.0

        # Daempfung (d): multiplikativer Geschwindigkeits-Verlust pro Tick
        # (0=sofortiger Stillstand, 1=keine Daempfung/reine Energieerhaltung).
        # Bewusst hoch/"steif" als Startwert -- global gilt: erst steif und
        # stabil, dann schrittweise weicher testen.
        self.damping = 0.80

        # Ein/Aus-Schalter je Kraft/Mechanismus (Checkboxen, siehe ui.py):
        # bewusst ALLE False als Default, damit man mit einer leeren Szene
        # (keine Kraft aktiv, nichts bewegt sich) startet und Kraft fuer Kraft
        # dazuschalten kann, um genau zu isolieren, welche einen Kollaps
        # verursacht -- statt das per Kopfrechnen/Skript zu vermuten.
        self.enable_core_plotnode_spring = False       # Feder (a)
        self.enable_plotnode_plotnode_spring = False   # Feder (b)
        self.enable_pressure = False                   # Druck (f)
        self.enable_plot_node_repulsion = False        # Kollisionsabstossung (Abschnitt D)
        self.enable_field_cores = False                # Potentialfeld auf Kerne (c)
        self.enable_field_plotnodes = False             # Potentialfeld auf PlotNodes
        self.enable_core_cell_containment = False      # harte Zellgrenze (Kerne)
        self.enable_wilderness_containment = False     # harte Wildnisgrenze (PlotNodes)

        # -- Basiswerte, auf die die Normierung bei Multiplikator 1.0 abbildet --
        self._BASE_CIV_SPACING_FACTOR = 8.0
        self._BASE_SPACING_SPRING_STIFFNESS = 1.2
        self._BASE_TRAFFIC_SPRING_STIFFNESS = 1.0
        self._BASE_PRESSURE_STRENGTH = 0.8
        self._BASE_CORE_MASS = 1.0
        self._BASE_PLOT_NODE_MASS = 1.0
        self._BASE_PLOT_NODE_REPULSION_STRENGTH = 4.0
        self._BASE_GRAVITY_STRENGTH = 0.01
        self._BASE_CITY_REPULSION_STRENGTH = 0.5
        self._BASE_HEIGHT_COST_FACTOR = 3.0
        self._BASE_TIER_FACTOR = 1.0

        self.spring_traffic_shrink = 0.002
        # Ruhelaenge der PlotNode<->PlotNode-Feder (b) schrumpft mit Traffic
        # bis zu 30% (Nutzer-Vorgabe) -- Bodenwert 0.70, nicht 1.0.
        self.spring_min_shrink_fraction = 0.70
        # Eigene, deutlich langsamer nachziehende EMA-Rate nur fuer den
        # Feder-Schrumpf (siehe ridge_traffic_shrink_ema in traffic.py) --
        # bewusst sanfter als traffic_decay=0.15 in traffic.py, damit die
        # Ruhelaenge beim ersten Auftreten von Traffic auf einer Kante nicht
        # in einem Schritt springt, sondern langsam eingleitet.
        self.spring_shrink_ema_decay = 0.05

        # Weiche Rueckstellkraft fuer die Wildnisgrenze (PlotNodes), ersetzt
        # den frueheren harten Positions-Snap in _physics_step: der Snap war
        # eine Viele-zu-eins-Abbildung (mehrere plot_nodes vom selben
        # Randabschnitt projizieren auf denselben Punkt) und liess plot_nodes
        # sichtbar zu Haufen/Linien entlang der Grenze kollabieren, siehe
        # physics._apply_spring_forces Abschnitt E.
        self.wilderness_push_stiffness = 1.5

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
    exit_code = app.exec()
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
