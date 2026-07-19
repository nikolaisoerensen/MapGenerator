"""
Path: tools/biome_lab/topology.py

Aufbau des statischen Wege-Netzes ("was am Anfang passiert"): Plotkern-
Sampling (best-candidate), Sonder-Nodes an Stadt-/Wildnis-/Kartengrenze,
Voronoi-Triangulation der Plotkerne (in einem isolierten Worker-Prozess,
siehe unten), Dijkstra-Distanzgraph fuer die Traffic-Simulation
(traffic.py) sowie die Node-/Core-Registries, die physics.py und draw.py
lesen.

Voronoi-Isolation: scipy.spatial.Voronoi (Qhull) kann bei degenerierten
Punktmengen nativ abstuerzen (Windows STATUS_STACK_BUFFER_OVERRUN). Damit
so ein Absturz nicht die ganze Qt-App mitreisst, laeuft die eigentliche
Voronoi-Berechnung dauerhaft in einem separaten Worker-Prozess; stirbt er,
wird er transparent neu gestartet (_ensure_voronoi_worker).
"""
import random as random_mod

import numpy as np
from scipy.spatial import Voronoi, cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box, LineString, Point
import multiprocessing as mp
import os

from .logging_setup import logger, LOG_DIR
from .models import PlotNode, PlotCore


def _voronoi_worker_loop(conn):
    """Laeuft dauerhaft in einem separaten Prozess (einmalig gestartet).
    Wartet in einer Schleife auf Punktmengen und sendet Ergebnisse zurueck.
    Ein nativer Qhull-Crash (Windows STATUS_STACK_BUFFER_OVERRUN) beendet
    nur diesen einen Prozess -- die Haupt-App bleibt am Leben und kann
    per _ensure_voronoi_worker() einen neuen Worker nachstarten."""
    from scipy.spatial import Voronoi as _Voronoi
    while True:
        try:
            points = conn.recv()
        except (EOFError, OSError):
            break
        if points is None:  # Shutdown-Signal
            break
        try:
            vor = _Voronoi(points)
            conn.send({
                "vertices": vor.vertices,
                "ridge_points": vor.ridge_points,
                "ridge_vertices": vor.ridge_vertices,
                "point_region": vor.point_region,
                "regions": vor.regions,
            })
        except Exception as e:
            conn.send({"error": str(e)})


def nearest_point_on_polyline(pos, polyline, closed=True):
    """Vektorisierte Projektion eines Punkts auf die naechstgelegene Stelle
    einer Polylinie (offen oder geschlossen).

    Numerisch identisch zur frueheren Python-Schleifen-Implementierung
    (ein Segment nach dem anderen, np.dot/np.clip pro Segment), aber als
    einzige numpy-Vektor-Operation ueber ALLE Segmente gleichzeitig statt
    einer Python-for-Schleife. Das war der dominante Hotspot im Live-Tick:
    Wildnis-/Kartenrand-/Stadtgrenz-Nodes rufen diese Funktion jeden Tick
    auf, und Wildnis-Konturen aus Marching-Squares koennen mehrere hundert
    Segmente haben -- die alte Python-Schleife brauchte dafuer >300ms/Tick,
    die vektorisierte Variante <1ms."""
    pos = np.asarray(pos, dtype=float)
    poly = np.asarray(polyline, dtype=float)
    n = len(poly)

    if n == 0:
        return pos
    if n == 1:
        return poly[0].copy()

    if closed:
        a = poly
        b = np.roll(poly, -1, axis=0)
    else:
        a = poly[:-1]
        b = poly[1:]

    ab = b - a
    ab_len_sq = np.einsum("ij,ij->i", ab, ab)
    safe_len_sq = np.where(ab_len_sq > 1e-12, ab_len_sq, 1.0)
    t = np.einsum("ij,ij->i", pos - a, ab) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    candidates = a + ab * t[:, None]
    degenerate = ab_len_sq <= 1e-12
    if np.any(degenerate):
        candidates[degenerate] = a[degenerate]

    dists_sq = np.einsum("ij,ij->i", candidates - pos, candidates - pos)
    best = int(np.argmin(dists_sq))
    return candidates[best]


def nearest_point_on_segments(pos, a, b):
    """Wie nearest_point_on_polyline, nimmt aber bereits in Start-/End-Punkte
    aufgeteilte Segment-Arrays (a, b) entgegen. Erspart den np.roll-Aufbau bei
    wiederholten Aufrufen auf DERSELBEN, zwischen Ticks unveraenderten
    Polylinie (z.B. eine Wildnisgrenzkontur) -- np.roll allein war in Profilen
    ein spuerbarer Anteil der physics._contain_plot_node-Kosten, weil dort pro
    Tick potenziell hunderte plot_nodes gegen dieselbe(n) Kontur(en) pruefen."""
    pos = np.asarray(pos, dtype=float)
    ab = b - a
    ab_len_sq = np.einsum("ij,ij->i", ab, ab)
    safe_len_sq = np.where(ab_len_sq > 1e-12, ab_len_sq, 1.0)
    t = np.einsum("ij,ij->i", pos - a, ab) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    candidates = a + ab * t[:, None]
    degenerate = ab_len_sq <= 1e-12
    if np.any(degenerate):
        candidates[degenerate] = a[degenerate]

    dists_sq = np.einsum("ij,ij->i", candidates - pos, candidates - pos)
    best = int(np.argmin(dists_sq))
    return candidates[best]


def extract_intersection_points(geom):
    """Zieht alle (x,y)-Punkte aus einem beliebigen shapely-Intersections-
    Ergebnis (Point/MultiPoint/LineString/MultiLineString/GeometryCollection).
    Bewusst per explizitem geom_type-Dispatch statt `hasattr(geom, "coords")`
    -- die `coords`-Property EXISTIERT auf jeder shapely-Basisklasse, wirft
    aber erst beim tatsaechlichen Zugriff ein NotImplementedError fuer
    Multi-Part-Geometrien (z.B. MultiLineString bei einer seltenen
    kollinearen Ueberlappung zwischen Ridge-Kante und Konturkante) --
    `hasattr` faengt diese Exception NICHT ab (sie kommt erst beim
    Property-Zugriff, nicht bei der Existenzpruefung), also crasht der
    naive Ansatz genau in diesem Randfall."""
    gt = geom.geom_type
    if gt == "Point":
        return [(geom.x, geom.y)]
    if gt == "MultiPoint":
        return [(g.x, g.y) for g in geom.geoms]
    if gt in ("LineString", "LinearRing"):
        return list(geom.coords)
    if gt in ("MultiLineString", "GeometryCollection"):
        points = []
        for g in geom.geoms:
            points.extend(extract_intersection_points(g))
        return points
    return []


def polygon_area(vertices):
    """Flaeche eines (nicht notwendigerweise konvexen, aber einfachen)
    Polygons ueber die Gausssche Trapezformel/Shoelace-Formel. Wird vom
    Innendruck-Term (physics._apply_spring_forces) benutzt: ein Kern soll
    seiner Zell-Flaeche entgegen Kompression eine Rueckstellkraft entgegen-
    setzen, dafuer muss die AKTUELLE Flaeche bekannt sein."""
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    return float(abs(np.sum(x * y2 - x2 * y)) * 0.5)


class TopologyMixin:
    """Statischer Aufbau von Plotkernen, Sonder-Nodes, Voronoi-Wege-Netz und
    Traffic-Distanzgraph. Wird von PlotPhysicsLab (app.py) zusammen mit den
    uebrigen Mixins eingebunden."""

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
        bleibt. (Aktuell ungenutzt -- die synchrone Voronoi-Berechnung in
        _build_voronoi_mesh() laeuft im Hauptprozess; dieser Pfad steht als
        Absturz-sichere Alternative bereit.)"""
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

    # ------------------------------------------------------- Plotkern-Setup --
    def _best_candidate_sample(self, valid_mask, count, base_spacing, civ_spacing_factor, civ_map, k=15):
        """Mitchell's-Best-Candidate-Sampling: pro Plotkern werden k
        Zufallskandidaten gezogen, der Kandidat mit dem groessten Abstand zu
        bereits gewaehlten Punkten (relativ zur civ-abhaengigen Soll-Distanz)
        gewinnt. Erzeugt eine deutlich gleichmaessigere Verteilung als
        einfaches Reject-Sampling."""
        ys, xs = np.nonzero(valid_mask)
        if len(xs) == 0:
            return []

        count = min(int(count), len(xs))
        chosen = []
        chosen_arr = np.empty((0, 2), dtype=float)
        pool_size = len(xs)

        for _ in range(count):
            idxs = np.random.randint(0, pool_size, size=min(k, pool_size))
            best_pos = None
            best_score = -np.inf

            for idx in idxs:
                x = float(xs[idx])
                y = float(ys[idx])

                civ_value = float(civ_map[int(y), int(x)])
                target_spacing = max(base_spacing * (1.0 - civ_spacing_factor * civ_value), 2.5)

                if len(chosen_arr):
                    dist = float(np.sqrt(np.min((chosen_arr[:, 0] - x) ** 2 + (chosen_arr[:, 1] - y) ** 2)))
                else:
                    dist = target_spacing * 10.0

                score = dist - target_spacing
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)

            if best_pos is None:
                idx = idxs[0]
                best_pos = (float(xs[idx]), float(ys[idx]))

            chosen.append(best_pos)
            chosen_arr = np.array(chosen, dtype=float)

        return chosen

    def _region_id_at(self, x, y):
        """Liest region_map an der (gerundeten, in Karten-Grenzen geklemmten)
        Pixel-Position (x,y). -1 falls region_map (noch) nicht existiert."""
        region_map = getattr(self, "region_map", None)
        if region_map is None:
            return -1
        px = int(np.clip(round(x), 0, self.map_size - 1))
        py = int(np.clip(round(y), 0, self.map_size - 1))
        return int(region_map[py, px])

    def _nearest_point_on_polyline(self, pos, polyline, closed=True):
        """Instanz-Wrapper um die modul-globale, vektorisierte Funktion (siehe
        oben) -- so bleibt der Aufrufstil ``self._nearest_point_on_polyline(...)``
        fuer physics.py identisch zur fruehreren Implementierung."""
        return nearest_point_on_polyline(pos, polyline, closed=closed)

    def _nearest_point_on_segments(self, pos, a, b):
        """Instanz-Wrapper um nearest_point_on_segments (siehe oben)."""
        return nearest_point_on_segments(pos, a, b)

    def _polygon_area(self, vertices):
        """Instanz-Wrapper um polygon_area (siehe oben)."""
        return polygon_area(vertices)

    # civ=1 -> 25% von base_spacing, civ=0 -> 100% von base_spacing, linear
    # dazwischen. Bewusst NICHT an den Slider plot_civ_spacing_factor
    # gekoppelt -- der steuert weiterhin nur die best-candidate-Sampling-
    # dichte (unveraendertes bestehendes Verhalten), waehrend hier die
    # exakten vom Nutzer vorgegebenen Zahlen (25%/100%) fest verdrahtet sind.
    CIV_RESTLENGTH_STEEPNESS = 0.75

    def _spring_rest_length(self, civ_value, base_spacing):
        """Zivilisationsabhaengige Ruhelaenge, gemeinsam genutzt von Core<->
        PlotNode-Federn (a), dem Innendruck-Ideal-Radius (f) und als Basis
        fuer die PlotNode<->PlotNode-Ruhelaenge (b, dort zusaetzlich per
        Traffic geschrumpft). NIE 0 -- jede Feder hat eine echte, positive
        Ruhelaenge (siehe Analyse-Notiz: rest_length=0 war die eigentliche
        Kollaps-Ursache)."""
        civ_factor = max(1.0 - self.CIV_RESTLENGTH_STEEPNESS * float(civ_value), 0.25)
        return max(float(base_spacing) * civ_factor, 2.0)

    # Schritt 2b (Seed-Relaxation): Anzahl Iterationen, Integrationsschritt
    # und Mindestabstand zu Wildnis-/Civ-Kontur bzw. Kartenrand. Bewusst
    # als feste Konstanten statt UI-Slider -- reine Vorverarbeitung, kein
    # Live-Physik-Parameter, den der Nutzer waehrend des Tickens abstimmen
    # wuerde.
    SEED_RELAX_ITERATIONS = 8
    SEED_RELAX_STEP = 0.5
    SEED_RELAX_MARGIN = 3.0
    SEED_RELAX_NEIGHBOR_COUNT = 6

    def _reset_plot_nodes(self):
        """Startet die Schritt-fuer-Schritt-Generierung (siehe
        _start_step_through_generation/_gen_step_1.._gen_step_9 unten) --
        fuehrt NUR Schritt 1 aus und wartet danach auf weitere Klicks (siehe
        _advance_generation_step, vom wiederverwendeten Play-Button
        aufgerufen, siehe ui.py._toggle_play). Name/Signatur bleiben
        unveraendert, weil der bestehende 'Reset Plot Nodes'-Button und
        scene._regenerate_scene diese Methode direkt aufrufen."""
        self._start_step_through_generation()

    def _log_step(self, message):
        """Diagnostik-Ausgabe pro Generierungs-Schritt: sowohl ins Logfile
        (logs/plot_physics_lab.log, siehe logging_setup.py) als auch direkt
        sichtbar im step_log-Panel (siehe ui.py), falls vorhanden -- damit
        der Nutzer nicht zwischen App und Logdatei wechseln muss, um die
        Ursache eines Fehlers Schritt fuer Schritt einzugrenzen."""
        logger.info(message)
        widget = getattr(self, "step_log_widget", None)
        if widget is not None:
            widget.appendPlainText(message)

    def _start_step_through_generation(self):
        """Setzt die 11-Schritte-Warteschlange auf und fuehrt sofort NUR
        Schritt 1 aus. Die restlichen 10 Schritte laufen erst durch
        wiederholte _advance_generation_step()-Aufrufe (ein Klick =
        ein Schritt), damit sich der Zustand nach JEDEM einzelnen Schritt
        visuell inspizieren laesst -- siehe Analyse-Notiz zur wiederholt
        falsch diagnostizierten Kollinearitaets-Linie: ohne Schritt-fuer-
        Schritt-Kontrolle laesst sich nicht zuverlaessig sagen, in welchem
        der elf Schritte ein Fehler tatsaechlich entsteht."""
        # Physik-Ticks IMMER pausieren, wenn eine neue Schritt-Generierung
        # beginnt -- sonst koennte der QTimer (siehe app.py) waehrend des
        # Durchklickens auf einer erst halb aufgebauten Topologie ticken.
        self.playing = False
        self._generation_step_index = 0
        self._pending_generation_steps = [
            ("Hintergrund/Regionen", self._gen_step_1_background),
            ("Seed-Punkte verteilen", self._gen_step_2_plot_cores),
            ("Seed-Punkte relaxieren", self._gen_step_2b_relax_seed_points),
            ("Stadtkerne setzen", self._gen_step_2c_place_city_cores),
            ("Voronoi (global, geklippt)", self._gen_step_4_voronoi_clipped),
            ("Stadtgrenze verteilen", self._gen_step_city_boundary_distribute),
            ("Wildnisgrenze schneiden+snappen", self._gen_step_5_wilderness_snap),
            ("Wildniskerne umklassifizieren", self._gen_step_6_wilderness_cores),
            ("Graph aufbauen (Dijkstra)", self._gen_step_7_build_graph),
            ("Konsistenzcheck", self._gen_step_8_consistency_check),
            ("Finalisierung", self._gen_step_9_finalize),
        ]
        widget = getattr(self, "step_log_widget", None)
        if widget is not None:
            widget.clear()
        self._advance_generation_step()
        update_label = getattr(self, "_update_play_button_label", None)
        if update_label is not None:
            update_label()

    def _advance_generation_step(self):
        """Fuehrt genau EINEN der 11 Generierungs-Schritte aus und zeichnet
        danach neu, damit jeder Klick sofort sichtbar wird. Kein-Op, wenn
        gerade kein Schritt-Durchlauf ansteht (siehe ui.py._toggle_play,
        das diese Methode nur aufruft, solange _pending_generation_steps
        noch Eintraege hat)."""
        steps = getattr(self, "_pending_generation_steps", None) or []
        if self._generation_step_index >= len(steps):
            return
        name, step_fn = steps[self._generation_step_index]
        self._generation_step_index += 1
        total = len(steps)
        self._log_step(f"--- Schritt {self._generation_step_index}/{total}: {name} ---")
        try:
            step_fn()
        except Exception as e:
            self._log_step(f"FEHLER in Schritt {self._generation_step_index} ({name}): {e}")
            logger.exception("Fehler in Generierungs-Schritt %s", name)
        # Bug-Fix (Nutzer-Feedback 2026-07-15): ridge_vertex_positions/
        # current_ridge_edges wurden bisher nur am Ende von Schritt 10
        # (Finalisierung) bzw. jeden Physik-Tick aktualisiert -- direkt nach
        # einem Zwischen-Schritt wie der Wildnisgrenzen-Snap zeigten die
        # generischen blauen 'voronoi_vertices'-Punkte (aus diesem veralteten
        # Snapshot) noch die ALTE Position, waehrend der neue cyan
        # wilderness_border_nodes-Marker (liest live plot_nodes direkt) schon
        # an der NEUEN Position erschien -- sah aus wie ein neu erzeugter
        # Punkt, obwohl kein neuer PlotNode entstanden war. no-op, solange
        # noch kein Voronoi-Netz existiert (Schritte vor Schritt 4).
        self._refresh_live_ridge_edges()
        self._mark_static_layer_dirty()
        self._redraw()

    def _gen_step_1_background(self):
        """Schritt 1/11: Terrain-Hintergrund + Regionen-Partitionierung neu
        aufbauen (siehe scene._recompute_background), aller Node-/Topologie-
        Zustand geleert. Prueft direkt hier, VOR jedem Node-Sampling, ob ein
        Wildnis-Polygon bereits eine verdaechtig lange/gerade Kante hat --
        moeglicher Bruecken-Artefakt aus dem buffer(0)-Reparatur-Schritt in
        scene._build_wilderness_boundary_points, der bislang unbemerkt direkt
        zu den auffaelligen geraden wilderness_node-Ketten fuehren kann."""
        self._recompute_background()
        self.next_node_id = 0

        self.nodes = []
        self.plot_nodes = []
        self.vertex_to_plot_node = {}
        self._plot_node_to_vertex = {}
        self.core_registry = {}

        self.boundary_owner = {}

        self.wilderness_node_ids = set()
        self.map_border_node_ids = set()
        self._core_cell_plot_node_ids = {}
        self._core_type_by_id = {}
        self._plot_node_wilderness_cache = {}

        self._static_vertex_positions = None
        self._static_ridge_edges = []
        self._static_num_vertices = 0
        self._static_boundary_entries = []
        self._static_boundary_settlement = []
        self._static_node_entry = {}
        self._static_predecessors = None
        self._static_distances = None

        self.ridge_traffic_history = {}
        self.ridge_traffic_shrink_ema = {}
        self.current_ridge_edges = []
        self.ridge_vertex_positions = None
        self.path_cache = {}

        self.selected_core_id = None
        self.selected_node_id = None
        self.iteration = 0
        self.topology_ready = False

        num_civ = int(getattr(self, "num_civ_regions", 0))
        num_wild = int(getattr(self, "num_wild_regions", 0))
        self._log_step(f"region_map: {num_civ} civ-Regionen, {num_wild} Wildnis-Regionen")

        polygon_region_ids = getattr(self, "_wilderness_polygon_region_ids", None) or []
        for poly_idx, poly in enumerate(self._wilderness_polygons):
            coords = np.array(poly.exterior.coords, dtype=float)
            seg_lengths = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
            if len(seg_lengths) == 0:
                continue
            median_len = float(np.median(seg_lengths))
            max_len = float(np.max(seg_lengths))
            rid = polygon_region_ids[poly_idx] if poly_idx < len(polygon_region_ids) else -1
            suspicious = median_len > 1e-6 and max_len > max(10.0 * median_len, 60.0)
            flag = " <-- VERDAECHTIG LANGE/GERADE KANTE" if suspicious else ""
            self._log_step(
                f"  Polygon {poly_idx} (civ-Region {rid}): {len(coords) - 1} Kanten, "
                f"median_len={median_len:.1f}px, max_len={max_len:.1f}px{flag}")

    def _gen_step_2_plot_cores(self):
        """Schritt 2/11: Seed-Punkte GLEICHMAESSIG UEBER DIE GESAMTE KARTE
        sampeln, mit civ-abhaengiger Dichte, deren civ-Wert aber bei 0.30
        GEFLOORT wird (Nutzer-Design: keine Unterscheidung civ/Wildnis mehr
        beim Sampling -- Wildnisflaechen bekommen so trotzdem nie einen
        groesseren Abstand als civ=0.30 ergaebe, wichtig fuer ein sauberes
        Voronoi ohne riesige Luecken). Alle Punkte starten als
        'standard_plot_node'; die eigentliche civ/Wildnis-Klassifikation
        passiert erst als Post-Process in Schritt 6, nachdem das Voronoi
        (Schritt 4) und die Wildnisgrenzen-Klassifikation (Schritt 5)
        bereits stehen."""
        min_buffer_to_city_px = 10.0
        # Bug-Fix: city_mask ist -1 ausserhalb, sonst settlement.location_id
        # (der ERSTE erzeugte Settlement hat location_id==0!) -- ">0" wuerde
        # dieses erste Settlement faelschlich als "kein Stadtgebiet" werten,
        # ">=0" ist die im Rest der Codebase korrekt verwendete Konvention.
        city_inside = self.city_mask >= 0
        dist_to_city = distance_transform_edt(~city_inside)

        edge_margin_px = int(round(self.PLOT_CORE_EDGE_MARGIN))
        map_edge_mask = np.ones_like(self.civ_map, dtype=bool)
        if edge_margin_px > 0:
            map_edge_mask[:edge_margin_px, :] = False
            map_edge_mask[-edge_margin_px:, :] = False
            map_edge_mask[:, :edge_margin_px] = False
            map_edge_mask[:, -edge_margin_px:] = False

        valid_mask = (
                (~city_inside)
                & (dist_to_city >= min_buffer_to_city_px)
                & map_edge_mask
        )

        effective_civ_map = np.maximum(self.civ_map, 0.30)

        positions = self._best_candidate_sample(
            valid_mask=valid_mask,
            count=int(self.plot_nodes_count),
            base_spacing=self.plot_base_spacing,
            civ_spacing_factor=self.plot_civ_spacing_factor,
            civ_map=effective_civ_map,
        )

        for x, y in positions:
            node = PlotNode(
                node_id=self.next_node_id,
                node_location=(float(x), float(y)),
                connector_ids=[],
                connector_distances=[],
                connector_elevations=[],
                connector_move_costs=[],
                connector_edge_ids=[],
                settlement_id=-1,
                node_type="standard_plot_node",
                neighbor_core_ids=[],
                neighbor_node_ids=[],
                traffic_weight=random_mod.uniform(3.0, 5.0),
            )
            self.nodes.append(node)
            self.next_node_id += 1

        self._log_step(f"Seed-Punkte: {len(positions)} gesamt (civ geflooret auf min. 0.30)")

        edge_violations = sum(
            1 for x, y in positions
            if x < edge_margin_px or x > self.map_size - edge_margin_px
            or y < edge_margin_px or y > self.map_size - edge_margin_px
        )
        self._log_step(f"  Rand-Abstand-Check: {edge_violations} Kerne < {edge_margin_px}px vom Kartenrand")

    def _gen_step_2b_relax_seed_points(self):
        """Schritt 3/11 (2b): gleicht die in Schritt 2 gesampelten Seed-Punkte
        untereinander aus (Feder-artige Relaxation, civ-abhaengige
        Soll-Distanz ueber dieselbe Formel wie die Live-Federn, siehe
        physics._rest_length_core_plotnode_batch) und drueckt sie danach
        mindestens SEED_RELAX_MARGIN px von der Wildnis-/Civ-Kontur und vom
        Kartenrand nach innen (Nutzer-Feedback: Schritt 2 allein war 'etwas
        schlecht verteilt').

        Jeder Punkt bekommt Federn nur zu seinen SEED_RELAX_NEIGHBOR_COUNT
        naechsten Nachbarn (k-nearest, scipy cKDTree), NICHT zu allen Paaren
        in einem festen Radius -- ein erster Versuch mit einem festen Radius
        (2x plot_base_spacing) UND aufsummierten (statt gemittelten) Kraeften
        verschlechterte die Verteilung messbar (Variationskoeffizient des
        Nachbarabstands stieg von ~0.15 auf ~0.9 nach der 'Relaxation' --
        das genaue Gegenteil des Ziels), weil in dichten Bereichen leicht
        50-100+ Nachbarn in den Radius fielen und ihre AUFSUMMIERTEN
        Federkraefte (jede einzeln bis MAX_SPRING_RESULTANT gedeckelt, aber
        die Summe nicht) zu grossen, instabilen Spruengen fuehrten. Mit fixer
        Nachbarzahl UND gemitteltem (nicht aufsummiertem) Kraftvektor bleibt
        die Verschiebung pro Punkt pro Iteration unabhaengig von der
        lokalen Dichte im selben Groessenbereich wie eine einzelne Feder.

        Reine Vorverarbeitung vor dem Voronoi (Schritt 4) -- keine
        Geschwindigkeiten/Masse noetig, einfache explizite Integration
        reicht (kein volles F=ma wie im Live-Tick)."""
        points = [n for n in self.nodes if n.node_type == "standard_plot_node"]
        if len(points) < 2:
            self._log_step("Zu wenige Seed-Punkte fuer Relaxation, uebersprungen.")
            return

        positions = np.array([n.node_location for n in points], dtype=float)
        polygons = list(self._wilderness_polygons)
        polylines = [np.array(poly.exterior.coords, dtype=float) for poly in polygons]
        margin = self.SEED_RELAX_MARGIN
        lo, hi = margin, float(self.map_size) - margin
        k = min(self.SEED_RELAX_NEIGHBOR_COUNT + 1, len(positions))

        def _push_from_contours(pos_arr):
            for idx in range(len(pos_arr)):
                for polyline in polylines:
                    p = pos_arr[idx]
                    nearest = self._nearest_point_on_polyline(p, polyline, closed=True)
                    dist = float(np.hypot(nearest[0] - p[0], nearest[1] - p[1]))
                    if dist < margin:
                        direction = p - nearest
                        dnorm = float(np.hypot(direction[0], direction[1]))
                        direction = direction / dnorm if dnorm > 1e-9 else np.array([1.0, 0.0])
                        pos_arr[idx] = nearest + direction * margin
            return pos_arr

        for _iteration in range(self.SEED_RELAX_ITERATIONS):
            if k > 1:
                tree = cKDTree(positions)
                _dists, neighbor_idx = tree.query(positions, k=k)
                self_idx = np.repeat(np.arange(len(positions)), k - 1)
                neighbor_flat = neighbor_idx[:, 1:].ravel()  # Spalte 0 ist der Punkt selbst

                rest_lengths_per_point = self._rest_length_core_plotnode_batch(positions)
                rest_lengths = 0.5 * (rest_lengths_per_point[self_idx] + rest_lengths_per_point[neighbor_flat])

                force_a, _force_b = self._spring_force_batch(
                    positions[self_idx], positions[neighbor_flat], rest_lengths,
                    stiffness=self.plotnode_plotnode_spring_stiffness, growth_rate=0.10)

                # GEMITTELT (nicht aufsummiert) ueber die k Nachbarn -- macht
                # die Verschiebung pro Punkt unabhaengig von k/lokaler Dichte.
                net_force = np.zeros_like(positions)
                np.add.at(net_force, self_idx, force_a)
                avg_force = net_force / float(k - 1)
                positions = positions + avg_force * self.SEED_RELAX_STEP

            positions[:, 0] = np.clip(positions[:, 0], lo, hi)
            positions[:, 1] = np.clip(positions[:, 1], lo, hi)
            positions = _push_from_contours(positions)

        for node, pos in zip(points, positions):
            node.node_location = (float(pos[0]), float(pos[1]))

        self._log_step(
            f"Seed-Relaxation: {len(points)} Punkte, {self.SEED_RELAX_ITERATIONS} Iterationen, "
            f"k={k - 1} Nachbarn/Punkt, Randabstand>={margin:.0f}px erzwungen")

    def _gen_step_2c_place_city_cores(self):
        """Schritt 4/11 (2c): pro Siedlung einen dedizierten Voronoi-Seed-Punkt
        ('Stadtkern'/city_core) exakt an ihrer Position setzen -- Nutzer-
        Design 2026-07-15, ersetzt die kreuzungsbasierte Stadtgrenz-
        Erkennung komplett. Grund: in einem einzigen geschlossenen,
        zusammenhaengenden Voronoi-Netz (siehe Schritt 4) laesst sich keine
        Kante mehr 'aufbrechen', ohne ein Loch zu erzeugen -- die
        kreuzungsbasierte Erkennung fand deshalb bei ~40% der Siedlungen
        (kleine Stadtkontur faellt komplett in EINE Voronoi-Zelle, keine
        Kante kreuzt sie je) gar keine Stadttore. Ein dedizierter Seed hat
        dagegen IMMER eine eigene Voronoi-Zelle mit echten Nachbar-plot_
        nodes -- Schritt 5 (Stadtgrenze verteilen) verteilt genau diese auf
        die Stadtkontur, garantiert nie leer.

        Laeuft NACH der Seed-Relaxation (2b), damit der Stadtkern nicht von
        deren Federkraeften mitverschoben wird (existiert zu diesem
        Zeitpunkt schlicht noch nicht) -- und VOR dem Voronoi (Schritt 4),
        damit er als vollwertiger Seed-Punkt daran teilnimmt.

        WICHTIG: muss ueber self.nodes.append() + self.next_node_id-
        Inkrement laufen, GENAU wie jeder andere self.nodes-Eintrag (siehe
        _gen_step_2_plot_cores) -- _sync_core_registry (Schritt 10) indiziert
        core_registry ueber enumerate(self.nodes) und loest neighbor_core_ids
        (echte node_ids) darueber auf; das funktioniert nur, weil node_id
        aktuell IMMER exakt dem Index in self.nodes entspricht. Ein Einfuegen
        ausser der Reihe oder eine wiederverwendete ID wuerde diese implizite
        Invariante lautlos brechen."""
        for settlement in self.settlements:
            node = PlotNode(
                node_id=self.next_node_id,
                node_location=(float(settlement.x), float(settlement.y)),
                connector_ids=[],
                connector_distances=[],
                connector_elevations=[],
                connector_move_costs=[],
                connector_edge_ids=[],
                settlement_id=settlement.location_id,
                node_type="city_core",
                neighbor_core_ids=[],
                neighbor_node_ids=[],
            )
            self.nodes.append(node)
            self.next_node_id += 1

        self._log_step(f"Stadtkerne: {len(self.settlements)} gesetzt (je einer pro Siedlung)")

    def _gen_step_4_voronoi_clipped(self):
        """Schritt 5/11: EIN einziges globales Voronoi ueber ALLE Seed-Punkte
        (Schritt 2), geklippt auf das Kartenrechteck (Kartenrand minus
        PLOT_CORE_EDGE_MARGIN nach innen) -- ersetzt die fruehere Regionen-
        Partitionierung + das separate Wildnis-Dreiecksnetz + das Naht-
        Stitching komplett (Nutzer-Design: kein Zusammenfuegen getrennter
        Netze mehr im Nachhinein -- ein einziges, von Anfang an
        zusammenhaengendes Netz, das an seinen echten Raendern sauber
        geklippt ist). Vertices, die erst durch das Klippen entstehen,
        werden hier direkt als 'map_border_node' klassifiziert und ab
        sofort NIE mehr bewegt (siehe physics._physics_step)."""
        if not self._build_voronoi_mesh():
            self.topology_ready = False
            self._log_step("Voronoi FEHLGESCHLAGEN (siehe Logfile fuer Details).")
            return
        self._log_step(
            f"Voronoi (global, geklippt): {len(self.plot_nodes)} plot_nodes, "
            f"{len(self.current_ridge_edges)} Ridge-Kanten, "
            f"{len(self.map_border_node_ids)} davon map_border_node")

    def _gen_step_city_boundary_distribute(self):
        """Schritt 6/11: verteilt fuer jeden Stadtkern (city_core, Schritt 4/11)
        dessen EIGENE Voronoi-Zellen-Nachbarn (die plot_nodes, die diese
        Zelle begrenzen) auf die tatsaechliche Stadtkontur
        (self._city_polygons[settlement_id]) und macht sie unbeweglich.

        Ersetzt die fruehere kreuzungsbasierte Erkennung komplett (Nutzer-
        Design 2026-07-15): ein dedizierter Seed-Punkt hat IMMER eine eigene
        Voronoi-Zelle mit echten Nachbarn (im Gegensatz zur Kreuzungssuche,
        die bei kleinen Stadtkonturen, die komplett in einer einzigen
        Voronoi-Zelle liegen, oft gar nichts fand -- ~40% der Siedlungen
        blieben ohne Stadttore). Kein Crossing-Test noetig: neighbor_
        core_ids auf einem plot_node ist per Konstruktion (siehe
        _build_plot_node_registry) mit echten node_ids befuellt, ein
        direkter Scan reicht.

        Laeuft VOR der Wildnisgrenzen-Erkennung (Schritt 6): Staedte liegen
        strukturell tief im Civ-Gebiet, bereits stadt-klassifizierte
        Endpunkte werden durch Schritt 6s 'beide Endpunkte noch
        standard_plot_node'-Wache sauber uebersprungen.

        Jeder verteilte Knoten bekommt node_type='city_border_node' UND
        settlement_id UND einen self.boundary_owner-Eintrag -- Dijkstra-
        Boundary-Entries (Schritt 8), Traffic-Sim (traffic.py) und
        Rendering (draw.py) haengen alle an boundary_owner, nicht am
        node_type allein."""
        city_cores = [n for n in self.nodes if n.node_type == "city_core"]
        if not city_cores:
            self._log_step("Keine Stadtkerne vorhanden, Stadtgrenzen-Verteilung uebersprungen.")
            return
        city_polygons = getattr(self, "_city_polygons", None) or {}

        def _snap_to_contour(pos, polylines):
            best_point, best_dist = None, np.inf
            for polyline in polylines:
                point = self._nearest_point_on_polyline(pos, polyline, closed=True)
                dist = float(np.hypot(point[0] - pos[0], point[1] - pos[1]))
                if dist < best_dist:
                    best_dist, best_point = dist, point
            return best_point

        distributed_count = 0
        fallback_count = 0
        empty_settlements = []

        for city_core in city_cores:
            settlement_id = city_core.settlement_id
            own_neighbors = [
                pn for pn in self.plot_nodes
                if city_core.node_id in pn.neighbor_core_ids and pn.node_type == "standard_plot_node"
            ]
            if not own_neighbors:
                empty_settlements.append(settlement_id)
                logger.warning(
                    "Stadtkern settlement_id=%s hat keine eigenen Voronoi-Nachbarn.", settlement_id)
                continue

            polygons = city_polygons.get(settlement_id) or []
            polylines = [np.array(poly.exterior.coords, dtype=float) for poly in polygons] if polygons else []

            for pn in own_neighbors:
                # Fallback (Regression gefunden 2026-07-15): sehr kleine
                # Staedte (wenige Raster-Pixel) liefern manchmal KEIN
                # gueltiges Marching-Squares-Polygon (zu klein/degeneriert
                # fuer eine sinnvolle Kontur, selbst nach dem CITY_MIN_AREA-
                # Fix). Statt die Siedlung dadurch komplett ohne Stadttore
                # zu lassen (Netz-Trennung!), werden ihre Voronoi-Zellen-
                # Nachbarn trotzdem zu city_border_node/boundary_owner --
                # nur OHNE Repositionierung, da keine sinnvolle Kontur zum
                # Snappen existiert; sie bleiben einfach an ihrer aktuellen
                # Voronoi-Position stehen und werden unbeweglich.
                if polylines:
                    snapped = _snap_to_contour(pn.node_location, polylines)
                    if snapped is not None:
                        pn.node_location = (float(snapped[0]), float(snapped[1]))
                    else:
                        fallback_count += 1
                else:
                    fallback_count += 1
                pn.node_type = "city_border_node"
                pn.settlement_id = settlement_id
                self.boundary_owner[pn.node_id] = settlement_id
                distributed_count += 1

        self._log_step(
            f"Stadtgrenzen-Verteilung: {len(city_cores)} Stadtkerne, "
            f"{distributed_count} plot_nodes zu city_border_node verteilt "
            f"({fallback_count} ohne Kontur-Snap, nur Fixierung)"
            + (f", OHNE Tore: {empty_settlements}" if empty_settlements else ""))

    def _gen_step_5_wilderness_snap(self):
        """Schritt 7/11: Jede Ridge-Kante finden, die die Wildnisgrenze
        (self._wilderness_polygons, bereits korrekt geschlossen seit dem
        Stage-1-Padding-Fix) tatsaechlich KREUZT -- ALLE Kreuzungspunkte
        einer Kante werden erfasst (nicht nur der erste), aber es werden
        IMMER nur die beiden bestehenden Endpunkte verschoben, nie neue
        Knoten erzeugt:

        - GENAU 1 Kreuzung (Normalfall, Endpunkte auf verschiedenen Seiten):
          der NAEHERE der beiden Endpunkte wird exakt auf die Kontur
          gesnappt und als 'wilderness_node' klassifiziert -- der andere
          bleibt unangetastet.
        - 2+ Kreuzungen (beide Endpunkte auf derselben Seite, die Kante
          durchquert eine schmale gegenteilige Landzunge): BEIDE Endpunkte
          werden gesnappt (node_a auf die ihm naechste, node_b auf die ihm
          naechste Kreuzung) und als 'wilderness_node' klassifiziert.

        Nutzer-Entscheidung 2026-07-15 (nach Live-Test): eine fruehere
        Version erzeugte bei 3+ Kreuzungen zusaetzlich neue Zwischen-Knoten
        fuer jede mittlere Kreuzung (Kette aus Teilstuecken) -- das machte
        das Netz live unuebersichtlicher, als es half. Jetzt bewusst
        einfach gehalten: das bestehende Netz "verschiebt sich" an der
        Grenze, es waechst nie. Eine seltene mittlere Kreuzung auf einer
        3+-Kreuzungs-Kante bleibt unrepraesentiert -- akzeptierter
        Kompromiss, siehe Analyse-Notiz zur vorherigen Version.

        Nur Kanten zwischen zwei noch unklassifizierten 'standard_plot_node'
        werden betrachtet, damit bereits gesnappte Randknoten nicht ein
        zweites Mal verschoben werden (macht die Verarbeitung reihenfolge-
        abhaengig bei Knoten, die von mehreren kreuzenden Kanten geteilt
        werden -- dieselbe Einschraenkung, die schon vorher galt)."""
        if self._static_vertex_positions is None or not self.plot_nodes:
            self._log_step("Kein Voronoi-Netz vorhanden, Wildnisgrenzen-Snap uebersprungen.")
            return
        polygons = list(self._wilderness_polygons)
        if not polygons:
            self._log_step("Keine Wildnis-Polygone vorhanden, Wildnisgrenzen-Snap uebersprungen.")
            return

        plot_node_by_id = {pn.node_id: pn for pn in self.plot_nodes}
        boundaries = [poly.exterior for poly in polygons]
        polylines = [np.array(b.coords, dtype=float) for b in boundaries]

        def _snap_to_contour(pos):
            best_point = None
            best_dist = np.inf
            for polyline in polylines:
                point = self._nearest_point_on_polyline(pos, polyline, closed=True)
                dist = float(np.hypot(point[0] - pos[0], point[1] - pos[1]))
                if dist < best_dist:
                    best_dist = dist
                    best_point = point
            return best_point

        snapped_count = 0
        checked_edges = 0

        for i, j, p1, p2, cost in self._static_ridge_edges:
            pid_a = self.vertex_to_plot_node.get(i)
            pid_b = self.vertex_to_plot_node.get(j)
            if pid_a is None or pid_b is None or pid_a == pid_b:
                continue
            node_a = plot_node_by_id.get(pid_a)
            node_b = plot_node_by_id.get(pid_b)
            if node_a is None or node_b is None:
                continue
            if node_a.node_type != "standard_plot_node" or node_b.node_type != "standard_plot_node":
                continue

            seg = LineString([p1, p2])
            seg_vec = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
            seg_len_sq = float(np.dot(seg_vec, seg_vec))

            crossing_ts = []
            for boundary in boundaries:
                if not seg.intersects(boundary):
                    continue
                intersection = seg.intersection(boundary)
                if intersection.is_empty:
                    continue
                pts = extract_intersection_points(intersection)

                for px, py in pts:
                    t = float(np.dot((px - p1[0], py - p1[1]), seg_vec) / seg_len_sq) if seg_len_sq > 1e-12 else 0.0
                    crossing_ts.append((t, (float(px), float(py))))

            if not crossing_ts:
                continue
            checked_edges += 1

            crossing_ts.sort(key=lambda item: item[0])
            deduped = []
            for t, pt in crossing_ts:
                if deduped and abs(t - deduped[-1][0]) < 1e-6:
                    continue
                deduped.append((t, pt))
            crossing_ts = deduped

            if len(crossing_ts) == 1:
                _, (ix, iy) = crossing_ts[0]
                dist_a = float(np.hypot(node_a.node_location[0] - ix, node_a.node_location[1] - iy))
                dist_b = float(np.hypot(node_b.node_location[0] - ix, node_b.node_location[1] - iy))
                target_node = node_a if dist_a <= dist_b else node_b

                snapped = _snap_to_contour(target_node.node_location)
                if snapped is not None:
                    target_node.node_location = (float(snapped[0]), float(snapped[1]))
                    target_node.node_type = "wilderness_node"
                    self.wilderness_node_ids.add(target_node.node_id)
                    snapped_count += 1
                continue

            # 2+ Kreuzungen: nur die beiden bestehenden Endpunkte werden
            # verschoben (node_a auf die ihm naechste, node_b auf die ihm
            # naechste Kreuzung) -- dazwischenliegende Kreuzungen bleiben
            # bewusst unrepraesentiert, siehe Docstring.
            snapped_a = _snap_to_contour(node_a.node_location)
            if snapped_a is not None:
                node_a.node_location = (float(snapped_a[0]), float(snapped_a[1]))
            node_a.node_type = "wilderness_node"
            self.wilderness_node_ids.add(node_a.node_id)

            snapped_b = _snap_to_contour(node_b.node_location)
            if snapped_b is not None:
                node_b.node_location = (float(snapped_b[0]), float(snapped_b[1]))
            node_b.node_type = "wilderness_node"
            self.wilderness_node_ids.add(node_b.node_id)
            snapped_count += 2

        self._log_step(
            f"Wildnisgrenzen-Snap: {checked_edges} kreuzende Ridge-Kanten gefunden, "
            f"{snapped_count} plot_nodes zu wilderness_node")

    def _gen_step_6_wilderness_cores(self):
        """Schritt 8/11: einfache Nachklassifikation -- jeder verbliebene
        Seed-Punkt ('standard_plot_node', also weder map_border_node noch
        wilderness_node), dessen civ-Wert an seiner Position unter
        WILDERNESS_CIV_THRESHOLD liegt, wird zu 'wilderness_core'
        umbenannt. Reine Label-Aenderung, keine Neupositionierung -- der
        Seed-Punkt sitzt ja bereits an seiner endgueltigen Stelle
        (Nutzer-Design: 'die Plotkerne die in der Wildnis liegen
        umdeklariert in Wildniskerne')."""
        reclassified = 0
        for node in self.nodes:
            if node.node_type != "standard_plot_node":
                continue
            x, y = node.node_location
            px = int(np.clip(round(x), 0, self.map_size - 1))
            py = int(np.clip(round(y), 0, self.map_size - 1))
            if float(self.civ_map[py, px]) < self.WILDERNESS_CIV_THRESHOLD:
                node.node_type = "wilderness_core"
                reclassified += 1
        self._log_step(f"Wildniskerne: {reclassified} Seed-Punkte zu wilderness_core umklassifiziert")

    def _gen_step_7_build_graph(self):
        """Schritt 9/11: Dijkstra-Graph aus dem fertigen, vollstaendig
        klassifizierten Netz aufbauen -- ERST NACHDEM Schritt 5
        (Wildnisgrenzen-Snap, verschiebt manche plot_node-Positionen) und
        Schritt 6 (Wildniskern-Umklassifikation) abgeschlossen sind, damit
        die Distanzen/Kosten nicht auf inzwischen veralteten Positionen
        beruhen. Jeder self.nodes-Eintrag (nur noch Seed-Punkte/Kerne, seit
        dem Umbau auf crossing-basierte Stadtgrenz-Nodes lebt city_border_
        node nicht mehr in self.nodes, siehe _gen_step_city_boundary_snap)
        bekommt eine Entry-Kante zur naechsten Voronoi-Vertex im GLOBALEN,
        flachen Vertex-Index-Raum -- keine Regionen-Aufteilung mehr noetig,
        das gesamte Netz ist seit Schritt 4 ohnehin schon EIN
        zusammenhaengendes Voronoi."""
        vertex_positions = self._static_vertex_positions
        ridge_edges = self._static_ridge_edges
        if vertex_positions is None or not ridge_edges:
            self._log_step("Kein Voronoi-Netz vorhanden, Graph-Aufbau uebersprungen.")
            return
        num_vertices = self._static_num_vertices

        global_tree = cKDTree(vertex_positions)
        node_entry = {}
        entry_edges = []
        for idx, node in enumerate(self.nodes):
            node_pos = np.array(node.node_location, dtype=float)
            entry_idx = num_vertices + idx
            node_entry[node.node_id] = entry_idx
            dist, vertex_idx = global_tree.query(node_pos)
            entry_edges.append((entry_idx, int(vertex_idx), max(float(dist), 1e-3)))

        total_graph_nodes = num_vertices + len(self.nodes)
        rows, cols, costs = [], [], []
        for i, j, _p1, _p2, cost in ridge_edges:
            rows.extend([i, j])
            cols.extend([j, i])
            costs.extend([cost, cost])
        for entry_idx, vertex_idx, cost in entry_edges:
            rows.extend([entry_idx, vertex_idx])
            cols.extend([vertex_idx, entry_idx])
            costs.extend([cost, cost])

        # city_border_node lebt seit dem Umbau auf crossing-basierte
        # Stadtgrenz-Nodes (siehe _gen_step_city_boundary_snap) als
        # klassifizierter plot_node in self.plot_nodes, nicht mehr als
        # eigener self.nodes-Eintrag -- node_entry (nur aus self.nodes
        # gebaut, siehe oben) findet solche IDs also nie. Sie sind aber
        # bereits vollwertige Vertices im flachen Voronoi-Vertex-Raum
        # (0..num_vertices-1, siehe _plot_node_to_vertex), brauchen also gar
        # keine Entry-Kante -- ihr Vertex-Index selbst ist der Graph-Knoten.
        boundary_entries = []
        boundary_settlement = []
        for node_id, settlement_id in self.boundary_owner.items():
            if node_id in node_entry:
                boundary_entries.append(node_entry[node_id])
                boundary_settlement.append(settlement_id)
            elif node_id in self._plot_node_to_vertex:
                boundary_entries.append(self._plot_node_to_vertex[node_id])
                boundary_settlement.append(settlement_id)

        if not boundary_entries:
            self._log_step("Keine Boundary-Entries gefunden, Graph unvollstaendig.")
            return

        try:
            graph = csr_matrix((costs, (rows, cols)), shape=(total_graph_nodes, total_graph_nodes))
            distances, predecessors = dijkstra(graph, indices=boundary_entries, return_predecessors=True)
        except Exception as e:
            self._log_step(f"Dijkstra-Berechnung fehlgeschlagen: {e}")
            logger.error("Initiale Dijkstra-Berechnung fehlgeschlagen: %s", e)
            return

        self._static_boundary_entries = boundary_entries
        self._static_boundary_settlement = boundary_settlement
        self._static_node_entry = node_entry
        self._static_predecessors = predecessors
        self._static_distances = distances

        self._log_step(
            f"Graph: {total_graph_nodes} Knoten gesamt ({num_vertices} Voronoi-Vertices + "
            f"{len(self.nodes)} Seed/Grenz-Nodes), {len(boundary_entries)} Boundary-Entries")

    def _gen_step_8_consistency_check(self):
        """Schritt 10/11: Regressionstest gegen genau die Fehlerklasse, die
        dieser Umbau beheben soll -- nach echtem Klippen (Schritt 4) darf
        KEIN Vertex mehr ausserhalb des Kartenrechtecks liegen; jeder
        map_border_node/wilderness_node muss nahe an seiner jeweiligen
        Grenze sitzen; kein standard_plot_node darf noch in der Wildnis
        liegen (Schritt 6 haette ihn sonst umklassifiziert); volle
        Erreichbarkeit zwischen allen Siedlungen."""
        inset = self.PLOT_CORE_EDGE_MARGIN
        size = float(self.map_size)
        vertex_positions = self._static_vertex_positions
        outside_count = 0
        if vertex_positions is not None and len(vertex_positions):
            tol = 0.5
            outside_count = int(np.count_nonzero(
                (vertex_positions[:, 0] < inset - tol) | (vertex_positions[:, 0] > size - inset + tol) |
                (vertex_positions[:, 1] < inset - tol) | (vertex_positions[:, 1] > size - inset + tol)
            ))

        border_far = 0
        for node in self.plot_nodes:
            if node.node_type != "map_border_node":
                continue
            x, y = node.node_location
            d = min(x - inset, size - inset - x, y - inset, size - inset - y)
            if abs(d) > 2.0:
                border_far += 1

        wilderness_far = 0
        polygons = list(self._wilderness_polygons)
        for node in self.plot_nodes:
            if node.node_type != "wilderness_node":
                continue
            if not polygons:
                wilderness_far += 1
                continue
            best = min(Point(node.node_location).distance(poly.exterior) for poly in polygons)
            if best > 2.0:
                wilderness_far += 1

        leftover_in_wilderness = 0
        for node in self.nodes:
            if node.node_type != "standard_plot_node":
                continue
            x, y = node.node_location
            px = int(np.clip(round(x), 0, self.map_size - 1))
            py = int(np.clip(round(y), 0, self.map_size - 1))
            if float(self.civ_map[py, px]) < self.WILDERNESS_CIV_THRESHOLD:
                leftover_in_wilderness += 1

        dist = self._static_distances
        boundary_entries = self._static_boundary_entries
        boundary_settlement = self._static_boundary_settlement
        all_reachable = True
        if dist is not None and len(boundary_entries):
            settlement_ids = sorted(set(boundary_settlement))
            for row_idx, from_sid in enumerate(boundary_settlement):
                row = dist[row_idx]
                for to_sid in settlement_ids:
                    if to_sid == from_sid:
                        continue
                    cols = [c for c, sid in enumerate(boundary_settlement) if sid == to_sid]
                    best = min(row[boundary_entries[c]] for c in cols)
                    if not np.isfinite(best):
                        all_reachable = False

        self._log_step(
            f"Konsistenzcheck: {outside_count} Vertices ausserhalb der Klip-Box, "
            f"{border_far} map_border_node >2px vom Rand, "
            f"{wilderness_far} wilderness_node >2px von der Kontur, "
            f"{leftover_in_wilderness} standard_plot_node noch in Wildnis, "
            f"alle Siedlungen erreichbar={all_reachable}")

    def _gen_step_9_finalize(self):
        """Schritt 11/11: Registries synchronisieren, Zellgrenzen aufbauen,
        plot_nodes einmalig gegen die Wildnisgrenze bereinigen (siehe
        _sanitize_plot_node_positions), Rendering-Snapshot sofort
        synchronisieren (siehe Analyse-Notiz zum Iteration-3-Render-Bug),
        Geschwindigkeiten nullen."""
        self.topology_ready = True
        self._sync_core_registry()
        self._core_type_by_id = {node.node_id: node.node_type for node in self.nodes}
        self._build_core_cell_plot_node_ids()

        before = np.array([pn.node_location for pn in self.plot_nodes], dtype=float)
        self._sanitize_plot_node_positions()
        after = np.array([pn.node_location for pn in self.plot_nodes], dtype=float)
        if len(before) == len(after) and len(before):
            deltas = np.hypot(after[:, 0] - before[:, 0], after[:, 1] - before[:, 1])
            moved = int(np.count_nonzero(deltas > 1e-6))
            self._log_step(
                f"Sanitize: {moved}/{len(before)} plot_nodes korrigiert, "
                f"max_korrektur={float(deltas.max()):.1f}px")

        # Bug-Fix: ridge_vertex_positions/current_ridge_edges (einzige Quelle
        # fuer draw.py) muessen SOFORT die post-sanitize Positionen spiegeln --
        # sonst zeigt das Rendering bis zum ersten _refresh_live_ridge_edges()
        # in _tick() (nur alle TRAFFIC_RECOMPUTE_INTERVAL Ticks, siehe
        # physics.py) den veralteten Pre-Sanitize-Snapshot und "springt" dann
        # sichtbar auf die korrekten Positionen um.
        self._refresh_live_ridge_edges()
        for node in self.nodes:
            node.velocity = (0.0, 0.0)
        for node in self.plot_nodes:
            node.velocity = (0.0, 0.0)
        self._log_step(f"Finalisiert: {len(self.plot_nodes)} plot_nodes, {len(self.nodes)} Nodes gesamt.")

    def _sanitize_plot_node_positions(self):
        """Restkorrektur NACH Schritt 5: manche zivilisationsnahen plot_nodes
        bleiben trotz Schritt 5 auf der falschen (Wildnis-)Seite -- Schritt 5
        erkennt nur Ridge-Kanten, bei denen BEIDE Endpunkte noch
        unklassifizierte 'standard_plot_node' sind; ist eine Kante schon
        durch eine FRUEHER im selben Durchlauf verarbeitete Nachbarkante
        'verbraucht' (der eine Endpunkt ist bereits wilderness_node/
        map_border_node), bekommt der andere -- ggf. ebenfalls falsch
        liegende -- Endpunkt nie eine eigene Pruefung. Diese Methode faengt
        genau diese uebrig gebliebenen Faelle ab: jeder verbliebene falsch
        liegende Knoten wird auf den naechstgelegenen Konturpunkt projiziert
        (minimale Verschiebungsdistanz) und als wilderness_node klassifiziert.

        WICHTIG (Nutzer-Feedback 2026-07-15, per Diagnose-Skript verifiziert):
        ein erster Versuch snappte hier stattdessen auf den Schnittpunkt der
        Kante zu einem echten, bereits korrekt sitzenden Graph-Nachbarn --
        das machte es NOCH schlimmer, weil dieser Nachbar-Schnittpunkt oft
        SEHR VIEL weiter entfernt liegt als der global naechste Konturpunkt,
        und die dadurch groessere Verschiebung mehr von den UEBRIGEN
        (unveraenderten) Kanten dieses Knotens neu ueber die gezackte Kontur
        springen liess. Minimale Verschiebung (naechster Punkt, vgl. alte
        _contain_plot_node-Logik) ist der robustere Ansatz.

        WICHTIG (Nutzer-Feedback 2026-07-15, live getestet): _is_wrong_side
        hatte hier eine invertierte Bedingung -- self._wilderness_polygons'
        INNERES ist trotz des Namens die CIV-Region (siehe scene.
        _build_wilderness_boundary_points, mask = civ_map >=
        WILDERNESS_CIV_THRESHOLD), also bedeutet _point_in_polygon()==True
        'liegt korrekt auf der Civ-Seite', nicht 'liegt falsch'. Die erste
        Version pruefte genau umgekehrt und stufte dadurch praktisch das
        GESAMTE civ-nahe Netz als 'falsch liegend' ein -- sichtbar live als
        kompletter Kollaps des Civ-Netzes auf die Wildnisgrenze in Schritt 9.
        Korrekt: falsch liegend heisst ausserhalb JEDES civ-Polygons.

        map_border_node/wilderness_node werden hier nicht erneut geprueft --
        sitzen seit Schritt 4/5 bereits an ihrer endgueltigen Position."""
        prepared_wilderness = self._prepare_wilderness_polygons()
        if not prepared_wilderness:
            return

        type_by_id = getattr(self, "_core_type_by_id", {})
        polygons = list(self._wilderness_polygons)
        polylines = [np.array(poly.exterior.coords, dtype=float) for poly in polygons]

        def _is_civ_adjacent(pn):
            return any(type_by_id.get(cid) == "standard_plot_node" for cid in pn.neighbor_core_ids)

        def _is_wrong_side(pn):
            pos = np.array(pn.node_location, dtype=float)
            return not any(
                self._point_in_polygon(pos, coords, shifted)
                for coords, shifted, _b in prepared_wilderness
            )

        wrong_side_nodes = [
            pn for pn in self.plot_nodes
            if pn.node_type == "standard_plot_node" and _is_civ_adjacent(pn) and _is_wrong_side(pn)
        ]
        if not wrong_side_nodes:
            return

        def _snap_to_contour(pos):
            best_point, best_dist = None, np.inf
            for polyline in polylines:
                point = self._nearest_point_on_polyline(pos, polyline, closed=True)
                dist = float(np.hypot(point[0] - pos[0], point[1] - pos[1]))
                if dist < best_dist:
                    best_dist, best_point = dist, point
            return best_point

        corrected = 0
        for pn in wrong_side_nodes:
            snapped = _snap_to_contour(np.array(pn.node_location, dtype=float))
            if snapped is None:
                continue
            pn.node_location = (float(snapped[0]), float(snapped[1]))
            pn.node_type = "wilderness_node"
            self.wilderness_node_ids.add(pn.node_id)
            corrected += 1

        self._log_step(
            f"Sanitize (Restfaelle nach Schritt 5): {len(wrong_side_nodes)} falsch liegende plot_nodes "
            f"gefunden, {corrected} auf naechsten Konturpunkt projiziert und zu wilderness_node")

    def _build_core_cell_plot_node_ids(self):
        """Merkt sich fuer jeden regulaeren Plotkern (standard_plot_node) die
        nach Winkel sortierten IDs seiner benachbarten plot_nodes -- das
        Polygon aus deren AKTUELLEN (ggf. je Tick bewegten) Positionen ist
        die harte Bewegungsgrenze in physics._physics_step, damit ein Kern
        nie ueber seine eigene
        Plotgrenze wandert. Wildniskerne brauchen das nicht (keine Physik).

        An der Dichte-Grenze zwischen den eng gepackten Civ-Kernen und den
        weit gestreuten Wildniskernen (siehe _generate_wilderness_cores)
        koennen einzelne Voronoi-Kanten sehr lang/duenn ('Sliver') werden --
        core.neighbor_node_ids enthaelt dann fuer manche Kerne einen einzelnen,
        weit entfernten plot_node, der geometrisch zwar korrekt ein Nachbar
        ist, das angle-sortierte Polygon aber zu einer entarteten Form mit
        einer riesigen 'Spitze' verzerrt. Ausreisser (deutlich weiter weg als
        der Median der restlichen Nachbarn) werden deshalb aus dem Polygon
        entfernt, solange mindestens 3 Punkte uebrig bleiben."""
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
        self._core_cell_plot_node_ids = {}

        for node in self.nodes:
            if node.node_type != "standard_plot_node":
                continue
            core = self.core_registry.get(node.node_id)
            if core is None or len(core.neighbor_node_ids) < 3:
                continue

            cx, cy = node.node_location
            ids = [nid for nid in core.neighbor_node_ids if nid in plot_node_by_id]
            if len(ids) < 3:
                continue
            positions = np.array([plot_node_by_id[nid].node_location for nid in ids], dtype=float)

            dists = np.hypot(positions[:, 0] - cx, positions[:, 1] - cy)
            median_dist = float(np.median(dists))
            if median_dist > 1e-6:
                keep = dists <= median_dist * 3.0
                if np.count_nonzero(keep) >= 3:
                    ids = [ids[i] for i in range(len(ids)) if keep[i]]
                    positions = positions[keep]

            angles = np.arctan2(positions[:, 1] - cy, positions[:, 0] - cx)
            order = np.argsort(angles)
            self._core_cell_plot_node_ids[node.node_id] = [ids[i] for i in order]

    def _active_physics_cores(self):
        """Liefert nur die Kern-Nodes, die an Feder-/Feldkraeften teilnehmen:
        alle regulaeren Nodes MINUS Stadtgrenz-Nodes (self.boundary_owner)
        MINUS die als 'wilderness_core' oder 'city_core' markierten.
        city_core (Schritt 4/11, siehe _gen_step_2c_place_city_cores) muss
        exakt an der Siedlungsposition stehen bleiben -- genau wie
        wilderness_core dauerhaft von Feder-/Feldphysik ausgeschlossen.
        map_border_node/wilderness_node/city_border_node leben seit dem
        Umbau auf ein einziges geklipptes Voronoi nicht mehr in self.nodes,
        sondern als klassifizierte plot_nodes in self.plot_nodes (siehe
        _build_voronoi_mesh/_gen_step_5_wilderness_snap/
        _gen_step_city_boundary_distribute) -- brauchen hier also keinen
        gesonderten Ausschluss mehr."""
        excluded_ids = set(self.boundary_owner.keys())
        return [
            node for node in self.nodes
            if node.node_id not in excluded_ids
            and node.node_type not in ("wilderness_core", "city_core")
        ]

    def _build_plot_node_registry(self, vertex_positions, ridge_vertices_list, ridge_points, points, core_nodes):
        """Baut die plot_nodes (Voronoi-Kreuzungen) samt Nachbarschaftslisten.

        Kein radialer Clamp mehr noetig: vertex_positions kommt bereits
        fertig ans Kartenrechteck geklippt aus _build_voronoi_mesh (echtes
        shapely-Klippen an der Quelle) -- Ausreisser-Koordinaten jenseits
        der Kartengrenze koennen so gar nicht mehr entstehen. Der fruehere
        radiale Clamp war eine Kompensation genau fuer dieses inzwischen an
        der Quelle geloeste Problem (Umkreismittelpunkte fast-kollinearer
        Dreiecke nahe der konvexen Huelle, die quasi Richtung Unendlich
        schossen) und wurde beim Umbau auf ein einziges geklipptes globales
        Voronoi entfernt."""
        registry = {}
        vertex_to_plot_node = {}

        def pos_key(pos):
            return (round(float(pos[0]), 6), round(float(pos[1]), 6))

        for ridge_idx, ridge in enumerate(ridge_vertices_list):
            if len(ridge) != 2:
                continue

            i, j = ridge
            if i < 0 or j < 0:
                continue

            if i >= len(vertex_positions) or j >= len(vertex_positions):
                continue

            core_a, core_b = ridge_points[ridge_idx]
            for vidx in (i, j):
                raw_pos = vertex_positions[vidx]
                pos = (float(raw_pos[0]), float(raw_pos[1]))
                key = pos_key(pos)

                if key not in registry:
                    node = PlotNode(
                        node_id=self.next_node_id,
                        node_location=(float(pos[0]), float(pos[1])),
                        connector_ids=[],
                        connector_distances=[],
                        connector_elevations=[],
                        connector_move_costs=[],
                        connector_edge_ids=[],
                        settlement_id=-1,
                        node_type="standard_plot_node",
                        neighbor_core_ids=[],
                        neighbor_node_ids=[],
                    )
                    registry[key] = node
                    self.next_node_id += 1

                node = registry[key]
                vertex_to_plot_node[vidx] = node.node_id

                for core_id in (core_a, core_b):
                    if core_id not in node.neighbor_core_ids:
                        node.neighbor_core_ids.append(int(core_id))

        id_to_node = {node.node_id: node for node in registry.values()}
        for ridge in ridge_vertices_list:
            if len(ridge) != 2:
                continue

            i, j = ridge
            if i < 0 or j < 0:
                continue

            node_id_i = vertex_to_plot_node.get(i)
            node_id_j = vertex_to_plot_node.get(j)
            if node_id_i is None or node_id_j is None or node_id_i == node_id_j:
                continue

            node_i = id_to_node[node_id_i]
            node_j = id_to_node[node_id_j]

            if node_id_j not in node_i.neighbor_node_ids:
                node_i.neighbor_node_ids.append(node_id_j)
            if node_id_i not in node_j.neighbor_node_ids:
                node_j.neighbor_node_ids.append(node_id_i)

        return list(registry.values()), vertex_to_plot_node

    def _sync_core_registry(self):
        """Baut das Core-Registry komplett neu auf. Guenstig genug (nur
        len(self.nodes) Eintraege) und hat den Vorteil, dass core.location
        immer die aktuelle, durch die Physik bewegte Position widerspiegelt
        statt einer beim ersten Aufbau eingefrorenen. Wird NUR bei
        Topologie-Aenderungen aufgerufen (nicht mehr jeden Tick, siehe
        physics._tick -- fuer reine Positions-Updates genuegt
        _sync_core_positions)."""
        self.core_registry = {
            idx: PlotCore(core_id=idx, location=tuple(node.node_location),
                          region_id=getattr(node, "region_id", -1))
            for idx, node in enumerate(self.nodes)
        }

        for node in self.plot_nodes:
            for core_id in node.neighbor_core_ids:
                core = self.core_registry.get(core_id)
                if core is not None and node.node_id not in core.neighbor_node_ids:
                    core.neighbor_node_ids.append(node.node_id)

    def _sync_core_positions(self):
        """Aktualisiert NUR core.location aus dem aktuellen self.nodes-Zustand,
        ohne Nachbarschaften/Federn neu zu bauen. Wird jeden Tick aufgerufen,
        wo sich nur Positionen aendern, nicht die Topologie."""
        for idx, node in enumerate(self.nodes):
            core = self.core_registry.get(idx)
            if core is not None:
                core.location = tuple(node.node_location)

    def _build_voronoi_mesh(self):
        """Baut EIN globales Voronoi ueber alle 'standard_plot_node'-Seed-
        Punkte, geklippt auf das Kartenrechteck [inset, map_size-inset]^2
        (shapely box) -- ersetzt die fruehere Regionen-Partitionierung +
        das separate Wildnis-Dreiecksnetz + das Naht-Stitching komplett
        (Nutzer-Design: kein Zusammenfuegen getrennter Netze mehr im
        Nachhinein).

        Fuer jede Ridge:
        - beide Endpunkte endlich UND innerhalb der Box -> unveraendert,
          Original-Vertex-Indizes bleiben erhalten.
        - ein Endpunkt bei -1 (unendliche Ridge): Richtung ueber die
          Standard-Rezeptur rekonstruiert (Senkrechte auf die Verbindung
          der beiden erzeugenden Punkte, vom Punktwolken-Zentrum weg
          orientiert), weit verlaengert, dann geklippt.
        - jeder Endpunkt ausserhalb der Box (endlich oder der verlaengerte
          Strahl): gegen die Box geklippt (shapely .intersection()) -- der
          Schnittpunkt wird ein NEUER Vertex, gesammelt in
          boundary_vertex_indices.

        Baut danach die plot_nodes (siehe _build_plot_node_registry, ohne
        radialen Clamp -- echtes Klippen macht ihn ueberfluessig) und
        klassifiziert jeden plot_node, dessen Vertex aus dem Klippen
        entstand, direkt als 'map_border_node' (ab sofort permanent
        unbeweglich, siehe physics._physics_step). Baut NUR das Netz --
        der Dijkstra-Graph wird separat in _gen_step_7_build_graph
        aufgebaut, NACHDEM Wildnisgrenzen-Snap (Schritt 5) und
        Wildniskern-Umklassifikation (Schritt 6) abgeschlossen sind.

        Seed-Punkte: 'standard_plot_node' UND 'city_core' (Schritt 4/11,
        siehe _gen_step_2c_place_city_cores) -- ein Stadtkern MUSS als
        echter Voronoi-Seed teilnehmen, sonst hat er keine eigene Zelle und
        _gen_step_city_boundary_distribute findet keine Nachbarn zum
        Verteilen (Regression gefunden 2026-07-15: ohne diesen Filter blieb
        JEDE Siedlung ohne Stadttore)."""
        all_core_nodes = [n for n in self.nodes if n.node_type in ("standard_plot_node", "city_core")]
        if len(all_core_nodes) < 4:
            logger.error("Zu wenige Kern-Nodes fuer Topologie.")
            return False

        points = np.array([n.node_location for n in all_core_nodes], dtype=float)
        node_ids = [n.node_id for n in all_core_nodes]

        try:
            vor = Voronoi(points)
        except Exception as e:
            logger.error("Voronoi-Berechnung fehlgeschlagen: %s", e)
            return False

        inset = self.PLOT_CORE_EDGE_MARGIN
        x0, y0 = inset, inset
        x1, y1 = float(self.map_size) - inset, float(self.map_size) - inset
        clip_box = box(x0, y0, x1, y1)

        center = points.mean(axis=0)
        far_distance = float(self.map_size) * 4.0

        vertex_positions_list = [np.asarray(v, dtype=float) for v in vor.vertices]
        clip_point_index = {}

        def _clip_index_for(pt):
            key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
            idx = clip_point_index.get(key)
            if idx is not None:
                return idx
            idx = len(vertex_positions_list)
            vertex_positions_list.append(np.array([float(pt[0]), float(pt[1])], dtype=float))
            clip_point_index[key] = idx
            return idx

        def _clip_segment(a, b):
            """None: beide Enden bereits innerhalb (Original-Indizes nutzen).
            'OUTSIDE': Segment beruehrt die Box gar nicht (verwerfen).
            Sonst: (pa, pb, a_inside, b_inside), pa/pb in derselben
            Reihenfolge wie a/b (ggf. geklippt)."""
            a_inside = clip_box.covers(Point(a))
            b_inside = clip_box.covers(Point(b))
            if a_inside and b_inside:
                return None
            seg = LineString([a, b])
            clipped = seg.intersection(clip_box)
            if clipped.is_empty:
                return "OUTSIDE"
            coords = list(clipped.coords)
            if len(coords) < 2:
                return "OUTSIDE"
            p_first, p_last = np.array(coords[0]), np.array(coords[-1])
            if np.hypot(*(p_first - a)) <= np.hypot(*(p_last - a)):
                pa_c, pb_c = p_first, p_last
            else:
                pa_c, pb_c = p_last, p_first
            return (pa_c, pb_c, a_inside, b_inside)

        ridge_vertices_list = []
        ridge_points = []
        boundary_vertex_indices = set()

        for ridge_idx, (i, j) in enumerate(vor.ridge_vertices):
            p1_local, p2_local = vor.ridge_points[ridge_idx]

            if i >= 0 and j >= 0:
                a_pos = np.asarray(vor.vertices[i], dtype=float)
                b_pos = np.asarray(vor.vertices[j], dtype=float)
                result = _clip_segment(a_pos, b_pos)
                if result is None:
                    idx_a, idx_b = i, j
                elif result == "OUTSIDE":
                    continue
                else:
                    pa_c, pb_c, a_inside, b_inside = result
                    idx_a = i if a_inside else _clip_index_for(pa_c)
                    idx_b = j if b_inside else _clip_index_for(pb_c)
                    if not a_inside:
                        boundary_vertex_indices.add(idx_a)
                    if not b_inside:
                        boundary_vertex_indices.add(idx_b)
            else:
                finite_idx = j if i < 0 else i
                finite_vertex = np.asarray(vor.vertices[finite_idx], dtype=float)
                t = points[p2_local] - points[p1_local]
                norm_t = float(np.linalg.norm(t))
                if norm_t < 1e-9:
                    continue
                t = t / norm_t
                n = np.array([-t[1], t[0]])
                midpoint = (points[p1_local] + points[p2_local]) / 2.0
                direction = n if np.dot(midpoint - center, n) > 0 else -n
                far_point = finite_vertex + direction * far_distance

                result = _clip_segment(finite_vertex, far_point)
                if result is None or result == "OUTSIDE":
                    continue
                pa_c, pb_c, a_inside, b_inside = result
                idx_a = finite_idx if a_inside else _clip_index_for(pa_c)
                idx_b = finite_idx if b_inside else _clip_index_for(pb_c)
                if not a_inside:
                    boundary_vertex_indices.add(idx_a)
                if not b_inside:
                    boundary_vertex_indices.add(idx_b)

            if idx_a == idx_b:
                continue
            ridge_vertices_list.append((idx_a, idx_b))
            ridge_points.append((node_ids[p1_local], node_ids[p2_local]))

        if not ridge_vertices_list:
            logger.error("Keine gueltigen Ridge-Kanten nach dem Klippen gefunden.")
            return False

        # Kompaktieren: vertex_positions_list enthaelt noch ALLE rohen
        # vor.vertices (auch nie geklippte, von keiner ueberlebenden Ridge
        # referenzierte Ausreisser weit ausserhalb der Karte, z.B. Vertices
        # von komplett verworfenen Huellen-Ridges) plus die neuen Klip-Punkte.
        # Nur die tatsaechlich benutzten Indizes behalten und neu
        # durchnummerieren, sonst wuerden diese toten Ausreisser als "Vertex
        # ausserhalb der Klip-Box" im Konsistenzcheck auftauchen, obwohl sie
        # nirgends im Graphen/Rendering referenziert werden.
        used_indices = sorted({idx for pair in ridge_vertices_list for idx in pair})
        remap = {old: new for new, old in enumerate(used_indices)}
        vertex_positions = np.array([vertex_positions_list[old] for old in used_indices], dtype=float)
        ridge_vertices_list = [(remap[a], remap[b]) for a, b in ridge_vertices_list]
        boundary_vertex_indices = {remap[old] for old in boundary_vertex_indices if old in remap}
        num_vertices = len(vertex_positions)

        ridge_edges = []
        for i, j in ridge_vertices_list:
            p1 = vertex_positions[i]
            p2 = vertex_positions[j]
            seg_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if seg_len <= 1e-6:
                continue
            raw_slope = self._sampled_slope(p1, p2, seg_len)
            normalized_slope = min(1.0, raw_slope / 30.0)
            cost = seg_len * (1.0 + self.plot_height_cost_factor * normalized_slope)
            ridge_edges.append((i, j, p1, p2, cost))

        if not ridge_edges:
            logger.error("Keine gueltigen Ridge-Kanten fuer Graph gefunden.")
            return False

        self.plot_nodes, self.vertex_to_plot_node = self._build_plot_node_registry(
            vertex_positions=vertex_positions,
            ridge_vertices_list=ridge_vertices_list,
            ridge_points=ridge_points,
            points=None,
            core_nodes=all_core_nodes,
        )
        self._plot_node_to_vertex = {}
        for vidx, pid in self.vertex_to_plot_node.items():
            self._plot_node_to_vertex.setdefault(pid, vidx)

        plot_node_by_id = {pn.node_id: pn for pn in self.plot_nodes}
        self.map_border_node_ids = set()
        for vidx in boundary_vertex_indices:
            pid = self.vertex_to_plot_node.get(vidx)
            if pid is None:
                continue
            plot_node = plot_node_by_id.get(pid)
            if plot_node is not None:
                plot_node.node_type = "map_border_node"
                self.map_border_node_ids.add(pid)

        self._static_vertex_positions = vertex_positions
        self._static_ridge_edges = ridge_edges
        self._static_num_vertices = num_vertices

        self.ridge_vertex_positions = vertex_positions
        self.current_ridge_edges = ridge_edges

        logger.info(
            "Voronoi (global, geklippt) gebaut: %d Vertices, %d Ridge-Kanten, %d Plotnodes, %d map_border_node.",
            num_vertices, len(ridge_edges), len(self.plot_nodes), len(self.map_border_node_ids),
        )
        return True
