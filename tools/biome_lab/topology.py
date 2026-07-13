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
        _generate_static_topology() laeuft im Hauptprozess; dieser Pfad
        steht als Absturz-sichere Alternative bereit.)"""
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

    def _generate_city_boundary_nodes(self):
        city_mask = self.city_mask
        inside = city_mask > 0
        boundary = inside & (
                ~np.roll(inside, 1, axis=0)
                | ~np.roll(inside, -1, axis=0)
                | ~np.roll(inside, 1, axis=1)
                | ~np.roll(inside, -1, axis=1)
        )

        ys, xs = np.nonzero(boundary)
        target_count = int(round(3 + self.city_size * 4))

        by_settlement = {}
        for y, x in zip(ys, xs):
            settlement_id = int(city_mask[y, x])
            by_settlement.setdefault(settlement_id, []).append((float(x), float(y)))

        settlement_by_id = {s.location_id: s for s in self.settlements}
        boundary_nodes = []
        owner = {}
        self._city_contours = {}

        for settlement_id, pts in by_settlement.items():
            settlement = settlement_by_id.get(settlement_id)
            if settlement is None or not pts:
                continue

            angles = np.array([np.arctan2(y - settlement.y, x - settlement.x) for x, y in pts], dtype=float)
            order = np.argsort(angles)
            sorted_pts = [pts[i] for i in order]
            self._city_contours[settlement_id] = np.array(sorted_pts, dtype=float)

            n = len(sorted_pts)
            count = min(target_count, n)

            for k in range(count):
                x, y = sorted_pts[int(round(k * n / count)) % n]
                node = PlotNode(
                    node_id=self.next_node_id,
                    node_location=(float(x), float(y)),
                    connector_ids=[],
                    connector_distances=[],
                    connector_elevations=[],
                    connector_move_costs=[],
                    connector_edge_ids=[],
                    settlement_id=settlement_id,
                    node_type="city_border_node",
                    neighbor_core_ids=[],
                    neighbor_node_ids=[],
                )
                boundary_nodes.append(node)
                owner[node.node_id] = settlement_id
                self.next_node_id += 1

        return boundary_nodes, owner

    def _generate_wilderness_boundary_nodes(self, target_spacing=25.0):
        """Erzeugt PlotNodes entlang der Wildnisgrenze (civ-Kontur), die spaeter
        in physics._physics_step() ausschliesslich TANGENTIAL zur Kontur
        gleiten duerfen. Sampling-Abstand ~target_spacing px entlang jedes
        Polygon-Umrings, damit auch stark verschlungene Konturen gleichmaessig
        besetzt werden."""
        nodes = []
        node_ids = set()

        for poly in self._wilderness_polygons:
            coords = np.array(poly.exterior.coords, dtype=float)
            seg_lengths = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
            perimeter = float(np.sum(seg_lengths))
            if perimeter <= 1e-6:
                continue

            count = max(4, int(round(perimeter / target_spacing)))
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            sample_distances = np.linspace(0.0, perimeter, count, endpoint=False)

            for d in sample_distances:
                idx = int(np.searchsorted(cumulative, d, side="right") - 1)
                idx = int(np.clip(idx, 0, len(coords) - 2))
                seg_len = seg_lengths[idx]
                t = (d - cumulative[idx]) / seg_len if seg_len > 1e-9 else 0.0

                x = coords[idx, 0] + (coords[idx + 1, 0] - coords[idx, 0]) * t
                y = coords[idx, 1] + (coords[idx + 1, 1] - coords[idx, 1]) * t

                node = PlotNode(
                    node_id=self.next_node_id,
                    node_location=(float(x), float(y)),
                    connector_ids=[],
                    connector_distances=[],
                    connector_elevations=[],
                    connector_move_costs=[],
                    connector_edge_ids=[],
                    settlement_id=-1,
                    node_type="wilderness_node",
                    neighbor_core_ids=[],
                    neighbor_node_ids=[],
                )
                nodes.append(node)
                node_ids.add(node.node_id)
                self.next_node_id += 1

        return nodes, node_ids

    def _generate_wilderness_cores(self):
        """Samplet 'Wildniskerne' gleichmaessig (OHNE Civ-Gewichtung, civ_
        spacing_factor=0) im Wildnis-Bereich der Karte -- civ-Werte und
        Topografie sollen fuer ihre Verteilung keine Rolle spielen, nur der
        grosse, feste Zielabstand self.wilderness_core_spacing. Werden wie
        normale Plotkerne in die SPAETERE Voronoi-Berechnung einbezogen
        (siehe all_core_nodes in _generate_static_topology), bekommen aber
        nie Feder-/Feldphysik (siehe physics._physics_step) -- ihre finale
        Position kommt einmalig aus _relax_wilderness_cores()."""
        city_inside = self.city_mask > 0
        valid_mask = (self.civ_map < self.WILDERNESS_CIV_THRESHOLD) & (~city_inside)
        wilderness_pixel_area = int(np.count_nonzero(valid_mask))
        if wilderness_pixel_area == 0:
            return []

        spacing = max(float(self.wilderness_core_spacing), 2.5)
        target_count = max(3, int(round(wilderness_pixel_area / (spacing ** 2))))

        positions = self._best_candidate_sample(
            valid_mask=valid_mask,
            count=target_count,
            base_spacing=spacing,
            civ_spacing_factor=0.0,
            civ_map=self.civ_map,
        )

        nodes = []
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
                node_type="wilderness_core",
                neighbor_core_ids=[],
                neighbor_node_ids=[],
                traffic_weight=random_mod.uniform(1.0, 2.0),
            )
            nodes.append(node)
            self.next_node_id += 1
        return nodes

    def _relax_wilderness_cores(self):
        """Setzt jeden Wildniskern EINMALIG in die Mitte seiner Voronoi-Zelle:
        Zentroid aus seinen benachbarten plot_nodes, ergaenzt um nahe
        Kartengrenz-Nodes (die er sich mit angrenzenden Plotkernen teilt).
        Muss VOR dem finalen _generate_static_topology()-Aufruf laufen, da
        sich danach die Voronoi-Zellen dieser Kerne (und ihrer Nachbarn)
        aendern -- siehe Aufrufreihenfolge in _reset_plot_nodes(). Wildnis-
        kerne bewegen sich NIE per Tick-Physik, nur hier, einmal beim Reset."""
        node_by_id = {n.node_id: n for n in self.nodes}
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}

        map_border_positions = np.array(
            [node_by_id[nid].node_location for nid in self.map_border_node_ids
             if nid in node_by_id],
            dtype=float,
        ) if self.map_border_node_ids else np.empty((0, 2), dtype=float)
        border_tree = cKDTree(map_border_positions) if len(map_border_positions) else None
        radius = max(float(self.wilderness_core_spacing) * 1.5, 5.0)

        for node in self.nodes:
            if node.node_type != "wilderness_core":
                continue
            core = self.core_registry.get(node.node_id)
            if core is None:
                continue

            neighbor_positions = [
                plot_node_by_id[nid].node_location
                for nid in core.neighbor_node_ids if nid in plot_node_by_id
            ]
            if border_tree is not None:
                nearby_idx = border_tree.query_ball_point(node.node_location, radius)
                neighbor_positions.extend(map_border_positions[i] for i in nearby_idx)

            if not neighbor_positions:
                continue

            centroid = np.mean(np.array(neighbor_positions, dtype=float), axis=0)
            node.node_location = (float(centroid[0]), float(centroid[1]))

    def _generate_map_border_nodes(self, target_spacing=25.0):
        """Erzeugt PlotNodes entlang des Kartenrand-Rechtecks (MAP_EDGE_INSET
        nach innen versetzt), die in physics._physics_step() nur entlang
        dieser Rand-Linie gleiten duerfen."""
        inset = self.MAP_EDGE_INSET
        size = float(self.map_size)

        x0, x1 = inset, size - inset
        y0, y1 = inset, size - inset

        perimeter = 2.0 * ((x1 - x0) + (y1 - y0))
        count = max(8, int(round(perimeter / target_spacing)))

        nodes = []
        node_ids = set()

        for k in range(count):
            d = perimeter * k / count

            if d < (x1 - x0):
                x, y = x0 + d, y0
            elif d < (x1 - x0) + (y1 - y0):
                x, y = x1, y0 + (d - (x1 - x0))
            elif d < 2.0 * (x1 - x0) + (y1 - y0):
                x, y = x1 - (d - ((x1 - x0) + (y1 - y0))), y1
            else:
                x, y = x0, y1 - (d - (2.0 * (x1 - x0) + (y1 - y0)))

            node = PlotNode(
                node_id=self.next_node_id,
                node_location=(float(x), float(y)),
                connector_ids=[],
                connector_distances=[],
                connector_elevations=[],
                connector_move_costs=[],
                connector_edge_ids=[],
                settlement_id=-1,
                node_type="map_border_node",
                neighbor_core_ids=[],
                neighbor_node_ids=[],
            )
            nodes.append(node)
            node_ids.add(node.node_id)
            self.next_node_id += 1

        return nodes, node_ids

    def _nearest_point_on_polyline(self, pos, polyline, closed=True):
        """Instanz-Wrapper um die modul-globale, vektorisierte Funktion (siehe
        oben) -- so bleibt der Aufrufstil ``self._nearest_point_on_polyline(...)``
        fuer physics.py identisch zur fruehreren Implementierung."""
        return nearest_point_on_polyline(pos, polyline, closed=closed)

    def _nearest_point_on_segments(self, pos, a, b):
        """Instanz-Wrapper um nearest_point_on_segments (siehe oben)."""
        return nearest_point_on_segments(pos, a, b)

    def _reset_plot_nodes(self):
        self._recompute_background()
        self.next_node_id = 0

        self.nodes = []
        self.plot_nodes = []
        self.vertex_to_plot_node = {}
        self.core_registry = {}

        self.boundary_owner = {}
        self._city_contours = {}

        self.wilderness_node_ids = set()
        self.map_border_node_ids = set()
        self._wilderness_contour_cache = {}
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
        self._static_core_springs = []

        self.ridge_traffic_history = {}
        self.current_ridge_edges = []
        self.ridge_vertex_positions = None
        self.path_cache = {}

        self.selected_core_id = None
        self.selected_node_id = None
        self.iteration = 0
        self.topology_ready = False

        min_buffer_to_city_px = 10.0
        city_inside = self.city_mask > 0
        dist_to_city = distance_transform_edt(~city_inside)

        valid_mask = (
                (self.civ_map > 0.2)
                & (~city_inside)
                & (dist_to_city >= min_buffer_to_city_px)
        )

        positions = self._best_candidate_sample(
            valid_mask=valid_mask,
            count=int(self.plot_nodes_count),
            base_spacing=self.plot_base_spacing,
            civ_spacing_factor=self.plot_civ_spacing_factor,
            civ_map=self.civ_map,
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

        # Wildniskerne DIREKT nach den regulaeren Plotkernen anhaengen (und
        # VOR den Sonder-Nodes unten): all_core_nodes in
        # _generate_static_topology() filtert per node_type, aber core_id_a/
        # core_id_b in den Core<->Core-Federn sind Indizes INNERHALB von
        # all_core_nodes -- die bleiben nur dann identisch zu self.nodes-
        # Indizes/node_id, wenn alle Kern-Typen einen zusammenhaengenden
        # Praefix von self.nodes bilden (siehe physics._apply_spring_forces,
        # das ueber core_registry/node_id nachschlaegt).
        wilderness_core_nodes = self._generate_wilderness_cores()
        self.nodes.extend(wilderness_core_nodes)

        city_boundary_nodes, boundary_owner = self._generate_city_boundary_nodes()
        self.nodes.extend(city_boundary_nodes)
        self.boundary_owner.update(boundary_owner)

        wilderness_nodes, wilderness_node_ids = self._generate_wilderness_boundary_nodes()
        self.nodes.extend(wilderness_nodes)
        self.wilderness_node_ids = set(wilderness_node_ids)

        map_border_nodes, map_border_node_ids = self._generate_map_border_nodes()
        self.nodes.extend(map_border_nodes)
        self.map_border_node_ids = set(map_border_node_ids)

        if not self._generate_static_topology():
            self.topology_ready = False
            self._mark_static_layer_dirty()
            self._redraw()
            return

        if wilderness_core_nodes:
            # Lloyd-Relaxation in einer Iteration: erster Durchlauf liefert
            # die Nachbarschaft (core_registry.neighbor_node_ids), danach
            # werden Wildniskerne auf ihren Zellmittelpunkt gesetzt und die
            # Topologie mit den finalen Positionen neu berechnet.
            self._sync_core_registry()
            self._relax_wilderness_cores()
            if not self._generate_static_topology():
                self.topology_ready = False
                self._mark_static_layer_dirty()
                self._redraw()
                return

        self.topology_ready = True
        self._sync_core_registry()
        self._core_type_by_id = {node.node_id: node.node_type for node in self.nodes}
        self._build_core_cell_plot_node_ids()
        self._mark_static_layer_dirty()
        self._redraw()

    def _build_core_cell_plot_node_ids(self):
        """Merkt sich fuer jeden regulaeren Plotkern (standard_plot_node) die
        nach Winkel sortierten IDs seiner benachbarten plot_nodes -- das
        Polygon aus deren AKTUELLEN (ggf. je Tick driftenden, siehe
        physics._drift_plot_nodes) Positionen ist die harte Bewegungsgrenze
        in physics._physics_step, damit ein Kern nie ueber seine eigene
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
        alle regulaeren Nodes (keine Boundary-/Wilderness-/Kartenrand-Nodes)
        MINUS die als 'wilderness_core' markierten. Wilderness-Cores bleiben
        unbeweglich, tauchen aber weiterhin in Voronoi-Topologie und
        Traffic-Simulation auf."""
        excluded_ids = set(self.boundary_owner.keys()) | self.wilderness_node_ids | self.map_border_node_ids
        return [
            node for node in self.nodes
            if node.node_id not in excluded_ids and node.node_type != "wilderness_core"
        ]

    def _build_plot_node_registry(self, vertex_positions, ridge_vertices_list, ridge_points, points, core_nodes):
        """Baut die plot_nodes (Voronoi-Kreuzungen) samt Nachbarschaftslisten.

        Voronoi-Vertices nahe der konvexen Huelle des Punkt-Sets koennen
        numerisch riesige (aber technisch 'endliche', also nicht i/j==-1)
        Koordinaten haben -- Umkreismittelpunkte fast-kollinearer/sehr
        stumpfer Dreiecke schiessen quasi Richtung Unendlich. Das war schon
        immer so, war aber frueher harmlos, weil plot_nodes nie bewegt wurden
        und ausserhalb der festen Achsen-Limits einfach unsichtbar blieben.
        Seit physics._drift_plot_nodes/_contain_plot_node plot_nodes aktiv
        bewegen und pruefen, wuerde so ein Ausreisser Kerne/andere plot_nodes
        in einem Tick quer ueber die Karte reissen -- deshalb hier auf einen
        grosszuegigen Rand um die Karte geklemmt, BEVOR irgendwer damit
        rechnet."""
        registry = {}
        vertex_to_plot_node = {}
        clamp_margin = float(self.map_size) * 0.5

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
                pos = (
                    float(np.clip(raw_pos[0], -clamp_margin, self.map_size + clamp_margin)),
                    float(np.clip(raw_pos[1], -clamp_margin, self.map_size + clamp_margin)),
                )
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
            idx: PlotCore(core_id=idx, location=tuple(node.node_location))
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

    def _generate_static_topology(self):
        active_core_nodes = self._active_physics_cores()
        all_core_nodes = [node for node in self.nodes if node.node_type in ("standard_plot_node", "wilderness_core")]

        if len(all_core_nodes) < 2:
            logger.error("Zu wenige Kern-Nodes fuer Topologie.")
            return False

        points = np.array([node.node_location for node in all_core_nodes], dtype=float)

        try:
            vor = Voronoi(points)
        except Exception as e:
            logger.error("Voronoi-Berechnung fehlgeschlagen: %s", e)
            return False

        vertex_positions = np.asarray(vor.vertices, dtype=float)
        ridge_vertices_list = list(vor.ridge_vertices)
        ridge_points = list(vor.ridge_points)
        num_vertices = len(vertex_positions)

        if num_vertices == 0:
            logger.error("Voronoi lieferte keine Vertices.")
            return False

        ridge_edges = []
        for ridge_idx, ridge in enumerate(ridge_vertices_list):
            if len(ridge) != 2:
                continue

            i, j = ridge
            if i < 0 or j < 0:
                continue
            if i >= num_vertices or j >= num_vertices:
                continue

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

        tree = cKDTree(vertex_positions)

        node_entry = {}
        entry_edges = []

        for idx, node in enumerate(self.nodes):
            try:
                region_idx = vor.point_region[min(idx, len(vor.point_region) - 1)]
                region = vor.regions[region_idx] if region_idx < len(vor.regions) else []
                finite_vertices = [v for v in region if v >= 0]

                node_pos = np.array(node.node_location, dtype=float)

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
                logger.warning("Fehler bei Node-Entry fuer node_id=%s: %s", node.node_id, e)

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

        boundary_entries = []
        boundary_settlement = []

        for node_id, settlement_id in self.boundary_owner.items():
            if node_id in node_entry:
                boundary_entries.append(node_entry[node_id])
                boundary_settlement.append(settlement_id)

        if not boundary_entries:
            logger.error("Keine Boundary-Entries gefunden, Topologie unvollstaendig.")
            return False

        try:
            graph = csr_matrix((costs, (rows, cols)), shape=(total_graph_nodes, total_graph_nodes))
            distances, predecessors = dijkstra(
                graph,
                indices=boundary_entries,
                return_predecessors=True,
            )
        except Exception as e:
            logger.error("Initiale Dijkstra-Berechnung fehlgeschlagen: %s", e)
            return False

        self._static_vertex_positions = vertex_positions
        self._static_ridge_edges = ridge_edges
        self._static_num_vertices = num_vertices
        self._static_boundary_entries = boundary_entries
        self._static_boundary_settlement = boundary_settlement
        self._static_node_entry = node_entry
        self._static_predecessors = predecessors
        self._static_distances = distances

        self.ridge_vertex_positions = vertex_positions
        self.current_ridge_edges = ridge_edges

        self.plot_nodes, self.vertex_to_plot_node = self._build_plot_node_registry(
            vertex_positions=vertex_positions,
            ridge_vertices_list=ridge_vertices_list,
            ridge_points=ridge_points,
            points=points,
            core_nodes=all_core_nodes,
        )

        self._static_core_springs = []

        # core_id_a/core_id_b sind Indizes in `points` (== all_core_nodes),
        # unveraendert aus ridge_points/neighbor_core_ids uebernommen. Die
        # Rest-Laenge wird deshalb direkt aus `points` berechnet statt aus
        # self.core_registry: Letzteres ist an dieser Stelle noch leer (wird
        # erst danach in _reset_plot_nodes() ueber _sync_core_registry()
        # befuellt) und lieferte hier vorher immer None, wodurch
        # self._static_core_springs dauerhaft leer blieb.
        seen_core_pairs = set()
        for node in self.plot_nodes:
            core_ids = sorted(set(node.neighbor_core_ids))
            if len(core_ids) < 2:
                continue

            for a_idx in range(len(core_ids)):
                for b_idx in range(a_idx + 1, len(core_ids)):
                    core_id_a = core_ids[a_idx]
                    core_id_b = core_ids[b_idx]
                    pair = tuple(sorted((core_id_a, core_id_b)))
                    if pair in seen_core_pairs:
                        continue
                    seen_core_pairs.add(pair)

                    if core_id_a >= len(points) or core_id_b >= len(points):
                        continue

                    pos_a = points[core_id_a]
                    pos_b = points[core_id_b]
                    rest_length = float(np.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]))
                    if rest_length <= 1e-9:
                        continue

                    self._static_core_springs.append((core_id_a, core_id_b, pair, rest_length))

        logger.info(
            "Statische Topologie generiert: %d Vertices, %d Ridge-Kanten, %d Plotnodes, %d Core-Core-Federn.",
            num_vertices,
            len(ridge_edges),
            len(self.plot_nodes),
            len(self._static_core_springs),
        )
        return True
