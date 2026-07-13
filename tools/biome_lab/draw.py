"""
Path: tools/biome_lab/draw.py

Matplotlib-Rendering. Trennt zwei Kosten-Klassen strikt voneinander:

1) Statische Ebene (Terrain-Heatmap, Stadtgrenz-/Wildnisgrenz-Kontur,
   graues Grundstuecksgrenzen-Netz, Voronoi-Kreuzungspunkte, Settlements):
   haengt nur von Szene/Topologie ab, nicht von der Tick-Position der Nodes.
   Wird per ax.clear() + Neuaufbau EINMALIG erzeugt, wenn scene.py oder
   topology.py eine Aenderung melden (_mark_static_layer_dirty), nicht mehr
   bei jedem Tick.
2) Dynamische Ebene (bewegte Node-Scatter, Traffic-eingefaerbte Wege-Linien,
   Auswahl-Highlighting, Titel): wird jeden Tick per set_offsets/
   set_segments/set_array aktualisiert, OHNE die Axes zu leeren oder Artists
   neu zu erzeugen.

Diese Trennung ist der Hauptgrund, warum ein Redraw jetzt statt ~100-140ms
nur noch wenige ms braucht: ax.clear() + Neuaufbau aller Konturen/
Kollektionen war der teuerste Teil des alten redraw()s (siehe Profiling in
der PR-Beschreibung), obwohl Terrain/Konturen/Grundstuecksgrenzen sich
zwischen zwei Ticks nie aendern.

Die tatsaechliche Wellen-Geometrie einer Wege-Linie (_wiggly_path) haengt
nur von (Vertex-Positionen, Heightmap, Spline-Slidern) ab -- diese aendern
sich innerhalb eines Ticks nie -- und wird deshalb ueber
self._wiggly_path_cache zwischengespeichert statt bei jedem Redraw neu
berechnet zu werden (frueher der zweitteuerste Posten).
"""
import numpy as np
from matplotlib.collections import LineCollection


class DrawMixin:
    """Rendering + Canvas-Klick-Interaktion. Wird von PlotPhysicsLab (app.py)
    zusammen mit den uebrigen Mixins eingebunden."""

    # --------------------------------------------------- Dirty-Flags/Cache --
    def _init_render_state(self):
        self._static_layer_dirty = True
        self._overlay_dirty = True
        self._wiggly_path_cache = {}
        self._artists = {}

    def _mark_static_layer_dirty(self):
        """Von scene.py/topology.py aufgerufen, wenn sich Terrain, civ_map,
        city_mask oder die Wege-Topologie geaendert haben. Erzwingt beim
        naechsten _redraw() einen vollen Neuaufbau der statischen Ebene."""
        self._static_layer_dirty = True
        self._overlay_dirty = True
        self._wiggly_path_cache = {}

    def _mark_overlay_dirty(self):
        """Von field.py aufgerufen, wenn sich das Potentialfeld geaendert hat
        (Slider-Callback). Aktualisiert nur die Overlay-Bilder/den Quiver,
        nicht die komplette statische Ebene."""
        self._overlay_dirty = True

    # -------------------------------------------------------- Wege-Splines --
    def _wiggly_path(self, p1, p2):
        """Reine Geometrie-Berechnung einer Wege-Spline zwischen zwei Punkten
        (haengt nur von p1, p2, Heightmap und den Spline-Slidern ab). Wird
        ueber _wiggly_path_cached zwischengespeichert aufgerufen -- siehe
        Modul-Docstring."""
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

    def _wiggly_path_cached(self, i, j, p1, p2):
        """Cached-Wrapper um _wiggly_path, Schluessel = die (stabilen)
        Voronoi-Vertex-Indizes der Kante. Cache wird bei jedem statischen
        Rebuild und bei Aenderung von spline_wiggle_pct/spline_detail
        geleert (siehe ui.py)."""
        key = (i, j)
        cached = self._wiggly_path_cache.get(key)
        if cached is not None:
            return cached
        result = self._wiggly_path(tuple(p1), tuple(p2))
        self._wiggly_path_cache[key] = result
        return result

    # ---------------------------------------------------- Canvas-Interaktion --
    def _on_canvas_click(self, event):
        """Findet das naechstgelegene Element (Plotkern ODER Plotnode) zur
        Klickposition und schaltet dessen Highlighting um. Kerne haben
        Vorrang vor Plotnodes, falls beide in Reichweite liegen (rote Punkte
        sind meist weiter verteilt als die dichteren blauen Voronoi-Nodes).
        Ein Klick weit ausserhalb jedes Elements hebt beide Auswahlen auf."""
        if event.xdata is None or event.ydata is None:
            return
        pick_radius_px = 8.0

        excluded_ids = set(self.boundary_owner.keys()) | self.wilderness_node_ids | self.map_border_node_ids
        regular_nodes = [n for n in self.nodes if n.node_id not in excluded_ids]

        core_hit = None
        if regular_nodes:
            positions = np.array([n.node_location for n in regular_nodes])
            dists = np.hypot(positions[:, 0] - event.xdata, positions[:, 1] - event.ydata)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= pick_radius_px:
                core_hit = regular_nodes[best_idx].node_id

        node_hit = None
        if core_hit is None and getattr(self, "plot_nodes", None):
            positions = np.array([n.node_location for n in self.plot_nodes])
            dists = np.hypot(positions[:, 0] - event.xdata, positions[:, 1] - event.ydata)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= pick_radius_px:
                node_hit = self.plot_nodes[best_idx].node_id

        if core_hit is not None:
            self._selected_node_id = None
            self._selected_core_id = None if self._selected_core_id == core_hit else core_hit
        elif node_hit is not None:
            self._selected_core_id = None
            self._selected_node_id = None if self._selected_node_id == node_hit else node_hit
        else:
            self._selected_core_id = None
            self._selected_node_id = None

        self._redraw()

    # ------------------------------------------------------- Statische Ebene --
    def _rebuild_static_layer(self):
        """Voller ax.clear() + Neuaufbau. Nur aufgerufen, wenn
        _static_layer_dirty gesetzt ist (Szenen-/Topologie-Aenderung), nicht
        mehr bei jedem Tick."""
        ax = self.ax
        ax.clear()
        artists = self._artists = {}

        hmin = float(np.percentile(self.heightmap, 2))
        hmax = float(np.max(self.heightmap))
        terrain_vmin = hmin - 0.55 * (hmax - hmin)
        artists["terrain"] = ax.imshow(
            self.heightmap, cmap="terrain", origin="lower",
            vmin=terrain_vmin, vmax=hmax,
            extent=(0, self.map_size, 0, self.map_size), zorder=0)

        # Overlay-Platzhalter (Daten/Sichtbarkeit kommen in _refresh_overlays)
        artists["civ_overlay"] = ax.imshow(
            self.civ_map, cmap="hot", origin="lower", alpha=0.5, vmin=0.0, vmax=1.0,
            extent=(0, self.map_size, 0, self.map_size), zorder=1, visible=False)
        h, w = self.civ_map.shape
        artists["potential_overlay"] = ax.imshow(
            np.zeros((h, w)), cmap="viridis", origin="lower", alpha=0.4,
            extent=(0, self.map_size, 0, self.map_size), zorder=1, visible=False)
        artists["potential_quiver"] = None  # in _refresh_overlays angelegt

        inside = (self.city_mask >= 0).astype(float)
        if np.any(inside):
            ax.contour(inside, levels=[0.5], colors="gold", linewidths=2.0,
                       extent=(0, self.map_size, 0, self.map_size))

        wilderness = (self.civ_map < self.WILDERNESS_CIV_THRESHOLD).astype(float)
        if np.any(wilderness):
            ax.contour(wilderness, levels=[0.5], colors="cyan", linewidths=0.8, alpha=0.4,
                       extent=(0, self.map_size, 0, self.map_size))

        polygon_segments = [(p1, p2) for (i, j, p1, p2, cost) in self.current_ridge_edges]
        grey_edges = LineCollection(polygon_segments, colors="dimgray", linewidths=0.5, alpha=0.5, zorder=2)
        ax.add_collection(grey_edges)
        artists["grey_edges"] = grey_edges

        artists["traffic_active"] = LineCollection([], cmap="YlOrRd", zorder=4)
        ax.add_collection(artists["traffic_active"])
        artists["traffic_unused"] = LineCollection([], colors="lightgray", linewidths=0.35, zorder=4)
        ax.add_collection(artists["traffic_unused"])

        if self.ridge_vertex_positions is not None and len(self.ridge_vertex_positions) > 0:
            artists["voronoi_vertices"] = ax.scatter(
                self.ridge_vertex_positions[:, 0], self.ridge_vertex_positions[:, 1],
                s=6, c="blue", zorder=5, alpha=0.7)

        artists["standard_cores"] = ax.scatter([], [], s=14, c="red", zorder=6)
        artists["wilderness_cores"] = ax.scatter([], [], s=14, c="limegreen", zorder=6)
        artists["boundary_nodes"] = ax.scatter(
            [], [], s=25, c="gold", marker="^", zorder=6, edgecolors="black", linewidths=0.4)
        artists["wilderness_border_nodes"] = ax.scatter(
            [], [], s=20, c="cyan", marker="o", zorder=6, edgecolors="black", linewidths=0.3)
        artists["map_border_nodes"] = ax.scatter(
            [], [], s=22, c="dimgray", marker="s", zorder=6, edgecolors="white", linewidths=0.3)

        sx = [s.x for s in self.settlements]
        sy = [s.y for s in self.settlements]
        ax.scatter(sx, sy, s=220, c="crimson", marker="*", zorder=7, edgecolors="black")

        artists["sel_core_ring"] = ax.scatter([], [], s=90, facecolors="none",
                                               edgecolors="lime", linewidths=2.0, zorder=8)
        artists["sel_core_neighbors"] = ax.scatter([], [], s=40, c="lime",
                                                    edgecolors="black", linewidths=0.6, zorder=8)
        artists["sel_core_lines"] = LineCollection([], colors="lime", linewidths=1.2, alpha=0.9, zorder=8)
        ax.add_collection(artists["sel_core_lines"])

        artists["sel_node_ring"] = ax.scatter([], [], s=90, facecolors="none",
                                               edgecolors="magenta", linewidths=2.0, zorder=8)
        artists["sel_node_cores"] = ax.scatter([], [], s=50, c="magenta",
                                                edgecolors="black", linewidths=0.6, zorder=8)
        artists["sel_node_core_lines"] = LineCollection([], colors="magenta", linewidths=1.2, alpha=0.9, zorder=8)
        ax.add_collection(artists["sel_node_core_lines"])
        artists["sel_node_neighbors"] = ax.scatter([], [], s=40, c="deepskyblue",
                                                    edgecolors="black", linewidths=0.6, zorder=8)
        artists["sel_node_neighbor_lines"] = LineCollection(
            [], colors="deepskyblue", linewidths=1.0, alpha=0.8, zorder=8)
        ax.add_collection(artists["sel_node_neighbor_lines"])

        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_aspect("equal")
        artists["title"] = ax.set_title("")

    # ---------------------------------------------------------- Overlays --
    def _refresh_overlays(self):
        """Aktualisiert die (seltenen) Overlay-Layer: Civ-Heatmap und
        Potentialfeld-Visualisierung. Nur aufgerufen, wenn _overlay_dirty
        gesetzt ist (Checkbox-Toggle oder Feld-Neuberechnung), nicht bei
        jedem Tick."""
        artists = self._artists

        artists["civ_overlay"].set_visible(self.show_civ_overlay)
        if self.show_civ_overlay:
            artists["civ_overlay"].set_data(self.civ_map)

        show_potential = self.show_potential_overlay and self.potential_field is not None
        artists["potential_overlay"].set_visible(show_potential)
        if show_potential:
            magnitude = np.sqrt(self.potential_field[:, :, 0] ** 2 + self.potential_field[:, :, 1] ** 2)
            knee = np.percentile(magnitude, 95) if np.max(magnitude) > 0 else 1.0
            knee = max(knee, 1e-9)
            normalized = magnitude / knee
            magnitude_display = np.where(
                normalized <= 1.0, normalized, 1.0 + np.tanh(normalized - 1.0)) / 2.0
            artists["potential_overlay"].set_data(magnitude_display)
            self._refresh_potential_quiver()
        elif artists.get("potential_quiver") is not None:
            artists["potential_quiver"].set_visible(False)

    def _refresh_potential_quiver(self):
        """Legt den Potentialfeld-Quiver beim ersten Bedarf an (feste
        Positionen, da map_size innerhalb einer Szene konstant ist) und
        aktualisiert danach nur noch U/V/Farbe ueber set_UVC statt den
        Quiver komplett neu zu erzeugen."""
        step = 5
        h, w = self.potential_field.shape[:2]
        fx = self.potential_field[0:h:step, 0:w:step, 0]
        fy = self.potential_field[0:h:step, 0:w:step, 1]
        magnitude = np.sqrt(fx ** 2 + fy ** 2)

        quiver = self._artists.get("potential_quiver")
        if quiver is None:
            yy, xx = np.mgrid[0:h:step, 0:w:step]
            ARROW_LENGTH_SCALE = 6.0  # kleiner Wert = laengere Pfeile, groesser = kuerzere Pfeile
            ARROW_WIDTH = 0.0025  # Schaftdicke
            quiver = self.ax.quiver(
                xx, yy, fx, fy, magnitude,
                cmap="viridis", scale=ARROW_LENGTH_SCALE, scale_units="inches",
                width=ARROW_WIDTH, headwidth=3.0, headlength=5.0, headaxislength=4.5,
                pivot="tail", alpha=0.95, zorder=3, clim=(0.0, 1.2))
            self._artists["potential_quiver"] = quiver
        else:
            quiver.set_UVC(fx, fy, magnitude)
        quiver.set_visible(True)

    # ---------------------------------------------------------- Dynamische Ebene --
    def _update_dynamic_layer(self):
        """Aktualisiert alle Artists, die sich potenziell jeden Tick aendern
        (Node-Positionen, Traffic-Einfaerbung, Auswahl-Highlighting, Titel)
        ueber set_offsets/set_segments/set_array -- ohne Axes-Clear und ohne
        neue Artist-Objekte."""
        artists = self._artists
        MIN_TRAFFIC = 20.0 * self.plot_tier_factor
        TRAFFIC_MAX = 300.0 * self.plot_tier_factor

        # Grundstuecksgrenzen/Voronoi-Punkte folgen den (langsam driftenden,
        # siehe physics._drift_plot_nodes/_refresh_live_ridge_edges)
        # plot_node-Positionen -- deshalb hier per set_segments/set_offsets
        # aktualisiert statt nur einmalig in _rebuild_static_layer gezeichnet.
        grey_segments = [(p1, p2) for (i, j, p1, p2, cost) in self.current_ridge_edges]
        artists["grey_edges"].set_segments(grey_segments)
        if self.ridge_vertex_positions is not None and len(self.ridge_vertex_positions) > 0:
            artists["voronoi_vertices"].set_offsets(self.ridge_vertex_positions)

        strasse_count = weg_count = pfad_count = 0
        total_traffic = 0.0
        active_segs, active_norms, active_widths = [], [], []
        unused_segs = []
        for i, j, p1, p2, cost in self.current_ridge_edges:
            key = self._edge_key(i, j)
            traffic = self.ridge_traffic_history.get(key, 0.0)
            xs, ys = self._wiggly_path_cached(i, j, p1, p2)
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

        artists["traffic_active"].set_segments(active_segs)
        if active_segs:
            artists["traffic_active"].set_array(np.array(active_norms))
            artists["traffic_active"].set_linewidths(active_widths)
        artists["traffic_unused"].set_segments(unused_segs)

        excluded_ids = set(self.boundary_owner.keys()) | self.wilderness_node_ids | self.map_border_node_ids
        regular_nodes = [n for n in self.nodes if n.node_id not in excluded_ids]
        standard_cores = [n for n in regular_nodes if n.node_type != "wilderness_core"]
        wilderness_cores = [n for n in regular_nodes if n.node_type == "wilderness_core"]
        self._set_scatter_offsets(artists["standard_cores"], standard_cores)
        self._set_scatter_offsets(artists["wilderness_cores"], wilderness_cores)

        boundary_nodes = [n for n in self.nodes if n.node_id in self.boundary_owner]
        self._set_scatter_offsets(artists["boundary_nodes"], boundary_nodes)

        wilderness_border_nodes = [n for n in self.nodes if n.node_id in self.wilderness_node_ids]
        self._set_scatter_offsets(artists["wilderness_border_nodes"], wilderness_border_nodes)

        map_border_nodes = [n for n in self.nodes if n.node_id in self.map_border_node_ids]
        self._set_scatter_offsets(artists["map_border_nodes"], map_border_nodes)

        self._update_selection_highlight()

        artists["title"].set_text(
            f"Iteration {self.iteration} | Plotkerne: {len(regular_nodes)} | "
            f"Straße: {strasse_count} Weg: {weg_count} Pfad: {pfad_count} Total Traffic: {total_traffic:.1f}")

    @staticmethod
    def _set_scatter_offsets(scatter_artist, nodes):
        if nodes:
            offsets = np.array([n.node_location for n in nodes], dtype=float)
        else:
            offsets = np.empty((0, 2), dtype=float)
        scatter_artist.set_offsets(offsets)

    def _update_selection_highlight(self):
        """Aktualisiert Ring/Nachbar-Highlighting fuer die per Klick
        ausgewaehlten Kern/Node (siehe _on_canvas_click). Beide
        Artist-Gruppen werden immer auf den aktuellen Zustand gesetzt (leer,
        wenn nichts ausgewaehlt ist), statt bedingt neu erzeugt zu werden."""
        artists = self._artists

        core_ring_pos = np.empty((0, 2), dtype=float)
        core_neighbor_pos = np.empty((0, 2), dtype=float)
        core_lines = []
        if self._selected_core_id is not None and self.core_registry:
            core = self.core_registry.get(self._selected_core_id)
            if core is not None:
                cx, cy = core.location
                core_ring_pos = np.array([[cx, cy]], dtype=float)
                node_by_id = {n.node_id: n for n in self.plot_nodes}
                neighbor_pts = []
                for nid in core.neighbor_node_ids:
                    node = node_by_id.get(nid)
                    if node is None:
                        continue
                    nx, ny = node.node_location
                    neighbor_pts.append((nx, ny))
                    core_lines.append(((cx, cy), (nx, ny)))
                if neighbor_pts:
                    core_neighbor_pos = np.array(neighbor_pts, dtype=float)
        artists["sel_core_ring"].set_offsets(core_ring_pos)
        artists["sel_core_neighbors"].set_offsets(core_neighbor_pos)
        artists["sel_core_lines"].set_segments(core_lines)

        node_ring_pos = np.empty((0, 2), dtype=float)
        node_core_pos = np.empty((0, 2), dtype=float)
        node_core_lines = []
        node_neighbor_pos = np.empty((0, 2), dtype=float)
        node_neighbor_lines = []
        if self._selected_node_id is not None and self.plot_nodes:
            node_by_id = {n.node_id: n for n in self.plot_nodes}
            selected_node = node_by_id.get(self._selected_node_id)
            if selected_node is not None:
                nx, ny = selected_node.node_location
                node_ring_pos = np.array([[nx, ny]], dtype=float)

                core_pts = []
                for core_id in selected_node.neighbor_core_ids:
                    core = self.core_registry.get(core_id)
                    if core is None:
                        continue
                    cx, cy = core.location
                    core_pts.append((cx, cy))
                    node_core_lines.append(((nx, ny), (cx, cy)))
                if core_pts:
                    node_core_pos = np.array(core_pts, dtype=float)

                neighbor_pts = []
                for nid in selected_node.neighbor_node_ids:
                    neighbor = node_by_id.get(nid)
                    if neighbor is None:
                        continue
                    ox, oy = neighbor.node_location
                    neighbor_pts.append((ox, oy))
                    node_neighbor_lines.append(((nx, ny), (ox, oy)))
                if neighbor_pts:
                    node_neighbor_pos = np.array(neighbor_pts, dtype=float)

        artists["sel_node_ring"].set_offsets(node_ring_pos)
        artists["sel_node_cores"].set_offsets(node_core_pos)
        artists["sel_node_core_lines"].set_segments(node_core_lines)
        artists["sel_node_neighbors"].set_offsets(node_neighbor_pos)
        artists["sel_node_neighbor_lines"].set_segments(node_neighbor_lines)

    # ------------------------------------------------------------- Redraw --
    def _redraw(self):
        if not hasattr(self, "_artists"):
            self._init_render_state()

        if self._static_layer_dirty:
            self._rebuild_static_layer()
            self._static_layer_dirty = False
            self._overlay_dirty = True

        if self._overlay_dirty:
            self._refresh_overlays()
            self._overlay_dirty = False

        self._update_dynamic_layer()
        self.status_label.setText(f"Iteration: {self.iteration} | {self._perf_text()}")
        self.canvas.draw_idle()
