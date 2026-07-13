"""
Path: tools/biome_lab/physics.py

Live-Physik ("was live passiert"): Federkraefte, Potentialfeld-Bewegung und
der eigentliche Tick-Loop. Wird jeden Timer-Tick (siehe app.py, QTimer)
einmal komplett durchlaufen.

Performance-Hinweis zum Tick-Loop (_tick): fruehere Version rief nach jedem
physics_step() sowohl _sync_core_positions() ALS AUCH _sync_core_registry()
auf. Letzteres baut das komplette core_registry-Dict inkl. aller
PlotCore-Objekte und Nachbarschaftslisten neu auf (Allokationen + O(edges)
Python-Schleife) -- obwohl sich zwischen zwei Resets nur Positionen aendern,
nie die Nachbarschaft. _sync_core_positions() allein reicht fuer den
Tick-Loop voellig aus; _sync_core_registry() wird nur noch aufgerufen, wenn
sich die Topologie tatsaechlich aendert (topology._reset_plot_nodes).
"""
import numpy as np


class PhysicsMixin:
    """Federkraefte + Bewegungsintegration + Tick-Orchestrierung. Wird von
    PlotPhysicsLab (app.py) zusammen mit den uebrigen Mixins eingebunden."""

    # -------------------------------------------------------------- Federn --
    def _clamp_vector(self, vector, max_norm):
        vec = np.asarray(vector, dtype=float)
        norm = float(np.hypot(vec[0], vec[1]))
        if norm <= 1e-12 or norm <= max_norm:
            return vec
        return vec * (max_norm / norm)

    def _safe_exp_spring_magnitude(self, deviation, stiffness, growth_rate=0.12,
                                    max_exp_argument=30.0, max_force=None):
        """Exponentielle Federkennlinie: bei kleiner Abweichung von der
        Soll-Laenge sehr schwache Kraft (sanfter Start), bei grosser
        Abweichung ueberproportional wachsende Rueckstellkraft. Der
        Exponenten-Clip verhindert Overflow bei extremen Abweichungen."""
        if max_force is None:
            max_force = self.MAX_SPRING_RESULTANT

        scaled = float(deviation) * float(growth_rate)
        scaled = float(np.clip(scaled, -max_exp_argument, max_exp_argument))

        if deviation >= 0.0:
            magnitude = float(stiffness) * (np.exp(scaled) - 1.0)
        else:
            magnitude = -float(stiffness) * (np.exp(-scaled) - 1.0)

        return float(np.clip(magnitude, -max_force, max_force))

    def _spring_force_between_points(self, pos_a, pos_b, rest_length, stiffness,
                                      growth_rate=0.12, traffic_weight=0.0):
        pa = np.asarray(pos_a, dtype=float)
        pb = np.asarray(pos_b, dtype=float)

        delta = pb - pa
        dist = float(np.hypot(delta[0], delta[1]))
        if dist <= 1e-9:
            return np.zeros(2, dtype=float), np.zeros(2, dtype=float), dist

        direction = delta / dist
        deviation = dist - float(rest_length)
        signed_magnitude = self._safe_exp_spring_magnitude(
            deviation=deviation, stiffness=stiffness, growth_rate=growth_rate,
            max_exp_argument=30.0, max_force=self.MAX_SPRING_RESULTANT)

        force_a = direction * signed_magnitude
        force_b = -force_a
        return force_a, force_b, dist

    def _spring_force_batch(self, pos_a_arr, pos_b_arr, rest_lengths, stiffness, growth_rate=0.12):
        """Vektorisierte Federkraft fuer viele Paare gleichzeitig. Numerisch
        identisch zu _spring_force_between_points."""
        pos_a_arr = np.asarray(pos_a_arr, dtype=float)
        pos_b_arr = np.asarray(pos_b_arr, dtype=float)
        rest_lengths = np.asarray(rest_lengths, dtype=float)
        delta = pos_b_arr - pos_a_arr
        dist = np.hypot(delta[:, 0], delta[:, 1])
        safe_dist = np.where(dist <= 1e-9, 1.0, dist)
        direction = delta / safe_dist[:, None]
        deviation = dist - rest_lengths
        scaled = np.clip(deviation * growth_rate, -30.0, 30.0)
        magnitude = np.where(
            deviation >= 0.0,
            stiffness * (np.exp(scaled) - 1.0),
            -stiffness * (np.exp(-scaled) - 1.0))
        magnitude = np.clip(magnitude, -self.MAX_SPRING_RESULTANT, self.MAX_SPRING_RESULTANT)
        force_a = direction * magnitude[:, None]
        force_a = np.where((dist <= 1e-9)[:, None], 0.0, force_a)
        return force_a, -force_a

    def _dynamic_rest_length(self, pos_a, pos_b, base_length, traffic_weight=0.0):
        mid = 0.5 * (np.asarray(pos_a, dtype=float) + np.asarray(pos_b, dtype=float))
        civ_here = self._civ_at_continuous(mid)

        civ_factor = max(1.0 - self.plot_civ_spacing_factor * civ_here, 0.25)
        shrink = 1.0 - min(float(traffic_weight) * self.spring_traffic_shrink, 1.0 - self.spring_min_shrink_fraction)
        shrink = max(shrink, self.spring_min_shrink_fraction)

        rest = float(base_length) * civ_factor * shrink
        return max(rest, 2.0)

    def _dynamic_rest_length_batch(self, pos_a_arr, pos_b_arr, base_lengths, traffic_weights):
        """Vektorisierte Variante von _dynamic_rest_length fuer viele Paare gleichzeitig."""
        mids = 0.5 * (np.asarray(pos_a_arr, dtype=float) + np.asarray(pos_b_arr, dtype=float))
        civ_here = self._civ_at_continuous_batch(mids)
        civ_factor = np.maximum(1.0 - self.plot_civ_spacing_factor * civ_here, 0.25)
        shrink = 1.0 - np.minimum(np.asarray(traffic_weights, dtype=float) * self.spring_traffic_shrink,
                                   1.0 - self.spring_min_shrink_fraction)
        shrink = np.maximum(shrink, self.spring_min_shrink_fraction)
        rest = np.asarray(base_lengths, dtype=float) * civ_factor * shrink
        return np.maximum(rest, 2.0)

    def _apply_spring_forces(self):
        """Sammelt alle Federkraefte (Kern<->Zentroid, Core<->Core) pro Tick
        auf. Die einzelnen Beitraege sind ueber np.add.at vektorisiert pro
        Kategorie, die Paar-Listen selbst werden aus den (kleinen, weil
        sparse besetzten) Nachbarschaftslisten der Cores aufgebaut."""
        node_ids = [node.node_id for node in self.nodes]
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        core_ids = list(self.core_registry.keys())
        core_index = {cid: i for i, cid in enumerate(core_ids)}
        core_by_id = self.core_registry

        node_force_arr = np.zeros((len(node_ids), 2), dtype=float)
        core_force_arr = np.zeros((len(core_ids), 2), dtype=float)

        # 1) Kern <-> Zentroid seiner Voronoi-Nachbarknoten (plot_nodes):
        #    zieht jeden Plotkern sanft in die Mitte seiner umgebenden
        #    Voronoi-Zelle (Lloyd-Relaxation-artig). rest_length ist bewusst
        #    0 -- der Kern SOLL im Zentroid liegen, jede Abweichung wird ueber
        #    dieselbe Exponentialkennlinie wie alle anderen Federn sanktioniert.
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
        pairs_ni = []
        pos_a_list, centroid_list = [], []
        for node in self.nodes:
            if node.node_type != "standard_plot_node":
                continue  # Wildniskerne haben keine Feder-/Feldphysik
            core = core_by_id.get(node.node_id)
            if core is None or not core.neighbor_node_ids:
                continue
            neighbor_positions = [
                plot_node_by_id[nid].node_location
                for nid in core.neighbor_node_ids if nid in plot_node_by_id
            ]
            if not neighbor_positions:
                continue
            pairs_ni.append(node_index[node.node_id])
            pos_a_list.append(node.node_location)
            centroid_list.append(np.mean(np.array(neighbor_positions, dtype=float), axis=0))

        if pairs_ni:
            pos_a_arr = np.array(pos_a_list, dtype=float)
            centroid_arr = np.array(centroid_list, dtype=float)
            rest_lengths = np.zeros(len(pairs_ni), dtype=float)
            force_a, _force_b = self._spring_force_batch(
                pos_a_arr, centroid_arr, rest_lengths, self.spacing_spring_stiffness, growth_rate=0.15)
            np.add.at(node_force_arr, pairs_ni, force_a)

        # 2) Core <-> Core Federn (statisches Voronoi-Traffic-Netz)
        if self._static_core_springs:
            pairs_a, pairs_b, base_len_list = [], [], []
            for core_id_a, core_id_b, _pair_key, base_length in self._static_core_springs:
                core_a = core_by_id.get(core_id_a)
                core_b = core_by_id.get(core_id_b)
                if core_a is None or core_b is None:
                    continue
                pairs_a.append(core_index[core_id_a])
                pairs_b.append(core_index[core_id_b])
                base_len_list.append(base_length)

            if pairs_a:
                pos_a_arr = np.array([core_by_id[core_ids[i]].location for i in pairs_a], dtype=float)
                pos_b_arr = np.array([core_by_id[core_ids[i]].location for i in pairs_b], dtype=float)
                traffic_list = [0.0] * len(pairs_a)
                rest_lengths = self._dynamic_rest_length_batch(pos_a_arr, pos_b_arr, base_len_list, traffic_list)
                force_a, force_b = self._spring_force_batch(
                    pos_a_arr, pos_b_arr, rest_lengths, self.traffic_spring_stiffness, growth_rate=0.10)
                np.add.at(core_force_arr, pairs_a, force_a)
                np.add.at(core_force_arr, pairs_b, force_b)

        node_norms = np.hypot(node_force_arr[:, 0], node_force_arr[:, 1])
        scale = np.where(node_norms > self.MAX_SPRING_RESULTANT,
                          self.MAX_SPRING_RESULTANT / np.maximum(node_norms, 1e-12), 1.0)
        node_force_arr *= scale[:, None]

        core_norms = np.hypot(core_force_arr[:, 0], core_force_arr[:, 1])
        scale = np.where(core_norms > self.MAX_SPRING_RESULTANT,
                          self.MAX_SPRING_RESULTANT / np.maximum(core_norms, 1e-12), 1.0)
        core_force_arr *= scale[:, None]

        node_forces = {nid: node_force_arr[i] for i, nid in enumerate(node_ids)}
        core_forces = {cid: core_force_arr[i] for i, cid in enumerate(core_ids)}
        return node_forces, core_forces

    # --------------------------------------------------------- Bewegung --
    def _physics_step(self):
        """Einzige gueltige physics_step()-Implementierung. Bewegt alle
        Nodes/Cores in self.nodes (inkl. Randnode-Projektion auf Kontur).
        core_registry ist nur ein Positions-Spiegel und wird separat ueber
        _sync_core_positions() aktuell gehalten. Wird ausschliesslich von
        _tick() aufgerufen."""
        if not getattr(self, "topology_ready", False):
            return

        node_forces, core_forces = self._apply_spring_forces()

        node_positions = np.array([n.node_location for n in self.nodes], dtype=float)
        field_forces_all = self._sample_field_batch(node_positions)
        zero2 = np.zeros(2, dtype=float)
        spring_forces_all = np.array(
            [node_forces.get(n.node_id, zero2) + core_forces.get(n.node_id, zero2) for n in self.nodes],
            dtype=float
        )
        total_forces_all = spring_forces_all + field_forces_all
        norms = np.hypot(total_forces_all[:, 0], total_forces_all[:, 1])
        scale = np.where(norms > self.MAX_SPRING_RESULTANT, self.MAX_SPRING_RESULTANT / np.maximum(norms, 1e-12), 1.0)
        total_forces_all *= scale[:, None]
        disp_all = total_forces_all * self.PHYSICS_TIME_STEP
        disp_norms = np.hypot(disp_all[:, 0], disp_all[:, 1])
        disp_scale = np.where(disp_norms > self.MAX_DISPLACEMENT_PER_TICK,
                               self.MAX_DISPLACEMENT_PER_TICK / np.maximum(disp_norms, 1e-12), 1.0)
        disp_all *= disp_scale[:, None]
        free_positions_all = node_positions + disp_all

        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
        displacement_by_core_id = {}

        for idx, node in enumerate(self.nodes):
            free_pos = free_positions_all[idx]
            node_type = getattr(node, "node_type", "standard_plot_node")

            if node_type == "wilderness_core":
                continue  # keine Tick-Physik, Position kommt aus _relax_wilderness_cores

            if node_type == "wilderness_node":
                cached_idx = self._wilderness_contour_cache.get(node.node_id)
                polygons = getattr(self, "_wilderness_polygons", None) or []
                if cached_idx is not None and cached_idx < len(polygons):
                    best_contour = np.asarray(polygons[cached_idx].exterior.coords, dtype=float)
                    best_idx = cached_idx
                else:
                    best_contour = None
                    best_dist = np.inf
                    best_idx = None
                    for p_idx, poly in enumerate(polygons):
                        contour = np.asarray(poly.exterior.coords, dtype=float)
                        candidate = self._nearest_point_on_polyline(free_pos, contour, closed=True)
                        dist = float(np.hypot(candidate[0] - free_pos[0], candidate[1] - free_pos[1]))
                        if dist < best_dist:
                            best_dist = dist
                            best_contour = contour
                            best_idx = p_idx
                    if best_idx is not None:
                        self._wilderness_contour_cache[node.node_id] = best_idx

                if best_contour is not None:
                    constrained = self._nearest_point_on_polyline(free_pos, best_contour, closed=True)
                    node.node_location = (float(constrained[0]), float(constrained[1]))
                else:
                    clamped = np.array([
                        np.clip(free_pos[0], 0.0, self.map_size - 1.0),
                        np.clip(free_pos[1], 0.0, self.map_size - 1.0),
                    ], dtype=float)
                    node.node_location = (float(clamped[0]), float(clamped[1]))

            elif node_type == "map_border_node":
                inset = self.MAP_EDGE_INSET
                s = float(self.map_size)
                border_rect = np.array([
                    [inset, inset],
                    [s - inset, inset],
                    [s - inset, s - inset],
                    [inset, s - inset],
                ], dtype=float)
                constrained = self._nearest_point_on_polyline(free_pos, border_rect, closed=True)
                node.node_location = (float(constrained[0]), float(constrained[1]))

            elif node_type == "city_border_node":
                settlement_id = self.boundary_owner.get(node.node_id)
                contour = self._city_contours.get(settlement_id)
                if contour is not None and len(contour) >= 2:
                    constrained = self._nearest_point_on_polyline(free_pos, np.asarray(contour, dtype=float),
                                                                   closed=True)
                    node.node_location = (float(constrained[0]), float(constrained[1]))
                else:
                    clamped = np.array([
                        np.clip(free_pos[0], 0.0, self.map_size - 1.0),
                        np.clip(free_pos[1], 0.0, self.map_size - 1.0),
                    ], dtype=float)
                    node.node_location = (float(clamped[0]), float(clamped[1]))

            else:
                clamped = np.array([
                    np.clip(free_pos[0], 0.0, self.map_size - 1.0),
                    np.clip(free_pos[1], 0.0, self.map_size - 1.0),
                ], dtype=float)
                # Harte Bewegungsgrenze: ein Plotkern darf seine eigene
                # Voronoi-Zelle (Polygon aus den AKTUELLEN Positionen seiner
                # Nachbar-plot_nodes) nie verlassen -- realistisch soll das
                # nie moeglich sein. Die Feder zum Zentroid (siehe
                # _apply_spring_forces) ist nur die weiche Vorstufe davon.
                clamped = self._contain_core_in_cell(node, clamped, plot_node_by_id)
                displacement_by_core_id[node.node_id] = clamped - node_positions[idx]
                node.node_location = (float(clamped[0]), float(clamped[1]))

        # core_registry-Positionen werden NICHT hier bewegt: core_forces ist
        # bereits oben in spring_forces_all eingerechnet (core_id == node_id,
        # core_registry ist nur ein Positions-Spiegel von self.nodes). Ein
        # zusaetzliches Bewegen von core_registry waere sofort wirkungslos,
        # da _tick() direkt danach _sync_core_positions() aufruft und
        # core.location damit ohnehin wieder 1:1 aus self.nodes ueberschreibt.

        self._drift_plot_nodes(displacement_by_core_id, plot_node_by_id)

    # -------------------------------------------------- Kern-/Node-Grenzen --
    def _point_in_polygon(self, point, polygon, shifted=None):
        """Ray-Casting-Punkt-in-Polygon-Test, rein numpy, ohne Abhaengigkeit
        von einem gecachten/eingefrorenen Path-Objekt -- die Polygone hier
        werden aus LIVE-Positionen (driftende plot_nodes) aufgebaut und
        aendern sich potenziell jeden Tick. `shifted`=(xs2, ys2) darf
        vorberechnet uebergeben werden (siehe _prepare_wilderness_polygons),
        um den np.roll-Aufbau nicht bei JEDEM Aufruf auf DERSELBEN,
        zwischen Ticks unveraenderten Kontur zu wiederholen."""
        x, y = point
        poly = np.asarray(polygon, dtype=float)
        xs, ys = poly[:, 0], poly[:, 1]
        if shifted is None:
            xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
        else:
            xs2, ys2 = shifted
        crosses = (ys > y) != (ys2 > y)
        if not np.any(crosses):
            return False
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
        return bool(np.sum(crosses & (x < x_intersect)) % 2 == 1)

    def _prepare_wilderness_polygons(self):
        """Baut fuer jedes Wildnis-Polygon einmal pro Tick (nicht einmal PRO
        plot_node!) die Segment-Arrays (a, b, xs2, ys2) auf. Diese Konturen
        aendern sich zwischen zwei Resets nie -- np.roll auf potenziell
        hundert-Punkte-Konturen war in Profilen der dominante Kostenblock von
        _contain_plot_node, weil dort pro Tick viele plot_nodes dieselbe(n)
        Kontur(en) pruefen koennen."""
        prepared = []
        for poly in (getattr(self, "_wilderness_polygons", None) or []):
            coords = np.asarray(poly.exterior.coords, dtype=float)
            xs2 = np.roll(coords[:, 0], -1)
            ys2 = np.roll(coords[:, 1], -1)
            b = np.column_stack((xs2, ys2))
            prepared.append((coords, (xs2, ys2), b))
        return prepared

    def _contain_core_in_cell(self, node, pos, plot_node_by_id):
        """Projiziert pos auf den Rand des Zell-Polygons zurueck, falls der
        Kern es verlassen wuerde. Polygon-Eckpunkte sind die (ggf. driftenden)
        AKTUELLEN Positionen der in _build_core_cell_plot_node_ids gemerkten
        Nachbar-plot_nodes -- kein statischer Snapshot.

        Sicherheitsnetz: die Korrektur wird auf ein Vielfaches von
        MAX_DISPLACEMENT_PER_TICK gedeckelt. Trotz Ausreisser-Filterung in
        _build_core_cell_plot_node_ids kann ein Zell-Polygon (v.a. an der
        Civ-/Wildnis-Dichtegrenze) noch leicht verzerrt sein -- ohne Deckel
        wuerde eine einzelne entartete Kante den Kern in einem Tick quer
        ueber die Karte teleportieren, und ueber _drift_plot_nodes auch alle
        benachbarten plot_nodes gleich mit."""
        plot_node_ids = getattr(self, "_core_cell_plot_node_ids", {}).get(node.node_id)
        if not plot_node_ids:
            return pos
        polygon = [plot_node_by_id[nid].node_location for nid in plot_node_ids if nid in plot_node_by_id]
        if len(polygon) < 3:
            return pos
        polygon = np.array(polygon, dtype=float)
        if self._point_in_polygon(pos, polygon):
            return pos

        projected = self._nearest_point_on_polyline(pos, polygon, closed=True)
        correction = projected - pos
        dist = float(np.hypot(correction[0], correction[1]))
        max_step = self.MAX_DISPLACEMENT_PER_TICK * 3.0
        if dist > max_step and dist > 1e-9:
            projected = pos + correction * (max_step / dist)
        return projected

    def _contain_plot_node(self, plot_node, pos, prepared_wilderness):
        """Haelt zivilisationsnahe plot_nodes (mind. ein Nachbarkern ist ein
        regulaerer Plotkern) innerhalb der Wildnisgrenze -- 'Seifenblasen in
        einer Keksform'. Rein wildniseigene plot_nodes (alle Nachbarkerne sind
        Wildniskerne) werden NICHT geclippt, die duerfen frei in der Wildnis
        liegen (Wege/Gebiete durch die Wildnis).

        Nutzt einen Polygon-Index-Cache (_plot_node_wilderness_cache), damit
        ein bereits an der Grenze 'klebender' Knoten nicht jeden Tick erneut
        gegen ALLE Wildnis-Polygone geprueft werden muss -- einmal am Rand
        angekommen bleibt er (numerisch) dort, die teure Suche ueber alle
        Polygone lohnt sich nur beim ersten Mal bzw. wenn der Cache-Treffer
        nicht mehr passt. prepared_wilderness kommt aus
        _prepare_wilderness_polygons() (einmal pro Tick, nicht pro Node)."""
        if not prepared_wilderness:
            return pos
        type_by_id = getattr(self, "_core_type_by_id", {})
        is_civ_adjacent = any(
            type_by_id.get(cid) == "standard_plot_node" for cid in plot_node.neighbor_core_ids
        )
        if not is_civ_adjacent:
            return pos

        cache = self._plot_node_wilderness_cache
        cached_idx = cache.get(plot_node.node_id)
        if cached_idx is not None and cached_idx < len(prepared_wilderness):
            coords, shifted, b = prepared_wilderness[cached_idx]
            if self._point_in_polygon(pos, coords, shifted):
                return pos
            return self._nearest_point_on_segments(pos, coords, b)

        best_idx, best_dist, best_point = None, np.inf, pos
        for idx, (coords, shifted, b) in enumerate(prepared_wilderness):
            if self._point_in_polygon(pos, coords, shifted):
                cache[plot_node.node_id] = idx
                return pos
            candidate = self._nearest_point_on_segments(pos, coords, b)
            dist = float(np.hypot(candidate[0] - pos[0], candidate[1] - pos[1]))
            if dist < best_dist:
                best_dist, best_idx, best_point = dist, idx, candidate
        if best_idx is not None:
            cache[plot_node.node_id] = best_idx
        return best_point

    def _drift_plot_nodes(self, displacement_by_core_id, plot_node_by_id):
        """Plot_nodes bewegen sich mit, wenn die Plotkerne geschoben werden:
        jeder plot_node driftet um den gedaempften Mittelwert der Verschiebung
        seiner Nachbarkerne (PLOT_NODE_FOLLOW_FACTOR < 1, damit die Plots nur
        'etwas aufrutschen', nicht 1:1 mitwandern). Wildnis-adjazente
        plot_nodes bleiben unbewegt (ihre Nachbarkerne sind Wildniskerne,
        Verschiebung == 0)."""
        if not displacement_by_core_id:
            return
        follow_factor = getattr(self, "PLOT_NODE_FOLLOW_FACTOR", 0.35)
        prepared_wilderness = self._prepare_wilderness_polygons()

        for plot_node in self.plot_nodes:
            deltas = [displacement_by_core_id[cid] for cid in plot_node.neighbor_core_ids
                      if cid in displacement_by_core_id]
            if not deltas:
                continue
            mean_delta = np.mean(deltas, axis=0) * follow_factor
            if abs(mean_delta[0]) < 1e-9 and abs(mean_delta[1]) < 1e-9:
                continue
            x, y = plot_node.node_location
            new_pos = np.array([x + mean_delta[0], y + mean_delta[1]], dtype=float)
            new_pos = self._contain_plot_node(plot_node, new_pos, prepared_wilderness)
            plot_node.node_location = (float(new_pos[0]), float(new_pos[1]))

    def _refresh_live_ridge_edges(self):
        """Baut current_ridge_edges/ridge_vertex_positions (nur fuer draw.py
        relevant, siehe Modul-Docstring topology.py) aus den AKTUELLEN
        plot_node-Positionen neu auf, statt dem eingefrorenen Voronoi-Snapshot
        aus dem letzten Reset -- sonst wuerden die gezeichneten Wege-Linien
        von den (jetzt driftenden) blauen Punkten abdriften. Der Dijkstra-
        Distanzgraph (traffic.py) bleibt bewusst unberuehrt: der haengt an
        self._static_*, nicht an current_ridge_edges/ridge_vertex_positions."""
        if not getattr(self, "_static_ridge_edges", None):
            return
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
        vertex_to_plot_node = getattr(self, "vertex_to_plot_node", {})

        updated_edges = []
        for i, j, p1, p2, cost in self._static_ridge_edges:
            pid_i = vertex_to_plot_node.get(i)
            pid_j = vertex_to_plot_node.get(j)
            live_p1 = plot_node_by_id[pid_i].node_location if pid_i in plot_node_by_id else p1
            live_p2 = plot_node_by_id[pid_j].node_location if pid_j in plot_node_by_id else p2
            updated_edges.append((i, j, live_p1, live_p2, cost))

        self.current_ridge_edges = updated_edges
        if self.plot_nodes:
            self.ridge_vertex_positions = np.array([n.node_location for n in self.plot_nodes], dtype=float)
        self._wiggly_path_cache = {}

    # ------------------------------------------------------------- Tick-Loop --
    def _tick(self):
        if not self.playing:
            return

        self.iteration += 1
        self._tick_timings = {}

        self._start_timer("physics_step")
        self._physics_step()
        self._stop_timer("physics_step")

        self._start_timer("sync_core_positions")
        self._sync_core_positions()  # leichtgewichtig: nur core.location aktualisieren
        self._stop_timer("sync_core_positions")

        self._start_timer("simulate_traffic")
        if self.iteration % self.TRAFFIC_RECOMPUTE_INTERVAL == 0:
            self._simulate_traffic()
            self._refresh_live_ridge_edges()
        self._stop_timer("simulate_traffic")

        self._start_timer("redraw")
        self._redraw()
        self._stop_timer("redraw")
