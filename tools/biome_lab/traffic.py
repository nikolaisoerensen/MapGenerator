"""
Path: tools/biome_lab/traffic.py

Traffic-Simulation ("was live passiert", periodischer Teil): rang-distanz-
gewichtete Verkehrszuweisung ueber den in topology.py vorberechneten
Dijkstra-Distanzgraphen. Laeuft nur alle PhysicsMixin.TRAFFIC_RECOMPUTE_INTERVAL
Ticks (Topologie/Distanzen aendern sich selten), nicht bei jedem Tick.
"""
import numpy as np


class TrafficMixin:
    """Traffic-Zuweisung + Hoehen-/Steigungs-Hilfsfunktionen. Wird von
    PlotPhysicsLab (app.py) zusammen mit den uebrigen Mixins eingebunden."""

    def _edge_key(self, i, j):
        """Schluessel ist das (stabile) Voronoi-Vertex-Index-Paar, NICHT die
        Position: p1/p2 in current_ridge_edges sind seit _refresh_live_ridge_
        edges LIVE (driftende plot_nodes), ein positionsbasierter Schluessel
        wuerde nie mit dem in _simulate_traffic() unter dem statischen (i,j)
        gespeicherten Traffic-Wert uebereinstimmen."""
        return tuple(sorted((int(i), int(j))))

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

    def _rank_distance_weights(self, n):
        """50%, 25%, 12.5%, ... - letzter Rang bekommt den Rest exakt."""
        if n <= 0:
            return []
        if n == 1:
            return [1.0]

        weights = []
        remaining = 1.0
        for _ in range(n - 1):
            w = remaining * 0.5
            weights.append(w)
            remaining -= w
        weights.append(remaining)
        return weights

    def _simulate_traffic(self):
        vertex_positions = self._static_vertex_positions
        ridge_edges = self._static_ridge_edges
        num_vertices = self._static_num_vertices
        predecessors = self._static_predecessors
        distances = self._static_distances
        boundary_entries = self._static_boundary_entries
        boundary_settlement = self._static_boundary_settlement
        node_entry = self._static_node_entry
        path_cache = self.path_cache

        if not self.topology_ready or vertex_positions is None or not ridge_edges:
            self.ridge_traffic_history = {}
            return

        def trace_and_add_predecessors(row_index, source_entry, target_entry, amount, contrib):
            cache_key = (row_index, source_entry, target_entry)
            cached_keys = path_cache.get(cache_key)

            if cached_keys is None:
                cached_keys = []
                current = target_entry

                while current != source_entry and current >= 0:
                    prev = predecessors[row_index, current]
                    if prev < 0:
                        break
                    if current < num_vertices and prev < num_vertices:
                        cached_keys.append(self._edge_key(current, prev))
                    current = prev

                path_cache[cache_key] = cached_keys

            for key in cached_keys:
                contrib[key] = contrib.get(key, 0.0) + amount

        city_boundary_indices = {}
        for row, (settlement_id, entry) in enumerate(zip(boundary_settlement, boundary_entries)):
            city_boundary_indices.setdefault(settlement_id, []).append((row, entry))

        fresh_contrib = {}

        for node in self.nodes:
            if node.node_id in self.boundary_owner:
                continue

            entry_idx = node_entry.get(node.node_id)
            if entry_idx is None:
                continue

            node_distances = distances[:, entry_idx]
            per_settlement_best = {}

            for settlement_id, rows in city_boundary_indices.items():
                best = None
                for row, _entry in rows:
                    d = node_distances[row]
                    if not np.isfinite(d):
                        continue
                    if best is None or d < best[0]:
                        best = (d, row)
                if best is not None:
                    per_settlement_best[settlement_id] = best

            if not per_settlement_best:
                continue

            ranked = sorted(per_settlement_best.items(), key=lambda kv: kv[1][0])
            weights = self._rank_distance_weights(len(ranked))
            traffic_weight = float(getattr(node, "traffic_weight", 4.0))

            for rank, (settlement_id, (_dist, row)) in enumerate(ranked):
                amount = weights[rank] * traffic_weight
                trace_and_add_predecessors(row, boundary_entries[row], entry_idx, amount, fresh_contrib)

        settlement_ids = [s.location_id for s in self.settlements]
        if len(settlement_ids) > 1:
            for settlement_id_from in settlement_ids:
                own_rows = [row for row, sid in enumerate(boundary_settlement) if sid == settlement_id_from]
                if not own_rows:
                    continue

                other_best = {}
                for settlement_id_to in settlement_ids:
                    if settlement_id_to == settlement_id_from:
                        continue

                    cols_to = [row for row, sid in enumerate(boundary_settlement) if sid == settlement_id_to]
                    best = None

                    for row_from in own_rows:
                        for col_row in cols_to:
                            target_entry = boundary_entries[col_row]
                            d = distances[row_from, target_entry]
                            if not np.isfinite(d):
                                continue
                            if best is None or d < best[0]:
                                best = (d, row_from, target_entry)

                    if best is not None:
                        other_best[settlement_id_to] = best

                if not other_best:
                    continue

                ranked = sorted(other_best.items(), key=lambda kv: kv[1][0])
                weights = self._rank_distance_weights(len(ranked))

                for rank, (_settlement_id_to, (_d, row_from, target_entry)) in enumerate(ranked):
                    amount = weights[rank] * self.plot_intercity_traffic
                    trace_and_add_predecessors(row_from, boundary_entries[row_from], target_entry, amount,
                                                fresh_contrib)

        traffic_decay = 0.15
        keep_factor = 1.0 - traffic_decay
        new_history = {}

        all_keys = set(self.ridge_traffic_history.keys()) | set(fresh_contrib.keys())
        for key in all_keys:
            old_value = self.ridge_traffic_history.get(key, 0.0)
            new_value = old_value * keep_factor + fresh_contrib.get(key, 0.0) * traffic_decay
            if new_value > 1e-6:
                new_history[key] = new_value

        self.ridge_traffic_history = new_history
