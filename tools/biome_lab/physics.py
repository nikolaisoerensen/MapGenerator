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

    def _rest_length_core_plotnode_batch(self, positions):
        """Ruhelaenge fuer die Core<->PlotNode-Federn (a): civ-abhaengig ueber
        topology._spring_rest_length, civ am Kern selbst abgefragt (nicht am
        Mittelpunkt -- der Kern definiert die 'Ortsklasse' der Zelle)."""
        civ_here = self._civ_at_continuous_batch(positions)
        civ_factor = np.maximum(1.0 - self.CIV_RESTLENGTH_STEEPNESS * civ_here, 0.25)
        return np.maximum(self.plot_base_spacing * civ_factor, 2.0)

    def _rest_length_plotnode_plotnode_batch(self, pos_a_arr, pos_b_arr, traffic_values):
        """Ruhelaenge fuer die PlotNode<->PlotNode-Federn (b): civ-abhaengig
        am Kantenmittelpunkt (wie a), zusaetzlich um bis zu 30% verkuerzt bei
        hohem Traffic -- stark befahrene Wege ziehen sich direkter/gerader,
        analog zu einer echten Straße gegenueber einem Trampelpfad."""
        mids = 0.5 * (np.asarray(pos_a_arr, dtype=float) + np.asarray(pos_b_arr, dtype=float))
        civ_here = self._civ_at_continuous_batch(mids)
        civ_factor = np.maximum(1.0 - self.CIV_RESTLENGTH_STEEPNESS * civ_here, 0.25)
        base = self.plot_base_spacing * civ_factor
        shrink = 1.0 - np.minimum(np.asarray(traffic_values, dtype=float) * self.spring_traffic_shrink,
                                   1.0 - self.spring_min_shrink_fraction)
        shrink = np.maximum(shrink, self.spring_min_shrink_fraction)
        return np.maximum(base * shrink, 2.0)

    def _apply_spring_forces(self):
        """Sammelt alle Federkraefte + Innendruck pro Tick auf, fuer Kerne UND
        plot_nodes (beide sind jetzt vollwertige Physik-Koerper mit eigener
        Masse/Geschwindigkeit, siehe _physics_step). Drei Kategorien:
          A) Core<->PlotNode (civ-abhaengige Ruhelaenge > 0)
          B) PlotNode<->PlotNode (civ- UND traffic-abhaengige Ruhelaenge > 0)
          C) Innendruck je Kern-Zelle (Flaechenerhalt, nur nach aussen)
        Keine der Federn hat rest_length=0 mehr -- das war die eigentliche
        Kollaps-Ursache (siehe Debugging-Notiz zu Cluster-Kollaps: zwei
        Punktgruppen, die sich gegenseitig zum jeweils anderen Mittelwert
        ziehen, kollabieren ohne verankerte, positive Ruhelaenge)."""
        active_cores = [n for n in self.nodes if n.node_type == "standard_plot_node"]
        core_ids = [n.node_id for n in active_cores]
        core_index = {nid: i for i, nid in enumerate(core_ids)}
        node_by_id_all = {n.node_id: n for n in self.nodes}

        plot_ids = [n.node_id for n in self.plot_nodes]
        plot_index = {nid: i for i, nid in enumerate(plot_ids)}
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}

        core_force_arr = np.zeros((len(core_ids), 2), dtype=float)
        plot_force_arr = np.zeros((len(plot_ids), 2), dtype=float)

        # ---- A) Core <-> PlotNode ----
        # core_registry.neighbor_node_ids ist fuer ALLE self.nodes-Eintraege
        # befuellt (siehe topology._sync_core_registry), nicht nur aktive
        # Kerne -- Wildniskerne nehmen als UNBEWEGLICHER Anker an der Feder
        # teil (sie erhalten selbst keine Kraft, ziehen aber ihre plot_node-
        # Nachbarn), regulaere Sonder-Nodes (Boundary-Typen) werden ausge-
        # schlossen (die haben eigene Kontur-Kinematik, keine Federphysik).
        if self.enable_core_plotnode_spring:
            pairs_core_id, pairs_plot_idx = [], []
            pos_core_list, pos_plot_list = [], []
            for cid, core in self.core_registry.items():
                core_node = node_by_id_all.get(cid)
                if core_node is None or core_node.node_type not in ("standard_plot_node", "wilderness_core"):
                    continue
                for pid in core.neighbor_node_ids:
                    plot_node = plot_node_by_id.get(pid)
                    if plot_node is None:
                        continue
                    pairs_core_id.append(cid)
                    pairs_plot_idx.append(plot_index[pid])
                    pos_core_list.append(core_node.node_location)
                    pos_plot_list.append(plot_node.node_location)

            if pairs_core_id:
                pos_core_arr = np.array(pos_core_list, dtype=float)
                pos_plot_arr = np.array(pos_plot_list, dtype=float)
                rest_arr = self._rest_length_core_plotnode_batch(pos_core_arr)
                force_core, force_plot = self._spring_force_batch(
                    pos_core_arr, pos_plot_arr, rest_arr, self.core_plotnode_spring_stiffness, growth_rate=0.15)

                plot_idx_arr = np.array(pairs_plot_idx, dtype=int)
                np.add.at(plot_force_arr, plot_idx_arr, force_plot)

                active_mask = np.array([cid in core_index for cid in pairs_core_id], dtype=bool)
                if np.any(active_mask):
                    core_idx_arr = np.array(
                        [core_index[cid] for cid in pairs_core_id if cid in core_index], dtype=int)
                    np.add.at(core_force_arr, core_idx_arr, force_core[active_mask])

        # ---- B) PlotNode <-> PlotNode ----
        if self.enable_plotnode_plotnode_spring:
            vertex_by_plot_node = getattr(self, "_plot_node_to_vertex", {})
            seen_pairs = set()
            pairs_i, pairs_j = [], []
            pos_a_list, pos_b_list, traffic_list = [], [], []
            for plot_node in self.plot_nodes:
                for other_id in plot_node.neighbor_node_ids:
                    if other_id == plot_node.node_id:
                        continue
                    pair = tuple(sorted((plot_node.node_id, other_id)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    other = plot_node_by_id.get(other_id)
                    if other is None:
                        continue
                    pairs_i.append(plot_index[plot_node.node_id])
                    pairs_j.append(plot_index[other_id])
                    pos_a_list.append(plot_node.node_location)
                    pos_b_list.append(other.node_location)

                    vi = vertex_by_plot_node.get(plot_node.node_id)
                    vj = vertex_by_plot_node.get(other_id)
                    if vi is not None and vj is not None:
                        # Bewusst die langsamer nachziehende Schrumpf-EMA
                        # (nicht ridge_traffic_history, das bleibt fuer die
                        # Traffic-Einfaerbung in draw.py schnell reagierend) --
                        # verhindert den Sprung von 0 auf vollen EMA-Wert beim
                        # ersten Auftreten von Traffic auf einer Kante.
                        traffic_list.append(self.ridge_traffic_shrink_ema.get(self._edge_key(vi, vj), 0.0))
                    else:
                        traffic_list.append(0.0)

            if pairs_i:
                pos_a_arr = np.array(pos_a_list, dtype=float)
                pos_b_arr = np.array(pos_b_list, dtype=float)
                rest_arr = self._rest_length_plotnode_plotnode_batch(pos_a_arr, pos_b_arr, traffic_list)
                force_a, force_b = self._spring_force_batch(
                    pos_a_arr, pos_b_arr, rest_arr, self.plotnode_plotnode_spring_stiffness, growth_rate=0.10)
                np.add.at(plot_force_arr, pairs_i, force_a)
                np.add.at(plot_force_arr, pairs_j, force_b)

        # ---- C) Innendruck + D) Kollisionsabstossung, beide je Kern-Zelle ----
        # Federn A/B decken nur DIREKT ueber eine Voronoi-Kante benachbarte
        # Paare ab. Zwei plot_nodes derselben Zelle, die NICHT direkt
        # benachbart sind (z.B. gegenueberliegende Ecken eines Vierecks),
        # haben sonst KEINE Ruhelaenge, die sie auseinanderhaelt -- empirisch
        # beobachtet: sie koennen exakt aufeinanderfallen und darueber ihre
        # angeschlossenen Kerne mit zusammenziehen. Deshalb hier zusaetzlich
        # ALLE Paare innerhalb derselben Zelle (nicht nur Kanten-Nachbarn)
        # mit einer Mindestabstands-Abstossung versehen, skaliert am
        # lokalen ideal_radius (gleiche civ-abhaengige Groesse wie der
        # Druckterm) statt an einem globalen Fixwert -- so passt sich der
        # Mindestabstand automatisch an dichte Innenstadt- vs. weite
        # Wildnis-Zellen an.
        pressure_strength = self.pressure_strength if self.enable_pressure else 0.0
        repulsion_strength = self.plot_node_repulsion_strength if self.enable_plot_node_repulsion else 0.0
        if pressure_strength > 1e-9 or repulsion_strength > 1e-9:
            for node in active_cores:
                ids_for_cell = self._core_cell_plot_node_ids.get(node.node_id)
                if not ids_for_cell:
                    continue
                ids_present = [pid for pid in ids_for_cell if pid in plot_node_by_id]
                if len(ids_present) < 3:
                    continue
                positions = np.array(
                    [plot_node_by_id[pid].node_location for pid in ids_present], dtype=float)
                ideal_radius = self._spring_rest_length(
                    self._civ_at_continuous(node.node_location), self.plot_base_spacing)

                if pressure_strength > 1e-9:
                    area = self._polygon_area(positions)
                    ideal_area = np.pi * ideal_radius ** 2
                    if area < 1e-6:
                        pressure_mag = pressure_strength
                    else:
                        pressure_mag = pressure_strength * max(ideal_area / area - 1.0, 0.0)
                    if pressure_mag > 1e-9:
                        core_pos = np.array(node.node_location, dtype=float)
                        deltas = positions - core_pos
                        dists = np.hypot(deltas[:, 0], deltas[:, 1])
                        safe_dists = np.where(dists > 1e-9, dists, 1.0)
                        directions = deltas / safe_dists[:, None]
                        forces = directions * pressure_mag
                        idx_arr = np.array([plot_index[pid] for pid in ids_present], dtype=int)
                        np.add.at(plot_force_arr, idx_arr, forces)

                if repulsion_strength > 1e-9:
                    min_sep = max(ideal_radius * 0.6, 3.0)
                    n = len(ids_present)
                    for a_idx in range(n):
                        for b_idx in range(a_idx + 1, n):
                            delta = positions[a_idx] - positions[b_idx]
                            dist = float(np.hypot(delta[0], delta[1]))
                            if dist >= min_sep:
                                continue
                            if dist > 1e-6:
                                direction = delta / dist
                            else:
                                pid_a = ids_present[a_idx]
                                angle = (pid_a * 2654435761) % 360 * np.pi / 180.0
                                direction = np.array([np.cos(angle), np.sin(angle)])
                            safe_dist = max(dist, 0.5)
                            magnitude = repulsion_strength * (min_sep / safe_dist - 1.0)
                            magnitude = min(magnitude, self.MAX_SPRING_RESULTANT)
                            push = direction * magnitude
                            ia, ib = plot_index[ids_present[a_idx]], plot_index[ids_present[b_idx]]
                            plot_force_arr[ia] += push
                            plot_force_arr[ib] -= push

        # ---- E) Weiche Wildnisgrenze (ersetzt harten Snap in _physics_step) ----
        # _contain_plot_node projiziert auf den naehesten Punkt einer festen
        # Kontur -- eine Viele-zu-eins-Abbildung: mehrere plot_nodes vom
        # selben Randabschnitt (z.B. einer konkaven Ecke) projizieren auf
        # denselben Punkt und kollabierten dort sichtbar zu einer Linie/einem
        # Haufen, wenn das Ergebnis wie frueher jeden Tick DIREKT als neue
        # Position uebernommen wurde. Hier stattdessen nur als Zielpunkt
        # fuer eine normale Federkraft genutzt (rest_length=0, nur nach
        # innen ziehend) -- die laeuft durch dieselbe F=ma-Integration wie
        # alle anderen Kraefte, so dass z.B. die PlotNode<->PlotNode-
        # Kollisionsabstossung (Block D) gleichzeitig gegensteuern und ein
        # Zusammenlaufen auf identische Punkte verhindern kann.
        if self.enable_wilderness_containment:
            prepared_wilderness = self._prepare_wilderness_polygons()
            if prepared_wilderness:
                for plot_node in self.plot_nodes:
                    pos = np.asarray(plot_node.node_location, dtype=float)
                    target = self._contain_plot_node(plot_node, pos, prepared_wilderness, cap=False)
                    correction = target - pos
                    dist = float(np.hypot(correction[0], correction[1]))
                    if dist <= 1e-9:
                        continue
                    direction = correction / dist
                    magnitude = self._safe_exp_spring_magnitude(
                        deviation=dist, stiffness=self.wilderness_push_stiffness, growth_rate=0.15)
                    magnitude = max(magnitude, 0.0)  # nur nach innen, nie nach aussen
                    idx = plot_index[plot_node.node_id]
                    plot_force_arr[idx] += direction * magnitude

        core_norms = np.hypot(core_force_arr[:, 0], core_force_arr[:, 1])
        scale = np.where(core_norms > self.MAX_SPRING_RESULTANT,
                          self.MAX_SPRING_RESULTANT / np.maximum(core_norms, 1e-12), 1.0)
        core_force_arr *= scale[:, None]

        plot_norms = np.hypot(plot_force_arr[:, 0], plot_force_arr[:, 1])
        scale = np.where(plot_norms > self.MAX_SPRING_RESULTANT,
                          self.MAX_SPRING_RESULTANT / np.maximum(plot_norms, 1e-12), 1.0)
        plot_force_arr *= scale[:, None]

        core_forces = {nid: core_force_arr[i] for i, nid in enumerate(core_ids)}
        plot_forces = {nid: plot_force_arr[i] for i, nid in enumerate(plot_ids)}
        return core_forces, plot_forces

    # --------------------------------------------------------- Bewegung --
    def _reflect_velocity_on_correction(self, velocity, pos_before, pos_after):
        """Nach einer harten Grenzkorrektur (Zell-/Wildnis-Containment) wird
        NUR die Geschwindigkeitskomponente ENTGEGEN der Korrekturrichtung
        entfernt (inelastischer Stoss, kein voller Stillstand) -- tangentiale
        Geschwindigkeit (Entlanggleiten der Grenze) bleibt erhalten. Ohne das
        wuerde ein Koerper an seiner Grenze im naechsten Tick sofort wieder
        mit voller Geschwindigkeit dagegenlaufen."""
        correction = np.asarray(pos_after, dtype=float) - np.asarray(pos_before, dtype=float)
        dist = float(np.hypot(correction[0], correction[1]))
        if dist <= 1e-9:
            return velocity
        direction = correction / dist
        v_along = float(np.dot(velocity, direction))
        if v_along < 0.0:
            return velocity - v_along * direction
        return velocity

    def _physics_step(self):
        """Einzige gueltige physics_step()-Implementierung. Echte F=ma-
        Integration (semi-implizites/symplektisches Euler: erst Geschwindig-
        keit aus Kraft+Masse+Daempfung aktualisieren, dann Position aus
        Geschwindigkeit) fuer aktive Kerne UND plot_nodes -- beide sind
        vollwertige Massepunkte mit eigener velocity (siehe models.PlotNode).
        city_border_node bleibt bewusst ausserhalb des Feder-Masse-Systems:
        das ist eine gezielte Zwangsbedingung (Gleiten auf einer festen
        Kontur), kein freier Koerper -- seine Bewegung kommt weiterhin rein
        aus dem Potentialfeld, ohne Masse/Traegheit. map_border_node und
        wilderness_node sind seit dem Umbau auf ein einziges geklipptes
        Voronoi (siehe topology._build_voronoi_mesh/_gen_step_5_wilderness_snap)
        PERMANENT unbeweglich -- keine Kinematik, keine Federphysik, gar
        keine Bewegung mehr (Nutzer-Design: 'unbeweglich von jetzt an, keine
        Physik'); ihre Position steht endgueltig fest, sobald sie klassifiziert
        wurden. Wilderness-Kerne bekommen wie bisher nie Physik. Wird
        ausschliesslich von _tick() aufgerufen."""
        if not getattr(self, "topology_ready", False):
            return

        core_forces, plot_forces = self._apply_spring_forces()
        dt = self.PHYSICS_TIME_STEP
        max_speed = self.MAX_DISPLACEMENT_PER_TICK / dt
        zero2 = np.zeros(2, dtype=float)

        # city_border_node: seit dem Umbau auf crossing-basierte Stadtgrenz-
        # Nodes (siehe topology._gen_step_city_boundary_snap) lebt dieser Typ
        # nicht mehr in self.nodes, sondern als klassifizierter plot_node in
        # self.plot_nodes -- PERMANENT unbeweglich, genau wie map_border_node/
        # wilderness_node (siehe movable_plot_nodes-Filter weiter unten). Die
        # fruehere kinematische Potentialfeld-Zwangsbedingung entfaellt damit
        # komplett, kein Ersatz-Codepfad noetig.

        # ---- Aktive Kerne: F=ma, Masse core_mass ----
        active_cores = [n for n in self.nodes if n.node_type == "standard_plot_node"]
        if active_cores:
            plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
            core_positions = np.array([n.node_location for n in active_cores], dtype=float)
            core_velocities = np.array([n.velocity for n in active_cores], dtype=float)

            core_field = self._sample_field_batch(core_positions) if self.enable_field_cores else np.zeros_like(core_positions)
            core_total_force = (np.array([core_forces.get(n.node_id, zero2) for n in active_cores], dtype=float)
                                 + core_field)
            core_force_norms = np.hypot(core_total_force[:, 0], core_total_force[:, 1])
            core_force_scale = np.where(core_force_norms > self.MAX_SPRING_RESULTANT,
                                         self.MAX_SPRING_RESULTANT / np.maximum(core_force_norms, 1e-12), 1.0)
            core_total_force *= core_force_scale[:, None]

            core_accel = core_total_force / self.core_mass
            core_velocities = (core_velocities + core_accel * dt) * self.damping
            core_speed = np.hypot(core_velocities[:, 0], core_velocities[:, 1])
            core_speed_scale = np.where(core_speed > max_speed, max_speed / np.maximum(core_speed, 1e-12), 1.0)
            core_velocities *= core_speed_scale[:, None]

            core_free = core_positions + core_velocities * dt
            core_free[:, 0] = np.clip(core_free[:, 0], 0.0, self.map_size - 1.0)
            core_free[:, 1] = np.clip(core_free[:, 1], 0.0, self.map_size - 1.0)

            for idx, node in enumerate(active_cores):
                # Harte Bewegungsgrenze: ein Plotkern darf seine eigene
                # Voronoi-Zelle (Polygon aus den AKTUELLEN Positionen seiner
                # Nachbar-plot_nodes) nie verlassen -- realistisch soll das
                # nie moeglich sein. Die Federn (a) sind nur die weiche
                # Vorstufe davon.
                pos_before = core_free[idx]
                if self.enable_core_cell_containment:
                    pos_after = self._contain_core_in_cell(node, pos_before, plot_node_by_id)
                else:
                    pos_after = pos_before
                velocity = self._reflect_velocity_on_correction(core_velocities[idx], pos_before, pos_after)
                node.node_location = (float(pos_after[0]), float(pos_after[1]))
                node.velocity = (float(velocity[0]), float(velocity[1]))

        # core_registry-Positionen werden NICHT hier bewegt: das ist nur ein
        # Positions-Spiegel von self.nodes, siehe _sync_core_positions in
        # _tick() -- ein zusaetzliches Bewegen waere sofort wirkungslos.

        # ---- PlotNodes: F=ma, Masse plot_node_mass ----
        # map_border_node/wilderness_node/city_border_node NICHT integrieren
        # -- alle drei sind seit dem Umbau auf ein einziges geklipptes
        # Voronoi + crossing-basierte Grenz-Erkennung permanent unbeweglich
        # (siehe Docstring oben), duerfen aber weiterhin als Feder-Anker fuer
        # ihre beweglichen Nachbarn wirken (siehe _apply_spring_forces, das
        # unveraendert ALLE plot_nodes-Nachbarschaften durchgeht).
        movable_plot_nodes = [n for n in self.plot_nodes
                               if n.node_type not in ("map_border_node", "wilderness_node", "city_border_node")]
        if movable_plot_nodes:
            plot_positions = np.array([n.node_location for n in movable_plot_nodes], dtype=float)
            plot_velocities = np.array([n.velocity for n in movable_plot_nodes], dtype=float)

            plot_field = self._sample_field_batch(plot_positions) if self.enable_field_plotnodes else np.zeros_like(plot_positions)
            plot_total_force = (np.array([plot_forces.get(n.node_id, zero2) for n in movable_plot_nodes], dtype=float)
                                 + plot_field)
            plot_force_norms = np.hypot(plot_total_force[:, 0], plot_total_force[:, 1])
            plot_force_scale = np.where(plot_force_norms > self.MAX_SPRING_RESULTANT,
                                         self.MAX_SPRING_RESULTANT / np.maximum(plot_force_norms, 1e-12), 1.0)
            plot_total_force *= plot_force_scale[:, None]

            plot_accel = plot_total_force / self.plot_node_mass
            plot_velocities = (plot_velocities + plot_accel * dt) * self.damping
            plot_speed = np.hypot(plot_velocities[:, 0], plot_velocities[:, 1])
            plot_speed_scale = np.where(plot_speed > max_speed, max_speed / np.maximum(plot_speed, 1e-12), 1.0)
            plot_velocities *= plot_speed_scale[:, None]

            plot_free = plot_positions + plot_velocities * dt
            plot_free[:, 0] = np.clip(plot_free[:, 0], 0.0, self.map_size - 1.0)
            plot_free[:, 1] = np.clip(plot_free[:, 1], 0.0, self.map_size - 1.0)

            # Kein harter Positions-Snap mehr hier: die Wildnisgrenze wirkt
            # jetzt als weiche Kraft in _apply_spring_forces (Abschnitt E) --
            # siehe dortiger Kommentar zur Viele-zu-eins-Kollaps-Ursache.
            for idx, plot_node in enumerate(movable_plot_nodes):
                pos_after = plot_free[idx]
                plot_node.node_location = (float(pos_after[0]), float(pos_after[1]))
                plot_node.velocity = (float(plot_velocities[idx][0]), float(plot_velocities[idx][1]))

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
        ueber die Karte teleportieren."""
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

    def _contain_plot_node(self, plot_node, pos, prepared_wilderness, cap=True):
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
        _prepare_wilderness_polygons() (einmal pro Tick, nicht pro Node).

        cap=False wird nur fuer die einmalige Sanierung direkt nach dem Reset
        genutzt (topology._sanitize_plot_node_positions): manche plot_nodes
        starten auf einem an die Kartengrenze geklemmten, eindeutig
        ungueltigen Ausreisser-Vertex (siehe _build_plot_node_registry) --
        die volle Korrektur in einem Schritt ist hier sicher und noetig. Im
        laufenden Tick-Betrieb (cap=True) bleibt die Korrektur gedeckelt,
        damit auch ein (nach Ausreisser-Filterung) noch leicht verzerrtes
        Polygon niemals einen Tick-Sprung quer ueber die Karte ausloesen
        kann."""
        if not prepared_wilderness:
            return pos
        type_by_id = getattr(self, "_core_type_by_id", {})
        is_civ_adjacent = any(
            type_by_id.get(cid) == "standard_plot_node" for cid in plot_node.neighbor_core_ids
        )
        if not is_civ_adjacent:
            return pos

        cache = self._plot_node_wilderness_cache
        # BEKANNTE EINSCHRAENKUNG (siehe Analyse-Notiz): Projektion auf die
        # naeheste Stelle einer festen Kontur ist eine Viele-zu-eins-
        # Abbildung -- mehrere plot_nodes, die vom selben Rand-Abschnitt
        # (z.B. einer konkaven Ecke) angezogen werden, koennen auf denselben
        # Punkt projizieren. Ein kleinerer Deckel verlangsamt das nur, loest
        # es nicht (empirisch getestet: fuehrt zu MEHR betroffenen Paaren,
        # da mehr Zeit in der Naehe der Grenze verbracht wird). Ein echter
        # Fix braucht eine weiche, kraftbasierte Grenz-Rueckstellung statt
        # eines harten Snaps -- vorerst bewusst grosszuegig gedeckelt, damit
        # die Grenze wenigstens zuverlaessig eingehalten wird.
        max_step = getattr(self, "plot_base_spacing", 60.0)

        def _capped(candidate):
            if not cap:
                return candidate
            correction = candidate - pos
            dist = float(np.hypot(correction[0], correction[1]))
            if dist > max_step and dist > 1e-9:
                return pos + correction * (max_step / dist)
            return candidate

        cached_idx = cache.get(plot_node.node_id)
        if cached_idx is not None and cached_idx < len(prepared_wilderness):
            coords, shifted, b = prepared_wilderness[cached_idx]
            if self._point_in_polygon(pos, coords, shifted):
                return pos
            return _capped(self._nearest_point_on_segments(pos, coords, b))

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
        return _capped(best_point)

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
        self._stop_timer("simulate_traffic")

        # Vom Traffic-Intervall entkoppelt: das Rendering (draw.py) haengt
        # ausschliesslich an ridge_vertex_positions/current_ridge_edges, die
        # sonst bis zu TRAFFIC_RECOMPUTE_INTERVAL Ticks hinter den echten
        # plot_node-Positionen zurueckbleiben wuerden (sichtbarer "Sprung").
        # Die teure Dijkstra-Traffic-Simulation bleibt oben weiter gedrosselt.
        self._start_timer("refresh_ridge_edges")
        self._refresh_live_ridge_edges()
        self._stop_timer("refresh_ridge_edges")

        self._start_timer("redraw")
        self._redraw()
        self._stop_timer("redraw")
