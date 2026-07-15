"""
Path: tools/biome_lab/scene.py

Szenen-Aufbau ("was am Anfang passiert"): einfache Terrain-Heightmap,
Settlements, Civ-Map und die daraus abgeleitete Wildnisgrenze. Wird beim
Programmstart sowie bei jedem Klick auf "Regenerate" / "Reset Plot Nodes"
neu aufgebaut. Laeuft komplett standalone ohne DataLODManager/
GenerationOrchestrator/CalculatorDispatcher - nur eine einfache Heightmap
(reine Terrain-Rauschgenerierung, Default-Parameter aus value_default.py),
3 Settlements, civ_map und ein grobes Landschafts-Voronoi als Hintergrund.

SceneMixin haelt keinen eigenen State-Container: alle Ergebnisse haengen
als Instanzattribute (self.heightmap, self.civ_map, self.city_mask, ...) an
der PlotPhysicsLab-Instanz, damit die anderen Mixins (field, topology,
physics, draw) direkt darauf zugreifen koennen.
"""
import random as random_mod

import numpy as np
from scipy.ndimage import label, gaussian_filter, grey_closing, grey_opening
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

from core.terrain_generator import BaseTerrainGenerator
from core.settlement_generator import SettlementGenerator, TerrainSuitabilityAnalyzer, CityBoundaryAnalyzer
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


class SceneMixin:
    """Terrain/Settlement/Civ-Map-Aufbau. Wird von PlotPhysicsLab (app.py)
    zusammen mit den uebrigen Mixins eingebunden."""

    def _regenerate_scene(self, seed=None):
        if seed is None:
            seed = random_mod.randint(1, 999_999)
        self.heightmap = generate_default_heightmap(self.map_size, seed)
        dz_dy, dz_dx = np.gradient(self.heightmap)
        self.slopemap = np.stack([dz_dx, dz_dy], axis=-1).astype(np.float32)
        self.water_map = (self.heightmap < np.percentile(self.heightmap, 8)).astype(np.float32)

        suitability_analyzer = TerrainSuitabilityAnalyzer(terrain_factor_villages=1.0, map_size=self.map_size)
        suitability_map = suitability_analyzer.create_combined_suitability(
            self.heightmap, self.slopemap, self.water_map)

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
        """Baut civ_map, city_mask und die Wildnisgrenze aus der aktuellen
        Heightmap + den aktuellen Hintergrund-Slidern neu auf und markiert
        anschliessend die statische Zeichen-Ebene (draw.py) als veraltet,
        da sich Terrain-Kontur/Stadtgrenze/Wildnisgrenze geaendert haben."""
        self.city_reach_factor = 4.0 + self.city_size * 6.0
        self.civ_influence_range = 0.15 + self.city_size * 0.30
        self.plot_intercity_traffic = 20.0 + self.city_size * 40.0

        self.gen.civ_influence_range = self.civ_influence_range
        self.gen.civ_influence_decay = self.civ_influence_decay
        self.gen.city_reach_factor = self.city_reach_factor
        self.gen.terrain_factor_villages = 1.0

        self.civ_map = self.gen.calculate_civilization_mapping(
            self.heightmap, self.slopemap, self.settlements, [], [])

        self.civ_map = gaussian_filter(self.civ_map, sigma=2.5)
        self.civ_map = grey_closing(self.civ_map, size=(7, 7))
        self.civ_map = grey_opening(self.civ_map, size=(5, 5))

        self._remove_civ_islands()  # verhindert stadtlose civ-Inseln in der Wildnis

        # Regionen-Map: rein diagnostisch/informativ seit dem Umbau auf ein
        # einziges globales, geklipptes Voronoi (siehe
        # topology._build_voronoi_mesh) -- die eigentliche Netz-Konstruktion
        # partitioniert nicht mehr nach Regionen, region_map wird nur noch
        # fuer den Schritt-1-Log und die region_id-Markierung an
        # Stadtgrenz-Nodes verwendet (siehe topology._region_id_at).
        # region_map: 0 = ungueltig/nie belegt (jedes Pixel ist entweder civ
        # oder Wildnis), 1..num_civ_regions = civ-Bloecke,
        # num_civ_regions+1.. = Wildnis-Bloecke.
        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        civ_labeled, num_civ_regions = label(civ_mask)
        wild_labeled, num_wild_regions = label(~civ_mask)
        self.region_map = np.where(
            civ_mask, civ_labeled, wild_labeled + num_civ_regions
        ).astype(np.int32)
        self.num_civ_regions = int(num_civ_regions)
        self.num_wild_regions = int(num_wild_regions)

        boundary_analyzer = CityBoundaryAnalyzer(
            self.gen.terrain_factor_villages, self.city_reach_factor, shader_manager=None)
        self.city_mask, _ = boundary_analyzer.compute_city_boundaries(
            self.heightmap, self.slopemap, self.settlements)

        # ---- Dynamische Wildnis-Grenzpunkte statt fester Knoten ----
        self._wilderness_boundary_polygon = self._build_wilderness_boundary_points()
        self._city_polygons = self._build_city_boundary_polygons()

        self._compute_potential_field()
        self._mark_static_layer_dirty()

    def _remove_civ_islands(self):
        """Entfernt civ-Flaechen, die keine Stadt enthalten (Inseln in der
        Wildnis). Jede verbleibende zusammenhaengende civ-Komponente ist damit
        garantiert mit mindestens einer Stadt verbunden. Getrennte Staedte auf
        verschiedenen Kontinenten bleiben erlaubt, solange jede fuer sich
        mindestens eine Stadt einschliesst."""
        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        labeled, num_features = label(civ_mask)
        if num_features == 0:
            return

        valid_labels = set()
        h, w = labeled.shape
        for settlement in self.settlements:
            sx, sy = int(round(settlement.x)), int(round(settlement.y))
            sx = np.clip(sx, 0, w - 1)
            sy = np.clip(sy, 0, h - 1)
            lbl = labeled[sy, sx]
            if lbl > 0:
                valid_labels.add(int(lbl))
            else:
                y0, y1 = max(0, sy - 5), min(h, sy + 6)
                x0, x1 = max(0, sx - 5), min(w, sx + 6)
                nearby = labeled[y0:y1, x0:x1]
                nearby_labels = nearby[nearby > 0]
                if len(nearby_labels):
                    valid_labels.add(int(nearby_labels[0]))

        island_mask = np.isin(labeled, list(valid_labels), invert=True) & (labeled > 0)
        if np.any(island_mask):
            self.civ_map[island_mask] = self.WILDERNESS_CIV_THRESHOLD * 0.5

    def _build_wilderness_boundary_points(self):
        """Baut ein echtes Polygon der Zivilisationsflaeche via Marching-Squares.

        Marching-Squares-Konturen aus komplexen Terrain-Formen sind oft nicht
        einfach (self-intersecting) -- Shapely markiert sie dann als invalid.
        Statt solche Konturen komplett zu verwerfen (was dazu fuehrt, dass nur
        ein Teil der Wildnisgrenzen ueberhaupt Nodes bekommt), reparieren wir
        sie per buffer(0) (Standard-Shapely-Trick fuer Self-Intersections).

        Bug-Fix: find_contours schliesst eine Kontur NICHT automatisch, wenn
        die civ-Flaeche bis an den Rand des Arrays reicht -- die zurueck-
        gegebene Punktfolge ist dann OFFEN (erster != letzter Punkt). Polygon(pts)
        schliesst jeden Ring aber IMMER mit einer geraden Linie vom letzten
        zurueck zum ersten Punkt -- bei einer offenen, randberuehrenden Kontur
        wird das zu einer willkuerlichen geraden Sehne quer durchs Gelaende
        statt dem echten (am Kartenrand entlanglaufenden) Rand. Fix: mask mit
        1px Hintergrund umranden, bevor find_contours laeuft -- dann beruehrt
        JEDE Flaeche garantiert nie den Array-Rand, jede Kontur ist zwingend
        geschlossen, und die Koordinaten werden danach um den Padding-Offset
        zurueckverschoben."""
        mask = (self.civ_map >= self.WILDERNESS_CIV_THRESHOLD).astype(np.float32)
        padded_mask = np.pad(mask, 1, mode="constant", constant_values=0.0)
        contours = measure.find_contours(padded_mask, level=0.5)
        polygons = []
        for c in contours:
            pts = np.column_stack([c[:, 1] - 1.0, c[:, 0] - 1.0])  # (row,col) -> (x,y), Padding-Offset entfernt
            if len(pts) < 4:
                continue
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            candidates = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
            for cand in candidates:
                if cand.is_valid and not cand.is_empty and cand.area > self.WILDERNESS_MIN_AREA:
                    polygons.append(cand)
        self._wilderness_polygons = polygons
        # Jedes Polygon einmalig seiner civ-Region zuordnen (representative_point
        # liegt garantiert INNERHALB des Polygons, also auf der civ-Seite) --
        # nur noch fuer den Schritt-1-Diagnostik-Log genutzt (siehe
        # topology._gen_step_1_background), seit die eigentliche Netz-
        # Konstruktion nicht mehr nach Regionen partitioniert.
        region_map = getattr(self, "region_map", None)
        self._wilderness_polygon_region_ids = []
        for poly in polygons:
            region_id = -1
            if region_map is not None:
                rp = poly.representative_point()
                px = int(np.clip(round(rp.x), 0, self.map_size - 1))
                py = int(np.clip(round(rp.y), 0, self.map_size - 1))
                region_id = int(region_map[py, px])
            self._wilderness_polygon_region_ids.append(region_id)

        if polygons:
            return np.vstack([np.array(p.exterior.coords) for p in polygons])
        return np.empty((0, 2), dtype=np.float64)

    def _build_city_boundary_polygons(self):
        """Baut fuer JEDE Siedlung ein eigenes, geschlossenes Polygon ihres
        Stadtgebiets -- exakt dasselbe Marching-Squares-Verfahren wie
        _build_wilderness_boundary_points (1px Hintergrund-Padding gegen
        offene Konturen am Array-Rand, buffer(0)-Reparatur fuer Self-
        Intersections), nur pro Siedlung auf `city_mask == settlement.
        location_id` statt auf `civ_map >= Schwelle`. Ersetzt das fruehere
        Pre-Sampling einzelner Punkte um den Rasterrand (siehe ehemals
        _generate_city_boundary_nodes) -- Stadtgrenz-Nodes entstehen jetzt
        wie wilderness_node aus echten Voronoi-Kanten-Kreuzungen mit diesem
        Polygon (siehe topology._gen_step_city_boundary_snap).

        Liefert dict[settlement_id] -> Liste von Polygonen (i.d.R. genau 1,
        aber ein Stadtgebiet kann theoretisch in getrennte Blobs zerfallen,
        analog zu den mehreren moeglichen Wildnis-Polygonen)."""
        city_polygons = {}
        for settlement in self.settlements:
            sid = settlement.location_id
            mask = (self.city_mask == sid).astype(np.float32)
            if not np.any(mask):
                city_polygons[sid] = []
                continue

            padded_mask = np.pad(mask, 1, mode="constant", constant_values=0.0)
            contours = measure.find_contours(padded_mask, level=0.5)
            polygons = []
            for c in contours:
                pts = np.column_stack([c[:, 1] - 1.0, c[:, 0] - 1.0])
                if len(pts) < 4:
                    continue
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                candidates = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
                for cand in candidates:
                    if cand.is_valid and not cand.is_empty and cand.area > self.CITY_MIN_AREA:
                        polygons.append(cand)
            city_polygons[sid] = polygons
        return city_polygons
