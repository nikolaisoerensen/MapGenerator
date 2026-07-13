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

        boundary_analyzer = CityBoundaryAnalyzer(
            self.gen.terrain_factor_villages, self.city_reach_factor, shader_manager=None)
        self.city_mask, _ = boundary_analyzer.compute_city_boundaries(
            self.heightmap, self.slopemap, self.settlements)

        # ---- Dynamische Wildnis-Grenzpunkte statt fester Knoten ----
        self._wilderness_boundary_polygon = self._build_wilderness_boundary_points()

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
        sie per buffer(0) (Standard-Shapely-Trick fuer Self-Intersections)."""
        mask = (self.civ_map >= self.WILDERNESS_CIV_THRESHOLD).astype(np.float32)
        contours = measure.find_contours(mask, level=0.5)
        polygons = []
        for c in contours:
            pts = np.column_stack([c[:, 1], c[:, 0]])  # (row,col) -> (x,y)
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
        if polygons:
            return np.vstack([np.array(p.exterior.coords) for p in polygons])
        return np.empty((0, 2), dtype=np.float64)
