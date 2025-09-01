"""
Water System Generation
Handles river networks, lakes, and hydrological flow simulation
Includes cached flow calculations and water erosion
"""

import numpy as np
from typing import Dict
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt

class RiverSimulator:
    """Memory-optimierte Fluss-Simulation - UNVERÄNDERT außer Cache-Management"""

    def __init__(self, heightmap: np.ndarray, rain_map: np.ndarray,
                 sea_level: float = 0.15, rain_scale: float = 10.0,
                 lake_threshold: float = 0.5):
        self.heightmap = heightmap
        self.rain_map = rain_map
        self.sea_level = sea_level
        self.rain_scale = rain_scale
        self.lake_threshold = lake_threshold
        self.h, self.w = heightmap.shape

        # Cache-Variablen
        self._flow_direction = None
        self._flow_accumulation = None
        self._height_hash = None
        self._rain_hash = None

    def _invalidate_cache_if_needed(self):
        """Prüft Änderungen an heightmap oder rain_map - UNVERÄNDERT"""
        current_height_hash = hash(self.heightmap.tobytes())
        current_rain_hash = hash((self.rain_map.tobytes(), self.rain_scale))
        if (self._height_hash != current_height_hash or
                self._rain_hash != current_rain_hash):
            self._flow_direction = None
            self._flow_accumulation = None
            self._cache_valid = False
            self._height_hash = current_height_hash
            self._rain_hash = current_rain_hash

    def calculate_flow_direction(self) -> np.ndarray:
        """Cached Flow Direction Berechnung - UNVERÄNDERT"""
        self._invalidate_cache_if_needed()

        if self._flow_direction is not None:
            return self._flow_direction

        flow_dir = np.zeros((self.h, self.w), dtype=np.int8)
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        for y in range(self.h):
            for x in range(self.w):
                if y == 0 or y == self.h - 1 or x == 0 or x == self.w - 1:
                    flow_dir[y, x] = -1
                    continue

                max_slope = 0
                best_direction = 0
                current_height = self.heightmap[y, x]

                for i, (dy, dx) in enumerate(directions):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        slope = (current_height - self.heightmap[ny, nx]) / (1.0 if (dy == 0 or dx == 0) else 1.414)
                        if slope > max_slope:
                            max_slope = slope
                            best_direction = i

                flow_dir[y, x] = best_direction

        self._flow_direction = flow_dir
        return flow_dir

    def generate_flow_accumulation(self, iterations: int = 50) -> np.ndarray:
        """In-Place Flow Accumulation - UNVERÄNDERT"""
        self._invalidate_cache_if_needed()

        if self._flow_accumulation is not None:
            return self._flow_accumulation

        flow_acc = self.calculate_water_volume()
        flow_dir = self.calculate_flow_direction()
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        height_indices = np.unravel_index(
            np.argsort(self.heightmap, axis=None)[::-1],
            self.heightmap.shape
        )

        for y, x in zip(height_indices[0], height_indices[1]):
            if flow_dir[y, x] == -1:
                continue

            direction = flow_dir[y, x]
            dy, dx = directions[direction]
            target_y, target_x = y + dy, x + dx

            if 0 <= target_y < self.h and 0 <= target_x < self.w:
                flow_acc[target_y, target_x] += flow_acc[y, x]

        self._flow_accumulation = flow_acc
        self._cache_valid = True
        return flow_acc

    def calculate_water_volume(self) -> np.ndarray:
        """Berechnet Wasservolumen - UNVERÄNDERT"""
        return self.rain_map * self.rain_scale

    def find_depressions(self) -> np.ndarray:
        """Watershed-basierte Seenerkennung - UNVERÄNDERT"""
        from scipy.ndimage import watershed_ift
        from scipy import ndimage

        markers = np.zeros_like(self.heightmap, dtype=np.int32)
        markers[self.heightmap < np.percentile(self.heightmap, 20)] = 1

        watersheds = watershed_ift(self.heightmap.astype(np.uint8), markers)
        depression_sizes = ndimage.sum(watersheds == 1, watersheds, range(2))

        return (watersheds == 1) & (depression_sizes[watersheds] > 10)

    def adjust_terrain_for_lakes(self, heightmap: np.ndarray, lakes: np.ndarray) -> np.ndarray:
        """Glättet Terrain für Seen - UNVERÄNDERT"""
        from scipy import ndimage

        smoothed = ndimage.gaussian_filter(heightmap, sigma=1.5)
        adjusted = heightmap - (lakes * 0.5)
        result = np.where(lakes, smoothed, heightmap)
        result = np.where(lakes, adjusted, result)
        return result

    def create_river_network(self, min_flow_threshold: float = 0.1, max_flow_threshold: float = 0.9) -> Dict[
        str, np.ndarray]:
        """Optimierte Version mit regionalen Thresholds - UNVERÄNDERT"""
        self._invalidate_cache_if_needed()

        flow_acc = self.generate_flow_accumulation()
        flow_normalized = self._normalize_array(flow_acc)

        threshold_map = np.zeros_like(flow_normalized)
        for x in range(self.w):
            west_factor = x / self.w
            threshold_map[:, x] = min_flow_threshold * (1 + west_factor * 2)

        streams = (flow_normalized > threshold_map) & (flow_normalized < 0.4)
        lakes = self.find_depressions() & (flow_normalized > min_flow_threshold / 2)
        rivers = (flow_normalized >= 0.4) & (flow_normalized < max_flow_threshold)
        major_rivers = flow_normalized >= max_flow_threshold

        return {
            'streams': streams.astype(float),
            'rivers': rivers.astype(float),
            'major_rivers': major_rivers.astype(float),
            'lakes': lakes.astype(float),
            'all_water': ((flow_normalized > min_flow_threshold) | lakes).astype(float)
        }

    def create_hierarchical_water_lines(self, min_flow_threshold=0.1, max_flow_threshold=0.9):
        flow_acc = self.generate_flow_accumulation()
        flow_normalized = self._normalize_array(flow_acc)

        # Verschiedene Wassertypen mit unterschiedlichen Eigenschaften
        water_lines = {
            'stream_coords': [],
            'river_coords': [],
            'major_river_coords': [],
            'stream_widths': [],
            'river_widths': [],
            'major_river_widths': []
        }

        height, width = flow_normalized.shape

        for y in range(height):
            for x in range(width):
                flow_val = flow_normalized[y, x]

                if flow_val > min_flow_threshold:
                    # Bestimme Wassertyp und Breite
                    if flow_val >= max_flow_threshold:
                        water_lines['major_river_coords'].append((x, y))
                        water_lines['major_river_widths'].append(3.0 + flow_val * 2.0)
                    elif flow_val >= 0.4:
                        water_lines['river_coords'].append((x, y))
                        water_lines['river_widths'].append(1.5 + flow_val * 1.5)
                    else:
                        water_lines['stream_coords'].append((x, y))
                        water_lines['stream_widths'].append(0.5 + flow_val * 1.0)

        return water_lines

    def calculate_valley_moisture_enhancement(self, base_rain_map, flow_accumulation):
        # Täler sammeln und leiten Feuchtigkeit weiter (Punkt 19)
        flow_normalized = self._normalize_array(flow_accumulation)

        # Erweitere Täler um ihre unmittelbare Umgebung
        valley_mask = flow_normalized > 0.1
        expanded_valleys = binary_dilation(valley_mask, iterations=2)

        # Feuchtigkeits-Enhancement basierend auf Tal-Nähe
        valley_distance = distance_transform_edt(~valley_mask)
        moisture_enhancement = np.exp(-valley_distance / 3.0)  # Exponentieller Abfall

        # Verstärke Regen in und um Täler
        enhanced_rain = base_rain_map * (1.0 + moisture_enhancement * 0.5)

        return enhanced_rain

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        """Hilfsfunktion für Normalisierung - UNVERÄNDERT"""
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)