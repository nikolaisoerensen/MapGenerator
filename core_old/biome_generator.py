"""
Biome Classification System
Classifies terrain into biomes based on elevation, temperature, and precipitation
Supports Gaussian weight-based classification and supersampling
"""

import numpy as np
from scipy.ndimage import binary_dilation
from dataclasses import dataclass

class Biome:
    def __init__(self, name, index, color, height_center, height_width,
                 rain_center, rain_width, temp_center, temp_width):
        self.name = name
        self.index = index
        self.color = color
        self.height_center = height_center
        self.height_width = height_width
        self.rain_center = rain_center
        self.rain_width = rain_width
        self.temp_center = temp_center
        self.temp_width = temp_width

    def get_weight(self, height, rain, temp):
        """Berechnet Gaußsche Gewichtung für dieses Biom"""
        h_weight = np.exp(-0.5 * ((height - self.height_center) / self.height_width) ** 2)
        r_weight = np.exp(-0.5 * ((rain - self.rain_center) / self.rain_width) ** 2)
        t_weight = np.exp(-0.5 * ((temp - self.temp_center) / self.temp_width) ** 2)
        return h_weight * r_weight * t_weight

# Neue Biom-Definitionen mit Gaußschen Parametern
biomes = [
    Biome("ocean", 0, "#000080", 0.05, 0.08, 0.5, 0.3, 0.5, 0.3),
    Biome("beach", 1, "#fffff0", 0.12, 0.03, 0.4, 0.2, 0.6, 0.2),
    Biome("alpine", 2, "#e8f0f0", 0.92, 0.08, 0.5, 0.3, 0.3, 0.2),
    Biome("tundra", 3, "#b8dbc9", 0.5, 0.25, 0.2, 0.15, 0.15, 0.15),
    Biome("taiga", 4, "#3b7a57", 0.6, 0.3, 0.8, 0.2, 0.2, 0.15),
    Biome("desert", 5, "#fff200", 0.4, 0.2, 0.1, 0.1, 0.8, 0.2),
    Biome("swamp", 6, "#a1a875", 0.3, 0.15, 0.9, 0.1, 0.7, 0.2),
    Biome("grass", 7, "#7cfc00", 0.4, 0.2, 0.3, 0.15, 0.6, 0.2),
    Biome("forest", 8, "#228b22", 0.5, 0.25, 0.7, 0.2, 0.5, 0.2),
    Biome("steppe", 9, "#c2b280", 0.45, 0.2, 0.45, 0.15, 0.55, 0.2),
    ]

class BiomeClassifier:
    def __init__(self, world, temperature_map: np.ndarray, rain_map: np.ndarray):
        self.world = world
        self.heightmap = world.heightmap
        self.temperature_map = temperature_map
        self.rain_map = rain_map

    def _is_connected_to_border(self):
        """Flood-fill vom Rand für Ocean-Bereiche"""
        ocean_mask = self.heightmap <= self.world.scale * 0.15

        seeds = np.zeros_like(ocean_mask, dtype=bool)
        seeds[0, :] = seeds[-1, :] = seeds[:, 0] = seeds[:, -1] = True
        seeds = seeds & ocean_mask

        from scipy.ndimage import binary_dilation
        connected = seeds.copy()
        for _ in range(max(self.heightmap.shape)):
            new_connected = binary_dilation(connected) & ocean_mask
            if np.array_equal(connected, new_connected):
                break
            connected = new_connected

        return connected

    def _is_adjacent_to_ocean(self, y, x, biome_map):
        """Prüft Ocean-Nachbarschaft"""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < biome_map.shape[0] and
                        0 <= nx < biome_map.shape[1] and
                        biome_map[ny, nx] == 0):
                    return True
        return False

    def generate_supersampled_biome_map(self) -> np.ndarray:
        """Generiert 2x2 Supersampling"""
        ocean_connected = self._is_connected_to_border()
        h, w = self.heightmap.shape
        super_map = np.zeros((h * 2, w * 2), dtype=int)

        for y in range(h):
            for x in range(w):
                if ocean_connected[y, x]:
                    super_map[y * 2:y * 2 + 2, x * 2:x * 2 + 2] = 0
                    continue

                elevation = self.heightmap[y, x] / self.world.scale
                rain = self.rain_map[y, x]
                temp = self.temperature_map[y, x]

                weights = []
                for i, biome in enumerate(biomes[1:], 1):
                    weight = biome.get_weight(elevation, rain, temp)
                    if weight > 0.01:
                        weights.append((i, weight))

                if not weights:
                    super_map[y * 2:y * 2 + 2, x * 2:x * 2 + 2] = 7
                    continue

                total_weight = sum(w[1] for w in weights)
                weights = [(idx, w / total_weight) for idx, w in weights]
                weights.sort(key=lambda x: x[1], reverse=True)

                pixels = [0, 0, 0, 0]
                remaining_pixels = 4

                for biome_idx, weight in weights[:4]:
                    pixel_count = max(1, round(weight * 4))
                    pixel_count = min(pixel_count, remaining_pixels)

                    for i in range(pixel_count):
                        if remaining_pixels > 0:
                            pixels[4 - remaining_pixels] = biome_idx
                            remaining_pixels -= 1

                while remaining_pixels > 0:
                    pixels[4 - remaining_pixels] = weights[0][0]
                    remaining_pixels -= 1

                super_map[y * 2, x * 2] = pixels[0]
                super_map[y * 2, x * 2 + 1] = pixels[1]
                super_map[y * 2 + 1, x * 2] = pixels[2]
                super_map[y * 2 + 1, x * 2 + 1] = pixels[3]

        # Beach-Korrektur
        for y in range(0, h * 2, 2):
            for x in range(0, w * 2, 2):
                orig_y, orig_x = y // 2, x // 2
                elevation = self.heightmap[orig_y, orig_x] / self.world.scale

                if (elevation <= 0.20 and not ocean_connected[orig_y, orig_x] and
                        self._is_adjacent_to_ocean_super(y // 2, x // 2, super_map)):
                    super_map[y:y + 2, x:x + 2] = 1

        return super_map

    def _is_adjacent_to_ocean_super(self, y, x, super_map):
        """Prüft Ocean-Nachbarschaft auf Supersampling-Level"""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < super_map.shape[0] // 2 and 0 <= nx < super_map.shape[1] // 2):
                    if np.any(super_map[ny * 2:ny * 2 + 2, nx * 2:nx * 2 + 2] == 0):
                        return True
        return False


    def generate_biome_map(self) -> np.ndarray:
        """Vektorisierte Biom-Karte mit Broadcasting"""
        ocean_connected = self._is_connected_to_border()

        # Normalisierte Parameter für alle Pixel
        elevation = self.heightmap / self.world.scale
        rain = self.rain_map
        temp = self.temperature_map

        # Vektorisierte Gewichtsberechnung für alle Biome gleichzeitig
        biome_weights = np.zeros((self.heightmap.shape[0], self.heightmap.shape[1], len(biomes)))

        # Ocean direkt setzen
        biome_weights[ocean_connected, 0] = 1.0

        # Alle anderen Biome vektorisiert berechnen
        non_ocean_mask = ~ocean_connected
        for i, biome in enumerate(biomes[1:], 1):
            # Broadcasting für alle Pixel gleichzeitig
            h_weight = np.exp(-0.5 * ((elevation - biome.height_center) / biome.height_width) ** 2)
            r_weight = np.exp(-0.5 * ((rain - biome.rain_center) / biome.rain_width) ** 2)
            t_weight = np.exp(-0.5 * ((temp - biome.temp_center) / biome.temp_width) ** 2)

            biome_weights[:, :, i] = h_weight * r_weight * t_weight
            biome_weights[ocean_connected, i] = 0.0  # Ocean überschreibt alles

        # Dominantes Biom finden (vektorisiert)
        biome_map = np.argmax(biome_weights, axis=2)

        # Beach-Korrektur (vektorisiert wo möglich)
        elevation_mask = elevation <= 0.20
        non_ocean_mask = biome_map != 0

        # Ocean-Nachbarschaft prüfen (bleibt als Loop, aber deutlich weniger Pixel)
        potential_beaches = elevation_mask & non_ocean_mask
        beach_candidates = np.where(potential_beaches)

        for y, x in zip(beach_candidates[0], beach_candidates[1]):
            if self._is_adjacent_to_ocean(y, x, biome_map):
                biome_map[y, x] = 1  # Beach

        return biome_map