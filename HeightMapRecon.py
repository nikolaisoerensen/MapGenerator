"""
Modulares Python-Tool für prozedurale Terrain- und Biomgenerierung
Basiert auf Simplex Noise und bietet regional kontrollierbare Weltgenerierung
"""

import numpy as np
from opensimplex import OpenSimplex
from scipy import ndimage
from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from memory_profiler import profile
import matplotlib.pyplot as plt


# =================== HILFSFUNKTIONEN ===================
def normalize_array(arr, new_min=0.0, new_max=1.0):
    arr = np.array(arr)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.full_like(arr, new_min)
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return new_min + normalized * (new_max - new_min)

def create_circular_mask(shape: Tuple[int, int], center: Tuple[float, float],
                         radius: float, falloff: float = 0.1) -> np.ndarray:
    """Erstellt kreisförmige Maske mit weichem Falloff"""
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cy, cx = center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Optimierte Maskenerstellung
    mask = np.clip(1.0 - (distance - radius * (1 - falloff)) / (radius * falloff), 0.0, 1.0)
    mask[distance <= radius * (1 - falloff)] = 1.0

    return mask

# =================== 1. BASE TERRAIN GENERATOR ===================

class BaseTerrainGenerator:
    """Grundlegender Terrain-Generator mit Simplex Noise"""

    def __init__(self, width: int = 256, height: int = 256, seed: int = 42):
        self.width = width
        self.height = height
        self.seed = seed
        self.noise_gen = OpenSimplex(seed)
        np.random.seed(seed)

    def generate_heightmap(self,
                           octaves: int = 6,
                           frequency: float = 0.01,
                           persistence: float = 0.5,
                           lacunarity: float = 2.0,
                           scale: float = 500.0,
                           redistribute_power: float = 1.0) -> np.ndarray:
        """
        Generiert Heightmap mit Simplex Noise

        Args:
            octaves: Anzahl der Noise-Schichten
            frequency: Basis-Frequenz des Noise
            persistence: Amplituden-Reduktion pro Oktave
            lacunarity: Frequenz-Multiplikator pro Oktave
            scale: Skalierungsfaktor für finale Höhen
            redistribute_power: Exponential redistribution (>1 = mehr Berge, <1 = mehr Täler)
        """
        heightmap = np.zeros((self.height, self.width))

        # Multi-Oktaven Noise
        amplitude = 1.0
        max_amplitude = 0.0

        for octave in range(octaves):
            # Vektorisierte Noise-Generierung
            noise_layer = np.zeros((self.height, self.width))
            for y in range(self.height):
                for x in range(self.width):
                    noise_layer[y, x] = self.noise_gen.noise2(x * frequency, y * frequency)

            heightmap += amplitude * noise_layer
            max_amplitude += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        # Normalisierung auf Basis der maximalen Amplitude
        heightmap /= max_amplitude
        heightmap = normalize_array(heightmap, 0, 1)

        # Höhenverteilung anpassen
        if redistribute_power != 1.0:
            heightmap = np.power(heightmap, redistribute_power)

        return heightmap * scale


# =================== 2. TERRAIN MODIFIER ===================

class TerrainModifier:
    """Modifiziert Terrain basierend auf regionalen Masken"""

    @staticmethod
    def apply_smoothing(heightmap: np.ndarray, mask: np.ndarray,
                        intensity: float = 5.0, preserve_detail: bool = True) -> np.ndarray:
        """Intelligente Glättung mit Detailerhaltung"""
        if preserve_detail:
            # Bilateral Filter für Detailerhaltung
            from scipy.ndimage import gaussian_filter
            # Approximation eines Bilateral Filters
            smoothed = gaussian_filter(heightmap, sigma=intensity)
            detail_map = heightmap - gaussian_filter(heightmap, sigma=intensity * 0.3)
            result = smoothed + detail_map * 0.3
        else:
            result = ndimage.gaussian_filter(heightmap, sigma=intensity)

        return heightmap * (1 - mask) + result * mask

    @staticmethod
    def apply_elevation_change(heightmap: np.ndarray, mask: np.ndarray,
                               elevation_delta: float, blend_mode: str = 'add') -> np.ndarray:
        """Erweiterte Höhenänderung mit verschiedenen Blend-Modi"""
        if blend_mode == 'add':
            return heightmap + (elevation_delta * mask)
        elif blend_mode == 'multiply':
            factor = 1.0 + (elevation_delta * mask)
            return heightmap * factor
        elif blend_mode == 'overlay':
            # Soft overlay für natürlichere Übergänge
            delta_map = elevation_delta * mask
            return np.where(mask > 0.5,
                            heightmap + delta_map * (1 - heightmap / 100.0),
                            heightmap + delta_map)
        else:
            return heightmap + (elevation_delta * mask)

    @staticmethod
    def apply_coastal_erosion(heightmap: np.ndarray, mask: np.ndarray,
                              erosion_strength: float = 0.8) -> np.ndarray:
        """Simuliert Küstenerosion für realistische Küstenlinien"""
        # Gradient-basierte Erosion
        grad_y, grad_x = np.gradient(heightmap)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Stärkere Erosion bei steileren Hängen
        erosion_factor = gradient_magnitude * erosion_strength * mask
        eroded = heightmap - erosion_factor

        # Glättung für natürliche Küstenlinien
        eroded = ndimage.gaussian_filter(eroded, sigma=2.0)

        return heightmap * (1 - mask) + eroded * mask


# =================== 3. RIVER SIMULATOR ===================

class RiverSimulator:
    """Verbesserte Fluss-Simulation mit realistischeren Algorithmen"""

    def __init__(self,
                 heightmap: np.ndarray,
                 rain_map: np.ndarray,
                 sea_level: float = 0.15):
        self.heightmap = heightmap
        self.rain_map = rain_map
        self.sea_level = sea_level
        self.h, self.w = heightmap.shape
        self._flow_direction = None
        self._flow_accumulation = None

    def calculate_flow_direction(self) -> np.ndarray:
        """Berechnet D8-Fließrichtungen (8 Nachbarn)"""
        if self._flow_direction is not None:
            return self._flow_direction

        # D8 Richtungen: N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        flow_dir = np.zeros((self.h, self.w), dtype=np.int8)

        for y in range(1, self.h - 1):
            for x in range(1, self.w - 1):
                current_height = self.heightmap[y, x]
                max_slope = 0
                best_direction = 0

                for i, (dy, dx) in enumerate(directions):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.h and 0 <= nx < self.w:
                        neighbor_height = self.heightmap[ny, nx]

                        # Diagonale Distanzen berücksichtigen
                        distance = 1.0 if (dy == 0 or dx == 0) else 1.414
                        slope = (current_height - neighbor_height) / distance

                        if slope > max_slope:
                            max_slope = slope
                            best_direction = i

                flow_dir[y, x] = best_direction

        self._flow_direction = flow_dir
        return flow_dir

    def generate_flow_accumulation(self) -> np.ndarray:
        """Optimierte Flow Accumulation mit D8-Algorithmus"""
        if self._flow_accumulation is not None:
            return self._flow_accumulation

        flow_acc = self.rain_map.copy()  # Start mit Rain Map statt 1.0
        flow_dir = self.calculate_flow_direction()

        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        # Sortiere von hoch zu niedrig
        height_order = np.unravel_index(np.argsort(self.heightmap, axis=None)[::-1],
                                        self.heightmap.shape)

        for y, x in zip(height_order[0], height_order[1]):
            if 0 < y < self.h - 1 and 0 < x < self.w - 1:
                direction = flow_dir[y, x]
                dy, dx = directions[direction]
                target_y, target_x = y + dy, x + dx

                if 0 <= target_y < self.h and 0 <= target_x < self.w:
                    flow_acc[target_y, target_x] += flow_acc[y, x]

        self._flow_accumulation = flow_acc
        return flow_acc

    def create_river_network(self, min_flow_threshold: float = 2.0) -> np.ndarray:
        """Erstellt River Map (1.0 = Wasser vorhanden)"""
        flow_acc = self.generate_flow_accumulation()

        # River Map: Binär (Wasser ja/nein)
        river_map = (flow_acc > min_flow_threshold).astype(float)

        return river_map


# =================== 4. RAIN MAP GENERATOR ===================

class RainMapGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def generate_rain_map(self, heightmap: np.ndarray) -> np.ndarray:
        """Generiert Regenkarte mit West-Ost Gradient und Orographic Effect"""
        rain_map = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                # West-Ost Gradient (West = mehr Regen)
                west_factor = 1.0 - (x / self.width)  # 1.0 (West) -> 0.0 (Ost)
                base_rain = 0.3 + west_factor * 0.7  # 0.3 bis 1.0

                # Orographic Effect (Luv/Lee)
                if x > 0:
                    elevation_diff = heightmap[y, x] - heightmap[y, x-1]
                    # Steigung nach oben = mehr Regen (Luv)
                    orographic = max(0, elevation_diff * 0.01)
                    base_rain += orographic

                rain_map[y, x] = np.clip(base_rain, 0.0, 1.0)

        return rain_map


# =================== 5. TEMPERATURE MAP GENERATOR ===================

class TemperatureMapGenerator:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def generate_temperature_map(self, heightmap: np.ndarray) -> np.ndarray:
        """Generiert Temperaturkarte mit Nord-Süd Gradient"""
        temperature_map = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                # Nord-Süd Gradient (Süden = wärmer)
                latitude_factor = 1.0 - (y / self.height)  # 1.0 (Norden) -> 0.0 (Süden)
                base_temp = 0.3 + (1.0 - latitude_factor) * 0.5  # 0.3 (Nord) bis 0.8 (Süd)

                # Höhen-Effekt (höher = kälter)
                elevation_factor = -(heightmap[y, x] / 100.0) * 0.3

                temperature = base_temp + elevation_factor
                temperature_map[y, x] = np.clip(temperature, 0.0, 1.0)

        return temperature_map


# =================== 6. BIOME CLASSIFIER ===================

class Biome:
    def __init__(self, name, index, color, min_height, max_height, min_rain, max_rain, min_temp, max_temp, needs_water=False):
        self.name = name
        self.index = index
        self.color = color
        self.min_height = min_height
        self.max_height = max_height
        self.min_rain = min_rain
        self.max_rain = max_rain
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.needs_water = needs_water

    def matches(self, height, rain, temp, has_water, is_border):
        height_match = self.min_height <= height <= self.max_height
        rain_match = self.min_rain <= rain <= self.max_rain
        temp_match = self.min_temp <= temp <= self.max_temp
        water_match = (not self.needs_water) or has_water

        return height_match and rain_match and temp_match and water_match

# Definiere Biome (korrigiert)
biomes = [
    Biome("ocean",     0, "#000080", 0.0, 0.15, 0.0, 1.0, 0.0, 1.0),    # Nur an Kartenrand
    Biome("beach",     1, "#fffff0", 0.10, 0.25, 0.0, 1.0, 0.0, 1.0),  # Nur neben Ocean
    Biome("swamp",     2, "#a1a875", 0.15, 0.6, 0.6, 1.0, 0.4, 1.0, needs_water=True),
    Biome("forest",    3, "#228b22", 0.15, 0.8, 0.5, 1.0, 0.3, 0.8),
    Biome("grassland", 4, "#7cfc00", 0.15, 0.7, 0.3, 0.7, 0.3, 0.8),
    Biome("desert",    5, "#fff200", 0.15, 0.8, 0.0, 0.3, 0.5, 1.0),
    Biome("tundra",    6, "#b8dbc9", 0.15, 0.9, 0.2, 0.8, 0.0, 0.4),
    Biome("alpine",    7, "#e8f0f0", 0.8, 1.0, 0.0, 1.0, 0.0, 1.0),
]

class BiomeClassifier:
    """Klassifiziert Biome basierend auf Höhe, Regen und Temperatur"""

    def __init__(self, heightmap: np.ndarray, rain_map: np.ndarray,
                 temperature_map: np.ndarray, river_map: np.ndarray, scale: float):
        self.heightmap = heightmap
        self.rain_map = rain_map
        self.temperature_map = temperature_map
        self.river_map = river_map
        self.scale = scale
        self.h, self.w = heightmap.shape

    def _is_border_pixel(self, y: int, x: int) -> bool:
        """Prüft ob Pixel am Kartenrand liegt"""
        return y == 0 or y == self.h-1 or x == 0 or x == self.w-1

    def _is_connected_to_border_ocean(self, y: int, x: int, visited: np.ndarray = None) -> bool:
        """Flood-fill: Prüft ob Ocean-Pixel mit Kartenrand verbunden ist"""
        if visited is None:
            visited = np.zeros_like(self.heightmap, dtype=bool)

        if visited[y, x] or self.heightmap[y, x] > self.scale * 0.15:
            return False

        visited[y, x] = True

        if self._is_border_pixel(y, x):
            return True

        # Prüfe Nachbarn
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.h and 0 <= nx < self.w:
                if self._is_connected_to_border_ocean(ny, nx, visited):
                    return True

        return False

    def _is_adjacent_to_ocean(self, y: int, x: int, ocean_mask: np.ndarray) -> bool:
        """Prüft ob Pixel neben Ocean liegt"""
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.h and 0 <= nx < self.w:
                if ocean_mask[ny, nx]:
                    return True
        return False

    def generate_biome_map(self) -> np.ndarray:
        """Generiert Biome-Karte"""
        biome_map = np.zeros((self.h, self.w), dtype=int)

        # 1. Ocean-Maske erstellen (nur Randverbundene)
        ocean_mask = np.zeros((self.h, self.w), dtype=bool)
        visited_global = np.zeros_like(self.heightmap, dtype=bool)

        for y in range(self.h):
            for x in range(self.w):
                if (self.heightmap[y, x] <= self.scale * 0.15 and
                    not visited_global[y, x] and
                    self._is_connected_to_border_ocean(y, x)):
                    # Markiere alle verbundenen Ocean-Pixel
                    self._flood_fill_ocean(y, x, ocean_mask, visited_global)

        # 2. Biome zuweisen
        for y in range(self.h):
            for x in range(self.w):
                elevation = self.heightmap[y, x] / self.scale
                rain = self.rain_map[y, x]
                temp = self.temperature_map[y, x]
                has_water = self.river_map[y, x] > 0
                is_border = self._is_border_pixel(y, x)

                # Ocean hat Priorität
                if ocean_mask[y, x]:
                    biome_map[y, x] = 0  # Ocean
                    continue

                # Beach nur neben Ocean
                if (elevation <= 0.25 and
                    self._is_adjacent_to_ocean(y, x, ocean_mask)):
                    biome_map[y, x] = 1  # Beach
                    continue

                # Andere Biome
                for biome in biomes[2:]:  # Skip Ocean und Beach
                    if biome.matches(elevation, rain, temp, has_water, is_border):
                        biome_map[y, x] = biome.index
                        break
                else:
                    biome_map[y, x] = 4  # Grassland fallback

        return biome_map

    def _flood_fill_ocean(self, start_y: int, start_x: int,
                         ocean_mask: np.ndarray, visited: np.ndarray):
        """Flood-fill für Ocean-Bereiche"""
        stack = [(start_y, start_x)]

        while stack:
            y, x = stack.pop()

            if (visited[y, x] or
                self.heightmap[y, x] > self.scale * 0.15):
                continue

            visited[y, x] = True
            ocean_mask[y, x] = True

            # Nachbarn hinzufügen
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.h and 0 <= nx < self.w and not visited[ny, nx]:
                    stack.append((ny, nx))


# =================== 7. REGION CLASS ===================

@dataclass
class BoundingBox:
    """Begrenzungsbox für Regionen"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Region:
    """Definiert regionale Einflüsse und Eigenschaften"""

    def __init__(self, name: str, bounding_box: BoundingBox):
        self.name = name
        self.bbox = bounding_box
        self.elevation_modifiers: List[Callable] = []
        self.rain_modifiers: List[Callable] = []
        self.temperature_modifiers: List[Callable] = []

    def create_region_mask(self, width: int, height: int) -> np.ndarray:
        """Erstellt Maske für diese Region"""
        mask = np.zeros((height, width))
        mask[self.bbox.y_min:self.bbox.y_max,
        self.bbox.x_min:self.bbox.x_max] = 1.0
        return mask

    def add_elevation_modifier(self, modifier_func: Callable):
        self.elevation_modifiers.append(modifier_func)

    def add_rain_modifier(self, modifier_func: Callable):
        self.rain_modifiers.append(modifier_func)

    def add_temperature_modifier(self, modifier_func: Callable):
        self.temperature_modifiers.append(modifier_func)

    def apply_elevation_modifiers(self, heightmap: np.ndarray) -> np.ndarray:
        result = heightmap.copy()
        mask = self.create_region_mask(heightmap.shape[1], heightmap.shape[0])

        for modifier in self.elevation_modifiers:
            result = modifier(result, mask)

        return result

    def apply_rain_modifiers(self, rain_map: np.ndarray) -> np.ndarray:
        result = rain_map.copy()
        mask = self.create_region_mask(rain_map.shape[1], rain_map.shape[0])

        for modifier in self.rain_modifiers:
            result = modifier(result, mask)

        return result

    def apply_temperature_modifiers(self, temp_map: np.ndarray) -> np.ndarray:
        result = temp_map.copy()
        mask = self.create_region_mask(temp_map.shape[1], temp_map.shape[0])

        for modifier in self.temperature_modifiers:
            result = modifier(result, mask)

        return result


# =================== HAUPTKLASSE: WORLD GENERATOR ===================

class WorldGenerator:
    """Hauptklasse zur Weltgenerierung"""

    def __init__(self, width: int = 128, height: int = 128, seed: int = 42, sea_level: float = 0.0):
        self.width = width
        self.height = height
        self.seed = seed
        self.sea_level = sea_level

        # Generatoren
        self.terrain_gen = BaseTerrainGenerator(width, height, seed)

        # Karten
        self.heightmap = None
        self.river_map = None
        self.rain_map = None
        self.temperature_map = None
        self.biome_map = None

        # Regionen
        self.regions: List[Region] = []

    def add_region(self, region: Region):
        """Fügt Region hinzu"""
        self.regions.append(region)

    def generate_world(self, terrain_params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Generiert komplette Welt"""
        if terrain_params is None:
            terrain_params = {}

        print("Generiere Heightmap...")
        self.heightmap = self.terrain_gen.generate_heightmap(**terrain_params)

        # Wende regionale Höhen-Modifier an
        for region in self.regions:
            self.heightmap = region.apply_elevation_modifiers(self.heightmap)

        print("Generiere Regenkarte...")
        rain_gen = RainMapGenerator(self.width, self.height)
        self.rain_map = rain_gen.generate_rain_map(self.heightmap)

        # Wende regionale Regen-Modifier an
        for region in self.regions:
            self.rain_map = region.apply_rain_modifiers(self.rain_map)

        print("Simuliere Flüsse...")
        river_sim = RiverSimulator(self.heightmap, self.rain_map, sea_level=self.sea_level)
        self.river_map = river_sim.create_river_network()

        print("Generiere Temperaturkarte...")
        temp_gen = TemperatureMapGenerator(self.width, self.height)
        self.temperature_map = temp_gen.generate_temperature_map(self.heightmap)

        # Wende regionale Temperatur-Modifier an
        for region in self.regions:
            self.temperature_map = region.apply_temperature_modifiers(self.temperature_map)

        print("Klassifiziere Biome...")
        biome_classifier = BiomeClassifier(
            self.heightmap, self.rain_map, self.temperature_map,
            self.river_map, terrain_params.get("scale", 100.0)
        )
        self.biome_map = biome_classifier.generate_biome_map()

        print("Weltgenerierung abgeschlossen!")

        return {
            'heightmap': self.heightmap,
            'river_map': self.river_map,
            'rain_map': self.rain_map,
            'temperature_map': self.temperature_map,
            'biome_map': self.biome_map
        }

    def visualize_world(self, figsize: Tuple[int, int] = (15, 12)):
        """Visualisiert alle generierten Karten"""
        if self.heightmap is None:
            print("Keine Welt generiert! Rufe zuerst generate_world() auf.")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Heightmap
        im1 = axes[0, 0].imshow(self.heightmap, cmap='terrain')
        axes[0, 0].set_title('Höhenkarte')
        plt.colorbar(im1, ax=axes[0, 0])

        # River Map
        im2 = axes[0, 1].imshow(self.river_map, cmap='Blues')
        axes[0, 1].set_title('Flussnetzwerk')
        plt.colorbar(im2, ax=axes[0, 1])

        # Rain Map
        im3 = axes[0, 2].imshow(self.rain_map, cmap='YlGnBu')
        axes[0, 2].set_title('Regenkarte')
        plt.colorbar(im3, ax=axes[0, 2])

        # Temperature Map
        im4 = axes[1, 0].imshow(self.temperature_map, cmap='RdYlBu_r')
        axes[1, 0].set_title('Temperaturkarte')
        plt.colorbar(im4, ax=axes[1, 0])

        # Biome Map
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch

        biome_colors = [mcolors.to_rgb(biome.color) for biome in biomes]
        biome_names = [biome.name for biome in biomes]

        cmap = mcolors.ListedColormap(biome_colors)
        norm = mcolors.BoundaryNorm(boundaries=list(range(len(biomes) + 1)), ncolors=len(biomes))

        im5 = axes[1, 1].imshow(self.biome_map, cmap=cmap, norm=norm)
        axes[1, 1].set_title('Biom-Karte')

        legend_elements = [
            Patch(facecolor=biome_colors[i], edgecolor='black', label=biome_names[i])
            for i in range(len(biomes))
        ]
        axes[1, 1].legend(handles=legend_elements, loc='upper right', fontsize='small')

        # Kombinierte Ansicht
        height_norm = normalize_array(self.heightmap, 0, 1)
        temp_norm = normalize_array(self.temperature_map, 0, 1)
        rain_norm = self.rain_map

        rgb_composite = np.stack([height_norm, temp_norm, rain_norm], axis=2)
        axes[1, 2].imshow(rgb_composite)
        axes[1, 2].set_title('RGB-Komposition\n(R=Höhe, G=Temp, B=Regen)')

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()


# =================== BEISPIEL-VERWENDUNG ===================

@profile
def create_example_world():
    terrain_params = {
        'octaves': 6,
        'frequency': 0.032,
        'persistence': 0.5,
        'lacunarity': 2.0,
        'scale': 100.0,
        'redistribute_power': 1.5  # Für ausgeprägtere Täler und Senken
    }

    # Weltgenerator erstellen
    world_gen = WorldGenerator(width=64, height=64, seed=np.random.randint(0, 1000), sea_level=10.0)

    # Heightmap generieren (mit terrain_params)
    world_gen.heightmap = world_gen.terrain_gen.generate_heightmap(**terrain_params)

    # Restliche Welt generieren mit hydro_params
    world_gen.generate_world(terrain_params=terrain_params)

    # Visualisierung
    world_gen.visualize_world()

    return world_gen


if __name__ == "__main__":
    print("Erstelle Beispielwelt...")
    example_world = create_example_world()
    print("Fertig! Die Karten werden angezeigt.")