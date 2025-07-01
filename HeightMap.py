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
from scipy.ndimage import gaussian_filter


# =================== HILFSFUNKTIONEN ===================
def get_neighbors(self, y, x):
    """Gibt gültige Nachbarkoordinaten mit Pufferrand"""
    neighbors = []
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-Nachbarn
        ny, nx = y+dy, x+dx
        if 0 <= ny < self.h and 0 <= nx < self.w:
            neighbors.append((ny, nx))
    return neighbors

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
        """Vektorisierte Heightmap-Generierung"""

        # Koordinaten-Meshgrid einmalig erstellen
        x_coords, y_coords = np.meshgrid(np.arange(self.width), np.arange(self.height))

        heightmap = np.zeros((self.height, self.width))
        amplitude = 1.0
        max_amplitude = 0.0

        for octave in range(octaves):
            # Vektorisierte Noise-Berechnung
            x_scaled = x_coords * frequency
            y_scaled = y_coords * frequency

            # Batch-Noise-Generierung (falls OpenSimplex es unterstützt)
            # Fallback: Flatten und reshape für bessere Performance
            coords_flat = np.column_stack([x_scaled.flatten(), y_scaled.flatten()])
            noise_flat = np.array([self.noise_gen.noise2(x, y) for x, y in coords_flat])
            noise_layer = noise_flat.reshape((self.height, self.width))

            heightmap += amplitude * noise_layer
            max_amplitude += amplitude

            amplitude *= persistence
            frequency *= lacunarity

        # Vektorisierte Normalisierung und Redistribution
        heightmap /= max_amplitude
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

        if redistribute_power != 1.0:
            heightmap = np.power(heightmap, redistribute_power)

        return heightmap * scale


# =================== 2. TERRAIN MODIFIER ===================

class TerrainModifier:
    """Modifiziert Terrain basierend auf regionalen Masken"""

    @staticmethod
    def apply_elevation_change(heightmap: np.ndarray, mask: np.ndarray, elevation_delta: float,
                               blend_mode: str = 'add') -> np.ndarray:
        result = heightmap + (elevation_delta * mask)  # vereinfacht

        return result


    @staticmethod
    def apply_smoothing(heightmap: np.ndarray, mask: np.ndarray,
                        intensity: float = 5.0, preserve_detail: bool = True) -> np.ndarray:
        """Intelligente Glättung mit Detailerhaltung"""
        if preserve_detail:
            # Bilateral Filter für Detailerhaltung
            # Approximation eines Bilateral Filters
            smoothed = gaussian_filter(heightmap, sigma=intensity)
            detail_map = heightmap - gaussian_filter(heightmap, sigma=intensity * 0.3)
            result = smoothed + detail_map * 0.3
        else:
            result = ndimage.gaussian_filter(heightmap, sigma=intensity)

        return heightmap * (1 - mask) + result * mask

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

    @staticmethod
    def apply_flat_coastal_plain(heightmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Platzhalter für Norddeutschland-ähnliche flache Küstenebenen
        TODO: Implementiere spezifische Logik für flache Küstenregionen
        """
        # Momentan: Starke Glättung + leichte Absenkung
        smoothed = TerrainModifier.apply_smoothing(heightmap, mask, intensity=8.0)
        lowered = TerrainModifier.apply_elevation_change(smoothed, mask, -15.0)
        return lowered

    @staticmethod
    def apply_mountain_range(heightmap: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Platzhalter für Gebirgsregionen
        TODO: Implementiere Ridge-Noise oder spezifische Gebirgsmuster
        """
        # Momentan: Erhöhung + verstärkte Variation
        elevated = TerrainModifier.apply_elevation_change(heightmap, mask, 40.0)
        return elevated


# =================== 3. RIVER SIMULATOR ===================

class RiverSimulator:
    """Memory-optimierte Fluss-Simulation mit erweitertem Caching für Terrain- und Regenparameter"""

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
        """Prüft Änderungen an heightmap oder rain_map/rain_scale"""
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
        """Cached Flow Direction Berechnung mit Randbehandlung"""
        self._invalidate_cache_if_needed()

        if self._flow_direction is not None:
            return self._flow_direction

        # Original-Logik mit Randmarkierung
        flow_dir = np.zeros((self.h, self.w), dtype=np.int8)
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]

        for y in range(self.h):
            for x in range(self.w):
                if y == 0 or y == self.h - 1 or x == 0 or x == self.w - 1:
                    flow_dir[y, x] = -1  # Markiere Ränder
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
        """In-Place Flow Accumulation mit Cache-Support"""
        self._invalidate_cache_if_needed()

        if self._flow_accumulation is not None:
            return self._flow_accumulation

        flow_acc = self.calculate_water_volume()  # Nutzt rain_map + rain_scale
        flow_dir = self.calculate_flow_direction()
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        # Höhenbasiertes Sorting (einmalig)
        height_indices = np.unravel_index(
            np.argsort(self.heightmap, axis=None)[::-1],
            self.heightmap.shape
        )

        # In-Place Accumulation mit Randprüfung
        for y, x in zip(height_indices[0], height_indices[1]):
            if flow_dir[y, x] == -1:  # Überspringe Ränder
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
        """Berechnet das Wasservolumen mit skaliertem Feuchtigkeitswert"""
        return self.rain_map * self.rain_scale

    def find_depressions(self) -> np.ndarray:
        """Watershed-basierte Seenerkennung"""
        from scipy.ndimage import watershed_ift

        # KORREKTUR: Watershed statt lokale Minima
        markers = np.zeros_like(self.heightmap, dtype=np.int32)
        markers[self.heightmap < np.percentile(self.heightmap, 20)] = 1

        watersheds = watershed_ift(self.heightmap.astype(np.uint8), markers)
        depression_sizes = ndimage.sum(watersheds == 1, watersheds, range(2))

        return (watersheds == 1) & (depression_sizes[watersheds] > 10)

    def adjust_terrain_for_lakes(self, heightmap: np.ndarray, lakes: np.ndarray) -> np.ndarray:
        """Glättet und senkt das Terrain für Seenbereiche"""
        # Glättung in Seenbereichen
        smoothed = ndimage.gaussian_filter(heightmap, sigma=1.5)

        # Leichte Absenkung der Seenbereiche
        adjusted = heightmap - (lakes * 0.5)

        # Kombiniere die Anpassungen
        result = np.where(lakes, smoothed, heightmap)
        result = np.where(lakes, adjusted, result)

        return result

    def create_river_network(self, min_flow_threshold: float = 0.1, max_flow_threshold: float = 0.9) -> Dict[
        str, np.ndarray]:
        """Optimierte Version mit Cache-Support und regionalen Thresholds"""
        self._invalidate_cache_if_needed()

        flow_acc = self.generate_flow_accumulation()
        flow_normalized = self._normalize_array(flow_acc)

        # Regionale Thresholds (West -> Ost Anstieg)
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

    @staticmethod
    def _normalize_array(arr: np.ndarray) -> np.ndarray:
        """Hilfsfunktion für Normalisierung 0-1"""
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # +1e-8 gegen Division durch 0


# =================== 4. rain MAP GENERATOR ===================

class RainMapGenerator:
    def __init__(self, world):
        self.world = world

    def generate_wind_field(self, base_wind_speed=1.0, wind_terrain_factor=0.4, smoothing_sigma=1.0):
        """
        Generiert ein 2D-Windvektorfeld, das von Westen kommt und durch Höhen abgelenkt wird.
        Höhe lenkt Windvektoren um – Bergflanken erzeugen Ableitung, Täler bündeln den Wind.
        """
        heightmap = self.world.heightmap
        h, w = heightmap.shape

        # Noise für Windvariation an der Westkante
        noise_gen = OpenSimplex(self.world.seed + 100)

        # 1. Höhen-Gradient berechnen
        grad_y, grad_x = np.gradient(heightmap)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-8  # +epsilon zur Vermeidung von Division durch 0

        # 2. Terrain-Normalenfeld erzeugen
        norm_x = grad_x / grad_mag
        norm_y = grad_y / grad_mag

        # 3. Basis-Windvektor (Ost) anpassen durch Gelände
        wind_x = np.zeros((h, w))
        wind_y = np.zeros((h, w))

        for y in range(h):
            # Windstärke an der Westkante mit Noise variieren
            y_noise = noise_gen.noise2(0, y * 0.1)
            west_wind_speed = base_wind_speed * (1.0 + y_noise * 0.15)

            for x in range(w):
                # Basisrichtung: von Westen (Ostwind)
                wind = np.array([west_wind_speed, 0.0])

                # Terrain beeinflusst Wind: seitlich ablenken proportional zum Gradient
                deflect = np.array([norm_y[y, x], -norm_x[y, x]])  # 90°-Drehung des Gradienten
                wind += deflect * wind_terrain_factor

                # Normieren auf tatsächliche Geschwindigkeit
                speed = np.linalg.norm(wind)
                wind_unit = wind / speed

                wind_x[y, x] = wind_unit[0] * speed
                wind_y[y, x] = wind_unit[1] * speed

        # Optional glätten (Flow natürlicher)
        from scipy.ndimage import gaussian_filter
        wind_x = gaussian_filter(wind_x, sigma=smoothing_sigma)
        wind_y = gaussian_filter(wind_y, sigma=smoothing_sigma)

        # Speichern
        self.wind_x = wind_x
        self.wind_y = wind_y

        return wind_x, wind_y

    def generate_orographic_rain_with_wind(self, base_wind_speed=1.0, wind_terrain_factor=0.3,
                                           rain_threshold=0.6, initial_moisture=1.0):
        """Erweiterte Regensimulation mit Windfeld und Feuchtigkeitstransport"""
        heightmap = self.world.heightmap
        height, width = heightmap.shape

        wind_x, wind_y = self.generate_wind_field(base_wind_speed, wind_terrain_factor)

        rain_map = np.zeros((height, width))
        moisture_map = np.zeros((height, width))
        rain_indicator = np.zeros((height, width))

        for y in range(height):
            moisture = initial_moisture
            rain_ind = 0.0

            for x in range(width):
                current_elevation = heightmap[y, x]

                # Höhenveränderung berechnen
                if x > 0:
                    height_diff = current_elevation - heightmap[y, x - 1]

                    # Regenindikator: steigt bergauf, sinkt bergab
                    rain_ind += height_diff * 0.02
                    rain_ind = max(0.0, rain_ind)

                # Regen wenn Schwelle überschritten und Feuchtigkeit vorhanden
                if rain_ind > rain_threshold and moisture > 0:
                    rainfall = min(
                        moisture * rain_ind * self.wind_speed[y, x] * 0.1,
                        moisture * 0.7
                    )
                    rain_map[y, x] = rainfall
                    moisture -= rainfall
                    rain_ind *= 0.7  # Regenindikator reduzieren nach Regen

                moisture_map[y, x] = moisture
                rain_indicator[y, x] = rain_ind

                # Feuchtigkeitsregeneration
                moisture = min(1.0, moisture + 0.03)

        # Speichere Windfeld für Visualisierung
        self.moisture_map = moisture_map

        return np.clip(rain_map, 0.0, 1.0)

    def generate_orographic_rain(self, wind_speed=1.5, shadow_factor=0.9):
        """Fallback zur alten Methode - ruft neue Windfeld-Methode auf"""
        return self.generate_orographic_rain_with_wind(
            base_wind_speed=wind_speed,
            wind_terrain_factor=0.3,
            rain_threshold=0.6
        )

    def create_water_map(self) -> np.ndarray:
        """Erstellt binäre Wasser-Karte (0/1) aus Flussnetzwerk"""
        if not hasattr(self.world, 'water_network'):
            return np.zeros_like(self.world.heightmap)

        water_map = np.zeros_like(self.world.heightmap)
        for water_type in ['streams', 'rivers', 'major_rivers', 'lakes']:
            if water_type in self.world.water_network:
                water_map = np.maximum(water_map, self.world.water_network[water_type])

        return (water_map > 0).astype(float)

    @property
    def wind_speed(self):
        """Dynamische Berechnung der Windgeschwindigkeit aus wind_x/wind_y."""
        if self.wind_x is None or self.wind_y is None:
            raise ValueError("Windfeld wurde noch nicht generiert. Rufe zuerst generate_wind_field() auf.")
        return np.sqrt(self.wind_x ** 2 + self.wind_y ** 2)

    @property
    def wind_direction(self):
        """Dynamische Berechnung der Windrichtung (in Radiant)."""
        if self.wind_x is None or self.wind_y is None:
            raise ValueError("Windfeld wurde noch nicht generiert. Rufe zuerst generate_wind_field() auf.")
        return np.arctan2(self.wind_y, self.wind_x)


# =================== 5. TEMPERATURE MAP GENERATOR ===================
class TemperatureMapGenerator:
    def __init__(self, world):
        self.world = world

    def generate_temperature_map(self):
        heightmap = self.world.heightmap
        height, width = heightmap.shape
        temperature_map = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                elevation = heightmap[y, x] / self.world.scale

                # Süd-Nord Gradient (y=0 ist Norden, y=height ist Süden)
                latitude_factor = (height - y) / height  # 1.0 = Süden, 0.0 = Norden
                base_temp = latitude_factor * 0.1  # 10 % Anteil

                # Höhen-Abkühlung (invertiert: 0.0 kalt, 1.0 warm)
                elevation_factor = (1.0 - elevation) * 0.5  # 50 % Anteil

                # Hanglage (Südhänge wärmer)
                if y < height - 1:
                    south_slope = heightmap[y, x] - heightmap[y + 1, x]
                    slope_factor = south_slope * 0.4  # 40 % Anteil
                else:
                    slope_factor = 0

                temperature = base_temp + elevation_factor + slope_factor
                temperature_map[y, x] = np.clip(temperature, 0.0, 1.0)

        temperature_map = self.smooth_temperature_map(temperature_map)
        return temperature_map

    def smooth_temperature_map(self, temp_map: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """
        Glättet die Temperaturkarte mit einem gaußschen Filter.
        sigma: Je höher, desto stärker die Glättung. Typisch: 0.5 bis 3.0
        """
        smoothed = gaussian_filter(temp_map, sigma=sigma)
        return np.clip(smoothed, 0.0, 1.0)

# =================== 6. BIOME CLASSIFIER ===================

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
    Biome("grassland", 7, "#7cfc00", 0.4, 0.2, 0.3, 0.15, 0.6, 0.2),
    Biome("forest", 8, "#228b22", 0.5, 0.25, 0.7, 0.2, 0.5, 0.2),
    Biome("steppe", 9, "#c2b280", 0.45, 0.2, 0.45, 0.15, 0.55, 0.2),
    ]

# =================== HAUPTKLASSE: WORLD ===================

class World:
    def __init__(self, heightmap, width, height, scale, seed=None):
        self.heightmap = heightmap
        self.width = width
        self.height = height
        self.scale = scale
        self.seed = seed
        self.temperature_map = None
        self.rain_map = None

class BiomeClassifier:
    """Klassifiziert Biome basierend auf Höhe und Feuchtigkeit"""

    def __init__(self, world: World, temperature_map: np.ndarray, rain_map: np.ndarray):
        self.world = world
        self.heightmap = world.heightmap
        self.temperature_map = temperature_map
        self.rain_map = rain_map

    def _is_connected_to_border(self):
        """Flood-fill vom Rand für Ocean-Bereiche"""
        ocean_mask = self.heightmap <= self.world.scale * 0.15

        # Markiere alle Randpixel als Seeds
        seeds = np.zeros_like(ocean_mask, dtype=bool)
        seeds[0, :] = seeds[-1, :] = seeds[:, 0] = seeds[:, -1] = True
        seeds = seeds & ocean_mask

        # Flood-fill
        from scipy.ndimage import binary_dilation
        connected = seeds.copy()
        for _ in range(max(self.heightmap.shape)):
            new_connected = binary_dilation(connected) & ocean_mask
            if np.array_equal(connected, new_connected):
                break
            connected = new_connected

        return connected

    def generate_supersampled_biome_map(self) -> np.ndarray:
        """Generiert 2x2 Supersampling mit gewichteten Biom-Anteilen"""
        ocean_connected = self._is_connected_to_border()
        h, w = self.heightmap.shape
        super_map = np.zeros((h * 2, w * 2), dtype=int)

        for y in range(h):
            for x in range(w):
                if ocean_connected[y, x]:
                    # Ocean füllt alle 4 Subpixel
                    super_map[y * 2:y * 2 + 2, x * 2:x * 2 + 2] = 0
                    continue

                elevation = self.heightmap[y, x] / self.world.scale
                rain = self.rain_map[y, x]
                temp = self.temperature_map[y, x]

                # Berechne Gewichte für alle Biome (außer Ocean)
                weights = []
                for i, biome in enumerate(biomes[1:], 1):
                    weight = biome.get_weight(elevation, rain, temp)
                    if weight > 0.01:  # Nur relevante Biome
                        weights.append((i, weight))

                if not weights:
                    super_map[y * 2:y * 2 + 2, x * 2:x * 2 + 2] = 7  # Fallback grassland
                    continue

                # Normalisiere und sortiere nach Gewicht
                total_weight = sum(w[1] for w in weights)
                weights = [(idx, w / total_weight) for idx, w in weights]
                weights.sort(key=lambda x: x[1], reverse=True)

                # Verteile 4 Subpixel basierend auf Gewichten
                pixels = [0, 0, 0, 0]  # 4 Subpixel
                remaining_pixels = 4

                for biome_idx, weight in weights[:4]:  # Max 4 verschiedene Biome
                    pixel_count = max(1, round(weight * 4))
                    pixel_count = min(pixel_count, remaining_pixels)

                    for i in range(pixel_count):
                        if remaining_pixels > 0:
                            pixels[4 - remaining_pixels] = biome_idx
                            remaining_pixels -= 1

                # Fülle verbleibende Pixel mit stärkstem Biom
                while remaining_pixels > 0:
                    pixels[4 - remaining_pixels] = weights[0][0]
                    remaining_pixels -= 1

                # Setze 2x2 Block
                super_map[y * 2, x * 2] = pixels[0]
                super_map[y * 2, x * 2 + 1] = pixels[1]
                super_map[y * 2 + 1, x * 2] = pixels[2]
                super_map[y * 2 + 1, x * 2 + 1] = pixels[3]

        # Beach-Korrektur auf Supersampling-Level
        for y in range(0, h * 2, 2):
            for x in range(0, w * 2, 2):
                orig_y, orig_x = y // 2, x // 2
                elevation = self.heightmap[orig_y, orig_x] / self.world.scale

                if (elevation <= 0.20 and not ocean_connected[orig_y, orig_x] and
                        self._is_adjacent_to_ocean_super(y // 2, x // 2, super_map)):
                    super_map[y:y + 2, x:x + 2] = 1  # Beach

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

    def _is_adjacent_to_ocean(self, y, x, biome_map):
        """Check 8 neighbors for ocean biome"""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < biome_map.shape[0] and
                        0 <= nx < biome_map.shape[1] and
                        biome_map[ny, nx] == 0):  # Ocean index
                    return True
        return False

    @property
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

# =================== 7. Erosion CLASS ===================

class HydraulicErosion:
    def __init__(self, erosion_strength=0.4, valley_depth_factor=2.0, tributary_factor=0.6):
        self.erosion_strength = erosion_strength
        self.valley_depth_factor = valley_depth_factor
        self.tributary_factor = tributary_factor

    def erode_hierarchical(self, heightmap, flow_accumulation):
        """Hierarchische Erosion für Tal-Systeme"""
        eroded = heightmap.copy()
        flow_norm = normalize_array(flow_accumulation, 0, 1)

        # Verschiedene Fluss-Hierarchien
        major_rivers = flow_norm > 0.8
        rivers = (flow_norm > 0.4) & (flow_norm <= 0.8)
        streams = (flow_norm > 0.1) & (flow_norm <= 0.4)

        # Hauptflüsse: Tiefe Täler
        if np.any(major_rivers):
            major_erosion = flow_norm * self.valley_depth_factor * self.erosion_strength
            eroded[major_rivers] -= major_erosion[major_rivers]

            # Tal-Verbreiterung
            major_mask = gaussian_filter(major_rivers.astype(float), sigma=3.0) > 0.1
            eroded[major_mask] -= major_erosion[major_mask] * 0.4

        # Nebenflüsse: Mittlere Erosion
        if np.any(rivers):
            river_erosion = flow_norm * self.valley_depth_factor * 0.6 * self.erosion_strength
            eroded[rivers] -= river_erosion[rivers]

            # Verbindung zu Hauptflüssen
            connection_mask = gaussian_filter(rivers.astype(float), sigma=1.5) > 0.2
            eroded[connection_mask] -= river_erosion[connection_mask] * self.tributary_factor

        # Bäche: Sanfte Erosion
        if np.any(streams):
            stream_erosion = flow_norm * 0.3 * self.erosion_strength
            eroded[streams] -= stream_erosion[streams]

        return eroded

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

        # Platzhalter für Modifier-Funktionen
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
        """Fügt Höhen-Modifier hinzu"""
        self.elevation_modifiers.append(modifier_func)

    def add_rain_modifier(self, modifier_func: Callable):
        """Fügt Feuchtigkeits-Modifier hinzu"""
        self.rain_modifiers.append(modifier_func)

    def add_temperature_modifier(self, modifier_func: Callable):
        """Fügt Temperatur-Modifier hinzu"""
        self.temperature_modifiers.append(modifier_func)

    def apply_elevation_modifiers(self, heightmap: np.ndarray) -> np.ndarray:
        result = heightmap.copy()
        mask = self.create_region_mask(heightmap.shape[1], heightmap.shape[0])

        for modifier in self.elevation_modifiers:
            result = modifier(result, mask)

        # KORREKTUR: Cache invalidieren
        if hasattr(self, '_flow_direction'):
            self._flow_direction = None
            self._flow_accumulation = None

        return result

    def apply_rain_modifiers(self, rain_map: np.ndarray) -> np.ndarray:
        """Wendet alle Feuchtigkeits-Modifier an"""
        result = rain_map.copy()
        mask = self.create_region_mask(rain_map.shape[1], rain_map.shape[0])

        for modifier in self.rain_modifiers:
            result = modifier(result, mask)

        return result

    def apply_temperature_modifiers(self, temp_map: np.ndarray) -> np.ndarray:
        """Wendet alle Temperatur-Modifier an"""
        result = temp_map.copy()
        mask = self.create_region_mask(temp_map.shape[1], temp_map.shape[0])

        for modifier in self.temperature_modifiers:
            result = modifier(result, mask)

        return result



# =================== HAUPTKLASSE: WORLD GENERATOR ===================

class WorldGenerator:
    """Hauptklasse zur Weltgenerierung"""

    def __init__(self, width: int = 128, height: int = 128, seed: int = 42, sea_level:float = 0.0):
        self.width = width
        self.height = height
        self.seed = seed
        self.sea_level = sea_level
        self.hydro_params = {}

        # Generatoren
        self.terrain_gen = BaseTerrainGenerator(width, height, seed)

        # Karten
        self.heightmap = None
        self.original_heightmap = None
        self.flow_map = None
        self.rain_map = None
        self.temperature_map = None
        self.biome_map = None

        # Regionen
        self.regions: List[Region] = []

    def add_region(self, region: Region):
        """Fügt Region hinzu"""
        self.regions.append(region)

    def generate_world(self,
                       terrain_params: Optional[Dict] = None,
                       hydro_params: Optional[Dict] = None,
                       iterations: int = 3) -> Dict[str, np.ndarray]:
        """Generiert komplette Welt mit Feedback-Loops"""
        if terrain_params is None:
            terrain_params = {}
        if hydro_params is None:
            hydro_params = {}

        self.hydro_params = hydro_params

        print("Generiere Heightmap...")
        self.heightmap = self.terrain_gen.generate_heightmap(**terrain_params)
        # Tiefe Kopie, da sonst beide auf das gleiche Objekt zeigen
        self.original_heightmap = self.heightmap.copy()

        # Pit-Filling sofort nach Generierung
        self.heightmap = self._fill_pits_global(self.heightmap)

        world = World(self.heightmap, self.width, self.height, terrain_params.get("scale", 1.0), seed=self.seed)

        # Feedback-Loop für realistische Entwicklung
        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}...")

            # Regen basierend auf aktueller Topographie mit Windfeld
            rain_gen = RainMapGenerator(world)
            self.rain_map = rain_gen.generate_orographic_rain_with_wind(
                base_wind_speed=hydro_params.get('wind_speed', 1.0),
                wind_terrain_factor=hydro_params.get('wind_terrain_factor', 0.3),
                rain_threshold=hydro_params.get('rain_threshold', 0.6)
            )

            # Speichere Windfeld-Daten
            self.wind_x = rain_gen.wind_x
            self.wind_y = rain_gen.wind_y
            self.moisture_map = rain_gen.moisture_map

            if (self.rain_map.max() - self.rain_map.min()) != 0:
                self.rain_map = (self.rain_map - self.rain_map.min()) / (self.rain_map.max() - self.rain_map.min())

            # Flow-Simulation mit Regen-Kopplung
            river_sim = RiverSimulator(self.heightmap, self.rain_map,
                                       sea_level=self.sea_level,
                                       rain_scale=hydro_params.get('rain_scale', 5.0),
                                       lake_threshold=hydro_params.get('lake_threshold', 0.5))

            self.flow_map = river_sim.generate_flow_accumulation()

            # Hierarchische Erosion
            erosion = HydraulicErosion(
                erosion_strength=hydro_params.get('erosion_strength', 0.4),
                valley_depth_factor=hydro_params.get('valley_depth_factor', 2.0),
                tributary_factor=hydro_params.get('tributary_factor', 0.6)
            )

            self.heightmap = erosion.erode_hierarchical(self.heightmap, self.flow_map)
            world.heightmap = self.heightmap

        # Finale Wasserfeatures
        water_network = river_sim.create_river_network()
        self.heightmap = self._adjust_lakes_realistic(self.heightmap, water_network['lakes'])
        self.water_network = water_network

        # Rest der Generierung...
        print("Generiere Temperaturkarte...")
        temp_gen = TemperatureMapGenerator(world)
        self.temperature_map = temp_gen.generate_temperature_map()

        print("Klassifiziere Biome...")
        biome_classifier = BiomeClassifier(world, self.temperature_map, self.rain_map)
        self.biome_map = biome_classifier.generate_biome_map
        self.biome_map_super = biome_classifier.generate_supersampled_biome_map()

        return {
            'heightmap': self.heightmap,
            'flow_map': self.flow_map,
            'rain_map': self.rain_map,
            'temperature_map': self.temperature_map,
            'biome_map': self.biome_map
        }

    def _fill_pits_global(self, heightmap, max_fill=5.0):  # max_fill: 2.0-10.0
        """Globales Pit-Filling mit Begrenzung"""
        filled = np.copy(heightmap)
        for _ in range(10):
            old_filled = filled.copy()
            filled = ndimage.maximum_filter(filled, size=3)
            filled = np.minimum(filled, heightmap + max_fill)
            if np.allclose(filled, old_filled):
                break
        return filled

    def _adjust_lakes_realistic(self, heightmap, lakes, lake_depth=3.0):  # lake_depth: 1.0-6.0
        """Realistische Seen-Anpassung mit Drainage"""
        adjusted = heightmap.copy()

        if np.any(lakes):
            # Seen-Zentren finden
            from scipy.ndimage import center_of_mass, label
            lake_labels, num_lakes = label(lakes)

            for lake_id in range(1, num_lakes + 1):
                lake_mask = lake_labels == lake_id

                # See-Zentrum als tiefster Punkt
                lake_center = center_of_mass(lake_mask)
                center_y, center_x = int(lake_center[0]), int(lake_center[1])

                # Drainage-Level bestimmen
                lake_heights = heightmap[lake_mask]
                drainage_level = np.percentile(lake_heights, 10) - lake_depth

                # Graduelle Absenkung zum Zentrum
                for y, x in zip(*np.where(lake_mask)):
                    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                    depth_factor = np.exp(-distance * 0.1)
                    adjusted[y, x] = max(drainage_level,
                                         heightmap[y, x] - lake_depth * depth_factor)

        return adjusted

    def visualize_world(self, figsize: Tuple[int, int] = (15, 12)):
        """Visualisiert alle generierten Karten"""
        if self.heightmap is None:
            print("Keine Welt generiert! Rufe zuerst generate_world() auf.")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        self._plot_heightmap_original(axes[0, 0])
        self._plot_heightmap_erosion(axes[0, 1])
        self._plot_flow_map(axes[0, 2])
        self._plot_rain_map(axes[1, 0])
        self._plot_temperature_map(axes[1, 1])
        self._plot_biome_map(axes[1, 2])

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()

    def _plot_heightmap_original(self, ax):
        im = ax.imshow(self.original_heightmap, cmap='terrain')
        ax.set_title('Höhenkarte original:')
        ax.axvline(x=self.width - 0.5, color='red', linestyle='--', alpha=0.7)
        plt.colorbar(im, ax=ax)

    def _plot_heightmap_erosion(self, ax):
        im = ax.imshow(self.heightmap, cmap='terrain')
        ax.set_title('Höhenkarte erodiert:')
        plt.colorbar(im, ax=ax)

    def _plot_flow_map(self, ax):
        """Zeigt topografische Linien mit Windfluss (Pfeile + optional Flowlines)"""
        contour_levels = np.linspace(self.heightmap.min(), self.heightmap.max(), 25)
        ax.contour(self.heightmap, levels=contour_levels, colors='gray', linewidths=0.4, alpha=0.7)

        step = 4
        y_coords, x_coords = np.meshgrid(
            range(0, self.height, step),
            range(0, self.width, step),
            indexing='ij'
        )

        if hasattr(self, 'wind_x') and hasattr(self, 'wind_y'):
            u = self.wind_x
            v = self.wind_y
            speed = np.sqrt(u**2 + v**2)

            # Flowlines (streamplot)
            ax.streamplot(
                np.arange(self.width),
                np.arange(self.height),
                u, v,
                color=speed, linewidth=0.6, density=2.5,
                cmap='Blues', arrowsize=0.8
            )

        ax.set_title('Windfluss & Topografie')
        ax.set_aspect('equal')
        ax.invert_yaxis()

    def _plot_rain_map(self, ax):
        im = ax.imshow(self.rain_map, cmap='YlGnBu')
        ax.set_title('Regenkarte')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

    def _plot_temperature_map(self, ax):
        im = ax.imshow(self.temperature_map, cmap='RdYlBu_r')
        ax.set_title('Temperaturkarte')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)

    def _plot_biome_map(self, ax):
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch

        # Topographische Linien
        contour_levels = np.linspace(self.heightmap.min(), self.heightmap.max(), 25)
        ax.contour(self.heightmap, levels=contour_levels, colors='gray', linewidths=0.4, alpha=0.7)

        biome_colors = [mcolors.to_rgb(biome.color) for biome in biomes]
        biome_names = [biome.name for biome in biomes]

        cmap = mcolors.ListedColormap(biome_colors)
        norm = mcolors.BoundaryNorm(boundaries=list(range(len(biomes) + 1)), ncolors=len(biomes))

        if hasattr(self, 'water_network'):
            water_display = np.zeros_like(self.heightmap)
            water_display = np.maximum(water_display, self.water_network['streams'] * 0.3)
            water_display = np.maximum(water_display, self.water_network['rivers'] * 0.6)
            water_display = np.maximum(water_display, self.water_network['major_rivers'] * 0.9)
            water_display = np.maximum(water_display, self.water_network['lakes'] * 1.0)

            from matplotlib.colors import LinearSegmentedColormap
            water_cmap = LinearSegmentedColormap.from_list('water', ['#00000000', '#1E90FF'])
        if hasattr(self, 'biome_map_super'):
            ax.imshow(self.biome_map_super, cmap=cmap, norm=norm, alpha=0.7)
        ax.imshow(water_display, cmap=water_cmap, alpha=0.8, vmin=0, vmax=1, interpolation='bilinear')

        im = ax.imshow(self.biome_map, cmap=cmap, norm=norm)
        ax.set_title('Biom-Karte')
        ax.set_aspect('equal')

        legend_elements = [
            Patch(facecolor=biome_colors[i], edgecolor='black', label=biome_names[i])
            for i in range(len(biomes))
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

# =================== BEISPIEL-VERWENDUNG ===================

@profile
def create_example_world():
    terrain_params = {
        'octaves': 6,                 # Anzahl der Oktaven für das Fraktal – mehr = detailreicher
        'frequency': 0.032,           # Basisfrequenz der Noise – höher = kleinere Features
        'persistence': 0.5,           # Amplitudenfall pro Oktave – kleiner = flacher, weicher
        'lacunarity': 2.0,            # Frequenzanstieg pro Oktave – größer = steilere, komplexere Formen
        'scale': 100.0,               # Gesamtskalierung der Höhe – z. B. 100 = moderate Höhen
        'redistribute_power': 1.3     # Höhenverteilung – <1 = mehr flache Gebiete, >1 = mehr Extreme
    }

    hydro_params = {
        'wind_speed': 0.7,         # 0.3 (wenig Regenverteilung) bis ca. 1.7 (starke Regenverteilung)
        'wind_terrain_factor': 0.3,        # 0.1 (Wind kaum beeinflusst von Terrain) bis 0.8 (stark beeinflusst)
        'rain_threshold': 0.5,        # 0.2 (viel Regen, schnell) bis 0.8 (Regen nur bei starkem Anstieg)
        'shadow_factor': 0.7,         # Noch nicht verwendet (könnte für Leeseiten oder Regenminderung dienen)
        'rain_scale': 3.0,            # Skaliert die Menge an Wasser in Flusssystemen (z. B. 2–10 üblich)
        'erosion_strength': 0.8       # 0.1 (kaum Erosion) bis 1.0 (sehr starke Täler)
    }

    world_gen = WorldGenerator(width=100, height=100, seed=np.random.randint(0, 1000), sea_level=10.0)
    world_gen.generate_world(terrain_params=terrain_params, hydro_params=hydro_params, iterations=8)
    world_gen.visualize_world()
    return world_gen

if __name__ == "__main__":
    print("Erstelle Beispielwelt...")
    example_world = create_example_world()
    print("Fertig! Die Karten werden angezeigt.")

