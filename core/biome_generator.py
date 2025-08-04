"""
Path: core/biome_generator.py

Funktionsweise: Klassifikation von Biomen auf Basis von Höhe, Temperatur, Niederschlag und Slope
- BaseGenerator-Integration mit einheitlicher API und LOD-System
- Gauß-basierte Klassifizierung mit Gewichtungen je Biomtyp
- Verwendung eines vektorbasierten Klassifikators für performante Zuordnung auf großen Karten
- Zwei-Ebenen-System: Base-Biomes und Super-Biomes
- Supersampling für weichere Übergänge zwischen allen Biomen (die vier dominantesten Anteile pro Zelle)
- Super-Biomes überschreiben Base-Biomes basierend auf speziellen Bedingungen
- Optional Dependencies: soil_moist_map und water_biomes_map mit intelligenten Fallback-Werten

Parameter Input (aus value_default.py BIOME):
- biome_wetness_factor (Gewichtung der Bodenfeuchtigkeit)
- biome_temp_factor (Gewichtung der Temperaturwerte)
- sea_level (Meeresspiegel-Höhe in Metern)
- bank_width (Radius für Ufer-Biome in Pixeln)
- edge_softness (Globaler Weichheits-Faktor für alle Super-Biome-Übergänge, 0.1-2.0)
- alpine_level (Basis-Höhe für Alpine-Zone in Metern)
- snow_level (Basis-Höhe für Schneegrenze in Metern)
- cliff_slope (Grenzwert für Klippen-Klassifikation in Grad)

data_manager Input:
- map_seed (Globaler Karten-Seed für reproduzierbare Zufallswerte)
- heightmap (2D-Array in meter Altitude) - REQUIRED
- slopemap (2D-Array in m/m mit dz/dx, dz/dy) - REQUIRED
- temp_map (2D-Array in °C) - REQUIRED
- soil_moist_map (2D-Array in Bodenfeuchtigkeit %) - OPTIONAL (Fallback: Höhen-basiert)
- water_biomes_map (2D-Array mit Wasser-Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake) - OPTIONAL (Fallback: alle 0)

Output:
- biome_map (2D-Array mit Index der jeweils dominantesten Biomklasse)
- biome_map_super (2D-Array, 2x supersampled zur Darstellung gemischter Biome)
- super_biome_mask (2D-Array, Maske welche Pixel von Super-Biomes überschrieben wurden)

BASE_BIOME_CLASSIFICATIONS = {
    'ice_cap': 'T=-40--5 | soil_moist=0-300 | h=0-8000 | slope=0-90',
    'tundra': 'T=-15-5 | soil_moist=100-600 | h=0-2000 | slope=0-30',
    'taiga': 'T=-10-15 | soil_moist=300-1200 | h=50-2500 | slope=0-45',
    'grassland': 'T=0-25 | soil_moist=200-800 | h=10-1500 | slope=0-15',
    'temperate_forest': 'T=5-25 | soil_moist=600-2000 | h=0-2000 | slope=0-60',
    'mediterranean': 'T=8-30 | soil_moist=300-900 | h=0-1200 | slope=0-45',
    'desert': 'T=10-50 | soil_moist=0-250 | h=0-2000 | slope=0-30',
    'semi_arid': 'T=5-35 | soil_moist=200-600 | h=0-1800 | slope=0-25',
    'tropical_rainforest': 'T=20-35 | soil_moist=1500-4000 | h=0-1500 | slope=0-70',
    'tropical_seasonal': 'T=18-35 | soil_moist=800-2000 | h=0-1200 | slope=0-40',
    'savanna': 'T=15-35 | soil_moist=400-1200 | h=0-1800 | slope=0-20',
    'montane_forest': 'T=0-20 | soil_moist=800-3000 | h=800-3500 | slope=5-80',
    'swamp': 'T=5-35 | soil_moist=800-3000 | h=0-200 | slope=0-5',
    'coastal_dunes': 'T=5-35 | soil_moist=300-1500 | h=0-100 | slope=5-45',
    'badlands': 'T=-5-45 | soil_moist=0-400 | h=200-2500 | slope=15-90'
}

SUPER_BIOME_CONDITIONS = {
    'ocean': {
        'condition': 'lokales Minimum + Randverbindung + h < sea_level',
        'description': 'Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind und unter sea_level liegen',
        'priority': 0
    },
    'lake': {
        'condition': 'water_biomes_map == 4',
        'description': 'Direkt aus water_biomes_map übernommen',
        'priority': 1
    },
    'grand_river': {
        'condition': 'water_biomes_map == 3',
        'description': 'Direkt aus water_biomes_map übernommen',
        'priority': 2
    },
    'river': {
        'condition': 'water_biomes_map == 2',
        'description': 'Direkt aus water_biomes_map übernommen',
        'priority': 3
    },
    'creek': {
        'condition': 'water_biomes_map == 1',
        'description': 'Direkt aus water_biomes_map übernommen',
        'priority': 4
    },
    'cliff': {
        'condition': 'slope_degrees > cliff_slope',
        'description': 'Klippe ab einem Grenzwert cliff_slope mit weichen Übergängen',
        'priority': 5,
        'soft_transition': True,
        'probability_formula': 'sigmoid((slope_degrees - cliff_slope) / edge_softness)'
    },
    'beach': {
        'condition': 'Nähe zu Ocean + h <= sea_level + 5',
        'description': 'Innerhalb bank_width Distanz zu Ocean-Pixeln mit weichen Rändern',
        'priority': 6,
        'soft_transition': True,
        'probability_formula': 'max(0, 1 - (distance_to_ocean / bank_width)^edge_softness)'
    },
    'lake_edge': {
        'condition': 'Nähe zu Lake + nicht selbst Lake',
        'description': 'Innerhalb bank_width Distanz zu Lake-Pixeln mit weichen Rändern',
        'priority': 7,
        'soft_transition': True,
        'probability_formula': 'max(0, 1 - (distance_to_lake / bank_width)^edge_softness)'
    },
    'river_bank': {
        'condition': 'Nähe zu River/Grand River/Creek + nicht selbst Wasser',
        'description': 'Innerhalb bank_width Distanz zu Fließgewässern mit weichen Rändern',
        'priority': 8,
        'soft_transition': True,
        'probability_formula': 'max(0, 1 - (distance_to_water / bank_width)^edge_softness)'
    },
    'snow_level': {
        'condition': 'h > snow_level + 500*(1 + temp_map(x,y)/10)',
        'description': 'Schneegrenze mit graduellen, temperaturabhängigen Übergängen',
        'priority': 9,
        'soft_transition': True,
        'probability_formula': 'sigmoid((h - (snow_level + 500*(1 + temp/10))) / (100 * edge_softness))'
    },
    'alpine_level': {
        'condition': 'h > alpine_level + 500*(1 + temp_map(x,y)/10)',
        'description': 'Alpine Zone mit graduellen, temperaturabhängigen Übergängen',
        'priority': 10,
        'soft_transition': True,
        'probability_formula': 'sigmoid((h - (alpine_level + 500*(1 + temp/10))) / (200 * edge_softness))'
    }
}

Klassen:
BiomeClassificationSystem (BaseGenerator)
    Funktionsweise: Hauptklasse für Biom-Klassifikation mit BaseGenerator-API und LOD-System
    Aufgabe: Koordiniert Base-Biome und Super-Biome Zuordnung mit Supersampling und Progress-Updates
    Methoden: generate(), _execute_generation(), _load_default_parameters(), _get_dependencies()

BaseBiomeClassifier
    Funktionsweise: Gauß-basierte Klassifizierung von 15 Grundbiomen mit LOD-Optimierung
    Aufgabe: Erstellt biome_map basierend auf Höhe, Temperatur, Niederschlag und Slope
    Methoden: calculate_gaussian_fitness(), weight_environmental_factors(), assign_dominant_biome()

SuperBiomeOverrideSystem
    Funktionsweise: Überschreibt Base-Biomes mit speziellen Bedingungen (Ocean, Cliff, Beach, etc.)
    Aufgabe: Erstellt super_biome_mask für prioritätsbasierte Biom-Überschreibung
    Methoden: detect_ocean_connectivity(), apply_proximity_biomes(), calculate_elevation_biomes()

SupersamplingManager
    Funktionsweise: 2x2 Supersampling mit diskretisierter Zufalls-Rotation (nur bei LOD256+)
    Aufgabe: Erstellt biome_map_super für weiche Übergänge zwischen Biomen bei höheren LODs
    Methoden: apply_rotational_supersampling(), calculate_soft_transitions(), optimize_spatial_distribution()

ProximityBiomeCalculator
    Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
    Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
    Methoden: calculate_distance_fields(), apply_gaussian_proximity(), blend_with_base_biomes()
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from collections import deque
from core.base_generator import BaseGenerator


class BiomeData:
    """
    Funktionsweise: Container für alle Biome-Daten mit LOD-Informationen
    Aufgabe: Speichert biome_map, biome_map_super, super_biome_mask und Metainformationen
    """
    def __init__(self):
        self.biome_map = None           # (height, width) - Dominante Biom-Indices
        self.biome_map_super = None     # (height*2, width*2) - Supersampeltes Biom-Map (nur bei LOD256+)
        self.super_biome_mask = None    # (height, width) - Super-Biome-Override-Maske
        self.lod_level = "LOD64"        # Aktueller LOD-Level
        self.actual_size = 64           # Tatsächliche Kartengröße
        self.supersampling_enabled = False  # Ob Supersampling angewendet wurde
        self.parameters = {}            # Verwendete Parameter für Cache-Management


class BaseBiomeClassifier:
    """
    Funktionsweise: Gauß-basierte Klassifizierung von 15 Grundbiomen mit LOD-Optimierung
    Aufgabe: Erstellt biome_map basierend auf Höhe, Temperatur, Niederschlag und Slope
    """

    def __init__(self, biome_wetness_factor=1.0, biome_temp_factor=1.0):
        """
        Funktionsweise: Initialisiert Base-Biome-Classifier mit Gewichtungsfaktoren
        Aufgabe: Setup der Gauß-basierten Biom-Klassifikation
        Parameter: biome_wetness_factor, biome_temp_factor - Gewichtung von Feuchtigkeit und Temperatur
        """
        self.wetness_factor = biome_wetness_factor
        self.temp_factor = biome_temp_factor

        # Base-Biome Definitionen exakt wie dokumentiert
        self.base_biomes = {
            'ice_cap': {'temp': (-40, -5), 'moisture': (0, 300), 'elevation': (0, 8000), 'slope': (0, 90)},
            'tundra': {'temp': (-15, 5), 'moisture': (100, 600), 'elevation': (0, 2000), 'slope': (0, 30)},
            'taiga': {'temp': (-10, 15), 'moisture': (300, 1200), 'elevation': (50, 2500), 'slope': (0, 45)},
            'grassland': {'temp': (0, 25), 'moisture': (200, 800), 'elevation': (10, 1500), 'slope': (0, 15)},
            'temperate_forest': {'temp': (5, 25), 'moisture': (600, 2000), 'elevation': (0, 2000), 'slope': (0, 60)},
            'mediterranean': {'temp': (8, 30), 'moisture': (300, 900), 'elevation': (0, 1200), 'slope': (0, 45)},
            'desert': {'temp': (10, 50), 'moisture': (0, 250), 'elevation': (0, 2000), 'slope': (0, 30)},
            'semi_arid': {'temp': (5, 35), 'moisture': (200, 600), 'elevation': (0, 1800), 'slope': (0, 25)},
            'tropical_rainforest': {'temp': (20, 35), 'moisture': (1500, 4000), 'elevation': (0, 1500), 'slope': (0, 70)},
            'tropical_seasonal': {'temp': (18, 35), 'moisture': (800, 2000), 'elevation': (0, 1200), 'slope': (0, 40)},
            'savanna': {'temp': (15, 35), 'moisture': (400, 1200), 'elevation': (0, 1800), 'slope': (0, 20)},
            'montane_forest': {'temp': (0, 20), 'moisture': (800, 3000), 'elevation': (800, 3500), 'slope': (5, 80)},
            'swamp': {'temp': (5, 35), 'moisture': (800, 3000), 'elevation': (0, 200), 'slope': (0, 5)},
            'coastal_dunes': {'temp': (5, 35), 'moisture': (300, 1500), 'elevation': (0, 100), 'slope': (5, 45)},
            'badlands': {'temp': (-5, 45), 'moisture': (0, 400), 'elevation': (200, 2500), 'slope': (15, 90)}
        }

        # Biome-Namen zu Index-Mapping
        self.biome_names = list(self.base_biomes.keys())
        self.biome_indices = {name: i for i, name in enumerate(self.biome_names)}

    def calculate_gaussian_fitness(self, temp_map, soil_moist_map, heightmap, slopemap, progress_callback=None):
        """
        Funktionsweise: Berechnet Gauß-Passung für alle Biome an allen Positionen mit Progress-Updates
        Aufgabe: Erstellt Fitness-Maps für jeden Biom-Typ basierend auf Umweltparametern
        Parameter: temp_map, soil_moist_map, heightmap, slopemap - Alle Umwelt-Daten
        Parameter: progress_callback - Callback-Funktion für Progress-Updates
        Returns: numpy.ndarray - Fitness-Werte (height, width, num_biomes)
        """
        height, width = temp_map.shape
        num_biomes = len(self.base_biomes)
        fitness_maps = np.zeros((height, width, num_biomes), dtype=np.float32)

        # Slope in Grad konvertieren
        slope_degrees = np.degrees(np.arctan(np.sqrt(slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2)))

        # Progress-Update: Start der Biom-Fitness-Berechnung
        if progress_callback:
            progress_callback("Base Biomes", 20, "Calculating Gaussian fitness for all biomes...")

        for biome_idx, (biome_name, biome_params) in enumerate(self.base_biomes.items()):
            temp_range = biome_params['temp']
            moisture_range = biome_params['moisture']
            elevation_range = biome_params['elevation']
            slope_range = biome_params['slope']

            # Gauß-Fitness für jeden Parameter
            temp_fitness = self._gaussian_range_fitness(temp_map, temp_range)
            moisture_fitness = self._gaussian_range_fitness(soil_moist_map, moisture_range)
            elevation_fitness = self._gaussian_range_fitness(heightmap, elevation_range)
            slope_fitness = self._gaussian_range_fitness(slope_degrees, slope_range)

            # Gewichtete Kombination wie dokumentiert
            combined_fitness = (
                    temp_fitness * 0.30 * self.temp_factor +
                    moisture_fitness * 0.35 * self.wetness_factor +
                    elevation_fitness * 0.20 +
                    slope_fitness * 0.15
            )

            fitness_maps[:, :, biome_idx] = combined_fitness

            # Progress-Update pro Biom
            if progress_callback:
                biome_progress = 20 + (biome_idx + 1) * 15 // num_biomes
                progress_callback("Base Biomes", biome_progress, f"Calculated fitness for {biome_name}")

        return fitness_maps

    def _gaussian_range_fitness(self, data_map, value_range):
        """
        Funktionsweise: Berechnet Gauß-Fitness für gegebenen Wertebereich
        Aufgabe: Optimale Fitness in Range-Mitte, abfallend zu Rändern
        Parameter: data_map, value_range - Daten-Array und (min, max) Bereich
        Returns: numpy.ndarray - Fitness-Werte zwischen 0 und 1
        """
        min_val, max_val = value_range
        range_center = (min_val + max_val) / 2
        range_width = max_val - min_val

        if range_width == 0:
            # Singulär-Punkt: perfekte Fitness nur bei exaktem Wert
            return np.where(data_map == min_val, 1.0, 0.0)

        # Gauß-Funktion: Maximum bei Center, Sigma = range_width/4
        sigma = range_width / 4.0
        normalized_distance = (data_map - range_center) / sigma
        fitness = np.exp(-0.5 * normalized_distance ** 2)

        # Außerhalb des Bereichs: stark reduzierte Fitness
        outside_range = (data_map < min_val) | (data_map > max_val)
        fitness[outside_range] *= 0.1

        return fitness

    def assign_dominant_biome(self, fitness_maps, progress_callback=None):
        """
        Funktionsweise: Weist dominantes Biom basierend auf höchster Fitness zu
        Aufgabe: Erstellt biome_map mit Index des dominantesten Bioms pro Pixel
        Parameter: fitness_maps - Fitness-Arrays aller Biome
        Parameter: progress_callback - Callback-Funktion für Progress-Updates
        Returns: numpy.ndarray - Biome-Indices (height, width)
        """
        if progress_callback:
            progress_callback("Base Biomes", 35, "Assigning dominant biomes...")

        # Argmax über Biom-Dimension
        dominant_biomes = np.argmax(fitness_maps, axis=2)
        return dominant_biomes.astype(np.uint8)


class SuperBiomeOverrideSystem:
    """
    Funktionsweise: Überschreibt Base-Biomes mit speziellen Bedingungen (Ocean, Cliff, Beach, etc.)
    Aufgabe: Erstellt super_biome_mask für prioritätsbasierte Biom-Überschreibung
    """

    def __init__(self, sea_level=0.0, bank_width=10.0, edge_softness=1.0, alpine_level=2000.0, snow_level=3000.0,
                 cliff_slope=45.0):
        """
        Funktionsweise: Initialisiert Super-Biome-Override-System mit allen Parametern
        Aufgabe: Setup aller Super-Biome-Schwellwerte und Übergangs-Parameter
        """
        self.sea_level = sea_level
        self.bank_width = bank_width
        self.edge_softness = edge_softness
        self.alpine_level = alpine_level
        self.snow_level = snow_level
        self.cliff_slope = cliff_slope

        # Super-Biome Indices (nach Base-Biomes)
        self.super_biome_offset = 15  # Erste 15 sind Base-Biomes
        self.super_biomes = {
            'ocean': self.super_biome_offset + 0,
            'lake': self.super_biome_offset + 1,
            'grand_river': self.super_biome_offset + 2,
            'river': self.super_biome_offset + 3,
            'creek': self.super_biome_offset + 4,
            'cliff': self.super_biome_offset + 5,
            'beach': self.super_biome_offset + 6,
            'lake_edge': self.super_biome_offset + 7,
            'river_bank': self.super_biome_offset + 8,
            'snow_level': self.super_biome_offset + 9,
            'alpine_level': self.super_biome_offset + 10
        }

    def detect_ocean_connectivity(self, heightmap, water_biomes_map, progress_callback=None):
        """
        Funktionsweise: Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind und unter sea_level liegen
        Aufgabe: Identifiziert Ocean-Bereiche durch Rand-Konnektivität
        Parameter: heightmap, water_biomes_map - Höhen und Wasser-Klassifikation
        Parameter: progress_callback - Callback-Funktion für Progress-Updates
        Returns: numpy.ndarray - Ocean-Maske (bool)
        """
        if progress_callback:
            progress_callback("Super Biomes", 45, "Detecting ocean connectivity...")

        height, width = heightmap.shape
        ocean_mask = np.zeros((height, width), dtype=bool)

        # Finde alle lokalen Minima unter sea_level
        potential_ocean = heightmap < self.sea_level

        if not np.any(potential_ocean):
            return ocean_mask

        # Flood-Fill von Kartenrändern
        visited = np.zeros((height, width), dtype=bool)
        queue = deque()

        # Alle Rand-Pixel die unter sea_level sind als Seeds
        for y in range(height):
            for x in range(width):
                is_edge = (x == 0 or x == width - 1 or y == 0 or y == height - 1)
                if is_edge and heightmap[y, x] < self.sea_level:
                    queue.append((x, y))
                    visited[y, x] = True
                    ocean_mask[y, x] = True

        # Flood-Fill
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (0 <= nx < width and 0 <= ny < height and
                        not visited[ny, nx] and heightmap[ny, nx] < self.sea_level):
                    visited[ny, nx] = True
                    ocean_mask[ny, nx] = True
                    queue.append((nx, ny))

        return ocean_mask

    def apply_proximity_biomes(self, water_biomes_map, heightmap, progress_callback=None):
        """
        Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
        Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
        Parameter: water_biomes_map, heightmap - Wasser-Klassifikation und Höhen
        Parameter: progress_callback - Callback-Funktion für Progress-Updates
        Returns: dict - Proximity-Masken für verschiedene Ufer-Biome
        """
        if progress_callback:
            progress_callback("Super Biomes", 50, "Calculating proximity biomes...")

        height, width = water_biomes_map.shape
        proximity_masks = {}

        # Ocean-Mask für Beach-Berechnung
        ocean_mask = self.detect_ocean_connectivity(heightmap, water_biomes_map)

        # Beach: Nähe zu Ocean + h <= sea_level + 5
        if np.any(ocean_mask):
            ocean_distance = distance_transform_edt(~ocean_mask)
            beach_condition = ((ocean_distance <= self.bank_width) &
                               (heightmap <= self.sea_level + 5) &
                               (~ocean_mask))

            # Weiche Übergänge mit edge_softness
            beach_probability = np.maximum(0, 1 - np.power(ocean_distance / self.bank_width, self.edge_softness))
            beach_probability[~beach_condition] = 0
            proximity_masks['beach'] = beach_probability
        else:
            proximity_masks['beach'] = np.zeros((height, width), dtype=np.float32)

        # Lake Edge: Nähe zu Lake + nicht selbst Lake
        lake_mask = water_biomes_map == 4
        if np.any(lake_mask):
            lake_distance = distance_transform_edt(~lake_mask)
            lake_edge_condition = ((lake_distance <= self.bank_width) & (~lake_mask))

            lake_edge_probability = np.maximum(0, 1 - np.power(lake_distance / self.bank_width, self.edge_softness))
            lake_edge_probability[~lake_edge_condition] = 0
            proximity_masks['lake_edge'] = lake_edge_probability
        else:
            proximity_masks['lake_edge'] = np.zeros((height, width), dtype=np.float32)

        # River Bank: Nähe zu River/Grand River/Creek + nicht selbst Wasser
        river_mask = (water_biomes_map >= 1) & (water_biomes_map <= 3)
        if np.any(river_mask):
            river_distance = distance_transform_edt(~river_mask)
            river_bank_condition = ((river_distance <= self.bank_width) & (water_biomes_map == 0))

            river_bank_probability = np.maximum(0, 1 - np.power(river_distance / self.bank_width, self.edge_softness))
            river_bank_probability[~river_bank_condition] = 0
            proximity_masks['river_bank'] = river_bank_probability
        else:
            proximity_masks['river_bank'] = np.zeros((height, width), dtype=np.float32)

        return proximity_masks

    def calculate_elevation_biomes(self, heightmap, temp_map, slopemap, progress_callback=None):
        """
        Funktionsweise: Berechnet Höhen-basierte Super-Biomes (Snow, Alpine, Cliff)
        Aufgabe: Erstellt Höhen- und Slope-abhängige Super-Biome mit weichen Übergängen
        Parameter: heightmap, temp_map, slopemap - Höhen, Temperatur und Slope-Daten
        Parameter: progress_callback - Callback-Funktion für Progress-Updates
        Returns: dict - Elevation-basierte Super-Biome Masken
        """
        if progress_callback:
            progress_callback("Super Biomes", 55, "Calculating elevation biomes...")

        height, width = heightmap.shape
        elevation_masks = {}

        # Slope in Grad konvertieren
        slope_degrees = np.degrees(np.arctan(np.sqrt(slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2)))

        # Cliff: slope_degrees > cliff_slope mit weichen Übergängen
        cliff_probability = self._sigmoid((slope_degrees - self.cliff_slope) / self.edge_softness)
        cliff_probability = np.maximum(0, cliff_probability)
        elevation_masks['cliff'] = cliff_probability

        # Snow Level: h > snow_level + 500*(1 + temp_map(x,y)/10) mit temperaturabhängigen Übergängen
        temp_adjusted_snow_level = self.snow_level + 500 * (1 + temp_map / 10)
        snow_height_diff = heightmap - temp_adjusted_snow_level
        snow_probability = self._sigmoid(snow_height_diff / (100 * self.edge_softness))
        snow_probability = np.maximum(0, snow_probability)
        elevation_masks['snow_level'] = snow_probability

        # Alpine Level: h > alpine_level + 500*(1 + temp_map(x,y)/10) mit temperaturabhängigen Übergängen
        temp_adjusted_alpine_level = self.alpine_level + 500 * (1 + temp_map / 10)
        alpine_height_diff = heightmap - temp_adjusted_alpine_level
        alpine_probability = self._sigmoid(alpine_height_diff / (200 * self.edge_softness))
        alpine_probability = np.maximum(0, alpine_probability)
        elevation_masks['alpine_level'] = alpine_probability

        return elevation_masks

    def _sigmoid(self, x):
        """
        Funktionsweise: Sigmoid-Funktion für weiche Übergänge
        Aufgabe: Konvertiert kontinuierliche Werte zu Wahrscheinlichkeiten
        Parameter: x - Input-Werte
        Returns: Sigmoid-Output zwischen 0 und 1
        """
        # Numerisch stabile Sigmoid-Implementation
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))


class SupersamplingManager:
    """
    Funktionsweise: 2x2 Supersampling mit diskretisierter Zufalls-Rotation (nur bei LOD256+)
    Aufgabe: Erstellt biome_map_super für weiche Übergänge zwischen Biomen bei höheren LODs
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Supersampling-Manager mit Zufalls-Seed
        Aufgabe: Setup der reproduzierbaren Supersampling-Rotation
        Parameter: map_seed (int) - Seed für reproduzierbare Zufälligkeit
        """
        self.map_seed = map_seed

    def should_apply_supersampling(self, lod_level, target_size):
        """
        Funktionsweise: Bestimmt ob Supersampling für gegebenes LOD angewendet werden soll
        Aufgabe: Performance-Optimierung - Supersampling nur bei höheren LODs
        Parameter: lod_level (str), target_size (int) - LOD-Level und Zielgröße
        Returns: bool - True wenn Supersampling angewendet werden soll
        """
        # Supersampling nur bei LOD256+ oder FINAL mit Größe >= 256
        if lod_level in ["LOD256", "LOD512", "LOD1024"] or (lod_level == "FINAL" and target_size >= 256):
            return True
        return False

    def apply_rotational_supersampling(self, biome_map, super_biome_probabilities=None, progress_callback=None):
        """
        Funktionsweise: Wendet 2x2 Supersampling mit diskretisierter Zufalls-Rotation an
        Aufgabe: Erstellt 2x supersampeltes Biom-Map mit optimierter räumlicher Verteilung
        Parameter: biome_map, super_biome_probabilities - Base-Biomes und Super-Biome-Wahrscheinlichkeiten
        Parameter: progress_callback - Callback-Funktion für Progress-Updates
        Returns: numpy.ndarray - 2x supersampeltes Biom-Map
        """
        if progress_callback:
            progress_callback("Supersampling", 80, "Applying 2x2 supersampling with rotational distribution...")

        height, width = biome_map.shape
        super_height, super_width = height * 2, width * 2
        biome_map_super = np.zeros((super_height, super_width), dtype=np.uint8)

        # Progress-Update für Supersampling-Schleife
        total_pixels = height * width
        processed_pixels = 0

        for y in range(height):
            for x in range(width):
                # Diskretisierte Rotations-Zuweisung
                rotation_hash = (self.map_seed + 12345 + x * 997 + y * 991) % 4

                # Sub-Pixel-Anordnung basierend auf Rotation
                if rotation_hash == 0:  # 0° Rotation
                    sub_order = [(0, 0), (0, 1), (1, 0), (1, 1)]  # TL, TR, BL, BR
                elif rotation_hash == 1:  # 90° Rotation
                    sub_order = [(1, 0), (0, 0), (1, 1), (0, 1)]  # BL, TL, BR, TR
                elif rotation_hash == 2:  # 180° Rotation
                    sub_order = [(1, 1), (1, 0), (0, 1), (0, 0)]  # BR, BL, TR, TL
                else:  # 270° Rotation
                    sub_order = [(0, 1), (1, 1), (0, 0), (1, 0)]  # TR, BR, TL, BL

                base_biome = biome_map[y, x]

                # Sub-Pixel-Zuweisung mit optimierter Zufälligkeit
                for i, (sub_y, sub_x) in enumerate(sub_order):
                    target_y = y * 2 + sub_y
                    target_x = x * 2 + sub_x

                    # Sub-Pixel-Seed für Wahrscheinlichkeits-Berechnung
                    sub_seed = (self.map_seed + 54321 + x * 4 + y * 4 + i * 3571) % 1000
                    probability = sub_seed / 1000.0  # [0.000, 0.999]

                    # Super-Biome-Zuweisung prüfen
                    assigned_biome = base_biome

                    if super_biome_probabilities is not None:
                        for super_biome_name, prob_map in super_biome_probabilities.items():
                            if y < prob_map.shape[0] and x < prob_map.shape[1]:
                                super_prob = prob_map[y, x]
                                if probability < super_prob:
                                    # Super-Biome zuweisen (Index wird später gemappt)
                                    assigned_biome = self._get_super_biome_index(super_biome_name)
                                    break

                    biome_map_super[target_y, target_x] = assigned_biome

                # Progress-Update alle 10% der Pixel
                processed_pixels += 1
                if processed_pixels % (total_pixels // 10) == 0 and progress_callback:
                    progress_pct = 80 + int((processed_pixels / total_pixels) * 15)
                    progress_callback("Supersampling", progress_pct, f"Supersampling progress: {processed_pixels}/{total_pixels} pixels")

        return biome_map_super

    def _get_super_biome_index(self, super_biome_name):
        """
        Funktionsweise: Konvertiert Super-Biome-Namen zu Index
        Aufgabe: Mapping von Super-Biome-Namen zu numerischen Indices
        """
        super_biome_offset = 15  # Nach Base-Biomes
        super_biome_mapping = {
            'ocean': super_biome_offset + 0,
            'lake': super_biome_offset + 1,
            'grand_river': super_biome_offset + 2,
            'river': super_biome_offset + 3,
            'creek': super_biome_offset + 4,
            'cliff': super_biome_offset + 5,
            'beach': super_biome_offset + 6,
            'lake_edge': super_biome_offset + 7,
            'river_bank': super_biome_offset + 8,
            'snow_level': super_biome_offset + 9,
            'alpine_level': super_biome_offset + 10
        }

        return super_biome_mapping.get(super_biome_name, 0)


class BiomeClassificationSystem(BaseGenerator):
    """
    Funktionsweise: Hauptklasse für Biom-Klassifikation mit BaseGenerator-API und LOD-System
    Aufgabe: Koordiniert Base-Biome und Super-Biome Zuordnung mit Supersampling und Progress-Updates
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Biome-Classification-System mit BaseGenerator und Sub-Komponenten
        Aufgabe: Setup aller Biom-Klassifikations-Systeme
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Biom-Verteilung
        """
        super().__init__(map_seed)

        # Standard-Parameter (werden durch _load_default_parameters überschrieben)
        self.biome_wetness_factor = 1.0
        self.biome_temp_factor = 1.0
        self.sea_level = 10.0
        self.bank_width = 3.0
        self.edge_softness = 1.0
        self.alpine_level = 1500.0
        self.snow_level = 2000.0
        self.cliff_slope = 60.0

    def _load_default_parameters(self):
        """
        Funktionsweise: Lädt BIOME-Parameter aus value_default.py
        Aufgabe: Standard-Parameter für Biome-Generierung
        Returns: dict - Alle Standard-Parameter für Biome
        """
        from gui.config.value_default import BIOME

        return {
            'biome_wetness_factor': BIOME.BIOME_WETNESS_FACTOR["default"],
            'biome_temp_factor': BIOME.BIOME_TEMP_FACTOR["default"],
            'sea_level': BIOME.SEA_LEVEL["default"],
            'bank_width': BIOME.BANK_WIDTH["default"],
            'edge_softness': BIOME.EDGE_SOFTNESS["default"],
            'alpine_level': BIOME.ALPINE_LEVEL["default"],
            'snow_level': BIOME.SNOW_LEVEL["default"],
            'cliff_slope': BIOME.CLIFF_SLOPE["default"]
        }

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Holt benötigte Dependencies mit intelligenten Fallback-Werten
        Aufgabe: Dependency-Resolution für Biome-Generierung mit optionalen Inputs
        Parameter: data_manager - DataManager-Instanz
        Returns: dict - Alle Input-Daten (required + optional mit Fallbacks)
        """
        if not data_manager:
            raise Exception("DataManager required for Biome generation")

        dependencies = {}

        # REQUIRED Dependencies - müssen vorhanden sein
        required_deps = ['heightmap', 'slopemap', 'temp_map']

        for dep_name in required_deps:
            # Versuche verschiedene DataManager-Bereiche
            data = None

            # Terrain-Daten
            if dep_name in ['heightmap', 'slopemap']:
                terrain_data = data_manager.get_terrain_data("complete")
                if terrain_data:
                    data = getattr(terrain_data, dep_name, None)
                if data is None:
                    data = data_manager.get_terrain_data(dep_name)

            # Weather-Daten
            elif dep_name == 'temp_map':
                data = data_manager.get_weather_data(dep_name)

            if data is None:
                raise Exception(f"Required dependency '{dep_name}' not available in DataManager")

            dependencies[dep_name] = data

        # OPTIONAL Dependencies - erstelle Fallback-Werte wenn nicht vorhanden
        heightmap = dependencies['heightmap']

        # soil_moist_map: Fallback basierend auf Höhe und Temperatur
        soil_moist_map = data_manager.get_water_data('soil_moist_map')
        if soil_moist_map is None:
            self.logger.warning("soil_moist_map not available, creating elevation-based fallback")
            soil_moist_map = self._create_fallback_soil_moisture(heightmap, dependencies['temp_map'])
        dependencies['soil_moist_map'] = soil_moist_map

        # water_biomes_map: Fallback - alle Pixel als "kein Wasser" (0)
        water_biomes_map = data_manager.get_water_data('water_biomes_map')
        if water_biomes_map is None:
            self.logger.warning("water_biomes_map not available, creating fallback (no water)")
            water_biomes_map = np.zeros_like(heightmap, dtype=np.uint8)
        dependencies['water_biomes_map'] = water_biomes_map

        self.logger.debug(f"Dependencies loaded - heightmap: {heightmap.shape}, temp_map: {dependencies['temp_map'].shape}")
        self.logger.debug(f"Optional deps - soil_moist_map: {'fallback' if soil_moist_map is dependencies.get('soil_moist_map') else 'original'}, water_biomes_map: {'fallback' if np.all(water_biomes_map == 0) else 'original'}")

        return dependencies

    def _create_fallback_soil_moisture(self, heightmap, temp_map):
        """
        Funktionsweise: Erstellt intelligente Fallback soil_moist_map basierend auf Höhe und Temperatur
        Aufgabe: Realistische Bodenfeuchtigkeit wenn Water-Generator noch nicht gelaufen ist
        Parameter: heightmap, temp_map - Höhen- und Temperaturdaten
        Returns: numpy.ndarray - Fallback soil_moist_map
        """
        height, width = heightmap.shape
        soil_moist_map = np.zeros((height, width), dtype=np.float32)

        # Normalisierte Höhe [0, 1]
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range > 0:
            norm_height = (heightmap - min_height) / height_range
        else:
            norm_height = np.zeros_like(heightmap)

        # Bodenfeuchtigkeit basierend auf Höhe und Temperatur
        for y in range(height):
            for x in range(width):
                elevation_factor = norm_height[y, x]
                temperature = temp_map[y, x]

                # Höhere Lagen: weniger Feuchtigkeit
                # Niedrigere Temperaturen: mehr Feuchtigkeit
                elevation_moisture = (1.0 - elevation_factor) * 500  # 0-500 range
                temp_moisture = max(0, (20 - temperature) * 20)  # Kühlere Bereiche mehr Feuchtigkeit

                # Kombiniere beide Faktoren
                total_moisture = elevation_moisture + temp_moisture

                # Realistische Range: 0-1000 gH2O/m³
                soil_moist_map[y, x] = min(1000, max(0, total_moisture))

        return soil_moist_map

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Funktionsweise: Führt Biome-Generierung mit Progress-Updates aus
        Aufgabe: Kernlogik der Biome-Generierung mit allen 4 Hauptschritten
        Parameter: lod, dependencies, parameters
        Returns: BiomeData-Objekt mit allen Biome-Outputs
        """
        heightmap = dependencies['heightmap']
        slopemap = dependencies['slopemap']
        temp_map = dependencies['temp_map']
        soil_moist_map = dependencies['soil_moist_map']
        water_biomes_map = dependencies['water_biomes_map']

        # Parameter aktualisieren
        self.biome_wetness_factor = parameters['biome_wetness_factor']
        self.biome_temp_factor = parameters['biome_temp_factor']
        self.sea_level = parameters['sea_level']
        self.bank_width = parameters['bank_width']
        self.edge_softness = parameters['edge_softness']
        self.alpine_level = parameters['alpine_level']
        self.snow_level = parameters['snow_level']
        self.cliff_slope = parameters['cliff_slope']

        # LOD-Größe bestimmen
        target_size = self._get_lod_size(lod, heightmap.shape[0])

        # Alle Arrays auf Zielgröße interpolieren falls nötig
        if heightmap.shape[0] != target_size:
            heightmap = self._interpolate_array(heightmap, target_size)
            slopemap = self._interpolate_array(slopemap, target_size)
            temp_map = self._interpolate_array(temp_map, target_size)
            soil_moist_map = self._interpolate_array(soil_moist_map, target_size)
            water_biomes_map = self._interpolate_array(water_biomes_map, target_size)

        # Schritt 1: Base-Biome Klassifikation (20% - 40%)
        self._update_progress("Base Biomes", 20, "Starting Gaussian biome classification...")
        base_classifier = BaseBiomeClassifier(self.biome_wetness_factor, self.biome_temp_factor)
        fitness_maps = base_classifier.calculate_gaussian_fitness(
            temp_map, soil_moist_map, heightmap, slopemap, self._update_progress
        )
        base_biome_map = base_classifier.assign_dominant_biome(fitness_maps, self._update_progress)

        # Schritt 2: Super-Biome Override (40% - 65%)
        self._update_progress("Super Biomes", 40, "Calculating super-biome overrides...")
        super_biome_system = SuperBiomeOverrideSystem(
            self.sea_level, self.bank_width, self.edge_softness,
            self.alpine_level, self.snow_level, self.cliff_slope
        )

        # Wasser-Biome direkt übernehmen
        super_biome_mask = np.zeros_like(base_biome_map, dtype=np.uint8)

        # Priority 0-4: Wasser-Biome
        super_biome_mask[water_biomes_map == 4] = super_biome_system.super_biomes['lake']  # Lake
        super_biome_mask[water_biomes_map == 3] = super_biome_system.super_biomes['grand_river']  # Grand River
        super_biome_mask[water_biomes_map == 2] = super_biome_system.super_biomes['river']  # River
        super_biome_mask[water_biomes_map == 1] = super_biome_system.super_biomes['creek']  # Creek

        # Ocean-Detection
        ocean_mask = super_biome_system.detect_ocean_connectivity(heightmap, water_biomes_map, self._update_progress)
        super_biome_mask[ocean_mask] = super_biome_system.super_biomes['ocean']

        # Proximity-Biomes
        proximity_masks = super_biome_system.apply_proximity_biomes(water_biomes_map, heightmap, self._update_progress)

        # Elevation-Biomes
        elevation_masks = super_biome_system.calculate_elevation_biomes(heightmap, temp_map, slopemap, self._update_progress)

        # Finale Biom-Map: Base-Biomes + Super-Biome-Overrides
        final_biome_map = base_biome_map.copy()

        # Wasser-Biome überschreiben (höchste Priorität)
        water_override = super_biome_mask > 0
        final_biome_map[water_override] = super_biome_mask[water_override]

        # Schritt 3: Super-Biome Integration (65% - 75%)
        self._update_progress("Integration", 65, "Integrating proximity and elevation biomes...")
        all_super_masks = {**proximity_masks, **elevation_masks}
        integrated_biome_map = self._integrate_super_biomes(final_biome_map, all_super_masks)

        # Schritt 4: Supersampling (75% - 95%) - nur bei höheren LODs
        supersampling_manager = SupersamplingManager(self.map_seed)

        if supersampling_manager.should_apply_supersampling(lod, target_size):
            self._update_progress("Supersampling", 75, "Applying supersampling for enhanced detail...")
            biome_map_super = supersampling_manager.apply_rotational_supersampling(
                integrated_biome_map, all_super_masks, self._update_progress
            )
            supersampling_enabled = True
        else:
            self._update_progress("Supersampling", 75, "Skipping supersampling for this LOD level")
            biome_map_super = None
            supersampling_enabled = False

        # BiomeData-Objekt erstellen
        biome_data = BiomeData()
        biome_data.biome_map = integrated_biome_map
        biome_data.biome_map_super = biome_map_super
        biome_data.super_biome_mask = super_biome_mask
        biome_data.lod_level = lod
        biome_data.actual_size = target_size
        biome_data.supersampling_enabled = supersampling_enabled
        biome_data.parameters = parameters.copy()

        self.logger.debug(f"Biome generation complete - LOD: {lod}, size: {target_size}, supersampling: {supersampling_enabled}")

        return biome_data

    def _integrate_super_biomes(self, base_biome_map, super_biome_probabilities):
        """
        Funktionsweise: Integriert Super-Biome-Wahrscheinlichkeiten in Base-Biome-Map
        Aufgabe: Probabilistische Super-Biome-Überschreibung mit weichen Übergängen
        Parameter: base_biome_map, super_biome_probabilities - Base-Biomes und Super-Biome-Wahrscheinlichkeiten
        Returns: numpy.ndarray - Integrierte Biom-Map
        """
        height, width = base_biome_map.shape
        integrated_map = base_biome_map.copy()

        # Prioritäts-basierte Integration (nach dokumentierter Reihenfolge)
        priority_order = ['cliff', 'beach', 'lake_edge', 'river_bank', 'snow_level', 'alpine_level']

        for super_biome_name in priority_order:
            if super_biome_name in super_biome_probabilities:
                prob_map = super_biome_probabilities[super_biome_name]

                # Deterministisch: hohe Wahrscheinlichkeit überschreibt
                high_prob_mask = prob_map > 0.7
                super_biome_index = self._get_super_biome_index(super_biome_name)
                integrated_map[high_prob_mask] = super_biome_index

        return integrated_map

    def _get_super_biome_index(self, super_biome_name):
        """
        Funktionsweise: Konvertiert Super-Biome-Namen zu Index
        Aufgabe: Einheitliche Index-Zuordnung für Super-Biomes
        """
        super_biome_offset = 15
        super_biome_mapping = {
            'ocean': super_biome_offset + 0,
            'lake': super_biome_offset + 1,
            'grand_river': super_biome_offset + 2,
            'river': super_biome_offset + 3,
            'creek': super_biome_offset + 4,
            'cliff': super_biome_offset + 5,
            'beach': super_biome_offset + 6,
            'lake_edge': super_biome_offset + 7,
            'river_bank': super_biome_offset + 8,
            'snow_level': super_biome_offset + 9,
            'alpine_level': super_biome_offset + 10
        }

        return super_biome_mapping.get(super_biome_name, 0)

    def _save_to_data_manager(self, data_manager, result, parameters):
        """
        Funktionsweise: Speichert Biome-Ergebnisse im DataManager
        Aufgabe: Automatische Speicherung aller Biome-Outputs mit Parameter-Tracking
        Parameter: data_manager, result (BiomeData), parameters
        """
        if isinstance(result, BiomeData):
            # BiomeData-Objekt in einzelne Arrays aufteilen für DataManager
            data_manager.set_biome_data("biome_map", result.biome_map, parameters)

            if result.biome_map_super is not None:
                data_manager.set_biome_data("biome_map_super", result.biome_map_super, parameters)

            data_manager.set_biome_data("super_biome_mask", result.super_biome_mask, parameters)

            self.logger.debug(f"BiomeData object saved to DataManager (supersampling: {result.supersampling_enabled})")
        else:
            # Fallback für Legacy-Format (Tuple)
            if hasattr(result, '__len__') and len(result) >= 2:
                biome_map, biome_map_super, super_biome_mask = result[:3]
                data_manager.set_biome_data("biome_map", biome_map, parameters)
                if biome_map_super is not None:
                    data_manager.set_biome_data("biome_map_super", biome_map_super, parameters)
                data_manager.set_biome_data("super_biome_mask", super_biome_mask, parameters)
                self.logger.debug("Legacy biome data saved to DataManager")

    def _get_lod_size(self, lod, original_size):
        """
        Funktionsweise: Bestimmt Zielgröße basierend auf LOD-Level
        Aufgabe: LOD-System für Biome mit gleicher Logik wie Weather
        """
        lod_sizes = {"LOD64": 64, "LOD128": 128, "LOD256": 256, "LOD512": 512, "LOD1024": 1024}

        if lod == "FINAL":
            return original_size
        else:
            return lod_sizes.get(lod, 64)

    def _interpolate_array(self, array, target_size):
        """
        Funktionsweise: Interpoliert 2D-Array auf neue Größe mittels bilinearer Interpolation
        Aufgabe: LOD-Upscaling für alle Input-Arrays
        """
        if len(array.shape) == 2:
            # 2D Array (heightmap, temp_map, etc.)
            return self._interpolate_2d(array, target_size)
        elif len(array.shape) == 3 and array.shape[2] == 2:
            # 3D Array mit 2 Kanälen (slopemap)
            result = np.zeros((target_size, target_size, 2), dtype=array.dtype)
            result[:, :, 0] = self._interpolate_2d(array[:, :, 0], target_size)
            result[:, :, 1] = self._interpolate_2d(array[:, :, 1], target_size)
            return result
        else:
            raise ValueError(f"Unsupported array shape for interpolation: {array.shape}")

    def _interpolate_2d(self, array, target_size):
        """
        Funktionsweise: Bilineare Interpolation für 2D-Arrays
        Aufgabe: Smooth Upscaling ohne Artefakte
        """
        old_size = array.shape[0]
        if old_size == target_size:
            return array.copy()

        scale_factor = (old_size - 1) / (target_size - 1)
        interpolated = np.zeros((target_size, target_size), dtype=array.dtype)

        for new_y in range(target_size):
            for new_x in range(target_size):
                old_x = new_x * scale_factor
                old_y = new_y * scale_factor

                x0, y0 = int(old_x), int(old_y)
                x1, y1 = min(x0 + 1, old_size - 1), min(y0 + 1, old_size - 1)

                fx, fy = old_x - x0, old_y - y0

                # Bilineare Interpolation
                h00, h10 = array[y0, x0], array[y0, x1]
                h01, h11 = array[y1, x0], array[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                interpolated[new_y, new_x] = h0 * (1 - fy) + h1 * fy

        return interpolated

    # ===== LEGACY-KOMPATIBILITÄT =====
    # Alle alten Methoden bleiben für Rückwärts-Kompatibilität erhalten

    def classify_biomes(self, heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map, biome_wetness_factor,
                        biome_temp_factor, sea_level, bank_width, edge_softness, alpine_level, snow_level, cliff_slope,
                        map_seed):
        """
        Funktionsweise: Legacy-Methode für direkte Biome-Klassifikation (KOMPATIBILITÄT)
        Aufgabe: Erhält bestehende API für Rückwärts-Kompatibilität
        """
        # Konvertiert alte API zur neuen API
        dependencies = {
            'heightmap': heightmap,
            'slopemap': slopemap,
            'temp_map': temp_map,
            'soil_moist_map': soil_moist_map,
            'water_biomes_map': water_biomes_map
        }
        parameters = {
            'biome_wetness_factor': biome_wetness_factor,
            'biome_temp_factor': biome_temp_factor,
            'sea_level': sea_level,
            'bank_width': bank_width,
            'edge_softness': edge_softness,
            'alpine_level': alpine_level,
            'snow_level': snow_level,
            'cliff_slope': cliff_slope
        }

        # Seed aktualisieren falls nötig
        if map_seed != self.map_seed:
            self.update_seed(map_seed)

        biome_data = self._execute_generation("LOD64", dependencies, parameters)

        # Legacy-Format zurückgeben (Tuple)
        return biome_data.biome_map, biome_data.super_biome_mask

    def apply_supersampling(self, biome_map, super_biome_probabilities=None):
        """
        Funktionsweise: Legacy-Methode für Supersampling
        Aufgabe: Erhält bestehende API für Supersampling
        """
        supersampling_manager = SupersamplingManager(self.map_seed)
        biome_map_super = supersampling_manager.apply_rotational_supersampling(biome_map, super_biome_probabilities)
        return biome_map_super

    def integrate_super_biomes(self, base_biome_map, super_biome_probabilities):
        """
        Funktionsweise: Legacy-Methode für Super-Biome-Integration
        Aufgabe: Erhält bestehende API für Super-Biome-Integration
        """
        return self._integrate_super_biomes(base_biome_map, super_biome_probabilities)

    def generate_complete_biomes(self, heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map,
                                 biome_wetness_factor, biome_temp_factor, sea_level, bank_width, edge_softness,
                                 alpine_level, snow_level, cliff_slope, map_seed):
        """
        Funktionsweise: Legacy-Methode für komplette Biome-Generierung
        Aufgabe: One-Stop Funktion für alle Biom-Outputs (KOMPATIBILITÄT)
        """
        # Konvertiert alte API zur neuen API
        dependencies = {
            'heightmap': heightmap,
            'slopemap': slopemap,
            'temp_map': temp_map,
            'soil_moist_map': soil_moist_map,
            'water_biomes_map': water_biomes_map
        }
        parameters = {
            'biome_wetness_factor': biome_wetness_factor,
            'biome_temp_factor': biome_temp_factor,
            'sea_level': sea_level,
            'bank_width': bank_width,
            'edge_softness': edge_softness,
            'alpine_level': alpine_level,
            'snow_level': snow_level,
            'cliff_slope': cliff_slope
        }

        # Seed aktualisieren falls nötig
        if map_seed != self.map_seed:
            self.update_seed(map_seed)

        biome_data = self._execute_generation("LOD256", dependencies, parameters)  # Höheres LOD für Legacy

        # Legacy-Format zurückgeben (Tuple)
        return biome_data.biome_map, biome_data.biome_map_super, biome_data.super_biome_mask

    def get_biome_statistics(self, biome_data):
        """
        Funktionsweise: Legacy-Methode für Biome-Statistiken
        Aufgabe: Analyse-Funktionen für Biom-System-Debugging
        """
        if isinstance(biome_data, BiomeData):
            biome_map = biome_data.biome_map
            biome_map_super = biome_data.biome_map_super
            super_biome_mask = biome_data.super_biome_mask
        else:
            biome_map, biome_map_super, super_biome_mask = biome_data

        # Biom-Verteilung
        unique_biomes, biome_counts = np.unique(biome_map, return_counts=True)
        biome_distribution = dict(zip(unique_biomes, biome_counts))

        # Super-Biome-Verteilung
        unique_super, super_counts = np.unique(super_biome_mask[super_biome_mask > 0], return_counts=True)
        super_distribution = dict(zip(unique_super, super_counts)) if len(unique_super) > 0 else {}

        # Supersampling-Statistiken
        supersampling_distribution = {}
        if biome_map_super is not None:
            super_unique, super_super_counts = np.unique(biome_map_super, return_counts=True)
            supersampling_distribution = dict(zip(super_unique, super_super_counts))

        stats = {
            'base_biomes': {
                'total_pixels': int(np.prod(biome_map.shape)),
                'unique_biomes': len(unique_biomes),
                'distribution': {int(k): int(v) for k, v in biome_distribution.items()},
                'diversity_index': self._calculate_diversity_index(biome_counts)
            },
            'super_biomes': {
                'override_pixels': int(np.sum(super_biome_mask > 0)),
                'unique_super_biomes': len(unique_super),
                'distribution': {int(k): int(v) for k, v in super_distribution.items()}
            },
            'supersampling': {
                'enabled': biome_map_super is not None,
                'super_resolution': biome_map_super.shape if biome_map_super is not None else None,
                'unique_biomes_super': len(supersampling_distribution),
                'distribution': {int(k): int(v) for k, v in supersampling_distribution.items()}
            }
        }

        return stats

    def _calculate_diversity_index(self, counts):
        """
        Funktionsweise: Berechnet Shannon-Diversity-Index für Biom-Verteilung
        Aufgabe: Diversitäts-Metrik für Biom-Verteilungs-Analyse
        """
        total = np.sum(counts)
        if total == 0:
            return 0.0

        proportions = counts / total
        # Shannon-Index: -sum(p * log(p))
        diversity = -np.sum(proportions * np.log(proportions + 1e-10))  # +epsilon für log(0) Vermeidung

        return float(diversity)

    def get_biome_name_mapping(self):
        """
        Funktionsweise: Gibt Mapping zwischen Biom-Indices und Namen zurück
        Aufgabe: Hilfsfunktion für Biom-Name-Auflösung
        Returns: dict - Index zu Name Mapping
        """
        base_biomes = [
            'ice_cap', 'tundra', 'taiga', 'grassland', 'temperate_forest', 'mediterranean',
            'desert', 'semi_arid', 'tropical_rainforest', 'tropical_seasonal', 'savanna',
            'montane_forest', 'swamp', 'coastal_dunes', 'badlands'
        ]

        super_biomes = [
            'ocean', 'lake', 'grand_river', 'river', 'creek', 'cliff',
            'beach', 'lake_edge', 'river_bank', 'snow_level', 'alpine_level'
        ]

        mapping = {}

        # Base-Biomes (0-14)
        for i, name in enumerate(base_biomes):
            mapping[i] = name

        # Super-Biomes (15-25)
        for i, name in enumerate(super_biomes):
            mapping[15 + i] = name

        return mapping


# =============================================================================
# Legacy-Klassen für Rückwärts-Kompatibilität
# Alle ursprünglichen Klassen bleiben funktional, verwenden aber die neue
# BaseGenerator-Integration intern
# =============================================================================

class ProximityBiomeCalculator:
    """
    Funktionsweise: Legacy-Klasse für Proximity-Biome-Berechnung (KOMPATIBILITÄT)
    Aufgabe: Berechnet Proximity-basierte Super-Biomes für Legacy-Code
    """

    def __init__(self, bank_width=10.0, edge_softness=1.0):
        """
        Funktionsweise: Initialisiert Proximity-Calculator mit Distanz- und Softness-Parametern
        Aufgabe: Setup der Proximity-Biome-Berechnung
        """
        self.bank_width = bank_width
        self.edge_softness = edge_softness

    def calculate_distance_fields(self, water_biomes_map):
        """
        Funktionsweise: Berechnet Distanzfelder zu verschiedenen Gewässertypen
        Aufgabe: Erstellt Distanz-Maps für alle Wasser-Kategorien
        Parameter: water_biomes_map - Wasser-Klassifikation
        Returns: dict - Distanzfelder für jeden Gewässertyp
        """
        distance_fields = {}

        # Lake
        lake_mask = water_biomes_map == 4
        distance_fields['lake'] = distance_transform_edt(~lake_mask) if np.any(lake_mask) else None

        # Rivers (alle Fließgewässer)
        river_mask = (water_biomes_map >= 1) & (water_biomes_map <= 3)
        distance_fields['river'] = distance_transform_edt(~river_mask) if np.any(river_mask) else None

        return distance_fields

    def apply_gaussian_proximity(self, distance_fields, water_biomes_map):
        """
        Funktionsweise: Wendet Gaussian-Proximity auf Distanzfelder an
        Aufgabe: Erstellt weiche Proximity-Übergänge mit edge_softness
        Parameter: distance_fields, water_biomes_map - Distanzfelder und Wasser-Klassifikation
        Returns: dict - Proximity-Wahrscheinlichkeiten
        """
        proximity_maps = {}
        height, width = water_biomes_map.shape

        # Lake Edge
        if distance_fields['lake'] is not None:
            lake_distance = distance_fields['lake']
            lake_edge_condition = ((lake_distance <= self.bank_width) & (water_biomes_map != 4))
            lake_edge_prob = np.maximum(0, 1 - np.power(lake_distance / self.bank_width, self.edge_softness))
            lake_edge_prob[~lake_edge_condition] = 0
            proximity_maps['lake_edge'] = lake_edge_prob
        else:
            proximity_maps['lake_edge'] = np.zeros((height, width), dtype=np.float32)

        # River Bank
        if distance_fields['river'] is not None:
            river_distance = distance_fields['river']
            river_bank_condition = ((river_distance <= self.bank_width) & (water_biomes_map == 0))
            river_bank_prob = np.maximum(0, 1 - np.power(river_distance / self.bank_width, self.edge_softness))
            river_bank_prob[~river_bank_condition] = 0
            proximity_maps['river_bank'] = river_bank_prob
        else:
            proximity_maps['river_bank'] = np.zeros((height, width), dtype=np.float32)

        return proximity_maps

    def blend_with_base_biomes(self, base_biomes, proximity_maps):
        """
        Funktionsweise: Legacy-Methode für Biome-Blending
        Aufgabe: Kompatibilität mit altem Code
        """
        # Delegiert an neue BiomeClassificationSystem-Implementation
        biome_system = BiomeClassificationSystem()
        return biome_system._integrate_super_biomes(base_biomes, proximity_maps)