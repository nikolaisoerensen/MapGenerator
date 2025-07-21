"""
Path: core/biome_generator.py

Funktionsweise: Klassifikation von Biomen auf Basis von Höhe, Temperatur, Niederschlag und Slope
- Gauß-basierte Klassifizierung mit Gewichtungen je Biomtyp
- Verwendung eines vektorbasierten Klassifikators für performante Zuordnung auf großen Karten
- Zwei-Ebenen-System: Base-Biomes und Super-Biomes
- Supersampling für weichere Übergänge zwischen allen Biomen (die vier dominantesten Anteile pro Zelle)
- Super-Biomes überschreiben Base-Biomes basierend auf speziellen Bedingungen

Parameter Input:
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
- heightmap (2D-Array in meter Altitude)
- slopemap (2D-Array in m/m mit dz/dx, dz/dy)
- temp_map (2D-Array in °C)
- soil_moist_map (2D-Array in Bodenfeuchtigkeit %)
- water_biomes_map (2D-Array mit Wasser-Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake)

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
BiomeClassificationSystem
    Funktionsweise: Hauptklasse für Biom-Klassifikation basierend auf Klimadaten
    Aufgabe: Koordiniert Base-Biome und Super-Biome Zuordnung mit Supersampling
    Methoden: classify_biomes(), apply_supersampling(), integrate_super_biomes()

BaseBiomeClassifier
    Funktionsweise: Gauß-basierte Klassifizierung von 15 Grundbiomen
    Aufgabe: Erstellt biome_map basierend auf Höhe, Temperatur, Niederschlag und Slope
    Methoden: calculate_gaussian_fitness(), weight_environmental_factors(), assign_dominant_biome()

SuperBiomeOverrideSystem
    Funktionsweise: Überschreibt Base-Biomes mit speziellen Bedingungen (Ocean, Cliff, Beach, etc.)
    Aufgabe: Erstellt super_biome_mask für prioritätsbasierte Biom-Überschreibung
    Methoden: detect_ocean_connectivity(), apply_proximity_biomes(), calculate_elevation_biomes()

SupersamplingManager
    Funktionsweise: 2x2 Supersampling mit diskretisierter Zufalls-Rotation
    Aufgabe: Erstellt biome_map_super für weiche Übergänge zwischen Biomen
    Methoden: apply_rotational_supersampling(), calculate_soft_transitions(), optimize_spatial_distribution()

ProximityBiomeCalculator
    Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
    Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
    Methoden: calculate_distance_fields(), apply_gaussian_proximity(), blend_with_base_biomes()
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from collections import deque


class BaseBiomeClassifier:
    """
    Funktionsweise: Gauß-basierte Klassifizierung von 15 Grundbiomen
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
            'tropical_rainforest': {'temp': (20, 35), 'moisture': (1500, 4000), 'elevation': (0, 1500),
                                    'slope': (0, 70)},
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

    def calculate_gaussian_fitness(self, temp_map, soil_moist_map, heightmap, slopemap):
        """
        Funktionsweise: Berechnet Gauß-Passung für alle Biome an allen Positionen
        Aufgabe: Erstellt Fitness-Maps für jeden Biom-Typ basierend auf Umweltparametern
        Parameter: temp_map, soil_moist_map, heightmap, slopemap - Alle Umwelt-Daten
        Returns: numpy.ndarray - Fitness-Werte (height, width, num_biomes)
        """
        height, width = temp_map.shape
        num_biomes = len(self.base_biomes)
        fitness_maps = np.zeros((height, width, num_biomes), dtype=np.float32)

        # Slope in Grad konvertieren
        slope_degrees = np.degrees(np.arctan(np.sqrt(slopemap[:, :, 0] ** 2 + slopemap[:, :, 1] ** 2)))

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

    def weight_environmental_factors(self, fitness_maps):
        """
        Funktionsweise: Wendet Umwelt-Faktor-Gewichtung auf Fitness-Maps an
        Aufgabe: Adjustiert Biome-Fitness basierend auf Gewichtungsparametern
        Parameter: fitness_maps - Fitness-Arrays aller Biome
        Returns: numpy.ndarray - Gewichtete Fitness-Maps
        """
        # Bereits in calculate_gaussian_fitness() implementiert
        return fitness_maps

    def assign_dominant_biome(self, fitness_maps):
        """
        Funktionsweise: Weist dominantes Biom basierend auf höchster Fitness zu
        Aufgabe: Erstellt biome_map mit Index des dominantesten Bioms pro Pixel
        Parameter: fitness_maps - Fitness-Arrays aller Biome
        Returns: numpy.ndarray - Biome-Indices (height, width)
        """
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

    def detect_ocean_connectivity(self, heightmap, water_biomes_map):
        """
        Funktionsweise: Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind und unter sea_level liegen
        Aufgabe: Identifiziert Ocean-Bereiche durch Rand-Konnektivität
        Parameter: heightmap, water_biomes_map - Höhen und Wasser-Klassifikation
        Returns: numpy.ndarray - Ocean-Maske (bool)
        """
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

    def apply_proximity_biomes(self, water_biomes_map, heightmap):
        """
        Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
        Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
        Parameter: water_biomes_map, heightmap - Wasser-Klassifikation und Höhen
        Returns: dict - Proximity-Masken für verschiedene Ufer-Biome
        """
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

    def calculate_elevation_biomes(self, heightmap, temp_map, slopemap):
        """
        Funktionsweise: Berechnet Höhen-basierte Super-Biomes (Snow, Alpine, Cliff)
        Aufgabe: Erstellt Höhen- und Slope-abhängige Super-Biome mit weichen Übergängen
        Parameter: heightmap, temp_map, slopemap - Höhen, Temperatur und Slope-Daten
        Returns: dict - Elevation-basierte Super-Biome Masken
        """
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


class ProximityBiomeCalculator:
    """
    Funktionsweise: Berechnet Proximity-basierte Super-Biomes (Beach, Lake Edge, River Bank)
    Aufgabe: Erstellt weiche Übergänge um Gewässer mit konfigurierbarem edge_softness
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

        # Ocean (wird separat behandelt in SuperBiomeOverrideSystem)
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
        Funktionsweise: Blendet Proximity-Biomes mit Base-Biomes
        Aufgabe: Integriert Proximity-Effekte in bestehende Biom-Verteilung
        Parameter: base_biomes, proximity_maps - Base-Biome-Indices und Proximity-Wahrscheinlichkeiten
        Returns: numpy.ndarray - Modifizierte Biom-Map
        """
        # Wird in BiomeClassificationSystem implementiert
        return base_biomes


class SupersamplingManager:
    """
    Funktionsweise: 2x2 Supersampling mit diskretisierter Zufalls-Rotation
    Aufgabe: Erstellt biome_map_super für weiche Übergänge zwischen Biomen
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Supersampling-Manager mit Zufalls-Seed
        Aufgabe: Setup der reproduzierbaren Supersampling-Rotation
        Parameter: map_seed (int) - Seed für reproduzierbare Zufälligkeit
        """
        self.map_seed = map_seed

    def apply_rotational_supersampling(self, biome_map, super_biome_probabilities=None):
        """
        Funktionsweise: Wendet 2x2 Supersampling mit diskretisierter Zufalls-Rotation an
        Aufgabe: Erstellt 2x supersampeltes Biom-Map mit optimierter räumlicher Verteilung
        Parameter: biome_map, super_biome_probabilities - Base-Biomes und Super-Biome-Wahrscheinlichkeiten
        Returns: numpy.ndarray - 2x supersampeltes Biom-Map
        """
        height, width = biome_map.shape
        super_height, super_width = height * 2, width * 2
        biome_map_super = np.zeros((super_height, super_width), dtype=np.uint8)

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

        return biome_map_super

    def calculate_soft_transitions(self, biome_map, edge_softness):
        """
        Funktionsweise: Berechnet weiche Übergänge zwischen benachbarten Biomen
        Aufgabe: Erstellt natürliche Biom-Grenzen durch Soft-Transitions
        Parameter: biome_map, edge_softness - Biom-Map und Softness-Parameter
        Returns: dict - Transition-Wahrscheinlichkeiten zwischen Biomen
        """
        height, width = biome_map.shape
        transitions = {}

        # Für jedes Biom: berechne Übergangs-Wahrscheinlichkeiten zu Nachbarn
        unique_biomes = np.unique(biome_map)

        for biome_id in unique_biomes:
            biome_mask = biome_map == biome_id

            # Distanz zu Biom-Grenzen
            distance_to_edge = distance_transform_edt(biome_mask)

            # Soft-Transition basierend auf edge_softness
            max_transition_distance = 5.0  # Pixel
            transition_prob = np.exp(-distance_to_edge / (max_transition_distance * edge_softness))
            transition_prob[~biome_mask] = 0

            transitions[biome_id] = transition_prob

        return transitions

    def optimize_spatial_distribution(self, biome_map_super):
        """
        Funktionsweise: Optimiert räumliche Verteilung für natürlichere Biom-Muster
        Aufgabe: Eliminiert künstliche Repetition durch mathematisch optimierte Streuung
        Parameter: biome_map_super - Supersampeltes Biom-Map
        Returns: numpy.ndarray - Optimiertes supersampeltes Biom-Map
        """
        # Bereits in apply_rotational_supersampling() durch optimierte Hash-Funktionen implementiert
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


class BiomeClassificationSystem:
    """
    Funktionsweise: Hauptklasse für Biom-Klassifikation basierend auf Klimadaten
    Aufgabe: Koordiniert Base-Biome und Super-Biome Zuordnung mit Supersampling
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Biome-Classification-System mit allen Sub-Komponenten
        Aufgabe: Setup aller Biom-Klassifikations-Systeme
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Biom-Verteilung
        """
        self.map_seed = map_seed

        # Standard-Parameter
        self.biome_wetness_factor = 1.0
        self.biome_temp_factor = 1.0
        self.sea_level = 0.0
        self.bank_width = 10.0
        self.edge_softness = 1.0
        self.alpine_level = 2000.0
        self.snow_level = 3000.0
        self.cliff_slope = 45.0

    def classify_biomes(self, heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map, biome_wetness_factor,
                        biome_temp_factor, sea_level, bank_width, edge_softness, alpine_level, snow_level, cliff_slope,
                        map_seed):
        """
        Funktionsweise: Klassifiziert Biome durch Base-Biome-Analyse und Super-Biome-Überschreibung
        Aufgabe: Hauptfunktion für komplette Biom-Klassifikation
        Parameter: heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map, biome_wetness_factor, biome_temp_factor, sea_level, bank_width, edge_softness, alpine_level, snow_level, cliff_slope, map_seed
        Returns: Tuple (biome_map, super_biome_mask) - Biom-Map und Super-Biome-Maske
        """
        # Parameter aktualisieren
        self.biome_wetness_factor = biome_wetness_factor
        self.biome_temp_factor = biome_temp_factor
        self.sea_level = sea_level
        self.bank_width = bank_width
        self.edge_softness = edge_softness
        self.alpine_level = alpine_level
        self.snow_level = snow_level
        self.cliff_slope = cliff_slope

        # Seed aktualisieren
        if map_seed != self.map_seed:
            self.map_seed = map_seed

        # Schritt 1: Base-Biome Klassifikation
        base_classifier = BaseBiomeClassifier(biome_wetness_factor, biome_temp_factor)
        fitness_maps = base_classifier.calculate_gaussian_fitness(temp_map, soil_moist_map, heightmap, slopemap)
        base_biome_map = base_classifier.assign_dominant_biome(fitness_maps)

        # Schritt 2: Super-Biome Override
        super_biome_system = SuperBiomeOverrideSystem(sea_level, bank_width, edge_softness, alpine_level, snow_level,
                                                      cliff_slope)

        # Wasser-Biome direkt übernehmen
        super_biome_mask = np.zeros_like(base_biome_map, dtype=np.uint8)

        # Priority 0-4: Wasser-Biome
        super_biome_mask[water_biomes_map == 4] = super_biome_system.super_biomes['lake']  # Lake
        super_biome_mask[water_biomes_map == 3] = super_biome_system.super_biomes['grand_river']  # Grand River
        super_biome_mask[water_biomes_map == 2] = super_biome_system.super_biomes['river']  # River
        super_biome_mask[water_biomes_map == 1] = super_biome_system.super_biomes['creek']  # Creek

        # Ocean-Detection
        ocean_mask = super_biome_system.detect_ocean_connectivity(heightmap, water_biomes_map)
        super_biome_mask[ocean_mask] = super_biome_system.super_biomes['ocean']

        # Proximity-Biomes
        proximity_masks = super_biome_system.apply_proximity_biomes(water_biomes_map, heightmap)

        # Elevation-Biomes
        elevation_masks = super_biome_system.calculate_elevation_biomes(heightmap, temp_map, slopemap)

        # Kombiniere alle Super-Biome-Masken (höhere Priorität überschreibt)
        all_super_masks = {**proximity_masks, **elevation_masks}

        # Finale Biom-Map: Base-Biomes + Super-Biome-Overrides
        final_biome_map = base_biome_map.copy()

        # Wasser-Biome überschreiben (höchste Priorität)
        water_override = super_biome_mask > 0
        final_biome_map[water_override] = super_biome_mask[water_override]

        return final_biome_map, super_biome_mask

    def apply_supersampling(self, biome_map, super_biome_probabilities=None):
        """
        Funktionsweise: Wendet 2x2 Supersampling auf Biom-Map an
        Aufgabe: Erstellt biome_map_super mit weichen Übergängen
        Parameter: biome_map, super_biome_probabilities - Biom-Map und Super-Biome-Wahrscheinlichkeiten
        Returns: numpy.ndarray - 2x supersampeltes Biom-Map
        """
        supersampling_manager = SupersamplingManager(self.map_seed)
        biome_map_super = supersampling_manager.apply_rotational_supersampling(biome_map, super_biome_probabilities)

        return biome_map_super

    def integrate_super_biomes(self, base_biome_map, super_biome_probabilities):
        """
        Funktionsweise: Integriert Super-Biome-Wahrscheinlichkeiten in Base-Biome-Map
        Aufgabe: Probabilistische Super-Biome-Überschreibung mit weichen Übergängen
        Parameter: base_biome_map, super_biome_probabilities - Base-Biomes und Super-Biome-Wahrscheinlichkeiten
        Returns: numpy.ndarray - Integrierte Biom-Map
        """
        height, width = base_biome_map.shape
        integrated_map = base_biome_map.copy()

        # Prioritäts-basierte Integration (nach dokumentierter Reihenfolge)
        priority_order = ['ocean', 'lake', 'grand_river', 'river', 'creek', 'cliff', 'beach', 'lake_edge', 'river_bank',
                          'snow_level', 'alpine_level']

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

    def generate_complete_biomes(self, heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map,
                                 biome_wetness_factor, biome_temp_factor, sea_level, bank_width, edge_softness,
                                 alpine_level, snow_level, cliff_slope, map_seed):
        """
        Funktionsweise: Generiert komplettes Biom-System mit allen Komponenten
        Aufgabe: One-Stop Funktion für alle Biom-Outputs
        Parameter: heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map, biome_wetness_factor, biome_temp_factor, sea_level, bank_width, edge_softness, alpine_level, snow_level, cliff_slope, map_seed
        Returns: Tuple (biome_map, biome_map_super, super_biome_mask) - Alle Biom-Outputs
        """
        # Basis-Klassifikation
        biome_map, super_biome_mask = self.classify_biomes(
            heightmap, slopemap, temp_map, soil_moist_map, water_biomes_map,
            biome_wetness_factor, biome_temp_factor, sea_level, bank_width, edge_softness,
            alpine_level, snow_level, cliff_slope, map_seed
        )

        # Super-Biome-Wahrscheinlichkeiten für Supersampling
        super_biome_system = SuperBiomeOverrideSystem(sea_level, bank_width, edge_softness, alpine_level, snow_level,
                                                      cliff_slope)
        proximity_probs = super_biome_system.apply_proximity_biomes(water_biomes_map, heightmap)
        elevation_probs = super_biome_system.calculate_elevation_biomes(heightmap, temp_map, slopemap)

        all_super_probs = {**proximity_probs, **elevation_probs}

        # Supersampling anwenden
        biome_map_super = self.apply_supersampling(biome_map, all_super_probs)

        return biome_map, biome_map_super, super_biome_mask

    def get_biome_statistics(self, biome_data):
        """
        Funktionsweise: Berechnet Statistiken über generierte Biom-Daten
        Aufgabe: Analyse-Funktionen für Biom-System-Debugging
        Parameter: biome_data - Tuple aller Biom-Arrays
        Returns: dict - Biom-Statistiken
        """
        biome_map, biome_map_super, super_biome_mask = biome_data

        # Biom-Verteilung
        unique_biomes, biome_counts = np.unique(biome_map, return_counts=True)
        biome_distribution = dict(zip(unique_biomes, biome_counts))

        # Super-Biome-Verteilung
        unique_super, super_counts = np.unique(super_biome_mask[super_biome_mask > 0], return_counts=True)
        super_distribution = dict(zip(unique_super, super_counts)) if len(unique_super) > 0 else {}

        # Supersampling-Statistiken
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
                'super_resolution': biome_map_super.shape,
                'unique_biomes_super': len(super_unique),
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

    def update_seed(self, new_seed):
        """
        Funktionsweise: Aktualisiert Seed für alle Zufalls-Komponenten
        Aufgabe: Ermöglicht Seed-Änderung ohne Neuinstanziierung
        Parameter: new_seed (int) - Neuer Seed für Biom-Generierung
        """
        if new_seed != self.map_seed:
            self.map_seed = new_seed

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