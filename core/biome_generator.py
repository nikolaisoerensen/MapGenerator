"""
Path: core/biome_generator.py
Date Changed: 24.08.2025

Funktionsweise: Komplexe Ökosystem-Klassifikation mit Multi-Generator-Integration
- BiomeClassificationSystem koordiniert geologische Simulation mit numerischem LOD-System
- 15 Base-Biomes nach Whittaker-Diagramm + 11 Super-Biomes mit Priority-Override-System
- 2x2-Supersampling mit diskretisierter Zufalls-Rotation für weiche Übergänge
- 3-stufiges Fallback-System: GPU-Shader → CPU-Fallback → Simple-Fallback

Parameter Input (aus value_default.py BIOME):
- biome_temp_factor (Gewichtung der Temperaturwerte, 0.0-3.0)
- biome_wetness_factor (Gewichtung der Bodenfeuchtigkeit, 0.0-3.0)
- elevation_factor (Gewichtung der Höhenwerte, 0.0-3.0)
- soil_moisture_factor (Gewichtung der Bodenfeuchtigkeit, 0.0-3.0)
- sea_level (Meeresspiegel-Höhe in Metern)
- alpine_level (Basis-Höhe für Alpine-Zone in Metern)
- snow_level (Basis-Höhe für Schneegrenze in Metern)
- cliff_slope (Grenzwert für Klippen-Klassifikation in Grad)
- edge_softness (Globaler Weichheits-Faktor für alle Super-Biome-Übergänge, 0.1-2.0)
- bank_width (Radius für Ufer-Biome in Pixeln)
- supersampling_quality (Supersampling-Level, 0.1-2.0)
- biome_seed (Reproduzierbare Zufallsvariation)

Dependencies (über DataLODManager):
- heightmap (von terrain_generator)
- temp_map (von weather_generator)
- precip_map (von weather_generator)
- soil_moist_map (von water_generator)
- water_biomes_map (von water_generator)

Output:
- BiomeData-Objekt mit biome_map, biome_map_super, super_biome_mask, validity_state und LOD-Metadaten
- DataLODManager-Storage für nachfolgende Generatoren (settlement)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from collections import deque
import logging


class BiomeData:
    """
    Container für alle Biome-Daten mit Validity-System und Cache-Management
    """
    def __init__(self):
        self.biome_map = None                    # 2D numpy.uint8 array, Index der dominantesten Biom-Klasse
        self.biome_map_super = None             # 2D numpy.uint8 array, 2x supersampled für weiche Übergänge
        self.super_biome_mask = None            # 2D numpy.bool array, Override-Bereiche-Maske
        self.climate_classification = None       # 2D numpy.uint8 array, Whittaker-Klimazone-Zuordnung
        self.biome_statistics = None            # Dict mit Verteilungs-Prozenten und Diversity-Metriken
        self.lod_level = 1                      # Numerisches LOD-Level
        self.actual_size = 32                   # Tatsächliche Kartengröße
        self.validity_state = {}                # Validity-flags pro LOD-Level und Output-Type
        self.parameter_hash = None              # Parameter-Hash für Cache-Invalidation
        self.performance_stats = {}             # Performance-Metriken


class BiomeClassificationSystem:
    """
    Hauptklasse für Multi-Factor-Biome-Classification mit vollständiger Manager-Integration
    """

    def __init__(self, shader_manager=None, data_lod_manager=None):
        """
        Initialisiert Biome-Classification-System mit Manager-Integration
        """
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager
        self.logger = logging.getLogger(__name__)

        # Standard-Parameter
        self.biome_temp_factor = 1.0
        self.biome_wetness_factor = 1.0
        self.elevation_factor = 1.0
        self.soil_moisture_factor = 1.0
        self.sea_level = 10.0
        self.alpine_level = 1500.0
        self.snow_level = 2000.0
        self.cliff_slope = 60.0
        self.edge_softness = 1.0
        self.bank_width = 3.0
        self.supersampling_quality = 1.0
        self.biome_seed = 42

        # Base-Biome Definitionen nach Whittaker-Diagramm
        self.base_biomes = self._initialize_base_biomes()

        # Super-Biome System
        self.super_biomes = self._initialize_super_biomes()

        # Sub-Komponenten
        self.base_biome_classifier = None
        self.super_biome_override_system = None
        self.supersampling_manager = None

    def _initialize_base_biomes(self):
        """
        Initialisiert 15 Base-Biome nach wissenschaftlichen Standards
        """
        return {
            'ice_cap': {'temp': (-40, -5), 'precip': (0, 300), 'elevation': (0, 8000), 'moisture': (0, 200)},
            'tundra': {'temp': (-15, 5), 'precip': (100, 600), 'elevation': (0, 2000), 'moisture': (100, 400)},
            'taiga': {'temp': (-10, 15), 'precip': (300, 1200), 'elevation': (50, 2500), 'moisture': (300, 800)},
            'grassland': {'temp': (0, 25), 'precip': (200, 800), 'elevation': (10, 1500), 'moisture': (200, 600)},
            'temperate_forest': {'temp': (5, 25), 'precip': (600, 2000), 'elevation': (0, 2000), 'moisture': (400, 1000)},
            'mediterranean': {'temp': (8, 30), 'precip': (300, 900), 'elevation': (0, 1200), 'moisture': (200, 600)},
            'desert': {'temp': (10, 50), 'precip': (0, 250), 'elevation': (0, 2000), 'moisture': (0, 100)},
            'semi_arid': {'temp': (5, 35), 'precip': (200, 600), 'elevation': (0, 1800), 'moisture': (100, 400)},
            'tropical_rainforest': {'temp': (20, 35), 'precip': (1500, 4000), 'elevation': (0, 1500), 'moisture': (800, 1500)},
            'tropical_seasonal': {'temp': (18, 35), 'precip': (800, 2000), 'elevation': (0, 1200), 'moisture': (400, 1000)},
            'savanna': {'temp': (15, 35), 'precip': (400, 1200), 'elevation': (0, 1800), 'moisture': (200, 600)},
            'montane_forest': {'temp': (0, 20), 'precip': (800, 3000), 'elevation': (800, 3500), 'moisture': (600, 1200)},
            'swamp': {'temp': (5, 35), 'precip': (800, 3000), 'elevation': (0, 200), 'moisture': (800, 1500)},
            'coastal_dunes': {'temp': (5, 35), 'precip': (300, 1500), 'elevation': (0, 100), 'moisture': (200, 800)},
            'badlands': {'temp': (-5, 45), 'precip': (0, 400), 'elevation': (200, 2500), 'moisture': (0, 200)}
        }

    def _initialize_super_biomes(self):
        """
        Initialisiert 11 Super-Biomes mit Priority-Hierarchie
        """
        return {
            'ocean': {'priority': 0, 'condition': 'flood_fill + sea_level'},
            'lake': {'priority': 1, 'condition': 'water_biomes_map == 4'},
            'grand_river': {'priority': 2, 'condition': 'water_biomes_map == 3'},
            'river': {'priority': 3, 'condition': 'water_biomes_map == 2'},
            'creek': {'priority': 4, 'condition': 'water_biomes_map == 1'},
            'cliff': {'priority': 5, 'condition': 'slope > cliff_slope', 'soft_transition': True},
            'beach': {'priority': 6, 'condition': 'ocean_proximity + elevation', 'soft_transition': True},
            'lake_edge': {'priority': 7, 'condition': 'lake_proximity', 'soft_transition': True},
            'river_bank': {'priority': 8, 'condition': 'river_proximity', 'soft_transition': True},
            'snow_level': {'priority': 9, 'condition': 'elevation + temperature', 'soft_transition': True},
            'alpine_level': {'priority': 10, 'condition': 'elevation + temperature', 'soft_transition': True}
        }

    def calculate_biomes(self, multi_input_data, parameters, lod_level):
        """
        Haupteintragspunkt für Biome-Classification mit vollständiger Manager-Integration
        """
        self.logger.info(f"Starting biome classification for LOD {lod_level}")

        # Parameter laden
        self._update_parameters(parameters)

        # Input-Data validieren
        validated_input = self._validate_multi_input(multi_input_data, lod_level)
        if validated_input is None:
            return None

        # Biome-Data-Objekt erstellen
        biome_data = BiomeData()
        biome_data.lod_level = lod_level
        biome_data.actual_size = validated_input['heightmap'].shape[0]
        biome_data.parameter_hash = self._calculate_parameter_hash(parameters)

        try:
            # 3-stufige Fallback-Strategie
            success = False

            # Stufe 1: GPU-Shader (Optimal)
            if self.shader_manager and not success:
                success = self._try_gpu_classification(biome_data, validated_input)
                if success:
                    self.logger.info("GPU-Shader classification successful")

            # Stufe 2: CPU-Fallback (Gut)
            if not success:
                success = self._try_cpu_classification(biome_data, validated_input)
                if success:
                    self.logger.info("CPU-Fallback classification successful")

            # Stufe 3: Simple-Fallback (Minimal)
            if not success:
                success = self._try_simple_classification(biome_data, validated_input)
                if success:
                    self.logger.info("Simple-Fallback classification successful")

            if not success:
                self.logger.error("All classification methods failed")
                return None

            # Statistics berechnen
            biome_data.biome_statistics = self._calculate_biome_statistics(biome_data)

            # Validity-State setzen
            biome_data.validity_state = self._validate_biome_data(biome_data)

            return biome_data

        except Exception as e:
            self.logger.error(f"Critical error in biome classification: {e}")
            return self._create_emergency_fallback(validated_input)

    def _validate_multi_input(self, multi_input_data, lod_level):
        """
        Multi-Generator Cross-Validation mit Consistency-Checks
        """
        required_inputs = ['heightmap', 'temp_map', 'precip_map', 'soil_moist_map', 'water_biomes_map']
        validated_data = {}

        for input_name in required_inputs:
            if input_name not in multi_input_data or multi_input_data[input_name] is None:
                # Graceful-Degradation mit Default-Values
                if input_name == 'soil_moist_map':
                    validated_data[input_name] = self._create_fallback_soil_moisture(
                        multi_input_data.get('heightmap'), multi_input_data.get('temp_map'))
                elif input_name == 'water_biomes_map':
                    validated_data[input_name] = np.zeros_like(multi_input_data.get('heightmap', np.zeros((64, 64))), dtype=np.uint8)
                else:
                    self.logger.error(f"Required input {input_name} missing and no fallback available")
                    return None
            else:
                validated_data[input_name] = multi_input_data[input_name]

        # Physical-Plausibility-Validation
        if not self._validate_input_consistency(validated_data):
            self.logger.warning("Input data consistency validation failed")

        return validated_data

    def _validate_input_consistency(self, data):
        """
        Cross-System-Physical-Plausibility zwischen Temperature-Elevation-Data
        """
        try:
            heightmap = data['heightmap']
            temp_map = data['temp_map']

            # Temperature-Elevation-Konsistenz prüfen
            mean_temp_low = np.mean(temp_map[heightmap < np.percentile(heightmap, 25)])
            mean_temp_high = np.mean(temp_map[heightmap > np.percentile(heightmap, 75)])

            # Höhere Lagen sollten tendenziell kälter sein
            if mean_temp_low <= mean_temp_high:
                return True
            else:
                self.logger.warning("Temperature-elevation relationship appears inverted")
                return False

        except Exception as e:
            self.logger.warning(f"Input consistency validation failed: {e}")
            return False

    def _try_gpu_classification(self, biome_data, input_data):
        """
        GPU-Shader-basierte parallele Multi-Factor-Classification
        """
        try:
            if not self.shader_manager:
                return False

            # GPU-Request für komplette Biome-Classification
            gpu_request = {
                'operation_type': 'multi_factor_biome_classification',
                'input_data': {
                    'heightmap': input_data['heightmap'],
                    'temp_map': input_data['temp_map'],
                    'precip_map': input_data['precip_map'],
                    'soil_moist_map': input_data['soil_moist_map'],
                    'water_biomes_map': input_data['water_biomes_map']
                },
                'parameters': {
                    'biome_temp_factor': self.biome_temp_factor,
                    'biome_wetness_factor': self.biome_wetness_factor,
                    'elevation_factor': self.elevation_factor,
                    'soil_moisture_factor': self.soil_moisture_factor,
                    'sea_level': self.sea_level,
                    'alpine_level': self.alpine_level,
                    'snow_level': self.snow_level,
                    'cliff_slope': self.cliff_slope,
                    'edge_softness': self.edge_softness,
                    'bank_width': self.bank_width,
                    'supersampling_quality': self.supersampling_quality,
                    'biome_seed': self.biome_seed
                },
                'lod_level': biome_data.lod_level
            }

            # Shader-Manager-Request
            response = self.shader_manager.request_biome_classification(gpu_request)

            if response.get('success', False):
                # GPU-Output übernehmen
                biome_data.biome_map = response['output_data']['biome_map']
                biome_data.biome_map_super = response['output_data'].get('biome_map_super')
                biome_data.super_biome_mask = response['output_data']['super_biome_mask']
                biome_data.climate_classification = response['output_data'].get('climate_classification')
                biome_data.performance_stats.update(response.get('performance_metrics', {}))

                return True
            else:
                self.logger.warning(f"GPU classification failed: {response.get('error_details', 'Unknown error')}")
                return False

        except Exception as e:
            self.logger.warning(f"GPU classification error: {e}")
            return False

    def _try_cpu_classification(self, biome_data, input_data):
        """
        CPU-Fallback mit optimierten NumPy-Vectorization und Multiprocessing
        """
        try:
            self.logger.info("Starting CPU-based biome classification")

            # Sub-Komponenten initialisieren
            if not self.base_biome_classifier:
                self.base_biome_classifier = BaseBiomeClassifier(
                    self.biome_temp_factor, self.biome_wetness_factor,
                    self.elevation_factor, self.soil_moisture_factor)

            if not self.super_biome_override_system:
                self.super_biome_override_system = SuperBiomeOverrideSystem(
                    self.sea_level, self.bank_width, self.edge_softness,
                    self.alpine_level, self.snow_level, self.cliff_slope)

            # Schritt 1: Base-Biome-Classification
            base_biome_map = self.base_biome_classifier.classify_base_biomes(
                input_data['heightmap'], input_data['temp_map'],
                input_data['precip_map'], input_data['soil_moist_map'])

            # Schritt 2: Super-Biome-Override
            super_biome_mask, super_biome_probabilities = self.super_biome_override_system.apply_super_biome_overrides(
                input_data['heightmap'], input_data['temp_map'],
                input_data['water_biomes_map'], input_data['soil_moist_map'])

            # Schritt 3: Integration
            final_biome_map = self._integrate_biome_layers(base_biome_map, super_biome_mask)

            # Schritt 4: Supersampling (nur bei höheren LODs)
            biome_map_super = None
            if biome_data.actual_size >= 256:
                if not self.supersampling_manager:
                    self.supersampling_manager = SupersamplingManager(self.biome_seed, self.supersampling_quality)
                biome_map_super = self.supersampling_manager.apply_supersampling(
                    final_biome_map, super_biome_probabilities)

            # Climate-Classification
            climate_classification = self._create_climate_classification(
                input_data['temp_map'], input_data['precip_map'])

            # Results setzen
            biome_data.biome_map = final_biome_map
            biome_data.biome_map_super = biome_map_super
            biome_data.super_biome_mask = super_biome_mask
            biome_data.climate_classification = climate_classification

            return True

        except Exception as e:
            self.logger.warning(f"CPU classification error: {e}")
            return False

    def _try_simple_classification(self, biome_data, input_data):
        """
        Simple-Fallback mit Height-Temperature-basierter Classification
        """
        try:
            self.logger.info("Starting simple fallback classification")

            height, width = input_data['heightmap'].shape
            biome_map = np.zeros((height, width), dtype=np.uint8)

            # Vereinfachte Height-Temperature-basierte Zuordnung
            heightmap = input_data['heightmap']
            temp_map = input_data['temp_map']

            # Normalisierte Werte
            norm_height = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
            norm_temp = (temp_map - np.min(temp_map)) / (np.max(temp_map) - np.min(temp_map))

            # Einfache Regeln
            for y in range(height):
                for x in range(width):
                    h = norm_height[y, x]
                    t = norm_temp[y, x]

                    if h > 0.8:  # Hohe Berge
                        if t < 0.3:
                            biome_map[y, x] = 0  # ice_cap
                        else:
                            biome_map[y, x] = 11  # montane_forest
                    elif h > 0.6:  # Mittlere Höhen
                        if t < 0.4:
                            biome_map[y, x] = 1  # tundra
                        else:
                            biome_map[y, x] = 2  # taiga
                    elif t < 0.3:  # Kalt
                        biome_map[y, x] = 2  # taiga
                    elif t > 0.7:  # Warm
                        if h < 0.2:
                            biome_map[y, x] = 8  # tropical_rainforest
                        else:
                            biome_map[y, x] = 10  # savanna
                    else:  # Gemäßigt
                        biome_map[y, x] = 4  # temperate_forest

            # Wasser-Biomes direkt übernehmen
            water_mask = input_data['water_biomes_map'] > 0
            biome_map[water_mask] = 15 + input_data['water_biomes_map'][water_mask]  # Super-Biome-Offset

            # Simple-Super-Biome-Mask
            super_biome_mask = np.zeros_like(biome_map, dtype=bool)
            super_biome_mask[water_mask] = True

            biome_data.biome_map = biome_map
            biome_data.biome_map_super = None  # Kein Supersampling bei Simple-Fallback
            biome_data.super_biome_mask = super_biome_mask
            biome_data.climate_classification = self._create_simple_climate_classification(temp_map)

            return True

        except Exception as e:
            self.logger.error(f"Simple classification error: {e}")
            return False

    def _create_emergency_fallback(self, input_data):
        """
        Minimal-Biome-System für Critical-Failures
        """
        height, width = input_data['heightmap'].shape

        biome_data = BiomeData()
        biome_data.biome_map = np.full((height, width), 4, dtype=np.uint8)  # Alles temperate_forest
        biome_data.biome_map_super = None
        biome_data.super_biome_mask = np.zeros((height, width), dtype=bool)
        biome_data.climate_classification = np.full((height, width), 2, dtype=np.uint8)  # Temperate
        biome_data.lod_level = 1
        biome_data.actual_size = min(height, width)
        biome_data.validity_state = {'emergency_fallback': True}
        biome_data.biome_statistics = {'emergency': True}
        biome_data.performance_stats = {'fallback_used': 'emergency'}

        return biome_data

    def _create_fallback_soil_moisture(self, heightmap, temp_map):
        """
        Erstellt Fallback soil_moist_map basierend auf Höhe und Temperatur
        """
        if heightmap is None or temp_map is None:
            return np.full((64, 64), 300.0, dtype=np.float32)

        height, width = heightmap.shape
        soil_moist_map = np.zeros((height, width), dtype=np.float32)

        # Höhen-basierte Feuchtigkeit (niedrigere Lagen = mehr Feuchtigkeit)
        norm_height = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap) + 1e-6)
        elevation_moisture = (1.0 - norm_height) * 400  # 0-400 range

        # Temperatur-basierte Feuchtigkeit (kühlere Bereiche = mehr Feuchtigkeit)
        temp_moisture = np.maximum(0, (20 - temp_map) * 15)

        # Kombiniere beide Faktoren
        soil_moist_map = elevation_moisture + temp_moisture
        soil_moist_map = np.clip(soil_moist_map, 0, 1000)

        return soil_moist_map.astype(np.float32)

    def _integrate_biome_layers(self, base_biome_map, super_biome_mask):
        """
        Integriert Base-Biomes mit Super-Biome-Override
        """
        final_biome_map = base_biome_map.copy()

        # Super-Biomes überschreiben Base-Biomes wo Maske aktiv
        super_mask_active = super_biome_mask > 0
        final_biome_map[super_mask_active] = super_biome_mask[super_mask_active]

        return final_biome_map

    def _create_climate_classification(self, temp_map, precip_map):
        """
        Erstellt Whittaker-Klimazone-Zuordnung
        """
        height, width = temp_map.shape
        climate_map = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                temp = temp_map[y, x]
                precip = precip_map[y, x]

                # Whittaker-Klimazonen
                if temp < -10:
                    climate_map[y, x] = 0  # Arctic
                elif temp < 20:
                    if precip < 200:
                        climate_map[y, x] = 4  # Arid
                    elif precip < 1000:
                        climate_map[y, x] = 1  # Boreal/Temperate
                    else:
                        climate_map[y, x] = 2  # Temperate
                else:
                    if precip < 600:
                        climate_map[y, x] = 4  # Arid
                    else:
                        climate_map[y, x] = 3  # Tropical

        return climate_map

    def _create_simple_climate_classification(self, temp_map):
        """
        Vereinfachte Klimaklassifikation nur basierend auf Temperatur
        """
        height, width = temp_map.shape
        climate_map = np.zeros((height, width), dtype=np.uint8)

        climate_map[temp_map < 0] = 0    # Arctic
        climate_map[(temp_map >= 0) & (temp_map < 20)] = 2    # Temperate
        climate_map[temp_map >= 20] = 3  # Tropical

        return climate_map

    def _calculate_biome_statistics(self, biome_data):
        """
        Berechnet umfassende Biome-Statistiken und Diversity-Metriken
        """
        try:
            stats = {}

            # Base-Biome-Distribution
            unique_biomes, counts = np.unique(biome_data.biome_map, return_counts=True)
            total_pixels = np.prod(biome_data.biome_map.shape)

            base_distribution = {}
            for biome_id, count in zip(unique_biomes, counts):
                if biome_id < 15:  # Base-Biomes
                    percentage = (count / total_pixels) * 100
                    base_distribution[int(biome_id)] = {
                        'count': int(count),
                        'percentage': float(percentage)
                    }

            stats['base_biomes'] = {
                'distribution': base_distribution,
                'diversity_index': self._calculate_shannon_diversity(counts),
                'total_pixels': int(total_pixels),
                'unique_count': len([b for b in unique_biomes if b < 15])
            }

            # Super-Biome-Statistics
            if biome_data.super_biome_mask is not None:
                super_pixels = np.sum(biome_data.super_biome_mask > 0)
                stats['super_biomes'] = {
                    'override_pixels': int(super_pixels),
                    'coverage_percentage': float((super_pixels / total_pixels) * 100)
                }

            # Climate-Zone-Statistics
            if biome_data.climate_classification is not None:
                climate_unique, climate_counts = np.unique(biome_data.climate_classification, return_counts=True)
                climate_dist = {}
                for zone_id, count in zip(climate_unique, climate_counts):
                    climate_dist[int(zone_id)] = float((count / total_pixels) * 100)

                stats['climate_zones'] = {
                    'distribution': climate_dist,
                    'unique_zones': len(climate_unique)
                }

            # Supersampling-Statistics
            if biome_data.biome_map_super is not None:
                stats['supersampling'] = {
                    'enabled': True,
                    'resolution': biome_data.biome_map_super.shape,
                    'quality_factor': 4  # 2x2 supersampling
                }
            else:
                stats['supersampling'] = {'enabled': False}

            return stats

        except Exception as e:
            self.logger.warning(f"Statistics calculation failed: {e}")
            return {'error': 'Statistics calculation failed'}

    def _calculate_shannon_diversity(self, counts):
        """
        Berechnet Shannon-Diversity-Index für Biom-Verteilung
        """
        total = np.sum(counts)
        if total == 0:
            return 0.0

        proportions = counts / total
        # Shannon-Index: -sum(p * log(p))
        diversity = -np.sum(proportions * np.log(proportions + 1e-10))  # +epsilon für log(0) Vermeidung
        return float(diversity)

    def _validate_biome_data(self, biome_data):
        """
        Validiert Biome-Data-Integrity und erstellt Validity-State
        """
        validity_state = {}

        try:
            # Basic Data-Validation
            if biome_data.biome_map is not None:
                validity_state['biome_map_valid'] = True
                validity_state['biome_range_valid'] = np.all((biome_data.biome_map >= 0) & (biome_data.biome_map <= 25))
            else:
                validity_state['biome_map_valid'] = False

            # Super-Biome-Validation
            if biome_data.super_biome_mask is not None:
                validity_state['super_biome_mask_valid'] = True
            else:
                validity_state['super_biome_mask_valid'] = False

            # Supersampling-Validation
            if biome_data.biome_map_super is not None:
                expected_super_size = biome_data.actual_size * 2
                actual_super_shape = biome_data.biome_map_super.shape
                validity_state['supersampling_size_valid'] = (
                    actual_super_shape[0] == expected_super_size and
                    actual_super_shape[1] == expected_super_size
                )
            else:
                validity_state['supersampling_size_valid'] = True  # Optional

            validity_state['overall_valid'] = all([
                validity_state.get('biome_map_valid', False),
                validity_state.get('biome_range_valid', False)
            ])

        except Exception as e:
            self.logger.warning(f"Biome data validation failed: {e}")
            validity_state['validation_error'] = str(e)
            validity_state['overall_valid'] = False

        return validity_state

    def _update_parameters(self, parameters):
        """
        Aktualisiert alle Biome-Parameter aus Parameter-Dictionary
        """
        if not parameters:
            return

        self.biome_temp_factor = parameters.get('biome_temp_factor', self.biome_temp_factor)
        self.biome_wetness_factor = parameters.get('biome_wetness_factor', self.biome_wetness_factor)
        self.elevation_factor = parameters.get('elevation_factor', self.elevation_factor)
        self.soil_moisture_factor = parameters.get('soil_moisture_factor', self.soil_moisture_factor)
        self.sea_level = parameters.get('sea_level', self.sea_level)
        self.alpine_level = parameters.get('alpine_level', self.alpine_level)
        self.snow_level = parameters.get('snow_level', self.snow_level)
        self.cliff_slope = parameters.get('cliff_slope', self.cliff_slope)
        self.edge_softness = parameters.get('edge_softness', self.edge_softness)
        self.bank_width = parameters.get('bank_width', self.bank_width)
        self.supersampling_quality = parameters.get('supersampling_quality', self.supersampling_quality)
        self.biome_seed = parameters.get('biome_seed', self.biome_seed)

    def _calculate_parameter_hash(self, parameters):
        """
        Erstellt Hash für Parameter-Cache-Invalidation
        """
        import hashlib
        param_str = str(sorted(parameters.items()))
        return hashlib.md5(param_str.encode()).hexdigest()


class BaseBiomeClassifier:
    """
    Gauß-basierte Klassifizierung von 15 Grundbiomen mit Multi-Factor-Weighting
    """

    def __init__(self, temp_factor=1.0, wetness_factor=1.0, elevation_factor=1.0, soil_moisture_factor=1.0):
        """
        Initialisiert Base-Biome-Classifier mit Gewichtungsfaktoren
        """
        self.temp_factor = temp_factor
        self.wetness_factor = wetness_factor
        self.elevation_factor = elevation_factor
        self.soil_moisture_factor = soil_moisture_factor

        # Base-Biome Definitionen (15 Biome nach Whittaker-Standards)
        self.biome_definitions = {
            0: {'name': 'ice_cap', 'temp': (-40, -5), 'precip': (0, 300), 'elevation': (0, 8000), 'moisture': (0, 200)},
            1: {'name': 'tundra', 'temp': (-15, 5), 'precip': (100, 600), 'elevation': (0, 2000), 'moisture': (100, 400)},
            2: {'name': 'taiga', 'temp': (-10, 15), 'precip': (300, 1200), 'elevation': (50, 2500), 'moisture': (300, 800)},
            3: {'name': 'grassland', 'temp': (0, 25), 'precip': (200, 800), 'elevation': (10, 1500), 'moisture': (200, 600)},
            4: {'name': 'temperate_forest', 'temp': (5, 25), 'precip': (600, 2000), 'elevation': (0, 2000), 'moisture': (400, 1000)},
            5: {'name': 'mediterranean', 'temp': (8, 30), 'precip': (300, 900), 'elevation': (0, 1200), 'moisture': (200, 600)},
            6: {'name': 'desert', 'temp': (10, 50), 'precip': (0, 250), 'elevation': (0, 2000), 'moisture': (0, 100)},
            7: {'name': 'semi_arid', 'temp': (5, 35), 'precip': (200, 600), 'elevation': (0, 1800), 'moisture': (100, 400)},
            8: {'name': 'tropical_rainforest', 'temp': (20, 35), 'precip': (1500, 4000), 'elevation': (0, 1500), 'moisture': (800, 1500)},
            9: {'name': 'tropical_seasonal', 'temp': (18, 35), 'precip': (800, 2000), 'elevation': (0, 1200), 'moisture': (400, 1000)},
            10: {'name': 'savanna', 'temp': (15, 35), 'precip': (400, 1200), 'elevation': (0, 1800), 'moisture': (200, 600)},
            11: {'name': 'montane_forest', 'temp': (0, 20), 'precip': (800, 3000), 'elevation': (800, 3500), 'moisture': (600, 1200)},
            12: {'name': 'swamp', 'temp': (5, 35), 'precip': (800, 3000), 'elevation': (0, 200), 'moisture': (800, 1500)},
            13: {'name': 'coastal_dunes', 'temp': (5, 35), 'precip': (300, 1500), 'elevation': (0, 100), 'moisture': (200, 800)},
            14: {'name': 'badlands', 'temp': (-5, 45), 'precip': (0, 400), 'elevation': (200, 2500), 'moisture': (0, 200)}
        }

    def classify_base_biomes(self, heightmap, temp_map, precip_map, soil_moist_map):
        """
        Klassifiziert Base-Biomes mit wissenschaftlich fundierter Multi-Factor-Analysis
        """
        height, width = heightmap.shape
        fitness_maps = np.zeros((height, width, 15), dtype=np.float32)

        # Für jedes Base-Biome Fitness berechnen
        for biome_id, biome_def in self.biome_definitions.items():
            # Gauß-Fitness für jeden Faktor
            temp_fitness = self._calculate_gaussian_fitness(temp_map, biome_def['temp'])
            precip_fitness = self._calculate_gaussian_fitness(precip_map, biome_def['precip'])
            elevation_fitness = self._calculate_gaussian_fitness(heightmap, biome_def['elevation'])
            moisture_fitness = self._calculate_gaussian_fitness(soil_moist_map, biome_def['moisture'])

            # Gewichtete Kombination (Temperature 30%, Precipitation 35%, Elevation 20%, Moisture 15%)
            combined_fitness = (
                temp_fitness * 0.30 * self.temp_factor +
                precip_fitness * 0.35 * self.wetness_factor +
                elevation_fitness * 0.20 * self.elevation_factor +
                moisture_fitness * 0.15 * self.soil_moisture_factor
            )

            fitness_maps[:, :, biome_id] = combined_fitness

        # Dominantes Biome pro Pixel
        dominant_biomes = np.argmax(fitness_maps, axis=2)
        return dominant_biomes.astype(np.uint8)

    def _calculate_gaussian_fitness(self, data_map, value_range):
        """
        Berechnet Gauß-Fitness für gegebenen Wertebereich
        """
        min_val, max_val = value_range
        range_center = (min_val + max_val) / 2
        range_width = max_val - min_val

        if range_width == 0:
            return np.where(data_map == min_val, 1.0, 0.0)

        # Gauß-Funktion: Maximum bei Center, Sigma = range_width/4
        sigma = range_width / 4.0
        normalized_distance = (data_map - range_center) / sigma
        fitness = np.exp(-0.5 * normalized_distance ** 2)

        # Außerhalb des Bereichs: stark reduzierte Fitness
        outside_range = (data_map < min_val) | (data_map > max_val)
        fitness[outside_range] *= 0.1

        return fitness


class SuperBiomeOverrideSystem:
    """
    Priority-basiertes Override-System mit 11 speziellen Biom-Bedingungen
    """

    def __init__(self, sea_level=10.0, bank_width=3.0, edge_softness=1.0,
                 alpine_level=1500.0, snow_level=2000.0, cliff_slope=60.0):
        """
        Initialisiert Super-Biome-Override-System mit Unified-Edge-Softness-Control
        """
        self.sea_level = sea_level
        self.bank_width = bank_width
        self.edge_softness = edge_softness
        self.alpine_level = alpine_level
        self.snow_level = snow_level
        self.cliff_slope = cliff_slope

        # Super-Biome-Offset (nach 15 Base-Biomes)
        self.super_biome_offset = 15

    def apply_super_biome_overrides(self, heightmap, temp_map, water_biomes_map, soil_moist_map):
        """
        Wendet alle Super-Biome-Overrides in Priority-Reihenfolge an
        """
        height, width = heightmap.shape
        super_biome_mask = np.zeros((height, width), dtype=np.uint8)
        super_biome_probabilities = {}

        # Priority 0-4: Water-basierte Super-Biomes (höchste Priorität)
        super_biome_mask[water_biomes_map == 4] = self.super_biome_offset + 1  # Lake
        super_biome_mask[water_biomes_map == 3] = self.super_biome_offset + 2  # Grand River
        super_biome_mask[water_biomes_map == 2] = self.super_biome_offset + 3  # River
        super_biome_mask[water_biomes_map == 1] = self.super_biome_offset + 4  # Creek

        # Ocean-Detection (Priority 0)
        ocean_mask = self._detect_ocean_connectivity(heightmap, water_biomes_map)
        super_biome_mask[ocean_mask] = self.super_biome_offset + 0  # Ocean

        # Priority 5-6: Topographie-basierte Super-Biomes
        cliff_probabilities = self._calculate_cliff_probabilities(heightmap)
        super_biome_probabilities['cliff'] = cliff_probabilities

        beach_probabilities = self._calculate_beach_probabilities(heightmap, ocean_mask)
        super_biome_probabilities['beach'] = beach_probabilities

        # Priority 7-8: Proximity-basierte Super-Biomes
        lake_edge_probabilities = self._calculate_lake_edge_probabilities(water_biomes_map)
        super_biome_probabilities['lake_edge'] = lake_edge_probabilities

        river_bank_probabilities = self._calculate_river_bank_probabilities(water_biomes_map)
        super_biome_probabilities['river_bank'] = river_bank_probabilities

        # Priority 9-10: Höhen-basierte Super-Biomes
        snow_probabilities = self._calculate_snow_level_probabilities(heightmap, temp_map)
        super_biome_probabilities['snow_level'] = snow_probabilities

        alpine_probabilities = self._calculate_alpine_level_probabilities(heightmap, temp_map)
        super_biome_probabilities['alpine_level'] = alpine_probabilities

        return super_biome_mask, super_biome_probabilities

    def _detect_ocean_connectivity(self, heightmap, water_biomes_map):
        """
        Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind
        """
        height, width = heightmap.shape
        ocean_mask = np.zeros((height, width), dtype=bool)

        # Potentielle Ocean-Bereiche (unter sea_level)
        potential_ocean = heightmap < self.sea_level

        if not np.any(potential_ocean):
            return ocean_mask

        # Flood-Fill von Kartenrändern
        visited = np.zeros((height, width), dtype=bool)
        queue = deque()

        # Alle Rand-Pixel unter sea_level als Seeds
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

    def _calculate_cliff_probabilities(self, heightmap):
        """
        Berechnet Cliff-Probabilities mit Slope-Threshold und Edge-Softness
        """
        # Gradient berechnen
        grad_y, grad_x = np.gradient(heightmap)
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        slope_degrees = np.degrees(np.arctan(slope_magnitude))

        # Sigmoid-Probability basierend auf cliff_slope und edge_softness
        cliff_probabilities = self._sigmoid((slope_degrees - self.cliff_slope) / self.edge_softness)
        cliff_probabilities = np.maximum(0, cliff_probabilities)

        return cliff_probabilities

    def _calculate_beach_probabilities(self, heightmap, ocean_mask):
        """
        Beach: Nähe zu Ocean + h <= sea_level + 5 mit weichen Übergängen
        """
        if not np.any(ocean_mask):
            return np.zeros_like(heightmap, dtype=np.float32)

        # Distance-Transform zu Ocean
        ocean_distance = distance_transform_edt(~ocean_mask)

        # Beach-Bedingung
        elevation_condition = heightmap <= (self.sea_level + 5)
        proximity_condition = ocean_distance <= self.bank_width
        beach_condition = elevation_condition & proximity_condition & (~ocean_mask)

        # Probability mit Edge-Softness
        beach_probabilities = np.maximum(0, 1 - np.power(ocean_distance / self.bank_width, self.edge_softness))
        beach_probabilities[~beach_condition] = 0

        return beach_probabilities

    def _calculate_lake_edge_probabilities(self, water_biomes_map):
        """
        Lake Edge: Nähe zu Lake + nicht selbst Lake
        """
        lake_mask = water_biomes_map == 4

        if not np.any(lake_mask):
            return np.zeros_like(water_biomes_map, dtype=np.float32)

        lake_distance = distance_transform_edt(~lake_mask)
        lake_edge_condition = (lake_distance <= self.bank_width) & (~lake_mask)

        lake_edge_probabilities = np.maximum(0, 1 - np.power(lake_distance / self.bank_width, self.edge_softness))
        lake_edge_probabilities[~lake_edge_condition] = 0

        return lake_edge_probabilities

    def _calculate_river_bank_probabilities(self, water_biomes_map):
        """
        River Bank: Nähe zu River/Grand River/Creek + nicht selbst Wasser
        """
        river_mask = (water_biomes_map >= 1) & (water_biomes_map <= 3)

        if not np.any(river_mask):
            return np.zeros_like(water_biomes_map, dtype=np.float32)

        river_distance = distance_transform_edt(~river_mask)
        river_bank_condition = (river_distance <= self.bank_width) & (water_biomes_map == 0)

        river_bank_probabilities = np.maximum(0, 1 - np.power(river_distance / self.bank_width, self.edge_softness))
        river_bank_probabilities[~river_bank_condition] = 0

        return river_bank_probabilities

    def _calculate_snow_level_probabilities(self, heightmap, temp_map):
        """
        Snow Level: h > snow_level + 500*(1 + temp_map(x,y)/10) mit temperaturabhängigen Übergängen
        """
        temp_adjusted_snow_level = self.snow_level + 500 * (1 + temp_map / 10)
        snow_height_diff = heightmap - temp_adjusted_snow_level

        snow_probabilities = self._sigmoid(snow_height_diff / (100 * self.edge_softness))
        snow_probabilities = np.maximum(0, snow_probabilities)

        return snow_probabilities

    def _calculate_alpine_level_probabilities(self, heightmap, temp_map):
        """
        Alpine Level: h > alpine_level + 500*(1 + temp_map(x,y)/10) mit temperaturabhängigen Übergängen
        """
        temp_adjusted_alpine_level = self.alpine_level + 500 * (1 + temp_map / 10)
        alpine_height_diff = heightmap - temp_adjusted_alpine_level

        alpine_probabilities = self._sigmoid(alpine_height_diff / (200 * self.edge_softness))
        alpine_probabilities = np.maximum(0, alpine_probabilities)

        return alpine_probabilities

    def _sigmoid(self, x):
        """
        Numerisch stabile Sigmoid-Funktion für weiche Übergänge
        """
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class SupersamplingManager:
    """
    2x2-Supersampling mit diskretisierter Zufalls-Rotation für Natural-Biome-Transitions
    """

    def __init__(self, biome_seed=42, supersampling_quality=1.0):
        """
        Initialisiert Supersampling-Manager mit reproduzierbarer Randomization
        """
        self.biome_seed = biome_seed
        self.supersampling_quality = supersampling_quality

    def apply_supersampling(self, biome_map, super_biome_probabilities):
        """
        Wendet 2x2-Supersampling mit diskretisierter Rotation an
        """
        height, width = biome_map.shape
        super_height, super_width = height * 2, width * 2
        biome_map_super = np.zeros((super_height, super_width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Diskretisierte Rotations-Zuweisung mit Primzahlen
                rotation_hash = (self.biome_seed + 12345 + x * 997 + y * 991) % 4

                # Sub-Pixel-Anordnung basierend auf Rotation
                if rotation_hash == 0:    # 0° Rotation
                    sub_order = [(0, 0), (0, 1), (1, 0), (1, 1)]  # TL, TR, BL, BR
                elif rotation_hash == 1:  # 90° Rotation
                    sub_order = [(1, 0), (0, 0), (1, 1), (0, 1)]  # BL, TL, BR, TR
                elif rotation_hash == 2:  # 180° Rotation
                    sub_order = [(1, 1), (1, 0), (0, 1), (0, 0)]  # BR, BL, TR, TL
                else:                     # 270° Rotation
                    sub_order = [(0, 1), (1, 1), (0, 0), (1, 0)]  # TR, BR, TL, BL

                base_biome = biome_map[y, x]

                # Sub-Pixel-Zuweisung mit Super-Biome-Probabilities
                for i, (sub_y, sub_x) in enumerate(sub_order):
                    target_y = y * 2 + sub_y
                    target_x = x * 2 + sub_x

                    # Sub-Pixel-Seed für Probabilistic-Assignment
                    sub_seed = (self.biome_seed + 54321 + x * 4 + y * 4 + i * 3571) % 1000
                    probability = sub_seed / 1000.0  # [0.000, 0.999]

                    # Super-Biome-Assignment prüfen
                    assigned_biome = base_biome

                    if super_biome_probabilities:
                        for super_biome_name, prob_map in super_biome_probabilities.items():
                            if y < prob_map.shape[0] and x < prob_map.shape[1]:
                                super_prob = prob_map[y, x] * self.supersampling_quality
                                if probability < super_prob:
                                    assigned_biome = self._get_super_biome_index(super_biome_name)
                                    break

                    biome_map_super[target_y, target_x] = assigned_biome

        return biome_map_super

    def _get_super_biome_index(self, super_biome_name):
        """
        Konvertiert Super-Biome-Namen zu Index
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