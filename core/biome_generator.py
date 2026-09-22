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
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter
from collections import deque
from typing import Dict, Any, List, Optional
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
        # Eignungsfeld (Godot-Export, Punkt b): nicht nur der Gewinner,
        # sondern die drei bestpassenden Basisbiome je Ort mit ihrem Anteil.
        self.biom_top3_ids = None               # 3D numpy.uint8 (H,W,3), Biom-Kennungen, absteigend
        self.biom_top3_anteil = None            # 3D numpy.float32 (H,W,3), Anteile, Summe 1
        self.biom_eindeutigkeit = None          # 2D numpy.float32, rohe Eignungssumme vor Normalisierung
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

    def set_active_parameters(self, parameters: Dict[str, Any]):
        """Setzt die Parameter, die alle _calc_*-Methoden bis zur nächsten frischen
        Anfrage verwenden (vom GenerationOrchestrator aufgerufen). Biome speichert
        Parameter als Instanz-Attribute (self.biome_temp_factor etc.), nicht als
        eigenes dict - _update_parameters() übernimmt das bereits."""
        self._update_parameters(parameters)

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, calculate_biomes() ohne
        injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from managers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    def _ensure_sub_components(self):
        """Lazy-Initialisierung der Sub-Komponenten mit den aktuellen Parametern
        (wie zuvor in _try_cpu_classification(), jetzt auch von den einzelnen
        _calc_*-Methoden genutzt)."""
        if not self.base_biome_classifier:
            self.base_biome_classifier = BaseBiomeClassifier(
                self.biome_temp_factor, self.biome_wetness_factor,
                self.elevation_factor, self.soil_moisture_factor)
        if not self.super_biome_override_system:
            self.super_biome_override_system = SuperBiomeOverrideSystem(
                self.sea_level, self.bank_width, self.edge_softness,
                self.alpine_level, self.snow_level, self.cliff_slope,
                shader_manager=self.shader_manager, data_lod_manager=self.data_lod_manager)
        if not self.supersampling_manager:
            self.supersampling_manager = SupersamplingManager(
                self.biome_seed, self.supersampling_quality, shader_manager=self.shader_manager)

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

            # Höhere Lagen sollten tendenziell kälter sein - die Bedingung war
            # umgekehrt: "low <= high" (Tiefland kälter/gleich Hochland) wurde als
            # gültig durchgewunken, das eigentlich realistische "low > high"
            # (Tiefland wärmer) loggte fälschlich eine "inverted"-Warnung.
            if mean_temp_low >= mean_temp_high:
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

        Läuft über die einzeln aufrufbaren _calc_*-Methoden (siehe
        managers/calculator_graph.py - Biome-Calculator-Knoten #23-#27 aus
        docs/generation_pipeline_dependencies.md). Die echte GUI-Pipeline
        (GenerationOrchestrator) ruft dieselben Methoden ab jetzt einzeln über den
        globalen CalculatorDispatcher auf (Tracker #16 LOD-Lockstep-Umbau) - der
        Effekt ist identisch, da beide Wege denselben Storage nutzen. Externes
        Verhalten (Signatur/Rückgabe, Mutation des übergebenen biome_data) bleibt
        unverändert.
        """
        try:
            self.logger.info("Starting CPU-based biome classification")

            self._ensure_data_lod_manager()
            self._ensure_sub_components()

            lod_level = biome_data.lod_level

            # Standalone-Convenience-Pfad: input_data kommt hier als direktes dict,
            # nicht aus dem DataLODManager - für die _calc_*-Methoden (die jetzt
            # IMMER aus dem Storage lesen) gespiegelt, analog zu Geology/Water.
            self.data_lod_manager.set_calculator_output(
                "terrain.redistribution", lod_level, {"heightmap": input_data['heightmap']})
            self.data_lod_manager.set_calculator_output(
                "weather.temperature", lod_level, {"temp_map": input_data['temp_map']})
            self.data_lod_manager.set_calculator_output(
                "weather.precipitation", lod_level, {"precip_map": input_data['precip_map']})
            self.data_lod_manager.set_calculator_output(
                "water.soil_moisture", lod_level, {"soil_moist_map": input_data['soil_moist_map']})
            # water.manning_flow (GEMALTE Klassifikation), nicht
            # water.flow_network (Zentrallinie) - siehe
            # core/water_generator.py._calc_manning_flow().
            self.data_lod_manager.set_calculator_output(
                "water.manning_flow", lod_level, {"water_biomes_map": input_data['water_biomes_map']})

            for calculator_id in (
                "biome.base_classification", "biome.super_override", "biome.integrate_layers",
                "biome.supersampling", "biome.climate_classification",
            ):
                getattr(self, "_calc_" + calculator_id.split(".", 1)[1])(calculator_id, lod_level)

            # Results setzen
            biome_data.biome_map = self.data_lod_manager.get_calculator_output(
                "biome.integrate_layers", "biome_map", lod_level)
            biome_data.biome_map_super = self.data_lod_manager.get_calculator_output(
                "biome.supersampling", "biome_map_super", lod_level)
            biome_data.super_biome_mask = self.data_lod_manager.get_calculator_output(
                "biome.super_override", "super_biome_mask", lod_level)
            biome_data.climate_classification = self.data_lod_manager.get_calculator_output(
                "biome.climate_classification", "climate_classification", lod_level)

            return True

        except Exception as e:
            self.logger.warning(f"CPU classification error: {e}")
            return False

    def assemble_biome_data(self, lod_level: int, parameters: Dict[str, Any]):
        """
        Funktionsweise: Baut das finale BiomeData-Objekt aus den einzeln
        gespeicherten Calculator-Outputs zusammen (inkl. Statistics/Validity-State)
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald alle 5 Biome-
            Calculator-Knoten ein LOD abgeschlossen haben (siehe Task 18 im
            LOD-Lockstep-Umbau)
        """
        final_biome_map = self.data_lod_manager.get_calculator_output(
            "biome.integrate_layers", "biome_map", lod_level)
        biome_map_super = self.data_lod_manager.get_calculator_output(
            "biome.supersampling", "biome_map_super", lod_level)
        super_biome_mask = self.data_lod_manager.get_calculator_output(
            "biome.super_override", "super_biome_mask", lod_level)
        climate_classification = self.data_lod_manager.get_calculator_output(
            "biome.climate_classification", "climate_classification", lod_level)
        top3_ids = self.data_lod_manager.get_calculator_output(
            "biome.base_classification", "biom_top3_ids", lod_level)
        top3_anteil = self.data_lod_manager.get_calculator_output(
            "biome.base_classification", "biom_top3_anteil", lod_level)
        eindeutigkeit = self.data_lod_manager.get_calculator_output(
            "biome.base_classification", "biom_eindeutigkeit", lod_level)

        if final_biome_map is None or super_biome_mask is None:
            raise ValueError(f"assemble_biome_data: fehlende Calculator-Outputs für LOD {lod_level}")

        biome_data = BiomeData()
        biome_data.lod_level = lod_level
        biome_data.actual_size = final_biome_map.shape[0]
        biome_data.parameter_hash = self._calculate_parameter_hash(parameters)
        biome_data.biome_map = final_biome_map
        biome_data.biome_map_super = biome_map_super
        biome_data.super_biome_mask = super_biome_mask
        biome_data.climate_classification = climate_classification
        biome_data.biom_top3_ids = top3_ids
        biome_data.biom_top3_anteil = top3_anteil
        biome_data.biom_eindeutigkeit = eindeutigkeit
        biome_data.biome_statistics = self._calculate_biome_statistics(biome_data)
        biome_data.validity_state = self._validate_biome_data(biome_data)

        return biome_data

    def _temp_map_juli(self, lod_level: int):
        """Julitemperatur (Periode 3 von 6, siehe weather_generator.py
        _calc_temperature()-Kommentar zu m/6-Zeitpunkten) statt des
        Jahresmittels aus 'temp_map' - siehe Kommentar bei 'temp_map_juli'
        in _get_prepared_biome_inputs()."""
        monatlich = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        if monatlich is not None and len(monatlich) > 3 and monatlich[3] is not None:
            return monatlich[3]
        return self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map", lod_level)

    def _get_prepared_biome_inputs(self, lod_level: int, needed: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Holt NUR die tatsächlich angeforderten Biome-Dependencies
        (Terrain/Weather/Water-Outputs) für dieses LOD. `needed` MUSS pro
        _calc_*-Methode exakt deren echte CALCULATOR_GRAPH-Abhängigkeiten
        widerspiegeln - sonst würde z.B. biome.climate_classification (haengt
        laut Graph NUR von weather.temperature/weather.precipitation ab)
        fälschlich auch auf water.soil_moisture/water.manning_flow warten,
        obwohl der Dispatcher diesen Knoten längst für bereit hält.
        """
        if needed is None:
            needed = ["heightmap", "temp_map", "precip_map", "soil_moist_map", "water_biomes_map"]

        fetchers = {
            "heightmap": lambda: self.data_lod_manager.get_calculator_combined_heightmap(lod_level),
            "temp_map": lambda: self.data_lod_manager.get_calculator_output(
                "weather.temperature", "temp_map", lod_level),
            # JULITEMPERATUR, NICHT JAHRESMITTEL (2026-08-11, Nutzerbefund via
            # Macchia-Fehlklassifikation, siehe docs/OFFENE_PUNKTE.md 9.1).
            # `weather.temperature`s "temp_map" ist das JAHRESMITTEL ueber alle
            # 6 saisonalen Perioden (weather_generator.py _calc_temperature(),
            # `temp_map = np.mean(np.stack(monthly_temp_maps, ...)`). Sowohl
            # BaseBiomeClassifier.biome_definitions ("temp ist die
            # JULITEMPERATUR", core/biome_generator.py Kommentar bei
            # self.biome_definitions) als auch die Alpin-/Firn-Schwellen
            # (BAUMGRENZE_JULI_C/FIRN_JULI_C) sind aber explizit gegen Juli
            # kalibriert - mit dem Jahresmittel verglichen liegt jeder Wert um
            # etwa die halbe Jahresspanne zu kalt. Gemessen am Beispiel
            # Macchia (Jahresmittel/Spanne 16.9/17.5, Juli-Referenz laut
            # docs/BIOME_MATRIX.md 25.6 Grad): der Median-Landpixel kam mit
            # dem Jahresmittel auf 16.4 Grad statt Juli, wodurch Steineichen-
            # wald (Bereich 21-28 Grad) fast ueberall ausserhalb seines
            # Bereichs lag und komplett aus den Top-Biomen verschwand, waehrend
            # Bruchwald (Bereich 14-22, toleranter nach unten) 47% der Region
            # dominierte - beide Biome sind fuer Macchia als Affinitaet
            # gelistet, aber in völlig falschem Verhaeltnis.
            # `temp_map_monthly[3]` ist bereits vorhanden (Periode 3 = Juli bei
            # TICKS_JE_JAHR=6 und Zeitpunkt m/6, siehe dortiger Kommentar) -
            # kein neuer Rechenweg noetig, nur die richtige bereits berechnete
            # Schicht lesen. Fallback auf "temp_map" nur, falls
            # temp_map_monthly aus irgendeinem Grund fehlt/leer ist (alter
            # Einzelschicht-Fallback-Pfad ohne WELTKARTE_AKTIV o.ae.).
            "temp_map_juli": lambda: self._temp_map_juli(lod_level),
            "precip_map": lambda: self.data_lod_manager.get_calculator_output(
                "weather.precipitation", "precip_map", lod_level),
            "soil_moist_map": lambda: self.data_lod_manager.get_calculator_output(
                "water.soil_moisture", "soil_moist_map", lod_level),
            # GEMALTE Wasser-Klassifikation (water.manning_flow), nicht die ein
            # Pixel breite Zentrallinie aus water.flow_network - Biome sollen
            # den Fluss in seiner tatsaechlichen Breite sehen. Siehe
            # core/water_generator.py._calc_manning_flow().
            "water_biomes_map": lambda: self.data_lod_manager.get_calculator_output(
                "water.manning_flow", "water_biomes_map", lod_level),
        }

        values = {key: fetchers[key]() for key in needed}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"Biome: fehlende Dependencies für LOD {lod_level}: {', '.join(missing)}")

        return values

    def _calc_preseed_hint(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'biome.preseed_hint' - billiger Vorab-Biome-Schätzwert
        NUR aus Slope (Süd-/Nordhang) + Breitengrad, OHNE jede Wetter-/Wasser-
        Abhängigkeit (siehe calculator_graph.py: hängt nur von terrain.redistribution/
        terrain.slope ab). Löst das Henne-Ei-Problem für water.soil_moisture in
        der allerersten LOD-Runde (Biome-Preseed-Plan Punkt B) - ab der zweiten
        LOD-Stufe nutzt water.soil_moisture stattdessen die ECHTE biome_map der
        Vorstufe, dieser Knoten ist dann nur noch fürs allererste LOD relevant.
        Bewusst grobe Regel-Heuristik statt Gauß-Fitness wie bei der echten
        Klassifikation - dient nur als Kapazitäts-/Verdunstungs-Hinweis, nicht
        als sichtbare Anzeige.
        """
        # erosion.slope, nicht terrain.slope: der Vorab-Biom-Schaetzwert
        # bewertet Steilheit, und die entsteht erst durch die Erosion.
        # terrain.slope kennt nur das unerodierte Rauschen.
        slopemap = self.data_lod_manager.get_calculator_output(
            "erosion.slope", "slopemap", lod_level)
        if slopemap is None:
            raise ValueError(f"biome.preseed_hint: slopemap für LOD {lod_level} nicht verfügbar")

        latitude = self.data_lod_manager.get_map_latitude()
        height, width = slopemap.shape[:2]

        # Nord/Süd-Hangausrichtung: dz/drow > 0 heißt bergauf Richtung Norden
        # (Zeile height-1 = Norden, siehe core/terrain_generator.py.
        # _calculate_cpu_slopes()s np.gradient-Konvention) = Südhang (trockener,
        # mehr Sonne - siehe die in dieser Session umgesetzte Hangausrichtungs-
        # Solar-Kopplung in weather_generator.py). south_facing in [-1,1]:
        # +1 = voll Südhang, -1 = voll Nordhang, 0 = eben/Ost-West-Hang.
        dz_dx = slopemap[:, :, 0]
        dz_dy = slopemap[:, :, 1]
        slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        south_facing = np.divide(dz_dy, slope_magnitude, out=np.zeros_like(dz_dy),
                                  where=slope_magnitude > 1e-6)

        # Feuchte-Score: negativ=trockener, positiv=feuchter - Südhang senkt
        # den Score, Nordhang hebt ihn, stärker gewichtet je steiler der Hang.
        # ---------------------------------------------------------------
        # FEUCHTE-SCORE
        #
        # Bis 2026-07-29 war das EINE Zeile: Hangausrichtung mal Steilheit.
        # Zwei Folgen davon, beide gemessen:
        #   - auf flachem Gelaende ist slope_magnitude null, also der Score
        #     null, also immer die Basisklasse. Eine ebene Karte wurde
        #     durchgehend einfarbig.
        #   - kein Hoehen- oder Lagebezug ging ein. Sumpf und Bergwald koennen
        #     so gar nicht entstehen, weil die Groessen, die sie definieren,
        #     im Score nicht vorkamen.
        #
        # Alles Folgende liegt VOR der Wetteriteration bereits vor - der
        # Knoten bleibt damit frei von water.*/weather.*-Abhaengigkeiten und
        # loest weiterhin das Henne-Ei-Problem, fuer das er gebaut wurde.
        # ---------------------------------------------------------------
        aspect_score = -south_facing * np.clip(slope_magnitude, 0.0, 1.0) * 0.6

        heightmap = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        if heightmap is None or heightmap.shape[:2] != (height, width):
            # Ohne Hoehendaten bleibt es beim reinen Hang-Score - kein harter
            # Fehler, aber dann fehlen Sumpf und Hoehenguertel.
            moisture_score = aspect_score
            hoehe_ueber_sohle = None
            hoehe_m = None
        else:
            hoehe_m = np.asarray(heightmap, dtype=np.float32)

            # HOEHE UEBER DER LOKALEN TALSOHLE. Der Schluessel gegen
            # "einzelne Sumpfpixel": Talsohlen sind zusammenhaengende
            # FLAECHEN, also ist auch diese Groesse zusammenhaengend. "Tief"
            # ist hier relativ zur Umgebung, nicht absolut - ein Hochtal auf
            # 2000 m ist fuer seinen Bach genauso Talsohle wie eine Ebene auf
            # 100 m.
            fenster = max(5, int(min(height, width) * 0.12) | 1)
            sohle = ndimage.minimum_filter(hoehe_m, size=fenster, mode='nearest')
            sohle = ndimage.uniform_filter(sohle, size=fenster, mode='nearest')
            spanne = float(hoehe_m.max() - hoehe_m.min()) or 1.0
            hoehe_ueber_sohle = np.clip((hoehe_m - sohle) / (0.25 * spanne), 0.0, 1.0)

            # Tief ueber der Sohle = feucht (Wasser sammelt sich dort),
            # hoch = trocken. Doppelt so stark gewichtet wie die
            # Hangausrichtung, weil es der belastbarere Hinweis ist.
            talnaehe_score = (0.5 - hoehe_ueber_sohle) * 1.2
            moisture_score = aspect_score + talnaehe_score

        # ---------------------------------------------------------------
        # KLIMAZONEN nach Breitengrad - jetzt SIEBEN Baender statt drei.
        #
        # Die alte Dreiteilung (0-31 / 32-63 / 64-90 Grad) kannte den
        # subtropischen TROCKENGUERTEL nicht: bei 20-30 Grad stand
        # tropical_seasonal mit Feuchtekapazitaet 70, also feucht - genau die
        # Zone, die trocken sein muesste. Gemessen war das der Grund, warum
        # der Wuestenguertel im Niederschlag nie ankam.
        #
        # Je Band: (trocken, basis, feucht).
        # ---------------------------------------------------------------
        zonen = (
            (10.0,  (9, 9, 8)),      # aequatorial: trop_seasonal / trop_rainforest
            (20.0,  (10, 9, 8)),     # Monsun: savanna / trop_seasonal / rainforest
            (33.0,  (6, 7, 10)),     # WUESTENGUERTEL: desert / semi_arid / savanna
            (45.0,  (7, 5, 4)),      # subtropisch: semi_arid / mediterran / Wald
            (58.0,  (7, 3, 4)),      # gemaessigt: semi_arid / grassland / Wald
            (68.0,  (3, 2, 2)),      # boreal: grassland / taiga
            (91.0,  (0, 1, 2)),      # polar: ice_cap / tundra / taiga
        )
        abs_lat = min(abs(latitude), 90.0)
        for grenze, (dry_id, base_id, wet_id) in zonen:
            if abs_lat < grenze:
                break

        preseed_biome_map = np.full((height, width), base_id, dtype=np.uint8)
        preseed_biome_map[moisture_score < -0.15] = dry_id
        preseed_biome_map[moisture_score > 0.15] = wet_id

        if hoehe_m is not None:
            # ---------------------------------------------------------------
            # HOEHENGUERTEL. Er ueberschreibt die Zone lokal - dieselbe Regel
            # wie die Breite, nur senkrecht.
            #
            # Die Grenzen WANDERN mit dem Breitengrad: die Baumgrenze liegt am
            # Aequator bei rund 3500 m, bei 60 Grad bei rund 700 m. Genau das
            # fehlte bisher, und es ist der Grund, warum alpine und cliff nicht
            # stimmen konnten - sie hingen an festen Hoehen.
            baumgrenze = float(np.interp(abs_lat, [0, 30, 45, 60, 75, 90],
                                          [3600, 3200, 2200, 900, 400, 0]))
            bergwald_unten = baumgrenze * 0.45

            bergwald = (hoehe_m >= bergwald_unten) & (hoehe_m < baumgrenze)
            preseed_biome_map[bergwald] = 11        # montane_forest
            preseed_biome_map[hoehe_m >= baumgrenze] = 1   # tundra (alpin)
            preseed_biome_map[hoehe_m >= baumgrenze * 1.35] = 0   # ice_cap

            # ---------------------------------------------------------------
            # SUMPF. Nicht nur flach+tief+nass, sondern ausdruecklich AUCH
            # breitengradabhaengig (Nutzer-Einwand 2026-07-29, und er stimmt):
            # die grossen Moore liegen in Westsibirien, am Hudson Bay, in
            # Finnland und Kanada. In kalten Zonen ist die Verdunstung klein
            # gegenueber dem Niederschlag, das Wasser bleibt oben. Im
            # Trockenguertel braeuchte es dafuer ein Vielfaches an Zufluss.
            #
            # Umgesetzt als Schwelle, die mit der Breite MILDER wird - kein
            # Verbot, nur eine Gewichtung.
            sumpf_schwelle = float(np.interp(abs_lat, [0, 20, 35, 50, 70, 90],
                                              [0.06, 0.02, 0.02, 0.10, 0.14, 0.05]))
            sumpfig = ((hoehe_ueber_sohle < sumpf_schwelle)
                       & (slope_magnitude < 0.05)
                       & (hoehe_m < baumgrenze * 0.6))
            preseed_biome_map[sumpfig] = 12         # swamp

        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"preseed_biome_map": preseed_biome_map})

    def _calc_base_classification(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'biome.base_classification' (#23) - Sibling zu super_override"""
        self._ensure_sub_components()
        inputs = self._get_prepared_biome_inputs(
            lod_level, needed=["heightmap", "temp_map_juli", "precip_map", "soil_moist_map"])
        # region_map seit 2026-08-07: sie traegt die Regionsaffinitaet. Ohne
        # sie klassifiziert der Aufruf wie zuvor, nur nach Klima.
        region_map = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "region_map", lod_level)
        # klassifiziere_mit_eignungsfeld() statt classify_base_biomes(): es
        # liefert dasselbe base_biome_map und zusaetzlich das Eignungsfeld,
        # ohne die Gauss-Rechnung ein zweites Mal zu fahren.
        (base_biome_map, top3_ids, top3_anteil,
         eindeutigkeit) = self.base_biome_classifier.klassifiziere_mit_eignungsfeld(
            inputs['heightmap'], inputs['temp_map_juli'], inputs['precip_map'],
            inputs['soil_moist_map'], region_map=region_map)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
            "base_biome_map": base_biome_map,
            "biom_top3_ids": top3_ids,
            "biom_top3_anteil": top3_anteil,
            "biom_eindeutigkeit": eindeutigkeit,
        })

    def _calc_super_override(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'biome.super_override' (#24) - Sibling zu base_classification"""
        self._ensure_sub_components()
        inputs = self._get_prepared_biome_inputs(
            lod_level, needed=["heightmap", "temp_map_juli", "water_biomes_map", "soil_moist_map"])
        # see_eis (docs/OFFENE_PUNKTE.md 3.6) - OHNE Pflichtpruefung, gleiches
        # Muster wie region_map/seegrad anderswo: nur auf der Weltkarte
        # vorhanden, None auf dem alten Pfad laesst Ocean unveraendert.
        see_eis = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "see_eis", lod_level)
        if see_eis is not None and see_eis.shape[0] != inputs['heightmap'].shape[0]:
            from scipy import ndimage
            faktor = inputs['heightmap'].shape[0] / see_eis.shape[0]
            see_eis = ndimage.zoom(see_eis.astype(np.uint8), faktor, order=0).astype(bool)
        super_biome_mask, super_biome_probabilities = self.super_biome_override_system.apply_super_biome_overrides(
            inputs['heightmap'], inputs['temp_map_juli'], inputs['water_biomes_map'], inputs['soil_moist_map'],
            see_eis=see_eis)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level,
            {"super_biome_mask": super_biome_mask, "super_biome_probabilities": super_biome_probabilities})

    def _calc_integrate_layers(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'biome.integrate_layers' (#25)"""
        base_biome_map = self.data_lod_manager.get_calculator_output(
            "biome.base_classification", "base_biome_map", lod_level)
        super_biome_mask = self.data_lod_manager.get_calculator_output(
            "biome.super_override", "super_biome_mask", lod_level)
        if base_biome_map is None or super_biome_mask is None:
            raise ValueError(f"biome.integrate_layers: fehlende Inputs für LOD {lod_level}")

        final_biome_map = self._integrate_biome_layers(base_biome_map, super_biome_mask)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"biome_map": final_biome_map})

    def _calc_supersampling(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'biome.supersampling' (#26)"""
        self._ensure_sub_components()
        final_biome_map = self.data_lod_manager.get_calculator_output(
            "biome.integrate_layers", "biome_map", lod_level)
        super_biome_probabilities = self.data_lod_manager.get_calculator_output(
            "biome.super_override", "super_biome_probabilities", lod_level)
        if final_biome_map is None or super_biome_probabilities is None:
            raise ValueError(f"biome.supersampling: fehlende Inputs für LOD {lod_level}")

        biome_map_super = self.supersampling_manager.apply_supersampling(final_biome_map, super_biome_probabilities)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"biome_map_super": biome_map_super})

    def _calc_climate_classification(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'biome.climate_classification' (#27) - haengt nur von
        temp_map/precip_map ab, kann parallel zu allem anderen in Biome laufen.
        """
        inputs = self._get_prepared_biome_inputs(lod_level, needed=["temp_map", "precip_map"])
        climate_classification = self._create_climate_classification(
            inputs['temp_map'], inputs['precip_map'])
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"climate_classification": climate_classification})

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
        Erstellt Whittaker-Klimazone-Zuordnung - 3-stufiges Fallback-System
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "biome", "climateClassification", {"temp_map": temp_map, "precip_map": precip_map}, {})
                if result.get("success"):
                    return result["climate_map"]
            except Exception as e:
                self.logger.warning(f"GPU climate classification failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        return self._create_climate_classification_cpu(temp_map, precip_map)

    def _create_climate_classification_cpu(self, temp_map, precip_map):
        """CPU-Implementierung der Whittaker-Klimazone-Zuordnung.
        Precip-Schwellen 2026-07-24 auf precip_maps aktuelle Größenordnung
        herunterskaliert (Faktor 50/4000, siehe _rescale_precip_moisture_
        ranges() - precip_map ist keine Jahresmenge mehr, ~50 ist der
        typische Maximalwert, siehe PRECIP_ANNUAL_SCALE_FACTOR in
        core/weather_generator.py)."""
        height, width = temp_map.shape
        climate_map = np.zeros((height, width), dtype=np.uint8)
        precip_scale = 50.0 / 4000.0

        for y in range(height):
            for x in range(width):
                temp = temp_map[y, x]
                precip = precip_map[y, x]

                # Whittaker-Klimazonen
                if temp < -10:
                    climate_map[y, x] = 0  # Arctic
                elif temp < 20:
                    if precip < 200 * precip_scale:
                        climate_map[y, x] = 4  # Arid
                    elif precip < 1000 * precip_scale:
                        climate_map[y, x] = 1  # Boreal/Temperate
                    else:
                        climate_map[y, x] = 2  # Temperate
                else:
                    if precip < 600 * precip_scale:
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

        # Base-Biome Definitionen (15 Biome nach Whittaker-Standards). 'temp' und
        # 'elevation' passen direkt zu temp_map (°C) und heightmap (m). 'precip' und
        # 'moisture' sind hier bewusst die KLASSISCHEN Whittaker-Referenzwerte in
        # Jahres-mm (0-4000) bzw. einer Jahres-Feuchte-Skala (0-1500) - siehe
        # _rescale_precip_moisture_ranges() für die tatsächlich verwendeten,
        # auf dieses Projekt skalierten Werte.
        # moisture_capacity (0-100, gleiche Skala wie soil_moist_map): oberes
        # Limit, das die Bodenfeuchte für dieses Biom nie überschreitet (siehe
        # WaterSystemGenerator._calc_soil_moisture() - Biome-Preseed-Plan).
        # evaporation_factor (1.0=neutral): wie schnell dieses Biom bei Wärme
        # austrocknet, skaliert den Trocknungs-Term dort. Grobe Startwerte,
        # tendenziell (aber nicht streng) invers zueinander kalibriert - wer
        # wenig Wasser halten kann, verdunstet meist auch schneller,
        # empirisch nachjustierbar.
        # =====================================================================
        # DIE FUENFZEHN EUROPAEISCHEN GRUNDBIOME (2026-08-07)
        # =====================================================================
        #
        # Die alte Tabelle war weltweit gedacht und hatte fuenf Arten, die auf
        # einer 21-km-Insel in Europa nichts zu suchen haben (Ice Cap,
        # Tropical Rainforest, Tropical Seasonal, Savanna, Badlands), und drei
        # zu grobe (Grassland, Temperate Forest, Mediterranean) - damit sah das
        # halbe Festland gleich aus.
        #
        # `temp` ist die JULITEMPERATUR in Grad, `precip` der JAHRES-
        # niederschlag in mm. Beides liefert das Wettersystem seit dem
        # 2026-08-07 als Festlegung: die Temperatur trifft ihre Regionsziele
        # auf 1 K, der Niederschlag auf rund 5 %.
        #
        # Herleitung und Bezugsorte: docs/BIOME_MATRIX.md.
        self.biome_definitions = {
            0: {'name': 'hochmoor', 'temp': (10, 17), 'precip': (1000, 3000),
                'elevation': (0, 900), 'moisture': (800, 1500),
                'moisture_capacity': 100.0, 'evaporation_factor': 0.3},
            1: {'name': 'bruchwald', 'temp': (14, 22), 'precip': (800, 2000),
                'elevation': (0, 400), 'moisture': (700, 1400),
                'moisture_capacity': 95.0, 'evaporation_factor': 0.4},
            2: {'name': 'feuchtwiese', 'temp': (13, 22), 'precip': (700, 1600),
                'elevation': (0, 200), 'moisture': (600, 1200),
                'moisture_capacity': 85.0, 'evaporation_factor': 0.7},
            3: {'name': 'grasland', 'temp': (13, 22), 'precip': (500, 900),
                'elevation': (0, 1200), 'moisture': (300, 700),
                'moisture_capacity': 55.0, 'evaporation_factor': 1.0},
            4: {'name': 'heide', 'temp': (13, 19), 'precip': (500, 800),
                'elevation': (0, 600), 'moisture': (200, 500),
                'moisture_capacity': 40.0, 'evaporation_factor': 1.1},
            5: {'name': 'fjell', 'temp': (2, 12), 'precip': (400, 2500),
                'elevation': (300, 2000), 'moisture': (300, 1000),
                'moisture_capacity': 50.0, 'evaporation_factor': 0.5},
            6: {'name': 'nadelwald', 'temp': (12, 18), 'precip': (450, 900),
                'elevation': (50, 1400), 'moisture': (350, 800),
                'moisture_capacity': 65.0, 'evaporation_factor': 0.6},
            7: {'name': 'mischwald', 'temp': (15, 20), 'precip': (500, 900),
                'elevation': (0, 1000), 'moisture': (350, 800),
                'moisture_capacity': 70.0, 'evaporation_factor': 0.7},
            8: {'name': 'buchenwald', 'temp': (16, 21), 'precip': (600, 1000),
                'elevation': (0, 1200), 'moisture': (450, 900),
                'moisture_capacity': 80.0, 'evaporation_factor': 0.6},
            9: {'name': 'eichenwald', 'temp': (18, 23), 'precip': (500, 800),
                'elevation': (0, 800), 'moisture': (300, 700),
                'moisture_capacity': 70.0, 'evaporation_factor': 0.8},
            10: {'name': 'bergwald', 'temp': (10, 17), 'precip': (700, 2000),
                 'elevation': (500, 2200), 'moisture': (500, 1100),
                 'moisture_capacity': 75.0, 'evaporation_factor': 0.5},
            11: {'name': 'macchia', 'temp': (22, 30), 'precip': (350, 700),
                 'elevation': (0, 900), 'moisture': (150, 450),
                 'moisture_capacity': 40.0, 'evaporation_factor': 1.4},
            12: {'name': 'steineichenwald', 'temp': (21, 28), 'precip': (550, 900),
                 'elevation': (0, 1000), 'moisture': (300, 700),
                 'moisture_capacity': 55.0, 'evaporation_factor': 1.1},
            13: {'name': 'trockensteppe', 'temp': (22, 32), 'precip': (250, 500),
                 'elevation': (0, 1200), 'moisture': (100, 350),
                 'moisture_capacity': 30.0, 'evaporation_factor': 1.5},
            14: {'name': 'halbwueste', 'temp': (24, 36), 'precip': (0, 300),
                 'elevation': (0, 1500), 'moisture': (0, 200),
                 'moisture_capacity': 18.0, 'evaporation_factor': 1.8},
        }

        # =====================================================================
        # WELCHE BIOME ZU WELCHER REGION GEHOEREN
        # =====================================================================
        #
        # Der Nutzer wollte "eine matrix fuer die regionen. bei jeder moeglichen
        # temperatur und wassermenge gibt es dann ein biome fuer die region."
        #
        # Umgesetzt als AFFINITAET statt als neun getrennter Matrizen, und zwar
        # aus einem Grund: neun Matrizen mit argmax(Region) erzeugen eine harte
        # Biomkante entlang der Voronoi-Zellen. Ein Bonus auf die Eignung wird
        # dagegen ueber die Regionsgewichte weich ueberblendet - dieselbe
        # Bauform, die auch das Gelaende und das Klima benutzen.
        #
        # WARUM ES NOETIG IST: Morobora und Nebelrode liegen nur 2 K und 40 mm
        # auseinander, sollen aber Nadel- gegen Buchenwald sein. Ueber
        # Temperatur und Niederschlag allein sind sie nicht zu trennen.
        #
        # Die Liste je Region ist die Erwartung aus docs/BIOME_MATRIX.md
        # Abschnitt 3 - dieselbe, gegen die auch geprueft wird.
        self.regions_biome_affinitaet = {
            "Clonagh":         ("hochmoor", "grasland", "feuchtwiese",
                                   "heide", "bruchwald"),
            "Skerrheim":          ("hochmoor", "nadelwald", "bergwald", "fjell"),
            "Morobora":              ("nadelwald", "mischwald", "hochmoor",
                                   "grasland"),
            "Estrande":     ("feuchtwiese", "eichenwald", "grasland",
                                   "bruchwald"),
            "Nevadin":          ("bergwald", "fjell", "buchenwald", "grasland"),
            "Nebelrode":      ("buchenwald", "mischwald", "grasland",
                                   "bergwald"),
            "Samarcia":             ("trockensteppe", "halbwueste", "macchia",
                                   "steineichenwald"),
            "Macchia":         ("steineichenwald", "macchia", "grasland",
                                   "bruchwald"),
            "Thalassia": ("macchia", "trockensteppe", "steineichenwald"),
        }

        self._rescale_precip_moisture_ranges()

    def _rescale_precip_moisture_ranges(self):
        """
        Skaliert 'precip'- und 'moisture'-Bereiche von den klassischen
        Jahres-Whittaker-Werten auf die tatsächliche Größenordnung von
        precip_map/soil_moist_map in diesem Projekt herunter.

        Revision 2026-07-24 (nach Nutzer-Korrektur der Kalibrierung vom
        2026-07-23): precip_map ist KEIN Jahreswert mehr, sondern eine
        Akkumulation über eine simulierte saisonale Periode, kalibriert auf
        ~50mm als typischen Maximalwert (PRECIP_ANNUAL_SCALE_FACTOR=0.5 in
        core/weather_generator.py, seltene Ausreißer erlaubt) - die
        precip-Bereiche unten müssen deshalb WIEDER (wie vor der
        Zwischenrevision) auf diese kleine Größenordnung herunterskaliert
        werden, sonst läge jeder real erreichbare precip_map-Wert weit
        unterhalb des Minimums fast aller Biome-Bereiche.
        """
        # 2026-08-07: precip_map ist seit dem festgelegten Niederschlag eine
        # TICKSUMME in echten Millimetern. Die Biomtabelle steht dagegen in
        # JAHRES-Millimetern (Hochmoor 1000-3000).
        #
        # Der Faktor haengt damit an der Ticklaenge und wird von dort geholt -
        # er darf NICHT als Zahl hier stehen. Sonst zeigte eine Umstellung auf
        # Monatsticks stillschweigend jedes Biom um Faktor zwei verschoben.
        #
        # Die frueher hier stehende 50/4000-Kruecke stammte aus der Zeit, als
        # precip_map eine Simulationsakkumulation ohne physikalische Einheit war
        # und auf "~50 als typischer Hoechstwert" geeicht wurde. Mit einer
        # Festlegung in Millimetern entfaellt diese Eichung.
        from core.weather_generator import MONATE_JE_TICK
        precip_scale = MONATE_JE_TICK / 12.0
        moisture_scale = 100.0 / 1500.0

        for biome_def in self.biome_definitions.values():
            p_min, p_max = biome_def['precip']
            biome_def['precip'] = (p_min * precip_scale, p_max * precip_scale)

            m_min, m_max = biome_def['moisture']
            biome_def['moisture'] = (m_min * moisture_scale, m_max * moisture_scale)

    def _affinitaetsbonus(self, form, region_map):
        """
        Ein Eignungsbonus je Biom, aus der Regionszugehoerigkeit.

        Rueckgabe: (H, W, 15) oder None, wenn keine Regionskarte vorliegt.

        WEICH, NICHT HART. Der Bonus wird als Feld gebildet und dann geglaettet
        - genau wie die Klimafelder. Neun getrennte Matrizen mit argmax(Region)
        haetten eine harte Biomkante entlang der Voronoi-Zellen erzeugt.
        """
        if region_map is None:
            return None
        from scipy import ndimage
        from core.terrain_weltkarte import alle_regionen

        R = np.asarray(region_map)
        if R.shape != form[:2]:
            from scipy.ndimage import zoom
            R = np.round(zoom(R.astype(np.float32),
                              form[0] / R.shape[0], order=0)).astype(np.int16)

        namen = {d["name"]: i for i, d in self.biome_definitions.items()}
        bonus = np.zeros(form[:2] + (15,), dtype=np.float32)
        for i, (_z, _s, r) in enumerate(alle_regionen()):
            g = R == i
            if not g.any():
                continue
            for biom in self.regions_biome_affinitaet.get(r["name"], ()):
                if biom in namen:
                    bonus[g, namen[biom]] = AFFINITAETSBONUS

        sigma = max(form[0] / 40.0, 1.0)
        for k in range(15):
            bonus[:, :, k] = ndimage.gaussian_filter(bonus[:, :, k], sigma)
        return bonus

    def _eignungsfelder(self, heightmap, temp_map, precip_map, soil_moist_map,
                        region_map=None):
        """
        Die rohen Eignungswerte aller 15 Basisbiome je Pixel: (H, W, 15).

        Frueher stand diese Rechnung direkt in classify_base_biomes() und ihr
        Ergebnis ueberlebte den abschliessenden argmax nicht - 14 von 15
        Werten waren nach dem Aufruf verloren. Fuer den Godot-Export (Punkt b:
        "23 % Macchia, 21 % Grasland, ...") werden sie aber gebraucht, und
        zwar ohne die Rechnung ein zweites Mal zu fahren. Darum jetzt eine
        eigene Methode mit zwei Abnehmern.
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

        # Der Regionsbonus - er entscheidet dort, wo das Klima allein nicht
        # trennt (Morobora gegen Nebelrode: 2 K und 40 mm auseinander).
        bonus = self._affinitaetsbonus((height, width), region_map)
        if bonus is not None:
            fitness_maps = fitness_maps + bonus

        return fitness_maps

    def classify_base_biomes(self, heightmap, temp_map, precip_map, soil_moist_map,
                             region_map=None):
        """
        Klassifiziert Base-Biomes mit wissenschaftlich fundierter Multi-Factor-Analysis

        `region_map` ist seit 2026-08-07 dazugekommen: sie traegt die
        Regionsaffinitaet (siehe _affinitaetsbonus). Ohne sie klassifiziert die
        Methode wie zuvor, nur nach Klima.
        """
        fitness_maps = self._eignungsfelder(
            heightmap, temp_map, precip_map, soil_moist_map, region_map=region_map)
        # Dominantes Biome pro Pixel
        dominant_biomes = np.argmax(fitness_maps, axis=2)
        return dominant_biomes.astype(np.uint8)

    def klassifiziere_mit_eignungsfeld(self, heightmap, temp_map, precip_map,
                                       soil_moist_map, region_map=None):
        """
        Wie classify_base_biomes(), liefert aber zusaetzlich das Eignungsfeld.

        Return: (dominant, top3_ids, top3_anteil, eindeutigkeit)

        * `dominant`      (H, W) uint8  - genau das, was classify_base_biomes()
                                          auch liefert; der Aufruf ist ein
                                          vollwertiger Ersatz.
        * `top3_ids`      (H, W, 3) uint8   - die Kennungen der drei am besten
                                          passenden Basisbiome, absteigend
                                          sortiert. Kanal 0 ist der Gewinner
                                          und damit identisch mit `dominant`.
        * `top3_anteil`   (H, W, 3) float32 - ihre Anteile, auf Summe 1
                                          normalisiert. Das ist die
                                          "23 % / 21 % / 12 %"-Angabe.
        * `eindeutigkeit` (H, W) float32 - die ROHE Summe aller 15
                                          Eignungswerte, VOR der Normalisierung.

        Warum die Eindeutigkeit mitkommt: das Normalisieren auf 100 % loescht
        eine echte Information. An einem alpinen Ort sind die Eignungen hoch
        und eng (Fjell 48 %, Bergwald 38 %), in einem mitteleuropaeischen
        Uebergangsraum flach und breit (Spitzenreiter 14 %, vierzehn Biome
        ueber 2 %). Nach der Normalisierung sehen beide gleich aus. Die rohe
        Summe schwankt zwischen diesen Faellen um den Faktor 6 und ist damit
        ein brauchbares Mass dafuer, wie eindeutig ein Ort klimatisch ist -
        spaeter das Stellrad dafuer, wie stark eine Biomgrenze ausfranst.

        SCHAERFUNG: die Eignungen werden vor dem Normalisieren mit
        EIGNUNG_SCHAERFUNG potenziert. Ungeschaerft deckten die besten drei
        nur 38-48 % ab, der Rest verteilte sich auf ein Dutzend Biome - eine
        Bodentextur daraus waere ueberall dieselbe Graubraunmischung. Die
        Potenz aendert die Reihenfolge nicht, nur den Abstand: bei Exponent 3
        decken die besten drei 73-100 % ab. Der Exponent ist ein Regler fuer
        die Weichheit der Biomgrenzen, kein Fehlerkorrektur - bei 1 laufen
        alle Biome ineinander, bei 6 ist man praktisch wieder beim argmax.
        """
        fitness_maps = self._eignungsfelder(
            heightmap, temp_map, precip_map, soil_moist_map, region_map=region_map)

        dominant = np.argmax(fitness_maps, axis=2).astype(np.uint8)
        eindeutigkeit = fitness_maps.sum(axis=2).astype(np.float32)

        gewicht = np.maximum(fitness_maps, 0.0) ** EIGNUNG_SCHAERFUNG

        # argpartition statt argsort: es genuegt, die besten EIGNUNG_ANZAHL
        # nach vorn zu holen: bei 1024x1024x15 ist das der Unterschied
        # zwischen einmal und viermal durch den Speicher.
        vorne = np.argpartition(-gewicht, EIGNUNG_ANZAHL - 1,
                                axis=2)[:, :, :EIGNUNG_ANZAHL]
        werte = np.take_along_axis(gewicht, vorne, axis=2)
        reihenfolge = np.argsort(-werte, axis=2)
        top3_ids = np.take_along_axis(vorne, reihenfolge, axis=2).astype(np.uint8)
        top3_werte = np.take_along_axis(werte, reihenfolge, axis=2)

        summe = top3_werte.sum(axis=2, keepdims=True)
        top3_anteil = np.divide(top3_werte, summe,
                                out=np.zeros_like(top3_werte),
                                where=summe > 0).astype(np.float32)

        return dominant, top3_ids, top3_anteil, eindeutigkeit

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


# Wie stark die Regionszugehoerigkeit die Biomwahl beeinflusst.
#
# Die Klimaeignung liegt bei 0..1; 0.35 heisst also, dass ein regionstypisches
# Biom rund ein Drittel Vorsprung bekommt. Genug, um Morobora von Nebelrode zu
# trennen (2 K und 40 mm auseinander), zu wenig, um das Klima zu ueberstimmen -
# ein Gipfel im Nevadin wird trotzdem Fjell und nicht Buchenwald.
# Die Baumgrenze und die Firngrenze als JULITEMPERATUR, nicht als Hoehe.
#
# In Norwegen auf 60 Grad Nord liegt die Baumgrenze bei rund 900 bis 1100 m,
# und dort herrscht im Juli die 10-Grad-Isotherme - daher der Wert. Die
# Firngrenze liegt dort, wo auch im Hochsommer nichts mehr abtaut.
#
# STAND 2026-08-07: auf dem DAMALIGEN Gelaende/Klimastand loeste keine der
# beiden Regeln aus (kaelteste Julitemperatur an Land 10.4 Grad). Diese
# Aussage ist SEIT DER TEMPERATUR-DIREKTNORMIERUNG (1.10/1.11 in docs/
# OFFENE_PUNKTE.md, dieselbe Session) UEBERHOLT - nachgemessen 2026-08-11:
# Julitemperatur an Land reicht inzwischen bis -12 Grad, `alpine_level`
# realisiert sich auf ~8-9% der Landflaeche, `snow_level` auf ~1%, beide mit
# plausibler Korrelation zu Hoehe/Kaelte. KEIN Regler hier wurde dafuer
# geaendert - die Klimakalibrierung an anderer Stelle hat das nebenbei
# geloest. Vor einer erneuten Schwellenaenderung hier immer erst mit dem
# AKTUELLEN Klimastand nachmessen (docs/OFFENE_PUNKTE.md 2.7).
BAUMGRENZE_JULI_C = 10.0
FIRN_JULI_C = 0.0

AFFINITAETSBONUS = 0.35

# --- Eignungsfeld fuer den Godot-Export ------------------------------------
#
# Wie viele Biome je Ort ausgegeben werden und wie stark ihr Abstand vorher
# gespreizt wird. Beide Zahlen sind Regler, keine Naturkonstanten - die
# Begruendung steht ausfuehrlich bei klassifiziere_mit_eignungsfeld().
#
# Die 3 haengt an Terrain3D: seine Control-Textur haelt je Pixel zwei
# Textur-Kennungen, und weil vier benachbarte Texel bilinear gemischt werden,
# kommen im gerenderten Bild bis zu acht zusammen. Drei je Ort sind damit
# darstellbar, ohne dass die Mischung zu Matsch wird.
EIGNUNG_ANZAHL = 3
EIGNUNG_SCHAERFUNG = 3.0


class SuperBiomeOverrideSystem:
    """
    Priority-basiertes Override-System mit 11 speziellen Biom-Bedingungen
    """

    def __init__(self, sea_level=10.0, bank_width=3.0, edge_softness=1.0,
                 alpine_level=1500.0, snow_level=2000.0, cliff_slope=60.0,
                 shader_manager=None, data_lod_manager=None):
        """
        Initialisiert Super-Biome-Override-System mit Unified-Edge-Softness-Control
        """
        self.sea_level = sea_level
        self.bank_width = bank_width
        self.edge_softness = edge_softness
        self.alpine_level = alpine_level
        self.snow_level = snow_level
        self.cliff_slope = cliff_slope
        self.shader_manager = shader_manager
        # Für _calculate_cliff_probabilities()'s live map_distance_km-Wert
        # (siehe [[project-terrain-review]] 4f) - optional, da diese Klasse
        # auch standalone/in Tests ohne echten Manager instanziiert wird.
        self.data_lod_manager = data_lod_manager

        # Super-Biome-Offset (nach 15 Base-Biomes)
        self.super_biome_offset = 15

    def apply_super_biome_overrides(self, heightmap, temp_map, water_biomes_map, soil_moist_map,
                                    see_eis=None):
        """
        Wendet alle Super-Biome-Overrides in Priority-Reihenfolge an

        `see_eis` (optional, (H,W) bool, core.terrain_weltkarte.seegliederung(),
        docs/OFFENE_PUNKTE.md 3.6) - Seeeis vor der Morobora-Kueste, NACH der
        Ocean-Zuweisung angewandt (ueberschreibt "Ocean" dort, wo Eis ist).
        None (alter Nicht-Weltkarten-Pfad) laesst Ocean unveraendert.
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

        # ABGESCHLOSSENE BECKEN UNTER DEM MEERESSPIEGEL (2026-08-25).
        #
        # Nutzerbefund mit Bild: *"hier ist die kontur vom 0 m level zu sehen
        # aber das basin ist mit biomen. eigentlich muesste das in der karte
        # blau dargestellt werden (und in 3D)."*
        #
        # URSACHE: `_detect_ocean_connectivity()` ist eine FLUTFUELLUNG VOM
        # KARTENRAND. Eine Senke unter 0 m, die keine Verbindung zum offenen
        # Meer hat, wird davon nie erreicht - und wenn die Wassersimulation
        # sie auch nicht als See fuellt (`water_biomes_map == 4`), faellt sie
        # durch beide Raster und wird nach Hoehe/Temperatur/Niederschlag als
        # LANDBIOM klassifiziert. Gemessen am 2026-08-25 (128 px, Seed
        # 20260804): 83 Pixel in 23 Becken, groesstes 33 Pixel, Tiefen bis
        # -73.5 m - mitten im Bild, mit Wald oder Samarcia eingefaerbt.
        #
        # ZUWEISUNG ALS SEE, NICHT ALS OZEAN, und das ist Absicht: diese
        # Becken sind per Definition NICHT mit dem Meer verbunden. Sie als
        # Ozean zu fuehren waere eine falsche Aussage ueber die
        # Erreichbarkeit - Seewege, Kuestenlogik und die Seegliederung lesen
        # `ocean`.
        #
        # HIER und nicht in der Anzeige, damit 2D und 3D dieselbe Karte
        # bekommen (stehende Regel in CLAUDE.md).
        #
        # WAS DAS BEWUSST NICHT ABBILDET: eine echte Trockensenke unter dem
        # Meeresspiegel (Totes Meer, Death Valley) gibt es in diesem Modell
        # damit nicht mehr. Das ist der Preis fuer die Eindeutigkeit "unter 0
        # ist Wasser", auf der der ganze Rest des Projekts aufbaut
        # (`land = H > 0` steht an Dutzenden Stellen).
        unterwasser = heightmap < self.sea_level
        ohne_zuweisung = super_biome_mask == 0
        abgeschlossen = unterwasser & ohne_zuweisung
        if abgeschlossen.any():
            super_biome_mask[abgeschlossen] = self.super_biome_offset + 1  # Lake
            logging.getLogger(__name__).info(
                "%d Pixel unter dem Meeresspiegel ohne Anbindung ans Meer "
                "als See eingestuft (abgeschlossene Becken)",
                int(abgeschlossen.sum()))

        # SEEEIS (2026-08-11) - eigene, sichtbare Kategorie statt eines
        # unbenutzten Datenfelds. Nutzer: "wie willst du das meereis
        # shadern, wie willst du es sonst darstellen?" Harte Zuweisung wie
        # bei Lake/River/Ocean daneben (kein Wahrscheinlichkeitsfeld noetig -
        # `see_eis` ist bereits eine diskrete Ja/Nein-Entscheidung je
        # Seezelle), nur auf tatsaechlichem Meer (see_eis ist zwar schon auf
        # Land False, aber ocean_mask kann an Fluss-/Seemuendungen von
        # water_biomes_map abweichen - beide zusammen sind sicherer als
        # eines allein).
        if see_eis is not None and see_eis.shape == (height, width):
            super_biome_mask[ocean_mask & see_eis] = self.super_biome_offset + 11  # Sea Ice

        # Priority 5-6: Topographie-basierte Super-Biomes
        cliff_probabilities = self._calculate_cliff_probabilities(heightmap)
        super_biome_probabilities['cliff'] = cliff_probabilities

        beach_probabilities = self._calculate_beach_probabilities(heightmap, ocean_mask)
        super_biome_probabilities['beach'] = beach_probabilities

        # Priority 7-8: Proximity-basierte Super-Biomes
        #
        # UFER GIBT ES NUR AN LAND (2026-08-25).
        #
        # `_calculate_lake_edge_probabilities()` und
        # `_calculate_river_bank_probabilities()` messen den ABSTAND zu Seen
        # bzw. Fluessen und bekommen die Heightmap gar nicht zu sehen. Ihr
        # Wahrscheinlichkeitsfeld reicht deshalb in BEIDE Richtungen - auch
        # ins Wasser hinein. Im Supersampling wurden daraus echte Pixel:
        # gemessen 3142 `river_bank` und 467 `lake_edge` UNTER dem
        # Meeresspiegel, zusammen 8 % aller Unterwasserpixel. Ein Flussufer
        # mitten auf dem Meer.
        #
        # In `biome_map` fiel das nicht auf, weil dort die harten
        # Wasserzuweisungen (Ocean/Lake/River) gewinnen - erst das
        # Supersampling setzt die Wahrscheinlichkeiten in Pixel um und
        # ueberschreibt damit auch Wasser. Dieselbe Klasse wie der
        # Snow-/Alpine-Fall, der weiter oben schon einmal auffiel.
        #
        # `beach` ist mitmaskiert, aber mit Spielraum: ein Strand DARF knapp
        # unter der Nulllinie liegen (die Bedingung dort ist
        # `h <= sea_level + 5`), nur nicht im tiefen Wasser.
        an_land = heightmap >= self.sea_level

        lake_edge_probabilities = self._calculate_lake_edge_probabilities(water_biomes_map)
        super_biome_probabilities['lake_edge'] = lake_edge_probabilities * an_land

        river_bank_probabilities = self._calculate_river_bank_probabilities(water_biomes_map)
        super_biome_probabilities['river_bank'] = river_bank_probabilities * an_land

        # Strand: bis 5 m unter der Nulllinie erlaubt, tiefer nicht.
        super_biome_probabilities['beach'] = (
            super_biome_probabilities['beach']
            * (heightmap >= self.sea_level - 5.0))

        # Priority 9-10: Höhen-basierte Super-Biomes
        snow_probabilities = self._calculate_snow_level_probabilities(heightmap, temp_map)
        alpine_probabilities = self._calculate_alpine_level_probabilities(heightmap, temp_map)

        # NUR AUF LAND (2026-08-11, Nutzer-Befund am laufenden Programm:
        # "vereinzelte weisse Punkte im Meer"). Beide Wahrscheinlichkeiten
        # haengen NUR an der Julitemperatur (siehe dortige Docstrings,
        # 2.2-Umbau) - ohne Landfilter erfuellt jedes hinreichend KALTE
        # Meerespixel (auf der Weltkarte z.B. See-Nord vor der Morobora, siehe
        # SEE_MITTEL_NORD in weather_generator.py) rein rechnerisch dieselbe
        # Bedingung wie ein Gipfel. `_apply_supersampling_cpu()` ueberschreibt
        # dann STOCHASTISCH einzelne Sub-Pixel des schon korrekt als "Ocean"
        # erkannten `ocean_mask` mit "Snow Level"/"Alpine Level" - genau die
        # vereinzelten weissen/grauen Punkte im offenen Meer statt eines
        # zusammenhaengenden Seeeis-Bilds.
        snow_probabilities = np.where(ocean_mask, 0.0, snow_probabilities)
        alpine_probabilities = np.where(ocean_mask, 0.0, alpine_probabilities)
        super_biome_probabilities['snow_level'] = snow_probabilities
        super_biome_probabilities['alpine_level'] = alpine_probabilities

        return super_biome_mask, super_biome_probabilities

    def _detect_ocean_connectivity(self, heightmap, water_biomes_map):
        """
        Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind -
        3-stufiges Fallback-System
        """
        # Potentielle Ocean-Bereiche (unter sea_level) - Kurzschluss vorab, damit
        # weder GPU noch CPU-Pfad bei einer garantiert leeren Maske aufgerufen wird
        if not np.any(heightmap < self.sea_level):
            return np.zeros(heightmap.shape, dtype=bool)

        # GPU-Pfad bewusst DEAKTIVIERT (shaders/biome/oceanConnectivity{Seed,
        # Propagate}.comp existieren, sind korrekt/verifiziert - siehe Session-
        # Analyse): eine echte, korrekte Wellenausbreitung (nicht Jump-Flooding, das
        # bei verwinkelten Küstenlinien falsch durch "Wände" springen könnte) braucht
        # pro Aufruf 4*(width+height) sequenzielle GPU-Passes, um bei stark
        # gewundenen Küstenlinien Konvergenz zu garantieren. Gemessen: bei 256px
        # ~390ms GPU vs. ~50ms CPU-BFS - die Python-deque-BFS besucht jeden
        # erreichbaren Pixel genau einmal und ist hier schlicht die schnellere
        # Lösung. GPU-Pfad bleibt aus, bis ein Abbruchkriterium (frühzeitiges Ende
        # bei Konvergenz) implementiert ist.
        if False and self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "biome", "oceanConnectivity", {"heightmap": heightmap, "sea_level": self.sea_level}, {})
                if result.get("success"):
                    return result["ocean_mask"]
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"GPU ocean connectivity failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        return self._detect_ocean_connectivity_cpu(heightmap)

    def _detect_ocean_connectivity_cpu(self, heightmap):
        """CPU-Implementierung: Flood-Fill von lokalen Minima die mit Kartenrand verbunden sind"""
        height, width = heightmap.shape
        ocean_mask = np.zeros((height, width), dtype=bool)

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

        ============================================================
        WARUM AN DER KUESTE FAST KEINE KLIPPEN ENTSTEHEN
        (Nutzerfrage 2026-08-25: *"ich sehe die cliffs nicht so ganz.
        wird die slopemap vor oder nach den vektoren berechnet?"*)
        ============================================================

        ERST DIE REIHENFOLGE, weil sie der naheliegende Verdacht war und
        NICHT die Ursache ist:

          terrain.redistribution  ruft weltfeld(), und DORT wird die
                                  Vektorkueste angewandt (VEKTOR_KUESTE_AKTIV
                                  in core/terrain_weltkarte.py)
          erosion.slope           rechnet auf der KOMBINIERTEN Heightmap
          diese Funktion          bekommt ebenfalls die kombinierte Heightmap
                                  (get_calculator_combined_heightmap, siehe
                                  _get_prepared_biome_inputs)

        Die Vektorkueste steckt also in jeder Heightmap, die hier ankommt.
        Nachgemessen am 2026-08-25: die kombinierte Heightmap war sogar
        BITGLEICH mit terrain.redistribution/heightmap (Erosion ist per
        EROSION_AKTIV abgeschaltet). **Die Reihenfolge ist richtig.**

        Diese Funktion benutzt ausserdem gar nicht die `slopemap`, sondern
        rechnet ihren eigenen Gradienten - eine Fehlerquelle weniger.

        DIE ECHTE URSACHE sind zwei Dinge, die zusammenkommen:

        1. DAS PROFIL IST FEINER ALS DAS RASTER. Die gemessenen Kuesten-
           profile sind nur auf ihren ersten 50-100 m steil:

               Moher-Klippen       0-50 m  48.9 Grad
               Amalfi-Steilkueste  0-50 m  27.4 Grad
               Santorini-Kliff     0-50 m  26.5 Grad
               Algarve-Klippen     0-50 m  24.9 Grad
               Luce-Bay-Straende   0-50 m   4.5 Grad

           Und 50 m sind:  0.30 px bei 128 px Karte,  0.60 px bei 256,
           1.20 px bei 512,  2.40 px bei 1024,  4.81 px bei 2048.
           **Unterhalb von etwa 1024 px ist die Klippe schmaler als ein
           Pixel** und kann im Raster gar nicht steil werden. Dieselbe
           Grenze wie in docs/OFFENE_PUNKTE.md 6.19.

        2. DIE MEISTEN ARCHETYPEN SIND GAR NICHT SO STEIL. Von den fuenf
           oben ueberschreitet nur Moher 45 Grad. Selbst bei perfekter
           Aufloesung waeren die uebrigen keine "Klippe" im Sinne dieser
           Schwelle.

        GEMESSEN, an der Wasserlinie (< 200 m) gegen Inland (> 1 km):

               256 px:  Kueste p90 26.2 Grad, >45 Grad 0.16 %  |  Inland 2.80 %
               512 px:  Kueste p90 29.3 Grad, >45 Grad 0.73 %  |  Inland 6.63 %
              1024 px:  Kueste p90 29.7 Grad, >45 Grad 1.93 %  |  Inland 11.43 %

           **Die Kueste ist der FLACHSTE Teil der Karte**, gemessen an der
           Dichte steiler Pixel - nicht der steilste. Der p90 bleibt bei
           allen drei Groessen zwischen 26 und 30 Grad.

        WAS DARAUS FOLGT, falls jemand spaeter wieder hier landet:

          * Klippen sind bei diesem Massstab ein GEBIRGS-Merkmal, kein
            Kuestenmerkmal. Das ist keine Fehlfunktion.
          * Wer Kuestenklippen sehen will, muss `cliff_slope` auf etwa
            30 Grad senken - dort liegt der gemessene p90 der Kueste.
            Bei 45 Grad (Vorgabe seit 2026-08-25) trifft man sie nicht.
          * Eine feinere Karte hilft, aber langsam: von 256 auf 1024 px
            steigt der Anteil steiler Kuestenpixel nur von 0.16 auf 1.93 %.
          * Am Raster zu drehen bringt hier weniger als am Profil. Wer
            steilere Kuesten WILL, muss die Vorlagen aendern - und die sind
            aus echten DEMs gemessen (core/vektor_kueste.py).
        """
        # Gradient berechnen - ohne spacing rechnet np.gradient() mit 1 Pixel = 1m
        # Horizontal-Abstand, obwohl ein Pixel real ~50-300m abdeckt (siehe
        # core/terrain_generator.py SlopeCalculator für denselben Bug an anderer
        # Stelle) - das ließ praktisch die gesamte Karte als Steilhang (>80°)
        # erscheinen, unabhängig vom cliff_slope-Parameter. map_distance_km kommt
        # live von Terrains "Map Distance"-Slider statt der vorherigen statischen
        # TERRAIN.WORLD_SIZE_KM-Konstante (siehe [[project-terrain-review]] 4f).
        if self.data_lod_manager is not None:
            map_distance_km = self.data_lod_manager.get_map_distance_km()
        else:
            from gui.config.value_default import TERRAIN
            map_distance_km = TERRAIN.WORLD_SIZE_KM
        spacing = (map_distance_km * 1000.0) / heightmap.shape[0]
        grad_y, grad_x = np.gradient(heightmap, spacing)
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
        Firn: wo die Julitemperatur unter FIRN_JULI_C faellt.

        TEMPERATURREGEL STATT HOEHENREGEL (2026-08-07). Die alte Fassung
        rechnete `h > snow_level + 500*(1 + T/10)` - bei 15 Grad also ueber
        2750 m. Der hoechste Punkt dieser Welt liegt bei 728 m; die Regel loeste
        nie aus, und niemand konnte das sehen, weil sie stillschweigend 0
        lieferte.

        Die Temperaturregel ist ausserdem die richtige Bauform: eine
        Schneegrenze ist keine Hoehenlinie. Sie folgt Hangausrichtung,
        Schattenwurf und Regionsklima - und wird damit von selbst
        unregelmaessig, genau wie der Nutzer es wollte ("mit varianz
        entsprechend der temperatur, damit es keine gerade linie ist").
        """
        return np.maximum(0.0, self._sigmoid(
            (FIRN_JULI_C - temp_map) / max(1.0 * self.edge_softness, 0.1)))

    def _calculate_alpine_level_probabilities(self, heightmap, temp_map):
        """
        Alpin: oberhalb der Baumgrenze, also wo die Julitemperatur unter
        BAUMGRENZE_JULI_C faellt. Begruendung siehe
        _calculate_snow_level_probabilities.
        """
        return np.maximum(0.0, self._sigmoid(
            (BAUMGRENZE_JULI_C - temp_map) / max(1.0 * self.edge_softness, 0.1)))

    def _alt_alpine_level_probabilities(self, heightmap, temp_map):
        """Die alte Hoehenregel - steht nur noch zum Vergleich hier."""
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

    def __init__(self, biome_seed=42, supersampling_quality=1.0, shader_manager=None):
        """
        Initialisiert Supersampling-Manager mit reproduzierbarer Randomization
        """
        self.biome_seed = biome_seed
        self.supersampling_quality = supersampling_quality
        self.shader_manager = shader_manager

    _PROBABILITY_KEYS = ("cliff", "beach", "lake_edge", "river_bank", "snow_level", "alpine_level")

    def apply_supersampling(self, biome_map, super_biome_probabilities):
        """
        Wendet 2x2-Supersampling mit diskretisierter Rotation an - 3-stufiges
        Fallback-System (GPU nur, wenn super_biome_probabilities exakt die von
        apply_super_biome_overrides() gelieferten 6 Keys in bekannter Reihenfolge hat).
        """
        if self.shader_manager and super_biome_probabilities and \
                tuple(super_biome_probabilities.keys()) == self._PROBABILITY_KEYS:
            try:
                result = self.shader_manager.request_shader_operation(
                    "biome", "supersampling",
                    {
                        "biome_map": biome_map,
                        "cliff_prob": super_biome_probabilities["cliff"],
                        "beach_prob": super_biome_probabilities["beach"],
                        "lake_edge_prob": super_biome_probabilities["lake_edge"],
                        "river_bank_prob": super_biome_probabilities["river_bank"],
                        "snow_prob": super_biome_probabilities["snow_level"],
                        "alpine_prob": super_biome_probabilities["alpine_level"],
                        "biome_seed": self.biome_seed,
                        "supersampling_quality": self.supersampling_quality,
                    },
                    {}
                )
                if result.get("success"):
                    return result["biome_map_super"]
            except Exception as e:
                logging.getLogger(__name__).warning(f"GPU supersampling failed: {e}, falling back to CPU")

        return self._apply_supersampling_cpu(biome_map, super_biome_probabilities)

    @staticmethod
    def _sub_zufall(seed, x, y, i):
        """
        Ein Wert in [0,1) je Teilpixel, der WIRKLICH von Ort und Ecke abhängt.

        HIER SASSEN DIE DIAGONALEN BÄNDER (Nutzerbild vom 2026-08-10).
        Die Vorlage rechnete:

            sub_seed = (seed + 54321 + x * 4 + y * 4 + i * 3571) % 1000

        `x * 4 + y * 4` ist `4·(x+y)`. Der "Zufallswert" hing damit nur von der
        SUMME der Koordinaten ab - jede Diagonale bekam denselben Wert, und
        wegen `% 1000` wiederholte sich das alle 250 Diagonalen. Wo eine
        Super-Biom-Wahrscheinlichkeit über null lag, kippte deshalb nicht ein
        gestreutes Muster, sondern ein ganzer diagonaler Streifen - quer über
        Land UND offenes Meer.

        Ersetzt durch eine Durchmischung mit drei verschiedenen großen
        Primzahlen je Achse. Zwei benachbarte Punkte, egal in welcher Richtung,
        bekommen damit unabhängige Werte.
        """
        h = (np.uint64(seed & 0xFFFFFFFF) * np.uint64(2654435761)
             + x.astype(np.uint64) * np.uint64(73856093)
             + y.astype(np.uint64) * np.uint64(19349663)
             + np.uint64(int(i) * 83492791))
        h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
        return ((h >> np.uint64(11)) & np.uint64(0xFFFFF)).astype(np.float64) / float(1 << 20)

    def _apply_supersampling_cpu(self, biome_map, super_biome_probabilities):
        """
        CPU-Implementierung des 2x2-Supersampling mit diskretisierter Rotation.

        VOLLSTÄNDIG VEKTORISIERT (2026-08-10). Die Vorlage war eine doppelte
        Python-Schleife über jedes Pixel, innen noch einmal über vier Teilpixel
        und sechs Wahrscheinlichkeitskarten - bei 1024 px sind das über 25
        Millionen Durchläufe im Interpreter. Dieselbe Rechnung als Feldoperation
        ist um Größenordnungen billiger und Zeile für Zeile dieselbe Formel,
        abgesehen von `_sub_zufall` (siehe dort, das war ein Fehler).
        """
        height, width = biome_map.shape
        biome_map_super = np.zeros((height * 2, width * 2), dtype=np.uint8)
        yy, xx = np.mgrid[0:height, 0:width]

        # Diskretisierte Rotations-Zuweisung mit Primzahlen - unverändert.
        rotation = (self.biome_seed + 12345 + xx * 997 + yy * 991) % 4
        anordnung = {
            0: [(0, 0), (0, 1), (1, 0), (1, 1)],   # TL, TR, BL, BR
            1: [(1, 0), (0, 0), (1, 1), (0, 1)],   # BL, TL, BR, TR
            2: [(1, 1), (1, 0), (0, 1), (0, 0)],   # BR, BL, TR, TL
            3: [(0, 1), (1, 1), (0, 0), (1, 0)],   # TR, BR, TL, BL
        }

        namen = list(super_biome_probabilities.keys()) if super_biome_probabilities else []
        for i in range(4):
            wert = self._sub_zufall(self.biome_seed + 54321, xx, yy, i)
            zugewiesen = biome_map.astype(np.uint8).copy()
            # Rückwärts, damit der ERSTE Treffer gewinnt - wie das `break` in
            # der Schleifenfassung.
            for name in reversed(namen):
                karte = super_biome_probabilities[name]
                if karte.shape[:2] != (height, width):
                    continue
                trifft = wert < karte * self.supersampling_quality
                zugewiesen[trifft] = self._get_super_biome_index(name)

            # Die Ecke, die dieses i bei der jeweiligen Rotation belegt.
            for r, ecken in anordnung.items():
                sub_y, sub_x = ecken[i]
                gilt = rotation == r
                ziel_y = (yy[gilt] * 2 + sub_y)
                ziel_x = (xx[gilt] * 2 + sub_x)
                biome_map_super[ziel_y, ziel_x] = zugewiesen[gilt]

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
            'alpine_level': super_biome_offset + 10,
            'sea_ice': super_biome_offset + 11,
        }

        return super_biome_mapping.get(super_biome_name, 0)