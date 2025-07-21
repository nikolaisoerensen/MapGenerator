"""
Path: core/__init__.py

Funktionsweise: Core-Module Initialisierung für Map Generator
Aufgabe: Stellt alle Generator-Klassen für GUI-Tabs zur Verfügung
Imports: Alle Haupt-Generator-Klassen aus den einzelnen Core-Modulen
"""

# Terrain Generation
from .terrain_generator import (
    BaseTerrainGenerator,
    SimplexNoiseGenerator,
    ShadowCalculator
)

# Geology Generation
from .geology_generator import (
    GeologyGenerator,
    RockTypeClassifier,
    MassConservationManager
)

# Weather Generation
from .weather_generator import (
    WeatherSystemGenerator,
    TemperatureCalculator,
    WindFieldSimulator,
    PrecipitationSystem,
    AtmosphericMoistureManager
)

# Water/Hydrology Generation
from .water_generator import (
    HydrologySystemGenerator,
    LakeDetectionSystem,
    FlowNetworkBuilder,
    ManningFlowCalculator,
    ErosionSedimentationSystem,
    SoilMoistureCalculator,
    EvaporationCalculator
)

# Biome Generation
from .biome_generator import (
    BiomeClassificationSystem,
    BaseBiomeClassifier,
    SuperBiomeOverrideSystem,
    SupersamplingManager,
    ProximityBiomeCalculator
)

# Settlement Generation
from .settlement_generator import (
    SettlementGenerator,
    TerrainSuitabilityAnalyzer,
    PathfindingSystem,
    CivilizationInfluenceMapper,
    PlotNodeSystem,
    Location,
    Plot,
    PlotNode
)

# Version Info
__version__ = "1.0.0"
__author__ = "Map Generator Team"

# Alle verfügbaren Generator-Klassen
__all__ = [
    # Terrain
    'BaseTerrainGenerator',
    'SimplexNoiseGenerator',
    'ShadowCalculator',

    # Geology
    'GeologyGenerator',
    'RockTypeClassifier',
    'MassConservationManager',

    # Weather
    'WeatherSystemGenerator',
    'TemperatureCalculator',
    'WindFieldSimulator',
    'PrecipitationSystem',
    'AtmosphericMoistureManager',

    # Water
    'HydrologySystemGenerator',
    'LakeDetectionSystem',
    'FlowNetworkBuilder',
    'ManningFlowCalculator',
    'ErosionSedimentationSystem',
    'SoilMoistureCalculator',
    'EvaporationCalculator',

    # Biome
    'BiomeClassificationSystem',
    'BaseBiomeClassifier',
    'SuperBiomeOverrideSystem',
    'SupersamplingManager',
    'ProximityBiomeCalculator',

    # Settlement
    'SettlementGenerator',
    'TerrainSuitabilityAnalyzer',
    'PathfindingSystem',
    'CivilizationInfluenceMapper',
    'PlotNode',
    'Location',
    'Plot'
]

# Convenience Funktionen für häufig verwendete Generator-Kombinationen
def get_basic_generators():
    """
    Funktionsweise: Gibt Standard-Generator-Set für einfache Map-Erstellung zurück
    Return: dict mit Basis-Generatoren
    """
    return {
        'terrain': BaseTerrainGenerator(),
        'geology': GeologyGenerator(),
        'weather': WeatherSystemGenerator(),
        'water': HydrologySystemGenerator(),
        'settlement': SettlementGenerator(),
        'biome': BiomeClassificationSystem()
    }

def get_all_generators():
    """
    Funktionsweise: Gibt alle verfügbaren Generatoren zurück
    Return: dict mit allen Generator-Klassen
    """
    return {
        'terrain': {
            'main': BaseTerrainGenerator(),
            'noise': SimplexNoiseGenerator(),
            'shadow': ShadowCalculator()
        },
        'geology': {
            'main': GeologyGenerator(),
            'classifier': RockTypeClassifier(),
            'conservation': MassConservationManager()
        },
        'settlement': {
            'main': SettlementGenerator(),
            'suitability': TerrainSuitabilityAnalyzer(),
            'pathfinding': PathfindingSystem(),
            'influence': CivilizationInfluenceMapper()
        },
        'weather': {
            'main': WeatherSystemGenerator(),
            'temperature': TemperatureCalculator(),
            'wind': WindFieldSimulator(),
            'precipitation': PrecipitationSystem(),
            'moisture': AtmosphericMoistureManager()
        },
        'water': {
            'main': HydrologySystemGenerator(),
            'lakes': LakeDetectionSystem(),
            'flow': FlowNetworkBuilder(),
            'manning': ManningFlowCalculator(),
            'erosion': ErosionSedimentationSystem(),
            'soil': SoilMoistureCalculator(),
            'evaporation': EvaporationCalculator()
        },
        'biome': {
            'main': BiomeClassificationSystem(),
            'base': BaseBiomeClassifier(),
            'super': SuperBiomeOverrideSystem(),
            'sampling': SupersamplingManager(),
            'proximity': ProximityBiomeCalculator()
        }
    }

# Hilfsfunktionen für Generator-Validation
def validate_generator_dependencies():
    """
    Funktionsweise: Validiert ob alle Generator-Dependencies verfügbar sind
    Return: bool - True wenn alle Dependencies OK
    """
    try:
        # Test-Imports für externe Dependencies
        import numpy as np
        from opensimplex import OpenSimplex
        from scipy.spatial import Delaunay
        from scipy.interpolate import splprep, splev
        from scipy.ndimage import gaussian_filter, distance_transform_edt

        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def get_generator_info():
    """
    Funktionsweise: Gibt Informationen über alle verfügbaren Generatoren zurück
    Return: dict mit Generator-Informationen
    """
    return {
        'terrain': {
            'description': 'Simplex-Noise basierte Terrain-Generierung mit Multi-Scale Layering',
            'outputs': ['heightmap', 'slopemap', 'shademap'],
            'dependencies': []
        },
        'geology': {
            'description': 'Geologische Schichten und Gesteinstypen mit Massenerhaltung',
            'outputs': ['rock_map', 'hardness_map'],
            'dependencies': ['heightmap', 'slopemap']
        },
        'settlement': {
            'description': 'Intelligente Settlement-Platzierung mit Civilization-Mapping',
            'outputs': ['settlement_list', 'landmark_list', 'roadsite_list', 'plot_map', 'civ_map'],
            'dependencies': ['heightmap', 'slopemap']
        },
        'weather': {
            'description': 'Dynamisches Wetter- und Feuchtigkeitssystem',
            'outputs': ['wind_map', 'temp_map', 'precip_map', 'humid_map'],
            'dependencies': ['heightmap', 'shade_map']
        },
        'water': {
            'description': 'Hydrologie-System mit Erosion und Sedimentation',
            'outputs': ['water_map', 'flow_map', 'flow_speed', 'cross_section', 'soil_moist_map',
                       'erosion_map', 'sedimentation_map', 'rock_map_updated', 'evaporation_map',
                       'ocean_outflow', 'water_biomes_map'],
            'dependencies': ['heightmap', 'slopemap', 'hardness_map', 'rock_map', 'precip_map',
                           'temp_map', 'wind_map', 'humid_map']
        },
        'biome': {
            'description': 'Biome-Klassifikation mit Base- und Super-Biomes',
            'outputs': ['biome_map', 'biome_map_super', 'super_biome_mask'],
            'dependencies': ['heightmap', 'slopemap', 'temp_map', 'soil_moist_map', 'water_biomes_map']
        }
    }