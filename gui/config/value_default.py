"""
Path: gui/config/value_default.py

Funktionsweise: Zentrale Parameter-Defaults für alle Slider und Controls
- Min/Max/Step/Default Werte für alle Generator-Parameter aktualisiert
- Neue Parameter für alle Core-Module integriert
- Organisiert nach Generator-Typen (TERRAIN, GEOLOGY, SETTLEMENT, WEATHER, WATER, BIOME)
- Validation-Rules und Parameter-Constraints
- Einheitliche Decimal-Precision und Suffix-Definitionen
"""


class TERRAIN:
    """Parameter für core/terrain_generator.py"""
    MAPSIZEMIN = 32
    MAPSIZEMAX = 1024

    MAPSIZE = {"min": MAPSIZEMIN, "max": MAPSIZEMAX, "default": 128, "step": 32}
    AMPLITUDE = {"min": 30, "max": 6000.0, "default": 2000.0, "step": 10, "suffix": "m"}
    OCTAVES = {"min": 1, "max": 12, "default": 8, "step": 1}
    FREQUENCY = {"min": 0.001, "max": 0.1, "default": 0.037, "step": 0.001}
    PERSISTENCE = {"min": 0.1, "max": 1.0, "default": 0.68, "step": 0.01}
    LACUNARITY = {"min": 1.1, "max": 4.0, "default": 2.3, "step": 0.1}
    REDISTRIBUTE_POWER = {"min": 0.5, "max": 4.0, "default": 2.5, "step": 0.1}
    MAP_SEED = {"min": 0, "max": 999999, "default": 542595, "step": 1}

class GEOLOGY:
    """Parameter für core/geology_generator.py"""
    SEDIMENTARY_HARDNESS = {"min": 1, "max": 100, "default": 30, "step": 1}
    IGNEOUS_HARDNESS = {"min": 1, "max": 100, "default": 80, "step": 1}
    METAMORPHIC_HARDNESS = {"min": 1, "max": 100, "default": 65, "step": 1}
    RIDGE_WARPING = {"min": 0.0, "max": 2.0, "default": 0.5, "step": 0.1}
    BEVEL_WARPING = {"min": 0.0, "max": 2.0, "default": 0.3, "step": 0.1}
    METAMORPH_FOLIATION = {"min": 0.0, "max": 1.0, "default": 0.4, "step": 0.1}
    METAMORPH_FOLDING = {"min": 0.0, "max": 1.0, "default": 0.6, "step": 0.1}
    IGNEOUS_FLOWING = {"min": 0.0, "max": 1.0, "default": 0.7, "step": 0.1}

class SETTLEMENT:
    """Parameter für core/settlement_generator.py"""
    SETTLEMENTS = {"min": 1, "max": 5, "default": 3, "step": 1}
    LANDMARKS = {"min": 0, "max": 6, "default": 3, "step": 1}
    ROADSITES = {"min": 0, "max": 6, "default": 3, "step": 1}
    PLOTNODES = {"min": 50, "max": 5000, "default": 1000, "step": 10}
    CIV_INFLUENCE_DECAY = {"min": 0.1, "max": 2.0, "default": 0.8, "step": 0.1}
    TERRAIN_FACTOR_VILLAGES = {"min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1}
    ROAD_SLOPE_TO_DISTANCE_RATIO = {"min": 0.1, "max": 3.0, "default": 1.5, "step": 0.1}
    LANDMARK_WILDERNESS = {"min": 0.1, "max": 0.8, "default": 0.3, "step": 0.05}
    PLOTSIZE = {"min": 0.5, "max": 5.0, "default": 2.0, "step": 0.1}


class WEATHER:
    """Parameter für core/weather_generator.py"""
    AIR_TEMP_ENTRY = {"min": -30, "max": 40, "default": 15, "step": 1, "suffix": "°C"}
    SOLAR_POWER = {"min": 0, "max": 50, "default": 20, "step": 1, "suffix": "°C"}
    ALTITUDE_COOLING = {"min": 2, "max": 100, "default": 6, "step": 1, "suffix": "°C/km"}
    THERMIC_EFFECT = {"min": 0.0, "max": 2.0, "default": 0.8, "step": 0.1}
    WIND_SPEED_FACTOR = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    TERRAIN_FACTOR = {"min": 0.0, "max": 2.0, "default": 1.2, "step": 0.1}


class WATER:
    """Parameter für core/water_generator.py"""
    LAKE_VOLUME_THRESHOLD = {"min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01, "suffix": "m"}
    RAIN_THRESHOLD = {"min": 1.0, "max": 20.0, "default": 5.0, "step": 0.1, "suffix": "gH2O/m²"}
    MANNING_COEFFICIENT = {"min": 0.01, "max": 0.1, "default": 0.03, "step": 0.005}
    EROSION_STRENGTH = {"min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1}
    SEDIMENT_CAPACITY_FACTOR = {"min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01}
    EVAPORATION_BASE_RATE = {"min": 0.0001, "max": 0.01, "default": 0.002, "step": 0.0001, "suffix": "m/Tag"}
    DIFFUSION_RADIUS = {"min": 1.0, "max": 20.0, "default": 5.0, "step": 0.5, "suffix": "Pixel"}
    SETTLING_VELOCITY = {"min": 0.001, "max": 0.1, "default": 0.01, "step": 0.001, "suffix": "m/s"}


class BIOME:
    """Parameter für core/biome_generator.py"""
    BIOME_WETNESS_FACTOR = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    BIOME_TEMP_FACTOR = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    SEA_LEVEL = {"min": 0, "max": 200, "default": 10, "step": 5, "suffix": "m"}
    BANK_WIDTH = {"min": 1, "max": 20, "default": 3, "step": 1, "suffix": "Pixel"}
    EDGE_SOFTNESS = {"min": 0.1, "max": 3.0, "default": 1.0, "step": 0.1}
    ALPINE_LEVEL = {"min": 500, "max": 3000, "default": 1500, "step": 50, "suffix": "m"}
    SNOW_LEVEL = {"min": 800, "max": 4000, "default": 2000, "step": 50, "suffix": "m"}
    CLIFF_SLOPE = {"min": 30, "max": 80, "default": 60, "step": 1, "suffix": "°"}


# Validation Rules für Parameter-Abhängigkeiten
class VALIDATION_RULES:
    """
    Funktionsweise: Definiert Parameter-Abhängigkeiten und Validation-Rules
    - Cross-Parameter Validation (z.B. Snow_Level > Alpine_Level)
    - Generator-Dependencies (welche Inputs werden benötigt)
    - Warning-Thresholds für Performance-kritische Parameter
    """

    # Terrain Parameter Validation
    TERRAIN_CONSTRAINTS = {
        "octaves_frequency": "octaves * frequency < 1.0",  # Verhindert zu hochfrequente Noise
        "redistribute_extreme": "redistribute_power != 1.0 or amplitude < 150"  # Warning bei extremen Werten
    }

    # Biome Parameter Validation
    BIOME_CONSTRAINTS = {
        "elevation_order": "alpine_level < snow_level",  # Alpine Zone muss unter Schneegrenze sein
        "sea_level_reasonable": "sea_level <= amplitude * 0.3"  # Meeresspiegel nicht zu hoch
    }

    # Performance Warnings
    PERFORMANCE_WARNINGS = {
        "large_map": "size >= 1024",  # Warnung bei großen Karten
        "high_detail": "octaves >= 10",  # Warnung bei sehr detaillierten Terrains
        "many_settlements": "settlements + landmarks + roadsites > 15"  # Warnung bei vielen Objekten
    }

    # Generator Dependencies
    DEPENDENCIES = {
        "geology": ["heightmap", "slopemap"],
        "settlement": ["heightmap", "slopemap", "water_map"],
        "weather": ["heightmap", "shademap", "soil_moist_map"],
        "water": ["heightmap", "slopemap", "hardness_map", "rock_map", "precip_map", "temp_map", "wind_map",
                  "humid_map"],
        "biome": ["heightmap", "slopemap", "temp_map", "soil_moist_map", "water_biomes_map"]
    }


# Utility Functions für Parameter-Handling
def get_parameter_config(generator_type, parameter_name):
    """
    Funktionsweise: Holt Parameter-Konfiguration für spezifischen Generator und Parameter
    Aufgabe: Zentrale Zugriffsfunktion für alle GUI-Komponenten
    Parameter: generator_type (str), parameter_name (str)
    Return: dict mit min/max/default/step/suffix
    """
    generator_classes = {
        "terrain": TERRAIN,
        "geology": GEOLOGY,
        "settlement": SETTLEMENT,
        "weather": WEATHER,
        "water": WATER,
        "biome": BIOME
    }

    if generator_type not in generator_classes:
        raise ValueError(f"Unknown generator type: {generator_type}")

    generator_class = generator_classes[generator_type]

    if not hasattr(generator_class, parameter_name.upper()):
        raise ValueError(f"Unknown parameter {parameter_name} for {generator_type}")

    return getattr(generator_class, parameter_name.upper())


def validate_parameter_set(generator_type, parameters):
    """
    Funktionsweise: Validiert kompletten Parameter-Satz für einen Generator
    Aufgabe: Prüft Cross-Parameter Constraints und Dependencies
    Parameter: generator_type (str), parameters (dict)
    Return: (is_valid: bool, warnings: list, errors: list)
    """
    warnings = []
    errors = []

    # Implementation würde hier Parameter-spezifische Validation durchführen
    # Beispiel für Terrain:
    if generator_type == "terrain":
        if parameters.get("octaves", 1) * parameters.get("frequency", 0.01) >= 1.0:
            warnings.append("Hohe Octaves * Frequency kann zu Noise-Artefakten führen")

    if generator_type == "biome":
        alpine = parameters.get("alpine_level", 1500)
        snow = parameters.get("snow_level", 2000)
        if alpine >= snow:
            errors.append("Alpine Level muss unter Snow Level liegen")

    return len(errors) == 0, warnings, errors