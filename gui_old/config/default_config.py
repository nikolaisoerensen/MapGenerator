#!/usr/bin/env python3
"""
Path: MapGenerator/gui/config/default_config.py
__init__.py existiert in "config"

Zentrale Konfigurationsdatei für alle Standard-Parameter
Eliminiert hardcodierte Werte in den Tabs
"""


class DefaultConfig:
    """
    Funktionsweise: Zentrale Verwaltung aller Standard-Parameter
    - Eliminiert hardcodierte Parameter in Tab-Dateien
    - Einfache Wartung und Konsistenz
    - Parameter-Validierung Regeln definiert
    """

    # Terrain Parameter
    TERRAIN_DEFAULTS = {
        'size': 128,
        'height': 200,
        'octaves': 8,
        'frequency': 0.037,
        'persistence': 0.68,
        'lacunarity': 2.3,
        'redistribute_power': 2.5,
        'seed': 542595
    }

    # Terrain Parameter Ranges (für Validierung)
    TERRAIN_RANGES = {
        'size': {'min': 64, 'max': 1024},
        'height': {'min': 0, 'max': 400},
        'octaves': {'min': 1, 'max': 8},
        'frequency': {'min': 0.001, 'max': 0.050},
        'persistence': {'min': 0.1, 'max': 1.0},
        'lacunarity': {'min': 1.0, 'max': 4.0},
        'redistribute_power': {'min': 0.5, 'max': 3.0},
        'seed': {'min': 0, 'max': 999999}
    }

    # Geology Parameter
    GEOLOGY_DEFAULTS = {
        'rock_types': ['Sedimentäres Gestein', 'Metamorphes Gestein', 'Magmatisches Gestein'],
        'hardness_values': [30, 60, 80],
        'ridge_warping': 0.25,
        'bevel_warping': 0.15
    }

    GEOLOGY_RANGES = {
        'hardness_values': {'min': 1, 'max': 100, 'type': 'list_of_int'},
        'ridge_warping': {'min': 0.0, 'max': 1.0},
        'bevel_warping': {'min': 0.0, 'max': 1.0},
        'rock_types': {'type': 'list_of_str'}  # Spezielle Behandlung für Listen
    }

    # Settlement Parameter
    SETTLEMENT_DEFAULTS = {
        'villages': 3,
        'landmarks': 2,
        'pubs': 2,
        'connections': 3,
        'village_size': 15,
        'village_influence': 25,
        'landmark_influence': 20
    }

    SETTLEMENT_RANGES = {
        'villages': {'min': 1, 'max': 5},
        'landmarks': {'min': 1, 'max': 5},
        'pubs': {'min': 1, 'max': 5},
        'connections': {'min': 1, 'max': 5},
        'village_size': {'min': 5, 'max': 30},
        'village_influence': {'min': 10, 'max': 50},
        'landmark_influence': {'min': 5, 'max': 40}
    }

    # Weather Parameter
    WEATHER_DEFAULTS = {
        'max_humidity': 70,
        'rain_amount': 5.0,
        'evaporation': 2.0,
        'wind_speed': 8.0,
        'wind_terrain_influence': 4.0,
        'avg_temperature': 15
    }

    WEATHER_RANGES = {
        'max_humidity': {'min': 30, 'max': 100},
        'rain_amount': {'min': 1.0, 'max': 10.0},
        'evaporation': {'min': 0.0, 'max': 5.0},
        'wind_speed': {'min': 0.0, 'max': 20.0},
        'wind_terrain_influence': {'min': 0.0, 'max': 10.0},
        'avg_temperature': {'min': -10, 'max': 40}
    }

    # Water Parameter
    WATER_DEFAULTS = {
        'lake_fill': 40,
        'sea_level': 15,
        'sediment_amount': 50,
        'water_speed': 8.0,
        'rock_dependency': 60
    }

    WATER_RANGES = {
        'lake_fill': {'min': 0, 'max': 100},
        'sea_level': {'min': 0, 'max': 50},
        'sediment_amount': {'min': 0, 'max': 100},
        'water_speed': {'min': 1.0, 'max': 20.0},
        'rock_dependency': {'min': 0, 'max': 100}
    }

    # UI Defaults
    UI_DEFAULTS = {
        'auto_simulate': False,
        'window_width': 1500,
        'window_height': 1000,
        'min_width': 1500,
        'min_height': 1000
    }

    @classmethod
    def get_defaults_for_tab(cls, tab_name):
        """
        Funktionsweise: Gibt Standard-Parameter für einen Tab zurück
        Args:
            tab_name (str): Name des Tabs ('terrain', 'geology', etc.)
        Returns:
            dict: Standard-Parameter für den Tab
        """
        defaults_map = {
            'terrain': cls.TERRAIN_DEFAULTS,
            'geology': cls.GEOLOGY_DEFAULTS,
            'settlement': cls.SETTLEMENT_DEFAULTS,
            'weather': cls.WEATHER_DEFAULTS,
            'water': cls.WATER_DEFAULTS,
            'ui': cls.UI_DEFAULTS
        }
        return defaults_map.get(tab_name, {}).copy()

    _RANGES_MAP = {
        'terrain': None,
        'geology': None,
        'settlement': None,
        'weather': None,
        'water': None
    }

    @classmethod
    def get_ranges_for_tab(cls, tab_name):
        if cls._RANGES_MAP[tab_name] is None:
            if tab_name == 'terrain':
                cls._RANGES_MAP[tab_name] = cls.TERRAIN_RANGES
            elif tab_name == 'geology':
                cls._RANGES_MAP[tab_name] = cls.GEOLOGY_RANGES
            elif tab_name == 'settlement':
                cls._RANGES_MAP[tab_name] = cls.SETTLEMENT_RANGES
            elif tab_name == 'water':
                cls._RANGES_MAP[tab_name] = cls.WATER_RANGES
            elif tab_name == 'weather':
                cls._RANGES_MAP[tab_name] = cls.WEATHER_RANGES
        return cls._RANGES_MAP[tab_name]

    @classmethod
    def validate_parameter(cls, tab_name, param_name, value):
        """
        Funktionsweise: Validiert einen einzelnen Parameter
        Args:
            tab_name (str): Tab-Name
            param_name (str): Parameter-Name
            value: Zu prüfender Wert
        Returns:
            tuple: (is_valid, corrected_value, error_message)
        """
        ranges = cls.get_ranges_for_tab(tab_name)

        if param_name not in ranges:
            return True, value, None  # Kein Range definiert = gültig

        param_range = ranges[param_name]

        # NEUE LOGIK: Spezielle Behandlung für Listen
        if 'type' in param_range:
            param_type = param_range['type']

            if param_type == 'list_of_str':
                # Listen von Strings (z.B. rock_types)
                if not isinstance(value, list):
                    defaults = cls.get_defaults_for_tab(tab_name)
                    return False, defaults.get(param_name, []), \
                        f"{param_name}: Muss Liste von Strings sein"

                # Prüfe ob alle Elemente Strings sind
                if not all(isinstance(item, str) for item in value):
                    # Konvertiere zu Strings
                    corrected_value = [str(item) for item in value]
                    return False, corrected_value, \
                        f"{param_name}: Alle Elemente zu Strings konvertiert"

                return True, value, None

            elif param_type == 'list_of_int':
                # Listen von Integers (z.B. hardness_values)
                if not isinstance(value, list):
                    defaults = cls.get_defaults_for_tab(tab_name)
                    return False, defaults.get(param_name, []), \
                        f"{param_name}: Muss Liste von Zahlen sein"

                # Validiere jedes Element der Liste
                corrected_list = []
                all_valid = True
                min_val = param_range.get('min', 0)
                max_val = param_range.get('max', 100)

                for item in value:
                    try:
                        item_int = int(item)
                        if item_int < min_val:
                            item_int = min_val
                            all_valid = False
                        elif item_int > max_val:
                            item_int = max_val
                            all_valid = False
                        corrected_list.append(item_int)
                    except (ValueError, TypeError):
                        corrected_list.append(min_val)
                        all_valid = False

                if not all_valid:
                    return False, corrected_list, \
                        f"{param_name}: Einige Werte korrigiert (Range: {min_val}-{max_val})"

                return True, value, None

        # NORMALE VALIDIERUNG für Einzelwerte (nicht-Listen)
        expected_types = {
            'size': int, 'height': int, 'octaves': int,
            'frequency': float, 'persistence': float, 'lacunarity': float,
            'redistribute_power': float, 'seed': int,
            'villages': int, 'landmarks': int, 'pubs': int, 'connections': int,
            'village_size': int, 'village_influence': int, 'landmark_influence': int,
            'max_humidity': int, 'rain_amount': float, 'evaporation': float,
            'wind_speed': float, 'wind_terrain_influence': float, 'avg_temperature': int,
            'lake_fill': int, 'sea_level': int, 'sediment_amount': int,
            'water_speed': float, 'rock_dependency': int,
            'ridge_warping': float, 'bevel_warping': float
        }

        # Typ-Konvertierung
        if param_name in expected_types:
            expected_type = expected_types[param_name]
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    defaults = cls.get_defaults_for_tab(tab_name)
                    return False, defaults.get(param_name, 0), \
                        f"{param_name}: Ungültiger Datentyp, erwartet {expected_type.__name__}"

        # Range-Validierung (nur für Einzelwerte mit min/max)
        if 'min' in param_range and 'max' in param_range:
            min_val = param_range['min']
            max_val = param_range['max']

            if value < min_val:
                return False, min_val, f"{param_name}: Wert {value} unter Minimum {min_val}"
            elif value > max_val:
                return False, max_val, f"{param_name}: Wert {value} über Maximum {max_val}"

        return True, value, None

    @classmethod
    def validate_parameters(cls, tab_name, params):
        """
        Funktionsweise: Validiert alle Parameter eines Tabs
        Args:
            tab_name (str): Tab-Name
            params (dict): Parameter-Dictionary
        Returns:
            tuple: (is_valid, corrected_params, error_messages)
        """
        corrected_params = params.copy()
        error_messages = []
        all_valid = True

        for param_name, value in params.items():
            is_valid, corrected_value, error_msg = cls.validate_parameter(tab_name, param_name, value)

            if not is_valid:
                all_valid = False
                corrected_params[param_name] = corrected_value
                if error_msg:
                    error_messages.append(error_msg)

        return all_valid, corrected_params, error_messages