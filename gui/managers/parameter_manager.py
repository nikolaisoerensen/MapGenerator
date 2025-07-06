#!/usr/bin/env python3
"""
Path: MapGenerator/gui/managers/parameter_manager.py
__init__.py existiert in "managers"

Parameter Manager - Ersetzt monolithischen WorldState
Spezialisierte Manager für verschiedene Parameter-Kategorien
"""

from gui.config.default_config import DefaultConfig
from gui.utils.error_handler import ErrorHandler, safe_execute


class BaseParameterManager:
    """
    Funktionsweise: Basis-Klasse für alle Parameter-Manager
    - Standardisierte Parameter-Verwaltung
    - Automatische Validierung
    - Error Handling Integration
    """

    def __init__(self, tab_name):
        self.tab_name = tab_name
        self.error_handler = ErrorHandler()
        self._parameters = DefaultConfig.get_defaults_for_tab(tab_name)
        self._ranges = DefaultConfig.get_ranges_for_tab(tab_name)

    @safe_execute('handle_parameter_error', fallback_return={})
    def get_parameters(self):
        """Gibt alle Parameter zurück"""
        return self._parameters.copy()

    @safe_execute('handle_parameter_error')
    def set_parameters(self, params):
        """
        Funktionsweise: Setzt Parameter mit automatischer Validierung
        Args:
            params (dict): Neue Parameter
        """
        # Validierung
        is_valid, corrected_params, error_messages = DefaultConfig.validate_parameters(
            self.tab_name, params
        )

        if not is_valid:
            for error_msg in error_messages:
                self.error_handler.logger.warning(f"{self.tab_name}: {error_msg}")

        # Verwende korrigierte Parameter
        self._parameters.update(corrected_params)
        return is_valid

    @safe_execute('handle_parameter_error')
    def set_parameter(self, param_name, value):
        """
        Funktionsweise: Setzt einzelnen Parameter mit Validierung
        Args:
            param_name (str): Parameter-Name
            value: Neuer Wert
        Returns:
            bool: True wenn gültig, False wenn korrigiert
        """
        is_valid, corrected_value, error_msg = DefaultConfig.validate_parameter(
            self.tab_name, param_name, value
        )

        if not is_valid:
            self.error_handler.logger.warning(f"{self.tab_name}: {error_msg}")

        self._parameters[param_name] = corrected_value
        return is_valid

    def get_parameter(self, param_name, default=None):
        """Gibt einzelnen Parameter zurück"""
        return self._parameters.get(param_name, default)

    @safe_execute('handle_parameter_error')
    def reset_to_defaults(self):
        """Setzt alle Parameter auf Standard-Werte zurück"""
        self._parameters = DefaultConfig.get_defaults_for_tab(self.tab_name)

    def get_parameter_range(self, param_name):
        """Gibt Min/Max Range für Parameter zurück"""
        return self._ranges.get(param_name, {'min': 0, 'max': 100})


class TerrainParameterManager(BaseParameterManager):
    """
    Funktionsweise: Spezialisierter Manager für Terrain-Parameter
    - Terrain-spezifische Validierung
    - Seed-Management
    """

    def __init__(self):
        super().__init__('terrain')

    def randomize_seed(self):
        """Generiert zufälligen Seed"""
        import random
        new_seed = random.randint(0, 999999)
        self.set_parameter('seed', new_seed)
        return new_seed

    def get_heightmap_params(self):
        """Gibt nur Heightmap-relevante Parameter zurück"""
        heightmap_keys = ['size', 'height', 'octaves', 'frequency',
                          'persistence', 'lacunarity', 'redistribute_power', 'seed']
        return {key: self._parameters[key] for key in heightmap_keys if key in self._parameters}


class GeologyParameterManager(BaseParameterManager):
    """Spezialisierter Manager für Geology-Parameter"""

    def __init__(self):
        super().__init__('geology')

    def add_rock_type(self, name, hardness):
        """
        Funktionsweise: Fügt neuen Gesteinstyp hinzu
        Args:
            name (str): Name des Gesteins
            hardness (int): Härte-Wert
        """
        rock_types = self._parameters.get('rock_types', [])
        hardness_values = self._parameters.get('hardness_values', [])

        if name not in rock_types:
            rock_types.append(name)
            hardness_values.append(hardness)

            self._parameters['rock_types'] = rock_types
            self._parameters['hardness_values'] = hardness_values

    def remove_rock_type(self, name):
        """Entfernt Gesteinstyp"""
        rock_types = self._parameters.get('rock_types', [])
        hardness_values = self._parameters.get('hardness_values', [])

        if name in rock_types:
            index = rock_types.index(name)
            rock_types.pop(index)
            if index < len(hardness_values):
                hardness_values.pop(index)

            self._parameters['rock_types'] = rock_types
            self._parameters['hardness_values'] = hardness_values


class SettlementParameterManager(BaseParameterManager):
    """Spezialisierter Manager für Settlement-Parameter"""

    def __init__(self):
        super().__init__('settlement')

    def get_total_settlements(self):
        """Berechnet Gesamtzahl der Settlements"""
        return (self._parameters.get('villages', 0) +
                self._parameters.get('landmarks', 0) +
                self._parameters.get('pubs', 0))

    def validate_settlement_density(self):
        """
        Funktionsweise: Prüft ob Settlement-Dichte realistisch ist
        Returns:
            tuple: (is_valid, warning_message)
        """
        total = self.get_total_settlements()
        if total > 10:
            return False, "Zu viele Settlements für realistische Darstellung"
        return True, None


class WeatherParameterManager(BaseParameterManager):
    """Spezialisierter Manager für Weather-Parameter"""

    def __init__(self):
        super().__init__('weather')

    def get_climate_classification(self):
        """
        Funktionsweise: Klassifiziert Klima basierend auf Parametern
        Returns:
            str: Klima-Typ (Tropisch, Gemäßigt, etc.)
        """
        temp = self._parameters.get('avg_temperature', 15)
        humidity = self._parameters.get('max_humidity', 70)

        if temp > 25 and humidity > 80:
            return "Tropisch"
        elif temp > 15 and humidity > 60:
            return "Gemäßigt"
        elif temp < 5:
            return "Kalt"
        elif humidity < 40:
            return "Trocken"
        else:
            return "Gemäßigt"


class WaterParameterManager(BaseParameterManager):
    """Spezialisierter Manager für Water-Parameter"""

    def __init__(self):
        super().__init__('water')

    def calculate_water_coverage(self):
        """
        Funktionsweise: Berechnet geschätzten Wasser-Anteil der Karte
        Returns:
            float: Prozent-Anteil Wasser (0-100)
        """
        sea_level = self._parameters.get('sea_level', 15)
        lake_fill = self._parameters.get('lake_fill', 40)

        # Vereinfachte Berechnung
        sea_coverage = sea_level * 0.5  # Grobe Schätzung
        lake_coverage = lake_fill * 0.1

        return min(sea_coverage + lake_coverage, 100)


class UIStateManager:
    """
    Funktionsweise: Manager für UI-spezifische Zustände
    - Fenster-Geometrie
    - Auto-Simulation Status
    - View-Einstellungen
    """

    def __init__(self):
        self.error_handler = ErrorHandler()
        self._ui_state = DefaultConfig.get_defaults_for_tab('ui')
        self._window_geometry = None

    @safe_execute('handle_worldstate_error')
    def set_auto_simulate(self, enabled):
        """Setzt Auto-Simulation Status"""
        self._ui_state['auto_simulate'] = enabled

    def get_auto_simulate(self):
        """Gibt Auto-Simulation Status zurück"""
        return self._ui_state.get('auto_simulate', False)

    @safe_execute('handle_worldstate_error')
    def set_window_geometry(self, geometry):
        """Speichert Fenster-Geometrie"""
        self._window_geometry = geometry

    def get_window_geometry(self):
        """Gibt Fenster-Geometrie zurück"""
        return self._window_geometry

    def get_default_window_size(self):
        """Gibt Standard-Fenstergröße zurück"""
        return (self._ui_state['window_width'], self._ui_state['window_height'])

    def get_minimum_window_size(self):
        """Gibt Minimum-Fenstergröße zurück"""
        return (self._ui_state['min_width'], self._ui_state['min_height'])


class WorldParameterManager:
    """
    Funktionsweise: Haupt-Manager der alle spezialisierten Manager koordiniert
    - Ersetzt den monolithischen WorldState
    - Bessere Trennung der Verantwortlichkeiten
    - Einfachere Wartung und Erweiterung
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Spezialisierte Manager erstellen
        self.terrain = TerrainParameterManager()
        self.geology = GeologyParameterManager()
        self.settlement = SettlementParameterManager()
        self.weather = WeatherParameterManager()
        self.water = WaterParameterManager()
        self.ui_state = UIStateManager()

        self._initialized = True

    def get_manager_for_tab(self, tab_name):
        """
        Funktionsweise: Gibt den passenden Manager für einen Tab zurück
        Args:
            tab_name (str): Name des Tabs
        Returns:
            BaseParameterManager: Entsprechender Manager
        """
        manager_map = {
            'terrain': self.terrain,
            'geology': self.geology,
            'settlement': self.settlement,
            'weather': self.weather,
            'water': self.water
        }
        return manager_map.get(tab_name.lower())

    def reset_all_to_defaults(self):
        """Setzt alle Parameter auf Standard-Werte zurück"""
        self.terrain.reset_to_defaults()
        self.geology.reset_to_defaults()
        self.settlement.reset_to_defaults()
        self.weather.reset_to_defaults()
        self.water.reset_to_defaults()

    def export_all_parameters(self):
        """
        Funktionsweise: Exportiert alle Parameter für Speicherung
        Returns:
            dict: Alle Parameter strukturiert nach Tabs
        """
        return {
            'terrain': self.terrain.get_parameters(),
            'geology': self.geology.get_parameters(),
            'settlement': self.settlement.get_parameters(),
            'weather': self.weather.get_parameters(),
            'water': self.water.get_parameters(),
            'ui_state': {
                'auto_simulate': self.ui_state.get_auto_simulate(),
                'window_geometry': self.ui_state.get_window_geometry()
            }
        }

    def import_all_parameters(self, data):
        """
        Funktionsweise: Importiert Parameter aus gespeicherten Daten
        Args:
            data (dict): Parameter-Dictionary zum Importieren
        """
        if 'terrain' in data:
            self.terrain.set_parameters(data['terrain'])
        if 'geology' in data:
            self.geology.set_parameters(data['geology'])
        if 'settlement' in data:
            self.settlement.set_parameters(data['settlement'])
        if 'weather' in data:
            self.weather.set_parameters(data['weather'])
        if 'water' in data:
            self.water.set_parameters(data['water'])
        if 'ui_state' in data:
            ui_data = data['ui_state']
            if 'auto_simulate' in ui_data:
                self.ui_state.set_auto_simulate(ui_data['auto_simulate'])
            if 'window_geometry' in ui_data:
                self.ui_state.set_window_geometry(ui_data['window_geometry'])

    # Backwards-Kompatibilität mit alter WorldState API
    def get_terrain_params(self):
        """Backwards-Kompatibilität"""
        return self.terrain.get_parameters()

    def set_terrain_params(self, params):
        """Backwards-Kompatibilität"""
        return self.terrain.set_parameters(params)

    def get_geology_params(self):
        """Backwards-Kompatibilität"""
        return self.geology.get_parameters()

    def set_geology_params(self, params):
        """Backwards-Kompatibilität"""
        return self.geology.set_parameters(params)

    def get_settlement_params(self):
        """Backwards-Kompatibilität"""
        return self.settlement.get_parameters()

    def set_settlement_params(self, params):
        """Backwards-Kompatibilität"""
        return self.settlement.set_parameters(params)

    def get_weather_params(self):
        """Backwards-Kompatibilität"""
        return self.weather.get_parameters()

    def set_weather_params(self, params):
        """Backwards-Kompatibilität"""
        return self.weather.set_parameters(params)

    def get_water_params(self):
        """Backwards-Kompatibilität"""
        return self.water.get_parameters()

    def set_water_params(self, params):
        """Backwards-Kompatibilität"""
        return self.water.set_parameters(params)

    def get_auto_simulate(self):
        """Backwards-Kompatibilität"""
        return self.ui_state.get_auto_simulate()

    def set_auto_simulate(self, enabled):
        """Backwards-Kompatibilität"""
        return self.ui_state.set_auto_simulate(enabled)

    def get_window_geometry(self):
        """Backwards-Kompatibilität"""
        return self.ui_state.get_window_geometry()

    def set_window_geometry(self, geometry):
        """Backwards-Kompatibilität"""
        return self.ui_state.set_window_geometry(geometry)