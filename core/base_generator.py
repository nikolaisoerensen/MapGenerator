"""
Path: core/base_generator.py

Funktionsweise: Universelle Basis-Klasse für alle Core-Generatoren
- Einheitliches Interface mit generate() Methode
- Automatische Parameter-Loading aus value_default.py
- DataManager-Integration für Dependencies und Speicherung
- Progress-Callback-System für Loading-Tab
- LOD-System-Unterstützung
- Legacy-Kompatibilität durch abstrakte Methoden

Verwendung:
class MyGenerator(BaseGenerator):
    def _load_default_parameters(self): ...
    def _get_dependencies(self, data_manager): ...
    def _execute_generation(self, lod, dependencies, parameters): ...
    def _save_to_data_manager(self, data_manager, result, parameters): ...
"""

import logging

class BaseGenerator:
    """
    Funktionsweise: Universelle Basis-Klasse für alle Core-Generatoren
    Aufgabe: Einheitliches Interface, LOD-System, Progress-Callbacks, DataManager-Integration
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Basis-Generator mit Standard-Eigenschaften
        Aufgabe: Setup für alle Generator-spezifischen Subklassen
        Parameter: map_seed (int) - Seed für reproduzierbare Generierung
        """
        self.map_seed = map_seed
        self.is_calculating = False
        self.progress_callback = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate(self, lod="LOD64", progress=None, data_manager=None, **parameters):
        """
        Funktionsweise: Universelle Haupt-Generierungsmethode für alle Generatoren
        Aufgabe: Einheitliches Interface mit LOD-System, Progress und DataManager
        Parameter: lod (str) - LOD-Level für Generierung
        Parameter: progress (function) - Callback für Progress-Updates
        Parameter: data_manager - DataManager-Instanz für Dependencies/Speicherung
        Parameter: **parameters - Generator-spezifische Parameter (überschreiben Defaults)
        Returns: Generator-spezifisches Ergebnis (dict oder Tuple)
        """
        # 1. Initialisierung
        self.progress_callback = progress
        self.is_calculating = True
        generator_name = self.__class__.__name__.replace('Generator', '')
        self._update_progress("Initialization", 0, f"Starting {generator_name} generation")

        try:
            # 2. Parameter-Setup
            self._update_progress("Parameters", 5, "Loading default parameters...")
            default_params = self._load_default_parameters()
            final_params = {**default_params, **parameters}
            self.logger.debug(f"Using parameters: {final_params}")

            # 3. Dependencies aus DataManager holen
            self._update_progress("Dependencies", 10, "Resolving dependencies...")
            dependencies = self._get_dependencies(data_manager)
            self.logger.debug(f"Dependencies resolved: {list(dependencies.keys())}")

            # 4. Generator-spezifische Berechnung
            self._update_progress("Generation", 15, f"Executing {generator_name} generation...")
            result = self._execute_generation(lod, dependencies, final_params)

            # 5. DataManager-Speicherung
            if data_manager:
                self._update_progress("Storage", 95, "Saving to DataManager...")
                self._save_to_data_manager(data_manager, result, final_params)
                self.logger.debug("Results saved to DataManager")

            self._update_progress("Complete", 100, f"{generator_name} generation complete")
            return result

        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            self._update_progress("Error", 0, f"Generation failed: {str(e)}")
            raise

        finally:
            self.is_calculating = False

    def _load_default_parameters(self):
        """
        Funktionsweise: Lädt Standard-Parameter aus value_default.py
        Aufgabe: Generator-spezifische Parameter-Extraktion
        Returns: dict - Standard-Parameter für diesen Generator
        Hinweis: MUSS in Subklasse implementiert werden
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _load_default_parameters")

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Holt benötigte Dependencies aus DataManager
        Aufgabe: Automatische Dependency-Resolution für Generator
        Parameter: data_manager - DataManager-Instanz
        Returns: dict - Benötigte Input-Daten (key: data_name, value: data_array)
        Hinweis: MUSS in Subklasse implementiert werden
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _get_dependencies")

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Funktionsweise: Führt generator-spezifische Berechnung aus
        Aufgabe: Kernlogik des Generators mit Progress-Updates
        Parameter: lod (str) - LOD-Level
        Parameter: dependencies (dict) - Input-Daten aus DataManager
        Parameter: parameters (dict) - Finale Parameter-Konfiguration
        Returns: Generator-spezifisches Ergebnis (dict empfohlen)
        Hinweis: MUSS in Subklasse implementiert werden
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _execute_generation")

    def _save_to_data_manager(self, data_manager, result, parameters):
        """
        Funktionsweise: Speichert Ergebnis im DataManager (generische Implementierung)
        Aufgabe: Automatische Speicherung aller Generator-Outputs ohne spezifische Imports
        Parameter: data_manager - DataManager-Instanz
        Parameter: result - Generierungs-Ergebnis aus _execute_generation
        Parameter: parameters (dict) - Verwendete Parameter für Cache-Management
        Hinweis: Kann in Subklasse überschrieben werden für spezifische Speicherung
        """
        generator_type = self.__class__.__name__.replace('Generator', '').lower()

        # Bestimme DataManager-Speicher-Methode basierend auf Generator-Typ
        save_method_map = {
            'baseterrain': 'set_terrain_data_complete',
            'terrain': 'set_terrain_data_complete',
            'geology': 'set_geology_data',
            'weather': 'set_weather_data',
            'water': 'set_water_data',
            'biome': 'set_biome_data',
            'settlement': 'set_settlement_data'
        }

        # Spezielle Behandlung für verschiedene Result-Typen
        if hasattr(result, '__dict__') and hasattr(result, 'heightmap'):
            # TerrainData-ähnliches Objekt
            if hasattr(data_manager, 'set_terrain_data_complete'):
                data_manager.set_terrain_data_complete(result, parameters)
                self.logger.debug("TerrainData object saved to DataManager")

        elif isinstance(result, dict):
            # Dictionary-basierte Ergebnisse (Geology, Weather, etc.)
            save_method_prefix = save_method_map.get(generator_type, 'set_data')

            for key, value in result.items():
                save_method = getattr(data_manager, save_method_prefix, None)
                if save_method:
                    try:
                        save_method(key, value, parameters)
                        self.logger.debug(f"{generator_type} data '{key}' saved to DataManager")
                    except Exception as e:
                        self.logger.warning(f"Failed to save {key}: {e}")

        elif hasattr(result, '__len__') and len(result) >= 2:
            # Tuple/List-basierte Ergebnisse (Legacy-Format)
            self.logger.warning(f"Legacy tuple result format detected for {generator_type}")

        else:
            self.logger.warning(f"Unknown result format for {generator_type}: {type(result)}")

    def _update_progress(self, step_name, progress_percent, detail_message):
        """
        Funktionsweise: Sendet Progress-Update an Callback-Funktion
        Aufgabe: Einheitliche Progress-Updates für alle Generatoren
        Parameter: step_name (str) - Name des aktuellen Schritts
        Parameter: progress_percent (int) - Fortschritt in Prozent (0-100)
        Parameter: detail_message (str) - Detaillierte Beschreibung
        """
        if self.progress_callback:
            try:
                self.progress_callback(step_name, progress_percent, detail_message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def update_seed(self, new_seed):
        """
        Funktionsweise: Aktualisiert Seed für Generator
        Aufgabe: Ermöglicht Seed-Änderung ohne Neuinstanziierung
        Parameter: new_seed (int) - Neuer Seed für reproduzierbare Generierung
        Hinweis: Kann in Subklasse überschrieben werden für spezifische Seed-Updates
        """
        if new_seed != self.map_seed:
            self.map_seed = new_seed
            self.logger.debug(f"Seed updated to {new_seed}")

    def get_generator_info(self):
        """
        Funktionsweise: Gibt Informationen über den Generator zurück
        Aufgabe: Debugging und Monitoring-Support
        Returns: dict - Generator-Metadaten
        """
        return {
            'name': self.__class__.__name__,
            'seed': self.map_seed,
            'is_calculating': self.is_calculating,
            'has_progress_callback': self.progress_callback is not None
        }

    def validate_dependencies(self, data_manager, required_dependencies):
        """
        Funktionsweise: Validiert ob alle benötigten Dependencies verfügbar sind
        Aufgabe: Dependency-Check vor Generierung
        Parameter: data_manager - DataManager-Instanz
        Parameter: required_dependencies (list) - Liste benötigter Daten-Keys
        Returns: tuple (all_available: bool, missing: list)
        """
        if not data_manager:
            return False, required_dependencies

        missing = []
        for dep in required_dependencies:
            # Prüfe verschiedene DataManager-Bereiche
            found = False
            for get_method in ['get_terrain_data', 'get_geology_data', 'get_weather_data',
                               'get_water_data', 'get_biome_data', 'get_settlement_data']:
                if hasattr(data_manager, get_method):
                    try:
                        data = getattr(data_manager, get_method)(dep)
                        if data is not None:
                            found = True
                            break
                    except:
                        continue

            if not found:
                missing.append(dep)

        return len(missing) == 0, missing