"""
Path: gui/managers/parameter_manager.py

Funktionsweise: Zentrale Koordination aller Parameter zwischen Tabs für Export, Reproduzierbarkeit und Cross-Tab-Dependencies
Aufgabe: Parameter-Communication, Export/Import, Preset-Management, Synchronisation zwischen Tabs
Features: Parameter-Hub, Export-Manager, Preset-System, Change-Tracking, Validation
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, QDateTime, QTimer


@dataclass
class ParameterChangeEvent:
    """Event für Parameter-Änderungen"""
    source_tab: str
    parameter_name: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.time)


class ParameterCommunicationHub(QObject):
    """
    Funktionsweise: Zentrale Koordination aller Parameter zwischen Tabs
    Aufgabe: Cross-Tab-Parameter-Sharing, Change-Tracking, Dependency-Management
    Features: Tab-Registration, Parameter-Broadcasting, Change-Listeners
    """

    # Signals für Parameter-Communication
    parameter_changed = pyqtSignal(str, str, object, object)  # (tab, param, old_val, new_val)
    tab_parameters_updated = pyqtSignal(str, dict)            # (tab_name, all_parameters)
    validation_status_changed = pyqtSignal(str, bool, list)   # (tab_name, is_valid, errors)

    def __init__(self):
        super().__init__()

        self.registered_tabs = {}           # {tab_name: tab_instance}
        self.parameter_dependencies = {}    # {tab_name: [dependency_tabs]}
        self.parameter_cache = {}          # {tab_name: last_parameters}
        self.change_listeners = {}         # {tab_name: [listener_functions]}
        self.change_history = []           # List[ParameterChangeEvent]
        self.parameter_constraints = {}    # {tab_name: {param: constraint_func}}

        self.logger = logging.getLogger(__name__)

    def register_tab(self, tab_name: str, tab_instance: Any, dependencies: Optional[List[str]] = None):
        """
        Funktionsweise: Registriert Tab mit optionalen Dependencies
        Parameter: tab_name, tab_instance, dependencies
        """
        self.registered_tabs[tab_name] = tab_instance

        if dependencies:
            self.parameter_dependencies[tab_name] = dependencies

        # Initial parameter cache
        if hasattr(tab_instance, 'get_current_parameters'):
            try:
                initial_params = tab_instance.get_current_parameters()
                self.parameter_cache[tab_name] = initial_params
            except Exception as e:
                self.logger.warning(f"Failed to get initial parameters for {tab_name}: {e}")
                self.parameter_cache[tab_name] = {}
        else:
            self.parameter_cache[tab_name] = {}

        self.logger.info(f"Registered tab {tab_name} with dependencies: {dependencies}")

    def get_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Funktionsweise: Sammelt Parameter aller Tabs für Export
        Return: dict {tab_name: parameters}
        """
        all_params = {}

        for tab_name, tab in self.registered_tabs.items():
            if hasattr(tab, 'get_current_parameters'):
                try:
                    params = tab.get_current_parameters()
                    all_params[tab_name] = params
                    # Cache aktualisieren
                    self.parameter_cache[tab_name] = params
                except Exception as e:
                    self.logger.error(f"Failed to get parameters from {tab_name}: {e}")
                    all_params[tab_name] = self.parameter_cache.get(tab_name, {})
            else:
                all_params[tab_name] = self.parameter_cache.get(tab_name, {})

        return all_params

    def get_tab_parameters(self, tab_name: str) -> Dict[str, Any]:
        """
        Funktionsweise: Holt Parameter eines spezifischen Tabs
        Parameter: tab_name
        Return: Parameters dict
        """
        if tab_name not in self.registered_tabs:
            return {}

        tab = self.registered_tabs[tab_name]

        if hasattr(tab, 'get_current_parameters'):
            try:
                params = tab.get_current_parameters()
                self.parameter_cache[tab_name] = params
                return params
            except Exception as e:
                self.logger.error(f"Failed to get parameters from {tab_name}: {e}")

        return self.parameter_cache.get(tab_name, {})

    def set_tab_parameters(self, tab_name: str, parameters: Dict[str, Any],
                          validate: bool = True, notify_listeners: bool = True):
        """
        Funktionsweise: Setzt Parameter für spezifischen Tab
        Parameter: tab_name, parameters, validate, notify_listeners
        """
        if tab_name not in self.registered_tabs:
            self.logger.warning(f"Tab {tab_name} not registered")
            return False

        tab = self.registered_tabs[tab_name]
        old_params = self.parameter_cache.get(tab_name, {})

        # Validation (optional)
        if validate:
            is_valid, errors = self._validate_parameters(tab_name, parameters)
            if not is_valid:
                self.logger.error(f"Parameter validation failed for {tab_name}: {errors}")
                self.validation_status_changed.emit(tab_name, False, errors)
                return False
            else:
                self.validation_status_changed.emit(tab_name, True, [])

        # Parameter setzen
        if hasattr(tab, 'set_parameters'):
            try:
                tab.set_parameters(parameters)
                self.parameter_cache[tab_name] = parameters

                # Change-Tracking
                self._track_parameter_changes(tab_name, old_params, parameters)

                # Listeners benachrichtigen
                if notify_listeners:
                    self._notify_parameter_change(tab_name, parameters)

                self.tab_parameters_updated.emit(tab_name, parameters)
                return True

            except Exception as e:
                self.logger.error(f"Failed to set parameters for {tab_name}: {e}")
                return False
        else:
            self.logger.warning(f"Tab {tab_name} does not support set_parameters")
            return False

    def add_parameter_change_listener(self, tab_name: str, listener_func: Callable):
        """
        Funktionsweise: Fügt Listener für Parameter-Änderungen hinzu
        Parameter: tab_name, listener_func
        """
        if tab_name not in self.change_listeners:
            self.change_listeners[tab_name] = []
        self.change_listeners[tab_name].append(listener_func)

    def remove_parameter_change_listener(self, tab_name: str, listener_func: Callable):
        """
        Funktionsweise: Entfernt Parameter-Change-Listener
        Parameter: tab_name, listener_func
        """
        if tab_name in self.change_listeners:
            try:
                self.change_listeners[tab_name].remove(listener_func)
            except ValueError:
                pass

    def add_parameter_constraint(self, tab_name: str, parameter_name: str,
                                constraint_func: Callable[[Any], bool]):
        """
        Funktionsweise: Fügt Constraint für Parameter hinzu
        Parameter: tab_name, parameter_name, constraint_func
        """
        if tab_name not in self.parameter_constraints:
            self.parameter_constraints[tab_name] = {}
        self.parameter_constraints[tab_name][parameter_name] = constraint_func

    def get_dependency_parameters(self, tab_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Funktionsweise: Sammelt Parameter aller Dependencies für einen Tab
        Parameter: tab_name
        Return: dict {dependency_tab: parameters}
        """
        dependency_params = {}
        dependencies = self.parameter_dependencies.get(tab_name, [])

        for dep_tab in dependencies:
            params = self.get_tab_parameters(dep_tab)
            if params:
                dependency_params[dep_tab] = params

        return dependency_params

    def get_parameter_change_history(self, tab_name: Optional[str] = None,
                                   limit: Optional[int] = None) -> List[ParameterChangeEvent]:
        """
        Funktionsweise: Holt Parameter-Change-History
        Parameter: tab_name (optional), limit (optional)
        Return: List von ParameterChangeEvents
        """
        history = self.change_history

        if tab_name:
            history = [event for event in history if event.source_tab == tab_name]

        if limit:
            history = history[-limit:]

        return history

    def clear_parameter_history(self, tab_name: Optional[str] = None):
        """
        Funktionsweise: Löscht Parameter-Change-History
        Parameter: tab_name (optional - wenn None, dann alles)
        """
        if tab_name:
            self.change_history = [
                event for event in self.change_history
                if event.source_tab != tab_name
            ]
        else:
            self.change_history.clear()

    def _validate_parameters(self, tab_name: str, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Funktionsweise: Validiert Parameter gegen Constraints
        Parameter: tab_name, parameters
        Return: (is_valid, error_messages)
        """
        errors = []
        constraints = self.parameter_constraints.get(tab_name, {})

        for param_name, value in parameters.items():
            if param_name in constraints:
                constraint_func = constraints[param_name]
                try:
                    if not constraint_func(value):
                        errors.append(f"Parameter {param_name} failed constraint validation")
                except Exception as e:
                    errors.append(f"Constraint validation error for {param_name}: {e}")

        return len(errors) == 0, errors

    def _track_parameter_changes(self, tab_name: str, old_params: Dict[str, Any],
                                new_params: Dict[str, Any]):
        """
        Funktionsweise: Verfolgt Parameter-Änderungen für History
        Parameter: tab_name, old_params, new_params
        """
        for param_name, new_value in new_params.items():
            old_value = old_params.get(param_name)

            if old_value != new_value:
                event = ParameterChangeEvent(
                    source_tab=tab_name,
                    parameter_name=param_name,
                    old_value=old_value,
                    new_value=new_value
                )

                self.change_history.append(event)
                self.parameter_changed.emit(tab_name, param_name, old_value, new_value)

        # History-Limit (max 1000 Events)
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]

    def _notify_parameter_change(self, source_tab: str, parameters: Dict[str, Any]):
        """
        Funktionsweise: Benachrichtigt Listener über Parameter-Änderungen
        Parameter: source_tab, parameters
        """
        for tab_name, listeners in self.change_listeners.items():
            if tab_name != source_tab:
                for listener in listeners:
                    try:
                        listener(source_tab, parameters)
                    except Exception as e:
                        self.logger.error(f"Parameter change listener failed: {e}")


class ParameterExportManager:
    """
    Funktionsweise: Export/Import-System für Parameter-Sets
    Aufgabe: JSON-Export/Import, Preset-Management, Metadata-Handling
    Features: Versioning, Metadata, Validation, Template-Support
    """

    def __init__(self, parameter_hub: ParameterCommunicationHub):
        self.parameter_hub = parameter_hub
        self.logger = logging.getLogger(__name__)

    def export_parameters_json(self, filename: str, include_metadata: bool = True,
                              tab_filter: Optional[List[str]] = None) -> bool:
        """
        Funktionsweise: Exportiert alle Parameter als JSON
        Parameter: filename, include_metadata, tab_filter
        Return: bool - Export erfolgreich
        """
        try:
            all_params = self.parameter_hub.get_all_parameters()

            # Tab-Filter anwenden
            if tab_filter:
                all_params = {tab: params for tab, params in all_params.items() if tab in tab_filter}

            export_data = {
                "parameters": all_params
            }

            if include_metadata:
                export_data["metadata"] = self._create_export_metadata(all_params)

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Parameter template exported to {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Template export failed: {e}")
            return False

    def create_parameter_preset(self, preset_name: str, description: str = "",
                               tab_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt Parameter-Preset aus aktuellen Werten
        Parameter: preset_name, description, tab_filter
        Return: Preset dict
        """
        all_params = self.parameter_hub.get_all_parameters()

        if tab_filter:
            all_params = {tab: params for tab, params in all_params.items() if tab in tab_filter}

        preset = {
            "preset_info": {
                "name": preset_name,
                "description": description,
                "creation_date": QDateTime.currentDateTime().toString(),
                "parameter_count": sum(len(params) for params in all_params.values()),
                "included_tabs": list(all_params.keys())
            },
            "parameters": all_params
        }

        return preset

    def apply_parameter_preset(self, preset: Dict[str, Any], validate: bool = True) -> bool:
        """
        Funktionsweise: Wendet Parameter-Preset an
        Parameter: preset, validate
        Return: bool - Anwendung erfolgreich
        """
        try:
            parameters = preset.get("parameters", {})

            success_count = 0
            for tab_name, tab_params in parameters.items():
                if self.parameter_hub.set_tab_parameters(tab_name, tab_params, validate=validate):
                    success_count += 1

            self.logger.info(f"Preset applied: {success_count}/{len(parameters)} tabs successful")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Preset application failed: {e}")
            return False

    def _create_export_metadata(self, parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt Metadata für Parameter-Export
        Parameter: parameters
        Return: Metadata dict
        """
        return {
            "export_timestamp": QDateTime.currentDateTime().toString(),
            "parameter_dependencies": dict(self.parameter_hub.parameter_dependencies),
            "map_generator_version": "1.0",
            "exported_tabs": list(parameters.keys()),
            "total_parameters": sum(len(params) for params in parameters.values()),
            "export_format_version": "1.0"
        }

    def _validate_import_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Funktionsweise: Validiert Import-Metadata für Kompatibilität
        Parameter: metadata
        Return: bool - Metadata valid
        """
        try:
            # Format-Version Check
            format_version = metadata.get("export_format_version", "1.0")
            if format_version != "1.0":
                self.logger.warning(f"Import format version mismatch: {format_version}")

            # Tab-Existenz Check
            exported_tabs = metadata.get("exported_tabs", [])
            missing_tabs = [tab for tab in exported_tabs if tab not in self.parameter_hub.registered_tabs]
            if missing_tabs:
                self.logger.warning(f"Missing tabs for import: {missing_tabs}")

            return True

        except Exception as e:
            self.logger.error(f"Metadata validation failed: {e}")
            return False


class ParameterPresetManager:
    """
    Funktionsweise: Management-System für Parameter-Presets
    Aufgabe: Preset-Storage, Kategorisierung, Suche, Versioning
    Features: File-based Storage, Kategorien, Tags, Suche
    """

    def __init__(self, preset_directory: str = "presets"):
        self.preset_directory = Path(preset_directory)
        self.preset_directory.mkdir(exist_ok=True)

        self.presets_cache = {}  # {preset_name: preset_data}
        self.categories = set()  # Set von Kategorien

        self.logger = logging.getLogger(__name__)
        self._load_presets()

    def save_preset(self, preset_name: str, preset_data: Dict[str, Any],
                   category: str = "general", tags: Optional[List[str]] = None) -> bool:
        """
        Funktionsweise: Speichert Preset in File-System
        Parameter: preset_name, preset_data, category, tags
        Return: bool - Speichern erfolgreich
        """
        try:
            # Preset-Metadata erweitern
            enhanced_preset = preset_data.copy()
            enhanced_preset["preset_info"]["category"] = category
            enhanced_preset["preset_info"]["tags"] = tags or []
            enhanced_preset["preset_info"]["file_version"] = "1.0"

            # Filename mit Kategorie
            filename = f"{category}_{preset_name}.json"
            filepath = self.preset_directory / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(enhanced_preset, f, indent=2, ensure_ascii=False)

            # Cache aktualisieren
            self.presets_cache[preset_name] = enhanced_preset
            self.categories.add(category)

            self.logger.info(f"Preset saved: {preset_name} in category {category}")
            return True

        except Exception as e:
            self.logger.error(f"Preset save failed: {e}")
            return False

    def load_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Funktionsweise: Lädt Preset aus Cache oder File
        Parameter: preset_name
        Return: Preset dict oder None
        """
        if preset_name in self.presets_cache:
            return self.presets_cache[preset_name]

        # Try loading from file
        for filepath in self.preset_directory.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    preset_data = json.load(f)

                if preset_data.get("preset_info", {}).get("name") == preset_name:
                    self.presets_cache[preset_name] = preset_data
                    return preset_data

            except Exception as e:
                self.logger.warning(f"Failed to load preset from {filepath}: {e}")

        return None

    def delete_preset(self, preset_name: str) -> bool:
        """
        Funktionsweise: Löscht Preset aus File-System und Cache
        Parameter: preset_name
        Return: bool - Löschen erfolgreich
        """
        try:
            # Aus Cache entfernen
            if preset_name in self.presets_cache:
                del self.presets_cache[preset_name]

            # File finden und löschen
            for filepath in self.preset_directory.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)

                    if preset_data.get("preset_info", {}).get("name") == preset_name:
                        filepath.unlink()
                        self.logger.info(f"Preset deleted: {preset_name}")
                        return True

                except Exception:
                    continue

            return False

        except Exception as e:
            self.logger.error(f"Preset deletion failed: {e}")
            return False

    def list_presets(self, category: Optional[str] = None,
                    tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Funktionsweise: Listet verfügbare Presets mit Filterung
        Parameter: category, tags
        Return: List von Preset-Info dicts
        """
        preset_list = []

        for preset_name, preset_data in self.presets_cache.items():
            preset_info = preset_data.get("preset_info", {})

            # Category-Filter
            if category and preset_info.get("category") != category:
                continue

            # Tag-Filter
            if tags:
                preset_tags = set(preset_info.get("tags", []))
                if not any(tag in preset_tags for tag in tags):
                    continue

            preset_list.append({
                "name": preset_name,
                "description": preset_info.get("description", ""),
                "category": preset_info.get("category", "general"),
                "tags": preset_info.get("tags", []),
                "creation_date": preset_info.get("creation_date", ""),
                "parameter_count": preset_info.get("parameter_count", 0)
            })

        return sorted(preset_list, key=lambda x: x["creation_date"], reverse=True)

    def search_presets(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Funktionsweise: Durchsucht Presets nach Namen/Beschreibung
        Parameter: search_term
        Return: List von matching Preset-Info dicts
        """
        search_term = search_term.lower()
        results = []

        for preset_name, preset_data in self.presets_cache.items():
            preset_info = preset_data.get("preset_info", {})

            # Suche in Name, Beschreibung und Tags
            searchable_text = (
                f"{preset_name} "
                f"{preset_info.get('description', '')} "
                f"{' '.join(preset_info.get('tags', []))}"
            ).lower()

            if search_term in searchable_text:
                results.append({
                    "name": preset_name,
                    "description": preset_info.get("description", ""),
                    "category": preset_info.get("category", "general"),
                    "tags": preset_info.get("tags", []),
                    "creation_date": preset_info.get("creation_date", ""),
                    "parameter_count": preset_info.get("parameter_count", 0)
                })

        return sorted(results, key=lambda x: x["creation_date"], reverse=True)

    def get_categories(self) -> List[str]:
        """
        Funktionsweise: Gibt alle verfügbaren Kategorien zurück
        Return: List von Kategorie-Namen
        """
        return sorted(list(self.categories))

    def _load_presets(self):
        """
        Funktionsweise: Lädt alle Presets aus dem Preset-Directory
        """
        try:
            for filepath in self.preset_directory.glob("*.json"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)

                    preset_info = preset_data.get("preset_info", {})
                    preset_name = preset_info.get("name")

                    if preset_name:
                        self.presets_cache[preset_name] = preset_data

                        # Kategorie zur Liste hinzufügen
                        category = preset_info.get("category", "general")
                        self.categories.add(category)

                except Exception as e:
                    self.logger.warning(f"Failed to load preset from {filepath}: {e}")

            self.logger.info(f"Loaded {len(self.presets_cache)} presets from {self.preset_directory}")

        except Exception as e:
            self.logger.error(f"Preset loading failed: {e}")


# Utility Functions für Parameter-Manager

def create_standard_parameter_hub() -> ParameterCommunicationHub:
    """
    Funktionsweise: Factory für Standard-Parameter-Hub mit Constraints
    Return: Konfigurierter ParameterCommunicationHub
    """
    hub = ParameterCommunicationHub()

    # Standard-Constraints hinzufügen
    _add_standard_constraints(hub)

    return hub


def _add_standard_constraints(hub: ParameterCommunicationHub):
    """
    Funktionsweise: Fügt Standard-Parameter-Constraints hinzu
    Parameter: hub
    """
    # Terrain Constraints
    hub.add_parameter_constraint("terrain", "size", lambda x: 64 <= x <= 2048 and (x & (x-1)) == 0)  # Power of 2
    hub.add_parameter_constraint("terrain", "amplitude", lambda x: 0 < x <= 1000)
    hub.add_parameter_constraint("terrain", "octaves", lambda x: 1 <= x <= 10)

    # Geology Constraints
    hub.add_parameter_constraint("geology", "sedimentary_hardness", lambda x: 0 <= x <= 100)
    hub.add_parameter_constraint("geology", "igneous_hardness", lambda x: 0 <= x <= 100)
    hub.add_parameter_constraint("geology", "metamorphic_hardness", lambda x: 0 <= x <= 100)


def register_standard_map_tabs(hub: ParameterCommunicationHub, tabs: Dict[str, Any]):
    """
    Funktionsweise: Registriert Standard-Map-Tabs mit Dependencies
    Parameter: hub, tabs dict
    """
    # Standard Dependencies für Map-Generator
    dependencies = {
        'terrain': [],
        'geology': ['terrain'],
        'water': ['terrain', 'geology'],
        'weather': ['terrain'],
        'biome': ['terrain', 'geology', 'water', 'weather'],
        'settlement': ['terrain', 'geology', 'water', 'biome']
    }

    for tab_name, tab_instance in tabs.items():
        tab_deps = dependencies.get(tab_name, [])
        hub.register_tab(tab_name, tab_instance, tab_deps)


class ParameterUpdateManager(QObject):
    """
    Funktionsweise: Race-Condition Prevention für Parameter-Updates
    Aufgabe: Debounced Parameter-Updates, Prevents Race-Conditions zwischen UI und Logic
    Features: Debouncing, Update-Queue, Validation-Timing, Generation-Coordination
    """

    # Signals für Update-Management
    validation_requested = pyqtSignal()
    generation_requested = pyqtSignal()
    update_completed = pyqtSignal()

    def __init__(self, parent_tab: QObject, debounce_ms: int = 500):
        super().__init__()

        self.parent_tab = parent_tab
        self.debounce_ms = debounce_ms

        # Timers für Debouncing
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self._perform_validation)

        self.generation_timer = QTimer()
        self.generation_timer.setSingleShot(True)
        self.generation_timer.timeout.connect(self._perform_generation)

        # Update-Queue
        self.pending_validation = False
        self.pending_generation = False

        self.logger = logging.getLogger(__name__)

    def request_validation(self):
        """
        Funktionsweise: Request Parameter-Validation mit Debouncing
        """
        self.pending_validation = True
        self.validation_timer.start(self.debounce_ms // 2)  # Shorter debounce for validation

    def request_generation(self):
        """
        Funktionsweise: Request Generation mit Debouncing
        """
        self.pending_generation = True
        self.generation_timer.start(self.debounce_ms)

    def _perform_validation(self):
        """
        Funktionsweise: Führt Parameter-Validation aus
        """
        if not self.pending_validation:
            return

        self.pending_validation = False

        try:
            if hasattr(self.parent_tab, 'validate_current_parameters'):
                self.parent_tab.validate_current_parameters()
                self.validation_requested.emit()
        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")

    def _perform_generation(self):
        """
        Funktionsweise: Führt Generation aus
        """
        if not self.pending_generation:
            return

        self.pending_generation = False

        try:
            if hasattr(self.parent_tab, 'generate'):
                self.parent_tab.generate()
                self.generation_requested.emit()
        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")

    def cancel_pending_updates(self):
        """
        Funktionsweise: Bricht alle pending Updates ab
        """
        self.validation_timer.stop()
        self.generation_timer.stop()
        self.pending_validation = False
        self.pending_generation = False

    def cleanup(self):
        """
        Funktionsweise: Cleanup für Tab-Wechsel
        """
        self.cancel_pending_updates()
        self.update_completed.emit()

    def import_parameters_json(self, filename: str, validate: bool = True,
                              selective_import: Optional[List[str]] = None) -> bool:
        """
        Funktionsweise: Importiert Parameter aus JSON und setzt sie in Tabs
        Parameter: filename, validate, selective_import
        Return: bool - Import erfolgreich
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            parameters = import_data.get("parameters", {})
            metadata = import_data.get("metadata", {})

            # Metadata-Validation
            if validate and metadata:
                if not self._validate_import_metadata(metadata):
                    self.logger.warning("Import metadata validation failed")

            # Selective Import
            if selective_import:
                parameters = {tab: params for tab, params in parameters.items() if tab in selective_import}

            # Parameter setzen
            success_count = 0
            for tab_name, tab_params in parameters.items():
                if self.parameter_hub.set_tab_parameters(tab_name, tab_params, validate=validate):
                    success_count += 1

            self.logger.info(f"Parameters imported: {success_count}/{len(parameters)} tabs successful")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Parameter import failed: {e}")
            return False

    def export_parameter_template(self, filename: str, tab_names: List[str]) -> bool:
        """
        Funktionsweise: Exportiert Parameter-Template mit Default-Werten
        Parameter: filename, tab_names
        Return: bool - Export erfolgreich
        """
        try:
            template_data = {
                "template_info": {
                    "name": "Parameter Template",
                    "description": "Template with default parameter values",
                    "creation_date": QDateTime.currentDateTime().toString(),
                    "included_tabs": tab_names
                },
                "parameters": {}
            }

            # Default-Parameter sammeln
            for tab_name in tab_names:
                if tab_name in self.parameter_hub.registered_tabs:
                    tab = self.parameter_hub.registered_tabs[tab_name]
                    if hasattr(tab, 'get_default_parameters'):
                        try:
                            default_params = tab.get_default_parameters()
                            template_data["parameters"][tab_name] = default_params
                        except Exception as e:
                            self.logger.warning(f"Failed to get default parameters for {tab_name}: {e}")

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(template_data, f, indent=2, ensure_ascii=False)

                self.logger.info(f"Parameter template exported to {filename}")
                return True

        except Exception as e:
            self.logger.error(f"Template export failed: {e}")
            return False