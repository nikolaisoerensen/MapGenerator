"""
Path: gui/managers/parameter_manager.py

Funktionsweise: Zentrale Koordination aller Parameter zwischen Tabs für Export, Reproduzierbarkeit und Cross-Tab-Dependencies
Aufgabe: Parameter-Communication, Export/Import, Preset-Management, Synchronisation zwischen Tabs
Features: Parameter-Hub, Export-Manager, Preset-System, Change-Tracking, Validation
"""


import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from PyQt5.QtCore import QObject, pyqtSignal, QTimer


@dataclass
class ParameterChangeEvent:
    """Event für Parameter-Änderungen"""
    source_tab: str
    parameter_name: str
    old_value: Any
    new_value: Any
    timestamp: float = field(default_factory=time.time)


class ParameterManager(QObject):
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

    def set_parameter(self, tab_name: str, parameter_name: str, value: Any):
        """
        Funktionsweise: Setzt einzelnen Parameter eines Tabs aus UI-Änderungen
        Aufgabe: Gegenstück zu BaseMapTab.parameter_ui_changed - aktualisiert Cache,
                 Change-History und benachrichtigt alle Tabs über parameter_changed
        Parameter: tab_name, parameter_name, value
        """
        if tab_name not in self.parameter_cache:
            self.parameter_cache[tab_name] = {}

        old_value = self.parameter_cache[tab_name].get(parameter_name)

        if old_value == value:
            return

        self.parameter_cache[tab_name][parameter_name] = value

        self.change_history.append(ParameterChangeEvent(
            source_tab=tab_name,
            parameter_name=parameter_name,
            old_value=old_value,
            new_value=value
        ))

        self.parameter_changed.emit(tab_name, parameter_name, old_value, value)

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