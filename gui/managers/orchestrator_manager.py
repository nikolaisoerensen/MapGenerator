"""
Path: gui/managers/orchestrator_manager.py

Funktionsweise: Vermeidet Code-Duplikation der Orchestrator-Integration zwischen allen Generator-Tabs
Aufgabe: Standard-Orchestrator-Handler, Request-Building, Signal-Management für wiederverwendbare Integration
Features: Standard-Handler, Request-Builder, Signal-Coordination, Error-Handling
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QMetaObject, Qt, Q_ARG


@dataclass
class OrchestratorRequest:
    """Standard-Request für GenerationOrchestrator"""
    generator_type: str
    parameters: Dict[str, Any]
    target_lod: str
    source_tab: str
    priority: int = 10
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class StandardOrchestratorHandler(QObject):
    """
    Funktionsweise: Wiederverwendbare Orchestrator-Integration für alle Generator-Tabs
    Aufgabe: Eliminiert Code-Duplikation zwischen Generator-Tabs
    Features: Standard-Signal-Handler, Thread-safe UI-Updates, Error-Recovery
    """

    # Signals für Handler-Communication
    handler_error = pyqtSignal(str, str)  # (generator_type, error_message)
    generation_status_changed = pyqtSignal(str, str, str)  # (generator_type, status, detail)

    def __init__(self, parent_tab: QObject, generator_type: str):
        super().__init__()

        self.parent_tab = parent_tab
        self.generator_type = generator_type
        self.orchestrator = getattr(parent_tab, 'generation_orchestrator', None)
        self.connected_signals = []

        self.logger = logging.getLogger(__name__)

        # UI-Update-Method-Namen (können von Parent-Tab überschrieben werden)
        self.ui_update_methods = {
            'generation_started': '_update_ui_generation_started',
            'generation_completed': '_update_ui_generation_completed',
            'generation_progress': '_update_ui_generation_progress',
            'lod_progression_completed': '_update_ui_lod_progression_completed'
        }

    def setup_standard_handlers(self):
        """
        Funktionsweise: Verbindet alle Standard-Orchestrator-Signals
        """
        if not self.orchestrator:
            self.logger.warning(f"No orchestrator available for {self.generator_type}")
            return False

        # Standard-Signal-Verbindungen
        signal_mappings = [
            ('generation_started', self._on_generation_started),
            ('generation_completed', self._on_generation_completed),
            ('generation_progress', self._on_generation_progress)
        ]

        # Optional: LOD-Progression (nicht alle Orchestrator haben das)
        if hasattr(self.orchestrator, 'lod_progression_completed'):
            signal_mappings.append(('lod_progression_completed', self._on_lod_progression_completed))

        # Verbindungen erstellen
        for signal_name, handler_method in signal_mappings:
            if hasattr(self.orchestrator, signal_name):
                signal = getattr(self.orchestrator, signal_name)
                connection = signal.connect(handler_method)
                self.connected_signals.append((signal_name, signal, connection))
                self.logger.debug(f"Connected {signal_name} for {self.generator_type}")
            else:
                self.logger.warning(f"Signal {signal_name} not found in orchestrator")

        return len(self.connected_signals) > 0

    def set_custom_ui_methods(self, method_mapping: Dict[str, str]):
        """
        Funktionsweise: Erlaubt Custom UI-Update-Methods
        Parameter: method_mapping {event_type: method_name}
        """
        self.ui_update_methods.update(method_mapping)

    @pyqtSlot(str, str)
    def _on_generation_started(self, generator_type: str, lod_level: str):
        """
        Funktionsweise: Standard Generation-Started Handler
        Parameter: generator_type, lod_level
        """
        if generator_type != self.generator_type:
            return

        self.generation_status_changed.emit(generator_type, "started", lod_level)

        # Thread-safe UI-Update
        method_name = self.ui_update_methods['generation_started']
        if hasattr(self.parent_tab, method_name):
            QMetaObject.invokeMethod(
                self.parent_tab, method_name,
                Qt.QueuedConnection,
                Q_ARG(str, lod_level)
            )
        else:
            self.logger.warning(f"UI update method {method_name} not found in parent tab")

    @pyqtSlot(str, str, bool)
    def _on_generation_completed(self, generator_type: str, lod_level: str, success: bool):
        """
        Funktionsweise: Standard Generation-Completed Handler
        Parameter: generator_type, lod_level, success
        """
        if generator_type != self.generator_type:
            return

        status = "completed" if success else "failed"
        self.generation_status_changed.emit(generator_type, status, lod_level)

        # Thread-safe UI-Update
        method_name = self.ui_update_methods['generation_completed']
        if hasattr(self.parent_tab, method_name):
            QMetaObject.invokeMethod(
                self.parent_tab, method_name,
                Qt.QueuedConnection,
                Q_ARG(str, lod_level),
                Q_ARG(bool, success)
            )
        else:
            self.logger.warning(f"UI update method {method_name} not found in parent tab")

    @pyqtSlot(str, str, int, str)
    def _on_generation_progress(self, generator_type: str, lod_level: str,
                                progress_percent: int, detail: str):
        """
        Funktionsweise: Standard Generation-Progress Handler
        Parameter: generator_type, lod_level, progress_percent, detail
        """
        if generator_type != self.generator_type:
            return

        self.generation_status_changed.emit(generator_type, "progress", f"{progress_percent}%")

        # Thread-safe UI-Update
        method_name = self.ui_update_methods['generation_progress']
        if hasattr(self.parent_tab, method_name):
            QMetaObject.invokeMethod(
                self.parent_tab, method_name,
                Qt.QueuedConnection,
                Q_ARG(str, lod_level),
                Q_ARG(int, progress_percent),
                Q_ARG(str, detail)
            )
        else:
            self.logger.warning(f"UI update method {method_name} not found in parent tab")

    @pyqtSlot(str, str)
    def _on_lod_progression_completed(self, generator_type: str, final_lod: str):
        """
        Funktionsweise: Standard LOD-Progression Handler
        Parameter: generator_type, final_lod
        """
        if generator_type != self.generator_type:
            return

        self.generation_status_changed.emit(generator_type, "lod_complete", final_lod)

        # Thread-safe UI-Update
        method_name = self.ui_update_methods['lod_progression_completed']
        if hasattr(self.parent_tab, method_name):
            QMetaObject.invokeMethod(
                self.parent_tab, method_name,
                Qt.QueuedConnection,
                Q_ARG(str, final_lod)
            )
        else:
            self.logger.warning(f"UI update method {method_name} not found in parent tab")

    def cleanup_connections(self):
        """
        Funktionsweise: Safe Disconnect aller Orchestrator-Signals
        """
        disconnected_count = 0

        for signal_name, signal, connection in self.connected_signals:
            try:
                if self.orchestrator:
                    signal.disconnect(connection)
                    disconnected_count += 1
            except (TypeError, RuntimeError) as e:
                self.logger.debug(f"Signal {signal_name} disconnect failed: {e}")

        self.connected_signals.clear()
        self.logger.debug(f"Disconnected {disconnected_count} orchestrator signals for {self.generator_type}")

    def get_connection_status(self) -> Dict[str, bool]:
        """
        Funktionsweise: Gibt Status aller Signal-Verbindungen zurück
        Return: dict {signal_name: is_connected}
        """
        status = {}
        for signal_name, signal, connection in self.connected_signals:
            try:
                # Test connection by checking if signal exists
                status[signal_name] = hasattr(self.orchestrator, signal_name)
            except Exception:
                status[signal_name] = False
        return status


class OrchestratorRequestBuilder:
    """
    Funktionsweise: Builder-Pattern für Orchestrator-Requests
    Aufgabe: Konsistente Request-Erstellung, Validation, Standard-Parameter
    Features: Builder-Pattern, Validation, Default-Values, Request-Templates
    """

    @staticmethod
    def build_standard_request(generator_type: str, parameters: Dict[str, Any],
                               target_lod: str, source_tab: str, priority: int = 10) -> OrchestratorRequest:
        """
        Funktionsweise: Baut Standard-Orchestrator-Request
        Parameter: generator_type, parameters, target_lod, source_tab, priority
        Return: OrchestratorRequest
        """
        return OrchestratorRequest(
            generator_type=generator_type,
            parameters=parameters.copy(),  # Copy to prevent mutations
            target_lod=target_lod,
            source_tab=source_tab,
            priority=priority
        )

    @staticmethod
    def build_terrain_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                              source_tab: str = "terrain") -> OrchestratorRequest:
        """
        Funktionsweise: Spezialisierter Terrain-Request
        Parameter: parameters, target_lod, source_tab
        Return: OrchestratorRequest
        """
        # Terrain-spezifische Parameter-Validation
        validated_params = OrchestratorRequestBuilder._validate_terrain_parameters(parameters)

        return OrchestratorRequest(
            generator_type="terrain",
            parameters=validated_params,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=10  # High priority für Terrain als Base-Layer
        )

    @staticmethod
    def build_geology_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                              source_tab: str = "geology") -> OrchestratorRequest:
        """
        Funktionsweise: Spezialisierter Geology-Request
        Parameter: parameters, target_lod, source_tab
        Return: OrchestratorRequest
        """
        validated_params = OrchestratorRequestBuilder._validate_geology_parameters(parameters)

        return OrchestratorRequest(
            generator_type="geology",
            parameters=validated_params,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=8  # Medium-high priority
        )

    @staticmethod
    def build_batch_request(requests: List[OrchestratorRequest]) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt Batch-Request für mehrere Generatoren
        Parameter: requests
        Return: Batch-Request dict
        """
        return {
            "batch_id": f"batch_{int(time.time())}",
            "requests": [
                {
                    "generator_type": req.generator_type,
                    "parameters": req.parameters,
                    "target_lod": req.target_lod,
                    "source_tab": req.source_tab,
                    "priority": req.priority
                }
                for req in requests
            ],
            "batch_timestamp": time.time()
        }

    @staticmethod
    def validate_request(request: OrchestratorRequest) -> tuple[bool, List[str]]:
        """
        Funktionsweise: Validiert Orchestrator-Request
        Parameter: request
        Return: (is_valid, error_messages)
        """
        errors = []

        # Required fields
        if not request.generator_type:
            errors.append("Missing generator_type")
        if not request.parameters:
            errors.append("Missing parameters")
        if not request.target_lod:
            errors.append("Missing target_lod")
        if not request.source_tab:
            errors.append("Missing source_tab")

        # LOD validation
        valid_lods = ["LOD64", "LOD128", "LOD256", "FINAL"]
        if request.target_lod not in valid_lods:
            errors.append(f"Invalid target_lod: {request.target_lod}")

        # Priority range
        if not (1 <= request.priority <= 10):
            errors.append(f"Priority must be 1-10, got: {request.priority}")

        return len(errors) == 0, errors

    @staticmethod
    def _validate_terrain_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Funktionsweise: Validiert und korrigiert Terrain-Parameter
        Parameter: parameters
        Return: Validierte Parameter
        """
        validated = parameters.copy()

        # Required Terrain-Parameter mit Defaults
        defaults = {
            "size": 512,
            "amplitude": 100,
            "octaves": 6,
            "frequency": 0.01,
            "persistence": 0.5,
            "lacunarity": 2.0,
            "redistribute_power": 1.0,
            "map_seed": 12345
        }

        # Fehlende Parameter mit Defaults ergänzen
        for param, default_value in defaults.items():
            if param not in validated:
                validated[param] = default_value

        # Parameter-Range-Validation
        if validated["size"] < 64:
            validated["size"] = 64
        elif validated["size"] > 2048:
            validated["size"] = 2048

        if validated["octaves"] < 1:
            validated["octaves"] = 1
        elif validated["octaves"] > 10:
            validated["octaves"] = 10

        return validated

    @staticmethod
    def _validate_geology_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Funktionsweise: Validiert und korrigiert Geology-Parameter
        Parameter: parameters
        Return: Validierte Parameter
        """
        validated = parameters.copy()

        # Required Geology-Parameter mit Defaults
        defaults = {
            "sedimentary_hardness": 30,
            "igneous_hardness": 80,
            "metamorphic_hardness": 65,
            "ridge_warping": 0.5,
            "bevel_warping": 0.3,
            "metamorph_foliation": 0.4,
            "metamorph_folding": 0.2,
            "igneous_flowing": 0.6
        }

        # Fehlende Parameter mit Defaults ergänzen
        for param, default_value in defaults.items():
            if param not in validated:
                validated[param] = default_value

        # Hardness-Parameter auf 0-100 begrenzen
        hardness_params = ["sedimentary_hardness", "igneous_hardness", "metamorphic_hardness"]
        for param in hardness_params:
            if validated[param] < 0:
                validated[param] = 0
            elif validated[param] > 100:
                validated[param] = 100

        return validated


class OrchestratorErrorHandler:
    """
    Funktionsweise: Error-Handling für Orchestrator-Integration
    Aufgabe: Error-Recovery, Retry-Logic, Fallback-Strategies
    Features: Automatic-Retry, Error-Classification, Recovery-Strategies
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.error_counts = {}  # {generator_type: error_count}
        self.last_errors = {}  # {generator_type: last_error_message}

        self.logger = logging.getLogger(__name__)

    def handle_generation_error(self, generator_type: str, error_message: str,
                                request: OrchestratorRequest) -> Optional[OrchestratorRequest]:
        """
        Funktionsweise: Behandelt Generation-Error mit Retry-Logic
        Parameter: generator_type, error_message, request
        Return: Retry-Request oder None
        """
        # Error-Count tracking
        self.error_counts[generator_type] = self.error_counts.get(generator_type, 0) + 1
        self.last_errors[generator_type] = error_message

        error_count = self.error_counts[generator_type]

        # Error-Classification
        error_type = self._classify_error(error_message)

        self.logger.warning(f"Generation error for {generator_type} (attempt {error_count}): {error_message}")

        # Retry-Logic basierend auf Error-Type
        if error_type == "retryable" and error_count <= self.max_retries:
            # Create retry request mit modifizierten Parametern
            retry_request = self._create_retry_request(request, error_count)

            self.logger.info(f"Scheduling retry {error_count}/{self.max_retries} for {generator_type}")
            return retry_request

        elif error_type == "parameter_error":
            # Versuche Parameter-Korrektur
            corrected_request = self._attempt_parameter_correction(request, error_message)
            if corrected_request and error_count <= self.max_retries:
                self.logger.info(f"Attempting parameter correction for {generator_type}")
                return corrected_request

        # Keine Retry-Möglichkeit
        self.logger.error(f"Generation failed permanently for {generator_type}: {error_message}")
        return None

    def reset_error_count(self, generator_type: str):
        """
        Funktionsweise: Setzt Error-Count für Generator zurück
        Parameter: generator_type
        """
        self.error_counts.pop(generator_type, None)
        self.last_errors.pop(generator_type, None)

    def get_error_summary(self) -> Dict[str, Any]:
        """
        Funktionsweise: Gibt Error-Summary für alle Generatoren zurück
        Return: Error-Summary dict
        """
        return {
            "error_counts": dict(self.error_counts),
            "last_errors": dict(self.last_errors),
            "total_errors": sum(self.error_counts.values())
        }

    def _classify_error(self, error_message: str) -> str:
        """
        Funktionsweise: Klassifiziert Error-Type für Retry-Strategy
        Parameter: error_message
        Return: Error-Type string
        """
        error_msg_lower = error_message.lower()

        # Memory-Errors (meist retryable mit niedrigerem LOD)
        if any(keyword in error_msg_lower for keyword in ["memory", "out of memory", "allocation"]):
            return "memory_error"

        # Parameter-Errors (korrigierbar)
        if any(keyword in error_msg_lower for keyword in ["parameter", "invalid", "range", "constraint"]):
            return "parameter_error"

        # Network/IO-Errors (retryable)
        if any(keyword in error_msg_lower for keyword in ["timeout", "connection", "io", "file"]):
            return "retryable"

        # Critical-Errors (nicht retryable)
        if any(keyword in error_msg_lower for keyword in ["critical", "fatal", "corrupted"]):
            return "critical"

        # Default: retryable
        return "retryable"

    def _create_retry_request(self, original_request: OrchestratorRequest,
                              attempt_number: int) -> OrchestratorRequest:
        """
        Funktionsweise: Erstellt Retry-Request mit angepassten Parametern
        Parameter: original_request, attempt_number
        Return: Modified OrchestratorRequest
        """
        retry_request = OrchestratorRequest(
            generator_type=original_request.generator_type,
            parameters=original_request.parameters.copy(),
            target_lod=original_request.target_lod,
            source_tab=original_request.source_tab,
            priority=max(1, original_request.priority - attempt_number)  # Lower priority for retries
        )

        # Bei Memory-Errors: LOD reduzieren
        if attempt_number > 1:
            lod_downgrade = {
                "FINAL": "LOD256",
                "LOD256": "LOD128",
                "LOD128": "LOD64",
                "LOD64": "LOD64"  # Minimum
            }
            retry_request.target_lod = lod_downgrade.get(retry_request.target_lod, "LOD64")

        return retry_request

    def _attempt_parameter_correction(self, request: OrchestratorRequest,
                                      error_message: str) -> Optional[OrchestratorRequest]:
        """
        Funktionsweise: Versucht Parameter-Korrektur basierend auf Error-Message
        Parameter: request, error_message
        Return: Corrected request oder None
        """
        corrected_params = request.parameters.copy()
        correction_made = False

        # Size-Parameter-Korrektur
        if "size" in error_message.lower():
            if "too large" in error_message.lower():
                current_size = corrected_params.get("size", 512)
                corrected_params["size"] = max(64, current_size // 2)
                correction_made = True
            elif "too small" in error_message.lower():
                current_size = corrected_params.get("size", 512)
                corrected_params["size"] = min(2048, current_size * 2)
                correction_made = True

        # Octaves-Parameter-Korrektur
        if "octaves" in error_message.lower():
            corrected_params["octaves"] = min(6, corrected_params.get("octaves", 6))
            correction_made = True

        if correction_made:
            return OrchestratorRequest(
                generator_type=request.generator_type,
                parameters=corrected_params,
                target_lod=request.target_lod,
                source_tab=request.source_tab,
                priority=request.priority
            )

        return None


class OrchestratorIntegrationManager:
    """
    Funktionsweise: High-Level Manager für komplette Orchestrator-Integration
    Aufgabe: Koordiniert Handler, Request-Builder und Error-Handler für komplette Integration
    Features: Multi-Tab-Management, Batch-Operations, Status-Monitoring
    """

    def __init__(self):
        self.handlers = {}  # {generator_type: StandardOrchestratorHandler}
        self.error_handler = OrchestratorErrorHandler()
        self.request_queue = []  # Pending requests
        self.active_generations = {}  # {generator_type: request}

        self.logger = logging.getLogger(__name__)

    def register_tab(self, tab_instance: QObject, generator_type: str) -> bool:
        """
        Funktionsweise: Registriert Tab für Orchestrator-Integration
        Parameter: tab_instance, generator_type
        Return: bool - Registration erfolgreich
        """
        try:
            handler = StandardOrchestratorHandler(tab_instance, generator_type)

            if handler.setup_standard_handlers():
                self.handlers[generator_type] = handler
                self.logger.info(f"Registered orchestrator handler for {generator_type}")
                return True
            else:
                self.logger.error(f"Failed to setup handlers for {generator_type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to register {generator_type}: {e}")
            return False

    def request_generation(self, generator_type: str, parameters: Dict[str, Any],
                           target_lod: str = "FINAL", source_tab: str = None) -> bool:
        """
        Funktionsweise: Requests Generation über registrierten Handler
        Parameter: generator_type, parameters, target_lod, source_tab
        Return: bool - Request erfolgreich
        """
        if generator_type not in self.handlers:
            self.logger.error(f"No handler registered for {generator_type}")
            return False

        # Request erstellen
        if generator_type == "terrain":
            request = OrchestratorRequestBuilder.build_terrain_request(
                parameters, target_lod, source_tab or "terrain"
            )
        elif generator_type == "geology":
            request = OrchestratorRequestBuilder.build_geology_request(
                parameters, target_lod, source_tab or "geology"
            )
        else:
            request = OrchestratorRequestBuilder.build_standard_request(
                generator_type, parameters, target_lod, source_tab or generator_type
            )

        # Request validieren
        is_valid, errors = OrchestratorRequestBuilder.validate_request(request)
        if not is_valid:
            self.logger.error(f"Invalid request for {generator_type}: {errors}")
            return False

        # Request an Orchestrator senden
        handler = self.handlers[generator_type]
        orchestrator = handler.orchestrator

        if not orchestrator:
            self.logger.error(f"No orchestrator available for {generator_type}")
            return False

        try:
            request_id = orchestrator.request_generation(
                generator_type=request.generator_type,
                parameters=request.parameters,
                target_lod=request.target_lod,
                source_tab=request.source_tab,
                priority=request.priority
            )

            if request_id:
                self.active_generations[generator_type] = request
                self.logger.info(f"Generation requested for {generator_type}: {request_id}")
                return True
            else:
                self.logger.error(f"Orchestrator rejected request for {generator_type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to request generation for {generator_type}: {e}")

            # Error-Handler für Retry-Logic
            retry_request = self.error_handler.handle_generation_error(
                generator_type, str(e), request
            )

            if retry_request:
                # Schedule retry
                self.request_queue.append(retry_request)
                self.logger.info(f"Scheduled retry for {generator_type}")

            return False

    def request_batch_generation(self, requests: List[Dict[str, Any]]) -> Dict[str, bool]:
        """
        Funktionsweise: Requests mehrere Generationen als Batch
        Parameter: requests - List von {generator_type, parameters, target_lod}
        Return: dict {generator_type: success}
        """
        results = {}

        for request_data in requests:
            generator_type = request_data["generator_type"]
            parameters = request_data["parameters"]
            target_lod = request_data.get("target_lod", "FINAL")
            source_tab = request_data.get("source_tab")

            success = self.request_generation(generator_type, parameters, target_lod, source_tab)
            results[generator_type] = success

        self.logger.info(f"Batch generation requested: {sum(results.values())}/{len(results)} successful")
        return results

    def get_generation_status(self) -> Dict[str, Any]:
        """
        Funktionsweise: Gibt Status aller aktiven Generationen zurück
        Return: Status dict
        """
        status = {
            "active_generations": list(self.active_generations.keys()),
            "registered_handlers": list(self.handlers.keys()),
            "pending_requests": len(self.request_queue),
            "error_summary": self.error_handler.get_error_summary()
        }

        # Handler-Status
        for generator_type, handler in self.handlers.items():
            status[f"{generator_type}_connections"] = handler.get_connection_status()

        return status

    def cleanup_all_handlers(self):
        """
        Funktionsweise: Cleanup aller registrierten Handler
        """
        for generator_type, handler in self.handlers.items():
            try:
                handler.cleanup_connections()
            except Exception as e:
                self.logger.error(f"Failed to cleanup handler for {generator_type}: {e}")

        self.handlers.clear()
        self.active_generations.clear()
        self.request_queue.clear()

        self.logger.info("All orchestrator handlers cleaned up")


# Utility Functions für Orchestrator-Manager

def create_standard_orchestrator_integration() -> OrchestratorIntegrationManager:
    """
    Funktionsweise: Factory für Standard-Orchestrator-Integration
    Return: Konfigurierter OrchestratorIntegrationManager
    """
    return OrchestratorIntegrationManager()


def setup_tab_orchestrator_integration(tab_instance: QObject, generator_type: str,
                                       integration_manager: Optional[
                                           OrchestratorIntegrationManager] = None) -> StandardOrchestratorHandler:
    """
    Funktionsweise: Setup für einzelnen Tab mit Orchestrator-Integration
    Parameter: tab_instance, generator_type, integration_manager
    Return: StandardOrchestratorHandler
    """
    if integration_manager:
        integration_manager.register_tab(tab_instance, generator_type)
        return integration_manager.handlers.get(generator_type)
    else:
        # Standalone-Handler
        handler = StandardOrchestratorHandler(tab_instance, generator_type)
        handler.setup_standard_handlers()
        return handler


def register_standard_map_generator_tabs(integration_manager: OrchestratorIntegrationManager,
                                         tabs: Dict[str, QObject]) -> Dict[str, bool]:
    """
    Funktionsweise: Registriert alle Standard-Map-Generator-Tabs
    Parameter: integration_manager, tabs {generator_type: tab_instance}
    Return: dict {generator_type: registration_success}
    """
    results = {}

    # Standard-Tab-Types
    standard_generators = ["terrain", "geology", "water", "weather", "biome", "settlement"]

    for generator_type in standard_generators:
        if generator_type in tabs:
            success = integration_manager.register_tab(tabs[generator_type], generator_type)
            results[generator_type] = success
        else:
            results[generator_type] = False

    return results