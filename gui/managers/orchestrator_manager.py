"""
Path: gui/managers/orchestrator_manager.py

Funktionsweise: Homogene Orchestrator-Integration für alle Generator-Tabs mit vereinheitlichter Signal-Architektur
Aufgabe: Standard-Orchestrator-Handler, Request-Building, Signal-Management für wiederverwendbare Integration zwischen allen Tabs
Features: Direkte Signal-Connections, Thread-safe UI-Updates, Request-Validation, einheitliche Tab-Integration
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


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
    Funktionsweise: Einheitliche Orchestrator-Integration für alle Generator-Tabs mit direkten Signals
    Aufgabe: Eliminiert Code-Duplikation zwischen allen Generator-Tabs mit homogener Signal-Architektur
    """

    # HOMOGENE SIGNALS für alle Tabs (einheitliche Signatures entsprechend Descriptoren)
    generation_completed = pyqtSignal(str, dict)  # (result_id, result_data)
    lod_progression_completed = pyqtSignal(str, str)  # (result_id, lod_level)
    generation_progress = pyqtSignal(int, str)  # (progress, message)

    def __init__(self, parent_tab: QObject, generator_type: str):
        super().__init__()

        self.parent_tab = parent_tab
        self.generator_type = generator_type
        self.orchestrator = getattr(parent_tab, 'generation_orchestrator', None)
        self.connected_signals = []

        self.logger = logging.getLogger(__name__)

    def setup_standard_handlers(self) -> bool:
        """
        Funktionsweise: Verbindet alle Standard-Orchestrator-Signals automatisch
        Return: bool - Setup erfolgreich
        """
        if not self.orchestrator:
            self.logger.warning(f"No orchestrator available for {self.generator_type}")
            return False

        # Standard-Signal-Verbindungen (entsprechend harmonisierter Orchestrator-Signals)
        signal_mappings = [
            ('generation_completed', self._on_generation_completed),
            ('generation_progress', self._on_generation_progress),
            ('lod_progression_completed', self._on_lod_progression_completed)
        ]

        # Verbindungen erstellen
        for signal_name, handler_method in signal_mappings:
            if hasattr(self.orchestrator, signal_name):
                signal = getattr(self.orchestrator, signal_name)
                try:
                    connection = signal.connect(handler_method)
                    self.connected_signals.append((signal_name, signal, handler_method))
                    self.logger.debug(f"Connected {signal_name} for {self.generator_type}")
                except Exception as e:
                    self.logger.error(f"Failed to connect {signal_name}: {e}")
            else:
                self.logger.warning(f"Signal {signal_name} not found in orchestrator")

        return len(self.connected_signals) > 0

    @pyqtSlot(str, dict)
    def _on_generation_completed(self, result_id: str, result_data: dict):
        """
        Funktionsweise: Internal Handler für Generation-Completed
        Emittiert: generation_completed Signal für Tab-Connections
        """
        generator_type = result_data.get('generator_type', '')
        if generator_type == self.generator_type:
            self.generation_completed.emit(result_id, result_data)

    @pyqtSlot(str, str)
    def _on_lod_progression_completed(self, result_id: str, lod_level: str):
        """
        Funktionsweise: Internal Handler für LOD-Progression
        Emittiert: lod_progression_completed Signal für Tab-Connections
        """
        # LOD-Progression ist immer relevant für den entsprechenden Generator-Type
        self.lod_progression_completed.emit(result_id, lod_level)

    @pyqtSlot(int, str)
    def _on_generation_progress(self, progress: int, message: str):
        """
        Funktionsweise: Internal Handler für Generation-Progress
        Emittiert: generation_progress Signal für Tab-Connections
        """
        # Progress-Updates sind immer relevant
        self.generation_progress.emit(progress, message)

    def cleanup_connections(self):
        """
        Funktionsweise: Safe Disconnect aller Orchestrator-Signals
        """
        disconnected_count = 0

        for signal_name, signal, handler_method in self.connected_signals:
            try:
                if self.orchestrator:
                    signal.disconnect(handler_method)
                    disconnected_count += 1
            except (TypeError, RuntimeError) as e:
                self.logger.debug(f"Signal {signal_name} disconnect failed: {e}")

        self.connected_signals.clear()
        self.logger.debug(f"Disconnected {disconnected_count} orchestrator signals for {self.generator_type}")


class OrchestratorRequestBuilder:
    """
    Funktionsweise: Builder-Pattern für alle Generator-Types mit typ-sicherer Request-Erstellung
    Aufgabe: Konsistente Request-Erstellung für alle 6 Standard-Generator-Types
    """

    @staticmethod
    def build_terrain_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                             source_tab: str = "terrain") -> OrchestratorRequest:
        """
        Funktionsweise: Terrain-Request mit Parameter-Validation
        """
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
        Funktionsweise: Geology-Request mit Parameter-Validation
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
    def build_weather_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                             source_tab: str = "weather") -> OrchestratorRequest:
        """
        Funktionsweise: Weather-Request mit Parameter-Validation
        """
        validated_params = OrchestratorRequestBuilder._validate_weather_parameters(parameters)

        return OrchestratorRequest(
            generator_type="weather",
            parameters=validated_params,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=7  # Medium priority
        )

    @staticmethod
    def build_water_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                           source_tab: str = "water") -> OrchestratorRequest:
        """
        Funktionsweise: Water-Request mit Parameter-Validation
        """
        validated_params = OrchestratorRequestBuilder._validate_water_parameters(parameters)

        return OrchestratorRequest(
            generator_type="water",
            parameters=validated_params,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=6  # Medium priority
        )

    @staticmethod
    def build_biome_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                           source_tab: str = "biome") -> OrchestratorRequest:
        """
        Funktionsweise: Biome-Request mit Parameter-Validation
        """
        validated_params = OrchestratorRequestBuilder._validate_biome_parameters(parameters)

        return OrchestratorRequest(
            generator_type="biome",
            parameters=validated_params,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=5  # Medium-low priority
        )

    @staticmethod
    def build_settlement_request(parameters: Dict[str, Any], target_lod: str = "FINAL",
                                source_tab: str = "settlement") -> OrchestratorRequest:
        """
        Funktionsweise: Settlement-Request mit Parameter-Validation
        """
        validated_params = OrchestratorRequestBuilder._validate_settlement_parameters(parameters)

        return OrchestratorRequest(
            generator_type="settlement",
            parameters=validated_params,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=4  # Low priority (letzter in Chain)
        )

    @staticmethod
    def build_batch_request(requests: List[OrchestratorRequest]) -> Dict[str, Any]:
        """
        Funktionsweise: Erstellt Batch-Request für mehrere Generatoren
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
        """Validiert und korrigiert Terrain-Parameter"""
        validated = parameters.copy()

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
        """Validiert und korrigiert Geology-Parameter"""
        validated = parameters.copy()

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

    @staticmethod
    def _validate_weather_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert und korrigiert Weather-Parameter"""
        validated = parameters.copy()

        defaults = {
            "air_temp_entry": 15.0,
            "solar_power": 1.0,
            "altitude_cooling": 0.65,
            "thermic_effect": 1.0,
            "wind_speed_factor": 1.0,
            "terrain_factor": 1.0
        }

        for param, default_value in defaults.items():
            if param not in validated:
                validated[param] = default_value

        return validated

    @staticmethod
    def _validate_water_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert und korrigiert Water-Parameter"""
        validated = parameters.copy()

        defaults = {
            "lake_volume_threshold": 100,
            "rain_threshold": 0.1,
            "manning_coefficient": 0.03,
            "erosion_strength": 1.0,
            "sediment_capacity_factor": 1.0,
            "evaporation_base_rate": 0.01,
            "diffusion_radius": 3,
            "settling_velocity": 0.1
        }

        for param, default_value in defaults.items():
            if param not in validated:
                validated[param] = default_value

        return validated

    @staticmethod
    def _validate_biome_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert und korrigiert Biome-Parameter"""
        validated = parameters.copy()

        defaults = {
            "biome_wetness_factor": 1.0,
            "biome_temp_factor": 1.0,
            "sea_level": 0.0,
            "bank_width": 5,
            "edge_softness": 1.0,
            "alpine_level": 2000,
            "snow_level": 2500,
            "cliff_slope": 30
        }

        for param, default_value in defaults.items():
            if param not in validated:
                validated[param] = default_value

        return validated

    @staticmethod
    def _validate_settlement_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validiert und korrigiert Settlement-Parameter"""
        validated = parameters.copy()

        defaults = {
            "settlements": 8,
            "landmarks": 5,
            "roadsites": 15,
            "plotnodes": 200,
            "civ_influence_decay": 0.8,
            "terrain_factor_villages": 1.0,
            "road_slope_to_distance_ratio": 1.0,
            "landmark_wilderness": 0.2,
            "plotsize": 100
        }

        for param, default_value in defaults.items():
            if param not in validated:
                validated[param] = default_value

        return validated


# Utility Functions für vereinfachte Tab-Integration

def setup_tab_orchestrator_integration(tab_instance: QObject, generator_type: str) -> StandardOrchestratorHandler:
    """
    Funktionsweise: Einheitliches Setup für alle Generator-Tabs
    Parameter: tab_instance, generator_type
    Return: StandardOrchestratorHandler
    """
    handler = StandardOrchestratorHandler(tab_instance, generator_type)

    if handler.setup_standard_handlers():
        logging.getLogger(__name__).info(f"Orchestrator handler setup completed for {generator_type}")
        return handler
    else:
        logging.getLogger(__name__).error(f"Failed to setup orchestrator handler for {generator_type}")
        return handler