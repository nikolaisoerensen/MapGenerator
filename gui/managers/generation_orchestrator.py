"""
    Path: gui/managers/generation_orchestrator.py

    GENERATION ORCHESTRATOR - THREADING UND LOD-PROGRESSION
    ========================================================

    OVERVIEW:
    Koordiniert alle 6 Map-Generatoren durch intelligente Dependency-Resolution,
    numerische LOD-Progression und CPU-optimierte Threading ohne UI-Blocking.

    CORE FEATURES:
    - Numerische LOD-Progression (1→2→3→4→n) basierend auf map_size
    - CPU-adaptive Parallelisierung (cpu_count - 2 Threads)
    - Dependency-Queue mit automatischer Resolution
    - Parameter-Impact-Analysis für selective Cache-Invalidation
    - DataManager-Integration für automatic Result-Storage

    LOD-SYSTEM:
    Numerische LOD-Levels: 1, 2, 3, 4, 5, 6, 7...
    LOD-Size-Berechnung: map_size_min * 2^(lod_level-1)
    Target-LOD: calculate_max_lod_for_size(map_size_parameter)

    SIGNAL ARCHITECTURE:
    Tab → Orchestrator:
    - request_generation(request) → request_id

    Orchestrator → Tab:
    - generation_started(generator_type: str, lod_level: int)
    - generation_completed(result_id: str, result_data: dict)
    - lod_progression_completed(result_id: str, lod_level: int)
    - generation_progress(progress: int, message: str)

    DEPENDENCY FLOW:
    Terrain → [Geology, Weather, Water] → Biome
    Terrain + Geology → Settlement

    REQUEST SYSTEM:
    ```python
    request = OrchestratorRequestBuilder.build_request(
        parameters=tab.get_parameters(),
        generator_type=GeneratorType.TERRAIN,
        target_lod=5,  # Berechnet aus map_size
        source_tab="terrain"
    )
    THREADING:

    CPU-adaptive Limits: max(1, cpu_count - 2)
    Background-Processing ohne UI-Block
    Thread-per-LOD-Level für maximale Parallelisierung
    Graceful Shutdown mit 3s Timeout

    INTEGRATION:

    DataManager: Automatic Result-Storage nach jeder LOD-Completion
    BaseMapTab: generate() ruft Orchestrator auf
    ParameterManager: Parameter-Changes triggern Impact-Analysis
"""
from dataclasses import dataclass

from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker, QTimer
from typing import Dict, List, Set, Any
import logging
import time
from enum import Enum
import os

def get_orchestrator_error_decorators():
    """
    Funktionsweise: Lazy Loading von Orchestrator Error Decorators
    Aufgabe: Lädt Threading, Memory-Critical und Core-Generation Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import threading_handler, memory_critical_handler, core_generation_handler
        return threading_handler, memory_critical_handler, core_generation_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

threading_handler, memory_critical_handler, core_generation_handler = get_orchestrator_error_decorators()

class GeneratorType(Enum):
    """Generator-Typen für Dependency-Management"""
    TERRAIN = "terrain"
    GEOLOGY = "geology"
    WEATHER = "weather"
    WATER = "water"
    BIOME = "biome"
    SETTLEMENT = "settlement"


class GenerationRequest:
    """
    Funktionsweise: Encapsulation einer Generator-Anfrage mit allen Metadaten
    Aufgabe: Typ-sichere Übertragung von Generation-Parametern zwischen Komponenten
    """
    def __init__(self, generator_type: GeneratorType, parameters: Dict[str, Any],
                 target_lod: int, source_tab: str, priority: int = 5):
        self.generator_type = generator_type
        self.parameters = parameters
        self.target_lod = target_lod
        self.source_tab = source_tab
        self.priority = priority
        self.timestamp = time.time()
        self.request_id = f"{generator_type.value}_{target_lod}_{int(self.timestamp)}"

@dataclass
class OrchestratorRequest:
    """Standard-Request für GenerationOrchestrator"""
    generator_type: str
    parameters: Dict[str, Any]
    target_lod: int
    source_tab: str
    priority: int = 5
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class OrchestratorRequestBuilder:
    """Unified Request-Builder für alle Generator-Types"""

    @staticmethod
    def build_request(parameters: Dict[str, Any], generator_type: GeneratorType,
                      target_lod: int = None, source_tab: str = None) -> OrchestratorRequest:
        """Erstellt Request mit automatischer Target-LOD-Berechnung"""
        if target_lod is None:
            from gui.managers.data_lod_manager import calculate_max_lod_for_size
            from gui.config.value_default import TERRAIN
            map_size = parameters.get("size", TERRAIN.MAPSIZE["default"])
            target_lod = calculate_max_lod_for_size(map_size)

        if source_tab is None:
            source_tab = generator_type.value

        # Direkte Priority-Zuweisung
        priority = 8 if generator_type == GeneratorType.TERRAIN else 5

        return OrchestratorRequest(
            generator_type=generator_type.value,
            parameters=parameters,
            target_lod=target_lod,
            source_tab=source_tab,
            priority=priority
        )


class ThreadState:
    """
    Funktionsweise: Status-Container für einzelnen Generator-Thread
    Aufgabe: Speichert aktuellen Status, Progress, Timing für einen Generator
    """

    def __init__(self, generator_type: str):
        self.generator_type = generator_type
        self.status = "Idle"  # Idle, Queued, Generating, Completed, Failed, Timeout
        self.current_lod = "None"
        self.progress = 0
        self.start_time = None
        self.error_message = ""

    def get_runtime(self) -> str:
        """Return: Runtime als formatierter String"""
        if self.start_time is None:
            return "0s"

        runtime = time.time() - self.start_time
        if runtime < 60:
            return f"{runtime:.0f}s"
        else:
            return f"{runtime/60:.1f}m"


class GenerationStateTracker:
    """
    Funktionsweise: Trackt Status aller 6 Generator-Threads für UI-Display
    Aufgabe: Zentrale Status-Verwaltung für Thread-Status-Display
    """

    def __init__(self):
        self.thread_states = {}  # generator_type -> ThreadState
        self.generator_types = [gen.value for gen in GeneratorType]

        # Initialisiere alle Thread-States
        for gen_type in self.generator_types:
            self.thread_states[gen_type] = ThreadState(gen_type)

    def set_request_queued(self, request: GenerationRequest):
        """Setzt Thread-Status auf 'Queued'"""
        state = self.thread_states[request.generator_type.value]
        state.status = "Queued"
        state.current_lod = request.target_lod.value
        state.progress = 0
        state.start_time = request.timestamp

    def set_request_active(self, request: GenerationRequest):
        """Setzt Thread-Status auf 'Generating'"""
        state = self.thread_states[request.generator_type.value]
        state.status = "Generating"
        state.current_lod = request.target_lod.value
        state.progress = 0

    def set_request_completed(self, generator_type: str, lod_level: str):
        """Setzt Thread-Status auf 'Completed'"""
        if generator_type in self.thread_states:
            state = self.thread_states[generator_type]
            state.status = "Completed"
            state.current_lod = lod_level
            state.progress = 100

    def set_request_failed(self, request: GenerationRequest, error_message: str):
        """Setzt Thread-Status auf 'Failed'"""
        state = self.thread_states[request.generator_type.value]
        state.status = "Failed"
        state.error_message = error_message
        state.progress = 0

    def update_progress(self, generator_type: str, lod_level: str, progress: int):
        """Aktualisiert Progress für Generator"""
        if generator_type in self.thread_states:
            state = self.thread_states[generator_type]
            state.progress = progress
            state.current_lod = lod_level

    def get_all_thread_status(self) -> List[Dict[str, Any]]:
        """
        Funktionsweise: Gibt Status aller 6 Threads für UI-Display zurück
        Return: Liste von Thread-Status-Dicts für UI-Integration
        """
        status_list = []
        for gen_type in self.generator_types:
            state = self.thread_states[gen_type]
            status_list.append({
                "generator": gen_type.title(),
                "status": state.status,
                "current_lod": state.current_lod,
                "progress": state.progress,
                "runtime": state.get_runtime(),
                "error": state.error_message
            })
        return status_list


class DependencyQueue:
    """
    Funktionsweise: Intelligente Queue für Generation-Requests mit Dependency-Resolution
    Aufgabe: Verwaltet wartende Requests und prüft automatisch Dependencies
    """

    def __init__(self):
        self.queued_requests = {}  # request_id -> GenerationRequest
        self.dependency_cache = {}  # generator_type -> required_dependencies

        # Dependency-Matrix definieren
        self.dependencies = {
            GeneratorType.TERRAIN: set(),
            GeneratorType.GEOLOGY: {GeneratorType.TERRAIN},
            GeneratorType.WEATHER: {GeneratorType.TERRAIN},
            GeneratorType.WATER: {GeneratorType.TERRAIN, GeneratorType.GEOLOGY, GeneratorType.WEATHER},
            GeneratorType.BIOME: {GeneratorType.TERRAIN, GeneratorType.WEATHER, GeneratorType.WATER},
            GeneratorType.SETTLEMENT: {GeneratorType.TERRAIN, GeneratorType.WATER, GeneratorType.BIOME}
        }

    def add_request(self, request: GenerationRequest):
        """
        Funktionsweise: Fügt Request zur Queue hinzu
        Parameter: request - GenerationRequest
        """
        self.queued_requests[request.request_id] = request

    def remove_request(self, request_id: str) -> GenerationRequest:
        """
        Funktionsweise: Entfernt Request aus Queue
        Parameter: request_id
        Return: Entfernter Request oder None
        """
        return self.queued_requests.pop(request_id, None)

    def get_available_requests(self, completed_generators: Dict[str, Set[str]], active_limit: int = 3) -> List[GenerationRequest]:
        """
        Funktionsweise: Findet alle Requests die gestartet werden können
        Parameter: completed_generators - Dict[generator_type, Set[completed_lods]], active_limit
        Return: Liste startbarer Requests (respektiert active_limit)
        """
        available = []

        for request in self.queued_requests.values():
            if len(available) >= active_limit:
                break

            if self.can_start_request(request, completed_generators):
                available.append(request)

        # Nach Priority sortieren
        available.sort(key=lambda r: (-r.priority, r.timestamp))
        return available[:active_limit]

    def can_start_request(self, request: GenerationRequest, completed_generators: Dict[str, Set[str]]) -> bool:
        """
        Funktionsweise: Prüft ob Request gestartet werden kann basierend auf Dependencies
        Parameter: request, completed_generators
        Return: bool - Request kann gestartet werden
        """
        required_deps = self.dependencies.get(request.generator_type, set())

        for dep_type in required_deps:
            dep_key = dep_type.value
            # Mindestens LOD64 muss verfügbar sein
            if dep_key not in completed_generators or "LOD64" not in completed_generators[dep_key]:
                return False

        return True

    def get_timed_out_requests(self, current_time: float, timeout_seconds: int = 600) -> List[GenerationRequest]:
        """
        Funktionsweise: Findet alle Requests die zu lange in Queue warten
        Parameter: current_time, timeout_seconds
        Return: Liste von timed-out Requests
        """
        timed_out = []
        for request in self.queued_requests.values():
            if current_time - request.timestamp > timeout_seconds:
                timed_out.append(request)
        return timed_out

    def clear(self):
        """Leert komplette Queue"""
        self.queued_requests.clear()

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Funktionsweise: Gibt aktuellen Queue-Status zurück
        Return: Dict mit Queue-Statistiken
        """
        return {
            "total_queued": len(self.queued_requests),
            "by_generator": {gen.value: sum(1 for r in self.queued_requests.values() if r.generator_type == gen)
                           for gen in GeneratorType}
        }


class GenerationOrchestrator(QObject):
    """
    Funktionsweise: Zentrale Orchestrierung aller Map-Generation mit homogener Signal-Architektur für alle Tabs
    Aufgabe: Koordiniert alle 6 Generatoren mit einheitlichen Signals für StandardOrchestratorHandler-Integration
    """

    # Harmonisierte Signals mit numerischen LODs
    generation_completed = pyqtSignal(str, dict)  # (result_id, result_data)
    lod_progression_completed = pyqtSignal(str, int)  # (result_id, lod_level) - int!

    # Tab-kompatible Signals
    generation_started = pyqtSignal(str, int)  # (generator_type, lod_level)
    generation_failed = pyqtSignal(str, int, str)  # (generator_type, lod_level, error)
    generation_progress = pyqtSignal(int, str)  # (progress, message)

    # Zusätzliche Signals für erweiterte Funktionalität
    dependency_invalidated = pyqtSignal(str, list)  # (generator_type: str, affected_generators: List[str])
    batch_generation_completed = pyqtSignal(bool, str)  # (success: bool, summary_message: str)
    queue_status_changed = pyqtSignal(list)  # (thread_status_list: List[Dict])

    def __init__(self, data_manager):
        super().__init__()
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)

        # Dependency-Tree Definition (basierend auf Core-Generator Requirements)
        self.dependency_tree = {
            GeneratorType.TERRAIN: set(),  # Keine Dependencies
            GeneratorType.GEOLOGY: {GeneratorType.TERRAIN},
            GeneratorType.WEATHER: {GeneratorType.TERRAIN},
            GeneratorType.WATER: {GeneratorType.TERRAIN, GeneratorType.GEOLOGY, GeneratorType.WEATHER},
            GeneratorType.BIOME: {GeneratorType.TERRAIN, GeneratorType.WEATHER, GeneratorType.WATER},
            GeneratorType.SETTLEMENT: {GeneratorType.TERRAIN, GeneratorType.WATER, GeneratorType.BIOME}
        }

        # Parameter-Impact-Matrix (welche Parameter-Änderungen invalidieren nachgelagerte Generatoren)
        self.impact_matrix = {
            GeneratorType.TERRAIN: {
                "high_impact": ["map_seed", "size", "amplitude", "octaves", "frequency"],
                "medium_impact": ["persistence", "lacunarity"],
                "low_impact": ["redistribute_power"]
            },
            GeneratorType.GEOLOGY: {
                "high_impact": ["sedimentary_hardness", "igneous_hardness", "metamorphic_hardness"],
                "medium_impact": ["ridge_warping", "bevel_warping"],
                "low_impact": ["metamorph_foliation", "igneous_flowing"]
            },
            GeneratorType.WEATHER: {
                "high_impact": ["air_temp_entry", "solar_power", "altitude_cooling"],
                "medium_impact": ["thermic_effect", "wind_speed_factor"],
                "low_impact": ["terrain_factor"]
            },
            GeneratorType.WATER: {
                "high_impact": ["lake_volume_threshold", "rain_threshold", "erosion_strength"],
                "medium_impact": ["manning_coefficient", "sediment_capacity_factor"],
                "low_impact": ["evaporation_base_rate", "diffusion_radius"]
            },
            GeneratorType.BIOME: {
                "high_impact": ["biome_wetness_factor", "biome_temp_factor", "sea_level"],
                "medium_impact": ["alpine_level", "snow_level", "bank_width"],
                "low_impact": ["edge_softness", "cliff_slope"]
            },
            GeneratorType.SETTLEMENT: {
                "high_impact": ["settlements", "landmarks", "roadsites", "plotnodes"],
                "medium_impact": ["civ_influence_decay", "terrain_factor_villages"],
                "low_impact": ["plotsize", "landmark_wilderness", "road_slope_to_distance_ratio"]
            }
        }

        # Generator-Instanzen (Lazy Loading)
        self._generator_instances = {}

        # Threading-Management
        self.generation_threads = {}
        self.thread_mutex = QMutex()
        self.active_requests = set()

        # LOD-Progression Tracking mit DataManager-Integration
        self.lod_progression_queue = {}  # generator_type -> [LOD64, LOD128, LOD256, FINAL]
        self.lod_completion_status = {}  # generator_type -> {LOD64: True, LOD128: False, ...}

        # Dependency-Queue System
        self.dependency_queue = DependencyQueue()
        self.state_tracker = GenerationStateTracker()
        self.processing_requests = set()

        # Request-Tracking für result_id Management
        self.active_request_mapping = {}  # request_id -> GenerationRequest

        # Performance-Monitoring
        self.generation_timings = {}
        self.memory_usage_tracking = {}

        # Setup
        self.setup_lod_tracking()
        self.setup_threading()
        self.setup_dependency_resolution()

        cpu_count = os.cpu_count() or 4
        if cpu_count <= 2:
            self.max_parallel_generations = cpu_count
        else:
            self.max_parallel_generations = max(1, cpu_count - 2)  # 2 Kerne frei lassen

        self.logger.info(f"Parallel generation limit: {self.max_parallel_generations}")

    def setup_lod_tracking(self):
        """
        Funktionsweise: Initialisiert LOD-Completion-Tracking für alle Generatoren - NUMERISCH
        Aufgabe: Baseline für LOD-Progression-Management mit DataManager-Integration
        """
        for generator_type in GeneratorType:
            # Numerisches LOD-Tracking statt String-basiert
            self.lod_completion_status[generator_type.value] = {}
            # Wird dynamisch gefüllt wenn LODs completed werden
            # Beispiel: {"terrain": {1: True, 2: True, 3: False, 4: False}}

    def setup_threading(self):
        """
        Funktionsweise: Konfiguriert Threading-System für Background-LOD-Enhancement
        Aufgabe: Thread-Pool-Setup ohne UI-Blocking
        """
        # Thread-Management
        self.generation_threads = {}
        self.thread_mutex = QMutex()
        self.active_requests = set()

    def setup_dependency_resolution(self):
        """
        Funktionsweise: Konfiguriert Dependency-Queue-Resolution-System
        Aufgabe: Timer für kontinuierliche Queue-Verarbeitung und Deadlock-Prevention
        """
        # Queue-Resolution Timer
        self.queue_resolution_timer = QTimer()
        self.queue_resolution_timer.timeout.connect(self.resolve_dependency_queue)
        self.queue_resolution_timer.start(2000)  # Alle 2 Sekunden

        # Timeout-Management Timer
        self.timeout_check_timer = QTimer()
        self.timeout_check_timer.timeout.connect(self.check_generation_timeouts)
        self.timeout_check_timer.start(30000)  # Alle 30 Sekunden

    @core_generation_handler("generation_request")
    def request_generation(self, request) -> str:
        """
        Funktionsweise: Einheitlicher Entry-Point für alle Generator-Requests von allen Tabs
        Parameter: request - OrchestratorRequest von OrchestratorRequestBuilder
        Return: request_id für Request-Tracking
        """
        # Convert zu internem GenerationRequest falls nötig
        if hasattr(request, 'generator_type') and hasattr(request, 'parameters'):
            # OrchestratorRequest von orchestrator_manager
            try:
                gen_type_enum = GeneratorType(request.generator_type)
            except ValueError as e:
                self.logger.error(f"Invalid generator_type: {e}")
                return None

            internal_request = GenerationRequest(
                gen_type_enum,
                request.parameters,
                request.target_lod,
                request.source_tab,
                request.priority
            )
        else:
            # Fallback für Legacy-Calls
            self.logger.warning("Legacy request format detected")
            return None

        # Parameter-Impact-Analyse
        affected_generators = self.calculate_parameter_impact(internal_request.generator_type, internal_request.parameters)

        # Cache-Invalidation für betroffene Generatoren
        if affected_generators:
            self.invalidate_downstream_dependencies(internal_request.generator_type, affected_generators)

        # Request zur DependencyQueue hinzufügen
        self.dependency_queue.add_request(internal_request)
        self.state_tracker.set_request_queued(internal_request)

        # Request-Mapping für result_id Management
        self.active_request_mapping[internal_request.request_id] = internal_request

        # Queue-Status-Update emittieren
        self.emit_queue_status_update()

        self.logger.info(f"Generation queued: {internal_request.request_id}")
        return internal_request.request_id

    @threading_handler("dependency_resolution")
    def resolve_dependency_queue(self):
        """
        Funktionsweise: Kontinuierliche Dependency-Queue-Resolution mit Threading-Protection
        Aufgabe: Startet alle verfügbaren Requests aus Queue, respektiert Dependencies und Limits
        """
        available_requests = self.dependency_queue.get_available_requests(
            completed_generators=self.get_completed_generators(),
            active_limit=3  # Max 3 parallele Generationen
        )

        for request in available_requests:
            try:
                # Request aus Queue entfernen und zu aktiven verschieben
                self.dependency_queue.remove_request(request.request_id)
                self.processing_requests.add(request)

                # State-Tracker aktualisieren
                self.state_tracker.set_request_active(request)

                # LOD-Progression starten
                success = self.start_lod_progression(request)

                if not success:
                    self.state_tracker.set_request_failed(request, "Failed to start LOD progression")
                    self.processing_requests.discard(request)

            except Exception as e:
                self.logger.error(f"Failed to start request {request.request_id}: {e}")
                self.state_tracker.set_request_failed(request, str(e))
                self.processing_requests.discard(request)

        # Queue-Status-Update emittieren
        self.emit_queue_status_update()

    def check_generation_timeouts(self):
        """
        Funktionsweise: Prüft und behandelt Timeout-Situationen für alle aktiven Generationen
        Aufgabe: Deadlock-Prevention durch Timeout-Management
        """
        current_time = time.time()
        timed_out_requests = []

        # Prüfe aktive Requests auf Timeout
        for request in list(self.processing_requests):
            if current_time - request.timestamp > 300:  # 5 Minuten Timeout
                timed_out_requests.append(request)
                self.logger.warning(f"Request {request.request_id} timed out")

        # Prüfe Queue-Requests auf Timeout
        queue_timeouts = self.dependency_queue.get_timed_out_requests(current_time, timeout_seconds=600)

        # Timeout-Requests aufräumen
        for request in timed_out_requests + queue_timeouts:
            self.cleanup_timed_out_request(request)

        if timed_out_requests or queue_timeouts:
            self.emit_queue_status_update()

    def cleanup_timed_out_request(self, request: GenerationRequest):
        """
        Funktionsweise: Räumt timed-out Request auf und stoppt zugehörigen Thread
        Parameter: request - Request der timed-out ist
        """
        # Request aus aktiven entfernen
        self.processing_requests.discard(request)

        # Request-Mapping aufräumen
        self.active_request_mapping.pop(request.request_id, None)

        # Thread stoppen falls vorhanden
        thread_key = f"{request.generator_type.value}_{request.target_lod.value}"
        with QMutexLocker(self.thread_mutex):
            if thread_key in self.generation_threads:
                thread = self.generation_threads[thread_key]
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(3000)
                del self.generation_threads[thread_key]

        # State-Tracker aktualisieren
        self.state_tracker.set_request_failed(request, "Timeout")

        self.logger.info(f"Cleaned up timed-out request: {request.request_id}")

    def get_completed_generators(self) -> Dict[str, Set[str]]:
        """
        Funktionsweise: Sammelt alle abgeschlossenen Generatoren pro LOD-Level mit DataManager-Integration
        Return: Dict[generator_type, Set[completed_lod_levels]]
        """
        completed = {}
        for generator_type in GeneratorType:
            completed[generator_type.value] = set()

            # DataManager-Integration für verfügbare LOD-Levels
            if generator_type == GeneratorType.TERRAIN:
                # Terrain-spezifische LOD-Prüfung über DataManager
                if self.data_manager.has_terrain_lod("LOD64"):
                    completed[generator_type.value].add("LOD64")
                if self.data_manager.has_terrain_lod("LOD128"):
                    completed[generator_type.value].add("LOD128")
                if self.data_manager.has_terrain_lod("LOD256"):
                    completed[generator_type.value].add("LOD256")
                if self.data_manager.has_terrain_lod("FINAL"):
                    completed[generator_type.value].add("FINAL")
            else:
                # Andere Generatoren: Standard-Status-Check
                status = self.lod_completion_status.get(generator_type.value, {})
                for lod_level, is_completed in status.items():
                    if is_completed:
                        completed[generator_type.value].add(lod_level)

        return completed

    def emit_queue_status_update(self):
        """
        Funktionsweise: Emittiert aktuellen Queue-Status für UI-Updates
        Aufgabe: Stellt 6-Thread-Status für UI-Display bereit
        """
        status_list = self.state_tracker.get_all_thread_status()
        self.queue_status_changed.emit(status_list)

    def start_lod_progression(self, request: GenerationRequest) -> bool:
        """
        Funktionsweise: Startet LOD-Progression für eine Generation-Request mit DataManager-Integration
        Parameter: request
        Return: bool - Success
        """
        generator_type = request.generator_type
        target_lod = request.target_lod

        # LOD-Progression-Queue erstellen basierend auf verfügbaren DataManager-Daten
        lod_sequence = self.create_lod_sequence(generator_type, target_lod)

        if not lod_sequence:
            return False

        # Queue speichern für Tracking
        self.lod_progression_queue[generator_type.value] = lod_sequence

        # Erste LOD-Stufe starten
        return self.execute_next_lod_level(generator_type, request.parameters, request.request_id)

    def create_lod_sequence(self, generator_type: GeneratorType, target_lod: int) -> List[int]:
        """Erstellt numerische LOD-Sequence von aktuellem Level bis Target"""
        current_lod = self.data_manager.get_current_lod_level(generator_type.value)

        if current_lod >= target_lod:
            return []  # Bereits erreicht oder überschritten

        return list(range(current_lod + 1, target_lod + 1))

    def execute_next_lod_level(self, generator_type: GeneratorType, parameters: Dict[str, Any], request_id: str) -> bool:
        """
        Funktionsweise: Führt nächstes LOD-Level in der Progression aus
        Parameter: generator_type, parameters, request_id
        Return: bool - Success
        """
        queue_key = generator_type.value

        if queue_key not in self.lod_progression_queue or not self.lod_progression_queue[queue_key]:
            # Progression abgeschlossen - Final Completion Signal emittieren
            self.emit_final_completion_signal(request_id, generator_type)
            return True

        # Nächstes LOD-Level aus Queue
        current_lod = self.lod_progression_queue[queue_key].pop(0)

        # Generator-Instanz holen
        generator_instance = self.get_generator_instance(generator_type)

        if not generator_instance:
            self.logger.error(f"No generator instance available for {generator_type.value}")
            return False

        # Generation in Thread starten
        thread = GenerationThread(
            generator_instance=generator_instance,
            generator_type=generator_type,
            lod_level=current_lod,
            parameters=parameters,
            data_manager=self.data_manager,
            request_id=request_id,
            parent=self
        )

        # Thread-Signals verbinden
        thread.generation_completed.connect(self.on_lod_generation_completed)
        thread.generation_progress.connect(self.on_generation_progress)

        # Thread starten
        with QMutexLocker(self.thread_mutex):
            thread_key = f"{generator_type.value}_{current_lod.value}"
            self.generation_threads[thread_key] = thread

        thread.start()

        return True

    def on_lod_generation_completed(self, request_id: str, generator_type: str, lod_level: str, success: bool,
                                    result_data: dict):
        """
        Funktionsweise: Callback für abgeschlossene LOD-Generation mit DataManager-Integration
        Parameter: request_id, generator_type, lod_level, success, result_data
        """
        # LOD-Status aktualisieren
        if success:
            self.lod_completion_status[generator_type][lod_level] = True
            self.state_tracker.set_request_completed(generator_type, lod_level)

            # DataManager-Integration: Ergebnis automatisch speichern
            self.save_generation_result_to_data_manager(generator_type, lod_level, result_data)

        # LOD-Progression-Signal emittieren (für sofortige UI-Updates mit bestem verfügbarem LOD)
        self.lod_progression_completed.emit(request_id, lod_level)

        # Thread aufräumen
        thread_key = f"{generator_type}_{lod_level}"
        with QMutexLocker(self.thread_mutex):
            if thread_key in self.generation_threads:
                del self.generation_threads[thread_key]

        if success:
            # Nächstes LOD-Level starten
            try:
                generator_type_enum = GeneratorType(generator_type)

                # Original-Request aus Mapping holen für Parameter
                original_request = self.active_request_mapping.get(request_id)
                if original_request:
                    self.execute_next_lod_level(generator_type_enum, original_request.parameters, request_id)
                else:
                    self.logger.warning(f"Could not find original request for {request_id}")

            except ValueError:
                self.logger.error(f"Invalid generator_type in callback: {generator_type}")
        else:
            # Bei Fehler: Request aus aktiven entfernen
            original_request = self.active_request_mapping.get(request_id)
            if original_request:
                self.processing_requests.discard(original_request)
                self.active_request_mapping.pop(request_id, None)
                self.state_tracker.set_request_failed(original_request, result_data.get("error", "Unknown error"))

        # Queue-Resolution triggern falls neue Dependencies verfügbar
        QTimer.singleShot(100, self.resolve_dependency_queue)

    def save_generation_result_to_data_manager(self, generator_type: str, lod_level: str, result_data: dict):
        """
        Funktionsweise: Speichert Generation-Ergebnis automatisch im DataManager
        Parameter: generator_type, lod_level, result_data
        """
        try:
            generator_output = result_data.get("generator_output")
            parameters_used = result_data.get("parameters_used", {})

            if generator_type == "terrain" and generator_output:
                # TerrainData-Objekt speichern
                self.data_manager.set_terrain_data_complete(generator_output, parameters_used)
                self.logger.debug(f"Terrain data saved to DataManager for {lod_level}")

            elif generator_type == "geology" and generator_output:
                # Geology-Daten speichern (rock_map, hardness_map)
                if hasattr(generator_output, 'rock_map'):
                    self.data_manager.set_geology_data("rock_map", generator_output.rock_map, parameters_used)
                if hasattr(generator_output, 'hardness_map'):
                    self.data_manager.set_geology_data("hardness_map", generator_output.hardness_map, parameters_used)

            elif generator_type == "weather" and generator_output:
                # Weather-Daten speichern
                weather_outputs = ["wind_map", "temp_map", "precip_map", "humid_map"]
                for output_key in weather_outputs:
                    if hasattr(generator_output, output_key):
                        output_data = getattr(generator_output, output_key)
                        self.data_manager.set_weather_data(output_key, output_data, parameters_used)

            elif generator_type == "water" and generator_output:
                # Water-Daten speichern
                water_outputs = ["water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
                               "erosion_map", "sedimentation_map", "rock_map_updated", "evaporation_map",
                               "ocean_outflow", "water_biomes_map"]
                for output_key in water_outputs:
                    if hasattr(generator_output, output_key):
                        output_data = getattr(generator_output, output_key)
                        self.data_manager.set_water_data(output_key, output_data, parameters_used)

            elif generator_type == "biome" and generator_output:
                # Biome-Daten speichern
                biome_outputs = ["biome_map", "biome_map_super", "super_biome_mask"]
                for output_key in biome_outputs:
                    if hasattr(generator_output, output_key):
                        output_data = getattr(generator_output, output_key)
                        self.data_manager.set_biome_data(output_key, output_data, parameters_used)

            elif generator_type == "settlement" and generator_output:
                # Settlement-Daten speichern
                settlement_outputs = ["settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map"]
                for output_key in settlement_outputs:
                    if hasattr(generator_output, output_key):
                        output_data = getattr(generator_output, output_key)
                        self.data_manager.set_settlement_data(output_key, output_data, parameters_used)

        except Exception as e:
            self.logger.error(f"Failed to save generation result to DataManager: {e}")

    def on_generation_progress(self, progress: int, message: str):
        """
        Funktionsweise: Callback für Generation-Progress mit vereinfachten Parametern
        Parameter: progress, message
        """
        # Vereinfachtes Progress-Signal emittieren (homogen für alle Tabs)
        self.generation_progress.emit(progress, message)

    def emit_final_completion_signal(self, request_id: str, generator_type: GeneratorType):
        """
        Funktionsweise: Emittiert Final-Completion-Signal wenn alle LOD-Level abgeschlossen sind
        Parameter: request_id, generator_type
        """
        # Original-Request aus Mapping holen
        original_request = self.active_request_mapping.get(request_id)
        if not original_request:
            self.logger.warning(f"Could not find original request for final completion: {request_id}")
            return

        # Generator-Output aus DataManager holen
        generator_data = self.get_generator_data_from_data_manager(generator_type.value)

        # result_data für Tab-Kompatibilität zusammenstellen
        result_data = {
            "generator_type": generator_type.value,
            "lod_level": original_request.target_lod.value,
            "success": True,
            "data": generator_data,
            "source_tab": original_request.source_tab,
            "timestamp": time.time()
        }

        # Final-Completion-Signal emittieren (homogen für alle Tabs)
        self.generation_completed.emit(request_id, result_data)

        # Request aus aktiven entfernen
        self.processing_requests.discard(original_request)
        self.active_request_mapping.pop(request_id, None)

        self.logger.info(f"Final completion emitted for {request_id}")

    def get_generator_data_from_data_manager(self, generator_type: str) -> dict:
        """
        Funktionsweise: Holt Generator-Daten aus DataManager für Final-Completion
        Parameter: generator_type
        Return: dict mit allen Generator-Outputs
        """
        if generator_type == "terrain":
            return {
                "heightmap": self.data_manager.get_terrain_data("heightmap"),
                "slopemap": self.data_manager.get_terrain_data("slopemap"),
                "shadowmap": self.data_manager.get_terrain_data("shadowmap"),
                "terrain_data_complete": self.data_manager.get_terrain_data("complete")
            }
        elif generator_type == "geology":
            return {
                "rock_map": self.data_manager.get_geology_data("rock_map"),
                "hardness_map": self.data_manager.get_geology_data("hardness_map")
            }
        elif generator_type == "weather":
            return {
                "wind_map": self.data_manager.get_weather_data("wind_map"),
                "temp_map": self.data_manager.get_weather_data("temp_map"),
                "precip_map": self.data_manager.get_weather_data("precip_map"),
                "humid_map": self.data_manager.get_weather_data("humid_map")
            }
        elif generator_type == "water":
            return {
                "water_map": self.data_manager.get_water_data("water_map"),
                "flow_map": self.data_manager.get_water_data("flow_map"),
                "soil_moist_map": self.data_manager.get_water_data("soil_moist_map"),
                "water_biomes_map": self.data_manager.get_water_data("water_biomes_map")
            }
        elif generator_type == "biome":
            return {
                "biome_map": self.data_manager.get_biome_data("biome_map"),
                "biome_map_super": self.data_manager.get_biome_data("biome_map_super"),
                "super_biome_mask": self.data_manager.get_biome_data("super_biome_mask")
            }
        elif generator_type == "settlement":
            return {
                "settlement_list": self.data_manager.get_settlement_data("settlement_list"),
                "plot_map": self.data_manager.get_settlement_data("plot_map"),
                "civ_map": self.data_manager.get_settlement_data("civ_map")
            }
        else:
            return {}

    def get_generator_instance(self, generator_type: GeneratorType):
        """
        Funktionsweise: Lazy Loading von Generator-Instanzen
        Parameter: generator_type
        Return: Generator-Instanz oder None
        """
        if generator_type.value not in self._generator_instances:
            try:
                if generator_type == GeneratorType.TERRAIN:
                    from core.terrain_generator import BaseTerrainGenerator
                    self._generator_instances[generator_type.value] = BaseTerrainGenerator()
                elif generator_type == GeneratorType.GEOLOGY:
                    from core.geology_generator import GeologyGenerator
                    self._generator_instances[generator_type.value] = GeologyGenerator()
                elif generator_type == GeneratorType.WEATHER:
                    from core.weather_generator import WeatherSystemGenerator
                    self._generator_instances[generator_type.value] = WeatherSystemGenerator()
                elif generator_type == GeneratorType.WATER:
                    from core.water_generator import HydrologySystemGenerator
                    self._generator_instances[generator_type.value] = HydrologySystemGenerator()
                elif generator_type == GeneratorType.BIOME:
                    from core.biome_generator import BiomeClassificationSystem
                    self._generator_instances[generator_type.value] = BiomeClassificationSystem()
                elif generator_type == GeneratorType.SETTLEMENT:
                    from core.settlement_generator import SettlementGenerator
                    self._generator_instances[generator_type.value] = SettlementGenerator()

            except ImportError as e:
                self.logger.error(f"Failed to import generator for {generator_type.value}: {e}")
                return None

        return self._generator_instances.get(generator_type.value)

    def calculate_parameter_impact(self, generator_type: GeneratorType,
                                 new_parameters: Dict[str, Any]) -> Set[GeneratorType]:
        """
        Funktionsweise: Berechnet welche nachgelagerte Generatoren von Parameter-Änderungen betroffen sind
        Parameter: generator_type, new_parameters
        Return: Set von betroffenen GeneratorType
        """
        # Aktuelle Parameter aus DataManager holen
        current_params = self.get_current_parameters(generator_type)

        affected_generators = set()
        impact_config = self.impact_matrix.get(generator_type, {})

        # Parameter-Änderungen analysieren
        for param_name, new_value in new_parameters.items():
            current_value = current_params.get(param_name)

            if current_value != new_value:
                # Impact-Level bestimmen
                impact_level = self.get_parameter_impact_level(generator_type, param_name)

                if impact_level == "high_impact":
                    # High-Impact: Alle nachgelagerten Generatoren betroffen
                    affected_generators.update(self.get_downstream_generators(generator_type))
                elif impact_level == "medium_impact":
                    # Medium-Impact: Direkt abhängige Generatoren
                    affected_generators.update(self.get_direct_dependents(generator_type))

        return affected_generators

    def get_parameter_impact_level(self, generator_type: GeneratorType, param_name: str) -> str:
        """
        Funktionsweise: Bestimmt Impact-Level eines spezifischen Parameters
        Parameter: generator_type, param_name
        Return: "high_impact", "medium_impact", "low_impact"
        """
        impact_config = self.impact_matrix.get(generator_type, {})

        for impact_level, param_list in impact_config.items():
            if param_name in param_list:
                return impact_level

        return "low_impact"  # Default für unbekannte Parameter

    def get_downstream_generators(self, generator_type: GeneratorType) -> Set[GeneratorType]:
        """
        Funktionsweise: Findet alle Generatoren die (direkt oder indirekt) von diesem abhängen
        Parameter: generator_type
        Return: Set aller abhängigen Generatoren
        """
        downstream = set()

        for dependent_type, dependencies in self.dependency_tree.items():
            if generator_type in dependencies:
                downstream.add(dependent_type)
                downstream.update(self.get_downstream_generators(dependent_type))

        return downstream

    def get_direct_dependents(self, generator_type: GeneratorType) -> Set[GeneratorType]:
        """
        Funktionsweise: Findet Generatoren die direkt von diesem abhängen
        Parameter: generator_type
        Return: Set direkt abhängiger Generatoren
        """
        dependents = set()

        for dependent_type, dependencies in self.dependency_tree.items():
            if generator_type in dependencies:
                dependents.add(dependent_type)

        return dependents

    def invalidate_downstream_dependencies(self, generator_type: GeneratorType,
                                         affected_generators: Set[GeneratorType]):
        """
        Funktionsweise: Invalidiert Cache und LOD-Status für alle betroffenen Generatoren
        Parameter: generator_type, affected_generators
        """
        for affected_type in affected_generators:
            # DataManager Cache invalidieren
            self.data_manager.invalidate_cache(affected_type.value)

            # LOD-Status zurücksetzen
            self.reset_lod_status(affected_type)

            self.logger.info(f"Invalidated {affected_type.value} due to {generator_type.value} changes")

        # Signal emittieren
        affected_names = [gen.value for gen in affected_generators]
        self.dependency_invalidated.emit(generator_type.value, affected_names)

    def reset_lod_status(self, generator_type: GeneratorType):
        """
        Funktionsweise: Setzt LOD-Completion-Status für Generator zurück
        Parameter: generator_type
        """
        self.lod_completion_status[generator_type.value] = {}
        # Wird dynamisch gefüllt bei LOD-Completions

    def get_current_parameters(self, generator_type: GeneratorType) -> Dict[str, Any]:
        """
        Funktionsweise: Holt aktuelle Parameter für Generator (Placeholder - würde aus DataManager/Parameter-System geholt)
        Parameter: generator_type
        Return: Parameter-Dictionary
        """
        # Placeholder - würde normalerweise aus DataManager oder Parameter-System geholt werden
        return {}

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt Memory-Usage von allen aktiven Generationen
        Return: Memory-Usage Summary mit DataManager-Integration
        """
        return {
            "data_manager_usage": self.data_manager.get_memory_usage(),
            "active_threads": len(self.generation_threads),
            "processing_requests": len(self.processing_requests),
            "queued_requests": len(self.dependency_queue.queued_requests)
        }

    def cancel_generation(self, generator_type: str) -> bool:
        """
        Funktionsweise: Bricht laufende Generation für spezifischen Generator ab
        Parameter: generator_type
        Return: bool - Cancellation erfolgreich
        """
        cancelled_requests = []

        # Aktive Requests finden und abbrechen
        for request in list(self.processing_requests):
            if request.generator_type.value == generator_type:
                cancelled_requests.append(request)

        # Queue-Requests finden und entfernen
        queue_requests_to_cancel = [req for req in self.dependency_queue.queued_requests.values()
                                   if req.generator_type.value == generator_type]

        # Alle gefundenen Requests abbrechen
        for request in cancelled_requests + queue_requests_to_cancel:
            self.cleanup_timed_out_request(request)

        if cancelled_requests or queue_requests_to_cancel:
            self.logger.info(f"Cancelled {len(cancelled_requests + queue_requests_to_cancel)} requests for {generator_type}")
            return True

        return False

    def cleanup_resources(self):
        """
        Funktionsweise: Cleanup für alle Threads und Resources
        Aufgabe: Graceful Shutdown bei App-Beendigung
        """
        # Timer stoppen
        if hasattr(self, 'queue_resolution_timer'):
            self.queue_resolution_timer.stop()
        if hasattr(self, 'timeout_check_timer'):
            self.timeout_check_timer.stop()

        # Alle Threads stoppen
        with QMutexLocker(self.thread_mutex):
            for thread in self.generation_threads.values():
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(3000)  # 3s Timeout

        # Queues leeren
        if hasattr(self, 'dependency_queue'):
            self.dependency_queue.clear()
        self.processing_requests.clear()
        self.lod_progression_queue.clear()
        self.active_request_mapping.clear()

        self.logger.info("Generation orchestrator cleanup completed")


class GenerationThread(QThread):
    """
    Funktionsweise: Worker-Thread für einzelne Generator-Ausführung mit homogenen Signals
    Aufgabe: Background-Generation ohne UI-Blocking
    """

    generation_completed = pyqtSignal(str, dict)  # (request_id, result_data)
    generation_progress = pyqtSignal(int, str)  # (progress, message) - homogen für alle Tabs

    def __init__(self, generator_instance, generator_type: GeneratorType, lod_level: int,
                 parameters: Dict[str, Any], data_manager, request_id: str, parent=None):
        super().__init__(parent)
        self.generator_instance = generator_instance
        self.generator_type = generator_type
        self.lod_level = lod_level
        self.parameters = parameters
        self.data_manager = data_manager
        self.request_id = request_id

    def run(self):
        """
        Funktionsweise: Thread-Execution für Generator mit homogenen Signals
        """
        try:
            # Progress-Callback definieren
            def progress_callback(step_name, progress_percent, detail_message):
                self.generation_progress.emit(progress_percent, f"{step_name}: {detail_message}")

            # Generator ausführen mit BaseGenerator-Interface
            if hasattr(self.generator_instance, 'generate'):
                # BaseGenerator-Interface
                result = self.generator_instance.generate(
                    lod=self.lod_level.value,
                    dependencies={},  # Dependencies werden über DataManager automatisch geholt
                    parameters=self.parameters,
                    data_manager=self.data_manager,
                    progress=progress_callback
                )
            else:
                # Legacy-Interface fallback
                result = self.generator_instance.generate_complete(
                    lod=self.lod_level.value,
                    progress=progress_callback,
                    **self.parameters
                )

            # Success basierend auf Result
            success = result is not None

            # result_data für homogene Tab-Integration
            result_data = {
                "generator_type": self.generator_type.value,
                "lod_level": self.lod_level.value,
                "parameters_used": self.parameters,
                "timestamp": time.time(),
                "success": success,
                "generator_output": result
            }

            self.generation_completed.emit(
                self.request_id,
                self.generator_type.value,
                self.lod_level.value,
                success,
                result_data
            )

        except Exception as e:
            # Error-Handling mit homogenen Signals
            self.generation_completed.emit(
                self.request_id,
                self.generator_type.value,
                self.lod_level.value,
                False,
                {"error": str(e)}
            )