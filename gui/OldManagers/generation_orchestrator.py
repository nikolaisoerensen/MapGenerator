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

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker, QTimer
from typing import Dict, List, Set, Any, Callable
import functools
import logging
import time
from enum import Enum
import os

from gui.OldManagers.calculator_graph import CALCULATOR_GRAPH, CalculatorDispatcher
from gui.OldManagers.shader_manager import ShaderManager

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

def _protective_handler(*args, **kwargs):
    """
    Funktionsweise: Schützender Dekorator für Orchestrator-Methoden, die als QTimer-Slot,
                    Signal-Slot oder im Hintergrund laufen
    Aufgabe: Fängt jede Exception in der dekorierten Methode ab, loggt sie als WARNING und
             gibt None zurück, damit ein ungefangener Fehler nie den Qt-Event-Loop beendet (qFatal)
    Hinweis: Die Handler aus gui.utils.error_handler reichen Exceptions nach dem Logging erneut
             weiter (raise). Für QTimer- und Signal-Slots würde das einen nativen Absturz auslösen,
             deshalb kapselt der Orchestrator seine Slots bewusst mit dieser schluckenden Variante.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*call_args, **call_kwargs):
            try:
                return func(*call_args, **call_kwargs)
            except Exception as e:
                logging.getLogger(func.__module__).warning(
                    f"Unhandled exception in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator

def get_orchestrator_error_decorators():
    """
    Funktionsweise: Liefert die schützenden Dekoratoren für alle Orchestrator-Slots
    Aufgabe: threading_handler, memory_critical_handler und core_generation_handler kapseln
             QTimer-, Signal- und Hintergrund-Methoden so, dass keine ungefangene Exception
             den Qt-Event-Loop beenden kann
    Return: Tuple von drei Decorator-Fabriken
    """
    return _protective_handler, _protective_handler, _protective_handler

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
            from gui.OldManagers.data_lod_manager import calculate_max_lod_for_size
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
        state.current_lod = request.target_lod
        state.progress = 0
        state.start_time = request.timestamp

    def set_request_active(self, request: GenerationRequest):
        """Setzt Thread-Status auf 'Generating'"""
        state = self.thread_states[request.generator_type.value]
        state.status = "Generating"
        state.current_lod = request.target_lod
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


# DependencyQueue (6-Knoten-Generator-Ebene) wurde durch CalculatorDispatcher
# (siehe gui/OldManagers/calculator_graph.py, 34 Calculator-Knoten) ersetzt -
# Tracker #16 LOD-Lockstep-Umbau.


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
    # settlement.plot_nodes-spezifisch (siehe [[project-settlement-plot-physics-rebuild]]
    # Teil F) - traegt einen Snapshot des noch konvergierenden Wege-/Plot-Netzes
    # (Node-Positionen/-Typen), damit der Settlement-Tab die Physik-Iterationen
    # live mitverfolgen kann, statt nur die fertige plot_map am Ende zu sehen.
    settlement_plot_live_update = pyqtSignal(object)  # snapshot dict

    # Zusätzliche Signals für erweiterte Funktionalität
    dependency_invalidated = pyqtSignal(str, list)  # (generator_type: str, affected_generators: List[str])
    batch_generation_completed = pyqtSignal(bool, str)  # (success: bool, summary_message: str)
    queue_status_changed = pyqtSignal(list)  # (thread_status_list: List[Dict])
    # calculator_id -> {"status": "idle"/"waiting"/"calculating"/"finished"/"error",
    # "target_lod": int, "completed_lod": int, "error_message": str oder None} -
    # granularer Status ALLER 34 Calculator-Knoten für PipelineStatusPanel
    # (siehe _calculator_status_snapshot())
    calculator_status_changed = pyqtSignal(dict)

    def __init__(self, data_lod_manager, shader_manager=None):
        super().__init__()
        self.data_lod_manager = data_lod_manager
        # Ein einziger, geteilter ShaderManager (und damit ein einziger GPUWorker/GL-
        # Kontext) für alle Generatoren - vorher gab es NIRGENDS eine Injektion, jeder
        # Generator instanziierte mit shader_manager=None (Default) und lief dadurch
        # immer auf CPU-Fallback, unabhängig von jeglicher Shader-Implementierung
        # (siehe Shader-Inventur/GPU-Fundament-Fix dieser Session). map_editor.py
        # übergibt hier seinen bereits für die 3D-Ansicht erzeugten ShaderManager;
        # ohne injizierten Manager (Standalone-Nutzung/Tests) wird einer angelegt.
        self.shader_manager = shader_manager if shader_manager is not None else ShaderManager()
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
                "high_impact": ["map_seed", "map_size", "amplitude", "octaves", "frequency"],
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
                "medium_impact": ["civ_influence_decay", "terrain_factor_villages",
                                  "plot_base_spacing", "plot_civ_spacing_factor"],
                "low_impact": ["plot_height_cost_factor", "landmark_wilderness", "road_slope_to_distance_ratio"]
            }
        }

        # Zuletzt verwendete Parameter je Generator - Baseline für
        # calculate_parameter_impact()/get_current_parameters(). Ohne das
        # war get_current_parameters() ein Platzhalter, der immer {} zurückgab,
        # wodurch Parameter-Vergleiche wirkungslos waren.
        self._last_parameters: Dict[GeneratorType, Dict[str, Any]] = {}

        # Generator-Instanzen (Lazy Loading)
        self._generator_instances = {}

        # Threading-Management (jetzt pro CALCULATOR-Knoten statt pro Generator -
        # siehe CalculatorThread/_dispatch_calculator())
        self.calculator_threads = {}  # f"{calculator_id}_{lod_level}" -> CalculatorThread
        self.thread_mutex = QMutex()
        self.in_flight_calculators = set()  # calculator_id, die gerade in einem Thread laufen
        self._announced_generator_rounds = set()  # {(generator_name, lod_level)} - Dedup für generation_started
        self._calculator_errors = {}  # calculator_id -> error_message, für PipelineStatusPanel "Error"-Status

        # Globaler Calculator-Dispatcher (Tracker #16 LOD-Lockstep-Umbau) - löst das
        # bisherige lod_progression_queue/lod_completion_status/DependencyQueue-Trio
        # ab. Verwaltet ALLE 34 Calculator-Knoten aus allen 6 Generatoren
        # gemeinsam, mit echter Runden-Synchronisation (siehe
        # gui/OldManagers/calculator_graph.py). executors werden nur für den
        # synchronen run_all_rounds()-Pfad gebraucht (Tests/Fallback) - die
        # eigentliche GUI-Pipeline dispatcht jeden Knoten einzeln asynchron über
        # CalculatorThread und ruft mark_completed() aus dessen Completion-Signal.
        self.calculator_dispatcher = CalculatorDispatcher(executors=self._build_calculator_executors())

        # Parameter je Generator aus der letzten request_generation()-Anfrage -
        # Grundlage für assemble_*_data(lod, parameters) und
        # generator_instance.set_active_parameters(parameters)
        self._active_parameters: Dict[str, Dict[str, Any]] = {}

        # Aktive Requests je Generator (für Cancel/Timeout/Signal-Payload) -
        # ersetzt das bisherige DependencyQueue+processing_requests-Duo
        self.active_requests_by_generator: Dict[str, GenerationRequest] = {}

        self.state_tracker = GenerationStateTracker()

        # Request-Tracking für result_id Management
        self.active_request_mapping = {}  # request_id -> GenerationRequest

        # Performance-Monitoring
        self.generation_timings = {}
        self.memory_usage_tracking = {}

        # Setup
        self.setup_threading()
        self.setup_dependency_resolution()

        cpu_count = os.cpu_count() or 4
        if cpu_count <= 2:
            self.max_parallel_generations = cpu_count
        else:
            self.max_parallel_generations = max(1, cpu_count - 2)  # 2 Kerne frei lassen

        self.logger.info(f"Parallel generation limit: {self.max_parallel_generations}")

    def setup_threading(self):
        """
        Funktionsweise: Konfiguriert Threading-System für Background-LOD-Enhancement
        Aufgabe: Thread-Pool-Setup ohne UI-Blocking
        """
        # Thread-Management (siehe auch __init__ - hier nur nochmal explizit für
        # Klarheit, falls setup_threading() später erneut aufgerufen wird)
        self.calculator_threads = {}
        self.thread_mutex = QMutex()
        self.in_flight_calculators = set()

    def setup_dependency_resolution(self):
        """
        Funktionsweise: Konfiguriert den Runden-Advancement-Timer für den globalen
        CalculatorDispatcher
        Aufgabe: Timer für kontinuierliches Voranschreiten der Runden und
            Deadlock-Prevention
        """
        # Runden-Advancement-Timer - prüft periodisch, ob neue Calculator-Knoten
        # bereit sind (z.B. falls ein Completion-Signal aus irgendeinem Grund
        # keinen sofortigen Re-Trigger ausgelöst hat)
        self.queue_resolution_timer = QTimer()
        self.queue_resolution_timer.timeout.connect(self.advance_calculator_dispatch)
        self.queue_resolution_timer.start(2000)  # Alle 2 Sekunden

        # Timeout-Management Timer
        self.timeout_check_timer = QTimer()
        self.timeout_check_timer.timeout.connect(self.check_generation_timeouts)
        self.timeout_check_timer.start(30000)  # Alle 30 Sekunden

    @core_generation_handler("generation_request")
    def request_generation(self, generator_type, parameters=None, target_lod=None,
                           source_tab=None, priority: int = 5) -> str:
        """
        Funktionsweise: Einheitlicher Entry-Point für alle Generator-Requests von allen Tabs
        Aufgabe: Nimmt Generator-Anfragen als Keyword-Argumente entgegen und überführt sie in
                 einen internen GenerationRequest für die Dependency-Queue
        Parameter:
            generator_type - Generator-Typ als String ("terrain", "geology", "weather", "water",
                             "biome", "settlement") oder bereits als GeneratorType
            parameters     - Parameter-Dict des anfragenden Tabs (None wird zu leerem Dict)
            target_lod     - Ziel-LOD als int; None oder nicht-numerisch wird aus map_size
                             berechnet, andernfalls LOD 1
            source_tab     - Name des anfragenden Tabs für Tracking und Logging
            priority       - Priorität in der Dependency-Queue
        Return: request_id für Request-Tracking, oder None bei ungültigem Generator-Typ
        """
        # Generator-Typ robust in Enum überführen (String oder bereits Enum)
        if isinstance(generator_type, GeneratorType):
            gen_type_enum = generator_type
        else:
            try:
                gen_type_enum = GeneratorType(str(generator_type))
            except ValueError:
                self.logger.error(f"Invalid generator_type: {generator_type}")
                return None

        # Parameter absichern
        if parameters is None:
            parameters = {}

        # Ziel-LOD robust bestimmen: int direkt, numerischer String, aus map_size, sonst
        # (für alle Nicht-Terrain-Generatoren) aus der zuletzt ANGEFRAGTEN Terrain-
        # map_size, sonst Minimum.
        # Nachgelagerte Generatoren haben keinen eigenen map_size-Parameter - ihre Auflösung
        # richtet sich nach der Terrain-Auflösung. Wichtig: hier darf NICHT das bereits
        # ABGESCHLOSSENE Terrain-LOD (get_current_lod_level) verwendet werden - beim
        # Auto-Start werden alle 6 Generatoren praktisch gleichzeitig angefragt, BEVOR
        # Terrain überhaupt zu rechnen begonnen hat, d.h. dessen completed LOD ist noch
        # 0 und alle nachgelagerten Generatoren blieben für immer bei target_lod=1
        # hängen (Terrain lief dann z.B. bis LOD3, der Rest nie über LOD1 hinaus).
        # Terrain wird in der Auto-Start-Schleife zuerst angefragt, seine map_size steht
        # also in _last_parameters bereits, bevor die übrigen 5 Generatoren dran sind.
        if not isinstance(target_lod, int):
            try:
                target_lod = int(target_lod)
            except (TypeError, ValueError):
                map_size = parameters.get("map_size")
                if map_size is None:
                    map_size = self._last_parameters.get(GeneratorType.TERRAIN, {}).get("map_size")
                try:
                    from gui.OldManagers.data_lod_manager import calculate_max_lod_for_size
                    target_lod = calculate_max_lod_for_size(int(map_size))
                except (TypeError, ValueError):
                    self.logger.info("target_lod nicht bestimmbar, nutze LOD 1")
                    target_lod = 1

        # Eine noch laufende oder wartende Generation für DENSELBEN Generator wird
        # hart abgebrochen, bevor der neue Request gestellt wird - verhindert dass
        # ein alter Thread mit veralteten Parametern parallel weiterläuft und seine
        # (dann falschen) Ergebnisse noch in den DataLODManager schreibt.
        self.cancel_generation(gen_type_enum.value)

        # Internen Request bauen
        internal_request = GenerationRequest(
            gen_type_enum,
            parameters,
            target_lod,
            source_tab or gen_type_enum.value,
            priority
        )

        # Parameter-Impact-Analyse: welche NACHGELAGERTEN Generatoren betroffen sind
        affected_generators = self.calculate_parameter_impact(internal_request.generator_type,
                                                              internal_request.parameters)

        # Der angefragte Generator MUSS bei geänderten Parametern auch sich selbst
        # invalidieren - sonst sieht create_lod_sequence() weiterhin den alten
        # current_lod >= target_lod und die Generation tut schlicht nichts (das
        # war der Grund, warum ein zweiter Generate-Klick wirkungslos blieb und
        # nur ein Programmneustart wieder half). Baseline-Vergleich verhindert
        # unnötige Neu-Generierung bei unverändertem Re-Klick.
        previous_parameters = self.get_current_parameters(gen_type_enum)
        self_changed = (not previous_parameters) or any(
            previous_parameters.get(key) != value for key, value in internal_request.parameters.items()
        )
        if self_changed:
            affected_generators = set(affected_generators)
            affected_generators.add(gen_type_enum)

        # Cache-Invalidation für betroffene Generatoren (inkl. sich selbst bei Parameter-Änderung).
        # target_lod wird mitgegeben, damit betroffene Generatoren automatisch auf
        # dasselbe (ggf. neue, höhere) Ziel-LOD "mitziehen" statt auf ihrem alten
        # Ziel-LOD stehen zu bleiben (z.B. wenn eine map_size-Änderung an Terrain
        # alle 5 anderen Generatoren invalidiert - die sollen dann auch dasselbe
        # neue Final-LOD anstreben, nicht für immer beim alten hängen bleiben).
        if affected_generators:
            self.invalidate_downstream_dependencies(internal_request.generator_type, affected_generators, target_lod)

        # Baseline für künftige Parameter-Vergleiche aktualisieren
        self._last_parameters[gen_type_enum] = dict(internal_request.parameters)

        # Generator-Instanz holen und mit aktuellen Parametern versorgen - alle
        # _calc_*-Methoden lesen diese über self._current_parameters/Instanz-
        # Attribute (siehe set_active_parameters() in core/*_generator.py)
        generator_instance = self.get_generator_instance(gen_type_enum)
        if generator_instance is None:
            self.logger.error(f"No generator instance available for {gen_type_enum.value}")
            return None
        generator_instance.set_active_parameters(internal_request.parameters)

        self._active_parameters[gen_type_enum.value] = internal_request.parameters
        self.active_requests_by_generator[gen_type_enum.value] = internal_request
        self.active_request_mapping[internal_request.request_id] = internal_request

        # Alte Error-States dieses Generators löschen - ein neuer Request soll
        # eine zuvor fehlgeschlagene Knoten-Anzeige im PipelineStatusPanel
        # nicht für immer auf "Error" stehen lassen
        for cid in self._calculator_ids_for(gen_type_enum.value):
            self._calculator_errors.pop(cid, None)

        # Beim globalen Calculator-Dispatcher anfragen: setzt das Ziel-LOD für
        # ALLE Calculator-Knoten dieses Generators (siehe CalculatorDispatcher.
        # request()) - ersetzt das bisherige DependencyQueue.add_request()
        self.calculator_dispatcher.request(gen_type_enum.value, target_lod)
        self.state_tracker.set_request_queued(internal_request)

        # Queue-Status-Update emittieren
        self.emit_queue_status_update()

        self.logger.info(f"Generation queued: {internal_request.request_id} -> target_lod {target_lod}")

        # Sofort versuchen, statt auf den nächsten 2s-Timer-Tick zu warten
        QTimer.singleShot(0, self.advance_calculator_dispatch)

        return internal_request.request_id

    @threading_handler("dependency_resolution")
    def advance_calculator_dispatch(self):
        """
        Funktionsweise: Treibt den globalen CalculatorDispatcher voran - dispatcht
        alle aktuell bereiten, noch nicht laufenden Calculator-Knoten als eigene
        CalculatorThread-Instanzen
        Aufgabe: Ersetzt das bisherige resolve_dependency_queue() (6-Knoten-
        Generator-Ebene) durch echte Runden-Synchronisation auf allen 34
        Calculator-Knoten (Tracker #16 LOD-Lockstep-Umbau) - ein Knoten läuft
        erst, wenn alle seine Abhängigkeiten DIESELBE Runde erreicht haben, und
        keine Runde N+1 beginnt, bevor Runde N für alle angefragten Knoten
        abgeschlossen ist.
        """
        first_run = not getattr(self, "_first_resolution_logged", False)
        if first_run:
            self._first_resolution_logged = True
            self.logger.info("Calculator dispatch: erster Advancement-Tick")

        ready = self.calculator_dispatcher.get_next_ready_batch()
        round_n = self.calculator_dispatcher.current_round

        for calculator_id in ready:
            if calculator_id in self.in_flight_calculators:
                continue
            try:
                self._dispatch_calculator(calculator_id, round_n)
            except Exception as e:
                self.logger.error(f"Failed to dispatch calculator {calculator_id}: {e}")

        self.emit_queue_status_update()

    @threading_handler("timeout_check")
    def check_generation_timeouts(self):
        """
        Funktionsweise: Prüft und behandelt Timeout-Situationen für alle aktiven Generationen
        Aufgabe: Deadlock-Prevention durch Timeout-Management
        """
        current_time = time.time()
        timed_out_generators = []

        for generator_name, request in list(self.active_requests_by_generator.items()):
            if current_time - request.timestamp <= 300:  # 5 Minuten Timeout
                continue
            calc_ids = self._calculator_ids_for(generator_name)
            still_pending = any(
                self.calculator_dispatcher.target_lod[cid] > 0
                and self.calculator_dispatcher.completed_lod[cid] < self.calculator_dispatcher.target_lod[cid]
                for cid in calc_ids
            )
            if still_pending:
                timed_out_generators.append(generator_name)
                self.logger.warning(f"Generation for '{generator_name}' timed out")

        for generator_name in timed_out_generators:
            self.cleanup_timed_out_generator(generator_name)

        if timed_out_generators:
            self.emit_queue_status_update()

    def cleanup_timed_out_generator(self, generator_name: str):
        """
        Funktionsweise: Räumt einen timed-out Generator auf
        Aufgabe: Stoppt alle in-flight Calculator-Threads dieses Generators und
            setzt sein Ziel-LOD zurück auf 0 (kein weiterer Dispatch-Versuch)
        """
        calc_ids = set(self._calculator_ids_for(generator_name))

        with QMutexLocker(self.thread_mutex):
            for thread_key, thread in list(self.calculator_threads.items()):
                if thread.calculator_id not in calc_ids:
                    continue
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(3000)
                del self.calculator_threads[thread_key]
                self.in_flight_calculators.discard(thread.calculator_id)

        for cid in calc_ids:
            self.calculator_dispatcher.target_lod[cid] = 0
            self._calculator_errors[cid] = "Timeout"

        request = self.active_requests_by_generator.pop(generator_name, None)
        if request:
            self.active_request_mapping.pop(request.request_id, None)
            self.state_tracker.set_request_failed(request, "Timeout")

        self.logger.info(f"Cleaned up timed-out generator: {generator_name}")
        self.emit_queue_status_update()

    def _calculator_ids_for(self, generator_name: str) -> List[str]:
        """Alle Calculator-Knoten-IDs, die zu diesem Generator gehören."""
        return [cid for cid, spec in CALCULATOR_GRAPH.items() if spec.generator == generator_name]

    def _sync_state_tracker_from_dispatcher(self):
        """
        Funktionsweise: Leitet den Per-Generator-Status (für PipelineStatusPanel/
        Tabs) aus dem feingranularen Calculator-Dispatcher-Zustand ab
        Aufgabe: Die UI zeigt weiterhin 6 Generator-Zeilen, obwohl intern jetzt 34
            Calculator-Knoten einzeln fortschreiten - current_lod eines Generators
            ist das Minimum über seine Knoten (der langsamste bestimmt den
            sichtbaren Fortschritt), progress ist der Anteil abgeschlossener
            Knoten-Runden am Gesamt-Soll.
        """
        for generator_type in GeneratorType:
            gname = generator_type.value
            calc_ids = self._calculator_ids_for(gname)
            target = max((self.calculator_dispatcher.target_lod[cid] for cid in calc_ids), default=0)
            if target == 0:
                continue  # nicht angefragt - Status unverändert lassen (z.B. "Idle")

            completed_levels = [self.calculator_dispatcher.completed_lod[cid] for cid in calc_ids]
            min_completed = min(completed_levels)

            state = self.state_tracker.thread_states[gname]
            state.current_lod = min_completed

            total_steps = len(calc_ids) * target
            done_steps = sum(min(c, target) for c in completed_levels)
            state.progress = int((done_steps / total_steps) * 100) if total_steps else 0

            if min_completed >= target:
                state.status = "Completed"
            elif any(cid in self.in_flight_calculators for cid in calc_ids):
                state.status = "Generating"
            else:
                state.status = "Queued"

    def emit_queue_status_update(self):
        """
        Funktionsweise: Emittiert aktuellen Queue-Status für UI-Updates
        Aufgabe: Stellt 6-Thread-Status UND granularen 34-Knoten-Status für
            UI-Display bereit
        """
        self._sync_state_tracker_from_dispatcher()
        status_list = self.state_tracker.get_all_thread_status()
        self.queue_status_changed.emit(status_list)
        self.calculator_status_changed.emit(self._calculator_status_snapshot())

    def _calculator_status_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Funktionsweise: Baut den granularen Status ALLER 34 Calculator-Knoten
            für die PipelineStatusPanel-Anzeige
        Aufgabe: Wird bei JEDER emit_queue_status_update()-Ausführung neu
            gebaut - insbesondere sofort innerhalb von request_generation(),
            noch bevor irgendein Thread neu gestartet wird. Ein Knoten, dessen
            target_lod gerade z.B. durch eine map_size-Änderung gestiegen ist,
            zeigt dadurch sofort "waiting" statt weiter fälschlich "finished"
            für das inzwischen überholte, niedrigere LOD zu behaupten.
        Return: calculator_id -> {"status", "target_lod", "completed_lod", "error_message"}
        """
        snapshot = {}
        for calculator_id in CALCULATOR_GRAPH:
            target = self.calculator_dispatcher.target_lod[calculator_id]
            completed = self.calculator_dispatcher.completed_lod[calculator_id]
            error_message = self._calculator_errors.get(calculator_id)

            if error_message:
                status = "error"
            elif target == 0:
                status = "idle"
            elif calculator_id in self.in_flight_calculators:
                status = "calculating"
            elif completed >= target:
                status = "finished"
            else:
                status = "waiting"

            snapshot[calculator_id] = {
                "status": status,
                "target_lod": target,
                "completed_lod": completed,
                "error_message": error_message,
            }
        return snapshot

    def _build_calculator_executors(self) -> Dict[str, Callable]:
        """
        Funktionsweise: Baut die executors-Dict für den CalculatorDispatcher-
        Konstruktor (Pflicht-Parameter, siehe CalculatorDispatcher.__init__)
        Aufgabe: Nur für den synchronen run_all_rounds()-Fallback (Tests) sowie
            als Validierungs-Grundlage relevant - die echte GUI-Pipeline
            dispatcht jeden Knoten asynchron über CalculatorThread (siehe
            _dispatch_calculator()), ruft diese Executors also nie direkt auf
        """
        return {cid: self._run_calculator_sync for cid in CALCULATOR_GRAPH}

    def _run_calculator_sync(self, calculator_id: str, lod_level: int):
        """Synchron: löst Generator-Instanz + _calc_*-Methode auf und ruft sie direkt auf."""
        generator_name = CALCULATOR_GRAPH[calculator_id].generator
        generator_instance = self.get_generator_instance(GeneratorType(generator_name))
        method = getattr(generator_instance, "_calc_" + calculator_id.split(".", 1)[1])
        method(calculator_id, lod_level)

    def _dispatch_calculator(self, calculator_id: str, lod_level: int):
        """
        Funktionsweise: Startet EINEN Calculator-Knoten als eigenen Background-Thread
        Aufgabe: Asynchroner Dispatch-Baustein des LOD-Lockstep-Umbaus (Tracker
            #16) - ersetzt das bisherige "ein Thread pro (Generator, LOD)" durch
            "ein Thread pro (Calculator-Knoten, LOD)", damit z.B.
            Settlement-Phasen #28-#33 nicht mehr hinter dem GESAMTEN
            Biome-Generator in der Warteschlange stehen
        """
        generator_name = CALCULATOR_GRAPH[calculator_id].generator
        generator_type = GeneratorType(generator_name)
        generator_instance = self.get_generator_instance(generator_type)

        if not generator_instance:
            self.logger.error(f"No generator instance available for {generator_name}")
            return

        # generation_started nur EINMAL pro (Generator, Runde) emittieren -
        # PipelineStatusPanel zeigt ohnehin nur Generator-Ebene an, nicht jeden
        # einzelnen Calculator-Knoten. Reines "läuft gerade ein anderer Knoten
        # dieses Generators" reicht nicht als Dedup-Kriterium, da z.B. Terrains
        # Knoten strikt sequenziell laufen (nie gleichzeitig in-flight) - ohne
        # dieses Set würde das Signal bei jedem einzelnen Knoten neu feuern.
        round_key = (generator_name, lod_level)
        if round_key not in self._announced_generator_rounds:
            self._announced_generator_rounds.add(round_key)
            self.generation_started.emit(generator_name, lod_level)

        self.in_flight_calculators.add(calculator_id)

        thread = CalculatorThread(
            generator_instance=generator_instance,
            calculator_id=calculator_id,
            lod_level=lod_level,
            parent=self
        )
        thread.calculator_completed.connect(self.on_calculator_completed)
        # Hinweis: ansonsten kein generelles Progress-Signal verdrahtet - die
        # _calc_*-Methoden rufen zwar self._update_progress() intern auf, das
        # war aber schon im alten GenerationThread nie an ein Qt-Signal
        # angebunden (progress_callback dort war ebenfalls nie an eine
        # calculate_*-Methode übergeben) - vorbestehende Lücke, keine
        # Regression durch diesen Umbau. EINZIGE Ausnahme (bewusst eng
        # begrenzt, siehe [[project-settlement-plot-physics-rebuild]] Teil F):
        # settlement.plot_nodes bekommt einen echten Live-Callback, damit die
        # bis zu 100 Physik-Iterationen im Settlement-Tab sichtbar mitlaufen -
        # kein genereller Umbau der Progress-Infrastruktur für alle Generatoren.
        if calculator_id == "settlement.plot_nodes":
            thread.settlement_plot_live_update.connect(self.settlement_plot_live_update.emit)

        thread_key = f"{calculator_id}@{lod_level}"
        with QMutexLocker(self.thread_mutex):
            self.calculator_threads[thread_key] = thread

        thread.start()

    def on_calculator_completed(self, calculator_id: str, lod_level: int, success: bool, error_message: str):
        """
        Funktionsweise: Callback für abgeschlossenen Calculator-Knoten
        Aufgabe: Aktualisiert den globalen CalculatorDispatcher, baut bei
            vollständiger Generator-Runde das Domain-Objekt zusammen und treibt
            die nächste Runde an (Tracker #16 LOD-Lockstep-Umbau)
        """
        self.in_flight_calculators.discard(calculator_id)

        thread_key = f"{calculator_id}@{lod_level}"
        with QMutexLocker(self.thread_mutex):
            self.calculator_threads.pop(thread_key, None)

        generator_name = CALCULATOR_GRAPH[calculator_id].generator

        if success:
            self.calculator_dispatcher.mark_completed(calculator_id, lod_level)
            try:
                self._maybe_assemble_generator(generator_name, lod_level)
            except Exception as e:
                self.logger.error(f"Failed to assemble '{generator_name}' data for LOD {lod_level}: {e}")
        else:
            self.logger.error(f"Calculator '{calculator_id}' failed at LOD {lod_level}: {error_message}")
            self._calculator_errors[calculator_id] = error_message
            request = self.active_requests_by_generator.get(generator_name)
            if request:
                self.state_tracker.set_request_failed(request, error_message)
            self.generation_failed.emit(generator_name, lod_level, error_message)
            # Ziel-LOD für den GESAMTEN Generator zurücksetzen - ein
            # fehlgeschlagener Knoten macht die restliche Runde für diesen
            # Generator ohnehin unmöglich (nachgelagerte Knoten desselben
            # Generators würden sonst für immer auf ihn warten)
            for cid in self._calculator_ids_for(generator_name):
                self.calculator_dispatcher.target_lod[cid] = 0

        QTimer.singleShot(50, self.advance_calculator_dispatch)

    def _maybe_assemble_generator(self, generator_name: str, lod_level: int):
        """
        Funktionsweise: Prüft ob ALLE Calculator-Knoten eines Generators
        lod_level erreicht haben, und baut in diesem Fall das fertige
        Domain-Objekt zusammen
        Aufgabe: Ersetzt save_generation_result_to_data_lod_manager()/
            emit_final_completion_signal() aus dem alten 6-Knoten-Dispatch -
            läuft jetzt pro Generator UND pro Runde (nicht nur beim finalen
            Ziel-LOD), damit lod_progression_completed weiterhin für
            Zwischen-LODs feuert wie bisher
        """
        calc_ids = self._calculator_ids_for(generator_name)
        if not all(self.calculator_dispatcher.completed_lod[cid] >= lod_level for cid in calc_ids):
            return  # noch nicht alle Knoten dieser Runde fertig

        generator_type = GeneratorType(generator_name)
        generator_instance = self.get_generator_instance(generator_type)
        parameters = self._active_parameters.get(generator_name, {})

        assemble_method = getattr(generator_instance, f"assemble_{generator_name}_data")
        generator_output = assemble_method(lod_level, parameters)

        complete_setters = {
            "terrain": self.data_lod_manager.set_terrain_data_complete_lod,
            "geology": self.data_lod_manager.set_geology_data_complete_lod,
            "weather": self.data_lod_manager.set_weather_data_complete_lod,
            "water": self.data_lod_manager.set_water_data_complete_lod,
            "biome": self.data_lod_manager.set_biome_data_complete_lod,
            "settlement": self.data_lod_manager.set_settlement_data_complete_lod,
        }
        complete_setters[generator_name](generator_output, lod_level, parameters)
        self.logger.debug(f"{generator_name}-Daten für LOD {lod_level} abgelegt")

        request = self.active_requests_by_generator.get(generator_name)
        request_id = request.request_id if request else f"{generator_name}_{lod_level}"
        target_lod = max((self.calculator_dispatcher.target_lod[cid] for cid in calc_ids), default=0)

        if lod_level >= target_lod:
            # Finale Completion - komplette Ziel-LOD-Progression abgeschlossen
            generator_data = self.get_generator_data_from_data_lod_manager(generator_name)
            result_data = {
                "generator_type": generator_name,
                "lod_level": lod_level,
                "success": True,
                "data": generator_data,
                "source_tab": request.source_tab if request else generator_name,
                "timestamp": time.time(),
            }
            self.generation_completed.emit(request_id, result_data)
            self.active_requests_by_generator.pop(generator_name, None)
            if request:
                self.active_request_mapping.pop(request.request_id, None)
            self.logger.info(f"Final completion emitted for {generator_name} (LOD {lod_level})")
        else:
            # Zwischen-LOD - sofortige UI-Updates mit bestem verfügbarem LOD
            self.lod_progression_completed.emit(request_id, lod_level)

    def on_generation_progress(self, progress: int, message: str):
        """
        Funktionsweise: Callback für Generation-Progress mit vereinfachten Parametern
        Parameter: progress, message
        """
        # Vereinfachtes Progress-Signal emittieren (homogen für alle Tabs)
        self.generation_progress.emit(progress, message)

    def get_generator_data_from_data_lod_manager(self, generator_type: str) -> dict:
        """
        Funktionsweise: Holt Generator-Daten aus DataManager für Final-Completion
        Parameter: generator_type
        Return: dict mit allen Generator-Outputs
        """
        if generator_type == "terrain":
            return {
                "heightmap": self.data_lod_manager.get_terrain_data("heightmap"),
                "slopemap": self.data_lod_manager.get_terrain_data("slopemap"),
                "shadowmap": self.data_lod_manager.get_terrain_data("shadowmap"),
                "terrain_data_complete": self.data_lod_manager.get_terrain_data("complete")
            }
        elif generator_type == "geology":
            return {
                "rock_map": self.data_lod_manager.get_geology_data("rock_map"),
                "hardness_map": self.data_lod_manager.get_geology_data("hardness_map")
            }
        elif generator_type == "weather":
            return {
                "wind_map": self.data_lod_manager.get_weather_data("wind_map"),
                "temp_map": self.data_lod_manager.get_weather_data("temp_map"),
                "precip_map": self.data_lod_manager.get_weather_data("precip_map"),
                "humid_map": self.data_lod_manager.get_weather_data("humid_map")
            }
        elif generator_type == "water":
            return {
                "water_map": self.data_lod_manager.get_water_data("water_map"),
                "flow_map": self.data_lod_manager.get_water_data("flow_map"),
                "soil_moist_map": self.data_lod_manager.get_water_data("soil_moist_map"),
                "water_biomes_map": self.data_lod_manager.get_water_data("water_biomes_map")
            }
        elif generator_type == "biome":
            return {
                "biome_map": self.data_lod_manager.get_biome_data("biome_map"),
                "biome_map_super": self.data_lod_manager.get_biome_data("biome_map_super"),
                "super_biome_mask": self.data_lod_manager.get_biome_data("super_biome_mask")
            }
        elif generator_type == "settlement":
            return {
                "settlement_list": self.data_lod_manager.get_settlement_data("settlement_list"),
                "plot_map": self.data_lod_manager.get_settlement_data("plot_map"),
                "civ_map": self.data_lod_manager.get_settlement_data("civ_map")
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
                # data_lod_manager injizieren: alle _calc_*-Methoden lesen/schreiben
                # jetzt über DataLODManager.get_calculator_output()/
                # set_calculator_output() (siehe Tracker #16 LOD-Lockstep-Umbau) -
                # ohne diese Injektion bleibt self.data_lod_manager auf dem
                # Generator None und jeder Calculator-Aufruf schlägt fehl.
                # shader_manager injizieren, wo der jeweilige Generator-Konstruktor das
                # unterstützt (Terrain/Weather/Water/Biome/Settlement) - Geology hat
                # bisher KEINE Shader-Anbindung implementiert (siehe Shader-Inventur),
                # ihr Konstruktor kennt den Parameter noch nicht.
                if generator_type == GeneratorType.TERRAIN:
                    from core.terrain_generator import BaseTerrainGenerator
                    self._generator_instances[generator_type.value] = BaseTerrainGenerator(
                        data_lod_manager=self.data_lod_manager, shader_manager=self.shader_manager)
                elif generator_type == GeneratorType.GEOLOGY:
                    from core.geology_generator import GeologySystemGenerator
                    self._generator_instances[generator_type.value] = GeologySystemGenerator(
                        data_lod_manager=self.data_lod_manager)
                elif generator_type == GeneratorType.WEATHER:
                    from core.weather_generator import WeatherSystemGenerator
                    self._generator_instances[generator_type.value] = WeatherSystemGenerator(
                        data_lod_manager=self.data_lod_manager, shader_manager=self.shader_manager)
                elif generator_type == GeneratorType.WATER:
                    from core.water_generator import HydrologySystemGenerator
                    self._generator_instances[generator_type.value] = HydrologySystemGenerator(
                        data_lod_manager=self.data_lod_manager, shader_manager=self.shader_manager)
                elif generator_type == GeneratorType.BIOME:
                    from core.biome_generator import BiomeClassificationSystem
                    self._generator_instances[generator_type.value] = BiomeClassificationSystem(
                        data_lod_manager=self.data_lod_manager, shader_manager=self.shader_manager)
                elif generator_type == GeneratorType.SETTLEMENT:
                    from core.settlement_generator import SettlementGenerator
                    self._generator_instances[generator_type.value] = SettlementGenerator(
                        data_lod_manager=self.data_lod_manager, shader_manager=self.shader_manager)

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
                                         affected_generators: Set[GeneratorType],
                                         target_lod: int = None):
        """
        Funktionsweise: Invalidiert Cache und LOD-Status für alle betroffenen Generatoren
        Parameter: generator_type, affected_generators
        Parameter: target_lod - wenn gesetzt, ziehen betroffene Generatoren automatisch
            auf dasselbe Ziel-LOD mit (z.B. eine map_size-Änderung an Terrain hebt das
            Final-LOD an - die dadurch invalidierten 5 anderen Generatoren sollen
            dasselbe neue Final-LOD anstreben, statt für immer bei ihrem alten,
            niedrigeren Ziel-LOD zu verharren). request() kann Ziel-LODs nur anheben,
            nie absenken (siehe CalculatorDispatcher.request()), also risikolos hier.
        """
        for affected_type in affected_generators:
            # DataManager Cache invalidieren
            self.data_lod_manager.invalidate_cache_lod(affected_type.value)

            # LOD-Status zurücksetzen
            self.reset_lod_status(affected_type)

            if target_lod is not None:
                self.calculator_dispatcher.request(affected_type.value, target_lod)

            self.logger.info(f"Invalidated {affected_type.value} due to {generator_type.value} changes")

        # Signal emittieren
        affected_names = [gen.value for gen in affected_generators]
        self.dependency_invalidated.emit(generator_type.value, affected_names)

    def reset_lod_status(self, generator_type: GeneratorType):
        """
        Funktionsweise: Setzt den Rechenstand für Generator zurück
        Parameter: generator_type
        Aufgabe: Setzt completed_lod für ALLE Calculator-Knoten dieses Generators
            im globalen CalculatorDispatcher auf 0 zurück (ersetzt das alte
            lod_completion_status[generator_type.value] = {}) - über
            reset_completed(), nicht direktes Dict-Schreiben, siehe dortigen
            Docstring (sonst Endlosschleife in get_next_ready_batch() bei jeder
            Regenerierung nach einem bereits vollständig abgeschlossenen Lauf)
        """
        for cid in self._calculator_ids_for(generator_type.value):
            self.calculator_dispatcher.reset_completed(cid)

    def get_current_parameters(self, generator_type: GeneratorType) -> Dict[str, Any]:
        """
        Funktionsweise: Holt die beim letzten erfolgreich gestellten Request
        verwendeten Parameter für Generator (Baseline für Parameter-Impact-Vergleich)
        Parameter: generator_type
        Return: Parameter-Dictionary
        """
        return self._last_parameters.get(generator_type, {})

    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt Memory-Usage von allen aktiven Generationen
        Return: Memory-Usage Summary mit DataManager-Integration
        """
        return {
            "data_lod_manager_usage": self.data_lod_manager.get_memory_usage(),
            "active_threads": len(self.calculator_threads),
            "active_requests": len(self.active_requests_by_generator),
            "pending_calculators": len(self.calculator_dispatcher.get_pending_nodes()),
        }

    def cancel_generation(self, generator_type: str) -> bool:
        """
        Funktionsweise: Bricht laufende Generation für spezifischen Generator ab
        Parameter: generator_type
        Return: bool - Cancellation erfolgreich
        Aufgabe: Stoppt alle in-flight Calculator-Threads dieses Generators und
            räumt dessen Request-Tracking auf - rührt bewusst NICHT an
            target_lod/completed_lod im CalculatorDispatcher (das ist
            invalidate_downstream_dependencies()/reset_lod_status() vorbehalten,
            die nur bei tatsächlich wirkungsvollen Parameter-Änderungen greifen)
        """
        calc_ids = set(self._calculator_ids_for(generator_type))
        in_flight_for_generator = [cid for cid in calc_ids if cid in self.in_flight_calculators]
        request = self.active_requests_by_generator.get(generator_type)

        if not in_flight_for_generator and not request:
            return False

        with QMutexLocker(self.thread_mutex):
            for thread_key, thread in list(self.calculator_threads.items()):
                if thread.calculator_id not in calc_ids:
                    continue
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(3000)
                del self.calculator_threads[thread_key]
                self.in_flight_calculators.discard(thread.calculator_id)

        if request:
            self.active_requests_by_generator.pop(generator_type, None)
            self.active_request_mapping.pop(request.request_id, None)
            self.state_tracker.set_request_failed(request, "Cancelled")

        self.logger.info(f"Cancelled generation for {generator_type}")
        return True

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
            for thread in self.calculator_threads.values():
                if thread.isRunning():
                    thread.terminate()
                    thread.wait(3000)  # 3s Timeout
            self.calculator_threads.clear()

        self.in_flight_calculators.clear()
        self.active_requests_by_generator.clear()
        self.active_request_mapping.clear()

        self.logger.info("Generation orchestrator cleanup completed")


class CalculatorThread(QThread):
    """
    Funktionsweise: Worker-Thread für EINEN einzelnen Calculator-Knoten (nicht
    mehr für einen kompletten Generator-LOD-Durchlauf)
    Aufgabe: Background-Ausführung von generator_instance._calc_<name>() ohne
        UI-Blocking - Grundbaustein des LOD-Lockstep-Umbaus (Tracker #16): jeder
        der 34 Calculator-Knoten läuft als eigener, unabhängig dispatchbarer
        Thread, gesteuert vom globalen CalculatorDispatcher (siehe
        gui/OldManagers/calculator_graph.py). Die _calc_*-Methode liest ihre
        Inputs selbst über DataLODManager.get_calculator_output()/
        get_terrain_data_combined() und persistiert ihren Output selbst über
        set_calculator_output() (siehe core/*_generator.py) - gibt also nichts
        zurück, anders als das alte GenerationThread, das ein komplettes
        Domain-Objekt zurückbekam.
    """

    calculator_completed = pyqtSignal(str, int, bool, str)  # (calculator_id, lod_level, success, error_message)
    # settlement.plot_nodes-spezifisch (siehe [[project-settlement-plot-physics-rebuild]]
    # Teil F, GenerationOrchestrator.settlement_plot_live_update) - trägt einen
    # Snapshot-Payload (object), gesetzt via generator_instance.live_plot_callback.
    settlement_plot_live_update = pyqtSignal(object)

    def __init__(self, generator_instance, calculator_id: str, lod_level: int, parent=None):
        super().__init__(parent)
        self.generator_instance = generator_instance
        self.calculator_id = calculator_id
        self.lod_level = lod_level

    def run(self):
        """Funktionsweise: Thread-Execution für einen einzelnen Calculator-Knoten"""
        try:
            # Live-Fortschritts-Callback nur für settlement.plot_nodes (siehe
            # settlement_plot_live_update oben) - self.settlement_plot_live_update.emit
            # läuft hier zwar im Worker-Thread, Qt liefert die verbundene Ziel-
            # Methode (GenerationOrchestrator.settlement_plot_live_update.emit)
            # aber automatisch per QueuedConnection an den GUI-Thread aus, da
            # Sender und Empfänger in unterschiedlichen Threads leben - der
            # Snapshot-Payload wird dabei kopiert (kein geteilter, mutable
            # Zustand über die Thread-Grenze).
            if self.calculator_id == "settlement.plot_nodes" and hasattr(self.generator_instance, "live_plot_callback"):
                self.generator_instance.live_plot_callback = self._emit_live_plot_update

            method_name = "_calc_" + self.calculator_id.split(".", 1)[1]
            method = getattr(self.generator_instance, method_name)
            method(self.calculator_id, self.lod_level)

            self.calculator_completed.emit(self.calculator_id, self.lod_level, True, "")

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Calculator '{self.calculator_id}' failed at LOD {self.lod_level}: {e}")
            self.calculator_completed.emit(self.calculator_id, self.lod_level, False, str(e))
        finally:
            if self.calculator_id == "settlement.plot_nodes" and hasattr(self.generator_instance, "live_plot_callback"):
                self.generator_instance.live_plot_callback = None

    def _emit_live_plot_update(self, snapshot):
        self.settlement_plot_live_update.emit(snapshot)
