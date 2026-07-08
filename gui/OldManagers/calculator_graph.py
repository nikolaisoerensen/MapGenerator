"""
Path: gui/OldManagers/calculator_graph.py

Funktionsweise: Zentrale Graph-Definition der 29 echten Rechenklassen (Calculators)
aus docs/generation_pipeline_dependencies.md - feingranularer als die bisherige
6-Generator-Sicht des Orchestrators (dessen dependency_tree/DependencyQueue nur
"Terrain/Geology/Weather/Water/Biome/Settlement" als Knoten kennt).

Aufgabe: Grundlage für den LOD-Lockstep-Umbau (Tracker #16, siehe
docs/generation_pipeline_dependencies.md und docs/session_handover_2026-07-08.md).
Zwei Scheduler-Klassen mit unterschiedlichem Zweck:
- CalculatorRoundScheduler: verwaltet nur GENERATOR-INTERNE Teilmengen (z.B. Terrain:
  noise -> redistribution -> {slope, shadow}), genutzt von core/*_generator.py als
  Zwischenschritt beim Zerlegen jedes einzelnen Generators.
- CalculatorDispatcher: verwaltet den KOMPLETTEN Graph generatorübergreifend und
  löst den bisherigen 6-Knoten dependency_tree in generation_orchestrator.py ab -
  damit können z.B. Settlement-Phasen #28-#33 starten, ohne auf Biome zu warten
  (nur #34 plot_nodes braucht biome_map).

water.erosion_feedback (Water->Terrain-Rückkopplung, "Problem #22" in den Docs) ist
laut Dokumentation bekannt kaputt (produziert keine Wirkung) und deshalb hier
bewusst NICHT als aktiver Knoten aufgenommen - 29 aktive Calculators, nicht 30.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass(frozen=True)
class CalculatorSpec:
    calculator_id: str
    generator: str  # "terrain"/"geology"/"weather"/"water"/"biome"/"settlement" (GeneratorType.value)
    depends_on: List[str] = field(default_factory=list)
    output_keys: List[str] = field(default_factory=list)


_CALCULATOR_SPECS = [
    # --- Terrain (#1-#4) ---
    CalculatorSpec("terrain.noise", "terrain", [], ["noise_grid"]),
    CalculatorSpec("terrain.redistribution", "terrain", ["terrain.noise"], ["heightmap"]),
    CalculatorSpec("terrain.slope", "terrain", ["terrain.redistribution"], ["slopemap"]),
    CalculatorSpec("terrain.shadow", "terrain", ["terrain.redistribution"], ["shadowmap"]),

    # --- Geology (#5-#10) ---
    CalculatorSpec("geology.classify_elevation", "geology", ["terrain.redistribution"], ["rock_map_raw"]),
    CalculatorSpec("geology.slope_hardening", "geology",
                   ["geology.classify_elevation", "terrain.slope"], ["rock_map_hardened"]),
    CalculatorSpec("geology.blend_zones", "geology", ["geology.slope_hardening"], ["rock_map_blended"]),
    CalculatorSpec("geology.tectonic_deformation", "geology",
                   ["geology.blend_zones"], ["rock_map_deformed", "height_delta"]),
    # Nicht in docs/generation_pipeline_dependencies.md aufgeführt (dort undokumentiert
    # geblieben) - RockTypeClassifier.apply_faceted_boundaries(), letzter Formungsschritt
    # vor Mass-Conservation (facettiert auch die Tektonik-Effekte, siehe
    # core/geology_generator.py Docstring).
    CalculatorSpec("geology.faceted_boundaries", "geology",
                   ["geology.tectonic_deformation"], ["rock_map_faceted"]),
    CalculatorSpec("geology.mass_conservation", "geology", ["geology.faceted_boundaries"], ["rock_map"]),
    CalculatorSpec("geology.hardness", "geology", ["geology.mass_conservation"], ["hardness_map"]),

    # --- Weather (#11-#14) ---
    CalculatorSpec("weather.temperature", "weather",
                   ["terrain.redistribution", "terrain.shadow"], ["temp_map"]),
    CalculatorSpec("weather.wind", "weather",
                   ["terrain.redistribution", "weather.temperature", "terrain.shadow"], ["wind_map"]),
    CalculatorSpec("weather.humidity", "weather",
                   ["terrain.redistribution", "weather.temperature", "weather.wind"], ["humid_map"]),
    CalculatorSpec("weather.precipitation", "weather",
                   ["weather.humidity", "weather.temperature", "weather.wind", "terrain.redistribution"],
                   ["precip_map"]),

    # --- Water (#15-#21, #22 erosion_feedback bewusst ausgeschlossen - siehe Docstring) ---
    CalculatorSpec("water.lake_detection", "water", ["terrain.redistribution"], ["lake_map"]),
    CalculatorSpec("water.flow_network", "water",
                   ["terrain.redistribution", "weather.precipitation", "water.lake_detection"],
                   ["flow_accumulation", "water_biomes_map"]),
    CalculatorSpec("water.steepest_descent", "water", ["terrain.redistribution"], ["flow_directions"]),
    CalculatorSpec("water.manning_flow", "water",
                   ["water.flow_network", "terrain.slope", "terrain.redistribution"],
                   ["flow_speed", "cross_section"]),
    CalculatorSpec("water.erosion_sedimentation", "water",
                   ["water.flow_network", "water.manning_flow", "water.steepest_descent", "geology.hardness"],
                   ["erosion_map", "sedimentation_map"]),
    CalculatorSpec("water.soil_moisture", "water", ["water.flow_network"], ["soil_moist_map"]),
    CalculatorSpec("water.evaporation", "water",
                   ["weather.temperature", "weather.wind", "weather.humidity", "water.flow_network"],
                   ["evaporation_map"]),

    # --- Biome (#23-#27) ---
    CalculatorSpec("biome.base_classification", "biome",
                   ["terrain.redistribution", "weather.temperature", "weather.precipitation",
                    "water.soil_moisture"], ["base_biome_map"]),
    CalculatorSpec("biome.super_override", "biome",
                   ["terrain.redistribution", "weather.temperature", "water.flow_network",
                    "water.soil_moisture"], ["super_biome_mask", "super_biome_probabilities"]),
    CalculatorSpec("biome.integrate_layers", "biome",
                   ["biome.base_classification", "biome.super_override"], ["biome_map"]),
    CalculatorSpec("biome.supersampling", "biome", ["biome.integrate_layers"], ["biome_map_super"]),
    CalculatorSpec("biome.climate_classification", "biome",
                   ["weather.temperature", "weather.precipitation"], ["climate_classification"]),

    # --- Settlement (#28-#34) ---
    CalculatorSpec("settlement.suitability", "settlement",
                   ["terrain.redistribution", "terrain.slope", "water.flow_network"],
                   ["combined_suitability_map"]),
    CalculatorSpec("settlement.settlements", "settlement",
                   ["settlement.suitability", "terrain.redistribution"], ["settlement_list"]),
    CalculatorSpec("settlement.pathfinding", "settlement",
                   ["settlement.settlements", "terrain.slope"], ["roads"]),
    CalculatorSpec("settlement.roadsites", "settlement", ["settlement.pathfinding"], ["roadsite_list"]),
    CalculatorSpec("settlement.civ_influence", "settlement",
                   ["terrain.redistribution", "terrain.slope", "settlement.settlements",
                    "settlement.pathfinding", "settlement.roadsites"], ["civ_map"]),
    CalculatorSpec("settlement.landmarks", "settlement",
                   ["settlement.civ_influence", "terrain.redistribution", "terrain.slope"], ["landmark_list"]),
    CalculatorSpec("settlement.plot_nodes", "settlement",
                   ["settlement.civ_influence", "settlement.settlements", "terrain.redistribution",
                    "biome.integrate_layers"], ["plot_nodes", "plots", "plot_map"]),
]

CALCULATOR_GRAPH: Dict[str, CalculatorSpec] = {spec.calculator_id: spec for spec in _CALCULATOR_SPECS}

# Tatsächliche Zählung: 34 durchnummerierte Schritte (#1-#34) aus
# docs/generation_pipeline_dependencies.md, minus dem bekannt kaputten #22
# (water.erosion_feedback, oben ausgeschlossen), plus geology.faceted_boundaries
# (in den Docs nicht erfasst, aber real im Code vorhanden - siehe core/geology_generator.py)
# = 34 aktive Knoten.
assert len(CALCULATOR_GRAPH) == 34, f"Erwartet 34 aktive Calculators, gefunden {len(CALCULATOR_GRAPH)}"


class CalculatorRoundScheduler:
    """
    Führt eine Teilmenge von CALCULATOR_GRAPH-Knoten rundenweise aus - pro Runde
    genau ein LOD-Level, in Abhängigkeits-Reihenfolge. Abhängigkeiten AUSSERHALB
    der verwalteten Teilmenge (z.B. geology.classify_elevation haengt von
    terrain.redistribution ab, aber dieser Scheduler verwaltet evtl. nur die
    Geology-Knoten) gelten als bereits erfüllt - der Aufrufer ist dafür
    verantwortlich, den Executor erst zu starten, wenn diese externen
    Abhängigkeiten tatsächlich für das gewünschte LOD vorliegen (das übernimmt
    heute weiterhin generation_orchestrator.py's bestehende Generator-Dispatch-
    Logik, solange nicht alle 6 Generatoren zerlegt sind).
    """

    def __init__(self, calculator_ids: List[str], executors: Dict[str, Callable[[dict], None]]):
        unknown = [cid for cid in calculator_ids if cid not in CALCULATOR_GRAPH]
        if unknown:
            raise ValueError(f"Unbekannte Calculator-IDs: {unknown}")
        missing_executors = [cid for cid in calculator_ids if cid not in executors]
        if missing_executors:
            raise ValueError(f"Fehlende Executor-Funktionen für: {missing_executors}")

        self.calculator_ids = list(calculator_ids)
        self.executors = executors
        self.completed_lod: Dict[str, int] = {cid: 0 for cid in calculator_ids}

    def run_round(self, target_lod: int, context: dict) -> dict:
        """
        Führt alle verwalteten Calculators für GENAU ein LOD-Level aus.
        context: gemeinsames Dict, aus dem Executor-Funktionen ihre Eingaben lesen
        und in das sie ihre Outputs schreiben (Data-Key -> Wert). Jeder Executor
        bekommt exakt dieses eine context-Dict übergeben.
        Returns: aktualisiertes context-Dict.
        """
        remaining = set(self.calculator_ids)
        while remaining:
            ready = [
                cid for cid in remaining
                if all(
                    dep not in self.calculator_ids or self.completed_lod[dep] >= target_lod
                    for dep in CALCULATOR_GRAPH[cid].depends_on
                )
            ]
            if not ready:
                raise RuntimeError(
                    f"Zirkuläre oder von außerhalb dieser Scheduler-Instanz "
                    f"unerfüllbare Abhängigkeit unter verbleibenden Knoten: {remaining}")

            for cid in ready:
                self.executors[cid](context)
                self.completed_lod[cid] = target_lod
                remaining.remove(cid)

        return context


class CalculatorDispatcher:
    """
    Globaler Runden-Scheduler über den KOMPLETTEN CALCULATOR_GRAPH (alle 6
    Generatoren gemeinsam) - löst den bisherigen 6-Knoten dependency_tree in
    GenerationOrchestrator ab (Tracker #16 / LOD-Lockstep-Umbau).

    Kernidee: "Runde N" heißt nicht "alle Knoten werden gleichzeitig auf LOD N
    gebracht", sondern "alle Knoten, die für LOD N bereit sind/werden, laufen -
    in Abhängigkeits-Kaskade - bis nichts mehr für LOD N bereit wird, DANN erst
    beginnt Runde N+1". Ein Knoten wie geology.classify_elevation kann also
    innerhalb derselben Runde laufen wie terrain.redistribution, sobald diese
    fertig ist (kein künstliches Warten auf einen globalen Rundenabschluss) -
    entscheidend ist nur, dass KEIN Knoten LOD N+1 erreicht, bevor nicht jeder
    für LOD N erreichbare Knoten sein LOD N abgeschlossen hat.

    Nutzt bewusst NUR die primitiven Bausteine (get_ready_nodes/mark_completed),
    nicht einen synchronen "alles durchlaufen"-Loop als einzige Schnittstelle -
    so kann die Anbindung in GenerationOrchestrator (Task 18) jeden bereiten
    Knoten als eigenen asynchronen Thread dispatchen und mark_completed() erst
    aus dessen Qt-Completion-Signal aufrufen, statt blockierend zu warten.
    run_all_rounds() ist eine synchrone Convenience-Variante für Tests und
    einfache Nicht-GUI-Nutzung.
    """

    def __init__(self, executors: Dict[str, Callable[[str, int], None]]):
        """
        executors: calculator_id -> callable(calculator_id, lod_level) -> None.
        Jeder Executor ist dafür verantwortlich, seinen Output selbst über
        DataLODManager.set_calculator_output() zu persistieren (siehe
        core/*_generator.py _calc_*-Methoden nach der Umstellung in den
        Tasks 12-17).
        """
        missing = [cid for cid in CALCULATOR_GRAPH if cid not in executors]
        if missing:
            raise ValueError(f"Fehlende Executor-Funktionen für: {missing}")

        self.executors = executors
        self.completed_lod: Dict[str, int] = {cid: 0 for cid in CALCULATOR_GRAPH}
        self.target_lod: Dict[str, int] = {cid: 0 for cid in CALCULATOR_GRAPH}  # 0 = nicht angefragt
        self._next_round = 1  # Fortsetzungspunkt für wiederholte run_all_rounds()-Aufrufe

    def request(self, generator: str, target_lod: int):
        """
        Setzt das Ziel-LOD für alle Calculator-Knoten EINES Generators (z.B. wenn
        ein Tab "Generieren" klickt oder Auto-Start beim App-Start alle 6 anfragt).
        Ein Knoten mit target_lod=0 gilt als nicht angefragt und wird vom
        Scheduler übersprungen (bleibt ewig "nicht bereit" für abhängige Knoten).
        """
        for cid, spec in CALCULATOR_GRAPH.items():
            if spec.generator == generator:
                self.target_lod[cid] = max(self.target_lod[cid], target_lod)

    def get_ready_nodes(self, round_n: int) -> List[str]:
        """
        Alle Knoten, die JETZT für Runde round_n ausgeführt werden können:
        eigenes target_lod >= round_n, noch nicht auf round_n abgeschlossen, und
        alle Abhängigkeiten haben round_n bereits erreicht. Ein Knoten, dessen
        Generator nie angefragt wurde (target_lod=0 bei einer Abhängigkeit),
        wird nie bereit - das ist beabsichtigt (kein Auto-Request von Upstream-
        Generatoren hier, das entscheidet der Aufrufer/Auto-Start explizit).
        """
        ready = []
        for cid, spec in CALCULATOR_GRAPH.items():
            if self.target_lod[cid] < round_n:
                continue
            if self.completed_lod[cid] >= round_n:
                continue
            if all(self.completed_lod[dep] >= round_n for dep in spec.depends_on):
                ready.append(cid)
        return ready

    def mark_completed(self, calculator_id: str, round_n: int):
        """Markiert einen Knoten als für round_n abgeschlossen (nach Thread-Completion)."""
        self.completed_lod[calculator_id] = max(self.completed_lod[calculator_id], round_n)

    def reset_completed(self, calculator_id: str):
        """
        Setzt einen Knoten auf completed_lod=0 zurück (Parameter-Änderung/
        Invalidierung, siehe GenerationOrchestrator.reset_lod_status()). MUSS
        über diese Methode laufen statt completed_lod[...] direkt zu schreiben:
        get_next_ready_batch() lässt _next_round unbegrenzt weiterlaufen, sobald
        einmal is_fully_done() erreicht wurde (_next_round steht dann auf
        höchstem target_lod + 1). Ein Knoten, dessen completed_lod danach ohne
        Rewind auf 0 zurückfällt, hat für IMMER target_lod < round_n - er wird
        von der "bereits erledigt"-Prüfung nie wieder erfasst, obwohl
        is_fully_done() ihn weiterhin als offen zählt: get_next_ready_batch()
        läuft dann in eine echte Endlosschleife auf dem aufrufenden (GUI-)Thread.
        Das rundenweise Vorlaufen weiter unten holt den zurückgesetzten Knoten
        günstig wieder ein, da bereits abgeschlossene Runden anderer Knoten
        sofort übersprungen werden (completed_lod >= round_n).
        """
        self.completed_lod[calculator_id] = 0
        self._next_round = 1

    def get_pending_nodes(self) -> List[str]:
        """Alle angefragten, aber noch nicht auf ihr Ziel-LOD gebrachten Knoten."""
        return [
            cid for cid in CALCULATOR_GRAPH
            if self.target_lod[cid] > 0 and self.completed_lod[cid] < self.target_lod[cid]
        ]

    def is_fully_done(self) -> bool:
        """True, wenn jeder angefragte Knoten sein Ziel-LOD erreicht hat."""
        return len(self.get_pending_nodes()) == 0

    @property
    def current_round(self) -> int:
        """Die Runde, die get_next_ready_batch() als nächstes zu vervollständigen versucht."""
        return self._next_round

    def get_next_ready_batch(self) -> List[str]:
        """
        Nicht-blockierende Kernmethode für asynchronen Dispatch (siehe
        GenerationOrchestrator.advance_calculator_dispatch(), Task 18): advanced
        self._next_round über bereits vollständig abgeschlossene Runden hinweg
        und gibt die aktuell bereiten Knoten für die (ggf. neu erreichte)
        nächste Runde zurück, OHNE sie selbst auszuführen - das übernimmt der
        Aufrufer (z.B. über eigene QThreads), der danach mark_completed()
        aufruft und diese Methode erneut abfragt (current_round liefert die
        Runde, für die die zurückgegebenen IDs bereit sind).

        Gibt eine leere Liste zurück, wenn entweder alles erledigt ist
        (is_fully_done() prüfen) oder aktuell nichts Neues bereit ist (z.B. weil
        alles gerade angeforderte bereits läuft/in einem Thread hängt, oder ein
        echtes Deadlock vorliegt, weil eine Abhängigkeit nie angefragt wurde).
        """
        round_n = self._next_round
        while True:
            still_pending_for_round = [
                cid for cid, spec in CALCULATOR_GRAPH.items()
                if self.target_lod[cid] >= round_n and self.completed_lod[cid] < round_n
            ]

            if not still_pending_for_round:
                # Runde round_n ist (soweit überhaupt angefragt) vollständig erledigt
                round_n += 1
                self._next_round = round_n
                if self.is_fully_done():
                    return []
                continue

            return self.get_ready_nodes(round_n)

    def run_all_rounds(self, max_rounds: int = 1000):
        """
        Synchrone Convenience-Variante: dispatcht Runde für Runde blockierend,
        bis kein Knoten mehr Fortschritt machen kann. Für Tests/einfache
        Nicht-GUI-Nutzung - die echte GUI-Anbindung (Task 18) nutzt
        get_next_ready_batch()/mark_completed() direkt für asynchronen
        Thread-Dispatch, ohne zu blockieren.

        Wiederholt aufrufbar (z.B. erst request("terrain", 1), run_all_rounds(),
        dann später request("terrain", 2), run_all_rounds() erneut, wenn neue
        Ziele nachträglich gesetzt werden) - merkt sich die zuletzt erreichte
        Runde in self._next_round, damit ein erneuter Aufruf nicht fälschlich
        sofort abbricht, nur weil Runde 1 schon vollständig erledigt war.
        """
        iterations = 0

        while iterations < max_rounds:
            iterations += 1

            ready = self.get_next_ready_batch()
            if not ready:
                break

            round_n = self._next_round
            for cid in ready:
                self.executors[cid](cid, round_n)
                self.mark_completed(cid, round_n)

        if not self.is_fully_done():
            raise RuntimeError(
                f"run_all_rounds beendet ohne alle Ziele zu erreichen (evtl. max_rounds zu "
                f"niedrig oder nie angefragte Abhängigkeit): {self.get_pending_nodes()}")
