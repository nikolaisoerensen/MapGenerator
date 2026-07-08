"""
Path: gui/utils/progress_weighting.py

Gewichtetes Kostenmodell für den globalen Footer-Ladebalken.

Jede LOD-Stufe verdoppelt den Rechenaufwand einer Generator-Klasse:
    cost(generator, lod) = base_cost[generator] * 2**(lod - 1)

base_cost ist ein Platzhalter pro Generator (nicht pro Einzelklasse - die
Orchestrator-Signale melden aktuell nur generator_type + lod_level, keine
Sub-Klassen-Fortschritte). Sobald echte Benchmark-Zeiten vorliegen, hier
austauschen.
"""

from typing import Dict, Set, Tuple

# Platzhalter-Basiskosten pro Generator bei LOD 1 (relative Einheiten).
# Höherer Wert = teurer. Reihenfolge folgt der Pipeline-Abhängigkeit.
DEFAULT_BASE_COST: Dict[str, float] = {
    "terrain": 1.0,
    "geology": 1.2,
    "weather": 0.8,
    "water": 1.5,
    "biome": 1.0,
    "settlement": 0.6,
}


class WeightedProgressCalculator:
    """
    Funktionsweise: Berechnet den Gesamtfortschritt der Pipeline gewichtet
    nach Generator-Basiskosten und LOD-Verdopplung.
    Aufgabe: Liefert 0-100 Fortschrittswert für den Footer-Ladebalken.
    """

    def __init__(self, base_cost: Dict[str, float] = None, max_lod: int = 7):
        self.base_cost = dict(base_cost or DEFAULT_BASE_COST)
        self.max_lod = max_lod

    def cell_cost(self, generator_type: str, lod_level: int) -> float:
        base = self.base_cost.get(generator_type, 1.0)
        return base * (2 ** (max(lod_level, 1) - 1))

    def total_cost(self, target_lod: int, generators: Set[str] = None) -> float:
        generators = generators or set(self.base_cost.keys())
        return sum(
            self.cell_cost(gen, lod)
            for gen in generators
            for lod in range(1, target_lod + 1)
        )

    def completed_cost(self, completed: Dict[str, Dict[int, str]]) -> float:
        """
        completed: {generator_type: {lod_level: "success"|"failed"}} - typischerweise
        MapEditorWindow.tab_generation_status. Nur "success" zählt als abgeschlossen.
        """
        total = 0.0
        for generator_type, lod_status in completed.items():
            for lod_level, status in lod_status.items():
                if status == "success":
                    try:
                        lod_int = int(lod_level)
                    except (TypeError, ValueError):
                        continue
                    total += self.cell_cost(generator_type, lod_int)
        return total

    def progress_percent(self, completed: Dict[str, Dict[int, str]], target_lod: int,
                          generators: Set[str] = None) -> Tuple[int, float, float]:
        """Returns (percent_0_100, completed_cost, total_cost)."""
        total = self.total_cost(target_lod, generators)
        if total <= 0:
            return 0, 0.0, 0.0
        done = min(self.completed_cost(completed), total)
        return int(round((done / total) * 100)), done, total
