"""
Path: gui/widgets/pipeline_status_panel.py

PipelineStatusPanel ist die feste linke Spalte des Map-Editors. Zeigt den
Status ALLER 34 Calculator-Knoten aus CALCULATOR_GRAPH (nicht mehr nur der
6 Generatoren) einzeln an, gruppiert nach Generator, in einer scrollbaren
Liste - damit sichtbar ist, welcher einzelne Rechenschritt (heightmap,
slopemap, flow_accumulation, ...) gerade läuft, statt nur des groben
6-Zeilen-Generator-Status.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QScrollArea

from gui.widgets.widgets import StatusIndicator
from gui.OldManagers.calculator_graph import CALCULATOR_GRAPH

GENERATOR_ORDER = ["terrain", "geology", "weather", "water", "biome", "settlement"]


class PipelineStatusPanel(QWidget):
    """
    Funktionsweise: Feste Spalte mit einem StatusIndicator pro Calculator-
    Knoten (34 Knoten aus CALCULATOR_GRAPH), gruppiert nach Generator und in
    einer QScrollArea, da 34 Zeilen die feste Spaltenbreite sonst sprengen.
    Aufgabe: Zeigt Idle/Waiting/Calculating/Finished/Error je Knoten an,
        unabhängig vom aktuell gewählten Haupt-Tab.
    Kommunikation: Keine eigenen Signals - wird ausschließlich über
        apply_snapshot() aus GenerationOrchestrator.calculator_status_changed
        aktualisiert (siehe MapEditorWindow._on_calculator_status_changed).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.indicators = {}  # calculator_id -> StatusIndicator
        self._setup_ui()

    def _setup_ui(self):
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(6)

        title = QLabel("Pipeline-Status")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        outer_layout.addWidget(title)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        outer_layout.addWidget(separator)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(3)

        nodes_by_generator = {}
        for calculator_id, spec in CALCULATOR_GRAPH.items():
            nodes_by_generator.setdefault(spec.generator, []).append((calculator_id, spec))

        for generator in GENERATOR_ORDER:
            nodes = nodes_by_generator.get(generator, [])
            if not nodes:
                continue

            header = QLabel(generator.capitalize())
            header.setStyleSheet(
                "font-size: 11px; font-weight: bold; color: #7f8c8d; margin-top: 4px;")
            content_layout.addWidget(header)

            for calculator_id, spec in nodes:
                label = "/".join(spec.output_keys) if spec.output_keys else calculator_id
                indicator = StatusIndicator(label)
                indicator.setToolTip(calculator_id)
                indicator.set_unknown()
                self.indicators[calculator_id] = indicator
                content_layout.addWidget(indicator)

        content_layout.addStretch()
        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)

        self.setLayout(outer_layout)

    def apply_snapshot(self, snapshot: dict):
        """
        Aktualisiert alle Indicator-Zeilen aus einem Status-Snapshot.
        Parameter: snapshot - calculator_id -> {"status": "idle"/"waiting"/
            "calculating"/"finished"/"error", "target_lod": int,
            "completed_lod": int, "error_message": str oder None}.
            Kommt aus GenerationOrchestrator._calculator_status_snapshot(),
            das bei JEDER emit_queue_status_update()-Ausführung neu gebaut
            wird - insbesondere sofort innerhalb von request_generation(),
            bevor irgendein Thread neu startet. Ein Knoten, dessen Ziel-LOD
            gerade gestiegen ist (z.B. durch eine map_size-Änderung), springt
            dadurch sofort von "Finished" auf "Waiting" um, statt fälschlich
            auf dem alten "Finished"-Stand für das inzwischen überholte LOD
            stehen zu bleiben.
        """
        for calculator_id, info in snapshot.items():
            indicator = self.indicators.get(calculator_id)
            if not indicator:
                continue

            status = info.get("status", "idle")
            target = info.get("target_lod", 0)
            completed = info.get("completed_lod", 0)

            if status == "calculating":
                indicator.set_pending(f"Calculating (LOD {target})")
            elif status == "finished":
                indicator.set_success(f"Finished (LOD {completed})")
            elif status == "error":
                indicator.set_error(info.get("error_message") or "Error")
            elif status == "waiting":
                indicator.set_warning(f"Waiting (LOD {completed} → {target})")
            else:  # "idle" - nicht angefragt
                indicator.set_unknown()

    def reset_all(self):
        for indicator in self.indicators.values():
            indicator.set_unknown()
