"""
Path: gui/widgets/pipeline_status_panel.py

PipelineStatusPanel ist die feste linke Spalte des Map-Editors. Sie zeigt
den globalen Pipeline-Status aller Generatoren (unabhängig vom aktuell
gewählten Haupt-Tab) und bleibt beim Tab-Wechsel unverändert sichtbar.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame

from gui.widgets.widgets import StatusIndicator

PIPELINE_GENERATORS = ["terrain", "geology", "weather", "water", "biome", "settlement"]


class PipelineStatusPanel(QWidget):
    """
    Funktionsweise: Feste Spalte mit einem StatusIndicator pro Generator.
    Aufgabe: Zeigt Queued/Calculating/Finished-Status je Generator, damit
    der Fortschritt der Gesamt-Pipeline unabhängig vom aktiven Tab sichtbar bleibt.
    Kommunikation: Keine Signals - nur Display, wird von MapEditorWindow aktualisiert.
    """

    def __init__(self, generators=None, parent=None):
        super().__init__(parent)
        self.generators = generators or PIPELINE_GENERATORS
        self.indicators = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("Pipeline-Status")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        layout.addWidget(separator)

        for generator_type in self.generators:
            indicator = StatusIndicator(generator_type.capitalize())
            indicator.set_unknown()
            self.indicators[generator_type] = indicator
            layout.addWidget(indicator)

        layout.addStretch()
        self.setLayout(layout)

    def set_queued(self, generator_type: str):
        if generator_type in self.indicators:
            self.indicators[generator_type].set_warning("Queued")

    def set_calculating(self, generator_type: str, lod_level=None):
        if generator_type in self.indicators:
            message = f"Calculating (LOD {lod_level}...)" if lod_level is not None else "Calculating..."
            self.indicators[generator_type].set_pending(message)

    def set_finished(self, generator_type: str, lod_level=None):
        if generator_type in self.indicators:
            message = f"Finished (LOD {lod_level})" if lod_level is not None else "Finished"
            self.indicators[generator_type].set_success(message)

    def set_failed(self, generator_type: str, message: str = "Failed"):
        if generator_type in self.indicators:
            self.indicators[generator_type].set_error(message)

    def reset_all(self):
        for indicator in self.indicators.values():
            indicator.set_unknown()
