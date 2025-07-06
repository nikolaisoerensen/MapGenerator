#!/usr/bin/env python3
"""
Path: MapGenerator/gui/widgets/base_map_canvas.py
__init__.py existiert in "widgets"

Basis-Klasse für alle Map-Canvas Widgets
Eliminiert Code-Duplikation bei Matplotlib-Integration
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class BaseMapCanvas(QWidget):
    """
    Funktionsweise: Basis-Klasse für alle Karten-Darstellungen
    - Stellt gemeinsame Matplotlib-Integration bereit
    - Vereinheitlicht Achsen-Setup und Canvas-Handling
    - Reduziert Code-Duplikation zwischen allen Tabs
    - Subklassen müssen nur update_map() implementieren
    """

    def __init__(self, figsize=(8, 8), title="Map Preview"):
        """
        Args:
            figsize (tuple): Größe der Matplotlib Figure
            title (str): Standard-Titel für die Karte
        """
        super().__init__()
        self.default_title = title
        self.init_ui(figsize)

    def init_ui(self, figsize):
        """
        Funktionsweise: Erstellt das grundlegende Canvas-Layout
        - Erstellt Matplotlib Figure und Canvas
        - Setzt Layout auf
        - Initialisiert Achsen mit Standardwerten
        """
        layout = QVBoxLayout()

        # Matplotlib Figure und Canvas erstellen
        self.figure = Figure(figsize=figsize, facecolor='white')
        self.canvas = FigureCanvas(self.figure)

        # Standard-Achse erstellen
        self.ax = self.figure.add_subplot(111)
        self.setup_axes()

        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_axes(self):
        """
        Funktionsweise: Konfiguriert die Standard-Achsen-Eigenschaften
        - Setzt Grenzen und Labels
        - Aktiviert Gitter
        - Wird von Subklassen überschrieben für spezielle Layouts
        """
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title(self.default_title)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)

    def clear_and_setup(self):
        """
        Funktionsweise: Löscht die Achse und stellt sie wieder her
        - Wird vor jeder Kartenaktualisierung aufgerufen
        - Verhindert Überlagerung von alten und neuen Plots
        """
        self.ax.clear()
        self.setup_axes()

    def draw(self):
        """
        Funktionsweise: Zeichnet die Karte neu
        - Wrapper um matplotlib canvas.draw()
        - Kann für Performance-Optimierungen erweitert werden
        """
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"Fehler beim Zeichnen der Karte: {e}")

    def update_map(self, **params):
        """
        Funktionsweise: Abstract method für Karten-Updates
        - Muss von Subklassen implementiert werden
        - Erhält alle Parameter als kwargs
        - Sollte clear_and_setup() aufrufen und mit draw() enden
        """
        raise NotImplementedError("Subklassen müssen update_map() implementieren")

    def set_title(self, title):
        """Setzt den Titel der Karte"""
        self.ax.set_title(title)
        self.draw()

    def add_legend(self, handles, labels, **kwargs):
        """
        Funktionsweise: Fügt eine Legende hinzu
        - Standardisierte Legende-Positionierung
        - kwargs werden an matplotlib legend() weitergegeben
        """
        default_kwargs = {
            'loc': 'upper right',
            'fontsize': 8,
            'framealpha': 0.9
        }
        default_kwargs.update(kwargs)

        self.ax.legend(handles, labels, **default_kwargs)


class MultiPlotMapCanvas(BaseMapCanvas):
    """
    Funktionsweise: Erweiterte Canvas-Klasse für mehrere Subplots
    - Für komplexere Visualisierungen (z.B. Weather Tab)
    - Verwaltet mehrere Achsen automatisch
    """

    def __init__(self, subplot_config=(1, 2), figsize=(12, 6), titles=None):
        """
        Args:
            subplot_config (tuple): (rows, cols) für Subplot-Layout
            figsize (tuple): Größe der Figure
            titles (list): Liste der Titel für jeden Subplot
        """
        self.subplot_config = subplot_config
        self.subplot_titles = titles or []
        super().__init__(figsize, "Multi-Plot Map")

    def setup_axes(self):
        """
        Funktionsweise: Erstellt mehrere Subplots
        - Basierend auf subplot_config
        - Speichert Achsen in self.axes Liste
        """
        rows, cols = self.subplot_config
        self.axes = []

        for i in range(rows * cols):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            # Titel setzen wenn verfügbar
            if i < len(self.subplot_titles):
                ax.set_title(self.subplot_titles[i])

            self.axes.append(ax)

        # Erstes Subplot als Haupt-Achse für Kompatibilität
        if self.axes:
            self.ax = self.axes[0]

    def clear_and_setup(self):
        """Löscht alle Subplots und erstellt sie neu"""
        self.figure.clear()
        self.setup_axes()

    def get_subplot(self, index):
        """
        Funktionsweise: Gibt ein spezifisches Subplot zurück
        Args:
            index (int): Index des gewünschten Subplots
        Returns:
            matplotlib.axes.Axes: Das angeforderte Subplot
        """
        if 0 <= index < len(self.axes):
            return self.axes[index]
        return self.ax  # Fallback zum Haupt-Subplot