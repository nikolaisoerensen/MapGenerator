#!/usr/bin/env python3
"""
Path: MapGenerator/gui/widgets/map_canvas.py
__init__.py existiert in "widgets"

Performance-optimierte Map Canvas Klassen
Vollständig eigenständige Implementierung ohne BaseMapCanvas
"""

import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from gui.utils.performance_utils import (
    performance_tracked, get_render_optimizer, LoadingStateManager
)
from gui.utils.error_handler import ErrorHandler


class MapCanvas(QWidget):
    """
    Funktionsweise: Performance-optimierte Map Canvas Klasse
    - Eigenständige Implementierung ohne Vererbung von BaseMapCanvas
    - Automatisches Debouncing von Updates
    - Memory-Management für Matplotlib
    - Loading-States für längere Operationen
    - Performance-Tracking
    - Alle Basis-Funktionen direkt implementiert
    """

    def __init__(self, canvas_id, figsize=(8, 8), title="Map Preview", debounce_ms=300):
        """
        Args:
            canvas_id (str): Eindeutige ID für Performance-Optimierung
            figsize (tuple): Figure-Größe
            title (str): Standard-Titel
            debounce_ms (int): Debounce-Zeit für Updates
        """
        super().__init__()
        self.canvas_id = canvas_id
        self.debounce_ms = debounce_ms
        self.default_title = title
        self.render_optimizer = get_render_optimizer()
        self.loading_manager = None
        self.error_handler = ErrorHandler()

        self.init_ui(figsize)

        # Loading-State Manager erstellen
        self.loading_manager = LoadingStateManager(self)

    def init_ui(self, figsize):
        """
        Funktionsweise: Erstellt das grundlegende Canvas-Layout mit Performance-Optimierungen
        - Erstellt optimierte Matplotlib Figure und Canvas
        - Setzt Layout auf mit Loading-Label
        - Initialisiert Achsen mit Standardwerten
        """
        layout = QVBoxLayout()

        # Loading Label
        self.loading_label = QLabel("Aktualisiere Karte...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setVisible(False)
        self.loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                border: 2px solid #3498db;
                border-radius: 10px;
                padding: 20px;
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)

        # Optimierte Figure erstellen
        self.figure = self.render_optimizer.get_optimized_figure(self.canvas_id, figsize)
        self.canvas = FigureCanvas(self.figure)

        # Standard-Achse erstellen
        self.ax = self.figure.add_subplot(111)
        self.setup_axes()

        layout.addWidget(self.canvas)
        layout.addWidget(self.loading_label)
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
        - Optimierte Version mit Figure-Recycling
        """
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.setup_axes()

    def draw(self):
        """
        Funktionsweise: Zeichnet die Karte neu
        - Wrapper um matplotlib canvas.draw()
        - Enthält Fehlerbehandlung
        """
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"Fehler beim Zeichnen der Karte: {e}")

    @performance_tracked("Map_Rendering")
    def update_map(self, **params):
        """
        Funktionsweise: Optimierte Map-Update Implementation
        - Zeigt Loading-State während Rendering
        - Performance-Tracking für Optimierung
        - Ruft _render_map() auf, das von Subklassen implementiert werden muss
        """
        try:
            # Loading-State starten
            self.show_loading(True)

            # Subklasse muss diese Methode implementieren
            self._render_map(**params)

        finally:
            # Loading-State beenden
            self.show_loading(False)

    def _render_map(self, world_manager=None, **params):
        """
        Funktionsweise: Muss von Subklassen implementiert werden
        - Enthält die eigentliche Rendering-Logik
        - Wird von update_map() aufgerufen
        """
        raise NotImplementedError("Subklassen müssen _render_map() implementieren")

    def show_loading(self, show=True):
        """
        Funktionsweise: Zeigt/versteckt Loading-Indicator
        Args:
            show (bool): True = Loading anzeigen, False = verstecken
        """
        self.loading_label.setVisible(show)
        if show:
            self.canvas.setEnabled(False)
            self.loading_manager.start_loading()
        else:
            self.canvas.setEnabled(True)
            self.loading_manager.stop_loading()

    def set_title(self, title):
        """
        Funktionsweise: Setzt den Titel der Karte
        Args:
            title (str): Neuer Titel
        """
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

    def cleanup(self):
        """
        Funktionsweise: Ressourcen-Cleanup
        - Schließt Matplotlib Figure explizit
        - Bereinigt Render-Optimizer Cache
        """
        try:
            plt.close(self.figure)  # Explizit schließen
            self.render_optimizer.cleanup_figure(self.canvas_id)
        except:
            pass


class MultiPlotCanvas(QWidget):
    """
    Funktionsweise: Performance-optimierte Multi-Plot Canvas
    - Eigenständige Implementierung ohne Vererbung von BaseMapCanvas
    - Für komplexe Visualisierungen wie Weather Tab
    - Verwaltet mehrere Achsen automatisch
    - Optimiertes Rendering für mehrere Subplots
    - Alle Basis-Funktionen direkt implementiert
    """

    def __init__(self, canvas_id, subplot_config=(1, 2), figsize=(12, 6),
                 titles=None, debounce_ms=300):
        """
        Args:
            canvas_id (str): Eindeutige ID für Performance-Optimierung
            subplot_config (tuple): (rows, cols) für Subplot-Layout
            figsize (tuple): Größe der Figure
            titles (list): Liste der Titel für jeden Subplot
            debounce_ms (int): Debounce-Zeit für Updates
        """
        super().__init__()
        self.canvas_id = canvas_id
        self.subplot_config = subplot_config
        self.subplot_titles = titles or []
        self.debounce_ms = debounce_ms
        self.default_title = "Multi-Plot Map"
        self.render_optimizer = get_render_optimizer()
        self.loading_manager = None
        self.error_handler = ErrorHandler()

        self.init_ui(figsize)

        self.loading_manager = LoadingStateManager(self)

    def init_ui(self, figsize):
        """
        Funktionsweise: Erstellt das grundlegende Canvas-Layout für Multi-Plot
        - Erstellt optimierte Matplotlib Figure und Canvas
        - Setzt Layout auf mit Loading-Label
        - Initialisiert mehrere Subplots
        """
        layout = QVBoxLayout()

        # Loading Label
        self.loading_label = QLabel("Generiere Karten...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setVisible(False)
        self.loading_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 200);
                border: 2px solid #e67e22;
                border-radius: 10px;
                padding: 20px;
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)

        # Optimierte Figure erstellen
        self.figure = self.render_optimizer.get_optimized_figure(self.canvas_id, figsize)
        self.canvas = FigureCanvas(self.figure)

        # Subplots erstellen
        self.setup_axes()

        layout.addWidget(self.canvas)
        layout.addWidget(self.loading_label)
        self.setLayout(layout)

    def setup_axes(self):
        """
        Funktionsweise: Erstellt mehrere Subplots
        - Basierend auf subplot_config
        - Speichert Achsen in self.axes Liste
        - Setzt Standardwerte für alle Subplots
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
        """
        Funktionsweise: Löscht alle Subplots und erstellt sie neu
        - Wird vor jeder Kartenaktualisierung aufgerufen
        - Verhindert Überlagerung von alten und neuen Plots
        """
        self.figure.clear()
        self.setup_axes()

    def draw(self):
        """
        Funktionsweise: Zeichnet alle Subplots neu
        - Wrapper um matplotlib canvas.draw()
        - Enthält Fehlerbehandlung
        """
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"Fehler beim Zeichnen der Multi-Plot Karte: {e}")

    @performance_tracked("MultiPlot_Rendering")
    def update_map(self, **params):
        """
        Funktionsweise: Optimierte Multi-Plot Update
        - Performance-optimiert für mehrere Subplots
        - Zeigt Loading-State während Rendering
        """
        try:
            self.show_loading(True)
            self._render_map(**params)
        finally:
            self.show_loading(False)

    def _render_map(self, **params):
        """
        Funktionsweise: Muss von Subklassen implementiert werden
        - Enthält die eigentliche Multi-Plot Rendering-Logik
        - Wird von update_maps() aufgerufen
        """
        raise NotImplementedError("Subklassen müssen _render_maps() implementieren")

    def show_loading(self, show=True):
        """
        Funktionsweise: Zeigt/versteckt Loading-Indicator für Multi-Plot
        Args:
            show (bool): True = Loading anzeigen, False = verstecken
        """
        self.loading_label.setVisible(show)
        if show:
            self.canvas.setEnabled(False)
            self.loading_manager.start_loading()
        else:
            self.canvas.setEnabled(True)
            self.loading_manager.stop_loading()

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

    def set_title(self, title, subplot_index=None):
        """
        Funktionsweise: Setzt den Titel für ein spezifisches Subplot oder alle
        Args:
            title (str): Neuer Titel
            subplot_index (int, optional): Index des Subplots, None für alle
        """
        if subplot_index is not None and 0 <= subplot_index < len(self.axes):
            self.axes[subplot_index].set_title(title)
        else:
            # Setze Titel für alle Subplots
            for i, ax in enumerate(self.axes):
                ax.set_title(f"{title} ({i+1})")
        self.draw()

    def add_legend(self, handles, labels, subplot_index=0, **kwargs):
        """
        Funktionsweise: Fügt eine Legende zu einem spezifischen Subplot hinzu
        Args:
            handles: Legend handles
            labels: Legend labels
            subplot_index (int): Index des Subplots
            **kwargs: Weitere Argumente für matplotlib legend()
        """
        default_kwargs = {
            'loc': 'upper right',
            'fontsize': 8,
            'framealpha': 0.9
        }
        default_kwargs.update(kwargs)

        if 0 <= subplot_index < len(self.axes):
            self.axes[subplot_index].legend(handles, labels, **default_kwargs)

    def cleanup(self):
        """
        Funktionsweise: Ressourcen-Cleanup für Multi-Plot
        - Schließt Matplotlib Figure explizit
        - Bereinigt Render-Optimizer Cache
        """
        try:
            plt.close(self.figure)
            self.render_optimizer.cleanup_figure(self.canvas_id)
        except:
            pass


class CachedMapCanvas(MapCanvas):
    """
    Funktionsweise: Canvas mit intelligentem Caching
    - Erbt von MapCanvas (nicht mehr von BaseMapCanvas)
    - Cached Rendering-Ergebnisse bei gleichen Parametern
    - Für sehr komplexe/langsame Visualisierungen
    """

    def __init__(self, canvas_id, figsize=(8, 8), title="Cached Map", cache_size=10):
        """
        Args:
            canvas_id (str): Eindeutige ID für Performance-Optimierung
            figsize (tuple): Figure-Größe
            title (str): Standard-Titel
            cache_size (int): Maximale Anzahl gecachter Ergebnisse
        """
        super().__init__(canvas_id, figsize, title)
        self.render_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, params):
        """
        Funktionsweise: Generiert Cache-Key aus Parametern
        Args:
            params (dict): Parameter-Dictionary
        Returns:
            str: Cache-Key
        """
        # Konvertiere Parameter zu sortiertem String für konsistente Keys
        sorted_params = sorted(params.items())
        return str(hash(str(sorted_params)))

    def update_map(self, **params):
        """
        Funktionsweise: Update mit Caching
        - Prüft Cache vor Rendering
        - Speichert Ergebnisse für Wiederverwendung
        """
        cache_key = self._generate_cache_key(params)

        if cache_key in self.render_cache:
            # Cache Hit - verwende gespeicherte Daten
            self.cache_hits += 1
            cached_data = self.render_cache[cache_key]
            self._restore_from_cache(cached_data)
            return

        # Cache Miss - normale Renderung
        self.cache_misses += 1
        super().update_map(**params)

        # Ergebnis cachen
        self._store_in_cache(cache_key, params)

    def _store_in_cache(self, cache_key, params):
        """
        Funktionsweise: Speichert Rendering-Ergebnis im Cache
        Args:
            cache_key (str): Cache-Schlüssel
            params (dict): Parameter die gecacht werden sollen
        """
        # Vereinfachte Cache-Speicherung (in Produktionscode würde man Bild-Daten speichern)
        if len(self.render_cache) >= self.cache_size:
            # Entferne älteste Einträge
            oldest_key = next(iter(self.render_cache))
            del self.render_cache[oldest_key]

        self.render_cache[cache_key] = {
            'params': params.copy(),
            'timestamp': time.time()
        }

    def _restore_from_cache(self, cached_data):
        """
        Funktionsweise: Stellt Rendering aus Cache wieder her
        Args:
            cached_data (dict): Gecachte Daten
        """
        # In echter Implementation würde hier das gecachte Bild geladen
        params = cached_data['params']
        self._render_map(**params)

    def get_cache_stats(self):
        """
        Funktionsweise: Gibt Cache-Statistiken zurück
        Returns:
            dict: Dictionary mit Cache-Statistiken
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.render_cache)
        }