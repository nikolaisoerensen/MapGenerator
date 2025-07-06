#!/usr/bin/env python3
"""
Path: MapGenerator/gui/utils/performance_utils.py
__init__.py existiert in "utils"

Performance Utilities für World Generator
Debouncing, Throttling und Memory-Management
"""

import time
import gc
from functools import wraps
from PyQt5.QtCore import QTimer, QObject, pyqtSignal, Qt
from gui.utils.error_handler import ErrorHandler


class DebounceTimer(QObject):
    """
    Funktionsweise: Debouncing für UI-Events
    - Verhindert zu häufige Map-Updates bei Slider-Bewegungen
    - Wartet bis User mit Eingaben aufhört
    - Reduziert CPU-Last erheblich
    """

    # Signal wird nach Debounce-Zeit ausgesendet
    triggered = pyqtSignal()

    def __init__(self, delay_ms=300):
        """
        Args:
            delay_ms (int): Wartezeit in Millisekunden
        """
        super().__init__()
        self.delay_ms = delay_ms
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.triggered.emit)

    def trigger(self):
        """
        Funktionsweise: Triggert das Debouncing
        - Stoppt vorherigen Timer
        - Startet neuen Timer mit delay_ms
        - Signal wird erst nach Wartezeit ausgesendet
        """
        self.timer.stop()
        self.timer.start(self.delay_ms)

    def stop(self):
        """Stoppt das Debouncing"""
        self.timer.stop()


class ThrottleTimer(QObject):
    """
    Funktionsweise: Throttling für regelmäßige Updates
    - Begrenzt Update-Frequenz auf maximum Rate
    - Für kontinuierliche Animationen/Updates
    """

    triggered = pyqtSignal()

    def __init__(self, interval_ms=100):
        """
        Args:
            interval_ms (int): Minimum-Intervall zwischen Updates
        """
        super().__init__()
        self.interval_ms = interval_ms
        self.last_trigger = 0
        self.pending = False
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._handle_timeout)

    def trigger(self):
        """
        Funktionsweise: Triggert Throttling
        - Führt sofort aus wenn genug Zeit vergangen
        - Plant Ausführung für später wenn zu früh
        """
        current_time = time.time() * 1000  # ms
        time_since_last = current_time - self.last_trigger

        if time_since_last >= self.interval_ms:
            # Genug Zeit vergangen - sofort ausführen
            self.last_trigger = current_time
            self.triggered.emit()
        elif not self.pending:
            # Zu früh - für später planen
            self.pending = True
            remaining_time = self.interval_ms - time_since_last
            self.timer.start(int(remaining_time))

    def _handle_timeout(self):
        """Interne Timeout-Behandlung"""
        self.pending = False
        self.last_trigger = time.time() * 1000
        self.triggered.emit()


class MapRenderingOptimizer:
    """
    Funktionsweise: Optimiert Matplotlib-Rendering
    - Verhindert Memory-Leaks
    - Recycelt Figure-Objekte
    - Optimiert Update-Performance
    """

    def __init__(self):
        self.error_handler = ErrorHandler()
        self._figure_cache = {}
        self._last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 Minuten

    def get_optimized_figure(self, canvas_id, figsize=(8, 8)):
        """
        Funktionsweise: Gibt optimierte Figure zurück
        - Recycelt bestehende Figures
        - Verhindert Memory-Leaks durch Figure-Wiederverwendung

        Args:
            canvas_id (str): Eindeutige ID für Canvas
            figsize (tuple): Gewünschte Figure-Größe
        Returns:
            matplotlib.figure.Figure: Optimierte Figure
        """
        try:
            # Cleanup alte Figures wenn nötig
            self._cleanup_if_needed()

            # Prüfe Cache
            if canvas_id in self._figure_cache:
                figure = self._figure_cache[canvas_id]
                # Prüfe ob Figure noch gültig und richtige Größe hat
                if hasattr(figure, 'canvas') and figure.get_figwidth() == figsize[0]:
                    figure.clear()  # Lösche Inhalte aber behalte Figure
                    return figure

            # Erstelle neue Figure und cache sie
            import matplotlib.pyplot as plt
            figure = plt.figure(figsize=figsize, facecolor='white')
            self._figure_cache[canvas_id] = figure
            return figure

        except Exception as e:
            self.error_handler.logger.error(f"Figure-Optimierung fehlgeschlagen: {e}")
            # Fallback zu normaler Figure
            import matplotlib.pyplot as plt
            return plt.figure(figsize=figsize, facecolor='white')

    def cleanup_figure(self, canvas_id):
        """
        Funktionsweise: Räumt spezifische Figure auf
        Args:
            canvas_id (str): ID der zu löschenden Figure
        """
        if canvas_id in self._figure_cache:
            try:
                figure = self._figure_cache[canvas_id]
                figure.clear()
                import matplotlib.pyplot as plt
                plt.close(figure)
                del self._figure_cache[canvas_id]
            except Exception as e:
                self.error_handler.logger.warning(f"Figure cleanup Fehler: {e}")

    def _cleanup_if_needed(self):
        """Automatische Bereinigung alter Figures"""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._last_cleanup = current_time

            # Garbage Collection erzwingen
            gc.collect()

            self.error_handler.logger.debug(f"Figure Cache: {len(self._figure_cache)} Figures")

    def cleanup_all(self):
        """Räumt alle Figures auf"""
        import matplotlib.pyplot as plt
        for canvas_id in list(self._figure_cache.keys()):
            try:
                figure = self._figure_cache[canvas_id]
                plt.close(figure)
            except:
                pass
        self._figure_cache.clear()
        gc.collect()


class PerformanceMonitor:
    """
    Funktionsweise: Überwacht Performance-Metriken
    - Misst Rendering-Zeiten
    - Erkennt Performance-Probleme
    - Loggt Performance-Statistiken
    """

    def __init__(self):
        self.error_handler = ErrorHandler()
        self.render_times = {}
        self.performance_warnings = set()

    def start_timing(self, operation_name):
        """Startet Zeitmessung für Operation"""
        self.render_times[operation_name] = time.time()

    def end_timing(self, operation_name):
        """
        Funktionsweise: Beendet Zeitmessung und loggt Ergebnis
        Args:
            operation_name (str): Name der Operation
        Returns:
            float: Verstrichene Zeit in Sekunden
        """
        if operation_name not in self.render_times:
            return 0

        elapsed = time.time() - self.render_times[operation_name]

        # Performance-Warnung bei langsamen Operations
        if elapsed > 2.0 and operation_name not in self.performance_warnings:
            self.error_handler.logger.warning(
                f"Langsame Operation: {operation_name} dauerte {elapsed:.2f}s"
            )
            self.performance_warnings.add(operation_name)
        elif elapsed < 0.5:
            # Entferne Warnung bei verbesserter Performance
            self.performance_warnings.discard(operation_name)

        self.error_handler.logger.debug(f"{operation_name}: {elapsed:.3f}s")

        del self.render_times[operation_name]
        return elapsed


def debounced_method(delay_ms=300):
    """
    Funktionsweise: Decorator für Debouncing von Methoden
    - Verhindert zu häufige Ausführung
    - Pro Instanz ein separater DebounceTimer

    Args:
        delay_ms (int): Debounce-Zeit in Millisekunden
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Timer-Attribut pro Methode erstellen
            timer_attr = f'_debounce_{func.__name__}'

            if not hasattr(self, timer_attr):
                timer = DebounceTimer(delay_ms)
                timer.triggered.connect(lambda: func(self, *args, **kwargs))
                setattr(self, timer_attr, timer)

            # Trigger das Debouncing
            getattr(self, timer_attr).trigger()

        return wrapper

    return decorator


def throttled_method(interval_ms=100):
    """
    Funktionsweise: Decorator für Throttling von Methoden
    - Begrenzt Ausführungsfrequenz

    Args:
        interval_ms (int): Minimum-Intervall zwischen Ausführungen
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            timer_attr = f'_throttle_{func.__name__}'

            if not hasattr(self, timer_attr):
                timer = ThrottleTimer(interval_ms)
                timer.triggered.connect(lambda: func(self, *args, **kwargs))
                setattr(self, timer_attr, timer)

            getattr(self, timer_attr).trigger()

        return wrapper

    return decorator


def performance_tracked(operation_name=None):
    """
    Funktionsweise: Decorator für Performance-Tracking
    - Misst automatisch Ausführungszeit
    - Loggt Performance-Metriken

    Args:
        operation_name (str): Name für Tracking (default: function name)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Performance Monitor erstellen falls nicht vorhanden
            if not hasattr(self, '_performance_monitor'):
                self._performance_monitor = PerformanceMonitor()

            op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"

            self._performance_monitor.start_timing(op_name)
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                self._performance_monitor.end_timing(op_name)

        return wrapper

    return decorator


class LoadingStateManager:
    """
    Funktionsweise: Verwaltet Loading-States für UI
    - Zeigt Spinner/Progress während längerer Operationen
    - Deaktiviert UI-Elemente während Updates
    """

    def __init__(self, widget):
        self.widget = widget
        self.is_loading = False
        self.disabled_widgets = []

    def start_loading(self, disable_widgets=None):
        """
        Funktionsweise: Startet Loading-State
        Args:
            disable_widgets (list): Widgets die deaktiviert werden sollen
        """
        if self.is_loading:
            return

        self.is_loading = True

        # Widgets deaktivieren
        if disable_widgets:
            for widget in disable_widgets:
                if widget.isEnabled():
                    widget.setEnabled(False)
                    self.disabled_widgets.append(widget)

        # Cursor ändern
        self.widget.setCursor(Qt.WaitCursor)

    def stop_loading(self):
        """Stoppt Loading-State"""
        if not self.is_loading:
            return

        self.is_loading = False

        # Widgets wieder aktivieren
        for widget in self.disabled_widgets:
            widget.setEnabled(True)
        self.disabled_widgets.clear()

        # Cursor zurücksetzen
        self.widget.unsetCursor()


# Globaler Render-Optimizer (Singleton)
_render_optimizer = None


def get_render_optimizer():
    """
    Funktionsweise: Gibt globalen Render-Optimizer zurück
    - Singleton Pattern für app-weite Optimierung
    """
    global _render_optimizer
    if _render_optimizer is None:
        _render_optimizer = MapRenderingOptimizer()
    return _render_optimizer