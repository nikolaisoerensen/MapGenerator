import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel
from PyQt5.QtCore import pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from gui.config.gui_default import CanvasSettings

def _validate_input_data(data):
    """
    Funktionsweise: Überprüft ob die eingehenden Daten für die Darstellung geeignet sind
    Aufgabe: Validierung von numpy-Arrays auf Typ, Dimensionen und numerische Werte
    Parameter: data - Zu prüfende Daten
    Rückgabe: bool - True wenn Daten valid sind, False sonst
    """
    if not isinstance(data, np.ndarray):
        return False

    if data.ndim != 2:
        return False

    if data.shape[0] < 10 or data.shape[1] < 10:
        return False

    if not np.issubdtype(data.dtype, np.number):
        return False

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False

    return True

def _calculate_contour_levels(heightmap):
    """
    Funktionsweise: Berechnet intelligente Contour-Level basierend auf maximaler Höhe
    Aufgabe: Dynamische Höhenlinien-Abstände je nach Terrain-Höhe
    Parameter: heightmap (numpy.ndarray) - Höhendaten
    Rückgabe: numpy.ndarray - Array mit Contour-Levels
    """
    max_height = heightmap.max()
    min_height = heightmap.min()

    if max_height > 1000:
        interval = 100  # Alle 100m bei hohen Bergen
    elif max_height > 500:
        interval = 50   # Alle 50m bei mittleren Höhen
    else:
        interval = 25   # Alle 25m bei niedrigen Höhen

    # Start bei nächstem Intervall über min_height
    start = np.ceil(min_height / interval) * interval
    end = np.floor(max_height / interval) * interval

    if start > end:
        # Fallback wenn Bereich zu klein
        return np.linspace(min_height, max_height, 5)

    return np.arange(start, end + interval, interval)

class MapDisplay2D(QWidget):
    """
    Funktionsweise: 2D-Visualisierung von Heightmaps und anderen Generator-Outputs mit Matplotlib
    Aufgabe: Interaktive 2D-Darstellung mit Zoom, Pan und Measure-Tools
    """

    # Signals für Tool-Interaktion
    coordinates_changed = pyqtSignal(float, float)  # (x, y)
    measurement_completed = pyqtSignal(float)  # (distance)
    export_requested = pyqtSignal(str)  # (format)

    def __init__(self, parent=None):
        """
        Funktionsweise: Initialisiert 2D Map-Display mit Matplotlib Canvas
        Aufgabe: Setup von Canvas, Tools und Event-Handlers
        """
        super().__init__(parent)

        self.current_data = None
        self.current_layer = "heightmap"
        self.contour_lines_enabled = True
        self.measure_mode = False
        self.measure_start = None
        self.measure_line = None
        self.measure_texts = []  # Liste für Measure-Text-Labels
        self.current_colorbar = None  # Referenz auf aktuelle Colorbar
        self.zoom_limits = None  # Zoom-Grenzen basierend auf Daten

        self._setup_ui()
        self._setup_matplotlib()
        self._connect_events()

    def _setup_ui(self):
        """
        Funktionsweise: Erstellt UI-Layout mit Canvas und Control-Buttons
        Aufgabe: Layout-Setup gemäß gui_default.py CanvasSettings
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Matplotlib Canvas
        self.figure = Figure(figsize=(12, 8), dpi=CanvasSettings.CANVAS_2D["dpi"])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Tool-Controls
        tool_layout = QHBoxLayout()

        self.contour_checkbox = QCheckBox("Contour Lines")
        self.contour_checkbox.setChecked(self.contour_lines_enabled)
        self.contour_checkbox.toggled.connect(self._toggle_contour_lines)
        tool_layout.addWidget(self.contour_checkbox)

        self.measure_button = QPushButton("Measure Distance")
        self.measure_button.setCheckable(True)
        self.measure_button.toggled.connect(self._toggle_measure_mode)
        tool_layout.addWidget(self.measure_button)

        self.export_button = QPushButton("Export PNG")
        self.export_button.clicked.connect(self._export_png)
        tool_layout.addWidget(self.export_button)

        # Koordinaten-Display
        self.coord_label = QLabel("Coordinates: (0, 0)")
        tool_layout.addWidget(self.coord_label)

        tool_layout.addStretch()
        layout.addLayout(tool_layout)

    def _setup_matplotlib(self):
        """
        Funktionsweise: Konfiguriert Matplotlib-Darstellung mit Standard-Settings
        Aufgabe: Axes-Setup, Colormap und Styling aus gui_default.py
        """
        self.figure.patch.set_facecolor(CanvasSettings.CANVAS_2D["background_color"])

        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(CanvasSettings.CANVAS_2D["background_color"])

        # Standard-Colormap für Heightmaps
        self.heightmap_cmap = plt.cm.terrain
        self.biome_cmap = plt.cm.Set3

        self.figure.tight_layout()

    def _connect_events(self):
        """
        Funktionsweise: Verbindet Matplotlib-Events mit Interaktions-Handlers
        Aufgabe: Setup von Mouse-Events für Zoom, Pan und Measure-Tools
        """
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)

    def update_display(self, data, layer_type="heightmap"):
        """
        Funktionsweise: Aktualisiert 2D-Display mit neuen Generator-Daten
        Aufgabe: Rendering von verschiedenen Datentypen (Heightmap, Biomes, etc.)
        Parameter: data (numpy.ndarray) - Daten zum Anzeigen, layer_type (str) - Datentyp
        """
        if data is None:
            return

        # Datenvalidierung
        if not _validate_input_data(data):
            print(f"Warnung: Ungültige Daten für {layer_type} erhalten")
            return

        self.current_data = data
        self.current_layer = layer_type

        # Zoom-Grenzen basierend auf Daten setzen
        self.zoom_limits = {
            'x_min': 0, 'x_max': data.shape[1],
            'y_min': 0, 'y_max': data.shape[0],
            'min_zoom_range': min(data.shape) * 0.05,  # Minimum 5% der kleineren Dimension
            'max_zoom_range': max(data.shape) * 1.0   # Maximum 100% der größeren Dimension
        }

        # Measure-Tools beim Layer-Wechsel zurücksetzen
        self._reset_measure_tools()

        # Alte Colorbar entfernen
        if self.current_colorbar:
            self.current_colorbar.remove()
            self.current_colorbar = None

        self.ax.clear()

        if layer_type == "heightmap":
            self._render_heightmap(data)
        elif layer_type == "biome_map":
            self._render_biome_map(data)
        elif layer_type == "water_map":
            self._render_water_map(data)
        elif layer_type == "temperature_map":
            self._render_temperature_map(data)
        elif layer_type == "precipitation_map":
            self._render_precipitation_map(data)
        else:
            self._render_generic_map(data)

        self._apply_styling()
        self.canvas.draw()

    def _reset_measure_tools(self):
        """
        Funktionsweise: Setzt alle Measure-Tools zurück
        Aufgabe: Cleanup von Measure-Line, Text-Labels und Status-Variablen
        """
        if self.measure_line:
            self.measure_line.remove()
            self.measure_line = None

        # Alle Measure-Texte entfernen
        for text in self.measure_texts:
            if text in self.ax.texts:
                text.remove()
        self.measure_texts.clear()

        self.measure_start = None

    def _render_heightmap(self, heightmap):
        """
        Funktionsweise: Rendert Heightmap mit Terrain-Colormap und optionalen Contour-Lines
        Aufgabe: Spezialisierte Darstellung für Terrain-Daten
        Parameter: heightmap (numpy.ndarray) - Höhendaten zum Rendern
        """
        im = self.ax.imshow(heightmap, cmap=self.heightmap_cmap, origin='lower', interpolation='bilinear')

        if self.contour_lines_enabled:
            contour_levels = _calculate_contour_levels(heightmap)
            contours = self.ax.contour(heightmap, levels=contour_levels,
                                       colors=CanvasSettings.CANVAS_2D["contour_colors"],
                                       linewidths=0.5, alpha=0.7)
            self.ax.clabel(contours, inline=True, fontsize=8)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Elevation (m)')

    def _render_biome_map(self, biome_map):
        """
        Funktionsweise: Rendert Biome-Map mit kategorialer Colormap
        Aufgabe: Spezialisierte Darstellung für Biome-Klassifikation
        Parameter: biome_map (numpy.ndarray) - Biome-Daten zum Rendern
        """
        im = self.ax.imshow(biome_map, cmap=self.biome_cmap, origin='lower', interpolation='nearest')

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Biome Type')

        # Biome-Namen als Colorbar-Labels
        biome_names = ['Ocean', 'Tundra', 'Taiga', 'Desert', 'Temperate', 'Tropical']
        unique_values = np.unique(biome_map)
        if len(unique_values) <= len(biome_names):
            self.current_colorbar.set_ticks(unique_values)
            self.current_colorbar.set_ticklabels([biome_names[int(val)] for val in unique_values])

    def _render_water_map(self, water_map):
        """
        Funktionsweise: Rendert Water-Map mit Blau-Farbschema
        Aufgabe: Spezialisierte Darstellung für Wasser-Daten
        Parameter: water_map (numpy.ndarray) - Wasser-Daten zum Rendern
        """
        im = self.ax.imshow(water_map, cmap=plt.cm.Blues, origin='lower', interpolation='bilinear')

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Water Depth (m)')

    def _render_temperature_map(self, temp_map):
        """
        Funktionsweise: Rendert Temperatur-Map mit Rot-Blau Colormap
        Aufgabe: Spezialisierte Darstellung für Temperatur-Daten
        Parameter: temp_map (numpy.ndarray) - Temperatur-Daten zum Rendern
        """
        im = self.ax.imshow(temp_map, cmap=plt.cm.RdBu_r, origin='lower', interpolation='bilinear')

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Temperature (°C)')

    def _render_precipitation_map(self, precip_map):
        """
        Funktionsweise: Rendert Niederschlags-Map mit Grün-Farbschema
        Aufgabe: Spezialisierte Darstellung für Niederschlags-Daten
        Parameter: precip_map (numpy.ndarray) - Niederschlags-Daten zum Rendern
        """
        im = self.ax.imshow(precip_map, cmap=plt.cm.Greens, origin='lower', interpolation='bilinear')

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Precipitation (mm)')

    def _render_generic_map(self, data):
        """
        Funktionsweise: Rendert unbekannte Datentypen mit Standard-Colormap
        Aufgabe: Fallback-Rendering für beliebige numerische Daten
        Parameter: data (numpy.ndarray) - Beliebige Daten zum Rendern
        """
        im = self.ax.imshow(data, cmap=plt.cm.viridis, origin='lower', interpolation='bilinear')

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Value')

    def _apply_styling(self):
        """
        Funktionsweise: Wendet einheitliches Styling auf Axes an
        Aufgabe: Konsistente Darstellung gemäß gui_default.py ColorSchemes
        """
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.grid(True, alpha=0.3, color=CanvasSettings.CANVAS_2D["grid_color"])

        # Title basierend auf aktuellem Layer
        layer_titles = {
            "heightmap": "Terrain Elevation",
            "biome_map": "Biome Distribution",
            "water_map": "Water Bodies",
            "temperature_map": "Temperature Field",
            "precipitation_map": "Precipitation Field"
        }

        title = layer_titles.get(self.current_layer, "Map Data")
        self.ax.set_title(title, fontsize=14, fontweight='bold')

    def _toggle_contour_lines(self, enabled):
        """
        Funktionsweise: Schaltet Contour-Lines ein/aus und aktualisiert Display
        Aufgabe: Toggle-Funktionalität für Höhenlinien-Darstellung
        Parameter: enabled (bool) - True wenn Contour-Lines angezeigt werden sollen
        """
        self.contour_lines_enabled = enabled
        if self.current_data is not None:
            self.update_display(self.current_data, self.current_layer)

    def _toggle_measure_mode(self, enabled):
        """
        Funktionsweise: Aktiviert/Deaktiviert Measure-Tool für Distanz-Messung
        Aufgabe: Toggle zwischen normalem und Measure-Modus
        Parameter: enabled (bool) - True wenn Measure-Modus aktiv sein soll
        """
        self.measure_mode = enabled
        if not enabled:
            self._reset_measure_tools()
            self.canvas.draw()

    def _on_mouse_press(self, event):
        """
        Funktionsweise: Handler für Mouse-Press Events
        Aufgabe: Startet Measure-Operation oder andere Interaktionen
        Parameter: event - Matplotlib MouseEvent
        """
        if event.inaxes != self.ax:
            return

        # Error-Handling für ungültige Koordinaten
        if event.xdata is None or event.ydata is None:
            return

        if self.measure_mode:
            self.measure_start = (event.xdata, event.ydata)
            if self.measure_line:
                self.measure_line.remove()
                self.measure_line = None

    def _on_mouse_move(self, event):
        """
        Funktionsweise: Handler für Mouse-Move Events
        Aufgabe: Aktualisiert Koordinaten-Display und Measure-Line
        Parameter: event - Matplotlib MouseEvent
        """
        if event.inaxes != self.ax:
            return

        # Error-Handling für ungültige Koordinaten
        if event.xdata is None or event.ydata is None:
            return

        # Koordinaten-Update
        x, y = event.xdata, event.ydata
        self.coord_label.setText(f"Coordinates: ({x:.1f}, {y:.1f})")
        self.coordinates_changed.emit(x, y)

        # Measure-Line Update
        if self.measure_mode and self.measure_start:
            if self.measure_line:
                self.measure_line.remove()

            start_x, start_y = self.measure_start
            self.measure_line = self.ax.plot([start_x, x], [start_y, y], 'r-', linewidth=2, alpha=0.7)[0]

            # Distanz berechnen
            distance = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)

            # Alten Distanz-Text entfernen
            for text in self.measure_texts:
                if text in self.ax.texts:
                    text.remove()
            self.measure_texts.clear()

            # Neuen Distanz-Text hinzufügen
            text_obj = self.ax.text((start_x + x) / 2, (start_y + y) / 2, f'{distance:.1f}',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            self.measure_texts.append(text_obj)

            self.canvas.draw()

    def _on_mouse_release(self, event):
        """
        Funktionsweise: Handler für Mouse-Release Events
        Aufgabe: Finalisiert Measure-Operation
        Parameter: event - Matplotlib MouseEvent
        """
        if event.inaxes != self.ax:
            return

        # Error-Handling für ungültige Koordinaten
        if event.xdata is None or event.ydata is None:
            return

        if self.measure_mode and self.measure_start:
            start_x, start_y = self.measure_start
            end_x, end_y = event.xdata, event.ydata

            distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            self.measurement_completed.emit(distance)

            self.measure_start = None

    def _on_mouse_scroll(self, event):
        """
        Funktionsweise: Handler für Mouse-Scroll Events für Zoom-Funktionalität
        Aufgabe: Implementiert Zoom-in/Zoom-out mit Mausrad und Zoom-Begrenzungen
        Parameter: event - Matplotlib ScrollEvent
        """
        if event.inaxes != self.ax:
            return

        # Error-Handling für ungültige Koordinaten
        if event.xdata is None or event.ydata is None:
            return

        if self.zoom_limits is None:
            return

        # Zoom-Faktor
        zoom_factor = 1.1 if event.step > 0 else 1 / 1.1

        # Aktuelle Limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Zoom-Zentrum (Maus-Position)
        x_center = event.xdata
        y_center = event.ydata

        # Neue Ranges berechnen
        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor

        # Zoom-Grenzen prüfen
        min_range = self.zoom_limits['min_zoom_range']
        max_range = self.zoom_limits['max_zoom_range']

        if x_range < min_range or y_range < min_range:
            return  # Zu weit hineingezoomt
        if x_range > max_range or y_range > max_range:
            return  # Zu weit herausgezoomt

        # Neue Limits berechnen
        new_xlim = [x_center - x_range / 2, x_center + x_range / 2]
        new_ylim = [y_center - y_range / 2, y_center + y_range / 2]

        # Limits innerhalb der Datengrenzen halten
        if new_xlim[0] < self.zoom_limits['x_min']:
            offset = self.zoom_limits['x_min'] - new_xlim[0]
            new_xlim[0] += offset
            new_xlim[1] += offset
        if new_xlim[1] > self.zoom_limits['x_max']:
            offset = new_xlim[1] - self.zoom_limits['x_max']
            new_xlim[0] -= offset
            new_xlim[1] -= offset

        if new_ylim[0] < self.zoom_limits['y_min']:
            offset = self.zoom_limits['y_min'] - new_ylim[0]
            new_ylim[0] += offset
            new_ylim[1] += offset
        if new_ylim[1] > self.zoom_limits['y_max']:
            offset = new_ylim[1] - self.zoom_limits['y_max']
            new_ylim[0] -= offset
            new_ylim[1] -= offset

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def _export_png(self):
        """
        Funktionsweise: Exportiert aktuelle Darstellung als hochauflösende PNG
        Aufgabe: Export-Quality Rendering für High-Resolution Output

        TODO: Implementierung in zukünftigem Stadium
        - File-Dialog für Speicherpfad
        - Tatsächliches Speichern der Export-Figure
        - Verschiedene Export-Formate (PNG, PDF, SVG)
        - Export-Einstellungen (DPI, Größe)
        """
        # Temporäre Deaktivierung - wird später implementiert
        pass

        # if self.current_data is None:
        #     return
        #
        # # Temporäre Figure für High-Resolution Export
        # export_fig = Figure(figsize=(16, 12), dpi=300)
        # export_ax = export_fig.add_subplot(111)
        #
        # # Aktuellen Content auf Export-Axes rendern
        # if self.current_layer == "heightmap":
        #     export_ax.imshow(self.current_data, cmap=self.heightmap_cmap, origin='lower')
        # elif self.current_layer == "biome_map":
        #     export_ax.imshow(self.current_data, cmap=self.biome_cmap, origin='lower')
        # else:
        #     export_ax.imshow(self.current_data, cmap=plt.cm.viridis, origin='lower')
        #
        # export_ax.set_title(f"Exported {self.current_layer}", fontsize=16)
        # export_fig.tight_layout()
        #
        # # Export-Signal mit temporärer Figure
        # self.export_requested.emit("PNG")

    def reset_view(self):
        """
        Funktionsweise: Setzt Zoom und Pan auf Standard-Ansicht zurück
        Aufgabe: Reset zu vollständiger Map-Ansicht
        """
        if self.current_data is not None:
            self.ax.set_xlim(0, self.current_data.shape[1])
            self.ax.set_ylim(0, self.current_data.shape[0])
            self.canvas.draw()