"""
Path: gui/widgets/map_display_2d.py

Funktionsweise: 2D Map-Rendering mit Matplotlib Integration
- Heightmap-Visualization mit Contour-Lines
- Multi-Layer Rendering (Terrain + Overlays)
- Interactive Tools (Zoom, Pan, Measure)
- Export-Quality Rendering für High-Resolution Output

Kommunikationskanäle:
- Input: numpy arrays von data_manager
- Config: gui_default.CanvasSettings für Render-Parameter
- Output: Interactive 2D-Display mit Tool-Integration
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel
from PyQt5.QtCore import pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from gui.config.gui_default import CanvasSettings, ColorSchemes


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

        self.current_data = data
        self.current_layer = layer_type

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

    def _render_heightmap(self, heightmap):
        """
        Funktionsweise: Rendert Heightmap mit Terrain-Colormap und optionalen Contour-Lines
        Aufgabe: Spezialisierte Darstellung für Terrain-Daten
        Parameter: heightmap (numpy.ndarray) - Höhendaten zum Rendern
        """
        im = self.ax.imshow(heightmap, cmap=self.heightmap_cmap, origin='lower', interpolation='bilinear')

        if self.contour_lines_enabled:
            contour_levels = np.linspace(heightmap.min(), heightmap.max(), 10)
            contours = self.ax.contour(heightmap, levels=contour_levels,
                                       colors=CanvasSettings.CANVAS_2D["contour_colors"],
                                       linewidths=0.5, alpha=0.7)
            self.ax.clabel(contours, inline=True, fontsize=8)

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Elevation (m)')

    def _render_biome_map(self, biome_map):
        """
        Funktionsweise: Rendert Biome-Map mit kategorialer Colormap
        Aufgabe: Spezialisierte Darstellung für Biome-Klassifikation
        Parameter: biome_map (numpy.ndarray) - Biome-Daten zum Rendern
        """
        im = self.ax.imshow(biome_map, cmap=self.biome_cmap, origin='lower', interpolation='nearest')

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Biome Type')

        # Biome-Namen als Colorbar-Labels
        biome_names = ['Ocean', 'Tundra', 'Taiga', 'Desert', 'Temperate', 'Tropical']
        unique_values = np.unique(biome_map)
        if len(unique_values) <= len(biome_names):
            cbar.set_ticks(unique_values)
            cbar.set_ticklabels([biome_names[int(val)] for val in unique_values])

    def _render_water_map(self, water_map):
        """
        Funktionsweise: Rendert Water-Map mit Blau-Farbschema
        Aufgabe: Spezialisierte Darstellung für Wasser-Daten
        Parameter: water_map (numpy.ndarray) - Wasser-Daten zum Rendern
        """
        im = self.ax.imshow(water_map, cmap=plt.cm.Blues, origin='lower', interpolation='bilinear')

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Water Depth (m)')

    def _render_temperature_map(self, temp_map):
        """
        Funktionsweise: Rendert Temperatur-Map mit Rot-Blau Colormap
        Aufgabe: Spezialisierte Darstellung für Temperatur-Daten
        Parameter: temp_map (numpy.ndarray) - Temperatur-Daten zum Rendern
        """
        im = self.ax.imshow(temp_map, cmap=plt.cm.RdBu_r, origin='lower', interpolation='bilinear')

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Temperature (°C)')

    def _render_precipitation_map(self, precip_map):
        """
        Funktionsweise: Rendert Niederschlags-Map mit Grün-Farbschema
        Aufgabe: Spezialisierte Darstellung für Niederschlags-Daten
        Parameter: precip_map (numpy.ndarray) - Niederschlags-Daten zum Rendern
        """
        im = self.ax.imshow(precip_map, cmap=plt.cm.Greens, origin='lower', interpolation='bilinear')

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Precipitation (mm)')

    def _render_generic_map(self, data):
        """
        Funktionsweise: Rendert unbekannte Datentypen mit Standard-Colormap
        Aufgabe: Fallback-Rendering für beliebige numerische Daten
        Parameter: data (numpy.ndarray) - Beliebige Daten zum Rendern
        """
        im = self.ax.imshow(data, cmap=plt.cm.viridis, origin='lower', interpolation='bilinear')

        cbar = self.figure.colorbar(im, ax=self.ax)
        cbar.set_label('Value')

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
        if not enabled and self.measure_line:
            self.measure_line.remove()
            self.measure_line = None
            self.canvas.draw()

    def _on_mouse_press(self, event):
        """
        Funktionsweise: Handler für Mouse-Press Events
        Aufgabe: Startet Measure-Operation oder andere Interaktionen
        Parameter: event - Matplotlib MouseEvent
        """
        if event.inaxes != self.ax:
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
            self.ax.text((start_x + x) / 2, (start_y + y) / 2, f'{distance:.1f}',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            self.canvas.draw()

    def _on_mouse_release(self, event):
        """
        Funktionsweise: Handler für Mouse-Release Events
        Aufgabe: Finalisiert Measure-Operation
        Parameter: event - Matplotlib MouseEvent
        """
        if self.measure_mode and self.measure_start and event.inaxes == self.ax:
            start_x, start_y = self.measure_start
            end_x, end_y = event.xdata, event.ydata

            distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
            self.measurement_completed.emit(distance)

            self.measure_start = None

    def _on_mouse_scroll(self, event):
        """
        Funktionsweise: Handler für Mouse-Scroll Events für Zoom-Funktionalität
        Aufgabe: Implementiert Zoom-in/Zoom-out mit Mausrad
        Parameter: event - Matplotlib ScrollEvent
        """
        if event.inaxes != self.ax:
            return

        # Zoom-Faktor
        zoom_factor = 1.1 if event.step > 0 else 1 / 1.1

        # Aktuelle Limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Zoom-Zentrum (Maus-Position)
        x_center = event.xdata
        y_center = event.ydata

        # Neue Limits berechnen
        x_range = (xlim[1] - xlim[0]) / zoom_factor
        y_range = (ylim[1] - ylim[0]) / zoom_factor

        new_xlim = [x_center - x_range / 2, x_center + x_range / 2]
        new_ylim = [y_center - y_range / 2, y_center + y_range / 2]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def _export_png(self):
        """
        Funktionsweise: Exportiert aktuelle Darstellung als hochauflösende PNG
        Aufgabe: Export-Quality Rendering für High-Resolution Output
        """
        if self.current_data is None:
            return

        # Temporäre Figure für High-Resolution Export
        export_fig = Figure(figsize=(16, 12), dpi=300)
        export_ax = export_fig.add_subplot(111)

        # Aktuellen Content auf Export-Axes rendern
        if self.current_layer == "heightmap":
            export_ax.imshow(self.current_data, cmap=self.heightmap_cmap, origin='lower')
        elif self.current_layer == "biome_map":
            export_ax.imshow(self.current_data, cmap=self.biome_cmap, origin='lower')
        else:
            export_ax.imshow(self.current_data, cmap=plt.cm.viridis, origin='lower')

        export_ax.set_title(f"Exported {self.current_layer}", fontsize=16)
        export_fig.tight_layout()

        # Export-Signal mit temporärer Figure
        self.export_requested.emit("PNG")

    def reset_view(self):
        """
        Funktionsweise: Setzt Zoom und Pan auf Standard-Ansicht zurück
        Aufgabe: Reset zu vollständiger Map-Ansicht
        """
        if self.current_data is not None:
            self.ax.set_xlim(0, self.current_data.shape[1])
            self.ax.set_ylim(0, self.current_data.shape[0])
            self.canvas.draw()