import numpy as np
from scipy.ndimage import zoom
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QLabel
from PyQt5.QtCore import pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.figure import Figure
from gui.config.gui_default import CanvasSettings, ColorSchemes

# Farbschema fuer PlotPhysicsSystem-Kerne/-Nodes (siehe
# [[project-settlement-plot-physics-rebuild]]) - geteilt zwischen der Live-
# Konvergenz-Vorschau (draw_plot_physics_snapshot) und dem finalen
# "eingefrorenen" Ergebnis (overlay_plot_boundaries), damit beide Ansichten
# optisch konsistent bleiben.
PLOT_CORE_COLOR_BY_TYPE = {
    "standard_plot_node": "#3498db", "wilderness_core": "#2ecc71", "city_core": "#e74c3c",
}
PLOT_NODE_COLOR_BY_TYPE = {
    "standard_plot_node": "#bdc3c7", "wilderness_node": "#27ae60",
    "map_border_node": "#7f8c8d", "city_border_node": "#c0392b",
}

def _validate_input_data(data):
    """
    Funktionsweise: Überprüft ob die eingehenden Daten für die Darstellung geeignet sind
    Aufgabe: Validierung von numpy-Arrays auf Typ, Dimensionen und numerische Werte
    Parameter: data - Zu prüfende Daten
    Rückgabe: bool - True wenn Daten valid sind, False sonst
    """
    if not isinstance(data, np.ndarray):
        return False

    # Echte 2D-Daten (Skalarfelder), RGB/RGBA-Bilddaten (z.B. Geology
    # rock_map, (H,W,3) Gesteinsanteile) ODER (H,W,2) Vektorfelder (wind_map -
    # eigener Pfeil-Renderer _render_wind_map()) sind gültig für die
    # Darstellung. Andere Kanalzahlen (z.B. slopemap) sind NICHT direkt
    # darstellbar - der Aufrufer muss sie vorher auf ein darstellbares Format
    # reduzieren (siehe TerrainTab update_display_mode()).
    if data.ndim == 3 and data.shape[2] not in (2, 3, 4):
        return False
    elif data.ndim not in (2, 3):
        return False

    if data.shape[0] < 10 or data.shape[1] < 10:
        return False

    if not np.issubdtype(data.dtype, np.number):
        return False

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False

    return True

def _get_layer_range(layer_key):
    """
    Funktionsweise: Liefert (colormap, vmin, vmax, scale) für einen Layer aus
    CanvasSettings.CANVAS_2D["layer_ranges"], oder (None, None, None, "linear")
    wenn kein Eintrag existiert (Aufrufer fällt dann auf Auto-Skalierung
    zurück). Aufgabe: Zentraler Lookup, von allen _render_*-Methoden genutzt,
    damit Farbskalen zwischen Frames (z.B. der saisonalen Monats-Animation im
    Weather-Tab) stabil bleiben statt pro Frame neu zu skalieren. scale ist
    "linear" (Default, bestehende 3-Tupel-Einträge) oder "log" (z.B.
    erosion_map/sedimentation_map - stark rechtsschiefe Werteverteilung,
    lineare Skala ließ den typischen Wertebereich kaum unterscheidbar
    erscheinen) - siehe _render_generic_map()/map_display_3d.py's
    _colorize_layer() für die tatsächliche Normalisierung.
    """
    entry = CanvasSettings.CANVAS_2D.get("layer_ranges", {}).get(layer_key)
    if entry is None:
        return None, None, None, "linear"
    if len(entry) == 3:
        return entry[0], entry[1], entry[2], "linear"
    return entry

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


def rasterize_plot_boundaries_rgba(plot_nodes, plot_edges, plot_cores, wilderness_polygons,
                                    map_size, resolution=512):
    """
    Funktionsweise: Rendert dieselbe Plot-Geometrie wie MapDisplay2D.
    overlay_plot_boundaries() (Kantennetz/Straßen-Tiers/Wildnisgrenzen/Kerne/
    Nodes), aber headless auf eine feste (resolution, resolution, 4)-RGBA-
    Textur statt in ein Qt-Canvas - der transparente Hintergrund (alpha=0)
    laesst das Terrain ueberall durchscheinen, wo nichts gezeichnet wurde.
    Aufgabe: Gemeinsame Rasterisierungs-Basis fuer den 3D-"Skin"-Textur-
    Upload (siehe map_display_3d.py's _render_settlement_plot_skin(),
    [[project-settlement-plot-physics-rebuild]] Teil 4) - "die 2D-Darstellung
    als Skin auf das Terrain legen" (Nutzer-Vorgabe) statt eigener 3D-
    Wireframe-/Marker-Geometrie.
    Parameter: plot_nodes/plot_edges/plot_cores/wilderness_polygons wie
    overlay_plot_boundaries(), map_size (int) - Kartengroesse in Pixeln
    (Koordinatensystem der node_location-Werte), resolution (int) -
    Ziel-Texturaufloesung.
    Return: (resolution, resolution, 4) uint8 RGBA-Array, Zeile 0 = y=0
    (row-index==y-Konvention wie heightmap/civ_map - siehe Docstring-
    Kommentar an der Aufrufstelle in map_display_3d.py).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not plot_nodes:
        return np.zeros((resolution, resolution, 4), dtype=np.uint8)

    dpi = 100
    fig = Figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    node_by_id = {n.node_id: n for n in plot_nodes}
    grey_segments, path_segments, road_segments = [], [], []
    for edge in (plot_edges or {}).values():
        a = node_by_id.get(edge.node_a)
        b = node_by_id.get(edge.node_b)
        if a is None or b is None:
            continue
        seg = (a.node_location, b.node_location)
        if edge.classification == "road":
            road_segments.append(seg)
        elif edge.classification == "path":
            path_segments.append(seg)
        else:
            grey_segments.append(seg)

    for segments, color, width, alpha in (
        (grey_segments, 'dimgray', 0.7, 0.6),
        (path_segments, '#f1c40f', 1.4, 0.9),
        (road_segments, '#e67e22', 3.0, 0.95),
    ):
        if segments:
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width, alpha=alpha, zorder=3))

    outline_segments = []
    for poly_coords in (wilderness_polygons or []):
        coords = np.asarray(poly_coords, dtype=float)
        if len(coords) < 2:
            continue
        outline_segments.extend((tuple(coords[i]), tuple(coords[i + 1])) for i in range(len(coords) - 1))
    if outline_segments:
        ax.add_collection(LineCollection(outline_segments, colors='magenta', linewidths=1.8, alpha=0.85, zorder=3))

    xs = [n.node_location[0] for n in plot_nodes]
    ys = [n.node_location[1] for n in plot_nodes]
    colors = [PLOT_NODE_COLOR_BY_TYPE.get(n.node_type, "#bdc3c7") for n in plot_nodes]
    ax.scatter(xs, ys, c=colors, marker='.', s=8, alpha=0.7, zorder=4)

    for node_type, color in PLOT_CORE_COLOR_BY_TYPE.items():
        cxs = [c.node_location[0] for c in (plot_cores or []) if c.node_type == node_type]
        cys = [c.node_location[1] for c in (plot_cores or []) if c.node_type == node_type]
        if cxs:
            ax.scatter(cxs, cys, c=color, marker='o', s=45, edgecolors='white', linewidths=0.6, zorder=5)

    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    # buffer_rgba() liefert Zeile 0 = oberer Bildrand (screen-space) = y=map_size;
    # geflippt, damit Zeile 0 = y=0 gilt (row-index==y, wie heightmap/civ_map).
    return np.flipud(buffer)

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
        self._contour_reference_heightmap = None  # für Contour-Overlay auf Nicht-Heightmap-Layern
        self._water_biomes_reference = None  # für See/Fluss-Farbunterscheidung in _render_water_map
        self.shadow_overlay_enabled = False
        self.shadow_angle_index = None
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

        # Contour Lines wird jetzt über die globale Shell-Checkbox (Spalte 2)
        # gesteuert (siehe set_contour_overlay()) - intern bleibt das Checkbox
        # verdrahtet, ist aber ausgeblendet, um Doppel-Bedienelemente zu vermeiden.
        self.contour_checkbox = QCheckBox("Contour Lines")
        self.contour_checkbox.setChecked(self.contour_lines_enabled)
        self.contour_checkbox.toggled.connect(self._toggle_contour_lines)
        self.contour_checkbox.setVisible(False)
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

        # Standard-Colormap für Heightmaps: plt.cm.terrain, aber ohne den
        # Blau-Anteil (0-25% der Original-Colormap, sieht wie Wasser aus - bei
        # elevation_vmax=4000 lag das Blau bei 0-1000m, Grün setzte erst ab 1000m
        # ein). Nimmt nur den Grün-bis-Weiß-Teil (ab 25%) und streckt ihn auf den
        # vollen 0-1 Bereich, damit Grün jetzt die niedrigste Farbe ist (0m) und
        # Weiß weiterhin die höchste (elevation_vmax). Die Farbskala am Rand
        # (colorbar in _render_heightmap) nutzt automatisch dieselbe Colormap.
        self.heightmap_cmap = ListedColormap(plt.cm.terrain(np.linspace(0.25, 1.0, 256)))

        # Feste Biome-Colormap aus ColorSchemes.BIOME_COLOR_TABLE (dieselbe Quelle
        # wie BiomeLegendDialog) - vorher plt.cm.Set3, ein generischer 12-Farben-
        # Colormap, der sich bei jedem Rendern automatisch auf den in der jeweiligen
        # Karte vorkommenden Wertebereich neu normalisierte. Dadurch stimmten Karte
        # und Legende nie zuverlässig überein, und ab Index 12 wiederholten sich
        # Farben. Jetzt: Index -> Farbe ist fix (26 Einträge, vmin/vmax beim
        # imshow() entsprechend fest gesetzt, siehe _render_biome_map).
        self.biome_cmap = ListedColormap([hex_color for _, hex_color in ColorSchemes.BIOME_COLOR_TABLE])

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
        elif layer_type == "slopemap":
            self._render_slopemap(data)
        elif layer_type == "rock_map":
            self._render_rock_map(data)
        elif layer_type == "biome_map":
            self._render_biome_map(data)
        elif layer_type == "water_map":
            self._render_water_map(data)
        elif layer_type == "temp_map":
            self._render_temperature_map(data)
        elif layer_type == "precip_map":
            self._render_precipitation_map(data)
        elif layer_type == "wind_map":
            self._render_wind_map(data)
        else:
            self._render_generic_map(data, layer_type)

        # Contour-Lines-Overlay auf Nicht-Heightmap-Layern (Heightmap zeichnet
        # ihre eigenen Konturen bereits in _render_heightmap direkt aus den
        # Layer-Daten selbst - dieselbe Quelle, kein zweites Overlay nötig).
        if (layer_type != "heightmap" and self.contour_lines_enabled
                and self._contour_reference_heightmap is not None):
            self._draw_contour_lines(self._contour_reference_heightmap)

        self._apply_styling()
        self.canvas.draw()

    def set_contour_reference_heightmap(self, heightmap):
        """
        Öffentlicher Hook: hinterlegt die Heightmap, die als Referenz für das
        Contour-Overlay dient, wenn gerade ein anderer Layer angezeigt wird
        (z.B. Water > Flowmap). Wird von BaseMapTab._push_data_to_current_display()
        bei jedem Display-Update mitgeschickt. Löst kein eigenes Redraw aus -
        die Layer-Daten kommen im selben Zug über update_display().
        """
        self._contour_reference_heightmap = heightmap

    def set_water_biomes_reference(self, water_biomes_map):
        """
        Öffentlicher Hook (analog zu set_contour_reference_heightmap): hinterlegt
        water_biomes_map (0=kein Wasser, 1-3=Creek/River/Grand River, 4=Lake), damit
        _render_water_map See-Pixel farblich von Fluss-Pixeln absetzen kann. Wird von
        WaterTab.update_display_mode() vor jedem "water_map"-Display-Update mitgeschickt
        - kein eigenes Redraw, die Layer-Daten kommen im selben Zug über update_display().
        """
        self._water_biomes_reference = water_biomes_map

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
        im = self.ax.imshow(
            heightmap, cmap=self.heightmap_cmap, origin='lower', interpolation='bilinear',
            vmin=CanvasSettings.CANVAS_2D["elevation_vmin"],
            vmax=CanvasSettings.CANVAS_2D["elevation_vmax"]
        )

        if self.contour_lines_enabled:
            self._draw_contour_lines(heightmap)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Elevation (m)')

    def _draw_contour_lines(self, heightmap):
        """
        Funktionsweise: Zeichnet Höhenlinien über das aktuell aktive Axes
        Aufgabe: Gemeinsamer Contour-Rendering-Schritt für Heightmap-Layer
        (direkt aus den eigenen Layer-Daten) und alle anderen Layer (aus der
        per set_contour_reference_heightmap() hinterlegten Referenz-Heightmap)
        Parameter: heightmap (numpy.ndarray) - Höhendaten für die Konturlinien

        Verschiedene Layer/Generatoren liegen nicht zwangsläufig in derselben
        Auflösung vor wie die aktuell angezeigte Karte (z.B. Biome-Supersampling
        2x, Geology-Zwischenschritte auf Bruchteilen der Kartengröße). ax.contour()
        zeichnet ohne explizite Achsen einfach in Pixel-Index-Koordinaten der
        heightmap - bei einer kleineren Referenz-Heightmap als der angezeigte
        Layer erscheinen die Konturen dadurch nur in einem Teilbereich der Karte
        (z.B. exakt einem Viertel bei halber Auflösung je Achse). Deshalb hier
        auf die tatsächliche Auflösung des angezeigten Layers hochskalieren.
        """
        target_shape = self.current_data.shape[:2] if self.current_data is not None else heightmap.shape
        if heightmap.shape[:2] != target_shape:
            zoom_factors = (target_shape[0] / heightmap.shape[0], target_shape[1] / heightmap.shape[1])
            heightmap = zoom(heightmap, zoom_factors, order=1)

        contour_levels = _calculate_contour_levels(heightmap)
        contours = self.ax.contour(heightmap, levels=contour_levels,
                                   colors=CanvasSettings.CANVAS_2D["contour_colors"],
                                   linewidths=0.5, alpha=0.7)
        self.ax.clabel(contours, inline=True, fontsize=8)

    def _render_slopemap(self, slope_degrees):
        """
        Funktionsweise: Rendert Steigungs-Magnitude (bereits in Grad, 2D) mit
        Steepness-Colormap
        Aufgabe: Spezialisierte Darstellung für Slope-Daten. Erwartet ein
        bereits auf 2D reduziertes Grad-Array (siehe TerrainTab.update_display_mode,
        das (H,W,2) dx/dy-Gradienten vorher zu einer Magnitude in Grad umrechnet -
        MapDisplay2D kann nur echte 2D-Bilder zeichnen).
        Parameter: slope_degrees (numpy.ndarray) - Steigung in Grad zum Rendern
        """
        _, vmin, vmax, _ = _get_layer_range("slopemap")
        im = self.ax.imshow(slope_degrees, cmap=plt.cm.YlOrRd, origin='lower', interpolation='bilinear',
                             vmin=vmin, vmax=vmax)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Slope (°)')

    def _render_rock_map(self, rock_map):
        """
        Funktionsweise: Rendert Gesteinstyp-Anteile als RGB-Bild
        Aufgabe: rock_map ist (H,W,3) - je Kanal der Anteil eines Gesteinstyps
        (z.B. igneous/sedimentary/metamorphic). imshow zeigt (H,W,3)-Arrays
        direkt als Echtfarbbild, dafür OHNE Colorbar (keine skalare Werte-Achse
        bei RGB-Composite-Daten). rock_map kommt vom Generator als uint8 mit
        R+G+B=255 (Mass-Conservation, siehe core/geology_generator.py) - vorher
        wurde direkt auf [0,1] geclippt, wodurch praktisch jeder Kanalwert (>1)
        auf 1.0 kappte und die Karte fast überall reinweiß erschien.
        Parameter: rock_map (numpy.ndarray) - (H,W,3) Gesteinsanteile, uint8 [0,255]
        """
        display_data = np.clip(rock_map.astype(np.float32) / 255.0, 0.0, 1.0)
        self.ax.imshow(display_data, origin='lower', interpolation='nearest')

    def _render_biome_map(self, biome_map):
        """
        Funktionsweise: Rendert Biome-Map mit kategorialer Colormap
        Aufgabe: Spezialisierte Darstellung für Biome-Klassifikation
        Parameter: biome_map (numpy.ndarray) - Biome-Daten zum Rendern
        """
        n_categories = len(ColorSchemes.BIOME_COLOR_TABLE)
        im = self.ax.imshow(biome_map, cmap=self.biome_cmap, origin='lower', interpolation='nearest',
                             vmin=0, vmax=n_categories - 1)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Biome Type')

        # Biome-Namen als Colorbar-Labels, nur für tatsächlich vorkommende Indizes.
        # biome_map kann Base-Biome-Indizes (0-14) UND Super-Biome-Indizes (15-25,
        # SuperBiomeOverrideSystem.super_biome_offset in core/biome_generator.py)
        # gleichzeitig enthalten (z.B. super_biome_mask, biome_map_super) - die
        # Namensliste muss deshalb beide Bereiche abdecken.
        unique_values = np.unique(biome_map)
        valid_values = [val for val in unique_values if 0 <= int(val) < n_categories]
        if valid_values:
            self.current_colorbar.set_ticks(valid_values)
            self.current_colorbar.set_ticklabels(
                [ColorSchemes.BIOME_COLOR_TABLE[int(val)][0] for val in valid_values])

    def _render_water_map(self, water_map):
        """
        Funktionsweise: Rendert Water-Map mit Blau-Farbschema
        Aufgabe: Spezialisierte Darstellung für Wasser-Daten
        Parameter: water_map (numpy.ndarray) - Wasser-Daten zum Rendern

        See-Pixel (water_biomes_map-Klasse 4, siehe core/water_generator.py
        _classify_water_bodies) werden - falls set_water_biomes_reference() vorher
        aufgerufen wurde - zusätzlich in einer klar abgesetzten Farbe overlayed, damit
        Seen auf den ersten Blick von Flüssen unterscheidbar sind (beide teilten sich
        vorher exakt dieselbe Blues-Tiefenskala, ein flacher See sah dadurch identisch
        zu einem schmalen Bach aus). water_biomes_reference bleibt optional/None-
        toleriert (Rückwärtskompatibilität, falls kein Aufrufer es setzt).
        """
        _, vmin, vmax, _ = _get_layer_range("water_map")
        im = self.ax.imshow(water_map, cmap=plt.cm.Blues, origin='lower', interpolation='bilinear',
                             vmin=vmin, vmax=vmax)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Water Depth (m)')

        water_biomes_map = self._water_biomes_reference
        if water_biomes_map is not None and water_biomes_map.shape[:2] == water_map.shape[:2]:
            lake_mask = np.ma.masked_where(water_biomes_map != 4, np.ones_like(water_map))
            self.ax.imshow(lake_mask, cmap=ListedColormap(['#0b3d91']), origin='lower',
                            interpolation='nearest', vmin=0, vmax=1, alpha=0.85)

    def _render_temperature_map(self, temp_map):
        """
        Funktionsweise: Rendert Temperatur-Map mit Rot-Blau Colormap
        Aufgabe: Spezialisierte Darstellung für Temperatur-Daten
        Parameter: temp_map (numpy.ndarray) - Temperatur-Daten zum Rendern
        """
        _, vmin, vmax, _ = _get_layer_range("temp_map")
        im = self.ax.imshow(temp_map, cmap=plt.cm.RdBu_r, origin='lower', interpolation='bilinear',
                             vmin=vmin, vmax=vmax)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Temperature (°C)')

    def _render_precipitation_map(self, precip_map):
        """
        Funktionsweise: Rendert Niederschlags-Map mit Grün-Farbschema
        Aufgabe: Spezialisierte Darstellung für Niederschlags-Daten
        Parameter: precip_map (numpy.ndarray) - Niederschlags-Daten zum Rendern
        """
        _, vmin, vmax, _ = _get_layer_range("precip_map")
        im = self.ax.imshow(precip_map, cmap=plt.cm.Greens, origin='lower', interpolation='bilinear',
                             vmin=vmin, vmax=vmax)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Precipitation (mm)')

    def _render_wind_map(self, wind_map):
        """
        Funktionsweise: Windstärke als Heatmap-Hintergrund + Stromlinien
        (matplotlib streamplot, folgt dem tatsächlichen Strömungsverlauf und
        macht Wirbel/Verwirbelungen deutlich sichtbar - reine Pfeile an
        diskreten Punkten zeigen das nicht) + zusätzliche Richtungs-Pfeile
        (matplotlib quiver) auf einem gegenüber vorher verdichteten,
        weiterhin map_size-unabhängigen Raster (32 -> 48 Punkte je Achse).
        Aufgabe: Ersetzt die frühere reine Windstärke-Heatmap (wind_map kam
        vorher bereits als Magnitude reduziert an) - zeigt jetzt zusätzlich
        die tatsächliche Windrichtung. Pfeillänge UND -farbe skalieren mit
        der lokalen Windstärke.
        Parameter: wind_map (numpy.ndarray) - (H,W,2) u/v-Windkomponenten in m/s
        """
        height, width = wind_map.shape[0], wind_map.shape[1]
        magnitude = np.sqrt(wind_map[:, :, 0] ** 2 + wind_map[:, :, 1] ** 2)

        _, vmin, vmax, _ = _get_layer_range("wind_map")
        im = self.ax.imshow(magnitude, cmap=plt.cm.Blues, origin='lower',
                             interpolation='bilinear', alpha=0.5, vmin=vmin, vmax=vmax)
        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Wind Speed (m/s)')

        # Stromlinien auf ein moderates, von map_size entkoppeltes Raster
        # (max. 256) resampled - streamplot integriert Pfadlinien statt nur
        # Einzelpunkte zu samplen, das skaliert bei sehr großen Karten
        # (1024x1024) sonst spürbar schlechter als quiver. streamplot
        # verlangt exakt äquidistante Koordinaten-Arrays - deshalb per
        # bilinearer zoom()-Interpolation auf ein reguläres Zielraster
        # resampled statt (wie beim quiver) einzelne Pixel per gerundetem
        # Integer-Index herauszugreifen (das würde bei windschiefer
        # Rundung ungleiche Abstände erzeugen -> matplotlib-Fehler).
        stream_grid = min(256, height, width)
        if stream_grid >= 4:
            zoom_factors = (stream_grid / height, stream_grid / width)
            su = zoom(wind_map[:, :, 0], zoom_factors, order=1)
            sv = zoom(wind_map[:, :, 1], zoom_factors, order=1)
            smag = zoom(magnitude, zoom_factors, order=1)
            sx = np.linspace(0, width - 1, stream_grid)
            sy = np.linspace(0, height - 1, stream_grid)
            self.ax.streamplot(sx, sy, su, sv,
                                color=smag, cmap=plt.cm.plasma, density=1.3,
                                linewidth=0.8, arrowsize=0.9)

        grid = 48
        y_idx = np.linspace(0, height - 1, min(grid, height)).astype(int)
        x_idx = np.linspace(0, width - 1, min(grid, width)).astype(int)
        yy, xx = np.meshgrid(y_idx, x_idx, indexing='ij')
        u = wind_map[yy, xx, 0]
        v = wind_map[yy, xx, 1]
        mag_sampled = magnitude[yy, xx]

        self.ax.quiver(xx, yy, u, v, mag_sampled, cmap=plt.cm.plasma,
                        angles='xy', scale_units='xy', width=0.0022, alpha=0.85)

    def _render_generic_map(self, data, layer_type=None):
        """
        Funktionsweise: Rendert unbekannte Datentypen mit Standard-Colormap
        Aufgabe: Fallback-Rendering für beliebige numerische Daten - nutzt
        eine feste Farbskala aus CanvasSettings.CANVAS_2D["layer_ranges"],
        falls layer_type dort einen Eintrag hat (z.B. humid_map,
        hardness_map, erosion_map, sedimentation_map, flow_map,
        soil_moist_map), sonst Auto-Skalierung wie bisher.
        Parameter: data (numpy.ndarray) - Beliebige Daten zum Rendern,
        layer_type (str) - Layer-Key für den Farbskalen-Lookup
        """
        cmap_name, vmin, vmax, scale = _get_layer_range(layer_type) if layer_type else (None, None, None, "linear")
        cmap = plt.cm.get_cmap(cmap_name) if cmap_name else plt.cm.viridis

        if scale == "log" and vmin is not None and vmax is not None:
            # Stark rechtsschiefe Werteverteilung (die meisten Pixel exakt 0,
            # z.B. erosion_map/sedimentation_map) - LogNorm kann keine
            # nicht-positiven Werte verarbeiten, daher vor dem Rendern auf ein
            # kleines Epsilon (deutlich unter vmin) geklemmt, damit reine
            # Nullen sauber unter die Skala fallen (unterste Farbe) statt als
            # matplotlibs "bad"-Farbe für den Großteil der Karte zu erscheinen.
            epsilon = vmin * 0.01
            plot_data = np.maximum(data, epsilon)
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            im = self.ax.imshow(plot_data, cmap=cmap, origin='lower', interpolation='bilinear', norm=norm)
        else:
            im = self.ax.imshow(data, cmap=cmap, origin='lower', interpolation='bilinear',
                                 vmin=vmin, vmax=vmax)

        self.current_colorbar = self.figure.colorbar(im, ax=self.ax)
        self.current_colorbar.set_label('Value')

    def overlay_settlements(self, settlement_list, landmark_list=None, roadsite_list=None):
        """
        Funktionsweise: Zeichnet Settlement-/Landmark-/Roadsite-Positionen als
        Marker über das aktuell angezeigte Bild
        Aufgabe: Overlay für BiomeTab/SettlementTab - erwartet Objekte mit
        .x/.y Attributen (core.settlement_generator.Location) oder (x,y)-Tupel
        Parameter: settlement_list, landmark_list, roadsite_list - Listen von Locations
        """
        if self.current_data is None:
            return

        def _coords(items):
            xs, ys = [], []
            for item in items or []:
                x = getattr(item, 'x', None)
                y = getattr(item, 'y', None)
                if x is None and isinstance(item, (tuple, list)) and len(item) >= 2:
                    x, y = item[0], item[1]
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            return xs, ys

        settle_x, settle_y = _coords(settlement_list)
        if settle_x:
            self.ax.scatter(settle_x, settle_y, c='red', marker='o', s=40,
                             edgecolors='white', linewidths=0.5, label='Settlements', zorder=5)

        landmark_x, landmark_y = _coords(landmark_list)
        if landmark_x:
            self.ax.scatter(landmark_x, landmark_y, c='gold', marker='^', s=30,
                             edgecolors='black', linewidths=0.5, label='Landmarks', zorder=5)

        roadsite_x, roadsite_y = _coords(roadsite_list)
        if roadsite_x:
            self.ax.scatter(roadsite_x, roadsite_y, c='saddlebrown', marker='s', s=15,
                             edgecolors='black', linewidths=0.3, label='Roadsites', zorder=4)

        if settle_x or landmark_x or roadsite_x:
            self.ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.canvas.draw()

    def overlay_roads(self, roads, color='darkorange', linewidth=1.2, alpha=0.85, zorder=4):
        """
        Funktionsweise: Zeichnet Road-Pfade als Linien über das aktuell
        angezeigte Bild
        Aufgabe: Overlay für SettlementTab "Road Network" - mehrfach mit
        unterschiedlicher color aufrufbar, um Hauptstraßen/Landmark-
        Anbindungen/Außenverbindungen optisch zu unterscheiden (siehe
        SettlementTab.update_settlement_display())
        Parameter: roads (List[List[Tuple]]) - Liste von Pfaden, je Pfad eine
        Liste von (x,y[,...])-Punkten
        """
        if self.current_data is None or not roads:
            return

        for path in roads:
            if not path or len(path) < 2:
                continue
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            self.ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)

        self.canvas.draw()

    def overlay_city_boundary_contour(self, city_mask, color='gold', linewidth=2.2):
        """
        Funktionsweise: Zeichnet die Stadtgrenze (siehe CityBoundaryAnalyzer in
        core/settlement_generator.py) als geschlossene Kontur statt als
        gefuellte Flaeche
        Aufgabe: Overlay fuer SettlementTab "City Boundary" - bewusst deutlich
        und "stadtmauerartig" (Nutzer-Vorgabe), kombinierbar mit jedem anderen
        Basis-Layer statt eines eigenen exklusiven Anzeigemodus
        Parameter: city_mask (numpy.ndarray[int]) - Settlement-ID pro Pixel,
        -1 = ausserhalb jeder Stadt (siehe city_mask-Konvention im Core)
        """
        if self.current_data is None or city_mask is None:
            return

        inside = (city_mask >= 0).astype(np.float32)
        if not np.any(inside):
            return

        self.ax.contour(inside, levels=[0.5], colors=[color], linewidths=linewidth, zorder=6)
        self.canvas.draw()

    def draw_plot_physics_snapshot(self, snapshot: dict):
        """
        Funktionsweise: Zeichnet einen Zwischenzustand der noch nicht
        konvergierten Plot-Physik (siehe core.settlement_generator.
        PlotPhysicsSystem._report_live_state(), [[project-settlement-plot-physics-rebuild]]
        Teil F) - Plotkerne (standard_plot_node/wilderness_core/city_core)
        als farbige Punkte, PlotNode-Kreuzungen als kleine graue Punkte.
        Entfernt vorherige Snapshot-Marker, bevor neu gezeichnet wird, damit
        sich nicht Dutzende Layer aus früheren Iterationen überlagern.
        Aufgabe: Live-Fortschrittsanzeige während der Physik-Konvergenz,
        analog zu draw.py im ursprünglichen tools/biome_lab/ Physics Lab.
        Parameter: snapshot (dict) - "core_positions"/"plot_node_positions":
        je eine Liste von (x, y, node_type)-Tupeln.
        """
        if self.current_data is None:
            return

        for artist in getattr(self, '_plot_physics_scatter_artists', []):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._plot_physics_scatter_artists = []

        plot_node_positions = snapshot.get("plot_node_positions") or []
        if plot_node_positions:
            xs = [p[0] for p in plot_node_positions]
            ys = [p[1] for p in plot_node_positions]
            colors = [PLOT_NODE_COLOR_BY_TYPE.get(p[2], "#bdc3c7") for p in plot_node_positions]
            artist = self.ax.scatter(xs, ys, c=colors, marker='.', s=6, alpha=0.7, zorder=4)
            self._plot_physics_scatter_artists.append(artist)

        core_positions = snapshot.get("core_positions") or []
        for node_type, color in PLOT_CORE_COLOR_BY_TYPE.items():
            xs = [p[0] for p in core_positions if p[2] == node_type]
            ys = [p[1] for p in core_positions if p[2] == node_type]
            if xs:
                artist = self.ax.scatter(xs, ys, c=color, marker='o', s=18,
                                          edgecolors='white', linewidths=0.3, zorder=5)
                self._plot_physics_scatter_artists.append(artist)

        self.canvas.draw()

    def overlay_plot_boundaries(self, plot_nodes, plot_edges=None, plot_cores=None, wilderness_polygons=None):
        """
        Funktionsweise: Zeichnet das konvergierte/eingefrorene Endergebnis von
        PlotPhysicsSystem (siehe [[project-settlement-plot-physics-rebuild]]
        Teil 3) - graues Voronoi-Kantennetz, Straßen/Wege nach
        PlotEdge.classification eingefärbt, Wildnisgrenz-Polygone, sowie
        Plotkerne/-Nodes farbig nach node_type. Ported aus tools/biome_lab/
        draw.py's _rebuild_static_layer()/_update_dynamic_layer(), reduziert
        auf einen einzigen statischen Zeichen-Durchlauf (kein Tick-Redraw
        nötig, das Ergebnis liegt bereits konvergiert vor).
        Aufgabe: Ersetzt die vorherige generische Nearest-Core-ID-Rasterdarstellung
        von "plot_map" als Basis-Layer für den "Plot Boundaries"-Anzeigemodus.
        Parameter: plot_nodes (List[PlotNode]) - Voronoi-Kreuzungen/Randnodes,
        plot_edges (Dict[int, PlotEdge]) - Kanten mit .node_a/.node_b/.classification,
        plot_cores (List[PlotNode]) - Plotkerne (node_type standard_plot_node/
        wilderness_core/city_core), wilderness_polygons (List[(N,2) array]) -
        Aussenkontur-Punkte je Wildnisgebiet.
        """
        if self.current_data is None or not plot_nodes:
            return

        for artist in getattr(self, '_plot_boundary_artists', []):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._plot_boundary_artists = []

        node_by_id = {n.node_id: n for n in plot_nodes}

        grey_segments, path_segments, road_segments = [], [], []
        for edge in (plot_edges or {}).values():
            a = node_by_id.get(edge.node_a)
            b = node_by_id.get(edge.node_b)
            if a is None or b is None:
                continue
            seg = (a.node_location, b.node_location)
            if edge.classification == "road":
                road_segments.append(seg)
            elif edge.classification == "path":
                path_segments.append(seg)
            else:
                grey_segments.append(seg)

        for segments, color, width, alpha in (
            (grey_segments, 'dimgray', 0.5, 0.5),
            (path_segments, '#f1c40f', 1.0, 0.85),
            (road_segments, '#e67e22', 2.2, 0.9),
        ):
            if segments:
                lc = LineCollection(segments, colors=color, linewidths=width, alpha=alpha, zorder=3)
                self.ax.add_collection(lc)
                self._plot_boundary_artists.append(lc)

        outline_segments = []
        for poly_coords in (wilderness_polygons or []):
            coords = np.asarray(poly_coords, dtype=float)
            if len(coords) < 2:
                continue
            outline_segments.extend((tuple(coords[i]), tuple(coords[i + 1])) for i in range(len(coords) - 1))
        if outline_segments:
            lc = LineCollection(outline_segments, colors='magenta', linewidths=1.3, alpha=0.85, zorder=3)
            self.ax.add_collection(lc)
            self._plot_boundary_artists.append(lc)

        xs = [n.node_location[0] for n in plot_nodes]
        ys = [n.node_location[1] for n in plot_nodes]
        colors = [PLOT_NODE_COLOR_BY_TYPE.get(n.node_type, "#bdc3c7") for n in plot_nodes]
        artist = self.ax.scatter(xs, ys, c=colors, marker='.', s=5, alpha=0.6, zorder=4)
        self._plot_boundary_artists.append(artist)

        for node_type, color in PLOT_CORE_COLOR_BY_TYPE.items():
            cxs = [c.node_location[0] for c in (plot_cores or []) if c.node_type == node_type]
            cys = [c.node_location[1] for c in (plot_cores or []) if c.node_type == node_type]
            if cxs:
                artist = self.ax.scatter(cxs, cys, c=color, marker='o', s=20,
                                          edgecolors='white', linewidths=0.4, zorder=5)
                self._plot_boundary_artists.append(artist)

        self.canvas.draw()

    def overlay_river_network(self, flow_map):
        """
        Funktionsweise: Überlagert Fluss-Netzwerk basierend auf Flow-Magnitude
        Aufgabe: Zellen mit hohem Wasserabfluss (>90. Perzentil) als Flüsse
        einfärben, Rest transparent lassen
        Parameter: flow_map (numpy.ndarray) - Wasserabfluss-Werte
        """
        if self.current_data is None or not isinstance(flow_map, np.ndarray) or flow_map.ndim != 2:
            return

        threshold = np.percentile(flow_map, 90) if np.any(flow_map > 0) else np.inf
        river_mask = np.ma.masked_less_equal(flow_map, threshold)

        self.ax.imshow(river_mask, cmap=plt.cm.Blues, origin='lower',
                        interpolation='bilinear', alpha=0.7, vmin=threshold)
        self.canvas.draw()

    def overlay_elevation_contours(self, heightmap):
        """
        Funktionsweise: Zeichnet Höhenlinien über das aktuell angezeigte Bild
        Aufgabe: Overlay für Nicht-Height-Layer (z.B. Biome-Map), damit
        Gelände-Referenz sichtbar bleibt
        Parameter: heightmap (numpy.ndarray) - Höhendaten für die Konturlinien
        """
        if self.current_data is None or not isinstance(heightmap, np.ndarray) or heightmap.ndim != 2:
            return

        contour_levels = _calculate_contour_levels(heightmap)
        contours = self.ax.contour(heightmap, levels=contour_levels, colors='white',
                                    linewidths=0.5, alpha=0.6)
        self.ax.clabel(contours, inline=True, fontsize=7)
        self.canvas.draw()

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
            "slopemap": "Terrain Slope",
            "rock_map": "Rock Type Distribution",
            "biome_map": "Biome Distribution",
            "water_map": "Water Bodies",
            "temperature_map": "Temperature Field",
            "precipitation_map": "Precipitation Field",
            "temp_map": "Temperature Field",
            "precip_map": "Precipitation Field",
            "humid_map": "Humidity Field",
            "wind_map": "Wind Speed",
            "suitability_map": "Settlement Suitability",
            "civ_map": "Civilization Influence",
            "plot_map": "Plot Boundaries"
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

    def set_contour_overlay(self, enabled: bool):
        """
        Öffentlicher Hook für das globale Shell-Checkbox "Contour Lines"
        (siehe MapEditorWindow-Spalte 2). Ersetzt das interne Checkbox als
        Bedienelement - dieses bleibt intern verdrahtet, wird aber ausgeblendet.
        """
        if self.contour_lines_enabled == enabled:
            return
        self.contour_lines_enabled = enabled
        self.contour_checkbox.blockSignals(True)
        self.contour_checkbox.setChecked(enabled)
        self.contour_checkbox.blockSignals(False)
        if self.current_data is not None:
            self.update_display(self.current_data, self.current_layer)

    def set_shadow_overlay(self, enabled: bool, angle_index: int = None):
        """
        Platzhalter-Hook für das globale Shell-Checkbox "Shadows". Hillshade-
        Rendering ist noch nicht implementiert - speichert nur den Zustand,
        damit eine spätere Shading-Implementierung hier andocken kann.
        """
        self.shadow_overlay_enabled = enabled
        self.shadow_angle_index = angle_index

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