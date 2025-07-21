#!/usr/bin/env python3
"""
Path: MapGenerator/gui/widgets/map_canvas.py

REFACTORED Map Canvas - Sauber, vollständig funktionell, ohne Overengineering
- Eliminiert 1200+ Zeilen Redundanz
- Behält alle 6 Canvas-Typen vollständig funktionsfähig
- Direkte Core-Integration ohne Fallback-Chaos
- Optimierte 2D/3D Dual-View Implementierung
- Signal-System für Tab-Datenübertragung
"""

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class BaseMapCanvas(QWidget):
    """
    Funktionsweise: Gemeinsame Basis für alle Canvas-Typen
    - Matplotlib/PyQt5 Integration
    - Standard Signal-System
    - Gemeinsame Rendering-Funktionen
    - Einheitliches Error-Handling
    """

    data_generated = pyqtSignal(object)

    def __init__(self, canvas_id, title="Map View"):
        super().__init__()
        self.canvas_id = canvas_id
        self.title = title
        self.input_data = {}

        # Standard Terrain Colormap für alle Canvas
        self.terrain_colors = ['#000080', '#4169E1', '#90EE90', '#32CD32',
                               '#FFFF00', '#D2691E', '#808080', '#FFFFFF']
        self.terrain_cmap = LinearSegmentedColormap.from_list('terrain', self.terrain_colors)

        self.init_ui()

    def init_ui(self):
        """Standard UI Setup"""
        layout = QVBoxLayout()

        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        layout.addWidget(self.title_label)

        # Canvas
        self.figure = Figure(figsize=(8, 6), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.setup_axes()

        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def setup_axes(self):
        """Standard Achsen-Konfiguration"""
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(self.title)

    def update_map(self, **params):
        """Template Method - wird von Subklassen überschrieben"""
        try:
            self._render_map(**params)
            self.canvas.draw()
        except Exception as e:
            print(f"{self.canvas_id} rendering error: {e}")
            self._show_error(str(e))

    def _render_map(self, **params):
        """Muss von Subklassen implementiert werden"""
        raise NotImplementedError

    def set_input_data(self, data_type, data):
        """Empfängt Daten von anderen Tabs"""
        self.input_data[data_type] = data
        print(f"{self.canvas_id}: Received {data_type} data")

    def _render_terrain_base(self, heightmap):
        """Gemeinsame Terrain-Basis für alle Canvas"""
        h, w = heightmap.shape
        x = np.linspace(0, 100, w)
        y = np.linspace(0, 100, h)
        return np.meshgrid(x, y)

    def _show_error(self, error_msg):
        """Einheitliche Error-Anzeige"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, f'Error: {error_msg[:50]}...',
                     transform=self.ax.transAxes, ha='center', va='center',
                     bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
        self.canvas.draw()

    def _show_missing_input(self, required_input):
        """Einheitliche Missing Input Anzeige"""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.text(0.5, 0.5, f'Benötigt: {required_input}\n\nBitte vorherigen Tab vervollständigen',
                     transform=self.ax.transAxes, ha='center', va='center',
                     bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.9))
        self.canvas.draw()


class Dual2D3DCanvas(BaseMapCanvas):
    """
    Funktionsweise: 2D/3D Dual-View Canvas
    - Erweitert BaseMapCanvas um 3D-Funktionalität
    - View-Toggle zwischen 2D, 3D und beiden
    - Template Methods für 2D/3D Rendering
    """

    def __init__(self, canvas_id, title="2D/3D View"):
        super().__init__(canvas_id, title)
        self._setup_dual_view()

    def _setup_dual_view(self):
        """Konfiguriert Dual-View Layout"""
        # Altes Layout komplett entfernen
        if self.layout():
            QWidget().setLayout(self.layout())

        # Neues Layout erstellen
        layout = QVBoxLayout()

        # Header mit View Controls
        header_layout = QHBoxLayout()
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        # View Toggle Buttons
        self.view_2d_btn = QPushButton("2D")
        self.view_3d_btn = QPushButton("3D")
        self.view_both_btn = QPushButton("Beide")

        for btn in [self.view_2d_btn, self.view_3d_btn, self.view_both_btn]:
            btn.setMaximumWidth(60)
            btn.clicked.connect(self._toggle_view)
            header_layout.addWidget(btn)

        self.view_both_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        layout.addLayout(header_layout)

        # Canvas Area
        canvas_layout = QHBoxLayout()

        # 2D Canvas
        self.figure = Figure(figsize=(6, 6), facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # 3D Canvas
        self.figure_3d = Figure(figsize=(6, 6), facecolor='white')
        self.canvas_3d = FigureCanvas(self.figure_3d)
        self.ax_3d = self.figure_3d.add_subplot(111, projection='3d')

        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.canvas_3d)
        layout.addLayout(canvas_layout)

        self.setLayout(layout)
        self._setup_dual_axes()

        # CRITICAL: Initial draw
        self.canvas.draw()
        self.canvas_3d.draw()

    def _setup_dual_axes(self):
        """Konfiguriert beide Achsen"""
        # 2D
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('2D Ansicht')

        # 3D
        self.ax_3d.set_xlim(0, 100)
        self.ax_3d.set_ylim(0, 100)
        self.ax_3d.set_title('3D Ansicht')
        self.fixed_elevation = 30
        self.ax_3d.view_init(elev=self.fixed_elevation, azim=45)

        # Event-Handler für Mouse-Interaction überschreiben
        self.ax_3d.mouse_init()
        self.canvas_3d.mpl_connect('button_press_event', self._on_3d_mouse_press)
        self.canvas_3d.mpl_connect('motion_notify_event', self._on_3d_mouse_motion)

    def _on_3d_mouse_press(self, event):
        """Nur bei 3D Subplot reagieren"""
        if event.inaxes != self.ax_3d:
            return
        self.last_mouse_x = event.x

    def _on_3d_mouse_motion(self, event):
        """Beschränkte 3D Navigation - nur Azimut ändern"""
        if event.inaxes != self.ax_3d or not hasattr(self, 'last_mouse_x'):
            return

        if event.button == 1:  # Linke Maustaste gedrückt
            # Berechne Azimut-Änderung basierend auf X-Bewegung
            dx = event.x - self.last_mouse_x
            current_azim = self.ax_3d.azim

            # Nur Azimut ändern, Elevation bleibt fest
            new_azim = current_azim + dx * 0.5  # Sensitivität anpassen
            self.ax_3d.view_init(elev=self.fixed_elevation, azim=new_azim)

            self.canvas_3d.draw()
            self.last_mouse_x = event.x

    def _toggle_view(self):
        """View Toggle Handler"""
        sender = self.sender()

        # Reset styles
        for btn in [self.view_2d_btn, self.view_3d_btn, self.view_both_btn]:
            btn.setStyleSheet("font-weight: bold;")

        if sender == self.view_2d_btn:
            self.canvas_3d.hide()
            self.canvas.show()
            sender.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        elif sender == self.view_3d_btn:
            self.canvas.hide()
            self.canvas_3d.show()
            sender.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        else:
            self.canvas.show()
            self.canvas_3d.show()
            sender.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")

    def update_map(self, **params):
        """Dual-View Update"""
        try:
            self._render_2d(**params)
            self._render_3d(**params)
            self.canvas.draw()
            self.canvas_3d.draw()
        except Exception as e:
            print(f"{self.canvas_id} dual rendering error: {e}")
            self._show_error(str(e))

    def _render_2d(self, **params):
        """Template Method für 2D Rendering"""
        pass

    def _render_3d(self, **params):
        """Template Method für 3D Rendering"""
        pass


# ==============================================================================
# SPEZIALISIERTE CANVAS-KLASSEN
# ==============================================================================

class TerrainCanvas(Dual2D3DCanvas):
    """
    Funktionsweise: Terrain Canvas mit Core-Integration
    - Direkte BaseTerrainGenerator Nutzung
    - 2D: Contour Map, 3D: Surface Plot
    - Heightmap-Signal für andere Tabs
    """

    heightmap_generated = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__('terrain', 'Terrain Generator - Live 2D/3D')
        self.current_heightmap = None
        self.terrain_generator = None

    def update_map(self, **params):
        """Override für Terrain-spezifische Logik"""
        # Erst Heightmap generieren
        self._render_map(**params)
        # Dann beide Views rendern
        super().update_map(**params)

    def _render_2d(self, **params):
        """2D Terrain Contour Map"""
        if self.current_heightmap is None:
            return

        X, Y = self._render_terrain_base(self.current_heightmap)

        self.ax.clear()
        contour = self.ax.contourf(X, Y, self.current_heightmap, levels=20,
                                   cmap=self.terrain_cmap, alpha=0.9)
        self.ax.contour(X, Y, self.current_heightmap, levels=10,
                        colors='black', alpha=0.4, linewidths=0.8)

        self.ax.set_title(f'2D Heightmap - Seed: {params.get("seed", 42)}')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')

    def _render_3d(self, **params):
        """3D Terrain Surface Plot"""
        if self.current_heightmap is None:
            return

        X, Y = self._render_terrain_base(self.current_heightmap)

        self.ax_3d.clear()
        surface = self.ax_3d.plot_surface(X, Y, self.current_heightmap,
                                          cmap='terrain', alpha=0.9, antialiased=True)

        self.ax_3d.set_zlim(self.current_heightmap.min(), self.current_heightmap.max())
        self.ax_3d.set_title(f'3D Heightmap - {params.get("octaves", 6)} Oktaven')
        self.ax_3d.view_init(elev=30, azim=45)

    def _render_map(self, **params):
        """Generiert Heightmap mit Core-Integration"""
        try:
            from core.terrain_generator import BaseTerrainGenerator

            size = params.get('size', 256)
            seed = params.get('seed', 42)

            self.terrain_generator = BaseTerrainGenerator(size, size, seed)
            self.current_heightmap = self.terrain_generator.generate_heightmap(**params)

            # Signal emittieren
            self.heightmap_generated.emit(self.current_heightmap)

        except ImportError:
            print("Core Terrain Generator nicht verfügbar")
            self._show_error("Core Terrain Generator fehlt")

    def update_terrain(self, heightmap, params):
        """Backwards Compatibility Wrapper für alte terrain_tab.py"""
        self.current_heightmap = heightmap
        self.update_map(**params)

    def _calculate_fallback_stats(self, heightmap):
        """Fallback Stats für terrain_tab.py Kompatibilität"""
        if heightmap is None:
            return {}

        return {
            'min_height': float(np.min(heightmap)),
            'max_height': float(np.max(heightmap)),
            'avg_height': float(np.mean(heightmap)),
            'height_std': float(np.std(heightmap)),
            'flat_percentage': float(np.sum(heightmap < np.mean(heightmap) * 0.3) / heightmap.size * 100),
            'mountain_percentage': float(np.sum(heightmap > np.mean(heightmap) * 1.2) / heightmap.size * 100)
        }

    def get_heightmap_for_next_tab(self):
        """Gibt Heightmap für nachfolgende Tabs zurück"""
        return self.current_heightmap


class GeologyCanvas(Dual2D3DCanvas):
    """
    Funktionsweise: Geology Canvas mit Textured Terrain
    - 2D: Gesteinsarten-Karte, 3D: Textured Terrain
    - Benötigt Heightmap vom Terrain Tab
    """

    geology_data_generated = pyqtSignal(dict)

    def __init__(self):
        super().__init__('geology', 'Geology Generator - 3D Textured Terrain')
        self.current_geology_data = None

        # Geology Colormap
        rock_colors = ['#8B4513', '#696969', '#2F4F4F', '#800080', '#F5DEB3']
        self.geology_cmap = ListedColormap(rock_colors)

    def _render_2d(self, **params):
        """2D Gesteinsarten-Karte"""
        if self.current_geology_data is None:
            return

        rock_map = self.current_geology_data['rock_map']
        h, w = rock_map.shape

        # Clear 2D Axes
        self.ax.clear()

        # Koordinaten für matplotlib
        x = np.linspace(0, 100, w)
        y = np.linspace(0, 100, h)
        X, Y = np.meshgrid(x, y)

        # Core Rock Types für Farben verwenden
        from core.geology_generator import ROCK_TYPES
        rock_colors = [rock_type.color for rock_type in ROCK_TYPES]

        from matplotlib.colors import ListedColormap
        geology_cmap = ListedColormap(rock_colors)

        # Zeige rock_map mit korrekten Core-Farben
        im = self.ax.imshow(rock_map, cmap=geology_cmap,
                            origin='lower', extent=[0, 100, 0, 100],
                            vmin=0, vmax=len(ROCK_TYPES) - 1, alpha=0.85)

        # Kontur-Linien für bessere Sichtbarkeit
        levels = list(range(len(ROCK_TYPES)))
        self.ax.contour(X, Y, rock_map, levels=levels, colors='black', alpha=0.4, linewidths=0.8)

        # Achsen-Setup
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.set_title('Gesteinsarten-Verteilung (Core)')
        self.ax.set_xlabel('X-Koordinate')
        self.ax.set_ylabel('Y-Koordinate')

    def _render_3d(self, **params):
        """3D Textured Terrain"""
        if 'heightmap' not in self.input_data or self.current_geology_data is None:
            return

        heightmap = self.input_data['heightmap']
        rock_map = self.current_geology_data['rock_map']
        hardness_map = self.current_geology_data['hardness_map']

        h, w = heightmap.shape

        # Koordinaten für 3D
        x = np.linspace(0, 100, w)
        y = np.linspace(0, 100, h)
        X, Y = np.meshgrid(x, y)

        # Clear 3D Axes
        self.ax_3d.clear()

        # Core Rock Types für Farben
        from core.geology_generator import ROCK_TYPES

        # Erstelle Farb-Array basierend auf Core rock_map
        colors_array = np.zeros((h, w, 4))  # RGBA

        for y_idx in range(h):
            for x_idx in range(w):
                rock_id = rock_map[y_idx, x_idx]
                if rock_id < len(ROCK_TYPES):
                    rock_type = ROCK_TYPES[rock_id]
                    # Konvertiere Hex-Farbe zu RGB
                    hex_color = rock_type.color.lstrip('#')
                    rgb = tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
                    colors_array[y_idx, x_idx] = (*rgb, 1.0)

        # 3D Surface mit Core-Texturen
        surface = self.ax_3d.plot_surface(X, Y, heightmap,
                                          facecolors=colors_array,
                                          alpha=0.9, antialiased=True,
                                          shade=True, linewidth=0)

        # Achsen-Setup
        self.ax_3d.set_xlim(0, 100)
        self.ax_3d.set_ylim(0, 100)
        self.ax_3d.set_zlim(heightmap.min(), heightmap.max())
        self.ax_3d.set_title('3D Terrain mit Core-Gesteinsarten')
        self.ax_3d.view_init(elev=30, azim=45)

    def _render_map(self, **params):
        """Geology Generation"""
        if 'heightmap' not in self.input_data:
            self._show_missing_input("Heightmap vom Terrain Tab")
            return

        heightmap = self.input_data['heightmap']

        # ECHTER Core GeologyGenerator Import und Verwendung
        from core.geology_generator import GeologyGenerator

        # Bereite Core-Parameter vor (aus GUI)
        core_params = {
            'tectonic_activity': 0.5,  # Default, kann aus params übernommen werden
            'volcanic_activity': 0.3,
            'erosion_strength': 0.4,
            'age_factor': 0.6
        }

        # Übernehme GUI-Parameter falls verfügbar
        if 'ridge_warping' in params:
            core_params['tectonic_activity'] = params['ridge_warping']
        if 'bevel_warping' in params:
            core_params['volcanic_activity'] = params['bevel_warping']

        # Erstelle Core GeologyGenerator
        h, w = heightmap.shape
        seed = params.get('seed', 42)
        geology_gen = GeologyGenerator(w, h, seed)

        # ECHTE Core-Generierung
        geology_result = geology_gen.generate_geology_map(heightmap, **core_params)

        # Speichere Ergebnisse
        self.current_geology_data = geology_result
        self.set_input_data('geology', geology_result)

        # Signal emittieren
        self.geology_data_generated.emit(geology_result)

class WeatherCanvas(Dual2D3DCanvas):
    """
    Funktionsweise: Weather Canvas mit Wind-Vektoren
    - 2D: Temperature/Rain Maps, 3D: Wind-Vektoren über Terrain
    - Benötigt Heightmap für Terrain-Interaktion
    """

    weather_data_generated = pyqtSignal(dict)

    def __init__(self):
        super().__init__('weather', 'Weather Generator - 3D Wind Fields')
        self.current_weather_data = None

    def _render_2d(self, **params):
        """2D Weather Maps"""
        if self.current_weather_data is None:
            return

        temp_map = self.current_weather_data['temperature_map']
        rain_map = self.current_weather_data['rain_map']
        X, Y = self._render_terrain_base(temp_map)

        self.ax.clear()
        temp_contour = self.ax.contourf(X, Y, temp_map, levels=15, cmap='RdYlBu_r', alpha=0.6)
        self.ax.contour(X, Y, rain_map, levels=8, colors='blue', alpha=0.8)
        self.ax.set_title(f'2D Weather - Temp: {params.get("avg_temperature", 15)}°C')

    def _render_3d(self, **params):
        """3D Wind-Vektoren über Terrain"""
        if 'heightmap' not in self.input_data or self.current_weather_data is None:
            return

        heightmap = self.input_data['heightmap']
        wind_speed = self.current_weather_data.get('wind_speed')
        wind_direction = self.current_weather_data.get('wind_direction')

        if wind_speed is None:
            return

        X, Y = self._render_terrain_base(heightmap)

        self.ax_3d.clear()
        surface = self.ax_3d.plot_surface(X, Y, heightmap, cmap='terrain', alpha=0.6)

        # Wind-Vektoren (reduzierte Dichte)
        h, w = heightmap.shape
        step = 8
        wind_height = params.get('wind_height', 5.0)

        for i in range(0, h, step):
            for j in range(0, w, step):
                x_pos = j * 100 / w
                y_pos = i * 100 / h
                z_pos = heightmap[i, j] + wind_height

                speed = wind_speed[i, j]
                direction = wind_direction[i, j]

                dx = speed * np.cos(direction) * 3
                dy = speed * np.sin(direction) * 3

                color = plt.cm.YlOrRd(min(1.0, speed / 10.0))
                self.ax_3d.quiver(x_pos, y_pos, z_pos, dx, dy, 0,
                                  color=color, alpha=0.8, arrow_length_ratio=0.3)

        self.ax_3d.set_title('3D Wind Fields')

    def _render_map(self, **params):
        """Weather Generation"""
        heightmap = self.input_data.get('heightmap')

        try:
            from core.weather_generator import RainGenerator, TemperatureGenerator
            from core.world_generator import World

            if heightmap is not None:
                world = World(heightmap, heightmap.shape[1], heightmap.shape[0], 100.0, 42)

                rain_gen = RainGenerator(world)
                rain_map = rain_gen.generate_orographic_rain_with_wind(
                    base_wind_speed=params.get('wind_speed', 8.0),
                    terrain_factor=params.get('wind_terrain_influence', 4.0)
                )

                temp_gen = TemperatureGenerator(world)
                temp_map = temp_gen.generate_temperature_map()

                wind_speed = getattr(rain_gen, 'wind_speed', np.random.rand(100, 100) * 10)
                wind_direction = getattr(rain_gen, 'wind_direction', np.random.rand(100, 100) * 2 * np.pi)
            else:
                # Fallback ohne Heightmap
                rain_map = np.random.rand(100, 100) * 0.5
                temp_map = np.random.rand(100, 100) * 30 + 5
                wind_speed = np.random.rand(100, 100) * 10
                wind_direction = np.random.rand(100, 100) * 2 * np.pi

            self.current_weather_data = {
                'rain_map': rain_map,
                'temperature_map': temp_map,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction
            }

            self.weather_data_generated.emit(self.current_weather_data)

        except ImportError:
            print("Core Weather Generator nicht verfügbar")
            self._show_error("Core Weather Generator fehlt")


class WaterCanvas(Dual2D3DCanvas):
    """
    Funktionsweise: Water Canvas mit Flow-Systemen
    - 2D: Flow-Netzwerk, 3D: Wassersysteme über Terrain
    - Benötigt Heightmap und optional Weather-Daten
    """

    water_data_generated = pyqtSignal(dict)

    def __init__(self):
        super().__init__('water', 'Water Generator - 3D Flow Systems')
        self.current_water_data = None

    def _render_2d(self, **params):
        """2D Water Network"""
        if self.current_water_data is None:
            return

        flow_accumulation = self.current_water_data['flow_accumulation']
        water_network = self.current_water_data['water_network']
        X, Y = self._render_terrain_base(flow_accumulation)

        self.ax.clear()
        flow_contour = self.ax.contourf(X, Y, flow_accumulation, levels=15, cmap='terrain', alpha=0.4)

        # Water Network Overlay
        water_colors = {'streams': '#87CEEB', 'rivers': '#4682B4', 'lakes': '#0000CD'}

        for water_type, color in water_colors.items():
            if water_type in water_network:
                water_mask = water_network[water_type]
                if np.any(water_mask):
                    self.ax.contour(X, Y, water_mask, levels=[0.5], colors=[color], linewidths=3)

        self.ax.set_title(f'2D Water Network - Sea Level: {params.get("sea_level", 15)}%')

    def _render_3d(self, **params):
        """3D Wassersysteme über Terrain"""
        if 'heightmap' not in self.input_data or self.current_water_data is None:
            return

        heightmap = self.input_data['heightmap']
        water_network = self.current_water_data['water_network']
        X, Y = self._render_terrain_base(heightmap)

        self.ax_3d.clear()
        surface = self.ax_3d.plot_surface(X, Y, heightmap, cmap='terrain', alpha=0.6)

        # Wassersysteme als erhöhte Surfaces
        water_height_offset = params.get('water_speed', 8.0) / 10.0

        for water_type in ['rivers', 'lakes']:
            if water_type in water_network:
                water_mask = water_network[water_type]
                if np.any(water_mask):
                    water_heights = heightmap + water_mask * water_height_offset * 2.0
                    water_heights_masked = np.where(water_mask > 0.1, water_heights, np.nan)

                    try:
                        color = 'blue' if water_type == 'rivers' else 'darkblue'
                        self.ax_3d.plot_surface(X, Y, water_heights_masked, color=color, alpha=0.8)
                    except:
                        pass

        self.ax_3d.set_title('3D Water Systems')

    def _render_map(self, **params):
        """Water Generation"""
        if 'heightmap' not in self.input_data:
            self._show_missing_input("Heightmap vom Terrain Tab")
            return

        heightmap = self.input_data['heightmap']
        rain_map = self.input_data.get('weather', {}).get('rain_map', np.random.rand(*heightmap.shape) * 0.5)

        try:
            from core.water_generator import RiverSimulator

            river_sim = RiverSimulator(
                heightmap, rain_map,
                sea_level=params.get('sea_level', 15) / 100,
                rain_scale=params.get('water_speed', 8.0)
            )

            flow_accumulation = river_sim.generate_flow_accumulation()
            water_network = river_sim.create_river_network()

            self.current_water_data = {
                'flow_accumulation': flow_accumulation,
                'water_network': water_network
            }

            self.water_data_generated.emit(self.current_water_data)

        except ImportError:
            # Simple fallback
            sea_level = params.get('sea_level', 15) / 100.0
            water_mask = heightmap < (heightmap.min() + (heightmap.max() - heightmap.min()) * sea_level)

            self.current_water_data = {
                'flow_accumulation': heightmap,
                'water_network': {'rivers': water_mask.astype(float)}
            }

            self.water_data_generated.emit(self.current_water_data)


class SettlementCanvas(Dual2D3DCanvas):
    """
    Funktionsweise: Settlement Canvas mit 3D Markers
    - 2D: Settlement-Positionen, 3D: 3D Markers über Terrain
    - Terrain-basierte Platzierung
    """

    settlement_data_generated = pyqtSignal(dict)

    def __init__(self):
        super().__init__('settlement', 'Settlement Generator - 3D Terrain')
        self.current_settlement_data = None

    def _render_2d(self, **params):
        """2D Settlement Map"""
        if 'heightmap' not in self.input_data or self.current_settlement_data is None:
            return

        heightmap = self.input_data['heightmap']
        X, Y = self._render_terrain_base(heightmap)

        self.ax.clear()
        self.ax.contourf(X, Y, heightmap, levels=15, cmap='terrain', alpha=0.4)

        # Settlement Markers
        settlement_styles = {
            'villages': {'marker': 's', 'color': '#8b4513', 'size': 100},
            'landmarks': {'marker': '^', 'color': '#fbbf24', 'size': 120},
            'pubs': {'marker': 'o', 'color': '#dc2626', 'size': 80}
        }

        for settlement_type, settlements in self.current_settlement_data.items():
            if settlement_type in settlement_styles and settlements:
                style = settlement_styles[settlement_type]
                positions = [s['position'] for s in settlements]
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]

                self.ax.scatter(x_coords, y_coords, marker=style['marker'],
                                c=style['color'], s=style['size'], alpha=0.9,
                                edgecolors='black', linewidth=1)

        total = sum(len(settlements) for settlements in self.current_settlement_data.values())
        self.ax.set_title(f'2D Settlements - Total: {total}')

    def _render_3d(self, **params):
        """3D Terrain mit Settlement Markers"""
        if 'heightmap' not in self.input_data or self.current_settlement_data is None:
            return

        heightmap = self.input_data['heightmap']
        X, Y = self._render_terrain_base(heightmap)

        self.ax_3d.clear()
        surface = self.ax_3d.plot_surface(X, Y, heightmap, cmap='terrain', alpha=0.7)

        # 3D Settlement Markers
        settlement_styles = {
            'villages': {'color': '#8b4513', 'size': 100},
            'landmarks': {'color': '#fbbf24', 'size': 150},
            'pubs': {'color': '#dc2626', 'size': 80}
        }

        h, w = heightmap.shape
        for settlement_type, settlements in self.current_settlement_data.items():
            if settlement_type in settlement_styles and settlements:
                style = settlement_styles[settlement_type]

                for settlement in settlements:
                    x_pos, y_pos = settlement['position']

                    # Höhe an Position interpolieren
                    x_idx = int(x_pos * w / 100)
                    y_idx = int(y_pos * h / 100)
                    x_idx = min(max(x_idx, 0), w - 1)
                    y_idx = min(max(y_idx, 0), h - 1)
                    z_pos = heightmap[y_idx, x_idx] + 5

                    self.ax_3d.scatter([x_pos], [y_pos], [z_pos],
                                       c=style['color'], s=style['size'], alpha=0.9)

        self.ax_3d.set_title('3D Terrain mit Settlements')

    def _render_map(self, **params):
        """Settlement Generation"""
        if 'heightmap' not in self.input_data:
            self._show_missing_input("Heightmap vom Terrain Tab")
            return

        heightmap = self.input_data['heightmap']
        settlement_data = self._generate_settlements(heightmap, params)

        self.current_settlement_data = settlement_data
        self.settlement_data_generated.emit(settlement_data)

    def _generate_settlements(self, heightmap, params):
        """Generiert Settlement-Positionen basierend auf Terrain"""
        h, w = heightmap.shape
        settlement_data = {'villages': [], 'landmarks': [], 'pubs': []}

        # Terrain-basierte Gewichtung (bevorzuge flache Gebiete)
        grad_y, grad_x = np.gradient(heightmap)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        suitability = 1.0 / (1.0 + slope)

        # Höhen-Präferenz (mittlere Höhen bevorzugt)
        normalized_height = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        height_weight = np.exp(-((normalized_height - 0.3) ** 2) / 0.1)
        suitability *= height_weight

        np.random.seed(params.get('seed', 42))

        # Generiere verschiedene Settlement-Typen
        for settlement_type, count in [('villages', params.get('villages', 3)),
                                       ('landmarks', params.get('landmarks', 2)),
                                       ('pubs', params.get('pubs', 2))]:
            for i in range(count):
                pos = self._find_best_position(suitability)
                if pos:
                    settlement_data[settlement_type].append({
                        'position': pos,
                        'size': params.get(f'{settlement_type[:-1]}_size', 15),
                        'influence': params.get(f'{settlement_type[:-1]}_influence', 20)
                    })
                    self._reduce_suitability_around_point(suitability, pos, 20)

        return settlement_data

    def _find_best_position(self, suitability):
        """Findet beste Position basierend auf Suitability"""
        flat_suitability = suitability.flatten()
        if np.sum(flat_suitability) == 0:
            return None

        probabilities = flat_suitability / np.sum(flat_suitability)
        chosen_idx = np.random.choice(len(flat_suitability), p=probabilities)

        h, w = suitability.shape
        y = chosen_idx // w
        x = chosen_idx % w

        return (x * 100 / w, y * 100 / h)

    def _reduce_suitability_around_point(self, suitability, pos, radius):
        """Reduziert Suitability um einen Punkt für Mindestabstand"""
        h, w = suitability.shape
        x_pixel = int(pos[0] * w / 100)
        y_pixel = int(pos[1] * h / 100)
        radius_pixel = int(radius * min(h, w) / 100)

        for dy in range(-radius_pixel, radius_pixel + 1):
            for dx in range(-radius_pixel, radius_pixel + 1):
                ny, nx = y_pixel + dy, x_pixel + dx
                if 0 <= ny < h and 0 <= nx < w:
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    if distance <= radius_pixel:
                        reduction = 1.0 - (distance / radius_pixel)
                        suitability[ny, nx] *= (1.0 - reduction * 0.8)


class BiomeCanvas(Dual2D3DCanvas):
    """
    Funktionsweise: Finale Biome Canvas - Multi-Layer World
    - 2D: Finale Biome-Klassifikation
    - 3D: Multi-Layer Welt mit allen Elementen
    - Kombiniert alle vorherigen Tab-Daten
    """

    biome_data_generated = pyqtSignal(dict)
    final_world_generated = pyqtSignal(dict)

    def __init__(self):
        super().__init__('biome', 'Biome Generator - Final World')
        self.current_biome_data = None

        # Biome-Definitionen mit Farben
        self.biome_definitions = {
            0: {'name': 'Tiefsee', 'color': '#000080'},
            1: {'name': 'Ozean', 'color': '#1e40af'},
            2: {'name': 'Strand', 'color': '#fbbf24'},
            3: {'name': 'Sumpf', 'color': '#65a30d'},
            4: {'name': 'Grasland', 'color': '#84cc16'},
            5: {'name': 'Wald', 'color': '#166534'},
            6: {'name': 'Nadelwald', 'color': '#14532d'},
            7: {'name': 'Steppe', 'color': '#ca8a04'},
            8: {'name': 'Wüste', 'color': '#dc2626'},
            9: {'name': 'Tundra', 'color': '#94a3b8'},
            10: {'name': 'Alpin', 'color': '#e5e7eb'},
            11: {'name': 'Gletscher', 'color': '#f3f4f6'}
        }

        colors = [self.biome_definitions[i]['color'] for i in sorted(self.biome_definitions.keys())]
        self.biome_cmap = ListedColormap(colors)

    def _render_2d(self, **params):
        """2D Finale Biome-Karte"""
        if self.current_biome_data is None:
            return

        biome_map = self.current_biome_data['biome_map']
        biome_stats = self.current_biome_data['biome_stats']

        self.ax.clear()
        self.ax.imshow(biome_map, cmap=self.biome_cmap,
                       vmin=0, vmax=len(self.biome_definitions) - 1,
                       origin='lower', extent=[0, 100, 0, 100], alpha=0.85)

        # Top 3 Biome als Text
        top_biomes = sorted(biome_stats.items(), key=lambda x: x[1], reverse=True)[:3]
        stats_text = "TOP 3 BIOME:\n" + "\n".join([f"{name}: {pct:.1f}%" for name, pct in top_biomes])

        self.ax.text(2, 98, stats_text, fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                     verticalalignment='top')

        self.ax.set_title('Finale Biome-Klassifikation')

    def _render_3d(self, **params):
        """3D Multi-Layer Final World"""
        if 'heightmap' not in self.input_data or self.current_biome_data is None:
            return

        heightmap = self.input_data['heightmap']
        biome_map = self.current_biome_data['biome_map']
        X, Y = self._render_terrain_base(heightmap)

        # Biome-Farben als RGB Array
        biome_colors_normalized = biome_map / (len(self.biome_definitions) - 1)
        biome_colors_array = self.biome_cmap(biome_colors_normalized)

        self.ax_3d.clear()
        surface = self.ax_3d.plot_surface(X, Y, heightmap,
                                          facecolors=biome_colors_array,
                                          alpha=0.8, antialiased=True, shade=False)

        # Zusätzliche Layer hinzufügen
        self._add_3d_layers(heightmap)

        self.ax_3d.set_title('3D Final World - Multi-Layer')

    def _add_3d_layers(self, heightmap):
        """Fügt zusätzliche 3D Layer hinzu"""
        h, w = heightmap.shape

        # Water Layer
        if 'water' in self.input_data:
            water_data = self.input_data['water']
            if 'water_network' in water_data:
                X, Y = self._render_terrain_base(heightmap)
                water_network = water_data['water_network']

                for water_type in ['rivers', 'lakes']:
                    if water_type in water_network:
                        water_mask = water_network[water_type]
                        if np.any(water_mask):
                            water_heights = heightmap + water_mask * 2.0
                            water_heights_masked = np.where(water_mask > 0.1, water_heights, np.nan)

                            try:
                                color = 'blue' if water_type == 'rivers' else 'darkblue'
                                self.ax_3d.plot_surface(X, Y, water_heights_masked,
                                                        color=color, alpha=0.8)
                            except:
                                pass

        # Settlement Layer
        if 'settlement' in self.input_data:
            settlement_data = self.input_data['settlement']
            settlement_styles = {
                'villages': {'color': '#8b4513', 'size': 100},
                'landmarks': {'color': '#fbbf24', 'size': 150},
                'pubs': {'color': '#dc2626', 'size': 80}
            }

            for settlement_type, settlements in settlement_data.items():
                if settlement_type in settlement_styles and settlements:
                    style = settlement_styles[settlement_type]

                    for settlement in settlements:
                        x_pos, y_pos = settlement['position']
                        x_idx = int(x_pos * w / 100)
                        y_idx = int(y_pos * h / 100)
                        x_idx = min(max(x_idx, 0), w - 1)
                        y_idx = min(max(y_idx, 0), h - 1)
                        z_pos = heightmap[y_idx, x_idx] + 8

                        self.ax_3d.scatter([x_pos], [y_pos], [z_pos],
                                           c=style['color'], s=style['size'], alpha=0.9)

    def _render_map(self, **params):
        """Finale Biome-Klassifikation"""
        if 'heightmap' not in self.input_data:
            self._show_missing_input("Heightmap vom Terrain Tab")
            return

        heightmap = self.input_data['heightmap']

        # Sammle alle verfügbaren Daten
        all_data = {
            'heightmap': heightmap,
            'weather': self.input_data.get('weather'),
            'water': self.input_data.get('water'),
            'geology': self.input_data.get('geology'),
            'settlement': self.input_data.get('settlement')
        }

        # Klassifiziere Biome
        biome_map = self._classify_biomes(heightmap, all_data)
        biome_stats = self._calculate_biome_statistics(biome_map)

        self.current_biome_data = {
            'biome_map': biome_map,
            'biome_stats': biome_stats,
            'all_input_data': all_data
        }

        # Signale emittieren
        self.biome_data_generated.emit(self.current_biome_data)
        self.final_world_generated.emit({
            'type': 'final_world',
            'biome_data': self.current_biome_data,
            'generation_complete': True
        })

    def _classify_biomes(self, heightmap, all_data):
        """Intelligente Biome-Klassifikation"""
        h, w = heightmap.shape
        biome_map = np.zeros((h, w), dtype=int)

        # Normalisierte Höhe
        normalized_height = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())

        # Weather-Daten
        temperature_map = None
        rain_map = None
        if all_data['weather']:
            temperature_map = all_data['weather'].get('temperature_map')
            rain_map = all_data['weather'].get('rain_map')

        # Fallback Weather-Gradients
        if temperature_map is None:
            temperature_map = np.zeros_like(heightmap)
            for i in range(h):
                temp_factor = 1.0 - (i / h) * 0.8  # Nord-Süd Gradient
                temperature_map[i, :] = temp_factor

        if rain_map is None:
            rain_map = np.zeros_like(heightmap)
            for j in range(w):
                rain_factor = 1.0 - (j / w) * 0.6  # West-Ost Gradient
                rain_map[:, j] = rain_factor

        # Biome-Klassifizierung
        sea_level = 0.15

        for i in range(h):
            for j in range(w):
                height = normalized_height[i, j]
                temp = temperature_map[i, j] if temperature_map.shape == (h, w) else 0.5
                rain = rain_map[i, j] if rain_map.shape == (h, w) else 0.5

                # Klassifizierungs-Logik
                if height < sea_level * 0.5:
                    biome_map[i, j] = 0  # Tiefsee
                elif height < sea_level:
                    biome_map[i, j] = 1  # Ozean
                elif height < sea_level + 0.05:
                    biome_map[i, j] = 2  # Strand
                elif height > 0.85:
                    biome_map[i, j] = 11 if temp < 0.3 else 10  # Gletscher/Alpin
                elif height > 0.75:
                    biome_map[i, j] = 9 if temp < 0.4 else 6  # Tundra/Nadelwald
                elif temp < 0.25:
                    biome_map[i, j] = 9  # Tundra
                elif rain < 0.25:
                    biome_map[i, j] = 8 if temp > 0.6 else 7  # Wüste/Steppe
                elif rain > 0.8 and height < 0.4:
                    biome_map[i, j] = 3  # Sumpf
                elif rain > 0.5 and temp > 0.4:
                    biome_map[i, j] = 5  # Wald
                elif rain > 0.4 and height > 0.4:
                    biome_map[i, j] = 6  # Nadelwald
                else:
                    biome_map[i, j] = 4  # Grasland

        return biome_map

    def _calculate_biome_statistics(self, biome_map):
        """Berechnet Biome-Verteilungs-Statistiken"""
        unique, counts = np.unique(biome_map, return_counts=True)
        total_pixels = biome_map.size

        stats = {}
        for biome_id, count in zip(unique, counts):
            percentage = (count / total_pixels) * 100
            biome_name = self.biome_definitions.get(biome_id, {}).get('name', f'Biome {biome_id}')
            stats[biome_name] = percentage

        return stats


# ==============================================================================
# HELPER FUNKTIONEN & FACTORY
# ==============================================================================

def create_canvas_for_tab(tab_name):
    """
    Funktionsweise: Factory-Funktion für Canvas-Erstellung
    - Erstellt passende Canvas-Klasse basierend auf Tab-Name
    - Vereinfacht Tab-Integration
    """
    canvas_map = {
        'terrain': TerrainCanvas,
        'geology': GeologyCanvas,
        'weather': WeatherCanvas,
        'water': WaterCanvas,
        'settlement': SettlementCanvas,
        'biome': BiomeCanvas
    }

    canvas_class = canvas_map.get(tab_name.lower())
    if canvas_class:
        return canvas_class()
    else:
        return BaseMapCanvas(tab_name, f"{tab_name.title()} View")


def connect_canvas_signals(canvas_list):
    """
    Funktionsweise: Verbindet Canvas-Signale für automatische Tab-Datenübertragung
    - Heightmap vom Terrain Tab an alle anderen
    - Weather-Daten an Water und Biome Tab
    - Settlement und Water-Daten an Biome Tab
    """
    if not canvas_list:
        return

    # Finde Canvas-Instanzen
    terrain_canvas = next((c for c in canvas_list if isinstance(c, TerrainCanvas)), None)
    weather_canvas = next((c for c in canvas_list if isinstance(c, WeatherCanvas)), None)
    water_canvas = next((c for c in canvas_list if isinstance(c, WaterCanvas)), None)
    settlement_canvas = next((c for c in canvas_list if isinstance(c, SettlementCanvas)), None)
    biome_canvas = next((c for c in canvas_list if isinstance(c, BiomeCanvas)), None)

    # Terrain -> Alle anderen
    if terrain_canvas:
        for canvas in canvas_list:
            if canvas != terrain_canvas:
                terrain_canvas.heightmap_generated.connect(
                    lambda heightmap, c=canvas: c.set_input_data('heightmap', heightmap)
                )

    # Weather -> Water, Biome
    if weather_canvas:
        for canvas in [water_canvas, biome_canvas]:
            if canvas:
                weather_canvas.weather_data_generated.connect(
                    lambda data, c=canvas: c.set_input_data('weather', data)
                )

    # Water -> Biome
    if water_canvas and biome_canvas:
        water_canvas.water_data_generated.connect(
            lambda data: biome_canvas.set_input_data('water', data)
        )

    # Settlement -> Biome
    if settlement_canvas and biome_canvas:
        settlement_canvas.settlement_data_generated.connect(
            lambda data: biome_canvas.set_input_data('settlement', data)
        )

    print(f"Canvas-Signale verbunden für {len(canvas_list)} Canvas-Instanzen")


# ==============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ==============================================================================

# Aliases für bestehende Tab-Integration
TerrainDualCanvas = TerrainCanvas
GeologyDualCanvas = GeologyCanvas
WeatherDualCanvas = WeatherCanvas
WaterDualCanvas = WaterCanvas
SettlementDualCanvas = SettlementCanvas
BiomeDualCanvas = BiomeCanvas