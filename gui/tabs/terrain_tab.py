#!/usr/bin/env python3
"""
Path: MapGenerator/gui/tabs/terrain_py.py
__init__.py existiert in "tabs"

World Generator GUI - Terrain Tab (Vollständig Refactored)
Tab 1: Terrain/Heightmap Parameter
Alle Verbesserungen aus Schritt 1-3 implementiert
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QSpinBox, QPushButton,
                             QFrame, QGridLayout, QSpacerItem, QSizePolicy, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt

# Schritt 1: Neue gemeinsame Widgets
from gui.widgets.parameter_slider import ParameterSlider
from gui.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui.widgets.map_canvas import MapCanvas
from gui.utils.error_handler import ErrorHandler, safe_execute, TabErrorContext
from gui.utils.performance_utils import debounced_method, performance_tracked
try:
    from gui.managers.parameter_manager import WorldParameterManager
except ImportError:
    from gui.world_state import WorldState as WorldParameterManager


class TerrainMapCanvas(MapCanvas):
    """
    Funktionsweise: Terrain Map Canvas mit Performance-Optimierung
    - Erbt von OptimizedMapCanvas für automatisches Debouncing
    - Implementiert Heightmap-spezifische Visualisierung
    - Seed-basierte reproduzierbare Generierung
    """

    def __init__(self):
        super().__init__('terrain_map', title='Heightmap Preview')

    @performance_tracked("Terrain_Rendering")
    @safe_execute('handle_map_rendering_error')
    def _render_map(self, size, height, octaves, frequency, persistence, lacunarity, redistribute_power, seed=42):
        """
        Funktionsweise: Rendert Terrain-Heightmap mit Error Handling
        - Verwendet TabErrorContext für robuste Fehlerbehandlung
        - Performance-optimiert für verschiedene Terrain-Größen
        - Reproduzierbare Ergebnisse durch Seed
        """
        with TabErrorContext('Terrain', 'Heightmap Rendering'):
            self.clear_and_setup()

            try:
                # Farbintensität basierend auf Höhen-Parameter
                intensity = (height / 400.0) * 0.5 + 0.3  # 0.3 bis 0.8
                color_value = plt.cm.terrain(intensity)
                self.ax.set_facecolor(color_value)

                # Optimierte Noise-Visualisierung basierend auf Größe
                grid_size = min(max(size // 8, 25), 100)  # Adaptive Auflösung
                x = np.linspace(0, 100, grid_size)
                y = np.linspace(0, 100, grid_size)
                X, Y = np.meshgrid(x, y)

                # Reproduzierbarer Pseudo-Noise mit Seed
                np.random.seed(seed)

                # Multi-Oktaven Noise-Simulation
                Z = self._generate_heightmap_noise(X, Y, octaves, frequency, persistence, lacunarity)

                # Redistribution für realistische Höhenverteilung
                Z = self._apply_height_redistribution(Z, redistribute_power)

                # Höhen-Konturen zeichnen
                contour_levels = min(octaves * 2, 15)  # Adaptive Kontur-Anzahl
                contours = self.ax.contour(X, Y, Z, levels=contour_levels,
                                           alpha=0.7, colors='darkgreen', linewidths=0.8)

                # Höhenlinien-Labels bei hoher Detailstufe
                if size >= 256 and octaves >= 4:
                    self.ax.clabel(contours, inline=True, fontsize=8, fmt='%1.1f')

                # Titel mit aktuellen Parametern und Statistiken
                terrain_stats = self._calculate_terrain_stats(Z, height)
                title = f'Heightmap Preview (Size: {size}x{size}, Seed: {seed})\n{terrain_stats}'
                self.set_title(title)

            except Exception as e:
                self.error_handler.logger.error(f"Heightmap-Generierung fehlgeschlagen: {e}")
                # Fallback: Einfache Darstellung
                self.ax.set_facecolor('lightgreen')
                self.ax.text(0.5, 0.5, 'Heightmap\n(Vereinfachte Ansicht)',
                             transform=self.ax.transAxes, ha='center', va='center')

            self.draw()

    def _generate_heightmap_noise(self, X, Y, octaves, frequency, persistence, lacunarity):
        """
        Funktionsweise: Generiert multi-oktaven Perlin-ähnlichen Noise
        - Simuliert realistische Terrain-Features
        - Berücksichtigt alle Noise-Parameter
        """
        Z = np.zeros_like(X)
        amplitude = 1.0
        freq = frequency

        for octave in range(int(octaves)):
            # Oktaven-basierter Noise
            noise_layer = (np.sin(X * freq * 10) * np.cos(Y * freq * 10) +
                           np.sin(X * freq * 15 + 1.5) * np.cos(Y * freq * 12 + 2.1))

            Z += noise_layer * amplitude

            # Frequency und Amplitude für nächste Oktave
            amplitude *= persistence
            freq *= lacunarity

            # Kleine Zufallsverschiebung für mehr Variation
            freq += np.random.uniform(-0.001, 0.001)

        return Z

    def _apply_height_redistribution(self, Z, power):
        """
        Funktionsweise: Wendet Höhen-Redistribution an
        - Macht flache Bereiche flacher, steile Bereiche steiler
        - Simuliert realistische Terrain-Verteilung
        """
        # Normalisierung auf 0-1
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min()) if Z.max() != Z.min() else Z

        # Power-Redistribution
        Z_redistributed = Z_norm ** power

        # Zurück-Skalierung
        return Z_redistributed * (Z.max() - Z.min()) + Z.min()

    def _calculate_terrain_stats(self, Z, max_height):
        """
        Funktionsweise: Berechnet Terrain-Statistiken für Info-Anzeige
        Returns:
            str: Formatierte Statistik-Info
        """
        try:
            elevation_range = Z.max() - Z.min()
            avg_elevation = Z.mean()
            steep_areas = np.sum(np.abs(np.gradient(Z)[0]) > 0.1) / Z.size * 100

            return f"Höhenspanne: {elevation_range:.2f} | Steile Bereiche: {steep_areas:.1f}%"
        except:
            return "Höhenprofil: Standard"


class TerrainControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Terrain Control Panel mit allen Verbesserungen
    - Neue ParameterSlider (Schritt 1)
    - WorldParameterManager Integration (Schritt 2)
    - Performance-Optimierung mit Debouncing (Schritt 3)
    - Seed-Management und Terrain-Validierung
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.error_handler = ErrorHandler()

        # Verwende neuen WorldParameterManager
        self.world_manager = WorldParameterManager()
        self.terrain_manager = self.world_manager.terrain

        self.init_ui()

    @safe_execute('handle_parameter_error')
    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Simplex Höhenprofil")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Seed Management
        self.setup_seed_controls(layout)

        # Auto-Simulation Control
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_basic_parameters(layout)
        self.setup_noise_parameters(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation mit Mixin (Terrain ist erster Tab)
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="Hauptmenü", next_text="Weiter")

        self.setLayout(layout)

        # Initial preview
        self.update_preview()

    def setup_seed_controls(self, layout):
        """Erstellt Seed-Management Controls"""
        seed_layout = QGridLayout()
        seed_label = QLabel("Terrain Seed:")

        params = self.terrain_manager.get_parameters()

        self.seed_input = QSpinBox()
        self.seed_input.setMinimum(0)
        self.seed_input.setMaximum(999999)
        self.seed_input.setValue(params.get('seed', 42))
        self.seed_input.valueChanged.connect(self.on_parameter_changed)

        self.random_seed_btn = QPushButton("Zufällig")
        self.random_seed_btn.setStyleSheet(
            "QPushButton { background-color: #9b59b6; color: white; font-weight: bold; padding: 8px; }")
        self.random_seed_btn.clicked.connect(self.randomize_seed)

        seed_widget = QWidget()
        seed_layout.addWidget(seed_label, 0, 0)
        seed_layout.addWidget(self.seed_input, 0, 1)
        seed_layout.addWidget(self.random_seed_btn, 0, 2)
        seed_widget.setLayout(seed_layout)
        layout.addWidget(seed_widget)

    def setup_simulation_controls(self, layout):
        """Erstellt Auto-Simulation Controls"""
        sim_control_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Automat. Simulieren")
        self.auto_simulate_checkbox.setChecked(self.world_manager.ui_state.get_auto_simulate())
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Simulieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.simulate_now_btn.clicked.connect(self.simulate_now)
        self.simulate_now_btn.setEnabled(not self.world_manager.ui_state.get_auto_simulate())

        sim_control_widget = QWidget()
        sim_control_layout.addWidget(self.auto_simulate_checkbox)
        sim_control_layout.addWidget(self.simulate_now_btn)
        sim_control_widget.setLayout(sim_control_layout)
        layout.addWidget(sim_control_widget)

    def setup_basic_parameters(self, layout):
        """Erstellt grundlegende Terrain-Parameter"""
        basic_group = QGroupBox("Basis-Parameter")
        basic_layout = QVBoxLayout()

        params = self.terrain_manager.get_parameters()

        # Verwende neue ParameterSlider-Klasse
        self.size_slider = ParameterSlider("Kartengröße", 64, 1024, params['size'])
        self.height_slider = ParameterSlider("Max. Höhe", 0, 400, params['height'], suffix="m")

        basic_sliders = [self.size_slider, self.height_slider]

        for slider in basic_sliders:
            basic_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

    def setup_noise_parameters(self, layout):
        """Erstellt Noise-spezifische Parameter"""
        noise_group = QGroupBox("Noise-Parameter")
        noise_layout = QVBoxLayout()

        params = self.terrain_manager.get_parameters()

        self.octaves_slider = ParameterSlider("Oktaven", 1, 8, params['octaves'])
        self.frequency_slider = ParameterSlider("Grundfrequenz", 0.001, 0.050,
                                                params['frequency'], decimals=3)
        self.persistence_slider = ParameterSlider("Persistence", 0.1, 1.0,
                                                  params['persistence'], decimals=2)
        self.lacunarity_slider = ParameterSlider("Lacunarity", 1.0, 4.0,
                                                 params['lacunarity'], decimals=1)
        self.redistribute_slider = ParameterSlider("Höhen-Redistribution", 0.5, 3.0,
                                                   params['redistribute_power'], decimals=1)

        noise_sliders = [self.octaves_slider, self.frequency_slider, self.persistence_slider,
                         self.lacunarity_slider, self.redistribute_slider]

        for slider in noise_sliders:
            noise_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)

    @safe_execute('handle_parameter_error')
    def on_auto_simulate_changed(self, state):
        """Auto-Simulation Checkbox geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.error_handler.logger.info("Terrain Auto-Simulation aktiviert")
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.error_handler.logger.info("Terrain Auto-Simulation deaktiviert")
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(200)  # Kurzes Debouncing für Terrain
    def on_parameter_changed(self):
        """Parameter wurden geändert - mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

    @safe_execute('handle_parameter_error')
    def simulate_now(self):
        """Manuelle Simulation"""
        self.error_handler.logger.info("Terrain Simulation gestartet!")
        self.update_preview()

    @safe_execute('handle_parameter_error')
    def randomize_seed(self):
        """
        Funktionsweise: Generiert zufälligen Seed mit Auto-Update
        - Verwendet TerrainParameterManager für Seed-Generierung
        - Triggert Update falls Auto-Simulation aktiviert
        """
        new_seed = self.terrain_manager.randomize_seed()
        self.seed_input.setValue(new_seed)

        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()
        else:
            self.error_handler.logger.info(f"Neuer Seed: {new_seed} - Klicken Sie 'Jetzt Simulieren'")

    @performance_tracked("Terrain_Preview_Update")
    @safe_execute('handle_map_rendering_error')
    def update_preview(self):
        """
        Funktionsweise: Aktualisiert die Terrain-Kartenvorschau
        - Performance-optimiert mit Tracking
        - Verwendet optimierte Map Canvas
        - Robuste Parameter-Validierung
        """
        with TabErrorContext('Terrain', 'Preview Update'):
            params = self.get_parameters()

            # Validiere und speichere Parameter
            self.terrain_manager.set_parameters(params)

            # Aktualisiere Karte (mit automatischem Debouncing)
            self.map_canvas.update_map(**params)

    def get_parameters(self):
        """
        Funktionsweise: Sammelt alle Terrain Parameter
        - Verwendet neue ParameterSlider.get_value() Methode
        - Robuste Parameter-Sammlung mit Fallback
        """
        try:
            return {
                'size': self.size_slider.get_value(),
                'height': self.height_slider.get_value(),
                'octaves': self.octaves_slider.get_value(),
                'frequency': self.frequency_slider.get_value(),
                'persistence': self.persistence_slider.get_value(),
                'lacunarity': self.lacunarity_slider.get_value(),
                'redistribute_power': self.redistribute_slider.get_value(),
                'seed': self.seed_input.value()
            }
        except Exception as e:
            self.error_handler.handle_parameter_error('Terrain', 'parameter_collection', e)
            return self.terrain_manager.get_parameters()

    def reset_to_defaults(self):
        """
        Funktionsweise: Setzt alle Parameter auf Standardwerte zurück
        - Verwendet centralized DefaultConfig
        - Aktualisiert UI-Elemente
        """
        try:
            self.terrain_manager.reset_to_defaults()
            default_params = self.terrain_manager.get_parameters()

            # UI-Slider auf Standardwerte setzen
            self.size_slider.set_value(default_params['size'])
            self.height_slider.set_value(default_params['height'])
            self.octaves_slider.set_value(default_params['octaves'])
            self.frequency_slider.set_value(default_params['frequency'])
            self.persistence_slider.set_value(default_params['persistence'])
            self.lacunarity_slider.set_value(default_params['lacunarity'])
            self.redistribute_slider.set_value(default_params['redistribute_power'])
            self.seed_input.setValue(default_params['seed'])

            self.update_preview()
            self.error_handler.logger.info("Terrain auf Standardwerte zurückgesetzt")

        except Exception as e:
            self.error_handler.handle_parameter_error('Terrain', 'reset_defaults', e)

    # Navigation Methoden (von NavigationMixin erforderlich)
    def next_menu(self):
        """Wechselt zum nächsten Tab (Geology)"""
        try:
            params = self.get_parameters()
            self.terrain_manager.set_parameters(params)
            self.error_handler.logger.info("Terrain Parameter gespeichert")

            next_tab = TabNavigationHelper.get_next_tab('TerrainWindow')
            if next_tab:
                self.navigate_to_tab(next_tab[0], next_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Terrain', 'Geology', e)

    def prev_menu(self):
        """Geht zurück zum Hauptmenü (Terrain ist erster Tab)"""
        try:
            params = self.get_parameters()
            self.terrain_manager.set_parameters(params)

            # Direkte Navigation zum Hauptmenü
            from gui.main_menu import MainMenuWindow

            # Speichere Geometrie
            current_geometry = self.window().geometry()

            # Erstelle Hauptmenü
            self.main_menu = MainMenuWindow()
            self.main_menu.setGeometry(current_geometry)
            self.main_menu.show()

            # Schließe aktuelles Fenster
            self.window().close()

        except Exception as e:
            print(f"Fehler beim Wechsel zum Hauptmenü: {e}")
            # Fallback: Einfach schließen
            self.window().close()

    def restart_generation(self):
        """
        Funktionsweise: Überschreibt NavigationMixin Methode
        - Setzt Parameter zurück statt Tab-Wechsel
        - Bleibt auf Terrain Tab
        """
        self.error_handler.logger.info("Terrain neu gestartet")
        self.reset_to_defaults()

    @safe_execute('handle_parameter_error')
    def quick_generate(self):
        """Schnellgenerierung mit Terrain-Statistiken"""
        params = self.get_parameters()
        self.terrain_manager.set_parameters(params)
        heightmap_params = self.terrain_manager.get_heightmap_params()
        self.error_handler.logger.info(f"Terrain Schnellgenerierung: {heightmap_params}")


class TerrainWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für Terrain-Tab
    - Verwendet optimierte Komponenten
    - Erweiterte Styling für erste Tab-Erfahrung
    """

    def __init__(self):
        super().__init__()
        self.error_handler = ErrorHandler()
        self.init_ui()

    @safe_execute('handle_worldstate_error')
    def init_ui(self):
        self.setWindowTitle("World Generator - Terrain Setup")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Linke Seite - Optimierte Karte (70%)
        self.map_canvas = TerrainMapCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Controls (30%)
        self.control_panel = TerrainControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Erweiterte Styling für Terrain
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            /* Spezielle Styles für Terrain Parameter */
            QGroupBox[title="Basis-Parameter"] {
                border-color: #27ae60;
            }
            QGroupBox[title="Noise-Parameter"] {
                border-color: #8e44ad;
            }
            /* Slider Styling */
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        if hasattr(self, 'map_canvas'):
            self.map_canvas.cleanup()
        super().closeEvent(event)