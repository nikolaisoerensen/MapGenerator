#!/usr/bin/env python3
"""
Path: MapGenerator/gui/tabs/water_tab.py
__init__.py existiert in "tabs"

World Generator GUI - Water Tab (Vollständig Refactored)
Tab 5: Flüsse, Seen und Wasserflächen
Alle Verbesserungen aus Schritt 1-3 implementiert
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QPushButton,
                             QFrame, QSpacerItem, QSizePolicy,
                             QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt

from gui.widgets.parameter_slider import ParameterSlider
from gui.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui.utils.error_handler import ErrorHandler, safe_execute, TabErrorContext
from gui.widgets.map_canvas import MapCanvas
from gui.utils.performance_utils import debounced_method, performance_tracked
try:
    from gui.managers.parameter_manager import WorldParameterManager
except ImportError:
    from gui.world_state import WorldState as WorldParameterManager


class WaterMapCanvas(MapCanvas):
    """
    Funktionsweise: Water Map Canvas mit Performance-Optimierung
    - Erbt von OptimizedMapCanvas für automatisches Debouncing
    - Komplexe Wassersystem-Visualisierung
    - Intelligente Erosions- und Sediment-Simulation
    """

    def __init__(self):
        super().__init__('water_map', title='Wassersystem')

    @performance_tracked("Water_Rendering")
    def _render_map(self, lake_fill, sea_level, sediment_amount, water_speed, rock_dependency):
        """
        Funktionsweise: Rendert komplexes Wassersystem
        - Terrain, Seen, Flüsse, Erosion in einem System
        - Performance-optimiert für komplexe Berechnungen
        """
        with TabErrorContext('Water', 'Water System Rendering'):
            self.clear_and_setup()

            # Terrain Hintergrund generieren
            X, Y, height_field = self._generate_terrain()

            # === MEERESSPIEGEL ===
            self._render_sea_level(X, Y, height_field, sea_level)

            # === SEEN ===
            self._render_lakes(X, Y, height_field, lake_fill, sea_level)

            # === FLUSS-NETZWERK ===
            self._render_river_network(water_speed, rock_dependency)

            # === EROSIONS-VISUALISIERUNG ===
            self._render_erosion_effects(sediment_amount, water_speed)

            # === WASSERDATEN-INFO ===
            self._add_water_info(lake_fill, sea_level, sediment_amount)

            self.set_title('Wassersystem: Flüsse, Seen & Erosion')
            self._add_water_legend()
            self.draw()

    def _generate_terrain(self):
        """
        Funktionsweise: Generiert Basis-Terrain für Wassersystem
        Returns:
            tuple: X, Y, height_field für weitere Berechnungen
        """
        x = np.linspace(0, 100, 80)
        y = np.linspace(0, 100, 80)
        X, Y = np.meshgrid(x, y)

        # Simuliertes Höhenfeld mit realistischen Features
        height_field = (np.sin(X * 0.08) * np.cos(Y * 0.06) +
                        np.sin(X * 0.12) * np.cos(Y * 0.10) * 0.5 +
                        np.random.normal(0, 0.05, X.shape))  # Weniger Rauschen
        height_field = (height_field + 1.2) * 0.4  # Normalisierung auf 0-0.8

        # Terrain-Konturen zeichnen
        terrain_contour = self.ax.contourf(X, Y, height_field, levels=20,
                                           cmap='terrain', alpha=0.6)
        return X, Y, height_field

    def _render_sea_level(self, X, Y, height_field, sea_level):
        """Rendert Meeresspiegel und Ozeane"""
        try:
            sea_height = sea_level / 100.0
            ocean_mask = height_field <= sea_height

            if np.any(ocean_mask):
                # Ozean in verschiedenen Blautönen je nach Tiefe
                ocean_depth = sea_height - height_field
                ocean_depth = np.where(ocean_mask, ocean_depth, 0)

                # Tiefere Bereiche dunkler
                self.ax.contourf(X, Y, ocean_depth, levels=10,
                                 cmap='Blues', alpha=0.8, vmin=0, vmax=sea_height)

        except Exception as e:
            self.error_handler.logger.warning(f"Meeresspiegel-Rendering Fehler: {e}")

    def _render_lakes(self, X, Y, height_field, lake_fill, sea_level):
        """Rendert Seen in geeigneten Vertiefungen"""
        try:
            sea_height = sea_level / 100.0
            lake_threshold = sea_height + (lake_fill / 100.0) * 0.3

            # Potentielle Seen-Bereiche
            potential_lakes = (height_field > sea_height) & (height_field < lake_threshold)

            # Intelligente Seen-Platzierung basierend auf Topographie
            lake_centers = self._find_lake_positions(height_field, potential_lakes, lake_fill)

            for i, (lx, ly) in enumerate(lake_centers):
                # Seen-Größe basierend auf lake_fill und lokaler Topographie
                local_depression = 1.0 - height_field[int(ly * 0.8), int(lx * 0.8)]
                lake_size = (lake_fill / 20.0) * (1 + local_depression)

                circle = Circle((lx, ly), lake_size, color='#1e40af', alpha=0.9)
                self.ax.add_patch(circle)

                # Seen-Label
                self.ax.text(lx, ly, f'S{i + 1}', ha='center', va='center',
                             fontsize=8, color='white', weight='bold')

        except Exception as e:
            self.error_handler.logger.warning(f"Seen-Rendering Fehler: {e}")

    def _find_lake_positions(self, height_field, potential_lakes, lake_fill):
        """
        Funktionsweise: Findet optimale Positionen für Seen
        - Basiert auf Topographie und lake_fill Parameter
        Returns:
            list: Liste von (x, y) Positionen für Seen
        """
        lake_positions = []
        max_lakes = min(int(lake_fill / 20) + 1, 5)  # 1-5 Seen basierend auf Parameter

        # Vordefinierte gute Positionen, gefiltert nach Topographie
        candidate_positions = [(25, 75), (70, 60), (45, 30), (80, 20), (15, 40)]

        for pos in candidate_positions[:max_lakes]:
            x, y = pos
            # Prüfe ob Position für See geeignet ist
            grid_x, grid_y = int(x * 0.8), int(y * 0.8)
            if 0 <= grid_x < potential_lakes.shape[1] and 0 <= grid_y < potential_lakes.shape[0]:
                if potential_lakes[grid_y, grid_x]:
                    lake_positions.append(pos)

        return lake_positions

    def _render_river_network(self, water_speed, rock_dependency):
        """Rendert komplexes Fluss-Netzwerk"""
        try:
            # Hauptfluss (West nach Ost) mit naturalistischer Kurve
            main_river_x = np.linspace(5, 95, 60)
            # Realistische Flussmäander
            meander_amplitude = 15 + (water_speed / 10) * 5
            main_river_y = (50 + np.sin(main_river_x * 0.08) * meander_amplitude +
                            np.random.normal(0, 1, len(main_river_x)))

            # Flussbreite basierend auf Wassergeschwindigkeit und Geologie
            base_width = max(1, water_speed / 3.0)
            geological_factor = 1 + (100 - rock_dependency) / 100.0  # Weichere Gesteine = breitere Flüsse
            river_width = base_width * geological_factor

            # Hauptfluss zeichnen
            self.ax.plot(main_river_x, main_river_y, color='#1e40af',
                         linewidth=river_width, alpha=0.9, label='Hauptfluss')

            # Nebenflüsse mit intelligenter Platzierung
            self._render_tributaries(main_river_x, main_river_y, river_width, rock_dependency)

            # Fluss-Features (Wasserfälle, Stromschnellen)
            if rock_dependency > 70:
                self._add_river_features(main_river_x, main_river_y)

        except Exception as e:
            self.error_handler.logger.warning(f"Fluss-Rendering Fehler: {e}")

    def _render_tributaries(self, main_river_x, main_river_y, main_width, rock_dependency):
        """Rendert Nebenflüsse mit realistischer Geometrie"""
        try:
            tributary_count = 3 + int(rock_dependency / 30)  # Mehr Nebenflüsse bei weicheren Gesteinen

            for i in range(min(tributary_count, 6)):  # Maximum 6 Nebenflüsse
                # Zufällige Einmündung in Hauptfluss
                confluence_idx = np.random.randint(10, len(main_river_x) - 10)
                confluence_x = main_river_x[confluence_idx]
                confluence_y = main_river_y[confluence_idx]

                # Nebenfluss-Ursprung
                if i % 2 == 0:  # Nord-Zuflüsse
                    source_x = confluence_x + np.random.uniform(-20, 20)
                    source_y = confluence_y + np.random.uniform(20, 40)
                else:  # Süd-Zuflüsse
                    source_x = confluence_x + np.random.uniform(-20, 20)
                    source_y = confluence_y - np.random.uniform(20, 40)

                # Nebenfluss-Verlauf
                trib_length = 20 + np.random.randint(0, 20)
                trib_x = np.linspace(source_x, confluence_x, trib_length)
                trib_y = np.linspace(source_y, confluence_y, trib_length)

                # Naturalistic Kurven
                trib_y += np.sin(trib_x * 0.2 + i) * 3

                # Nebenfluss-Breite (schmaler als Hauptfluss)
                trib_width = main_width * (0.3 + np.random.uniform(0, 0.3))

                self.ax.plot(trib_x, trib_y, color='#3b82f6',
                             linewidth=trib_width, alpha=0.8)

        except Exception as e:
            self.error_handler.logger.warning(f"Nebenfluss-Rendering Fehler: {e}")

    def _add_river_features(self, river_x, river_y):
        """Fügt spezielle Fluss-Features hinzu"""
        try:
            # Wasserfälle bei steilen Stellen (simuliert)
            for i in range(2):
                fall_idx = np.random.randint(10, len(river_x) - 10)
                fall_x, fall_y = river_x[fall_idx], river_y[fall_idx]

                # Wasserfall-Symbol
                self.ax.plot(fall_x, fall_y, marker='v', color='white',
                             markersize=6, markeredgecolor='blue', markeredgewidth=2)

        except Exception as e:
            self.error_handler.logger.warning(f"Fluss-Features Fehler: {e}")

    def _render_erosion_effects(self, sediment_amount, water_speed):
        """Rendert Erosions- und Sediment-Effekte"""
        try:
            if sediment_amount > 30:
                # Delta-Bildung am Meer
                delta_positions = [(95, 45), (90, 55)]  # Flussmündungen

                for dx, dy in delta_positions:
                    delta_size = sediment_amount / 20.0
                    delta = patches.FancyBboxPatch(
                        (dx - delta_size / 2, dy - delta_size / 2),
                        delta_size, delta_size,
                        boxstyle="round,pad=0.3",
                        facecolor='#fbbf24', alpha=0.7,
                        edgecolor='#92400e'
                    )
                    self.ax.add_patch(delta)

                    self.ax.text(dx, dy, 'Δ', ha='center', va='center',
                                 fontsize=10, weight='bold', color='brown')

            # Erosions-Linien entlang Flüsse bei hoher Wasserkraft
            if water_speed > 10:
                self._add_erosion_patterns()

        except Exception as e:
            self.error_handler.logger.warning(f"Erosions-Rendering Fehler: {e}")

    def _add_erosion_patterns(self):
        """Fügt Erosions-Muster hinzu"""
        try:
            # Seitenerosion entlang Hauptfluss
            erosion_x = np.linspace(20, 80, 30)
            erosion_y = 50 + np.sin(erosion_x * 0.1) * 12

            # Erosions-Linien
            for offset in [-3, 3]:
                self.ax.plot(erosion_x, erosion_y + offset,
                             color='#8B4513', linestyle=':', alpha=0.6, linewidth=1)

        except Exception as e:
            self.error_handler.logger.warning(f"Erosions-Muster Fehler: {e}")

    def _add_water_info(self, lake_fill, sea_level, sediment_amount):
        """Fügt Wasser-Informationen zur Karte hinzu"""
        try:
            # Wasser-Coverage Berechnung
            water_coverage = min(sea_level * 0.5 + lake_fill * 0.1, 100)

            # Info-Text in oberer linker Ecke
            info_text = f"Wasser: {water_coverage:.0f}%\n Meereshöhe: {sea_level}%\nSediment: {sediment_amount}%"

            self.ax.text(5, 95, info_text, fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                         verticalalignment='top')

        except Exception as e:
            self.error_handler.logger.warning(f"Wasser-Info Fehler: {e}")

    def _add_water_legend(self):
        """Fügt professionelle Wasser-Legende hinzu"""
        try:
            legend_elements = [
                plt.Line2D([0], [0], color='#1e40af', linewidth=3, label='Hauptfluss'),
                plt.Line2D([0], [0], color='#3b82f6', linewidth=2, label='Nebenfluss'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1e40af',
                           markersize=8, label='See'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#1e3a8a', label='Ozean'),
                plt.Rectangle((0, 0), 1, 1, facecolor='#fbbf24', alpha=0.6, label='Sediment/Delta')
            ]

            self.ax.legend(handles=legend_elements, loc='upper left', fontsize=8,
                           framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0, 0.85))

        except Exception as e:
            self.error_handler.logger.warning(f"Wasser-Legende Fehler: {e}")


class WaterControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Water Control Panel mit allen Verbesserungen
    - Neue ParameterSlider (Schritt 1)
    - WorldParameterManager Integration (Schritt 2)
    - Performance-Optimierung mit Debouncing (Schritt 3)
    - Wasser-Coverage Berechnung und Validierung
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.error_handler = ErrorHandler()

        # Verwende neuen WorldParameterManager
        self.world_manager = WorldParameterManager()
        self.water_manager = self.world_manager.water

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Flüsse & Wasserflächen")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Auto-Simulation Control
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_water_body_parameters(layout)
        self.setup_erosion_parameters(layout)

        # Wasser-Coverage Info
        self.setup_water_coverage_info(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation mit Mixin
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="Zurück", next_text="Weiter")

        self.setLayout(layout)

        # Initial preview
        self.update_preview()

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

    def setup_water_body_parameters(self, layout):
        """Erstellt Wasserkörper-Parameter"""
        water_group = QGroupBox("Wasserkörper")
        water_layout = QVBoxLayout()

        params = self.water_manager.get_parameters()

        self.lake_fill_slider = ParameterSlider("Seefüllung", 0, 100,
                                                params['lake_fill'], suffix="%")
        self.sea_level_slider = ParameterSlider("Meeresspiegel", 0, 50,
                                                params['sea_level'], suffix="%")

        water_sliders = [self.lake_fill_slider, self.sea_level_slider]

        for slider in water_sliders:
            water_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        water_group.setLayout(water_layout)
        layout.addWidget(water_group)

    def setup_erosion_parameters(self, layout):
        """Erstellt Tal-Erosion Parameter"""
        erosion_group = QGroupBox("Tal-Erosion")
        erosion_layout = QVBoxLayout()

        params = self.water_manager.get_parameters()

        self.sediment_slider = ParameterSlider("Sedimentmenge", 0, 100,
                                               params['sediment_amount'], suffix="%")
        self.water_speed_slider = ParameterSlider("Wassergeschwindigkeit", 1.0, 20.0,
                                                  params['water_speed'], decimals=1, suffix="m/s")
        self.rock_dependency_slider = ParameterSlider("Gesteinsabhängigkeit", 0, 100,
                                                      params['rock_dependency'], suffix="%")

        erosion_sliders = [self.sediment_slider, self.water_speed_slider,
                           self.rock_dependency_slider]

        for slider in erosion_sliders:
            erosion_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        erosion_group.setLayout(erosion_layout)
        layout.addWidget(erosion_group)

    def setup_water_coverage_info(self, layout):
        """Erstellt Wasser-Coverage Informations-Panel"""
        coverage_group = QGroupBox("Wasser-Coverage")
        coverage_layout = QVBoxLayout()

        self.coverage_label = QLabel("Berechne...")
        self.coverage_label.setStyleSheet("""
            QLabel {
                background-color: #e8f6ff;
                border: 2px solid #1e40af;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        self.coverage_label.setAlignment(Qt.AlignCenter)

        coverage_layout.addWidget(self.coverage_label)
        coverage_group.setLayout(coverage_layout)
        layout.addWidget(coverage_group)

    @safe_execute('handle_parameter_error')
    def on_auto_simulate_changed(self, state):
        """Auto-Simulation Checkbox geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.error_handler.logger.info("Water Auto-Simulation aktiviert")
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.error_handler.logger.info("Water Auto-Simulation deaktiviert")
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(250)  # Debouncing für Water-Berechnungen
    def on_parameter_changed(self):
        """Parameter wurden geändert - mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

        # Aktualisiere Water-Coverage sofort (ohne Debouncing)
        self.update_water_coverage()

    @safe_execute('handle_parameter_error')
    def simulate_now(self):
        """Manuelle Simulation"""
        self.error_handler.logger.info("Water Simulation gestartet!")
        self.update_preview()

    @performance_tracked("Water_Preview_Update")
    @safe_execute('handle_map_rendering_error')
    def update_preview(self):
        """
        Funktionsweise: Aktualisiert die Wasser-Kartenvorschau
        - Performance-optimiert für komplexe Wassersystem-Berechnungen
        - Validiert Wasser-Parameter
        """
        with TabErrorContext('Water', 'Preview Update'):
            params = self.get_parameters()

            # Validiere und speichere Parameter
            self.water_manager.set_parameters(params)

            # Aktualisiere Water-Coverage
            self.update_water_coverage()

            # Aktualisiere Karte (mit automatischem Debouncing)
            self.map_canvas.update_map(**params)

    def update_water_coverage(self):
        """
        Funktionsweise: Aktualisiert Water-Coverage Berechnung in Echtzeit
        - Berechnet geschätzten Wasser-Anteil der Karte
        - Visuelles Feedback für realistische Werte
        """
        try:
            params = self.get_parameters()
            self.water_manager.set_parameters(params)

            # Hole Water-Coverage vom Manager
            coverage = self.water_manager.calculate_water_coverage()

            # Bewerte Coverage für Feedback
            if coverage < 20:
                coverage_type = "Trocken"
                color = "#e67e22"
            elif coverage < 40:
                coverage_type = "Moderat"
                color = "#f39c12"
            elif coverage < 60:
                coverage_type = "Ausgewogen"
                color = "#3498db"
            else:
                coverage_type = "Sehr Feucht"
                color = "#2980b9"

            self.coverage_label.setText(f"{coverage_type}\n{coverage:.1f}% Wasser")
            self.coverage_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {color}20;
                    border: 2px solid {color};
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 12px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
            """)

        except Exception as e:
            self.error_handler.logger.warning(f"Water-Coverage Update Fehler: {e}")

    def get_parameters(self):
        """
        Funktionsweise: Sammelt alle Water Parameter
        - Verwendet neue ParameterSlider API
        - Robuste Parameter-Sammlung
        """
        try:
            return {
                'lake_fill': self.lake_fill_slider.get_value(),
                'sea_level': self.sea_level_slider.get_value(),
                'sediment_amount': self.sediment_slider.get_value(),
                'water_speed': self.water_speed_slider.get_value(),
                'rock_dependency': self.rock_dependency_slider.get_value()
            }
        except Exception as e:
            self.error_handler.handle_parameter_error('Water', 'parameter_collection', e)
            return self.water_manager.get_parameters()

    # Navigation Methoden (von NavigationMixin erforderlich)
    def next_menu(self):
        """Wechselt zum nächsten Tab (Biome)"""
        try:
            params = self.get_parameters()
            self.water_manager.set_parameters(params)
            self.error_handler.logger.info("Water Parameter gespeichert")

            next_tab = TabNavigationHelper.get_next_tab('WaterWindow')
            if next_tab:
                self.navigate_to_tab(next_tab[0], next_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Water', 'Biome', e)

    def prev_menu(self):
        """Wechselt zum vorherigen Tab (Weather)"""
        try:
            params = self.get_parameters()
            self.water_manager.set_parameters(params)

            prev_tab = TabNavigationHelper.get_prev_tab('WaterWindow')
            if prev_tab:
                self.navigate_to_tab(prev_tab[0], prev_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Water', 'Weather', e)

    @safe_execute('handle_parameter_error')
    def quick_generate(self):
        """Schnellgenerierung mit Water-Coverage Info"""
        params = self.get_parameters()
        self.water_manager.set_parameters(params)
        coverage = self.water_manager.calculate_water_coverage()
        self.error_handler.logger.info(f"Water Schnellgenerierung: {coverage:.1f}% Coverage - {params}")


class WaterWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für Water-Tab
    - Verwendet optimierte Map Canvas
    - Robuste Initialisierung für komplexes Wassersystem
    """

    def __init__(self):
        super().__init__()
        self.error_handler = ErrorHandler()
        self.init_ui()

    @safe_execute('handle_worldstate_error')
    def init_ui(self):
        self.setWindowTitle("World Generator - Flüsse & Wasserflächen")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Linke Seite - Optimierte Karte (70%)
        self.map_canvas = WaterMapCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Controls (30%)
        self.control_panel = WaterControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Styling für Water-spezifische Elemente
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
                /* Spezielle Styles für Water Parameter */
                QGroupBox[title="Wasserkörper"] {
                    border-color: #1e40af;
                }
                QGroupBox[title="Tal-Erosion"] {
                    border-color: #92400e;
                }
                QGroupBox[title="Wasser-Coverage"] {
                    border-color: #3b82f6;
                    background-color: #f8fafc;
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