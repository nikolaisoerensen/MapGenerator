#!/usr/bin/env python3
"""
Path: MapGenerator/gui/tabs/geology_tab.py
__init__.py existiert in "tabs"

World Generator GUI - Geology Tab (Vollständig Refactored)
Tab 2: Gesteinsarten und geologische Eigenschaften
Alle Verbesserungen aus Schritt 1-3 implementiert - KORRIGIERTE VERSION
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QPushButton,
                             QFrame, QSpacerItem, QSizePolicy,
                             QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.colors import ListedColormap

# Schritt 1: Neue gemeinsame Widgets
from gui.widgets.parameter_slider import ParameterSlider
from gui.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui.utils.error_handler import ErrorHandler, safe_execute, TabErrorContext
from gui.widgets.map_canvas import MapCanvas
from gui.utils.performance_utils import debounced_method, performance_tracked

try:
    from gui.managers.parameter_manager import WorldParameterManager
except ImportError:
    from gui.world_state import WorldState as WorldParameterManager


class RockTypeWidget(QWidget):
    """
    Funktionsweise: Widget für einzelne Gesteinsart-Konfiguration
    - Checkbox für Aktivierung/Deaktivierung
    - Härte-Slider mit Live-Anzeige
    - Konsistente Parameter-Schnittstelle
    """

    def __init__(self, name, default_hardness=50):
        super().__init__()
        self.name = name
        self.init_ui(default_hardness)

    def init_ui(self, default_hardness):
        layout = QHBoxLayout()

        # Checkbox für Aktivierung
        self.enabled_cb = QCheckBox(self.name)
        self.enabled_cb.setChecked(True)
        layout.addWidget(self.enabled_cb)

        # Härte-Slider mit ParameterSlider
        self.hardness_slider = ParameterSlider("", 1, 100, default_hardness, suffix="")
        self.hardness_slider.setMaximumWidth(200)
        layout.addWidget(self.hardness_slider)

        self.setLayout(layout)

    def is_enabled(self):
        return self.enabled_cb.isChecked()

    def get_hardness(self):
        return self.hardness_slider.get_value()

    def set_hardness(self, value):
        self.hardness_slider.set_value(value)


class GeologyMapCanvas(MapCanvas):
    """
    Funktionsweise: Geology Map Canvas mit Performance-Optimierung
    - Erbt von OptimizedMapCanvas für automatisches Debouncing
    - Implementiert geologische Schichten-Visualisierung
    - Intelligente Gesteins-Klassifizierung und -Visualisierung
    """

    def __init__(self):
        super().__init__('geology_map', title='Geologische Karte')
        self.rock_type_colors = {
            'Sedimentäres Gestein': '#8B4513',  # Braun
            'Metamorphes Gestein': '#696969',  # Grau
            'Magmatisches Gestein': '#2F4F4F',  # DunklerGrau
            'Vulkanisches Gestein': '#800080',  # Lila
            'Kalkstein': '#F5DEB3'  # Beige
        }

    @performance_tracked("Geology_Rendering")
    @safe_execute('handle_map_rendering_error')
    def _render_map(self, rock_types, hardness_values, ridge_warping, bevel_warping):
        """
        Funktionsweise: Rendert geologische Karte mit Error Handling
        - Verwendet TabErrorContext für robuste Fehlerbehandlung
        - Performance-optimiert für komplexe geologische Berechnungen
        - Visualisiert Gesteinsschichten mit realistischen Formationen
        """
        with TabErrorContext('Geology', 'Geological Map Rendering'):
            self.clear_and_setup()

            # Koordinaten-Mesh generieren
            x = np.linspace(0, 100, 60)
            y = np.linspace(0, 100, 60)
            X, Y = np.meshgrid(x, y)

            # === GEOLOGISCHE FORMATIONEN ===
            geological_layers = self._generate_geological_layers(X, Y, ridge_warping, bevel_warping)

            # === GESTEINS-KLASSIFIZIERUNG ===
            rock_classification = self._classify_rock_formations(geological_layers, rock_types, hardness_values)

            # === GEOLOGISCHE KARTE RENDERN ===
            self._render_geological_map(X, Y, rock_classification, rock_types)

            # === TEKTONISCHE STRUKTUREN ===
            self._render_tectonic_features(X, Y, ridge_warping, bevel_warping)

            # === HÄRTE-VISUALISIERUNG ===
            self._render_hardness_overlay(X, Y, rock_classification, hardness_values)

            # === GEOLOGISCHE STATISTIKEN ===
            self._add_geological_statistics(rock_types, hardness_values, rock_classification)

            self.set_title('Geologische Karte: Gesteinsarten & Tektonik')
            self._add_geology_legend(rock_types)
            self.draw()

    def _generate_geological_layers(self, X, Y, ridge_warping, bevel_warping):
        """
        Funktionsweise: Generiert realistische geologische Schichten
        - Multi-Oktaven Geological Noise
        - Berücksichtigt tektonische Deformation
        Returns:
            dict: Verschiedene geologische Formationen
        """
        try:
            # Sedimentäre Schichten (horizontal)
            sediment_layer = np.sin(X * 0.08) * np.cos(Y * 0.06) + \
                             np.sin(X * 0.12) * np.cos(Y * 0.10) * 0.5

            # Metamorphe Deformation (durch Ridge Warping beeinflusst)
            metamorphic_layer = np.sin(X * 0.15 + ridge_warping * 3) * \
                                np.cos(Y * 0.12 + ridge_warping * 2) + \
                                np.random.normal(0, 0.1, X.shape)

            # Magmatische Intrusionen (durch Bevel Warping beeinflusst)
            igneous_layer = np.sin(X * 0.05 + bevel_warping * 4) * \
                            np.cos(Y * 0.08 + bevel_warping * 3) + \
                            np.sin(X * 0.20) * np.cos(Y * 0.18) * 0.3

            # Vulkanische Aktivität (lokale Hotspots)
            volcanic_centers = [(25, 75), (70, 30), (45, 50)]
            volcanic_layer = np.zeros_like(X)

            for vcx, vcy in volcanic_centers:
                distance = np.sqrt((X - vcx) ** 2 + (Y - vcy) ** 2)
                volcanic_layer += np.exp(-distance / (10 + bevel_warping * 5)) * \
                                  (1 + ridge_warping * 0.5)

            return {
                'sedimentary': sediment_layer,
                'metamorphic': metamorphic_layer,
                'igneous': igneous_layer,
                'volcanic': volcanic_layer
            }

        except Exception as e:
            self.error_handler.logger.warning(f"Geologische Schicht-Generierung Fehler: {e}")
            # Fallback: Einfache Schichten
            return {
                'sedimentary': np.sin(X * 0.1) * np.cos(Y * 0.1),
                'metamorphic': np.random.normal(0, 0.2, X.shape),
                'igneous': np.zeros_like(X),
                'volcanic': np.zeros_like(X)
            }

    def _classify_rock_formations(self, geological_layers, rock_types, hardness_values):
        """
        Funktionsweise: Klassifiziert Gesteinsformationen
        - Basiert auf dominanten geologischen Prozessen
        - Berücksichtigt aktivierte Gesteinsarten
        Returns:
            numpy.ndarray: Rock type classification map
        """
        try:
            if not rock_types:
                # Fallback wenn keine Gesteinsarten aktiviert
                return np.zeros_like(geological_layers['sedimentary'], dtype=int)

            sediment = geological_layers['sedimentary']
            metamorphic = geological_layers['metamorphic']
            igneous = geological_layers['igneous']
            volcanic = geological_layers['volcanic']

            # Normalisierung der Schichten
            layers = [sediment, metamorphic, igneous, volcanic]
            for i, layer in enumerate(layers):
                if layer.max() != layer.min():
                    layers[i] = (layer - layer.min()) / (layer.max() - layer.min())

            sediment, metamorphic, igneous, volcanic = layers

            # Gesteins-Klassifizierung basierend auf dominanten Prozessen
            classification = np.zeros_like(sediment, dtype=int)

            # Mappings zwischen verfügbaren Gesteinsarten und Schichten
            rock_layer_mapping = {
                'Sedimentäres Gestein': sediment,
                'Metamorphes Gestein': metamorphic,
                'Magmatisches Gestein': igneous,
                'Vulkanisches Gestein': volcanic,
                'Kalkstein': sediment * 0.8  # Spezielle sedimentäre Variante
            }

            # Klassifiziere jede Position
            for i in range(sediment.shape[0]):
                for j in range(sediment.shape[1]):
                    # Finde dominante Formation für diese Position
                    formation_strengths = []

                    for idx, rock_type in enumerate(rock_types):
                        if rock_type in rock_layer_mapping:
                            strength = rock_layer_mapping[rock_type][i, j]
                            formation_strengths.append((strength, idx))

                    if formation_strengths:
                        # Wähle stärkste Formation
                        dominant_formation = max(formation_strengths, key=lambda x: x[0])
                        classification[i, j] = dominant_formation[1]

            return classification

        except Exception as e:
            self.error_handler.logger.warning(f"Gesteins-Klassifizierung Fehler: {e}")
            return np.zeros_like(geological_layers['sedimentary'], dtype=int)

    def _render_geological_map(self, X, Y, rock_classification, rock_types):
        """Rendert die Hauptgeologie-Karte mit Gesteinsarten"""
        try:
            if not rock_types:
                # Fallback: Einfache Terrain-Darstellung
                self.ax.set_facecolor('lightgray')
                self.ax.text(0.5, 0.5, 'Keine Gesteinsarten\naktiviert',
                             transform=self.ax.transAxes, ha='center', va='center',
                             fontsize=14, weight='bold')
                return

            # Erstelle Custom Colormap basierend auf aktivierten Gesteinsarten
            colors = []
            for rock_type in rock_types:
                color = self.rock_type_colors.get(rock_type, '#CCCCCC')
                colors.append(color)

            if colors:
                rock_cmap = ListedColormap(colors)

                # Render Gesteins-Karte
                rock_map = self.ax.imshow(rock_classification, cmap=rock_cmap,
                                          vmin=0, vmax=len(rock_types) - 1,
                                          origin='lower', extent=[0, 100, 0, 100],
                                          alpha=0.8, interpolation='bilinear')

                # Kontur-Linien für bessere Abgrenzung
                self.ax.contour(X, Y, rock_classification, levels=len(rock_types),
                                colors='black', alpha=0.3, linewidths=0.5)

        except Exception as e:
            self.error_handler.logger.warning(f"Geologische Karten-Rendering Fehler: {e}")

    def _render_tectonic_features(self, X, Y, ridge_warping, bevel_warping):
        """Rendert tektonische Strukturen"""
        try:
            # Verwerfungslinien bei starker Ridge Warping
            if ridge_warping > 0.5:
                fault_lines_x = [20, 80, 50]
                fault_lines_y = [10 + ridge_warping * 20, 90 - ridge_warping * 15, 50]

                for i, (fx, fy) in enumerate(zip(fault_lines_x, fault_lines_y)):
                    # Verwerfungslinie
                    fault_x = np.linspace(fx - 15, fx + 15, 20)
                    fault_y = np.full_like(fault_x, fy) + np.sin(fault_x * 0.3) * 3

                    self.ax.plot(fault_x, fault_y, color='red', linewidth=2,
                                 alpha=0.8, linestyle='--', label='Verwerfung' if i == 0 else "")

            # Falten-Strukturen bei starker Bevel Warping
            if bevel_warping > 0.4:
                fold_x = np.linspace(10, 90, 50)
                fold_y = 30 + np.sin(fold_x * 0.15 + bevel_warping * 2) * (5 + bevel_warping * 10)

                self.ax.plot(fold_x, fold_y, color='orange', linewidth=2,
                             alpha=0.7, label='Falten-Struktur')

                # Parallel-Falte
                fold_y2 = 70 + np.sin(fold_x * 0.15 + bevel_warping * 2 + np.pi) * (3 + bevel_warping * 8)
                self.ax.plot(fold_x, fold_y2, color='orange', linewidth=2, alpha=0.7)

        except Exception as e:
            self.error_handler.logger.warning(f"Tektonische Strukturen Fehler: {e}")

    def _render_hardness_overlay(self, X, Y, rock_classification, hardness_values):
        """Rendert Gesteinshärte als Overlay"""
        try:
            if not hardness_values:
                return

            # Erstelle Härte-Karte basierend auf Klassifizierung
            hardness_map = np.zeros_like(rock_classification, dtype=float)

            for i in range(len(hardness_values)):
                mask = (rock_classification == i)
                hardness_map[mask] = hardness_values[i]

            # Härte-Konturen (nur bei signifikanten Unterschieden)
            if len(set(hardness_values)) > 1:
                hardness_contours = self.ax.contour(X, Y, hardness_map,
                                                    levels=5, colors='white',
                                                    alpha=0.4, linewidths=1)

                # Härte-Labels
                self.ax.clabel(hardness_contours, inline=True, fontsize=8,
                               fmt='H:%1.0f', colors='white')

        except Exception as e:
            self.error_handler.logger.warning(f"Härte-Overlay Fehler: {e}")

    def _add_geological_statistics(self, rock_types, hardness_values, rock_classification):
        """Fügt geologische Statistiken zur Karte hinzu"""
        try:
            if not rock_types:
                return

            # Berechne Gesteinsverteilung
            unique, counts = np.unique(rock_classification, return_counts=True)
            total_pixels = rock_classification.size

            # Durchschnittliche Härte
            avg_hardness = np.mean(hardness_values) if hardness_values else 50

            # Tektonische Aktivität (basierend auf Parametern)
            # Diese Info würde normalerweise aus den Parametern kommen

            stats_text = f"GEOLOGIE:\n"
            stats_text += f"Avg Härte: {avg_hardness:.0f}\n"
            stats_text += f"{len(rock_types)} Gesteinsarten"

            self.ax.text(2, 98, stats_text, fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                         verticalalignment='top', zorder=20)

        except Exception as e:
            self.error_handler.logger.warning(f"Geologische Statistiken Fehler: {e}")

    def _add_geology_legend(self, rock_types):
        """Fügt professionelle Geologie-Legende hinzu"""
        try:
            legend_elements = []

            # Gesteinsarten
            for rock_type in rock_types:
                color = self.rock_type_colors.get(rock_type, '#CCCCCC')
                legend_elements.append(
                    mpatches.Patch(color=color, label=rock_type)
                )

            # Tektonische Features
            legend_elements.extend([
                plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                           label='Verwerfung'),
                plt.Line2D([0], [0], color='orange', linewidth=2,
                           label='Falten-Struktur')
            ])

            if legend_elements:
                self.ax.legend(handles=legend_elements, loc='center left',
                               bbox_to_anchor=(1.02, 0.5), fontsize=8,
                               framealpha=0.95, edgecolor='gray')

        except Exception as e:
            self.error_handler.logger.warning(f"Geologie-Legende Fehler: {e}")


class GeologyControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Geology Control Panel mit allen Verbesserungen
    - Neue ParameterSlider (Schritt 1)
    - WorldParameterManager Integration (Schritt 2)
    - Performance-Optimierung mit Debouncing (Schritt 3)
    - Konsistente Navigation und Error Handling
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.error_handler = ErrorHandler()

        # Verwende neuen WorldParameterManager
        self.world_manager = WorldParameterManager()
        self.geology_manager = self.world_manager.geology

        self.init_ui()

    @safe_execute('handle_parameter_error')
    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Geologie & Gesteinsarten")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #8b4513;")
        layout.addWidget(title)

        # Auto-Simulation Control
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_rock_type_parameters(layout)
        self.setup_deformation_parameters(layout)

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

    def setup_rock_type_parameters(self, layout):
        """Erstellt Gesteinsarten-Parameter mit RockTypeWidgets"""
        rock_group = QGroupBox("Gesteinsarten & Härte")
        rock_layout = QVBoxLayout()

        # Standard Gesteinsarten mit Default-Härten
        rock_types_defaults = [
            ("Sedimentäres Gestein", 30),
            ("Metamorphes Gestein", 60),
            ("Magmatisches Gestein", 80),
            ("Vulkanisches Gestein", 70),
            ("Kalkstein", 25)
        ]

        self.rock_widgets = []
        params = self.geology_manager.get_parameters()
        stored_rock_types = params.get('rock_types', [])
        stored_hardness = params.get('hardness_values', [])

        for i, (rock_name, default_hardness) in enumerate(rock_types_defaults):
            # Verwende gespeicherte Werte falls vorhanden
            if i < len(stored_hardness):
                hardness = stored_hardness[i]
            else:
                hardness = default_hardness

            rock_widget = RockTypeWidget(rock_name, hardness)

            # Setze Aktivierungsstatus
            if stored_rock_types and rock_name not in stored_rock_types:
                rock_widget.enabled_cb.setChecked(False)

            self.rock_widgets.append(rock_widget)
            rock_layout.addWidget(rock_widget)

            # Event-Verbindungen
            rock_widget.enabled_cb.stateChanged.connect(self.on_parameter_changed)
            rock_widget.hardness_slider.valueChanged.connect(self.on_parameter_changed)

        rock_group.setLayout(rock_layout)
        layout.addWidget(rock_group)

    def setup_deformation_parameters(self, layout):
        """Erstellt tektonische Deformations-Parameter"""
        deform_group = QGroupBox("Tektonische Deformation")
        deform_layout = QVBoxLayout()

        params = self.geology_manager.get_parameters()

        # Ridge Warping
        self.ridge_slider = ParameterSlider("Ridge Warping", 0.0, 1.0,
                                            params.get('ridge_warping', 0.25), decimals=2)

        # Bevel Warping
        self.bevel_slider = ParameterSlider("Bevel Warping", 0.0, 1.0,
                                            params.get('bevel_warping', 0.15), decimals=2)

        deform_sliders = [self.ridge_slider, self.bevel_slider]

        for slider in deform_sliders:
            deform_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        deform_group.setLayout(deform_layout)
        layout.addWidget(deform_group)

    @safe_execute('handle_parameter_error')
    def on_auto_simulate_changed(self, state):
        """Auto-Simulation Checkbox geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.error_handler.logger.info("Geology Auto-Simulation aktiviert")
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.error_handler.logger.info("Geology Auto-Simulation deaktiviert")
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(250)  # Debouncing für Geology-Berechnungen
    def on_parameter_changed(self):
        """Parameter wurden geändert - mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

    @safe_execute('handle_parameter_error')
    def simulate_now(self):
        """Manuelle Simulation"""
        self.error_handler.logger.info("Geology Simulation gestartet!")
        self.update_preview()

    @performance_tracked("Geology_Preview_Update")
    @safe_execute('handle_map_rendering_error')
    def update_preview(self):
        """
        Funktionsweise: Aktualisiert die Geologie-Kartenvorschau
        - Performance-optimiert mit Tracking
        - Verwendet optimierte Map Canvas
        - Robuste Parameter-Validierung
        """
        with TabErrorContext('Geology', 'Preview Update'):
            params = self.get_parameters()

            # Validiere und speichere Parameter
            self.geology_manager.set_parameters(params)

            # Aktualisiere Karte (mit automatischem Debouncing)
            self.map_canvas.update_map(**params)

    def get_parameters(self):
        """
        Funktionsweise: Sammelt alle Geology Parameter
        - Sammelt aktivierte Gesteinsarten und ihre Härten
        - Robuste Parameter-Sammlung mit Fallback
        """
        try:
            rock_types = []
            hardness_values = []

            for widget in self.rock_widgets:
                if widget.is_enabled():
                    rock_types.append(widget.name)
                    hardness_values.append(widget.get_hardness())

            return {
                'rock_types': rock_types,
                'hardness_values': hardness_values,
                'ridge_warping': self.ridge_slider.get_value(),
                'bevel_warping': self.bevel_slider.get_value()
            }

        except Exception as e:
            # Korrigierte API: tab_name, param_name, error
            self.error_handler.handle_parameter_error('Geology', 'parameter_collection', e)
            # Fallback zu Standard-Parametern
            return {
                'rock_types': ['Sedimentäres Gestein', 'Metamorphes Gestein'],
                'hardness_values': [30, 60],
                'ridge_warping': 0.25,
                'bevel_warping': 0.15
            }

    # Navigation Methoden (von NavigationMixin erforderlich)
    def next_menu(self):
        """Wechselt zum nächsten Tab (Settlement)"""
        try:
            params = self.get_parameters()
            self.geology_manager.set_parameters(params)
            self.error_handler.logger.info("Geology Parameter gespeichert")

            next_tab = TabNavigationHelper.get_next_tab('GeologyWindow')
            if next_tab:
                self.navigate_to_tab(next_tab[0], next_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Geology', 'Settlement', e)

    def prev_menu(self):
        """Wechselt zum vorherigen Tab (Terrain)"""
        try:
            params = self.get_parameters()
            self.geology_manager.set_parameters(params)

            prev_tab = TabNavigationHelper.get_prev_tab('GeologyWindow')
            if prev_tab:
                self.navigate_to_tab(prev_tab[0], prev_tab[1])
        except Exception as e:
            self.error_handler.handle_tab_navigation_error('Geology', 'Terrain', e)

    @safe_execute('handle_parameter_error')
    def quick_generate(self):
        """Schnellgenerierung mit Geologie-Statistiken"""
        try:
            params = self.get_parameters()
            self.geology_manager.set_parameters(params)
            rock_count = len(params['rock_types'])
            avg_hardness = np.mean(params['hardness_values']) if params['hardness_values'] else 50
            self.error_handler.logger.info(
                f"Geology Schnellgenerierung: {rock_count} Gesteinsarten, Ø Härte: {avg_hardness:.0f}")
        except Exception as e:
            self.error_handler.handle_parameter_error('Geology', 'quick_generate', e)


class GeologyWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für Geology-Tab
    - Verwendet optimierte Map Canvas
    - Konsistente Initialisierung und Styling
    """

    def __init__(self):
        super().__init__()
        self.error_handler = ErrorHandler()
        self.init_ui()

    @safe_execute('handle_worldstate_error')
    def init_ui(self):
        self.setWindowTitle("World Generator - Geologie & Gesteinsarten")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Linke Seite - Optimierte Karte (70%)
        self.map_canvas = GeologyMapCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Controls (30%)
        self.control_panel = GeologyControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Styling für Geology-spezifische Elemente
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
            }
            QLabel {
                color: #1e293b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 12px;
                background-color: rgba(255, 255, 255, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #334155;
            }
            /* Spezielle Styles für Geology Parameter */
            QGroupBox[title*="Gesteinsarten"] {
                border-color: #8b4513;
                background-color: rgba(139, 69, 19, 0.05);
            }
            QGroupBox[title*="Tektonische"] {
                border-color: #dc2626;
                background-color: rgba(220, 38, 38, 0.05);
            }
            /* Button Hover-Effekte */
            QPushButton:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            /* Geology Titel-Style */
            QLabel[text*="Geologie"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8b4513, stop:1 #d2691e);
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
            /* RockTypeWidget Styling */
            QCheckBox {
                font-weight: bold;
                color: #4a5568;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #a0aec0;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: #8b4513;
                border-color: #8b4513;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMCAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggMkw0IDZMMiA0IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }
            QCheckBox::indicator:unchecked:hover {
                border-color: #8b4513;
                background-color: rgba(139, 69, 19, 0.1);
            }
        """)

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        try:
            if hasattr(self, 'map_canvas'):
                self.map_canvas.cleanup()
        except Exception as e:
            self.error_handler.logger.warning(f"Geology Cleanup Fehler: {e}")
        finally:
            # Wichtig: Event an Parent weiterleiten
            super().closeEvent(event)