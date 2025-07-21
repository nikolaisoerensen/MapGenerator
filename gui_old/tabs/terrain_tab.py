#!/usr/bin/env python3
"""
Vollständig integrierter Terrain Tab - Core-Integration mit 2D/3D Ansicht
BEISPIEL für die Integration aller anderen Tabs
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QSpinBox, QPushButton,
                             QFrame, QGridLayout, QSpacerItem, QSizePolicy, QCheckBox,
                             QGroupBox, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

from gui_old.managers.parameter_manager import WorldParameterManager
from gui_old.widgets.map_canvas import TerrainCanvas
from gui_old.widgets.parameter_slider import ParameterSlider
from gui_old.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from core import BaseTerrainGenerator, create_preview_heightmap, validate_terrain_parameters


class TerrainControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Erweiterte Control Panel mit Core-Parameter Integration
    - Alle Original Terrain-Parameter verfügbar
    - Live-Update mit der integrierten Canvas
    - Datenübertragung an nachfolgende Tabs
    - Erweiterte Statistiken und Info-Displays
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas                            # Referenz zur 2D/3D Canvas
        self.world_manager = WorldParameterManager()            # Singleton Parameter-Manager
        params = self.world_manager.terrain.get_parameters()    # Lade gespeicherte Parameter
        self.world_manager.terrain.set_parameters(params)       # Setze sie (mit Validierung)

        # Verbinde Canvas Signal für Datenübertragung
        self.map_canvas.heightmap_generated.connect(self.on_heightmap_generated)
        self.map_canvas.data_generated.connect(self.on_data_generated)

        # Debouncing Timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.perform_live_update)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # === HEADER ===
        title = QLabel("Terrain Generator")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 15px; color: #2e8b57;")
        layout.addWidget(title)

        # === SEED MANAGEMENT ===
        self.setup_seed_controls(layout)

        # === AUTO-SIMULATION ===
        self.setup_simulation_controls(layout)

        # === PARAMETER GRUPPEN ===
        self.setup_basic_parameters(layout)
        self.setup_noise_parameters(layout)
        self.setup_advanced_parameters(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # === NAVIGATION ===
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="Hauptmenü", next_text="Geology →")

        self.setLayout(layout)

        # Initial preview
        self.perform_live_update()

    def setup_seed_controls(self, layout):
        """Erweiterte Seed-Kontrollen"""
        seed_group = QGroupBox("Terrain Seed")
        seed_layout = QGridLayout()

        params = self.world_manager.terrain.get_parameters()

        self.seed_input = QSpinBox()
        self.seed_input.setMinimum(0)
        self.seed_input.setMaximum(999999)
        self.seed_input.setValue(params.get('seed', 42))
        self.seed_input.setStyleSheet("font-size: 12px; padding: 5px;")
        self.seed_input.valueChanged.connect(self.on_parameter_changed)

        self.random_seed_btn = QPushButton("Zufällig")
        self.random_seed_btn.setStyleSheet(
            "QPushButton { background-color: #9b59b6; color: white; font-weight: bold; padding: 8px; }")
        self.random_seed_btn.clicked.connect(self.randomize_seed)

        seed_layout.addWidget(QLabel("Seed:"), 0, 0)
        seed_layout.addWidget(self.seed_input, 0, 1)
        seed_layout.addWidget(self.random_seed_btn, 0, 2)

        seed_group.setLayout(seed_layout)
        layout.addWidget(seed_group)

    def setup_simulation_controls(self, layout):
        """Auto-Simulation Controls"""
        sim_group = QGroupBox("⚡ Live Update")
        sim_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Auto-Simulation")
        self.auto_simulate_checkbox.setChecked(
            WorldParameterManager().ui_state.get_auto_simulate()
        )
        self.auto_simulate_checkbox.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Generieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #e67e22; color: white; font-weight: bold; padding: 10px; }")
        self.simulate_now_btn.clicked.connect(self.simulate_now)
        self.simulate_now_btn.setEnabled(not self.world_manager.get_auto_simulate())

        sim_layout.addWidget(self.auto_simulate_checkbox)
        sim_layout.addWidget(self.simulate_now_btn)
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

    def setup_basic_parameters(self, layout):
        """Basis-Parameter Gruppe"""
        basic_group = QGroupBox("Basis-Parameter")
        basic_layout = QVBoxLayout()

        params = self.world_manager.get_terrain_params()

        # KARTENGRÖSSE mit 32er Steps
        self.size_slider = ParameterSlider("Kartengröße", 64, 512, params['size'], step=32)

        # HÖHE mit 10er Steps: 0, 10, 20, 30, ..., 400
        self.height_slider = ParameterSlider("Max. Höhe", 0, 400, params['height'], suffix="m", step=10)

        basic_sliders = [self.size_slider, self.height_slider]

        for slider in basic_sliders:
            basic_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

    def setup_noise_parameters(self, layout):
        """Noise-Parameter Gruppe"""
        noise_group = QGroupBox("Simplex Noise Parameter")
        noise_layout = QVBoxLayout()

        params = self.world_manager.get_terrain_params()

        self.octaves_slider = ParameterSlider("Oktaven (Detail)", 1, 10, params['octaves'])
        self.frequency_slider = ParameterSlider("Grundfrequenz", 0.001, 0.050,
                                                params['frequency'], decimals=3)
        self.persistence_slider = ParameterSlider("Persistence (Rauheit)", 0.1, 1.0,
                                                  params['persistence'], decimals=2)
        self.lacunarity_slider = ParameterSlider("Lacunarity (Detailskalierung)", 1.0, 4.0,
                                                 params['lacunarity'], decimals=1)

        noise_sliders = [self.octaves_slider, self.frequency_slider,
                         self.persistence_slider, self.lacunarity_slider]

        for slider in noise_sliders:
            noise_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)

    def setup_advanced_parameters(self, layout):
        """Erweiterte Parameter"""
        advanced_group = QGroupBox("Erweitert")
        advanced_layout = QVBoxLayout()

        params = self.world_manager.get_terrain_params()

        self.redistribute_slider = ParameterSlider("Höhen-Redistribution", 0.5, 3.0,
                                                   params['redistribute_power'], decimals=1)

        advanced_layout.addWidget(self.redistribute_slider)
        self.redistribute_slider.valueChanged.connect(self.on_parameter_changed)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

    def on_heightmap_generated(self, heightmap):
        """Callback wenn neue Heightmap generiert wurde"""
        try:
            if self.map_canvas.terrain_generator:
                stats = self.map_canvas.terrain_generator.get_terrain_stats(heightmap)
            else:
                stats = self.map_canvas._calculate_fallback_stats(heightmap)

            # Aktualisiere Statistiken Display
            self.update_statistics_display(stats)

            # Speichere in WorldState für nachfolgende Tabs
            self.world_manager.set_terrain_params(self.get_parameters())

        except Exception as e:
            print(f"Heightmap callback Fehler: {e}")

    def on_data_generated(self, data):
        """Callback für data_generated Signal"""
        pass

    def randomize_seed(self):
        """Generiert zufälligen Seed"""
        import random
        new_seed = random.randint(0, 999999)
        self.seed_input.setValue(new_seed)

    def on_auto_simulate_changed(self, state):
        """Auto-Simulation Toggle"""
        is_checked = state == 2
        self.world_manager.set_auto_simulate(is_checked)

        self.simulate_now_btn.setEnabled(not is_checked)

        if is_checked:
            print("✓ Terrain Auto-Update aktiviert")
            self.perform_live_update()
        else:
            print("Terrain Auto-Update deaktiviert")

    def on_parameter_changed(self):
        """Parameter geändert - mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_timer.stop()
            self.update_timer.start(300)  # 300ms Debouncing für Terrain

    def simulate_now(self):
        """Manuelle Generierung"""
        print("Terrain Generierung gestartet!")
        self.perform_live_update()

    def perform_live_update(self):
        """Führt Live Update der Terrain Canvas durch"""
        try:
            params = self.get_parameters()

            self.map_canvas.update_map(**params)

            print(f"✓ Terrain generiert: Seed {params.get('seed', 42)}, Size {params.get('size', 256)}")
        except Exception as e:
            print(f"Live Update Fehler: {e}")

    def get_parameters(self):
        """Sammelt alle Terrain Parameter"""
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
            print(f"Parameter sammeln Fehler: {e}")
            return self.world_manager.get_terrain_params()

    # Navigation Methods (NavigationMixin Interface)
    def next_menu(self):
        """Wechselt zum Geology Tab ohne Pickle-Serialisierung"""
        params = self.get_parameters()
        self.world_manager.set_terrain_params(params)

        # Heightmap direkt im world_manager speichern ohne Serialisierung
        heightmap = self.map_canvas.get_heightmap_for_next_tab()
        if heightmap is not None:
            # Speichere Heightmap direkt im Parameter-Manager
            self.world_manager.terrain._heightmap_cache = heightmap
            params['heightmap_available'] = True
            self.world_manager.set_terrain_params(params)

        # Navigation
        from gui_old.tabs.geology_tab import GeologyWindow
        current_geometry = self.window().geometry()

        self.geology_window = GeologyWindow()
        self.geology_window.setGeometry(current_geometry)
        self.geology_window.show()
        self.window().close()

    def prev_menu(self):
        """Zurück zum Hauptmenü"""
        try:
            params = self.get_parameters()
            self.world_manager.set_terrain_params(params)

            from gui_old.main_menu import MainMenuWindow
            current_geometry = self.window().geometry()
            self.main_menu = MainMenuWindow()
            self.main_menu.setGeometry(current_geometry)
            self.main_menu.show()
            self.window().close()

        except Exception as e:
            print(f"Navigation zu Hauptmenü Fehler: {e}")

    def quick_generate(self):
        """Schnellgenerierung mit finaler Heightmap"""
        try:
            params = self.get_parameters()
            self.world_manager.set_terrain_params(params)

            # Finale Generierung mit voller Auflösung (falls Live Preview kleiner war)
            final_generator = BaseTerrainGenerator(params['size'], params['size'], params['seed'])
            final_heightmap = final_generator.generate_heightmap(**params)
            final_stats = final_generator.get_terrain_stats(final_heightmap)

            print("Finale Terrain Generierung abgeschlossen!")
            print(f"   Finale Größe: {params['size']}x{params['size']}")
            print(f"   Höhenbereich: {final_stats['min_height']:.1f}m - {final_stats['max_height']:.1f}m")
            print(f"   Berge: {final_stats.get('mountain_percentage', 0):.1f}%")
            print(f"   Flach: {final_stats.get('flat_percentage', 0):.1f}%")

        except Exception as e:
            print(f"Schnellgenerierung Fehler: {e}")


class TerrainWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für integrierten Terrain Tab
    - Verwendet neue 2D/3D Canvas
    - Erweiterte Controls mit Core-Integration
    - Optimiert für Live-Updates und Datenübertragung
    """

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Integrierter Terrain Generator (2D/3D Live)")
        self.setGeometry(100, 100, 1600, 1100)
        self.setMinimumSize(1600, 1100)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # === LINKE SEITE: 2D/3D CANVAS (75%) ===
        self.map_canvas = TerrainCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setLineWidth(2)
        main_layout.addWidget(separator)

        # === RECHTE SEITE: ERWEITERTE CONTROLS (25%) ===
        self.control_panel = TerrainControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(400)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # === ERWEITERTE STYLING ===
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f0f8ff, stop:1 #e6f3ff);
            }
            QLabel {
                color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: rgba(255, 255, 255, 0.95);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #2980b9;
                font-size: 13px;
            }
            QGroupBox[title*="Basis"] {
                border-color: #27ae60;
            }
            QGroupBox[title*="Noise"] {
                border-color: #8e44ad;
            }
            QGroupBox[title*="Erweitert"] {
                border-color: #e67e22;
            }
            QGroupBox[title*="Statistiken"] {
                border-color: #34495e;
                background-color: rgba(236, 240, 241, 0.95);
            }
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ecf0f1, stop:1 #d5dbdb);
                margin: 2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #3498db, stop:1 #2980b9);
                border: 1px solid #2980b9;
                width: 20px;
                margin: -2px 0;
                border-radius: 5px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #5dade2, stop:1 #3498db);
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            QPushButton:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            QPushButton:pressed {
                transform: translateY(1px);
            }
        """)

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        try:
            # Matplotlib figures schließen
            plt.close('all')
        except:
            pass
        super().closeEvent(event)


# === MAIN EXECUTION FÜR TESTING ===
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Teste den integrierten Terrain Tab
    window = TerrainWindow()
    window.show()

    sys.exit(app.exec_())