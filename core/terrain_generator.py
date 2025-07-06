#!/usr/bin/env python3
"""
World Generator GUI - Hauptprogramm
Erste Seite: Terrain/Heightmap Parameter
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QSlider, QSpinBox, QPushButton,
                             QFrame, QGridLayout, QSpacerItem, QSizePolicy, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MapCanvas(QWidget):
    """Widget für die Kartendarstellung"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Matplotlib Figure und Canvas erstellen
        self.figure = Figure(figsize=(8, 8), facecolor='white')
        self.canvas = FigureCanvas(self.figure)

        # Grünes Quadrat als Platzhalter
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_facecolor('lightgreen')
        self.ax.set_title('Heightmap Preview')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        # Gitter für bessere Orientierung
        self.ax.grid(True, alpha=0.3)

        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_map(self, size, height, octaves, frequency, persistence, lacunarity, redistribute_power):
        """Aktualisiert die Karte basierend auf Parametern (Placeholder)"""
        self.ax.clear()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)

        # Einfache Visualisierung der Parameter
        # Hier würde später die echte Heightmap-Generierung stehen

        # Farbintensität basierend auf Parametern
        intensity = (height / 400.0) * 0.5 + 0.3  # 0.3 bis 0.8
        color_value = plt.cm.terrain(intensity)
        self.ax.set_facecolor(color_value)

        # Einfache "Noise"-Visualisierung
        x = np.linspace(0, 100, size // 4)
        y = np.linspace(0, 100, size // 4)
        X, Y = np.meshgrid(x, y)

        # Pseudo-Noise basierend auf Parametern
        Z = np.sin(X * frequency * 10) * np.cos(Y * frequency * 10) * persistence
        Z = Z ** redistribute_power

        self.ax.contour(X, Y, Z, levels=octaves, alpha=0.6, colors='darkgreen')

        self.ax.set_title(f'Heightmap Preview (Size: {size}x{size})')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()


class ParameterSlider(QWidget):
    """Custom Widget für Slider mit Label und Spinbox"""

    def __init__(self, label_text, min_val, max_val, default_val, decimals=0, step=1):
        super().__init__()
        self.decimals = decimals
        self.step_size = step
        self.init_ui(label_text, min_val, max_val, default_val)

    def init_ui(self, label_text, min_val, max_val, default_val):
        layout = QGridLayout()

        # Label
        self.label = QLabel(label_text)
        layout.addWidget(self.label, 0, 0, 1, 2)

        # Slider
        self.slider = QSlider(Qt.Horizontal)

        # Für Dezimalwerte: Slider mit 100x Multiplikator
        if self.decimals > 0:
            self.slider.setMinimum(int(min_val * (10 ** self.decimals)))
            self.slider.setMaximum(int(max_val * (10 ** self.decimals)))
            self.slider.setValue(int(default_val * (10 ** self.decimals)))
        else:
            self.slider.setMinimum(min_val)
            self.slider.setMaximum(max_val)
            self.slider.setValue(default_val)

        layout.addWidget(self.slider, 1, 0)

        # SpinBox
        if self.decimals > 0:
            self.spinbox = QSpinBox()
            self.spinbox.setMinimum(int(min_val * (10 ** self.decimals)))
            self.spinbox.setMaximum(int(max_val * (10 ** self.decimals)))
            self.spinbox.setValue(int(default_val * (10 ** self.decimals)))
            self.spinbox.setSuffix(f" ×10⁻{self.decimals}")
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setMinimum(min_val)
            self.spinbox.setMaximum(max_val)
            self.spinbox.setValue(default_val)

        self.spinbox.setMaximumWidth(100)
        layout.addWidget(self.spinbox, 1, 1)

        # Verbindungen
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spinbox.valueChanged.connect(self.on_spinbox_changed)

        self.setLayout(layout)

    def on_slider_changed(self, value):
        self.spinbox.setValue(value)

    def on_spinbox_changed(self, value):
        self.slider.setValue(value)

    def get_value(self):
        """Gibt den aktuellen Wert zurück"""
        if self.decimals > 0:
            return self.slider.value() / (10 ** self.decimals)
        return self.slider.value()

    def set_value(self, value):
        """Setzt einen neuen Wert"""
        if self.decimals > 0:
            self.slider.setValue(int(value * (10 ** self.decimals)))
        else:
            self.slider.setValue(value)


class ControlPanel(QWidget):
    """Rechte Seite mit den Parametern"""

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        # Import hier um zirkuläre Imports zu vermeiden
        from gui.world_state import WorldState
        self.world_state = WorldState()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Simplex Höhenprofil")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)

        # Seed Input
        seed_layout = QGridLayout()
        seed_label = QLabel("Seed:")
        self.seed_input = QSpinBox()
        self.seed_input.setMinimum(0)
        self.seed_input.setMaximum(999999)
        self.seed_input.setValue(self.world_state.terrain_params['seed'])
        self.random_seed_btn = QPushButton("Zufällig")
        self.random_seed_btn.clicked.connect(self.randomize_seed)

        seed_widget = QWidget()
        seed_layout.addWidget(seed_label, 0, 0)
        seed_layout.addWidget(self.seed_input, 0, 1)
        seed_layout.addWidget(self.random_seed_btn, 0, 2)
        seed_widget.setLayout(seed_layout)
        layout.addWidget(seed_widget)

        # Auto-Simulation Control
        sim_control_layout = QHBoxLayout()
        self.auto_simulate_checkbox = QCheckBox("Automat. Simulieren")
        self.auto_simulate_checkbox.setChecked(self.world_state.get_auto_simulate())
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Simulieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.simulate_now_btn.clicked.connect(self.simulate_now)
        self.simulate_now_btn.setEnabled(not self.world_state.get_auto_simulate())

        sim_control_widget = QWidget()
        sim_control_layout.addWidget(self.auto_simulate_checkbox)
        sim_control_layout.addWidget(self.simulate_now_btn)
        sim_control_widget.setLayout(sim_control_layout)
        layout.addWidget(sim_control_widget)

        # Parameter Slider mit Werten aus WorldState
        params = self.world_state.get_terrain_params()
        self.size_slider = ParameterSlider("Größe", 64, 1024, params['size'])
        self.height_slider = ParameterSlider("Höhe", 0, 400, params['height'])
        self.octaves_slider = ParameterSlider("Oktaven", 1, 8, params['octaves'])
        self.frequency_slider = ParameterSlider("Frequency", 1, 50, params['frequency'] * 1000, decimals=3)
        self.persistence_slider = ParameterSlider("Persistence", 1, 10, params['persistence'] * 10, decimals=1)
        self.lacunarity_slider = ParameterSlider("Lacunarity", 10, 40, params['lacunarity'] * 10, decimals=1)
        self.redistribute_slider = ParameterSlider("Redistribute Power", 5, 30, params['redistribute_power'] * 10, decimals=1)

        # Slider zu Layout hinzufügen
        sliders = [self.size_slider, self.height_slider, self.octaves_slider,
                   self.frequency_slider, self.persistence_slider,
                   self.lacunarity_slider, self.redistribute_slider]

        for slider in sliders:
            layout.addWidget(slider)
            slider.slider.valueChanged.connect(self.on_parameter_changed)

        # Initial preview
        self.update_preview()

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation Buttons
        from gui.navigation_widget import NavigationWidget
        self.navigation = NavigationWidget(show_prev=False, show_next=True, next_text="Weiter")
        self.navigation.next_clicked.connect(self.next_menu)
        self.navigation.quick_gen_clicked.connect(self.quick_generate)
        self.navigation.exit_clicked.connect(self.exit_app)
        layout.addWidget(self.navigation)

        self.setLayout(layout)

    def on_auto_simulate_changed(self, state):
        """Auto-Simulation Checkbox geändert"""
        is_checked = state == 2
        self.world_state.set_auto_simulate(is_checked)
        
        if is_checked:
            print("Auto-Simulation aktiviert")
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            print("Auto-Simulation deaktiviert")
            self.simulate_now_btn.setEnabled(True)

    def on_parameter_changed(self):
        """Parameter wurden geändert"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()

    def simulate_now(self):
        """Manuell ausgelöste Simulation"""
        print("Manuelle Simulation gestartet!")
        self.update_preview()

    def randomize_seed(self):
        """Generiert einen zufälligen Seed"""
        import random
        new_seed = random.randint(0, 999999)
        self.seed_input.setValue(new_seed)
        if self.auto_simulate_checkbox.isChecked():
            self.update_preview()
        else:
            print("Seed geändert - Klicken Sie 'Jetzt Simulieren' für Aktualisierung")

    def update_preview(self):
        """Aktualisiert die Kartenvorschau"""
        params = self.get_parameters()
        self.world_state.set_terrain_params(params)
        self.map_canvas.update_map(**params)

    def get_parameters(self):
        """Sammelt alle Parameter"""
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

    def quick_generate(self):
        """Schnellgenerierung - alle Schritte überspringen"""
        print("Schnellgenerierung gestartet!")
        params = self.get_parameters()
        self.world_state.set_terrain_params(params)
        print("Parameter:", params)

    def next_menu(self):
        """Zum nächsten Menü (Geologie)"""
        print("Wechsle zum Geology Menü")
        params = self.get_parameters()
        self.world_state.set_terrain_params(params)
        print("Parameter gespeichert:", params)

        # Fenstergeometrie speichern
        self.world_state.set_window_geometry(self.window().geometry())

        from gui.tabs.geology_tab import GeologyWindow
        self.geology_window = GeologyWindow()
        self.geology_window.setGeometry(self.world_state.get_window_geometry())
        self.geology_window.show()
        self.window().close()

    def exit_app(self):
        """Zurück zum Hauptmenü"""
        from gui.main_menu import MainMenuWindow
        self.main_menu = MainMenuWindow()
        self.main_menu.setGeometry(self.world_state.get_window_geometry())
        self.main_menu.show()
        self.window().close()


class TerrainWindow(QMainWindow):
    """Hauptfenster der Anwendung"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Terrain Setup")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout (Horizontal Split)
        main_layout = QHBoxLayout()

        # Linke Seite - Karte (70%)
        self.map_canvas = MapCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Controls (30%)
        self.control_panel = ControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #333;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
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
        # Hier könnte man die Proportionen erzwingen falls nötig