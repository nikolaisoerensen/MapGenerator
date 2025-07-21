#!/usr/bin/env python3
"""
Geology Tab - 3D Textured Terrain
REPARIERTE VERSION - Alle Crash-Bugs behoben
"""

import sys
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSpacerItem, QSizePolicy,
    QCheckBox, QGroupBox, QProgressBar, QMainWindow, QFrame
)

# KORRIGIERTE IMPORTS
from gui_old.widgets.map_canvas import GeologyCanvas  # Richtiger Name!
from gui_old.widgets.parameter_slider import ParameterSlider
from gui_old.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui_old.managers.parameter_manager import WorldParameterManager

# Versuche Performance Utils (optional)
try:
    from gui_old.utils.performance_utils import debounced_method, performance_tracked
except ImportError:
    # Fallback wenn Performance Utils fehlen
    def debounced_method(ms):
        def decorator(func):
            return func

        return decorator


    def performance_tracked(name):
        def decorator(func):
            return func

        return decorator


class RockTypeWidget(QWidget):
    """Widget für einzelne Gesteinsart-Konfiguration"""

    def __init__(self, name, default_hardness=50):
        super().__init__()
        self.name = name
        self.init_ui(default_hardness)

    def init_ui(self, default_hardness):
        layout = QHBoxLayout()

        self.enabled_cb = QCheckBox(self.name)
        self.enabled_cb.setChecked(True)
        layout.addWidget(self.enabled_cb)

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


class GeologyControlPanel(QWidget, NavigationMixin):
    """Geology Control Panel mit 3D Textur-Integration"""

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.world_manager = WorldParameterManager()
        self.geology_manager = self.world_manager.geology

        # REPARATUR: Debounce Timer initialisieren
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.perform_update)

        self.geology_heightmap()

        self.init_ui()


    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Geology & 3D Textured Terrain")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #8b4513;")
        layout.addWidget(title)

        # Input Status
        self.input_status_label = QLabel("Warte auf Terrain-Daten...")
        self.input_status_label.setStyleSheet("""
            QLabel {
                background-color: #fff3cd;
                border: 2px solid #ffeaa7;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.input_status_label)

        # Auto-Simulation
        self.setup_simulation_controls(layout)

        # Parameter Gruppen
        self.setup_rock_type_parameters(layout)
        self.setup_deformation_parameters(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="← Terrain", next_text="Settlement →")

        self.setLayout(layout)

        # Initial preview
        self.update_preview()

    def geology_heightmap(self):
        """
        Ersetzt geology_heightmap() in GeologyControlPanel
        Lädt Heightmap direkt aus world_manager ohne Pickle
        """
        terrain_manager = self.world_manager.terrain

        # Prüfe ob Heightmap im Cache verfügbar
        if hasattr(terrain_manager, '_heightmap_cache'):
            heightmap = terrain_manager._heightmap_cache
            self.map_canvas.set_input_data('heightmap', heightmap)
            terrain_params = terrain_manager.get_parameters()
            self.map_canvas.set_input_data('terrain_params', terrain_params)
            return True

        # Prüfe ob Heightmap regeneriert werden kann
        terrain_params = terrain_manager.get_parameters()
        if terrain_params.get('heightmap_available', False):
            from core.terrain_generator import BaseTerrainGenerator
            size = terrain_params.get('size', 256)
            seed = terrain_params.get('seed', 42)

            generator = BaseTerrainGenerator(size, size, seed)
            heightmap = generator.generate_heightmap(**terrain_params)

            self.map_canvas.set_input_data('heightmap', heightmap)
            self.map_canvas.set_input_data('terrain_params', terrain_params)
            return True

        return False

    def setup_simulation_controls(self, layout):
        """Auto-Simulation Controls"""
        sim_control_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Auto-Simulation")
        self.auto_simulate_checkbox.setChecked(
            self.world_manager.ui_state.get_auto_simulate()
        )
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Simulieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        # REPARATUR: Korrekte Signal-Verbindung (ohne Klammern!)
        self.simulate_now_btn.clicked.connect(self.simulate_now)
        self.simulate_now_btn.setEnabled(not self.world_manager.ui_state.get_auto_simulate())

        sim_control_widget = QWidget()
        sim_control_layout.addWidget(self.auto_simulate_checkbox)
        sim_control_layout.addWidget(self.simulate_now_btn)
        sim_control_widget.setLayout(sim_control_layout)
        layout.addWidget(sim_control_widget)

    def setup_rock_type_parameters(self, layout):
        """Gesteinsarten-Parameter"""
        rock_group = QGroupBox("Gesteinsarten & Härte")
        rock_layout = QVBoxLayout()

        # Standard Gesteinsarten
        rock_types_defaults = [
            ("Sedimentäres Gestein", 30),
            ("Metamorphes Gestein", 60),
            ("Magmatisches Gestein", 80)
        ]

        self.rock_widgets = []
        params = self.geology_manager.get_parameters()
        stored_rock_types = params.get('rock_types', [])
        stored_hardness = params.get('hardness_values', [])

        for i, (rock_name, default_hardness) in enumerate(rock_types_defaults):
            hardness = stored_hardness[i] if i < len(stored_hardness) else default_hardness

            rock_widget = RockTypeWidget(rock_name, hardness)

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
        """Tektonische Deformations-Parameter"""
        deform_group = QGroupBox("Tektonische Deformation")
        deform_layout = QVBoxLayout()

        params = self.geology_manager.get_parameters()

        self.ridge_slider = ParameterSlider("Ridge Warping", 0.0, 1.0,
                                            params.get('ridge_warping', 0.25), decimals=2)
        self.bevel_slider = ParameterSlider("Bevel Warping", 0.0, 1.0,
                                            params.get('bevel_warping', 0.15), decimals=2)

        deform_sliders = [self.ridge_slider, self.bevel_slider]

        for slider in deform_sliders:
            deform_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        deform_group.setLayout(deform_layout)
        layout.addWidget(deform_group)

    def on_auto_simulate_changed(self, state):
        """Auto-Simulation geändert"""
        is_checked = state == 2
        self.world_manager.ui_state.set_auto_simulate(is_checked)

        if is_checked:
            self.simulate_now_btn.setEnabled(False)
            self.update_preview()
        else:
            self.simulate_now_btn.setEnabled(True)

    @debounced_method(250)
    def on_parameter_changed(self):
        """Parameter geändert mit Debouncing"""
        if self.auto_simulate_checkbox.isChecked():
            self.update_timer.start(250)  # Trigger debounced update

    def simulate_now(self):
        """Manuelle Simulation"""
        print("🔧 Geology: Manuelle Simulation gestartet")
        self.update_preview()

    def perform_update(self):
        """Debounced Update-Ausführung"""
        self.update_preview()

    @performance_tracked("Geology_Preview_Update")
    def update_preview(self):
        """
        REPARIERTE VERSION: Aktualisiert 3D Textured Terrain Preview
        """
        print("🏔️ Geology: Update Preview gestartet")

        # REPARATUR: Prüfe Heightmap-Input
        if not self._has_heightmap_input():
            self.input_status_label.setText("⚠ Keine Terrain-Daten - Gehe zu Terrain Tab")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    border: 2px solid #dc3545;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)

            # FALLBACK: Einfache Demo-Karte ohne Terrain
            self._show_demo_geology()
            return

        # Parameter sammeln
        geology_params = self.get_parameters()
        print(f"🔧 Geology Parameter: {len(geology_params)} params")

        try:
            from core.geology_generator import GeologyGenerator

            # Hole Heightmap vom Canvas
            heightmap = self.map_canvas.input_data['heightmap']
            h, w = heightmap.shape
            geology_gen = GeologyGenerator(w, h, geology_params.get('seed', 42))

            # ECHTE Geology generieren
            geology_result = geology_gen.generate_geology_map(heightmap, **geology_params)

            # Canvas aktualisieren
            self.map_canvas.set_input_data('geology', geology_result)
            self.map_canvas.update_map(**geology_params)

            self.input_status_label.setText("✅ Geology generiert mit Core")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #d4edda;
                    border: 2px solid #28a745;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)

        except ImportError:
            print("⚠ Core Geology Generator nicht verfügbar - verwende Fallback")
            self._generate_fallback_geology()

        except Exception as e:
            print(f"❌ Geology Generation Fehler: {e}")
            self.input_status_label.setText(f"❌ Fehler: {str(e)[:30]}...")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    border: 2px solid #dc3545;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)

    def _has_heightmap_input(self):
        """
        VERBESSERTE VERSION: Prüft ob Heightmap verfügbar ist
        """
        has_heightmap = (hasattr(self.map_canvas, 'input_data') and
                         'heightmap' in self.map_canvas.input_data and
                         self.map_canvas.input_data['heightmap'] is not None)

        if has_heightmap:
            heightmap = self.map_canvas.input_data['heightmap']
            print(f"Heightmap verfügbar: {heightmap.shape}")
        else:
            print("Keine Heightmap verfügbar")

        return has_heightmap

    def _show_demo_geology(self):
        """Zeigt Demo-Geology ohne Terrain-Daten"""
        print("🎭 Zeige Demo-Geology")
        demo_params = self.get_parameters()
        demo_params['demo_mode'] = True
        self.map_canvas.update_map(**demo_params)

    def _generate_fallback_geology(self):
        """Fallback Geology ohne Core"""
        print("🔄 Fallback Geology Generation")

        if self._has_heightmap_input():
            heightmap = self.map_canvas.input_data['heightmap']

            # Einfache Rock-Map basierend auf Höhe
            normalized_height = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
            rock_map = np.zeros_like(heightmap, dtype=int)

            # Einfache Höhen-basierte Klassifikation
            rock_map[normalized_height < 0.3] = 0  # Sedimentär (niedrig)
            rock_map[(normalized_height >= 0.3) & (normalized_height < 0.7)] = 1  # Metamorph (mittel)
            rock_map[normalized_height >= 0.7] = 2  # Magmatisch (hoch)

            geology_result = {'rock_map': rock_map}
            self.map_canvas.set_input_data('geology', geology_result)

            params = self.get_parameters()
            self.map_canvas.update_map(**params)

            self.input_status_label.setText("✅ Fallback Geology generiert")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3cd;
                    border: 2px solid #ffeaa7;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)

    def _generate_fallback_heightmap(self):
        """
        FALLBACK: Generiert neue Heightmap wenn keine vom Terrain Tab kommt
        """
        try:
            from core import BaseTerrainGenerator

            # Hole Terrain-Parameter (ohne heightmap_data)
            terrain_params = self.world_manager.terrain.get_parameters()

            # Entferne heightmap_data wenn vorhanden
            if 'heightmap_data' in terrain_params:
                del terrain_params['heightmap_data']

            print(f"Generiere Fallback-Heightmap mit Parametern: {terrain_params}")

            # Generiere neue Heightmap
            size = terrain_params.get('size', 256)
            seed = terrain_params.get('seed', 42)

            generator = BaseTerrainGenerator(size, size, seed)
            heightmap = generator.generate_heightmap(**terrain_params)

            print(f"Fallback-Heightmap generiert: {heightmap.shape}")

            # Setze in Canvas
            self.map_canvas.set_input_data('heightmap', heightmap)
            self.map_canvas.set_input_data('terrain_params', terrain_params)

            return True

        except ImportError:
            print("Core TerrainGenerator nicht verfügbar")
            return False
        except Exception as e:
            print(f"Fallback-Heightmap Generation fehlgeschlagen: {e}")
            return False

    def get_parameters(self):
        """Sammelt alle Geology Parameter"""
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
            'bevel_warping': self.bevel_slider.get_value(),
            'seed': 42  # Default seed
        }

    def set_heightmap_input(self, heightmap):
        """Empfängt Heightmap vom Terrain Tab"""
        print(f"📨 Geology: Heightmap empfangen {heightmap.shape}")
        if hasattr(self.map_canvas, 'set_input_data'):
            self.map_canvas.set_input_data('heightmap', heightmap)
            self.update_preview()

    # Navigation Methods
    def next_menu(self):
        """Wechselt zu Settlement Tab"""
        params = self.get_parameters()
        self.geology_manager.set_parameters(params)
        print("➡️ Navigation: Geology → Settlement")

        next_tab = TabNavigationHelper.get_next_tab('GeologyWindow')
        if next_tab:
            self.navigate_to_tab(next_tab[0], next_tab[1])

    def prev_menu(self):
        """Zurück zu Terrain Tab"""
        params = self.get_parameters()
        self.geology_manager.set_parameters(params)
        print("⬅️ Navigation: Geology → Terrain")

        prev_tab = TabNavigationHelper.get_prev_tab('GeologyWindow')
        if prev_tab:
            self.navigate_to_tab(prev_tab[0], prev_tab[1])

    def quick_generate(self):
        """Schnellgenerierung"""
        params = self.get_parameters()
        self.geology_manager.set_parameters(params)
        rock_count = len(params['rock_types'])
        print(f"⚡ Geology Quick Generate: {rock_count} Gesteinsarten")


class GeologyWindow(QMainWindow):
    """Geology Tab Hauptfenster - REPARIERT"""

    def __init__(self):
        super().__init__()
        print("🏗️ GeologyWindow: Initialisierung gestartet")
        try:
            self.init_ui()
            print("✅ GeologyWindow: Erfolgreich initialisiert")
        except Exception as e:
            print(f"❌ GeologyWindow Init Fehler: {e}")
            import traceback
            traceback.print_exc()
            raise

    def init_ui(self):
        self.setWindowTitle("World Generator - Geology & 3D Textured Terrain")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Linke Seite: 3D Textured Canvas (70%)
        try:
            print("🖼️ Erstelle GeologyCanvas...")
            self.map_canvas = GeologyCanvas()
            print("✅ GeologyCanvas erfolgreich erstellt")
        except Exception as e:
            print(f"❌ GeologyCanvas Fehler: {e}")
            # Fallback: Einfaches Widget
            self.map_canvas = QWidget()
            self.map_canvas.setStyleSheet("background-color: #f0f0f0;")
            label = QLabel("Geology Canvas konnte nicht geladen werden")
            layout = QVBoxLayout()
            layout.addWidget(label)
            self.map_canvas.setLayout(layout)

        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite: Controls (30%)
        try:
            print("🎛️ Erstelle GeologyControlPanel...")
            self.control_panel = GeologyControlPanel(self.map_canvas)
            self.control_panel.setMaximumWidth(350)
            print("✅ GeologyControlPanel erfolgreich erstellt")
        except Exception as e:
            print(f"❌ GeologyControlPanel Fehler: {e}")
            # Fallback: Einfaches Panel
            self.control_panel = QWidget()
            self.control_panel.setMaximumWidth(350)

        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Geology-spezifisches Styling
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
            }
            QGroupBox[title*="Gesteinsarten"] {
                border-color: #8b4513;
                background-color: rgba(139, 69, 19, 0.05);
            }
            QGroupBox[title*="Tektonische"] {
                border-color: #dc2626;
                background-color: rgba(220, 38, 38, 0.05);
            }
        """)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        try:
            print("🧹 GeologyWindow: Cleanup...")
            if hasattr(self, 'map_canvas') and hasattr(self.map_canvas, 'cleanup'):
                self.map_canvas.cleanup()
        except Exception as e:
            print(f"Geology Cleanup Fehler: {e}")
        finally:
            super().closeEvent(event)


# Test-Funktion
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    print("🧪 Teste Geology Tab...")
    app = QApplication(sys.argv)

    try:
        window = GeologyWindow()
        window.show()
        print("✅ Geology Tab Test erfolgreich!")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"❌ Geology Tab Test fehlgeschlagen: {e}")
        import traceback

        traceback.print_exc()