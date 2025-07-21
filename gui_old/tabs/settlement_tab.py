#!/usr/bin/env python3
"""
Settlement Tab - 3D Terrain mit Settlement Markers
ERSETZT bestehende settlement_tab.py komplett
"""

import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QSpacerItem, QSizePolicy,
    QCheckBox, QGroupBox
)
from PyQt5.QtCore import Qt

from gui_old.widgets.map_canvas import SettlementDualCanvas
from gui_old.widgets.parameter_slider import ParameterSlider
from gui_old.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui_old.utils.performance_utils import debounced_method, performance_tracked
from gui_old.managers.parameter_manager import WorldParameterManager


class SettlementControlPanel(QWidget, NavigationMixin):
    """Settlement Control Panel mit 3D Terrain Integration"""

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas
        self.world_manager = WorldParameterManager()
        self.settlement_manager = self.world_manager.settlement

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Settlements & 3D Terrain")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
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
        self.setup_count_parameters(layout)
        self.setup_property_parameters(layout)
        self.setup_placement_parameters(layout)

        # Settlement Preview
        self.setup_settlement_preview(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Navigation
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="← Geology", next_text="Weather →")

        self.setLayout(layout)

    def setup_simulation_controls(self, layout):
        """Auto-Simulation Controls"""
        sim_control_layout = QHBoxLayout()

        self.auto_simulate_checkbox = QCheckBox("Auto-Simulation")
        self.auto_simulate_checkbox.setChecked(
            WorldParameterManager().ui_state.get_auto_simulate()
        )
        self.auto_simulate_checkbox.stateChanged.connect(self.on_auto_simulate_changed)

        self.simulate_now_btn = QPushButton("Jetzt Simulieren")
        self.simulate_now_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }")
        self.simulate_now_btn.clicked.connect(self.update_preview())
        self.simulate_now_btn.setEnabled(not self.world_manager.ui_state.get_auto_simulate())

        sim_control_widget = QWidget()
        sim_control_layout.addWidget(self.auto_simulate_checkbox)
        sim_control_layout.addWidget(self.simulate_now_btn)
        sim_control_widget.setLayout(sim_control_layout)
        layout.addWidget(sim_control_widget)

    def setup_count_parameters(self, layout):
        """Anzahl-Parameter"""
        count_group = QGroupBox("Anzahl Settlements")
        count_layout = QVBoxLayout()

        params = self.settlement_manager.get_parameters()

        self.villages_slider = ParameterSlider("Dörfer", 1, 5, params['villages'])
        self.landmarks_slider = ParameterSlider("Landmarks", 1, 5, params['landmarks'])
        self.pubs_slider = ParameterSlider("Pubs", 1, 5, params['pubs'])
        self.connections_slider = ParameterSlider("Außenverbindungen", 1, 5, params['connections'])

        count_sliders = [self.villages_slider, self.landmarks_slider,
                         self.pubs_slider, self.connections_slider]

        for slider in count_sliders:
            count_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        count_group.setLayout(count_layout)
        layout.addWidget(count_group)

    def setup_property_parameters(self, layout):
        """Eigenschaften-Parameter"""
        props_group = QGroupBox("Settlement Eigenschaften")
        props_layout = QVBoxLayout()

        params = self.settlement_manager.get_parameters()

        self.village_size_slider = ParameterSlider("Dorf-Größe", 5, 30, params['village_size'])
        self.village_influence_slider = ParameterSlider("Dorf-Einfluss", 10, 50, params['village_influence'])
        self.landmark_influence_slider = ParameterSlider("Landmark-Einfluss", 5, 40, params['landmark_influence'])

        props_sliders = [self.village_size_slider, self.village_influence_slider,
                         self.landmark_influence_slider]

        for slider in props_sliders:
            props_layout.addWidget(slider)
            slider.valueChanged.connect(self.on_parameter_changed)

        props_group.setLayout(props_layout)
        layout.addWidget(props_group)

    def setup_placement_parameters(self, layout):
        """Placement-Parameter (neue für Terrain-Integration)"""
        placement_group = QGroupBox("Terrain-basierte Platzierung")
        placement_layout = QVBoxLayout()

        self.prefer_flat_cb = QCheckBox("Bevorzuge flache Gebiete")
        self.prefer_flat_cb.setChecked(True)
        self.prefer_flat_cb.stateChanged.connect(self.on_parameter_changed)

        self.avoid_water_cb = QCheckBox("Vermeide Wassernähe")
        self.avoid_water_cb.setChecked(False)
        self.avoid_water_cb.stateChanged.connect(self.on_parameter_changed)

        self.height_preference_slider = ParameterSlider("Höhen-Präferenz", 0.0, 1.0, 0.3, decimals=2)
        self.height_preference_slider.valueChanged.connect(self.on_parameter_changed)

        placement_layout.addWidget(self.prefer_flat_cb)
        placement_layout.addWidget(self.avoid_water_cb)
        placement_layout.addWidget(self.height_preference_slider)

        placement_group.setLayout(placement_layout)
        layout.addWidget(placement_group)

    def setup_settlement_preview(self, layout):
        """Settlement Preview Info"""
        preview_group = QGroupBox("Settlement Vorschau")
        preview_layout = QVBoxLayout()

        self.preview_label = QLabel("Generiere Settlements...")
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        self.preview_label.setAlignment(Qt.AlignLeft)
        self.preview_label.setWordWrap(True)

        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

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
            self.update_preview()

    @performance_tracked("Settlement_Preview_Update")
    def update_preview(self):
        """Aktualisiert Settlement Preview"""
        params = self.get_parameters()

        # Parameter validieren und speichern
        self.settlement_manager.set_parameters(params)

        # Prüfe Settlement-Dichte
        is_valid, warning = self.settlement_manager.validate_settlement_density()

        # Input Status prüfen
        if not self._has_terrain_input():
            self.input_status_label.setText("⚠ Keine Terrain-Daten verfügbar")
            self.input_status_label.setStyleSheet("""
                QLabel {
                    background-color: #f8d7da;
                    border: 2px solid #dc3545;
                    border-radius: 8px;
                    padding: 8px;
                    font-weight: bold;
                }
            """)
            return

        # Update Input Status
        self.input_status_label.setText("✓ Terrain-Daten verfügbar - 3D Placement aktiv")
        self.input_status_label.setStyleSheet("""
            QLabel {
                background-color: #d4edda;
                border: 2px solid #28a745;
                border-radius: 8px;
                padding: 8px;
                font-weight: bold;
            }
        """)

        # Update Preview Info
        self.update_settlement_preview(params)

        # Canvas Update
        self.map_canvas.update_map(**params)

    def _has_terrain_input(self):
        """Prüft ob Terrain-Input verfügbar ist"""
        return hasattr(self.map_canvas, 'input_data') and 'heightmap' in self.map_canvas.input_data

    def update_settlement_preview(self, params):
        """Aktualisiert Settlement Preview Text"""
        try:
            total_settlements = self.settlement_manager.get_total_settlements()

            preview_text = "SETTLEMENT PREVIEW:\n\n"
            preview_text += f"Dörfer: {params['villages']}\n"
            preview_text += f"Landmarks: {params['landmarks']}\n"
            preview_text += f"Pubs: {params['pubs']}\n"
            preview_text += f"Verbindungen: {params['connections']}\n"
            preview_text += f"Total: {total_settlements} Settlements\n\n"

            # Placement Info
            if params.get('prefer_flat', True):
                preview_text += "• Bevorzugt flache Gebiete\n"
            if params.get('avoid_water', False):
                preview_text += "• Vermeidet Wassernähe\n"

            preview_text += f"• Höhen-Präferenz: {params.get('height_preference', 0.3):.1f}"

            self.preview_label.setText(preview_text)

        except Exception as e:
            self.preview_label.setText(f"Preview Fehler: {e}")

    def get_parameters(self):
        """Sammelt alle Settlement Parameter"""
        return {
            'villages': self.villages_slider.get_value(),
            'landmarks': self.landmarks_slider.get_value(),
            'pubs': self.pubs_slider.get_value(),
            'connections': self.connections_slider.get_value(),
            'village_size': self.village_size_slider.get_value(),
            'village_influence': self.village_influence_slider.get_value(),
            'landmark_influence': self.landmark_influence_slider.get_value(),
            # Neue Terrain-Integration Parameter
            'prefer_flat': self.prefer_flat_cb.isChecked(),
            'avoid_water': self.avoid_water_cb.isChecked(),
            'height_preference': self.height_preference_slider.get_value()
            }

    # Navigation Methods
    def next_menu(self):
        """Wechselt zu Weather Tab"""
        params = self.get_parameters()
        self.settlement_manager.set_parameters(params)

        next_tab = TabNavigationHelper.get_next_tab('SettlementWindow')
        if next_tab:
            self.navigate_to_tab(next_tab[0], next_tab[1])

    def prev_menu(self):
        """Zurück zu Geology Tab"""
        params = self.get_parameters()
        self.settlement_manager.set_parameters(params)

        prev_tab = TabNavigationHelper.get_prev_tab('SettlementWindow')
        if prev_tab:
            self.navigate_to_tab(prev_tab[0], prev_tab[1])

    def quick_generate(self):
        """Schnellgenerierung"""
        params = self.get_parameters()
        self.settlement_manager.set_parameters(params)

class SettlementWindow(QMainWindow):
    """Settlement Tab Hauptfenster"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Settlements & 3D Terrain")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Linke Seite: 3D Terrain mit Settlements (70%)
        self.map_canvas = SettlementDualCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite: Controls (30%)
        self.control_panel = SettlementControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(350)
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Settlement-spezifisches Styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox[title*="Settlement"] {
                border-color: #8b4513;
                background-color: rgba(139, 69, 19, 0.05);
            }
        """)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        if hasattr(self, 'map_canvas'):
            self.map_canvas.cleanup()
        super().closeEvent(event)