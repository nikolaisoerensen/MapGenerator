"""
Path: gui/tabs/terrain_tab.py

Funktionsweise: Terrain-Editor mit erweiteter Core-Integration
- Erbt von BaseMapTab für gemeinsame Features
- Direkte Integration mit core/terrain_generator.py (alle neuen Parameter)
- Spezialisierte Widgets: TerrainParameterPanel, TerrainStatisticsWidget
- Live 2D/3D Preview über map_display_2d/3d.py erweitert
- Real-time Terrain-Statistics (Höhenverteilung, Steigungen, Verschattung)
- Output: heightmap, slopemap, shademap für nachfolgende Generatoren
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import TERRAIN, get_parameter_config, validate_parameter_set
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton
from core.terrain_generator import BaseTerrainGenerator, SimplexNoiseGenerator, ShadowCalculator

def get_terrain_error_decorators():
    """
    Funktionsweise: Lazy Loading von Terrain Tab Error Decorators
    Aufgabe: Lädt Memory-Critical, Parameter und GPU-Shader Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import memory_critical_handler, parameter_handler, gpu_shader_handler
        return memory_critical_handler, parameter_handler, gpu_shader_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

memory_critical_handler, parameter_handler, gpu_shader_handler = get_terrain_error_decorators()

class TerrainTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für Terrain-Generation mit allen neuen Core-Features
    Aufgabe: Koordiniert UI, Parameter-Management und Core-Generator Integration
    Output: heightmap, slopemap, shademap für alle nachfolgenden Generatoren
    """

    def __init__(self, data_manager, navigation_manager, shader_manager):
        super().__init__(data_manager, navigation_manager, shader_manager)
        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.terrain_generator = BaseTerrainGenerator()
        self.noise_generator = SimplexNoiseGenerator()
        self.shadow_calculator = ShadowCalculator()

        # Parameter-Tracking für Cache-Validation
        self.current_parameters = {}

        # Setup UI
        self.setup_terrain_ui()
        self.setup_parameter_validation()

        # Initial Parameter Load
        self.load_default_parameters()

    def setup_terrain_ui(self):
        """
        Funktionsweise: Erstellt spezialisierte UI für Terrain-Generator
        Aufgabe: Parameter-Slider, Preview-Canvas, Statistics-Widget
        """
        # Parameter Panel erstellen
        self.parameter_panel = self.create_terrain_parameter_panel()
        self.control_panel.addWidget(self.parameter_panel)

        # Statistics Widget
        self.statistics_widget = TerrainStatisticsWidget()
        self.control_panel.addWidget(self.statistics_widget)

        # Visualization Controls
        self.visualization_controls = self.create_visualization_controls()
        self.control_panel.addWidget(self.visualization_controls)

        # Status und Navigation (von BaseMapTab)
        self.setup_input_status()
        self.setup_navigation()

    def create_terrain_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Terrain-Parametern
        Aufgabe: Slider für alle neuen Core-Parameter (amplitude, map_seed, etc.)
        Return: QGroupBox mit allen Parameter-Slidern
        """
        panel = QGroupBox("Terrain Parameters")
        layout = QVBoxLayout()

        # Dictionary für alle Parameter-Slider
        self.parameter_sliders = {}

        # Alle Terrain-Parameter aus value_default.py
        terrain_params = [
            "size", "amplitude", "octaves", "frequency",
            "persistence", "lacunarity", "redistribute_power", "map_seed"
        ]

        for param_name in terrain_params:
            param_config = get_parameter_config("terrain", param_name)

            # Parameter-Slider mit Konfiguration erstellen
            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 1),
                suffix=param_config.get("suffix", "")
            )

            # Signal-Verbindung für Auto-Update
            slider.valueChanged.connect(self.on_parameter_changed)

            self.parameter_sliders[param_name] = slider
            layout.addWidget(slider)

        panel.setLayout(layout)
        return panel

    def create_visualization_controls(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Controls für Visualization-Modi
        Aufgabe: Switcher zwischen heightmap, slopemap, shademap Darstellung
        Return: QGroupBox mit Visualization-Controls
        """
        panel = QGroupBox("Visualization")
        layout = QVBoxLayout()

        # Radio Buttons für verschiedene Darstellungsarten
        self.display_mode = QButtonGroup()

        self.heightmap_radio = QRadioButton("Heightmap")
        self.heightmap_radio.setChecked(True)
        self.heightmap_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.heightmap_radio, 0)
        layout.addWidget(self.heightmap_radio)

        self.slopemap_radio = QRadioButton("Slope Map")
        self.slopemap_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.slopemap_radio, 1)
        layout.addWidget(self.slopemap_radio)

        self.shademap_radio = QRadioButton("Shadow Map")
        self.shademap_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.shademap_radio, 2)
        layout.addWidget(self.shademap_radio)

        # Shadow Angle Slider (für Shademap)
        self.shadow_angle_slider = ParameterSlider(
            label="Shadow Angle",
            min_val=0,
            max_val=5,
            default_val=2,
            step=1,
            suffix=""
        )
        self.shadow_angle_slider.valueChanged.connect(self.update_shadow_display)
        self.shadow_angle_slider.setEnabled(False)  # Initial disabled
        layout.addWidget(self.shadow_angle_slider)

        panel.setLayout(layout)
        return panel

    def setup_parameter_validation(self):
        """
        Funktionsweise: Setup für Parameter-Validation und Cross-Parameter Constraints
        Aufgabe: Verbindet Validation-System mit Parameter-Änderungen
        """
        # Timer für Debounced Validation
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self.validate_current_parameters)

        # Validation Status Widget
        self.validation_status = StatusIndicator("Parameter Validation")
        self.control_panel.addWidget(self.validation_status)

    def load_default_parameters(self):
        """
        Funktionsweise: Lädt Default-Parameter in alle Slider
        Aufgabe: Initialisiert UI mit Standard-Werten aus value_default.py
        """
        for param_name, slider in self.parameter_sliders.items():
            param_config = get_parameter_config("terrain", param_name)
            slider.setValue(param_config["default"])

        # Initial Parameter-Set speichern
        self.current_parameters = self.get_current_parameters()

    def get_current_parameters(self) -> dict:
        """
        Funktionsweise: Sammelt aktuelle Parameter-Werte von allen Slidern
        Return: dict mit allen aktuellen Parameter-Werten
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    @pyqtSlot()
    def on_parameter_changed(self):
        """
        Funktionsweise: Slot für Parameter-Änderungen mit Debouncing
        Aufgabe: Triggert Validation und Auto-Generation nach kurzer Verzögerung
        """
        # Parameter aktualisieren
        self.current_parameters = self.get_current_parameters()

        # Debounced Validation starten
        self.validation_timer.start(500)  # 500ms Delay

        # Auto-Simulation triggern (wenn aktiviert)
        if self.auto_simulation_enabled:
            self.auto_simulation_timer.start(1000)  # 1s Delay für Generation

    @parameter_handler
    def validate_current_parameters(self):
        """
        Funktionsweise: Validiert aktuelle Parameter auf Constraints und Performance
        Aufgabe: Prüft Cross-Parameter Validation und zeigt Warnings/Errors
        Besonderheit: Parameter Handler schützt vor Validation-Fehlern
        """
        # Import hier um Circular Dependencies zu vermeiden
        try:
            from gui.config.value_default import validate_parameter_set
            is_valid, warnings, errors = validate_parameter_set("terrain", self.current_parameters)
        except ImportError:
            # Fallback bei fehlendem Config-Modul
            is_valid, warnings, errors = True, [], []

        if errors:
            self.validation_status.set_error(f"Errors: {'; '.join(errors)}")
        elif warnings:
            self.validation_status.set_warning(f"Warnings: {'; '.join(warnings)}")
        else:
            self.validation_status.set_success("Parameters valid")

    @memory_critical_handler("terrain_generation")
    def generate_terrain(self):
        """
        Funktionsweise: Hauptmethode für Terrain-Generation mit Core-Integration
        Aufgabe: Ruft alle Core-Generatoren auf und speichert Results im DataManager
        Besonderheit: Memory-Critical Handler schützt vor Memory-Errors bei großen Arrays
        """
        try:
            self.logger.info("Starting terrain generation...")

            # Timing für Performance-Messung starten
            self.start_generation_timing()

            # Parameter für Core-Generator vorbereiten
            params = self.current_parameters.copy()
            map_size = int(params.get("size", 512))  # Fallback für size

            # LOD-Level vom DataManager übernehmen
            current_map_size = self.data_manager.get_current_map_size()
            if current_map_size != map_size:
                params["size"] = current_map_size
                self.logger.info(f"Using LOD map size: {current_map_size}")

            # 1. Heightmap Generation
            heightmap = self.terrain_generator.generate_heightmap(
                map_size=params["size"],
                amplitude=params.get("amplitude", 100.0),
                octaves=int(params.get("octaves", 6)),
                frequency=params.get("frequency", 0.01),
                persistence=params.get("persistence", 0.5),
                lacunarity=params.get("lacunarity", 2.0),
                redistribute_power=params.get("redistribute_power", 1.0),
                map_seed=int(params.get("map_seed", 42))
            )

            # 2. Slope Calculation
            slopemap = self.terrain_generator.calculate_slopes(heightmap)

            # 3. Shadow Calculation (6 Sonnenwinkel)
            shademap = self.shadow_calculator.calculate_shadows_multi_angle(
                heightmap=heightmap,
                sun_angles=6
            )

            # Results im DataManager speichern
            self.data_manager.set_terrain_data("heightmap", heightmap, params)
            self.data_manager.set_terrain_data("slopemap", slopemap, params)
            self.data_manager.set_terrain_data("shademap", shademap, params)

            # Display aktualisieren
            self.update_terrain_display()

            # Statistics aktualisieren
            self.statistics_widget.update_statistics(heightmap, slopemap, shademap)

            # Timing beenden
            self.end_generation_timing(True)

            self.logger.info("Terrain generation completed successfully")

        except Exception as e:
            self.handle_generation_error(e)
            self.end_generation_timing(False, str(e))
            raise  # Re-raise für Error Handler

    @gpu_shader_handler("terrain_display")
    def update_terrain_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt heightmap, slopemap oder shademap je nach Selection
        Besonderheit: GPU-Shader Handler schützt vor OpenGL/Display-Fehlern
        """
        current_mode = self.display_mode.checkedId()

        if current_mode == 0:  # Heightmap
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.display_heightmap(heightmap)

        elif current_mode == 1:  # Slopemap
            slopemap = self.data_manager.get_terrain_data("slopemap")
            if slopemap is not None:
                self.map_display.display_slopemap(slopemap)

        elif current_mode == 2:  # Shademap
            shademap = self.data_manager.get_terrain_data("shademap")
            if shademap is not None:
                angle_index = int(self.shadow_angle_slider.getValue())
                self.map_display.display_shademap(shademap, angle_index)

    @pyqtSlot()
    def update_display_mode(self):
        """
        Funktionsweise: Slot für Visualization-Mode Änderungen
        Aufgabe: Aktiviert/Deaktiviert Shadow-Angle Slider, aktualisiert Display
        """
        current_mode = self.display_mode.checkedId()

        # Shadow Angle Slider nur bei Shademap aktivieren
        self.shadow_angle_slider.setEnabled(current_mode == 2)

        # Display aktualisieren
        self.update_terrain_display()

    @pyqtSlot()
    def update_shadow_display(self):
        """
        Funktionsweise: Aktualisiert Shademap-Display bei Angle-Änderung
        Aufgabe: Zeigt spezifischen Sonnenwinkel der 6-Winkel Shademap
        """
        if self.display_mode.checkedId() == 2:  # Nur bei Shademap-Mode
            self.update_terrain_display()


class TerrainStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Widget für Real-time Terrain-Statistiken
    Aufgabe: Zeigt Höhenverteilung, Steigungsstatistiken, Verschattungsinfo
    """

    def __init__(self):
        super().__init__("Terrain Statistics")
        self.setup_ui()

    def setup_ui(self):
        """
        Funktionsweise: Erstellt UI für Statistik-Anzeige
        """
        layout = QVBoxLayout()

        # Statistics Labels
        self.height_stats = QLabel("Height: -")
        self.slope_stats = QLabel("Slope: -")
        self.shadow_stats = QLabel("Shadow: -")

        layout.addWidget(self.height_stats)
        layout.addWidget(self.slope_stats)
        layout.addWidget(self.shadow_stats)

        self.setLayout(layout)

    def update_statistics(self, heightmap: np.ndarray, slopemap: np.ndarray, shademap: np.ndarray):
        """
        Funktionsweise: Berechnet und zeigt aktuelle Terrain-Statistiken
        Parameter: heightmap, slopemap, shademap (numpy arrays)
        """
        # Height Statistics
        height_min = np.min(heightmap)
        height_max = np.max(heightmap)
        height_mean = np.mean(heightmap)

        # Slope Statistics (Convert to degrees)
        slope_degrees = np.arctan(np.sqrt(slopemap[:,:,0]**2 + slopemap[:,:,1]**2)) * 180 / np.pi
        slope_max = np.max(slope_degrees)
        slope_mean = np.mean(slope_degrees)

        # Shadow Statistics (Average über alle 6 Winkel)
        shadow_mean = np.mean(shademap)
        shadow_min = np.min(shademap)

        # Update Labels
        self.height_stats.setText(f"Height: {height_min:.1f}m - {height_max:.1f}m (μ={height_mean:.1f}m)")
        self.slope_stats.setText(f"Slope: 0° - {slope_max:.1f}° (μ={slope_mean:.1f}°)")
        self.shadow_stats.setText(f"Shadow: {shadow_min:.2f} - 1.00 (μ={shadow_mean:.2f})")


# Integration in BaseMapTab erweitern
class BaseMapTab(QWidget):
    """
    Basis-Klasse erweitert für alle neuen Features
    Muss hier definiert werden um Circular Import zu vermeiden
    """

    def __init__(self, data_manager, navigation_manager, shader_manager):
        super().__init__()

        self.data_manager = data_manager
        self.navigation_manager = navigation_manager
        self.shader_manager = shader_manager

        # Auto-Simulation Setup
        self.auto_simulation_enabled = False
        self.auto_simulation_timer = QTimer()
        self.auto_simulation_timer.setSingleShot(True)
        self.auto_simulation_timer.timeout.connect(self.generate_terrain)

        # UI Setup (70/30 Layout)
        self.setup_common_ui()
        self.setup_auto_simulation()

    def setup_common_ui(self):
        """70/30 Layout Setup"""
        layout = QHBoxLayout()

        # Canvas (70%)
        self.canvas_container = QFrame()
        self.canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Control Panel (30%)
        self.control_panel = QVBoxLayout()
        control_widget = QFrame()
        control_widget.setLayout(self.control_panel)
        control_widget.setMaximumWidth(400)

        # Splitter für resizable Layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas_container)
        splitter.addWidget(control_widget)
        splitter.setSizes([700, 300])  # 70/30 initial ratio

        layout.addWidget(splitter)
        self.setLayout(layout)

        # Map Display wird von Sub-Classes erstellt
        from gui.widgets.map_display_2d import MapDisplay2D
        self.map_display = MapDisplay2D()

        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.map_display)
        self.canvas_container.setLayout(canvas_layout)

    def setup_auto_simulation(self):
        """Auto-Simulation Controls Setup"""
        auto_panel = QGroupBox("Auto Simulation")
        layout = QVBoxLayout()

        self.auto_simulation_checkbox = QCheckBox("Auto Update")
        self.auto_simulation_checkbox.toggled.connect(self.toggle_auto_simulation)
        layout.addWidget(self.auto_simulation_checkbox)

        self.manual_generate_button = BaseButton("Generate Manually")
        self.manual_generate_button.clicked.connect(self.generate_terrain)
        layout.addWidget(self.manual_generate_button)

        auto_panel.setLayout(layout)
        self.control_panel.addWidget(auto_panel)

    @pyqtSlot(bool)
    def toggle_auto_simulation(self, enabled: bool):
        """Toggle Auto-Simulation on/off"""
        self.auto_simulation_enabled = enabled

    def generate_terrain(self):
        """Placeholder - wird von Sub-Classes überschrieben"""
        pass