"""
Path: gui/tabs/geology_tab.py

Funktionsweise: Geologie-Editor mit erweiteter Core-Integration und 3D Textured Terrain
- Input: heightmap, slopemap von terrain_tab über data_manager
- Core-Integration: geology_generator.py mit allen neuen Parametern
- 3D-Rendering mit Gesteinstyp-Texturen (RGB rock_map Visualization)
- Parameter: Rock-Types Hardness, Tektonische Deformation (ridge_warping, etc.)
- Output: rock_map (RGB), hardness_map für water_generator
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import GEOLOGY, get_parameter_config, validate_parameter_set, VALIDATION_RULES
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton
from core.geology_generator import GeologyGenerator, RockTypeClassifier, MassConservationManager

def get_geology_error_decorators():
    """
    Funktionsweise: Lazy Loading von Geology Tab Error Decorators
    Aufgabe: Lädt Memory-Critical, Parameter und GPU-Shader Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import memory_critical_handler, parameter_handler, data_management_handler
        return memory_critical_handler, parameter_handler, data_management_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator, noop_decorator

memory_critical_handler, parameter_handler, data_management_handler = get_geology_error_decorators()

class GeologyTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für Geology-Generation mit allen neuen Core-Features
    Aufgabe: Koordiniert UI, Parameter-Management und Core-Generator Integration
    Input: heightmap, slopemap von terrain_tab
    Output: rock_map (RGB), hardness_map für nachfolgende Generatoren
    """

    def __init__(self, data_manager, navigation_manager, shader_manager):
        super().__init__(data_manager, navigation_manager, shader_manager)
        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.geology_generator = GeologyGenerator()
        self.rock_classifier = RockTypeClassifier()
        self.mass_conservation = MassConservationManager()

        # Parameter-Tracking
        self.current_parameters = {}

        # Setup UI
        self.setup_geology_ui()
        self.setup_dependency_checking()

        # Initial Parameter Load
        self.load_default_parameters()

        # Input Status überwachen
        self.check_input_dependencies()

    def setup_geology_ui(self):
        """
        Funktionsweise: Erstellt spezialisierte UI für Geology-Generator
        Aufgabe: Parameter-Slider, Rock-Map Preview, Hardness-Visualization
        """
        # Parameter Panel
        self.parameter_panel = self.create_geology_parameter_panel()
        self.control_panel.addWidget(self.parameter_panel)

        # Rock Distribution Widget
        self.rock_distribution_widget = RockDistributionWidget()
        self.control_panel.addWidget(self.rock_distribution_widget)

        # Visualization Controls
        self.visualization_controls = self.create_geology_visualization_controls()
        self.control_panel.addWidget(self.visualization_controls)

        # Input Dependencies Status
        self.setup_input_status()
        self.setup_navigation()

    def create_geology_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Geology-Parametern
        Aufgabe: Slider für alle neuen Core-Parameter (ridge_warping, metamorph_*, etc.)
        Return: QGroupBox mit allen Parameter-Slidern
        """
        panel = QGroupBox("Geology Parameters")
        layout = QVBoxLayout()

        self.parameter_sliders = {}

        # Rock Hardness Parameters
        hardness_group = QGroupBox("Rock Hardness")
        hardness_layout = QVBoxLayout()

        hardness_params = ["sedimentary_hardness", "igneous_hardness", "metamorphic_hardness"]
        for param_name in hardness_params:
            param_config = get_parameter_config("geology", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 1),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            hardness_layout.addWidget(slider)

        hardness_group.setLayout(hardness_layout)
        layout.addWidget(hardness_group)

        # Deformation Parameters
        deformation_group = QGroupBox("Tectonic Deformation")
        deformation_layout = QVBoxLayout()

        deformation_params = [
            "ridge_warping", "bevel_warping", "metamorph_foliation",
            "metamorph_folding", "igneous_flowing"
        ]

        for param_name in deformation_params:
            param_config = get_parameter_config("geology", param_name)

            slider = ParameterSlider(
                label=param_name.replace("_", " ").title(),
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 0.1),
                suffix=param_config.get("suffix", "")
            )

            slider.valueChanged.connect(self.on_parameter_changed)
            self.parameter_sliders[param_name] = slider
            deformation_layout.addWidget(slider)

        deformation_group.setLayout(deformation_layout)
        layout.addWidget(deformation_group)

        panel.setLayout(layout)
        return panel

    def create_geology_visualization_controls(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Controls für Geology-Visualization
        Aufgabe: Switcher zwischen rock_map (RGB), hardness_map, einzelne Gesteinstypen
        Return: QGroupBox mit Visualization-Controls
        """
        panel = QGroupBox("Geology Visualization")
        layout = QVBoxLayout()

        # Display Mode Selection
        self.display_mode = QButtonGroup()

        self.rock_map_radio = QRadioButton("Rock Map (RGB)")
        self.rock_map_radio.setChecked(True)
        self.rock_map_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.rock_map_radio, 0)
        layout.addWidget(self.rock_map_radio)

        self.hardness_radio = QRadioButton("Hardness Map")
        self.hardness_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.hardness_radio, 1)
        layout.addWidget(self.hardness_radio)

        self.sedimentary_radio = QRadioButton("Sedimentary (Red)")
        self.sedimentary_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.sedimentary_radio, 2)
        layout.addWidget(self.sedimentary_radio)

        self.igneous_radio = QRadioButton("Igneous (Green)")
        self.igneous_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.igneous_radio, 3)
        layout.addWidget(self.igneous_radio)

        self.metamorphic_radio = QRadioButton("Metamorphic (Blue)")
        self.metamorphic_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.metamorphic_radio, 4)
        layout.addWidget(self.metamorphic_radio)

        # 3D Terrain Toggle
        self.terrain_3d_checkbox = QCheckBox("Show 3D Terrain")
        self.terrain_3d_checkbox.toggled.connect(self.toggle_3d_terrain)
        layout.addWidget(self.terrain_3d_checkbox)

        panel.setLayout(layout)
        return panel

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking
        Aufgabe: Überwacht verfügbare Inputs von terrain_tab
        """
        # Required Dependencies für Geology
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["geology"]

        # Dependency Status Widget
        self.dependency_status = StatusIndicator("Input Dependencies")
        self.control_panel.addWidget(self.dependency_status)

        # Data Manager Signals verbinden
        self.data_manager.data_updated.connect(self.on_data_updated)

    def load_default_parameters(self):
        """
        Funktionsweise: Lädt Default-Parameter in alle Slider
        """
        for param_name, slider in self.parameter_sliders.items():
            param_config = get_parameter_config("geology", param_name)
            slider.setValue(param_config["default"])

        self.current_parameters = self.get_current_parameters()

    def get_current_parameters(self) -> dict:
        """
        Funktionsweise: Sammelt aktuelle Parameter-Werte von allen Slidern
        Return: dict mit allen aktuellen Parameter-Werten für Core-Generator
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    @parameter_handler
    def on_parameter_changed(self):
        """
        Funktionsweise: Slot für Parameter-Änderungen
        Aufgabe: Update Rock-Distribution Widget, Auto-Generation triggern
        """
        self.current_parameters = self.get_current_parameters()

        # Rock Distribution Widget aktualisieren
        self.rock_distribution_widget.update_distribution(self.current_parameters)

        # Auto-Simulation triggern
        if self.auto_simulation_enabled:
            self.auto_simulation_timer.start(1000)

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """
        Funktionsweise: Slot für Data-Updates von anderen Generatoren
        Aufgabe: Prüft Dependencies und aktiviert/deaktiviert Generation
        """
        if generator_type == "terrain" and data_key in self.required_dependencies:
            self.check_input_dependencies()

    def check_input_dependencies(self):
        """
        Funktionsweise: Prüft ob alle Required Dependencies verfügbar sind
        Aufgabe: Aktiviert/Deaktiviert Generation Button und zeigt Status
        """
        is_complete, missing = self.data_manager.check_dependencies("geology", self.required_dependencies)

        if is_complete:
            self.dependency_status.set_success("All inputs available")
            self.manual_generate_button.setEnabled(True)
        else:
            self.dependency_status.set_warning(f"Missing: {', '.join(missing)}")
            self.manual_generate_button.setEnabled(False)

        return is_complete

    @memory_critical_handler("geology_generation")
    def generate_geology(self):
        """
        Funktionsweise: Hauptmethode für Geology-Generation mit Core-Integration
        Aufgabe: Ruft alle Core-Generatoren auf und speichert Results im DataManager
        """
        try:
            # Dependencies prüfen
            if not self.check_input_dependencies():
                self.logger.warning("Cannot generate geology - missing dependencies")
                return

            self.logger.info("Starting geology generation...")

            self.start_generation_timing()

            # Input-Daten von DataManager holen
            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")

            if heightmap is None or slopemap is None:
                raise ValueError("Required terrain data not available")

            # Parameter für Core-Generator vorbereiten
            params = self.current_parameters.copy()

            # Rock-Type Hardness Values für Core-Generator
            rock_types = {
                "sedimentary": params["sedimentary_hardness"],
                "igneous": params["igneous_hardness"],
                "metamorphic": params["metamorphic_hardness"]
            }

            # 1. Rock Distribution Generation
            rock_map = self.geology_generator.generate_rock_distribution(
                heightmap=heightmap,
                slopemap=slopemap,
                rock_types=rock_types,
                ridge_warping=params["ridge_warping"],
                bevel_warping=params["bevel_warping"],
                metamorph_foliation=params["metamorph_foliation"],
                metamorph_folding=params["metamorph_folding"],
                igneous_flowing=params["igneous_flowing"]
            )

            # 2. Hardness Map Calculation
            hardness_map = self.geology_generator.calculate_hardness_map(
                rock_map=rock_map,
                hardness_values=rock_types
            )

            # 3. Mass Conservation Validation
            rock_map_normalized = self.mass_conservation.normalize_rock_masses(rock_map)
            mass_valid = self.mass_conservation.validate_conservation(rock_map_normalized)

            if not mass_valid:
                self.logger.warning("Mass conservation validation failed - applying correction")
                rock_map_normalized = self.mass_conservation.redistribute_masses(rock_map_normalized)

            # Results im DataManager speichern
            self.data_manager.set_geology_data("rock_map", rock_map_normalized, params)
            self.data_manager.set_geology_data("hardness_map", hardness_map, params)

            # Display aktualisieren
            self.update_geology_display()

            # Rock Distribution Statistics aktualisieren
            self.rock_distribution_widget.update_statistics(rock_map_normalized, hardness_map)

            # Timing beenden
            self.end_generation_timing(True)

            self.logger.info("Geology generation completed successfully")


        except Exception as e:
            self.handle_generation_error(e)
            self.end_generation_timing(False, str(e))
            raise

    @data_management_handler("geology_display")
    def update_geology_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Zeigt rock_map, hardness_map oder einzelne Gesteinstypen
        """
        current_mode = self.display_mode.checkedId()

        rock_map = self.data_manager.get_geology_data("rock_map")
        hardness_map = self.data_manager.get_geology_data("hardness_map")

        if rock_map is None:
            return

        if current_mode == 0:  # Rock Map (RGB)
            self.map_display.display_rock_map_rgb(rock_map)

        elif current_mode == 1:  # Hardness Map
            if hardness_map is not None:
                self.map_display.display_hardness_map(hardness_map)

        elif current_mode == 2:  # Sedimentary (Red Channel)
            sedimentary_map = rock_map[:, :, 0] / 255.0
            self.map_display.display_single_rock_type(sedimentary_map, "Sedimentary", "Reds")

        elif current_mode == 3:  # Igneous (Green Channel)
            igneous_map = rock_map[:, :, 1] / 255.0
            self.map_display.display_single_rock_type(igneous_map, "Igneous", "Greens")

        elif current_mode == 4:  # Metamorphic (Blue Channel)
            metamorphic_map = rock_map[:, :, 2] / 255.0
            self.map_display.display_single_rock_type(metamorphic_map, "Metamorphic", "Blues")

        # 3D Terrain Overlay wenn aktiviert
        if self.terrain_3d_checkbox.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                self.map_display.overlay_3d_terrain(heightmap)

    @pyqtSlot()
    def update_display_mode(self):
        """
        Funktionsweise: Slot für Visualization-Mode Änderungen
        Aufgabe: Aktualisiert Display basierend auf neuer Selection
        """
        self.update_geology_display()

    @pyqtSlot(bool)
    def toggle_3d_terrain(self, enabled: bool):
        """
        Funktionsweise: Toggle für 3D Terrain Overlay
        Aufgabe: Aktiviert/Deaktiviert 3D Heightmap Overlay über Geology-Data
        """
        self.update_geology_display()

    # Override BaseMapTab method
    def generate_terrain(self):
        """Override von BaseMapTab für Geology-spezifische Generation"""
        self.generate_geology()


class RockDistributionWidget(QGroupBox):
    """
    Funktionsweise: Widget für Rock-Distribution Visualization und Statistics
    Aufgabe: Zeigt aktuelle Gesteins-Verteilung, Mass-Conservation Status
    """

    def __init__(self):
        super().__init__("Rock Distribution")
        self.setup_ui()

    def setup_ui(self):
        """
        Funktionsweise: Erstellt UI für Rock-Distribution Display
        """
        layout = QVBoxLayout()

        # Hardness Preview
        hardness_group = QGroupBox("Rock Hardness Preview")
        hardness_layout = QGridLayout()

        self.sedimentary_bar = QProgressBar()
        self.sedimentary_bar.setStyleSheet("QProgressBar::chunk { background-color: #d2691e; }")
        self.sedimentary_label = QLabel("Sedimentary: 30")
        hardness_layout.addWidget(self.sedimentary_label, 0, 0)
        hardness_layout.addWidget(self.sedimentary_bar, 0, 1)

        self.igneous_bar = QProgressBar()
        self.igneous_bar.setStyleSheet("QProgressBar::chunk { background-color: #228b22; }")
        self.igneous_label = QLabel("Igneous: 80")
        hardness_layout.addWidget(self.igneous_label, 1, 0)
        hardness_layout.addWidget(self.igneous_bar, 1, 1)

        self.metamorphic_bar = QProgressBar()
        self.metamorphic_bar.setStyleSheet("QProgressBar::chunk { background-color: #4169e1; }")
        self.metamorphic_label = QLabel("Metamorphic: 65")
        hardness_layout.addWidget(self.metamorphic_label, 2, 0)
        hardness_layout.addWidget(self.metamorphic_bar, 2, 1)

        hardness_group.setLayout(hardness_layout)
        layout.addWidget(hardness_group)

        # Distribution Statistics
        self.distribution_stats = QLabel("Distribution: Not generated")
        layout.addWidget(self.distribution_stats)

        # Mass Conservation Status
        self.mass_conservation_status = StatusIndicator("Mass Conservation")
        layout.addWidget(self.mass_conservation_status)

        self.setLayout(layout)

    def update_distribution(self, parameters: dict):
        """
        Funktionsweise: Aktualisiert Rock-Hardness Preview basierend auf Parametern
        Parameter: parameters (dict mit hardness values)
        """
        # Progress Bars aktualisieren (0-100 Scale)
        sed_hardness = parameters.get("sedimentary_hardness", 30)
        ign_hardness = parameters.get("igneous_hardness", 80)
        met_hardness = parameters.get("metamorphic_hardness", 65)

        self.sedimentary_bar.setValue(sed_hardness)
        self.sedimentary_label.setText(f"Sedimentary: {sed_hardness}")

        self.igneous_bar.setValue(ign_hardness)
        self.igneous_label.setText(f"Igneous: {ign_hardness}")

        self.metamorphic_bar.setValue(met_hardness)
        self.metamorphic_label.setText(f"Metamorphic: {met_hardness}")

    def update_statistics(self, rock_map: np.ndarray, hardness_map: np.ndarray):
        """
        Funktionsweise: Aktualisiert Statistiken nach Generation
        Parameter: rock_map (RGB array), hardness_map (2D array)
        """
        # Rock Distribution Percentages berechnen
        total_pixels = rock_map.shape[0] * rock_map.shape[1]

        sedimentary_pct = np.sum(rock_map[:, :, 0]) / (total_pixels * 255) * 100
        igneous_pct = np.sum(rock_map[:, :, 1]) / (total_pixels * 255) * 100
        metamorphic_pct = np.sum(rock_map[:, :, 2]) / (total_pixels * 255) * 100

        # Mass Conservation Check
        mass_sums = np.sum(rock_map, axis=2)
        mass_conservation_valid = np.allclose(mass_sums, 255)

        # Update Statistics Display
        self.distribution_stats.setText(
            f"Distribution: Sed {sedimentary_pct:.1f}%, Ign {igneous_pct:.1f}%, Met {metamorphic_pct:.1f}%"
        )

        if mass_conservation_valid:
            self.mass_conservation_status.set_success("Mass conserved (R+G+B=255)")
        else:
            self.mass_conservation_status.set_warning("Mass conservation violation detected")