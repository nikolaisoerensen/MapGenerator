"""
Path: gui/tabs/geology_tab.py

Funktionsweise: Geologie-Editor mit erweiteter Core-Integration und 3D Textured Terrain - VOLLSTÄNDIG REPARIERT
- Input: heightmap, slopemap von terrain_tab über data_manager
- Core-Integration: geology_generator.py mit allen neuen Parametern
- 3D-Rendering mit Gesteinstyp-Texturen (RGB rock_map Visualization)
- Parameter: Rock-Types Hardness, Tektonische Deformation (ridge_warping, etc.)
- Output: rock_map (RGB), hardness_map für water_generator
- FIXES: Orchestrator-Init, Button-API-Access, Dependency-Resolution, Thread-Safety
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import GEOLOGY, get_parameter_config, validate_parameter_set, VALIDATION_RULES
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton, DependencyResolver
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
    Funktionsweise: Hauptklasse für Geology-Generation mit allen neuen Core-Features - VOLLSTÄNDIG REPARIERT
    Aufgabe: Koordiniert UI, Parameter-Management und Core-Generator Integration
    Input: heightmap, slopemap von terrain_tab
    Output: rock_map (RGB), hardness_map für nachfolgende Generatoren
    FIXES: Orchestrator-Init, Dependency-Resolver, Button-API-Access, Thread-Safety
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)

        self.logger = logging.getLogger(__name__)

        # Core-Generator Instanzen
        self.geology_generator = GeologyGenerator()
        self.rock_classifier = RockTypeClassifier()
        self.mass_conservation = MassConservationManager()

        # Parameter-Tracking
        self.current_parameters = {}

        # LOD-Tracking für Orchestrator-Integration hinzufügen
        self.target_lod = "FINAL"
        self.generation_in_progress = False
        self.available_lods = set()

        # Dependency-Resolver für robuste Dependencies
        self.dependency_resolver = DependencyResolver(self.data_manager)

        # Setup UI
        self.setup_geology_ui()
        self.setup_dependency_checking()

        self.setup_orchestrator_integration()

        # Initial Parameter Load
        self.load_default_parameters()

        # Input Status überwachen
        self.check_input_dependencies()

    def setup_geology_ui(self):
        """
        Funktionsweise: Erstellt spezialisierte UI für Geology-Generator
        Aufgabe: Parameter-Slider, Rock-Distribution
        """
        # Parameter Panel
        self.parameter_panel = self.create_geology_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)

        # LOD Control Panel
        self.lod_control_panel = self.create_lod_control_panel()
        self.control_panel.layout().addWidget(self.lod_control_panel)

        # Rock Distribution Widget
        self.rock_distribution_widget = RockDistributionWidget()
        self.control_panel.layout().addWidget(self.rock_distribution_widget)

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

    def create_lod_control_panel(self) -> QGroupBox:
        """Erstellt LOD-Control Panel für Geology Quality Selection"""
        panel = QGroupBox("Quality / LOD Control")
        layout = QVBoxLayout()

        # Target-LOD Selection
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Quality:"))

        self.target_lod_combo = QComboBox()
        self.target_lod_combo.addItems(["LOD64 (Fast Preview)", "LOD128 (Medium)", "LOD256 (High)", "FINAL (Best)"])
        self.target_lod_combo.setCurrentIndex(3)
        self.target_lod_combo.currentTextChanged.connect(self.on_target_lod_changed)
        target_layout.addWidget(self.target_lod_combo)

        layout.addLayout(target_layout)

        # Generation Progress
        self.generation_progress = QProgressBar()
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        self.generation_progress.setTextVisible(True)
        self.generation_progress.setFormat("Ready")
        layout.addWidget(self.generation_progress)

        panel.setLayout(layout)
        return panel

    def create_visualization_controls(self):
        """
        Funktionsweise: Erstellt Geology-spezifische Visualization-Controls - ÜBERSCHRIEBEN
        Aufgabe: Erweitert Base-Controls um Geology-Modi
        Return: QWidget mit Geology-Visualization-Controls
        """
        widget = super().create_visualization_controls()
        layout = widget.layout()

        # Geology-spezifische Modi hinzufügen
        self.hardness_radio = QRadioButton("Hardness")
        self.hardness_radio.setStyleSheet("font-size: 11px;")
        self.hardness_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.hardness_radio, 1)
        layout.insertWidget(1, self.hardness_radio)

        self.rock_types_radio = QRadioButton("Rock Types")
        self.rock_types_radio.setStyleSheet("font-size: 11px;")
        self.rock_types_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.rock_types_radio, 2)
        layout.insertWidget(2, self.rock_types_radio)

        # 3D Terrain Toggle
        self.terrain_3d_checkbox = QCheckBox("3D Terrain")
        self.terrain_3d_checkbox.setStyleSheet("font-size: 10px;")
        self.terrain_3d_checkbox.toggled.connect(self.toggle_3d_terrain)
        layout.addWidget(self.terrain_3d_checkbox)

        return widget

    def setup_dependency_checking(self):
        """
        Funktionsweise: Setup für Input-Dependency Checking
        Aufgabe: Überwacht verfügbare Inputs von terrain_tab
        """
        self.required_dependencies = VALIDATION_RULES.DEPENDENCIES["geology"]

        self.dependency_status = StatusIndicator("Input Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

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

    def set_parameters(self, parameters: dict):
        """
        Funktionsweise: Setzt Parameter-Werte mit Synchronisation (Fix für Problem 8)
        Parameter: parameters (dict)
        """
        for param_name, value in parameters.items():
            if param_name in self.parameter_sliders:
                self.parameter_sliders[param_name].setValue(value)

        # Parameter-State synchronisieren (Fix für Problem 8)
        self.current_parameters = self.get_current_parameters()

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
        Funktionsweise: Prüft ob alle Required Dependencies verfügbar sind - REPARIERT
        Aufgabe: Aktiviert/Deaktiviert Generation Button und zeigt Status
        REPARIERT: Robuste Dependency-Resolution mit Retry-Logic (Fix für Problem 20)
        """
        # Nutze Dependency-Resolver für robuste Dependencies (Fix für Problem 20)
        is_complete, missing = self.dependency_resolver.resolve_dependencies("geology", self.required_dependencies)

        if is_complete:
            self.dependency_status.set_success("All inputs available")
            # FIX für Problem 9: Public API für Button-Control
            if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                self.auto_simulation_panel.set_manual_button_enabled(True)
        else:
            self.dependency_status.set_warning(f"Missing: {', '.join(missing)}")
            # FIX für Problem 9: Public API für Button-Control
            if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
                self.auto_simulation_panel.set_manual_button_enabled(False)

        return is_complete

    def generate(self):
        """Hauptmethode für Geology-Generation mit Orchestrator Integration"""
        if not self.generation_orchestrator:
            self.logger.error("No GenerationOrchestrator available")
            self.handle_generation_error(Exception("GenerationOrchestrator not available"))
            return

        if self.generation_in_progress:
            self.logger.info("Generation already in progress, ignoring request")
            return

        if not self.check_input_dependencies():
            self.logger.warning("Cannot generate geology - missing dependencies")
            return

        try:
            self.logger.info(f"Starting geology generation with target LOD: {self.target_lod}")

            self.start_generation_timing()
            self.generation_in_progress = True

            request_id = self.generation_orchestrator.request_generation(
                generator_type="geology",
                parameters=self.current_parameters.copy(),
                target_lod=self.target_lod,
                source_tab="geology",
                priority=10
            )

            if request_id:
                self.logger.info(f"Geology generation requested: {request_id}")
            else:
                raise Exception("Failed to request generation from orchestrator")

        except Exception as e:
            self.generation_in_progress = False
            self.handle_generation_error(e)
            raise

    @pyqtSlot(str)
    def on_target_lod_changed(self, combo_text: str):
        """Slot für Target-LOD-Änderungen"""
        if "LOD64" in combo_text:
            self.target_lod = "LOD64"
        elif "LOD128" in combo_text:
            self.target_lod = "LOD128"
        elif "LOD256" in combo_text:
            self.target_lod = "LOD256"
        elif "FINAL" in combo_text:
            self.target_lod = "FINAL"

        self.logger.info(f"Target LOD changed to: {self.target_lod}")

    def setup_orchestrator_integration(self):
        """Setup für GenerationOrchestrator Integration"""
        if not self.generation_orchestrator:
            self.logger.warning("No GenerationOrchestrator provided to GeologyTab")
            return

        self.generation_orchestrator.generation_started.connect(self.on_orchestrator_generation_started)
        self.generation_orchestrator.generation_completed.connect(self.on_orchestrator_generation_completed)
        self.generation_orchestrator.generation_progress.connect(self.on_orchestrator_generation_progress)

    @pyqtSlot(str, str)
    def on_orchestrator_generation_started(self, generator_type: str, lod_level: str):
        """Handler für Generation-Start vom Orchestrator"""
        if generator_type != "geology":
            return

        QMetaObject.invokeMethod(self, "_update_ui_generation_started",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, lod_level))

    @pyqtSlot(str)
    def _update_ui_generation_started(self, lod_level: str):
        """UI-Update für Generation-Start in Main-Thread"""
        self.generation_progress.setValue(0)
        self.generation_progress.setFormat(f"Generating {lod_level}...")

    @pyqtSlot(str, str, bool)
    def on_orchestrator_generation_completed(self, generator_type: str, lod_level: str, success: bool):
        """Handler für Generation-Completion vom Orchestrator"""
        if generator_type != "geology":
            return

        QMetaObject.invokeMethod(self, "_update_ui_generation_completed",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, lod_level),
                                 Q_ARG(bool, success))

    @pyqtSlot(str, bool)
    def _update_ui_generation_completed(self, lod_level: str, success: bool):
        """UI-Update für Generation-Completion in Main-Thread"""
        if success:
            self.available_lods.add(lod_level)
            self.generation_progress.setValue(100)
            self.generation_progress.setFormat(f"{lod_level} Complete")

            self.update_geology_display()

            rock_map = self.data_manager.get_geology_data("rock_map")
            hardness_map = self.data_manager.get_geology_data("hardness_map")
            if rock_map is not None and hardness_map is not None:
                self.rock_distribution_widget.update_statistics(rock_map, hardness_map)
        else:
            self.generation_progress.setFormat(f"{lod_level} Failed")

        if lod_level == self.target_lod:
            self.generation_in_progress = False
            self.end_generation_timing(success)

    @pyqtSlot(str, str, int, str)
    def on_orchestrator_generation_progress(self, generator_type: str, lod_level: str, progress_percent: int,
                                            detail: str):
        """Handler für Generation-Progress vom Orchestrator"""
        if generator_type != "weather":
            return

        QMetaObject.invokeMethod(self, "_update_ui_generation_progress",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, lod_level),
                                 Q_ARG(int, progress_percent))

    @pyqtSlot(str, int)
    def _update_ui_generation_progress(self, lod_level: str, progress_percent: int):
        """UI-Update für Generation-Progress in Main-Thread"""
        self.generation_progress.setValue(progress_percent)
        self.generation_progress.setFormat(f"{lod_level} - {progress_percent}%")

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

        current_display = self.get_current_display()
        if not current_display:
            self.logger.debug("No display available")
            return

        if current_mode == 0:  # Heightmap (Base)
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                current_display.update_display(heightmap, "heightmap")

        elif current_mode == 1:  # Hardness Map
            if hardness_map is not None:
                current_display.update_display(hardness_map, "hardness_map")

        elif current_mode == 2:  # Rock Types
            current_display.update_display(rock_map, "rock_map")

        # 3D Terrain Overlay wenn aktiviert
        if hasattr(self, 'terrain_3d_checkbox') and self.terrain_3d_checkbox.isChecked():
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None and hasattr(current_display, 'overlay_3d_terrain'):
                current_display.overlay_3d_terrain(heightmap)

    @pyqtSlot()
    def update_display_mode(self):
        """
        Funktionsweise: Handler für Geology Display-Mode-Änderungen - ÜBERSCHRIEBEN
        Aufgabe: Geology-spezifische Visualization-Modi
        """
        if not hasattr(self, 'display_mode'):
            return

        current_mode = self.display_mode.checkedId()
        current_display = self.get_current_display()

        if not current_display:
            return

        try:
            if current_mode == 0:  # Heightmap (Base)
                heightmap = self.data_manager.get_terrain_data("heightmap")
                if heightmap is not None:
                    current_display.update_display(heightmap, "heightmap")

            elif current_mode == 1:  # Hardness Map
                hardness_map = self.data_manager.get_geology_data("hardness_map")
                if hardness_map is not None:
                    current_display.update_display(hardness_map, "hardness_map")

            elif current_mode == 2:  # Rock Types
                rock_map = self.data_manager.get_geology_data("rock_map")
                if rock_map is not None:
                    current_display.update_display(rock_map, "rock_map")

        except Exception as e:
            self.logger.error(f"Geology display mode update failed: {e}")

    @pyqtSlot(bool)
    def toggle_3d_terrain(self, enabled: bool):
        """
        Funktionsweise: Toggle für 3D Terrain Overlay
        Aufgabe: Aktiviert/Deaktiviert 3D Heightmap Overlay über Geology-Data
        """
        self.update_geology_display()

    def cleanup_resources(self):
        """
        Funktionsweise: Cleanup-Methode für Resource-Management - ERWEITERT
        Aufgabe: Wird beim Tab-Wechsel oder Schließen aufgerufen
        ERWEITERT: Dependency-Resolver Cleanup
        """
        # Dependency-Resolver cleanup
        if hasattr(self, 'dependency_resolver'):
            self.dependency_resolver.reset_retries("geology")

        # Parent Cleanup aufrufen
        super().cleanup_resources()


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


# Zusätzliche Utility-Funktionen für Geology-Tab:

def calculate_rock_type_distribution(rock_map: np.ndarray) -> dict:
    """
    Funktionsweise: Berechnet Verteilung der Gesteinstypen
    Parameter: rock_map (RGB numpy array)
    Return: dict mit Prozentanteilen pro Gesteinstyp
    """
    total_pixels = rock_map.shape[0] * rock_map.shape[1]

    distribution = {
        "sedimentary": np.sum(rock_map[:, :, 0]) / (total_pixels * 255) * 100,
        "igneous": np.sum(rock_map[:, :, 1]) / (total_pixels * 255) * 100,
        "metamorphic": np.sum(rock_map[:, :, 2]) / (total_pixels * 255) * 100
    }

    return distribution


def validate_mass_conservation(rock_map: np.ndarray, tolerance: float = 1.0) -> bool:
    """
    Funktionsweise: Validiert Mass-Conservation für Rock-Map
    Parameter: rock_map (RGB array), tolerance (float)
    Return: bool - Mass Conservation erfüllt
    """
    mass_sums = np.sum(rock_map, axis=2)
    target_mass = 255  # RGB sollte zu 255 summieren

    # Check ob alle Pixel innerhalb Tolerance sind
    deviations = np.abs(mass_sums - target_mass)
    max_deviation = np.max(deviations)

    return max_deviation <= tolerance


def create_hardness_visualization(hardness_map: np.ndarray) -> np.ndarray:
    """
    Funktionsweise: Erstellt Visualization für Hardness-Map
    Parameter: hardness_map (2D numpy array)
    Return: RGB visualization array
    """
    # Normalisiere Hardness-Werte auf 0-1
    normalized = (hardness_map - np.min(hardness_map)) / (np.max(hardness_map) - np.min(hardness_map))

    # Erstelle Colormap: Soft -> Hard (Blau -> Rot)
    visualization = np.zeros((hardness_map.shape[0], hardness_map.shape[1], 3), dtype=np.uint8)

    # Blau für weiche Gesteine
    visualization[:, :, 2] = (255 * (1 - normalized)).astype(np.uint8)

    # Rot für harte Gesteine
    visualization[:, :, 0] = (255 * normalized).astype(np.uint8)

    # Grün für mittlere Härten
    middle_mask = (normalized > 0.3) & (normalized < 0.7)
    visualization[middle_mask, 1] = (128 * (1 - np.abs(normalized[middle_mask] - 0.5) * 2)).astype(np.uint8)

    return visualization