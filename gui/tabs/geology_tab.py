"""
Path: gui/tabs/geology_tab.py

Funktionsweise: Geology-Editor mit vollständiger BaseTab-Integration - CLEAN IMPLEMENTATION
- Nutzt BaseTab's generate() als einheitliche Hauptfunktion
- Vollständige Integration: GenerationOrchestrator, ParameterManager, DataLODManager
- Auto-Tab-Generation: Automatische Generation bei Dependencies-Erfüllung
- Eliminiert alle QMetaObject-Patterns und redundante Handler
- Input: heightmap, slopemap von terrain_tab
- Output: rock_map (RGB), hardness_map für nachfolgende Generatoren
"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QRadioButton,
    QCheckBox, QProgressBar, QGridLayout, QComboBox
)
from PyQt5.QtCore import pyqtSlot
import numpy as np
import logging

from .base_tab import BaseMapTab
from gui.config.value_default import GEOLOGY
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from core.geology_generator import GeologySystemGenerator, RockTypeClassifier, MassConservationManager


class GeologyTab(BaseMapTab):
    """
    Funktionsweise: Geology-Generator Tab mit vollständiger BaseTab-Integration
    Aufgabe: Koordiniert Geology-Generation über einheitliche BaseTab-Architektur
    Input: heightmap, slopemap von terrain_tab
    Output: rock_map (RGB), hardness_map über DataLODManager
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        # Required dependencies für BaseTab
        self.required_dependencies = ["heightmap", "slopemap"]

        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)

        self.logger = logging.getLogger(__name__)

        # Core-Generator (nur für direkte Fallback-Generation)
        self.geology_generator = GeologySystemGenerator()
        self.rock_classifier = RockTypeClassifier()
        self.mass_conservation = MassConservationManager()

        # Enhanced Features - WIEDERHERGESTELLT
        self.dependency_validation_enabled = True
        self.data_validation_enabled = True


        # Setup standardisierte Orchestrator-Integration
        self.setup_standard_orchestrator_handlers("geology")

        # Setup UI mit standardisierten Patterns
        self.create_parameter_controls()
        self.create_geology_specific_widgets()
        self.create_enhanced_geology_ui()

        # Load default parameters
        self._load_default_geology_parameters()

        # Initial dependency check
        self.check_input_dependencies()

    def create_parameter_controls(self):
        """
        Funktionsweise: Erstellt Geology-Parameter-Controls - STANDARDISIERT
        Aufgabe: Nutzt BaseTab's control_panel für einheitliches Layout
        """
        # Rock Hardness Parameters
        hardness_panel = QGroupBox("Rock Hardness")
        hardness_layout = QVBoxLayout()

        self.parameter_sliders = {}

        hardness_params = [
            ("sedimentary_hardness", "Sedimentary Hardness"),
            ("igneous_hardness", "Igneous Hardness"),
            ("metamorphic_hardness", "Metamorphic Hardness")
        ]

        for param_key, param_label in hardness_params:
            param_config = GEOLOGY.__dict__[param_key.upper()]

            slider = ParameterSlider(
                label=param_label,
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 1),
                suffix=param_config.get("suffix", "")
            )
            slider.valueChanged.connect(self._on_parameter_changed)
            self.parameter_sliders[param_key] = slider
            hardness_layout.addWidget(slider)

        hardness_panel.setLayout(hardness_layout)
        self.control_panel.layout().addWidget(hardness_panel)

        # Tectonic Deformation Parameters
        deformation_panel = QGroupBox("Tectonic Deformation")
        deformation_layout = QVBoxLayout()

        deformation_params = [
            ("ridge_warping", "Ridge Warping"),
            ("bevel_warping", "Bevel Warping"),
            ("metamorph_foliation", "Metamorphic Foliation"),
            ("metamorph_folding", "Metamorphic Folding"),
            ("igneous_flowing", "Igneous Flowing")
        ]

        for param_key, param_label in deformation_params:
            param_config = GEOLOGY.__dict__[param_key.upper()]

            slider = ParameterSlider(
                label=param_label,
                min_val=param_config["min"],
                max_val=param_config["max"],
                default_val=param_config["default"],
                step=param_config.get("step", 0.1),
                suffix=param_config.get("suffix", "")
            )
            slider.valueChanged.connect(self._on_parameter_changed)
            self.parameter_sliders[param_key] = slider
            deformation_layout.addWidget(slider)

        deformation_panel.setLayout(deformation_layout)
        self.control_panel.layout().addWidget(deformation_panel)

    def create_enhanced_geology_ui(self):
        """
        Funktionsweise: Erstellt erweiterte Geology UI-Komponenten - WIEDERHERGESTELLT
        Aufgabe: LOD-Control, Progress-Tracking, Dependency-Status
        """
        # LOD Control Panel - WIEDERHERGESTELLT
        self.lod_control_panel = self.create_geology_lod_control_panel()
        self.control_panel.layout().addWidget(self.lod_control_panel)

        # Enhanced Dependency Status - WIEDERHERGESTELLT
        self.dependency_status = StatusIndicator("Input Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def create_geology_lod_control_panel(self):
        """
        Funktionsweise: LOD-Control Panel für Geology - WIEDERHERGESTELLT
        Aufgabe: Target-LOD Selection und detaillierte Progress-Anzeige
        """
        panel = QGroupBox("Geology Quality Control")
        layout = QVBoxLayout()

        # Target-LOD Selection - WIEDERHERGESTELLT
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Target Quality:"))

        self.target_lod_combo = QComboBox()
        self.target_lod_combo.addItems([
            "LOD64 (Fast Preview)",
            "LOD128 (Medium Quality)",
            "LOD256 (High Quality)",
            "FINAL (Best Quality)"
        ])
        self.target_lod_combo.setCurrentIndex(3)  # Default: FINAL
        self.target_lod_combo.currentTextChanged.connect(self._on_target_lod_changed)
        target_layout.addWidget(self.target_lod_combo)

        layout.addLayout(target_layout)

        # Detaillierte Progress Bar - WIEDERHERGESTELLT
        self.generation_progress = QProgressBar()
        self.generation_progress.setRange(0, 100)
        self.generation_progress.setValue(0)
        self.generation_progress.setTextVisible(True)
        self.generation_progress.setFormat("Ready")
        layout.addWidget(self.generation_progress)

        panel.setLayout(layout)
        return panel

    def _on_target_lod_changed(self, combo_text: str):
        """Target-LOD aus Combo-Box extrahieren"""
        if "LOD64" in combo_text:
            self.target_lod = "LOD64"
        elif "LOD128" in combo_text:
            self.target_lod = "LOD128"
        elif "LOD256" in combo_text:
            self.target_lod = "LOD256"
        elif "FINAL" in combo_text:
            self.target_lod = "FINAL"
        else:
            self.target_lod = "FINAL"  # Fallback

    def create_geology_specific_widgets(self):
        """Erstellt Geology-spezifische Widgets"""
        # Rock Distribution Widget
        self.rock_distribution_widget = RockDistributionWidget()
        self.control_panel.layout().addWidget(self.rock_distribution_widget)

    def create_visualization_controls(self):
        """
        Funktionsweise: Erstellt Geology-spezifische Display-Controls - ERWEITERT BaseTab
        Aufgabe: Fügt Geology-Modi zu BaseTab's Standard-Controls hinzu
        """
        widget = super().create_visualization_controls()
        layout = widget.layout()

        # Rock Types Mode hinzufügen
        self.rock_types_radio = QRadioButton("Rock Types")
        self.rock_types_radio.setStyleSheet("font-size: 11px;")
        self.rock_types_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.rock_types_radio, 1)
        layout.insertWidget(1, self.rock_types_radio)

        # Hardness Mode hinzufügen
        self.hardness_radio = QRadioButton("Hardness")
        self.hardness_radio.setStyleSheet("font-size: 11px;")
        self.hardness_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.hardness_radio, 2)
        layout.insertWidget(2, self.hardness_radio)

        # 3D Terrain Overlay Toggle
        self.terrain_3d_checkbox = QCheckBox("3D Terrain")
        self.terrain_3d_checkbox.setStyleSheet("font-size: 10px;")
        self.terrain_3d_checkbox.toggled.connect(self.update_display_mode)
        layout.addWidget(self.terrain_3d_checkbox)

        return widget

    def update_display_mode(self):
        """
        Funktionsweise: Geology Display-Mode Handler - NUTZT BaseTab's Renderer
        Aufgabe: Delegiert an BaseTab's _render_current_mode für konsistente Rendering
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
                    self._render_current_mode(0, current_display, heightmap, "heightmap")

            elif current_mode == 1:  # Rock Types
                rock_map = self.data_manager.get_geology_data("rock_map")
                if rock_map is not None:
                    self._render_current_mode(1, current_display, rock_map, "rock_map")

            elif current_mode == 2:  # Hardness Map
                hardness_map = self.data_manager.get_geology_data("hardness_map")
                if hardness_map is not None:
                    self._render_current_mode(2, current_display, hardness_map, "hardness_map")

            # 3D Terrain Overlay wenn aktiviert
            if hasattr(self, 'terrain_3d_checkbox') and self.terrain_3d_checkbox.isChecked():
                heightmap = self.data_manager.get_terrain_data("heightmap")
                if heightmap is not None and hasattr(current_display, 'overlay_3d_terrain'):
                    current_display.overlay_3d_terrain(heightmap)

        except Exception as e:
            self.logger.error(f"Geology display mode update failed: {e}")

    def generate_geology_system(self):
        """
        Funktionsweise: Core Geology-Generation - WIRD VON BaseTab's generate() AUFGERUFEN
        Aufgabe: Implementiert tab-spezifische Generator-Logic mit Input-Validation
        """
        try:
            # Input-Validation
            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")

            if heightmap is None or slopemap is None:
                raise ValueError("Missing terrain data for geology generation")

            parameters = self.get_current_parameters()

            # Core-Generation ausführen
            result = self.geology_generator.generate(parameters, heightmap, slopemap)

            # Results über DataLODManager speichern
            current_lod = self.data_manager.get_current_lod_level("geology")

            self.data_manager.set_geology_data_lod("rock_map", result.rock_map, current_lod, parameters)
            self.data_manager.set_geology_data_lod("hardness_map", result.hardness_map, current_lod, parameters)

            # Rock Distribution Widget aktualisieren
            self.rock_distribution_widget.update_statistics(result.rock_map, result.hardness_map)

            # Signal für andere Tabs (automatisch durch BaseTab)
            self.data_updated.emit("geology", "rock_map")
            self.data_updated.emit("geology", "hardness_map")

            self.logger.info(f"Geology generation completed at LOD {current_lod}")
            return True

        except Exception as e:
            self.logger.error(f"Geology generation failed: {e}")
            raise

    def get_current_parameters(self):
        """
        Funktionsweise: Sammelt aktuelle Parameter von allen Slidern
        Return: dict mit allen Geology-Parametern
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    def _on_parameter_changed(self):
        """
        Funktionsweise: Parameter-Change Handler - NUTZT BaseTab's Auto-Simulation
        Aufgabe: Triggert BaseTab's Auto-Simulation-System
        """
        # Parameter-Change an BaseTab weiterleiten
        parameters = self.get_current_parameters()
        self.parameter_changed.emit("geology", parameters)

        # Rock Distribution Preview aktualisieren
        self.rock_distribution_widget.update_distribution(parameters)

        # Auto-Simulation über BaseTab triggern
        if self.auto_simulation_enabled:
            self.auto_simulation_timer.start(500)  # 500ms debounce

    def _load_default_geology_parameters(self):
        """Lädt Default-Parameter in alle Slider"""
        for param_name, slider in self.parameter_sliders.items():
            param_config = GEOLOGY.__dict__[param_name.upper()]
            slider.setValue(param_config["default"])

    def check_input_dependencies(self):
        """
        Funktionsweise: Dependency-Check für Geology - ÜBERSCHREIBT BaseTab
        Aufgabe: Prüft ob Terrain-Daten verfügbar sind
        """
        try:
            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")

            dependencies_met = heightmap is not None and slopemap is not None

            # UI-Status aktualisieren
            if hasattr(self, 'auto_simulation_panel'):
                self.auto_simulation_panel.set_manual_button_enabled(dependencies_met)

            # Dependency-Status anzeigen
            if dependencies_met:
                status_msg = "All terrain inputs available"
                if hasattr(self, 'error_status'):
                    self.error_status.set_success(status_msg)
            else:
                missing = []
                if heightmap is None:
                    missing.append("heightmap")
                if slopemap is None:
                    missing.append("slopemap")

                status_msg = f"Missing terrain data: {', '.join(missing)}"
                if hasattr(self, 'error_status'):
                    self.error_status.set_warning(status_msg)

            return dependencies_met

        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False

    def on_external_data_updated(self, generator_type: str, data_key: str):
        """
        Funktionsweise: External Data Handler - ÜBERSCHREIBT BaseTab
        Aufgabe: Reagiert auf Terrain-Updates und triggert Auto-Generation
        """
        if generator_type == "terrain" and data_key in self.required_dependencies:
            dependencies_met = self.check_input_dependencies()

            # AUTO-TAB-GENERATION: Automatisch generieren wenn Dependencies erfüllt
            if dependencies_met and self.auto_simulation_enabled:
                self.logger.info("Dependencies satisfied - triggering auto-generation")
                self.auto_simulation_timer.start(1000)  # 1s delay für Auto-Generation


class RockDistributionWidget(QGroupBox):
    """
    Funktionsweise: Widget für Rock-Distribution Visualization und Statistics
    Aufgabe: Zeigt Rock-Hardness Preview und Mass-Conservation Status
    """

    def __init__(self):
        super().__init__("Rock Distribution")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Rock-Distribution Display"""
        layout = QVBoxLayout()

        # Hardness Preview Bars
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
        Funktionsweise: Aktualisiert Hardness Preview basierend auf Parametern
        Parameter: parameters (dict mit hardness values)
        """
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
        try:
            # Rock Distribution Percentages berechnen
            total_pixels = rock_map.shape[0] * rock_map.shape[1]

            sedimentary_pct = np.sum(rock_map[:, :, 0]) / (total_pixels * 255) * 100
            igneous_pct = np.sum(rock_map[:, :, 1]) / (total_pixels * 255) * 100
            metamorphic_pct = np.sum(rock_map[:, :, 2]) / (total_pixels * 255) * 100

            # Mass Conservation Check
            mass_sums = np.sum(rock_map, axis=2)
            mass_conservation_valid = np.allclose(mass_sums, 255, atol=5)

            # Update Statistics Display
            self.distribution_stats.setText(
                f"Distribution: Sed {sedimentary_pct:.1f}%, Ign {igneous_pct:.1f}%, Met {metamorphic_pct:.1f}%"
            )

            if mass_conservation_valid:
                self.mass_conservation_status.set_success("Mass conserved (R+G+B=255)")
            else:
                self.mass_conservation_status.set_warning("Mass conservation violation detected")

        except Exception as e:
            self.distribution_stats.setText("Statistics calculation failed")
            self.mass_conservation_status.set_error(f"Error: {str(e)}")

    def generate_geology_system(self):
        """
        Funktionsweise: Core Geology-Generation - WIRD VON BaseTab's generate() AUFGERUFEN
        Aufgabe: Implementiert tab-spezifische Generator-Logic mit robuster Input-Validation
        """
        try:
            # ENHANCED INPUT-VALIDATION - WIEDERHERGESTELLT für Robustheit
            if not self._validate_input_data():
                raise ValueError("Input validation failed - terrain data invalid or missing")

            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")
            parameters = self.get_current_parameters()

            # Core-Generation ausführen
            result = self.geology_generator.generate(parameters, heightmap, slopemap)

            # ROCK_MAP-VALIDATION - WIEDERHERGESTELLT für Data-Integrity
            if not self._validate_rock_map(result.rock_map):
                self.logger.warning("Generated rock_map failed validation - applying corrections")
                result.rock_map = self._repair_rock_map(result.rock_map)

            # Results über DataLODManager speichern
            current_lod = getattr(self, 'target_lod', 'FINAL')

            self.data_manager.set_geology_data_lod("rock_map", result.rock_map, current_lod, parameters)
            self.data_manager.set_geology_data_lod("hardness_map", result.hardness_map, current_lod, parameters)

            # Rock Distribution Widget aktualisieren
            self.rock_distribution_widget.update_statistics(result.rock_map, result.hardness_map)

            # Signal für andere Tabs (automatisch durch BaseTab)
            self.data_updated.emit("geology", "rock_map")
            self.data_updated.emit("geology", "hardness_map")

            self.logger.info(f"Geology generation completed at LOD {current_lod}")
            return True

        except Exception as e:
            self.logger.error(f"Geology generation failed: {e}")
            raise

    def _validate_input_data(self) -> bool:
        """
        Funktionsweise: Robuste Input-Validation - WIEDERHERGESTELLT
        Aufgabe: Validiert Terrain-Inputs vor Generation
        """
        try:
            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")

            if heightmap is None:
                self.logger.error("Missing heightmap from terrain generation")
                return False

            if slopemap is None:
                self.logger.error("Missing slopemap from terrain generation")
                return False

            # Shape Validation
            if heightmap.shape != slopemap.shape[:2]:
                self.logger.error(f"Shape mismatch: heightmap {heightmap.shape} vs slopemap {slopemap.shape[:2]}")
                return False

            # Data Type Validation
            if not isinstance(heightmap, np.ndarray) or not isinstance(slopemap, np.ndarray):
                self.logger.error("Terrain data must be numpy arrays")
                return False

            # Size Validation
            h, w = heightmap.shape
            if h < 32 or w < 32 or h > 2048 or w > 2048:
                self.logger.error(f"Invalid terrain size: {h}x{w}")
                return False

            # Data Range Validation
            if np.any(np.isnan(heightmap)) or np.any(np.isinf(heightmap)):
                self.logger.error("Invalid values in heightmap (NaN/Inf)")
                return False

            self.logger.info(f"Geology input validation passed: {heightmap.shape}")
            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False

    def _validate_rock_map(self, rock_map: np.ndarray) -> bool:
        """
        Funktionsweise: Rock_map-Validation - WIEDERHERGESTELLT für Data-Integrity
        Parameter: rock_map (numpy array)
        Return: bool - Rock_map ist valide
        """
        try:
            if not isinstance(rock_map, np.ndarray):
                self.logger.error(f"Rock_map is not numpy array: {type(rock_map)}")
                return False

            if len(rock_map.shape) != 3 or rock_map.shape[2] != 3:
                self.logger.error(f"Rock_map wrong shape: {rock_map.shape}, expected (H, W, 3)")
                return False

            # Data Range Validation
            if np.any(rock_map < 0) or np.any(rock_map > 255):
                self.logger.error(f"Rock_map values out of range [0,255]: {np.min(rock_map)}-{np.max(rock_map)}")
                return False

            # Mass Conservation Check
            mass_sums = np.sum(rock_map, axis=2)
            mass_deviation = np.abs(mass_sums - 255)
            max_deviation = np.max(mass_deviation)

            if max_deviation > 10:  # 10 Tolerance für Rundungsfehler
                self.logger.warning(f"Rock_map mass conservation issue: max deviation {max_deviation}")
                return False

            self.logger.info(f"Rock_map validation passed: {rock_map.shape}, mass_dev={max_deviation:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"Rock_map validation exception: {e}")
            return False

    def _repair_rock_map(self, rock_map: np.ndarray) -> np.ndarray:
        """
        Funktionsweise: Rock_map-Repair für fehlerhafte Daten - WIEDERHERGESTELLT
        Parameter: rock_map (potentially corrupted)
        Return: repaired rock_map
        """
        try:
            if rock_map is None or not isinstance(rock_map, np.ndarray):
                return self._create_fallback_rock_map()

            # Ensure correct shape
            if len(rock_map.shape) != 3 or rock_map.shape[2] != 3:
                return self._create_fallback_rock_map()

            # Clamp values to valid range
            rock_map = np.clip(rock_map, 0, 255)

            # Mass Conservation Repair
            mass_sums = np.sum(rock_map, axis=2)
            invalid_pixels = mass_sums != 255

            if np.any(invalid_pixels):
                self.logger.info(f"Repairing {np.sum(invalid_pixels)} pixels with mass conservation issues")

                # Normalize to 255 for invalid pixels
                for i in range(rock_map.shape[2]):
                    rock_map[invalid_pixels, i] = (rock_map[invalid_pixels, i] / mass_sums[invalid_pixels]) * 255

                # Handle division by zero cases
                zero_mass_pixels = mass_sums == 0
                if np.any(zero_mass_pixels):
                    rock_map[zero_mass_pixels, 0] = 128  # Sedimentary
                    rock_map[zero_mass_pixels, 1] = 64  # Igneous
                    rock_map[zero_mass_pixels, 2] = 63  # Metamorphic

            return rock_map.astype(np.uint8)

        except Exception as e:
            self.logger.error(f"Rock_map repair failed: {e}")
            return self._create_fallback_rock_map()

    def _create_fallback_rock_map(self) -> np.ndarray:
        """Erstellt Fallback-Rock_map bei Validation/Repair-Fehlern"""
        try:
            # Use terrain size if available
            heightmap = self.data_manager.get_terrain_data("heightmap")
            if heightmap is not None:
                h, w = heightmap.shape
            else:
                h, w = 256, 256  # Default size

            rock_map = np.zeros((h, w, 3), dtype=np.uint8)
            rock_map[:, :, 0] = 128  # Sedimentary
            rock_map[:, :, 1] = 64  # Igneous
            rock_map[:, :, 2] = 63  # Metamorphic (Sum = 255)

            self.logger.info(f"Created fallback rock_map: {rock_map.shape}")
            return rock_map

        except Exception as e:
            self.logger.error(f"Fallback rock_map creation failed: {e}")
            # Ultimate fallback

    def check_input_dependencies(self):
        """
        Funktionsweise: Enhanced Dependency-Check - ERWEITERT für robustere Validation
        Aufgabe: Prüft Dependencies mit detailliertem Status-Feedback
        """
        try:
            heightmap = self.data_manager.get_terrain_data("heightmap")
            slopemap = self.data_manager.get_terrain_data("slopemap")

            dependencies_met = True
            status_messages = []

            # Individual Dependency Checks
            if heightmap is None:
                dependencies_met = False
                status_messages.append("Missing heightmap")
            else:
                status_messages.append(f"Heightmap: {heightmap.shape}")

            if slopemap is None:
                dependencies_met = False
                status_messages.append("Missing slopemap")
            else:
                status_messages.append(f"Slopemap: {slopemap.shape}")

            # Enhanced Validation wenn beide verfügbar
            if heightmap is not None and slopemap is not None:
                if heightmap.shape != slopemap.shape[:2]:
                    dependencies_met = False
                    status_messages.append("Shape mismatch between terrain data")
                else:
                    status_messages.append("Data consistency: OK")

            # UI-Status Updates
            if hasattr(self, 'auto_simulation_panel'):
                self.auto_simulation_panel.set_manual_button_enabled(dependencies_met)

            # Enhanced Dependency Status Display - WIEDERHERGESTELLT
            if hasattr(self, 'dependency_status'):
                if dependencies_met:
                    self.dependency_status.set_success(f"Dependencies: {' | '.join(status_messages)}")
                else:
                    self.dependency_status.set_warning(f"Issues: {' | '.join(status_messages)}")

            return dependencies_met

        except Exception as e:
            self.logger.error(f"Enhanced dependency check failed: {e}")
            if hasattr(self, 'dependency_status'):
                self.dependency_status.set_error(f"Dependency check error: {str(e)}")
            return False

    # ENHANCED SIGNAL-HANDLER - WIEDERHERGESTELLT für detailliertes User-Feedback

    def _on_generation_started_geology(self, generator_type: str, lod_level: int):
        """Enhanced Generation-Start Handler für Geology"""
        if generator_type != "geology":
            return

        if hasattr(self, 'generation_progress'):
            self.generation_progress.setValue(0)
            self.generation_progress.setFormat(f"Generating Geology LOD {lod_level}...")

    def _on_generation_progress_geology(self, generator_type: str, lod_level: int, progress_percent: int, detail: str):
        """Enhanced Progress Handler für Geology"""
        if generator_type != "geology":
            return

        if hasattr(self, 'generation_progress'):
            self.generation_progress.setValue(progress_percent)
            self.generation_progress.setFormat(f"LOD {lod_level}: {detail} ({progress_percent}%)")

    def _on_generation_completed_geology(self, generator_type: str, lod_level: int, success: bool):
        """Enhanced Completion Handler für Geology"""
        if generator_type != "geology":
            return

        if hasattr(self, 'generation_progress'):
            if success:
                self.generation_progress.setValue(100)
                self.generation_progress.setFormat(f"Geology LOD {lod_level} Complete")

                # Sofortiges Statistics-Update - WIEDERHERGESTELLT
                try:
                    rock_map = self.data_manager.get_geology_data("rock_map")
                    hardness_map = self.data_manager.get_geology_data("hardness_map")

                    if rock_map is not None and hardness_map is not None:
                        self.rock_distribution_widget.update_statistics(rock_map, hardness_map)

                        # Post-Generation Validation - WIEDERHERGESTELLT
                        if self.data_validation_enabled:
                            if self._validate_rock_map(rock_map):
                                self.logger.info("Post-generation rock_map validation: PASSED")
                            else:
                                self.logger.warning("Post-generation rock_map validation: FAILED")

                except Exception as e:
                    self.logger.debug(f"Post-generation updates failed: {e}")
            else:
                self.generation_progress.setFormat(f"Geology LOD {lod_level} Failed")

    def setup_standard_orchestrator_handlers(self, generator_type: str):
        """
        Funktionsweise: Erweiterte Orchestrator-Integration für Geology - ERWEITERT BaseTab
        Aufgabe: Fügt enhanced Signal-Handler zu BaseTab's Standard-Handlern hinzu
        """
        # BaseTab's Standard-Handler
        super().setup_standard_orchestrator_handlers(generator_type)

        # Geology-spezifische Handler
        if self.generation_orchestrator:
            try:
                self.generation_orchestrator.generation_started.connect(self._on_generation_started_geology)
                self.generation_orchestrator.generation_progress.connect(self._on_generation_progress_geology)
                self.generation_orchestrator.generation_completed.connect(self._on_generation_completed_geology)

                self.logger.debug("Enhanced geology orchestrator handlers connected")
            except Exception as e:
                self.logger.warning(f"Enhanced geology orchestrator handler setup failed: {e}")