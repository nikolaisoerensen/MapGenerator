"""
Path: gui/tabs/terrain_tab.py
Date changed: 24.08.2025

TerrainTab implementiert die Terrain-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den TerrainGenerator aus core/terrain_generator.py. Als Basis-Generator
ohne Dependencies liefert er heightmap, slopemap und shadowmap für alle nachgelagerten Systeme.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QLabel
)
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any, Optional

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import (
    ParameterSlider, RandomSeedButton
)
from gui.config.value_default import TERRAIN, get_parameter_config, validate_parameter_set

class TerrainTab(BaseMapTab):
    """
    Terrain-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert heightmap, slopemap und shadowmap Generation als Basis für alle anderen Generatoren.
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "terrain"
        self.required_dependencies = []  # Terrain hat keine Dependencies

        # Terrain-spezifische Attribute (vor super(), da create_parameter_controls
        # während BaseMapTab.setup_ui() darauf zugreift und sie befüllt)
        self.parameter_sliders = {}
        self.generation_button = None
        self.progress_bar = None
        self.system_status = None
        self.statistics_display = None
        self.display_mode_group = None
        self.current_display_mode = "height"

        # LOD-System Tracking
        self.current_lod = 1
        self.max_lod = 6

        self.logger = logging.getLogger("TerrainTab")

        # Manager-Integration
        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )

        # Registrierung beim ParameterManager: parameter_sliders sind durch
        # create_parameter_controls() (innerhalb von super().__init__()) bereits
        # befüllt, get_current_parameters() liefert damit sofort die Default-Werte
        # als Startwert des Caches.
        if self.parameter_manager:
            self.parameter_manager.register_tab(self.generator_type, self)

        self.logger.info("TerrainTab initialized")

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls für Terrain-Generation.
        Implementiert Required-Method von BaseMapTab.
        Stellt bei abgetrenntem Panel-Layout ein neues Layout wieder her,
        damit die Parameter-Erstellung nie leer abbricht.
        """
        if not self.control_panel:
            self.logger.error("Parameter creation skipped: control_panel is None")
            return

        if self.control_panel.layout() is None:
            repaired_layout = QVBoxLayout()
            repaired_layout.setContentsMargins(5, 5, 5, 5)
            repaired_layout.setSpacing(10)
            self.control_panel.setLayout(repaired_layout)
            self.control_panel_content_layout = repaired_layout
            self.logger.info("Control panel layout was detached - reinstalled")

        try:
            # Terrain Parameters GroupBox
            self._create_terrain_parameters()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_terrain_parameters(self):
        """Erstellt Terrain Parameter Controls"""
        terrain_group = QGroupBox("Terrain Parameters")
        terrain_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        terrain_layout = QVBoxLayout()

        # Parameter-Definitionen aus value_default.TERRAIN
        parameter_configs = [
            ("map_size", "Map Size", TERRAIN.MAPSIZE),
            ("amplitude", "Height Amplitude", TERRAIN.AMPLITUDE),
            ("octaves", "Detail Octaves", TERRAIN.OCTAVES),
            ("frequency", "Base Frequency", TERRAIN.FREQUENCY),
            ("persistence", "Detail Persistence", TERRAIN.PERSISTENCE),
            ("lacunarity", "Frequency Scaling", TERRAIN.LACUNARITY),
            ("redistribute_power", "Height Redistribution", TERRAIN.REDISTRIBUTE_POWER),
            ("map_seed", "Map Seed", TERRAIN.MAP_SEED)
        ]

        for param_key, label, config in parameter_configs:
            if param_key == "map_seed":
                # Seed Parameter mit RandomSeedButton
                seed_layout = self._create_seed_parameter(param_key, label, config)
                terrain_layout.addLayout(seed_layout)
            else:
                # Standard Parameter Slider
                slider = ParameterSlider(
                    label=label,
                    min_val=config["min"],
                    max_val=config["max"],
                    default_val=config["default"],
                    step=config["step"],
                    suffix=config.get("suffix", ""),
                    description=config.get("description", "")
                )

                # Parameter-Change Handler
                slider.valueChanged.connect(
                    lambda value, key=param_key: self._on_parameter_changed(key, value)
                )

                self.parameter_sliders[param_key] = slider
                terrain_layout.addWidget(slider)

        terrain_group.setLayout(terrain_layout)
        self.control_panel.layout().addWidget(terrain_group)

    def _create_seed_parameter(self, param_key: str, label: str, config: Dict):
        """Erstellt Seed Parameter mit RandomSeedButton"""
        seed_layout = QHBoxLayout()

        # Seed Slider
        seed_slider = ParameterSlider(
            label=label,
            min_val=config["min"],
            max_val=config["max"],
            default_val=config["default"],
            step=config["step"],
            description=config.get("description", "")
        )

        seed_slider.valueChanged.connect(
            lambda value: self._on_parameter_changed(param_key, value)
        )

        # Random Seed Button
        random_button = RandomSeedButton()
        random_button.seed_generated.connect(
            lambda seed: self._set_random_seed(param_key, seed)
        )

        self.parameter_sliders[param_key] = seed_slider

        seed_layout.addWidget(seed_slider)
        seed_layout.addWidget(random_button)

        return seed_layout

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Terrain-Statistics. Generation Control (Berechnen-Button, Ladebalken,
        System-Status) entfällt hier bewusst - das übernehmen jetzt der globale
        [GENERIEREN]-Button und die Pipeline-Status-Spalte im Shell-Layout.
        """
        stats_group = QGroupBox("Terrain Statistics")
        stats_group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        stats_layout = QVBoxLayout()

        # Statistics Labels
        self.height_range_label = QLabel("Height Range: No data")
        self.slope_stats_label = QLabel("Slope Statistics: No data")
        self.shadow_coverage_label = QLabel("Shadow Coverage: No data")
        self.performance_label = QLabel("Performance: No data")

        # Styling für Statistics
        for label in [self.height_range_label, self.slope_stats_label,
                     self.shadow_coverage_label, self.performance_label]:
            label.setStyleSheet("font-size: 10px; color: #2c3e50; padding: 2px;")

        stats_layout.addWidget(self.height_range_label)
        stats_layout.addWidget(self.slope_stats_label)
        stats_layout.addWidget(self.shadow_coverage_label)
        stats_layout.addWidget(self.performance_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

    def create_visualization_controls(self):
        """
        Erstellt Terrain-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Display Mode Controls (Height/Slope)
        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """Erstellt Height/Slope Display Mode Controls"""
        layout = QHBoxLayout()

        # Display Mode Button Group
        self.display_mode_group = QButtonGroup()

        height_radio = QRadioButton("Height")
        height_radio.setChecked(True)
        height_radio.toggled.connect(lambda checked: self._on_display_mode_changed("height", checked))
        self.display_mode_group.addButton(height_radio, 0)

        slope_radio = QRadioButton("Slope")
        slope_radio.toggled.connect(lambda checked: self._on_display_mode_changed("slope", checked))
        self.display_mode_group.addButton(slope_radio, 1)

        layout.addWidget(height_radio)
        layout.addWidget(slope_radio)

        return layout

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            # Parameter an ParameterManager weiterleiten
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            # Cross-Parameter Validation
            self._validate_parameter_constraints()

            self.logger.debug(f"Parameter changed: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

    def _set_random_seed(self, param_key: str, seed_value: int):
        """Setzt zufälligen Seed-Wert"""
        try:
            if param_key in self.parameter_sliders:
                self.parameter_sliders[param_key].setValue(seed_value)
                self._on_parameter_changed(param_key, seed_value)

        except Exception as e:
            self.logger.error(f"Random seed setting failed: {e}")

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Werte aller Terrain-Parameter-Slider.
        Wird vom ParameterManager als zentrale Quelle für die Terrain-Parameter
        genutzt (register_tab()/get_tab_parameters() rufen diese Methode auf,
        siehe gui/OldManagers/parameter_manager.py).
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    def _validate_parameter_constraints(self):
        """Validiert Cross-Parameter Constraints"""
        try:
            parameters = self.get_current_parameters()

            # Validation über value_default.py
            is_valid, warnings, errors = validate_parameter_set("terrain", parameters)

            # UI-Status Updates
            if errors:
                if self.system_status:
                    self.system_status.set_error(f"Parameter errors: {', '.join(errors)}")
            elif warnings:
                if self.system_status:
                    self.system_status.set_warning(f"Warnings: {', '.join(warnings)}")
            else:
                if self.system_status:
                    self.system_status.set_success("Parameters valid")

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")

    def generate(self):
        """
        Überschreibt BaseMapTab: Dependency-Check + Parameter-Validation vor
        der eigentlichen Generation. Wird jetzt ausschließlich über den
        globalen [GENERIEREN]-Button im Shell-Footer ausgelöst (kein eigener
        Berechnen-Button mehr im Parameter-Panel).
        """
        try:
            # Dependency Check (Terrain hat keine Dependencies)
            if not self.check_input_dependencies():
                self.logger.warning("Input dependencies not met")
                return

            # Parameter Validation
            self._validate_parameter_constraints()

            super().generate()

            self.logger.info("Terrain generation requested")

        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")
            if self.system_status:
                self.system_status.set_error(f"Generation failed: {e}")

    def _on_display_mode_changed(self, mode: str, checked: bool):
        """Handler für Display Mode Changes"""
        if checked:
            self.current_display_mode = mode
            self.update_display_mode()
            self.logger.debug(f"Display mode changed to: {mode}")

    # =============================================================================
    # DISPLAY UPDATE SYSTEM
    # =============================================================================

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab Display-Update für Terrain-spezifische Modi.
        Implementiert Height/Slope Display-Switching.
        """
        try:
            if not self.data_lod_manager:
                return

            current_display = self.get_current_display()
            if not current_display:
                return

            # Daten basierend auf Display-Mode holen
            if self.current_display_mode == "height":
                # Kombiniert (Geology-Tektonik + Water-Erosion/-Sedimentation),
                # nicht die unbearbeitete Terrain-Rohausgabe - siehe
                # DataLODManager.get_terrain_data_combined()
                data = self.data_lod_manager.get_terrain_data_combined("heightmap")
                data_type = "heightmap"
                display_data = data
            elif self.current_display_mode == "slope":
                data = self.data_lod_manager.get_terrain_data("slopemap")
                data_type = "slopemap"
                # slopemap ist (H,W,2) dx/dy-Gradient - MapDisplay2D kann nur
                # echte 2D-Bilder zeichnen. Für die Anzeige auf eine
                # Steigungs-Magnitude in Grad reduzieren (dieselbe Umrechnung
                # wie in _update_statistics()); die rohen (H,W,2)-Daten bleiben
                # für Change-Detection/Statistics unverändert.
                display_data = None
                if data is not None and hasattr(data, 'shape') and len(data.shape) == 3:
                    slope_magnitude = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
                    display_data = np.degrees(np.arctan(slope_magnitude)).astype(np.float32)
            else:
                return

            # Display Update mit Change-Detection
            if data is not None and display_data is not None and hasattr(current_display, 'update_display'):
                display_id = f"TerrainTab_{self.current_view}_{data_type}"

                if hasattr(self.data_lod_manager, 'display_update_manager'):
                    needs_update = self.data_lod_manager.display_update_manager.needs_update(
                        display_id, data, data_type
                    )

                    if needs_update:
                        self._push_data_to_current_display(display_data, data_type)
                        self.data_lod_manager.display_update_manager.mark_updated(
                            display_id, data, data_type
                        )

                        # Statistics Update
                        self._update_statistics(data, data_type)
                else:
                    # Fallback ohne Change-Detection
                    self._push_data_to_current_display(display_data, data_type)
                    self._update_statistics(data, data_type)

        except Exception as e:
            self.logger.debug(f"Display mode update failed: {e}")

    def _update_statistics(self, data, data_type: str):
        """Aktualisiert Terrain Statistics basierend auf aktuellen Daten"""
        try:
            if data_type == "heightmap" and hasattr(data, 'shape'):
                # Height Statistics
                height_min = float(np.min(data))
                height_max = float(np.max(data))
                height_mean = float(np.mean(data))
                height_std = float(np.std(data))

                self.height_range_label.setText(
                    f"Height Range: {height_min:.1f}m - {height_max:.1f}m "
                    f"(Mean: {height_mean:.1f}m ± {height_std:.1f}m)"
                )

                # Performance Metrics
                data_size_mb = (data.nbytes / (1024 * 1024))
                self.performance_label.setText(
                    f"Performance: {data.shape[0]}×{data.shape[1]} "
                    f"({data_size_mb:.1f}MB)"
                )

            elif data_type == "slopemap" and hasattr(data, 'shape') and len(data.shape) == 3:
                # Slope Statistics (data ist (H,W,2) für dx/dy Gradienten)
                slope_magnitude = np.sqrt(data[:,:,0]**2 + data[:,:,1]**2)
                max_slope = float(np.max(slope_magnitude))
                mean_slope = float(np.mean(slope_magnitude))

                # Konvertierung zu Degrees
                max_slope_deg = np.degrees(np.arctan(max_slope))
                mean_slope_deg = np.degrees(np.arctan(mean_slope))

                self.slope_stats_label.setText(
                    f"Slope Statistics: Max {max_slope_deg:.1f}°, "
                    f"Mean {mean_slope_deg:.1f}°"
                )

            # Shadow Coverage (falls verfügbar)
            shadow_data = self.data_lod_manager.get_terrain_data("shadowmap")
            if shadow_data is not None and hasattr(shadow_data, 'shape'):
                if len(shadow_data.shape) == 3:  # (H,W,7) für 7 Sonnenwinkel
                    shadow_min = float(np.min(shadow_data))
                    shadow_max = float(np.max(shadow_data))
                    shadow_mean = float(np.mean(shadow_data))

                    self.shadow_coverage_label.setText(
                        f"Shadow Coverage: {shadow_min:.2f} - {shadow_max:.2f} "
                        f"(Mean: {shadow_mean:.2f})"
                    )

        except Exception as e:
            self.logger.debug(f"Statistics update failed: {e}")

    # =============================================================================
    # GENERATION PROGRESS TRACKING
    # =============================================================================

    @pyqtSlot(int, str)
    def on_generation_progress(self, progress: int, message: str):
        """
        Überschreibt BaseMapTab Progress Handler für LOD-spezifisches Progress.
        """
        if not self.generation_active:
            return

        try:
            # LOD-Level aus Progress ableiten (0-100 Progress → LOD 1-6)
            lod_level = max(1, min(6, int((progress / 100) * self.max_lod) + 1))

            if self.progress_bar:
                # LOD-spezifischer Progress Text
                phase_text = "Heightmap Generation"
                if progress > 60:
                    phase_text = "Shadow Calculation"
                elif progress > 30:
                    phase_text = "Slope Calculation"

                self.progress_bar.set_lod_progress(lod_level, self.max_lod, phase_text)

            # Generation Button Status
            if self.generation_button:
                self.generation_button.set_loading(True)

            # System Status Update
            if self.system_status:
                self.system_status.set_pending(f"Generating LOD {lod_level}/{self.max_lod} ({progress}%)")

            self.current_lod = lod_level

        except Exception as e:
            self.logger.error(f"Progress tracking failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """
        Überschreibt BaseMapTab Completion Handler für Terrain-spezifische Completion.
        """
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type != self.generator_type:
            return

        try:
            # Progress Bar Reset
            if self.progress_bar:
                if success:
                    self.progress_bar.set_progress(100, "Completed", "")
                else:
                    self.progress_bar.reset()

            # Generation Button aktivieren
            if self.generation_button:
                self.generation_button.set_loading(False)

            # System Status
            if self.system_status:
                if success:
                    self.system_status.set_success(f"Generation completed (LOD {self.max_lod})")
                else:
                    self.system_status.set_error("Generation failed")

            # Display Update nach Completion
            if success:
                self.update_display_mode()

                # Map-Size Sync zu anderen Tabs
                if self.data_lod_manager and hasattr(self.data_lod_manager, 'sync_map_size'):
                    current_map_size = self.parameter_sliders.get('map_size', {}).getValue()
                    if current_map_size:
                        self.data_lod_manager.sync_map_size(int(current_map_size))

            # Parent-Class Completion Handler
            super().on_generation_completed(result_id, result_data)

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER SYNCHRONISATION
    # =============================================================================

    def update_parameter_ui(self, param_name: str, value):
        """
        Überschreibt BaseMapTab Parameter-UI Update für Terrain-Parameter.
        Synchronisiert UI-Controls mit ParameterManager-Updates.
        """
        try:
            if param_name in self.parameter_sliders:
                # Update ohne Signal-Emission (verhindert Loop)
                slider = self.parameter_sliders[param_name]
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

                self.logger.debug(f"Parameter UI updated: {param_name} = {value}")

            # Cross-Parameter Validation nach Update
            self._validate_parameter_constraints()

        except Exception as e:
            self.logger.error(f"Parameter UI update failed: {e}")

    # =============================================================================
    # DEPENDENCY SYSTEM (überschrieben, da Terrain keine Dependencies hat)
    # =============================================================================

    def check_input_dependencies(self) -> bool:
        """
        Überschreibt BaseMapTab Dependency Check.
        Terrain hat keine Input-Dependencies, gibt immer True zurück.
        """
        return True  # Terrain ist Basis-Generator ohne Dependencies

    # =============================================================================
    # RESOURCE MANAGEMENT
    # =============================================================================

    def cleanup_resources(self):
        """
        Erweitert BaseMapTab Cleanup für Terrain-spezifische Resources.
        """
        try:
            self.logger.debug("Cleaning up terrain-specific resources")

            # Terrain-spezifische Cleanup
            self.parameter_sliders.clear()
            self.current_lod = 1

            # Progress Reset
            if self.progress_bar:
                self.progress_bar.reset()

            # Status Reset
            if self.system_status:
                self.system_status.set_success("Ready")

            # Parent Cleanup
            super().cleanup_resources()

        except Exception as e:
            self.logger.error(f"Terrain cleanup failed: {e}")


def terrain_tab():
    """
    Factory-Funktion für TerrainTab-Erstellung.
    Wird von der Main-Application für Tab-Initialisierung verwendet.
    """
    return TerrainTab
