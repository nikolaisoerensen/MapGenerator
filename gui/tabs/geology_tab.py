"""
Path: gui/tabs/geology_tab.py

GeologyTab implementiert die Geology-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den GeologySystemGenerator aus core/geology_generator.py. Als von
Terrain abhängiger Generator (heightmap_combined, slopemap) liefert er rock_map und
hardness_map für Water und alle nachgelagerten Systeme.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QCheckBox, QLabel, QGridLayout, QProgressBar
)
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from gui.config.value_default import GEOLOGY


class GeologyTab(BaseMapTab):
    """
    Geology-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert rock_map und hardness_map Generation auf Basis der Terrain-Daten
    (heightmap_combined, slopemap).
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "geology"
        self.required_dependencies = ["heightmap", "slopemap"]

        # Geology-spezifische Attribute (vor super(), da create_parameter_controls
        # und create_visualization_controls während BaseMapTab.setup_ui() darauf
        # zugreifen und sie befüllen)
        self.parameter_sliders = {}
        self.rock_distribution_widget = None
        self.dependency_status = None
        self.display_mode_group = None
        self.terrain_3d_checkbox = None
        self.current_display_mode = "height"

        self.logger = logging.getLogger("GeologyTab")

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

        # Initialer Dependency-Check für die Status-Anzeige
        self.check_input_dependencies()

        self.logger.info("GeologyTab initialized")

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls für Geology-Generation.
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
            self._create_hardness_parameters()
            self._create_deformation_parameters()
            self._create_dependency_status()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_hardness_parameters(self):
        """Erstellt Rock-Hardness Parameter Controls"""
        hardness_group = QGroupBox("Rock Hardness")
        hardness_group.setFont(QFont("Arial", 10, QFont.Bold))
        hardness_layout = QVBoxLayout()

        hardness_params = [
            ("sedimentary_hardness", "Sedimentary Hardness", GEOLOGY.SEDIMENTARY_HARDNESS),
            ("igneous_hardness", "Igneous Hardness", GEOLOGY.IGNEOUS_HARDNESS),
            ("metamorphic_hardness", "Metamorphic Hardness", GEOLOGY.METAMORPHIC_HARDNESS)
        ]

        for param_key, label, config in hardness_params:
            slider = ParameterSlider(
                label=label,
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 1),
                suffix=config.get("suffix", ""),
                description=config.get("description", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            hardness_layout.addWidget(slider)

        hardness_group.setLayout(hardness_layout)
        self.control_panel.layout().addWidget(hardness_group)

    def _create_deformation_parameters(self):
        """Erstellt Tectonic-Deformation Parameter Controls"""
        deformation_group = QGroupBox("Tectonic Deformation")
        deformation_group.setFont(QFont("Arial", 10, QFont.Bold))
        deformation_layout = QVBoxLayout()

        deformation_params = [
            ("ridge_warping", "Ridge Warping", GEOLOGY.RIDGE_WARPING),
            ("bevel_warping", "Bevel Warping", GEOLOGY.BEVEL_WARPING),
            ("metamorph_foliation", "Metamorphic Foliation", GEOLOGY.METAMORPH_FOLIATION),
            ("metamorph_folding", "Metamorphic Folding", GEOLOGY.METAMORPH_FOLDING),
            ("igneous_flowing", "Igneous Flowing", GEOLOGY.IGNEOUS_FLOWING)
        ]

        for param_key, label, config in deformation_params:
            slider = ParameterSlider(
                label=label,
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 0.1),
                suffix=config.get("suffix", ""),
                description=config.get("description", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            deformation_layout.addWidget(slider)

        deformation_group.setLayout(deformation_layout)
        self.control_panel.layout().addWidget(deformation_group)

    def _create_dependency_status(self):
        """Erstellt Dependency-Status-Anzeige (Verfügbarkeit der Terrain-Inputs)"""
        self.dependency_status = StatusIndicator("Input Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit der
        Rock-Distribution-Anzeige (Härte-Vorschau + Mass-Conservation-Status).
        Generation Control entfällt hier bewusst - das übernehmen der globale
        [GENERIEREN]-Button und die Pipeline-Status-Spalte im Shell-Layout.
        """
        self.rock_distribution_widget = RockDistributionWidget()
        layout.addWidget(self.rock_distribution_widget)

    def create_visualization_controls(self):
        """
        Erstellt Geology-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)

        controls_layout.addWidget(self._create_vertical_separator())

        overlay_layout = self._create_overlay_controls()
        controls_layout.addLayout(overlay_layout)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """Erstellt Height/Rock-Types/Hardness Display Mode Controls"""
        layout = QHBoxLayout()

        self.display_mode_group = QButtonGroup()

        height_radio = QRadioButton("Height")
        height_radio.setChecked(True)
        height_radio.toggled.connect(lambda checked: self._on_display_mode_changed("height", checked))
        self.display_mode_group.addButton(height_radio, 0)

        rock_types_radio = QRadioButton("Rock Types")
        rock_types_radio.toggled.connect(lambda checked: self._on_display_mode_changed("rock_map", checked))
        self.display_mode_group.addButton(rock_types_radio, 1)

        hardness_radio = QRadioButton("Hardness")
        hardness_radio.toggled.connect(lambda checked: self._on_display_mode_changed("hardness_map", checked))
        self.display_mode_group.addButton(hardness_radio, 2)

        layout.addWidget(height_radio)
        layout.addWidget(rock_types_radio)
        layout.addWidget(hardness_radio)

        return layout

    def _create_overlay_controls(self):
        """Erstellt 3D-Terrain-Overlay-Toggle"""
        layout = QHBoxLayout()

        self.terrain_3d_checkbox = QCheckBox("3D Terrain")
        self.terrain_3d_checkbox.toggled.connect(self.update_display_mode)
        layout.addWidget(self.terrain_3d_checkbox)

        return layout

    def _create_vertical_separator(self):
        """Erstellt vertikalen Separator für UI-Layout"""
        separator = QWidget()
        separator.setFixedWidth(1)
        separator.setStyleSheet("background-color: #bdc3c7;")
        return separator

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            if self.rock_distribution_widget:
                self.rock_distribution_widget.update_distribution(self.get_current_parameters())

            self.logger.debug(f"Parameter changed: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

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
        Überschreibt BaseMapTab Display-Update für Geology-spezifische Modi.
        Implementiert Height/Rock-Types/Hardness Display-Switching sowie den
        optionalen 3D-Terrain-Overlay.
        """
        try:
            if not self.data_lod_manager:
                return

            current_display = self.get_current_display()
            if not current_display:
                return

            if self.current_display_mode == "height":
                # Kombiniert, nicht die unbearbeitete Terrain-Rohausgabe - siehe
                # DataLODManager.get_terrain_data_combined()
                data = self.data_lod_manager.get_terrain_data_combined("heightmap")
                data_type = "heightmap"
            elif self.current_display_mode == "rock_map":
                data = self.data_lod_manager.get_geology_data("rock_map")
                data_type = "rock_map"
            elif self.current_display_mode == "hardness_map":
                data = self.data_lod_manager.get_geology_data("hardness_map")
                data_type = "hardness_map"
            else:
                return

            if data is not None and hasattr(current_display, 'update_display'):
                display_id = f"GeologyTab_{self.current_view}_{data_type}"

                if hasattr(self.data_lod_manager, 'display_update_manager'):
                    needs_update = self.data_lod_manager.display_update_manager.needs_update(
                        display_id, data, data_type
                    )

                    if needs_update:
                        self._push_data_to_current_display(data, data_type)
                        self.data_lod_manager.display_update_manager.mark_updated(
                            display_id, data, data_type
                        )
                else:
                    self._push_data_to_current_display(data, data_type)

            # 3D Terrain Overlay wenn aktiviert
            if self.terrain_3d_checkbox and self.terrain_3d_checkbox.isChecked():
                heightmap = self.data_lod_manager.get_terrain_data_combined("heightmap")
                if heightmap is not None and hasattr(current_display.display, 'overlay_3d_terrain'):
                    current_display.display.overlay_3d_terrain(heightmap)

        except Exception as e:
            self.logger.debug(f"Geology display mode update failed: {e}")

    # =============================================================================
    # GENERATION
    # =============================================================================

    def generate(self):
        """
        Überschreibt BaseMapTab: Dependency-Check vor der eigentlichen Generation.
        Wird ausschließlich über den globalen [GENERIEREN]-Button im Shell-Footer
        ausgelöst (kein eigener Berechnen-Button mehr im Parameter-Panel).
        """
        try:
            if not self.check_input_dependencies():
                self.logger.warning("Input dependencies (heightmap/slopemap) not met")
                return

            super().generate()

            self.logger.info("Geology generation requested")

        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """
        Überschreibt BaseMapTab Completion Handler für Geology-spezifische Completion.
        Aktualisiert Rock-Distribution-Statistics nach erfolgreicher Generation.
        """
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type != self.generator_type:
            return

        try:
            if success:
                self.update_display_mode()

                rock_map = self.data_lod_manager.get_geology_data("rock_map")
                hardness_map = self.data_lod_manager.get_geology_data("hardness_map")

                if rock_map is not None and hardness_map is not None and self.rock_distribution_widget:
                    self.rock_distribution_widget.update_statistics(rock_map, hardness_map)

            super().on_generation_completed(result_id, result_data)

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER SYNCHRONISATION
    # =============================================================================

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Werte aller Geology-Parameter-Slider.
        Wird vom ParameterManager als zentrale Quelle für die Geology-Parameter
        genutzt (register_tab()/get_tab_parameters() rufen diese Methode auf,
        siehe gui/OldManagers/parameter_manager.py).
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    def update_parameter_ui(self, param_name: str, value):
        """
        Überschreibt BaseMapTab Parameter-UI Update für Geology-Parameter.
        Synchronisiert UI-Controls mit ParameterManager-Updates.
        """
        try:
            if param_name in self.parameter_sliders:
                slider = self.parameter_sliders[param_name]
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)

                self.logger.debug(f"Parameter UI updated: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter UI update failed: {e}")

    # =============================================================================
    # DEPENDENCY SYSTEM
    # =============================================================================

    def check_input_dependencies(self) -> bool:
        """
        Überschreibt BaseMapTab Dependency Check.
        Prüft ob die Terrain-Inputs (heightmap, slopemap) verfügbar sind.
        """
        try:
            heightmap = self.data_lod_manager.get_terrain_data("heightmap")
            slopemap = self.data_lod_manager.get_terrain_data("slopemap")

            dependencies_met = heightmap is not None and slopemap is not None

            if self.dependency_status:
                if dependencies_met:
                    self.dependency_status.set_success("Terrain inputs available")
                else:
                    missing = []
                    if heightmap is None:
                        missing.append("heightmap")
                    if slopemap is None:
                        missing.append("slopemap")
                    self.dependency_status.set_warning(f"Missing terrain data: {', '.join(missing)}")

            return dependencies_met

        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False

    # =============================================================================
    # RESOURCE MANAGEMENT
    # =============================================================================

    def cleanup_resources(self):
        """
        Erweitert BaseMapTab Cleanup für Geology-spezifische Resources.
        """
        try:
            self.logger.debug("Cleaning up geology-specific resources")

            self.parameter_sliders.clear()

            super().cleanup_resources()

        except Exception as e:
            self.logger.error(f"Geology cleanup failed: {e}")


class RockDistributionWidget(QGroupBox):
    """
    Widget für Rock-Distribution Visualization und Statistics.
    Zeigt eine Härte-Vorschau (live während Parameter-Änderungen) und die
    Verteilungs-/Mass-Conservation-Statistik nach abgeschlossener Generation.
    """

    def __init__(self):
        super().__init__("Rock Distribution")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Rock-Distribution Display"""
        layout = QVBoxLayout()

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

        self.distribution_stats = QLabel("Distribution: Not generated")
        layout.addWidget(self.distribution_stats)

        self.mass_conservation_status = StatusIndicator("Mass Conservation")
        layout.addWidget(self.mass_conservation_status)

        self.setLayout(layout)

    def update_distribution(self, parameters: dict):
        """
        Aktualisiert Hardness Preview basierend auf aktuellen Parametern.
        Parameter: parameters (dict mit hardness values)
        """
        sed_hardness = parameters.get("sedimentary_hardness", 30)
        ign_hardness = parameters.get("igneous_hardness", 80)
        met_hardness = parameters.get("metamorphic_hardness", 65)

        self.sedimentary_bar.setValue(int(sed_hardness))
        self.sedimentary_label.setText(f"Sedimentary: {sed_hardness}")

        self.igneous_bar.setValue(int(ign_hardness))
        self.igneous_label.setText(f"Igneous: {ign_hardness}")

        self.metamorphic_bar.setValue(int(met_hardness))
        self.metamorphic_label.setText(f"Metamorphic: {met_hardness}")

    def update_statistics(self, rock_map: np.ndarray, hardness_map: np.ndarray):
        """
        Aktualisiert Statistiken nach abgeschlossener Generation.
        Parameter: rock_map (RGB array), hardness_map (2D array)
        """
        try:
            total_pixels = rock_map.shape[0] * rock_map.shape[1]

            sedimentary_pct = np.sum(rock_map[:, :, 0]) / (total_pixels * 255) * 100
            igneous_pct = np.sum(rock_map[:, :, 1]) / (total_pixels * 255) * 100
            metamorphic_pct = np.sum(rock_map[:, :, 2]) / (total_pixels * 255) * 100

            mass_sums = np.sum(rock_map, axis=2)
            mass_conservation_valid = np.allclose(mass_sums, 255, atol=5)

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
