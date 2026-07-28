"""
Path: gui/tabs/erosion_tab.py

ErosionTab ist die UI des Erosion-Generators (core/erosion_generator.py) - des
hydraulischen FELDVERFAHRENS nach dem Vorbild LanLou123/Webgl-Erosion.

Der Tab sitzt zwischen Geology und Weather, weil genau dort der Generator in
der Pipeline steht: erst wird das Gelände geformt, danach rechnen Wetter,
Wasser, Biome und Siedlungen auf dem fertigen Relief. Bis 2026-07-28 lag die
Erosion im Water-Tab und damit hinter Weather - das Wetter sah die Täler nie.

Er liefert erosion_map, sedimentation_map, thermal_erosion_map,
thermal_deposition_map (die vier gelände­formenden Karten, die
DataLODManager.get_calculator_combined_heightmap() verrechnet) sowie
sediment_load_map, water_depth_map und flow_velocity_map als Diagnosefelder.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QLabel
)
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from gui.config.value_default import get_parameter_config


class ErosionTab(BaseMapTab):
    """
    Erosion-Generator Tab mit vollständiger BaseMapTab-Integration.

    Die Anzeigemodi sind bewusst mehr als die reinen Ergebniskarten: gerade
    beim Kalibrieren eines Erosionsmodells sagt der Zwischenzustand
    (Sedimentfracht, Wasserstand, Fließgeschwindigkeit) mehr über das Verhalten
    aus als das Endergebnis - und "Net Change" ist die Karte, an der eine
    unplausible Spitze sofort auffällt.
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager,
                 shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "erosion"
        # Genau die zwei Karten, die der Generator liest (siehe
        # ErosionSystemGenerator._calc_hydraulic()). Bewusst KEINE Weather-
        # Abhängigkeit: die Erosion modelliert geologische Zeit und läuft vor
        # dem Wetter.
        self.required_dependencies = ["heightmap", "hardness_map"]

        self.parameter_sliders = {}
        self.erosion_stats = None
        self.dependency_status = None
        self.gpu_status = None
        self.display_mode_group = None
        self.current_display_mode = "height"
        self._display_modes_by_id = {}

        self.logger = logging.getLogger("ErosionTab")

        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )

        if self.parameter_manager:
            self.parameter_manager.register_tab(self.generator_type, self)

        self.check_input_dependencies()
        self.logger.info("ErosionTab initialized")

    # =============================================================================
    # PARAMETER-CONTROLS
    # =============================================================================

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls. Implementiert Required-Method von
        BaseMapTab. Stellt bei abgetrenntem Panel-Layout ein neues Layout
        wieder her, damit die Parameter-Erstellung nie leer abbricht (gleiches
        Muster wie WaterTab).
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
            # Gruppiert nach dem, was der Nutzer beim Kalibrieren zusammen
            # anfasst - nicht nach der internen Struktur des Simulators.
            self._create_parameter_group(
                "Fluvial Erosion",
                ["erosion_capacity", "erosion_strength", "deposition_rate"])
            self._create_parameter_group(
                "Water Cycle", ["rainfall", "evaporation_rate"])
            self._create_parameter_group(
                "Slope Collapse",
                ["thermal_strength", "talus_angle_scale", "thermal_variant"])
            self._create_parameter_group(
                "Geology Coupling", ["hardness_influence"])
            self._create_parameter_group(
                "Simulation",
                ["simulation_resolution", "convergence_threshold", "max_steps", "smoothing"])
            self._create_dependency_status()
            self._create_gpu_status()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_parameter_group(self, title: str, param_keys):
        """Erstellt eine Parameter-GroupBox für eine Teilmenge der Erosion-Parameter."""
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout = QVBoxLayout()

        for param_key in param_keys:
            config = get_parameter_config("erosion", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", 0.01),
                suffix=config.get("suffix", ""),
                description=config.get("description", "")
            )
            slider.valueChanged.connect(
                lambda value, key=param_key: self._on_parameter_changed(key, value)
            )
            self.parameter_sliders[param_key] = slider
            layout.addWidget(slider)

        group.setLayout(layout)
        self.control_panel.layout().addWidget(group)

    def _create_dependency_status(self):
        """Verfügbarkeit von Terrain-Heightmap und Geology-Härte."""
        self.dependency_status = StatusIndicator("Erosion Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def _create_gpu_status(self):
        """
        GPU-Status. Für dieses Modell ist die Anzeige mehr als Kosmetik: ohne
        GPU begrenzt der Generator die Simulationsauflösung auf 256 px (siehe
        HydraulicFieldSimulator.MAX_CPU_RESOLUTION), weil mehrere tausend
        Schritte auf 512² auf der CPU Stunden dauern würden. Die Statistik
        zeigt anschließend, auf welcher Auflösung tatsächlich gerechnet wurde.
        """
        self.gpu_status = StatusIndicator("GPU Status")
        if self.shader_manager and getattr(self.shader_manager, 'gpu_available', False):
            self.gpu_status.set_success("GPU available")
        else:
            self.gpu_status.set_warning("CPU only - Auflösung wird auf 256 begrenzt")
        self.control_panel.layout().addWidget(self.gpu_status)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """Befüllt das Statistics-Tab (Spalte 3)."""
        self.erosion_stats = ErosionStatisticsWidget()
        layout.addWidget(self.erosion_stats)

    # =============================================================================
    # ANZEIGEMODI
    # =============================================================================

    def create_visualization_controls(self):
        """Erstellt die Radio-Buttons über dem Canvas."""
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addLayout(self._create_display_mode_controls())
        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """
        Neun Anzeigemodi. Zwei davon gibt es sonst nirgends im Programm:

        * "Net Change" - die signierte Höhenänderung mit divergierender
          Farbskala um 0. Beim Partikelverfahren wurde einmal ein
          Sedimentations-Peak gemeldet, der sich als Durchsatz-Hotspot ohne
          jede Geländeauffälligkeit entpuppte; auf dieser Karte wäre das in
          einem Blick zu sehen gewesen.
        * "Sediment Load" - die noch im Wasser gelöste Fracht, also der
          Zustand, der das Feldverfahren überhaupt vom Partikelverfahren
          unterscheidet. Im Vorbild sind das die cyanfarbenen Frachtspuren.
        """
        layout = QHBoxLayout()
        self.display_mode_group = QButtonGroup()

        modes = [
            ("height", "Height"),
            ("erosion_map", "Erosion"),
            ("sedimentation_map", "Sedimentation"),
            ("net_change_map", "Net Change"),
            ("sediment_load_map", "Sediment Load"),
            ("water_depth_map", "Water Depth"),
            ("flow_velocity_map", "Flow Velocity"),
            ("thermal_erosion_map", "Thermal Erosion"),
            ("thermal_deposition_map", "Thermal Deposition"),
        ]

        for button_id, (mode_key, label) in enumerate(modes):
            radio = QRadioButton(label)
            if button_id == 0:
                radio.setChecked(True)
            self.display_mode_group.addButton(radio, button_id)
            layout.addWidget(radio)

        self._display_modes_by_id = {button_id: mode_key
                                     for button_id, (mode_key, _) in enumerate(modes)}
        # idClicked statt toggled: toggled feuert beim Umschalten zweimal
        # (einmal mit checked=False für das abgewählte Radio).
        self.display_mode_group.idClicked.connect(self._on_display_mode_selected)
        return layout

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        try:
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)
            if self.erosion_stats:
                self.erosion_stats.update_parameter_preview(self.get_current_parameters())
            self.logger.debug(f"Parameter changed: {param_name} = {value}")
        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

    def _on_display_mode_selected(self, button_id: int):
        mode = self._display_modes_by_id.get(button_id)
        if mode is None:
            return
        self.current_display_mode = mode
        self.update_display_mode()
        self.logger.debug(f"Display mode changed to: {mode}")

    # =============================================================================
    # DISPLAY
    # =============================================================================

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab. "Net Change" hat keinen eigenen Storage-Key -
        es ist die Differenz der beiden Ergebniskarten und wird hier gebildet,
        statt ein weiteres Array durch die ganze Pipeline zu schleusen, das
        dieselbe Information ein drittes Mal enthielte.
        """
        try:
            if not self.data_lod_manager:
                return
            current_display = self.get_current_display()
            if not current_display:
                return

            if self.current_display_mode == "height":
                data = self.data_lod_manager.get_terrain_data_combined("heightmap")
                data_type = "heightmap_combined"
            elif self.current_display_mode == "net_change_map":
                erosion = self.data_lod_manager.get_erosion_data("erosion_map")
                sedimentation = self.data_lod_manager.get_erosion_data("sedimentation_map")
                data = None
                if erosion is not None and sedimentation is not None:
                    data = (sedimentation.astype(np.float32) - erosion.astype(np.float32))
                data_type = "net_change_map"
            else:
                data = self.data_lod_manager.get_erosion_data(self.current_display_mode)
                data_type = self.current_display_mode

            if data is not None and hasattr(current_display, 'update_display'):
                display_id = f"ErosionTab_{self.current_view}_{data_type}"
                if hasattr(self.data_lod_manager, 'display_update_manager'):
                    manager = self.data_lod_manager.display_update_manager
                    if manager.needs_update(display_id, data, data_type):
                        self._push_data_to_current_display(data, data_type)
                        manager.mark_updated(display_id, data, data_type)
                else:
                    self._push_data_to_current_display(data, data_type)

        except Exception as e:
            self.logger.debug(f"Erosion display mode update failed: {e}")

    # =============================================================================
    # GENERATION
    # =============================================================================

    def generate(self):
        """Überschreibt BaseMapTab: Dependency-Check vor der Generation."""
        try:
            if not self.check_input_dependencies():
                self.logger.warning("Input dependencies (terrain/geology) not met")
                return
            super().generate()
            self.logger.info("Erosion generation requested")
        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """Aktualisiert Anzeige und Statistik nach abgeschlossener Generation."""
        try:
            super().on_generation_completed(result_id, result_data)
            if self.erosion_stats:
                self.erosion_stats.update_generation_statistics(self.data_lod_manager)
            self.update_display_mode()
        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER-SCHNITTSTELLE
    # =============================================================================

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Slider-Werte. Zentrale Quelle für den
        ParameterManager (register_tab()/get_tab_parameters()).

        `thermal_variant` ist als Slider 0/1 modelliert (der vorhandene
        ParameterSlider kann keine Auswahl darstellen) und wird hier in den
        String übersetzt, den der Simulator erwartet - damit bleibt die
        Bedeutung im Generator lesbar statt als magische Zahl.
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        parameters["thermal_variant"] = (
            "flux" if parameters.get("thermal_variant", 0) >= 0.5 else "gather")
        return parameters

    def update_parameter_ui(self, param_name: str, value):
        """Synchronisiert UI-Controls mit ParameterManager-Updates."""
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
    # DEPENDENCIES
    # =============================================================================

    def check_input_dependencies(self) -> bool:
        """Prüft Terrain-Heightmap und Geology-Härte."""
        try:
            values = {
                "heightmap": self.data_lod_manager.get_terrain_data("heightmap"),
                "hardness_map": self.data_lod_manager.get_geology_data("hardness_map"),
            }
            missing = [key for key, value in values.items() if value is None]
            dependencies_met = not missing

            if self.dependency_status:
                if dependencies_met:
                    self.dependency_status.set_success("All dependencies available")
                else:
                    self.dependency_status.set_warning(f"Missing: {', '.join(missing)}")
            return dependencies_met
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return False


class ErosionStatisticsWidget(QWidget):
    """
    Zeigt Erosion-Parameter und die Kennzahlen des letzten Laufs.

    Die Kennzahlen sind bewusst die, an denen man ein Erosionsmodell beurteilt
    und nicht die, die am eindrucksvollsten klingen: hat der Lauf konvergiert,
    wie weit hat sich das Gelände tatsächlich verformt, und geht die
    Massenbilanz auf. Die tatsächliche Simulationsauflösung steht dabei, weil
    sie ohne GPU unter dem eingestellten Wert liegt - eine stille Abweichung
    wäre hier besonders irreführend.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("ErosionStatisticsWidget")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        parameter_group = QGroupBox("Erosion Parameters")
        parameter_layout = QVBoxLayout()
        self.capacity_label = QLabel("Sediment Capacity: -")
        self.strength_label = QLabel("Erosion Strength: -")
        self.deposition_label = QLabel("Deposition Rate: -")
        self.rainfall_label = QLabel("Rainfall: -")
        self.hardness_label = QLabel("Hardness Influence: -")
        for widget in (self.capacity_label, self.strength_label, self.deposition_label,
                       self.rainfall_label, self.hardness_label):
            parameter_layout.addWidget(widget)
        parameter_group.setLayout(parameter_layout)
        layout.addWidget(parameter_group)

        result_group = QGroupBox("Simulation Results")
        result_layout = QVBoxLayout()
        self.resolution_label = QLabel("Simulation Resolution: -")
        self.steps_label = QLabel("Steps: -")
        self.balance_label = QLabel("Mass Balance: -")
        self.change_label = QLabel("Terrain Change: -")
        self.load_label = QLabel("Sediment in Transit: -")
        for widget in (self.resolution_label, self.steps_label, self.balance_label,
                       self.change_label, self.load_label):
            result_layout.addWidget(widget)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        layout.addStretch()
        self.setLayout(layout)

    def update_parameter_preview(self, parameters: Dict[str, Any]):
        try:
            def show(label, text, key, fmt="{:.2f}"):
                value = parameters.get(key)
                label.setText(f"{text}: -" if value is None
                              else f"{text}: " + fmt.format(value))

            show(self.capacity_label, "Sediment Capacity", "erosion_capacity")
            show(self.strength_label, "Erosion Strength", "erosion_strength")
            show(self.deposition_label, "Deposition Rate", "deposition_rate")
            show(self.rainfall_label, "Rainfall", "rainfall")
            influence = parameters.get("hardness_influence")
            self.hardness_label.setText(
                "Hardness Influence: -" if influence is None
                else f"Hardness Influence: {influence * 100:.0f}%")
        except Exception as e:
            self.logger.debug(f"Parameter preview update failed: {e}")

    def update_generation_statistics(self, data_lod_manager):
        """
        Liest die Kennzahlen direkt aus dem Erosion-Storage. `converged=False`
        wird ausdrücklich als solches angezeigt statt verschwiegen - ein Lauf,
        der in die Schrittgrenze gelaufen ist, hat ein anderes Ergebnis als
        einer, der ausgelaufen ist, und der Nutzer soll das sehen.
        """
        try:
            if data_lod_manager is None:
                return

            resolution = data_lod_manager.get_erosion_data("simulation_resolution")
            steps = data_lod_manager.get_erosion_data("steps_taken")
            converged = data_lod_manager.get_erosion_data("converged")
            balance = data_lod_manager.get_erosion_data("mass_balance")

            if resolution:
                self.resolution_label.setText(
                    f"Simulation Resolution: {int(resolution)}x{int(resolution)}")
            if steps is not None:
                note = "konvergiert" if converged else "Schrittgrenze erreicht"
                self.steps_label.setText(f"Steps: {int(steps)} ({note})")
            if balance is not None:
                self.balance_label.setText(
                    f"Mass Balance: {float(balance) * 100:+.2f}% "
                    f"(Rest verlässt die Karte über den Rand)")

            erosion = data_lod_manager.get_erosion_data("erosion_map")
            sedimentation = data_lod_manager.get_erosion_data("sedimentation_map")
            if erosion is not None and sedimentation is not None:
                net = sedimentation.astype(np.float64) - erosion
                self.change_label.setText(
                    f"Terrain Change: {float(net.min()):+,.0f} … {float(net.max()):+,.0f} m")

            load = data_lod_manager.get_erosion_data("sediment_load_map")
            if load is not None:
                self.load_label.setText(
                    f"Sediment in Transit: max {float(load.max()):.2f} m, "
                    f"Mittel {float(load.mean()):.3f} m")

        except Exception as e:
            self.logger.debug(f"Generation statistics update failed: {e}")
