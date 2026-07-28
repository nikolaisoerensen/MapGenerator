"""
Path: gui/tabs/water_tab.py

WaterTab implementiert die Water-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den HydrologySystemGenerator aus core/water_generator.py. Als von
Terrain, Geology und Weather abhängiger Generator liefert er water_map, flow_map, flow_speed,
cross_section, soil_moist_map, erosion_map, sedimentation_map, evaporation_map, ocean_outflow
und water_biomes_map für Biome und Settlement.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QButtonGroup, QCheckBox, QLabel
)
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from gui.config.value_default import get_parameter_config


class WaterTab(BaseMapTab):
    """
    Water-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert das komplette Hydrologie-System (Flow-Netzwerk, Erosion/Sedimentation,
    Bodenfeuchtigkeit, Verdunstung) auf Basis der Terrain-, Geology- und Weather-Daten.
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):

        # Generator-Konfiguration vor BaseMapTab.__init__()
        self.generator_type = "water"
        # Genau die Karten, die Water tatsächlich liest (siehe
        # HydrologySystemGenerator._get_prepared_water_inputs()). slopemap und
        # rock_map sind 2026-07-27 entfernt: beide wurden verlangt, aber von
        # keiner Water-Berechnung je benutzt - der Tab blockierte die
        # Generierung dadurch, bis Geology seine rock_map geliefert hatte,
        # obwohl Water sie nicht braucht.
        self.required_dependencies = ["heightmap", "hardness_map",
                                      "precip_map", "temp_map", "wind_map", "humid_map"]

        # Water-spezifische Attribute (vor super(), da create_parameter_controls
        # und create_visualization_controls während BaseMapTab.setup_ui() darauf
        # zugreifen und sie befüllen)
        self.parameter_sliders = {}
        self.hydrology_stats = None
        self.dependency_status = None
        self.gpu_status = None
        self.display_mode_group = None
        self.current_display_mode = "height"

        self.logger = logging.getLogger("WaterTab")

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

        self.logger.info("WaterTab initialized")

    def create_parameter_controls(self):
        """
        Erstellt alle Parameter-Controls für Water-Generation.
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
            # rain_threshold und manning_coefficient sind 2026-07-27 entfernt -
            # beide hatten seit dem Pipe-Modell-Umbau keinen Effekt mehr auf
            # das Ergebnis (siehe gui/config/value_default.py WATER für die
            # volle Begründung).
            self._create_parameter_group("Lake Detection", ["lake_volume_threshold"], 100.0)
            self._create_parameter_group(
                "Flow Dynamics", ["river_abundance", "diffusion_radius"], 0.001)
            # === ALTBESTAND (stillgelegt 2026-07-28) ===
            # Die Erosions-Regler sind in den Erosion-Tab gewandert
            # (gui/tabs/erosion_tab.py). Sie haetten hier keinen Effekt mehr,
            # weil Water kein Gelaende mehr formt.
            # self._create_parameter_group(
            #     "Erosion & Sedimentation",
            #     ["erosion_strength", "erosion_passes", "sediment_capacity_factor",
            #      "settling_velocity", "thermal_erosion_strength"], 0.01)
            self._create_parameter_group("Evaporation", ["evaporation_base_rate"], 0.0001)
            self._create_dependency_status()
            self._create_gpu_status()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _create_parameter_group(self, title: str, param_keys, default_step: float):
        """Erstellt eine Parameter-GroupBox für eine Teilmenge der Water-Parameter"""
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout = QVBoxLayout()

        for param_key in param_keys:
            config = get_parameter_config("water", param_key)

            slider = ParameterSlider(
                label=param_key.replace("_", " ").title(),
                min_val=config["min"],
                max_val=config["max"],
                default_val=config["default"],
                step=config.get("step", default_step),
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
        """Erstellt Dependency-Status-Anzeige (Verfügbarkeit der Terrain/Geology/Weather-Inputs)"""
        self.dependency_status = StatusIndicator("Water Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def _create_gpu_status(self):
        """
        Erstellt GPU-Status-Anzeige. Die Hydrologie-Simulation läuft aktuell durchgehend
        über den CPU-Fallback (ShaderManager bietet keine Water-spezifischen
        GPU-Compute-Shader an) - die Anzeige spiegelt das ehrlich wider.
        """
        self.gpu_status = StatusIndicator("GPU Status")
        if self.shader_manager and getattr(self.shader_manager, 'gpu_available', False):
            self.gpu_status.set_success("GPU available")
        else:
            self.gpu_status.set_warning("Using CPU fallback")
        self.control_panel.layout().addWidget(self.gpu_status)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit den
        Hydrology-Statistics (Parameter-Preview + Generation-Results).
        """
        self.hydrology_stats = HydrologyStatisticsWidget()
        layout.addWidget(self.hydrology_stats)

    def create_visualization_controls(self):
        """
        Erstellt Water-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab.
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """Erstellt Height/Water-Outputs Display Mode Controls"""
        layout = QHBoxLayout()

        self.display_mode_group = QButtonGroup()

        # Thermal Erosion/Deposition ergänzt (2026-07-27): beide formen das
        # Gelände sichtbar um und haben einen eigenen Slider, waren aber in
        # keinem Anzeigemodus zu sehen - der Effekt des Sliders liess sich
        # dadurch nur indirekt über die kombinierte Heightmap beurteilen.
        modes = [
            ("height", "Height"),
            ("water_map", "Water Depth"),
            ("flow_map", "Flow"),
            # === ALTBESTAND (stillgelegt 2026-07-28) ===
            # Erosion, Sedimentation und die beiden Thermal-Karten haben einen
            # eigenen Tab (gui/tabs/erosion_tab.py). Hier wuerden sie nur noch
            # leere Karten zeigen, weil WaterData sie nicht mehr fuehrt.
            # ("erosion_map", "Erosion"),
            # ("sedimentation_map", "Sedimentation"),
            # ("thermal_erosion_map", "Thermal Erosion"),
            # ("thermal_deposition_map", "Thermal Deposition"),
            ("soil_moist_map", "Soil Moisture"),
            ("evaporation_map", "Evaporation"),
        ]

        for button_id, (mode_key, label) in enumerate(modes):
            radio = QRadioButton(label)
            if button_id == 0:
                radio.setChecked(True)
            # buttonClicked statt toggled: toggled feuert beim Umschalten
            # ZWEIMAL (einmal mit checked=False für das abgewählte Radio).
            self.display_mode_group.addButton(radio, button_id)
            layout.addWidget(radio)

        self._display_modes_by_id = {button_id: mode_key
                                      for button_id, (mode_key, _) in enumerate(modes)}
        self.display_mode_group.idClicked.connect(self._on_display_mode_selected)

        return layout

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            if self.hydrology_stats:
                self.hydrology_stats.update_parameter_preview(self.get_current_parameters())

            self.logger.debug(f"Parameter changed: {param_name} = {value}")

        except Exception as e:
            self.logger.error(f"Parameter change handling failed: {e}")

    def _on_display_mode_selected(self, button_id: int):
        """Handler für Display-Mode-Wechsel (an QButtonGroup.idClicked, nicht
        an QRadioButton.toggled - letzteres feuert beim Umschalten zweimal)."""
        mode = self._display_modes_by_id.get(button_id)
        if mode is None:
            return
        self.current_display_mode = mode
        self.update_display_mode()
        self.logger.debug(f"Display mode changed to: {mode}")

    # =============================================================================
    # DISPLAY UPDATE SYSTEM
    # =============================================================================

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab Display-Update für Water-spezifische Modi.
        Implementiert Height/Water-Depth/Flow/Erosion/Sedimentation/Soil-Moisture
        Display-Switching sowie River-Network- und 3D-Terrain-Overlays.
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
                data_type = "heightmap_combined"
            else:
                data = self.data_lod_manager.get_water_data(self.current_display_mode)
                data_type = self.current_display_mode

            # See/Fluss-Farbunterscheidung (siehe [[project-water-flood-calibration]]):
            # water_biomes_map als Referenz hinterlegen, BEVOR water_map gerendert wird -
            # analog zu set_contour_reference_heightmap weiter unten in _push_data_to_
            # current_display(). Nur für den water_map-Modus relevant, aber billig genug,
            # um es einfach immer mitzuschicken (_render_water_map/_render_overlay
            # ignorieren es sonst). An BEIDE Displays pushen (nicht nur current_display) -
            # gleiches Muster wie die Shademap in base_tab.py: self.map_display_3d
            # existiert immer, auch wenn gerade 2D sichtbar ist, damit beim Umschalten
            # zu 3D die See-Färbung sofort stimmt statt erst nach erneutem Radio-
            # Button-Klick (User-Report: Seen in 3D nicht sichtbar).
            water_biomes_map = self.data_lod_manager.get_water_data("water_biomes_map")
            if hasattr(current_display.display, 'set_water_biomes_reference'):
                current_display.display.set_water_biomes_reference(water_biomes_map)
            if (self.map_display_3d and self.map_display_3d is not current_display
                    and hasattr(self.map_display_3d.display, 'set_water_biomes_reference')):
                self.map_display_3d.display.set_water_biomes_reference(water_biomes_map)

            if data is not None and hasattr(current_display, 'update_display'):
                display_id = f"WaterTab_{self.current_view}_{data_type}"

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

        except Exception as e:
            self.logger.debug(f"Water display mode update failed: {e}")

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
                self.logger.warning("Input dependencies (terrain/geology/weather) not met")
                return

            super().generate()

            self.logger.info("Water generation requested")

        except Exception as e:
            self.logger.error(f"Generation request failed: {e}")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """
        Überschreibt BaseMapTab Completion Handler für Water-spezifische Completion.
        Aktualisiert Hydrology-Statistics nach erfolgreicher Generation.
        """
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type != self.generator_type:
            return

        try:
            if success:
                self.update_display_mode()

                if self.hydrology_stats:
                    results = {
                        "water_map": self.data_lod_manager.get_water_data("water_map"),
                        "flow_speed": self.data_lod_manager.get_water_data("flow_speed"),
                        "erosion_map": self.data_lod_manager.get_water_data("erosion_map"),
                        "sedimentation_map": self.data_lod_manager.get_water_data("sedimentation_map"),
                        "thermal_erosion_map": self.data_lod_manager.get_water_data("thermal_erosion_map"),
                        "ocean_outflow": self.data_lod_manager.get_water_data("ocean_outflow"),
                    }
                    if any(value is not None for value in results.values()):
                        self.hydrology_stats.update_generation_statistics(results)

            super().on_generation_completed(result_id, result_data)

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {e}")

    # =============================================================================
    # PARAMETER SYNCHRONISATION
    # =============================================================================

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Sammelt die aktuellen Werte aller Water-Parameter-Slider.
        Wird vom ParameterManager als zentrale Quelle für die Water-Parameter
        genutzt (register_tab()/get_tab_parameters() rufen diese Methode auf,
        siehe gui/OldManagers/parameter_manager.py).
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()
        return parameters

    def update_parameter_ui(self, param_name: str, value):
        """
        Überschreibt BaseMapTab Parameter-UI Update für Water-Parameter.
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
        Prüft ob alle Terrain-/Geology-/Weather-Inputs verfügbar sind.
        """
        try:
            values = {
                "heightmap": self.data_lod_manager.get_terrain_data("heightmap"),
                "hardness_map": self.data_lod_manager.get_geology_data("hardness_map"),
                "precip_map": self.data_lod_manager.get_weather_data("precip_map"),
                "temp_map": self.data_lod_manager.get_weather_data("temp_map"),
                "wind_map": self.data_lod_manager.get_weather_data("wind_map"),
                "humid_map": self.data_lod_manager.get_weather_data("humid_map"),
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

    # =============================================================================
    # RESOURCE MANAGEMENT
    # =============================================================================

    def cleanup_resources(self):
        """
        Erweitert BaseMapTab Cleanup für Water-spezifische Resources.
        """
        try:
            self.logger.debug("Cleaning up water-specific resources")

            self.parameter_sliders.clear()

            super().cleanup_resources()

        except Exception as e:
            self.logger.error(f"Water cleanup failed: {e}")


class HydrologyStatisticsWidget(QGroupBox):
    """
    Widget für Hydrologie-Statistiken und Parameter-Preview.
    Zeigt Water-Parameter und Generation-Results (Water Coverage, Flow Speed, Erosion).
    """

    def __init__(self):
        super().__init__("Hydrology Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Hydrologie-Statistiken"""
        layout = QVBoxLayout()

        preview_group = QGroupBox("Parameter Preview")
        preview_layout = QVBoxLayout()

        self.lake_threshold_label = QLabel("Lake Threshold: -")
        self.erosion_strength_label = QLabel("Erosion Strength: -")
        self.thermal_strength_label = QLabel("Thermal Erosion Strength: -")
        self.river_abundance_label = QLabel("River Abundance: -")

        preview_layout.addWidget(self.lake_threshold_label)
        preview_layout.addWidget(self.erosion_strength_label)
        preview_layout.addWidget(self.thermal_strength_label)
        preview_layout.addWidget(self.river_abundance_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        stats_group = QGroupBox("Generation Results")
        stats_layout = QVBoxLayout()

        self.water_coverage_label = QLabel("Water Coverage: -")
        self.avg_flow_speed_label = QLabel("Avg Flow Speed: -")
        self.total_erosion_label = QLabel("Total Erosion: -")
        self.net_terrain_change_label = QLabel("Net Terrain Change: -")
        self.ocean_outflow_label = QLabel("Ocean Outflow: -")

        stats_layout.addWidget(self.water_coverage_label)
        stats_layout.addWidget(self.avg_flow_speed_label)
        stats_layout.addWidget(self.total_erosion_label)
        stats_layout.addWidget(self.net_terrain_change_label)
        stats_layout.addWidget(self.ocean_outflow_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        self.setLayout(layout)

    def update_parameter_preview(self, parameters: dict):
        """
        Aktualisiert die Parameter-Preview. Zeigt nur Parameter, die das
        Ergebnis tatsächlich beeinflussen - der frühere "Manning Coefficient"
        wurde hier prominent angezeigt, obwohl er seit dem Pipe-Modell-Umbau
        keinerlei Wirkung mehr hatte (siehe gui/config/value_default.py WATER).
        """
        lake_threshold = parameters.get("lake_volume_threshold")
        erosion_strength = parameters.get("erosion_strength")
        thermal_strength = parameters.get("thermal_erosion_strength")
        river_abundance = parameters.get("river_abundance")

        self.lake_threshold_label.setText(
            "Lake Threshold: -" if lake_threshold is None else f"Lake Threshold: {lake_threshold:,.0f} m³")
        self.erosion_strength_label.setText(
            "Erosion Strength: -" if erosion_strength is None else f"Erosion Strength: {erosion_strength:.1f}")
        self.thermal_strength_label.setText(
            "Thermal Erosion Strength: -" if thermal_strength is None
            else f"Thermal Erosion Strength: {thermal_strength:.1f}")
        self.river_abundance_label.setText(
            "River Abundance: -" if river_abundance is None
            else f"River Abundance: {river_abundance * 100:.0f}% der Wasserläufe")

    def update_generation_statistics(self, results: dict):
        """
        Aktualisiert Statistiken nach abgeschlossener Generation.
        Parameter: results (dict mit water_map/flow_speed/erosion_map/
            sedimentation_map/thermal_erosion_map/ocean_outflow)

        Einheiten-Korrektur 2026-07-27: "Total Erosion" wurde als "m/Jahr"
        beschriftet. Die Droplet-Erosion liefert kumulierte METER über die
        LOD-Kette, keine Jahresrate - es gibt im gesamten Modell keinen
        Zeitbezug, gegen den eine Rate definiert wäre.
        """
        water_map = results.get("water_map")
        if water_map is not None:
            water_pixels = np.sum(water_map > 0.01)  # > 1cm Wassertiefe
            total_pixels = water_map.shape[0] * water_map.shape[1]
            water_coverage = (water_pixels / total_pixels) * 100
            self.water_coverage_label.setText(f"Water Coverage: {water_coverage:.1f}%")

        flow_speed = results.get("flow_speed")
        if flow_speed is not None and np.any(flow_speed > 0):
            avg_speed = np.mean(flow_speed[flow_speed > 0])
            self.avg_flow_speed_label.setText(f"Avg Flow Speed: {avg_speed:.2f} m/s")

        erosion_map = results.get("erosion_map")
        thermal_erosion_map = results.get("thermal_erosion_map")
        if erosion_map is not None:
            total_erosion = float(np.sum(erosion_map))
            thermal_note = ""
            if thermal_erosion_map is not None:
                thermal_note = f" (+ {float(np.sum(thermal_erosion_map)):,.0f} m Böschung)"
            self.total_erosion_label.setText(
                f"Erosions-Durchsatz: {total_erosion:,.0f} m{thermal_note}")

        sedimentation_map = results.get("sedimentation_map")
        if erosion_map is not None and sedimentation_map is not None:
            # Netto-Höhenänderung der Karte: was abgetragen und nicht wieder
            # abgelagert wurde, hat die Karte verlassen (Partikel-Restfracht
            # am Kartenrand, siehe DropletErosionSystem "Massenerhaltung").
            #
            # Die beiden Karten sind kumulierter DURCHSATZ, nicht
            # Netto-Änderung: eine Zelle kann dieselbe Fracht über den Lauf
            # hinweg mehrfach abgeben und wieder aufnehmen (siehe
            # DropletErosionSystem.simulate_erosion_sedimentation). Ihre Summen
            # sind deshalb als "Durchsatz" beschriftet, und die Kennzahl, die
            # tatsächlich beschreibt, wie stark sich das Gelände verformt hat,
            # ist die Spanne der Netto-Änderung pro Zelle - genau die Zahl, an
            # der sich eine unplausible Spitze erkennen lässt.
            net_per_cell = np.asarray(sedimentation_map, dtype=np.float64) - erosion_map
            net_change = float(net_per_cell.sum())
            self.net_terrain_change_label.setText(
                f"Net Terrain Change: {net_change:,.0f} m "
                f"(pro Zelle {float(net_per_cell.min()):+,.0f} … "
                f"{float(net_per_cell.max()):+,.0f} m)")

        ocean_outflow = results.get("ocean_outflow")
        if ocean_outflow is not None:
            self.ocean_outflow_label.setText(f"Ocean Outflow: {float(ocean_outflow):,.0f} m³")
