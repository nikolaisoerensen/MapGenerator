"""
Path: gui/tabs/geology_tab.py

GeologyTab implementiert die Geology-Generator UI mit vollständiger BaseMapTab-Integration
und direkter Anbindung an den GeologySystemGenerator aus core/geology_generator.py. Als von
Terrain abhängiger Generator (heightmap, slopemap) liefert er rock_map und hardness_map für
Water und alle nachgelagerten Systeme.

Rework für das 3D-Gesteinsstapel-Modell (siehe core/geology_generator.py und den
zugehörigen Umsetzungsplan): 15 Slider in 6 Gruppen statt der früheren 8 in 2 Gruppen,
neue Diagnose-Anzeigemodi (isolierte Δz-Komponenten) und eine Cross-Section-Ansicht
(vertikaler Schichtstapel-Schnitt entlang X oder Y), damit jeder Tektonik-Slider einzeln
sichtbar/verstehbar wird statt nur im kombinierten Endergebnis.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QRadioButton,
    QButtonGroup, QLabel, QProgressBar
)
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtGui import QFont
import logging
import numpy as np
from typing import Dict, Any

from gui.tabs.base_tab import BaseMapTab
from gui.widgets.widgets import ParameterSlider, StatusIndicator
from gui.config.value_default import GEOLOGY
from core.geology_layers import ALL_ROCK_TYPES, N_LAYERS


class GeologyTab(BaseMapTab):
    """
    Geology-Generator Tab mit vollständiger BaseMapTab-Integration.
    Implementiert rock_map und hardness_map Generation auf Basis der Terrain-Daten
    (heightmap, slopemap).
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
        self.current_display_mode = "height"
        # Cross-Section-Achse/Position: die Bedien-Widgets dafür leben jetzt
        # in der globalen Shell-Zeile neben "Contour Lines" (gui/map_editor.py,
        # UI-Aufräumung Teil 2), GeologyTab hält nur noch den reinen Zustand,
        # den map_editor.py über set_cross_section_axis()/
        # set_cross_section_position() setzt (Muster wie set_contour_overlay()).
        self.current_cross_section_axis = "x"
        self.current_cross_section_position = 0.5

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
            self._create_tilt_parameters()
            self._create_folding_parameters()
            self._create_faulting_parameters()
            self._create_intrusion_parameters()
            self._create_metamorphism_parameters()
            self._create_dependency_status()

            self.logger.debug("Parameter controls created successfully")

        except Exception as e:
            self.logger.error(f"Parameter control creation failed: {e}")

    def _add_slider_group(self, title: str, params: list):
        """Baut eine QGroupBox mit einer Reihe von ParameterSlider-Widgets
        (gemeinsames Muster für alle 6 Slider-Gruppen unten)."""
        group = QGroupBox(title)
        group.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout = QVBoxLayout()

        for param_key, label, config in params:
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
            layout.addWidget(slider)

        group.setLayout(layout)
        self.control_panel.layout().addWidget(group)

    def _create_hardness_parameters(self):
        self._add_slider_group("Rock Hardness", [
            ("sedimentary_hardness", "Sedimentary Hardness", GEOLOGY.SEDIMENTARY_HARDNESS),
            ("igneous_hardness", "Igneous Hardness", GEOLOGY.IGNEOUS_HARDNESS),
            ("metamorphic_hardness", "Metamorphic Hardness", GEOLOGY.METAMORPHIC_HARDNESS),
        ])

    def _create_tilt_parameters(self):
        self._add_slider_group("Tilt", [
            ("tilt_intensity", "Tilt Intensity", GEOLOGY.TILT_INTENSITY),
            ("tilt_direction", "Tilt Direction", GEOLOGY.TILT_DIRECTION),
        ])

    def _create_folding_parameters(self):
        self._add_slider_group("Folding", [
            ("fold_intensity", "Fold Intensity", GEOLOGY.FOLD_INTENSITY),
            ("fold_detail", "Fold Detail", GEOLOGY.FOLD_DETAIL),
        ])

    def _create_faulting_parameters(self):
        self._add_slider_group("Faulting", [
            ("fault_intensity", "Fault Intensity", GEOLOGY.FAULT_INTENSITY),
            ("fault_detail", "Fault Detail", GEOLOGY.FAULT_DETAIL),
            ("fault_edge_softness", "Fault Edge Softness", GEOLOGY.FAULT_EDGE_SOFTNESS),
        ])

    def _create_intrusion_parameters(self):
        self._add_slider_group("Intrusions", [
            ("intrusion_density", "Intrusion Density", GEOLOGY.INTRUSION_DENSITY),
            ("intrusion_size", "Intrusion Size", GEOLOGY.INTRUSION_SIZE),
            ("intrusion_detail", "Intrusion Detail", GEOLOGY.INTRUSION_DETAIL),
        ])

    def _create_metamorphism_parameters(self):
        self._add_slider_group("Metamorphism", [
            ("metamorphic_overprint_intensity", "Metamorphic Overprint Intensity",
             GEOLOGY.METAMORPHIC_OVERPRINT_INTENSITY),
            ("foliation_detail", "Foliation Detail", GEOLOGY.FOLIATION_DETAIL),
        ])

    def _create_dependency_status(self):
        """Erstellt Dependency-Status-Anzeige (Verfügbarkeit der Terrain-Inputs)"""
        self.dependency_status = StatusIndicator("Input Dependencies")
        self.control_panel.layout().addWidget(self.dependency_status)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        Überschreibt BaseMapTab: befüllt das Statistics-Tab (Spalte 3) mit der
        Rock-Distribution-Anzeige (Härte-Vorschau + Formations-Statistik).
        Generation Control entfällt hier bewusst - das übernehmen der globale
        [GENERIEREN]-Button und die Pipeline-Status-Spalte im Shell-Layout.
        """
        self.rock_distribution_widget = RockDistributionWidget()
        layout.addWidget(self.rock_distribution_widget)

    def create_visualization_controls(self):
        """
        Erstellt Geology-spezifische Visualization Controls.
        Überschreibt Optional-Method von BaseMapTab. Die Cross-Section-
        Achsen-/Positions-Regler leben NICHT mehr hier, sondern in der
        globalen Shell-Zeile neben "Contour Lines" (gui/map_editor.py,
        UI-Aufräumung Teil 2) - hier bleibt nur die Display-Mode-Auswahl
        (inkl. des "Cross-Section"-Radios selbst). Der frühere 3D-Terrain-
        Overlay-Checkbox-Toggle wurde entfernt (Nutzer-Wunsch: nie
        funktional, rief eine nie implementierte overlay_3d_terrain()-
        Methode auf MapDisplay3D auf - reine tote UI).
        """
        controls_widget = QWidget()
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        display_mode_layout = self._create_display_mode_controls()
        controls_layout.addLayout(display_mode_layout)

        controls_widget.setLayout(controls_layout)
        return controls_widget

    def _create_display_mode_controls(self):
        """
        Erstellt die Display-Mode-Radios: die drei bisherigen (Height/Rock
        Outcrop/Hardness) plus die Diagnose-Modi für die isolierten
        Verformungs-Komponenten (Terrain Hub/Tilt/Fold/Fault/Intrusion Only)
        sowie Cross-Section - direkte Antwort auf die Kernbeschwerde der
        Geology-Review-Runde ("ich will erkennen können wie sich Ridge
        Warping etc. auswirkt"). 5 Spalten statt 4, damit die 9 Modi in
        2 statt 3 Zeilen passen (Platz durch den Wegfall des 3D-Terrain-
        Toggles freigeworden).
        """
        layout = QGridLayout()
        self.display_mode_group = QButtonGroup()

        modes = [
            ("Height", "height", True),
            ("Rock Outcrop", "rock_map", False),
            ("Hardness", "hardness_map", False),
            ("Terrain Hub Only", "terrain_hub_delta", False),
            ("Tilt Only", "tilt_delta", False),
            ("Fold Only", "fold_delta", False),
            ("Fault Only", "fault_delta", False),
            ("Intrusion Only", "intrusion_delta", False),
            ("Cross-Section", "cross_section", False),
        ]
        for i, (label, mode_key, checked) in enumerate(modes):
            radio = QRadioButton(label)
            radio.setChecked(checked)
            radio.toggled.connect(lambda is_checked, key=mode_key: self._on_display_mode_changed(key, is_checked))
            self.display_mode_group.addButton(radio, i)
            layout.addWidget(radio, i // 5, i % 5)
            if mode_key == "cross_section":
                # Referenz für map_editor.py, das dieses Radio (zusammen mit
                # der Shell-Zeile für Achse/Position) im 3D-Modus ausblendet,
                # da Cross-Section dort nicht darstellbar ist.
                self.cross_section_mode_radio = radio

        return layout

    # =============================================================================
    # EVENT HANDLERS
    # =============================================================================

    def _on_parameter_changed(self, param_name: str, value: float):
        """Handler für Parameter-Änderungen"""
        try:
            if self.parameter_manager:
                self.parameter_ui_changed.emit(self.generator_type, param_name, value)

            if self.rock_distribution_widget:
                self.rock_distribution_widget.update_hardness_preview(self.get_current_parameters())

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
    # ÖFFENTLICHE API FÜR DIE GLOBALE CROSS-SECTION-SHELL-ZEILE
    # (gui/map_editor.py, neben "Contour Lines" - Muster wie set_contour_
    # overlay() bei anderen Tabs)
    # =============================================================================

    def set_cross_section_axis(self, axis: str):
        """Von map_editor.py aufgerufen, wenn der Nutzer im globalen Shell-
        Bereich zwischen "Cut along X"/"Cut along Y" umschaltet."""
        self.current_cross_section_axis = axis
        if self.current_display_mode == "cross_section":
            self.update_display_mode()

    def set_cross_section_position(self, position: float):
        """Von map_editor.py aufgerufen, wenn der globale Positions-Slider
        bewegt wird. `position` ist ein Bruchteil [0-1] entlang der jeweils
        anderen Achse, unabhängig von der aktuellen Kartenauflösung."""
        self.current_cross_section_position = position
        if self.current_display_mode == "cross_section":
            self.update_display_mode()

    # =============================================================================
    # DISPLAY UPDATE SYSTEM
    # =============================================================================

    # 2D-Anzeigemodi, die direkt einer DataLODManager-Abfrage entsprechen
    # (Modus-Schlüssel -> (Abruf-Methode, Anzeige-data_type-String)). Height/
    # Rock-Outcrop/Hardness bleiben Spezialfälle (siehe update_display_mode()),
    # Cross-Section ebenfalls (braucht mehrere Datenquellen gleichzeitig).
    _DELTA_DISPLAY_MODES = {
        "terrain_hub_delta": "terrain_hub",
        "tilt_delta": "tilt",
        "fold_delta": "fold",
        "fault_delta": "fault",
        "intrusion_delta": "intrusion",
    }

    def update_display_mode(self):
        """
        Überschreibt BaseMapTab Display-Update für Geology-spezifische Modi.
        Implementiert Height/Rock-Outcrop/Hardness/Δz-Diagnose/Cross-Section
        Display-Switching sowie den optionalen 3D-Terrain-Overlay.
        """
        try:
            if not self.data_lod_manager:
                return

            current_display = self.get_current_display()
            if not current_display:
                return

            if self.current_display_mode == "cross_section":
                self._update_cross_section_display()
            else:
                data, data_type = self._resolve_display_data()
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

        except Exception as e:
            self.logger.debug(f"Geology display mode update failed: {e}")

    def _resolve_display_data(self):
        """Liefert (data, data_type) für alle Nicht-Cross-Section-Anzeigemodi."""
        if self.current_display_mode == "height":
            # Kombiniert, nicht die unbearbeitete Terrain-Rohausgabe - siehe
            # DataLODManager.get_terrain_data_combined()
            return self.data_lod_manager.get_terrain_data_combined("heightmap"), "heightmap_combined"
        if self.current_display_mode == "rock_map":
            return self.data_lod_manager.get_geology_data("rock_map"), "rock_map"
        if self.current_display_mode == "hardness_map":
            return self.data_lod_manager.get_geology_data("hardness_map"), "hardness_map"
        if self.current_display_mode in self._DELTA_DISPLAY_MODES:
            component = self._DELTA_DISPLAY_MODES[self.current_display_mode]
            return self.data_lod_manager.get_geology_delta_component(component), self.current_display_mode
        return None, None

    def _update_cross_section_display(self):
        """
        Baut das Daten-Paket für die Cross-Section-Ansicht (kein einzelnes
        2D-Raster, sondern Schichtgrenzen + Terrain-Höhe + Achse/Position) und
        pusht es unter dem eigenen data_type "geology_cross_section" - siehe
        MapDisplay2D._render_geology_cross_section() für die Interpretation.
        """
        layer_boundaries = self.data_lod_manager.get_geology_layer_boundaries()
        terrain_height = self.data_lod_manager.get_terrain_data("heightmap")
        if layer_boundaries is None or terrain_height is None:
            return

        position = self.current_cross_section_position
        payload = {
            "layer_boundaries": layer_boundaries,
            "terrain_height": terrain_height,
            # Für die Intrusions-Ausbeulung im Querschnitt (siehe
            # MapDisplay2D._render_geology_cross_section()) - optional, kann
            # None sein, falls Geology noch nicht bis zum Intrusions-Knoten
            # durchgelaufen ist.
            "intrusion_distance_map": self.data_lod_manager.get_geology_intrusion_distance_map(),
            "axis": self.current_cross_section_axis,
            "position": position,
        }
        self._push_data_to_current_display(payload, "geology_cross_section")

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

                layer_id_map = self.data_lod_manager.get_geology_layer_id_map()
                hardness_map = self.data_lod_manager.get_geology_data("hardness_map")

                if layer_id_map is not None and hardness_map is not None and self.rock_distribution_widget:
                    self.rock_distribution_widget.update_statistics(layer_id_map, hardness_map)

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
        siehe managers/parameter_manager.py).
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
    Formations-Verteilungs-Statistik nach abgeschlossener Generation. Die
    frühere "Mass Conservation" (R+G+B=255)-Statusanzeige entfällt - im
    3D-Gesteinsstapel-Modell ist jeder Pixel genau EIN diskreter Gesteinstyp,
    kein Mischverhältnis mehr, das konserviert werden müsste.
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
        self.distribution_stats.setWordWrap(True)
        layout.addWidget(self.distribution_stats)

        self.layer_status = StatusIndicator("Layer Assignment")
        layout.addWidget(self.layer_status)

        self.setLayout(layout)

    def update_hardness_preview(self, parameters: dict):
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

    def update_statistics(self, layer_id_map: np.ndarray, hardness_map: np.ndarray):
        """
        Aktualisiert Statistiken nach abgeschlossener Generation - Anteil der
        3 Härte-Kategorien und der dominanten Einzelformation, berechnet über
        layer_id_map (diskrete Schicht-/Intrusions-Zuordnung je Pixel), nicht
        mehr über RGB-Kanal-Summen (die im neuen Modell keine Mischverhältnisse
        mehr codieren).
        """
        try:
            total_pixels = layer_id_map.size
            category_counts = {"sedimentary": 0, "igneous": 0, "metamorphic": 0}
            for layer_index, layer in enumerate(ALL_ROCK_TYPES):
                count = int(np.sum(layer_id_map == layer_index))
                category_counts[layer.category] += count

            pct = {k: (v / total_pixels * 100.0 if total_pixels else 0.0) for k, v in category_counts.items()}

            dominant_index = int(np.bincount(layer_id_map.ravel(), minlength=len(ALL_ROCK_TYPES)).argmax())
            dominant_name = ALL_ROCK_TYPES[dominant_index].name

            self.distribution_stats.setText(
                f"Distribution: Sed {pct['sedimentary']:.1f}%, Ign {pct['igneous']:.1f}%, "
                f"Met {pct['metamorphic']:.1f}% - Dominant: {dominant_name}"
            )

            valid = bool(np.all((layer_id_map >= 0) & (layer_id_map <= N_LAYERS)))
            if valid:
                self.layer_status.set_success("All pixels assigned a valid rock layer")
            else:
                self.layer_status.set_warning("Some pixels have an invalid layer id")

        except Exception as e:
            self.distribution_stats.setText("Statistics calculation failed")
            self.layer_status.set_error(f"Error: {str(e)}")
