"""
Path: gui/tabs/terrain_tab.py

Funktionsweise: Terrain-Editor mit GenerationOrchestrator Integration - VOLLSTÄNDIG REFACTORED
- Erbt von BaseMapTab für gemeinsame Features
- NEUE INTEGRATION: GenerationOrchestrator statt direkte Core-Calls
- Spezialisierte Widgets: TerrainParameterPanel, TerrainStatisticsWidget
- Live 2D/3D Preview über map_display_2d/3d.py erweitert
- Real-time Terrain-Statistics (Höhenverteilung, Steigungen, Verschattung)
- Output: heightmap, slopemap, shademap für nachfolgende Generatoren
- REFACTORED: Modulare Architektur, Standard-Orchestrator-Handler, Parameter-Update-Manager,
             Vereinfachtes LOD-System, Zentraler Display-Renderer, Explizite Imports
"""

from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QCheckBox, QComboBox, QRadioButton
)
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QFont

import numpy as np
import logging
import time

from .base_tab import BaseMapTab
from gui.config.value_default import TERRAIN, get_parameter_config, validate_parameter_set
from gui.widgets.widgets import ParameterSlider, StatusIndicator, BaseButton, RandomSeedButton, ParameterUpdateManager

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


class TerrainConstants:
    """Konstanten für Terrain-Tab"""
    MAX_PROGRESS = 100
    SHADOW_ANGLES = 6
    DEFAULT_SHADOW_ANGLE = 2

    @staticmethod
    def validate_angle_index(index: int) -> bool:
        """Validiert Shadow-Angle-Index"""
        return 0 <= index < TerrainConstants.SHADOW_ANGLES


class TerrainTab(BaseMapTab):
    """
    Funktionsweise: Hauptklasse für Terrain-Generation mit Standard-Orchestrator-Handler - VOLLSTÄNDIG REFACTORED
    Aufgabe: Koordiniert UI, Parameter-Management und Orchestrator-basierte Generation
    Output: heightmap, slopemap, shademap für alle nachfolgenden Generatoren
    REFACTORED: Standard-Orchestrator-Handler, Vereinfachtes LOD-System, Zentraler Renderer
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):

        self.logger = logging.getLogger(__name__)

        # Terrain-Konstanten
        self.terrain_constants = TerrainConstants()
        super().__init__(data_manager, navigation_manager, shader_manager, generation_orchestrator)

        # Parameter-Tracking für Cache-Validation mit Sync
        self.current_parameters = {}
        self.last_generated_parameters = {}

        # Parameter-Update-Manager für Race-Condition Prevention
        self.parameter_manager = ParameterUpdateManager(self)

        # Setup Standard-Orchestrator-Handler
        self.setup_standard_orchestrator_handlers("terrain")

        # Setup UI
        self.setup_terrain_ui()
        self.setup_parameter_validation()

        # Initial Parameter Load
        self.load_default_parameters()

    def setup_terrain_ui(self):
        """
        Funktionsweise: Erstellt spezialisierte UI für Terrain-Generator - REFACTORED
        Aufgabe: Parameter-Slider, Statistics-Widget, Standard-LOD-Controls
        """
        # Parameter Panel erstellen
        self.parameter_panel = self.create_terrain_parameter_panel()
        self.control_panel.layout().addWidget(self.parameter_panel)

        # Standard LOD Control Panel aus BaseTab nutzen
        self.lod_control_panel = self.create_standard_lod_panel()
        self.control_panel.layout().addWidget(self.lod_control_panel)

        # Statistics Widget
        self.statistics_widget = TerrainStatisticsWidget()
        self.control_panel.layout().addWidget(self.statistics_widget)

    def create_terrain_parameter_panel(self) -> QGroupBox:
        """
        Funktionsweise: Erstellt Parameter-Panel mit allen Terrain-Parametern - REFACTORED
        Aufgabe: Slider für alle Core-Parameter + Random-Seed Button
        Return: QGroupBox mit allen Parameter-Slidern
        REFACTORED: Nur eine Definition, RandomSeedButton korrekt integriert
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

            # Spezial-Layout für map_seed mit Random-Button
            if param_name == "map_seed":
                seed_container = QWidget()
                seed_layout = QHBoxLayout()
                seed_layout.setContentsMargins(0, 0, 0, 0)

                # Parameter-Slider
                slider = ParameterSlider(
                    label="Map Seed",
                    min_val=param_config["min"],
                    max_val=param_config["max"],
                    default_val=param_config["default"],
                    step=param_config.get("step", 1),
                    suffix=param_config.get("suffix", "")
                )

                # Random-Seed Button
                random_button = RandomSeedButton()
                random_button.seed_generated.connect(slider.setValue)

                seed_layout.addWidget(slider)
                seed_layout.addWidget(random_button)
                seed_container.setLayout(seed_layout)

                slider.valueChanged.connect(self.on_parameter_changed)
                self.parameter_sliders[param_name] = slider
                layout.addWidget(seed_container)

            else:
                # Normale Parameter-Slider
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
                layout.addWidget(slider)

        panel.setLayout(layout)
        return panel

    def setup_parameter_validation(self):
        """
        Funktionsweise: Setup für Parameter-Validation und Cross-Parameter Constraints
        Aufgabe: Verbindet Validation-System mit Parameter-Änderungen
        """
        self.validation_status = StatusIndicator("Parameter Validation")
        self.control_panel.layout().addWidget(self.validation_status)

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
        self.last_generated_parameters = self.current_parameters.copy()

    def get_current_parameters(self) -> dict:
        """
        Funktionsweise: Sammelt aktuelle Parameter-Werte von allen Slidern
        Return: dict mit allen aktuellen Parameter-Werten inkl. Size-Weiterleitung
        """
        parameters = {}
        for param_name, slider in self.parameter_sliders.items():
            parameters[param_name] = slider.getValue()

        if "size" in parameters:
            parameters["size"] = int(parameters["size"])

        return parameters

    def set_parameters(self, parameters: dict):
        """
        Funktionsweise: Setzt Parameter-Werte mit Synchronisation
        Parameter: parameters (dict)
        Aufgabe: External Parameter-Updates mit State-Synchronisation
        """
        for param_name, value in parameters.items():
            if param_name in self.parameter_sliders:
                self.parameter_sliders[param_name].setValue(value)

        # Parameter-State synchronisieren
        self.current_parameters = self.get_current_parameters()
        self.last_generated_parameters = self.current_parameters.copy()

    @pyqtSlot()
    def on_parameter_changed(self):
        """
        Funktionsweise: Slot für Parameter-Änderungen mit Debouncing - REFACTORED
        Aufgabe: Triggert Validation und Orchestrator-basierte Auto-Generation
        REFACTORED: Nutzt ParameterUpdateManager gegen Race-Conditions
        """
        self.current_parameters = self.get_current_parameters()

        # Debounced Updates über ParameterUpdateManager
        self.parameter_manager.request_validation()

        # Auto-Simulation triggern (wenn aktiviert)
        if self.auto_simulation_enabled and not self.generation_in_progress:
            if self.has_significant_parameter_change():
                self.parameter_manager.request_generation()

    def has_significant_parameter_change(self) -> bool:
        """
        Funktionsweise: Prüft ob Parameter-Änderung signifikant genug für Neu-Generation ist
        Return: bool - Signifikante Änderung
        """
        for param_name, current_value in self.current_parameters.items():
            last_value = self.last_generated_parameters.get(param_name, None)

            if last_value is None or abs(current_value - last_value) > 0.001:
                return True

        return False

    def validate_current_parameters(self):
        """
        Funktionsweise: Validiert aktuelle Parameter auf Constraints und Performance
        Aufgabe: Prüft Cross-Parameter Validation und zeigt Warnings/Errors
        Besonderheit: Parameter Handler schützt vor Validation-Fehlern
        """
        try:
            is_valid, warnings, errors = validate_parameter_set("terrain", self.current_parameters)
        except ImportError:
            is_valid, warnings, errors = True, [], []

        if errors:
            self.validation_status.set_error(f"Errors: {'; '.join(errors)}")
        elif warnings:
            self.validation_status.set_warning(f"Warnings: {'; '.join(warnings)}")
        else:
            self.validation_status.set_success("Parameters valid")

    def generate(self):
        """
        Funktionsweise: Hauptmethode für Terrain-Generation mit Orchestrator Integration - REFACTORED
        Aufgabe: Startet Terrain-Generation über GenerationOrchestrator mit Target-LOD
        Besonderheit: Memory-Critical Handler schützt vor Memory-Errors bei großen Arrays
        REFACTORED: Standard-Orchestrator-Handler nutzen
        """
        try:
            if not self.generation_orchestrator:
                self.logger.error("No GenerationOrchestrator available")
                self.handle_generation_error(Exception("GenerationOrchestrator not available"))
                return

            if self.generation_in_progress:
                self.logger.info("Generation already in progress, ignoring request")
                return

            self.logger.info(f"Starting terrain generation with target LOD: {self.target_lod}")

            self.start_generation_timing()
            self.generation_in_progress = True

            params = self.current_parameters.copy()

            request_id = self.generation_orchestrator.request_generation(
                generator_type="terrain",
                parameters=params,
                target_lod=self.target_lod,
                source_tab="terrain",
                priority=10
            )

            if request_id:
                self.logger.info(f"Terrain generation requested: {request_id}")
                self.last_generated_parameters = params.copy()
            else:
                self.logger.error("Failed to request terrain generation")
                self.handle_generation_error(Exception("Failed to request generation"))

        except Exception as e:
            self.handle_generation_error(e)
            self.generation_in_progress = False
            raise

    def create_visualization_controls(self):
        """
        Funktionsweise: Erstellt Terrain-spezifische Visualization-Controls - ÜBERSCHRIEBEN
        Aufgabe: Erweitert Base-Controls um Terrain-spezifische Modi
        Return: QWidget mit erweiterten Visualization-Controls
        """
        widget = super().create_visualization_controls()
        layout = widget.layout()

        # Terrain-spezifische Modi hinzufügen
        self.slopemap_radio = QRadioButton("Slope")
        self.slopemap_radio.setStyleSheet("font-size: 11px;")
        self.slopemap_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.slopemap_radio, 1)
        layout.insertWidget(1, self.slopemap_radio)

        self.shademap_radio = QRadioButton("Shadow")
        self.shademap_radio.setStyleSheet("font-size: 11px;")
        self.shademap_radio.toggled.connect(self.update_display_mode)
        self.display_mode.addButton(self.shademap_radio, 2)
        layout.insertWidget(2, self.shademap_radio)

        # Shadow Angle Slider (kompakt)
        self.shadow_angle_slider = ParameterSlider(
            label="Angle",
            min_val=0,
            max_val=self.terrain_constants.SHADOW_ANGLES - 1,
            default_val=self.terrain_constants.DEFAULT_SHADOW_ANGLE,
            step=1
        )
        self.shadow_angle_slider.valueChanged.connect(self.update_display_mode)
        self.shadow_angle_slider.setVisible(False)
        layout.addWidget(self.shadow_angle_slider)

        return widget

    def update_display_mode(self):
        """
        Funktionsweise: Handler für Terrain Display-Mode-Änderungen - REFACTORED
        Aufgabe: Nutzt zentralen Renderer aus BaseTab
        REFACTORED: Zentraler _render_current_mode eliminiert Code-Duplikation
        """
        if not hasattr(self, 'display_mode'):
            return

        current_mode = self.display_mode.checkedId()
        current_display = self.get_current_display()

        if not current_display:
            self.logger.debug("No display available for mode update")
            return

        # Shadow-Angle-Slider nur bei Shadow-Mode anzeigen
        if hasattr(self, 'shadow_angle_slider'):
            self.shadow_angle_slider.setVisible(current_mode == 2)

        try:
            if current_mode == 0:  # Heightmap
                heightmap = self.data_manager.get_terrain_data("heightmap")
                if heightmap is not None:
                    self._render_current_mode(current_mode, current_display, heightmap, "heightmap")

            elif current_mode == 1:  # Slopemap
                slopemap = self.data_manager.get_terrain_data("slopemap")
                if slopemap is not None:
                    self._render_current_mode(current_mode, current_display, slopemap, "slopemap")

            elif current_mode == 2:  # Shademap mit Memory-optimiertem Slicing
                try:
                    shademap = self.data_manager.get_terrain_data("shadowmap")
                    if shademap is not None and hasattr(self, 'shadow_angle_slider'):
                        angle_index = int(self.shadow_angle_slider.getValue())

                        if self.terrain_constants.validate_angle_index(angle_index):
                            if len(shademap.shape) == 3 and shademap.shape[2] > angle_index:
                                # Memory-optimiertes Slicing mit Views statt Copies
                                angle_shadowmap = shademap[:, :, angle_index:angle_index+1].squeeze()
                                self._render_current_mode(current_mode, current_display, angle_shadowmap, "shadowmap")
                            else:
                                self._render_current_mode(current_mode, current_display, shademap, "shadowmap")
                        else:
                            self.logger.warning(f"Invalid shadow angle index: {angle_index}")

                except Exception as e:
                    self.logger.error(f"Shademap display error: {e}")

        except Exception as e:
            self.logger.error(f"Display mode update failed: {e}")

    def update_terrain_display(self):
        """
        Funktionsweise: Aktualisiert Display basierend auf aktuellem Visualization-Mode
        Aufgabe: Delegiert an update_display_mode für konsistente Rendering-Logic
        """
        self.update_display_mode()

    def update_terrain_statistics(self):
        """
        Funktionsweise: Aktualisiert Terrain-Statistics mit neuesten Daten
        """
        heightmap = self.data_manager.get_terrain_data("heightmap")
        slopemap = self.data_manager.get_terrain_data("slopemap")
        shadowmap = self.data_manager.get_terrain_data("shadowmap")

        if heightmap is not None and slopemap is not None and shadowmap is not None:
            best_lod = self.get_best_available_lod()
            self.statistics_widget.update_statistics(heightmap, slopemap, shadowmap, best_lod)

    def get_best_available_lod(self) -> str:
        """
        Funktionsweise: Findet höchstes verfügbares LOD-Level
        Return: str - Bestes verfügbares LOD oder None
        """
        lod_priority = ["FINAL", "LOD256", "LOD128", "LOD64"]

        for lod in lod_priority:
            if lod in self.available_lods:
                return lod

        return "Unknown"

    # STANDARD ORCHESTRATOR UI-UPDATE METHODS (von BaseTab übernommen):

    @pyqtSlot(str)
    def _update_ui_generation_started(self, lod_level: str):
        """UI-Update für Generation-Start - TERRAIN-SPEZIFISCH"""
        super()._update_ui_generation_started(lod_level)
        self.logger.debug(f"Terrain generation started: {lod_level}")

    @pyqtSlot(str, bool)
    def _update_ui_generation_completed(self, lod_level: str, success: bool):
        """UI-Update für Generation-Completion - TERRAIN-SPEZIFISCH"""
        super()._update_ui_generation_completed(lod_level, success)

        if success:
            try:
                self.update_terrain_display()
                self.update_terrain_statistics()
            except Exception as e:
                self.logger.error(f"Post-generation update failed: {e}")

        self.logger.info(f"Terrain generation completed: {lod_level} success={success}")

    @pyqtSlot(str, int, str)
    def _update_ui_generation_progress(self, lod_level: str, progress_percent: int, detail: str):
        """UI-Update für Generation-Progress - TERRAIN-SPEZIFISCH"""
        super()._update_ui_generation_progress(lod_level, progress_percent, detail)

        if hasattr(self, 'auto_simulation_panel'):
            self.auto_simulation_panel.set_generation_status("progress", detail)

    @pyqtSlot(str)
    def _update_ui_lod_progression_completed(self, final_lod: str):
        """UI-Update für LOD-Progression-Completion - TERRAIN-SPEZIFISCH"""
        super()._update_ui_lod_progression_completed(final_lod)
        self.logger.info(f"Terrain LOD progression completed: {final_lod}")

    def get_generation_status_summary(self) -> dict:
        """
        Funktionsweise: Sammelt aktuellen Generation-Status für Export/Debug
        Return: dict mit aktuellem Status
        """
        return {
            "available_lods": list(self.available_lods),
            "target_lod": self.target_lod,
            "generation_in_progress": self.generation_in_progress,
            "current_parameters": self.current_parameters,
            "last_generated_parameters": self.last_generated_parameters,
            "best_available_lod": self.get_best_available_lod()
        }

    def force_regeneration(self):
        """
        Funktionsweise: Erzwingt komplette Neu-Generation (für Debug/Development)
        Aufgabe: Löscht alle LODs und startet von LOD64 neu
        """
        self.available_lods.clear()

        if self.data_manager:
            self.data_manager.invalidate_cache("terrain")

        self.generate()

    def cleanup_resources(self):
        """
        Funktionsweise: Cleanup-Methode für Resource-Management - ERWEITERT
        Aufgabe: Wird beim Tab-Wechsel oder Schließen aufgerufen
        ERWEITERT: Parameter-Manager cleanup
        """
        # Parameter-Manager cleanup
        if hasattr(self, 'parameter_manager'):
            self.parameter_manager.cleanup()

        # Parent Cleanup aufrufen
        super().cleanup_resources()


class TerrainStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Widget für Real-time Terrain-Statistiken - REFACTORED
    Aufgabe: Zeigt Höhenverteilung, Steigungsstatistiken, Verschattungsinfo
    REFACTORED: Optimierte Statistics-Berechnung, bessere Performance
    """

    def __init__(self):
        super().__init__("Terrain Statistics")
        self.current_lod = "Unknown"
        self.cached_statistics = {}  # Cache für wiederholte Berechnungen
        self.setup_ui()

    def setup_ui(self):
        """
        Funktionsweise: Erstellt UI für Statistik-Anzeige
        """
        layout = QVBoxLayout()

        # LOD-Info
        self.lod_info = QLabel("LOD: Unknown")
        self.lod_info.setStyleSheet("font-weight: bold; color: #3498db;")
        layout.addWidget(self.lod_info)

        # Statistics Labels
        self.height_stats = QLabel("Height: -")
        self.slope_stats = QLabel("Slope: -")
        self.shadow_stats = QLabel("Shadow: -")
        self.performance_stats = QLabel("Performance: -")

        layout.addWidget(self.height_stats)
        layout.addWidget(self.slope_stats)
        layout.addWidget(self.shadow_stats)
        layout.addWidget(self.performance_stats)

        self.setLayout(layout)

    def update_statistics(self, heightmap: np.ndarray, slopemap: np.ndarray,
                         shadowmap: np.ndarray, lod_level: str = "Unknown"):
        """
        Funktionsweise: Berechnet und zeigt aktuelle Terrain-Statistiken - OPTIMIERT
        Parameter: heightmap, slopemap, shadowmap (numpy arrays), lod_level (str)
        OPTIMIERT: Caching für wiederholte Berechnungen
        """
        # Cache-Key basierend auf Array-Shapes und LOD
        cache_key = f"{heightmap.shape}_{slopemap.shape}_{shadowmap.shape}_{lod_level}"

        if cache_key in self.cached_statistics:
            stats = self.cached_statistics[cache_key]
        else:
            stats = self._calculate_statistics(heightmap, slopemap, shadowmap)
            self.cached_statistics[cache_key] = stats

        self.current_lod = lod_level
        self.lod_info.setText(f"LOD: {lod_level} ({heightmap.shape[0]}x{heightmap.shape[1]})")

        # Update Labels
        self.height_stats.setText(f"Height: {stats['height_min']:.1f}m - {stats['height_max']:.1f}m (μ={stats['height_mean']:.1f}m, σ={stats['height_std']:.1f}m)")
        self.slope_stats.setText(f"Slope: 0° - {stats['slope_max']:.1f}° (μ={stats['slope_mean']:.1f}°)")
        self.shadow_stats.setText(f"Shadow: {stats['shadow_min']:.2f} - 1.00 (μ={stats['shadow_mean']:.2f})")
        self.performance_stats.setText(f"Data Size: {stats['data_size_mb']:.1f} MB")

    def _calculate_statistics(self, heightmap: np.ndarray, slopemap: np.ndarray,
                            shadowmap: np.ndarray) -> dict:
        """
        Funktionsweise: Berechnet Statistics einmalig für Caching
        Parameter: heightmap, slopemap, shadowmap
        Return: dict mit allen Statistics
        """
        # Height Statistics
        height_min = float(np.min(heightmap))
        height_max = float(np.max(heightmap))
        height_mean = float(np.mean(heightmap))
        height_std = float(np.std(heightmap))

        # Slope Statistics (Convert to degrees)
        if len(slopemap.shape) == 3 and slopemap.shape[2] >= 2:
            slope_magnitude = np.sqrt(slopemap[:,:,0]**2 + slopemap[:,:,1]**2)
        else:
            slope_magnitude = slopemap

        slope_degrees = np.arctan(slope_magnitude) * 180 / np.pi
        slope_max = float(np.max(slope_degrees))
        slope_mean = float(np.mean(slope_degrees))

        # Shadow Statistics (Average über alle Winkel falls Multi-Angle)
        if len(shadowmap.shape) == 3:
            shadow_mean = float(np.mean(shadowmap))
            shadow_min = float(np.min(shadowmap))
        else:
            shadow_mean = float(np.mean(shadowmap))
            shadow_min = float(np.min(shadowmap))

        # Performance Statistics
        data_size_mb = (heightmap.nbytes + slopemap.nbytes + shadowmap.nbytes) / (1024 * 1024)

        return {
            'height_min': height_min,
            'height_max': height_max,
            'height_mean': height_mean,
            'height_std': height_std,
            'slope_max': slope_max,
            'slope_mean': slope_mean,
            'shadow_min': shadow_min,
            'shadow_mean': shadow_mean,
            'data_size_mb': data_size_mb
        }

    def clear_cache(self):
        """Löscht Statistics-Cache"""
        self.cached_statistics.clear()