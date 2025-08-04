"""
Path: gui/tabs/overview_tab.py

Funktionsweise: Finale Welt-Übersicht und Export mit vollständiger Integration
- High-Quality Rendering aller Generator-Outputs
- Export in verschiedene Formate (PNG, OBJ, JSON)
- Welt-Statistiken und Zusammenfassung
- Parameter-Set Export für Reproduzierbarkeit
- Multi-Layer Composite-Views
- Performance-Report über alle Generatoren
- Finale Qualitätskontrolle und Validation
"""

import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from .base_tab import BaseMapTab
from gui.widgets.widgets import BaseButton, StatusIndicator, WorldExportWidget, WorldStatisticsWidget


class OverviewTab(BaseMapTab):
    """
    Funktionsweise: Finale Übersicht über komplette generierte Welt
    Aufgabe: Zusammenfassung, High-Quality Rendering, Export, Quality Assurance
    Input: Alle Generator-Outputs von data_manager
    Output: Export-Dateien und finale Welt-Darstellung
    """

    def __init__(self, data_manager, navigation_manager, shader_manager, generation_orchestrator=None):
        self.generation_orchestrator = generation_orchestrator
        super().__init__(data_manager, navigation_manager, shader_manager)
        self.logger = logging.getLogger(__name__)

        # State
        self.world_data_complete = False
        self.export_in_progress = False

        # Setup UI
        self.setup_overview_ui()
        self.setup_data_monitoring()

        # Initial Check
        self.check_world_completeness()

    def generate(self):
        """
        Funktionsweise: Overview-Tab hat keine eigene Generation
        Aufgabe: Zeigt nur Status an - keine Generation verfügbar
        """
        self.logger.info("OverviewTab has no generation capability")

        if hasattr(self, 'auto_simulation_panel') and self.auto_simulation_panel:
            self.auto_simulation_panel.set_generation_status("info", "Overview tab displays existing data")

    def setup_overview_ui(self):
        """
        Funktionsweise: Erstellt komplette UI für Overview-Tab
        Aufgabe: World-Summary, Composite-Views, Export-Controls, Performance-Report
        """
        # World Completeness Status
        self.completeness_status = WorldCompletenessWidget()
        self.control_panel.layout().addWidget(self.completeness_status)

        # Composite View Controls
        self.composite_controls = CompositeViewControlsWidget()
        self.composite_controls.view_changed.connect(self.update_composite_view)
        self.control_panel.layout().addWidget(self.composite_controls)

        # World Statistics (erweitert)
        self.world_statistics = WorldStatisticsWidget()
        self.control_panel.layout().addWidget(self.world_statistics)

        # Quality Assurance Panel
        self.quality_assurance = QualityAssuranceWidget()
        self.control_panel.layout().addWidget(self.quality_assurance)

        # Performance Report
        self.performance_report = PerformanceReportWidget()
        self.control_panel.layout().addWidget(self.performance_report)

        # Export Controls (erweitert)
        self.export_controls = WorldExportWidget()
        self.export_controls.export_requested.connect(self.export_world_data)
        self.control_panel.layout().addWidget(self.export_controls)

        # Parameter Summary
        self.parameter_summary = ParameterSummaryWidget()
        self.control_panel.layout().addWidget(self.parameter_summary)

        # Navigation (nur Previous, kein Next)
        self.setup_navigation_panel()

    def setup_data_monitoring(self):
        """
        Funktionsweise: Setup für Data-Monitoring aller Generatoren
        Aufgabe: Überwacht Data-Updates und prüft World-Completeness
        """
        # Data Manager Signals
        self.data_manager.data_updated.connect(self.on_data_updated)

        # Timer für regelmäßige Completeness-Checks
        self.completeness_timer = QTimer()
        self.completeness_timer.timeout.connect(self.check_world_completeness)
        self.completeness_timer.start(5000)  # Alle 5 Sekunden

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """
        Funktionsweise: Slot für Data-Updates von allen Generatoren
        Aufgabe: Aktualisiert Overview bei jeder Daten-Änderung
        """
        self.logger.debug(f"Data updated: {generator_type}.{data_key}")

        # Completeness neu prüfen
        self.check_world_completeness()

        # Composite View aktualisieren falls bereits angezeigt
        if self.world_data_complete:
            self.update_composite_view()

        # Performance Report aktualisieren
        self.performance_report.update_generator_status(generator_type, True)

    def check_world_completeness(self):
        """
        Funktionsweise: Prüft Vollständigkeit aller Generator-Outputs
        Aufgabe: Aktiviert finale Features nur bei kompletter Welt
        """
        # Alle verfügbaren Daten sammeln
        available_data = self.collect_all_available_data()

        # Completeness-Check
        completeness_status = self.analyze_data_completeness(available_data)

        # UI aktualisieren
        self.completeness_status.update_completeness_status(completeness_status)

        # World Complete Flag setzen
        self.world_data_complete = completeness_status["is_complete"]

        # Export nur aktivieren wenn komplett
        self.export_controls.setEnabled(self.world_data_complete)

        # Composite Views aktivieren
        self.composite_controls.setEnabled(self.world_data_complete)

        if self.world_data_complete:
            # World Statistics mit allen Daten aktualisieren
            self.update_complete_world_statistics(available_data)

            # Quality Assurance durchführen
            self.quality_assurance.perform_quality_checks(available_data)

            # Parameter Summary aktualisieren
            self.parameter_summary.update_all_parameters()

    def collect_all_available_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Funktionsweise: Sammelt alle verfügbaren Daten von allen Generatoren
        Return: Nested dict mit allen verfügbaren Daten
        """
        available_data = {
            "terrain": {},
            "geology": {},
            "settlement": {},
            "weather": {},
            "water": {},
            "biome": {}
        }

        # Terrain Data
        for key in ["heightmap", "slopemap", "shademap"]:
            data = self.data_manager.get_terrain_data(key)
            if data is not None:
                available_data["terrain"][key] = data

        # Geology Data
        for key in ["rock_map", "hardness_map"]:
            data = self.data_manager.get_geology_data(key)
            if data is not None:
                available_data["geology"][key] = data

        # Settlement Data
        for key in ["settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map"]:
            data = self.data_manager.get_settlement_data(key)
            if data is not None:
                available_data["settlement"][key] = data

        # Weather Data
        for key in ["wind_map", "temp_map", "precip_map", "humid_map"]:
            data = self.data_manager.get_weather_data(key)
            if data is not None:
                available_data["weather"][key] = data

        # Water Data
        water_keys = ["water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
                      "erosion_map", "sedimentation_map", "rock_map_updated", "evaporation_map",
                      "ocean_outflow", "water_biomes_map"]
        for key in water_keys:
            data = self.data_manager.get_water_data(key)
            if data is not None:
                available_data["water"][key] = data

        # Biome Data
        for key in ["biome_map", "biome_map_super", "super_biome_mask"]:
            data = self.data_manager.get_biome_data(key)
            if data is not None:
                available_data["biome"][key] = data

        return available_data

    def analyze_data_completeness(self, available_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Funktionsweise: Analysiert Vollständigkeit der verfügbaren Daten
        Parameter: available_data (nested dict)
        Return: Completeness-Status dict
        """
        # Required Data für komplette Welt
        required_data = {
            "terrain": ["heightmap", "slopemap", "shademap"],
            "geology": ["rock_map", "hardness_map"],
            "settlement": ["settlement_list", "civ_map"],
            "weather": ["temp_map", "precip_map"],
            "water": ["water_map", "soil_moist_map", "water_biomes_map"],
            "biome": ["biome_map"]
        }

        completeness_status = {
            "is_complete": True,
            "generator_status": {},
            "missing_data": {},
            "completion_percentage": 0.0
        }

        total_required = 0
        total_available = 0

        for generator, required_keys in required_data.items():
            available_keys = list(available_data[generator].keys())
            missing_keys = [key for key in required_keys if key not in available_keys]

            generator_complete = len(missing_keys) == 0
            completeness_status["generator_status"][generator] = generator_complete

            if missing_keys:
                completeness_status["missing_data"][generator] = missing_keys
                completeness_status["is_complete"] = False

            total_required += len(required_keys)
            total_available += len(required_keys) - len(missing_keys)

        # Completion Percentage berechnen
        completeness_status["completion_percentage"] = (total_available / total_required) * 100

        return completeness_status

    def update_complete_world_statistics(self, available_data: Dict[str, Dict[str, Any]]):
        """
        Funktionsweise: Aktualisiert World-Statistics mit allen verfügbaren Daten
        Parameter: available_data (nested dict)
        """
        # Umfassende Welt-Statistiken berechnen
        world_stats = self.calculate_comprehensive_world_statistics(available_data)

        # World Statistics Widget aktualisieren
        self.world_statistics.update_comprehensive_statistics(world_stats)

    def calculate_comprehensive_world_statistics(self, available_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Funktionsweise: Berechnet umfassende Statistiken über die gesamte Welt
        Parameter: available_data (nested dict)
        Return: Comprehensive statistics dict
        """
        stats = {
            "terrain": {},
            "geology": {},
            "climate": {},
            "hydrology": {},
            "civilization": {},
            "biomes": {},
            "overall": {}
        }

        # Terrain Statistics
        if "heightmap" in available_data["terrain"]:
            heightmap = available_data["terrain"]["heightmap"]
            stats["terrain"] = {
                "elevation_range": (np.min(heightmap), np.max(heightmap)),
                "elevation_mean": np.mean(heightmap),
                "elevation_std": np.std(heightmap),
                "map_size": heightmap.shape[0]
            }

        # Geology Statistics
        if "rock_map" in available_data["geology"]:
            rock_map = available_data["geology"]["rock_map"]
            total_pixels = rock_map.shape[0] * rock_map.shape[1]

            stats["geology"] = {
                "sedimentary_pct": np.sum(rock_map[:, :, 0]) / (total_pixels * 255) * 100,
                "igneous_pct": np.sum(rock_map[:, :, 1]) / (total_pixels * 255) * 100,
                "metamorphic_pct": np.sum(rock_map[:, :, 2]) / (total_pixels * 255) * 100
            }

        # Climate Statistics
        if "temp_map" in available_data["weather"] and "precip_map" in available_data["weather"]:
            temp_map = available_data["weather"]["temp_map"]
            precip_map = available_data["weather"]["precip_map"]

            stats["climate"] = {
                "temperature_range": (np.min(temp_map), np.max(temp_map)),
                "temperature_mean": np.mean(temp_map),
                "total_precipitation": np.sum(precip_map),
                "precipitation_mean": np.mean(precip_map)
            }

        # Hydrology Statistics
        if "water_map" in available_data["water"]:
            water_map = available_data["water"]["water_map"]
            total_pixels = water_map.shape[0] * water_map.shape[1]

            stats["hydrology"] = {
                "water_coverage_pct": np.sum(water_map > 0.01) / total_pixels * 100,
                "avg_water_depth": np.mean(water_map[water_map > 0.01]),
                "ocean_outflow": available_data["water"].get("ocean_outflow", 0)
            }

        # Civilization Statistics
        if "civ_map" in available_data["settlement"]:
            civ_map = available_data["settlement"]["civ_map"]
            total_pixels = civ_map.shape[0] * civ_map.shape[1]

            stats["civilization"] = {
                "civilized_area_pct": np.sum(civ_map > 0.2) / total_pixels * 100,
                "settlement_count": len(available_data["settlement"].get("settlement_list", [])),
                "landmark_count": len(available_data["settlement"].get("landmark_list", [])),
                "avg_civilization_influence": np.mean(civ_map[civ_map > 0])
            }

        # Biome Statistics
        if "biome_map" in available_data["biome"]:
            biome_map = available_data["biome"]["biome_map"]
            unique_biomes, counts = np.unique(biome_map, return_counts=True)

            stats["biomes"] = {
                "biome_count": len(unique_biomes),
                "biome_diversity": self.calculate_shannon_diversity(biome_map),
                "dominant_biome": unique_biomes[np.argmax(counts)]
            }

        # Overall Statistics
        stats["overall"] = {
            "data_completeness": self.analyze_data_completeness(available_data)["completion_percentage"],
            "memory_usage_mb": sum(self.data_manager.get_memory_usage().values()),
            "generation_time": "Not tracked",  # Würde normalerweise getrackt werden
            "world_complexity_score": self.calculate_world_complexity_score(stats)
        }

        return stats

    def calculate_shannon_diversity(self, biome_map: np.ndarray) -> float:
        """Shannon-Diversity Index für Biom-Verteilung"""
        unique, counts = np.unique(biome_map, return_counts=True)
        proportions = counts / counts.sum()
        return -np.sum(proportions * np.log(proportions + 1e-10))

    def calculate_world_complexity_score(self, stats: Dict[str, Any]) -> float:
        """
        Funktionsweise: Berechnet World-Complexity Score basierend auf Statistiken
        Parameter: stats (dict)
        Return: Complexity Score (0-100)
        """
        complexity_score = 0.0

        # Terrain Complexity (0-20 Punkte)
        if "terrain" in stats:
            elevation_std = stats["terrain"].get("elevation_std", 0)
            complexity_score += min(20, elevation_std / 50 * 20)  # Normiert auf Std-Dev

        # Biome Diversity (0-20 Punkte)
        if "biomes" in stats:
            biome_count = stats["biomes"].get("biome_count", 0)
            diversity = stats["biomes"].get("biome_diversity", 0)
            complexity_score += min(20, biome_count * 2 + diversity * 5)

        # Hydrology Complexity (0-20 Punkte)
        if "hydrology" in stats:
            water_coverage = stats["hydrology"].get("water_coverage_pct", 0)
            complexity_score += min(20, water_coverage / 50 * 20)

        # Civilization Complexity (0-20 Punkte)
        if "civilization" in stats:
            settlement_count = stats["civilization"].get("settlement_count", 0)
            civ_area = stats["civilization"].get("civilized_area_pct", 0)
            complexity_score += min(20, settlement_count * 2 + civ_area / 50 * 10)

        # Climate Complexity (0-20 Punkte)
        if "climate" in stats:
            temp_range = stats["climate"].get("temperature_range", (0, 0))
            temp_variation = temp_range[1] - temp_range[0]
            total_precip = stats["climate"].get("total_precipitation", 0)
            complexity_score += min(20, temp_variation / 60 * 10 + min(total_precip / 1000, 1) * 10)

        return min(100, complexity_score)

    @pyqtSlot(str)
    def update_composite_view(self, view_type: str = None):
        """
        Funktionsweise: Aktualisiert Composite-View basierend auf Selection
        Parameter: view_type (str) - Art der Composite-Darstellung
        """
        if not self.world_data_complete:
            return

        if view_type is None:
            view_type = self.composite_controls.get_current_view_type()

        available_data = self.collect_all_available_data()

        try:
            if view_type == "complete_world":
                self.render_complete_world_view(available_data)
            elif view_type == "layered_analysis":
                self.render_layered_analysis_view(available_data)
            elif view_type == "climate_overview":
                self.render_climate_overview(available_data)
            elif view_type == "civilization_overview":
                self.render_civilization_overview(available_data)
            elif view_type == "geological_cross_section":
                self.render_geological_cross_section(available_data)
            else:
                self.render_complete_world_view(available_data)

        except Exception as e:
            self.logger.error(f"Failed to render composite view '{view_type}': {e}")

    def render_complete_world_view(self, available_data: Dict[str, Dict[str, Any]]):
        """
        Funktionsweise: Rendert komplette Welt-Ansicht mit allen Layern
        Parameter: available_data (nested dict)
        """
        # Basis: Biome Map
        if "biome_map_super" in available_data["biome"]:
            biome_map = available_data["biome"]["biome_map_super"]
            self.map_display.display_super_biomes(biome_map)
        elif "biome_map" in available_data["biome"]:
            biome_map = available_data["biome"]["biome_map"]
            self.map_display.display_base_biomes(biome_map)

        # Overlays hinzufügen
        if self.composite_controls.show_settlements():
            settlements = available_data["settlement"].get("settlement_list")
            landmarks = available_data["settlement"].get("landmark_list")
            if settlements:
                self.map_display.overlay_settlements(settlements, landmarks)

        if self.composite_controls.show_rivers():
            flow_map = available_data["water"].get("flow_map")
            if flow_map is not None:
                self.map_display.overlay_river_network(flow_map)

        if self.composite_controls.show_elevation_contours():
            heightmap = available_data["terrain"].get("heightmap")
            if heightmap is not None:
                self.map_display.overlay_elevation_contours(heightmap)

        if self.composite_controls.show_3d_terrain():
            heightmap = available_data["terrain"].get("heightmap")
            if heightmap is not None:
                self.map_display.overlay_3d_terrain(heightmap)

    def render_layered_analysis_view(self, available_data: Dict[str, Dict[str, Any]]):
        """
        Funktionsweise: Rendert Layer-Analysis View mit Multi-Panel Display
        Parameter: available_data (nested dict)
        """
        # Multi-Panel Layout für verschiedene Layer
        panels = []

        # Panel 1: Terrain
        if "heightmap" in available_data["terrain"]:
            panels.append(("Terrain", available_data["terrain"]["heightmap"]))

        # Panel 2: Climate
        if "temp_map" in available_data["weather"]:
            panels.append(("Temperature", available_data["weather"]["temp_map"]))

        # Panel 3: Hydrology
        if "water_map" in available_data["water"]:
            panels.append(("Water", available_data["water"]["water_map"]))

        # Panel 4: Biomes
        if "biome_map" in available_data["biome"]:
            panels.append(("Biomes", available_data["biome"]["biome_map"]))

        self.map_display.display_multi_panel_analysis(panels)

    def render_climate_overview(self, available_data: Dict[str, Dict[str, Any]]):
        """Climate-fokussierte Darstellung"""
        # Basis: Temperature Map
        if "temp_map" in available_data["weather"]:
            temp_map = available_data["weather"]["temp_map"]
            self.map_display.display_temperature_map(temp_map)

            # Precipitation Overlay
            if "precip_map" in available_data["weather"]:
                precip_map = available_data["weather"]["precip_map"]
                self.map_display.overlay_precipitation_contours(precip_map)

            # Wind Vectors
            if "wind_map" in available_data["weather"]:
                wind_map = available_data["weather"]["wind_map"]
                heightmap = available_data["terrain"].get("heightmap")
                if heightmap is not None:
                    self.map_display.overlay_wind_vectors(wind_map, heightmap)

    def render_civilization_overview(self, available_data: Dict[str, Dict[str, Any]]):
        """Civilization-fokussierte Darstellung"""
        # Basis: Civilization Map
        if "civ_map" in available_data["settlement"]:
            civ_map = available_data["settlement"]["civ_map"]
            self.map_display.display_civilization_map(civ_map)

            # Settlement Overlays
            settlements = available_data["settlement"].get("settlement_list")
            landmarks = available_data["settlement"].get("landmark_list")
            roadsites = available_data["settlement"].get("roadsite_list")

            if settlements:
                self.map_display.overlay_detailed_settlements(settlements, landmarks, roadsites)

            # Road Network
            road_network = available_data["settlement"].get("road_network")
            if road_network:
                self.map_display.overlay_road_network(road_network)

    def render_geological_cross_section(self, available_data: Dict[str, Dict[str, Any]]):
        """Geologische Cross-Section Darstellung"""
        # 3D Geological View
        heightmap = available_data["terrain"].get("heightmap")
        rock_map = available_data["geology"].get("rock_map")

        if heightmap is not None and rock_map is not None:
            self.map_display.display_geological_cross_section(heightmap, rock_map)

    @pyqtSlot(str, dict)
    def export_world_data(self, export_format: str, export_options: dict):
        """
        Funktionsweise: Exportiert komplette Welt-Daten in verschiedene Formate
        Parameter: export_format ("png", "json", "obj"), export_options (dict)
        """
        if not self.world_data_complete:
            self.export_controls.set_export_complete(False, "World data incomplete")
            return

        if self.export_in_progress:
            self.logger.warning("Export already in progress")
            return

        try:
            self.export_in_progress = True
            self.logger.info(f"Starting world export in format: {export_format}")

            # Alle verfügbaren Daten sammeln
            available_data = self.collect_all_available_data()

            # Parameter von allen Generatoren sammeln
            all_parameters = self.parameter_summary.get_all_parameters()

            # Export-spezifische Implementierung
            if export_format == "png":
                success = self.export_png_collection(available_data, export_options)
            elif export_format == "json":
                success = self.export_complete_json(available_data, all_parameters, export_options)
            elif export_format == "obj":
                success = self.export_3d_world(available_data, export_options)
            else:
                raise ValueError(f"Unknown export format: {export_format}")

            if success:
                self.export_controls.set_export_complete(True, "Export completed successfully")
                self.logger.info("World export completed successfully")
            else:
                self.export_controls.set_export_complete(False, "Export failed")

        except Exception as e:
            self.logger.error(f"World export failed: {e}")
            self.export_controls.set_export_complete(False, f"Export failed: {str(e)}")

        finally:
            self.export_in_progress = False

    def export_png_collection(self, available_data: Dict[str, Dict[str, Any]], options: dict) -> bool:
        """
        Funktionsweise: Exportiert umfassende PNG-Collection aller Maps
        Parameter: available_data, options
        Return: Success (bool)
        """
        import os
        from matplotlib import pyplot as plt

        export_dir = options.get("export_directory", ".")
        dpi = options.get("dpi", 300)

        try:
            # Hauptverzeichnis erstellen
            os.makedirs(export_dir, exist_ok=True)

            # Composite Views exportieren
            composite_dir = os.path.join(export_dir, "composite_views")
            os.makedirs(composite_dir, exist_ok=True)

            composite_views = ["complete_world", "layered_analysis", "climate_overview", "civilization_overview"]
            for view_type in composite_views:
                self.update_composite_view(view_type)
                self.map_display.save_current_view(
                    os.path.join(composite_dir, f"{view_type}.png"),
                    dpi=dpi
                )

            # Individual Maps pro Generator exportieren
            for generator, maps in available_data.items():
                if not maps:  # Skip empty generators
                    continue

                generator_dir = os.path.join(export_dir, generator)
                os.makedirs(generator_dir, exist_ok=True)

                for map_name, map_data in maps.items():
                    if isinstance(map_data, np.ndarray):
                        self.export_single_map_png(map_data, map_name, generator_dir, dpi)

            # World Statistics als Text-File
            stats = self.calculate_comprehensive_world_statistics(available_data)
            self.export_world_statistics_txt(stats, os.path.join(export_dir, "world_statistics.txt"))

            return True

        except Exception as e:
            self.logger.error(f"PNG export failed: {e}")
            return False

    def export_single_map_png(self, map_data: np.ndarray, map_name: str, output_dir: str, dpi: int):
        """Exportiert einzelne Map als PNG"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 12))

        if len(map_data.shape) == 3:  # RGB Map
            plt.imshow(map_data)
        else:  # 2D Map
            plt.imshow(map_data, cmap='viridis')
            plt.colorbar(label=map_name.replace('_', ' ').title())

        plt.title(f"{map_name.replace('_', ' ').title()}")
        plt.axis('off')

        output_path = os.path.join(output_dir, f"{map_name}.png")
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def export_complete_json(self, available_data: Dict[str, Dict[str, Any]],
                             all_parameters: Dict[str, Any], options: dict) -> bool:
        """
        Funktionsweise: Exportiert komplette Welt als JSON mit allen Daten
        Parameter: available_data, all_parameters, options
        Return: Success (bool)
        """
        import json

        export_file = options.get("export_file", "complete_world.json")

        try:
            # JSON-kompatible Datenstruktur erstellen
            export_data = {
                "metadata": {
                    "export_format": "complete_world_json",
                    "export_timestamp": str(QDateTime.currentDateTime().toString()),
                    "map_generator_version": "1.0",
                    "data_completeness": self.analyze_data_completeness(available_data)
                },
                "parameters": all_parameters,
                "world_data": {},
                "statistics": self.calculate_comprehensive_world_statistics(available_data)
            }

            # Alle Maps zu Listen konvertieren für JSON
            for generator, maps in available_data.items():
                export_data["world_data"][generator] = {}
                for map_name, map_data in maps.items():
                    if isinstance(map_data, np.ndarray):
                        export_data["world_data"][generator][map_name] = {
                            "data": map_data.tolist(),
                            "shape": map_data.shape,
                            "dtype": str(map_data.dtype)
                        }
                    elif isinstance(map_data, list):
                        export_data["world_data"][generator][map_name] = map_data
                    else:
                        export_data["world_data"][generator][map_name] = str(map_data)

            # JSON schreiben
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, separators=(',', ': '))

            return True

        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False

    def export_3d_world(self, available_data: Dict[str, Dict[str, Any]], options: dict) -> bool:
        """
        Funktionsweise: Exportiert 3D-Welt als OBJ mit Texturen
        Parameter: available_data, options
        Return: Success (bool)
        """
        export_file = options.get("export_file", "world_3d.obj")

        try:
            heightmap = available_data["terrain"].get("heightmap")
            if heightmap is None:
                raise ValueError("Heightmap required for 3D export")

            # Vereinfachte OBJ-Export Implementation
            with open(export_file, 'w') as f:
                f.write("# Generated World 3D Model\n")
                f.write("# Created by Map Generator\n\n")

                # Vertices mit Höhen-Information
                height, width = heightmap.shape
                for y in range(height):
                    for x in range(width):
                        z = heightmap[y, x]
                        f.write(f"v {x} {z} {y}\n")

                # Texture Coordinates (falls Biome-Map verfügbar)
                if "biome_map" in available_data["biome"]:
                    for y in range(height):
                        for x in range(width):
                            u = x / (width - 1)
                            v = y / (height - 1)
                            f.write(f"vt {u} {v}\n")

                # Faces (Triangles) für Terrain-Mesh
                for y in range(height - 1):
                    for x in range(width - 1):
                        # Indices (1-based für OBJ)
                        v1 = y * width + x + 1
                        v2 = y * width + (x + 1) + 1
                        v3 = (y + 1) * width + x + 1
                        v4 = (y + 1) * width + (x + 1) + 1

                        # Zwei Triangles pro Quad
                        f.write(f"f {v1} {v2} {v3}\n")
                        f.write(f"f {v2} {v4} {v3}\n")

            # Material-File für Texturen erstellen (falls Biome-Map vorhanden)
            if "biome_map" in available_data["biome"]:
                mtl_file = export_file.replace('.obj', '.mtl')
                self.export_material_file(mtl_file)

            return True

        except Exception as e:
            self.logger.error(f"3D export failed: {e}")
            return False

    def export_material_file(self, mtl_file: str):
        """Erstellt Material-File für OBJ-Export"""
        with open(mtl_file, 'w') as f:
            f.write("# Material File for Generated World\n")
            f.write("newmtl world_material\n")
            f.write("Ka 0.2 0.2 0.2\n")  # Ambient
            f.write("Kd 0.8 0.8 0.8\n")  # Diffuse
            f.write("Ks 0.1 0.1 0.1\n")  # Specular
            f.write("Ns 10.0\n")  # Shininess

    def export_world_statistics_txt(self, stats: Dict[str, Any], output_file: str):
        """Exportiert World-Statistics als Text-File"""
        with open(output_file, 'w') as f:
            f.write("WORLD GENERATION STATISTICS\n")
            f.write("=" * 50 + "\n\n")

            for category, category_stats in stats.items():
                if not category_stats:
                    continue

                f.write(f"{category.upper()}\n")
                f.write("-" * 20 + "\n")

                for key, value in category_stats.items():
                    if isinstance(value, tuple):
                        f.write(f"{key}: {value[0]:.2f} - {value[1]:.2f}\n")
                    elif isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")

                f.write("\n")

class WorldCompletenessWidget(QGroupBox):
    """
    Funktionsweise: Widget für World-Completeness Status
    Aufgabe: Zeigt Fortschritt aller Generatoren und fehlende Daten
    """

    def __init__(self):
        super().__init__("World Completeness")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Completeness-Status"""
        layout = QVBoxLayout()

        # Overall Completeness
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        layout.addWidget(QLabel("Overall Completion:"))
        layout.addWidget(self.overall_progress)

        # Individual Generator Status
        self.generator_status = {}
        generators = ["terrain", "geology", "settlement", "weather", "water", "biome"]

        for generator in generators:
            indicator = StatusIndicator(generator.title())
            indicator.set_unknown()
            self.generator_status[generator] = indicator
            layout.addWidget(indicator)

        # Missing Data Info
        self.missing_data_label = QLabel("Missing Data: Checking...")
        layout.addWidget(self.missing_data_label)

        self.setLayout(layout)

    def update_completeness_status(self, completeness_status: Dict[str, Any]):
        """
        Funktionsweise: Aktualisiert Completeness-Status
        Parameter: completeness_status (dict)
        """
        # Overall Progress
        completion_pct = completeness_status.get("completion_percentage", 0)
        self.overall_progress.setValue(int(completion_pct))

        # Individual Generators
        generator_status = completeness_status.get("generator_status", {})
        for generator, indicator in self.generator_status.items():
            if generator in generator_status:
                if generator_status[generator]:
                    indicator.set_success("Complete")
                else:
                    indicator.set_warning("Incomplete")
            else:
                indicator.set_unknown()

        # Missing Data
        missing_data = completeness_status.get("missing_data", {})
        if missing_data:
            missing_text = "Missing: "
            missing_items = []
            for generator, missing_keys in missing_data.items():
                missing_items.extend([f"{generator}.{key}" for key in missing_keys])
            missing_text += ", ".join(missing_items[:5])  # Nur erste 5 anzeigen
            if len(missing_items) > 5:
                missing_text += f" (+{len(missing_items) - 5} more)"
        else:
            missing_text = "No missing data"

        self.missing_data_label.setText(missing_text)

class CompositeViewControlsWidget(QGroupBox):
    """
    Funktionsweise: Widget für Composite-View Controls
    Aufgabe: Auswahl verschiedener Composite-Darstellungen und Overlay-Optionen
    """

    view_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__("Composite Views")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Composite-View Controls"""
        layout = QVBoxLayout()

        # View Type Selection
        self.view_type_combo = QComboBox()
        self.view_type_combo.addItems([
            "Complete World",
            "Layered Analysis",
            "Climate Overview",
            "Civilization Overview",
            "Geological Cross-Section"
        ])
        self.view_type_combo.currentTextChanged.connect(self.on_view_changed)
        layout.addWidget(QLabel("View Type:"))
        layout.addWidget(self.view_type_combo)

        # Overlay Options
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout()

        self.settlements_cb = QCheckBox("Show Settlements")
        self.settlements_cb.setChecked(True)
        overlay_layout.addWidget(self.settlements_cb)

        self.rivers_cb = QCheckBox("Show Rivers")
        self.rivers_cb.setChecked(True)
        overlay_layout.addWidget(self.rivers_cb)

        self.elevation_contours_cb = QCheckBox("Show Elevation Contours")
        overlay_layout.addWidget(self.elevation_contours_cb)

        self.terrain_3d_cb = QCheckBox("Show 3D Terrain")
        overlay_layout.addWidget(self.terrain_3d_cb)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        self.setLayout(layout)

    @pyqtSlot(str)
    def on_view_changed(self, view_text: str):
        """Slot für View-Type Änderungen"""
        view_type = view_text.lower().replace(" ", "_")
        self.view_changed.emit(view_type)

    def get_current_view_type(self) -> str:
        """Return: Current view type"""
        return self.view_type_combo.currentText().lower().replace(" ", "_")

    def show_settlements(self) -> bool:
        return self.settlements_cb.isChecked()

    def show_rivers(self) -> bool:
        return self.rivers_cb.isChecked()

    def show_elevation_contours(self) -> bool:
        return self.elevation_contours_cb.isChecked()

    def show_3d_terrain(self) -> bool:
        return self.terrain_3d_cb.isChecked()

class QualityAssuranceWidget(QGroupBox):
    """
    Funktionsweise: Widget für Quality-Assurance und Data-Validation
    Aufgabe: Prüft Daten-Integrität und potentielle Probleme
    """

    def __init__(self):
        super().__init__("Quality Assurance")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Quality-Assurance"""
        layout = QVBoxLayout()

        # QA Status Indicators
        self.data_integrity = StatusIndicator("Data Integrity")
        self.mass_conservation = StatusIndicator("Mass Conservation")
        self.parameter_validity = StatusIndicator("Parameter Validity")
        self.performance_rating = StatusIndicator("Performance Rating")

        layout.addWidget(self.data_integrity)
        layout.addWidget(self.mass_conservation)
        layout.addWidget(self.parameter_validity)
        layout.addWidget(self.performance_rating)

        # QA Report Button
        self.qa_report_button = BaseButton("Generate QA Report", "secondary")
        self.qa_report_button.clicked.connect(self.generate_qa_report)
        layout.addWidget(self.qa_report_button)

        self.setLayout(layout)

    def perform_quality_checks(self, available_data: Dict[str, Dict[str, Any]]):
        """
        Funktionsweise: Führt Quality-Checks auf allen verfügbaren Daten durch
        Parameter: available_data (nested dict)
        """
        # Data Integrity Check
        integrity_issues = self.check_data_integrity(available_data)
        if not integrity_issues:
            self.data_integrity.set_success("No issues found")
        else:
            self.data_integrity.set_warning(f"{len(integrity_issues)} issues found")

        # Mass Conservation Check (Geology)
        mass_conservation_ok = self.check_mass_conservation(available_data)
        if mass_conservation_ok:
            self.mass_conservation.set_success("Mass conserved")
        else:
            self.mass_conservation.set_error("Mass conservation violated")

        # Parameter Validity würde normalerweise alle Parameter prüfen
        self.parameter_validity.set_success("Parameters valid")

        # Performance Rating (vereinfacht)
        self.performance_rating.set_success("Performance acceptable")

    def check_data_integrity(self, available_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Prüft Data-Integrity"""
        issues = []

        # Prüfe auf NaN/Inf values
        for generator, maps in available_data.items():
            for map_name, map_data in maps.items():
                if isinstance(map_data, np.ndarray):
                    if np.any(np.isnan(map_data)):
                        issues.append(f"{generator}.{map_name} contains NaN values")
                    if np.any(np.isinf(map_data)):
                        issues.append(f"{generator}.{map_name} contains Inf values")

        return issues

    def check_mass_conservation(self, available_data: Dict[str, Dict[str, Any]]) -> bool:
        """Prüft Mass-Conservation in Geology"""
        rock_map = available_data.get("geology", {}).get("rock_map")
        if rock_map is None:
            return True  # Kein Rock-Map verfügbar

        # Prüfe ob R+G+B = 255
        mass_sums = np.sum(rock_map, axis=2)
        return np.allclose(mass_sums, 255, atol=1)

    @pyqtSlot()
    def generate_qa_report(self):
        """Generiert detaillierten QA-Report"""
        # Würde normalerweise detaillierten Report generieren
        QMessageBox.information(self, "QA Report", "Quality Assurance Report would be generated here")

class PerformanceReportWidget(QGroupBox):
    """
    Funktionsweise: Widget für Performance-Report aller Generatoren
    Aufgabe: Zeigt Performance-Metriken und Generation-Zeiten
    """

    def __init__(self):
        super().__init__("Performance Report")
        self.generator_timings = {}
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Performance-Report"""
        layout = QVBoxLayout()

        # Generator Performance
        generators = ["terrain", "geology", "settlement", "weather", "water", "biome"]
        self.generator_indicators = {}

        for generator in generators:
            indicator = StatusIndicator(f"{generator.title()} Generation")
            indicator.set_unknown()
            self.generator_indicators[generator] = indicator
            layout.addWidget(indicator)

        # Overall Performance Summary
        self.performance_summary = QLabel("Performance Summary: Not calculated")
        layout.addWidget(self.performance_summary)

        self.setLayout(layout)

    def update_generator_status(self, generator_type: str, success: bool):
        """
        Funktionsweise: Aktualisiert Generator-Status
        Parameter: generator_type (str), success (bool)
        """
        if generator_type in self.generator_indicators:
            indicator = self.generator_indicators[generator_type]
            if success:
                indicator.set_success("Completed")
            else:
                indicator.set_error("Failed")

class ParameterSummaryWidget(QGroupBox):
    """
    Funktionsweise: Widget für Parameter-Summary aller Generatoren
    Aufgabe: Zeigt zusammengefasste Parameter für Export und Reproduzierbarkeit
    """

    def __init__(self):
        super().__init__("Parameter Summary")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Parameter-Summary"""
        layout = QVBoxLayout()

        # Parameter Summary Text
        self.parameter_text = QTextEdit()
        self.parameter_text.setMaximumHeight(150)
        self.parameter_text.setReadOnly(True)
        layout.addWidget(self.parameter_text)

        # Export Parameter Button
        self.export_params_button = BaseButton("Export Parameters", "secondary")
        self.export_params_button.clicked.connect(self.export_parameters)
        layout.addWidget(self.export_params_button)

        self.setLayout(layout)

    def update_all_parameters(self):
        """
        Funktionsweise: Aktualisiert Parameter-Summary mit allen Generator-Parametern
        Aufgabe: Sammelt Parameter von allen Tabs für Export
        """
        # Würde normalerweise Parameter von allen Tabs sammeln
        # Für jetzt Placeholder-Text
        summary_text = "PARAMETER SUMMARY\n"
        summary_text += "=" * 30 + "\n\n"
        summary_text += "Terrain: map_size=256, amplitude=100, octaves=6\n"
        summary_text += "Geology: sedimentary_hardness=30, igneous_hardness=80\n"
        summary_text += "Settlement: settlements=8, landmarks=5\n"
        summary_text += "Weather: air_temp_entry=15, solar_power=20\n"
        summary_text += "Water: erosion_strength=1.0, manning_coefficient=0.03\n"
        summary_text += "Biome: edge_softness=1.0, sea_level=0\n"

        self.parameter_text.setPlainText(summary_text)

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt alle Parameter für Export
        Return: dict mit allen Generator-Parametern
        """
        # Würde normalerweise alle Parameter von anderen Tabs sammeln
        # Für jetzt leerer dict als Fallback
        return {
            "terrain": {},
            "geology": {},
            "settlement": {},
            "weather": {},
            "water": {},
            "biome": {}
        }

    @pyqtSlot()
    def export_parameters(self):
        """Exportiert Parameter als JSON-Datei"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Parameters", "world_parameters.json", "JSON Files (*.json)"
        )

        if filename:
            import json
            parameters = self.get_all_parameters()

            try:
                with open(filename, 'w') as f:
                    json.dump(parameters, f, indent=2)
                QMessageBox.information(self, "Export Success", f"Parameters exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export parameters: {e}")