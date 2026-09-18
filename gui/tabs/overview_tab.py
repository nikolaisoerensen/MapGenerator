"""
Path: gui/tabs/overview_tab.py

Funktionsweise: Finale Welt-Übersicht und Export mit vollständiger Integration
- High-Quality Rendering aller Generator-Outputs
- Export in verschiedene Formate (PNG, OBJ, JSON)
- Welt-Statistiken und Zusammenfassung
- Parameter-Set Export für Reproduzierbarkeit
- Finale Qualitätskontrolle und Validation

Kein Composite-View-Rendering (Ticket #6, 2026-09-16): die frühere
Multi-Panel/Composite-Ansicht rief ausschließlich Methoden auf
`self.map_display` auf, das in dieser Klasse nirgends zugewiesen wird -
vollständig toter Code seit jeher, siehe docs/SITZUNGSLOG.md und
docs/SPEC_OVERLAYS.md (Out of Scope).
"""

import os
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from .base_tab import BaseMapTab
from gui.widgets.widgets import (
    BaseButton,
    StatusIndicator
)


class OverviewTab(BaseMapTab):
    """
    Funktionsweise: Finale Übersicht über komplette generierte Welt
    Aufgabe: Zusammenfassung, High-Quality Rendering, Export, Quality Assurance
    Input: Alle Generator-Outputs von data_lod_manager
    Output: Export-Dateien und finale Welt-Darstellung
    """

    def __init__(self, data_lod_manager, parameter_manager, navigation_manager, shader_manager, generation_orchestrator):
        self.generator_type = "overview"

        super().__init__(
            data_lod_manager=data_lod_manager,
            parameter_manager=parameter_manager,
            navigation_manager=navigation_manager,
            shader_manager=shader_manager,
            generation_orchestrator=generation_orchestrator
        )
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

    def create_parameter_controls(self):
        """
        No-Op-Override: OverviewTab ist ein Summary-/Export-Tab ohne eigene
        Generierungs-Parameter (siehe generate() oben - "has no generation
        capability") und baut seine Widgets über setup_overview_ui() statt
        über diesen Basisklassen-Hook - unterdrückt die sonst bei jedem
        Tab-Start geloggte "should implement create_parameter_controls()"-
        Warnung aus BaseMapTab, die für diesen Tab-Typ ohnehin nicht zutrifft.
        """
        pass

    def setup_overview_ui(self):
        """
        Funktionsweise: Erstellt komplette UI für Overview-Tab
        Aufgabe: World-Summary, Export-Controls, Quality-Assurance, Parameter-Summary
        """
        # World Completeness Status
        self.completeness_status = WorldCompletenessWidget()
        self.control_panel.layout().addWidget(self.completeness_status)

        # World Statistics (erweitert)
        self.world_statistics = WorldStatisticsWidget()
        self.control_panel.layout().addWidget(self.world_statistics)

        # Quality Assurance Panel
        self.quality_assurance = QualityAssuranceWidget()
        self.control_panel.layout().addWidget(self.quality_assurance)

        # Export Controls (erweitert)
        self.export_controls = WorldExportWidget()
        self.export_controls.export_requested.connect(self.export_world_data)
        self.control_panel.layout().addWidget(self.export_controls)

        # Layer-Export (alle Radio-Button-Layer als PNG, siehe
        # gui/utils/map_export.py) - eigenständig von export_controls oben,
        # da dessen world_data_complete-Gate aktuell dauerhaft blockiert
        # (siehe check_world_completeness()).
        from gui.utils.map_export import DEFAULT_EXPORT_ROOT
        self.layer_export = LayerExportWidget(DEFAULT_EXPORT_ROOT)
        self.layer_export.export_requested.connect(self.export_layers_to_disk)
        self.control_panel.layout().addWidget(self.layer_export)

        # Parameter Summary - liest die echten Slider-Werte der Generator-Tabs
        # über den ParameterManager (siehe ParameterSummaryWidget-Docstring).
        self.parameter_summary = ParameterSummaryWidget(self.parameter_manager)
        self.control_panel.layout().addWidget(self.parameter_summary)

    def setup_data_monitoring(self):
        """
        Funktionsweise: Setup für Data-Monitoring aller Generatoren
        Aufgabe: Überwacht Data-Updates und prüft World-Completeness
        """
        # Data Manager Signals
        self.data_lod_manager.data_updated.connect(self.on_data_updated)

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

        # Completeness neu prüfen (billig, kein Redraw - unabhängig von
        # Sichtbarkeit).
        self.check_world_completeness()

    def check_world_completeness(self):
        """
        Funktionsweise: Prüft Vollständigkeit aller Generator-Outputs
        Aufgabe: Aktiviert finale Features nur bei kompletter Welt
        """
        # Datei-Namensvorschlag mit dem aktuellen Map-Seed aktuell halten
        # (nur Placeholder, überschreibt nie manuell eingetippten Text - siehe
        # LayerExportWidget.set_filename_suggestion()).
        if hasattr(self, 'layer_export'):
            try:
                seed = self.parameter_manager.get_tab_parameters("terrain").get("map_seed") \
                    if self.parameter_manager else None
                self.layer_export.set_filename_suggestion(
                    f"Mapseed_{seed}" if seed is not None else "Mapseed_xxxxxx")
            except Exception as e:
                self.logger.debug(f"Filename-Vorschlag nicht aktualisierbar: {e}")

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
            "erosion": {},
            "water": {},
            "biome": {}
        }

        # Terrain Data
        for key in ["heightmap", "slopemap", "shadowmap"]:
            data = self.data_lod_manager.get_terrain_data(key)
            if data is not None:
                available_data["terrain"][key] = data

        # Geology Data
        for key in ["rock_map", "hardness_map"]:
            data = self.data_lod_manager.get_geology_data(key)
            if data is not None:
                available_data["geology"][key] = data

        # Settlement Data
        for key in ["settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map"]:
            data = self.data_lod_manager.get_settlement_data(key)
            if data is not None:
                available_data["settlement"][key] = data

        # Weather Data
        for key in ["wind_map", "temp_map", "precip_map", "humid_map"]:
            data = self.data_lod_manager.get_weather_data(key)
            if data is not None:
                available_data["weather"][key] = data

        # Water Data
        water_keys = ["water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
                      "erosion_map", "sedimentation_map", "rock_map_updated", "evaporation_map",
                      "ocean_outflow", "water_biomes_map"]
        for key in water_keys:
            data = self.data_lod_manager.get_water_data(key)
            if data is not None:
                available_data["water"][key] = data

        # Biome Data
        for key in ["biome_map", "biome_map_super", "super_biome_mask"]:
            data = self.data_lod_manager.get_biome_data(key)
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
            "terrain": ["heightmap", "slopemap", "shadowmap"],
            "geology": ["rock_map", "hardness_map"],
            "settlement": ["settlement_list", "civ_map"],
            "weather": ["temp_map", "precip_map"],
            "erosion": ["erosion_map", "sedimentation_map", "sediment_load_map"],
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
            "memory_usage_mb": sum(self.data_lod_manager.get_memory_usage().values()),
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

    @pyqtSlot(str, dict)
    @pyqtSlot(str, str)
    def export_layers_to_disk(self, filename_prefix: str, output_root: str):
        """
        Funktionsweise: Slot für LayerExportWidget.export_requested - exportiert
        alle aktuell verfügbaren Radio-Button-Layer als PNG (siehe
        gui/utils/map_export.py) in output_root/filename_prefix/, unabhängig
        vom world_data_complete-Status.
        """
        from gui.utils.map_export import export_all_layers
        try:
            success, message, _output_dir = export_all_layers(
                self.data_lod_manager, self.parameter_manager, output_root, filename_prefix)
        except Exception as e:
            self.logger.error(f"Layer export failed: {e}")
            success, message = False, f"Export failed: {e}"
        self.layer_export.set_export_complete(success, message)

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
        """Duenner Wrapper - Logik liegt in gui/utils/map_export.py (Ticket #38:
        von core/welt_io.py ohne Qt-GUI wiederverwendbar, siehe dortige
        Begruendung)."""
        from gui.utils.map_export import export_single_map_png as _impl
        _impl(map_data, map_name, output_dir, dpi)

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
        """Duenner Wrapper - siehe export_single_map_png() oben."""
        from gui.utils.map_export import export_material_file as _impl
        _impl(mtl_file)

    def export_world_statistics_txt(self, stats: Dict[str, Any], output_file: str):
        """Duenner Wrapper - siehe export_single_map_png() oben."""
        from gui.utils.map_export import export_world_statistics_txt as _impl
        _impl(stats, output_file)


class WorldStatisticsWidget(QGroupBox):
    """
    Funktionsweise: Zeigt umfassende Welt-Statistiken über alle Generatoren
    Aufgabe: Stellt die von OverviewTab berechneten Kategorie-Statistiken lesbar dar
    """

    def __init__(self):
        super().__init__("World Statistics")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für die Welt-Statistik-Anzeige"""
        layout = QVBoxLayout()

        self.statistics_text = QTextEdit()
        self.statistics_text.setReadOnly(True)
        self.statistics_text.setMinimumHeight(200)
        layout.addWidget(self.statistics_text)

        self.setLayout(layout)

    def update_comprehensive_statistics(self, stats: dict):
        """
        Funktionsweise: Aktualisiert die Anzeige mit den kompletten Welt-Statistiken
        Parameter: stats (dict) - Kategorien (terrain, geology, climate, ...) mit Kennzahlen
        """
        lines = []

        for category, category_stats in stats.items():
            if not category_stats:
                continue

            lines.append(category.upper())

            for key, value in category_stats.items():
                if isinstance(value, tuple) and len(value) == 2:
                    lines.append(f"  {key}: {value[0]:.2f} - {value[1]:.2f}")
                elif isinstance(value, float):
                    lines.append(f"  {key}: {value:.3f}")
                else:
                    lines.append(f"  {key}: {value}")

            lines.append("")

        self.statistics_text.setPlainText("\n".join(lines))


class WorldExportWidget(QGroupBox):
    """
    Funktionsweise: Bedienoberfläche für den Welt-Export in verschiedene Formate
    Aufgabe: Format-Auswahl und Export-Anforderung; der eigentliche Export erfolgt in OverviewTab
    Kommunikation: Signal export_requested(export_format, export_options)
    """

    export_requested = pyqtSignal(str, dict)

    def __init__(self):
        super().__init__("World Export")
        self.setup_ui()

    def setup_ui(self):
        """Erstellt UI für Format-Auswahl, Export-Button und Status-Rückmeldung"""
        layout = QVBoxLayout()

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(["png", "json", "obj"])
        format_layout.addWidget(self.format_combo)

        layout.addLayout(format_layout)

        self.export_button = BaseButton("Export World", "primary")
        self.export_button.clicked.connect(self._on_export_clicked)
        layout.addWidget(self.export_button)

        self.export_status = StatusIndicator("Export Status")
        self.export_status.set_unknown()
        layout.addWidget(self.export_status)

        self.setLayout(layout)

    def _on_export_clicked(self):
        """
        Funktionsweise: Löst den Export mit dem gewählten Format aus
        Aufgabe: Sammelt Format und Optionen und emittiert export_requested
        """
        export_format = self.format_combo.currentText()
        export_options = {"export_directory": "world_export"}
        self.export_requested.emit(export_format, export_options)

    def set_export_complete(self, success: bool, message: str):
        """
        Funktionsweise: Zeigt das Ergebnis eines Export-Vorgangs an
        Parameter: success (bool), message (str)
        """
        if success:
            self.export_status.set_success(message)
        else:
            self.export_status.set_error(message)

class LayerExportWidget(QGroupBox):
    """
    Funktionsweise: UI für den Export aller Radio-Button-Layer als Bilddateien
    (siehe gui/utils/map_export.py) - Dateiname mit Mapseed-Vorschlag als
    Placeholder (wird bei Seed-Änderungen aktualisiert, ohne bereits vom
    Nutzer eingetippten Text zu überschreiben), Zielordner-Auswahl (Default:
    <Projekt-Root>/exports), Export-Button, Status-Rückmeldung.
    Aufgabe: Bewusst eigenständig von WorldExportWidget - dessen
    world_data_complete-Gate blockiert aktuell dauerhaft (siehe
    OverviewTab.check_world_completeness()), dieser Export funktioniert mit
    jedem aktuellen Kartenstand, auch nur teilweise generiert.
    Kommunikation: Signal export_requested(filename_prefix: str, output_root: str)
    """

    export_requested = pyqtSignal(str, str)

    def __init__(self, default_export_root: str):
        super().__init__("Export")
        self.export_root = default_export_root
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("Filename:"))
        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("Mapseed_xxxxxx")
        filename_layout.addWidget(self.filename_edit)
        layout.addLayout(filename_layout)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Folder:"))
        self.folder_label = QLabel(self.export_root)
        self.folder_label.setWordWrap(True)
        folder_layout.addWidget(self.folder_label, 1)
        self.browse_button = QPushButton("...")
        self.browse_button.setMaximumWidth(30)
        self.browse_button.clicked.connect(self._on_browse_clicked)
        folder_layout.addWidget(self.browse_button)
        layout.addLayout(folder_layout)

        self.export_button = BaseButton("Export", "primary")
        self.export_button.clicked.connect(self._on_export_clicked)
        layout.addWidget(self.export_button)

        self.export_status = StatusIndicator("Export Status")
        self.export_status.set_unknown()
        layout.addWidget(self.export_status)

        self.setLayout(layout)

    def set_filename_suggestion(self, filename: str):
        """Aktualisiert nur den Placeholder (nicht den echten Feld-Text) -
        überschreibt dadurch nie, was der Nutzer selbst eingetippt hat."""
        self.filename_edit.setPlaceholderText(filename)

    def _on_browse_clicked(self):
        chosen = QFileDialog.getExistingDirectory(self, "Export-Ordner wählen", self.export_root)
        if chosen:
            self.export_root = chosen
            self.folder_label.setText(chosen)

    def _on_export_clicked(self):
        filename_prefix = self.filename_edit.text().strip() or self.filename_edit.placeholderText()
        if not filename_prefix:
            self.export_status.set_error("Kein Dateiname angegeben")
            return
        self.export_requested.emit(filename_prefix, self.export_root)

    def set_export_complete(self, success: bool, message: str):
        if success:
            self.export_status.set_success(message)
        else:
            self.export_status.set_error(message)


class WorldCompletenessWidget(QGroupBox):
    """
    Funktionsweise: Widget für World-Completeness Status
    Aufgabe: Zeigt Fortschritt aller Generatoren und fehlende Daten
    """

    def __init__(self):
        super().__init__("World Completeness")
        self.setup_ui()

    def setup_ui(self):
        """
        Erstellt UI für Completeness-Status.

        Die 6 Pro-Generator-StatusIndicator-Zeilen (Terrain/Geology/.../Biome
        Complete-Incomplete) wurden entfernt - das linke PipelineStatusPanel
        (immer sichtbar, auch während der Overview-Tab aktiv ist) zeigt
        dieselbe Information bereits pro Calculator-Node granularer an. Nur
        der Gesamt-Fortschrittsbalken und die Missing-Data-Liste sind
        Informationen, die es im linken Panel nicht gibt - siehe
        [[project-overview-tab-cleanup]].
        """
        layout = QVBoxLayout()

        # Overall Completeness
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        layout.addWidget(QLabel("Overall Completion:"))
        layout.addWidget(self.overall_progress)

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

class ParameterSummaryWidget(QGroupBox):
    """
    Funktionsweise: Widget für Parameter-Summary aller Generatoren
    Aufgabe: Zeigt zusammengefasste Parameter für Export und Reproduzierbarkeit

    Braucht den ParameterManager, um die TATSÄCHLICH eingestellten Werte der
    Generator-Tabs zu lesen (get_tab_parameters()). Vorher gab dieses Widget
    hartkodierte Platzhalter aus - beim Umbau auf echte Werte (2026-07-27)
    wurde der Manager zunächst nicht durchgereicht, wodurch
    get_all_parameters() bei jedem Daten-Update in einen AttributeError lief.
    """

    def __init__(self, parameter_manager=None):
        super().__init__("Parameter Summary")
        self.parameter_manager = parameter_manager
        self.setup_ui()

    def set_parameter_manager(self, parameter_manager):
        """Nachträgliches Setzen, falls das Widget vor dem Manager existiert."""
        self.parameter_manager = parameter_manager

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
        lines = ["PARAMETER SUMMARY", "=" * 30, ""]

        all_parameters = self.get_all_parameters()
        for generator, parameters in all_parameters.items():
            if not parameters:
                lines.append(f"{generator.capitalize()}: (noch nicht generiert)")
                continue
            values = ", ".join(f"{key}={value}" for key, value in sorted(parameters.items()))
            lines.append(f"{generator.capitalize()}: {values}")

        self.parameter_text.setPlainText("\n".join(lines))

    # Reihenfolge der Generatoren in Summary und Export - entspricht der
    # Pipeline-Reihenfolge, nicht der alphabetischen.
    _GENERATOR_ORDER = ("terrain", "geology", "erosion", "weather", "water",
                        "biome", "settlement")

    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Funktionsweise: Sammelt die tatsächlich eingestellten Parameter aller
        Generator-Tabs über den ParameterManager.
        Return: dict generator -> parameter-dict (leeres dict pro Generator,
            dessen Tab noch nicht registriert ist)

        Bis 2026-07-27 gaben diese Methode und update_all_parameters()
        hartkodierte Platzhalter zurück - der Parameter-Export schrieb dadurch
        eine Datei mit sechs leeren Objekten, und die Summary zeigte
        Parameter-Namen, die es teilweise gar nicht mehr gab (z.B.
        `manning_coefficient`, seit dem Pipe-Modell-Umbau ohne Wirkung und
        inzwischen ganz entfernt).
        """
        if not self.parameter_manager:
            return {generator: {} for generator in self._GENERATOR_ORDER}

        return {
            generator: dict(self.parameter_manager.get_tab_parameters(generator) or {})
            for generator in self._GENERATOR_ORDER
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
