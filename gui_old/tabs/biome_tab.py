#!/usr/bin/env python3
"""
Path: MapGenerator/gui/tabs/biome_tab.py
__init__.py existiert in "tabs"

World Generator GUI - Biome Tab
Tab 6: Biome Klassifizierung und Weltansicht
Alle Verbesserungen aus Schritt 1-3 implementiert
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QLabel, QPushButton,
                             QFrame, QSpacerItem, QSizePolicy,
                             QCheckBox, QGroupBox, QTextEdit)
from matplotlib.colors import ListedColormap
from gui_old.widgets.navigation_mixin import NavigationMixin, TabNavigationHelper
from gui_old.widgets.map_canvas import BiomeCanvas
from gui_old.utils.performance_utils import performance_tracked
from gui_old.managers.parameter_manager import WorldParameterManager


class BiomeMapCanvas(BiomeCanvas):
    """
    Funktionsweise: Finale Biome Map Canvas mit allen Weltdaten
    - Kombiniert alle vorherigen Tab-Parameter für finale Welt-Visualisierung
    - Performance-optimiert für komplexe Multi-Layer Rendering
    - Intelligente Biome-Klassifizierung basierend auf allen Parametern
    """

    def __init__(self):
        super().__init__('biome_final_map', title='Finale Biom-Karte', debounce_ms=500)
        self.biome_definitions = self._init_biome_definitions()

    def _init_biome_definitions(self):
        """
        Funktionsweise: Definiert alle verfügbaren Biome mit Eigenschaften
        Returns:
            dict: Biome-Definitionen mit Farben und Klassifizierungs-Regeln
        """
        return {
            0: {'name': 'Tiefsee', 'color': '#000080', 'min_height': 0.0, 'max_height': 0.10},
            1: {'name': 'Ozean', 'color': '#1e40af', 'min_height': 0.10, 'max_height': 0.15},
            2: {'name': 'Strand', 'color': '#fbbf24', 'min_height': 0.15, 'max_height': 0.25},
            3: {'name': 'Sumpf', 'color': '#65a30d', 'min_height': 0.15, 'max_height': 0.35},
            4: {'name': 'Grasland', 'color': '#84cc16', 'min_height': 0.25, 'max_height': 0.60},
            5: {'name': 'Wald', 'color': '#166534', 'min_height': 0.30, 'max_height': 0.70},
            6: {'name': 'Nadelwald', 'color': '#14532d', 'min_height': 0.40, 'max_height': 0.80},
            7: {'name': 'Steppe', 'color': '#ca8a04', 'min_height': 0.35, 'max_height': 0.65},
            8: {'name': 'Wüste', 'color': '#dc2626', 'min_height': 0.20, 'max_height': 0.60},
            9: {'name': 'Tundra', 'color': '#94a3b8', 'min_height': 0.30, 'max_height': 0.75},
            10: {'name': 'Alpin', 'color': '#e5e7eb', 'min_height': 0.75, 'max_height': 1.0},
            11: {'name': 'Gletscher', 'color': '#f3f4f6', 'min_height': 0.85, 'max_height': 1.0}
        }

    @performance_tracked("Biome_Final_Rendering")
    def _render_map(self, world_manager):
        """
        Funktionsweise: Rendert finale Welt-Karte mit allen Layern
        - Kombiniert alle Tab-Parameter für komplette Welt-Simulation
        - Multi-Layer Rendering: Terrain -> Biome -> Water -> Settlements
        """
        
        self.clear_and_setup()

        # Sammle alle Parameter von allen Tabs
        all_params = self._collect_all_parameters(world_manager)

        # === LAYER 1: TERRAIN & HEIGHT ===
        X, Y, height_field = self._generate_terrain(all_params)

        # === LAYER 2: CLIMATE & WEATHER ===
        temperature_field, precipitation_field = self._generate_climate_fields(X, Y, all_params)

        # === LAYER 3: BIOME CLASSIFICATION ===
        biome_map = self._classify_biomes(height_field, temperature_field, precipitation_field, all_params)

        # === LAYER 4: WATER SYSTEMS ===
        self._render_water_systems(X, Y, height_field, all_params)

        # === LAYER 5: SETTLEMENTS & FEATURES ===
        self._render_world_features(all_params)

        # === BIOME MAP RENDERING ===
        self._render_biome_map(X, Y, biome_map)

        # === FINAL TOUCHES ===
        self._add_world_statistics(all_params, biome_map)
        self._add_comprehensive_legend()

        self.set_title('Finale Welt: Alle Biome & Features integriert')
        self.draw()

    def _collect_all_parameters(self, world_manager):
        """Sammelt Parameter von allen Tabs für integrierte Simulation"""
        return {
            'terrain': world_manager.terrain.get_parameters(),
            'geology': world_manager.geology.get_parameters(),
            'settlement': world_manager.settlement.get_parameters(),
            'weather': world_manager.weather.get_parameters(),
            'water': world_manager.water.get_parameters()
        }

    def _generate_terrain(self, all_params):
        """
        Funktionsweise: Generiert integriertes Terrain basierend auf allen Parametern
        - Berücksichtigt Geologie für Terrain-Modifikation
        - Realistic Height Distribution
        """
        # Basis-Koordinaten
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        X, Y = np.meshgrid(x, y)

        # Terrain-Parameter
        terrain_params = all_params.get('terrain', {})
        np.random.seed(terrain_params.get('seed', 42))

        # Multi-Scale Terrain mit geologischen Einflüssen
        base_terrain = (np.sin(X * 0.08) * np.cos(Y * 0.06) +
                        np.sin(X * 0.12) * np.cos(Y * 0.10) * 0.5 +
                        np.random.normal(0, 0.08, X.shape))

        # Geologische Modifikation
        geology_params = all_params.get('geology', {})
        rock_hardness = np.mean(geology_params.get('hardness_values', [50])) / 100.0
        ridge_effect = geology_params.get('ridge_warping', 0.25)

        # Härtere Gesteine = steilere Berge
        height_field = base_terrain * (0.8 + rock_hardness * 0.4)
        height_field += np.sin(X * 0.05 + ridge_effect) * np.cos(Y * 0.05) * 0.2

        # Normalisierung
        height_field = (height_field + 2) / 4  # 0-1 Range

        return X, Y, height_field

    def _generate_climate_fields(self, X, Y, all_params):
        """Generiert Temperatur- und Niederschlags-Felder"""
        weather_params = all_params.get('weather', {})

        # Temperatur: Nord-Süd Gradient + Höhen-Effekt
        avg_temp = weather_params.get('avg_temperature', 15)
        lat_gradient = np.linspace(0.2, 1.0, 100)  # Kälter im Norden
        temperature_field = np.zeros_like(X)

        for i in range(100):
            temperature_field[i, :] = (avg_temp / 40.0) * lat_gradient[i]

        # Niederschlag: West-Ost Gradient + Topographie
        max_humidity = weather_params.get('max_humidity', 70) / 100.0
        rain_amount = weather_params.get('rain_amount', 5.0) / 10.0

        precipitation_field = np.zeros_like(X)
        for j in range(100):
            # Feuchte Luft von Westen
            precipitation_field[:, j] = max_humidity * (1.0 - j / 100.0 * 0.8) * rain_amount

        return temperature_field, precipitation_field

    def _classify_biomes(self, height_field, temperature_field, precipitation_field, all_params):
        """
        Funktionsweise: Intelligente Biome-Klassifizierung
        - Basiert auf Höhe, Temperatur, Niederschlag
        - Berücksichtigt Wassersystem-Parameter
        """
        biome_map = np.zeros_like(height_field, dtype=int)
        water_params = all_params.get('water', {})
        sea_level = water_params.get('sea_level', 15) / 100.0

        for i in range(100):
            for j in range(100):
                h = height_field[i, j]
                t = temperature_field[i, j]
                p = precipitation_field[i, j]

                # Biome-Klassifizierungs-Logik
                if h < sea_level * 0.5:  # Tiefsee
                    biome_map[i, j] = 0
                elif h < sea_level:  # Ozean
                    biome_map[i, j] = 1
                elif h < sea_level + 0.1 and i > 85:  # Strand (Küste)
                    biome_map[i, j] = 2
                elif h > 0.85:  # Hochgebirge
                    if t < 0.3:
                        biome_map[i, j] = 11  # Gletscher
                    else:
                        biome_map[i, j] = 10  # Alpin
                elif h > 0.75:  # Gebirge
                    if t < 0.4:
                        biome_map[i, j] = 9  # Tundra
                    else:
                        biome_map[i, j] = 6  # Nadelwald
                elif t < 0.25:  # Kalt
                    biome_map[i, j] = 9  # Tundra
                elif p < 0.25:  # Trocken
                    if t > 0.6:
                        biome_map[i, j] = 8  # Wüste
                    else:
                        biome_map[i, j] = 7  # Steppe
                elif p > 0.8 and h < 0.4:  # Sehr feucht, niedrig
                    biome_map[i, j] = 3  # Sumpf
                elif p > 0.5 and t > 0.4:  # Feucht und warm
                    biome_map[i, j] = 5  # Wald
                elif p > 0.4 and h > 0.4:  # Mittel feucht, höher
                    biome_map[i, j] = 6  # Nadelwald
                else:  # Standard
                    biome_map[i, j] = 4  # Grasland

        return biome_map

    def _render_biome_map(self, X, Y, biome_map):
        """Rendert die finale Biome-Karte"""
        # Custom Colormap aus Biome-Definitionen
        colors = [self.biome_definitions[i]['color'] for i in sorted(self.biome_definitions.keys())]
        biome_cmap = ListedColormap(colors)

        # Biome-Karte rendern
        self.ax.imshow(biome_map, cmap=biome_cmap,
                       vmin=0, vmax=len(self.biome_definitions) - 1,
                       origin='lower', extent=[0, 100, 0, 100], alpha=0.85)


    def _render_water_systems(self, X, Y, height_field, all_params):
        """Rendert integrierte Wassersysteme"""
        water_params = all_params.get('water', {})

        # Flüsse (vereinfacht)
        river_x = np.linspace(10, 90, 40)
        river_y = 50 + np.sin(river_x * 0.12) * 15
        water_speed = water_params.get('water_speed', 8.0)
        river_width = max(2, water_speed / 4.0)

        self.ax.plot(river_x, river_y, color='#1e40af',
                     linewidth=river_width, alpha=0.9, zorder=10)

        # Seen (basierend auf Topographie)
        lake_fill = water_params.get('lake_fill', 40)
        lake_positions = [(25, 75), (70, 60), (45, 30)]

        for i, (lx, ly) in enumerate(lake_positions[:int(lake_fill / 30 + 1)]):
            lake_size = lake_fill / 25.0
            circle = plt.Circle((lx, ly), lake_size, color='#3b82f6',
                                alpha=0.9, zorder=10)
            self.ax.add_patch(circle)

    def _render_world_features(self, all_params):
        """Rendert Settlements und weitere Features"""
        settlement_params = all_params.get('settlement', {})

        # Dörfer
        villages = settlement_params.get('villages', 3)
        village_positions = [(30, 65), (60, 40), (75, 75)]
        for i in range(min(villages, len(village_positions))):
            x, y = village_positions[i]
            self.ax.plot(x, y, marker='s', color='#8b4513',
                         markersize=8, alpha=0.9, zorder=15)
            self.ax.text(x, y + 3, f'Dorf {i + 1}', ha='center', va='bottom',
                         fontsize=8, weight='bold', zorder=16)

        # Landmarks
        landmarks = settlement_params.get('landmarks', 2)
        landmark_positions = [(40, 80), (80, 30)]
        for i in range(min(landmarks, len(landmark_positions))):
            x, y = landmark_positions[i]
            self.ax.plot(x, y, marker='^', color='#fbbf24',
                         markersize=10, alpha=0.9, zorder=15)
            self.ax.text(x, y + 4, f'Landmark {i + 1}', ha='center', va='bottom',
                         fontsize=8, weight='bold', zorder=16)

        # Straßen zwischen Settlements
        connections = settlement_params.get('connections', 3)
        for i in range(min(connections, 2)):
            start_pos = village_positions[i] if i < len(village_positions) else (50, 50)
            end_x, end_y = (100 if i % 2 == 0 else 0, start_pos[1])
            self.ax.plot([start_pos[0], end_x], [start_pos[1], end_y],
                         color='#7c2d12', linewidth=2, alpha=0.7, zorder=12)

    def _add_world_statistics(self, all_params, biome_map):
        """Fügt Welt-Statistiken zur Karte hinzu"""
        # Biome-Verteilung berechnen
        unique, counts = np.unique(biome_map, return_counts=True)
        total_pixels = biome_map.size

        # Top 3 Biome finden
        top_biomes = []
        for i in np.argsort(counts)[-3:]:
            biome_id = unique[i]
            percentage = (counts[i] / total_pixels) * 100
            biome_name = self.biome_definitions.get(biome_id, {}).get('name', f'Biome {biome_id}')
            top_biomes.append(f"{biome_name}: {percentage:.1f}%")

        # Statistik-Text
        stats_text = "WELT-ÜBERSICHT:\n" + "\n".join(reversed(top_biomes))

        self.ax.text(2, 98, stats_text, fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                     verticalalignment='top', zorder=20)

    def _add_comprehensive_legend(self):
        """Fügt umfassende Legende für finale Welt hinzu"""
        legend_elements = []

        # Biome-Legende (Top 6 häufigste)
        main_biomes = [1, 4, 5, 8, 9, 10]  # Ozean, Grasland, Wald, Wüste, Tundra, Alpin
        for biome_id in main_biomes:
            if biome_id in self.biome_definitions:
                biome = self.biome_definitions[biome_id]
                legend_elements.append(
                    mpatches.Patch(color=biome['color'], label=biome['name'])
                )

        # Features
        legend_elements.extend([
            plt.Line2D([0], [0], color='#1e40af', linewidth=3, label='Fluss'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3b82f6',
                       markersize=6, label='See'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#8b4513',
                       markersize=6, label='Dorf'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#fbbf24',
                       markersize=8, label='Landmark')
        ])

        self.ax.legend(handles=legend_elements, loc='center left',
                       bbox_to_anchor=(1.02, 0.5), fontsize=8,
                       framealpha=0.95, edgecolor='gray')


class BiomeControlPanel(QWidget, NavigationMixin):
    """
    Funktionsweise: Finale Control Panel für Biome Tab
    - Kein Parameter-Input (verwendet alle vorherigen Tab-Parameter)
    - Zeigt Welt-Statistiken und Export-Optionen
    - Finale Navigation (Fertigstellen/Exportieren)
    """

    def __init__(self, map_canvas):
        super().__init__()
        self.map_canvas = map_canvas

        # Verwende WorldParameterManager für finale Zusammenfassung
        self.world_manager = WorldParameterManager()

        self.init_ui()


    def init_ui(self):
        layout = QVBoxLayout()

        # Titel
        title = QLabel("Finale Welt-Generierung")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px; color: #2563eb;")
        layout.addWidget(title)

        # Welt-Zusammenfassung
        self.setup_world_summary(layout)

        # Generierung Controls
        self.setup_generation_controls(layout)

        # Export & Finalisierung
        self.setup_export_controls(layout)

        # Spacer
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Finale Navigation (Zurück/Fertigstellen)
        self.setup_navigation(layout, show_prev=True, show_next=True,
                              prev_text="Zurück", next_text="Fertigstellen")

        self.setLayout(layout)

        # Initial world rendering
        self.generate_final_world()

    def setup_world_summary(self, layout):
        """Erstellt Welt-Zusammenfassungs-Panel"""
        summary_group = QGroupBox("Welt-Zusammenfassung")
        summary_layout = QVBoxLayout()

        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(140)
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8fafc;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Segoe UI', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)

        self.update_world_summary()

        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

    def setup_generation_controls(self, layout):
        """Erstellt Generierungs-Control Panel"""
        gen_group = QGroupBox("Welt-Generierung")
        gen_layout = QVBoxLayout()

        self.regenerate_btn = QPushButton("Welt Neu Generieren")
        self.regenerate_btn.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; padding: 12px; }")
        self.regenerate_btn.clicked.connect(self.generate_final_world)

        self.randomize_all_btn = QPushButton("Alle Parameter Randomisieren")
        self.randomize_all_btn.setStyleSheet(
            "QPushButton { background-color: #7c3aed; color: white; font-weight: bold; padding: 12px; }")
        self.randomize_all_btn.clicked.connect(self.randomize_all_parameters)

        gen_layout.addWidget(self.regenerate_btn)
        gen_layout.addWidget(self.randomize_all_btn)
        gen_group.setLayout(gen_layout)
        layout.addWidget(gen_group)

    def setup_export_controls(self, layout):
        """Erstellt Export-Control Panel"""
        export_group = QGroupBox("Export & Speichern")
        export_layout = QVBoxLayout()

        self.export_image_btn = QPushButton("Als Bild Exportieren")
        self.export_image_btn.setStyleSheet(
            "QPushButton { background-color: #dc2626; color: white; font-weight: bold; padding: 10px; }")
        self.export_image_btn.clicked.connect(self.export_world_image)

        self.export_data_btn = QPushButton("Parameter Exportieren")
        self.export_data_btn.setStyleSheet(
            "QPushButton { background-color: #ea580c; color: white; font-weight: bold; padding: 10px; }")
        self.export_data_btn.clicked.connect(self.export_world_data)

        export_layout.addWidget(self.export_image_btn)
        export_layout.addWidget(self.export_data_btn)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

    @performance_tracked("Final_World_Generation")
    def generate_final_world(self):
        """
        Funktionsweise: Generiert finale Welt mit allen Tab-Parametern
        - Sammelt alle Parameter von allen Tabs
        - Triggert finale Welt-Visualisierung
        """

        # Aktualisiere Zusammenfassung
        self.update_world_summary()

        # Triggere finale Karten-Generierung
        self.map_canvas.update_map(world_manager=self.world_manager)

    def update_world_summary(self):
        """Aktualisiert Welt-Zusammenfassung mit allen Parametern"""
        all_params = self.world_manager.export_all_parameters()

        # Formatierte Zusammenfassung erstellen
        summary = "FINALE WELT-PARAMETER:\n\n"

        # Terrain
        terrain = all_params.get('terrain', {})
        summary += f"TERRAIN: {terrain.get('size', 256)}x{terrain.get('size', 256)}, "
        summary += f"Max.Höhe: {terrain.get('height', 100)}m, Seed: {terrain.get('seed', 42)}\n"

        # Weather
        weather = all_params.get('weather', {})
        climate = self.world_manager.weather.get_climate_classification()
        summary += f"KLIMA: {climate}, {weather.get('avg_temperature', 15)}°C, "
        summary += f"Feuchtigkeit: {weather.get('max_humidity', 70)}%\n"

        # Water
        water = all_params.get('water', {})
        water_coverage = self.world_manager.water.calculate_water_coverage()
        summary += f"WASSER: {water_coverage:.1f}% Coverage, "
        summary += f"Meereshöhe: {water.get('sea_level', 15)}%\n"

        # Settlements
        settlement = all_params.get('settlement', {})
        total_settlements = self.world_manager.settlement.get_total_settlements()
        summary += f"SIEDLUNGEN: {total_settlements} Total "
        summary += f"({settlement.get('villages', 3)} Dörfer, {settlement.get('landmarks', 2)} Landmarks)\n"

        # Geology
        geology = all_params.get('geology', {})
        rock_types = len(geology.get('rock_types', []))
        summary += f"GEOLOGIE: {rock_types} Gesteinsarten\n\n"

        summary += "Bereit für finale Generierung!"

        self.summary_text.setPlainText(summary)

    def randomize_all_parameters(self):
        """Randomisiert alle Parameter für zufällige Welt-Generierung"""
        # Randomisiere Terrain Seed
        self.world_manager.terrain.randomize_seed()

        # TODO: Weitere Randomisierung für andere Parameter
        # (würde in echter Implementation alle Parameter zufällig setzen)

        self.generate_final_world()

    def export_world_image(self):
        """Exportiert finale Welt als Bild"""
        pass

    def export_world_data(self):
        """Exportiert alle Parameter als JSON/XML"""
        pass

    # Navigation Methoden (von NavigationMixin erforderlich)
    def next_menu(self):
        """Finale Fertigstellung - Zurück zum Hauptmenü"""
        TabNavigationHelper.go_to_main_menu(self.window(), self.world_manager)

    def prev_menu(self):
        """Wechselt zum vorherigen Tab (Water)"""
        prev_tab = TabNavigationHelper.get_prev_tab('BiomeWindow')
        if prev_tab:
            self.navigate_to_tab(prev_tab[0], prev_tab[1])


class BiomeWindow(QMainWindow):
    """
    Funktionsweise: Hauptfenster für Biome-Tab (Finale Version)
    - Verwendet optimierte Map Canvas für komplexe Multi-Layer Rendering
    - Erweiterte Fenster-Konfiguration für finale Welt-Anzeige
    """

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Finale Biom-Klassifizierung")
        self.setGeometry(100, 100, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()

        # Linke Seite - Finale Welt-Karte (75%)
        self.map_canvas = BiomeMapCanvas()
        main_layout.addWidget(self.map_canvas, 7)

        # Trennlinie
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Rechte Seite - Finale Controls (25%)
        self.control_panel = BiomeControlPanel(self.map_canvas)
        self.control_panel.setMaximumWidth(380)  # Etwas breiter für finale Controls
        main_layout.addWidget(self.control_panel, 3)

        central_widget.setLayout(main_layout)

        # Finale Styling für Biome Tab
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8fafc, stop:1 #e2e8f0);
            }
            QLabel {
                color: #1e293b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cbd5e1;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 12px;
                background-color: rgba(255, 255, 255, 0.8);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #334155;
            }
            /* Spezielle Styles für finale Biome Gruppen */
            QGroupBox[title*="Welt-Zusammenfassung"] {
                border-color: #3b82f6;
                background-color: rgba(59, 130, 246, 0.05);
            }
            QGroupBox[title*="Welt-Generierung"] {
                border-color: #059669;
                background-color: rgba(5, 150, 105, 0.05);
            }
            QGroupBox[title*="Export"] {
                border-color: #dc2626;
                background-color: rgba(220, 38, 38, 0.05);
            }
            /* Button Hover-Effekte */
            QPushButton:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            /* Finale Welt Titel-Style */
            QLabel[text*="Finale Welt"] {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3b82f6, stop:1 #8b5cf6);
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
        """)

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Cleanup beim Schließen"""
        try:
            if hasattr(self, 'map_canvas'):
                self.map_canvas.cleanup()
        except Exception as e:
            print(f"Cleanup Fehler: {e}")
        finally:
            # Wichtig: Event an Parent weiterleiten
            super().closeEvent(event)