"""""
# =============================================================================
# Legacy-Kompatibilität für bestehende Imports
# =============================================================================

# Alle ursprünglichen Klassen bleiben für Rückwärts-Kompatibilität verfügbar
# Sie delegieren intern an die neuen BaseGenerator-Implementierungen

# TerrainSuitabilityAnalyzer - bereits neu implementiert
# PathfindingSystem - bereits neu implementiert  
# CivilizationInfluenceMapper - bereits neu implementiert
# PlotNodeSystem - bereits neu implementiert
# Location, PlotNode, Plot - bereits neu implementiert

# Für bestehenden Code der direkt auf diese Klassen zugreift:
Path: core/settlement_generator.py

Funktionsweise: Intelligente Settlement-Platzierung mit BaseGenerator-Integration und LOD-System
- BaseGenerator-Integration mit einheitlicher API und LOD-System
- Terrain-Suitability Analysis (Steigung, Höhe, Wasser-Nähe)
- Locations: 
    Settlements: Städte oder Dörfer die an bestimmten Orten vorkommen können (Täler, flache Hügel). Settlements verringern die Terrainverformung in der Nähe etwas. Je nach Radius (Siedlungsgröße) ist der Einfluss auf die Umgebung größer/kleiner. Die Form der Stadt soll zB Linsenförmig sein und über die Slopemap erzeugt werden. Zwischen Settlements gibt es einen Minimalabstand je nach map_size und Anzahl von Settlements. Innerhalb der Stadtgrenzen ist civ_map = 1, außerhalb nimmt der Einfluss ab.
    Roads: Nachdem Settlements entstanden sind werden die ersten Wege zwischen den Ortschaften geplottet. Dazu soll der Weg des geringsten Widerstands gefunden werden (Pathfinding via slopemap-cost). Danach werden die Straßen etwas gebogen über sanfte Splineinterpolation zwischen zB jedem 3.Waypoint. Erzeugen sehr geringen Einfluss entlang der Wege (z.B. 0.3).
    Roadsites: z.B. Taverne, Handelsposten, Wegschrein, Zollhaus, Galgenplatz, Markt, besondere Industrie. Entstehen in einem Bereich von 30%-70% Weglänge zwischen Settlements entlang von Roads. Der civ_map-Einfluss ist wesentlich geringer als der von Städten.
    Landmarks: z.B. Burgen, Kloster, mystische Stätte etc. entstehen in Regionen mit einem civ_map value < thresholds (landmark_wilderness). Erzeugen einen ähnlich geringen Einfluss wie Roadsites. Außerdem werden beide nur in niedrigeren Höhen und Slopes generiert.
    Wilderness: Bereiche unterhalb eines civ_map-Werts unterhalb von 0.2 werden genullt und als Wilderness deklariert. Hier spawnen keine Plotnodes. Hier sollen in der  späteren Spielentwicklung Questevents stattfinden.
    civ_map-Logik: civ_map wird mit 0.0 initialisiert. Jeder Quellpunkt trägt akkumulativ zum civ-Wert bei. Einflussverteilung um Quellpunkt über radialen Decay-Kernel (z.B. Gauß, linear fallend oder benutzerdefinierte Kurve). Decay ist stärker an Hanglagen, so dass Zivilisation nicht auf Berge reicht. Decayradius und Initialwert abhängig von Location-Typ: Stadt-Grenzpunkte starten bei 0.8 (innerhalb der Stadt ist 1.0), Roadwaypoints addieren 0.2 bis max. 0.5, Roadsite/Landmarks 0.4. Optional bei sehr hohen Berechnungzeiten kann die Einflussverteilung mit GPU-Shadermasken erfolgen.
    Plotnodes: Es wird eine feste Anzahl an Plotnodes generiert (plotnodes-parameter). Gleichmäßige Verteilung auf alle Bereiche außerhalb von Städten und Wilderness. Die Plotnodes verbinden sich mit mit Nachbarnodes über Delaunay-Triangulation. Dann verbinden sich die Delaunay-Dreiecke mit benachbarten Dreiecken zu Grundstücken. Die Plotnode-Civwerte werden zusammengerechnet und wenn sie einen Wert (plot-size-parameter) überschreiten ist die Größe erreicht. So werden Grundstücke in Region mit hohem Civ-wert kleiner. Über Abstoßungslogik können die Nodes "physisch" umarrangiert werden. Kanten mit geringem Winkel sollen sich glätten und die Zwischenpunkte können verschwinden. Sehr spitze Winkel lockern sich ebenso. Plotnode-Eigenschaften:
        node_id, node_location, connector_id (list of nodes), connector_distance (x,y entfernung), connector_elevation (akkumulierter höhenunterschied zu connector), connector_movecost (movecost abhängig von biomes)
    Plots: Plots bestehend aus Plotnodes haben folgende Eigenschaften:
        biome_amount: akkumulierte Menge eines jeden Bioms in den Grenzen des Plots
        resource_amount: später im Spiel sich verändernde Menge an natürlichen Rohstoffen.
        plot_area: Größe des Plots
        plot_distance: Anzahl der Nodepunkte*Distance Entfernung zu

Parameter Input (aus value_default.py SETTLEMENT):
- settlements, landmarks, roadsites, plotnodes: number of each type
- civ_influence_decay: Influence around Locationtypes decays of distance
- terrain_factor_villages: terrain influence on settlement suitability
- road_slope_to_distance_ratio: rather short roads or steep roads
- landmark_wilderness: wilderness area size by changing cutoff-threshold
- plotsize: how much accumulated civ-value to form plot

data_manager Input:
- map_seed (Globaler Karten-Seed für reproduzierbare Settlement-Platzierung)
- heightmap (2D-Array in meter Altitude) - REQUIRED
- slopemap (2D-Array in m/m mit dz/dx, dz/dy) - REQUIRED
- water_map (2D-Array mit Wasser-Klassifikation) - REQUIRED
- biome_map (2D-Array mit Biom-Indices) - OPTIONAL (Fallback: Höhen-basiert)

Output:
- settlement_list (List[Location] - Alle Settlements)
- landmark_list (List[Location] - Alle Landmarks)
- roadsite_list (List[Location] - Alle Roadsites)
- plot_map (2D-Array mit Plot-IDs)
- civ_map (2D-Array mit Zivilisations-Einfluss)

Klassen:
SettlementGenerator (BaseGenerator)
    Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung mit BaseGenerator-API und LOD-System
    Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map mit Progress-Updates
    Methoden: generate(), _execute_generation(), _load_default_parameters(), _get_dependencies()

TerrainSuitabilityAnalyzer
    Funktionsweise: Analysiert Terrain-Eignung für Settlements basierend auf Steigung, Höhe, Wasser-Nähe
    Aufgabe: Erstellt Suitability-Map für optimale Settlement-Platzierung
    Methoden: analyze_slope_suitability(), calculate_water_proximity(), evaluate_elevation_fitness()

PathfindingSystem   
    Funktionsweise: Findet Wege geringsten Widerstands zwischen Settlements für Straßen
    Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation und LOD-Optimierung
    Methoden: find_least_resistance_path(), apply_spline_smoothing(), calculate_movement_cost()

CivilizationInfluenceMapper    
    Funktionsweise: Berechnet civ_map durch radialen Decay von Settlement/Road/Landmark-Punkten
    Aufgabe: Erstellt realistische Zivilisations-Verteilung mit Decay-Kernels
    Methoden: apply_settlement_influence(), calculate_road_influence(), apply_decay_kernel()

PlotNodeSystem    
    Funktionsweise: Generiert Plotnodes mit Delaunay-Triangulation und Grundstücks-Bildung
    Aufgabe: Erstellt Grundstücks-System für späteres Gameplay mit LOD-abhängiger Dichte
    Methoden: generate_plot_nodes(), create_delaunay_triangulation(), merge_to_plots(), optimize_node_positions()
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
import heapq
from collections import namedtuple, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
from core.base_generator import BaseGenerator


class SettlementData:
    """
    Funktionsweise: Container für alle Settlement-Daten mit LOD-Informationen
    Aufgabe: Speichert settlement_list, landmark_list, roadsite_list, plot_map, civ_map und Metainformationen
    """
    def __init__(self):
        self.settlement_list = []       # List[Location] - Alle Settlements
        self.landmark_list = []         # List[Location] - Alle Landmarks  
        self.roadsite_list = []         # List[Location] - Alle Roadsites
        self.plot_map = None           # (height, width) - Plot-IDs
        self.civ_map = None            # (height, width) - Zivilisations-Einfluss
        self.plot_nodes = []           # List[PlotNode] - Alle PlotNodes
        self.plots = []                # List[Plot] - Alle Plots
        self.roads = []                # List[List[Tuple]] - Alle Road-Pfade
        self.lod_level = "LOD64"       # Aktueller LOD-Level
        self.actual_size = 64          # Tatsächliche Kartengröße
        self.parameters = {}           # Verwendete Parameter für Cache-Management


@dataclass
class Location:
    """
    Funktionsweise: Datenstruktur für alle Arten von Locations (Settlements, Landmarks, Roadsites)
    Aufgabe: Einheitliche Repräsentation aller Siedlungs-Objekte
    """
    location_id: int
    x: float
    y: float
    location_type: str  # 'settlement', 'landmark', 'roadsite'
    radius: float
    civ_influence: float
    properties: Dict = None


@dataclass
class PlotNode:
    """
    Funktionsweise: Datenstruktur für Plotnodes mit allen Verbindungs-Informationen
    Aufgabe: Repräsentiert einzelne Nodes im Grundstücks-System
    """
    node_id: int
    node_location: Tuple[float, float]
    connector_id: List[int]
    connector_distance: List[float]
    connector_elevation: List[float]
    connector_movecost: List[float]


@dataclass
class Plot:
    """
    Funktionsweise: Datenstruktur für Plots mit allen Eigenschaften
    Aufgabe: Repräsentiert Grundstücke bestehend aus PlotNodes
    """
    plot_id: int
    nodes: List[PlotNode]
    biome_amount: Dict[str, float]
    resource_amount: Dict[str, float]
    plot_area: float
    plot_distance: float


class TerrainSuitabilityAnalyzer:
    """
    Funktionsweise: Analysiert Terrain-Eignung für Settlements basierend auf Steigung, Höhe, Wasser-Nähe
    Aufgabe: Erstellt Suitability-Map für optimale Settlement-Platzierung mit LOD-Optimierung
    """

    def __init__(self, terrain_factor_villages=1.0, lod_level="LOD64"):
        """
        Funktionsweise: Initialisiert Suitability-Analyzer mit Terrain-Gewichtungsfaktor und LOD
        Aufgabe: Setup der Terrain-Analyse-Parameter mit LOD-Anpassung
        Parameter: terrain_factor_villages (float), lod_level (str) - Gewichtung und LOD-Level
        """
        self.terrain_factor = terrain_factor_villages
        self.lod_level = lod_level
        
        # LOD-abhängige Optimierungen
        self.lod_optimizations = {
            "LOD64": {"analysis_detail": 0.5, "max_distance_check": 20},
            "LOD128": {"analysis_detail": 0.7, "max_distance_check": 30},
            "LOD256": {"analysis_detail": 1.0, "max_distance_check": 40},
            "FINAL": {"analysis_detail": 1.0, "max_distance_check": 50}
        }

    def analyze_slope_suitability(self, slopemap, progress_callback=None):
        """
        Funktionsweise: Bewertet Terrain-Eignung basierend auf Slope-Steilheit mit Progress-Updates
        Aufgabe: Erstellt Slope-Suitability-Map für Settlement-Platzierung
        Parameter: slopemap (numpy.ndarray), progress_callback - Slope-Daten und Progress-Callback
        Returns: numpy.ndarray - Slope-Suitability zwischen 0 (ungeeignet) und 1 (ideal)
        """
        if progress_callback:
            progress_callback("Terrain Analysis", 5, "Analyzing slope suitability...")

        height, width = slopemap.shape[:2]
        slope_suitability = np.zeros((height, width), dtype=np.float32)
        
        # LOD-abhängige Detailgrad
        detail_level = self.lod_optimizations[self.lod_level]["analysis_detail"]

        for y in range(height):
            for x in range(width):
                # Slope-Magnitude berechnen
                dz_dx = slopemap[y, x, 0]
                dz_dy = slopemap[y, x, 1]
                slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

                # Optimal: flache Bereiche (slope < 0.1)
                # Akzeptabel: leichte Steigung (slope < 0.5)
                # Ungeeignet: steile Hänge (slope > 1.0)
                if slope_magnitude < 0.1:
                    slope_suitability[y, x] = 1.0
                elif slope_magnitude < 0.5:
                    slope_suitability[y, x] = 1.0 - (slope_magnitude - 0.1) / 0.4 * 0.5
                elif slope_magnitude < 1.0:
                    slope_suitability[y, x] = 0.5 - (slope_magnitude - 0.5) / 0.5 * 0.5
                else:
                    slope_suitability[y, x] = 0.0

                # LOD-Anpassung: Weniger Detail bei niedrigen LODs
                if detail_level < 1.0:
                    slope_suitability[y, x] = np.round(slope_suitability[y, x] / detail_level) * detail_level

        return slope_suitability

    def calculate_water_proximity(self, water_map, progress_callback=None):
        """
        Funktionsweise: Berechnet Eignung basierend auf Nähe zu Wasserquellen mit LOD-Optimierung
        Aufgabe: Erstellt Water-Proximity-Suitability für Settlement-Platzierung
        Parameter: water_map (numpy.ndarray), progress_callback - Wasser-Daten und Progress-Callback
        Returns: numpy.ndarray - Water-Proximity-Suitability
        """
        if progress_callback:
            progress_callback("Terrain Analysis", 10, "Calculating water proximity...")

        height, width = water_map.shape
        water_suitability = np.zeros((height, width), dtype=np.float32)
        
        # LOD-abhängige Maximal-Distanz
        max_distance = self.lod_optimizations[self.lod_level]["max_distance_check"]

        # Finde alle Wasser-Pixel
        water_pixels = np.where(water_map > 0)

        for y in range(height):
            for x in range(width):
                if len(water_pixels[0]) == 0:
                    water_suitability[y, x] = 0.0
                    continue

                # Minimale Distanz zu Wasser berechnen (mit LOD-Limit)
                distances = np.sqrt((water_pixels[1] - x) ** 2 + (water_pixels[0] - y) ** 2)
                min_distance = np.min(distances)
                
                # Bei niedrigen LODs: früher abbrechen
                if min_distance > max_distance:
                    water_suitability[y, x] = 0.0
                    continue

                # Optimal: 2-10 Pixel Entfernung
                # Akzeptabel: bis 20 Pixel
                # Ungeeignet: > 30 Pixel oder direkt auf Wasser
                if min_distance == 0:
                    water_suitability[y, x] = 0.0  # Direkt auf Wasser
                elif min_distance < 2:
                    water_suitability[y, x] = min_distance / 2.0  # Zu nah
                elif min_distance <= 10:
                    water_suitability[y, x] = 1.0  # Optimal
                elif min_distance <= 20:
                    water_suitability[y, x] = 1.0 - (min_distance - 10) / 10 * 0.5
                elif min_distance <= 30:
                    water_suitability[y, x] = 0.5 - (min_distance - 20) / 10 * 0.5
                else:
                    water_suitability[y, x] = 0.0

        return water_suitability

    def evaluate_elevation_fitness(self, heightmap, progress_callback=None):
        """
        Funktionsweise: Bewertet Terrain-Eignung basierend auf Höhenlage
        Aufgabe: Erstellt Elevation-Suitability für Settlement-Platzierung
        Parameter: heightmap (numpy.ndarray), progress_callback - Höhendaten und Progress-Callback
        Returns: numpy.ndarray - Elevation-Suitability
        """
        if progress_callback:
            progress_callback("Terrain Analysis", 15, "Evaluating elevation fitness...")

        height, width = heightmap.shape
        elevation_suitability = np.zeros((height, width), dtype=np.float32)

        # Höhen-Statistiken
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            elevation_suitability.fill(1.0)
            return elevation_suitability

        for y in range(height):
            for x in range(width):
                # Normalisierte Höhe [0, 1]
                norm_height = (heightmap[y, x] - min_height) / height_range

                # Optimal: 20-60% der Höhenrange (Täler und niedrige Hügel)
                # Akzeptabel: bis 80%
                # Ungeeignet: > 80% (hohe Berge) oder < 10% (Sümpfe/Seen)
                if norm_height < 0.1:
                    elevation_suitability[y, x] = norm_height / 0.1 * 0.3
                elif norm_height <= 0.2:
                    elevation_suitability[y, x] = 0.3 + (norm_height - 0.1) / 0.1 * 0.4
                elif norm_height <= 0.6:
                    elevation_suitability[y, x] = 1.0  # Optimal
                elif norm_height <= 0.8:
                    elevation_suitability[y, x] = 1.0 - (norm_height - 0.6) / 0.2 * 0.5
                else:
                    elevation_suitability[y, x] = 0.5 - (norm_height - 0.8) / 0.2 * 0.5

        return elevation_suitability

    def create_combined_suitability(self, heightmap, slopemap, water_map, progress_callback=None):
        """
        Funktionsweise: Kombiniert alle Suitability-Faktoren zu finaler Suitability-Map
        Aufgabe: Erstellt finale Settlement-Suitability durch gewichtete Kombination
        Parameter: heightmap, slopemap, water_map, progress_callback - Alle Terrain-Daten und Progress
        Returns: numpy.ndarray - Kombinierte Suitability-Map
        """
        slope_suit = self.analyze_slope_suitability(slopemap, progress_callback)
        water_suit = self.calculate_water_proximity(water_map, progress_callback)
        elevation_suit = self.evaluate_elevation_fitness(heightmap, progress_callback)

        # Gewichtete Kombination
        weights = {
            'slope': 0.4 * self.terrain_factor,
            'water': 0.35,
            'elevation': 0.25 * self.terrain_factor
        }

        combined_suitability = (
                slope_suit * weights['slope'] +
                water_suit * weights['water'] +
                elevation_suit * weights['elevation']
        )

        # Normalisierung auf [0, 1]
        max_possible = sum(weights.values())
        if max_possible > 0:
            combined_suitability /= max_possible

        return combined_suitability


class PathfindingSystem:
    """
    Funktionsweise: Findet Wege geringsten Widerstands zwischen Settlements für Straßen
    Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation und LOD-Optimierung
    """

    def __init__(self, road_slope_to_distance_ratio=1.0, lod_level="LOD64"):
        """
        Funktionsweise: Initialisiert Pathfinding-System mit Slope-Distance-Gewichtung und LOD
        Aufgabe: Setup der Pathfinding-Parameter mit LOD-Anpassung
        Parameter: road_slope_to_distance_ratio (float), lod_level (str) - Gewichtung und LOD
        """
        self.slope_distance_ratio = road_slope_to_distance_ratio
        self.lod_level = lod_level
        
        # LOD-abhängige Pathfinding-Optimierungen
        self.lod_settings = {
            "LOD64": {"max_search_nodes": 500, "path_resolution": 2},
            "LOD128": {"max_search_nodes": 1000, "path_resolution": 1},
            "LOD256": {"max_search_nodes": 2000, "path_resolution": 1},
            "FINAL": {"max_search_nodes": 5000, "path_resolution": 1}
        }

    def calculate_movement_cost(self, slopemap, x, y):
        """
        Funktionsweise: Berechnet Bewegungskosten für einzelnen Punkt basierend auf Slope
        Aufgabe: Kostenfunktion für A*-Pathfinding
        Parameter: slopemap, x, y - Slope-Daten und Koordinaten
        Returns: float - Bewegungskosten für diesen Punkt
        """
        height, width = slopemap.shape[:2]

        if x < 0 or x >= width or y < 0 or y >= height:
            return float('inf')

        # Slope-Magnitude berechnen
        dz_dx = slopemap[y, x, 0]
        dz_dy = slopemap[y, x, 1]
        slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        # Basis-Kosten: 1.0 + Slope-Penalty
        base_cost = 1.0
        slope_penalty = slope_magnitude * self.slope_distance_ratio

        return base_cost + slope_penalty

    def find_least_resistance_path(self, slopemap, start_pos, end_pos, progress_callback=None):
        """
        Funktionsweise: A*-Pathfinding für Weg geringsten Widerstands zwischen zwei Punkten mit LOD-Optimierung
        Aufgabe: Findet optimalen Straßenverlauf zwischen Settlements
        Parameter: slopemap, start_pos, end_pos, progress_callback - Slope-Daten, Positionen und Progress
        Returns: List[Tuple] - Wegpunkte vom Start zum Ziel
        """
        height, width = slopemap.shape[:2]
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        end_x, end_y = int(end_pos[0]), int(end_pos[1])
        
        # LOD-Einstellungen
        max_nodes = self.lod_settings[self.lod_level]["max_search_nodes"]
        path_resolution = self.lod_settings[self.lod_level]["path_resolution"]

        # A*-Datenstrukturen
        open_set = [(0, start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self._heuristic((start_x, start_y), (end_x, end_y))}
        
        nodes_explored = 0

        while open_set and nodes_explored < max_nodes:
            current_f, current_x, current_y = heapq.heappop(open_set)
            nodes_explored += 1

            if current_x == end_x and current_y == end_y:
                # Pfad rekonstruieren
                path = []
                while (current_x, current_y) in came_from:
                    path.append((current_x, current_y))
                    current_x, current_y = came_from[(current_x, current_y)]
                path.append((start_x, start_y))
                return list(reversed(path))

            # Nachbarn prüfen (8-Connectivity mit LOD-Resolution)
            for dx in range(-path_resolution, path_resolution + 1, path_resolution):
                for dy in range(-path_resolution, path_resolution + 1, path_resolution):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_x = current_x + dx
                    neighbor_y = current_y + dy

                    if (neighbor_x < 0 or neighbor_x >= width or
                            neighbor_y < 0 or neighbor_y >= height):
                        continue

                    # Bewegungskosten berechnen
                    movement_cost = self.calculate_movement_cost(slopemap, neighbor_x, neighbor_y)
                    if movement_cost == float('inf'):
                        continue

                    # Diagonale Bewegung kostet mehr
                    if dx != 0 and dy != 0:
                        movement_cost *= 1.414

                    tentative_g_score = g_score.get((current_x, current_y), float('inf')) + movement_cost

                    if tentative_g_score < g_score.get((neighbor_x, neighbor_y), float('inf')):
                        came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                        g_score[(neighbor_x, neighbor_y)] = tentative_g_score
                        f_score[(neighbor_x, neighbor_y)] = tentative_g_score + self._heuristic(
                            (neighbor_x, neighbor_y), (end_x, end_y))
                        heapq.heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))

        # Kein Pfad gefunden oder Node-Limit erreicht - direkte Linie als Fallback
        if progress_callback:
            progress_callback("Road Building", 30, f"Pathfinding fallback after {nodes_explored} nodes")
        
        return [(start_x, start_y), (end_x, end_y)]

    def _heuristic(self, pos1, pos2):
        """
        Funktionsweise: Heuristik-Funktion für A*-Algorithmus
        Aufgabe: Schätzt Kosten vom aktuellen Punkt zum Ziel
        Parameter: pos1, pos2 - Aktuelle und Ziel-Position
        Returns: float - Geschätzte Kosten
        """
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def apply_spline_smoothing(self, path, smoothing_factor=3, progress_callback=None):
        """
        Funktionsweise: Wendet Spline-Interpolation auf Pfad an für sanfte Straßenführung
        Aufgabe: Glättet Straßenverlauf zwischen Wegpunkten
        Parameter: path (List[Tuple]), smoothing_factor (int), progress_callback - Pfad, Glättung und Progress
        Returns: List[Tuple] - Geglätteter Pfad
        """
        if len(path) < 4:
            return path

        # LOD-abhängige Spline-Qualität
        if self.lod_level == "LOD64":
            smoothing_factor = max(5, smoothing_factor)  # Weniger Punkte bei LOD64

        # Nur jeden N-ten Punkt für Spline verwenden
        control_points = path[::smoothing_factor]
        if path[-1] not in control_points:
            control_points.append(path[-1])

        if len(control_points) < 3:
            return path

        # Koordinaten extrahieren
        x_coords = [p[0] for p in control_points]
        y_coords = [p[1] for p in control_points]

        try:
            # Spline interpolieren
            tck, u = splprep([x_coords, y_coords], s=0)

            # Neue Punkte entlang Spline generieren
            u_new = np.linspace(0, 1, len(path))
            smoothed_coords = splev(u_new, tck)

            smoothed_path = [(int(x), int(y)) for x, y in zip(smoothed_coords[0], smoothed_coords[1])]
            return smoothed_path
        except:
            # Fallback bei Spline-Fehlern
            return path


class CivilizationInfluenceMapper:
    """
    Funktionsweise: Berechnet civ_map durch radialen Decay von Settlement/Road/Landmark-Punkten
    Aufgabe: Erstellt realistische Zivilisations-Verteilung mit Decay-Kernels
    """

    def __init__(self, civ_influence_decay=1.0):
        """
        Funktionsweise: Initialisiert Civilization-Influence-Mapper mit Decay-Parameter
        Aufgabe: Setup der Zivilisations-Einfluss-Berechnung
        Parameter: civ_influence_decay (float) - Stärke des Einfluss-Abfalls mit Distanz
        """
        self.decay_factor = civ_influence_decay

    def apply_settlement_influence(self, civ_map, settlements, slopemap, progress_callback=None):
        """
        Funktionsweise: Wendet Settlement-Einfluss auf civ_map an mit radialem Decay
        Aufgabe: Berechnet Zivilisations-Einfluss von Städten und Dörfern
        Parameter: civ_map, settlements, slopemap, progress_callback - Civ-Map, Settlement-Liste, Slope-Daten und Progress
        Returns: numpy.ndarray - Aktualisierte civ_map
        """
        if progress_callback:
            progress_callback("Civilization Mapping", 55, f"Applying influence for {len(settlements)} settlements...")

        height, width = civ_map.shape

        for i, settlement in enumerate(settlements):
            center_x = int(settlement.x)
            center_y = int(settlement.y)
            radius = settlement.radius
            base_influence = settlement.civ_influence

            # Einflussbereich berechnen
            for y in range(max(0, center_y - int(radius) - 5), min(height, center_y + int(radius) + 6)):
                for x in range(max(0, center_x - int(radius) - 5), min(width, center_x + int(radius) + 6)):
                    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                    if distance <= radius:
                        # Innerhalb Stadt: maximaler Einfluss (1.0)
                        civ_map[y, x] = max(civ_map[y, x], 1.0)
                    else:
                        # Außerhalb: radialer Decay mit Slope-Modifikation
                        slope_modifier = self._calculate_slope_decay_modifier(slopemap, x, y)
                        decay_distance = (distance - radius) * self.decay_factor * slope_modifier

                        if decay_distance < radius * 2:
                            influence = base_influence * np.exp(-decay_distance / radius)
                            civ_map[y, x] = max(civ_map[y, x], influence)

            # Progress-Update pro Settlement-Batch
            if progress_callback and (i + 1) % max(1, len(settlements) // 4) == 0:
                progress = 55 + (i + 1) * 10 // len(settlements)
                progress_callback("Civilization Mapping", progress, f"Processed {i + 1}/{len(settlements)} settlements")

        return civ_map

    def calculate_road_influence(self, civ_map, roads, slopemap, progress_callback=None):
        """
        Funktionsweise: Wendet Road-Einfluss auf civ_map an entlang der Straßenverläufe
        Aufgabe: Berechnet Zivilisations-Einfluss entlang von Straßen
        Parameter: civ_map, roads, slopemap, progress_callback - Civ-Map, Straßen-Pfade, Slope-Daten und Progress
        Returns: numpy.ndarray - Aktualisierte civ_map
        """
        if progress_callback:
            progress_callback("Civilization Mapping", 65, f"Applying road influence for {len(roads)} roads...")

        road_influence = 0.2
        max_road_civ = 0.5
        road_width = 2

        for road_idx, road in enumerate(roads):
            for point in road:
                x, y = int(point[0]), int(point[1])

                # Einfluss um Straßenpunkt
                for dy in range(-road_width, road_width + 1):
                    for dx in range(-road_width, road_width + 1):
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < civ_map.shape[1] and 0 <= ny < civ_map.shape[0]:
                            distance = np.sqrt(dx ** 2 + dy ** 2)
                            if distance <= road_width:
                                influence = road_influence * (1 - distance / road_width)
                                new_value = min(max_road_civ, civ_map[ny, nx] + influence)
                                civ_map[ny, nx] = new_value

        return civ_map

    def apply_decay_kernel(self, civ_map, location, influence_value, radius, slopemap):
        """
        Funktionsweise: Wendet radialen Decay-Kernel um einzelne Location an
        Aufgabe: Generische Einfluss-Verteilung für beliebige Locations
        Parameter: civ_map, location, influence_value, radius, slopemap
        Returns: numpy.ndarray - Aktualisierte civ_map
        """
        height, width = civ_map.shape
        center_x, center_y = int(location.x), int(location.y)

        for y in range(max(0, center_y - int(radius) - 2), min(height, center_y + int(radius) + 3)):
            for x in range(max(0, center_x - int(radius) - 2), min(width, center_x + int(radius) + 3)):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                if distance <= radius:
                    # Slope-Modifikation
                    slope_modifier = self._calculate_slope_decay_modifier(slopemap, x, y)
                    decay_distance = distance * self.decay_factor * slope_modifier

                    influence = influence_value * np.exp(-decay_distance / radius)
                    civ_map[y, x] = max(civ_map[y, x], influence)

        return civ_map

    def _calculate_slope_decay_modifier(self, slopemap, x, y):
        """
        Funktionsweise: Berechnet Slope-basierten Modifier für Einfluss-Decay
        Aufgabe: Verstärkt Decay an Hanglagen, so dass Zivilisation nicht auf Berge reicht
        Parameter: slopemap, x, y - Slope-Daten und Koordinaten
        Returns: float - Slope-Decay-Modifier (>1 = stärkerer Decay)
        """
        height, width = slopemap.shape[:2]

        if x < 0 or x >= width or y < 0 or y >= height:
            return 2.0  # Starker Decay außerhalb der Map

        # Slope-Magnitude berechnen
        dz_dx = slopemap[y, x, 0]
        dz_dy = slopemap[y, x, 1]
        slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        # Modifier: 1.0 (flach) bis 3.0 (sehr steil)
        slope_modifier = 1.0 + slope_magnitude * 2.0
        return min(3.0, slope_modifier)


class PlotNodeSystem:
    """
    Funktionsweise: Generiert Plotnodes mit Delaunay-Triangulation und Grundstücks-Bildung
    Aufgabe: Erstellt Grundstücks-System für späteres Gameplay mit LOD-abhängiger Dichte
    """

    def __init__(self, plotsize=1.0, lod_level="LOD64"):
        """
        Funktionsweise: Initialisiert PlotNode-System mit Plot-Größen-Parameter und LOD
        Aufgabe: Setup der Grundstücks-Generierung mit LOD-Anpassung
        Parameter: plotsize (float), lod_level (str) - Akkumulierter Civ-Wert für Plot-Größe und LOD
        """
        self.plotsize_threshold = plotsize
        self.lod_level = lod_level
        self.next_node_id = 0
        self.next_plot_id = 0
        
        # LOD-abhängige PlotNode-Dichte
        self.lod_density_factors = {
            "LOD64": 0.3,    # 30% der gewünschten Nodes
            "LOD128": 0.6,   # 60% der gewünschten Nodes
            "LOD256": 0.9,   # 90% der gewünschten Nodes
            "FINAL": 1.0     # 100% der gewünschten Nodes
        }

    def generate_plot_nodes(self, civ_map, plotnodes_count, settlements, progress_callback=None):
        """
        Funktionsweise: Generiert PlotNodes gleichmäßig verteilt außerhalb von Städten und Wilderness
        Aufgabe: Erstellt initiale PlotNode-Verteilung für Grundstücks-System mit LOD-Optimierung
        Parameter: civ_map, plotnodes_count, settlements, progress_callback - Civ-Map, Anzahl Nodes, Settlements und Progress
        Returns: List[PlotNode] - Generierte PlotNodes
        """
        if progress_callback:
            progress_callback("Plot Generation", 70, "Generating plot nodes...")

        height, width = civ_map.shape
        nodes = []
        
        # LOD-abhängige Node-Anzahl
        density_factor = self.lod_density_factors[self.lod_level]
        adjusted_count = int(plotnodes_count * density_factor)

        # Gültige Bereiche finden (nicht in Städten, nicht in Wilderness)
        valid_positions = []

        for y in range(height):
            for x in range(width):
                civ_value = civ_map[y, x]

                # Wilderness ausschließen (< 0.2)
                if civ_value < 0.2:
                    continue

                # Stadt-Bereiche ausschließen (= 1.0)
                if civ_value >= 1.0:
                    continue

                # Mindestabstand zu Settlements prüfen
                too_close = False
                for settlement in settlements:
                    distance = np.sqrt((x - settlement.x) ** 2 + (y - settlement.y) ** 2)
                    if distance < settlement.radius * 1.2:
                        too_close = True
                        break

                if not too_close:
                    valid_positions.append((x, y, civ_value))

        # PlotNodes gleichmäßig verteilen
        if len(valid_positions) < adjusted_count:
            adjusted_count = len(valid_positions)

        # Sampling für gleichmäßige Verteilung
        sampled_positions = self._sample_uniform_distribution(valid_positions, adjusted_count)

        for x, y, civ_value in sampled_positions:
            node = PlotNode(
                node_id=self.next_node_id,
                node_location=(float(x), float(y)),
                connector_id=[],
                connector_distance=[],
                connector_elevation=[],
                connector_movecost=[]
            )
            nodes.append(node)
            self.next_node_id += 1

        if progress_callback:
            progress_callback("Plot Generation", 75, f"Generated {len(nodes)} plot nodes")

        return nodes

    def create_delaunay_triangulation(self, nodes, heightmap, biome_map=None, progress_callback=None):
        """
        Funktionsweise: Erstellt Delaunay-Triangulation zwischen PlotNodes mit MoveCost-Berechnung
        Aufgabe: Verbindet PlotNodes über Delaunay-Dreiecke für Grundstücks-Bildung
        Parameter: nodes, heightmap, biome_map, progress_callback - PlotNode-Liste, Höhendaten, Biom-Map und Progress
        Returns: List[PlotNode] - Nodes mit aktualisierten Verbindungen
        """
        if progress_callback:
            progress_callback("Plot Generation", 80, "Creating Delaunay triangulation...")

        if len(nodes) < 3:
            return nodes

        # Koordinaten extrahieren
        points = np.array([node.node_location for node in nodes])

        try:
            # Delaunay-Triangulation
            tri = Delaunay(points)

            # Verbindungen aus Triangulation extrahieren
            for node in nodes:
                node.connector_id = []
                node.connector_distance = []
                node.connector_elevation = []
                node.connector_movecost = []

            for simplex in tri.simplices:
                # Jedes Dreieck verbindet 3 Nodes
                for i in range(3):
                    for j in range(i + 1, 3):
                        node_a_idx = simplex[i]
                        node_b_idx = simplex[j]

                        node_a = nodes[node_a_idx]
                        node_b = nodes[node_b_idx]

                        # Verbindung A->B
                        if node_b.node_id not in node_a.connector_id:
                            distance = self._calculate_distance(node_a.node_location, node_b.node_location)
                            elevation_diff = self._calculate_elevation_difference(
                                node_a.node_location, node_b.node_location, heightmap
                            )
                            movecost = self._calculate_biome_movecost(
                                node_a.node_location, node_b.node_location, biome_map, distance, elevation_diff
                            )

                            node_a.connector_id.append(node_b.node_id)
                            node_a.connector_distance.append(distance)
                            node_a.connector_elevation.append(elevation_diff)
                            node_a.connector_movecost.append(movecost)

                        # Verbindung B->A
                        if node_a.node_id not in node_b.connector_id:
                            distance = self._calculate_distance(node_b.node_location, node_a.node_location)
                            elevation_diff = self._calculate_elevation_difference(
                                node_b.node_location, node_a.node_location, heightmap
                            )
                            movecost = self._calculate_biome_movecost(
                                node_b.node_location, node_a.node_location, biome_map, distance, elevation_diff
                            )

                            node_b.connector_id.append(node_a.node_id)
                            node_b.connector_distance.append(distance)
                            node_b.connector_elevation.append(elevation_diff)
                            node_b.connector_movecost.append(movecost)

            return nodes

        except Exception as e:
            # Fallback: keine Verbindungen
            if progress_callback:
                progress_callback("Plot Generation", 80, f"Delaunay triangulation failed: {e}")
            return nodes

    def merge_to_plots(self, nodes, civ_map, progress_callback=None):
        """
        Funktionsweise: Fusioniert PlotNodes zu Plots basierend auf akkumuliertem Civ-Wert
        Aufgabe: Erstellt Grundstücke durch Node-Gruppierung nach Civ-Wert-Schwellwert
        Parameter: nodes, civ_map, progress_callback - PlotNode-Liste, Civ-Map und Progress
        Returns: List[Plot] - Generierte Plots
        """
        if progress_callback:
            progress_callback("Plot Generation", 85, "Merging nodes to plots...")

        plots = []
        used_nodes = set()

        for start_node in nodes:
            if start_node.node_id in used_nodes:
                continue

            # Neuen Plot starten
            plot_nodes = [start_node]
            used_nodes.add(start_node.node_id)
            accumulated_civ = self._get_node_civ_value(start_node, civ_map)

            # Nachbarn hinzufügen bis Schwellwert erreicht
            candidates = [start_node]

            while candidates and accumulated_civ < self.plotsize_threshold:
                current_node = candidates.pop(0)

                # Nachbarn des aktuellen Nodes prüfen
                for neighbor_id in current_node.connector_id:
                    if neighbor_id in used_nodes:
                        continue

                    neighbor_node = self._find_node_by_id(nodes, neighbor_id)
                    if neighbor_node is None:
                        continue

                    neighbor_civ = self._get_node_civ_value(neighbor_node, civ_map)

                    # Node zum Plot hinzufügen
                    plot_nodes.append(neighbor_node)
                    used_nodes.add(neighbor_id)
                    accumulated_civ += neighbor_civ
                    candidates.append(neighbor_node)

                    if accumulated_civ >= self.plotsize_threshold:
                        break

            # Plot erstellen
            if len(plot_nodes) > 0:
                plot = Plot(
                    plot_id=self.next_plot_id,
                    nodes=plot_nodes,
                    biome_amount={},  # Wird später gefüllt
                    resource_amount={},  # Wird später gefüllt
                    plot_area=self._calculate_plot_area(plot_nodes),
                    plot_distance=self._calculate_plot_distance(plot_nodes)
                )
                plots.append(plot)
                self.next_plot_id += 1

        if progress_callback:
            progress_callback("Plot Generation", 90, f"Created {len(plots)} plots from {len(nodes)} nodes")

        return plots

    def optimize_node_positions(self, nodes, iterations=5, progress_callback=None):
        """
        Funktionsweise: Optimiert PlotNode-Positionen durch Abstoßungslogik und Winkel-Glättung
        Aufgabe: Verbessert Node-Anordnung für natürlichere Grundstücks-Formen
        Parameter: nodes, iterations, progress_callback - PlotNode-Liste, Iterationen und Progress
        Returns: List[PlotNode] - Optimierte PlotNodes
        """
        if progress_callback:
            progress_callback("Plot Generation", 92, f"Optimizing node positions ({iterations} iterations)...")

        # LOD-abhängige Iterations-Anzahl
        if self.lod_level == "LOD64":
            iterations = max(2, iterations // 2)

        for iteration in range(iterations):
            # Abstoßungslogik zwischen zu nahen Nodes
            for i, node_a in enumerate(nodes):
                for j, node_b in enumerate(nodes[i + 1:], i + 1):
                    distance = self._calculate_distance(node_a.node_location, node_b.node_location)

                    if distance < 3.0:  # Zu nah
                        # Abstoßungsvektor berechnen
                        dx = node_b.node_location[0] - node_a.node_location[0]
                        dy = node_b.node_location[1] - node_a.node_location[1]

                        if distance > 0:
                            # Normalisieren und Abstoßung anwenden
                            dx /= distance
                            dy /= distance

                            repulsion_strength = (3.0 - distance) * 0.1

                            node_a.node_location = (
                                node_a.node_location[0] - dx * repulsion_strength,
                                node_a.node_location[1] - dy * repulsion_strength
                            )
                            node_b.node_location = (
                                node_b.node_location[0] + dx * repulsion_strength,
                                node_b.node_location[1] + dy * repulsion_strength
                            )

            # Progress-Update pro Iteration
            if progress_callback and iteration % max(1, iterations // 3) == 0:
                progress = 92 + iteration * 3 // iterations
                progress_callback("Plot Generation", progress, f"Optimization iteration {iteration + 1}/{iterations}")

        return nodes

    def _calculate_biome_movecost(self, pos1, pos2, biome_map, distance, elevation_diff):
        """
        Funktionsweise: Berechnet Bewegungskosten basierend auf Biom-Typen entlang des Pfades
        Aufgabe: Biom-abhängige MoveCost für PlotNode-Verbindungen
        Parameter: pos1, pos2, biome_map, distance, elevation_diff
        Returns: float - Biom-adjustierte Bewegungskosten
        """
        if biome_map is None:
            # Fallback: nur Distanz + Elevation
            return distance + abs(elevation_diff) * 0.5

        # Vereinfachte Biom-Kosten-Matrix
        biome_costs = {
            0: 1.0,   # ice_cap - schwer
            1: 0.8,   # tundra - moderat
            2: 0.6,   # taiga - leicht
            3: 0.4,   # grassland - sehr leicht
            4: 0.6,   # temperate_forest - moderat
            5: 0.5,   # mediterranean - leicht
            6: 1.2,   # desert - schwer
            7: 0.7,   # semi_arid - moderat
            8: 0.9,   # tropical_rainforest - schwer
            9: 0.7,   # tropical_seasonal - moderat
            10: 0.5,  # savanna - leicht
            11: 0.8,  # montane_forest - moderat
            12: 1.5,  # swamp - sehr schwer
            13: 0.6,  # coastal_dunes - moderat
            14: 1.1   # badlands - schwer
        }

        # Biom-Typ an Mittelpunkt der Verbindung
        mid_x = int((pos1[0] + pos2[0]) / 2)
        mid_y = int((pos1[1] + pos2[1]) / 2)

        if (0 <= mid_x < biome_map.shape[1] and 0 <= mid_y < biome_map.shape[0]):
            biome_type = biome_map[mid_y, mid_x]
            biome_cost_factor = biome_costs.get(biome_type, 1.0)
        else:
            biome_cost_factor = 1.0

        return distance * biome_cost_factor + abs(elevation_diff) * 0.5

    def _sample_uniform_distribution(self, positions, count):
        """
        Funktionsweise: Sampelt Positionen für gleichmäßige räumliche Verteilung
        Aufgabe: Verhindert Clustering von PlotNodes
        Parameter: positions, count - Verfügbare Positionen und gewünschte Anzahl
        Returns: List[Tuple] - Gleichmäßig verteilte Positionen
        """
        if len(positions) <= count:
            return positions

        # Einfaches Grid-basiertes Sampling
        positions.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x

        step = len(positions) // count
        sampled = []

        for i in range(0, len(positions), step):
            if len(sampled) < count:
                sampled.append(positions[i])

        return sampled

    def _calculate_distance(self, pos1, pos2):
        """
        Funktionsweise: Berechnet Euklidische Distanz zwischen zwei Positionen
        Aufgabe: Standard-Distanzberechnung für PlotNode-Verbindungen
        """
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _calculate_elevation_difference(self, pos1, pos2, heightmap):
        """
        Funktionsweise: Berechnet akkumulierten Höhenunterschied zwischen zwei Positionen
        Aufgabe: Höhendifferenz für MoveCost-Berechnung
        """
        x1, y1 = int(pos1[0]), int(pos1[1])
        x2, y2 = int(pos2[0]), int(pos2[1])

        height, width = heightmap.shape

        if (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height):
            return heightmap[y2, x2] - heightmap[y1, x1]

        return 0.0

    def _get_node_civ_value(self, node, civ_map):
        """
        Funktionsweise: Holt Civ-Wert für PlotNode-Position aus civ_map
        Aufgabe: Civ-Wert-Lookup für Plot-Größen-Berechnung
        """
        x, y = int(node.node_location[0]), int(node.node_location[1])
        height, width = civ_map.shape

        if 0 <= x < width and 0 <= y < height:
            return civ_map[y, x]

        return 0.0

    def _find_node_by_id(self, nodes, node_id):
        """
        Funktionsweise: Findet PlotNode anhand der ID
        Aufgabe: Node-Lookup für Nachbar-Suche
        """
        for node in nodes:
            if node.node_id == node_id:
                return node
        return None

    def _calculate_plot_area(self, plot_nodes):
        """
        Funktionsweise: Berechnet approximative Fläche eines Plots
        Aufgabe: Plot-Größen-Berechnung für Gameplay-Eigenschaften
        """
        if len(plot_nodes) < 3:
            return 1.0

        # Simplified: Anzahl Nodes als Flächen-Approximation
        return float(len(plot_nodes))

    def _calculate_plot_distance(self, plot_nodes):
        """
        Funktionsweise: Berechnet durchschnittliche Entfernung zwischen Plot-Nodes
        Aufgabe: Plot-Distanz-Metrik für Gameplay
        """
        if len(plot_nodes) < 2:
            return 0.0

        total_distance = 0.0
        connections = 0

        for node in plot_nodes:
            total_distance += sum(node.connector_distance)
            connections += len(node.connector_distance)

        if connections > 0:
            return total_distance / connections

        return 0.0


class SettlementGenerator(BaseGenerator):
    """
    Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung mit BaseGenerator-API und LOD-System
    Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map mit Progress-Updates
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Settlement-Generator mit BaseGenerator und Sub-Komponenten
        Aufgabe: Setup aller Settlement-Systeme und Rng-Seed
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Settlement-Platzierung
        """
        super().__init__(map_seed)
        random.seed(map_seed)
        np.random.seed(map_seed)

        self.next_location_id = 0

        # Standard-Parameter (werden durch _load_default_parameters überschrieben)
        self.settlements = 3
        self.landmarks = 3
        self.roadsites = 3
        self.plotnodes = 1000
        self.civ_influence_decay = 0.8
        self.terrain_factor_villages = 1.0
        self.road_slope_to_distance_ratio = 1.5
        self.landmark_wilderness = 0.3
        self.plotsize = 2.0

    def _load_default_parameters(self):
        """
        Funktionsweise: Lädt SETTLEMENT-Parameter aus value_default.py
        Aufgabe: Standard-Parameter für Settlement-Generierung
        Returns: dict - Alle Standard-Parameter für Settlement
        """
        from gui.config.value_default import SETTLEMENT

        return {
            'settlements': SETTLEMENT.SETTLEMENTS["default"],
            'landmarks': SETTLEMENT.LANDMARKS["default"],
            'roadsites': SETTLEMENT.ROADSITES["default"],
            'plotnodes': SETTLEMENT.PLOTNODES["default"],
            'civ_influence_decay': SETTLEMENT.CIV_INFLUENCE_DECAY["default"],
            'terrain_factor_villages': SETTLEMENT.TERRAIN_FACTOR_VILLAGES["default"],
            'road_slope_to_distance_ratio': SETTLEMENT.ROAD_SLOPE_TO_DISTANCE_RATIO["default"],
            'landmark_wilderness': SETTLEMENT.LANDMARK_WILDERNESS["default"],
            'plotsize': SETTLEMENT.PLOTSIZE["default"]
        }

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Holt benötigte Dependencies mit intelligenten Fallback-Werten
        Aufgabe: Dependency-Resolution für Settlement-Generierung mit optionalen Inputs
        Parameter: data_manager - DataManager-Instanz
        Returns: dict - Alle Input-Daten (required + optional mit Fallbacks)
        """
        if not data_manager:
            raise Exception("DataManager required for Settlement generation")

        dependencies = {}

        # REQUIRED Dependencies - müssen vorhanden sein
        required_deps = ['heightmap', 'slopemap', 'water_map']
        
        for dep_name in required_deps:
            data = None
            
            # Terrain-Daten
            if dep_name in ['heightmap', 'slopemap']:
                terrain_data = data_manager.get_terrain_data("complete")
                if terrain_data:
                    data = getattr(terrain_data, dep_name, None)
                if data is None:
                    data = data_manager.get_terrain_data(dep_name)
            
            # Water-Daten  
            elif dep_name == 'water_map':
                data = data_manager.get_water_data('water_map')
                if data is None:
                    # Fallback: versuche water_biomes_map
                    data = data_manager.get_water_data('water_biomes_map')

            if data is None:
                raise Exception(f"Required dependency '{dep_name}' not available in DataManager")

            dependencies[dep_name] = data

        # OPTIONAL Dependencies - erstelle Fallback-Werte wenn nicht vorhanden
        heightmap = dependencies['heightmap']

        # biome_map: Fallback basierend auf Höhe (für MoveCost-Berechnung)
        biome_map = data_manager.get_biome_data('biome_map')
        if biome_map is None:
            self.logger.warning("biome_map not available, creating height-based fallback")
            biome_map = self._create_fallback_biome_map(heightmap)
        dependencies['biome_map'] = biome_map

        self.logger.debug(f"Dependencies loaded - heightmap: {heightmap.shape}, water_map: {dependencies['water_map'].shape}")
        self.logger.debug(f"Optional deps - biome_map: {'fallback' if biome_map is dependencies.get('biome_map') else 'original'}")

        return dependencies

    def _create_fallback_biome_map(self, heightmap):
        """
        Funktionsweise: Erstellt intelligente Fallback biome_map basierend auf Höhenlage
        Aufgabe: Einfache Biom-Klassifikation wenn Biome-Generator noch nicht gelaufen ist
        Parameter: heightmap - Höhendaten
        Returns: numpy.ndarray - Fallback biome_map
        """
        height, width = heightmap.shape
        biome_map = np.zeros((height, width), dtype=np.uint8)

        # Normalisierte Höhe [0, 1]
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range > 0:
            norm_height = (heightmap - min_height) / height_range
        else:
            norm_height = np.zeros_like(heightmap)

        # Einfache höhenbasierte Biom-Zuordnung
        for y in range(height):
            for x in range(width):
                h = norm_height[y, x]

                if h < 0.2:
                    biome_map[y, x] = 3    # grassland (niedrig)
                elif h < 0.4:
                    biome_map[y, x] = 4    # temperate_forest (mittel-niedrig)
                elif h < 0.6:
                    biome_map[y, x] = 2    # taiga (mittel)
                elif h < 0.8:
                    biome_map[y, x] = 1    # tundra (mittel-hoch)
                else:
                    biome_map[y, x] = 0    # ice_cap (hoch)

        return biome_map

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Funktionsweise: Führt Settlement-Generierung mit Progress-Updates aus
        Aufgabe: Kernlogik der Settlement-Generierung mit allen 5 Hauptphasen
        Parameter: lod, dependencies, parameters
        Returns: SettlementData-Objekt mit allen Settlement-Outputs
        """
        heightmap = dependencies['heightmap']
        slopemap = dependencies['slopemap']
        water_map = dependencies['water_map']
        biome_map = dependencies['biome_map']

        # Parameter aktualisieren
        self.settlements = parameters['settlements']
        self.landmarks = parameters['landmarks']
        self.roadsites = parameters['roadsites']
        self.plotnodes = parameters['plotnodes']
        self.civ_influence_decay = parameters['civ_influence_decay']
        self.terrain_factor_villages = parameters['terrain_factor_villages']
        self.road_slope_to_distance_ratio = parameters['road_slope_to_distance_ratio']
        self.landmark_wilderness = parameters['landmark_wilderness']
        self.plotsize = parameters['plotsize']

        # LOD-Größe bestimmen
        target_size = self._get_lod_size(lod, heightmap.shape[0])

        # Alle Arrays auf Zielgröße interpolieren falls nötig
        if heightmap.shape[0] != target_size:
            heightmap = self._interpolate_array(heightmap, target_size)
            slopemap = self._interpolate_array(slopemap, target_size)
            water_map = self._interpolate_array(water_map, target_size)
            biome_map = self._interpolate_array(biome_map, target_size)

        # Phase 1: Settlement Generation (5% - 25%)
        self._update_progress("Settlement Placement", 5, "Analyzing terrain suitability...")
        settlement_list = self._generate_settlements(heightmap, slopemap, water_map, lod)

        # Phase 2: Road Network Creation (25% - 40%)
        self._update_progress("Road Building", 25, "Creating road networks between settlements...")
        roads = self._create_road_network(settlement_list, slopemap, lod)

        # Phase 3: Roadsite Placement (40% - 50%)
        self._update_progress("Roadsite Placement", 40, "Placing roadsites along roads...")
        roadsite_list = self._place_roadsites(roads, lod)

        # Phase 4: Civilization Mapping (50% - 70%)
        self._update_progress("Civilization Mapping", 50, "Creating civilization influence map...")
        civ_map = self._create_civilization_map(heightmap, slopemap, settlement_list, roads, roadsite_list)

        # Phase 5: Landmark & Plot Generation (70% - 95%)
        self._update_progress("Landmarks & Plots", 70, "Placing landmarks in wilderness areas...")
        landmark_list = self._place_landmarks(civ_map, heightmap, slopemap, lod)

        plot_nodes, plots = self._generate_plots(civ_map, settlement_list, heightmap, biome_map, lod)
        plot_map = self._create_plot_map(heightmap.shape, plots)

        # SettlementData-Objekt erstellen
        settlement_data = SettlementData()
        settlement_data.settlement_list = settlement_list
        settlement_data.landmark_list = landmark_list
        settlement_data.roadsite_list = roadsite_list
        settlement_data.plot_map = plot_map
        settlement_data.civ_map = civ_map
        settlement_data.plot_nodes = plot_nodes
        settlement_data.plots = plots
        settlement_data.roads = roads
        settlement_data.lod_level = lod
        settlement_data.actual_size = target_size
        settlement_data.parameters = parameters.copy()

        self.logger.debug(f"Settlement generation complete - LOD: {lod}, size: {target_size}")
        self.logger.debug(f"Generated: {len(settlement_list)} settlements, {len(roads)} roads, {len(plots)} plots")

        return settlement_data

    def _generate_settlements(self, heightmap, slopemap, water_map, lod):
        """
        Funktionsweise: Generiert Settlement-Positionen basierend auf Terrain-Suitability
        Aufgabe: Platziert Städte und Dörfer an optimal geeigneten Locations
        """
        # LOD-abhängige Settlement-Anzahl
        lod_factors = {"LOD64": 0.5, "LOD128": 0.8, "LOD256": 1.0, "FINAL": 1.0}
        adjusted_count = max(1, int(self.settlements * lod_factors.get(lod, 1.0)))

        analyzer = TerrainSuitabilityAnalyzer(self.terrain_factor_villages, lod)
        suitability_map = analyzer.create_combined_suitability(heightmap, slopemap, water_map, self._update_progress)

        settlements = []
        height, width = heightmap.shape

        # Mindestabstand zwischen Settlements berechnen
        map_diagonal = np.sqrt(height ** 2 + width ** 2)
        min_distance = max(10, map_diagonal / (adjusted_count + 1))

        attempts = 0
        max_attempts = adjusted_count * 20

        while len(settlements) < adjusted_count and attempts < max_attempts:
            attempts += 1

            # Position mit höchster Suitability finden
            best_positions = self._find_best_settlement_positions(suitability_map, settlements, min_distance)

            if not best_positions:
                break

            # Zufällige Position aus besten wählen
            x, y = random.choice(best_positions[:min(10, len(best_positions))])

            # Settlement erstellen
            settlement_size = random.uniform(0.5, 1.5)  # Variiert Stadtgröße
            radius = 3 + settlement_size * 2
            civ_influence = 0.8

            settlement = Location(
                location_id=self.next_location_id,
                x=float(x),
                y=float(y),
                location_type='settlement',
                radius=radius,
                civ_influence=civ_influence,
                properties={'size': settlement_size}
            )

            settlements.append(settlement)
            self.next_location_id += 1

            # Suitability um neue Settlement reduzieren
            self._reduce_suitability_around_point(suitability_map, x, y, min_distance)

            # Progress-Update
            if self._update_progress:
                progress = 5 + (len(settlements) * 15) // adjusted_count
                self._update_progress("Settlement Placement", progress, f"Placed {len(settlements)}/{adjusted_count} settlements")

        return settlements

    def _create_road_network(self, settlements, slopemap, lod):
        """
        Funktionsweise: Erstellt Straßennetzwerk zwischen Settlements mit LOD-optimiertem Pathfinding
        """
        if len(settlements) < 2:
            return []

        pathfinder = PathfindingSystem(self.road_slope_to_distance_ratio, lod)
        roads = []

        # Minimum Spanning Tree für Settlement-Verbindungen
        connected = [settlements[0]]
        unconnected = settlements[1:]

        road_count = 0
        total_roads = len(settlements) - 1

        while unconnected:
            best_connection = None
            best_distance = float('inf')

            for connected_settlement in connected:
                for unconnected_settlement in unconnected:
                    distance = np.sqrt(
                        (connected_settlement.x - unconnected_settlement.x) ** 2 +
                        (connected_settlement.y - unconnected_settlement.y) ** 2
                    )

                    if distance < best_distance:
                        best_distance = distance
                        best_connection = (connected_settlement, unconnected_settlement)

            if best_connection:
                start_settlement, end_settlement = best_connection

                # Pfad finden mit Progress-Callback
                path = pathfinder.find_least_resistance_path(
                    slopemap,
                    (start_settlement.x, start_settlement.y),
                    (end_settlement.x, end_settlement.y),
                    self._update_progress
                )

                # Spline-Glättung anwenden
                smoothed_path = pathfinder.apply_spline_smoothing(path, smoothing_factor=3, progress_callback=self._update_progress)
                roads.append(smoothed_path)

                connected.append(end_settlement)
                unconnected.remove(end_settlement)

                road_count += 1

                # Progress-Update für Straßen
                if self._update_progress:
                    progress = 25 + (road_count * 10) // total_roads
                    self._update_progress("Road Building", progress, f"Built {road_count}/{total_roads} roads")

        return roads

    def _place_roadsites(self, roads, lod):
        """
        Funktionsweise: Platziert Roadsites entlang von Straßen zwischen 30-70% der Weglänge
        """
        roadsites = []

        if not roads or self.roadsites == 0:
            return roadsites

        # LOD-abhängige Roadsite-Anzahl
        lod_factors = {"LOD64": 0.3, "LOD128": 0.6, "LOD256": 1.0, "FINAL": 1.0}
        adjusted_count = max(0, int(self.roadsites * lod_factors.get(lod, 1.0)))

        sites_per_road = max(1, adjusted_count // len(roads)) if adjusted_count > 0 else 0

        for road_idx, road in enumerate(roads):
            if len(road) < 3:
                continue

            road_length = len(road)

            for _ in range(sites_per_road):
                if len(roadsites) >= adjusted_count:
                    break

                # Position zwischen 30-70% der Weglänge
                position_ratio = random.uniform(0.3, 0.7)
                position_index = int(position_ratio * road_length)
                position_index = max(0, min(road_length - 1, position_index))

                x, y = road[position_index]

                roadsite_type = random.choice([
                    'tavern', 'trading_post', 'shrine', 'toll_house',
                    'gallows', 'market', 'industry'
                ])

                roadsite = Location(
                    location_id=self.next_location_id,
                    x=float(x),
                    y=float(y),
                    location_type='roadsite',
                    radius=1.5,
                    civ_influence=0.4,
                    properties={'roadsite_type': roadsite_type}
                )

                roadsites.append(roadsite)
                self.next_location_id += 1

        if self._update_progress:
            self._update_progress("Roadsite Placement", 45, f"Placed {len(roadsites)} roadsites")

        return roadsites

    def _create_civilization_map(self, heightmap, slopemap, settlements, roads, roadsites):
        """
        Funktionsweise: Erstellt finale civ_map durch Kombination aller Zivilisations-Einflüsse
        """
        height, width = heightmap.shape
        civ_map = np.zeros((height, width), dtype=np.float32)

        influence_mapper = CivilizationInfluenceMapper(self.civ_influence_decay)

        # Settlement-Einfluss anwenden
        civ_map = influence_mapper.apply_settlement_influence(civ_map, settlements, slopemap, self._update_progress)

        # Road-Einfluss anwenden
        civ_map = influence_mapper.calculate_road_influence(civ_map, roads, slopemap, self._update_progress)

        # Roadsite-Einfluss anwenden
        for roadsite in roadsites:
            civ_map = influence_mapper.apply_decay_kernel(
                civ_map, roadsite, roadsite.civ_influence, roadsite.radius, slopemap
            )

        # Wilderness definieren (< 0.2 wird auf 0.0 gesetzt)
        wilderness_mask = civ_map < 0.2
        civ_map[wilderness_mask] = 0.0

        return civ_map

    def _place_landmarks(self, civ_map, heightmap, slopemap, lod):
        """
        Funktionsweise: Platziert Landmarks in Wilderness-Bereichen mit niedrigem Civ-Wert
        """
        landmarks = []

        # LOD-abhängige Landmark-Anzahl
        lod_factors = {"LOD64": 0.5, "LOD128": 0.8, "LOD256": 1.0, "FINAL": 1.0}
        adjusted_count = max(0, int(self.landmarks * lod_factors.get(lod, 1.0)))

        if adjusted_count == 0:
            return landmarks

        height, width = civ_map.shape
        valid_positions = []

        for y in range(height):
            for x in range(width):
                civ_value = civ_map[y, x]

                # Wilderness-Schwellwert prüfen
                if civ_value >= self.landmark_wilderness:
                    continue

                # Niedrige Höhen und Slopes bevorzugen
                elevation_ok = self._check_elevation_suitability(heightmap, x, y)
                slope_ok = self._check_slope_suitability(slopemap, x, y)

                if elevation_ok and slope_ok:
                    valid_positions.append((x, y))

        # Landmarks gleichmäßig verteilen
        if len(valid_positions) < adjusted_count:
            adjusted_count = len(valid_positions)

        sampled_positions = self._sample_landmark_positions(valid_positions, adjusted_count)

        for x, y in sampled_positions:
            landmark_type = random.choice(['castle', 'monastery', 'mystic_site', 'ruins'])

            landmark = Location(
                location_id=self.next_location_id,
                x=float(x),
                y=float(y),
                location_type='landmark',
                radius=2.0,
                civ_influence=0.4,
                properties={'landmark_type': landmark_type}
            )

            landmarks.append(landmark)
            self.next_location_id += 1

        if self._update_progress:
            self._update_progress("Landmarks & Plots", 72, f"Placed {len(landmarks)} landmarks")

        return landmarks

    def _generate_plots(self, civ_map, settlements, heightmap, biome_map, lod):
        """
        Funktionsweise: Generiert Grundstücks-System mit PlotNodes und Delaunay-Triangulation
        """
        plot_system = PlotNodeSystem(self.plotsize, lod)

        # PlotNodes generieren
        nodes = plot_system.generate_plot_nodes(civ_map, self.plotnodes, settlements, self._update_progress)

        # Delaunay-Triangulation mit BiomeMap-Integration
        nodes = plot_system.create_delaunay_triangulation(nodes, heightmap, biome_map, self._update_progress)

        # Node-Positionen optimieren
        nodes = plot_system.optimize_node_positions(nodes, iterations=5, progress_callback=self._update_progress)

        # Plots aus Nodes erstellen
        plots = plot_system.merge_to_plots(nodes, civ_map, self._update_progress)

        return nodes, plots

    def _save_to_data_manager(self, data_manager, result, parameters):
        """
        Funktionsweise: Speichert Settlement-Ergebnisse im DataManager
        Aufgabe: Automatische Speicherung aller Settlement-Outputs mit Parameter-Tracking
        Parameter: data_manager, result (SettlementData), parameters
        """
        if isinstance(result, SettlementData):
            # SettlementData-Objekt in einzelne Arrays/Listen aufteilen für DataManager
            data_manager.set_settlement_data("settlement_list", result.settlement_list, parameters)
            data_manager.set_settlement_data("landmark_list", result.landmark_list, parameters)
            data_manager.set_settlement_data("roadsite_list", result.roadsite_list, parameters)
            data_manager.set_settlement_data("plot_map", result.plot_map, parameters)
            data_manager.set_settlement_data("civ_map", result.civ_map, parameters)

            # Zusätzliche Daten für erweiterte Funktionalität
            data_manager.set_settlement_data("plot_nodes", result.plot_nodes, parameters)
            data_manager.set_settlement_data("plots", result.plots, parameters)
            data_manager.set_settlement_data("roads", result.roads, parameters)

            self.logger.debug(f"SettlementData object saved to DataManager - {len(result.settlement_list)} settlements, {len(result.plots)} plots")
        else:
            # Fallback für Legacy-Format (Tuple)
            if hasattr(result, '__len__') and len(result) >= 5:
                settlement_list, landmark_list, roadsite_list, plot_map, civ_map = result[:5]
                data_manager.set_settlement_data("settlement_list", settlement_list, parameters)
                data_manager.set_settlement_data("landmark_list", landmark_list, parameters)
                data_manager.set_settlement_data("roadsite_list", roadsite_list, parameters)
                data_manager.set_settlement_data("plot_map", plot_map, parameters)
                data_manager.set_settlement_data("civ_map", civ_map, parameters)
                self.logger.debug("Legacy settlement data saved to DataManager")

    def _get_lod_size(self, lod, original_size):
        """
        Funktionsweise: Bestimmt Zielgröße basierend auf LOD-Level
        Aufgabe: LOD-System für Settlement mit gleicher Logik wie andere Generatoren
        """
        lod_sizes = {"LOD64": 64, "LOD128": 128, "LOD256": 256, "LOD512": 512, "LOD1024": 1024}

        if lod == "FINAL":
            return original_size
        else:
            return lod_sizes.get(lod, 64)

    def _interpolate_array(self, array, target_size):
        """
        Funktionsweise: Interpoliert 2D-Array auf neue Größe mittels bilinearer Interpolation
        Aufgabe: LOD-Upscaling für alle Input-Arrays
        """
        if len(array.shape) == 2:
            # 2D Array (heightmap, water_map, biome_map)
            return self._interpolate_2d(array, target_size)
        elif len(array.shape) == 3 and array.shape[2] == 2:
            # 3D Array mit 2 Kanälen (slopemap)
            result = np.zeros((target_size, target_size, 2), dtype=array.dtype)
            result[:, :, 0] = self._interpolate_2d(array[:, :, 0], target_size)
            result[:, :, 1] = self._interpolate_2d(array[:, :, 1], target_size)
            return result
        else:
            raise ValueError(f"Unsupported array shape for interpolation: {array.shape}")

    def _interpolate_2d(self, array, target_size):
        """
        Funktionsweise: Bilineare Interpolation für 2D-Arrays
        Aufgabe: Smooth Upscaling ohne Artefakte
        """
        old_size = array.shape[0]
        if old_size == target_size:
            return array.copy()

        scale_factor = (old_size - 1) / (target_size - 1)
        interpolated = np.zeros((target_size, target_size), dtype=array.dtype)

        for new_y in range(target_size):
            for new_x in range(target_size):
                old_x = new_x * scale_factor
                old_y = new_y * scale_factor

                x0, y0 = int(old_x), int(old_y)
                x1, y1 = min(x0 + 1, old_size - 1), min(y0 + 1, old_size - 1)

                fx, fy = old_x - x0, old_y - y0

                # Bilineare Interpolation
                h00, h10 = array[y0, x0], array[y0, x1]
                h01, h11 = array[y1, x0], array[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                interpolated[new_y, new_x] = h0 * (1 - fy) + h1 * fy

        return interpolated

    # Hilfsmethoden für Settlement-Platzierung
    def _find_best_settlement_positions(self, suitability_map, existing_settlements, min_distance):
        """
        Funktionsweise: Findet beste verfügbare Positionen für Settlement-Platzierung
        """
        height, width = suitability_map.shape
        candidates = []

        threshold = np.percentile(suitability_map, 75)  # Top 25%

        for y in range(height):
            for x in range(width):
                if suitability_map[y, x] < threshold:
                    continue

                # Mindestabstand zu existierenden Settlements prüfen
                too_close = False
                for settlement in existing_settlements:
                    distance = np.sqrt((x - settlement.x) ** 2 + (y - settlement.y) ** 2)
                    if distance < min_distance:
                        too_close = True
                        break

                if not too_close:
                    candidates.append((x, y, suitability_map[y, x]))

        candidates.sort(key=lambda c: c[2], reverse=True)
        return [(x, y) for x, y, _ in candidates]

    def _reduce_suitability_around_point(self, suitability_map, center_x, center_y, radius):
        """
        Funktionsweise: Reduziert Suitability um gegebenen Punkt für Mindestabstände
        """
        height, width = suitability_map.shape

        for y in range(max(0, int(center_y - radius)), min(height, int(center_y + radius + 1))):
            for x in range(max(0, int(center_x - radius)), min(width, int(center_x + radius + 1))):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                if distance <= radius:
                    reduction_factor = 1.0 - (distance / radius) * 0.8
                    suitability_map[y, x] *= reduction_factor

    def _check_elevation_suitability(self, heightmap, x, y):
        """
        Funktionsweise: Prüft ob Elevation für Landmark geeignet ist
        """
        height, width = heightmap.shape

        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            return True

        norm_height = (heightmap[y, x] - min_height) / height_range
        return norm_height < 0.7  # Landmarks nur in unteren 70% der Höhen

    def _check_slope_suitability(self, slopemap, x, y):
        """
        Funktionsweise: Prüft ob Slope für Landmark geeignet ist
        """
        height, width = slopemap.shape[:2]

        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        dz_dx = slopemap[y, x, 0]
        dz_dy = slopemap[y, x, 1]
        slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        return slope_magnitude < 0.5  # Landmarks nur bei moderaten Slopes

    def _sample_landmark_positions(self, positions, count):
        """
        Funktionsweise: Sampelt Landmark-Positionen für gleichmäßige Verteilung
        """
        if len(positions) <= count:
            return positions

        step = len(positions) // count
        sampled = []

        for i in range(0, len(positions), step):
            if len(sampled) < count:
                sampled.append(positions[i])

        return sampled

    def _create_plot_map(self, map_shape, plots):
        """
        Funktionsweise: Erstellt Plot-Map mit Plot-IDs für jede Map-Position
        """
        height, width = map_shape
        plot_map = np.zeros((height, width), dtype=np.int32)

        for plot in plots:
            for node in plot.nodes:
                x, y = int(node.node_location[0]), int(node.node_location[1])

                if 0 <= x < width and 0 <= y < height:
                    plot_map[y, x] = plot.plot_id

        return plot_map

    # ===== LEGACY-KOMPATIBILITÄT =====
    # Alle alten Methoden bleiben für Rückwärts-Kompatibilität erhalten

    def generate_settlements(self, heightmap, slopemap, water_map, settlements_count, terrain_factor_villages):
        """
        Funktionsweise: Legacy-Methode für direkte Settlement-Generierung (KOMPATIBILITÄT)
        """
        dependencies = {
            'heightmap': heightmap,
            'slopemap': slopemap,
            'water_map': water_map,
            'biome_map': self._create_fallback_biome_map(heightmap)
        }
        parameters = {
            'settlements': settlements_count,
            'landmarks': self.landmarks,
            'roadsites': self.roadsites,
            'plotnodes': self.plotnodes,
            'civ_influence_decay': self.civ_influence_decay,
            'terrain_factor_villages': terrain_factor_villages,
            'road_slope_to_distance_ratio': self.road_slope_to_distance_ratio,
            'landmark_wilderness': self.landmark_wilderness,
            'plotsize': self.plotsize
        }

        settlement_data = self._execute_generation("LOD64", dependencies, parameters)
        return settlement_data.settlement_list

    def create_road_network(self, settlements, slopemap, road_slope_to_distance_ratio):
        """
        Funktionsweise: Legacy-Methode für Road-Network-Erstellung
        """
        pathfinder = PathfindingSystem(road_slope_to_distance_ratio, "LOD64")
        return self._create_road_network(settlements, slopemap, "LOD64")

    def place_landmarks(self, civ_map, landmarks_count, landmark_wilderness, heightmap, slopemap):
        """
        Funktionsweise: Legacy-Methode für Landmark-Platzierung
        """
        self.landmarks = landmarks_count
        self.landmark_wilderness = landmark_wilderness
        return self._place_landmarks(civ_map, heightmap, slopemap, "LOD64")

    def place_roadsites(self, roads, roadsites_count):
        """
        Funktionsweise: Legacy-Methode für Roadsite-Platzierung
        """
        self.roadsites = roadsites_count
        return self._place_roadsites(roads, "LOD64")

    def generate_plots(self, civ_map, plotnodes_count, plotsize, settlements, heightmap):
        """
        Funktionsweise: Legacy-Methode für Plot-Generierung
        """
        self.plotnodes = plotnodes_count
        self.plotsize = plotsize
        biome_map = self._create_fallback_biome_map(heightmap)
        return self._generate_plots(civ_map, settlements, heightmap, biome_map, "LOD64")

    def create_civilization_map(self, heightmap, slopemap, settlements, roads, landmarks, roadsites, civ_influence_decay):
        """
        Funktionsweise: Legacy-Methode für Civilization-Map-Erstellung
        """
        self.civ_influence_decay = civ_influence_decay
        return self._create_civilization_map(heightmap, slopemap, settlements, roads, roadsites)

    def generate_complete_settlements(self, heightmap, slopemap, water_map, map_seed, settlements,
                                      landmarks, roadsites, plotnodes, civ_influence_decay, terrain_factor_villages,
                                      road_slope_to_distance_ratio, landmark_wilderness, plotsize):
        """
        Funktionsweise: Legacy-Methode für komplette Settlement-Generierung (KOMPATIBILITÄT)
        """
        # Konvertiert alte API zur neuen API
        dependencies = {
            'heightmap': heightmap,
            'slopemap': slopemap,
            'water_map': water_map,
            'biome_map': self._create_fallback_biome_map(heightmap)
        }
        parameters = {
            'settlements': settlements,
            'landmarks': landmarks,
            'roadsites': roadsites,
            'plotnodes': plotnodes,
            'civ_influence_decay': civ_influence_decay,
            'terrain_factor_villages': terrain_factor_villages,
            'road_slope_to_distance_ratio': road_slope_to_distance_ratio,
            'landmark_wilderness': landmark_wilderness,
            'plotsize': plotsize
        }

        # Seed aktualisieren falls nötig
        if map_seed != self.map_seed:
            self.update_seed(map_seed)

        settlement_data = self._execute_generation("LOD256", dependencies, parameters)  # Höheres LOD für Legacy

        # Legacy-Format zurückgeben (Tuple)
        return (settlement_data.settlement_list, settlement_data.landmark_list, settlement_data.roadsite_list,
                settlement_data.plot_map, settlement_data.civ_map)

    def get_settlement_statistics(self, settlement_data):
        """
        Funktionsweise: Legacy-Methode für Settlement-Statistiken
        Aufgabe: Analyse-Funktionen für Settlement-System-Debugging
        """
        if isinstance(settlement_data, SettlementData):
            settlement_list = settlement_data.settlement_list
            landmark_list = settlement_data.landmark_list
            roadsite_list = settlement_data.roadsite_list
            plot_map = settlement_data.plot_map
            civ_map = settlement_data.civ_map
            plots = settlement_data.plots
            roads = settlement_data.roads
        else:
            # Legacy Tuple-Format
            settlement_list, landmark_list, roadsite_list, plot_map, civ_map = settlement_data[:5]
            plots = []
            roads = []

        # Settlement-Statistiken
        settlement_types = {}
        for settlement in settlement_list:
            settlement_types[settlement.location_type] = settlement_types.get(settlement.location_type, 0) + 1

        # Landmark-Statistiken
        landmark_types = {}
        for landmark in landmark_list:
            landmark_type = landmark.properties.get('landmark_type', 'unknown') if landmark.properties else 'unknown'
            landmark_types[landmark_type] = landmark_types.get(landmark_type, 0) + 1

        # Roadsite-Statistiken
        roadsite_types = {}
        for roadsite in roadsite_list:
            roadsite_type = roadsite.properties.get('roadsite_type', 'unknown') if roadsite.properties else 'unknown'
            roadsite_types[roadsite_type] = roadsite_types.get(roadsite_type, 0) + 1

        # Civ-Map-Statistiken
        civ_stats = {
            'min': float(np.min(civ_map)),
            'max': float(np.max(civ_map)),
            'mean': float(np.mean(civ_map)),
            'std': float(np.std(civ_map)),
            'wilderness_pixels': int(np.sum(civ_map < 0.2)),
            'civilized_pixels': int(np.sum(civ_map >= 1.0))
        }

        # Plot-Statistiken
        plot_stats = {
            'total_plots': len(plots),
            'total_nodes': sum(len(plot.nodes) for plot in plots) if plots else 0,
            'avg_plot_size': np.mean([plot.plot_area for plot in plots]) if plots else 0.0,
            'unique_plot_ids': len(np.unique(plot_map[plot_map > 0])) if plot_map is not None else 0
        }

        # Road-Statistiken
        road_stats = {
            'total_roads': len(roads),
            'total_road_length': sum(len(road) for road in roads),
            'avg_road_length': np.mean([len(road) for road in roads]) if roads else 0.0
        }

        stats = {
            'settlements': {
                'total': len(settlement_list),
                'types': settlement_types,
                'avg_radius': np.mean([s.radius for s in settlement_list]) if settlement_list else 0.0,
                'avg_influence': np.mean([s.civ_influence for s in settlement_list]) if settlement_list else 0.0
            },
            'landmarks': {
                'total': len(landmark_list),
                'types': landmark_types
            },
            'roadsites': {
                'total': len(roadsite_list),
                'types': roadsite_types
            },
            'civilization_map': civ_stats,
            'plots': plot_stats,
            'roads': road_stats,
            'map_coverage': {
                'wilderness_percentage': (civ_stats['wilderness_pixels'] / np.prod(civ_map.shape)) * 100,
                'civilized_percentage': (civ_stats['civilized_pixels'] / np.prod(civ_map.shape)) * 100
            }
        }

        return stats

    def update_seed(self, new_seed):
        """
        Funktionsweise: Aktualisiert Seed für alle Settlement-Komponenten
        Aufgabe: Seed-Update mit Re-Initialisierung der Random-Generatoren
        Parameter: new_seed (int) - Neuer Seed
        """
        if new_seed != self.map_seed:
            super().update_seed(new_seed)
            # Random-Generatoren mit neuem Seed re-initialisieren
            random.seed(new_seed)
            np.random.seed(new_seed)
            # Location-ID-Counter zurücksetzen für reproduzierbare IDs
            self.next_location_id = 0

    def get_settlement_info(self):
        """
        Funktionsweise: Gibt Informationen über den Settlement-Generator zurück
        Aufgabe: Debugging und Monitoring-Support für Settlement-System
        Returns: dict - Settlement-Generator-Metadaten
        """
        base_info = super().get_generator_info()

        settlement_info = {
            **base_info,
            'settlement_config': {
                'settlements': self.settlements,
                'landmarks': self.landmarks,
                'roadsites': self.roadsites,
                'plotnodes': self.plotnodes
            },
            'generation_parameters': {
                'civ_influence_decay': self.civ_influence_decay,
                'terrain_factor_villages': self.terrain_factor_villages,
                'road_slope_to_distance_ratio': self.road_slope_to_distance_ratio,
                'landmark_wilderness': self.landmark_wilderness,
                'plotsize': self.plotsize
            },
            'next_location_id': self.next_location_id
        }

        return settlement_info

    def validate_settlement_parameters(self, parameters):
        """
        Funktionsweise: Validiert Settlement-Parameter für sinnvolle Werte
        Aufgabe: Parameter-Validation vor Generierung
        Parameter: parameters (dict) - Zu validierende Parameter
        Returns: tuple (is_valid: bool, warnings: list, errors: list)
        """
        warnings = []
        errors = []

        # Settlement-Anzahl-Validation
        if parameters.get('settlements', 3) <= 0:
            errors.append("Settlement count must be positive")
        elif parameters.get('settlements', 3) > 10:
            warnings.append("High settlement count may cause performance issues")

        # PlotNode-Validation
        if parameters.get('plotnodes', 1000) > 5000:
            warnings.append("High plotnode count may cause memory issues")

        # Wilderness-Threshold-Validation
        wilderness = parameters.get('landmark_wilderness', 0.3)
        if wilderness < 0.1 or wilderness > 0.8:
            warnings.append("Landmark wilderness threshold outside recommended range (0.1-0.8)")

        # Road-Slope-Ratio-Validation
        road_ratio = parameters.get('road_slope_to_distance_ratio', 1.5)
        if road_ratio < 0.1 or road_ratio > 5.0:
            warnings.append("Road slope to distance ratio outside practical range")

        # Plotsize-Validation
        plotsize = parameters.get('plotsize', 2.0)
        if plotsize < 0.5 or plotsize > 10.0:
            warnings.append("Plot size outside recommended range (0.5-10.0)")

        return len(errors) == 0, warnings, errors

# =============================================================================
# Utility Functions für Settlement-System
# =============================================================================

def create_settlement_summary(settlement_data):
    """
    Funktionsweise: Erstellt Zusammenfassung der Settlement-Generierung
    Parameter: settlement_data (SettlementData oder Tuple)
    Returns: dict - Übersichtliche Zusammenfassung aller Settlement-Aspekte
    """
    if isinstance(settlement_data, SettlementData):
        return {
            'lod_level': settlement_data.lod_level,
            'map_size': settlement_data.actual_size,
            'settlements': len(settlement_data.settlement_list),
            'landmarks': len(settlement_data.landmark_list),
            'roadsites': len(settlement_data.roadsite_list),
            'roads': len(settlement_data.roads),
            'plots': len(settlement_data.plots),
            'plot_nodes': len(settlement_data.plot_nodes),
            'parameters_used': settlement_data.parameters
        }
    else:
        # Legacy Tuple-Format
        settlement_list, landmark_list, roadsite_list, plot_map, civ_map = settlement_data[:5]
        return {
            'lod_level': 'unknown',
            'map_size': civ_map.shape[0] if civ_map is not None else 0,
            'settlements': len(settlement_list),
            'landmarks': len(landmark_list),
            'roadsites': len(roadsite_list),
            'roads': 0,  # Nicht verfügbar in Legacy-Format
            'plots': len(np.unique(plot_map[plot_map > 0])) if plot_map is not None else 0,
            'plot_nodes': 0,  # Nicht verfügbar in Legacy-Format
            'parameters_used': {}
        }

def export_settlement_data(settlement_data, format_type='dict'):
    """
    Funktionsweise: Exportiert Settlement-Daten in verschiedene Formate
    Parameter: settlement_data, format_type ('dict', 'json', 'summary')
    Returns: Exportierte Daten im gewünschten Format
    """
    if not isinstance(settlement_data, SettlementData):
        raise ValueError("export_settlement_data requires SettlementData object")

    if format_type == 'dict':
        return {
            'settlements': [
                {
                    'id': s.location_id,
                    'x': s.x,
                    'y': s.y,
                    'type': s.location_type,
                    'radius': s.radius,
                    'influence': s.civ_influence,
                    'properties': s.properties
                } for s in settlement_data.settlement_list
            ],
            'landmarks': [
                {
                    'id': l.location_id,
                    'x': l.x,
                    'y': l.y,
                    'type': l.location_type,
                    'landmark_type': l.properties.get('landmark_type', 'unknown') if l.properties else 'unknown',
                    'influence': l.civ_influence
                } for l in settlement_data.landmark_list
            ],
            'roadsites': [
                {
                    'id': r.location_id,
                    'x': r.x,
                    'y': r.y,
                    'type': r.location_type,
                    'roadsite_type': r.properties.get('roadsite_type', 'unknown') if r.properties else 'unknown',
                    'influence': r.civ_influence
                } for r in settlement_data.roadsite_list
            ],
            'plots': [
                {
                    'id': p.plot_id,
                    'area': p.plot_area,
                    'nodes': len(p.nodes),
                    'distance': p.plot_distance
                } for p in settlement_data.plots
            ],
            'metadata': {
                'lod_level': settlement_data.lod_level,
                'map_size': settlement_data.actual_size,
                'generation_parameters': settlement_data.parameters
            }
        }
    elif format_type == 'summary':
        return create_settlement_summary(settlement_data)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")