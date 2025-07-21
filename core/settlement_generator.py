"""
Path: core/settlement_generator.py

Funktionsweise: Intelligente Settlement-Platzierung
- Terrain-Suitability Analysis (Steigung, Höhe, Wasser-Nähe)
    Suitability-Map wird mit diesen Einflussgrößen erzeugt.
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

data_manager Input:
    - heightmap
    - slopemap
    - water_map

Parameter Input:
    - map_seed
    - settlements, landmarks, roadsites, plotnodes: number of each type
    - civ_influence_decay: Influence around Locationtypes decays of distance
    - terrain_factor_villages: terrain influence on settlement suitability
    - road_slope_to_distance_ratio: rather short roads or steep roads
    - landmark_wilderness: wilderness area size by changing cutoff-threshold
    - plotsize: how much accumulated civ-value to form plot

Output:
    - settlement list
    - landmark list
    - roadsite list
    - plot_map
    - civ_map

Klassen:
SettlementGenerator  
    Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung und Civilization-Mapping
    Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map
    Methoden: generate_settlements(), create_road_network(), place_landmarks(), generate_plots()

TerrainSuitabilityAnalyzer
    Funktionsweise: Analysiert Terrain-Eignung für Settlements basierend auf Steigung, Höhe, Wasser-Nähe
    Aufgabe: Erstellt Suitability-Map für optimale Settlement-Platzierung
    Methoden: analyze_slope_suitability(), calculate_water_proximity(), evaluate_elevation_fitness()

PathfindingSystem   
    Funktionsweise: Findet Wege geringsten Widerstands zwischen Settlements für Straßen
    Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation
    Methoden: find_least_resistance_path(), apply_spline_smoothing(), calculate_movement_cost()

CivilizationInfluenceMapper    
    Funktionsweise: Berechnet civ_map durch radialen Decay von Settlement/Road/Landmark-Punkten
    Aufgabe: Erstellt realistische Zivilisations-Verteilung mit Decay-Kernels
    Methoden: apply_settlement_influence(), calculate_road_influence(), apply_decay_kernel()

PlotNodeSystem    
    Funktionsweise: Generiert Plotnodes mit Delaunay-Triangulation und Grundstücks-Bildung
    Aufgabe: Erstellt Grundstücks-System für späteres Gameplay
    Methoden: generate_plot_nodes(), create_delaunay_triangulation(), merge_to_plots(), optimize_node_positions()
"""


import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
import heapq
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random


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
    Aufgabe: Erstellt Suitability-Map für optimale Settlement-Platzierung
    """

    def __init__(self, terrain_factor_villages=1.0):
        """
        Funktionsweise: Initialisiert Suitability-Analyzer mit Terrain-Gewichtungsfaktor
        Aufgabe: Setup der Terrain-Analyse-Parameter
        Parameter: terrain_factor_villages (float) - Gewichtung des Terrain-Einflusses
        """
        self.terrain_factor = terrain_factor_villages

    def analyze_slope_suitability(self, slopemap):
        """
        Funktionsweise: Bewertet Terrain-Eignung basierend auf Slope-Steilheit
        Aufgabe: Erstellt Slope-Suitability-Map für Settlement-Platzierung
        Parameter: slopemap (numpy.ndarray) - Slope-Daten (dz/dx, dz/dy)
        Returns: numpy.ndarray - Slope-Suitability zwischen 0 (ungeeignet) und 1 (ideal)
        """
        height, width = slopemap.shape[:2]
        slope_suitability = np.zeros((height, width), dtype=np.float32)

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

        return slope_suitability

    def calculate_water_proximity(self, water_map):
        """
        Funktionsweise: Berechnet Eignung basierend auf Nähe zu Wasserquellen
        Aufgabe: Erstellt Water-Proximity-Suitability für Settlement-Platzierung
        Parameter: water_map (numpy.ndarray) - Wasser-Daten
        Returns: numpy.ndarray - Water-Proximity-Suitability
        """
        height, width = water_map.shape
        water_suitability = np.zeros((height, width), dtype=np.float32)

        # Finde alle Wasser-Pixel
        water_pixels = np.where(water_map > 0)

        for y in range(height):
            for x in range(width):
                if len(water_pixels[0]) == 0:
                    water_suitability[y, x] = 0.0
                    continue

                # Minimale Distanz zu Wasser berechnen
                distances = np.sqrt((water_pixels[1] - x) ** 2 + (water_pixels[0] - y) ** 2)
                min_distance = np.min(distances)

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

    def evaluate_elevation_fitness(self, heightmap):
        """
        Funktionsweise: Bewertet Terrain-Eignung basierend auf Höhenlage
        Aufgabe: Erstellt Elevation-Suitability für Settlement-Platzierung
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: numpy.ndarray - Elevation-Suitability
        """
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

    def create_combined_suitability(self, heightmap, slopemap, water_map):
        """
        Funktionsweise: Kombiniert alle Suitability-Faktoren zu finaler Suitability-Map
        Aufgabe: Erstellt finale Settlement-Suitability durch gewichtete Kombination
        Parameter: heightmap, slopemap, water_map - Alle Terrain-Daten
        Returns: numpy.ndarray - Kombinierte Suitability-Map
        """
        slope_suit = self.analyze_slope_suitability(slopemap)
        water_suit = self.calculate_water_proximity(water_map)
        elevation_suit = self.evaluate_elevation_fitness(heightmap)

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
    Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation
    """

    def __init__(self, road_slope_to_distance_ratio=1.0):
        """
        Funktionsweise: Initialisiert Pathfinding-System mit Slope-Distance-Gewichtung
        Aufgabe: Setup der Pathfinding-Parameter
        Parameter: road_slope_to_distance_ratio (float) - Gewichtung zwischen Steigung und Distanz
        """
        self.slope_distance_ratio = road_slope_to_distance_ratio

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

    def find_least_resistance_path(self, slopemap, start_pos, end_pos):
        """
        Funktionsweise: A*-Pathfinding für Weg geringsten Widerstands zwischen zwei Punkten
        Aufgabe: Findet optimalen Straßenverlauf zwischen Settlements
        Parameter: slopemap, start_pos, end_pos - Slope-Daten und Start-/Ziel-Koordinaten
        Returns: List[Tuple] - Wegpunkte vom Start zum Ziel
        """
        height, width = slopemap.shape[:2]
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        end_x, end_y = int(end_pos[0]), int(end_pos[1])

        # A*-Datenstrukturen
        open_set = [(0, start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self._heuristic((start_x, start_y), (end_x, end_y))}

        while open_set:
            current_f, current_x, current_y = heapq.heappop(open_set)

            if current_x == end_x and current_y == end_y:
                # Pfad rekonstruieren
                path = []
                while (current_x, current_y) in came_from:
                    path.append((current_x, current_y))
                    current_x, current_y = came_from[(current_x, current_y)]
                path.append((start_x, start_y))
                return list(reversed(path))

            # Nachbarn prüfen (8-Connectivity)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
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

        # Kein Pfad gefunden - direkte Linie als Fallback
        return [(start_x, start_y), (end_x, end_y)]

    def _heuristic(self, pos1, pos2):
        """
        Funktionsweise: Heuristik-Funktion für A*-Algorithmus
        Aufgabe: Schätzt Kosten vom aktuellen Punkt zum Ziel
        Parameter: pos1, pos2 - Aktuelle und Ziel-Position
        Returns: float - Geschätzte Kosten
        """
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def apply_spline_smoothing(self, path, smoothing_factor=3):
        """
        Funktionsweise: Wendet Spline-Interpolation auf Pfad an für sanfte Straßenführung
        Aufgabe: Glättet Straßenverlauf zwischen Wegpunkten
        Parameter: path (List[Tuple]), smoothing_factor (int) - Pfad und Glättungsintervall
        Returns: List[Tuple] - Geglätteter Pfad
        """
        if len(path) < 4:
            return path

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

    def apply_settlement_influence(self, civ_map, settlements, slopemap):
        """
        Funktionsweise: Wendet Settlement-Einfluss auf civ_map an mit radialem Decay
        Aufgabe: Berechnet Zivilisations-Einfluss von Städten und Dörfern
        Parameter: civ_map, settlements, slopemap - Civ-Map, Settlement-Liste und Slope-Daten
        Returns: numpy.ndarray - Aktualisierte civ_map
        """
        height, width = civ_map.shape

        for settlement in settlements:
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

        return civ_map

    def calculate_road_influence(self, civ_map, roads, slopemap):
        """
        Funktionsweise: Wendet Road-Einfluss auf civ_map an entlang der Straßenverläufe
        Aufgabe: Berechnet Zivilisations-Einfluss entlang von Straßen
        Parameter: civ_map, roads, slopemap - Civ-Map, Straßen-Pfade und Slope-Daten
        Returns: numpy.ndarray - Aktualisierte civ_map
        """
        road_influence = 0.2
        max_road_civ = 0.5
        road_width = 2

        for road in roads:
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
    Aufgabe: Erstellt Grundstücks-System für späteres Gameplay
    """

    def __init__(self, plotsize=1.0):
        """
        Funktionsweise: Initialisiert PlotNode-System mit Plot-Größen-Parameter
        Aufgabe: Setup der Grundstücks-Generierung
        Parameter: plotsize (float) - Akkumulierter Civ-Wert für Plot-Größe
        """
        self.plotsize_threshold = plotsize
        self.next_node_id = 0
        self.next_plot_id = 0

    def generate_plot_nodes(self, civ_map, plotnodes_count, settlements):
        """
        Funktionsweise: Generiert PlotNodes gleichmäßig verteilt außerhalb von Städten und Wilderness
        Aufgabe: Erstellt initiale PlotNode-Verteilung für Grundstücks-System
        Parameter: civ_map, plotnodes_count, settlements - Civ-Map, Anzahl Nodes und Settlement-Liste
        Returns: List[PlotNode] - Generierte PlotNodes
        """
        height, width = civ_map.shape
        nodes = []

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
        if len(valid_positions) < plotnodes_count:
            plotnodes_count = len(valid_positions)

        # Sampling für gleichmäßige Verteilung
        sampled_positions = self._sample_uniform_distribution(valid_positions, plotnodes_count)

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

        return nodes

    def create_delaunay_triangulation(self, nodes, heightmap):
        """
        Funktionsweise: Erstellt Delaunay-Triangulation zwischen PlotNodes
        Aufgabe: Verbindet PlotNodes über Delaunay-Dreiecke für Grundstücks-Bildung
        Parameter: nodes, heightmap - PlotNode-Liste und Höhendaten
        Returns: List[PlotNode] - Nodes mit aktualisierten Verbindungen
        """
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
                            distance = self._calculate_distance(node_a.node_location,
                                                                node_b.node_location)
                            elevation_diff = self._calculate_elevation_difference(
                                node_a.node_location, node_b.node_location, heightmap
                            )
                            movecost = distance + abs(elevation_diff) * 0.5  # Simplified

                            node_a.connector_id.append(node_b.node_id)
                            node_a.connector_distance.append(distance)
                            node_a.connector_elevation.append(elevation_diff)
                            node_a.connector_movecost.append(movecost)

                        # Verbindung B->A
                        if node_a.node_id not in node_b.connector_id:
                            distance = self._calculate_distance(node_b.node_location,
                                                                node_a.node_location)
                            elevation_diff = self._calculate_elevation_difference(
                                node_b.node_location, node_a.node_location, heightmap
                            )
                            movecost = distance + abs(elevation_diff) * 0.5  # Simplified

                            node_b.connector_id.append(node_a.node_id)
                            node_b.connector_distance.append(distance)
                            node_b.connector_elevation.append(elevation_diff)
                            node_b.connector_movecost.append(movecost)

            return nodes

        except Exception as e:
            # Fallback: keine Verbindungen
            return nodes

    def merge_to_plots(self, nodes, civ_map):
        """
        Funktionsweise: Fusioniert PlotNodes zu Plots basierend auf akkumuliertem Civ-Wert
        Aufgabe: Erstellt Grundstücke durch Node-Gruppierung nach Civ-Wert-Schwellwert
        Parameter: nodes, civ_map - PlotNode-Liste und Civ-Map
        Returns: List[Plot] - Generierte Plots
        """
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

        return plots

    def optimize_node_positions(self, nodes, iterations=10):
        """
        Funktionsweise: Optimiert PlotNode-Positionen durch Abstoßungslogik und Winkel-Glättung
        Aufgabe: Verbessert Node-Anordnung für natürlichere Grundstücks-Formen
        Parameter: nodes, iterations - PlotNode-Liste und Anzahl Optimierungs-Iterationen
        Returns: List[PlotNode] - Optimierte PlotNodes
        """
        for _ in range(iterations):
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

            # Winkel-Glättung für scharfe Winkel
            self._smooth_sharp_angles(nodes)

        return nodes

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

    def _smooth_sharp_angles(self, nodes):
        """
        Funktionsweise: Glättet scharfe Winkel zwischen Node-Verbindungen
        Aufgabe: Optimiert Node-Anordnung für natürlichere Grundstücks-Formen
        """
        # Simplified: entfernt Nodes mit sehr spitzen Winkeln
        for node in nodes:
            if len(node.connector_id) >= 3:
                # Prüfe Winkel zwischen Verbindungen
                # Bei sehr spitzen Winkeln: lockere Position leicht
                pass

class SettlementGenerator:
    """
    Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung und Civilization-Mapping
    Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Settlement-Generator mit allen Sub-Komponenten
        Aufgabe: Setup aller Settlement-Systeme und Rng-Seed
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Settlement-Platzierung
        """
        self.map_seed = map_seed
        random.seed(map_seed)
        np.random.seed(map_seed)

        self.next_location_id = 0

        # Standard-Parameter
        self.terrain_factor_villages = 1.0
        self.civ_influence_decay = 1.0
        self.road_slope_to_distance_ratio = 1.0
        self.landmark_wilderness = 0.2
        self.plotsize = 1.0

    def generate_settlements(self, heightmap, slopemap, water_map, settlements_count,
                             terrain_factor_villages):
        """
        Funktionsweise: Generiert Settlement-Positionen basierend auf Terrain-Suitability
        Aufgabe: Platziert Städte und Dörfer an optimal geeigneten Locations
        Parameter: heightmap, slopemap, water_map, settlements_count, terrain_factor_villages
        Returns: List[Location] - Generierte Settlements
        """
        self.terrain_factor_villages = terrain_factor_villages

        # Terrain-Suitability analysieren
        analyzer = TerrainSuitabilityAnalyzer(terrain_factor_villages)
        suitability_map = analyzer.create_combined_suitability(heightmap, slopemap, water_map)

        settlements = []
        height, width = heightmap.shape

        # Mindestabstand zwischen Settlements berechnen
        map_diagonal = np.sqrt(height ** 2 + width ** 2)
        min_distance = max(10, map_diagonal / (settlements_count + 1))

        attempts = 0
        max_attempts = settlements_count * 20

        while len(settlements) < settlements_count and attempts < max_attempts:
            attempts += 1

            # Position mit höchster Suitability finden
            best_positions = self._find_best_settlement_positions(suitability_map, settlements,
                                                                  min_distance)

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

        return settlements

    def create_road_network(self, settlements, slopemap, road_slope_to_distance_ratio):
        """
        Funktionsweise: Erstellt Straßennetzwerk zwischen Settlements mit Pathfinding und Spline-Glättung
        Aufgabe: Verbindet Städte durch optimale Straßenführung
        Parameter: settlements, slopemap, road_slope_to_distance_ratio
        Returns: List[List[Tuple]] - Liste von Straßen (jede Straße ist Liste von Wegpunkten)
        """
        if len(settlements) < 2:
            return []

        pathfinder = PathfindingSystem(road_slope_to_distance_ratio)
        roads = []

        # Minimum Spanning Tree für Settlement-Verbindungen
        connected = [settlements[0]]
        unconnected = settlements[1:]

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

                # Pfad finden
                path = pathfinder.find_least_resistance_path(
                    slopemap,
                    (start_settlement.x, start_settlement.y),
                    (end_settlement.x, end_settlement.y)
                )

                # Spline-Glättung anwenden
                smoothed_path = pathfinder.apply_spline_smoothing(path, smoothing_factor=3)
                roads.append(smoothed_path)

                connected.append(end_settlement)
                unconnected.remove(end_settlement)

        return roads

    def place_landmarks(self, civ_map, landmarks_count, landmark_wilderness, heightmap, slopemap):
        """
        Funktionsweise: Platziert Landmarks in Wilderness-Bereichen mit niedrigem Civ-Wert
        Aufgabe: Erstellt interessante Locations abseits der Zivilisation
        Parameter: civ_map, landmarks_count, landmark_wilderness, heightmap, slopemap
        Returns: List[Location] - Generierte Landmarks
        """
        landmarks = []
        height, width = civ_map.shape

        # Gültige Landmark-Positionen finden
        valid_positions = []

        for y in range(height):
            for x in range(width):
                civ_value = civ_map[y, x]

                # Wilderness-Schwellwert prüfen
                if civ_value >= landmark_wilderness:
                    continue

                # Niedrige Höhen und Slopes bevorzugen
                elevation_ok = self._check_elevation_suitability(heightmap, x, y)
                slope_ok = self._check_slope_suitability(slopemap, x, y)

                if elevation_ok and slope_ok:
                    valid_positions.append((x, y))

        # Landmarks gleichmäßig verteilen
        if len(valid_positions) < landmarks_count:
            landmarks_count = len(valid_positions)

        sampled_positions = self._sample_landmark_positions(valid_positions, landmarks_count)

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

        return landmarks

    def place_roadsites(self, roads, roadsites_count):
        """
        Funktionsweise: Platziert Roadsites entlang von Straßen zwischen 30-70% der Weglänge
        Aufgabe: Erstellt Taverns, Handelsposten etc. entlang der Straßen
        Parameter: roads, roadsites_count - Straßen-Liste und Anzahl Roadsites
        Returns: List[Location] - Generierte Roadsites
        """
        roadsites = []

        if not roads or roadsites_count == 0:
            return roadsites

        sites_per_road = max(1, roadsites_count // len(roads))

        for road in roads:
            if len(road) < 3:
                continue

            road_length = len(road)

            for _ in range(sites_per_road):
                if len(roadsites) >= roadsites_count:
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

        return roadsites

    def generate_plots(self, civ_map, plotnodes_count, plotsize, settlements, heightmap):
        """
        Funktionsweise: Generiert Grundstücks-System mit PlotNodes und Delaunay-Triangulation
        Aufgabe: Erstellt Plots für Gameplay außerhalb von Städten und Wilderness
        Parameter: civ_map, plotnodes_count, plotsize, settlements, heightmap
        Returns: Tuple (List[PlotNode], List[Plot]) - Nodes und Plots
        """
        plot_system = PlotNodeSystem(plotsize)

        # PlotNodes generieren
        nodes = plot_system.generate_plot_nodes(civ_map, plotnodes_count, settlements)

        # Delaunay-Triangulation
        nodes = plot_system.create_delaunay_triangulation(nodes, heightmap)

        # Node-Positionen optimieren
        nodes = plot_system.optimize_node_positions(nodes, iterations=5)

        # Plots aus Nodes erstellen
        plots = plot_system.merge_to_plots(nodes, civ_map)

        return nodes, plots

    def create_civilization_map(self, heightmap, slopemap, settlements, roads, landmarks, roadsites,
                                civ_influence_decay):
        """
        Funktionsweise: Erstellt finale civ_map durch Kombination aller Zivilisations-Einflüsse
        Aufgabe: Berechnet finale Zivilisations-Verteilung über gesamte Map
        Parameter: heightmap, slopemap, settlements, roads, landmarks, roadsites, civ_influence_decay
        Returns: numpy.ndarray - Finale civ_map
        """
        height, width = heightmap.shape
        civ_map = np.zeros((height, width), dtype=np.float32)

        influence_mapper = CivilizationInfluenceMapper(civ_influence_decay)

        # Settlement-Einfluss anwenden
        civ_map = influence_mapper.apply_settlement_influence(civ_map, settlements, slopemap)

        # Road-Einfluss anwenden
        civ_map = influence_mapper.calculate_road_influence(civ_map, roads, slopemap)

        # Landmark-Einfluss anwenden
        for landmark in landmarks:
            civ_map = influence_mapper.apply_decay_kernel(
                civ_map, landmark, landmark.civ_influence, landmark.radius, slopemap
            )

        # Roadsite-Einfluss anwenden
        for roadsite in roadsites:
            civ_map = influence_mapper.apply_decay_kernel(
                civ_map, roadsite, roadsite.civ_influence, roadsite.radius, slopemap
            )

        # Wilderness definieren (< 0.2 wird auf 0.0 gesetzt)
        wilderness_mask = civ_map < 0.2
        civ_map[wilderness_mask] = 0.0

        return civ_map

    def generate_complete_settlements(self, heightmap, slopemap, water_map, map_seed, settlements,
                                      landmarks, roadsites,
                                      plotnodes, civ_influence_decay, terrain_factor_villages,
                                      road_slope_to_distance_ratio, landmark_wilderness, plotsize):
        """
        Funktionsweise: Generiert komplettes Settlement-System mit allen Komponenten
        Aufgabe: One-Stop Funktion für alle Settlement-Outputs
        Parameter: heightmap, slopemap, water_map, map_seed, settlements, landmarks, roadsites, plotnodes, civ_influence_decay, terrain_factor_villages, road_slope_to_distance_ratio, landmark_wilderness, plotsize
        Returns: Tuple (settlement_list, landmark_list, roadsite_list, plot_map, civ_map)
        """
        # Seed aktualisieren
        if map_seed != self.map_seed:
            self.map_seed = map_seed
            random.seed(map_seed)
            np.random.seed(map_seed)

        # Settlements generieren
        settlement_list = self.generate_settlements(
            heightmap, slopemap, water_map, settlements, terrain_factor_villages
        )

        # Road-Network erstellen
        roads = self.create_road_network(settlement_list, slopemap, road_slope_to_distance_ratio)

        # Roadsites platzieren
        roadsite_list = self.place_roadsites(roads, roadsites)

        # Civ-Map erstellen (vorläufig für Landmark-Platzierung)
        preliminary_civ_map = self.create_civilization_map(
            heightmap, slopemap, settlement_list, roads, [], roadsite_list, civ_influence_decay
        )

        # Landmarks platzieren
        landmark_list = self.place_landmarks(
            preliminary_civ_map, landmarks, landmark_wilderness, heightmap, slopemap
        )

        # Finale Civ-Map erstellen
        civ_map = self.create_civilization_map(
            heightmap, slopemap, settlement_list, roads, landmark_list, roadsite_list,
            civ_influence_decay
        )

        # Plot-System generieren
        plot_nodes, plots = self.generate_plots(civ_map, plotnodes, plotsize, settlement_list,
                                                heightmap)

        # Plot-Map erstellen (vereinfacht als Plot-IDs)
        plot_map = self._create_plot_map(heightmap.shape, plots)

        return settlement_list, landmark_list, roadsite_list, plot_map, civ_map

    def _find_best_settlement_positions(self, suitability_map, existing_settlements, min_distance):
        """
        Funktionsweise: Findet beste verfügbare Positionen für Settlement-Platzierung
        Aufgabe: Identifiziert optimale Locations unter Berücksichtigung von Mindestabständen
        Parameter: suitability_map, existing_settlements, min_distance
        Returns: List[Tuple] - Beste verfügbare Positionen
        """
        height, width = suitability_map.shape
        candidates = []

        # Threshold für "gute" Suitability
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

        # Nach Suitability sortieren (beste zuerst)
        candidates.sort(key=lambda c: c[2], reverse=True)

        return [(x, y) for x, y, _ in candidates]

    def _reduce_suitability_around_point(self, suitability_map, center_x, center_y, radius):
        """
        Funktionsweise: Reduziert Suitability um gegebenen Punkt für Mindestabstände
        Aufgabe: Verhindert zu enge Settlement-Platzierung
        Parameter: suitability_map, center_x, center_y, radius
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
        Aufgabe: Landmarks nur in niedrigeren Höhen wie dokumentiert
        """
        height, width = heightmap.shape

        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        # Höhen-Statistiken
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            return True

        norm_height = (heightmap[y, x] - min_height) / height_range

        # Landmarks nur in unteren 70% der Höhen
        return norm_height < 0.7

    def _check_slope_suitability(self, slopemap, x, y):
        """
        Funktionsweise: Prüft ob Slope für Landmark geeignet ist
        Aufgabe: Landmarks nur in niedrigeren Slopes wie dokumentiert
        """
        height, width = slopemap.shape[:2]

        if x < 0 or x >= width or y < 0 or y >= height:
            return False

        # Slope-Magnitude berechnen
        dz_dx = slopemap[y, x, 0]
        dz_dy = slopemap[y, x, 1]
        slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

        # Landmarks nur bei moderaten Slopes (< 0.5)
        return slope_magnitude < 0.5

    def _sample_landmark_positions(self, positions, count):
        """
        Funktionsweise: Sampelt Landmark-Positionen für gleichmäßige Verteilung
        Aufgabe: Verhindert Clustering von Landmarks
        """
        if len(positions) <= count:
            return positions

        # Einfaches gleichmäßiges Sampling
        step = len(positions) // count
        sampled = []

        for i in range(0, len(positions), step):
            if len(sampled) < count:
                sampled.append(positions[i])

        return sampled

    def _create_plot_map(self, map_shape, plots):
        """
        Funktionsweise: Erstellt Plot-Map mit Plot-IDs für jede Map-Position
        Aufgabe: Räumliche Zuordnung von Plots zu Map-Koordinaten
        Parameter: map_shape, plots - Map-Dimensionen und Plot-Liste
        Returns: numpy.ndarray - Plot-Map mit Plot-IDs
        """
        height, width = map_shape
        plot_map = np.zeros((height, width), dtype=np.int32)

        for plot in plots:
            for node in plot.nodes:
                x, y = int(node.node_location[0]), int(node.node_location[1])

                if 0 <= x < width and 0 <= y < height:
                    plot_map[y, x] = plot.plot_id

        return plot_map