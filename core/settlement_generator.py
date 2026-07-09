"""
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
- heightmap (2D-Array in meter Altitude)
- slopemap (2D-Array in m/m mit dz/dx, dz/dy)
- water_map (2D-Array mit Wasser-Klassifikation)
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
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt
import heapq
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import random


class SettlementData:
    """
    Funktionsweise: Container für alle Settlement-Daten mit Status-System und LOD-Management
    Aufgabe: Speichert alle internen und externen Maps mit Validity-State und Parameter-Hash
    """

    def __init__(self):
        # Externe Outputs
        self.settlement_list = []  # List[Location] - Alle Settlements
        self.landmark_list = []  # List[Location] - Alle Landmarks
        self.roadsite_list = []  # List[Location] - Alle Roadsites
        self.plot_map = None  # (height, width) - Plot-IDs
        self.civ_map = None  # (height, width) - Zivilisations-Einfluss
        self.plot_nodes = []  # List[PlotNode] - Alle PlotNodes
        self.plots = []  # List[Plot] - Alle Plots
        self.roads = []  # List[List[Tuple]] - Alle Road-Pfade
        self.city_mask = None  # (height, width) - Settlement-ID pro Pixel, -1 = ausserhalb jeder Stadt
        self.voronoi_cell_map = None  # (height, width) - Landschafts-Plot-Zell-ID pro Pixel, -1 = Stadt/Wilderness
        self.street_mask = None  # (height, width) bool - innerstaedtisches Strassenraster
        self.house_parcel_map = None  # (height, width) - kartenweit eindeutige Hausparzellen-ID, -1 = keine Parzelle
        self.landmark_roads = []  # List[List[Tuple]] - Landmark-Anbindungen ans Strassennetz
        self.outer_roads = []  # List[List[Tuple]] - Aussenverbindungen zur Kartengrenze
        self.plot_edges = {}  # Dict[int, PlotEdge] - adressierbares Kanten-Registry mit Traffic/Klassifikation

        # Interne Daten
        self.combined_suitability_map = None  # Terrain-Suitability für Settlement-Platzierung

        # Status-Attribute für jeden Berechnungsschritt
        self.terrain_suitability_valid = False
        self.settlements_valid = False
        self.road_network_valid = False
        self.roadsites_valid = False
        self.civilization_mapping_valid = False
        self.landmarks_valid = False
        self.plots_valid = False

        # LOD-Tracking
        self.lod_level = "LOD64"  # Aktueller LOD-Level
        self.actual_size = 64  # Tatsächliche Kartengröße
        self.validity_state = {}  # Validity-State pro LOD-Level
        self.parameter_hash = None  # Parameter-Hash für Cache-Validation
        self.parameters = {}  # Verwendete Parameter für Cache-Management

    def is_step_valid(self, step: str) -> bool:
        """
        Funktionsweise: Prüft ob einzelner Berechnungsschritt valid ist
        Parameter: step (str) - Name des Berechnungsschritts
        Returns: bool - True wenn Schritt valid
        """
        step_mapping = {
            'terrain_suitability': self.terrain_suitability_valid,
            'settlements': self.settlements_valid,
            'road_network': self.road_network_valid,
            'roadsites': self.roadsites_valid,
            'civilization_mapping': self.civilization_mapping_valid,
            'landmarks': self.landmarks_valid,
            'plots': self.plots_valid
        }
        return step_mapping.get(step, False)

    def invalidate_step(self, step: str):
        """
        Funktionsweise: Invalidiert einzelnen Berechnungsschritt
        Parameter: step (str) - Name des zu invalidierenden Schritts
        """
        if step == 'terrain_suitability':
            self.terrain_suitability_valid = False
        elif step == 'settlements':
            self.settlements_valid = False
        elif step == 'road_network':
            self.road_network_valid = False
        elif step == 'roadsites':
            self.roadsites_valid = False
        elif step == 'civilization_mapping':
            self.civilization_mapping_valid = False
        elif step == 'landmarks':
            self.landmarks_valid = False
        elif step == 'plots':
            self.plots_valid = False

    def get_step_status(self, step: str) -> dict:
        """
        Funktionsweise: Gibt detaillierten Status eines Berechnungsschritts zurück
        Parameter: step (str) - Name des Berechnungsschritts
        Returns: dict - Detaillierter Status mit Metadaten
        """
        status = {
            'valid': self.is_step_valid(step),
            'lod_level': self.lod_level,
            'data_available': False,
            'details': {}
        }

        if step == 'terrain_suitability':
            status['data_available'] = self.combined_suitability_map is not None
        elif step == 'settlements':
            status['data_available'] = len(self.settlement_list) > 0
            status['details']['settlement_count'] = len(self.settlement_list)
        elif step == 'road_network':
            status['data_available'] = len(self.roads) > 0
            status['details']['road_count'] = len(self.roads)
        elif step == 'roadsites':
            status['data_available'] = len(self.roadsite_list) > 0
            status['details']['roadsite_count'] = len(self.roadsite_list)
        elif step == 'civilization_mapping':
            status['data_available'] = self.civ_map is not None
        elif step == 'landmarks':
            status['data_available'] = len(self.landmark_list) > 0
            status['details']['landmark_count'] = len(self.landmark_list)
        elif step == 'plots':
            status['data_available'] = len(self.plots) > 0
            status['details']['plot_count'] = len(self.plots)

        return status


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
    connector_edge_id: List[int]
    settlement_id: int = -1  # >=0: dieser Node IST der Marktplatz von Settlement.location_id
    # (Nutzer-Vorgabe: "eine Stadt hat je einen Marktplatz, der zugleich ein
    # Node ist") - ein echter, in der Delaunay-Triangulation/im Traffic-Graph
    # ganz normal teilnehmender PlotNode statt eines separaten virtuellen
    # Anker-Knotens. -1 = normaler Plot-Node ohne Siedlungsbezug.


@dataclass
class PlotEdge:
    """
    Funktionsweise: Adressierbare Kante zwischen zwei PlotNodes (z.B. "Plotnode
    234 und 260 teilen sich Kante 839", Nutzer-Vorgabe) - Grundlage für die
    Familien-/Verkehrssimulation in PlotNodeSystem.simulate_plot_traffic().
    Aufgabe: Traegt Laenge, kumulierte Hoehenueberbrueckung (Wegintegral statt
    reiner Endpunkt-Differenz) und den daraus abgeleiteten Traffic-Wert, der
    die Kante am Ende als "none"/"path"/"road" klassifiziert.
    """
    edge_id: int
    node_a: int
    node_b: int
    length: float
    height_cost: float  # kumulierte |Höhenänderung| entlang der Linie (Wegintegral)
    movement_cost: float  # length * (1 + height_cost_factor * mittlere Steigung)
    traffic: float = 0.0  # fraktional: eine PlotNode teilt ihre "Familien-Masse" per Rang-Distanz-Gewicht auf mehrere Ziele auf
    classification: str = "none"  # "none" | "path" | "road"


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

    def __init__(self, terrain_factor_villages=1.0, map_size=64):
        """
        Funktionsweise: Initialisiert Suitability-Analyzer mit Terrain-Gewichtungsfaktor und LOD
        Aufgabe: Setup der Terrain-Analyse-Parameter mit LOD-Anpassung
        Parameter: terrain_factor_villages (float), map_size (int) - Gewichtung und tatsächliche Pixel-Größe
        """
        self.terrain_factor = terrain_factor_villages
        self.map_size = map_size

        # Größenabhängige Optimierungen (map_size ist die tatsächliche Pixel-Auflösung
        # der übergebenen Arrays, nicht das alte string-basierte LOD-Label)
        if map_size <= 64:
            self.analysis_detail, self.max_distance_check = 0.5, 20
        elif map_size <= 128:
            self.analysis_detail, self.max_distance_check = 0.7, 30
        elif map_size <= 256:
            self.analysis_detail, self.max_distance_check = 1.0, 40
        else:
            self.analysis_detail, self.max_distance_check = 1.0, 50

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
        detail_level = self.analysis_detail

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

        # Vektorisiert statt O(H*W*Wasserpixel) Python-Doppelschleife (jeder Pixel
        # berechnete zuvor seine Distanz zu JEDEM Wasser-Pixel einzeln neu) -
        # distance_transform_edt() liefert dieselbe "Distanz zum naechsten
        # True-Pixel" exakt, nur in O(H*W). War die Hauptursache fuer sehr lange
        # bzw. haengende Settlement-Generierung bei groesseren/wasserreichen
        # Karten (siehe docs/backlog.md Ticket #4 Performance-Hinweis).
        water_mask = water_map > 0
        if not np.any(water_mask):
            return np.zeros((height, width), dtype=np.float32)

        min_distance = distance_transform_edt(~water_mask)

        # LOD-abhängige Maximal-Distanz: alles darueber bleibt 0.0 (wie vorher)
        min_distance = np.where(min_distance > self.max_distance_check, np.inf, min_distance)

        # Optimal: 2-10 Pixel Entfernung
        # Akzeptabel: bis 20 Pixel
        # Ungeeignet: > 30 Pixel oder direkt auf Wasser
        water_suitability = np.zeros((height, width), dtype=np.float32)
        near_mask = min_distance < 2
        water_suitability[near_mask] = (min_distance[near_mask] / 2.0).astype(np.float32)  # zu nah (inkl. direkt auf Wasser = 0)
        optimal_mask = (min_distance >= 2) & (min_distance <= 10)
        water_suitability[optimal_mask] = 1.0  # Optimal
        good_mask = (min_distance > 10) & (min_distance <= 20)
        if np.any(good_mask):
            water_suitability[good_mask] = 1.0 - (min_distance[good_mask] - 10) / 10 * 0.5
        ok_mask = (min_distance > 20) & (min_distance <= 30)
        if np.any(ok_mask):
            water_suitability[ok_mask] = 0.5 - (min_distance[ok_mask] - 20) / 10 * 0.5
        # > 30 (oder > max_distance_check, oben auf inf gesetzt) bleibt 0.0

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


def _voronoi_edge_distance_map(cell_map):
    """
    Distanz in Pixeln zur naechsten Voronoi-Zellgrenze (inkl. Stadtgrenze, da
    Stadt-Pixel im cell_map bereits als -1 maskiert sind - ein Uebergang von Stadt
    zu Landschafts-Zelle zaehlt hier bewusst ebenfalls als "Grenze", Strassen aus
    der Stadt heraus sollen sich ja ebenso an ihr orientieren).
    Grundlage fuer den Edge-Bias in PathfindingSystem (Nutzer-Vorgabe: Wege
    zwischen Siedlungen sollen entlang der Plot-Grenzen verlaufen statt geradewegs
    durch die Zellen).
    """
    edge_mask = np.zeros(cell_map.shape, dtype=bool)
    edge_mask[:-1, :] |= cell_map[:-1, :] != cell_map[1:, :]
    edge_mask[1:, :] |= cell_map[:-1, :] != cell_map[1:, :]
    edge_mask[:, :-1] |= cell_map[:, :-1] != cell_map[:, 1:]
    edge_mask[:, 1:] |= cell_map[:, :-1] != cell_map[:, 1:]

    if not np.any(edge_mask):
        return np.full(cell_map.shape, np.inf, dtype=np.float32)

    return distance_transform_edt(~edge_mask).astype(np.float32)


class PathfindingSystem:
    """
    Funktionsweise: Findet Wege geringsten Widerstands zwischen Settlements für Straßen
    Aufgabe: Erstellt realistische Straßenverbindungen mit Spline-Interpolation und LOD-Optimierung
    """

    def __init__(self, road_slope_to_distance_ratio=1.0, map_size=64,
                 edge_distance_map=None, edge_bias=0.0, edge_bias_scale=16.0):
        """
        Funktionsweise: Initialisiert Pathfinding-System mit Slope-Distance-Gewichtung und LOD
        Aufgabe: Setup der Pathfinding-Parameter mit LOD-Anpassung
        Parameter: road_slope_to_distance_ratio (float), map_size (int) - Gewichtung und tatsächliche Pixel-Größe
        Parameter: edge_distance_map - optionale (H,W)-Distanz zur naechsten Voronoi-
            Zellgrenze (siehe _voronoi_edge_distance_map()); None = kein Edge-Bias
            (Legacy-Verhalten, reine Slope-Kosten).
        Parameter: edge_bias - Staerke der Bevorzugung von Zellgrenzen (0 = aus)
        Parameter: edge_bias_scale - charakteristische Distanz (Pixel), ueber die
            der Edge-Bias von "billig direkt auf der Grenze" zu "voller Straf-
            aufschlag weit von jeder Grenze" saettigt (typischerweise die
            Voronoi-Seed-Spacing-Groessenordnung).
        """
        self.slope_distance_ratio = road_slope_to_distance_ratio
        self.map_size = map_size
        self.edge_distance_map = edge_distance_map
        self.edge_bias = edge_bias
        self.edge_bias_scale = max(1e-3, edge_bias_scale)

        # Größenabhängige Pathfinding-Optimierungen
        if map_size <= 64:
            self.max_search_nodes, self.path_resolution = 500, 2
        elif map_size <= 128:
            self.max_search_nodes, self.path_resolution = 1000, 1
        elif map_size <= 256:
            self.max_search_nodes, self.path_resolution = 2000, 1
        else:
            self.max_search_nodes, self.path_resolution = 5000, 1

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
        cost = base_cost + slope_penalty

        # Edge-Bias: guenstiger nahe einer Voronoi-Zellgrenze, saettigt Richtung
        # (1 + edge_bias) je weiter man sich von jeder Grenze entfernt.
        if self.edge_distance_map is not None and self.edge_bias > 0:
            distance_to_edge = self.edge_distance_map[y, x]
            if np.isfinite(distance_to_edge):
                cost *= 1.0 + self.edge_bias * (
                    distance_to_edge / (distance_to_edge + self.edge_bias_scale))

        return cost

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
        max_nodes = self.max_search_nodes
        path_resolution = self.path_resolution

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
        if self.map_size <= 64:
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


_VORONOI_NEIGHBOR_STEPS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.4142135), (1, -1, 1.4142135), (-1, 1, 1.4142135), (1, 1, 1.4142135),
)


def _terrain_cost_voronoi(heightmap, slopemap, seed_positions, terrain_factor, max_cost=None, valid_mask=None):
    """
    Multi-Source-Dijkstra ueber das Pixelgrid: jeder Pixel wird dem Seed mit der
    geringsten terrain-cost-gewichteten Distanz zugeordnet - CPU-Referenz fuer den
    spaeteren GPU-JFA-Shader (siehe shaders/water/jumpFloodLakes.comp fuer exakt
    dasselbe Muster bei der Lake-Detection, dort mit Hoehen- statt Slope-Kosten).

    effective_distance = geometrische Distanz * (1 + terrain_factor * slope_magnitude),
    dieselbe Formel-Familie wie im alten Settlement-Deskriptor
    (slope_factor = 1 + terrain_factor * slope_angle; effective_distance = distance * slope_factor).

    Parameter:
        seed_positions: Liste/Array von (x, y)-Tupeln, ein Seed pro Eintrag (Index = seed_id)
        max_cost: optionale Kappungsgrenze - Pixel jenseits davon bleiben unassigned (-1).
            Laesst den Flood bei kleinen, lokal begrenzten Grenzen (z.B. Stadtgrenzen)
            frueh terminieren statt die ganze Karte zu fluten.
        valid_mask: optionale bool-Maske (H,W) - der Flood propagiert nur innerhalb
            dieser Maske (genutzt vom Block-System, um Hausparzellen strikt auf
            das Stadt-Footprint zu begrenzen statt in die Landschaft auszulaufen).
    Returns: (nearest_seed_map int32 (H,W) mit -1 = unassigned, cost_map float32 (H,W))
    """
    height, width = heightmap.shape
    slope_magnitude = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2).astype(np.float32)

    nearest_seed = np.full((height, width), -1, dtype=np.int32)
    cost_map = np.full((height, width), np.inf, dtype=np.float32)

    heap = []
    for seed_id, (sx, sy) in enumerate(seed_positions):
        ix, iy = int(round(sx)), int(round(sy))
        if 0 <= ix < width and 0 <= iy < height and cost_map[iy, ix] > 0.0:
            if valid_mask is not None and not valid_mask[iy, ix]:
                continue
            cost_map[iy, ix] = 0.0
            nearest_seed[iy, ix] = seed_id
            heapq.heappush(heap, (0.0, ix, iy, seed_id))

    while heap:
        cost, x, y, seed_id = heapq.heappop(heap)
        if cost > cost_map[y, x]:
            continue  # veralteter Heap-Eintrag, bereits durch billigeren Pfad ueberholt
        for dx, dy, base_step in _VORONOI_NEIGHBOR_STEPS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if valid_mask is not None and not valid_mask[ny, nx]:
                continue
            step_cost = base_step * (1.0 + terrain_factor * slope_magnitude[ny, nx])
            new_cost = cost + step_cost
            if max_cost is not None and new_cost > max_cost:
                continue
            if new_cost < cost_map[ny, nx]:
                cost_map[ny, nx] = new_cost
                nearest_seed[ny, nx] = seed_id
                heapq.heappush(heap, (new_cost, nx, ny, seed_id))

    return nearest_seed, cost_map


def _terrain_cost_voronoi_gpu_or_cpu(shader_manager, heightmap, slopemap, seed_positions, terrain_factor, max_cost=None):
    """
    Versucht den terrain-cost-gewichteten Multi-Source-Flood auf der GPU (JFA-
    Approximation, siehe shaders/settlement/terrainCostFlood.comp), faellt bei
    fehlendem shader_manager oder GPU-Fehlern auf die exakte CPU-Dijkstra-
    Referenz zurueck (_terrain_cost_voronoi()) - dasselbe GPU->CPU-Fallback-
    Muster wie in core/water_generator.py (hier ohne separaten Simple-Fallback,
    da die CPU-Variante bereits die volle Referenzimplementierung ist).
    Returns: (nearest_seed_map int32 (H,W), cost_map float32 (H,W)) - identische
    Signatur zu _terrain_cost_voronoi(), damit beide Pfade austauschbar sind.
    """
    if shader_manager:
        try:
            result = shader_manager.request_shader_operation(
                "settlement", "terrainCostFlood",
                {"slopemap": slopemap, "seed_positions": seed_positions,
                 "terrain_factor": terrain_factor, "max_cost": max_cost},
                {}
            )
            if result.get("success"):
                return result["nearest_seed_map"], result["cost_map"]
        except Exception as e:
            logging.warning(f"GPU terrain-cost-flood fehlgeschlagen: {e}, Fallback auf CPU")

    return _terrain_cost_voronoi(heightmap, slopemap, seed_positions, terrain_factor, max_cost=max_cost)


class CityBoundaryAnalyzer:
    """
    Funktionsweise: Bestimmt die Stadtgrenze je Settlement ueber eine terrain-cost-
    gewichtete Distanz vom Stadtkern - auf flacher Distanz reicht die Stadt weiter,
    Haenge bremsen die Ausdehnung ab (Nutzer-Vorgabe: Mischung aus Abstand zum
    Stadtkern und Uberwindung von Hoehe).
    Aufgabe: Liefert eine city_mask (Settlement-ID pro Pixel, -1 = ausserhalb jeder
    Stadt) als harte Grenze zwischen Stadt-Innerem (feine Hausparzellen, siehe
    spaeteres Block-System) und Landschaft (grobe Voronoi-Felder, siehe
    LandscapeVoronoiSystem).
    """

    def __init__(self, terrain_factor=1.0, reach_factor=4.0, shader_manager=None):
        self.terrain_factor = terrain_factor
        self.reach_factor = reach_factor
        self.shader_manager = shader_manager

    def compute_city_boundaries(self, heightmap, slopemap, settlements, progress_callback=None):
        """
        Pro Settlement ein eigener kostenbegrenzter Flood (max_cost = radius *
        reach_factor) statt ein einziger globaler Flood ueber alle Seeds - bei nur
        wenigen Settlements (max 5, siehe SETTLEMENT.SETTLEMENTS) bleibt das
        guenstig, weil jeder Flood frueh an seiner eigenen Grenze abbricht, und
        erlaubt unterschiedlich grosse Staedte (je nach settlement.radius) mit
        jeweils eigenem max_cost statt einem gemeinsamen Cutoff. Nutzt denselben
        GPU/CPU-Flood wie LandscapeVoronoiSystem (siehe
        _terrain_cost_voronoi_gpu_or_cpu()), pro Settlement einzeln aufgerufen
        (Single-Seed), Ergebnisse werden ueber city_cost_map settlementuebergreifend
        gemergt (bester/niedrigster Cost gewinnt bei ueberlappenden Reichweiten).
        """
        height, width = heightmap.shape
        city_mask = np.full((height, width), -1, dtype=np.int32)
        city_cost_map = np.full((height, width), np.inf, dtype=np.float32)

        city_settlements = [s for s in settlements if s.location_type == 'settlement']
        for i, settlement in enumerate(city_settlements):
            if progress_callback:
                progress_callback(
                    "City Boundary", 30 + (i * 3) // max(1, len(city_settlements)),
                    f"Computing city boundary {i + 1}/{len(city_settlements)}...")

            max_cost = settlement.radius * self.reach_factor
            seed_nearest, seed_cost = _terrain_cost_voronoi_gpu_or_cpu(
                self.shader_manager, heightmap, slopemap, [(settlement.x, settlement.y)],
                self.terrain_factor, max_cost=max_cost)

            reached = seed_nearest >= 0
            better = reached & (seed_cost < city_cost_map)
            city_cost_map[better] = seed_cost[better]
            city_mask[better] = settlement.location_id

        return city_mask, city_cost_map


class LandscapeVoronoiSystem:
    """
    Funktionsweise: Grobe Feld-Voronoi-Zellen fuer die Landschaft ausserhalb von
    Staedten (siehe CityBoundaryAnalyzer) und Wilderness (siehe Wilderness-
    Threshold in CivilizationInfluenceMapper). Seedpunkte stossen sich
    physikalisch ab, Hoehe/Slope modulieren die Abstossung (steile Gegenden ->
    dichtere Seeds -> kleinere Zellen - Nutzer-Vorgabe: "an steilen hügeligen
    Gegenden sind die Plots in der Realität kleiner"). Zellzuordnung ueber
    denselben terrain-cost-gewichteten Multi-Source-Flood wie CityBoundaryAnalyzer
    (siehe _terrain_cost_voronoi()).

    LOD-Kontinuitaet ("Gummiband", Nutzer-Vorgabe): SettlementGenerator uebergibt
    die Seed-Positionen der letzten LOD-Stufe als Warm-Start (siehe
    SettlementGenerator._calc_landscape_voronoi()) - die Relaxation baut jede
    LOD-Stufe auf der vorigen auf statt komplett neu zu wuerfeln, damit Plot-
    Grenzen zwischen LOD-Stufen nicht springen und sich Strassenwege ueber die
    Iterationen hinweg straffen koennen.
    """

    def __init__(self, terrain_factor=1.0, base_spacing=16.0, relax_iterations=4, shader_manager=None):
        self.terrain_factor = terrain_factor
        self.base_spacing = base_spacing
        self.relax_iterations = relax_iterations
        self.shader_manager = shader_manager

    def _local_min_spacing(self, slopemap, x, y):
        """Steilere Stellen -> kleinerer Mindestabstand -> dichtere Seeds -> kleinere Zellen."""
        dz_dx = slopemap[y, x, 0]
        dz_dy = slopemap[y, x, 1]
        slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        return self.base_spacing / (1.0 + self.terrain_factor * slope_magnitude)

    def generate_seeds(self, heightmap, slopemap, city_mask, target_count,
                        previous_seeds=None, rng=None):
        """
        Warm-Start aus previous_seeds (bereits auf die aktuelle Kartengroesse
        skaliert - siehe SettlementGenerator._calc_landscape_voronoi()), aufgefuellt
        per Rejection-Sampling bis target_count erreicht ist. Seeds werden nie
        innerhalb einer Stadtgrenze plaziert (city_mask >= 0). Reicht der Platz bei
        gegebenem Mindestabstand nicht fuer target_count Seeds, bricht das
        Rejection-Sampling nach max_attempts einfach mit weniger Seeds ab, statt
        zu haengen - fuer kleine LOD-Stufen mit wenigen freien Pixeln erwuenscht.
        """
        rng = rng or random
        height, width = heightmap.shape

        # Spatial-Hash-Grid statt linearem Scan ueber alle bisherigen Seeds:
        # bei sehr steilem Terrain kann _local_min_spacing() auf einen winzigen
        # Bruchteil von base_spacing schrumpfen, wodurch weit mehr Seeds passen
        # als bei flachem Terrain angenommen - ein linearer "for (sx,sy) in seeds"-
        # Scan pro Versuch wurde dann selbst bei nur ~1000 Zielwerten spuerbar
        # langsam (siehe docs/backlog.md Ticket #4, "hängt bei LOD1"-Report).
        # Zellgroesse = base_spacing (flaches Terrain als obere Schranke) haelt
        # die pro Versuch zu pruefende Nachbarschaft klein und konstant.
        cell_size = max(1e-3, self.base_spacing)
        grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

        def cell_of(x, y):
            return (int(x // cell_size), int(y // cell_size))

        def is_too_close(x, y, min_spacing):
            cx, cy = cell_of(x, y)
            span = int(min_spacing // cell_size) + 1
            for gx in range(cx - span, cx + span + 1):
                for gy in range(cy - span, cy + span + 1):
                    for (sx, sy) in grid.get((gx, gy), ()):
                        if (sx - x) ** 2 + (sy - y) ** 2 < min_spacing ** 2:
                            return True
            return False

        def insert(x, y):
            grid.setdefault(cell_of(x, y), []).append((x, y))

        seeds = []
        if previous_seeds is not None:
            for (x, y) in previous_seeds:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < width and 0 <= iy < height and city_mask[iy, ix] < 0:
                    seeds.append((float(x), float(y)))
                    insert(float(x), float(y))
                if len(seeds) >= target_count:
                    break

        max_attempts = max(50, target_count * 30)
        attempts = 0
        while len(seeds) < target_count and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(0, width - 1)
            y = rng.uniform(0, height - 1)
            ix, iy = int(x), int(y)
            if city_mask[iy, ix] >= 0:
                continue
            min_spacing = self._local_min_spacing(slopemap, ix, iy)
            if is_too_close(x, y, min_spacing):
                continue
            seeds.append((x, y))
            insert(x, y)

        return seeds

    def relax(self, seeds, heightmap, slopemap, city_mask, progress_callback=None):
        """
        Iterative paarweise Abstossung ueber cKDTree-Nachbarschaftssuche. Ziel-
        abstand pro Seed-Paar = Mittelwert der beiden lokalen Mindestabstaende
        (terrain-moduliert). Seeds, die durch die Abstossung in eine Stadtgrenze
        gedrueckt wuerden, bleiben auf ihrer alten Position stehen statt die
        Stadtgrenze zu verletzen.
        """
        if len(seeds) < 2:
            return seeds

        height, width = heightmap.shape
        positions = np.array(seeds, dtype=np.float64)
        # Einmal vorab (statt pro Paar ueber _local_min_spacing()) - bei vielen
        # Tausend Nachbar-Paaren pro Iteration dominierte vorher der reine
        # Python/numpy-Scalar-Call-Overhead (np.clip()/np.hypot() pro Paar in
        # einer Python-Schleife) die Laufzeit, nicht die Relaxation selbst -
        # siehe Profiling in docs/backlog.md Ticket #4 ("hängt bei LOD1").
        slope_magnitude = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2)

        for iteration in range(self.relax_iterations):
            if progress_callback:
                progress_callback(
                    "Landscape Voronoi", 35 + (iteration * 5) // max(1, self.relax_iterations),
                    f"Relaxing plot seeds (iteration {iteration + 1}/{self.relax_iterations})...")

            tree = cKDTree(positions)
            max_spacing = self.base_spacing * 2.0
            pairs = tree.query_pairs(r=max_spacing, output_type='ndarray')
            if len(pairs) == 0:
                break

            displacement = np.zeros_like(positions)

            i_idx, j_idx = pairs[:, 0], pairs[:, 1]
            pos_i, pos_j = positions[i_idx], positions[j_idx]

            ix = np.clip(pos_i[:, 0], 0, width - 1).astype(np.intp)
            iy = np.clip(pos_i[:, 1], 0, height - 1).astype(np.intp)
            jx = np.clip(pos_j[:, 0], 0, width - 1).astype(np.intp)
            jy = np.clip(pos_j[:, 1], 0, height - 1).astype(np.intp)

            spacing_i = self.base_spacing / (1.0 + self.terrain_factor * slope_magnitude[iy, ix])
            spacing_j = self.base_spacing / (1.0 + self.terrain_factor * slope_magnitude[jy, jx])
            target_spacing = 0.5 * (spacing_i + spacing_j)

            delta = pos_i - pos_j
            dist = np.hypot(delta[:, 0], delta[:, 1])
            valid = (dist >= 1e-6) & (dist < target_spacing)

            push = np.zeros_like(dist)
            push[valid] = (target_spacing[valid] - dist[valid]) / target_spacing[valid] * 0.5
            direction = np.zeros_like(delta)
            direction[valid] = delta[valid] / dist[valid, np.newaxis]

            contribution = direction * push[:, np.newaxis]
            np.add.at(displacement, i_idx, contribution)
            np.add.at(displacement, j_idx, -contribution)

            new_positions = positions + displacement
            new_positions[:, 0] = np.clip(new_positions[:, 0], 0, width - 1)
            new_positions[:, 1] = np.clip(new_positions[:, 1], 0, height - 1)

            # Stadtgrenzen respektieren: Seeds, die hineingedrueckt wuerden, bleiben stehen
            for idx in range(len(new_positions)):
                nx, ny = int(new_positions[idx, 0]), int(new_positions[idx, 1])
                if city_mask[ny, nx] >= 0:
                    new_positions[idx] = positions[idx]

            positions = new_positions

        return [(float(x), float(y)) for x, y in positions]

    def assign_cells(self, heightmap, slopemap, seeds, city_mask, progress_callback=None):
        """
        Zellzuordnung ueber denselben terrain-cost-Flood wie die Stadtgrenze
        (siehe _terrain_cost_voronoi_gpu_or_cpu() - versucht GPU-JFA, faellt auf
        CPU-Dijkstra zurueck). Der Flood selbst kennt die Stadtgrenze nicht
        (propagiert ungehindert durch das gesamte Grid) - Stadt-Pixel werden
        deshalb nachtraeglich aus dem Ergebnis maskiert, damit Landschafts-Zellen
        nie mit Stadt-Innerem ueberlappen (das gehoert dem separaten Block-System,
        siehe CityBlockSystem).
        """
        if progress_callback:
            progress_callback("Landscape Voronoi", 42, "Assigning Voronoi cells...")
        height, width = heightmap.shape
        if not seeds:
            return np.full((height, width), -1, dtype=np.int32)
        cell_map, _ = _terrain_cost_voronoi_gpu_or_cpu(self.shader_manager, heightmap, slopemap, seeds, self.terrain_factor)
        cell_map[city_mask >= 0] = -1
        return cell_map


class CityBlockSystem:
    """
    Funktionsweise: Feines innerstaedtisches Strassenraster + Hausparzellen
    innerhalb einer Stadtgrenze (siehe CityBoundaryAnalyzer) - ein eigener,
    dichterer Mechanismus als das grobe Landschafts-Voronoi (siehe
    LandscapeVoronoiSystem), naeher am Referenzbild orientiert: zuerst ein
    Strassen-Skelett (Minimum-Spanning-Tree zwischen Haus-Ankerpunkten), dann
    richten sich die Hausparzellen an diesen Strassen aus statt an abstrakten
    Zellgrenzen (Nutzer-Vorgabe: "Häuser richten sich nach Wegen aus, nicht
    nach abstrakten Zellgrenzen").

    Skaliert mit der verfuegbaren Pixel-Flaeche der Stadt (Nutzer-Vorgabe: bei
    wenigen Pixeln pro Stadt nur wenige Haeuser, waechst mit dem LOD mit) -
    target_houses ergibt sich aus footprint_area / house_spacing**2 statt einem
    festen Parameter, damit winzige Staedte bei niedrigem LOD nicht ueberfuellt
    werden.
    """

    def __init__(self, house_spacing=4.0):
        self.house_spacing = house_spacing

    def build_for_settlement(self, footprint_mask, rng):
        """
        Parameter: footprint_mask - bool (H,W), True wo city_mask == diese
            Settlement-ID (siehe CityBoundaryAnalyzer). rng - random.Random-
            Instanz fuer reproduzierbare Anker-Auswahl.
        Returns: dict mit street_mask (bool H,W), house_parcel_map
            (int32 H,W, lokale Parcel-ID, -1 = keine Parzelle/Strasse), anchors
        """
        height, width = footprint_mask.shape
        ys, xs = np.nonzero(footprint_mask)
        if len(xs) == 0:
            return {
                "street_mask": np.zeros((height, width), dtype=bool),
                "house_parcel_map": np.full((height, width), -1, dtype=np.int32),
                "anchors": [],
            }

        area = len(xs)
        target_houses = max(1, int(area / max(1.0, self.house_spacing ** 2)))
        anchors = self._sample_anchors(xs, ys, target_houses, rng)

        street_mask = np.zeros((height, width), dtype=bool)
        if len(anchors) >= 2:
            street_mask = self._build_street_skeleton(anchors, footprint_mask)

        # Hausparzellen: naechster Anker innerhalb des Footprints (rein geometrisch,
        # kein Terrain-Cost noetig auf dieser kleinen Skala), Strassenpixel ausgenommen.
        parcel_map = np.full((height, width), -1, dtype=np.int32)
        if anchors:
            flat_heightmap = np.zeros((height, width), dtype=np.float32)
            flat_slopemap = np.zeros((height, width, 2), dtype=np.float32)
            nearest, _ = _terrain_cost_voronoi(
                flat_heightmap, flat_slopemap, anchors, terrain_factor=0.0, valid_mask=footprint_mask)
            parcel_map[:] = nearest
            parcel_map[street_mask] = -1

        return {"street_mask": street_mask, "house_parcel_map": parcel_map, "anchors": anchors}

    def _sample_anchors(self, xs, ys, target_count, rng):
        """Rejection-Sampling mit Mindestabstand ueber die Footprint-Pixel selbst
        (statt kontinuierlichem uniform-Sampling) - garantiert, dass jeder Anker
        tatsaechlich innerhalb des (moeglicherweise sehr kleinen und unregel-
        maessig geformten) Stadt-Footprints liegt. Spatial-Hash-Grid statt
        linearem "all(...)"-Scan ueber alle bisherigen Anker (gleiches Muster
        wie LandscapeVoronoiSystem.generate_seeds(), siehe dortigen Kommentar
        zum Performance-Fix) - haelt die pro Kandidat zu pruefende Nachbarschaft
        klein und konstant statt mit der Anker-Anzahl zu wachsen."""
        candidates = list(zip(xs.tolist(), ys.tolist()))
        rng.shuffle(candidates)

        cell_size = max(1e-3, self.house_spacing)
        grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

        def cell_of(x, y):
            return (int(x // cell_size), int(y // cell_size))

        def is_too_close(x, y):
            cx, cy = cell_of(x, y)
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for (ax, ay) in grid.get((gx, gy), ()):
                        if (x - ax) ** 2 + (y - ay) ** 2 < self.house_spacing ** 2:
                            return True
            return False

        anchors = []
        for (x, y) in candidates:
            if len(anchors) >= target_count:
                break
            if not is_too_close(x, y):
                anchors.append((float(x), float(y)))
                grid.setdefault(cell_of(x, y), []).append((float(x), float(y)))
        if not anchors and candidates:
            anchors.append((float(candidates[0][0]), float(candidates[0][1])))
        return anchors

    def _build_street_skeleton(self, anchors, footprint_mask):
        """Minimum-Spanning-Tree zwischen Haus-Ankerpunkten als Strassen-Skelett."""
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.sparse import csr_matrix

        points = np.array(anchors)
        dist_matrix = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
        mst = minimum_spanning_tree(csr_matrix(dist_matrix)).toarray()

        street_mask = np.zeros(footprint_mask.shape, dtype=bool)
        edges = np.transpose(np.nonzero(mst))
        for i, j in edges:
            self._rasterize_line(street_mask, points[i], points[j], footprint_mask)
        return street_mask

    @staticmethod
    def _rasterize_line(street_mask, p0, p1, footprint_mask):
        """Lineare Rasterung zwischen zwei Ankerpunkten, auf den Footprint begrenzt."""
        height, width = street_mask.shape
        x0, y0 = p0
        x1, y1 = p1
        steps = max(1, int(round(np.hypot(x1 - x0, y1 - y0))))
        for t in np.linspace(0, 1, steps + 1):
            x = int(round(x0 + (x1 - x0) * t))
            y = int(round(y0 + (y1 - y0) * t))
            if 0 <= x < width and 0 <= y < height and footprint_mask[y, x]:
                street_mask[y, x] = True


class SettlementGenerator:
    """
    Funktionsweise: Hauptklasse für intelligente Settlement-Platzierung mit BaseGenerator-API und LOD-System
    Aufgabe: Koordiniert alle Settlement-Aspekte und erstellt civ_map mit Progress-Updates
    """

    def __init__(self, map_seed=42, shader_manager=None, data_lod_manager=None):
        """
        Funktionsweise: Initialisiert Settlement-Generator mit BaseGenerator und Sub-Komponenten
        Aufgabe: Setup aller Settlement-Systeme und Rng-Seed
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Settlement-Platzierung
        Parameter: shader_manager - optionaler ShaderManager für GPU-Compute (siehe
            CityBoundaryAnalyzer/LandscapeVoronoiSystem, shaders/settlement/terrainCostFlood.comp).
            None (Standalone/Tests) bedeutet reine CPU-Referenz, kein Verhaltensunterschied
            außer Performance.
        Parameter: data_lod_manager - DataLODManager für feingranularen Calculator-Storage
            (siehe set_calculator_output()/get_calculator_output()). Die echte Pipeline
            injiziert immer eine Instanz über GenerationOrchestrator.get_generator_instance();
            bleibt sie None (Standalone/Tests), wird beim ersten Bedarf lazy eine eigene erzeugt.
        """
        self.map_seed = map_seed
        self.logger = logging.getLogger(self.__class__.__name__)
        random.seed(map_seed)
        np.random.seed(map_seed)

        self.next_location_id = 0
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager

        # Progress-Callback (step_name, progress_percent, detail_message) -> None.
        # War nie initialisiert - _execute_generation() ruft self._update_progress()
        # an mehreren Stellen unbedingt auf (kein "if self._update_progress:"-Guard),
        # wodurch jede Settlement-Generierung sofort mit AttributeError abbrach.
        # No-op statt None, damit auch die unguarded Call-Sites sicher sind; ein
        # echter Callback kann jederzeit durch Zuweisung überschrieben werden.
        self._update_progress = lambda *args, **kwargs: None

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
        self.city_reach_factor = 4.0
        self.voronoi_base_spacing = 16.0
        self.voronoi_relax_iterations = 4
        self.road_voronoi_edge_bias = 1.5
        self.house_spacing = 4.0
        self.civ_influence_range = 0.30
        self.plot_base_spacing = 10.0
        self.plot_civ_spacing_factor = 3.0
        self.plot_height_cost_factor = 2.0
        self.plot_path_traffic_threshold = 25
        self.plot_road_traffic_threshold = 75
        self.plot_intercity_traffic = 30
        self.plot_traffic_attraction = 0.05

    def set_active_parameters(self, parameters):
        """
        Setzt die Parameter, die alle calculate_*()/_calc_*-Methoden bis zur
        nächsten frischen Anfrage verwenden (vom GenerationOrchestrator
        aufgerufen). Settlement speichert Parameter als Instanz-Attribute
        (self.settlements etc.), nicht als eigenes dict - entspricht dem, was
        _execute_generation() vorher direkt inline gemacht hat.
        """
        self.settlements = parameters['settlements']
        self.landmarks = parameters['landmarks']
        self.roadsites = parameters['roadsites']
        self.plotnodes = parameters['plotnodes']
        self.civ_influence_decay = parameters['civ_influence_decay']
        self.terrain_factor_villages = parameters['terrain_factor_villages']
        self.road_slope_to_distance_ratio = parameters['road_slope_to_distance_ratio']
        self.landmark_wilderness = parameters['landmark_wilderness']
        self.plotsize = parameters['plotsize']
        self.city_reach_factor = parameters['city_reach_factor']
        self.voronoi_base_spacing = parameters['voronoi_base_spacing']
        self.voronoi_relax_iterations = parameters['voronoi_relax_iterations']
        self.road_voronoi_edge_bias = parameters['road_voronoi_edge_bias']
        self.house_spacing = parameters['house_spacing']
        self.civ_influence_range = parameters['civ_influence_range']
        self.plot_base_spacing = parameters['plot_base_spacing']
        self.plot_civ_spacing_factor = parameters['plot_civ_spacing_factor']
        self.plot_height_cost_factor = parameters['plot_height_cost_factor']
        self.plot_path_traffic_threshold = parameters['plot_path_traffic_threshold']
        self.plot_road_traffic_threshold = parameters['plot_road_traffic_threshold']
        self.plot_intercity_traffic = parameters['plot_intercity_traffic']
        self.plot_traffic_attraction = parameters['plot_traffic_attraction']

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, _execute_generation() ohne
        injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from gui.OldManagers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

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
            'plotsize': SETTLEMENT.PLOTSIZE["default"],
            'city_reach_factor': SETTLEMENT.CITY_REACH_FACTOR["default"],
            'voronoi_base_spacing': SETTLEMENT.VORONOI_BASE_SPACING["default"],
            'voronoi_relax_iterations': SETTLEMENT.VORONOI_RELAX_ITERATIONS["default"],
            'road_voronoi_edge_bias': SETTLEMENT.ROAD_VORONOI_EDGE_BIAS["default"],
            'house_spacing': SETTLEMENT.HOUSE_SPACING["default"],
            'civ_influence_range': SETTLEMENT.CIV_INFLUENCE_RANGE["default"],
            'plot_base_spacing': SETTLEMENT.PLOT_BASE_SPACING["default"],
            'plot_civ_spacing_factor': SETTLEMENT.PLOT_CIV_SPACING_FACTOR["default"],
            'plot_height_cost_factor': SETTLEMENT.PLOT_HEIGHT_COST_FACTOR["default"],
            'plot_path_traffic_threshold': SETTLEMENT.PLOT_PATH_TRAFFIC_THRESHOLD["default"],
            'plot_road_traffic_threshold': SETTLEMENT.PLOT_ROAD_TRAFFIC_THRESHOLD["default"],
            'plot_intercity_traffic': SETTLEMENT.PLOT_INTERCITY_TRAFFIC["default"],
            'plot_traffic_attraction': SETTLEMENT.PLOT_TRAFFIC_ATTRACTION["default"]
        }

    def _get_dependencies(self, data_manager, lod_level=None):
        """
        Funktionsweise: Holt benötigte Dependencies mit intelligenten Fallback-Werten
        Aufgabe: Dependency-Resolution für Settlement-Generierung mit optionalen Inputs
        Parameter: data_manager - DataManager-Instanz
        Parameter: lod_level - Exaktes LOD-Ceiling für alle Fetches, analog zu den anderen
            5 Generatoren (GenerationThread.run() übergibt dort überall self.lod_level).
            Vorher holte diese Methode für jede Dependency unabhängig das jeweils
            "beste global verfügbare LOD" (get_terrain_data("complete")/get_water_data()/
            get_biome_data() ohne LOD-Argument) - dadurch konnte Settlement z.B. eine
            Terrain-Heightmap von LOD 5 mit einer Biome-Map von LOD 2 mischen, statt wie
            die anderen Generatoren konsistent auf einem LOD zu bleiben. None (Default)
            erhält das alte "bestes verfügbares LOD"-Verhalten für Aufrufer außerhalb
            des Orchestrators (z.B. Legacy-Skripte).
        Returns: dict - Alle Input-Daten (required + optional mit Fallbacks)
        """
        if not data_manager:
            raise Exception("DataManager required for Settlement generation")

        dependencies = {}

        # Kombiniert (Geology-Tektonik + Water-Erosion/-Sedimentation) statt der
        # unbearbeiteten Terrain-Rohausgabe - dieselbe Datenbasis, die auch die
        # anderen 5 Generatoren als heightmap_combined bekommen (siehe
        # DataLODManager.get_terrain_data_combined()). Vorher las diese Methode
        # TerrainData.heightmap direkt aus dem "complete"-Objekt - das ist die rohe,
        # unbearbeitete Heightmap.
        heightmap = data_manager.get_terrain_data_combined("heightmap", lod_level)
        slopemap = data_manager.get_terrain_data_lod("slopemap", lod_level)

        if heightmap is None:
            raise Exception("Required dependency 'heightmap' not available in DataManager")
        if slopemap is None:
            raise Exception("Required dependency 'slopemap' not available in DataManager")

        dependencies['heightmap'] = heightmap
        dependencies['slopemap'] = slopemap

        # Water-Daten, mit Fallback auf water_biomes_map falls water_map (noch) nicht existiert
        water_map = data_manager.get_water_data_lod('water_map', lod_level)
        if water_map is None:
            water_map = data_manager.get_water_data_lod('water_biomes_map', lod_level)
        if water_map is None:
            raise Exception("Required dependency 'water_map' not available in DataManager")
        dependencies['water_map'] = water_map

        # OPTIONAL Dependency - erstelle Fallback-Wert wenn nicht vorhanden
        # biome_map: Fallback basierend auf Höhe (für MoveCost-Berechnung)
        biome_map = data_manager.get_biome_data_lod('biome_map', lod_level)
        if biome_map is None:
            self.logger.warning("biome_map not available, creating height-based fallback")
            biome_map = self._create_fallback_biome_map(heightmap)
        dependencies['biome_map'] = biome_map

        self.logger.debug(f"Dependencies loaded - heightmap: {heightmap.shape}, water_map: {dependencies['water_map'].shape}")

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
        Aufgabe: Kernlogik der Settlement-Generierung mit allen 7 Hauptphasen und Status-Tracking
        Parameter: lod, dependencies, parameters
        Returns: SettlementData-Objekt mit allen Settlement-Outputs und Status-Tracking
        """
        heightmap = dependencies['heightmap']
        slopemap = dependencies['slopemap']
        water_map = dependencies['water_map']
        biome_map = dependencies['biome_map']

        self.set_active_parameters(parameters)
        self._ensure_data_lod_manager()

        # LOD-Größe bestimmen
        target_size = self._get_lod_size(lod, heightmap.shape[0])

        # Alle Arrays auf Zielgröße interpolieren falls nötig
        if heightmap.shape[0] != target_size:
            heightmap = self._interpolate_array(heightmap, target_size)
            slopemap = self._interpolate_array(slopemap, target_size)
            water_map = self._interpolate_array(water_map, target_size)
            biome_map = self._interpolate_array(biome_map, target_size)

        try:
            # Standalone-Convenience-Pfad (Legacy-Kompatibilität + Tests): dependencies
            # kommen hier als direktes dict, nicht aus dem DataLODManager - für die
            # _calc_*-Methoden (die jetzt IMMER aus dem Storage lesen) gespiegelt,
            # analog zu Geology/Water/Biome. Erwartet lod als int (siehe
            # _get_prepared_settlement_inputs()/set_calculator_output() - der
            # String-LOD-Pfad ist nur noch für ungenutzten Legacy-Code relevant).
            self.data_lod_manager.set_calculator_output("terrain.redistribution", lod, {"heightmap": heightmap})
            self.data_lod_manager.set_calculator_output("terrain.slope", lod, {"slopemap": slopemap})
            self.data_lod_manager.set_calculator_output(
                "water.flow_network", lod, {"water_biomes_map": water_map})
            self.data_lod_manager.set_calculator_output(
                "biome.integrate_layers", lod, {"biome_map": biome_map})

            # Läuft über die einzeln aufrufbaren _calc_*-Methoden (siehe
            # gui/OldManagers/calculator_graph.py - Settlement-Calculator-Knoten
            # #28-#34 aus docs/generation_pipeline_dependencies.md). Die echte
            # GUI-Pipeline (GenerationOrchestrator) ruft dieselben Methoden ab jetzt
            # einzeln über den globalen CalculatorDispatcher auf (Tracker #16
            # LOD-Lockstep-Umbau) - nur #34 (plot_nodes) braucht biome_map, die
            # anderen 6 Phasen können unabhängig von Biome starten.
            for calculator_id in (
                "settlement.suitability", "settlement.settlements", "settlement.city_boundary",
                "settlement.city_blocks", "settlement.landscape_voronoi", "settlement.pathfinding",
                "settlement.outer_roads", "settlement.roadsites", "settlement.civ_influence",
                "settlement.landmarks", "settlement.landmark_roads", "settlement.plot_nodes",
            ):
                getattr(self, "_calc_" + calculator_id.split(".", 1)[1])(calculator_id, lod)

            settlement_data = self.assemble_settlement_data(lod, parameters)

            self.logger.debug(f"Settlement generation complete - LOD: {lod}, size: {target_size}")
            self.logger.debug(
                f"Generated: {len(settlement_data.settlement_list)} settlements, "
                f"{len(settlement_data.roads)} roads, {len(settlement_data.plots)} plots")

            return settlement_data

        except Exception as e:
            self.logger.error(f"Settlement generation failed: {e}")
            raise

    def assemble_settlement_data(self, lod_level: int, parameters) -> SettlementData:
        """
        Funktionsweise: Baut das finale SettlementData-Objekt aus den einzeln
        gespeicherten Calculator-Outputs zusammen
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald alle 7
            Settlement-Calculator-Knoten ein LOD abgeschlossen haben (siehe
            Task 18 im LOD-Lockstep-Umbau)
        """
        combined_suitability_map = self.data_lod_manager.get_calculator_output(
            "settlement.suitability", "combined_suitability_map", lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        city_mask = self.data_lod_manager.get_calculator_output("settlement.city_boundary", "city_mask", lod_level)
        voronoi_cell_map = self.data_lod_manager.get_calculator_output(
            "settlement.landscape_voronoi", "voronoi_cell_map", lod_level)
        street_mask = self.data_lod_manager.get_calculator_output("settlement.city_blocks", "street_mask", lod_level)
        house_parcel_map = self.data_lod_manager.get_calculator_output(
            "settlement.city_blocks", "house_parcel_map", lod_level)
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        landmark_roads = self.data_lod_manager.get_calculator_output(
            "settlement.landmark_roads", "landmark_roads", lod_level)
        outer_roads = self.data_lod_manager.get_calculator_output(
            "settlement.outer_roads", "outer_roads", lod_level)
        roadsite_list = self.data_lod_manager.get_calculator_output(
            "settlement.roadsites", "roadsite_list", lod_level)
        civ_map = self.data_lod_manager.get_calculator_output("settlement.civ_influence", "civ_map", lod_level)
        landmark_list = self.data_lod_manager.get_calculator_output(
            "settlement.landmarks", "landmark_list", lod_level)
        plot_nodes = self.data_lod_manager.get_calculator_output(
            "settlement.plot_nodes", "plot_nodes", lod_level)
        plots = self.data_lod_manager.get_calculator_output("settlement.plot_nodes", "plots", lod_level)
        plot_map = self.data_lod_manager.get_calculator_output("settlement.plot_nodes", "plot_map", lod_level)
        plot_edges = self.data_lod_manager.get_calculator_output("settlement.plot_nodes", "plot_edges", lod_level)

        if combined_suitability_map is None or settlement_list is None or civ_map is None:
            raise ValueError(f"assemble_settlement_data: fehlende Calculator-Outputs für LOD {lod_level}")

        settlement_data = SettlementData()
        settlement_data.lod_level = lod_level
        settlement_data.actual_size = combined_suitability_map.shape[0]
        settlement_data.parameters = parameters.copy()
        settlement_data.combined_suitability_map = combined_suitability_map
        settlement_data.settlement_list = settlement_list
        settlement_data.city_mask = city_mask
        settlement_data.voronoi_cell_map = voronoi_cell_map
        settlement_data.street_mask = street_mask
        settlement_data.house_parcel_map = house_parcel_map
        settlement_data.roads = roads if roads is not None else []
        settlement_data.landmark_roads = landmark_roads if landmark_roads is not None else []
        settlement_data.outer_roads = outer_roads if outer_roads is not None else []
        settlement_data.roadsite_list = roadsite_list if roadsite_list is not None else []
        settlement_data.civ_map = civ_map
        settlement_data.landmark_list = landmark_list if landmark_list is not None else []
        settlement_data.plot_nodes = plot_nodes if plot_nodes is not None else []
        settlement_data.plots = plots if plots is not None else []
        settlement_data.plot_map = plot_map
        settlement_data.plot_edges = plot_edges if plot_edges is not None else {}

        settlement_data.terrain_suitability_valid = True
        settlement_data.settlements_valid = True
        settlement_data.road_network_valid = roads is not None
        settlement_data.roadsites_valid = roadsite_list is not None
        settlement_data.civilization_mapping_valid = True
        settlement_data.landmarks_valid = landmark_list is not None
        settlement_data.plots_valid = plot_nodes is not None

        return settlement_data

    def _get_prepared_settlement_inputs(self, lod_level: int) -> Dict[str, Any]:
        """
        Holt alle Settlement-Dependencies (Terrain/Water-Outputs) für dieses LOD.
        water_map wird bewusst direkt aus water.flow_network's water_biomes_map
        gelesen (Calculator-Graph-Ebene), nicht aus der zusammengesetzten
        Domain-Ebene (DataLODManager.get_water_data_lod('water_map')) - funktional
        äquivalent für die reine Wasser-Präsenz-Prüfung in
        TerrainSuitabilityAnalyzer.calculate_water_proximity() (prüft nur
        `water_map > 0`), aber verfügbar sobald DIESER EINE Water-Knoten fertig
        ist, ohne auf die vollständige Water-Generator-Assemblierung zu warten.
        """
        heightmap = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        slopemap = self.data_lod_manager.get_calculator_output("terrain.slope", "slopemap", lod_level)
        water_map = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "water_biomes_map", lod_level)

        missing = [name for name, value in (
            ("heightmap", heightmap), ("slopemap", slopemap), ("water_map", water_map)
        ) if value is None]
        if missing:
            raise ValueError(f"Settlement: fehlende Dependencies für LOD {lod_level}: {', '.join(missing)}")

        return {"heightmap": heightmap, "slopemap": slopemap, "water_map": water_map}

    def _calc_suitability(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.suitability' (#28)"""
        self._update_progress("Terrain Analysis", 5, "Analyzing terrain suitability for settlements...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        suitability_map = self.calculate_terrain_suitability(
            inputs["heightmap"], inputs["slopemap"], inputs["water_map"], lod_level)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"combined_suitability_map": suitability_map})

    def _calc_settlements(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.settlements' (#29)"""
        self._update_progress("Settlement Placement", 15, "Placing settlements based on suitability...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        suitability_map = self.data_lod_manager.get_calculator_output(
            "settlement.suitability", "combined_suitability_map", lod_level)
        if suitability_map is None:
            raise ValueError(f"settlement.settlements: combined_suitability_map für LOD {lod_level} nicht verfügbar")

        settlement_list = self.calculate_settlements(suitability_map, inputs["heightmap"], lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"settlement_list": settlement_list})

    def _calc_city_boundary(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.city_boundary' (NEU) - terrain-cost-gewichtete
        Stadtgrenze je Settlement, Grundlage fuer die Trennung Stadt-Innen (spaeteres
        Block-System) vs. Landschaft (LandscapeVoronoiSystem)."""
        self._update_progress("City Boundary", 20, "Computing city boundaries...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        if settlement_list is None:
            raise ValueError(f"settlement.city_boundary: settlement_list für LOD {lod_level} nicht verfügbar")

        analyzer = CityBoundaryAnalyzer(self.terrain_factor_villages, self.city_reach_factor, self.shader_manager)
        city_mask, city_cost_map = analyzer.compute_city_boundaries(
            inputs["heightmap"], inputs["slopemap"], settlement_list, self._update_progress)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"city_mask": city_mask, "city_cost_map": city_cost_map})

    def _calc_city_blocks(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.city_blocks' (NEU) - innerstädtisches
        Straßenraster + Hausparzellen je Settlement (siehe CityBlockSystem),
        strikt auf die jeweilige Stadtgrenze (#35, city_mask) begrenzt.
        Parzellen-IDs werden über alle Settlements hinweg fortlaufend eindeutig
        gemacht (globaler next_parcel_id-Zähler), damit house_parcel_map
        kartenweit als eine einzige ID-Ebene genutzt werden kann."""
        self._update_progress("City Blocks", 22, "Generating street grid and house parcels...")
        city_mask = self.data_lod_manager.get_calculator_output("settlement.city_boundary", "city_mask", lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        if city_mask is None or settlement_list is None:
            raise ValueError(f"settlement.city_blocks: fehlende Inputs für LOD {lod_level}")

        height, width = city_mask.shape
        street_mask = np.zeros((height, width), dtype=bool)
        house_parcel_map = np.full((height, width), -1, dtype=np.int32)

        block_system = CityBlockSystem(self.house_spacing)
        next_parcel_id = 0
        for settlement in settlement_list:
            if settlement.location_type != 'settlement':
                continue
            footprint_mask = city_mask == settlement.location_id
            if not np.any(footprint_mask):
                continue

            rng = random.Random(self.map_seed + lod_level + settlement.location_id)
            result = block_system.build_for_settlement(footprint_mask, rng)

            street_mask |= result["street_mask"]
            local_parcels = result["house_parcel_map"]
            has_parcel = local_parcels >= 0
            if has_parcel.any():
                house_parcel_map[has_parcel] = local_parcels[has_parcel] + next_parcel_id
                next_parcel_id += int(local_parcels[has_parcel].max()) + 1

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
            "street_mask": street_mask, "house_parcel_map": house_parcel_map,
        })

    def _calc_landscape_voronoi(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.landscape_voronoi' (NEU) - grobe Feld-Voronoi-
        Zellen ausserhalb der Stadtgrenzen, warm-gestartet aus der vorigen LOD-Stufe
        (siehe LandscapeVoronoiSystem-Docstring fürs "Gummiband"-Verhalten)."""
        self._update_progress("Landscape Voronoi", 30, "Generating landscape plot seeds...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        city_mask = self.data_lod_manager.get_calculator_output("settlement.city_boundary", "city_mask", lod_level)
        if city_mask is None:
            raise ValueError(f"settlement.landscape_voronoi: city_mask für LOD {lod_level} nicht verfügbar")

        height, width = inputs["heightmap"].shape
        previous_seeds_raw = self.data_lod_manager.get_calculator_output(
            "settlement.landscape_voronoi", "voronoi_seed_positions", lod_level - 1)
        previous_seeds = None
        if previous_seeds_raw:
            # Seeds werden relativ (0..1) gespeichert (siehe unten) - dadurch unabhängig
            # von der absoluten Pixelgröße der vorigen LOD-Stufe auf die aktuelle Karte skalierbar.
            previous_seeds = [(rx * (width - 1), ry * (height - 1)) for rx, ry in previous_seeds_raw]

        voronoi = LandscapeVoronoiSystem(
            self.terrain_factor_villages, self.voronoi_base_spacing, self.voronoi_relax_iterations,
            self.shader_manager)
        rng = random.Random(self.map_seed + lod_level)
        seeds = voronoi.generate_seeds(
            inputs["heightmap"], inputs["slopemap"], city_mask, self.plotnodes,
            previous_seeds=previous_seeds, rng=rng)
        seeds = voronoi.relax(seeds, inputs["heightmap"], inputs["slopemap"], city_mask, self._update_progress)
        cell_map = voronoi.assign_cells(inputs["heightmap"], inputs["slopemap"], seeds, city_mask, self._update_progress)

        # Relativ (0..1) speichern, damit der Warm-Start beim nächsten (größeren) LOD
        # unabhängig von der absoluten Pixelgröße dieser Stufe skaliert werden kann.
        relative_seeds = [(x / max(1, width - 1), y / max(1, height - 1)) for x, y in seeds]

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
            "voronoi_seed_positions": relative_seeds,
            "voronoi_cell_map": cell_map,
        })

    def _calc_pathfinding(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.pathfinding' (#30)"""
        self._update_progress("Road Building", 25, "Creating road networks between settlements...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        voronoi_cell_map = self.data_lod_manager.get_calculator_output(
            "settlement.landscape_voronoi", "voronoi_cell_map", lod_level)
        if settlement_list is None:
            raise ValueError(f"settlement.pathfinding: settlement_list für LOD {lod_level} nicht verfügbar")

        roads = self.calculate_road_network(settlement_list, inputs["slopemap"], lod_level, voronoi_cell_map)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"roads": roads})

    def _calc_roadsites(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.roadsites' (#31)"""
        self._update_progress("Roadsite Placement", 40, "Placing roadsites along roads...")
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        if roads is None:
            raise ValueError(f"settlement.roadsites: roads für LOD {lod_level} nicht verfügbar")

        roadsite_list = self.calculate_roadsites(roads, lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"roadsite_list": roadsite_list})

    def _calc_civ_influence(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.civ_influence' (#32)"""
        self._update_progress("Civilization Mapping", 50, "Creating civilization influence map...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        roadsite_list = self.data_lod_manager.get_calculator_output(
            "settlement.roadsites", "roadsite_list", lod_level)
        if settlement_list is None or roads is None or roadsite_list is None:
            raise ValueError(f"settlement.civ_influence: fehlende Inputs für LOD {lod_level}")

        civ_map = self.calculate_civilization_mapping(
            inputs["heightmap"], inputs["slopemap"], settlement_list, roads, roadsite_list)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"civ_map": civ_map})

    def _calc_landmarks(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.landmarks' (#33)"""
        self._update_progress("Landmark Placement", 65, "Placing landmarks in wilderness areas...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        civ_map = self.data_lod_manager.get_calculator_output("settlement.civ_influence", "civ_map", lod_level)
        if civ_map is None:
            raise ValueError(f"settlement.landmarks: civ_map für LOD {lod_level} nicht verfügbar")

        landmark_list = self.calculate_landmarks(civ_map, inputs["heightmap"], inputs["slopemap"], lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"landmark_list": landmark_list})

    def _calc_landmark_roads(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.landmark_roads' (NEU) - deterministische
        Dijkstra-Anbindung jedes Landmarks an den nächstgelegenen Punkt des
        Hauptstraßennetzes (Nutzer-Vorgabe: kein Zufallsmechanismus in Phase 1,
        das dekorative Zusatz-Wegenetz kommt erst in Phase 2)."""
        self._update_progress("Landmark Roads", 68, "Connecting landmarks to road network...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        landmark_list = self.data_lod_manager.get_calculator_output(
            "settlement.landmarks", "landmark_list", lod_level)
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        if landmark_list is None or roads is None:
            raise ValueError(f"settlement.landmark_roads: fehlende Inputs für LOD {lod_level}")

        landmark_roads = self.calculate_landmark_roads(landmark_list, roads, inputs["slopemap"], lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"landmark_roads": landmark_roads})

    def _calc_outer_roads(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.outer_roads' (NEU) - 2-3 Außenverbindungen
        von Siedlungen zur Kartengrenze an plausiblen Positionen (nicht
        Bergspitze/Meer, siehe calculate_outer_connections())."""
        self._update_progress("Outer Roads", 27, "Connecting settlements to map border...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        suitability_map = self.data_lod_manager.get_calculator_output(
            "settlement.suitability", "combined_suitability_map", lod_level)
        if settlement_list is None or suitability_map is None:
            raise ValueError(f"settlement.outer_roads: fehlende Inputs für LOD {lod_level}")

        outer_roads = self.calculate_outer_connections(
            settlement_list, suitability_map, inputs["water_map"], inputs["slopemap"], lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"outer_roads": outer_roads})

    def _calc_plot_nodes(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'settlement.plot_nodes' (#34) - einzige Settlement-Phase,
        die biome_map braucht (siehe docs/generation_pipeline_dependencies.md).
        """
        self._update_progress("Plot Generation", 75, "Generating plot system...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        civ_map = self.data_lod_manager.get_calculator_output("settlement.civ_influence", "civ_map", lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        biome_map = self.data_lod_manager.get_calculator_output(
            "biome.integrate_layers", "biome_map", lod_level)
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        if civ_map is None or settlement_list is None or biome_map is None:
            raise ValueError(f"settlement.plot_nodes: fehlende Inputs für LOD {lod_level}")

        height, width = inputs["heightmap"].shape
        previous_positions_relative = self.data_lod_manager.get_calculator_output(
            "settlement.plot_nodes", "plot_node_positions", lod_level - 1)
        previous_node_positions = None
        if previous_positions_relative:
            # Relativ (0..1) gespeichert (siehe unten), damit der Warm-Start
            # unabhängig von der absoluten Pixelgröße der vorigen LOD-Stufe auf
            # die aktuelle Karte skaliert werden kann (analog zu
            # LandscapeVoronoiSystem._calc_landscape_voronoi()).
            previous_node_positions = [(rx * (width - 1), ry * (height - 1)) for rx, ry in previous_positions_relative]

        plot_nodes, plots, plot_edges = self.calculate_plots(
            civ_map, settlement_list, inputs["heightmap"], biome_map, lod_level, roads,
            previous_node_positions=previous_node_positions)
        plot_map = self._create_plot_map(inputs["heightmap"].shape, plots)

        relative_positions = [
            (x / max(1, width - 1), y / max(1, height - 1)) for x, y in
            (node.node_location for node in plot_nodes)
        ]
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level,
            {"plot_nodes": plot_nodes, "plots": plots, "plot_map": plot_map, "plot_edges": plot_edges,
             "plot_node_positions": relative_positions})

    def calculate_terrain_suitability(self, heightmap, slopemap, water_map, lod):
        """
        Funktionsweise: Berechnet Terrain-Suitability für optimale Settlement-Platzierung
        Aufgabe: Kombiniert Slope-, Water-Proximity- und Elevation-Fitness zu finaler Suitability-Map
        Parameter: heightmap, slopemap, water_map, lod - Alle Terrain-Daten und LOD-Level
        Returns: numpy.ndarray - Kombinierte Suitability-Map für Settlement-Platzierung
        """
        analyzer = TerrainSuitabilityAnalyzer(self.terrain_factor_villages, heightmap.shape[0])
        combined_suitability = analyzer.create_combined_suitability(heightmap, slopemap, water_map, self._update_progress)
        return combined_suitability

    def calculate_settlements(self, suitability_map, heightmap, lod):
        """
        Funktionsweise: Platziert Settlements basierend auf Terrain-Suitability mit LOD-abhängiger Anzahl
        Aufgabe: Erstellt Settlement-Liste mit optimaler Positionierung und Mindestabständen
        Parameter: suitability_map, heightmap, lod - Suitability-Daten und LOD-Level
        Returns: List[Location] - Alle platzierten Settlements
        """
        # LOD-abhängige Settlement-Anzahl
        lod_factors = {"LOD64": 0.5, "LOD128": 0.8, "LOD256": 1.0, "FINAL": 1.0}
        adjusted_count = max(1, int(self.settlements * lod_factors.get(lod, 1.0)))

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
                progress = 15 + (len(settlements) * 10) // adjusted_count
                self._update_progress("Settlement Placement", progress, f"Placed {len(settlements)}/{adjusted_count} settlements")

        return settlements

    def calculate_road_network(self, settlements, slopemap, lod, voronoi_cell_map=None):
        """
        Funktionsweise: Erstellt Straßennetzwerk zwischen Settlements mit LOD-optimiertem Pathfinding
        Aufgabe: Findet optimale Straßenverbindungen mit Spline-Interpolation
        Parameter: settlements, slopemap, lod - Settlement-Liste, Slope-Daten und LOD-Level
        Parameter: voronoi_cell_map - optionale Landschafts-Voronoi-Zellzuordnung
            (settlement.landscape_voronoi, #36) - wenn vorhanden, bevorzugen die
            Straßen den Verlauf entlang der Zellgrenzen (Nutzer-Vorgabe); None
            faellt auf reines Slope-Cost-Pathfinding zurueck (Legacy-Verhalten).
        Returns: List[List[Tuple]] - Alle Road-Pfade als Wegpunkt-Listen
        """
        if len(settlements) < 2:
            return []

        edge_distance_map = None
        if voronoi_cell_map is not None:
            edge_distance_map = _voronoi_edge_distance_map(voronoi_cell_map)

        pathfinder = PathfindingSystem(
            self.road_slope_to_distance_ratio, slopemap.shape[0],
            edge_distance_map=edge_distance_map, edge_bias=self.road_voronoi_edge_bias,
            edge_bias_scale=self.voronoi_base_spacing)
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
                    progress = 25 + (road_count * 15) // total_roads
                    self._update_progress("Road Building", progress, f"Built {road_count}/{total_roads} roads")

        return roads

    def calculate_roadsites(self, roads, lod):
        """
        Funktionsweise: Platziert Roadsites entlang von Straßen zwischen 30-70% der Weglänge
        Aufgabe: Erstellt Roadsite-Liste mit verschiedenen Typen (Tavern, Trading Post, etc.)
        Parameter: roads, lod - Road-Pfade und LOD-Level
        Returns: List[Location] - Alle platzierten Roadsites
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

    def calculate_civilization_mapping(self, heightmap, slopemap, settlements, roads, roadsites):
        """
        Funktionsweise: Erstellt civ_map durch radialen Decay von Settlement/Road/Roadsite-Punkten
        Aufgabe: Berechnet Zivilisations-Einfluss mit Slope-abhängigem Decay und Wilderness-Definition
        Parameter: heightmap, slopemap, settlements, roads, roadsites - Alle Zivilisations-Quellen
        Returns: numpy.ndarray - Civilization-Influence-Map mit Wilderness-Bereichen
        """
        height, width = heightmap.shape
        civ_map = np.zeros((height, width), dtype=np.float32)

        # effective_radius als Bruchteil der Kartendiagonale statt der winzigen
        # settlement.radius (4-6px unabhaengig von map_size) als Decay-Laengenskala -
        # Nutzer-Vorgabe: "ich will das wirklich ein großer Radius um die Stadt
        # beeinflusst wird [...] alles andere ist ja Wilderness und die macht etwa
        # die Hälfte der Karte aus". Skaliert automatisch mit jeder LOD-Stufe.
        map_diagonal = np.sqrt(height ** 2 + width ** 2)
        effective_radius = map_diagonal * self.civ_influence_range
        influence_mapper = CivilizationInfluenceMapper(self.civ_influence_decay, effective_radius)

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

    def calculate_landmarks(self, civ_map, heightmap, slopemap, lod):
        """
        Funktionsweise: Platziert Landmarks in Wilderness-Bereichen mit niedrigem civ_map-Wert
        Aufgabe: Erstellt Landmark-Liste (Castle, Monastery, etc.) mit Elevation/Slope-Constraints
        Parameter: civ_map, heightmap, slopemap, lod - Civilization-Map, Terrain-Daten und LOD-Level
        Returns: List[Location] - Alle platzierten Landmarks
        """
        landmarks = []

        # LOD-abhängige Landmark-Anzahl
        lod_factors = {"LOD64": 0.5, "LOD128": 0.8, "LOD256": 1.0, "FINAL": 1.0}
        adjusted_count = max(0, int(self.landmarks * lod_factors.get(lod, 1.0)))

        if adjusted_count == 0:
            return landmarks

        height, width = civ_map.shape

        # Vektorisiert statt Pixel-fuer-Pixel-Python-Schleife: die alten Helper
        # _check_elevation_suitability()/_check_slope_suitability() riefen pro
        # Pixel (H*W-mal!) erneut np.min()/np.max() auf die komplette heightmap
        # auf - zweitgroesster Performance-Fund neben calculate_water_proximity()
        # (siehe docs/backlog.md Ticket #4 Performance-Hinweis).
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height
        if height_range == 0:
            elevation_ok = np.ones((height, width), dtype=bool)
        else:
            norm_height = (heightmap - min_height) / height_range
            elevation_ok = norm_height < 0.7  # Landmarks nur in unteren 70% der Höhen

        slope_magnitude = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2)
        slope_ok = slope_magnitude < 0.5  # Landmarks nur bei moderaten Slopes

        wilderness_ok = civ_map < self.landmark_wilderness
        valid_mask = wilderness_ok & elevation_ok & slope_ok
        valid_ys, valid_xs = np.nonzero(valid_mask)
        valid_positions = list(zip(valid_xs.tolist(), valid_ys.tolist()))

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
            self._update_progress("Landmark Placement", 70, f"Placed {len(landmarks)} landmarks")

        return landmarks

    def calculate_landmark_roads(self, landmarks, roads, slopemap, lod):
        """
        Funktionsweise: Verbindet jedes Landmark deterministisch per A*-Pathfinding
        mit dem nächstgelegenen Wegpunkt des bestehenden Hauptstraßennetzes.
        Aufgabe: Landmark-Anbindung ohne Zufallsmechanismus (Nutzer-Vorgabe -
        das dekorative Zusatz-Wegenetz ist bewusst auf Phase 2 verschoben).
        Parameter: landmarks, roads, slopemap, lod - Landmark-Liste, bestehende
            Road-Pfade, Slope-Daten und LOD-Level
        Returns: List[List[Tuple]] - Ein Pfad pro Landmark zum Straßennetz
        """
        if not landmarks or not roads:
            return []

        road_points = [pt for road in roads for pt in road]
        if not road_points:
            return []

        pathfinder = PathfindingSystem(self.road_slope_to_distance_ratio, slopemap.shape[0])
        landmark_roads = []

        for landmark in landmarks:
            distances = [(landmark.x - px) ** 2 + (landmark.y - py) ** 2 for px, py in road_points]
            nearest_idx = int(np.argmin(distances))
            target = road_points[nearest_idx]

            path = pathfinder.find_least_resistance_path(
                slopemap, (landmark.x, landmark.y), target, self._update_progress)
            smoothed_path = pathfinder.apply_spline_smoothing(
                path, smoothing_factor=3, progress_callback=self._update_progress)
            landmark_roads.append(smoothed_path)

        return landmark_roads

    def calculate_outer_connections(self, settlements, suitability_map, water_map, slopemap, lod, count=None):
        """
        Funktionsweise: Verbindet Siedlungen mit 2-3 Punkten am Kartenrand an
        plausiblen Positionen (nicht Bergspitze/Meer, siehe Nutzer-Vorgabe).
        Aufgabe: Randpunkt-Auswahl über Suitability + Wasser-Ausschluss, über
        den Kartenumfang verteilt (Mindestabstand), dann A*-Pathfinding von der
        jeweils nächstgelegenen Siedlung.
        Parameter: settlements, suitability_map, water_map, slopemap, lod, count
            - Settlement-Liste, Terrain-Eignung, Wasser-Maske, Slope-Daten,
              LOD-Level und optionale feste Anzahl Außenverbindungen (Default:
              2-3, abhängig von der Settlement-Anzahl)
        Returns: List[List[Tuple]] - Ein Pfad pro Außenverbindung
        """
        settlements_only = [s for s in settlements if s.location_type == 'settlement']
        if not settlements_only:
            return []

        height, width = suitability_map.shape
        connection_count = count if count is not None else min(3, max(2, len(settlements_only)))

        border_candidates = []
        for x in range(width):
            border_candidates.append((x, 0))
            border_candidates.append((x, height - 1))
        for y in range(height):
            border_candidates.append((0, y))
            border_candidates.append((width - 1, y))

        # Wasser-Punkte (Meer/See am Kartenrand) ausschließen, Rest nach Suitability sortieren
        scored = [
            (suitability_map[y, x], x, y) for (x, y) in border_candidates if water_map[y, x] <= 0
        ]
        if not scored:
            return []
        scored.sort(key=lambda item: item[0], reverse=True)

        # Ausgewählte Randpunkte über den Kartenumfang verteilen (Mindestabstand),
        # damit nicht alle Verbindungen in derselben Ecke landen
        chosen = []
        map_perimeter = 2 * (width + height)
        min_spacing = map_perimeter / (connection_count * 2)
        for _, x, y in scored:
            if len(chosen) >= connection_count:
                break
            if all((x - cx) ** 2 + (y - cy) ** 2 >= min_spacing ** 2 for cx, cy in chosen):
                chosen.append((x, y))

        pathfinder = PathfindingSystem(self.road_slope_to_distance_ratio, slopemap.shape[0])
        outer_roads = []

        for (bx, by) in chosen:
            nearest_settlement = min(
                settlements_only, key=lambda s: (s.x - bx) ** 2 + (s.y - by) ** 2)

            path = pathfinder.find_least_resistance_path(
                slopemap, (nearest_settlement.x, nearest_settlement.y), (bx, by), self._update_progress)
            smoothed_path = pathfinder.apply_spline_smoothing(
                path, smoothing_factor=3, progress_callback=self._update_progress)
            outer_roads.append(smoothed_path)

        return outer_roads

    def calculate_plots(self, civ_map, settlements, heightmap, biome_map, lod, roads=None,
                        previous_node_positions=None):
        """
        Funktionsweise: Generiert PlotNode-System mit Delaunay-Triangulation,
        Plot-Fusion und Familien-/Verkehrssimulation über den Edge-Graph
        Aufgabe: Erstellt Grundstücks-System basierend auf akkumuliertem Civ-Wert mit LOD-optimierter Dichte

        Zwei-Pass-Ablauf für die traffic-getriebene "Gummiband"-Anziehung
        (Nutzer-Vorgabe): Pass 1 baut Graph+Traffic auf der (warmgestarteten)
        Ausgangsgeometrie auf, nur um zu wissen, welche Kanten stark genutzt
        sind. apply_traffic_attraction() verschiebt die Nodes daraufhin leicht
        zueinander (Gegenkraft: derselbe civ-gewichtete Mindestabstand wie bei
        der Generierung). Pass 2 baut Graph+Traffic auf der ANGEPASSTEN
        Geometrie neu auf - das ist der tatsächlich zurückgegebene/angezeigte
        Zustand. Traffic wird dabei nie über Aufrufe hinweg akkumuliert (siehe
        simulate_plot_traffic()); nur die minimal verschobenen Positionen
        wandern über previous_node_positions in die nächste LOD-Stufe weiter,
        wo sich stark genutzte Wege dadurch schrittweise weiter begradigen/
        verkürzen können - das Kräftegleichgewicht ist bewusst nur grob
        kalibriert und soll später feinjustiert werden.
        Parameter: civ_map, settlements, heightmap, biome_map, lod, roads -
            Alle Plot-Daten, LOD-Level und bestehendes Straßennetz (für
            Inter-City-Traffic, siehe simulate_plot_traffic())
        Parameter: previous_node_positions - optionale (x,y)-Positionen der
            vorigen LOD-Stufe, bereits auf die aktuelle Kartengröße skaliert
            (siehe _calc_plot_nodes()) - Warm-Start für generate_plot_nodes()
        Returns: Tuple[List[PlotNode], List[Plot], Dict[int, PlotEdge]] -
            PlotNode-Liste, finale Plot-Liste und das Edge-Registry mit
            simuliertem Traffic/Klassifikation (beides aus Pass 2)
        """
        plot_system = PlotNodeSystem(self.plotsize, heightmap.shape[0])

        # PlotNodes generieren (civ-wert-abhängige Abstoßung + Warm-Start, siehe generate_plot_nodes())
        nodes = plot_system.generate_plot_nodes(
            civ_map, self.plotnodes, settlements,
            base_spacing=self.plot_base_spacing, civ_spacing_factor=self.plot_civ_spacing_factor,
            previous_nodes=previous_node_positions,
            rng=random.Random(self.map_seed + int(heightmap.shape[0])),
            progress_callback=self._update_progress)

        # Pass 1: Graph + Traffic nur zur Ermittlung stark genutzter Kanten
        nodes, edge_registry = plot_system.create_delaunay_triangulation(
            nodes, heightmap, biome_map, height_cost_factor=self.plot_height_cost_factor,
            progress_callback=self._update_progress)
        edge_registry = plot_system.simulate_plot_traffic(
            nodes, edge_registry, settlements, roads or [],
            path_traffic_threshold=self.plot_path_traffic_threshold,
            road_traffic_threshold=self.plot_road_traffic_threshold,
            intercity_traffic=self.plot_intercity_traffic,
            progress_callback=self._update_progress)

        # Gummiband-Anziehung entlang stark genutzter Kanten anwenden
        nodes = plot_system.apply_traffic_attraction(
            nodes, edge_registry, civ_map, self.plot_base_spacing, self.plot_civ_spacing_factor,
            self.plot_traffic_attraction, progress_callback=self._update_progress)

        # Pass 2: Graph + Traffic auf der angepassten Geometrie neu aufbauen (finaler Zustand)
        nodes, edge_registry = plot_system.create_delaunay_triangulation(
            nodes, heightmap, biome_map, height_cost_factor=self.plot_height_cost_factor,
            progress_callback=self._update_progress)
        edge_registry = plot_system.simulate_plot_traffic(
            nodes, edge_registry, settlements, roads or [],
            path_traffic_threshold=self.plot_path_traffic_threshold,
            road_traffic_threshold=self.plot_road_traffic_threshold,
            intercity_traffic=self.plot_intercity_traffic,
            progress_callback=self._update_progress)

        # Plots aus Nodes erstellen
        plots = plot_system.merge_to_plots(nodes, civ_map, self._update_progress)

        return nodes, plots, edge_registry

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
            data_manager.set_settlement_data("combined_suitability_map", result.combined_suitability_map, parameters)

            # Komplettes SettlementData-Objekt auch speichern
            data_manager.set_settlement_data("settlement_data_complete", result, parameters)

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
        # Legacy-String-LOD ("LOD64" etc.) für Rückwärtskompatibilität
        if isinstance(lod, str):
            if lod == "FINAL":
                return original_size
            lod_sizes = {"LOD64": 64, "LOD128": 128, "LOD256": 256, "LOD512": 512, "LOD1024": 1024}
            return lod_sizes.get(lod, original_size)

        # Modernes numerisches LOD-System: data_lod_manager liefert die Arrays
        # bereits in der zur angeforderten LOD-Stufe passenden Pixel-Auflösung.
        return original_size

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
        return self.calculate_road_network(settlements, slopemap, "LOD64")

    def place_landmarks(self, civ_map, landmarks_count, landmark_wilderness, heightmap, slopemap):
        """
        Funktionsweise: Legacy-Methode für Landmark-Platzierung
        """
        self.landmarks = landmarks_count
        self.landmark_wilderness = landmark_wilderness
        return self.calculate_landmarks(civ_map, heightmap, slopemap, "LOD64")

    def place_roadsites(self, roads, roadsites_count):
        """
        Funktionsweise: Legacy-Methode für Roadsite-Platzierung
        """
        self.roadsites = roadsites_count
        return self.calculate_roadsites(roads, "LOD64")

    def generate_plots(self, civ_map, plotnodes_count, plotsize, settlements, heightmap):
        """
        Funktionsweise: Legacy-Methode für Plot-Generierung
        """
        self.plotnodes = plotnodes_count
        self.plotsize = plotsize
        biome_map = self._create_fallback_biome_map(heightmap)
        return self.calculate_plots(civ_map, settlements, heightmap, biome_map, "LOD64")

    def create_civilization_map(self, heightmap, slopemap, settlements, roads, landmarks, roadsites, civ_influence_decay):
        """
        Funktionsweise: Legacy-Methode für Civilization-Map-Erstellung
        """
        self.civ_influence_decay = civ_influence_decay
        return self.calculate_civilization_mapping(heightmap, slopemap, settlements, roads, roadsites)

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
        # Auf Defaults aufsetzen statt eines fest kodierten Dicts, damit neu
        # hinzugekommene Parameter (z.B. city_reach_factor, plot_base_spacing)
        # diese Legacy-Methode nicht mit KeyError in set_active_parameters()
        # brechen - nur die von der alten Signatur tatsächlich übergebenen
        # Werte überschreiben die Defaults.
        parameters = self._load_default_parameters()
        parameters.update({
            'settlements': settlements,
            'landmarks': landmarks,
            'roadsites': roadsites,
            'plotnodes': plotnodes,
            'civ_influence_decay': civ_influence_decay,
            'terrain_factor_villages': terrain_factor_villages,
            'road_slope_to_distance_ratio': road_slope_to_distance_ratio,
            'landmark_wilderness': landmark_wilderness,
            'plotsize': plotsize
        })

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
        raise ValueError(
            f"Unsupported export format: {format_type}")  # Nachbarn prüfen (8-Connectivity mit LOD-Resolution)
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

    def __init__(self, civ_influence_decay=1.0, effective_radius=50.0):
        """
        Funktionsweise: Initialisiert Civilization-Influence-Mapper mit Decay-Parameter
        Aufgabe: Setup der Zivilisations-Einfluss-Berechnung
        Parameter: civ_influence_decay (float) - Stärke des Einfluss-Abfalls mit Distanz
        Parameter: effective_radius (float) - Decay-Längenskala in Pixeln für den
            Einfluss-Abfall (siehe SettlementGenerator.calculate_civilization_mapping():
            map_diagonal * civ_influence_range) - ersetzt die vorher genutzte
            settlement.radius (4-6px, unabhängig von map_size) als Skala, damit die
            Reichweite tatsächlich mit der Kartengröße mitwächst.
        """
        self.decay_factor = civ_influence_decay
        self.effective_radius = max(1e-3, effective_radius)

    def apply_settlement_influence(self, civ_map, settlements, slopemap, progress_callback=None):
        """
        Funktionsweise: Wendet Settlement-Einfluss auf civ_map an mit radialem Decay
        Aufgabe: Berechnet Zivilisations-Einfluss von Städten und Dörfern
        Parameter: civ_map, settlements, slopemap, progress_callback - Civ-Map, Settlement-Liste, Slope-Daten und Progress
        Returns: numpy.ndarray - Aktualisierte civ_map

        Vektorisiert über die gesamte Karte pro Settlement (statt einer lokalen
        Python-Doppelschleife über ein enges Fenster um settlement.radius) - mit
        effective_radius jetzt oft ein nennenswerter Bruchteil der Kartengröße
        (siehe __init__-Docstring) wäre das enge alte Fenster ohnehin zu klein
        gewesen, und ein entsprechend vergrößertes Fenster mit Pixel-für-Pixel
        Python-Aufrufen wäre für große effective_radius sehr langsam geworden.
        """
        if progress_callback:
            progress_callback("Civilization Mapping", 55, f"Applying influence for {len(settlements)} settlements...")

        height, width = civ_map.shape
        slope_magnitude = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2)

        # Normalisiert auf die tatsächliche Terrain-Skala dieser Heightmap statt
        # eines festen Faktors auf rohe Höhenmeter-pro-Pixel: slope_magnitude
        # liegt bei üblicher Amplitude/Redistribution oft im Bereich 10-90+,
        # wodurch der alte "min(3.0, 1+slope*2)"-Modifier auf >95% der Pixel
        # sofort am Deckel saturierte - flache und gebirgige Gegenden wurden
        # dadurch kaum unterschieden (Nutzer-Beobachtung: "Civ Influence
        # verbreitet sich über flache Gebiete besser als Berge hoch und
        # runter" - das griff bisher praktisch nirgends spürbar). Das
        # 75.-Perzentil dieser konkreten Karte als Referenz macht den Modifier
        # automatisch adaptiv zu jeder Amplitude/map_size statt eines
        # hartcodierten "typischen" Werts.
        typical_slope = np.percentile(slope_magnitude, 75)
        normalized_slope = slope_magnitude / max(float(typical_slope), 1e-6)
        slope_modifier = np.minimum(5.0, 1.0 + normalized_slope * 2.0)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

        for i, settlement in enumerate(settlements):
            center_x, center_y = settlement.x, settlement.y
            radius = settlement.radius
            base_influence = settlement.civ_influence

            distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
            decay_distance = np.maximum(0.0, distance - radius) * self.decay_factor * slope_modifier
            influence = base_influence * np.exp(-decay_distance / self.effective_radius)
            influence = np.where(distance <= radius, 1.0, influence)  # Innerhalb Stadt: maximaler Einfluss
            civ_map = np.maximum(civ_map, influence)

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

    def __init__(self, plotsize=1.0, map_size=64):
        """
        Funktionsweise: Initialisiert PlotNode-System mit Plot-Größen-Parameter und LOD
        Aufgabe: Setup der Grundstücks-Generierung mit LOD-Anpassung
        Parameter: plotsize (float), map_size (int) - Akkumulierter Civ-Wert für Plot-Größe und tatsächliche Pixel-Größe
        """
        self.plotsize_threshold = plotsize
        self.map_size = map_size
        self.next_node_id = 0
        self.next_plot_id = 0

        # Größenabhängige PlotNode-Dichte
        if map_size <= 64:
            self.density_factor = 0.3  # 30% der gewünschten Nodes
        elif map_size <= 128:
            self.density_factor = 0.6  # 60% der gewünschten Nodes
        elif map_size <= 256:
            self.density_factor = 0.9  # 90% der gewünschten Nodes
        else:
            self.density_factor = 1.0  # 100% der gewünschten Nodes

    def generate_plot_nodes(self, civ_map, plotnodes_count, settlements, base_spacing=10.0,
                             civ_spacing_factor=3.0, previous_nodes=None, rng=None, progress_callback=None):
        """
        Funktionsweise: Generiert PlotNodes mit civ-wert-abhängiger Abstoßung
        außerhalb von Städten und Wilderness
        Aufgabe: Hoher Civ-Wert (näher an der Stadt) -> kleinerer Mindestabstand
        -> dichtere Nodes -> kleinere Plots; niedriger Civ-Wert (Richtung
        Wilderness) -> größerer Mindestabstand -> größere Plots (Nutzer-
        Vorgabe). Ersetzt die vorherige civ-blinde Gleichverteilung
        (_sample_uniform_distribution()) samt der wirkungslosen festen
        3px-Abstoßung in optimize_node_positions() - Mindestabstand wird jetzt
        direkt beim Sampling durchgesetzt statt nachträglich per Post-Hoc-
        Optimierung. Spatial-Hash-Grid für die Nachbarschaftsprüfung (gleiches
        Muster wie LandscapeVoronoiSystem.generate_seeds() - siehe dortigen
        Performance-Kommentar zu O(n)-Scans bei vielen Nodes).
        Parameter: civ_map, plotnodes_count, settlements - Civ-Map, Ziel-Anzahl, Settlement-Liste
        Parameter: base_spacing, civ_spacing_factor - Mindestabstand bei civ=0
            bzw. Stärke der civ-abhängigen Kompression (min_spacing =
            base_spacing / (1 + civ_spacing_factor * civ_value))
        Parameter: previous_nodes - optionale Liste von (x,y)-Positionen der
            vorigen LOD-Stufe (bereits auf die aktuelle Kartengröße skaliert,
            siehe SettlementGenerator._calc_plot_nodes()) - Warm-Start, damit
            die traffic-getriebene Anziehung (apply_traffic_attraction()) über
            LOD-Stufen hinweg wirken kann statt bei jeder Stufe komplett neu zu
            würfeln (analog zu LandscapeVoronoiSystem.generate_seeds())
        Returns: List[PlotNode] - Generierte PlotNodes
        """
        if progress_callback:
            progress_callback("Plot Generation", 80, "Generating plot nodes...")

        rng = rng or random
        height, width = civ_map.shape
        adjusted_count = int(plotnodes_count * self.density_factor)

        # Vektorisiert statt Pixel-für-Pixel-Python-Schleife: Wilderness (< 0.2)
        # und Stadt-Kern (>= 1.0) ausschließen, dazu Mindestabstand zu allen Settlements.
        valid_mask = (civ_map >= 0.2) & (civ_map < 1.0)
        if settlements:
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            for settlement in settlements:
                too_close = (xx - settlement.x) ** 2 + (yy - settlement.y) ** 2 < (settlement.radius * 1.2) ** 2
                valid_mask &= ~too_close

        cell_size = max(1e-3, base_spacing)
        grid: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

        def cell_of(x, y):
            return (int(x // cell_size), int(y // cell_size))

        def is_too_close(x, y, min_spacing):
            cx, cy = cell_of(x, y)
            span = int(min_spacing // cell_size) + 1
            for gx in range(cx - span, cx + span + 1):
                for gy in range(cy - span, cy + span + 1):
                    for (sx, sy) in grid.get((gx, gy), ()):
                        if (sx - x) ** 2 + (sy - y) ** 2 < min_spacing ** 2:
                            return True
            return False

        nodes = []

        # Marktplatz-Node je Settlement (Nutzer-Vorgabe: "eine Stadt hat je
        # einen Marktplatz, der zugleich ein Node ist") - unconditional
        # eingefügt (ignoriert valid_mask, da Stadt-Kerne civ_map>=1.0 sonst
        # ausgeschlossen wären), damit diese Nodes ganz normal an Delaunay-
        # Triangulation und Traffic-Graph teilnehmen statt eines separaten
        # virtuellen Anker-Knotens (siehe simulate_plot_traffic()). In den
        # Abstands-Grid eingetragen, damit reguläre Nodes trotzdem Abstand
        # dazu halten.
        for settlement in settlements:
            if settlement.location_type != 'settlement':
                continue
            node = PlotNode(
                node_id=self.next_node_id, node_location=(float(settlement.x), float(settlement.y)),
                connector_id=[], connector_distance=[], connector_elevation=[],
                connector_movecost=[], connector_edge_id=[], settlement_id=settlement.location_id
            )
            nodes.append(node)
            grid.setdefault(cell_of(settlement.x, settlement.y), []).append(
                (float(settlement.x), float(settlement.y)))
            self.next_node_id += 1

        # Marktplätze zählen nicht gegen das reguläre Node-Budget (plotnodes_count
        # ist "wie viele Feld-Nodes zusätzlich zu den Marktplätzen").
        adjusted_count += len(nodes)

        if previous_nodes:
            for (x, y) in previous_nodes:
                if len(nodes) >= adjusted_count:
                    break
                ix, iy = int(round(x)), int(round(y))
                if not (0 <= ix < width and 0 <= iy < height) or not valid_mask[iy, ix]:
                    continue
                civ_value = float(civ_map[iy, ix])
                min_spacing = base_spacing / (1.0 + civ_spacing_factor * civ_value)
                if is_too_close(x, y, min_spacing):
                    continue
                node = PlotNode(
                    node_id=self.next_node_id, node_location=(float(x), float(y)),
                    connector_id=[], connector_distance=[], connector_elevation=[],
                    connector_movecost=[], connector_edge_id=[]
                )
                nodes.append(node)
                grid.setdefault(cell_of(x, y), []).append((float(x), float(y)))
                self.next_node_id += 1

        max_attempts = max(200, adjusted_count * 30)
        attempts = 0
        while len(nodes) < adjusted_count and attempts < max_attempts:
            attempts += 1
            x = rng.uniform(0, width - 1)
            y = rng.uniform(0, height - 1)
            ix, iy = int(x), int(y)
            if not valid_mask[iy, ix]:
                continue

            civ_value = float(civ_map[iy, ix])
            min_spacing = base_spacing / (1.0 + civ_spacing_factor * civ_value)
            if is_too_close(x, y, min_spacing):
                continue

            node = PlotNode(
                node_id=self.next_node_id,
                node_location=(float(x), float(y)),
                connector_id=[], connector_distance=[], connector_elevation=[],
                connector_movecost=[], connector_edge_id=[]
            )
            nodes.append(node)
            grid.setdefault(cell_of(x, y), []).append((x, y))
            self.next_node_id += 1

        if progress_callback:
            progress_callback("Plot Generation", 85, f"Generated {len(nodes)} plot nodes")

        return nodes

    def create_delaunay_triangulation(self, nodes, heightmap, biome_map=None,
                                       height_cost_factor=2.0, progress_callback=None):
        """
        Funktionsweise: Erstellt Delaunay-Triangulation zwischen PlotNodes mit MoveCost-Berechnung
        Aufgabe: Verbindet PlotNodes über Delaunay-Dreiecke für Grundstücks-Bildung UND
        baut parallel das adressierbare PlotEdge-Registry auf (Nutzer-Vorgabe:
        "Plotnode 234 und 260 teilen sich eine Kante, diese ist dann Kante 839") -
        Grundlage für build_edge_graph_and_simulate_traffic().
        Parameter: nodes, heightmap, biome_map, height_cost_factor, progress_callback -
            PlotNode-Liste, Höhendaten, Biom-Map, Gewichtung der mittleren
            Pfad-Steigung in den Kantenkosten, und Progress
        Returns: Tuple[List[PlotNode], Dict[int, PlotEdge]] - Nodes mit
            aktualisierten Verbindungen und das Edge-Registry (edge_id -> PlotEdge)
        """
        if progress_callback:
            progress_callback("Plot Generation", 87, "Creating Delaunay triangulation...")

        edge_registry: Dict[int, PlotEdge] = {}

        if len(nodes) < 3:
            return nodes, edge_registry

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
                node.connector_edge_id = []

            edge_lookup: Dict[Tuple[int, int], int] = {}  # (min_id,max_id) -> edge_id

            def _connect(node_a, node_b):
                distance = self._calculate_distance(node_a.node_location, node_b.node_location)
                elevation_diff = self._calculate_elevation_difference(
                    node_a.node_location, node_b.node_location, heightmap
                )
                movecost = self._calculate_biome_movecost(
                    node_a.node_location, node_b.node_location, biome_map, distance, elevation_diff
                )

                key = (min(node_a.node_id, node_b.node_id), max(node_a.node_id, node_b.node_id))
                if key not in edge_lookup:
                    # Wegintegral statt Endpunkt-Differenz: eine gerade Linie über
                    # Hügel-und-Tal hätte sonst netto ~0 Höhendifferenz, obwohl
                    # tatsächlich Auf- und Abstieg zurückgelegt werden müsste.
                    height_cost = self._cumulative_height_cost(
                        node_a.node_location, node_b.node_location, heightmap)
                    avg_slope = height_cost / max(distance, 1e-6)
                    movement_cost = distance * (1.0 + height_cost_factor * avg_slope)

                    edge_id = len(edge_registry)
                    edge_registry[edge_id] = PlotEdge(
                        edge_id=edge_id, node_a=key[0], node_b=key[1],
                        length=distance, height_cost=height_cost, movement_cost=movement_cost)
                    edge_lookup[key] = edge_id
                edge_id = edge_lookup[key]

                node_a.connector_id.append(node_b.node_id)
                node_a.connector_distance.append(distance)
                node_a.connector_elevation.append(elevation_diff)
                node_a.connector_movecost.append(movecost)
                node_a.connector_edge_id.append(edge_id)

            for simplex in tri.simplices:
                # Jedes Dreieck verbindet 3 Nodes
                for i in range(3):
                    for j in range(i + 1, 3):
                        node_a = nodes[simplex[i]]
                        node_b = nodes[simplex[j]]

                        if node_b.node_id not in node_a.connector_id:
                            _connect(node_a, node_b)
                        if node_a.node_id not in node_b.connector_id:
                            _connect(node_b, node_a)

            return nodes, edge_registry

        except Exception as e:
            # Fallback: keine Verbindungen
            if progress_callback:
                progress_callback("Plot Generation", 87, f"Delaunay triangulation failed: {e}")
            return nodes, edge_registry

    def merge_to_plots(self, nodes, civ_map, progress_callback=None):
        """
        Funktionsweise: Fusioniert PlotNodes zu Plots basierend auf akkumuliertem Civ-Wert
        Aufgabe: Erstellt Grundstücke durch Node-Gruppierung nach Civ-Wert-Schwellwert
        Parameter: nodes, civ_map, progress_callback - PlotNode-Liste, Civ-Map und Progress
        Returns: List[Plot] - Generierte Plots
        """
        if progress_callback:
            progress_callback("Plot Generation", 90, "Merging nodes to plots...")

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
            progress_callback("Plot Generation", 92, f"Created {len(plots)} plots from {len(nodes)} nodes")

        return plots

    @staticmethod
    def _rank_distance_weights(n):
        """
        Funktionsweise: Chancendegressions-Formel für die Stadtwahl nach
        Rang-Distanz (Nutzer-Vorgabe): die i-nächste Siedlung bekommt
        P(i) = 0.5^i (i=1 nächste, i=2 übernächste, ...), die entfernteste
        Siedlung (Rang n) bekommt denselben Wert wie Rang n-1, damit die
        Summe exakt 1.0 ergibt (2 Städte: 50/50, 3: 50/25/25, 4: 50/25/12.5/12.5, ...).
        Parameter: n - Anzahl erreichbarer Siedlungen
        Returns: List[float] - Gewichte, aufsteigend nach Rang (Index 0 = nächste)
        """
        if n <= 0:
            return []
        if n == 1:
            return [1.0]
        weights = [0.5 ** i for i in range(1, n)]
        weights.append(0.5 ** (n - 1))
        return weights

    def simulate_plot_traffic(self, nodes, edge_registry, settlements, road_network,
                               path_traffic_threshold=25, road_traffic_threshold=75,
                               intercity_traffic=30, progress_callback=None):
        """
        Funktionsweise: Simuliert Wege-/Straßen-Entstehung über den PlotEdge-Graph
        Aufgabe: Jede PlotNode (außer den Marktplätzen selbst) entspricht einer
        Familie, die ihren Verkehr nach Rang-Distanz gewichtet auf ALLE
        erreichbaren Marktplätze verteilt statt ausschließlich auf den
        nächstgelegenen (Nutzer-Vorgabe, Chancendegressions-Formel - siehe
        _rank_distance_weights()): die nächste Siedlung bekommt 50% der
        "Familien-Masse" dieser PlotNode, die übernächste 25%, usw. Jede
        durchquerte Kante auf dem jeweiligen Weg wird um das entsprechende
        Gewicht erhöht (Kosten siehe PlotEdge.movement_cost). Zusätzlich
        laufen intercity_traffic Personen pro Richtung zwischen den durch das
        bestehende Straßennetz (road_network, settlement.pathfinding)
        verbundenen Siedlungspaaren über denselben Graphen. Kanten werden ab
        path_traffic_threshold zu "path", ab road_traffic_threshold zu "road".

        Jede Siedlung hat einen eigenen Marktplatz-PlotNode (siehe
        generate_plot_nodes()), der ganz normal im selben Graphen liegt - kein
        separater virtueller Anker-Knoten mehr nötig.

        Bleibt trotz Wegintegral-Kosten pro Kante günstig: der Graph hat nur
        so viele Knoten wie es PlotNodes gibt (typischerweise <= wenige
        Tausend, siehe SETTLEMENT.PLOTNODES) statt Pixel-Grid-Größe - Dijkstra
        darauf ist Größenordnungen billiger als die Pixel-Floods anderswo im
        Settlement-Rework (_terrain_cost_voronoi()).

        Parameter: nodes, edge_registry - PlotNode-Liste und Dict[edge_id, PlotEdge]
            (siehe create_delaunay_triangulation())
        Parameter: settlements, road_network - Settlement-Liste und bestehende
            Road-Pfade (settlement.pathfinding) zur Bestimmung "benachbarter"
            Siedlungspaare für den Inter-City-Traffic
        Returns: Dict[int, PlotEdge] - edge_registry mit aktualisiertem
            traffic/classification (traffic ist jetzt fraktional/float, da
            eine PlotNode ihre "Familien-Masse" auf mehrere Ziele aufteilt)
        """
        if progress_callback:
            progress_callback("Plot Generation", 93, "Simulating family/trade traffic...")

        if not nodes or not edge_registry:
            return edge_registry

        node_index = {node.node_id: i for i, node in enumerate(nodes)}
        num_nodes = len(nodes)

        marketplace_index_by_settlement_id = {
            n.settlement_id: node_index[n.node_id] for n in nodes if n.settlement_id >= 0
        }
        settlements_with_marketplace = [
            s for s in settlements if s.location_type == 'settlement'
            and s.location_id in marketplace_index_by_settlement_id
        ]
        if not settlements_with_marketplace:
            return edge_registry

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        rows, cols, costs = [], [], []
        for edge in edge_registry.values():
            i, j = node_index[edge.node_a], node_index[edge.node_b]
            rows += [i, j]
            cols += [j, i]
            costs += [edge.movement_cost, edge.movement_cost]
        graph = csr_matrix((costs, (rows, cols)), shape=(num_nodes, num_nodes))

        edge_lookup = {(e.node_a, e.node_b): e.edge_id for e in edge_registry.values()}

        def _add_traffic_along_path(predecessors, source_idx, target_idx, amount):
            current = target_idx
            while current != source_idx and current >= 0:
                prev = predecessors[current]
                if prev < 0:
                    break
                a, b = (current, prev) if current < prev else (prev, current)
                key = (nodes[a].node_id, nodes[b].node_id)
                edge_id = edge_lookup.get(key)
                if edge_id is not None:
                    edge_registry[edge_id].traffic += amount
                current = prev

        # 1) Rang-Distanz-gewichtete Verkehrsverteilung (siehe _rank_distance_weights()):
        # Distanzen von JEDEM Marktplatz zu JEDER PlotNode auf einmal berechnen
        # (kein "min_only" mehr - wir brauchen die volle Rangliste, nicht nur
        # den naechsten), dann pro PlotNode nach Distanz sortieren und gewichtet
        # verteilen.
        marketplace_indices = [marketplace_index_by_settlement_id[s.location_id]
                                for s in settlements_with_marketplace]
        distances, predecessors = dijkstra(graph, indices=marketplace_indices, return_predecessors=True)

        for i in range(num_nodes):
            if nodes[i].settlement_id >= 0:
                continue  # Marktplätze selbst erzeugen keinen Familienverkehr

            node_distances = distances[:, i]
            reachable = [k for k in range(len(marketplace_indices)) if np.isfinite(node_distances[k])]
            if not reachable:
                continue
            reachable.sort(key=lambda k: node_distances[k])
            weights = self._rank_distance_weights(len(reachable))
            for rank, k in enumerate(reachable):
                _add_traffic_along_path(predecessors[k], marketplace_indices[k], i, weights[rank])

        # 2) Inter-City-Traffic: entlang des bestehenden Straßennetzes
        # verbundene Siedlungspaare tauschen intercity_traffic Personen pro
        # Richtung aus (ueber den Plot-Graphen geroutet, direkt zwischen den
        # beiden Marktplatz-Nodes).
        connected_pairs = self._infer_connected_settlement_pairs(settlements_with_marketplace, road_network)
        for settlement_a, settlement_b in connected_pairs:
            src = marketplace_index_by_settlement_id[settlement_a.location_id]
            dst = marketplace_index_by_settlement_id[settlement_b.location_id]
            dist_single, pred_single = dijkstra(graph, indices=[src], return_predecessors=True)
            if np.isfinite(dist_single[0, dst]):
                _add_traffic_along_path(pred_single[0], src, dst, intercity_traffic)

        # 3) Klassifikation nach Traffic-Schwellwerten
        for edge in edge_registry.values():
            if edge.traffic >= road_traffic_threshold:
                edge.classification = "road"
            elif edge.traffic >= path_traffic_threshold:
                edge.classification = "path"
            else:
                edge.classification = "none"

        if progress_callback:
            path_count = sum(1 for e in edge_registry.values() if e.classification != "none")
            progress_callback("Plot Generation", 95, f"Traffic simulation done ({path_count} paths/roads)")

        return edge_registry

    def _infer_connected_settlement_pairs(self, settlements, road_network):
        """
        Funktionsweise: Bestimmt "benachbarte" Siedlungspaare für den Inter-
        City-Traffic anhand des bereits bestehenden Straßennetzes
        (settlement.pathfinding, MST-Verbindungen) statt einer eigenen
        Nachbarschafts-Definition
        Parameter: settlements, road_network - Settlement-Liste und Road-Pfade
        Returns: List[Tuple[Location, Location]] - Verbundene Settlement-Paare
        """
        pairs = []
        if not road_network:
            return pairs

        for road in road_network:
            if len(road) < 2:
                continue
            start_point, end_point = road[0], road[-1]
            start_settlement = min(
                settlements, key=lambda s: (s.x - start_point[0]) ** 2 + (s.y - start_point[1]) ** 2)
            end_settlement = min(
                settlements, key=lambda s: (s.x - end_point[0]) ** 2 + (s.y - end_point[1]) ** 2)
            if start_settlement is not end_settlement:
                pairs.append((start_settlement, end_settlement))

        return pairs

    def apply_traffic_attraction(self, nodes, edge_registry, civ_map, base_spacing,
                                  civ_spacing_factor, attraction_strength, progress_callback=None):
        """
        Funktionsweise: "Gummiband"-Anziehungskraft zwischen Nodes, deren
        verbindende Kante Traffic trägt - stark genutzte Wege werden dadurch
        über die LOD-Iterationen hinweg schrittweise kürzer/gerader
        (Nutzer-Vorgabe: "verwendete Wege werden noch stärker verwendet").
        Gegenkraft ist derselbe civ-gewichtete Mindestabstand wie bei
        generate_plot_nodes() (Terrain/Civ-Dichte/Nähe zu anderen Plots) - eine
        Kante zieht ihre beiden Nodes nie näher zusammen, als dieser
        Mindestabstand erlaubt, das verhindert ein Kollabieren.

        Traffic selbst wird NICHT über Aufrufe hinweg akkumuliert (jeder
        Durchlauf berechnet frischen Traffic aus der aktuellen Geometrie, siehe
        simulate_plot_traffic()) - nur die daraus resultierende, leicht
        verschobene Node-Position wird über den Warm-Start
        (generate_plot_nodes(previous_nodes=...)) an die nächste LOD-Stufe
        weitergereicht. Das Gleichgewicht zwischen Anziehung und Mindestabstand
        ist bewusst nur grob kalibriert (attraction_strength) - Feintuning ist
        laut Nutzer-Vorgabe ein späterer Schritt.
        Parameter: nodes, edge_registry - aktuelle PlotNode-/PlotEdge-Listen
            (Traffic muss bereits über simulate_plot_traffic() gesetzt sein)
        Parameter: civ_map, base_spacing, civ_spacing_factor - dieselbe
            Mindestabstands-Formel wie generate_plot_nodes()
        Parameter: attraction_strength - Bewegung in Pixel pro Traffic-Punkt
        Returns: List[PlotNode] - Nodes mit angepassten Positionen (Marktplatz-
            Nodes bleiben immer an ihrer Settlement-Position fixiert)
        """
        if progress_callback:
            progress_callback("Plot Generation", 91, "Applying traffic attraction (rubber-band)...")

        if not edge_registry:
            return nodes

        height, width = civ_map.shape
        node_by_id = {n.node_id: n for n in nodes}
        positions = {nid: np.array(n.node_location, dtype=np.float64) for nid, n in node_by_id.items()}
        displacement = {nid: np.zeros(2) for nid in positions}

        for edge in edge_registry.values():
            if edge.traffic <= 0 or edge.node_a not in positions or edge.node_b not in positions:
                continue

            pos_a, pos_b = positions[edge.node_a], positions[edge.node_b]
            delta = pos_b - pos_a
            dist = float(np.hypot(delta[0], delta[1]))
            if dist < 1e-6:
                continue

            civ_a = self._get_node_civ_value(node_by_id[edge.node_a], civ_map)
            civ_b = self._get_node_civ_value(node_by_id[edge.node_b], civ_map)
            min_spacing = base_spacing / (1.0 + civ_spacing_factor * 0.5 * (civ_a + civ_b))

            # Nie ueber den Mindestabstand hinaus zusammenziehen (Gegenkraft)
            pull = max(0.0, min(dist - min_spacing, attraction_strength * edge.traffic))
            direction = delta / dist

            # Marktplatz-Nodes bleiben fixiert (sie SIND die Settlement-Position,
            # siehe generate_plot_nodes()) - trifft die Anziehung auf einen
            # Marktplatz, bewegt sich stattdessen das andere Ende komplett
            # (statt der sonst üblichen 50/50-Aufteilung).
            a_pinned = node_by_id[edge.node_a].settlement_id >= 0
            b_pinned = node_by_id[edge.node_b].settlement_id >= 0
            if a_pinned and b_pinned:
                continue
            elif a_pinned:
                displacement[edge.node_b] -= direction * pull
            elif b_pinned:
                displacement[edge.node_a] += direction * pull
            else:
                displacement[edge.node_a] += direction * pull * 0.5
                displacement[edge.node_b] -= direction * pull * 0.5

        for node in nodes:
            if node.settlement_id >= 0:
                continue  # Marktplatz bleibt an der Settlement-Position
            new_pos = positions[node.node_id] + displacement[node.node_id]
            new_pos[0] = np.clip(new_pos[0], 0, width - 1)
            new_pos[1] = np.clip(new_pos[1], 0, height - 1)
            node.node_location = (float(new_pos[0]), float(new_pos[1]))

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
            0: 1.0,  # ice_cap - schwer
            1: 0.8,  # tundra - moderat
            2: 0.6,  # taiga - leicht
            3: 0.4,  # grassland - sehr leicht
            4: 0.6,  # temperate_forest - moderat
            5: 0.5,  # mediterranean - leicht
            6: 1.2,  # desert - schwer
            7: 0.7,  # semi_arid - moderat
            8: 0.9,  # tropical_rainforest - schwer
            9: 0.7,  # tropical_seasonal - moderat
            10: 0.5,  # savanna - leicht
            11: 0.8,  # montane_forest - moderat
            12: 1.5,  # swamp - sehr schwer
            13: 0.6,  # coastal_dunes - moderat
            14: 1.1  # badlands - schwer
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

    def _cumulative_height_cost(self, pos1, pos2, heightmap):
        """
        Funktionsweise: Berechnet die kumulierte Höhenüberbrückung entlang der
        geraden Linie zwischen zwei Positionen (Wegintegral)
        Aufgabe: Liefert die tatsächlich zurückzulegende Auf-/Abstiegssumme für
        PlotEdge.height_cost - eine Linie über Hügel-und-Tal hat ~0 Netto-
        Höhendifferenz (siehe _calculate_elevation_difference()), muss aber
        trotzdem als teuer gelten (Nutzer-Vorgabe: "kumulierte
        Höhenüberbrückung, also über Wegintegral")
        Parameter: pos1, pos2, heightmap - Start/Ziel und Höhendaten
        Returns: float - Summe der absoluten Höhenänderungen entlang der Linie
        """
        height, width = heightmap.shape
        x1, y1 = pos1
        x2, y2 = pos2
        steps = max(1, int(round(np.hypot(x2 - x1, y2 - y1))))

        xs = np.clip(np.round(np.linspace(x1, x2, steps + 1)).astype(int), 0, width - 1)
        ys = np.clip(np.round(np.linspace(y1, y2, steps + 1)).astype(int), 0, height - 1)
        heights_along_path = heightmap[ys, xs]

        return float(np.sum(np.abs(np.diff(heights_along_path))))

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
