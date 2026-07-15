"""
# =============================================================================
# Legacy-Kompatibilität für bestehende Imports
# =============================================================================

# Alle ursprünglichen Klassen bleiben für Rückwärts-Kompatibilität verfügbar
# Sie delegieren intern an die neuen BaseGenerator-Implementierungen

# TerrainSuitabilityAnalyzer - bereits neu implementiert
# PathfindingSystem - bereits neu implementiert
# CivilizationInfluenceMapper - bereits neu implementiert
# PlotPhysicsSystem - ersetzt das fruehere PlotNodeSystem (Delaunay) sowie
#   LandscapeVoronoiSystem/CityBlockSystem, siehe [[project-settlement-plot-physics-rebuild]]
# Location, PlotNode, PlotCore, PlotEdge - bereits neu implementiert

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
    Plotnodes (PlotPhysicsSystem, siehe [[project-settlement-plot-physics-rebuild]] - ersetzt die
    frühere Delaunay-Triangulation): eine feste Anzahl Plotkerne wird via Mitchell's-Best-Candidate-
    Sampling civ-abhängig über die Karte verteilt (plotnodes-Parameter), zusätzlich ein dedizierter
    Kern pro Siedlung. Über diese Kerne (plus Stadt-/Wildnisgrenz-Sonderfälle) wird EIN einziges,
    auf das Kartenrechteck geklipptes Voronoi-Diagramm gelegt - die Voronoi-Kreuzungen sind die
    PlotNodes. Ein Feder-Masse-Physiksystem (Kräfte + Potentialfeld aus civ_map-Gradient/Stadt-
    Gravitation/Wildnis-Abstoßung) lässt das Netz bis zur Konvergenz (oder max. 100 Iterationen)
    entspannen; eine rang-distanz-gewichtete Verkehrssimulation über den Kanten-Graphen liefert
    Traffic-Werte, aus denen die Straßen-Tier-Klassifikation (Straße/Weg/Pfad, PlotEdge.classification)
    entsteht. PlotNode-Eigenschaften:
        node_id, node_location, connector_ids/connector_distances/connector_elevations/
        connector_move_costs (Listen), node_type (standard_plot_node/wilderness_core/city_core/
        wilderness_node/map_border_node/city_border_node), traffic_weight

Parameter Input (aus value_default.py SETTLEMENT):
- settlements, landmarks, roadsites, plotnodes: number of each type
- civ_influence_decay: Influence around Locationtypes decays of distance
- terrain_factor_villages: terrain influence on settlement suitability
- road_slope_to_distance_ratio: rather short roads or steep roads
- landmark_wilderness: wilderness area size by changing cutoff-threshold
- plot_base_spacing, plot_civ_spacing_factor, plot_height_cost_factor: plot-physics
  spacing/pressure tuning (see PlotPhysicsSystem)

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

PlotPhysicsSystem
    Funktionsweise: Baut ein einziges geclipptes globales Voronoi-Mesh über Plotkern-Seeds und lässt
    Feder-Masse-Physik + Potentialfeld bis zur Konvergenz laufen (Port aus tools/biome_lab/, siehe
    [[project-settlement-plot-physics-rebuild]]) - ersetzt PlotNodeSystem (Delaunay),
    LandscapeVoronoiSystem und CityBlockSystem vollständig.
    Aufgabe: Erstellt Grundstücks-/Wege-System nur am finalen LOD (keine Zwischen-LOD-Berechnung mehr)
    Methoden: generate(), build_plot_map()
"""

import numpy as np
from scipy.spatial import Delaunay, Voronoi, cKDTree
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, grey_closing, grey_opening
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import box, LineString, Point, Polygon, MultiPolygon
from skimage import measure
import heapq
import logging
from dataclasses import dataclass, field
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
    Funktionsweise: Ein Knoten im Wege-Netz - entweder eine Voronoi-Kreuzung
    zwischen Plotkernen (node_type="standard_plot_node") oder einer der drei
    Sondertypen, die nur TANGENTIAL entlang ihrer jeweiligen Kontur gleiten
    (bzw. seit dem Umbau auf ein einziges geklipptes Voronoi permanent
    unbeweglich sind, siehe PlotPhysicsSystem._physics_step):
      - "wilderness_node": auf der civ-Kontur (Wildnisgrenze)
      - "map_border_node": auf dem Kartenrand-Rechteck
      - "city_border_node": auf der Stadtgrenzkontur
    "wilderness_core"/"city_core" markieren dagegen normale Plotkerne (siehe
    PlotCore), die von der Feder-/Feldphysik ausgeschlossen sind.
    Aufgabe: Repräsentiert einzelne Nodes im Grundstücks-System. Feldnamen/
    -struktur 1:1 aus tools/biome_lab/models.py übernommen (Physics-Lab-
    Neuaufbau der Settlement-Plot-Generierung, siehe PlotPhysicsSystem) -
    ersetzt die frühere Delaunay-basierte PlotNode-Form (Singular-Feldnamen
    connector_id/connector_distance/... aus dem entfernten PlotNodeSystem).
    """
    node_id: int
    node_location: Tuple[float, float]
    connector_ids: List[int]
    connector_distances: List[float]
    connector_elevations: List[float]
    connector_move_costs: List[float]
    connector_edge_ids: List[int]
    settlement_id: int = -1  # >=0: dieser Node IST der Marktplatz von Settlement.location_id
    node_type: str = "standard_plot_node"
    neighbor_core_ids: List[int] = field(default_factory=list)
    neighbor_node_ids: List[int] = field(default_factory=list)
    traffic_weight: float = 4.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    # Regionen-Partitionierung (siehe PlotPhysicsSystem._region_id_at): welcher
    # zusammenhängenden civ-/Wildnis-Fläche dieser Node angehört. Rein
    # diagnostisch. region_id_secondary ist nur für echte Nahtstellen-Nodes
    # (node_type="wilderness_node") gesetzt, die zwei Regionen berühren.
    region_id: int = -1
    region_id_secondary: int = -1


@dataclass
class PlotCore:
    """
    Funktionsweise: Ein Plotkern (Voronoi-Seed-Punkt) - entweder ein regulärer
    Plot ("standard_plot_node"-Kern, nimmt an Feder-/Feldphysik teil),
    "wilderness_core" (tief in der Wildnis, physikfrei) oder "city_core"
    (Siedlungsposition selbst, physikfrei, garantiert eigene Voronoi-Zelle).
    Aufgabe: Leichtgewichtiges Gegenstück zu PlotNode für die Plotkerne
    selbst - ergänzt um core_type und dieselben Nachbarschafts-Listen wie
    PlotNode. 1:1 aus tools/biome_lab/models.py übernommen.
    """
    core_id: int
    location: Tuple[float, float]
    core_type: str = "standard_core"
    neighbor_core_ids: List[int] = field(default_factory=list)
    neighbor_node_ids: List[int] = field(default_factory=list)
    region_id: int = -1


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


# ==========================================================================
# PlotPhysicsSystem: ersetzt LandscapeVoronoiSystem + CityBlockSystem (und,
# weiter unten im File, das alte PlotNodeSystem) durch ein einziges,
# physikbasiertes Plot-Generierungssystem. Ported aus tools/biome_lab/
# (topology.py, physics.py, field.py, traffic.py, models.py) - siehe
# [[project-settlement-plot-physics-rebuild]]. tools/biome_lab/ bleibt als
# eigenstaendiges Sandbox-Tool fuer weitere Design-Iteration bestehen.
#
# Abweichungen vom interaktiven Lab-Original (Produktions-Anpassungen):
# - Kein Qt/QTimer/Mixin-Aufbau - eine Klasse mit generate() statt Live-Ticks.
# - Kein 11-Schritt-Klick-Modus (reine Lab-Debug-Funktion).
# - Kein Voronoi-Worker-Subprocess (war im Lab bereits ungenutzt).
# - Physik laeuft bis zur Konvergenz oder wird nach MAX_PHYSICS_ITERATIONS
#   eingefroren (Nutzer-Vorgabe) statt bis der Nutzer manuell pausiert.
# - Traffic-Kalibrierung: traffic_weight wird nach der finalen node_type-
#   Klassifikation fest zugewiesen (standard_plot_node=2.0, wilderness_core=
#   1.0, city_core=erhoeht gegenueber dem alten 4.0-Default), statt
#   zufaellig 3.0-5.0 fuer alle Nicht-Stadt-Knoten.
# - Alle Kraft-Schalter sind permanent aktiv (im Lab Debug-Checkboxen).
# - Optionaler progress_callback fuer Live-Fortschrittsanzeige waehrend der
#   Physik-Konvergenz (siehe Teil F des Rebuild-Plans).
# ==========================================================================


def _nearest_point_on_polyline(pos, polyline, closed=True):
    """Vektorisierte Projektion eines Punkts auf die naechstgelegene Stelle
    einer Polylinie (offen oder geschlossen). 1:1 aus tools/biome_lab/topology.py."""
    pos = np.asarray(pos, dtype=float)
    poly = np.asarray(polyline, dtype=float)
    n = len(poly)

    if n == 0:
        return pos
    if n == 1:
        return poly[0].copy()

    if closed:
        a = poly
        b = np.roll(poly, -1, axis=0)
    else:
        a = poly[:-1]
        b = poly[1:]

    ab = b - a
    ab_len_sq = np.einsum("ij,ij->i", ab, ab)
    safe_len_sq = np.where(ab_len_sq > 1e-12, ab_len_sq, 1.0)
    t = np.einsum("ij,ij->i", pos - a, ab) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    candidates = a + ab * t[:, None]
    degenerate = ab_len_sq <= 1e-12
    if np.any(degenerate):
        candidates[degenerate] = a[degenerate]

    dists_sq = np.einsum("ij,ij->i", candidates - pos, candidates - pos)
    best = int(np.argmin(dists_sq))
    return candidates[best]


def _nearest_point_on_segments(pos, a, b):
    """Wie _nearest_point_on_polyline, nimmt aber bereits in Start-/End-Punkte
    aufgeteilte Segment-Arrays (a, b) entgegen. 1:1 aus tools/biome_lab/topology.py."""
    pos = np.asarray(pos, dtype=float)
    ab = b - a
    ab_len_sq = np.einsum("ij,ij->i", ab, ab)
    safe_len_sq = np.where(ab_len_sq > 1e-12, ab_len_sq, 1.0)
    t = np.einsum("ij,ij->i", pos - a, ab) / safe_len_sq
    t = np.clip(t, 0.0, 1.0)
    candidates = a + ab * t[:, None]
    degenerate = ab_len_sq <= 1e-12
    if np.any(degenerate):
        candidates[degenerate] = a[degenerate]

    dists_sq = np.einsum("ij,ij->i", candidates - pos, candidates - pos)
    best = int(np.argmin(dists_sq))
    return candidates[best]


def _extract_intersection_points(geom):
    """Zieht alle (x,y)-Punkte aus einem beliebigen shapely-Intersections-
    Ergebnis. 1:1 aus tools/biome_lab/topology.py (siehe dortiger Docstring
    zur expliziten geom_type-Dispatch-Begruendung)."""
    gt = geom.geom_type
    if gt == "Point":
        return [(geom.x, geom.y)]
    if gt == "MultiPoint":
        return [(g.x, g.y) for g in geom.geoms]
    if gt in ("LineString", "LinearRing"):
        return list(geom.coords)
    if gt in ("MultiLineString", "GeometryCollection"):
        points = []
        for g in geom.geoms:
            points.extend(_extract_intersection_points(g))
        return points
    return []


def _polygon_area(vertices):
    """Flaeche eines Polygons ueber die Shoelace-Formel. 1:1 aus
    tools/biome_lab/topology.py."""
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    return float(abs(np.sum(x * y2 - x2 * y)) * 0.5)


class PlotPhysicsSystem:
    """
    Ersetzt LandscapeVoronoiSystem/CityBlockSystem/PlotNodeSystem: baut ein
    einziges geclipptes globales Voronoi-Mesh ueber Plotkern-Seeds auf und
    laesst Feder-Masse-Physik + Potentialfeld bis zur Konvergenz laufen.
    Liefert am Ende PlotNode/PlotEdge-Objekte (Straßen-Tier-Klassifikation
    inklusive) fuer SettlementData.plot_nodes/plot_edges/plot_map.
    """

    # ---- Konstanten (Werte 1:1 aus tools/biome_lab/app.py uebernommen -
    # dort ueber Log-Regler bei Multiplikator 1.0 erreicht, hier direkt als
    # Produktions-Default, da es in der Produktion keine Live-Regler gibt) ----
    TRAFFIC_RECOMPUTE_INTERVAL = 3
    WILDERNESS_CIV_THRESHOLD = 0.20
    WILDERNESS_MIN_AREA = 50
    CITY_MIN_AREA = 5.0
    SOFTENING = 5.0
    PHYSICS_TIME_STEP = 0.25
    MAX_SPRING_RESULTANT = 4.0
    MAX_DISPLACEMENT_PER_TICK = 2.0
    PLOT_CORE_EDGE_MARGIN = 5.0
    CIV_RESTLENGTH_STEEPNESS = 0.75
    SEED_RELAX_ITERATIONS = 8
    SEED_RELAX_STEP = 0.5
    SEED_RELAX_MARGIN = 3.0
    SEED_RELAX_NEIGHBOR_COUNT = 6

    # ---- NEU (existierte im Lab nicht - dort lief Physik bis der Nutzer
    # manuell pausierte, siehe Modul-Docstring) ----
    MAX_PHYSICS_ITERATIONS = 100  # Nutzer-Vorgabe
    # Konvergenz: gilt als "eingeschwungen", sobald die maximale Node-
    # Verschiebung ueber CONVERGENCE_STABLE_TICKS aufeinanderfolgende Ticks
    # unter CONVERGENCE_MAX_DISPLACEMENT bleibt. Platzhalter-Werte - Nutzer
    # will das Ergebnis erst live sehen, bevor final kalibriert wird.
    CONVERGENCE_MAX_DISPLACEMENT = 0.05
    CONVERGENCE_STABLE_TICKS = 5

    # ---- Traffic-Tier-Schwellen (aus tools/biome_lab/draw.py uebernommen,
    # dort Anzeige-Konzern - hier Teil der eigentlichen Generator-Logik,
    # siehe PlotEdge.classification) ----
    TIER_STRASSE_THRESHOLD = 170.0
    TIER_WEG_THRESHOLD = 90.0
    TIER_MIN_TRAFFIC = 20.0

    def __init__(self, map_size, plot_nodes_count=200, plot_base_spacing=60.0,
                 plot_civ_spacing_factor=8.0, plot_height_cost_factor=3.0,
                 shader_manager=None, progress_callback=None, map_seed=None,
                 live_state_callback=None):
        self.map_size = int(map_size)
        self.plot_nodes_count = int(plot_nodes_count)
        self.plot_base_spacing = float(plot_base_spacing)
        self.plot_civ_spacing_factor = float(plot_civ_spacing_factor)
        self.plot_height_cost_factor = float(plot_height_cost_factor)
        self.shader_manager = shader_manager  # aktuell ungenutzt (Teil G, GPU-Shader, ist nachgelagert)
        # Live-Snapshot-Callback (iteration, node positions/types) fuer die
        # Fortschrittsanzeige waehrend der Physik-Konvergenz - siehe
        # [[project-settlement-plot-physics-rebuild]] Teil F. Separat vom
        # generischen progress_callback (phase/percent/message), da die GUI
        # hier tatsaechliche Node-Positionen zum Nachzeichnen braucht.
        self.live_state_callback = live_state_callback
        self.progress_callback = progress_callback
        if map_seed is not None:
            random.seed(map_seed)

        # Kraft-Schalter: in der Produktion permanent aktiv (im Lab Debug-
        # Checkboxen, um einzelne Kraefte zu isolieren, siehe Modul-Docstring).
        self.enable_core_plotnode_spring = True
        self.enable_plotnode_plotnode_spring = True
        self.enable_pressure = True
        self.enable_plot_node_repulsion = True
        self.enable_field_cores = True
        self.enable_field_plotnodes = True
        self.enable_core_cell_containment = True
        self.enable_wilderness_containment = True

        # Physik-/Feld-Basiswerte (1:1 aus tools/biome_lab/app.py's
        # _BASE_*-Konstanten bei Multiplikator 1.0).
        self.core_plotnode_spring_stiffness = 1.2
        self.plotnode_plotnode_spring_stiffness = 1.0
        self.pressure_strength = 0.8
        self.core_mass = 1.0
        self.plot_node_mass = 1.0
        self.plot_node_repulsion_strength = 4.0
        self.plot_gravity_strength = 0.01
        self.plot_city_repulsion_strength = 0.5
        self.norm_potential_strength = 1.0
        self.damping = 0.80
        self.wilderness_push_stiffness = 1.5
        self.spring_traffic_shrink = 0.002
        self.spring_min_shrink_fraction = 0.70
        self.spring_shrink_ema_decay = 0.05

        # NEUE Traffic-Kalibrierung (Nutzer-Vorgabe, siehe Modul-Docstring):
        # feste Werte je finalem node_type statt zufaellig 3.0-5.0 fuer alle
        # Nicht-Stadt-Knoten. city_core-Wert liegt ueber dem bisherigen
        # 4.0-Default - Platzhalter, zur Kalibrierung nach Live-Test.
        self.traffic_weight_standard_plot = 2.0
        self.traffic_weight_wilderness = 1.0
        self.traffic_weight_city_core = 6.0
        self.plot_intercity_traffic = 30.0

        # Laufzeit-Zustand (1:1 Struktur wie tools/biome_lab/app.py's __init__)
        self.next_node_id = 0
        self.nodes = []
        self.plot_nodes = []
        self.vertex_to_plot_node = {}
        self._plot_node_to_vertex = {}
        self.core_registry = {}
        self.boundary_owner = {}
        self.wilderness_node_ids = set()
        self.map_border_node_ids = set()
        self._core_cell_plot_node_ids = {}
        self._core_type_by_id = {}
        self._plot_node_wilderness_cache = {}

        self._static_vertex_positions = None
        self._static_ridge_edges = []
        self._static_num_vertices = 0
        self._static_boundary_entries = []
        self._static_boundary_settlement = []
        self._static_node_entry = {}
        self._static_predecessors = None
        self._static_distances = None

        self.ridge_traffic_history = {}
        self.ridge_traffic_shrink_ema = {}
        self.path_cache = {}
        self.potential_field = None

        self.iteration = 0
        self.topology_ready = False

        # Wird in generate() gesetzt.
        self.heightmap = None
        self.slopemap = None
        self.civ_map = None
        self.city_mask = None
        self.settlements = []
        self.region_map = None
        self.num_civ_regions = 0
        self.num_wild_regions = 0
        self._wilderness_polygons = []
        self._wilderness_polygon_region_ids = []
        self._city_polygons = {}

    # ==================================================================
    # Oeffentliche Einstiegsmethode
    # ==================================================================
    def generate(self, heightmap, slopemap, civ_map, city_mask, settlements):
        """
        Baut das komplette Wege-/Plot-Netz auf und laesst die Physik bis zur
        Konvergenz (oder MAX_PHYSICS_ITERATIONS) laufen. Ersetzt den
        interaktiven 11-Schritt-Klick-Modus des Physics Lab durch einen
        einzigen durchgehenden, headless Aufruf.
        Parameter: heightmap/slopemap (H,W) bzw (H,W,2), civ_map (H,W) 0..1,
        city_mask (H,W) int (Settlement-ID pro Pixel, -1 ausserhalb jeder
        Stadt - siehe CityBoundaryAnalyzer), settlements (List[Location]).
        Returns: bool - True bei Erfolg (Voronoi-Aufbau kann bei zu wenigen
        Plotkernen fehlschlagen, siehe _gen_step_4_voronoi_clipped).
        """
        self.heightmap = heightmap
        self.slopemap = slopemap
        self.civ_map = civ_map
        self.city_mask = city_mask
        self.settlements = settlements

        self._gen_step_1_background()
        self._gen_step_2_plot_cores()
        self._gen_step_2b_relax_seed_points()
        self._gen_step_2c_place_city_cores()
        if not self._gen_step_4_voronoi_clipped():
            return False
        self._gen_step_city_boundary_distribute()
        self._gen_step_5_wilderness_snap()
        self._gen_step_6_wilderness_cores()
        self._assign_traffic_weights()
        self._gen_step_7_build_graph()
        self._gen_step_9_finalize()

        self._compute_potential_field()
        self._run_physics_to_convergence()
        return True

    def _report_progress(self, phase, percent, message):
        if self.progress_callback:
            try:
                self.progress_callback(phase, percent, message)
            except Exception:
                logging.debug("PlotPhysicsSystem progress_callback failed", exc_info=True)

    def _report_live_state(self):
        """Baut einen leichten, kopierten Snapshot des aktuellen (noch nicht
        konvergierten) Netzes und reicht ihn an live_state_callback weiter -
        siehe [[project-settlement-plot-physics-rebuild]] Teil F. Nur Plain-
        Python-Listen/Tupel (keine geteilten Referenzen auf self.nodes/
        self.plot_nodes selbst), damit der Snapshot sicher über eine Qt-
        Thread-Grenze wandern kann, während self.nodes im Hintergrund weiter
        mutiert wird."""
        if not self.live_state_callback:
            return
        try:
            snapshot = {
                "iteration": self.iteration,
                "map_size": self.map_size,
                "core_positions": [(n.node_location[0], n.node_location[1], n.node_type) for n in self.nodes],
                "plot_node_positions": [
                    (n.node_location[0], n.node_location[1], n.node_type) for n in self.plot_nodes],
            }
            self.live_state_callback(snapshot)
        except Exception:
            logging.debug("PlotPhysicsSystem live_state_callback failed", exc_info=True)

    # ==================================================================
    # Topologie: Plotkern-Setup (aus tools/biome_lab/topology.py)
    # ==================================================================
    def _best_candidate_sample(self, valid_mask, count, base_spacing, civ_spacing_factor, civ_map, k=15):
        """Mitchell's-Best-Candidate-Sampling. 1:1 aus tools/biome_lab/topology.py."""
        ys, xs = np.nonzero(valid_mask)
        if len(xs) == 0:
            return []

        count = min(int(count), len(xs))
        chosen = []
        chosen_arr = np.empty((0, 2), dtype=float)
        pool_size = len(xs)

        for _ in range(count):
            idxs = np.random.randint(0, pool_size, size=min(k, pool_size))
            best_pos = None
            best_score = -np.inf

            for idx in idxs:
                x = float(xs[idx])
                y = float(ys[idx])

                civ_value = float(civ_map[int(y), int(x)])
                target_spacing = max(base_spacing * (1.0 - civ_spacing_factor * civ_value), 2.5)

                if len(chosen_arr):
                    dist = float(np.sqrt(np.min((chosen_arr[:, 0] - x) ** 2 + (chosen_arr[:, 1] - y) ** 2)))
                else:
                    dist = target_spacing * 10.0

                score = dist - target_spacing
                if score > best_score:
                    best_score = score
                    best_pos = (x, y)

            if best_pos is None:
                idx = idxs[0]
                best_pos = (float(xs[idx]), float(ys[idx]))

            chosen.append(best_pos)
            chosen_arr = np.array(chosen, dtype=float)

        return chosen

    def _region_id_at(self, x, y):
        region_map = self.region_map
        if region_map is None:
            return -1
        px = int(np.clip(round(x), 0, self.map_size - 1))
        py = int(np.clip(round(y), 0, self.map_size - 1))
        return int(region_map[py, px])

    def _nearest_point_on_polyline(self, pos, polyline, closed=True):
        return _nearest_point_on_polyline(pos, polyline, closed=closed)

    def _nearest_point_on_segments(self, pos, a, b):
        return _nearest_point_on_segments(pos, a, b)

    def _polygon_area(self, vertices):
        return _polygon_area(vertices)

    def _spring_rest_length(self, civ_value, base_spacing):
        """Zivilisationsabhaengige Ruhelaenge, NIE 0. 1:1 aus
        tools/biome_lab/topology.py."""
        civ_factor = max(1.0 - self.CIV_RESTLENGTH_STEEPNESS * float(civ_value), 0.25)
        return max(float(base_spacing) * civ_factor, 2.0)

    def _gen_step_1_background(self):
        """Baut civ-abhaengige Regionen-Partitionierung + Wildnis-/Stadt-
        Polygone (Marching-Squares) auf, leert allen Node-/Topologie-Zustand.
        Entspricht tools/biome_lab/scene.py's _recompute_background() +
        topology.py's _gen_step_1_background() zusammengefasst - in der
        Produktion kommen civ_map/city_mask/heightmap bereits fertig als
        Argumente von generate() an, muessen hier nicht mehr selbst
        berechnet werden."""
        self.next_node_id = 0
        self.nodes = []
        self.plot_nodes = []
        self.vertex_to_plot_node = {}
        self._plot_node_to_vertex = {}
        self.core_registry = {}
        self.boundary_owner = {}
        self.wilderness_node_ids = set()
        self.map_border_node_ids = set()
        self._core_cell_plot_node_ids = {}
        self._core_type_by_id = {}
        self._plot_node_wilderness_cache = {}
        self._static_vertex_positions = None
        self._static_ridge_edges = []
        self._static_num_vertices = 0
        self._static_boundary_entries = []
        self._static_boundary_settlement = []
        self._static_node_entry = {}
        self._static_predecessors = None
        self._static_distances = None
        self.ridge_traffic_history = {}
        self.ridge_traffic_shrink_ema = {}
        self.path_cache = {}
        self.iteration = 0
        self.topology_ready = False

        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        civ_labeled, num_civ_regions = label(civ_mask)
        wild_labeled, num_wild_regions = label(~civ_mask)
        self.region_map = np.where(civ_mask, civ_labeled, wild_labeled + num_civ_regions).astype(np.int32)
        self.num_civ_regions = int(num_civ_regions)
        self.num_wild_regions = int(num_wild_regions)

        self._wilderness_polygons, self._wilderness_polygon_region_ids = self._build_wilderness_boundary_polygons()
        self._city_polygons = self._build_city_boundary_polygons()

        self._report_progress("plot_physics", 0, "Topologie: Hintergrund/Regionen aufgebaut")

    def _build_wilderness_boundary_polygons(self):
        """Baut echte Polygone der Zivilisationsflaeche via Marching-Squares.
        1:1 aus tools/biome_lab/scene.py's _build_wilderness_boundary_points()
        (dort Rueckgabe als Punktwolke fuer Rendering - hier direkt die
        Polygon-Liste, da kein Rendering-Anwendungsfall besteht)."""
        mask = (self.civ_map >= self.WILDERNESS_CIV_THRESHOLD).astype(np.float32)
        padded_mask = np.pad(mask, 1, mode="constant", constant_values=0.0)
        contours = measure.find_contours(padded_mask, level=0.5)
        polygons = []
        for c in contours:
            pts = np.column_stack([c[:, 1] - 1.0, c[:, 0] - 1.0])
            if len(pts) < 4:
                continue
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            candidates = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
            for cand in candidates:
                if cand.is_valid and not cand.is_empty and cand.area > self.WILDERNESS_MIN_AREA:
                    polygons.append(cand)

        region_ids = []
        for poly in polygons:
            region_id = -1
            if self.region_map is not None:
                rp = poly.representative_point()
                px = int(np.clip(round(rp.x), 0, self.map_size - 1))
                py = int(np.clip(round(rp.y), 0, self.map_size - 1))
                region_id = int(self.region_map[py, px])
            region_ids.append(region_id)
        return polygons, region_ids

    def _build_city_boundary_polygons(self):
        """Baut fuer JEDE Siedlung ein eigenes Polygon ihres Stadtgebiets
        via Marching-Squares auf city_mask == settlement.location_id. 1:1
        aus tools/biome_lab/scene.py's _build_city_boundary_polygons()."""
        city_polygons = {}
        for settlement in self.settlements:
            sid = settlement.location_id
            mask = (self.city_mask == sid).astype(np.float32)
            if not np.any(mask):
                city_polygons[sid] = []
                continue

            padded_mask = np.pad(mask, 1, mode="constant", constant_values=0.0)
            contours = measure.find_contours(padded_mask, level=0.5)
            polygons = []
            for c in contours:
                pts = np.column_stack([c[:, 1] - 1.0, c[:, 0] - 1.0])
                if len(pts) < 4:
                    continue
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    continue
                candidates = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
                for cand in candidates:
                    if cand.is_valid and not cand.is_empty and cand.area > self.CITY_MIN_AREA:
                        polygons.append(cand)
            city_polygons[sid] = polygons
        return city_polygons

    def _gen_step_2_plot_cores(self):
        """Sampled Seed-Punkte gleichmaessig ueber die gesamte Karte, mit
        civ-abhaengiger Dichte (auf civ=0.30 geflooret). 1:1 aus
        tools/biome_lab/topology.py's _gen_step_2_plot_cores() - traffic_weight
        wird hier NICHT mehr zufaellig gesetzt (siehe _assign_traffic_weights,
        laeuft nach der finalen node_type-Klassifikation)."""

        min_buffer_to_city_px = 10.0
        city_inside = self.city_mask >= 0
        dist_to_city = distance_transform_edt(~city_inside)

        edge_margin_px = int(round(self.PLOT_CORE_EDGE_MARGIN))
        map_edge_mask = np.ones_like(self.civ_map, dtype=bool)
        if edge_margin_px > 0:
            map_edge_mask[:edge_margin_px, :] = False
            map_edge_mask[-edge_margin_px:, :] = False
            map_edge_mask[:, :edge_margin_px] = False
            map_edge_mask[:, -edge_margin_px:] = False

        valid_mask = (~city_inside) & (dist_to_city >= min_buffer_to_city_px) & map_edge_mask
        effective_civ_map = np.maximum(self.civ_map, 0.30)

        positions = self._best_candidate_sample(
            valid_mask=valid_mask, count=self.plot_nodes_count, base_spacing=self.plot_base_spacing,
            civ_spacing_factor=self.plot_civ_spacing_factor, civ_map=effective_civ_map)

        for x, y in positions:
            node = PlotNode(
                node_id=self.next_node_id, node_location=(float(x), float(y)),
                connector_ids=[], connector_distances=[], connector_elevations=[],
                connector_move_costs=[], connector_edge_ids=[], settlement_id=-1,
                node_type="standard_plot_node", neighbor_core_ids=[], neighbor_node_ids=[])
            self.nodes.append(node)
            self.next_node_id += 1

        self._report_progress("plot_physics", 5, f"Topologie: {len(positions)} Seed-Punkte verteilt")

    def _gen_step_2b_relax_seed_points(self):
        """Feder-artige Relaxation der Seed-Punkte gegen ihre k-naechsten
        Nachbarn, danach Mindestabstand zur Wildnis-/Civ-Kontur und zum
        Kartenrand erzwungen. 1:1 aus tools/biome_lab/topology.py."""
        points = [n for n in self.nodes if n.node_type == "standard_plot_node"]
        if len(points) < 2:
            return

        positions = np.array([n.node_location for n in points], dtype=float)
        polylines = [np.array(poly.exterior.coords, dtype=float) for poly in self._wilderness_polygons]
        margin = self.SEED_RELAX_MARGIN
        lo, hi = margin, float(self.map_size) - margin
        k = min(self.SEED_RELAX_NEIGHBOR_COUNT + 1, len(positions))

        def _push_from_contours(pos_arr):
            for idx in range(len(pos_arr)):
                for polyline in polylines:
                    p = pos_arr[idx]
                    nearest = self._nearest_point_on_polyline(p, polyline, closed=True)
                    dist = float(np.hypot(nearest[0] - p[0], nearest[1] - p[1]))
                    if dist < margin:
                        direction = p - nearest
                        dnorm = float(np.hypot(direction[0], direction[1]))
                        direction = direction / dnorm if dnorm > 1e-9 else np.array([1.0, 0.0])
                        pos_arr[idx] = nearest + direction * margin
            return pos_arr

        for _iteration in range(self.SEED_RELAX_ITERATIONS):
            if k > 1:
                tree = cKDTree(positions)
                _dists, neighbor_idx = tree.query(positions, k=k)
                self_idx = np.repeat(np.arange(len(positions)), k - 1)
                neighbor_flat = neighbor_idx[:, 1:].ravel()

                rest_lengths_per_point = self._rest_length_core_plotnode_batch(positions)
                rest_lengths = 0.5 * (rest_lengths_per_point[self_idx] + rest_lengths_per_point[neighbor_flat])

                force_a, _force_b = self._spring_force_batch(
                    positions[self_idx], positions[neighbor_flat], rest_lengths,
                    stiffness=self.plotnode_plotnode_spring_stiffness, growth_rate=0.10)

                net_force = np.zeros_like(positions)
                np.add.at(net_force, self_idx, force_a)
                avg_force = net_force / float(k - 1)
                positions = positions + avg_force * self.SEED_RELAX_STEP

            positions[:, 0] = np.clip(positions[:, 0], lo, hi)
            positions[:, 1] = np.clip(positions[:, 1], lo, hi)
            positions = _push_from_contours(positions)

        for node, pos in zip(points, positions):
            node.node_location = (float(pos[0]), float(pos[1]))

        self._report_progress("plot_physics", 10, f"Topologie: {len(points)} Seed-Punkte relaxiert")

    def _gen_step_2c_place_city_cores(self):
        """Pro Siedlung einen dedizierten Voronoi-Seed-Punkt (city_core)
        exakt an ihrer Position setzen. 1:1 aus tools/biome_lab/topology.py
        (siehe dortiger Docstring zur Herleitung: garantiert im Gegensatz
        zur frueheren kreuzungsbasierten Erkennung IMMER eine eigene
        Voronoi-Zelle mit echten Nachbar-plot_nodes)."""

        for settlement in self.settlements:
            node = PlotNode(
                node_id=self.next_node_id, node_location=(float(settlement.x), float(settlement.y)),
                connector_ids=[], connector_distances=[], connector_elevations=[],
                connector_move_costs=[], connector_edge_ids=[], settlement_id=settlement.location_id,
                node_type="city_core", neighbor_core_ids=[], neighbor_node_ids=[])
            self.nodes.append(node)
            self.next_node_id += 1

        self._report_progress("plot_physics", 15, f"Topologie: {len(self.settlements)} Stadtkerne gesetzt")

    # ==================================================================
    # Topologie: Voronoi-Mesh (aus tools/biome_lab/topology.py)
    # ==================================================================
    def _build_plot_node_registry(self, vertex_positions, ridge_vertices_list, ridge_points):
        """Baut die plot_nodes (Voronoi-Kreuzungen) samt Nachbarschaftslisten.
        1:1 aus tools/biome_lab/topology.py (radialer Clamp entfaellt, da
        vertex_positions bereits fertig geklippt aus _build_voronoi_mesh
        kommt)."""

        registry = {}
        vertex_to_plot_node = {}

        def pos_key(pos):
            return (round(float(pos[0]), 6), round(float(pos[1]), 6))

        for ridge_idx, ridge in enumerate(ridge_vertices_list):
            if len(ridge) != 2:
                continue
            i, j = ridge
            if i < 0 or j < 0 or i >= len(vertex_positions) or j >= len(vertex_positions):
                continue

            core_a, core_b = ridge_points[ridge_idx]
            for vidx in (i, j):
                raw_pos = vertex_positions[vidx]
                pos = (float(raw_pos[0]), float(raw_pos[1]))
                key = pos_key(pos)

                if key not in registry:
                    node = PlotNode(
                        node_id=self.next_node_id, node_location=(float(pos[0]), float(pos[1])),
                        connector_ids=[], connector_distances=[], connector_elevations=[],
                        connector_move_costs=[], connector_edge_ids=[], settlement_id=-1,
                        node_type="standard_plot_node", neighbor_core_ids=[], neighbor_node_ids=[])
                    registry[key] = node
                    self.next_node_id += 1

                node = registry[key]
                vertex_to_plot_node[vidx] = node.node_id

                for core_id in (core_a, core_b):
                    if core_id not in node.neighbor_core_ids:
                        node.neighbor_core_ids.append(int(core_id))

        id_to_node = {node.node_id: node for node in registry.values()}
        for ridge in ridge_vertices_list:
            if len(ridge) != 2:
                continue
            i, j = ridge
            if i < 0 or j < 0:
                continue
            node_id_i = vertex_to_plot_node.get(i)
            node_id_j = vertex_to_plot_node.get(j)
            if node_id_i is None or node_id_j is None or node_id_i == node_id_j:
                continue
            node_i = id_to_node[node_id_i]
            node_j = id_to_node[node_id_j]
            if node_id_j not in node_i.neighbor_node_ids:
                node_i.neighbor_node_ids.append(node_id_j)
            if node_id_i not in node_j.neighbor_node_ids:
                node_j.neighbor_node_ids.append(node_id_i)

        return list(registry.values()), vertex_to_plot_node

    def _sync_core_registry(self):
        """Baut das Core-Registry komplett neu auf. 1:1 aus
        tools/biome_lab/topology.py."""

        self.core_registry = {
            idx: PlotCore(core_id=idx, location=tuple(node.node_location),
                          region_id=getattr(node, "region_id", -1))
            for idx, node in enumerate(self.nodes)
        }
        for node in self.plot_nodes:
            for core_id in node.neighbor_core_ids:
                core = self.core_registry.get(core_id)
                if core is not None and node.node_id not in core.neighbor_node_ids:
                    core.neighbor_node_ids.append(node.node_id)

    def _sync_core_positions(self):
        for idx, node in enumerate(self.nodes):
            core = self.core_registry.get(idx)
            if core is not None:
                core.location = tuple(node.node_location)

    def _build_voronoi_mesh(self):
        """Baut EIN globales Voronoi ueber alle 'standard_plot_node'/
        'city_core'-Seed-Punkte, geklippt auf das Kartenrechteck. 1:1 aus
        tools/biome_lab/topology.py's _build_voronoi_mesh() - OHNE die dort
        vorhandene (bereits ungenutzte) Worker-Subprocess-Option, siehe
        Modul-Docstring."""
        all_core_nodes = [n for n in self.nodes if n.node_type in ("standard_plot_node", "city_core")]
        if len(all_core_nodes) < 4:
            logging.error("PlotPhysicsSystem: zu wenige Kern-Nodes fuer Topologie.")
            return False

        points = np.array([n.node_location for n in all_core_nodes], dtype=float)
        node_ids = [n.node_id for n in all_core_nodes]

        try:
            vor = Voronoi(points)
        except Exception as e:
            logging.error(f"PlotPhysicsSystem: Voronoi-Berechnung fehlgeschlagen: {e}")
            return False

        inset = self.PLOT_CORE_EDGE_MARGIN
        x0, y0 = inset, inset
        x1, y1 = float(self.map_size) - inset, float(self.map_size) - inset
        clip_box = box(x0, y0, x1, y1)

        center = points.mean(axis=0)
        far_distance = float(self.map_size) * 4.0

        vertex_positions_list = [np.asarray(v, dtype=float) for v in vor.vertices]
        clip_point_index = {}

        def _clip_index_for(pt):
            key = (round(float(pt[0]), 4), round(float(pt[1]), 4))
            idx = clip_point_index.get(key)
            if idx is not None:
                return idx
            idx = len(vertex_positions_list)
            vertex_positions_list.append(np.array([float(pt[0]), float(pt[1])], dtype=float))
            clip_point_index[key] = idx
            return idx

        def _clip_segment(a, b):
            a_inside = clip_box.covers(Point(a))
            b_inside = clip_box.covers(Point(b))
            if a_inside and b_inside:
                return None
            seg = LineString([a, b])
            clipped = seg.intersection(clip_box)
            if clipped.is_empty:
                return "OUTSIDE"
            coords = list(clipped.coords)
            if len(coords) < 2:
                return "OUTSIDE"
            p_first, p_last = np.array(coords[0]), np.array(coords[-1])
            if np.hypot(*(p_first - a)) <= np.hypot(*(p_last - a)):
                pa_c, pb_c = p_first, p_last
            else:
                pa_c, pb_c = p_last, p_first
            return (pa_c, pb_c, a_inside, b_inside)

        ridge_vertices_list = []
        ridge_points = []
        boundary_vertex_indices = set()

        for ridge_idx, (i, j) in enumerate(vor.ridge_vertices):
            p1_local, p2_local = vor.ridge_points[ridge_idx]

            if i >= 0 and j >= 0:
                a_pos = np.asarray(vor.vertices[i], dtype=float)
                b_pos = np.asarray(vor.vertices[j], dtype=float)
                result = _clip_segment(a_pos, b_pos)
                if result is None:
                    idx_a, idx_b = i, j
                elif result == "OUTSIDE":
                    continue
                else:
                    pa_c, pb_c, a_inside, b_inside = result
                    idx_a = i if a_inside else _clip_index_for(pa_c)
                    idx_b = j if b_inside else _clip_index_for(pb_c)
                    if not a_inside:
                        boundary_vertex_indices.add(idx_a)
                    if not b_inside:
                        boundary_vertex_indices.add(idx_b)
            else:
                finite_idx = j if i < 0 else i
                finite_vertex = np.asarray(vor.vertices[finite_idx], dtype=float)
                t = points[p2_local] - points[p1_local]
                norm_t = float(np.linalg.norm(t))
                if norm_t < 1e-9:
                    continue
                t = t / norm_t
                n = np.array([-t[1], t[0]])
                midpoint = (points[p1_local] + points[p2_local]) / 2.0
                direction = n if np.dot(midpoint - center, n) > 0 else -n
                far_point = finite_vertex + direction * far_distance

                result = _clip_segment(finite_vertex, far_point)
                if result is None or result == "OUTSIDE":
                    continue
                pa_c, pb_c, a_inside, b_inside = result
                idx_a = finite_idx if a_inside else _clip_index_for(pa_c)
                idx_b = finite_idx if b_inside else _clip_index_for(pb_c)
                if not a_inside:
                    boundary_vertex_indices.add(idx_a)
                if not b_inside:
                    boundary_vertex_indices.add(idx_b)

            if idx_a == idx_b:
                continue
            ridge_vertices_list.append((idx_a, idx_b))
            ridge_points.append((node_ids[p1_local], node_ids[p2_local]))

        if not ridge_vertices_list:
            logging.error("PlotPhysicsSystem: keine gueltigen Ridge-Kanten nach dem Klippen gefunden.")
            return False

        used_indices = sorted({idx for pair in ridge_vertices_list for idx in pair})
        remap = {old: new for new, old in enumerate(used_indices)}
        vertex_positions = np.array([vertex_positions_list[old] for old in used_indices], dtype=float)
        ridge_vertices_list = [(remap[a], remap[b]) for a, b in ridge_vertices_list]
        boundary_vertex_indices = {remap[old] for old in boundary_vertex_indices if old in remap}
        num_vertices = len(vertex_positions)

        ridge_edges = []
        for i, j in ridge_vertices_list:
            p1 = vertex_positions[i]
            p2 = vertex_positions[j]
            seg_len = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            if seg_len <= 1e-6:
                continue
            raw_slope = self._sampled_slope(p1, p2, seg_len)
            normalized_slope = min(1.0, raw_slope / 30.0)
            cost = seg_len * (1.0 + self.plot_height_cost_factor * normalized_slope)
            ridge_edges.append((i, j, p1, p2, cost))

        if not ridge_edges:
            logging.error("PlotPhysicsSystem: keine gueltigen Ridge-Kanten fuer Graph gefunden.")
            return False

        self.plot_nodes, self.vertex_to_plot_node = self._build_plot_node_registry(
            vertex_positions=vertex_positions, ridge_vertices_list=ridge_vertices_list, ridge_points=ridge_points)
        self._plot_node_to_vertex = {}
        for vidx, pid in self.vertex_to_plot_node.items():
            self._plot_node_to_vertex.setdefault(pid, vidx)

        plot_node_by_id = {pn.node_id: pn for pn in self.plot_nodes}
        self.map_border_node_ids = set()
        for vidx in boundary_vertex_indices:
            pid = self.vertex_to_plot_node.get(vidx)
            if pid is None:
                continue
            plot_node = plot_node_by_id.get(pid)
            if plot_node is not None:
                plot_node.node_type = "map_border_node"
                self.map_border_node_ids.add(pid)

        self._static_vertex_positions = vertex_positions
        self._static_ridge_edges = ridge_edges
        self._static_num_vertices = num_vertices
        return True

    def _gen_step_4_voronoi_clipped(self):
        if not self._build_voronoi_mesh():
            self.topology_ready = False
            return False
        self._report_progress(
            "plot_physics", 20,
            f"Topologie: Voronoi ({len(self.plot_nodes)} plot_nodes, "
            f"{len(self.map_border_node_ids)} davon Kartenrand)")
        return True

    def _gen_step_city_boundary_distribute(self):
        """Verteilt fuer jeden Stadtkern dessen eigene Voronoi-Zellen-Nachbarn
        auf die tatsaechliche Stadtkontur und macht sie unbeweglich. 1:1 aus
        tools/biome_lab/topology.py."""
        city_cores = [n for n in self.nodes if n.node_type == "city_core"]
        if not city_cores:
            return

        def _snap_to_contour(pos, polylines):
            best_point, best_dist = None, np.inf
            for polyline in polylines:
                point = self._nearest_point_on_polyline(pos, polyline, closed=True)
                dist = float(np.hypot(point[0] - pos[0], point[1] - pos[1]))
                if dist < best_dist:
                    best_dist, best_point = dist, point
            return best_point

        distributed_count = 0
        for city_core in city_cores:
            settlement_id = city_core.settlement_id
            own_neighbors = [
                pn for pn in self.plot_nodes
                if city_core.node_id in pn.neighbor_core_ids and pn.node_type == "standard_plot_node"
            ]
            if not own_neighbors:
                logging.warning(f"PlotPhysicsSystem: Stadtkern settlement_id={settlement_id} hat keine eigenen Voronoi-Nachbarn.")
                continue

            polygons = self._city_polygons.get(settlement_id) or []
            polylines = [np.array(poly.exterior.coords, dtype=float) for poly in polygons] if polygons else []

            for pn in own_neighbors:
                if polylines:
                    snapped = _snap_to_contour(pn.node_location, polylines)
                    if snapped is not None:
                        pn.node_location = (float(snapped[0]), float(snapped[1]))
                pn.node_type = "city_border_node"
                pn.settlement_id = settlement_id
                self.boundary_owner[pn.node_id] = settlement_id
                distributed_count += 1

        self._report_progress(
            "plot_physics", 25, f"Topologie: {distributed_count} plot_nodes zu Stadtgrenze verteilt")

    def _gen_step_5_wilderness_snap(self):
        """Findet jede Ridge-Kante, die die Wildnisgrenze kreuzt, und snapt
        die naeheren (oder bei 2+ Kreuzungen beide) Endpunkte auf die Kontur.
        1:1 aus tools/biome_lab/topology.py."""
        if self._static_vertex_positions is None or not self.plot_nodes:
            return
        polygons = list(self._wilderness_polygons)
        if not polygons:
            return

        plot_node_by_id = {pn.node_id: pn for pn in self.plot_nodes}
        boundaries = [poly.exterior for poly in polygons]
        polylines = [np.array(b.coords, dtype=float) for b in boundaries]

        def _snap_to_contour(pos):
            best_point, best_dist = None, np.inf
            for polyline in polylines:
                point = self._nearest_point_on_polyline(pos, polyline, closed=True)
                dist = float(np.hypot(point[0] - pos[0], point[1] - pos[1]))
                if dist < best_dist:
                    best_dist, best_point = dist, point
            return best_point

        snapped_count = 0
        for i, j, p1, p2, cost in self._static_ridge_edges:
            pid_a = self.vertex_to_plot_node.get(i)
            pid_b = self.vertex_to_plot_node.get(j)
            if pid_a is None or pid_b is None or pid_a == pid_b:
                continue
            node_a = plot_node_by_id.get(pid_a)
            node_b = plot_node_by_id.get(pid_b)
            if node_a is None or node_b is None:
                continue
            if node_a.node_type != "standard_plot_node" or node_b.node_type != "standard_plot_node":
                continue

            seg = LineString([p1, p2])
            seg_vec = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
            seg_len_sq = float(np.dot(seg_vec, seg_vec))

            crossing_ts = []
            for boundary in boundaries:
                if not seg.intersects(boundary):
                    continue
                intersection = seg.intersection(boundary)
                if intersection.is_empty:
                    continue
                pts = _extract_intersection_points(intersection)
                for px, py in pts:
                    t = float(np.dot((px - p1[0], py - p1[1]), seg_vec) / seg_len_sq) if seg_len_sq > 1e-12 else 0.0
                    crossing_ts.append((t, (float(px), float(py))))

            if not crossing_ts:
                continue

            crossing_ts.sort(key=lambda item: item[0])
            deduped = []
            for t, pt in crossing_ts:
                if deduped and abs(t - deduped[-1][0]) < 1e-6:
                    continue
                deduped.append((t, pt))
            crossing_ts = deduped

            if len(crossing_ts) == 1:
                _, (ix, iy) = crossing_ts[0]
                dist_a = float(np.hypot(node_a.node_location[0] - ix, node_a.node_location[1] - iy))
                dist_b = float(np.hypot(node_b.node_location[0] - ix, node_b.node_location[1] - iy))
                target_node = node_a if dist_a <= dist_b else node_b

                snapped = _snap_to_contour(target_node.node_location)
                if snapped is not None:
                    target_node.node_location = (float(snapped[0]), float(snapped[1]))
                    target_node.node_type = "wilderness_node"
                    self.wilderness_node_ids.add(target_node.node_id)
                    snapped_count += 1
                continue

            snapped_a = _snap_to_contour(node_a.node_location)
            if snapped_a is not None:
                node_a.node_location = (float(snapped_a[0]), float(snapped_a[1]))
            node_a.node_type = "wilderness_node"
            self.wilderness_node_ids.add(node_a.node_id)

            snapped_b = _snap_to_contour(node_b.node_location)
            if snapped_b is not None:
                node_b.node_location = (float(snapped_b[0]), float(snapped_b[1]))
            node_b.node_type = "wilderness_node"
            self.wilderness_node_ids.add(node_b.node_id)
            snapped_count += 2

        self._sanitize_plot_node_positions()
        self._report_progress("plot_physics", 30, f"Topologie: {snapped_count} plot_nodes zu Wildnisgrenze gesnappt")

    def _sanitize_plot_node_positions(self):
        """Restkorrektur: manche zivilisationsnahe plot_nodes bleiben trotz
        Wildnisgrenzen-Snap auf der falschen Seite (Kanten, die schon durch
        eine frueher verarbeitete Nachbarkante 'verbraucht' wurden). 1:1 aus
        tools/biome_lab/topology.py (siehe dortiger Docstring zur
        wichtigen Innen/Aussen-Konvention: _wilderness_polygons' Inneres
        ist die CIV-Region, nicht die Wildnis)."""
        prepared = self._prepare_wilderness_polygons()
        if not prepared:
            return

        type_by_id = {node.node_id: node.node_type for node in self.nodes}
        polygons = list(self._wilderness_polygons)
        polylines = [np.array(poly.exterior.coords, dtype=float) for poly in polygons]

        def _is_civ_adjacent(pn):
            return any(type_by_id.get(cid) == "standard_plot_node" for cid in pn.neighbor_core_ids)

        def _is_wrong_side(pn):
            pos = np.array(pn.node_location, dtype=float)
            return not any(self._point_in_polygon(pos, coords, shifted) for coords, shifted, _b in prepared)

        wrong_side_nodes = [
            pn for pn in self.plot_nodes
            if pn.node_type == "standard_plot_node" and _is_civ_adjacent(pn) and _is_wrong_side(pn)
        ]
        if not wrong_side_nodes:
            return

        def _snap_to_contour(pos):
            best_point, best_dist = None, np.inf
            for polyline in polylines:
                point = self._nearest_point_on_polyline(pos, polyline, closed=True)
                dist = float(np.hypot(point[0] - pos[0], point[1] - pos[1]))
                if dist < best_dist:
                    best_dist, best_point = dist, point
            return best_point

        for pn in wrong_side_nodes:
            snapped = _snap_to_contour(np.array(pn.node_location, dtype=float))
            if snapped is None:
                continue
            pn.node_location = (float(snapped[0]), float(snapped[1]))
            pn.node_type = "wilderness_node"
            self.wilderness_node_ids.add(pn.node_id)

    def _gen_step_6_wilderness_cores(self):
        """Jeder verbliebene 'standard_plot_node', dessen civ-Wert unter
        WILDERNESS_CIV_THRESHOLD liegt, wird zu 'wilderness_core'
        umklassifiziert (reine Label-Aenderung). 1:1 aus
        tools/biome_lab/topology.py."""
        reclassified = 0
        for node in self.nodes:
            if node.node_type != "standard_plot_node":
                continue
            x, y = node.node_location
            px = int(np.clip(round(x), 0, self.map_size - 1))
            py = int(np.clip(round(y), 0, self.map_size - 1))
            if float(self.civ_map[py, px]) < self.WILDERNESS_CIV_THRESHOLD:
                node.node_type = "wilderness_core"
                reclassified += 1
        self._report_progress("plot_physics", 35, f"Topologie: {reclassified} Wildniskerne umklassifiziert")

    def _assign_traffic_weights(self):
        """NEU (ersetzt die zufaellige Zuweisung aus tools/biome_lab/
        topology.py:543, siehe Modul-Docstring): traffic_weight wird nach
        der FINALEN node_type-Klassifikation fest zugewiesen (Nutzer-
        Vorgabe: Plots=2.0, Wildniskerne=1.0, Stadtkerne erhoeht). Muss NACH
        _gen_step_6_wilderness_cores laufen, da standard_plot_node erst dort
        final von wilderness_core unterschieden wird."""
        for node in self.nodes:
            if node.node_type == "standard_plot_node":
                node.traffic_weight = self.traffic_weight_standard_plot
            elif node.node_type == "wilderness_core":
                node.traffic_weight = self.traffic_weight_wilderness
            elif node.node_type == "city_core":
                node.traffic_weight = self.traffic_weight_city_core

    def _gen_step_7_build_graph(self):
        """Baut den Dijkstra-Graph aus dem fertigen, klassifizierten Netz.
        1:1 aus tools/biome_lab/topology.py."""
        vertex_positions = self._static_vertex_positions
        ridge_edges = self._static_ridge_edges
        if vertex_positions is None or not ridge_edges:
            return
        num_vertices = self._static_num_vertices

        global_tree = cKDTree(vertex_positions)
        node_entry = {}
        entry_edges = []
        for idx, node in enumerate(self.nodes):
            node_pos = np.array(node.node_location, dtype=float)
            entry_idx = num_vertices + idx
            node_entry[node.node_id] = entry_idx
            dist, vertex_idx = global_tree.query(node_pos)
            entry_edges.append((entry_idx, int(vertex_idx), max(float(dist), 1e-3)))

        total_graph_nodes = num_vertices + len(self.nodes)
        rows, cols, costs = [], [], []
        for i, j, _p1, _p2, cost in ridge_edges:
            rows.extend([i, j])
            cols.extend([j, i])
            costs.extend([cost, cost])
        for entry_idx, vertex_idx, cost in entry_edges:
            rows.extend([entry_idx, vertex_idx])
            cols.extend([vertex_idx, entry_idx])
            costs.extend([cost, cost])

        boundary_entries = []
        boundary_settlement = []
        for node_id, settlement_id in self.boundary_owner.items():
            if node_id in node_entry:
                boundary_entries.append(node_entry[node_id])
                boundary_settlement.append(settlement_id)
            elif node_id in self._plot_node_to_vertex:
                boundary_entries.append(self._plot_node_to_vertex[node_id])
                boundary_settlement.append(settlement_id)

        if not boundary_entries:
            logging.warning("PlotPhysicsSystem: keine Boundary-Entries gefunden, Graph unvollstaendig.")
            return

        try:
            graph = csr_matrix((costs, (rows, cols)), shape=(total_graph_nodes, total_graph_nodes))
            distances, predecessors = dijkstra(graph, indices=boundary_entries, return_predecessors=True)
        except Exception as e:
            logging.error(f"PlotPhysicsSystem: Dijkstra-Berechnung fehlgeschlagen: {e}")
            return

        self._static_boundary_entries = boundary_entries
        self._static_boundary_settlement = boundary_settlement
        self._static_node_entry = node_entry
        self._static_predecessors = predecessors
        self._static_distances = distances
        self._report_progress("plot_physics", 40, f"Topologie: Dijkstra-Graph ({total_graph_nodes} Knoten) aufgebaut")

    def _gen_step_9_finalize(self):
        """Registries synchronisieren, Zellgrenzen aufbauen, plot_nodes
        einmalig gegen die Wildnisgrenze bereinigen, Geschwindigkeiten
        nullen. 1:1 aus tools/biome_lab/topology.py."""
        self.topology_ready = True
        self._sync_core_registry()
        self._core_type_by_id = {node.node_id: node.node_type for node in self.nodes}
        self._build_core_cell_plot_node_ids()

        for node in self.nodes:
            node.velocity = (0.0, 0.0)
        for node in self.plot_nodes:
            node.velocity = (0.0, 0.0)
        self._report_progress(
            "plot_physics", 45, f"Topologie finalisiert: {len(self.plot_nodes)} plot_nodes, {len(self.nodes)} Nodes")

    def _build_core_cell_plot_node_ids(self):
        """Merkt sich fuer jeden regulaeren Plotkern die nach Winkel
        sortierten IDs seiner benachbarten plot_nodes (harte Bewegungsgrenze
        in _physics_step). 1:1 aus tools/biome_lab/topology.py."""
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
        self._core_cell_plot_node_ids = {}

        for node in self.nodes:
            if node.node_type != "standard_plot_node":
                continue
            core = self.core_registry.get(node.node_id)
            if core is None or len(core.neighbor_node_ids) < 3:
                continue

            cx, cy = node.node_location
            ids = [nid for nid in core.neighbor_node_ids if nid in plot_node_by_id]
            if len(ids) < 3:
                continue
            positions = np.array([plot_node_by_id[nid].node_location for nid in ids], dtype=float)

            dists = np.hypot(positions[:, 0] - cx, positions[:, 1] - cy)
            median_dist = float(np.median(dists))
            if median_dist > 1e-6:
                keep = dists <= median_dist * 3.0
                if np.count_nonzero(keep) >= 3:
                    ids = [ids[i] for i in range(len(ids)) if keep[i]]
                    positions = positions[keep]

            angles = np.arctan2(positions[:, 1] - cy, positions[:, 0] - cx)
            order = np.argsort(angles)
            self._core_cell_plot_node_ids[node.node_id] = [ids[i] for i in order]

    def _sampled_slope(self, p1, p2, length):
        if length < 1e-6:
            return 0.0
        if length <= 12.0:
            return abs(self._height_at(p1) - self._height_at(p2)) / length
        num_segments = max(1, int(np.ceil(length / 10.0)))
        t = np.linspace(0, 1, num_segments + 1)
        xs = p1[0] + (p2[0] - p1[0]) * t
        ys = p1[1] + (p2[1] - p1[1]) * t
        h, w = self.heightmap.shape
        ix = np.clip(xs.round().astype(int), 0, w - 1)
        iy = np.clip(ys.round().astype(int), 0, h - 1)
        heights = self.heightmap[iy, ix]
        cumulative_height_change = float(np.sum(np.abs(np.diff(heights))))
        return cumulative_height_change / length

    def _height_at(self, pos):
        x, y = int(round(pos[0])), int(round(pos[1]))
        h, w = self.heightmap.shape
        if 0 <= y < h and 0 <= x < w:
            return float(self.heightmap[y, x])
        return 0.0

    # ==================================================================
    # Physik: Federn, Integration (aus tools/biome_lab/physics.py)
    # ==================================================================
    def _civ_at_continuous(self, pos):
        h, w = self.civ_map.shape
        x = float(np.clip(pos[0], 0.0, w - 1.0))
        y = float(np.clip(pos[1], 0.0, h - 1.0))
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        tx, ty = x - x0, y - y0
        return float(
            self.civ_map[y0, x0] * (1.0 - tx) * (1.0 - ty) + self.civ_map[y0, x1] * tx * (1.0 - ty)
            + self.civ_map[y1, x0] * (1.0 - tx) * ty + self.civ_map[y1, x1] * tx * ty)

    def _civ_at_continuous_batch(self, positions):
        h, w = self.civ_map.shape
        pos = np.asarray(positions, dtype=float)
        x = np.clip(pos[:, 0], 0.0, w - 1.0)
        y = np.clip(pos[:, 1], 0.0, h - 1.0)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        tx = x - x0
        ty = y - y0
        return (self.civ_map[y0, x0] * (1 - tx) * (1 - ty) + self.civ_map[y0, x1] * tx * (1 - ty)
                + self.civ_map[y1, x0] * (1 - tx) * ty + self.civ_map[y1, x1] * tx * ty)

    def _safe_exp_spring_magnitude(self, deviation, stiffness, growth_rate=0.12, max_exp_argument=30.0, max_force=None):
        if max_force is None:
            max_force = self.MAX_SPRING_RESULTANT
        scaled = float(deviation) * float(growth_rate)
        scaled = float(np.clip(scaled, -max_exp_argument, max_exp_argument))
        if deviation >= 0.0:
            magnitude = float(stiffness) * (np.exp(scaled) - 1.0)
        else:
            magnitude = -float(stiffness) * (np.exp(-scaled) - 1.0)
        return float(np.clip(magnitude, -max_force, max_force))

    def _spring_force_batch(self, pos_a_arr, pos_b_arr, rest_lengths, stiffness, growth_rate=0.12):
        pos_a_arr = np.asarray(pos_a_arr, dtype=float)
        pos_b_arr = np.asarray(pos_b_arr, dtype=float)
        rest_lengths = np.asarray(rest_lengths, dtype=float)
        delta = pos_b_arr - pos_a_arr
        dist = np.hypot(delta[:, 0], delta[:, 1])
        safe_dist = np.where(dist <= 1e-9, 1.0, dist)
        direction = delta / safe_dist[:, None]
        deviation = dist - rest_lengths
        scaled = np.clip(deviation * growth_rate, -30.0, 30.0)
        magnitude = np.where(
            deviation >= 0.0, stiffness * (np.exp(scaled) - 1.0), -stiffness * (np.exp(-scaled) - 1.0))
        magnitude = np.clip(magnitude, -self.MAX_SPRING_RESULTANT, self.MAX_SPRING_RESULTANT)
        force_a = direction * magnitude[:, None]
        force_a = np.where((dist <= 1e-9)[:, None], 0.0, force_a)
        return force_a, -force_a

    def _rest_length_core_plotnode_batch(self, positions):
        civ_here = self._civ_at_continuous_batch(positions)
        civ_factor = np.maximum(1.0 - self.CIV_RESTLENGTH_STEEPNESS * civ_here, 0.25)
        return np.maximum(self.plot_base_spacing * civ_factor, 2.0)

    def _rest_length_plotnode_plotnode_batch(self, pos_a_arr, pos_b_arr, traffic_values):
        mids = 0.5 * (np.asarray(pos_a_arr, dtype=float) + np.asarray(pos_b_arr, dtype=float))
        civ_here = self._civ_at_continuous_batch(mids)
        civ_factor = np.maximum(1.0 - self.CIV_RESTLENGTH_STEEPNESS * civ_here, 0.25)
        base = self.plot_base_spacing * civ_factor
        shrink = 1.0 - np.minimum(np.asarray(traffic_values, dtype=float) * self.spring_traffic_shrink,
                                   1.0 - self.spring_min_shrink_fraction)
        shrink = np.maximum(shrink, self.spring_min_shrink_fraction)
        return np.maximum(base * shrink, 2.0)

    def _edge_key(self, i, j):
        return tuple(sorted((int(i), int(j))))

    def _apply_spring_forces(self):
        """Sammelt alle Federkraefte + Innendruck + Kollisionsabstossung +
        weiche Wildnisgrenze pro Tick. 1:1 aus tools/biome_lab/physics.py's
        _apply_spring_forces()."""
        active_cores = [n for n in self.nodes if n.node_type == "standard_plot_node"]
        core_ids = [n.node_id for n in active_cores]
        core_index = {nid: i for i, nid in enumerate(core_ids)}
        node_by_id_all = {n.node_id: n for n in self.nodes}

        plot_ids = [n.node_id for n in self.plot_nodes]
        plot_index = {nid: i for i, nid in enumerate(plot_ids)}
        plot_node_by_id = {n.node_id: n for n in self.plot_nodes}

        core_force_arr = np.zeros((len(core_ids), 2), dtype=float)
        plot_force_arr = np.zeros((len(plot_ids), 2), dtype=float)

        if self.enable_core_plotnode_spring:
            pairs_core_id, pairs_plot_idx = [], []
            pos_core_list, pos_plot_list = [], []
            for cid, core in self.core_registry.items():
                core_node = node_by_id_all.get(cid)
                if core_node is None or core_node.node_type not in ("standard_plot_node", "wilderness_core"):
                    continue
                for pid in core.neighbor_node_ids:
                    plot_node = plot_node_by_id.get(pid)
                    if plot_node is None:
                        continue
                    pairs_core_id.append(cid)
                    pairs_plot_idx.append(plot_index[pid])
                    pos_core_list.append(core_node.node_location)
                    pos_plot_list.append(plot_node.node_location)

            if pairs_core_id:
                pos_core_arr = np.array(pos_core_list, dtype=float)
                pos_plot_arr = np.array(pos_plot_list, dtype=float)
                rest_arr = self._rest_length_core_plotnode_batch(pos_core_arr)
                force_core, force_plot = self._spring_force_batch(
                    pos_core_arr, pos_plot_arr, rest_arr, self.core_plotnode_spring_stiffness, growth_rate=0.15)

                plot_idx_arr = np.array(pairs_plot_idx, dtype=int)
                np.add.at(plot_force_arr, plot_idx_arr, force_plot)

                active_mask = np.array([cid in core_index for cid in pairs_core_id], dtype=bool)
                if np.any(active_mask):
                    core_idx_arr = np.array(
                        [core_index[cid] for cid in pairs_core_id if cid in core_index], dtype=int)
                    np.add.at(core_force_arr, core_idx_arr, force_core[active_mask])

        if self.enable_plotnode_plotnode_spring:
            vertex_by_plot_node = self._plot_node_to_vertex
            seen_pairs = set()
            pairs_i, pairs_j = [], []
            pos_a_list, pos_b_list, traffic_list = [], [], []
            for plot_node in self.plot_nodes:
                for other_id in plot_node.neighbor_node_ids:
                    if other_id == plot_node.node_id:
                        continue
                    pair = tuple(sorted((plot_node.node_id, other_id)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    other = plot_node_by_id.get(other_id)
                    if other is None:
                        continue
                    pairs_i.append(plot_index[plot_node.node_id])
                    pairs_j.append(plot_index[other_id])
                    pos_a_list.append(plot_node.node_location)
                    pos_b_list.append(other.node_location)

                    vi = vertex_by_plot_node.get(plot_node.node_id)
                    vj = vertex_by_plot_node.get(other_id)
                    if vi is not None and vj is not None:
                        traffic_list.append(self.ridge_traffic_shrink_ema.get(self._edge_key(vi, vj), 0.0))
                    else:
                        traffic_list.append(0.0)

            if pairs_i:
                pos_a_arr = np.array(pos_a_list, dtype=float)
                pos_b_arr = np.array(pos_b_list, dtype=float)
                rest_arr = self._rest_length_plotnode_plotnode_batch(pos_a_arr, pos_b_arr, traffic_list)
                force_a, force_b = self._spring_force_batch(
                    pos_a_arr, pos_b_arr, rest_arr, self.plotnode_plotnode_spring_stiffness, growth_rate=0.10)
                np.add.at(plot_force_arr, pairs_i, force_a)
                np.add.at(plot_force_arr, pairs_j, force_b)

        pressure_strength = self.pressure_strength if self.enable_pressure else 0.0
        repulsion_strength = self.plot_node_repulsion_strength if self.enable_plot_node_repulsion else 0.0
        if pressure_strength > 1e-9 or repulsion_strength > 1e-9:
            for node in active_cores:
                ids_for_cell = self._core_cell_plot_node_ids.get(node.node_id)
                if not ids_for_cell:
                    continue
                ids_present = [pid for pid in ids_for_cell if pid in plot_node_by_id]
                if len(ids_present) < 3:
                    continue
                positions = np.array([plot_node_by_id[pid].node_location for pid in ids_present], dtype=float)
                ideal_radius = self._spring_rest_length(self._civ_at_continuous(node.node_location), self.plot_base_spacing)

                if pressure_strength > 1e-9:
                    area = self._polygon_area(positions)
                    ideal_area = np.pi * ideal_radius ** 2
                    if area < 1e-6:
                        pressure_mag = pressure_strength
                    else:
                        pressure_mag = pressure_strength * max(ideal_area / area - 1.0, 0.0)
                    if pressure_mag > 1e-9:
                        core_pos = np.array(node.node_location, dtype=float)
                        deltas = positions - core_pos
                        dists = np.hypot(deltas[:, 0], deltas[:, 1])
                        safe_dists = np.where(dists > 1e-9, dists, 1.0)
                        directions = deltas / safe_dists[:, None]
                        forces = directions * pressure_mag
                        idx_arr = np.array([plot_index[pid] for pid in ids_present], dtype=int)
                        np.add.at(plot_force_arr, idx_arr, forces)

                if repulsion_strength > 1e-9:
                    min_sep = max(ideal_radius * 0.6, 3.0)
                    n = len(ids_present)
                    for a_idx in range(n):
                        for b_idx in range(a_idx + 1, n):
                            delta = positions[a_idx] - positions[b_idx]
                            dist = float(np.hypot(delta[0], delta[1]))
                            if dist >= min_sep:
                                continue
                            if dist > 1e-6:
                                direction = delta / dist
                            else:
                                pid_a = ids_present[a_idx]
                                angle = (pid_a * 2654435761) % 360 * np.pi / 180.0
                                direction = np.array([np.cos(angle), np.sin(angle)])
                            safe_dist = max(dist, 0.5)
                            magnitude = repulsion_strength * (min_sep / safe_dist - 1.0)
                            magnitude = min(magnitude, self.MAX_SPRING_RESULTANT)
                            push = direction * magnitude
                            ia, ib = plot_index[ids_present[a_idx]], plot_index[ids_present[b_idx]]
                            plot_force_arr[ia] += push
                            plot_force_arr[ib] -= push

        if self.enable_wilderness_containment:
            prepared_wilderness = self._prepare_wilderness_polygons()
            if prepared_wilderness:
                for plot_node in self.plot_nodes:
                    pos = np.asarray(plot_node.node_location, dtype=float)
                    target = self._contain_plot_node(plot_node, pos, prepared_wilderness, cap=False)
                    correction = target - pos
                    dist = float(np.hypot(correction[0], correction[1]))
                    if dist <= 1e-9:
                        continue
                    direction = correction / dist
                    magnitude = self._safe_exp_spring_magnitude(
                        deviation=dist, stiffness=self.wilderness_push_stiffness, growth_rate=0.15)
                    magnitude = max(magnitude, 0.0)
                    idx = plot_index[plot_node.node_id]
                    plot_force_arr[idx] += direction * magnitude

        core_norms = np.hypot(core_force_arr[:, 0], core_force_arr[:, 1])
        scale = np.where(core_norms > self.MAX_SPRING_RESULTANT,
                          self.MAX_SPRING_RESULTANT / np.maximum(core_norms, 1e-12), 1.0)
        core_force_arr *= scale[:, None]

        plot_norms = np.hypot(plot_force_arr[:, 0], plot_force_arr[:, 1])
        scale = np.where(plot_norms > self.MAX_SPRING_RESULTANT,
                          self.MAX_SPRING_RESULTANT / np.maximum(plot_norms, 1e-12), 1.0)
        plot_force_arr *= scale[:, None]

        core_forces = {nid: core_force_arr[i] for i, nid in enumerate(core_ids)}
        plot_forces = {nid: plot_force_arr[i] for i, nid in enumerate(plot_ids)}
        return core_forces, plot_forces

    def _reflect_velocity_on_correction(self, velocity, pos_before, pos_after):
        correction = np.asarray(pos_after, dtype=float) - np.asarray(pos_before, dtype=float)
        dist = float(np.hypot(correction[0], correction[1]))
        if dist <= 1e-9:
            return velocity
        direction = correction / dist
        v_along = float(np.dot(velocity, direction))
        if v_along < 0.0:
            return velocity - v_along * direction
        return velocity

    def _physics_step(self):
        """Semi-implizite/symplektische Euler-Integration fuer aktive Kerne
        UND plot_nodes. Returns: float - maximale Node-Verschiebung diesen
        Tick (0.0 falls nichts integriert wurde), fuer die neue Konvergenz-
        Erkennung (siehe _run_physics_to_convergence, existiert im Lab
        nicht). 1:1 aus tools/biome_lab/physics.py's _physics_step(), NUR um
        den Verschiebungs-Rueckgabewert erweitert."""
        if not self.topology_ready:
            return 0.0

        core_forces, plot_forces = self._apply_spring_forces()
        dt = self.PHYSICS_TIME_STEP
        max_speed = self.MAX_DISPLACEMENT_PER_TICK / dt
        zero2 = np.zeros(2, dtype=float)
        max_displacement = 0.0

        active_cores = [n for n in self.nodes if n.node_type == "standard_plot_node"]
        if active_cores:
            plot_node_by_id = {n.node_id: n for n in self.plot_nodes}
            core_positions = np.array([n.node_location for n in active_cores], dtype=float)
            core_velocities = np.array([n.velocity for n in active_cores], dtype=float)

            core_field = self._sample_field_batch(core_positions) if self.enable_field_cores else np.zeros_like(core_positions)
            core_total_force = (np.array([core_forces.get(n.node_id, zero2) for n in active_cores], dtype=float)
                                 + core_field)
            core_force_norms = np.hypot(core_total_force[:, 0], core_total_force[:, 1])
            core_force_scale = np.where(core_force_norms > self.MAX_SPRING_RESULTANT,
                                         self.MAX_SPRING_RESULTANT / np.maximum(core_force_norms, 1e-12), 1.0)
            core_total_force *= core_force_scale[:, None]

            core_accel = core_total_force / self.core_mass
            core_velocities = (core_velocities + core_accel * dt) * self.damping
            core_speed = np.hypot(core_velocities[:, 0], core_velocities[:, 1])
            core_speed_scale = np.where(core_speed > max_speed, max_speed / np.maximum(core_speed, 1e-12), 1.0)
            core_velocities *= core_speed_scale[:, None]

            core_free = core_positions + core_velocities * dt
            core_free[:, 0] = np.clip(core_free[:, 0], 0.0, self.map_size - 1.0)
            core_free[:, 1] = np.clip(core_free[:, 1], 0.0, self.map_size - 1.0)

            for idx, node in enumerate(active_cores):
                pos_before = core_free[idx]
                if self.enable_core_cell_containment:
                    pos_after = self._contain_core_in_cell(node, pos_before, plot_node_by_id)
                else:
                    pos_after = pos_before
                velocity = self._reflect_velocity_on_correction(core_velocities[idx], pos_before, pos_after)
                displacement = float(np.hypot(pos_after[0] - node.node_location[0], pos_after[1] - node.node_location[1]))
                max_displacement = max(max_displacement, displacement)
                node.node_location = (float(pos_after[0]), float(pos_after[1]))
                node.velocity = (float(velocity[0]), float(velocity[1]))

        movable_plot_nodes = [n for n in self.plot_nodes
                               if n.node_type not in ("map_border_node", "wilderness_node", "city_border_node")]
        if movable_plot_nodes:
            plot_positions = np.array([n.node_location for n in movable_plot_nodes], dtype=float)
            plot_velocities = np.array([n.velocity for n in movable_plot_nodes], dtype=float)

            plot_field = self._sample_field_batch(plot_positions) if self.enable_field_plotnodes else np.zeros_like(plot_positions)
            plot_total_force = (np.array([plot_forces.get(n.node_id, zero2) for n in movable_plot_nodes], dtype=float)
                                 + plot_field)
            plot_force_norms = np.hypot(plot_total_force[:, 0], plot_total_force[:, 1])
            plot_force_scale = np.where(plot_force_norms > self.MAX_SPRING_RESULTANT,
                                         self.MAX_SPRING_RESULTANT / np.maximum(plot_force_norms, 1e-12), 1.0)
            plot_total_force *= plot_force_scale[:, None]

            plot_accel = plot_total_force / self.plot_node_mass
            plot_velocities = (plot_velocities + plot_accel * dt) * self.damping
            plot_speed = np.hypot(plot_velocities[:, 0], plot_velocities[:, 1])
            plot_speed_scale = np.where(plot_speed > max_speed, max_speed / np.maximum(plot_speed, 1e-12), 1.0)
            plot_velocities *= plot_speed_scale[:, None]

            plot_free = plot_positions + plot_velocities * dt
            plot_free[:, 0] = np.clip(plot_free[:, 0], 0.0, self.map_size - 1.0)
            plot_free[:, 1] = np.clip(plot_free[:, 1], 0.0, self.map_size - 1.0)

            for idx, plot_node in enumerate(movable_plot_nodes):
                pos_after = plot_free[idx]
                displacement = float(np.hypot(
                    pos_after[0] - plot_node.node_location[0], pos_after[1] - plot_node.node_location[1]))
                max_displacement = max(max_displacement, displacement)
                plot_node.node_location = (float(pos_after[0]), float(pos_after[1]))
                plot_node.velocity = (float(plot_velocities[idx][0]), float(plot_velocities[idx][1]))

        return max_displacement

    def _point_in_polygon(self, point, polygon, shifted=None):
        x, y = point
        poly = np.asarray(polygon, dtype=float)
        xs, ys = poly[:, 0], poly[:, 1]
        if shifted is None:
            xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
        else:
            xs2, ys2 = shifted
        crosses = (ys > y) != (ys2 > y)
        if not np.any(crosses):
            return False
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
        return bool(np.sum(crosses & (x < x_intersect)) % 2 == 1)

    def _prepare_wilderness_polygons(self):
        prepared = []
        for poly in (self._wilderness_polygons or []):
            coords = np.asarray(poly.exterior.coords, dtype=float)
            xs2 = np.roll(coords[:, 0], -1)
            ys2 = np.roll(coords[:, 1], -1)
            b = np.column_stack((xs2, ys2))
            prepared.append((coords, (xs2, ys2), b))
        return prepared

    def _contain_core_in_cell(self, node, pos, plot_node_by_id):
        plot_node_ids = self._core_cell_plot_node_ids.get(node.node_id)
        if not plot_node_ids:
            return pos
        polygon = [plot_node_by_id[nid].node_location for nid in plot_node_ids if nid in plot_node_by_id]
        if len(polygon) < 3:
            return pos
        polygon = np.array(polygon, dtype=float)
        if self._point_in_polygon(pos, polygon):
            return pos

        projected = self._nearest_point_on_polyline(pos, polygon, closed=True)
        correction = projected - pos
        dist = float(np.hypot(correction[0], correction[1]))
        max_step = self.MAX_DISPLACEMENT_PER_TICK * 3.0
        if dist > max_step and dist > 1e-9:
            projected = pos + correction * (max_step / dist)
        return projected

    def _contain_plot_node(self, plot_node, pos, prepared_wilderness, cap=True):
        if not prepared_wilderness:
            return pos
        type_by_id = self._core_type_by_id
        is_civ_adjacent = any(
            type_by_id.get(cid) == "standard_plot_node" for cid in plot_node.neighbor_core_ids)
        if not is_civ_adjacent:
            return pos

        cache = self._plot_node_wilderness_cache
        max_step = self.plot_base_spacing

        def _capped(candidate):
            if not cap:
                return candidate
            correction = candidate - pos
            dist = float(np.hypot(correction[0], correction[1]))
            if dist > max_step and dist > 1e-9:
                return pos + correction * (max_step / dist)
            return candidate

        cached_idx = cache.get(plot_node.node_id)
        if cached_idx is not None and cached_idx < len(prepared_wilderness):
            coords, shifted, b = prepared_wilderness[cached_idx]
            if self._point_in_polygon(pos, coords, shifted):
                return pos
            return _capped(self._nearest_point_on_segments(pos, coords, b))

        best_idx, best_dist, best_point = None, np.inf, pos
        for idx, (coords, shifted, b) in enumerate(prepared_wilderness):
            if self._point_in_polygon(pos, coords, shifted):
                cache[plot_node.node_id] = idx
                return pos
            candidate = self._nearest_point_on_segments(pos, coords, b)
            dist = float(np.hypot(candidate[0] - pos[0], candidate[1] - pos[1]))
            if dist < best_dist:
                best_dist, best_idx, best_point = dist, idx, candidate
        if best_idx is not None:
            cache[plot_node.node_id] = best_idx
        return _capped(best_point)

    # ==================================================================
    # Potentialfeld (aus tools/biome_lab/field.py)
    # ==================================================================
    def _compute_potential_field(self):
        h, w = self.civ_map.shape
        field_arr = np.zeros((h, w, 2), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]

        gy, gx = np.gradient(self.civ_map)
        grad_norm = np.sqrt(gx ** 2 + gy ** 2)
        GRAD_SATURATION = 0.05
        grad_strength = 1.0 - np.exp(-grad_norm / GRAD_SATURATION)
        mask = grad_norm > 1e-10
        gx_dir = np.zeros_like(gx)
        gy_dir = np.zeros_like(gy)
        gx_dir[mask] = gx[mask] / grad_norm[mask]
        gy_dir[mask] = gy[mask] / grad_norm[mask]
        civ_strength = np.clip(1.0 - self.civ_map, 0.0, 1.0) * 0.5
        field_arr[:, :, 0] += gx_dir * grad_strength * civ_strength
        field_arr[:, :, 1] += gy_dir * grad_strength * civ_strength

        if self.settlements:
            eps = self.SOFTENING
            for settlement in self.settlements:
                sx, sy = settlement.x, settlement.y
                dx = xx - sx
                dy = yy - sy
                dist = np.maximum(np.hypot(dx, dy), eps)
                weight = 140.0 / np.sqrt(dist)
                field_arr[:, :, 0] -= (dx / dist) * weight * self.plot_gravity_strength
                field_arr[:, :, 1] -= (dy / dist) * weight * self.plot_gravity_strength

        hill_fx, hill_fy = self._compute_wilderness_hill_term(xx, yy)
        field_arr[:, :, 0] += hill_fx
        field_arr[:, :, 1] += hill_fy

        PUSH_CAP = 3.0
        wall_spacing = 5.0
        for settlement in self.settlements:
            sx, sy = settlement.x, settlement.y
            dx = xx - sx
            dy = yy - sy
            dist = np.maximum(np.hypot(dx, dy), 1e-6)
            ratio = np.minimum(wall_spacing / dist, 6.0)
            push = np.minimum(self.plot_city_repulsion_strength * ratio ** 3, PUSH_CAP)
            field_arr[:, :, 0] += (dx / dist) * push
            field_arr[:, :, 1] += (dy / dist) * push

        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        dist_out = distance_transform_edt(~civ_mask)
        dist_in = distance_transform_edt(civ_mask)
        signed_dist = np.where(civ_mask, dist_in, -dist_out)
        gy_s, gx_s = np.gradient(gaussian_filter(signed_dist, sigma=3.0))
        norm_s = np.sqrt(gx_s ** 2 + gy_s ** 2)
        mask_s = norm_s > 1e-10
        wgx = np.zeros_like(gx_s)
        wgy = np.zeros_like(gy_s)
        wgx[mask_s] = gx_s[mask_s] / norm_s[mask_s]
        wgy[mask_s] = gy_s[mask_s] / norm_s[mask_s]
        wild_scale = 25.0
        push_strength = np.where(
            signed_dist < 0, 1.0 - np.exp(-np.abs(signed_dist) / wild_scale),
            np.exp(-np.maximum(signed_dist, 0) / wild_scale))
        field_arr[:, :, 0] += wgx * push_strength * 0.6
        field_arr[:, :, 1] += wgy * push_strength * 0.6

        BORDER_MARGIN = 25.0
        BORDER_STRENGTH = 0.4

        def _edge_push(dist_to_edge):
            t = np.clip(1.0 - dist_to_edge / BORDER_MARGIN, 0.0, 1.0)
            return t ** 2 * BORDER_STRENGTH

        field_arr[:, :, 0] += _edge_push(xx.astype(float)) - _edge_push((w - 1 - xx).astype(float))
        field_arr[:, :, 1] += _edge_push(yy.astype(float)) - _edge_push((h - 1 - yy).astype(float))

        field_arr *= self.norm_potential_strength
        self.potential_field = field_arr

    def _compute_wilderness_hill_term(self, xx, yy):
        civ_mask_hill = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        dist_out = distance_transform_edt(~civ_mask_hill)

        HILL_SATURATION_DIST = 40.0
        HILL_MAX_STRENGTH = 0.5
        monotonic_strength = HILL_MAX_STRENGTH * (1.0 - np.exp(-dist_out / HILL_SATURATION_DIST))
        monotonic_strength = np.where(civ_mask_hill, 0.0, monotonic_strength)

        hgy, hgx = np.gradient(gaussian_filter(self.heightmap, sigma=4.0))
        hnorm = np.sqrt(hgx ** 2 + hgy ** 2)
        hmask = hnorm > 1e-10
        hgx_dir = np.zeros_like(hgx)
        hgy_dir = np.zeros_like(hgy)
        hgx_dir[hmask] = hgx[hmask] / hnorm[hmask]
        hgy_dir[hmask] = hgy[hmask] / hnorm[hmask]

        return hgx_dir * monotonic_strength, hgy_dir * monotonic_strength

    def _sample_field_batch(self, positions):
        if self.potential_field is None or len(positions) == 0:
            return np.zeros((len(positions), 2), dtype=float)
        h, w = self.potential_field.shape[:2]
        pos = np.asarray(positions, dtype=float)
        x = np.clip(pos[:, 0], 0.5, w - 1.5)
        y = np.clip(pos[:, 1], 0.5, h - 1.5)
        ix = x.astype(int)
        iy = y.astype(int)
        fx = x - ix
        fy = y - iy
        dx = np.where(ix + 1 < w, 1, 0)
        dy = np.where(iy + 1 < h, 1, 0)
        f00 = self.potential_field[iy, ix]
        f10 = self.potential_field[iy, ix + dx]
        f01 = self.potential_field[iy + dy, ix]
        f11 = self.potential_field[iy + dy, ix + dx]
        w00 = ((1 - fx) * (1 - fy))[:, None]
        w10 = (fx * (1 - fy))[:, None]
        w01 = ((1 - fx) * fy)[:, None]
        w11 = (fx * fy)[:, None]
        return f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11

    # ==================================================================
    # Traffic (aus tools/biome_lab/traffic.py)
    # ==================================================================
    def _rank_distance_weights(self, n):
        """50%, 25%, 12.5%, ... - letzter Rang bekommt den Rest exakt.
        Identisch zu SettlementGenerator/PlotNodeSystem's eigener
        _rank_distance_weights() (core/settlement_generator.py) - hier
        eigenstaendig gehalten statt cross-class aufgerufen, um
        PlotPhysicsSystem unabhaengig von SettlementGenerator-Internas zu
        halten."""
        if n <= 0:
            return []
        if n == 1:
            return [1.0]
        weights = []
        remaining = 1.0
        for _ in range(n - 1):
            w = remaining * 0.5
            weights.append(w)
            remaining -= w
        weights.append(remaining)
        return weights

    def _simulate_traffic(self):
        """1:1 aus tools/biome_lab/traffic.py's _simulate_traffic()."""
        vertex_positions = self._static_vertex_positions
        ridge_edges = self._static_ridge_edges
        num_vertices = self._static_num_vertices
        predecessors = self._static_predecessors
        distances = self._static_distances
        boundary_entries = self._static_boundary_entries
        boundary_settlement = self._static_boundary_settlement
        node_entry = self._static_node_entry
        path_cache = self.path_cache

        if not self.topology_ready or vertex_positions is None or not ridge_edges:
            self.ridge_traffic_history = {}
            self.ridge_traffic_shrink_ema = {}
            return

        def trace_and_add_predecessors(row_index, source_entry, target_entry, amount, contrib):
            cache_key = (row_index, source_entry, target_entry)
            cached_keys = path_cache.get(cache_key)
            if cached_keys is None:
                cached_keys = []
                current = target_entry
                while current != source_entry and current >= 0:
                    prev = predecessors[row_index, current]
                    if prev < 0:
                        break
                    if current < num_vertices and prev < num_vertices:
                        cached_keys.append(self._edge_key(current, prev))
                    current = prev
                path_cache[cache_key] = cached_keys
            for key in cached_keys:
                contrib[key] = contrib.get(key, 0.0) + amount

        city_boundary_indices = {}
        for row, (settlement_id, entry) in enumerate(zip(boundary_settlement, boundary_entries)):
            city_boundary_indices.setdefault(settlement_id, []).append((row, entry))

        fresh_contrib = {}

        for node in self.nodes:
            if node.node_id in self.boundary_owner:
                continue
            entry_idx = node_entry.get(node.node_id)
            if entry_idx is None:
                continue

            node_distances = distances[:, entry_idx]
            per_settlement_best = {}
            for settlement_id, rows in city_boundary_indices.items():
                best = None
                for row, _entry in rows:
                    d = node_distances[row]
                    if not np.isfinite(d):
                        continue
                    if best is None or d < best[0]:
                        best = (d, row)
                if best is not None:
                    per_settlement_best[settlement_id] = best

            if not per_settlement_best:
                continue

            ranked = sorted(per_settlement_best.items(), key=lambda kv: kv[1][0])
            weights = self._rank_distance_weights(len(ranked))
            traffic_weight = float(getattr(node, "traffic_weight", 4.0))

            for rank, (settlement_id, (_dist, row)) in enumerate(ranked):
                amount = weights[rank] * traffic_weight
                trace_and_add_predecessors(row, boundary_entries[row], entry_idx, amount, fresh_contrib)

        settlement_ids = [s.location_id for s in self.settlements]
        if len(settlement_ids) > 1:
            for settlement_id_from in settlement_ids:
                own_rows = [row for row, sid in enumerate(boundary_settlement) if sid == settlement_id_from]
                if not own_rows:
                    continue
                other_best = {}
                for settlement_id_to in settlement_ids:
                    if settlement_id_to == settlement_id_from:
                        continue
                    cols_to = [row for row, sid in enumerate(boundary_settlement) if sid == settlement_id_to]
                    best = None
                    for row_from in own_rows:
                        for col_row in cols_to:
                            target_entry = boundary_entries[col_row]
                            d = distances[row_from, target_entry]
                            if not np.isfinite(d):
                                continue
                            if best is None or d < best[0]:
                                best = (d, row_from, target_entry)
                    if best is not None:
                        other_best[settlement_id_to] = best

                if not other_best:
                    continue
                ranked = sorted(other_best.items(), key=lambda kv: kv[1][0])
                weights = self._rank_distance_weights(len(ranked))
                for rank, (_settlement_id_to, (_d, row_from, target_entry)) in enumerate(ranked):
                    amount = weights[rank] * self.plot_intercity_traffic
                    trace_and_add_predecessors(row_from, boundary_entries[row_from], target_entry, amount, fresh_contrib)

        traffic_decay = 0.15
        keep_factor = 1.0 - traffic_decay
        new_history = {}
        all_keys = set(self.ridge_traffic_history.keys()) | set(fresh_contrib.keys())
        for key in all_keys:
            old_value = self.ridge_traffic_history.get(key, 0.0)
            new_value = old_value * keep_factor + fresh_contrib.get(key, 0.0) * traffic_decay
            if new_value > 1e-6:
                new_history[key] = new_value
        self.ridge_traffic_history = new_history

        shrink_keep_factor = 1.0 - self.spring_shrink_ema_decay
        new_shrink_ema = {}
        for key in all_keys:
            old_shrink = self.ridge_traffic_shrink_ema.get(key, 0.0)
            new_shrink = old_shrink * shrink_keep_factor + fresh_contrib.get(key, 0.0) * self.spring_shrink_ema_decay
            if new_shrink > 1e-6:
                new_shrink_ema[key] = new_shrink
        self.ridge_traffic_shrink_ema = new_shrink_ema

    # ==================================================================
    # NEU: Konvergenz-Loop (ersetzt den Live-QTimer-Tick des Lab, siehe
    # Modul-Docstring - existiert im Original nicht)
    # ==================================================================
    def _run_physics_to_convergence(self):
        """Laesst _physics_step()+_simulate_traffic() headless laufen, bis
        entweder die maximale Node-Verschiebung ueber
        CONVERGENCE_STABLE_TICKS aufeinanderfolgende Ticks unter
        CONVERGENCE_MAX_DISPLACEMENT bleibt, oder MAX_PHYSICS_ITERATIONS
        erreicht ist (Nutzer-Vorgabe: 'run to convergence oder wenn das
        nicht passiert dann einfrieren nach 100 Iterationen'). Ruft bei
        gesetztem progress_callback alle TRAFFIC_RECOMPUTE_INTERVAL
        Iterationen zurueck, damit die GUI den fortschreitenden Zustand
        live anzeigen kann (siehe [[project-settlement-plot-physics-rebuild]] Teil F)."""
        stable_ticks = 0
        for iteration in range(1, self.MAX_PHYSICS_ITERATIONS + 1):
            self.iteration = iteration
            max_displacement = self._physics_step()
            self._sync_core_positions()

            if iteration % self.TRAFFIC_RECOMPUTE_INTERVAL == 0:
                self._simulate_traffic()
                self._report_progress(
                    "plot_physics",
                    45 + int(50 * iteration / self.MAX_PHYSICS_ITERATIONS),
                    f"Physik-Iteration {iteration}/{self.MAX_PHYSICS_ITERATIONS} "
                    f"(max. Verschiebung {max_displacement:.3f}px)")
                self._report_live_state()

            if max_displacement < self.CONVERGENCE_MAX_DISPLACEMENT:
                stable_ticks += 1
                if stable_ticks >= self.CONVERGENCE_STABLE_TICKS:
                    logging.info(
                        f"PlotPhysicsSystem: konvergiert nach {iteration} Iterationen "
                        f"(max. Verschiebung < {self.CONVERGENCE_MAX_DISPLACEMENT}px).")
                    break
            else:
                stable_ticks = 0
        else:
            logging.info(
                f"PlotPhysicsSystem: MAX_PHYSICS_ITERATIONS ({self.MAX_PHYSICS_ITERATIONS}) erreicht, "
                f"eingefroren ohne volle Konvergenz.")

        self._simulate_traffic()  # finale Traffic-Zuweisung mit den konvergierten Positionen
        self._classify_road_tiers()
        self._report_progress("plot_physics", 100, f"Physik abgeschlossen nach {self.iteration} Iterationen")

    # ==================================================================
    # NEU: Strassen-Tier-Klassifikation (aus tools/biome_lab/draw.py in die
    # Generator-Logik verschoben, siehe Modul-Docstring)
    # ==================================================================
    def _classify_road_tiers(self):
        """Baut aus ridge_traffic_history + _static_ridge_edges die finalen
        PlotEdge-Objekte mit Traffic-Wert und Tier-Klassifikation
        ("none"/"path"/"road" - Strasse/Weg werden beide als "road"
        klassifiziert, Pfad als "path", analog zur bestehenden Produktions-
        Konvention in PlotEdge.classification; die feinere Strasse/Weg/Pfad-
        Unterscheidung bleibt zusaetzlich in PlotEdge.properties verfuegbar).
        Wird von SettlementGenerator._calc_plot_nodes() gelesen, um
        SettlementData.plot_edges zu befuellen."""

        edges = {}
        edge_id = 0
        for i, j, p1, p2, cost in self._static_ridge_edges:
            pid_i = self.vertex_to_plot_node.get(i)
            pid_j = self.vertex_to_plot_node.get(j)
            if pid_i is None or pid_j is None or pid_i == pid_j:
                continue
            key = self._edge_key(i, j)
            traffic = self.ridge_traffic_history.get(key, 0.0)

            if traffic >= self.TIER_STRASSE_THRESHOLD:
                tier, classification = "strasse", "road"
            elif traffic >= self.TIER_WEG_THRESHOLD:
                tier, classification = "weg", "road"
            elif traffic >= self.TIER_MIN_TRAFFIC:
                tier, classification = "pfad", "path"
            else:
                tier, classification = "none", "none"

            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            edges[edge_id] = PlotEdge(
                edge_id=edge_id, node_a=pid_i, node_b=pid_j, length=length,
                height_cost=float(cost - length), movement_cost=float(cost),
                traffic=float(traffic), classification=classification)
            edge_id += 1

        self.plot_edges = edges

    def build_plot_map(self):
        """
        Baut eine (H,W) int32 plot_map: jedes Pixel bekommt die node_id des
        naechstgelegenen Plotkerns (self.nodes - standard_plot_node/
        wilderness_core/city_core), per cKDTree-Nearest-Neighbor ueber die
        ganze Karte. Ersetzt die fruehere, sehr sparse Plot-Map des alten
        PlotNodeSystem (_create_plot_map malte nur einzelne Pixel an den
        PlotNode-Positionen selbst, keine Flaechenfuellung) - da jeder
        Plotkern hier per Konstruktion bereits eine eigene Voronoi-Zelle
        besitzt, ist eine flaechendeckende Nearest-Core-Zuordnung die
        naturgemaesse, korrekte Entsprechung."""
        if not self.nodes:
            return np.full((self.map_size, self.map_size), -1, dtype=np.int32)

        core_positions = np.array([n.node_location for n in self.nodes], dtype=float)
        core_ids = np.array([n.node_id for n in self.nodes], dtype=np.int32)
        tree = cKDTree(core_positions)

        yy, xx = np.mgrid[0:self.map_size, 0:self.map_size]
        query_points = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
        _dist, nearest_idx = tree.query(query_points)
        plot_map = core_ids[nearest_idx].reshape(self.map_size, self.map_size)
        return plot_map


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

        # Live-Fortschritts-Callback nur für settlement.plot_nodes (siehe
        # [[project-settlement-plot-physics-rebuild]] Teil F) - wird von
        # CalculatorThread._emit_live_plot_update() gesetzt, solange dieser
        # eine Calculator-Knoten läuft, danach wieder auf None zurückgesetzt.
        # None (Default/Standalone/Tests) bedeutet: PlotPhysicsSystem meldet
        # keine Zwischenzustände, läuft aber sonst identisch durch.
        self.live_plot_callback = None

        # Standard-Parameter (werden durch _load_default_parameters überschrieben)
        self.settlements = 3
        self.landmarks = 3
        self.roadsites = 3
        self.plotnodes = 1000
        self.civ_influence_decay = 0.8
        self.terrain_factor_villages = 1.0
        self.road_slope_to_distance_ratio = 1.5
        self.landmark_wilderness = 0.3
        self.city_reach_factor = 4.0
        self.civ_influence_range = 0.30
        self.plot_base_spacing = 10.0
        self.plot_civ_spacing_factor = 3.0
        self.plot_height_cost_factor = 2.0

    def set_active_parameters(self, parameters):
        """
        Setzt die Parameter, die alle calculate_*()/_calc_*-Methoden bis zur
        nächsten frischen Anfrage verwenden (vom GenerationOrchestrator
        aufgerufen). Settlement speichert Parameter als Instanz-Attribute
        (self.settlements etc.), nicht als eigenes dict - entspricht dem, was
        _execute_generation() vorher direkt inline gemacht hat.
        Mit Defaults gemergt (analog zu core/geology_generator.py) - die GUI
        (settlement_tab.py) exponiert nicht jeden hier gelesenen Schlüssel als
        Slider (z.B. city_reach_factor/civ_influence_range), ein reines
        parameters['key'] würde bei jedem GUI-getriggerten Request mit
        KeyError abbrechen.
        """
        parameters = {**self._load_default_parameters(), **parameters}
        self.settlements = parameters['settlements']
        self.landmarks = parameters['landmarks']
        self.roadsites = parameters['roadsites']
        self.plotnodes = parameters['plotnodes']
        self.civ_influence_decay = parameters['civ_influence_decay']
        self.terrain_factor_villages = parameters['terrain_factor_villages']
        self.road_slope_to_distance_ratio = parameters['road_slope_to_distance_ratio']
        self.landmark_wilderness = parameters['landmark_wilderness']
        self.city_reach_factor = parameters['city_reach_factor']
        self.civ_influence_range = parameters['civ_influence_range']
        self.plot_base_spacing = parameters['plot_base_spacing']
        self.plot_civ_spacing_factor = parameters['plot_civ_spacing_factor']
        self.plot_height_cost_factor = parameters['plot_height_cost_factor']

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
            'city_reach_factor': SETTLEMENT.CITY_REACH_FACTOR["default"],
            'civ_influence_range': SETTLEMENT.CIV_INFLUENCE_RANGE["default"],
            'plot_base_spacing': SETTLEMENT.PLOT_BASE_SPACING["default"],
            'plot_civ_spacing_factor': SETTLEMENT.PLOT_CIV_SPACING_FACTOR["default"],
            'plot_height_cost_factor': SETTLEMENT.PLOT_HEIGHT_COST_FACTOR["default"]
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

    def _calc_pathfinding(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'settlement.pathfinding' (#30) - liefert die frühe,
        einfache Zwischen-Siedlungs-Bootstrap-Route, die civ_influence als
        Einfluss-Quelle braucht (siehe _calc_civ_influence). Das sichtbare,
        game-relevante Straßennetz kommt seit [[project-settlement-plot-physics-rebuild]]
        aus PlotPhysicsSystem (settlement.plot_nodes, siehe _calc_plot_nodes) -
        dieser frühe Bootstrap-Pfad ist bewusst NICHT ersetzt, weil
        PlotPhysicsSystem selbst civ_map als Eingabe braucht und daher
        zwangsläufig NACH civ_influence laufen muss (Zirkelbezug sonst).
        voronoi_cell_map (früher aus settlement.landscape_voronoi, jetzt
        entfernt) entfällt ersatzlos - calculate_road_network() fällt dafür
        bereits dokumentiert auf reines Slope-Cost-Pathfinding zurück.
        """
        self._update_progress("Road Building", 25, "Creating road networks between settlements...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        if settlement_list is None:
            raise ValueError(f"settlement.pathfinding: settlement_list für LOD {lod_level} nicht verfügbar")

        roads = self.calculate_road_network(settlement_list, inputs["slopemap"], lod_level, None)
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
        Calculator-Node 'settlement.plot_nodes' (#34) - nutzt PlotPhysicsSystem
        (siehe [[project-settlement-plot-physics-rebuild]]) statt des früheren
        Delaunay-basierten PlotNodeSystem. Läuft NUR am finalen LOD (Nutzer-
        Vorgabe: keine Zwischen-LOD-Berechnung mehr für Plots/Städte - erst am
        Ende, wenn alles andere fertig ist, entsteht das Wege-/Plot-Netz in
        einem einzigen, bis zur Konvergenz laufenden Durchlauf). An allen
        Zwischen-LODs wird lediglich ein leeres Platzhalter-Ergebnis
        geschrieben, damit nichts (z.B. eine GUI-Statusabfrage) auf fehlende
        Daten trifft - kein anderer Calculator-Knoten hängt von
        settlement.plot_nodes ab (siehe calculator_graph.py), es entsteht also
        keine echte Wartezeit für irgendetwas anderes.
        """
        is_final_lod = lod_level >= self.data_lod_manager.get_max_lod_for_map_size()
        if not is_final_lod:
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
                "plot_nodes": [], "plots": [], "plot_map": None, "plot_edges": {},
                "plot_node_positions": [], "plot_cores": [], "wilderness_polygons": [],
            })
            return

        self._update_progress("Plot Generation", 75, "Generating plot system (physics)...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        civ_map = self.data_lod_manager.get_calculator_output("settlement.civ_influence", "civ_map", lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        city_mask = self.data_lod_manager.get_calculator_output("settlement.city_boundary", "city_mask", lod_level)
        if civ_map is None or settlement_list is None or city_mask is None:
            raise ValueError(f"settlement.plot_nodes: fehlende Inputs für LOD {lod_level}")

        height, width = inputs["heightmap"].shape
        plot_system = PlotPhysicsSystem(
            map_size=height, plot_nodes_count=self.plotnodes, plot_base_spacing=self.plot_base_spacing,
            plot_civ_spacing_factor=self.plot_civ_spacing_factor,
            plot_height_cost_factor=self.plot_height_cost_factor, shader_manager=self.shader_manager,
            progress_callback=self._update_progress, map_seed=self.map_seed,
            live_state_callback=self.live_plot_callback)
        ok = plot_system.generate(inputs["heightmap"], inputs["slopemap"], civ_map, city_mask, settlement_list)

        if not ok:
            self.logger.warning(f"settlement.plot_nodes: PlotPhysicsSystem.generate() fehlgeschlagen für LOD {lod_level}")
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
                "plot_nodes": [], "plots": [], "plot_map": None, "plot_edges": {},
                "plot_node_positions": [], "plot_cores": [], "wilderness_polygons": [],
            })
            return

        plot_map = plot_system.build_plot_map()
        relative_positions = [
            (x / max(1, width - 1), y / max(1, height - 1)) for x, y in
            (node.node_location for node in plot_system.plot_nodes)
        ]
        # wilderness_polygons als reine (N,2)-Koordinatenarrays statt Shapely-
        # Polygon-Objekte exportiert - Konsumenten (map_display_2d.py's
        # overlay_plot_boundaries(), siehe [[project-settlement-plot-physics-rebuild]]
        # Teil 3) brauchen kein shapely, nur die Aussenkontur-Punkte.
        wilderness_polygons = [
            np.asarray(poly.exterior.coords, dtype=float) for poly in plot_system._wilderness_polygons
        ]
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level,
            {"plot_nodes": plot_system.plot_nodes, "plots": [], "plot_map": plot_map,
             "plot_edges": plot_system.plot_edges, "plot_node_positions": relative_positions,
             "plot_cores": plot_system.nodes, "wilderness_polygons": wilderness_polygons})

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
            edge_distance_map=edge_distance_map)
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
        parameters = self._load_default_parameters()
        parameters.update({
            'settlements': settlements_count,
            'landmarks': self.landmarks,
            'roadsites': self.roadsites,
            'plotnodes': self.plotnodes,
            'civ_influence_decay': self.civ_influence_decay,
            'terrain_factor_villages': terrain_factor_villages,
            'road_slope_to_distance_ratio': self.road_slope_to_distance_ratio,
            'landmark_wilderness': self.landmark_wilderness
        })

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

    def create_civilization_map(self, heightmap, slopemap, settlements, roads, landmarks, roadsites, civ_influence_decay):
        """
        Funktionsweise: Legacy-Methode für Civilization-Map-Erstellung
        """
        self.civ_influence_decay = civ_influence_decay
        return self.calculate_civilization_mapping(heightmap, slopemap, settlements, roads, roadsites)

    def generate_complete_settlements(self, heightmap, slopemap, water_map, map_seed, settlements,
                                      landmarks, roadsites, plotnodes, civ_influence_decay, terrain_factor_villages,
                                      road_slope_to_distance_ratio, landmark_wilderness):
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
            'landmark_wilderness': landmark_wilderness
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
                'landmark_wilderness': self.landmark_wilderness
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

