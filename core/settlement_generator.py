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

from core.wegsuche_schnell import NUMBA_DA as _NUMBA_WEGSUCHE_DA

# GEWICHTETE HEURISTIK (Punkt 2.4 der Leistungsliste, docs/PERFORMANCE_2026-08-23.md).
#
# f = g + w*h. Mit w > 1 wird der Suchbaum schmaler; der gefundene Weg ist
# dafuer hoechstens w-mal teurer als der optimale - das ist eine Schranke,
# kein Erfahrungswert, und smoke_test_wegsuche_schnell prueft sie.
#
# STEHT BEWUSST AUF 1.0. Gemessen am 2026-08-23 auf echtem Gelaende brachte
# w = 1.2 und w = 1.5 zwar 1.1x bzw. 1.7x weniger Rechenzeit, aber der
# numba-Kern ist mit w = 1.0 bereits bei 0.007 s je Route - der Gewinn ist
# absolut bedeutungslos, und die Optimalitaet des Weges ist es nicht: eine
# Handelsstrasse, die 20 % teurer verlaeuft als noetig, widerspricht dem
# Bereitschaftstest in calculate_road_network(), der Wegkosten mit
# Bereitschaft VERGLEICHT. Der Regler bleibt fuer den Fall, dass sehr viel
# groessere Karten kommen.
WEGSUCHE_H_GEWICHT = 1.0
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
        self.roads = []  # List[List[Tuple]] - Alle Road-Pfade (Land)
        self.sea_roads = []  # List[List[Tuple]] - Seewege, docs/SIEDLUNGEN_ENTWURF.md §4.4
        self.city_mask = None  # (height, width) - Settlement-ID pro Pixel, -1 = ausserhalb jeder Stadt
        self.voronoi_cell_map = None  # (height, width) - Landschafts-Plot-Zell-ID pro Pixel, -1 = Stadt/Wilderness
        self.street_mask = None  # (height, width) bool - innerstaedtisches Strassenraster
        self.house_parcel_map = None  # (height, width) - kartenweit eindeutige Hausparzellen-ID, -1 = keine Parzelle
        self.landmark_roads = []  # List[List[Tuple]] - Landmark-Anbindungen ans Strassennetz
        self.plot_edges = {}  # Dict[int, PlotEdge] - adressierbares Kanten-Registry mit Traffic/Klassifikation
        self.potential_field = None  # (height, width, 2) - PlotPhysicsSystem-Kraftfeld, siehe [[project-settlement-physics-lab-parity]]
        # Vorher NIE über set_settlement_data_complete_lod() dekomponiert
        # worden, obwohl settlement_tab.py sie schon länger über
        # get_settlement_data() abfragte - dadurch blieben Plotkerne/
        # Wildnisgrenzen/PlotNode-Positionen in der 2D/3D-Anzeige unsichtbar
        # (Nutzer-Report "dann sehe ich keine plotkerne, keine plotnodes,
        # nichts"), siehe [[project-settlement-physics-lab-parity]].
        self.plot_cores = []  # List[PlotNode] - node_type in {standard_plot_node, wilderness_core, city_core}
        self.wilderness_polygons = []  # List[(N,2) array] - Aussenkontur-Punkte je Wildnisgebiet
        self.plot_node_positions = []  # List[(x_norm, y_norm)] - PlotNode-Positionen, normiert auf [0,1]

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


# Drei Raenge, alle klein (docs/SIEDLUNGEN_ENTWURF.md §1) - Unterscheidung ist
# eine der Rang-Spanne, nicht Stadt gegen Metropole.
RANG_HAEUSER = {"dorf": (15, 25), "siedlung": (25, 35), "stadt": (35, 50)}
RANG_REIHENFOLGE = ("dorf", "siedlung", "stadt")


# =============================================================================
# KULTURKATALOGE - docs/KULTUREN_UND_ORTE.md, Abschnitt "AUSWAHL DES NUTZERS"
# =============================================================================
#
# Der verbindliche Satz: 5 Roadsite- und 5 Landmark-Arten je der neun Kulturen
# (45 + 45). Kulturnamen sind exakt die `volk`-Werte aus
# core/terrain_weltkarte.py REGIONEN.
#
# JEDE ART TRAEGT EINE PLATZIERUNGSVORLIEBE (docs/SIEDLUNGEN_ENTWURF.md §4.6
# fuer Roadsites, §4.7 fuer Landmarks nennt nur die KATEGORIEN, nicht die
# Zuordnung je Art - die folgt hier aus dem Namen selbst: "Furtstein an der
# Flussquerung" will an eine Furt, "Warte auf dem Kamm" auf einen Passpunkt,
# "Osteria an der Kreuzung" ausdruecklich an eine Kreuzung; was keiner
# Kategorie eindeutig zuzuordnen ist, gilt als "strecke" - der allgemeine
# Zwischenstueck-Fall, den es nach dem Entwurf ebenfalls gibt ("auf langen
# Zwischenstuecken ohne Ort, damit eine Tagesreise einen Halt hat").
#
# ROADSITE-KATEGORIEN: furt (Furt/Bruecke/Faehre), pass (Passhoehe/Kamm/
# Bergsattel), kreuzung (Wegscheide/Marktflecken/Grenzuebergang), strecke
# (alles andere - der Rest eines langen Weges).
ROADSITE_KATALOG = {
    "Kelten": [
        ("Furtstein an der Flussquerung", "furt"),
        ("Rastkreuz an der Wegscheide", "kreuzung"),
        ("Bardenlager", "strecke"),
        ("Pilgerherberge", "strecke"),
        ("Zollringwall", "kreuzung"),
    ],
    "Wikinger": [
        ("Faehrstelle ueber den Fjord", "furt"),
        ("Sennhuette", "pass"),
        ("Handelsplatz am Strand", "strecke"),
        ("Kohlenmeiler", "strecke"),
        ("Salzsiederei", "strecke"),
    ],
    "Slawen": [
        ("Bohlenweg durchs Sumpfland", "strecke"),
        ("Pelzhaendlerlager", "strecke"),
        ("Blockhausherberge", "strecke"),
        ("Grenzverhau aus Staemmen", "kreuzung"),
        ("Faehre am Strom", "furt"),
    ],
    "Franken": [
        ("Zollbruecke", "furt"),
        ("Wechselstall fuer Pferde", "strecke"),
        ("Fischerweiler", "strecke"),
        ("Weinschenke", "strecke"),
        ("Muehlenwehr", "furt"),
    ],
    "Alemannen": [
        ("Passhospiz", "pass"),
        ("Wechselstall fuer Saumtiere", "pass"),
        ("Klause mit Wegzoll", "kreuzung"),
        ("Holzriese", "strecke"),
        ("Kaesespeicher", "strecke"),
    ],
    "Sachsen": [
        ("Warte auf dem Kamm", "pass"),
        ("Gerichtslinde", "kreuzung"),
        ("Wuestung", "strecke"),
        ("Kruggasthof", "strecke"),
        ("Kalkofen", "strecke"),
    ],
    "Andalusier": [
        ("Funduq — Karawanserei", "strecke"),
        ("Aljibe — Zisterne am Weg", "strecke"),
        ("Canada — Herdenweg", "strecke"),
        ("Zoco — Marktflecken", "kreuzung"),
        ("Oelmuehle", "strecke"),
    ],
    "Italiener": [
        ("Via-Rest mit Meilenstein", "strecke"),
        ("Fischtrockenplatz", "strecke"),
        ("Weinpresse", "strecke"),
        ("Rastplatz auf dem Bergsattel", "pass"),
        ("Osteria an der Kreuzung", "kreuzung"),
    ],
    "Byzantiner": [
        ("Skala — Anlegebucht", "furt"),
        ("Zisternenhof", "strecke"),
        ("Schwammtaucherlager", "strecke"),
        ("Eselspfad mit Stuetzmauern", "pass"),
        ("Xenodocheion", "strecke"),
    ],
}

# LANDMARK-KATEGORIEN: gipfel (Kuppe/Fels/Hoehe), kueste (Kliff/Kueste/
# Riff/Insel), quelle (Wasser als Heiligtum: Quelle, Fluss, Bruecke),
# abgelegen (der Rest - "duerfen ausdruecklich weitab jedes Weges liegen").
LANDMARK_KATALOG = {
    "Kelten": [
        ("Steinkreis auf der Kuppe", "gipfel"),
        ("Heilige Quelle mit Opfergaben", "quelle"),
        ("Ogham-Stein als Grenzmal", "abgelegen"),
        ("Ganggrab", "abgelegen"),
        ("Bienenkorbzellen am Kliff", "kueste"),
    ],
    "Wikinger": [
        ("Runenstein", "abgelegen"),
        ("Langhaus des Jarls", "abgelegen"),
        ("Hoergr — Steinaltar auf der Hoehe", "gipfel"),
        ("Thingplatz", "abgelegen"),
        ("Gestrandetes Langschiff", "kueste"),
    ],
    "Slawen": [
        ("Gorod — Ringwallburg", "abgelegen"),
        ("Heiliger Hain", "abgelegen"),
        ("Wehrturm aus Blockholz", "abgelegen"),
        ("Verlassene Brandrodung", "abgelegen"),
        ("Baerenhoehle mit Opferstelle", "abgelegen"),
    ],
    "Franken": [
        ("Steinerne Abtei", "abgelegen"),
        ("Rest einer Koenigspfalz", "abgelegen"),
        ("Salzgarten am Aestuar", "kueste"),
        ("Kliffkapelle", "kueste"),
        ("Aquaeduktstueck der Roemer", "abgelegen"),
    ],
    "Alemannen": [
        ("Bergkloster auf dem Sattel", "gipfel"),
        ("Trutzburg auf dem Felskopf", "gipfel"),
        ("Gletscherzunge", "gipfel"),
        ("Eisenerzgrube", "abgelegen"),
        ("Wildheu-Alm", "gipfel"),
    ],
    "Sachsen": [
        ("Stumpf der gefaellten Irminsul", "abgelegen"),
        ("Missionskirche aus Bruchstein", "abgelegen"),
        ("Silbergrube", "abgelegen"),
        ("Alte Landwehr", "abgelegen"),
        ("Opferstein im Buchenwald", "abgelegen"),
    ],
    "Andalusier": [
        ("Hisn — Felsenburg", "gipfel"),
        ("Alcazaba-Ruine", "gipfel"),
        ("Noria — Schoepfrad am Fluss", "quelle"),
        ("Atalaya — Signalturm", "gipfel"),
        ("Nekropole am Wadi", "abgelegen"),
    ],
    "Italiener": [
        ("Roemische Bruecke", "quelle"),
        ("Bergdorfkastell", "gipfel"),
        ("Basilika mit Campanile", "abgelegen"),
        ("Terrassierte Olivenhaenge", "abgelegen"),
        ("Schwefelquelle", "quelle"),
    ],
    "Byzantiner": [
        ("Kastro — Inselfestung", "kueste"),
        ("Klippenkloster", "kueste"),
        ("Antiker Tempel als Steinbruch", "abgelegen"),
        ("Antikes Amphitheater", "abgelegen"),
        ("Schiffswrackriff", "kueste"),
    ],
}


def _naechste_kultur(x, y, settlements):
    """Kultur der naechstgelegenen Siedlung - Grundlage fuer die Typwahl von
    Roadsites/Landmarks, die selbst keine eigene Kulturzuordnung tragen."""
    orte = [s for s in settlements if s.location_type == 'settlement' and s.culture]
    if not orte:
        return None
    abstaende = [(s.x - x) ** 2 + (s.y - y) ** 2 for s in orte]
    return orte[int(np.argmin(abstaende))].culture


@dataclass
class Location:
    """
    Funktionsweise: Datenstruktur für alle Arten von Locations (Settlements, Landmarks, Roadsites)
    Aufgabe: Einheitliche Repräsentation aller Siedlungs-Objekte

    `culture`/`rank`/`house_count` (2026-08-10, docs/SIEDLUNGEN_ENTWURF.md §1+3):
    nur fuer location_type == 'settlement' belegt. `culture` ist der Name aus
    core.terrain_weltkarte (`volk`-Feld je Region), `rank` einer von
    'dorf'/'siedlung'/'stadt' (15-25/25-35/35-50 Haeuser), `house_count` die
    konkrete gezogene Zahl. Roadsites/Landmarks tragen ihre eigene Kultur ueber
    `properties['culture']` statt eines eigenen Feldes - sie sind zahlreicher
    und kuerzerlebig in der Auswertung.
    """
    location_id: int
    x: float
    y: float
    location_type: str  # 'settlement', 'landmark', 'roadsite'
    radius: float
    civ_influence: float
    properties: Dict = None
    culture: str = ""
    rank: str = ""
    house_count: int = 0
    # Stadttyp (2026-08-13, docs/OFFENE_PUNKTE.md 5.16): "bergdorf",
    # "marktstadt", "agrarstadt" oder "sonstige" - nur fuer
    # location_type == 'settlement' belegt. Bestimmt die erlaubte Groesse
    # (STADTTYPEN[...]["rang_erlaubt"]) und das Handelsinteresse gegenueber
    # anderen Orten (handelsgewicht()).
    settlement_type: str = "sonstige"


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
    # Lauf-Durchschnitt (nicht die abklingende EMA von `traffic`) ueber die
    # gesamte Konvergenz-Simulation - treibt die Traffic-Gradient-
    # Strassenfaerbung (hell-orange -> dunkelrot), siehe
    # PlotPhysicsSystem._classify_road_tiers()/_simulate_traffic() und
    # [[project-settlement-physics-lab-parity]]. Bewusst ein eigenes Feld
    # statt `traffic` zu ueberladen, damit die Tier-Klassifikation (die
    # weiterhin die abklingende EMA nutzt) unveraendert bleibt.
    traffic_avg: float = 0.0


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


# Referenzhoehe der Hoehen-Daempfung (Meter). Ab hier ist der Standortwert auf
# die Haelfte gefallen; oberhalb von rund 1800 m ist praktisch nichts mehr
# uebrig. Frei gewaehlt, kein Regler (docs/SIEDLUNGEN_ENTWURF.md nennt keinen) -
# passt zu den Regionshoehen der Weltkarte (Nevadin reicht nach der
# Neueichung vom 2026-08-10 bis rund 1000 m ueber die Reliefspanne).
ELEVATION_DAEMPFUNG_M = 600.0

# Gewicht je Wassertyp fuer den staerksten Einzelfaktor ("Wasser am Ort").
# Grossfluss/See sind der beste Standort (Muehlen, Transport in Masse),
# Meereskueste vergleichbar einer Flussmuendung, der Bach am schwaechsten -
# er traegt keine Lastschiffe. Werte aus water.manning_flow's water_biomes_map:
# 0=kein Wasser, 1=Bach, 2=Fluss, 3=Grossfluss, 4=See.
WASSERTYP_GEWICHT = {4: 1.0, 3: 1.0, 2: 0.8, 1: 0.5}
KUeSTE_GEWICHT = 0.75


class TerrainSuitabilityAnalyzer:
    """
    Eignungsfeld nach docs/SIEDLUNGEN_ENTWURF.md Abschnitt 2 - fuenf Faktoren:

        Wasser am Ort         staerkster Einzelfaktor (Wassertyp x Naehe)
        Ebener Grund          stark
        Ackerland im Umkreis  stark (traegt die GROESSE, nicht die Lage)
        Hoehenlage             daempfend (multiplikativ)
        Erreichbarkeit          daempfend (multiplikativ, siehe unten)

    ERREICHBARKEIT IST EIN RUECKSCHRITT, KEIN FUENFTER TERM VON ANFANG AN.
    Das Wegenetz haengt an den Siedlungen, die Erreichbarkeit haengt am
    Wegenetz - die Spezifikation loest das in EINEM Rueckschritt: die erste
    Platzierung laeuft ohne Erreichbarkeit (Faktor neutral 1.0), nach dem
    Netzbau wird der Rang einmal nachkorrigiert, ohne Orte zu verschieben
    (Abschnitt 5 des Entwurfs). `create_combined_suitability` nimmt deshalb
    ein optionales `reachability_map` entgegen statt es selbst zu berechnen -
    der Aufrufer entscheidet, ob gerade die erste oder die nachkorrigierte
    Runde laeuft.

    VOLLSTAENDIG VEKTORISIERT (2026-08-10). Die Vorlage hatte fuer Slope und
    Hoehe je eine Python-Doppelschleife ueber jeden Pixel einzeln - bei 512 px
    ueber eine Viertelmillion Iterationen fuer zwei Faktoren, die sich beide
    als einfache Feldformel schreiben lassen.
    """

    def __init__(self, terrain_factor_villages=1.0, map_size=64):
        self.terrain_factor = terrain_factor_villages
        self.map_size = map_size

        # Groessenabhaengige maximale Wasser-Suchdistanz (Pixel) - map_size ist
        # die tatsaechliche Pixel-Aufloesung der uebergebenen Arrays.
        if map_size <= 64:
            self.max_distance_check = 20
        elif map_size <= 128:
            self.max_distance_check = 30
        elif map_size <= 256:
            self.max_distance_check = 40
        else:
            self.max_distance_check = 50

        # Radius des "Umkreis" fuer den Ackerland-Faktor, in Pixeln - grob eine
        # Tagesreise zu Fuss um die Felder, skaliert mit der Aufloesung wie die
        # Wasser-Suchdistanz.
        self.farmland_radius_px = max(3, int(round(self.max_distance_check * 0.6)))

    def analyze_slope_suitability(self, slopemap, progress_callback=None):
        """Ebener Grund: 1.0 unter 0.1 Steigung, linear auf 0 bei 1.0."""
        if progress_callback:
            progress_callback("Terrain Analysis", 5, "Analyzing slope suitability...")

        dz_dx = slopemap[:, :, 0].astype(np.float64)
        dz_dy = slopemap[:, :, 1].astype(np.float64)
        hang = np.hypot(dz_dx, dz_dy)

        eignung = np.select(
            [hang < 0.1, hang < 0.5, hang < 1.0],
            [np.ones_like(hang),
             1.0 - (hang - 0.1) / 0.4 * 0.5,
             0.5 - (hang - 0.5) / 0.5 * 0.5],
            default=0.0)
        return np.clip(eignung, 0.0, 1.0).astype(np.float32)

    def calculate_water_proximity(self, water_map, heightmap=None, progress_callback=None):
        """
        Wasser am Ort: je Wassertyp eine eigene Distanzkarte, das Maximum aus
        Typgewicht x Naehe-Abklingen gewinnt. Ein Dorf direkt an einem
        Grossfluss zaehlt damit hoeher als eines gleich nah an einem Bach -
        das war der Vorlage nicht bekannt, die jedes `water_map > 0`-Pixel
        gleich behandelte.
        """
        if progress_callback:
            progress_callback("Terrain Analysis", 10, "Calculating water proximity...")

        height, width = water_map.shape
        bestwert = np.zeros((height, width), dtype=np.float64)

        def abklingen(distanz):
            # Gleiche Kurve wie die Vorlage: 0 direkt am Wasser (kein
            # Ueberschwemmungsrisiko), Optimum 2-10 Pixel, Ausklingen bis 30.
            w = np.zeros_like(distanz)
            nah = distanz < 2
            w[nah] = distanz[nah] / 2.0
            w[(distanz >= 2) & (distanz <= 10)] = 1.0
            gut = (distanz > 10) & (distanz <= 20)
            w[gut] = 1.0 - (distanz[gut] - 10) / 10 * 0.5
            ok = (distanz > 20) & (distanz <= self.max_distance_check)
            w[ok] = np.maximum(0.0, 0.5 - (distanz[ok] - 20) / 10 * 0.5)
            return w

        for typ, gewicht in WASSERTYP_GEWICHT.items():
            maske = water_map == typ
            if not np.any(maske):
                continue
            distanz = distance_transform_edt(~maske)
            bestwert = np.maximum(bestwert, gewicht * abklingen(distanz))

        if heightmap is not None:
            kueste = heightmap <= 0.0
            if np.any(kueste) and not np.all(kueste):
                distanz = distance_transform_edt(~kueste)
                bestwert = np.maximum(bestwert, KUeSTE_GEWICHT * abklingen(distanz))

        return np.clip(bestwert, 0.0, 1.0).astype(np.float32)

    def evaluate_elevation_fitness(self, heightmap, progress_callback=None):
        """
        Hoehenlage, DAEMPFEND statt einer Wohlfuehlzone in der Mitte der
        Hoehenspanne - die Vorlage bevorzugte 20-60% der lokalen Hoehenspanne
        UNABHAENGIG von der absoluten Hoehe; auf einer flachen Karte war damit
        auch der hoechste Punkt "optimal". Jetzt eine feste, physikalisch
        gemeinte Kurve: je hoeher ueber dem Meer, desto kuerzer die
        Wachstumszeit, desto weniger Ertrag (docs/SIEDLUNGEN_ENTWURF.md §2).
        """
        if progress_callback:
            progress_callback("Terrain Analysis", 15, "Evaluating elevation fitness...")

        ueber_null = np.maximum(heightmap.astype(np.float64), 0.0)
        return (1.0 / (1.0 + (ueber_null / ELEVATION_DAEMPFUNG_M) ** 2)).astype(np.float32)

    def evaluate_farmland_radius(self, flat_suit, elevation_suit, land_mask, progress_callback=None):
        """
        Ackerland im Umkreis: Anteil an flacher, tiefer Flaeche im Radius um
        jeden Punkt - bestimmt, WIEVIELE Menschen der Ort ernaehren kann,
        waehrend Wasser die LAGE traegt (docs/SIEDLUNGEN_ENTWURF.md §2).

        Ein Boxfilter statt eines echten Kreises: bei den hier ueblichen
        Radien (wenige Pixel) ist der Unterschied zur Kreisscheibe gering,
        `uniform_filter` ist separierbar und braucht O(1) je Pixel statt
        O(Radius^2).
        """
        if progress_callback:
            progress_callback("Terrain Analysis", 12, "Evaluating farmland radius...")

        from scipy.ndimage import uniform_filter

        ackerland = (flat_suit >= 0.6) & (elevation_suit >= 0.4) & land_mask
        anteil = uniform_filter(ackerland.astype(np.float64),
                                size=2 * self.farmland_radius_px + 1, mode="nearest")
        return anteil.astype(np.float32)

    def create_combined_suitability(self, heightmap, slopemap, water_map,
                                    reachability_map=None, progress_callback=None):
        """
        Fuenf Faktoren zur Standortguete. Wasser/Ebene/Ackerland gehen additiv
        gewichtet ein (Wasser am staerksten, die Begruendung siehe
        WASSERTYP_GEWICHT), Hoehe und Erreichbarkeit wirken DAEMPFEND -
        multiplikativ auf das Ergebnis, nicht als weiterer additiver Term -
        weil sie im Entwurf ausdruecklich als daempfende Faktoren beschrieben
        sind, nicht als weitere Qualitaeten, die sich aufaddieren.
        """
        land_mask = heightmap > 0.0
        wasser_suit = self.calculate_water_proximity(water_map, heightmap, progress_callback)
        flach_suit = self.analyze_slope_suitability(slopemap, progress_callback)
        hoehe_suit = self.evaluate_elevation_fitness(heightmap, progress_callback)
        acker_suit = self.evaluate_farmland_radius(flach_suit, hoehe_suit, land_mask,
                                                   progress_callback)

        gewichte = {'wasser': 0.45, 'flach': 0.30 * self.terrain_factor,
                   'acker': 0.25 * self.terrain_factor}
        summe_gewichte = sum(gewichte.values()) or 1.0
        lage_guete = (wasser_suit * gewichte['wasser']
                     + flach_suit * gewichte['flach']
                     + acker_suit * gewichte['acker']) / summe_gewichte

        daempfung = hoehe_suit
        if reachability_map is not None:
            daempfung = daempfung * np.clip(reachability_map, 0.0, 1.0)

        combined = np.where(land_mask, lage_guete * daempfung, 0.0)
        return combined.astype(np.float32)

    def stadttyp_eignungen(self, heightmap, slopemap, water_map,
                            progress_callback=None):
        """
        Eine Eignungskarte je Stadttyp (Nutzer-Vorgabe 2026-08-13,
        docs/OFFENE_PUNKTE.md 5.16) - Rueckgabe dict typ -> (H,W) float32 in
        0..1, ausserhalb von Land ueberall 0.

        BAUT AUF DEN VORHANDENEN TEILFAKTOREN AUF, rechnet nichts neu:
        `calculate_water_proximity`, `analyze_slope_suitability`,
        `evaluate_elevation_fitness` und `evaluate_farmland_radius` liefern
        bereits genau die vier Groessen, aus denen sich die Typen ableiten
        lassen. Eine zweite, eigene Gelaendeanalyse waere eine zweite Wahrheit.

        Die Kriterien folgen der Vorgabe woertlich:
          Bergdorf   "In den Bergen"                    -> hohe Lage, steiler
          Marktstadt "liegt am Wasser oder kann viele Staedte gut erreichen"
          Agrarstadt "Hat viel flaches Land und fruchtbare Biome in der Naehe"
          sonstige   "alle staedte die nicht reinpassen" -> konstante Grundguete

        `hoehe_suit` ist eine EIGNUNG (hoch = gute, also maessige Hoehe), nicht
        die Hoehe selbst - fuer das Bergdorf wird sie deshalb invertiert.

        FRUCHTBARKEIT OHNE BIOMKARTE: `biome_map` steht diesem Knoten nicht zur
        Verfuegung (settlement.settlements haengt laut Calculator-Graph an
        settlement.suitability und terrain.redistribution, nicht an biome.*).
        Als Ersatz dient `acker_suit` (evaluate_farmland_radius), das genau
        dafuer gedacht ist - flaches, nicht zu hoch gelegenes Umland. Eine
        echte Biom-Abhaengigkeit waere eine neue Graph-Kante und ein eigener
        Schritt.
        """
        land_mask = heightmap > 0.0
        wasser_suit = self.calculate_water_proximity(water_map, heightmap, progress_callback)
        flach_suit = self.analyze_slope_suitability(slopemap, progress_callback)
        hoehe_suit = self.evaluate_elevation_fitness(heightmap, progress_callback)
        acker_suit = self.evaluate_farmland_radius(flach_suit, hoehe_suit, land_mask,
                                                   progress_callback)

        # BERGIGKEIT AUS DER ECHTEN HOEHE, RELATIV ZU DIESER KARTE - nicht aus
        # `hoehe_suit`. Der erste Anlauf nahm `1 - hoehe_suit`, was falsch war
        # und gemessen fast nichts lieferte: `evaluate_elevation_fitness()` ist
        # eine EIGNUNG (hoch = angenehme Wohnhoehe), ihr Median liegt auf
        # dieser Karte bei 0.94, die Invertierung also bei 0.06 - nur 2.6 % der
        # Landflaeche kamen ueberhaupt als Bergdorf in Frage, obwohl 34.6 %
        # des Landes ueber 200 m liegen.
        #
        # Stattdessen der Rang der Hoehe zwischen Median und 95. Perzentil des
        # LANDES: relativ zur jeweiligen Karte, damit eine flache Steppenwelt
        # ebenso ihre "Berge" hat wie das Nevadin, und unabhaengig von
        # absoluten Metergrenzen, die je Region ohnehin verschieden gemeint
        # waeren.
        land_hoehen = heightmap[land_mask]
        if land_hoehen.size:
            unten = float(np.percentile(land_hoehen, 50))
            oben = float(np.percentile(land_hoehen, 95))
        else:
            unten, oben = 0.0, 1.0
        spanne = max(oben - unten, 1e-6)
        bergig = np.clip((heightmap - unten) / spanne, 0.0, 1.0)
        steil = 1.0 - np.clip(flach_suit, 0.0, 1.0)

        eignungen = {
            # In den Bergen: hohe Lage UND spuerbare Hangneigung. Beides
            # multiplikativ, damit ein flaches Hochplateau nicht schon als
            # Bergdorf zaehlt.
            "bergdorf": bergig * (0.4 + 0.6 * steil),
            # Am Wasser (der staerkste Anteil) - "oder kann viele Staedte gut
            # erreichen" steckt in der Ebenheit, die zugleich fuer gute
            # Wegverbindungen steht.
            "marktstadt": np.clip(wasser_suit, 0.0, 1.0) * (0.6 + 0.4 * np.clip(flach_suit, 0.0, 1.0)),
            # Viel flaches Land plus fruchtbares Umland.
            "agrarstadt": np.clip(acker_suit, 0.0, 1.0) * (0.5 + 0.5 * np.clip(flach_suit, 0.0, 1.0)),
            # Fischersiedlung: unmittelbar am Wasser, aber OHNE den
            # Ebenheits-/Hinterlandanteil der Marktstadt. Genau das ist der
            # Unterschied zwischen beiden: eine Marktstadt braucht ein
            # Umland und gute Landverbindungen, ein Fischerdorf braucht nur
            # die Kueste - und steht deshalb auch dort, wo fuer eine
            # Marktstadt nichts zu holen waere (Steilkueste, kleine Insel).
            # Der Deckel haelt sie unter der Marktstadt, wo BEIDE moeglich
            # sind: an einer guten Hafenlage mit Hinterland soll die
            # Marktstadt gewinnen, die Fischersiedlung bekommt die Reste.
            "fischersiedlung": np.clip(wasser_suit, 0.0, 1.0) * FISCHER_DECKEL,
            # Auffangtyp: konstant mittelmaessig. Er gewinnt genau dort, wo
            # kein anderer Typ ueber diese Schwelle kommt - deshalb ein fester
            # Wert und keine eigene Gelaendeformel.
            "sonstige": np.full(heightmap.shape, TYP_GRUNDGUETE, dtype=np.float32),
        }
        return {typ: np.where(land_mask, np.clip(k, 0.0, 1.0), 0.0).astype(np.float32)
                for typ, k in eignungen.items()}


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


# Wasserkosten-Stufen fuer das Kostenfeld, docs/SIEDLUNGEN_ENTWURF.md §4.1.
# Furt/kurze Bruecke bis zur Muendungstiefe-Groessenordnung bleibt machbar,
# aber teuer; tieferes Wasser ist fuer LANDwege gesperrt (Seewege siehe
# calculate_road_network()).
WASSERKOSTEN_FLACH = 8.0     # 0 bis -5 m
WASSERKOSTEN_TIEF = 25.0     # -5 bis -10 m
WASSER_SPERRE_M = -10.0      # tiefer: gesperrt
WEGERABATT = 0.4             # auf einem bereits gebauten Weg


# Steigungskosten fuer den Wegebau (2026-08-13, docs/OFFENE_PUNKTE.md 5.19).
#
# Die Kosten wachsen EXPONENTIELL mit dem Neigungswinkel in Grad, nicht mehr
# quadratisch mit dem Gradientenbetrag. Nutzer-Vorgabe: "keiner wuerde eine
# strasse bauen die zB mehr als x Grad steigung hat. und 5 Grad weniger ist
# schon wesentlich besser quasi. also irgendwann wird es einfach
# unpassierbar."
#
# STEIGUNG_SKALA_GRAD ist die Skala des Exponenten: je STEIGUNG_SKALA_GRAD
# Grad mehr vervielfachen sich die Zusatzkosten um e. Bei 8 Grad bedeutet
# das zwischen 15 und 20 Grad rund den doppelten Preis - genau das gewuenschte
# "5 Grad weniger ist wesentlich besser".
STEIGUNG_SKALA_GRAD = 8.0

# Ab hier gilt ein Hang als fuer den Wegebau unbrauchbar.
MAX_WEG_STEIGUNG_GRAD = 30.0

# ... und kostet dann WEGEBAU_UNMOEGLICH statt np.inf. BEWUSST ENDLICH:
# eine harte Sperre wuerde ganze Landesteile abschneiden, wenn ein Ort hinter
# einem durchgehend steilen Wall liegt - der Ort waere dann gar nicht mehr
# ans Netz anzubinden. Mit einem sehr hohen, aber endlichen Wert nimmt A*
# einen solchen Uebergang nur, wenn es wirklich keine Alternative gibt, und
# sucht sonst zuverlaessig den Umweg.
WEGEBAU_UNMOEGLICH = 500.0


def bau_kostenfeld(heightmap, slopemap, slope_distance_ratio, weg_maske=None):
    """
    Das Kostenfeld EINMAL bauen, docs/SIEDLUNGEN_ENTWURF.md §4.1 ("Kostenfeld
    zuerst") - nicht wie in der Vorlage je A*-Schritt neu aus slopemap
    ausrechnen (`calculate_movement_cost` tat das bei jedem einzelnen
    Nachbarn). Ebener Grund kostet 1.0, Wasser in drei Stufen, ein bereits
    vorhandener Weg kostet nur WEGERABATT so viel wie sonst - "der wichtigste
    Trick": Wege buendeln sich zu Hauptstrecken, statt parallel zu laufen.

    STEIGUNGSKOSTEN SEIT 2026-08-13 EXPONENTIELL (docs/OFFENE_PUNKTE.md 5.19,
    Nutzerbefund "die hoehenkosten sind zu niedrig. es gibt strassen die ueber
    hohe berge gehen"). Die alte Formel `1 + ratio * hang^2` war als Strafe
    praktisch wirkungslos: gemessen kostete ein 30-Grad-Hang nur das
    **1.5-fache** eines ebenen Pixels, ein 40-Grad-Hang das 2.06-fache. Ein
    Umweg von schon 50 % Mehrlaenge war damit teurer als die Direttissima
    ueber den Berg - genau das, was der Nutzer auf der Karte sah.

    Jetzt: `1 + ratio * (exp(winkel / STEIGUNG_SKALA_GRAD) - 1)`, gerechnet
    ueber den WINKEL in Grad statt ueber den Gradientenbetrag. Ab
    MAX_WEG_STEIGUNG_GRAD gilt der Hang als unbrauchbar und kostet
    WEGEBAU_UNMOEGLICH.

    Rueckgabe: (H,W) float64, np.inf wo gesperrt (Wasser tiefer als
    WASSER_SPERRE_M).
    """
    dz_dx = slopemap[:, :, 0].astype(np.float64)
    dz_dy = slopemap[:, :, 1].astype(np.float64)
    hang = np.hypot(dz_dx, dz_dy)
    winkel_grad = np.degrees(np.arctan(hang))
    kosten = 1.0 + slope_distance_ratio * (
        np.expm1(winkel_grad / STEIGUNG_SKALA_GRAD))
    kosten = np.where(winkel_grad >= MAX_WEG_STEIGUNG_GRAD,
                      np.maximum(kosten, WEGEBAU_UNMOEGLICH), kosten)

    if heightmap is not None:
        h = heightmap.astype(np.float64)
        flach = (h <= 0.0) & (h > -5.0)
        tief = (h <= -5.0) & (h > WASSER_SPERRE_M)
        gesperrt = h <= WASSER_SPERRE_M
        kosten = np.where(flach, WASSERKOSTEN_FLACH, kosten)
        kosten = np.where(tief, WASSERKOSTEN_TIEF, kosten)
        kosten = np.where(gesperrt, np.inf, kosten)

    if weg_maske is not None and np.any(weg_maske):
        kosten = np.where(weg_maske, kosten * WEGERABATT, kosten)

    return kosten


# Seeweg-Kostenfeld, docs/SIEDLUNGEN_ENTWURF.md §4.4 - das SPIEGELBILD des
# Landkostenfelds: Land ist gesperrt, Flachwasser teuer (an der Kueste
# entlangtasten soll sich nicht lohnen), richtiges tiefes Wasser billig.
SEEWEG_KOSTEN_FLACH = 3.0     # 0 bis SEEWEG_TIEFE_ZIEL_M
SEEWEG_TIEFE_ZIEL_M = -10.0   # ab hier "echtes" tiefes Wasser


def bau_seekostenfeld(heightmap, seegrad=None):
    """
    Kostenfeld fuer Seewege - Land gesperrt, Flachwasser teuer, tiefes Wasser
    billig. Keine Hangkosten (der Meeresboden ist fuer die Route irrelevant).

    Mit `seegrad` (docs/OFFENE_PUNKTE.md 3.3, "Seegrad als Grundlage fuer
    Seewege - 'ab Grad 1' statt 'ab 10 m Tiefe'") zaehlt die See-VORONOI-
    GLIEDERUNG statt der reinen Hoehe: Grad 0 (Kuestenzelle) teuer,
    ab Grad 1 billig. OHNE seegrad (alter Nicht-Weltkarten-Pfad) bleibt die
    Hoehenschwelle SEEWEG_TIEFE_ZIEL_M erhalten.
    """
    h = heightmap.astype(np.float64)
    if seegrad is not None:
        tief = seegrad >= 1
    else:
        tief = h <= SEEWEG_TIEFE_ZIEL_M
    kosten = np.where(h > 0.0, np.inf, np.where(tief, 1.0, SEEWEG_KOSTEN_FLACH))
    return kosten


def _naechster_kuestenpunkt(x, y, heightmap):
    """
    Naechstes Wasserpixel zu (x,y) - der tatsaechliche Ausgangspunkt eines
    Seewegs.

    WARUM NOTWENDIG. "Land ist gesperrt" (§4.4) gilt fuer den Seeweg-Kosten-
    feld woertlich - bau_seekostenfeld() setzt jedes Landpixel auf np.inf.
    Eine Siedlung steht aber so gut wie nie GENAU auf der Wasserlinie,
    sondern ein paar Pixel landeinwaerts. Ohne dieses Snapping haette A* am
    Ausgangspunkt selbst schon nur unendlich teure Nachbarn und faende NIE
    einen Weg, egal wie nah das Meer liegt - gemessen: zwei Hafenstaedte
    beiderseits eines 10 Pixel breiten, tiefen Kanals bekamen 0 Seewege statt
    des erwarteten einen.

    Die eigentliche Route laeuft weiterhin STRIKT durchs Wasser (§4.4 bleibt
    woertlich gueltig); nur die kurze Verbindung Siedlung->Kueste wird separat
    als gerade Strecke angehaengt, nicht durch das Seeweg-A* selbst gesucht.
    """
    height, width = heightmap.shape
    yi = int(np.clip(round(y), 0, height - 1))
    xi = int(np.clip(round(x), 0, width - 1))
    if heightmap[yi, xi] <= 0.0:
        return x, y
    wasser = heightmap <= 0.0
    if not np.any(wasser):
        return None
    _abstand, index = distance_transform_edt(~wasser, return_indices=True)
    return float(index[1][yi, xi]), float(index[0][yi, xi])


def _seeweg_anteil_tief(pfad, heightmap, seegrad=None):
    """Anteil der Pfadpunkte in echtem tiefen Wasser - die Auflage aus §4.4:
    'der Weg muss den groessten Teil seiner Laenge in Wasser ab 10 m Tiefe
    liegen. Ein Seeweg, der sich an der Kueste entlangtastet, waere kein
    Seeweg, sondern ein schlechter Landweg.'

    Mit `seegrad` gilt statt der Hoehenschwelle "ab Grad 1" (docs/OFFENE_PUNKTE.md
    3.3) - dieselbe Umstellung wie in `bau_seekostenfeld()`."""
    if not pfad:
        return 0.0
    height, width = heightmap.shape
    tief = 0
    for x, y in pfad:
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        if seegrad is not None:
            if seegrad[yi, xi] >= 1:
                tief += 1
        elif heightmap[yi, xi] <= SEEWEG_TIEFE_ZIEL_M:
            tief += 1
    return tief / len(pfad)


# Rang als Zahl fuer die Bereitschaftsformel (docs/SIEDLUNGEN_ENTWURF.md §4.3):
# Bereitschaft = Rang(A) * Rang(B) * (gleiche Kultur ? 1.0 : 0.45). Zwei
# Staedte (3*3=9) verbinden sich damit praktisch immer, zwei Doerfer
# verschiedener Kultur (1*1*0.45=0.45) fast nie.
RANG_ZAHL = {"dorf": 1, "siedlung": 2, "stadt": 3}


# =============================================================================
# STADTTYPEN (Nutzer-Vorgabe 2026-08-13, docs/OFFENE_PUNKTE.md 5.16)
# =============================================================================
#
# Jede Siedlung bekommt einen von vier Typen. Der Typ folgt aus der LAGE (in
# den Bergen -> Bergdorf, am Wasser/gut erreichbar -> Marktstadt, viel flaches
# Ackerland -> Agrarstadt, sonst "sonstige") und bestimmt danach zweierlei:
# die moegliche Groesse und das Handelsinteresse gegenueber anderen Orten.
#
# WARUM DER TYP NACH DER PLATZIERUNG BESTIMMT WIRD und nicht davor: der
# Nutzer hat die bestehende Platzierung ausdruecklich als gut bezeichnet
# ("Die Siedlungen sind ziemlich gut gesetzt worden bisher"). Sie bleibt
# deshalb unveraendert; die Typzuweisung liest nur die Lage der bereits
# gesetzten Orte aus. Einzige Ausnahme ist die Marktstadt, die je Region
# genau einmal vorkommt - dort wird unter den vorhandenen Orten der mit der
# besten Marktstadt-Eignung ausgewaehlt.
#
# `rang_erlaubt` schraenkt ein, welche Groessen ein Typ annehmen darf:
#   Bergdorf   "Klein bis mittel"  -> dorf/siedlung
#   Marktstadt "Mittel bis gross"  -> siedlung/stadt
#   Agrarstadt "Mittel bis gross"  -> siedlung/stadt
#   sonstige   alles
# Grundguete des Auffangtyps "sonstige". Ein Ort wird nur dann Bergdorf/
# Marktstadt/Agrarstadt, wenn seine Lage dort BESSER als dieser Wert ist -
# sonst bleibt es ein gewoehnlicher Ort. Zu niedrig gewaehlt bekaeme fast
# jeder Ort einen Sondertyp, zu hoch gaebe es nur noch "sonstige".
TYP_GRUNDGUETE = 0.42

# Obergrenze der Landmark-Kategorie "abgelegen" (siehe landmark_eignungen()).
# Sie ist der Auffangtyp: ein markanter Gipfel oder ein Kliff soll sie
# stechen, eine unauffaellige Wildnis nicht. Ohne Deckel gewaenne sie fast
# ueberall, weil `civ_map` auf weiten Teilen der Karte 0 ist.
ABGELEGEN_DECKEL = 0.55

# Daempfungsfaktor, mit dem eine Landmark-Kategorie nach jeder Wahl INNERHALB
# derselben Region abgewertet wird (siehe calculate_landmarks()). Sorgt fuer
# gemischte Landmarks je Region, statt dass die flaechenmaessig staerkste
# Kategorie alles belegt. 1.0 waere keine Vielfalt, 0.0 ein hartes Verbot
# jeder Wiederholung - beides unerwuenscht.
KATEGORIE_WIEDERHOLUNG = 0.45

# Ab wievielen zusammentreffenden Wegen eine Kreuzung als echte WEGSCHEIDE
# gilt (siehe kreuzungsgrade(), Nutzer-Vorgabe "taverne an einer kreuzung mit
# min. drei wegen"). Solche Kreuzungen werden bei der Roadsite-Platzierung
# bevorzugt; die uebrigen bleiben als schwaechere Kandidaten erhalten.
KREUZUNG_MIN_WEGE = 3

# Schwelle fuer den bedarfsgetriebenen Netzausbau (siehe kanten_nach_bedarf(),
# docs/OFFENE_PUNKTE.md 5.21): eine zusaetzliche Strecke wird nur gebaut, wenn
# ihr Gesamtnutzen fuer ALLE Handelspaare mindestens das so-und-sovielfache
# ihrer Baukosten betraegt. Groesser = sparsameres Netz.
NETZAUSBAU_MINDESTNUTZEN = 3.0

# Hoechstzahl zusaetzlicher Strecken je Kultur aus dem Bedarfsausbau - eine
# Sicherung dagegen, dass aus dem sparsamen Gabriel-Netz ein Vollgraph wird.
NETZAUSBAU_MAX_KANTEN = 3

# Obergrenze der Fischersiedlungs-Eignung (siehe stadttyp_eignungen()). Haelt
# sie unter der Marktstadt, wo beide moeglich waeren - an einer guten
# Hafenlage MIT Hinterland soll die Marktstadt gewinnen. Ueber TYP_GRUNDGUETE
# (0.42), damit sie den Auffangtyp "sonstige" an der Kueste dennoch sticht.
FISCHER_DECKEL = 0.62

# Fixe Kosten fuer den Wechsel Land<->Schiff, JE HAFEN (also zweimal je
# Seeweg). Nutzer-Vorgabe 2026-08-13: "die kosten allgemein auf ein schiff
# umzusteigen sind quasi fix und dann sind die kosten auf der see halbwegs
# guenstig. ich will nur nicht das alle nur noch per see transportieren,
# deshalb muss die 'umsteigekosten' eingestellt werden, so dass nur 35%
# seehandel besteht oder sowas".
#
# IN KILOMETERN ANGEGEBEN, NICHT IN ROHEN KOSTENPUNKTEN. Ein ebenes Pixel
# kostet 1.0, die Kosten eines Weges wachsen also mit seiner PIXELzahl - bei
# doppelter Aufloesung kostet derselbe Weg doppelt so viel. Ein fester
# Kostenwert waere damit bei 1024 px eine ganz andere Bremse als bei 384 px.
# Gemessen ist das beim Eichen aufgefallen: dieselbe Zahl ergab bei 320 px und
# 384 px deutlich verschiedene Seehandelsanteile. Ueber `hafenkosten(mpp)`
# unten wird daraus ein aufloesungsunabhaengiger Wert.
#
# Bedeutung: ein Hafenwechsel ist so teuer wie so viele Kilometer ebener
# Landweg. Groesser = weniger Seehandel.
# GEEICHT, nicht geraten: gemessen ueber vier Faelle (320/384 px, vier
# Seeds) faellt der Seehandelsanteil monoton 68 % (0 km) -> 50 % (22) ->
# 43 % (30) -> 33 % (45) -> 20 % (70). 40 km trifft das 35-%-Ziel im MITTEL.
#
# EHRLICHE EINSCHRAENKUNG: der Anteil streut je Karte stark (bei 45 km
# zwischen 0 % und 53 %). Das ist nicht zu beheben und auch richtig so - eine
# Karte ohne vorgelagerte Inseln hat keinen Seehandel, egal wie billig die
# Haefen sind, und eine Inselwelt hat viel. Der Regler stellt den DURCHSCHNITT
# ein, nicht den Wert jeder einzelnen Karte. Genau das entspricht der
# Nutzer-Vorgabe: "so dass nur 35% seehandel besteht oder sowas (insgesamt,
# manche inselorte sind natuerlich bei 100% seehandel)".
HAFEN_UMSTEIGEKOSTEN_KM = 40.0


def hafenkosten(meter_pro_pixel):
    """Umsteigekosten je Hafen in Kostenpunkten, aus HAFEN_UMSTEIGEKOSTEN_KM.

    Ein ebenes Pixel kostet 1.0 - die Umrechnung ist deshalb schlicht "wie
    viele Pixel sind diese Kilometer". Dadurch bremst der Hafenwechsel bei
    jeder Kartengroesse gleich stark, statt bei feiner Aufloesung faktisch zu
    verschwinden."""
    return HAFEN_UMSTEIGEKOSTEN_KM * 1000.0 / max(float(meter_pro_pixel), 1e-6)

# Zielanteil des Handels, der ueber See laufen soll (Nutzer: "35%").
SEEHANDEL_ZIEL = 0.35

STADTTYPEN = {
    "bergdorf":   {"name": "Bergdorf",   "rang_erlaubt": ("dorf", "siedlung")},
    "marktstadt": {"name": "Marktstadt", "rang_erlaubt": ("siedlung", "stadt")},
    "agrarstadt": {"name": "Agrarstadt", "rang_erlaubt": ("siedlung", "stadt")},
    # Fischersiedlung (2026-08-13, Nutzervorschlag: "vielleicht macht ein typ
    # 'fischersiedlung' noch sinn fuer kleine hafenstaedte oder seedoerfern
    # auf inseln?"). Sie fuellt eine echte Luecke: die Marktstadt ist per
    # Definition mittel bis gross und hoechstens einmal je Region - eine
    # Insel mit drei Haeusern kann also gar keine sein, braucht aber einen
    # Hafen, sonst ist sie ueberhaupt nicht ans Netz anzubinden. Deshalb
    # AUSDRUECKLICH klein ("die ortschaften sind immer recht klein"): nur
    # dorf/siedlung, nie stadt.
    "fischersiedlung": {"name": "Fischersiedlung", "rang_erlaubt": ("dorf", "siedlung")},
    "sonstige":   {"name": "Ort",        "rang_erlaubt": ("dorf", "siedlung", "stadt")},
}


def _handelsinteresse_einseitig(a, b):
    """
    Wie stark Ort `a` am Handel mit Ort `b` interessiert ist - die EINE
    Richtung, exakt nach der Nutzer-Vorgabe vom 2026-08-13.

    Die Vorgabe ist je Typ einseitig formuliert und ergibt paarweise zwei
    verschiedene Werte (Bergdorf sieht die eigene Marktstadt mit 5, die
    Marktstadt das Bergdorf mit 3). `handelsgewicht()` unten bildet daraus
    die SUMME - so vom Nutzer entschieden.

    Wo die Vorgabe groessenabhaengige Spannen nennt ("von 2 (kleiner Ort) bis
    6 (grosser Ort)"), ist die Groesse des ZIELS `b` gemeint: wie attraktiv
    der Handelspartner ist. Ein nicht ausdruecklich geregelter Fall bekommt
    1 - kein Handel waere falsch, Orte handeln immer ein wenig.
    """
    gleiche_kultur = bool(a.culture) and a.culture == b.culture
    typ_a = getattr(a, "settlement_type", "sonstige") or "sonstige"
    typ_b = getattr(b, "settlement_type", "sonstige") or "sonstige"
    # 0..1 ueber die drei Raenge - Grundlage der groessenabhaengigen Spannen
    groesse_b = (RANG_ZAHL.get(b.rank, 1) - 1) / 2.0

    if typ_a == "bergdorf":
        # "Handel 2 zu jeder Marktstadt, 2 zu sonstigen Staedten eigener
        #  Fraktion, 5 zur Marktstadt eigener Fraktion."
        if typ_b == "marktstadt":
            return 5.0 if gleiche_kultur else 2.0
        return 2.0 if gleiche_kultur else 1.0

    if typ_a == "marktstadt":
        # "10 Handel zu Marktstaedten, 3 Handel zu Staedten eigener Fraktion."
        if typ_b == "marktstadt":
            return 10.0
        return 3.0 if gleiche_kultur else 1.0

    if typ_a == "fischersiedlung":
        # Eigene Handelswerte (Nutzer: "mit eigenen tradewerten"). Eine
        # Fischersiedlung lebt vom Absatz an groessere Orte und vom
        # Kuestenhandel untereinander: zur Marktstadt am staerksten (dort
        # geht der Fang hin), zu anderen Fischersiedlungen mittel (Austausch
        # entlang der Kueste), zum Binnenland wenig.
        if typ_b == "marktstadt":
            return 6.0 if gleiche_kultur else 3.0
        if typ_b == "fischersiedlung":
            return 3.0 if gleiche_kultur else 2.0
        return 1.5 if gleiche_kultur else 1.0

    if typ_a == "agrarstadt":
        # "Handel zu Staedten von 2 (kleiner Ort) bis 6 (grosser Ort) mit
        #  Faktor 1.5 fuer eigene Fraktion."
        wert = 2.0 + 4.0 * groesse_b
        return wert * (1.5 if gleiche_kultur else 1.0)

    # "alle staedte die nicht reinpassen. Handel ist groessenabhaengig von
    #  1 fremde Fraktion klein bis 4 fremde Fraktion gross mit faktor 1.5
    #  fuer eigene Fraktion."
    wert = 1.0 + 3.0 * groesse_b
    return wert * (1.5 if gleiche_kultur else 1.0)


# Kantenlaenge des GROBEN Gitters, auf dem die Erreichbarkeit gerechnet wird.
# Siehe erreichbarkeits_matrix() - fuer eine Eignungs-RANGFOLGE genuegt ein
# sehr grobes Feld, gemessen 0.997 Rangkorrelation gegen die volle Aufloesung
# bei 92-fachem Tempo.
ERREICHBARKEIT_GITTER_PX = 128


def erreichbarkeits_matrix(kostenfeld, positionen, grob_px=ERREICHBARKEIT_GITTER_PX):
    """
    Wegkosten-Matrix (n,n) zwischen allen `positionen` (Liste von (x, y)).

    WARUM MULTI-SOURCE-DIJKSTRA UND NICHT A* JE PAAR (2026-08-13,
    docs/OFFENE_PUNKTE.md 5.17): A*s einziger Vorteil ist seine Zielheuristik,
    die auf EIN Ziel zulenkt. Fuer eine ganze Kostenmatrix gibt es kein
    einzelnes Ziel; A* faellt dort auf Dijkstra zurueck, macht das aber
    n*(n-1)/2 mal statt n mal. Eine Dijkstra-Welle je Startpunkt
    (`skimage.graph.MCP_Geometric.find_costs`) liefert dagegen in EINEM Lauf
    die Kosten zu ALLEN uebrigen Punkten. Gemessen bei 12 Orten/1024 px:
    4.07 s fuer alle Wellen gegen 14.30 s hochgerechnet fuer die A*-Paare.

    WARUM AUF EINEM GROBEN GITTER: hier wird BEWERTET, nicht gezeichnet - es
    zaehlt die Rangfolge ("welcher Ort ist besser angebunden"), nicht der
    Meterwert und schon gar nicht die Weggeometrie. Gemessen (30 Orte,
    1024 px): auf 128 px gerechnet ist die Rangfolge der Zentralitaet zu
    0.997 dieselbe wie auf voller Aufloesung, bei 92-fachem Tempo (7.8 s ->
    0.08 s). Die tatsaechlichen WEGE entstehen weiterhin per feinem A* in
    calculate_road_network() - das hier ersetzt sie nicht.

    Nicht erreichbare Paare stehen als np.inf in der Matrix.
    """
    from scipy.ndimage import zoom
    from skimage.graph import MCP_Geometric

    n = len(positionen)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    feld = np.asarray(kostenfeld, dtype=np.float64)
    feld = np.where(np.isfinite(feld), feld, 1e6)
    size = feld.shape[0]

    faktor = min(1.0, float(grob_px) / max(1, size))
    if faktor < 1.0:
        grob = zoom(feld, faktor, order=1)
        grob = np.maximum(grob, 1e-6)
    else:
        grob = feld
        faktor = 1.0

    hoehe_g, breite_g = grob.shape
    knoten = []
    for x, y in positionen:
        xi = int(np.clip(round(float(x) * faktor), 0, breite_g - 1))
        yi = int(np.clip(round(float(y) * faktor), 0, hoehe_g - 1))
        knoten.append((yi, xi))

    matrix = np.full((n, n), np.inf, dtype=np.float64)
    for i, start in enumerate(knoten):
        mcp = MCP_Geometric(grob, fully_connected=True)
        kosten, _ = mcp.find_costs([start])
        for j, ziel in enumerate(knoten):
            matrix[i, j] = kosten[ziel]
    # Zurueck auf die Skala des feinen Gitters, damit die Zahlen mit
    # Pfadkosten anderswo vergleichbar bleiben.
    if faktor < 1.0:
        matrix = matrix / faktor
    np.fill_diagonal(matrix, 0.0)
    return matrix


def zentralitaet(matrix):
    """
    Je Ort ein Wert in 0..1: wie gut er die uebrigen Orte erreicht (1 = am
    besten angebunden). Grundlage ist die Summe der Wegkosten zu allen
    anderen (Closeness) - unerreichbare Ziele zaehlen mit dem hoechsten
    vorkommenden endlichen Wert, damit eine abgeschnittene Insel nicht
    versehentlich als "gut angebunden" durchgeht (inf wuerde beim
    Normieren sonst zu NaN).
    """
    n = len(matrix)
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    endlich = matrix[np.isfinite(matrix)]
    strafe = float(endlich.max()) if endlich.size else 1.0
    gefuellt = np.where(np.isfinite(matrix), matrix, strafe)
    summe = gefuellt.sum(axis=1)
    lo, hi = float(summe.min()), float(summe.max())
    if hi - lo < 1e-9:
        return np.ones(n, dtype=np.float64)
    # kleine Summe = gut erreichbar -> invertieren
    return 1.0 - (summe - lo) / (hi - lo)


def handelsgewicht(a, b):
    """
    Handelsinteresse der KANTE zwischen zwei Orten: die Summe beider
    einseitiger Interessen (Nutzerentscheidung 2026-08-13 - Alternativen
    waeren Maximum oder Mittel gewesen).

    Tritt im Bereitschaftstest des Wegenetzes an die Stelle des frueheren
    `rang_a * rang_b * kulturfaktor`.
    """
    return _handelsinteresse_einseitig(a, b) + _handelsinteresse_einseitig(b, a)
BEREITSCHAFT_FREMDKULTUR = 0.45


def netzdistanzen(kanten, anzahl):
    """
    Kuerzeste Wege ZWISCHEN ALLEN ORTEN UEBER DAS GEBAUTE NETZ (nicht Luftlinie,
    nicht Direktkosten) - (n,n)-Matrix, np.inf wo unverbunden.

    Laeuft auf dem GRAPHEN der Orte, nicht auf der Pixelkarte: bei rund 30
    Knoten ist das ein Wimpernschlag, waehrend dieselbe Frage auf Pixelebene
    Sekunden kosten wuerde. Genau deshalb ist der Grenznutzen-Ausbau unten
    ueberhaupt bezahlbar (docs/OFFENE_PUNKTE.md 5.21).

    Parameter: `kanten` als dict {(i,j): kosten} mit i<j, `anzahl` = Zahl der Orte.
    """
    import heapq

    nachbarn = [[] for _ in range(anzahl)]
    for (i, j), kosten in kanten.items():
        if not np.isfinite(kosten):
            continue
        nachbarn[i].append((j, float(kosten)))
        nachbarn[j].append((i, float(kosten)))

    D = np.full((anzahl, anzahl), np.inf, dtype=np.float64)
    for start in range(anzahl):
        D[start, start] = 0.0
        halde = [(0.0, start)]
        while halde:
            dist, k = heapq.heappop(halde)
            if dist > D[start, k]:
                continue
            for nachbar, kosten in nachbarn[k]:
                neu = dist + kosten
                if neu < D[start, nachbar]:
                    D[start, nachbar] = neu
                    heapq.heappush(halde, (neu, nachbar))
    return D


def umwegfaktoren(netz_D, direkt_C):
    """
    Je Ortspaar: Netzdistanz geteilt durch die Kosten des DIREKTEN Weges.

    Der billige Detektor fuer "hier fehlt eine Verbindung" (docs/OFFENE_PUNKTE
    5.21, Nutzerbeobachtung am Kartenbild: zwei Doerfer beiderseits eines
    Berges, die nur ueber den ganzen Umweg unten herum zusammenkommen). Ein
    Wert nahe 1 heisst "das Netz bildet den direkten Weg gut ab", ein grosser
    Wert heisst "die beiden sind eigentlich nah beieinander, das Netz macht
    einen weiten Bogen".

    Unverbundene Paare bekommen np.inf, Paare ohne endlichen Direktweg NaN
    (dort ist die Frage sinnlos - es gibt keinen Weg, den man abkuerzen
    koennte).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        faktor = netz_D / direkt_C
    faktor = np.where(np.isfinite(direkt_C) & (direkt_C > 0), faktor, np.nan)
    np.fill_diagonal(faktor, 1.0)
    return faktor


def kanten_traffic(kanten, handel_W, anzahl):
    """
    Wieviel Handel laeuft ueber jede Kante des Netzes.

    Fuer jedes Ortspaar wird der kuerzeste Weg IM NETZ bestimmt und sein
    Handelsvolumen `w_ij` auf alle Kanten dieses Weges aufgeschlagen. Das ist
    gewichtete Kanten-Betweenness.

    Zwei Verwendungen (docs/OFFENE_PUNKTE.md 5.22 und 6.27):
      * der Anteil des Handels, der ueber SEEwege laeuft - die Groesse, gegen
        die die Hafen-Umsteigekosten geeicht werden ("nur 35 % seehandel")
      * die "Traffic"-Angabe, die spaeter beim Anklicken einer Strasse
        angezeigt werden soll. Vorher waere jede solche Zahl erfunden.

    Rueckgabe: dict {(i,j): volumen} mit i<j, nur fuer Kanten mit Verkehr.
    """
    import heapq

    nachbarn = [[] for _ in range(anzahl)]
    for (i, j), kosten in kanten.items():
        if not np.isfinite(kosten):
            continue
        nachbarn[i].append((j, float(kosten)))
        nachbarn[j].append((i, float(kosten)))

    W = np.asarray(handel_W, dtype=np.float64)
    traffic = {}
    for start in range(anzahl):
        dist = np.full(anzahl, np.inf)
        vorgaenger = np.full(anzahl, -1, dtype=np.int64)
        dist[start] = 0.0
        halde = [(0.0, start)]
        while halde:
            d, k = heapq.heappop(halde)
            if d > dist[k]:
                continue
            for nachbar, kosten in nachbarn[k]:
                neu = d + kosten
                if neu < dist[nachbar]:
                    dist[nachbar] = neu
                    vorgaenger[nachbar] = k
                    heapq.heappush(halde, (neu, nachbar))
        # Volumen jedes Ziels entlang seines Weges zurueckverfolgen. Nur
        # start<ziel, damit jede Kante genau einmal je Paar zaehlt.
        for ziel in range(anzahl):
            if ziel <= start or not np.isfinite(dist[ziel]):
                continue
            volumen = float(W[start, ziel])
            if volumen <= 0:
                continue
            k = ziel
            while vorgaenger[k] >= 0:
                v = int(vorgaenger[k])
                schluessel = (min(k, v), max(k, v))
                traffic[schluessel] = traffic.get(schluessel, 0.0) + volumen
                k = v
    return traffic


def seehandel_anteil(kanten, handel_W, anzahl, seekanten):
    """
    Anteil des Handelsvolumens, das ueber SEEwege laeuft (0..1).

    Die Groesse, gegen die die Hafen-Umsteigekosten geeicht werden
    (Nutzer-Vorgabe 2026-08-13: "ich will nur nicht das alle nur noch per see
    transportieren, deshalb muss die 'umsteigekosten' eingestellt werden, so
    dass nur 35% seehandel besteht oder sowas").

    Gezaehlt wird ueber `kanten_traffic()`, also ueber das tatsaechlich
    durchlaufende Volumen - nicht ueber die blosse ANZAHL der Seewege. Zwei
    kaum genutzte Faehren sollen nicht so viel zaehlen wie eine stark
    befahrene Hauptroute.
    """
    traffic = kanten_traffic(kanten, handel_W, anzahl)
    gesamt = sum(traffic.values())
    if gesamt <= 0:
        return 0.0
    see = sum(v for k, v in traffic.items() if k in seekanten)
    return see / gesamt


def kanten_nach_bedarf(direkt_C, handel_W, bestehende, kandidaten,
                       mindest_nutzen=NETZAUSBAU_MINDESTNUTZEN, hoechstens=None):
    """
    Waehlt zusaetzliche Verbindungen nach ihrem GESAMTNUTZEN fuer alle
    Handelspaare (docs/OFFENE_PUNKTE.md 5.21).

    DAS PROBLEM, DAS DAMIT GELOEST WIRD (Nutzerbeobachtung 2026-08-13): das
    bisherige Verfahren entscheidet je PAAR - lohnt sich fuer A und B eine
    direkte Strecke? Ein Pass ueber einen Bergruecken lohnt sich fuer kein
    einzelnes Paar, er ist fuer jedes fuer sich zu teuer. Dass er fuer ZEHN
    Paare zusammen der groesste Gewinn waere, sieht ein paarweises Verfahren
    strukturell nicht: "aber nicht das 10 doerfer daran interessiert sind eine
    verbindung oben zu haben".

    Der Nutzen einer Kandidatenkante e ist die Summe ueber ALLE Ortspaare:

        Nutzen(e) = SUMME_ij  handel_ij * (netzdistanz_ohne_e - netzdistanz_mit_e)

    also: um wieviel verkuerzt diese eine Strecke die Wege aller Handelspaare
    zusammen, gewichtet mit ihrem Handelsvolumen. Gebaut wird gierig die Kante
    mit dem besten Verhaeltnis Nutzen zu Baukosten, danach werden die
    Netzdistanzen neu bestimmt und die naechste gesucht.

    ERSETZT DEN GABRIEL-GRAPHEN NICHT, sondern ergaenzt ihn (Nutzer-Vorgabe:
    "bisher sieht das ziemlich gut aus, halte dich etwa daran, aber wir wollen
    das nur etwas besser machen"). `bestehende` ist das bereits gebaute Netz.

    Parameter:
        direkt_C     (n,n) Wegkosten zwischen den Orten (erreichbarkeits_matrix)
        handel_W     (n,n) Handelsinteresse je Paar (handelsgewicht)
        bestehende   dict {(i,j): kosten}, i<j - das schon gebaute Netz
        kandidaten   Liste von (i,j)-Paaren, die gebaut werden koennten
        mindest_nutzen  Schwelle fuer Nutzen/Baukosten; darunter wird nichts gebaut
        hoechstens   Obergrenze fuer die Zahl neuer Kanten (None = unbegrenzt)

    Rueckgabe: Liste der gewaehlten (i, j)-Paare, in der Reihenfolge des Baus.
    """
    anzahl = len(direkt_C)
    netz = dict(bestehende)
    offen = [(min(i, j), max(i, j)) for i, j in kandidaten
             if (min(i, j), max(i, j)) not in netz and i != j]
    gewaehlt = []

    W = np.asarray(handel_W, dtype=np.float64)
    while offen:
        D = netzdistanzen(netz, anzahl)
        # Unverbundenes zaehlt mit einem hohen, aber endlichen Ersatzwert:
        # sonst waere jede Differenz gegen inf entweder inf oder NaN, und der
        # Vergleich zwischen zwei Kandidaten, die BEIDE etwas verbinden,
        # unmoeglich.
        endlich = D[np.isfinite(D)]
        ersatz = (float(endlich.max()) * 4.0 + 1.0) if endlich.size else 1.0
        D_e = np.where(np.isfinite(D), D, ersatz)

        bester, bester_wert = None, 0.0
        for (i, j) in offen:
            kosten = float(direkt_C[i, j])
            if not np.isfinite(kosten) or kosten <= 0:
                continue
            probe = dict(netz)
            probe[(i, j)] = kosten
            D_neu = netzdistanzen(probe, anzahl)
            D_neu_e = np.where(np.isfinite(D_neu), D_neu, ersatz)
            ersparnis = float(np.sum(W * np.maximum(D_e - D_neu_e, 0.0))) / 2.0
            wert = ersparnis / kosten
            if wert > bester_wert:
                bester, bester_wert = (i, j), wert

        if bester is None or bester_wert < mindest_nutzen:
            break
        netz[bester] = float(direkt_C[bester[0], bester[1]])
        gewaehlt.append(bester)
        offen.remove(bester)
        if hoechstens is not None and len(gewaehlt) >= hoechstens:
            break
    return gewaehlt


def _gabriel_kandidaten(punkte):
    """
    Gabriel-Graph ueber `punkte` (N,2): Kandidatenpaare fuer das Wegenetz,
    docs/SIEDLUNGEN_ENTWURF.md §4.2. Zwei Punkte A,B sind Kandidaten, wenn im
    Kreis mit Durchmesser AB kein dritter Punkt liegt - ergibt ein sparse,
    zusammenhaengendes Netz mit typisch 2-3 Nachbarn je Ort statt eines Sterns
    oder einer Vollverknuepfung.

    O(n^3), das ist bei Siedlungszahlen im niedrigen Zehnerbereich (2-5 je
    Kultur, bis zu neun Kulturen) unproblematisch - eine Beschleunigung waere
    hier vorzeitige Optimierung.

    Rueckgabe: Liste von (i, j)-Indexpaaren, i<j.
    """
    n = len(punkte)
    kandidaten = []
    for i in range(n):
        for j in range(i + 1, n):
            mitte = (punkte[i] + punkte[j]) / 2.0
            radius2 = float(np.sum((punkte[i] - punkte[j]) ** 2)) / 4.0
            frei = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if float(np.sum((punkte[k] - mitte) ** 2)) < radius2:
                    frei = False
                    break
            if frei:
                kandidaten.append((i, j))
    return kandidaten


def kreuzungen_finden(roads, sea_roads, settlements, shape, mindestabstand_siedlung=None):
    """
    Kreuzungen NACH dem Routing, docs/SIEDLUNGEN_ENTWURF.md §4.5: "alle
    Wegepixel, an denen sich zwei Strecken treffen und die NICHT auf einem
    Ort liegen". Sie entstehen von selbst durch den Wegerabatt aus §4.1 -
    dort, wo zwei Strecken ein Stueck gemeinsam gehen und sich wieder trennen.

    Rein geometrisch: jeder Weg (Land wie See) rasterisiert mit seiner
    eigenen ID, ein Pixel zaehlt als Kreuzung, wenn es von mehr als einer
    ID beruehrt wird. Naheliegende Kreuzungspixel werden zu EINER Kreuzung
    zusammengefasst (`scipy.ndimage.label` auf der Kreuzungsmaske) - sonst
    meldete ein einziges reales Treffen, wo zwei Strecken ein paar Pixel
    breit gemeinsam laufen, mehrere "Kreuzungen" dicht nebeneinander.

    Rueckgabe: Liste von (x, y)-Punkten.
    """
    from scipy import ndimage

    height, width = shape
    id_karte = np.full((height, width), -1, dtype=np.int32)
    beruehrt_mehrfach = np.zeros((height, width), dtype=bool)

    alle_wege = list(roads) + list(sea_roads)
    for weg_id, weg in enumerate(alle_wege):
        for x, y in weg:
            xi = int(np.clip(round(x), 0, width - 1))
            yi = int(np.clip(round(y), 0, height - 1))
            bisherige = id_karte[yi, xi]
            if bisherige == -1:
                id_karte[yi, xi] = weg_id
            elif bisherige != weg_id:
                beruehrt_mehrfach[yi, xi] = True

    if not np.any(beruehrt_mehrfach):
        return []

    # Auf einem Ort liegende Treffer sind keine Kreuzung, sondern der Ort
    # selbst (jeder Weg endet ja an einer Siedlung - dort treffen sich
    # zwangslaeufig alle an diesem Ort ankommenden Strecken).
    if mindestabstand_siedlung is None:
        mindestabstand_siedlung = max(2.0, min(height, width) / 64.0)
    for s in settlements:
        if s.location_type != 'settlement':
            continue
        y0 = max(0, int(s.y - mindestabstand_siedlung))
        y1 = min(height, int(s.y + mindestabstand_siedlung) + 1)
        x0 = max(0, int(s.x - mindestabstand_siedlung))
        x1 = min(width, int(s.x + mindestabstand_siedlung) + 1)
        beruehrt_mehrfach[y0:y1, x0:x1] = False

    beschriftet, anzahl = ndimage.label(beruehrt_mehrfach)
    kreuzungen = []
    for i in range(1, anzahl + 1):
        ys, xs = np.nonzero(beschriftet == i)
        kreuzungen.append((float(xs.mean()), float(ys.mean())))
    return kreuzungen


def kreuzungsgrade(roads, sea_roads, kreuzungen, shape, radius=None):
    """
    WIEVIELE verschiedene Wege treffen sich an jeder Kreuzung.

    `kreuzungen_finden()` liefert nur die ORTE - "hier beruehren sich
    mindestens zwei Wege". Fuer die Nutzer-Vorgabe 2026-08-13 ("roadsites
    haben auch bestimmte kriterien, zB taverne ... an einer kreuzung mit min.
    drei wegen") reicht das nicht: eine Taverne gehoert an eine echte
    Wegscheide, nicht an jede Stelle, an der sich zwei Strecken streifen.

    Gezaehlt werden verschiedene Weg-IDs in einem kleinen Umkreis um den
    Kreuzungspunkt. Der Umkreis ist noetig, weil `kreuzungen_finden()` die
    Kreuzung als SCHWERPUNKT einer Pixelgruppe zurueckgibt - der genaue
    Mittelpunkt muss selbst gar nicht auf jedem beteiligten Weg liegen.

    Rueckgabe: Liste von int, gleiche Reihenfolge und Laenge wie `kreuzungen`.
    """
    height, width = shape
    if radius is None:
        radius = max(2, int(min(height, width) / 128))

    alle_wege = list(roads) + list(sea_roads)
    # Weg-IDs in ein Raster legen, dann je Kreuzung das Fenster auslesen -
    # billiger als je Kreuzung alle Wege durchzugehen.
    raster = [[set() for _ in range(width)] for _ in range(height)]
    for weg_id, weg in enumerate(alle_wege):
        for x, y in weg:
            xi = int(np.clip(round(x), 0, width - 1))
            yi = int(np.clip(round(y), 0, height - 1))
            raster[yi][xi].add(weg_id)

    grade = []
    for x, y in kreuzungen:
        xi = int(np.clip(round(x), 0, width - 1))
        yi = int(np.clip(round(y), 0, height - 1))
        ids = set()
        for yy in range(max(0, yi - radius), min(height, yi + radius + 1)):
            for xx in range(max(0, xi - radius), min(width, xi + radius + 1)):
                ids |= raster[yy][xx]
        grade.append(len(ids))
    return grade


class PathfindingSystem:
    """
    A*-Wegesuche auf einem VORBERECHNETEN Kostenfeld (siehe bau_kostenfeld()).
    Erstellt realistische Straßenverbindungen mit Spline-Interpolation und
    LOD-Optimierung.
    """

    def __init__(self, cost_field, map_size=64,
                 edge_distance_map=None, edge_bias=0.0, edge_bias_scale=16.0):
        """
        Parameter: cost_field - (H,W) float, aus bau_kostenfeld(); np.inf =
            gesperrt.
        Parameter: map_size (int) - tatsaechliche Pixel-Groesse, fuer die
            LOD-Suchbudgets.
        Parameter: edge_distance_map - optionale (H,W)-Distanz zur naechsten
            Voronoi-Zellgrenze (siehe _voronoi_edge_distance_map()); None =
            kein Edge-Bias.
        Parameter: edge_bias - Staerke der Bevorzugung von Zellgrenzen (0 = aus)
        Parameter: edge_bias_scale - charakteristische Distanz (Pixel), ueber
            die der Edge-Bias von "billig direkt auf der Grenze" zu "voller
            Strafaufschlag weit von jeder Grenze" saettigt.
        """
        self.cost_field = cost_field
        self.map_size = map_size
        self.edge_distance_map = edge_distance_map
        self.edge_bias = edge_bias
        self.edge_bias_scale = max(1e-3, edge_bias_scale)

        # Größenabhängige Pathfinding-Optimierungen.
        #
        # SUCHBUDGET DEUTLICH ANGEHOBEN (2026-08-10, Nutzer-Vorgabe: Strassen
        # sollen "moeglichst realistisch durch die taeler meandern"). Die
        # alten Budgets (500/1000/2000/5000) stammen aus der Zeit vor dem
        # Kostenfeld-Umbau (§4.1) - ein rein linearer Hangkosten-Aufschlag
        # liess A* fast immer zuegig zum Ziel finden. Das neue Kostenfeld ist
        # streckenweise sehr viel schaerfer (Hangkosten QUADRATISCH, Wasser
        # bis 25x, ganze Bereiche unendlich teuer): ein Weg, der einem Grat
        # wirklich ausweichen muss, braucht dafuer einen laengeren,
        # unoffensichtlichen Umweg - und genau den findet A* nur, wenn ihm
        # das Budget nicht vorher ausgeht.
        #
        # Gemessen an einem synthetischen Grat mit einem einzigen Pass (siehe
        # tests/smoke_test_settlement_roads.py, Abschnitt "Taeler"): bei den
        # ALTEN Budgets brach die Suche IMMER ab und lieferte den nutzlosen
        # Geradlinien-Fallback (siehe find_least_resistance_path()-Docstring)
        # - selbst dort, wo ein guter Weg klar existierte. Notwendiges Budget
        # wuchs dabei ungefaehr mit dem QUADRAT der Kantenlaenge:
        #   100 px -> 10000 noetig, 256 px -> 51200 noetig, 512 px -> 200000 noetig
        # (grob size*size, mit Sicherheitsmarge hier auf 1.3*size*size gesetzt).
        # Bei 512 px kostete ein einzelner derart schwieriger Kandidat rund
        # 3 s - vertretbar, weil das nur EINMAL am finalen LOD laeuft und die
        # meisten Kandidaten (kein derart erzwungener Umweg noetig) weit
        # darunter bleiben. Nach oben gedeckelt, damit ein pathologischer Fall
        # nicht unbegrenzt Zeit kostet.
        if map_size <= 64:
            self.max_search_nodes, self.path_resolution = 4000, 2
        else:
            self.max_search_nodes = min(600_000, int(1.3 * map_size * map_size))
            self.path_resolution = 1

    def calculate_movement_cost(self, x, y):
        """Bewegungskosten fuer einen Punkt - liest aus dem Kostenfeld, rechnet nichts neu."""
        height, width = self.cost_field.shape[:2]

        if x < 0 or x >= width or y < 0 or y >= height:
            return float('inf')

        cost = float(self.cost_field[y, x])

        # Edge-Bias: guenstiger nahe einer Voronoi-Zellgrenze, saettigt Richtung
        # (1 + edge_bias) je weiter man sich von jeder Grenze entfernt.
        if self.edge_distance_map is not None and self.edge_bias > 0 and np.isfinite(cost):
            distance_to_edge = self.edge_distance_map[y, x]
            if np.isfinite(distance_to_edge):
                cost *= 1.0 + self.edge_bias * (
                    distance_to_edge / (distance_to_edge + self.edge_bias_scale))

        return cost

    def _a_stern(self, start_x, start_y, end_x, end_y, max_nodes,
                 schnell=True):
        """
        A* - seit 2026-08-23 zuerst ueber den mit numba uebersetzten Kern
        (core/wegsuche_schnell.py), mit der Python-Fassung darunter als
        Rueckfall.

        GEMESSEN: 14x schneller bei Punkt-fuer-Punkt identischem Pfad und
        identischer Kostensumme (tests/smoke_test_wegsuche_schnell.py, alle
        fuenf Gruppen gruen, darunter der harte Fall "Ebene" mit lauter
        Kostengleichstaenden). Auf einer 1024-px-Karte fiel der Median je
        Route von 0.85 s auf unter 0.06 s.

        Der Rueckfall meldet sich LAUT (Logzeile in wegsuche_schnell), wenn
        numba fehlt - ein stiller Rueckfall auf einen 14x langsameren Pfad
        waere von Erfolg nicht zu unterscheiden, und genau dieser Fehler ist
        in diesem Projekt schon zweimal wochenlang unbemerkt geblieben
        (siehe CLAUDE.md).

        Der reine A*-Suchlauf mit festem Knotenbudget. Gibt den Pfad zurueck,
        wenn er das Ziel innerhalb von `max_nodes` erreicht, sonst None.

        OPTIMALITAET HAENGT NICHT AM BUDGET. A* mit einer zulaessigen
        Heuristik expandiert Knoten in Reihenfolge ihrer wahren optimalen
        Kosten - sobald das Ziel EXPANDIERT (nicht nur erreicht) wird, ist der
        gefundene Pfad der optimale, unabhaengig davon, wie gross `max_nodes`
        war. Ein kleineres Budget kann also nur FRUEHER aufgeben, nie einen
        schlechteren-aber-erfolgreichen Pfad liefern - das macht die
        Zwei-Stufen-Eskalation in find_least_resistance_path() sicher.

        VEKTORISIERUNG WAR HIER KEINE OPTION (A* ist von Natur aus
        sequenziell - jeder Schritt haengt vom vorigen ab), also wurden zwei
        klassische Python-A*-Kosten stattdessen direkt angegriffen (gemessen
        an einer echten 512-px-Welt mit 41 Siedlungen, cProfile):

        1. GESCHLOSSENE MENGE. Die Vorlage kannte keine - ein Zellenupdate,
           das eine bereits im `open_set` liegende, aber noch nicht
           expandierte Zelle erneut mit besserem g_score einfuegte,
           HINTERLIESS den alten (schlechteren) Heap-Eintrag einfach liegen
           ("lazy deletion" ohne die dazugehoerige Pruefung). Jeder so ver-
           waiste Eintrag wurde beim Poppen trotzdem als "neuer" Knoten voll
           expandiert. Jetzt: `closed` haelt bereits final expandierte Zellen
           fest, ein Popup mit veraltetem f_score wird sofort uebersprungen.
        2. METHODENAUFRUFE IM HEISSESTEN INNENPFAD. `calculate_movement_cost()`
           lief 10.48 MILLIONEN mal (6.4 s reine Aufrufzeit), `_heuristic()`
           1.39 Millionen mal (3.4 s) - beides fuer eine einzelne simple
           Feldabfrage bzw. eine Wurzel. Direkt inline gerechnet mit lokalen
           Variablen statt `self.`-Attributzugriffen je Aufruf.
        """
        height, width = self.cost_field.shape[:2]
        cost_field = self.cost_field
        path_resolution = self.path_resolution
        # Edge-Bias nur einbeziehen, wenn er ueberhaupt aktiv ist (bei den
        # Aufrufen aus calculate_road_network() immer aus) - sonst waere die
        # Inline-Fassung fuer den haeufigsten Fall unnoetig komplizierter.
        edge_map = self.edge_distance_map if self.edge_bias > 0 else None

        # DER SCHNELLE PFAD ZUERST.
        #
        # `schnell=False` erzwingt die Python-Fassung. Das ist KEIN
        # Debug-Schalter, sondern die Voraussetzung dafuer, dass
        # tests/smoke_test_wegsuche_schnell.py ueberhaupt etwas prueft:
        # ohne ihn verglich der Test nach dem Einbau numba gegen numba und
        # war gruen, ohne noch irgendetwas zuzusichern (2026-08-23, beim
        # Einbau sofort bemerkt - dieselbe Falle wie beim adaptiven Mesh,
        # siehe CLAUDE.md "Gruene Tests koennen eine tote Funktion
        # verdecken").
        if not schnell:
            return self._a_stern_python(start_x, start_y, end_x, end_y,
                                        max_nodes)
        from core.wegsuche_schnell import wegsuche as _wegsuche_schnell
        _schnell = _wegsuche_schnell(
            cost_field, (start_x, start_y), (end_x, end_y), max_nodes,
            schritt=path_resolution, kante=edge_map,
            kante_bias=self.edge_bias if edge_map is not None else 0.0,
            kante_skala=self.edge_bias_scale,
            h_gewicht=WEGSUCHE_H_GEWICHT)
        if _schnell is not None:
            return _schnell
        if _NUMBA_WEGSUCHE_DA:
            # numba war da und hat NICHTS gefunden - dann findet die
            # Python-Fassung auch nichts (identische Suche, siehe
            # smoke_test_wegsuche_schnell). Den langsamen Lauf sparen.
            return None
        return self._a_stern_python(start_x, start_y, end_x, end_y, max_nodes)

    def _a_stern_python(self, start_x, start_y, end_x, end_y, max_nodes):
        """
        Die urspruengliche Fassung in reinem Python.

        Bleibt als Rueckfall UND als Pruefmassstab: sie ist die Definition
        dessen, was ein richtiger Pfad ist, und der numba-Kern in
        core/wegsuche_schnell.py wird in
        tests/smoke_test_wegsuche_schnell.py Punkt fuer Punkt gegen sie
        gemessen.
        """
        height, width = self.cost_field.shape[:2]
        cost_field = self.cost_field
        path_resolution = self.path_resolution
        edge_map = self.edge_distance_map if self.edge_bias > 0 else None
        edge_bias = self.edge_bias
        edge_scale = self.edge_bias_scale

        open_set = [(0.0, start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0.0}
        closed = set()

        nodes_explored = 0

        while open_set and nodes_explored < max_nodes:
            current_f, current_x, current_y = heapq.heappop(open_set)
            current_key = (current_x, current_y)
            if current_key in closed:
                continue  # veralteter Heap-Eintrag - diese Zelle ist schon fertig expandiert
            closed.add(current_key)
            nodes_explored += 1

            if current_x == end_x and current_y == end_y:
                path = []
                k = current_key
                while k in came_from:
                    path.append(k)
                    k = came_from[k]
                path.append((start_x, start_y))
                return list(reversed(path))

            current_g = g_score[current_key]

            for dx in range(-path_resolution, path_resolution + 1, path_resolution):
                for dy in range(-path_resolution, path_resolution + 1, path_resolution):
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_x = current_x + dx
                    neighbor_y = current_y + dy

                    if (neighbor_x < 0 or neighbor_x >= width or
                            neighbor_y < 0 or neighbor_y >= height):
                        continue

                    neighbor_key = (neighbor_x, neighbor_y)
                    if neighbor_key in closed:
                        continue

                    movement_cost = float(cost_field[neighbor_y, neighbor_x])
                    if movement_cost == float('inf'):
                        continue
                    if edge_map is not None:
                        distance_to_edge = edge_map[neighbor_y, neighbor_x]
                        if np.isfinite(distance_to_edge):
                            movement_cost *= 1.0 + edge_bias * (
                                distance_to_edge / (distance_to_edge + edge_scale))

                    if dx != 0 and dy != 0:
                        movement_cost *= 1.414

                    tentative_g_score = current_g + movement_cost

                    if tentative_g_score < g_score.get(neighbor_key, float('inf')):
                        came_from[neighbor_key] = current_key
                        g_score[neighbor_key] = tentative_g_score
                        dxh = neighbor_x - end_x
                        dyh = neighbor_y - end_y
                        h = (dxh * dxh + dyh * dyh) ** 0.5
                        heapq.heappush(open_set, (tentative_g_score + h, neighbor_x, neighbor_y))

        return None

    def find_least_resistance_path(self, start_pos, end_pos, progress_callback=None):
        """
        Funktionsweise: A*-Pathfinding für Weg geringsten Widerstands zwischen zwei Punkten mit LOD-Optimierung
        Aufgabe: Findet optimalen Straßenverlauf zwischen Settlements
        Parameter: start_pos, end_pos, progress_callback - Positionen und Progress
        Returns: (path, erreicht) - `path` immer eine Liste von Wegpunkten,
            `erreicht` True nur, wenn A* das Ziel wirklich fand.

        WARUM `erreicht` NOTWENDIG IST (2026-08-10, gefunden beim Testen der
        Seewege). Wenn A* das Ziel nicht erreicht (Wasser/Berg sperrt jeden
        Weg, oder das Node-Limit greift zuerst), lieferte diese Methode schon
        immer die GERADE LUFTLINIE `[start, end]` als Fallback zurueck - eine
        Geometrie, kein echter Pfad. Ein Aufrufer, der die "Kosten" dieses
        Fallbacks bildet, indem er die Kostenfeld-Werte der zurueckgegebenen
        Punkte aufsummiert, sieht davon aber NUR den Start- und Zielpunkt -
        die eigentlich unpassierbaren Zellen DAZWISCHEN kommen in der
        zweielementigen Liste gar nicht vor. Ergebnis: eine Route durch
        gesperrtes tiefes Wasser wurde als "billig" gemessen (nur der
        Zielpunkt zaehlte), bestand den Bereitschaftstest und wurde als
        Landstrasse quer durchs Meer gebaut, statt dass ein Seeweg gesucht
        wurde. Ein Aufrufer, dem die Erreichbarkeit wichtig ist (jeder
        Bereitschafts-/Kulturzusammenhangstest in calculate_road_network()),
        muss deshalb `erreicht` statt der Pfadlaenge pruefen.

        ZWEI STUFEN STATT EINES GROSSEN BUDGETS (2026-08-10). Das Suchbudget
        wurde fuer das neue, schaerfere Kostenfeld deutlich angehoben (siehe
        __init__), damit Strassen einem Grat wirklich bis zum Pass ausweichen
        koennen. Gemessen an einer realen 512-px-Welt kostete das aber: die
        Siedlungskette bis zu den Roadsites brauchte 116.7 s statt 6.3 s bei
        128 px - obwohl die MEISTEN Kandidatenpaare gar keinen grossen Umweg
        brauchen und mit einem kleinen Budget laengst fertig waeren, bezahlte
        JEDER Aufruf das volle Budget, sobald A* aus irgendeinem Grund bis zum
        Ende suchen musste.
        Deshalb zuerst mit einem kleinen Budget (das alte, 500/1000/2000/5000)
        versuchen; nur wenn DAS scheitert, mit dem vollen Budget neu ansetzen.
        Dank der Optimalitaets-Eigenschaft von A* (siehe _a_stern()-Docstring)
        liefert das GARANTIERT denselben Pfad wie eine Suche, die sofort mit
        dem vollen Budget gestartet waere - nur eben in der Mehrzahl der
        Faelle sehr viel schneller.
        """
        start_x, start_y = int(start_pos[0]), int(start_pos[1])
        end_x, end_y = int(end_pos[0]), int(end_pos[1])

        schnelles_budget = min(self.max_search_nodes, self._SCHNELLES_BUDGET.get(
            "gross" if self.map_size > 64 else "klein", 5000))
        pfad = self._a_stern(start_x, start_y, end_x, end_y, schnelles_budget)
        if pfad is None and schnelles_budget < self.max_search_nodes:
            pfad = self._a_stern(start_x, start_y, end_x, end_y, self.max_search_nodes)

        if pfad is not None:
            return pfad, True

        # Kein Pfad gefunden - direkte Linie als Fallback (siehe Docstring:
        # NIE als echte Route behandeln, `erreicht` ist False).
        if progress_callback:
            progress_callback("Road Building", 30, "Pathfinding fallback - kein Pfad im Budget gefunden")
        return [(start_x, start_y), (end_x, end_y)], False

    # Kleines Erstbudget fuer die schnelle erste Stufe (siehe
    # find_least_resistance_path()) - dieselben Werte, die vor dem
    # Kostenfeld-Umbau als EINZIGES Budget dienten und dort fuer die meisten
    # Faelle ausreichten.
    _SCHNELLES_BUDGET = {"klein": 4000, "gross": 5000}

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
        except Exception as fehler:
            # Stiller Ersatzpfad: der Weg wird trotzdem gezeichnet, nur ohne
            # Glättung - sieht auf der Karte plausibel aus, ist aber
            # nicht der beabsichtigte sanfte Verlauf (CLAUDE.md "jeder
            # stille Rueckfall braucht eine laute Logzeile").
            logging.getLogger(__name__).warning(
                "Spline-Glättung fehlgeschlagen (%s, %d Kontrollpunkte) "
                "- Weg bleibt ungeglättet.", fehler, len(control_points))
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
        # -1.0 STATT np.inf FUER "UNERREICHT" (2026-08-10, OFFENE_PUNKTE 5.13).
        #
        # np.inf ist fuer echte Kosten korrekt gemeint ("unerreichbarer Pixel
        # hat unendliche Kosten"), aber ein stiller Landmine: jede kuenftige
        # Anzeige (Colorbar-Normierung liest feld.min()/.max()) oder
        # Serialisierung (JSON kennt kein Infinity) waere daran zerbrochen,
        # ohne dass beim Schreiben dieser Methode irgendetwas darauf
        # hingewiesen haette - genau das meldete der generische Pipeline-Test
        # (`smoke_test_pipeline_outputs.py`, Kategorie "NICHT-ENDLICH").
        #
        # -1.0 spiegelt genau die Konvention, die `city_mask` bereits hat
        # ("-1 = ausserhalb jeder Stadt"): city_mask==-1 GENAU DORT, wo
        # city_cost_map==-1.0 - ein Aufrufer kann beide gleich lesen.
        city_cost_map = np.full((height, width), -1.0, dtype=np.float32)

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
            # "-1.0 gilt als schlechter als jede echte Kostenzahl" statt eines
            # blossen `<` - eine reale Kostenzahl ist nie negativ, aber `<`
            # allein wuerde das erste Settlement nie gewinnen lassen (jede
            # echte Zahl ist groesser als -1.0).
            noch_unerreicht = city_cost_map < 0.0
            better = reached & (noch_unerreicht | (seed_cost < city_cost_map))
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
    # Stagnationsabbruch - siehe _run_physics_to_convergence().
    STAGNATION_FENSTER = 10
    STAGNATION_ANTEIL = 0.10          # 10 % Rueckgang je Fenster als Mindestmass
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
                 core_plotnode_spring_stiffness=1.2, plotnode_plotnode_spring_stiffness=1.0,
                 pressure_strength=0.8, core_mass=1.0, plot_node_mass=1.0,
                 plot_node_repulsion_strength=4.0, plot_gravity_strength=0.01,
                 plot_city_repulsion_strength=0.5, potential_strength=1.0,
                 damping=0.80, plot_tier_factor=1.0,
                 enable_core_plotnode_spring=True, enable_plotnode_plotnode_spring=True,
                 enable_pressure=True, enable_plot_node_repulsion=True,
                 enable_field_cores=True, enable_field_plotnodes=True,
                 enable_core_cell_containment=True, enable_wilderness_containment=True,
                 shader_manager=None, progress_callback=None, map_seed=None,
                 live_state_callback=None):
        self.map_size = int(map_size)
        # Skalierungs-Faktoren fuer Map-Groessen-Unabhaengigkeit (siehe
        # [[project-settlement-scale-invariance]]): 128px ist die Referenz-
        # Groesse, fuer die alle folgenden Distanz-/Flaechen-Konstanten
        # urspruenglich kalibriert wurden (Production-Default, siehe
        # gui/config/value_default.py TERRAIN.MAPSIZE). scale_factor skaliert
        # Distanzen linear mit map_size, area_scale_factor skaliert Flaechen
        # quadratisch - ohne das blieb z.B. eine Stadt bei 1024px ~64x so
        # gross (relativ zur Kartenflaeche) wie bei 128px, da radius/Flaechen-
        # Schwellen vorher absolute Pixelwerte waren.
        self.scale_factor = self.map_size / 128.0
        self.area_scale_factor = self.scale_factor ** 2

        self.plot_nodes_count = int(plot_nodes_count)
        self.plot_base_spacing = float(plot_base_spacing) * self.scale_factor
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

        # Kraft-Schalter: jetzt per Konstruktor-Parameter statt fest
        # verdrahtet (siehe [[project-settlement-physics-lab-parity]]) -
        # Production-Default bleibt "alles an" (anders als das Lab, das
        # bewusst mit allem AUS startet, um Kraefte einzeln zu isolieren -
        # das ist ein Debug-Workflow, kein sinnvoller Production-Default).
        self.enable_core_plotnode_spring = bool(enable_core_plotnode_spring)
        self.enable_plotnode_plotnode_spring = bool(enable_plotnode_plotnode_spring)
        self.enable_pressure = bool(enable_pressure)
        self.enable_plot_node_repulsion = bool(enable_plot_node_repulsion)
        self.enable_field_cores = bool(enable_field_cores)
        self.enable_field_plotnodes = bool(enable_field_plotnodes)
        self.enable_core_cell_containment = bool(enable_core_cell_containment)
        self.enable_wilderness_containment = bool(enable_wilderness_containment)

        # Physik-/Feld-Werte (Basis 1:1 aus tools/biome_lab/app.py's
        # _BASE_*-Konstanten bei Multiplikator 1.0) - jetzt per Konstruktor-
        # Parameter statt fest verdrahtet, siehe
        # [[project-settlement-physics-lab-parity]].
        self.core_plotnode_spring_stiffness = float(core_plotnode_spring_stiffness)
        self.plotnode_plotnode_spring_stiffness = float(plotnode_plotnode_spring_stiffness)
        self.pressure_strength = float(pressure_strength)
        self.core_mass = float(core_mass)
        self.plot_node_mass = float(plot_node_mass)
        self.plot_node_repulsion_strength = float(plot_node_repulsion_strength)
        self.plot_gravity_strength = float(plot_gravity_strength)
        self.plot_city_repulsion_strength = float(plot_city_repulsion_strength)
        self.potential_strength = float(potential_strength)
        self.damping = float(damping)
        # NEU (existierte im Lab nur implizit bei Multiplikator 1.0, siehe
        # TIER_*_THRESHOLD-Konstanten oben) - skaliert die Traffic-Tier-
        # Schwellen, siehe _classify_road_tiers().
        self.plot_tier_factor = float(plot_tier_factor)
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
        # Traffic-Tier-Schwellen (TIER_STRASSE_THRESHOLD etc.) sind Verkehrs-
        # WERT-Schwellen (akkumuliertes Rang-Distanz-Gewicht ueber den Plot-
        # Graphen), keine Raumdistanzen - bleiben bewusst unskaliert, bereits
        # ueber plot_tier_factor unabhaengig vom map_size einstellbar.

        # Klassen-Konstanten, die urspruenglich absolute Pixel-/Flaechenwerte
        # waren, hier per Instanz-Attribut (gleicher Name, ueberschattet die
        # Klassen-Konstante) auf die tatsaechliche Kartengroesse skaliert -
        # siehe [[project-settlement-scale-invariance]]. Alle bestehenden
        # Zugriffe (self.WILDERNESS_MIN_AREA etc.) bleiben unveraendert.
        self.WILDERNESS_MIN_AREA = self.WILDERNESS_MIN_AREA * self.area_scale_factor
        self.CITY_MIN_AREA = self.CITY_MIN_AREA * self.area_scale_factor
        self.SOFTENING = self.SOFTENING * self.scale_factor
        self.MAX_DISPLACEMENT_PER_TICK = self.MAX_DISPLACEMENT_PER_TICK * self.scale_factor
        self.PLOT_CORE_EDGE_MARGIN = self.PLOT_CORE_EDGE_MARGIN * self.scale_factor
        self.SEED_RELAX_MARGIN = self.SEED_RELAX_MARGIN * self.scale_factor

        # Vorher lokale Variablen/Inline-Konstanten an einzelnen Call-Sites -
        # zu skalierten Instanz-Attributen befoerdert, damit sie hier zentral
        # skaliert werden koennen (siehe jeweilige Nutzungsstelle).
        self.min_buffer_to_city_px = 10.0 * self.scale_factor        # _gen_step_2_plot_cores
        self.MIN_SEP_FLOOR = 3.0 * self.scale_factor                 # Kern-Node-Abstoßungs-Untergrenze
        self.MIN_REST_LENGTH = 2.0 * self.scale_factor               # Feder-Ruhelaengen-Untergrenze
        self.wall_spacing = 5.0 * self.scale_factor                  # Stadtmauer-Abstoßung
        self.wild_scale = 25.0 * self.scale_factor                   # Wildnisgrenzen-Abklingdistanz
        self.border_margin = 25.0 * self.scale_factor                # Kartenrand-Abstoßung
        self.hill_saturation_dist = 40.0 * self.scale_factor         # Hoehen-Gradient-Saettigungsdistanz
        # Siedlungs-"Gravitation" im Potentialfeld (weight = coeff/sqrt(dist)):
        # damit dieselbe RELATIVE Position (z.B. "10% der Kartendiagonale von
        # der Stadt entfernt") bei jeder Kartengroesse dieselbe Kraft erfaehrt,
        # muss der Koeffizient mit sqrt(scale_factor) skalieren, nicht linear -
        # siehe [[project-settlement-scale-invariance]] fuer die Herleitung.
        self.gravity_coeff = 140.0 * float(np.sqrt(self.scale_factor))

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
        # Lauf-Durchschnitt ueber die gesamte Konvergenz-Simulation (siehe
        # PlotEdge.traffic_avg, [[project-settlement-physics-lab-parity]]) -
        # getrennt von der abklingenden EMA oben, die weiterhin die Tier-
        # Klassifikation treibt.
        self.ridge_traffic_sum = {}
        self.ridge_traffic_sample_count = 0
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
        self.ridge_traffic_sum = {}
        self.ridge_traffic_sample_count = 0
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

        city_inside = self.city_mask >= 0
        dist_to_city = distance_transform_edt(~city_inside)

        edge_margin_px = int(round(self.PLOT_CORE_EDGE_MARGIN))
        map_edge_mask = np.ones_like(self.civ_map, dtype=bool)
        if edge_margin_px > 0:
            map_edge_mask[:edge_margin_px, :] = False
            map_edge_mask[-edge_margin_px:, :] = False
            map_edge_mask[:, :edge_margin_px] = False
            map_edge_mask[:, -edge_margin_px:] = False

        valid_mask = (~city_inside) & (dist_to_city >= self.min_buffer_to_city_px) & map_edge_mask
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
        ohne_nachbarn = []
        for city_core in city_cores:
            settlement_id = city_core.settlement_id
            own_neighbors = [
                pn for pn in self.plot_nodes
                if city_core.node_id in pn.neighbor_core_ids and pn.node_type == "standard_plot_node"
            ]
            if not own_neighbors:
                # EINE Sammelzeile statt einer je Stadt. Im Lauf vom
                # 2026-08-22 standen hier sieben identische WARNINGs
                # untereinander; das ist Laerm, nicht Diagnose. Der
                # eigentliche Befund - WIEVIELE von WIEVIELEN - steht jetzt
                # unten in einer Zeile.
                ohne_nachbarn.append(settlement_id)
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

        if ohne_nachbarn:
            logging.warning(
                "PlotPhysicsSystem: %d von %d Stadtkernen ohne eigene "
                "Voronoi-Nachbarn (settlement_id %s) - ihre Grenze bleibt "
                "unbesetzt. Vermutlich derselbe Grund, aus dem die "
                "Physikschleife nicht konvergiert; wird mit dem neuen "
                "Plot-System geklaert (docs/PERFORMANCE_2026-08-23.md 3.2).",
                len(ohne_nachbarn), len(city_cores),
                ", ".join(str(i) for i in ohne_nachbarn))
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
        return np.maximum(self.plot_base_spacing * civ_factor, self.MIN_REST_LENGTH)

    def _rest_length_plotnode_plotnode_batch(self, pos_a_arr, pos_b_arr, traffic_values):
        mids = 0.5 * (np.asarray(pos_a_arr, dtype=float) + np.asarray(pos_b_arr, dtype=float))
        civ_here = self._civ_at_continuous_batch(mids)
        civ_factor = np.maximum(1.0 - self.CIV_RESTLENGTH_STEEPNESS * civ_here, 0.25)
        base = self.plot_base_spacing * civ_factor
        shrink = 1.0 - np.minimum(np.asarray(traffic_values, dtype=float) * self.spring_traffic_shrink,
                                   1.0 - self.spring_min_shrink_fraction)
        shrink = np.maximum(shrink, self.spring_min_shrink_fraction)
        return np.maximum(base * shrink, self.MIN_REST_LENGTH)

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
                    min_sep = max(ideal_radius * 0.6, self.MIN_SEP_FLOOR)
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
                weight = self.gravity_coeff / np.sqrt(dist)
                field_arr[:, :, 0] -= (dx / dist) * weight * self.plot_gravity_strength
                field_arr[:, :, 1] -= (dy / dist) * weight * self.plot_gravity_strength

        hill_fx, hill_fy = self._compute_wilderness_hill_term(xx, yy)
        field_arr[:, :, 0] += hill_fx
        field_arr[:, :, 1] += hill_fy

        PUSH_CAP = 3.0  # Kraft-Betrag-Kappung, keine Distanz - bewusst unskaliert
        wall_spacing = self.wall_spacing
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
        gy_s, gx_s = np.gradient(gaussian_filter(signed_dist, sigma=3.0 * self.scale_factor))
        norm_s = np.sqrt(gx_s ** 2 + gy_s ** 2)
        mask_s = norm_s > 1e-10
        wgx = np.zeros_like(gx_s)
        wgy = np.zeros_like(gy_s)
        wgx[mask_s] = gx_s[mask_s] / norm_s[mask_s]
        wgy[mask_s] = gy_s[mask_s] / norm_s[mask_s]
        wild_scale = self.wild_scale
        push_strength = np.where(
            signed_dist < 0, 1.0 - np.exp(-np.abs(signed_dist) / wild_scale),
            np.exp(-np.maximum(signed_dist, 0) / wild_scale))
        field_arr[:, :, 0] += wgx * push_strength * 0.6
        field_arr[:, :, 1] += wgy * push_strength * 0.6

        BORDER_MARGIN = self.border_margin
        BORDER_STRENGTH = 0.4

        def _edge_push(dist_to_edge):
            t = np.clip(1.0 - dist_to_edge / BORDER_MARGIN, 0.0, 1.0)
            return t ** 2 * BORDER_STRENGTH

        field_arr[:, :, 0] += _edge_push(xx.astype(float)) - _edge_push((w - 1 - xx).astype(float))
        field_arr[:, :, 1] += _edge_push(yy.astype(float)) - _edge_push((h - 1 - yy).astype(float))

        field_arr *= self.potential_strength
        self.potential_field = field_arr

    def _compute_wilderness_hill_term(self, xx, yy):
        civ_mask_hill = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        dist_out = distance_transform_edt(~civ_mask_hill)

        HILL_SATURATION_DIST = self.hill_saturation_dist
        HILL_MAX_STRENGTH = 0.5
        monotonic_strength = HILL_MAX_STRENGTH * (1.0 - np.exp(-dist_out / HILL_SATURATION_DIST))
        monotonic_strength = np.where(civ_mask_hill, 0.0, monotonic_strength)

        hgy, hgx = np.gradient(gaussian_filter(self.heightmap, sigma=4.0 * self.scale_factor))
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
            self.ridge_traffic_sum = {}
            self.ridge_traffic_sample_count = 0
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

        # Lauf-Durchschnitt ueber die gesamte Konvergenz-Simulation (siehe
        # PlotEdge.traffic_avg, [[project-settlement-physics-lab-parity]]) -
        # reitet auf demselben Aufruf-Rhythmus wie die EMA unten mit, aendert
        # aber deren Verhalten nicht (separate Akkumulatoren).
        self.ridge_traffic_sample_count += 1
        for key, amount in fresh_contrib.items():
            self.ridge_traffic_sum[key] = self.ridge_traffic_sum.get(key, 0.0) + amount

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
        import time as _t
        # Teilzeiten der Physikschleife. Das Log sagte bisher nur, dass die
        # 100 Iterationen ausgeschoepft wurden - nicht, wie sich die 20 s auf
        # Schritt, Verkehr und Fortschrittsmeldung verteilen. Ohne diese
        # Aufteilung ist nicht zu entscheiden, ob eine hoehere Schrittweite
        # oder ein selteneres _simulate_traffic() der Hebel ist.
        _zeit = {"physics_step": 0.0, "sync": 0.0, "traffic": 0.0,
                 "fortschritt": 0.0}
        _t_ges = _t.perf_counter()
        stable_ticks = 0
        # ABBRUCH BEI STAGNATION (Punkt 3.4 der Leistungsliste,
        # docs/PERFORMANCE_2026-08-23.md).
        #
        # Diese Schleife lief bisher IMMER die vollen 100 Iterationen, weil
        # sie die Konvergenzschranke nie erreichte - im Log des Nutzers vom
        # 2026-08-22 steht dazu "MAX_PHYSICS_ITERATIONS (100) erreicht,
        # eingefroren ohne volle Konvergenz", und das kostete 20 s.
        #
        # Ein System, dessen groesste Verschiebung ueber
        # STAGNATION_FENSTER Iterationen um weniger als STAGNATION_ANTEIL
        # faellt, wird auch in den restlichen Iterationen nicht mehr
        # konvergieren. Der Endzustand ist derselbe "eingefroren ohne
        # Konvergenz", nur frueher erreicht.
        #
        # DIE PLOT-PHYSIK WIRD ERSETZT (Nutzer-Vorgabe 2026-08-23: "bei
        # settlements machen wir fuer die plots ein neues system, also hier
        # nur rudimentaer fehlerhafte systeme fixen oder ausschalten").
        # Deshalb hier bewusst KEIN Umbau der Kraftrechnung, sondern nur
        # der Abbruch und die Verlaufsaufzeichnung - die sagt dem neuen
        # System, woran das alte gescheitert ist.
        verlauf = []
        for iteration in range(1, self.MAX_PHYSICS_ITERATIONS + 1):
            self.iteration = iteration
            _a = _t.perf_counter()
            max_displacement = self._physics_step()
            _b = _t.perf_counter(); _zeit["physics_step"] += _b - _a
            self._sync_core_positions()
            _c = _t.perf_counter(); _zeit["sync"] += _c - _b

            if iteration % self.TRAFFIC_RECOMPUTE_INTERVAL == 0:
                self._simulate_traffic()
                _d = _t.perf_counter(); _zeit["traffic"] += _d - _c
                self._report_progress(
                    "plot_physics",
                    45 + int(50 * iteration / self.MAX_PHYSICS_ITERATIONS),
                    f"Physik-Iteration {iteration}/{self.MAX_PHYSICS_ITERATIONS} "
                    f"(max. Verschiebung {max_displacement:.3f}px)")
                self._report_live_state()
                _zeit["fortschritt"] += _t.perf_counter() - _d

            verlauf.append(float(max_displacement))
            if (len(verlauf) >= 2 * self.STAGNATION_FENSTER
                    and iteration % self.STAGNATION_FENSTER == 0):
                jetzt = min(verlauf[-self.STAGNATION_FENSTER:])
                davor = min(verlauf[-2 * self.STAGNATION_FENSTER:
                                    -self.STAGNATION_FENSTER])
                if jetzt > davor * (1.0 - self.STAGNATION_ANTEIL):
                    logging.info(
                        "PlotPhysicsSystem: Stagnation nach %d Iterationen "
                        "(groesste Verschiebung %.3f px, davor %.3f px - "
                        "Rueckgang unter %.0f %%). Eingefroren; die "
                        "restlichen %d Iterationen haetten daran nichts "
                        "geaendert.",
                        iteration, jetzt, davor,
                        100.0 * self.STAGNATION_ANTEIL,
                        self.MAX_PHYSICS_ITERATIONS - iteration)
                    break

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

        _t_schleife = _t.perf_counter() - _t_ges
        _a = _t.perf_counter()
        self._simulate_traffic()  # finale Traffic-Zuweisung mit den konvergierten Positionen
        self._classify_road_tiers()
        _zeit["abschluss"] = _t.perf_counter() - _a

        _log = logging.getLogger("Pipeline")
        _log.info("--- settlement.plot_nodes Physik: %d Iterationen, %.3fs ---",
                  self.iteration, _t_schleife + _zeit["abschluss"])
        if verlauf:
            # Der VERLAUF der groessten Verschiebung, nicht nur ihr Endwert.
            # Faellt sie und stagniert, ist CONVERGENCE_MAX_DISPLACEMENT zu
            # streng; springt sie, ist die Schrittweite zu gross. Das ist
            # die Frage, die das Nachfolgesystem beantworten muss.
            stichprobe = verlauf[::max(len(verlauf) // 8, 1)][:8]
            _log.info("      %-38s %s",
                      "[max. Verschiebung px]",
                      " ".join(f"{v:.3f}" for v in stichprobe)
                      + f" ... {verlauf[-1]:.3f}")
        for _n, _d2 in sorted(_zeit.items(), key=lambda x: -x[1]):
            _log.info("      %-38s %8.3fs  %5.1f%%  (%.1f ms je Iteration)",
                      _n, _d2, 100.0 * _d2 / max(_t_schleife + _zeit["abschluss"], 1e-9),
                      1000.0 * _d2 / max(self.iteration, 1))
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
            # Lauf-Durchschnitt (nicht die abklingende EMA oben) ueber die
            # gesamte Konvergenz-Simulation, fuer die Traffic-Gradient-
            # Strassenfaerbung - siehe _simulate_traffic() und
            # [[project-settlement-physics-lab-parity]].
            avg_traffic = self.ridge_traffic_sum.get(key, 0.0) / max(1, self.ridge_traffic_sample_count)

            if traffic >= self.TIER_STRASSE_THRESHOLD * self.plot_tier_factor:
                tier, classification = "strasse", "road"
            elif traffic >= self.TIER_WEG_THRESHOLD * self.plot_tier_factor:
                tier, classification = "weg", "road"
            elif traffic >= self.TIER_MIN_TRAFFIC * self.plot_tier_factor:
                tier, classification = "pfad", "path"
            else:
                tier, classification = "none", "none"

            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            edges[edge_id] = PlotEdge(
                edge_id=edge_id, node_a=pid_i, node_b=pid_j, length=length,
                height_cost=float(cost - length), movement_cost=float(cost),
                traffic=float(traffic), classification=classification,
                traffic_avg=float(avg_traffic))
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

        # Map-Groessen-Skalierungsfaktoren (siehe
        # [[project-settlement-scale-invariance]]) - sicherer Default 1.0
        # (kein Effekt) fuer den Fall, dass ein Location-Objekt vor dem
        # ersten _get_prepared_settlement_inputs()-Aufruf erzeugt wird
        # (sollte in der echten Pipeline nie vorkommen); echte Werte werden
        # dort aus der tatsaechlichen Kartengroesse gesetzt.
        self.scale_factor = 1.0
        self.area_scale_factor = 1.0

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
        self.plotnodes = 200
        self.civ_influence_decay = 0.8
        self.terrain_factor_villages = 1.0
        self.road_slope_to_distance_ratio = 1.5
        self.landmark_wilderness = 0.3
        self.city_size = 0.5
        self.city_reach_factor = 0.6 + 0.5 * 3.4
        self.civ_influence_range = 0.15 + 0.5 * 0.30
        self.plot_intercity_traffic = 10.0 + 0.5 * 40.0
        self.plot_base_spacing = 20.0
        self.plot_civ_spacing_factor = 8.0
        self.plot_height_cost_factor = 3.0

        # Plot Physics - Advanced (siehe [[project-settlement-physics-lab-parity]]),
        # 1:1 aus PlotPhysicsSystem's Basiswerten uebernommen.
        self.core_plotnode_spring_stiffness = 1.2
        self.plotnode_plotnode_spring_stiffness = 1.0
        self.pressure_strength = 0.8
        self.core_mass = 1.0
        self.plot_node_mass = 1.0
        self.plot_node_repulsion_strength = 4.0
        self.plot_gravity_strength = 0.01
        self.plot_city_repulsion_strength = 0.5
        self.potential_strength = 1.0
        self.damping = 0.80
        self.plot_tier_factor = 1.0

        # Kraft-Schalter. Springs/Pressure seit 2026-08-11 aus (siehe
        # _load_default_parameters()-Kommentar fuer die volle Begruendung -
        # dieselben Werte hier nur als Fallback, falls je eine Instanz ohne
        # set_active_parameters() genutzt wird).
        self.enable_core_plotnode_spring = False
        self.enable_plotnode_plotnode_spring = False
        self.enable_pressure = False
        self.enable_plot_node_repulsion = True
        self.enable_field_cores = True
        self.enable_field_plotnodes = True
        self.enable_core_cell_containment = True
        self.enable_wilderness_containment = True

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

        # City Size (an tools/biome_lab/scene.py's _recompute_background()
        # angelehnte Ableitung, siehe [[project-settlement-physics-lab-parity]]):
        # ein Regler leitet city_reach_factor/civ_influence_range/
        # plot_intercity_traffic gemeinsam ab, statt sie einzeln zu slidern.
        #
        # WICHTIG - Basiswerte bewusst NICHT 1:1 vom Lab übernommen (siehe
        # [[project-settlement-scale-invariance]] für die volle Historie):
        # die Lab-Formel (city_reach_factor = 4.0 + city_size*6.0) war für
        # eine deutlich größere Referenz-Karte kalibriert und ließ Städte bei
        # Productions map_size=128 fast verdreifachen (5.7% -> 17.0% der
        # Kartenfläche). Erste Korrektur (Basis 1.0 statt 4.0, city_size=0.5
        # -> 4.0) reproduzierte zwar den alten Production-Default, war laut
        # Live-Test des Nutzers aber SELBST noch ~3x zu groß (Zielgröße:
        # Stadt ~0.5-1.5% der Kartenfläche). Fläche skaliert ungefähr mit
        # reach_factor^2 (empirisch bestätigt: 4.0->5.7%, 7.0->17.0%,
        # Verhältnis 17/5.7=2.98 ≈ (7/4)^2=3.06), daher Basis nochmal durch
        # sqrt(3)≈1.73 geteilt (city_size=0.5 -> 2.3 statt 4.0). Reine
        # Kalibrierungsanpassung, keine strukturelle Änderung - bei Bedarf
        # anhand von Live-Tests weiter nachjustierbar.
        self.city_size = parameters['city_size']
        self.city_reach_factor = 0.6 + self.city_size * 3.4
        self.civ_influence_range = 0.15 + self.city_size * 0.30
        self.plot_intercity_traffic = 10.0 + self.city_size * 40.0

        self.plot_base_spacing = parameters['plot_base_spacing']
        self.plot_civ_spacing_factor = parameters['plot_civ_spacing_factor']
        self.plot_height_cost_factor = parameters['plot_height_cost_factor']

        # Plot Physics - Advanced (siehe [[project-settlement-physics-lab-parity]])
        self.core_plotnode_spring_stiffness = parameters['core_plotnode_spring_stiffness']
        self.plotnode_plotnode_spring_stiffness = parameters['plotnode_plotnode_spring_stiffness']
        self.pressure_strength = parameters['pressure_strength']
        self.core_mass = parameters['core_mass']
        self.plot_node_mass = parameters['plot_node_mass']
        self.plot_node_repulsion_strength = parameters['plot_node_repulsion_strength']
        self.plot_gravity_strength = parameters['plot_gravity_strength']
        self.plot_city_repulsion_strength = parameters['plot_city_repulsion_strength']
        self.potential_strength = parameters['potential_strength']
        self.damping = parameters['damping']
        self.plot_tier_factor = parameters['plot_tier_factor']

        # Kraft-Schalter
        self.enable_core_plotnode_spring = parameters['enable_core_plotnode_spring']
        self.enable_plotnode_plotnode_spring = parameters['enable_plotnode_plotnode_spring']
        self.enable_pressure = parameters['enable_pressure']
        self.enable_plot_node_repulsion = parameters['enable_plot_node_repulsion']
        self.enable_field_cores = parameters['enable_field_cores']
        self.enable_field_plotnodes = parameters['enable_field_plotnodes']
        self.enable_core_cell_containment = parameters['enable_core_cell_containment']
        self.enable_wilderness_containment = parameters['enable_wilderness_containment']

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, _execute_generation() ohne
        injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from managers.data_lod_manager import DataLODManager
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
            # city_size ersetzt city_reach_factor/civ_influence_range als
            # Slider (beide werden jetzt in set_active_parameters() daraus
            # abgeleitet) - die beiden Konstanten bleiben als Config-Eintraege
            # bestehen (z.B. fuer Legacy-Leser dieses dicts), werden aber
            # nicht mehr fuer die Instanz-Attribute genutzt.
            'city_size': SETTLEMENT.CITY_SIZE["default"],
            'city_reach_factor': SETTLEMENT.CITY_REACH_FACTOR["default"],
            'civ_influence_range': SETTLEMENT.CIV_INFLUENCE_RANGE["default"],
            'plot_base_spacing': SETTLEMENT.PLOT_BASE_SPACING["default"],
            'plot_civ_spacing_factor': SETTLEMENT.PLOT_CIV_SPACING_FACTOR["default"],
            'plot_height_cost_factor': SETTLEMENT.PLOT_HEIGHT_COST_FACTOR["default"],
            # Plot Physics - Advanced (siehe [[project-settlement-physics-lab-parity]])
            'core_plotnode_spring_stiffness': SETTLEMENT.CORE_PLOTNODE_SPRING_STIFFNESS["default"],
            'plotnode_plotnode_spring_stiffness': SETTLEMENT.PLOTNODE_PLOTNODE_SPRING_STIFFNESS["default"],
            'pressure_strength': SETTLEMENT.PRESSURE_STRENGTH["default"],
            'core_mass': SETTLEMENT.CORE_MASS["default"],
            'plot_node_mass': SETTLEMENT.PLOT_NODE_MASS["default"],
            'plot_node_repulsion_strength': SETTLEMENT.PLOT_NODE_REPULSION_STRENGTH["default"],
            'plot_gravity_strength': SETTLEMENT.PLOT_GRAVITY_STRENGTH["default"],
            'plot_city_repulsion_strength': SETTLEMENT.PLOT_CITY_REPULSION_STRENGTH["default"],
            'potential_strength': SETTLEMENT.POTENTIAL_STRENGTH["default"],
            'damping': SETTLEMENT.DAMPING["default"],
            'plot_tier_factor': SETTLEMENT.PLOT_TIER_FACTOR["default"],
            # Kraft-Schalter (reine Booleans, kein ParameterSlider/Config-Slot)
            #
            # PHYSIK VEREINFACHT (2026-08-11, Nutzer-Vorgabe): "sobald die
            # Physik losgeht geht alles kaputt" - zwei Federn (core<->plotnode,
            # plotnode<->plotnode) und die Verkehrs-Kontraktion darauf machten
            # das System instabil. Die Verkehrs-Kontraktion (_rest_length_
            # plotnode_plotnode_batch(), schrumpft die Ruhelaenge der
            # plotnode<->plotnode-Feder anhand von ridge_traffic_shrink_ema)
            # haengt AUSSCHLIESSLICH an enable_plotnode_plotnode_spring - mit
            # der Feder aus ist sie automatisch mit weg, ohne dass ihr Code
            # geloescht werden musste. Nutzer-Vorgabe woertlich: "einfach
            # zurueck zur Abstossung der Nodes untereinander reicht" -
            # enable_plot_node_repulsion bleibt daher an, ebenso die
            # Wildnis-/Stadtgrenzen-Eindaemmung (enable_*_containment,
            # "wildnisgrenze, stadtgrenze, alles gut" - das ist die bereits
            # VORHER berechnete Grenze selbst, nur ihre physik-seitige
            # Ruecksetzkraft laeuft hier mit). enable_pressure (Flaechendruck
            # je Zelle) ebenfalls aus - eine weitere Kraft neben der
            # gewuenschten reinen Abstossung, nicht Teil der Vorgabe.
            'enable_core_plotnode_spring': False,
            'enable_plotnode_plotnode_spring': False,
            'enable_pressure': False,
            'enable_plot_node_repulsion': True,
            'enable_field_cores': True,
            'enable_field_plotnodes': True,
            'enable_core_cell_containment': True,
            'enable_wilderness_containment': True,
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
            # In erosion.slope spiegeln, weil die _calc_*-Methoden dieses
            # Generators von dort lesen (siehe
            # _get_prepared_settlement_inputs()).
            self.data_lod_manager.set_calculator_output(
                "erosion.slope", lod, {"slopemap": slopemap})
            # water.manning_flow (GEMALTE Klassifikation), nicht
            # water.flow_network (Zentrallinie) - siehe
            # core/water_generator.py._calc_manning_flow().
            self.data_lod_manager.set_calculator_output(
                "water.manning_flow", lod, {"water_biomes_map": water_map})
            self.data_lod_manager.set_calculator_output(
                "biome.integrate_layers", lod, {"biome_map": biome_map})

            # Läuft über die einzeln aufrufbaren _calc_*-Methoden (siehe
            # managers/calculator_graph.py - Settlement-Calculator-Knoten
            # #28-#34 aus docs/generation_pipeline_dependencies.md). Die echte
            # GUI-Pipeline (GenerationOrchestrator) ruft dieselben Methoden ab jetzt
            # einzeln über den globalen CalculatorDispatcher auf (Tracker #16
            # LOD-Lockstep-Umbau) - nur #34 (plot_nodes) braucht biome_map, die
            # anderen 6 Phasen können unabhängig von Biome starten.
            for calculator_id in (
                "settlement.suitability", "settlement.settlements", "settlement.city_boundary",
                "settlement.city_blocks", "settlement.landscape_voronoi", "settlement.pathfinding",
                "settlement.roadsites", "settlement.civ_influence",
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
        sea_roads = self.data_lod_manager.get_calculator_output(
            "settlement.pathfinding", "sea_roads", lod_level)
        landmark_roads = self.data_lod_manager.get_calculator_output(
            "settlement.landmark_roads", "landmark_roads", lod_level)
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
        potential_field = self.data_lod_manager.get_calculator_output(
            "settlement.plot_nodes", "potential_field", lod_level)
        plot_cores = self.data_lod_manager.get_calculator_output("settlement.plot_nodes", "plot_cores", lod_level)
        wilderness_polygons = self.data_lod_manager.get_calculator_output(
            "settlement.plot_nodes", "wilderness_polygons", lod_level)
        plot_node_positions = self.data_lod_manager.get_calculator_output(
            "settlement.plot_nodes", "plot_node_positions", lod_level)

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
        settlement_data.sea_roads = sea_roads if sea_roads is not None else []
        settlement_data.landmark_roads = landmark_roads if landmark_roads is not None else []
        settlement_data.roadsite_list = roadsite_list if roadsite_list is not None else []
        settlement_data.civ_map = civ_map
        settlement_data.landmark_list = landmark_list if landmark_list is not None else []
        settlement_data.plot_nodes = plot_nodes if plot_nodes is not None else []
        settlement_data.plots = plots if plots is not None else []
        settlement_data.plot_map = plot_map
        settlement_data.plot_edges = plot_edges if plot_edges is not None else {}
        settlement_data.potential_field = potential_field
        settlement_data.plot_cores = plot_cores if plot_cores is not None else []
        settlement_data.wilderness_polygons = wilderness_polygons if wilderness_polygons is not None else []
        settlement_data.plot_node_positions = plot_node_positions if plot_node_positions is not None else []

        settlement_data.terrain_suitability_valid = True
        settlement_data.settlements_valid = True
        settlement_data.road_network_valid = roads is not None
        settlement_data.roadsites_valid = roadsite_list is not None
        settlement_data.civilization_mapping_valid = True
        settlement_data.landmarks_valid = landmark_list is not None
        settlement_data.plots_valid = plot_nodes is not None

        return settlement_data

    def _is_final_lod(self, calculator_id: str, lod_level: int) -> bool:
        """
        Prüft, ob lod_level die letzte/finale Runde für DIESEN Calculator-Lauf
        ist - gemeinsame Grundlage für ALLE 10 Settlement-Calculator-Knoten
        (siehe [[project-settlement-scale-invariance]]), nicht nur
        settlement.plot_nodes wie zuvor. WICHTIG: alle 10 Knoten MÜSSEN
        dasselbe Kriterium nutzen (dieselbe true_max_lod-Quelle) - sonst könnte
        ein nachgelagerter Knoten in einer Runde "final" sein, in der ein
        vorgelagerter Knoten (dessen Output er liest) es noch nicht ist, und
        ein leeres Platzhalter-Ergebnis als echte Daten lesen.

        Liest das Ziel-LOD aus DataLODManager.get_calculator_target_lod() (von
        GenerationOrchestrator.request_generation() für jeden Calculator-Knoten
        gesetzt, sobald der Request gestellt wird - stabil über den gesamten
        Lauf, unabhängig vom Fortschritt anderer Generatoren).

        Frühere Version leitete "final" stattdessen aus der GERADE
        VERFÜGBAREN (noch wachsenden) Terrain-Heightmap-Größe ab
        (calculate_max_lod_for_size(aktuelle_heightmap.shape[0])) - das verglich
        de facto jede Zwischen-Runde mit sich selbst und erkannte JEDE Runde
        fälschlich als "final" (z.B. Runde 1 mit einer LOD-1-großen Heightmap:
        max_lod_for_size(32px) == 1 == lod_level, "final" also sofort wahr),
        wodurch PlotPhysicsSystem (bis zu 100 Iterationen) bei jeder LOD-Runde
        statt nur einmal am Ende komplett neu lief. Fallback auf die alte
        Heightmap-Ableitung bleibt für Aufrufe außerhalb von
        GenerationOrchestrator (z.B. Standalone-Tests) erhalten, für die nie
        ein Ziel-LOD gesetzt wurde.
        """
        target = self.data_lod_manager.get_calculator_target_lod(calculator_id)
        if target is not None:
            return lod_level >= target

        from managers.data_lod_manager import calculate_max_lod_for_size
        full_heightmap = self.data_lod_manager.get_terrain_data("heightmap")
        if full_heightmap is not None:
            true_max_lod = calculate_max_lod_for_size(full_heightmap.shape[0])
        else:
            true_max_lod = self.data_lod_manager.get_max_lod_for_map_size()
        return lod_level >= true_max_lod

    def _get_prepared_settlement_inputs(self, lod_level: int) -> Dict[str, Any]:
        """
        Holt alle Settlement-Dependencies (Terrain/Water-Outputs) für dieses LOD.
        water_map wird bewusst direkt aus water.manning_flow's water_biomes_map
        gelesen (Calculator-Graph-Ebene), nicht aus der zusammengesetzten
        Domain-Ebene (DataLODManager.get_water_data_lod('water_map')) - funktional
        äquivalent für die reine Wasser-Präsenz-Prüfung in
        TerrainSuitabilityAnalyzer.calculate_water_proximity() (prüft nur
        `water_map > 0`), aber verfügbar sobald DIESER EINE Water-Knoten fertig
        ist, ohne auf die vollständige Water-Generator-Assemblierung zu warten.
        water.manning_flow (nicht water.flow_network) liefert die GEMALTE
        Klassifikation in tatsaechlicher Flussbreite - Siedlungsnaehe soll sich
        an der echten Gewaesserflaeche orientieren, nicht an der ein Pixel
        breiten Zentrallinie (siehe core/water_generator.py._calc_manning_flow()).

        Setzt außerdem self.scale_factor/self.area_scale_factor aus der
        tatsächlichen finalen Heightmap-Größe (siehe
        [[project-settlement-scale-invariance]]) - alle Methoden, die danach im
        selben Calculator-Durchlauf aufgerufen werden (calculate_settlements/
        calculate_roadsites/calculate_landmarks/calculate_civilization_mapping),
        nutzen dieses Ambient-State für map-größen-unabhängige Radien. Diese
        Methode wird von 8 der 9 gegateten Calculator-Knoten selbst aufgerufen;
        _calc_roadsites ist die einzige Ausnahme (holt nur "roads"), bekommt
        self.scale_factor aber transitiv gesetzt, weil settlement.roadsites im
        CALCULATOR_GRAPH von settlement.pathfinding->settlement.settlements->
        settlement.suitability abhängt und _calc_suitability in derselben
        Runde IMMER zuerst läuft (siehe calculator_graph.py) - falls sich diese
        Reihenfolge je ändert, muss das hier erneut geprüft werden.
        """
        heightmap = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        # erosion.slope: Wegekosten und Baubarkeit muessen die tatsaechlichen
        # Rinnen und Kaemme kennen, nicht das unerodierte Gelaende.
        slopemap = self.data_lod_manager.get_calculator_output(
            "erosion.slope", "slopemap", lod_level)
        water_map = self.data_lod_manager.get_calculator_output(
            "water.manning_flow", "water_biomes_map", lod_level)

        missing = [name for name, value in (
            ("heightmap", heightmap), ("slopemap", slopemap), ("water_map", water_map)
        ) if value is None]
        if missing:
            raise ValueError(f"Settlement: fehlende Dependencies für LOD {lod_level}: {', '.join(missing)}")

        self.scale_factor = heightmap.shape[0] / 128.0
        self.area_scale_factor = self.scale_factor ** 2

        # Kulturregionen (2026-08-10, docs/SIEDLUNGEN_ENTWURF.md §3). OHNE
        # Pflichtpruefung: terrain.redistribution ist zwar bereits eine echte
        # Abhaengigkeit von settlement.settlements (calculator_graph.py), aber
        # nur im WELTKARTE_AKTIV-Pfad liefert sie ueberhaupt region_map -
        # calculate_settlements() faellt bei None auf eine einzige namenlose
        # Kultur zurueck (altes Verhalten).
        region_map = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "region_map", lod_level)

        # Seegrad (docs/OFFENE_PUNKTE.md 3.3, "ab Grad 1 statt ab 10 m Tiefe"):
        # gleiches OHNE-Pflichtpruefung-Muster wie region_map - nur auf der
        # Weltkarte vorhanden.
        seegrad = self.data_lod_manager.get_calculator_output(
            "terrain.redistribution", "seegrad", lod_level)

        return {"heightmap": heightmap, "slopemap": slopemap, "water_map": water_map,
                "region_map": region_map, "seegrad": seegrad}

    def _calc_suitability(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'settlement.suitability' (#28) - läuft NUR am finalen
        LOD (siehe _is_final_lod(), [[project-settlement-scale-invariance]]) -
        vorher wurde die gesamte Settlement-Kette bei JEDER LOD-Runde komplett
        neu berechnet (auf der jeweils kleineren Zwischenauflösung), sichtbar
        als "wächst in Stufen" während der Generierung. Kein anderer
        Calculator-Knoten außerhalb von settlement.* hängt von hier ab, es
        entsteht also keine Wartezeit für irgendetwas anderes.
        """
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(
                calculator_id, lod_level, {"combined_suitability_map": None})
            return
        self._update_progress("Terrain Analysis", 5, "Analyzing terrain suitability for settlements...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        suitability_map = self.calculate_terrain_suitability(
            inputs["heightmap"], inputs["slopemap"], inputs["water_map"], lod_level,
            region_map=inputs.get("region_map"))
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"combined_suitability_map": suitability_map})

    def _calc_settlements(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.settlements' (#29) - siehe _is_final_lod()."""
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"settlement_list": []})
            return
        self._update_progress("Settlement Placement", 15, "Placing settlements based on suitability...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        suitability_map = self.data_lod_manager.get_calculator_output(
            "settlement.suitability", "combined_suitability_map", lod_level)
        if suitability_map is None:
            raise ValueError(f"settlement.settlements: combined_suitability_map für LOD {lod_level} nicht verfügbar")

        # Eignungskarten je Stadttyp (2026-08-13, docs/OFFENE_PUNKTE.md 5.16).
        # Aus DENSELBEN Eingaben wie die allgemeine Standortguete - der
        # Analyzer rechnet die vier Teilfaktoren dafuer ohnehin schon.
        typ_eignungen = None
        try:
            # Derselbe Analyzer-Aufbau wie in calculate_terrain_suitability() -
            # er haelt keinen Zustand ueber den Aufruf hinaus, ein zweites
            # Exemplar ist deshalb unbedenklich und billiger als es
            # durchzureichen.
            analyzer = TerrainSuitabilityAnalyzer(
                self.terrain_factor_villages, inputs["heightmap"].shape[0])
            typ_eignungen = analyzer.stadttyp_eignungen(
                inputs["heightmap"], inputs["slopemap"], inputs["water_map"],
                self._update_progress)
        except Exception as fehler:
            # LAUT melden statt still auf "alles sonstige" zurueckzufallen -
            # ohne diese Zeile waere ein Fehler hier von einer Karte ohne
            # Sondertypen nicht zu unterscheiden (CLAUDE.md).
            self.logger.warning(
                "Stadttyp-Eignungen fehlgeschlagen (%s) - alle Orte bleiben "
                "beim Auffangtyp 'sonstige'", fehler)

        # Kostenfeld fuer die Erreichbarkeit der Marktstadt-Wahl (5.17).
        # Dasselbe Feld, das auch das Wegenetz benutzt - eine zweite
        # Kostendefinition waere eine zweite Wahrheit. Die Erreichbarkeit
        # selbst rechnet darauf grob (siehe erreichbarkeits_matrix()).
        kostenfeld = None
        try:
            kostenfeld = bau_kostenfeld(inputs["heightmap"], inputs["slopemap"],
                                        self.road_slope_to_distance_ratio)
        except Exception as fehler:
            self.logger.warning(
                "Kostenfeld fuer die Marktstadt-Erreichbarkeit nicht gebaut (%s) - "
                "es zaehlt nur die Wasserlage", fehler)

        settlement_list = self.calculate_settlements(
            suitability_map, inputs["heightmap"], lod_level,
            region_map=inputs.get("region_map"), typ_eignungen=typ_eignungen,
            kostenfeld=kostenfeld)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"settlement_list": settlement_list})

    def _calc_city_boundary(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.city_boundary' (NEU) - terrain-cost-gewichtete
        Stadtgrenze je Settlement, Grundlage fuer die Trennung Stadt-Innen (spaeteres
        Block-System) vs. Landschaft (LandscapeVoronoiSystem). Siehe _is_final_lod()."""
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(
                calculator_id, lod_level, {"city_mask": None, "city_cost_map": None})
            return
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
        Seit 2026-08-10 liefert calculate_road_network() zwei Listen: `roads`
        (Land, Gabriel-Graph + Kostenfeld + Bereitschaftstest,
        docs/SIEDLUNGEN_ENTWURF.md §4) und `sea_roads` (§4.4, fuer
        Kulturpaare ohne endlichen Landweg).
        voronoi_cell_map (früher aus settlement.landscape_voronoi, jetzt
        entfernt) entfällt ersatzlos - calculate_road_network() fällt dafür
        bereits dokumentiert auf reines Slope-Cost-Pathfinding zurück.
        Siehe _is_final_lod().
        """
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(
                calculator_id, lod_level, {"roads": [], "sea_roads": []})
            return
        self._update_progress("Road Building", 25, "Creating road networks between settlements...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        if settlement_list is None:
            raise ValueError(f"settlement.pathfinding: settlement_list für LOD {lod_level} nicht verfügbar")

        roads, sea_roads = self.calculate_road_network(
            settlement_list, inputs["heightmap"], inputs["slopemap"], lod_level, None,
            seegrad=inputs.get("seegrad"))
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"roads": roads, "sea_roads": sea_roads})

    def _calc_roadsites(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'settlement.roadsites' (#31) - siehe _is_final_lod().
        Braucht seit dem Umbau auf den 45-Arten-Katalog (docs/SIEDLUNGEN_ENTWURF.md
        §4.6) zusaetzlich heightmap (Furt-/Passerkennung) und settlement_list
        (Kulturzuordnung je Standort ueber _naechste_kultur()) - ruft dafuer
        jetzt selbst _get_prepared_settlement_inputs() statt sich wie zuvor
        nur auf das transitiv gesetzte self.scale_factor zu verlassen.
        """
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"roadsite_list": []})
            return
        self._update_progress("Roadsite Placement", 40, "Placing roadsites along roads...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        sea_roads = self.data_lod_manager.get_calculator_output(
            "settlement.pathfinding", "sea_roads", lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        if roads is None or settlement_list is None:
            raise ValueError(f"settlement.roadsites: fehlende Inputs für LOD {lod_level}")

        roadsite_list = self.calculate_roadsites(
            roads, sea_roads or [], settlement_list, inputs["heightmap"], lod_level,
            region_map=inputs.get("region_map"))
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"roadsite_list": roadsite_list})

    def _calc_civ_influence(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.civ_influence' (#32) - siehe _is_final_lod()."""
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"civ_map": None})
            return
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
        """Calculator-Node 'settlement.landmarks' (#33) - siehe _is_final_lod()."""
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"landmark_list": []})
            return
        self._update_progress("Landmark Placement", 65, "Placing landmarks in wilderness areas...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        civ_map = self.data_lod_manager.get_calculator_output("settlement.civ_influence", "civ_map", lod_level)
        settlement_list = self.data_lod_manager.get_calculator_output(
            "settlement.settlements", "settlement_list", lod_level)
        if civ_map is None or settlement_list is None:
            raise ValueError(f"settlement.landmarks: fehlende Inputs für LOD {lod_level}")

        landmark_list = self.calculate_landmarks(
            civ_map, inputs["heightmap"], inputs["slopemap"], inputs["water_map"],
            settlement_list, lod_level, region_map=inputs.get("region_map"))
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"landmark_list": landmark_list})

    def _calc_landmark_roads(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'settlement.landmark_roads' (NEU) - deterministische
        Dijkstra-Anbindung jedes Landmarks an den nächstgelegenen Punkt des
        Hauptstraßennetzes (Nutzer-Vorgabe: kein Zufallsmechanismus in Phase 1,
        das dekorative Zusatz-Wegenetz kommt erst in Phase 2). Siehe _is_final_lod()."""
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"landmark_roads": []})
            return
        self._update_progress("Landmark Roads", 68, "Connecting landmarks to road network...")
        inputs = self._get_prepared_settlement_inputs(lod_level)
        landmark_list = self.data_lod_manager.get_calculator_output(
            "settlement.landmarks", "landmark_list", lod_level)
        roads = self.data_lod_manager.get_calculator_output("settlement.pathfinding", "roads", lod_level)
        if landmark_list is None or roads is None:
            raise ValueError(f"settlement.landmark_roads: fehlende Inputs für LOD {lod_level}")

        landmark_roads = self.calculate_landmark_roads(
            landmark_list, roads, inputs["heightmap"], inputs["slopemap"], lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"landmark_roads": landmark_roads})

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
        keine echte Wartezeit für irgendetwas anderes. Siehe _is_final_lod()
        (jetzt von allen 10 Settlement-Knoten geteilt, nicht mehr nur diesem).
        """
        if not self._is_final_lod(calculator_id, lod_level):
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
                "plot_nodes": [], "plots": [], "plot_map": None, "plot_edges": {},
                "plot_node_positions": [], "plot_cores": [], "wilderness_polygons": [],
                "potential_field": None,
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

        # plot_nodes_count BEWUSST NICHT zusaetzlich mit area_scale_factor
        # hochskaliert (frueherer Versuch, per Messung als echter Bug
        # verworfen, siehe [[project-settlement-scale-invariance]]):
        # plot_base_spacing skaliert bereits linear mit der Kartengroesse
        # (siehe PlotPhysicsSystem.__init__), wodurch dieselbe Node-ANZAHL
        # bei groesserer Karte automatisch proportional groessere Zellen mit
        # derselben relativen Dichte ergibt - EIN Skalierungs-Hebel reicht.
        # Eine ZUSAETZLICHE quadratische Vervierfachung der Node-Anzahl (bei
        # map_size=256 z.B. 210->840) ueberfuellte den verfuegbaren Platz so
        # stark, dass die Feder-/Abstossungsphysik selbst mit deutlich mehr
        # Iterationen (400 statt 100, getestet) nicht mehr konvergierte -
        # Naechster-Nachbar-Abstand kollabierte auf ~7% des Ziel-Abstands
        # statt der ueblichen ~30% (leicht unterkonvergiert ist normal, siehe
        # Baseline-Messung bei 128px/200 Nodes) - sichtbar als chaotisches,
        # ueberfuelltes Wegenetz ("komplett zerschossen").
        plot_system = PlotPhysicsSystem(
            map_size=height, plot_nodes_count=self.plotnodes, plot_base_spacing=self.plot_base_spacing,
            plot_civ_spacing_factor=self.plot_civ_spacing_factor,
            plot_height_cost_factor=self.plot_height_cost_factor,
            core_plotnode_spring_stiffness=self.core_plotnode_spring_stiffness,
            plotnode_plotnode_spring_stiffness=self.plotnode_plotnode_spring_stiffness,
            pressure_strength=self.pressure_strength, core_mass=self.core_mass,
            plot_node_mass=self.plot_node_mass,
            plot_node_repulsion_strength=self.plot_node_repulsion_strength,
            plot_gravity_strength=self.plot_gravity_strength,
            plot_city_repulsion_strength=self.plot_city_repulsion_strength,
            potential_strength=self.potential_strength, damping=self.damping,
            plot_tier_factor=self.plot_tier_factor,
            enable_core_plotnode_spring=self.enable_core_plotnode_spring,
            enable_plotnode_plotnode_spring=self.enable_plotnode_plotnode_spring,
            enable_pressure=self.enable_pressure,
            enable_plot_node_repulsion=self.enable_plot_node_repulsion,
            enable_field_cores=self.enable_field_cores,
            enable_field_plotnodes=self.enable_field_plotnodes,
            enable_core_cell_containment=self.enable_core_cell_containment,
            enable_wilderness_containment=self.enable_wilderness_containment,
            shader_manager=self.shader_manager,
            progress_callback=self._update_progress, map_seed=self.map_seed,
            live_state_callback=self.live_plot_callback)
        plot_system.plot_intercity_traffic = self.plot_intercity_traffic
        ok = plot_system.generate(inputs["heightmap"], inputs["slopemap"], civ_map, city_mask, settlement_list)

        if not ok:
            self.logger.warning(f"settlement.plot_nodes: PlotPhysicsSystem.generate() fehlgeschlagen für LOD {lod_level}")
            self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {
                "plot_nodes": [], "plots": [], "plot_map": None, "plot_edges": {},
                "plot_node_positions": [], "plot_cores": [], "wilderness_polygons": [],
                "potential_field": None,
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
             "plot_cores": plot_system.nodes, "wilderness_polygons": wilderness_polygons,
             # Fuer die neue Potenzialfeld-Anzeige (Checkbox in settlement_tab.py,
             # siehe [[project-settlement-physics-lab-parity]]) - vorher nie nach
             # aussen exportiert, wurde nach generate() intern verworfen.
             "potential_field": plot_system.potential_field})

    def calculate_terrain_suitability(self, heightmap, slopemap, water_map, lod,
                                       reachability_map=None, region_map=None):
        """
        Fuenf-Faktor-Eignungsfeld (docs/SIEDLUNGEN_ENTWURF.md §2). `reachability_map`
        bleibt None fuer die erste Platzierungsrunde (Faktor neutral) - siehe
        TerrainSuitabilityAnalyzer.create_combined_suitability().

        Multipliziert danach mit `_randfaktor()` (docs/OFFENE_PUNKTE.md 5.14) -
        eine weiche Absenkung nahe der Kastengrenze des 3x3-Ausschnittsgitters,
        nur wenn `region_map` vorliegt (WELTKARTE_AKTIV).
        """
        analyzer = TerrainSuitabilityAnalyzer(self.terrain_factor_villages, heightmap.shape[0])
        combined_suitability = analyzer.create_combined_suitability(
            heightmap, slopemap, water_map, reachability_map=reachability_map,
            progress_callback=self._update_progress)
        randfaktor = self._randfaktor(heightmap.shape[0], region_map)
        if randfaktor is not None:
            combined_suitability = combined_suitability * randfaktor
        return combined_suitability

    def _randfaktor(self, size, region_map):
        """
        Weiche Randstrafe zur Kastengrenze des 3x3-Ausschnittsgitters
        (docs/OFFENE_PUNKTE.md 5.14, urspr. TODO §D2 Vorschlag 2: "Abschlag im Eignungswert in den
        letzten ~200m vor der Kastengrenze. Er entscheidet nur bei sonst
        gleichwertigen Plaetzen. Ein hartes Verbot waere schaedlich: Staedte
        sollen an Fluessen und Kuesten liegen, und eine Gitterlinie laeuft
        davon voellig unabhaengig."). Faktor 0.5 direkt auf der Linie, linear
        auf 1.0 ab 200m Abstand - eine Absenkung, kein Ausschluss.

        None wenn `region_map` fehlt (alter Nicht-Weltkarten-Pfad ohne
        Gitter) - dasselbe Signal, das schon `calculate_settlements()` fuer
        denselben Fall benutzt.
        """
        if region_map is None:
            return None
        import core.terrain_weltkarte as rw
        linien = np.asarray(rw.gitterlinien_px(size))
        achse = np.arange(size, dtype=np.float64)
        abstand_1d = np.min(np.abs(achse[:, None] - linien[None, :]), axis=1)
        abstand_px = np.minimum(abstand_1d[None, :], abstand_1d[:, None])
        mpp = rw.WELT_KM * 1000.0 / size
        abstand_m = abstand_px * mpp
        schwelle_m = 200.0
        return (0.5 + 0.5 * np.clip(abstand_m / schwelle_m, 0.0, 1.0)).astype(np.float32)

    def _grenzabstand_m(self, x, y, size):
        """Abstand eines einzelnen Punkts zur naechsten Kastengrenze, in
        Metern - fuer die Roadsite-/Landmark-Kandidatenreihung (5.14)."""
        import core.terrain_weltkarte as rw
        linien = rw.gitterlinien_px(size)
        dx = min(abs(x - l) for l in linien)
        dy = min(abs(y - l) for l in linien)
        mpp = rw.WELT_KM * 1000.0 / size
        return min(dx, dy) * mpp

    def _fern_zuerst(self, punkte, size, schwelle_m=200.0, xy=lambda p: (p[0], p[1])):
        """
        Stabile Umsortierung: Kandidaten mit >= schwelle_m Randabstand zuerst,
        sonst unveraendert (innerhalb jeder Gruppe bleibt die vorherige,
        bereits zufaellige Reihenfolge erhalten). Genau die "weiche Strafe,
        kein Verbot" aus 5.14 fuer Punktlisten (statt eines Eignungsfeld-
        Faktors wie bei Siedlungen): naeher-am-Rand wird nur nachrangig, nicht
        gestrichen - fehlen ferne Kandidaten, werden die nahen trotzdem
        benutzt. `xy` holt (x, y) aus einem Listenelement - Vorgabe passt fuer
        (x, y)-Paare (Roadsites), Landmarks brauchen (kategorie, x, y).
        """
        def _fern(p):
            x, y = xy(p)
            return self._grenzabstand_m(x, y, size) < schwelle_m
        return sorted(punkte, key=_fern)

    def _rang_zuweisen(self, werte_mit_rauschen):
        """
        Rang je Settlement EINER Kultur, docs/SIEDLUNGEN_ENTWURF.md §3.

        Der beste Wert wird immer 'stadt' - das ist genau "mindestens ein Ort
        je Kultur ist Stadt". Vom Rest (r = n-1 Orte) wird die obere Haelfte
        'siedlung', der Rest 'dorf':
            r=1 (n=2): 0 siedlung, 1 dorf      -> [stadt, dorf]
            r=2 (n=3): 1 siedlung, 1 dorf      -> [stadt, siedlung, dorf]
            r=3 (n=4): 1 siedlung, 2 dorf
            r=4 (n=5): 2 siedlung, 2 dorf
        `werte_mit_rauschen` ist der Eignungswert am Standort PLUS Rauschen
        (§2: "Gezogen wird mit Rauschen um den Wert herum, sodass gelegentlich
        ein Dorf an bester Lage sitzt und eine Stadt an mittelmaessiger") - der
        Rang folgt also nicht dem rohen Eignungswert, sondern dieser bereits
        verrauschten Fassung.
        """
        n = len(werte_mit_rauschen)
        reihenfolge = sorted(range(n), key=lambda i: -werte_mit_rauschen[i])
        raenge = [""] * n
        if n == 0:
            return raenge
        raenge[reihenfolge[0]] = "stadt"
        rest = reihenfolge[1:]
        siedlung_anzahl = len(rest) // 2
        for i, idx in enumerate(rest):
            raenge[idx] = "siedlung" if i < siedlung_anzahl else "dorf"
        return raenge

    def _typen_zuweisen(self, orte, typ_eignungen, raenge, kostenfeld=None):
        """
        Weist den Orten EINER Kultur ihre Stadttypen zu und liefert die
        (moeglicherweise angepassten) Raenge zurueck.

        Reihenfolge, und warum:
        1. **Marktstadt zuerst.** Sie ist auf eine je Region beschraenkt
           ("nur eine Marktstadt pro Region moeglich") - waere sie ein
           gewoehnlicher Kandidat, koennten zwei Orte sie gleichzeitig
           beanspruchen. Es gewinnt der Ort mit der hoechsten
           Marktstadt-Eignung an seiner Position, und nur wenn diese ueber
           TYP_GRUNDGUETE liegt: eine Region ohne brauchbare Hafenlage
           bekommt lieber gar keine Marktstadt als eine schlechte.
        2. **Alle uebrigen** nehmen den Typ mit der hoechsten Eignung an ihrem
           Ort. "sonstige" ist dabei ein normaler Mitbewerber mit fester
           Guete - gewinnt er, passte schlicht kein Sondertyp.
        3. **Rang gegen den Typ pruefen.** `rang_erlaubt` je Typ setzt die
           Vorgabe "Bergdorf klein bis mittel" / "Marktstadt mittel bis gross"
           durch. Ein zu hoher Rang wird auf den hoechsten erlaubten gesenkt,
           ein zu niedriger auf den niedrigsten erlaubten angehoben.
           **Die Marktstadt ist davon ausgenommen, ihren Rang zu VERLIEREN** -
           sie ist der Handelsknoten der Region und soll nicht als Dorf enden.
        """
        if not orte:
            return raenge

        raenge = list(raenge)
        offen = list(range(len(orte)))

        def eignung_bei(typ, ort):
            karte = typ_eignungen.get(typ)
            if karte is None:
                return 0.0
            yi = int(np.clip(round(ort.y), 0, karte.shape[0] - 1))
            xi = int(np.clip(round(ort.x), 0, karte.shape[1] - 1))
            return float(karte[yi, xi])

        # 1. Marktstadt - hoechstens eine, und nur bei brauchbarer Lage.
        #
        # "liegt am Wasser ODER kann viele Staedte gut erreichen" - das ODER
        # ist woertlich gemeint, deshalb das MAXIMUM aus beidem und kein
        # Produkt: eine Binnenstadt am Knotenpunkt aller Wege ist ebenso eine
        # Marktstadt wie ein Hafen abseits der Hauptrouten.
        #
        # Die Erreichbarkeit kommt aus der Wegkosten-Matrix zwischen den
        # bereits gesetzten Orten (erreichbarkeits_matrix()/zentralitaet(),
        # docs/OFFENE_PUNKTE.md 5.17) - nicht aus einer Gelaendenaeherung.
        # Fehlt sie (kein Kostenfeld uebergeben), bleibt es bei der reinen
        # Wasserlage; das ist dann eine schwaechere, aber nicht falsche
        # Bewertung.
        zentral = None
        if kostenfeld is not None and len(orte) > 1:
            try:
                matrix = erreichbarkeits_matrix(
                    kostenfeld, [(o.x, o.y) for o in orte])
                zentral = zentralitaet(matrix)
            except Exception as fehler:
                self.logger.warning(
                    "Erreichbarkeit fuer die Marktstadt-Wahl fehlgeschlagen (%s) - "
                    "es zaehlt nur die Wasserlage", fehler)

        beste_markt, bester_wert = None, TYP_GRUNDGUETE
        for i in offen:
            wasserlage = eignung_bei("marktstadt", orte[i])
            wert = wasserlage
            if zentral is not None:
                wert = max(wasserlage, float(zentral[i]))
            if wert > bester_wert:
                beste_markt, bester_wert = i, wert
        if beste_markt is not None:
            orte[beste_markt].settlement_type = "marktstadt"
            offen.remove(beste_markt)

        # 2. Alle uebrigen: bester Typ an ihrem Ort (ohne Marktstadt)
        for i in offen:
            kandidaten = [(eignung_bei(typ, orte[i]), typ)
                          for typ in ("bergdorf", "agrarstadt", "sonstige")]
            kandidaten.sort(key=lambda p: -p[0])
            orte[i].settlement_type = kandidaten[0][1]

        # 3. Rang an den Typ anpassen
        for i, ort in enumerate(orte):
            erlaubt = STADTTYPEN[ort.settlement_type]["rang_erlaubt"]
            if raenge[i] in erlaubt:
                continue
            if ort.settlement_type == "marktstadt":
                # nur anheben, nie senken (siehe Docstring) - auf den
                # NIEDRIGSTEN erlaubten Rang, nicht direkt auf den hoechsten
                # (Ticket #84: erlaubt[-1] erzeugte eine zweite "stadt" je
                # Kultur, weil _rang_zuweisen() bereits zuvor genau einen
                # Ort gekroent hatte).
                raenge[i] = erlaubt[0] if RANG_ZAHL.get(raenge[i], 1) < RANG_ZAHL[erlaubt[0]] \
                    else raenge[i]
                if raenge[i] not in erlaubt:
                    raenge[i] = erlaubt[0]
            elif RANG_ZAHL.get(raenge[i], 1) > RANG_ZAHL[erlaubt[-1]]:
                raenge[i] = erlaubt[-1]
            else:
                raenge[i] = erlaubt[0]
        return raenge

    def calculate_settlements(self, suitability_map, heightmap, lod, region_map=None,
                               typ_eignungen=None, kostenfeld=None):
        """
        Platziert Settlements je Kultur (docs/SIEDLUNGEN_ENTWURF.md §2+3).

        ANZAHL JE KULTUR: 2 bis 5, abgeleitet aus der Eignungssumme der
        Kulturregion verglichen mit der bestausgestatteten Region ("die Summe
        der Eignung ueber der Region, verglichen mit allen neun"):

            norm_c = Eignungssumme(Kultur c) / max(Eignungssumme ueber alle Kulturen)
            anzahl_c = round(2 + 3 * norm_c)     # 2 bei norm=0, 5 bei norm=1

        Der `settlements`-Regler (frueher eine absolute Gesamtzahl) wirkt jetzt
        als MULTIPLIKATOR auf diese Ableitung, neutral bei seiner Vorgabe 3
        (siehe gui/config/value_default.py SETTLEMENT.SETTLEMENTS) - der
        Regler bleibt drehbar, bestimmt aber nicht mehr die Zahl direkt.

        OHNE region_map (alter Nicht-Weltkarten-Pfad, WELTKARTE_AKTIV=False):
        eine einzige namenlose Kultur, `self.settlements` Orte direkt - das
        alte Verhalten bleibt fuer diesen Pfad erhalten.
        """
        # KOPIE, NICHT DAS ORIGINAL. `_reduce_suitability_around_point` mutiert
        # das Array in-place - ohne Kopie waere das der gecachte Calculator-
        # Output "settlement.suitability"/combined_suitability_map selbst
        # (Python/NumPy reichen Arrays per Referenz), der dann fuer jede
        # spaetere Anzeige/Wiederverwendung schon mit Loechern um jede
        # Siedlung herum daestuende. War schon in der Vorlage so angelegt,
        # hier beim Umbau auf mehrere Kulturschleifen mit-behoben.
        suitability_map = suitability_map.copy()
        height, width = heightmap.shape
        zufall_s = self._knoten_zufall("settlements")

        if region_map is None:
            kulturen = [("", np.ones((height, width), dtype=bool),
                        max(1, int(round(self.settlements))))]
        else:
            import core.terrain_weltkarte as rw
            region_map = np.asarray(region_map)
            summen = {}
            masken = {}
            for i, (_z, _s, r) in enumerate(rw.alle_regionen()):
                maske = (region_map == i) & (heightmap > 0.0)
                if not np.any(maske):
                    continue
                masken[i] = maske
                summen[i] = float(suitability_map[maske].sum())

            hoechste_summe = max(summen.values()) if summen else 0.0
            multiplikator = self.settlements / 3.0
            kulturen = []
            for i, summe in summen.items():
                norm_c = summe / hoechste_summe if hoechste_summe > 0 else 0.0
                anzahl = int(round(2 + 3 * norm_c))
                anzahl = int(np.clip(round(anzahl * multiplikator), 2, 5))
                name = rw.alle_regionen()[i][2]["volk"]
                kulturen.append((name, masken[i], anzahl))
            # Feste Reihenfolge (Regionsindex) statt Dict-Iterationsreihenfolge -
            # sonst haengt die Platzierungsreihenfolge (und damit, wer sich wem
            # gegenueber den Mindestabstand sichert) am Python-Hash statt am Seed.
            kulturen.sort(key=lambda k: k[0])

        gesamt_anzahl = sum(a for _n, _m, a in kulturen) or 1

        settlements = []
        for kultur_name, kultur_maske, anzahl_ziel in kulturen:
            # MINDESTABSTAND JE KULTUR, NICHT EIN GEMEINSAMER GLOBALER.
            #
            # Ein einziger, aus der GESAMTzahl aller Orte auf der Karte
            # abgeleiteter Mindestabstand liess kleine Regionen ihr Ziel
            # verfehlen: gemessen bei 96 px, Seed 12345, bekam Italiener/
            # Macchia (klein, kuestennah) nur 1 statt der vorgesehenen 2-5
            # Orte - der Abstand war fuer die GANZE Karte bemessen, nicht fuer
            # diese eine, kleinere Flaeche. Jede Kultur bekommt stattdessen
            # ihren eigenen Abstand aus der WURZEL ihrer eigenen Landflaeche -
            # eine kleine Region lässt ihre Orte enger stehen.
            kultur_flaeche = float(np.count_nonzero(kultur_maske))
            min_distance = max(4.0, np.sqrt(kultur_flaeche) / (anzahl_ziel + 1))

            neue_dieser_kultur = []
            # Erreicht die Kultur ihr Ziel bei diesem Abstand nicht (Region zu
            # klein/zerklueftet fuer so viele Orte), wird der Abstand bis zu
            # zweimal halbiert, bevor mit weniger als dem Ziel weitergemacht
            # wird - lieber ein kleinerer, aber tatsaechlich erreichter
            # Abstand als stillschweigend zu wenige Orte.
            for _versuch in range(3):
                if len(neue_dieser_kultur) >= anzahl_ziel:
                    break
                # SPERRMASKE EINMAL JE min_distance AUFBAUEN, DANACH NUR NOCH
                # LOKAL ERWEITERN (2026-08-11, Pipeline-Audit-Befund).
                #
                # Vorher baute _find_best_settlement_positions bei JEDEM
                # Versuch die Sperrflaeche neu auf - eine Python-Schleife
                # ueber ALLE bisher platzierten Siedlungen (karteweit, nicht
                # nur diese Kultur), die je Siedlung ein GANZES (H,W)-Array
                # anlegte. Gemessen: 17.5x langsamer bei 256->512px (4x mehr
                # Pixel) statt der erwarteten ~4x - die wiederholte
                # Vollraster-Allokation je Siedlung UND je Versuch summierte
                # sich quadratisch mit der Siedlungszahl. `settlement.
                # settlements` brauchte dadurch 3.37s statt 0.23s bei 512px.
                # Die Sperrflaeche einer Siedlung ist aber, genau wie bei
                # _reduce_suitability_around_point(), nur eine lokal
                # begrenzte Kreisscheibe - denselben Trick wendet jetzt auch
                # diese Maske an: einmalig je min_distance aus allen
                # bisherigen Siedlungen aufbauen (billig, da lokal begrenzt),
                # danach pro neu platzierter Siedlung nur noch EIN lokales
                # Update statt eines kompletten Neuaufbaus.
                gesperrt_mask = np.zeros(suitability_map.shape, dtype=bool)
                for s in settlements:
                    self._markiere_gesperrt(gesperrt_mask, s.x, s.y, min_distance)
                attempts, max_attempts = 0, anzahl_ziel * 20
                while len(neue_dieser_kultur) < anzahl_ziel and attempts < max_attempts:
                    attempts += 1
                    best_positions = self._find_best_settlement_positions(
                        suitability_map, gesperrt_mask, erlaubt_mask=kultur_maske)
                    if not best_positions:
                        break
                    x, y = zufall_s.choice(best_positions[:min(10, len(best_positions))])

                    # Rang-Rauschen (§2) direkt am gewaehlten Standort - "an
                    # bester Lage" heisst hoechster Eignungswert, das Rauschen
                    # kann das verschieben, ohne die Lage selbst zu aendern.
                    rang_wert = float(suitability_map[int(y), int(x)]) + zufall_s.uniform(-0.25, 0.25)

                    platzhalter = Location(
                        location_id=self.next_location_id, x=float(x), y=float(y),
                        location_type='settlement', radius=0.0, civ_influence=0.8,
                        properties={'rang_wert': rang_wert}, culture=kultur_name)
                    self.next_location_id += 1
                    neue_dieser_kultur.append(platzhalter)
                    settlements.append(platzhalter)
                    # Nur die neue Siedlung lokal in die Sperrmaske eintragen,
                    # nicht sie komplett neu aufbauen (siehe Kommentar oben).
                    self._markiere_gesperrt(gesperrt_mask, x, y, min_distance)
                    # UNTERDRUECKUNGSRADIUS GROESSER ALS DER MINDESTABSTAND
                    # (2026-08-10, Nutzer-Vorgabe: "die staedte gleichmaessiger
                    # verteilen, etwas weniger geklumpt"). Gemessen VORHER:
                    # mittlerer Nachbarabstand nur 0.76x dessen, was eine
                    # gleichmaessige Verteilung ergaebe (5 von 9 Kulturen unter
                    # 0.65x - deutlich geklumpt). Grund: _reduce_suitability_
                    # around_point() liess die Eignung ausserhalb von
                    # min_distance komplett unberuehrt - bei einer Region mit
                    # EINEM starken Eignungshuegel blieb direkt ausserhalb des
                    # Ausschlussradius noch reichlich hohe Eignung uebrig, und
                    # der naechste Ort setzte sich an den Rand genau dieses
                    # Huegels statt in einen anderen Teil der Region - eine
                    # Perlenkette am Huegelrand statt einer Streuung. Die
                    # harte Ausschlusszone (min_distance, fuer die
                    # Ziel-Trefferquote wichtig) bleibt unveraendert; nur die
                    # WEICHE Eignungs-Absenkung wirkt jetzt ueber einen
                    # groesseren Radius, sodass die Suche nach dem naechsten
                    # Ort tatsaechlich in einen anderen Bereich ausweicht.
                    self._reduce_suitability_around_point(
                        suitability_map, x, y, min_distance * 2.5)

                    if self._update_progress:
                        progress = 15 + (len(settlements) * 10) // gesamt_anzahl
                        self._update_progress(
                            "Settlement Placement", progress,
                            f"Placed {len(settlements)}/{gesamt_anzahl} settlements")
                min_distance = max(2.0, min_distance * 0.5)

            # Rang erst, wenn ALLE Orte dieser Kultur stehen - er ist eine
            # Aussage ueber die Kultur als Ganzes ("bester Ort wird Stadt"),
            # nicht ueber einen einzelnen Platzierungsschritt.
            raenge = self._rang_zuweisen([s.properties['rang_wert'] for s in neue_dieser_kultur])

            # STADTTYPEN (2026-08-13, docs/OFFENE_PUNKTE.md 5.16). Erst hier,
            # wenn alle Orte dieser Kultur stehen - "nur eine Marktstadt pro
            # Region" ist eine Aussage ueber die Gruppe, nicht ueber einen
            # einzelnen Ort. Der Typ kann den Rang anschliessend noch
            # verschieben (ein Bergdorf darf keine 'stadt' sein).
            if typ_eignungen:
                raenge = self._typen_zuweisen(neue_dieser_kultur, typ_eignungen,
                                              raenge, kostenfeld=kostenfeld)

            for settlement, rang in zip(neue_dieser_kultur, raenge):
                lo, hi = RANG_HAEUSER[rang]
                haeuser = int(round(zufall_s.uniform(lo, hi)))
                groesse01 = (haeuser - 15) / (50 - 15)
                settlement.rank = rang
                settlement.house_count = haeuser
                # (3 + groesse01*2) haelt denselben Radienbereich wie die
                # fruehere, vom Rang unabhaengige Zufallsstreuung - jetzt aus
                # der tatsaechlichen Haeuserzahl, nicht danaben her gewuerfelt.
                settlement.radius = (3 + groesse01 * 2) * self.scale_factor
                settlement.civ_influence = {"dorf": 0.6, "siedlung": 0.7, "stadt": 0.8}[rang]

        return settlements

    def calculate_road_network(self, settlements, heightmap, slopemap, lod, voronoi_cell_map=None,
                               seegrad=None):
        """
        Wegenetz nach docs/SIEDLUNGEN_ENTWURF.md §4.1-§4.3. Ablauf, in dieser
        Reihenfolge:

          1. KOSTENFELD (§4.1) - einmal, ueber bau_kostenfeld(). Wasser in drei
             Stufen, Hangkosten quadratisch, ein bestehender Weg verbilligt
             sich selbst (WEGERABATT) - "Wege buendeln sich zu Hauptstrecken".
          2. GABRIEL-GRAPH (§4.2) - welche Ortspaare ueberhaupt Kandidaten
             sind: A-B nur, wenn im Kreis ueber ihrer Verbindungsstrecke kein
             dritter Ort liegt. Kein Stern, keine Vollverknuepfung.
          3. BEREITSCHAFTSTEST (§4.3) je Kandidat, absteigend nach
             Bereitschaft abgearbeitet (die eifrigsten Verbindungen zuerst -
             sie werden ohnehin fast immer gebaut und damit zur Haupttrasse,
             auf die sich schwaechere Kandidaten per Wegerabatt aufbuendeln
             koennen):

                Bereitschaft = Rang(A) * Rang(B) * (gleiche Kultur ? 1.0 : 0.45)
                Wegkosten    = Pfadkosten-Summe / Luftlinienabstand

             `Wegkosten` ist damit dimensionslos und mit `Bereitschaft`
             vergleichbar - ein flaches, einfaches Stueck Land liegt nahe 1.0,
             ein Gebirge oder Wasser treibt es weit darueber. Gebaut wird bei
             Bereitschaft > Wegkosten. Rang zaehlt dorf=1/siedlung=2/stadt=3,
             sodass zwei Staedte (9.0) praktisch immer verbinden, zwei Doerfer
             verschiedener Kultur (0.45) fast nie - genau die vom Entwurf
             genannten Faelle.
          4. KULTURZUSAMMENHANG (§4.3, Ausnahme) - je Kultur wird geprueft, ob
             ihre Orte nach Schritt 3 EINEN zusammenhaengenden Teilgraphen
             bilden. Falls nicht, werden die guenstigsten fehlenden
             Verbindungen nachgetragen, unabhaengig von der Bereitschaft -
             "egal was sie kosten". Findet sich dabei KEIN endlich teurer
             Landweg (Wasser trennt die Komponenten vollstaendig), wird
             stattdessen ein SEEWEG versucht (§4.4) - siehe bau_seekostenfeld().

        `voronoi_cell_map` wie bisher optional fuer den Randbias entlang von
        Landschafts-Voronoi-Zellgrenzen. `seegrad` (docs/OFFENE_PUNKTE.md 3.3)
        steuert das Seeweg-Kostenfeld und die Tiefwasser-Auflage aus §4.4 -
        "ab Grad 1" statt "ab 10 m Tiefe", siehe bau_seekostenfeld()/
        _seeweg_anteil_tief(). None faellt auf die alte Hoehenschwelle zurueck.

        Returns: (roads, sea_roads) - je List[List[Tuple]]. Seewege getrennt
        zurueckgegeben, weil sie "anders gezeichnet werden - gestrichelt, in
        einem eigenen Blau" (§4.4), nicht weil sie technisch etwas anderes
        waeren.
        """
        if len(settlements) < 2:
            return [], []

        edge_distance_map = None
        if voronoi_cell_map is not None:
            edge_distance_map = _voronoi_edge_distance_map(voronoi_cell_map)

        # Teilschritt-Messung (managers/teilschritte.py). 60.3 s auf einer
        # Zeile im Pipeline-Log liessen offen, ob die A*-Laeufe, das
        # Kostenfeld oder der Bedarfsausbau die Zeit fressen. Zusaetzlich
        # zaehlt `_a_stern_zaehler` die einzelnen Routen mit - ohne die Zahl
        # laesst sich nicht sagen, ob ein Lauf teuer ist oder es viele sind.
        from managers.teilschritte import Teilschritte, schritt as _s
        _ts = Teilschritte("settlement.pathfinding",
                           fortschritt=self._update_progress, von=25, bis=90,
                           plan=[("kostenfeld", 2.0), ("gabriel_kandidaten", 1.0),
                                 ("routen_bewerten", 70.0), ("seekostenfeld", 2.0),
                                 ("kulturzusammenhang", 15.0),
                                 ("bedarfsausbau", 10.0)])
        self._a_stern_zaehler = [0, 0.0]

        with _s(_ts, "kostenfeld", "Kostenfeld"):
            basis_kostenfeld = bau_kostenfeld(heightmap, slopemap, self.road_slope_to_distance_ratio)
        weg_maske = np.zeros(basis_kostenfeld.shape, dtype=bool)

        def route(a, b):
            """(Pfad, Pfadkosten) fuer ein Ortspaar - nutzt den aktuellen Wegerabatt."""
            import time as _t
            _t0 = _t.perf_counter()
            feld = (np.where(weg_maske, basis_kostenfeld * WEGERABATT, basis_kostenfeld)
                   if np.any(weg_maske) else basis_kostenfeld)
            pathfinder = PathfindingSystem(feld, slopemap.shape[0],
                                           edge_distance_map=edge_distance_map)
            pfad, erreicht = pathfinder.find_least_resistance_path(
                (a.x, a.y), (b.x, b.y), self._update_progress)
            # Nicht erreicht -> unendlich, EGAL was die (bei einem Fallback
            # bedeutungslose) Pfadsumme sagen wuerde. Siehe Docstring von
            # find_least_resistance_path().
            kosten = (sum(pathfinder.calculate_movement_cost(x, y) for x, y in pfad[1:])
                     if erreicht else float('inf'))
            self._a_stern_zaehler[0] += 1
            self._a_stern_zaehler[1] += _t.perf_counter() - _t0
            return pathfinder, pfad, kosten

        def merke(pf, pfad):
            """Einen bereits gerouteten Pfad tatsaechlich bauen: glaetten,
            in die Wegemaske eintragen (kuenftige Routen guenstiger machen),
            der Ausgabeliste hinzufuegen. Nimmt Pfad/Pathfinder ENTGEGEN statt
            selbst neu zu routen - der Aufrufer hat sie fuer die
            Bereitschaftspruefung ohnehin schon berechnet."""
            geglaettet = pf.apply_spline_smoothing(
                pfad, smoothing_factor=3, progress_callback=self._update_progress)
            for x, y in pfad:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < weg_maske.shape[0] and 0 <= xi < weg_maske.shape[1]:
                    weg_maske[yi, xi] = True
            roads.append(geglaettet)
            return geglaettet

        roads = []
        gebaut = set()  # {frozenset({id_a, id_b})}

        # Tragen die Orte ueberhaupt Stadttypen? Nur dann greifen die
        # Handelsgewichte (siehe bereitschaft_von()). Im alten
        # Nicht-Weltkarten-Pfad bleibt es bei der Rang-Formel.
        typen_vorhanden = any(
            getattr(s, "settlement_type", "sonstige") != "sonstige" for s in settlements)

        # ---------------------------------------------------- 2: Gabriel-Graph
        with _s(_ts, "gabriel_kandidaten", "Kandidatenpaare"):
            kandidaten_indizes = _gabriel_kandidaten(
                np.array([[s.x, s.y] for s in settlements], dtype=np.float64))
        _rb = _s(_ts, "routen_bewerten", "Kandidaten routen und bewerten")
        _rb.__enter__()

        # BEREITSCHAFT HAENGT NICHT AN DEN WEGKOSTEN - sie kann also VORAB
        # sortiert werden, waehrend die Wegkosten erst BEIM Abarbeiten
        # berechnet werden (nicht vorab in einem zweiten Durchlauf). Das ist
        # kein Stildetail: nur so wirkt der Wegerabatt aus Schritt 1 auch auf
        # die ENTSCHEIDUNG spaeterer, schwaecherer Kandidaten - eine Verbindung
        # neben einer bereits gebauten Haupttrasse wird dadurch tatsaechlich
        # eher gebaut, nicht nur geometrisch an sie herangezogen. Ein erster
        # Entwurf werte alle Kandidaten VOR jedem Bau aus - der Rabatt griff
        # dort nie in die Entscheidung, nur noch in die spaeter neu
        # geroutete Geometrie.
        def bereitschaft_von(i, j):
            """
            Wie sehr zwei Orte diese Verbindung wollen.

            Seit 2026-08-13 (docs/OFFENE_PUNKTE.md 5.16) das HANDELSGEWICHT
            aus den Stadttypen (`handelsgewicht()`, Summe beider einseitiger
            Interessen) statt des frueheren `rang_a * rang_b * kulturfaktor`.
            Der Mechanismus dahinter ist unveraendert: der Wert wird gegen die
            laengenbezogenen Wegkosten gehalten, und der Wegerabatt auf bereits
            gebauten Trassen wirkt weiterhin auf die ENTSCHEIDUNG mit.

            Die Kultur steckt jetzt IM Handelsgewicht (eigene Fraktion zaehlt
            je nach Typ 1.5-fach oder mit eigenen Zahlen), ein zusaetzlicher
            `BEREITSCHAFT_FREMDKULTUR`-Faktor waere doppelt gezaehlt.

            Rueckfall auf die alte Formel, wenn KEIN Ort einen Typ traegt
            (alter Nicht-Weltkarten-Pfad) - dort gibt es weder Kulturen noch
            Typen, und alle Handelsgewichte waeren gleich.
            """
            a, b = settlements[i], settlements[j]
            if typen_vorhanden:
                return handelsgewicht(a, b)
            rang_a = RANG_ZAHL.get(a.rank, 1)
            rang_b = RANG_ZAHL.get(b.rank, 1)
            kulturfaktor = 1.0 if a.culture == b.culture else BEREITSCHAFT_FREMDKULTUR
            return rang_a * rang_b * kulturfaktor

        kandidaten_indizes.sort(key=lambda ij: -bereitschaft_von(*ij))

        # ---------------------------------------------------- 3: Bereitschaftstest
        road_count, total_roads = 0, max(1, len(kandidaten_indizes))
        for i, j in kandidaten_indizes:
            a, b = settlements[i], settlements[j]
            luftlinie = max(1.0, float(np.hypot(a.x - b.x, a.y - b.y)))
            pf, pfad, pfadkosten = route(a, b)
            wegkosten = pfadkosten / luftlinie
            if bereitschaft_von(i, j) > wegkosten:
                merke(pf, pfad)
                gebaut.add(frozenset((a.location_id, b.location_id)))
            road_count += 1
            if self._update_progress:
                progress = 25 + (road_count * 12) // total_roads
                self._update_progress("Road Building", progress,
                                      f"Bewertet {road_count}/{total_roads} Kandidaten")

        # ---------------------------------------------------- 4: Kulturzusammenhang
        _rb.__exit__(None, None, None)
        sea_roads = []
        with _s(_ts, "seekostenfeld", "Seekostenfeld"):
            seekostenfeld = bau_seekostenfeld(heightmap, seegrad=seegrad)
            seepfadfinder = PathfindingSystem(seekostenfeld, slopemap.shape[0])

        def see_route(a, b):
            """
            Seeweg zwischen zwei SIEDLUNGEN, nicht zwischen zwei Wasserpixeln.
            A* laeuft nur zwischen den naechstgelegenen Kuestenpunkten (das
            eigentliche Seekostenfeld haelt Land strikt gesperrt); die kurzen
            Verbindungen Siedlung->Kueste an beiden Enden sind gerade Strecken,
            siehe _naechster_kuestenpunkt().
            """
            ka = _naechster_kuestenpunkt(a.x, a.y, heightmap)
            kb = _naechster_kuestenpunkt(b.x, b.y, heightmap)
            if ka is None or kb is None:
                return [(a.x, a.y), (b.x, b.y)], float('inf')
            see_pfad, erreicht = seepfadfinder.find_least_resistance_path(
                ka, kb, self._update_progress)
            kosten = (sum(seepfadfinder.calculate_movement_cost(x, y) for x, y in see_pfad[1:])
                     if erreicht else float('inf'))
            pfad = [(a.x, a.y)] + see_pfad + [(b.x, b.y)]
            return pfad, kosten

        kulturen = {}
        for idx, s in enumerate(settlements):
            kulturen.setdefault(s.culture, []).append(idx)

        _kz = _s(_ts, "kulturzusammenhang", "Kulturzusammenhang")
        _kz.__enter__()
        for kultur, indizes in kulturen.items():
            if len(indizes) < 2:
                continue
            # Zusammenhangskomponenten NUR ueber bereits gebaute Kanten
            # innerhalb dieser Kultur (Union-Find).
            eltern = {idx: idx for idx in indizes}

            def find(x):
                while eltern[x] != x:
                    eltern[x] = eltern[eltern[x]]
                    x = eltern[x]
                return x

            for idx_a in indizes:
                for idx_b in indizes:
                    if idx_a >= idx_b:
                        continue
                    paar = frozenset((settlements[idx_a].location_id, settlements[idx_b].location_id))
                    if paar in gebaut:
                        wa, wb = find(idx_a), find(idx_b)
                        if wa != wb:
                            eltern[wa] = wb

            # Solange mehr als eine Komponente uebrig ist: die tatsaechlich
            # GUENSTIGSTE Verbindung (Pfadkosten, nicht nur Luftlinie)
            # zwischen zwei VERSCHIEDENEN Komponenten nachtragen - "egal was
            # sie kosten", also ohne Bereitschaftstest. Bei diesen kleinen
            # Gruppen (2-5 Orte je Kultur) ist ein volles Routing je
            # verbleibendem Paar noch billig. Findet sich kein endlicher
            # Landweg (Wasser trennt vollstaendig), wird stattdessen ein
            # Seeweg versucht (§4.4) - eine Union bleibt hier auch dann
            # bestehen, WENN nur der Seeweg gelingt, sonst haette die naechste
            # Runde denselben unmoeglichen Landweg wieder als "billigsten"
            # gewaehlt und liefe endlos im Kreis.
            while len({find(idx) for idx in indizes}) > 1:
                beste = None
                for idx_a in indizes:
                    for idx_b in indizes:
                        if idx_a >= idx_b or find(idx_a) == find(idx_b):
                            continue
                        a, b = settlements[idx_a], settlements[idx_b]
                        pf, pfad, pfadkosten = route(a, b)
                        if beste is None or pfadkosten < beste[0]:
                            beste = (pfadkosten, idx_a, idx_b, pf, pfad)
                if beste is None:
                    break
                kosten, idx_a, idx_b, pf, pfad = beste
                a, b = settlements[idx_a], settlements[idx_b]
                paar = frozenset((a.location_id, b.location_id))

                if np.isfinite(kosten):
                    if paar not in gebaut:
                        merke(pf, pfad)
                        gebaut.add(paar)
                    eltern[find(idx_a)] = find(idx_b)
                    continue

                # Kein endlicher Landweg - Seeweg versuchen. §4.4 Auflage: der
                # groesste Teil der Laenge muss in echtem tiefen Wasser liegen,
                # sonst waere es "kein Seeweg, sondern ein schlechter Landweg".
                # ">=" statt ">": bei kurzen Ueberfahrten (wenige Pfadpunkte)
                # landet der Anteil leicht GENAU auf der Schwelle (z.B. 3 von
                # 5 Punkten = 0.6) - eine echte Mehrheit soll nicht an einem
                # Rundungs-Gleichstand scheitern.
                see_pfad, see_kosten = see_route(a, b)
                if np.isfinite(see_kosten) and _seeweg_anteil_tief(
                        see_pfad, heightmap, seegrad=seegrad) >= 0.5:
                    if paar not in gebaut:
                        sea_roads.append(see_pfad)
                        gebaut.add(paar)
                    eltern[find(idx_a)] = find(idx_b)
                else:
                    # Auch per Seeweg nicht zu verbinden (z.B. Kueste liegt
                    # dazwischen, kein tiefes Wasser erreichbar) - dieses Paar
                    # gilt als erledigt, damit die Schleife nicht haengen
                    # bleibt; die Kultur bleibt fuer diese zwei Orte getrennt.
                    eltern[find(idx_a)] = find(idx_b)

        # ------------------------------------ 4: Ausbau nach GESAMTBEDARF
        #
        # Bis hierher ist jede Entscheidung PAARWEISE gefallen: lohnt sich
        # fuer A und B diese Strecke? Ein Pass ueber einen Bergruecken lohnt
        # sich fuer kein einzelnes Paar - fuer zehn Paare zusammen waere er
        # der groesste Gewinn. Genau das sah das Verfahren strukturell nicht
        # (Nutzerbeobachtung 2026-08-13 am Kartenbild, docs/OFFENE_PUNKTE.md
        # 5.21: "aber nicht das 10 doerfer daran interessiert sind eine
        # verbindung oben zu haben").
        #
        # ERGAENZT das bisherige Netz, ersetzt es nicht (Nutzer-Vorgabe:
        # "bisher sieht das ziemlich gut aus, halte dich etwa daran, aber wir
        # wollen das nur etwas besser machen"). Laeuft auf dem Ortsgraphen mit
        # wenigen Dutzend Knoten und kostet daher fast nichts; teuer ist nur
        # das anschliessende Routen der wenigen tatsaechlich gewaehlten Kanten.
        _kz.__exit__(None, None, None)
        _ba = _s(_ts, "bedarfsausbau", "Netzausbau nach Bedarf")
        _ba.__enter__()
        try:
            roads = self._netz_nach_bedarf_ausbauen(
                settlements, roads, gebaut, kulturen, route, merke)
        except Exception as fehler:                       # pragma: no cover
            self.logger.warning(
                "Bedarfsgetriebener Netzausbau uebersprungen (%s) - das Netz "
                "bleibt beim paarweisen Ergebnis", fehler)
        _ba.__exit__(None, None, None)

        anzahl, dauer = self._a_stern_zaehler
        _ts.bericht()
        logging.getLogger("Pipeline").info(
            "      %-38s %8.3fs  %d Laeufe, %.1f ms je Lauf",
            "[davon A*-Routen]", dauer, anzahl,
            1000.0 * dauer / max(anzahl, 1))

        return roads, sea_roads

    def _seehandel_messen(self, settlements, kulturen, route, see_route,
                          heightmap, seegrad):
        """
        Welcher Anteil des Handels laeuft ueber See - die Groesse, gegen die
        `HAFEN_UMSTEIGEKOSTEN` geeicht ist (docs/OFFENE_PUNKTE.md 5.22).

        Reines Messen, kein Bauen: die Funktion veraendert das Netz nicht. Sie
        existiert, damit die Eichung ueberpruefbar ist statt behauptet - ohne
        sie waere "35 % Seehandel" eine Zahl, die niemand nachrechnen kann.

        Rueckgabe: (anteil, anzahl_seekanten, anzahl_landkanten).
        """
        gesamt_traffic = {}
        see_gesamt = land_gesamt = 0
        for kultur, indizes in kulturen.items():
            if len(indizes) < 2:
                continue
            n = len(indizes)
            C = np.full((n, n), np.inf)
            W = np.zeros((n, n))
            seekanten = set()
            for a in range(n):
                for b in range(a + 1, n):
                    ort_a, ort_b = settlements[indizes[a]], settlements[indizes[b]]
                    W[a, b] = W[b, a] = handelsgewicht(ort_a, ort_b)
                    _pf, _pfad, land_kosten = route(ort_a, ort_b)
                    see_pfad, see_kosten = see_route(ort_a, ort_b)
                    see_gesamt_kosten = (see_kosten + 2.0 * hafenkosten(self.meters_per_pixel)
                                         if np.isfinite(see_kosten) else np.inf)
                    if see_gesamt_kosten < land_kosten:
                        C[a, b] = C[b, a] = see_gesamt_kosten
                        seekanten.add((a, b))
                        see_gesamt += 1
                    else:
                        C[a, b] = C[b, a] = land_kosten
                        if np.isfinite(land_kosten):
                            land_gesamt += 1
            np.fill_diagonal(C, 0.0)
            kanten = {(a, b): float(C[a, b]) for a in range(n) for b in range(a + 1, n)
                      if np.isfinite(C[a, b])}
            if not kanten:
                continue
            anteil = seehandel_anteil(kanten, W, n, seekanten)
            traffic = kanten_traffic(kanten, W, n)
            gesamt_traffic[kultur] = (anteil, sum(traffic.values()))

        if not gesamt_traffic:
            return 0.0, see_gesamt, land_gesamt
        # Ueber die Kulturen nach ihrem Handelsvolumen gewichtet mitteln -
        # eine Kultur mit zwei Orten soll den Gesamtanteil nicht so stark
        # bestimmen wie eine mit acht.
        summe = sum(v for _a, v in gesamt_traffic.values())
        if summe <= 0:
            return 0.0, see_gesamt, land_gesamt
        anteil = sum(a * v for a, v in gesamt_traffic.values()) / summe
        return anteil, see_gesamt, land_gesamt

    def _netz_nach_bedarf_ausbauen(self, settlements, roads, gebaut, kulturen,
                                   route, merke):
        """
        Schritt 4 des Wegenetzes: zusaetzliche Strecken nach ihrem Nutzen fuer
        ALLE Handelspaare (siehe `kanten_nach_bedarf()`, docs/OFFENE_PUNKTE.md
        5.21).

        Je Kultur getrennt, wie die uebrigen Schritte auch. Die Kostenmatrix
        entsteht aus den tatsaechlichen A*-Pfadkosten zwischen den Orten
        dieser Kultur - dieselbe Quelle wie beim paarweisen Bau, damit beide
        Schritte dieselbe Wirklichkeit sehen.
        """
        for kultur, indizes in kulturen.items():
            if len(indizes) < 3:
                # Unter drei Orten gibt es keinen "Umweg ueber Dritte", den
                # eine zusaetzliche Kante abkuerzen koennte.
                continue

            n = len(indizes)
            C = np.full((n, n), np.inf, dtype=np.float64)
            W = np.zeros((n, n), dtype=np.float64)
            pfade = {}
            for a in range(n):
                for b in range(a + 1, n):
                    ort_a = settlements[indizes[a]]
                    ort_b = settlements[indizes[b]]
                    _pf, pfad, kosten = route(ort_a, ort_b)
                    C[a, b] = C[b, a] = kosten
                    W[a, b] = W[b, a] = handelsgewicht(ort_a, ort_b)
                    pfade[(a, b)] = (_pf, pfad)
            np.fill_diagonal(C, 0.0)

            bestehend = {}
            for a in range(n):
                for b in range(a + 1, n):
                    paar = frozenset((settlements[indizes[a]].location_id,
                                      settlements[indizes[b]].location_id))
                    if paar in gebaut and np.isfinite(C[a, b]):
                        bestehend[(a, b)] = float(C[a, b])

            kandidaten = [(a, b) for a in range(n) for b in range(a + 1, n)
                          if (a, b) not in bestehend and np.isfinite(C[a, b])]
            if not kandidaten:
                continue

            gewaehlt = kanten_nach_bedarf(
                C, W, bestehend, kandidaten,
                mindest_nutzen=NETZAUSBAU_MINDESTNUTZEN,
                hoechstens=NETZAUSBAU_MAX_KANTEN)

            for a, b in gewaehlt:
                paar = frozenset((settlements[indizes[a]].location_id,
                                  settlements[indizes[b]].location_id))
                if paar in gebaut:
                    continue
                _pf, pfad = pfade[(a, b)]
                merke(_pf, pfad)
                gebaut.add(paar)
                self.logger.debug(
                    "Bedarfsausbau %s: Strecke %s-%s ergaenzt",
                    kultur, settlements[indizes[a]].location_id,
                    settlements[indizes[b]].location_id)
        return roads

    def _knoten_zufall(self, name: str):
        """
        Ein eigener Zufallsgenerator je Knoten, aus map_seed abgeleitet.

        WARUM NICHT DAS GLOBALE `random`. Der Seed wird einmal im Konstruktor
        gesetzt, aber `random` ist Modulzustand: bis ein spaeter Knoten an die
        Reihe kommt, haengt er davon ab, WIEVIELE Zufallszahlen die Knoten
        davor gezogen haben. Das unterscheidet sich zwischen GPU- und CPU-Pfad,
        weil die verschiedene Codewege nehmen.

        Gemessen am 2026-08-05: roadsite_list war im GPU-Durchlauf leer und im
        CPU-Durchlauf gefuellt - bei identischem Seed und identischer Karte.
        Damit war die Vorgabe verletzt, dass alles aus Seed und Reglern
        reproduzierbar sein muss.

        Ein Generator je Knoten haengt nur am Seed und am Knotennamen, nicht an
        der Ausfuehrungsgeschichte.
        """
        streuung = 0
        for zeichen in name:
            streuung = (streuung * 131 + ord(zeichen)) & 0x7FFFFFFF
        return random.Random((int(self.map_seed) << 8) ^ streuung)

    def calculate_roadsites(self, roads, sea_roads, settlements, heightmap, lod, region_map=None):
        """
        Roadsites nach docs/SIEDLUNGEN_ENTWURF.md §4.6: "bevorzugt an
        Kreuzungen (ein Gasthof lebt vom Verkehr), an Furten und
        Passhoehen, auf langen Zwischenstuecken ohne Ort". Typ kommt aus dem
        45-Arten-Katalog der naechstgelegenen Kultur (ROADSITE_KATALOG,
        docs/KULTUREN_UND_ORTE.md), nach Platzierungskategorie passend
        gewaehlt - eine Kreuzung bei den Kelten wird eher "Zollringwall" als
        "Bardenlager", eine Passhoehe bei den Alemannen eher "Passhospiz".

        Mit `region_map` (WELTKARTE_AKTIV) wird PRO REGION ausgewaehlt statt
        ein einziges Mal fuer die ganze Karte (Nutzer-Vorgabe 2026-08-10:
        "pro region ein paar, so 1-4 jeweils") - vorher gab ein einziges
        globales Ziel (z.B. 3 bei Standard-Reglerstellung) ueber alle neun
        Regionen zusammen nur eine Handvoll Roadsites auf der gesamten
        Weltkarte. Die Kategorie-Prioritaet aus §4.6 bleibt je Region
        erhalten; nur die ZIELZAHL wird jetzt neun Mal statt einmal
        ausgewertet.

        Parameter: roads, sea_roads, settlements, heightmap, lod, region_map
        Returns: List[Location] - Alle platzierten Roadsites
        """
        roadsites = []
        zufall = self._knoten_zufall("roadsites")

        alle_wege = list(roads) + list(sea_roads)
        if not alle_wege or self.roadsites == 0:
            return roadsites

        height, width = heightmap.shape
        size = height

        # ---- Kandidatentypen sammeln, in der Prioritaet aus §4.6 ----
        kreuzungen = list(kreuzungen_finden(roads, sea_roads, settlements, (height, width)))
        # Grad JETZT bestimmen, solange die Reihenfolge noch der von
        # kreuzungen_finden() entspricht - _fern_zuerst() sortiert gleich um.
        kreuzung_grade = kreuzungsgrade(roads, sea_roads, kreuzungen, (height, width))

        furt_punkte, pass_punkte, strecke_punkte = [], [], []
        for weg in roads:  # Furt/Pass nur auf Landwegen sinnvoll
            if len(weg) < 3:
                continue
            hoehen = []
            for x, y in weg:
                xi = int(np.clip(round(x), 0, width - 1))
                yi = int(np.clip(round(y), 0, height - 1))
                h = float(heightmap[yi, xi])
                hoehen.append(h)
                if WASSER_SPERRE_M < h <= 0.0:
                    furt_punkte.append((x, y))
            # Passpunkt: der hoechste Punkt des Weges - aber nur, wenn der
            # Weg wirklich steigt (Spanne > 30 m), sonst waere "hoechster
            # Punkt" nur Rauschen auf flachem Land.
            if hoehen and (max(hoehen) - min(hoehen)) > 30.0:
                pass_punkte.append(weg[int(np.argmax(hoehen))])
            pos = min(len(weg) - 1, int(zufall.uniform(0.3, 0.7) * len(weg)))
            strecke_punkte.append(weg[pos])
        for weg in sea_roads:
            if len(weg) >= 3:
                pos = min(len(weg) - 1, int(zufall.uniform(0.3, 0.7) * len(weg)))
                strecke_punkte.append(weg[pos])

        for punkte in (furt_punkte, pass_punkte, strecke_punkte):
            zufall.shuffle(punkte)

        # 5.14: innerhalb jeder Kategorie randferne Kandidaten zuerst - eine
        # weiche Praeferenz (Tie-Breaker), keine Ausduennung. Die Kategorie-
        # Prioritaet selbst (Kreuzung > Furt > Pass > Strecke) bleibt die
        # PRIMAERE Ordnung unten in `kandidaten`.
        if region_map is not None:
            # Kreuzungen mitsamt ihrem Grad umsortieren, sonst zeigt der Grad
            # anschliessend auf die falsche Kreuzung.
            paare = self._fern_zuerst(
                [(x, y, g) for (x, y), g in zip(kreuzungen, kreuzung_grade)],
                size, xy=lambda p: (p[0], p[1]))
            kreuzungen = [(x, y) for x, y, _g in paare]
            kreuzung_grade = [g for _x, _y, g in paare]
            furt_punkte = self._fern_zuerst(furt_punkte, size)
            pass_punkte = self._fern_zuerst(pass_punkte, size)
            strecke_punkte = self._fern_zuerst(strecke_punkte, size)

        kandidaten = (
            # ECHTE WEGSCHEIDEN ZUERST (2026-08-13, docs/OFFENE_PUNKTE.md
            # 5.20). Nutzer-Vorgabe: "roadsites haben auch bestimmte kriterien
            # (zB taverne ... an einer kreuzung mit min. drei wegen etc)".
            # `kreuzungen_finden()` meldet jede Stelle, an der sich MINDESTENS
            # ZWEI Wege beruehren - darunter viele, an denen zwei Strecken
            # sich nur streifen. Eine Taverne gehoert aber an eine richtige
            # Wegscheide. `kreuzungsgrade()` zaehlt deshalb nach, wie viele
            # verschiedene Wege je Kreuzung zusammenkommen, und Kreuzungen ab
            # KREUZUNG_MIN_WEGE stehen VOR den uebrigen.
            [("kreuzung", x, y) for (x, y), grad in zip(kreuzungen, kreuzung_grade)
             if grad >= KREUZUNG_MIN_WEGE]
            + [("kreuzung", x, y) for (x, y), grad in zip(kreuzungen, kreuzung_grade)
               if grad < KREUZUNG_MIN_WEGE]
            + [("furt", x, y) for x, y in furt_punkte]
            + [("pass", x, y) for x, y in pass_punkte]
            + [("strecke", x, y) for x, y in strecke_punkte]
        )

        # ---- Zielzahl(en): ohne region_map ein einziges Ziel fuer die ganze
        # Karte (altes Verhalten), mit region_map eines je der neun Regionen ----
        lod_factors = {"LOD64": 0.3, "LOD128": 0.6, "LOD256": 1.0, "FINAL": 1.0}
        skala = lod_factors.get(lod, 1.0)
        if region_map is None:
            gebiete = [(np.ones((height, width), dtype=bool),
                       max(0, int(self.roadsites * skala)))]
        else:
            import core.terrain_weltkarte as rw
            region_map = np.asarray(region_map)
            multiplikator = self.roadsites / 3.0
            gebiete = []
            for i, (_z, _s, _r) in enumerate(rw.alle_regionen()):
                maske = (region_map == i)
                if not np.any(maske):
                    continue
                anzahl = max(0, int(round(zufall.randint(1, 4) * multiplikator * skala)))
                gebiete.append((maske, anzahl))

        # ---- Auswahl mit Mindestabstand, damit sie sich nicht drängen ----
        min_abstand = max(3.0, min(height, width) / 40.0)
        gewaehlt = []
        for maske, anzahl_ziel in gebiete:
            if anzahl_ziel <= 0:
                continue
            hinzugefuegt = 0
            for kategorie, x, y in kandidaten:
                if hinzugefuegt >= anzahl_ziel:
                    break
                xi = int(np.clip(round(x), 0, width - 1))
                yi = int(np.clip(round(y), 0, height - 1))
                if not maske[yi, xi]:
                    continue
                if any((x - gx) ** 2 + (y - gy) ** 2 < min_abstand ** 2
                      for _k, gx, gy in gewaehlt):
                    continue
                gewaehlt.append((kategorie, x, y))
                hinzugefuegt += 1

        # ---- Typ aus dem Katalog der naechstgelegenen Kultur ----
        for kategorie, x, y in gewaehlt:
            kultur = _naechste_kultur(x, y, settlements)
            katalog = ROADSITE_KATALOG.get(kultur, []) if kultur else []
            passend = [name for name, kat in katalog if kat == kategorie]
            auswahl = passend or [name for name, _kat in katalog]
            name = zufall.choice(auswahl) if auswahl else "Rastplatz"

            roadsite = Location(
                location_id=self.next_location_id, x=float(x), y=float(y),
                location_type='roadsite', radius=1.5 * self.scale_factor,
                civ_influence=0.4, properties={'roadsite_type': name, 'kategorie': kategorie},
                culture=kultur or "")
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

    def calculate_landmarks(self, civ_map, heightmap, slopemap, water_map, settlements, lod, region_map=None):
        """
        Landmarks nach docs/SIEDLUNGEN_ENTWURF.md §4.7: "unabhaengig vom Netz,
        nach eigenen Kriterien: Gipfel, Kliffs, Quellen, abgelegene Stellen -
        duerfen ausdruecklich weitab jedes Weges liegen". Typ kommt aus dem
        45-Arten-Katalog der naechstgelegenen Kultur (LANDMARK_KATALOG,
        docs/KULTUREN_UND_ORTE.md), nach Kategorie passend gewaehlt.

        VIER KATEGORIEN, VIER MASKEN. Die alte Fassung hatte eine einzige
        pauschale Hoehen-Obergrenze (unterste 70%) - das schloss Gipfel-Arten
        ("Trutzburg auf dem Felskopf", "Atalaya — Signalturm") von genau den
        Stellen aus, die ihr Name verlangt. Jetzt eine Basis (Wildnis + kein
        Extremhang), darauf vier verschiedene Feinauswahlen.

        Mit `region_map` (WELTKARTE_AKTIV) PRO REGION ausgewaehlt, siehe
        `calculate_roadsites()` fuer dieselbe Begruendung (Nutzer-Vorgabe
        2026-08-10: "pro region ein paar, so 1-4 jeweils"). Kategorien bleiben
        global gemischt (kein Prioritaetsranking zwischen ihnen, anders als
        bei Roadsites) - die Randstrafe (5.14) wirkt hier deshalb erst NACH
        dem Mischen aller Kategorien, als letzter, schwaechster Tie-Breaker.

        Parameter: civ_map, heightmap, slopemap, water_map, settlements, lod, region_map
        Returns: List[Location] - Alle platzierten Landmarks
        """
        landmarks = []
        zufall = self._knoten_zufall("landmarks")

        lod_factors = {"LOD64": 0.5, "LOD128": 0.8, "LOD256": 1.0, "FINAL": 1.0}
        skala = lod_factors.get(lod, 1.0)

        height, width = civ_map.shape
        size = height

        # ---- Zielzahl(en) zuerst, damit der Kandidatenpool passend gross ist ----
        if region_map is None:
            gebiete = [(np.ones((height, width), dtype=bool),
                       max(0, int(self.landmarks * skala)))]
        else:
            import core.terrain_weltkarte as rw
            region_map = np.asarray(region_map)
            multiplikator = self.landmarks / 3.0
            gebiete = []
            for i, (_z, _s, _r) in enumerate(rw.alle_regionen()):
                maske = (region_map == i)
                if not np.any(maske):
                    continue
                anzahl = max(0, int(round(zufall.randint(1, 4) * multiplikator * skala)))
                gebiete.append((maske, anzahl))
        gesamt_ziel = sum(a for _m, a in gebiete)
        if gesamt_ziel == 0:
            return landmarks

        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height
        norm_height = ((heightmap - min_height) / height_range
                       if height_range > 0 else np.zeros((height, width)))

        slope_magnitude = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2)
        basis = (civ_map < self.landmark_wilderness) & (slope_magnitude < 0.5)

        schwelle_px = max(2.0, min(height, width) / 20.0)
        land = heightmap > 0.0
        kueste_dist = distance_transform_edt(land) if np.any(~land) else np.full((height, width), np.inf)
        wasser_maske = water_map > 0
        quelle_dist = (distance_transform_edt(~wasser_maske) if np.any(wasser_maske)
                       else np.full((height, width), np.inf))

        kategorie_masken = {
            # "gipfel"/"abgelegen" fehlte bisher der Landfilter - auf der
            # Weltkarte (negative Hoehen = Meer) normiert `norm_height` ueber
            # die GESAMTE Hoehenspanne inklusive Meerestiefen; ein Meerespixel
            # mit civ_map~0 und Hangneigung~0 (offene See ist flach) erfuellte
            # damit klaglos "abgelegen" (norm_height < 0.7 trifft auf fast
            # jedes Meerespixel zu) - Landmarks landeten im offenen Meer.
            "gipfel": basis & land & (norm_height > 0.6),
            "kueste": basis & land & (kueste_dist < schwelle_px),
            "quelle": basis & land & (quelle_dist < schwelle_px),
            "abgelegen": basis & land & (norm_height < 0.7),
        }

        # AUSWAHL NACH EIGNUNG STATT PER ZUFALLSZIEHUNG (2026-08-13,
        # docs/OFFENE_PUNKTE.md 5.18).
        #
        # Die alte Fassung zog aus jeder binaeren Kategoriemaske einen
        # ZUFAELLIGEN Pool und mischte ihn. Ein "Gipfel" landete damit auf
        # irgendeinem Pixel oberhalb 60 % der Hoehenspanne statt auf dem
        # Gipfel. Jetzt liefert `landmark_eignungen()` je Kategorie eine
        # kontinuierliche Guete, und je Region wird schlicht das Beste
        # genommen - mit Mindestabstand, damit nicht alle auf demselben Grat
        # sitzen.
        #
        # Das Rauschen bleibt als leichter Stoerterm erhalten (wie bei der
        # Rangvergabe der Siedlungen, SIEDLUNGEN_ENTWURF §2): sonst saehe
        # jede Karte mit gleichem Seed nicht nur gleich aus, sondern jede
        # Region auch immer nach demselben Muster.
        eignungen = self.landmark_eignungen(civ_map, heightmap, slopemap, water_map)

        min_abstand = max(3.0, min(height, width) / 20.0)
        gewaehlt = []
        for regionsmaske, anzahl_ziel in gebiete:
            if anzahl_ziel <= 0:
                continue
            # Beste Kategorie je Pixel dieser Region, plus etwas Rauschen
            beste_kat = None
            bester_wert = None
            for kategorie, karte in eignungen.items():
                wert = np.where(regionsmaske & kategorie_masken[kategorie], karte, 0.0)
                if bester_wert is None:
                    bester_wert = wert.copy()
                    beste_kat = np.where(wert > 0, kategorie, "")
                else:
                    besser = wert > bester_wert
                    bester_wert = np.where(besser, wert, bester_wert)
                    beste_kat = np.where(besser, kategorie, beste_kat)
            if bester_wert is None or not np.any(bester_wert > 0):
                continue

            # `zufall` ist ein `random.Random` (siehe _knoten_zufall()), KEIN
            # numpy-RandomState - es kennt kein `size=`. Das Rauschfeld
            # deshalb ueber einen aus demselben Generator geseedeten
            # numpy-Generator, damit es weiterhin nur am Seed und am
            # Knotennamen haengt und nicht an der Ausfuehrungsgeschichte.
            rausch_quelle = np.random.RandomState(zufall.randrange(2 ** 31))
            rausch = rausch_quelle.uniform(0.92, 1.08, size=bester_wert.shape)
            punkte_wert = bester_wert * rausch

            hinzugefuegt = 0
            arbeits_wert = punkte_wert.copy()
            # KATEGORIE-VIELFALT JE REGION. Ohne sie gewinnt auf einer Insel
            # fast immer dieselbe Kategorie: gemessen 14 von 20 Landmarks
            # "kueste", weil eine 21-km-Insel eben viel Kueste hat und die
            # Kuesteneignung dort flaechendeckend hoch ist. Nach jeder Wahl
            # wird die gewaehlte Kategorie in DIESER Region gedaempft - bei
            # 1-4 Landmarks je Region genuegt das fuer eine Mischung, ohne
            # eine Kategorie hart zu verbieten (eine Region ganz ohne Berge
            # soll auch weiterhin kein Gipfel-Landmark erzwingen muessen).
            kategorie_daempfung = {k: 1.0 for k in eignungen}
            for _ in range(anzahl_ziel * 4):
                if hinzugefuegt >= anzahl_ziel:
                    break
                if not np.any(arbeits_wert > 0):
                    break
                yi, xi = np.unravel_index(np.argmax(arbeits_wert), arbeits_wert.shape)
                if arbeits_wert[yi, xi] <= 0:
                    break
                x, y = int(xi), int(yi)
                if any((x - gx) ** 2 + (y - gy) ** 2 < min_abstand ** 2
                       for _k, gx, gy in gewaehlt):
                    arbeits_wert[yi, xi] = 0.0
                    continue
                kategorie = str(beste_kat[yi, xi])
                gewaehlt.append((kategorie, x, y))
                hinzugefuegt += 1
                # Umgebung sperren, damit die naechste Wahl woanders landet
                y0, y1 = max(0, y - int(min_abstand)), min(height, y + int(min_abstand) + 1)
                x0, x1 = max(0, x - int(min_abstand)), min(width, x + int(min_abstand) + 1)
                arbeits_wert[y0:y1, x0:x1] = 0.0
                # ... und diese Kategorie regionsweit abwerten
                if kategorie in kategorie_daempfung:
                    kategorie_daempfung[kategorie] *= KATEGORIE_WIEDERHOLUNG
                    betroffen = (beste_kat == kategorie)
                    arbeits_wert[betroffen] *= KATEGORIE_WIEDERHOLUNG

        for kategorie, x, y in gewaehlt:
            kultur = _naechste_kultur(x, y, settlements)
            katalog = LANDMARK_KATALOG.get(kultur, []) if kultur else []
            passend = [name for name, kat in katalog if kat == kategorie]
            auswahl = passend or [name for name, _kat in katalog]
            name = zufall.choice(auswahl) if auswahl else "Verlassene Staette"

            landmark = Location(
                location_id=self.next_location_id, x=float(x), y=float(y),
                location_type='landmark', radius=2.0 * self.scale_factor,
                civ_influence=0.4, properties={'landmark_type': name, 'kategorie': kategorie},
                culture=kultur or "")
            landmarks.append(landmark)
            self.next_location_id += 1

        if self._update_progress:
            self._update_progress("Landmark Placement", 70, f"Placed {len(landmarks)} landmarks")

        return landmarks

    def landmark_eignungen(self, civ_map, heightmap, slopemap, water_map):
        """
        Eine EIGNUNGSKARTE (0..1) je Landmark-Kategorie statt einer binaeren
        Maske (2026-08-13, docs/OFFENE_PUNKTE.md 5.18).

        WARUM DAS DIE EIGENTLICHE VERBESSERUNG IST: die bisherigen
        `kategorie_masken` waren ja/nein-Felder, aus denen anschliessend
        ZUFAELLIG gezogen wurde. Ein "Gipfel"-Landmark landete damit auf
        irgendeinem Pixel oberhalb 60 % der Hoehenspanne - nicht auf dem
        Gipfel. Eine "Kueste"-Landmark auf irgendeinem kuestennahen Pixel -
        nicht am markanten Kliff. Genau das meinte der Nutzer mit
        "Landmarks sind schlecht". Mit einer kontinuierlichen Eignung laesst
        sich stattdessen die BESTE Stelle waehlen.

        Die vier Kategorien entsprechen denen des Katalogs (LANDMARK_KATALOG):

          gipfel     echtes lokales Hoehenmaximum, nicht nur "hoch gelegen"
          kueste     nah am Wasser UND markant (Steilkueste schlaegt Flachufer)
          quelle     nah am Wasser, aber hoch gelegen - ein Ursprung, nicht
                     die Muendung
          abgelegen  weit weg von jeder Zivilisation

        `civ_map` daempft alle vier: "in der naehe von staedten ist oft
        weniger hoch" (Nutzer). Fuer "abgelegen" ist sie zugleich das
        Hauptkriterium.
        """
        from scipy.ndimage import maximum_filter, gaussian_filter

        hoehe = np.asarray(heightmap, dtype=np.float32)
        land = hoehe > 0.0
        if not np.any(land):
            leer = np.zeros(hoehe.shape, dtype=np.float32)
            return {k: leer.copy() for k in ("gipfel", "kueste", "quelle", "abgelegen")}

        size = min(hoehe.shape)
        # Hoehenrang NUR ueber Land - ueber die ganze Karte gerechnet wuerde
        # die Meerestiefe die Spanne dominieren und fast jedes Landpixel als
        # "hoch" erscheinen lassen (derselbe Fehler wie bei den Stadttypen).
        land_hoehen = hoehe[land]
        unten = float(np.percentile(land_hoehen, 40))
        oben = float(np.percentile(land_hoehen, 98))
        hoehenrang = np.clip((hoehe - unten) / max(oben - unten, 1e-6), 0.0, 1.0)

        # GIPFEL: ein echtes lokales Maximum. `maximum_filter` liefert je
        # Pixel den hoechsten Wert der Umgebung; wo er dem eigenen Wert
        # entspricht, steht ein Gipfel. Weich gemacht ueber die Differenz,
        # damit auch "fast Gipfel" noch Werte bekommen und nicht nur ein
        # einzelnes Pixel je Bergkuppe.
        radius = max(3, int(size / 40))
        umgebungsmax = maximum_filter(hoehe, size=2 * radius + 1)
        vorsprung = np.clip(1.0 - (umgebungsmax - hoehe) / max(1.0, 0.15 * (oben - unten)),
                            0.0, 1.0)
        gipfel = hoehenrang * vorsprung

        # KUESTE: nah am Meer und markant. Die Markanz kommt aus der
        # Hangneigung - ein Kliff ist interessanter als ein Sandstrand.
        hang = np.sqrt(slopemap[..., 0] ** 2 + slopemap[..., 1] ** 2)
        hang_norm = np.clip(hang / max(float(np.percentile(hang[land], 90)), 1e-6), 0.0, 1.0)
        kuestennaehe_px = max(2.0, size / 40.0)
        dist_meer = distance_transform_edt(land) if np.any(~land) else np.full(hoehe.shape, np.inf)
        nah_am_meer = np.clip(1.0 - dist_meer / kuestennaehe_px, 0.0, 1.0)
        kueste = nah_am_meer * (0.35 + 0.65 * hang_norm)

        # QUELLE: nah an Suesswasser, aber hoch gelegen. Ohne den
        # Hoehenanteil waere jede Flussmuendung eine "Quelle".
        wasser = np.asarray(water_map) > 0
        if np.any(wasser):
            dist_wasser = distance_transform_edt(~wasser)
            nah_am_wasser = np.clip(1.0 - dist_wasser / kuestennaehe_px, 0.0, 1.0)
        else:
            nah_am_wasser = np.zeros(hoehe.shape, dtype=np.float32)
        quelle = nah_am_wasser * (0.3 + 0.7 * hoehenrang)

        # ABGELEGEN: fern jeder Zivilisation - und BEWUSST GEDECKELT.
        #
        # Ohne den Deckel gewinnt diese Kategorie fast ueberall: `civ_map` ist
        # auf weiten Teilen der Karte schlicht 0 (dort wohnt niemand), die
        # Einsamkeit also 1.0. Gemessen mit leerem Zivilisationsfeld lag
        # "abgelegen" auf 100 % der Landflaeche bei 1.0, waehrend ein echter
        # Gipfel nur 0.32 % der Flaeche ueber 0.5 bringt - ohne Deckel waere
        # JEDES Landmark "abgelegen" geworden und die drei ortsgebundenen
        # Kategorien haetten nie gezogen.
        #
        # Mit ABGELEGEN_DECKEL ist sie der Auffangtyp, genau wie
        # TYP_GRUNDGUETE bei den Stadttypen: ein markanter Gipfel (bis 1.0)
        # oder ein Kliff (bis ~0.9) sticht sie, eine unauffaellige Wildnis
        # nicht.
        civ = np.clip(np.asarray(civ_map, dtype=np.float32), 0.0, 1.0)
        einsamkeit = 1.0 - gaussian_filter(civ, sigma=max(1.0, size / 128.0))
        abgelegen = np.clip(einsamkeit, 0.0, 1.0) ** 2 * ABGELEGEN_DECKEL

        # Zivilisationsdaempfung fuer die drei ORTSGEBUNDENEN Kategorien -
        # ein Gipfel mitten in der Stadt ist kein Landmark. "abgelegen" hat
        # sie bereits als Hauptkriterium und wird nicht doppelt gedaempft.
        naehe_daempfung = np.clip(1.0 - 0.7 * civ, 0.0, 1.0)

        eignungen = {
            "gipfel": gipfel * naehe_daempfung,
            "kueste": kueste * naehe_daempfung,
            "quelle": quelle * naehe_daempfung,
            "abgelegen": abgelegen,
        }
        return {k: np.where(land, np.clip(v, 0.0, 1.0), 0.0).astype(np.float32)
                for k, v in eignungen.items()}

    def calculate_landmark_roads(self, landmarks, roads, heightmap, slopemap, lod):
        """
        Funktionsweise: Verbindet jedes Landmark deterministisch per A*-Pathfinding
        mit dem nächstgelegenen Wegpunkt des bestehenden Hauptstraßennetzes.
        Aufgabe: Landmark-Anbindung ohne Zufallsmechanismus (Nutzer-Vorgabe -
        das dekorative Zusatz-Wegenetz ist bewusst auf Phase 2 verschoben).
        Parameter: landmarks, roads, heightmap, slopemap, lod - Landmark-Liste,
            bestehende Road-Pfade, Hoehen-/Slope-Daten und LOD-Level
        Returns: List[List[Tuple]] - Ein Pfad pro Landmark zum Straßennetz
        """
        if not landmarks or not roads:
            return []

        road_points = [pt for road in roads for pt in road]
        if not road_points:
            return []

        weg_maske = np.zeros(heightmap.shape, dtype=bool)
        for px, py in road_points:
            xi, yi = int(round(px)), int(round(py))
            if 0 <= yi < weg_maske.shape[0] and 0 <= xi < weg_maske.shape[1]:
                weg_maske[yi, xi] = True
        kostenfeld = bau_kostenfeld(heightmap, slopemap, self.road_slope_to_distance_ratio, weg_maske)
        pathfinder = PathfindingSystem(kostenfeld, slopemap.shape[0])
        landmark_roads = []

        for landmark in landmarks:
            distances = [(landmark.x - px) ** 2 + (landmark.y - py) ** 2 for px, py in road_points]
            nearest_idx = int(np.argmin(distances))
            target = road_points[nearest_idx]

            path, _erreicht = pathfinder.find_least_resistance_path(
                (landmark.x, landmark.y), target, self._update_progress)
            smoothed_path = pathfinder.apply_spline_smoothing(
                path, smoothing_factor=3, progress_callback=self._update_progress)
            landmark_roads.append(smoothed_path)

        return landmark_roads

    # calculate_outer_connections() ENTFERNT (2026-08-10, OFFENE_PUNKTE 5.11).
    # Verband Siedlungen mit 2-3 Punkten am KARTENRAND - eine Annahme, die zu
    # keiner Insel/Region passt: es gibt kein sinnvolles "Draussen", zu dem
    # eine Strasse fuehren sollte. docs/SIEDLUNGEN_ENTWURF.md kennt nur
    # Siedlung-Siedlung-, Siedlung-See- (Seewege) und Roadsite/Landmark-
    # Anbindungen - keine Kartenrand-Anbindung. War ein Leftover aus einer
    # frueheren Konzeptphase.

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
    def _find_best_settlement_positions(self, suitability_map, gesperrt_mask,
                                        erlaubt_mask=None):
        """
        Beste verfuegbare Positionen, absteigend nach Eignung.

        VEKTORISIERT (2026-08-10). Die Vorlage war eine Python-Doppelschleife
        ueber JEDEN Pixel, darin fuer jeden Kandidaten noch eine Schleife ueber
        alle bisherigen Siedlungen - bei wachsender Siedlungszahl O(H*W*N) in
        reinem Python, aufgerufen einmal PRO VERSUCH. `erlaubt_mask` ist neu:
        beschraenkt die Suche auf eine Kulturregion, siehe calculate_settlements().

        `gesperrt_mask` (2026-08-11, Pipeline-Audit): fertige (H,W)-Bool-
        Sperrflaeche statt einer Liste bestehender Siedlungen - der Aufrufer
        baut sie einmal je min_distance auf und erweitert sie danach lokal
        (siehe `_markiere_gesperrt`), statt sie hier bei JEDEM Versuch aus
        allen Siedlungen neu zu berechnen. Gemessen: `settlement.settlements`
        skalierte dadurch bei 256->512px (4x Pixel) um 17.5x statt der
        erwarteten ~4x - jetzt behoben.
        """
        gueltig = np.ones(suitability_map.shape, dtype=bool) if erlaubt_mask is None else erlaubt_mask
        if not np.any(gueltig):
            return []
        schwelle = np.percentile(suitability_map[gueltig], 75)
        kandidat = gueltig & (suitability_map >= schwelle)

        if gesperrt_mask is not None:
            kandidat &= ~gesperrt_mask

        ys, xs = np.nonzero(kandidat)
        if len(xs) == 0:
            return []
        reihenfolge = np.argsort(-suitability_map[ys, xs])
        return [(int(xs[i]), int(ys[i])) for i in reihenfolge]

    def _markiere_gesperrt(self, gesperrt_mask, center_x, center_y, radius):
        """
        Traegt EINE Kreisscheibe in eine bestehende Sperrmaske ein - lokal
        begrenzt auf ihre Bounding-Box, wie `_reduce_suitability_around_point`.
        Ermoeglicht, die Sperrflaeche mehrerer Siedlungen inkrementell
        aufzubauen, ohne je Siedlung ein volles (H,W)-Array anzulegen (siehe
        `_find_best_settlement_positions`-Docstring).
        """
        height, width = gesperrt_mask.shape
        y0 = max(0, int(center_y - radius))
        y1 = min(height, int(center_y + radius + 1))
        x0 = max(0, int(center_x - radius))
        x1 = min(width, int(center_x + radius + 1))
        if y1 <= y0 or x1 <= x0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        innerhalb = (xx - center_x) ** 2 + (yy - center_y) ** 2 < radius ** 2
        gesperrt_mask[y0:y1, x0:x1] |= innerhalb

    def _reduce_suitability_around_point(self, suitability_map, center_x, center_y, radius):
        """
        Eignung um einen Punkt herum absenken, fuer Mindestabstaende.
        VEKTORISIERT (2026-08-10) - siehe _find_best_settlement_positions().
        """
        height, width = suitability_map.shape
        y0 = max(0, int(center_y - radius))
        y1 = min(height, int(center_y + radius + 1))
        x0 = max(0, int(center_x - radius))
        x1 = min(width, int(center_x + radius + 1))
        if y1 <= y0 or x1 <= x0:
            return

        yy, xx = np.mgrid[y0:y1, x0:x1]
        distanz = np.hypot(xx - center_x, yy - center_y)
        innerhalb = distanz <= radius
        faktor = 1.0 - (distanz / radius) * 0.8
        block = suitability_map[y0:y1, x0:x1]
        block[innerhalb] *= faktor[innerhalb]

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

    def create_road_network(self, settlements, heightmap, slopemap, road_slope_to_distance_ratio):
        """
        Funktionsweise: Legacy-Methode für Road-Network-Erstellung. Unbenutzt
        im Rest des Projekts (kein Aufrufer gefunden) - nur der Vollstaendigkeit
        halber an die neue calculate_road_network()-Signatur angepasst
        (heightmap fuer die Wasserkosten-Stufen, §4.1), damit sie nicht als
        stiller Aufruf-Landmine liegen bleibt.
        """
        self.road_slope_to_distance_ratio = road_slope_to_distance_ratio
        return self.calculate_road_network(settlements, heightmap, slopemap, "LOD64")

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
    except Exception as fehler:
        # Stiller Ersatzpfad, siehe apply_spline_smoothing() oben fuer die
        # volle Begruendung.
        logging.getLogger(__name__).warning(
            "Spline-Glättung fehlgeschlagen (%s, %d Kontrollpunkte) - "
            "Weg bleibt ungeglättet.", fehler, len(control_points))
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

