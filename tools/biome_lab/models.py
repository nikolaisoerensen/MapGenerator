"""
Path: tools/biome_lab/models.py

Datenklassen des Physics Lab: Location, PlotNode, PlotCore.

PlotNode / Location wurden als eigenstaendige Kopien aus
core/settlement_generator.py uebernommen (dortige @dataclass-Definitionen,
siehe Datei-Docstring "Plotnode-Eigenschaften: node_id, node_location,
connector_id, connector_distance, connector_elevation, connector_movecost").
Bewusst entkoppelt vom SettlementGenerator-Modul, damit das Physics-Lab keine
Abhaengigkeit zu dessen DataLODManager/BaseGenerator-Infrastruktur hat.

PlotNode wurde um node_type (Debug-Highlight), neighbor_core_ids und
neighbor_node_ids erweitert; zusaetzlich gibt es PlotCore fuer die Plotkerne
selbst (beide Seiten der Klick-Navigation: Kern -> Nodes, Node -> Nachbarn).
"""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Location:
    """Eigenstaendige Kopie aus core/settlement_generator.py (dortige
    @dataclass Location). Wird fuer settlement-bezogene Positionsdaten
    (PlotPhysicsLab.settlements) benoetigt."""
    location_id: int
    x: float
    y: float
    location_type: str  # 'settlement', 'landmark', 'roadsite'
    radius: float
    civ_influence: float
    properties: Dict = None


@dataclass
class PlotNode:
    """Ein Knoten im Wege-Netz: entweder eine Voronoi-Kreuzung zwischen
    Plotkernen (node_type='standard_plot_node') oder einer der drei
    Sondertypen, die in physics.py nur TANGENTIAL entlang ihrer jeweiligen
    Kontur gleiten duerfen:
      - 'wilderness_node': gleitet auf der civ-Kontur (Wildnisgrenze)
      - 'map_border_node': gleitet auf dem Kartenrand-Rechteck
      - 'city_border_node': gleitet auf der Stadtgrenzkontur
    'wilderness_core' markiert dagegen einen normalen Plotkern (siehe
    PlotCore), der tief in der Wildnis liegt und deshalb von der
    Feder-/Feldphysik ausgeschlossen ist (siehe topology._active_physics_cores)."""
    node_id: int
    node_location: tuple[float, float]
    connector_ids: list[int]
    connector_distances: list[float]
    connector_elevations: list[float]
    connector_move_costs: list[float]
    connector_edge_ids: list[int]
    settlement_id: int = -1
    node_type: str = "standard_plot_node"
    neighbor_core_ids: list[int] = field(default_factory=list)
    neighbor_node_ids: list[int] = field(default_factory=list)
    traffic_weight: float = 4.0


@dataclass
class PlotCore:
    """Leichte Klasse fuer die Plotkerne selbst. Ergaenzt um core_type und
    dieselben Nachbarschafts-Listen wie PlotNode, damit ein Klick auf einen
    Kern weiss, welche Nodes/Kerne zu ihm gehoeren."""
    core_id: int
    location: tuple[float, float]
    core_type: str = "standard_core"
    neighbor_core_ids: list[int] = field(default_factory=list)
    neighbor_node_ids: list[int] = field(default_factory=list)
