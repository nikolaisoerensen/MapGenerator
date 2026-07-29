"""
Path: core/water_generator.py

Funktionsweise: Dynamisches Hydrologiesystem mit Erosion, Sedimentation und bidirektionaler Terrain-Modifikation

AUSFÜHRUNGSREIHENFOLGE PRO LOD-RUNDE (Nutzer-Vorgabe 2026-07-27, im
CALCULATOR_GRAPH über echte Kanten erzwungen - siehe dortigen Reihenfolge-Block):

  1. water.erosion_sedimentation  Droplet-Erosion (DropletErosionSystem)
  2. water.thermal_erosion        Böschungswinkel (ThermalErosionSystem)
  --- ab hier ist das Gelände dieser Runde final ---
  3. water.lake_detection         Senken/Seen (LakeDetectionSystem)
  4. water.flow_network           Wasserkreislauf (PipeFlowSimulator)
  5. water.manning_flow           Fluss-Breite (ManningFlowCalculator)
  6. water.soil_moisture          Bodenfeuchte (SoilMoistureCalculator)
  7. water.evaporation            Verdunstung (EvaporationCalculator)

Schritt 1 und 2 formen das GELÄNDE und sind bewusst UNABHÄNGIG vom
Weather-Niederschlag: die Erosions-Partikel spawnen höhengewichtet direkt aus
der Heightmap (Erosion modelliert geologische Zeit). Erst ab Schritt 3 kommt
der ECHTE Niederschlag ins Spiel - Seen, Flüsse, Bodenfeuchte und alles
Nachgelagerte entstehen auf dem bereits erodierten Gelände.

Verfahren im Einzelnen:
- Lake-Detection über Priority-Flood-Wasserscheide, Becken bis zum Spill-Point gefüllt
- Flussnetzwerk über virtuelle-Rohre-Hydraulik (Mei/Decaudin/Lefebvre 2007)
- Droplet-Erosion nach Sebastian Lagues Referenzalgorithmus
- Böschungswinkel-Erosion (Angle of Repose), härteabhängig
- Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
- Evaporation aus Weather-Daten (temp_map, wind_map, humid_map), wirkt als
  Senke im Pipe-Modell zurück

Parameter Input (alle über WaterTab-Slider, siehe gui/config/value_default.py WATER):
- lake_volume_threshold (Mindest-Seevolumen in m³)
- river_abundance (Anteil der wasserführenden Pixel, der als Fluss gilt, 0..1)
- erosion_strength (Erosionsintensitäts-Multiplikator)
- sediment_capacity_factor (Transportkapazität eines Erosions-Partikels)
- settling_velocity (Anteil des Überschuss-Sediments, der pro Schritt absinkt)
- thermal_erosion_strength (Stärke des Böschungswinkel-Abrutschens)
- diffusion_radius (Grundwasser-Ausbreitungsradius in Pixeln)
- evaporation_base_rate (Basis-Verdunstungsrate in m/Tag)

Dependencies (über DataLODManager):
- heightmap (von terrain_generator für Orographic-Effects und Flow-Pathfinding)
- hardness_map (von geology_generator für Erosions-Resistance)
- precip_map (von weather_generator für Precipitation-driven Water-Sources)
- temp_map (von weather_generator für Temperature-based Evaporation)
- wind_map (von weather_generator für Wind-enhanced Evaporation)
- humid_map (von weather_generator für Temperature-based Evaporation)

Output:
- WaterData-Objekt mit den 12 Feldern aus HydrologySystemGenerator.WATER_DATA_KEYS
  plus validity_state und LOD-Metadaten
- Bidirektionale Terrain-Integration: DataLODManager.get_terrain_data_combined() liest erosion_map/
  sedimentation_map und liefert base_heightmap - erosion_map + sedimentation_map an alle
  nachgelagerten Generatoren; jeder neue Water-LOD-Durchlauf bekommt seinerseits diese bereits
  erodierte Heightmap als Input und akkumuliert seine frische Erosion/Sedimentation on top
  (Erosion rechnet nur in der letzten LOD-Runde, siehe HydrologySystemGenerator._is_final_lod())
- DataLODManager-Storage für nachfolgende Generatoren (biome, settlement)
"""

import numpy as np
from scipy.ndimage import (gaussian_filter, distance_transform_edt, map_coordinates,
                            minimum_filter, maximum_filter)
from skimage.morphology import reconstruction as _skimage_reconstruction
from skimage.segmentation import watershed as _skimage_watershed
from typing import Dict, Any, List, Optional
import logging

# Biom-abhängige Boden-Feuchte-Kapazität/Verdunstung (Biome-Preseed-Plan
# Punkt C) - Index 0-14 spiegelt BaseBiomeClassifier.biome_definitions
# (core/biome_generator.py) 1:1 (gleiche Reihenfolge/Werte, absichtlich hier
# dupliziert statt importiert - identisches, bereits etabliertes Muster wie
# _BIOME_ROUGHNESS_DAMPING in weather_generator.py, um core/water_generator.py
# nicht von core/biome_generator.py abhängig zu machen). Index 15-19 deckt
# echte Wasser-Kacheln ab (ocean/lake/grand_river/river/creek - dieselbe
# Super-Biome-Reihenfolge wie _BIOME_ROUGHNESS_DAMPING), die im
# Trocknungs-Term ohnehin übersprungen werden, aber eine volle Kapazität
# brauchen, damit der abschließende Kapazitäts-Clip echtes Wasser nicht
# fälschlich unter 100 deckelt.
_BIOME_MOISTURE_CAPACITY = np.array([
    40.0, 45.0, 60.0, 55.0, 75.0, 40.0, 20.0, 30.0, 95.0, 70.0,
    50.0, 70.0, 100.0, 35.0, 15.0,   # 0-14: ice_cap..badlands
    100.0, 100.0, 100.0, 100.0, 100.0,   # 15-19: ocean/lake/grand_river/river/creek
], dtype=np.float32)
_BIOME_EVAPORATION_FACTOR = np.array([
    0.2, 0.5, 0.6, 1.0, 0.6, 1.3, 1.8, 1.4, 0.5, 0.8,
    1.1, 0.6, 0.3, 1.3, 1.7,   # 0-14: ice_cap..badlands
    0.0, 0.0, 0.0, 0.0, 0.0,   # 15-19: ungenutzt (Wasser-Kacheln ausgenommen)
], dtype=np.float32)

# Geteilte physikalische Konstanten (Pipe-Modell-Umbau 2026-07-25, siehe
# PipeFlowSimulator) - vorher in _calculate_stream_power_erosion() inline
# dupliziert; jetzt eine gemeinsame Quelle fuer sowohl die Pipe-Fluss-
# Simulation als auch die Stream-Power-Erosion, verhindert Drift zwischen
# zwei Kopien derselben Naturkonstante.
GRAVITY = 9.81      # m/s^2
RHO_WATER = 1000.0  # kg/m^3


class WaterData:
    """
    Funktionsweise: Container für alle Water-Daten mit Metainformationen und LOD-System
    Aufgabe: Speichert die 12 Hydrologie-Outputs (siehe
        HydrologySystemGenerator.WATER_DATA_KEYS - dieselbe Liste steuert die
        Ablage im DataLODManager) mit LOD-Level und Validity-State.
    """
    def __init__(self):
        self.water_map = None              # (height, width) - Gewässertiefen in m
        self.flow_map = None               # (height, width) - Durchfluss-Betrag in m³/s
        self.flow_speed = None             # (height, width) - Fließgeschwindigkeit in m/s
        self.cross_section = None          # (height, width) - Flussquerschnitt in m²
        self.soil_moist_map = None         # (height, width) - Bodenfeuchtigkeit in %
        # Erosion/Sedimentation sind KUMULIERTE METER über die LOD-Kette,
        # keine Jahresraten - das Modell kennt keinen Zeitbezug, gegen den
        # eine Rate definiert wäre (Kommentar bis 2026-07-27 fälschlich
        # "m/Jahr", ebenso die Beschriftung im Water-Tab).
        # === ALTBESTAND (stillgelegt 2026-07-28) ===
        # Die vier gelaendeformenden Karten gehoeren jetzt ErosionData
        # (core/erosion_generator.py). Die Attribute bleiben vorerst als
        # Platzhalter stehen, damit aelterer Code, der sie liest, None statt
        # eines AttributeError bekommt; sie werden nie mehr befuellt und
        # koennen mit dem uebrigen Altbestand geloescht werden.
        self.erosion_map = None
        self.sedimentation_map = None
        self.thermal_erosion_map = None
        self.thermal_deposition_map = None
        self.evaporation_map = None        # (height, width) - Verdunstung in gH2O/m²/Tag
        self.ocean_outflow = None          # Scalar - Randabfluss über den Durchlauf in m³
        self.water_biomes_map = None       # (height, width) - Wasser-Klassifikation 0-4

        # LOD-System Integration
        self.lod_level = 1                 # Aktueller LOD-Level (numerisch)
        self.actual_size = 32              # Tatsächliche Kartengröße
        self.validity_state = "valid"      # Validity-State für Cache-Management
        self.parameter_hash = None         # Parameter-Hash für Cache-Invalidation
        self.parameters = {}               # Verwendete Parameter für Cache-Management

# 3x3-Nachbarschaft OHNE Zentrum - Grundlage für die vektorisierte
# Minima-Suche (minimum_filter mit diesem Footprint liefert das Minimum der 8
# Nachbarn, das Zentrum selbst geht bewusst nicht ein).
_NEIGHBOR_FOOTPRINT_8 = np.array([[True, True, True],
                                  [True, False, True],
                                  [True, True, True]], dtype=bool)

# Die 8 Nachbar-Verschiebungen als (dy, dx) - eine Quelle für alle
# Nachbarschafts-Sweeps dieses Moduls (Spill-Höhen, Minima-Prüfung).
_NEIGHBOR_OFFSETS_8 = ((-1, -1), (-1, 0), (-1, 1),
                       (0, -1), (0, 1),
                       (1, -1), (1, 0), (1, 1))


def fill_depressions(heightmap):
    """
    Füllt jede abflusslose Senke bis zu ihrem Überlaufpunkt auf und liefert die
    FÜLLHÖHE pro Zelle zurück (>= 0, in Metern) - nicht das gefüllte Gelände.
    Der Aufrufer entscheidet, was er damit macht; DropletErosionSystem bucht
    sie als Sedimentation, was physikalisch genau stimmt: Material füllt die
    Senke.

    Verfahren: morphologische Rekonstruktion durch Erosion
    (skimage.morphology.reconstruction) - dasselbe Ergebnis wie
    Priority-Flood/Wang-Liu, nur in Cython statt in Python. Der Kartenrand
    bleibt offen (Startwerte dort = Originalhöhe), Wasser kann die Karte also
    verlassen; alles im Inneren wird bis zur niedrigsten Überlaufkante
    angehoben.

    WOZU: Die Droplet-Erosion erzeugt zwangsläufig geschlossene Gruben - ein
    Partikel gräbt sich ein und der nächste fällt hinein. Ohne Gegenmittel
    zerfällt die Karte in einzelne Krater ohne Abfluss, die sich nie zu einem
    Tiefpunkt verbinden (Nutzer-Report (a)). Gemessen bei 128²/96²:

        nach Droplet-Erosion          230 lokale Minima
        + Senkenfüllung                 0 lokale Minima      (2.4 ms bei 96²)

    Der Nebeneffekt ist der eigentliche Gewinn: die gefüllten Bereiche sind
    EBEN - mittlere Hangneigung 4.0 m/px innerhalb der Füllung gegenüber
    32.9 m/px außerhalb, bei 27.8% Flächenanteil. Das ist genau das Zielbild
    "Sedimentation erzeugt Ebenen, also flache Landschaften mit einzelnen
    Hügeln, die durch die Ebenen brechen".

    Ein Kanal, der bis zum Kartenrand entwässert, ist per Definition KEINE
    Senke und bleibt unangetastet - eingeschnittene Flussläufe überleben die
    Füllung, nur geschlossene Gruben verschwinden.

    Parameter: heightmap (H,W)
    Return: (H,W) float64, Füllhöhe je Zelle (0 wo nichts gefüllt wurde)
    """
    dem = heightmap.astype(np.float64, copy=False)
    if dem.shape[0] < 3 or dem.shape[1] < 3:
        return np.zeros(dem.shape, dtype=np.float64)

    # Saat: überall das Maximum (= "erstmal alles voll"), nur der Rand behält
    # seine echte Höhe. Die Rekonstruktion durch Erosion drückt die Saat dann
    # so weit herunter, wie das Gelände es erlaubt - übrig bleibt genau die
    # bis zum Überlauf gefüllte Oberfläche.
    seed = np.full_like(dem, dem.max())
    seed[0, :] = dem[0, :]
    seed[-1, :] = dem[-1, :]
    seed[:, 0] = dem[:, 0]
    seed[:, -1] = dem[:, -1]

    filled = _skimage_reconstruction(seed, dem, method='erosion')
    return np.maximum(0.0, filled - dem)


def _detect_local_minima_full(heightmap):
    """
    Identifiziert alle STRIKTEN lokalen Minima einer Heightmap (jeder der 8
    Nachbarn liegt echt höher), ohne die Randzellen - siehe
    LakeDetectionSystem._detect_local_minima, das dies als dünner Wrapper
    aufruft. Modulweite Funktion, damit sie auch außerhalb von
    LakeDetectionSystem wiederverwendbar ist.

    Vektorisiert über scipy.ndimage.minimum_filter (2026-07-27) - die
    vorherige Python-Doppelschleife mit innerem 3x3-Scan kostete bei 512²
    bereits 0.16 s und wuchs linear mit der Pixelzahl weiter, bei jedem
    LOD erneut. Semantisch identisch: `heightmap < min(8 Nachbarn)` ist exakt
    die Bedingung "kein Nachbar <= current_height" der alten Schleife, und
    der Rand wird wie zuvor ausgeschlossen (die alte Schleife lief über
    range(1, n-1)).
    Return: Liste von (x, y)-Tupeln, aufsteigend nach (y, x) - dieselbe
    Reihenfolge, die die zeilenweise Doppelschleife erzeugte (die Basin-IDs
    hängen von dieser Reihenfolge ab).
    """
    height, width = heightmap.shape
    if height < 3 or width < 3:
        return []

    neighbor_min = minimum_filter(
        heightmap, footprint=_NEIGHBOR_FOOTPRINT_8, mode='constant', cval=np.inf)
    is_minimum = heightmap < neighbor_min

    # Rand ausschließen (dort ist die 8er-Nachbarschaft unvollständig).
    is_minimum[0, :] = False
    is_minimum[-1, :] = False
    is_minimum[:, 0] = False
    is_minimum[:, -1] = False

    ys, xs = np.nonzero(is_minimum)   # np.nonzero liefert bereits (y, x)-sortiert
    return list(zip(xs.tolist(), ys.tolist()))


def _apply_priority_flood_watershed(heightmap, seeds):
    """
    Wasserscheiden-Zuordnung per Priority-Flood/Immersionssimulation (Vincent-Soille-
    Watershed-Transform): jede Zelle wird dem Becken zugeordnet, dessen Flutung sie
    zuerst erreicht (Multi-Source-Dijkstra mit Höhe als Kosten über heapq). Modulweite
    Funktion (siehe LakeDetectionSystem._apply_jump_flooding, das dies als dünner
    Wrapper aufruft) - derselbe Algorithmus-Stammbaum wie RichDEM's Priority-Flood-
    Depression-Filling (Barnes et al. 2014, https://github.com/r-barnes/richdem),
    das ebenfalls Senken "to the level of their lowest outlet or spill-point" füllt.

    Ersetzt die vormalige Jump-Flooding-Variante, die Zellen rein per "current_height
    >= seed_height AND kürzeste Luftlinien-Distanz" zuwies - das ist keine echte
    Erreichbarkeits-Prüfung (kein monotoner Abwärtspfad zum Seed nötig), sondern nur
    eine grobe obere Schranke, die praktisch immer erfüllt ist (jeder Punkt der Karte
    liegt höher als IRGENDEIN lokales Minimum irgendwo auf der Karte). Dadurch wurde
    de facto ein reines Luftlinien-Voronoi-Diagramm über die gesamte Karte gelegt,
    das auch Berggipfel dem nächstgelegenen Tal zuschlug, unabhängig von Bergkämmen
    dazwischen - empirisch bestätigt: 99.98-100% der Karte wurden einem Becken
    zugewiesen, ungeachtet der Topographie.

    Die Priority-Flood-Wasserscheide stoppt dagegen an echten Wasserscheiden (Grate),
    weil jede Zelle vom zuerst dort ankommenden (= niedrigsten) Flutungs-Frontpunkt
    beansprucht wird - das ist die Standard-Definition eines Einzugsgebiets.

    Implementierung seit 2026-07-27: skimage.segmentation.watershed (8er-
    Konnektivität, Marker = die übergebenen Seeds). Das ist derselbe
    Immersions-Watershed, den die vorherige handgeschriebene heapq-Schleife
    nachbildete - nur in Cython statt in Python. Gegen die alte Fassung
    verifiziert: bei 128² und 256² auf gefiltertem Rauschen stimmen die
    Becken-Zuordnungen zu 100.00% überein, bei 14-facher Geschwindigkeit
    (0.33 s -> 0.024 s bei 256²). Der Vollständigkeits-Charakter bleibt
    erhalten: jede Zelle wird genau einem Becken zugewiesen, solange
    mindestens ein Seed existiert.
    """
    height, width = heightmap.shape
    if not seeds:
        return np.full((height, width), -1, dtype=np.int32)

    # skimage-Marker sind 1-basiert (0 = "noch nicht zugeordnet"), die
    # Becken-IDs dieses Moduls 0-basiert mit -1 als "kein Becken".
    markers = np.zeros((height, width), dtype=np.int32)
    seed_x = np.fromiter((s[0] for s in seeds), dtype=np.intp, count=len(seeds))
    seed_y = np.fromiter((s[1] for s in seeds), dtype=np.intp, count=len(seeds))
    markers[seed_y, seed_x] = np.arange(1, len(seeds) + 1, dtype=np.int32)

    basin_map = _skimage_watershed(
        heightmap, markers=markers, connectivity=_NEIGHBOR_FOOTPRINT_8 | np.eye(3, dtype=bool))
    return (basin_map.astype(np.int32) - 1)


def _compute_spill_elevations(heightmap, basin_id_map, num_basins):
    """
    Berechnet pro Becken-ID die Überlauf-/Spill-Point-Höhe (niedrigster Punkt, an dem
    ein Becken in ein Nachbarbecken oder über den Kartenrand überläuft) - extrahiert
    aus LakeDetectionSystem._classify_lake_basins()s erster Schleife, dort weiterhin
    per Aufruf dieser Funktion genutzt (siehe dortigen Docstring für die volle
    Herleitung, warum der Spill-Point statt Becken-Minimum/-Maximum der korrekte
    Wasserspiegel-Bezugspunkt ist). Modulweite Funktion, damit sie auch außerhalb von
    LakeDetectionSystem wiederverwendbar ist (siehe compute_full_watershed()).
    Rückgabe: 1D-Array (Länge num_basins), np.inf für Becken ohne Nachbar-Kreuzung
    (sollte bei einer vollständigen Karte nicht vorkommen, außer num_basins==0).

    Vektorisiert (2026-07-27): 8 verschobene Vollbild-Vergleiche + np.minimum.at
    statt der vorherigen Python-Doppelschleife mit innerem 8er-Scan (bei 512²
    2.0 s, wachsend mit der Pixelzahl, bei jedem LOD erneut). Semantisch
    identisch, inklusive der beiden Sonderfälle: Randzellen tragen ihre eigene
    Höhe bei (offener Abfluss über den Kartenrand) und Nachbarn außerhalb der
    Karte werden übersprungen statt als fremdes Becken gewertet.
    """
    height, width = heightmap.shape
    spill_elevation = np.full(max(num_basins, 0), np.inf, dtype=np.float64)
    if num_basins <= 0:
        return spill_elevation

    heights = heightmap.astype(np.float64, copy=False)
    basins = basin_id_map

    # Sonderfall 1: Becken berührt den Kartenrand -> offener Abfluss, die
    # eigene Höhe der Randzelle ist eine gültige Überlauf-Höhe.
    edge_mask = np.zeros((height, width), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_cells = edge_mask & (basins >= 0)
    if np.any(edge_cells):
        np.minimum.at(spill_elevation, basins[edge_cells], heights[edge_cells])

    # Sonderfall 2: Kreuzung in ein Nachbarbecken -> der höhere der beiden
    # Punkte ist die Sattelhöhe, die überwunden werden muss.
    for dy, dx in _NEIGHBOR_OFFSETS_8:
        # Überlappender Ausschnitt beider Gitter für diese Verschiebung -
        # Zellen ohne realen Nachbarn in dieser Richtung fallen dadurch
        # automatisch weg (entspricht dem Bounds-Check der alten Schleife).
        y_from, y_to = max(0, -dy), height - max(0, dy)
        x_from, x_to = max(0, -dx), width - max(0, dx)
        if y_from >= y_to or x_from >= x_to:
            continue

        own = basins[y_from:y_to, x_from:x_to]
        other = basins[y_from + dy:y_to + dy, x_from + dx:x_to + dx]
        crossing = np.maximum(heights[y_from:y_to, x_from:x_to],
                              heights[y_from + dy:y_to + dy, x_from + dx:x_to + dx])

        relevant = (own >= 0) & (own != other)
        if np.any(relevant):
            np.minimum.at(spill_elevation, own[relevant], crossing[relevant])

    return spill_elevation


class LakeDetectionSystem:
    """
    Funktionsweise: Identifiziert Seen über eine Priority-Flood-Wasserscheide
    (siehe _apply_priority_flood_watershed) und füllt jedes Becken bis zu
    seinem Überlaufpunkt.
    Aufgabe: Findet alle potentiellen Seestandorte und deren Einzugsgebiete.

    meters_per_pixel ist Konstruktor-/Aufruf-Parameter, weil
    lake_volume_threshold seit 2026-07-27 ein echtes Volumen in m³ ist: ohne
    die Zellfläche wäre die Schwelle in "Meter-Pixel" und damit direkt von der
    Auflösung abhängig (dieselbe Geländeform ergäbe bei map_size 512 etwa
    viermal so viele Seen wie bei 128, ohne dass ein Slider das anzeigt).
    """

    def __init__(self, lake_volume_threshold=0.1, shader_manager=None, meters_per_pixel=1.0):
        self.lake_volume_threshold = lake_volume_threshold
        self.shader_manager = shader_manager
        self.meters_per_pixel = meters_per_pixel

    def _cell_area_m2(self) -> float:
        """Reale Grundfläche einer Zelle in m² - Umrechnungsfaktor zwischen der
        aufsummierten Wassertiefe (Meter-Pixel) und einem echten Volumen."""
        return float(self.meters_per_pixel) ** 2

    def detect_lakes(self, heightmap, parameters, meters_per_pixel=None):
        """
        Funktionsweise: GPU-accelerated Lake-Detection mit Fallback-Strategie
        Aufgabe: GPU-Pfad mit CPU-Fallback für robuste Lake-Detection
        meters_per_pixel: überschreibt für diesen Aufruf den Konstruktor-Wert
            (der Aufrufer kennt die LOD-abhängige Auflösung, siehe
            HydrologySystemGenerator._calc_lake_detection()).
        """
        if meters_per_pixel is not None:
            self.meters_per_pixel = meters_per_pixel

        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "jumpFloodLakes",
                    {
                        "heightmap": heightmap,
                        "lake_volume_threshold": self.lake_volume_threshold,
                        "meters_per_pixel": self.meters_per_pixel,
                    },
                    parameters
                )
                if result.get("success"):
                    return result["lake_map"], result["valid_lakes"]
            except Exception as e:
                logging.warning(f"GPU lake detection failed: {e}, falling back to CPU")

        return self._cpu_lake_detection(heightmap)

    def _cpu_lake_detection(self, heightmap):
        """
        CPU-Pfad: lokale Minima -> Priority-Flood-Wasserscheide -> Becken bis
        zum Überlaufpunkt füllen. Alle drei Schritte sind vektorisiert
        (siehe die jeweiligen Modulfunktionen).

        Der frühere `except Exception -> _simple_lake_detection()`-Fallback ist
        entfernt (2026-07-27): _simple_lake_detection() markierte einzelne
        Senken-Pixel als "See" und lieferte eine erfundene valid_lakes-Liste
        mit einem Pseudo-See in der Kartenmitte. Das ist kein Ergebnis, das
        irgendein Konsument sinnvoll verwenden kann - es sah nur so aus, als
        wäre die Berechnung gelungen, und verdeckte damit jeden echten
        Programmfehler in diesem Pfad. Ein Fehler propagiert jetzt bis zum
        Aufrufer.
        """
        height, width = heightmap.shape

        local_minima = self._detect_local_minima(heightmap)
        if not local_minima:
            return np.full((height, width), -1, dtype=np.int32), []

        basin_map = self._apply_jump_flooding(heightmap, local_minima)
        return self._classify_lake_basins(heightmap, basin_map, local_minima)

    def _detect_local_minima(self, heightmap):
        """Identifiziert alle lokalen Minima als potentielle See-Seeds (dünner
        Wrapper um die modulweite _detect_local_minima_full(), siehe dort -
        Extraktion für Wiederverwendung durch compute_full_watershed())."""
        return _detect_local_minima_full(heightmap)

    def _apply_jump_flooding(self, heightmap, lake_seeds):
        """Wasserscheiden-Zuordnung (dünner Wrapper um die modulweite
        _apply_priority_flood_watershed(), siehe dort für die volle
        Algorithmus-Herleitung - Extraktion für Wiederverwendung durch
        compute_full_watershed())."""
        return _apply_priority_flood_watershed(heightmap, lake_seeds)

    def _classify_lake_basins(self, heightmap, lake_map, lake_seeds):
        """
        Klassifiziert See-Becken nach Volumen und validiert Threshold.

        Ein Becken (Einzugsgebiet, aus _apply_jump_flooding) ist meist viel größer als
        der eigentliche See darin - der Großteil ist trockenes Gelände, das nur ins
        Becken entwässert. Der tatsächliche Wasserspiegel steigt bis zum niedrigsten
        Punkt am Beckenrand (Spill-Point/Sattelpunkt, dort läuft er ins Nachbarbecken
        oder über den Kartenrand über), NICHT bis zur Höhe irgendeines beliebigen
        Randpixels. Nur Pixel unterhalb dieses Spill-Points sind tatsächlich unter
        Wasser - vorher wurde stattdessen nur gegen die Höhe des Becken-Minimums selbst
        geprüft (`terrain_height <= seed_height`), was praktisch nie erfüllt war (per
        Definition liegt ein striktes lokales Minimum unter all seinen Nachbarn, sodass
        nur das Minimum-Pixel selbst die Bedingung erfüllte) - dadurch war total_volume
        für jedes Becken quasi immer 0 und es konnte nie ein See entstehen.

        Ein zwischenzeitlicher Fix-Versuch (paralleler Branch) nahm stattdessen die
        MAXIMALE Höhe innerhalb des Beckens als Wasserspiegel - das behebt zwar das
        total_volume=0-Problem, öffnet aber das GEGENTEIL-Problem wieder: da jeder
        Punkt im Becken per Definition <= dem Becken-Maximum liegt, zählt dann
        wieder das GESAMTE Becken (inklusive Berggipfel) als überflutet. Der
        Spill-Point (niedrigster RAND, nicht höchster Punkt) ist der einzige Wert,
        der beide Probleme gleichzeitig vermeidet.

        Spill-Elevation-Berechnung ist nach _compute_spill_elevations() extrahiert
        (siehe dort - modulweit, auch von compute_full_watershed() genutzt).

        Vektorisiert (2026-07-27): die vorherige Fassung lief pro Seed einmal
        über die GANZE Karte (`np.where(lake_map == lake_id)`) und war damit
        O(Seeds x Pixel). Da die Zahl lokaler Minima auf zerklüftetem Terrain
        etwa proportional zur Pixelzahl wächst, war dieser eine Block faktisch
        quadratisch in der Pixelzahl - gemessen 0.60 s bei 256², 13.03 s bei
        512² (Faktor 22 bei 4-facher Pixelzahl), hochgerechnet mehrere Minuten
        bei 1024², und das pro LOD-Stufe. Er war damit der harte Deckel für die
        nutzbare map_size auf dem CPU-Pfad.

        Jetzt: EIN Durchlauf über die Karte, Volumen und Pixelzahl pro Becken
        gleichzeitig per np.bincount. Ergebnis ist bitidentisch zur alten
        Fassung, inklusive der Vergabereihenfolge der neuen Lake-IDs (die alte
        Schleife vergab sie aufsteigend nach Becken-ID, `np.cumsum` über die
        qualifizierten Becken reproduziert genau das).

        Die volume_threshold-Umrechnung: `lake_volume_threshold` ist seit
        2026-07-27 ein echtes Volumen in m³ (siehe WATER.LAKE_VOLUME_THRESHOLD)
        - `total_volume` unten ist die Summe der Wassertiefen über alle
        überfluteten Pixel und wird deshalb mit der Zellfläche multipliziert.
        Vorher war die Schwelle in "Meter-Pixel" und skalierte dadurch direkt
        mit der Auflösung: dieselbe Geländeform ergab bei 512 etwa viermal so
        viele Seen wie bei 128.
        """
        height, width = heightmap.shape
        num_basins = len(lake_seeds)
        filtered_lake_map = np.full((height, width), -1, dtype=np.int32)
        if num_basins == 0:
            return filtered_lake_map, []

        spill_elevation = _compute_spill_elevations(heightmap, lake_map, num_basins)

        heights = heightmap.astype(np.float64, copy=False)
        # Spill-Höhe pro Pixel; Becken ohne endliche Spill-Höhe (kein Nachbar,
        # kein Kartenrand) bekommen -inf und werden dadurch nirgends überflutet.
        spill_lookup = np.where(np.isfinite(spill_elevation), spill_elevation, -np.inf)
        has_basin = lake_map >= 0
        basin_ids = np.where(has_basin, lake_map, 0)
        spill_per_pixel = spill_lookup[basin_ids]

        submerged = has_basin & (heights <= spill_per_pixel)
        submerged_basins = basin_ids[submerged]
        submerged_depth = spill_per_pixel[submerged] - heights[submerged]

        pixel_counts = np.bincount(submerged_basins, minlength=num_basins)
        depth_sums = np.bincount(submerged_basins, weights=submerged_depth, minlength=num_basins)

        cell_area = self._cell_area_m2()
        volumes = depth_sums * cell_area
        qualifies = (pixel_counts > 0) & (volumes >= self.lake_volume_threshold)

        # Neue, lückenlose Lake-IDs in aufsteigender Becken-Reihenfolge -
        # identisch zur Vergabe der alten Schleife (`len(valid_lakes)`).
        new_ids = np.full(num_basins, -1, dtype=np.int32)
        new_ids[qualifies] = np.arange(int(qualifies.sum()), dtype=np.int32)

        keep = submerged & qualifies[basin_ids]
        filtered_lake_map[keep] = new_ids[basin_ids[keep]]

        valid_lakes = [
            {
                'seed': lake_seeds[basin_id],
                'volume': float(volumes[basin_id]),
                'pixels': int(pixel_counts[basin_id]),
            }
            for basin_id in np.nonzero(qualifies)[0]
        ]

        return filtered_lake_map, valid_lakes


# Anteil des Maximal-Durchflusses dieser Karte, ab dem eine Zelle überhaupt
# als "wasserführend" gilt - siehe _river_flow_percentile_threshold().
#
# Empirisch gewählt (2026-07-27, 64² gefiltertes Rauschen, 200 Pipe-Schritte):
# der Median-Durchfluss liegt je nach Niederschlag bei 1.5-4% des
# Kartenmaximums. Eine Schwelle von 2% trennt dort sauber zwischen
# "Wasserlauf" und "numerischem Rinnsal" und lässt den Anteil wasserführender
# Zellen wieder auf das Klima reagieren - gemessen:
#   precip 25 -> 85% wasserführend, Fluss-Anteil 8.5% der Karte
#   precip 13 -> 79%                                7.9%
#   precip  5 -> 42%                                4.2%
#   precip  1 -> 38%                                3.8%
# Mit der vorherigen Bedingung `> 0` waren es in ALLEN vier Fällen exakt 100%
# wasserführend und damit stur 10.0% Fluss-Anteil, unabhängig vom Klima.
# Der river_abundance-Slider bleibt über seinen vollen Bereich wirksam
# (0.0 -> 0.4%, 0.3 -> 24%, 1.0 -> 79% der Karte bei precip 13).
WET_CELL_DISCHARGE_FRACTION = 0.02


def _river_flow_percentile_threshold(flow_accumulation, river_abundance):
    """
    Funktionsweise: Leitet einen absoluten Fluss-Klassifikations-Schwellwert aus
    einem 0..1-"Wieviel-Anteil-der-wasserführenden-Pixel-soll-Fluss-sein"-Regler
    UND der TATSÄCHLICHEN Durchfluss-Verteilung DIESER Karte ab.

    Aufgabe: river_abundance=0.0 -> nur die stärksten ~0.5% der wasserführenden
    Pixel gelten als Fluss (sehr restriktiv); river_abundance=1.0 -> praktisch
    jedes wasserführende Pixel gilt als Fluss (sehr freizügig). Die Perzentil-
    Grenzen werden leicht eingeklemmt (0.5..99.5), damit die Extremwerte nicht
    buchstäblich Minimum/Maximum eines einzelnen Pixels treffen.

    Die Nässe-Maske (2026-07-27): "wasserführend" heisst Durchfluss über
    WET_CELL_DISCHARGE_FRACTION des Kartenmaximums, nicht mehr schlicht
    `> 0`. Unter dem Pipe-Modell ist der Durchfluss an praktisch JEDER Zelle
    numerisch grösser als null (gemessen: 100.0% der Pixel) - das Perzentil
    lief damit über die gesamte Karte statt über die wasserführenden Zellen,
    und river_abundance=0.10 klassifizierte exakt 10% der GESAMTKARTE als
    Fluss. Niederschlag, Terrain und Seed hatten dadurch keinerlei Einfluss
    mehr darauf, wie viel Fluss auf der Karte ist; der Slider bestimmte es
    allein. Mit der Schwelle bezieht sich der Regler wieder auf das, was sein
    Name und seine Beschreibung sagen.

    Parameter: flow_accumulation (H,W) float array, river_abundance (float 0..1)
    Return: float - absoluter Schwellwert in derselben Einheit wie flow_accumulation
    """
    flow = np.asarray(flow_accumulation, dtype=np.float64)
    max_flow = float(flow.max()) if flow.size else 0.0
    if max_flow <= 0.0:
        return float('inf')

    wet = flow[flow > max_flow * WET_CELL_DISCHARGE_FRACTION]
    if wet.size == 0:
        return float('inf')

    percentile = 100.0 * (1.0 - np.clip(river_abundance, 0.0, 1.0))
    percentile = float(np.clip(percentile, 0.5, 99.5))
    return float(np.percentile(wet, percentile))


class PipeFlowSimulator:
    """
    Funktionsweise: Virtuelles-Rohre-Hydraulikmodell (Mei/Decaudin/Lefebvre
    2007) - kontinuierlicher 4-Richtungs-Fluss zwischen Nachbarzellen pro
    Schritt, mit einer pro Schritt neu berechneten Massenerhaltungs-Skalierung
    K (siehe _pipe_step_cpu). Ersetzt das vorherige D8-Steilster-Abstieg-
    Routing (_calculate_steepest_descent/_redirect_basin_flow_to_spill/
    _accumulate_upstream_flow, bis 2026-07-25) - Nutzer-Feedback: Flüsse
    rendern als 45°-Zickzack, Erosion sieht "extrem pixelig" aus, weil jede
    Zelle immer nur auf EINEN von 8 Nachbarn zeigt.

    Aufgabe: Senken füllen und laufen im Pipe-Modell von selbst über - kein
    diskreter "Sink"-Zustand und keine gesonderte Wasserscheiden-Umleitung
    mehr nötig (die K-Skalierung verhindert strukturell, dass eine Zelle mehr
    Wasser abgibt als sie hat; das GESAMTSYSTEM ist dadurch bereits
    unconditionally stable, kein zusätzlicher CFL-Zeitschritt-Check nötig).

    Zustandsdarstellung pro Zelle: Wassertiefe d (Meter) + 4-Richtungs-
    Ausfluss f=(f_L,f_R,f_T,f_B) (m³/s, nicht-negativ, Reihenfolge fix: Links/
    Rechts/Oben/Unten). Geschwindigkeit (v_x,v_y) wird JEDEN Schritt aus dem
    Fluss abgeleitet, nicht gespeichert (reine Diagnosegröße für Erosion +
    Anzeige).

    Kartenrand = offene Grenze (Geisterzelle mit derselben Geländehöhe wie
    die Randzelle selbst, aber Wassertiefe 0) - Wasser am Kartenrand
    entwässert dadurch mit einem Gefälle, das exakt der eigenen Wassertiefe
    entspricht (Standard-"open boundary"-Randbedingung für Flachwassermodelle,
    kein willkürliches Extra-Gefälle nötig, kein Sonderfall im Code - siehe
    Padding-Trick in _pipe_step_cpu).
    """

    # Virtuelle Rohr-Querschnittsfläche (m²) - Kalibrierungskonstante wie
    # ErosionSedimentationSystem.EROSION_RATE_SCALE, noch NICHT gegen die
    # laufende App kalibriert (Startwert, weiteres Nachjustieren anhand des
    # visuellen Eindrucks erwartet).
    PIPE_CROSS_SECTION_AREA = 0.6  # m²

    # "Effektive" Sekunden Simulationszeit, die auf das GESAMTE LOD-
    # Iterationsbudget (lod_iterations['flow']) verteilt werden - ergibt einen
    # adaptiven Zeitschritt dt = PIPE_TIME_SCALE_S / n_steps.
    #
    # Seit der Umstellung auf zeitbasierten Regen (2026-07-27, siehe
    # RAIN_TO_DEPTH_RATE unten) ist DIESER Wert - nicht die Schrittzahl - die
    # Grösse, die bestimmt, wie viel Wasser insgesamt auf die Karte fällt.
    # Die Schrittzahl steuert nur noch die zeitliche Auflösung, mit der
    # dieselbe Simulationsdauer durchgerechnet wird.
    PIPE_TIME_SCALE_S = 1800.0

    # Umrechnung precip_map (gH2O/m²) -> Meter Wasser PRO SEKUNDE
    # Simulationszeit.
    #
    # Bis 2026-07-27 war das ein Zuwachs PRO SCHRITT (RAIN_TO_DEPTH_SCALE),
    # nicht pro Sekunde. Damit hing die gesamte Wassermenge direkt an der
    # Iterationszahl: gemessen bei sonst identischen Bedingungen 0.54 m
    # mittlere Wassertiefe bei 50 Schritten gegenüber 4.18 m bei 400 - Faktor
    # 8 allein durch das LOD-Level. Da jede LOD-Stufe zusätzlich den
    # Wasserstand der Vorstufe übernimmt (previous_depth), summierte sich das
    # über die LOD-Kette weiter auf, und die Karte lief mit steigendem LOD
    # sichtbar voll. Der Zahlenwert ist so gewählt, dass 1800 s Simulationszeit
    # dieselbe Wassermenge einbringen wie früher die bei 128 px kalibrierte
    # Stufe von 200 Schritten (200 * 1/1000 / 1800 s).
    RAIN_TO_DEPTH_RATE = (200.0 / 1000.0) / PIPE_TIME_SCALE_S  # m Wasser pro (gH2O/m²) pro Sekunde

    # Umrechnung potentielle Verdunstung (gH2O/m²/Tag, siehe
    # EvaporationCalculator.calculate_potential_evaporation()) -> Meter Wasser
    # pro Sekunde Simulationszeit.
    #
    # Verdunstung ist neben dem Randabfluss die zweite Senke des Modells.
    # Ohne sie hatte eine Karte mit ausgeprägten Becken überhaupt keinen Weg,
    # Wasser wieder zu verlieren - der Wasserstand konnte dort nur steigen.
    #
    # Der Faktor ist bewusst eine STILISIERTE Kalibrierungskonstante, keine
    # physikalische Umrechnung: PIPE_TIME_SCALE_S (1800 s) ist "effektive"
    # Simulationszeit, keine reale Uhrzeit. Eine physikalisch exakte
    # Umrechnung (1 gH2O/m² = 1e-6 m Wassersäule, verteilt auf 86400 s)
    # ergäbe über 1800 s eine Verdunstung in der Größenordnung 1e-8 m - ein
    # Wert, der im Modell exakt nichts täte und den Regler damit nur
    # scheinbar wirksam machte. Stattdessen ist der Faktor so gewählt, dass
    # die Verdunstung bei den Default-Parametern (evaporation_base_rate
    # 0.002, gemässigtes Klima -> ~30-60 gH2O/m²/Tag) rund ein Viertel des
    # mittleren Regeneintrags aufzehrt: spürbar, aber nicht dominant.
    EVAPORATION_TO_DEPTH_RATE = RAIN_TO_DEPTH_RATE * 0.25 / 50.0

    # Mindest-Wassertiefe für die Geschwindigkeits-Ableitung (verhindert
    # Divisions-Explosion in Trockenzellen).
    MIN_DEPTH_FOR_VELOCITY = 1e-4  # m

    def __init__(self, shader_manager=None):
        self.shader_manager = shader_manager

    def simulate(self, heightmap, precip_map, potential_evaporation, lake_map, parameters,
                 lod_iterations, meters_per_pixel, previous_depth=None, previous_flux=None):
        """
        Funktionsweise: GPU-Pfad mit CPU-Fallback
        Aufgabe: Treibt lod_iterations['flow'] Schritte des Pipe-Modells.

        potential_evaporation: (H,W) gH2O/m²/Tag aus
            EvaporationCalculator.calculate_potential_evaporation() - die
            zweite Senke des Modells neben dem Randabfluss (siehe
            EVAPORATION_TO_DEPTH_RATE).
        previous_depth/previous_flux: konvergierter Zustand der vorherigen
            (kleineren) LOD-Stufe - None startet bei
            vollständig trockener Karte.

        Rückgabe: dict mit water_depth, velocity_x, velocity_y, discharge_map
        (|Netto-Fluss|, ersetzt die alte flow_accumulation-Semantik "kumulierte
        Regenmenge" durch "wie viel Wasser fließt hier gerade durch"),
        edge_outflow und evaporated_volume (beide Skalare, m³ über den
        gesamten Durchlauf - zusammen mit dem Regeneintrag die
        Massenbilanz) sowie depth_state/flux_state (interner Roh-Zustand für
        die nächste LOD-Stufe).
        """
        n_steps = max(1, int(lod_iterations["flow"]))
        dt_seconds = self.PIPE_TIME_SCALE_S / n_steps

        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "pipeFlowNetwork",
                    {
                        "heightmap": heightmap,
                        "precip_map": precip_map,
                        "potential_evaporation": potential_evaporation,
                        "previous_depth": previous_depth,
                        "previous_flux": previous_flux,
                        "meters_per_pixel": meters_per_pixel,
                        "dt_seconds": dt_seconds,
                        "n_steps": n_steps,
                        "pipe_cross_section_area": self.PIPE_CROSS_SECTION_AREA,
                        "rain_to_depth_rate": self.RAIN_TO_DEPTH_RATE,
                        "evaporation_to_depth_rate": self.EVAPORATION_TO_DEPTH_RATE,
                    },
                    parameters
                )
                if result.get("success"):
                    return result
            except Exception as e:
                logging.warning(f"GPU pipe flow simulation failed: {e}, falling back to CPU")

        # CPU-Pfad. Kein Simple-Fallback mehr (2026-07-27) - siehe
        # FlowNetworkBuilder.build_flow_network().
        return self._cpu_simulate(
            heightmap, precip_map, potential_evaporation, n_steps, dt_seconds,
            meters_per_pixel, previous_depth, previous_flux)

    def _cpu_simulate(self, heightmap, precip_map, potential_evaporation, n_steps, dt_seconds,
                       meters_per_pixel, previous_depth=None, previous_flux=None):
        height, width = heightmap.shape
        depth = previous_depth.astype(np.float64).copy() if previous_depth is not None \
            else np.zeros((height, width), dtype=np.float64)
        flux = previous_flux.astype(np.float64).copy() if previous_flux is not None \
            else np.zeros((height, width, 4), dtype=np.float64)

        # Quelle und Senke beide als Rate PRO SEKUNDE - mit dt multipliziert
        # ergibt das den Zuwachs/Verlust pro Schritt, unabhängig davon, in wie
        # viele Schritte dieselbe Simulationsdauer zerlegt wird.
        rain_rate = precip_map.astype(np.float64) * self.RAIN_TO_DEPTH_RATE
        evaporation_rate = np.maximum(
            0.0, potential_evaporation.astype(np.float64)) * self.EVAPORATION_TO_DEPTH_RATE

        rain_per_step = rain_rate * dt_seconds
        evaporation_per_step = evaporation_rate * dt_seconds

        pipe_area = self.PIPE_CROSS_SECTION_AREA
        cell_area = meters_per_pixel * meters_per_pixel
        total_edge_outflow_volume = 0.0
        total_evaporated_volume = 0.0

        vx = np.zeros((height, width), dtype=np.float64)
        vy = np.zeros((height, width), dtype=np.float64)
        flux_x_net = np.zeros((height, width), dtype=np.float64)
        flux_y_net = np.zeros((height, width), dtype=np.float64)

        for _ in range(n_steps):
            depth, flux, vx, vy, flux_x_net, flux_y_net, edge_outflow_step, evaporated_step = \
                self._pipe_step_cpu(
                    heightmap, depth, flux, rain_per_step, evaporation_per_step, dt_seconds,
                    pipe_area, meters_per_pixel, cell_area)
            total_edge_outflow_volume += edge_outflow_step
            total_evaporated_volume += evaporated_step

        discharge_map = np.sqrt(flux_x_net ** 2 + flux_y_net ** 2)
        return {
            "success": True,
            "water_depth": depth.astype(np.float32),
            "velocity_x": vx.astype(np.float32),
            "velocity_y": vy.astype(np.float32),
            "discharge_map": discharge_map.astype(np.float32),
            "edge_outflow": total_edge_outflow_volume,
            "evaporated_volume": total_evaporated_volume,
            "depth_state": depth.astype(np.float32),
            "flux_state": flux.astype(np.float32),
        }

    def _pipe_step_cpu(self, heightmap, depth, flux, rain_per_step, evaporation_per_step, dt,
                        pipe_area, pipe_length, cell_area):
        """
        EIN Zeitschritt des virtuellen-Rohre-Modells. Zwei logisch getrennte
        Teilschritte (Fluss-Update braucht Nachbarn AKTUELLE Tiefe,
        Tiefe-Update braucht Nachbarn GERADE berechneten Fluss - derselbe
        Grund, warum das Referenzprojekt thermalterrainflux-frag.glsl von
        thermalapply-frag.glsl trennt, und dieselbe Lehre wie
        ErosionSedimentationSystem._transport_sediment_optimized()s
        Gather-nicht-Scatter-Fix) - hier über vollständig materialisierte
        Zwischen-Arrays, kein In-Place-Mutation während des Sweeps.

        Reihenfolge innerhalb des Schritts: Regen rein -> Fluss -> Verdunstung
        raus. Die Verdunstung greift am Ende auf den bereits transportierten
        Wasserstand zu und ist durch den vorhandenen Wasserstand begrenzt
        (eine trockene Zelle kann nicht verdunsten) - deshalb wird die
        TATSÄCHLICH verdunstete Menge zurückgegeben, nicht die potentielle.
        """
        d1 = depth + rain_per_step

        b_pad = np.pad(heightmap.astype(np.float64), 1, mode='edge')
        d_pad = np.pad(d1, 1, mode='constant', constant_values=0.0)
        surface_pad = b_pad + d_pad
        surface = surface_pad[1:-1, 1:-1]
        surface_l = surface_pad[1:-1, 0:-2]
        surface_r = surface_pad[1:-1, 2:]
        surface_t = surface_pad[0:-2, 1:-1]
        surface_b = surface_pad[2:, 1:-1]

        accel = dt * pipe_area * GRAVITY / pipe_length
        f_l = np.maximum(0.0, flux[:, :, 0] + accel * (surface - surface_l))
        f_r = np.maximum(0.0, flux[:, :, 1] + accel * (surface - surface_r))
        f_t = np.maximum(0.0, flux[:, :, 2] + accel * (surface - surface_t))
        f_b = np.maximum(0.0, flux[:, :, 3] + accel * (surface - surface_b))

        # Massenerhaltungs-Skalierung K, JEDEN Schritt neu berechnet (siehe
        # Klassen-Docstring) - strukturelle Ersetzung des vorherigen
        # nachträglichen "Force-Settle am Loop-Ende"; macht das gesamte
        # System unconditionally stable (eine Zelle kann nie mehr Wasser
        # abgeben als sie hat).
        total_out = f_l + f_r + f_t + f_b
        k = np.minimum(1.0, (d1 * cell_area) / np.maximum(dt * total_out, 1e-12))
        f_l = f_l * k
        f_r = f_r * k
        f_t = f_t * k
        f_b = f_b * k

        # Zufluss von Nachbarn = deren jeweils ENTGEGENGESETZTER Ausfluss;
        # Geisterzellen am Rand liefern 0 Zufluss (offener Abfluss, siehe
        # Klassen-Docstring).
        fr_pad = np.pad(f_r, 1, mode='constant', constant_values=0.0)
        fl_pad = np.pad(f_l, 1, mode='constant', constant_values=0.0)
        fb_pad = np.pad(f_b, 1, mode='constant', constant_values=0.0)
        ft_pad = np.pad(f_t, 1, mode='constant', constant_values=0.0)

        in_from_left = fr_pad[1:-1, 0:-2]
        in_from_right = fl_pad[1:-1, 2:]
        in_from_top = fb_pad[0:-2, 1:-1]
        in_from_bottom = ft_pad[2:, 1:-1]

        incoming = in_from_left + in_from_right + in_from_top + in_from_bottom
        outgoing = f_l + f_r + f_t + f_b

        d_transported = np.maximum(0.0, d1 + dt / cell_area * (incoming - outgoing))

        # Verdunstung als zweite Senke - nie mehr als vorhanden ist.
        evaporated = np.minimum(d_transported, evaporation_per_step)
        d2 = d_transported - evaporated

        d_avg = np.maximum(0.5 * (d1 + d2), self.MIN_DEPTH_FOR_VELOCITY)
        flux_x_net = 0.5 * (in_from_left - f_l + f_r - in_from_right)
        flux_y_net = 0.5 * (in_from_top - f_t + f_b - in_from_bottom)
        vx = flux_x_net / (pipe_length * d_avg)
        vy = flux_y_net / (pipe_length * d_avg)

        # Rand-Abfluss dieses Schritts (m³) - Fluss durch die äusserste
        # Zellreihe/-spalte, der keinen realen Nachbarn mehr hat.
        edge_outflow_step = dt * (
            float(f_l[:, 0].sum()) + float(f_r[:, -1].sum()) +
            float(f_t[0, :].sum()) + float(f_b[-1, :].sum())
        )
        evaporated_step = float(evaporated.sum()) * cell_area

        new_flux = np.stack([f_l, f_r, f_t, f_b], axis=-1)
        return d2, new_flux, vx, vy, flux_x_net, flux_y_net, edge_outflow_step, evaporated_step


class FlowNetworkBuilder:
    """
    Funktionsweise: Baut das Flussnetzwerk über das virtuelle-Rohre-
    Hydraulikmodell (siehe PipeFlowSimulator) statt des früheren D8-
    Steilster-Abstieg-Routings.
    Aufgabe: Erstellt flow_accumulation (jetzt Durchfluss-Betrag statt
    kumulierter Regenmenge, siehe PipeFlowSimulator.simulate()) und
    water_biomes_map mit Flusssystemen, die nicht mehr auf 45°-Schritte
    begrenzt sind.
    """

    def __init__(self, river_abundance=0.10, shader_manager=None):
        self.river_abundance = river_abundance
        self.shader_manager = shader_manager
        self.pipe_simulator = PipeFlowSimulator(shader_manager=shader_manager)

    def build_flow_network(self, heightmap, precip_map, potential_evaporation, lake_map, parameters,
                            lod_iterations, meters_per_pixel, previous_depth=None, previous_flux=None):
        """
        Funktionsweise: Treibt PipeFlowSimulator über lod_iterations['flow']
        Schritte, klassifiziert anschließend Wasserkörper aus dem
        resultierenden Durchfluss-Betrag.
        Aufgabe: lake_map bleibt NUR für die Wasserkörper-Klassifikation
        relevant (_classify_water_bodies) - eine gesonderte Wasserscheiden-
        Umleitung (wie beim alten D8-Modell) ist nicht mehr nötig, Senken
        füllen/laufen im Pipe-Modell strukturell von selbst über (siehe
        PipeFlowSimulator-Docstring).
        Rückgabe: dict (siehe PipeFlowSimulator.simulate()), ergänzt um
        "water_biomes_map".

        Der frühere `except Exception -> _simple_flow_network()`-Fallback ist
        entfernt (2026-07-27): er setzte die "Wassertiefe" einfach mit der
        Niederschlagsmenge gleich und lieferte ein Fluss-Netzwerk ohne einen
        einzigen Fluss - ein Ergebnis, das nach aussen wie eine gelungene
        Simulation aussah, aber keine war, und das jeden echten Fehler in
        PipeFlowSimulator verdeckte. Damit entfiel auch der letzte Leser von
        `rain_threshold`, weshalb dieser Parameter samt Slider entfernt wurde.
        Der GPU->CPU-Fallback innerhalb von PipeFlowSimulator.simulate()
        bleibt bestehen (erwartete Umgebungsbedingung).
        """
        sim_result = self.pipe_simulator.simulate(
            heightmap, precip_map, potential_evaporation, lake_map, parameters,
            lod_iterations, meters_per_pixel,
            previous_depth=previous_depth, previous_flux=previous_flux)
        sim_result["water_biomes_map"] = self._classify_water_bodies(
            sim_result["discharge_map"], lake_map)
        return sim_result

    # Vielfache der Creek-Schwelle, ab denen ein Wasserlauf als River bzw.
    # Grand River gilt (Klassifikationsstufen 2 und 3).
    RIVER_THRESHOLD_FACTOR = 4.0
    GRAND_RIVER_THRESHOLD_FACTOR = 20.0

    def _classify_water_bodies(self, flow_accumulation, lake_map):
        """
        Klassifiziert Wasserkörper basierend auf Flussgröße.
        Vektorisiert (2026-07-27) - vorher eine Python-Doppelschleife über die
        ganze Karte; np.select bildet dieselbe if/elif-Kaskade exakt ab.
        Seen (lake_map >= 0) haben Vorrang und werden nicht überschrieben.
        """
        # Schwelle wird PRO KARTE aus der tatsächlichen flow_accumulation-
        # Verteilung + self.river_abundance abgeleitet (siehe
        # _river_flow_percentile_threshold()) statt eines festen absoluten
        # Werts - dadurch bleibt der ANTEIL der wasserführenden Pixel, der als
        # Fluss gilt, unabhängig von Kartengröße/Seed/Niederschlagsmenge.
        creek_threshold = _river_flow_percentile_threshold(flow_accumulation, self.river_abundance)

        is_lake = lake_map >= 0
        water_biomes_map = np.select(
            [
                is_lake,
                flow_accumulation >= creek_threshold * self.GRAND_RIVER_THRESHOLD_FACTOR,
                flow_accumulation >= creek_threshold * self.RIVER_THRESHOLD_FACTOR,
                flow_accumulation >= creek_threshold,
            ],
            [4, 3, 2, 1],
            default=0,
        ).astype(np.uint8)
        return water_biomes_map


class ManningFlowCalculator:
    """
    Funktionsweise: Malt Fluss-Breite in water_biomes_map basierend auf dem
    Kanal-Querschnitt (calculate_channel_width/paint_channel_width).

    Bis 2026-07-25 (D8 -> Pipe-Modell-Umbau, Stufe 3) löste diese Klasse
    zusätzlich unabhängig die Manning-Gleichung für flow_speed/water_depth
    (calculate_flow_properties/_cpu_manning_calculation/
    _optimize_channel_geometry/_simple_flow_calculation) - das ist jetzt
    redundant: der Pipe-Modell (PipeFlowSimulator) liefert diese Größen
    bereits als echte, simulierte Werte direkt aus water.flow_network. Eine
    zweite, unabhängige Schätzung derselben physikalischen Größe würde
    zwangsläufig vom tatsächlich simulierten Wasserstand abweichen - dieselbe
    "zwei Wahrheiten für dieselbe Größe"-Falle, die bereits einmal in
    _calculate_stream_power_erosion()s alter, algebraisch kürzenden
    Geschwindigkeits-Approximation gejagt wurde (siehe dortiger Docstring).
    cross_section wird jetzt bei jedem Aufruf (siehe HydrologySystemGenerator.
    _calc_manning_flow()) frisch aus der Kontinuitätsgleichung (Fläche =
    Durchfluss/Geschwindigkeit) der simulierten Werte abgeleitet.
    """

    # Tal-Breite-Analyse (siehe _analyze_valley_width_full): 8 Strahlen à max.
    # 24 Schritte, ein Strahl endet an der ersten Zelle, die mehr als
    # VALLEY_RIM_RISE_M über dem Startpunkt liegt (= Talrand).
    VALLEY_RAY_COUNT = 8
    VALLEY_MAX_STEPS = 25          # exklusive Obergrenze, wie range(1, 25)
    VALLEY_RIM_RISE_M = 20.0

    def __init__(self, shader_manager=None):
        self.shader_manager = shader_manager

    @classmethod
    def _analyze_valley_width_full(cls, heightmap):
        """
        Tal-Breite für JEDE Zelle in einem Rutsch (Rückgabe (H,W) float32,
        Einheit Pixel).

        Ersetzt die vorherige Pro-Pixel-Methode _analyze_valley_width(), die
        pro Fluss-Pixel 8 x 24 Einzelabfragen in Python machte - bei 512² und
        einem Fluss-Anteil von 10% waren das ~2.0 s, wachsend mit Pixelzahl x
        Fluss-Anteil. Da (dx, dy) pro Strahl konstant sind, ist `int(dx*step)`
        ein KONSTANTER ganzzahliger Versatz pro (Strahl, Schritt) - der Scan
        lässt sich damit als 8 x 24 verschobene Vollbild-Vergleiche schreiben.

        Semantisch identisch zur alten Fassung, inklusive beider
        Abbruchbedingungen: ein Strahl, der aus der Karte läuft, trägt 0 bei
        (kein Talrand gefunden), und ein Strahl endet beim ERSTEN Treffer.
        Rückgabe ist wie zuvor `max über alle Strahlen * 2`.
        """
        height, width = heightmap.shape
        heights = heightmap.astype(np.float64, copy=False)
        rim_threshold = heights + cls.VALLEY_RIM_RISE_M

        max_distance = np.zeros((height, width), dtype=np.float64)

        for angle in np.linspace(0, 2 * np.pi, cls.VALLEY_RAY_COUNT):
            dx, dy = np.cos(angle), np.sin(angle)

            # active: Strahl läuft an dieser Zelle noch (weder Treffer noch
            # Kartenrand); distance: Schrittzahl des Treffers, 0 = kein Treffer.
            active = np.ones((height, width), dtype=bool)
            distance = np.zeros((height, width), dtype=np.float64)

            for step in range(1, cls.VALLEY_MAX_STEPS):
                offset_x, offset_y = int(dx * step), int(dy * step)

                # Zellen, für die dieser Versatz aus der Karte führt: Strahl
                # endet dort (entspricht dem `break` der alten Schleife).
                in_bounds = np.zeros((height, width), dtype=bool)
                y_from, y_to = max(0, -offset_y), height - max(0, offset_y)
                x_from, x_to = max(0, -offset_x), width - max(0, offset_x)
                if y_from < y_to and x_from < x_to:
                    in_bounds[y_from:y_to, x_from:x_to] = True
                active &= in_bounds
                if not np.any(active):
                    break

                sampled = np.zeros((height, width), dtype=np.float64)
                sampled[y_from:y_to, x_from:x_to] = heights[
                    y_from + offset_y:y_to + offset_y, x_from + offset_x:x_to + offset_x]

                hit = active & (sampled > rim_threshold)
                distance[hit] = step
                active &= ~hit

            max_distance = np.maximum(max_distance, distance)

        return (max_distance * 2.0).astype(np.float32)

    # Tal-Breite (Pixel) -> Breite-zu-Tiefe-Verhältnis des Kanals. Enge
    # Kerbtäler zwingen einen schmalen, tiefen Querschnitt, weite Talböden
    # erlauben einen breiten, flachen.
    VALLEY_WIDTH_TO_RATIO = ((5.0, 2.0), (20.0, 5.0), (np.inf, 10.0))

    def calculate_channel_width(self, cross_section, heightmap, stream_mask):
        """
        Funktionsweise: Leitet die Flussbreite aus dem bereits vorhandenen
        Querschnitt (cross_section, aus der Manning-Kontinuitätsgleichung) her,
        statt sie separat zu schätzen: area = width * depth und
        width = ratio * depth ergibt width = sqrt(ratio * area). Läuft
        einheitlich für GPU- wie CPU-cross_section, da die Ratio nur von der
        Heightmap abhängt, nicht vom Fließweg.
        Aufgabe: Nur stream_mask-Pixel mit positivem Querschnitt bekommen einen
        Wert, alle anderen bleiben 0.
        Return: (height, width) float32, Flussbreite in realen Metern
        """
        valley_width = self._analyze_valley_width_full(heightmap)

        ratio = np.empty(valley_width.shape, dtype=np.float32)
        remaining = np.ones(valley_width.shape, dtype=bool)
        for upper_bound, ratio_value in self.VALLEY_WIDTH_TO_RATIO:
            in_band = remaining & (valley_width < upper_bound)
            ratio[in_band] = ratio_value
            remaining &= ~in_band

        area = np.asarray(cross_section, dtype=np.float32)
        applicable = stream_mask & (area > 0)
        return np.where(applicable, np.sqrt(ratio * np.maximum(area, 0.0)), 0.0).astype(np.float32)

    def paint_channel_width(self, water_biomes_map, channel_width, meters_per_pixel, flow_speed=None):
        """
        Funktionsweise: Dilatiert jedes Fluss-Zentrallinien-Pixel (Creek/River/
        Grand River, Werte 1-3 in water_biomes_map) um seine berechnete
        Flussbreite/2 via distance_transform_edt (nearest-centerline-Lookup) -
        übernimmt die Klassifikations-Stufe vom nächsten Zentrallinien-Pixel,
        malt also nur die räumliche Ausdehnung, klassifiziert nicht neu.
        Lake-Pixel (4) und bereits klassifizierte Fluss-Pixel bleiben
        unverändert (kein Überschreiben mit einer schwächeren Stufe).
        Aufgabe: channel_width ist in realen Metern (aus der Manning-
        Kontinuitätsgleichung) - Umrechnung in Pixel-Radius über
        `meters_per_pixel`. Dieser Wert wird seit 2026-07-27 vom Aufrufer
        übergeben (HydrologySystemGenerator._meters_per_pixel(), eine Quelle
        für alle Water-Knoten) statt hier aus einer km-Angabe und der
        Array-Höhe rekonstruiert zu werden - die Rekonstruktion setzte
        stillschweigend voraus, dass die übergebene Karte exakt die
        LOD-Zielauflösung hat.

        flow_speed (optional, (H,W) m/s aus calculate_flow_properties): schnell
        fließende Reaches malen sich schmaler als ihr roher channel_width-Wert
        nahelegt - reine Akkumulationsmenge (flow_accumulation, treibt
        channel_width über cross_section) sagt nichts über die Fließ-
        geschwindigkeit aus, aber bei gleicher Wassermenge kann sich schnell
        fließendes Wasser nicht so breit "aufstauen" wie langsames (Nutzer-
        Beobachtung: "dort wo die Wasserbewegung schneller ist, kann nicht so
        viel Wasser sich ansammeln"). Skaliert NUR die gemalte Breite, NICHT
        die Klassifikations-Schwellen selbst (die bleiben akkumulationsbasiert
        - physikalisch richtig für "wie viel Wasser", siehe Docstring oben,
        "malt nur die räumliche Ausdehnung, klassifiziert nicht neu").
        None (Default) reproduziert exakt das alte Verhalten.
        Return: neue water_biomes_map (Kopie, Original bleibt unverändert)
        """
        stream_mask = (water_biomes_map >= 1) & (water_biomes_map <= 3)
        if not np.any(stream_mask):
            return water_biomes_map

        channel_width_visual = channel_width
        if flow_speed is not None:
            reference_speed = float(np.median(flow_speed[stream_mask]))
            if reference_speed > 1e-6:
                # Boden 0.3: schnelle Reaches wirken schmaler, verschwinden aber
                # nie ganz. Median über alle Fluss-Pixel als Referenz, damit der
                # Faktor relativ zur tatsächlichen Verteilung dieser Karte skaliert
                # statt gegen eine feste, kartengrößen-unabhängige Konstante.
                effective_width_factor = np.clip(
                    reference_speed / np.maximum(flow_speed, 0.01), 0.3, 1.0)
                channel_width_visual = channel_width * effective_width_factor

        distances, nearest_indices = distance_transform_edt(~stream_mask, return_indices=True)
        nearest_y, nearest_x = nearest_indices
        nearest_width_m = channel_width_visual[nearest_y, nearest_x]
        nearest_radius_px = (nearest_width_m / 2.0) / meters_per_pixel

        painted = water_biomes_map.copy()
        paintable = (water_biomes_map == 0) & (distances <= nearest_radius_px)
        painted[paintable] = water_biomes_map[nearest_y[paintable], nearest_x[paintable]]
        return painted


class DropletErosionSystem:
    """
    Funktionsweise: Droplet-basierte hydraulische Erosion (Sebastian-Lague-
    Referenzalgorithmus, https://github.com/SebLague/Hydraulic-Erosion,
    Formeln gegen Erosion.cs verifiziert) - ersetzt die vorherige
    ErosionSedimentationSystem (Stream-Power-Formel + Semi-Lagrange/
    MacCormack-Sedimenttransport entlang des eingefrorenen Pipe-Modell-
    Geschwindigkeitsfelds, bis 2026-07-25).

    Nutzer-Feedback, das zu diesem Umbau führte: das alte Eulersche System
    hatte keine Rückkopplung INNERHALB eines Durchlaufs - eine Zelle, die
    sich durch Ablagerung füllte, machte das Antriebs-Geschwindigkeitsfeld
    nicht flacher, da dieses vor der Schleife einmalig eingefroren wurde. Ein
    Partikel-Modell löst das strukturell: jeder Partikel mutiert die
    Arbeits-Heightmap DIREKT waehrend seines Lebenswegs - eine Zelle, die
    sich fuellt, ist beim naechsten Abtasten sofort messbar flacher, ohne
    dass ein expliziter Hoehen-Deckel noetig waere.

    Vollstaendig entkoppelt von water.flow_network/PipeFlowSimulator - Partikel
    spawnen GLEICHVERTEILT ueber die Karte, unabhaengig von simuliertem
    Niederschlag/Abfluss. Laeuft dadurch VOR water.lake_detection/
    water.flow_network in der Ausfuehrungsreihenfolge (siehe
    _execute_generation()) - Erosion modelliert geologische Zeit, der
    Wasserkreislauf simuliert den heutigen Zustand AUF dem bereits erodierten
    Gelaende.

    Gleichverteilter Spawn (2026-07-27, vorher hoehen-gewichtet): Regen faellt
    ueberall gleich. Die Erosion nimmt bergab VON SELBST zu, weil ein Partikel
    dort Geschwindigkeit und Sedimentkapazitaet aufgebaut hat und sich
    Fliesswege buendeln - Gipfel und Kaemme bleiben weitgehend unberuehrt, weil
    dorthin nur die Partikel gelangen, die direkt dort "vom Himmel fallen".
    Der vorherige hoehen-gewichtete Spawn liess 69% aller Partikel in der
    oberen Gelaendehaelfte starten und zerkraterte damit genau die Gipfel
    (Nutzer-Report "Berge sind zerfressen an den Spitzen").

    DREI-STUFEN-VERFAHREN pro Lauf (siehe simulate_erosion_sedimentation()):
    abwechselnd Senken fuellen und erodieren, mit einer Fuellung als ABSCHLUSS.
    Ohne die Fuellung graebt sich jedes Partikel seine eigene Grube und die
    Karte zerfaellt in einzelne Krater ohne Abfluss; ohne die abschliessende
    Fuellung reisst die letzte Erosionsrunde sie wieder auf. Gemessen bei 128²:
    375 lokale Minima ohne Fuellung, 462 bei Fuellung nur VOR der Erosion,
    0 mit abschliessender Fuellung - bei gleichzeitig 32% Ebenen-Anteil.
    Siehe fill_depressions().

    Massenerhaltung: bewusst NICHT exakt erzwungen (Nutzer-Vorgabe 2026-07-25:
    "ich brauche keine strikte Massenerhaltung, ich will das wie Sebastian
    Lague"). Bei JEDEM Abbruchpfad eines Partikels (Kartenrand, Max-
    Lebenszeit, Null-Richtung) wird die verbleibende Restfracht VERWORFEN,
    identisch zur Referenz (Erosion.cs verwirft sie beim Partikel-Tod
    ebenfalls stillschweigend) - `sedimentation_map.sum()` liegt dadurch
    strukturell UNTER `erosion_map.sum()`, das ist beabsichtigt. Eine
    frühere Fassung dieser Klasse hatte hier stattdessen zwanghaft
    abgesetzt, um exakte Massenerhaltung zu erzwingen (uebertragen aus einer
    Vorgabe fuer das separate Pipe-Flow-System) - genau dieser Zwang war die
    Ursache fuer sichtbare kuenstliche Huegel (Nutzer-Report), da die
    komplette Restfracht eines Partikels an EINER Stelle abgeladen wurde.
    Entfernt, seit klar ist, dass diese Klasse KEINE exakte Massenerhaltung
    braucht.
    """

    # Traegheit der Fliessrichtung: dir = dir*INERTIA - grad*(1-INERTIA).
    #
    # Auf steilem Gelaende ist |grad| gross und dominiert den Term - das
    # Partikel laeuft praktisch gerade bergab. Auf einer flachen Ebene ist
    # grad ~ 0, dort dominiert die vorherige Richtung - das Partikel wandert
    # und kruemmt sich langsam, also MAEANDERT es. Der Effekt reguliert sich
    # damit von selbst und braucht keine Fallunterscheidung "Berg oder Ebene".
    #
    # 0.30 statt des Referenzwerts 0.05 (Nutzer-Vorgabe 2026-07-27: "Fluesse
    # ... meandern dort"): 0.05 ist fast reines Gradientenfolgen und erzeugt
    # auch auf der Ebene schnurgerade Rinnen. Gemessen bei 128² mit 4
    # Durchgaengen: Ebenen-Anteil 34.8% -> 38.3%, zusammenhaengendes Kanalnetz
    # bleibt gross (147 px).
    DROPLET_INERTIA = 0.30
    DROPLET_SEDIMENT_MIN_CAPACITY_M = 0.01
    DROPLET_ERODE_SPEED = 0.3
    DROPLET_EVAPORATE_SPEED = 0.01
    DROPLET_INITIAL_WATER_VOLUME = 1.0
    DROPLET_INITIAL_SPEED = 1.0
    DROPLET_MIN_WATER = 0.01

    # Partikeldichte (Partikel pro Pixel), fuer die
    # DROPLET_INITIAL_WATER_VOLUME kalibriert ist. Weicht die tatsaechliche
    # Dichte davon ab, wird das Wasser pro Tropfen umgekehrt proportional
    # skaliert - siehe initial_water_volume().
    DROPLET_REFERENCE_DENSITY = 0.3

    # --- Partikelgeometrie ---
    #
    # LEBENSWEG in Metern: ein Partikel soll dieselbe REALE Strecke
    # zuruecklegen, egal bei welcher Aufloesung gerechnet wird - sonst
    # schrumpfen Erosionsstrukturen relativ zur Karte, je feiner das Gitter
    # (bei 30 festen Schritten deckt ein Partikel auf 64 px die halbe Karte
    # ab, auf 1024 px nur 3%).
    #
    # Der Bezugswert ist gegenueber der ersten Fassung VERDREIFACHT (30 -> 90
    # Schritte bei der Default-Konfiguration 128 px auf 10 km). Der Lebensweg
    # ist der staerkste Hebel fuer eine verzweigte, baumartige Erosionskarte:
    # Partikel muessen weit genug laufen, um sich zu GEMEINSAMEN Kanaelen zu
    # buendeln. Gemessen bei 128², 40k Partikel, 4 Durchgaenge - groesste
    # zusammenhaengende Komponente des Kanalnetzes: 30 Schritte -> 83 px,
    # 90 Schritte -> 153 px (Nutzer-Report (d) "Erosionskarte ist eine
    # homogene Flaeche anstatt ... wie ein Baum mit Aesten").
    DROPLET_MAX_LIFETIME_M = 90 * (10_000.0 / 128.0)
    DROPLET_MAX_LIFETIME_STEPS_MIN = 20
    DROPLET_MAX_LIFETIME_STEPS_MAX = 200

    # PINSELRADIUS dagegen in PIXELN, bewusst NICHT metrisch: er ist ein
    # numerischer Glaettungskernel (ueber wie viele Nachbarzellen der Abtrag
    # EINES Ereignisses verschmiert wird), keine Landschaftsgroesse. Metrisch
    # definiert wuchs er bei 512 px auf 12 px an - die Pinselflaeche und damit
    # die Laufzeit steigt quadratisch (452 statt 13 Zellen pro Ereignis), was
    # hohe Partikelzahlen unbezahlbar macht. Und er zerfaserte das Ergebnis:
    # r=1 konzentriert zwar stark (Top-5%-Anteil 0.39), zerlegt das Kanalnetz
    # aber in Bruchstuecke (39 px statt 147 px). 2 px ist der gemessene
    # Kompromiss.
    DROPLET_ERODE_RADIUS_PX = 2.0
    # Exponent auf die normierte Hoehe fuer die Spawn-Wahrscheinlichkeit.
    # 0.0 = GLEICHVERTEILT (Default seit 2026-07-27), 1.0 = linear zur Hoehe.
    #
    # Vorher 1.0 ("Partikel fallen auf die Berge"): gemessen starteten damit
    # 69% aller Partikel in der oberen Gelaendehaelfte und zerkraterten genau
    # die Gipfel. Regen faellt real ueberall gleich; dass die Erosion trotzdem
    # bergab zunimmt, ergibt sich von selbst (Geschwindigkeit, Kapazitaet,
    # Buendelung der Fliesswege) - siehe Klassen-Docstring.
    # Bleibt als Regler erhalten, falls ein orographischer Effekt gewuenscht
    # ist (Werte um 0.2-0.4 waeren dafuer plausibel).
    DROPLET_SPAWN_HEIGHT_POWER = 0.0
    # Stilisierte "Spielgefuehl"-Gravitationskonstante DER REFERENZ (Erosion.cs:
    # "public float gravity = 4;") - BEWUSST NICHT die reale 9.81 m/s^2
    # (Modul-Konstante GRAVITY, siehe PipeFlowSimulator): alle anderen
    # Referenz-Default-Werte (capacityFactor, erodeSpeed etc.) sind als ein
    # zusammenhaengendes, aufeinander kalibriertes Parameter-Set um DIESEN
    # stilisierten Wert herum gewaehlt - ein Tausch gegen die reale
    # Erdbeschleunigung wuerde die Geschwindigkeits-/Kapazitaets-Balance des
    # gesamten Sets verschieben, ohne die uebrigen Werte neu zu kalibrieren.
    DROPLET_GRAVITY = 4.0
    # Relief-relativer Deckel fuer Abtrag/Ablagerung PRO SCHRITT (nicht in der
    # Referenz - zusaetzliche Vorsichtsmassnahme gegen einen einzelnen
    # pathologischen Schritt bei sehr steilem Gelaende).
    #
    # Von 2% auf 0.5% des Reliefs gesenkt (2026-07-27). 2% bedeutete bei
    # 4000 m Relief 80 m Abtrag in EINEM Schritt - kein Sicherheitsnetz,
    # sondern eine zweite Erosionsquelle. Der Deckel ist zugleich der
    # staerkste Hebel dafuer, dass sich die Erosion in LINIEN sammelt statt
    # sich flaechig zu verteilen; gemessen bei 128²: Anteil der Erosion in den
    # staerksten 5% der Zellen 0.231 (bei 2%) gegenueber 0.347 (bei 0.2%).
    # 0.5% ist der Kompromiss aus Kanalbildung und noch sichtbarer Wirkung
    # einzelner Partikel.
    DROPLET_CAP_RELIEF_FRACTION = 0.005
    DROPLET_CAP_MIN_M = 0.05

    # HINWEIS: Ein Budget pro Zelle und Schritt als fester Anteil des Reliefs
    # war hier einmal implementiert und ist wieder entfallen. Es wirkte nicht
    # monoton - gemessen bei 256 px, 5 Partikeln/Pixel, Relief 916 m, hoechste
    # Gelaendenadel ueber allen 8 Nachbarn:
    #
    #     ohne Budget      187 m
    #     2% des Reliefs   280 m   (schlechter als ohne)
    #     5% des Reliefs   134 m
    #     10% des Reliefs   37 m   (dafuer Sedimentation 28 058 statt 10 304 m)
    #
    # Ein Deckel, der das Ergebnis in beide Richtungen verschieben kann, ist
    # kein Regler, sondern ein zusaetzlicher freier Parameter. An seine Stelle
    # tritt die geometrische Grenze, die die Pro-Partikel-Regel ohnehin schon
    # kennt (siehe _scatter_deposit_vec/_scatter_erode_vec) - sie braucht
    # keinen kalibrierten Wert.
    # Haerte-Kopplung (nicht in der Referenz vorhanden): ERODIERBARKEIT als
    # Anteil des geometrisch erlaubten Abtrags, Wertebereich (0, 1].
    #
    # Der Faktor multipliziert das Ergebnis der `-delta_height`-Klemme. Genau
    # deshalb ist die Obergrenze 1.0 und nicht mehr 3.0: ein Faktor > 1 wuerde
    # die Klemme aufheben und das Partikel unter seinen eigenen Abfluss graben
    # lassen - das war (zusammen mit erosion_strength) die Ursache der Krater.
    # Ein Faktor <= 1 ist dagegen unbedenklich: das Ergebnis bleibt
    # <= -delta_height, die Zusage haelt.
    #
    # Bei der Referenzhaerte (50) ist die Erodierbarkeit 1.0 (das volle
    # Gefaelle darf abgetragen werden), haerteres Gestein liegt darunter.
    # Weiches Gestein bekommt keinen Bonus ueber das geometrisch Moegliche
    # hinaus - es widersteht nur nicht.
    #
    # Der EXPONENT ist der eigentliche Hebel der Haerte-Differenzierung, nicht
    # die Untergrenze: bei linearer Kennlinie (Exponent 1) liefert Haerte 95
    # den Faktor 50/95 = 0.53, was ueber jeder sinnvollen Untergrenze liegt -
    # ein Absenken von MIN_FACTOR aendert dort gemessen exakt nichts.
    # Zusaetzlich wirkt eine negative Rueckkopplung: haerteres Gestein bleibt
    # steiler, wodurch das Gefaelle (und damit die Klemme) groesser bleibt und
    # den Unterschied wieder einholt. Mit Exponent 1 blieb vom Faktor 1.9 im
    # Einzelereignis nur 1.17 im Gesamtabtrag uebrig; der Exponent gleicht
    # diese Kompression aus.
    DROPLET_ERODE_CAP_HARDNESS_REFERENCE = 50.0
    DROPLET_ERODE_CAP_HARDNESS_EXPONENT = 2.5
    DROPLET_ERODE_CAP_HARDNESS_MIN_FACTOR = 0.05
    DROPLET_ERODE_CAP_HARDNESS_MAX_FACTOR = 1.0

    # Standard-Anzahl Durchgänge, falls der Aufrufer keinen Wert liefert -
    # siehe simulate_erosion_sedimentation() für die Bedeutung.
    DEFAULT_EROSION_PASSES = 4

    def __init__(self, erosion_strength=1.0, sediment_capacity_factor=4.0, deposit_speed=0.3,
                 shader_manager=None, erosion_passes=DEFAULT_EROSION_PASSES):
        self.erosion_strength = erosion_strength
        self.capacity_factor = sediment_capacity_factor
        # Name bewusst "settling_velocity" beibehalten (nicht "deposit_speed")
        # - minimaler Diff in HydrologySystemGenerator._update_parameters(),
        # die dieses Attribut bereits unter diesem Namen synchronisiert.
        self.settling_velocity = deposit_speed
        self.erosion_passes = erosion_passes
        self.shader_manager = shader_manager

    def simulate_erosion_sedimentation(self, heightmap, hardness_map, parameters, lod_iterations,
                                        meters_per_pixel=1.0):
        """
        Führt den vollständigen Erosionslauf aus und liefert
        (erosion_map, sedimentation_map). Die übergebene heightmap wird NICHT
        mutiert (es wird auf einer internen float64-Kopie gearbeitet).

        ABLAUF - abwechselnd füllen und erodieren, mit einer Füllung als
        ABSCHLUSS:

            für jeden der N Durchgänge:
                Senken füllen        -> als Sedimentation verbucht
                Teilmenge der Partikel laufen lassen
            abschliessend Senken füllen

        Warum diese Reihenfolge, gemessen bei 128² mit 40k Partikeln:

            nur erodieren                              375 Krater, 19.7% Ebenen
            füllen -> erodieren                        417 Krater, 31.4% Ebenen
            4x (füllen -> erodieren)                   462 Krater, 32.2% Ebenen
            4x (füllen -> erodieren) -> füllen           0 Krater, 32.3% Ebenen

        Füllen VOR dem Erodieren erzeugt die Ebenen, aber die Erosion danach
        reisst die Gruben wieder auf - erst die abschliessende Füllung liefert
        beides. Und das Füllen ZWISCHEN den Durchgängen ist nicht nur Kosmetik:
        es stellt vor jedem Durchgang eine durchgehende Entwässerung her, so
        dass die Partikel des nächsten Durchgangs bis zum Kartenrand
        durchlaufen können, statt in den Gruben des vorherigen zu enden. Das
        ist die Voraussetzung dafür, dass sich überhaupt zusammenhängende
        Bachlinien bilden.

        `erosion_passes` steuert N. N=1 bedeutet: einmal erodieren, einmal
        abschliessend füllen.
        """
        num_droplets = max(1, int(lod_iterations['erosion_particles']))
        passes = max(1, int(self.erosion_passes))

        work = heightmap.astype(np.float64).copy()
        erosion_map = np.zeros_like(work)
        sedimentation_map = np.zeros_like(work)

        # Relief EINMAL aus dem Ausgangsgelände - der Deckel soll über den
        # gesamten Lauf konstant bleiben und nicht mitwandern, während sich
        # das Relief durch die eigene Erosion verändert.
        relief = float(heightmap.max() - heightmap.min())
        cap_per_step = max(self.DROPLET_CAP_MIN_M, self.DROPLET_CAP_RELIEF_FRACTION * relief)

        droplets_per_pass = max(1, num_droplets // passes)
        # Wasser pro Tropfen aus der GESAMT-Partikelzahl, nicht pro Durchgang -
        # die Durchgaenge teilen sich dieselbe Regenmenge auf.
        initial_water = self.initial_water_volume(num_droplets, work.size)

        for pass_index in range(passes):
            self._apply_fill(work, sedimentation_map)
            spawn_positions = self._sample_spawn_positions(
                work, droplets_per_pass, parameters, pass_index=pass_index)
            self._run_droplets(work, hardness_map, erosion_map, sedimentation_map,
                                spawn_positions, meters_per_pixel, cap_per_step, parameters,
                                initial_water)

        # Abschliessende Füllung - ohne sie bleiben die Gruben des letzten
        # Durchgangs stehen (gemessen: 462 statt 0 lokale Minima).
        self._apply_fill(work, sedimentation_map)

        # BEIDE KARTEN SIND KUMULIERTER DURCHSATZ, NICHT NETTO-AENDERUNG.
        #
        # Dieselbe Zelle kann in aufeinanderfolgenden Schritten abgetragen und
        # wieder beschickt werden; beide Vorgaenge addieren sich in ihrer
        # jeweiligen Karte auf. An einem Konvergenzpunkt - typischerweise
        # dort, wo ein Kanal die Karte verlaesst - schleusen ueber alle
        # Durchgaenge tausende Partikelwege Material durch dieselbe Zelle.
        # Gemessen bei 256 px, 5 Partikeln/Pixel, Relief 916 m:
        #
        #     Zelle (227,14): 10 304 m Sedimentation, 10 125 m Erosion
        #                     -> netto 179 m, Nadel ueber den Nachbarn: -0.3 m
        #
        # Im Gelaende steht dort also nichts Auffaelliges. Genau das war die
        # auf der 512er-Karte gemeldete "Sedimentations-Spitze": ein
        # Durchsatz-Hotspot neben einem Median von 107 m, der die lineare
        # Farbskala saettigt.
        #
        # Die naheliegende Antwort - beide Karten nach Vorzeichen in eine
        # Netto-Aenderung aufloesen - ist GEMESSEN VERWORFEN. Sie waere
        # bilanziell exakt (`heightmap - erosion + sedimentation` bleibt
        # unveraendert, weil nur die Differenz eingeht), loescht aber genau
        # das Merkmal, das der Nutzer sehen will: in einem Kanal sind Abtrag
        # und Beschickung annaehernd gleich gross, netto bleibt dort fast
        # nichts stehen. Bei 128 px, 81 920 Partikeln:
        #
        #                              brutto   netto
        #     groesstes Kanalnetz      210 px   95 px
        #     Top-5%-Anteil (Baum)      0.555   0.314
        #     Erosions-Schwerpunkt      0.457   0.600   (Gelaende 0.531)
        #
        # Netto zeigt also die Verzweigung nicht mehr und verlagert die
        # Erosion ueber die mittlere Gelaendehoehe - das Gegenteil des
        # Zielbilds. Die Karten bleiben deshalb brutto; die grosse Spannweite
        # ist eine Frage der Farbskala und wird in der Anzeige logarithmisch
        # aufgeloest (wie bei der Niederschlagskarte).
        return erosion_map.astype(np.float32), sedimentation_map.astype(np.float32)

    def simulate_droplets_only(self, heightmap, hardness_map, spawn_positions, meters_per_pixel,
                                cap_per_step=None):
        """
        EIN Partikel-Durchgang OHNE Senkenfüllung, auf einer Kopie der
        Heightmap. Return: (erosion_map, sedimentation_map) als float32.

        Gedacht für alles, was gezielt das PARTIKEL-Verhalten prüfen will und
        durch die Füllung verfälscht würde - insbesondere die Massenbilanz:
        die Füllung wird als Sedimentation verbucht, `sedimentation.sum()`
        läge damit über `erosion.sum()` und die Aussage "Restfracht am
        Kartenrand wird verworfen" wäre nicht mehr messbar. Genutzt von
        smoke_test_water_edge_sediment.py und
        smoke_test_water_drainage_erosion.py.

        Der reguläre Weg ist simulate_erosion_sedimentation() - nur der
        enthält die Senkenfüllung und damit das, was die Karte tatsächlich
        bekommt.
        """
        work = heightmap.astype(np.float64).copy()
        erosion_map = np.zeros_like(work)
        sedimentation_map = np.zeros_like(work)
        if cap_per_step is None:
            relief = float(heightmap.max() - heightmap.min())
            cap_per_step = max(self.DROPLET_CAP_MIN_M, self.DROPLET_CAP_RELIEF_FRACTION * relief)

        self._cpu_simulate(work, hardness_map, erosion_map, sedimentation_map,
                            spawn_positions, meters_per_pixel, cap_per_step)
        return erosion_map.astype(np.float32), sedimentation_map.astype(np.float32)


    @staticmethod
    def _apply_fill(work, sedimentation_map):
        """Senken in `work` auffüllen und die Füllhöhe als Sedimentation
        verbuchen. Mutiert beide Arrays in place, damit die Buchhaltung
        `work == heightmap - erosion_map + sedimentation_map` exakt erhalten
        bleibt (darauf verlässt sich der Aufrufer, siehe
        DataLODManager.get_terrain_data_combined())."""
        fill_amount = fill_depressions(work)
        if fill_amount.any():
            work += fill_amount
            sedimentation_map += fill_amount

    def _run_droplets(self, work, hardness_map, erosion_map, sedimentation_map,
                       spawn_positions, meters_per_pixel, cap_per_step, parameters,
                       initial_water):
        """Einen Durchgang Partikel über `work` laufen lassen - GPU bevorzugt,
        CPU als Rückfallebene. Beide mutieren work/erosion_map/
        sedimentation_map in place.

        GPU-Pfad ist implementiert und registriert
        (shader_manager._dispatch_droplet_erosion, DISPATCH_TABLE-Eintrag
        ("water", "dropletErosion"); Shader: dropletErosionStep.comp +
        dropletErosionApply.comp). Das Wettlauf-Problem vieler gleichzeitig in
        dieselbe Höhen-Textur schreibender Partikel-Threads ist dort über
        Lockstep-Wellen und Fixed-Point-Integer-Atomics gelöst. Er ist bei
        grossen Auflösungen der vorgesehene Weg, nicht nur eine Beschleunigung
        - der CPU-Pfad bleibt die verifizierbare Referenz.

        Die Senkenfüllung zwischen den Durchgängen läuft immer auf der CPU;
        der GPU-Pfad gibt dafür pro Durchgang eine Höhenkarte zurück, was
        gegenüber der Partikel-Simulation nicht ins Gewicht fällt.
        """
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "dropletErosion",
                    {
                        "heightmap": work, "hardness_map": hardness_map,
                        "spawn_positions": spawn_positions, "meters_per_pixel": meters_per_pixel,
                        "erosion_strength": self.erosion_strength,
                        "capacity_factor": self.capacity_factor,
                        "deposit_speed": self.settling_velocity,
                        "cap_per_step": cap_per_step,
                        "initial_water": initial_water,
                    },
                    parameters
                )
                if result.get("success"):
                    erosion_map += result["erosion_map"]
                    sedimentation_map += result["sedimentation_map"]
                    work -= result["erosion_map"]
                    work += result["sedimentation_map"]
                    return
            except Exception as e:
                logging.warning(f"GPU droplet erosion failed: {e}, falling back to CPU")

        # CPU-Pfad. Kein Simple-Fallback mehr (2026-07-27): er lieferte
        # stillschweigend "gar keine Erosion" und liess damit einen echten
        # Programmfehler wie eine Karte ohne Erosionsbedarf aussehen.
        self._cpu_simulate_lockstep(work, hardness_map, erosion_map, sedimentation_map,
                                     spawn_positions, meters_per_pixel, cap_per_step,
                                     initial_water)

    def _cpu_simulate_lockstep(self, work, hardness_map, erosion_map, sedimentation_map,
                                spawn_positions, meters_per_pixel, cap_per_step,
                                initial_water=None):
        """
        EINEN Durchgang Partikel laufen lassen - alle Partikel GLEICHZEITIG,
        Schritt für Schritt (Lockstep). Mutiert work/erosion_map/
        sedimentation_map in place.

        Identisches Verfahren wie der GPU-Pfad
        (shaders/water/dropletErosionStep.comp + dropletErosionApply.comp):
        pro Runde bewegen sich alle lebenden Partikel einen Schritt, alle
        lesen dabei den Geländestand vom ANFANG der Runde und ihre Beiträge
        werden erst danach verrechnet. Damit rechnen CPU und GPU dasselbe
        Modell - vorher lief die CPU strikt sequentiell (jedes Partikel sah
        die Spuren aller vorherigen sofort), die GPU im Lockstep, und die
        beiden Pfade konnten sich nie exakt entsprechen.

        Warum überhaupt: die sequentielle Fassung braucht für 20.000 Partikel
        bei 128² rund 53 s. Die vom Nutzer gewünschten ~80.000 Partikel wären
        damit auch am finalen LOD unbrauchbar, und gerade die hohe
        Partikelzahl ist das, was ein fein verästeltes Kanalnetz erzeugt
        (statistische Konvergenz vieler schwacher Wege statt weniger starker).
        Die sequentielle Fassung bleibt als Referenz erhalten
        (_cpu_simulate(), genutzt von simulate_droplets_only()).

        Numerisch bewusst in float64, damit die Buchhaltung
        `work == heightmap - erosion_map + sedimentation_map` exakt bleibt.
        """
        size_y, size_x = work.shape
        max_steps, erode_radius_px = self.resolve_pixel_geometry(meters_per_pixel)
        brush_offsets, brush_weights = self._build_erosion_brush(erode_radius_px)
        hardness = hardness_map.astype(np.float64)

        pos_x = np.ascontiguousarray(spawn_positions[:, 0], dtype=np.float64)
        pos_y = np.ascontiguousarray(spawn_positions[:, 1], dtype=np.float64)
        dir_x = np.zeros_like(pos_x)
        dir_y = np.zeros_like(pos_y)
        speed = np.full_like(pos_x, self.DROPLET_INITIAL_SPEED)
        if initial_water is None:
            initial_water = self.initial_water_volume(pos_x.size, work.size)
        water = np.full_like(pos_x, initial_water)

        # Scratch-Puffer für die Zell-Grenze (siehe _cell_budget_scale):
        # geforderte Summe und geometrische Grenze je Zelle. Einmal angelegt
        # und über alle Schritte wiederverwendet; die Streu-Funktionen nullen
        # sie jeweils nur an den berührten Stellen wieder.
        requested_scratch = np.zeros_like(work)
        sediment = np.zeros_like(pos_x)
        alive = np.ones(pos_x.shape, dtype=bool)

        for _ in range(max_steps):
            if not alive.any():
                break
            idx = np.nonzero(alive)[0]

            # Geometrische Grenze dieses Schritts, EINMAL fuer die ganze Karte
            # aus dem eingefrorenen Gelaendestand - alle Partikel des Schritts
            # rechnen gegen genau diesen Stand (siehe _deposit_headroom).
            deposit_headroom = self._deposit_headroom(work)

            px, py = pos_x[idx], pos_y[idx]
            x0, y0, fx, fy = self._bilinear_corners_vec(work, px, py)
            height, grad_x, grad_y = self._height_and_gradient_vec(work, x0, y0, fx, fy)

            new_dir_x = dir_x[idx] * self.DROPLET_INERTIA - grad_x * (1.0 - self.DROPLET_INERTIA)
            new_dir_y = dir_y[idx] * self.DROPLET_INERTIA - grad_y * (1.0 - self.DROPLET_INERTIA)
            length = np.hypot(new_dir_x, new_dir_y)

            # Richtungslose Partikel sterben (Referenz-Verhalten: keine
            # Zufalls-Neuausrichtung).
            has_direction = length >= 1e-9
            safe_len = np.where(has_direction, length, 1.0)
            new_dir_x /= safe_len
            new_dir_y /= safe_len

            new_x = px + new_dir_x
            new_y = py + new_dir_y
            in_bounds = (new_x >= 0) & (new_x < size_x - 1) & (new_y >= 0) & (new_y < size_y - 1)

            # Ein Partikel wirkt in diesem Schritt nur, wenn es eine Richtung
            # hat UND im Gebiet bleibt - sonst endet sein Weg hier und die
            # Restfracht wird verworfen (identisch zur Referenz Erosion.cs).
            acts = has_direction & in_bounds
            if acts.any():
                act = np.nonzero(acts)[0]
                a_idx = idx[act]
                ax0, ay0, afx, afy = x0[act], y0[act], fx[act], fy[act]
                new_height, _, _ = self._height_and_gradient_vec(
                    work, *self._bilinear_corners_vec(work, new_x[act], new_y[act]))
                delta_height = new_height - height[act]

                hardness_here = np.maximum(1.0, self._bilinear_sample_vec(hardness, ax0, ay0, afx, afy))
                hardness_factor = np.clip(
                    (self.DROPLET_ERODE_CAP_HARDNESS_REFERENCE / hardness_here)
                    ** self.DROPLET_ERODE_CAP_HARDNESS_EXPONENT,
                    self.DROPLET_ERODE_CAP_HARDNESS_MIN_FACTOR,
                    self.DROPLET_ERODE_CAP_HARDNESS_MAX_FACTOR)

                sed_act = sediment[a_idx]
                capacity = np.maximum(
                    -delta_height * speed[a_idx] * water[a_idx]
                    * self.capacity_factor * self.erosion_strength,
                    self.DROPLET_SEDIMENT_MIN_CAPACITY_M)

                depositing = (sed_act > capacity) | (delta_height > 0)

                # --- Ablagern ---
                deposit_amount = np.where(
                    delta_height > 0,
                    np.minimum(delta_height, sed_act),
                    (sed_act - capacity) * self.settling_velocity)
                deposit_amount = np.clip(deposit_amount, 0.0, cap_per_step)
                deposit_amount = np.where(depositing, deposit_amount, 0.0)

                # --- Eintiefen (die -delta_height-Klemme gewinnt immer, siehe
                # _walk_one_droplet() fuer die volle Begruendung) ---
                erode_amount = np.minimum(
                    (capacity - sed_act) * self.DROPLET_ERODE_SPEED, -delta_height) * hardness_factor
                erode_amount = np.clip(erode_amount, 0.0, cap_per_step)
                erode_amount = np.where(depositing, 0.0, erode_amount)

                # Die Ablagerung wird auf das begrenzt, was EINE Zelle in
                # diesem Schritt aufnehmen kann (siehe _deposit_headroom);
                # beide Streu-Operationen liefern die tatsaechlich angewandte
                # Menge zurueck - nur die geht in die Sediment-Bilanz des
                # Partikels ein.
                actually_deposited = self._scatter_deposit_vec(
                    work, sedimentation_map, ax0, ay0, afx, afy, deposit_amount,
                    deposit_headroom, requested_scratch)
                actually_eroded = self._scatter_erode_vec(
                    work, erosion_map, px[act], py[act],
                    erode_amount, brush_offsets, brush_weights, requested_scratch)

                sediment[a_idx] = sed_act - actually_deposited + actually_eroded
                speed[a_idx] = np.sqrt(np.maximum(
                    0.0, speed[a_idx] ** 2 + delta_height * self.DROPLET_GRAVITY))
                water[a_idx] *= (1.0 - self.DROPLET_EVAPORATE_SPEED)
                pos_x[a_idx] = new_x[act]
                pos_y[a_idx] = new_y[act]
                dir_x[a_idx] = new_dir_x[act]
                dir_y[a_idx] = new_dir_y[act]

            # Sterbefaelle dieses Schritts eintragen.
            dead = idx[~acts]
            alive[dead] = False
            still = idx[acts]
            alive[still] = (water[still] >= self.DROPLET_MIN_WATER) & np.isfinite(speed[still])

    @staticmethod
    def _bilinear_corners_vec(field, pos_x, pos_y):
        """Vektorisierte Fassung von _bilinear_corners() - identische Formel
        und identisches Klemmen auf den gueltigen Eckbereich."""
        size_y, size_x = field.shape
        x0 = np.clip(np.floor(pos_x).astype(np.intp), 0, size_x - 2)
        y0 = np.clip(np.floor(pos_y).astype(np.intp), 0, size_y - 2)
        return x0, y0, pos_x - x0, pos_y - y0

    @staticmethod
    def _bilinear_sample_vec(field, x0, y0, fx, fy):
        """Vektorisierte bilineare Abtastung eines Skalarfelds."""
        v_nw = field[y0, x0]
        v_ne = field[y0, x0 + 1]
        v_sw = field[y0 + 1, x0]
        v_se = field[y0 + 1, x0 + 1]
        return (v_nw * (1 - fx) * (1 - fy) + v_ne * fx * (1 - fy) +
                v_sw * (1 - fx) * fy + v_se * fx * fy)

    @classmethod
    def _height_and_gradient_vec(cls, work, x0, y0, fx, fy):
        """Vektorisierte Fassung von _height_and_gradient() (Referenz-Formel
        aus Erosion.cs::CalculateHeightAndGradient)."""
        h_nw = work[y0, x0]
        h_ne = work[y0, x0 + 1]
        h_sw = work[y0 + 1, x0]
        h_se = work[y0 + 1, x0 + 1]
        gradient_x = (h_ne - h_nw) * (1 - fy) + (h_se - h_sw) * fy
        gradient_y = (h_sw - h_nw) * (1 - fx) + (h_se - h_ne) * fx
        height = (h_nw * (1 - fx) * (1 - fy) + h_ne * fx * (1 - fy) +
                  h_sw * (1 - fx) * fy + h_se * fx * fy)
        return height, gradient_x, gradient_y

    @staticmethod
    def _deposit_headroom(work):
        """
        Wie weit darf JEDE Zelle in EINEM Lockstep-Schritt steigen? Bis zur
        Hoehe ihres hoechsten Nachbarn.

        Das ist dieselbe Aussage, die die Pro-Partikel-Regel schon trifft -
        `min(delta_height, sediment)` fuellt bis zur Hoehe der Zielzelle - nur
        auf Zellebene formuliert und damit auch dann gueltig, wenn K Partikel
        im selben Schritt gegen DENSELBEN eingefrorenen Gelaendestand rechnen.
        Ohne sie bekommt die Zelle das K-fache (gemessen bei 512 px: bis zu
        541 Tropfen auf einer Zelle in einem Schritt).

        Ein sequentieller Lauf (SebLague/Hydraulic-Erosion Erosion.cs, hier
        _walk_one_droplet) braucht die Regel nicht: dort sieht Partikel N die
        Wirkung von N-1 sofort, sein eigenes `delta_height` ist entsprechend
        kleiner, die Zelle saettigt von selbst. Lagues GPU-Fassung schreibt
        ohne Atomics und verliert Beitraege - auch das verhindert Spitzen,
        umgeht das Problem aber, statt es zu loesen. Wer verlustfrei
        akkumuliert (np.add.at bzw. imageAtomicAdd), braucht sie explizit.

        Die Grenze hat KEINEN kalibrierten Parameter: sie folgt allein aus dem
        Gelaende. Ein frueher hier stehender Deckel als fester Anteil des
        Reliefs wirkte nicht einmal monoton (siehe Kommentar bei
        DROPLET_CAP_RELIEF_FRACTION).

        Gemessen bei 128 px, 81 920 Partikeln, gegen den Zustand ohne Grenze -
        die Regel verbessert JEDE Kennzahl, nicht nur die Spitzen:

            hoechste Gelaendenadel   66.2 m -> 17.2 m
            Top-5%-Anteil (Baum)      0.468 -> 0.555
            groesstes Kanalnetz      177 px -> 210 px
            Erosions-Schwerpunkt      0.443 -> 0.457  (Gelaende 0.531)

        BEWUSST NUR AUF DER ABLAGERUNGSSEITE. Die spiegelbildliche Regel fuer
        den Abtrag ("grabe nicht unter den niedrigsten Nachbarn") ist zwar
        genauso herleitbar, trifft aber genau die Rinnensohlen: eine Zelle im
        Kanal LIEGT bereits unter ihren Nachbarn, ihre Grenze ist also nahe
        null, und die Eintiefung koennte sich nur noch als Welle vom
        Kartenrand nach oben fortpflanzen. Gemessen kostete das das halbe
        Kanalnetz (210 -> 56 px) und schob den Erosions-Schwerpunkt von 0.457
        auf 0.610, also ueber die mittlere Gelaendehoehe - das Gegenteil des
        Zielbilds. Zu tief gegrabene Zellen sind ausserdem harmlos, weil die
        abschliessende Senkenfuellung sie wieder schliesst; ueberhoehte Zellen
        raeumt dagegen nichts weg. Diese Asymmetrie ist der Grund, warum die
        Grenze hier nur einseitig gilt.
        """
        highest = maximum_filter(work, footprint=_NEIGHBOR_FOOTPRINT_8, mode='nearest')
        return np.maximum(highest - work, 0.0)

    @staticmethod
    def _cell_budget_scale(requested, headroom, targets):
        """
        Skalierungsfaktor je Zelle, damit die SUMME aller Beitraege eines
        Schritts die geometrische Grenze dieser Zelle (siehe _step_headroom)
        nicht ueberschreitet: k = min(1, Grenze/gefordert).

        Wortgleiches Vorbild im selben Projekt: die Massenerhaltungs-Skalierung
        im Pipe-Modell (PipeFlowSimulator._pipe_step_cpu,
        `k = min(1, (d1*cell_area) / (dt*total_out))`), die dort verhindert,
        dass eine Zelle mehr Wasser abgibt als sie hat.

        `requested` ist ein Scratch-Puffer in Kartengroesse; er wird am Ende
        nur an den TATSAECHLICH beruehrten Stellen wieder genullt (O(Partikel)
        statt O(Pixel)), damit er ueber alle Schritte wiederverwendbar bleibt.
        """
        return np.minimum(1.0, headroom[targets] / np.maximum(requested[targets], 1e-12))

    @classmethod
    def _scatter_deposit_vec(cls, work, sedimentation_map, x0, y0, fx, fy, amount,
                              headroom, requested):
        """
        Ablagerung bilinear auf die 4 Eckpunkte verteilen, begrenzt durch die
        GEOMETRISCHE AUFNAHMEFAEHIGKEIT der Zelle in diesem Schritt.

        Warum diese Grenze noetig ist: im Lockstep lesen alle Partikel eines
        Schritts denselben Gelaendestand und ihre Beitraege werden anschliessend
        verlustfrei aufsummiert. Die Pro-Partikel-Regel "hoechstens bis zur
        Hoehe der naechsten Zelle auffuellen" (min(delta_height, sediment))
        wird damit von K Partikeln gleichzeitig gegen DENSELBEN Stand
        ausgewertet - die Zelle bekommt am Ende das K-fache. Gemessen bei
        512 px: bis zu 541 Tropfen auf einer Zelle in einem Schritt.

        Der sequentielle Referenzpfad (_walk_one_droplet, wie
        SebLague/Hydraulic-Erosion Erosion.cs) hat das Problem nicht: dort
        sieht Partikel N die Ablagerung von N-1 sofort, sein eigenes
        `delta_height` ist entsprechend kleiner, die Regel ist
        selbstbegrenzend. Lagues GPU-Fassung schreibt ohne Atomics, verliert
        dadurch Beitraege und kann ebenfalls keine Spitze aufbauen - beides
        umgeht das Problem, statt es zu loesen.

        `headroom` ist die Pro-Partikel-Grenze, gegen die aggregiert wird:

        * Fuellt das Partikel eine Senke (`delta_height > 0`), ist sie
          `delta_height` - mehr als bis auf die Hoehe der Zielzelle kann die
          Zelle auch sequentiell nicht steigen, egal wie viele Partikel
          kommen. Aggregiert wird per Maximum, nicht per Summe.
        * Laedt das Partikel dagegen Ueberfracht auf einem Hang ab
          (`delta_height <= 0`), gibt es keine geometrische Saettigung: jedes
          Partikel bringt eigenes Material mit, und genau diese Summe baut die
          Ebenen. Dieser Zweig wird mit `inf` uebergeben, also nicht begrenzt.

        Rueckgabe: die TATSAECHLICH abgelagerte Menge je Partikel - nur die
        darf dem Partikel vom Sediment abgezogen werden, sonst stimmt die
        Bilanz nicht mehr.
        """
        applied = np.zeros_like(amount)
        if not np.any(amount > 0):
            return applied

        corners = ((y0, x0, (1 - fx) * (1 - fy)), (y0, x0 + 1, fx * (1 - fy)),
                   (y0 + 1, x0, (1 - fx) * fy), (y0 + 1, x0 + 1, fx * fy))

        # Durchgang 1: geforderte Menge je Zelle aufsummieren.
        for ty, tx, weight in corners:
            np.add.at(requested, (ty, tx), amount * weight)

        # Durchgang 2: anteilig herunterskaliert anwenden.
        for ty, tx, weight in corners:
            contribution = amount * weight * cls._cell_budget_scale(
                requested, headroom, (ty, tx))
            np.add.at(work, (ty, tx), contribution)
            np.add.at(sedimentation_map, (ty, tx), contribution)
            applied += contribution

        for ty, tx, _ in corners:
            requested[ty, tx] = 0.0
        return applied

    @classmethod
    def _scatter_erode_vec(cls, work, erosion_map, pos_x, pos_y,
                            amount, brush_offsets, brush_weights, requested):
        """
        Abtrag ueber den kreisfoermigen Pinsel verteilen, begrenzt durch
        dieselbe geometrische Regel wie die Ablagerung (siehe
        _scatter_deposit_vec() fuer die Begruendung).

        `requested` wird hier nur noch als Scratch-Puffer der Symmetrie halber
        gefuehrt; eine Zell-Grenze gibt es auf der Abtragsseite bewusst NICHT
        (Begruendung in _deposit_headroom: sie trifft die Rinnensohlen und
        kostete gemessen das halbe Kanalnetz).

        Rueckgabe: die TATSAECHLICH abgetragene Menge je Partikel. Sie kann
        unter `amount` liegen, wenn der Pinsel den Kartenrand ueberlappt (dort
        liegende Gewichte fallen weg, sie werden nicht umverteilt). Nur dieser
        Wert darf dem Partikel als Sediment gutgeschrieben werden.
        """
        actually_eroded = np.zeros_like(amount)
        if not np.any(amount > 0):
            return actually_eroded

        size_y, size_x = work.shape
        cx = np.rint(pos_x).astype(np.intp)
        cy = np.rint(pos_y).astype(np.intp)

        # Gueltige (Partikel, Pinselzelle)-Paare einmal bestimmen und fuer
        # beide Durchgaenge wiederverwenden.
        targets = []
        for (dy, dx), weight in zip(brush_offsets, brush_weights):
            ty, tx = cy + dy, cx + dx
            valid = (ty >= 0) & (ty < size_y) & (tx >= 0) & (tx < size_x) & (amount > 0)
            if not valid.any():
                continue
            targets.append((ty[valid], tx[valid], valid, weight))

        for ty, tx, valid, weight in targets:
            contribution = amount[valid] * weight
            np.add.at(work, (ty, tx), -contribution)
            np.add.at(erosion_map, (ty, tx), contribution)
            actually_eroded[valid] += contribution

        return actually_eroded

    def _sample_spawn_positions(self, heightmap, num_droplets, parameters, pass_index=0):
        """
        Spawn-Positionen der Partikel. RNG deterministisch aus water_seed
        geseedet (bestehendes Seed-Muster dieses Projekts), damit Seed+Karte
        reproduzierbare Partikel-Wege ergeben; `pass_index` geht in den Seed
        ein, damit die Durchgänge einer Mehrfach-Runde nicht alle exakt
        dieselben Startpunkte benutzen.

        DROPLET_SPAWN_HEIGHT_POWER == 0 (Default) bedeutet gleichverteilt -
        dann wird direkt und ohne Gewichtungs-Umweg gezogen (deutlich
        billiger als rng.choice() mit expliziter Wahrscheinlichkeitsverteilung
        über size² Zellen, was bei 80k Partikeln spürbar ist).
        Ein Wert > 0 gewichtet wie zuvor zur Höhe hin.
        """
        seed = int(parameters.get('water_seed', 12345)) if parameters else 12345
        rng = np.random.RandomState(seed + 1000 * int(pass_index))

        h = heightmap.astype(np.float64)
        size_y, size_x = h.shape

        if self.DROPLET_SPAWN_HEIGHT_POWER <= 0.0:
            pos_x = rng.uniform(0.0, size_x - 1.0001, num_droplets)
            pos_y = rng.uniform(0.0, size_y - 1.0001, num_droplets)
            return np.stack([pos_x, pos_y], axis=-1)

        norm = (h - h.min()) / max(h.max() - h.min(), 1e-9)
        # +Epsilon: tiefste Punkte spawnen noch gelegentlich, verhindert
        # einen Nullvektor auf komplett flacher Karte.
        prob = np.power(norm, self.DROPLET_SPAWN_HEIGHT_POWER) + 1e-6
        prob_flat = (prob / prob.sum()).ravel()

        flat_indices = rng.choice(h.size, size=num_droplets, replace=True, p=prob_flat)
        spawn_y, spawn_x = np.unravel_index(flat_indices, h.shape)

        jitter_x = rng.uniform(0.0, 1.0, num_droplets)
        jitter_y = rng.uniform(0.0, 1.0, num_droplets)
        pos_x = np.clip(spawn_x.astype(np.float64) + jitter_x, 0.0, size_x - 1.0001)
        pos_y = np.clip(spawn_y.astype(np.float64) + jitter_y, 0.0, size_y - 1.0001)
        return np.stack([pos_x, pos_y], axis=-1)

    @staticmethod
    def _build_erosion_brush(radius):
        """Kreisfoermiger Erosions-Pinsel (Gewicht linear mit dem Abstand
        abfallend, auf Summe 1 normiert) - EINMAL pro Simulationslauf
        vorberechnet, nicht pro Partikel/Schritt (reine Funktion des
        Radius)."""
        offsets = []
        weights = []
        r = int(np.ceil(radius))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= radius:
                    offsets.append((dy, dx))
                    weights.append(max(0.0, 1.0 - dist / radius))
        weights = np.array(weights, dtype=np.float64)
        total = weights.sum()
        if total > 0:
            weights /= total
        return offsets, weights

    @staticmethod
    def _bilinear_corners(field, pos_x, pos_y):
        """Liefert (x0,y0,fx,fy) fuer die bilineare Abtastung von field an
        (pos_x,pos_y) - geteilte Ecken-Berechnung fuer Hoehe/Gradient/
        Ablagerung."""
        size_y, size_x = field.shape
        x0 = int(np.floor(pos_x))
        y0 = int(np.floor(pos_y))
        x0 = min(max(x0, 0), size_x - 2)
        y0 = min(max(y0, 0), size_y - 2)
        fx = pos_x - x0
        fy = pos_y - y0
        return x0, y0, fx, fy

    @classmethod
    def _bilinear_value(cls, field, pos_x, pos_y):
        """Bilineare Abtastung eines Skalarfelds (z.B. hardness_map) an
        (pos_x,pos_y)."""
        x0, y0, fx, fy = cls._bilinear_corners(field, pos_x, pos_y)
        v_nw, v_ne = field[y0, x0], field[y0, x0 + 1]
        v_sw, v_se = field[y0 + 1, x0], field[y0 + 1, x0 + 1]
        return (v_nw * (1 - fx) * (1 - fy) + v_ne * fx * (1 - fy) +
                v_sw * (1 - fx) * fy + v_se * fx * fy)

    @classmethod
    def _height_and_gradient(cls, work, pos_x, pos_y):
        """Bilineare Hoehen-/Gradienten-Abtastung (Referenz-Formel, siehe
        Erosion.cs::CalculateHeightAndGradient)."""
        x0, y0, fx, fy = cls._bilinear_corners(work, pos_x, pos_y)
        h_nw, h_ne = work[y0, x0], work[y0, x0 + 1]
        h_sw, h_se = work[y0 + 1, x0], work[y0 + 1, x0 + 1]
        gradient_x = (h_ne - h_nw) * (1 - fy) + (h_se - h_sw) * fy
        gradient_y = (h_sw - h_nw) * (1 - fx) + (h_se - h_ne) * fx
        height = (h_nw * (1 - fx) * (1 - fy) + h_ne * fx * (1 - fy) +
                  h_sw * (1 - fx) * fy + h_se * fx * fy)
        return height, gradient_x, gradient_y

    @classmethod
    def _deposit_bilinear(cls, work, accumulator_map, pos_x, pos_y, amount):
        """Verteilt `amount` bilinear-gewichtet auf die 4 Eckpunkte der
        aktuellen Zelle - mutiert `work` (Arbeits-Heightmap, gibt dem
        naechsten Abtast-Schritt sofort die Rueckkopplung) UND
        `accumulator_map` (erosion_map ODER sedimentation_map, je nach
        Aufrufer) gleichzeitig."""
        if amount <= 0:
            return
        x0, y0, fx, fy = cls._bilinear_corners(work, pos_x, pos_y)
        w_nw = (1 - fx) * (1 - fy)
        w_ne = fx * (1 - fy)
        w_sw = (1 - fx) * fy
        w_se = fx * fy

        work[y0, x0] += amount * w_nw
        work[y0, x0 + 1] += amount * w_ne
        work[y0 + 1, x0] += amount * w_sw
        work[y0 + 1, x0 + 1] += amount * w_se

        accumulator_map[y0, x0] += amount * w_nw
        accumulator_map[y0, x0 + 1] += amount * w_ne
        accumulator_map[y0 + 1, x0] += amount * w_sw
        accumulator_map[y0 + 1, x0 + 1] += amount * w_se

    @staticmethod
    def _erode_brush(work, erosion_map, pos_x, pos_y, amount, brush):
        """Traegt `amount` ueber den kreisfoermigen Pinsel ab - mutiert
        `work` UND `erosion_map`. Gibt die TATSAECHLICH abgetragene Menge
        zurueck (kann bei Pinsel-Ueberlappung mit dem Kartenrand kleiner als
        `amount` sein, da dort liegende Pinsel-Gewichte schlicht wegfallen,
        nicht umverteilt werden) - der Aufrufer darf NUR diesen tatsaechlichen
        Wert dem Partikel als aufgenommenes Sediment gutschreiben, sonst
        waere die Massenbilanz nicht mehr exakt."""
        if amount <= 0:
            return 0.0
        size_y, size_x = work.shape
        offsets, weights = brush
        cx = int(round(pos_x))
        cy = int(round(pos_y))
        total_eroded = 0.0
        for (dy, dx), weight in zip(offsets, weights):
            ty, tx = cy + dy, cx + dx
            if 0 <= ty < size_y and 0 <= tx < size_x:
                delta = amount * weight
                work[ty, tx] -= delta
                erosion_map[ty, tx] += delta
                total_eroded += delta
        return total_eroded

    @classmethod
    def initial_water_volume(cls, num_droplets, pixel_count):
        """
        Wassermenge pro Tropfen, umgekehrt proportional zur Partikeldichte.

        Der Gedanke: die Partikelzahl soll bestimmen, wie FEIN die Erosion
        aufgeloest wird - nicht, WIE VIEL insgesamt erodiert wird. Dieselbe
        Regenmenge auf mehr Tropfen verteilt bedeutet weniger Wasser pro
        Tropfen; die Sedimentkapazitaet haengt direkt daran
        (capacity ~ -delta_height * speed * water * factor), womit jeder
        einzelne Tropfen entsprechend schwaecher wirkt.

        Ohne diese Kopplung kehrt sich der erwartete Effekt um. Gemessen bei
        128², 4 Durchgaengen, Anteil der Erosion in den staerksten 5% der
        Zellen (hoeher = staerker verzweigt statt flaechig):

            Partikel   ohne Kopplung   mit Kopplung
              20.000       0.243          0.278
              80.000       0.215          0.403
             240.000       0.142          0.455

        Ohne Kopplung traegt jeder zusaetzliche Tropfen zusaetzliches Material
        ab, die Karte wird flaechig abgeschliffen und die Kanaele verschwinden
        wieder. Mit Kopplung konvergiert das Ergebnis statistisch gegen ein
        immer feiner verzweigtes Netz - genau das, was viele schwache Partikel
        leisten sollen.

        Return: float - Anfangs-Wasservolumen eines Tropfens
        """
        density = max(float(num_droplets) / max(float(pixel_count), 1.0), 1e-9)
        return cls.DROPLET_INITIAL_WATER_VOLUME * (cls.DROPLET_REFERENCE_DENSITY / density)

    @classmethod
    def resolve_pixel_geometry(cls, meters_per_pixel):
        """
        Liefert die Partikelgeometrie für diese Auflösung:
        - Lebensweg aus DROPLET_MAX_LIFETIME_M (metrisch, damit ein Partikel
          bei jeder Auflösung dieselbe reale Strecke läuft)
        - Pinselradius als fester Pixelwert (Glättungskernel, siehe
          DROPLET_ERODE_RADIUS_PX)
        Return: (max_lifetime_steps: int, erode_radius_px: float)

        Gemeinsame Quelle für CPU- und GPU-Pfad
        (shader_manager._dispatch_droplet_erosion importiert sie), damit beide
        garantiert dieselbe Geometrie verwenden.
        """
        meters_per_pixel = max(float(meters_per_pixel), 1e-9)
        lifetime_steps = int(np.clip(round(cls.DROPLET_MAX_LIFETIME_M / meters_per_pixel),
                                     cls.DROPLET_MAX_LIFETIME_STEPS_MIN,
                                     cls.DROPLET_MAX_LIFETIME_STEPS_MAX))
        return lifetime_steps, cls.DROPLET_ERODE_RADIUS_PX

    def _cpu_simulate(self, work, hardness_map, erosion_map, sedimentation_map,
                       spawn_positions, meters_per_pixel, cap_per_step, initial_water=None):
        """EINEN Durchgang Partikel laufen lassen - mutiert work/erosion_map/
        sedimentation_map in place (der Aufrufer besitzt sie über alle
        Durchgänge hinweg, siehe simulate_erosion_sedimentation()).

        Sequentielle Python-Schleife über die Partikel - strukturell anders als
        jede andere CPU-Implementierung dieser Datei (ThermalErosionSystem etc.
        sind voll-vektorisierte Grid-Sweeps); ein Droplet-Weg ist inhärent
        sequentiell (jeder Schritt braucht den Zustand, den der VORHERIGE
        Schritt hinterlassen hat). Echter, dokumentierter
        Performance-Charakter, nicht versteckt."""
        hardness = hardness_map.astype(np.float64)
        max_lifetime_steps, erode_radius_px = self.resolve_pixel_geometry(meters_per_pixel)
        brush = self._build_erosion_brush(erode_radius_px)
        if initial_water is None:
            initial_water = self.initial_water_volume(len(spawn_positions), work.size)

        for pos_x, pos_y in spawn_positions:
            self._walk_one_droplet(work, hardness, erosion_map, sedimentation_map,
                                    float(pos_x), float(pos_y), cap_per_step, brush,
                                    max_lifetime_steps, initial_water)

    def _walk_one_droplet(self, work, hardness_map, erosion_map, sedimentation_map,
                           pos_x, pos_y, cap_per_step, brush, max_lifetime_steps,
                           initial_water):
        """EIN Partikel-Lebensweg (siehe Klassen-Docstring/Plan fuer die volle
        Formel-Herleitung, Schritt-fuer-Schritt gegen Erosion.cs verifiziert).
        Mutiert work/erosion_map/sedimentation_map in place.
        max_lifetime_steps kommt aus resolve_pixel_geometry() - die Weglaenge
        ist in Metern definiert, nicht in Pixeln (siehe dortige Konstanten)."""
        size_y, size_x = work.shape
        dir_x, dir_y = 0.0, 0.0
        speed = self.DROPLET_INITIAL_SPEED
        water = initial_water
        sediment = 0.0

        for _ in range(max_lifetime_steps):
            height, grad_x, grad_y = self._height_and_gradient(work, pos_x, pos_y)

            dir_x = dir_x * self.DROPLET_INERTIA - grad_x * (1.0 - self.DROPLET_INERTIA)
            dir_y = dir_y * self.DROPLET_INERTIA - grad_y * (1.0 - self.DROPLET_INERTIA)
            length = np.sqrt(dir_x * dir_x + dir_y * dir_y)
            if length < 1e-9:
                break  # Referenz-Verhalten: keine Zufalls-Neuausrichtung
            dir_x /= length
            dir_y /= length

            new_x = pos_x + dir_x
            new_y = pos_y + dir_y

            if new_x < 0 or new_x >= size_x - 1 or new_y < 0 or new_y >= size_y - 1:
                break  # Restfracht wird unten (Zwangs-Absetzung) an (pos_x,pos_y) abgesetzt

            new_height, _, _ = self._height_and_gradient(work, new_x, new_y)
            delta_height = new_height - height

            hardness_here = max(1.0, float(self._bilinear_value(hardness_map, pos_x, pos_y)))
            hardness_factor = np.clip(
                (self.DROPLET_ERODE_CAP_HARDNESS_REFERENCE / hardness_here)
                ** self.DROPLET_ERODE_CAP_HARDNESS_EXPONENT,
                self.DROPLET_ERODE_CAP_HARDNESS_MIN_FACTOR, self.DROPLET_ERODE_CAP_HARDNESS_MAX_FACTOR)

            # erosion_strength wirkt auf die KAPAZITAET - also darauf, wie viel
            # das Partikel ueberhaupt tragen WILL. Nicht als Nachmultiplikator
            # auf die abgetragene Menge, siehe die Klemme unten.
            sediment_capacity = max(
                -delta_height * speed * water * self.capacity_factor * self.erosion_strength,
                self.DROPLET_SEDIMENT_MIN_CAPACITY_M)

            if sediment > sediment_capacity or delta_height > 0:
                # Ablagern (bergauf ODER Kapazitaet ueberschritten)
                if delta_height > 0:
                    amount_to_deposit = min(delta_height, sediment)
                else:
                    amount_to_deposit = (sediment - sediment_capacity) * self.settling_velocity
                amount_to_deposit = min(amount_to_deposit, cap_per_step)
                sediment -= amount_to_deposit
                self._deposit_bilinear(work, sedimentation_map, pos_x, pos_y, amount_to_deposit)
            else:
                # Eintiefen. Die `-delta_height`-KLEMME ist die zentrale
                # Schutzzusage des Referenzalgorithmus (Erosion.cs): ein
                # Partikel traegt nie mehr ab als das Gefaelle zur naechsten
                # Zelle - es kann sich damit nie unter seinen eigenen Abfluss
                # graben.
                #
                # Bis 2026-07-27 wurde direkt NACH dieser Klemme noch mit
                # erosion_strength (Default 2.5) und hardness_factor (bis 3.0)
                # multipliziert, also mit bis zum 7.5-fachen. Genau das hob die
                # Zusage auf: das Partikel grub sich eine Grube, kletterte im
                # naechsten Schritt wieder heraus (delta_height > 0) und lud
                # seine Fracht am Grubenrand ab - der erhoehte Wall um jeden
                # Krater. Gemessen bei 128²: 66 -> 368 lokale Minima, bis 304 m
                # tiefe Loecher (Nutzer-Report (a) "viele einzelne Krater").
                #
                # Jetzt: erosion_strength wirkt auf die Kapazitaet (oben), die
                # Haerte als ERODIERBARKEIT in (0,1] auf das Ergebnis der
                # Klemme. Ein Faktor <= 1 kann die Klemme nicht aufheben - das
                # Ergebnis bleibt <= -delta_height, egal wie weich das Gestein
                # ist. Weiches Gestein widersteht nur nicht, es gewinnt nichts
                # ueber das geometrisch Moegliche hinaus.
                amount_to_erode = min(
                    (sediment_capacity - sediment) * self.DROPLET_ERODE_SPEED,
                    -delta_height) * hardness_factor
                amount_to_erode = min(amount_to_erode, cap_per_step)
                actually_eroded = self._erode_brush(work, erosion_map, pos_x, pos_y, amount_to_erode, brush)
                sediment += actually_eroded

            speed = np.sqrt(max(0.0, speed * speed + delta_height * self.DROPLET_GRAVITY))
            water *= (1.0 - self.DROPLET_EVAPORATE_SPEED)

            pos_x, pos_y = new_x, new_y

            if water < self.DROPLET_MIN_WATER or not np.isfinite(speed):
                break

        # Restfracht wird bei JEDEM Abbruchpfad VERWORFEN, identisch zur
        # Referenz (Erosion.cs) - siehe Klassen-Docstring "Massenerhaltung".
        # KEIN Zwangs-Absetzen mehr (frueher hier vorhanden, war die Ursache
        # sichtbarer kuenstlicher Huegel).


class ThermalErosionSystem:
    """
    Funktionsweise: Böschungswinkel-Erosion (Angle-of-Repose, "Phase 6",
    Nutzer-Vorgabe 2026-07-25) - unabhängiger, rein geometrischer Pass:
    Material bewegt sich lateral zu einem Nachbarn, sobald die Steigung
    dorthin den härte-abhängigen kritischen Böschungswinkel überschreitet.
    Läuft NACH der Fluss-Erosion (water.erosion_sedimentation, siehe
    water.thermal_erosion-Knoten in calculator_graph.py) auf der bereits
    fluvial eingeschnittenen Landschaft - Fluss-Erosion schneidet zuerst das
    schmale V-Kerbtal, Thermal Erosion kollabiert/verbreitert es danach je
    nach Härte: härteres Gestein -> höherer Böschungswinkel -> widersteht
    seitlichem Abrutschen besser -> steile V-Wände bleiben erhalten;
    weicheres/lockereres Material -> niedrigerer Winkel -> kollabiert zu
    Schutthalden/U-Form. EIN Mechanismus erfüllt damit sowohl "härtere
    Materialien bleiben besser zurück" als auch "V- UND U-Täler je nach
    Bedingung".

    4er-Nachbarschaft (konsistent mit dem Pipe-Modell, siehe
    PipeFlowSimulator) - Kartenrand ist eine GESCHLOSSENE Grenze (anders als
    beim Wasser: Material soll nicht von der Karte "abrutschen", Geisterzelle
    hat dieselbe Höhe wie die Randzelle selbst -> kein Gefälle, kein
    Transfer über den Rand). Jede Iteration ist eine vollständig
    materialisierte Gather-Operation (kein In-Place-Scatter während des
    Sweeps - dieselbe Lehre wie PipeFlowSimulator/
    _transport_sediment_maccormack) - dadurch ist jede transportierte
    Materialeinheit EXAKT einer Quelle und einem Ziel zugeordnet, die
    Massenbilanz ist strukturell exakt (kein Renormierungs-Schritt nötig,
    anders als beim Semi-Lagrange-Sedimenttransport).
    """

    # Böschungswinkel-Bereich (Grad) - fester interner Bereich, nur die
    # Stärke (TRANSFER_RATE-Multiplikator) ist nutzerseitig einstellbar
    # (siehe WATER.THERMAL_EROSION_STRENGTH). 15° in der Größenordnung von
    # losem Geröll/Sand, 60° in der Größenordnung von sehr hartem,
    # standfestem Gestein (grobe Anlehnung an reale Schüttwinkel-
    # Größenordnungen, kein exakter geotechnischer Wert).
    REPOSE_ANGLE_MIN_DEG = 15.0
    REPOSE_ANGLE_MAX_DEG = 60.0
    HARDNESS_REFERENCE_MIN = 1.0
    HARDNESS_REFERENCE_MAX = 100.0

    # HINWEIS ZUR MAP-DISTANCE-ABHÄNGIGKEIT (untersucht 2026-07-27):
    #
    # Der kritische Höhenunterschied zwischen zwei Nachbarzellen ist
    # `zellbreite * tan(boeschungswinkel)` - eine geometrisch exakte
    # Beziehung. Gemessen (64², 20 Iterationen, Härte 50) ergibt das
    # 652 717 m Gesamtabtrag bei 1 km Kartenausdehnung, 12 026 m bei 10 km
    # und exakt 0.00 m bei 100 km. Das SIEHT nach einem Skalierungsfehler
    # aus, ist aber korrektes Verhalten: 4000 m Relief auf 100 km Breite
    # sind eine sehr sanfte Landschaft (mittleres Gefälle ~4%), in der kein
    # Hang die 15°-Schwelle des weichsten Materials erreicht - es gibt dort
    # schlicht nichts abzurutschen.
    #
    # Ein Versuch, den Winkel stattdessen gegen eine FESTE Bezugs-Zellgröße
    # auszuwerten, wurde verworfen: er macht die Beziehung physikalisch
    # falsch und hebt die Härte-Differenzierung auf - mit ihr fiel im
    # V-Kerbtal-Test (smoke_test_water_thermal_erosion.py) sowohl weiches
    # als auch hartes Gestein unter die Schwelle, aus "V- oder U-Tal je nach
    # Härte" wurde "nie ein Kollaps". Die Geometrie bleibt deshalb exakt.
    #
    # Wer bei sehr grosser Kartenausdehnung sichtbare Böschungseffekte will,
    # muss die Reliefenergie erhöhen (TERRAIN.AMPLITUDE) - Map Distance und
    # Amplitude sind zwei unabhängige Regler, und ihr Verhältnis ist genau
    # das, was hier zählt.

    # Anteil des Steigungs-Überschusses (über dem Böschungswinkel), der PRO
    # ITERATION lateral bewegt wird - < 1, damit ein einzelner Transfer nie
    # über das Gleichgewicht mit EINEM Nachbarn hinausschiesst.
    TRANSFER_RATE = 0.5

    # Relief-relativer Deckel PRO ITERATION - verhindert, dass eine Zelle mit
    # mehreren gleichzeitig "hungrigen" Nachbarn in einem Schritt mehr abgibt,
    # als plausibel ist.
    #
    # Von 1% auf 0.1% des Reliefs gesenkt (2026-07-27, Nutzer-Report
    # "thermal erosion zerfrisst die hügel sehr stark"). Der Deckel ist hier
    # nicht nur Sicherheitsnetz, sondern die tatsaechlich bindende Groesse:
    # bei 78 m/px und einem Boeschungswinkel von 15° liegt die Schwelle bei
    # 21 m Hoehenunterschied, der Ueberschuss auf steilen Haengen aber
    # regelmaessig bei 10-40 m - der Deckel greift also in fast jeder
    # Iteration. Gemessen bei 128², Relief 947 m, 40 Iterationen:
    #
    #     Deckel      Gesamtabtrag   max. pro Zelle
    #     1%   (9.5m)      48 679 m        287.9 m   <- 30% des Reliefs!
    #     0.5% (4.7m)      44 496 m        189.4 m
    #     0.2% (1.9m)      31 676 m         75.8 m
    #     0.1% (0.9m)      22 334 m         37.9 m
    #
    # Weil der Deckel bindet, gilt naeherungsweise
    # Gesamtverlagerung ~ Deckel x Iterationszahl - deshalb ist die
    # Iterationszahl bewusst NICHT mehr an die Kantenlaenge gekoppelt
    # (siehe HydrologySystemGenerator.THERMAL_ITERATIONS), sonst haette
    # dieselbe Karte bei feinerer Aufloesung ein Vielfaches an Abtrag.
    CAP_RELIEF_FRACTION = 0.001
    CAP_MIN_M = 0.02

    def __init__(self, thermal_strength=1.0, shader_manager=None):
        self.thermal_strength = thermal_strength
        self.shader_manager = shader_manager

    def _repose_angle_deg(self, hardness_map):
        t = np.clip(
            (hardness_map - self.HARDNESS_REFERENCE_MIN) /
            (self.HARDNESS_REFERENCE_MAX - self.HARDNESS_REFERENCE_MIN), 0.0, 1.0)
        return self.REPOSE_ANGLE_MIN_DEG + t * (self.REPOSE_ANGLE_MAX_DEG - self.REPOSE_ANGLE_MIN_DEG)

    def simulate(self, heightmap, hardness_map, parameters, iterations, meters_per_pixel):
        """GPU-Pfad mit CPU-Fallback, siehe Klassen-
        Docstring. Rückgabe: (thermal_erosion_map, thermal_deposition_map),
        beide (H,W) nicht-negativ, in Metern."""
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "thermalErosion",
                    {
                        "heightmap": heightmap, "hardness_map": hardness_map,
                        "meters_per_pixel": meters_per_pixel, "iterations": iterations,
                        "transfer_rate": self.TRANSFER_RATE, "thermal_strength": self.thermal_strength,
                        "repose_angle_min_deg": self.REPOSE_ANGLE_MIN_DEG,
                        "repose_angle_max_deg": self.REPOSE_ANGLE_MAX_DEG,
                        "hardness_reference_min": self.HARDNESS_REFERENCE_MIN,
                        "hardness_reference_max": self.HARDNESS_REFERENCE_MAX,
                        "cap_relief_fraction": self.CAP_RELIEF_FRACTION, "cap_min_m": self.CAP_MIN_M,
                    },
                    parameters
                )
                if result.get("success"):
                    return result["thermal_erosion_map"], result["thermal_deposition_map"]
            except Exception as e:
                logging.warning(f"GPU thermal erosion failed: {e}, falling back to CPU")

        # CPU-Pfad. Kein Simple-Fallback mehr (2026-07-27) - gleiche
        # Begruendung wie bei DropletErosionSystem.
        return self._cpu_simulate(heightmap, hardness_map, iterations, meters_per_pixel)

    def _cpu_simulate(self, heightmap, hardness_map, iterations, meters_per_pixel):
        height_work = heightmap.astype(np.float64).copy()
        tan_angle = np.tan(np.radians(self._repose_angle_deg(hardness_map.astype(np.float64))))
        relief = float(heightmap.max() - heightmap.min())
        cap_per_step = max(self.CAP_MIN_M, self.CAP_RELIEF_FRACTION * relief)

        erosion_map = np.zeros_like(height_work)
        deposition_map = np.zeros_like(height_work)

        for _ in range(max(1, iterations)):
            # Geschlossene Grenze: Geisterzelle = eigene Höhe/eigener Winkel
            # (kein Gefälle über den Rand, siehe Klassen-Docstring).
            h_pad = np.pad(height_work, 1, mode='edge')
            tan_pad = np.pad(tan_angle, 1, mode='edge')
            h = h_pad[1:-1, 1:-1]
            tan_here = tan_pad[1:-1, 1:-1]

            h_l, tan_l = h_pad[1:-1, 0:-2], 0.5 * (tan_here + tan_pad[1:-1, 0:-2])
            h_r, tan_r = h_pad[1:-1, 2:], 0.5 * (tan_here + tan_pad[1:-1, 2:])
            h_t, tan_t = h_pad[0:-2, 1:-1], 0.5 * (tan_here + tan_pad[0:-2, 1:-1])
            h_b, tan_b = h_pad[2:, 1:-1], 0.5 * (tan_here + tan_pad[2:, 1:-1])

            diff_l = np.maximum(0.0, (h - h_l) - meters_per_pixel * tan_l)
            diff_r = np.maximum(0.0, (h - h_r) - meters_per_pixel * tan_r)
            diff_t = np.maximum(0.0, (h - h_t) - meters_per_pixel * tan_t)
            diff_b = np.maximum(0.0, (h - h_b) - meters_per_pixel * tan_b)

            scale = self.TRANSFER_RATE * self.thermal_strength
            raw_l, raw_r, raw_t, raw_b = diff_l * scale, diff_r * scale, diff_t * scale, diff_b * scale

            total_raw = raw_l + raw_r + raw_t + raw_b
            k = np.minimum(1.0, cap_per_step / np.maximum(total_raw, 1e-12))
            out_l, out_r, out_t, out_b = raw_l * k, raw_r * k, raw_t * k, raw_b * k

            # Zufluss von Nachbarn = deren jeweils ENTGEGENGESETZTER Ausfluss;
            # Geisterzellen am (geschlossenen) Rand liefern 0 Zufluss.
            out_l_pad = np.pad(out_l, 1, mode='constant', constant_values=0.0)
            out_r_pad = np.pad(out_r, 1, mode='constant', constant_values=0.0)
            out_t_pad = np.pad(out_t, 1, mode='constant', constant_values=0.0)
            out_b_pad = np.pad(out_b, 1, mode='constant', constant_values=0.0)

            incoming = (out_r_pad[1:-1, 0:-2] + out_l_pad[1:-1, 2:] +
                        out_b_pad[0:-2, 1:-1] + out_t_pad[2:, 1:-1])
            outgoing = out_l + out_r + out_t + out_b

            height_work = height_work + incoming - outgoing
            erosion_map += outgoing
            deposition_map += incoming

        return erosion_map.astype(np.float32), deposition_map.astype(np.float32)


class SoilMoistureCalculator:
    """
    Funktionsweise: Berechnet Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
    Aufgabe: Erstellt soil_moist_map für Biome-System und Weather-Evaporation
    """

    def __init__(self, diffusion_radius=2.0, shader_manager=None, capillary_sigma=0.5):
        self.diffusion_radius = diffusion_radius
        self.shader_manager = shader_manager
        # Kapillare Ausbreitung: fester, KLEINER Radius (nicht die einstellbare
        # diffusion_radius, die nur die "Grundwasser"-Komponente steuert) - war
        # vorher hart auf 2.0 kodiert. Bei einem dichteren Fluss-Netzwerk
        # (Creek/River-Klassifikation deckt nach der precip_map-Neukalibrierung
        # empirisch ~40% der Karte ab statt vorher deutlich weniger) überlappte
        # sigma=2.0 zwischen benachbarten Wasser-Quellen so stark, dass praktisch
        # die GESAMTE Karte auf >90% Feuchte kam (empirisch: mean=63.8%,
        # nur 1.7% der Pixel unter 10% - "Soil Moisture überall 100%"-Report,
        # siehe [[project-water-flood-calibration]]). 0.5 verifiziert gegen
        # denselben Test: mean sinkt auf ~52%, 4-7% der Pixel bleiben unter 10%.
        self.capillary_sigma = capillary_sigma

    def calculate_soil_moisture(self, water_biomes_map, flow_accumulation, parameters, water_mask_source=None):
        """
        Funktionsweise: GPU-Pfad mit CPU-Fallback (Multi-Radius-Gauss).
        Aufgabe: Feuchtigkeits-Verteilung rund um Gewässer.

        Der frühere dritte "Simple"-Fallback (_simple_soil_moisture_calculation)
        ist entfernt (2026-07-27): er fing jede Exception des CPU-Pfads ab und
        lieferte eine grob gerasterte Ersatzkarte, wodurch ein echter
        Programmfehler wie ein gültiges Ergebnis aussah. Der CPU-Pfad ist reine
        numpy/scipy-Arithmetik ohne erwartbare Fehlerquelle - ein Fehler dort
        ist ein Bug und propagiert jetzt. Der GPU->CPU-Fallback bleibt, weil
        "keine GPU verfügbar" eine erwartete Umgebungsbedingung ist.

        water_mask_source (optional): separate Karte, die statt water_biomes_map
        für die 100%-Feuchte-QUELLFLÄCHE genutzt wird (die ungemalte
        Fluss-Zentrallinie, siehe _calc_soil_moisture in
        HydrologySystemGenerator) - entkoppelt die räumliche Ausdehnung der
        Boden-Feuchte von der visuellen Flussbreite. Default None =
        water_biomes_map selbst als Quelle nutzen.
        water_biomes_map bleibt für alles andere (Klassifikations-Stufe pro
        Pixel) unverändert die gemalte Karte.
        """
        if water_mask_source is None:
            water_mask_source = water_biomes_map

        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "soilMoistureGaussian",
                    {
                        "water_biomes_map": water_biomes_map,
                        "water_mask_source": water_mask_source,
                        "flow_accumulation": flow_accumulation,
                        "diffusion_radius": self.diffusion_radius,
                        "capillary_sigma": self.capillary_sigma
                    },
                    parameters
                )
                if result.get("success"):
                    return result["soil_moisture"]
            except Exception as e:
                logging.warning(f"GPU soil moisture calculation failed: {e}, falling back to CPU")

        return self._cpu_soil_moisture_calculation(water_biomes_map, flow_accumulation, water_mask_source)

    # Grundwasser-Beitrag pro Wasser-Klassifikationsstufe:
    # water_type -> (Basis-Feuchte in %, Zuschlag pro Durchfluss-Einheit,
    #                Obergrenze des Zuschlags in %).
    # Seen (4) haben keinen durchfluss-abhängigen Anteil - ein stehendes
    # Gewässer speist das Grundwasser unabhängig davon, wie viel durchfliesst.
    _GROUNDWATER_BY_WATER_TYPE = {
        4: (80.0, 0.0, 0.0),
        3: (60.0, 0.1, 20.0),
        2: (40.0, 0.2, 20.0),
        1: (20.0, 0.3, 10.0),
    }

    def _cpu_soil_moisture_calculation(self, water_biomes_map, flow_accumulation, water_mask_source=None):
        """
        Gaussian-Diffusion mit zwei Radien (kapillar eng, Grundwasser weit).
        Die Grundwasser-Quellstärke pro Pixel ist seit 2026-07-27 vektorisiert
        (vorher eine Python-Doppelschleife über die ganze Karte) - dieselbe
        Tabelle, nur als Masken-Zuweisung statt if/elif-Kaskade, siehe
        _GROUNDWATER_BY_WATER_TYPE.
        """
        if water_mask_source is None:
            water_mask_source = water_biomes_map

        normalized_flow = self._normalized_discharge(flow_accumulation)

        # Direkte Wasserpräsenz: maximale Feuchtigkeit - nutzt water_mask_source
        # (Zentrallinie), nicht die ggf. breiter gemalte water_biomes_map.
        water_mask = water_mask_source > 0

        # Kapillare Ausbreitung (enger Filter)
        capillary_source = np.where(water_mask, 100.0, 0.0).astype(np.float32)
        capillary_moisture = gaussian_filter(capillary_source, sigma=self.capillary_sigma)

        # Grundwasser-Effekte (weiter Filter) - Klassifikations-Stufe (Creek/
        # River/Grand River/Lake) an der Zentrallinie ist identisch zur
        # gemalten Fläche (Breiten-Malen ändert nur die räumliche Ausdehnung,
        # nicht die Stufe selbst), daher weiterhin water_mask_source nutzen.
        groundwater_source = np.zeros(water_mask_source.shape, dtype=np.float32)
        for water_type, (base, per_flow, cap) in self._GROUNDWATER_BY_WATER_TYPE.items():
            type_mask = water_mask_source == water_type
            if not np.any(type_mask):
                continue
            groundwater_source[type_mask] = base + np.minimum(
                cap, normalized_flow[type_mask] * per_flow)

        groundwater_moisture = gaussian_filter(groundwater_source, sigma=self.diffusion_radius)

        # Kombiniere beide Effekte (Maximum)
        combined_moisture = np.maximum(capillary_moisture, groundwater_moisture)
        combined_moisture[water_mask] = 100.0

        return combined_moisture

    @staticmethod
    def _normalized_discharge(flow_accumulation):
        """
        Skaliert den Durchfluss auf ein maßstabsunabhängiges 0..100-Band
        (Perzentil-basiert, 99. Perzentil der wasserführenden Zellen = 100).

        Nötig, weil `flow_accumulation` seit dem Pipe-Modell-Umbau ein echter
        Volumenstrom ist, dessen Größenordnung direkt von Zellfläche und
        Zeitschritt abhängt: gemessen schwankt der Maximalwert allein zwischen
        map_distance_km 1 und 10 um Faktor 35. Die festen Zuschlags-Faktoren
        oben (0.1/0.2/0.3 pro Einheit, gedeckelt bei 10-20%) waren gegen die
        alte, dimensionslose Akkumulationsmenge kalibriert und liefen unter dem
        Pipe-Modell IMMER in ihren Deckel - der Durchfluss hatte damit faktisch
        keinen Einfluss mehr auf die Bodenfeuchte. Die Normierung stellt genau
        diesen Einfluss wieder her, ohne die Faktoren an eine Kartengröße zu
        binden.
        """
        flow = np.asarray(flow_accumulation, dtype=np.float32)
        wet = flow[flow > 0]
        if wet.size == 0:
            return np.zeros_like(flow)
        reference = float(np.percentile(wet, 99.0))
        if reference <= 0:
            return np.zeros_like(flow)
        return np.clip(flow / reference, 0.0, 1.0) * 100.0


class EvaporationCalculator:
    """
    Funktionsweise: Berechnet Verdunstung aus temp_map, wind_map und humid_map
    (Magnus-Formel).

    Zwei Abnehmer derselben Formel:
    1. calculate_potential_evaporation() - die reine, NICHT durch eine
       Wasserfläche begrenzte Rate. Sie geht als zweite Senke (neben dem
       Randabfluss) direkt in die Pipe-Simulation ein, siehe
       PipeFlowSimulator.EVAPORATION_TO_DEPTH_RATE. Weil sie nur von
       Wetterdaten abhängt und nicht von der Wasser-Klassifikation, entsteht
       dabei kein Zyklus im Calculator-Graph.
    2. calculate_evaporation() - dieselbe Rate, zusätzlich durch den
       Gewässertyp gedeckelt (ein Bach kann nicht beliebig viel abgeben).
       Das ist der angezeigte water.evaporation-Output.

    Bis 2026-07-27 gab es nur Variante 2, und ihr Ergebnis wurde von
    niemandem gelesen: die evaporation_map war ein toter Output, und das
    Pipe-Modell hatte ausser dem Kartenrand keine Senke - Wasser konnte in
    abflusslosen Becken nur steigen. evaporation_base_rate war damit ein
    Slider ohne jede Wirkung auf das Kartenbild.
    """

    def __init__(self, evaporation_base_rate=0.002, shader_manager=None):
        self.base_rate = evaporation_base_rate
        self.shader_manager = shader_manager

    def calculate_potential_evaporation(self, temp_map, wind_map, humid_map):
        """
        Potentielle Verdunstung in gH2O/m²/Tag - ohne Begrenzung durch eine
        Wasserfläche, also der Wert, den eine unbegrenzt verfügbare
        Wasseroberfläche an diesem Ort abgeben würde.
        Return: (H,W) float32, nicht-negativ.
        """
        temperature = temp_map.astype(np.float64, copy=False)
        humidity = humid_map.astype(np.float64, copy=False)
        wind_speed = np.hypot(wind_map[:, :, 0].astype(np.float64),
                              wind_map[:, :, 1].astype(np.float64))

        # Magnus-Formel für die maximale Wasserdampfdichte. Der Ausdruck ist
        # strikt positiv (Exponentialfunktion), eine Division durch 0 ist
        # ausgeschlossen.
        max_vapor_density = 5.0 * np.exp(0.06 * temperature)
        relative_humidity = np.minimum(1.0, humidity / max_vapor_density)

        humidity_factor = 1.0 - relative_humidity
        temp_factor = np.where(temperature > 0, np.exp(temperature / 20.0), 0.1)
        wind_factor = 1.0 + wind_speed * 0.2

        potential = self.base_rate * humidity_factor * temp_factor * wind_factor * 1000.0
        return np.maximum(0.0, potential).astype(np.float32)

    def calculate_evaporation(self, temp_map, wind_map, humid_map, water_biomes_map, parameters):
        """
        Funktionsweise: GPU-accelerated Evaporation mit Magnus-Formel
        Aufgabe: GPU-Pfad mit CPU-Fallback für realistische Verdunstung
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "atmosphericEvaporation",
                    {
                        "temp_map": temp_map,
                        "wind_map": wind_map,
                        "humid_map": humid_map,
                        "water_biomes_map": water_biomes_map,
                        "base_rate": self.base_rate
                    },
                    parameters
                )
                if result.get("success"):
                    return result["evaporation_map"]
            except Exception as e:
                logging.warning(f"GPU evaporation calculation failed: {e}, falling back to CPU")

        # CPU-Pfad. Kein dritter "Simple"-Fallback mehr (2026-07-27, gleiche
        # Begründung wie bei SoilMoistureCalculator.calculate_soil_moisture):
        # er lieferte eine feste Pauschalrate und liess damit einen echten
        # Programmfehler wie ein gültiges Ergebnis aussehen.
        return self._cpu_evaporation_calculation(temp_map, wind_map, humid_map, water_biomes_map)

    # Obergrenze der Verdunstung je Wasser-Klassifikationsstufe
    # (gH2O/m²/Tag) - ein Bach kann nicht beliebig viel Wasser abgeben, ein
    # See dagegen praktisch schon (np.inf = keine Begrenzung).
    _EVAPORATION_LIMIT_BY_WATER_TYPE = np.array([
        0.0,      # 0 = kein Wasser
        50.0,     # 1 = Creek
        100.0,    # 2 = River
        200.0,    # 3 = Grand River
        np.inf,   # 4 = Lake
    ], dtype=np.float64)

    def _cpu_evaporation_calculation(self, temp_map, wind_map, humid_map, water_biomes_map):
        """
        Verdunstung über die Magnus-Formel (calculate_potential_evaporation),
        zusätzlich durch den Gewässertyp gedeckelt. Vollständig vektorisiert
        (2026-07-27 - vorher zwei aufeinanderfolgende Python-Doppelschleifen
        über die ganze Karte, zusammen 0.29 s bei 512², bei jedem LOD erneut).
        Formeln unverändert.
        Return: (H,W) float32, Verdunstung in gH2O/m²/Tag.
        """
        potential = self.calculate_potential_evaporation(temp_map, wind_map, humid_map)
        return self._limit_by_available_water(potential, water_biomes_map)

    def _limit_by_available_water(self, evaporation_map, water_biomes_map):
        """
        Begrenzt die Verdunstung durch die verfügbare Wasseroberfläche und
        setzt sie auf trockenen Zellen auf 0. Vektorisiert über eine
        Lookup-Tabelle (_EVAPORATION_LIMIT_BY_WATER_TYPE).
        """
        water_type = np.clip(np.asarray(water_biomes_map, dtype=np.intp),
                             0, len(self._EVAPORATION_LIMIT_BY_WATER_TYPE) - 1)
        limits = self._EVAPORATION_LIMIT_BY_WATER_TYPE[water_type]
        return np.minimum(evaporation_map, limits).astype(np.float32)


class HydrologySystemGenerator:
    """
    Funktionsweise: Hauptklasse für dynamisches Hydrologiesystem
    Aufgabe: Koordiniert alle hydrologischen Prozesse mit LOD-System und Multi-Dependency-Resolution
    """

    def __init__(self, map_seed=42, shader_manager=None, data_lod_manager=None):
        self.map_seed = map_seed
        self.logger = logging.getLogger(self.__class__.__name__)
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager

        # Sub-System Initialisierung
        self.lake_detection = LakeDetectionSystem(shader_manager=shader_manager)
        self.flow_network = FlowNetworkBuilder(shader_manager=shader_manager)
        self.manning_calculator = ManningFlowCalculator(shader_manager=shader_manager)
        self.erosion_system = DropletErosionSystem(shader_manager=shader_manager)
        self.soil_moisture = SoilMoistureCalculator(shader_manager=shader_manager)
        self.evaporation = EvaporationCalculator(shader_manager=shader_manager)
        self.thermal_erosion = ThermalErosionSystem(shader_manager=shader_manager)

        # Parameter der aktuell laufenden Generierungs-Anfrage - vom
        # GenerationOrchestrator einmal pro frischer Anfrage über
        # set_active_parameters() gesetzt, bleibt über alle LOD-Runden dieser
        # Anfrage hinweg konstant.
        self._current_parameters: Dict[str, Any] = {}

        # Fortschritts-Callback (phase, progress, message) - wird von
        # CalculatorThread für die Dauer eines Knotens gesetzt und speist das
        # generation_progress-Signal des Orchestrators. Muss als Attribut
        # existieren, damit CalculatorThread es überhaupt findet (hasattr-Test).
        self.progress_callback = None

    def set_active_parameters(self, parameters: Dict[str, Any]):
        """Setzt die Parameter, die alle _calc_*-Methoden bis zur nächsten frischen
        Anfrage verwenden (vom GenerationOrchestrator aufgerufen). Water speichert
        einen Teil der Parameter zusätzlich als Instanz-Attribute auf den Sub-
        Kalkulator-Objekten (self.erosion_system.capacity_factor,
        self.soil_moisture.diffusion_radius
        etc.) - _update_parameters() überträgt das (identisches Muster wie
        core/biome_generator.py set_active_parameters()/_update_parameters()).
        Fehlte hier bisher: alle _calc_*-Calculator-Knoten rufen Methoden auf
        diesen Sub-Objekten auf, die für die meisten Werte self.X liest statt aus
        dem live durchgereichten parameters-Dict - ohne diesen Aufruf blieb jedes
        Sub-Objekt für immer bei seinem Konstruktor-Default hängen, unabhängig
        vom UI-Slider (Live-App-Report: "buchstäblich jeder Water-Parameter hat
        keinen Effekt", siehe [[project-water-parameter-sync-bug]]).
        """
        self._current_parameters = parameters
        self._update_parameters(parameters)

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, calculate_hydrology() ohne
        injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from gui.OldManagers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    # _load_default_parameters() gelöscht (2026-07-27): las
    # WATER.EROSION_ITERATIONS_PER_LOD und WATER.WATER_SEED - beide existieren
    # in gui/config/value_default.py nicht (mehr), der Aufruf endete also
    # zwangsläufig in einem AttributeError, den das umgebende
    # `except ImportError` nicht fing. Die Methode hatte keinen einzigen
    # Aufrufer; Defaults kommen ausschließlich über
    # WaterTab.get_current_parameters() -> set_active_parameters(), mit den
    # Fallback-Literalen in _update_parameters() als letztem Sicherheitsnetz.

    # _get_dependencies() geloescht (2026-07-27): kein Aufrufer. Die Pipeline
    # laeuft ueber den CalculatorDispatcher, in dem jede _calc_*-Methode ihre
    # Inputs ueber _get_prepared_water_inputs() selbst aus dem
    # Calculator-Storage holt - und zwar genau die, die ihr Knoten laut
    # CALCULATOR_GRAPH wirklich braucht. Die geloeschte Methode verlangte
    # darueber hinaus slopemap und rock_map, die Water beide nie benutzt hat.

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Führt Water-Generierung mit LOD-optimierten Algorithmen aus.
        Erosion und Böschungswinkel rechnen nur in der letzten LOD-Runde
        (siehe _is_final_lod()); die früheren Runden schreiben Nullkarten. Eine
        LOD-übergreifende Kumulation gibt es deshalb nicht mehr.
        """
        self.logger.info(f"Starting water generation for LOD {lod}")

        # Genau die 6 Karten, die Water tatsächlich liest. slopemap und
        # rock_map sind 2026-07-27 aus der Schnittstelle entfernt - sie wurden
        # verlangt und durchgereicht, aber von keiner Water-Berechnung je
        # benutzt (zusätzliche `dependencies`-Einträge werden hier still
        # ignoriert, bestehende Aufrufer brechen dadurch nicht).
        heightmap = dependencies['heightmap']
        hardness_map = dependencies['hardness_map']
        precip_map = dependencies['precip_map']
        temp_map = dependencies['temp_map']
        wind_map = dependencies['wind_map']
        humid_map = dependencies['humid_map']

        self._update_parameters(parameters)
        self._ensure_data_lod_manager()
        self.set_active_parameters(parameters)

        target_size = self._get_lod_size(lod, heightmap.shape[0])

        try:
            # Standalone-Convenience-Pfad (Legacy-Kompatibilität + Tests): dependencies
            # kommen hier als direktes dict, nicht aus dem DataLODManager - für die
            # _calc_*-Methoden (die jetzt IMMER aus dem Storage lesen) gespiegelt,
            # analog zu Geology/Weather. Erwartet lod als int (die einzig noch aktiv
            # genutzte LOD-Form - der alte String-LOD-Pfad ist nur noch in der
            # separat markierten Legacy-Methode generate_hydrology_system() relevant,
            # die keinen calculator-graph-basierten Storage mehr nutzt).
            self.data_lod_manager.set_calculator_output("terrain.redistribution", lod, {"heightmap": heightmap})
            self.data_lod_manager.set_calculator_output(
                "geology.hardness", lod, {"hardness_map": hardness_map})
            self.data_lod_manager.set_calculator_output(
                "weather.precipitation", lod, {"precip_map": precip_map})
            self.data_lod_manager.set_calculator_output(
                "weather.temperature", lod, {"temp_map": temp_map})
            self.data_lod_manager.set_calculator_output("weather.wind", lod, {"wind_map": wind_map})
            self.data_lod_manager.set_calculator_output(
                "weather.humidity", lod, {"humid_map": humid_map})

            # Läuft über die einzeln aufrufbaren _calc_*-Methoden (siehe
            # gui/OldManagers/calculator_graph.py - Water-Calculator-Knoten #15-#21
            # aus docs/generation_pipeline_dependencies.md, #22 erosion_feedback
            # bewusst ausgeschlossen - bekannt kaputt). Die echte GUI-Pipeline
            # (GenerationOrchestrator) ruft dieselben Methoden ab jetzt einzeln über
            # den globalen CalculatorDispatcher auf (Tracker #16 LOD-Lockstep-Umbau)
            # - der Effekt ist identisch, da beide Wege denselben Storage nutzen.
            # water.steepest_descent (D8-Fließrichtung) entfaellt seit dem
            # Pipe-Modell-Umbau 2026-07-25 komplett - siehe PipeFlowSimulator.
            #
            # Reihenfolge identisch zu der, die CALCULATOR_GRAPH seit
            # 2026-07-27 auch im parallelen Dispatcher erzwingt (siehe dortigen
            # Reihenfolge-Block): erst das Gelaende formen (Droplet-Erosion +
            # Boeschungswinkel, beide UNABHAENGIG vom Weather-Niederschlag),
            # dann den Wasserkreislauf mit dem ECHTEN Regen auf genau diesem
            # veraenderten Gelaende simulieren. Standalone- und GUI-Pfad
            # liefern dadurch dieselbe Reihenfolge und damit dasselbe Ergebnis.
            # === ALTBESTAND (stillgelegt 2026-07-28) ===
            # "water.erosion_sedimentation" und "water.thermal_erosion" sind
            # hier entfallen: das Gelaende formt seit 2026-07-28 der eigene
            # Erosion-Generator (core/erosion_generator.py), der VOR Weather
            # laeuft. Water simuliert nur noch den Wasserkreislauf auf dem
            # bereits fertigen Gelaende.
            for calculator_id in (
                "water.lake_detection", "water.flow_network",
                "water.manning_flow", "water.soil_moisture", "water.evaporation",
            ):
                getattr(self, "_calc_" + calculator_id.split(".", 1)[1])(calculator_id, lod)

            water_data = self.assemble_water_data(lod, parameters)

            self.logger.debug(f"Water generation complete - LOD: {lod}, size: {water_data.actual_size}")
            return water_data

        except Exception as e:
            # KEIN Ersatzergebnis mehr (2026-07-28). Der Zweig lieferte bei
            # jedem Fehler Nullkarten mit validity_state="fallback" - eine
            # gescheiterte Generierung sah damit aus wie eine gelungene auf
            # einer wasserlosen Karte. Genau das ist gerade passiert: ein
            # NameError im Zusammenbau (Rest des Erosions-Umbaus) wurde zu
            # "Fluss-Anteil 0.000, mittlere Tiefe 0.0000 m" und waere ohne den
            # Skalierungs-Regressionstest nicht aufgefallen.
            #
            # Es ist derselbe Aufraeum-Schritt, der 2026-07-27 fuer alle
            # uebrigen `except Exception -> Ersatzergebnis`-Faelle dieser Datei
            # gemacht wurde - dieser eine war uebersehen worden.
            self.logger.error(f"Water generation failed: {e}")
            raise

    def assemble_water_data(self, lod_level: int, parameters: Dict[str, Any]) -> WaterData:
        """
        Funktionsweise: Baut das finale WaterData-Objekt aus den einzeln
        gespeicherten Calculator-Outputs zusammen (inkl. der reinen
        Output-Formatierungs-Schritte water_map/ocean_outflow, die keine eigenen
        Calculator-Knoten sind)
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald alle 7 Water-
            Calculator-Knoten ein LOD abgeschlossen haben (siehe Task 18 im
            LOD-Lockstep-Umbau)
        """
        flow_accumulation = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "flow_accumulation", lod_level)
        # GEMALTE Klassifikation (water.manning_flow), nicht die Zentrallinie -
        # WaterData.water_biomes_map ist die Fassung, die Anzeige, Biome und
        # Settlement sehen sollen (siehe _calc_manning_flow()).
        water_biomes_map = self.data_lod_manager.get_calculator_output(
            "water.manning_flow", "water_biomes_map", lod_level)
        # water_depth kommt seit dem Pipe-Modell-Umbau direkt aus
        # water.flow_network (echte simulierte Tiefe, nicht mehr Mannings
        # Kanalgeometrie-Schaetzung) - siehe _calc_flow_network().
        water_depth = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "water_depth", lod_level)
        ocean_outflow = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "ocean_outflow", lod_level)
        # flow_speed kommt seit Stufe 3 (Manning-Schrumpfung) direkt aus dem
        # Pipe-Modell-Geschwindigkeitsfeld statt aus water.manning_flow (das
        # hat seit dem Umbau keine eigene flow_speed-Schaetzung mehr, siehe
        # _calc_manning_flow()).
        velocity_x = self.data_lod_manager.get_calculator_output("water.flow_network", "velocity_x", lod_level)
        velocity_y = self.data_lod_manager.get_calculator_output("water.flow_network", "velocity_y", lod_level)
        flow_speed = None
        if velocity_x is not None and velocity_y is not None:
            flow_speed = np.sqrt(velocity_x.astype(np.float64) ** 2 + velocity_y.astype(np.float64) ** 2)
        cross_section = self.data_lod_manager.get_calculator_output(
            "water.manning_flow", "cross_section", lod_level)
        # === ALTBESTAND (stillgelegt 2026-07-28) ===
        # Die vier gelaendeformenden Karten gehoeren jetzt dem Erosion-
        # Generator (core/erosion_generator.py). WaterData fuehrt sie nicht
        # mehr; Anzeige und get_terrain_data_combined() lesen sie aus dem
        # Erosion-Storage.
        # erosion_map = self.data_lod_manager.get_calculator_output(
        #     "water.erosion_sedimentation", "erosion_map", lod_level)
        # sedimentation_map = self.data_lod_manager.get_calculator_output(
        #     "water.erosion_sedimentation", "sedimentation_map", lod_level)
        # thermal_erosion_map = self.data_lod_manager.get_calculator_output(
        #     "water.thermal_erosion", "thermal_erosion_map", lod_level)
        # thermal_deposition_map = self.data_lod_manager.get_calculator_output(
        #     "water.thermal_erosion", "thermal_deposition_map", lod_level)
        soil_moist_map = self.data_lod_manager.get_calculator_output(
            "water.soil_moisture", "soil_moist_map", lod_level)
        evaporation_map = self.data_lod_manager.get_calculator_output(
            "water.evaporation", "evaporation_map", lod_level)

        missing = [name for name, value in (
            ("flow_accumulation", flow_accumulation), ("water_biomes_map", water_biomes_map),
            ("water_depth", water_depth), ("ocean_outflow", ocean_outflow),
            ("flow_speed", flow_speed), ("cross_section", cross_section),
            ("soil_moist_map", soil_moist_map),
            ("evaporation_map", evaporation_map),
        ) if value is None]
        if missing:
            raise ValueError(f"assemble_water_data: fehlende Calculator-Outputs für LOD {lod_level}: "
                              f"{', '.join(missing)}")

        target_size = flow_accumulation.shape[0]

        self._update_progress("Finalization", 98, "Creating water depth map...")
        water_map = self._create_water_depth_map(water_biomes_map, water_depth)

        water_data = WaterData()
        water_data.water_map = water_map
        water_data.flow_map = flow_accumulation
        water_data.flow_speed = flow_speed
        water_data.cross_section = cross_section
        water_data.soil_moist_map = soil_moist_map
        # === ALTBESTAND (stillgelegt 2026-07-28) ===
        # Die vier gelaendeformenden Karten fuellt jetzt der Erosion-Generator
        # (core/erosion_generator.py); WaterData fuehrt sie nicht mehr.
        # water_data.erosion_map = erosion_map
        # water_data.sedimentation_map = sedimentation_map
        # water_data.thermal_erosion_map = thermal_erosion_map
        # water_data.thermal_deposition_map = thermal_deposition_map
        water_data.evaporation_map = evaporation_map
        water_data.ocean_outflow = ocean_outflow
        water_data.water_biomes_map = water_biomes_map
        water_data.lod_level = lod_level
        water_data.actual_size = target_size
        water_data.parameters = parameters.copy()
        water_data.validity_state = "valid"
        water_data.parameter_hash = self._calculate_parameter_hash(parameters)

        return water_data

    def _get_prepared_water_inputs(self, lod_level: int, needed: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Holt NUR die tatsächlich angeforderten Water-Dependencies
        (Terrain/Geology/Weather-Outputs) für dieses LOD und bringt sie auf
        Water's eigene Ziel-Auflösung (_get_lod_size()). `needed` (Teilmenge von
        "heightmap"/"hardness_map"/"precip_map"/"temp_map"/"wind_map"/
        "humid_map") MUSS pro _calc_*-Methode exakt deren echte
        CALCULATOR_GRAPH-Abhängigkeiten widerspiegeln - sonst würde z.B.
        water.lake_detection (haengt laut Graph NUR von terrain.redistribution
        ab) hier fälschlich auch auf geology.hardness/weather.* warten, obwohl
        der Dispatcher diesen Knoten bereits für bereit hält, sobald nur die
        Heightmap existiert. Alle nicht angeforderten Werte werden gar nicht
        erst abgefragt (kein unnötiges Warten auf Generatoren, die dieser
        Knoten laut Graph nicht braucht).
        target_size wird immer mitgeliefert (basiert auf heightmap, falls
        angefordert, sonst auf dem ersten verfügbaren angeforderten Wert).
        """
        if needed is None:
            needed = ["heightmap", "hardness_map", "precip_map", "temp_map", "wind_map", "humid_map"]

        fetchers = {
            "heightmap": lambda: self.data_lod_manager.get_calculator_combined_heightmap(lod_level),
            "hardness_map": lambda: self.data_lod_manager.get_calculator_output(
                "geology.hardness", "hardness_map", lod_level),
            "precip_map": lambda: self.data_lod_manager.get_calculator_output(
                "weather.precipitation", "precip_map", lod_level),
            "temp_map": lambda: self.data_lod_manager.get_calculator_output(
                "weather.temperature", "temp_map", lod_level),
            "wind_map": lambda: self.data_lod_manager.get_calculator_output("weather.wind", "wind_map", lod_level),
            "humid_map": lambda: self.data_lod_manager.get_calculator_output(
                "weather.humidity", "humid_map", lod_level),
            # Mittlere Besonnung 0..1 (0 = dauerhaft verschattet). Die rohe
            # shadowmap ist (H,W,7) - ein Kanal je Sonnenwinkel-Voreinstellung;
            # gemittelt wird direkt hier, weil Water nur die durchschnittliche
            # Besonnung braucht und _interpolate_array() ohnehin nur 2D- und
            # 2-/3-kanalige Arrays kennt. Siehe _apply_biome_soil_drying().
            "insolation": lambda: self._mean_insolation(
                self.data_lod_manager.get_calculator_output(
                    "terrain.shadow", "shadowmap", lod_level)),
        }

        values = {key: fetchers[key]() for key in needed}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(f"Water: fehlende Dependencies für LOD {lod_level}: {', '.join(missing)}")

        reference = values["heightmap"] if values.get("heightmap") is not None else next(iter(values.values()))
        target_size = self._get_lod_size(lod_level, reference.shape[0])

        result = {key: self._interpolate_array(value, target_size) for key, value in values.items()}
        result["target_size"] = target_size
        return result

    @staticmethod
    def _mean_insolation(shadowmap):
        """
        Mittlere Besonnung je Pixel (0..1) aus der rohen shadowmap.

        terrain.shadow liefert (H,W,7): einen Kanal je Sonnenwinkel-
        Voreinstellung, Wert 0 = im Schatten, sonst der Beleuchtungsfaktor
        des Hangs. Der Mittelwert über die Winkel ist die Größe, die für die
        Austrocknung zählt - er enthält sowohl die Hangausrichtung als auch
        die Verschattung durch Nachbarberge.
        Gibt None zurück, wenn keine shadowmap vorliegt (der Aufrufer behandelt
        das als "keine Besonnungs-Information").
        """
        if shadowmap is None:
            return None
        if shadowmap.ndim == 3:
            return shadowmap.mean(axis=2).astype(np.float32)
        return shadowmap.astype(np.float32)

    # Fallback-Kartenausdehnung, falls DataLODManager noch keinen Live-Wert vom
    # Terrain-Tab bekommen hat (Standalone-Nutzung/Tests) - identisch zu
    # TERRAIN.MAP_DISTANCE_KM["default"].
    FALLBACK_MAP_DISTANCE_KM = 10.0

    def _is_final_lod(self, calculator_id: str, lod_level: int) -> bool:
        """
        Ist `lod_level` die letzte Runde für diesen Calculator-Lauf?

        Identisches Kriterium und identische Quelle wie
        SettlementGenerator._is_final_lod() (core/settlement_generator.py):
        das beim Request gesetzte Ziel-LOD aus
        DataLODManager.get_calculator_target_lod(). Dieser Wert steht fest,
        sobald der Request gestellt wurde, und ist damit unabhängig vom
        Fortschritt anderer Generatoren - anders als eine Ableitung aus der
        gerade verfügbaren (noch wachsenden) Heightmap-Größe, die jede
        Zwischenrunde fälschlich für final halten würde.

        Ohne gesetztes Ziel-LOD (Standalone-Aufrufe, Tests) wird auf die
        tatsächliche Heightmap-Größe zurückgegriffen.

        WOZU (2026-07-27): Erosion und Böschungswinkel rechnen nur noch in der
        letzten Runde. Vorher liefen sie bei JEDER LOD-Stufe und wurden
        aufaddiert - bei LOD 1 (32 px auf 10 km) ist der Erosions-Pinsel aber
        3% der Kartenbreite breit, und diese groben, runden Strukturen wurden
        anschliessend hochskaliert und waren dauerhaft im Ergebnis. Sie liessen
        sich durch keine spätere Verfeinerung mehr entfernen (Nutzer-Report:
        gleichförmig grosse Krater über die ganze Karte).
        """
        target = self.data_lod_manager.get_calculator_target_lod(calculator_id)
        if target is not None:
            return lod_level >= target

        from gui.OldManagers.data_lod_manager import calculate_max_lod_for_size
        full_heightmap = self.data_lod_manager.get_terrain_data("heightmap")
        if full_heightmap is not None:
            return lod_level >= calculate_max_lod_for_size(full_heightmap.shape[0])
        return lod_level >= self.data_lod_manager.get_max_lod_for_map_size()

    def _meters_per_pixel(self, target_size: int) -> float:
        """
        Reale Kantenlänge einer Zelle in Metern für dieses LOD.

        Eine Quelle für alle Water-Knoten (vorher drei wörtlich identische
        Inline-Kopien in _calc_flow_network/_calc_erosion_sedimentation/
        _calc_thermal_erosion, jede mit einem eigenen, stillen 1000.0-Fallback).
        Der Fallback ist jetzt die tatsächliche Default-Kartenausdehnung statt
        eines glatten Zahlenwerts: 1000 m/px entsprach bei 128 px einer
        128-km-Karte und damit einem völlig anderen Maßstab als dem, den der
        Terrain-Tab per Default einstellt - Erosion und Fließverhalten wären in
        diesem Fall lautlos gegen die falsche Größenordnung kalibriert gewesen.
        """
        map_distance_km = self.data_lod_manager.get_map_distance_km()
        if not map_distance_km or map_distance_km <= 0:
            map_distance_km = self.FALLBACK_MAP_DISTANCE_KM
        return (map_distance_km * 1000.0) / target_size

    def _resize_nearest(self, array, target_size):
        """
        Nearest-Neighbor-Resize für Label-/Integer-Arrays wie lake_map (Basin-
        IDs inkl. -1-Sentinel für "kein See") - die vorhandene bilineare
        _interpolate_2d() würde zwischen benachbarten Basin-IDs (z.B. 2 und 5)
        sinnlose Zwischenwerte (3.5) erzeugen. Gebraucht als Absicherung, wenn
        lake_map (best-verfügbares LOD <= angefordertem Level, siehe
        get_calculator_output()) von einer anderen tatsächlichen Auflösung
        stammt als die aktuell verwendete heightmap für dasselbe LOD - z.B.
        wenn map_size zwischen zwei Generierungs-Durchläufen geändert wurde
        und noch ein andersgroßer lake_map-Eintrag im Calculator-Cache liegt.
        """
        old_size = array.shape[0]
        if old_size == target_size:
            return array
        indices = np.clip((np.arange(target_size) * (old_size / target_size)).astype(np.int64), 0, old_size - 1)
        return array[np.ix_(indices, indices)]

    def _calc_lake_detection(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'water.lake_detection' (#15)

        `full_basin_map` (ungefilterte Wasserscheiden-Zuordnung, nur fuer die
        D8-Wasserscheiden-Umleitung gebraucht) entfaellt seit dem Pipe-
        Modell-Umbau 2026-07-25 - Senken fuellen/laufen im Pipe-Modell von
        selbst ueber, keine gesonderte Umleitung mehr noetig (siehe
        PipeFlowSimulator). Nur noch das volumen-gefilterte `lake_map` (fuer
        die Wasserkoerper-Klassifikation "ist das ein sichtbarer See") bleibt.
        """
        self._update_progress("Lake Detection", 40, "Detecting local minima...")
        inputs = self._get_prepared_water_inputs(lod_level, needed=["heightmap"])
        lake_map, _valid_lakes = self.lake_detection.detect_lakes(
            inputs["heightmap"], self._current_parameters,
            meters_per_pixel=self._meters_per_pixel(inputs["target_size"]))
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"lake_map": lake_map})

    def _calc_flow_network(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'water.flow_network' (#16) - treibt PipeFlowSimulator
        (siehe dort, ersetzt das fruehere D8-Routing) fuer lod_iterations['flow']
        Schritte.

        previous_depth/previous_flux: konvergierter Pipe-Zustand der
        vorherigen (kleineren) LOD-Stufe, aus den eigenen depth_state/
        flux_state-Outputs der letzten Runde gelesen - jede LOD-Stufe baut so
        auf dem bereits eingeschwungenen Wasserstand der Vorstufe auf statt
        bei einer komplett trockenen Karte neu zu starten (analog zu
        Vorstufe). depth_state/flux_state/
        ocean_outflow sind seit 2026-07-27 regulaer in CALCULATOR_GRAPH
        deklariert (vorher undeklarierte Zusatz-Keys) - dadurch erfasst der
        Calculator-Storage-Purge in DataLODManager.invalidate_cache_lod() sie
        mit, statt einen Wasserstand aus einem verworfenen Lauf stehen zu
        lassen.

        water_biomes_map ist auf dieser Stufe die ZENTRALLINIE (ein Pixel
        breit); die gemalte Fassung liefert water.manning_flow als eigenen
        Output (siehe _calc_manning_flow()).
        """
        self._update_progress("Flow Network", 55, "Simulating pipe flow...")
        inputs = self._get_prepared_water_inputs(
            lod_level, needed=["heightmap", "precip_map", "temp_map", "wind_map", "humid_map"])
        target_size = inputs["target_size"]
        lod_iterations = self._get_lod_iterations(target_size)
        meters_per_pixel = self._meters_per_pixel(target_size)

        # Potentielle Verdunstung als zweite Senke des Wasserkreislaufs (die
        # erste ist der Randabfluss) - haengt NUR von Wetterdaten ab, nicht
        # von der Wasser-Klassifikation, erzeugt also keinen Zyklus mit
        # water.evaporation. Siehe EvaporationCalculator-Docstring.
        potential_evaporation = self.evaporation.calculate_potential_evaporation(
            inputs["temp_map"], inputs["wind_map"], inputs["humid_map"])

        lake_map = self.data_lod_manager.get_calculator_output("water.lake_detection", "lake_map", lod_level)
        if lake_map is None:
            raise ValueError(f"water.flow_network: lake_map für LOD {lod_level} nicht verfügbar")
        if lake_map.shape[0] != inputs["heightmap"].shape[0]:
            lake_map = self._resize_nearest(lake_map, inputs["heightmap"].shape[0])

        # WARMSTART nur, wenn es wirklich einen vorigen Durchgang gibt.
        #
        # Frueher kam der Zustand aus der vorigen LOD-Runde. Ohne Leiter waere
        # der naheliegende Ersatz "lies deinen eigenen letzten Output" - der
        # ist aber bei einer ZWEITEN Generierung mit demselben Manager der des
        # VORIGEN LAUFS, nicht des vorigen Durchgangs. Genau so gemessen: der
        # Kreislauf startete beim zweiten Lauf warm und lieferte das Dreifache
        # (Reset-Test 2299 gegen 6598). Die Durchgangsnummer trennt beides
        # sauber - im ersten Durchgang startet der Kreislauf immer trocken.
        erster_durchgang = self.data_lod_manager.get_feedback_pass() <= 1

        previous_depth = None if erster_durchgang else             self.data_lod_manager.get_calculator_output(calculator_id, "depth_state", lod_level)
        if previous_depth is not None and previous_depth.shape[0] != target_size:
            previous_depth = self._interpolate_2d(previous_depth, target_size)
        previous_flux = None if erster_durchgang else             self.data_lod_manager.get_calculator_output(calculator_id, "flux_state", lod_level)
        if previous_flux is not None and previous_flux.shape[0] != target_size:
            previous_flux = np.stack(
                [self._interpolate_2d(previous_flux[:, :, i], target_size) for i in range(4)], axis=-1)

        sim_result = self.flow_network.build_flow_network(
            inputs["heightmap"], inputs["precip_map"], potential_evaporation, lake_map,
            self._current_parameters, lod_iterations, meters_per_pixel,
            previous_depth=previous_depth, previous_flux=previous_flux
        )
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level,
            {
                "flow_accumulation": sim_result["discharge_map"],
                "water_biomes_map": sim_result["water_biomes_map"],
                "water_depth": sim_result["water_depth"],
                "velocity_x": sim_result["velocity_x"],
                "velocity_y": sim_result["velocity_y"],
                # ocean_outflow ersetzt die vorherige
                # _calculate_ocean_outflow()-Berechnung (die flow_directions
                # brauchte, das es nicht mehr gibt) durch den direkt vom
                # Pipe-Modell mitgefuehrten Rand-Abfluss; evaporated_volume
                # ist die zweite Senke; depth_state/flux_state sind der
                # interne Roh-Zustand fuer die naechste LOD-Stufe (siehe
                # Docstring oben). Alle in CALCULATOR_GRAPH deklariert.
                "ocean_outflow": sim_result["edge_outflow"],
                "evaporated_volume": sim_result["evaporated_volume"],
                "depth_state": sim_result["depth_state"],
                "flux_state": sim_result["flux_state"],
            })

    def _calc_manning_flow(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'water.manning_flow' (#18) - seit dem Pipe-Modell-
        Umbau (Stufe 3) auf reine Fluss-Breiten-Malerei geschrumpft
        (calculate_channel_width/paint_channel_width, siehe
        ManningFlowCalculator-Docstring). Die frühere, unabhängige Manning-
        Kanalgeometrie-Suche für flow_speed/water_depth ist gelöscht - beide
        kommen jetzt direkt aus water.flow_network (echte simulierte Werte,
        siehe PipeFlowSimulator), eine zweite, davon abweichende
        Schätzung derselben physikalischen Größe wäre die "zwei Wahrheiten"-
        Falle, die bereits einmal in _calculate_stream_power_erosion() gejagt
        wurde. cross_section wird direkt aus der Kontinuitätsgleichung
        (Fläche = Durchfluss / Geschwindigkeit) aus den simulierten Werten
        abgeleitet, statt separat geloest zu werden - gleiche physikalische
        Beziehung, nur andere Eingabequelle.

        Outputs (alle unter DIESER calculator_id, siehe CALCULATOR_GRAPH):
        cross_section, channel_width und water_biomes_map - letzteres die
        GEMALTE Wasser-Klassifikation. Bis 2026-07-27 schrieb diese Methode
        die gemalte Fassung stattdessen in den water_biomes_map-Output-Slot
        von water.flow_network zurueck und sicherte die ueberschriebene
        Zentrallinie unter einem undeklarierten Zusatz-Key. Das war eine
        Wettlaufsituation: der CalculatorDispatcher startet alle in derselben
        Runde bereiten Knoten parallel, und water.evaporation/
        biome.super_override/settlement.suitability hingen nur von
        water.flow_network ab - welche der beiden Fassungen sie lasen, war
        thread-timing-abhaengig. Beide Fassungen liegen jetzt als getrennte,
        deklarierte Outputs zweier verschiedener Knoten vor
        (get_calculator_output() ist auf (LOD, calculator_id, key)
        geschluesselt), und jeder Konsument haengt explizit von dem Knoten ab,
        dessen Fassung er braucht.
        """
        self._update_progress("Manning Flow", 70, "Painting channel width...")
        inputs = self._get_prepared_water_inputs(lod_level, needed=["heightmap"])
        flow_accumulation = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "flow_accumulation", lod_level)
        velocity_x = self.data_lod_manager.get_calculator_output("water.flow_network", "velocity_x", lod_level)
        velocity_y = self.data_lod_manager.get_calculator_output("water.flow_network", "velocity_y", lod_level)
        centerline_map = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "water_biomes_map", lod_level)
        missing = [name for name, value in (
            ("flow_accumulation", flow_accumulation), ("velocity_x", velocity_x),
            ("velocity_y", velocity_y), ("water_biomes_map", centerline_map),
        ) if value is None]
        if missing:
            raise ValueError(f"water.manning_flow: fehlende Inputs für LOD {lod_level}: {', '.join(missing)}")

        flow_speed = np.sqrt(velocity_x.astype(np.float64) ** 2 + velocity_y.astype(np.float64) ** 2)
        # Kontinuitätsgleichung: Fläche = Durchfluss / Geschwindigkeit -
        # gegen Divisions-Explosion bei fast trockenen Zellen (beide ~0)
        # gedeckelt, dort ist die Fläche ohnehin irrelevant (kein Fluss).
        cross_section = np.where(
            flow_speed > 0.1, flow_accumulation / np.maximum(flow_speed, 0.1), 0.0
        ).astype(np.float32)

        # River Stage 3: Flussbreite aus dem Querschnitt malen. Ohne einen
        # einzigen Fluss-Pixel (z.B. extrem trockene Karte) bleibt die gemalte
        # Fassung identisch zur Zentrallinie und channel_width durchgehend 0 -
        # die Outputs werden trotzdem geschrieben, damit nachgelagerte Knoten
        # nie auf einen fehlenden Key laufen.
        stream_mask = (centerline_map >= 1) & (centerline_map <= 3)
        channel_width = np.zeros(centerline_map.shape, dtype=np.float32)
        painted_map = centerline_map.copy()
        if np.any(stream_mask):
            channel_width = self.manning_calculator.calculate_channel_width(
                cross_section, inputs["heightmap"], stream_mask)
            painted_map = self.manning_calculator.paint_channel_width(
                centerline_map, channel_width,
                self._meters_per_pixel(inputs["target_size"]), flow_speed=flow_speed)

        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level,
            {
                "cross_section": cross_section,
                "channel_width": channel_width,
                "water_biomes_map": painted_map,
            })

    # =========================================================================
    # === ALTBESTAND DROPLET-EROSION (stillgelegt 2026-07-28) =================
    #
    # Die beiden folgenden Calculator-Knoten haben KEINEN Aufrufer mehr: weder
    # CALCULATOR_GRAPH noch _execute_generation() kennen sie. Ersetzt durch den
    # Knoten erosion.hydraulic in core/erosion_generator.py (Feldverfahren:
    # Pipe-Hydraulik + mitstroemendes Sedimentfeld), der zwischen Geology und
    # Weather laeuft.
    #
    # Sie bleiben als LAUFFAEHIGER Code stehen statt auskommentiert, weil genau
    # das den A/B-Vergleich moeglich macht: smoke_test_water_edge_sediment.py,
    # smoke_test_water_drainage_erosion.py und
    # smoke_test_water_erosion_quality.py sprechen DropletErosionSystem direkt
    # an und laufen unveraendert weiter. Ein zeilenweises Auskommentieren
    # einer 700-Zeilen-Klasse haette die Datei unlesbar gemacht und genau
    # diesen Vergleich zerstoert.
    #
    # KOENNEN VOLLSTAENDIG GELOESCHT WERDEN, sobald das Feldmodell freigegeben
    # ist - zusammen mit DropletErosionSystem und den zugehoerigen Tests.
    # =========================================================================
    def _calc_erosion_sedimentation(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'water.erosion_sedimentation' (#19) - vollstaendig
        entkoppelt von water.flow_network: Partikel spawnen gleichverteilt
        ueber die Karte, kein Zugriff auf simulierten Abfluss/Geschwindigkeit
        noetig. Laeuft deshalb VOR water.lake_detection/water.flow_network in
        der Ausfuehrungsreihenfolge - Erosion modelliert geologische Zeit, der
        Wasserkreislauf simuliert den heutigen Zustand AUF dem bereits
        erodierten Gelaende.

        Rechnet NUR in der letzten LOD-Runde (siehe _is_final_lod()); alle
        frueheren Runden schreiben Nullkarten. Vorher lief die Erosion bei
        jeder Stufe und wurde ueber previous_erosion_map/
        previous_sedimentation_map aufaddiert - die groben Strukturen der
        niedrigen Stufen (bei LOD 1 ist der Pinsel 3% der Kartenbreite breit)
        wurden hochskaliert und waren danach unentfernbar im Ergebnis. Mit dem
        Gate entfaellt diese Kumulation ersatzlos; die volle Partikelzahl
        entfaellt stattdessen auf die eine Runde, die tatsaechlich zaehlt.

        Die Nullkarten der frueheren Runden sind kein Platzhalter, sondern die
        physikalisch richtige Aussage "auf dieser Stufe wurde noch nicht
        erodiert" - get_calculator_combined_heightmap() zieht sie ab bzw.
        addiert sie und erhaelt korrekt das unerodierte Gelaende.
        """
        # Den Vorstand BEIDER gelaendeformender Knoten fuer dieses LOD
        # verwerfen, BEVOR die Heightmap gelesen wird.
        #
        # get_calculator_combined_heightmap() zieht sowohl erosion_map als auch
        # thermal_erosion_map ab und addiert beide Ablagerungskarten. Bei einer
        # zweiten Generierung auf demselben LOD (Parameteraenderung, erneuter
        # Klick auf GENERIEREN) wuerde sonst auf dem bereits geformten Gelaende
        # des vorherigen Laufs weitererodiert und der Abtrag mit jedem Lauf
        # anwachsen.
        #
        # Warum hier auch der Thermal-Knoten mit weggeraeumt wird, obwohl er
        # das gleich darauf selbst tut: er laeuft laut CALCULATOR_GRAPH NACH
        # diesem Knoten. Raeumte jeder nur sich selbst, laese die Erosion die
        # Thermal-Karten des VERWORFENEN Laufs noch mit - gemessen im
        # Reset-Regressionstest als 1.6% Abweichung der Erosionssumme. Beide
        # Karten beschreiben dasselbe Gelaende und muessen gemeinsam auf den
        # Ausgangsstand zurueck. Siehe
        # DataLODManager.clear_calculator_node_output().
        for terrain_forming_id in self.TERRAIN_FORMING_CALCULATORS:
            self.data_lod_manager.clear_calculator_node_output(terrain_forming_id, lod_level)

        inputs = self._get_prepared_water_inputs(lod_level, needed=["heightmap", "hardness_map"])
        target_size = inputs["target_size"]

        if not self._is_final_lod(calculator_id, lod_level):
            zeros = np.zeros((target_size, target_size), dtype=np.float32)
            self.data_lod_manager.set_calculator_output(
                calculator_id, lod_level,
                {"erosion_map": zeros, "sedimentation_map": zeros.copy()})
            return

        self._update_progress("Erosion-Sedimentation", 5, "Simulating hydraulic droplets...")
        lod_iterations = self._get_lod_iterations(target_size)
        meters_per_pixel = self._meters_per_pixel(target_size)

        erosion_map, sedimentation_map = self.erosion_system.simulate_erosion_sedimentation(
            inputs["heightmap"], inputs["hardness_map"], self._current_parameters,
            lod_iterations, meters_per_pixel=meters_per_pixel
        )

        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"erosion_map": erosion_map, "sedimentation_map": sedimentation_map})

    def _calc_thermal_erosion(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'water.thermal_erosion' - Böschungswinkel-Erosion
        ("Phase 6", siehe ThermalErosionSystem), läuft NACH water.erosion_
        sedimentation. inputs["heightmap"] (über _get_prepared_water_inputs ->
        get_calculator_combined_heightmap()) liest bereits die fluvial
        eroderte/sedimentierte Landschaft DIESES Durchlaufs - Fluss-Erosion
        schneidet zuerst das schmale V-Kerbtal, Thermal Erosion kollabiert/
        verbreitert es danach je nach Härte (siehe ThermalErosionSystem-
        Docstring). Kein Sonderfall nötig: get_calculator_combined_heightmap()
        liest water.erosion_sedimentation's Output direkt aus dem Calculator-
        Storage, nicht erst nach dem finalen assemble_water_data().

        Rechnet wie water.erosion_sedimentation NUR in der letzten LOD-Runde
        (siehe _is_final_lod()) - beide Knoten formen das Gelände und dürfen
        keine grob aufgelösten Zwischenstände einbacken. Die LOD-Kumulation
        entfällt damit ersatzlos.
        """
        # Wie bei water.erosion_sedimentation: eigenen Vorstand dieses LODs
        # verwerfen, bevor die Heightmap gelesen wird - sie enthaelt sonst die
        # Boeschungs-Erosion des vorherigen Laufs (siehe
        # DataLODManager.clear_calculator_node_output()). Die FLUVIALEN Karten
        # bleiben dabei unberuehrt; genau die soll dieser Knoten ja sehen.
        self.data_lod_manager.clear_calculator_node_output(calculator_id, lod_level)

        inputs = self._get_prepared_water_inputs(lod_level, needed=["heightmap", "hardness_map"])
        target_size = inputs["target_size"]

        if not self._is_final_lod(calculator_id, lod_level):
            zeros = np.zeros((target_size, target_size), dtype=np.float32)
            self.data_lod_manager.set_calculator_output(
                calculator_id, lod_level,
                {"thermal_erosion_map": zeros, "thermal_deposition_map": zeros.copy()})
            return

        self._update_progress("Thermal Erosion", 20, "Simulating angle of repose...")
        lod_iterations = self._get_lod_iterations(target_size)
        meters_per_pixel = self._meters_per_pixel(target_size)

        thermal_erosion_map, thermal_deposition_map = self.thermal_erosion.simulate(
            inputs["heightmap"], inputs["hardness_map"], self._current_parameters,
            lod_iterations["thermal"], meters_per_pixel
        )

        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level,
            {"thermal_erosion_map": thermal_erosion_map, "thermal_deposition_map": thermal_deposition_map})

    def _calc_soil_moisture(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'water.soil_moisture' (#20).

        Liest BEIDE Wasser-Klassifikationen (siehe _calc_manning_flow):
        - water.manning_flow/water_biomes_map (GEMALT) fuer die Klassifikations-
          Stufe pro Pixel,
        - water.flow_network/water_biomes_map (ZENTRALLINIE) als 100%-Feuchte-
          Quellflaeche.
        Ohne diese Trennung waechst die Boden-Feuchte-Ausdehnung automatisch
        mit, sobald Fluesse breiter gemalt werden (gemeldeter Effekt: "fast
        100% fast ueberall"). Beide Knoten sind deklarierte Abhaengigkeiten
        dieses Knotens, beide Karten liegen daher garantiert fuer DIESES LOD
        vor - die frueher noetige Nearest-Neighbor-Reconciliation zwischen
        zwei unterschiedlich weit fortgeschrittenen LOD-Staenden entfaellt
        damit ersatzlos.
        """
        self._update_progress("Soil Moisture", 88, "Calculating gaussian diffusion...")
        flow_accumulation = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "flow_accumulation", lod_level)
        water_mask_source = self.data_lod_manager.get_calculator_output(
            "water.flow_network", "water_biomes_map", lod_level)
        water_biomes_map = self.data_lod_manager.get_calculator_output(
            "water.manning_flow", "water_biomes_map", lod_level)
        missing = [name for name, value in (
            ("flow_accumulation", flow_accumulation),
            ("water_biomes_map (Zentrallinie)", water_mask_source),
            ("water_biomes_map (gemalt)", water_biomes_map),
        ) if value is None]
        if missing:
            raise ValueError(f"water.soil_moisture: fehlende Inputs für LOD {lod_level}: {', '.join(missing)}")

        soil_moist_map = self.soil_moisture.calculate_soil_moisture(
            water_biomes_map, flow_accumulation, self._current_parameters,
            water_mask_source=water_mask_source)

        # Biom-abhängige Trocknung + Kapazitäts-Grenze (Biome-Preseed-Plan
        # Punkt C) - best-effort: ohne verfügbare temp_map/Biom-Quelle bleibt
        # das bisherige Verhalten (reine Distanz-zu-Wasser-Diffusion, 0-100)
        # unverändert erhalten, kein harter Fehler.
        temp_map = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map", lod_level)
        if temp_map is not None:
            biome_hint_map = None
            # Echte Biome-Klassifikation des VORIGEN Rueckkopplungs-Durchgangs
            # (siehe FEEDBACK_PASSES in calculator_graph.py). Kein Zyklus: der
            # Wert stammt aus einem abgeschlossenen Durchgang, nicht aus dem
            # laufenden. Im ersten Durchgang bewusst None -> Pre-Biome unten;
            # ohne diese Abfrage laege beim zweiten LAUF die Biome-Karte des
            # vorigen Laufs vor und das Ergebnis haenge daran, wie oft man
            # schon generiert hat.
            biome_hint_map = None
            if self.data_lod_manager.get_feedback_pass() > 1:
                biome_hint_map = self.data_lod_manager.get_calculator_output(
                    "biome.integrate_layers", "biome_map", lod_level)
            if biome_hint_map is None:
                # Allererste LOD-Runde (oder Biome wurde für die Vorstufe nie
                # angefragt) - billiger Slope+Breitengrad-Schätzwert statt
                # der echten Klassifikation (siehe core/biome_generator.py.
                # _calc_preseed_hint()).
                biome_hint_map = self.data_lod_manager.get_calculator_output(
                    "biome.preseed_hint", "preseed_biome_map", lod_level)
            if biome_hint_map is not None:
                insolation = self._mean_insolation(self.data_lod_manager.get_calculator_output(
                    "terrain.shadow", "shadowmap", lod_level))
                soil_moist_map = self._apply_biome_soil_drying(
                    soil_moist_map, temp_map, biome_hint_map, water_mask_source, insolation)

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"soil_moist_map": soil_moist_map})

    # Glättungsradius (Pixel) für die Boden-Feuchte-Kapazitätsgrenze - siehe
    # _apply_biome_soil_drying(). Klein gehalten (ähnliche Größenordnung wie
    # FLOODPLAIN_BLUR_SIGMA_PX), soll Biom-Grenzen weich überblenden, nicht
    # die Kapazitäts-Unterschiede zwischen weit auseinanderliegenden Biomen
    # verwischen.
    SOIL_CAPACITY_SMOOTHING_SIGMA_PX = 2.0

    # Wie stark die Besonnung die Austrocknung skaliert (siehe
    # _apply_biome_soil_drying). Der Faktor ist RELATIV zum Kartenmittel: ein
    # durchschnittlich besonnter Hang bleibt bei 1.0, ein dauerhaft
    # verschatteter Nordhang trocknet nur mit INSOLATION_DRYING_MIN, ein voll
    # besonnter Südhang mit INSOLATION_DRYING_MAX.
    INSOLATION_DRYING_MIN = 0.4
    INSOLATION_DRYING_MAX = 1.6

    def _apply_biome_soil_drying(self, soil_moist_map: np.ndarray, temp_map: np.ndarray,
                                  biome_index_map: np.ndarray, water_mask_source: np.ndarray,
                                  insolation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Wendet eine biom- und besonnungsabhängige Trocknung + Kapazitäts-Grenze
        auf die per Gauß-Diffusion berechnete soil_moist_map an - Wüste
        trocknet schneller aus und kann insgesamt weniger Wasser halten als
        z.B. Sumpf, und ein verschatteter Hang trocknet langsamer als ein voll
        besonnter. Echte Wasser-Kacheln (water_mask_source > 0) trocknen NICHT
        aus (ein Fluss verdunstet nicht einfach, siehe EvaporationCalculator
        für die separate, echte Wasserflächen-Verdunstung).

        `insolation` (0..1, Mittel über die Sonnenwinkel, siehe
        _mean_insolation()) setzt die Nutzer-Vorgabe um: "die ebenen,
        nordhänge (bereiche wo wenig sonne hinkommt) sind eher feucht". Die
        shadowmap enthält dabei nicht nur die Hangausrichtung, sondern auch
        die Verschattung durch Nachbarberge - ein Nordhang UND ein enges Tal
        bleiben beide feuchter. Ohne shadowmap (None) bleibt das Verhalten
        unverändert; der Term ist dann schlicht 1.0.

        biome_index_map/temp_map/insolation können von einer ANDEREN
        tatsächlichen Auflösung stammen (Vorstufen-LOD bzw. best-verfügbares
        LOD) - Nearest-Neighbor-Resize, wie _resize_nearest() für lake_map.
        """
        target_size = soil_moist_map.shape[0]
        if biome_index_map.shape[0] != target_size:
            biome_index_map = self._resize_nearest(biome_index_map, target_size)
        if temp_map.shape[0] != target_size:
            temp_map = self._resize_nearest(temp_map, target_size)
        if insolation is not None and insolation.shape[0] != target_size:
            insolation = self._resize_nearest(insolation, target_size)

        biome_idx = np.clip(biome_index_map.astype(np.int32), 0, len(_BIOME_EVAPORATION_FACTOR) - 1)
        evap_factor = _BIOME_EVAPORATION_FACTOR[biome_idx]
        # Kapazität räumlich geglättet statt der rohen kategorialen Pro-Biom-
        # Tabelle (Nutzer-Feedback 2026-07-25: "Bodenfeuchte ist sehr on/off").
        # Der harte np.clip(..., capacity) unten erzeugte an JEDER Biom-Grenze
        # eine scharfe Kante/ein Plateau - die eigentlich glatte Gauß-
        # Diffusion sah dadurch stufig statt kontinuierlich aus. Ein kleiner
        # Gauß-Filter auf die Kapazitäts-Karte selbst (nicht auf soil_moist_
        # map - die Diffusion bleibt unverändert) lässt die Kapazitätsgrenze
        # über SOIL_CAPACITY_SMOOTHING_SIGMA_PX Pixel sanft überblenden,
        # statt abrupt zu springen.
        capacity_raw = _BIOME_MOISTURE_CAPACITY[biome_idx]
        capacity = gaussian_filter(capacity_raw, sigma=self.SOIL_CAPACITY_SMOOTHING_SIGMA_PX, mode='nearest')

        # Nur bei Wärme über einem Referenzwert - kalibrierbare Konstanten,
        # DRYING_COEFF so gewählt, dass ein Wüsten-Pixel (evap_factor≈1.8) bei
        # deutlicher Wärme (+20°C über Referenz) spürbar, aber nicht
        # schlagartig unter seine Kapazität fällt.
        REFERENCE_TEMP_C = 15.0
        DRYING_COEFF = 0.8
        drying = np.maximum(0.0, temp_map - REFERENCE_TEMP_C) * evap_factor * DRYING_COEFF

        # Besonnungs-Term: relativ zum Kartenmittel, damit er die Verteilung
        # ÜBER die Karte formt und nicht das Gesamtniveau verschiebt (eine
        # global dunkle Karte soll nicht pauschal feuchter werden - dafür ist
        # die Temperatur zuständig).
        drying = drying * self._insolation_drying_factor(insolation)

        dried = soil_moist_map - drying
        result = np.clip(dried, 0.0, capacity).astype(np.float32)
        # Echte Wasser-Kacheln umgehen sowohl die Trocknung ALS AUCH die
        # Kapazitäts-Grenze (der Clip oben würde sie sonst fälschlich auf die
        # Kapazität des dort zufällig klassifizierten Land-Biomes deckeln,
        # z.B. einen Fluss mitten in der Wüste auf 20 statt 100) - ein Fluss
        # trocknet nicht aus, nur die diffuse Umgebung.
        water_mask = water_mask_source > 0
        result[water_mask] = soil_moist_map[water_mask]
        return result

    def _insolation_drying_factor(self, insolation):
        """
        Multiplikator für die Austrocknung aus der mittleren Besonnung.
        1.0 = durchschnittlich besonnt, < 1 = schattiger als der
        Kartendurchschnitt (bleibt feuchter), > 1 = sonniger.
        Ohne Besonnungs-Information (None) oder auf einer gleichmäßig
        beleuchteten Karte ist der Faktor überall exakt 1.0.
        """
        if insolation is None:
            return 1.0
        values = np.asarray(insolation, dtype=np.float64)
        mean_insolation = float(values.mean())
        if not np.isfinite(mean_insolation) or mean_insolation <= 1e-6:
            return 1.0
        return np.clip(values / mean_insolation,
                       self.INSOLATION_DRYING_MIN, self.INSOLATION_DRYING_MAX)

    def _calc_evaporation(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'water.evaporation' (#21) - Sibling zu soil_moisture.
        Nutzt die GEMALTE Wasser-Klassifikation (water.manning_flow), da
        Verdunstung von der tatsaechlichen Wasser-OBERFLAECHE abhaengt, nicht
        von der ein Pixel breiten Zentrallinie - siehe _calc_manning_flow().
        """
        self._update_progress("Evaporation", 96, "Calculating atmospheric evaporation...")
        inputs = self._get_prepared_water_inputs(lod_level, needed=["temp_map", "wind_map", "humid_map"])
        water_biomes_map = self.data_lod_manager.get_calculator_output(
            "water.manning_flow", "water_biomes_map", lod_level)
        if water_biomes_map is None:
            raise ValueError(f"water.evaporation: water_biomes_map für LOD {lod_level} nicht verfügbar")

        evaporation_map = self.evaporation.calculate_evaporation(
            inputs["temp_map"], inputs["wind_map"], inputs["humid_map"], water_biomes_map,
            self._current_parameters)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"evaporation_map": evaporation_map})

    def _update_parameters(self, parameters):
        """
        Aktualisiert alle Sub-System-Parameter aus dem Slider-Dict.

        Die Fallback-Literale sind ein Sicherheitsnetz für fehlende Dict-Keys
        (in der echten Pipeline liefert WaterTab.get_current_parameters()
        immer alle Slider) - gegen die aktuellen Defaults in
        gui/config/value_default.py abgeglichen, um Drift bei künftigen
        Kalibrierungsrunden zu vermeiden.

        manning_coefficient und rain_threshold sind hier entfernt (2026-07-27):
        beide hatten seit dem Pipe-Modell-Umbau keinen Effekt mehr auf das
        Ergebnis (manning_n wurde von keiner Methode mehr gelesen,
        rain_threshold nur noch im inzwischen entfernten Simple-Fallback des
        Flow-Netzwerks). Sie sind zusammen mit ihren Slidern aus WATER und
        WaterTab gelöscht, statt als wirkungslose Regler stehen zu bleiben.
        """
        self.lake_detection.lake_volume_threshold = parameters.get('lake_volume_threshold', 5000.0)
        self.flow_network.river_abundance = parameters.get('river_abundance', 0.10)
        self.erosion_system.erosion_strength = parameters.get('erosion_strength', 2.5)
        self.erosion_system.capacity_factor = parameters.get('sediment_capacity_factor', 4.0)
        self.erosion_system.settling_velocity = parameters.get('settling_velocity', 0.3)
        self.erosion_system.erosion_passes = parameters.get('erosion_passes', 4)
        self.soil_moisture.diffusion_radius = parameters.get('diffusion_radius', 2.0)
        self.evaporation.base_rate = parameters.get('evaporation_base_rate', 0.002)
        self.thermal_erosion.thermal_strength = parameters.get('thermal_erosion_strength', 1.0)

    # Alle Array-/Skalar-Felder von WaterData, die in den Domain-Storage
    # gehören - eine Quelle statt zweier handgepflegter Listen (hier und in
    # data_lod_manager.set_water_data_complete_lod()). thermal_erosion_map/
    # thermal_deposition_map fehlten hier bis 2026-07-27.
    # Erosion-Umbau 2026-07-28: erosion_map, sedimentation_map,
    # thermal_erosion_map und thermal_deposition_map sind hier entfallen - sie
    # gehoeren jetzt ErosionSystemGenerator.EROSION_DATA_KEYS.
    WATER_DATA_KEYS = (
        "water_map", "flow_map", "flow_speed", "cross_section", "soil_moist_map",
        "evaporation_map", "ocean_outflow", "water_biomes_map",
    )

    # === ALTBESTAND (stillgelegt 2026-07-28) ===
    # Water formt kein Gelaende mehr - das macht der Erosion-Generator
    # (core/erosion_generator.py), der seinen eigenen Vorstand selbst wegraeumt.
    # TERRAIN_FORMING_CALCULATORS = ("water.erosion_sedimentation", "water.thermal_erosion")

    # _save_to_data_manager() geloescht (2026-07-27): kein Aufrufer. Der
    # GenerationOrchestrator schreibt Water-Ergebnisse ueber
    # assemble_water_data() + DataLODManager.set_water_data_complete_lod().
    # WATER_DATA_KEYS oben bleibt - genau diese Liste liest der DataLODManager.

    # update_seed() geloescht (2026-07-27): rief `super().update_seed()` auf,
    # obwohl HydrologySystemGenerator von `object` erbt - der Aufruf endete
    # zwangslaeufig in `AttributeError: 'super' object has no attribute
    # 'update_seed'`. Der einzige Aufrufer war die ebenfalls geloeschte
    # Legacy-Methode generate_hydrology_system(). Der Seed kommt in der
    # echten Pipeline ueber den Konstruktor bzw. den Parameter-Dict, nicht
    # ueber einen Setter.

    def _get_lod_size(self, lod: int, original_size: int) -> int:
        """
        Bestimmt die Zielgröße dieses LOD-Levels. Der frühere String-LOD-Pfad
        ("LOD64"/"LOD128"/"LOD256"/"FINAL") ist mit seinem letzten Nutzer
        (generate_hydrology_system(), 2026-07-27 gelöscht) entfallen.
        """
        from gui.config.value_default import TERRAIN
        target_size = TERRAIN.MAPSIZEMIN * (2 ** (int(lod) - 1))
        return min(target_size, original_size)

    # --- Iterations-/Partikelbudgets, alle größenbezogen statt als Tabelle ---
    #
    # Bis 2026-07-27 war das eine feste Tabelle pro LOD-Level mit einem
    # `else`-Zweig für "LOD 4 und höher". Der Docstring versprach ~0.3
    # Partikel/Pixel, was nur bis LOD 4 (256 px) stimmte: bei 512 px waren es
    # noch 0.076, bei 1024 px nur noch 0.019 - die Erosion war dort also pro
    # Pixel 16x schwächer als bei 256 px, ohne dass irgendein Regler das
    # angezeigt hätte. Alle Budgets leiten sich jetzt aus der tatsächlichen
    # Kantenlänge ab.

    # Partikel pro Pixel.
    #
    # Von 0.3 auf 5.0 angehoben (2026-07-27, Nutzer-Vorgabe "viele viele
    # weniger starke Partikel"): seit die Erosion nur noch in der letzten
    # LOD-Runde rechnet, entfällt das gesamte Budget auf genau einen Durchlauf,
    # und seit die Wassermenge pro Tropfen umgekehrt mit der Dichte skaliert
    # (DropletErosionSystem.initial_water_volume()) verbessert eine höhere
    # Partikelzahl das Ergebnis, statt es nur teurer zu machen. Gemessen bei
    # 128², Anteil der Erosion in den stärksten 5% der Zellen:
    # 20k -> 0.278, 80k -> 0.403, 240k -> 0.455.
    #
    # 5.0 entspricht bei der Default-Kartengröße (128 px) rund 80.000
    # Partikeln. Laufzeit auf dem vektorisierten CPU-Pfad dort ~6 s; sie wächst
    # mit Pixelzahl x Schrittzahl, weshalb bei sehr großen finalen Auflösungen
    # der GPU-Pfad der vorgesehene Weg ist.
    EROSION_PARTICLES_PER_PIXEL = 5.0

    # Pipe-Fluss-Schritte: reine KONVERGENZ-Iterationen. Sie steuern seit der
    # Umstellung auf zeitbasierten Regen (siehe PipeFlowSimulator.simulate())
    # nicht mehr die Wassermenge, sondern nur noch, wie weit sich das System
    # einschwingt. Wasser legt pro Schritt höchstens eine Zelle zurück, die
    # nötige Schrittzahl wächst also linear mit der Kantenlänge - Bezugspunkt
    # ist die alte, gegen 128 px kalibrierte Stufe (200 Schritte).
    FLOW_STEPS_PER_EDGE_PIXEL = 200 / 128
    FLOW_STEPS_MIN = 50
    FLOW_STEPS_MAX = 2000

    # Böschungswinkel-Iterationen: FESTE Zahl, bewusst nicht an die
    # Kantenlänge gekoppelt.
    #
    # Der Prozess ist deckel-begrenzt (siehe
    # ThermalErosionSystem.CAP_RELIEF_FRACTION): die verlagerte Menge ist
    # näherungsweise `Deckel x Iterationszahl`, nicht das Erreichen eines
    # Gleichgewichts. Eine mit der Kantenlänge wachsende Iterationszahl
    # bedeutete deshalb schlicht mehr Abtrag bei feinerer Auflösung - bei
    # 512 px das Vierfache von 128 px auf derselben Karte. Mit einem festen
    # Wert bleibt die Gesamtverlagerung über alle Auflösungen vergleichbar.
    THERMAL_ITERATIONS = 40

    def _get_lod_iterations(self, edge_pixels: int) -> Dict[str, int]:
        """
        Iterations-/Partikelbudgets für eine Karte mit `edge_pixels`
        Kantenlänge, alle aus dieser Kantenlänge abgeleitet (siehe Konstanten
        oben). Der Parameter ist bewusst die TATSÄCHLICHE Zielauflösung
        (inputs["target_size"]) statt des LOD-Levels: _get_lod_size() deckelt
        das nominelle LOD-Ziel auf die vorhandene Kartengrösse, und die
        Budgets müssen zur tatsächlich gerechneten Auflösung passen, nicht zur
        nominellen.

        Der 'manning'-Key ist entfernt (2026-07-27) - er wurde nirgends
        gelesen, seit ManningFlowCalculator auf reine Breiten-Malerei
        geschrumpft ist.

        Achtung Laufzeit: `erosion_particles` wächst quadratisch mit der
        Kantenlänge, und DropletErosionSystem._cpu_simulate() ist eine
        sequentielle Python-Schleife (Partikel x Lebenszeit-Schritte x
        Pinselfläche). Bei grossen finalen Auflösungen ist der GPU-Pfad
        (shaders/water/dropletErosionStep.comp, registriert in
        shader_manager.DISPATCH_TABLE) deshalb nicht optional, sondern der
        vorgesehene Weg; der CPU-Pfad bleibt die verifizierbare Referenz.
        """
        return {
            'erosion_particles': max(1, int(round(edge_pixels ** 2 * self.EROSION_PARTICLES_PER_PIXEL))),
            'flow': int(np.clip(round(edge_pixels * self.FLOW_STEPS_PER_EDGE_PIXEL),
                                self.FLOW_STEPS_MIN, self.FLOW_STEPS_MAX)),
            'thermal': self.THERMAL_ITERATIONS,
        }

    def _interpolate_array(self, array, target_size):
        """Interpoliert Arrays aller Typen auf neue Größe"""
        if array is None:
            return None

        if len(array.shape) == 2:
            # 2D Array (heightmap, temp_map, etc.)
            return self._interpolate_2d(array, target_size)
        elif len(array.shape) == 3:
            if array.shape[2] == 2:
                # 3D Array mit 2 Kanälen (wind_map, slopemap)
                result = np.zeros((target_size, target_size, 2), dtype=array.dtype)
                result[:, :, 0] = self._interpolate_2d(array[:, :, 0], target_size)
                result[:, :, 1] = self._interpolate_2d(array[:, :, 1], target_size)
                return result
            elif array.shape[2] == 3:
                # 3D Array mit 3 Kanälen (rock_map RGB)
                result = np.zeros((target_size, target_size, 3), dtype=array.dtype)
                for channel in range(3):
                    result[:, :, channel] = self._interpolate_2d(array[:, :, channel], target_size)

                # Massenerhaltung für rock_map: R+G+B=255
                if array.dtype == np.uint8:  # Vermutlich rock_map
                    result = self._ensure_mass_conservation(result)

                return result

        raise ValueError(f"Unsupported array shape for interpolation: {array.shape}")

    def _interpolate_2d(self, array, target_size):
        """
        Bilineare Interpolation für 2D-Arrays.

        Vektorisiert über scipy.ndimage.map_coordinates (2026-07-27) - die
        vorherige Python-Doppelschleife kostete bei 512² 0.50 s und wurde von
        _get_prepared_water_inputs() für JEDES Input-Array JEDES Water-Knotens
        aufgerufen (bis zu 7 Knoten x mehrere Karten pro LOD-Stufe).
        Abtastgitter und Randbehandlung sind identisch: dieselbe
        scale_factor-Formel (old-1)/(target-1), und `mode='nearest'` entspricht
        dem Klemmen von x1/y1 auf old_size-1 in der alten Fassung.

        Integer-Eingaben werden jetzt gerundet statt abgeschnitten. Die alte
        Fassung schrieb das Zwischenergebnis direkt in ein Array vom dtype der
        Eingabe, wodurch jeder interpolierte Wert systematisch abgerundet wurde
        (ein konstanter Abwärts-Bias von bis zu 1 pro Pixel). Für Float-Karten
        - alles, was Water tatsächlich interpoliert - ändert sich nichts.
        """
        old_size = array.shape[0]
        if old_size == target_size:
            return array.copy()
        if target_size < 2:
            raise ValueError(f"_interpolate_2d: target_size muss >= 2 sein, war {target_size}")

        scale_factor = (old_size - 1) / (target_size - 1)
        axis = np.arange(target_size, dtype=np.float64) * scale_factor
        grid_y, grid_x = np.meshgrid(axis, axis, indexing='ij')

        interpolated = map_coordinates(
            array.astype(np.float64, copy=False), [grid_y, grid_x], order=1, mode='nearest')

        if np.issubdtype(array.dtype, np.integer):
            return np.rint(interpolated).astype(array.dtype)
        return interpolated.astype(array.dtype, copy=False)

    def _ensure_mass_conservation(self, rock_map):
        """
        Stellt sicher, dass R+G+B=255 für rock_map nach der Interpolation gilt.
        Vektorisiert (2026-07-27) - vorher eine Python-Doppelschleife über die
        ganze Karte. Verhalten unverändert, inklusive der Gleichverteilung
        (85/85/85) für Pixel mit Gesamtsumme 0.
        """
        channels = np.asarray(rock_map, dtype=np.float32)
        total = channels.sum(axis=2, keepdims=True)

        normalized = np.divide(channels, total, out=np.zeros_like(channels), where=total > 0) * 255.0
        return np.where(total > 0, normalized, 85.0).astype(np.uint8)

    def _create_water_depth_map(self, water_biomes_map, water_depth):
        """
        Erstellt die Wasser-Tiefen-Map fuer die Anzeige direkt aus der ECHTEN,
        vom Pipe-Modell simulierten Wassertiefe (water_depth, siehe
        PipeFlowSimulator) statt der frueheren Schaetzung aus
        flow_accumulation*0.01 (See) bzw. cross_section/10 (Fluss) - beide
        waren reine Proxys fuer eine Groesse, die jetzt direkt vorliegt.
        Trockene Pixel (water_biomes_map==0) bleiben 0, auch wenn dort
        numerisch eine winzige Restfeuchte simuliert wurde (z.B. frisch
        gefallener, noch nicht abgeflossener Regen).
        """
        water_map = np.where(water_biomes_map > 0, water_depth, 0.0).astype(np.float32)
        return np.clip(water_map, 0.0, 50.0)

    # === ENTFERNT 2026-07-28: _create_minimal_water_data() ===
    # Lieferte bei einem Fehler Nullkarten mit validity_state="fallback". Der
    # einzige Aufrufer war der `except Exception`-Zweig in
    # _execute_generation(), der jetzt weiterwirft - ein Programmfehler soll
    # als Fehler ankommen und nicht als leere, aber "erfolgreiche" Wasserkarte.
    # Begruendung und der konkrete Fall, der das ausgeloest hat, stehen dort.

    def _calculate_parameter_hash(self, parameters):
        """Berechnet Hash für Parameter-basierte Cache-Invalidation"""
        import hashlib
        param_str = str(sorted(parameters.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def set_progress_callback(self, callback):
        """Setzt den Fortschritts-Callback (phase, progress, message) -
        identisches Muster wie core/geology_generator.py. Wird von
        CalculatorThread für die Dauer eines Knotens gesetzt."""
        self.progress_callback = callback

    def _update_progress(self, phase, percentage, message):
        """Progress-Update für UI-Integration. Ein fehlschlagender Callback
        darf die Generierung nie abbrechen (gleiche Absicherung wie in
        geology_generator.py)."""
        if self.progress_callback:
            try:
                self.progress_callback(phase, percentage, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def calculate_hydrology(self, dependencies: Dict[str, Any], parameters: Dict[str, Any],
                           lod_level: int) -> WaterData:
        """
        Hauptmethode für Water-System-Generierung mit numerischem LOD-Level.

        Args:
            dependencies: dict mit heightmap, hardness_map, precip_map,
                         temp_map, wind_map, humid_map (bereits auf lod_level
                         vorskaliert vom DataLODManager)
            parameters: Alle Water-Parameter aus ParameterManager
            lod_level: Numerisches LOD-Level (1-6+)
            Erosion/Böschungswinkel rechnen nur in der letzten LOD-Runde
            (siehe _is_final_lod()) - es gibt keine LOD-übergreifende
            Kumulation mehr.

        Returns:
            WaterData: Vollständiges Wassersystem mit allen 10 Outputs
        """
        return self._execute_generation(lod_level, dependencies, parameters)

    # ===== ENTFERNTER LEGACY-BLOCK (2026-07-27) =====
    # generate_hydrology_system(), simulate_water_cycle(),
    # update_erosion_sedimentation(), get_hydrology_statistics() und
    # validate_mass_conservation() sind geloescht - keine davon hatte einen
    # Aufrufer im Projekt (nur descriptor.py erwaehnt sie, und das ist reine
    # Dokumentation). Sie waren zudem inhaltlich veraltet:
    # generate_hydrology_system() nahm rain_threshold/manning_coefficient
    # entgegen (beide wirkungslos und inzwischen entfernt) und fuhr ueber den
    # String-LOD-Pfad ("LOD64"), update_erosion_sedimentation() rechnete mit
    # einer eigenen, von der echten Pipeline abweichenden Formel, und
    # validate_mass_conservation() pruefte eine rock_map, die Water gar nicht
    # mehr anfasst. Mit ihnen entfaellt der letzte Nutzer des String-LOD-Pfads
    # in _get_lod_size()/_get_lod_iterations().
