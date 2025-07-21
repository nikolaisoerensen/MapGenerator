"""
Path: core/water_generator.py

Funktionsweise: Dynamisches Hydrologiesystem mit Erosion und Sedimentation
- Lake-Detection durch Jump Flooding Algorithm für parallele Senken-Identifikation
- Flussnetzwerk-Aufbau durch Steepest Descent mit Upstream-Akkumulation
- Strömungsberechnung nach Manning-Gleichung mit adaptiven Querschnitten
- Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
- Stream Power Erosion mit Hjulström-Sundborg Transport
- Realistische Sedimentation mit Transportkapazitäts-Überschreitung
- Evaporation nach Penman-Gleichung mit Wind- und Temperatureffekten

Parameter Input:
- lake_volume_threshold (Mindestvolumen für Seebildung, default 0.1m)
- rain_threshold (Niederschlagsschwelle für Quellbildung, default 5.0 gH2O/m²)
- manning_coefficient (Rauheitskoeffizient für Fließgeschwindigkeit, default 0.03)
- erosion_strength (Erosionsintensität-Multiplikator, default 1.0)
- sediment_capacity_factor (Transportkapazitäts-Faktor, default 0.1)
- evaporation_base_rate (Basis-Verdunstungsrate, default 0.002 m/Tag)
- diffusion_radius (Bodenfeuchtigkeit-Ausbreitungsradius, default 5.0 Pixel)
- settling_velocity (Sediment-Sinkgeschwindigkeit, default 0.01 m/s)

data_manager Input:
- map_seed
- heightmap
- slopemap
- hardness_map (Gesteinshärte-Verteilung)
- rock_map (RGB-Feld, Gesteinsmassen - R=Sedimentary, G=Igneous, B=Metamorphic mit R+G+B=255)
- precip_map (2D-Feld, Niederschlag in gH2O/m²)
- temp_map (2D-Feld, Lufttemperatur in °C)
- wind_map (2D-Feld, Windvektoren in m/s)
- humid_map (2D-Feld, Luftfeuchtigkeit in gH2O/m³)

Output:
- water_map (2D-Feld, Gewässertiefen) in m
- flow_map (2D-Feld, Volumenstrom) in m³/s
- flow_speed (2D-Feld, Fließgeschwindigkeit) in m/s
- cross_section (2D-Feld, Flusquerschnitt) in m²
- soil_moist_map (2D-Feld, Bodenfeuchtigkeit) in %
- erosion_map (2D-Feld, Erosionsrate) in m/Jahr
- sedimentation_map (2D-Feld, Sedimentationsrate) in m/Jahr
- rock_map_updated (RGB-Feld, Gesteinsmassen-Verteilung) - R=Sedimentary, G=Igneous, B=Metamorphic mit R+G+B=255
- evaporation_map (2D-Feld, Verdunstung) in gH2O/m²/Tag
- ocean_outflow (Skalär, Wasserabfluss ins Meer) in m³/s
- water_biomes_map (2D-Array mit Wasser-Klassifikation: 0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake)

Klassen:
HydrologySystemGenerator
    Funktionsweise: Hauptklasse für dynamisches Hydrologiesystem mit Erosion und Sedimentation
    Aufgabe: Koordiniert alle hydrologischen Prozesse und Massentransport
    Methoden: generate_hydrology_system(), simulate_water_cycle(), update_erosion_sedimentation()

LakeDetectionSystem
    Funktionsweise: Identifiziert Seen durch Jump Flooding Algorithm für parallele Senken-Identifikation
    Aufgabe: Findet alle potentiellen Seestandorte und deren Einzugsgebiete
    Methoden: detect_local_minima(), apply_jump_flooding(), classify_lake_basins()

FlowNetworkBuilder
    Funktionsweise: Baut Flussnetzwerk durch Steepest Descent mit Upstream-Akkumulation
    Aufgabe: Erstellt flow_map und water_biomes_map mit realistischen Flusssystemen
    Methoden: calculate_steepest_descent(), accumulate_upstream_flow(), classify_water_bodies()

ManningFlowCalculator
    Funktionsweise: Berechnet Strömung nach Manning-Gleichung mit adaptiven Querschnitten
    Aufgabe: Erstellt flow_speed und cross_section für realistische Fließgeschwindigkeiten
    Methoden: solve_manning_equation(), optimize_channel_geometry(), calculate_hydraulic_radius()

ErosionSedimentationSystem
    Funktionsweise: Simuliert Stream Power Erosion mit Hjulström-Sundborg Transport
    Aufgabe: Modifiziert heightmap und rock_map durch realistische Erosions-/Sedimentationsprozesse
    Methoden: calculate_stream_power(), transport_sediment(), apply_mass_conservation()

SoilMoistureCalculator
    Funktionsweise: Berechnet Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
    Aufgabe: Erstellt soil_moist_map für Biome-System und Weather-Evaporation
    Methoden: apply_gaussian_diffusion(), calculate_groundwater_effects(), integrate_moisture_sources()

EvaporationCalculator
    Funktionsweise: Berechnet Evaporation nach atmosphärischen Bedingungen
    Aufgabe: Erstellt evaporation_map basierend auf temp_map, humid_map und wind_map
    Methoden: calculate_atmospheric_evaporation(), apply_wind_effects(), limit_by_available_water()
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from collections import deque
import heapq

class WaterData:
    """
    Funktionsweise: Container für alle Water-Daten mit Metainformationen
    Aufgabe: Speichert alle 11 Hydrologie-Outputs und LOD-Informationen
    """
    def __init__(self):
        self.water_map = None              # (height, width) - Gewässertiefen in m
        self.flow_map = None               # (height, width) - Volumenstrom in m³/s
        self.flow_speed = None             # (height, width) - Fließgeschwindigkeit in m/s
        self.cross_section = None          # (height, width) - Flusquerschnitt in m²
        self.soil_moist_map = None         # (height, width) - Bodenfeuchtigkeit in %
        self.erosion_map = None            # (height, width) - Erosionsrate in m/Jahr
        self.sedimentation_map = None      # (height, width) - Sedimentationsrate in m/Jahr
        self.rock_map_updated = None       # (height, width, 3) - Aktualisierte Gesteinsmassen RGB
        self.evaporation_map = None        # (height, width) - Verdunstung in gH2O/m²/Tag
        self.ocean_outflow = None          # Scalar - Wasserabfluss ins Meer in m³/s
        self.water_biomes_map = None       # (height, width) - Wasser-Klassifikation 0-4
        self.lod_level = "LOD64"           # Aktueller LOD-Level
        self.actual_size = 64              # Tatsächliche Kartengröße
        self.parameters = {}               # Verwendete Parameter für Cache-Management

class LakeDetectionSystem:
    """
    Funktionsweise: Identifiziert Seen durch Jump Flooding Algorithm für parallele Senken-Identifikation
    Aufgabe: Findet alle potentiellen Seestandorte und deren Einzugsgebiete
    """

    def __init__(self, lake_volume_threshold=0.1):
        """
        Funktionsweise: Initialisiert Lake-Detection-System mit Volumen-Schwellwert
        Aufgabe: Setup der See-Identifikation
        Parameter: lake_volume_threshold (float) - Mindestvolumen für Seebildung
        """
        self.lake_volume_threshold = lake_volume_threshold

    def detect_local_minima(self, heightmap):
        """
        Funktionsweise: Identifiziert alle lokalen Minima als potentielle See-Seeds
        Aufgabe: Findet alle Senken in der Heightmap für Jump Flooding
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: List[Tuple] - Liste der lokalen Minima-Koordinaten
        """
        height, width = heightmap.shape
        local_minima = []

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_height = heightmap[y, x]

                # Prüfe alle 8 Nachbarn
                is_minimum = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        neighbor_height = heightmap[y + dy, x + dx]
                        if neighbor_height <= current_height:
                            is_minimum = False
                            break

                    if not is_minimum:
                        break

                if is_minimum:
                    local_minima.append((x, y))

        return local_minima

    def apply_jump_flooding(self, heightmap, lake_seeds):
        """
        Funktionsweise: Jump Flooding Algorithm für parallele Senken-Füllung
        Aufgabe: Bestimmt Einzugsgebiete für alle See-Seeds in O(log n) Zeit
        Parameter: heightmap, lake_seeds - Höhendaten und See-Seed-Positionen
        Returns: numpy.ndarray - Lake-ID für jeden Pixel (-1 = kein See)
        """
        height, width = heightmap.shape
        lake_map = np.full((height, width), -1, dtype=np.int32)

        if not lake_seeds:
            return lake_map

        # Initialisierung: Jeder See-Seed markiert sich selbst
        for i, (seed_x, seed_y) in enumerate(lake_seeds):
            lake_map[seed_y, seed_x] = i

        # Jump Flooding mit exponentiell abnehmenden Sprungdistanzen
        max_dim = max(height, width)
        jump_distance = 1
        while jump_distance < max_dim:
            jump_distance *= 2

        while jump_distance >= 1:
            new_lake_map = np.copy(lake_map)

            for y in range(height):
                for x in range(width):
                    current_height = heightmap[y, x]
                    closest_seed = lake_map[y, x]
                    closest_distance = float('inf')

                    # Prüfe alle Jump-Nachbarn
                    for dy in [-jump_distance, 0, jump_distance]:
                        for dx in [-jump_distance, 0, jump_distance]:
                            ny, nx = y + dy, x + dx

                            if 0 <= ny < height and 0 <= nx < width:
                                neighbor_seed = lake_map[ny, nx]

                                if neighbor_seed >= 0:
                                    seed_x, seed_y = lake_seeds[neighbor_seed]
                                    seed_height = heightmap[seed_y, seed_x]

                                    # Kann Wasser zu diesem Seed fließen?
                                    if current_height >= seed_height:
                                        distance = np.sqrt((x - seed_x) ** 2 + (y - seed_y) ** 2)

                                        if distance < closest_distance:
                                            closest_distance = distance
                                            closest_seed = neighbor_seed

                    new_lake_map[y, x] = closest_seed

            lake_map = new_lake_map
            jump_distance //= 2

        return lake_map

    def classify_lake_basins(self, heightmap, lake_map, lake_seeds):
        """
        Funktionsweise: Klassifiziert See-Becken nach Volumen und validiert Threshold
        Aufgabe: Filtert See-Kandidaten nach Mindestvolumen
        Parameter: heightmap, lake_map, lake_seeds - Höhen, Lake-Map und Seeds
        Returns: Tuple (filtered_lake_map, valid_lakes) - Gefilterte Seen
        """
        height, width = heightmap.shape
        filtered_lake_map = np.full((height, width), -1, dtype=np.int32)
        valid_lakes = []

        for lake_id, (seed_x, seed_y) in enumerate(lake_seeds):
            # Berechne Einzugsgebiet-Volumen
            lake_pixels = np.where(lake_map == lake_id)

            if len(lake_pixels[0]) == 0:
                continue

            # Volumen-Berechnung: Differenz zwischen Wasserstand und Terrain
            seed_height = heightmap[seed_y, seed_x]
            total_volume = 0.0

            for py, px in zip(lake_pixels[0], lake_pixels[1]):
                terrain_height = heightmap[py, px]
                if terrain_height <= seed_height:
                    water_depth = seed_height - terrain_height
                    total_volume += water_depth

            # Volume-Threshold prüfen
            if total_volume >= self.lake_volume_threshold:
                # See ist gültig
                for py, px in zip(lake_pixels[0], lake_pixels[1]):
                    filtered_lake_map[py, px] = len(valid_lakes)

                valid_lakes.append({
                    'seed': (seed_x, seed_y),
                    'volume': total_volume,
                    'pixels': len(lake_pixels[0])
                })

        return filtered_lake_map, valid_lakes


class FlowNetworkBuilder:
    """
    Funktionsweise: Baut Flussnetzwerk durch Steepest Descent mit Upstream-Akkumulation
    Aufgabe: Erstellt flow_map und water_biomes_map mit realistischen Flusssystemen
    """

    def __init__(self, rain_threshold=5.0):
        """
        Funktionsweise: Initialisiert Flow-Network-Builder mit Niederschlags-Schwellwert
        Aufgabe: Setup der Flussnetzwerk-Erstellung
        Parameter: rain_threshold (float) - Niederschlagsschwelle für Quellbildung
        """
        self.rain_threshold = rain_threshold

    def calculate_steepest_descent(self, heightmap):
        """
        Funktionsweise: Berechnet Steepest Descent Flow-Richtungen für jeden Pixel
        Aufgabe: Bestimmt optimale Fließrichtung zu steilstem Nachbarn
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: numpy.ndarray - Flow-Richtungen (8-directional encoding)
        """
        height, width = heightmap.shape
        flow_directions = np.zeros((height, width), dtype=np.int8)

        # 8-Richtungs-Encoding: 0=E, 1=SE, 2=S, 3=SW, 4=W, 5=NW, 6=N, 7=NE
        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        for y in range(height):
            for x in range(width):
                current_height = heightmap[y, x]
                steepest_gradient = 0.0
                steepest_direction = -1  # -1 = Senke/keine Richtung

                for direction, (dx, dy) in enumerate(direction_offsets):
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor_height = heightmap[ny, nx]

                        # Gradient berechnen (Höhendifferenz / Distanz)
                        height_diff = current_height - neighbor_height
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        gradient = height_diff / distance

                        if gradient > steepest_gradient:
                            steepest_gradient = gradient
                            steepest_direction = direction

                flow_directions[y, x] = steepest_direction

        return flow_directions

    def accumulate_upstream_flow(self, flow_directions, precip_map, lake_map):
        """
        Funktionsweise: Akkumuliert Upstream-Flow iterativ für Flussnetzwerk-Aufbau
        Aufgabe: Berechnet Wassermengen durch Upstream-Akkumulation
        Parameter: flow_directions, precip_map, lake_map - Flow-Richtungen, Niederschlag, Seen
        Returns: numpy.ndarray - Akkumulierte Flow-Mengen
        """
        height, width = flow_directions.shape
        flow_accumulation = np.zeros((height, width), dtype=np.float32)

        # Initiale Wassermengen: Niederschlag über Schwellwert
        initial_water = np.where(precip_map > self.rain_threshold, precip_map, 0)
        flow_accumulation = np.copy(initial_water)

        # Iterative Akkumulation (bis Konvergenz)
        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        max_iterations = height * width // 10
        for iteration in range(max_iterations):
            new_accumulation = np.copy(flow_accumulation)
            changed = False

            for y in range(height):
                for x in range(width):
                    # Sammle Wasser von allen Zellen die hierher fließen
                    incoming_water = 0.0

                    # Prüfe alle Nachbarn
                    for source_dir, (dx, dy) in enumerate(direction_offsets):
                        sx, sy = x - dx, y - dy  # Source-Position

                        if 0 <= sx < width and 0 <= sy < height:
                            source_flow_dir = flow_directions[sy, sx]

                            # Fließt Source-Zelle zu aktueller Position?
                            if source_flow_dir == source_dir:
                                incoming_water += flow_accumulation[sy, sx]

                    # See-Handling: Seen akkumulieren Wasser ohne Weiterleitung
                    if lake_map[y, x] >= 0:
                        new_accumulation[y, x] = initial_water[y, x] + incoming_water
                    else:
                        new_accumulation[y, x] = initial_water[y, x] + incoming_water

                    if abs(new_accumulation[y, x] - flow_accumulation[y, x]) > 0.01:
                        changed = True

            flow_accumulation = new_accumulation

            if not changed:
                break

        return flow_accumulation

    def classify_water_bodies(self, flow_accumulation, lake_map):
        """
        Funktionsweise: Klassifiziert Wasserkörper basierend auf Flussgröße
        Aufgabe: Erstellt water_biomes_map mit verschiedenen Gewässertypen
        Parameter: flow_accumulation, lake_map - Flow-Mengen und Seen
        Returns: numpy.ndarray - Water-Biomes (0=kein Wasser, 1=Creek, 2=River, 3=Grand River, 4=Lake)
        """
        height, width = flow_accumulation.shape
        water_biomes_map = np.zeros((height, width), dtype=np.uint8)

        # Seen zuerst markieren
        water_biomes_map[lake_map >= 0] = 4  # Lake

        # Flow-basierte Klassifikation
        for y in range(height):
            for x in range(width):
                if lake_map[y, x] >= 0:
                    continue  # Bereits als See markiert

                flow_amount = flow_accumulation[y, x]

                if flow_amount >= 100.0:  # Grand River
                    water_biomes_map[y, x] = 3
                elif flow_amount >= 20.0:  # River
                    water_biomes_map[y, x] = 2
                elif flow_amount >= 5.0:  # Creek
                    water_biomes_map[y, x] = 1
                # else: 0 = kein Wasser

        return water_biomes_map


class ManningFlowCalculator:
    """
    Funktionsweise: Berechnet Strömung nach Manning-Gleichung mit adaptiven Querschnitten
    Aufgabe: Erstellt flow_speed und cross_section für realistische Fließgeschwindigkeiten
    """

    def __init__(self, manning_coefficient=0.03):
        """
        Funktionsweise: Initialisiert Manning-Flow-Calculator mit Rauheitskoeffizient
        Aufgabe: Setup der Manning-Gleichungs-Berechnung
        Parameter: manning_coefficient (float) - Rauheitskoeffizient für Fließgeschwindigkeit
        """
        self.manning_n = manning_coefficient

    def solve_manning_equation(self, flow_accumulation, slopemap, heightmap):
        """
        Funktionsweise: Löst Manning-Gleichung für Fließgeschwindigkeit v = (1/n) * R^(2/3) * S^(1/2)
        Aufgabe: Berechnet realistische Fließgeschwindigkeiten
        Parameter: flow_accumulation, slopemap, heightmap - Flow-Mengen, Slopes, Höhen
        Returns: Tuple (flow_speed, cross_section) - Geschwindigkeit und Querschnitt
        """
        height, width = flow_accumulation.shape
        flow_speed = np.zeros((height, width), dtype=np.float32)
        cross_section = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                flow_rate = flow_accumulation[y, x]  # m³/s

                if flow_rate < 1.0:  # Zu wenig Wasser für Manning-Berechnung
                    continue

                # Slope-Magnitude berechnen
                slope_x = slopemap[y, x, 0] if x < slopemap.shape[1] else 0
                slope_y = slopemap[y, x, 1] if y < slopemap.shape[0] else 0
                slope = np.sqrt(slope_x ** 2 + slope_y ** 2)
                slope = max(0.001, slope)  # Minimum-Slope für Berechnung

                # Optimaler Querschnitt durch iterative Lösung
                optimal_area, hydraulic_radius = self.optimize_channel_geometry(
                    flow_rate, slope, heightmap, x, y
                )

                # Manning-Geschwindigkeit berechnen
                if hydraulic_radius > 0:
                    velocity = (1.0 / self.manning_n) * (hydraulic_radius ** (2.0 / 3.0)) * (slope ** 0.5)

                    # Kontinuitätsgleichung: Q = A * v
                    if velocity > 0:
                        required_area = flow_rate / velocity
                        cross_section[y, x] = required_area
                        flow_speed[y, x] = velocity

        return flow_speed, cross_section

    def optimize_channel_geometry(self, flow_rate, slope, heightmap, x, y):
        """
        Funktionsweise: Optimiert Kanal-Geometrie für gegebene Bedingungen
        Aufgabe: Bestimmt optimale Breite-zu-Tiefe-Verhältnis basierend auf Geländeform
        Parameter: flow_rate, slope, heightmap, x, y - Flow-Rate, Slope und Position
        Returns: Tuple (area, hydraulic_radius) - Optimaler Querschnitt
        """
        # Tal-Breite analysieren
        valley_width = self._analyze_valley_width(heightmap, x, y)

        # Breite-zu-Tiefe-Verhältnis basierend auf Tal-Konfinierung
        if valley_width < 5:  # Enges Tal
            width_to_depth_ratio = 2.0  # Tiefe, schmale Flüsse
        elif valley_width < 20:  # Mittleres Tal
            width_to_depth_ratio = 5.0
        else:  # Weites Tal/Ebene
            width_to_depth_ratio = 10.0  # Breite, flache Flüsse

        # Iterative Optimierung für hydraulisch optimale Form
        best_area = 0
        best_hydraulic_radius = 0

        for depth in np.linspace(0.1, 5.0, 20):
            width = depth * width_to_depth_ratio
            area = width * depth
            wetted_perimeter = width + 2 * depth
            hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0

            # Manning-Geschwindigkeit für diese Geometrie
            velocity = (1.0 / self.manning_n) * (hydraulic_radius ** (2.0 / 3.0)) * (slope ** 0.5)
            theoretical_flow = area * velocity

            # Optimum: Geometrie die am besten zu gegebenem Flow passt
            if abs(theoretical_flow - flow_rate) < abs(best_area * velocity - flow_rate):
                best_area = area
                best_hydraulic_radius = hydraulic_radius

        return best_area, best_hydraulic_radius

    def calculate_hydraulic_radius(self, width, depth):
        """
        Funktionsweise: Berechnet hydraulischen Radius R = A / P
        Aufgabe: Standard hydraulischer Radius für rechteckigen Querschnitt
        Parameter: width, depth - Breite und Tiefe des Kanals
        Returns: float - Hydraulischer Radius
        """
        area = width * depth
        wetted_perimeter = width + 2 * depth

        if wetted_perimeter > 0:
            return area / wetted_perimeter
        else:
            return 0.0

    def _analyze_valley_width(self, heightmap, center_x, center_y):
        """
        Funktionsweise: Analysiert Tal-Breite durch Suche nach Geländeanstiegen
        Aufgabe: Bestimmt Konfinierung des Flusses durch Topographie
        """
        height, width = heightmap.shape

        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            return 10.0  # Default-Breite

        center_height = heightmap[center_y, center_x]
        height_threshold = center_height + 20.0  # 20m Anstieg als Tal-Grenze

        # Suche in alle Richtungen nach Tal-Rändern
        max_distance = 0

        for angle in np.linspace(0, 2 * np.pi, 16):  # 16 Richtungen
            dx = np.cos(angle)
            dy = np.sin(angle)

            distance = 0
            for step in range(1, 50):  # Max 50 Pixel Suchradius
                test_x = center_x + int(dx * step)
                test_y = center_y + int(dy * step)

                if test_x < 0 or test_x >= width or test_y < 0 or test_y >= height:
                    break

                test_height = heightmap[test_y, test_x]
                if test_height > height_threshold:
                    distance = step
                    break

            max_distance = max(max_distance, distance)

        return max_distance * 2  # Tal-Breite = 2 * Radius


class SoilMoistureCalculator:
    """
    Funktionsweise: Berechnet Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
    Aufgabe: Erstellt soil_moist_map für Biome-System und Weather-Evaporation
    """

    def __init__(self, diffusion_radius=5.0):
        """
        Funktionsweise: Initialisiert Soil-Moisture-Calculator mit Diffusions-Radius
        Aufgabe: Setup der Bodenfeuchtigkeit-Berechnung
        Parameter: diffusion_radius (float) - Ausbreitungsradius der Bodenfeuchtigkeit
        """
        self.diffusion_radius = diffusion_radius

    def apply_gaussian_diffusion(self, water_biomes_map, flow_accumulation):
        """
        Funktionsweise: Wendet Gaussian-Diffusion von Gewässern in Umgebung an
        Aufgabe: Simuliert kapillare Ausbreitung und Grundwasser-Effekte
        Parameter: water_biomes_map, flow_accumulation - Wasser-Klassifikation und Flow-Mengen
        Returns: numpy.ndarray - Bodenfeuchtigkeit in %
        """
        height, width = water_biomes_map.shape
        soil_moisture = np.zeros((height, width), dtype=np.float32)

        # Direkte Wasserpräsenz: maximale Feuchtigkeit
        water_mask = water_biomes_map > 0
        soil_moisture[water_mask] = 100.0

        # Kapillare Ausbreitung (enger Filter)
        capillary_source = np.zeros_like(soil_moisture)
        capillary_source[water_mask] = 100.0
        capillary_moisture = gaussian_filter(capillary_source, sigma=2.0)

        # Grundwasser-Effekte (weiter Filter)
        groundwater_source = np.zeros_like(soil_moisture)

        # Gewichte basierend auf Wasser-Typ
        for y in range(height):
            for x in range(width):
                water_type = water_biomes_map[y, x]
                flow_amount = flow_accumulation[y, x]

                if water_type == 4:  # Lake
                    groundwater_source[y, x] = 80.0
                elif water_type == 3:  # Grand River
                    groundwater_source[y, x] = 60.0 + min(20.0, flow_amount * 0.1)
                elif water_type == 2:  # River
                    groundwater_source[y, x] = 40.0 + min(20.0, flow_amount * 0.2)
                elif water_type == 1:  # Creek
                    groundwater_source[y, x] = 20.0 + min(10.0, flow_amount * 0.3)

        groundwater_moisture = gaussian_filter(groundwater_source, sigma=self.diffusion_radius)

        # Kombiniere beide Effekte (Maximum)
        combined_moisture = np.maximum(capillary_moisture, groundwater_moisture)

        # Direkte Wasserpräsenz überschreibt alles
        combined_moisture[water_mask] = 100.0

        return combined_moisture

def calculate_groundwater_effects(self, heightmap, water_biomes_map):
        """
        Funktionsweise: Berechnet Grundwasser-Effekte basierend auf Topographie
        Aufgabe: Simuliert Grundwasser-Strömung in niedrigere Bereiche
        Parameter: heightmap, water_biomes_map - Höhen und Wasser-Klassifikation
        Returns: numpy.ndarray - Grundwasser-Beitrag zur Bodenfeuchtigkeit
        """
        height, width = heightmap.shape
        groundwater_effect = np.zeros((height, width), dtype=np.float32)
        
        # Finde alle Wasser-Quellen
        water_sources = np.where(water_biomes_map > 0)
        
        for water_y, water_x in zip(water_sources[0], water_sources[1]):
            water_height = heightmap[water_y, water_x]
            water_type = water_biomes_map[water_y, water_x]
            
            # Grundwasser-Stärke basierend auf Wasser-Typ
            if water_type == 4:  # Lake
                base_strength = 50.0
            elif water_type == 3:  # Grand River
                base_strength = 30.0
            elif water_type == 2:  # River
                base_strength = 20.0
            else:  # Creek
                base_strength = 10.0
            
            # Ausbreitung in niedrigere Bereiche
            for y in range(max(0, water_y - 20), min(height, water_y + 21)):
                for x in range(max(0, water_x - 20), min(width, water_x + 21)):
                    target_height = heightmap[y, x]
                    
                    # Nur wenn Ziel niedriger liegt
                    if target_height <= water_height:
                        distance = np.sqrt((x - water_x)**2 + (y - water_y)**2)
                        height_diff = water_height - target_height
                        
                        if distance > 0:
                            # Grundwasser-Intensität durch Distanz und Höhendifferenz
                            intensity = base_strength * (height_diff / max(1.0, distance))
                            intensity = min(intensity, base_strength)  # Cap maximum
                            
                            groundwater_effect[y, x] = max(groundwater_effect[y, x], intensity)
        
        return groundwater_effect
    
def integrate_moisture_sources(self, base_moisture, precipitation_moisture, groundwater_moisture):
    """
    Funktionsweise: Integriert alle Feuchtigkeits-Quellen zu finaler soil_moist_map
    Aufgabe: Kombiniert Wasser-Diffusion, Niederschlag und Grundwasser
    Parameter: base_moisture, precipitation_moisture, groundwater_moisture - Verschiedene Feuchtigkeits-Quellen
    Returns: numpy.ndarray - Finale Bodenfeuchtigkeit
    """
    # Kombiniere alle Quellen (additiv bis Maximum 100%)
    total_moisture = base_moisture + precipitation_moisture * 0.1 + groundwater_moisture

    # Begrenze auf [0, 100]%
    total_moisture = np.clip(total_moisture, 0, 100)

    return total_moisture


class EvaporationCalculator:
    """
    Funktionsweise: Berechnet Evaporation nach atmosphärischen Bedingungen
    Aufgabe: Erstellt evaporation_map basierend auf temp_map, humid_map und wind_map
    """
    
    def __init__(self, evaporation_base_rate=0.002):
        """
        Funktionsweise: Initialisiert Evaporation-Calculator mit Basis-Verdunstungsrate
        Aufgabe: Setup der Evaporations-Berechnung
        Parameter: evaporation_base_rate (float) - Basis-Verdunstungsrate in m/Tag
        """
        self.base_rate = evaporation_base_rate
    
    def calculate_atmospheric_evaporation(self, temp_map, humid_map, wind_map, water_biomes_map):
        """
        Funktionsweise: Berechnet Evaporation basierend auf atmosphärischen Sättigungseffekten
        Aufgabe: Realistische Verdunstung abhängig von Luftfeuchtigkeit, Temperatur und Wind
        Parameter: temp_map, humid_map, wind_map, water_biomes_map - Atmosphärische Bedingungen und Wasser
        Returns: numpy.ndarray - Evaporation in gH2O/m²/Tag
        """
        height, width = temp_map.shape
        evaporation_map = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                # Nur an Wasseroberflächen
                if water_biomes_map[y, x] == 0:
                    continue
                
                temperature = temp_map[y, x]
                humidity = humid_map[y, x]
                wind_speed = np.sqrt(wind_map[y, x, 0]**2 + wind_map[y, x, 1]**2)
                
                # Maximale Wasserdampfdichte (Magnus-Formel)
                max_vapor_density = 5.0 * np.exp(0.06 * temperature)
                
                # Aktuelle relative Feuchtigkeit
                if max_vapor_density > 0:
                    relative_humidity = humidity / max_vapor_density
                    relative_humidity = min(1.0, relative_humidity)
                else:
                    relative_humidity = 1.0
                
                # Evaporation reduziert sich mit steigender Luftfeuchtigkeit
                humidity_factor = 1.0 - relative_humidity
                
                # Temperatur-Faktor (exponentiell mit Temperatur)
                temp_factor = np.exp(temperature / 20.0) if temperature > 0 else 0.1
                
                # Wind-Faktor (lineare Verstärkung)
                wind_factor = 1.0 + wind_speed * 0.2
                
                # Basis-Evaporation
                evaporation_rate = self.base_rate * humidity_factor * temp_factor * wind_factor * 1000  # Conversion zu gH2O/m²/Tag
                
                evaporation_map[y, x] = evaporation_rate
        
        return evaporation_map
    
    def apply_wind_effects(self, base_evaporation, wind_map):
        """
        Funktionsweise: Wendet Wind-Verstärkung auf Evaporation an
        Aufgabe: Wind beschleunigt Dampftransport von Wasseroberfläche
        Parameter: base_evaporation, wind_map - Basis-Evaporation und Wind-Daten
        Returns: numpy.ndarray - Wind-verstärkte Evaporation
        """
        height, width = base_evaporation.shape
        wind_enhanced_evap = np.copy(base_evaporation)
        
        for y in range(height):
            for x in range(width):
                if base_evaporation[y, x] > 0:
                    wind_speed = np.sqrt(wind_map[y, x, 0]**2 + wind_map[y, x, 1]**2)
                    
                    # Wind-Verstärkung: +20% pro m/s Wind
                    wind_enhancement = 1.0 + wind_speed * 0.2
                    wind_enhanced_evap[y, x] *= wind_enhancement
        
        return wind_enhanced_evap
    
    def limit_by_available_water(self, evaporation_map, water_biomes_map, flow_accumulation):
        """
        Funktionsweise: Begrenzt Evaporation durch verfügbare Wasseroberfläche
        Aufgabe: Physikalische Begrenzung der Verdunstung durch Wasserverfügbarkeit
        Parameter: evaporation_map, water_biomes_map, flow_accumulation - Evaporation, Wasser-Typ, Flow-Mengen
        Returns: numpy.ndarray - Begrenzte Evaporation
        """
        height, width = evaporation_map.shape
        limited_evaporation = np.copy(evaporation_map)
        
        for y in range(height):
            for x in range(width):
                water_type = water_biomes_map[y, x]
                potential_evap = evaporation_map[y, x]
                
                if water_type == 0:  # Kein Wasser
                    limited_evaporation[y, x] = 0.0
                elif water_type == 1:  # Creek
                    # Kleine Gewässer: begrenzte Oberfläche
                    max_evap = min(potential_evap, 50.0)
                    limited_evaporation[y, x] = max_evap
                elif water_type == 2:  # River
                    max_evap = min(potential_evap, 100.0)
                    limited_evaporation[y, x] = max_evap
                elif water_type == 3:  # Grand River
                    max_evap = min(potential_evap, 200.0)
                    limited_evaporation[y, x] = max_evap
                else:  # Lake
                    # Seen: große Oberfläche, keine Begrenzung
                    limited_evaporation[y, x] = potential_evap
        
        return limited_evaporation


class ErosionSedimentationSystem:
    """
    Funktionsweise: Simuliert Stream Power Erosion mit Hjulström-Sundborg Transport
    Aufgabe: Modifiziert heightmap und rock_map durch realistische Erosions-/Sedimentationsprozesse
    """
    
    def __init__(self, erosion_strength=1.0, sediment_capacity_factor=0.1, settling_velocity=0.01):
        """
        Funktionsweise: Initialisiert Erosion-Sedimentation-System mit Erosions-Parametern
        Aufgabe: Setup der Erosions- und Sedimentations-Berechnung
        Parameter: erosion_strength, sediment_capacity_factor, settling_velocity - Erosions-Parameter
        """
        self.erosion_strength = erosion_strength
        self.capacity_factor = sediment_capacity_factor
        self.settling_velocity = settling_velocity
    
    def calculate_stream_power(self, flow_accumulation, flow_speed, slopemap, hardness_map):
        """
        Funktionsweise: Berechnet Stream Power Erosion E = K * (τ - τc)
        Aufgabe: Erosionsrate basierend auf Scherspannung und Gesteinshärte
        Parameter: flow_accumulation, flow_speed, slopemap, hardness_map - Flow-Daten und Gesteinshärte
        Returns: numpy.ndarray - Erosionsrate in m/Jahr
        """
        height, width = flow_accumulation.shape
        erosion_map = np.zeros((height, width), dtype=np.float32)
        
        # Konstanten
        rho_water = 1000.0  # kg/m³
        gravity = 9.81      # m/s²
        
        for y in range(height):
            for x in range(width):
                flow_rate = flow_accumulation[y, x]
                velocity = flow_speed[y, x]
                
                if flow_rate < 1.0 or velocity < 0.1:  # Zu wenig Wasser/Geschwindigkeit
                    continue
                
                # Scherspannung τ = ρ * g * h * S
                slope_x = slopemap[y, x, 0] if x < slopemap.shape[1] else 0
                slope_y = slopemap[y, x, 1] if y < slopemap.shape[0] else 0
                slope = np.sqrt(slope_x**2 + slope_y**2)
                
                # Approximiere Wassertiefe aus Flow-Rate und Geschwindigkeit
                if velocity > 0:
                    water_depth = flow_rate / (velocity * 10.0)  # Angenommene Breite: 10m
                else:
                    water_depth = 0
                
                # Scherspannung
                shear_stress = rho_water * gravity * water_depth * slope
                
                # Kritische Scherspannung basierend auf Gesteinshärte
                hardness = hardness_map[y, x] if x < hardness_map.shape[1] and y < hardness_map.shape[0] else 50.0
                critical_shear = hardness * 10.0  # Pa (empirisch)
                
                # Erosion nur wenn Scherspannung kritischen Wert überschreitet
                if shear_stress > critical_shear:
                    excess_stress = shear_stress - critical_shear
                    
                    # Stream Power Erosion: E = K * (τ - τc) * v²
                    erosion_rate = self.erosion_strength * excess_stress * (velocity**2) / hardness
                    
                    # Begrenzung auf realistische Werte (max 0.1 m/Jahr)
                    erosion_rate = min(erosion_rate * 1e-6, 0.1)  # Skalierung
                    
                    erosion_map[y, x] = erosion_rate
        
        return erosion_map
    
    def transport_sediment(self, erosion_map, flow_speed, flow_directions, flow_accumulation):
        """
        Funktionsweise: Transportiert Sediment entlang Fließrichtungen mit Hjulström-Diagramm
        Aufgabe: Sediment-Transport und Kapazitäts-basierte Ablagerung
        Parameter: erosion_map, flow_speed, flow_directions, flow_accumulation - Erosion und Flow-Daten
        Returns: numpy.ndarray - Sedimentationsrate in m/Jahr
        """
        height, width = erosion_map.shape
        sedimentation_map = np.zeros((height, width), dtype=np.float32)
        sediment_load = np.zeros((height, width), dtype=np.float32)
        
        # Transport-Kapazität berechnen (Hjulström-Diagramm vereinfacht)
        transport_capacity = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                velocity = flow_speed[y, x]
                
                # Transportkapazität ∝ v^2.5 (nach Hjulström)
                if velocity > 0.1:  # Mindestgeschwindigkeit für Transport
                    transport_capacity[y, x] = self.capacity_factor * (velocity ** 2.5)
        
        # Upstream-zu-Downstream Transport simulation
        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        
        # Mehrere Iterationen für Sediment-Transport
        for iteration in range(10):
            new_sediment_load = np.copy(sediment_load)
            
            for y in range(height):
                for x in range(width):
                    # Erosion fügt Sediment hinzu
                    new_sediment_load[y, x] += erosion_map[y, x]
                    
                    # Transport zu Downstream-Zelle
                    flow_dir = flow_directions[y, x]
                    
                    if 0 <= flow_dir < 8:
                        dx, dy = direction_offsets[flow_dir]
                        nx, ny = x + dx, y + dy
                        
                        if 0 <= nx < width and 0 <= ny < height:
                            # Transport-Effizienz basierend auf Geschwindigkeit
                            velocity = flow_speed[y, x]
                            if velocity > 0.1:
                                transport_efficiency = min(1.0, velocity / 2.0)  # 100% bei 2 m/s
                                
                                transported_sediment = new_sediment_load[y, x] * transport_efficiency
                                new_sediment_load[y, x] -= transported_sediment
                                new_sediment_load[ny, nx] += transported_sediment
                    
                    # Sedimentation wenn Transportkapazität überschritten
                    current_load = new_sediment_load[y, x]
                    capacity = transport_capacity[y, x]
                    
                    if current_load > capacity:
                        excess_sediment = current_load - capacity
                        settling_rate = excess_sediment * self.settling_velocity
                        
                        sedimentation_map[y, x] += settling_rate
                        new_sediment_load[y, x] -= settling_rate
            
            sediment_load = new_sediment_load
        
        return sedimentation_map
    
    def apply_mass_conservation(self, rock_map, erosion_map, sedimentation_map):
        """
        Funktionsweise: Wendet Massenerhaltung auf rock_map an (R+G+B=255)
        Aufgabe: Erhält Gesteinsmassen-Verhältnisse bei Erosion/Sedimentation
        Parameter: rock_map, erosion_map, sedimentation_map - Gesteine und Erosions-/Sedimentations-Änderungen
        Returns: numpy.ndarray - Aktualisierte rock_map mit Massenerhaltung
        """
        height, width = rock_map.shape[:2]
        updated_rock_map = np.copy(rock_map).astype(np.float32)
        
        for y in range(height):
            for x in range(width):
                erosion_rate = erosion_map[y, x]
                sedimentation_rate = sedimentation_map[y, x]
                
                # Net-Änderung: Sedimentation - Erosion
                net_change = sedimentation_rate - erosion_rate
                
                if net_change != 0:
                    # Aktuelle Gesteinsverhältnisse
                    current_rocks = updated_rock_map[y, x, :].copy()
                    total_mass = np.sum(current_rocks)
                    
                    if total_mass > 0:
                        # Verhältnisse erhalten
                        ratios = current_rocks / total_mass
                        
                        # Änderung proportional anwenden
                        change_amount = net_change * 10.0  # Skalierung für sichtbare Effekte
                        updated_rocks = current_rocks + ratios * change_amount
                        
                        # Negative Werte verhindern
                        updated_rocks = np.maximum(updated_rocks, 1.0)
                        
                        # Re-normalisierung auf 255
                        new_total = np.sum(updated_rocks)
                        if new_total > 0:
                            normalized_rocks = (updated_rocks / new_total) * 255
                            updated_rock_map[y, x, :] = normalized_rocks
        
        return updated_rock_map.astype(np.uint8)


class HydrologySystemGenerator:
    """
    Funktionsweise: Hauptklasse für dynamisches Hydrologiesystem mit Erosion und Sedimentation
    Aufgabe: Koordiniert alle hydrologischen Prozesse und Massentransport
    """
    
    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Hydrologie-System mit allen Sub-Komponenten
        Aufgabe: Setup aller hydrologischen Systeme
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Hydrologie
        """
        self.map_seed = map_seed
        
        # Standard-Parameter
        self.lake_volume_threshold = 0.1
        self.rain_threshold = 5.0
        self.manning_coefficient = 0.03
        self.erosion_strength = 1.0
        self.sediment_capacity_factor = 0.1
        self.evaporation_base_rate = 0.002
        self.diffusion_radius = 5.0
        self.settling_velocity = 0.01
    
    def generate_hydrology_system(self, heightmap, slopemap, hardness_map, rock_map, precip_map, temp_map, wind_map, humid_map, lake_volume_threshold, rain_threshold, manning_coefficient, erosion_strength, sediment_capacity_factor, evaporation_base_rate, diffusion_radius, settling_velocity, map_seed):
        """
        Funktionsweise: Generiert komplettes Hydrologie-System mit allen Komponenten
        Aufgabe: Hauptfunktion für Hydrologie-Generierung mit allen Parametern
        Parameter: heightmap, slopemap, hardness_map, rock_map, precip_map, temp_map, wind_map, humid_map, lake_volume_threshold, rain_threshold, manning_coefficient, erosion_strength, sediment_capacity_factor, evaporation_base_rate, diffusion_radius, settling_velocity, map_seed
        Returns: Tuple - Alle Hydrologie-Outputs
        """
        # Parameter aktualisieren
        self.lake_volume_threshold = lake_volume_threshold
        self.rain_threshold = rain_threshold
        self.manning_coefficient = manning_coefficient
        self.erosion_strength = erosion_strength
        self.sediment_capacity_factor = sediment_capacity_factor
        self.evaporation_base_rate = evaporation_base_rate
        self.diffusion_radius = diffusion_radius
        self.settling_velocity = settling_velocity
        
        # Schritt 1: Lake-Detection
        lake_system = LakeDetectionSystem(lake_volume_threshold)
        lake_seeds = lake_system.detect_local_minima(heightmap)
        lake_map = lake_system.apply_jump_flooding(heightmap, lake_seeds)
        filtered_lake_map, valid_lakes = lake_system.classify_lake_basins(heightmap, lake_map, lake_seeds)
        
        # Schritt 2: Flow-Network
        flow_builder = FlowNetworkBuilder(rain_threshold)
        flow_directions = flow_builder.calculate_steepest_descent(heightmap)
        flow_accumulation = flow_builder.accumulate_upstream_flow(flow_directions, precip_map, filtered_lake_map)
        water_biomes_map = flow_builder.classify_water_bodies(flow_accumulation, filtered_lake_map)
        
        # Schritt 3: Manning-Flow
        manning_calc = ManningFlowCalculator(manning_coefficient)
        flow_speed, cross_section = manning_calc.solve_manning_equation(flow_accumulation, slopemap, heightmap)
        
        # Schritt 4: Erosion-Sedimentation
        erosion_system = ErosionSedimentationSystem(erosion_strength, sediment_capacity_factor, settling_velocity)
        erosion_map = erosion_system.calculate_stream_power(flow_accumulation, flow_speed, slopemap, hardness_map)
        sedimentation_map = erosion_system.transport_sediment(erosion_map, flow_speed, flow_directions, flow_accumulation)
        rock_map_updated = erosion_system.apply_mass_conservation(rock_map, erosion_map, sedimentation_map)
        
        # Schritt 5: Soil-Moisture
        moisture_calc = SoilMoistureCalculator(diffusion_radius)
        base_soil_moisture = moisture_calc.apply_gaussian_diffusion(water_biomes_map, flow_accumulation)
        groundwater_moisture = moisture_calc.calculate_groundwater_effects(heightmap, water_biomes_map)
        precipitation_moisture = precip_map * 0.5  # Niederschlag trägt zur Bodenfeuchtigkeit bei
        soil_moist_map = moisture_calc.integrate_moisture_sources(base_soil_moisture, precipitation_moisture, groundwater_moisture)
        
        # Schritt 6: Evaporation
        evap_calc = EvaporationCalculator(evaporation_base_rate)
        base_evaporation = evap_calc.calculate_atmospheric_evaporation(temp_map, humid_map, wind_map, water_biomes_map)
        wind_enhanced_evap = evap_calc.apply_wind_effects(base_evaporation, wind_map)
        evaporation_map = evap_calc.limit_by_available_water(wind_enhanced_evap, water_biomes_map, flow_accumulation)
        
        # Schritt 7: Water-Map und Ocean-Outflow
        water_map = self._create_water_depth_map(water_biomes_map, flow_accumulation, cross_section)
        ocean_outflow = self._calculate_ocean_outflow(flow_accumulation, flow_directions, heightmap.shape)
        
        return (water_map, flow_accumulation, flow_speed, cross_section, soil_moist_map, 
                erosion_map, sedimentation_map, rock_map_updated, evaporation_map, 
                ocean_outflow, water_biomes_map)
    
    def _create_water_depth_map(self, water_biomes_map, flow_accumulation, cross_section):
        """
        Funktionsweise: Erstellt Wasser-Tiefen-Map aus Flow-Daten
        Aufgabe: Konvertiert Flow-Accumulation zu Wassertiefen
        """
        height, width = water_biomes_map.shape
        water_map = np.zeros((height, width), dtype=np.float32)
        
        for y in range(height):
            for x in range(width):
                water_type = water_biomes_map[y, x]
                
                if water_type == 0:  # Kein Wasser
                    continue
                elif water_type == 4:  # Lake
                    # See-Tiefe basierend auf Akkumulation
                    depth = min(5.0, flow_accumulation[y, x] * 0.01)
                    water_map[y, x] = depth
                else:  # Flüsse
                    # Fluss-Tiefe aus Querschnitt ableiten
                    area = cross_section[y, x] if x < cross_section.shape[1] and y < cross_section.shape[0] else 0
                    
                    if area > 0:
                        # Approximierte Tiefe (angenommene Breite: 10m für Flüsse)
                        estimated_depth = area / 10.0
                        water_map[y, x] = min(3.0, estimated_depth)
        
        return water_map
    
    def _calculate_ocean_outflow(self, flow_accumulation, flow_directions, map_shape):
        """
        Funktionsweise: Berechnet Wasser-Abfluss ins Meer (an Kartenrändern)
        Aufgabe: Summiert allen Wasser-Abfluss der die Karte verlässt
        """
        height, width = map_shape
        total_outflow = 0.0
        
        # Prüfe alle Rand-Pixel
        for y in range(height):
            for x in range(width):
                # Ist Pixel am Rand?
                is_edge = (x == 0 or x == width - 1 or y == 0 or y == height - 1)
                
                if is_edge:
                    flow_dir = flow_directions[y, x]
                    
                    # Fließt Wasser über den Rand hinaus?
                    if flow_dir >= 0:
                        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
                        dx, dy = direction_offsets[flow_dir]
                        target_x, target_y = x + dx, y + dy
                        
                        # Ziel außerhalb der Karte?
                        if target_x < 0 or target_x >= width or target_y < 0 or target_y >= height:
                            total_outflow += flow_accumulation[y, x]
        
        return total_outflow
    
    def simulate_water_cycle(self, current_hydrology, time_step=1.0):
        """
        Funktionsweise: Simuliert zeitliche Entwicklung des Wasser-Kreislaufs
        Aufgabe: Dynamische Updates für Animation/Simulation
        Parameter: current_hydrology, time_step - Aktuelle Hydrologie-Daten und Zeitschritt
        Returns: Tuple - Aktualisierte Hydrologie-Daten
        """
        # Vereinfachte zeitliche Evolution
        (water_map, flow_map, flow_speed, cross_section, soil_moist_map, 
         erosion_map, sedimentation_map, rock_map_updated, evaporation_map, 
         ocean_outflow, water_biomes_map) = current_hydrology
        
        # Bodenfeuchtigkeit durch Evaporation reduzieren
        evap_loss = evaporation_map * time_step * 0.1
        new_soil_moist = np.maximum(0, soil_moist_map - evap_loss)
        
        # Erosion akkumulieren (sehr langsam)
        accumulated_erosion = erosion_map * time_step * 0.001
        
        return (water_map, flow_map, flow_speed, cross_section, new_soil_moist, 
                accumulated_erosion, sedimentation_map, rock_map_updated, evaporation_map, 
                ocean_outflow, water_biomes_map)
    
    def update_erosion_sedimentation(self, heightmap, rock_map, erosion_map, sedimentation_map, time_step=1.0):
        """
        Funktionsweise: Aktualisiert Heightmap und Rock-Map durch Erosion/Sedimentation
        Aufgabe: Anwendung der Erosions-/Sedimentations-Effekte auf Terrain
        Parameter: heightmap, rock_map, erosion_map, sedimentation_map, time_step
        Returns: Tuple (new_heightmap, new_rock_map) - Modifiziertes Terrain
        """
        # Heightmap durch Erosion/Sedimentation modifizieren
        net_height_change = (sedimentation_map - erosion_map) * time_step * 0.1
        new_heightmap = heightmap + net_height_change
        
        # Rock-Map durch Massenerhaltung aktualisieren
        erosion_system = ErosionSedimentationSystem(self.erosion_strength, self.sediment_capacity_factor, self.settling_velocity)
        new_rock_map = erosion_system.apply_mass_conservation(rock_map, erosion_map, sedimentation_map)
        
        return new_heightmap, new_rock_map
    
    def get_hydrology_statistics(self, hydrology_data):
        """
        Funktionsweise: Berechnet Statistiken über generierte Hydrologie-Daten
        Aufgabe: Analyse-Funktionen für Hydrologie-System-Debugging
        Parameter: hydrology_data - Tuple aller Hydrologie-Arrays
        Returns: dict - Hydrologie-Statistiken
        """
        (water_map, flow_map, flow_speed, cross_section, soil_moist_map, 
         erosion_map, sedimentation_map, rock_map_updated, evaporation_map, 
         ocean_outflow, water_biomes_map) = hydrology_data
        
        # Wasser-Klassifikations-Statistiken
        water_types, type_counts = np.unique(water_biomes_map, return_counts=True)
        water_classification = dict(zip(water_types, type_counts))
        
        # Flow-Geschwindigkeiten
        active_flow_mask = flow_speed > 0
        
        stats = {
            'water_coverage': {
                'total_water_pixels': int(np.sum(water_biomes_map > 0)),
                'lakes': int(water_classification.get(4, 0)),
                'grand_rivers': int(water_classification.get(3, 0)),
                'rivers': int(water_classification.get(2, 0)),
                'creeks': int(water_classification.get(1, 0)),
                'dry_land': int(water_classification.get(0, 0))
            },
            'flow_dynamics': {
                'max_flow_rate': float(np.max(flow_map)),
                'total_flow_volume': float(np.sum(flow_map)),
                'max_flow_speed': float(np.max(flow_speed)),
                'avg_flow_speed': float(np.mean(flow_speed[active_flow_mask])) if np.any(active_flow_mask) else 0.0,
                'ocean_outflow': float(ocean_outflow)
            },
            'erosion_sedimentation': {
                'total_erosion': float(np.sum(erosion_map)),
                'total_sedimentation': float(np.sum(sedimentation_map)),
                'max_erosion_rate': float(np.max(erosion_map)),
                'max_sedimentation_rate': float(np.max(sedimentation_map)),
                'net_terrain_change': float(np.sum(sedimentation_map) - np.sum(erosion_map))
            },
            'moisture_evaporation': {
                'avg_soil_moisture': float(np.mean(soil_moist_map)),
                'max_soil_moisture': float(np.max(soil_moist_map)),
                'total_evaporation': float(np.sum(evaporation_map)),
                'max_evaporation_rate': float(np.max(evaporation_map))
            },
            'rock_composition': {
                'sedimentary_percent': float(np.mean(rock_map_updated[:, :, 0]) / 255 * 100),
                'igneous_percent': float(np.mean(rock_map_updated[:, :, 1]) / 255 * 100),
                'metamorphic_percent': float(np.mean(rock_map_updated[:, :, 2]) / 255 * 100)
            }
        }
        
        return stats
    
    def update_seed(self, new_seed):
        """
        Funktionsweise: Aktualisiert Seed für alle Zufalls-Komponenten
        Aufgabe: Ermöglicht Seed-Änderung ohne Neuinstanziierung
        Parameter: new_seed (int) - Neuer Seed für Hydrologie-Generierung
        """
        if new_seed != self.map_seed:
            self.map_seed = new_seed
    
    def validate_mass_conservation(self, rock_map_original, rock_map_updated):
        """
        Funktionsweise: Validiert Massenerhaltung in der Rock-Map
        Aufgabe: Überprüft ob R+G+B=255 bei allen Pixeln erhalten bleibt
        Parameter: rock_map_original, rock_map_updated - Original und aktualisierte Rock-Map
        Returns: dict - Validierungs-Ergebnisse
        """
        # Prüfe Summen
        original_sums = np.sum(rock_map_original, axis=2)
        updated_sums = np.sum(rock_map_updated, axis=2)
        
        # Validierung
        original_valid = np.all(original_sums == 255)
        updated_valid = np.all(updated_sums == 255)
        
        # Massenerhaltung global
        total_original = np.sum(rock_map_original)
        total_updated = np.sum(rock_map_updated)
        mass_difference = abs(total_updated - total_original)
        
        results = {
            'original_mass_conservation': original_valid,
            'updated_mass_conservation': updated_valid,
            'total_mass_difference': float(mass_difference),
            'mass_conservation_ratio': float(total_updated / total_original) if total_original > 0 else 0.0,
            'invalid_pixels': int(np.sum(updated_sums != 255))
        }
        
        return results