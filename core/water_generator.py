"""
Path: core/water_generator.py

Funktionsweise: Dynamisches Hydrologiesystem mit Erosion, Sedimentation und bidirektionaler Terrain-Modifikation
- Lake-Detection durch Jump Flooding Algorithm für parallele Senken-Identifikation
- Flussnetzwerk-Aufbau durch Steepest Descent mit Upstream-Akkumulation
- Strömungsberechnung nach Manning-Gleichung mit adaptiven Querschnitten
- Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
- Stream Power Erosion mit iterativer Terrain-Modifikation über LOD-Levels
- Realistische Sedimentation mit kumulative Akkumulation
- Evaporation basierend auf statischen Weather-Daten (temp_map, wind_map, humid_map)

Parameter Input:
- lake_volume_threshold (Mindestvolumen für Seebildung, default 0.1m)
- rain_threshold (Niederschlagsschwelle für Quellbildung, default 0.02 gH2O/m²)
- manning_coefficient (Rauheitskoeffizient für Fließgeschwindigkeit, default 0.03)
- erosion_strength (Erosionsintensitäts-Multiplikator, default 1.0)
- sediment_capacity_factor (Transportkapazitäts-Faktor, default 0.1)
- evaporation_base_rate (Basis-Verdunstungsrate, default 0.002 m/Tag)
- diffusion_radius (Bodenfeuchtigkeit-Ausbreitungsradius, default 5.0 Pixel)
- settling_velocity (Sediment-Sinkgeschwindigkeit, default 0.01 m/s)
- erosion_iterations_per_lod (Anzahl Erosions-Zyklen pro LOD-Level, default 10)
- water_seed (Reproduzierbare Zufallsvariation)

Dependencies (über DataLODManager):
- heightmap (von terrain_generator für Orographic-Effects und Flow-Pathfinding)
- hardness_map (von geology_generator für Erosions-Resistance)
- precip_map (von weather_generator für Precipitation-driven Water-Sources)
- temp_map (von weather_generator für Temperature-based Evaporation)
- wind_map (von weather_generator für Wind-enhanced Evaporation)

Output:
- WaterData-Objekt mit water_map, flow_map, flow_speed, soil_moist_map, water_biomes_map,
  erosion_map, sedimentation_map, validity_state und LOD-Metadaten
- Bidirektionale Terrain-Integration: DataLODManager.get_terrain_data_combined() liest erosion_map/
  sedimentation_map und liefert base_heightmap - erosion_map + sedimentation_map an alle
  nachgelagerten Generatoren; jeder neue Water-LOD-Durchlauf bekommt seinerseits diese bereits
  erodierte Heightmap als Input und akkumuliert seine frische Erosion/Sedimentation on top
  (siehe calculate_hydrology(previous_erosion_map=..., previous_sedimentation_map=...))
- DataLODManager-Storage für nachfolgende Generatoren (biome, settlement)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix
from collections import deque
from typing import Dict, Any
import heapq
import logging

from gui.OldManagers.calculator_graph import CalculatorRoundScheduler

class WaterData:
    """
    Funktionsweise: Container für alle Water-Daten mit Metainformationen und LOD-System
    Aufgabe: Speichert alle 11 Hydrologie-Outputs mit LOD-Level und Validity-State
    """
    def __init__(self):
        self.water_map = None              # (height, width) - Gewässertiefen in m
        self.flow_map = None               # (height, width) - Volumenstrom in m³/s
        self.flow_speed = None             # (height, width) - Fließgeschwindigkeit in m/s
        self.cross_section = None          # (height, width) - Flusquerschnitt in m²
        self.soil_moist_map = None         # (height, width) - Bodenfeuchtigkeit in %
        self.erosion_map = None            # (height, width) - Erosionsrate in m/Jahr
        self.sedimentation_map = None      # (height, width) - Sedimentationsrate in m/Jahr
        self.evaporation_map = None        # (height, width) - Verdunstung in gH2O/m²/Tag
        self.ocean_outflow = None          # Scalar - Wasserabfluss ins Meer in m³/s
        self.water_biomes_map = None       # (height, width) - Wasser-Klassifikation 0-4

        # LOD-System Integration
        self.lod_level = "LOD64"           # Aktueller LOD-Level
        self.actual_size = 64              # Tatsächliche Kartengröße
        self.validity_state = "valid"      # Validity-State für Cache-Management
        self.parameter_hash = None         # Parameter-Hash für Cache-Invalidation
        self.parameters = {}               # Verwendete Parameter für Cache-Management

class LakeDetectionSystem:
    """
    Funktionsweise: Identifiziert Seen durch Jump Flooding Algorithm für parallele Senken-Identifikation
    Aufgabe: Findet alle potentiellen Seestandorte und deren Einzugsgebiete
    """

    def __init__(self, lake_volume_threshold=0.1, shader_manager=None):
        self.lake_volume_threshold = lake_volume_threshold
        self.shader_manager = shader_manager

    def detect_lakes(self, heightmap, parameters):
        """
        Funktionsweise: GPU-accelerated Lake-Detection mit Fallback-Strategie
        Aufgabe: 3-stufiges Fallback-System für robuste Lake-Detection
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "jumpFloodLakes",
                    {"heightmap": heightmap, "lake_volume_threshold": self.lake_volume_threshold},
                    parameters
                )
                if result.get("success"):
                    return result["lake_map"], result["valid_lakes"]
            except Exception as e:
                logging.warning(f"GPU lake detection failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        try:
            return self._cpu_lake_detection(heightmap)
        except Exception as e:
            logging.warning(f"CPU lake detection failed: {e}, using simple fallback")

        # Simple-Fallback (Minimal)
        return self._simple_lake_detection(heightmap)

    def _cpu_lake_detection(self, heightmap):
        """CPU-optimierte Lake-Detection mit Jump Flooding Algorithm"""
        height, width = heightmap.shape

        # Lokale Minima finden
        local_minima = self._detect_local_minima(heightmap)
        if not local_minima:
            return np.full((height, width), -1, dtype=np.int32), []

        # Jump Flooding Algorithm
        lake_map = self._apply_jump_flooding(heightmap, local_minima)

        # Lake-Klassifikation
        filtered_lake_map, valid_lakes = self._classify_lake_basins(heightmap, lake_map, local_minima)

        return filtered_lake_map, valid_lakes

    def _simple_lake_detection(self, heightmap):
        """Simple-Fallback: Basis Lake-Detection ohne komplexe Algorithmen"""
        height, width = heightmap.shape
        lake_map = np.full((height, width), -1, dtype=np.int32)

        # Sehr einfache Senken-Detection
        for y in range(1, height-1):
            for x in range(1, width-1):
                current = heightmap[y, x]
                neighbors = [
                    heightmap[y-1, x], heightmap[y+1, x],
                    heightmap[y, x-1], heightmap[y, x+1]
                ]
                if all(current < neighbor for neighbor in neighbors):
                    lake_map[y, x] = 0

        return lake_map, [{"seed": (width//2, height//2), "volume": 1.0}]

    def _detect_local_minima(self, heightmap):
        """Identifiziert alle lokalen Minima als potentielle See-Seeds"""
        height, width = heightmap.shape
        local_minima = []

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_height = heightmap[y, x]
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

    def _apply_jump_flooding(self, heightmap, lake_seeds):
        """Jump Flooding Algorithm für parallele Senken-Füllung in O(log n) Zeit"""
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

    def _classify_lake_basins(self, heightmap, lake_map, lake_seeds):
        """Klassifiziert See-Becken nach Volumen und validiert Threshold"""
        height, width = heightmap.shape
        filtered_lake_map = np.full((height, width), -1, dtype=np.int32)
        valid_lakes = []

        for lake_id, (seed_x, seed_y) in enumerate(lake_seeds):
            lake_pixels = np.where(lake_map == lake_id)

            if len(lake_pixels[0]) == 0:
                continue

            # Volumen-Berechnung
            seed_height = heightmap[seed_y, seed_x]
            total_volume = 0.0

            for py, px in zip(lake_pixels[0], lake_pixels[1]):
                terrain_height = heightmap[py, px]
                if terrain_height <= seed_height:
                    water_depth = seed_height - terrain_height
                    total_volume += water_depth

            # Volume-Threshold prüfen
            if total_volume >= self.lake_volume_threshold:
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

    def __init__(self, rain_threshold=5.0, shader_manager=None):
        self.rain_threshold = rain_threshold
        self.shader_manager = shader_manager

    def build_flow_network(self, heightmap, precip_map, lake_map, parameters, lod_iterations):
        """
        Funktionsweise: GPU-accelerated Flow-Network mit LOD-optimierten Iterationen
        Aufgabe: 3-stufiges Fallback-System für robuste Flow-Network-Generation
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "steepestDescentFlow",
                    {
                        "heightmap": heightmap,
                        "precip_map": precip_map,
                        "lake_map": lake_map,
                        "rain_threshold": self.rain_threshold,
                        "max_iterations": lod_iterations["flow"]
                    },
                    parameters
                )
                if result.get("success"):
                    flow_accumulation = result["flow_accumulation"]
                    water_biomes_map = self._classify_water_bodies(flow_accumulation, lake_map)
                    return flow_accumulation, water_biomes_map
            except Exception as e:
                logging.warning(f"GPU flow network failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        try:
            return self._cpu_flow_network(heightmap, precip_map, lake_map, lod_iterations)
        except Exception as e:
            logging.warning(f"CPU flow network failed: {e}, using simple fallback")

        # Simple-Fallback (Minimal)
        return self._simple_flow_network(heightmap, precip_map, lake_map)

    def _cpu_flow_network(self, heightmap, precip_map, lake_map, lod_iterations):
        """CPU-optimierte Flow-Network-Generation"""
        # Steepest Descent berechnen
        flow_directions = self._calculate_steepest_descent(heightmap)

        # Upstream-Akkumulation mit LOD-optimierten Iterationen
        flow_accumulation = self._accumulate_upstream_flow(
            flow_directions, precip_map, lake_map, lod_iterations["flow"]
        )

        # Water-Biomes klassifizieren
        water_biomes_map = self._classify_water_bodies(flow_accumulation, lake_map)

        return flow_accumulation, water_biomes_map

    def _simple_flow_network(self, heightmap, precip_map, lake_map):
        """Simple-Fallback: Basis Flow-Network ohne komplexe Akkumulation"""
        height, width = heightmap.shape
        flow_accumulation = np.where(precip_map > self.rain_threshold, precip_map, 0)
        water_biomes_map = np.zeros((height, width), dtype=np.uint8)
        water_biomes_map[lake_map >= 0] = 4  # Lake
        return flow_accumulation, water_biomes_map

    def _calculate_steepest_descent(self, heightmap):
        """Berechnet Steepest Descent Flow-Richtungen für jeden Pixel"""
        height, width = heightmap.shape
        flow_directions = np.zeros((height, width), dtype=np.int8)

        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        for y in range(height):
            for x in range(width):
                current_height = heightmap[y, x]
                steepest_gradient = 0.0
                steepest_direction = -1

                for direction, (dx, dy) in enumerate(direction_offsets):
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < width and 0 <= ny < height:
                        neighbor_height = heightmap[ny, nx]
                        height_diff = current_height - neighbor_height
                        distance = np.sqrt(dx ** 2 + dy ** 2)
                        gradient = height_diff / distance

                        if gradient > steepest_gradient:
                            steepest_gradient = gradient
                            steepest_direction = direction

                flow_directions[y, x] = steepest_direction

        return flow_directions

    def _accumulate_upstream_flow(self, flow_directions, precip_map, lake_map, max_iterations):
        """LOD-optimierte Upstream-Akkumulation mit begrenzten Iterationen"""
        height, width = flow_directions.shape
        flow_accumulation = np.zeros((height, width), dtype=np.float32)

        # Initiale Wassermengen
        initial_water = np.where(precip_map > self.rain_threshold, precip_map, 0)
        flow_accumulation = np.copy(initial_water)

        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        for iteration in range(max_iterations):
            new_accumulation = np.copy(flow_accumulation)
            changed = False

            for y in range(height):
                for x in range(width):
                    incoming_water = 0.0

                    # Sammle Wasser von allen Zellen die hierher fließen
                    for source_dir, (dx, dy) in enumerate(direction_offsets):
                        sx, sy = x - dx, y - dy

                        if 0 <= sx < width and 0 <= sy < height:
                            source_flow_dir = flow_directions[sy, sx]

                            if source_flow_dir == source_dir:
                                incoming_water += flow_accumulation[sy, sx]

                    # See-Handling
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

    def _classify_water_bodies(self, flow_accumulation, lake_map):
        """Klassifiziert Wasserkörper basierend auf Flussgröße"""
        height, width = flow_accumulation.shape
        water_biomes_map = np.zeros((height, width), dtype=np.uint8)

        # Seen zuerst markieren
        water_biomes_map[lake_map >= 0] = 4  # Lake

        # Flow-basierte Klassifikation
        for y in range(height):
            for x in range(width):
                if lake_map[y, x] >= 0:
                    continue

                flow_amount = flow_accumulation[y, x]

                # Schwellen skaliert zur precip_map/rain_threshold-Größenordnung
                # (siehe WATER.RAIN_THRESHOLD in gui/config/value_default.py)
                if flow_amount >= 0.4:  # Grand River
                    water_biomes_map[y, x] = 3
                elif flow_amount >= 0.08:  # River
                    water_biomes_map[y, x] = 2
                elif flow_amount >= 0.02:  # Creek
                    water_biomes_map[y, x] = 1

        return water_biomes_map


class ManningFlowCalculator:
    """
    Funktionsweise: Berechnet Strömung nach Manning-Gleichung mit adaptiven Querschnitten
    Aufgabe: Erstellt flow_speed und cross_section für realistische Fließgeschwindigkeiten
    """

    def __init__(self, manning_coefficient=0.03, shader_manager=None):
        self.manning_n = manning_coefficient
        self.shader_manager = shader_manager

    def calculate_flow_properties(self, flow_accumulation, slopemap, heightmap, parameters, lod_iterations):
        """
        Funktionsweise: GPU-accelerated Manning-Flow mit adaptiven Querschnitten
        Aufgabe: 3-stufiges Fallback-System für realistische Fließgeschwindigkeiten
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "manningFlowCalculation",
                    {
                        "flow_accumulation": flow_accumulation,
                        "slopemap": slopemap,
                        "heightmap": heightmap,
                        "manning_n": self.manning_n,
                        "depth_tests": lod_iterations["manning"]
                    },
                    parameters
                )
                if result.get("success"):
                    return result["flow_speed"], result["cross_section"]
            except Exception as e:
                logging.warning(f"GPU Manning calculation failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        try:
            return self._cpu_manning_calculation(flow_accumulation, slopemap, heightmap, lod_iterations)
        except Exception as e:
            logging.warning(f"CPU Manning calculation failed: {e}, using simple fallback")

        # Simple-Fallback (Minimal)
        return self._simple_flow_calculation(flow_accumulation)

    def _cpu_manning_calculation(self, flow_accumulation, slopemap, heightmap, lod_iterations):
        """CPU-optimierte Manning-Gleichung mit adaptiven Querschnitten"""
        height, width = flow_accumulation.shape
        flow_speed = np.zeros((height, width), dtype=np.float32)
        cross_section = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                flow_rate = flow_accumulation[y, x]

                if flow_rate < 1.0:
                    continue

                # Slope-Magnitude berechnen
                slope_x = slopemap[y, x, 0] if x < slopemap.shape[1] else 0
                slope_y = slopemap[y, x, 1] if y < slopemap.shape[0] else 0
                slope = np.sqrt(slope_x ** 2 + slope_y ** 2)
                slope = max(0.001, slope)

                # Optimaler Querschnitt
                optimal_area, hydraulic_radius = self._optimize_channel_geometry(
                    flow_rate, slope, heightmap, x, y, lod_iterations["manning"]
                )

                # Manning-Geschwindigkeit
                if hydraulic_radius > 0:
                    velocity = (1.0 / self.manning_n) * (hydraulic_radius ** (2.0 / 3.0)) * (slope ** 0.5)

                    if velocity > 0:
                        required_area = flow_rate / velocity
                        cross_section[y, x] = required_area
                        flow_speed[y, x] = velocity

        return flow_speed, cross_section

    def _simple_flow_calculation(self, flow_accumulation):
        """Simple-Fallback: Basis Fließgeschwindigkeits-Approximation"""
        height, width = flow_accumulation.shape
        flow_speed = np.sqrt(flow_accumulation + 1e-6) * 0.1  # Vereinfachte Geschwindigkeit
        cross_section = flow_accumulation * 0.01  # Vereinfachter Querschnitt
        return flow_speed, cross_section

    def _optimize_channel_geometry(self, flow_rate, slope, heightmap, x, y, depth_tests):
        """LOD-optimierte Kanal-Geometrie-Optimierung"""
        valley_width = self._analyze_valley_width(heightmap, x, y)

        if valley_width < 5:
            width_to_depth_ratio = 2.0
        elif valley_width < 20:
            width_to_depth_ratio = 5.0
        else:
            width_to_depth_ratio = 10.0

        best_area = 0
        best_hydraulic_radius = 0

        for depth in np.linspace(0.1, 5.0, depth_tests):
            width = depth * width_to_depth_ratio
            area = width * depth
            wetted_perimeter = width + 2 * depth
            hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0

            velocity = (1.0 / self.manning_n) * (hydraulic_radius ** (2.0 / 3.0)) * (slope ** 0.5)
            theoretical_flow = area * velocity

            if abs(theoretical_flow - flow_rate) < abs(best_area * velocity - flow_rate):
                best_area = area
                best_hydraulic_radius = hydraulic_radius

        return best_area, best_hydraulic_radius

    def _analyze_valley_width(self, heightmap, center_x, center_y):
        """Vereinfachte Tal-Breite-Analyse für bessere Performance"""
        height, width = heightmap.shape

        if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
            return 10.0

        center_height = heightmap[center_y, center_x]
        height_threshold = center_height + 20.0

        max_distance = 0

        for angle in np.linspace(0, 2 * np.pi, 8):
            dx = np.cos(angle)
            dy = np.sin(angle)

            distance = 0
            for step in range(1, 25):
                test_x = center_x + int(dx * step)
                test_y = center_y + int(dy * step)

                if test_x < 0 or test_x >= width or test_y < 0 or test_y >= height:
                    break

                test_height = heightmap[test_y, test_x]
                if test_height > height_threshold:
                    distance = step
                    break

            max_distance = max(max_distance, distance)

        return max_distance * 2


class ErosionSedimentationSystem:
    """
    Funktionsweise: Simuliert Stream Power Erosion mit iterativen Terrain-Modifikationen
    Aufgabe: Modifiziert Landschaft durch erosion_map und sedimentation_map mit realistischem Sediment-Transport
    """

    def __init__(self, erosion_strength=1.0, sediment_capacity_factor=0.1, settling_velocity=0.01, shader_manager=None):
        self.erosion_strength = erosion_strength
        self.capacity_factor = sediment_capacity_factor
        self.settling_velocity = settling_velocity
        self.shader_manager = shader_manager

    def simulate_erosion_sedimentation(self, flow_accumulation, flow_speed, flow_directions, hardness_map, parameters, lod_iterations):
        """
        Funktionsweise: GPU-accelerated Erosion-Sedimentation mit iterativen Terrain-Modifikationen
        Aufgabe: 3-stufiges Fallback-System für realistische Landschafts-Evolution
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                erosion_result = self.shader_manager.request_shader_operation(
                    "water", "streamPowerErosion",
                    {
                        "flow_accumulation": flow_accumulation,
                        "flow_speed": flow_speed,
                        "hardness_map": hardness_map,
                        "erosion_strength": self.erosion_strength
                    },
                    parameters
                )

                if erosion_result.get("success"):
                    erosion_map = erosion_result["erosion_map"]

                    transport_result = self.shader_manager.request_shader_operation(
                        "water", "sedimentTransport",
                        {
                            "erosion_map": erosion_map,
                            "flow_speed": flow_speed,
                            "flow_directions": flow_directions,
                            "flow_accumulation": flow_accumulation,
                            "capacity_factor": self.capacity_factor,
                            "settling_velocity": self.settling_velocity,
                            "iterations": lod_iterations["sediment"]
                        },
                        parameters
                    )

                    if transport_result.get("success"):
                        return erosion_map, transport_result["sedimentation_map"]

            except Exception as e:
                logging.warning(f"GPU erosion simulation failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        try:
            return self._cpu_erosion_sedimentation(flow_accumulation, flow_speed, flow_directions, hardness_map, lod_iterations)
        except Exception as e:
            logging.warning(f"CPU erosion simulation failed: {e}, using simple fallback")

        # Simple-Fallback (Minimal)
        return self._simple_erosion_sedimentation(flow_accumulation, hardness_map)

    def _cpu_erosion_sedimentation(self, flow_accumulation, flow_speed, flow_directions, hardness_map, lod_iterations):
        """CPU-optimierte Erosion-Sedimentation mit Stream Power"""
        # Stream Power Erosion berechnen
        erosion_map = self._calculate_stream_power_erosion(flow_accumulation, flow_speed, hardness_map)

        # Sediment-Transport mit LOD-optimierten Iterationen
        sedimentation_map = self._transport_sediment_optimized(
            erosion_map, flow_speed, flow_directions, flow_accumulation, lod_iterations["sediment"]
        )

        return erosion_map, sedimentation_map

    def _simple_erosion_sedimentation(self, flow_accumulation, hardness_map):
        """Simple-Fallback: Basis Erosion ohne komplexe Transport-Berechnungen"""
        height, width = flow_accumulation.shape
        erosion_map = np.zeros((height, width), dtype=np.float32)
        sedimentation_map = np.zeros((height, width), dtype=np.float32)

        # Sehr vereinfachte Erosion basierend auf Flow-Stärke
        erosion_factor = self.erosion_strength * 0.001
        for y in range(height):
            for x in range(width):
                flow_strength = flow_accumulation[y, x]
                hardness = hardness_map[y, x] if x < hardness_map.shape[1] and y < hardness_map.shape[0] else 50.0

                if flow_strength > 10.0:
                    erosion_rate = erosion_factor * flow_strength / max(hardness, 1.0)
                    erosion_map[y, x] = min(erosion_rate, 0.1)  # Max 0.1m/Jahr

                    # Einfache Sedimentation in niedrigeren Bereichen
                    if y < height - 1 and flow_strength > 5.0:
                        sedimentation_map[y + 1, x] = erosion_rate * 0.5

        return erosion_map, sedimentation_map

    def _calculate_stream_power_erosion(self, flow_accumulation, flow_speed, hardness_map):
        """Stream Power Erosion: E = K * (τ - τc) mit Scherspannung"""
        height, width = flow_accumulation.shape
        erosion_map = np.zeros((height, width), dtype=np.float32)

        rho_water = 1000.0  # kg/m³
        gravity = 9.81      # m/s²

        for y in range(height):
            for x in range(width):
                flow_rate = flow_accumulation[y, x]
                velocity = flow_speed[y, x]

                # Flow-Rate-Schwelle skaliert zur precip_map/rain_threshold-Größenordnung
                # (siehe WATER.RAIN_THRESHOLD in gui/config/value_default.py)
                if flow_rate < 0.004 or velocity < 0.1:
                    continue

                # Approximiere Wassertiefe aus Flow-Rate und Geschwindigkeit
                water_depth = flow_rate / (velocity * 10.0) if velocity > 0 else 0

                # Vereinfachte Slope-Schätzung
                slope = velocity / 10.0  # Grobe Approximation

                # Scherspannung
                shear_stress = rho_water * gravity * water_depth * slope

                # Kritische Scherspannung basierend auf Gesteinshärte
                hardness = hardness_map[y, x] if x < hardness_map.shape[1] and y < hardness_map.shape[0] else 50.0
                critical_shear = hardness * 10.0

                # Erosion nur wenn Scherspannung kritischen Wert überschreitet
                if shear_stress > critical_shear:
                    excess_stress = shear_stress - critical_shear
                    erosion_rate = self.erosion_strength * excess_stress * (velocity**2) / hardness
                    erosion_rate = min(erosion_rate * 1e-6, 0.1)  # Skalierung und Begrenzung
                    erosion_map[y, x] = erosion_rate

        return erosion_map

    def _transport_sediment_optimized(self, erosion_map, flow_speed, flow_directions, flow_accumulation, sediment_iterations):
        """LOD-optimierte Sediment-Transport mit reduzierten Iterationen"""
        height, width = erosion_map.shape
        sedimentation_map = np.zeros((height, width), dtype=np.float32)
        sediment_load = np.zeros((height, width), dtype=np.float32)

        # Transport-Kapazität berechnen
        transport_capacity = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                velocity = flow_speed[y, x]
                if velocity > 0.1:
                    transport_capacity[y, x] = self.capacity_factor * (velocity ** 2.5)

        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

        # LOD-spezifische Iterationen
        for iteration in range(sediment_iterations):
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
                            velocity = flow_speed[y, x]
                            if velocity > 0.1:
                                transport_efficiency = min(1.0, velocity / 2.0)
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


class SoilMoistureCalculator:
    """
    Funktionsweise: Berechnet Bodenfeuchtigkeit durch Gaussian-Diffusion von Gewässern
    Aufgabe: Erstellt soil_moist_map für Biome-System und Weather-Evaporation
    """

    def __init__(self, diffusion_radius=5.0, shader_manager=None):
        self.diffusion_radius = diffusion_radius
        self.shader_manager = shader_manager

    def calculate_soil_moisture(self, water_biomes_map, flow_accumulation, parameters):
        """
        Funktionsweise: GPU-accelerated Soil-Moisture mit Multi-Radius-Filter
        Aufgabe: 3-stufiges Fallback-System für realistische Feuchtigkeits-Verteilung
        """
        # GPU-Shader (Optimal)
        if self.shader_manager:
            try:
                result = self.shader_manager.request_shader_operation(
                    "water", "soilMoistureGaussian",
                    {
                        "water_biomes_map": water_biomes_map,
                        "flow_accumulation": flow_accumulation,
                        "diffusion_radius": self.diffusion_radius
                    },
                    parameters
                )
                if result.get("success"):
                    return result["soil_moisture"]
            except Exception as e:
                logging.warning(f"GPU soil moisture calculation failed: {e}, falling back to CPU")

        # CPU-Fallback (Gut)
        try:
            return self._cpu_soil_moisture_calculation(water_biomes_map, flow_accumulation)
        except Exception as e:
            logging.warning(f"CPU soil moisture calculation failed: {e}, using simple fallback")

        # Simple-Fallback (Minimal)
        return self._simple_soil_moisture_calculation(water_biomes_map)

    def _cpu_soil_moisture_calculation(self, water_biomes_map, flow_accumulation):
        """CPU-optimierte Gaussian-Diffusion mit Multi-Radius-Filter"""
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
        combined_moisture[water_mask] = 100.0

        return combined_moisture

    def _simple_soil_moisture_calculation(self, water_biomes_map):
        """Simple-Fallback: Basis Bodenfeuchtigkeit ohne Gaussian-Diffusion"""
        height, width = water_biomes_map.shape
        soil_moisture = np.zeros((height, width), dtype=np.float32)

        # Direkte Wasserpräsenz
        soil_moisture[water_biomes_map > 0] = 100.0

        # Einfache radiale Ausbreitung
        for y in range(height):
            for x in range(width):
                if water_biomes_map[y, x] > 0:
                    # Vereinfachte Nachbarschaft-Feuchtigkeit
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                distance = np.sqrt(dx*dx + dy*dy)
                                if distance > 0:
                                    moisture = 50.0 / distance
                                    soil_moisture[ny, nx] = max(soil_moisture[ny, nx], moisture)

        return np.clip(soil_moisture, 0, 100)


class EvaporationCalculator:
    """
    Funktionsweise: Berechnet Evaporation basierend auf statischen Weather-Daten
    Aufgabe: Erstellt evaporation_map durch Integration von temp_map, wind_map und humid_map
    """

    def __init__(self, evaporation_base_rate=0.002, shader_manager=None):
        self.base_rate = evaporation_base_rate
        self.shader_manager = shader_manager

    def calculate_evaporation(self, temp_map, wind_map, humid_map, water_biomes_map, parameters):
        """
        Funktionsweise: GPU-accelerated Evaporation mit Magnus-Formel
        Aufgabe: 3-stufiges Fallback-System für realistische Verdunstung
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

        # CPU-Fallback (Gut)
        try:
            return self._cpu_evaporation_calculation(temp_map, wind_map, humid_map, water_biomes_map)
        except Exception as e:
            logging.warning(f"CPU evaporation calculation failed: {e}, using simple fallback")

        # Simple-Fallback (Minimal)
        return self._simple_evaporation_calculation(water_biomes_map)

    def _cpu_evaporation_calculation(self, temp_map, wind_map, humid_map, water_biomes_map):
        """CPU-optimierte Evaporation mit Magnus-Formel"""
        height, width = temp_map.shape
        evaporation_map = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                if water_biomes_map[y, x] == 0:
                    continue

                temperature = temp_map[y, x]
                humidity = humid_map[y, x]
                wind_speed = np.sqrt(wind_map[y, x, 0]**2 + wind_map[y, x, 1]**2)

                # Magnus-Formel für maximale Wasserdampfdichte
                max_vapor_density = 5.0 * np.exp(0.06 * temperature)

                # Relative Feuchtigkeit
                if max_vapor_density > 0:
                    relative_humidity = humidity / max_vapor_density
                    relative_humidity = min(1.0, relative_humidity)
                else:
                    relative_humidity = 1.0

                # Evaporation-Faktoren
                humidity_factor = 1.0 - relative_humidity
                temp_factor = np.exp(temperature / 20.0) if temperature > 0 else 0.1
                wind_factor = 1.0 + wind_speed * 0.2

                # Basis-Evaporation
                evaporation_rate = self.base_rate * humidity_factor * temp_factor * wind_factor * 1000
                evaporation_map[y, x] = evaporation_rate

        return self._limit_by_available_water(evaporation_map, water_biomes_map)

    def _simple_evaporation_calculation(self, water_biomes_map):
        """Simple-Fallback: Fixed Evaporation-Rate ohne atmosphärische Komplexität"""
        evaporation_map = np.zeros_like(water_biomes_map, dtype=np.float32)
        evaporation_map[water_biomes_map > 0] = self.base_rate * 500  # Fixed rate
        return evaporation_map

    def _limit_by_available_water(self, evaporation_map, water_biomes_map):
        """Begrenzt Evaporation durch verfügbare Wasseroberfläche"""
        height, width = evaporation_map.shape
        limited_evaporation = np.copy(evaporation_map)

        for y in range(height):
            for x in range(width):
                water_type = water_biomes_map[y, x]
                potential_evap = evaporation_map[y, x]

                if water_type == 0:
                    limited_evaporation[y, x] = 0.0
                elif water_type == 1:  # Creek
                    limited_evaporation[y, x] = min(potential_evap, 50.0)
                elif water_type == 2:  # River
                    limited_evaporation[y, x] = min(potential_evap, 100.0)
                elif water_type == 3:  # Grand River
                    limited_evaporation[y, x] = min(potential_evap, 200.0)
                # Lake: keine Begrenzung

        return limited_evaporation


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
        self.erosion_system = ErosionSedimentationSystem(shader_manager=shader_manager)
        self.soil_moisture = SoilMoistureCalculator(shader_manager=shader_manager)
        self.evaporation = EvaporationCalculator(shader_manager=shader_manager)

    def _load_default_parameters(self):
        """Lädt WATER-Parameter aus value_default.py"""
        try:
            from gui.config.value_default import WATER
            return {
                'lake_volume_threshold': WATER.LAKE_VOLUME_THRESHOLD["default"],
                'rain_threshold': WATER.RAIN_THRESHOLD["default"],
                'manning_coefficient': WATER.MANNING_COEFFICIENT["default"],
                'erosion_strength': WATER.EROSION_STRENGTH["default"],
                'sediment_capacity_factor': WATER.SEDIMENT_CAPACITY_FACTOR["default"],
                'evaporation_base_rate': WATER.EVAPORATION_BASE_RATE["default"],
                'diffusion_radius': WATER.DIFFUSION_RADIUS["default"],
                'settling_velocity': WATER.SETTLING_VELOCITY["default"],
                'erosion_iterations_per_lod': WATER.EROSION_ITERATIONS_PER_LOD.get("default", 10),
                'water_seed': WATER.WATER_SEED.get("default", 12345)
            }
        except ImportError:
            # Fallback defaults
            return {
                'lake_volume_threshold': 0.1,
                'rain_threshold': 0.02,
                'manning_coefficient': 0.03,
                'erosion_strength': 1.0,
                'sediment_capacity_factor': 0.1,
                'evaporation_base_rate': 0.002,
                'diffusion_radius': 5.0,
                'settling_velocity': 0.01,
                'erosion_iterations_per_lod': 10,
                'water_seed': 12345
            }

    def _get_dependencies(self, data_manager):
        """Holt alle 8 benötigten Dependencies aus DataManager"""
        if not data_manager:
            raise Exception("DataManager required for Water generation")

        dependencies = {}

        try:
            # Terrain-Dependencies (2)
            heightmap = data_manager.get_terrain_data_combined("heightmap")
            slopemap = data_manager.get_terrain_data("slopemap")

            if heightmap is None:
                raise Exception("Heightmap dependency not available - run Terrain generator first")
            if slopemap is None:
                raise Exception("Slopemap dependency not available - run Terrain generator first")

            dependencies['heightmap'] = heightmap
            dependencies['slopemap'] = slopemap

            # Geology-Dependencies (2)
            hardness_map = data_manager.get_geology_data("hardness_map")
            rock_map = data_manager.get_geology_data("rock_map")

            if hardness_map is None:
                raise Exception("Hardness_map dependency not available - run Geology generator first")
            if rock_map is None:
                raise Exception("Rock_map dependency not available - run Geology generator first")

            dependencies['hardness_map'] = hardness_map
            dependencies['rock_map'] = rock_map

            # Weather-Dependencies (4)
            precip_map = data_manager.get_weather_data("precip_map")
            temp_map = data_manager.get_weather_data("temp_map")
            wind_map = data_manager.get_weather_data("wind_map")
            humid_map = data_manager.get_weather_data("humid_map")

            if precip_map is None:
                raise Exception("Precip_map dependency not available - run Weather generator first")
            if temp_map is None:
                raise Exception("Temp_map dependency not available - run Weather generator first")
            if wind_map is None:
                raise Exception("Wind_map dependency not available - run Weather generator first")
            if humid_map is None:
                raise Exception("Humid_map dependency not available - run Weather generator first")

            dependencies['precip_map'] = precip_map
            dependencies['temp_map'] = temp_map
            dependencies['wind_map'] = wind_map
            dependencies['humid_map'] = humid_map

            self.logger.debug("All 8 Water dependencies loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load dependencies: {e}")
            raise

        return dependencies

    def _execute_generation(self, lod, dependencies, parameters,
                           previous_erosion_map=None, previous_sedimentation_map=None):
        """
        Führt Water-Generierung mit LOD-optimierten Algorithmen aus.
        previous_erosion_map/previous_sedimentation_map: kumulierte Erosion/Sedimentation aus dem
        letzten abgeschlossenen Water-LOD (vom Aufrufer über DataLODManager geholt). Wenn gesetzt,
        wird die in dieser Passage frisch berechnete Erosion/Sedimentation dazuaddiert statt sie zu
        ersetzen - so erodiert jeder LOD-Durchlauf die bereits erodierte Landschaft weiter, statt
        immer wieder bei der Basis-Heightmap neu anzufangen.
        """
        self.logger.info(f"Starting water generation for LOD {lod}")

        # Dependencies extrahieren
        heightmap = dependencies['heightmap']
        slopemap = dependencies['slopemap']
        hardness_map = dependencies['hardness_map']
        rock_map = dependencies['rock_map']
        precip_map = dependencies['precip_map']
        temp_map = dependencies['temp_map']
        wind_map = dependencies['wind_map']
        humid_map = dependencies['humid_map']

        # Parameter aktualisieren
        self._update_parameters(parameters)

        # LOD-Größe bestimmen
        target_size = self._get_lod_size(lod, heightmap.shape[0])

        # Alle Arrays auf Zielgröße interpolieren
        heightmap = self._interpolate_array(heightmap, target_size)
        slopemap = self._interpolate_array(slopemap, target_size)
        hardness_map = self._interpolate_array(hardness_map, target_size)
        rock_map = self._interpolate_array(rock_map, target_size)
        precip_map = self._interpolate_array(precip_map, target_size)
        temp_map = self._interpolate_array(temp_map, target_size)
        wind_map = self._interpolate_array(wind_map, target_size)
        humid_map = self._interpolate_array(humid_map, target_size)

        # LOD-spezifische Iterationsanzahl
        lod_iterations = self._get_lod_iterations(lod)

        try:
            # Läuft intern über CalculatorRoundScheduler statt einer flachen Aufruf-
            # Sequenz (siehe gui/OldManagers/calculator_graph.py - Water-Calculator-
            # Knoten #15-#21 aus docs/generation_pipeline_dependencies.md, #22
            # erosion_feedback bewusst ausgeschlossen - bekannt kaputt). Phase des
            # LOD-Lockstep-Umbaus (Tracker #16): nur Water-intern zerlegt, externes
            # Verhalten (calculate_hydrology()/_execute_generation() Signatur/
            # Rückgabe inkl. previous_erosion_map/previous_sedimentation_map-
            # Akkumulation) bleibt unverändert.
            context = {
                "heightmap": heightmap, "slopemap": slopemap, "hardness_map": hardness_map,
                "rock_map": rock_map, "precip_map": precip_map, "temp_map": temp_map,
                "wind_map": wind_map, "humid_map": humid_map,
                "parameters": parameters, "lod_iterations": lod_iterations,
                "target_size": target_size,
                "previous_erosion_map": previous_erosion_map,
                "previous_sedimentation_map": previous_sedimentation_map,
            }

            scheduler = CalculatorRoundScheduler(
                calculator_ids=[
                    "water.lake_detection", "water.flow_network", "water.steepest_descent",
                    "water.manning_flow", "water.erosion_sedimentation", "water.soil_moisture",
                    "water.evaporation",
                ],
                executors={
                    "water.lake_detection": self._calc_lake_detection,
                    "water.flow_network": self._calc_flow_network,
                    "water.steepest_descent": self._calc_steepest_descent,
                    "water.manning_flow": self._calc_manning_flow,
                    "water.erosion_sedimentation": self._calc_erosion_sedimentation,
                    "water.soil_moisture": self._calc_soil_moisture,
                    "water.evaporation": self._calc_evaporation,
                },
            )
            scheduler.run_round(target_lod=lod, context=context)

            # Finale Berechnungen (reine Output-Formatierung, kein eigener
            # Calculator-Knoten aus docs/generation_pipeline_dependencies.md)
            self._update_progress("Finalization", 98, "Creating water depth map...")
            water_map = self._create_water_depth_map(
                context["water_biomes_map"], context["flow_accumulation"], context["cross_section"])
            ocean_outflow = self._calculate_ocean_outflow(
                context["flow_accumulation"], context["flow_directions"], heightmap.shape)

            # WaterData-Objekt erstellen
            water_data = WaterData()
            water_data.water_map = water_map
            water_data.flow_map = context["flow_accumulation"]
            water_data.flow_speed = context["flow_speed"]
            water_data.cross_section = context["cross_section"]
            water_data.soil_moist_map = context["soil_moist_map"]
            water_data.erosion_map = context["erosion_map"]
            water_data.sedimentation_map = context["sedimentation_map"]
            water_data.evaporation_map = context["evaporation_map"]
            water_data.ocean_outflow = ocean_outflow
            water_data.water_biomes_map = context["water_biomes_map"]
            water_data.lod_level = lod
            water_data.actual_size = target_size
            water_data.parameters = parameters.copy()
            water_data.validity_state = "valid"
            water_data.parameter_hash = self._calculate_parameter_hash(parameters)

            self.logger.debug(f"Water generation complete - LOD: {lod}, size: {target_size}")
            return water_data

        except Exception as e:
            self.logger.error(f"Water generation failed: {e}")
            # Fallback zu minimal water data
            return self._create_minimal_water_data(target_size, lod, parameters)

    def _calc_lake_detection(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'water.lake_detection' (#15)"""
        self._update_progress("Lake Detection", 5, "Detecting local minima...")
        lake_map, valid_lakes = self.lake_detection.detect_lakes(context["heightmap"], context["parameters"])
        context["lake_map"] = lake_map
        context["valid_lakes"] = valid_lakes

    def _calc_flow_network(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'water.flow_network' (#16)"""
        self._update_progress("Flow Network", 20, "Calculating steepest descent...")
        flow_accumulation, water_biomes_map = self.flow_network.build_flow_network(
            context["heightmap"], context["precip_map"], context["lake_map"], context["parameters"],
            context["lod_iterations"]
        )
        context["flow_accumulation"] = flow_accumulation
        context["water_biomes_map"] = water_biomes_map

    def _calc_steepest_descent(self, context: Dict[str, Any]) -> None:
        """
        Calculator-Node 'water.steepest_descent' (#17) - Sibling zu flow_network,
        haengt nur von heightmap ab. flow_network berechnet Flow-Directions intern
        nochmal fuer die eigene Upstream-Akkumulation (privates Detail von
        FlowNetworkBuilder) - dieser Knoten liefert das fuer die
        Erosion-Sedimentation benoetigte flow_directions-Array.
        """
        context["flow_directions"] = self.flow_network._calculate_steepest_descent(context["heightmap"])

    def _calc_manning_flow(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'water.manning_flow' (#18)"""
        self._update_progress("Manning Flow", 45, "Solving Manning equation...")
        flow_speed, cross_section = self.manning_calculator.calculate_flow_properties(
            context["flow_accumulation"], context["slopemap"], context["heightmap"], context["parameters"],
            context["lod_iterations"]
        )
        context["flow_speed"] = flow_speed
        context["cross_section"] = cross_section

    def _calc_erosion_sedimentation(self, context: Dict[str, Any]) -> None:
        """
        Calculator-Node 'water.erosion_sedimentation' (#19) - inkl. Kumulation
        über LOD-Durchläufe (previous_erosion_map/previous_sedimentation_map),
        analog zu Geology._calc_tectonic_deformation().
        """
        self._update_progress("Erosion-Sedimentation", 65, "Calculating stream power...")
        erosion_map, sedimentation_map = self.erosion_system.simulate_erosion_sedimentation(
            context["flow_accumulation"], context["flow_speed"], context["flow_directions"],
            context["hardness_map"], context["parameters"], context["lod_iterations"]
        )

        target_size = context["target_size"]
        previous_erosion_map = context["previous_erosion_map"]
        previous_sedimentation_map = context["previous_sedimentation_map"]

        # Kumulative Akkumulation über LOD-Durchläufe: die frische Erosion/Sedimentation
        # dieser Passage kommt zur bereits akkumulierten Menge aus dem letzten LOD dazu.
        if previous_erosion_map is not None:
            if previous_erosion_map.shape[0] != target_size:
                previous_erosion_map = self._interpolate_2d(previous_erosion_map, target_size)
            erosion_map = erosion_map + previous_erosion_map
        if previous_sedimentation_map is not None:
            if previous_sedimentation_map.shape[0] != target_size:
                previous_sedimentation_map = self._interpolate_2d(previous_sedimentation_map, target_size)
            sedimentation_map = sedimentation_map + previous_sedimentation_map

        context["erosion_map"] = erosion_map
        context["sedimentation_map"] = sedimentation_map

    def _calc_soil_moisture(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'water.soil_moisture' (#20)"""
        self._update_progress("Soil Moisture", 88, "Calculating gaussian diffusion...")
        context["soil_moist_map"] = self.soil_moisture.calculate_soil_moisture(
            context["water_biomes_map"], context["flow_accumulation"], context["parameters"]
        )

    def _calc_evaporation(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'water.evaporation' (#21) - Sibling zu soil_moisture"""
        self._update_progress("Evaporation", 96, "Calculating atmospheric evaporation...")
        context["evaporation_map"] = self.evaporation.calculate_evaporation(
            context["temp_map"], context["wind_map"], context["humid_map"], context["water_biomes_map"],
            context["parameters"]
        )

    def _update_parameters(self, parameters):
        """Aktualisiert alle Sub-System Parameter"""
        self.lake_detection.lake_volume_threshold = parameters.get('lake_volume_threshold', 0.1)
        self.flow_network.rain_threshold = parameters.get('rain_threshold', 5.0)
        self.manning_calculator.manning_n = parameters.get('manning_coefficient', 0.03)
        self.erosion_system.erosion_strength = parameters.get('erosion_strength', 1.0)
        self.erosion_system.capacity_factor = parameters.get('sediment_capacity_factor', 0.1)
        self.erosion_system.settling_velocity = parameters.get('settling_velocity', 0.01)
        self.soil_moisture.diffusion_radius = parameters.get('diffusion_radius', 5.0)
        self.evaporation.base_rate = parameters.get('evaporation_base_rate', 0.002)

    def _save_to_data_manager(self, data_manager, result, parameters):
        """Speichert alle 11 Water-Outputs im DataManager"""
        if isinstance(result, WaterData):
            data_manager.set_water_data("water_map", result.water_map, parameters)
            data_manager.set_water_data("flow_map", result.flow_map, parameters)
            data_manager.set_water_data("flow_speed", result.flow_speed, parameters)
            data_manager.set_water_data("cross_section", result.cross_section, parameters)
            data_manager.set_water_data("soil_moist_map", result.soil_moist_map, parameters)
            data_manager.set_water_data("erosion_map", result.erosion_map, parameters)
            data_manager.set_water_data("sedimentation_map", result.sedimentation_map, parameters)
            data_manager.set_water_data("evaporation_map", result.evaporation_map, parameters)
            data_manager.set_water_data("ocean_outflow", result.ocean_outflow, parameters)
            data_manager.set_water_data("water_biomes_map", result.water_biomes_map, parameters)

            self.logger.debug("WaterData object with 11 outputs saved to DataManager")
        else:
            self.logger.warning(f"Unknown water result format: {type(result)}")

    def update_seed(self, new_seed):
        """Aktualisiert Seed für alle Water-Komponenten"""
        if new_seed != self.map_seed:
            super().update_seed(new_seed)

    def _get_lod_size(self, lod, original_size):
        """
        Bestimmt Zielgröße basierend auf LOD-Level.
        Unterstützt sowohl das numerische LOD-System (int, alle aktuellen Aufrufer über
        GenerationOrchestrator/GenerationThread) als auch die alten String-LODs
        ("LOD64"/"LOD128"/"LOD256"/"FINAL", nur noch für generate_hydrology_system()).
        """
        if isinstance(lod, str):
            if lod == "FINAL":
                return original_size
            lod_sizes = {"LOD64": 64, "LOD128": 128, "LOD256": 256}
            return lod_sizes.get(lod, 64)

        from gui.config.value_default import TERRAIN
        target_size = TERRAIN.MAPSIZEMIN * (2 ** (lod - 1))
        return min(target_size, original_size)

    def _get_lod_iterations(self, lod):
        """
        Bestimmt LOD-spezifische Iterationsanzahl für Performance-Optimierung.
        Unterstützt sowohl das numerische LOD-System (int) als auch die alten
        String-LODs (nur noch für generate_hydrology_system()).
        """
        if isinstance(lod, str):
            if lod == "LOD64":
                return {'flow': 50, 'sediment': 3, 'manning': 5}
            elif lod == "LOD128":
                return {'flow': 100, 'sediment': 5, 'manning': 10}
            elif lod == "LOD256":
                return {'flow': 200, 'sediment': 7, 'manning': 15}
            else:  # FINAL
                return {'flow': 400, 'sediment': 10, 'manning': 20}

        # Numerisches LOD: gleiche Stufen wie bisher, jetzt nach LOD-Level statt String
        if lod <= 1:
            return {'flow': 50, 'sediment': 3, 'manning': 5}
        elif lod == 2:
            return {'flow': 100, 'sediment': 5, 'manning': 10}
        elif lod == 3:
            return {'flow': 200, 'sediment': 7, 'manning': 15}
        else:
            return {'flow': 400, 'sediment': 10, 'manning': 20}

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
        """Bilineare Interpolation für 2D-Arrays"""
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

    def _ensure_mass_conservation(self, rock_map):
        """Stellt sicher dass R+G+B=255 für rock_map nach Interpolation"""
        height, width = rock_map.shape[:2]
        conserved_map = np.copy(rock_map).astype(np.float32)

        for y in range(height):
            for x in range(width):
                r, g, b = conserved_map[y, x, :]
                total = r + g + b

                if total > 0:
                    # Normalisierung auf 255
                    conserved_map[y, x, 0] = (r / total) * 255
                    conserved_map[y, x, 1] = (g / total) * 255
                    conserved_map[y, x, 2] = (b / total) * 255
                else:
                    # Gleichverteilung bei total=0
                    conserved_map[y, x, :] = [85, 85, 85]

        return conserved_map.astype(np.uint8)

    def _create_water_depth_map(self, water_biomes_map, flow_accumulation, cross_section):
        """Erstellt Wasser-Tiefen-Map aus Flow-Daten"""
        height, width = water_biomes_map.shape
        water_map = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                water_type = water_biomes_map[y, x]

                if water_type == 0:  # Kein Wasser
                    continue
                elif water_type == 4:  # Lake
                    depth = min(5.0, flow_accumulation[y, x] * 0.01)
                    water_map[y, x] = depth
                else:  # Flüsse
                    area = cross_section[y, x] if x < cross_section.shape[1] and y < cross_section.shape[0] else 0

                    if area > 0:
                        estimated_depth = area / 10.0
                        water_map[y, x] = min(3.0, estimated_depth)

        return water_map

    def _calculate_ocean_outflow(self, flow_accumulation, flow_directions, map_shape):
        """Berechnet Wasser-Abfluss ins Meer (an Kartenrändern)"""
        height, width = map_shape
        total_outflow = 0.0

        # Prüfe alle Rand-Pixel
        for y in range(height):
            for x in range(width):
                is_edge = (x == 0 or x == width - 1 or y == 0 or y == height - 1)

                if is_edge:
                    flow_dir = flow_directions[y, x]

                    if flow_dir >= 0:
                        direction_offsets = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
                        if flow_dir < len(direction_offsets):
                            dx, dy = direction_offsets[flow_dir]
                            target_x, target_y = x + dx, y + dy

                            if target_x < 0 or target_x >= width or target_y < 0 or target_y >= height:
                                total_outflow += flow_accumulation[y, x]

        return total_outflow

    def _create_minimal_water_data(self, target_size, lod, parameters):
        """Erstellt minimale WaterData bei Critical-Failures"""
        water_data = WaterData()
        water_data.water_map = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.flow_map = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.flow_speed = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.cross_section = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.soil_moist_map = np.ones((target_size, target_size), dtype=np.float32) * 30.0  # 30% default
        water_data.erosion_map = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.sedimentation_map = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.evaporation_map = np.zeros((target_size, target_size), dtype=np.float32)
        water_data.ocean_outflow = 0.0
        water_data.water_biomes_map = np.zeros((target_size, target_size), dtype=np.uint8)
        water_data.lod_level = lod
        water_data.actual_size = target_size
        water_data.parameters = parameters.copy()
        water_data.validity_state = "fallback"
        water_data.parameter_hash = self._calculate_parameter_hash(parameters)
        return water_data

    def _calculate_parameter_hash(self, parameters):
        """Berechnet Hash für Parameter-basierte Cache-Invalidation"""
        import hashlib
        param_str = str(sorted(parameters.items()))
        return hashlib.md5(param_str.encode()).hexdigest()

    def _update_progress(self, phase, percentage, message):
        """Progress-Update für UI-Integration"""
        if hasattr(self, 'progress_callback') and self.progress_callback:
            self.progress_callback(phase, percentage, message)

    def calculate_hydrology(self, dependencies: Dict[str, Any], parameters: Dict[str, Any],
                           lod_level: int, previous_erosion_map=None,
                           previous_sedimentation_map=None) -> WaterData:
        """
        Hauptmethode für Water-System-Generierung mit numerischem LOD-Level.

        Args:
            dependencies: dict mit heightmap, slopemap, hardness_map, rock_map,
                         precip_map, temp_map, wind_map, humid_map (bereits auf
                         lod_level vorskaliert vom DataLODManager)
            parameters: Alle Water-Parameter aus ParameterManager
            lod_level: Numerisches LOD-Level (1-6+)
            previous_erosion_map, previous_sedimentation_map: kumulierte Werte aus dem
                letzten abgeschlossenen Water-LOD (None beim allerersten Lauf) - werden
                zur in dieser Passage neu berechneten Erosion/Sedimentation addiert, damit
                die Landschaftsveränderung über LOD-Durchläufe hinweg iterativ akkumuliert
                statt bei jedem Lauf wieder bei der Basis-Heightmap neu zu starten.

        Returns:
            WaterData: Vollständiges Wassersystem mit allen 10 Outputs
        """
        return self._execute_generation(lod_level, dependencies, parameters,
                                       previous_erosion_map, previous_sedimentation_map)

    # ===== LEGACY-KOMPATIBILITÄT =====
    # Alle alten Methoden bleiben für Rückwärts-Kompatibilität erhalten

    def generate_hydrology_system(self, heightmap, slopemap, hardness_map, rock_map, precip_map, temp_map,
                                  wind_map, humid_map, lake_volume_threshold, rain_threshold, manning_coefficient,
                                  erosion_strength, sediment_capacity_factor, evaporation_base_rate,
                                  diffusion_radius, settling_velocity, map_seed):
        """Legacy-Methode für direkte Hydrologie-Generierung (KOMPATIBILITÄT)"""
        # Konvertiert alte API zur neuen API
        dependencies = {
            'heightmap': heightmap,
            'slopemap': slopemap,
            'hardness_map': hardness_map,
            'rock_map': rock_map,
            'precip_map': precip_map,
            'temp_map': temp_map,
            'wind_map': wind_map,
            'humid_map': humid_map
        }
        parameters = {
            'lake_volume_threshold': lake_volume_threshold,
            'rain_threshold': rain_threshold,
            'manning_coefficient': manning_coefficient,
            'erosion_strength': erosion_strength,
            'sediment_capacity_factor': sediment_capacity_factor,
            'evaporation_base_rate': evaporation_base_rate,
            'diffusion_radius': diffusion_radius,
            'settling_velocity': settling_velocity
        }

        # Seed aktualisieren falls nötig
        if map_seed is not None:
            self.update_seed(map_seed)

        water_data = self._execute_generation("LOD64", dependencies, parameters)

        # Legacy-Format zurückgeben (Tuple mit 11 Elementen)
        return (water_data.water_map, water_data.flow_map, water_data.flow_speed,
                water_data.cross_section, water_data.soil_moist_map, water_data.erosion_map,
                water_data.sedimentation_map, None, water_data.evaporation_map,  # rock_map_updated entfernt
                water_data.ocean_outflow, water_data.water_biomes_map)

    def simulate_water_cycle(self, current_hydrology, time_step=1.0):
        """Legacy-Methode für Water-Cycle-Updates"""
        if isinstance(current_hydrology, WaterData):
            soil_moist_map = current_hydrology.soil_moist_map
            evaporation_map = current_hydrology.evaporation_map
            erosion_map = current_hydrology.erosion_map
        else:
            # Legacy Tuple-Format
            (water_map, flow_map, flow_speed, cross_section, soil_moist_map,
             erosion_map, sedimentation_map, rock_map_updated, evaporation_map,
             ocean_outflow, water_biomes_map) = current_hydrology

        # Bodenfeuchtigkeit durch Evaporation reduzieren
        evap_loss = evaporation_map * time_step * 0.1
        new_soil_moist = np.maximum(0, soil_moist_map - evap_loss)

        # Erosion akkumulieren (sehr langsam)
        accumulated_erosion = erosion_map * time_step * 0.001

        if isinstance(current_hydrology, WaterData):
            # WaterData-Format aktualisieren
            updated_data = WaterData()
            updated_data.__dict__.update(current_hydrology.__dict__)
            updated_data.soil_moist_map = new_soil_moist
            updated_data.erosion_map = accumulated_erosion
            return updated_data
        else:
            # Legacy Tuple-Format zurückgeben
            return (water_map, flow_map, flow_speed, cross_section, new_soil_moist,
                    accumulated_erosion, sedimentation_map, rock_map_updated, evaporation_map,
                    ocean_outflow, water_biomes_map)

    def update_erosion_sedimentation(self, heightmap, rock_map, erosion_map, sedimentation_map, time_step=1.0):
        """Legacy-Methode für Erosion/Sedimentation-Updates"""
        net_height_change = (sedimentation_map - erosion_map) * time_step * 0.1
        new_heightmap = heightmap + net_height_change

        # Simplified mass conservation ohne komplexe Geology-Integration
        new_rock_map = rock_map  # Keep unchanged für Legacy-Kompatibilität

        return new_heightmap, new_rock_map

    def get_hydrology_statistics(self, hydrology_data):
        """Legacy-Methode für Hydrologie-Statistiken"""
        if isinstance(hydrology_data, WaterData):
            water_biomes_map = hydrology_data.water_biomes_map
            flow_map = hydrology_data.flow_map
            flow_speed = hydrology_data.flow_speed
            erosion_map = hydrology_data.erosion_map
            sedimentation_map = hydrology_data.sedimentation_map
            soil_moist_map = hydrology_data.soil_moist_map
            evaporation_map = hydrology_data.evaporation_map
            ocean_outflow = hydrology_data.ocean_outflow
        else:
            # Legacy Tuple-Format
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
            }
        }

        return stats

    def validate_mass_conservation(self, rock_map_original, rock_map_updated):
        """Legacy-Methode für Massenerhaltungs-Validation"""
        if rock_map_original is None or rock_map_updated is None:
            return {
                'original_mass_conservation': False,
                'updated_mass_conservation': False,
                'total_mass_difference': 0.0,
                'mass_conservation_ratio': 1.0,
                'invalid_pixels': 0
            }

        original_sums = np.sum(rock_map_original, axis=2)
        updated_sums = np.sum(rock_map_updated, axis=2)

        original_valid = np.all(original_sums == 255)
        updated_valid = np.all(updated_sums == 255)

        total_original = np.sum(rock_map_original)
        total_updated = np.sum(rock_map_updated)
        mass_difference = abs(total_updated - total_original)

        results = {
            'original_mass_conservation': original_valid,
            'updated_mass_conservation': updated_valid,
            'total_mass_difference': float(mass_difference),
            'mass_conservation_ratio': float(total_updated / total_original) if total_original > 0 else 1.0,
            'invalid_pixels': int(np.sum(updated_sums != 255))
        }

        return results