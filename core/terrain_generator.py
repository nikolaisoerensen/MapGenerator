"""
Path: core/terrain_generator.py

Funktionsweise: Simplex-Noise basierte Terrain-Generierung
- BaseTerrainGenerator mit Octaves/Persistence/Lacunarity
- Multi-Scale Noise-Layering für realistische Landschaften
- Höhen-Redistribution für natürliche Höhenverteilung
- Deformation mittels ridge warping
- Performance-optimiert für Live-Preview (low LOD), button-calculated-Preview (medium LOD) und final-Preview (high LOD) und Auslagerung auf GPU mit Shader
- Berechnung der Verschattung mit Raycasts und 6 Sonnenwinkeln. Die 6 Sonnenwinkel können mit Slider durchgeschaltet werden

Parameter Input:
- map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power, map_seed

Output:
- heightmap array
- slopemap 2D array (dz/dx and dz/dy)
- shademap array

Klassen:
BaseTerrainGenerator
    Funktionsweise: Hauptklasse für Simplex-Noise basierte Terrain-Generierung mit Multi-Scale Layering
    Aufgabe: Koordiniert alle Terrain-Generierungsschritte und verwaltet Parameter
    Methoden: generate_heightmap(), apply_redistribution(), calculate_slopes(), generate_shadows()

SimplexNoiseGenerator   
    Funktionsweise: Erzeugt OpenSimplex-Noise mit konfigurierbaren Octaves/Persistence/Lacunarity
    Aufgabe: Basis-Noise-Funktionen für alle anderen Module
    Methoden: noise_2d(), multi_octave_noise(), ridge_noise()

ShadowCalculator    
    Funktionsweise: Berechnet Verschattung mit Raycasts für 6 verschiedene Sonnenwinkel
    Aufgabe: Erstellt shademap für Weather-System und visuelle Darstellung
    Methoden: calculate_shadows_multi_angle(), raycast_shadow(), combine_shadow_angles()
"""

import numpy as np
from opensimplex import OpenSimplex
import threading
import time
import os
from core.base_generator import BaseGenerator

class TerrainData:
    """
    Funktionsweise: Container für alle Terrain-Daten mit Metainformationen
    Aufgabe: Speichert Heightmap, Slopemap, Shadowmap und LOD-Informationen
    """

    def __init__(self):
        self.heightmap = None
        self.slopemap = None
        self.shadowmap = None
        self.lod_level = "LOD64"
        self.actual_size = 64
        self.calculated_sun_angles = []  # Welche Sonnenwinkel bereits berechnet
        self.parameters = {}  # Speichert verwendete Parameter

class SimplexNoiseGenerator:
    """
    Funktionsweise: Erzeugt OpenSimplex-Noise mit konfigurierbaren Octaves/Persistence/Lacunarity
    Aufgabe: Basis-Noise-Funktionen für alle anderen Module
    """

    def __init__(self, seed=42):
        """
        Funktionsweise: Initialisiert OpenSimplex-Generator mit gegebenem Seed
        Aufgabe: Setup des Noise-Generators für reproduzierbare Ergebnisse
        Parameter: seed (int) - Seed für reproduzierbaren Noise
        """
        self.generator = OpenSimplex(seed=seed)

    def noise_2d(self, x, y):
        """
        Funktionsweise: Erzeugt einzelnen Noise-Wert für gegebene 2D-Koordinaten
        Aufgabe: Basis-Noise-Funktion für einzelne Punkte
        Parameter: x, y (float) - 2D-Koordinaten
        Returns: float - Noise-Wert zwischen -1 und 1
        """
        return self.generator.noise2(x, y)

    def multi_octave_noise(self, x, y, octaves, persistence, lacunarity, frequency):
        """
        Funktionsweise: Erzeugt Multi-Octave Noise durch Kombination mehrerer Noise-Layer
        Aufgabe: Komplexere Noise-Patterns durch Octave-Layering
        Parameter: x, y, octaves, persistence, lacunarity, frequency - Noise-Parameter
        Returns: float - Kombinierter Noise-Wert
        """
        value = 0.0
        amplitude = 1.0
        current_frequency = frequency
        max_amplitude = 0.0

        for _ in range(octaves):
            value += amplitude * self.noise_2d(x * current_frequency, y * current_frequency)
            max_amplitude += amplitude
            amplitude *= persistence
            current_frequency *= lacunarity

        # Normalisierung auf [-1, 1]
        return value / max_amplitude if max_amplitude > 0 else 0

    def ridge_noise(self, x, y, octaves, persistence, lacunarity, frequency):
        """
        Funktionsweise: Erzeugt Ridge-Noise für scharfe Gebirgskämme und Täler
        Aufgabe: Spezialisierte Noise-Variante für dramatische Terrain-Features
        Parameter: x, y, octaves, persistence, lacunarity, frequency - Noise-Parameter
        Returns: float - Ridge-Noise-Wert
        """
        value = 0.0
        amplitude = 1.0
        current_frequency = frequency
        max_amplitude = 0.0

        for _ in range(octaves):
            noise_val = abs(self.noise_2d(x * current_frequency, y * current_frequency))
            ridge_val = 1.0 - noise_val  # Invertierung für Ridge-Effekt
            value += amplitude * ridge_val
            max_amplitude += amplitude
            amplitude *= persistence
            current_frequency *= lacunarity

        return value / max_amplitude if max_amplitude > 0 else 0


    def generate_noise_grid(self, size, frequency, octaves, persistence, lacunarity, offset_x=0, offset_y=0):
        """
        Funktionsweise: Generiert komplettes Noise-Grid auf einmal statt Pixel für Pixel
        Aufgabe: Performance-Optimierung durch Batch-Verarbeitung ohne Python-Loops
        Parameter: size (int) - Grid-Größe, frequency/octaves/persistence/lacunarity - Noise-Parameter
        Parameter: offset_x/offset_y (float) - Verschiebung für nahtlose Kacheln
        Returns: numpy.ndarray - Komplettes Noise-Grid mit Werten zwischen -1 und 1
        """
        # Koordinaten-Arrays für gesamtes Grid erstellen
        x_coords = np.linspace(0, 1, size) + offset_x
        y_coords = np.linspace(0, 1, size) + offset_y

        # Meshgrid für vektorisierte Berechnung
        X, Y = np.meshgrid(x_coords, y_coords)

        # Ergebnis-Array initialisieren
        noise_grid = np.zeros((size, size), dtype=np.float32)

        # Multi-Octave Berechnung
        amplitude = 1.0
        current_frequency = frequency
        max_amplitude = 0.0

        for octave in range(octaves):
            # Aktuelle Frequenz-Koordinaten
            freq_X = X * current_frequency
            freq_Y = Y * current_frequency

            # Noise für gesamtes Grid berechnen (vektorisiert)
            octave_noise = np.zeros_like(noise_grid)
            for y in range(size):
                for x in range(size):
                    octave_noise[y, x] = self.generator.noise2(freq_X[y, x], freq_Y[y, x])

            # Octave zum Gesamtergebnis hinzufügen
            noise_grid += amplitude * octave_noise
            max_amplitude += amplitude

            # Parameter für nächste Octave
            amplitude *= persistence
            current_frequency *= lacunarity

        # Normalisierung auf [-1, 1]
        if max_amplitude > 0:
            noise_grid /= max_amplitude

        return noise_grid


    def interpolate_existing_grid(self, existing_grid, new_size):
        """
        Funktionsweise: Interpoliert bestehende LOD-Daten auf höhere Auflösung mittels bilinearer Interpolation
        Aufgabe: Progressive LOD-Verbesserung ohne Neuberechnung aller Werte
        Parameter: existing_grid (numpy.ndarray) - Bestehende niedrig-aufgelöste Daten
        Parameter: new_size (int) - Zielgröße für Interpolation
        Returns: numpy.ndarray - Interpolierte Daten in neuer Auflösung
        """
        old_size = existing_grid.shape[0]

        # Wenn Größen gleich sind, einfach kopieren
        if old_size == new_size:
            return existing_grid.copy()

        # Skalierungsfaktor berechnen
        scale_factor = (old_size - 1) / (new_size - 1)

        # Neues Grid initialisieren
        interpolated_grid = np.zeros((new_size, new_size), dtype=np.float32)

        # Für jeden Punkt im neuen Grid
        for new_y in range(new_size):
            for new_x in range(new_size):
                # Entsprechende Position im alten Grid
                old_x = new_x * scale_factor
                old_y = new_y * scale_factor

                # Ganzzahlige Koordinaten für Interpolation
                x0 = int(old_x)
                y0 = int(old_y)
                x1 = min(x0 + 1, old_size - 1)
                y1 = min(y0 + 1, old_size - 1)

                # Interpolations-Gewichte
                fx = old_x - x0
                fy = old_y - y0

                # Bilineare Interpolation
                h00 = existing_grid[y0, x0]
                h10 = existing_grid[y0, x1]
                h01 = existing_grid[y1, x0]
                h11 = existing_grid[y1, x1]

                # Interpolation in x-Richtung
                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                # Interpolation in y-Richtung
                interpolated_grid[new_y, new_x] = h0 * (1 - fy) + h1 * fy

        return interpolated_grid


    def add_detail_noise(self, base_grid, detail_frequency, detail_amplitude):
        """
        Funktionsweise: Fügt hochfrequente Detail-Noise zu bestehender interpolierter Basis hinzu
        Aufgabe: Verfeinert interpolierte LOD-Daten mit lokalen Details
        Parameter: base_grid (numpy.ndarray) - Basis-Grid aus Interpolation
        Parameter: detail_frequency (float) - Frequenz für Detail-Noise
        Parameter: detail_amplitude (float) - Stärke der Detail-Noise (meist 10-30% der Original-Amplitude)
        Returns: numpy.ndarray - Verfeinertes Grid mit Details
        """
        size = base_grid.shape[0]

        # Detail-Noise mit höherer Frequenz generieren
        detail_grid = self.generate_noise_grid(
            size=size,
            frequency=detail_frequency,
            octaves=2,  # Weniger Octaves für Details
            persistence=0.5,
            lacunarity=2.0,
            offset_x=0,
            offset_y=0
        )

        # Detail-Noise mit reduzierter Amplitude zur Basis hinzufügen
        enhanced_grid = base_grid + (detail_grid * detail_amplitude)

        return enhanced_grid


class ShadowCalculator:
    """
    Funktionsweise: Berechnet Verschattung mit Raycasts für 6 verschiedene Sonnenwinkel
    Aufgabe: Erstellt shademap für Weather-System und visuelle Darstellung
    """

    def __init__(self):
        """
        Funktionsweise: Initialisiert Shadow-Calculator mit Standard-Sonnenwinkel-Konfiguration
        Aufgabe: Setup der 7 Sonnenwinkel für Tagesverlauf-Simulation
        """
        # 7 Sonnenwinkel für Tagesverlauf (in Grad)
        self.sun_angles = [
            (10, 75),  # Morgendämmerung
            (25, 90),  # Morgen
            (45, 120),  # Vormittag
            (70, 180),  # Mittag
            (45, 240),  # Nachmittag
            (25, 270),  # Abend
            (10, 285)  # Späte Dämmerung
        ]

        # Gewichtung durch atmosphärische Durchdringung (Mittag erhält höchste Gewichtung)
        self.sun_weights = [0.06, 0.2, 0.6, 0.9, 0.6, 0.2, 0.06]

    def calculate_shadows_multi_angle(self, heightmap):
        """
        Funktionsweise: Berechnet Verschattung für alle 6 Sonnenwinkel und kombiniert sie
        Aufgabe: Erstellt realistische Verschattung durch mehrere Sonnenwinkel
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: numpy.ndarray - Kombinierte Verschattung (0=Schatten, 1=Vollsonne)
        """
        height, width = heightmap.shape
        combined_shadows = np.zeros((height, width), dtype=np.float32)
        total_weight = sum(self.sun_weights)

        for i, (elevation, azimuth) in enumerate(self.sun_angles):
            shadow_map = self.raycast_shadow(heightmap, elevation, azimuth)
            combined_shadows += shadow_map * self.sun_weights[i]

        # Normalisierung auf [0, 1]
        combined_shadows /= total_weight
        return combined_shadows

    def calculate_shadows_with_lod(self, heightmap, lod_level="LOD64", shadow_resolution=64):
        """
        Funktionsweise: Berechnet Verschattung mit LOD-System und fester Shadow-Auflösung
        Aufgabe: Optimierte Shadow-Berechnung - Shadows immer in niedriger Auflösung, dann interpoliert
        Parameter: heightmap (numpy.ndarray) - Höhendaten in beliebiger Auflösung
        Parameter: lod_level (str) - LOD-Level für Sonnenwinkel-Auswahl
        Parameter: shadow_resolution (int) - Feste Auflösung für Shadow-Berechnung (Standard: 64)
        Returns: numpy.ndarray - Shadow-Map in gleicher Auflösung wie heightmap
        """
        original_size = heightmap.shape[0]

        # Heightmap für Shadow-Berechnung auf niedrige Auflösung reduzieren
        if original_size > shadow_resolution:
            # Downscale heightmap für Shadow-Berechnung
            scale_factor = (original_size - 1) / (shadow_resolution - 1)
            shadow_heightmap = np.zeros((shadow_resolution, shadow_resolution), dtype=np.float32)

            for y in range(shadow_resolution):
                for x in range(shadow_resolution):
                    # Position im Original-Heightmap
                    orig_x = x * scale_factor
                    orig_y = y * scale_factor

                    # Bilineare Interpolation für Downscaling
                    shadow_heightmap[y, x] = self._interpolate_height(heightmap, orig_x, orig_y)
        else:
            shadow_heightmap = heightmap

        # Passende Sonnenwinkel für LOD holen
        sun_angles, sun_weights = self.get_sun_angles_for_lod(lod_level)

        # Shadow-Berechnung in niedriger Auflösung
        low_res_shadows = np.zeros((shadow_heightmap.shape[0], shadow_heightmap.shape[1]), dtype=np.float32)
        total_weight = sum(sun_weights)

        for i, (elevation, azimuth) in enumerate(sun_angles):
            shadow_map = self.raycast_shadow(shadow_heightmap, elevation, azimuth)
            low_res_shadows += shadow_map * sun_weights[i]

        # Normalisierung
        low_res_shadows /= total_weight

        # Upscale auf Original-Größe falls nötig
        if original_size > shadow_resolution:
            final_shadows = self._upscale_shadows(low_res_shadows, original_size)
        else:
            final_shadows = low_res_shadows

        return final_shadows

    def get_sun_angles_for_lod(self, lod_level):
        """
        Funktionsweise: Gibt passende Sonnenwinkel-Auswahl für LOD-Level zurück
        Aufgabe: Performance-Optimierung durch weniger Sonnenwinkel bei niedrigen LODs
        Parameter: lod_level (str) - "LOD64", "LOD128", "LOD256", "FINAL"
        Returns: Tuple (sun_angles_list, sun_weights_list) - Gefilterte Winkel und Gewichtungen
        """
        if lod_level == "LOD64":
            # Nur Mittag
            indices = [3]  # (70, 180)
        elif lod_level == "LOD128":
            # Mittag + Vormittag + Nachmittag
            indices = [2, 3, 4]  # (45, 120), (70, 180), (45, 240)
        elif lod_level == "LOD256":
            # + Morgen + Abend
            indices = [1, 2, 3, 4, 5]  # (25, 90), (45, 120), (70, 180), (45, 240), (25, 270)
        else:  # FINAL
            # Alle 7 Winkel
            indices = list(range(7))

        selected_angles = [self.sun_angles[i] for i in indices]
        selected_weights = [self.sun_weights[i] for i in indices]

        return selected_angles, selected_weights

    def calculate_shadows_progressive(self, heightmap, lod_level, existing_shadows=None, existing_lod="LOD64"):
        """
        Funktionsweise: Berechnet nur neue Sonnenwinkel und kombiniert mit bestehenden Shadows
        Aufgabe: Progressive Shadow-Verbesserung ohne Neuberechnung aller Winkel
        Parameter: heightmap (numpy.ndarray) - Aktuelle Höhendaten
        Parameter: lod_level (str) - Ziel-LOD-Level
        Parameter: existing_shadows (numpy.ndarray) - Bestehende Shadow-Daten (optional)
        Parameter: existing_lod (str) - LOD-Level der bestehenden Shadows
        Returns: numpy.ndarray - Erweiterte Shadow-Map
        """
        # Neue Sonnenwinkel für Ziel-LOD
        new_angles, new_weights = self.get_sun_angles_for_lod(lod_level)

        # Bestehende Sonnenwinkel
        if existing_shadows is not None:
            old_angles, old_weights = self.get_sun_angles_for_lod(existing_lod)

            # Finde nur die NEUEN Winkel (die noch nicht berechnet wurden)
            old_angle_set = set(old_angles)
            additional_angles = []
            additional_weights = []

            for angle, weight in zip(new_angles, new_weights):
                if angle not in old_angle_set:
                    additional_angles.append(angle)
                    additional_weights.append(weight)

            if not additional_angles:
                # Keine neuen Winkel, nur bestehende Shadows zurückgeben
                return existing_shadows

            # Berechne nur die zusätzlichen Winkel
            additional_shadows = np.zeros_like(heightmap, dtype=np.float32)
            for elevation, azimuth in additional_angles:
                shadow_map = self.raycast_shadow(heightmap, elevation, azimuth)
                additional_shadows += shadow_map

            # Normiere die zusätzlichen Shadows
            if additional_weights:
                additional_shadows /= len(additional_weights)

            # Kombiniere bestehende und neue Shadows
            total_old_weight = sum(old_weights)
            total_additional_weight = sum(additional_weights)
            total_weight = total_old_weight + total_additional_weight

            combined_shadows = (
                                           existing_shadows * total_old_weight + additional_shadows * total_additional_weight) / total_weight

            return combined_shadows
        else:
            # Keine bestehenden Shadows, normale Berechnung
            return self.calculate_shadows_with_lod(heightmap, lod_level)

    def raycast_shadow(self, heightmap, sun_elevation, sun_azimuth):
        """
        Funktionsweise: Berechnet Verschattung für einen einzelnen Sonnenwinkel mit Raycasting
        Aufgabe: Raycast-basierte Verschattungsberechnung für gegebenen Sonnenstand
        Parameter: heightmap, sun_elevation, sun_azimuth - Höhendaten und Sonnenposition
        Returns: numpy.ndarray - Verschattung für diesen Sonnenwinkel
        """
        height, width = heightmap.shape
        shadow_map = np.ones((height, width), dtype=np.float32)

        # Sonnenrichtung berechnen
        elevation_rad = np.radians(sun_elevation)
        azimuth_rad = np.radians(sun_azimuth)

        sun_x = np.cos(elevation_rad) * np.sin(azimuth_rad)
        sun_y = np.cos(elevation_rad) * np.cos(azimuth_rad)
        sun_z = np.sin(elevation_rad)

        # Raycast für jeden Pixel
        for y in range(height):
            for x in range(width):
                if self._is_in_shadow(heightmap, x, y, sun_x, sun_y, sun_z):
                    shadow_map[y, x] = 0.0  # Vollschatten
                else:
                    # Berechne partielle Verschattung basierend auf Neigung
                    slope_factor = self._calculate_slope_shading(heightmap, x, y, sun_x, sun_y, sun_z)
                    shadow_map[y, x] = slope_factor

        return shadow_map

    def _upscale_shadows(self, low_res_shadows, target_size):
        """
        Funktionsweise: Skaliert Shadow-Map von niedriger auf hohe Auflösung mittels bilinearer Interpolation
        Aufgabe: Effiziente Shadow-Interpolation für finale Darstellung
        Parameter: low_res_shadows (numpy.ndarray) - Shadow-Daten in niedriger Auflösung
        Parameter: target_size (int) - Zielgröße für Upscaling
        Returns: numpy.ndarray - Interpolierte Shadow-Map in Zielauflösung
        """
        low_size = low_res_shadows.shape[0]
        scale_factor = (low_size - 1) / (target_size - 1)

        upscaled_shadows = np.zeros((target_size, target_size), dtype=np.float32)

        for y in range(target_size):
            for x in range(target_size):
                # Position im Low-Res Shadow-Map
                low_x = x * scale_factor
                low_y = y * scale_factor

                # Ganzzahlige Koordinaten
                x0, y0 = int(low_x), int(low_y)
                x1, y1 = min(x0 + 1, low_size - 1), min(y0 + 1, low_size - 1)

                # Interpolations-Gewichte
                fx = low_x - x0
                fy = low_y - y0

                # Bilineare Interpolation
                s00 = low_res_shadows[y0, x0]
                s10 = low_res_shadows[y0, x1]
                s01 = low_res_shadows[y1, x0]
                s11 = low_res_shadows[y1, x1]

                s0 = s00 * (1 - fx) + s10 * fx
                s1 = s01 * (1 - fx) + s11 * fx

                upscaled_shadows[y, x] = s0 * (1 - fy) + s1 * fy

        return upscaled_shadows

    def _is_in_shadow(self, heightmap, x, y, sun_x, sun_y, sun_z):
        """
        Funktionsweise: Prüft ob ein Pixel durch Raycast im Schatten liegt
        Aufgabe: Raycast-Test von Pixel zur Sonne mit Terrain-Kollision
        Parameter: heightmap, x, y, sun_x, sun_y, sun_z - Terrain und Raycast-Parameter
        Returns: bool - True wenn Pixel im Schatten liegt
        """
        height, width = heightmap.shape
        current_height = heightmap[y, x]

        # Raycast-Parameter
        step_size = 0.5
        max_distance = max(width, height) * 2

        # Raycast entlang Sonnenrichtung
        for distance in np.arange(step_size, max_distance, step_size):
            # Aktuelle Position im Raycast
            ray_x = x + sun_x * distance
            ray_y = y + sun_y * distance
            ray_z = current_height + sun_z * distance

            # Prüfe ob Ray außerhalb der Map ist
            if ray_x < 0 or ray_x >= width or ray_y < 0 or ray_y >= height:
                break

            # Interpoliere Höhe an aktueller Ray-Position
            terrain_height = self._interpolate_height(heightmap, ray_x, ray_y)

            # Kollision mit Terrain?
            if ray_z <= terrain_height:
                return True  # Im Schatten

        return False  # Nicht im Schatten

    def _interpolate_height(self, heightmap, x, y):
        """
        Funktionsweise: Interpoliert Höhenwert an nicht-ganzzahligen Koordinaten
        Aufgabe: Bilineare Interpolation für smooth Raycast-Kollision
        Parameter: heightmap, x, y - Terrain und Interpolations-Koordinaten
        Returns: float - Interpolierte Höhe
        """
        height, width = heightmap.shape

        # Begrenze Koordinaten
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        # Ganzzahlige Koordinaten
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

        # Interpolations-Gewichte
        fx = x - x0
        fy = y - y0

        # Bilineare Interpolation
        h00 = heightmap[y0, x0]
        h10 = heightmap[y0, x1]
        h01 = heightmap[y1, x0]
        h11 = heightmap[y1, x1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fy) + h1 * fy

    def _calculate_slope_shading(self, heightmap, x, y, sun_x, sun_y, sun_z):
        """
        Funktionsweise: Berechnet Verschattung basierend auf Oberflächenneigung
        Aufgabe: Slope-basierte Shading für realistische Beleuchtung
        Parameter: heightmap, x, y, sun_x, sun_y, sun_z - Terrain und Licht-Parameter
        Returns: float - Shading-Faktor zwischen 0 und 1
        """
        height, width = heightmap.shape

        # Berechne Oberflächennormale
        if x > 0 and x < width - 1 and y > 0 and y < height - 1:
            dz_dx = (heightmap[y, x + 1] - heightmap[y, x - 1]) * 0.5
            dz_dy = (heightmap[y + 1, x] - heightmap[y - 1, x]) * 0.5
        else:
            dz_dx = 0
            dz_dy = 0

        # Normale berechnen
        normal = np.array([-dz_dx, -dz_dy, 1.0])
        normal = normal / np.linalg.norm(normal)

        # Sonnenrichtung
        sun_dir = np.array([sun_x, sun_y, sun_z])
        sun_dir = sun_dir / np.linalg.norm(sun_dir)

        # Dot-Product für Beleuchtungsstärke
        dot_product = np.dot(normal, sun_dir)

        # Clamp auf [0, 1]
        return max(0.0, dot_product)

    def combine_shadow_angles(self, shadow_maps, weights=None):
        """
        Funktionsweise: Kombiniert mehrere Schatten-Maps zu einer finalen Verschattung
        Aufgabe: Gewichtete Kombination verschiedener Sonnenwinkel
        Parameter: shadow_maps (list), weights (list) - Schatten-Maps und Gewichtungen
        Returns: numpy.ndarray - Kombinierte Verschattung
        """
        if weights is None:
            weights = [1.0] * len(shadow_maps)

        combined = np.zeros_like(shadow_maps[0], dtype=np.float32)
        total_weight = sum(weights)

        for shadow_map, weight in zip(shadow_maps, weights):
            combined += shadow_map * weight

        return combined / total_weight if total_weight > 0 else combined

class BaseTerrainGenerator(BaseGenerator):
    """
    Funktionsweise: Hauptklasse für Simplex-Noise basierte Terrain-Generierung mit Multi-Scale Layering und LOD-System
    Aufgabe: Koordiniert alle Terrain-Generierungsschritte, verwaltet Parameter und Threading
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Terrain-Generator mit allen Sub-Komponenten und LOD-System
        Aufgabe: Setup von Noise-Generator, Shadow-Calculator und Threading-System
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Ergebnisse
        """
        super().__init__(map_seed)

        self.noise_generator = SimplexNoiseGenerator(seed=map_seed)
        self.shadow_calculator = ShadowCalculator()

        # Standard-Parameter
        self.map_size = 256
        self.amplitude = 100
        self.octaves = 4
        self.frequency = 0.01
        self.persistence = 0.5
        self.lacunarity = 2.0
        self.redistribute_power = 1.0

        # LOD-System
        self.lod_sizes = {"LOD64": 64, "LOD128": 128, "LOD256": 256}
        self.current_terrain_data = None

        # Threading-System
        self.is_calculating = False
        self.calculation_thread = None
        self.progress_callback = None

    def generate_terrain(self, lod="LOD64", progress=None, background=False,
                         existing_data=None, map_size=None, amplitude=None, octaves=None,
                         frequency=None, persistence=None, lacunarity=None,
                         redistribute_power=None, map_seed=None):
        """
        Funktionsweise: Einheitliche Terrain-Generierung mit LOD-Support und Threading
        Aufgabe: Hauptmethode für alle Terrain-Generierung mit progressiver Verbesserung
        Parameter: lod ("LOD64"/"LOD128"/"LOD256"/"FINAL") - LOD-Level
        Parameter: progress (function) - Callback für Fortschritts-Updates
        Parameter: background (bool) - Threading mit niedriger CPU-Priorität
        Parameter: existing_data (TerrainData) - Bestehende Daten für progressive Verbesserung
        Parameter: map_size, amplitude, etc. - Überschreibt Standard-Parameter wenn gegeben
        Returns: TerrainData - Komplette Terrain-Daten
        """
        # Parameter aktualisieren falls übergeben
        self._update_parameters(map_size, amplitude, octaves, frequency,
                                persistence, lacunarity, redistribute_power, map_seed)

        # Threading-Modus
        if background:
            return self._generate_terrain_threaded(lod, progress, existing_data)
        else:
            return self._generate_terrain_direct(lod, progress, existing_data)

    def _generate_terrain_direct(self, lod, progress, existing_data):
        """
        Funktionsweise: Direkte Terrain-Generierung im Hauptthread
        Aufgabe: Synchrone Berechnung mit Progress-Updates
        Parameter: lod, progress, existing_data - siehe generate_terrain()
        Returns: TerrainData - Generierte Terrain-Daten
        """
        self.progress_callback = progress
        self.is_calculating = True

        try:
            # Zielgröße bestimmen
            if lod == "FINAL":
                target_size = self.map_size
            else:
                target_size = self.lod_sizes[lod]

            # Progress-Update
            self._update_progress("Initialization", 0, f"Starting {lod} generation (size: {target_size})")

            # Neue TerrainData erstellen oder bestehende erweitern
            if existing_data is None:
                terrain_data = TerrainData()
                terrain_data.parameters = self._get_current_parameters()
            else:
                terrain_data = existing_data

            # 1. Heightmap generieren/verbessern
            self._update_progress("Heightmap", 10, "Generating heightmap...")
            terrain_data.heightmap = self._progressive_heightmap_generation(
                target_size, terrain_data.heightmap if existing_data else None
            )

            # 2. Redistribution anwenden
            self._update_progress("Redistribution", 30, "Applying height redistribution...")
            terrain_data.heightmap = self.apply_redistribution(
                terrain_data.heightmap, self.redistribute_power
            )

            # 3. Slopemap berechnen
            self._update_progress("Slopes", 50, "Calculating slope map...")
            terrain_data.slopemap = self.calculate_slopes(terrain_data.heightmap)

            # 4. Shadowmap generieren/erweitern
            self._update_progress("Shadows", 70, f"Calculating shadows for {lod}...")
            terrain_data.shadowmap = self._progressive_shadow_generation(
                terrain_data.heightmap, lod, terrain_data.shadowmap if existing_data else None,
                terrain_data.lod_level if existing_data else "LOD64"
            )

            # 5. Metadaten aktualisieren
            terrain_data.lod_level = lod
            terrain_data.actual_size = target_size
            terrain_data.calculated_sun_angles = self.shadow_calculator.get_sun_angles_for_lod(lod)[0]

            self._update_progress("Complete", 100, f"{lod} terrain generation complete!")
            self.current_terrain_data = terrain_data

            return terrain_data

        finally:
            self.is_calculating = False

    def _generate_terrain_threaded(self, lod, progress, existing_data):
        """
        Funktionsweise: Terrain-Generierung in separatem Thread mit niedriger Priorität
        Aufgabe: Hintergrund-Berechnung ohne GUI-Blocking
        Parameter: lod, progress, existing_data - siehe generate_terrain()
        Returns: threading.Thread - Thread-Objekt für Kontrolle
        """

        def worker():
            # Thread-Priorität reduzieren (falls unterstützt)
            try:
                if hasattr(os, 'nice'):
                    os.nice(10)  # Unix/Linux: Niedrigere Priorität
            except:
                pass

            # Terrain-Generierung mit CPU-Yields
            result = self._generate_terrain_with_yields(lod, progress, existing_data)
            return result

        # Thread starten
        self.calculation_thread = threading.Thread(target=worker, daemon=True)
        self.calculation_thread.start()

        return self.calculation_thread

    def _generate_terrain_with_yields(self, lod, progress, existing_data):
        """
        Funktionsweise: Terrain-Generierung mit regelmäßigen CPU-Yields
        Aufgabe: Hintergrund-Berechnung mit verbesserter System-Responsivität
        Parameter: lod, progress, existing_data - siehe generate_terrain()
        Returns: TerrainData - Generierte Terrain-Daten
        """
        self.progress_callback = progress
        self.is_calculating = True

        try:
            # Zielgröße bestimmen
            if lod == "FINAL":
                target_size = self.map_size
            else:
                target_size = self.lod_sizes[lod]

            self._update_progress("Initialization", 0, f"Starting background {lod} generation")
            self._yield_cpu()

            # Terrain-Daten vorbereiten
            if existing_data is None:
                terrain_data = TerrainData()
                terrain_data.parameters = self._get_current_parameters()
            else:
                terrain_data = existing_data

            # 1. Heightmap mit Yields
            self._update_progress("Heightmap", 10, "Generating heightmap in background...")
            terrain_data.heightmap = self._progressive_heightmap_generation_with_yields(
                target_size, terrain_data.heightmap if existing_data else None
            )

            # 2. Redistribution
            self._update_progress("Redistribution", 30, "Applying redistribution...")
            terrain_data.heightmap = self.apply_redistribution(
                terrain_data.heightmap, self.redistribute_power
            )
            self._yield_cpu()

            # 3. Slopes
            self._update_progress("Slopes", 50, "Calculating slopes...")
            terrain_data.slopemap = self._calculate_slopes_with_yields(terrain_data.heightmap)

            # 4. Shadows
            self._update_progress("Shadows", 70, "Calculating shadows...")
            terrain_data.shadowmap = self._progressive_shadow_generation(
                terrain_data.heightmap, lod, terrain_data.shadowmap if existing_data else None,
                terrain_data.lod_level if existing_data else "LOD64"
            )

            # 5. Finalisierung
            terrain_data.lod_level = lod
            terrain_data.actual_size = target_size
            terrain_data.calculated_sun_angles = self.shadow_calculator.get_sun_angles_for_lod(lod)[0]

            self._update_progress("Complete", 100, f"Background {lod} generation complete!")
            self.current_terrain_data = terrain_data

            return terrain_data

        finally:
            self.is_calculating = False

    def _load_default_parameters(self):
        """
        Funktionsweise: Lädt TERRAIN-Parameter aus value_default.py
        Aufgabe: Standard-Parameter für Terrain-Generierung
        Returns: dict - Alle Standard-Parameter für Terrain
        """
        from gui.config.value_default import TERRAIN

        return {
            'map_size': TERRAIN.SIZE["default"],
            'amplitude': TERRAIN.AMPLITUDE["default"],
            'octaves': TERRAIN.OCTAVES["default"],
            'frequency': TERRAIN.FREQUENCY["default"],
            'persistence': TERRAIN.PERSISTENCE["default"],
            'lacunarity': TERRAIN.LACUNARITY["default"],
            'redistribute_power': TERRAIN.REDISTRIBUTE_POWER["default"],
            'map_seed': TERRAIN.MAP_SEED["default"]
        }

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Terrain braucht keine Dependencies - ist der erste Generator
        Aufgabe: Leere Dependencies für Basis-Generator
        Parameter: data_manager - DataManager-Instanz (wird nicht verwendet)
        Returns: dict - Leeres Dependencies-Dict
        """
        # Terrain ist der erste Generator und braucht keine Dependencies
        return {}

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Funktionsweise: Führt Terrain-Generierung mit bestehender Logik aus
        Aufgabe: Wrapper um bestehende Terrain-Generation mit Progress-Updates
        Parameter: lod, dependencies, parameters
        Returns: TerrainData-Objekt
        """
        # Parameter aktualisieren
        self._update_parameters(
            map_size=parameters['map_size'],
            amplitude=parameters['amplitude'],
            octaves=parameters['octaves'],
            frequency=parameters['frequency'],
            persistence=parameters['persistence'],
            lacunarity=parameters['lacunarity'],
            redistribute_power=parameters['redistribute_power'],
            map_seed=parameters['map_seed']
        )

        # Progress-Updates für BaseGenerator-Kompatibilität
        def terrain_progress_wrapper(step_name, progress_percent, detail_message):
            # Konvertiere Terrain-spezifische Progress zu BaseGenerator-Progress
            base_progress = int(15 + (progress_percent * 0.8))  # 15-95% Bereich
            self._update_progress(step_name, base_progress, detail_message)

        # Bestehende generate_terrain Methode verwenden
        terrain_data = self.generate_terrain(
            lod=lod,
            progress=terrain_progress_wrapper,
            background=False,
            existing_data=None,
            map_size=parameters['map_size'],
            amplitude=parameters['amplitude'],
            octaves=parameters['octaves'],
            frequency=parameters['frequency'],
            persistence=parameters['persistence'],
            lacunarity=parameters['lacunarity'],
            redistribute_power=parameters['redistribute_power'],
            map_seed=parameters['map_seed']
        )

        return terrain_data

    def _save_to_data_manager(self, data_manager, result, parameters):
        """
        Funktionsweise: Speichert TerrainData-Objekt im DataManager
        Aufgabe: Automatische Speicherung aller Terrain-Outputs
        Parameter: data_manager, result (TerrainData), parameters
        """
        if isinstance(result, TerrainData):
            # Verwende die neue set_terrain_data_complete Methode
            data_manager.set_terrain_data_complete(result, parameters)
            self.logger.debug("TerrainData object saved to DataManager")
        else:
            # Fallback für Legacy-Format
            if hasattr(result, '__len__') and len(result) == 3:
                heightmap, slopemap, shadowmap = result
                data_manager.set_terrain_data("heightmap", heightmap, parameters)
                data_manager.set_terrain_data("slopemap", slopemap, parameters)
                data_manager.set_terrain_data("shadowmap", shadowmap, parameters)
                self.logger.debug("Legacy terrain data saved to DataManager")
            else:
                self.logger.warning(f"Unknown terrain result format: {type(result)}")

    def _progressive_heightmap_generation(self, target_size, existing_heightmap=None):
        """
        Funktionsweise: Generiert Heightmap progressiv oder von Grund auf
        Aufgabe: Wiederverwenden bestehender Daten mit Detail-Verfeinerung
        Parameter: target_size (int) - Zielgröße für Heightmap
        Parameter: existing_heightmap (numpy.ndarray) - Bestehende Heightmap zum Erweitern
        Returns: numpy.ndarray - Generierte/erweiterte Heightmap
        """
        if existing_heightmap is not None:
            # Progressive Verbesserung
            # Schritt 1: Interpolation auf neue Größe
            interpolated = self.noise_generator.interpolate_existing_grid(
                existing_heightmap, target_size
            )

            # Schritt 2: Detail-Noise hinzufügen
            # Frequency anpassen: Höhere Frequenz für feinere Details
            detail_frequency = self.frequency * (target_size / existing_heightmap.shape[0])
            detail_amplitude = self.amplitude * 0.25  # 25% der Original-Amplitude für Details

            final_heightmap = self.noise_generator.add_detail_noise(
                interpolated, detail_frequency, detail_amplitude
            )
        else:
            # Vollständige Generierung für erste LOD-Stufe
            final_heightmap = self._generate_full_heightmap(target_size)

        return final_heightmap

    def _progressive_heightmap_generation_with_yields(self, target_size, existing_heightmap=None):
        """
        Funktionsweise: Wie _progressive_heightmap_generation aber mit CPU-Yields
        Aufgabe: Hintergrund-freundliche Heightmap-Generierung
        Parameter: target_size, existing_heightmap - siehe _progressive_heightmap_generation
        Returns: numpy.ndarray - Generierte Heightmap
        """
        if existing_heightmap is not None:
            # Interpolation mit Yield
            self._yield_cpu()
            interpolated = self.noise_generator.interpolate_existing_grid(
                existing_heightmap, target_size
            )

            # Detail-Noise mit Yield
            self._yield_cpu()
            detail_frequency = self.frequency * (target_size / existing_heightmap.shape[0])
            detail_amplitude = self.amplitude * 0.25

            final_heightmap = self.noise_generator.add_detail_noise(
                interpolated, detail_frequency, detail_amplitude
            )
        else:
            # Vollständige Generierung mit Yields
            final_heightmap = self._generate_full_heightmap_with_yields(target_size)

        return final_heightmap

    def _progressive_shadow_generation(self, heightmap, lod_level, existing_shadows=None, existing_lod="LOD64"):
        """
        Funktionsweise: Generiert Shadows progressiv oder vollständig mit LOD-System
        Aufgabe: Optimierte Shadow-Berechnung mit Wiederverwendung bestehender Daten
        Parameter: heightmap (numpy.ndarray) - Aktuelle Höhendaten
        Parameter: lod_level (str) - Ziel-LOD für Sonnenwinkel-Auswahl
        Parameter: existing_shadows (numpy.ndarray) - Bestehende Shadow-Daten
        Parameter: existing_lod (str) - LOD der bestehenden Shadow-Daten
        Returns: numpy.ndarray - Generierte/erweiterte Shadow-Map
        """
        if existing_shadows is not None and existing_lod != lod_level:
            # Progressive Shadow-Verbesserung
            return self.shadow_calculator.calculate_shadows_progressive(
                heightmap, lod_level, existing_shadows, existing_lod
            )
        else:
            # Vollständige Shadow-Berechnung
            return self.shadow_calculator.calculate_shadows_with_lod(
                heightmap, lod_level, shadow_resolution=64
            )

    def _generate_full_heightmap(self, size):
        """
        Funktionsweise: Generiert komplette Heightmap von Grund auf
        Aufgabe: Vollständige Noise-Generierung für erste LOD-Stufe
        Parameter: size (int) - Größe der zu generierenden Heightmap
        Returns: numpy.ndarray - Vollständig generierte Heightmap
        """
        # Frequency für aktuelle Größe anpassen
        adjusted_frequency = self.frequency * (64 / size)  # Referenz: LOD64

        # Noise-Grid generieren
        noise_grid = self.noise_generator.generate_noise_grid(
            size, adjusted_frequency, self.octaves, self.persistence, self.lacunarity
        )

        # Auf [0, amplitude] skalieren
        heightmap = (noise_grid + 1.0) * 0.5 * self.amplitude

        return heightmap

    def _generate_full_heightmap_with_yields(self, size):
        """
        Funktionsweise: Wie _generate_full_heightmap aber mit CPU-Yields für Background-Berechnung
        Aufgabe: Hintergrund-freundliche vollständige Heightmap-Generierung
        Parameter: size (int) - Größe der Heightmap
        Returns: numpy.ndarray - Generierte Heightmap
        """
        # Yield vor intensiver Berechnung
        self._yield_cpu()

        adjusted_frequency = self.frequency * (64 / size)
        noise_grid = self.noise_generator.generate_noise_grid(
            size, adjusted_frequency, self.octaves, self.persistence, self.lacunarity
        )

        # Yield nach Noise-Generierung
        self._yield_cpu()

        heightmap = (noise_grid + 1.0) * 0.5 * self.amplitude
        return heightmap

    def _calculate_slopes_with_yields(self, heightmap):
        """
        Funktionsweise: Berechnet Slope-Map mit CPU-Yields für bessere Background-Performance
        Aufgabe: Hintergrund-freundliche Slope-Berechnung
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: numpy.ndarray - Slope-Map
        """
        # Standard-Slope-Berechnung mit gelegentlichen Yields
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        yield_counter = 0
        for y in range(height):
            for x in range(width):
                # CPU-Yield alle 1000 Pixel
                yield_counter += 1
                if yield_counter % 1000 == 0:
                    self._yield_cpu()

                # Standard Slope-Berechnung (wie im Original)
                if x > 0 and x < width - 1:
                    dz_dx = (heightmap[y, x + 1] - heightmap[y, x - 1]) * 0.5
                elif x == 0:
                    dz_dx = heightmap[y, x + 1] - heightmap[y, x]
                else:
                    dz_dx = heightmap[y, x] - heightmap[y, x - 1]

                if y > 0 and y < height - 1:
                    dz_dy = (heightmap[y + 1, x] - heightmap[y - 1, x]) * 0.5
                elif y == 0:
                    dz_dy = heightmap[y + 1, x] - heightmap[y, x]
                else:
                    dz_dy = heightmap[y, x] - heightmap[y - 1, x]

                slopemap[y, x, 0] = dz_dx
                slopemap[y, x, 1] = dz_dy

        return slopemap

    def _calculate_lod_steps(self, final_size):
        """
        Funktionsweise: Berechnet optimale LOD-Schritte nach Option A
        Aufgabe: Bestimmt Zwischen-Auflösungen für progressive Generierung
        Parameter: final_size (int) - Finale Zielgröße
        Returns: list - Liste der LOD-Schritte [64, 128, 256, final_size]
        """
        steps = []

        # Immer mit 64 beginnen (außer finale Größe ist kleiner)
        if final_size >= 64:
            steps.append(64)

        # 128 hinzufügen wenn Zielgröße >= 128
        if final_size >= 128:
            steps.append(128)

        # 256 hinzufügen wenn Zielgröße >= 256
        if final_size >= 256:
            steps.append(256)

        # Finale Größe hinzufügen wenn sie nicht bereits in der Liste ist
        if final_size not in steps:
            steps.append(final_size)

        return steps

    def _update_parameters(self, map_size=None, amplitude=None, octaves=None,
                           frequency=None, persistence=None, lacunarity=None,
                           redistribute_power=None, map_seed=None):
        """
        Funktionsweise: Aktualisiert Generator-Parameter falls neue Werte übergeben werden
        Aufgabe: Flexible Parameter-Übernahme ohne Standardwerte zu überschreiben
        Parameter: map_size, amplitude, etc. - Neue Parameter (None = keine Änderung)
        """
        if map_size is not None:
            self.map_size = map_size
        if amplitude is not None:
            self.amplitude = amplitude
        if octaves is not None:
            self.octaves = octaves
        if frequency is not None:
            self.frequency = frequency
        if persistence is not None:
            self.persistence = persistence
        if lacunarity is not None:
            self.lacunarity = lacunarity
        if redistribute_power is not None:
            self.redistribute_power = redistribute_power
        if map_seed is not None and map_seed != self.map_seed:
            self.map_seed = map_seed
            self.noise_generator = SimplexNoiseGenerator(seed=map_seed)

    def _get_current_parameters(self):
        """
        Funktionsweise: Gibt aktuelle Parameter als Dictionary zurück
        Aufgabe: Parameter-Speicherung für TerrainData-Metadaten
        Returns: dict - Aktuelle Generator-Parameter
        """
        return {
            'map_size': self.map_size,
            'amplitude': self.amplitude,
            'octaves': self.octaves,
            'frequency': self.frequency,
            'persistence': self.persistence,
            'lacunarity': self.lacunarity,
            'redistribute_power': self.redistribute_power,
            'map_seed': self.map_seed
        }

    def _update_progress(self, step_name, progress_percent, detail_message):
        """
        Funktionsweise: Sendet Progress-Update an Callback-Funktion wenn vorhanden
        Aufgabe: GUI-Updates für Ladebalken und Status-Anzeigen
        Parameter: step_name (str) - Name des aktuellen Schritts
        Parameter: progress_percent (int) - Fortschritt in Prozent (0-100)
        Parameter: detail_message (str) - Detaillierte Beschreibung
        """
        if self.progress_callback:
            try:
                self.progress_callback(step_name, progress_percent, detail_message)
            except:
                pass  # Ignore callback errors

    def _yield_cpu(self):
        """
        Funktionsweise: Gibt CPU-Zeit an andere Prozesse ab
        Aufgabe: Verhindert 100% CPU-Auslastung bei Background-Berechnung
        """
        time.sleep(0.001)  # 1ms Pause für andere Prozesse

    # BESTEHENDE METHODEN (unverändert für Kompatibilität):

    def generate_heightmap(self, map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power,
                           map_seed):
        """
        Funktionsweise: Legacy-Methode für direkte Heightmap-Generierung (KOMPATIBILITÄT)
        Aufgabe: Erhält bestehende API für Rückwärts-Kompatibilität
        """
        self._update_parameters(map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power,
                                map_seed)
        return self._generate_full_heightmap(map_size)

    def apply_redistribution(self, heightmap, redistribute_power):
        """
        Funktionsweise: Wendet Power-Redistribution auf Heightmap an für natürlichere Höhenverteilung
        Aufgabe: Modifiziert Höhenverteilung für realistische Terrain-Charakteristika
        Parameter: heightmap (numpy.ndarray), redistribute_power (float) - Heightmap und Power-Faktor
        Returns: numpy.ndarray - Redistributed Heightmap
        """
        if redistribute_power == 1.0:
            return heightmap

        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            return heightmap

        normalized = (heightmap - min_height) / height_range
        redistributed = np.power(normalized, redistribute_power)
        result = redistributed * height_range + min_height

        return result

    def calculate_slopes(self, heightmap):
        """
        Funktionsweise: Berechnet Slope-Map mit dz/dx und dz/dy Gradienten
        Aufgabe: Erstellt Slope-Informationen für Erosion, Settlement und Biome-Systeme
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: numpy.ndarray - Slope-Map mit Shape (height, width, 2) für dz/dx und dz/dy
        """
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                if x > 0 and x < width - 1:
                    dz_dx = (heightmap[y, x + 1] - heightmap[y, x - 1]) * 0.5
                elif x == 0:
                    dz_dx = heightmap[y, x + 1] - heightmap[y, x]
                else:
                    dz_dx = heightmap[y, x] - heightmap[y, x - 1]

                if y > 0 and y < height - 1:
                    dz_dy = (heightmap[y + 1, x] - heightmap[y - 1, x]) * 0.5
                elif y == 0:
                    dz_dy = heightmap[y + 1, x] - heightmap[y, x]
                else:
                    dz_dy = heightmap[y, x] - heightmap[y - 1, x]

                slopemap[y, x, 0] = dz_dx
                slopemap[y, x, 1] = dz_dy

        return slopemap

    def generate_shadows(self, heightmap):
        """
        Funktionsweise: Legacy-Methode für Shadow-Generierung (KOMPATIBILITÄT)
        Aufgabe: Erhält bestehende API für Rückwärts-Kompatibilität
        """
        return self.shadow_calculator.calculate_shadows_multi_angle(heightmap)

    def generate_complete_terrain(self, map_size, amplitude, octaves, frequency, persistence, lacunarity,
                                  redistribute_power, map_seed):
        """
        Funktionsweise: Legacy-Methode für komplette Terrain-Generierung (KOMPATIBILITÄT)
        Aufgabe: Erhält bestehende API, verwendet intern neues LOD-System
        """
        # Parameter setzen
        self._update_parameters(map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power,
                                map_seed)

        # Neue generate_terrain Methode verwenden
        terrain_data = self.generate_terrain(lod="FINAL", map_size=map_size)

        # Legacy-Format zurückgeben
        return terrain_data.heightmap, terrain_data.slopemap, terrain_data.shadowmap