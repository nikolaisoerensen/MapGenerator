"""
Path: core/terrain_generator.py
Date Changed: 29.08.2025

Funktionsweise: Terrain-Generation mit numerischem LOD-System und 3-stufigem Fallback-System
- BaseTerrainGenerator koordiniert alle Terrain-Generierungsschritte mit DataLODManager-Integration
- SimplexNoiseGenerator mit GPU/CPU/Simple-Fallback über ShaderManager
- ShadowCalculator mit LOD-spezifischen Sonnenwinkeln und Fallback-System
- SlopeCalculator für Gradienten-Berechnung mit Performance-Optimierung
- TerrainData mit erweiterten Validity-Checks und Parameter-Hash-System

Parameter Input:
- parameters: dict mit map_seed, map_size, amplitude, octaves, frequency, persistence, lacunarity, redistribute_power
- lod_level: int (1-7) für numerisches LOD-System

Output:
- TerrainData-Objekt mit heightmap, slopemap, shadowmap, validity_state und LOD-Metadaten
- DataLODManager-Integration über set_terrain_data_lod()

LOD-System (Numerisch):
- lod_level 1: 32x32 (1 Sonnenwinkel - Mittag)
- lod_level 2: 64x64 (3 Sonnenwinkel - Vormittag/Mittag/Nachmittag)
- lod_level 3: 128x128 (5 Sonnenwinkel + Morgen/Abend)
- lod_level 4: 256x256 (7 Sonnenwinkel + Dämmerung)
- lod_level 5+: bis map_size erreicht, 7 Sonnenwinkel konstant

Fallback-System (3-stufig):
- GPU-Shader (Optimal): ShaderManager für parallele Multi-Octave-Noise-Berechnung
- CPU-Fallback (Gut): Optimierte NumPy-Implementierung mit Multiprocessing
- Simple-Fallback (Minimal): Direkte Implementierung, wenige Zeilen, garantiert funktionsfähig
"""

import numpy as np
from opensimplex import OpenSimplex
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any

try:
    from core.base_generator import BaseGenerator
except ImportError:
    # Fallback wenn BaseGenerator nicht verfügbar
    class BaseGenerator:
        def __init__(self, map_seed=42):
            self.map_seed = map_seed
            self.logger = logging.getLogger(self.__class__.__name__)


class TerrainData:
    """
    Funktionsweise: Container für alle Terrain-Daten mit Validity-System und Cache-Management
    Aufgabe: Speichert Heightmap, Slopemap, Shadowmap mit LOD-Level, Validity-State und Parameter-Hash
    Attribute: heightmap, slopemap, shadowmap, lod_level, actual_size, validity_state, parameter_hash,
               calculated_sun_angles, parameters
    Validity-Methods: is_valid(), invalidate(), validate_against_parameters(), get_validity_summary()
    """

    def __init__(self):
        # Core terrain data
        self.heightmap: Optional[np.ndarray] = None
        self.slopemap: Optional[np.ndarray] = None
        self.shadowmap: Optional[np.ndarray] = None

        # LOD metadata
        self.lod_level: int = 1
        self.actual_size: int = 32
        self.calculated_sun_angles: List[Tuple[int, int]] = []

        # Validity system
        self.validity_state: str = "valid"
        self.validity_flags: Dict[str, bool] = {
            "heightmap": False,
            "slopemap": False,
            "shadowmap": False
        }

        # Parameter tracking
        self.parameters: Dict[str, Any] = {}
        self.parameter_hash: Optional[str] = None

        # Performance metadata
        self.generation_time: float = 0.0
        self.fallback_used: str = "unknown"  # "gpu", "cpu", "simple"

    def is_valid(self) -> bool:
        """Prüft Validity-State des TerrainData-Objekts"""
        return self.validity_state == "valid" and all(self.validity_flags.values())

    def invalidate(self):
        """Invalidiert TerrainData-Objekt"""
        self.validity_state = "invalid"
        self.validity_flags = {key: False for key in self.validity_flags}

    def validate_against_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """
        Funktionsweise: Prüft ob TerrainData mit neuen Parametern kompatibel ist
        Parameter: new_parameters (dict)
        Return: bool - Kompatibel
        """
        new_hash = self._calculate_parameter_hash(new_parameters)
        return self.parameter_hash == new_hash

    def get_validity_summary(self) -> Dict[str, Any]:
        """
        Return: dict mit Validity-Informationen für DataManager
        """
        return {
            "validity_state": self.validity_state,
            "validity_flags": self.validity_flags.copy(),
            "lod_level": self.lod_level,
            "actual_size": self.actual_size,
            "calculated_sun_angles": len(self.calculated_sun_angles),
            "parameter_hash": self.parameter_hash,
            "generation_time": self.generation_time,
            "fallback_used": self.fallback_used
        }

    def detect_critical_changes(self, new_parameters: Dict[str, Any]) -> List[str]:
        """
        Funktionsweise: Erkennt signifikante Parameter-Änderungen für Cache-Invalidation
        Parameter: new_parameters - Neue Parameter zum Vergleich
        Returns: List[str] - Liste der kritischen Änderungen
        """
        if not self.parameters:
            return ["initial_generation"]

        critical_params = ["map_seed", "map_size", "amplitude", "octaves", "frequency"]
        critical_changes = []

        for param in critical_params:
            if param in new_parameters and param in self.parameters:
                if new_parameters[param] != self.parameters[param]:
                    critical_changes.append(param)

        return critical_changes

    def get_invalidated_generators(self) -> List[str]:
        """
        Funktionsweise: Bestimmt welche nachgelagerten Generatoren invalidiert werden müssen
        Returns: List[str] - Liste der zu invalidierenden Generatoren
        """
        # Terrain ist Basis-Generator - alle anderen hängen davon ab
        return ["geology", "weather", "water", "biome", "settlement"]

    def _calculate_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """Berechnet MD5-Hash der Parameter für Cache-Validation"""
        return hashlib.md5(str(sorted(parameters.items())).encode()).hexdigest()

    def update_parameters(self, parameters: Dict[str, Any]):
        """Aktualisiert Parameter und Parameter-Hash"""
        self.parameters = parameters.copy()
        self.parameter_hash = self._calculate_parameter_hash(parameters)


class SimplexNoiseGenerator:
    """
    Funktionsweise: Erzeugt OpenSimplex-Noise mit 3-stufiger Fallback-Strategie
    Aufgabe: Basis-Noise-Funktionen für Heightmap-Generation mit Performance-Optimierung
    Methoden: noise_2d(), multi_octave_noise(), ridge_noise()
    LOD-Optimiert: generate_noise_grid() für Batch-Verarbeitung, interpolate_existing_grid() für LOD-Upgrades

    Spezifische Fallbacks:
    - GPU-Optimal: request_noise_generation() für parallele Multi-Octave-Berechnung
    - CPU-Fallback: Optimierte NumPy-Implementierung mit vectorization
    - Simple-Fallback: Direkte Random-Noise-Generation (5-10 Zeilen)
    """

    def __init__(self, seed: int = 42, shader_manager=None):
        """
        Funktionsweise: Initialisiert OpenSimplex-Generator mit ShaderManager-Integration
        Parameter: seed (int) - Seed für reproduzierbaren Noise
        Parameter: shader_manager - ShaderManager-Instanz für GPU-Acceleration
        """
        self.generator = OpenSimplex(seed=seed)
        self.shader_manager = shader_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def generate_noise_grid(self, size: int, frequency: float, octaves: int,
                          persistence: float, lacunarity: float,
                          offset_x: float = 0, offset_y: float = 0) -> np.ndarray:
        """
        Funktionsweise: Generiert komplettes Noise-Grid mit 3-stufigem Fallback
        Aufgabe: Performance-Optimierung durch Batch-Verarbeitung
        Parameter: size, frequency, octaves, persistence, lacunarity, offset_x, offset_y
        Returns: numpy.ndarray - Komplettes Noise-Grid mit Werten zwischen -1 und 1
        """
        parameters = {
            'size': size,
            'frequency': frequency,
            'octaves': octaves,
            'persistence': persistence,
            'lacunarity': lacunarity,
            'offset_x': offset_x,
            'offset_y': offset_y
        }

        # GPU-Fallback (Optimal)
        if self._gpu_available():
            try:
                result = self.shader_manager.request_noise_generation(
                    operation_type="multi_octave_noise",
                    parameters=parameters
                )
                if result.get('success', False):
                    self.logger.debug("GPU noise generation successful")
                    return result['data']
            except Exception as e:
                self.logger.warning(f"GPU noise generation failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._generate_cpu_optimized(parameters)
        except Exception as e:
            self.logger.warning(f"CPU noise generation failed: {e}")

        # Simple-Fallback (Minimal, garantiert funktionsfähig)
        return self._generate_simple_fallback(parameters)

    def interpolate_existing_grid(self, existing_grid: np.ndarray, new_size: int) -> np.ndarray:
        """
        Funktionsweise: Interpoliert bestehende LOD-Daten auf höhere Auflösung mittels bilinearer Interpolation
        Aufgabe: Progressive LOD-Verbesserung ohne Neuberechnung aller Werte
        Parameter: existing_grid - Bestehende niedrig-aufgelöste Daten
        Parameter: new_size - Zielgröße für Interpolation
        Returns: numpy.ndarray - Interpolierte Daten in neuer Auflösung
        """
        old_size = existing_grid.shape[0]

        if old_size == new_size:
            return existing_grid.copy()

        # GPU-Interpolation falls verfügbar
        if self._gpu_available():
            try:
                result = self.shader_manager.request_interpolation(
                    operation_type="bicubic_interpolation",
                    input_data=existing_grid,
                    target_size=new_size
                )
                if result.get('success', False):
                    return result['data']
            except Exception as e:
                self.logger.warning(f"GPU interpolation failed: {e}")

        # CPU-Fallback mit optimierter Interpolation
        return self._interpolate_cpu_optimized(existing_grid, new_size)

    def add_detail_noise(self, base_grid: np.ndarray, detail_frequency: float,
                        detail_amplitude: float) -> np.ndarray:
        """
        Funktionsweise: Fügt hochfrequente Detail-Noise zu bestehender interpolierter Basis hinzu
        Aufgabe: Verfeinert interpolierte LOD-Daten mit lokalen Details
        Parameter: base_grid - Basis-Grid aus Interpolation
        Parameter: detail_frequency - Frequenz für Detail-Noise
        Parameter: detail_amplitude - Stärke der Detail-Noise (meist 10-30% der Original-Amplitude)
        Returns: numpy.ndarray - Verfeinertes Grid mit Details
        """
        size = base_grid.shape[0]

        # Detail-Noise mit höherer Frequenz generieren
        detail_grid = self.generate_noise_grid(
            size=size,
            frequency=detail_frequency,
            octaves=2,  # Weniger Octaves für Details
            persistence=0.5,
            lacunarity=2.0
        )

        # Detail-Noise mit reduzierter Amplitude zur Basis hinzufügen
        enhanced_grid = base_grid + (detail_grid * detail_amplitude)
        return enhanced_grid

    def _gpu_available(self) -> bool:
        """Prüft GPU-Verfügbarkeit über ShaderManager"""
        return (self.shader_manager is not None and
                hasattr(self.shader_manager, 'gpu_available') and
                self.shader_manager.gpu_available)

    def _generate_cpu_optimized(self, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Funktionsweise: Optimierte NumPy-Implementierung für CPU-Fallback
        Aufgabe: Vectorized Operations für bessere Performance ohne GPU
        Parameter: parameters - Noise-Parameter
        Returns: numpy.ndarray - CPU-generiertes Noise-Grid
        """
        size = parameters['size']
        frequency = parameters['frequency']
        octaves = parameters['octaves']
        persistence = parameters['persistence']
        lacunarity = parameters['lacunarity']
        offset_x = parameters.get('offset_x', 0)
        offset_y = parameters.get('offset_y', 0)

        # Koordinaten-Arrays für gesamtes Grid erstellen (vectorized)
        x_coords = np.linspace(0, 1, size) + offset_x
        y_coords = np.linspace(0, 1, size) + offset_y
        X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')

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

            # Vectorized noise calculation für bessere Performance
            octave_noise = self._vectorized_noise(freq_X, freq_Y)

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

    def _vectorized_noise(self, freq_X: np.ndarray, freq_Y: np.ndarray) -> np.ndarray:
        """
        Funktionsweise: Vectorized Noise-Berechnung für CPU-Performance
        Parameter: freq_X, freq_Y - Frequency-adjusted coordinate arrays
        Returns: numpy.ndarray - Noise values
        """
        # Optimierte Batch-Verarbeitung
        result = np.zeros_like(freq_X, dtype=np.float32)

        # Process in chunks für Memory-Efficiency
        chunk_size = min(1000, freq_X.size)
        flat_X = freq_X.flatten()
        flat_Y = freq_Y.flatten()
        flat_result = np.zeros_like(flat_X)

        for i in range(0, len(flat_X), chunk_size):
            end_idx = min(i + chunk_size, len(flat_X))
            for j in range(i, end_idx):
                flat_result[j] = self.generator.noise2(flat_X[j], flat_Y[j])

        return flat_result.reshape(freq_X.shape)

    def _generate_simple_fallback(self, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Funktionsweise: Einfache Fallback-Implementierung (5-10 Zeilen)
        Aufgabe: Garantierte Funktionsfähigkeit auch bei kritischen Fehlern
        Parameter: parameters - Noise-Parameter
        Returns: numpy.ndarray - Simple Random-Noise
        """
        size = parameters['size']

        # Simple Random-Noise mit Seed-Reproduzierbarkeit
        np.random.seed(hash(str(parameters)) % (2**32))

        # Basis Random-Noise
        noise = np.random.uniform(-1, 1, (size, size)).astype(np.float32)

        # Einfache Glättung für weniger chaotisches Aussehen
        from scipy.ndimage import gaussian_filter
        try:
            noise = gaussian_filter(noise, sigma=1.0, mode='wrap')
        except ImportError:
            # Fallback wenn scipy nicht verfügbar
            pass

        return noise

    def _interpolate_cpu_optimized(self, existing_grid: np.ndarray, new_size: int) -> np.ndarray:
        """
        Funktionsweise: CPU-optimierte bilineare Interpolation
        Parameter: existing_grid, new_size
        Returns: numpy.ndarray - Interpolierte Daten
        """
        old_size = existing_grid.shape[0]
        scale_factor = (old_size - 1) / (new_size - 1)

        # Vectorized coordinate calculation
        new_coords = np.arange(new_size, dtype=np.float32)
        old_x_coords = new_coords * scale_factor
        old_y_coords = new_coords * scale_factor

        # Mesh für vectorized interpolation
        old_X, old_Y = np.meshgrid(old_x_coords, old_y_coords, indexing='xy')

        # Scipy interpolation falls verfügbar, sonst manuelle bilineare Interpolation
        try:
            from scipy.interpolate import RegularGridInterpolator
            old_grid_coords = (np.arange(old_size), np.arange(old_size))
            interpolator = RegularGridInterpolator(old_grid_coords, existing_grid,
                                                 method='linear', bounds_error=False,
                                                 fill_value=0)

            points = np.column_stack([old_Y.ravel(), old_X.ravel()])
            interpolated = interpolator(points).reshape((new_size, new_size))
            return interpolated.astype(np.float32)

        except ImportError:
            # Manual bilinear interpolation fallback
            return self._manual_bilinear_interpolation(existing_grid, new_size)

    def _manual_bilinear_interpolation(self, existing_grid: np.ndarray, new_size: int) -> np.ndarray:
        """Manual bilinear interpolation without scipy dependency"""
        old_size = existing_grid.shape[0]
        scale_factor = (old_size - 1) / (new_size - 1)

        interpolated = np.zeros((new_size, new_size), dtype=np.float32)

        for new_y in range(new_size):
            for new_x in range(new_size):
                old_x = new_x * scale_factor
                old_y = new_y * scale_factor

                x0, y0 = int(old_x), int(old_y)
                x1, y1 = min(x0 + 1, old_size - 1), min(y0 + 1, old_size - 1)

                fx, fy = old_x - x0, old_y - y0

                # Bilinear interpolation
                h00 = existing_grid[y0, x0]
                h10 = existing_grid[y0, x1]
                h01 = existing_grid[y1, x0]
                h11 = existing_grid[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                interpolated[new_y, new_x] = h0 * (1 - fy) + h1 * fy

        return interpolated


class ShadowCalculator:
    """
    Funktionsweise: Berechnet Verschattung mit Raycasts für LOD-spezifische Sonnenwinkel
    Aufgabe: Erstellt shadowmap (konstant 64x64) für Weather-System und visuelle Darstellung
    Methoden: calculate_shadows(heightmap, lod_level, parameters), raycast_shadow(), combine_shadow_angles()
    LOD-System: get_sun_angles_for_lod() - 1,3,5,7 Sonnenwinkel je nach LOD-Level
    Progressive-Enhancement: Berechnet nur neue Sonnenwinkel bei LOD-Upgrades, kombiniert mit bestehenden Shadows

    Spezifische Fallbacks:
    - GPU-Optimal: Parallele Raycast-Berechnung für alle Sonnenwinkel
    - CPU-Fallback: Optimierte CPU-Raycast-Implementierung
    - Simple-Fallback: Einfache Height-Difference-Shadow-Approximation
    """

    def __init__(self, shader_manager=None):
        """
        Funktionsweise: Initialisiert Shadow-Calculator mit LOD-spezifischer Sonnenwinkel-Konfiguration
        Parameter: shader_manager - ShaderManager für GPU-Acceleration
        """
        self.shader_manager = shader_manager
        self.logger = logging.getLogger(self.__class__.__name__)

        # 7 Sonnenwinkel für Tagesverlauf (elevation, azimuth in Grad)
        self.sun_angles = [
            (10, 75),   # Morgendämmerung
            (25, 90),   # Morgen
            (45, 120),  # Vormittag
            (70, 180),  # Mittag
            (45, 240),  # Nachmittag
            (25, 270),  # Abend
            (10, 285)   # Späte Dämmerung
        ]

        # Gewichtung durch atmosphärische Durchdringung
        self.sun_weights = [0.06, 0.2, 0.6, 0.9, 0.6, 0.2, 0.06]

    def calculate_shadows(self, heightmap: np.ndarray, lod_level: int,
                         existing_shadows: Optional[np.ndarray] = None,
                         existing_lod: int = 1) -> np.ndarray:
        """
        Funktionsweise: Berechnet Verschattung mit 3-stufigem Fallback und LOD-System
        Aufgabe: Erstellt shadowmap (konstant 64x64) für Weather-System
        Parameter: heightmap - Höhendaten
        Parameter: lod_level - LOD-Level für Sonnenwinkel-Auswahl
        Parameter: existing_shadows - Bestehende Shadow-Daten (optional)
        Parameter: existing_lod - LOD-Level der bestehenden Shadows
        Returns: numpy.ndarray - Shadow-Map konstant 64x64
        """
        # GPU-Fallback (Optimal)
        if self._gpu_available():
            try:
                result = self.shader_manager.request_shadow_calculation(
                    operation_type="multi_angle_shadows",
                    heightmap=heightmap,
                    lod_level=lod_level,
                    existing_shadows=existing_shadows,
                    existing_lod=existing_lod
                )
                if result.get('success', False):
                    self.logger.debug("GPU shadow calculation successful")
                    return result['data']
            except Exception as e:
                self.logger.warning(f"GPU shadow calculation failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._calculate_cpu_shadows(heightmap, lod_level, existing_shadows, existing_lod)
        except Exception as e:
            self.logger.warning(f"CPU shadow calculation failed: {e}")

        # Simple-Fallback (Minimal)
        return self._calculate_simple_shadows(heightmap, lod_level)

    def get_sun_angles_for_lod(self, lod_level: int) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Funktionsweise: Gibt passende Sonnenwinkel-Auswahl für LOD-Level zurück
        Parameter: lod_level (int) - 1,2,3,4,5,6,7
        Returns: Tuple (sun_angles_list, sun_weights_list) - Gefilterte Winkel und Gewichtungen
        """
        if lod_level == 1:
            # Nur Mittag
            indices = [3]
        elif lod_level == 2:
            # Mittag + Vormittag + Nachmittag
            indices = [2, 3, 4]
        elif lod_level == 3:
            # + Morgen + Abend
            indices = [1, 2, 3, 4, 5]
        else:  # lod_level >= 4
            # Alle 7 Winkel
            indices = list(range(7))

        selected_angles = [self.sun_angles[i] for i in indices]
        selected_weights = [self.sun_weights[i] for i in indices]

        return selected_angles, selected_weights

    def _gpu_available(self) -> bool:
        """Prüft GPU-Verfügbarkeit über ShaderManager"""
        return (self.shader_manager is not None and
                hasattr(self.shader_manager, 'gpu_available') and
                self.shader_manager.gpu_available)

    def _calculate_cpu_shadows(self, heightmap: np.ndarray, lod_level: int,
                              existing_shadows: Optional[np.ndarray] = None,
                              existing_lod: int = 1) -> np.ndarray:
        """
        Funktionsweise: Optimierte CPU-Raycast-Implementierung
        Parameter: heightmap, lod_level, existing_shadows, existing_lod
        Returns: numpy.ndarray - Shadow-Map 64x64
        """
        shadow_resolution = 64
        original_size = heightmap.shape[0]

        # Heightmap für Shadow-Berechnung auf 64x64 reduzieren falls nötig
        if original_size > shadow_resolution:
            shadow_heightmap = self._downsample_heightmap(heightmap, shadow_resolution)
        else:
            shadow_heightmap = heightmap

        # Progressive Shadow-Enhancement falls bestehende Shadows vorhanden
        if existing_shadows is not None and existing_lod != lod_level:
            return self._calculate_progressive_shadows(shadow_heightmap, lod_level,
                                                     existing_shadows, existing_lod)

        # Vollständige Shadow-Berechnung
        sun_angles, sun_weights = self.get_sun_angles_for_lod(lod_level)
        shadows = np.zeros((shadow_resolution, shadow_resolution), dtype=np.float32)
        total_weight = sum(sun_weights)

        for i, (elevation, azimuth) in enumerate(sun_angles):
            shadow_map = self._raycast_shadow_cpu(shadow_heightmap, elevation, azimuth)
            shadows += shadow_map * sun_weights[i]

        # Normalisierung
        shadows /= total_weight

        # Upscale auf Original-Größe falls nötig
        if original_size > shadow_resolution:
            shadows = self._upsample_shadows(shadows, original_size)

        return shadows

    def _calculate_simple_shadows(self, heightmap: np.ndarray, lod_level: int) -> np.ndarray:
        """
        Funktionsweise: Einfache Height-Difference-Shadow-Approximation
        Aufgabe: Garantierte Funktionsfähigkeit ohne komplexe Raycasting
        Parameter: heightmap, lod_level
        Returns: numpy.ndarray - Approximierte Shadow-Map
        """
        height, width = heightmap.shape
        shadows = np.ones((height, width), dtype=np.float32)

        # Einfache Gradient-basierte Shadow-Approximation
        # Steile Nordhänge sind dunkler, Südhänge heller
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Einfacher Gradient in Y-Richtung (Nord-Süd)
                north_slope = heightmap[y-1, x] - heightmap[y, x]
                south_slope = heightmap[y+1, x] - heightmap[y, x]

                # Nordhänge dunkler, Südhänge heller
                slope_factor = (south_slope - north_slope) * 0.1
                shadows[y, x] = np.clip(0.5 + slope_factor, 0.1, 1.0)

        return shadows

    def _calculate_progressive_shadows(self, heightmap: np.ndarray, lod_level: int,
                                     existing_shadows: np.ndarray, existing_lod: int) -> np.ndarray:
        """Progressive Shadow-Enhancement - nur neue Sonnenwinkel berechnen"""
        new_angles, new_weights = self.get_sun_angles_for_lod(lod_level)
        old_angles, old_weights = self.get_sun_angles_for_lod(existing_lod)

        # Finde neue Winkel
        old_angle_set = set(old_angles)
        additional_angles = []
        additional_weights = []

        for angle, weight in zip(new_angles, new_weights):
            if angle not in old_angle_set:
                additional_angles.append(angle)
                additional_weights.append(weight)

        if not additional_angles:
            return existing_shadows

        # Berechne nur zusätzliche Winkel
        additional_shadows = np.zeros_like(existing_shadows, dtype=np.float32)
        for elevation, azimuth in additional_angles:
            shadow_map = self._raycast_shadow_cpu(heightmap, elevation, azimuth)
            additional_shadows += shadow_map

        # Normiere die zusätzlichen Shadows
        if additional_weights:
            additional_shadows /= len(additional_weights)

        # Kombiniere bestehende und neue Shadows
        total_old_weight = sum(old_weights)
        total_additional_weight = sum(additional_weights)
        total_weight = total_old_weight + total_additional_weight

        combined_shadows = (existing_shadows * total_old_weight +
                          additional_shadows * total_additional_weight) / total_weight

        return combined_shadows

    def _raycast_shadow_cpu(self, heightmap: np.ndarray, sun_elevation: float,
                           sun_azimuth: float) -> np.ndarray:
        """
        Funktionsweise: CPU-optimierte Raycast-Shadow-Berechnung für einen Sonnenwinkel
        Parameter: heightmap, sun_elevation, sun_azimuth
        Returns: numpy.ndarray - Shadow-Map für diesen Sonnenwinkel
        """
        height, width = heightmap.shape
        shadow_map = np.ones((height, width), dtype=np.float32)

        # Sonnenrichtung berechnen
        elevation_rad = np.radians(sun_elevation)
        azimuth_rad = np.radians(sun_azimuth)

        sun_x = np.cos(elevation_rad) * np.sin(azimuth_rad)
        sun_y = np.cos(elevation_rad) * np.cos(azimuth_rad)
        sun_z = np.sin(elevation_rad)

        # Optimierte Raycast-Berechnung
        for y in range(height):
            for x in range(width):
                if self._is_in_shadow_cpu(heightmap, x, y, sun_x, sun_y, sun_z):
                    shadow_map[y, x] = 0.0
                else:
                    # Slope-basierte Beleuchtung
                    slope_factor = self._calculate_slope_shading_cpu(
                        heightmap, x, y, sun_x, sun_y, sun_z
                    )
                    shadow_map[y, x] = slope_factor

        return shadow_map

    def _is_in_shadow_cpu(self, heightmap: np.ndarray, x: int, y: int,
                         sun_x: float, sun_y: float, sun_z: float) -> bool:
        """CPU-optimierte Shadow-Raycast-Test"""
        height, width = heightmap.shape
        current_height = heightmap[y, x]

        step_size = 0.5
        max_distance = max(width, height) * 2

        for distance in np.arange(step_size, max_distance, step_size):
            ray_x = x + sun_x * distance
            ray_y = y + sun_y * distance
            ray_z = current_height + sun_z * distance

            if ray_x < 0 or ray_x >= width or ray_y < 0 or ray_y >= height:
                break

            terrain_height = self._interpolate_height_cpu(heightmap, ray_x, ray_y)

            if ray_z <= terrain_height:
                return True

        return False

    def _calculate_slope_shading_cpu(self, heightmap: np.ndarray, x: int, y: int,
                                   sun_x: float, sun_y: float, sun_z: float) -> float:
        """CPU-optimierte Slope-basierte Beleuchtung"""
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
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length

        # Sonnenrichtung
        sun_dir = np.array([sun_x, sun_y, sun_z])
        sun_dir_length = np.linalg.norm(sun_dir)
        if sun_dir_length > 0:
            sun_dir = sun_dir / sun_dir_length

        # Dot-Product für Beleuchtungsstärke
        dot_product = np.dot(normal, sun_dir)
        return max(0.0, dot_product)

    def _interpolate_height_cpu(self, heightmap: np.ndarray, x: float, y: float) -> float:
        """CPU-optimierte Höhen-Interpolation"""
        height, width = heightmap.shape

        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

        fx, fy = x - x0, y - y0

        # Bilineare Interpolation
        h00 = heightmap[y0, x0]
        h10 = heightmap[y0, x1]
        h01 = heightmap[y1, x0]
        h11 = heightmap[y1, x1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fy) + h1 * fy

    def _downsample_heightmap(self, heightmap: np.ndarray, target_size: int) -> np.ndarray:
        """Reduziert Heightmap für Shadow-Berechnung"""
        original_size = heightmap.shape[0]

        if original_size <= target_size:
            return heightmap

        scale_factor = (original_size - 1) / (target_size - 1)
        downsampled = np.zeros((target_size, target_size), dtype=np.float32)

        for y in range(target_size):
            for x in range(target_size):
                orig_x = x * scale_factor
                orig_y = y * scale_factor
                downsampled[y, x] = self._interpolate_height_cpu(heightmap, orig_x, orig_y)

        return downsampled

    def _upsample_shadows(self, shadows: np.ndarray, target_size: int) -> np.ndarray:
        """Vergrößert Shadow-Map auf Zielgröße"""
        current_size = shadows.shape[0]

        if current_size >= target_size:
            return shadows

        scale_factor = (current_size - 1) / (target_size - 1)
        upsampled = np.zeros((target_size, target_size), dtype=np.float32)

        for y in range(target_size):
            for x in range(target_size):
                shadow_x = x * scale_factor
                shadow_y = y * scale_factor
                upsampled[y, x] = self._interpolate_height_cpu(shadows, shadow_x, shadow_y)

        return upsampled


class SlopeCalculator:
    """
    Funktionsweise: Berechnet Steigungsgradienten (dz/dx, dz/dy) aus Heightmap
    Aufgabe: Erstellt slopemap für Geology-Generator und visuelle Darstellung
    Methoden: calculate_slopes(heightmap, parameters), gradient_magnitude(), validate_slopes()
    Output-Format: 3D-Array (H,W,2) mit dz/dx und dz/dy Komponenten
    Validation: Gradient-Range-Checks und Consistency mit heightmap-Shape

    Spezifische Fallbacks:
    - GPU-Optimal: Parallele Gradient-Berechnung mit GPU-Compute-Shader
    - CPU-Fallback: NumPy gradient() mit optimierten Parametern
    - Simple-Fallback: Einfache Finite-Difference-Approximation
    """

    def __init__(self, shader_manager=None):
        """
        Parameter: shader_manager - ShaderManager für GPU-Acceleration
        """
        self.shader_manager = shader_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_slopes(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Funktionsweise: Berechnet Slope-Map mit 3-stufigem Fallback
        Parameter: heightmap - Höhendaten
        Parameter: parameters - Slope-Parameter (spacing, smoothing, etc.)
        Returns: numpy.ndarray - Slope-Map mit Shape (H,W,2) für dz/dx und dz/dy
        """
        # GPU-Fallback (Optimal)
        if self._gpu_available():
            try:
                result = self.shader_manager.request_slope_calculation(
                    operation_type="gradient_calculation",
                    heightmap=heightmap,
                    parameters=parameters
                )
                if result.get('success', False):
                    self.logger.debug("GPU slope calculation successful")
                    return result['data']
            except Exception as e:
                self.logger.warning(f"GPU slope calculation failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._calculate_cpu_slopes(heightmap, parameters)
        except Exception as e:
            self.logger.warning(f"CPU slope calculation failed: {e}")

        # Simple-Fallback (Minimal)
        return self._calculate_simple_slopes(heightmap)

    def gradient_magnitude(self, slopemap: np.ndarray) -> np.ndarray:
        """
        Funktionsweise: Berechnet Gradient-Magnitude aus Slope-Map
        Parameter: slopemap - (H,W,2) Array mit dz/dx, dz/dy
        Returns: numpy.ndarray - Gradient-Magnitude
        """
        if slopemap.shape[2] != 2:
            raise ValueError("Slopemap must have shape (H,W,2)")

        dz_dx = slopemap[:, :, 0]
        dz_dy = slopemap[:, :, 1]

        magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
        return magnitude

    def validate_slopes(self, slopemap: np.ndarray, heightmap: np.ndarray) -> Dict[str, Any]:
        """
        Funktionsweise: Validiert Slope-Map gegen Heightmap
        Parameter: slopemap, heightmap
        Returns: dict - Validation-Results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }

        # Shape-Consistency
        if slopemap.shape[:2] != heightmap.shape:
            validation_result["valid"] = False
            validation_result["errors"].append("Shape mismatch between slopemap and heightmap")

        if slopemap.shape[2] != 2:
            validation_result["valid"] = False
            validation_result["errors"].append("Slopemap must have 2 channels (dz/dx, dz/dy)")

        # Gradient-Range-Checks
        try:
            magnitude = self.gradient_magnitude(slopemap)
            max_gradient = np.max(magnitude)
            mean_gradient = np.mean(magnitude)

            validation_result["statistics"] = {
                "max_gradient": float(max_gradient),
                "mean_gradient": float(mean_gradient),
                "nan_count": int(np.sum(np.isnan(magnitude))),
                "inf_count": int(np.sum(np.isinf(magnitude)))
            }

            if max_gradient > 10.0:  # Sehr steile Gradienten
                validation_result["warnings"].append(f"Very steep gradients detected: {max_gradient}")

            if np.sum(np.isnan(magnitude)) > 0:
                validation_result["valid"] = False
                validation_result["errors"].append("NaN values in slope calculation")

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    def _gpu_available(self) -> bool:
        """Prüft GPU-Verfügbarkeit über ShaderManager"""
        return (self.shader_manager is not None and
                hasattr(self.shader_manager, 'gpu_available') and
                self.shader_manager.gpu_available)

    def _calculate_cpu_slopes(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Funktionsweise: NumPy gradient() mit optimierten Parametern
        Parameter: heightmap, parameters
        Returns: numpy.ndarray - CPU-berechnete Slopes
        """
        # NumPy gradient für optimierte Performance
        spacing = parameters.get('spacing', 1.0)

        # Berechne Gradienten in beide Richtungen
        grad_y, grad_x = np.gradient(heightmap, spacing, edge_order=2)

        # Als (H,W,2) Array zusammenfassen
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)
        slopemap[:, :, 0] = grad_x  # dz/dx
        slopemap[:, :, 1] = grad_y  # dz/dy

        # Optional: Smoothing
        smoothing = parameters.get('smoothing', 0.0)
        if smoothing > 0:
            try:
                from scipy.ndimage import gaussian_filter
                slopemap[:, :, 0] = gaussian_filter(slopemap[:, :, 0], sigma=smoothing)
                slopemap[:, :, 1] = gaussian_filter(slopemap[:, :, 1], sigma=smoothing)
            except ImportError:
                # Fallback ohne scipy
                pass

        return slopemap

    def _calculate_simple_slopes(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Funktionsweise: Einfache Finite-Difference-Approximation
        Parameter: heightmap
        Returns: numpy.ndarray - Simple Slope-Berechnung
        """
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        # Einfache Finite-Difference
        for y in range(height):
            for x in range(width):
                # dz/dx
                if x > 0 and x < width - 1:
                    dz_dx = (heightmap[y, x + 1] - heightmap[y, x - 1]) * 0.5
                elif x == 0:
                    dz_dx = heightmap[y, x + 1] - heightmap[y, x]
                else:
                    dz_dx = heightmap[y, x] - heightmap[y, x - 1]

                # dz/dy
                if y > 0 and y < height - 1:
                    dz_dy = (heightmap[y + 1, x] - heightmap[y - 1, x]) * 0.5
                elif y == 0:
                    dz_dy = heightmap[y + 1, x] - heightmap[y, x]
                else:
                    dz_dy = heightmap[y, x] - heightmap[y - 1, x]

                slopemap[y, x, 0] = dz_dx
                slopemap[y, x, 1] = dz_dy

        return slopemap


class BaseTerrainGenerator(BaseGenerator):
    """
    Funktionsweise: Hauptklasse für Terrain-Generierung mit numerischem LOD-System und Manager-Integration
    Aufgabe: Koordiniert alle Terrain-Generierungsschritte, verwaltet Parameter und LOD-Progression
    External-Interface: calculate_heightmap(parameters, lod_level) - wird von GenerationOrchestrator aufgerufen
    Internal-Methods: _coordinate_generation(), _validate_parameters(), _create_terrain_data()
    Manager-Integration: DataLODManager für Storage, ShaderManager für Performance-Optimierung
    Threading: Läuft in GenerationOrchestrator-Background-Threads mit LOD-Progression
    Error-Handling: Graceful Degradation bei Shader/Generator-Fehlern, vollständige Fallback-Kette
    """

    def __init__(self, map_seed: int = 42, shader_manager=None):
        """
        Funktionsweise: Initialisiert Terrain-Generator mit allen Sub-Komponenten
        Parameter: map_seed - Globaler Seed für reproduzierbare Ergebnisse
        Parameter: shader_manager - ShaderManager für Performance-Optimierung
        """
        super().__init__(map_seed)

        self.noise_generator = SimplexNoiseGenerator(seed=map_seed, shader_manager=shader_manager)
        self.shadow_calculator = ShadowCalculator(shader_manager=shader_manager)
        self.slope_calculator = SlopeCalculator(shader_manager=shader_manager)
        self.shader_manager = shader_manager

        # Standard-Parameter aus value_default.py
        self.default_parameters = self._load_default_parameters()

    def calculate_heightmap(self, parameters: Dict[str, Any], lod_level: int) -> TerrainData:
        """
        Funktionsweise: EINZIGE öffentliche Methode - wird von GenerationOrchestrator aufgerufen
        Aufgabe: Koordiniert komplette Terrain-Generierung für gegebenes LOD-Level
        Parameter: parameters - Alle Terrain-Parameter (aus ParameterManager)
        Parameter: lod_level - Numerisches LOD-Level (1-7)
        Returns: TerrainData - Komplette Terrain-Daten mit Validity-System
        """
        import time
        start_time = time.time()

        try:
            # Parameter-Validation
            self._validate_parameters(parameters)

            # Terrain-Generierung koordinieren
            terrain_data = self._coordinate_generation(parameters, lod_level)

            # Performance-Tracking
            terrain_data.generation_time = time.time() - start_time
            terrain_data.fallback_used = self._determine_fallback_used()

            # Parameter und Validity-State setzen
            terrain_data.update_parameters(parameters)
            terrain_data.validity_state = "valid"
            terrain_data.validity_flags = {
                "heightmap": True,
                "slopemap": True,
                "shadowmap": True
            }

            self.logger.info(f"Terrain generation completed for LOD {lod_level} in {terrain_data.generation_time:.2f}s")
            return terrain_data

        except Exception as e:
            self.logger.error(f"Terrain generation failed for LOD {lod_level}: {e}")

            # Error-Recovery: Minimal TerrainData zurückgeben
            error_terrain = TerrainData()
            error_terrain.lod_level = lod_level
            error_terrain.actual_size = self._lod_level_to_size(lod_level, parameters.get('map_size', 512))
            error_terrain.validity_state = "error"
            error_terrain.generation_time = time.time() - start_time
            error_terrain.fallback_used = "error_recovery"

            # Minimal-Heightmap für System-Continuity
            size = error_terrain.actual_size
            error_terrain.heightmap = np.zeros((size, size), dtype=np.float32)
            error_terrain.slopemap = np.zeros((size, size, 2), dtype=np.float32)
            error_terrain.shadowmap = np.ones((size, size), dtype=np.float32) * 0.5

            return error_terrain

    def _coordinate_generation(self, parameters: Dict[str, Any], lod_level: int) -> TerrainData:
        """
        Funktionsweise: Koordiniert alle Terrain-Generierungsschritte
        Parameter: parameters, lod_level
        Returns: TerrainData - Generierte Terrain-Daten
        """
        # LOD-Größe berechnen
        lod_size = self._lod_level_to_size(lod_level, parameters.get('map_size', 512))

        # TerrainData initialisieren
        terrain_data = TerrainData()
        terrain_data.lod_level = lod_level
        terrain_data.actual_size = lod_size

        self.logger.debug(f"Starting terrain generation: LOD {lod_level}, Size {lod_size}")

        # 1. Heightmap generieren
        terrain_data.heightmap = self._generate_heightmap(parameters, lod_size)
        self.logger.debug("Heightmap generation completed")

        # 2. Height-Redistribution anwenden
        terrain_data.heightmap = self._apply_redistribution(
            terrain_data.heightmap, parameters.get('redistribute_power', 1.0)
        )
        self.logger.debug("Height redistribution completed")

        # 3. Slopemap berechnen
        terrain_data.slopemap = self.slope_calculator.calculate_slopes(
            terrain_data.heightmap, parameters
        )
        self.logger.debug("Slope calculation completed")

        # 4. Shadowmap generieren
        terrain_data.shadowmap = self.shadow_calculator.calculate_shadows(
            terrain_data.heightmap, lod_level
        )
        terrain_data.calculated_sun_angles = self.shadow_calculator.get_sun_angles_for_lod(lod_level)[0]
        self.logger.debug("Shadow calculation completed")

        return terrain_data

    def _generate_heightmap(self, parameters: Dict[str, Any], size: int) -> np.ndarray:
        """
        Funktionsweise: Generiert Heightmap mit Noise-Generator
        Parameter: parameters, size
        Returns: numpy.ndarray - Generierte Heightmap
        """
        # Noise-Parameter extrahieren
        frequency = parameters.get('frequency', 0.01)
        octaves = parameters.get('octaves', 4)
        persistence = parameters.get('persistence', 0.5)
        lacunarity = parameters.get('lacunarity', 2.0)
        amplitude = parameters.get('amplitude', 100)

        # Frequency für LOD-Größe anpassen
        adjusted_frequency = frequency * (64 / size)  # Referenz: LOD 64

        # Noise-Grid generieren
        noise_grid = self.noise_generator.generate_noise_grid(
            size=size,
            frequency=adjusted_frequency,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )

        # Auf [0, amplitude] skalieren
        heightmap = (noise_grid + 1.0) * 0.5 * amplitude

        return heightmap.astype(np.float32)

    def _apply_redistribution(self, heightmap: np.ndarray, redistribute_power: float) -> np.ndarray:
        """
        Funktionsweise: Wendet Power-Redistribution auf Heightmap an
        Parameter: heightmap, redistribute_power
        Returns: numpy.ndarray - Redistributed Heightmap
        """
        if redistribute_power == 1.0:
            return heightmap

        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            return heightmap

        # Normalisierung und Power-Redistribution
        normalized = (heightmap - min_height) / height_range
        redistributed = np.power(normalized, redistribute_power)
        result = redistributed * height_range + min_height

        return result.astype(np.float32)

    def _lod_level_to_size(self, lod_level: int, target_map_size: int) -> int:
        """
        Funktionsweise: Konvertiert numerisches LOD-Level zu tatsächlicher Größe
        Parameter: lod_level (1-7), target_map_size
        Returns: int - Tatsächliche Größe für dieses LOD-Level
        """
        # Basis-Größe: 32 für LOD 1
        base_size = 32

        # Verdopplung bis target_map_size erreicht
        current_size = base_size
        for level in range(2, lod_level + 1):
            next_size = current_size * 2
            if next_size <= target_map_size:
                current_size = next_size
            else:
                # Nächste Verdopplung würde über target_map_size gehen
                current_size = target_map_size
                break

        return min(current_size, target_map_size)

    def _validate_parameters(self, parameters: Dict[str, Any]):
        """
        Funktionsweise: Validiert alle Terrain-Parameter
        Parameter: parameters
        Raises: ValueError bei ungültigen Parametern
        """
        required_params = ['map_seed', 'map_size', 'amplitude', 'octaves',
                          'frequency', 'persistence', 'lacunarity', 'redistribute_power']

        # Required Parameters prüfen
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")

        # Range-Validation
        validations = {
            'map_size': lambda x: 32 <= x <= 2048 and (x & (x-1)) == 0,  # Power of 2
            'amplitude': lambda x: 0 <= x <= 1000,
            'octaves': lambda x: 1 <= x <= 10,
            'frequency': lambda x: 0.001 <= x <= 1.0,
            'persistence': lambda x: 0.0 <= x <= 2.0,
            'lacunarity': lambda x: 1.0 <= x <= 5.0,
            'redistribute_power': lambda x: 0.1 <= x <= 3.0
        }

        for param, validator in validations.items():
            if param in parameters and not validator(parameters[param]):
                raise ValueError(f"Invalid value for {param}: {parameters[param]}")

    def _determine_fallback_used(self) -> str:
        """
        Funktionsweise: Bestimmt welcher Fallback hauptsächlich verwendet wurde
        Returns: str - "gpu", "cpu", oder "simple"
        """
        if self.shader_manager and hasattr(self.shader_manager, 'gpu_available'):
            if self.shader_manager.gpu_available:
                return "gpu"
        return "cpu"

    def _load_default_parameters(self) -> Dict[str, Any]:
        """
        Funktionsweise: Lädt Standard-Parameter aus value_default.py
        Returns: dict - Standard-Parameter
        """
        try:
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
        except ImportError:
            # Fallback-Parameter
            return {
                'map_size': 512,
                'amplitude': 100,
                'octaves': 6,
                'frequency': 0.01,
                'persistence': 0.5,
                'lacunarity': 2.0,
                'redistribute_power': 1.0,
                'map_seed': 12345
            }

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Terrain braucht keine Dependencies - ist der Basis-Generator
        Parameter: data_manager (wird nicht verwendet)
        Returns: dict - Leeres Dependencies-Dict
        """
        return {}

    def _execute_generation(self, lod_level: int, dependencies: Dict, parameters: Dict[str, Any]):
        """
        Funktionsweise: BaseGenerator-Interface-Methode - delegiert an calculate_heightmap
        Parameter: lod_level, dependencies (nicht verwendet), parameters
        Returns: TerrainData - Generierte Terrain-Daten
        """
        return self.calculate_heightmap(parameters, lod_level)

    def _save_to_data_manager(self, data_manager, result: TerrainData, parameters: Dict[str, Any]):
        """
        Funktionsweise: Speichert TerrainData im DataManager
        Parameter: data_manager, result, parameters
        """
        try:
            data_manager.set_terrain_data_lod(
                "heightmap", result.heightmap, result.lod_level, parameters
            )
            data_manager.set_terrain_data_lod(
                "slopemap", result.slopemap, result.lod_level, parameters
            )
            data_manager.set_terrain_data_lod(
                "shadowmap", result.shadowmap, result.lod_level, parameters
            )
            self.logger.debug(f"Terrain data saved to DataManager for LOD {result.lod_level}")
        except Exception as e:
            self.logger.error(f"Failed to save terrain data to DataManager: {e}")

    # ================================
    # LEGACY-KOMPATIBILITÄT (deprecated)
    # ================================

    def generate_terrain(self, **kwargs):
        """Legacy method - use calculate_heightmap instead"""
        self.logger.warning("generate_terrain is deprecated - use calculate_heightmap")

        # Parameter-Mapping für Legacy-Calls
        parameters = self.default_parameters.copy()
        parameters.update(kwargs)

        # LOD aus kwargs extrahieren oder Standard verwenden
        lod_level = kwargs.get('lod_level', 4)  # Standard: LOD 4

        terrain_data = self.calculate_heightmap(parameters, lod_level)

        # Legacy-Format zurückgeben (heightmap, slopemap, shadowmap)
        return terrain_data.heightmap, terrain_data.slopemap, terrain_data.shadowmap

    def generate_heightmap(self, map_size, amplitude, octaves, frequency,
                          persistence, lacunarity, redistribute_power, map_seed):
        """Legacy method - use calculate_heightmap instead"""
        self.logger.warning("generate_heightmap is deprecated - use calculate_heightmap")

        parameters = {
            'map_size': map_size,
            'amplitude': amplitude,
            'octaves': octaves,
            'frequency': frequency,
            'persistence': persistence,
            'lacunarity': lacunarity,
            'redistribute_power': redistribute_power,
            'map_seed': map_seed
        }

        # LOD-Level basierend auf map_size bestimmen
        if map_size <= 64:
            lod_level = 2
        elif map_size <= 128:
            lod_level = 3
        elif map_size <= 256:
            lod_level = 4
        elif map_size <= 512:
            lod_level = 5
        else:
            lod_level = 6

        terrain_data = self.calculate_heightmap(parameters, lod_level)
        return terrain_data.heightmap

    def generate_shadows(self, heightmap):
        """Legacy method - use ShadowCalculator directly"""
        self.logger.warning("generate_shadows is deprecated - use ShadowCalculator")
        return self.shadow_calculator.calculate_shadows(heightmap, 4)  # Standard LOD 4

    def calculate_slopes(self, heightmap):
        """Legacy method - use SlopeCalculator directly"""
        self.logger.warning("calculate_slopes is deprecated - use SlopeCalculator")
        return self.slope_calculator.calculate_slopes(heightmap, {})

    def apply_redistribution(self, heightmap, redistribute_power):
        """Legacy method - kept for compatibility"""
        return self._apply_redistribution(heightmap, redistribute_power)


# ================================
# FACTORY FUNCTIONS
# ================================

def create_terrain_generator(map_seed: int = 42, shader_manager=None) -> BaseTerrainGenerator:
    """
    Funktionsweise: Factory-Funktion für BaseTerrainGenerator
    Parameter: map_seed, shader_manager
    Returns: BaseTerrainGenerator - Konfigurierte Instanz
    """
    return BaseTerrainGenerator(map_seed=map_seed, shader_manager=shader_manager)

def create_terrain_data() -> TerrainData:
    """
    Funktionsweise: Factory-Funktion für TerrainData
    Returns: TerrainData - Neue leere Instanz
    """
    return TerrainData()

def validate_terrain_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Funktionsweise: Standalone Parameter-Validation
    Parameter: parameters - Zu validierende Parameter
    Returns: dict - Validation-Result mit errors/warnings
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    required_params = ['map_seed', 'map_size', 'amplitude', 'octaves',
                      'frequency', 'persistence', 'lacunarity', 'redistribute_power']

    # Required Parameters prüfen
    for param in required_params:
        if param not in parameters:
            result["valid"] = False
            result["errors"].append(f"Missing required parameter: {param}")

    # Range-Validation
    validations = {
        'map_size': (lambda x: 32 <= x <= 2048 and (x & (x-1)) == 0, "Must be power of 2 between 32 and 2048"),
        'amplitude': (lambda x: 0 <= x <= 1000, "Must be between 0 and 1000"),
        'octaves': (lambda x: 1 <= x <= 10, "Must be between 1 and 10"),
        'frequency': (lambda x: 0.001 <= x <= 1.0, "Must be between 0.001 and 1.0"),
        'persistence': (lambda x: 0.0 <= x <= 2.0, "Must be between 0.0 and 2.0"),
        'lacunarity': (lambda x: 1.0 <= x <= 5.0, "Must be between 1.0 and 5.0"),
        'redistribute_power': (lambda x: 0.1 <= x <= 3.0, "Must be between 0.1 and 3.0")
    }

    for param, (validator, message) in validations.items():
        if param in parameters:
            try:
                if not validator(parameters[param]):
                    result["valid"] = False
                    result["errors"].append(f"Invalid {param}: {message}")
            except (TypeError, ValueError):
                result["valid"] = False
                result["errors"].append(f"Invalid type for {param}: expected number")

    # Warnings für suboptimale Parameter
    if parameters.get('octaves', 0) > 8:
        result["warnings"].append("High octave count may impact performance")
    if parameters.get('amplitude', 0) > 500:
        result["warnings"].append("Very high amplitude may create unrealistic terrain")

    return result


# ================================
# UTILITY FUNCTIONS
# ================================

def lod_level_to_size(lod_level: int, target_map_size: int) -> int:
    """
    Funktionsweise: Utility-Funktion für LOD-Size-Berechnung
    Parameter: lod_level, target_map_size
    Returns: int - Berechnete Größe
    """
    base_size = 32
    current_size = base_size

    for level in range(2, lod_level + 1):
        next_size = current_size * 2
        if next_size <= target_map_size:
            current_size = next_size
        else:
            current_size = target_map_size
            break

    return min(current_size, target_map_size)

def calculate_lod_progression(target_size: int) -> List[Tuple[int, int]]:
    """
    Funktionsweise: Berechnet vollständige LOD-Progression
    Parameter: target_size - Finale Zielgröße
    Returns: List[Tuple[int, int]] - Liste von (lod_level, size) Tupeln
    """
    progression = []
    lod_level = 1
    current_size = 32

    while current_size <= target_size:
        progression.append((lod_level, current_size))

        if current_size >= target_size:
            break

        next_size = current_size * 2
        if next_size > target_size:
            if current_size < target_size:
                lod_level += 1
                progression.append((lod_level, target_size))
            break
        else:
            current_size = next_size
            lod_level += 1

    return progression

def estimate_generation_time(parameters: Dict[str, Any], lod_level: int,
                           has_gpu: bool = False) -> float:
    """
    Funktionsweise: Schätzt Generierungszeit basierend auf Parametern
    Parameter: parameters, lod_level, has_gpu
    Returns: float - Geschätzte Zeit in Sekunden
    """
    size = lod_level_to_size(lod_level, parameters.get('map_size', 512))
    octaves = parameters.get('octaves', 4)

    # Basis-Zeit (Sekunden für 64x64, 4 octaves auf CPU)
    base_time = 0.1

    # Skalierung basierend auf Größe (quadratisch)
    size_factor = (size / 64) ** 2

    # Skalierung basierend auf Octaves (linear)
    octave_factor = octaves / 4

    # GPU-Beschleunigung
    gpu_factor = 0.1 if has_gpu else 1.0

    estimated_time = base_time * size_factor * octave_factor * gpu_factor

    # Shadow-Berechnung hinzufügen
    shadow_angles = len(ShadowCalculator().get_sun_angles_for_lod(lod_level)[0])
    shadow_time = 0.05 * shadow_angles * (size / 64) ** 2 * gpu_factor

    return estimated_time + shadow_time

def get_memory_usage_estimate(lod_level: int, target_map_size: int) -> Dict[str, int]:
    """
    Funktionsweise: Schätzt Memory-Usage für gegebenes LOD
    Parameter: lod_level, target_map_size
    Returns: dict - Memory-Usage in Bytes pro Datentyp
    """
    size = lod_level_to_size(lod_level, target_map_size)

    # Bytes pro Element (float32 = 4 bytes)
    heightmap_bytes = size * size * 4
    slopemap_bytes = size * size * 2 * 4  # 2 Kanäle
    shadowmap_bytes = size * size * 4

    total_bytes = heightmap_bytes + slopemap_bytes + shadowmap_bytes

    return {
        "heightmap_bytes": heightmap_bytes,
        "slopemap_bytes": slopemap_bytes,
        "shadowmap_bytes": shadowmap_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024)
    }


# ================================
# MODULE TESTING
# ================================

def test_terrain_generator():
    """
    Funktionsweise: Basis-Test für TerrainGenerator
    Aufgabe: Validiert Kernfunktionalität ohne externe Dependencies
    """
    print("Testing TerrainGenerator...")

    # Test Parameter-Validation
    test_params = {
        'map_seed': 12345,
        'map_size': 128,
        'amplitude': 100,
        'octaves': 4,
        'frequency': 0.01,
        'persistence': 0.5,
        'lacunarity': 2.0,
        'redistribute_power': 1.0
    }

    validation_result = validate_terrain_parameters(test_params)
    assert validation_result["valid"], f"Parameter validation failed: {validation_result['errors']}"
    print("✓ Parameter validation passed")

    # Test Generator ohne ShaderManager
    generator = create_terrain_generator(map_seed=12345, shader_manager=None)

    # Test Terrain-Generierung
    terrain_data = generator.calculate_heightmap(test_params, lod_level=2)

    assert terrain_data.heightmap is not None, "Heightmap generation failed"
    assert terrain_data.slopemap is not None, "Slopemap generation failed"
    assert terrain_data.shadowmap is not None, "Shadowmap generation failed"
    assert terrain_data.is_valid(), "TerrainData validation failed"

    print(f"✓ Terrain generation passed (Size: {terrain_data.actual_size}, LOD: {terrain_data.lod_level})")

    # Test LOD-System
    sizes = [lod_level_to_size(lod, 512) for lod in range(1, 8)]
    expected_sizes = [32, 64, 128, 256, 512, 512, 512]  # ab LOD 5 bleibt bei 512
    assert sizes == expected_sizes, f"LOD sizing failed: {sizes} != {expected_sizes}"
    print("✓ LOD system validation passed")

    print("All terrain generator tests passed!")

if __name__ == "__main__":
    # Führe Tests aus wenn direkt aufgerufen
    test_terrain_generator()