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
    - GPU-Optimal: ShaderManager.process_noise_generation() für parallele Multi-Octave-Berechnung
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
        self.seed = seed
        self.shader_manager = shader_manager
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_seed(self, seed: int) -> None:
        """
        Setzt den Noise-Seed neu, falls er vom aktuellen abweicht. Nötig, weil
        BaseTerrainGenerator (und damit dieser SimplexNoiseGenerator) über
        GenerationOrchestrator.get_generator_instance() dauerhaft gecacht wird
        - ohne explizites Reseed blieb der Seed für die gesamte App-Laufzeit
        auf dem Wert eingefroren, mit dem der Generator beim allerersten
        Generate-Klick konstruiert wurde (Änderungen am map_seed-Slider hatten
        dadurch keine Wirkung auf die Heightmap).
        """
        if seed != self.seed:
            self.generator = OpenSimplex(seed=seed)
            self.seed = seed

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

        # GPU-Pfad: shaders/terrain/noiseGeneration.comp implementiert jetzt echtes
        # Gradient-Noise (Hash-basierte OpenSimplex-Variante, siehe Shader-Kommentar)
        # statt der ursprünglichen sin(x)*cos(y)-Platzhalter-Formel - re-aktiviert,
        # nachdem die alte "if False and"-Deaktivierung (aus einer Zeit, in der
        # shader_manager praktisch immer None war und der Shader selbst nur der
        # Platzhalter war) beide Voraussetzungen nicht mehr zutreffen.
        if self._gpu_available() and offset_x == 0 and offset_y == 0:
            try:
                result = self.shader_manager.process_noise_generation(
                    size=size, octaves=octaves, frequency=frequency,
                    persistence=persistence, lacunarity=lacunarity, seed=self.seed
                )
                if result is not None:
                    self.logger.debug("GPU noise generation successful")
                    return result
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

        # Keine GPU-Beschleunigung: ShaderManager bietet keine Interpolations-Methode an
        # (im Gegensatz zu Noise-Generierung und Shadow-Raycast gibt es hier keine
        # process_*-Entsprechung) - direkt auf die CPU-Interpolation gehen.
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

        # Koordinaten-Arrays für gesamtes Grid erstellen (vectorized).
        # Pixel-Koordinaten (0..size), NICHT auf [0,1] normalisiert: bei
        # normalisierten Koordinaten landet frequency*coord (z.B. 0.037*1=0.037)
        # in einem winzigen Ausschnitt des Noise-Raums nahe (0,0), wo Simplex-
        # Noise praktisch konstant ist - das Ergebnis war eine fast flache
        # Heightmap ohne echte Berge, unabhängig von amplitude/redistribute_power.
        x_coords = np.arange(size, dtype=np.float64) + offset_x
        y_coords = np.arange(size, dtype=np.float64) + offset_y
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


_SOLAR_K = np.pi / 180.0
# Referenztage für die 6 Zwei-Monats-Perioden (01.Jan/01.Mär/01.Mai/01.Jul/
# 01.Sep/01.Nov), Tageszahl via "(monat-1)*30.3+datum" (geoastro.de-Formel).
_SEASONAL_REFERENCE_DAYS_OF_YEAR = [1.0, 61.6, 122.2, 182.8, 243.4, 304.0]
_SEASONAL_DAYTIME_HOURS = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]


def calculate_solar_position(day_of_year: float, hour: float, latitude: float,
                              longitude: float) -> Tuple[float, float]:
    """
    Funktionsweise: Echte astronomische Sonnenstandsberechnung (Deklination,
    Zeitgleichung, Stundenwinkel -> Höhe/Azimut), Quelle: geoastro.de/SME/
    tk/index.htm.
    Parameter: day_of_year - Tageszahl im Jahr (1-365, kann fraktional sein),
    hour - Stunde inkl. Bruchteil (z.B. 14.5 = 14:30), latitude/longitude -
    Grad (longitude Ost positiv)
    Returns: (elevation_deg, azimuth_deg) - Elevation auf >=0 geklemmt (Sonne
    unter dem Horizont wird nicht negativ zurückgegeben)
    """
    declination = -23.45 * np.cos(_SOLAR_K * 360 * (day_of_year + 10) / 365)
    equation_of_time = 60 * (
        -0.171 * np.sin(0.0337 * day_of_year + 0.465)
        - 0.1299 * np.sin(0.01787 * day_of_year - 0.168)
    )
    hour_angle = 15 * (hour - (15.0 - longitude) / 15.0 - 12 + equation_of_time / 60)

    x = (np.sin(_SOLAR_K * latitude) * np.sin(_SOLAR_K * declination)
         + np.cos(_SOLAR_K * latitude) * np.cos(_SOLAR_K * declination)
         * np.cos(_SOLAR_K * hour_angle))
    x = float(np.clip(x, -1.0, 1.0))
    elevation = np.arcsin(x) / _SOLAR_K

    denom = np.cos(_SOLAR_K * latitude) * np.sin(np.arccos(x))
    if abs(denom) < 1e-9:
        azimuth = 180.0
    else:
        y = float(np.clip(
            -(np.sin(_SOLAR_K * latitude) * x - np.sin(_SOLAR_K * declination)) / denom,
            -1.0, 1.0))
        azimuth_raw = np.arccos(y) / _SOLAR_K
        solar_noon_hour = 12 + (15.0 - longitude) / 15.0 - equation_of_time / 60
        azimuth = azimuth_raw if hour <= solar_noon_hour else 360.0 - azimuth_raw

    return max(0.0, float(elevation)), float(azimuth)


def generate_seasonal_sun_angles(month_index: int, latitude: float,
                                  longitude: float) -> List[Tuple[float, float]]:
    """
    Funktionsweise: 7 (elevation, azimuth)-Paare für eine der 6 saisonalen
    Zwei-Monats-Perioden, über calculate_solar_position() für den jeweiligen
    Referenztag und 7 Tageszeiten (6-18 Uhr) berechnet - ersetzt
    ShadowCalculator.sun_angles 1:1 für diesen Monat (gleiche Länge 7,
    dieselbe LOD-Index-Filterung in get_sun_angles_for_lod() bleibt gültig).
    Parameter: month_index - 0..5 (Jan/Feb .. Nov/Dez), latitude/longitude - Grad
    """
    day_of_year = _SEASONAL_REFERENCE_DAYS_OF_YEAR[month_index]
    return [calculate_solar_position(day_of_year, h, latitude, longitude)
            for h in _SEASONAL_DAYTIME_HOURS]


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
                         existing_lod: int = 1,
                         sun_angles_override: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Funktionsweise: Berechnet Verschattung mit 3-stufigem Fallback und LOD-System
        Aufgabe: Erstellt shadowmap (konstant 64x64) für Weather-System
        Parameter: heightmap - Höhendaten
        Parameter: lod_level - LOD-Level für Sonnenwinkel-Auswahl
        Parameter: existing_shadows - Bestehende Shadow-Daten (optional)
        Parameter: existing_lod - LOD-Level der bestehenden Shadows
        Parameter: sun_angles_override - ersetzt self.sun_angles für diesen
            Aufruf (z.B. saisonal berechnete Sonnenwinkel via
            generate_seasonal_sun_angles() für Weathers Monats-Simulation) -
            self.sun_weights bleibt unverändert, get_sun_angles_for_lod()s
            Index-Filterung bleibt gültig (gleiche Länge 7). None = bisheriges
            Verhalten (feste self.sun_angles-Tabelle, ein Tag für alle Monate).
        Returns: numpy.ndarray - Shadow-Map konstant 64x64
        """
        # GPU-Fallback (Optimal) - ShaderManager.process_shadow_raycast() rechnet nur
        # einen Sonnenwinkel pro Aufruf, daher hier über die LOD-Winkel loopen und wie
        # im CPU-Pfad gewichtet kombinieren. Progressive Shadow-Enhancement (existing_
        # shadows/existing_lod) wird hier nicht nachgebildet, da calculate_heightmap()
        # sie in der aktuellen Pipeline nie mit gesetzten Werten aufruft.
        if self._gpu_available():
            try:
                return self._calculate_gpu_shadows(heightmap, lod_level, sun_angles_override)
            except Exception as e:
                self.logger.warning(f"GPU shadow calculation failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._calculate_cpu_shadows(heightmap, lod_level, existing_shadows, existing_lod,
                                                sun_angles_override)
        except Exception as e:
            self.logger.warning(f"CPU shadow calculation failed: {e}")

        # Simple-Fallback (Minimal)
        return self._calculate_simple_shadows(heightmap, lod_level)

    def get_sun_angles_for_lod(
            self, lod_level: int,
            sun_angles_override: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Funktionsweise: Gibt passende Sonnenwinkel-Auswahl für LOD-Level zurück
        Parameter: lod_level (int) - 1,2,3,4,5,6,7
        Parameter: sun_angles_override - falls gesetzt, wird diese 7er-Liste
            statt self.sun_angles indiziert (siehe calculate_shadows()) -
            self.sun_weights bleibt in jedem Fall die Quelle der Gewichtung.
        Returns: Tuple (sun_angles_list, sun_weights_list) - Gefilterte Winkel und Gewichtungen
        """
        angles_source = sun_angles_override if sun_angles_override is not None else self.sun_angles

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

        selected_angles = [angles_source[i] for i in indices]
        selected_weights = [self.sun_weights[i] for i in indices]

        return selected_angles, selected_weights

    def _gpu_available(self) -> bool:
        """Prüft GPU-Verfügbarkeit über ShaderManager"""
        return (self.shader_manager is not None and
                hasattr(self.shader_manager, 'gpu_available') and
                self.shader_manager.gpu_available)

    def _calculate_cpu_shadows(self, heightmap: np.ndarray, lod_level: int,
                              existing_shadows: Optional[np.ndarray] = None,
                              existing_lod: int = 1,
                              sun_angles_override: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Funktionsweise: Optimierte CPU-Raycast-Implementierung
        Parameter: heightmap, lod_level, existing_shadows, existing_lod, sun_angles_override
        Returns: numpy.ndarray - Shadow-Map in heightmap-Auflösung (intern bei 64x64 gerechnet)
        """
        shadow_resolution = 64
        original_size = heightmap.shape[0]

        # Heightmap für Shadow-Berechnung immer auf 64x64 bringen (in beide Richtungen -
        # vorher wurde bei original_size < 64, z.B. LOD1 mit 32x32, nicht hochskaliert,
        # wodurch shadow_heightmap bei 32x32 blieb während "shadows" weiter hart auf
        # 64x64 initialisiert wurde: "operands could not be broadcast together with
        # shapes (64,64) (32,32) (64,64)" bei jedem LOD1-Lauf)
        shadow_heightmap = self._resize_2d(heightmap, shadow_resolution)

        # Progressive Shadow-Enhancement falls bestehende Shadows vorhanden
        if existing_shadows is not None and existing_lod != lod_level:
            return self._calculate_progressive_shadows(shadow_heightmap, lod_level,
                                                     existing_shadows, existing_lod)

        # Vollständige Shadow-Berechnung
        sun_angles, sun_weights = self.get_sun_angles_for_lod(lod_level, sun_angles_override)
        shadows = np.zeros((shadow_resolution, shadow_resolution), dtype=np.float32)
        total_weight = sum(sun_weights)

        for i, (elevation, azimuth) in enumerate(sun_angles):
            shadow_map = self._raycast_shadow_cpu(shadow_heightmap, elevation, azimuth)
            shadows += shadow_map * sun_weights[i]

        # Normalisierung
        shadows /= total_weight

        # Auf Original-Größe zurückskalieren falls nötig (in beide Richtungen)
        if original_size != shadow_resolution:
            shadows = self._resize_2d(shadows, original_size)

        return shadows

    def _calculate_gpu_shadows(self, heightmap: np.ndarray, lod_level: int,
                              sun_angles_override: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
        """
        Funktionsweise: GPU-Raycast-Implementierung über ShaderManager.process_shadow_raycast()
        Aufgabe: Rechnet - wie der CPU-Pfad - pro LOD-Sonnenwinkel einen Raycast-Pass und
                 kombiniert die Ergebnisse gewichtet; process_shadow_raycast() selbst rechnet
                 nur einen Winkel pro Aufruf (kein Batch-Modus vorhanden).
        Parameter: heightmap, lod_level, sun_angles_override
        Returns: numpy.ndarray - Shadow-Map in heightmap-Auflösung (intern bei 64x64 gerechnet)
        """
        shadow_resolution = 64
        original_size = heightmap.shape[0]
        shadow_heightmap = self._resize_2d(heightmap, shadow_resolution)

        sun_angles, sun_weights = self.get_sun_angles_for_lod(lod_level, sun_angles_override)
        shadows = np.zeros((shadow_resolution, shadow_resolution), dtype=np.float32)
        total_weight = sum(sun_weights)

        for (elevation, azimuth), weight in zip(sun_angles, sun_weights):
            shadow_map = self.shader_manager.process_shadow_raycast(
                shadow_heightmap, elevation, azimuth, shadow_resolution
            )
            if shadow_map is None:
                raise RuntimeError("process_shadow_raycast returned no data")
            shadows += shadow_map * weight

        shadows /= total_weight

        if original_size != shadow_resolution:
            shadows = self._resize_2d(shadows, original_size)

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

    def _resize_2d(self, grid: np.ndarray, target_size: int) -> np.ndarray:
        """
        Bilineare Größenänderung eines 2D-Grids in beide Richtungen (vereinheitlicht
        die vormals getrennten, nur-eine-Richtung-fähigen _downsample_heightmap()/
        _upsample_shadows() - deren Richtungs-Guards ließen z.B. eine 32x32-Heightmap
        bei einer Ziel-Shadow-Resolution von 64 unverändert bei 32x32, siehe
        _calculate_cpu_shadows()).
        """
        original_size = grid.shape[0]

        if original_size == target_size:
            return grid

        scale_factor = (original_size - 1) / (target_size - 1)
        resized = np.zeros((target_size, target_size), dtype=np.float32)

        for y in range(target_size):
            for x in range(target_size):
                orig_x = x * scale_factor
                orig_y = y * scale_factor
                resized[y, x] = self._interpolate_height_cpu(grid, orig_x, orig_y)

        return resized


class SlopeCalculator:
    """
    Funktionsweise: Berechnet Steigungsgradienten (dz/dx, dz/dy) aus Heightmap
    Aufgabe: Erstellt slopemap für Geology-Generator und visuelle Darstellung
    Methoden: calculate_slopes(heightmap, parameters), gradient_magnitude(), validate_slopes()
    Output-Format: 3D-Array (H,W,2) mit dz/dx und dz/dy Komponenten
    Validation: Gradient-Range-Checks und Consistency mit heightmap-Shape

    Spezifische Fallbacks:
    - Kein GPU-Pfad: ShaderManager bietet keine Slope-/Gradient-Berechnung an
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
        # Keine GPU-Beschleunigung: ShaderManager bietet keine Slope-/Gradient-Methode
        # an (im Gegensatz zu Noise-Generierung und Shadow-Raycast gibt es hier keine
        # process_*-Entsprechung) - direkt auf die CPU-Berechnung gehen.

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
        # NumPy gradient für optimierte Performance. spacing = reale Meter pro Pixel,
        # NICHT 1.0 - die Karte deckt immer TERRAIN.WORLD_SIZE_KM x WORLD_SIZE_KM ab
        # (siehe gui/widgets/map_display_3d.py), unabhängig von der Pixelauflösung.
        # Ein fester spacing=1.0 hieß: 1m Höhenunterschied zwischen Nachbarpixeln wird
        # wie 1m realer Horizontal-Abstand behandelt, obwohl ein Pixel bei typischen
        # Kartengrößen tatsächlich ~50-300m Horizontal-Abstand abdeckt - das ergab
        # Gradienten um ~10-15 (entspricht ~85-89°) auf praktisch der gesamten Karte.
        if 'spacing' in parameters:
            spacing = parameters['spacing']
        else:
            from gui.config.value_default import TERRAIN
            world_size_m = TERRAIN.WORLD_SIZE_KM * 1000.0
            spacing = world_size_m / heightmap.shape[0]

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


class BaseTerrainGenerator:
    """
    Funktionsweise: Hauptklasse für Terrain-Generierung mit numerischem LOD-System und Manager-Integration
    Aufgabe: Koordiniert alle Terrain-Generierungsschritte, verwaltet Parameter und LOD-Progression
    External-Interface: calculate_heightmap(parameters, lod_level) - wird von GenerationOrchestrator aufgerufen
    Internal-Methods: _coordinate_generation(), _validate_parameters(), _create_terrain_data()
    Manager-Integration: DataLODManager für Storage, ShaderManager für Performance-Optimierung
    Threading: Läuft in GenerationOrchestrator-Background-Threads mit LOD-Progression
    Error-Handling: Graceful Degradation bei Shader/Generator-Fehlern, vollständige Fallback-Kette
    """

    def __init__(self, map_seed: int = 42, shader_manager=None, data_lod_manager=None):
        """
        Funktionsweise: Initialisiert Terrain-Generator mit allen Sub-Komponenten
        Parameter: map_seed - Globaler Seed für reproduzierbare Ergebnisse
        Parameter: shader_manager - ShaderManager für Performance-Optimierung
        Parameter: data_lod_manager - DataLODManager für feingranularen Calculator-
            Storage (siehe set_calculator_output()/get_calculator_output()). Die echte
            Pipeline injiziert immer eine Instanz über GenerationOrchestrator.
            get_generator_instance(); bleibt sie None (Standalone-Nutzung/Tests), wird
            beim ersten Bedarf lazy eine eigene erzeugt (siehe _ensure_data_lod_manager()).
        """
        self.map_seed = map_seed
        self.logger = logging.getLogger(self.__class__.__name__)

        self.noise_generator = SimplexNoiseGenerator(seed=map_seed, shader_manager=shader_manager)
        self.shadow_calculator = ShadowCalculator(shader_manager=shader_manager)
        self.slope_calculator = SlopeCalculator(shader_manager=shader_manager)
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager

        # Parameter der aktuell laufenden Generierungs-Anfrage - vom
        # GenerationOrchestrator einmal pro frischer Anfrage über
        # set_active_parameters() gesetzt, bleibt über alle LOD-Runden dieser
        # Anfrage hinweg konstant (ersetzt das frühere context-dict-basierte
        # Parameter-Handling, das nur innerhalb EINES calculate_heightmap()-Aufrufs
        # existierte - die _calc_*-Methoden werden jetzt vom globalen
        # CalculatorDispatcher einzeln aufgerufen, nicht mehr alle zusammen).
        self._current_parameters: Dict[str, Any] = {}

        # Standard-Parameter aus value_default.py
        self.default_parameters = self._load_default_parameters()

    def set_active_parameters(self, parameters: Dict[str, Any]):
        """Setzt die Parameter, die alle _calc_*-Methoden bis zur nächsten frischen
        Anfrage verwenden (vom GenerationOrchestrator aufgerufen)."""
        self._current_parameters = parameters

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, calculate_heightmap() ohne
        injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from gui.OldManagers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    def calculate_heightmap(self, parameters: Dict[str, Any], lod_level: int) -> TerrainData:
        """
        Funktionsweise: Standalone-Convenience-Entry-Point (Legacy-Kompatibilität + Tests)
        Aufgabe: Führt alle 4 Terrain-Calculator-Knoten synchron für EIN LOD aus und
            liefert das fertige TerrainData-Objekt. Die echte GUI-Pipeline
            (GenerationOrchestrator) ruft dieselben _calc_*-Methoden ab jetzt einzeln
            über den globalen CalculatorDispatcher auf (siehe
            gui/OldManagers/calculator_graph.py, Tracker #16 LOD-Lockstep-Umbau) -
            der Effekt ist identisch, da beide Wege dieselben Methoden und denselben
            Storage nutzen.
        Parameter: parameters - Alle Terrain-Parameter (aus ParameterManager)
        Parameter: lod_level - Numerisches LOD-Level (1-7)
        Returns: TerrainData - Komplette Terrain-Daten mit Validity-System
        """
        import time
        start_time = time.time()

        try:
            self._validate_parameters(parameters)
            self._ensure_data_lod_manager()
            self.set_active_parameters(parameters)

            self.logger.debug(f"Starting terrain generation: LOD {lod_level}")

            self._calc_noise("terrain.noise", lod_level)
            self._calc_redistribution("terrain.redistribution", lod_level)
            self._calc_slope("terrain.slope", lod_level)
            self._calc_shadow("terrain.shadow", lod_level)

            terrain_data = self.assemble_terrain_data(lod_level, parameters)
            terrain_data.generation_time = time.time() - start_time

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

    def assemble_terrain_data(self, lod_level: int, parameters: Dict[str, Any]) -> TerrainData:
        """
        Funktionsweise: Baut das finale TerrainData-Objekt aus den einzeln gespeicherten
        Calculator-Outputs zusammen
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald alle 4 Terrain-
            Calculator-Knoten ein LOD abgeschlossen haben (siehe Task 18 im
            LOD-Lockstep-Umbau) - analog zu den entsprechenden assemble_*_data()-
            Methoden der anderen 5 Generatoren (nimmt parameters entgegen, damit
            der Orchestrator alle 6 assemble_*_data()-Methoden einheitlich
            aufrufen kann).
        Parameter: lod_level - Numerisches LOD-Level
        Parameter: parameters - Alle Terrain-Parameter (für update_parameters())
        Returns: TerrainData - Komplette, fertig validierte Terrain-Daten
        """
        heightmap = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        slopemap = self.data_lod_manager.get_calculator_output("terrain.slope", "slopemap", lod_level)
        shadowmap = self.data_lod_manager.get_calculator_output("terrain.shadow", "shadowmap", lod_level)

        if heightmap is None or slopemap is None or shadowmap is None:
            raise ValueError(f"assemble_terrain_data: fehlende Calculator-Outputs für LOD {lod_level}")

        terrain_data = TerrainData()
        terrain_data.lod_level = lod_level
        terrain_data.actual_size = heightmap.shape[0]
        terrain_data.heightmap = heightmap
        terrain_data.slopemap = slopemap
        terrain_data.shadowmap = shadowmap
        terrain_data.calculated_sun_angles = self.shadow_calculator.get_sun_angles_for_lod(lod_level)[0]
        terrain_data.fallback_used = self._determine_fallback_used()
        terrain_data.update_parameters(parameters)
        terrain_data.validity_state = "valid"
        terrain_data.validity_flags = {"heightmap": True, "slopemap": True, "shadowmap": True}

        return terrain_data

    def _calc_noise(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'terrain.noise' (#1): rohes Noise-Grid [-1,1]"""
        parameters = self._current_parameters
        size = self._lod_level_to_size(lod_level, parameters.get('map_size', 512))

        frequency = parameters.get('frequency', 0.01)
        octaves = parameters.get('octaves', 4)
        persistence = parameters.get('persistence', 0.5)
        lacunarity = parameters.get('lacunarity', 2.0)

        # Frequency für LOD-Größe anpassen
        adjusted_frequency = frequency * (64 / size)  # Referenz: LOD 64

        # map_seed war hier vorher nie gelesen worden - der Noise-Generator
        # behielt den Seed, mit dem er beim allerersten Generate-Klick
        # konstruiert wurde, für die gesamte App-Laufzeit (siehe
        # SimplexNoiseGenerator.set_seed()). Änderungen am map_seed-Slider
        # hatten dadurch nie eine sichtbare Wirkung auf die Heightmap.
        map_seed = parameters.get('map_seed', self.map_seed)
        self.noise_generator.set_seed(map_seed)

        noise_grid = self.noise_generator.generate_noise_grid(
            size=size,
            frequency=adjusted_frequency,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity
        )
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"noise_grid": noise_grid})
        self.logger.debug("Noise generation completed")

    def _calc_redistribution(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'terrain.redistribution' (#2): Noise -> amplitudenskalierte
        Heightmap -> Power-Redistribution, ergibt die finale heightmap.
        """
        parameters = self._current_parameters
        noise_grid = self.data_lod_manager.get_calculator_output("terrain.noise", "noise_grid", lod_level)
        if noise_grid is None:
            raise ValueError(f"terrain.redistribution: noise_grid für LOD {lod_level} nicht verfügbar")

        amplitude = parameters.get('amplitude', 100)

        # Auf [0, amplitude] skalieren
        heightmap = (noise_grid + 1.0) * 0.5 * amplitude
        heightmap = heightmap.astype(np.float32)

        heightmap = self._apply_redistribution(
            heightmap, parameters.get('redistribute_power', 1.0), amplitude
        )
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"heightmap": heightmap})
        self.logger.debug("Heightmap generation + redistribution completed")

    def _calc_slope(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'terrain.slope' (#3)"""
        heightmap = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        if heightmap is None:
            raise ValueError(f"terrain.slope: heightmap für LOD {lod_level} nicht verfügbar")

        slopemap = self.slope_calculator.calculate_slopes(heightmap, self._current_parameters)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"slopemap": slopemap})
        self.logger.debug("Slope calculation completed")

    def _calc_shadow(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'terrain.shadow' (#4)"""
        heightmap = self.data_lod_manager.get_calculator_output("terrain.redistribution", "heightmap", lod_level)
        if heightmap is None:
            raise ValueError(f"terrain.shadow: heightmap für LOD {lod_level} nicht verfügbar")

        shadowmap = self.shadow_calculator.calculate_shadows(heightmap, lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"shadowmap": shadowmap})
        self.logger.debug("Shadow calculation completed")

    def _apply_redistribution(self, heightmap: np.ndarray, redistribute_power: float,
                               amplitude: float = None) -> np.ndarray:
        """
        Funktionsweise: Wendet Power-Redistribution auf Heightmap an
        Parameter: heightmap, redistribute_power, amplitude - theoretische Maximalhöhe
        Returns: numpy.ndarray - Redistributed Heightmap

        Normalisiert gegen die theoretische Maximalhöhe (amplitude), nicht gegen
        min/max der jeweils erzeugten Stichprobe: Contrast-Stretching gegen das
        Sample-Minimum würde bei jedem redistribute_power-Wert wieder den vollen
        Wertebereich ausnutzen und könnte die Landschaft nie absolut Richtung 0m
        drücken - genau das soll ein hoher redistribute_power aber bewirken
        (wenige hohe Gipfel, der Großteil der Fläche nahe der Talsohle).
        Ohne amplitude (z.B. für bereits redistributierte/fremde Daten) fällt die
        Funktion auf den alten Contrast-Stretch-Modus zurück.
        """
        if redistribute_power == 1.0:
            return heightmap

        if amplitude and amplitude > 0:
            normalized = np.clip(heightmap / amplitude, 0.0, 1.0)
            redistributed = np.power(normalized, redistribute_power)
            result = redistributed * amplitude
            return result.astype(np.float32)

        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            return heightmap

        # Fallback: Contrast-Stretch gegen die Stichprobe (altes Verhalten)
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

        # Range-Validation - Grenzen synchron zu gui/config/value_default.py TERRAIN
        validations = {
            'map_size': lambda x: 32 <= x <= 1024 and x % 32 == 0,  # Vielfaches von MAPSIZEMIN (UI-Step)
            'amplitude': lambda x: 30 <= x <= 6000,
            'octaves': lambda x: 1 <= x <= 12,
            'frequency': lambda x: 0.001 <= x <= 0.1,
            'persistence': lambda x: 0.1 <= x <= 1.0,
            'lacunarity': lambda x: 1.1 <= x <= 4.0,
            'redistribute_power': lambda x: 0.5 <= x <= 4.0
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
                'map_size': TERRAIN.MAPSIZE["default"],
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
        'map_size': (lambda x: 32 <= x <= 2048 and x % 32 == 0, "Must be a multiple of 32 between 32 and 2048"),
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