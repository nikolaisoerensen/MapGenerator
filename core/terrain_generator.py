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
import math
from typing import Dict, List, Tuple, Optional, Any



# BREITE DER GEZEICHNETEN FLUSSLINIE, in Pixeln (nur `river_water`).
#
# r = GRUND + JE_DEKADE * log10(1 + Wassermenge). Gemessen reicht die Menge
# von 0.38 bis 725 je Knoten, das sind knapp drei Dekaden:
#
#     Bach    (Menge   1)  ->  r = 0.6 px
#     Zufluss (Menge  20)  ->  r = 1.3 px
#     Strom   (Menge 700)  ->  r = 2.0 px
#
# Bewusst flach: der Hauptstrom soll erkennbar dicker sein, aber die Karte
# nicht zulaufen. Die Werte gelten fuer jede Aufloesung gleich - eine Linie
# von 2 px ist bei 1024 px feiner als bei 384, was richtig ist, weil dort
# mehr Laeufe nebeneinanderliegen.
FLUSS_BREITE_GRUND_PX = 0.30
FLUSS_BREITE_JE_DEKADE_PX = 0.60


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

        # WELTKARTE: vier weitere Ausgaben von terrain.redistribution.
        #
        # Sie fehlten hier bis zum 2026-08-06, und das war der Grund, weshalb
        # weder die Regionenfarben noch die Fluesse im Programm zu sehen waren.
        # Der Weg ist zweistufig: ein _calc_-Knoten legt seine Ausgaben im
        # CALCULATOR-Speicher ab, aber die Reiter lesen aus dem DOMAIN-Speicher
        # (get_terrain_data), und dazwischen steht assemble_terrain_data() ->
        # set_terrain_data_complete_lod(). Was dort nicht aufgezaehlt ist,
        # existiert fuer die Anzeige nicht.
        #
        # Der Fehler war unsichtbar: die Reiter fragen ab, bekommen None und
        # zeichnen einfach nur das Gelaende - kein Absturz, keine Warnung. Mein
        # erster Test hat ihn nicht gefunden, weil er den Rueckgabewert von
        # _weltkarte_heightmap() prüfte und danach direkt den Renderer, also
        # genau um diese Stufe herum.
        #
        # Optional: der alte Pfad (WELTKARTE_AKTIV = False) liefert sie nicht.
        self.river_mask: Optional[np.ndarray] = None
        self.river_order: Optional[np.ndarray] = None
        self.river_generation: Optional[np.ndarray] = None
        self.river_water: Optional[np.ndarray] = None
        # Flussbaum als Linienzuege statt Raster (Ticket #37,
        # core/fluss_export.py) - List[Dict] mit punkte/ordnung/breite_m,
        # fuer den Vektorexport. KEIN np.ndarray (require_array=False in
        # set_terrain_data_complete_lod(), siehe dort).
        self.river_lines: Optional[list] = None
        self.region_map: Optional[np.ndarray] = None
        # (3, H, W): Jahresmittel, Jahresspanne, Niederschlag - siehe
        # _weltkarte_heightmap. Weich ueber die Regionsgrenzen gemischt.
        self.klima_map: Optional[np.ndarray] = None
        # Seegliederung (2026-08-11, docs/spezifikation/12_WASSER.md Abschnitt 7): seegrad 0 auf
        # Land, 1..4+ auf See; ufer_region_a/b die bis zu zwei naechstgelegenen
        # Regionen je Seezelle (core.terrain_weltkarte.seegliederung()).
        self.seegrad: Optional[np.ndarray] = None
        self.ufer_region_a: Optional[np.ndarray] = None
        self.ufer_region_b: Optional[np.ndarray] = None
        # Seetyp-Regeln (2026-08-11, docs/OFFENE_PUNKTE.md 3.6): bool, True wo
        # die naechste Uferregion die Morobora ist.
        self.see_eis: Optional[np.ndarray] = None
        # Kuesten-Archetypen (2026-08-12, docs/OFFENE_PUNKTE.md 3.8): lokaler
        # Index (0..2, -1 = keiner) INNERHALB der Region, ueber
        # core.terrain_weltkarte.KUESTEN_ARCHETYPEN[region_name][index]
        # nachschlagbar; kuesten_staerke die Blendstaerke (0..1) je Pixel.
        self.kuesten_archetyp: Optional[np.ndarray] = None
        self.kuesten_staerke: Optional[np.ndarray] = None
        # Spielkarten-Zerlegung (2026-08-13, docs/OFFENE_PUNKTE.md 5.15):
        # int16 0..8, welche der neun Spielkarten dieses Pixel traegt. Konvexe
        # Vielecke mit etwa gleicher Landmasse - siehe core/spielkarten.py.
        # NICHT dasselbe wie region_map: die Regionen bestimmen Gelaende und
        # Kultur, die Spielkarten sind der Zuschnitt fuer Anzeige und Export.
        self.spielkarte: Optional[np.ndarray] = None

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

    # ENTFERNT 2026-07-30: detect_critical_changes() und
    # get_invalidated_generators().
    #
    # Beide waren toter Code - eine repo-weite Suche (core/, gui/, descriptor.py,
    # smoke_test_*, _old_*) fand ausser der jeweiligen Definition keinen
    # Aufrufer. Beide waren zugleich handgepflegte Listen der Art, die
    # SPEZIFIKATION §4.5 ausdruecklich verbietet, und beide waren schon falsch:
    #
    #   detect_critical_changes()   fuehrte fuenf Parameter als "kritisch" und
    #       liess redistribute_power, persistence und lacunarity aus, seit
    #       2026-07-30 zusaetzlich die sieben erosion_filter_*-Regler. Wer die
    #       Liste liest und ihr glaubt, schliesst daraus, dass diese Regler das
    #       Terrain nicht invalidieren - was falsch ist.
    #   get_invalidated_generators()  fuehrte die nachgelagerten Generatoren
    #       von Hand statt sie aus CALCULATOR_GRAPH abzuleiten.
    #
    # Gefaehrlich war nicht, dass sie nichts taten, sondern dass sie
    # verbindlich aussahen.
    #
    # Die Invalidierung findet in GenerationOrchestrator.
    # invalidate_downstream_dependencies() statt; die betroffenen Generatoren
    # werden dort aus dem Knotengraphen abgeleitet
    # (managers/calculator_graph.py), nicht aus einer Liste. Und weil
    # _calculate_parameter_hash() den VOLLSTAENDIGEN Parametersatz hasht,
    # wirkt jeder neue Regler, ohne irgendwo eingetragen zu werden - ein Hash
    # ueber alles kann nichts vergessen, eine Liste schon.
    #
    # Nicht mit aufgeraeumt, weil ausserhalb dieser Aufgabe: auch
    # validate_against_parameters() und get_validity_summary() weiter unten
    # haben derzeit keinen Aufrufer. Sie stehen aber als Teil des
    # "Validity-Methods"-Satzes im Klassen-Docstring und in descriptor.py und
    # gehoeren zusammen mit den gleichnamigen Methoden von Geology/Weather
    # betrachtet, nicht einzeln. is_valid() ist in Gebrauch
    # (test_terrain_generator() und mehrere smoke_test_*).

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
        # Der Versatz war bis 2026-08-04 ein GPU-Ausschlusskriterium: der Shader
        # kannte ihn nicht, also fiel JEDER verschobene Ausschnitt still auf die
        # CPU zurueck. Kein Fehler im Log, nur ein Programm, das ein Vielfaches
        # laenger rechnet - dieselbe Falle wie beim SHADERS_ROOT-Umzug. Jetzt
        # hat noiseGeneration.comp u_offset_x/u_offset_y, und die Zoom-Pyramide
        # (Makro/Meso/Mikro auf dieselbe Weltstelle) laeuft auf der GPU.
        if self._gpu_available():
            try:
                result = self.shader_manager.process_noise_generation(
                    size=size, octaves=octaves, frequency=frequency,
                    persistence=persistence, lacunarity=lacunarity, seed=self.seed,
                    offset_x=offset_x, offset_y=offset_y
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


# WIE FEIN DER SCHLAGSCHATTEN GERECHNET WIRD.
#
# Bis 2026-08-07 stand hier die feste Zahl 64 - unabhaengig von der
# Kartengroesse. Der Schattenwurf wurde damit bei jeder Aufloesung gleich grob,
# waehrend das Gelaende immer feiner wurde; GPU- und CPU-Pfad liefen dadurch
# mit steigender Kartengroesse immer weiter auseinander (Korrelation 0.815 bei
# 128 px, nur noch 0.507 bei 512).
#
# Jetzt ein VIERTEL der Kartenkante, mindestens 64. Der Schatten waechst also
# mit der Karte mit. Vorschlag des Nutzers 2026-08-07.
SCHATTEN_TEILER = 4
SCHATTEN_MINDESTGITTER = 64

# Soll der Schlagschatten ueber die GPU laufen?
#
# JA. Bei gleichem Gitter ist der Shader dem vektorisierten CPU-Weg deutlich
# ueberlegen - gemessen 2026-08-07, reiner Raycast ueber 7 Sonnenstaende:
#
#   Gitter   CPU       GPU      Faktor
#   128      0.13 s    0.02 s     8x
#   256      1.14 s    0.02 s    51x
#   512     21.17 s    0.07 s   304x
#
# Ein frueherer Vergleich am selben Tag hatte den GPU-Weg als nutzlos
# ausgewiesen. Der war UNFAIR: die GPU rechnete auf 64 px, die CPU auf voller
# Aufloesung - zwei verschiedene Arbeitsmengen. Mit gleichem Gitter stimmen
# beide zu 99 % der Pixel ueberein; die Reste sind einzelne Randpixel des
# Schlagschattens (GLSL-Textursampling gegen numpy-Bilinear).
GPU_SCHATTEN = True


# Bezugsrelief fuer die Gewichtung des Erosionsfilters (Meter).
#
# Eine Region mit diesem Relief bekommt Gewicht 1.0, doppelt so viel Relief
# das Doppelte - geklemmt auf MIN..MAX, damit weder die Samarcia (183 m) leer
# ausgeht noch das Nevadin (1000 m) zerfranst.
#
# FEST und nicht aus dem Kartenmittel abgeleitet: eine Normierung auf das
# jeweilige Bild waere derselbe Fehler, den die Hoehenskala schon einmal hatte.
EROSION_BEZUGSRELIEF_M = 400.0
EROSION_GEWICHT_MIN = 0.35
EROSION_GEWICHT_MAX = 2.50


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

        # DIE PIXELGROESSE IN METERN. Ohne sie ist "wie steil ist der Hang"
        # nicht beantwortbar - siehe den Block bei _einfallswinkel(). 1.0 ist
        # KEINE brauchbare Vorgabe, sondern der Zustand vor dem 2026-08-07;
        # wer sie nicht setzt, bekommt eine Warnung.
        self._meters_per_pixel = 0.0

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
        if not self._meters_per_pixel:
            self.logger.warning(
                "ShadowCalculator: meters_per_pixel nicht gesetzt - Hangneigung "
                "und Schattenlaenge werden falsch. set_meters_per_pixel() "
                "aufrufen (siehe Kommentar bei _einfallswinkel).")

        # EIN ABLAUF FUER BEIDE PFADE (2026-08-07).
        #
        # Vorher gab es zwei getrennte Wege, die verschiedene GROESSEN
        # lieferten: der Shader nur die Verschattung, die CPU zusaetzlich den
        # Einfallswinkel. Jetzt ist die Aufteilung nach der Art der Rechnung
        # gemacht statt nach dem Geraet:
        #
        #   SCHLAGSCHATTEN   Strahlverfolgung, teuer, GPU 300-mal schneller
        #                    -> auf SCHATTEN_TEILER-tel der Kartenkante
        #   EINFALLSWINKEL   rein oertlich, billig, kein Gewinn durch die GPU
        #                    -> in voller Aufloesung, EINE Fassung fuer beide
        #
        # Damit ist die Paritaet Bauweise: der Einfallswinkel kann gar nicht
        # mehr auseinanderlaufen, und der Schlagschatten benutzt auf beiden
        # Wegen dieselben Grenzen (raycast_grenzen).
        try:
            return self._sonnenexposition(heightmap, lod_level,
                                          sun_angles_override)
        except Exception as e:
            self.logger.warning(f"Shadow calculation failed: {e}")

        # Simple-Fallback (Minimal)
        return self._calculate_simple_shadows(heightmap, lod_level)

    def _sonnenexposition(self, heightmap: np.ndarray, lod_level: int,
                          sun_angles_override=None) -> np.ndarray:
        """
        Schlagschatten mal Einfallswinkel, ueber die Sonnenstaende gewichtet.

        Der Schlagschatten entsteht auf einem groeberen Gitter (siehe
        SCHATTEN_TEILER) und wird hochskaliert - er hat ohnehin weiche Raender.
        Der Einfallswinkel wird in voller Aufloesung gerechnet; er traegt die
        feine Struktur, aus der spaeter die unregelmaessige Baumgrenze folgt.
        """
        size = heightmap.shape[0]
        gitter = max(SCHATTEN_MINDESTGITTER, size // SCHATTEN_TEILER)
        gitter = min(gitter, size)
        mpp_voll = float(self._meters_per_pixel) or 1.0
        mpp_gitter = mpp_voll * size / float(gitter)

        grob = self._resize_2d(heightmap, gitter) if gitter != size else heightmap
        grob = np.ascontiguousarray(grob, dtype=np.float32)

        winkel, gewichte = self.get_sun_angles_for_lod(lod_level, sun_angles_override)
        summe = float(sum(gewichte)) or 1.0

        schatten = np.zeros((gitter, gitter), dtype=np.float64)
        einfall = np.zeros((size, size), dtype=np.float64)
        gpu = GPU_SCHATTEN and self._gpu_available()

        for (elevation, azimuth), gewicht in zip(winkel, gewichte):
            sx, sy, sz = self._sonnenrichtung(elevation, azimuth)
            reichweite, schritt = self.raycast_grenzen(grob, mpp_gitter, sz)

            teil = None
            if gpu:
                try:
                    teil = self.shader_manager.process_shadow_raycast(
                        grob, elevation, azimuth, gitter,
                        max_distance=float(reichweite), step_size=float(schritt),
                        height_scale=1.0 / max(mpp_gitter, 1e-6))
                except Exception as fehler:
                    self.logger.warning("GPU-Schatten fehlgeschlagen: %s", fehler)
                    teil = None
                    gpu = False
            if teil is None:
                teil = self._verschattung_cpu(grob, mpp_gitter, sx, sy, sz)

            schatten += np.asarray(teil, dtype=np.float64) * gewicht
            einfall += self._einfallswinkel(heightmap, mpp_voll, sx, sy, sz) * gewicht

        schatten /= summe
        einfall /= summe
        if gitter != size:
            schatten = self._resize_2d(schatten.astype(np.float32), size)
        return (schatten * einfall).astype(np.float32)

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

        # Dieselben Konstanten wie der fruehere CPU-Hauptpfad (der inzwischen
        # geloeschte _is_in_shadow_cpu, siehe docs/OFFENE_PUNKTE.md 10.5:
        # step_size=0.5, max_distance=max(width,height)*2 - bei der hier immer
        # 64x64 großen shadow_heightmap also 128.0) statt der vorherigen
        # ShaderManager-eigenen Defaults (max_distance=100.0, step_size=1.0) -
        # sonst unterschieden sich GPU- und CPU-Schatten sichtbar je nachdem,
        # welcher Pfad gerade aktiv war. Siehe [[project-terrain-review]] 4b.
        # DER SHADER HAT DENSELBEN EINHEITENFEHLER (gefunden 2026-08-07).
        #
        # shadowRaycast.comp rechnet `currentHeight += sunDir.z * u_step_size`,
        # also Meter plus Pixel - genau der Fehler, der auch im CPU-Pfad steckte.
        # Gemessen: auf ebenem Wasser lieferte die GPU 0.411 statt der
        # analytisch erwarteten 0.719, weil der Strahl praktisch waagerecht lief
        # und jede Bodenwelle Schatten warf.
        #
        # ER LAESST SICH OHNE SHADER-AENDERUNG BEHEBEN: `u_height_scale`
        # multipliziert beide Hoehen im Shader. Setzt man ihn auf 1/mpp, stehen
        # die Hoehen in PIXELEINHEITEN, und der Vergleich mit `sunDir.z * step`
        # (ebenfalls Pixel) stimmt wieder. Eine Zeile statt eines Shader-Umbaus,
        # und der Shader bleibt fuer andere Aufrufer unveraendert.
        mpp_schatten = (float(getattr(self, "_meters_per_pixel", 0.0)) or 1.0) \
            * original_size / float(shadow_resolution)
        hoehenskala = 1.0 / max(mpp_schatten, 1e-6)

        for (elevation, azimuth), weight in zip(sun_angles, sun_weights):
            shadow_map = self.shader_manager.process_shadow_raycast(
                shadow_heightmap, elevation, azimuth, shadow_resolution,
                max_distance=float(shadow_resolution * 2), step_size=0.5,
                height_scale=hoehenskala
            )
            if shadow_map is None:
                raise RuntimeError("process_shadow_raycast returned no data")
            shadows += shadow_map * weight

        shadows /= total_weight

        if original_size != shadow_resolution:
            shadows = self._resize_2d(shadows, original_size)

        # DER EINFALLSWINKEL FEHLT DEM SHADER (2026-08-07).
        #
        # shadowRaycast.comp gibt nur 0.0 oder 1.0 zurueck - reine
        # Verschattung. Der CPU-Pfad multiplizierte zusaetzlich mit dem
        # Skalarprodukt aus Flaechennormale und Sonnenrichtung, und damit
        # lieferten die beiden Pfade GRUNDVERSCHIEDENE Groessen: gemessen
        # Landmittel 0.296 gegen 0.111, groesste Abweichung 0.86.
        #
        # Der Einfallswinkel wird deshalb HIER ergaenzt, in derselben Funktion,
        # die auch der CPU-Pfad benutzt. Zwei Gruende dafuer, statt den Shader
        # zu erweitern:
        #
        #   * Er ist eine rein oertliche Rechnung ohne Strahlverfolgung - auf
        #     der CPU in Millisekunden erledigt, kein Gewinn durch die GPU.
        #   * EINE Fassung fuer beide Pfade macht die Paritaet zur Bauweise
        #     statt zur Zusicherung, die man nachtraeglich pruefen muss.
        #
        # Er wird in VOLLER Aufloesung gerechnet, waehrend die Verschattung vom
        # Shader auf 64 px entsteht - der Schlagschatten ist also grob, die
        # Hangbeleuchtung fein. Das ist vertretbar, weil der Schlagschatten
        # ohnehin weiche Raender hat; die Hangbeleuchtung dagegen traegt die
        # feine Struktur, die spaeter die Baumgrenze unregelmaessig macht.
        mpp = float(getattr(self, "_meters_per_pixel", 0.0)) or 1.0
        einfall = np.zeros_like(shadows, dtype=np.float64)
        for (elevation, azimuth), weight in zip(sun_angles, sun_weights):
            sx, sy, sz = self._sonnenrichtung(elevation, azimuth)
            einfall += self._einfallswinkel(heightmap, mpp, sx, sy, sz) * weight
        einfall /= total_weight

        return (shadows * einfall).astype(np.float32)

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

        # Sonnenrichtung berechnen. Azimuth-Konvention (siehe
        # calculate_solar_position()): 0°=Norden, 90°=Osten, 180°=Süden,
        # 270°=Westen, im Uhrzeigersinn. Array-Konvention dieser Codebase
        # (verifiziert über _semi_lagrangian_advect()s source_y=y_idx-v*dt
        # PLUS den unveränderten prevailing_wind_direction-Code, der bei
        # Default 225° konsistent "Wind von Nordosten" ergibt, exakt wie vom
        # Nutzer beobachtet, OHNE jede Änderung - beweist Zeile height-1 =
        # Norden, Zeile 0 = Süden, NICHT umgekehrt wie ein Kommentar in
        # _old_01/core_old/weather_generator.py behauptet, der zu einem
        # früheren, inzwischen archivierten Implementierungsstand gehört und
        # sich als nicht mehr gültig herausstellte): sun_y MUSS daher bei
        # Süd-Azimut (180°) NEGATIV sein (Richtung abnehmende Zeile), exakt
        # was die Formel ohne Vorzeichen-Anpassung bereits liefert - ein
        # zwischenzeitlicher Fix-Versuch (Y negiert) beruhte auf der falschen,
        # archivierten Konvention und wurde nach dieser Verifikation wieder
        # zurückgenommen.
        sun_x, sun_y, sun_z = self._sonnenrichtung(sun_elevation, sun_azimuth)

        # SEIT 2026-08-07 vektorisiert und in Metern gerechnet, siehe den
        # Block bei _einfallswinkel(). Die alte Pixel-fuer-Pixel-Doppelschleife
        # (_is_in_shadow_cpu / _calculate_slope_shading_cpu) rief niemand mehr
        # auf und ist am 2026-08-16 geloescht (docs/OFFENE_PUNKTE.md 10.5) -
        # die Fehlergeschichte dazu steht weiter unten (SONNENEXPOSITION).
        mpp = float(getattr(self, "_meters_per_pixel", 0.0)) or 1.0
        shadow_map = (self._verschattung_cpu(heightmap, mpp, sun_x, sun_y, sun_z)
                      * self._einfallswinkel(heightmap, mpp, sun_x, sun_y, sun_z))
        return shadow_map.astype(np.float32)

    # =========================================================================
    # SONNENEXPOSITION - eine Stelle fuer beide Pfade (2026-08-07)
    # =========================================================================
    #
    # WAS HIER FALSCH WAR, und es war dreimal derselbe Fehler:
    #
    #  1. `_calculate_slope_shading_cpu` bildete den Gradienten als
    #     h[x+1] - h[x-1], also METER JE PIXEL. Bei 83 m/px erschien jeder Hang
    #     um Faktor 83 zu steil: gemessen 86.5 Grad Median statt 11.1, und
    #     98.8 % der Landflaeche galten als steiler als 60 Grad statt 0.7 %.
    #     Die Flaechennormale kippte damit fast in die Waagerechte und das
    #     Skalarprodukt mit der Sonne brach zusammen.
    #
    #  2. `_is_in_shadow_cpu` rechnete `ray_z = hoehe + sun_z * distance` -
    #     METER plus PIXEL. Der Sonnenstrahl stieg also um sun_z Meter je
    #     Pixel statt um sun_z * mpp. Die Sonne stand dadurch 83-mal zu tief,
    #     und fast die ganze Karte lag im Schatten.
    #
    #  3. Der GPU-Shader liefert NUR die Verschattung (0 oder 1) und gar
    #     keinen Einfallswinkel. Gemessen: Landmittel 0.296 auf der GPU gegen
    #     0.111 auf der CPU, groesste Abweichung 0.86 auf einer 0..1-Skala.
    #     Zwei verschiedene Groessen, und das Programm nimmt die GPU.
    #
    # Das Ergebnis war, dass Land DUNKLER war als Wasser (0.111 gegen 0.407) -
    # bei derselben Sonne physikalisch unmoeglich.
    #
    # ES BRAUCHT DIE PIXELGROESSE. Ohne sie ist die Frage "wie steil ist der
    # Hang" nicht beantwortbar; genau daran ist schon der Hangfehler vom
    # 2026-07-09 gescheitert, der in drei Dateien steckte.

    def set_meters_per_pixel(self, meters_per_pixel: float):
        """Wie gross ein Pixel in der Wirklichkeit ist. Siehe _einfallswinkel."""
        self._meters_per_pixel = float(meters_per_pixel)

    @staticmethod
    def _sonnenrichtung(sun_elevation: float, sun_azimuth: float):
        """Die Sonnenrichtung in der Konvention dieser Codebase.

        Azimut 0 = Norden, 90 = Osten, im Uhrzeigersinn. Zeile height-1 ist
        Norden (siehe die ausfuehrliche Herleitung in _raycast_shadow_cpu).
        """
        e = np.radians(sun_elevation)
        a = np.radians(sun_azimuth)
        return (np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e))

    @staticmethod
    def raycast_grenzen(heightmap: np.ndarray, meters_per_pixel: float,
                        sun_z: float):
        """
        Reichweite (Pixel) und Schrittweite fuer den Schattenstrahl.

        EINE Stelle fuer beide Pfade. Vorher hatte der CPU-Weg Schrittweite 1.0
        und eine geometrische Reichweite, der Shader 0.5 und die feste Grenze
        2 * Kartenbreite - bei gleichem Gitter wichen die Ergebnisse dadurch um
        bis zu 0.6 voneinander ab (gemessen 2026-08-07), obwohl beide dieselbe
        Frage beantworten sollten.

        DIE REICHWEITE FOLGT AUS DER GEOMETRIE: laenger als
        Hoehenspanne / tan(Sonnenhoehe) kann ein Schlagschatten nicht sein. Die
        feste Grenze von 2 * Kartenbreite war willkuerlich und der Grund, warum
        der Schattenwurf so lange brauchte.
        """
        mpp = max(float(meters_per_pixel), 1e-6)
        size = heightmap.shape[0]
        spanne = float(np.ptp(heightmap)) or 1.0
        reichweite = min(spanne / (max(sun_z, 1e-4) * mpp) + 2.0, 2.0 * size)
        return reichweite, 0.5

    @staticmethod
    def _einfallswinkel(heightmap: np.ndarray, meters_per_pixel: float,
                        sun_x: float, sun_y: float, sun_z: float) -> np.ndarray:
        """
        Der Kosinus des Einfallswinkels je Pixel, 0..1.

        Die Flaechennormale aus dem Gradienten IN METERN JE METER - das ist der
        Unterschied zur alten Fassung. Vektorisiert ueber die ganze Karte statt
        Pixel fuer Pixel: der alte Weg brauchte 4.9 Millionen Aufrufe einer
        Python-Funktion und war der groesste Einzelposten der Wetterrechnung.
        """
        mpp = max(float(meters_per_pixel), 1e-6)
        gy, gx = np.gradient(heightmap.astype(np.float64), mpp)
        # Normale (-dz/dx, -dz/dy, 1), normiert
        laenge = np.sqrt(gx * gx + gy * gy + 1.0)
        skalar = (-gx * sun_x - gy * sun_y + sun_z) / laenge
        return np.maximum(skalar, 0.0)

    @staticmethod
    def _verschattung_cpu(heightmap: np.ndarray, meters_per_pixel: float,
                          sun_x: float, sun_y: float, sun_z: float) -> np.ndarray:
        """
        Wer liegt im Schlagschatten? 1 = besonnt, 0 = verschattet.

        VEKTORISIERT und IN METERN. Der Strahl steigt je Pixelschritt um
        sun_z * mpp Meter - vorher um sun_z Meter, was die Sonne um den Faktor
        mpp zu tief stellte.

        DIE REICHWEITE FOLGT AUS DER GEOMETRIE statt aus einer festen Zahl:
        laenger als Hoehenspanne / tan(Sonnenhoehe) kann ein Schatten nicht
        sein. Die alte Grenze von 2 * Kartenbreite war willkuerlich und der
        Grund, warum der Schattenwurf so lange brauchte.
        """
        mpp = max(float(meters_per_pixel), 1e-6)
        if sun_z <= 1e-4:
            return np.zeros_like(heightmap, dtype=np.float32)

        size = heightmap.shape[0]
        reichweite_px, schritt = ShadowCalculator.raycast_grenzen(heightmap, mpp,
                                                                 sun_z)

        H = heightmap.astype(np.float64)
        gy, gx = np.mgrid[0:size, 0:size]
        im_schatten = np.zeros((size, size), dtype=bool)

        d = schritt
        while d < reichweite_px:
            sx = gx + sun_x * d
            sy = gy + sun_y * d
            drin = (sx >= 0) & (sx < size - 1) & (sy >= 0) & (sy < size - 1)
            if not drin.any():
                break
            xi = np.clip(sx, 0, size - 1.001)
            yi = np.clip(sy, 0, size - 1.001)
            x0 = xi.astype(np.int32)
            y0 = yi.astype(np.int32)
            fx = xi - x0
            fy = yi - y0
            gelaende = (H[y0, x0] * (1 - fx) * (1 - fy)
                        + H[y0, x0 + 1] * fx * (1 - fy)
                        + H[y0 + 1, x0] * (1 - fx) * fy
                        + H[y0 + 1, x0 + 1] * fx * fy)
            strahl = H + sun_z * d * mpp
            im_schatten |= drin & (strahl <= gelaende)
            d += schritt

        return (~im_schatten).astype(np.float32)

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
        Parameter: parameters - ungenutzt, nur für Signatur-Kompatibilität mit
        anderen calculate_*()-Methoden dieses Musters
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
        # NICHT 1.0 - die Karte deckt immer map_distance_km x map_distance_km ab
        # (siehe gui/widgets/map_display_3d.py), unabhängig von der Pixelauflösung.
        # Ein fester spacing=1.0 hieß: 1m Höhenunterschied zwischen Nachbarpixeln wird
        # wie 1m realer Horizontal-Abstand behandelt, obwohl ein Pixel bei typischen
        # Kartengrößen tatsächlich ~50-300m Horizontal-Abstand abdeckt - das ergab
        # Gradienten um ~10-15 (entspricht ~85-89°) auf praktisch der gesamten Karte.
        # map_distance_km kommt aus demselben parameters-Dict wie alle anderen
        # Terrain-Slider (live einstellbar seit [[project-terrain-review]] 4f),
        # Fallback auf TERRAIN.WORLD_SIZE_KM nur für Standalone-/Legacy-Aufrufer,
        # die parameters ohne diesen Key übergeben.
        from gui.config.value_default import TERRAIN
        map_distance_km = parameters.get('map_distance_km', TERRAIN.WORLD_SIZE_KM)
        world_size_m = map_distance_km * 1000.0
        spacing = world_size_m / heightmap.shape[0]

        # Berechne Gradienten in beide Richtungen
        grad_y, grad_x = np.gradient(heightmap, spacing, edge_order=2)

        # Als (H,W,2) Array zusammenfassen
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)
        slopemap[:, :, 0] = grad_x  # dz/dx
        slopemap[:, :, 1] = grad_y  # dz/dy

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

        # LADEBALKEN - bis 2026-08-23 hatte AUSGERECHNET dieser Generator
        # keinen. Der Orchestrator setzt das Attribut per hasattr()-Test
        # (generation_orchestrator._run_calculator), fand es hier nie, und
        # damit stand der Balken waehrend terrain.redistribution still -
        # also waehrend der laengsten Einzelphase der ganzen Pipeline (61 s
        # von 203 s gemessen). Es sah aus, als haenge das Programm.
        self.progress_callback = None

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

        # Live-Wert für DataLODManager.get_map_distance_km() aktuell halten -
        # Biome/Water/Weather/map_display_3d.py lesen darüber statt eines
        # statischen TERRAIN.WORLD_SIZE_KM-Imports (siehe
        # [[project-terrain-review]] 4f). self.data_lod_manager kann in
        # Standalone-/Test-Nutzung noch None sein (siehe
        # _ensure_data_lod_manager()) - dann bleibt der Default bestehen.
        if self.data_lod_manager is not None and 'map_distance_km' in parameters:
            self.data_lod_manager.set_map_distance_km(parameters['map_distance_km'])

        # Live-Wert für DataLODManager.get_map_seed() aktuell halten - Geology
        # liest darüber statt eines Konstruktor-Arguments (siehe
        # GeologySystemGenerator.set_active_parameters(), Nutzer-Bug-Report:
        # Intrusion/Fault/Tilt blieben bei jedem Map Seed identisch, weil
        # GeologySystemGenerator lazy vom GenerationOrchestrator OHNE
        # map_seed-Konstruktor-Argument instanziiert und für die gesamte
        # App-Session wiederverwendet wird).
        if self.data_lod_manager is not None and 'map_seed' in parameters:
            self.data_lod_manager.set_map_seed(parameters['map_seed'])

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, calculate_heightmap() ohne
        injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from managers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    def calculate_heightmap(self, parameters: Dict[str, Any], lod_level: int) -> TerrainData:
        """
        Funktionsweise: Standalone-Convenience-Entry-Point (Legacy-Kompatibilität + Tests)
        Aufgabe: Führt alle 4 Terrain-Calculator-Knoten synchron für EIN LOD aus und
            liefert das fertige TerrainData-Objekt. Die echte GUI-Pipeline
            (GenerationOrchestrator) ruft dieselben _calc_*-Methoden ab jetzt einzeln
            über den globalen CalculatorDispatcher auf (siehe
            managers/calculator_graph.py, Tracker #16 LOD-Lockstep-Umbau) -
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
        # Die vier Weltkarten-Ausgaben mitnehmen. OHNE Pflichtpruefung: im alten
        # Pfad (WELTKARTE_AKTIV = False) gibt es sie nicht, und ein fehlendes
        # Flussnetz darf den Zusammenbau nicht scheitern lassen.
        for schluessel in ("river_mask", "river_order", "river_generation", "river_water",
                           "river_lines",
                           "hinterland_height", "voronoi_map",
                           "region_map", "klima_map", "seegrad",
                           "ufer_region_a", "ufer_region_b", "see_eis",
                           "kuesten_archetyp", "kuesten_staerke", "spielkarte"):
            setattr(terrain_data, schluessel,
                    self.data_lod_manager.get_calculator_output(
                        "terrain.redistribution", schluessel, lod_level))

        terrain_data.calculated_sun_angles = self.shadow_calculator.get_sun_angles_for_lod(lod_level)[0]
        terrain_data.fallback_used = self._determine_fallback_used()
        terrain_data.update_parameters(parameters)
        terrain_data.validity_state = "valid"
        terrain_data.validity_flags = {"heightmap": True, "slopemap": True, "shadowmap": True}

        return terrain_data

    @staticmethod
    def _max_safe_octaves(adjusted_frequency: float, lacunarity: float, requested_octaves: int) -> int:
        """
        Funktionsweise: Größte Oktavenzahl n, für die
        adjusted_frequency * lacunarity**(n-1) <= 0.5 (Nyquist-Grenze, 0.5
        Zyklen/Pixel) gilt - geschlossene Form statt Schleife, da lacunarity
        pro Oktave multiplikativ wächst.
        Parameter: adjusted_frequency (float) - bereits größen-normalisierte
        Basisfrequenz (siehe _calc_noise), lacunarity (float), requested_octaves
        (int) - vom Slider angefragte Oktavenzahl, obere Grenze des Ergebnisses.
        Returns: int - mindestens 1, höchstens requested_octaves.
        """
        if requested_octaves <= 1 or adjusted_frequency <= 0 or lacunarity <= 1.0:
            return max(1, requested_octaves)
        if adjusted_frequency > 0.5:
            return 1
        max_n = 1 + math.floor(math.log(0.5 / adjusted_frequency) / math.log(lacunarity))
        return max(1, min(requested_octaves, max_n))

    def _calc_noise(self, calculator_id: str, lod_level: int) -> None:
        """Calculator-Node 'terrain.noise' (#1): rohes Noise-Grid [-1,1]"""
        parameters = self._current_parameters
        size = self._lod_level_to_size(lod_level, parameters.get('map_size', 512))

        frequency = parameters.get('frequency', 0.01)
        octaves = parameters.get('octaves', 4)
        persistence = parameters.get('persistence', 0.5)
        lacunarity = parameters.get('lacunarity', 2.0)

        # Frequenz in Zyklen pro Pixel. Zwei Wege, und der erste ist der
        # richtige:
        #
        # terrain_feature_size_m gibt die Groesse der Grundformen in METERN an.
        # Daraus folgt die Zyklenzahl ueber die Karte als Kartenbreite geteilt
        # durch Formgroesse, und daraus die Zyklen je Pixel. Damit haengt die
        # Landschaft an der Wirklichkeit und nicht am Bildausschnitt: ein
        # groesserer Ausschnitt zeigt MEHR Formen, nicht groessere.
        #
        # Der alte Weg `frequency * (64 / size)` haengt nur an der Pixelzahl.
        # Gemessen (smoke_test_terrain_scale_coupling.py): die Karte zeigte bei
        # jedem map_distance_km dieselben 4.74 Zyklen, die Formen waren also bei
        # 5 km 1064 m und bei 50 km 10638 m gross. Er bleibt als
        # Ueberschreibung erhalten, damit Labore und Altbestand weiterlaufen.
        feature_size_m = parameters.get('feature_size_m')
        if feature_size_m and feature_size_m > 0:
            karte_m = float(self._ensure_data_lod_manager().get_map_distance_km()) * 1000.0
            zyklen_ueber_karte = karte_m / float(feature_size_m)
            adjusted_frequency = zyklen_ueber_karte / float(size)
        else:
            adjusted_frequency = frequency * (64 / size)  # Referenz: LOD 64

        # Oktaven, die die Nyquist-Grenze (0.5 Zyklen/Pixel) überschreiten, fügen
        # nur noch Aliasing statt echtem Detail hinzu - bei Default-Werten
        # (frequency=0.037, lacunarity=2.3) liegt das schon ab Oktave 5 vor (siehe
        # OCTAVES-Beschreibung in value_default.py). Der UI-Slider erlaubt aber
        # weiterhin bis zu 8 Oktaven und Lacunarity bis 4.0 - hier statt eines
        # kaputten Ergebnisses still auf die tatsächlich sinnvolle Oktavenzahl
        # clampen (wirkt einheitlich auf GPU-/CPU-/Simple-Fallback, da alle drei
        # denselben effective_octaves-Wert von hier bekommen, statt den Fix
        # separat für jeden der drei Pfade nachzubauen).
        effective_octaves = self._max_safe_octaves(adjusted_frequency, lacunarity, octaves)
        if effective_octaves < octaves:
            self.logger.debug(
                f"Octaves clamped from {octaves} to {effective_octaves} "
                f"(frequency={adjusted_frequency:.4f}, lacunarity={lacunarity}: "
                f"higher octaves would exceed the 0.5 cycles/pixel Nyquist limit)")
        octaves = effective_octaves

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

        # WELTKARTE - die Weiche (docs/archiv/2026-08-04_INTEGRATIONSPLAN.md, Stufe P1).
        #
        # Ist sie aktiv, kommt die Heightmap aus core/terrain_weltkarte.py:
        # neun Regionen als Parameterfeld auf einer Plaetzchenform, mit Meer.
        # Der alte Pfad (Noise -> Potenz -> Erosionsfilter -> Flussnetz) bleibt
        # vollstaendig erhalten und laeuft, sobald der Schalter aus ist - er
        # muss lauffaehig bleiben, bis die Weltkarte abgenommen ist.
        weltkarte = self._weltkarte_heightmap(lod_level)
        if weltkarte is not None:
            self.data_lod_manager.set_calculator_output(
                calculator_id, lod_level, weltkarte)
            self.logger.debug("Weltkarte statt Noise-Gelaende erzeugt")
            return

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

        outputs = {"heightmap": heightmap}
        gefiltert = self._apply_erosion_filter(heightmap, amplitude)
        if gefiltert is not None:
            outputs["heightmap"] = gefiltert["heightmap"]
            outputs["ridge_map"] = gefiltert["ridge_map"]

        # Flussnetz NACH dem Erosionsfilter: dessen Ergebnis ist die Flaeche P,
        # in die eingeschnitten wird (SPEZIFIKATION §12).
        netz = self._apply_river_network(outputs["heightmap"], amplitude)
        if netz is not None:
            # Spanne erneut setzen: der Einschnitt drueckt die Talsohle unter
            # die Talsohlenhoehe (gemessen -1260 m bei den Alpen, §12). Mit
            # Potenz 1.0 ist das eine reine lineare Abbildung, die Form bleibt.
            outputs["heightmap"] = self._apply_redistribution(
                netz["heightmap"], 1.0, amplitude)
            outputs["river_mask"] = netz["river_mask"]
            outputs["river_order"] = netz["river_order"]

        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, outputs)
        self.logger.debug("Heightmap generation + redistribution completed")

    def _update_progress(self, phase: str, progress: int, message: str):
        """
        Ladebalken, gleiche Signatur wie in allen anderen core/*_generator.py.
        Fehler hier duerfen die Berechnung nicht abbrechen - eine Anzeige ist
        kein Rechenergebnis.
        """
        cb = getattr(self, "progress_callback", None)
        if cb is None:
            return
        try:
            cb(phase, int(progress), message)
        except Exception:                                        # noqa: BLE001
            self.logger.debug("progress_callback fehlgeschlagen", exc_info=True)

    def _weltkarte_heightmap(self, lod_level: int):
        """
        Die Heightmap der Regionenwelt, oder None wenn der Schalter aus ist.

        WAS HIER ANDERS IST ALS IM ALTEN PFAD, und warum es so sein muss:

        * KEINE FENSTERNORMIERUNG. Der alte Pfad streckt jede Karte auf
          0..AMPLITUDE. Hier steht die Hoehe in echten Metern, und sie darf
          NEGATIV sein - unter 0 ist Meer. Eine Normierung wuerde die
          Kuestenlinie verschieben und alle neun Regionseichungen entwerten.
        * DIE KARTENGROESSE STEHT FEST. Die Welt ist WELT_KM breit; ein anderer
          Wert von map_distance_km wuerde Regionsgroessen und Talabstaende
          gegeneinander verschieben. Der Wert wird deshalb im Manager gesetzt,
          damit alle nachgelagerten Knoten dieselbe Skala benutzen.
        * AMPLITUDE, FEATURE_SIZE_M und REDISTRIBUTE_POWER wirken nicht. Die
          Regionen bringen ihre eigenen mit. Die Regler bleiben stehen, damit
          der alte Pfad weiter bedienbar ist.
        """
        import gui.config.value_default as vd
        if not getattr(vd, "WELTKARTE_AKTIV", False):
            return None

        from core.terrain_weltkarte import weltfeld, WELT_KM

        parameters = self._current_parameters
        size = self._lod_level_to_size(lod_level, parameters.get("map_size", 512))
        seed = int(parameters.get("map_seed", self.map_seed))

        manager = self._ensure_data_lod_manager()
        if abs(float(manager.get_map_distance_km()) - WELT_KM) > 1e-6:
            manager.set_map_distance_km(WELT_KM)
            self.logger.info(
                "Weltkarte aktiv: map_distance_km auf %.1f km gesetzt", WELT_KM)

        # TEILSCHRITT-MESSUNG (2026-08-23). Ohne sie sagte das Pipeline-Log
        # nur "terrain.redistribution | 61.117s" - bei einem Aufrufbaum aus
        # rund einem Dutzend Funktionen ist das keine Diagnose, sondern eine
        # Zahl. Die drei Stufen hier (Weltfeld, Erosionsfilter, Fluesse) und
        # die elf Teilschritte in weltfeld() selbst stehen jetzt einzeln im
        # Log und treiben den Ladebalken.
        from core.terrain_weltkarte import WELTFELD_PLAN
        from managers.teilschritte import Teilschritte, schritt as _s
        schritte = Teilschritte(
            "terrain.redistribution", fortschritt=self._update_progress,
            von=5, bis=95,
            plan=list(WELTFELD_PLAN) + [("erosionsfilter", 8.0),
                                        ("weltfluesse", 25.0),
                                        ("ridge_und_klima", 2.0)])

        # DIE DREI ABSCHALTHAEKCHEN (Nutzerwunsch 2026-08-25):
        #
        #   *"kannst du mir einmal fuer flussnetzwerk und erosionfilter und
        #   kuestentypen jeweils checkboxen einfuegen, mit denen ich die
        #   effekte immer auch ausschalten kann?"*
        #
        # Zweck ist das VERGLEICHEN: welche Stufe traegt wieviel zum Bild
        # bei. Vorgabe ist ueberall AN, ein fehlender Parameter aendert also
        # nichts - Tools und Smoke-Tests rufen unveraendert auf.
        p = self._current_parameters or {}

        def _an(schluessel):
            return bool(p.get(schluessel, True))

        kuesten_an = _an("kuesten_archetypen_aktiv")
        filter_an = _an("erosion_filter_aktiv")
        fluesse_an = _an("river_network_aktiv")

        # DIE EINSTELLUNGEN DES REGIONSREITERS UND DER FORMREGLER.
        #
        # Bis zum 2026-08-26 kam hier NICHTS davon an: der Regionsreiter
        # hielt seine Ueberschreibungen fuer sich, und `weltfeld()` las
        # ausschliesslich den Katalog. Man konnte eine Region einstellen,
        # "Generieren" druecken und bekam dieselbe Karte - ohne Absturz und
        # ohne Meldung. Gefunden nur, indem nachgesehen wurde, WER die
        # Ueberschreibungen liest (niemand).
        #
        # `regionen_ueberschreibung` ist ein geschachteltes Dict
        # {Regionsname: {Regler: Wert}} und gilt NUR FUER DIESEN LAUF - der
        # Katalog bleibt die Vorgabe (Nutzerentscheidung 2026-08-25).
        ueberschreibung = p.get("regionen_ueberschreibung") or None
        kontinentform_regler = p.get("kontinentform")
        if ueberschreibung:
            self.logger.info(
                "Regionsueberschreibungen aktiv: %s",
                ", ".join(f"{n} ({', '.join(v)})"
                          for n, v in ueberschreibung.items()))
        if kontinentform_regler is not None:
            self.logger.info("Kontinentform-Regler auf %.2f",
                             float(kontinentform_regler))

        heightmap, felder = weltfeld(
            size, seed, shader_manager=self.shader_manager, schritte=schritte,
            kuesten_aktiv=kuesten_an,
            regionen_ueberschreibung=ueberschreibung,
            kontinentform_regler=(None if kontinentform_regler is None
                                  else float(kontinentform_regler)))
        mpp = WELT_KM * 1000.0 / float(size)

        # DER EROSIONSFILTER - nach dem Weltfeld, VOR dem Flussnetz.
        #
        # Diese Reihenfolge steht so im INTEGRATIONSPLAN (S4) und hat einen
        # Grund: der Filter erzeugt Grate und Rinnen, und die Taeler sollen in
        # genau diese Form geschnitten werden. Umgekehrt wuerde der Filter die
        # frisch eingegrabenen Taeler wieder zuschuetten.
        ridge_ersatz = None
        with _s(schritte, "erosionsfilter", "Erosionsfilter"):
            if filter_an:
                gefiltert = self._weltkarte_erosionsfilter(heightmap, felder)
            else:
                gefiltert = None
                self.logger.info("Erosionsfilter UEBERSPRUNGEN "
                                 "(erosion_filter_aktiv=False)")
        if gefiltert is not None:
            heightmap = gefiltert["heightmap"]
            ridge_ersatz = gefiltert["ridge_map"]

        fluss_maske = np.zeros((size, size), dtype=np.float32)
        fluss_ordnung = np.zeros((size, size), dtype=np.float32)
        fluss_generation = np.zeros((size, size), dtype=np.float32)
        fluss_wasser = np.zeros((size, size), dtype=np.float32)
        # Linienzuege fuer den Vektorexport (Ticket #37) - leer, solange
        # Fluesse aus/nicht aktiv sind, genau wie die vier Raster oben.
        fluss_linien_export: list = []

        if not fluesse_an:
            self.logger.info("Flussnetz und Taeler UEBERSPRUNGEN "
                             "(river_network_aktiv=False)")
        elif getattr(vd, "WELTFLUESSE_AKTIV", False):
            with _s(schritte, "weltfluesse", "Flussnetz und Taeler"):
                (heightmap, fluss_maske, fluss_ordnung, fluss_generation,
                 fluss_wasser, fluss_linien_export) = self._weltfluesse(
                     heightmap, felder, size, seed)

        # ridge_map ist ein Anzeige-Output des Erosionsfilters. Solange der bei
        # aktiver Weltkarte nicht laeuft, liefert die Hangneigung ein
        # brauchbares Ersatzbild - besser als ein fehlender Output, der die
        # Anzeige leer laesst (smoke_test_pipeline_outputs).
        # ridge_map: seit dem 2026-08-07 die echte des Erosionsfilters, wenn er
        # laeuft. Die Hangneigung als Ersatz war ein Notbehelf, solange er aus
        # war - sie zeigt Steilheit, nicht Grate.
        if ridge_ersatz is not None:
            ridge = ridge_ersatz
        else:
            gy, gx = np.gradient(heightmap.astype(np.float32), mpp)
            ridge = np.hypot(gx, gy).astype(np.float32)

        # region_map: welche der neun Regionen an diesem Pixel fuehrt (0..8 in
        # der Reihenfolge von alle_regionen(), Nordwest nach Suedost). Kommt aus
        # demselben Gewichtsfeld, das auch das Gelaende formt - der Siedlungs-
        # generator liest daran die Kultur ab, der Terrain-Reiter faerbt danach.
        # klima_map: drei Ebenen als EIN Output statt dreier Einzelausgaben.
        #
        #   [0] Jahresmitteltemperatur auf Meereshoehe, Grad
        #   [1] Jahresspanne (Juli minus Januar), Kelvin
        #   [2] Jahresniederschlag, mm
        #
        # Sie entstehen aus DENSELBEN Regionsgewichten wie das Gelaende, sind
        # also an den Regionsgrenzen bereits weich ueberblendet - genau die
        # Vorgabe des Nutzers vom 2026-08-07 ("wir muessen immer
        # Regionengrenzen sanft uebergehen lassen"). Wuerde das Wetter sie
        # ueber `region_map` (argmax) nachschlagen, gaebe es harte Kanten.
        _r_ctx = _s(schritte, "ridge_und_klima", "Grate und Klima")
        _r_ctx.__enter__()
        klima = np.stack([felder["temp_mittel_m0"],
                          felder["temp_spanne"],
                          felder["niederschlag_mm"]], axis=0).astype(np.float32)

        _r_ctx.__exit__(None, None, None)
        schritte.bericht()

        return {
            "heightmap": heightmap.astype(np.float32),
            "ridge_map": ridge,
            "river_mask": fluss_maske,
            "river_order": fluss_ordnung,
            "river_generation": fluss_generation,
            # Die EINE Flusskarte (Nutzervorgabe 2026-08-26): wieviel Wasser
            # an dieser Stelle berechnet wurde, in Niederschlag mal Flaeche.
            # Ersetzt die Rot/Gruen-Faerbung nach Generation als Leitansicht.
            "river_water": fluss_wasser,
            # Der Flussbaum als Linienzuege (Ticket #37, core/fluss_export.py)
            # - Liste von {"punkte","ordnung","breite_m"}-Dicts statt eines
            # Rasters, damit der Vektorexport scharfe Linien statt Treppen
            # bekommt. Siehe DOMAIN-Weiterleitung weiter unten (river_lines
            # in assemble_terrain_data()/TerrainData) und
            # set_terrain_data_complete_lod() fuer den Speicherpfad.
            "river_lines": fluss_linien_export,
            # DIE HOEHENFAKTOR-ANSICHT (Nutzerwunsch 2026-08-26:
            # *"kannst du mir die voronoiansicht als erstes bauen? ich
            # will den hoehenfaktor sehen koennen (3d und 2D)"*).
            #
            # `hinterlandhoehe_m`: je Pixel die gemessene Hoehe des
            # Hinterlands seines Kuestengebiets, in Metern (Band
            # 400-700 m, Zweipunktmethode des Nutzers). NaN auf See und
            # im alpinen Sonderfall.
            # `voronoi_map`: die Zellnummern, aus denen die Gebiete
            # gewachsen sind - dieselben Zellen, die auch die Regionen
            # bilden.
            "hinterland_height": felder.get("hinterlandhoehe_m"),
            "voronoi_map": felder.get("voronoi"),
            "region_map": felder["regionen"].astype(np.int16),
            "klima_map": klima,
            # Seegliederung (docs/spezifikation/12_WASSER.md Abschnitt 7, docs/OFFENE_PUNKTE.md
            # 3.1/3.2/3.6): seegrad 0 auf Land, 1..4+ auf See (Breitensuche
            # ueber den See-Voronoi-Zellgraphen); ufer_region_a/b die bis zu
            # zwei naechstgelegenen Regionen je Seezelle.
            "seegrad": felder["seegrad"],
            "ufer_region_a": felder["ufer_region_a"],
            "ufer_region_b": felder["ufer_region_b"],
            "see_eis": felder["see_eis"],
            # Kuesten-Archetypen (docs/OFFENE_PUNKTE.md 3.8) - lokaler Index
            # (0..2) INNERHALB der Region, zusammen mit region_map ueber
            # terrain_weltkarte.KUESTEN_ARCHETYPEN nachschlagbar; -1 = kein
            # Archetyp hier. kuesten_staerke ist die Blendstaerke selbst
            # (0..1) - die vom Nutzer angefragte "Strahlungstiefe".
            "kuesten_archetyp": felder.get("kuesten_archetyp"),
            "kuesten_staerke": felder.get("kuesten_staerke"),
            # Spielkarten-Zerlegung (docs/OFFENE_PUNKTE.md 5.15): neun konvexe
            # Vielecke mit etwa gleicher Landmasse, als Zuschnitt fuer die
            # Regionalansicht und spaeter den Godot-Export. Laeuft HIER, weil
            # sie genau die beiden Felder braucht, die an dieser Stelle
            # frisch vorliegen (fertige Heightmap und seegrad) - ein eigener
            # Calculator-Knoten muesste beide erneut anfordern.
            # Kostet rund 1 s bei 1024 px (gemessen), gegenueber den ~33 s
            # dieses Knotens vernachlaessigbar.
            "spielkarte": self._weltkarte_spielkarten(heightmap, felder, seed),
            # Regionsziel fuer die Windgeschwindigkeit (docs/spezifikation/10_REGIONEN.md B.5),
            # weich ueber die Regionsgrenzen gemischt wie klima_map - siehe
            # weather_generator.py._run_coupled_atmosphere_simulation fuer die
            # Verwendung als raeumlicher wind_speed_factor.
            "wind_ziel_map": felder["wind_mittel_ms"].astype(np.float32),
        }

    def _weltkarte_spielkarten(self, heightmap, felder, seed):
        """
        Die Zerlegung der Welt in neun Spielkarten (core/spielkarten.py,
        docs/OFFENE_PUNKTE.md 5.15) - konvexe Vielecke mit etwa gleicher
        Landmasse, wobei kuestennahe See zur Haelfte zaehlt.

        Rueckgabe: (H,W) int16 mit 0..8, oder None wenn die Zerlegung nicht
        moeglich ist (z.B. gar kein Land). **Ein Fehlschlag wird als WARNING
        geloggt und nicht still verschluckt** - ohne diese Zeile waere ein
        fehlendes Feld von einem absichtlich leeren nicht zu unterscheiden
        (CLAUDE.md, dieselbe Lehre wie beim adaptiven Mesh).

        Die Siedlungen sind hier bewusst NICHT dabei, obwohl `spielkarten.
        saatpunkte()` sie verarbeiten kann: sie entstehen erst weit spaeter in
        der Kette (settlement.settlements haengt ueber mehrere Stufen an
        diesem Knoten hier). Die Vorgabe "Schnittlinien zwischen den Staedten"
        braucht deshalb einen eigenen Umbau der Reihenfolge und ist bewusst
        aufgeschoben (Nutzer 2026-08-13: "die schnittlinie zwischen den
        staedten ist mit dem settlement-update verwandt und da gehen wir
        spaeter drauf ein").
        """
        try:
            from core import spielkarten
            ergebnis = spielkarten.zerlegen(
                heightmap, felder.get("seegrad"), siedlungen=None,
                anzahl=9, seed=int(seed))
            self.logger.debug(
                "Spielkarten: %d Runden, Massenspanne %.2f",
                ergebnis["runden"], ergebnis["spanne"])
            return ergebnis["karte"].astype(np.int16)
        except Exception as fehler:
            self.logger.warning(
                "Spielkarten-Zerlegung fehlgeschlagen (%s) - die Regionalansicht "
                "faellt auf das starre 3x3-Raster zurueck", fehler)
            return None

    def _weltfluesse(self, heightmap, felder, size, seed):
        """
        Flussnetz in drei Rechenstufen, dann die Taeler eingraben.

        Rueckgabe: (heightmap mit Taelern, maske, ordnung, generation, wasser,
        linien). `linien` ist der Baum ZUSAETZLICH als Linienzuege (Ticket
        #37, core/fluss_export.py) - der Graph wurde bis dahin nach der
        Rasterisierung weggeworfen, sodass Fluesse im Vektorexport
        (map_export.vektordaten()) komplett fehlten. Siehe Moduldocstring
        von core/fluss_export.py fuer Format und Begruendung.

        NUR UEBER WASSER GEZEICHNET. Die Laeufe reichen konstruktionsbedingt bis
        MUENDUNGSTIEFE_M (-50 m), damit ein Fluss sichtbar ins Meer muendet und
        die Muendungsrichtung stimmt. Alles unterhalb von 0 m wird in Maske und
        Ordnung weggelassen - dort ist Meer, kein Fluss.
        """
        from core.terrain_weltfluesse import (flussnetz, taeler_eingraben,
                                              MUENDUNGSTIEFE_M, ERBE_KOSTEN)
        import core.terrain_river_network as rn
        import core.terrain_weltkarte as rw
        from core.fluss_export import fluss_linien

        # DIE REGLER WIRKEN WIEDER (2026-08-06).
        #
        # Bis dahin nahm `flussnetz` ueberhaupt keine Parameter entgegen, und
        # die neun `river_*`-Regler der Oberflaeche bewegten nichts - gemessen
        # 0.00 m Hoehenaenderung und 0 abweichende Flusspixel bei allen neun.
        #
        # Fuenf haben eine echte Entsprechung im neuen Netz und sind hier
        # angeschlossen. Die uebrigen vier (`river_border_outflow`,
        # `river_divide_blend`, `river_plateau_flatten`, `river_meander`)
        # beschreiben Dinge, die es im Weltflussnetz nicht gibt - eine Insel
        # entwaessert ins Meer und nicht ueber den Kartenrand. Sie bleiben
        # gesperrt (gui/config/value_default.stillgelegte_regler).
        p = self._current_parameters

        def regler(name, vorgabe):
            wert = p.get(name)
            return float(wert) if wert is not None else float(vorgabe)

        netz = flussnetz(
            heightmap, seed,
            kosten_staerke=regler("river_cost_strength", 6.0),
            abstand_makro_m=regler("river_spacing_m", 1200.0),
            muendungstiefe_m=-abs(regler("river_mouth_depth_m",
                                         -MUENDUNGSTIEFE_M)),
            erbe_kosten=regler("river_inherit_cost", ERBE_KOSTEN),
            # WASSERMENGE STATT KNOTENZAHL (Block 1.1,
            # docs/spezifikation/12_WASSER.md). `felder` liegt hier seit jeher
            # vollstaendig vor - der Niederschlag wurde nur nie
            # weitergereicht, und das Flussnetz zaehlte deshalb Knoten
            # statt Wasser.
            niederschlag_mm=felder.get("niederschlag_mm"),
            # Fuer die Hauptstrom-Quote (Block 2, docs/spezifikation/12_WASSER.md).
            region_map=felder.get("regionen"))
        if netz is None:
            leer = np.zeros((size, size), dtype=np.float32)
            return heightmap, leer, leer.copy(), leer.copy(), leer.copy(), []

        geschnitten = taeler_eingraben(
            heightmap, netz, felder,
            breite_faktor=regler("river_valley_width", 0.35),
            tiefe_anteil=regler("river_incision_share", 0.30),
            form=regler("river_valley_form", 1.3),
            abstand_makro_m=regler("river_spacing_m", 1200.0))

        punkte, eltern = netz["punkte"], netz["eltern"]
        strahler = rn.strahler_order(eltern, netz["reihenfolge"])

        # LINIENZUEGE FUER DEN VEKTOREXPORT (Ticket #37, core/fluss_export.py).
        #
        # Derselbe Baum, VOR dem Wegwerfen in ein Raster, als Punktketten mit
        # Flussordnung und Breite - siehe Moduldocstring dort. mpp hier neu
        # berechnet (flussnetz() macht das intern genauso, gibt es aber nicht
        # zurueck).
        mpp = rw.WELT_KM * 1000.0 / size
        linien = fluss_linien(
            netz, strahler, felder, mpp,
            breite_faktor=regler("river_valley_width", 0.35),
            abstand_makro_m=regler("river_spacing_m", 1200.0))

        maske = np.zeros((size, size), dtype=np.float32)
        ordnung = np.zeros((size, size), dtype=np.float32)
        generation = np.zeros((size, size), dtype=np.float32)
        # DIE WASSERMENGE ALS EIGENE KARTE (2026-08-26).
        #
        # Nutzervorgabe: *"ich verstehe noch immer nicht die mehrteilung mit
        # roten und gruenen fluessen, jetzt wo wir quasi wassermengen und so
        # haben. koennen wir nur eine karte haben die darstellt wie viel
        # wasser fuer die fluesse berechnet wurde?"*
        #
        # `netz["flaeche"]` IST diese Groesse: sie akkumuliert seit dem
        # 2026-08-24 Niederschlag mal Flaeche flussabwaerts, nicht mehr
        # blosse Knotenzahl (siehe `baue_stufe` in terrain_weltfluesse.py).
        #
        # NICHT `flow_map` GENOMMEN, obwohl es die naheliegende Wahl waere:
        # die gehoert zur WASSER-Stufe (`get_water_data`), der Flussreiter
        # liest aber Terrain-Daten. Sie hier zu zeigen hiesse, den Reiter von
        # einer spaeteren Pipelinestufe abhaengig zu machen - er zeigte dann
        # nichts, solange die noch nicht gerechnet hat.
        wasser = np.zeros((size, size), dtype=np.float32)
        # DIESELBE SPLINE WIE BEIM EINSCHNEIDEN (2026-08-25).
        #
        # Hier stand `punkte[e]*(1-t) + punkte[i]*t`, also eine gerade Sehne
        # zwischen zwei Netzknoten - genauso wie in `taeler_eingraben()`. Der
        # Nutzer sah das als Zacken: *"die fluesse sind hier sehr zackig
        # gezeichnet"*. Gemessen am Hauptstrom (384 px, 40 Knoten) war der
        # groesste Richtungswechsel 143.5 Grad; mit der Spline sind es 33.3.
        #
        # WICHTIG, dass BEIDE Stellen dieselbe Funktion benutzen: die Maske
        # ist das, was man sieht, das Einschneiden das, was man begeht. Zwei
        # verschiedene Kurven waeren zwei Wahrheiten - der gezeichnete Fluss
        # laege dann neben seinem Tal.
        from core.terrain_weltfluesse import hauptkinder, kantenpunkte
        kinder = hauptkinder(eltern, netz["flaeche"])
        for i in range(len(punkte)):
            e = eltern[i]
            if e < 0:
                continue
            schritte = max(int(np.linalg.norm(punkte[i] - punkte[e]) * 2.0), 2)
            for p in kantenpunkte(punkte, eltern, kinder, e, i, schritte):
                y = int(np.clip(round(p[0]), 0, size - 1))
                x = int(np.clip(round(p[1]), 0, size - 1))
                if geschnitten[y, x] <= 0.0:
                    continue
                maske[y, x] = 1.0
                ordnung[y, x] = max(ordnung[y, x], float(strahler[i]))
                generation[y, x] = max(generation[y, x],
                                       float(3 - netz["lauf_stufe"][i]))
                # DIE LINIE WIRD MIT DER WASSERMENGE BREITER (2026-08-26).
                #
                # Nutzervorgabe: *"du solltest dort auch die dicke der linie
                # langsam steigen lassen mit der wassermenge. damit es
                # deutlicher ist."*
                #
                # NUR AUF DER ANZEIGEKARTE, nicht auf `maske`. `river_mask`
                # geht in die Biomklassifikation (Uferbiome, siehe
                # `river_bank` in core/biome_generator.py) - sie zu
                # verbreitern haette dort stillschweigend die Biomverteilung
                # verschoben. `river_water` ist ein reines Anzeigeprodukt und
                # darf breit sein.
                #
                # Logarithmisch, weil die Wassermenge es auch ist: gemessen
                # 0.38 bis 725 je Knoten (Median 1.33). Linear waere der
                # Hauptstrom 500-mal breiter als ein Bach.
                menge = float(netz["flaeche"][i])
                r = FLUSS_BREITE_GRUND_PX + FLUSS_BREITE_JE_DEKADE_PX * np.log10(
                    1.0 + max(menge, 0.0))
                rad = int(r)
                if rad <= 0:
                    wasser[y, x] = max(wasser[y, x], menge)
                else:
                    y0, y1 = max(0, y - rad), min(size, y + rad + 1)
                    x0, x1 = max(0, x - rad), min(size, x + rad + 1)
                    yy, xx = np.ogrid[y0:y1, x0:x1]
                    scheibe = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
                    ziel = wasser[y0:y1, x0:x1]
                    np.maximum(ziel, np.where(scheibe, menge, 0.0), out=ziel)
        return geschnitten, maske, ordnung, generation, wasser, linien

    def _apply_river_network(self, P: np.ndarray, amplitude: float):
        """
        Flussnetz-Skelett in die Flaeche P schneiden (SPEZIFIKATION §12,
        core/terrain_river_network.py).

        Wie beim Erosionsfilter bewusst KEIN eigener Calculator-Knoten: das
        Ergebnis ist die endgueltige Gelaendeform, und 20+ Lesestellen holen
        die Heightmap ueber ("terrain.redistribution", "heightmap"). Sie alle
        umzuhaengen ist das Risiko aus §4.5.

        Der Hoehenbereich wird danach NICHT hier zurueckgebildet - das macht
        _calc_redistribution ohnehin nach dem Erosionsfilter. Hier wird die
        Spanne zum Schluss noch einmal gesetzt, weil der Einschnitt die
        Talsohle unter die Talsohlenhoehe druecken kann.

        Returns: None wenn das Netz aus ist oder die Karte fuer den
        eingestellten Flussabstand zu klein ist.
        """
        from gui.config.value_default import FLUSSNETZ_AKTIV
        if not FLUSSNETZ_AKTIV:
            return None

        from core.terrain_river_network import carve_river_network

        parameters = self._current_parameters
        size = int(P.shape[0])
        km = float(self._ensure_data_lod_manager().get_map_distance_km())
        meters_per_pixel = km * 1000.0 / float(size)

        # Reglerwerte durchreichen; fehlt einer, gilt die Vorgabe des Moduls
        # (§4.1: durchreichen, nicht doppelt pflegen).
        netz_parameter = {}
        for parameter_key, modul_key in (
                ("river_spacing_m", "river_spacing_m"),
                ("river_incision_share", "incision_share"),
                ("river_valley_width", "valley_width_fraction"),
                ("river_valley_form", "valley_form"),
                ("river_meander", "meander"),
                ("river_divide_blend", "divide_blend"),
                ("river_cost_strength", "cost_strength"),
                ("river_border_outflow", "border_outflow")):
            if parameter_key in parameters:
                netz_parameter[modul_key] = parameters[parameter_key]

        # PLATEAU_FLATTEN ebnet die Flaeche ZWISCHEN den Taelern ein und laesst
        # den Gipfel stehen - der Regler, der Hochebene von Bergland trennt.
        # 0 = volles Relief (Bergland), hohe Werte = Hochebene.
        #
        # Nach oben begrenzt: bei exakt 1.0 waere P eine konstante Ebene, das
        # Flussnetz faende kein Gefaelle und schaltete sich ab.
        # Zahl der Auslaesse aus der Kartengroesse - auf 15 x 15 km liegen
        # keine drei unabhaengigen Flusssysteme (Nutzer, 2026-07-30).
        from gui.config.value_default import flussnetz_auslaesse
        netz_parameter.setdefault("outlet_count", flussnetz_auslaesse(km))

        flatten = float(np.clip(parameters.get("river_plateau_flatten", 0.0),
                                0.0, 0.9))
        if flatten > 0.0:
            P = P * (1.0 - flatten) + amplitude * flatten

        import time
        start = time.time()
        ergebnis = carve_river_network(
            P.astype(np.float64), meters_per_pixel, float(amplitude),
            int(parameters.get("map_seed", self.map_seed)), netz_parameter)
        if ergebnis is None:
            return None

        self.logger.debug(
            "River network: %d nodes, order up to %d, %dpx, %.2fs",
            ergebnis["node_count"], ergebnis["max_order"], size,
            time.time() - start)
        return ergebnis

    def _erosion_filter_parameters(self, size: int, map_distance_km: float) -> Dict[str, Any]:
        """
        Bildet die Regler des Terrain-Tabs auf die Parameter von
        core/terrain_erosion_filter.py ab.

        Eigene Methode, damit Messwerkzeuge denselben Weg gehen koennen wie die
        App. §4.2 ist der teuerste Fehlertyp dieses Projekts - dreimal an einem
        Tag wurde etwas anderes gemessen als lief.

        Fehlt ein Regler, gilt die Vorgabe des Filters. Es wird KEIN zweiter
        Satz Konstanten hier gefuehrt (§4.1: durchreichen, nicht doppelt
        pflegen).
        """
        from core.terrain_erosion_filter import ATEF_DEFAULTS

        parameters = self._current_parameters
        filter_parameters: Dict[str, Any] = {}
        for parameter_key, filter_key in (
                ("erosion_filter_strength", "erosion_strength"),
                ("erosion_filter_scale", "erosion_scale"),
                ("erosion_filter_detail", "erosion_detail"),
                ("erosion_filter_gully_weight", "erosion_gully_weight"),
                ("erosion_filter_octaves", "erosion_octaves")):
            if parameter_key in parameters:
                filter_parameters[filter_key] = parameters[parameter_key]

        # Die beiden Rundungen sind im Filter ein vec4; nur die ersten zwei
        # Komponenten sind Regler, die hinteren zwei bleiben bei den Werten des
        # Originals.
        vorgabe = ATEF_DEFAULTS["erosion_rounding"]
        filter_parameters["erosion_rounding"] = (
            float(parameters.get("erosion_filter_ridge_rounding", vorgabe[0])),
            float(parameters.get("erosion_filter_crease_rounding", vorgabe[1])),
            vorgabe[2], vorgabe[3])

        # RINNENGROESSE: der Regler steht in METERN, der Filter rechnet in
        # Kartenanteilen. Hier ist die eine Stelle, an der die drei Skalenebenen
        # (Aufloesung, reale Ausdehnung, Rinnengroesse) zusammengefuehrt werden.
        #
        # Ohne diese Umrechnung hing die Rinnengroesse an der Kartenausdehnung:
        # 5 / 15 / 50 km ergaben 631 / 1893 / 6310 m Rinnen, Faktor 10 ueber den
        # Bereich (smoke_test_terrain_scale_coupling.py, Lauf 2). Beim
        # Herauszoomen wurden die Rinnen groesser statt zahlreicher.
        #
        # erosion_filter_scale (der rohe Anteil) bleibt als Ueberschreibung
        # zulaessig und hat Vorrang - die Labore und der Paritaetstest gegen den
        # Shader brauchen den Wert, den das Original benutzt.
        if "erosion_scale" not in filter_parameters:
            karte_m = max(float(map_distance_km) * 1000.0, 1.0)
            groesse_m = float(parameters.get(
                "erosion_filter_gully_size_m", 0.15 * karte_m))
            anteil = groesse_m / karte_m

            # Untergrenze: eine Rinne, die schmaler als drei Pixel wird, ist
            # nicht mehr darstellbar und erzeugt nur Aliasing - dieselbe
            # Ueberlegung wie _max_safe_octaves() fuer die Oktaven, nur fuer die
            # Grundskala. Bewusst geometrisch (Pixel je Rinne) statt als fester
            # Meterwert, damit sie nicht an einer bestimmten Kartengroesse haengt.
            zell_skala = float(ATEF_DEFAULTS["erosion_cell_scale"])
            kleinster_anteil = 3.0 / max(zell_skala * float(size), 1.0)
            if anteil < kleinster_anteil:
                self.logger.debug(
                    "Erosion filter: gully size %.0f m raised to %.0f m - "
                    "below three pixels at %d px / %.1f km",
                    groesse_m, kleinster_anteil * karte_m, size, map_distance_km)
                anteil = kleinster_anteil
            filter_parameters["erosion_scale"] = min(anteil, 1.0)

        return filter_parameters

    def _weltkarte_erosionsfilter(self, heightmap, felder):
        """
        Der ATEF-Erosionsfilter fuer die Weltkarte, mit REGIONSGEWICHTUNG.

        Nutzerwunsch 2026-08-07: "der erosion filter sollte aber eigentlich
        schon einstellbar sein und ein automatischer faktor gewichtet das ganze
        pro region dann. also in den bergen wo mehr masse ist haben wir mehr
        features, baeche etc. und in den niederungen weniger."

        ZWEI UNTERSCHIEDE ZUM ALTEN PFAD, beide zwingend:

        1. KEINE HOEHENNORMIERUNG. `_apply_erosion_filter` zieht das Ergebnis
           zum Schluss auf 0..AMPLITUDE zurueck. Auf der Weltkarte waere das
           verheerend: dort steht die Hoehe in echten Metern und darf negativ
           sein, und eine Normierung wuerde die Kuestenlinie verschieben und
           alle neun Regionseichungen entwerten.

        2. DER FILTER SIEHT NUR LAND. Er normiert intern gegen die Spanne der
           uebergebenen Karte; mit dem Meeresboden bei -200 m waere diese
           Spanne zur Haelfte Wasser, und die Landstruktur bekaeme entsprechend
           weniger davon ab. Uebergeben wird deshalb max(H, 0).

        DIE GEWICHTUNG kommt aus `relief_m` - dem Feld, das ohnehin schon je
        Pixel vorliegt. Das Nevadin mit 1000 m Relief bekommt damit rund das
        Achtfache an Struktur wie das Clonagh mit 115 m.
        """
        from gui.config.value_default import EROSION_FILTER_AKTIV
        if not EROSION_FILTER_AKTIV:
            return None

        from core.terrain_erosion_filter import filter_heightmap
        import core.terrain_weltkarte as rw

        size = int(heightmap.shape[0])
        mpp = rw.WELT_KM * 1000.0 / float(size)
        parameter = self._erosion_filter_parameters(size, rw.WELT_KM)

        nur_land = np.maximum(heightmap, 0.0).astype(np.float32)
        ergebnis = filter_heightmap(nur_land, mpp, parameter)
        delta = np.asarray(ergebnis["height_delta"], dtype=np.float64)

        # REGIONSGEWICHT. Bezug ist EROSION_BEZUGSRELIEF_M, damit das Gewicht
        # eine Bedeutung hat und nicht am jeweiligen Kartenmittel haengt - eine
        # Normierung auf das Bild waere derselbe Fehler wie bei der Hoehenskala.
        relief = np.asarray(felder["relief_m"], dtype=np.float64)
        gewicht = np.clip(relief / EROSION_BEZUGSRELIEF_M,
                          EROSION_GEWICHT_MIN, EROSION_GEWICHT_MAX)

        # NUR UEBER WASSER. Unter der Wasserlinie gibt es keine Rinnen und
        # keine Grate; das Delta dort wuerde nur den Meeresboden aufrauhen.
        ueber_wasser = np.clip(heightmap / 50.0, 0.0, 1.0)

        return {
            "heightmap": (heightmap + delta * gewicht * ueber_wasser
                          ).astype(np.float32),
            "ridge_map": np.asarray(ergebnis["ridge_map"], dtype=np.float32),
        }

    def _apply_erosion_filter(self, heightmap: np.ndarray, amplitude: float):
        """
        ATEF-Erosionsfilter auf die fertig umverteilte Heightmap (SPEZIFIKATION
        §9, Portierung in core/terrain_erosion_filter.py).

        Absichtlich HIER und nicht als eigener Calculator-Knoten: der Filter
        liefert die endgueltige Geländeform, und 20+ Lesestellen in
        core/ und gui/ holen die Heightmap ueber
        ("terrain.redistribution", "heightmap"). Sie alle auf einen neuen Knoten
        umzuhaengen ist genau das Risiko, vor dem §4.5 warnt (fuenf
        handgepflegte Listen haben je einen Deadlock oder eine fehlende
        Invalidierung verursacht). So sehen Slope, Schatten, Geology, Weather,
        Water, Biome, die 2D-Anzeige, die 3D-Ansicht und der Export den Filter
        ohne eine einzige weitere Aenderung.

        Die ridge_map (-1 in Kerben, +1 auf Kaemmen) wird als zweiter Output
        desselben Knotens mitgespeichert - der Autor nennt sie ausdruecklich als
        Entwaesserungs-Eingang. Sie ist noch NICHT als Anzeige-Layer registriert
        und wird noch von niemandem gelesen.

        Returns: None wenn der Filter aus ist, sonst dict mit heightmap und
            ridge_map.
        """
        from gui.config.value_default import EROSION_FILTER_AKTIV
        if not EROSION_FILTER_AKTIV:
            return None

        from core.terrain_erosion_filter import ATEF_DEFAULTS, filter_heightmap

        size = int(heightmap.shape[0])
        km = float(self._ensure_data_lod_manager().get_map_distance_km())
        meters_per_pixel = km * 1000.0 / float(size)
        filter_parameters = self._erosion_filter_parameters(size, km)

        import time
        start = time.time()
        ergebnis = filter_heightmap(heightmap, meters_per_pixel, filter_parameters)
        gefiltert = heightmap + ergebnis["height_delta"]

        # Hoehenspanne wiederherstellen. §3.1 fuehrt "Hoehenspanne genau
        # BASE_ELEVATION_M .. AMPLITUDE" als erfuellt, und ein Delta obendrauf
        # reisst sie - gemessen wuchs das Relief um 6-8%. redistribute_power=1.0
        # macht _apply_redistribution() zur reinen linearen Abbildung auf die
        # Zielspanne, laesst die Form also unberuehrt.
        #
        # Zweiter Zweck: damit kann kein Reglerstand des Filters die Karte aus
        # ihrem Hoehenbereich schieben (§1, §4.7).
        gefiltert = self._apply_redistribution(gefiltert, 1.0, amplitude)

        self.logger.debug(
            "Erosion filter applied: %dpx, %d of %d octaves effective, %.2fs",
            size, ergebnis["effective_octaves"],
            filter_parameters.get("erosion_octaves",
                                  ATEF_DEFAULTS["erosion_octaves"]),
            time.time() - start)
        return {"heightmap": gefiltert, "ridge_map": ergebnis["ridge_map"]}

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

        # SCHATTEN WERDEN AUF DER OBERFLAECHE GEWORFEN, NICHT AUF DEM
        # MEERESBODEN.
        #
        # Der Meeresboden wirft keinen Schatten - dort steht Wasser, und dessen
        # Oberflaeche ist eben. Ohne diese Klemmung berechnete der Raycast die
        # Verschattung an der Unterwassertopografie, und das Wetter las sie als
        # Sonneneinstrahlung: `temp += (shadow - 0.5) * u_solar_power` im
        # Temperatur-Shader macht daraus einen Temperaturunterschied, den es
        # nicht geben kann.
        #
        # Das Gelaende selbst behaelt seine Tiefen - geklemmt wird nur, was in
        # den Raycast geht.
        import gui.config.value_default as vd
        if getattr(vd, "WELTKARTE_AKTIV", False):
            heightmap = np.maximum(heightmap, 0.0)

        # Die Pixelgroesse MUSS gesetzt sein, sonst rechnet der Schattenwurf mit
        # 1 m je Pixel und haelt jeden Hang fuer eine Wand (2026-08-07).
        manager = self._ensure_data_lod_manager()
        self.shadow_calculator.set_meters_per_pixel(
            float(manager.get_map_distance_km()) * 1000.0 / heightmap.shape[0])

        shadowmap = self.shadow_calculator.calculate_shadows(heightmap, lod_level)
        self.data_lod_manager.set_calculator_output(calculator_id, lod_level, {"shadowmap": shadowmap})
        self.logger.debug("Shadow calculation completed")

    def _apply_redistribution(self, heightmap: np.ndarray, redistribute_power: float,
                               amplitude: float = None,
                               base_elevation: float = None) -> np.ndarray:
        """
        Funktionsweise: Power-Redistribution UND Festlegung der Höhenspanne
        Parameter: heightmap, redistribute_power, amplitude - Gipfelhöhe in m,
            base_elevation - Talsohle in m (Default: TERRAIN.BASE_ELEVATION_M)
        Returns: numpy.ndarray - Heightmap, die von base_elevation bis amplitude reicht

        DIE SPANNE IST GARANTIERT: der höchste Punkt liegt exakt bei
        `amplitude`, der tiefste exakt bei `base_elevation`. `redistribute_power`
        steuert weiterhin die VERTEILUNG dazwischen (hoch = wenige Gipfel, viel
        Fläche nahe der Talsohle), nicht mehr die erreichte Höhe.

        Vorher wurde gegen die THEORETISCHE Amplitude normiert, und das verlor
        die Höhe an zwei Stellen (gemessen, Amplitude 4000 m, Default-Parameter):

            Simplex-fBm erreicht real nur -0.681 .. +0.651 statt -1 .. +1
                -> Gipfel schon bei 3302 m statt 4000 m
            danach (3302/4000)^3.5 = 0.511
                -> Gipfel bei 2044 m, Talsohle bei 6.5 m

        Aus 4000 m eingestellter Amplitude wurden also 2044 m Berge - der Regler
        log um fast die Hälfte. Der zweite Verlust ist der grössere und er
        WÄCHST mit redistribute_power: je mehr Fläche man in die Ebene drücken
        will, desto niedriger wurden zugleich die Gipfel, obwohl das zwei
        verschiedene Dinge sind.

        Der frühere Docstring begründete das Normieren gegen die theoretische
        Amplitude damit, ein Contrast-Stretch könne "die Landschaft nie absolut
        Richtung 0 m drücken". Das stimmt für einen Stretch NACH der Potenz -
        hier wird aber DAVOR normiert und danach auf die Zielspanne abgebildet.
        Die Potenz wirkt also unverändert auf die Verteilung; nur ihr
        Nebeneffekt auf die Gipfelhöhe ist weg.

        Preis dieser Entscheidung, bewusst in Kauf genommen: jede Karte reicht
        jetzt von der Talsohle bis zur Amplitude. Ein Seed, dessen Rauschen
        zufällig flacher ausfällt, ergibt keine flachere Landschaft mehr,
        sondern dieselbe Spanne mit anderer Form.
        """
        if amplitude is None or amplitude <= 0:
            # Ohne Zielhöhe (Legacy-Aufrufer) bleibt nur die reine Potenz auf
            # der vorhandenen Spanne - dann ist nichts zu garantieren.
            low, high = float(heightmap.min()), float(heightmap.max())
            if high - low < 1e-9:
                return heightmap
            normalized = (heightmap - low) / (high - low)
            return (np.power(normalized, redistribute_power) * (high - low)
                    + low).astype(np.float32)

        if base_elevation is None:
            from gui.config.value_default import TERRAIN
            base_elevation = TERRAIN.BASE_ELEVATION_M

        low, high = float(heightmap.min()), float(heightmap.max())
        if high - low < 1e-9:
            return np.full_like(heightmap, base_elevation, dtype=np.float32)

        # 1. auf die TATSÄCHLICHE Stichprobenspanne normieren (nicht auf die
        #    theoretische) - das holt den ersten Verlust zurück
        normalized = (heightmap - low) / (high - low)
        # 2. Verteilung formen - unverändert der bisherige Mechanismus
        redistributed = np.power(normalized, redistribute_power)
        # 3. auf die Zielspanne abbilden - das holt den zweiten Verlust zurück
        result = base_elevation + redistributed * (amplitude - base_elevation)
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

        # Range-Validation - Grenzen direkt aus gui/config/value_default.py TERRAIN
        # gelesen statt hier ein zweites Mal hart codiert, damit beide Quellen
        # strukturell nicht mehr auseinanderdriften können (vorher erlaubte
        # 'octaves' hier bis 12, obwohl der UI-Slider nur bis 8 geht).
        from gui.config.value_default import TERRAIN
        param_to_config = {
            'map_size': TERRAIN.MAPSIZE, 'amplitude': TERRAIN.AMPLITUDE,
            'octaves': TERRAIN.OCTAVES, 'frequency': TERRAIN.FREQUENCY,
            'persistence': TERRAIN.PERSISTENCE, 'lacunarity': TERRAIN.LACUNARITY,
            'redistribute_power': TERRAIN.REDISTRIBUTE_POWER,
            # Nicht in required_params (Slider ist neu, ältere/Standalone-
            # Aufrufer liefern ihn evtl. noch nicht mit) - wird trotzdem
            # geprüft, sobald vorhanden.
            'map_distance_km': TERRAIN.MAP_DISTANCE_KM,
        }

        for param, config in param_to_config.items():
            if param not in parameters:
                continue
            value = parameters[param]
            if not (config["min"] <= value <= config["max"]):
                raise ValueError(f"Invalid value for {param}: {value}")
            if param == 'map_size' and value % 32 != 0:
                raise ValueError(f"Invalid value for {param}: {value} (must be a multiple of 32)")

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

    # Test Generator ohne ShaderManager
    generator = create_terrain_generator(map_seed=12345, shader_manager=None)

    generator._validate_parameters(test_params)
    print("✓ Parameter validation passed")

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