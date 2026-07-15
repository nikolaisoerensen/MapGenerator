"""
Path: core/weather_generator.py
Date Changed: 25.08.2025

Funktionsweise: Dynamisches Wetter- und Feuchtigkeitssystem mit DataLODManager-Integration
- CFD-basierte Windsimulation mit Navier-Stokes-Gleichungen
- GPU-Shader-Integration mit 3-stufigem Fallback-System
- Numerisches LOD-System mit progressiver CFD-Komplexität
- Orographische Effekte mit Luv-/Lee-Berechnung
- Bidirektionale Terrain-Integration mit heightmap_combined

Parameter Input:
- air_temp_entry (Lufttemperatur bei Karteneintritt in °C)
- solar_power (max. solare Gewinne, default 20°C)
- altitude_cooling (Abkühlen der Luft pro km Altitude, default 6°C)
- thermic_effect (Thermische Verformung der Windvektoren durch shademap)
- wind_speed_factor (Windgeschwindigkeit je Luftdruckdifferenz)
- terrain_factor (Einfluss von Terrain auf Wind und Temperatur)

Dependencies (über DataLODManager):
- heightmap_combined (von terrain_generator, post-erosion wenn verfügbar)
- shadowmap (von terrain_generator für Sonneneinstrahlung)

Output:
- WeatherData-Objekt mit wind_map, temp_map, precip_map, humid_map
- DataLODManager-Storage für nachfolgende Generatoren (water, biome)

LOD-System (Numerisch):
- lod_level 1: 32x32, 3 CFD-Iterationen für schnelle Preview
- lod_level 2: 64x64, 5 CFD-Iterationen mit Enhanced-Effects
- lod_level 3: 128x128, 7 CFD-Iterationen mit Detailed-Orographics
- lod_level 4: 256x256, 10 CFD-Iterationen mit High-Quality-Physics
- lod_level 5: 512x512, 15 CFD-Iterationen mit Premium-Simulation
- lod_level 6+: bis map_size, 20 CFD-Iterationen mit Maximum-Quality
"""

import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.fft import dctn, idctn
import logging
from typing import Dict, Any, Optional, Tuple


class AtmosphereLayers:
    """
    Geometrie der 3 Höhenschichten für die gekoppelte Atmosphären-Simulation
    (siehe _run_coupled_atmosphere_simulation). Terrain-folgend: die absolute
    Höhe von Schicht L an Zelle (x,y) ist heightmap(x,y) + REF_ALTITUDE_AGL[L]
    - ein Berg hebt den gesamten Schichtstapel mit an, wodurch Orographie
    (Steigungsregen, Hangbeschleunigung) über alle 3 Schichten spürbar bleibt.

    Bänder (0-150m / 150-1200m / >1200m AGL) sind eine grobe Anlehnung an
    reale Grenzschicht/freie Troposphäre, keine exakt recherchierten Werte -
    mit dem Nutzer abgestimmter Startwert, leicht anpassbar.
    """
    GROUND = 0
    MID = 1
    HIGH = 2
    COUNT = 3
    NAMES = ("ground", "mid", "high")
    # Schicht-Mittelhöhen (AGL) für die Potentiell->Real-Temperatur-Umrechnung
    # (siehe potential_to_real_temperature) und für die höhenabhängige
    # Dämpfung der alten Einzelschicht-Terrainterme (siehe
    # _run_coupled_atmosphere_simulation Schritt 2).
    REF_ALTITUDE_AGL = (75.0, 675.0, 2200.0)
    # Nominelle Schicht-Dicken (0-150m / 150-1200m / 1200-3200m), nur für die
    # vertikale Fluss-Divergenz in _apply_continuity_correction genutzt (w/dicke) -
    # HIGH hat keine echte Obergrenze, 3200m ist ein repräsentativer Abschluss.
    THICKNESS_M = (150.0, 1050.0, 2000.0)


# Rauigkeits-Lookup-Tabelle je Biome-Kategorie (core/biome_generator.py:859-875
# Basis-Biome 0-14, :981-999 Wasser-Super-Biome 15-19) -> ungefährer
# Dämpfungs-Anteil [0,1] für bodennahen Wind, grob an WindNinjas
# Rauigkeitslängen-Konzept angelehnt (dichter Bewuchs bremst Wind nahe am
# Boden stärker als offenes Gelände/Wasser). Stilisierte, nicht
# meteorologisch kalibrierte Werte - siehe [[project-wind-roughness]] und
# WeatherSystemGenerator._get_roughness_damping().
_BIOME_ROUGHNESS_DAMPING = np.array([
    0.03, 0.08, 0.30, 0.12, 0.32, 0.18, 0.04, 0.08, 0.35, 0.28,
    0.14, 0.30, 0.22, 0.10, 0.05,   # 0-14: ice_cap..badlands
    0.00, 0.00, 0.02, 0.02, 0.02,   # 15-19: ocean/lake/grand_river/river/creek
], dtype=np.float32)

# Stärke des Grat-/Canyon-Speedup-Terms (WindNinja-Terrain-Shape-Effekt,
# siehe [[project-wind-ridge-speedup]]) - maximale multiplikative
# Geschwindigkeitsänderung (+/-) bei extremer (3-Sigma-)Krümmung, empirischer
# Startwert wie die übrigen internen Skalierungskonstanten dieser Datei.
_RIDGE_SPEEDUP_STRENGTH = 0.35


class WeatherData:
    """
    Container für alle Weather-Daten mit vollständiger LOD-Integration

    Attributes:
        wind_map: 2D numpy.float32 array (H,W,2), Windvektoren in m/s
        temp_map: 2D numpy.float32 array, Lufttemperatur in °C
        precip_map: 2D numpy.float32 array, Niederschlag in gH2O/m²
        humid_map: 2D numpy.float32 array, Luftfeuchtigkeit in gH2O/m³
        lod_level: int, Numerisches LOD-Level
        actual_size: int, Tatsächliche Kartengröße
        validity_state: dict, Cache-Invalidation-State
        parameter_hash: str, Parameter-Hash für Cache-Management
        performance_stats: dict, CFD-Performance-Metriken
    """

    def __init__(self):
        self.wind_map = None
        self.temp_map = None
        self.precip_map = None
        self.humid_map = None
        # Saisonale Rohwerte (je Liste aus 6 np.ndarray, eine pro Zwei-Monats-
        # Periode Jan/Feb..Nov/Dez) - für die animierte Weather-Tab-Anzeige.
        # temp_map/wind_map/humid_map/precip_map bleiben der saisonale
        # Mittelwert daraus, von Water/Biome/Default-GUI-Ansicht konsumiert.
        self.wind_map_monthly = None
        self.temp_map_monthly = None
        self.precip_map_monthly = None
        self.humid_map_monthly = None
        # 3-Schicht-Atmosphäre (siehe AtmosphereLayers/_run_coupled_atmosphere_
        # simulation) - rein ADDITIV zu den Feldern oben, die weiterhin die
        # GROUND-Schicht widerspiegeln (Rückwärtskompatibilität für 2D/3D-
        # Anzeige, water.evaporation, alle Biome-Knoten - siehe Docstring von
        # _calc_temperature). Kein bestehender Konsument liest diese Felder.
        self.wind_map_layers = None    # (3,H,W,2) float32 m/s
        self.temp_map_layers = None    # (3,H,W) float32 °C (reale, keine potentielle Temp)
        self.humid_map_layers = None   # (3,H,W) float32, gleiche Skala wie humid_map
        self.wind_map_layers_monthly = None   # Liste von 6x (3,H,W,2)
        self.temp_map_layers_monthly = None   # Liste von 6x (3,H,W)
        self.humid_map_layers_monthly = None  # Liste von 6x (3,H,W)
        self.lod_level = 1
        self.actual_size = 32
        self.validity_state = {"valid": True, "dependencies_satisfied": False}
        self.parameter_hash = ""
        self.performance_stats = {}

    def is_valid(self) -> bool:
        """Prüft Validity-State für Cache-Management"""
        return self.validity_state.get("valid", False)

    def invalidate(self):
        """Invalidiert Weather-Data für Cache-Management"""
        self.validity_state["valid"] = False

    def get_validity_summary(self) -> dict:
        """Liefert Validity-Summary für DataLODManager"""
        return {
            "valid": self.is_valid(),
            "lod_level": self.lod_level,
            "size": self.actual_size,
            "parameter_hash": self.parameter_hash
        }


class WeatherSystemGenerator:
    """
    Hauptklasse für dynamisches Wetter- und Feuchtigkeitssystem mit vollständiger Manager-Integration

    Koordiniert CFD-basierte Atmosphärensimulation mit GPU-Acceleration und 3-stufigem Fallback-System.
    Implementiert Navier-Stokes-Gleichungen für realistische Windfelder mit orographischen Effekten.
    """

    def __init__(self, map_seed: int = 42, shader_manager=None, data_lod_manager=None):
        """
        Initialisiert Weather-System mit Manager-Integration

        Args:
            map_seed: Seed für reproduzierbare Weather-Patterns
            shader_manager: ShaderManager für GPU-Acceleration
            data_lod_manager: DataLODManager für Input-Dependencies
        """
        self.map_seed = map_seed
        self.shader_manager = shader_manager
        self.data_lod_manager = data_lod_manager
        self.noise_generator = OpenSimplex(seed=map_seed)

        # Logger für Debug und Performance-Monitoring
        self.logger = logging.getLogger(__name__)

        # Sub-Komponenten
        self.temp_calculator = TemperatureCalculator()
        self.wind_simulator = WindFieldSimulator()
        self.precip_system = PrecipitationSystem()
        self.moisture_manager = AtmosphericMoistureManager()

        # Eigene ShadowCalculator-Instanz (separat von Terrains) für die
        # saisonalen Monats-Shadowmaps der 6-Monats-Simulation (siehe
        # _calc_temperature) - nutzt denselben shader_manager für GPU-
        # Beschleunigung, überschreibt aber nie den geteilten
        # terrain.shadow-Knoten-Output.
        from core.terrain_generator import ShadowCalculator
        self.shadow_calculator = ShadowCalculator(shader_manager=shader_manager)

        # Performance-Tracking
        self.performance_stats = {}

        # Cache für die separierbaren Poisson-Eigenwerte (1/lambda) der
        # DCT-basierten Kontinuitätskorrektur (_apply_continuity_correction),
        # pro (height, width) - siehe [[project-wind-poisson-projection]].
        # Spart die wiederholte cos()-Berechnung des Nenners, nicht die
        # eigentliche FFT-Kosten.
        self._poisson_eig_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Progress-Callback für UI-Updates
        self.progress_callback = None

        # Parameter der aktuell laufenden Generierungs-Anfrage - vom
        # GenerationOrchestrator einmal pro frischer Anfrage über
        # set_active_parameters() gesetzt, bleibt über alle LOD-Runden dieser
        # Anfrage hinweg konstant.
        self._current_parameters: Dict[str, Any] = {}

    def set_active_parameters(self, parameters: Dict[str, Any]):
        """Setzt die Parameter, die alle _calc_*-Methoden bis zur nächsten frischen
        Anfrage verwenden (vom GenerationOrchestrator aufgerufen)."""
        self._current_parameters = parameters

    def _generate_seasonal_parameters(self, base_parameters: Dict[str, Any],
                                       month_index: int) -> Dict[str, Any]:
        """
        Funktionsweise: month_index 0..5 = Jan/Feb .. Nov/Dez. Nutzt
        WEATHER.CLIMATE_ZONE_SEASONAL_OFFSETS als saisonale FORM um den
        User-Slider-Wert als Zentrum ("Jahres-Mittel" der Klimazone), statt
        einer beliebigen Sinus-Kurve - Werte bleiben dadurch klimazonen-
        typisch plausibel. Deterministisch aus map_seed (analog Fix #21:
        aus parameters gelesen, nicht aus dem konstruktionszeit-fixen
        self.map_seed, da GenerationOrchestrator.get_generator_instance()
        WeatherSystemGenerator ohne map_seed konstruiert).
        Aufgabe: Liefert ein vollständiges Parameter-Dict für einen einzelnen
        Monats-Durchlauf der Monats-Schleife in den _calc_*-Methoden.
        """
        from gui.config.value_default import WEATHER

        profile = WEATHER.CLIMATE_ZONE_SEASONAL_OFFSETS[WEATHER.CLIMATE_ZONE]
        map_seed = base_parameters.get('map_seed', self.map_seed)
        rng = np.random.default_rng((int(map_seed) * 1000 + month_index) % (2 ** 32))

        # altitude_cooling/thermic_effect/terrain_factor bleiben unverändert
        # (physikalische Konstanten, keine saisonalen Treiber)
        params = dict(base_parameters)

        def _apply_offset(key: str, config: Dict[str, Any], jitter_frac: float = 0.04) -> float:
            base_value = base_parameters.get(key, config["default"])
            offset = profile[key][month_index]
            jitter = rng.normal(0.0, jitter_frac * (config["max"] - config["min"]))
            return float(np.clip(base_value + offset + jitter, config["min"], config["max"]))

        params["air_temp_entry"] = _apply_offset("air_temp_entry", WEATHER.AIR_TEMP_ENTRY)
        params["solar_power"] = _apply_offset("solar_power", WEATHER.SOLAR_POWER)
        params["wind_speed_factor"] = _apply_offset("wind_speed_factor", WEATHER.WIND_SPEED_FACTOR)
        params["air_humidity_entry"] = _apply_offset("air_humidity_entry", WEATHER.AIR_HUMIDITY_ENTRY)

        # Windrichtung ist zirkular - Rotation der Basisrichtung um eine
        # saisonale Amplitude (moderat, "vorherrschende Richtung" soll nicht
        # beliebig kippen) statt eines linearen Offsets (0/360°-Wrap-Bug).
        base_direction = base_parameters.get(
            "prevailing_wind_direction", WEATHER.PREVAILING_WIND_DIRECTION["default"])
        wind_swing_deg = 35.0
        seasonal_strength = profile["air_temp_entry"][month_index] / 9.0  # -1..1-artige Normierung
        params["prevailing_wind_direction"] = (
            base_direction + wind_swing_deg * seasonal_strength + rng.normal(0.0, 8.0)
        ) % 360.0

        # Für _generate_pressure_noise() (CPU) und den GPU-Gegenpart in
        # shader_manager.py - macht das kleinräumige Windfeld-Rauschmuster
        # pro Monat optisch unterscheidbar statt für alle 6 Perioden identisch.
        params["month_index"] = month_index

        return params

    def _ensure_data_lod_manager(self):
        """Lazy-Fallback für Standalone-Nutzung (Tests, calculate_weather_system()
        ohne injizierten Manager) - die echte Pipeline injiziert immer einen über
        GenerationOrchestrator.get_generator_instance()."""
        if self.data_lod_manager is None:
            from gui.OldManagers.data_lod_manager import DataLODManager
            self.data_lod_manager = DataLODManager()
        return self.data_lod_manager

    def calculate_weather_system(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                                parameters: Dict[str, Any], lod_level: int) -> WeatherData:
        """
        Funktionsweise: Standalone-Convenience-Entry-Point (Legacy-Kompatibilität + Tests)
        Aufgabe: Führt alle 4 Weather-Calculator-Knoten synchron für EIN LOD aus und
            liefert das fertige WeatherData-Objekt. Die echte GUI-Pipeline
            (GenerationOrchestrator) ruft dieselben _calc_*-Methoden ab jetzt einzeln
            über den globalen CalculatorDispatcher auf (Tracker #16 LOD-Lockstep-
            Umbau) - der Effekt ist identisch, da beide Wege denselben Storage nutzen.
            Weather hat - anders als Geology/Water - keinen Cross-LOD-
            Akkumulationszustand, jeder Aufruf ist eigenständig.

        Args:
            heightmap_combined: Post-Erosion Heightmap vom Water-Generator oder Original-Heightmap
            shadowmap: Shadow-Map vom Terrain-Generator für Sonneneinstrahlung
            parameters: Alle Weather-Parameter aus ParameterManager
            lod_level: Numerisches LOD-Level (1-6+) für Progressive Enhancement

        Returns:
            WeatherData: Vollständiges Weather-System mit allen Outputs

        Raises:
            ValueError: Bei ungültigen Input-Dependencies oder Parameter-Ranges
            RuntimeError: Bei kritischen CFD-Solver-Failures
        """
        try:
            self.logger.debug(f"Starting weather generation - LOD {lod_level}, size: {heightmap_combined.shape}")

            self._validate_inputs(heightmap_combined, shadowmap, parameters, lod_level)
            self._ensure_data_lod_manager()
            self.set_active_parameters(parameters)

            # Standalone-Convenience-Pfad: heightmap_combined/shadowmap kommen hier
            # als direkte Parameter, nicht aus dem DataLODManager - für die
            # _calc_*-Methoden (die jetzt IMMER aus dem feingranularen Calculator-
            # Storage lesen, siehe get_calculator_combined_heightmap()) dort
            # gespiegelt, analog zu Geology/Water.
            self.data_lod_manager.set_calculator_output(
                "terrain.redistribution", lod_level, {"heightmap": heightmap_combined})
            self.data_lod_manager.set_calculator_output(
                "terrain.shadow", lod_level, {"shadowmap": shadowmap})

            for calculator_id in (
                "weather.temperature", "weather.wind", "weather.humidity", "weather.precipitation",
            ):
                getattr(self, "_calc_" + calculator_id.split(".", 1)[1])(calculator_id, lod_level)

            weather_data = self.assemble_weather_data(lod_level, parameters)

            self.logger.info(f"Weather generation completed successfully - LOD {lod_level}")

            return weather_data

        except Exception as e:
            self.logger.error(f"Weather generation failed: {str(e)}")
            # Error-Recovery: Fallback zu Simplified-Weather-System
            return self._create_fallback_weather_data(heightmap_combined.shape[0], lod_level, parameters)

    def assemble_weather_data(self, lod_level: int, parameters: Dict[str, Any]) -> WeatherData:
        """
        Funktionsweise: Baut das finale WeatherData-Objekt aus den einzeln
        gespeicherten Calculator-Outputs zusammen
        Aufgabe: Wird vom GenerationOrchestrator aufgerufen, sobald alle 4 Weather-
            Calculator-Knoten ein LOD abgeschlossen haben (siehe Task 18 im
            LOD-Lockstep-Umbau)
        """
        temp_map = self.data_lod_manager.get_calculator_output("weather.temperature", "temp_map", lod_level)
        wind_map = self.data_lod_manager.get_calculator_output("weather.wind", "wind_map", lod_level)
        humid_map = self.data_lod_manager.get_calculator_output("weather.humidity", "humid_map", lod_level)
        precip_map = self.data_lod_manager.get_calculator_output("weather.precipitation", "precip_map", lod_level)

        if temp_map is None or wind_map is None or humid_map is None or precip_map is None:
            raise ValueError(f"assemble_weather_data: fehlende Calculator-Outputs für LOD {lod_level}")

        # Saisonale Monats-Listen (für die animierte Weather-Tab-Anzeige) -
        # optional, da nur relevant, wenn die _calc_*-Methoden tatsächlich
        # über die Monats-Schleife gelaufen sind (z.B. immer der Fall bei der
        # echten Pipeline, aber None bei alten Cache-Einträgen).
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        wind_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_monthly", lod_level)
        humid_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_monthly", lod_level)
        precip_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.precipitation", "precip_map_monthly", lod_level)

        # 3-Schicht-Atmosphäre (siehe AtmosphereLayers/_run_coupled_atmosphere_
        # simulation, [[project-3layer-wind-cfd]]) - optional, None bei altem
        # Cache oder falls der gekoppelte Loop diese Runde in den Einzelschicht-
        # Fallback gefallen ist (siehe _calc_temperature-Docstring).
        temp_map_layers = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_layers", lod_level)
        wind_map_layers = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_layers", lod_level)
        humid_map_layers = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_layers", lod_level)
        temp_map_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_layers_monthly", lod_level)
        wind_map_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_layers_monthly", lod_level)
        humid_map_layers_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_layers_monthly", lod_level)

        target_size = temp_map.shape[0]
        weather_data = self._create_weather_data(
            wind_map, temp_map, precip_map, humid_map, lod_level, target_size, parameters,
            wind_map_monthly=wind_map_monthly, temp_map_monthly=temp_map_monthly,
            precip_map_monthly=precip_map_monthly, humid_map_monthly=humid_map_monthly,
            temp_map_layers=temp_map_layers, wind_map_layers=wind_map_layers,
            humid_map_layers=humid_map_layers, temp_map_layers_monthly=temp_map_layers_monthly,
            wind_map_layers_monthly=wind_map_layers_monthly,
            humid_map_layers_monthly=humid_map_layers_monthly)

        cfd_iterations = self._get_cfd_iterations(lod_level)
        self._update_performance_stats(weather_data, cfd_iterations)

        return weather_data

    def _get_prepared_terrain_inputs(self, lod_level: int):
        """
        Holt heightmap_combined/shadowmap für dieses LOD und bringt sie auf die
        Weather-eigene Ziel-Auflösung (siehe _get_lod_size()) - Ersatz für das
        frühere context-basierte Vorab-Interpolieren, das nur innerhalb EINES
        calculate_weather_system()-Aufrufs existierte. Jede _calc_*-Methode ruft
        das unabhängig auf (billige, reine Operation - kein gemeinsamer Zustand
        zwischen separaten Dispatch-Aufrufen nötig).
        """
        heightmap_combined = self.data_lod_manager.get_calculator_combined_heightmap(lod_level)
        shadowmap = self.data_lod_manager.get_calculator_output("terrain.shadow", "shadowmap", lod_level)
        if heightmap_combined is None or shadowmap is None:
            raise ValueError(f"Weather: heightmap_combined/shadowmap für LOD {lod_level} nicht verfügbar")

        target_size = self._get_lod_size(lod_level, heightmap_combined.shape[0])
        heightmap, shadowmap = self._prepare_input_data(heightmap_combined, shadowmap, target_size)
        return heightmap, shadowmap, target_size

    def _calc_temperature(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.temperature' (#11) - Wirtsknoten für die
        gekoppelte 3-Schicht-Atmosphären-Simulation (siehe
        _run_coupled_atmosphere_simulation, [[project-3layer-wind-cfd]]).

        Läuft intern über 6 saisonale Zwei-Monats-Perioden (Jan/Feb..Nov/Dez,
        siehe _generate_seasonal_parameters()). Jeder Monat bekommt seine
        eigene, astronomisch berechnete Shadowmap (generate_seasonal_sun_angles()
        + self.shadow_calculator, NICHT der geteilte terrain.shadow-Output -
        Winter bekommt einen flacheren, Sommer einen steileren Sonnenstand).

        WICHTIG (DAG-Design-Entscheidung): dieser Knoten hat im CALCULATOR_GRAPH
        keine depends_on-Kante zu weather.wind/weather.humidity - ist also
        topologisch GARANTIERT der erste der 3, der pro Runde fertig wird.
        Deshalb läuft der GESAMTE gekoppelte Temp+Wind+Feuchte+Niederschlag-Loop
        HIER statt in _calc_wind: würde der Loop stattdessen in _calc_wind
        laufen, könnte biome.super_override (hängt NUR von weather.temperature
        ab, nicht von weather.wind) einen nur-"geseedeten" temp_map-Wert lesen,
        bevor der Loop fertig ist - ein Race-Window, das mit dem Loop hier
        (wo kein Zwischenzustand existiert, der fälschlich als fertig gilt)
        gar nicht erst entsteht. Ergebnisse werden explizit unter ALLEN 4
        Calculator-IDs gespeichert (set_calculator_output nimmt calculator_id
        als reinen String ohne Validierung gegen den ausführenden Knoten
        entgegen) - _calc_wind/_calc_humidity/_calc_precipitation werden
        dadurch zu dünnen Pass-Throughs (siehe deren Docstrings).

        Die GROUND-Schicht füllt weiterhin temp_map/wind_map/humid_map/
        precip_map (+ _monthly) - exakt wie vor dem 3-Schicht-Umbau, für
        Rückwärtskompatibilität mit 2D/3D-Anzeige, water.evaporation und allen
        Biome-Knoten. Die vollen 3 Schichten landen zusätzlich, rein additiv,
        unter *_layers/*_layers_monthly (siehe WeatherData).

        Fallback: schlägt der gekoppelte Loop fehl (Exception - z.B. bei
        pathologischen Eingaben), fällt dieser Knoten auf die alte
        Einzelschicht-Temperatur-Logik zurück und schreibt NUR
        temp_map/temp_map_monthly/shadowmap_monthly - wind/humidity/
        precipitation bleiben dann unter ihren jeweiligen calculator_ids leer,
        wodurch die Pass-Through-Knoten IHREN eigenen alten Einzelschicht-
        Fallback auslösen (etabliertes 3-stufiges Fallback-Muster dieser Datei).
        """
        self._update_progress("Temperature", 20, "Calculating coupled 3-layer atmosphere...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        atmosphere_steps = self._get_atmosphere_loop_steps(lod_level)
        # Statisch über alle 6 Monate dieser Runde - einmal holen statt in
        # _run_coupled_atmosphere_simulation sechsmal neu abzufragen (siehe
        # [[project-wind-roughness]]).
        roughness_damping = self._get_roughness_damping(heightmap.shape, lod_level)

        from gui.config.value_default import WEATHER
        from core.terrain_generator import generate_seasonal_sun_angles
        latitude = self._current_parameters.get('map_latitude', WEATHER.MAP_LATITUDE["default"])
        longitude = self._current_parameters.get('map_longitude', WEATHER.MAP_LONGITUDE["default"])

        monthly_shadowmaps = []
        month_params_list = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            sun_angles = generate_seasonal_sun_angles(month_index, latitude, longitude)
            month_shadowmap = self.shadow_calculator.calculate_shadows(
                heightmap, lod_level, sun_angles_override=sun_angles)
            monthly_shadowmaps.append(month_shadowmap)
            month_params_list.append(month_params)

        try:
            monthly_temp_maps, monthly_wind_maps = [], []
            monthly_humid_maps, monthly_precip_maps = [], []
            monthly_temp_layers, monthly_wind_layers, monthly_humid_layers = [], [], []

            for month_index in range(6):
                result = self._run_coupled_atmosphere_simulation(
                    heightmap, monthly_shadowmaps[month_index], month_params_list[month_index],
                    target_size, n_steps=atmosphere_steps, roughness_damping=roughness_damping)

                monthly_temp_layers.append(result['temp_layers'])
                monthly_wind_layers.append(result['wind_layers'])
                monthly_humid_layers.append(result['humid_layers'])

                monthly_temp_maps.append(result['temp_layers'][AtmosphereLayers.GROUND])
                monthly_wind_maps.append(result['wind_layers'][AtmosphereLayers.GROUND])
                monthly_humid_maps.append(result['humid_layers'][AtmosphereLayers.GROUND])
                monthly_precip_maps.append(result['precip_map'])

            # Mehrgenerationen-Puffer für Feuchte (siehe frühere _calc_humidity-
            # Fassung, Verhalten hier 1:1 erhalten, nur an den neuen Aufrufort
            # verschoben): gewichteter Mittelwert statt additiver Akkumulation,
            # PRO Monatsindex, dämpft "kippt bei jedem Lauf komplett in trocken
            # oder nass"-Verhalten.
            previous_humid_monthly = self.data_lod_manager.get_calculator_output(
                "weather.humidity", "humid_map_monthly", lod_level - 1)
            if previous_humid_monthly is not None:
                for m in range(6):
                    prev_m = previous_humid_monthly[m]
                    if prev_m.shape[0] != monthly_humid_maps[m].shape[0]:
                        prev_m = self._interpolate_2d_bicubic(prev_m, monthly_humid_maps[m].shape[0])
                    monthly_humid_maps[m] = 0.4 * prev_m + 0.6 * monthly_humid_maps[m]

            temp_map = np.mean(np.stack(monthly_temp_maps, axis=0), axis=0).astype(np.float32)
            wind_map = np.mean(np.stack(monthly_wind_maps, axis=0), axis=0).astype(np.float32)
            humid_map = np.mean(np.stack(monthly_humid_maps, axis=0), axis=0).astype(np.float32)
            precip_map = np.mean(np.stack(monthly_precip_maps, axis=0), axis=0).astype(np.float32)

            temp_map_layers = np.mean(np.stack(monthly_temp_layers, axis=0), axis=0).astype(np.float32)
            wind_map_layers = np.mean(np.stack(monthly_wind_layers, axis=0), axis=0).astype(np.float32)
            humid_map_layers = np.mean(np.stack(monthly_humid_layers, axis=0), axis=0).astype(np.float32)

            self.data_lod_manager.set_calculator_output(
                "weather.temperature", lod_level,
                {"temp_map": temp_map, "temp_map_monthly": monthly_temp_maps,
                 "shadowmap_monthly": monthly_shadowmaps,
                 "temp_map_layers": temp_map_layers, "temp_map_layers_monthly": monthly_temp_layers})
            self.data_lod_manager.set_calculator_output(
                "weather.wind", lod_level,
                {"wind_map": wind_map, "wind_map_monthly": monthly_wind_maps,
                 "wind_map_layers": wind_map_layers, "wind_map_layers_monthly": monthly_wind_layers})
            self.data_lod_manager.set_calculator_output(
                "weather.humidity", lod_level,
                {"humid_map": humid_map, "humid_map_monthly": monthly_humid_maps,
                 "humid_map_layers": humid_map_layers, "humid_map_layers_monthly": monthly_humid_layers})
            self.data_lod_manager.set_calculator_output(
                "weather.precipitation", lod_level,
                {"precip_map": precip_map, "precip_map_monthly": monthly_precip_maps})

        except Exception as e:
            self.logger.warning(
                f"weather.temperature: gekoppelte 3-Schicht-Simulation fehlgeschlagen ({e}) - "
                f"Fallback auf Einzelschicht-Temperatur, Wind/Feuchte/Niederschlag "
                f"nutzen ihren eigenen Einzelschicht-Fallback")
            monthly_temp_maps = [
                self._calculate_temperature_field(
                    heightmap, monthly_shadowmaps[m], month_params_list[m], target_size)
                for m in range(6)
            ]
            temp_map = np.mean(np.stack(monthly_temp_maps, axis=0), axis=0).astype(np.float32)
            self.data_lod_manager.set_calculator_output(
                "weather.temperature", lod_level,
                {"temp_map": temp_map, "temp_map_monthly": monthly_temp_maps,
                 "shadowmap_monthly": monthly_shadowmaps})

    def _calc_wind(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.wind' (#12) - dünner Pass-Through. Der
        eigentliche Wind wird bereits vom gekoppelten 3-Schicht-Loop in
        _calc_temperature berechnet und unter dieser calculator_id gespeichert
        (siehe dortiger Docstring, [[project-3layer-wind-cfd]] für die DAG-
        Begründung). Ist der Wert schon da (Normalfall - weather.temperature
        hat keine Abhängigkeit zu diesem Knoten, läuft also garantiert
        zuerst), ist dieser Aufruf ein No-Op, der nur zügig zurückkehrt (damit
        der Orchestrator mark_completed('weather.wind') auslöst, worauf eigene
        Downstream-Konsumenten warten).

        Fallback: ist der Wert NICHT da (z.B. weil _calc_temperature's
        gekoppelter Loop fehlschlug), auf die alte Einzelschicht-CFD
        zurückfallen - etabliertes 3-stufiges Fallback-Muster dieser Datei.
        """
        existing = self.data_lod_manager.get_calculator_output(calculator_id, "wind_map", lod_level)
        if existing is not None:
            return

        self.logger.warning("weather.wind: kein Ergebnis vom gekoppelten Loop gefunden - Einzelschicht-Fallback")
        cfd_iterations = self._get_cfd_iterations(lod_level)
        self._update_progress("Wind Field", 30, f"Fallback CFD simulation with {cfd_iterations} iterations...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        shadowmap_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "shadowmap_monthly", lod_level)
        if temp_map_monthly is None:
            raise ValueError(f"weather.wind: temp_map_monthly für LOD {lod_level} nicht verfügbar")

        monthly_wind_maps = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            month_temp_map = temp_map_monthly[month_index]
            month_shadowmap = shadowmap_monthly[month_index] if shadowmap_monthly else shadowmap
            monthly_wind_maps.append(self._simulate_wind_field_cfd(
                heightmap, month_temp_map, month_shadowmap, month_params, target_size, cfd_iterations))

        wind_map = np.mean(np.stack(monthly_wind_maps, axis=0), axis=0).astype(np.float32)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"wind_map": wind_map, "wind_map_monthly": monthly_wind_maps})

    def _calc_humidity(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.humidity' (#13) - dünner Pass-Through, analog
        zu _calc_wind. Die eigentliche Feuchte (inkl. Mehrgenerationen-Puffer-
        Blending) wird bereits vom gekoppelten Loop in _calc_temperature
        berechnet und gespeichert.

        Fallback: alte Einzelschicht-Logik inkl. eigenem Mehrgenerationen-
        Puffer-Blend, falls _calc_temperature's Loop fehlschlug.
        """
        existing = self.data_lod_manager.get_calculator_output(calculator_id, "humid_map", lod_level)
        if existing is not None:
            return

        self.logger.warning("weather.humidity: kein Ergebnis vom gekoppelten Loop gefunden - Einzelschicht-Fallback")
        self._update_progress("Humidity", 60, "Fallback atmospheric moisture transport...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        wind_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_monthly", lod_level)
        if temp_map_monthly is None or wind_map_monthly is None:
            raise ValueError(f"weather.humidity: fehlende Inputs für LOD {lod_level}")

        hardness_map = self.data_lod_manager.get_calculator_output(
            "geology.hardness", "hardness_map", lod_level)
        if hardness_map is not None and hardness_map.shape[0] != heightmap.shape[0]:
            hardness_map = self._interpolate_2d_bicubic(hardness_map, heightmap.shape[0])

        previous_monthly = self.data_lod_manager.get_calculator_output(
            calculator_id, "humid_map_monthly", lod_level - 1)

        monthly_humid_maps = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            humid_map_m = self._calculate_atmospheric_moisture(
                heightmap, temp_map_monthly[month_index], wind_map_monthly[month_index],
                month_params, hardness_map=hardness_map)

            if previous_monthly is not None:
                prev_m = previous_monthly[month_index]
                if prev_m.shape[0] != humid_map_m.shape[0]:
                    prev_m = self._interpolate_2d_bicubic(prev_m, humid_map_m.shape[0])
                humid_map_m = 0.4 * prev_m + 0.6 * humid_map_m

            monthly_humid_maps.append(humid_map_m)

        humid_map = np.mean(np.stack(monthly_humid_maps, axis=0), axis=0).astype(np.float32)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"humid_map": humid_map, "humid_map_monthly": monthly_humid_maps})

    def _calc_precipitation(self, calculator_id: str, lod_level: int) -> None:
        """
        Calculator-Node 'weather.precipitation' (#14) - dünner Pass-Through,
        analog zu _calc_wind/_calc_humidity. Der eigentliche Niederschlag
        (akkumulierte Kondensation aus dem gekoppelten Loop + orographischer
        Zusatzbeitrag) wird bereits von _calc_temperature berechnet und
        gespeichert.

        Fallback: alte Einzelschicht-Logik (Luv/Lee + Magnus-Kondensation auf
        dem finalen Einzelschicht-Snapshot), falls _calc_temperature's Loop
        fehlschlug.
        """
        existing = self.data_lod_manager.get_calculator_output(calculator_id, "precip_map", lod_level)
        if existing is not None:
            return

        self.logger.warning(
            "weather.precipitation: kein Ergebnis vom gekoppelten Loop gefunden - Einzelschicht-Fallback")
        self._update_progress("Precipitation", 80, "Fallback orographic precipitation...")
        heightmap, shadowmap, target_size = self._get_prepared_terrain_inputs(lod_level)
        temp_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.temperature", "temp_map_monthly", lod_level)
        wind_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.wind", "wind_map_monthly", lod_level)
        humid_map_monthly = self.data_lod_manager.get_calculator_output(
            "weather.humidity", "humid_map_monthly", lod_level)
        if temp_map_monthly is None or wind_map_monthly is None or humid_map_monthly is None:
            raise ValueError(f"weather.precipitation: fehlende Inputs für LOD {lod_level}")

        monthly_precip_maps = []
        for month_index in range(6):
            month_params = self._generate_seasonal_parameters(self._current_parameters, month_index)
            monthly_precip_maps.append(self._calculate_precipitation_system(
                humid_map_monthly[month_index], temp_map_monthly[month_index],
                wind_map_monthly[month_index], heightmap, month_params))

        precip_map = np.mean(np.stack(monthly_precip_maps, axis=0), axis=0).astype(np.float32)
        self.data_lod_manager.set_calculator_output(
            calculator_id, lod_level, {"precip_map": precip_map, "precip_map_monthly": monthly_precip_maps})

    def _semi_lagrangian_advect(self, field: np.ndarray, u: np.ndarray, v: np.ndarray,
                                y_idx: np.ndarray, x_idx: np.ndarray, dt_scale: float = 0.1) -> np.ndarray:
        """
        Allgemeine Semi-Lagrange-Rückwärts-Advektion (siehe [[project-3layer-wind-cfd]],
        Referenz-Technik aus niels747/2D-Weather-Sandbox): jede Zelle sampelt ihren
        neuen Wert an position - geschwindigkeit*dt_scale, statt Vorwärts-Differenzen -
        numerisch stabil, kein CFL-Problem, gilt identisch für Geschwindigkeit,
        potentielle Temperatur und Feuchte. y_idx/x_idx sind vom Aufrufer
        vorberechnete np.mgrid-Koordinaten (wird pro Zeitschritt für alle 3
        Schichten/4 Felder wiederverwendet statt pro Aufruf neu gebaut).
        dt_scale folgt derselben Stabilitäts-Skalierung wie das bereits bestehende
        _transport_moisture_simple (dt*0.1).
        """
        height, width = field.shape[:2]
        source_x = np.clip(x_idx - u * dt_scale, 0, width - 1)
        source_y = np.clip(y_idx - v * dt_scale, 0, height - 1)
        return map_coordinates(field, [source_y, source_x], order=1, mode='nearest').astype(field.dtype)

    def _run_coupled_atmosphere_simulation(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                          month_params: Dict[str, Any], target_size: int,
                                          n_steps: int,
                                          roughness_damping: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Gekoppelte 3-Schicht-Atmosphären-Simulation (Boden/Mittel/Hoch, siehe
        AtmosphereLayers) - ersetzt das bisherige "Temp einmal -> Wind einmal ->
        Feuchte einmal"-Muster für EINEN Monat durch einen echten Pro-Zeitschritt-
        Loop, in dem Wind/Temperatur/Feuchte sich gegenseitig beeinflussen. Wird
        von _calc_temperature aufgerufen (siehe dortiger Docstring für die DAG-
        Begründung, warum der Loop dort statt in _calc_wind gehostet wird) -
        [[project-3layer-wind-cfd]].

        Pro Zeitschritt, pro Schicht:
          1. Semi-Lagrange-Advektion von u,v,theta (potentielle Temp),q
          2. Druckgradient (gemeinsames synoptisches Druckfeld je Monat, Terrain-
             Kopplung nur auf GROUND/gedämpft MID) + additive Terrain-Ablenkung
             UND multiplikativer Grat-/Canyon-Speedup (beide nur GROUND/gedämpft
             MID, siehe [[project-wind-ridge-speedup]]) + Rauigkeits-Dämpfung aus
             Biome (best-effort, siehe [[project-wind-roughness]]) + thermische
             Konvektion (Schatten-Term nur GROUND)
          3. Latentwärme: Kondensation aus Übersättigung (Magnus-Formel, identische
             Skala wie _calculate_precipitation_cpu/_calculate_atmospheric_moisture_cpu)
             wärmt, Verdunstung (nur GROUND) kühlt
        Danach, pro Zeitschritt EINMAL über alle Schichten:
          4. Vertikaler Austausch (Thermik-getrieben, symmetrische Relaxation
             zwischen Nachbarschichten - erhält die Schicht-Summe jeder Größe)
          5. Diffusion + 6-Richtungs-Kontinuitäts-Korrektur (horizontale Divergenz
             + vertikaler Fluss-Term aus Schritt 4)

        roughness_damping (optional, (H,W), Wertebereich [0,1]): best-effort
        Bodenrauigkeits-Dämpfung aus der Biome-Klassifikation, von
        _calc_temperature einmal pro LOD-Runde vorab per
        _get_roughness_damping() geholt (statisch über alle 6 Monate, daher
        hier als Parameter statt hier selbst erneut abgefragt). None (Default,
        z.B. wenn Biome in dieser Session nie angefragt wurde) reproduziert
        exakt das alte Verhalten ohne Rauigkeits-Term.

        Rückgabe: dict mit 'wind_layers' (3,H,W,2), 'temp_layers' (3,H,W, reale
        Temperatur), 'humid_layers' (3,H,W), 'precip_map' (H,W).
        """
        from gui.config.value_default import WEATHER  # TURBULENCE_STRENGTH-Default weiter unten

        height, width = heightmap.shape
        L = AtmosphereLayers
        altitude_cooling_rate = month_params['altitude_cooling'] / 1000.0  # °C/m (Parameter ist °C/km)

        # Latentwärme-Kopplungsstärke: °C pro gH2O/m³ Kondensation/Verdunstung.
        # Startwert (Bereich 0.05-0.15 laut Plan), empirisch, dokumentiert wie
        # frühere Kalibrierungen dieser Session - noch nicht am echten Nutzer-
        # Feedback nachjustiert.
        LATENT_HEAT_COEFFICIENT = 0.1
        # Vertikaler Austausch: Bruchteil pro Zeitschritt, der zwischen
        # Nachbarschichten geblendet wird, proportional zur Thermik-Stärke -
        # geklemmt, damit ein einzelner Zeitschritt nie mehr als 30% einer
        # Schicht "leert" (Stabilität, analog zur Diffusions-Rate anderswo in
        # dieser Datei).
        MAX_VERTICAL_EXCHANGE_FRACTION = 0.3
        VERTICAL_EXCHANGE_COEFF = 0.02
        # Wie stark sich der druckgetriebene Wind-Term pro Zeitschritt Richtung
        # Druck-Gleichgewicht bewegt, statt (wie im alten Einzelschicht-Code) ihn
        # jede Iteration hart zu überschreiben - mit echter Advektion (Schritt 1)
        # ist u/v jetzt eine prognostische, transportierte Größe, ein hartes
        # Überschreiben würde die advehierte Struktur jedes Mal zerstören.
        PRESSURE_RELAX_RATE = 0.3
        # Wind-Geschwindigkeits-Multiplikatoren für die Initialbedingung je
        # Schicht (Wind nimmt mit Höhe zu, geringere Bodenreibung) - nur
        # Startwert, der Loop entwickelt die tatsächliche Struktur.
        LAYER_WIND_SEED_MULT = (1.0, 1.3, 1.6)
        LAYER_HUMID_SEED_MULT = (1.0, 0.5, 0.2)
        # Terrain-Kopplung/-Ablenkung (die diese Session zuvor für die
        # Einzelschicht-CFD hinzugefügten Terme) wirkt gedämpft mit der Höhe -
        # voll auf GROUND, exponentiell gedämpft auf MID, gar nicht auf HIGH
        # (siehe [[project-3layer-wind-cfd]]: der primäre Terrain-
        # Beschleunigungseffekt entsteht jetzt aus der Kontinuitätskorrektur
        # über die terrain-folgende Schichtgeometrie selbst).
        TERRAIN_TERM_SCALE = (1.0, float(np.exp(-L.REF_ALTITUDE_AGL[L.MID] / 1000.0)), 0.0)
        # Vorticity Confinement (Fedkiw/Stam, siehe _apply_vorticity_confinement)
        # - Turbulenz entsteht überwiegend durch Bodenreibung/Oberflächen-Rauheit,
        # deshalb stärker auf GROUND, schwächer auf MID/HIGH (aber nicht 0 wie bei
        # den Terrain-Termen - auch die freie Atmosphäre zeigt etwas Verwirbelung).
        VORTICITY_LAYER_SCALE = (1.0, 0.6, 0.3)
        # Rauigkeits-Dämpfung: wie die Terrain-Terme voll auf GROUND, gedämpft
        # auf MID, keine auf HIGH (Bodenreibung wirkt per Definition nur nahe
        # der Oberfläche) - siehe [[project-wind-roughness]].
        ROUGHNESS_LAYER_SCALE = (1.0, 0.3, 0.0)

        y_idx, x_idx = np.mgrid[0:height, 0:width].astype(np.float64)
        slopemap = self._calculate_slopes_vectorized(heightmap)
        curvature_norm = self._calculate_curvature_normalized(heightmap)

        # Rand-Verstärkung für Vorticity Confinement (Nutzer-Wunsch: "an der
        # Kartengrenze mehr Varianz" - der Kartenrand ist, wo die synoptische
        # Randbedingung einströmt, dort real am wenigsten "ausgeglichen"). Faktor
        # 1.0 in der Kartenmitte, bis 2.5x direkt am Rand, exponentiell abklingend
        # über ~8% der Kartengröße - Startwerte, empirisch nachjustierbar.
        edge_dist = np.minimum.reduce([x_idx, width - 1 - x_idx, y_idx, height - 1 - y_idx])
        vorticity_edge_boost = (1.0 + 1.5 * np.exp(
            -edge_dist / max(0.08 * min(height, width), 1e-6))).astype(np.float32)

        # Gemeinsames synoptisches Druckfeld (eine Größenordnung, alle Schichten
        # spüren dieselbe großräumige Richtung + Monats-Rauschen - nur die
        # Terrain-Kopplung unten unterscheidet sich pro Schicht).
        wind_direction_deg = month_params.get('prevailing_wind_direction', 0.0)
        pressure_field = self._build_directional_pressure_field(height, width, wind_direction_deg)
        pressure_field = pressure_field + self._generate_pressure_noise(
            (height, width), month_params.get('month_index', 0)) * 0.15

        height_range = heightmap.max() - heightmap.min()
        height_normalized = (heightmap - heightmap.min()) / height_range if height_range > 1e-6 \
            else np.zeros_like(heightmap, dtype=np.float32)
        terrain_pressure_layers = [
            height_normalized * month_params['terrain_factor'] * 0.2 * TERRAIN_TERM_SCALE[i]
            for i in range(L.COUNT)
        ]

        # --- Initialbedingung ---
        surface_temp = self._calculate_temperature_field(heightmap, shadowmap, month_params, target_size)
        # Konstante potentielle Temperatur über alle Schichten als Start (gut
        # durchmischte Atmosphäre) - reale Temperatur pro Schicht ergibt sich
        # daraus automatisch kälter mit Höhe (siehe Klassen-Docstring von
        # AtmosphereLayers).
        theta = [surface_temp.astype(np.float32).copy() for _ in range(L.COUNT)]

        base_wind = self._simulate_wind_field_simple(heightmap, month_params)
        u = [(base_wind[:, :, 0] * LAYER_WIND_SEED_MULT[i]).astype(np.float32) for i in range(L.COUNT)]
        v = [(base_wind[:, :, 1] * LAYER_WIND_SEED_MULT[i]).astype(np.float32) for i in range(L.COUNT)]

        ground_temp_c = np.clip(theta[L.GROUND] - altitude_cooling_rate * L.REF_ALTITUDE_AGL[L.GROUND], -50, 60)
        sat_vp0 = 6.112 * np.exp(17.67 * ground_temp_c / (ground_temp_c + 243.5))
        ground_wind_speed0 = np.sqrt(u[L.GROUND] ** 2 + v[L.GROUND] ** 2)
        wind_factor0 = np.minimum(2.0, ground_wind_speed0 / 5.0)
        evap_rate0 = 0.5 * (sat_vp0 / 100.0) * (1.0 + wind_factor0)  # soil_moisture=50/100=0.5
        q = [(evap_rate0 * 140.0 * LAYER_HUMID_SEED_MULT[i]).astype(np.float32) for i in range(L.COUNT)]

        precip_accum = np.zeros((height, width), dtype=np.float32)

        for step in range(n_steps):
            if step % max(1, n_steps // 5) == 0:
                progress = 20 + (step / max(1, n_steps)) * 40
                self._update_progress("Atmosphere", int(progress), f"Coupled step {step + 1}/{n_steps}")

            t_real = [None] * L.COUNT
            for i in range(L.COUNT):
                # 1. Semi-Lagrange-Advektion (alle 4 Felder mit dem VOR der
                # Advektion gültigen Geschwindigkeitsfeld rückwärts-gesampelt -
                # Standard-Semi-Lagrange-Konsistenz).
                u_old, v_old = u[i], v[i]
                u[i] = self._semi_lagrangian_advect(u_old, u_old, v_old, y_idx, x_idx)
                v[i] = self._semi_lagrangian_advect(v_old, u_old, v_old, y_idx, x_idx)
                theta[i] = self._semi_lagrangian_advect(theta[i], u_old, v_old, y_idx, x_idx)
                q[i] = self._semi_lagrangian_advect(q[i], u_old, v_old, y_idx, x_idx)

                # 2a. Druckgradient - Relaxation statt hartem Reset (siehe
                # PRESSURE_RELAX_RATE-Kommentar oben).
                pressure_iter = pressure_field - terrain_pressure_layers[i]
                grad_x = np.zeros((height, width), dtype=np.float32)
                grad_y = np.zeros((height, width), dtype=np.float32)
                grad_x[:, 1:-1] = (pressure_iter[:, 2:] - pressure_iter[:, :-2]) * 0.5
                grad_y[1:-1, :] = (pressure_iter[2:, :] - pressure_iter[:-2, :]) * 0.5
                u_target = -grad_x * month_params['wind_speed_factor'] * 10.0
                v_target = -grad_y * month_params['wind_speed_factor'] * 10.0
                u[i] += (u_target - u[i]) * PRESSURE_RELAX_RATE
                v[i] += (v_target - v[i]) * PRESSURE_RELAX_RATE

                # 2b. Terrain-Ablenkung (gedämpft mit Höhe, siehe TERRAIN_TERM_SCALE)
                # + Grat-/Canyon-Speedup (WindNinja-Terrain-Shape-Effekt,
                # siehe [[project-wind-ridge-speedup]]): zusätzlich zur reinen
                # Hang-Ablenkung beschleunigt Wind multiplikativ über
                # konvexen Graten und bremst in konkaven Tälern, gleiche
                # Höhendämpfung wie die Ablenkung selbst.
                terrain_term = month_params['terrain_factor'] * 0.5 * TERRAIN_TERM_SCALE[i]
                if terrain_term != 0.0:
                    u[i] += slopemap[:, :, 1] * terrain_term
                    v[i] -= slopemap[:, :, 0] * terrain_term
                    ridge_factor = 1.0 + _RIDGE_SPEEDUP_STRENGTH * (-curvature_norm) * \
                        month_params['terrain_factor'] * TERRAIN_TERM_SCALE[i]
                    u[i] *= ridge_factor
                    v[i] *= ridge_factor

                # 2c. Thermische Konvektion (Schatten-Term nur GROUND - nur die
                # Bodenschicht "sieht" die Sonneneinstrahlung direkt, siehe
                # Docstring Schritt 3 in der Klassen-Beschreibung oben).
                t_real[i] = theta[i] - altitude_cooling_rate * L.REF_ALTITUDE_AGL[i]
                temp_grad_x = np.zeros((height, width), dtype=np.float32)
                temp_grad_y = np.zeros((height, width), dtype=np.float32)
                temp_grad_x[:, 1:-1] = (t_real[i][:, 2:] - t_real[i][:, :-2]) * 0.5
                temp_grad_y[1:-1, :] = (t_real[i][2:, :] - t_real[i][:-2, :]) * 0.5
                convection_strength = (t_real[i] - np.mean(t_real[i])) * month_params['thermic_effect'] * 0.08
                u[i] += temp_grad_x * 0.05 + convection_strength
                if i == L.GROUND:
                    if len(shadowmap.shape) == 3:
                        shadow_avg = np.mean(shadowmap, axis=2)
                    else:
                        shadow_avg = shadowmap
                    shadow_effect = (shadow_avg - 0.5) * month_params['thermic_effect'] * 0.15
                    v[i] += temp_grad_y * 0.05 + shadow_effect
                else:
                    v[i] += temp_grad_y * 0.05

                # 2d. Rauigkeits-Dämpfung aus Biome (best-effort, siehe
                # [[project-wind-roughness]]) - läuft VOR der Verdunstungs-
                # Windgeschwindigkeit in Schritt 3, damit rauigkeitsbedingt
                # gebremster Bodenwind auch die Verdunstungs-Verstärkung
                # konsistent mitreduziert.
                if roughness_damping is not None and ROUGHNESS_LAYER_SCALE[i] > 0.0:
                    damping = roughness_damping * ROUGHNESS_LAYER_SCALE[i]
                    u[i] *= (1.0 - damping)
                    v[i] *= (1.0 - damping)

                # 3. Latentwärme - identische Magnus-Skala wie
                # _calculate_precipitation_cpu/_calculate_atmospheric_moisture_cpu
                # (siehe [[project-precip-humidity-calibration]]), KEIN zweites
                # Feuchte-Einheitensystem.
                t_clamped = np.clip(t_real[i], -40, 50)
                rho_max = 5.0 * np.exp(0.06 * t_clamped)
                oversaturation = np.maximum(0.0, q[i] / np.maximum(rho_max, 1e-6) - 1.0)
                condensation = np.minimum(oversaturation * rho_max * 0.6, q[i])
                theta[i] += condensation * LATENT_HEAT_COEFFICIENT
                q[i] = np.maximum(q[i] - condensation, 0.0)
                if i in (L.GROUND, L.MID):
                    precip_accum += condensation

                if i == L.GROUND:
                    ground_temp_c = np.clip(t_real[i], -50, 60)
                    sat_vp = 6.112 * np.exp(17.67 * ground_temp_c / (ground_temp_c + 243.5))
                    wind_speed = np.sqrt(u[i] ** 2 + v[i] ** 2)
                    wind_factor = np.minimum(2.0, wind_speed / 5.0)
                    evap_rate = 0.5 * (sat_vp / 100.0) * (1.0 + wind_factor)
                    # Über n_steps verteilt, damit die kumulierte Zufuhr über den
                    # ganzen Loop näherungsweise dieselbe Größenordnung erreicht
                    # wie der bisherige Einzelschuss-Faktor 140.0.
                    evap_source = evap_rate * 140.0 / max(n_steps, 1)
                    q[L.GROUND] += evap_source
                    theta[L.GROUND] -= evap_source * LATENT_HEAT_COEFFICIENT

            # 4. Vertikaler Austausch (Thermik-getrieben, symmetrische Relaxation
            # - erhält die Schicht-Summe von theta/q/u/v exakt, siehe
            # Methoden-Docstring).
            w_gm = np.maximum(0.0, t_real[L.GROUND] - t_real[L.MID]) * month_params['thermic_effect'] \
                * VERTICAL_EXCHANGE_COEFF
            w_mh = np.maximum(0.0, t_real[L.MID] - t_real[L.HIGH]) * month_params['thermic_effect'] \
                * VERTICAL_EXCHANGE_COEFF
            f_gm = np.clip(w_gm, 0.0, MAX_VERTICAL_EXCHANGE_FRACTION)
            f_mh = np.clip(w_mh, 0.0, MAX_VERTICAL_EXCHANGE_FRACTION)

            for field_list in (theta, q, u, v):
                lower, mid_, upper = field_list[L.GROUND], field_list[L.MID], field_list[L.HIGH]
                delta_gm = f_gm * (lower - mid_)
                delta_mh = f_mh * (mid_ - upper)
                field_list[L.GROUND] = lower - delta_gm
                field_list[L.MID] = mid_ + delta_gm - delta_mh
                field_list[L.HIGH] = upper + delta_mh

            vertical_flux_terms = (
                -w_gm / AtmosphereLayers.THICKNESS_M[L.GROUND],
                (w_gm - w_mh) / AtmosphereLayers.THICKNESS_M[L.MID],
                w_mh / AtmosphereLayers.THICKNESS_M[L.HIGH],
            )

            # 5. Diffusion + Vorticity Confinement + 6-Richtungs-Kontinuitätskorrektur
            # pro Schicht. Vorticity Confinement läuft NACH der Diffusion (die
            # Diffusion glättet zuerst, numerische Stabilität bleibt erhalten) und
            # VOR der Kontinuitätskorrektur (die das Ergebnis danach wieder auf
            # Massenerhaltung ausbalanciert) - würde die Reihenfolge vertauscht,
            # würde die Diffusion die injizierte Verwirbelung im selben Schritt
            # wieder wegbügeln.
            turbulence_strength = month_params.get(
                'turbulence_strength', WEATHER.TURBULENCE_STRENGTH["default"])
            for i in range(L.COUNT):
                wind_field_i = np.stack([u[i], v[i]], axis=-1).astype(np.float32)
                wind_field_i = self._apply_wind_diffusion(wind_field_i, 0.1)
                if turbulence_strength > 0.0:
                    vorticity_strength_i = (
                        turbulence_strength * VORTICITY_LAYER_SCALE[i] * vorticity_edge_boost)
                    self._apply_vorticity_confinement(wind_field_i, vorticity_strength_i)
                self._apply_continuity_correction(wind_field_i, vertical_flux_term=vertical_flux_terms[i])
                u[i], v[i] = wind_field_i[:, :, 0], wind_field_i[:, :, 1]

        # --- Ergebnis zusammensetzen ---
        temp_layers = np.stack(
            [theta[i] - altitude_cooling_rate * L.REF_ALTITUDE_AGL[i] for i in range(L.COUNT)], axis=0
        ).astype(np.float32)
        wind_layers = np.stack(
            [np.stack([u[i], v[i]], axis=-1) for i in range(L.COUNT)], axis=0
        ).astype(np.float32)
        humid_layers = np.stack(q, axis=0).astype(np.float32)

        # Orographischer Zusatzbeitrag (Luv-/Lee, identische Formel wie
        # _calculate_precipitation_cpu) auf Basis der finalen GROUND-Schicht -
        # separat von precip_accum gehalten, um Magnus-Kondensation nicht
        # doppelt zu zählen (precip_accum hat die Kondensation bereits über
        # den ganzen Loop akkumuliert).
        wind_speed_ground = np.sqrt(wind_layers[L.GROUND, :, :, 0] ** 2 + wind_layers[L.GROUND, :, :, 1] ** 2)
        wind_norm_x = np.where(wind_speed_ground > 0.1, wind_layers[L.GROUND, :, :, 0] / wind_speed_ground, 0)
        wind_norm_y = np.where(wind_speed_ground > 0.1, wind_layers[L.GROUND, :, :, 1] / wind_speed_ground, 0)
        wind_slope_alignment = wind_norm_x * slopemap[:, :, 0] + wind_norm_y * slopemap[:, :, 1]
        orographic_factor = np.maximum(0, wind_slope_alignment) * wind_speed_ground * 0.3
        oro_precip = humid_layers[L.GROUND] * orographic_factor * 0.05

        precip_map = np.clip(precip_accum + oro_precip, 0.0, 500.0).astype(np.float32)

        return {
            'wind_layers': wind_layers,
            'temp_layers': temp_layers,
            'humid_layers': humid_layers,
            'precip_map': precip_map,
        }

    def _validate_inputs(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                        parameters: Dict[str, Any], lod_level: int):
        """
        Input-Data-Validation Pipeline für robuste Weather-Generation

        Prüft Physical-Range-Validation, Cross-Generator-Consistency und LOD-Compatibility.
        """
        # Shape-Consistency-Checks
        if heightmap_combined.shape != shadowmap.shape[:2]:
            raise ValueError(f"Shape mismatch: heightmap {heightmap_combined.shape} vs shadowmap {shadowmap.shape[:2]}")

        # Physical-Range-Validation
        if np.any(np.isnan(heightmap_combined)) or np.any(np.isinf(heightmap_combined)):
            raise ValueError("Invalid values in heightmap_combined")

        if np.any(np.isnan(shadowmap)) or np.any(np.isinf(shadowmap)):
            raise ValueError("Invalid values in shadowmap")

        # Parameter-Range-Validation
        required_params = ['air_temp_entry', 'solar_power', 'altitude_cooling',
                          'thermic_effect', 'wind_speed_factor', 'terrain_factor']

        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")

        # Physical-Plausibility-Checks
        if not (-50 <= parameters['air_temp_entry'] <= 60):
            raise ValueError(f"air_temp_entry {parameters['air_temp_entry']} outside physical range [-50, 60]°C")

        if not (0 <= parameters['solar_power'] <= 50):
            raise ValueError(f"solar_power {parameters['solar_power']} outside valid range [0, 50]°C")

        # LOD-Level-Validation
        if not (1 <= lod_level <= 10):
            raise ValueError(f"Invalid lod_level {lod_level}, must be in range [1, 10]")

    def _get_lod_size(self, lod_level: int, original_size: int) -> int:
        """
        Bestimmt Target-Size basierend auf numerischem LOD-Level

        LOD-System mit progressiver Grid-Verdopplung bis original_size erreicht
        """
        base_size = 32
        max_lod_before_original = 6

        if lod_level <= max_lod_before_original:
            # Verdopplung pro LOD-Level: 32 -> 64 -> 128 -> 256 -> 512 -> 1024
            # Bei Nicht-Zweierpotenz-Zielgrößen (z.B. 96) wird original_size (die
            # tatsächlich vom Terrain-Generator gelieferte, bereits korrekt geklemmte
            # Heightmap-Größe) schon vor Erreichen von max_lod_before_original
            # überschritten - ohne min() würde hier auf eine größere Auflösung
            # hochinterpoliert als das Terrain überhaupt hat.
            return min(base_size * (2 ** (lod_level - 1)), original_size)
        else:
            # Höhere LODs verwenden original_size
            return original_size

    def _get_cfd_iterations(self, lod_level: int) -> int:
        """
        Bestimmt CFD-Iterations basierend auf LOD-Level für Progressive Enhancement

        Steigende CFD-Komplexität: 3->5->7->10->15->20->25 Iterationen
        """
        iteration_mapping = {
            1: 3,   # LOD 32x32: 3 Iterationen für schnelle Preview
            2: 5,   # LOD 64x64: 5 Iterationen mit Enhanced-Effects
            3: 7,   # LOD 128x128: 7 Iterationen mit Detailed-Orographics
            4: 10,  # LOD 256x256: 10 Iterationen mit High-Quality-Physics
            5: 15,  # LOD 512x512: 15 Iterationen mit Premium-Simulation
            6: 20,  # LOD 1024x1024: 20 Iterationen mit Maximum-Quality
        }

        return iteration_mapping.get(lod_level, 25)  # 25+ Iterationen für höchste LODs

    def _get_atmosphere_loop_steps(self, lod_level: int) -> int:
        """
        Bestimmt die Zeitschritt-Anzahl für den gekoppelten 3-Schicht-Atmosphäre-
        Loop (_run_coupled_atmosphere_simulation, [[project-3layer-wind-cfd]]) -
        EIGENSTÄNDIG von _get_cfd_iterations() oben, das weiterhin nur die
        Einzelschicht-Fallback-Pfade treibt (_calc_wind/_calc_humidity/
        _calc_precipitation, falls der gekoppelte Loop fehlschlägt) und für einen
        anderen, bereits kalibrierten Algorithmus gilt - nicht automatisch mit
        hochziehen.

        Grob verdoppelt gegenüber der alten Tabelle: mit den alten, niedrigen
        Werten blieb zu wenig "Simulationszeit", damit sich die tatsächlich
        monats-variierenden Treiber (Windrichtungs-Drehung, Monats-Rauschen im
        Druckfeld/in der Bodentemperatur, monatliche Sonnenstand-Shadowmap) gegen
        das terrain-dominierte Gleichgewicht durchsetzen, bevor der Loop endet -
        empirisch bestätigt über eine paarweise räumliche Korrelation von
        humid_map_monthly von 0.72-0.96 (praktisch identisches Muster über alle
        6 Monate, nur das Niveau verschob sich). Startwerte, nutzerseitig als
        Performance-Kosten akzeptiert (vor dem geplanten GPU-Port).
        """
        step_mapping = {
            1: 8,   # LOD 32x32
            2: 12,  # LOD 64x64
            3: 18,  # LOD 128x128
            4: 25,  # LOD 256x256
            5: 35,  # LOD 512x512
            6: 50,  # LOD 1024x1024
        }
        return step_mapping.get(lod_level, 60)

    def _prepare_input_data(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                           target_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpoliert Input-Data auf Target-Size mit bicubic Interpolation

        Erhält Pattern-Preservation bei Auflösungs-Verdopplung
        """
        # Heightmap interpolieren falls nötig
        if heightmap_combined.shape[0] != target_size:
            heightmap = self._interpolate_2d_bicubic(heightmap_combined, target_size)
        else:
            heightmap = heightmap_combined.copy()

        # Shadowmap immer interpolieren (kann von anderem LOD kommen)
        if shadowmap.shape[0] != target_size:
            # Shadowmap ist 3D (H,W,angles) - jeden Kanal separat interpolieren
            if len(shadowmap.shape) == 3:
                interpolated_shadow = np.zeros((target_size, target_size, shadowmap.shape[2]),
                                             dtype=np.float32)
                for angle_idx in range(shadowmap.shape[2]):
                    interpolated_shadow[:, :, angle_idx] = self._interpolate_2d_bicubic(
                        shadowmap[:, :, angle_idx], target_size)
                shadowmap_interp = interpolated_shadow
            else:
                shadowmap_interp = self._interpolate_2d_bicubic(shadowmap, target_size)
        else:
            shadowmap_interp = shadowmap.copy()

        return heightmap, shadowmap_interp

    def _calculate_temperature_field(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                   parameters: Dict[str, Any], target_size: int) -> np.ndarray:
        """
        Temperaturfeld-Berechnung mit GPU-Shader-Integration und 3-stufigem Fallback

        Integriert Altitude-Cooling, Solar-Heating, Latitude-Gradient und Noise-Variation
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'temperature_calculation',
                    'input_data': {
                        'heightmap': heightmap,
                        'shadowmap': shadowmap
                    },
                    'parameters': {
                        'air_temp_entry': parameters['air_temp_entry'],
                        'solar_power': parameters['solar_power'],
                        'altitude_cooling': parameters['altitude_cooling'],
                        'map_seed': self.map_seed
                    },
                    'lod_level': target_size
                }

                result = self.shader_manager.request_temperature_calculation(shader_request)

                if result.success:
                    self.logger.debug("Temperature calculation completed on GPU")
                    return result.temperature_field
                else:
                    self.logger.warning(f"GPU temperature calculation failed: {result.error}")

        except Exception as e:
            self.logger.warning(f"GPU shader request failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._calculate_temperature_cpu_optimized(heightmap, shadowmap, parameters, target_size)
        except Exception as e:
            self.logger.error(f"CPU temperature calculation failed: {e}")

        # Simple-Fallback (Minimal)
        return self._calculate_temperature_simple(heightmap, parameters)

    def _calculate_temperature_cpu_optimized(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                           parameters: Dict[str, Any], target_size: int) -> np.ndarray:
        """
        CPU-optimierte Temperatur-Berechnung mit vectorized NumPy-Operations
        """
        # Basis-Temperatur
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Altitude-Cooling (vectorized)
        altitude_cooling_rate = parameters['altitude_cooling'] / 1000.0  # °C pro Meter (Parameter ist °C/km)
        temp_map -= heightmap * altitude_cooling_rate

        # Solar-Heating (vectorized)
        if len(shadowmap.shape) == 3:
            # Gewichtete Kombination aller Shadow-Angles
            shadow_weighted = np.mean(shadowmap, axis=2)
        else:
            shadow_weighted = shadowmap

        # Shadow-Map: 0 (Schatten) bis 1 (volle Sonne)
        solar_effect = (shadow_weighted - 0.5) * parameters['solar_power']
        temp_map += solar_effect

        # Latitude-Gradient (vectorized)
        height, width = heightmap.shape
        y_coords = np.arange(height).reshape(-1, 1)
        latitude_effect = (y_coords / (height - 1)) * 5.0  # 5°C Nord-Süd-Gradient
        temp_map += latitude_effect

        # Atmospheric Noise-Variation
        noise_variation = self._generate_atmospheric_noise(
            heightmap.shape, target_size, parameters.get('month_index', 0))
        temp_map += noise_variation

        return temp_map

    def _calculate_temperature_simple(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Simple-Fallback: Basic Temperature ohne komplexe Effekte
        """
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Nur Altitude-Cooling
        altitude_cooling_rate = parameters['altitude_cooling'] / 1000.0  # Parameter ist °C/km
        temp_map -= heightmap * altitude_cooling_rate

        return temp_map

    def _simulate_wind_field_cfd(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                shadowmap: np.ndarray, parameters: Dict[str, Any],
                                target_size: int, cfd_iterations: int) -> np.ndarray:
        """
        CFD-basierte Wind-Simulation mit Navier-Stokes-Gleichungen und GPU-Acceleration

        Implementiert vollständige CFD-Pipeline mit Pressure-Gradients, Terrain-Deflection
        und Thermal-Convection über multiple Iterationen
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'wind_field_cfd',
                    'input_data': {
                        'heightmap': heightmap,
                        'temp_map': temp_map,
                        'shadowmap': shadowmap
                    },
                    'parameters': {
                        'wind_speed_factor': parameters['wind_speed_factor'],
                        'terrain_factor': parameters['terrain_factor'],
                        'thermic_effect': parameters['thermic_effect'],
                        'prevailing_wind_direction': parameters.get('prevailing_wind_direction', 0.0),
                        'month_index': parameters.get('month_index', 0),
                        'cfd_iterations': cfd_iterations,
                        'map_seed': self.map_seed
                    },
                    'lod_level': target_size
                }

                result = self.shader_manager.request_wind_field_cfd(shader_request)

                if result.success:
                    self.logger.debug(f"Wind CFD completed on GPU with {cfd_iterations} iterations")
                    return result.wind_field
                else:
                    self.logger.warning(f"GPU wind CFD failed: {result.error}")

        except Exception as e:
            self.logger.warning(f"GPU wind CFD request failed: {e}")

        # CPU-Fallback (Gut)
        try:
            return self._simulate_wind_field_cpu_cfd(heightmap, temp_map, shadowmap, parameters,
                                                   cfd_iterations)
        except Exception as e:
            self.logger.error(f"CPU wind CFD failed: {e}")

        # Simple-Fallback (Minimal)
        return self._simulate_wind_field_simple(heightmap, parameters)

    def _build_directional_pressure_field(self, height: int, width: int,
                                           wind_direction_deg: float) -> np.ndarray:
        """
        Funktionsweise: Lineares Druckgefälle entlang einer beliebigen
        Windrichtung (Grad, math. Konvention: 0°=Wind Richtung +x/Ost,
        90°=+y) - ersetzt das früher hartcodierte West-Ost-Gefälle. Bei
        wind_direction_deg=0 identisch zur alten Formel (Projektion s
        entspricht dann x_coords), da 'prevailing_wind_direction' den
        Windursprung in Ost-Konvention beschreibt.
        Aufgabe: Treibt die Richtung, aus der der vorherrschende Wind weht,
        in die initiale Druckfeld-Konstruktion der CFD-Simulation ein - lokale
        Abweichung entsteht weiterhin über die bestehende Terrain-Ablenkung.
        """
        theta = np.radians(wind_direction_deg)
        dx, dy = np.cos(theta), np.sin(theta)
        y_idx, x_idx = np.mgrid[0:height, 0:width]
        s = x_idx * dx + y_idx * dy
        s_min, s_max = s.min(), s.max()
        s_range = s_max - s_min
        s_norm = (s - s_min) / s_range if s_range > 1e-9 else np.zeros_like(s, dtype=np.float32)
        return (1.0 - s_norm * 0.3).astype(np.float32)

    def _simulate_wind_field_cpu_cfd(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                   shadowmap: np.ndarray, parameters: Dict[str, Any],
                                   cfd_iterations: int) -> np.ndarray:
        """
        CPU-optimierte CFD-Simulation mit NumPy-Vectorization

        Implementiert vereinfachte Navier-Stokes mit Advection, Pressure-Gradients und Diffusion
        """
        height, width = heightmap.shape

        # Initialisierung
        wind_field = np.zeros((height, width, 2), dtype=np.float32)

        # Slopemap aus Heightmap berechnen (vectorized)
        slopemap = self._calculate_slopes_vectorized(heightmap)
        # Grat-/Canyon-Speedup-Term (siehe [[project-wind-ridge-speedup]]) -
        # nur der Speedup, keine Rauigkeits-Dämpfung hier: dieser Pfad ist
        # der seltene Einzelschicht-Fallback (nur bei Exception im primären
        # 3-Schicht-Loop), der schon bisher bewusst schlanker gehalten wird
        # (keine Advektion/Vorticity/Schichten) - Rauigkeit bräuchte
        # zusätzlich lod_level/data_lod_manager-Zugriff, den diese Methode
        # aktuell nicht entgegennimmt.
        curvature_norm = self._calculate_curvature_normalized(heightmap)

        # Initiales Druckfeld entlang der vorherrschenden Windrichtung (mit Noise)
        wind_direction_deg = parameters.get('prevailing_wind_direction', 0.0)
        pressure_field = self._build_directional_pressure_field(height, width, wind_direction_deg)

        # Noise-Modulation
        pressure_noise = self._generate_pressure_noise((height, width), parameters.get('month_index', 0))
        pressure_field += pressure_noise * 0.15

        # Druck-Terrain-Kopplung: hohe Punkte senken lokal den effektiven
        # Druck (grobe Orographie-Näherung - Wind beschleunigt an/über
        # Erhebungen, sammelt sich in Tälern). Anders als die bestehende
        # additive Terrain-Ablenkung weiter unten (die nur EINMALIG pro
        # Iteration auf wind_field draufaddiert wird, NACH der Gradienten-
        # Berechnung) fließt dieser Term VOR der Gradientenberechnung ins
        # Druckfeld selbst ein - dadurch bleibt die terrain-geprägte Struktur
        # über die Diffusion/Kontinuitäts-Korrektur der gesamten
        # Iterationskette hinweg erhalten, statt nur ein einmaliger Nudge auf
        # ein sonst uniformes Feld zu sein. Koeffizient 0.2 ist ein erster
        # Richtwert (kleiner als der bestehende additive Term mit 0.5, da
        # dieser Effekt jetzt über ALLE Iterationen wirkt statt einmalig -
        # sonst Gefahr von Überkompensation/Instabilität in
        # _apply_continuity_correction).
        height_range = heightmap.max() - heightmap.min()
        height_normalized = (heightmap - heightmap.min()) / height_range if height_range > 1e-6 \
            else np.zeros_like(heightmap, dtype=np.float32)
        terrain_pressure_term = height_normalized * parameters['terrain_factor'] * 0.2

        # CFD-Iterationen
        for iteration in range(cfd_iterations):
            # Progress-Update für längere CFD-Simulationen
            if iteration % max(1, cfd_iterations // 5) == 0:
                progress = 30 + (iteration / cfd_iterations) * 30
                self._update_progress("Wind CFD", int(progress),
                                    f"CFD iteration {iteration + 1}/{cfd_iterations}")

            # Druckgradienten berechnen (vectorized) - Terrain-Kopplung fließt
            # hier ein, VOR der Gradientenberechnung (siehe Kommentar oben).
            pressure_field_iter = pressure_field - terrain_pressure_term
            pressure_grad_x = np.zeros_like(pressure_field)
            pressure_grad_y = np.zeros_like(pressure_field)

            pressure_grad_x[:, 1:-1] = (pressure_field_iter[:, 2:] - pressure_field_iter[:, :-2]) * 0.5
            pressure_grad_y[1:-1, :] = (pressure_field_iter[2:, :] - pressure_field_iter[:-2, :]) * 0.5

            # Wind aus Druckgradienten
            wind_field[:, :, 0] = -pressure_grad_x * parameters['wind_speed_factor'] * 10.0
            wind_field[:, :, 1] = -pressure_grad_y * parameters['wind_speed_factor'] * 10.0

            # Terrain-Ablenkung (vectorized)
            terrain_factor = parameters['terrain_factor'] * 0.5
            wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor  # Slope Y -> Wind X
            wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor  # Slope X -> Wind Y

            # Grat-/Canyon-Speedup (siehe [[project-wind-ridge-speedup]])
            ridge_factor = 1.0 + _RIDGE_SPEEDUP_STRENGTH * (-curvature_norm) * parameters['terrain_factor']
            wind_field[:, :, 0] *= ridge_factor
            wind_field[:, :, 1] *= ridge_factor

            # Thermal-Convection
            self._apply_thermal_convection(wind_field, temp_map, shadowmap, parameters)

            # Wind-Diffusion für Stabilität (simplified)
            wind_field = self._apply_wind_diffusion(wind_field, 0.1)

            # Kontinuitäts-Correction für Massenerhaltung
            self._apply_continuity_correction(wind_field)

        return wind_field

    def _simulate_wind_field_simple(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Simple-Fallback: Basic Wind-Field ohne CFD-Komplexität
        """
        height, width = heightmap.shape
        wind_field = np.zeros((height, width, 2), dtype=np.float32)

        # Konstanter Wind entlang der vorherrschenden Windrichtung
        wind_direction_deg = parameters.get('prevailing_wind_direction', 0.0)
        theta = np.radians(wind_direction_deg)
        base_wind_speed = parameters['wind_speed_factor'] * 5.0
        wind_field[:, :, 0] = base_wind_speed * np.cos(theta)
        wind_field[:, :, 1] = base_wind_speed * np.sin(theta)

        # Basic Terrain-Ablenkung
        slopemap = self._calculate_slopes_vectorized(heightmap)
        terrain_factor = parameters['terrain_factor'] * 0.2
        wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor
        wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor

        return wind_field

    def _calculate_atmospheric_moisture(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                      wind_map: np.ndarray, parameters: Dict[str, Any],
                                      hardness_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Atmospheric-Moisture-Calculation mit Evaporation und Transport

        Implementiert Magnus-Formel für Sättigungsdampfdruck und Wind-enhanced Evaporation

        hardness_map: optional, nur vom CPU-Fallback für die hardness-gewichtete
        Diffusion genutzt (siehe _apply_humidity_diffusion) - der GPU-Pfad nutzt
        weiterhin humidityDiffusion.comp's einfachen Box-Blur ohne Hardness-
        Gewichtung (bekannte Lücke, analog zur bewusst zurückgestellten GPU-
        Watershed-Parität für #42 - eigener Folge-Task, kein Blocker hier).
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'atmospheric_moisture',
                    'input_data': {
                        'heightmap': heightmap,
                        'temp_map': temp_map,
                        'wind_map': wind_map
                    },
                    'parameters': parameters,
                    'lod_level': heightmap.shape[0]
                }

                result = self.shader_manager.request_atmospheric_moisture(shader_request)

                if result.success:
                    return result.humidity_field

        except Exception as e:
            self.logger.warning(f"GPU moisture calculation failed: {e}")

        # CPU-Fallback (Gut)
        return self._calculate_atmospheric_moisture_cpu(heightmap, temp_map, wind_map, parameters, hardness_map)

    def _calculate_atmospheric_moisture_cpu(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                          wind_map: np.ndarray, parameters: Dict[str, Any],
                                          hardness_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        CPU-optimierte Atmospheric-Moisture mit vectorized Operations
        """
        height, width = temp_map.shape

        # Initiale Soil-Moisture (vereinfacht - 50% überall)
        soil_moisture = np.full((height, width), 50.0, dtype=np.float32)

        # Evaporation-Rate basierend auf Temperatur (Magnus-Formel vereinfacht)
        # Sättigungsdampfdruck steigt exponentiell mit Temperatur
        temp_celsius = np.maximum(-50, np.minimum(60, temp_map))  # Clamp temperature
        saturation_vapor_pressure = 6.112 * np.exp(17.67 * temp_celsius / (temp_celsius + 243.5))

        # Wind-Speed für Enhanced-Evaporation
        wind_speed = np.sqrt(wind_map[:, :, 0]**2 + wind_map[:, :, 1]**2)
        wind_factor = np.minimum(2.0, wind_speed / 5.0)  # Cap bei 2x Enhancement

        # Evaporation-Rate
        evaporation_rate = (soil_moisture / 100.0) * (saturation_vapor_pressure / 100.0) * (1.0 + wind_factor)

        # Initiale Humidity aus Evaporation. Skalierungsfaktor kalibriert gegen
        # rho_max = 5*exp(0.06*T) aus _calculate_precipitation_cpu (Magnus-Formel
        # Sättigungsdampfdichte, Größenordnung ~5-17 bei -3..20°C): mit dem alten
        # Faktor 10 lag humid_map bei ~0.3-1.5 - relative_humidity = humid_map/rho_max
        # blieb dadurch IMMER unter ~0.09, "oversaturation" (>1.0, einziger Auslöser
        # für Kondensations-Niederschlag) konnte nie auftreten, unabhängig von
        # Temperatur/Wind/Terrain. precip_map bestand dadurch nur noch aus dem
        # (kleinen) orographischen Anteil. Faktor empirisch so gewählt, dass bei
        # Default-Parametern ein Mix aus über-/untersättigten Gebieten entsteht statt
        # eines globalen Alles-oder-Nichts-Zustands (siehe [[project-precip-humidity-calibration]]).
        humid_map = evaporation_rate * 140.0

        # Moisture-Transport (vereinfacht, 3 Iterationen)
        for _ in range(3):
            humid_map = self._transport_moisture_simple(humid_map, wind_map, dt=0.5)

        # Diffusion für smooth Distribution
        humid_map = self._apply_humidity_diffusion(humid_map, iterations=2, hardness_map=hardness_map)

        return humid_map

    def _calculate_precipitation_system(self, humid_map: np.ndarray, temp_map: np.ndarray,
                                      wind_map: np.ndarray, heightmap: np.ndarray,
                                      parameters: Dict[str, Any]) -> np.ndarray:
        """
        Precipitation-System mit Orographic-Effects und Condensation

        Implementiert Luv-/Lee-Effekte und Magnus-Formel für Condensation-Thresholds
        """
        try:
            # GPU-Shader-Request (Optimal)
            if self.shader_manager:
                shader_request = {
                    'operation_type': 'precipitation_calculation',
                    'input_data': {
                        'humid_map': humid_map,
                        'temp_map': temp_map,
                        'wind_map': wind_map,
                        'heightmap': heightmap
                    },
                    'parameters': parameters,
                    'lod_level': heightmap.shape[0]
                }

                result = self.shader_manager.request_precipitation_calculation(shader_request)

                if result.success:
                    return result.precipitation_field

        except Exception as e:
            self.logger.warning(f"GPU precipitation calculation failed: {e}")

        # CPU-Fallback (Gut)
        return self._calculate_precipitation_cpu(humid_map, temp_map, wind_map, heightmap, parameters)

    def _calculate_precipitation_cpu(self, humid_map: np.ndarray, temp_map: np.ndarray,
                                    wind_map: np.ndarray, heightmap: np.ndarray,
                                    parameters: Dict[str, Any]) -> np.ndarray:
        """
        CPU-optimierte Precipitation mit Orographic-Effects und Condensation-Logic
        """
        height, width = humid_map.shape
        precip_map = np.zeros((height, width), dtype=np.float32)

        # Slopemap für Orographic-Effects
        slopemap = self._calculate_slopes_vectorized(heightmap)

        # 1. Orographic Precipitation (Luv-/Lee-Effekte)
        wind_speed = np.sqrt(wind_map[:, :, 0]**2 + wind_map[:, :, 1]**2)

        # Wind-Slope-Alignment für Luv-Identifikation
        wind_norm_x = np.where(wind_speed > 0.1, wind_map[:, :, 0] / wind_speed, 0)
        wind_norm_y = np.where(wind_speed > 0.1, wind_map[:, :, 1] / wind_speed, 0)

        wind_slope_alignment = (wind_norm_x * slopemap[:, :, 0] +
                               wind_norm_y * slopemap[:, :, 1])

        # Orographic Enhancement an Luvhängen
        orographic_factor = np.maximum(0, wind_slope_alignment) * wind_speed * 0.3
        oro_precip = humid_map * orographic_factor * 0.05

        # 2. Condensation Precipitation (Magnus-Formel)
        # Sättigungsdampfdichte: rho_max = 5*exp(0.06*T)
        temp_celsius = np.maximum(-40, np.minimum(50, temp_map))
        rho_max = 5.0 * np.exp(0.06 * temp_celsius)

        # Relative Humidity
        relative_humidity = np.where(rho_max > 0, humid_map / rho_max, 0)

        # Precipitation bei Übersättigung (> 1.0)
        oversaturation = np.maximum(0, relative_humidity - 1.0)
        condensation_precip = oversaturation * rho_max * 0.6

        # 3. Kombiniere Precipitation-Sources
        precip_map = oro_precip + condensation_precip

        # Physical Limits
        precip_map = np.maximum(0, precip_map)  # Kein negativer Niederschlag
        precip_map = np.minimum(500, precip_map)  # Maximum 500 gH2O/m²

        return precip_map

    def _create_weather_data(self, wind_map: np.ndarray, temp_map: np.ndarray,
                           precip_map: np.ndarray, humid_map: np.ndarray,
                           lod_level: int, target_size: int,
                           parameters: Dict[str, Any],
                           wind_map_monthly=None, temp_map_monthly=None,
                           precip_map_monthly=None, humid_map_monthly=None,
                           temp_map_layers=None, wind_map_layers=None, humid_map_layers=None,
                           temp_map_layers_monthly=None, wind_map_layers_monthly=None,
                           humid_map_layers_monthly=None) -> WeatherData:
        """
        Erstellt WeatherData-Objekt mit vollständiger LOD-Integration und Validation
        """
        weather_data = WeatherData()
        weather_data.wind_map = wind_map
        weather_data.temp_map = temp_map
        weather_data.precip_map = precip_map
        weather_data.humid_map = humid_map
        weather_data.wind_map_monthly = wind_map_monthly
        weather_data.temp_map_monthly = temp_map_monthly
        weather_data.precip_map_monthly = precip_map_monthly
        weather_data.humid_map_monthly = humid_map_monthly
        # 3-Schicht-Atmosphäre, rein additiv (siehe [[project-3layer-wind-cfd]]) -
        # None, falls der gekoppelte Loop diese Runde nicht lief.
        weather_data.temp_map_layers = temp_map_layers
        weather_data.wind_map_layers = wind_map_layers
        weather_data.humid_map_layers = humid_map_layers
        weather_data.temp_map_layers_monthly = temp_map_layers_monthly
        weather_data.wind_map_layers_monthly = wind_map_layers_monthly
        weather_data.humid_map_layers_monthly = humid_map_layers_monthly
        weather_data.lod_level = lod_level
        weather_data.actual_size = target_size

        # Parameter-Hash für Cache-Management
        weather_data.parameter_hash = self._calculate_parameter_hash(parameters)

        # Validity-State setzen
        weather_data.validity_state = {
            "valid": True,
            "dependencies_satisfied": True,
            "lod_level": lod_level,
            "last_generation": "weather_system"
        }

        # Data-Quality-Validation
        self._validate_weather_output(weather_data)

        return weather_data

    def _validate_weather_output(self, weather_data: WeatherData):
        """
        Validiert Weather-Output für Data-Integrity und Physical-Plausibility
        """
        # NaN-Detection
        for field_name, field_data in [
            ("wind_map", weather_data.wind_map),
            ("temp_map", weather_data.temp_map),
            ("precip_map", weather_data.precip_map),
            ("humid_map", weather_data.humid_map)
        ]:
            if np.any(np.isnan(field_data)) or np.any(np.isinf(field_data)):
                self.logger.warning(f"Invalid values detected in {field_name}")
                weather_data.validity_state["valid"] = False

        # Physical-Range-Validation
        if not (-60 <= np.min(weather_data.temp_map) and np.max(weather_data.temp_map) <= 80):
            self.logger.warning("Temperature values outside physical range")

        if not (0 <= np.min(weather_data.precip_map)):
            self.logger.warning("Negative precipitation values detected")

        if not (0 <= np.min(weather_data.humid_map)):
            self.logger.warning("Negative humidity values detected")

        # Wind-Speed-Validation
        wind_speeds = np.sqrt(weather_data.wind_map[:, :, 0]**2 + weather_data.wind_map[:, :, 1]**2)
        if np.max(wind_speeds) > 100:  # > 100 m/s unrealistic
            self.logger.warning("Unrealistic wind speeds detected")

    def _update_performance_stats(self, weather_data: WeatherData, cfd_iterations: int):
        """
        Aktualisiert Performance-Statistics für Monitoring und Optimization
        """
        weather_data.performance_stats = {
            "cfd_iterations": cfd_iterations,
            "lod_level": weather_data.lod_level,
            "map_size": weather_data.actual_size,
            "generation_method": "gpu" if self.shader_manager else "cpu",
            "temp_range": {
                "min": float(np.min(weather_data.temp_map)),
                "max": float(np.max(weather_data.temp_map)),
                "mean": float(np.mean(weather_data.temp_map))
            },
            "wind_stats": {
                "max_speed": float(np.max(np.sqrt(weather_data.wind_map[:, :, 0]**2 +
                                                 weather_data.wind_map[:, :, 1]**2))),
                "mean_speed": float(np.mean(np.sqrt(weather_data.wind_map[:, :, 0]**2 +
                                                   weather_data.wind_map[:, :, 1]**2)))
            },
            "precipitation_total": float(np.sum(weather_data.precip_map))
        }

    def _create_fallback_weather_data(self, original_size: int, lod_level: int,
                                     parameters: Dict[str, Any]) -> WeatherData:
        """
        Error-Recovery: Erstellt Minimal-Weather-System bei kritischen Failures
        """
        target_size = self._get_lod_size(lod_level, original_size)

        # Minimal-Weather-Fields
        weather_data = WeatherData()
        weather_data.wind_map = np.zeros((target_size, target_size, 2), dtype=np.float32)
        weather_data.wind_map[:, :, 0] = 5.0  # Konstanter 5 m/s Ostwind

        weather_data.temp_map = np.full((target_size, target_size),
                                      parameters.get('air_temp_entry', 15.0), dtype=np.float32)
        weather_data.precip_map = np.full((target_size, target_size), 50.0, dtype=np.float32)
        weather_data.humid_map = np.full((target_size, target_size), 30.0, dtype=np.float32)

        weather_data.lod_level = lod_level
        weather_data.actual_size = target_size
        weather_data.validity_state = {"valid": False, "fallback": True}

        self.logger.warning("Fallback weather data created due to generation failure")

        return weather_data

    # ===== UTILITY METHODS =====

    def _interpolate_2d_bicubic(self, array: np.ndarray, target_size: int) -> np.ndarray:
        """
        Bicubic-Interpolation für Pattern-Preservation bei LOD-Upscaling
        """
        from scipy.ndimage import zoom

        old_size = array.shape[0]
        if old_size == target_size:
            return array.copy()

        scale_factor = target_size / old_size

        try:
            # SciPy zoom für bicubic-ähnliche Interpolation
            interpolated = zoom(array, scale_factor, order=3)

            # Exakte Größe sicherstellen
            if interpolated.shape[0] != target_size:
                interpolated = interpolated[:target_size, :target_size]

            return interpolated.astype(array.dtype)

        except Exception as e:
            self.logger.warning(f"Bicubic interpolation failed, using bilinear: {e}")
            # Fallback zu bilinearer Interpolation
            return self._interpolate_2d_bilinear(array, target_size)

    def _interpolate_2d_bilinear(self, array: np.ndarray, target_size: int) -> np.ndarray:
        """
        Bilineare Interpolation als Fallback für Bicubic
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

    def _calculate_slopes_vectorized(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Vectorized Slope-Calculation (dz/dx, dz/dy) für Performance.

        Durch reale Meter/Pixel geteilt (siehe core/terrain_generator.py
        SlopeCalculator._calculate_cpu_slopes für den identischen Bug an anderer
        Stelle) - ohne das war ein 1m Höhenunterschied zwischen Nachbarpixeln wie
        1m realer Horizontal-Abstand behandelt, obwohl ein Pixel bei typischen
        Kartengrößen tatsächlich ~50-300m abdeckt. Das ließ wind_slope_alignment
        (und darüber oro_precip) auf praktisch jedem Hang unrealistisch groß werden.
        """
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        from gui.config.value_default import TERRAIN
        spacing = (TERRAIN.WORLD_SIZE_KM * 1000.0) / height

        # dz/dx (vectorized)
        slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5 / spacing
        slopemap[:, 0, 0] = (heightmap[:, 1] - heightmap[:, 0]) / spacing
        slopemap[:, -1, 0] = (heightmap[:, -1] - heightmap[:, -2]) / spacing

        # dz/dy (vectorized)
        slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5 / spacing
        slopemap[0, :, 1] = (heightmap[1, :] - heightmap[0, :]) / spacing
        slopemap[-1, :, 1] = (heightmap[-1, :] - heightmap[-2, :]) / spacing

        return slopemap

    def _calculate_curvature_normalized(self, heightmap: np.ndarray) -> np.ndarray:
        """
        Normalisierte Gelände-Krümmung (diskreter Laplace der Heightmap,
        reales Meter/Pixel-Spacing wie _calculate_slopes_vectorized) für den
        Grat-/Canyon-Speedup-Term (siehe [[project-wind-ridge-speedup]],
        WindNinjas Terrain-Shape-Effekt): negativ auf konvexen Graten/Kuppen
        (Wind soll dort beschleunigen), positiv in konkaven Tälern (Wind soll
        dort bremsen).

        Auf [-1,1] per geclipptem Z-Score (3 Standardabweichungen) normiert
        statt fixer physikalischer Einheiten - die rohe Krümmung
        (Höhe/Meter²) hängt stark von TERRAIN.AMPLITUDE/WORLD_SIZE_KM ab, ein
        Z-Score-Clip macht den Speedup-Effekt unabhängig von der absoluten
        Terrain-Skalierung nutzbar (ähnliches Muster wie height_normalized an
        anderer Stelle dieser Datei, aber robust gegen Ausreißer statt Min/Max).
        Ränder bleiben bei 0 (kein Krümmungssignal) - dieselbe Vereinfachung
        wie das ungenutzte äußerste Pixel bei vielen zentralen Differenzen
        dieser Datei.
        """
        height, width = heightmap.shape
        from gui.config.value_default import TERRAIN
        spacing = (TERRAIN.WORLD_SIZE_KM * 1000.0) / height

        d2x = np.zeros((height, width), dtype=np.float32)
        d2y = np.zeros((height, width), dtype=np.float32)
        d2x[:, 1:-1] = (heightmap[:, 2:] - 2.0 * heightmap[:, 1:-1] + heightmap[:, :-2]) / (spacing ** 2)
        d2y[1:-1, :] = (heightmap[2:, :] - 2.0 * heightmap[1:-1, :] + heightmap[:-2, :]) / (spacing ** 2)
        laplacian = d2x + d2y

        scale = float(np.std(laplacian)) + 1e-9
        return np.clip(laplacian / (3.0 * scale), -1.0, 1.0).astype(np.float32)

    def _resize_nearest_labels(self, label_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Nearest-Neighbor-Resize für kategoriale/Label-Daten (z.B. biome_map) -
        bilineare/bikubische Interpolation wie _interpolate_2d_bicubic würde
        an Kategoriegrenzen unsinnige Zwischenwerte erzeugen. Siehe
        _get_roughness_damping() für den Aufrufkontext (analog zum
        bestehenden Muster in water_generator.py::_resize_nearest).
        """
        src_h, src_w = label_map.shape[:2]
        tgt_h, tgt_w = target_shape
        if (src_h, src_w) == (tgt_h, tgt_w):
            return label_map
        y_idx = np.clip((np.arange(tgt_h) * src_h / tgt_h).astype(int), 0, src_h - 1)
        x_idx = np.clip((np.arange(tgt_w) * src_w / tgt_w).astype(int), 0, src_w - 1)
        return label_map[np.ix_(y_idx, x_idx)]

    def _get_roughness_damping(self, target_shape: Tuple[int, int], lod_level: int) -> Optional[np.ndarray]:
        """
        Best-effort Bodenrauigkeits-Dämpfung aus der Biome-Klassifikation
        (siehe [[project-wind-roughness]], _BIOME_ROUGHNESS_DAMPING oben).

        Biome läuft NACH Weather im Calculator-Graph (weather.wind ist keine
        Abhängigkeit von biome.*) - ein aktueller Wert ist nur verfügbar,
        wenn Biome in DIESER Session bereits (mindestens für ein niedrigeres
        LOD) angefragt wurde, z.B. beim Auto-Start der vollen Pipeline. Bei
        isolierter Weather-Tab-Generierung bleibt Biome unangefragt - dann
        liefert dieser Aufruf None, und der Aufrufer wendet keine Dämpfung an
        (identisches Verhalten zu vor dieser Änderung).

        Bei einer Shape-Abweichung (z.B. weil Biome zuletzt bei einer anderen
        map_size lief - der DataLODManager-Cache für Calculator-Outputs wird
        von clear_all_data()/invalidate_cache_lod() nie geleert) wird per
        Nearest-Neighbor resampled statt zu verwerfen, analog zum
        bestehenden Muster in water_generator.py::_resize_nearest - eine
        stilisierte, leicht veraltete Rauigkeits-Näherung ist hier
        unproblematischer als bei echten Höhen-/Strömungsdaten.
        """
        biome_map = self.data_lod_manager.get_calculator_output(
            "biome.integrate_layers", "biome_map", lod_level)
        if biome_map is None:
            return None
        if biome_map.shape[:2] != tuple(target_shape):
            biome_map = self._resize_nearest_labels(biome_map, target_shape)
        ids = np.clip(biome_map.astype(np.int32), 0, len(_BIOME_ROUGHNESS_DAMPING) - 1)
        return _BIOME_ROUGHNESS_DAMPING[ids]

    def _generate_atmospheric_noise(self, shape: Tuple[int, int], target_size: int,
                                    month_index: int = 0) -> np.ndarray:
        """
        Generiert atmospheric Noise-Variation mit Edge-Enhancement.

        month_index verschiebt die Noise-Koordinaten um einen fixen Offset - exakt
        dasselbe Muster wie _generate_pressure_noise() (letzte Runde). Vorher nutzte
        dieser Aufruf IMMER dieselben Koordinaten (nur von der Pixelposition
        abhängig) - da dieses Rauschen die Bodentemperatur speist, die wiederum die
        initiale Feuchte-Quelle des gekoppelten 3-Schicht-Loops treibt, blieb das
        räumliche MUSTER von humid_map über alle 6 Monate praktisch identisch (nur
        das Niveau verschob sich über saisonale Parameter) - empirisch bestätigt:
        paarweise räumliche Korrelation zwischen Monaten lag bei 0.72-0.96. Offset
        (10 Einheiten pro Monat) liegt deutlich über der OpenSimplex-Kohärenzlänge
        bei dieser Frequenz, siehe [[project-3layer-wind-cfd]].

        Vektorisiert via noise2array() statt der früheren Python-Doppelschleife
        (identische Formel, inkl. Edge-Enhancement) - selber Grund wie die
        Vektorisierung der CFD-Hotpaths letzte Runde: wird jetzt pro Monat
        aufgerufen, nicht mehr nur einmal insgesamt.
        """
        height, width = shape
        offset = month_index * 10.0

        x_coords = np.arange(width, dtype=np.float64) / width * 3.0 + offset
        y_coords = np.arange(height, dtype=np.float64) / height * 3.0 + offset
        noise_field = self.noise_generator.noise2array(x_coords, y_coords).astype(np.float32)

        if target_size > 128:
            noise_field += self.noise_generator.noise2array(
                x_coords * 2, y_coords * 2).astype(np.float32) * 0.3

        # Edge-Factor für stärkere Variation an Kartenrändern (identische Formel
        # wie die frühere _calculate_edge_factor(), vektorisiert).
        y_idx, x_idx = np.mgrid[0:height, 0:width]
        dist_to_edge = np.minimum.reduce([x_idx, y_idx, width - 1 - x_idx, height - 1 - y_idx])
        max_dist = max(min(width, height) // 6, 1)
        edge_factor = np.where(
            dist_to_edge < max_dist,
            1.0 + (max_dist - dist_to_edge) / max_dist * 0.8,
            1.0
        ).astype(np.float32)

        return noise_field * edge_factor * 3.0

    def _generate_pressure_noise(self, shape: Tuple[int, int], month_index: int = 0) -> np.ndarray:
        """
        Generiert Pressure-Noise für CFD-Simulation.

        month_index verschiebt die Noise-Koordinaten um einen fixen Offset -
        vorher nutzte dieser Aufruf IMMER dieselben Koordinaten (nur von der
        Pixelposition abhängig), wodurch das kleinräumige, optisch dominante
        Rauschmuster des Windfelds für alle 6 saisonalen Perioden identisch
        blieb, während nur die (schwächere) großräumige Grundrichtung
        rotierte - der Wind wirkte dadurch trotz korrekt saisonal variierender
        Richtung optisch "eingefroren". Offset (10 Einheiten pro Monat) liegt
        deutlich über der OpenSimplex-Kohärenzlänge bei dieser Frequenz, das
        Muster ist pro Monat dadurch optisch unabhängig. Nutzt noise2array()
        (bereits an anderer Stelle dieser Session verifiziert) statt der
        früheren Doppel-Schleife - vektorisiert, kein Verhaltensrisiko.
        """
        height, width = shape
        offset = month_index * 10.0
        x = np.arange(width, dtype=np.float64) / width * 2.0 + offset
        y = np.arange(height, dtype=np.float64) / height * 2.0 + offset
        return self.noise_generator.noise2array(x, y).astype(np.float32)

    def _apply_thermal_convection(self, wind_field: np.ndarray, temp_map: np.ndarray,
                                 shadowmap: np.ndarray, parameters: Dict[str, Any]):
        """
        Wendet thermische Konvektion auf Wind-Field an (in-place)
        """
        height, width = temp_map.shape

        # Temperature-Gradients (vectorized)
        temp_grad_x = np.zeros_like(temp_map)
        temp_grad_y = np.zeros_like(temp_map)

        temp_grad_x[:, 1:-1] = (temp_map[:, 2:] - temp_map[:, :-2]) * 0.5
        temp_grad_y[1:-1, :] = (temp_map[2:, :] - temp_map[:-2, :]) * 0.5

        # Thermal-Convection-Strength
        avg_temp = np.mean(temp_map)
        temp_diff = temp_map - avg_temp
        convection_strength = temp_diff * parameters['thermic_effect'] * 0.08

        # Shadow-based thermal effects
        if len(shadowmap.shape) == 3:
            shadow_avg = np.mean(shadowmap, axis=2)
        else:
            shadow_avg = shadowmap

        shadow_effect = (shadow_avg - 0.5) * parameters['thermic_effect'] * 0.15

        # Apply thermal modifications
        wind_field[:, :, 0] += temp_grad_x * 0.05 + convection_strength
        wind_field[:, :, 1] += temp_grad_y * 0.05 + shadow_effect

    def _apply_wind_diffusion(self, wind_field: np.ndarray, diffusion_rate: float) -> np.ndarray:
        """
        Wendet Diffusion auf Wind-Field für numerical Stability an.

        Vektorisiert via np.roll statt der früheren Python-Doppelschleife (identische
        4-Nachbar-Mittelwert-Formel) - notwendig, weil der neue 3-Schicht-CFD-
        Zeitschritt-Loop (siehe [[project-3layer-wind-cfd]]) diese Funktion pro Schicht
        UND pro Zeitschritt aufruft statt wie bisher nur einmal pro Monat; bei größeren
        LODs wäre die reine Python-Schleife sonst spürbar limitierend. np.roll wraps am
        Rand um, aber die dadurch "falschen" Werte landen nur in Zeile/Spalte 0 bzw.
        -1 des Zwischenergebnisses, die am Ende explizit auf die unveränderten
        Original-Randwerte zurückgesetzt werden (identisch zum alten Verhalten, das
        Randpixel nie anfasste, da die Schleife bei range(1, height-1) begann).
        """
        neighbor_avg = (
            np.roll(wind_field, 1, axis=0) + np.roll(wind_field, -1, axis=0) +
            np.roll(wind_field, 1, axis=1) + np.roll(wind_field, -1, axis=1)
        ) * 0.25

        diffused_field = wind_field + (neighbor_avg - wind_field) * diffusion_rate
        diffused_field[0, :] = wind_field[0, :]
        diffused_field[-1, :] = wind_field[-1, :]
        diffused_field[:, 0] = wind_field[:, 0]
        diffused_field[:, -1] = wind_field[:, -1]

        return diffused_field

    def _apply_vorticity_confinement(self, wind_field: np.ndarray, vorticity_strength) -> None:
        """
        Vorticity Confinement (Fedkiw/Stam-Standardtechnik, siehe
        [[project-3layer-wind-cfd]]) - injiziert lokale Rotationsenergie zurück,
        die _apply_wind_diffusion pro Zeitschritt entfernt. Formel 1:1 aus
        niels747/2D-Weather-Sandbox's curlShader.frag + vorticityShader.frag
        übernommen (nicht neu hergeleitet), an unser (H,W,2)-Layout angepasst:

            curl = dv/dx - du/dy                              (skalare 2D-Rotation)
            force = normalize(grad(|curl|)) * curl * staerke   (Kraft senkrecht zum
                                                                 |curl|-Gradienten)

        Ohne diesen Fix war der Wind über weite Flächen fast parallel (empirisch
        ~7.8° mittlere Richtungsänderung zwischen Nachbarpixeln) - reine Diffusion
        + Kontinuitätskorrektur glätten aktiv jede kleinräumige Verwirbelung weg,
        die "Turbulenz" ausmacht.

        Modifiziert wind_field IN-PLACE (wie _apply_continuity_correction).
        vorticity_strength: Skalar ODER (H,W)-Array (z.B. für die Rand-
        Verstärkung, siehe vorticity_edge_boost in _run_coupled_atmosphere_
        simulation) - beides funktioniert per NumPy-Broadcasting unverändert.

        STABILITÄTS-KAPPUNG: die Referenz-Formel selbst hat keine eingebaute
        Grenze - bei uns wird sie über viele Zeitschritte (8-60+, siehe
        _get_atmosphere_loop_steps) wiederholt angewendet, ohne einen echten
        Zeitschritt-Skalierungsfaktor dt (anders als im Referenz-Shader, der pro
        Frame mit einem sehr kleinen, festen dt läuft). Ohne Kappung entsteht eine
        Rückkopplung (mehr Curl -> mehr Kraft -> mehr Curl im nächsten Schritt),
        die bei höheren LODs (mehr Iterationen UND feineres Gitter) beobachtbar
        eskalierte (empirisch: 12 m/s bei LOD1/8 Schritten, 247 m/s bei LOD3/18
        Schritten - eindeutige Instabilität, kein Rundungsfehler). Kraft-Betrag
        pro Aufruf auf denselben Größenordnungsbereich gekappt wie die übrigen
        Pro-Iterations-Terme in dieser Datei (thermische Konvektion, Terrain-
        Ablenkung: O(0.05-0.5)) - Turbulenz baut sich dadurch über mehrere
        Schritte graduell auf statt in einem Schritt zu explodieren.
        """
        u, v = wind_field[:, :, 0], wind_field[:, :, 1]

        dvdx = np.zeros_like(v)
        dudy = np.zeros_like(u)
        dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) * 0.5
        dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) * 0.5
        curl = dvdx - dudy

        abs_curl = np.abs(curl)
        grad_x = np.zeros_like(curl)
        grad_y = np.zeros_like(curl)
        grad_x[:, 1:-1] = (abs_curl[:, 2:] - abs_curl[:, :-2]) * 0.5
        grad_y[1:-1, :] = (abs_curl[2:, :] - abs_curl[:-2, :]) * 0.5
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6

        force_x = (grad_y / magnitude) * curl * vorticity_strength
        force_y = (-grad_x / magnitude) * curl * vorticity_strength

        FORCE_CAP = 1.0  # m/s pro Aufruf, siehe Docstring-Begründung oben
        force_magnitude = np.sqrt(force_x ** 2 + force_y ** 2)
        scale_down = np.minimum(1.0, FORCE_CAP / np.maximum(force_magnitude, 1e-6))
        force_x *= scale_down
        force_y *= scale_down

        wind_field[:, :, 0] += force_x
        wind_field[:, :, 1] += force_y

    def _get_poisson_inv_eigs(self, height: int, width: int) -> np.ndarray:
        """
        Liefert (gecached) 1/lambda(p,q) für den separierbaren 2D-Neumann-Laplace,
        den die DCT-Typ-2/3-Basis diagonalisiert: lambda(p,q) = 2cos(pi*p/H) +
        2cos(pi*q/W) - 4. lambda(0,0)=0 (Mittelwert-Freiheitsgrad von phi) wird auf
        einen Dummy-Wert gesetzt - div_hat[0,0] wird vor der Division ohnehin auf 0
        gezwungen (siehe _apply_continuity_correction), der Dummy-Wert hier wird also
        nie tatsächlich verwendet.
        """
        key = (height, width)
        cached = self._poisson_eig_cache.get(key)
        if cached is not None:
            return cached
        # Eigenwert-Berechnung selbst in float64 (nur ein einmaliger Aufwand pro
        # Grid-Größe, dank Cache), der resultierende Nenner wird für die
        # Multiplikation mit dem float32-DCT-Spektrum nach float32 zurückgecastet.
        p = np.arange(height, dtype=np.float64)
        q = np.arange(width, dtype=np.float64)
        lam = (2.0 * np.cos(np.pi * p / height) - 2.0)[:, None] + \
              (2.0 * np.cos(np.pi * q / width) - 2.0)[None, :]
        lam[0, 0] = 1.0
        inv_lam = (1.0 / lam).astype(np.float32)
        self._poisson_eig_cache[key] = inv_lam
        return inv_lam

    def _apply_continuity_correction(self, wind_field: np.ndarray, vertical_flux_term: Optional[np.ndarray] = None):
        """
        Erzwingt Massenerhaltung (~Divergenzfreiheit) per Poisson-Projektion -
        WindNinja-inspiriert (siehe [[project-wind-poisson-projection]]): WindNinja
        baut ein initiales Windfeld u0 und löst dann eine Poisson-Gleichung
        grad^2(phi) = div(u0) für ein Korrekturpotential phi (bei WindNinja per FEM+CG
        auf einem 3D-Terrain-Mesh), danach u_final = u0 - grad(phi). Für dieses
        reguläre Cartesian-Grid ist das direkte Äquivalent eine schnelle
        DCT-basierte Poisson-Lösung mit Neumann-Randbedingung (kein Wind-Quell-/
        Senken-Fluss über den Kartenrand) - derselbe Projektionsschritt wie in
        "Stable Fluids" (Stam).

        Ersetzt die frühere schwache lokale Iteration (nur 10% der lokalen Divergenz
        pro Aufruf entfernt, Ränder nie korrigiert) durch eine einmalige globale
        Lösung pro Aufruf, die das Feld in einem Schritt (bis auf
        Diskretisierungs-/Rundungsfehler) tatsächlich divergenzfrei macht - siehe
        Verifikation in [[project-wind-poisson-projection]].

        vertical_flux_term (optional, (H,W)): zusätzlicher Divergenz-Beitrag aus dem
        vertikalen Massenaustausch zwischen Atmosphären-Schichten (siehe
        _run_coupled_atmosphere_simulation Schritt 6, [[project-3layer-wind-cfd]]) -
        (w_oben - w_unten)/dicke. Positiv = Netto-Massenzufluss von oben/unten in
        diese Schicht (die Horizontal-Korrektur muss dann netto AUSWärts divergieren,
        um den Zufluss auszugleichen), negativ = Netto-Abfluss (Konvergenz nötig).
        Eine reine Neumann-Projektion kann nur Divergenz mit Domänen-Mittel ~0
        entfernen; ein echter über die Schicht gemittelter Netto-Massenfluss bleibt
        als realer Rest bestehen - das ist physikalisch korrekt (Nettomasse fließt
        tatsächlich in/aus dieser Schicht), kein Solver-Defekt.

        Divergenz/Gradient nutzen weiterhin das zentrale 2h-Schema dieser Datei,
        jetzt aber auch an den Rändern via Spiegel-Ghost-Zellen (f[-1]:=f[0],
        f[N]:=f[N-1] - die diskrete Neumann-Randbedingung), statt sie wie zuvor auf 0
        zu lassen.

        Poisson-Arithmetik läuft komplett in float32 (Messung: float64 brachte bei
        realistischen, bereits geglätteten Windfeldern - hier läuft _apply_wind_diffusion
        immer direkt davor - keine messbar bessere Divergenz-Reduktion (~96-98% in
        beiden Fällen), kostete aber ~2x mehr Laufzeit, siehe
        [[project-wind-poisson-projection]]. Die theoretische float32-Rauschgrenze bei
        kleinen Eigenwerten (Verstärkungsfaktor ~10^5 bei N=1024) betrifft primär
        adversarielle/reine Rauschfelder mit signifikanter Energie in der Domänen-Mittel-
        Komponente, nicht die hier tatsächlich vorkommenden glatten Felder.
        """
        height, width = wind_field.shape[:2]
        u = wind_field[:, :, 0]
        v = wind_field[:, :, 1]

        div = np.zeros((height, width), dtype=np.float32)
        div[:, 1:-1] = (u[:, 2:] - u[:, :-2]) * 0.5
        div[:, 0] = (u[:, 1] - u[:, 0]) * 0.5
        div[:, -1] = (u[:, -1] - u[:, -2]) * 0.5
        div[1:-1, :] += (v[2:, :] - v[:-2, :]) * 0.5
        div[0, :] += (v[1, :] - v[0, :]) * 0.5
        div[-1, :] += (v[-1, :] - v[-2, :]) * 0.5
        if vertical_flux_term is not None:
            # Netto-Zufluss von oben/unten wirkt wie negative horizontale Divergenz
            # (die Zelle "will" sich horizontal ausdehnen, um den Zufluss
            # auszugleichen) - siehe Docstring oben.
            div -= vertical_flux_term.astype(np.float32)

        inv_lambda = self._get_poisson_inv_eigs(height, width)
        div_hat = dctn(div, type=2, norm='ortho')
        div_hat[0, 0] = 0.0  # Mittelwert-Freiheitsgrad von phi, siehe Docstring
        phi = idctn((div_hat * inv_lambda).astype(np.float32), type=2, norm='ortho')

        grad_x = np.zeros((height, width), dtype=np.float32)
        grad_y = np.zeros((height, width), dtype=np.float32)
        grad_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) * 0.5
        grad_x[:, 0] = (phi[:, 1] - phi[:, 0]) * 0.5
        grad_x[:, -1] = (phi[:, -1] - phi[:, -2]) * 0.5
        grad_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) * 0.5
        grad_y[0, :] = (phi[1, :] - phi[0, :]) * 0.5
        grad_y[-1, :] = (phi[-1, :] - phi[-2, :]) * 0.5

        wind_field[:, :, 0] -= grad_x
        wind_field[:, :, 1] -= grad_y

    def _transport_moisture_simple(self, humid_map: np.ndarray, wind_field: np.ndarray,
                                  dt: float = 0.5) -> np.ndarray:
        """
        Simplified Moisture-Transport durch Advection.

        Vektorisiert via scipy.ndimage.map_coordinates statt der früheren Python-
        Doppelschleife mit manueller bilinearer Interpolation - identische Semi-
        Lagrange-Rückwärts-Sample-Formel (source = position - wind*dt*skalierung,
        bilinear an der Quellposition gesampelt), nur vektorisiert. Das ist zugleich
        die allgemeine Advektions-Primitive, die der neue 3-Schicht-CFD-Zeitschritt-
        Loop für Wind/Temperatur/Feuchte gleichermaßen braucht (siehe
        [[project-3layer-wind-cfd]]) - keine Wegwerf-Arbeit. mode='nearest' entspricht
        dem alten expliziten np.clip auf die Array-Grenzen. Randzeilen/-spalten bleiben
        wie im Original unverändert (Schleife begann bei range(1, height-1)).
        """
        height, width = humid_map.shape
        y_idx, x_idx = np.mgrid[0:height, 0:width].astype(np.float64)

        wind_x = wind_field[:, :, 0] * dt * 0.1  # Scaled für Stabilität
        wind_y = wind_field[:, :, 1] * dt * 0.1

        source_x = np.clip(x_idx - wind_x, 0, width - 1)
        source_y = np.clip(y_idx - wind_y, 0, height - 1)

        sampled = map_coordinates(humid_map, [source_y, source_x], order=1, mode='nearest')

        transported_humid = humid_map.copy()
        transported_humid[1:-1, 1:-1] = sampled[1:-1, 1:-1]
        return transported_humid.astype(humid_map.dtype)

    def _apply_humidity_diffusion(self, humid_map: np.ndarray, iterations: int = 2,
                                   hardness_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Humidity-Diffusion für natural Distribution - echte scipy.ndimage.gaussian_filter-
        Diffusion statt des früheren handgerollten 4-Nachbar-Box-Blurs (identisches
        iterations-Argument beibehalten, nur als Sigma-Skalierung statt Loop-Zähler
        genutzt - Iterations-Anzahl skalierte vorher grob die Diffusionsstärke,
        das übernimmt jetzt Sigma direkt).

        hardness_map (optional): normiert auf [0,1] und pixelweise als Blend-Gewicht
        genutzt - result = hardness_norm*original + (1-hardness_norm)*diffused. Weiches
        Gestein (niedrige hardness) lässt Feuchte stärker zu den Nachbarn diffundieren
        (mehr diffundierter Anteil), hartes Gestein hält Feuchte lokal fester (mehr
        Original-Anteil) - "sedimentation verteilt das besser als harter Stein"
        (Nutzer-Feedback). Ohne hardness_map (z.B. allererster Weather-Lauf, bevor
        Geology überhaupt etwas geliefert hat) wird gleichmäßig vollständig diffundiert.
        """
        sigma = 1.5 * max(1, iterations)
        diffused_map = gaussian_filter(humid_map, sigma=sigma)

        if hardness_map is None:
            return diffused_map

        h_min, h_max = float(hardness_map.min()), float(hardness_map.max())
        if h_max - h_min > 1e-6:
            hardness_norm = (hardness_map - h_min) / (h_max - h_min)
        else:
            hardness_norm = np.zeros_like(hardness_map)

        return hardness_norm * humid_map + (1.0 - hardness_norm) * diffused_map

    def _calculate_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Erstellt Parameter-Hash für Cache-Management
        """
        import hashlib

        # Sortierte Parameter für consistent Hashing
        sorted_params = sorted(parameters.items())
        param_string = str(sorted_params) + str(self.map_seed)

        return hashlib.md5(param_string.encode()).hexdigest()[:16]

    def _update_progress(self, phase: str, progress: int, message: str):
        """
        Progress-Update für UI-Integration
        """
        if self.progress_callback:
            self.progress_callback(phase, progress, message)
        else:
            self.logger.debug(f"Weather Progress [{progress}%]: {phase} - {message}")

    def update_seed(self, new_seed: int):
        """
        Aktualisiert Seed für alle Weather-Komponenten
        """
        if new_seed != self.map_seed:
            self.map_seed = new_seed
            self.noise_generator = OpenSimplex(seed=new_seed)
            self.logger.debug(f"Weather seed updated to {new_seed}")


# ===== SUB-KOMPONENTEN (modernisiert) =====

class TemperatureCalculator:
    """
    Modernisierte Temperature-Calculation mit GPU-Shader-Integration
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TemperatureCalculator")

    def calculate_temperature_with_orographic_effects(self, heightmap: np.ndarray,
                                                    shadowmap: np.ndarray,
                                                    parameters: Dict[str, Any]) -> np.ndarray:
        """
        Temperature-Calculation mit erweiterten orographischen Effekten
        """
        # Alle Temperature-Komponenten integrieren
        temp_map = self._calculate_base_temperature(heightmap, shadowmap, parameters)
        temp_map = self._apply_orographic_effects(temp_map, heightmap, parameters)
        temp_map = self._apply_latitude_gradient(temp_map, heightmap.shape)

        return temp_map

    def _calculate_base_temperature(self, heightmap: np.ndarray, shadowmap: np.ndarray,
                                   parameters: Dict[str, Any]) -> np.ndarray:
        """Base Temperature mit Altitude-Cooling und Solar-Heating"""
        # Basis-Temperatur
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Altitude-Cooling (Parameter ist °C/km)
        altitude_cooling = parameters['altitude_cooling'] / 1000.0
        temp_map -= heightmap * altitude_cooling

        # Solar-Heating aus Shadowmap
        if len(shadowmap.shape) == 3:
            # Multi-Angle Shadows - gewichtete Kombination
            shadow_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1, 0.05, 0.05])  # Mittag stärker gewichtet
            shadow_combined = np.average(shadowmap, axis=2, weights=shadow_weights[:shadowmap.shape[2]])
        else:
            shadow_combined = shadowmap

        solar_effect = (shadow_combined - 0.5) * parameters['solar_power']
        temp_map += solar_effect

        return temp_map

    def _apply_orographic_effects(self, temp_map: np.ndarray, heightmap: np.ndarray,
                                 parameters: Dict[str, Any]) -> np.ndarray:
        """Erweiterte orographische Temperature-Effects"""
        # Valley-Inversion-Effect (Täler kühler bei hohen Lagen)
        height, width = heightmap.shape

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                current_elevation = heightmap[y, x]

                # Nachbar-Elevations
                neighbor_elevations = [
                    heightmap[y-1, x], heightmap[y+1, x],
                    heightmap[y, x-1], heightmap[y, x+1]
                ]
                max_neighbor = max(neighbor_elevations)

                # Valley-Detection und Temperature-Inversion
                if current_elevation < max_neighbor - 50:  # 50m tiefer als Nachbarn
                    valley_effect = -2.0  # 2°C kühler in Tälern
                    temp_map[y, x] += valley_effect

        return temp_map

    def _apply_latitude_gradient(self, temp_map: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Latitude-Gradient: Nord-Süd Temperature-Variation"""
        height, width = shape

        # 5°C Unterschied von Süd (y=0) zu Nord (y=height)
        for y in range(height):
            latitude_factor = (y / (height - 1)) * 5.0
            temp_map[y, :] += latitude_factor

        return temp_map


class WindFieldSimulator:
    """
    CFD-basierte Wind-Simulation mit Navier-Stokes-Approximation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".WindFieldSimulator")

    def simulate_cfd_wind_field(self, heightmap: np.ndarray, temp_map: np.ndarray,
                               parameters: Dict[str, Any], iterations: int) -> np.ndarray:
        """
        Full CFD-Simulation mit Navier-Stokes-Equations
        """
        # Implementation würde hier folgen - der bestehende Code ist bereits eine gute Basis
        pass


class PrecipitationSystem:
    """
    Erweiterte Precipitation mit Orographic-Enhancement
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".PrecipitationSystem")


class AtmosphericMoistureManager:
    """
    Atmospheric-Moisture mit Magnus-Formel und Wind-Enhancement
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".AtmosphericMoistureManager")


# ===== LEGACY COMPATIBILITY =====

def generate_weather_system(heightmap, shade_map, soil_moist_map, air_temp_entry, solar_power,
                           altitude_cooling, thermic_effect, wind_speed_factor, terrain_factor,
                           flow_direction=None, flow_accumulation=None, map_seed=None):
    """
    Legacy-Kompatibilität für alte API
    """
    generator = WeatherSystemGenerator(map_seed=map_seed or 42)

    parameters = {
        'air_temp_entry': air_temp_entry,
        'solar_power': solar_power,
        'altitude_cooling': altitude_cooling,
        'thermic_effect': thermic_effect,
        'wind_speed_factor': wind_speed_factor,
        'terrain_factor': terrain_factor
    }

    weather_data = generator.calculate_weather_system(heightmap, shade_map, parameters, 3)

    # Legacy-Format zurückgeben (Tuple)
    return weather_data.wind_map, weather_data.temp_map, weather_data.precip_map, weather_data.humid_map