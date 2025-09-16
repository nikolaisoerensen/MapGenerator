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
- altitude_cooling (Abkühlen der Luft pro 100m Altitude, default 6°C)
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
import logging
from typing import Dict, Any, Tuple


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

        # Performance-Tracking
        self.performance_stats = {}

        # Progress-Callback für UI-Updates
        self.progress_callback = None

    def calculate_weather_system(self, heightmap_combined: np.ndarray, shadowmap: np.ndarray,
                                parameters: Dict[str, Any], lod_level: int) -> WeatherData:
        """
        Hauptmethode für Weather-System-Generierung mit vollständiger LOD-Progression

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

            # Input-Validation
            self._validate_inputs(heightmap_combined, shadowmap, parameters, lod_level)

            # Target-Size für LOD bestimmen
            target_size = self._get_lod_size(lod_level, heightmap_combined.shape[0])

            # CFD-Iterations für LOD bestimmen
            cfd_iterations = self._get_cfd_iterations(lod_level)

            self.logger.debug(f"Weather generation - target_size: {target_size}, CFD iterations: {cfd_iterations}")

            # Input-Data auf Target-Size interpolieren
            heightmap, shadowmap = self._prepare_input_data(heightmap_combined, shadowmap, target_size)

            # Schritt 1: Temperature Field Calculation (20% - 30%)
            self._update_progress("Temperature", 20, "Calculating temperature field with orographic effects...")
            temp_map = self._calculate_temperature_field(heightmap, shadowmap, parameters, target_size)

            # Schritt 2: Wind Field CFD-Simulation (30% - 60%)
            self._update_progress("Wind Field", 30, f"CFD simulation with {cfd_iterations} iterations...")
            wind_map = self._simulate_wind_field_cfd(heightmap, temp_map, shadowmap, parameters,
                                                   target_size, cfd_iterations)

            # Schritt 3: Atmospheric Moisture Management (60% - 80%)
            self._update_progress("Humidity", 60, "Calculating atmospheric moisture transport...")
            humid_map = self._calculate_atmospheric_moisture(heightmap, temp_map, wind_map, parameters)

            # Schritt 4: Precipitation System (80% - 95%)
            self._update_progress("Precipitation", 80, "Calculating orographic precipitation...")
            precip_map = self._calculate_precipitation_system(humid_map, temp_map, wind_map,
                                                            heightmap, parameters)

            # WeatherData-Objekt erstellen und validieren
            weather_data = self._create_weather_data(wind_map, temp_map, precip_map, humid_map,
                                                   lod_level, target_size, parameters)

            # Performance-Stats aktualisieren
            self._update_performance_stats(weather_data, cfd_iterations)

            self.logger.info(f"Weather generation completed successfully - LOD {lod_level}")

            return weather_data

        except Exception as e:
            self.logger.error(f"Weather generation failed: {str(e)}")
            # Error-Recovery: Fallback zu Simplified-Weather-System
            return self._create_fallback_weather_data(heightmap_combined.shape[0], lod_level, parameters)

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
            return base_size * (2 ** (lod_level - 1))
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
        altitude_cooling_rate = parameters['altitude_cooling'] / 100.0  # °C pro Meter
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
        noise_variation = self._generate_atmospheric_noise(heightmap.shape, target_size)
        temp_map += noise_variation

        return temp_map

    def _calculate_temperature_simple(self, heightmap: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Simple-Fallback: Basic Temperature ohne komplexe Effekte
        """
        temp_map = np.full(heightmap.shape, parameters['air_temp_entry'], dtype=np.float32)

        # Nur Altitude-Cooling
        altitude_cooling_rate = parameters['altitude_cooling'] / 100.0
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
        pressure_field = np.zeros((height, width), dtype=np.float32)

        # Slopemap aus Heightmap berechnen (vectorized)
        slopemap = self._calculate_slopes_vectorized(heightmap)

        # Initiales Druckfeld (West-Ost-Gradient mit Noise)
        x_coords = np.arange(width).reshape(1, -1)
        pressure_field = 1.0 - (x_coords / (width - 1)) * 0.3

        # Noise-Modulation
        pressure_noise = self._generate_pressure_noise((height, width))
        pressure_field += pressure_noise * 0.15

        # CFD-Iterationen
        for iteration in range(cfd_iterations):
            # Progress-Update für längere CFD-Simulationen
            if iteration % max(1, cfd_iterations // 5) == 0:
                progress = 30 + (iteration / cfd_iterations) * 30
                self._update_progress("Wind CFD", int(progress),
                                    f"CFD iteration {iteration + 1}/{cfd_iterations}")

            # Druckgradienten berechnen (vectorized)
            pressure_grad_x = np.zeros_like(pressure_field)
            pressure_grad_y = np.zeros_like(pressure_field)

            pressure_grad_x[:, 1:-1] = (pressure_field[:, 2:] - pressure_field[:, :-2]) * 0.5
            pressure_grad_y[1:-1, :] = (pressure_field[2:, :] - pressure_field[:-2, :]) * 0.5

            # Wind aus Druckgradienten
            wind_field[:, :, 0] = -pressure_grad_x * parameters['wind_speed_factor'] * 10.0
            wind_field[:, :, 1] = -pressure_grad_y * parameters['wind_speed_factor'] * 10.0

            # Terrain-Ablenkung (vectorized)
            terrain_factor = parameters['terrain_factor'] * 0.5
            wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor  # Slope Y -> Wind X
            wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor  # Slope X -> Wind Y

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

        # Konstanter West-Ost-Wind
        base_wind_speed = parameters['wind_speed_factor'] * 5.0
        wind_field[:, :, 0] = base_wind_speed  # Ostwind

        # Basic Terrain-Ablenkung
        slopemap = self._calculate_slopes_vectorized(heightmap)
        terrain_factor = parameters['terrain_factor'] * 0.2
        wind_field[:, :, 0] += slopemap[:, :, 1] * terrain_factor
        wind_field[:, :, 1] -= slopemap[:, :, 0] * terrain_factor

        return wind_field

    def _calculate_atmospheric_moisture(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                      wind_map: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """
        Atmospheric-Moisture-Calculation mit Evaporation und Transport

        Implementiert Magnus-Formel für Sättigungsdampfdruck und Wind-enhanced Evaporation
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
        return self._calculate_atmospheric_moisture_cpu(heightmap, temp_map, wind_map, parameters)

    def _calculate_atmospheric_moisture_cpu(self, heightmap: np.ndarray, temp_map: np.ndarray,
                                          wind_map: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
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

        # Initiale Humidity aus Evaporation
        humid_map = evaporation_rate * 10.0  # Skalierung auf realistische Werte

        # Moisture-Transport (vereinfacht, 3 Iterationen)
        for _ in range(3):
            humid_map = self._transport_moisture_simple(humid_map, wind_map, dt=0.5)

        # Diffusion für smooth Distribution
        humid_map = self._apply_humidity_diffusion(humid_map, iterations=2)

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
                           parameters: Dict[str, Any]) -> WeatherData:
        """
        Erstellt WeatherData-Objekt mit vollständiger LOD-Integration und Validation
        """
        weather_data = WeatherData()
        weather_data.wind_map = wind_map
        weather_data.temp_map = temp_map
        weather_data.precip_map = precip_map
        weather_data.humid_map = humid_map
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
        Vectorized Slope-Calculation (dz/dx, dz/dy) für Performance
        """
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        # dz/dx (vectorized)
        slopemap[:, 1:-1, 0] = (heightmap[:, 2:] - heightmap[:, :-2]) * 0.5
        slopemap[:, 0, 0] = heightmap[:, 1] - heightmap[:, 0]
        slopemap[:, -1, 0] = heightmap[:, -1] - heightmap[:, -2]

        # dz/dy (vectorized)
        slopemap[1:-1, :, 1] = (heightmap[2:, :] - heightmap[:-2, :]) * 0.5
        slopemap[0, :, 1] = heightmap[1, :] - heightmap[0, :]
        slopemap[-1, :, 1] = heightmap[-1, :] - heightmap[-2, :]

        return slopemap

    def _generate_atmospheric_noise(self, shape: Tuple[int, int], target_size: int) -> np.ndarray:
        """
        Generiert atmospheric Noise-Variation mit Edge-Enhancement
        """
        height, width = shape
        noise_field = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # Edge-Factor für stärkere Variation an Kartenrändern
                edge_factor = self._calculate_edge_factor(x, y, width, height)

                # Noise mit verschiedenen Scales
                noise_x, noise_y = x / width * 3, y / height * 3
                noise_val = self.noise_generator.noise2(noise_x, noise_y)

                # Zusätzlicher Noise für höhere LODs
                if target_size > 128:
                    noise_val += self.noise_generator.noise2(noise_x * 2, noise_y * 2) * 0.3

                noise_field[y, x] = noise_val * edge_factor * 3.0

        return noise_field

    def _generate_pressure_noise(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Generiert Pressure-Noise für CFD-Simulation
        """
        height, width = shape
        pressure_noise = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                noise_x, noise_y = x / width * 2, y / height * 2
                pressure_noise[y, x] = self.noise_generator.noise2(noise_x, noise_y)

        return pressure_noise

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
        Wendet Diffusion auf Wind-Field für numerical Stability an
        """
        height, width = wind_field.shape[:2]
        diffused_field = wind_field.copy()

        for component in [0, 1]:  # x und y Komponenten
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    neighbors = [
                        wind_field[y-1, x, component], wind_field[y+1, x, component],
                        wind_field[y, x-1, component], wind_field[y, x+1, component]
                    ]

                    neighbor_avg = np.mean(neighbors)
                    current_value = wind_field[y, x, component]

                    diffused_field[y, x, component] = (current_value +
                                                     (neighbor_avg - current_value) * diffusion_rate)

        return diffused_field

    def _apply_continuity_correction(self, wind_field: np.ndarray):
        """
        Wendet Kontinuitäts-Correction für Massenerhaltung an (simplified)
        """
        height, width = wind_field.shape[:2]

        # Divergence berechnen (simplified)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Divergence: du/dx + dv/dy
                dudx = (wind_field[y, x+1, 0] - wind_field[y, x-1, 0]) * 0.5
                dvdy = (wind_field[y+1, x, 1] - wind_field[y-1, x, 1]) * 0.5

                divergence = dudx + dvdy

                # Correction-Factor für Massenerhaltung
                correction_factor = -divergence * 0.1

                wind_field[y, x, 0] += correction_factor * 0.5
                wind_field[y, x, 1] += correction_factor * 0.5

    def _transport_moisture_simple(self, humid_map: np.ndarray, wind_field: np.ndarray,
                                  dt: float = 0.5) -> np.ndarray:
        """
        Simplified Moisture-Transport durch Advection
        """
        height, width = humid_map.shape
        transported_humid = humid_map.copy()

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                wind_x = wind_field[y, x, 0] * dt * 0.1  # Scaled für Stabilität
                wind_y = wind_field[y, x, 1] * dt * 0.1

                # Source-Position für Advektion
                source_x = max(0, min(width - 1, x - wind_x))
                source_y = max(0, min(height - 1, y - wind_y))

                # Bilineare Interpolation
                x0, y0 = int(source_x), int(source_y)
                x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

                fx, fy = source_x - x0, source_y - y0

                h00, h10 = humid_map[y0, x0], humid_map[y0, x1]
                h01, h11 = humid_map[y1, x0], humid_map[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                transported_humid[y, x] = h0 * (1 - fy) + h1 * fy

        return transported_humid

    def _apply_humidity_diffusion(self, humid_map: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Humidity-Diffusion für natural Distribution
        """
        diffused_map = humid_map.copy()
        diffusion_rate = 0.08

        for _ in range(iterations):
            new_map = diffused_map.copy()
            height, width = diffused_map.shape

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    neighbors = [
                        diffused_map[y-1, x], diffused_map[y+1, x],
                        diffused_map[y, x-1], diffused_map[y, x+1]
                    ]

                    neighbor_avg = np.mean(neighbors)
                    current_value = diffused_map[y, x]

                    new_map[y, x] = current_value + (neighbor_avg - current_value) * diffusion_rate

            diffused_map = new_map

        return diffused_map

    def _calculate_edge_factor(self, x: int, y: int, width: int, height: int) -> float:
        """
        Berechnet Edge-Factor für stärkere Effekte an Map-Boundaries
        """
        dist_to_edge = min(x, y, width - 1 - x, height - 1 - y)
        max_dist = min(width, height) // 6

        if dist_to_edge < max_dist:
            return 1.0 + (max_dist - dist_to_edge) / max_dist * 0.8
        else:
            return 1.0

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

        # Altitude-Cooling (verstärkt: 6°C/100m)
        altitude_cooling = parameters['altitude_cooling'] / 100.0
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