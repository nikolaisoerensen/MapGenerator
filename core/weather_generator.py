"""
Path: core/weather_generator.py

Funktionsweise: Dynamisches Wetter- und Feuchtigkeitssystem
- Terrain-basierte Temperaturberechnung (Altitude, Sonneneinstrahlung, Breitengrad (Input: heightmap, shade_map) der Luft.
- Windfeld-Simulation mit Ablenkung um Berge und Abbremsen durch Geländegradienten-Analyse (slopemap).
    - Konvektion: Wärmetransport durch Windströme
    - Orographischer Regen: Feuchtigkeitstransport
    - Precipation durch Abkühlen/Aufsteigen der Luft
    - Evaporation durch Bodenfeuchtigkeit (Input: soil_moist_map)
    - OpenSimplex-Noise für  Variation in Temperatur-, Feuchte- und Luftdruck an Kartengrenze
    - Regen wird ausgelöst durch Erreichen eines Feuchtigkeitsgrenzwertes von über 1.0. Dieser berechnet sich aus rho_max = 5*exp(0.06*T) mit T aus temp_map und der Luftfeuchtigkeit in der jeweiligen Zelle.

Parameter Input:
- air_temp_entry (Lufttemperatur bei Karteneintritt)
- solar_power (max. solare Gewinne (default 20°C))
- altitude_cooling (Abkühlen der Luft pro 100 m Altitude(default 6°C))
- thermic_effect (Thermische Verformung der Windvektoren durch shademap)
- wind_speed_factor (Windgeschwindigkeit je Luftdruckdifferenz)
- terrain_factor (Einfluss von Terrain auf Wind und Temperatur)
- flow_direction, flow_accumulation (optional: Wasserfluss-Informationen)

data_manager Input:
- map_seed
- heightmap
- shade_map
- soil_moist_map

Output:
- wind_map (Windvektoren (Richtung und Geschwindigkeit)) in m/s
- temp_map (2D-Feld, Lufttemperatur) in °C
- precip_map (2D-Feld, Regenfall) in gH20/m2
- humid_map (2D-Feld, Luftfeuchte) in gH20/m3

Klassen:
WeatherSystemGenerator
    Funktionsweise: Hauptklasse für dynamisches Wetter- und Feuchtigkeitssystem
    Aufgabe: Koordiniert Temperatur, Wind, Niederschlag und Luftfeuchtigkeit
    Methoden: generate_weather_system(), update_weather_cycle(), integrate_atmospheric_effects()

TemperatureCalculator
    Funktionsweise: Berechnet Lufttemperatur basierend auf Altitude, Sonneneinstrahlung und Breitengrad
    Aufgabe: Erstellt temp_map als Grundlage für alle Weather-Berechnungen
    Methoden: calculate_altitude_cooling(), apply_solar_heating(), add_latitude_gradient()

WindFieldSimulator
    Funktionsweise: Simuliert Windfeld mit Ablenkung um Berge und Geländegradienten-Analyse
    Aufgabe: Erstellt wind_map für Konvektion und Orographischen Regen
    Methoden: simulate_pressure_gradients(), apply_terrain_deflection(), calculate_thermal_effects()

PrecipitationSystem
    Funktionsweise: Berechnet Niederschlag durch Feuchtigkeits-Transport und Kondensation
    Aufgabe: Erstellt precip_map basierend auf Luftfeuchtigkeit und Geländeeffekten
    Methoden: calculate_orographic_precipitation(), simulate_moisture_transport(), trigger_precipitation_events()

AtmosphericMoistureManager
    Funktionsweise: Verwaltet Luftfeuchtigkeit durch Evaporation und Konvektion
    Aufgabe: Erstellt humid_map mit realistischer Feuchtigkeits-Verteilung
    Methoden: calculate_evaporation(), transport_moisture(), apply_humidity_diffusion()
"""

import numpy as np
from opensimplex import OpenSimplex
from core.base_generator import BaseGenerator

class WeatherData:
    """
    Funktionsweise: Container für alle Weather-Daten mit Metainformationen
    Aufgabe: Speichert wind_map, temp_map, precip_map, humid_map und LOD-Informationen
    """
    def __init__(self):
        self.wind_map = None      # (height, width, 2) - Windvektoren in m/s
        self.temp_map = None      # (height, width) - Temperatur in °C
        self.precip_map = None    # (height, width) - Niederschlag in gH2O/m²
        self.humid_map = None     # (height, width) - Luftfeuchtigkeit in gH2O/m³
        self.lod_level = "LOD64"  # Aktueller LOD-Level
        self.actual_size = 64     # Tatsächliche Kartengröße
        self.parameters = {}      # Verwendete Parameter für Cache-Management

class TemperatureCalculator:
    """
    Funktionsweise: Berechnet Lufttemperatur basierend auf Altitude, Sonneneinstrahlung und Breitengrad
    Aufgabe: Erstellt temp_map als Grundlage für alle Weather-Berechnungen
    """

    def __init__(self, air_temp_entry=15.0, solar_power=20.0, altitude_cooling=6.0):
        """
        Funktionsweise: Initialisiert Temperatur-Calculator mit Standard-Parametern
        Aufgabe: Setup der Temperatur-Berechnungs-Parameter
        Parameter: air_temp_entry, solar_power, altitude_cooling - Temperatur-Parameter
        """
        self.air_temp_entry = air_temp_entry
        self.solar_power = solar_power
        self.altitude_cooling = altitude_cooling  # °C pro 100m

    def calculate_altitude_cooling(self, heightmap):
        """
        Funktionsweise: Berechnet Temperaturabnahme durch Altitude (60°C pro km - 10x verstärkt)
        Aufgabe: Dramatische Höhen-Temperatur-Effekte für das System
        Parameter: heightmap (numpy.ndarray) - Höhendaten
        Returns: numpy.ndarray - Temperatur-Reduktion durch Altitude
        """
        # 60°C pro 1000m = 6°C pro 100m (10x verstärkt gegenüber real)
        altitude_temp_reduction = heightmap * (self.altitude_cooling / 100.0)
        return altitude_temp_reduction

    def apply_solar_heating(self, shade_map):
        """
        Funktionsweise: Wendet solare Erwärmung basierend auf Verschattung an
        Aufgabe: Sonneneinstrahlung beeinflusst lokale Temperatur
        Parameter: shade_map (numpy.ndarray) - Verschattungsdaten (0=Schatten, 1=Sonne)
        Returns: numpy.ndarray - Temperatur-Änderung durch Sonneneinstrahlung
        """
        # Shade-Map: 0 (Schatten) bis 1 (volle Sonne)
        # Bei voller Sonne: +solar_power/2, bei Schatten: -solar_power/2
        solar_temp_change = (shade_map - 0.5) * self.solar_power
        return solar_temp_change

    def add_latitude_gradient(self, map_shape):
        """
        Funktionsweise: Fügt Breitengrad-Gradienten hinzu (Äquator-zu-Pol-Simulation)
        Aufgabe: Nord-Süd Temperatur-Gradient über die Karte
        Parameter: map_shape (tuple) - Dimensionen der Karte
        Returns: numpy.ndarray - Temperatur-Gradient von Süd nach Nord
        """
        height, width = map_shape
        latitude_gradient = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            # y=0 (Süd) = 0°C Änderung, y=height (Nord) = +5°C
            latitude_temp_change = (y / (height - 1)) * 5.0
            latitude_gradient[y, :] = latitude_temp_change

        return latitude_gradient

    def calculate_base_temperature(self, heightmap, shade_map, noise_variation):
        """
        Funktionsweise: Berechnet Basis-Temperaturfeld aus allen Komponenten
        Aufgabe: Kombiniert Altitude, Solar und Latitude-Effekte zu temp_map
        Parameter: heightmap, shade_map, noise_variation - Alle Temperatur-Eingaben
        Returns: numpy.ndarray - Basis-Temperaturfeld
        """
        # Basis-Temperatur
        base_temp = np.full(heightmap.shape, self.air_temp_entry, dtype=np.float32)

        # Altitude-Cooling anwenden
        altitude_reduction = self.calculate_altitude_cooling(heightmap)
        base_temp -= altitude_reduction

        # Solar-Heating anwenden
        solar_change = self.apply_solar_heating(shade_map)
        base_temp += solar_change

        # Latitude-Gradient anwenden
        latitude_change = self.add_latitude_gradient(heightmap.shape)
        base_temp += latitude_change

        # Noise-Variation anwenden
        base_temp += noise_variation

        return base_temp


class WindFieldSimulator:
    """
    Funktionsweise: Simuliert Windfeld mit Ablenkung um Berge und Geländegradienten-Analyse
    Aufgabe: Erstellt wind_map für Konvektion und Orographischen Regen
    """

    def __init__(self, wind_speed_factor=1.0, terrain_factor=1.0, thermic_effect=1.0):
        """
        Funktionsweise: Initialisiert Wind-Simulator mit Geschwindigkeits- und Terrain-Faktoren
        Aufgabe: Setup der Wind-Simulations-Parameter
        Parameter: wind_speed_factor, terrain_factor, thermic_effect - Wind-Parameter
        """
        self.wind_speed_factor = wind_speed_factor
        self.terrain_factor = terrain_factor
        self.thermic_effect = thermic_effect

    def simulate_pressure_gradients(self, map_shape, noise_generator):
        """
        Funktionsweise: Simuliert Druckgradienten von West nach Ost mit Noise-Modulation
        Aufgabe: Grundlegendes Windfeld durch Druckdifferenzen
        Parameter: map_shape, noise_generator - Map-Dimensionen und Noise-Generator
        Returns: numpy.ndarray - Druckfeld für Wind-Berechnung
        """
        height, width = map_shape
        pressure_field = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # Basis-Druckgradient: West (hoch) nach Ost (niedrig)
                base_pressure = 1.0 - (x / (width - 1)) * 0.3

                # Noise-Modulation für Turbulenz
                noise_x = x / width * 4.0
                noise_y = y / height * 4.0
                noise_variation = noise_generator.noise2(noise_x, noise_y) * 0.15

                pressure_field[y, x] = base_pressure + noise_variation

        return pressure_field

    def apply_terrain_deflection(self, pressure_field, slopemap, heightmap):
        """
        Funktionsweise: Wendet Terrain-Ablenkung auf Windfeld an (Düsen-/Blockierungs-Effekte)
        Aufgabe: Realistische Wind-Terrain-Interaktion
        Parameter: pressure_field, slopemap, heightmap - Druck, Slopes und Höhen
        Returns: numpy.ndarray - Wind-Vektoren (vx, vy) in m/s
        """
        height, width = pressure_field.shape
        wind_field = np.zeros((height, width, 2), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # Druckgradient berechnen
                wind_x = 0.0
                wind_y = 0.0

                # X-Richtung (West-Ost Gradient)
                if x > 0 and x < width - 1:
                    pressure_grad_x = (pressure_field[y, x + 1] - pressure_field[y, x - 1]) * 0.5
                    wind_x = -pressure_grad_x * self.wind_speed_factor * 10.0

                # Y-Richtung (Nord-Süd Gradient)
                if y > 0 and y < height - 1:
                    pressure_grad_y = (pressure_field[y + 1, x] - pressure_field[y - 1, x]) * 0.5
                    wind_y = -pressure_grad_y * self.wind_speed_factor * 10.0

                # Terrain-Deflection anwenden
                if 0 <= y < height and 0 <= x < width:
                    # Slope-basierte Ablenkung
                    slope_x = slopemap[y, x, 0] if x < slopemap.shape[1] else 0
                    slope_y = slopemap[y, x, 1] if y < slopemap.shape[0] else 0

                    # Wind wird um Berghänge abgelenkt
                    deflection_factor = self.terrain_factor * 0.5
                    wind_x += slope_y * deflection_factor  # Slope in Y-Richtung lenkt Wind in X ab
                    wind_y -= slope_x * deflection_factor  # Slope in X-Richtung lenkt Wind in Y ab

                    # Düseneffekt in Tälern, Abbremsung an Bergen
                    elevation = heightmap[y, x]
                    elevation_factor = 1.0 - (elevation / 1000.0) * 0.3  # Bremsung in Höhe
                    wind_x *= max(0.2, elevation_factor)
                    wind_y *= max(0.2, elevation_factor)

                wind_field[y, x, 0] = wind_x
                wind_field[y, x, 1] = wind_y

        return wind_field

    def calculate_thermal_effects(self, wind_field, temp_map, shade_map):
        """
        Funktionsweise: Berechnet thermische Konvektion und modifiziert Windfeld
        Aufgabe: Thermisch induzierte Windkomponenten durch Temperaturunterschiede
        Parameter: wind_field, temp_map, shade_map - Wind, Temperatur und Verschattung
        Returns: numpy.ndarray - Modifiziertes Windfeld mit thermischen Effekten
        """
        height, width = wind_field.shape[:2]
        thermal_wind = np.copy(wind_field)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # Temperaturgradienten berechnen
                temp_grad_x = (temp_map[y, x + 1] - temp_map[y, x - 1]) * 0.5
                temp_grad_y = (temp_map[y + 1, x] - temp_map[y - 1, x]) * 0.5

                # Thermische Konvektion
                # Warme Bereiche erzeugen Aufwinde, kalte Abwinde
                local_temp = temp_map[y, x]
                avg_temp = np.mean(temp_map)
                temp_diff = local_temp - avg_temp

                # Konvektive Windkomponente
                convection_strength = temp_diff * self.thermic_effect * 0.1

                # Hangwind-Effekte
                shade_influence = (shade_map[y, x] - 0.5) * self.thermic_effect * 0.2

                # Thermische Modifikation anwenden
                thermal_wind[y, x, 0] += temp_grad_x * 0.1 + convection_strength
                thermal_wind[y, x, 1] += temp_grad_y * 0.1 + shade_influence

        return thermal_wind


class AtmosphericMoistureManager:
    """
    Funktionsweise: Verwaltet Luftfeuchtigkeit durch Evaporation und Konvektion
    Aufgabe: Erstellt humid_map mit realistischer Feuchtigkeits-Verteilung
    """

    def __init__(self):
        """
        Funktionsweise: Initialisiert Atmospheric-Moisture-Manager
        Aufgabe: Setup der Feuchtigkeits-Verwaltung
        """
        pass

    def calculate_evaporation(self, soil_moist_map, temp_map, wind_field):
        """
        Funktionsweise: Berechnet Evaporation von Bodenfeuchtigkeit in Atmosphäre
        Aufgabe: Feuchtigkeitseintrag durch Verdunstung
        Parameter: soil_moist_map, temp_map, wind_field - Bodenfeuchtigkeit, Temperatur, Wind
        Returns: numpy.ndarray - Evaporations-Rate
        """
        height, width = soil_moist_map.shape
        evaporation_map = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                soil_moisture = soil_moist_map[y, x]
                temperature = temp_map[y, x]
                wind_speed = np.sqrt(wind_field[y, x, 0] ** 2 + wind_field[y, x, 1] ** 2)

                # Evaporation steigt mit Temperatur und Windgeschwindigkeit
                temp_factor = max(0, (temperature - 0) / 30.0)  # 0°C - 30°C Bereich
                wind_factor = min(2.0, wind_speed / 5.0)  # Wind bis 5 m/s berücksichtigt
                moisture_factor = soil_moisture / 100.0  # Soil moisture als Prozent

                evaporation_rate = moisture_factor * temp_factor * (1.0 + wind_factor) * 10.0
                evaporation_map[y, x] = evaporation_rate

        return evaporation_map

    def transport_moisture(self, humid_map, wind_field, dt=1.0):
        """
        Funktionsweise: Transportiert Luftfeuchtigkeit entsprechend Windfeld (Advektion)
        Aufgabe: Feuchtigkeits-Transport durch Wind
        Parameter: humid_map, wind_field, dt - Feuchtigkeit, Wind, Zeitschritt
        Returns: numpy.ndarray - Transportierte Feuchtigkeits-Verteilung
        """
        height, width = humid_map.shape
        new_humid_map = np.copy(humid_map)

        # Simplified advection
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                wind_x = wind_field[y, x, 0] * dt
                wind_y = wind_field[y, x, 1] * dt

                # Source-Position für Advektion
                source_x = x - wind_x
                source_y = y - wind_y

                # Bounds checking
                source_x = max(0, min(width - 1, source_x))
                source_y = max(0, min(height - 1, source_y))

                # Bilineare Interpolation für smooth transport
                x0, y0 = int(source_x), int(source_y)
                x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

                fx = source_x - x0
                fy = source_y - y0

                # Interpolation
                h00 = humid_map[y0, x0]
                h10 = humid_map[y0, x1]
                h01 = humid_map[y1, x0]
                h11 = humid_map[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx

                new_humid_map[y, x] = h0 * (1 - fy) + h1 * fy

        return new_humid_map

    def apply_humidity_diffusion(self, humid_map, iterations=3):
        """
        Funktionsweise: Wendet Diffusion auf Feuchtigkeitsfeld an für natürliche Verteilung
        Aufgabe: Glättet extreme Feuchtigkeits-Gradienten
        Parameter: humid_map, iterations - Feuchtigkeit und Anzahl Diffusions-Schritte
        Returns: numpy.ndarray - Diffused Humidity-Map
        """
        height, width = humid_map.shape
        diffused_map = np.copy(humid_map)

        diffusion_rate = 0.1

        for _ in range(iterations):
            new_map = np.copy(diffused_map)

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Nachbarn mitteln
                    neighbors = [
                        diffused_map[y - 1, x], diffused_map[y + 1, x],
                        diffused_map[y, x - 1], diffused_map[y, x + 1]
                    ]

                    neighbor_avg = np.mean(neighbors)
                    current_value = diffused_map[y, x]

                    # Diffusion anwenden
                    new_map[y, x] = current_value + (neighbor_avg - current_value) * diffusion_rate

            diffused_map = new_map

        return diffused_map


class PrecipitationSystem:
    """
    Funktionsweise: Berechnet Niederschlag durch Feuchtigkeits-Transport und Kondensation
    Aufgabe: Erstellt precip_map basierend auf Luftfeuchtigkeit und Geländeeffekten
    """

    def __init__(self):
        """
        Funktionsweise: Initialisiert Precipitation-System
        Aufgabe: Setup der Niederschlags-Berechnung
        """
        pass

    def calculate_orographic_precipitation(self, humid_map, wind_field, heightmap, slopemap):
        """
        Funktionsweise: Berechnet orographischen Niederschlag durch Luv-/Lee-Effekte
        Aufgabe: Bergbedingte Niederschlags-Verteilung
        Parameter: humid_map, wind_field, heightmap, slopemap - Feuchtigkeit, Wind, Terrain
        Returns: numpy.ndarray - Orographischer Niederschlag
        """
        height, width = humid_map.shape
        oro_precip = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                if x >= slopemap.shape[1] or y >= slopemap.shape[0]:
                    continue

                humidity = humid_map[y, x]
                wind_x = wind_field[y, x, 0]
                wind_y = wind_field[y, x, 1]
                wind_speed = np.sqrt(wind_x ** 2 + wind_y ** 2)

                # Slope in Windrichtung
                slope_x = slopemap[y, x, 0]
                slope_y = slopemap[y, x, 1]

                # Dot-Product: Wind-Richtung vs Slope-Richtung
                wind_slope_alignment = (wind_x * slope_x + wind_y * slope_y) / max(0.1, wind_speed)

                # Aufwinde an Luvhängen verstärken Niederschlag
                if wind_slope_alignment > 0:  # Wind gegen Hang
                    orographic_factor = wind_slope_alignment * wind_speed * 0.5
                    oro_precip[y, x] = humidity * orographic_factor * 0.1

        return oro_precip

    def simulate_moisture_transport(self, humid_map, temp_map):
        """
        Funktionsweise: Simuliert Feuchtigkeits-Kondensation basierend auf Temperatur
        Aufgabe: Temperaturbedingte Niederschlags-Auslösung
        Parameter: humid_map, temp_map - Feuchtigkeit und Temperatur
        Returns: numpy.ndarray - Kondensations-Niederschlag
        """
        height, width = humid_map.shape
        condensation_precip = np.zeros((height, width), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                humidity = humid_map[y, x]
                temperature = temp_map[y, x]

                # Sättigungsdampfdichte: rho_max = 5*exp(0.06*T)
                rho_max = 5.0 * np.exp(0.06 * temperature)

                # Relative Feuchtigkeit
                if rho_max > 0:
                    relative_humidity = humidity / rho_max

                    # Niederschlag bei Übersättigung (> 1.0)
                    if relative_humidity > 1.0:
                        excess_moisture = humidity - rho_max
                        condensation_precip[y, x] = excess_moisture * 0.8  # 80% fällt als Regen

        return condensation_precip

    def trigger_precipitation_events(self, humid_map, temp_map, oro_precip, condensation_precip):
        """
        Funktionsweise: Kombiniert alle Niederschlags-Komponenten zu finalem precip_map
        Aufgabe: Finale Niederschlags-Berechnung
        Parameter: humid_map, temp_map, oro_precip, condensation_precip - Alle Niederschlags-Inputs
        Returns: numpy.ndarray - Finaler Niederschlag in gH2O/m²
        """
        # Kombiniere orographischen und Kondensations-Niederschlag
        total_precip = oro_precip + condensation_precip

        # Minimum/Maximum Limits
        total_precip = np.maximum(0, total_precip)  # Kein negativer Niederschlag
        total_precip = np.minimum(200, total_precip)  # Maximum 200 gH2O/m²

        return total_precip

class WeatherSystemGenerator(BaseGenerator):
    """
    Funktionsweise: Hauptklasse für dynamisches Wetter- und Feuchtigkeitssystem mit BaseGenerator-API
    Aufgabe: Koordiniert Temperatur, Wind, Niederschlag und Luftfeuchtigkeit mit einheitlicher API
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Weather-System mit BaseGenerator und Sub-Komponenten
        Aufgabe: Setup aller Weather-Systeme und Noise-Generator
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Weather-Patterns
        """
        super().__init__(map_seed)
        self.noise_generator = OpenSimplex(seed=map_seed)

        # Standard-Parameter (werden durch _load_default_parameters überschrieben)
        self.air_temp_entry = 15.0
        self.solar_power = 20.0
        self.altitude_cooling = 6.0
        self.thermic_effect = 1.0
        self.wind_speed_factor = 1.0
        self.terrain_factor = 1.0

    def _load_default_parameters(self):
        """
        Funktionsweise: Lädt WEATHER-Parameter aus value_default.py
        Aufgabe: Standard-Parameter für Weather-Generierung
        Returns: dict - Alle Standard-Parameter für Weather
        """
        from gui.config.value_default import WEATHER

        return {
            'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
            'solar_power': WEATHER.SOLAR_POWER["default"],
            'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
            'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
            'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
            'terrain_factor': WEATHER.TERRAIN_FACTOR["default"]
        }

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Holt heightmap, shademap und erstellt initiale soil_moist_map aus DataManager
        Aufgabe: Dependency-Resolution für Weather-Generierung
        Parameter: data_manager - DataManager-Instanz
        Returns: dict - Benötigte Terrain-Daten
        """
        if not data_manager:
            raise Exception("DataManager required for Weather generation")

        # TerrainData-Objekt holen
        terrain_data = data_manager.get_terrain_data("complete")
        if terrain_data is None:
            # Fallback: Einzelne Arrays holen
            heightmap = data_manager.get_terrain_data("heightmap")
            shademap = data_manager.get_terrain_data("shadowmap")

            if heightmap is None:
                raise Exception("Heightmap dependency not available in DataManager")
            if shademap is None:
                raise Exception("Shadowmap dependency not available in DataManager")
        else:
            heightmap = terrain_data.heightmap
            shademap = terrain_data.shadowmap

            if heightmap is None:
                raise Exception("Heightmap not available in TerrainData")
            if shademap is None:
                raise Exception("Shadowmap not available in TerrainData")

        # Initiale soil_moist_map erstellen (alles trocken)
        soil_moist_map = np.zeros_like(heightmap, dtype=np.float32)

        self.logger.debug(f"Dependencies loaded - heightmap: {heightmap.shape}, shademap: {shademap.shape}")

        return {
            'heightmap': heightmap,
            'shademap': shademap,
            'soil_moist_map': soil_moist_map
        }

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Funktionsweise: Führt Weather-Generierung mit Progress-Updates aus
        Aufgabe: Kernlogik der Weather-Generierung mit allen 4 Hauptschritten
        Parameter: lod, dependencies, parameters
        Returns: WeatherData-Objekt mit allen Weather-Outputs
        """
        heightmap = dependencies['heightmap']
        shademap = dependencies['shademap']
        soil_moist_map = dependencies['soil_moist_map']

        # Parameter aktualisieren
        self.air_temp_entry = parameters['air_temp_entry']
        self.solar_power = parameters['solar_power']
        self.altitude_cooling = parameters['altitude_cooling']
        self.thermic_effect = parameters['thermic_effect']
        self.wind_speed_factor = parameters['wind_speed_factor']
        self.terrain_factor = parameters['terrain_factor']

        # LOD-Größe bestimmen
        target_size = self._get_lod_size(lod, heightmap.shape[0])

        # Heightmap auf Zielgröße interpolieren falls nötig
        if heightmap.shape[0] != target_size:
            heightmap = self._interpolate_array(heightmap, target_size)
            soil_moist_map = self._interpolate_array(soil_moist_map, target_size)

        # Shademap IMMER auf Zielgröße interpolieren (kommt von LOD64)
        if shademap.shape[0] != target_size:
            shademap = self._interpolate_array(shademap, target_size)

        # Slopemap aus Heightmap ableiten
        slopemap = self._calculate_slopes(heightmap)

        # Schritt 1: Temperature Calculation (20% - 25%)
        self._update_progress("Temperature", 20, "Calculating temperature field...")
        temp_map = self._calculate_temperature(heightmap, shademap, target_size)

        # Schritt 2: Wind Field Simulation (25% - 55%)
        self._update_progress("Wind Field", 25, "Simulating pressure gradients...")
        wind_map = self._simulate_wind_field(heightmap, slopemap, temp_map, shademap, target_size)

        # Schritt 3: Humidity Management (55% - 80%)
        self._update_progress("Humidity", 55, "Calculating evaporation and moisture transport...")
        humid_map = self._calculate_humidity(soil_moist_map, temp_map, wind_map)

        # Schritt 4: Precipitation Calculation (80% - 95%)
        self._update_progress("Precipitation", 80, "Calculating precipitation patterns...")
        precip_map = self._calculate_precipitation(humid_map, temp_map, wind_map, heightmap, slopemap)

        # WeatherData-Objekt erstellen
        weather_data = WeatherData()
        weather_data.wind_map = wind_map
        weather_data.temp_map = temp_map
        weather_data.precip_map = precip_map
        weather_data.humid_map = humid_map
        weather_data.lod_level = lod
        weather_data.actual_size = target_size
        weather_data.parameters = parameters.copy()

        self.logger.debug(f"Weather generation complete - LOD: {lod}, size: {target_size}")

        return weather_data

    def _save_to_data_manager(self, data_manager, result, parameters):
        """
        Funktionsweise: Speichert Weather-Ergebnisse im DataManager
        Aufgabe: Automatische Speicherung aller Weather-Outputs mit Parameter-Tracking
        Parameter: data_manager, result (WeatherData), parameters
        """
        if isinstance(result, WeatherData):
            # WeatherData-Objekt in einzelne Arrays aufteilen für DataManager
            data_manager.set_weather_data("wind_map", result.wind_map, parameters)
            data_manager.set_weather_data("temp_map", result.temp_map, parameters)
            data_manager.set_weather_data("precip_map", result.precip_map, parameters)
            data_manager.set_weather_data("humid_map", result.humid_map, parameters)

            self.logger.debug("WeatherData object saved to DataManager")
        else:
            # Fallback für Legacy-Format (Tuple)
            if hasattr(result, '__len__') and len(result) == 4:
                wind_map, temp_map, precip_map, humid_map = result
                data_manager.set_weather_data("wind_map", wind_map, parameters)
                data_manager.set_weather_data("temp_map", temp_map, parameters)
                data_manager.set_weather_data("precip_map", precip_map, parameters)
                data_manager.set_weather_data("humid_map", humid_map, parameters)
                self.logger.debug("Legacy weather data saved to DataManager")

    def update_seed(self, new_seed):
        """
        Funktionsweise: Aktualisiert Seed für alle Weather-Komponenten
        Aufgabe: Seed-Update mit Re-Initialisierung des Noise-Generators
        Parameter: new_seed (int) - Neuer Seed
        """
        if new_seed != self.map_seed:
            super().update_seed(new_seed)
            # Noise-Generator mit neuem Seed re-initialisieren
            self.noise_generator = OpenSimplex(seed=new_seed)

    def _get_lod_size(self, lod, original_size):
        """
        Funktionsweise: Bestimmt Zielgröße basierend auf LOD-Level
        Aufgabe: LOD-System für Weather mit gleicher Logik wie Terrain
        """
        lod_sizes = {"LOD64": 64, "LOD128": 128, "LOD256": 256}

        if lod == "FINAL":
            return original_size
        else:
            return lod_sizes.get(lod, 64)

    def _interpolate_array(self, array, target_size):
        """
        Funktionsweise: Interpoliert 2D-Array auf neue Größe mittels bilinearer Interpolation
        Aufgabe: LOD-Upscaling für Heightmap, Shademap, etc.
        """
        if len(array.shape) == 2:
            # 2D Array (heightmap, temp_map, etc.)
            return self._interpolate_2d(array, target_size)
        elif len(array.shape) == 3 and array.shape[2] == 2:
            # 3D Array mit 2 Kanälen (wind_map)
            result = np.zeros((target_size, target_size, 2), dtype=array.dtype)
            result[:, :, 0] = self._interpolate_2d(array[:, :, 0], target_size)
            result[:, :, 1] = self._interpolate_2d(array[:, :, 1], target_size)
            return result
        else:
            raise ValueError(f"Unsupported array shape for interpolation: {array.shape}")

    def _interpolate_2d(self, array, target_size):
        """
        Funktionsweise: Bilineare Interpolation für 2D-Arrays
        Aufgabe: Smooth Upscaling ohne Artefakte
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

    def _calculate_temperature(self, heightmap, shademap, target_size):
        """
        Funktionsweise: Berechnet Temperaturfeld mit allen Einflüssen und LOD-Unterstützung
        Aufgabe: Integriert Altitude, Solar und Latitude-Effekte plus Noise
        """
        # Noise-Variation an Kartengrenze
        height, width = heightmap.shape
        noise_variation = np.zeros((height, width), dtype=np.float32)

        # Progress-Update für Temperature Sub-Steps
        self._update_progress("Temperature", 21, "Generating atmospheric noise variation...")

        for y in range(height):
            for x in range(width):
                # Stärkere Variation an Kartenrändern
                edge_factor = self._calculate_edge_factor(x, y, width, height)
                noise_val = self.noise_generator.noise2(x / width * 3, y / height * 3)
                noise_variation[y, x] = noise_val * edge_factor * 5.0

        self._update_progress("Temperature", 23, "Calculating altitude and solar effects...")

        # Temperatur-Calculator verwenden
        temp_calc = TemperatureCalculator(self.air_temp_entry, self.solar_power, self.altitude_cooling)
        temp_map = temp_calc.calculate_base_temperature(heightmap, shademap, noise_variation)

        return temp_map

    def _simulate_wind_field(self, heightmap, slopemap, temp_map, shademap, target_size):
        """
        Funktionsweise: Simuliert Windfeld mit Druckgradienten und Terrain-Interaktion
        Aufgabe: Erstellt realistisches Windfeld mit thermischen Effekten
        """
        # Wind-Simulator verwenden
        wind_sim = WindFieldSimulator(self.wind_speed_factor, self.terrain_factor, self.thermic_effect)

        # Druckfeld generieren
        self._update_progress("Wind Field", 30, "Generating pressure gradients...")
        pressure_field = wind_sim.simulate_pressure_gradients(heightmap.shape, self.noise_generator)

        # Terrain-Ablenkung anwenden
        self._update_progress("Wind Field", 40, "Applying terrain deflection...")
        wind_field = wind_sim.apply_terrain_deflection(pressure_field, slopemap, heightmap)

        # Thermische Effekte hinzufügen
        self._update_progress("Wind Field", 50, "Calculating thermal effects...")
        wind_field = wind_sim.calculate_thermal_effects(wind_field, temp_map, shademap)

        return wind_field

    def _calculate_humidity(self, soil_moist_map, temp_map, wind_field):
        """
        Funktionsweise: Berechnet Luftfeuchtigkeit durch Evaporation und Transport
        Aufgabe: Erstellt realistische Feuchtigkeits-Verteilung
        """
        moisture_manager = AtmosphericMoistureManager()

        # Evaporation von Bodenfeuchtigkeit
        self._update_progress("Humidity", 60, "Calculating evaporation from soil...")
        evaporation_map = moisture_manager.calculate_evaporation(soil_moist_map, temp_map, wind_field)

        # Initiale Feuchtigkeit basierend auf Evaporation
        humid_map = evaporation_map * 2.0  # Startfeuchtigkeit

        # Feuchtigkeits-Transport (mehrere Iterationen für Stabilität)
        self._update_progress("Humidity", 65, "Transporting moisture (iteration 1/3)...")
        for i in range(3):
            humid_map = moisture_manager.transport_moisture(humid_map, wind_field, dt=0.5)
            if i < 2:  # Nur für erste 2 Iterationen Progress-Update
                self._update_progress("Humidity", 65 + (i + 1) * 3, f"Transporting moisture (iteration {i + 2}/3)...")

        # Diffusion für natürliche Verteilung
        self._update_progress("Humidity", 75, "Applying humidity diffusion...")
        humid_map = moisture_manager.apply_humidity_diffusion(humid_map, iterations=2)

        return humid_map

    def _calculate_precipitation(self, humid_map, temp_map, wind_field, heightmap, slopemap):
        """
        Funktionsweise: Berechnet Niederschlag durch Kondensation und orographische Effekte
        Aufgabe: Erstellt Niederschlags-Verteilung
        """
        precip_system = PrecipitationSystem()

        # Orographischer Niederschlag
        self._update_progress("Precipitation", 85, "Calculating orographic precipitation...")
        oro_precip = precip_system.calculate_orographic_precipitation(
            humid_map, wind_field, heightmap, slopemap
        )

        # Kondensations-Niederschlag
        self._update_progress("Precipitation", 90, "Calculating condensation precipitation...")
        condensation_precip = precip_system.simulate_moisture_transport(humid_map, temp_map)

        # Finale Niederschlags-Berechnung
        self._update_progress("Precipitation", 93, "Combining precipitation sources...")
        precip_map = precip_system.trigger_precipitation_events(
            humid_map, temp_map, oro_precip, condensation_precip
        )

        return precip_map

    def _calculate_slopes(self, heightmap):
        """
        Funktionsweise: Berechnet Slope-Map aus Heightmap (dz/dx, dz/dy)
        Aufgabe: Ableitung der Slopes für Wind-Terrain-Interaktion
        """
        height, width = heightmap.shape
        slopemap = np.zeros((height, width, 2), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                # dz/dx berechnen
                if x > 0 and x < width - 1:
                    dz_dx = (heightmap[y, x + 1] - heightmap[y, x - 1]) * 0.5
                elif x == 0:
                    dz_dx = heightmap[y, x + 1] - heightmap[y, x]
                else:  # x == width - 1
                    dz_dx = heightmap[y, x] - heightmap[y, x - 1]

                # dz/dy berechnen
                if y > 0 and y < height - 1:
                    dz_dy = (heightmap[y + 1, x] - heightmap[y - 1, x]) * 0.5
                elif y == 0:
                    dz_dy = heightmap[y + 1, x] - heightmap[y, x]
                else:  # y == height - 1
                    dz_dy = heightmap[y, x] - heightmap[y - 1, x]

                slopemap[y, x, 0] = dz_dx
                slopemap[y, x, 1] = dz_dy

        return slopemap

    def _calculate_edge_factor(self, x, y, width, height):
        """
        Funktionsweise: Berechnet Edge-Faktor für stärkere Noise-Variation an Kartenrändern
        Aufgabe: Verstärkt atmosphärische Variationen an Map-Grenzen
        """
        # Distanz zu nächstem Rand
        dist_to_edge = min(x, y, width - 1 - x, height - 1 - y)
        max_dist = min(width, height) // 4

        if dist_to_edge < max_dist:
            edge_factor = 1.0 + (max_dist - dist_to_edge) / max_dist
        else:
            edge_factor = 1.0

        return edge_factor

    # ===== LEGACY-KOMPATIBILITÄT =====
    # Alle alten Methoden bleiben für Rückwärts-Kompatibilität erhalten

    def generate_weather_system(self, heightmap, shade_map, soil_moist_map, air_temp_entry, solar_power,
                                altitude_cooling, thermic_effect, wind_speed_factor, terrain_factor,
                                flow_direction=None, flow_accumulation=None, map_seed=None):
        """
        Funktionsweise: Legacy-Methode für direkte Weather-Generierung (KOMPATIBILITÄT)
        Aufgabe: Erhält bestehende API für Rückwärts-Kompatibilität
        """
        # Konvertiert alte API zur neuen API
        dependencies = {
            'heightmap': heightmap,
            'shademap': shade_map,
            'soil_moist_map': soil_moist_map
        }
        parameters = {
            'air_temp_entry': air_temp_entry,
            'solar_power': solar_power,
            'altitude_cooling': altitude_cooling,
            'thermic_effect': thermic_effect,
            'wind_speed_factor': wind_speed_factor,
            'terrain_factor': terrain_factor
        }

        # Seed aktualisieren falls nötig
        if map_seed is not None:
            self.update_seed(map_seed)

        weather_data = self._execute_generation("LOD64", dependencies, parameters)

        # Legacy-Format zurückgeben (Tuple)
        return weather_data.wind_map, weather_data.temp_map, weather_data.precip_map, weather_data.humid_map

    def update_weather_cycle(self, current_weather, heightmap, shade_map, soil_moist_map, time_step=1.0):
        """
        Funktionsweise: Legacy-Methode für Weather-Updates
        Aufgabe: Erhält bestehende API für zeitliche Entwicklung
        """
        wind_map, temp_map, precip_map, humid_map = current_weather

        # Slopemap berechnen
        slopemap = self._calculate_slopes(heightmap)

        # Feuchtigkeits-Update durch Transport
        moisture_manager = AtmosphericMoistureManager()
        humid_map = moisture_manager.transport_moisture(humid_map, wind_map, dt=time_step)

        # Evaporation hinzufügen
        evaporation = moisture_manager.calculate_evaporation(soil_moist_map, temp_map, wind_map)
        humid_map += evaporation * time_step

        # Neue Niederschlags-Berechnung
        precip_system = PrecipitationSystem()
        new_precip = precip_system.simulate_moisture_transport(humid_map, temp_map)

        # Feuchtigkeit durch Niederschlag reduzieren
        humid_map -= new_precip * 0.8
        humid_map = np.maximum(0, humid_map)  # Negative Feuchtigkeit verhindern

        return wind_map, temp_map, new_precip, humid_map

    def integrate_atmospheric_effects(self, weather_data, flow_direction=None, flow_accumulation=None):
        """
        Funktionsweise: Legacy-Methode für Wasserfluss-Integration
        Aufgabe: Erweiterte Wetter-Wasser-Interaktion
        """
        wind_map, temp_map, precip_map, humid_map = weather_data

        if flow_direction is not None and flow_accumulation is not None:
            # Flussnähe erhöht lokale Feuchtigkeit
            height, width = humid_map.shape

            for y in range(height):
                for x in range(width):
                    if x < flow_accumulation.shape[1] and y < flow_accumulation.shape[0]:
                        flow_value = flow_accumulation[y, x]

                        if flow_value > 0:
                            # Erhöhte Feuchtigkeit nahe Flüssen
                            humidity_boost = min(10.0, flow_value * 0.1)
                            humid_map[y, x] += humidity_boost

                            # Leichte Temperatur-Modifikation durch Wasser
                            temp_mod = -flow_value * 0.01  # Kühlung durch Wasser
                            temp_map[y, x] += temp_mod

        return wind_map, temp_map, precip_map, humid_map

    def get_weather_statistics(self, weather_data):
        """
        Funktionsweise: Legacy-Methode für Weather-Statistiken
        Aufgabe: Analyse-Funktionen für Weather-System-Debugging
        """
        if isinstance(weather_data, WeatherData):
            wind_map = weather_data.wind_map
            temp_map = weather_data.temp_map
            precip_map = weather_data.precip_map
            humid_map = weather_data.humid_map
        else:
            wind_map, temp_map, precip_map, humid_map = weather_data

        # Wind-Geschwindigkeiten berechnen
        wind_speeds = np.sqrt(wind_map[:, :, 0] ** 2 + wind_map[:, :, 1] ** 2)

        stats = {
            'temperature': {
                'min': float(np.min(temp_map)),
                'max': float(np.max(temp_map)),
                'mean': float(np.mean(temp_map)),
                'std': float(np.std(temp_map))
            },
            'precipitation': {
                'min': float(np.min(precip_map)),
                'max': float(np.max(precip_map)),
                'mean': float(np.mean(precip_map)),
                'total': float(np.sum(precip_map))
            },
            'humidity': {
                'min': float(np.min(humid_map)),
                'max': float(np.max(humid_map)),
                'mean': float(np.mean(humid_map))
            },
            'wind': {
                'min_speed': float(np.min(wind_speeds)),
                'max_speed': float(np.max(wind_speeds)),
                'mean_speed': float(np.mean(wind_speeds)),
                'dominant_direction': self._calculate_dominant_wind_direction(wind_map)
            }
        }

        return stats

    def _calculate_dominant_wind_direction(self, wind_map):
        """
        Funktionsweise: Berechnet dominante Windrichtung über gesamte Map
        Aufgabe: Analyse der Haupt-Windrichtung für Statistiken
        """
        # Durchschnittliche Windkomponenten
        avg_wind_x = np.mean(wind_map[:, :, 0])
        avg_wind_y = np.mean(wind_map[:, :, 1])

        # Winkel berechnen (in Grad)
        angle_rad = np.arctan2(avg_wind_y, avg_wind_x)
        angle_deg = np.degrees(angle_rad)

        # Auf [0, 360] normalisieren
        if angle_deg < 0:
            angle_deg += 360

        # Himmelsrichtung bestimmen
        directions = ['East', 'Northeast', 'North', 'Northwest',
                      'West', 'Southwest', 'South', 'Southeast']
        direction_index = int((angle_deg + 22.5) // 45) % 8

        return {
            'angle_degrees': float(angle_deg),
            'direction': directions[direction_index]
        }