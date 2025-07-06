"""
Weather System Generation
Handles rain distribution, wind simulation, and temperature mapping
Includes orographic effects and wind-based weather patterns
"""

import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter, distance_transform_edt

def calculate_hillshade(heightmap, sun_azimuth=75, sun_elevation=50, scale=1.0):
    """
    Berechnet Hillshade (Schattenwurf) für Heightmap
    sun_azimuth: 90° = Sonne im Süden, 180° = Osten, 0° = Westen
    sun_elevation: 60° = Sonnenstand über Horizont
    """
    # Konvertiere zu Radians
    azimuth_rad = np.radians(sun_azimuth)
    elevation_rad = np.radians(sun_elevation)

    # Berechne Gradienten (Steigung in x und y Richtung)
    grad_y, grad_x = np.gradient(heightmap / scale)

    # Berechne Oberflächennormalen
    # Slope (Steigung) und Aspect (Ausrichtung) der Oberfläche
    slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    aspect = np.arctan2(-grad_y, grad_x)  # Negative y wegen Koordinatensystem

    # Sonnenvektor in kartesischen Koordinaten
    sun_x = np.sin(azimuth_rad) * np.cos(elevation_rad)
    sun_y = np.cos(azimuth_rad) * np.cos(elevation_rad)
    sun_z = np.sin(elevation_rad)

    # Oberflächennormalen in kartesischen Koordinaten
    normal_x = -np.sin(slope) * np.sin(aspect)
    normal_y = -np.sin(slope) * np.cos(aspect)
    normal_z = np.cos(slope)

    # Dot Product: Winkel zwischen Sonne und Oberflächennormale
    illumination = (sun_x * normal_x +
                    sun_y * normal_y +
                    sun_z * normal_z)

    # Klemme auf [0,1] - negative Werte sind im Schatten
    hillshade = np.clip(illumination, 0, 1)

    return hillshade

def get_neighbors(heightmap, y, x):
    """Gibt gültige Nachbarkoordinaten zurück """
    h, w = heightmap.shape
    neighbors = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            neighbors.append((ny, nx))
    return neighbors

class RainGenerator:
    """Regenkarten-Generator mit Windfeld-Simulation """

    def __init__(self, world):
        self.world = world
        self.wind_x = None
        self.wind_y = None
        self.wind_speed = None
        self.wind_direction = None
        self.moisture_map = None
        self.air_temperature = None
        self.hillshade = None
        self.sediment_map = np.zeros_like(world.heightmap)

    def calculate_terrain_temperature(self, base_temp=15.0, elevation_lapse=25.0):  # Reduziert von 65 auf 25
        heightmap = self.world.heightmap
        height, width = heightmap.shape

        self.hillshade = calculate_hillshade(heightmap, sun_azimuth=0, sun_elevation=60)
        terrain_temp = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                elevation_km = heightmap[y, x] / (self.world.scale * 1000)

                latitude_factor = y / height
                latitude_temp = latitude_factor * 8.0  # Reduziert von 10 auf 8

                elevation_temp = -elevation_lapse * elevation_km

                solar_heating = self.hillshade[y, x] * 5.0  # Reduziert von 8 auf 5

                terrain_temp[y, x] = base_temp + latitude_temp + elevation_temp + solar_heating

        return terrain_temp

    def generate_wind_field_with_temperature(self, base_wind_speed=1.0, terrain_factor=0.3,
                                             initial_air_temp=15.0, temp_noise_amplitude=2.25):
        """
        Generiert 2D-Windfeld mit Terrain-Deflection und Lufttemperatur-Transport
        """
        heightmap = self.world.heightmap
        height, width = heightmap.shape

        wind_speed = np.zeros((height, width))
        wind_direction = np.zeros((height, width))
        self.wind_x = np.zeros((height, width))
        self.wind_y = np.zeros((height, width))
        self.air_temperature = np.zeros((height, width))

        terrain_temp = self.calculate_terrain_temperature()
        noise_gen = OpenSimplex(self.world.seed + 100)
        grad_y, grad_x = np.gradient(heightmap)

        for y in range(height):
            y_noise = noise_gen.noise2(0, y * 0.1)
            initial_temp = initial_air_temp + y_noise * temp_noise_amplitude  # 15 ± 2.25 = ±15%

            # Wind kommt jetzt von WESTEN (x=0) und geht nach OSTEN
            west_wind_strength = base_wind_speed * (1.0 + y_noise * 0.15)
            current_speed = west_wind_strength
            current_direction = 0.0  # 0° = Richtung Osten
            current_air_temp = initial_temp

            for x in range(width):
                if x > 0:
                    height_diff = heightmap[y, x] - heightmap[y, x - 1]
                    speed_change = -height_diff * terrain_factor * 0.01
                    current_speed = max(0.1, current_speed + speed_change)

                    terrain_deflection = np.arctan2(grad_y[y, x], grad_x[y, x]) * terrain_factor * 0.3
                    current_direction += terrain_deflection
                    current_direction *= 0.8

                    terrain_exchange_rate = 0.1
                    retention_rate = 0.9
                    current_air_temp = (current_air_temp * retention_rate +
                                        terrain_temp[y, x] * terrain_exchange_rate)

                wind_speed[y, x] = current_speed
                wind_direction[y, x] = current_direction
                self.air_temperature[y, x] = current_air_temp

                # Wind zeigt nach Osten (positive x-Richtung)
                self.wind_x[y, x] = current_speed * np.cos(current_direction)
                self.wind_y[y, x] = current_speed * np.sin(current_direction)

        self.wind_x = gaussian_filter(self.wind_x, sigma=1.0)
        self.wind_y = gaussian_filter(self.wind_y, sigma=1.0)
        self.air_temperature = gaussian_filter(self.air_temperature, sigma=1.5)

        self.wind_speed = np.sqrt(self.wind_x ** 2 + self.wind_y ** 2)
        self.wind_direction = np.arctan2(self.wind_y, self.wind_x)

        return self.wind_speed, self.wind_direction, self.air_temperature

    def generate_orographic_rain_with_wind(self, base_wind_speed=1.0, terrain_factor=0.3,
                                           rain_threshold=0.5, initial_moisture=1.0,
                                           moisture_recovery=0.03, diffusion_strength=0.3):
        """
        Regensimulation mit Windfeld-Integration und Temperatur
        """
        heightmap = self.world.heightmap
        height, width = heightmap.shape

        # Windfeld mit Temperatur generieren
        wind_speed, wind_direction, air_temp = self.generate_wind_field_with_temperature(
            base_wind_speed, terrain_factor
        )

        rain_map = np.zeros((height, width))
        moisture_map = np.zeros((height, width))
        rain_indicator = np.zeros((height, width))
        temp_rain = np.zeros((height, width))

        for y in range(height):
            moisture = initial_moisture
            rain_ind = 0.0

            for x in range(width):
                current_elevation = heightmap[y, x]

                if x > 0:
                    effective_height_diff = current_elevation - heightmap[y, x - 1]
                    wind_influence = abs(self.wind_x[y, x]) / (self.wind_speed[y, x] + 0.1)

                    # Temperatur-Effekt: kältere Luft kondensiert leichter
                    temp_factor = max(0.5, 1.5 - air_temp[y, x] / 20.0)  # Kältere Luft = mehr Regen

                    rain_ind += effective_height_diff * 0.02 * wind_influence * temp_factor
                    rain_ind = max(0.0, rain_ind)

                if rain_ind > rain_threshold and moisture > 0:
                    rainfall = min(
                        moisture * rain_ind * wind_speed[y, x] * 0.1,
                        moisture * 0.7
                    )
                    temp_rain[y, x] = rainfall
                    moisture -= rainfall
                    rain_ind *= 0.7

                moisture_map[y, x] = moisture
                rain_indicator[y, x] = rain_ind
                moisture = min(1.0, moisture + moisture_recovery)

        rain_map = temp_rain.copy()

        # Diffusions-Iterationen mit Temperatur-Einfluss
        for iteration in range(3):
            diffused_rain = gaussian_filter(rain_map, sigma=1.5) * diffusion_strength

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if rain_map[y, x] > 0:
                        wind_dir = self.wind_direction[y, x]
                        wind_str = self.wind_speed[y, x]

                        dx = int(np.round(np.cos(wind_dir)))
                        dy = int(np.round(np.sin(wind_dir)))

                        target_x = min(width - 1, max(0, x + dx))
                        target_y = min(height - 1, max(0, y + dy))

                        transfer = rain_map[y, x] * 0.1 * wind_str * diffusion_strength
                        rain_map[target_y, target_x] += transfer
                        rain_map[y, x] -= transfer * 0.5

            rain_map = rain_map * (1 - diffusion_strength) + diffused_rain

        self.moisture_map = moisture_map
        return np.clip(rain_map, 0.0, 1.0)

    def calculate_evaporation_map(self, water_network, temperature_map):
        evaporation = np.zeros_like(self.world.heightmap)

        # Verdunstung aus Gewässern (Punkt 5)
        if water_network:
            water_mask = np.zeros_like(self.world.heightmap)
            for water_type in ['streams', 'rivers', 'major_rivers', 'lakes']:
                if water_type in water_network:
                    water_mask = np.maximum(water_mask, water_network[water_type])

            # Verdunstungsrate abhängig von Temperatur und Wasserfläche
            evaporation += water_mask * temperature_map * 0.3

        # Verdunstung aus feuchten Böden
        if hasattr(self, 'moisture_map') and self.moisture_map is not None:
            soil_evaporation = self.moisture_map * temperature_map * 0.1
            evaporation += soil_evaporation

        return evaporation

    def calculate_thermal_convection(self, temperature_map):
        # Thermische Aufwinde in warmen Tälern (Punkt 6)
        grad_y, grad_x = np.gradient(self.world.heightmap)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Täler haben niedrige Slope, hohe Temperatur = starke Konvektion
        valley_mask = slope < np.percentile(slope, 30)  # Untere 30% = Täler
        warm_mask = temperature_map > np.percentile(temperature_map, 70)  # Obere 30% = warm

        convection_strength = valley_mask.astype(float) * warm_mask.astype(float) * temperature_map

        # Konvektion fördert lokale Niederschläge
        convection_rain = gaussian_filter(convection_strength, sigma=2.0) * 0.2

        return convection_rain

    def calculate_downstream_moisture(self, initial_rain_map, flow_direction, flow_accumulation):
        moisture_transport = initial_rain_map.copy()
        height, width = initial_rain_map.shape

        # Flow directions wie in WaterGenerator
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        # Sortiere Pixel nach Höhe (von hoch zu niedrig)
        height_order = np.unravel_index(
            np.argsort(self.world.heightmap, axis=None)[::-1],
            self.world.heightmap.shape
        )

        # Transportiere Feuchtigkeit bergabwärts
        for y, x in zip(height_order[0], height_order[1]):
            if flow_direction[y, x] == -1:  # Rand
                continue

            current_moisture = moisture_transport[y, x]
            if current_moisture <= 0.01:
                continue

            # Finde Ziel-Pixel
            direction = flow_direction[y, x]
            dy, dx = directions[direction]
            target_y, target_x = y + dy, x + dx

            if 0 <= target_y < height and 0 <= target_x < width:
                # Transport-Effizienz basierend auf Flow-Stärke
                transport_rate = min(0.8, flow_accumulation[y, x] / flow_accumulation.max() * 0.5 + 0.1)

                transported_moisture = current_moisture * transport_rate
                moisture_transport[target_y, target_x] += transported_moisture
                moisture_transport[y, x] -= transported_moisture

        return moisture_transport

    def get_final_soil_moisture(self):
        """Gibt finale Bodenfeuchtigkeit zurück (separiert von direktem Regen)"""
        if hasattr(self, 'final_soil_moisture'):
            return self.final_soil_moisture
        return np.zeros_like(self.world.heightmap)

    def set_soil_moisture(self, moisture_map):
        """Setzt finale Bodenfeuchtigkeit"""
        self.final_soil_moisture = moisture_map

    def create_water_map(self) -> np.ndarray:
        """Erstellt binäre Wasser-Karte"""
        if not hasattr(self.world, 'water_network'):
            return np.zeros_like(self.world.heightmap)

        water_map = np.zeros_like(self.world.heightmap)
        for water_type in ['streams', 'rivers', 'major_rivers', 'lakes']:
            if water_type in self.world.water_network:
                water_map = np.maximum(water_map, self.world.water_network[water_type])

        return (water_map > 0).astype(float)


class TemperatureGenerator:
    """Temperaturkarten-Generator basierend auf Höhe, Breitengrad und Sonneneinstrahlung"""

    def __init__(self, world):
        self.world = world

    def generate_temperature_map(self, use_air_temperature=True):
        """
        Generiert Temperaturkarte basierend auf:
        - Lufttemperatur (falls verfügbar aus RainGenerator)
        - Höhe und Position
        - Sonneneinstrahlung (Hillshade)
        """
        heightmap = self.world.heightmap
        height, width = heightmap.shape

        # Prüfe ob Lufttemperatur verfügbar ist
        if (use_air_temperature and hasattr(self.world, 'rain_generator') and
                hasattr(self.world.rain_generator, 'air_temperature')):
            # Verwende die transportierte Lufttemperatur als Basis
            temperature_map = self.world.rain_generator.air_temperature.copy()
            # Normalisiere auf [0,1] Bereich
            temp_min, temp_max = temperature_map.min(), temperature_map.max()
            temperature_map = (temperature_map - temp_min) / (temp_max - temp_min + 1e-8)
        else:
            # Fallback: Berechne wie vorher
            temperature_map = np.zeros((height, width))

            # Berechne Hillshade
            hillshade = calculate_hillshade(heightmap)

            for y in range(height):
                for x in range(width):
                    elevation = heightmap[y, x] / self.world.scale

                    # Nord-Süd Gradient korrigiert (y=0 ist Norden, y=height ist Süden)
                    latitude_factor = (height-y) / height  # 0.0 = Nord (kalt), 1.0 = Süd (warm)
                    base_temp = latitude_factor * 0.15  # 15% Anteil

                    # Höhen-Abkühlung
                    elevation_factor = (1.0 - elevation) * 0.45  # 45% Anteil

                    # Sonneneinstrahlung (Hillshade)
                    solar_factor = hillshade[y, x] * 0.4  # 40% Anteil

                    temperature = base_temp + elevation_factor + solar_factor
                    temperature_map[y, x] = np.clip(temperature, 0.0, 1.0)

        temperature_map = self.smooth_temperature_map(temperature_map)
        return temperature_map

    def smooth_temperature_map(self, temp_map: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        """Glättet die Temperaturkarte mit gaußschem Filter"""
        smoothed = gaussian_filter(temp_map, sigma=sigma)
        return np.clip(smoothed, 0.0, 1.0)