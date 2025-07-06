"""
Main World Generation Orchestrator
Coordinates terrain, weather, water, and biome generation
Provides visualization and regional control systems
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, LinearSegmentedColormap
from typing import Tuple, List

from matplotlib.patches import Patch
from memory_profiler import profile
from scipy import ndimage

from terrain_generator import BaseTerrainGenerator, HydraulicErosion
from weather_generator import RainGenerator, TemperatureGenerator, calculate_hillshade
from water_generator import RiverSimulator
from biome_generator import BiomeClassifier, biomes

def create_terrain_colormap():
    """Erstellt custom Terrain Colormap: hellgrün→grün→gelb→braun→grau"""
    colors = ['#90EE90', '#32CD32', '#FFFF00', '#D2691E', '#808080']  # hellgrün→grün→gelb→braun→grau
    return LinearSegmentedColormap.from_list('custom_terrain', colors, N=256)

class World:
    def __init__(self, heightmap, width, height, scale, seed=42):
        self.heightmap = heightmap
        self.width = width
        self.height = height
        self.scale = scale
        self.seed = seed
        self.temperature_map = None
        self.rain_map = None
        self.rain_generator = None

class WorldGenerator:
    """Hauptklasse zur Weltgenerierung"""

    def __init__(self, width: int = 128, height: int = 128, seed: int = 42, sea_level: float = 0.0):
        self.width = width
        self.height = height
        self.seed = seed
        self.sea_level = sea_level
        self.hydro_params = {}

        # Generatoren
        self.terrain_gen = BaseTerrainGenerator(width, height, seed)

        # Karten
        self.heightmap = None
        self.original_heightmap = None
        self.flow_map = None
        self.rain_map = None
        self.temperature_map = None
        self.biome_map = None

    def generate_world(self, terrain_params=None, hydro_params=None, iterations=3):
        if terrain_params is None:
            terrain_params = {}
        if hydro_params is None:
            hydro_params = {}

        self.hydro_params = hydro_params

        print("Generiere Heightmap...")
        self.heightmap = self.terrain_gen.generate_heightmap(**terrain_params)
        self.original_heightmap = self.heightmap.copy()

        self.heightmap = self._fill_pits_global(self.heightmap)
        world = World(self.heightmap, self.width, self.height, terrain_params.get("scale", 1.0), self.seed)

        for iteration in range(iterations):
            print(f"Iteration {iteration + 1}/{iterations}...")

            rain_gen = RainGenerator(world)
            world.rain_generator = rain_gen

            # Basis-Regen generieren
            base_rain = rain_gen.generate_orographic_rain_with_wind(
                base_wind_speed=hydro_params.get('wind_speed', 1.0),
                terrain_factor=hydro_params.get('wind_terrain_factor', 0.3),
                rain_threshold=hydro_params.get('rain_threshold', 0.6),
                moisture_recovery=hydro_params.get('rain_threshold', 0.6),
                diffusion_strength=0.7
            )

            river_sim = RiverSimulator(self.heightmap, base_rain,
                                       sea_level=self.sea_level,
                                       rain_scale=hydro_params.get('rain_scale', 5.0),
                                       lake_threshold=hydro_params.get('lake_threshold', 0.5))

            self.flow_map = river_sim.generate_flow_accumulation()
            flow_direction = river_sim.calculate_flow_direction()

            # Neue Features anwenden
            # Punkt 17: Downstream moisture transport
            enhanced_rain = rain_gen.calculate_downstream_moisture(base_rain, flow_direction, self.flow_map)
            rain_gen.set_soil_moisture(enhanced_rain)

            # Punkt 5 & 6: Verdunstung und thermische Konvektion
            if iteration > 0:  # Ab 2. Iteration, wenn Wasser existiert
                temp_gen = TemperatureGenerator(world)
                temp_map = temp_gen.generate_temperature_map()

                evaporation = rain_gen.calculate_evaporation_map(
                    getattr(self, 'water_network', None), temp_map
                )
                convection_rain = rain_gen.calculate_thermal_convection(temp_map)

                enhanced_rain += evaporation * 0.3 + convection_rain

            # Punkt 19: Valley moisture enhancement
            self.rain_map = river_sim.calculate_valley_moisture_enhancement(enhanced_rain, self.flow_map)

            # Normalisierung
            self.rain_map = (self.rain_map - self.rain_map.min()) / (self.rain_map.max() - self.rain_map.min())

            # Windfeld-Daten übertragen
            self.wind_speed = rain_gen.wind_speed
            self.wind_direction = rain_gen.wind_direction
            self.moisture_map = rain_gen.moisture_map
            self.air_temperature = rain_gen.air_temperature
            self.hillshade = rain_gen.hillshade

            erosion = HydraulicErosion(
                erosion_strength=hydro_params.get('erosion_strength', 0.4),
                valley_depth_factor=hydro_params.get('valley_depth_factor', 2.0),
                tributary_factor=hydro_params.get('tributary_factor', 0.6)
            )

            self.heightmap = erosion.erode_hierarchical(self.heightmap, self.flow_map)
            world.heightmap = self.heightmap

        water_network = river_sim.create_river_network()
        self.water_lines = river_sim.create_hierarchical_water_lines()  # Neue Linien-Daten
        self.heightmap = self._adjust_lakes_realistic(self.heightmap, water_network['lakes'])
        self.water_network = water_network

        print("Generiere Temperaturkarte...")
        temp_gen = TemperatureGenerator(world)
        self.temperature_map = temp_gen.generate_temperature_map()

        print("Klassifiziere Biome...")
        biome_classifier = BiomeClassifier(world, self.temperature_map, self.rain_map)
        self.biome_map = biome_classifier.generate_biome_map()
        self.biome_map_super = biome_classifier.generate_supersampled_biome_map()

        return {
            'heightmap': self.heightmap,
            'flow_map': self.flow_map,
            'rain_map': self.rain_map,
            'temperature_map': self.temperature_map,
            'biome_map': self.biome_map
        }

    def visualize_world(self, figsize: Tuple[int, int] = (18, 12)):
        """Erweiterte Visualisierung mit neuen Features"""
        if self.heightmap is None:
            print("Keine Welt generiert! Rufe zuerst generate_world() auf.")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        self._plot_heightmap_with_hillshade(axes[0, 0])
        self._plot_flow_map(axes[0, 1])
        self._plot_rain_map(axes[0, 2])
        self._plot_water_network(axes[1, 0])
        self._plot_air_temperature_map(axes[1, 1])
        self._plot_biome_map(axes[1, 2])

        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.show()

    def _fill_pits_global(self, heightmap, max_fill=5.0):  # max_fill: 2.0-10.0
        """Globales Pit-Filling mit Begrenzung"""
        filled = np.copy(heightmap)
        for _ in range(10):
            old_filled = filled.copy()
            filled = ndimage.maximum_filter(filled, size=3)
            filled = np.minimum(filled, heightmap + max_fill)
            if np.allclose(filled, old_filled):
                break
        return filled

    def _adjust_lakes_realistic(self, heightmap, lakes, lake_depth=3.0):  # lake_depth: 1.0-6.0
        """Realistische Seen-Anpassung mit Drainage"""
        adjusted = heightmap.copy()

        if np.any(lakes):
            # Seen-Zentren finden
            from scipy.ndimage import center_of_mass, label
            lake_labels, num_lakes = label(lakes)

            for lake_id in range(1, num_lakes + 1):
                lake_mask = lake_labels == lake_id

                # See-Zentrum als tiefster Punkt
                lake_center = center_of_mass(lake_mask)
                center_y, center_x = int(lake_center[0]), int(lake_center[1])

                # Drainage-Level bestimmen
                lake_heights = heightmap[lake_mask]
                drainage_level = np.percentile(lake_heights, 10) - lake_depth

                # Graduelle Absenkung zum Zentrum
                for y, x in zip(*np.where(lake_mask)):
                    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
                    depth_factor = np.exp(-distance * 0.1)
                    adjusted[y, x] = max(drainage_level,
                                         heightmap[y, x] - lake_depth * depth_factor)

        return adjusted

    def _add_terrain_map_contours(self, ax, heightmap_used, saturation=1.0, alpha=1.0):
        terrain_cmap = create_terrain_colormap()  # Neue custom colormap

        contour_levels = np.linspace(heightmap_used.min(), heightmap_used.max(), 15)
        im = ax.contourf(heightmap_used, levels=contour_levels, cmap=terrain_cmap, alpha=alpha, origin='lower',
                         extent=[0, 100, 0, 100])
        ax.contour(heightmap_used, levels=contour_levels, colors='gray', linewidths=0.8, alpha=0.7, origin='lower',
                   extent=[0, 100, 0, 100])
        highlighted_levels = contour_levels[::5]
        ax.contour(heightmap_used, levels=highlighted_levels, colors='gray', linewidths=1.5, alpha=0.9, origin='lower',
                   extent=[0, 100, 0, 100])
        return im

    def _add_terrain_map_gradient(self, ax, heightmap_used, saturation=1.0, alpha=1.0):
        """Terrain-Gradient mit Hillshade-Schattierung"""
        terrain_cmap = create_terrain_colormap()  # Neue custom colormap

        if self.hillshade is not None:
            height_norm = (heightmap_used - heightmap_used.min()) / (heightmap_used.max() - heightmap_used.min())
            terrain_colors = terrain_cmap(height_norm)
            shaded_intensity = self.hillshade * 0.7 + 0.3
            for i in range(3):
                terrain_colors[:, :, i] *= shaded_intensity
            im = ax.imshow(terrain_colors, alpha=alpha, origin='lower', extent=[0, 100, 0, 100])
        else:
            im = ax.imshow(heightmap_used, cmap=terrain_cmap, alpha=alpha, origin='lower', extent=[0, 100, 0, 100])
        return im, terrain_cmap

    def _plot_heightmap_with_hillshade(self, ax):
        """Höhenkarte mit Schattenwurf"""
        im, terrain_cmap = self._add_terrain_map_gradient(ax, self.heightmap, 1.0, 1.0)
        im.cmap= terrain_cmap #im.cmap wird nicht übernommen, deshalb hier nochmal gesetzt für Colorbar
        ax.set_title('Höhenkarte mit Schatten')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')

    def _plot_flow_map(self, ax):
        self._add_terrain_map_gradient(ax, self.heightmap, 0.3, 0.6)

        if hasattr(self, 'wind_speed'):
            step = 4

            # Berechne exakte Grid-Größen
            y_indices = np.arange(0, self.height, step)
            x_indices = np.arange(0, self.width, step)

            # Schneide Wind-Arrays auf Grid-Größe zu
            u = self.wind_speed[y_indices, :][:, x_indices] * np.cos(self.wind_direction[y_indices, :][:, x_indices])
            v = self.wind_speed[y_indices, :][:, x_indices] * np.sin(self.wind_direction[y_indices, :][:, x_indices])
            speed = self.wind_speed[y_indices, :][:, x_indices]

            # Grid für Koordinaten (skaliert auf 0-100)
            x_grid = x_indices * 100 / self.width
            y_grid = y_indices * 100 / self.height

            # Debug: Prüfe Shapes
            print(f"Grid shapes: x_grid={len(x_grid)}, y_grid={len(y_grid)}")
            print(f"Wind shapes: u={u.shape}, v={v.shape}, speed={speed.shape}")

            # Stelle sicher dass alle Arrays gleiche Shape haben
            assert u.shape == v.shape == speed.shape == (len(y_grid), len(x_grid)), \
                f"Shape mismatch: u={u.shape}, v={v.shape}, speed={speed.shape}, expected=({len(y_grid)}, {len(x_grid)})"

            strm = ax.streamplot(
                x_grid, y_grid, u, v,
                color=speed, linewidth=1, density=2.0,
                cmap='YlOrBr', arrowsize=0.5
            )

            # 3) Colorbar für Wind hinzufügen
            plt.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04, label='Windgeschwindigkeit')

        ax.set_title('Windfluss & Topografie')
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def _plot_rain_map(self, ax):
        self._add_terrain_map_gradient(ax, self.heightmap, 0.3, 0.6)

        cmap_blue = plt.get_cmap('Blues')
        colors_blue = cmap_blue(np.arange(cmap_blue.N))
        colors_blue[:int(cmap_blue.N * 0.1), 3] = 0
        colors_blue[int(cmap_blue.N * 0.1):int(cmap_blue.N * 0.3), 3] = 0.3
        transparent_blue = mcolors.ListedColormap(colors_blue)

        im1 = ax.imshow(self.rain_map, cmap=transparent_blue, alpha=0.7, origin='lower', extent=[0, 100, 0, 100])

        # Debug: Prüfe ob Bodenfeuchtigkeit existiert
        if (hasattr(self, 'rain_generator') and
                hasattr(self.rain_generator, 'final_soil_moisture')):

            cmap_purple = plt.get_cmap('Purples')
            colors_purple = cmap_purple(np.arange(cmap_purple.N))
            colors_purple[:int(cmap_purple.N * 0.1), 3] = 0
            colors_purple[int(cmap_purple.N * 0.1):int(cmap_purple.N * 0.3), 3] = 0.3
            transparent_purple = mcolors.ListedColormap(colors_purple)

            soil_moisture = self.rain_generator.get_final_soil_moisture()
            im2 = ax.imshow(soil_moisture, cmap=transparent_purple, alpha=0.5, origin='lower', extent=[0, 100, 0, 100])

            # 5) Zwei separate Colorbars
            cb1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Regen')
            cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.12, label='Bodenfeuchtigkeit')  # Weiter rechts
        else:
            plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label='Regen')
            print("DEBUG: Keine Bodenfeuchtigkeit gefunden")  # Debug

        ax.set_title('Regen (blau) + Bodenfeuchtigkeit (violett)')
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def _plot_water_network(self, ax):
        self._add_terrain_map_gradient(ax, self.heightmap, 0.2, 0.9)  # Stärkerer Terrain-Hintergrund

        if hasattr(self, 'water_lines'):
            from scipy.ndimage import gaussian_filter

            streams_img = np.zeros((self.height, self.width))
            rivers_img = np.zeros((self.height, self.width))
            major_rivers_img = np.zeros((self.height, self.width))

            for i, (x, y) in enumerate(self.water_lines['stream_coords']):
                if 0 <= y < self.height and 0 <= x < self.width:
                    streams_img[y, x] = 0.3

            for i, (x, y) in enumerate(self.water_lines['river_coords']):
                if 0 <= y < self.height and 0 <= x < self.width:
                    rivers_img[y, x] = 0.6

            for i, (x, y) in enumerate(self.water_lines['major_river_coords']):
                if 0 <= y < self.height and 0 <= x < self.width:
                    major_rivers_img[y, x] = 1.0

            # 2) Weniger Blur
            streams_blurred = gaussian_filter(streams_img, sigma=0.4)  # war 0.8
            rivers_blurred = gaussian_filter(rivers_img, sigma=0.6)  # war 1.2
            major_rivers_blurred = gaussian_filter(major_rivers_img, sigma=0.8)  # war 1.5

            ax.imshow(streams_blurred, cmap='Blues', alpha=0.4, vmin=0, vmax=1, origin='lower', extent=[0, 100, 0, 100])
            ax.imshow(rivers_blurred, cmap='Blues', alpha=0.7, vmin=0, vmax=1, origin='lower', extent=[0, 100, 0, 100])

            major_cmap = mcolors.LinearSegmentedColormap.from_list('dark_blue', ['#00000000', '#000080'])
            ax.imshow(major_rivers_blurred, cmap=major_cmap, alpha=0.9, vmin=0, vmax=1, origin='lower',
                      extent=[0, 100, 0, 100])

            if hasattr(self, 'water_network') and 'lakes' in self.water_network:
                lakes_blurred = gaussian_filter(self.water_network['lakes'].astype(float), sigma=0.5)  # weniger blur
                ax.imshow(lakes_blurred, cmap='Blues', alpha=0.8, vmin=0, vmax=1, origin='lower',
                          extent=[0, 100, 0, 100])

        ax.set_title('Hierarchisches Wassernetzwerk')
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def _plot_air_temperature_map(self, ax):
        """Lufttemperatur-Karte mit korrigierter Skalierung"""
        self._add_terrain_map_gradient(ax, self.heightmap, 0.3, 0.5)

        if hasattr(self, 'air_temperature') and self.air_temperature is not None:
            im = ax.imshow(self.air_temperature, cmap='RdYlBu_r', alpha=0.8, origin='lower', extent=[0, 100, 0, 100])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Lufttemperatur (°C)')
        else:
            im = ax.imshow(self.temperature_map, cmap='RdYlBu_r', alpha=0.8, origin='lower', extent=[0, 100, 0, 100])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Temperatur')

        ax.set_title('Lufttemperatur & Terrain')
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def _plot_biome_map(self, ax):
        """Biom-Karte mit korrigierten Achsen"""
        self._add_terrain_map_gradient(ax, self.heightmap, 0.2, 0.3)

        biome_colors = [mcolors.to_rgb(biome.color) for biome in biomes]
        biome_names = [biome.name for biome in biomes]

        cmap = mcolors.ListedColormap(biome_colors)
        norm = mcolors.BoundaryNorm(boundaries=list(range(len(biomes) + 1)), ncolors=len(biomes))

        if hasattr(self, 'water_network'):
            water_display = np.zeros_like(self.heightmap)
            water_display = np.maximum(water_display, self.water_network['streams'] * 0.3)
            water_display = np.maximum(water_display, self.water_network['rivers'] * 0.6)
            water_display = np.maximum(water_display, self.water_network['major_rivers'] * 0.9)
            water_display = np.maximum(water_display, self.water_network['lakes'] * 1.0)

            water_cmap = LinearSegmentedColormap.from_list('water', ['#00000000', '#1E90FF'])
            ax.imshow(water_display, cmap=water_cmap, alpha=0.8, vmin=0, vmax=1,
                      interpolation='bilinear', origin='lower', extent=[0, 100, 0, 100])

        # Biome mit korrigierten Achsen
        if hasattr(self, 'biome_map_super'):
            ax.imshow(self.biome_map_super, cmap=cmap, norm=norm, alpha=0.7, origin='lower', extent=[0, 100, 0, 100])
        else:
            ax.imshow(self.biome_map, cmap=cmap, norm=norm, alpha=0.7, origin='lower', extent=[0, 100, 0, 100])

        ax.set_title('Biom-Karte')
        ax.set_aspect('equal')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        legend_elements = [
            Patch(facecolor=biome_colors[i], edgecolor='black', label=biome_names[i])
            for i in range(len(biomes))
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

# =================== BEISPIEL-VERWENDUNG ===================

@profile
def create_example_world():
    """Beispiel mit neuen Features: Hillshade und Lufttemperatur"""
    terrain_params = {
        'octaves': 6,
        'frequency': 0.012,
        'persistence': 0.5,
        'lacunarity': 2.0,
        'scale': 100.0,
        'redistribute_power': 1.3
    }

    hydro_params = {
        'wind_speed': 1.2,
        'wind_terrain_factor': 0.4,
        'rain_threshold': 0.5,
        'rain_scale': 8.0,
        'erosion_strength': 0.8,
        'moisture_recovery': 0.08
    }

    print("Erstelle Welt mit Hillshade und Lufttemperatur...")
    world_gen = WorldGenerator(width=100, height=100, seed=np.random.randint(0, 1000), sea_level=10.0)
    world_gen.generate_world(terrain_params=terrain_params, hydro_params=hydro_params, iterations=2)

    print("Neue Features verfügbar:")
    if world_gen.hillshade is not None:
        print(f"  - Hillshade: {world_gen.hillshade.min():.3f} bis {world_gen.hillshade.max():.3f}")
    if world_gen.air_temperature is not None:
        print(
            f"  - Lufttemperatur: {world_gen.air_temperature.min():.1f}°C bis {world_gen.air_temperature.max():.1f}°C")

    world_gen.visualize_world()
    return world_gen


if __name__ == "__main__":
    print("Erstelle erweiterte Beispielwelt...")
    example_world = create_example_world()
    print("Fertig! Die Karten mit Hillshade und Lufttemperatur werden angezeigt.")