"""
Path: gui/config/gui_default.py

Funktionsweise: GUI-Layout und Styling-Konfiguration
- Window-Größen und Positionen für alle Tabs
- Button-Styling (Farben, Größen, Fonts)
- Canvas-Konfiguration (Split-Ratios, Render-Settings)
- Color-Schemes und Theme-Definitionen

Struktur:
class WindowSettings:
    MAIN_MENU = {"width": 800, "height": 600}
    MAP_EDITOR = {"width": 1500, "height": 1000}

class ButtonSettings:
    PRIMARY = {"color": "#27ae60", "hover": "#229954"}
    SECONDARY = {"color": "#3498db", "hover": "#2980b9"}

class CanvasSettings:
    SPLIT_RATIO = 0.7  # 70% Canvas, 30% Controls
"""


class WindowSettings:
    """
    Funktionsweise: Definiert Standard-Fenstergrößen und Positionen für alle GUI-Komponenten
    Aufgabe: Zentrale Konfiguration aller Window-Parameter für konsistente Darstellung
    """
    MAIN_MENU = {
        "width": 1200,
        "height": 800,
        "min_width": 600,
        "min_height": 600
    }

    MAP_EDITOR = {
        "width": 1500,
        "height": 1000,
        "min_width": 1200,
        "min_height": 800
    }


class ButtonSettings:
    """
    Funktionsweise: Definiert einheitliche Button-Styles für alle UI-Komponenten
    Aufgabe: Konsistente Farbgebung und Hover-Effekte für alle Buttons
    """
    PRIMARY = {
        "color": "#488852",
        "hover": "#5e8964",
        "font_size": 18,
        "font_weight": "bold"
    }

    SECONDARY = {
        "color": "#487188",
        "hover": "#5e7a89",
        "font_size": 18,
        "font_weight": "normal"
    }

    DANGER = {
        "color": "#884858",
        "hover": "#895e69",
        "font_size": 18,
        "font_weight": "bold"
    }


class CanvasSettings:
    """
    Funktionsweise: Konfiguriert Canvas-Layout und Render-Parameter für Map-Display
    Aufgabe: Einheitliche Split-Ratios und Render-Einstellungen für alle Map-Tabs
    """
    SPLIT_RATIO = 0.7  # 70% Canvas, 30% Controls

    # 2D Canvas Settings
    CANVAS_2D = {
        "background_color": "#2c3e50",
        "grid_color": "#34495e",
        # Kein Blau mehr (sah wie Wasser aus, siehe elevation_vmin/vmax unten) -
        # matplotlib zyklt durch diese Liste je Contour-Level, betraf also nicht
        # nur niedrige Höhen sondern Linien über die ganze Karte verteilt.
        "contour_colors": ["#7f8c8d", "#e74c3c", "#f39c12"],
        "dpi": 100,
        # Feste Höhen-Farbskala (Meter) statt Auto-Skalierung pro Heightmap,
        # damit z.B. ein 200m-Hügel und ein 3500m-Berg nicht dieselbe volle
        # Farbspanne bekommen - entspricht realen topografischen Karten.
        # Werte oberhalb von elevation_vmax clippen auf die höchste Farbe.
        "elevation_vmin": 0.0,
        "elevation_vmax": 4000.0
    }

    # 3D Canvas Settings
    CANVAS_3D = {
        "background_color": (0.17, 0.24, 0.31, 1.0),  # RGBA
        "light_position": (1.0, 1.0, 1.0),
        # Terrain-Mesh wird immer auf ein 10x10-Einheiten XZ-Footprint normiert
        # (siehe MapDisplay3D.terrain_scale_factor). 5.0 war kleiner als der
        # Mesh-Diagonal-Halbradius (~7.07) selbst - bei 45° FOV landeten
        # Vertices dadurch weit außerhalb von NDC [-1,1] und es wurde nichts
        # sichtbar. ~17 umfasst das Mesh bei 45° FOV mit Rand.
        "camera_distance": 17.0,
        "fov": 45.0
    }


class ColorSchemes:
    """
    Funktionsweise: Definiert Farbpaletten für verschiedene Map-Visualisierungen
    Aufgabe: Konsistente Farbgebung für Heightmaps, Biomes und andere Visualisierungen
    """
    TERRAIN = {
        "low": "#2980b9",  # Deep Blue
        "water": "#3498db",  # Blue
        "land": "#27ae60",  # Green
        "mountain": "#95a5a6",  # Gray
        "high": "#ecf0f1"  # White
    }

    # Einzige Quelle für Biome-Index -> (Name, Farbe), genutzt von
    # BiomeLegendDialog (gui/widgets/widgets.py) UND MapDisplay2D._render_biome_map
    # (gui/widgets/map_display_2d.py) - Index entspricht exakt den Werten aus
    # core/biome_generator.py (0-14 Base-Biomes, 15-25 Super-Biomes über
    # SuperBiomeOverrideSystem.super_biome_offset=15). Vorher hatten Legende und
    # Darstellung getrennte, auseinandergelaufene Farbdefinitionen.
    BIOME_COLOR_TABLE = [
        # Base Biomes (Index 0-14)
        ("Ice Cap", "#f8f9fa"),
        ("Tundra", "#e9ecef"),
        ("Taiga", "#228b22"),
        ("Grassland", "#90ee90"),
        ("Temperate Forest", "#006400"),
        ("Mediterranean", "#9acd32"),
        ("Desert", "#daa520"),
        ("Semi Arid", "#d2691e"),
        ("Tropical Rainforest", "#008000"),
        ("Tropical Seasonal", "#32cd32"),
        ("Savanna", "#bdb76b"),
        ("Montane Forest", "#2e8b57"),
        ("Swamp", "#556b2f"),
        ("Coastal Dunes", "#f4a460"),
        ("Badlands", "#a0522d"),
        # Super Biomes (Index 15-25)
        ("Ocean", "#0077be"),
        ("Lake", "#4da6ff"),
        ("Grand River", "#0066cc"),
        ("River", "#3399ff"),
        ("Creek", "#66b3ff"),
        ("Cliff", "#696969"),
        ("Beach", "#f5deb3"),
        ("Lake Edge", "#87ceeb"),
        ("River Bank", "#98fb98"),
        ("Snow Level", "#fffafa"),
        ("Alpine Level", "#d3d3d3"),
    ]


class LayoutSettings:
    """
    Funktionsweise: Definiert Standard-Layout-Parameter für alle GUI-Komponenten
    Aufgabe: Einheitliche Abstände, Margins und Padding-Werte
    """
    PADDING = 10
    MARGIN = 15
    BUTTON_HEIGHT = 50
    SLIDER_HEIGHT = 25
    LABEL_HEIGHT = 20

    CONTROL_PANEL_WIDTH = 300
    STATUS_BAR_HEIGHT = 25

class AppConstants:
    """Application-wide timing and behavior constants"""
    MEMORY_CHECK_INTERVAL_MS = 60000  # Memory monitoring interval
    CLEANUP_DELAY_MS = 2000  # Cleanup protection delay

# Application constants for timing and behavior
class EditorConstants:
    """MapEditor-specific constants for consistent behavior"""
    STATUS_UPDATE_INTERVAL_MS = 5000     # Status bar update frequency
    GENERATION_TIMEOUT_MS = 300000       # 5 minute timeout for generation
    TAB_SWITCH_DELAY_MS = 100           # Delay for smooth tab transitions
    ERROR_DISPLAY_DURATION_MS = 5000    # Error message display time