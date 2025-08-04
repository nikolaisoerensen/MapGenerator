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
        "min_width": 1000,
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
        "contour_colors": ["#3498db", "#e74c3c", "#f39c12"],
        "dpi": 100
    }

    # 3D Canvas Settings
    CANVAS_3D = {
        "background_color": (0.17, 0.24, 0.31, 1.0),  # RGBA
        "light_position": (1.0, 1.0, 1.0),
        "camera_distance": 5.0,
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

    BIOMES = {
        "ocean": "#2980b9",
        "desert": "#f39c12",
        "forest": "#27ae60",
        "tundra": "#95a5a6",
        "grassland": "#2ecc71"
    }


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