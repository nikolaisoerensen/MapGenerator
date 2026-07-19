"""
Path: gui/widgets/map_display_3d.py

Funktionsweise: 3D Terrain-Rendering mit Multi-Layer Support und Interactive Overlays
- Heightmap-basierte Terrain-Geometrie als Basis für alle Tabs
- Multi-Resolution Overlay-System (Heightmap + Shademap + spezifische Layer)
- Tab-spezifische Visualisierungen mit durchschaltbaren Overlays
- Fixed-Axis Camera (60° Elevation, Azimuth-Rotation um Z-Achse)

Tab-spezifische Rendering-Modi:
TERRAIN:
- Basis: Heightmap z(x,y) als 3D-Mesh
- Primary Overlay: 2D-Map Coloring auf Terrain-Oberfläche
- Secondary Overlay: Shademap (64x64) für Schattierung/Verdunklung
- Upscaling: Bilinear/Bicubic Interpolation von 64x64 auf Heightmap-Resolution

GEOLOGY:
- Basis: Heightmap-Terrain
- Overlays (durchschaltbar via Checkboxes):
  * Rock Map: RGB-Überlagerung für Gesteinstypen
  * Hardness Map: Materialfestigkeit-Visualisierung
- Rendering: Additive/Multiplicative Blending auf Terrain-Oberfläche

WEATHER:
- Basis: Heightmap-Terrain
- Surface Overlays (durchschaltbar via Checkboxes):
  * Precipitation Map: Niederschlags-Coloring
  * Temperature Map: Temperatur-Farbschema
  * Wind Map: Windgeschwindigkeits-Visualisierung
  * Humidity Map: Luftfeuchtigkeits-Darstellung
- Future Extensions:
  * Volumetrische Wolken-Rendering via Shader
  * Stromlinien-basierte Windvisualisierung mit animierten Partikeln

WATER:
- Basis: Heightmap-Terrain
- Multi-Layer Overlays (kombinierte Darstellung):
  * Water Map: Wasserkörper-Visualisierung
  * Soil Moisture Map: Bodenfeuchtigkeit
  * Erosion Map: Erosions-Patterns
  * Sedimentation Map: Sedimentablagerungen
- Rendering: Unterschiedliche Farb-Kodierung pro Layer, durchschaltbar

BIOME:
- Basis: Heightmap-Terrain
- Biome Map: 4x höhere Auflösung als Heightmap
- Rendering: 4 Biom-Pixel pro Terrain-Pixel in randomisierter Anordnung
- Upscaling: Entweder Core-Preprocessing oder Runtime-Tessellation

SETTLEMENT:
- Basis: Heightmap-Terrain
- Geometric Overlays:
  * Plot Boundaries: Wireframe-Mesh auf Terrain projiziert
  * Plot Nodes: Vertex-Marker an Eckpunkten
- Point Features:
  * Settlements: Größere gefärbte Kreise/Zylinder
  * Landmarks: Mittlere Marker mit Icons
  * Road Sites: Kleinere Verbindungs-Punkte
- Optional Overlay: Civ Map als Surface-Coloring (durchschaltbar)

Rendering-Pipeline:
1. Terrain-Mesh aus Heightmap generieren
2. Base-Coloring je nach Tab-Typ anwenden
3. Overlay-Blending basierend auf aktiven Checkboxes
4. Shadow-Map Integration (upscaled auf Terrain-Resolution)
5. Lighting und Shading via GLSL Shader

Kommunikationskanäle:
- Input: heightmaps und Overlay-Data von data_manager
- GPU: shader_manager für optimierte Rendering-Pipeline
- Controls: Camera-Settings aus gui_default.py
"""

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSlider, QLabel
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
from OpenGL.GL import shaders
import math
from gui.config.gui_default import CanvasSettings, ColorSchemes
from gui.config.value_default import TERRAIN
from gui.widgets.map_display_2d import _calculate_contour_levels, _get_layer_range

# 3D-interne Overlay-Layer-Namen (siehe self.overlay_data) -> Key in
# CanvasSettings.CANVAS_2D["layer_ranges"] (dieselbe Tabelle, die auch die
# 2D-Ansicht für feste Farbskalen nutzt, siehe map_display_2d.py). Layer ohne
# Eintrag hier (rock_map, biome_map, super_biome_mask, civ_map) sind
# kategorisch/RGB und werden in _colorize_layer() gesondert behandelt.
_LAYER_RANGE_KEY_MAP = {
    "temperature": "temp_map", "precipitation": "precip_map",
    "humidity": "humid_map", "wind": "wind_map",
    "water_map": "water_map", "soil_moisture": "soil_moist_map",
    "erosion": "erosion_map", "sedimentation": "sedimentation_map",
    "flow_map": "flow_map", "hardness_map": "hardness_map", "slope": "slopemap",
}


def _colorize_layer(data, layer_name):
    """
    Funktionsweise: Wandelt ein Overlay-Daten-Array in ein (H,W,3) uint8
    RGB-Array um, für den Upload als Overlay-Textur in _render_overlay().
    Nutzt für kontinuierliche Skalar-Layer dieselbe
    CanvasSettings.CANVAS_2D["layer_ranges"]-Tabelle wie die 2D-Ansicht
    (konsistente Farbgebung zwischen 2D und 3D), für kategorische Layer
    (rock_map, biome_map, super_biome_mask) eine direkte Index-/RGB-
    Umsetzung analog zu map_display_2d.py's _render_rock_map()/
    _render_biome_map().
    Parameter: data (numpy.ndarray) - Rohe Overlay-Daten, layer_name (str) -
    3D-interner Layer-Name (Key aus self.overlay_data[tab_type])
    Rückgabe: numpy.ndarray (H,W,3) uint8
    """
    if layer_name == "rock_map":
        # Bereits (H,W,3) Gesteinsanteile, uint8 - wie 2D direkt als RGB nutzen.
        rgb = np.clip(data.astype(np.float32), 0.0, 255.0).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        return rgb

    if layer_name in ("biome_map", "super_biome_mask"):
        # Index -> Farbe direkt aus derselben Tabelle wie die 2D-Legende.
        n_categories = len(ColorSchemes.BIOME_COLOR_TABLE)
        indices = np.clip(data.astype(np.int32), 0, n_categories - 1)
        lookup = np.array(
            [_hex_to_rgb(hex_color) for _, hex_color in ColorSchemes.BIOME_COLOR_TABLE], dtype=np.uint8)
        return lookup[indices]

    if data.ndim == 3 and data.shape[2] == 2:
        # Vektorfeld (z.B. wind) - Magnitude als Skalar-Grundlage nutzen.
        data = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)

    range_key = _LAYER_RANGE_KEY_MAP.get(layer_name)
    cmap_name, vmin, vmax, scale = _get_layer_range(range_key) if range_key else (None, None, None, "linear")
    if vmin is not None:
        if scale == "log":
            # Gleiche Epsilon-Floor-Logik wie 2D's _render_generic_map() (siehe
            # map_display_2d.py) - hält 2D/3D farblich konsistent für stark
            # rechtsschiefe Layer (erosion_map/sedimentation_map).
            epsilon = vmin * 0.01
            safe_data = np.maximum(data.astype(np.float64), epsilon)
            log_range = max(np.log(vmax) - np.log(vmin), 1e-9)
            norm = np.clip((np.log(safe_data) - np.log(vmin)) / log_range, 0.0, 1.0)
        else:
            norm = np.clip((data - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    else:
        cmap_name = "viridis"
        data_min, data_max = float(data.min()), float(data.max())
        norm = (data - data_min) / max(data_max - data_min, 1e-9)

    rgb = (plt.get_cmap(cmap_name)(norm)[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def _hex_to_rgb(hex_color):
    """Wandelt '#rrggbb' in ein (r,g,b) uint8-Tupel um."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _validate_heightmap(heightmap):
    """
    Funktionsweise: Validiert Heightmap-Daten für 3D-Rendering
    Aufgabe: Prüft numpy-Array auf korrekte Form und numerische Werte
    Parameter: heightmap - Zu prüfende Heightmap
    Rückgabe: bool - True wenn valide, False sonst
    """
    if not isinstance(heightmap, np.ndarray):
        return False

    if heightmap.ndim != 2:
        return False

    if heightmap.shape[0] < 10 or heightmap.shape[1] < 10:
        return False

    if not np.issubdtype(heightmap.dtype, np.number):
        return False

    if np.any(np.isnan(heightmap)) or np.any(np.isinf(heightmap)):
        return False

    return True


def _validate_overlay_data(overlay_data, expected_shape=None):
    """
    Funktionsweise: Validiert Overlay-Daten (Water, Biome, etc.)
    Aufgabe: Prüft Overlay-Arrays auf Kompatibilität mit Heightmap
    Parameter: overlay_data - Overlay-Array, expected_shape - Erwartete Dimensionen
    Rückgabe: bool - True wenn valide, False sonst
    """
    if overlay_data is None:
        return True  # None ist valide (bedeutet kein Overlay)

    if not isinstance(overlay_data, np.ndarray):
        return False

    if overlay_data.ndim < 2 or overlay_data.ndim > 3:
        return False

    if expected_shape and overlay_data.shape[:2] != expected_shape:
        # Biome-Maps können 4x Auflösung haben
        if overlay_data.shape[0] != expected_shape[0] * 4 or overlay_data.shape[1] != expected_shape[1] * 4:
            return False

    if not np.issubdtype(overlay_data.dtype, np.number):
        return False

    return True


def _validate_settlement_data(settlement_data):
    """
    Funktionsweise: Validiert Settlement-Positionsdaten
    Aufgabe: Prüft Settlement-Positionen auf korrekte 3D-Koordinaten
    Parameter: settlement_data - Liste oder Array von Positionen
    Rückgabe: bool - True wenn valide, False sonst
    """
    if settlement_data is None or len(settlement_data) == 0:
        return True

    try:
        # Kann Liste von Tupeln oder numpy Array sein
        if isinstance(settlement_data, list):
            for pos in settlement_data:
                if len(pos) != 3:
                    return False
                if not all(isinstance(coord, (int, float)) for coord in pos):
                    return False
        elif isinstance(settlement_data, np.ndarray):
            if settlement_data.shape[1] != 3:
                return False
        else:
            return False
    except:
        return False

    return True


def _create_perspective_matrix(fov, aspect, near, far):
    """
    Funktionsweise: Erstellt Perspective-Projection-Matrix
    Aufgabe: 3D zu 2D Transformation mit Perspektive
    Parameter: fov (float), aspect (float), near (float), far (float)
    Rückgabe: numpy.ndarray - 4x4 Projection-Matrix
    """
    fov_rad = math.radians(fov)
    f = 1.0 / math.tan(fov_rad / 2.0)

    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0

    return matrix


def _create_lookat_matrix(eye, target, up):
    """
    Funktionsweise: Erstellt View-Matrix für Camera-Position
    Aufgabe: Definiert Camera-Position und Blickrichtung
    Parameter: eye (array), target (array), up (array) - 3D-Vektoren
    Rückgabe: numpy.ndarray - 4x4 View-Matrix
    """
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    # Forward-Vektor (von Eye zu Target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Right-Vektor (Cross-Product von Forward und Up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Up-Vektor korrigieren
    up = np.cross(right, forward)

    # View-Matrix aufbauen
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = right
    matrix[1, :3] = up
    matrix[2, :3] = -forward

    # Translation
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(up, eye)
    matrix[2, 3] = np.dot(forward, eye)

    return matrix


def _create_model_matrix(translation=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
    """
    Funktionsweise: Erstellt Model-Matrix für Objekt-Transformationen
    Aufgabe: Position, Rotation und Skalierung von 3D-Objekten
    Parameter: translation, rotation, scale - 3D-Transformationen
    Rückgabe: numpy.ndarray - 4x4 Model-Matrix
    """
    # Translation-Matrix
    trans_matrix = np.eye(4, dtype=np.float32)
    trans_matrix[0, 3] = translation[0]
    trans_matrix[1, 3] = translation[1]
    trans_matrix[2, 3] = translation[2]

    # Rotation um Y-Achse (für Azimuth)
    rot_y = math.radians(rotation[1])
    rot_matrix = np.eye(4, dtype=np.float32)
    rot_matrix[0, 0] = math.cos(rot_y)
    rot_matrix[0, 2] = math.sin(rot_y)
    rot_matrix[2, 0] = -math.sin(rot_y)
    rot_matrix[2, 2] = math.cos(rot_y)

    # Scale-Matrix
    scale_matrix = np.eye(4, dtype=np.float32)
    scale_matrix[0, 0] = scale[0]
    scale_matrix[1, 1] = scale[1]
    scale_matrix[2, 2] = scale[2]

    # Kombiniere Matrizen: Scale * Rotation * Translation
    return trans_matrix @ rot_matrix @ scale_matrix


def _upscale_shademap(shademap, target_shape):
    """
    Funktionsweise: Skaliert 64x64 Shademap auf Heightmap-Auflösung
    Aufgabe: Bilinear-Interpolation für weiche Schatten-Übergänge
    Parameter: shademap (64x64), target_shape - Ziel-Dimensionen
    Rückgabe: numpy.ndarray - Hochskalierte Shademap
    """
    if shademap is None:
        return None

    # Einfache bilineare Interpolation
    from scipy.ndimage import zoom
    scale_y = target_shape[0] / shademap.shape[0]
    scale_x = target_shape[1] / shademap.shape[1]

    return zoom(shademap, (scale_y, scale_x), order=1)  # order=1 für bilinear


class MapDisplay3D(QOpenGLWidget):
    """
    Funktionsweise: 3D-Visualisierung von Heightmaps mit OpenGL-Rendering
    Aufgabe: Real-time 3D Terrain-Darstellung mit Camera-Controls und Multi-Layer Support
    """

    # Signals für 3D-Interaktion
    camera_changed = pyqtSignal(float, float, float)  # (rotation_x, rotation_y, zoom)
    vertex_selected = pyqtSignal(int, int)  # (x, y)
    rendering_error = pyqtSignal(str)  # Error-Messages

    # Reale Weltgröße, die die Karte immer abdeckt (unabhängig von map_size/
    # Pixelauflösung) - siehe _calculate_terrain_scaling(). Zentral in
    # gui/config/value_default.py TERRAIN.WORLD_SIZE_KM, damit SlopeCalculator
    # dieselbe Annahme verwendet.
    WORLD_SIZE_KM = TERRAIN.WORLD_SIZE_KM

    def __init__(self, parent=None):
        """
        Funktionsweise: Initialisiert 3D OpenGL-Widget mit Standard-Camera-Position
        Aufgabe: Setup von OpenGL-Context und Standard-Rendering-Parameter
        """
        super().__init__(parent)

        # Camera-Parameter - Fixed-Axis (60° Elevation, Azimuth-Rotation)
        self.camera_distance = CanvasSettings.CANVAS_3D["camera_distance"]
        # Positive Elevation = Blick von oben herab (Kartenansicht). Negativ
        # bedeutete Kamera unterhalb der Map, Blick von unten nach oben.
        self.camera_elevation = 55.0  # Feste Elevation
        self.camera_azimuth = 0.0  # Rotation um Z-Achse
        self.fov = CanvasSettings.CANVAS_3D["fov"]

        # Rendering-Daten
        self.heightmap = None
        self.shademap = None  # Shadow-Map (beliebige Auflösung, wird auf Heightmap-Größe hochskaliert)
        # Globaler Schatten-Toggle (Shell-Footer "Shadows"-Checkbox, siehe
        # BaseMapTab.set_shadow_overlay()/set_shadow_overlay() unten) - gilt
        # tab-übergreifend, da _render_terrain_base() (das Basis-Mesh, von
        # jedem Tab-Render-Pfad aufgerufen) den Schatten-Multiplikator anwendet.
        self.shadows_enabled = True
        # Globaler Contour-Lines-Toggle (Shell-Footer, siehe set_contour_overlay()
        # unten). Anders als 2D braucht 3D keine separate Referenz-Heightmap - das
        # Mesh basiert immer auf der kombinierten Heightmap (heightmap_combined),
        # unabhängig vom aktuell gezeigten renderMode/Overlay, FragPos.y im Shader
        # entspricht also immer der echten Roh-Höhe. contour_interval wird aus
        # derselben _calculate_contour_levels()-Logik wie 2D abgeleitet (siehe
        # update_heightmap()), damit beide Ansichten dieselben Höhenlinien-Abstände
        # zeigen.
        self.contours_enabled = True
        self.contour_interval = 25.0
        # See/Fluss-Farbunterscheidung im Water-Tab (analog zu 2D's
        # set_water_biomes_reference(), siehe map_display_2d.py _render_water_map()) -
        # ohne water_biomes_map teilen sich Seen und Flüsse dieselbe flache Blues-
        # Tiefenskala; ein flacher See (oft nur wenig über LAKE_VOLUME_THRESHOLD) sah
        # dadurch in 3D fast identisch zu unbewässertem Land aus (User-Report: "Seen
        # ... nicht zu sehen im 3D Modus"), da 2D diese fehlende Kontrastierung schon
        # länger über einen separaten Solid-Color-Overlay-Pass kompensiert, der in 3D
        # bisher komplett fehlte.
        self.water_biomes_reference = None
        # Separates Mini-Shader-Programm für die Wind-Vektor-Pfeile (Weather-Tab,
        # Layer "wind") - der Haupt-Terrain-Shader erwartet Normal/TexCoord/LightPos-
        # Varyings, die für simple farbige GL_LINES nicht gebraucht werden, siehe
        # shaders/3d_display/wind_vector.vert/.frag. Vorher zeigte "wind" in 3D nur
        # eine schwache Magnitude-Heatmap ohne Richtung ("nichts zu erkennen").
        self.wind_shader_program = None
        self.current_tab = "terrain"  # Aktueller Tab-Typ

        # Mesh-Daten
        self.mesh_vertices = None
        self.mesh_indices = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.vao = None  # Vertex Array Object

        # Shader-System
        self.shader_program = None
        self.shader_fallback_active = False

        # Matrix-Caching
        self.projection_matrix = None
        self.view_matrix = None
        self.model_matrix = None

        # Layer-Visibility für verschiedene Tabs
        self.layer_visibility = {
            "terrain": {"base": True, "slope": False},
            "geology": {"rock_map": True, "hardness_map": False},
            "weather": {"precipitation": True, "temperature": False, "wind": False, "humidity": False},
            "water": {"water_map": True, "soil_moisture": False, "erosion": False, "sedimentation": False,
                      "flow_map": False},
            "biome": {"biome_map": True, "super_biome_mask": False},
            "settlement": {"plots": True, "settlements": True, "landmarks": True, "roads": True, "civ_map": False}
        }

        # Overlay-Daten für verschiedene Tabs
        self.overlay_data = {
            "terrain": {"slope": None},
            "geology": {"rock_map": None, "hardness_map": None},
            "weather": {"precipitation": None, "temperature": None, "wind": None, "humidity": None},
            "water": {"water_map": None, "soil_moisture": None, "erosion": None, "sedimentation": None,
                      "flow_map": None},
            "biome": {"biome_map": None, "super_biome_mask": None},
            "settlement": {"plots": None, "settlements": [], "landmarks": [], "roads": [], "civ_map": None}
        }

        # Mouse-Interaction
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5

        # Animation-Timer für dynamische Effekte
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_time = 0.0

        # Mesh-Parameter
        self.terrain_scale_factor = 1.0
        self.terrain_height_scale = 1.0
        self.terrain_center_y = 0.0

    def initializeGL(self):
        """
        Funktionsweise: OpenGL-Initialisierung beim ersten Aufruf - ERWEITERT
        Aufgabe: Setup von OpenGL-State, Shaders und Rendering-Pipeline mit Error-Logging
        """
        try:
            print("DEBUG: Initializing OpenGL...")

            # OpenGL-Settings
            gl.glEnable(gl.GL_DEPTH_TEST)
            self._check_gl_error("after enabling depth test")

            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)
            self._check_gl_error("after enabling face culling")

            # Background-Color aus gui_default.py
            bg_color = CanvasSettings.CANVAS_3D["background_color"]
            gl.glClearColor(*bg_color)
            self._check_gl_error("after setting clear color")

            # Vertex Array Object erstellen
            self.vao = gl.glGenVertexArrays(1)
            self._check_gl_error("after creating VAO")

            # Shader-Programm laden
            print("DEBUG: Loading shaders...")
            self._load_shaders()
            self._compile_wind_shader()

            # Lighting-Setup
            if self.shader_program:
                print("DEBUG: Setting up lighting...")
                self._setup_lighting()
                print("DEBUG: OpenGL initialization completed successfully")
            else:
                print("DEBUG: OpenGL initialization completed but no shaders loaded")

        except Exception as e:
            error_msg = f"OpenGL initialization failed: {str(e)}"
            print(f"DEBUG: {error_msg}")
            self.rendering_error.emit(error_msg)

    def resizeGL(self, width, height):
        """
        Funktionsweise: Behandelt Fenster-Resize Events
        Aufgabe: Aktualisiert Viewport und Projection-Matrix
        Parameter: width, height (int) - Neue Fenster-Dimensionen
        """
        gl.glViewport(0, 0, width, height)
        self._update_projection_matrix()

    def paintGL(self):
        """
        Funktionsweise: Haupt-Rendering-Loop für jeden Frame
        Aufgabe: Rendert alle aktiven Layer in korrekter Reihenfolge
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not self._prepare_rendering():
            return

        # View-Matrix aktualisieren
        self._update_view_matrix()
        self._update_model_matrix()

        # Tab-spezifisches Rendering
        if self.current_tab == "terrain":
            self._render_terrain_tab()
        elif self.current_tab == "geology":
            self._render_geology_tab()
        elif self.current_tab == "weather":
            self._render_weather_tab()
        elif self.current_tab == "water":
            self._render_water_tab()
        elif self.current_tab == "biome":
            self._render_biome_tab()
        elif self.current_tab == "settlement":
            self._render_settlement_tab()

    def _prepare_rendering(self):
        """
        Funktionsweise: Bereitet Rendering vor und prüft Voraussetzungen
        Aufgabe: Validiert Shader, Daten und OpenGL-State
        Rückgabe: bool - True wenn Rendering möglich, False sonst
        """
        if self.heightmap is None:
            return False

        if self.shader_program is None and not self.shader_fallback_active:
            self._activate_fallback_rendering()

        if self.mesh_vertices is None:
            return False

        # Shader aktivieren (falls verfügbar)
        if self.shader_program:
            gl.glUseProgram(self.shader_program)
            self._upload_matrices()

        return True

    def _activate_fallback_rendering(self):
        """
        Funktionsweise: Aktiviert Fallback-Rendering ohne Shader
        Aufgabe: Einfaches Wireframe/Point-Rendering wenn Shader fehlschlagen
        """
        self.shader_fallback_active = True
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)  # Wireframe-Modus
        self.rendering_error.emit("Shader compilation failed - using wireframe fallback")

    def update_heightmap(self, heightmap, tab_type="terrain"):
        """
        Funktionsweise: Aktualisiert Heightmap und regeneriert Terrain-Mesh
        Aufgabe: Konvertiert 2D-Heightmap zu 3D-Mesh für OpenGL-Rendering
        Parameter: heightmap (numpy.ndarray), tab_type (str) - Neue Höhendaten und Tab-Typ
        """
        if not _validate_heightmap(heightmap):
            self.rendering_error.emit("Invalid heightmap data received")
            return

        self.heightmap = heightmap
        self.current_tab = tab_type
        self._calculate_terrain_scaling()

        # Contour-Intervall aus derselben Logik wie map_display_2d.py's
        # _calculate_contour_levels() ableiten (levels[1]-levels[0], da beide
        # Level-Arrays aus einem festen Intervall + np.arange() entstehen) -
        # eine Quelle der Wahrheit für 2D/3D-konsistente Höhenlinien-Abstände.
        contour_levels = _calculate_contour_levels(heightmap)
        if len(contour_levels) >= 2:
            self.contour_interval = float(contour_levels[1] - contour_levels[0])

        # _generate_terrain_mesh() erstellt/löscht OpenGL-Buffer direkt (glGenBuffers,
        # glDeleteBuffers, VBO-Upload) - das läuft hier NICHT innerhalb von paintGL(),
        # wo Qt automatisch den richtigen Kontext dieses Widgets aktiv setzt, sondern
        # als normaler Methodenaufruf (z.B. aus einem Signal-Handler). Ohne
        # makeCurrent() davor landen diese GL-Aufrufe im Kontext, der zufällig gerade
        # aktiv ist - bei mehreren MapDisplay3DWidget-Instanzen (eine pro Tab, siehe
        # base_tab.py) potenziell der FALSCHE Kontext oder gar keiner, was Buffer-
        # Erstellung fehlschlagen lässt oder Zustand eines anderen Tabs verändert.
        self.makeCurrent()
        try:
            self._generate_terrain_mesh()
        finally:
            self.doneCurrent()

        self.update()

    def update_shademap(self, shademap):
        """
        Funktionsweise: Aktualisiert Shademap für Terrain-Schattierung
        Aufgabe: Setzt neue Shadow-Daten für Upscaling auf Heightmap-Auflösung
        Parameter: shademap (numpy.ndarray, 2D) - Shadow-Map in beliebiger
        Auflösung (_upscale_shademap() skaliert bilinear auf die tatsächliche
        Heightmap-Größe hoch - anders als der frühere hartkodierte 64x64-Check
        hier nahelegte, der nie zu einer echten Rendering-Nutzung führte, da
        diese Methode bislang von keiner Aufrufstelle im Code erreicht wurde).
        """
        if shademap is not None and (not hasattr(shademap, 'ndim') or shademap.ndim != 2):
            self.rendering_error.emit("Shademap must be a 2D array")
            return

        self.shademap = shademap
        self.update()

    def set_shadow_overlay(self, checked: bool, angle: int = None):
        """
        Funktionsweise: Globaler Schatten-Toggle (Shell-Footer "Shadows"-
        Checkbox, siehe BaseMapTab.set_shadow_overlay()) - fehlte hier bisher
        komplett, weshalb der hasattr()-Guard im Aufrufer lautlos fehlschlug
        und der Toggle im 3D-Modus wirkungslos blieb.
        Aufgabe: Setzt self.shadows_enabled, das _render_terrain_base() (über
        _bind_shadow_texture()) bei jedem Frame abfragt.
        Parameter: checked (bool) - Schatten an/aus. angle (optional, derzeit
        ungenutzt) - der frühere manuelle Sonnenwinkel-Slider im Terrain-Tab
        wurde entfernt (siehe Kanban-Punkt "Shadow Angle entfernen"); die
        Shademap selbst wird bereits mit einem festen Standard-Sonnenwinkel
        geliefert (siehe BaseMapTab._push_data_to_current_display()).
        """
        self.shadows_enabled = checked
        self.update()

    def set_contour_overlay(self, checked: bool):
        """
        Funktionsweise: Globaler Contour-Lines-Toggle (Shell-Footer "Contour
        Lines"-Checkbox, siehe BaseMapTab.set_contour_overlay()) - fehlte hier
        bisher komplett (nur MapDisplay2D hatte set_contour_overlay()), weshalb
        der hasattr()-Guard im Aufrufer lautlos fehlschlug und Höhenlinien im
        3D-Modus nie sichtbar waren.
        Aufgabe: Setzt self.contours_enabled, das _render_terrain_base() bei
        jedem Frame als useContours-Uniform an den Fragment-Shader weiterreicht.
        """
        self.contours_enabled = checked
        self.update()

    def set_water_biomes_reference(self, water_biomes_map):
        """
        Funktionsweise: Hinterlegt water_biomes_map (0=kein Wasser, 1-3=Creek/
        River/Grand River, 4=Lake) für die See/Fluss-Farbunterscheidung in
        _render_overlay() - analog zu MapDisplay2D.set_water_biomes_reference().
        Fehlte hier bisher komplett (der hasattr()-Guard in water_tab.py
        schlug für 3D lautlos fehl), weshalb Seen in 3D nur die normale,
        oft kaum sichtbare Wassertiefen-Farbskala bekamen statt der klar
        abgesetzten See-Farbe.
        """
        self.water_biomes_reference = water_biomes_map
        self.update()

    def update_overlay_data(self, tab_type, layer_name, data):
        """
        Funktionsweise: Aktualisiert Overlay-Daten für spezifische Tabs
        Aufgabe: Setzt neue Overlay-Daten für verschiedene Visualisierungs-Layer
        Parameter: tab_type (str), layer_name (str), data - Tab, Layer und Daten
        """
        if tab_type not in self.overlay_data:
            self.rendering_error.emit(f"Unknown tab type: {tab_type}")
            return

        expected_shape = self.heightmap.shape if self.heightmap is not None else None

        if tab_type == "settlement" and layer_name in ["settlements", "landmarks", "roads"]:
            if not _validate_settlement_data(data):
                self.rendering_error.emit(f"Invalid settlement data for {layer_name}")
                return
        else:
            if not _validate_overlay_data(data, expected_shape):
                self.rendering_error.emit(f"Invalid overlay data for {tab_type}.{layer_name}")
                return

        self.overlay_data[tab_type][layer_name] = data

        # Animation starten falls nötig (z.B. für Wind)
        if tab_type == "weather" and layer_name == "wind" and data is not None:
            if self.layer_visibility["weather"]["wind"]:
                self.animation_timer.start(50)  # 20 FPS

        self.update()

    def set_layer_visibility(self, tab_type, layer_name, visible):
        """
        Funktionsweise: Schaltet Sichtbarkeit einzelner Render-Layer ein/aus
        Aufgabe: Toggle-Funktionalität für verschiedene Visualisierungs-Layer
        Parameter: tab_type (str), layer_name (str), visible (bool)
        """
        if tab_type not in self.layer_visibility:
            return

        if layer_name not in self.layer_visibility[tab_type]:
            return

        self.layer_visibility[tab_type][layer_name] = visible

        # Animation-Timer Management
        if tab_type == "weather" and layer_name == "wind":
            if visible and self.overlay_data["weather"]["wind"] is not None:
                self.animation_timer.start(50)
            elif not visible:
                # Prüfe ob andere Animationen laufen
                if not any(self.layer_visibility["weather"][key] for key in ["wind"]):
                    self.animation_timer.stop()

        self.update()

    def _calculate_terrain_scaling(self):
        """
        Funktionsweise: Berechnet automatische Skalierung basierend auf Heightmap-Daten
        Aufgabe: Dynamische Anpassung der Terrain-Größe und Höhen-Skalierung
        """
        if self.heightmap is None:
            return

        max_dimension = max(self.heightmap.shape)
        self.terrain_scale_factor = 10.0 / max_dimension  # Normiert auf 10 Einheiten

        # Feste, map_size- und sample-unabhängige Höhen-Skalierung: Die Karte
        # deckt real IMMER WORLD_SIZE_KM x WORLD_SIZE_KM ab (siehe
        # terrain_scale_factor: 10 Render-Einheiten = WORLD_SIZE_KM), egal wie
        # viele Pixel map_size hat. Vorher wurde relativ zu max_dimension
        # (Pixelanzahl!) und dem Min/Max der jeweiligen Stichprobe skaliert -
        # dadurch wirkten baugleiche Berge bei unterschiedlicher map_size oder
        # unterschiedlichem Seed nicht im selben Verhältnis zueinander, und
        # jede Heightmap wurde automatisch auf denselben visuellen Höheneindruck
        # gestreckt. Jetzt: 1 Render-Einheit entspricht immer WORLD_SIZE_KM/10 km.
        world_size_km = self.WORLD_SIZE_KM
        self.terrain_height_scale = 10.0 / (world_size_km * 1000.0)  # render units per meter

        # Vertikales Zentrum des Meshs in Welt-Y (Vertices bleiben unverändert
        # auf ihrer echten Höhe, siehe _generate_terrain_mesh) - die Kamera
        # zielt darauf statt fix auf (0,0,0). Ohne das lag reales Terrain
        # (Höhen typischerweise deutlich > 0) komplett außerhalb des
        # Kamera-Blickfelds und es wurde schlicht nichts gerendert.
        mid_height = (float(self.heightmap.min()) + float(self.heightmap.max())) / 2.0
        self.terrain_center_y = mid_height * self.terrain_height_scale

    def _load_shader_from_file(self, filepath):
        """
        Funktionsweise: Lädt Shader-Code aus Datei - ERWEITERT mit Debug-Logging
        Aufgabe: Liest GLSL-Shader aus externen .vert/.frag Dateien
        Parameter: filepath (str) - Pfad zur Shader-Datei
        Rückgabe: str - Shader-Code oder None bei Fehlern
        """
        print(f"DEBUG: Trying to load shader: {filepath}")

        # FIX: Korrekter Shader-Pfad
        corrected_path = filepath.replace("shader/", "shaders/3d_display/")
        print(f"DEBUG: Corrected shader path: {corrected_path}")

        import os
        if not os.path.exists(corrected_path):
            print(f"DEBUG: Shader file does not exist: {corrected_path}")
            print(f"DEBUG: Current working directory: {os.getcwd()}")

            # Versuche alternative Pfade
            alternative_paths = [
                filepath,  # Original
                f"shaders/{os.path.basename(filepath)}",  # Nur shaders/
                f"gui/shaders/{os.path.basename(filepath)}"  # gui/shaders/
            ]

            for alt_path in alternative_paths:
                print(f"DEBUG: Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    corrected_path = alt_path
                    print(f"DEBUG: Found shader at: {corrected_path}")
                    break
            else:
                self.rendering_error.emit(f"Shader file not found: {filepath}")
                return None

        try:
            with open(corrected_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"DEBUG: Successfully loaded shader, {len(content)} characters")
                return content
        except FileNotFoundError:
            self.rendering_error.emit(f"Shader file not found: {corrected_path}")
            return None
        except Exception as e:
            error_msg = f"Error reading shader file {corrected_path}: {str(e)}"
            print(f"DEBUG: {error_msg}")
            self.rendering_error.emit(error_msg)
            return None

    def _check_gl_error(self, context=""):
        """
        Funktionsweise: Prüft auf OpenGL-Errors - NEU
        Aufgabe: Debugging für OpenGL-Probleme
        Parameter: context (str) - Kontext-Info für Error-Messages
        Return: bool - True wenn kein Error
        """
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            error_msg = f"OpenGL Error {context}: {error}"
            print(f"DEBUG: {error_msg}")
            self.rendering_error.emit(error_msg)
            return False
        return True

    def _load_shaders(self):
        """
        Funktionsweise: Lädt und kompiliert Vertex- und Fragment-Shader aus Dateien
        Aufgabe: Erstellt Shader-Programm für 3D-Terrain-Rendering
        """
        # Hauptshader aus Dateien laden
        vertex_shader_code = self._load_shader_from_file("shader/terrain.vert")
        fragment_shader_code = self._load_shader_from_file("shader/terrain.frag")

        if vertex_shader_code is None or fragment_shader_code is None:
            # Fallback auf einfache Shader
            self._load_fallback_shaders()
            return

        try:
            vertex_shader = shaders.compileShader(vertex_shader_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_code, gl.GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            self.rendering_error.emit(f"Terrain shader compilation failed: {str(e)}")
            self._load_fallback_shaders()

    def _compile_wind_shader(self):
        """
        Funktionsweise: Kompiliert das separate Mini-Shader-Programm für die
        Wind-Vektor-Pfeile (siehe self.wind_shader_program oben).
        Aufgabe: Einmalig bei initializeGL() geladen, danach von
        _render_wind_vectors() pro Frame genutzt. Scheitert die Kompilierung,
        bleibt self.wind_shader_program None - _render_wind_vectors() no-opt
        dann still (Wind-Heatmap-Overlay bleibt trotzdem sichtbar).
        """
        vertex_shader_code = self._load_shader_from_file("shader/wind_vector.vert")
        fragment_shader_code = self._load_shader_from_file("shader/wind_vector.frag")

        if vertex_shader_code is None or fragment_shader_code is None:
            return

        try:
            vertex_shader = shaders.compileShader(vertex_shader_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_code, gl.GL_FRAGMENT_SHADER)
            self.wind_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            self.rendering_error.emit(f"Wind vector shader compilation failed: {str(e)}")
            self.wind_shader_program = None

    def _load_fallback_shaders(self):
        """
        Funktionsweise: Lädt einfache Fallback-Shader für Wireframe-Rendering
        Aufgabe: Backup-Shader wenn Hauptshader fehlschlagen
        """
        vertex_shader_code = self._load_shader_from_file("shader/simple.vert")
        fragment_shader_code = self._load_shader_from_file("shader/simple.frag")

        if vertex_shader_code is None or fragment_shader_code is None:
            self.shader_program = None
            return

        try:
            vertex_shader = shaders.compileShader(vertex_shader_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_code, gl.GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
            self.shader_fallback_active = True
        except Exception as e:
            self.rendering_error.emit(f"Fallback shader compilation failed: {str(e)}")
            self.shader_program = None

    def _setup_lighting(self):
        """
        Funktionsweise: Konfiguriert Lighting-Parameter für realistisches Shading
        Aufgabe: Setzt Light-Position aus gui_default.py und Standard-Material-Properties
        """
        if not self.shader_program:
            return

        gl.glUseProgram(self.shader_program)

        # Light-Position aus gui_default.py
        light_pos = CanvasSettings.CANVAS_3D["light_position"]
        light_pos_location = gl.glGetUniformLocation(self.shader_program, "lightPos")
        if light_pos_location >= 0:
            gl.glUniform3f(light_pos_location, *light_pos)

    def _generate_terrain_mesh(self):
        """
        Funktionsweise: Generiert 3D-Mesh aus 2D-Heightmap mit effizienter Vertex-Struktur
        Aufgabe: Erstellt Vertices, Normals und Indices für Terrain-Rendering - vollständig
        vektorisiert (vorher eine Python-Dreifachschleife: bei 256x256 = 65536 Vertices
        einzeln mit numpy.cross() berechnet, spürbar langsam bei realen Map-Sizes).
        Erzeugt bit-identische Ergebnisse zur alten Schleifen-Implementierung (gleiche
        Rand-Behandlung bei den Gradienten, gleiche Vertex-/Index-Reihenfolge).
        """
        if self.heightmap is None:
            return

        # Alte Buffer löschen
        self._cleanup_mesh_buffers()

        height, width = self.heightmap.shape
        heightmap = self.heightmap.astype(np.float32)

        # Vertex-Positionen (vectorized, (height, width) Grids)
        x_idx = np.arange(width, dtype=np.float32)
        y_idx = np.arange(height, dtype=np.float32)
        pos_x = np.broadcast_to(
            (x_idx / (width - 1) - 0.5) * width * self.terrain_scale_factor, (height, width))
        pos_z = np.broadcast_to(
            ((y_idx / (height - 1) - 0.5) * height * self.terrain_scale_factor)[:, None], (height, width))
        pos_y = heightmap * self.terrain_height_scale

        # Normalen: gleiche Rand-Behandlung wie die vorherige Pro-Vertex-Schleife
        # (Rand: einseitige Differenz, Innen: unhalbierte zentrale Differenz - deshalb
        # kein np.gradient(), das die Differenz innen halbiert).
        dz_dx = np.empty_like(heightmap)
        dz_dx[:, 1:-1] = heightmap[:, 2:] - heightmap[:, :-2]
        dz_dx[:, 0] = heightmap[:, 1] - heightmap[:, 0]
        dz_dx[:, -1] = heightmap[:, -1] - heightmap[:, -2]

        dz_dy = np.empty_like(heightmap)
        dz_dy[1:-1, :] = heightmap[2:, :] - heightmap[:-2, :]
        dz_dy[0, :] = heightmap[1, :] - heightmap[0, :]
        dz_dy[-1, :] = heightmap[-1, :] - heightmap[-2, :]

        dx = dz_dx * self.terrain_height_scale
        dy = dz_dy * self.terrain_height_scale

        # Cross-Product der beiden Tangenten entlang der Vertex-Nachbarn: Tangente
        # X-Richtung (Spalten) = (step, dx, 0), Tangente Z-Richtung (Zeilen) =
        # (0, dy, step) - cross(T_x, T_z) = (-dx*step, step^2, -dy*step).
        # Frühere Fassung (vor diesem Fix) hatte X/Z vertauscht und normal_z als
        # von dy unabhängige Konstante -step^2 - dadurch beeinflussten Nord/Süd-
        # Gefälle (Zeilen-Gradient dy) die Beleuchtung nie, nur Ost/West-Gefälle
        # (dx, fälschlich im X-Slot der alten Formel gelandet). Sichtbar geworden
        # als "Sonne kommt aus Osten" trotz Süd-Lichtposition, siehe [[project-3d-sun-normal-fix]] -
        # jede Lichtrichtung mit -Z-Anteil (Süden) beleuchtete dadurch praktisch
        # das gesamte Terrain gleichmäßig statt gezielt Süd-Hänge.
        step_size = self.terrain_scale_factor
        normal_x = -dx * step_size
        normal_y = np.full((height, width), step_size ** 2, dtype=np.float32)
        normal_z = -dy * step_size

        length = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
        safe_length = np.where(length > 0, length, 1.0)
        normal_x = np.where(length > 0, normal_x / safe_length, 0.0)
        normal_y = np.where(length > 0, normal_y / safe_length, 1.0)
        normal_z = np.where(length > 0, normal_z / safe_length, 0.0)

        # Texture-Coordinates
        tex_u = np.broadcast_to(x_idx / (width - 1), (height, width))
        tex_v = np.broadcast_to((y_idx / (height - 1))[:, None], (height, width))

        # Interleaved Vertex-Layout wie zuvor: [pos_x, pos_y, pos_z, nx, ny, nz, u, v]
        # pro Vertex, in derselben y-major/x-minor Reihenfolge wie die alte Schleife.
        vertex_grid = np.stack(
            [pos_x, pos_y, pos_z, normal_x, normal_y, normal_z, tex_u, tex_v], axis=-1)
        self.mesh_vertices = vertex_grid.reshape(-1).astype(np.float32)

        # Indices für Triangles (zwei pro Quad, gleiche Winkel-Reihenfolge wie zuvor)
        yy, xx = np.meshgrid(np.arange(height - 1), np.arange(width - 1), indexing='ij')
        top_left = yy * width + xx
        top_right = yy * width + (xx + 1)
        bottom_left = (yy + 1) * width + xx
        bottom_right = (yy + 1) * width + (xx + 1)

        triangle_1 = np.stack([top_left, bottom_left, top_right], axis=-1)
        triangle_2 = np.stack([top_right, bottom_left, bottom_right], axis=-1)
        indices = np.stack([triangle_1, triangle_2], axis=-2)

        self.mesh_indices = indices.reshape(-1).astype(np.uint32)

        # OpenGL-Buffers erstellen
        self._create_mesh_buffers()

    def _create_mesh_buffers(self):
        """
        Funktionsweise: Erstellt OpenGL-Buffers für Mesh-Daten
        Aufgabe: Upload von Vertex- und Index-Daten zur GPU
        """
        if self.mesh_vertices is None or self.mesh_indices is None:
            return

        # VAO binden
        gl.glBindVertexArray(self.vao)

        # Vertex Buffer Object
        self.vertex_buffer = glvbo.VBO(self.mesh_vertices)
        self.vertex_buffer.bind()

        # Index Buffer Object
        self.index_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh_indices.nbytes, self.mesh_indices, gl.GL_STATIC_DRAW)

        # Vertex-Attribute konfigurieren
        stride = 8 * 4  # 8 floats * 4 bytes

        # Position (location = 0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)

        # Normal (location = 1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(3 * 4))

        # Texture-Coords (location = 2)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(6 * 4))

        # VAO unbinden
        gl.glBindVertexArray(0)

    def _cleanup_mesh_buffers(self):
        """
        Funktionsweise: Löscht bestehende Mesh-Buffers für Memory-Management
        Aufgabe: Cleanup von GPU-Ressourcen vor Neuerstellung
        """
        if self.vertex_buffer:
            self.vertex_buffer.delete()
            self.vertex_buffer = None

        if self.index_buffer:
            gl.glDeleteBuffers(1, [self.index_buffer])
            self.index_buffer = None

    def _update_projection_matrix(self):
        """
        Funktionsweise: Aktualisiert Projection-Matrix bei Fenster-Resize
        Aufgabe: Setzt Perspective-Projection mit FOV aus gui_default.py
        """
        if self.height() == 0:
            return

        aspect_ratio = self.width() / self.height()
        self.projection_matrix = _create_perspective_matrix(
            fov=self.fov,
            aspect=aspect_ratio,
            near=0.1,
            far=100.0
        )

    def _update_view_matrix(self):
        """
        Funktionsweise: Aktualisiert View-Matrix basierend auf Fixed-Axis Camera
        Aufgabe: Berechnet Camera-Position für Elevation und Azimuth-Rotation
        """
        # Fixed-Axis Camera: Feste Elevation, variable Azimuth-Rotation
        elevation_rad = math.radians(self.camera_elevation)
        azimuth_rad = math.radians(self.camera_azimuth)

        # Camera-Position berechnen (relativ zum vertikalen Terrain-Zentrum,
        # nicht zum Welt-Ursprung - reale Höhen liegen nie bei Y=0)
        eye_x = self.camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        eye_y = self.terrain_center_y + self.camera_distance * math.sin(elevation_rad)
        eye_z = self.camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)

        eye = [eye_x, eye_y, eye_z]
        target = [0, self.terrain_center_y, 0]  # Blick zum Terrain-Zentrum
        up = [0, 1, 0]  # Y ist oben

        self.view_matrix = _create_lookat_matrix(eye, target, up)

    def _update_model_matrix(self):
        """
        Funktionsweise: Aktualisiert Model-Matrix für Terrain-Positionierung
        Aufgabe: Setzt Terrain-Position im World-Space
        """
        self.model_matrix = _create_model_matrix(
            translation=(0, 0, 0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1)
        )

    def _upload_matrices(self):
        """
        Funktionsweise: Uploaded Matrix-Daten zu GPU-Shadern
        Aufgabe: Setzt Uniform-Variablen für Vertex-Transformationen

        Die Matrizen werden in _create_perspective_matrix()/_create_lookat_matrix()/
        _create_model_matrix() in Standard-Zeilen-Major-Schreibweise aufgebaut
        (z.B. matrix[2,3] = Translations-Term). numpy .flatten() liefert diese
        Daten row-major. glUniformMatrix4fv() mit transpose=GL_FALSE erwartet
        column-major Daten - ohne Transpose landete z.B. bei der Projection-
        Matrix der W-Term (matrix[3,2]=-1) an der falschen Stelle, wodurch die
        perspektivische Division kaputt ging und nichts mehr sichtbar war.
        transpose=GL_TRUE lässt OpenGL die row-major Daten korrekt transponieren.
        """
        if not self.shader_program:
            return

        # Projection-Matrix
        if self.projection_matrix is not None:
            proj_location = gl.glGetUniformLocation(self.shader_program, "projection")
            if proj_location >= 0:
                gl.glUniformMatrix4fv(proj_location, 1, gl.GL_TRUE, self.projection_matrix.flatten())

        # View-Matrix
        if self.view_matrix is not None:
            view_location = gl.glGetUniformLocation(self.shader_program, "view")
            if view_location >= 0:
                gl.glUniformMatrix4fv(view_location, 1, gl.GL_TRUE, self.view_matrix.flatten())

        # Model-Matrix
        if self.model_matrix is not None:
            model_location = gl.glGetUniformLocation(self.shader_program, "model")
            if model_location >= 0:
                gl.glUniformMatrix4fv(model_location, 1, gl.GL_TRUE, self.model_matrix.flatten())

    def _render_terrain_tab(self):
        """
        Funktionsweise: Rendert Terrain-Tab mit Heightmap und Shademap
        Aufgabe: Basis-Terrain mit 2D-Map Coloring und Shadow-Integration
        """
        if not self.layer_visibility["terrain"]["base"]:
            return

        self._render_terrain_base()

        if self.layer_visibility["terrain"]["slope"]:
            self._render_overlay("terrain", "slope")

    def _render_geology_tab(self):
        """
        Funktionsweise: Rendert Geology-Tab mit Rock- und Hardness-Maps
        Aufgabe: Terrain mit Gesteins- und Festigkeits-Overlays
        """
        self._render_terrain_base()

        if self.layer_visibility["geology"]["rock_map"]:
            self._render_overlay("geology", "rock_map")

        if self.layer_visibility["geology"]["hardness_map"]:
            self._render_overlay("geology", "hardness_map")

    def _render_weather_tab(self):
        """
        Funktionsweise: Rendert Weather-Tab mit Klima-Daten
        Aufgabe: Terrain mit Precipitation, Temperature, Wind und Humidity-Overlays
        """
        self._render_terrain_base()

        weather_layers = ["precipitation", "temperature", "wind", "humidity"]
        for layer in weather_layers:
            if self.layer_visibility["weather"][layer]:
                self._render_overlay("weather", layer)
                if layer == "wind":
                    self._render_wind_vectors()

    def _render_water_tab(self):
        """
        Funktionsweise: Rendert Water-Tab mit Hydrologie-Daten
        Aufgabe: Terrain mit Water, Soil-Moisture, Erosion und Sedimentation-Overlays
        """
        self._render_terrain_base()

        water_layers = ["water_map", "soil_moisture", "erosion", "sedimentation", "flow_map"]
        for layer in water_layers:
            if self.layer_visibility["water"][layer]:
                self._render_overlay("water", layer)

    def _render_biome_tab(self):
        """
        Funktionsweise: Rendert Biome-Tab mit 4x Auflösung
        Aufgabe: Terrain mit hochauflösender Biome-Map
        """
        self._render_terrain_base()

        if self.layer_visibility["biome"]["biome_map"]:
            self._render_overlay("biome", "biome_map")

        if self.layer_visibility["biome"]["super_biome_mask"]:
            self._render_overlay("biome", "super_biome_mask")

    def _render_settlement_tab(self):
        """
        Funktionsweise: Rendert Settlement-Tab mit Plots und Markern
        Aufgabe: Terrain mit Plot-Geometrie und Settlement-Features
        """
        self._render_terrain_base()

        if self.layer_visibility["settlement"]["civ_map"]:
            self._render_overlay("settlement", "civ_map")

        if self.layer_visibility["settlement"]["plots"]:
            self._render_plot_boundaries()

        if self.layer_visibility["settlement"]["settlements"]:
            self._render_settlement_markers("settlements")

        if self.layer_visibility["settlement"]["landmarks"]:
            self._render_settlement_markers("landmarks")

        if self.layer_visibility["settlement"]["roads"]:
            self._render_settlement_markers("roads")

    def _render_terrain_base(self):
        """
        Funktionsweise: Rendert Basis-Terrain-Mesh
        Aufgabe: Grundlegendes Terrain-Rendering mit Standard-Shading
        """
        if self.vertex_buffer is None or self.mesh_indices is None:
            return

        if self.shader_program:
            # Shader-Parameter setzen
            render_mode_location = gl.glGetUniformLocation(self.shader_program, "renderMode")
            if render_mode_location >= 0:
                mode_value = {"terrain": 0, "geology": 1, "weather": 2, "water": 3, "biome": 4, "settlement": 5}
                gl.glUniform1i(render_mode_location, mode_value.get(self.current_tab, 0))

            height_scale_location = gl.glGetUniformLocation(self.shader_program, "heightScale")
            if height_scale_location >= 0:
                gl.glUniform1f(height_scale_location, self.terrain_height_scale)

            max_height_location = gl.glGetUniformLocation(self.shader_program, "maxHeight")
            if max_height_location >= 0:
                max_height = np.max(self.heightmap) * self.terrain_height_scale
                gl.glUniform1f(max_height_location, max_height)

            # Overlay-Parameter (falls verfügbar)
            use_overlay_location = gl.glGetUniformLocation(self.shader_program, "useOverlay")
            if use_overlay_location >= 0:
                gl.glUniform1i(use_overlay_location, 0)  # Default: kein Overlay

            overlay_strength_location = gl.glGetUniformLocation(self.shader_program, "overlayStrength")
            if overlay_strength_location >= 0:
                gl.glUniform1f(overlay_strength_location, 0.5)  # Default-Stärke

            # Contour-Lines jeden Frame neu setzen (wie useOverlay/useShadows oben -
            # Shader-Uniforms behalten sonst ihren letzten Wert über Frames hinweg).
            use_contours_location = gl.glGetUniformLocation(self.shader_program, "useContours")
            if use_contours_location >= 0:
                gl.glUniform1i(use_contours_location, 1 if self.contours_enabled else 0)

            contour_interval_location = gl.glGetUniformLocation(self.shader_program, "contourInterval")
            if contour_interval_location >= 0:
                gl.glUniform1f(contour_interval_location, self.contour_interval)

            # useShadows jeden Frame neu setzen (wie useOverlay oben) - Shader-Uniforms
            # behalten sonst ihren letzten Wert über Draw-Calls/Frames hinweg. Vorher
            # wurde hier IMMER auf 0 zurückgesetzt (_render_shadows() aktivierte den
            # Uniform separat, NACH diesem Draw-Call - zu spät, um noch zu wirken, siehe
            # Git-Historie) - jetzt wird die Schatten-Textur HIER, VOR dem Draw-Call,
            # erstellt/gebunden, gilt dadurch tab-übergreifend für jeden Aufrufer von
            # _render_terrain_base() (Terrain/Geology/Weather/Water/Biome/Settlement -
            # alle rendern über diese eine gemeinsame Basis-Mesh-Funktion), gesteuert vom
            # globalen Shell-Footer "Shadows"-Toggle (self.shadows_enabled, siehe
            # set_shadow_overlay()).
            shadow_texture_id = self._bind_shadow_texture()

        elif self.shader_fallback_active:
            # Fallback-Shader Parameter
            wireframe_color_location = gl.glGetUniformLocation(self.shader_program, "wireframeColor")
            if wireframe_color_location >= 0:
                gl.glUniform3f(wireframe_color_location, 0.7, 0.7, 0.7)  # Grau
            shadow_texture_id = None

        else:
            shadow_texture_id = None

        # VAO binden und rendern
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.mesh_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        if shadow_texture_id is not None:
            gl.glDeleteTextures(1, [shadow_texture_id])

    def _bind_shadow_texture(self):
        """
        Funktionsweise: Erstellt bei aktiviertem Schatten-Toggle eine Ein-Kanal-
        Textur aus der (auf Heightmap-Auflösung hochskalierten) Shademap, bindet
        sie an Texture-Unit 2 (Unit 1 ist bereits von overlayTexture belegt, siehe
        _render_overlay()) und setzt shadowMap/useShadows entsprechend. Setzt
        useShadows=0, wenn Schatten deaktiviert sind oder keine Shademap vorliegt.
        Aufgabe: Gemeinsamer Schatten-Setup-Schritt für _render_terrain_base(),
        wird VOR dessen Draw-Call aufgerufen (anders als die frühere _render_
        shadows(), die NACH dem Draw-Call lief und dadurch nie sichtbar wurde).
        Return: Textur-ID zum späteren Löschen (nach dem Draw-Call), oder None.
        """
        use_shadows_location = gl.glGetUniformLocation(self.shader_program, "useShadows")

        if not self.shadows_enabled or self.shademap is None or self.heightmap is None:
            if use_shadows_location >= 0:
                gl.glUniform1i(use_shadows_location, 0)
            return None

        upscaled_shadows = _upscale_shademap(self.shademap, self.heightmap.shape)
        if upscaled_shadows is None:
            if use_shadows_location >= 0:
                gl.glUniform1i(use_shadows_location, 0)
            return None

        shadow_data = np.ascontiguousarray(np.clip(upscaled_shadows, 0.0, 1.0).astype(np.float32))

        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R32F, shadow_data.shape[1], shadow_data.shape[0],
                         0, gl.GL_RED, gl.GL_FLOAT, shadow_data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glActiveTexture(gl.GL_TEXTURE2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        shadow_map_location = gl.glGetUniformLocation(self.shader_program, "shadowMap")
        if shadow_map_location >= 0:
            gl.glUniform1i(shadow_map_location, 2)
        if use_shadows_location >= 0:
            gl.glUniform1i(use_shadows_location, 1)

        return texture_id

    def _render_overlay(self, tab_type, layer_name):
        """
        Funktionsweise: Rendert Tab-spezifische Overlay-Layer als Textur auf
        dem Terrain-Mesh.
        Aufgabe: Färbt overlay_data über _colorize_layer() zu einer RGB-
        Textur ein, lädt sie hoch und zeichnet das Mesh ein zweites Mal mit
        useOverlay=1 (der Shader mischt Basis- und Overlay-Farbe intern PRO
        FRAGMENT, siehe getWeatherColor() etc. in terrain.frag - ein
        einziger zusätzlicher Draw-Call pro sichtbarem Layer reicht). Der
        zweite Draw-Call liegt auf EXAKT derselben Tiefe wie der Basis-Pass
        aus _render_terrain_base() (identisches Mesh) - Default-Tiefentest
        GL_LESS würde ihn deshalb komplett verdecken, daher hier temporär
        auf GL_LEQUAL umgeschaltet.
        Parameter: tab_type (str), layer_name (str) - Tab und Layer-Identifikation
        """
        overlay_data = self.overlay_data[tab_type][layer_name]
        if overlay_data is None or not self.shader_program or self.vertex_buffer is None:
            return

        try:
            rgb = _colorize_layer(np.asarray(overlay_data), layer_name)
        except Exception:
            return

        # See-Pixel klar absetzen (analog zu map_display_2d.py's _render_water_map()
        # Lake-Overlay-Pass, alpha=0.85) - ohne das teilen sich Seen und Flüsse
        # dieselbe flache Blues-Tiefenskala und flache Seen sind kaum von Land zu
        # unterscheiden (siehe set_water_biomes_reference()).
        if layer_name == "water_map" and self.water_biomes_reference is not None:
            water_biomes = np.asarray(self.water_biomes_reference)
            if water_biomes.shape[:2] == rgb.shape[:2]:
                lake_mask = water_biomes == 4
                lake_color = np.array([11, 61, 145], dtype=np.float32)  # #0b3d91
                rgb = rgb.astype(np.float32)
                rgb[lake_mask] = rgb[lake_mask] * 0.15 + lake_color * 0.85
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        texture_id = gl.glGenTextures(1)
        try:
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, rgb.shape[1], rgb.shape[0],
                             0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(rgb))
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

            overlay_tex_location = gl.glGetUniformLocation(self.shader_program, "overlayTexture")
            if overlay_tex_location >= 0:
                gl.glUniform1i(overlay_tex_location, 1)

            use_overlay_location = gl.glGetUniformLocation(self.shader_program, "useOverlay")
            if use_overlay_location >= 0:
                gl.glUniform1i(use_overlay_location, 1)

            overlay_strength_location = gl.glGetUniformLocation(self.shader_program, "overlayStrength")
            if overlay_strength_location >= 0:
                gl.glUniform1f(overlay_strength_location, 0.75)

            render_mode_location = gl.glGetUniformLocation(self.shader_program, "renderMode")
            if render_mode_location >= 0:
                mode_value = {"terrain": 0, "geology": 1, "weather": 2, "water": 3, "biome": 4, "settlement": 5}
                gl.glUniform1i(render_mode_location, mode_value.get(tab_type, 0))

            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glBindVertexArray(self.vao)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.mesh_indices), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glDepthFunc(gl.GL_LESS)

            if use_overlay_location >= 0:
                gl.glUniform1i(use_overlay_location, 0)
        finally:
            gl.glDeleteTextures(1, [texture_id])

    def _render_wind_vectors(self):
        """
        Funktionsweise: Zeichnet Windrichtungs-Pfeile als GL_LINES über dem
        Terrain-Mesh (analog zu map_display_2d.py's matplotlib-Quiver in
        _render_wind_map()) - die reine Magnitude-Heatmap aus _render_overlay()
        zeigt nur Windstärke, keine Richtung, und blendet auf dem Terrain kaum
        sichtbar ein (User-Report: "nichts zu erkennen").
        Aufgabe: Sampled ein grobes Raster aus overlay_data["weather"]["wind"]
        ((H,W,2) u/v m/s in Spalten-/Zeilen-Richtung, siehe _generate_terrain_
        mesh's pos_x/pos_z-Konvention: u->X/Ost-West, v->Z/Süd-Nord), platziert
        an jedem Sample-Punkt einen kleinen Pfeil (Schaft + zwei Widerhaken)
        knapp über der Terrain-Oberfläche. Länge/Farbe skalieren mit der
        lokalen Windstärke (weiß=schwach, orange-rot=stark).
        """
        if not self.wind_shader_program or self.heightmap is None:
            return

        wind_data = self.overlay_data.get("weather", {}).get("wind")
        if wind_data is None:
            return

        wind_data = np.asarray(wind_data)
        height, width = self.heightmap.shape
        if wind_data.ndim != 3 or wind_data.shape[2] != 2 or wind_data.shape[:2] != (height, width):
            return  # Shape-Mismatch (z.B. während eines LOD-Übergangs) - nächster Frame passt wieder

        grid = 14
        y_idx = np.linspace(0, height - 1, min(grid, height)).astype(int)
        x_idx = np.linspace(0, width - 1, min(grid, width)).astype(int)
        yy, xx = np.meshgrid(y_idx, x_idx, indexing='ij')

        u = wind_data[yy, xx, 0].astype(np.float32)
        v = wind_data[yy, xx, 1].astype(np.float32)
        magnitude = np.sqrt(u ** 2 + v ** 2)
        max_mag = float(magnitude.max())
        if max_mag < 1e-6:
            return

        pos_x = (xx.astype(np.float32) / (width - 1) - 0.5) * width * self.terrain_scale_factor
        pos_z = (yy.astype(np.float32) / (height - 1) - 0.5) * height * self.terrain_scale_factor
        terrain_h = self.heightmap[yy, xx].astype(np.float32)
        hover = 0.05  # etwas über der Oberfläche schweben, gegen Z-Fighting mit dem Mesh
        pos_y = terrain_h * self.terrain_height_scale + hover

        # Pfeillänge proportional zur lokalen Windstärke (relativ zum stärksten
        # Sample im aktuellen Raster), gedeckelt auf einen Bruchteil des
        # Rasterabstands, damit sich benachbarte Pfeile nicht überlappen.
        cell_span = self.terrain_scale_factor * max(width, height) / grid
        max_arrow_len = cell_span * 0.8
        safe_mag = np.where(magnitude > 1e-6, magnitude, 1.0)
        unit_x = u / safe_mag
        unit_z = v / safe_mag
        arrow_len = (magnitude / max_mag) * max_arrow_len
        dir_x = unit_x * arrow_len
        dir_z = unit_z * arrow_len

        # Widerhaken: unit_dir um +-150 Grad in der XZ-Ebene gedreht, kurze Länge.
        theta = np.radians(150.0)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        barb1_x = unit_x * cos_t - unit_z * sin_t
        barb1_z = unit_x * sin_t + unit_z * cos_t
        barb2_x = unit_x * cos_t + unit_z * sin_t
        barb2_z = -unit_x * sin_t + unit_z * cos_t
        barb_len = arrow_len * 0.35

        tip_x, tip_z = pos_x + dir_x, pos_z + dir_z

        # Farbe: weiß (schwach) -> orange-rot (stark), linear nach Magnitude.
        t = (magnitude / max_mag)[..., None]
        calm_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        strong_color = np.array([1.0, 0.25, 0.05], dtype=np.float32)
        color = calm_color * (1 - t) + strong_color * t

        def _seg(ax_, ay_, az_, bx_, by_, bz_):
            a = np.stack([ax_, ay_, az_], axis=-1)
            b = np.stack([bx_, by_, bz_], axis=-1)
            return a, b

        shaft_a, shaft_b = _seg(pos_x, pos_y, pos_z, tip_x, pos_y, tip_z)
        barb1_a, barb1_b = _seg(tip_x, pos_y, tip_z, tip_x + barb1_x * barb_len, pos_y, tip_z + barb1_z * barb_len)
        barb2_a, barb2_b = _seg(tip_x, pos_y, tip_z, tip_x + barb2_x * barb_len, pos_y, tip_z + barb2_z * barb_len)

        positions = np.stack([shaft_a, shaft_b, barb1_a, barb1_b, barb2_a, barb2_b], axis=-2)  # (grid,grid,6,3)
        colors = np.broadcast_to(color[:, :, None, :], positions.shape)

        vertex_data = np.concatenate([positions, colors], axis=-1).reshape(-1, 6).astype(np.float32)
        vertex_data = np.ascontiguousarray(vertex_data)

        vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        try:
            gl.glUseProgram(self.wind_shader_program)

            for name, matrix in (("model", self.model_matrix), ("view", self.view_matrix),
                                  ("projection", self.projection_matrix)):
                if matrix is None:
                    continue
                location = gl.glGetUniformLocation(self.wind_shader_program, name)
                if location >= 0:
                    gl.glUniformMatrix4fv(location, 1, gl.GL_TRUE, matrix.flatten())

            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STREAM_DRAW)

            stride = 6 * 4
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(3 * 4))

            gl.glLineWidth(2.0)
            gl.glDrawArrays(gl.GL_LINES, 0, vertex_data.shape[0])

            gl.glBindVertexArray(0)
        finally:
            gl.glDeleteBuffers(1, [vbo])
            gl.glDeleteVertexArrays(1, [vao])

    def _render_plot_boundaries(self):
        """
        Funktionsweise: Rendert das PlotPhysicsSystem-Ergebnis als texturierten
        "Skin" auf dem Terrain-Mesh statt eigener 3D-Wireframe-/Marker-
        Geometrie (Nutzer-Vorgabe: "die 2D-Darstellung als Skin auf das
        Terrain legen", siehe [[project-settlement-plot-physics-rebuild]]
        Teil 4) - overlay_data["settlement"]["plots"] enthält bereits die
        fertig gerasterte (H,W,4)-RGBA-Textur (siehe map_display_2d.py's
        rasterize_plot_boundaries_rgba(), gepusht von settlement_tab.py's
        apply_3d_overlays()). Hier nur noch Hochladen + zweiter Draw-Call mit
        useAlphaOverlay=1 (siehe terrain.frag) - echte Transparenz an
        unbemalten Stellen statt des pauschalen overlayStrength-Mix der
        übrigen Scalar-Overlays.
        Aufgabe: Zeichnet Plot-Kantennetz/Straßen-Tiers/Wildnisgrenzen/Kerne/
        Nodes auf der Terrain-Oberfläche.
        """
        rgba = self.overlay_data["settlement"]["plots"]
        if (rgba is None or not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] != 4
                or not self.shader_program or self.vertex_buffer is None):
            return

        texture_id = gl.glGenTextures(1)
        try:
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, rgba.shape[1], rgba.shape[0],
                             0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(rgba))
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

            overlay_tex_location = gl.glGetUniformLocation(self.shader_program, "overlayTexture")
            if overlay_tex_location >= 0:
                gl.glUniform1i(overlay_tex_location, 1)

            use_overlay_location = gl.glGetUniformLocation(self.shader_program, "useOverlay")
            if use_overlay_location >= 0:
                gl.glUniform1i(use_overlay_location, 1)

            use_alpha_overlay_location = gl.glGetUniformLocation(self.shader_program, "useAlphaOverlay")
            if use_alpha_overlay_location >= 0:
                gl.glUniform1i(use_alpha_overlay_location, 1)

            overlay_strength_location = gl.glGetUniformLocation(self.shader_program, "overlayStrength")
            if overlay_strength_location >= 0:
                gl.glUniform1f(overlay_strength_location, 1.0)

            render_mode_location = gl.glGetUniformLocation(self.shader_program, "renderMode")
            if render_mode_location >= 0:
                gl.glUniform1i(render_mode_location, 5)  # settlement

            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glBindVertexArray(self.vao)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.mesh_indices), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glDepthFunc(gl.GL_LESS)

            if use_overlay_location >= 0:
                gl.glUniform1i(use_overlay_location, 0)
            if use_alpha_overlay_location >= 0:
                gl.glUniform1i(use_alpha_overlay_location, 0)
        finally:
            gl.glDeleteTextures(1, [texture_id])

    def _render_settlement_markers(self, marker_type):
        """
        Funktionsweise: Rendert Settlement-Feature-Marker
        Aufgabe: 3D-Marker für Settlements, Landmarks und Roads
        Parameter: marker_type (str) - Typ der Settlement-Features
        """
        marker_data = self.overlay_data["settlement"][marker_type]
        if not marker_data:
            return

        # TODO: Implementierung verschiedener Marker-Typen
        # - Settlements: Größere Zylinder/Kugeln
        # - Landmarks: Icon-basierte Marker
        # - Roads: Kleinere Verbindungspunkte

    def _update_animation(self):
        """
        Funktionsweise: Aktualisiert Animation-Zeit für dynamische Effekte
        Aufgabe: Inkrementiert Animation-Timer für Wind-Vektoren und andere Animationen
        """
        self.animation_time += 0.1

        # Prüfe ob Animationen aktiv sind
        animations_active = (
            self.layer_visibility["weather"]["wind"] and
            self.overlay_data["weather"]["wind"] is not None
        )

        if animations_active:
            self.update()

    def mousePressEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Press Events
        Aufgabe: Startet Camera-Rotation und Vertex-Selection
        """
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Move Events für Fixed-Axis Camera-Controls
        Aufgabe: Implementiert Azimuth-Rotation um Z-Achse (feste Elevation)
        """
        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # Nur Azimuth-Rotation (um Z-Achse)
            self.camera_azimuth += dx * self.mouse_sensitivity

            # Normalisiere Azimuth auf 0-360°
            self.camera_azimuth = self.camera_azimuth % 360.0

            self.camera_changed.emit(self.camera_elevation, self.camera_azimuth, self.camera_distance)
            self.update()

        self.last_mouse_pos = event.pos()

    def wheelEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Wheel Events für Zoom-Funktionalität
        Aufgabe: Implementiert Zoom mit Distanz-Begrenzungen
        """
        if event.angleDelta().y() == 0:
            return

        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        new_distance = self.camera_distance * zoom_factor

        # Zoom-Grenzen
        min_distance = 2.0
        max_distance = 50.0
        self.camera_distance = max(min_distance, min(max_distance, new_distance))

        self.camera_changed.emit(self.camera_elevation, self.camera_azimuth, self.camera_distance)
        self.update()

    def reset_camera(self):
        """
        Funktionsweise: Setzt Camera auf Standard-Position zurück
        Aufgabe: Reset zu Default-View aus gui_default.py
        """
        self.camera_distance = CanvasSettings.CANVAS_3D["camera_distance"]
        self.camera_elevation = 55.0
        self.camera_azimuth = 0.0

        self.camera_changed.emit(self.camera_elevation, self.camera_azimuth, self.camera_distance)
        self.update()


class MapDisplay3DWidget(QWidget):
    """
    Funktionsweise: Wrapper-Widget für 3D-Display mit Tab-spezifischen Controls
    Aufgabe: Kombiniert 3D-Display mit dynamischen Layer-Controls basierend auf aktivem Tab
    """

    def __init__(self, parent=None):
        """
        Funktionsweise: Initialisiert 3D-Widget mit Control-Panel
        Aufgabe: Setup von 3D-Display und zugehörigen UI-Controls
        """
        super().__init__(parent)

        self.current_tab = "terrain"
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """
        Funktionsweise: Erstellt UI-Layout mit 3D-Display und Tab-spezifischen Controls
        Aufgabe: Layout-Setup für 3D-Rendering mit dynamischen Layer-Controls
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 3D-Display
        self.display_3d = MapDisplay3D()
        layout.addWidget(self.display_3d)

        # Dynamic Control-Panel (wird je nach Tab angepasst)
        self.control_layout = QHBoxLayout()
        self.control_widgets = {}

        # Camera-Reset Button (immer sichtbar)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.control_layout.addWidget(self.reset_camera_button)

        self.control_layout.addStretch()
        layout.addLayout(self.control_layout)

        # Initial Controls für Terrain-Tab
        self._setup_terrain_controls()

    def _setup_terrain_controls(self):
        """
        Funktionsweise: Erstellt Controls für Terrain-Tab
        Aufgabe: Basis-Terrain und Shadows-Sichtbarkeit (KEINE Layer-Auswahl -
        welcher Daten-Layer auf dem Mesh liegt, entscheiden ausschließlich die
        radio buttons über dem Canvas, siehe BaseMapTab._push_data_to_
        current_display()/_LAYER_SELECTION_KEYS_3D in gui/tabs/base_tab.py -
        die frühere separate "Slope"-Checkbox hier war dazu redundant und
        konnte unabhängig von der 2D-Auswahl aktiv bleiben; entfernt).
        """
        self._clear_tab_controls()

        # Terrain Base (Mesh-Sichtbarkeit selbst, keine Layer-Auswahl)
        terrain_checkbox = QCheckBox("Terrain")
        terrain_checkbox.setChecked(True)
        terrain_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("terrain", "base", checked)
        )
        self.control_widgets["terrain_base"] = terrain_checkbox
        self.control_layout.insertWidget(0, terrain_checkbox)

        # Shadows (Beleuchtungs-Sichtbarkeit, keine Layer-Auswahl) - ruft
        # set_shadow_overlay() auf (denselben globalen Toggle wie die Shell-
        # Footer-Checkbox, siehe MapDisplay3D.set_shadow_overlay()), nicht
        # mehr set_layer_visibility() - Schatten gelten jetzt tab-übergreifend
        # über die gemeinsame _render_terrain_base(), nicht mehr nur lokal
        # für den Terrain-Tab.
        shadows_checkbox = QCheckBox("Shadows")
        shadows_checkbox.setChecked(True)
        shadows_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_shadow_overlay(checked)
        )
        self.control_widgets["terrain_shadows"] = shadows_checkbox
        self.control_layout.insertWidget(1, shadows_checkbox)

    def _setup_geology_controls(self):
        """
        Funktionsweise: Geology-Tab hat keine eigenen 3D-Basis-Controls mehr -
        welcher Layer (Rock Types/Hardness) auf dem Mesh liegt, entscheiden
        ausschließlich die radio buttons über dem Canvas (siehe
        BaseMapTab._push_data_to_current_display()/_LAYER_SELECTION_KEYS_3D).
        Die früheren "Rock Types"/"Hardness"-Checkboxen hier waren dazu
        redundant (Mehrfachauswahl möglich, wo eigentlich immer nur einer der
        beiden Layer gleichzeitig Sinn ergibt) - entfernt.
        """
        self._clear_tab_controls()

    def _setup_weather_controls(self):
        """
        Funktionsweise: Weather-Tab hat keine eigenen 3D-Basis-Controls mehr -
        welcher Layer (Precipitation/Temperature/Wind/Humidity) auf dem Mesh
        liegt, entscheiden ausschließlich die radio buttons über dem Canvas
        (siehe BaseMapTab._push_data_to_current_display()/
        _LAYER_SELECTION_KEYS_3D). Die früheren 4 Checkboxen hier waren dazu
        redundant - entfernt.
        """
        self._clear_tab_controls()

    def _setup_water_controls(self):
        """
        Funktionsweise: Water-Tab hat keine eigenen 3D-Basis-Controls mehr -
        welcher Layer (Water/Soil Moisture/Erosion/Sedimentation/Flow) auf
        dem Mesh liegt, entscheiden ausschließlich die radio buttons über
        dem Canvas (siehe BaseMapTab._push_data_to_current_display()/
        _LAYER_SELECTION_KEYS_3D). Die früheren 5 Checkboxen hier waren dazu
        redundant (User-Report: mehrere gleichzeitig aktivierbar, z.B. Wasser-
        UND Fluss-Overlay gleichzeitig, unabhängig von der 2D-Auswahl) -
        entfernt.
        """
        self._clear_tab_controls()

    def _setup_biome_controls(self):
        """
        Funktionsweise: Biome-Tab hat keine eigenen 3D-Basis-Controls mehr -
        welcher Layer (Biome Map/Super Biomes) auf dem Mesh liegt,
        entscheiden ausschließlich die radio buttons über dem Canvas (siehe
        BaseMapTab._push_data_to_current_display()/_LAYER_SELECTION_KEYS_3D).
        Die früheren 2 Checkboxen hier waren dazu redundant - entfernt.
        """
        self._clear_tab_controls()

    def _setup_settlement_controls(self):
        """
        Funktionsweise: Erstellt Controls für Settlement-Tab
        Aufgabe: Plots, Settlements, Landmarks, Roads und Civ-Map Controls
        """
        self._clear_tab_controls()

        settlement_layers = [
            ("Plots", "plots", True),
            ("Settlements", "settlements", True),
            ("Landmarks", "landmarks", True),
            ("Roads", "roads", True),
            ("Civ Map", "civ_map", False)
        ]

        for i, (label, layer_name, default_checked) in enumerate(settlement_layers):
            checkbox = QCheckBox(label)
            checkbox.setChecked(default_checked)
            checkbox.toggled.connect(
                lambda checked, layer=layer_name: self.display_3d.set_layer_visibility("settlement", layer, checked)
            )
            self.control_widgets[f"settlement_{layer_name}"] = checkbox
            self.control_layout.insertWidget(i, checkbox)

    def _clear_tab_controls(self):
        """
        Funktionsweise: Entfernt alle Tab-spezifischen Controls
        Aufgabe: Cleanup vor Erstellung neuer Tab-Controls
        """
        for widget in self.control_widgets.values():
            widget.setParent(None)
            widget.deleteLater()
        self.control_widgets.clear()

    def _connect_signals(self):
        """
        Funktionsweise: Verbindet UI-Controls mit 3D-Display-Funktionen
        Aufgabe: Signal-Routing zwischen Controls und 3D-Rendering
        """
        self.reset_camera_button.clicked.connect(self.display_3d.reset_camera)

    def set_tab_type(self, tab_type):
        """
        Funktionsweise: Wechselt Tab-Typ und aktualisiert Controls
        Aufgabe: Dynamische Anpassung der UI-Controls basierend auf aktivem Tab
        Parameter: tab_type (str) - Neuer Tab-Typ
        """
        if tab_type == self.current_tab:
            return

        self.current_tab = tab_type

        # Tab-spezifische Controls erstellen
        if tab_type == "terrain":
            self._setup_terrain_controls()
        elif tab_type == "geology":
            self._setup_geology_controls()
        elif tab_type == "weather":
            self._setup_weather_controls()
        elif tab_type == "water":
            self._setup_water_controls()
        elif tab_type == "biome":
            self._setup_biome_controls()
        elif tab_type == "settlement":
            self._setup_settlement_controls()

    # Interface-Methoden für externe Updates
    def update_heightmap(self, heightmap, tab_type="terrain"):
        """
        Funktionsweise: Delegiert Heightmap-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Heightmap-Updates
        """
        self.set_tab_type(tab_type)
        self.display_3d.update_heightmap(heightmap, tab_type)

    def update_shademap(self, shademap):
        """
        Funktionsweise: Delegiert Shademap-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Shadow-Updates
        """
        self.display_3d.update_shademap(shademap)

    def update_overlay_data(self, tab_type, layer_name, data):
        """
        Funktionsweise: Delegiert Overlay-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Overlay-Updates
        """
        self.display_3d.update_overlay_data(tab_type, layer_name, data)

    def set_layer_visibility(self, tab_type, layer_name, visible):
        """
        Funktionsweise: Delegiert Layer-Sichtbarkeit an 3D-Display.
        Aufgabe: Interface-Methode für BaseMapTab._push_data_to_current_display() -
        fehlte hier bisher komplett (nur MapDisplay3D selbst hatte diese Methode),
        weshalb der hasattr()-Guard in base_tab.py auf self.map_display_3d.display
        (= diese Wrapper-Instanz, NICHT self.display_3d direkt) lautlos fehlschlug
        und die radio-button-Layer-Auswahl im 3D-Modus wirkungslos blieb.
        """
        self.display_3d.set_layer_visibility(tab_type, layer_name, visible)

    def set_shadow_overlay(self, checked: bool, angle: int = None):
        """
        Funktionsweise: Delegiert Schatten-Toggle an 3D-Display.
        Aufgabe: Interface-Methode für BaseMapTab.set_shadow_overlay() - gleicher
        fehlender Delegations-Grund wie set_layer_visibility() oben.
        """
        self.display_3d.set_shadow_overlay(checked, angle)

    def set_contour_overlay(self, checked: bool):
        """
        Funktionsweise: Delegiert Contour-Lines-Toggle an 3D-Display.
        Aufgabe: Interface-Methode für BaseMapTab.set_contour_overlay() - gleicher
        fehlender Delegations-Grund wie set_layer_visibility() oben.
        """
        self.display_3d.set_contour_overlay(checked)

    def set_water_biomes_reference(self, water_biomes_map):
        """
        Funktionsweise: Delegiert See/Fluss-Referenzdaten an 3D-Display.
        Aufgabe: Interface-Methode für water_tab.py - gleicher fehlender
        Delegations-Grund wie set_layer_visibility() oben.
        """
        self.display_3d.set_water_biomes_reference(water_biomes_map)
