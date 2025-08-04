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
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSlider, QLabel, QOpenGLWidget
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
from OpenGL.GL import shaders
import math
from gui.config.gui_default import CanvasSettings


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

    def __init__(self, parent=None):
        """
        Funktionsweise: Initialisiert 3D OpenGL-Widget mit Standard-Camera-Position
        Aufgabe: Setup von OpenGL-Context und Standard-Rendering-Parameter
        """
        super().__init__(parent)

        # Camera-Parameter - Fixed-Axis (60° Elevation, Azimuth-Rotation)
        self.camera_distance = CanvasSettings.CANVAS_3D["camera_distance"]
        self.camera_elevation = -60.0  # Feste Elevation
        self.camera_azimuth = 0.0  # Rotation um Z-Achse
        self.fov = CanvasSettings.CANVAS_3D["fov"]

        # Rendering-Daten
        self.heightmap = None
        self.shademap = None  # 64x64 Shadow-Map
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
            "terrain": {"base": True, "shadows": True},
            "geology": {"rock_map": True, "hardness_map": False},
            "weather": {"precipitation": True, "temperature": False, "wind": False, "humidity": False},
            "water": {"water_map": True, "soil_moisture": False, "erosion": False, "sedimentation": False},
            "biome": {"biome_map": True},
            "settlement": {"plots": True, "settlements": True, "landmarks": True, "roads": True, "civ_map": False}
        }

        # Overlay-Daten für verschiedene Tabs
        self.overlay_data = {
            "geology": {"rock_map": None, "hardness_map": None},
            "weather": {"precipitation": None, "temperature": None, "wind": None, "humidity": None},
            "water": {"water_map": None, "soil_moisture": None, "erosion": None, "sedimentation": None},
            "biome": {"biome_map": None},
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
        self._generate_terrain_mesh()
        self.update()

    def update_shademap(self, shademap):
        """
        Funktionsweise: Aktualisiert Shademap für Terrain-Schattierung
        Aufgabe: Setzt neue 64x64 Shadow-Daten für Upscaling
        Parameter: shademap (numpy.ndarray) - 64x64 Shadow-Map
        """
        if shademap is not None and shademap.shape != (64, 64):
            self.rendering_error.emit("Shademap must be 64x64 resolution")
            return

        self.shademap = shademap
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

        height_range = self.heightmap.max() - self.heightmap.min()
        if height_range > 0:
            self.terrain_height_scale = (max_dimension * 0.3) / height_range
        else:
            self.terrain_height_scale = 1.0

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
        Aufgabe: Erstellt Vertices, Normals und Indices für Terrain-Rendering
        """
        if self.heightmap is None:
            return

        # Alte Buffer löschen
        self._cleanup_mesh_buffers()

        height, width = self.heightmap.shape
        vertices = []
        indices = []

        # Vertices und Normals generieren
        for y in range(height):
            for x in range(width):
                # Position
                pos_x = (x / (width - 1) - 0.5) * width * self.terrain_scale_factor
                pos_y = self.heightmap[y, x] * self.terrain_height_scale
                pos_z = (y / (height - 1) - 0.5) * height * self.terrain_scale_factor

                # Normal berechnen
                normal_x, normal_y, normal_z = self._calculate_vertex_normal(x, y, width, height)

                # Texture-Coordinates
                tex_u = x / (width - 1)
                tex_v = y / (height - 1)

                vertices.extend([pos_x, pos_y, pos_z, normal_x, normal_y, normal_z, tex_u, tex_v])

        # Indices für Triangles generieren
        for y in range(height - 1):
            for x in range(width - 1):
                # Zwei Triangles pro Quad
                top_left = y * width + x
                top_right = y * width + (x + 1)
                bottom_left = (y + 1) * width + x
                bottom_right = (y + 1) * width + (x + 1)

                # Triangle 1
                indices.extend([top_left, bottom_left, top_right])
                # Triangle 2
                indices.extend([top_right, bottom_left, bottom_right])

        # Mesh-Daten speichern
        self.mesh_vertices = np.array(vertices, dtype=np.float32)
        self.mesh_indices = np.array(indices, dtype=np.uint32)

        # OpenGL-Buffers erstellen
        self._create_mesh_buffers()

    def _calculate_vertex_normal(self, x, y, width, height):
        """
        Funktionsweise: Berechnet Vertex-Normal für Lighting-Berechnungen
        Aufgabe: Erstellt Oberflächennormalen basierend auf benachbarten Heightmap-Werten
        Parameter: x, y (int), width, height (int) - Vertex-Position und Heightmap-Dimensionen
        Rückgabe: tuple - (normal_x, normal_y, normal_z)
        """
        # Gradient in X-Richtung
        if x > 0 and x < width - 1:
            dx = (self.heightmap[y, x + 1] - self.heightmap[y, x - 1]) * self.terrain_height_scale
        elif x == 0:
            dx = (self.heightmap[y, x + 1] - self.heightmap[y, x]) * self.terrain_height_scale
        else:
            dx = (self.heightmap[y, x] - self.heightmap[y, x - 1]) * self.terrain_height_scale

        # Gradient in Y-Richtung
        if y > 0 and y < height - 1:
            dy = (self.heightmap[y + 1, x] - self.heightmap[y - 1, x]) * self.terrain_height_scale
        elif y == 0:
            dy = (self.heightmap[y + 1, x] - self.heightmap[y, x]) * self.terrain_height_scale
        else:
            dy = (self.heightmap[y, x] - self.heightmap[y - 1, x]) * self.terrain_height_scale

        # Cross-Product für Normal
        step_size = self.terrain_scale_factor
        normal = np.cross([-step_size, 0, dy], [0, step_size, dx])

        # Normalisieren
        length = np.linalg.norm(normal)
        if length > 0:
            normal = normal / length
        else:
            normal = np.array([0, 1, 0])  # Default nach oben

        return normal[0], normal[1], normal[2]

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

        # Camera-Position berechnen
        eye_x = self.camera_distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
        eye_y = self.camera_distance * math.sin(elevation_rad)
        eye_z = self.camera_distance * math.cos(elevation_rad) * math.cos(azimuth_rad)

        eye = [eye_x, eye_y, eye_z]
        target = [0, 0, 0]  # Blick zum Zentrum
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
        """
        if not self.shader_program:
            return

        # Projection-Matrix
        if self.projection_matrix is not None:
            proj_location = gl.glGetUniformLocation(self.shader_program, "projection")
            if proj_location >= 0:
                gl.glUniformMatrix4fv(proj_location, 1, gl.GL_FALSE, self.projection_matrix.flatten())

        # View-Matrix
        if self.view_matrix is not None:
            view_location = gl.glGetUniformLocation(self.shader_program, "view")
            if view_location >= 0:
                gl.glUniformMatrix4fv(view_location, 1, gl.GL_FALSE, self.view_matrix.flatten())

        # Model-Matrix
        if self.model_matrix is not None:
            model_location = gl.glGetUniformLocation(self.shader_program, "model")
            if model_location >= 0:
                gl.glUniformMatrix4fv(model_location, 1, gl.GL_FALSE, self.model_matrix.flatten())

    def _render_terrain_tab(self):
        """
        Funktionsweise: Rendert Terrain-Tab mit Heightmap und Shademap
        Aufgabe: Basis-Terrain mit 2D-Map Coloring und Shadow-Integration
        """
        if not self.layer_visibility["terrain"]["base"]:
            return

        self._render_terrain_base()

        if self.layer_visibility["terrain"]["shadows"] and self.shademap is not None:
            self._render_shadows()

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

    def _render_water_tab(self):
        """
        Funktionsweise: Rendert Water-Tab mit Hydrologie-Daten
        Aufgabe: Terrain mit Water, Soil-Moisture, Erosion und Sedimentation-Overlays
        """
        self._render_terrain_base()

        water_layers = ["water_map", "soil_moisture", "erosion", "sedimentation"]
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

        elif self.shader_fallback_active:
            # Fallback-Shader Parameter
            wireframe_color_location = gl.glGetUniformLocation(self.shader_program, "wireframeColor")
            if wireframe_color_location >= 0:
                gl.glUniform3f(wireframe_color_location, 0.7, 0.7, 0.7)  # Grau

        # VAO binden und rendern
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.mesh_indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _render_shadows(self):
        """
        Funktionsweise: Rendert Shademap-Integration
        Aufgabe: Upscaling und Anwendung der 64x64 Shademap auf Terrain
        """
        if self.shademap is None or not self.shader_program:
            return

        # Shademap auf Heightmap-Auflösung hochskalieren
        upscaled_shadows = _upscale_shademap(self.shademap, self.heightmap.shape)

        if upscaled_shadows is None:
            return

        # Shadow-Textur erstellen und binden
        # TODO: Implementierung der Textur-Upload und Shader-Integration
        use_shadows_location = gl.glGetUniformLocation(self.shader_program, "useShadows")
        if use_shadows_location >= 0:
            gl.glUniform1i(use_shadows_location, 1)

    def _render_overlay(self, tab_type, layer_name):
        """
        Funktionsweise: Rendert Tab-spezifische Overlay-Layer
        Aufgabe: Anwendung von Overlay-Daten auf Terrain-Oberfläche
        Parameter: tab_type (str), layer_name (str) - Tab und Layer-Identifikation
        """
        overlay_data = self.overlay_data[tab_type][layer_name]
        if overlay_data is None:
            return

        # TODO: Implementierung overlay-spezifischer Rendering-Modi
        # - Texture-Blending für verschiedene Overlay-Typen
        # - Farbschema-Anwendung basierend auf Layer-Typ
        # - Multi-Channel Rendering für kombinierte Overlays

    def _render_plot_boundaries(self):
        """
        Funktionsweise: Rendert Plot-Grenzen als Wireframe-Mesh
        Aufgabe: Visualisierung von Settlement-Plot-Geometrie
        """
        plot_data = self.overlay_data["settlement"]["plots"]
        if plot_data is None:
            return

        # TODO: Implementierung von Plot-Boundary-Rendering
        # - Wireframe-Mesh auf Terrain projiziert
        # - Plot-Node-Marker an Eckpunkten

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

        if event.buttons() & Qt.LeftButton:
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
        self.camera_elevation = -60.0
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
        Aufgabe: Basis-Terrain und Shadows-Controls
        """
        self._clear_tab_controls()

        # Terrain Base
        terrain_checkbox = QCheckBox("Terrain")
        terrain_checkbox.setChecked(True)
        terrain_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("terrain", "base", checked)
        )
        self.control_widgets["terrain_base"] = terrain_checkbox
        self.control_layout.insertWidget(0, terrain_checkbox)

        # Shadows
        shadows_checkbox = QCheckBox("Shadows")
        shadows_checkbox.setChecked(True)
        shadows_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("terrain", "shadows", checked)
        )
        self.control_widgets["terrain_shadows"] = shadows_checkbox
        self.control_layout.insertWidget(1, shadows_checkbox)

    def _setup_geology_controls(self):
        """
        Funktionsweise: Erstellt Controls für Geology-Tab
        Aufgabe: Rock-Map und Hardness-Map Controls
        """
        self._clear_tab_controls()

        # Rock Map
        rock_checkbox = QCheckBox("Rock Types")
        rock_checkbox.setChecked(True)
        rock_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("geology", "rock_map", checked)
        )
        self.control_widgets["geology_rock"] = rock_checkbox
        self.control_layout.insertWidget(0, rock_checkbox)

        # Hardness Map
        hardness_checkbox = QCheckBox("Hardness")
        hardness_checkbox.setChecked(False)
        hardness_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("geology", "hardness_map", checked)
        )
        self.control_widgets["geology_hardness"] = hardness_checkbox
        self.control_layout.insertWidget(1, hardness_checkbox)

    def _setup_weather_controls(self):
        """
        Funktionsweise: Erstellt Controls für Weather-Tab
        Aufgabe: Precipitation, Temperature, Wind und Humidity-Controls
        """
        self._clear_tab_controls()

        weather_layers = [
            ("Precipitation", "precipitation", True),
            ("Temperature", "temperature", False),
            ("Wind", "wind", False),
            ("Humidity", "humidity", False)
        ]

        for i, (label, layer_name, default_checked) in enumerate(weather_layers):
            checkbox = QCheckBox(label)
            checkbox.setChecked(default_checked)
            checkbox.toggled.connect(
                lambda checked, layer=layer_name: self.display_3d.set_layer_visibility("weather", layer, checked)
            )
            self.control_widgets[f"weather_{layer_name}"] = checkbox
            self.control_layout.insertWidget(i, checkbox)

    def _setup_water_controls(self):
        """
        Funktionsweise: Erstellt Controls für Water-Tab
        Aufgabe: Water, Soil-Moisture, Erosion und Sedimentation-Controls
        """
        self._clear_tab_controls()

        water_layers = [
            ("Water", "water_map", True),
            ("Soil Moisture", "soil_moisture", False),
            ("Erosion", "erosion", False),
            ("Sedimentation", "sedimentation", False)
        ]

        for i, (label, layer_name, default_checked) in enumerate(water_layers):
            checkbox = QCheckBox(label)
            checkbox.setChecked(default_checked)
            checkbox.toggled.connect(
                lambda checked, layer=layer_name: self.display_3d.set_layer_visibility("water", layer, checked)
            )
            self.control_widgets[f"water_{layer_name}"] = checkbox
            self.control_layout.insertWidget(i, checkbox)

    def _setup_biome_controls(self):
        """
        Funktionsweise: Erstellt Controls für Biome-Tab
        Aufgabe: Biome-Map Control
        """
        self._clear_tab_controls()

        # Biome Map
        biome_checkbox = QCheckBox("Biome Map")
        biome_checkbox.setChecked(True)
        biome_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("biome", "biome_map", checked)
        )
        self.control_widgets["biome_map"] = biome_checkbox
        self.control_layout.insertWidget(0, biome_checkbox)

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