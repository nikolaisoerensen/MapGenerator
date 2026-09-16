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
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSlider, QLabel, \
    QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
from OpenGL.GL import shaders
import math
import time
from gui.config.gui_default import CanvasSettings, ColorSchemes
from gui.config.value_default import TERRAIN
from gui.widgets.map_display_2d import _calculate_contour_levels, _get_layer_range, compute_slope_compass_rgb
from gui.widgets.adaptive_terrain_mesh import build_adaptive_mesh, ist_fuer_adaptives_mesh_geeignet

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
    "thermal_erosion": "thermal_erosion_map",
    "thermal_deposition": "thermal_deposition_map",
    # Erosion-Tab (2026-07-28) - siehe gui/tabs/erosion_tab.py.
    "net_change": "net_change_map", "sediment_load": "sediment_load_map",
    "water_depth": "water_depth_map", "flow_velocity": "flow_velocity_map",
    "evaporation": "evaporation_map",
    "flow_map": "flow_map", "hardness_map": "hardness_map", "slope": "slopemap",
    "civ_map": "civ_map",
    "terrain_hub_delta": "terrain_hub_delta", "tilt_delta": "tilt_delta",
    "fold_delta": "fold_delta", "fault_delta": "fault_delta",
    "intrusion_delta": "intrusion_delta",
    # Die vier neuen Terrain-Karten (2026-08-26/27). OHNE Eintrag hier
    # findet `_colorize_layer()` keinen `range_key`, faellt still auf
    # Auto-Skalierung ohne Farbtafel zurueck - und 3D zeigt ANDERE Farben
    # als 2D, ohne Fehlermeldung. Genau der Rueckfall, den
    # smoke_test_layer_2d_3d_parity.py bewacht; er hat es gefunden,
    # nachdem die Eintraege in layer_ranges (gui_default.py) schon standen
    # und die Sache dadurch erledigt aussah.
    "river_water": "river_water", "river_order": "river_order",
    "hinterland_height": "hinterland_height",
    "voronoi_map": "voronoi_map",
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

    if layer_name == "slope" and data.ndim == 3 and data.shape[2] == 2:
        # Kompass-Farbrad (Hangausrichtung=Hue, Steilheit=Saettigung) -
        # identische Formel wie map_display_2d.py's _render_slopemap(),
        # Nutzer-Vorgabe "gleiche Farben fuer 2D und 3D".
        rgb_float = compute_slope_compass_rgb(data[:, :, 0], data[:, :, 1])
        return (np.clip(rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)

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
    elif cmap_name is not None:
        # Colormap aus CanvasSettings bekannt (z.B. "RdBu_r" für die
        # signierten Geology-Delta-Diagnose-Layer), aber kein fester Bereich
        # definiert (vmin=vmax=None) - symmetrische Max-Abs-Normierung um 0
        # statt reinem Daten-Min/Max, damit "kein Effekt" (0) bei einer
        # divergierenden Farbskala in der Mitte (weiß bei RdBu_r) liegt statt
        # zufällig irgendwo im Bereich zu landen.
        max_abs = max(float(np.abs(data).max()), 1e-9)
        norm = np.clip((data.astype(np.float64) + max_abs) / (2.0 * max_abs), 0.0, 1.0)
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
    # Anklicken von Orten/Wegen (docs/OFFENE_PUNKTE.md 6.29). Traegt das
    # Treffer-dict aus karten_auswahl.treffer_suchen() bzw. None beim Klick
    # ins Leere - der Reiter entscheidet, was er damit anzeigt.
    objekt_gewaehlt = pyqtSignal(object)

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
        # 180° statt 0°: bei azimuth=0 steht die Kamera auf der Nord-Seite
        # (+Z, siehe _generate_terrain_mesh()s pos_z-Formel: Zeile height-1 =
        # Norden) und blickt nach Süden - der Betrachter sähe damit
        # bevorzugt Nordhänge, UMGEKEHRT zur 2D-Darstellung (origin='lower',
        # Norden oben im Bild - ein Nutzer, der eine Nordup-Karte von Süden
        # her betrachtet). Nutzer-Beobachtung: 3D-Kamera bei Terrain/Geology
        # "auf Norden eingestellt", sollte "180° gedreht" sein - Kamera daher
        # auf die Süd-Seite (-Z), Blickrichtung Norden, exakt wie beim
        # gedachten Betrachter der 2D-Karte.
        self.camera_azimuth = 180.0  # Rotation um Z-Achse
        self.fov = CanvasSettings.CANVAS_3D["fov"]

        # BLICKPUNKT als eigener Zustand (2026-07-28). Vorher blickte
        # _update_view_matrix() fest auf [0, terrain_center_y, 0], die Kamera
        # konnte sich also nur um die Kartenmitte drehen und war an sie
        # gefesselt. Sobald der Blickpunkt beweglich ist, ergeben sich Panning
        # (Blickpunkt verschieben) und Fliegen (Blickpunkt UND Auge
        # verschieben) aus derselben Groesse.
        #
        # y wird in _calculate_auto_scaling() auf die Gelaendemitte gesetzt,
        # sobald eine Heightmap vorliegt.
        self.camera_target = [0.0, 0.0, 0.0]
        # Ob der Nutzer den Blickpunkt schon selbst bewegt hat. Siehe
        # _calculate_auto_scaling(): eine neue Heightmap darf die Kamera nur
        # dann nachzentrieren, wenn sie noch unberuehrt ist.
        self._camera_target_touched = False

        # Flugmodus: gedrueckte Tasten, ausgewertet von einem Timer statt
        # direkt im keyPressEvent. Sonst haengt die Fluggeschwindigkeit an der
        # Tastenwiederholrate des Betriebssystems, und mehrere gleichzeitig
        # gedrueckte Tasten (vorwaerts + aufsteigen) funktionieren nicht.
        self._pressed_keys = set()
        self.flight_timer = QTimer()
        self.flight_timer.timeout.connect(self._advance_flight)

        # Meter pro Sekunde in Render-Einheiten. 10 Einheiten = Kantenlaenge
        # der Karte, ein Flug quer ueber die Karte dauert damit rund 3 s.
        self.flight_speed = 3.5

        # Tastatur-Ereignisse erreichen ein Widget nur mit Fokus-Politik.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Rendering-Daten
        self.heightmap = None
        self.shademap = None  # Shadow-Map (beliebige Auflösung, wird auf Heightmap-Größe hochskaliert)
        # Licht-Position (Weltkoordinaten, +X=Ost/+Z=Norden/+Y=oben - siehe
        # _generate_terrain_mesh()s pos_x/pos_z-Formeln). Default aus
        # gui_default.py (statischer "Sonne im Süden"-Fallback), wird über
        # set_sun_direction() durch die tatsächlich berechnete Sonnenposition
        # (Breitengrad/Jahreszeit) ersetzt, sobald diese verfügbar ist - siehe
        # base_tab.py._push_data_to_current_display().
        self._light_pos = tuple(CanvasSettings.CANVAS_3D["light_position"])
        # Schatten sind dauerhaft aktiv (die frühere "Shadows"-Checkbox - sowohl
        # der globale Shell-Footer-Toggle als auch die lokale Terrain-3D-Tab-
        # Checkbox - wurde auf Nutzer-Wunsch entfernt, UI-Aufräumung Teil 2).
        # _render_terrain_base() (das Basis-Mesh, von jedem Tab-Render-Pfad
        # aufgerufen) wendet den Schatten-Multiplikator weiterhin an.
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
        # Live-Wert von Terrains "Map Distance"-Slider (siehe
        # DataLODManager.get_map_distance_km(), [[project-terrain-review]] 4f) -
        # dieses Widget hat keinen eigenen data_lod_manager-Zugriff (reines
        # Push-Rendering-Widget), daher über set_world_size_km() aktuell
        # gehalten, statt der vorherigen statischen Klassenkonstante
        # WORLD_SIZE_KM direkt zu lesen. Fallback-Default bleibt die
        # Konstante, bis der erste Push passiert ist.
        self.world_size_km = self.WORLD_SIZE_KM
        # Separates Mini-Shader-Programm für die Wind-Vektor-Pfeile (Weather-Tab,
        # Layer "wind") - der Haupt-Terrain-Shader erwartet Normal/TexCoord/LightPos-
        # Varyings, die für simple farbige GL_LINES nicht gebraucht werden, siehe
        # shaders/3d_display/wind_vector.vert/.frag. Vorher zeigte "wind" in 3D nur
        # eine schwache Magnitude-Heatmap ohne Richtung ("nichts zu erkennen").
        self.wind_shader_program = None

        # Eigenes Mini-Shader-Programm fuer die Wegbaender (6.28,
        # Nutzerfeedback 2026-08-16: das bisherige wind_shader_program ist
        # komplett unlit - die Baender sahen dadurch "fake" aus und passten
        # nicht zur Terrain-Beleuchtung). Braucht Normalen (fuer echtes
        # Licht) und eine Auswahl-Einfaerbung, die wind_vector.vert/.frag
        # nicht kennen - siehe shaders/3d_display/wegband.vert/.frag.
        self.wegband_shader_program = None
        self.current_tab = "terrain"  # Aktueller Tab-Typ

        # Mesh-Daten
        self.mesh_vertices = None
        self.mesh_indices = None
        self.vertex_buffer = None
        self.index_buffer = None

        # OVERLAY-TEXTUR-ZWISCHENSPEICHER (2026-08-13, Nutzerbefund "Kuestentyp
        # ruckelt im 3D, Slope auch noch"). Gemessen: `_colorize_layer()` fuer
        # Slope und `rasterize_kuesten_archetypen_rgba()` brauchen bei 1024px
        # je rund 0.6s - VOR diesem Fix wurde das bei JEDEM paintGL()-Aufruf
        # neu gerechnet, also bei jeder Mausbewegung waehrend des Drehens.
        # Schluessel (tab_type, layer_name) -> (Objekt-ID der Quelldaten,
        # Textur-ID). Objekt-Identitaet genuegt hier als Cache-Schluessel
        # (anders als beim Heightmap-Vergleich in update_heightmap(), wo
        # get_terrain_data_combined() bei JEDEM Aufruf eine Kopie liefert) -
        # overlay_data[tab_type][layer_name] wird nur bei einem echten Push
        # (update_overlay_data()) durch ein NEUES Objekt ersetzt, zwischen
        # zwei Frames waehrend des Kamera-Drehens bleibt es dasselbe Array.
        self._overlay_texture_cache = {}
        # Zwischenspeicher der Wegband-Geometrie (6.28) - sie je Frame neu
        # zu bauen waere derselbe Fehler wie bei den Overlays (6.21).
        self._wegband_cache = {}
        # Auswaehlbare Objekte in WELTkoordinaten, gefuellt von
        # setze_auswahlobjekte() (6.29). Getrennt von overlay_data, weil hier
        # die fertigen Weltpositionen stehen und nicht die Rohdaten.
        self._auswahl_orte = np.zeros((0, 3), dtype=np.float64)
        self._auswahl_orte_kennung = []
        self._auswahl_wege = []
        self._auswahl_wege_kennung = []
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
            "terrain": {"base": True, "slope": False,
                        "region_overlay": False, "kuesten_overlay": False,
                        # AUS als Vorgabe. Der Fluss-Reiter schaltet es
                        # ein, wenn sein Modus "Flussnetz" ist, und wieder
                        # AUS, sobald der Nutzer auf "Gelaende" oder
                        # "Ordnung" wechselt (Nutzerbefund 2026-08-24:
                        # *"wenn man auf Ordnung geht dann aendert sich
                        # nichts"* - das Netz lag ueber allem).
                        "river_overlay": False,
                        # Skalarkarten des Flussreiters (2026-08-26).
                        "river_water": False, "river_order": False,
                        # Hoehenfaktor-Ansicht (2026-08-26).
                        "hinterland_height": False, "voronoi_map": False},
            "geology": {"rock_map": True, "hardness_map": False,
                        "terrain_hub_delta": False, "tilt_delta": False, "fold_delta": False,
                        "fault_delta": False, "intrusion_delta": False},
            "erosion": {"erosion": True, "sedimentation": False, "net_change": False, "sediment_load": False,
                       "water_depth": False, "flow_velocity": False, "thermal_erosion": False,
                       "thermal_deposition": False},
            "weather": {"precipitation": True, "temperature": False, "wind": False, "humidity": False},
            "water": {"water_map": True, "soil_moisture": False, "erosion": False, "sedimentation": False,
                      "flow_map": False,
                      # 2026-08-26 nachgetragen: stand in beiden
                      # Registern des Reiters, fehlte aber hier - die
                      # Verdunstungskarte blieb im 3D unsichtbar.
                      "evaporation": False},
            "biome": {"biome_map": True, "super_biome_mask": False},
            "settlement": {"plots": True, "settlements": True, "landmarks": True, "roads": True, "civ_map": False,
                           # "uebersicht": globale Siedlungsuebersicht als RGBA-Skin
                           # (Staedte/Landmarken/Roadsites als Punkte, Land-/Seewege als
                           # Linien) - 2026-08-13, Nutzer-Vorgabe "3D Settlements global
                           # sollte jetzt umgesetzt werden". Ersetzt funktional die drei
                           # nie implementierten Marker-Layer darueber (siehe
                           # _render_settlement_markers(), ein leerer TODO-Stub).
                           "uebersicht": False,
                           # Wege als echte Bandgeometrie (6.28)
                           "wegbaender": False}
        }

        # Overlay-Daten für verschiedene Tabs
        self.overlay_data = {
            # `river_overlay` liegt unter "terrain", NICHT unter einem
            # eigenen "river"-Eintrag: `gui/tabs/river_tab.py` setzt
            # `generator_type = "terrain"` und meldet sich im 3D genau so
            # an. Ein eigener Tab-Typ waere nie erreicht worden.
            "terrain": {"slope": None, "region_overlay": None,
                        "kuesten_overlay": None, "river_overlay": None,
                        # Skalarkarten des Flussreiters (2026-08-26) - er
                        # meldet sich als "terrain" an, siehe base_tab.
                        "river_water": None, "river_order": None,
                        "hinterland_height": None, "voronoi_map": None},
            "geology": {"rock_map": None, "hardness_map": None,
                        "terrain_hub_delta": None, "tilt_delta": None, "fold_delta": None,
                        "fault_delta": None, "intrusion_delta": None},
            "erosion": {"erosion": None, "sedimentation": None, "net_change": None, "sediment_load": None,
                       "water_depth": None, "flow_velocity": None, "thermal_erosion": None,
                       "thermal_deposition": None},
            "weather": {"precipitation": None, "temperature": None, "wind": None, "humidity": None},
            "water": {"water_map": None, "soil_moisture": None, "erosion": None, "sedimentation": None,
                      "flow_map": None,
                      # 2026-08-26 nachgetragen, siehe layer_visibility oben.
                      "evaporation": None},
            "biome": {"biome_map": None, "super_biome_mask": None},
            "settlement": {"plots": None, "settlements": [], "landmarks": [], "roads": [], "civ_map": None,
                           "uebersicht": None, "wegbaender": None}
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

        # Fehlertoleranz (Meter) fuer die adaptive Mesh-Triangulierung -
        # siehe gui/widgets/adaptive_terrain_mesh.py. Nur bei quadratischen
        # Heightmaps mit Kantenlaenge 2^n+1 aktiv, sonst automatischer
        # Rueckfall auf das Gleichmaessig-Gitter (siehe _generate_terrain_mesh).
        self._adaptive_mesh_fehler_toleranz_m = 6.0

        # ALTERNATIVER NETZBAUER, normalerweise aus.
        #
        # Ein Aufrufer kann hier eine Funktion
        # `(heightmap, scale_factor, height_scale) -> (vertices, indices, stats)`
        # hinterlegen; sie hat dann Vorrang vor dem Quadtree. Gedacht fuer
        # `tools/mesh_werkstatt.py`, wo die Netzarten live verglichen werden -
        # und spaeter fuer eine Umschaltung in der App selbst.
        #
        # Voreinstellung None heisst: exakt das bisherige Verhalten. Liefert
        # der Bauer None, wird das LAUT gemeldet und auf das Quadtree
        # zurueckgefallen - ein stiller Rueckfall waere von Erfolg nicht zu
        # unterscheiden (siehe CLAUDE.md, dieselbe Falle wie bei den
        # GPU-Fallbacks und der 2^n+1-Bedingung).
        self._mesh_bauer = None

        # Index des angeklickten Weges in der Liste aus setze_auswahlobjekte()
        # (erst Landwege, dann Seewege - dieselbe Reihenfolge wie im
        # Zeichenpuffer). None = nichts gewaehlt.
        self._ausgewaehlter_weg = None

    def setze_mesh_bauer(self, bauer):
        """
        Netzbauer setzen (oder mit None zuruecksetzen) und Mesh neu bauen.
        """
        self._mesh_bauer = bauer
        if self.heightmap is not None:
            self.makeCurrent()
            try:
                # legt die GL-Buffer am Ende selbst an
                self._generate_terrain_mesh()
            finally:
                self.doneCurrent()
            self.update()

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
            # Kompensiert die Vorzeichen-Spiegelung der X-Zeile in
            # _update_view_matrix() (Ost-West-Spiegelungs-Fix, siehe dortiger
            # Kommentar) - eine Spiegelung im View-Space kehrt zwangsläufig
            # die Dreiecks-Wickelrichtung um, ohne dies würde Backface-Culling
            # plötzlich das gesamte sichtbare Terrain wegculled statt der
            # tatsächlichen Rückseiten.
            gl.glFrontFace(gl.GL_CW)
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
            self._compile_wegband_shader()

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
        elif self.current_tab == "erosion":
            self._render_erosion_tab()
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

        # OHNE SHADERPROGRAMM WIRD NICHT GEZEICHNET.
        #
        # Vorher lief es hier mit `return True` weiter und rief anschliessend
        # glDrawElements ohne aktives Programm - im Core-Profile undefiniert,
        # in der Praxis ein harter Prozessabbruch (0xC0000409). Lieber ein
        # leeres Fenster mit einer klaren Meldung als ein Absturz ohne
        # Traceback.
        if self.shader_program is None:
            if not getattr(self, "_shader_fehlt_gemeldet", False):
                self._shader_fehlt_gemeldet = True
                print("FEHLER: kein Shaderprogramm - es wird nichts gezeichnet. "
                      "Meist bedeutet das, dass die Dateien unter "
                      "shaders/3d_display/ nicht gefunden wurden.")
                self.rendering_error.emit(
                    "Kein Shaderprogramm - shaders/3d_display/ nicht gefunden?")
            return False

        gl.glUseProgram(self.shader_program)
        self._upload_matrices()

        return True

    def _activate_fallback_rendering(self):
        """
        Funktionsweise: Aktiviert Fallback-Rendering ohne Shader
        Aufgabe: Einfaches Wireframe/Point-Rendering wenn Shader fehlschlagen

        DER NAME VERSPRICHT MEHR, ALS DIE FUNKTION HAELT: sie schaltet nur den
        Polygonmodus auf Linien um. Ein Ersatz-Shaderprogramm baut sie nicht.
        Ohne Programm ist `glDrawElements` im Core-Profile aber undefiniert -
        am 2026-08-16 starb der Prozess dadurch hart mit 0xC0000409, ganz ohne
        Python-Traceback (Ursache waren nicht gefundene Shaderdateien, siehe
        `_load_shader_from_file`). Deshalb meldet `_prepare_rendering()` jetzt
        False, statt ohne Programm weiterzuzeichnen.
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

        # base_tab.py._push_data_to_current_display() ruft update_heightmap()
        # bei JEDEM 3D-Layer-Wechsel auf, auch reinen Textur-Wechseln wie
        # Height->Slope (siehe dortiger Kommentar: "Overlay-Layer werden
        # zusaetzlich IMMER an... gepusht") - dabei wird i.d.R. dieselbe,
        # inhaltlich unveraenderte kombinierte Heightmap erneut hereingereicht
        # (get_terrain_data_combined() baut bei jedem Aufruf ein NEUES Array
        # per .copy(), ein reiner Objekt-Identitaets-Vergleich wuerde also
        # immer "geaendert" sagen). Solange das Mesh gleichfoermig war, war
        # ein Rebuild bei jedem Wechsel gratis (rein vektorisiertes Numpy) -
        # seit der adaptiven Triangulierung (siehe adaptive_terrain_mesh.py)
        # kostet ein Rebuild bei 1024px real 2-3s Python-Rechenzeit, was sich
        # beim Umschalten auf einen anderen Skin (z.B. Slope) als spuerbares
        # Ruckeln zeigte (Nutzerbefund 2026-08-12) - der Rebuild war dabei
        # komplett unnoetig, da sich nur die Textur, nicht die Geometrie
        # aendert. Deshalb: Inhaltsvergleich statt Identitaet, Rebuild nur bei
        # tatsaechlicher Aenderung (oder wenn noch gar kein Mesh existiert).
        _t0 = time.time()
        unveraendert = (
            self.heightmap is not None
            and self.heightmap.shape == heightmap.shape
            and np.array_equal(self.heightmap, heightmap)
        )
        print(f"DEBUG: update_heightmap({tab_type}): Vergleich {time.time()-_t0:.3f}s, unveraendert={unveraendert}, mesh vorhanden={self.mesh_vertices is not None}")

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

        if unveraendert and self.mesh_vertices is not None:
            self.update()
            print(f"DEBUG: update_heightmap({tab_type}): Rebuild uebersprungen, gesamt {time.time()-_t0:.3f}s")
            return

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
        print(f"DEBUG: update_heightmap({tab_type}): Rebuild gesamt {time.time()-_t0:.3f}s")

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

    def set_sun_direction(self, elevation_deg: float, azimuth_deg: float, distance: float = 15.0):
        """
        Funktionsweise: Setzt die Licht-Position aus einem echten Sonnenstand
        (Elevation/Azimut, siehe core/terrain_generator.py.calculate_solar_position())
        statt des statischen gui_default.py-Defaults - macht die 3D-Beleuchtung
        breitengrad-/jahreszeitabhängig statt einer festen "Sonne im Süden,
        45°"-Annahme.
        Aufgabe: Weltraum-Konvention (siehe _generate_terrain_mesh()): +X=Ost,
        +Z=Norden, +Y=oben - dieselbe Azimut-Formel wie
        ShadowCalculator._raycast_shadow_cpu() (0°=Norden, 90°=Osten, 180°=
        Süden, 270°=Westen), nur auf die 3D-Weltachsen (X,Z statt der
        2D-Array-Achsen X,Y) übertragen. distance ist rein die Licht-Entfernung
        vom Ursprung (keine physikalische Einheit, nur groß genug für ein
        praktisch richtungsartiges Licht, ähnliche Größenordnung wie der
        bisherige statische Default).
        Parameter: elevation_deg - Sonnenhöhe über dem Horizont (0-90°),
        azimuth_deg - Kompass-Azimut, distance - Licht-Entfernung
        """
        elev_rad = np.radians(elevation_deg)
        azim_rad = np.radians(azimuth_deg)
        self._light_pos = (
            distance * np.cos(elev_rad) * np.sin(azim_rad),  # Ost-Komponente
            distance * np.sin(elev_rad),                      # oben
            distance * np.cos(elev_rad) * np.cos(azim_rad),  # Nord-Komponente
        )
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

    def set_world_size_km(self, world_size_km: float):
        """
        Funktionsweise: Aktualisiert die reale Kartenausdehnung in km, die
        _calculate_terrain_scaling() für die Höhen-Skalierung des Meshs nutzt -
        live von Terrains "Map Distance"-Slider gepusht (siehe
        DataLODManager.get_map_distance_km(), [[project-terrain-review]] 4f),
        da dieses Widget selbst keinen data_lod_manager hat.
        """
        if world_size_km and world_size_km != self.world_size_km:
            self.world_size_km = float(world_size_km)
            self._calculate_terrain_scaling()
            self.update()

    def overlay_river_generations(self, generation_map, zeige_mikro=False):
        """
        Das Flussnetz im 3D - Gegenstueck zur gleichnamigen 2D-Methode.

        WARUM ES SIE BRAUCHTE (Nutzerbefund 2026-08-24): *"dass man im 3D
        modus bei dem Flussnetzwerk keine fluesse sehn kann."*

        `gui/tabs/river_tab.py` ruft diese Methode ueber ein `hasattr` auf.
        In der 2D-Anzeige gab es sie seit dem 2026-08-06, in der 3D-Anzeige
        nie - der Aufruf fiel damit LAUTLOS aus, ohne Fehler und ohne
        Warnung. Genau das Muster, vor dem CLAUDE.md warnt: ein stiller
        Rueckfall ist von Erfolg nicht zu unterscheiden.

        Gezeichnet wird als RGBA-Textur auf dem Gelaende, mit derselben
        Farblogik wie in 2D (`rasterize_fluesse_rgba` in map_display_2d) -
        EINE Funktion fuer beide Ansichten, damit keine zweite Wahrheit
        entsteht.
        """
        if self.heightmap is None:
            return
        if not isinstance(generation_map, np.ndarray) or generation_map.ndim != 2:
            return
        self.set_layer_visibility("terrain", "river_overlay", True)
        self.update_overlay_data("terrain", "river_overlay", {
            "river_generation": generation_map,
            "heightmap": self.heightmap,
            "zeige_mikro": bool(zeige_mikro),
            # Im 3D verschwindet ein einzelnes Pixel auf der schraeg
            # betrachteten Textur - die 2D-Ansicht zeichnet mit `scatter`
            # ohnehin groessere Marker.
            "breite_px": 1,
        })
        self.update()

    def clear_river_overlay(self):
        """
        Das Flussnetz wieder abschalten.

        Der Fluss-Reiter ruft das, sobald sein Modus nicht mehr
        "Flussnetz" ist. Ohne diesen Weg blieb das Netz ueber JEDER
        Ansicht des Reiters liegen - der Nutzerbefund vom 2026-08-24
        (*"wenn man auf Ordnung geht dann aendert sich nichts und wenn man
        wieder auf gelaende geht aendert sich auch nichts"*).
        """
        self.set_layer_visibility("terrain", "river_overlay", False)
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
        elif layer_name in ("region_overlay", "kuesten_overlay", "wegbaender",
                            "river_overlay"):
            # Rohes Payload-Dict wie fuer den 2D-Renderer (regionen/heightmap/
            # ggf. kuesten_archetyp/kuesten_staerke), KEIN fertiges Array -
            # wird erst beim Zeichnen rasterisiert, siehe
            # _render_dict_rgba_overlay() (2026-08-13).
            if data is not None and not isinstance(data, dict):
                self.rendering_error.emit(f"Invalid overlay payload for {tab_type}.{layer_name}")
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
        # gestreckt. Jetzt: 1 Render-Einheit entspricht immer world_size_km/10 km.
        self.terrain_height_scale = 10.0 / (self.world_size_km * 1000.0)  # render units per meter

        # Vertikales Zentrum des Meshs in Welt-Y (Vertices bleiben unverändert
        # auf ihrer echten Höhe, siehe _generate_terrain_mesh) - die Kamera
        # zielt darauf statt fix auf (0,0,0). Ohne das lag reales Terrain
        # (Höhen typischerweise deutlich > 0) komplett außerhalb des
        # Kamera-Blickfelds und es wurde schlicht nichts gerendert.
        mid_height = (float(self.heightmap.min()) + float(self.heightmap.max())) / 2.0
        self.terrain_center_y = mid_height * self.terrain_height_scale

        # Blickpunkt nur nachziehen, solange ihn niemand bewegt hat. Sonst
        # wuerde jede neue Heightmap die Kamera aus der Position reissen, in
        # die der Nutzer sich gerade geflogen hat - und genau waehrend eines
        # Laufs trudeln laufend neue Heightmaps ein (Live-Vorschau).
        if not self._camera_target_touched:
            self.camera_target = [0.0, self.terrain_center_y, 0.0]

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

            # ALLE PFADE HIER WAREN RELATIV ZUM ARBEITSVERZEICHNIS.
            #
            # Damit lud das 3D-Display seine Shader NUR, wenn das Programm aus
            # dem Projektstamm gestartet wurde. Aufgefallen am 2026-08-16 beim
            # Start von `tools/mesh_werkstatt.py`: keiner der sechs Shader
            # wurde gefunden, danach zeichnete `_prepare_rendering()` ohne
            # Shaderprogramm weiter (`_activate_fallback_rendering()` schaltet
            # nur den Polygonmodus um) - und `glDrawElements` mit Programm 0
            # ist im Core-Profile undefiniert. Der Prozess starb hart mit
            # 0xC0000409 (STATUS_STACK_BUFFER_OVERRUN), ohne Python-Traceback.
            #
            # Deshalb zusaetzlich vom Projektstamm aus suchen, der aus
            # __file__ kommt (diese Datei liegt in gui/widgets/, also drei
            # Ebenen hoch). Die Reihenfolge bleibt: erst das bisherige
            # Verhalten, dann der absolute Pfad - was heute laeuft, laeuft
            # unveraendert weiter.
            wurzel = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            name = os.path.basename(filepath)

            # Versuche alternative Pfade
            alternative_paths = [
                filepath,  # Original
                f"shaders/{os.path.basename(filepath)}",  # Nur shaders/
                f"gui/shaders/{os.path.basename(filepath)}",  # gui/shaders/
                os.path.join(wurzel, "shaders", "3d_display", name),
                os.path.join(wurzel, "shaders", name),
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

    def _compile_wegband_shader(self):
        """
        Kompiliert das Mini-Shader-Programm fuer die Wegbaender (6.28,
        Nutzerfeedback 2026-08-16) - eigenes Paar statt des unlit
        wind_shader_program, siehe self.wegband_shader_program oben und
        shaders/3d_display/wegband.vert/.frag. Scheitert die Kompilierung,
        bleibt self.wegband_shader_program None - _render_wegbaender() no-opt
        dann still (die Punkt-/Linien-Uebersicht aus 6.23 bleibt trotzdem
        sichtbar).
        """
        vertex_shader_code = self._load_shader_from_file("shader/wegband.vert")
        fragment_shader_code = self._load_shader_from_file("shader/wegband.frag")

        if vertex_shader_code is None or fragment_shader_code is None:
            return

        try:
            vertex_shader = shaders.compileShader(vertex_shader_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_code, gl.GL_FRAGMENT_SHADER)
            self.wegband_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            self.rendering_error.emit(f"Wegband shader compilation failed: {str(e)}")
            self.wegband_shader_program = None

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

        # Initialer Licht-Positions-Wert (siehe self._light_pos) - wird JEDEN
        # Frame in _render_terrain_base() erneut gesetzt, sobald
        # set_sun_direction() die echte Sonnenposition geliefert hat (gleiches
        # Re-Set-pro-Frame-Muster wie useContours/useShadows dort).
        light_pos_location = gl.glGetUniformLocation(self.shader_program, "lightPos")
        if light_pos_location >= 0:
            gl.glUniform3f(light_pos_location, *self._light_pos)

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

        # Adaptive, fehler-getriebene Triangulierung (siehe adaptive_terrain_mesh.py):
        # wenige grosse Dreiecke auf flachen Flaechen (offenes Meer, Ebenen),
        # volle Pixel-Aufloesung an Klippen/Detailbereichen - statt vorher
        # ueberall exakt einem Vertex pro Pixel. Nur bei quadratischer
        # Heightmap mit Kantenlaenge 2^n+1 anwendbar (alle map_size-Werte
        # dieses Projekts erfuellen das); sonst automatischer Rueckfall auf
        # das bisherige Gleichmaessig-Gitter unten.
        adaptives_ergebnis = None

        # KUESTENSCHNITT (gui/widgets/kuesten_schnitt.py, docs/KUESTENMODELL.md
        # §8): das Gitter zellweise entlang der Nullkontur schneiden, damit die
        # Kuestenlinie eine echte Dreieckskante wird statt einer Rastertreppe.
        #
        # Gemessen bei 384 px: 4240 Konturvertices, davon 99.7 % NICHT auf
        # einer Pixelecke (Median-Versatz 0.164 px); das Quadtree liegt bei
        # 0.000004 px, also exakt darauf. Dicht (keine Kante an mehr als zwei
        # Dreiecken, Flaeche exakt), Konturhoehen exakt 0.
        #
        # PREIS, ehrlich benannt: der Schnitt geht vom VOLLEN Gitter aus und
        # kostet dadurch rund fuenfmal so viele Dreiecke wie das adaptive
        # Quadtree (das nur etwa ein Fuenftel des vollen Gitters braucht).
        # Deshalb umschaltbar, Vorgabe siehe value_default.py.
        try:
            from gui.config.value_default import KUESTEN_SCHNITT_AKTIV
        except ImportError:
            KUESTEN_SCHNITT_AKTIV = False
        if KUESTEN_SCHNITT_AKTIV and self._mesh_bauer is None:
            hat_kueste = bool((heightmap > 0).any() and (heightmap <= 0).any())
            if hat_kueste:
                from gui.widgets.kuesten_schnitt import baue_schnitt_mesh
                adaptives_ergebnis = baue_schnitt_mesh(
                    heightmap, self.terrain_scale_factor,
                    self.terrain_height_scale)
                if adaptives_ergebnis is None:
                    print("DEBUG: Kuestenschnitt lieferte nichts - "
                          "Rueckfall auf das Quadtree")

        # Alternativer Netzbauer (siehe setze_mesh_bauer) hat Vorrang.
        if adaptives_ergebnis is None and self._mesh_bauer is not None:
            adaptives_ergebnis = self._mesh_bauer(
                heightmap, self.terrain_scale_factor, self.terrain_height_scale)
            if adaptives_ergebnis is None:
                print("DEBUG: Alternativer Netzbauer lieferte nichts - "
                      "Rueckfall auf das Quadtree")

        if adaptives_ergebnis is None and ist_fuer_adaptives_mesh_geeignet(heightmap):
            adaptives_ergebnis = build_adaptive_mesh(
                heightmap, self.terrain_scale_factor, self.terrain_height_scale,
                fehler_toleranz_m=self._adaptive_mesh_fehler_toleranz_m)

        if adaptives_ergebnis is not None:
            self.mesh_vertices, self.mesh_indices, mesh_stats = adaptives_ergebnis
            herkunft = "aus Zwischenspeicher" if mesh_stats.get("aus_cache") else "neu gerechnet"
            # `blaetter` gibt es nur beim Quadtree - der alternative Netzbauer
            # liefert andere Kennzahlen, deshalb .get() statt [].
            zusatz = (f"{mesh_stats['blaetter']} Blaetter, "
                      if "blaetter" in mesh_stats else "")
            if "frei_verschoben" in mesh_stats:
                zusatz += (f"{mesh_stats['frei_verschoben']:.1%} Vertices frei "
                           f"verschoben (Median {mesh_stats['versatz_median_px']:.3f} px), ")
            print(f"DEBUG: Adaptives Terrain-Mesh ({herkunft}): "
                  f"{mesh_stats['dreiecke']}/{mesh_stats['voll_dreiecke']} "
                  f"Dreiecke ({mesh_stats['dreiecke'] / mesh_stats['voll_dreiecke']:.1%}), "
                  f"{zusatz}{mesh_stats['vertices']}/{mesh_stats['voll_vertices']} Vertices")
        else:
            print("DEBUG: Adaptives Mesh nicht anwendbar (Heightmap-Groesse) - Gleichmaessig-Gitter")
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

        # Gibt es ueberhaupt Meer? Nur dann wird die Wasserplatte gezeichnet -
        # bei einer Karte ohne negative Hoehen waere sie eine blaue Scheibe
        # quer durch das Tal. Die Platte wird bei jedem neuen Gelaende neu
        # aufgebaut, weil sich die Ausdehnung geaendert haben kann.
        #
        # ALTE VAO/VBO ERST LOESCHEN (2026-08-11, Pipeline-Log-Nutzerbefund -
        # GPU lief bei 1024px im Lauf der Generierung aus dem VRAM). Vorher
        # stand hier nur `self._wasser_vao = None` - das verwirft die Python-
        # Referenz, aber NICHT das GL-Objekt selbst (`glGenVertexArrays`/
        # `glGenBuffers` in _render_water_plane() unten legen dann bei jedem
        # naechsten Aufruf klaglos ein NEUES VAO/VBO an, weil `getattr(self,
        # "_wasser_vao", None) is None` wieder zutrifft). Bei jeder Mesh-
        # Neuerzeugung mit Wasser blieb das alte Paar orphaned auf der GPU
        # zurueck. Die einzelne Wasserplatte ist klein (6 Vertices), aber bei
        # wiederholten Regenerationen ohne Programmneustart summiert sich das -
        # und war neben dem Alle-Tabs-Redraw (6.14) ein zweiter Beitrag zum
        # beobachteten VRAM-Leck.
        if getattr(self, "_wasser_vao", None) is not None:
            gl.glDeleteVertexArrays(1, [self._wasser_vao])
        if getattr(self, "_wasser_vbo", None) is not None:
            gl.glDeleteBuffers(1, [self._wasser_vbo])
        self._hat_wasser = bool(np.any(heightmap < 0.0))
        self._wasser_vao = None
        self._wasser_vbo = None

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

        self._cleanup_overlay_texturen()
        # Wegband-Puffer mit freigeben: sie haengen an derselben Heightmap.
        # Der naechste Frame legt sie aus den zwischengespeicherten
        # Python-Arrays neu an, falls die Geometrie noch gilt.
        self._wegband_puffer_freigeben()

    def _wegband_puffer_freigeben(self):
        """
        GL-Puffer der Wegbaender loeschen.

        Gehoert zu der Zwischenspeicherung in `_render_wegbaender()`: solange
        die Geometrie gleich bleibt, ueberleben VAO/VBO/EBO viele Frames.
        Aendert sie sich - oder wird das Widget abgeraeumt -, muessen sie weg,
        sonst waechst der VRAM-Verbrauch bei jedem neuen Wegnetz. Genau dieser
        Fehler ist am 2026-08-11 schon einmal beim Wasser-VAO passiert (siehe
        _generate_terrain_mesh): dort stand nur `= None`, was die
        Python-Referenz verwirft, aber nicht das GL-Objekt.
        """
        gepuffert = self._wegband_cache.get("gl") if self._wegband_cache else None
        if not gepuffert:
            return
        vao, vbo, ebo = gepuffert
        try:
            gl.glDeleteBuffers(1, [vbo])
            gl.glDeleteBuffers(1, [ebo])
            gl.glDeleteVertexArrays(1, [vao])
        except Exception:                       # noqa: BLE001
            pass
        self._wegband_cache["gl"] = None

    def _cleanup_overlay_texturen(self):
        """
        Funktionsweise: Loescht alle im Overlay-Textur-Cache (siehe
        self._overlay_texture_cache in __init__) gehaltenen GL-Texturen und
        leert den Cache.
        Aufgabe: Verhindert, dass bei jeder Neugenerierung verwaiste
        Texturen im VRAM liegen bleiben - der Cache hebt Texturen ueber
        Objekt-Identitaet der Quelldaten auf (siehe _render_overlay()/
        _render_dict_rgba_overlay()); eine Neugenerierung ersetzt diese
        Objekte immer, das alte Eintraege danach nie wieder trifft. Wird von
        _cleanup_mesh_buffers() aus aufgerufen, also bei jedem Mesh-Neubau
        (_generate_terrain_mesh()), nicht erst beim Schliessen des Widgets.
        """
        for _, texture_id in self._overlay_texture_cache.values():
            gl.glDeleteTextures(1, [texture_id])
        self._overlay_texture_cache.clear()

    def _update_projection_matrix(self):
        """
        Funktionsweise: Aktualisiert Projection-Matrix bei Fenster-Resize
        Aufgabe: Setzt Perspective-Projection mit FOV aus gui_default.py
        """
        if self.height() == 0:
            return

        aspect_ratio = self.width() / self.height()

        # NEAR- UND FAR-PLANE WANDERN MIT DEM ZOOM (2026-08-24).
        #
        # Hier stand vorher fest `near=0.1, far=2000.0` mit dem Kommentar, das
        # Verhaeltnis 0.1:2000 sei "fuer einen 24-Bit-Tiefenpuffer
        # unkritisch". **Das war falsch, und es hat Geld gekostet:** die
        # Tiefengenauigkeit haengt fast allein an der NEAR-Plane, nicht am
        # Verhaeltnis. Nachgerechnet fuer einen 24-Bit-Puffer, Welt 10
        # Einheiten breit:
        #
        #     Kameraabstand 17 (Vorgabe):  0.000172 Welteinheiten
        #     Kameraabstand 30:            0.000536
        #     Kameraabstand 60:            0.002146
        #
        # Das Wegband schwebt 0.001 Einheiten ueber dem Gelaende. Ab
        # Kameraabstand ~45 ist die Tiefenaufloesung also GROESSER als der
        # Abstand - und genau das war der Nutzerbefund vom 2026-08-24: "der
        # weg verschwindet bei vielen kamera-bewegungs-aktionen ... auf
        # distanz kann er auch mal gestueckelt sein". Kein Fehler der
        # Wegdarstellung, sondern ein Praezisionsproblem der Projektion.
        #
        # Mitwandernd wird daraus (gemessen, gleiche Rechnung):
        #
        #     Kameraabstand 17:  0.000020  ( 8.6x genauer)
        #     Kameraabstand 30:  0.000035  (15.3x genauer)
        #     Kameraabstand 60:  0.000070  (30.8x genauer)
        #
        # `near = Abstand/20` ist reichlich sicher: das Gelaende ist 10
        # Einheiten breit, bei Abstand 17 liegt der naechste Punkt also rund
        # 10 Einheiten entfernt, die Near-Plane bei 0.85. Die Untergrenze
        # 0.1 haelt das bisherige Verhalten bei maximalem Hereinzoomen
        # (min_distance = 2.0, siehe zoom-Handler).
        abstand = float(getattr(self, "camera_distance", 17.0) or 17.0)
        near = max(0.1, abstand / 20.0)
        # Far grosszuegig hinter das Gelaende, aber nicht bei 2000 festgenagelt -
        # der Zuschlag deckt Panning und die Kartenausdehnung ab.
        far = abstand + 200.0

        self.projection_matrix = _create_perspective_matrix(
            fov=self.fov, aspect=aspect_ratio, near=near, far=far)

    # Die View-Matrix wird am Ende von _update_view_matrix() in ihrer X-Zeile
    # NEGIERT (Ost-West-Korrektur, dort ausfuehrlich begruendet). Bildschirm-
    # rechts entspricht deshalb NICHT cross(forward, up), sondern dessen
    # Gegenrichtung. Jede seitliche Bewegung - Panning wie Strafen - muss
    # dieses Vorzeichen mitnehmen, sonst laeuft sie spiegelverkehrt zu dem,
    # was der Nutzer auf dem Bildschirm sieht.
    SCREEN_RIGHT_SIGN = -1.0

    def _eye_offset(self):
        """Vektor vom Blickpunkt zum Auge, aus Distanz/Elevation/Azimut."""
        elevation_rad = math.radians(self.camera_elevation)
        azimuth_rad = math.radians(self.camera_azimuth)
        horizontal = self.camera_distance * math.cos(elevation_rad)
        return [horizontal * math.sin(azimuth_rad),
                self.camera_distance * math.sin(elevation_rad),
                horizontal * math.cos(azimuth_rad)]

    def _forward(self):
        """
        Die ECHTE Blickrichtung (vom Auge zum Blickpunkt), normiert.

        W/S fliegen hier entlang - mit Free Look (linke Maustaste) laesst sich
        die Neigung frei einstellen, man fliegt also dorthin, wo man hinsieht,
        wie ein Spectator im Shooter. Eine waagerechte Variante gab es hier
        kurzzeitig, solange die Neigung noch fest bei 55 Grad stand; mit
        beweglicher Neigung waere sie nur noch verwirrend.
        """
        offset = self._eye_offset()
        length = math.sqrt(sum(component * component for component in offset)) or 1.0
        return [-component / length for component in offset]

    def _screen_axes(self):
        """
        Die beiden Achsen der Bildebene (rechts, hoch) in Weltkoordinaten -
        die Ebene, in der Panning und Auf-/Abschweben stattfinden ("orthogonal
        zur Blickrichtung").
        """
        offset = self._eye_offset()
        length = math.sqrt(sum(component * component for component in offset)) or 1.0
        # Blickrichtung = vom Auge zum Blickpunkt.
        forward = [-component / length for component in offset]

        world_up = [0.0, 1.0, 0.0]
        right = [forward[1] * world_up[2] - forward[2] * world_up[1],
                 forward[2] * world_up[0] - forward[0] * world_up[2],
                 forward[0] * world_up[1] - forward[1] * world_up[0]]
        right_length = math.sqrt(sum(c * c for c in right)) or 1.0
        right = [c / right_length for c in right]

        # up AUS DEM UNGESPIEGELTEN right berechnen. Wendet man
        # SCREEN_RIGHT_SIGN vorher an, kippt es ueber das Kreuzprodukt in die
        # Hoch-Achse durch und die zeigt nach UNTEN - gemessen y = -0.57,
        # Leertaste senkte die Kamera statt sie zu heben. Die Spiegelung
        # betrifft ausschliesslich die Links/Rechts-Achse.
        up = [right[1] * forward[2] - right[2] * forward[1],
              right[2] * forward[0] - right[0] * forward[2],
              right[0] * forward[1] - right[1] * forward[0]]

        return [c * self.SCREEN_RIGHT_SIGN for c in right], up

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
        offset = self._eye_offset()
        target = list(self.camera_target)
        eye = [target[0] + offset[0], target[1] + offset[1], target[2] + offset[2]]
        up = [0, 1, 0]  # Y ist oben

        self.view_matrix = _create_lookat_matrix(eye, target, up)

        # Nutzer-Beobachtung: 3D-Terrain ist Ost-West gespiegelt gegenüber der
        # 2D-Ansicht ("links ist rechts und umgekehrt"). Ursache: die Kamera
        # steht bei azimuth=180° (siehe __init__-Kommentar, Fix für die
        # vorherige Nord-Süd-Verdrehung) auf der Süd-Seite und blickt nach
        # Norden - das ist unvermeidlich mit einer Ost-West-Spiegelung
        # verbunden, da ein reiner Kamera-Azimut-Wechsel bei fester Up-Achse
        # IMMER Tiefe (Nord/Süd) UND Links/Rechts gemeinsam vertauscht (wie
        # beim Herumgehen um einen Tisch: was vorher links war, ist von der
        # Gegenseite aus rechts) - beweisbar über right=cross(forward,up) in
        # _create_lookat_matrix(): bei azimuth=180 zeigt "right" auf -X
        # (Westen) statt +X (Osten). Durch Negieren NUR der X-Zeile (right-
        # Vektor + zugehörige Translation) wird ausschließlich dieser
        # Rechts/Links-Seiteneffekt aufgehoben, ohne die bereits korrekte
        # Nord-Süd-Blickrichtung erneut zu verändern - Osten landet dadurch
        # wieder auf der Bildschirm-rechten Seite, exakt wie in der
        # 2D-Ansicht. Ein reiner Vorzeichen-Wechsel im View-Space wie dieser
        # kehrt zwangsläufig die Dreiecks-Wickelrichtung um (Spiegelungen
        # kehren immer die Chiralität um) - deshalb wird in initializeGL()
        # (siehe glCullFace(GL_BACK) dort) zusätzlich glFrontFace(GL_CW)
        # gesetzt, um Backface-Culling weiterhin korrekt arbeiten zu lassen,
        # statt dass das gesamte Terrain plötzlich weggeculled würde.
        self.view_matrix[0, :] = -self.view_matrix[0, :]

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

        # Regionen/Kuestentypen als RGBA-Skin (2026-08-13, Nutzer-Vorgabe "die
        # 3D darstellung ALLER 2D maps, aber vor allem der Kuesten auf die 3D
        # Terrains bekommen") - dieselbe Rasterisierung wie 2D, siehe
        # _render_dict_rgba_overlay().
        if self.layer_visibility["terrain"].get("region_overlay"):
            self._render_dict_rgba_overlay("terrain", "region_overlay")
        if self.layer_visibility["terrain"].get("kuesten_overlay"):
            self._render_dict_rgba_overlay("terrain", "kuesten_overlay")

        # DIE SKALARKARTEN DES FLUSSREITERS (2026-08-26). Gewoehnliche
        # Overlays wie "slope" - `_colorize_layer()` holt ihre Farbskala aus
        # derselben `layer_ranges`-Tabelle wie die 2D-Ansicht, es braucht
        # also keinen eigenen Farbcode.
        if self.layer_visibility["terrain"].get("river_water"):
            self._render_overlay("terrain", "river_water")
        if self.layer_visibility["terrain"].get("river_order"):
            self._render_overlay("terrain", "river_order")

        # HOEHENFAKTOR UND VORONOI (2026-08-26). Nutzerwunsch ausdruecklich
        # "3d und 2D" - beide Wege in derselben Aenderung, siehe die stehende
        # Regel in CLAUDE.md.
        if self.layer_visibility["terrain"].get("hinterland_height"):
            self._render_overlay("terrain", "hinterland_height")
        if self.layer_visibility["terrain"].get("voronoi_map"):
            self._render_overlay("terrain", "voronoi_map")

        # DAS FLUSSNETZ (2026-08-24, Nutzerbefund *"dass man im 3D modus bei
        # dem Flussnetzwerk keine fluesse sehn kann"*).
        #
        # Es liegt im Terrain-Zweig, weil `gui/tabs/river_tab.py`
        # `generator_type = "terrain"` setzt und sich im 3D so anmeldet.
        # Gezeichnet wird nur, wenn der Fluss-Reiter Daten geschickt hat -
        # `_render_dict_rgba_overlay` kehrt bei leerem Slot von selbst um.
        if self.layer_visibility["terrain"].get("river_overlay", True):
            self._render_dict_rgba_overlay("terrain", "river_overlay")

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

        # Diagnose-Modi (Terrain Hub/Tilt/Fold/Fault/Intrusion Only) - siehe
        # gui/tabs/geology_tab.py _DELTA_DISPLAY_MODES. Signierte Verschiebungs-
        # werte (m), _colorize_layer() bekommt dieselben Layer-Namen wie die
        # 2D-Anzeige für ein konsistentes Farbschema.
        for delta_layer in ("terrain_hub_delta", "tilt_delta", "fold_delta",
                             "fault_delta", "intrusion_delta"):
            if self.layer_visibility["geology"][delta_layer]:
                self._render_overlay("geology", delta_layer)

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

        # "evaporation" 2026-08-26 nachgetragen - sie stand in beiden
        # Registern des Wasserreiters, fehlte aber sowohl in den
        # Vorgabe-Dicts als auch in dieser Liste. Gefunden vom neuen
        # tests/smoke_test_anzeige_register_3d.py bei seinem ERSTEN Lauf.
        water_layers = ["water_map", "soil_moisture", "erosion",
                        "sedimentation", "flow_map", "evaporation"]
        for layer in water_layers:
            if self.layer_visibility["water"][layer]:
                self._render_overlay("water", layer)

    def _render_erosion_tab(self):
        """
        Funktionsweise: Rendert den Erosion-Tab - Gelaende plus genau eines der
        acht Erosions-Overlays.
        Aufgabe: Die Radio-Buttons ueber dem Canvas sind die einzige Quelle der
        Wahrheit fuer die Sichtbarkeit (siehe BaseMapTab._push_data_to_current_display),
        deshalb wird hier schlicht ueber alle bekannten Layer iteriert.
        """
        self._render_terrain_base()
        for layer in ("erosion", "sedimentation", "net_change", "sediment_load",
                      "water_depth", "flow_velocity", "thermal_erosion", "thermal_deposition"):
            if self.layer_visibility["erosion"].get(layer):
                self._render_overlay("erosion", layer)

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

        # Globale Siedlungsuebersicht als Alpha-Skin (2026-08-13) - derselbe
        # Weg wie Plots/Regionen/Kuestentypen, gecacht ueber die Objekt-
        # Identitaet des RGBA-Arrays (siehe _overlay_cache_pruefen()).
        if self.layer_visibility["settlement"].get("uebersicht"):
            self._render_settlement_uebersicht()

        # Wege als echte Bandgeometrie (6.28) - NACH dem Skin, damit sie
        # darauf liegen, und mit eigenem Draw-Call.
        if self.layer_visibility["settlement"].get("wegbaender"):
            self._render_wegbaender()

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
                # Erosion teilt sich den Render-Modus mit Water: beide zeichnen
                # das Gelaende plus genau ein Skalarfeld-Overlay. Die Zahlen
                # sind Shader-Uniforms - ein neuer Wert braeuchte auch einen
                # neuen Zweig im Fragment-Shader.
                mode_value = {"terrain": 0, "geology": 1, "weather": 2, "water": 3,
                              "erosion": 3, "biome": 4, "settlement": 5}
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

            # lightPos jeden Frame neu setzen (wie useContours/useShadows) -
            # siehe set_sun_direction().
            light_pos_location = gl.glGetUniformLocation(self.shader_program, "lightPos")
            if light_pos_location >= 0:
                gl.glUniform3f(light_pos_location, *self._light_pos)

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

        self._render_wasserflaeche()

        if shadow_texture_id is not None:
            gl.glDeleteTextures(1, [shadow_texture_id])

    def _render_wasserflaeche(self):
        """
        Die Wasseroberflaeche als durchscheinende Platte auf 0 m.

        WARUM SIE NOETIG IST. Seit der Weltkarte kann die Heightmap negativ
        werden, und der Meeresboden ist dann zwar blau gefaerbt, liegt aber als
        SENKE da - aus schraeger Sicht sieht man in ein trockenes Becken. Die
        Platte schliesst es und macht aus der Senke ein Meer.

        Sie wird NACH dem Gelaende gezeichnet, mit Blending und ohne
        Tiefenschreiben: so verdeckt sie nichts, was ueber ihr liegt, und
        flaches Wasser laesst den Grund durchscheinen.

        Nur wenn es ueberhaupt Wasser gibt - bei einer Karte ohne negative
        Hoehen waere sie eine blaue Scheibe quer durch das Tal.
        """
        if self.shader_program is None or self.mesh_vertices is None:
            return
        if not getattr(self, "_hat_wasser", False):
            return

        if getattr(self, "_wasser_vao", None) is None:
            # Ausdehnung aus dem Gelaendenetz uebernehmen, damit die Platte
            # genau bis an den Kartenrand reicht.
            ecken = self.mesh_vertices.reshape(-1, 8)
            x0, x1 = float(ecken[:, 0].min()), float(ecken[:, 0].max())
            z0, z1 = float(ecken[:, 2].min()), float(ecken[:, 2].max())
            daten = np.array([
                [x0, 0.0, z0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [x1, 0.0, z0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [x1, 0.0, z1, 0.0, 1.0, 0.0, 1.0, 1.0],
                [x0, 0.0, z0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [x1, 0.0, z1, 0.0, 1.0, 0.0, 1.0, 1.0],
                [x0, 0.0, z1, 0.0, 1.0, 0.0, 0.0, 1.0],
            ], dtype=np.float32)
            self._wasser_vao = gl.glGenVertexArrays(1)
            self._wasser_vbo = gl.glGenBuffers(1)
            gl.glBindVertexArray(self._wasser_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._wasser_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, daten.nbytes, daten,
                            gl.GL_STATIC_DRAW)
            schritt = 8 * 4
            for platz, versatz in ((0, 0), (1, 3 * 4), (2, 6 * 4)):
                grosse = 2 if platz == 2 else 3
                gl.glVertexAttribPointer(platz, grosse, gl.GL_FLOAT,
                                         gl.GL_FALSE, schritt,
                                         gl.ctypes.c_void_p(versatz))
                gl.glEnableVertexAttribArray(platz)
            gl.glBindVertexArray(0)

        # renderMode kurz auf 6 stellen und danach auf den Wert des aktuellen
        # Tabs zuruecksetzen. ZurueckLESEN (glGetUniformiv) waere fehleranfaellig
        # und je nach Treiber verschieden - der Sollwert steht ohnehin fest.
        modi = {"terrain": 0, "geology": 1, "weather": 2, "water": 3,
                "erosion": 3, "biome": 4, "settlement": 5}
        ort = gl.glGetUniformLocation(self.shader_program, "renderMode")
        if ort >= 0:
            gl.glUniform1i(ort, 6)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glBindVertexArray(self._wasser_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

        if ort >= 0:
            gl.glUniform1i(ort, modi.get(self.current_tab, 0))

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

        # ZWISCHENGESPEICHERTE TEXTUR WIEDERVERWENDEN, WENN DIE QUELLDATEN
        # DIESELBEN SIND (2026-08-13, siehe self._overlay_texture_cache in
        # __init__ fuer die volle Begruendung - 0.6s/Frame ohne diesen Cache).
        # `water_map` haengt zusaetzlich von `water_biomes_reference` ab, das
        # unabhaengig vom Overlay selbst wechseln kann - beide Objekt-IDs
        # gehen in den Vergleich ein.
        cache_schluessel = (tab_type, layer_name)
        wasser_id = (id(self.water_biomes_reference)
                    if layer_name == "water_map" else None)
        daten_id = (id(overlay_data), wasser_id)
        alter_eintrag = self._overlay_texture_cache.get(cache_schluessel)

        if alter_eintrag is not None and alter_eintrag[0] == daten_id:
            texture_id = alter_eintrag[1]
        else:
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

            if alter_eintrag is not None:
                gl.glDeleteTextures(1, [alter_eintrag[1]])
            texture_id = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, rgb.shape[1], rgb.shape[0],
                             0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(rgb))
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            self._overlay_texture_cache[cache_schluessel] = (daten_id, texture_id)

        try:
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
                # Erosion teilt sich den Render-Modus mit Water: beide zeichnen
                # das Gelaende plus genau ein Skalarfeld-Overlay. Die Zahlen
                # sind Shader-Uniforms - ein neuer Wert braeuchte auch einen
                # neuen Zweig im Fragment-Shader.
                mode_value = {"terrain": 0, "geology": 1, "weather": 2, "water": 3,
                              "erosion": 3, "biome": 4, "settlement": 5}
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
            # KEIN glDeleteTextures mehr hier (2026-08-13) - die Textur bleibt
            # im Cache fuer den naechsten Frame stehen, siehe
            # self._overlay_texture_cache oben. Aufgeraeumt wird sie erst,
            # wenn neue Quelldaten sie ersetzen (oben, "alter_eintrag") oder
            # beim naechsten Mesh-Neubau (_cleanup_overlay_texturen(), von
            # _generate_terrain_mesh() aus aufgerufen).
            pass

    def _sample_field_bilinear(self, field: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                                width: int, height: int) -> np.ndarray:
        """
        Bilineares Sampling eines (H,W)- oder (H,W,C)-Feldes an beliebigen
        (nicht ganzzahligen) Pixelkoordinaten - für die Stromlinien-Integration
        in _render_wind_streamline_segments() gebraucht (Wind UND Heightmap an
        denselben Zwischenpositionen). Eigene Implementierung statt
        scipy.ndimage.map_coordinates, da dieses Modul scipy bisher nicht
        importiert und für diesen einen Zweck keine neue Abhängigkeit
        eingeführt werden soll.
        """
        x0 = np.clip(np.floor(xs).astype(np.int64), 0, width - 2)
        y0 = np.clip(np.floor(ys).astype(np.int64), 0, height - 2)
        fx = xs - x0
        fy = ys - y0
        f00 = field[y0, x0]
        f10 = field[y0, x0 + 1]
        f01 = field[y0 + 1, x0]
        f11 = field[y0 + 1, x0 + 1]
        if field.ndim == 3:
            fx = fx[:, None]
            fy = fy[:, None]
        top = f00 * (1 - fx) + f10 * fx
        bot = f01 * (1 - fx) + f11 * fx
        return top * (1 - fy) + bot * fy

    def _compute_wind_streamline_segments(self, wind_data: np.ndarray, width: int, height: int,
                                           global_max_mag: float):
        """
        Verfolgt Wind-Stromlinien per CPU-Vektorintegration (Euler, feste
        Schrittzahl, über alle Seeds gleichzeitig vektorisiert) - das 3D-
        Äquivalent zu map_display_2d.py's matplotlib streamplot in
        _render_wind_map() (siehe [[project-wind-3d-streamlines]]). Zeigt den
        tatsächlichen Strömungsverlauf/Wirbel, was diskrete Pfeile an
        einzelnen Punkten nicht leisten. Fester RNG-Seed für die Start-
        punkte, damit die Linien zwischen Frames nicht "springen" (nur
        Kamerabewegung ändert sich, nicht die zugrundeliegenden Daten).

        Rückgabe: (positions, colors) je (n_lines, n_steps, 2, 3) - direkt
        mit den Pfeil-Segmenten aus _render_wind_vectors() konkatenierbar
        (gleiches Vertex-Format: Position + Farbe, GL_LINES).
        """
        n_lines = 42
        n_steps = 22
        step_len = max(width, height) / 90.0

        rng = np.random.default_rng(1234)
        x = rng.uniform(1.0, width - 2.0, n_lines).astype(np.float64)
        y = rng.uniform(1.0, height - 2.0, n_lines).astype(np.float64)

        path_x = np.zeros((n_steps + 1, n_lines), dtype=np.float64)
        path_y = np.zeros((n_steps + 1, n_lines), dtype=np.float64)
        path_mag = np.zeros((n_steps + 1, n_lines), dtype=np.float32)
        path_x[0], path_y[0] = x, y
        uv0 = self._sample_field_bilinear(wind_data, x, y, width, height)
        path_mag[0] = np.hypot(uv0[:, 0], uv0[:, 1])

        # Nutzer-Beobachtung (Screenshot 2026-07-23): Stromlinien "knicken" am
        # Kartenrand ab und laufen daran entlang statt sauber abzuschneiden -
        # verursacht durch np.clip() hier, das eine aus dem Gitter
        # herauslaufende Position auf den Rand zurückfaltete (Linie "rutscht"
        # sichtbar am Rand entlang) statt die Linie dort enden zu lassen.
        # map_display_2d.py's matplotlib-streamplot hat dasselbe Problem
        # bereits über NaN-Zellen sauber gelöst (siehe _render_wind_map()) -
        # hier (reine Vertex-Positionen, kein NaN-Support in GL_LINES) über
        # ein "alive"-Flag pro Linie: sobald eine Linie das Gitter verlassen
        # würde, friert ihre Position für alle Folgeschritte ein (Segmente
        # danach haben Länge 0, unsichtbar), statt am Rand weiterzurutschen.
        alive = np.ones(n_lines, dtype=bool)
        cur_x, cur_y = x.copy(), y.copy()
        for step in range(1, n_steps + 1):
            uv = self._sample_field_bilinear(wind_data, cur_x, cur_y, width, height)
            mag = np.hypot(uv[:, 0], uv[:, 1])
            safe_mag = np.where(mag > 1e-6, mag, 1.0)
            new_x = cur_x + (uv[:, 0] / safe_mag) * step_len
            new_y = cur_y + (uv[:, 1] / safe_mag) * step_len
            alive &= (new_x >= 0) & (new_x <= width - 1) & (new_y >= 0) & (new_y <= height - 1)
            cur_x = np.where(alive, new_x, cur_x)
            cur_y = np.where(alive, new_y, cur_y)
            path_x[step], path_y[step] = cur_x, cur_y
            path_mag[step] = np.where(alive, mag, path_mag[step - 1])

        terrain_h = self._sample_field_bilinear(
            self.heightmap, path_x.ravel(), path_y.ravel(), width, height
        ).reshape(path_x.shape).astype(np.float32)

        pos_x = (path_x.astype(np.float32) / (width - 1) - 0.5) * width * self.terrain_scale_factor
        pos_z = (path_y.astype(np.float32) / (height - 1) - 0.5) * height * self.terrain_scale_factor
        hover = 0.05
        pos_y = terrain_h * self.terrain_height_scale + hover

        safe_global_max = max(global_max_mag, 1e-6)
        t = np.clip(path_mag / safe_global_max, 0.0, 1.0)[..., None]
        calm_color = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        strong_color = np.array([1.0, 0.25, 0.05], dtype=np.float32)
        vertex_color = calm_color * (1 - t) + strong_color * t  # (n_steps+1, n_lines, 3)

        a_pos = np.stack([pos_x[:-1], pos_y[:-1], pos_z[:-1]], axis=-1)  # (n_steps, n_lines, 3)
        b_pos = np.stack([pos_x[1:], pos_y[1:], pos_z[1:]], axis=-1)
        a_color = vertex_color[:-1]
        b_color = vertex_color[1:]

        positions = np.stack([a_pos, b_pos], axis=-2)  # (n_steps, n_lines, 2, 3)
        colors = np.stack([a_color, b_color], axis=-2)
        return positions, colors

    def _render_wind_vectors(self):
        """
        Funktionsweise: Zeichnet Windrichtungs-Pfeile UND Stromlinien als
        GL_LINES über dem Terrain-Mesh (analog zu map_display_2d.py's
        matplotlib-Quiver+Streamplot in _render_wind_map()) - die reine
        Magnitude-Heatmap aus _render_overlay() zeigt nur Windstärke, keine
        Richtung/Verwirbelung, und blendet auf dem Terrain kaum sichtbar ein
        (User-Report: "nichts zu erkennen").
        Aufgabe: Sampled ein Raster aus overlay_data["weather"]["wind"]
        ((H,W,2) u/v m/s in Spalten-/Zeilen-Richtung, siehe _generate_terrain_
        mesh's pos_x/pos_z-Konvention: u->X/Ost-West, v->Z/Süd-Nord), platziert
        an jedem Sample-Punkt einen kleinen Pfeil (Schaft + zwei Widerhaken)
        knapp über der Terrain-Oberfläche, und ergänzt per
        _compute_wind_streamline_segments() verfolgte Stromlinien, die dem
        tatsächlichen Strömungsverlauf folgen (siehe
        [[project-wind-3d-streamlines]]). Länge/Farbe skalieren mit der
        lokalen Windstärke (weiß=schwach, orange-rot=stark), auf derselben
        globalen Skala für Pfeile UND Stromlinien.
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

        # Windstärke-Skala: FESTER Referenzwert aus derselben zentralen
        # Quelle wie map_display_2d.py's _render_wind_map()-Colorbar
        # (CanvasSettings.CANVAS_2D["layer_ranges"]["wind_map"], 0-40 m/s),
        # NICHT mehr das Max des aktuellen Frames/dieser einen Karte.
        # Nutzer-Korrektur: mit einer pro-Karte-relativen Skala bedeutete
        # "volle Farbe/Länge" auf jeder Karte etwas anderes (die jeweils
        # windigste Stelle DIESER Karte), wodurch Pfeile zwischen
        # verschiedenen Karten/Frames nicht vergleichbar waren - exakt
        # dieselbe Bewandnis-Lücke wie beim 2D-Colorbar-Fix. Derselbe
        # Lookup wie 2D garantiert, dass beide Ansichten immer dieselbe
        # Skala zeigen, ohne eine zweite Konstante parallel zu pflegen.
        full_magnitude = np.hypot(wind_data[:, :, 0], wind_data[:, :, 1])
        if float(full_magnitude.max()) < 1e-6:
            return
        _, _, wind_vmax, _ = _get_layer_range("wind_map")
        global_max_mag = float(wind_vmax) if wind_vmax else float(full_magnitude.max())

        grid = 20  # dichter als zuvor (14) - Nutzer-Wunsch "kleinteiliger"
        y_idx = np.linspace(0, height - 1, min(grid, height)).astype(int)
        x_idx = np.linspace(0, width - 1, min(grid, width)).astype(int)
        yy, xx = np.meshgrid(y_idx, x_idx, indexing='ij')

        u = wind_data[yy, xx, 0].astype(np.float32)
        v = wind_data[yy, xx, 1].astype(np.float32)
        magnitude = np.sqrt(u ** 2 + v ** 2)

        pos_x = (xx.astype(np.float32) / (width - 1) - 0.5) * width * self.terrain_scale_factor
        pos_z = (yy.astype(np.float32) / (height - 1) - 0.5) * height * self.terrain_scale_factor
        terrain_h = self.heightmap[yy, xx].astype(np.float32)
        hover = 0.05  # etwas über der Oberfläche schweben, gegen Z-Fighting mit dem Mesh
        pos_y = terrain_h * self.terrain_height_scale + hover

        # Pfeillänge proportional zur lokalen Windstärke (relativ zur
        # globalen Windstärke-Skala), gedeckelt auf einen Bruchteil des
        # Rasterabstands, damit sich benachbarte Pfeile nicht überlappen.
        cell_span = self.terrain_scale_factor * max(width, height) / grid
        max_arrow_len = cell_span * 0.8
        safe_mag = np.where(magnitude > 1e-6, magnitude, 1.0)
        unit_x = u / safe_mag
        unit_z = v / safe_mag
        arrow_len = np.clip(magnitude / global_max_mag, 0.0, 1.0) * max_arrow_len
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

        # Farbe: weiß (schwach) -> orange-rot (stark), linear nach globaler Windstärke-Skala.
        t = np.clip(magnitude / global_max_mag, 0.0, 1.0)[..., None]
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

        arrow_positions = np.stack(
            [shaft_a, shaft_b, barb1_a, barb1_b, barb2_a, barb2_b], axis=-2)  # (grid,grid,6,3)
        arrow_colors = np.broadcast_to(color[:, :, None, :], arrow_positions.shape)

        stream_positions, stream_colors = self._compute_wind_streamline_segments(
            wind_data, width, height, global_max_mag)

        vertex_data = np.concatenate([
            np.concatenate([arrow_positions, arrow_colors], axis=-1).reshape(-1, 6),
            np.concatenate([stream_positions, stream_colors], axis=-1).reshape(-1, 6),
        ], axis=0).astype(np.float32)
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

    def setze_auswahlobjekte(self, orte, wege, welt_km=None):
        """
        Hinterlegt, was angeklickt werden kann (docs/OFFENE_PUNKTE.md 6.29).

        `orte` ist eine Liste von (x_px, y_px, kennung), `wege` eine Liste von
        (pfad_px, kennung). Die Umrechnung nach Weltkoordinaten passiert HIER
        und EINMAL - beim Klick soll nur noch projiziert und verglichen werden,
        nicht erst gerechnet.

        Verwendet dieselbe Pixel->Welt-Formel wie Terrain-Mesh und Wegbaender.
        Eine eigene Rechnung waere eine dritte Wahrheit; sie wuerde die
        Trefferflaechen unauffaellig gegen das verschieben, was man sieht.
        """
        self._auswahl_orte = np.zeros((0, 3), dtype=np.float64)
        self._auswahl_orte_kennung = []
        self._auswahl_wege = []
        self._auswahl_wege_kennung = []
        if self.heightmap is None:
            return

        from gui.widgets.wege_geometrie import _hoehe_an, SCHWEBE_ANTEIL
        hoehe_px, breite_px = self.heightmap.shape
        tsf = self.terrain_scale_factor
        ths = self.terrain_height_scale
        # Etwas ueber dem Boden, damit ein Ort auf einer Kuppe nicht im
        # Gelaende sitzt - derselbe Aufschlag wie bei den Wegbaendern.
        schwebe = SCHWEBE_ANTEIL * hoehe_px * tsf

        def nach_welt(xs, ys):
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            h = _hoehe_an(np.asarray(self.heightmap, dtype=np.float32), xs, ys)
            return np.stack([
                (np.clip(xs, 0, breite_px - 1) / (breite_px - 1) - 0.5) * breite_px * tsf,
                h * ths + schwebe,
                (np.clip(ys, 0, hoehe_px - 1) / (hoehe_px - 1) - 0.5) * hoehe_px * tsf,
            ], axis=1)

        if orte:
            xs = [float(o[0]) for o in orte]
            ys = [float(o[1]) for o in orte]
            self._auswahl_orte = nach_welt(xs, ys)
            self._auswahl_orte_kennung = [o[2] for o in orte]

        for pfad, kennung in (wege or []):
            punkte = np.asarray(pfad, dtype=np.float64).reshape(-1, 2)
            if len(punkte) < 2:
                continue
            self._auswahl_wege.append(nach_welt(punkte[:, 0], punkte[:, 1]))
            self._auswahl_wege_kennung.append(kennung)

    def _auswahl_pruefen(self, position):
        """Sucht das Objekt unter dem Mauszeiger und meldet es per Signal."""
        if self.model_matrix is None or self.view_matrix is None \
                or self.projection_matrix is None:
            return
        try:
            from gui.widgets.karten_auswahl import treffer_suchen
            treffer = treffer_suchen(
                position.x(), position.y(),
                self._auswahl_orte, self._auswahl_orte_kennung,
                self._auswahl_wege, self._auswahl_wege_kennung,
                self.model_matrix, self.view_matrix, self.projection_matrix,
                max(self.width(), 1), max(self.height(), 1))
            # GEWAEHLTEN WEG MERKEN und neu zeichnen lassen - sonst
            # passiert beim Klick zwar etwas im Textfeld, aber im Bild ist
            # nichts markiert (Nutzerbefund 2026-08-24: "nicht markierbar").
            # Der Shader kann die Einfaerbung seit dem 2026-08-16, sie wurde
            # nur nie eingeschaltet.
            vorher = self._ausgewaehlter_weg
            if treffer and treffer.get("art") == "weg":
                self._ausgewaehlter_weg = treffer.get("index")
            else:
                self._ausgewaehlter_weg = None
            if vorher != self._ausgewaehlter_weg:
                self.update()

            self.objekt_gewaehlt.emit(treffer)
        except Exception as fehler:                        # pragma: no cover
            # KEIN self.logger hier - diese Klasse hat keinen (geprueft), ein
            # Zugriff darauf wuerde im Fehlerfall SELBST scheitern und die
            # eigentliche Ursache verschlucken. Das vorhandene Fehlersignal
            # ist der Weg, den auch die uebrigen Methoden dieser Klasse gehen.
            self.rendering_error.emit(f"Auswahl fehlgeschlagen: {fehler}")

    # Farbe je Wegkategorie, gemeinsam mit der 2D-Textur-Fassung (6.23) -
    # eine zweite Farbwahl waere eine zweite Wahrheit.
    _WEGBAND_FARBEN = {"wege": (0.82, 0.47, 0.13),      # darkorange
                       "seewege": (0.25, 0.41, 0.88)}   # royalblue

    # Deckkraft der Baender (2026-08-16, Nutzerfeedback: "so dass die
    # unregelmaessige form sich schoen auf die textur schmiegt" statt
    # blickdicht auf dem Gelaende zu "kleben") - unter 1.0, damit ein Rest
    # Terraintextur durchscheint wie bei einem echten Decal.
    _WEGBAND_ALPHA = 0.88

    def _render_wegbaender(self):
        """
        Wege als echte Bandgeometrie statt als Textur (docs/OFFENE_PUNKTE.md
        6.28, Nutzerwunsch 2026-08-13: "es sieht nicht so schoen aus mit den
        strassen als textur ... ich will fuer den editor ein bisschen
        schoenere optik").

        NUTZT DAS EIGENE `wegband_shader_program` (2026-08-16), nicht mehr
        das geteilte `wind_shader_program`. Nutzerbefund an der ersten
        Fassung: "die wege sehen auch schlecht aus, passen nicht ins
        lighting, sehen total fake aus" - das unlit Position+Farbe-Programm
        der Windpfeile hat keine Normalen und keine Lichtrechnung. Ein
        eigenes Paar (shaders/3d_display/wegband.vert/.frag) statt den
        geteilten Shader zu erweitern, damit die Windpfeile im Wetter-Reiter
        unangetastet bleiben.

        Die Geometrie kommt aus `gui/widgets/wege_geometrie.py` (inklusive
        Normalen, bilinear aus dem Terrain-Normalenfeld) und wird ueber die
        Objekt-Identitaet der Wegliste zwischengespeichert - sie pro Frame
        neu zu bauen waere derselbe Fehler wie bei den Overlays (6.21), nur
        teurer. Land- und Seewege liegen im selben Puffer, werden aber mit
        ZWEI Draw-Calls gezeichnet (je Kategorie eine Farbe als Uniform statt
        als Vertex-Attribut - ein Wegband hat ohnehin nur eine Farbe, ein
        drittes Attribut haette den Puffer nur unnoetig vergroessert).
        """
        payload = self.overlay_data["settlement"].get("wegbaender")
        if not isinstance(payload, dict):
            return
        if not self.wegband_shader_program:
            # LAUT MELDEN, nicht still aussteigen. Ohne das ist "der Shader
            # liess sich nicht uebersetzen" von "es gibt gerade keine Wege"
            # nicht zu unterscheiden - dieselbe Falle wie bei den
            # GPU-Rueckfaellen und der 2^n+1-Bedingung (siehe CLAUDE.md).
            if not getattr(self, "_wegband_shader_fehlt_gemeldet", False):
                self._wegband_shader_fehlt_gemeldet = True
                print("FEHLER: wegband-Shaderprogramm fehlt - Wege werden "
                      "NICHT gezeichnet (Uebersetzung fehlgeschlagen?).")
                self.rendering_error.emit("Wegband-Shader fehlt - keine Wege in 3D")
            return
        if self.heightmap is None or self.model_matrix is None:
            return

        wege = payload.get("wege") or []
        seewege = payload.get("seewege") or []
        if not wege and not seewege:
            return

        schluessel = (id(payload), self.heightmap.shape,
                      round(float(self.terrain_scale_factor), 9),
                      round(float(self.terrain_height_scale), 12))
        gepuffert = self._wegband_cache.get("schluessel")
        if gepuffert != schluessel:
            from gui.widgets.wege_geometrie import (
                baue_wegbaender, WEG_BREITE_M, SEEWEG_BREITE_M)
            welt_km = float(getattr(self, "world_size_km", 21.3) or 21.3)

            v_alle, i_alle, kategorien = [], [], []
            weg_bereiche = {}
            vertex_versatz, index_versatz = 0, 0
            for name, liste, breite in (
                    ("wege", wege, WEG_BREITE_M),
                    ("seewege", seewege, SEEWEG_BREITE_M)):
                if not liste:
                    continue
                v, i, bereiche = baue_wegbaender(
                    liste, self.heightmap, welt_km,
                    self.terrain_scale_factor, self.terrain_height_scale,
                    breite_m=breite)
                if len(v) == 0:
                    continue
                kategorien.append((name, index_versatz, len(i)))
                # Je EINZELNEM Weg merken, wo er im gemeinsamen Indexpuffer
                # liegt - das ist die Grundlage fuers Einfaerben des
                # angeklickten Weges. Der Schluessel ist die Position in der
                # Auswahlliste (erst Landwege, dann Seewege), damit er zum
                # Index aus `treffer_suchen` passt.
                versatz_in_auswahl = 0 if name == "wege" else len(wege)
                for weg_index, start, anzahl in bereiche:
                    weg_bereiche[versatz_in_auswahl + weg_index] = (
                        index_versatz + start, anzahl)
                v_alle.append(v)
                i_alle.append(i + vertex_versatz)
                vertex_versatz += len(v)
                index_versatz += len(i)

            if not v_alle:
                self._wegband_puffer_freigeben()
                self._wegband_cache = {"schluessel": schluessel, "daten": None}
            else:
                self._wegband_puffer_freigeben()
                self._wegband_cache = {
                    "schluessel": schluessel,
                    "daten": (np.ascontiguousarray(np.concatenate(v_alle)),
                              np.ascontiguousarray(np.concatenate(i_alle)),
                              kategorien, weg_bereiche),
                }

        daten = self._wegband_cache.get("daten")
        if not daten:
            return
        vertex_data, index_data, kategorien, weg_bereiche = daten

        # GL-PUFFER NUR EINMAL JE GEOMETRIE, NICHT JE FRAME (2026-08-24).
        #
        # Vorher legte diese Funktion bei JEDEM Frame VAO/VBO/EBO neu an, lud
        # die kompletten Baender zur GPU und loeschte alles wieder. Kein Leck
        # - der finally-Zweig raeumte sauber auf -, aber der Cache
        # daruber sparte damit nur das Rechnen in Python, nicht die
        # Uebertragung. Jetzt haengen die Puffer am Cache und werden erst
        # freigegeben, wenn sich die Geometrie wirklich aendert.
        gepuffert = self._wegband_cache.get("gl")
        if gepuffert is None:
            vao = gl.glGenVertexArrays(1)
            vbo = gl.glGenBuffers(1)
            ebo = gl.glGenBuffers(1)
            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data,
                            gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_data.nbytes,
                            index_data, gl.GL_STATIC_DRAW)
            # [Position xyz | Normale xyz | Deckung] = 7 float je Vertex
            stride = 7 * 4
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
            gl.glEnableVertexAttribArray(1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                     gl.GLvoidp(3 * 4))
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 1, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                     gl.GLvoidp(6 * 4))
            gl.glBindVertexArray(0)
            self._wegband_cache["gl"] = (vao, vbo, ebo)
            gepuffert = self._wegband_cache["gl"]
        vao, vbo, ebo = gepuffert

        try:
            gl.glUseProgram(self.wegband_shader_program)
            for name, matrix in (("model", self.model_matrix),
                                 ("view", self.view_matrix),
                                 ("projection", self.projection_matrix)):
                if matrix is None:
                    continue
                ort = gl.glGetUniformLocation(self.wegband_shader_program, name)
                if ort >= 0:
                    gl.glUniformMatrix4fv(ort, 1, gl.GL_TRUE, matrix.flatten())

            licht_ort = gl.glGetUniformLocation(self.wegband_shader_program, "lightPos")
            if licht_ort >= 0:
                gl.glUniform3f(licht_ort, *self._light_pos)
            alpha_ort = gl.glGetUniformLocation(self.wegband_shader_program, "wegAlpha")
            if alpha_ort >= 0:
                gl.glUniform1f(alpha_ort, self._WEGBAND_ALPHA)
            # Auswahl-Einfaerbung (6.29/Nutzerwunsch) ist im Shader vorbereitet,
            # aber das Anklicken einzelner Wege setzt hier noch nichts - siehe
            # docs/OFFENE_PUNKTE.md 6.27 (Teil 2, Anklicken).
            ausgewaehlt_ort = gl.glGetUniformLocation(self.wegband_shader_program, "ausgewaehlt")
            if ausgewaehlt_ort >= 0:
                gl.glUniform1i(ausgewaehlt_ort, 0)

            gl.glBindVertexArray(vao)

            # Baender sind einseitig gewickelt wie das Terrain, aber sie von
            # unten zu sehen ist beim Drehen normal - deshalb hier KEIN
            # Backface-Culling.
            war_culling = gl.glIsEnabled(gl.GL_CULL_FACE)
            if war_culling:
                gl.glDisable(gl.GL_CULL_FACE)

            # ALPHA-BLENDING FUER DEN DECAL-EFFEKT (2026-08-16) - vorher
            # deckend (Alpha im Fragment-Shader war immer 1.0). GL_BLEND ist
            # sonst im gesamten 3D-Renderpfad nirgends aktiv, deshalb hier
            # einfach an- und danach wieder ausschalten statt den vorherigen
            # Zustand/Blendfaktor zu sichern.
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

            # TIEFENVERSATZ STATT WELTVERSATZ (2026-08-24).
            #
            # Bisher hielt allein `SCHWEBE_ANTEIL` das Band ueber dem
            # Gelaende - ein Versatz in der WELT. Der hat zwei Nachteile: bei
            # flachem Blickwinkel sieht man unter die Strasse, und er muss
            # gross genug fuer den schlimmsten Querhang sein, wirkt also
            # ueberall sonst zu hoch. `glPolygonOffset` verschiebt die
            # Fragmente nur im TIEFENPUFFER. Das Band liegt damit
            # geometrisch auf dem Boden und gewinnt trotzdem den Tiefentest.
            # Negative Werte ziehen zur Kamera hin.
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glPolygonOffset(-3.0, -6.0)

            for name, index_start, index_count in kategorien:
                farbe = self._WEGBAND_FARBEN.get(name, (0.7, 0.7, 0.7))
                farbe_ort = gl.glGetUniformLocation(self.wegband_shader_program, "wegFarbe")
                if farbe_ort >= 0:
                    gl.glUniform3f(farbe_ort, *farbe)
                gl.glDrawElements(gl.GL_TRIANGLES, index_count, gl.GL_UNSIGNED_INT,
                                  gl.GLvoidp(index_start * 4))

            # DER GEWAEHLTE WEG NOCH EINMAL, roetlich eingefaerbt.
            #
            # Zweiter Draw-Call statt einer Farbe je Vertex: die Auswahl
            # aendert sich pro Klick, die Geometrie aber nicht - ein
            # Vertexattribut muesste bei jeder Auswahl neu hochgeladen werden.
            # Der Shader hat dafuer den `ausgewaehlt`-Schalter.
            bereich = weg_bereiche.get(self._ausgewaehlter_weg)
            if bereich is not None and ausgewaehlt_ort >= 0:
                start, anzahl = bereich
                gl.glUniform1i(ausgewaehlt_ort, 1)
                gl.glDrawElements(gl.GL_TRIANGLES, anzahl, gl.GL_UNSIGNED_INT,
                                  gl.GLvoidp(start * 4))
                gl.glUniform1i(ausgewaehlt_ort, 0)

            gl.glPolygonOffset(0.0, 0.0)
            gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glDisable(gl.GL_BLEND)
            if war_culling:
                gl.glEnable(gl.GL_CULL_FACE)
            gl.glBindVertexArray(0)
        finally:
            # Puffer bleiben am Cache haengen, siehe oben - hier NICHT
            # loeschen, sonst ist der Sinn der Zwischenspeicherung dahin.
            gl.glBindVertexArray(0)

    def _render_plot_boundaries(self):
        """
        Funktionsweise: Rendert das PlotPhysicsSystem-Ergebnis als texturierten
        "Skin" auf dem Terrain-Mesh statt eigener 3D-Wireframe-/Marker-
        Geometrie (Nutzer-Vorgabe: "die 2D-Darstellung als Skin auf das
        Terrain legen", siehe [[project-settlement-plot-physics-rebuild]]
        Teil 4) - overlay_data["settlement"]["plots"] enthält bereits die
        fertig gerasterte (H,W,4)-RGBA-Textur (siehe map_display_2d.py's
        rasterize_plot_boundaries_rgba(), gepusht von settlement_tab.py's
        apply_3d_overlays()).
        Aufgabe: Zeichnet Plot-Kantennetz/Straßen-Tiers/Wildnisgrenzen/Kerne/
        Nodes auf der Terrain-Oberfläche.
        """
        rgba = self.overlay_data["settlement"]["plots"]
        if rgba is None or not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] != 4:
            return
        cache_schluessel = ("settlement", "plots")
        texture_id = self._overlay_cache_pruefen(cache_schluessel, id(rgba))
        if texture_id is None:
            texture_id = self._overlay_textur_hochladen(cache_schluessel, id(rgba), rgba)
        self._render_rgba_textur(texture_id)

    def _render_settlement_uebersicht(self):
        """
        Funktionsweise: Zeichnet die globale Siedlungsuebersicht (Staedte,
        Landmarken, Roadsites, Land- und Seewege) als fertig rasterisierten
        RGBA-Skin auf das Gelaende - gefuellt von
        SettlementTab.apply_3d_overlays() ueber
        `rasterize_settlements_rgba()` (map_display_2d.py), also aus
        derselben Zeichenlogik wie die 2D-Ansicht.
        Aufgabe: 3D-Darstellung des globalen Siedlungsreiters (Nutzer-Vorgabe
        2026-08-13). Bewusst ein Skin und keine echte 3D-Marker-Geometrie:
        `_render_settlement_markers()` war seit jeher ein leerer TODO-Stub,
        und eigene Zylinder/Icons haetten neuen GLSL- und Geometriecode
        gebraucht - der Alpha-Overlay-Pfad steht dagegen bereits und wird
        von Plots, Regionen und Kuestentypen genauso genutzt.
        """
        rgba = self.overlay_data["settlement"].get("uebersicht")
        if rgba is None or not isinstance(rgba, np.ndarray) or rgba.ndim != 3 or rgba.shape[2] != 4:
            return
        cache_schluessel = ("settlement", "uebersicht")
        texture_id = self._overlay_cache_pruefen(cache_schluessel, id(rgba))
        if texture_id is None:
            texture_id = self._overlay_textur_hochladen(cache_schluessel, id(rgba), rgba)
        self._render_rgba_textur(texture_id)

    def _render_dict_rgba_overlay(self, tab_type, layer_name):
        """
        Funktionsweise: Wie _render_plot_boundaries(), aber fuer Overlays, die
        NICHT schon fertig rasterisiert ankommen, sondern als dasselbe rohe
        Payload-Dict, das auch der 2D-Renderer bekommt (Nutzer-Vorgabe
        2026-08-13: "die 3D darstellung ALLER 2D maps, aber vor allem der
        Kuesten auf die 3D Terrains bekommen"). Rasterisiert ueber dieselben
        Funktionen wie die 2D-Seite (`rasterize_regions_rgba()`/
        `rasterize_kuesten_archetypen_rgba()` in map_display_2d.py) - EINE
        Farblogik fuer beide Ansichten statt einer zweiten, die auseinander-
        laufen koennte.

        DIE RASTERISIERUNG SELBST WIRD GECACHT (2026-08-13, Nutzerbefund
        "Kuestentyp ruckelt im 3D") - gemessen 0.6s bei 1024px, bei jedem
        paintGL()-Aufruf (jede Mausbewegung waehrend des Drehens) neu
        gerechnet, ohne dass sich die Daten geaendert haetten. Der
        Cache-Check laeuft deshalb VOR dem Rasterisieren, nicht nur vor dem
        Hochladen - ein reiner Textur-Cache haette das eigentliche Problem
        nicht geloest.
        Aufgabe: Baut bei Bedarf die RGBA-Textur aus dem Rohdaten-Dict und
        zeichnet sie wie jeden anderen Alpha-Overlay.
        """
        payload = self.overlay_data.get(tab_type, {}).get(layer_name)
        if not isinstance(payload, dict):
            return
        cache_schluessel = (tab_type, layer_name)
        # CACHE-SCHLUESSEL AUS DEM INHALT, NICHT AUS DEM DICT.
        #
        # `id(payload)` war fuer die Regionen-/Kuesten-Overlays richtig -
        # dort wird EIN Dict gebaut und wiederverwendet. Das Flussnetz
        # baut bei jedem Aufruf ein NEUES Dict (siehe
        # overlay_river_generations), damit war `id()` jedes Mal anders,
        # der Cache griff nie, und die Textur wurde bei JEDEM Frame neu
        # gerastert und hochgeladen. Gemessener Nutzerbefund 2026-08-24:
        # *"fluesse 3d ist ziemlich langsam"*.
        #
        # Die enthaltenen Arrays sind dagegen stabil - sie kommen direkt
        # aus dem DataLODManager und werden nur weitergereicht.
        if layer_name == "river_overlay":
            daten_id = (id(payload.get("river_generation")),
                        bool(payload.get("zeige_mikro")),
                        int(payload.get("breite_px", 1)))
        else:
            daten_id = id(payload)
        texture_id = self._overlay_cache_pruefen(cache_schluessel, daten_id)

        if texture_id is None:
            hoehe = payload.get("heightmap")
            if hoehe is None:
                return

            from gui.widgets.overlay_rasterizer import (
                rasterize_regions_rgba, rasterize_kuesten_archetypen_rgba,
                rasterize_fluesse_rgba)
            # DAS FLUSSNETZ ZUERST, denn es braucht `regionen` NICHT.
            # Die Pruefung auf `regionen` steht deshalb erst darunter -
            # stuende sie oben, kaeme dieser Zweig nie zum Zug.
            if layer_name == "river_overlay":
                gen = payload.get("river_generation")
                if gen is None:
                    return
                rgba = rasterize_fluesse_rgba(
                    gen, hoehe,
                    zeige_mikro=bool(payload.get("zeige_mikro")),
                    breite_px=int(payload.get("breite_px", 1)))
                texture_id = self._overlay_textur_hochladen(
                    cache_schluessel, daten_id, rgba)
                if texture_id is not None:
                    self._render_rgba_textur(texture_id)
                return

            regionen = payload.get("regionen")
            if regionen is None:
                return
            if layer_name == "region_overlay":
                rgba = rasterize_regions_rgba(regionen, hoehe, alpha=0.55, border_alpha=0.9)
            elif layer_name == "kuesten_overlay":
                archetyp = payload.get("kuesten_archetyp")
                if archetyp is None:
                    return
                rgba = rasterize_kuesten_archetypen_rgba(
                    regionen, hoehe, archetyp, payload.get("kuesten_staerke"))
            else:
                return
            texture_id = self._overlay_textur_hochladen(cache_schluessel, daten_id, rgba)

        self._render_rgba_textur(texture_id)

    def _overlay_cache_pruefen(self, cache_schluessel, daten_id):
        """Rueckgabe: die zwischengespeicherte Textur-ID, wenn die Quelldaten
        seit dem letzten Aufbau dieselben sind (Objekt-Identitaet - siehe
        self._overlay_texture_cache in __init__), sonst None."""
        eintrag = self._overlay_texture_cache.get(cache_schluessel)
        if eintrag is not None and eintrag[0] == daten_id:
            return eintrag[1]
        return None

    def _overlay_textur_hochladen(self, cache_schluessel, daten_id, rgba):
        """Laedt `rgba` als (H,W,4)-Textur hoch, loescht eine evtl. vorhandene
        aeltere Textur desselben Cache-Schluessels und hinterlegt die neue.
        Gibt die neue Textur-ID zurueck."""
        if (rgba is None or not isinstance(rgba, np.ndarray) or rgba.ndim != 3
                or rgba.shape[2] != 4):
            return None
        alt = self._overlay_texture_cache.get(cache_schluessel)
        if alt is not None:
            gl.glDeleteTextures(1, [alt[1]])
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, rgba.shape[1], rgba.shape[0],
                         0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, np.ascontiguousarray(rgba))
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        self._overlay_texture_cache[cache_schluessel] = (daten_id, texture_id)
        return texture_id

    def _render_rgba_textur(self, texture_id):
        """
        Funktionsweise: Zeichnet eine BEREITS HOCHGELADENE (H,W,4)-RGBA-Textur
        als zweiten Draw-Call mit useAlphaOverlay=1 (siehe terrain.frag) -
        echte Transparenz an unbemalten Stellen statt des pauschalen
        overlayStrength-Mix der uebrigen Scalar-Overlays. Verallgemeinert aus
        der urspruenglich plot-spezifischen Fassung (2026-08-13), damit jeder
        weitere vorgerasterte 2D-"Skin" (Regionen, Kuestentypen, ...) denselben
        Weg ohne neuen GLSL-Code nutzen kann. Die Textur selbst bleibt ueber
        Frames hinweg bestehen (siehe _overlay_textur_hochladen()) - hier wird
        nur noch gebunden und gezeichnet, kein Upload mehr.
        Aufgabe: Gemeinsamer Zeichenpfad fuer alle RGBA-Skin-Overlays.
        """
        if texture_id is None or not self.shader_program or self.vertex_buffer is None:
            return

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
            # IMMER 5 (Settlement), unabhaengig vom aufrufenden Tab -
            # terrain.frag behandelt nur in getSettlementColor() den
            # Alpha-Kanal der Overlay-Textur (useAlphaOverlay), jeder
            # andere renderMode-Zweig (0-4) mischt nur mit fester
            # overlayStrength und wuerde transparente Bereiche der
            # RGBA-Textur trotzdem einfaerben. Ohne neuen GLSL-Code zu
            # schreiben (Projekt-Vorgeschichte mit Shader-Aenderungen)
            # ist 5 deshalb fuer JEDEN Alpha-Skin-Overlay der einzig
            # richtige Wert, nicht nur fuer Settlement-Plots.
            gl.glUniform1i(render_mode_location, 5)

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
        # Fokus holen, sonst erreichen WASD/Leertaste dieses Widget nie -
        # die Tasten landen sonst beim zuletzt angeklickten Bedienelement.
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.last_mouse_pos = event.pos()

        # AUSWAHL nur bei der LINKEN Taste und nur, wenn etwas hinterlegt ist
        # (6.29). Die rechte/mittlere Taste drehen und schieben die Kamera -
        # dort waere eine Auswahl stoerend.
        if event.button() == Qt.MouseButton.LeftButton and (
                len(self._auswahl_orte) or self._auswahl_wege):
            self._auswahl_pruefen(event.pos())

    # Grenzen der Neigung. Bei genau +-90 Grad steht die Blickrichtung parallel
    # zur Hoch-Achse (0,1,0), das Kreuzprodukt in _create_lookat_matrix() wird
    # zu null und das Bild kippt weg - deshalb kurz davor abfangen.
    ELEVATION_LIMIT_DEG = 85.0

    def mouseMoveEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Move Events
        Aufgabe: Drei Gesten, siehe unten

        LINKE Maustaste       Free Look - Azimut UND Neigung, Drehung um die
                              EIGENE Position, wie ein Spectator im Shooter
        MITTLERE Maustaste    Panning in der Bildebene
        RECHTE Maustaste      die frühere Linksklick-Geste: Azimut-Drehung UM
                              DEN BLICKPUNKT, also um das Gelände herumgehen
        """
        if self.last_mouse_pos is None:
            return

        dx = event.pos().x() - self.last_mouse_pos.x()
        dy = event.pos().y() - self.last_mouse_pos.y()

        if event.buttons() & Qt.MouseButton.MiddleButton:
            self._pan(dx, dy)

        elif event.buttons() & Qt.MouseButton.LeftButton:
            # Free Look. Maus nach unten = nach unten schauen, also groessere
            # Elevation (die Kamera steht dann steiler ueber dem Blickpunkt).
            self._rotate_in_place(dx * self.mouse_sensitivity,
                                  dy * self.mouse_sensitivity)

        elif event.buttons() & Qt.MouseButton.RightButton:
            # Alte Linksklick-Geste: der BLICKPUNKT bleibt stehen, das Auge
            # wandert um ihn herum.
            self.camera_azimuth = (
                self.camera_azimuth + dx * self.mouse_sensitivity) % 360.0
            self.camera_changed.emit(
                self.camera_elevation, self.camera_azimuth, self.camera_distance)
            self.update()

        self.last_mouse_pos = event.pos()

    def _rotate_in_place(self, delta_azimuth, delta_elevation):
        """
        Dreht die Kamera um die EIGENE Position: das Auge bleibt stehen, der
        Blickpunkt wandert.

        Das ist der Unterschied zur Rechtsklick-Geste, bei der es umgekehrt
        ist. Beides braucht denselben Trick - Auge merken, Winkel aendern,
        Blickpunkt so nachziehen, dass das Auge wieder dort landet.
        """
        offset = self._eye_offset()
        eye = [self.camera_target[axis] + offset[axis] for axis in range(3)]

        self.camera_azimuth = (self.camera_azimuth + delta_azimuth) % 360.0
        self.camera_elevation = max(
            -self.ELEVATION_LIMIT_DEG,
            min(self.ELEVATION_LIMIT_DEG, self.camera_elevation + delta_elevation))

        new_offset = self._eye_offset()
        self.camera_target = [eye[axis] - new_offset[axis] for axis in range(3)]
        self._camera_target_touched = True

        self.camera_changed.emit(
            self.camera_elevation, self.camera_azimuth, self.camera_distance)
        self.update()

    def _pan(self, dx, dy):
        """
        Verschiebt den BLICKPUNKT in der Bildebene (Shift + linke Maustaste).

        Der Versatz skaliert mit `camera_distance`: aus der Nähe soll dieselbe
        Mausbewegung fein verschieben, aus der Ferne grob - sonst ist Panning
        entweder herausgezoomt zäh oder herangezoomt unbrauchbar hektisch.
        """
        right, up = self._screen_axes()
        self._camera_target_touched = True
        scale = self.camera_distance * 0.0022
        for axis in range(3):
            # dy invertiert: Bildschirm-y zeigt nach unten, die Welt-Hochachse
            # nach oben. Ohne das zieht die Karte in die falsche Richtung.
            self.camera_target[axis] += (-dx * right[axis] + dy * up[axis]) * scale

        self.camera_changed.emit(self.camera_elevation, self.camera_azimuth, self.camera_distance)
        self.update()

    # --- Flugmodus -----------------------------------------------------------
    #
    # W/S  vorwaerts/rueckwaerts entlang der echten Blickrichtung
    # A/D  seitwaerts strafen
    # Leer aufsteigen, Shift absinken - beides orthogonal zur Blickrichtung
    #
    # Gedreht wird ausschliesslich mit der Maus: links um die EIGENE Position
    # (Free Look), rechts um den BLICKPUNKT (die alte Geste). Dass beides
    # nebeneinander moeglich ist, ist der Grund, warum der Blickpunkt
    # ueberhaupt eigener Zustand werden musste.
    _FLIGHT_KEYS = frozenset({
        Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D,
        Qt.Key.Key_Space, Qt.Key.Key_Shift,
    })
    FLIGHT_TICK_MS = 16  # ~60 Hz

    def keyPressEvent(self, event):
        if event.key() in self._FLIGHT_KEYS and not event.isAutoRepeat():
            self._pressed_keys.add(event.key())
            if not self.flight_timer.isActive():
                self.flight_timer.start(self.FLIGHT_TICK_MS)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() in self._FLIGHT_KEYS and not event.isAutoRepeat():
            self._pressed_keys.discard(event.key())
            if not self._pressed_keys:
                self.flight_timer.stop()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        # Ohne das fliegt die Kamera weiter, wenn das Fenster den Fokus
        # verliert waehrend eine Taste gedrueckt ist - das keyReleaseEvent
        # kommt dann nie an.
        self._pressed_keys.clear()
        self.flight_timer.stop()
        # Projektion mitziehen - sie haengt am Kameraabstand, siehe
        # _update_projection_matrix().
        self._update_projection_matrix()
        super().focusOutEvent(event)

    def _advance_flight(self):
        """Ein Zeitschritt des Flugmodus, aus den gedrueckten Tasten."""
        if not self._pressed_keys:
            self.flight_timer.stop()
            return

        seconds = self.FLIGHT_TICK_MS / 1000.0
        self._camera_target_touched = True
        keys = self._pressed_keys
        moved = False

        forward_input = ((Qt.Key.Key_W in keys) - (Qt.Key.Key_S in keys))
        if forward_input:
            forward = self._forward()
            step = forward_input * self.flight_speed * seconds
            for axis in range(3):
                self.camera_target[axis] += forward[axis] * step
            moved = True

        # Shift ist reiner Abwaerts-Schub - das Panning haengt seit
        # 2026-07-28 an der mittleren Maustaste, nicht mehr an Shift.
        vertical_input = ((Qt.Key.Key_Space in keys) - (Qt.Key.Key_Shift in keys))
        if vertical_input:
            _, up = self._screen_axes()
            step = vertical_input * self.flight_speed * seconds
            for axis in range(3):
                self.camera_target[axis] += up[axis] * step
            moved = True

        # A/D STRAFEN seitwaerts, sie drehen nicht. Gedreht wird ausschliesslich
        # mit der Maus (links: um die eigene Position, rechts: um den
        # Blickpunkt) - so wie in einem Shooter.
        strafe_input = ((Qt.Key.Key_D in keys) - (Qt.Key.Key_A in keys))
        if strafe_input:
            # `right` aus _screen_axes() traegt SCREEN_RIGHT_SIGN bereits und
            # ist gegen die echte View-Matrix belegt (siehe Panning-Test).
            right, _ = self._screen_axes()
            step = strafe_input * self.flight_speed * seconds
            for axis in range(3):
                self.camera_target[axis] += right[axis] * step
            moved = True

        if moved:
            self.camera_changed.emit(
                self.camera_elevation, self.camera_azimuth, self.camera_distance)
            self.update()

    def wheelEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Wheel Events für Zoom-Funktionalität
        Aufgabe: Implementiert Zoom mit Distanz-Begrenzungen
        """
        if event.angleDelta().y() == 0:
            return

        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        new_distance = self.camera_distance * zoom_factor

        # Zoom-Grenzen. Obergrenze 2026-07-28 von 50 auf 400 angehoben: seit
        # der Blickpunkt beweglich ist, kann man weit von der Karte wegfliegen,
        # und 50 Einheiten (5 Kartenbreiten) waren dafuer zu eng. Die
        # Untergrenze bleibt - naeher als 2 Einheiten schneidet die Near-Plane
        # ins Gelaende.
        min_distance = 2.0
        max_distance = 400.0
        self.camera_distance = max(min_distance, min(max_distance, new_distance))

        # Die Projektion haengt jetzt am Kameraabstand (siehe
        # _update_projection_matrix) und muss deshalb hier mitgezogen werden.
        self._update_projection_matrix()

        self.camera_changed.emit(self.camera_elevation, self.camera_azimuth, self.camera_distance)
        self.update()

    def reset_camera(self):
        """
        Funktionsweise: Setzt Camera auf Standard-Position zurück
        Aufgabe: Reset zu Default-View aus gui_default.py
        """
        self.camera_distance = CanvasSettings.CANVAS_3D["camera_distance"]
        self.camera_elevation = 55.0
        self.camera_azimuth = 180.0  # siehe __init__-Kommentar
        # Blickpunkt zurueck auf die Kartenmitte - das ist die Rettungsleine,
        # wenn man sich im Flugmodus verirrt hat.
        self.camera_target = [0.0, self.terrain_center_y, 0.0]
        self._camera_target_touched = False
        self._pressed_keys.clear()
        self.flight_timer.stop()

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

        # Camera-Reset Button (immer sichtbar) - die Rettungsleine, wenn man
        # sich im Flugmodus verirrt hat.
        #
        # NoFocus wie beim [GENERIEREN]-Knopf: ein fokussierter QPushButton
        # verschluckt die Leertaste, und die gehoert im Flugmodus der Kamera
        # (siehe MapDisplay3D.keyPressEvent).
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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

        # Die frühere "Shadows"-Checkbox hier wurde auf Nutzer-Wunsch entfernt
        # (UI-Aufräumung Teil 2) - Schatten sind jetzt dauerhaft aktiv
        # (MapDisplay3D.shadows_enabled bleibt bei seinem Default True).

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
        elif tab_type == "erosion":
            # Erosion nutzt dieselben Bedienelemente wie Water (Gelaende +
            # genau ein Skalarfeld-Overlay).
            self._setup_water_controls()
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

    def set_sun_direction(self, elevation_deg, azimuth_deg, distance=15.0):
        """
        Funktionsweise: Delegiert Sonnenstand-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Licht-Richtungs-Updates, siehe
        MapDisplay3D.set_sun_direction()
        """
        self.display_3d.set_sun_direction(elevation_deg, azimuth_deg, distance)

    def update_overlay_data(self, tab_type, layer_name, data):
        """
        Funktionsweise: Delegiert Overlay-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Overlay-Updates
        """
        self.display_3d.update_overlay_data(tab_type, layer_name, data)

    def clear_river_overlay(self):
        """Das Flussnetz abschalten - Weiterleitung an die GL-Anzeige."""
        self.display_3d.clear_river_overlay()

    def overlay_river_generations(self, generation_map, zeige_mikro=False):
        """
        Das Flussnetz - Weiterleitung an die GL-Anzeige.

        DIESE WEITERLEITUNG IST DER PUNKT, an dem es beim ersten Anlauf
        scheiterte. `gui/tabs/base_tab.py` legt das 3D-Widget in einen
        `DisplayWrapper`, und `BaseMapTab._push_overlays()` (frueher:
        `river_tab._anzeigeziel()`) greift ueber `.display` darauf zu - das
        ist DIESE Klasse, nicht die innere `MapDisplay3D`. Die Methode allein
        in der GL-Klasse zu haben genuegt also nicht; `hasattr` schlaegt hier
        fehl und der Aufruf faellt lautlos aus - genau der Fehler, der
        behoben werden sollte.
        """
        self.display_3d.overlay_river_generations(generation_map, zeige_mikro)

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

    def set_world_size_km(self, world_size_km: float):
        """
        Funktionsweise: Delegiert die reale Kartenausdehnung (km) an das
        3D-Display (siehe [[project-terrain-review]] 4f).
        Aufgabe: Interface-Methode für base_tab.py, analog zu
        set_water_biomes_reference() oben.
        """
        self.display_3d.set_world_size_km(world_size_km)