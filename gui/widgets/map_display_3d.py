"""
Path: gui/widgets/map_display_3d.py

Funktionsweise: 3D Terrain-Rendering mit OpenGL
- Real-time 3D Heightmap-Mesh Rendering
- Camera-Controls (Orbit, Zoom, Pan)
- Multi-Layer Support (Terrain + Settlements + Water + Wind-Vectors)
- Lighting und Shading für realistisches Appearance

Kommunikationskanäle:
- Input: heightmaps und Overlay-Data von data_manager
- GPU: shader_manager für optimierte Rendering-Pipeline
- Controls: Camera-Settings aus gui_default.py
"""

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QSlider, QLabel
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtOpenGL import QOpenGLWidget
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
from OpenGL.GL import shaders
from gui.config.gui_default import CanvasSettings


class MapDisplay3D(QOpenGLWidget):
    """
    Funktionsweise: 3D-Visualisierung von Heightmaps mit OpenGL-Rendering
    Aufgabe: Real-time 3D Terrain-Darstellung mit Camera-Controls und Multi-Layer Support
    """

    # Signals für 3D-Interaktion
    camera_changed = pyqtSignal(float, float, float)  # (rotation_x, rotation_y, zoom)
    vertex_selected = pyqtSignal(int, int)  # (x, y)

    def __init__(self, parent=None):
        """
        Funktionsweise: Initialisiert 3D OpenGL-Widget mit Standard-Camera-Position
        Aufgabe: Setup von OpenGL-Context und Standard-Rendering-Parameter
        """
        super().__init__(parent)

        # Camera-Parameter aus gui_default.py
        self.camera_distance = CanvasSettings.CANVAS_3D["camera_distance"]
        self.camera_rotation_x = -60.0
        self.camera_rotation_y = 0.0
        self.fov = CanvasSettings.CANVAS_3D["fov"]

        # Rendering-Daten
        self.heightmap = None
        self.mesh_vertices = None
        self.mesh_indices = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.shader_program = None

        # Layer-Visibility
        self.show_terrain = True
        self.show_water = True
        self.show_settlements = True
        self.show_wind_vectors = False

        # Overlay-Daten
        self.water_map = None
        self.settlement_positions = []
        self.wind_vectors = None

        # Mouse-Interaction
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5

        # Animation-Timer für Wind-Vectors
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_time = 0.0

    def initializeGL(self):
        """
        Funktionsweise: OpenGL-Initialisierung beim ersten Aufruf
        Aufgabe: Setup von OpenGL-State, Shaders und Rendering-Pipeline
        """
        # OpenGL-Settings
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        # Background-Color aus gui_default.py
        bg_color = CanvasSettings.CANVAS_3D["background_color"]
        gl.glClearColor(*bg_color)

        # Shader-Programm laden
        self._load_shaders()

        # Lighting-Setup
        self._setup_lighting()

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

        if self.shader_program is None:
            return

        # Shader aktivieren
        gl.glUseProgram(self.shader_program)

        # View-Matrix setzen
        self._update_view_matrix()

        # Layer in korrekter Reihenfolge rendern
        if self.show_terrain and self.heightmap is not None:
            self._render_terrain()

        if self.show_water and self.water_map is not None:
            self._render_water()

        if self.show_settlements and self.settlement_positions:
            self._render_settlements()

        if self.show_wind_vectors and self.wind_vectors is not None:
            self._render_wind_vectors()

    def update_heightmap(self, heightmap):
        """
        Funktionsweise: Aktualisiert Heightmap und regeneriert Terrain-Mesh
        Aufgabe: Konvertiert 2D-Heightmap zu 3D-Mesh für OpenGL-Rendering
        Parameter: heightmap (numpy.ndarray) - Neue Höhendaten
        """
        self.heightmap = heightmap
        if heightmap is not None:
            self._generate_terrain_mesh()
            self.update()

    def update_water_map(self, water_map):
        """
        Funktionsweise: Aktualisiert Water-Layer für Overlay-Rendering
        Aufgabe: Setzt neue Wasser-Daten für 3D-Darstellung
        Parameter: water_map (numpy.ndarray) - Wasser-Tiefendaten
        """
        self.water_map = water_map
        self.update()

    def update_settlements(self, settlement_positions):
        """
        Funktionsweise: Aktualisiert Settlement-Marker für 3D-Darstellung
        Aufgabe: Setzt neue Settlement-Positionen für Marker-Rendering
        Parameter: settlement_positions (list) - Liste von (x, y, z) Positionen
        """
        self.settlement_positions = settlement_positions
        self.update()

    def update_wind_vectors(self, wind_vectors):
        """
        Funktionsweise: Aktualisiert Wind-Vector-Field für Animation
        Aufgabe: Setzt neue Wind-Daten und startet Animation
        Parameter: wind_vectors (numpy.ndarray) - Wind-Vektoren (vx, vy) pro Pixel
        """
        self.wind_vectors = wind_vectors
        if wind_vectors is not None and self.show_wind_vectors:
            self.animation_timer.start(50)  # 20 FPS Animation
        else:
            self.animation_timer.stop()
        self.update()

    def _load_shaders(self):
        """
        Funktionsweise: Lädt und kompiliert Vertex- und Fragment-Shader
        Aufgabe: Erstellt Shader-Programm für 3D-Terrain-Rendering
        """
        vertex_shader_code = """
        #version 330 core
        layout (location = 0) in vec3 position;
        layout (location = 1) in vec3 normal;
        layout (location = 2) in vec2 texCoord;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 lightPos;

        out vec3 FragPos;
        out vec3 Normal;
        out vec2 TexCoord;
        out vec3 LightPos;

        void main() {
            FragPos = vec3(model * vec4(position, 1.0));
            Normal = mat3(transpose(inverse(model))) * normal;
            TexCoord = texCoord;
            LightPos = lightPos;

            gl_Position = projection * view * vec4(FragPos, 1.0);
        }
        """

        fragment_shader_code = """
        #version 330 core
        out vec4 FragColor;

        in vec3 FragPos;
        in vec3 Normal;
        in vec2 TexCoord;
        in vec3 LightPos;

        uniform float heightScale;
        uniform float maxHeight;
        uniform bool useHeightColors;

        void main() {
            // Terrain-Coloring basierend auf Höhe
            vec3 color;
            if (useHeightColors) {
                float height = FragPos.y / maxHeight;
                if (height < 0.3) {
                    color = mix(vec3(0.2, 0.4, 0.8), vec3(0.8, 0.7, 0.4), height / 0.3);
                } else if (height < 0.7) {
                    color = mix(vec3(0.8, 0.7, 0.4), vec3(0.2, 0.6, 0.2), (height - 0.3) / 0.4);
                } else {
                    color = mix(vec3(0.2, 0.6, 0.2), vec3(0.9, 0.9, 0.9), (height - 0.7) / 0.3);
                }
            } else {
                color = vec3(0.6, 0.6, 0.6);
            }

            // Phong-Lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(LightPos - FragPos);

            // Ambient
            vec3 ambient = 0.3 * color;

            // Diffuse
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * color;

            // Specular
            vec3 viewDir = normalize(-FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = 0.3 * spec * vec3(1.0, 1.0, 1.0);

            vec3 result = ambient + diffuse + specular;
            FragColor = vec4(result, 1.0);
        }
        """

        try:
            vertex_shader = shaders.compileShader(vertex_shader_code, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_shader_code, gl.GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
        except Exception as e:
            print(f"Shader compilation failed: {e}")
            self.shader_program = None

    def _setup_lighting(self):
        """
        Funktionsweise: Konfiguriert Lighting-Parameter für realistisches Shading
        Aufgabe: Setzt Light-Position aus gui_default.py und Standard-Material-Properties
        """
        if self.shader_program:
            gl.glUseProgram(self.shader_program)

            # Light-Position aus gui_default.py
            light_pos = CanvasSettings.CANVAS_3D["light_position"]
            light_pos_location = gl.glGetUniformLocation(self.shader_program, "lightPos")
            gl.glUniform3f(light_pos_location, *light_pos)

    def _generate_terrain_mesh(self):
        """
        Funktionsweise: Generiert 3D-Mesh aus 2D-Heightmap
        Aufgabe: Erstellt Vertices, Normals und Indices für Terrain-Rendering
        """
        if self.heightmap is None:
            return

        height, width = self.heightmap.shape
        vertices = []
        indices = []

        # Vertices und Texture-Coordinates generieren
        for y in range(height):
            for x in range(width):
                # Position (x, height, z)
                pos_x = (x / (width - 1)) * 10.0 - 5.0  # -5 bis +5
                pos_y = self.heightmap[y, x] * 0.01  # Scale height
                pos_z = (y / (height - 1)) * 10.0 - 5.0  # -5 bis +5

                # Normal berechnen (simplified)
                normal_x = 0.0
                normal_y = 1.0
                normal_z = 0.0

                if x > 0 and x < width - 1:
                    normal_x = (self.heightmap[y, x - 1] - self.heightmap[y, x + 1]) * 0.1
                if y > 0 and y < height - 1:
                    normal_z = (self.heightmap[y - 1, x] - self.heightmap[y + 1, x]) * 0.1

                # Normalisieren
                length = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
                if length > 0:
                    normal_x /= length
                    normal_y /= length
                    normal_z /= length

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

        # VBOs erstellen
        self.mesh_vertices = np.array(vertices, dtype=np.float32)
        self.mesh_indices = np.array(indices, dtype=np.uint32)

        # OpenGL-Buffers aktualisieren
        if self.vertex_buffer:
            self.vertex_buffer.delete()
        if self.index_buffer:
            gl.glDeleteBuffers(1, [self.index_buffer])

        self.vertex_buffer = glvbo.VBO(self.mesh_vertices)

        self.index_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.mesh_indices.nbytes, self.mesh_indices, gl.GL_STATIC_DRAW)

    def _render_terrain(self):
        """
        Funktionsweise: Rendert Terrain-Mesh mit Shading und Texturing
        Aufgabe: OpenGL-Rendering des generierten Terrain-Meshes
        """
        if self.vertex_buffer is None or self.mesh_indices is None:
            return

        gl.glUseProgram(self.shader_program)

        # Shader-Parameter setzen
        height_scale_location = gl.glGetUniformLocation(self.shader_program, "heightScale")
        max_height_location = gl.glGetUniformLocation(self.shader_program, "maxHeight")
        use_height_colors_location = gl.glGetUniformLocation(self.shader_program, "useHeightColors")

        gl.glUniform1f(height_scale_location, 0.01)
        gl.glUniform1f(max_height_location, np.max(self.heightmap) * 0.01)
        gl.glUniform1i(use_height_colors_location, 1)

        # Vertex-Daten binden
        self.vertex_buffer.bind()

        # Vertex-Attribute setzen
        stride = 8 * 4  # 8 floats * 4 bytes per float

        # Position (location = 0)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)

        # Normal (location = 1)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(3 * 4))

        # Texture-Coords (location = 2)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.GLvoidp(6 * 4))

        # Index-Buffer binden und rendern
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.mesh_indices), gl.GL_UNSIGNED_INT, None)

        # Cleanup
        gl.glDisableVertexAttribArray(0)
        gl.glDisableVertexAttribArray(1)
        gl.glDisableVertexAttribArray(2)
        self.vertex_buffer.unbind()

    def _render_water(self):
        """
        Funktionsweise: Rendert Wasser-Layer als transparente Ebene über Terrain
        Aufgabe: Spezielle Darstellung für Wasser-Bereiche mit Transparenz
        """
        if self.water_map is None:
            return

        # Simplified water rendering - würde echte Wasser-Shader benötigen
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Wasser-spezifische Rendering-Logik hier

        gl.glDisable(gl.GL_BLEND)

    def _render_settlements(self):
        """
        Funktionsweise: Rendert Settlement-Marker als 3D-Objekte über Terrain
        Aufgabe: Visualisierung von Settlement-Positionen mit 3D-Markern
        """
        if not self.settlement_positions:
            return

        # Simplified settlement rendering - würde Icon/Mesh-Rendering benötigen
        gl.glPointSize(8.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glColor3f(1.0, 0.0, 0.0)  # Rot für Settlements

        for pos in self.settlement_positions:
            gl.glVertex3f(pos[0], pos[1], pos[2])

        gl.glEnd()

    def _render_wind_vectors(self):
        """
        Funktionsweise: Rendert animierte Wind-Vektoren als Linien über Terrain
        Aufgabe: Visualisierung von Wind-Flow mit animierten Vektor-Feldern
        """
        if self.wind_vectors is None:
            return

        # Simplified wind vector rendering
        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(0.0, 1.0, 0.0)  # Grün für Wind-Vektoren

        height, width = self.wind_vectors.shape[:2]
        step = max(1, min(width, height) // 20)  # Nur jeden N-ten Vektor anzeigen

        for y in range(0, height, step):
            for x in range(0, width, step):
                # Start-Position
                start_x = (x / (width - 1)) * 10.0 - 5.0
                start_z = (y / (height - 1)) * 10.0 - 5.0
                start_y = 0.0
                if self.heightmap is not None:
                    start_y = self.heightmap[y, x] * 0.01 + 0.5

                # End-Position basierend auf Wind-Vektor
                wind_x = self.wind_vectors[y, x, 0] * 0.5
                wind_z = self.wind_vectors[y, x, 1] * 0.5

                # Animation-Offset
                anim_offset = np.sin(self.animation_time + x * 0.1 + y * 0.1) * 0.1

                gl.glVertex3f(start_x, start_y + anim_offset, start_z)
                gl.glVertex3f(start_x + wind_x, start_y + wind_z + anim_offset, start_z + wind_z)

        gl.glEnd()

    def _update_projection_matrix(self):
        """
        Funktionsweise: Aktualisiert Projection-Matrix bei Fenster-Resize
        Aufgabe: Setzt Perspective-Projection mit FOV aus gui_default.py
        """
        if self.shader_program is None:
            return

        gl.glUseProgram(self.shader_program)

        aspect_ratio = self.width() / self.height() if self.height() > 0 else 1.0

        # Perspective-Matrix berechnen (simplified)
        projection_location = gl.glGetUniformLocation(self.shader_program, "projection")
        if projection_location >= 0:
            # Perspective-Matrix setzen (würde echte Matrix-Berechnung benötigen)
            pass

    def _update_view_matrix(self):
        """
        Funktionsweise: Aktualisiert View-Matrix basierend auf Camera-Position
        Aufgabe: Setzt Camera-Transform für Orbit-Controls
        """
        if self.shader_program is None:
            return

        # View-Matrix basierend auf Camera-Parametern berechnen
        view_location = gl.glGetUniformLocation(self.shader_program, "view")
        model_location = gl.glGetUniformLocation(self.shader_program, "model")

        if view_location >= 0 and model_location >= 0:
            # Matrix-Berechnungen (würde echte Matrix-Math benötigen)
            pass

    def _update_animation(self):
        """
        Funktionsweise: Aktualisiert Animation-Zeit für Wind-Vector-Animation
        Aufgabe: Inkrementiert Animation-Timer für flüssige Wind-Animation
        """
        self.animation_time += 0.1
        if self.show_wind_vectors:
            self.update()

    def mousePressEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Press Events
        Aufgabe: Startet Camera-Rotation oder andere Interaktionen
        """
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Move Events für Camera-Controls
        Aufgabe: Implementiert Orbit-Controls durch Mouse-Drag
        """
        if self.last_mouse_pos is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        if event.buttons() & Qt.LeftButton:
            # Orbit-Rotation
            self.camera_rotation_y += dx * self.mouse_sensitivity
            self.camera_rotation_x += dy * self.mouse_sensitivity

            # Begrenze X-Rotation
            self.camera_rotation_x = max(-90, min(90, self.camera_rotation_x))

            self.camera_changed.emit(self.camera_rotation_x, self.camera_rotation_y, self.camera_distance)
            self.update()

        self.last_mouse_pos = event.pos()

    def wheelEvent(self, event):
        """
        Funktionsweise: Handler für Mouse-Wheel Events für Zoom-Funktionalität
        Aufgabe: Implementiert Zoom-in/Zoom-out mit Mausrad
        """
        zoom_factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
        self.camera_distance *= zoom_factor

        # Begrenze Zoom-Bereich
        self.camera_distance = max(1.0, min(20.0, self.camera_distance))

        self.camera_changed.emit(self.camera_rotation_x, self.camera_rotation_y, self.camera_distance)
        self.update()

    def set_layer_visibility(self, layer_name, visible):
        """
        Funktionsweise: Schaltet Sichtbarkeit einzelner Render-Layer ein/aus
        Aufgabe: Toggle-Funktionalität für verschiedene Visualisierungs-Layer
        Parameter: layer_name (str), visible (bool) - Layer-Name und Sichtbarkeits-Status
        """
        if layer_name == "terrain":
            self.show_terrain = visible
        elif layer_name == "water":
            self.show_water = visible
        elif layer_name == "settlements":
            self.show_settlements = visible
        elif layer_name == "wind_vectors":
            self.show_wind_vectors = visible
            if visible and self.wind_vectors is not None:
                self.animation_timer.start(50)
            else:
                self.animation_timer.stop()

        self.update()

    def reset_camera(self):
        """
        Funktionsweise: Setzt Camera auf Standard-Position zurück
        Aufgabe: Reset zu Default-View aus gui_default.py
        """
        self.camera_distance = CanvasSettings.CANVAS_3D["camera_distance"]
        self.camera_rotation_x = -60.0
        self.camera_rotation_y = 0.0

        self.camera_changed.emit(self.camera_rotation_x, self.camera_rotation_y, self.camera_distance)
        self.update()


class MapDisplay3DWidget(QWidget):
    """
    Funktionsweise: Wrapper-Widget für 3D-Display mit Controls
    Aufgabe: Kombiniert 3D-Display mit Layer-Controls und Camera-Settings
    """

    def __init__(self, parent=None):
        """
        Funktionsweise: Initialisiert 3D-Widget mit Control-Panel
        Aufgabe: Setup von 3D-Display und zugehörigen UI-Controls
        """
        super().__init__(parent)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """
        Funktionsweise: Erstellt UI-Layout mit 3D-Display und Controls
        Aufgabe: Layout-Setup für 3D-Rendering mit Layer-Controls
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 3D-Display
        self.display_3d = MapDisplay3D()
        layout.addWidget(self.display_3d)

        # Control-Panel
        control_layout = QHBoxLayout()

        # Layer-Visibility Controls
        self.terrain_checkbox = QCheckBox("Terrain")
        self.terrain_checkbox.setChecked(True)
        control_layout.addWidget(self.terrain_checkbox)

        self.water_checkbox = QCheckBox("Water")
        self.water_checkbox.setChecked(True)
        control_layout.addWidget(self.water_checkbox)

        self.settlements_checkbox = QCheckBox("Settlements")
        self.settlements_checkbox.setChecked(True)
        control_layout.addWidget(self.settlements_checkbox)

        self.wind_checkbox = QCheckBox("Wind Vectors")
        self.wind_checkbox.setChecked(False)
        control_layout.addWidget(self.wind_checkbox)

        # Camera-Reset Button
        self.reset_camera_button = QPushButton("Reset Camera")
        control_layout.addWidget(self.reset_camera_button)

        control_layout.addStretch()
        layout.addLayout(control_layout)

    def _connect_signals(self):
        """
        Funktionsweise: Verbindet UI-Controls mit 3D-Display-Funktionen
        Aufgabe: Signal-Routing zwischen Controls und 3D-Rendering
        """
        self.terrain_checkbox.toggled.connect(lambda checked: self.display_3d.set_layer_visibility("terrain", checked))
        self.water_checkbox.toggled.connect(lambda checked: self.display_3d.set_layer_visibility("water", checked))
        self.settlements_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("settlements", checked))
        self.wind_checkbox.toggled.connect(
            lambda checked: self.display_3d.set_layer_visibility("wind_vectors", checked))

        self.reset_camera_button.clicked.connect(self.display_3d.reset_camera)

    def update_heightmap(self, heightmap):
        """
        Funktionsweise: Delegiert Heightmap-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Heightmap-Updates
        """
        self.display_3d.update_heightmap(heightmap)

    def update_water_map(self, water_map):
        """
        Funktionsweise: Delegiert Water-Map-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Water-Updates
        """
        self.display_3d.update_water_map(water_map)

    def update_settlements(self, settlement_positions):
        """
        Funktionsweise: Delegiert Settlement-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Settlement-Updates
        """
        self.display_3d.update_settlements(settlement_positions)

    def update_wind_vectors(self, wind_vectors):
        """
        Funktionsweise: Delegiert Wind-Vector-Update an 3D-Display
        Aufgabe: Interface-Methode für externe Wind-Updates
        """
        self.display_3d.update_wind_vectors(wind_vectors)