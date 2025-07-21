"""
Path: gui/managers/shader_manager.py

Funktionsweise: GPU-Compute Management für Performance-kritische Operationen
- OpenGL Compute Shader für Parallel-Processing
- Fallback auf CPU für Systeme ohne GPU-Support
- Optimierte Operationen: Noise-Generation, Erosion, Biome-Blending
- Memory-Management zwischen GPU und CPU

Einsatzgebiete:
- Terrain: Multi-Octave Simplex-Noise parallel
- Water: Hydraulic Erosion simulation
- Weather: Wind-Field calculations
- Biome: Vegetation distribution

Kommunikationskanäle:
- Input: Large numpy arrays für GPU-Processing
- Output: Processed arrays zurück an Data-Manager
"""

import numpy as np
from PyQt5.QtOpenGL import QOpenGLContext, QOpenGLShaderProgram, QOpenGLShader
from PyQt5.QtCore import QObject, pyqtSignal
import OpenGL.GL as gl

def get_shader_manager_error_decorators():
    """
    Funktionsweise: Lazy Loading von Shader Manager Error Decorators
    Aufgabe: Lädt GPU-Shader und Memory-Critical Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import gpu_shader_handler, initialization_handler
        return gpu_shader_handler, initialization_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator

gpu_shader_handler, initialization_handler = get_shader_manager_error_decorators()

class ShaderManager(QObject):
    """
    Funktionsweise: Verwaltet OpenGL Compute Shader für GPU-beschleunigte Berechnungen
    Aufgabe: Koordiniert GPU-Processing mit CPU-Fallback für alle Generator-Module
    """

    # Signals für Shader-Operations
    processing_started = pyqtSignal(str)  # (operation_name)
    processing_finished = pyqtSignal(str, bool)  # (operation_name, success)
    gpu_fallback_triggered = pyqtSignal(str)  # (reason)

    def __init__(self):
        """
        Funktionsweise: Initialisiert Shader-Manager und prüft GPU-Verfügbarkeit
        Aufgabe: Erkennt OpenGL-Support und lädt Standard-Shader
        """
        super().__init__()
        self.gpu_available = False
        self.context = None
        self.shaders = {}
        self.textures = {}

        self._check_gpu_support()
        if self.gpu_available:
            self._load_core_shaders()

    def _check_gpu_support(self):
        """
        Funktionsweise: Prüft ob GPU-Compute verfügbar ist
        Aufgabe: Erkennt OpenGL Compute Shader Support und setzt gpu_available Flag
        """
        try:
            self.context = QOpenGLContext.currentContext()
            if self.context and self.context.isValid():
                # Prüfe OpenGL Version für Compute Shader Support (4.3+)
                version_string = gl.glGetString(gl.GL_VERSION).decode()
                major, minor = map(int, version_string.split()[0].split('.'))

                if major > 4 or (major == 4 and minor >= 3):
                    self.gpu_available = True
                else:
                    self.gpu_fallback_triggered.emit("OpenGL 4.3+ required for Compute Shaders")
            else:
                self.gpu_fallback_triggered.emit("No valid OpenGL context")
        except Exception as e:
            self.gpu_fallback_triggered.emit(f"GPU check failed: {str(e)}")

    def _load_core_shaders(self):
        """
        Funktionsweise: Lädt die Kern-Shader für alle Generator-Module
        Aufgabe: Initialisiert Compute Shaders für Noise, Erosion und Biome-Processing
        """
        core_shaders = {
            "noise_generation": self._create_noise_shader(),
            "erosion_simulation": self._create_erosion_shader(),
            "biome_blending": self._create_biome_shader()
        }

        for name, shader_code in core_shaders.items():
            if self._compile_shader(name, shader_code):
                print(f"Shader '{name}' loaded successfully")
            else:
                print(f"Failed to load shader '{name}'")

    def process_noise_generation(self, size, octaves, frequency, persistence, lacunarity, seed):
        """
        Funktionsweise: GPU-beschleunigte Multi-Octave Simplex-Noise Generierung
        Aufgabe: Parallel Processing für Terrain-Noise mit allen Octaves
        Parameter: size, octaves, frequency, persistence, lacunarity, seed - Noise-Parameter
        Returns: numpy array mit generiertem Noise oder None bei Fehler
        """
        if not self.gpu_available:
            return self._cpu_fallback_noise(size, octaves, frequency, persistence, lacunarity, seed)

        self.processing_started.emit("noise_generation")

        try:
            # GPU-Buffer für Output erstellen
            output_texture = self._create_texture_2d(size, size, gl.GL_R32F)

            # Shader-Parameter setzen
            shader = self.shaders.get("noise_generation")
            if not shader:
                raise Exception("Noise shader not available")

            shader.bind()
            shader.setUniformValue("u_size", size)
            shader.setUniformValue("u_octaves", octaves)
            shader.setUniformValue("u_frequency", frequency)
            shader.setUniformValue("u_persistence", persistence)
            shader.setUniformValue("u_lacunarity", lacunarity)
            shader.setUniformValue("u_seed", seed)

            # Compute Shader ausführen
            work_groups_x = (size + 15) // 16  # 16x16 Work-Group Size
            work_groups_y = (size + 15) // 16
            gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # Ergebnis von GPU lesen
            result = self._read_texture_data(output_texture, size, size)

            self.processing_finished.emit("noise_generation", True)
            return result

        except Exception as e:
            self.processing_finished.emit("noise_generation", False)
            return self._cpu_fallback_noise(size, octaves, frequency, persistence, lacunarity, seed)

    def process_erosion_simulation(self, heightmap, water_map, velocity_map, sediment_map, dt):
        """
        Funktionsweise: GPU-beschleunigte Hydraulic Erosion Simulation
        Aufgabe: Parallel Processing der Erosions-Gleichungen für alle Pixel
        Parameter: heightmap, water_map, velocity_map, sediment_map, dt - Erosions-Daten
        Returns: Tuple (new_heightmap, new_water_map, new_velocity_map, new_sediment_map)
        """
        if not self.gpu_available:
            return self._cpu_fallback_erosion(heightmap, water_map, velocity_map, sediment_map, dt)

        self.processing_started.emit("erosion_simulation")

        try:
            size = heightmap.shape[0]

            # Input-Texturen erstellen
            height_tex = self._upload_texture_2d(heightmap, gl.GL_R32F)
            water_tex = self._upload_texture_2d(water_map, gl.GL_R32F)
            velocity_tex = self._upload_texture_2d(velocity_map, gl.GL_RG32F)
            sediment_tex = self._upload_texture_2d(sediment_map, gl.GL_R32F)

            # Output-Texturen erstellen
            new_height_tex = self._create_texture_2d(size, size, gl.GL_R32F)
            new_water_tex = self._create_texture_2d(size, size, gl.GL_R32F)
            new_velocity_tex = self._create_texture_2d(size, size, gl.GL_RG32F)
            new_sediment_tex = self._create_texture_2d(size, size, gl.GL_R32F)

            # Shader-Parameter setzen
            shader = self.shaders.get("erosion_simulation")
            shader.bind()
            shader.setUniformValue("u_size", size)
            shader.setUniformValue("u_dt", dt)

            # Compute Shader ausführen
            work_groups = (size + 15) // 16
            gl.glDispatchCompute(work_groups, work_groups, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # Ergebnisse lesen
            new_heightmap = self._read_texture_data(new_height_tex, size, size)
            new_water_map = self._read_texture_data(new_water_tex, size, size)
            new_velocity_map = self._read_texture_data(new_velocity_tex, size, size, channels=2)
            new_sediment_map = self._read_texture_data(new_sediment_tex, size, size)

            self.processing_finished.emit("erosion_simulation", True)
            return (new_heightmap, new_water_map, new_velocity_map, new_sediment_map)

        except Exception as e:
            self.processing_finished.emit("erosion_simulation", False)
            return self._cpu_fallback_erosion(heightmap, water_map, velocity_map, sediment_map, dt)

    def process_biome_blending(self, temperature_map, precipitation_map, elevation_map, moisture_map):
        """
        Funktionsweise: GPU-beschleunigte Biome-Klassifikation und Blending
        Aufgabe: Parallel Processing der Biome-Zuordnung für alle Pixel
        Parameter: temperature_map, precipitation_map, elevation_map, moisture_map - Klima-Daten
        Returns: numpy array mit Biome-Indices oder None bei Fehler
        """
        if not self.gpu_available:
            return self._cpu_fallback_biome_blending(temperature_map, precipitation_map, elevation_map, moisture_map)

        self.processing_started.emit("biome_blending")

        try:
            size = temperature_map.shape[0]

            # Input-Texturen uploaden
            temp_tex = self._upload_texture_2d(temperature_map, gl.GL_R32F)
            precip_tex = self._upload_texture_2d(precipitation_map, gl.GL_R32F)
            elev_tex = self._upload_texture_2d(elevation_map, gl.GL_R32F)
            moist_tex = self._upload_texture_2d(moisture_map, gl.GL_R32F)

            # Output-Texture erstellen
            biome_tex = self._create_texture_2d(size, size, gl.GL_R8UI)

            # Shader ausführen
            shader = self.shaders.get("biome_blending")
            shader.bind()
            shader.setUniformValue("u_size", size)

            work_groups = (size + 15) // 16
            gl.glDispatchCompute(work_groups, work_groups, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # Ergebnis lesen
            result = self._read_texture_data(biome_tex, size, size, data_type=gl.GL_UNSIGNED_BYTE)

            self.processing_finished.emit("biome_blending", True)
            return result

        except Exception as e:
            self.processing_finished.emit("biome_blending", False)
            return self._cpu_fallback_biome_blending(temperature_map, precipitation_map, elevation_map, moisture_map)

    def _create_noise_shader(self):
        """
        Funktionsweise: Erstellt GLSL Compute Shader Code für Noise-Generierung
        Aufgabe: Definiert GPU-Code für Multi-Octave Simplex-Noise
        Returns: String mit GLSL Compute Shader Code
        """
        return """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;
        layout(r32f, binding = 0) uniform image2D output_image;

        uniform int u_size;
        uniform int u_octaves;
        uniform float u_frequency;
        uniform float u_persistence;
        uniform float u_lacunarity;
        uniform int u_seed;

        // Simplex-Noise Implementierung hier
        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x >= u_size || coord.y >= u_size) return;

            float x = float(coord.x) / float(u_size);
            float y = float(coord.y) / float(u_size);

            float noise_value = 0.0;
            float amplitude = 1.0;
            float frequency = u_frequency;

            for (int i = 0; i < u_octaves; i++) {
                // Simplified noise - echte Simplex-Implementierung würde hier stehen
                noise_value += amplitude * sin(x * frequency + u_seed) * cos(y * frequency + u_seed);
                amplitude *= u_persistence;
                frequency *= u_lacunarity;
            }

            imageStore(output_image, coord, vec4(noise_value, 0, 0, 0));
        }
        """

    def _create_erosion_shader(self):
        """
        Funktionsweise: Erstellt GLSL Compute Shader Code für Erosions-Simulation
        Aufgabe: Definiert GPU-Code für Hydraulic Erosion
        Returns: String mit GLSL Compute Shader Code
        """
        return """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;

        layout(r32f, binding = 0) uniform image2D height_in;
        layout(r32f, binding = 1) uniform image2D water_in;
        layout(rg32f, binding = 2) uniform image2D velocity_in;
        layout(r32f, binding = 3) uniform image2D sediment_in;

        layout(r32f, binding = 4) uniform image2D height_out;
        layout(r32f, binding = 5) uniform image2D water_out;
        layout(rg32f, binding = 6) uniform image2D velocity_out;
        layout(r32f, binding = 7) uniform image2D sediment_out;

        uniform int u_size;
        uniform float u_dt;

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x >= u_size || coord.y >= u_size) return;

            // Simplified erosion - echte Implementierung würde hier stehen
            float height = imageLoad(height_in, coord).r;
            float water = imageLoad(water_in, coord).r;
            vec2 velocity = imageLoad(velocity_in, coord).rg;
            float sediment = imageLoad(sediment_in, coord).r;

            // Basic erosion calculation
            float new_height = height - water * u_dt * 0.1;
            float new_water = water * 0.99;
            vec2 new_velocity = velocity * 0.98;
            float new_sediment = sediment * 0.95;

            imageStore(height_out, coord, vec4(new_height, 0, 0, 0));
            imageStore(water_out, coord, vec4(new_water, 0, 0, 0));
            imageStore(velocity_out, coord, vec4(new_velocity, 0, 0));
            imageStore(sediment_out, coord, vec4(new_sediment, 0, 0, 0));
        }
        """

    def _create_biome_shader(self):
        """
        Funktionsweise: Erstellt GLSL Compute Shader Code für Biome-Blending
        Aufgabe: Definiert GPU-Code für Biome-Klassifikation
        Returns: String mit GLSL Compute Shader Code
        """
        return """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;

        layout(r32f, binding = 0) uniform image2D temp_in;
        layout(r32f, binding = 1) uniform image2D precip_in;
        layout(r32f, binding = 2) uniform image2D elev_in;
        layout(r32f, binding = 3) uniform image2D moist_in;
        layout(r8ui, binding = 4) uniform uimage2D biome_out;

        uniform int u_size;

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x >= u_size || coord.y >= u_size) return;

            float temp = imageLoad(temp_in, coord).r;
            float precip = imageLoad(precip_in, coord).r;
            float elev = imageLoad(elev_in, coord).r;
            float moist = imageLoad(moist_in, coord).r;

            // Simplified biome classification
            uint biome_id = 0;
            if (temp < 0) biome_id = 1; // Tundra
            else if (temp < 15 && moist > 0.6) biome_id = 2; // Taiga
            else if (temp > 25 && precip < 200) biome_id = 3; // Desert
            else biome_id = 4; // Temperate

            imageStore(biome_out, coord, uvec4(biome_id, 0, 0, 0));
        }
        """

    def _compile_shader(self, name, shader_code):
        """
        Funktionsweise: Kompiliert GLSL Compute Shader und speichert ihn
        Aufgabe: Lädt und kompiliert Shader-Code für GPU-Verwendung
        Parameter: name (str), shader_code (str) - Shader-Name und GLSL-Code
        Returns: bool - True wenn Kompilierung erfolgreich
        """
        try:
            shader_program = QOpenGLShaderProgram()

            if shader_program.addShaderFromSourceCode(QOpenGLShader.Compute, shader_code):
                if shader_program.link():
                    self.shaders[name] = shader_program
                    return True
                else:
                    print(f"Shader linking failed for '{name}': {shader_program.log()}")
            else:
                print(f"Shader compilation failed for '{name}': {shader_program.log()}")

        except Exception as e:
            print(f"Shader creation failed for '{name}': {e}")

        return False

    def _create_texture_2d(self, width, height, internal_format):
        """
        Funktionsweise: Erstellt OpenGL-Texture für GPU-Processing
        Aufgabe: Allokiert GPU-Memory für Texture-Daten
        Parameter: width, height, internal_format - Texture-Dimensionen und Format
        Returns: GLuint - Texture-ID
        """
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RED, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        return texture_id

    def _upload_texture_2d(self, data, internal_format):
        """
        Funktionsweise: Lädt numpy-Array als Texture auf GPU hoch
        Aufgabe: Transfer von CPU-Daten zu GPU-Memory
        Parameter: data (numpy.ndarray), internal_format - Daten und GPU-Format
        Returns: GLuint - Texture-ID
        """
        height, width = data.shape[:2]
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        if len(data.shape) == 2:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RED, gl.GL_FLOAT, data)
        else:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RG, gl.GL_FLOAT, data)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        return texture_id

    def _read_texture_data(self, texture_id, width, height, channels=1, data_type=gl.GL_FLOAT):
        """
        Funktionsweise: Liest Texture-Daten von GPU zurück zu CPU
        Aufgabe: Transfer von GPU-Ergebnissen zu numpy-Array
        Parameter: texture_id, width, height, channels, data_type - Texture-Parameter
        Returns: numpy.ndarray - GPU-Ergebnisse als Array
        """
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

        if data_type == gl.GL_FLOAT:
            dtype = np.float32
            gl_format = gl.GL_RED if channels == 1 else gl.GL_RG
        else:
            dtype = np.uint8
            gl_format = gl.GL_RED_INTEGER if channels == 1 else gl.GL_RG_INTEGER

        data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl_format, data_type)

        if channels == 1:
            return np.frombuffer(data, dtype=dtype).reshape((height, width))
        else:
            return np.frombuffer(data, dtype=dtype).reshape((height, width, channels))

    def _cpu_fallback_noise(self, size, octaves, frequency, persistence, lacunarity, seed):
        """
        Funktionsweise: CPU-Fallback für Noise-Generierung wenn GPU nicht verfügbar
        Aufgabe: Software-Implementation der Noise-Generierung
        Parameter: size, octaves, frequency, persistence, lacunarity, seed - Noise-Parameter
        Returns: numpy array mit generiertem Noise
        """
        np.random.seed(seed)
        result = np.zeros((size, size), dtype=np.float32)

        for y in range(size):
            for x in range(size):
                noise_value = 0.0
                amplitude = 1.0
                freq = frequency

                for _ in range(octaves):
                    noise_value += amplitude * (np.sin(x * freq) * np.cos(y * freq))
                    amplitude *= persistence
                    freq *= lacunarity

                result[y, x] = noise_value

        return result

    def _cpu_fallback_erosion(self, heightmap, water_map, velocity_map, sediment_map, dt):
        """
        Funktionsweise: CPU-Fallback für Erosions-Simulation
        Aufgabe: Software-Implementation der Hydraulic Erosion
        Returns: Tuple mit erodierten Maps
        """
        # Simplified CPU erosion
        new_heightmap = heightmap - water_map * dt * 0.1
        new_water_map = water_map * 0.99
        new_velocity_map = velocity_map * 0.98
        new_sediment_map = sediment_map * 0.95

        return (new_heightmap, new_water_map, new_velocity_map, new_sediment_map)

    def _cpu_fallback_biome_blending(self, temperature_map, precipitation_map, elevation_map, moisture_map):
        """
        Funktionsweise: CPU-Fallback für Biome-Klassifikation
        Aufgabe: Software-Implementation der Biome-Zuordnung
        Returns: numpy array mit Biome-Indices
        """
        size = temperature_map.shape[0]
        result = np.zeros((size, size), dtype=np.uint8)

        for y in range(size):
            for x in range(size):
                temp = temperature_map[y, x]
                precip = precipitation_map[y, x]

                if temp < 0:
                    result[y, x] = 1  # Tundra
                elif temp < 15 and moisture_map[y, x] > 0.6:
                    result[y, x] = 2  # Taiga
                elif temp > 25 and precip < 200:
                    result[y, x] = 3  # Desert
                else:
                    result[y, x] = 4  # Temperate

        return result

    def cleanup(self):
        """
        Funktionsweise: Räumt GPU-Ressourcen bei Anwendungs-Beendigung auf
        Aufgabe: Freigebung aller GPU-Texturen und Shader-Programme
        """
        for shader in self.shaders.values():
            if shader:
                shader.release()

        for texture_id in self.textures.values():
            if texture_id:
                gl.glDeleteTextures(1, [texture_id])

        self.shaders.clear()
        self.textures.clear()