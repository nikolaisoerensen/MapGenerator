"""
Path: managers/shader_manager.py

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
import functools
import logging

import numpy as np
import OpenGL.GL as gl
from opensimplex import OpenSimplex
from PyQt5.QtCore import QObject, pyqtSignal

def _protective_handler(*args, **kwargs):
    """
    Funktionsweise: Schützender Dekorator für Shader-Methoden mit GPU-Zugriff
    Aufgabe: Fängt jede Exception in der dekorierten Methode ab, loggt sie als WARNING und
             gibt None zurück, damit ein GPU-Fehler nie die aufrufende Verarbeitung abbricht
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*call_args, **call_kwargs):
            try:
                return func(*call_args, **call_kwargs)
            except Exception as e:
                logging.getLogger(func.__module__).warning(
                    f"Unhandled exception in {func.__name__}: {e}")
                return None
        return wrapper
    return decorator


def get_shader_manager_error_decorators():
    """
    Funktionsweise: Liefert die schützenden Dekoratoren für die Shader-Methoden
    Aufgabe: gpu_shader_handler und initialization_handler kapseln GPU-nahe Methoden so, dass
             ein Fehler geloggt wird und None zurückkommt, statt die Verarbeitung abzubrechen
    Return: Tuple von zwei Decorator-Fabriken
    """
    return _protective_handler, _protective_handler


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
        self.logger = logging.getLogger(__name__)
        self.gpu_available = False
        self.context = None
        self.shaders = {}
        self.textures = {}
        self._gpu_checked = False

    def _compile_shader(self, name, shader_code):
        """Direkte OpenGL-Shader-Kompilierung"""
        try:
            # Compute Shader erstellen
            shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
            gl.glShaderSource(shader, shader_code)
            gl.glCompileShader(shader)

            # Kompilierung prüfen
            if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
                error = gl.glGetShaderInfoLog(shader).decode()
                print(f"Shader compilation failed for '{name}': {error}")
                gl.glDeleteShader(shader)
                return False

            # Shader-Programm erstellen
            program = gl.glCreateProgram()
            gl.glAttachShader(program, shader)
            gl.glLinkProgram(program)

            # Linking prüfen
            if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
                error = gl.glGetProgramInfoLog(program).decode()
                print(f"Shader linking failed for '{name}': {error}")
                gl.glDeleteProgram(program)
                gl.glDeleteShader(shader)
                return False

            # Cleanup und speichern
            gl.glDeleteShader(shader)  # Shader kann nach Linking gelöscht werden
            self.shaders[name] = program
            return True

        except Exception as e:
            print(f"Shader creation failed for '{name}': {e}")
            return False

    def _use_shader(self, name):
        """Aktiviert Shader für Verwendung"""
        if name in self.shaders:
            gl.glUseProgram(self.shaders[name])
            return self.shaders[name]
        return None

    def _set_shader_uniform_int(self, program, name, value):
        """Setzt Integer-Uniform"""
        location = gl.glGetUniformLocation(program, name)
        gl.glUniform1i(location, value)

    def _set_shader_uniform_float(self, program, name, value):
        """Setzt Float-Uniform"""
        location = gl.glGetUniformLocation(program, name)
        gl.glUniform1f(location, value)

    def _check_gpu_support(self):
        """
        Funktionsweise: Prüft ob GPU-Compute verfügbar ist
        Aufgabe: Erkennt OpenGL Compute Shader Support und setzt gpu_available Flag
        """
        try:
            # Direkte OpenGL-Version-Prüfung statt QGLContext. version_string.split()[0]
            # kann "4.3" oder "4.3.0" (Patch-Level, z.B. Intel-Treiber) sein - nur die
            # ersten beiden Komponenten nehmen, sonst wirft das Unpacking ValueError und
            # gpu_available bleibt für IMMER False (dieser Check läuft nur einmal).
            version_string = gl.glGetString(gl.GL_VERSION).decode()
            major, minor = map(int, version_string.split()[0].split('.')[:2])

            if major > 4 or (major == 4 and minor >= 3):
                self.gpu_available = True
            else:
                self.gpu_fallback_triggered.emit("OpenGL 4.3+ required")
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
            "biome_blending": self._create_biome_shader(),
            "shadow_raycast": self._create_shadow_raycast_shader()
        }

        for name, shader_code in core_shaders.items():
            if self._compile_shader(name, shader_code):
                self.logger.info(f"Shader '{name}' loaded successfully")
            else:
                self.logger.warning(f"Failed to load shader '{name}'")

    def _ensure_gpu_checked(self):
        """
        Funktionsweise: Führt GPU-Prüfung und Laden der Kern-Shader genau einmal aus
        Aufgabe: Verzögert die GPU-Erkennung bis zum ersten Verarbeitungsaufruf, damit sie erst
                 läuft, wenn ein OpenGL-Kontext existiert (im Konstruktor gibt es noch keinen).
                 Wird von jeder process_*-Methode aufgerufen, prüft aber nur beim ersten Mal.
        """
        if self._gpu_checked:
            return
        self._gpu_checked = True

        self._check_gpu_support()
        if self.gpu_available:
            self._load_core_shaders()

    def process_noise_generation(self, size, octaves, frequency, persistence, lacunarity, seed, perm):
        """
        Funktionsweise: GPU-beschleunigte Multi-Octave OpenSimplex-Noise Generierung
        Aufgabe: Parallel Processing für Terrain-Noise mit allen Octaves
        Parameter: size, octaves, frequency, persistence, lacunarity, seed - Noise-Parameter
        Parameter: perm - 256-Element Permutationstabelle (z.B. OpenSimplex(seed)._perm), muss mit dem
                   CPU-Referenzpfad übereinstimmen, damit GPU- und CPU-Noise für denselben Seed identisch sind
        Returns: numpy array mit generiertem Noise oder None bei Fehler
        """
        self._ensure_gpu_checked()
        if not self.gpu_available:
            return self._cpu_fallback_noise(size, octaves, frequency, persistence, lacunarity, seed)

        self.processing_started.emit("noise_generation")

        output_texture = None
        perm_buffer = None
        try:
            program = self._use_shader("noise_generation")
            if not program:
                raise Exception("Noise shader not available")

            output_texture = self._create_texture_2d(size, size, gl.GL_R32F)
            perm_buffer = self._upload_ssbo(perm, dtype=np.int32)

            gl.glBindImageTexture(0, output_texture, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, perm_buffer)

            self._set_shader_uniform_int(program, "u_size", size)
            self._set_shader_uniform_int(program, "u_octaves", octaves)
            self._set_shader_uniform_float(program, "u_frequency", frequency)
            self._set_shader_uniform_float(program, "u_persistence", persistence)
            self._set_shader_uniform_float(program, "u_lacunarity", lacunarity)

            # Compute Shader ausführen
            work_groups_x = (size + 15) // 16  # 16x16 Work-Group Size
            work_groups_y = (size + 15) // 16
            gl.glDispatchCompute(work_groups_x, work_groups_y, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.GL_SHADER_STORAGE_BARRIER_BIT)

            # Ergebnis von GPU lesen
            result = self._read_texture_data(output_texture, size, size)

            self.processing_finished.emit("noise_generation", True)
            return result

        except Exception as e:
            self.logger.warning(f"GPU noise generation failed: {e}")
            self.processing_finished.emit("noise_generation", False)
            return self._cpu_fallback_noise(size, octaves, frequency, persistence, lacunarity, seed)
        finally:
            if output_texture is not None:
                gl.glDeleteTextures(1, [output_texture])
            if perm_buffer is not None:
                gl.glDeleteBuffers(1, [perm_buffer])

    def process_erosion_simulation(self, heightmap, water_map, velocity_map, sediment_map, dt):
        """
        Funktionsweise: GPU-beschleunigte Hydraulic Erosion Simulation
        Aufgabe: Parallel Processing der Erosions-Gleichungen für alle Pixel
        Parameter: heightmap, water_map, velocity_map, sediment_map, dt - Erosions-Daten
        Returns: Tuple (new_heightmap, new_water_map, new_velocity_map, new_sediment_map)
        """
        self._ensure_gpu_checked()
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
        self._ensure_gpu_checked()
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

    def process_shadow_raycast(self, heightmap, sun_elevation, sun_azimuth, shadowmap_size=64):
        """
        Funktionsweise: GPU-beschleunigte Shadow-Raycast-Berechnung
        Aufgabe: Parallel Processing der Shadow-Raycasts für einen Sonnenwinkel
        Parameter: heightmap (numpy.ndarray), sun_elevation (float), sun_azimuth (float),
                  shadowmap_size (int), max_distance (float), step_size (float), height_scale (float)
        Returns: numpy array mit Shadow-Map (64x64) oder None bei Fehler
        """
        self._ensure_gpu_checked()
        if not self.gpu_available:
            return self._cpu_fallback_shadow_raycast(heightmap, sun_elevation, sun_azimuth, shadowmap_size)

        try:
            # Shader aktivieren
            program = self._use_shader("shadow_raycast")
            if not program:
                raise Exception("Shadow raycast shader not available")

            height_tex = self._upload_texture_2d(heightmap, gl.GL_R32F)
            shadow_tex = self._create_texture_2d(shadowmap_size, shadowmap_size, gl.GL_R32F)

            # Texturen binden (Binding-Index muss mit Shader übereinstimmen)
            gl.glBindImageTexture(0, height_tex, 0, gl.GL_FALSE, 0, gl.GL_READ_ONLY, gl.GL_R32F)
            gl.glBindImageTexture(1, shadow_tex, 0, gl.GL_FALSE, 0, gl.GL_WRITE_ONLY, gl.GL_R32F)

            # Uniforms setzen
            self._set_shader_uniform_int(program, "u_heightmap_size", heightmap.shape[0])
            self._set_shader_uniform_int(program, "u_shadowmap_size", shadowmap_size)
            self._set_shader_uniform_float(program, "u_sun_elevation", sun_elevation)
            self._set_shader_uniform_float(program, "u_sun_azimuth", sun_azimuth)

            # Compute Shader ausführen
            work_groups = (shadowmap_size + 7) // 8
            gl.glDispatchCompute(work_groups, work_groups, 1)
            gl.glMemoryBarrier(gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

            # Ergebnis von GPU lesen
            result = self._read_texture_data(shadow_tex, shadowmap_size, shadowmap_size)

            # Cleanup
            gl.glDeleteTextures(1, [height_tex])
            gl.glDeleteTextures(1, [shadow_tex])

            self.processing_finished.emit("shadow_raycast", True)
            return result



        except Exception as e:
            self.logger.warning(f"GPU shadow processing failed: {e}")
            return self._cpu_fallback_shadow_raycast(heightmap, sun_elevation, sun_azimuth, shadowmap_size)

    def _create_noise_shader(self):
        """
        Funktionsweise: Erstellt GLSL Compute Shader Code für Noise-Generierung
        Aufgabe: Definiert GPU-Code für Multi-Octave OpenSimplex-Noise (2D, Kurt-Spencer-Algorithmus),
                 Port von opensimplex.internals._noise2/_extrapolate2 - siehe core/terrain_generator.py's
                 SimplexNoiseGenerator (CPU-Referenz, gleicher Algorithmus über das opensimplex-Package).
                 Die Permutationstabelle wird nicht im Shader neu berechnet, sondern von der CPU
                 (OpenSimplex(seed)._perm) als SSBO hochgeladen, damit GPU- und CPU-Pfad für denselben
                 Seed exakt dieselbe Permutation verwenden.
        Returns: String mit GLSL Compute Shader Code
        """
        return """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;
        layout(r32f, binding = 0) uniform writeonly image2D output_image;

        layout(std430, binding = 1) readonly buffer PermBuffer {
            int perm[256];
        };

        uniform int u_size;
        uniform int u_octaves;
        uniform float u_frequency;
        uniform float u_persistence;
        uniform float u_lacunarity;

        const float STRETCH_CONSTANT2 = -0.211324865405187;
        const float SQUISH_CONSTANT2 = 0.366025403784439;
        const float NORM_CONSTANT2 = 47.0;

        const vec2 GRAD2[8] = vec2[8](
            vec2(5.0, 2.0), vec2(2.0, 5.0),
            vec2(-5.0, 2.0), vec2(-2.0, 5.0),
            vec2(5.0, -2.0), vec2(2.0, -5.0),
            vec2(-5.0, -2.0), vec2(-2.0, -5.0)
        );

        float extrapolate2(int xsb, int ysb, float dx, float dy) {
            int index = (perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E) >> 1;
            vec2 g = GRAD2[index];
            return g.x * dx + g.y * dy;
        }

        // Direkter Port von opensimplex.internals._noise2 (Kurt-Spencer OpenSimplex 2D)
        float openSimplexNoise2(float x, float y) {
            float stretchOffset = (x + y) * STRETCH_CONSTANT2;
            float xs = x + stretchOffset;
            float ys = y + stretchOffset;

            int xsb = int(floor(xs));
            int ysb = int(floor(ys));

            float squishOffset = float(xsb + ysb) * SQUISH_CONSTANT2;
            float xb = float(xsb) + squishOffset;
            float yb = float(ysb) + squishOffset;

            float xins = xs - float(xsb);
            float yins = ys - float(ysb);
            float inSum = xins + yins;

            float dx0 = x - xb;
            float dy0 = y - yb;

            float value = 0.0;

            float dx1 = dx0 - 1.0 - SQUISH_CONSTANT2;
            float dy1 = dy0 - SQUISH_CONSTANT2;
            float attn1 = 2.0 - dx1 * dx1 - dy1 * dy1;
            if (attn1 > 0.0) {
                attn1 *= attn1;
                value += attn1 * attn1 * extrapolate2(xsb + 1, ysb, dx1, dy1);
            }

            float dx2 = dx0 - SQUISH_CONSTANT2;
            float dy2 = dy0 - 1.0 - SQUISH_CONSTANT2;
            float attn2 = 2.0 - dx2 * dx2 - dy2 * dy2;
            if (attn2 > 0.0) {
                attn2 *= attn2;
                value += attn2 * attn2 * extrapolate2(xsb, ysb + 1, dx2, dy2);
            }

            int xsvExt;
            int ysvExt;
            float dxExt;
            float dyExt;

            if (inSum <= 1.0) {
                float zins = 1.0 - inSum;
                if (zins > xins || zins > yins) {
                    if (xins > yins) {
                        xsvExt = xsb + 1; ysvExt = ysb - 1;
                        dxExt = dx0 - 1.0; dyExt = dy0 + 1.0;
                    } else {
                        xsvExt = xsb - 1; ysvExt = ysb + 1;
                        dxExt = dx0 + 1.0; dyExt = dy0 - 1.0;
                    }
                } else {
                    xsvExt = xsb + 1; ysvExt = ysb + 1;
                    dxExt = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT2;
                    dyExt = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT2;
                }
            } else {
                float zins = 2.0 - inSum;
                if (zins < xins || zins < yins) {
                    if (xins > yins) {
                        xsvExt = xsb + 2; ysvExt = ysb;
                        dxExt = dx0 - 2.0 - 2.0 * SQUISH_CONSTANT2;
                        dyExt = dy0 - 2.0 * SQUISH_CONSTANT2;
                    } else {
                        xsvExt = xsb; ysvExt = ysb + 2;
                        dxExt = dx0 - 2.0 * SQUISH_CONSTANT2;
                        dyExt = dy0 - 2.0 - 2.0 * SQUISH_CONSTANT2;
                    }
                } else {
                    dxExt = dx0; dyExt = dy0;
                    xsvExt = xsb; ysvExt = ysb;
                }
                xsb += 1;
                ysb += 1;
                dx0 = dx0 - 1.0 - 2.0 * SQUISH_CONSTANT2;
                dy0 = dy0 - 1.0 - 2.0 * SQUISH_CONSTANT2;
            }

            float attn0 = 2.0 - dx0 * dx0 - dy0 * dy0;
            if (attn0 > 0.0) {
                attn0 *= attn0;
                value += attn0 * attn0 * extrapolate2(xsb, ysb, dx0, dy0);
            }

            float attnExt = 2.0 - dxExt * dxExt - dyExt * dyExt;
            if (attnExt > 0.0) {
                attnExt *= attnExt;
                value += attnExt * attnExt * extrapolate2(xsvExt, ysvExt, dxExt, dyExt);
            }

            return value / NORM_CONSTANT2;
        }

        void main() {
            ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
            if (coord.x >= u_size || coord.y >= u_size) return;

            float x = float(coord.x);
            float y = float(coord.y);

            float noise_value = 0.0;
            float amplitude = 1.0;
            float frequency = u_frequency;
            float max_amplitude = 0.0;

            for (int i = 0; i < u_octaves; i++) {
                noise_value += amplitude * openSimplexNoise2(x * frequency, y * frequency);
                max_amplitude += amplitude;
                amplitude *= u_persistence;
                frequency *= u_lacunarity;
            }

            if (max_amplitude > 0.0) {
                noise_value /= max_amplitude;
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

    def _create_shadow_raycast_shader(self):
        """
        Funktionsweise: Erstellt GLSL Compute Shader Code für Shadow-Raycasting
        Aufgabe: Definiert GPU-Code für parallele Raycast-Shadow-Berechnung
        Returns: String mit GLSL Compute Shader Code
        """
        return """
        #version 430
        layout(local_size_x = 8, local_size_y = 8) in;

        layout(r32f, binding = 0) uniform readonly image2D heightmap;
        layout(r32f, binding = 1) uniform writeonly image2D shadowmap;

        uniform int u_heightmap_size;
        uniform int u_shadowmap_size;
        uniform float u_sun_elevation;
        uniform float u_sun_azimuth;
        uniform float u_max_distance;
        uniform float u_step_size;
        uniform float u_height_scale;

        const float PI = 3.14159265359;
        const float DEG_TO_RAD = PI / 180.0;

        float sampleHeight(vec2 uv) {
            vec2 coord = uv * float(u_heightmap_size - 1);
            ivec2 coord0 = ivec2(floor(coord));
            ivec2 coord1 = coord0 + ivec2(1, 1);

            coord0 = clamp(coord0, ivec2(0), ivec2(u_heightmap_size - 1));
            coord1 = clamp(coord1, ivec2(0), ivec2(u_heightmap_size - 1));

            float h00 = imageLoad(heightmap, coord0).r;
            float h10 = imageLoad(heightmap, ivec2(coord1.x, coord0.y)).r;
            float h01 = imageLoad(heightmap, ivec2(coord0.x, coord1.y)).r;
            float h11 = imageLoad(heightmap, coord1).r;

            vec2 frac = coord - vec2(coord0);
            float h0 = mix(h00, h10, frac.x);
            float h1 = mix(h01, h11, frac.x);

            return mix(h0, h1, frac.y);
        }

        float calculateShadow(vec2 shadowCoord) {
            vec2 startUV = shadowCoord / float(u_shadowmap_size - 1);
            float startHeight = sampleHeight(startUV) * u_height_scale;

            float elevRad = u_sun_elevation * DEG_TO_RAD;
            float azimRad = u_sun_azimuth * DEG_TO_RAD;

            vec3 sunDir = vec3(
                cos(elevRad) * sin(azimRad),
                cos(elevRad) * cos(azimRad),
                sin(elevRad)
            );

            vec2 rayStep = sunDir.xy * u_step_size / float(u_heightmap_size);
            float heightStep = sunDir.z * u_step_size;

            vec2 currentUV = startUV;
            float currentHeight = startHeight + heightStep;

            for (float distance = u_step_size; distance < u_max_distance; distance += u_step_size) {
                currentUV += rayStep;
                currentHeight += heightStep;

                if (currentUV.x < 0.0 || currentUV.x > 1.0 || 
                    currentUV.y < 0.0 || currentUV.y > 1.0) {
                    break;
                }

                float terrainHeight = sampleHeight(currentUV) * u_height_scale;

                if (currentHeight <= terrainHeight) {
                    return 0.0;
                }
            }

            return 1.0;
        }

        void main() {
            ivec2 shadowCoord = ivec2(gl_GlobalInvocationID.xy);

            if (shadowCoord.x >= u_shadowmap_size || shadowCoord.y >= u_shadowmap_size) {
                return;
            }

            float shadowValue = calculateShadow(vec2(shadowCoord));
            imageStore(shadowmap, shadowCoord, vec4(shadowValue, 0.0, 0.0, 1.0));
        }
        """

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
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl.GL_RED, gl.GL_FLOAT, data)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        return texture_id

    def _upload_ssbo(self, data, dtype=np.int32):
        """
        Funktionsweise: Lädt ein numpy-Array als Shader-Storage-Buffer-Object auf die GPU hoch
        Aufgabe: Transfer von CPU-Daten (z.B. Permutationstabelle) zu GPU-Memory für Compute-Shader
        Parameter: data (numpy.ndarray), dtype - Quelldaten und Ziel-GPU-Datentyp
        Returns: GLuint - Buffer-ID
        """
        buffer_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer_id)
        array = np.ascontiguousarray(data, dtype=dtype)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, array.nbytes, array, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        return buffer_id

    def _read_texture_data(self, texture_id, width, height, channels=1, data_type=gl.GL_FLOAT):
        """
        Funktionsweise: Liest Texture-Daten von GPU zurück zu CPU
        Aufgabe: Transfer von GPU-Ergebnissen zu numpy-Array
        Parameter: texture_id, width, height, channels, data_type - Texture-Parameter
        Returns: numpy.ndarray - GPU-Ergebnisse als Array
        """
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, gl.GL_RED, gl.GL_FLOAT)
        return np.frombuffer(data, dtype=np.float32).reshape((height, width))

    def _cpu_fallback_noise(self, size, octaves, frequency, persistence, lacunarity, seed):
        """
        Funktionsweise: CPU-Fallback für Noise-Generierung wenn GPU nicht verfügbar oder
                        die GPU-Dispatch fehlschlägt
        Aufgabe: Software-Implementation mit demselben OpenSimplex-Algorithmus und derselben
                 Normalisierung wie core/terrain_generator.py's SimplexNoiseGenerator._generate_cpu_optimized(),
                 damit dieser interne Fallback nie stillschweigend die alte sin(x)*cos(y)-Platzhalter-Noise
                 zurückgibt (das hatte zuvor near-flat Heightmaps verursacht, siehe _create_noise_shader())
        Parameter: size, octaves, frequency, persistence, lacunarity, seed - Noise-Parameter
        Returns: numpy array mit generiertem Noise
        """
        generator = OpenSimplex(seed=seed)
        coords = np.arange(size, dtype=np.float64)

        result = np.zeros((size, size), dtype=np.float32)
        amplitude = 1.0
        current_frequency = frequency
        max_amplitude = 0.0

        for _ in range(octaves):
            octave_noise = generator.noise2array(coords * current_frequency, coords * current_frequency)
            result += (amplitude * octave_noise).astype(np.float32)
            max_amplitude += amplitude
            amplitude *= persistence
            current_frequency *= lacunarity

        if max_amplitude > 0:
            result /= max_amplitude

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

    def _cpu_fallback_shadow_raycast(self, heightmap, sun_elevation, sun_azimuth,
                                     shadowmap_size, max_distance=100.0, step_size=1.0, height_scale=1.0):
        """
        Funktionsweise: CPU-Fallback für Shadow-Raycast-Berechnung
        Aufgabe: Software-Implementation der Shadow-Raycasts
        """
        import numpy as np

        heightmap_size = heightmap.shape[0]
        shadowmap = np.ones((shadowmap_size, shadowmap_size), dtype=np.float32)

        # Sun direction calculation
        elev_rad = np.radians(sun_elevation)
        azim_rad = np.radians(sun_azimuth)

        sun_dir_x = np.cos(elev_rad) * np.sin(azim_rad)
        sun_dir_y = np.cos(elev_rad) * np.cos(azim_rad)
        sun_dir_z = np.sin(elev_rad)

        # Process each shadow pixel
        for sy in range(shadowmap_size):
            for sx in range(shadowmap_size):
                # Convert shadow coord to heightmap UV
                start_uv_x = sx / (shadowmap_size - 1)
                start_uv_y = sy / (shadowmap_size - 1)

                # Bilinear interpolation for start height
                hm_x = start_uv_x * (heightmap_size - 1)
                hm_y = start_uv_y * (heightmap_size - 1)

                x0, y0 = int(hm_x), int(hm_y)
                x1, y1 = min(x0 + 1, heightmap_size - 1), min(y0 + 1, heightmap_size - 1)

                fx, fy = hm_x - x0, hm_y - y0

                h00 = heightmap[y0, x0]
                h10 = heightmap[y0, x1]
                h01 = heightmap[y1, x0]
                h11 = heightmap[y1, x1]

                h0 = h00 * (1 - fx) + h10 * fx
                h1 = h01 * (1 - fx) + h11 * fx
                start_height = (h0 * (1 - fy) + h1 * fy) * height_scale

                # Raycast
                ray_step_x = sun_dir_x * step_size / heightmap_size
                ray_step_y = sun_dir_y * step_size / heightmap_size
                height_step = sun_dir_z * step_size

                current_uv_x = start_uv_x
                current_uv_y = start_uv_y
                current_height = start_height + height_step

                distance = step_size
                while distance < max_distance:
                    current_uv_x += ray_step_x
                    current_uv_y += ray_step_y
                    current_height += height_step

                    # Check bounds
                    if current_uv_x < 0 or current_uv_x > 1 or current_uv_y < 0 or current_uv_y > 1:
                        break

                    # Sample terrain height
                    hm_x = current_uv_x * (heightmap_size - 1)
                    hm_y = current_uv_y * (heightmap_size - 1)

                    x0, y0 = int(hm_x), int(hm_y)
                    x1, y1 = min(x0 + 1, heightmap_size - 1), min(y0 + 1, heightmap_size - 1)

                    fx, fy = hm_x - x0, hm_y - y0

                    h00 = heightmap[y0, x0]
                    h10 = heightmap[y0, x1]
                    h01 = heightmap[y1, x0]
                    h11 = heightmap[y1, x1]

                    h0 = h00 * (1 - fx) + h10 * fx
                    h1 = h01 * (1 - fx) + h11 * fx
                    terrain_height = (h0 * (1 - fy) + h1 * fy) * height_scale

                    # Check intersection
                    if current_height <= terrain_height:
                        shadowmap[sy, sx] = 0.0  # In shadow
                        break

                    distance += step_size

        return shadowmap

    def cleanup(self):
        """
        Funktionsweise: Räumt GPU-Ressourcen bei Anwendungs-Beendigung auf
        Aufgabe: Freigebung aller GPU-Texturen und Shader-Programme, robust auch ohne aktiven
                 GPU-Kontext, damit das Beenden nie an einer GL-Operation scheitert
        """
        for shader in self.shaders.values():
            if shader:
                try:
                    shader.release()
                except Exception as e:
                    self.logger.warning(f"Shader-Release fehlgeschlagen: {e}")

        for texture_id in self.textures.values():
            if texture_id:
                try:
                    gl.glDeleteTextures(1, [texture_id])
                except Exception as e:
                    self.logger.warning(f"Textur-Freigabe fehlgeschlagen: {e}")

        self.shaders.clear()
        self.textures.clear()